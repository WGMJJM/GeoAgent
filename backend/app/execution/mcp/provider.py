"""持久 MCP 会话与按需工具注册；完整目录不直接进入模型上下文。"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from copy import deepcopy
from typing import Any

import httpx2
from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

from app.core.models import ErrorCategory, ToolError, ToolMetadata, ToolResult, ToolStatus
from app.execution.tools import RegisteredTool, ToolContext, ToolRegistry

from .config import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPProvider:
    def __init__(self, config: MCPServerConfig, registry: ToolRegistry, *, timeout_seconds: int) -> None:
        self.config = config
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self._tools: dict[str, types.Tool] = {}
        self._metadata: dict[str, ToolMetadata] = {}
        self._session: ClientSession | None = None
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._events: asyncio.Queue[str] = asyncio.Queue()
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        return self._session is not None and self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._ready = asyncio.Event()
        self._events = asyncio.Queue()
        self.last_error = None
        self._task = asyncio.create_task(self._run(), name=f"mcp-{self.config.id}")
        await self._ready.wait()

    def close(self) -> None:
        if self._task is not None and not self._task.done():
            self._events.put_nowait("stop")

    async def aclose(self) -> None:
        self.close()
        if self._task is not None:
            await self._task

    async def _run(self) -> None:
        # SDK 的 AnyIO 上下文在同一所有者任务内建立与退出；不跨任务关闭取消域。
        try:
            async with AsyncExitStack() as stack:
                if self.config.transport == "stdio":
                    parameters = StdioServerParameters(
                        command=self.config.command,
                        args=self.config.args,
                        cwd=self.config.cwd,
                        env=self.config.resolve_refs(self.config.env_refs),
                    )
                    streams = await stack.enter_async_context(stdio_client(parameters))
                else:
                    client = await stack.enter_async_context(httpx2.AsyncClient(
                        headers=self.config.resolve_refs(self.config.header_refs),
                        timeout=self.timeout_seconds,
                    ))
                    streams = await stack.enter_async_context(streamable_http_client(
                        self.config.url, http_client=client,
                    ))
                session = await stack.enter_async_context(ClientSession(
                    *streams,
                    read_timeout_seconds=self.timeout_seconds,
                    message_handler=self._notification,
                ))
                async with asyncio.timeout(self.timeout_seconds):
                    await session.initialize()
                    await self._refresh(session)
                self._session = session
                self._ready.set()
                while (event := await self._events.get()) != "stop":
                    if event == "refresh":
                        async with asyncio.timeout(self.timeout_seconds):
                            await self._refresh(session)
        except Exception as exc:
            # 外部服务故障不能使内置工具不可用；日志不展开可能含凭据的 SDK 异常。
            self.last_error = type(exc).__name__
            logger.warning("MCP 服务 %s 连接失败：%s", self.config.id, self.last_error)
        finally:
            self._session = None
            self._ready.set()
            for name in self._tools:
                self.registry.unregister(name)

    async def _notification(self, message) -> None:
        if isinstance(message, types.ToolListChangedNotification):
            self._events.put_nowait("refresh")
        elif isinstance(message, Exception):
            self.last_error = type(message).__name__
            self._session = None
            self._events.put_nowait("stop")

    async def _refresh(self, session: ClientSession) -> None:
        tools: dict[str, types.Tool] = {}
        cursor = None
        while True:
            result = await session.list_tools(params=types.PaginatedRequestParams(cursor=cursor))
            for tool in result.tools:
                name = f"mcp.{self.config.id}.{tool.name}"
                if name in tools:
                    raise ValueError("MCP 服务返回了重复工具名")
                tools[name] = tool
            cursor = result.next_cursor
            if cursor is None:
                break
        for name, previous in self._tools.items():
            if tools.get(name) != previous:
                self.registry.unregister(name)
        self._tools = tools
        self._metadata = {
            name: ToolMetadata(
                name=name,
                description=tool.description or tool.title or tool.name,
                input_schema=deepcopy(tool.input_schema),
                required_scopes=[self.config.scope],
                required_envs=[self.config.environment],
                risk_level=self.config.tool_risks.get(tool.name, self.config.risk_level),
                tags=["mcp", self.config.id],
            )
            for name, tool in tools.items()
        }

    def summaries(self) -> tuple[ToolMetadata, ...]:
        return tuple(self._metadata.values()) if self.available else ()

    def materialize(self, name: str) -> RegisteredTool:
        tool = self._tools[name]

        async def handler(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
            session = self._session
            if session is None or not self.available:
                return self._failure(context, "MCP_UNAVAILABLE", "MCP 服务当前未连接。")
            if self._tools.get(name) != tool:
                return self._failure(context, "MCP_TOOL_CHANGED", "工具定义已变化，请重新获取当前 Schema。")
            try:
                result = await session.call_tool(
                    tool.name, arguments, read_timeout_seconds=self.timeout_seconds,
                )
            except MCPError:
                return self._failure(context, "MCP_PROTOCOL_ERROR", "MCP 服务拒绝了调用或返回协议错误。")
            except Exception as exc:
                self.last_error = type(exc).__name__
                self._session = None
                self.close()
                return self._failure(context, "MCP_CALL_FAILED", "MCP 调用失败；服务已标记为不可用，操作不会自动重试。")
            output = {
                "server": self.config.id,
                "content": [item.model_dump(mode="json", by_alias=True) for item in result.content],
                "structured_content": result.structured_content,
            }
            return ToolResult(
                call_id=context.call_id or "",
                status=ToolStatus.FAILED if result.is_error else ToolStatus.SUCCESS,
                output=output,
                error=ToolError(code="MCP_TOOL_ERROR", category=ErrorCategory.EXTERNAL,
                                message="MCP 工具返回执行错误，详见工具结果。") if result.is_error else None,
            )

        return RegisteredTool(self._metadata[name], handler)

    @staticmethod
    def _failure(context: ToolContext, code: str, message: str) -> ToolResult:
        return ToolResult(call_id=context.call_id or "", status=ToolStatus.FAILED,
                          error=ToolError(code=code, category=ErrorCategory.EXTERNAL, message=message))
