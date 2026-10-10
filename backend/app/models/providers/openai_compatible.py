"""OpenAI-compatible Chat Completions Adapter。"""

from __future__ import annotations

import asyncio
import copy
import re
import ssl

from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from app.core.tokens import estimate_tokens
from app.models.adapter import ModelAdapter, ModelRequest, ModelResponse, ModelStreamChunk
from app.models.config import ModelConfig

_PROVIDER_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


class OpenAICompatibleAdapter(ModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        if not config.model or not config.model.strip():
            raise ValueError("OpenAI 兼容接口需要填写模型名称。")
        self.config = config
        self.count_tokens("")  # 启动时校验本地词表；配置错误不降级成另一种计数。
        self.timeout_seconds = config.timeout_seconds
        self.minimum_reasoning_effort = config.minimum_reasoning_effort
        self.completion_review_extra_body = config.completion_review_extra_body
        self.supports_stream = config.supports_stream
        self.supports_tools = config.supports_tools
        self.supports_json_object = config.supports_json_object
        self.supports_structured_output = config.supports_json_object
        self.supports_json_schema = config.supports_json_schema
        # 本地 Ollama、vLLM 等兼容服务通常不校验 API Key；官方或云端服务
        # 仍由用户在配置中填写真实密钥。OpenAI SDK 要求传入非空字符串，
        # 因此对无密钥的本地服务使用占位值，不会把它发送为业务凭据。
        base_url = config.base_url.strip() if config.base_url and config.base_url.strip() else None
        http_client = DefaultAsyncHttpxClient(verify=ssl.create_default_context()) if config.use_system_certificates else None
        self.client = AsyncOpenAI(api_key=config.api_key or "local", base_url=base_url, timeout=config.timeout_seconds, max_retries=0, http_client=http_client)

    def count_tokens(self, value: str) -> int:
        return estimate_tokens(value, self.config.tokenizer_file)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        reasoning_effort = _reasoning_effort(self.config, request)
        tools, name_to_alias, alias_to_name = _provider_tools(request.tools)
        timeout = request.timeout_seconds if request.timeout_seconds is not None else self.config.timeout_seconds
        async with asyncio.timeout(timeout):
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=_provider_messages(request.messages, name_to_alias),
                tools=tools or None if _capability(self, "supports_tools", True) else None,
                response_format=request.response_format if _capability(self, "supports_json_object", True) or _capability(self, "supports_json_schema", False) else None,
                temperature=request.temperature if request.temperature is not None else self.config.temperature,
                max_tokens=request.max_tokens,
                extra_body=request.extra_body,
                timeout=timeout,
                **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
            )
        message = response.choices[0].message
        tool_calls = _restore_tool_names([call.model_dump() for call in (message.tool_calls or [])], alias_to_name)
        return ModelResponse(content=message.content or "", tool_calls=tool_calls, input_tokens=response.usage.prompt_tokens if response.usage else None, output_tokens=response.usage.completion_tokens if response.usage else None, model=response.model, finish_reason=response.choices[0].finish_reason)

    async def stream(self, request: ModelRequest):
        if not _capability(self, "supports_stream", True):
            async for chunk in super().stream(request):
                yield chunk
            return
        tool_calls: dict[int, dict] = {}
        reasoning_effort = _reasoning_effort(self.config, request)
        tools, name_to_alias, alias_to_name = _provider_tools(request.tools)
        model = None
        input_tokens = None
        output_tokens = None
        finish_reason = None
        timeout = request.timeout_seconds if request.timeout_seconds is not None else self.config.timeout_seconds
        async with asyncio.timeout(timeout):
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=_provider_messages(request.messages, name_to_alias),
                tools=tools or None if _capability(self, "supports_tools", True) else None,
                response_format=request.response_format if _capability(self, "supports_json_object", True) or _capability(self, "supports_json_schema", False) else None,
                temperature=request.temperature if request.temperature is not None else self.config.temperature,
                max_tokens=request.max_tokens,
                extra_body=request.extra_body,
                stream=True,
                stream_options={"include_usage": True},
                timeout=timeout,
                **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
            )
            try:
                async for chunk in stream:
                    model = getattr(chunk, "model", None) or model
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        input_tokens = usage.prompt_tokens
                        output_tokens = usage.completion_tokens
                    if chunk.choices:
                        choice = chunk.choices[0]
                        finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                        delta = choice.delta
                        content = delta.content or ""
                        if content:
                            yield ModelStreamChunk(content=content, model=model)
                        for raw_call in delta.tool_calls or []:
                            index = raw_call.index if raw_call.index is not None else len(tool_calls)
                            call = tool_calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                            if raw_call.id:
                                call["id"] = raw_call.id
                            if raw_call.type:
                                call["type"] = raw_call.type
                            if raw_call.function:
                                if raw_call.function.name:
                                    call["function"]["name"] += raw_call.function.name
                                if raw_call.function.arguments:
                                    call["function"]["arguments"] += raw_call.function.arguments
                    # include_usage 的统计包通常位于 finish_reason 之后，choices 为空。
                    # 正文即时推送，决策聚合到统计包或正常 EOF；仍受现有请求超时约束。
                    if finish_reason and usage is not None:
                        break
            finally:
                await stream.close()
        # 某些兼容服务没有发送 finish_reason，但正常关闭了流；仍然给上层
        # 一个明确 terminal chunk，兼容旧服务，同时不会把“正文已到齐”当作结束。
        yield ModelStreamChunk(
            tool_calls=_restore_tool_names([tool_calls[index] for index in sorted(tool_calls)], alias_to_name),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            finish_reason=finish_reason,
            done=True,
        )

    async def close(self) -> None:
        await self.client.close()


def _capability(adapter: OpenAICompatibleAdapter, name: str, default: bool) -> bool:
    if name in getattr(adapter, "__dict__", {}):
        return bool(adapter.__dict__[name])
    config = getattr(adapter, "config", None)
    return bool(getattr(config, name, default))


def _reasoning_effort(config: ModelConfig, request: ModelRequest):
    effort = request.reasoning_effort or config.default_reasoning_effort
    if effort and effort not in config.reasoning_efforts:
        raise ValueError(f"模型 {config.model} 不支持思考程度 {effort}")
    return effort


def _provider_tools(tools: list[dict]) -> tuple[list[dict], dict[str, str], dict[str, str]]:
    """把内部命名空间转换为 OpenAI 兼容接口普遍接受的函数名。"""

    provider_tools = copy.deepcopy(tools)
    name_to_alias: dict[str, str] = {}
    alias_to_name: dict[str, str] = {}
    for tool in provider_tools:
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        name = function["name"]
        alias = _provider_tool_name(name)
        existing = alias_to_name.get(alias)
        if existing is not None and existing != name:
            raise ValueError(f"工具名称映射冲突：{existing} 与 {name}")
        function["name"] = alias
        name_to_alias[name] = alias
        alias_to_name[alias] = name
    return provider_tools, name_to_alias, alias_to_name


def _provider_messages(messages: list[dict], name_to_alias: dict[str, str]) -> list[dict]:
    """同步转换历史 assistant tool_calls，保持多轮工具协议一致。"""

    provider_messages = copy.deepcopy(messages)
    for message in provider_messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                function["name"] = name_to_alias.get(function["name"], _provider_tool_name(function["name"]))
    return provider_messages


def _restore_tool_names(tool_calls: list[dict], alias_to_name: dict[str, str]) -> list[dict]:
    """模型响应离开适配层前恢复 GeoAgent 的内部工具名称。"""

    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict) and function.get("name") in alias_to_name:
            function["name"] = alias_to_name[function["name"]]
    return tool_calls


def _provider_tool_name(name: str) -> str:
    return name if _PROVIDER_TOOL_NAME.fullmatch(name) else re.sub(r"[^a-zA-Z0-9_-]", "__", name)
