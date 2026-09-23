"""GeoAgent 唯一的模型决策与工具观察循环。"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.core.models import (
    AgentRequest,
    AgentResult,
    AgentResultStatus,
    Checkpoint,
    Run,
    RunStatus,
    ToolCall,
    ToolError,
    ToolResult,
    ToolStatus,
    new_id,
)
from app.execution.tools import ToolExecutor, ToolRegistry
from app.models import ModelAdapter, ModelRequest
from app.observability import EventType, TraceRecorder
from app.run.lifecycle import persist_result
from app.config import Settings
from app.state import StateStore

from .context import ContextBuilder

ASK_USER_TOOL = {
    "type": "function",
    "function": {
        "name": "agent.ask_user",
        "description": "在关键参数或资源无法安全确认时，向用户提出一个具体问题。",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "minLength": 1}},
            "required": ["question"],
            "additionalProperties": False,
        },
    },
}

SEARCH_HISTORY_TOOL = {
    "type": "function",
    "function": {
        "name": "conversation.search_history",
        "description": "按需在当前对话的原始消息中检索相关历史；仅返回同一对话消息。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

FIRST_STAGE_TOOLS = frozenset({"dataset.list", "dataset.inspect"})


@dataclass(slots=True)
class LoopPreparedRequest:
    """统一模型循环运行时的不可变请求绑定。"""

    request: AgentRequest
    run: Run


class AgentLoop:
    def __init__(
        self,
        store: StateStore,
        registry: ToolRegistry,
        executor: ToolExecutor,
        trace: TraceRecorder,
        settings: Settings,
        model_provider: Callable[[str | None], ModelAdapter | None],
        services_factory: Callable[[str | None], dict[str, Any]],
        *,
        context_services: dict[str, Any] | None = None,
        metrics=None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.executor = executor
        self.trace = trace
        self.settings = settings
        self.model_provider = model_provider
        self.services_factory = services_factory
        self.metrics = metrics
        context_services = context_services or {}
        self.context = ContextBuilder(
            store,
            profile_service=context_services.get("profile"),
            project_memory=context_services.get("project_memory"),
            conversation_memory=context_services.get("conversation_memory"),
        )
        self.model_adapter: ModelAdapter | None = None
        self.model_adapters: dict[str, ModelAdapter] = {}
        self.default_model_profile: str | None = None

    async def prepare_request(
        self,
        request: AgentRequest,
        *,
        metadata: dict[str, object] | None = None,
    ) -> LoopPreparedRequest:
        """不经过自然语言 Gate，直接为每条普通消息建立无 Task 的 Run。"""

        run = Run(
            conversation_id=request.conversation_id,
            agent_id="agent-loop",
            status=RunStatus.RUNNING,
            started_at=datetime.now(UTC),
            metadata={
                "request_id": request.request_id,
                "original_request": request.user_input,
                "original_request_message_id": _latest_user_message_id(self.store, request.conversation_id),
                "protocol_version": 1,
                **(metadata or {}),
            },
        )
        self.store.save_run(run)
        return LoopPreparedRequest(request=request, run=run)

    def prepare_resume(self, request: AgentRequest, run: Run) -> LoopPreparedRequest:
        return LoopPreparedRequest(request=request, run=run)

    async def run(
        self,
        request: AgentRequest,
        *,
        prepared: LoopPreparedRequest,
        resume_from: Checkpoint | None = None,
        continuation: dict[str, object] | None = None,
        on_model_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> AgentResult:
        del continuation
        run = prepared.run
        model = self.model_provider(request.model_profile)
        if model is None:
            return await self._finish(
                run,
                request=request,
                result=AgentResult(
                    agent_id=run.agent_id,
                    status=AgentResultStatus.FAILED,
                    summary="尚未配置可用模型，当前请求未执行。",
                    error="MODEL_NOT_CONFIGURED",
                    trace_id=run.id,
                ),
            )

        messages, cursor_id = self._initial_messages(request, resume_from, run)
        tools = self._tool_definitions()
        tool_call_count = run.tool_call_count
        for turn in range(1, self.settings.max_agent_turns + 1):
            current = self.store.get_run(run.id) or run
            current = current.model_copy(update={"status": RunStatus.RUNNING, "turn_count": current.turn_count + 1})
            self.store.save_run(current)
            try:
                response = await model.complete(
                    ModelRequest(
                        messages=messages,
                        tools=tools if tool_call_count < self.settings.max_tool_calls else [],
                        max_tokens=self.settings.max_tokens,
                    )
                )
            except Exception:
                return await self._finish(
                    current,
                    request=request,
                    result=AgentResult(
                        agent_id=current.agent_id,
                        status=AgentResultStatus.FAILED,
                        summary="模型服务请求失败；本轮没有执行未确认的操作。",
                        error="MODEL_UNAVAILABLE",
                        trace_id=current.id,
                    ),
                )

            if response.tool_calls:
                if tool_call_count >= self.settings.max_tool_calls:
                    return await self._finish(
                        current,
                        request=request,
                        result=AgentResult(
                            agent_id=current.agent_id,
                            status=AgentResultStatus.BLOCKED,
                            summary="已达到本次运行的工具调用上限，运行已安全停止。",
                            error="BUDGET_EXCEEDED",
                            trace_id=current.id,
                        ),
                    )
                assistant_calls, pending = _decode_tool_calls(response.tool_calls, current.id)
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": assistant_calls,
                    }
                )
                ask_question: str | None = None
                for provider_call_id, name, arguments, decode_error, persisted_id in pending:
                    if decode_error:
                        result = ToolResult(
                            call_id=persisted_id,
                            status=ToolStatus.FAILED,
                            error=ToolError(code="INVALID_TOOL_ARGUMENTS", message=decode_error),
                        )
                    elif name == "agent.ask_user":
                        problem = _validate_arguments(arguments, ASK_USER_TOOL["function"]["parameters"])
                        if problem:
                            result = ToolResult(call_id=persisted_id, status=ToolStatus.FAILED, error=ToolError(code="INVALID_TOOL_ARGUMENTS", message=problem))
                        else:
                            ask_question = str(arguments["question"]).strip()
                            result = ToolResult(call_id=persisted_id, status=ToolStatus.BLOCKED, output={"waiting_for_user": True, "question": ask_question})
                    elif tool_call_count >= self.settings.max_tool_calls:
                        result = ToolResult(
                            call_id=persisted_id,
                            status=ToolStatus.BLOCKED,
                            error=ToolError(code="BUDGET_EXCEEDED", message="已达到本次运行的工具调用上限。"),
                        )
                    elif name == "conversation.search_history":
                        problem = _validate_arguments(arguments, SEARCH_HISTORY_TOOL["function"]["parameters"])
                        if problem:
                            result = ToolResult(call_id=persisted_id, status=ToolStatus.FAILED, error=ToolError(code="INVALID_TOOL_ARGUMENTS", message=problem))
                        else:
                            limit = int(arguments.get("limit", 5))
                            if request.user_id and self.context.conversation_memory is not None:
                                found = self.context.conversation_memory.search_history(
                                    request.conversation_id,
                                    request.user_id,
                                    arguments["query"],
                                    limit=limit,
                                )
                            else:
                                found = self.store.search_messages(request.conversation_id, arguments["query"], limit=limit)
                            result = ToolResult(
                                call_id=persisted_id,
                                status=ToolStatus.SUCCESS,
                                output=[{"message_id": item.id, "role": item.role, "content": item.content[:3000]} for item in found],
                            )
                            tool_call_count += 1
                            latest = self.store.get_run(current.id) or current
                            current = latest.model_copy(update={"tool_call_count": tool_call_count})
                            self.store.save_run(current)
                    else:
                        try:
                            registered = self.registry.get(name)
                        except KeyError:
                            result = ToolResult(call_id=persisted_id, status=ToolStatus.FAILED, error=ToolError(code="UNKNOWN_TOOL", message=f"未注册工具：{name}"))
                        else:
                            problem = _validate_arguments(arguments, _exposed_tool_schema(name, registered.metadata.input_schema))
                            if name not in FIRST_STAGE_TOOLS:
                                problem = "该能力尚未在当前阶段开放。"
                            if problem:
                                result = ToolResult(call_id=persisted_id, status=ToolStatus.FAILED, error=ToolError(code="INVALID_TOOL_ARGUMENTS", message=problem))
                            else:
                                call = ToolCall(id=persisted_id, name=name, arguments=arguments, run_id=current.id, agent_id=current.agent_id)
                                result = await self.executor.execute(
                                    call,
                                    agent_id=current.agent_id,
                                    services=self.services_factory(request.user_id),
                                )
                                tool_call_count += 1
                                latest = self.store.get_run(current.id) or current
                                current = latest.model_copy(update={"tool_call_count": tool_call_count})
                                self.store.save_run(current)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": provider_call_id,
                            "content": _tool_observation(result),
                        }
                    )
                    if ask_question:
                        self._save_checkpoint(request, current, messages, cursor_id, "waiting_user")
                        return await self._finish(
                            current,
                            request=request,
                            result=AgentResult(
                                agent_id=current.agent_id,
                                status=AgentResultStatus.BLOCKED,
                                summary=ask_question,
                                error="WAITING_USER",
                                needs_input={"question": ask_question},
                                trace_id=current.id,
                            ),
                        )

                self._save_checkpoint(request, current, messages, cursor_id, f"tool_observation_{turn}")
                await self.trace.emit(
                    current.id,
                    EventType.DECISION_MADE,
                    "模型提出工具动作，执行结果已作为观察返回",
                    payload={"action": "tool_call", "tool_count": len(pending)},
                    agent_id=current.agent_id,
                )
                continue

            answer = response.content.strip()
            if not answer:
                return await self._finish(
                    current,
                    request=request,
                    result=AgentResult(
                        agent_id=current.agent_id,
                        status=AgentResultStatus.FAILED,
                        summary="模型返回了空内容，无法确认本轮结果。",
                        error="MODEL_PROTOCOL_ERROR",
                        trace_id=current.id,
                    ),
                )
            if on_model_delta is not None:
                await on_model_delta(answer)
            return await self._finish(
                current,
                request=request,
                result=AgentResult(
                    agent_id=current.agent_id,
                    status=AgentResultStatus.SUCCESS,
                    summary=answer,
                    trace_id=current.id,
                ),
            )

        current = self.store.get_run(run.id) or run
        return await self._finish(
            current,
            request=request,
            result=AgentResult(
                agent_id=current.agent_id,
                status=AgentResultStatus.BLOCKED,
                summary="已达到本次运行的模型决策轮数上限，运行已停止。",
                error="BUDGET_EXCEEDED",
                trace_id=current.id,
            ),
        )

    def _tool_definitions(self) -> list[dict[str, Any]]:
        definitions = []
        for item in self.registry.definitions():
            if item.name not in FIRST_STAGE_TOOLS:
                continue
            parameters = _exposed_tool_schema(item.name, item.input_schema)
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "description": item.description,
                        "parameters": parameters,
                    },
                }
            )
        return [*definitions, ASK_USER_TOOL, SEARCH_HISTORY_TOOL]

    def _initial_messages(self, request: AgentRequest, checkpoint: Checkpoint | None, run: Run) -> tuple[list[dict[str, Any]], str | None]:
        if checkpoint is None:
            return self.context.build(request, run=run), _latest_user_message_id(self.store, request.conversation_id)

        state = checkpoint.state
        saved = state.get("protocol_messages")
        protocol = [item for item in saved if isinstance(item, dict)] if isinstance(saved, list) else []
        cursor_id = state.get("message_cursor_id") if isinstance(state.get("message_cursor_id"), str) else None
        new_messages = self.store.list_messages_after(request.conversation_id, cursor_id)
        appended = [{"role": item.role, "content": item.content} for item in new_messages if item.role in {"user", "assistant"}]
        if not appended and not protocol:
            return self.context.build(request, run=run), _latest_user_message_id(self.store, request.conversation_id)
        protocol.extend(appended)
        built = self.context.build(request, run=run, protocol_messages=protocol, append_request=False)
        return built, _latest_user_message_id(self.store, request.conversation_id) or cursor_id

    def _save_checkpoint(
        self,
        request: AgentRequest,
        run: Run,
        messages: list[dict[str, Any]],
        cursor_id: str | None,
        phase: str,
    ) -> None:
        protocol_messages = [
            item
            for item in messages
            if item.get("role") in {"user", "assistant", "tool"}
        ]
        self.store.save_checkpoint(
            Checkpoint(
                run_id=run.id,
                phase=phase,
                state={
                    "schema_version": 1,
                    "request": request.model_dump(mode="json"),
                    "protocol_messages": protocol_messages,
                    "message_cursor_id": _latest_user_message_id(self.store, request.conversation_id) or cursor_id,
                },
            )
        )

    async def _finish(self, run: Run, result: AgentResult, *, request: AgentRequest) -> AgentResult:
        current = self.store.get_run(run.id) or run
        status = {
            AgentResultStatus.SUCCESS: RunStatus.COMPLETED,
            AgentResultStatus.PARTIAL: RunStatus.PARTIAL_COMPLETED,
            AgentResultStatus.BLOCKED: RunStatus.WAITING_USER if result.error == "WAITING_USER" else RunStatus.WAITING_APPROVAL if result.error == "APPROVAL_REQUIRED" else RunStatus.BUDGET_EXCEEDED if result.error == "BUDGET_EXCEEDED" else RunStatus.FAILED,
            AgentResultStatus.CANCELLED: RunStatus.CANCELLED,
            AgentResultStatus.FAILED: RunStatus.FAILED,
        }[result.status]
        persist_result(self.store, current, None, result, run_status=status, task_status=None)
        previous = self.store.latest_checkpoint(current.id)
        state = dict(previous.state) if previous is not None else {}
        state.update({"schema_version": 1, "request": request.model_dump(mode="json"), "result": result.model_dump(mode="json")})
        phase = "waiting_user" if status is RunStatus.WAITING_USER else "waiting_approval" if status is RunStatus.WAITING_APPROVAL else "run_completed"
        self.store.save_checkpoint(Checkpoint(run_id=current.id, phase=phase, state=state))
        event_type = EventType.RUN_COMPLETED if result.status is AgentResultStatus.SUCCESS else EventType.RUN_WAITING_USER if result.error == "WAITING_USER" else EventType.RUN_FAILED
        await self.trace.emit(current.id, event_type, result.summary, payload={"status": status.value, "error": result.error}, agent_id=current.agent_id)
        return result


def _decode_tool_calls(raw_calls: list[dict[str, Any]], run_id: str):
    messages: list[dict[str, Any]] = []
    decoded = []
    for raw in raw_calls:
        raw = raw if isinstance(raw, dict) else {}
        provider_id = str(raw.get("id") or new_id("provider_call"))
        function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
        name = str(function.get("name") or "")
        raw_arguments = function.get("arguments", "{}")
        error = None
        try:
            arguments = raw_arguments if isinstance(raw_arguments, dict) else json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            arguments = {}
            error = f"工具参数不是有效 JSON 对象：{exc}"
        persisted_id = f"{run_id}:{provider_id}"
        messages.append(
            {
                "id": provider_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        )
        decoded.append((provider_id, name, arguments, error, persisted_id))
    return messages, decoded


def _validate_arguments(value: Any, schema: dict[str, Any], path: str = "参数") -> str | None:
    expected = schema.get("type")
    valid = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected in valid and not valid[expected](value):
        return f"{path}类型错误，期望 {expected}。"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}不在允许值范围内。"
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            return f"{path}缺少必需字段：{', '.join(missing)}。"
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                return f"{path}包含未声明字段：{', '.join(sorted(extra))}。"
        for key, item in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, dict):
                problem = _validate_arguments(item, child_schema, f"{path}.{key}")
                if problem:
                    return problem
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        return f"{path}不能为空。"
    if isinstance(value, (int, float)):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}小于允许的最小值。"
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            return f"{path}必须大于 {schema['exclusiveMinimum']}。"
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            problem = _validate_arguments(item, schema["items"], f"{path}[{index}]")
            if problem:
                return problem
    return None


def _exposed_tool_schema(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    if name == "dataset.inspect":
        # 旧 Handler 的 path 分支会登记新文件；首阶段只开放已登记 ID 的只读检查。
        return {
            "type": "object",
            "properties": {"dataset_id": {"type": "string"}},
            "required": ["dataset_id"],
            "additionalProperties": False,
        }
    return schema


def _tool_observation(result: ToolResult) -> str:
    payload = {
        "status": result.status.value,
        "output": result.output,
        "datasets": result.datasets,
        "artifacts": result.artifacts,
        "warnings": result.warnings,
        "error": result.error.model_dump(mode="json") if result.error else None,
    }
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return text if len(text) <= 16000 else text[:16000] + "…(结果截断)"


def _latest_user_message_id(store: StateStore, conversation_id: str) -> str | None:
    for item in reversed(store.list_messages(conversation_id, limit=100)):
        if item.role == "user":
            return item.id
    return None


__all__ = ["AgentLoop", "LoopPreparedRequest", "FIRST_STAGE_TOOLS"]
