"""GeoAgent 唯一的模型决策与工具观察循环。"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.auth.policy import ToolDiscoveryContext
from app.config import Settings
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
from app.execution.tools import (
    TOOL_SEARCH_DEFINITION,
    ToolCatalog,
    ToolExecutor,
    ToolRegistry,
    validate_arguments,
)
from app.models import ModelAdapter, ModelRequest
from app.observability import EventType, TraceRecorder
from app.run.lifecycle import persist_result
from app.state import StateStore

from .context import ContextBuilder
from .delegation import DELEGATE_TOOL

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
        self.catalog = ToolCatalog(registry, executor.policy.is_discoverable)
        self.policy = executor.policy
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
        self.delegation = None

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

    def prepare_child_request(self, request: AgentRequest, child: Run) -> LoopPreparedRequest:
        run = child.model_copy(update={"status": RunStatus.RUNNING, "started_at": datetime.now(UTC)})
        self.store.save_run(run)
        self._save_checkpoint(request, run, [], None, "child_prepared", set(), [], pending_tool_calls=[])
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

        messages, cursor_id = self._initial_messages(request, resume_from, run, continuation)
        activated_names = self._restore_activated_names(resume_from, request, run)
        pending_approvals = self._pending_approvals(resume_from)
        tool_call_count = run.tool_call_count
        saved_pending = resume_from.state.get("pending_tool_calls", []) if resume_from else []
        resume_pending = [tuple(item) for item in saved_pending if isinstance(item, list) and len(item) == 6]

        if pending_approvals:
            if isinstance(continuation, dict) and continuation.get("type") == "user_input":
                self._save_checkpoint(
                    request,
                    run,
                    messages,
                    cursor_id,
                    "waiting_approval",
                    activated_names,
                    pending_approvals,
                )
                first = pending_approvals[0]
                return await self._finish(
                    run,
                    request=request,
                    result=AgentResult(
                        agent_id=run.agent_id,
                        status=AgentResultStatus.BLOCKED,
                        summary=f"已记录补充信息；工具 {first.get('tool_name', '')} 仍在等待审批。",
                        error="APPROVAL_REQUIRED",
                        needs_input={"approval_id": first.get("approval_id"), "tool": first.get("tool_name")},
                        trace_id=run.id,
                    ),
                )
            resumed = await self._resume_approval(
                request,
                run,
                messages,
                cursor_id,
                activated_names,
                pending_approvals,
                continuation,
            )
            if resumed is not None:
                return resumed

        remaining_turns = max(0, self.settings.max_agent_turns - run.turn_count)
        for turn in range(remaining_turns + bool(resume_pending)):
            services = self._execution_services(request, run)
            runtime_context = self._discovery_context(request, services, run)
            activated_names = self._available_activations(activated_names, runtime_context)
            tools = self._tool_definitions(runtime_context, activated_names)
            current = self.store.get_run(run.id) or run
            current = current.model_copy(update={"status": RunStatus.RUNNING, "turn_count": current.turn_count + (not resume_pending)})
            self.store.save_run(current)
            response = None
            try:
                if not resume_pending:
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

            if response is None or response.tool_calls:
                restoring_batch = bool(resume_pending)
                if restoring_batch:
                    pending, resume_pending = resume_pending, []
                else:
                    assistant_calls, pending = _decode_tool_calls(response.tool_calls, current.id)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response.content or "",
                            "tool_calls": assistant_calls,
                        }
                    )
                ask_question: str | None = None
                delegate_wait: ToolResult | None = None
                execution_uncertain = False
                next_activations: set[str] | None = None
                permitted_this_batch = frozenset(activated_names)
                if restoring_batch and resume_from:
                    permitted_this_batch = frozenset(resume_from.state.get("batch_activated_names", activated_names))
                    saved_next = resume_from.state.get("batch_next_activations")
                    next_activations = set(saved_next) if isinstance(saved_next, list) else None
                available_names_this_batch = self._available_tool_names(
                    self._discovery_context(request, self._execution_services(request, current), current),
                    permitted_this_batch,
                )
                mixed_delegation = len(pending) > 1 and any(item[1] == "agent.delegate" for item in pending)
                for index, (provider_call_id, name, arguments, decode_error, persisted_id, counted) in enumerate(pending):
                    within_budget = counted or tool_call_count < self.settings.max_tool_calls
                    if not within_budget:
                        result = _blocked_result(
                            persisted_id,
                            "BUDGET_EXCEEDED",
                            "已达到本次运行的工具调用上限。",
                        )
                    elif not counted:
                        tool_call_count += 1
                        latest = self.store.get_run(current.id) or current
                        current = latest.model_copy(update={"tool_call_count": tool_call_count})
                        self.store.save_run(current)
                        pending[index] = (provider_call_id, name, arguments, decode_error, persisted_id, True)

                    # 模型已提出的原子调用先保存；恢复用原 ID 读取结果，禁止重新询问模型制造新副作用。
                    self._save_checkpoint(request, current, messages, cursor_id, "tool_pending", activated_names,
                                          pending_approvals, pending_tool_calls=pending[index:],
                                          batch_activated_names=permitted_this_batch, batch_next_activations=next_activations)

                    if not within_budget:
                        pass
                    elif mixed_delegation:
                        result = _failed_result(persisted_id, "DELEGATION_MIXED_BATCH", "agent.delegate 必须独占一个模型工具批次。")
                    elif decode_error:
                        result = ToolResult(
                            call_id=persisted_id,
                            status=ToolStatus.FAILED,
                            error=ToolError(code="INVALID_TOOL_ARGUMENTS", message=decode_error),
                        )
                    elif name == "agent.delegate":
                        if current.parent_run_id:
                            result = _blocked_result(persisted_id, "SUBAGENT_DELEGATION_FORBIDDEN", "子 Agent 不能再次委派。")
                        elif self.delegation is None:
                            result = _failed_result(persisted_id, "DELEGATION_UNAVAILABLE", "委派调度器未配置。")
                        else:
                            result = await self.delegation.execute(arguments, request=request, parent=current,
                                                                   call_id=persisted_id, continuation=continuation)
                            if result.error and result.error.code in {"WAITING_USER", "APPROVAL_REQUIRED"}:
                                delegate_wait = result
                            elif isinstance(result.output, dict) and "delegation_id" in result.output:
                                latest = self.store.get_run(current.id) or current
                                current = latest.model_copy(update={"metadata": {**latest.metadata, "last_delegation_status": result.output["status"],
                                                                                 "delegation_dataset_ids": result.datasets,
                                                                                 "delegation_artifact_ids": result.artifacts}})
                                self.store.save_run(current)
                    elif name == "agent.ask_user":
                        problem = validate_arguments(arguments, ASK_USER_TOOL["function"]["parameters"])
                        if problem:
                            result = ToolResult(call_id=persisted_id, status=ToolStatus.FAILED, error=ToolError(code="INVALID_TOOL_ARGUMENTS", message=problem))
                        else:
                            ask_question = str(arguments["question"]).strip()
                            result = ToolResult(call_id=persisted_id, status=ToolStatus.BLOCKED, output={"waiting_for_user": True, "question": ask_question})
                    elif name == "tool.search":
                        problem = validate_arguments(arguments, TOOL_SEARCH_DEFINITION["function"]["parameters"])
                        if problem:
                            result = _failed_result(persisted_id, "INVALID_TOOL_ARGUMENTS", problem)
                        else:
                            search_services = self._execution_services(request, current)
                            search_context = self._discovery_context(request, search_services, current)
                            try:
                                search_output = self.catalog.tool_search(arguments, search_context)
                            except ValueError as exc:
                                result = _failed_result(persisted_id, "INVALID_TOOL_ARGUMENTS", str(exc))
                            else:
                                result = ToolResult(call_id=persisted_id, status=ToolStatus.SUCCESS, output=search_output)
                                # 本批次多个检索取并集；只在下一轮启用，不覆盖前一次结果。
                                if next_activations is None:
                                    next_activations = set()
                                next_activations.update(item["name"] for item in search_output["tools"])
                    elif name == "conversation.search_history":
                        problem = validate_arguments(arguments, SEARCH_HISTORY_TOOL["function"]["parameters"])
                        if current.parent_run_id:
                            result = _blocked_result(persisted_id, "SUBAGENT_HISTORY_FORBIDDEN", "子任务不能读取主会话历史。")
                        elif problem:
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
                    else:
                        try:
                            registered = self.registry.get(name)
                        except KeyError:
                            result = _failed_result(persisted_id, "UNKNOWN_TOOL", f"未注册工具：{name}")
                        else:
                            if self.registry.is_deferred(name) and name not in permitted_this_batch:
                                result = _blocked_result(
                                    persisted_id,
                                    "DEFERRED_TOOL_NOT_ACTIVE",
                                    "该延迟工具未在当前 Run 中激活，请先使用 tool.search。",
                                )
                            elif name not in available_names_this_batch:
                                result = _blocked_result(
                                    persisted_id,
                                    "TOOL_NOT_AVAILABLE",
                                    "当前用户权限或运行环境不允许使用该工具。",
                                )
                            else:
                                problem = validate_arguments(arguments, registered.metadata.input_schema)
                                if problem:
                                    result = _failed_result(persisted_id, "INVALID_TOOL_ARGUMENTS", problem)
                                else:
                                    execution_services = self._execution_services(request, current)
                                    execution_context = self._discovery_context(request, execution_services, current)
                                    allowed_now = self._available_tool_names(execution_context, permitted_this_batch)
                                    call = ToolCall(id=persisted_id, name=name, arguments=arguments, run_id=current.id, agent_id=current.agent_id)
                                    result = await self.executor.execute(
                                        call,
                                        agent_id=current.agent_id,
                                        services=execution_services,
                                        active_tool_names=allowed_now,
                                        discovery_context=execution_context,
                                    )
                                    if result.error and result.error.code == "APPROVAL_REQUIRED":
                                        approval_id = result.error.details.get("approval_id")
                                        if approval_id:
                                            approval = {
                                                "approval_id": str(approval_id),
                                                "run_id": current.id,
                                                "provider_call_id": provider_call_id,
                                                "call_id": persisted_id,
                                                "tool_name": name,
                                                "arguments": arguments,
                                            }
                                            pending_approvals.append(approval)
                    if current.parent_run_id and name not in {"agent.ask_user", "tool.search", "conversation.search_history", "agent.delegate"}:
                        approval_waiting = result.error is not None and result.error.code == "APPROVAL_REQUIRED" and result.error.details.get("approval_id")
                        if not approval_waiting and self.store.get_tool_call(persisted_id) is None:
                            self.store.save_tool_call(ToolCall(id=persisted_id, name=name, arguments=arguments, run_id=current.id), result)
                    _replace_tool_observation(messages, provider_call_id, _tool_observation(result))
                    remaining = pending[index:] if delegate_wait else pending[index + 1:]
                    self._save_checkpoint(request, current, messages, cursor_id, "tool_observation", activated_names,
                                          pending_approvals, pending_tool_calls=remaining,
                                          batch_activated_names=permitted_this_batch, batch_next_activations=next_activations)
                    if result.error and result.error.code == "SIDE_EFFECT_UNCERTAIN":
                        execution_uncertain = True
                        break
                if next_activations is not None:
                    activated_names = next_activations
                self._save_checkpoint(
                    request,
                    current,
                    messages,
                    cursor_id,
                    "tool_observation",
                    activated_names,
                    pending_approvals,
                    pending_tool_calls=remaining,
                )
                await self.trace.emit(
                    current.id,
                    EventType.DECISION_MADE,
                    "模型提出工具动作，执行结果已作为观察返回",
                    payload={"action": "tool_call", "tool_count": len(pending)},
                    agent_id=current.agent_id,
                )
                if delegate_wait is not None:
                    details = delegate_wait.error.details
                    return await self._finish(current, request=request, result=AgentResult(
                        agent_id=current.agent_id, task_id=(self.store.get_run(current.id) or current).task_id,
                        status=AgentResultStatus.BLOCKED, summary=delegate_wait.error.message,
                        error=delegate_wait.error.code, needs_input=details, trace_id=current.id))
                if execution_uncertain:
                    return await self._finish(current, request=request, result=AgentResult(
                        agent_id=current.agent_id, status=AgentResultStatus.FAILED,
                        summary="工具清理未完成，副作用无法确认；运行已停止，需要人工处理。",
                        error="SIDE_EFFECT_UNCERTAIN", trace_id=current.id))
                if ask_question:
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
                if pending_approvals:
                    first = pending_approvals[0]
                    return await self._finish(
                        current,
                        request=request,
                        result=AgentResult(
                            agent_id=current.agent_id,
                            status=AgentResultStatus.BLOCKED,
                            summary=f"工具 {first['tool_name']} 等待审批后才能继续。",
                            error="APPROVAL_REQUIRED",
                            needs_input={"approval_id": first["approval_id"], "tool": first["tool_name"]},
                            trace_id=current.id,
                        ),
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

    def _tool_definitions(
        self,
        context: ToolDiscoveryContext | None = None,
        activated_names: set[str] | frozenset[str] = frozenset(),
    ) -> list[dict[str, Any]]:
        context = context or ToolDiscoveryContext()
        definitions = [TOOL_SEARCH_DEFINITION, ASK_USER_TOOL]
        if context.allowed_tool_names is None:
            definitions.append(SEARCH_HISTORY_TOOL)
            if self.delegation is not None:
                definitions.append(DELEGATE_TOOL)
        available = self._available_tool_names(context, activated_names)
        for item in self.registry.definitions():
            if item.name not in available:
                continue
            definitions.append(_tool_definition(item.name, item.description, item.input_schema))
        return definitions

    def _discovery_context(self, request: AgentRequest, services: dict[str, Any], run: Run | None = None) -> ToolDiscoveryContext:
        context = self.policy.discovery_context(authenticated_user=bool(request.user_id), services=services)
        current = self.store.get_run(run.id) if run is not None else None
        parent = self.store.get_run(current.parent_run_id) if current is not None and current.parent_run_id else None
        return self.policy.restrict_to_run(context, current, parent)

    def _execution_services(self, request: AgentRequest, run: Run) -> dict[str, Any]:
        services = self.services_factory(request.user_id)
        if run.parent_run_id:
            services = dict(services)
            workspace = services.get("workspace")
            if hasattr(workspace, "for_run"):
                services["workspace"] = workspace.for_run(run.id)
            registry = services.get("registry")
            if hasattr(registry, "for_run"):
                services["registry"] = registry.for_run(request.dataset_ids, run.id)
        return services

    def _available_activations(
        self,
        activated_names: set[str] | frozenset[str],
        context: ToolDiscoveryContext,
    ) -> set[str]:
        available = set()
        for name in activated_names:
            if not self.registry.is_deferred(name):
                continue
            try:
                metadata = self.registry.get(name).metadata
            except KeyError:
                continue
            if self.policy.is_discoverable(metadata, context):
                available.add(name)
        return available

    def _available_tool_names(
        self,
        context: ToolDiscoveryContext,
        activated_names: set[str] | frozenset[str],
    ) -> frozenset[str]:
        names = set()
        for metadata in self.registry.definitions():
            if self.registry.is_deferred(metadata.name) and metadata.name not in activated_names:
                continue
            if self.policy.is_discoverable(metadata, context):
                names.add(metadata.name)
        return frozenset(names)

    def _restore_activated_names(self, checkpoint: Checkpoint | None, request: AgentRequest, run: Run | None = None) -> set[str]:
        if checkpoint is None:
            return set()
        raw = checkpoint.state.get("activated_tool_names")
        if not isinstance(raw, list):
            return set()
        services = self.services_factory(request.user_id)
        context = self._discovery_context(request, services, run)
        return self._available_activations({name for name in raw if isinstance(name, str)}, context)

    @staticmethod
    def _pending_approvals(checkpoint: Checkpoint | None) -> list[dict[str, Any]]:
        if checkpoint is None:
            return []
        raw = checkpoint.state.get("pending_approvals")
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    async def _resume_approval(
        self,
        request: AgentRequest,
        run: Run,
        messages: list[dict[str, Any]],
        cursor_id: str | None,
        activated_names: set[str],
        pending_approvals: list[dict[str, Any]],
        continuation: dict[str, object] | None,
    ) -> AgentResult | None:
        if not isinstance(continuation, dict) or continuation.get("type") != "approval_result":
            first = pending_approvals[0]
            return await self._finish(
                run,
                request=request,
                result=AgentResult(
                    agent_id=run.agent_id,
                    status=AgentResultStatus.BLOCKED,
                    summary=f"工具 {first.get('tool_name', '')} 仍在等待审批。",
                    error="APPROVAL_REQUIRED",
                    needs_input={"approval_id": first.get("approval_id"), "tool": first.get("tool_name")},
                    trace_id=run.id,
                ),
            )

        approval_id = continuation.get("approval_id")
        index = next((i for i, item in enumerate(pending_approvals) if item.get("approval_id") == approval_id), None)
        if index is None:
            return await self._finish(
                run,
                request=request,
                result=AgentResult(
                    agent_id=run.agent_id,
                    status=AgentResultStatus.FAILED,
                    summary="审批恢复上下文与当前运行不匹配，工具没有执行。",
                    error="APPROVAL_MISMATCH",
                    trace_id=run.id,
                ),
            )
        pending = pending_approvals.pop(index)
        if pending.get("run_id") != run.id:
            return await self._finish(
                run,
                request=request,
                result=AgentResult(agent_id=run.agent_id, status=AgentResultStatus.FAILED, summary="审批绑定的运行不匹配。", error="APPROVAL_MISMATCH", trace_id=run.id),
            )

        call_id = pending.get("call_id")
        provider_call_id = pending.get("provider_call_id")
        name = pending.get("tool_name")
        arguments = pending.get("arguments")
        if not isinstance(call_id, str) or not isinstance(provider_call_id, str) or not isinstance(name, str) or not isinstance(arguments, dict):
            return await self._finish(
                run,
                request=request,
                result=AgentResult(agent_id=run.agent_id, status=AgentResultStatus.FAILED, summary="审批保存的工具调用数据无效。", error="APPROVAL_MISMATCH", trace_id=run.id),
            )

        if continuation.get("approved") is True:
            services = self._execution_services(request, run)
            context = self._discovery_context(request, services, run)
            available = self._available_tool_names(context, activated_names)
            result = await self.executor.execute(
                ToolCall(id=call_id, name=name, arguments=arguments, run_id=run.id, agent_id=run.agent_id),
                agent_id=run.agent_id,
                services=services,
                approval_id=str(approval_id),
                active_tool_names=available,
                discovery_context=context,
            )
        elif continuation.get("approved") is False:
            result = _blocked_result(call_id, "APPROVAL_DENIED", "用户拒绝了此工具操作；工具没有执行。")
            if run.parent_run_id:
                self.store.save_tool_call(ToolCall(id=call_id, name=name, arguments=arguments, run_id=run.id), result)
        else:
            return await self._finish(
                run,
                request=request,
                result=AgentResult(agent_id=run.agent_id, status=AgentResultStatus.FAILED, summary="审批结果无效，工具没有执行。", error="APPROVAL_MISMATCH", trace_id=run.id),
            )

        _replace_tool_observation(messages, provider_call_id, _tool_observation(result))
        if pending_approvals:
            self._save_checkpoint(request, run, messages, cursor_id, "waiting_approval", activated_names, pending_approvals)
            first = pending_approvals[0]
            return await self._finish(
                run,
                request=request,
                result=AgentResult(
                    agent_id=run.agent_id,
                    status=AgentResultStatus.BLOCKED,
                    summary=f"工具 {first.get('tool_name', '')} 仍在等待审批。",
                    error="APPROVAL_REQUIRED",
                    needs_input={"approval_id": first.get("approval_id"), "tool": first.get("tool_name")},
                    trace_id=run.id,
                ),
            )
        self._save_checkpoint(request, run, messages, cursor_id, "approval_resolved", activated_names, [])
        return None

    def _initial_messages(
        self,
        request: AgentRequest,
        checkpoint: Checkpoint | None,
        run: Run,
        continuation: dict[str, object] | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if checkpoint is None:
            return self.context.build(request, run=run), _latest_user_message_id(self.store, request.conversation_id)

        state = checkpoint.state
        saved = state.get("protocol_messages")
        protocol = [dict(item) for item in saved if isinstance(item, dict)] if isinstance(saved, list) else []
        cursor_id = state.get("message_cursor_id") if isinstance(state.get("message_cursor_id"), str) else None
        if isinstance(continuation, dict) and continuation.get("type") == "user_input":
            content = continuation.get("content")
            if isinstance(content, str) and content.strip():
                protocol.append({"role": "user", "content": content.strip()})
        if not protocol:
            return self.context.build(request, run=run), _latest_user_message_id(self.store, request.conversation_id)
        built = self.context.build(request, run=run, protocol_messages=protocol, append_request=False)
        return built, cursor_id

    def _save_checkpoint(
        self,
        request: AgentRequest,
        run: Run,
        messages: list[dict[str, Any]],
        cursor_id: str | None,
        phase: str,
        activated_names: set[str] | frozenset[str],
        pending_approvals: list[dict[str, Any]],
        *,
        pending_tool_calls=None,
        batch_activated_names=None,
        batch_next_activations=None,
    ) -> None:
        protocol_messages = [
            item
            for item in messages
            if item.get("role") in {"user", "assistant", "tool"}
        ]
        previous = self.store.latest_checkpoint(run.id)
        pending_tool_calls = pending_tool_calls if pending_tool_calls is not None else (previous.state.get("pending_tool_calls", []) if previous else [])
        self.store.save_checkpoint(
            Checkpoint(
                run_id=run.id,
                phase=phase,
                state={
                    "schema_version": 1,
                    "request": request.model_dump(mode="json"),
                    "protocol_messages": protocol_messages,
                    "message_cursor_id": cursor_id,
                    "activated_tool_names": sorted(activated_names),
                    "pending_approvals": pending_approvals,
                    "pending_tool_calls": pending_tool_calls,
                    "batch_activated_names": sorted(batch_activated_names if batch_activated_names is not None else activated_names),
                    "batch_next_activations": sorted(batch_next_activations) if batch_next_activations is not None else None,
                    "tool_call_count": run.tool_call_count,
                },
            )
        )

    async def _finish(self, run: Run, result: AgentResult, *, request: AgentRequest) -> AgentResult:
        current = self.store.get_run(run.id) or run
        if current.task_id:
            result = result.model_copy(update={"task_id": current.task_id})
        if not current.parent_run_id and result.status is AgentResultStatus.SUCCESS and current.metadata.get("last_delegation_status"):
            delegated_status = AgentResultStatus(current.metadata["last_delegation_status"])
            result = result.model_copy(update={"status": delegated_status,
                                               "error": "DELEGATION_FAILED" if delegated_status is AgentResultStatus.FAILED else "CANCELLED" if delegated_status is AgentResultStatus.CANCELLED else None,
                                               "datasets": current.metadata.get("delegation_dataset_ids", []),
                                               "artifacts": current.metadata.get("delegation_artifact_ids", [])})
        status = {
            AgentResultStatus.SUCCESS: RunStatus.COMPLETED,
            AgentResultStatus.PARTIAL: RunStatus.PARTIAL_COMPLETED,
            AgentResultStatus.BLOCKED: RunStatus.WAITING_USER if result.error == "WAITING_USER" else RunStatus.WAITING_APPROVAL if result.error == "APPROVAL_REQUIRED" else RunStatus.BUDGET_EXCEEDED if result.error == "BUDGET_EXCEEDED" else RunStatus.FAILED,
            AgentResultStatus.CANCELLED: RunStatus.CANCELLED,
            AgentResultStatus.FAILED: RunStatus.FAILED,
        }[result.status]
        task = self.store.get_task(current.task_id) if current.task_id and not current.parent_run_id else None
        persist_result(self.store, current, task, result, run_status=status, task_status=None)
        previous = self.store.latest_checkpoint(current.id)
        state = dict(previous.state) if previous is not None else {}
        state.update({"schema_version": 1, "request": request.model_dump(mode="json"), "result": result.model_dump(mode="json")})
        phase = "waiting_user" if status is RunStatus.WAITING_USER else "waiting_approval" if status is RunStatus.WAITING_APPROVAL else "run_completed"
        self.store.save_checkpoint(Checkpoint(run_id=current.id, phase=phase, state=state))
        event_type = (
            EventType.RUN_COMPLETED
            if result.status is AgentResultStatus.SUCCESS
            else EventType.RUN_WAITING_USER
            if result.error == "WAITING_USER"
            else EventType.RUN_WAITING_APPROVAL
            if result.error == "APPROVAL_REQUIRED"
            else EventType.RUN_CANCELLED
            if result.status is AgentResultStatus.CANCELLED
            else EventType.RUN_FAILED
        )
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
            if isinstance(raw_arguments, dict):
                arguments = raw_arguments
                protocol_arguments = json.dumps(raw_arguments, ensure_ascii=False)
            elif isinstance(raw_arguments, str):
                protocol_arguments = raw_arguments
                arguments = json.loads(raw_arguments)
            else:
                protocol_arguments = json.dumps(raw_arguments, ensure_ascii=False)
                arguments = json.loads(protocol_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            arguments = {}
            protocol_arguments = raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments, ensure_ascii=False)
            error = f"工具参数不是有效 JSON 对象：{exc}"
        persisted_id = f"{run_id}:{provider_id}"
        messages.append(
            {
                "id": provider_id,
                "type": "function",
                "function": {"name": name, "arguments": protocol_arguments},
            }
        )
        decoded.append((provider_id, name, arguments, error, persisted_id, False))
    return messages, decoded


def _tool_definition(name: str, description: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema},
    }


def _failed_result(call_id: str, code: str, message: str) -> ToolResult:
    return ToolResult(call_id=call_id, status=ToolStatus.FAILED, error=ToolError(code=code, message=message))


def _blocked_result(call_id: str, code: str, message: str) -> ToolResult:
    return ToolResult(call_id=call_id, status=ToolStatus.BLOCKED, error=ToolError(code=code, message=message))


def _replace_tool_observation(messages: list[dict[str, Any]], provider_call_id: str, content: str) -> None:
    for item in reversed(messages):
        if item.get("role") == "tool" and item.get("tool_call_id") == provider_call_id:
            item["content"] = content
            return
    messages.append({"role": "tool", "tool_call_id": provider_call_id, "content": content})


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
    if isinstance(result.output, dict) and "delegation_id" in result.output:
        return text  # 委派观察已按字段裁剪，不从中间截断关键 ID/状态或破坏 JSON。
    return text if len(text) <= 16000 else text[:16000] + "…(结果截断)"


def _latest_user_message_id(store: StateStore, conversation_id: str) -> str | None:
    for item in reversed(store.list_messages(conversation_id, limit=100)):
        if item.role == "user":
            return item.id
    return None


__all__ = ["AgentLoop", "LoopPreparedRequest"]
