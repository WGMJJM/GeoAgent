"""把注册工具执行成可追踪 ToolResult。"""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import time
from threading import Event
from typing import Any

from app.auth import ApprovalService, PermissionPolicy
from app.core.models import (
    DatasetOutputPolicy,
    ErrorCategory,
    ToolCall,
    ToolError,
    ToolExecutionStatus,
    ToolResult,
    ToolStatus,
)
from app.execution.process import ProcessCancelled
from app.gis.errors import as_tool_error
from app.observability import EventType, TraceRecorder
from app.state import StateStore

from .model import ToolContext
from .registry import ToolRegistry


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        store: StateStore,
        trace: TraceRecorder,
        policy: PermissionPolicy | None = None,
        *,
        timeout_seconds: int = 120,
        metrics=None,
        approval_service: ApprovalService | None = None,
    ) -> None:
        self.registry = registry
        self.store = store
        self.trace = trace
        self.policy = policy or PermissionPolicy()
        self.timeout_seconds = timeout_seconds
        self.metrics = metrics
        self.approval_service = approval_service
        self.services: dict[str, Any] = {}
        self._cancel_events: dict[str, set[Event]] = {}

    async def execute(self, call: ToolCall, *, agent_id: str, services: dict[str, Any], approval_id: str | None = None) -> ToolResult:
        started = time.perf_counter()
        call = call.model_copy(update={"agent_id": agent_id})
        existing = self.store.get_tool_call(call.id)
        if existing is not None:
            execution_status, stored_result = existing
            if execution_status is ToolExecutionStatus.COMPLETED and stored_result is not None:
                return stored_result.model_copy(update={"call_id": call.id})
            if execution_status is ToolExecutionStatus.RUNNING:
                return _in_progress_result(call.id)
        if self.metrics:
            self.metrics.increment("tool_calls.started")
            self.metrics.increment("tool_call_count")
        try:
            registered = self.registry.get(call.name)
        except KeyError as exc:
            result = ToolResult(call_id=call.id, status=ToolStatus.FAILED, error=ToolError(code="UNKNOWN_TOOL", message=str(exc)))
            if self._claim(call):
                self.store.save_tool_call_result(call.id, result)
            if self.metrics:
                self.metrics.observe("tool_ms", (time.perf_counter() - started) * 1000)
                self.metrics.increment("tool_calls.failed")
            if call.run_id:
                await self.trace.emit(call.run_id, EventType.TOOL_FAILED, str(exc), payload={"tool": call.name, "status": result.status.value}, agent_id=agent_id)
            return result

        decision = self.policy.authorize(registered.metadata, call.arguments)
        if not decision.allowed:
            user_id = self.store.user_id_for_run(call.run_id) if call.run_id else None
            continuation_approval_id = approval_id
            current_run = self.store.get_run(call.run_id) if call.run_id else None
            if current_run is not None and continuation_approval_id is None:
                value = current_run.metadata.get("approval_id")
                continuation_approval_id = str(value) if value else None
            if decision.needs_approval and self.approval_service is not None and user_id and continuation_approval_id:
                consumed = self.approval_service.consume_if_matches(
                    continuation_approval_id,
                    user_id=user_id,
                    run_id=call.run_id or "",
                    tool_name=call.name,
                    arguments=call.arguments,
                )
                if consumed is not None:
                    await self.trace.emit(call.run_id or "unbound", EventType.APPROVAL_CONSUMED, "已消费一次性工具审批", payload={"approval_id": consumed.id, "tool": call.name, "status": consumed.status.value}, agent_id=agent_id)
                    decision = None
            if decision is not None:
                if decision.needs_approval and self.approval_service is not None and user_id and current_run is not None and call.run_id:
                    approval = self.approval_service.create_pending(
                        current_run,
                        call,
                        user_id=user_id,
                        risk_level=registered.metadata.risk_level,
                        reason=decision.reason,
                    )
                    approval_id = approval.id
                result = ToolResult(
                    call_id=call.id,
                    status=ToolStatus.BLOCKED,
                    error=ToolError(code="APPROVAL_REQUIRED", category=ErrorCategory.PERMISSION, message=decision.reason, details={"needs_approval": decision.needs_approval, "approval_id": approval_id}),
                )
                if self._claim(call):
                    self.store.save_tool_call_result(call.id, result)
                if self.metrics:
                    self.metrics.observe("tool_ms", (time.perf_counter() - started) * 1000)
                    self.metrics.increment("tool_calls.blocked")
                await self.trace.emit(call.run_id or "unbound", EventType.TOOL_FAILED, decision.reason, payload={"tool": call.name, "status": result.status.value, "approval_id": approval_id}, agent_id=agent_id)
                if approval_id:
                    await self.trace.emit(call.run_id or "unbound", EventType.APPROVAL_REQUESTED, "工具执行需要用户审批", payload={"approval_id": approval_id, "tool": call.name, "risk_level": registered.metadata.risk_level.value, "status": "PENDING"}, agent_id=agent_id)
                return result

        if not self._claim(call):
            return _in_progress_or_stored_result(self.store.get_tool_call(call.id), call.id)
        if call.run_id:
            await self.trace.emit(call.run_id, EventType.TOOL_STARTED, f"开始执行 {call.name}", payload={"tool": call.name, "arguments": _safe_arguments(call.arguments)}, agent_id=agent_id)
        cancel_event = Event()
        if call.run_id:
            self._cancel_events.setdefault(call.run_id, set()).add(cancel_event)
        context = ToolContext(run_id=call.run_id or "unbound", agent_id=agent_id, services=services, cancel_event=cancel_event)
        try:
            value = await asyncio.wait_for(asyncio.to_thread(registered.handler, call.arguments, context), timeout=self.timeout_seconds)
            if inspect.isawaitable(value):
                value = await asyncio.wait_for(value, timeout=self.timeout_seconds)
            result = _result_from_value(call.id, value)
        except asyncio.CancelledError:
            cancel_event.set()
            raise
        except ProcessCancelled:
            result = ToolResult(call_id=call.id, status=ToolStatus.CANCELLED, error=ToolError(code="EXECUTION_CANCELLED", category=ErrorCategory.EXECUTION, message="Tool 所属 Run 已取消。"))
        except subprocess.TimeoutExpired:
            result = ToolResult(
                call_id=call.id,
                status=ToolStatus.FAILED,
                retryable=registered.metadata.supports_retry,
                error=ToolError(
                    code="EXECUTION_TIMEOUT",
                    category=ErrorCategory.EXECUTION,
                    message=f"Tool 超过 {self.timeout_seconds}s 未完成。",
                    retryable=registered.metadata.supports_retry,
                    details={"timeout_seconds": self.timeout_seconds},
                ),
            )
        except TimeoutError:
            cancel_event.set()
            result = ToolResult(call_id=call.id, status=ToolStatus.FAILED, retryable=registered.metadata.supports_retry, error=ToolError(code="EXECUTION_TIMEOUT", category=ErrorCategory.EXECUTION, message=f"Tool 超过 {self.timeout_seconds}s 未完成。", retryable=registered.metadata.supports_retry))
        except Exception as exc:
            error = as_tool_error(exc)
            result = ToolResult(call_id=call.id, status=ToolStatus.FAILED, retryable=error.retryable, error=error)
        finally:
            if call.run_id:
                events = self._cancel_events.get(call.run_id)
                if events:
                    events.discard(cancel_event)
                    if not events:
                        self._cancel_events.pop(call.run_id, None)
        result = result.model_copy(update={"duration_ms": round((time.perf_counter() - started) * 1000, 2)})
        if self.metrics:
            self.metrics.observe("tool_ms", result.duration_ms or 0)
            if result.status is ToolStatus.CANCELLED:
                self.metrics.increment("tool_calls.cancelled")
            elif result.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS}:
                self.metrics.increment("tool_calls.completed")
            else:
                self.metrics.increment("tool_calls.failed")
            if result.error and result.error.code == "EXECUTION_TIMEOUT":
                self.metrics.increment("tool_calls.timed_out")
        self.store.save_tool_call_result(call.id, result)
        if call.run_id:
            event_type = EventType.TOOL_COMPLETED if result.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS} else EventType.TOOL_FAILED
            await self.trace.emit(call.run_id, event_type, f"{call.name}: {result.status.value}", payload={"tool": call.name, "status": result.status.value, "error": result.error.model_dump(mode="json") if result.error else None, "duration_ms": result.duration_ms}, agent_id=agent_id)
            if result.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS} and registered.metadata.dataset_output_policy is DatasetOutputPolicy.REQUIRED:
                for dataset_id in result.datasets:
                    await self.trace.emit(call.run_id, EventType.DATASET_CREATED, f"Dataset 已登记：{dataset_id}", payload={"dataset_id": dataset_id, "tool": call.name}, agent_id=agent_id)
            if result.status in {ToolStatus.SUCCESS, ToolStatus.PARTIAL_SUCCESS} and registered.metadata.produces_artifact:
                for artifact_id in result.artifacts:
                    await self.trace.emit(call.run_id, EventType.ARTIFACT_CREATED, f"Artifact 已发布：{artifact_id}", payload={"artifact_id": artifact_id, "tool": call.name}, agent_id=agent_id)
        return result

    def _claim(self, call: ToolCall) -> bool:
        """以 call.id 原子占用一次执行机会，防止并发重复副作用。"""

        return self.store.mark_tool_call_running(call)

    def cancel_run(self, run_id: str) -> None:
        for event in self._cancel_events.get(run_id, ()):
            event.set()


def _result_from_value(call_id: str, value: Any) -> ToolResult:
    if isinstance(value, ToolResult):
        return value.model_copy(update={"call_id": call_id})
    if not isinstance(value, dict):
        return ToolResult(call_id=call_id, status=ToolStatus.SUCCESS, output=value)
    status = ToolStatus(value.get("status", ToolStatus.SUCCESS)) if "status" in value else ToolStatus.SUCCESS
    return ToolResult(call_id=call_id, status=status, output=value.get("output", value), datasets=value.get("datasets", []), artifacts=value.get("artifacts", []), warnings=value.get("warnings", []), retryable=bool(value.get("retryable", False)))


def _in_progress_result(call_id: str) -> ToolResult:
    return ToolResult(
        call_id=call_id,
        status=ToolStatus.BLOCKED,
        error=ToolError(
            code="TOOL_CALL_IN_PROGRESS",
            category=ErrorCategory.EXECUTION,
            message="相同 call.id 的工具调用正在执行，当前请求不会重复触发副作用。",
            details={"recovery": "等待原调用完成后读取持久化结果。"},
        ),
    )


def _in_progress_or_stored_result(existing: tuple[ToolExecutionStatus, ToolResult | None] | None, call_id: str) -> ToolResult:
    if existing is not None and existing[0] is ToolExecutionStatus.COMPLETED and existing[1] is not None:
        return existing[1].model_copy(update={"call_id": call_id})
    return _in_progress_result(call_id)


def _safe_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    hidden = {"code", "api_key", "token"}
    return {key: "<redacted>" if key.casefold() in hidden else value for key, value in arguments.items()}
