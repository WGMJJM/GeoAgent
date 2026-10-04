"""有限自动恢复；业务修正仍由现有 Agent Loop 决定。"""

from __future__ import annotations

import asyncio
import math
import random
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
import httpx2
from openai import APIConnectionError, APIStatusError

from app.core.models import Run, ToolExecutionStatus
from app.observability import EventType

from .checkpoints import RunCheckpointCodec


def transient_error(error: Exception) -> bool:
    """只读协议状态及异常类型，不匹配错误文字或工具名称。"""
    if isinstance(error, (APIConnectionError, httpx.TransportError, httpx2.TransportError,
                          ConnectionError, TimeoutError)):
        return True
    if isinstance(error, (APIStatusError, httpx.HTTPStatusError, httpx2.HTTPStatusError)):
        response = error.response
        # 配额/计费类拒绝不会因为原样重试恢复。
        if isinstance(error, APIStatusError) and error.code in {"insufficient_quota", "billing_hard_limit_reached"}:
            return False
        return response.status_code in {408, 429} or 500 <= response.status_code < 600
    return False


def retry_after(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    if response is None:
        return None
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


class RecoveryController:
    def __init__(self, store, settings, trace) -> None:
        self.store, self.settings, self.trace = store, settings, trace

    def current(self, run_id: str) -> dict:
        checkpoint = self.store.latest_checkpoint(run_id)
        return checkpoint.state.get("recovery") or {} if checkpoint is not None else {}

    def begin(self, run_id: str, scope: str, operation_id: str) -> dict:
        previous = self.current(run_id)
        if previous.get("scope") == scope and previous.get("operation_id") == operation_id:
            return previous
        record = self.store.get_tool_call_record(operation_id) if scope == "tool" else None
        state = {"scope": scope, "operation_id": operation_id,
                 "retries_used": record[0].attempt - 1 if record else 0,
                 "next_retry_at": None, "status": "running", "retryable": False,
                 "published_text": False}
        self.store.update_recovery(run_id, state)
        return state

    def published(self, run_id: str) -> None:
        state = self.current(run_id)
        self.store.update_recovery(run_id, {**state, "published_text": True})

    async def schedule(self, run_id: str, *, retryable: bool, error_code: str,
                       server_delay: float | None = None) -> bool:
        state = self.current(run_id)
        state = {**state, "status": "failed", "retryable": retryable,
                 "error_code": error_code, "next_retry_at": None}
        self.store.update_recovery(run_id, state)
        if not retryable or state.get("published_text"):
            return False
        maximum = self.settings.max_tool_retries if state["scope"] == "tool" else self.settings.max_model_retries
        delay = min(self.settings.retry_max_delay_seconds,
                    self.settings.retry_delay_seconds * 2 ** state["retries_used"])
        delay += random.uniform(0, delay * 0.25)
        if server_delay is not None:
            delay = max(delay, server_delay)
        pending = {**state, "status": "waiting", "retries_used": state["retries_used"] + 1,
                   "next_retry_at": datetime.fromtimestamp(datetime.now(UTC).timestamp() + delay, UTC).isoformat()}
        reserved = self.store.reserve_recovery_retry(
            run_id, recovery=pending, max_retries=maximum,
            max_run_retries=self.settings.max_run_retries, max_tool_calls=self.settings.max_tool_calls,
        )
        if reserved:
            await self.trace.emit(run_id, EventType.RETRY_STARTED, "正在恢复临时失败的步骤",
                                  payload={"scope": state["scope"], "operation_id": state["operation_id"],
                                           "retry": pending["retries_used"], "delay_seconds": round(delay, 3),
                                           "error": error_code})
        return reserved

    async def wait(self, run_id: str) -> None:
        state = self.current(run_id)
        if state.get("status") == "waiting":
            remaining = (datetime.fromisoformat(state["next_retry_at"]) - datetime.now(UTC)).total_seconds()
            await asyncio.sleep(max(0, remaining))
            self.store.update_recovery(run_id, {**state, "status": "running", "next_retry_at": None})

    def clear(self, run_id: str) -> None:
        self.store.update_recovery(run_id, None)

    def can_resume(self, run: Run) -> bool:
        """只能自动接续有明确游标的运行，未知副作用和已发布正文除外。"""
        checkpoint = self.store.latest_checkpoint(run.id)
        if checkpoint is None or checkpoint.state.get("schema_version", 0) < RunCheckpointCodec.RECOVERY_VERSION:
            return False
        state = checkpoint.state.get("recovery") or {}
        if state.get("published_text") or any(
            status is ToolExecutionStatus.RUNNING for _, status, _ in self.store.list_tool_calls(run.id)
        ):
            return False
        if checkpoint.state.get("pending_approvals"):
            return False
        if run.status.value == "FAILED":
            if not state.get("retryable") or state.get("status") != "failed":
                return False
            maximum = self.settings.max_tool_retries if state["scope"] == "tool" else self.settings.max_model_retries
            if state["retries_used"] >= maximum:
                return False
        return bool(state or checkpoint.state.get("pending_tool_calls") or checkpoint.state.get("protocol_messages"))
