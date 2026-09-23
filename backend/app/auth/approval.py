"""一次性工具审批服务。

该服务只维护审批事实和精确匹配，不执行工具，也不决定生命周期。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.core.models import ApprovalRequest, ApprovalStatus, RiskLevel, Run, ToolCall, utc_now
from app.state import StateStore


def argument_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps({"tool": tool_name, "arguments": arguments}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_argument_preview(arguments: dict[str, Any], *, limit: int = 120) -> dict[str, Any]:
    hidden = ("password", "secret", "token", "api_key", "authorization", "credential", "code")

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): "<已隐藏>" if any(marker in str(key).casefold() for marker in hidden) else visit(item) for key, item in list(value.items())[:20]}
        if isinstance(value, list):
            return [visit(item) for item in value[:20]]
        if isinstance(value, str):
            return value[:limit]
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)[:limit]

    return visit(arguments)


class ApprovalService:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    def create_pending(self, run: Run, call: ToolCall, *, user_id: str, risk_level: RiskLevel, reason: str) -> ApprovalRequest:
        fingerprint = argument_fingerprint(call.name, call.arguments)
        existing = self.store.find_pending_approval(user_id=user_id, source_run_id=run.id, tool_name=call.name, argument_fingerprint=fingerprint)
        if existing is not None:
            return existing
        approval = ApprovalRequest(
            user_id=user_id,
            conversation_id=run.conversation_id,
            task_id=run.task_id,
            source_run_id=run.id,
            tool_call_id=call.id,
            tool_name=call.name,
            argument_fingerprint=fingerprint,
            risk_level=risk_level,
            argument_preview=safe_argument_preview(call.arguments),
            reason=reason,
        )
        self.store.save_approval(approval)
        return approval

    def get(self, approval_id: str, *, user_id: str) -> ApprovalRequest | None:
        return self.store.get_approval(approval_id, user_id=user_id)

    def list(self, user_id: str, *, status: ApprovalStatus | None = None, limit: int = 50) -> list[ApprovalRequest]:
        return self.store.list_approvals(user_id, status=status, limit=limit)

    def approve(self, approval_id: str, *, user_id: str, note: str | None = None, persist: bool = True) -> ApprovalRequest | None:
        approval = self.get(approval_id, user_id=user_id)
        if approval is None:
            return None
        if approval.status is not ApprovalStatus.PENDING:
            return approval
        updated = approval.model_copy(update={"status": ApprovalStatus.APPROVED, "decided_at": utc_now(), "decision_note": note})
        if persist:
            self.store.update_approval(updated)
        return updated

    def deny(self, approval_id: str, *, user_id: str, note: str | None = None, persist: bool = True) -> ApprovalRequest | None:
        approval = self.get(approval_id, user_id=user_id)
        if approval is None:
            return None
        if approval.status is not ApprovalStatus.PENDING:
            return approval
        updated = approval.model_copy(update={"status": ApprovalStatus.DENIED, "decided_at": utc_now(), "decision_note": note})
        if persist:
            self.store.update_approval(updated)
        return updated

    def bind_continuation(self, approval_id: str, *, user_id: str, continuation_run_id: str) -> ApprovalRequest | None:
        approval = self.get(approval_id, user_id=user_id)
        if approval is None:
            return None
        if approval.status is not ApprovalStatus.APPROVED:
            return approval
        updated = approval.model_copy(update={"continuation_run_id": continuation_run_id})
        self.store.update_approval(updated)
        return updated

    def consume_if_matches(
        self,
        approval_id: str,
        *,
        user_id: str,
        run_id: str | None = None,
        continuation_run_id: str | None = None,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ApprovalRequest | None:
        return self.store.consume_approval(
            approval_id=approval_id,
            user_id=user_id,
            run_id=run_id,
            continuation_run_id=continuation_run_id,
            tool_name=tool_name,
            argument_fingerprint=argument_fingerprint(tool_name, arguments),
        )

    def expire_if_unconsumed(self, approval_id: str, *, user_id: str) -> ApprovalRequest | None:
        approval = self.get(approval_id, user_id=user_id)
        if approval is None or approval.status is not ApprovalStatus.APPROVED:
            return approval
        expired = approval.model_copy(update={"status": ApprovalStatus.EXPIRED, "decided_at": approval.decided_at or utc_now()})
        self.store.update_approval(expired)
        return expired


__all__ = ["ApprovalService", "argument_fingerprint", "safe_argument_preview"]
