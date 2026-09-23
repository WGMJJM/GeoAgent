"""GeoAgent 的最小权限策略。

读操作和生成新结果默认允许；覆盖/删除/外部访问等需要显式 approval，而第一版
不会偷偷降级执行。这样工具失败可以被 Agent 看见并转化为 BLOCKED/ASK_USER。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.models import RiskLevel, ToolMetadata


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    needs_approval: bool = False
    reason: str = ""


class PermissionPolicy:
    def __init__(self, *, allow_approval: bool = False) -> None:
        self.allow_approval = allow_approval

    def authorize(self, metadata: ToolMetadata, arguments: dict) -> PermissionDecision:
        if metadata.risk_level in {RiskLevel.READ, RiskLevel.WRITE}:
            if metadata.risk_level is RiskLevel.WRITE and arguments.get("overwrite"):
                return PermissionDecision(False, True, "覆盖已有结果需要用户审批。")
            return PermissionDecision(True)
        if self.allow_approval:
            return PermissionDecision(True)
        return PermissionDecision(False, True, f"工具 {metadata.name} 的风险等级为 {metadata.risk_level}，需要审批。")

