"""GeoAgent 的最小权限策略。

读操作和生成新结果默认允许；覆盖/删除/外部访问等需要显式 approval，而第一版
不会偷偷降级执行。这样工具失败可以被 Agent 看见并转化为 BLOCKED/ASK_USER。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.core.models import RiskLevel, Run, ToolMetadata


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    needs_approval: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ToolDiscoveryContext:
    """由服务端基于认证身份和实际服务构造，不接受模型提供。"""

    granted_scopes: frozenset[str] = frozenset()
    available_envs: frozenset[str] = frozenset()
    allowed_tool_names: frozenset[str] | None = None


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

    def is_discoverable(self, metadata: ToolMetadata, context: ToolDiscoveryContext) -> bool:
        """只判断是否可向模型展示；真实执行仍由 authorize 和资源服务再次校验。"""

        return (
            (context.allowed_tool_names is None or metadata.name in context.allowed_tool_names)
            and set(metadata.required_scopes).issubset(context.granted_scopes)
            and set(metadata.required_envs).issubset(context.available_envs)
        )

    @staticmethod
    def restrict_to_run(context: ToolDiscoveryContext, run: Run | None, parent: Run | None = None) -> ToolDiscoveryContext:
        """只读取服务端持久化 Run 的限制，每次发现和执行均重新取交集。"""

        names = context.allowed_tool_names
        scopes, environments = context.granted_scopes, context.available_envs
        for current in (parent, run):
            if current is None:
                continue
            raw = current.metadata.get("allowed_tool_names")
            if isinstance(raw, list):
                allowed = frozenset(raw)
                names = allowed if names is None else names & allowed
        if run is not None and run.parent_run_id:
            scopes &= frozenset(run.metadata.get("parent_granted_scopes", []))
            environments &= frozenset(run.metadata.get("parent_available_envs", []))
        return ToolDiscoveryContext(scopes, environments, names)

    @staticmethod
    def discovery_context(*, authenticated_user: bool, services: Mapping[str, Any]) -> ToolDiscoveryContext:
        """根据可信认证结果和实际注入服务构造目录上下文。"""

        scopes = (
            frozenset({"dataset.read", "dataset.write", "workspace.read", "workspace.write", "artifact.create"})
            if authenticated_user
            else frozenset()
        )
        environments: set[str] = set()
        if services.get("registry") is not None and services.get("inspector") is not None:
            environments.add("gis.dataset")
        for service, environment in (
            ("vectors", "gis.vector"),
            ("rasters", "gis.raster"),
            ("crs", "gis.crs"),
            ("renderer", "gis.visualization"),
            ("workspace", "workspace"),
        ):
            if services.get(service) is not None:
                environments.add(environment)
        # 普通 Python/Shell 执行器不等于隔离环境；只有宿主明确注入经过验证的
        # 隔离运行时标记时才允许目录发现这两类工具。
        if services.get("isolated_python") is True:
            environments.add("isolated_python")
        if services.get("isolated_shell") is True:
            environments.add("isolated_shell")
        return ToolDiscoveryContext(scopes, frozenset(environments))

