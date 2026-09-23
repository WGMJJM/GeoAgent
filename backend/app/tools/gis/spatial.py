"""analysis.* 的矢量邻近与叠加 Tool。"""

from __future__ import annotations

from typing import Any

from app.execution.tools import ToolContext, ToolRegistry
from app.gis.analysis.proximity import distance_summary
from app.tools.gis import metadata

from .common import dataset_from_context


def register(registry: ToolRegistry) -> None:
    registry.register(metadata("analysis.distance", "计算要素到目标的距离分布和阈值覆盖数", tags=["gis", "analysis"]), distance)


def distance(arguments: dict[str, Any], context: ToolContext) -> dict:
    source = dataset_from_context(context, arguments.get("source_dataset_id"))
    target = dataset_from_context(context, arguments["target_dataset_id"]) if arguments.get("target_dataset_id") else None
    result = distance_summary(source, target, threshold=float(arguments["threshold"]) if arguments.get("threshold") is not None else None)
    return {"output": result, "datasets": [item.id for item in (source, target) if item]}

