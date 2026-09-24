"""analysis.zonal_statistics Tool。"""

from __future__ import annotations

from typing import Any

from app.execution.tools import ToolContext, ToolRegistry
from app.gis.analysis.statistics import zonal_statistics
from app.tools.gis import metadata

from .common import dataset_from_context


def register(registry: ToolRegistry) -> None:
    registry.register(metadata("analysis.zonal_statistics", "计算分区栅格均值、最小值和最大值 / Calculate zonal raster statistics", tags=["gis", "analysis", "raster"], required_envs=["gis.vector", "gis.raster"]), zonal, deferred=True)


def zonal(arguments: dict[str, Any], context: ToolContext) -> dict:
    zones = dataset_from_context(context, arguments.get("zones_dataset_id"))
    raster = dataset_from_context(context, arguments.get("raster_dataset_id"))
    result = zonal_statistics(zones, raster)
    return {"output": result, "datasets": [zones.id, raster.id]}

