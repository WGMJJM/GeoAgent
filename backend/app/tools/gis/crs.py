"""crs.* Tools。"""

from __future__ import annotations

from typing import Any

from app.core.models import DatasetKind
from app.execution.tools import ToolContext, ToolRegistry
from app.gis.crs.service import CRSService
from app.tools.gis import metadata

from .common import dataset_from_context, output_path, register_derived


def register(registry: ToolRegistry) -> None:
    registry.register(metadata("crs.inspect", "检查数据集坐标参考系和距离单位", tags=["gis", "crs"]), inspect_crs)
    registry.register(metadata("crs.reproject", "将矢量或栅格数据重投影到指定 CRS", write=True, tags=["gis", "crs"]), reproject)


def inspect_crs(arguments: dict[str, Any], context: ToolContext) -> dict:
    dataset = dataset_from_context(context, arguments.get("dataset_id"))
    result = CRSService(default_crs=context.services["settings"].default_crs).inspect(dataset)
    return {"output": {"dataset_id": dataset.id, **result}, "datasets": [dataset.id]}


def reproject(arguments: dict[str, Any], context: ToolContext) -> dict:
    dataset = dataset_from_context(context, arguments.get("dataset_id"))
    target_crs = arguments.get("target_crs") or CRSService(default_crs=context.services["settings"].default_crs).choose_projected_crs(dataset)
    suffix = ".tif" if dataset.kind is DatasetKind.RASTER else ".gpkg"
    target = output_path(context, arguments.get("output_path"), f"{dataset.name}_projected", suffix)
    if dataset.kind is DatasetKind.RASTER:
        result = context.services["rasters"].reproject(dataset, target_crs, target, run_id=context.run_id, tool_call_id=arguments.get("tool_call_id"))
    else:
        result = context.services["vectors"].reproject(dataset, target_crs, target, run_id=context.run_id, tool_call_id=arguments.get("tool_call_id"))
    registered = register_derived(context, result.path, name=target.stem, source_ids=[dataset.id], operation="crs.reproject", parameters={"target_crs": target_crs}, tool_call_id=arguments.get("tool_call_id"))
    return {"output": {"dataset": registered.model_dump(mode="json"), "target_crs": target_crs}, "datasets": [registered.id]}

