"""raster.* Tools。"""

from __future__ import annotations

from typing import Any

from app.execution.tools import ToolContext, ToolRegistry
from app.tools.gis import metadata

from .common import dataset_from_context, output_path, register_derived


def register(registry: ToolRegistry) -> None:
    registry.register(metadata("raster.inspect", "检查栅格尺寸、分辨率和 NoData / Inspect raster dimensions, resolution and NoData", tags=["gis", "raster"], required_envs=["gis.raster"]), inspect, deferred=True)
    registry.register(metadata("raster.clip", "使用矢量边界裁剪栅格 / Clip a raster with a vector boundary", write=True, tags=["gis", "raster"], required_envs=["gis.raster", "workspace"]), clip, deferred=True)
    registry.register(metadata("raster.reproject", "重投影栅格数据 / Reproject raster data", write=True, tags=["gis", "raster"], required_envs=["gis.raster", "workspace"]), reproject, deferred=True)
    registry.register(metadata("raster.slope", "从 DEM 计算坡度 / Calculate slope from a DEM", write=True, tags=["gis", "raster", "terrain"], required_envs=["gis.raster", "workspace"]), slope, deferred=True)


def inspect(arguments: dict[str, Any], context: ToolContext) -> dict:
    dataset = dataset_from_context(context, arguments.get("dataset_id"))
    result = context.services["rasters"].inspect(dataset)
    return {"output": result, "datasets": [dataset.id]}


def clip(arguments: dict[str, Any], context: ToolContext) -> dict:
    dataset = dataset_from_context(context, arguments.get("dataset_id"))
    mask_dataset = dataset_from_context(context, arguments.get("mask_dataset_id"))
    target = output_path(context, arguments.get("output_path"), f"{dataset.name}_clipped", ".tif")
    result = context.services["rasters"].clip(dataset, mask_dataset, target, run_id=context.run_id, tool_call_id=arguments.get("tool_call_id"))
    registered = register_derived(context, result.path, name=target.stem, source_ids=[dataset.id, mask_dataset.id], operation="raster.clip", parameters={"mask_dataset_id": mask_dataset.id}, tool_call_id=arguments.get("tool_call_id"))
    return {"output": registered.model_dump(mode="json"), "datasets": [registered.id]}


def reproject(arguments: dict[str, Any], context: ToolContext) -> dict:
    dataset = dataset_from_context(context, arguments.get("dataset_id"))
    target_crs = arguments["target_crs"]
    target = output_path(context, arguments.get("output_path"), f"{dataset.name}_projected", ".tif")
    result = context.services["rasters"].reproject(dataset, target_crs, target, run_id=context.run_id, tool_call_id=arguments.get("tool_call_id"))
    registered = register_derived(context, result.path, name=target.stem, source_ids=[dataset.id], operation="raster.reproject", parameters={"target_crs": target_crs}, tool_call_id=arguments.get("tool_call_id"))
    return {"output": registered.model_dump(mode="json"), "datasets": [registered.id]}


def slope(arguments: dict[str, Any], context: ToolContext) -> dict:
    dataset = dataset_from_context(context, arguments.get("dataset_id"))
    target = output_path(context, arguments.get("output_path"), f"{dataset.name}_slope", ".tif")
    result = context.services["rasters"].slope(dataset, target, run_id=context.run_id, tool_call_id=arguments.get("tool_call_id"))
    registered = register_derived(context, result.path, name=target.stem, source_ids=[dataset.id], operation="raster.slope", tool_call_id=arguments.get("tool_call_id"))
    return {"output": registered.model_dump(mode="json"), "datasets": [registered.id]}

