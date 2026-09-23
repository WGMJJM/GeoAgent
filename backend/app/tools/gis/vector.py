"""vector.* Tools。"""

from __future__ import annotations

from typing import Any

from app.execution.tools import ToolContext, ToolRegistry
from app.tools.gis import metadata

from .common import dataset_from_context, output_path, register_derived


def register(registry: ToolRegistry) -> None:
    registry.register(metadata("vector.validate", "检查矢量 geometry 合法性和要素数量", tags=["gis", "vector"]), validate)
    registry.register(metadata("vector.repair", "修复无效矢量 geometry", write=True, tags=["gis", "vector"]), repair)
    registry.register(metadata("vector.buffer", "按线性单位生成矢量缓冲区", write=True, tags=["gis", "vector"]), buffer)
    registry.register(metadata("vector.clip", "使用矢量边界裁剪数据", write=True, tags=["gis", "vector"]), clip)
    registry.register(metadata("vector.intersection", "计算两个矢量数据集的相交区域", write=True, tags=["gis", "vector"]), intersection)
    registry.register(metadata("vector.dissolve", "按字段或整体融合矢量要素", write=True, tags=["gis", "vector"]), dissolve)
    registry.register(metadata("vector.spatial_join", "按空间关系连接两个矢量数据集", write=True, tags=["gis", "vector"]), spatial_join)


def validate(arguments: dict[str, Any], context: ToolContext) -> dict:
    dataset = dataset_from_context(context, arguments.get("dataset_id"))
    result = context.services["vectors"].validate(dataset)
    return {"output": result, "datasets": [dataset.id], "warnings": ["检测到无效 geometry，请调用 vector.repair" ] if result.get("invalid_geometry_count", 0) else []}


def repair(arguments: dict[str, Any], context: ToolContext) -> dict:
    dataset = dataset_from_context(context, arguments.get("dataset_id"))
    target = output_path(context, arguments.get("output_path"), f"{dataset.name}_repaired")
    result = context.services["vectors"].repair(dataset, target, run_id=context.run_id, tool_call_id=arguments.get("tool_call_id"))
    registered = register_derived(context, result.path, name=target.stem, source_ids=[dataset.id], operation="vector.repair", tool_call_id=arguments.get("tool_call_id"))
    return {"output": registered.model_dump(mode="json"), "datasets": [registered.id]}


def buffer(arguments: dict[str, Any], context: ToolContext) -> dict:
    dataset = dataset_from_context(context, arguments.get("dataset_id"))
    distance = float(arguments.get("distance", 0))
    target = output_path(context, arguments.get("output_path"), f"{dataset.name}_buffer")
    result = context.services["vectors"].buffer(dataset, distance, target, run_id=context.run_id, tool_call_id=arguments.get("tool_call_id"))
    registered = register_derived(context, result.path, name=target.stem, source_ids=[dataset.id], operation="vector.buffer", parameters={"distance": distance}, tool_call_id=arguments.get("tool_call_id"))
    return {"output": {"dataset": registered.model_dump(mode="json"), "distance": distance}, "datasets": [registered.id]}


def clip(arguments: dict[str, Any], context: ToolContext) -> dict:
    dataset = dataset_from_context(context, arguments.get("dataset_id"))
    mask = dataset_from_context(context, arguments.get("mask_dataset_id"))
    target = output_path(context, arguments.get("output_path"), f"{dataset.name}_clipped")
    result = context.services["vectors"].clip(dataset, mask, target, run_id=context.run_id, tool_call_id=arguments.get("tool_call_id"))
    registered = register_derived(context, result.path, name=target.stem, source_ids=[dataset.id, mask.id], operation="vector.clip", parameters={"mask_dataset_id": mask.id}, tool_call_id=arguments.get("tool_call_id"))
    return {"output": registered.model_dump(mode="json"), "datasets": [registered.id]}


def intersection(arguments: dict[str, Any], context: ToolContext) -> dict:
    left = dataset_from_context(context, arguments.get("left_dataset_id"))
    right = dataset_from_context(context, arguments.get("right_dataset_id"))
    target = output_path(context, arguments.get("output_path"), f"{left.name}_intersection")
    result = context.services["vectors"].intersection(left, right, target, run_id=context.run_id, tool_call_id=arguments.get("tool_call_id"))
    registered = register_derived(context, result.path, name=target.stem, source_ids=[left.id, right.id], operation="vector.intersection", parameters={"right_dataset_id": right.id}, tool_call_id=arguments.get("tool_call_id"))
    return {"output": registered.model_dump(mode="json"), "datasets": [registered.id]}


def dissolve(arguments: dict[str, Any], context: ToolContext) -> dict:
    dataset = dataset_from_context(context, arguments.get("dataset_id"))
    by = arguments.get("by")
    target = output_path(context, arguments.get("output_path"), f"{dataset.name}_dissolved")
    result = context.services["vectors"].dissolve(dataset, by, target, run_id=context.run_id, tool_call_id=arguments.get("tool_call_id"))
    registered = register_derived(context, result.path, name=target.stem, source_ids=[dataset.id], operation="vector.dissolve", parameters={"by": by}, tool_call_id=arguments.get("tool_call_id"))
    return {"output": registered.model_dump(mode="json"), "datasets": [registered.id]}


def spatial_join(arguments: dict[str, Any], context: ToolContext) -> dict:
    left = dataset_from_context(context, arguments.get("left_dataset_id"))
    right = dataset_from_context(context, arguments.get("right_dataset_id"))
    target = output_path(context, arguments.get("output_path"), f"{left.name}_joined")
    result = context.services["vectors"].spatial_join(left, right, target, predicate=arguments.get("predicate", "intersects"), run_id=context.run_id, tool_call_id=arguments.get("tool_call_id"))
    registered = register_derived(context, result.path, name=target.stem, source_ids=[left.id, right.id], operation="vector.spatial_join", parameters={"predicate": arguments.get("predicate", "intersects")}, tool_call_id=arguments.get("tool_call_id"))
    return {"output": registered.model_dump(mode="json"), "datasets": [registered.id]}
