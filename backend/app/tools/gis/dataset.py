"""dataset.* Tools。"""

from __future__ import annotations

from typing import Any

from app.core.models import DatasetKind
from app.execution.tools import ToolContext, ToolRegistry
from app.tools.gis import metadata

from .common import dataset_from_context


def register(registry: ToolRegistry) -> None:
    registry.register(metadata("dataset.list", "列出已注册的空间数据集", tags=["gis", "dataset"]), list_datasets)
    registry.register(metadata("dataset.inspect", "检查数据集的 CRS、范围、字段和统计摘要", tags=["gis", "dataset"]), inspect_dataset)
    registry.register(metadata("dataset.register", "把 workspace 中的数据文件登记到 Dataset Registry", write=True, tags=["gis", "dataset"]), register_dataset)


def list_datasets(arguments: dict[str, Any], context: ToolContext) -> dict:
    kind = arguments.get("kind")
    registry = context.services["registry"]
    parsed = DatasetKind(kind) if kind else None
    datasets = registry.list(parsed)
    return {"output": {"datasets": [item.model_dump(mode="json") for item in datasets]}, "datasets": [item.id for item in datasets]}


def inspect_dataset(arguments: dict[str, Any], context: ToolContext) -> dict:
    registry = context.services["registry"]
    identifier = arguments.get("dataset_id") or arguments.get("path")
    dataset = registry.resolve(identifier) if identifier else None
    if dataset is None and arguments.get("path"):
        path = context.services["workspace"].resolve(arguments["path"], allow_missing=False)
        dataset = registry.register_path(path, name=arguments.get("name"), run_id=context.run_id)
    dataset = dataset or dataset_from_context(context, identifier)
    return {"output": dataset.model_dump(mode="json"), "datasets": [dataset.id]}


def register_dataset(arguments: dict[str, Any], context: ToolContext) -> dict:
    path = context.services["workspace"].resolve(arguments.get("path", ""), allow_missing=False)
    dataset = context.services["registry"].register_path(path, name=arguments.get("name"), run_id=context.run_id)
    return {"output": dataset.model_dump(mode="json"), "datasets": [dataset.id]}
