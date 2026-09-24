"""dataset.* Tools。"""

from __future__ import annotations

from typing import Any

from app.core.models import DatasetKind
from app.execution.tools import ToolContext, ToolRegistry
from app.tools.gis import metadata

from .common import dataset_from_context


def register(registry: ToolRegistry) -> None:
    registry.register(metadata("dataset.list", "列出已注册的空间数据集 / List registered spatial datasets", tags=["gis", "dataset"], required_scopes=["dataset.read"], required_envs=["gis.dataset"]), list_datasets)
    registry.register(metadata("dataset.inspect", "检查数据集 CRS、范围、字段和摘要 / Inspect dataset CRS, extent, fields and summary", tags=["gis", "dataset"], required_scopes=["dataset.read", "dataset.write", "workspace.read"], required_envs=["gis.dataset", "workspace"]), inspect_dataset)
    registry.register(metadata("dataset.register", "登记工作区中的数据文件 / Register a data file from the workspace", write=True, tags=["gis", "dataset"], required_scopes=["workspace.read", "dataset.write"], required_envs=["gis.dataset", "workspace"]), register_dataset, deferred=True)


def list_datasets(arguments: dict[str, Any], context: ToolContext) -> dict:
    kind = arguments.get("kind")
    registry = context.services["registry"]
    parsed = DatasetKind(kind) if kind else None
    datasets = registry.list(parsed)
    return {"output": {"datasets": [item.model_dump(mode="json") for item in datasets]}, "datasets": [item.id for item in datasets]}


def inspect_dataset(arguments: dict[str, Any], context: ToolContext) -> dict:
    dataset = dataset_from_context(context, arguments.get("dataset_id"))
    return {"output": dataset.model_dump(mode="json"), "datasets": [dataset.id]}


def register_dataset(arguments: dict[str, Any], context: ToolContext) -> dict:
    path = context.services["workspace"].resolve(arguments.get("path", ""), allow_missing=False)
    dataset = context.services["registry"].register_path(path, name=arguments.get("name"), run_id=context.run_id)
    return {"output": dataset.model_dump(mode="json"), "datasets": [dataset.id]}
