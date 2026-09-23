"""GIS Tool 的参数与输出辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.models import Dataset, ErrorCategory, new_id
from app.execution.tools import ToolContext
from app.gis.errors import GISFailure


def dataset_from_context(context: ToolContext, identifier: str | None) -> Dataset:
    registry = context.services["registry"]
    if not identifier:
        raise GISFailure("MISSING_DATASET", "Tool 缺少 dataset_id。", category=ErrorCategory.DATA)
    dataset = registry.resolve(identifier, user_id=context.services.get("user_id"))
    if dataset is None:
        raise GISFailure("MISSING_DATASET", f"未注册的数据集：{identifier}", category=ErrorCategory.DATA)
    return dataset


def output_path(context: ToolContext, requested: str | None, stem: str, suffix: str = ".gpkg", *, intermediate: bool = True) -> Path:
    workspace = context.services["workspace"]
    filename = requested or f"{stem}_{new_id('out').split('_', 1)[1]}{suffix}"
    directory = workspace.intermediate_dir if intermediate else workspace.output_dir
    target = workspace.resolve(directory / filename)
    if target.exists():
        raise FileExistsError(f"输出文件已存在，为避免覆盖请换一个 output_path：{target.name}")
    return target


def register_derived(
    context: ToolContext,
    path: str | Path,
    *,
    name: str | None,
    source_ids: list[str],
    operation: str,
    parameters: dict[str, Any] | None = None,
    tool_call_id: str | None = None,
) -> Dataset:
    registry = context.services["registry"]
    return registry.register_path(path, name=name, run_id=context.run_id, source_dataset_ids=source_ids, operation=operation, parameters=parameters, tool_call_id=tool_call_id)
