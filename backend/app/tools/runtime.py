"""python.execute / shell.execute Tools。"""

from __future__ import annotations

from typing import Any

from app.core.models import DatasetOutputPolicy, ErrorCategory, ToolMetadata
from app.execution.tools import ToolContext, ToolRegistry
from app.gis.errors import GISFailure


def register_runtime_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolMetadata(
            name="python.execute",
            description="在受信任的本地运行时执行 Python GIS 代码（默认关闭）",
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "要执行的 Python 代码"},
                    "dataset_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            risk_level="WRITE",
            supports_retry=False,
            dataset_output_policy=DatasetOutputPolicy.OPTIONAL,
            tags=["runtime", "python", "gis"],
        ),
        python_execute,
    )
    registry.register(
        ToolMetadata(
            name="shell.execute",
            description="在 workspace 中执行白名单 GIS CLI",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "单条白名单 GIS CLI 命令"},
                    "dataset_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            risk_level="WRITE",
            supports_retry=True,
            dataset_output_policy=DatasetOutputPolicy.OPTIONAL,
            tags=["runtime", "shell", "gis"],
        ),
        shell_execute,
    )


def python_execute(arguments: dict[str, Any], context: ToolContext) -> dict:
    if not context.services.get("allow_unsafe_python", False):
        raise GISFailure(
            "UNSAFE_PYTHON_DISABLED",
            "通用 Python 执行默认关闭；仅在受信任的本地运行时显式开启 GEOAGENT_ENABLE_UNSAFE_PYTHON 后使用。",
            category=ErrorCategory.PERMISSION,
        )
    result = context.services["python"].execute(arguments.get("code", ""), cancel_event=context.cancel_event)
    if result.returncode != 0:
        raise GISFailure("PYTHON_EXECUTION_FAILED", result.stderr or "Python 执行失败。", category=ErrorCategory.EXECUTION, details={"returncode": result.returncode}, retryable=False)
    datasets, warnings = _register_files(context, result.created_files, "python.execute", arguments.get("dataset_ids", []))
    return {"output": result.model_dump(mode="json"), "datasets": datasets, "warnings": warnings}


def shell_execute(arguments: dict[str, Any], context: ToolContext) -> dict:
    result = context.services["shell"].execute(arguments.get("command", ""), cancel_event=context.cancel_event)
    if result.returncode != 0:
        raise GISFailure("SHELL_EXECUTION_FAILED", result.stderr or "Shell 执行失败。", category=ErrorCategory.EXECUTION, details={"returncode": result.returncode}, retryable=True)
    datasets, warnings = _register_files(context, result.created_files, "shell.execute", arguments.get("dataset_ids", []))
    return {"output": result.model_dump(mode="json"), "datasets": datasets, "warnings": warnings}


def _register_files(context: ToolContext, files: list[str], operation: str, source_ids: list[str]) -> tuple[list[str], list[str]]:
    registry = context.services["registry"]
    datasets: list[str] = []
    warnings: list[str] = []
    for filename in files:
        try:
            dataset = registry.register_path(filename, run_id=context.run_id, source_dataset_ids=source_ids, operation=operation)
        except Exception as exc:
            warnings.append(f"未登记新文件 {filename}：{exc}")
        else:
            datasets.append(dataset.id)
    return datasets, warnings
