"""python.execute / shell.execute Tools。"""

from __future__ import annotations

import json
import mimetypes
from typing import Any

from app.core.models import Artifact, ArtifactKind, DatasetOutputPolicy, ErrorCategory, ToolMetadata
from app.execution.tools import ToolContext, ToolRegistry
from app.gis.errors import GISFailure


def register_runtime_tools(registry: ToolRegistry, *, python_environment: dict[str, Any] | None = None) -> None:
    registry.register(
        ToolMetadata(
            name="python.execute",
            description="按需编写并执行 Python，使用 GeoAgent 已安装库辅助自定义数据处理、分析和结果生成；受信任本地执行，不是安全沙箱 / Execute Python with installed libraries for custom data processing and analysis",
            required_scopes=["runtime.python.execute", "dataset.read", "dataset.write", "workspace.read", "workspace.write", "artifact.create"],
            required_envs=["runtime.python.local", "workspace"],
            input_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "minLength": 1,
                        "description": "完整 Python 代码，非 Markdown。全局 datasets 为本次 dataset_ids 经权限核验后的 ID→元数据字典，含 path、name、kind、format、crs、schema；用 datasets[id]['path'] 读取数据。output_dir、temp_dir 为 Path 对象，输出写 output_dir，临时文件写 temp_dir；通过 print 返回必要统计和说明。只使用实际已安装库，不安装包、不启动系统命令、不覆盖输入，也不读取 GeoAgent 内部配置或状态。失败不保证没有生成文件，修正前查看错误和 created_files。当前运行环境：" + json.dumps(python_environment or {}, ensure_ascii=False, separators=(",", ":")),
                    },
                    "dataset_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "uniqueItems": True,
                        "description": "本次代码要读取的已登记 Dataset ID；执行前按当前用户/子任务权限核验，不能填写任意路径或猜测 ID。不读取数据集时可省略。",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            risk_level="WRITE",
            supports_retry=False,
            dataset_output_policy=DatasetOutputPolicy.OPTIONAL,
            produces_artifact=True,
            tags=["runtime", "python", "gis"],
        ),
        python_execute,
        deferred=True,
    )
    registry.register(
        ToolMetadata(
            name="shell.execute",
            description="在隔离工作区中执行 GIS 命令（当前无已验证隔离环境） / Run GIS commands in an isolated workspace",
            required_scopes=["runtime.shell.execute", "workspace.read", "workspace.write"],
            required_envs=["isolated_shell", "workspace"],
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
            supports_retry=False,
            dataset_output_policy=DatasetOutputPolicy.OPTIONAL,
            tags=["runtime", "shell", "gis"],
        ),
        shell_execute,
        deferred=True,
    )


def python_execute(arguments: dict[str, Any], context: ToolContext) -> dict:
    if not context.services.get("allow_unsafe_python", False):
        raise GISFailure(
            "UNSAFE_PYTHON_DISABLED",
            "通用 Python 执行默认关闭；仅在受信任的本地运行时显式开启 GEOAGENT_ENABLE_UNSAFE_PYTHON 后使用。",
            category=ErrorCategory.PERMISSION,
        )
    inputs = {}
    for identifier in arguments.get("dataset_ids", []):
        dataset = context.services["registry"].get(identifier)
        if dataset is None:
            raise GISFailure("MISSING_DATASET", f"数据集不存在或无权读取：{identifier}", category=ErrorCategory.DATA)
        inputs[dataset.id] = dataset.model_dump(mode="json", include={"id", "name", "path", "kind", "format", "crs", "schema_"})
    result = context.services["python"].execute(
        arguments["code"],
        datasets=inputs,
        workspace=context.services["workspace"].for_run(context.run_id),
        cancel_event=context.cancel_event,
    )
    if result.returncode != 0:
        raise GISFailure("PYTHON_EXECUTION_FAILED", result.stderr or "Python 执行失败。", category=ErrorCategory.EXECUTION, details=result.model_dump(mode="json"), retryable=False)
    datasets, artifacts, warnings = _register_files(context, result.created_files, "python.execute", list(inputs), publish=True)
    return {"output": result.model_dump(mode="json"), "datasets": datasets, "artifacts": artifacts, "warnings": warnings}


def shell_execute(arguments: dict[str, Any], context: ToolContext) -> dict:
    result = context.services["shell"].execute(arguments.get("command", ""), cancel_event=context.cancel_event)
    if result.returncode != 0:
        raise GISFailure("SHELL_EXECUTION_FAILED", result.stderr or "Shell 执行失败。", category=ErrorCategory.EXECUTION, details={"returncode": result.returncode}, retryable=True)
    datasets, _, warnings = _register_files(context, result.created_files, "shell.execute", arguments.get("dataset_ids", []))
    return {"output": result.model_dump(mode="json"), "datasets": datasets, "warnings": warnings}


def _register_files(context: ToolContext, files: list[str], operation: str, source_ids: list[str], *, publish: bool = False) -> tuple[list[str], list[str], list[str]]:
    registry = context.services["registry"]
    datasets: list[str] = []
    artifacts: list[str] = []
    warnings: list[str] = []
    for filename in files:
        path = context.services["workspace"].assert_inside(filename)
        dataset_id = None
        try:
            dataset = registry.register_path(path, run_id=context.run_id, source_dataset_ids=source_ids, operation=operation, tool_call_id=context.call_id)
        except GISFailure as exc:
            # 非数据格式仍可作为文件产物发布，不另建扩展名白名单。
            if exc.error.code != "UNSUPPORTED_FORMAT" or not publish:
                warnings.append(f"未登记新文件 {filename}：{exc}")
        except Exception as exc:
            warnings.append(f"未登记新文件 {filename}：{exc}")
        else:
            dataset_id = dataset.id
            datasets.append(dataset.id)
        if publish:
            owner = context.services.get("user_id")
            if owner is None and not context.services.get("system_owned", False):
                raise PermissionError("生成文件产物必须绑定用户")
            artifact = Artifact(
                name=path.name,
                kind=ArtifactKind.DATASET if dataset_id else ArtifactKind.OTHER,
                path=str(path),
                media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                dataset_id=dataset_id,
                run_id=context.run_id,
                owner_user_id=owner,
                description="Python 辅助执行生成的文件",
                metadata={"tool_call_id": context.call_id, "source_dataset_ids": source_ids},
            )
            context.services["store"].save_artifact(artifact)
            artifacts.append(artifact.id)
    return datasets, artifacts, warnings
