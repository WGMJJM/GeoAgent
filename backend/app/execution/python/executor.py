"""在 GeoAgent workspace 中运行受信任的本地 Python 脚本。

WorkspaceManager 只提供路径边界，不构成通用 Python 的 OS 级安全沙箱。
是否允许 Agent 调用该能力由 GEOAGENT_ENABLE_UNSAFE_PYTHON 控制。
"""

from __future__ import annotations

import json
import platform
import sys
import uuid
from importlib.metadata import distributions
from threading import Event
from typing import Any

from app.execution.process import run_process
from app.execution.sandbox import WorkspaceManager

from .result import PythonExecutionResult


class PythonExecutor:
    def __init__(self, workspace: WorkspaceManager, *, timeout_seconds: int = 120) -> None:
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def environment() -> dict[str, Any]:
        """只描述当前解释器已安装的依赖，不导入库、不下载或安装包。"""

        return {
            "mode": "trusted_local",
            "python_version": platform.python_version(),
            "packages": dict(sorted(
                (item.metadata["Name"], item.version)
                for item in distributions() if item.metadata["Name"]
            )),
        }

    def execute(
        self,
        code: str,
        *,
        datasets: dict[str, dict[str, Any]] | None = None,
        workspace: WorkspaceManager | None = None,
        cancel_event: Event | None = None,
    ) -> PythonExecutionResult:
        if not code.strip():
            raise ValueError("Python code 不能为空。")
        workspace = workspace or self.workspace
        execution_id = f"python_{uuid.uuid4().hex[:10]}"
        output_dir = workspace.output_dir / execution_id
        output_dir.mkdir()
        script = workspace.temp_dir / f"{execution_id}.py"
        bindings = workspace.temp_dir / f"{execution_id}.json"
        script.write_text(code, encoding="utf-8")
        bindings.write_text(json.dumps({
            "datasets": datasets or {},
            "output_dir": str(output_dir),
            "temp_dir": str(workspace.temp_dir),
        }, ensure_ascii=False), encoding="utf-8")
        launcher = (
            "import json, runpy, sys; from pathlib import Path; "
            "context = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')); "
            "context['output_dir'] = Path(context['output_dir']); "
            "context['temp_dir'] = Path(context['temp_dir']); "
            "runpy.run_path(sys.argv[1], init_globals=context, run_name='__main__')"
        )
        try:
            # -I 排除宿主 PYTHONPATH/用户 site，不是文件或网络安全沙箱。
            completed = run_process(
                [sys.executable, "-I", "-B", "-X", "utf8", "-c", launcher, str(script), str(bindings)],
                cwd=output_dir,
                timeout_seconds=self.timeout_seconds,
                cancel_event=cancel_event,
                encoding="utf-8",
            )
        finally:
            script.unlink(missing_ok=True)
            bindings.unlink(missing_ok=True)
        created = [str(path) for path in sorted(output_dir.rglob("*")) if path.is_file()]
        return PythonExecutionResult(returncode=completed.returncode, stdout=completed.stdout[-20000:], stderr=completed.stderr[-20000:], created_files=created)
