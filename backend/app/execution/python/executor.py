"""在 GeoAgent workspace 中运行受信任的本地 Python 脚本。

WorkspaceManager 只提供路径边界，不构成通用 Python 的 OS 级安全沙箱。
是否允许 Agent 调用该能力由 GEOAGENT_ENABLE_UNSAFE_PYTHON 控制。
"""

from __future__ import annotations

import re
import sys
import uuid
from threading import Event

from app.execution.process import run_process
from app.execution.sandbox import WorkspaceManager

from .result import PythonExecutionResult


class PythonExecutor:
    def __init__(self, workspace: WorkspaceManager, *, timeout_seconds: int = 120) -> None:
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds

    def execute(self, code: str, *, cancel_event: Event | None = None) -> PythonExecutionResult:
        if not code.strip():
            raise ValueError("Python code 不能为空。")
        if re.search(r"(?:\.\.[\\/]|[A-Za-z]:[\\/]|\\\\)", code) or re.search(r"(?<![\w])/(?:etc|tmp|home|root|var|usr|bin|opt)(?:[\\/\s'\")]|$)", code):
            raise PermissionError("Python 代码包含明显的 workspace 外部绝对路径或路径穿越。")
        before = self.workspace.snapshot()
        script = self.workspace.temp_dir / f"run_{uuid.uuid4().hex[:10]}.py"
        script.write_text(code, encoding="utf-8")
        try:
            completed = run_process([sys.executable, str(script)], cwd=self.workspace.root, timeout_seconds=self.timeout_seconds, cancel_event=cancel_event)
        finally:
            script.unlink(missing_ok=True)
        created = [str(path) for path in self.workspace.discover_new_files(before) if path != script]
        return PythonExecutionResult(returncode=completed.returncode, stdout=completed.stdout[-20000:], stderr=completed.stderr[-20000:], created_files=created)
