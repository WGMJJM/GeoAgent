"""限制在 workspace 内且只允许 GIS CLI 白名单的 Shell 执行器。"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from threading import Event

from app.execution.process import run_process
from app.execution.sandbox import WorkspaceManager

from .result import ShellExecutionResult


class ShellExecutor:
    ALLOWED_COMMANDS = {"gdalinfo", "gdalwarp", "gdal_translate", "ogr2ogr", "ogrinfo", "gdaldem"}

    def __init__(self, workspace: WorkspaceManager, *, timeout_seconds: int = 120) -> None:
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds

    def execute(self, command: str, *, cancel_event: Event | None = None) -> ShellExecutionResult:
        tokens = _tokens(command)
        if not tokens:
            raise ValueError("Shell command 不能为空。")
        if any(operator in command for operator in ("&&", "||", ";", "|", ">", "<", "$(", "`")):
            raise PermissionError("Shell 不允许命令串联、重定向或命令替换。")
        executable = Path(tokens[0]).name.casefold()
        if executable not in self.ALLOWED_COMMANDS:
            raise PermissionError(f"Shell 仅允许 GIS CLI：{', '.join(sorted(self.ALLOWED_COMMANDS))}")
        for token in tokens[1:]:
            cleaned = token.strip('"\'')
            if re.match(r"^[A-Za-z]:[\\/]", cleaned) or cleaned.startswith("\\\\"):
                self.workspace.assert_inside(cleaned)
            elif not cleaned.startswith("-") and ("/" in cleaned or "\\" in cleaned):
                self.workspace.resolve(cleaned)
        before = self.workspace.snapshot()
        completed = run_process(tokens, cwd=self.workspace.root, timeout_seconds=self.timeout_seconds, cancel_event=cancel_event)
        return ShellExecutionResult(returncode=completed.returncode, stdout=completed.stdout[-20000:], stderr=completed.stderr[-20000:], created_files=[str(path) for path in self.workspace.discover_new_files(before)])


def _tokens(command: str) -> list[str]:
    try:
        tokens = shlex.split(command, posix=False)
    except ValueError as exc:
        raise ValueError(f"Shell command 引号不匹配：{exc}") from exc
    return [_unquote(token) for token in tokens]


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] in "\"'" and token[-1] == token[0]:
        return token[1:-1]
    if token[:1] in {"\"", "'"} or token[-1:] in {"\"", "'"}:
        raise ValueError("Shell command 引号不匹配。")
    return token
