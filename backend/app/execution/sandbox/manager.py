"""文件工作区边界。

GeoAgent 允许 Agent 读写 workspace 下的文件，但所有路径都会解析后检查是否
仍在 workspace 内。输入、派生数据、发布产物和临时脚本分开存放，便于恢复和清理。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


class WorkspaceManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.input_dir = self.root / "input"
        self.intermediate_dir = self.root / "intermediate"
        self.output_dir = self.root / "output"
        self.temp_dir = self.root / "temp"
        for directory in (self.input_dir, self.intermediate_dir, self.output_dir, self.temp_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def for_user(self, user_id: str | None) -> WorkspaceManager:
        """返回用户专属工作区；None 仅用于内部离线/系统数据。"""

        return WorkspaceManager(self.root / "users" / user_id) if user_id else self

    def resolve(self, path: str | Path, *, allow_missing: bool = True) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        self.assert_inside(candidate)
        if not allow_missing and not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate

    def assert_inside(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser().resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"路径必须位于 GeoAgent workspace 内：{candidate}") from exc
        return candidate

    def output_path(self, filename: str, *, intermediate: bool = True) -> Path:
        name = Path(filename).name
        directory = self.intermediate_dir if intermediate else self.output_dir
        target = self.assert_inside(directory / name)
        if target.exists():
            raise FileExistsError(f"输出文件已存在，为避免覆盖请换一个 output_path：{target.name}")
        return target

    def snapshot(self) -> set[Path]:
        return {path for path in self.root.rglob("*") if path.is_file()}

    def discover_new_files(self, before: Iterable[Path]) -> list[Path]:
        known = {Path(item).resolve() for item in before}
        return sorted((path for path in self.snapshot() if path not in known), key=str)
