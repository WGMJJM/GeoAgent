"""附件登记入口；文件实体仍由 WorkspaceManager 统一管理。"""

from pathlib import Path
from uuid import uuid4

from app.execution.sandbox import WorkspaceManager

SHAPEFILE_EXTENSIONS = frozenset({".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj"})
SHAPEFILE_REQUIRED_EXTENSIONS = frozenset({".shp", ".shx", ".dbf"})


class AttachmentService:
    def __init__(self, workspace: WorkspaceManager) -> None:
        self.workspace = workspace

    def accept(self, filename: str, content: bytes, *, user_id: str | None = None) -> Path:
        workspace = self.workspace.for_user(user_id)
        safe_name = Path(filename).name.strip()
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("文件名不能为空")
        if not content:
            raise ValueError("不能上传空文件")
        target = workspace.resolve(workspace.input_dir / safe_name)
        if target.exists():
            target = workspace.resolve(
                workspace.input_dir / f"{target.stem}-{uuid4().hex[:8]}{target.suffix}"
            )
        target.write_bytes(content)
        return target

    def accept_shapefile(
        self,
        files: list[tuple[str, bytes]],
        *,
        user_id: str | None = None,
    ) -> Path:
        """保存同名 Shapefile 文件组并返回主 .shp 路径。"""

        if not files:
            raise ValueError("Shapefile 文件组不能为空")
        normalized = [(Path(filename).name.strip(), content) for filename, content in files]
        if any(not filename or filename in {".", ".."} for filename, _ in normalized):
            raise ValueError("文件名不能为空")
        if any(not content for _, content in normalized):
            raise ValueError("不能上传空文件")
        stems = {Path(filename).stem.casefold() for filename, _ in normalized}
        if len(stems) != 1:
            raise ValueError("Shapefile 文件组必须使用相同的文件名")
        extensions = [Path(filename).suffix.casefold() for filename, _ in normalized]
        unsupported = sorted(set(extensions) - SHAPEFILE_EXTENSIONS)
        if unsupported:
            raise ValueError(f"不支持的 Shapefile 伴随文件：{', '.join(unsupported)}")
        if len(extensions) != len(set(extensions)):
            raise ValueError("Shapefile 文件组包含重复扩展名")
        missing = sorted(SHAPEFILE_REQUIRED_EXTENSIONS - set(extensions))
        if missing:
            raise ValueError(f"Shapefile 缺少必要文件：{', '.join(missing)}")

        workspace = self.workspace.for_user(user_id)
        stem = Path(normalized[0][0]).stem
        directory = workspace.resolve(workspace.input_dir / f"{stem}-{uuid4().hex[:8]}")
        directory.mkdir()
        primary: Path | None = None
        for filename, content in normalized:
            target = workspace.resolve(directory / filename)
            target.write_bytes(content)
            if target.suffix.casefold() == ".shp":
                primary = target
        assert primary is not None
        return primary
