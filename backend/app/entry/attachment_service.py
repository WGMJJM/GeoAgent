"""附件登记入口；文件实体仍由 WorkspaceManager 统一管理。"""

from pathlib import Path
from uuid import uuid4

from app.execution.sandbox import WorkspaceManager


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
