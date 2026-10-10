"""下载原文件；Shapefile 将同名伴随文件打包，不改变原数据。"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from app.entry.attachment_service import SHAPEFILE_EXTENSIONS, SHAPEFILE_REQUIRED_EXTENSIONS
from app.execution.sandbox import WorkspaceManager


def package_shapefile(path: Path, workspace: WorkspaceManager) -> Path:
    files = [
        workspace.resolve(item, allow_missing=False)
        for item in path.parent.iterdir()
        if item.stem.casefold() == path.stem.casefold()
        and item.suffix.casefold() in SHAPEFILE_EXTENSIONS
        and item.is_file()
    ]
    missing = SHAPEFILE_REQUIRED_EXTENSIONS - {item.suffix.casefold() for item in files}
    if missing:
        raise ValueError(f"Shapefile 缺少必要文件：{', '.join(sorted(missing))}")
    with NamedTemporaryFile(dir=workspace.temp_dir, suffix=".zip", delete=False) as temporary:
        archive = Path(temporary.name)
    try:
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as output:
            for item in files:
                output.write(item, arcname=item.name)
    except Exception:
        archive.unlink()
        raise
    return archive
