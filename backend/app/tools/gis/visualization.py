"""map.render Tool。"""

from __future__ import annotations

from typing import Any

from app.core.models import Artifact, ArtifactKind
from app.execution.tools import ToolContext, ToolRegistry
from app.tools.gis import metadata

from .common import dataset_from_context, output_path


def register(registry: ToolRegistry) -> None:
    registry.register(metadata("map.render", "将矢量数据渲染为可下载地图 / Render vector data as a downloadable map", write=True, artifact=True, tags=["gis", "visualization"], required_scopes=["dataset.read", "workspace.write", "artifact.create"], required_envs=["gis.vector", "gis.visualization", "workspace"]), render, deferred=True)


def render(arguments: dict[str, Any], context: ToolContext) -> dict:
    dataset = dataset_from_context(context, arguments.get("dataset_id"))
    target = output_path(context, arguments.get("output_path"), f"{dataset.name}_map", ".html", intermediate=False)
    path = context.services["renderer"].render(dataset, target, title=arguments.get("title"))
    owner_user_id = context.services.get("user_id")
    if owner_user_id is None and not context.services.get("system_owned", False):
        raise PermissionError("生成地图产物必须绑定用户")
    artifact = Artifact(name=path.name, kind=ArtifactKind.MAP, path=str(path), media_type="text/html", dataset_id=dataset.id, run_id=context.run_id, owner_user_id=owner_user_id, description="GeoAgent 生成的地图结果")
    context.services["store"].save_artifact(artifact)
    return {"output": artifact.model_dump(mode="json"), "artifacts": [artifact.id], "datasets": [dataset.id]}
