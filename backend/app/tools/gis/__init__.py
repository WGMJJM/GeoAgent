"""GeoAgent 第一批 GIS Tools。"""

from __future__ import annotations

from typing import Any

from app.core.models import DatasetOutputPolicy, RiskLevel, ToolMetadata
from app.execution.tools import ToolRegistry


def register_gis_tools(registry: ToolRegistry) -> None:
    """把 GIS 领域操作注册到通用 Tool Registry。"""

    for module in (dataset, crs, vector, raster, spatial, analysis, visualization):
        module.register(registry)


def metadata(
    name: str,
    description: str,
    *,
    write: bool = False,
    artifact: bool = False,
    tags: list[str] | None = None,
    input_schema: dict[str, Any] | None = None,
    dataset_output_policy: DatasetOutputPolicy | None = None,
    required_scopes: list[str] | None = None,
    required_envs: list[str] | None = None,
) -> ToolMetadata:
    output_policy = dataset_output_policy
    if output_policy is None:
        output_policy = DatasetOutputPolicy.NONE if artifact or name.endswith(".validate") else DatasetOutputPolicy.REQUIRED if write else DatasetOutputPolicy.NONE
    return ToolMetadata(
        name=name,
        description=description,
        input_schema=input_schema or _SCHEMAS.get(name, _schema({})),
        required_scopes=required_scopes if required_scopes is not None else (["dataset.read", "dataset.write", "workspace.write"] if write else ["dataset.read"]),
        required_envs=required_envs or [],
        risk_level=RiskLevel.WRITE if write else RiskLevel.READ,
        supports_retry=name.startswith(("dataset.inspect", "raster.inspect")),
        dataset_output_policy=output_policy,
        produces_artifact=artifact,
        tags=tags or ["gis"],
    )


def _schema(properties: dict[str, Any], *, required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


_DATASET_ID = {"type": "string", "description": "已登记 Dataset 的 ID、名称或文件名"}
_OUTPUT_PATH = {"type": "string", "description": "workspace 内的输出路径，可选"}
_SCHEMAS: dict[str, dict[str, Any]] = {
    "dataset.list": _schema({"kind": {"type": "string", "enum": ["VECTOR", "RASTER", "TABLE"]}}),
    "dataset.inspect": _schema(
        {
            "dataset_id": _DATASET_ID,
            "path": {"type": "string", "description": "workspace 内尚未登记的数据文件路径"},
            "name": {"type": "string", "description": "新登记数据集的显示名称，可选"},
        }
    ),
    "dataset.register": _schema(
        {"path": {"type": "string", "description": "workspace 内的数据文件路径"}, "name": {"type": "string"}},
        required=("path",),
    ),
    "crs.inspect": _schema({"dataset_id": _DATASET_ID}, required=("dataset_id",)),
    "crs.reproject": _schema(
        {"dataset_id": _DATASET_ID, "target_crs": {"type": "string"}, "output_path": _OUTPUT_PATH},
        required=("dataset_id",),
    ),
    "vector.validate": _schema({"dataset_id": _DATASET_ID}, required=("dataset_id",)),
    "vector.repair": _schema({"dataset_id": _DATASET_ID, "output_path": _OUTPUT_PATH}, required=("dataset_id",)),
    "vector.buffer": _schema(
        {"dataset_id": _DATASET_ID, "distance": {"type": "number", "exclusiveMinimum": 0}, "output_path": _OUTPUT_PATH},
        required=("dataset_id", "distance"),
    ),
    "vector.clip": _schema(
        {"dataset_id": _DATASET_ID, "mask_dataset_id": _DATASET_ID, "output_path": _OUTPUT_PATH},
        required=("dataset_id", "mask_dataset_id"),
    ),
    "vector.intersection": _schema(
        {"left_dataset_id": _DATASET_ID, "right_dataset_id": _DATASET_ID, "output_path": _OUTPUT_PATH},
        required=("left_dataset_id", "right_dataset_id"),
    ),
    "vector.dissolve": _schema(
        {"dataset_id": _DATASET_ID, "by": {"type": "string"}, "output_path": _OUTPUT_PATH},
        required=("dataset_id",),
    ),
    "vector.spatial_join": _schema(
        {"left_dataset_id": _DATASET_ID, "right_dataset_id": _DATASET_ID, "predicate": {"type": "string"}, "output_path": _OUTPUT_PATH},
        required=("left_dataset_id", "right_dataset_id"),
    ),
    "raster.inspect": _schema({"dataset_id": _DATASET_ID}, required=("dataset_id",)),
    "raster.clip": _schema(
        {"dataset_id": _DATASET_ID, "mask_dataset_id": _DATASET_ID, "output_path": _OUTPUT_PATH},
        required=("dataset_id", "mask_dataset_id"),
    ),
    "raster.reproject": _schema(
        {"dataset_id": _DATASET_ID, "target_crs": {"type": "string"}, "output_path": _OUTPUT_PATH},
        required=("dataset_id", "target_crs"),
    ),
    "raster.slope": _schema({"dataset_id": _DATASET_ID, "output_path": _OUTPUT_PATH}, required=("dataset_id",)),
    "analysis.distance": _schema(
        {
            "source_dataset_id": _DATASET_ID,
            "target_dataset_id": _DATASET_ID,
            "threshold": {"type": "number", "minimum": 0},
        },
        required=("source_dataset_id",),
    ),
    "analysis.zonal_statistics": _schema(
        {"zones_dataset_id": _DATASET_ID, "raster_dataset_id": _DATASET_ID},
        required=("zones_dataset_id", "raster_dataset_id"),
    ),
    "map.render": _schema(
        {"dataset_id": _DATASET_ID, "title": {"type": "string"}, "output_path": _OUTPUT_PATH},
        required=("dataset_id",),
    ),
}


from . import analysis, crs, dataset, raster, spatial, vector, visualization  # noqa: E402

__all__ = ["register_gis_tools"]
