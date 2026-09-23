"""GeoPandas-backed vector operations.

服务只负责确定性 GIS 计算，不做用户意图判断；Agent/Tool 层负责把它们暴露为
可审计的 ToolResult。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
from pyproj import CRS

from app.core.models import Dataset, DatasetKind, ErrorCategory, new_id
from app.gis.crs.validator import require_crs, require_projected
from app.gis.errors import GISFailure
from app.gis.vector.validator import validate_frame


class VectorService:
    def read(self, dataset: Dataset | str | Path) -> gpd.GeoDataFrame:
        path = Path(dataset.path if isinstance(dataset, Dataset) else dataset)
        try:
            return gpd.read_file(path)
        except Exception as exc:
            raise GISFailure("VECTOR_READ_FAILED", f"无法读取矢量数据：{exc}", category=ErrorCategory.DATA) from exc

    def validate(self, dataset: Dataset | str | Path) -> dict[str, Any]:
        frame = self.read(dataset)
        result = validate_frame(frame, allow_empty=True)
        result.update({"path": str(Path(dataset.path if isinstance(dataset, Dataset) else dataset).resolve()), "crs": str(frame.crs) if frame.crs else None})
        return result

    def repair(
        self,
        dataset: Dataset,
        output_path: str | Path,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> Dataset:
        frame = self.read(dataset).copy()
        if "geometry" not in frame:
            raise GISFailure("GEOMETRY_COLUMN_MISSING", "矢量数据没有 geometry 列。", category=ErrorCategory.GEOMETRY)
        try:
            from shapely import make_valid

            frame["geometry"] = frame.geometry.apply(lambda value: make_valid(value) if value is not None and not value.is_valid else value)
        except ImportError:
            frame["geometry"] = frame.geometry.apply(lambda value: value.buffer(0) if value is not None and not value.is_valid else value)
        self._write(frame, output_path)
        return _derived_dataset(dataset, output_path, run_id, "vector.repair", tool_call_id)

    def reproject(
        self,
        dataset: Dataset,
        target_crs: str,
        output_path: str | Path,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> Dataset:
        frame = self.read(dataset)
        require_crs(frame.crs)
        try:
            transformed = frame.to_crs(target_crs)
        except Exception as exc:
            raise GISFailure("REPROJECT_FAILED", f"重投影失败：{exc}", category=ErrorCategory.CRS) from exc
        self._write(transformed, output_path)
        return _derived_dataset(dataset, output_path, run_id, "crs.reproject", tool_call_id, target_crs=target_crs)

    def buffer(
        self,
        dataset: Dataset,
        distance: float,
        output_path: str | Path,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> Dataset:
        if distance <= 0:
            raise GISFailure("INVALID_DISTANCE", "buffer distance 必须大于 0。", category=ErrorCategory.INPUT)
        frame = self.read(dataset)
        require_projected(frame.crs)
        try:
            result = frame.copy()
            result["geometry"] = result.geometry.buffer(distance)
            validate_frame(result)
        except GISFailure:
            raise
        except Exception as exc:
            raise GISFailure("BUFFER_FAILED", f"缓冲区计算失败：{exc}", category=ErrorCategory.GEOMETRY) from exc
        self._write(result, output_path)
        return _derived_dataset(dataset, output_path, run_id, "vector.buffer", tool_call_id, distance=distance)

    def clip(
        self,
        dataset: Dataset,
        mask: Dataset,
        output_path: str | Path,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> Dataset:
        left = self.read(dataset)
        right = self.read(mask)
        _align_crs(left, right)
        try:
            result = gpd.clip(left, right)
            validate_frame(result)
        except GISFailure:
            raise
        except Exception as exc:
            raise GISFailure("CLIP_FAILED", f"裁剪失败：{exc}", category=ErrorCategory.GEOMETRY) from exc
        self._write(result, output_path)
        return _derived_dataset(dataset, output_path, run_id, "vector.clip", tool_call_id, mask_dataset_id=mask.id)

    def intersection(
        self,
        left_dataset: Dataset,
        right_dataset: Dataset,
        output_path: str | Path,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> Dataset:
        left = self.read(left_dataset)
        right = self.read(right_dataset)
        _align_crs(left, right)
        if (~left.geometry.is_valid.fillna(False)).any() or (~right.geometry.is_valid.fillna(False)).any():
            raise GISFailure("INVALID_GEOMETRY", "intersection 输入存在无效 geometry。", category=ErrorCategory.GEOMETRY, details={"repairable": True})
        try:
            result = gpd.overlay(left, right, how="intersection")
            validate_frame(result)
        except GISFailure:
            raise
        except Exception as exc:
            raise GISFailure("INTERSECTION_FAILED", f"相交分析失败：{exc}", category=ErrorCategory.GEOMETRY) from exc
        self._write(result, output_path)
        return _derived_dataset(left_dataset, output_path, run_id, "vector.intersection", tool_call_id, right_dataset_id=right_dataset.id)

    def dissolve(
        self,
        dataset: Dataset,
        by: str | None,
        output_path: str | Path,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> Dataset:
        frame = self.read(dataset)
        if by and by not in frame.columns:
            raise GISFailure("MISSING_FIELD", f"字段不存在：{by}", category=ErrorCategory.DATA, details={"available_fields": list(frame.columns)})
        result = frame.dissolve(by=by, as_index=False) if by else frame.dissolve(as_index=False)
        validate_frame(result)
        self._write(result, output_path)
        return _derived_dataset(dataset, output_path, run_id, "vector.dissolve", tool_call_id, by=by)

    def spatial_join(
        self,
        left_dataset: Dataset,
        right_dataset: Dataset,
        output_path: str | Path,
        *,
        predicate: str = "intersects",
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> Dataset:
        left = self.read(left_dataset)
        right = self.read(right_dataset)
        _align_crs(left, right)
        try:
            result = gpd.sjoin(left, right, how="left", predicate=predicate)
        except ImportError as exc:
            raise GISFailure("SPATIAL_INDEX_MISSING", "spatial_join 需要 GeoPandas 空间索引依赖。", category=ErrorCategory.RESOURCE) from exc
        except Exception as exc:
            raise GISFailure("SPATIAL_JOIN_FAILED", f"空间连接失败：{exc}", category=ErrorCategory.GEOMETRY) from exc
        self._write(result, output_path)
        return _derived_dataset(left_dataset, output_path, run_id, "vector.spatial_join", tool_call_id, right_dataset_id=right_dataset.id, predicate=predicate)

    def _write(self, frame: gpd.GeoDataFrame, output_path: str | Path) -> None:
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix.lower()
        try:
            if suffix == ".gpkg":
                frame.to_file(target, driver="GPKG", layer=target.stem[:60])
            elif suffix in {".geojson", ".json"}:
                frame.to_file(target, driver="GeoJSON")
            elif suffix == ".shp":
                frame.to_file(target, driver="ESRI Shapefile")
            else:
                raise GISFailure("UNSUPPORTED_FORMAT", f"不支持的矢量输出格式：{suffix}", category=ErrorCategory.DATA)
        except GISFailure:
            raise
        except Exception as exc:
            raise GISFailure("VECTOR_WRITE_FAILED", f"无法写出矢量结果：{exc}", category=ErrorCategory.DATA) from exc


def _align_crs(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame) -> None:
    if left.crs is None or right.crs is None:
        raise GISFailure("CRS_MISSING", "叠加分析要求两个输入都具备 CRS。", category=ErrorCategory.CRS)
    if not CRS.from_user_input(left.crs).equals(CRS.from_user_input(right.crs)):
        raise GISFailure(
            "CRS_MISMATCH",
            "空间叠加的两个数据集 CRS 不一致。",
            category=ErrorCategory.CRS,
            details={"left_crs": str(left.crs), "right_crs": str(right.crs), "repairable": True},
        )


def _derived_dataset(
    source: Dataset,
    output_path: str | Path,
    run_id: str | None,
    operation: str,
    tool_call_id: str | None,
    **parameters: Any,
) -> Dataset:
    target = Path(output_path).expanduser().resolve()
    return Dataset(
        id=new_id("ds"),
        name=target.stem,
        kind=DatasetKind.VECTOR,
        path=str(target),
        format=target.suffix.lower().lstrip("."),
        source_dataset_ids=[source.id],
        created_by_run_id=run_id,
        crs=None,
        metadata={"operation": operation, "parameters": parameters, "tool_call_id": tool_call_id},
    )
