"""Dataset Inspector：把大型空间数据压缩成可供 Agent 使用的元数据。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import rasterio
from pyproj import CRS

from app.core.models import BoundingBox, CRSInfo, Dataset, DatasetKind, DatasetSchema, new_id
from app.gis.errors import GISFailure

VECTOR_EXTENSIONS = {".shp", ".gpkg", ".geojson", ".json", ".kml", ".gml", ".zip"}
RASTER_EXTENSIONS = {".tif", ".tiff", ".img", ".vrt", ".asc"}
TABLE_EXTENSIONS = {".csv", ".tsv", ".parquet", ".jsonl"}


class DatasetInspector:
    """读取轻量统计信息，不把原始 Dataset 内容塞进模型上下文。"""

    def inspect(self, path: str | Path, *, dataset_id: str | None = None, name: str | None = None) -> Dataset:
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            raise GISFailure("MISSING_DATASET", f"找不到数据集：{file_path}", category="DATA")
        if not file_path.is_file():
            raise GISFailure("UNSUPPORTED_DATASET_PATH", f"数据集路径不是文件：{file_path}", category="DATA")

        kind = self.detect_kind(file_path)
        if kind is DatasetKind.VECTOR:
            dataset = self._inspect_vector(file_path, name=name)
        elif kind is DatasetKind.RASTER:
            dataset = self._inspect_raster(file_path, name=name)
        else:
            dataset = self._inspect_table(file_path, name=name)
        if dataset_id:
            dataset = dataset.model_copy(update={"id": dataset_id})
        return dataset

    @staticmethod
    def detect_kind(path: Path) -> DatasetKind:
        suffix = path.suffix.lower()
        if suffix in RASTER_EXTENSIONS:
            return DatasetKind.RASTER
        if suffix in VECTOR_EXTENSIONS:
            return DatasetKind.VECTOR
        if suffix in TABLE_EXTENSIONS:
            return DatasetKind.TABLE
        raise GISFailure("UNSUPPORTED_FORMAT", f"暂不支持的数据格式：{suffix or path.name}", category="DATA")

    def _inspect_vector(self, path: Path, *, name: str | None) -> Dataset:
        try:
            frame = gpd.read_file(path)
        except Exception as exc:
            raise GISFailure("VECTOR_READ_FAILED", f"无法读取矢量数据：{exc}", category="DATA") from exc
        crs = _crs_info(frame.crs)
        bounds = _bounds(frame.total_bounds)
        geometry_type = ", ".join(sorted(str(item) for item in frame.geometry.geom_type.dropna().unique())) or None
        invalid = int((~frame.geometry.is_valid.fillna(False)).sum()) if "geometry" in frame else None
        fields = {column: str(dtype) for column, dtype in frame.dtypes.items() if column != "geometry"}
        schema = DatasetSchema(
            fields=fields,
            geometry_type=geometry_type,
            feature_count=len(frame),
            invalid_geometry_count=invalid,
        )
        return Dataset(
            id=new_id("ds"),
            name=name or path.stem,
            kind=DatasetKind.VECTOR,
            path=str(path),
            format=path.suffix.lower().lstrip(".") or "vector",
            crs=crs,
            extent=bounds,
            schema=schema,
            metadata={"columns": list(frame.columns), "empty": frame.empty},
        )

    def _inspect_raster(self, path: Path, *, name: str | None) -> Dataset:
        try:
            with rasterio.open(path) as src:
                crs = _crs_info(src.crs)
                bounds = BoundingBox(min_x=src.bounds.left, min_y=src.bounds.bottom, max_x=src.bounds.right, max_y=src.bounds.top)
                schema = DatasetSchema(
                    width=src.width,
                    height=src.height,
                    bands=src.count,
                    resolution=(float(src.res[0]), float(src.res[1])),
                    nodata=_json_number(src.nodata),
                )
                metadata = {
                    "dtype": list(src.dtypes),
                    "driver": src.driver,
                    "transform": tuple(float(v) for v in src.transform),
                    "bounds": [bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y],
                }
        except Exception as exc:
            raise GISFailure("RASTER_READ_FAILED", f"无法读取栅格数据：{exc}", category="RASTER") from exc
        return Dataset(
            name=name or path.stem,
            kind=DatasetKind.RASTER,
            path=str(path),
            format=path.suffix.lower().lstrip(".") or "raster",
            crs=crs,
            extent=bounds,
            schema=schema,
            metadata=metadata,
        )

    def _inspect_table(self, path: Path, *, name: str | None) -> Dataset:
        try:
            if path.suffix.lower() == ".tsv":
                frame = pd.read_csv(path, sep="\t", nrows=1000)
            elif path.suffix.lower() == ".jsonl":
                frame = pd.read_json(path, lines=True, nrows=1000)
            else:
                frame = pd.read_csv(path, nrows=1000) if path.suffix.lower() != ".parquet" else pd.read_parquet(path)
        except Exception as exc:
            raise GISFailure("TABLE_READ_FAILED", f"无法读取表格数据：{exc}", category="DATA") from exc
        return Dataset(
            name=name or path.stem,
            kind=DatasetKind.TABLE,
            path=str(path),
            format=path.suffix.lower().lstrip(".") or "table",
            schema=DatasetSchema(fields={column: str(dtype) for column, dtype in frame.dtypes.items()}, feature_count=len(frame)),
            metadata={"columns": list(frame.columns), "sample_rows": frame.head(3).to_dict(orient="records")},
        )


def _crs_info(value: Any) -> CRSInfo | None:
    if value is None:
        return None
    try:
        crs = CRS.from_user_input(value)
        authority = crs.to_authority()
        return CRSInfo(
            authority=f"{authority[0]}:{authority[1]}" if authority else crs.to_string(),
            name=crs.name,
            is_geographic=bool(crs.is_geographic),
            linear_unit=(crs.axis_info[0].unit_name if crs.axis_info else None),
        )
    except Exception:
        return CRSInfo(authority=str(value))


def _bounds(values: Any) -> BoundingBox | None:
    try:
        values = [float(value) for value in values]
        if len(values) != 4 or any(not math.isfinite(value) for value in values):
            return None
        return BoundingBox(min_x=values[0], min_y=values[1], max_x=values[2], max_y=values[3])
    except Exception:
        return None


def _json_number(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        value = float(value)
        return int(value) if value.is_integer() else value
    except (TypeError, ValueError):
        return None
