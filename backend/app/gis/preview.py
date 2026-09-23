"""面向工作台的轻量 Dataset 预览读取模型。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import rasterio
from pydantic import BaseModel, ConfigDict, Field

from app.core.models import Dataset, DatasetKind
from app.execution.sandbox.manager import WorkspaceManager


class DatasetPreview(BaseModel):
    """UI read model；不暴露 Dataset 的完整路径、所有 metadata 或原始对象。"""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    kind: DatasetKind
    crs: str | None = None
    source_crs: str | None = None
    bbox: list[float] | None = None
    feature_count: int | None = None
    truncated: bool = False
    geojson: dict[str, Any] | None = None
    width: int | None = None
    height: int | None = None
    bands: int | None = None
    resolution: list[float] | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class DatasetPreviewService:
    def preview(
        self,
        dataset: Dataset,
        workspace: WorkspaceManager,
        *,
        max_features: int = 200,
        max_fields: int = 32,
        max_property_length: int = 160,
    ) -> DatasetPreview:
        path = workspace.resolve(dataset.path, allow_missing=False)
        source_crs = dataset.crs.authority if dataset.crs else None
        if dataset.kind is DatasetKind.VECTOR:
            return self._vector(dataset, path, source_crs, max_features, max_fields, max_property_length)
        if dataset.kind is DatasetKind.RASTER:
            return self._raster(dataset, path, source_crs)
        return self._table(dataset, path, source_crs, max_features, max_fields, max_property_length)

    def _vector(self, dataset: Dataset, path: Path, source_crs: str | None, limit: int, max_fields: int, max_property_length: int) -> DatasetPreview:
        frame = gpd.read_file(path, rows=limit)
        if len(frame.columns) > max_fields + 1:
            geometry_column = frame.geometry.name if frame.geometry.name in frame.columns else "geometry"
            columns = [item for item in frame.columns if item != geometry_column][:max_fields] + [geometry_column]
            frame = frame[columns]
        if frame.crs is not None and str(frame.crs) != "EPSG:4326":
            frame = frame.to_crs("EPSG:4326")
        geojson = json.loads(frame.to_json(drop_id=False))
        geojson = _compact_json(geojson, max_property_length)
        bbox = [float(value) for value in frame.total_bounds] if not frame.empty else None
        feature_count = dataset.schema.feature_count if dataset.schema else None
        return DatasetPreview(
            dataset_id=dataset.id,
            kind=dataset.kind,
            crs="EPSG:4326" if frame.crs is not None else source_crs,
            source_crs=source_crs,
            bbox=bbox,
            feature_count=feature_count,
            truncated=feature_count is not None and feature_count > len(frame),
            geojson=geojson,
            columns=[str(item) for item in frame.columns if item != frame.geometry.name],
        )

    def _raster(self, dataset: Dataset, path: Path, source_crs: str | None) -> DatasetPreview:
        with rasterio.open(path) as source:
            bbox = [float(source.bounds.left), float(source.bounds.bottom), float(source.bounds.right), float(source.bounds.top)]
            return DatasetPreview(
                dataset_id=dataset.id,
                kind=dataset.kind,
                crs=source_crs,
                source_crs=source_crs,
                bbox=bbox,
                width=source.width,
                height=source.height,
                bands=source.count,
                resolution=[float(source.res[0]), float(source.res[1])],
            )

    def _table(self, dataset: Dataset, path: Path, source_crs: str | None, limit: int, max_fields: int, max_property_length: int) -> DatasetPreview:
        if path.suffix.lower() == ".tsv":
            frame = pd.read_csv(path, sep="\t", nrows=limit)
        elif path.suffix.lower() == ".jsonl":
            frame = pd.read_json(path, lines=True, nrows=limit)
        else:
            frame = pd.read_csv(path, nrows=limit) if path.suffix.lower() != ".parquet" else pd.read_parquet(path).head(limit)
        frame = frame.iloc[:, :max_fields]
        rows = _compact_json(frame.to_dict(orient="records"), max_property_length)
        feature_count = dataset.schema.feature_count if dataset.schema else None
        return DatasetPreview(
            dataset_id=dataset.id,
            kind=dataset.kind,
            crs=source_crs,
            source_crs=source_crs,
            feature_count=feature_count,
            truncated=feature_count is not None and feature_count > len(frame),
            columns=[str(item) for item in frame.columns],
            rows=rows,
        )


def _compact_json(value: Any, max_length: int) -> Any:
    if isinstance(value, dict):
        return {str(key): _compact_json(item, max_length) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_json(item, max_length) for item in value]
    if isinstance(value, str) and len(value) > max_length:
        return value[:max_length] + "…"
    return value


__all__ = ["DatasetPreview", "DatasetPreviewService"]
