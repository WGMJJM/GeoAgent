"""面向工作台的轻量 Dataset 预览读取模型。"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageOps
from rasterio.enums import ColorInterp, Resampling
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
    media_type: str | None = None
    page_count: int | None = None
    text: str | None = None
    image_data_url: str | None = None
    render_note: str | None = None


class DatasetPreviewService:
    def preview(
        self,
        dataset: Dataset,
        workspace: WorkspaceManager,
        *,
        max_features: int = 200,
        max_fields: int = 32,
        max_property_length: int = 160,
        max_dimension: int = 1024,
    ) -> DatasetPreview:
        path = workspace.resolve(dataset.path, allow_missing=False)
        source_crs = dataset.crs.authority if dataset.crs else None
        if dataset.kind is DatasetKind.VECTOR:
            return self._vector(dataset, path, source_crs, max_features, max_fields, max_property_length)
        if dataset.kind is DatasetKind.RASTER:
            return self._raster(dataset, path, source_crs, max_dimension)
        if dataset.kind is DatasetKind.TABLE:
            return self._table(dataset, path, source_crs, max_features, max_fields, max_property_length)
        if dataset.kind is DatasetKind.DOCUMENT:
            return self._document(dataset)
        if dataset.kind is DatasetKind.IMAGE:
            return self._image(dataset, path, max_dimension)
        raise ValueError(f"暂不支持预览的数据类型：{dataset.kind}")

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

    def _raster(self, dataset: Dataset, path: Path, source_crs: str | None, max_dimension: int) -> DatasetPreview:
        with rasterio.open(path) as source:
            scale = min(1, max_dimension / max(source.width, source.height))
            height, width = max(1, round(source.height * scale)), max(1, round(source.width * scale))
            rgb = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            indexes = [source.colorinterp.index(color) + 1 for color in rgb] if all(color in source.colorinterp for color in rgb) else [1]
            pixels = source.read(indexes, out_shape=(len(indexes), height, width), masked=True, resampling=Resampling.nearest)
            valid = ~np.any(np.ma.getmaskarray(pixels), axis=0) & np.all(np.isfinite(pixels.data), axis=0)
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[:, :, 3] = np.where(valid, 255, 0)
            if source.colorinterp[0] == ColorInterp.palette and len(indexes) == 1:
                colors = source.colormap(1)
                palette = np.zeros((max(colors) + 1, 4), dtype=np.uint8)
                for value, color in colors.items():
                    palette[value] = color
                values = pixels.data[0]
                in_palette = valid & (values >= 0) & (values < len(palette))
                rgba[:, :, 3] = 0
                rgba[in_palette] = palette[values[in_palette].astype(np.intp)]
                note = "第 1 波段 · 文件内置色表"
            else:
                for channel in range(3):
                    band = pixels.data[channel if len(indexes) == 3 else 0]
                    if len(indexes) == 3 and band.dtype == np.uint8:
                        rgba[:, :, channel] = band
                    elif np.any(valid):
                        values = band[valid].astype(np.float64)
                        low, high = values.min(), values.max()
                        rgba[:, :, channel][valid] = np.clip((values - low) / (high - low) * 255, 0, 255).astype(np.uint8) if high > low else 127
                note = "RGB 波段" if len(indexes) == 3 else "第 1 波段 · 灰度拉伸（按预览样本）"
            if ColorInterp.alpha in source.colorinterp:
                alpha = source.read(source.colorinterp.index(ColorInterp.alpha) + 1, out_shape=(height, width), resampling=Resampling.nearest)
                rgba[:, :, 3] = np.minimum(rgba[:, :, 3], np.clip(alpha, 0, 255).astype(np.uint8))
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
                image_data_url=_png_data_url(Image.fromarray(rgba)),
                render_note=f"{note}；NoData 透明。仅供查看，不改变原始数据。",
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

    def _document(self, dataset: Dataset) -> DatasetPreview:
        return DatasetPreview(
            dataset_id=dataset.id,
            kind=dataset.kind,
            media_type=str(dataset.metadata.get("media_type") or "application/octet-stream"),
            page_count=dataset.metadata.get("page_count"),
            truncated=bool(dataset.metadata.get("text_truncated")),
            text=str(dataset.metadata.get("text_preview") or ""),
        )

    def _image(self, dataset: Dataset, path: Path, max_dimension: int) -> DatasetPreview:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source)
            width, height = image.size
            bands = len(image.getbands())
            image.thumbnail((max_dimension, max_dimension))
            data_url = _png_data_url(image.convert("RGBA"))
        return DatasetPreview(
            dataset_id=dataset.id,
            kind=dataset.kind,
            media_type=str(dataset.metadata.get("media_type") or "application/octet-stream"),
            width=width,
            height=height,
            bands=bands,
            image_data_url=data_url,
            render_note="等比例缩略图；不改变原始图片。",
        )


def _png_data_url(image: Image.Image) -> str:
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _compact_json(value: Any, max_length: int) -> Any:
    if isinstance(value, dict):
        return {str(key): _compact_json(item, max_length) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_json(item, max_length) for item in value]
    if isinstance(value, str) and len(value) > max_length:
        return value[:max_length] + "…"
    return value


__all__ = ["DatasetPreview", "DatasetPreviewService"]
