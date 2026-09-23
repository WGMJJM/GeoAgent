"""Rasterio-backed raster operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import Resampling, calculate_default_transform, reproject

from app.core.models import Dataset, DatasetKind, ErrorCategory, new_id
from app.gis.errors import GISFailure
from app.gis.raster.validator import raster_summary


class RasterService:
    def inspect(self, dataset: Dataset | str | Path) -> dict[str, Any]:
        path = Path(dataset.path if isinstance(dataset, Dataset) else dataset)
        try:
            with rasterio.open(path) as source:
                values = source.read(1, out_shape=(min(source.height, 512), min(source.width, 512)))
                return {
                    "path": str(path.resolve()),
                    "width": source.width,
                    "height": source.height,
                    "bands": source.count,
                    "crs": str(source.crs) if source.crs else None,
                    "resolution": [float(source.res[0]), float(source.res[1])],
                    "nodata": source.nodata,
                    "bounds": [source.bounds.left, source.bounds.bottom, source.bounds.right, source.bounds.top],
                    "statistics": raster_summary(values, source.nodata),
                }
        except Exception as exc:
            raise GISFailure("RASTER_READ_FAILED", f"无法读取栅格：{exc}", category=ErrorCategory.RASTER) from exc

    def clip(
        self,
        dataset: Dataset,
        mask_dataset: Dataset,
        output_path: str | Path,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> Dataset:
        shapes = gpd.read_file(mask_dataset.path)
        if shapes.empty:
            raise GISFailure("EMPTY_DATASET", "裁剪边界为空。", category=ErrorCategory.DATA)
        with rasterio.open(dataset.path) as source:
            if source.crs is None or shapes.crs is None:
                raise GISFailure("CRS_MISSING", "栅格裁剪要求输入 CRS 完整。", category=ErrorCategory.CRS)
            shapes = shapes.to_crs(source.crs)
            try:
                clipped, transform = mask(source, shapes.geometry, crop=True)
            except ValueError as exc:
                raise GISFailure("NO_OVERLAP", f"裁剪边界与栅格没有重叠：{exc}", category=ErrorCategory.RASTER) from exc
            profile = source.profile.copy()
            profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=transform)
            _write_raster(output_path, profile, clipped)
        return _derived_raster(dataset, output_path, run_id, "raster.clip", tool_call_id, mask_dataset_id=mask_dataset.id)

    def reproject(
        self,
        dataset: Dataset,
        target_crs: str,
        output_path: str | Path,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> Dataset:
        with rasterio.open(dataset.path) as source:
            if source.crs is None:
                raise GISFailure("CRS_MISSING", "栅格没有 CRS，无法重投影。", category=ErrorCategory.CRS)
            transform, width, height = calculate_default_transform(source.crs, target_crs, source.width, source.height, *source.bounds)
            profile = source.profile.copy()
            profile.update(crs=target_crs, transform=transform, width=width, height=height)
            target = Path(output_path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(target, "w", **profile) as destination:
                for band in range(1, source.count + 1):
                    reproject(
                        source=rasterio.band(source, band),
                        destination=rasterio.band(destination, band),
                        src_transform=source.transform,
                        src_crs=source.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest,
                    )
        return _derived_raster(dataset, output_path, run_id, "raster.reproject", tool_call_id, target_crs=target_crs)

    def slope(
        self,
        dataset: Dataset,
        output_path: str | Path,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> Dataset:
        with rasterio.open(dataset.path) as source:
            if source.crs is None:
                raise GISFailure("CRS_MISSING", "DEM 没有 CRS，无法解释坡度单位。", category=ErrorCategory.CRS)
            if source.crs.is_geographic:
                raise GISFailure("CRS_UNIT_MISMATCH", "坡度计算需要米制投影 CRS，请先重投影 DEM。", category=ErrorCategory.CRS, details={"source_crs": str(source.crs), "repairable": True})
            elevation = source.read(1).astype("float32")
            x_res, y_res = source.res
            gy, gx = np.gradient(elevation, y_res, x_res)
            slope_degrees = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
            profile = source.profile.copy()
            profile.update(dtype="float32", count=1, nodata=-9999.0)
            if source.nodata is not None:
                slope_degrees[elevation == source.nodata] = -9999.0
            _write_raster(output_path, profile, slope_degrees[np.newaxis, ...])
        return _derived_raster(dataset, output_path, run_id, "raster.slope", tool_call_id)


def _write_raster(output_path: str | Path, profile: dict[str, Any], values: np.ndarray) -> None:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(target, "w", **profile) as destination:
            destination.write(values)
    except Exception as exc:
        raise GISFailure("RASTER_WRITE_FAILED", f"无法写出栅格结果：{exc}", category=ErrorCategory.RASTER) from exc


def _derived_raster(
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
        kind=DatasetKind.RASTER,
        path=str(target),
        format=target.suffix.lower().lstrip("."),
        source_dataset_ids=[source.id],
        created_by_run_id=run_id,
        metadata={"operation": operation, "parameters": parameters, "tool_call_id": tool_call_id},
    )
