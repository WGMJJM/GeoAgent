"""CRS 识别、投影选择与重投影。"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from pyproj import CRS

from app.core.models import Dataset
from app.gis.crs.validator import require_crs
from app.gis.errors import GISFailure
from app.gis.vector.service import VectorService


class CRSService:
    def __init__(self, vector_service: VectorService | None = None, default_crs: str = "EPSG:3857") -> None:
        self.vector_service = vector_service or VectorService()
        self.default_crs = default_crs

    def inspect(self, dataset: Dataset | str | Path) -> dict:
        path = Path(dataset.path if isinstance(dataset, Dataset) else dataset)
        if path.suffix.lower() in {".shp", ".gpkg", ".geojson", ".json", ".kml", ".gml"}:
            frame = gpd.read_file(path)
            crs = require_crs(frame.crs)
            return _describe(crs)
        import rasterio

        with rasterio.open(path) as source:
            crs = require_crs(source.crs)
        return _describe(crs)

    def choose_projected_crs(self, dataset: Dataset | str | Path, preferred: str | None = None) -> str:
        if preferred:
            crs = require_crs(preferred)
            if crs.is_geographic:
                raise GISFailure("TARGET_CRS_NOT_PROJECTED", f"目标 CRS 不是投影 CRS：{preferred}")
            return crs.to_string()
        path = Path(dataset.path if isinstance(dataset, Dataset) else dataset)
        if isinstance(dataset, Dataset) and dataset.extent and dataset.crs and dataset.crs.is_geographic:
            center_x = (dataset.extent.min_x + dataset.extent.max_x) / 2
            center_y = (dataset.extent.min_y + dataset.extent.max_y) / 2
            if -180 <= center_x <= 180 and -80 <= center_y <= 84:
                zone = int((center_x + 180) // 6) + 1
                return f"EPSG:{32600 + zone if center_y >= 0 else 32700 + zone}"
        try:
            frame = gpd.read_file(path)
            if frame.crs and frame.crs.is_geographic and not frame.empty:
                estimate = frame.estimate_utm_crs()
                if estimate:
                    return estimate.to_string()
        except Exception:
            pass
        crs = require_crs(self.default_crs)
        if crs.is_geographic:
            raise GISFailure("DEFAULT_CRS_NOT_PROJECTED", f"默认 CRS 不是投影 CRS：{self.default_crs}")
        return crs.to_string()

    def reproject(
        self,
        dataset: Dataset | str | Path,
        target_crs: str,
        output_path: str | Path,
        *,
        run_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> Dataset:
        source = dataset if isinstance(dataset, Dataset) else None
        if source is None:
            from app.gis.dataset.inspector import DatasetInspector

            source = DatasetInspector().inspect(dataset)
        require_crs(source.crs.authority if source.crs else None)
        return self.vector_service.reproject(source, target_crs, output_path, run_id=run_id, tool_call_id=tool_call_id)


def _describe(crs: CRS) -> dict:
    authority = crs.to_authority()
    return {
        "authority": f"{authority[0]}:{authority[1]}" if authority else crs.to_string(),
        "name": crs.name,
        "is_geographic": bool(crs.is_geographic),
        "is_projected": bool(crs.is_projected),
        "linear_unit": crs.axis_info[0].unit_name if crs.axis_info else None,
    }
