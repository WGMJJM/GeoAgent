"""栅格分区统计的轻量实现，不依赖 rasterstats。"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask

from app.core.models import Dataset, ErrorCategory
from app.gis.errors import GISFailure


def zonal_statistics(zones: Dataset, raster: Dataset) -> dict[str, Any]:
    polygons = gpd.read_file(zones.path)
    if polygons.empty:
        raise GISFailure("EMPTY_DATASET", "分区数据为空。", category=ErrorCategory.DATA)
    rows: list[dict[str, Any]] = []
    with rasterio.open(raster.path) as source:
        if source.crs is None or polygons.crs is None:
            raise GISFailure("CRS_MISSING", "分区统计要求 CRS 完整。", category=ErrorCategory.CRS)
        polygons = polygons.to_crs(source.crs)
        for index, geometry in enumerate(polygons.geometry):
            try:
                values, _ = mask(source, [geometry], crop=True, filled=False)
                data = np.asarray(values[0].compressed() if np.ma.isMaskedArray(values[0]) else values[0]).astype(float)
                data = data[np.isfinite(data)]
                rows.append({"zone_index": index, "count": int(data.size), "mean": float(data.mean()) if data.size else None, "min": float(data.min()) if data.size else None, "max": float(data.max()) if data.size else None})
            except ValueError:
                rows.append({"zone_index": index, "count": 0, "mean": None, "min": None, "max": None})
    valid = [row for row in rows if row["count"]]
    return {"zone_count": len(rows), "zones_with_values": len(valid), "zones": rows}

