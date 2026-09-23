"""矢量结果的结构检查。"""

from __future__ import annotations

import geopandas as gpd

from app.gis.errors import GISFailure


def validate_frame(frame: gpd.GeoDataFrame, *, allow_empty: bool = False) -> dict:
    if frame.empty and not allow_empty:
        raise GISFailure("EMPTY_DATASET", "空间运算结果为空。", category="DATA")
    invalid = int((~frame.geometry.is_valid.fillna(False)).sum()) if "geometry" in frame else 0
    return {
        "feature_count": int(len(frame)),
        "geometry_types": sorted(str(item) for item in frame.geometry.geom_type.dropna().unique()),
        "invalid_geometry_count": invalid,
        "empty": bool(frame.empty),
    }

