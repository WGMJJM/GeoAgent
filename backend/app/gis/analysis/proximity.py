"""邻近度与可达性基础分析。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
from pyproj import CRS
from shapely.ops import unary_union

from app.core.models import Dataset, ErrorCategory
from app.gis.errors import GISFailure


def distance_summary(
    source: Dataset,
    target: Dataset | None = None,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    left = gpd.read_file(Path(source.path))
    if left.empty:
        raise GISFailure("EMPTY_DATASET", "距离分析的源数据为空。", category=ErrorCategory.DATA)
    if left.crs is None or left.crs.is_geographic:
        raise GISFailure("CRS_UNIT_MISMATCH", "距离分析需要使用米等线性单位的投影 CRS。", category=ErrorCategory.CRS)
    if target is None:
        distances = left.geometry.distance(unary_union(left.geometry))
    else:
        right = gpd.read_file(Path(target.path))
        if right.empty:
            raise GISFailure("EMPTY_DATASET", "距离分析的目标数据为空。", category=ErrorCategory.DATA)
        if right.crs is None or not CRS.from_user_input(left.crs).equals(CRS.from_user_input(right.crs)):
            raise GISFailure("CRS_MISMATCH", "距离分析的两个 CRS 不一致。", category=ErrorCategory.CRS)
        target_union = unary_union(right.geometry)
        distances = left.geometry.distance(target_union)
    values = [float(value) for value in distances]
    within = sum(value <= threshold for value in values) if threshold is not None else None
    return {
        "feature_count": len(values),
        "min_distance": min(values) if values else None,
        "max_distance": max(values) if values else None,
        "mean_distance": sum(values) / len(values) if values else None,
        "threshold": threshold,
        "within_threshold": within,
    }
