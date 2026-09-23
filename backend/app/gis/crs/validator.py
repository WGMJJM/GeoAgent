"""CRS 安全检查。"""

from __future__ import annotations

from typing import Any

from pyproj import CRS

from app.core.models import ErrorCategory
from app.gis.errors import GISFailure


def require_crs(value: Any) -> CRS:
    if value is None:
        raise GISFailure("CRS_MISSING", "数据集没有 CRS，无法安全进行空间计算。", category=ErrorCategory.CRS)
    try:
        return CRS.from_user_input(value)
    except Exception as exc:
        raise GISFailure("CRS_INVALID", f"无法解析 CRS：{value}", category=ErrorCategory.CRS) from exc


def require_projected(value: Any) -> CRS:
    crs = require_crs(value)
    if crs.is_geographic:
        raise GISFailure(
            "CRS_UNIT_MISMATCH",
            "当前 CRS 使用经纬度单位，不能直接使用米执行距离/缓冲区运算。",
            category=ErrorCategory.CRS,
            details={"source_crs": crs.to_string(), "reason": "geographic_crs"},
        )
    return crs

