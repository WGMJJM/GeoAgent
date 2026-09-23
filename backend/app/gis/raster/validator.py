"""栅格结果检查。"""

from __future__ import annotations

from typing import Any

import numpy as np


def raster_summary(array: np.ndarray, nodata: Any = None) -> dict[str, Any]:
    valid = array
    if nodata is not None:
        valid = array[array != nodata]
    valid = valid[np.isfinite(valid)] if valid.size else valid
    return {
        "cell_count": int(array.size),
        "valid_cell_count": int(valid.size),
        "min": float(valid.min()) if valid.size else None,
        "max": float(valid.max()) if valid.size else None,
        "mean": float(valid.mean()) if valid.size else None,
        "empty": bool(valid.size == 0),
    }

