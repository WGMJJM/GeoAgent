"""面向任务的 GIS 分析函数。"""

from .proximity import distance_summary
from .statistics import zonal_statistics

__all__ = ["distance_summary", "zonal_statistics"]

