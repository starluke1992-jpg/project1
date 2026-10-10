"""AKShare 数据接入：统一字段的取数函数、本地 CSV 缓存、数据检查和网络自检。"""
from .akshare_api import SOURCES, fetch
from .cache import SeriesCache
from .checks import check_series
from .connectivity import check_all

__all__ = ["SOURCES", "fetch", "SeriesCache", "check_series", "check_all"]
