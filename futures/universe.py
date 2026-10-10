"""B2 的品种池：每个调仓日按规则生成，不手工挑。

规则（docs/plan_A_B.md 3.4 节）：
  - 连续序列已有至少 min_days 个交易日（新品种上市满一年才纳入）；
  - 过去 60 日日均成交量（全部合约合计，单位：手）不低于 min_volume。
成交量口径各交易所略有差异（单边/双边），min_volume 设得比较宽松；首次用真实数据时请抽查。
"""
import pandas as pd


def eligibility(prices: pd.DataFrame, volumes: pd.DataFrame = None, min_days: int = 250,
                min_volume: float = 20000, window: int = 60) -> pd.DataFrame:
    """返回与 prices 同形状的布尔表：某天某品种是否可交易。只用到当天及以前的数据。"""
    listed = prices.notna().cumsum() >= min_days
    if volumes is None:
        return listed
    liquid = volumes.reindex_like(prices).rolling(window, min_periods=window // 2).mean() >= min_volume
    return listed & liquid
