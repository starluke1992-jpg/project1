"""危机 Alpha 检验：策略在股市危机里的表现，而不只是长期夏普。

  crisis_episodes  基准（如沪深300）每一段回撤超过阈值的区间里，策略赚了多少
  smile            按基准月度/季度收益分五组，看每组策略的平均收益（“微笑”形状）
  up_down_beta     基准上涨期和下跌期分别回归的 beta；危机 Alpha 要求下跌期 beta < 0
"""
import numpy as np
import pandas as pd


def benchmark_returns(df: pd.DataFrame) -> pd.Series:
    """由 data.py 格式的品种数据（price/adj）算基准日收益：复权价差 / 前一日真实价格。"""
    return (df["adj"].diff() / df["price"].shift()).dropna()


def crisis_episodes(strategy: pd.Series, bench: pd.Series, threshold: float = 0.15) -> pd.DataFrame:
    """找出基准从前高回撤超过 threshold 的每一段（前高 → 最低点），
    列出这段时间基准的跌幅和策略的累计收益（固定资金口径，按日收益相加）。"""
    strategy, bench = strategy.align(bench, join="inner")
    wealth = (1 + bench.fillna(0)).cumprod()
    rows = []
    peak_date, peak, trough_date, in_dd = wealth.index[0], wealth.iloc[0], None, False
    for date, w in wealth.items():
        if w >= peak:
            if in_dd:
                rows.append(_episode(peak_date, trough_date, wealth, strategy, recovered=date))
                in_dd = False
            peak_date, peak, trough_date = date, w, None
            continue
        if trough_date is None or w < wealth[trough_date]:
            trough_date = date
        if w / peak - 1 <= -threshold:
            in_dd = True
    if in_dd:
        rows.append(_episode(peak_date, trough_date, wealth, strategy, recovered=None))
    cols = ["peak", "trough", "recovered", "bench_return", "strategy_return"]
    return pd.DataFrame(rows, columns=cols)


def _episode(peak_date, trough_date, wealth, strategy, recovered):
    window = strategy.loc[peak_date:trough_date].iloc[1:]  # 前高当天的收益不算
    return {
        "peak": peak_date.date(),
        "trough": trough_date.date(),
        "recovered": recovered.date() if recovered is not None else None,
        "bench_return": round(float(wealth[trough_date] / wealth[peak_date] - 1), 4),
        "strategy_return": round(float(window.sum()), 4),
    }


def _period(strategy, bench, freq):
    strategy, bench = strategy.align(bench, join="inner")
    key = bench.index.to_period(freq)
    s = strategy.groupby(key).sum()
    b = (1 + bench).groupby(key).prod() - 1
    return s, b


def smile(strategy: pd.Series, bench: pd.Series, freq: str = "M", buckets: int = 5) -> pd.DataFrame:
    """把基准的月度（freq='Q' 为季度）收益从低到高分成 buckets 组，统计每组的平均收益。
    危机 Alpha 的特征：最差一组（第 1 组）策略平均收益为正，两端高、中间低。"""
    s, b = _period(strategy, bench, freq)
    group = pd.qcut(b.rank(method="first"), buckets, labels=range(1, buckets + 1))
    return pd.DataFrame({
        "bench_avg": b.groupby(group, observed=True).mean(),
        "strategy_avg": s.groupby(group, observed=True).mean(),
        "strategy_hit_rate": (s > 0).groupby(group, observed=True).mean(),
        "periods": b.groupby(group, observed=True).size(),
    }).round(4)


def up_down_beta(strategy: pd.Series, bench: pd.Series, freq: str = "M") -> dict:
    """基准上涨期、下跌期分别对基准回归的 beta，以及整体相关系数和最差 10% 时期的平均收益。"""
    s, b = _period(strategy, bench, freq)

    def beta(mask):
        if mask.sum() < 3 or b[mask].var() == 0:
            return np.nan
        return float(np.cov(s[mask], b[mask])[0, 1] / b[mask].var())

    worst = b <= b.quantile(0.1)
    return {
        "corr": round(float(s.corr(b)), 3),
        "beta_up": round(beta(b > 0), 3),
        "beta_down": round(beta(b < 0), 3),
        "worst10_bench_avg": round(float(b[worst].mean()), 4),
        "worst10_strategy_avg": round(float(s[worst].mean()), 4),
    }


def crisis_report(strategy: pd.Series, bench: pd.Series, threshold: float = 0.15) -> dict:
    return {
        "episodes": crisis_episodes(strategy, bench, threshold),
        "smile_monthly": smile(strategy, bench, "M"),
        "smile_quarterly": smile(strategy, bench, "Q"),
        "beta": up_down_beta(strategy, bench, "M"),
    }
