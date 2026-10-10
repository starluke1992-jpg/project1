"""趋势过滤：决定每个资产这期持有多少比例（0 / 0.5 / 1）。

ma_mom_exposure：方案 A v1 的规则，两个公开信号各占一半：
  收盘价 > 200 日均线；过去 252 日收益 > 同期现金收益。

negative_trend_exposure：作者"负向趋势因子"的近似实现（挑战者，用来和 v1 比较）。
  作者只说它综合了波动率、趋势力度和成交量/持仓量，公式没有公开，这里按描述做了一个可复现的版本：
    趋势力度 = 过去 60 日对数价格回归斜率的 t 值；
    波动放大 = 20 日波动 / 120 日波动 > 1.2 且趋势向下时，再扣一分；
    成交量放大 = 有成交量数据时，20 日均量 / 120 日均量 > 1.2 且趋势向下，再扣半分。
  综合分低于阈值（默认 −1）判定为"处于下跌趋势"，暴露设为 0，否则为 1。
  这不是作者的原公式，只是同一思路的公开版本。
"""
import numpy as np
import pandas as pd


def ma_mom_exposure(prices: pd.DataFrame, cash_index: pd.Series, ma: int = 200, mom: int = 252) -> pd.Series:
    """prices、cash_index 是截至信号日的价格（累计净值）序列，返回各资产的暴露系数。"""
    out = {}
    for a in prices.columns:
        p = prices[a].dropna()
        if len(p) < max(ma, mom + 1):
            out[a] = np.nan
            continue
        above_ma = p.iloc[-1] > p.iloc[-ma:].mean()
        c = cash_index.reindex(p.index).ffill()
        beat_cash = p.iloc[-1] / p.iloc[-mom - 1] > c.iloc[-1] / c.iloc[-mom - 1]
        out[a] = 0.5 * above_ma + 0.5 * beat_cash
    return pd.Series(out, dtype=float)


def _slope_tstat(logp: np.ndarray) -> float:
    n = len(logp)
    x = np.arange(n) - (n - 1) / 2
    beta = (x * (logp - logp.mean())).sum() / (x ** 2).sum()
    resid = logp - logp.mean() - beta * x
    se = np.sqrt((resid ** 2).sum() / (n - 2) / (x ** 2).sum())
    return float(beta / se) if se > 0 else 0.0


def negative_trend_score(price: pd.Series, volume: pd.Series = None, trend_window: int = 60,
                         vol_short: int = 20, vol_long: int = 120) -> float:
    p = price.dropna()
    if len(p) < max(trend_window, vol_long + 1):
        return np.nan
    logp = np.log(p.to_numpy(dtype=float))
    score = _slope_tstat(logp[-trend_window:])
    r = np.diff(logp)
    vol_ratio = r[-vol_short:].std(ddof=1) / max(r[-vol_long:].std(ddof=1), 1e-12)
    if score < 0 and vol_ratio > 1.2:
        score -= 1.0
    if volume is not None:
        v = volume.reindex(p.index).dropna()
        if len(v) >= vol_long and v.iloc[-vol_long:].mean() > 0:
            if score < 0 and v.iloc[-vol_short:].mean() / v.iloc[-vol_long:].mean() > 1.2:
                score -= 0.5
    return score


def negative_trend_exposure(prices: pd.DataFrame, volumes: pd.DataFrame = None, threshold: float = -1.0,
                            **kw) -> pd.Series:
    out = {}
    for a in prices.columns:
        vol = volumes[a] if volumes is not None and a in volumes.columns else None
        s = negative_trend_score(prices[a], vol, **kw)
        out[a] = np.nan if np.isnan(s) else float(s >= threshold)
    return pd.Series(out, dtype=float)
