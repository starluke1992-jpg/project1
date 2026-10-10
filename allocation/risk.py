"""风险估计：年化波动率、Ledoit-Wolf 收缩协方差。"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ann_vol(returns: pd.DataFrame, floor: float = 0.0) -> pd.Series:
    """各列日收益的年化波动率，不低于 floor（防止低波动时权重暴涨）。"""
    return (returns.std(ddof=1) * np.sqrt(TRADING_DAYS)).clip(lower=floor)


def ledoit_wolf(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf (2004) 向"等方差单位阵"收缩的协方差，年化。

    样本短、资产多时，样本协方差噪声很大；收缩估计把它往一个简单结构拉，误差更小。
    公式与 sklearn.covariance.LedoitWolf 相同。传入的 returns 不能有缺失值。
    """
    x = returns.to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    t, n = x.shape
    if n == 1:
        return pd.DataFrame([[x.var(ddof=0) * TRADING_DAYS]], index=returns.columns, columns=returns.columns)
    s = x.T @ x / t
    mu = np.trace(s) / n
    x2 = x ** 2
    beta = ((x2.T @ x2).sum() / t - (s ** 2).sum()) / (n * t)
    delta = ((s - mu * np.eye(n)) ** 2).sum() / n
    shrink = 0.0 if delta == 0 else min(beta, delta) / delta
    cov = shrink * mu * np.eye(n) + (1 - shrink) * s
    return pd.DataFrame(cov * TRADING_DAYS, index=returns.columns, columns=returns.columns)


def portfolio_vol(weights: pd.Series, cov: pd.DataFrame) -> float:
    w = weights.reindex(cov.index).fillna(0.0).to_numpy()
    return float(np.sqrt(max(w @ cov.to_numpy() @ w, 0.0)))
