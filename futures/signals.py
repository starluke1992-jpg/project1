"""B2 的趋势信号。每个函数输入复权价格（DataFrame，每列一个品种），输出同形状的信号，取值在 [−1, 1]。
T 日的信号只用到 T 日及以前的数据（测试里有截断检查）。

  tsmom_ensemble   v1：20/60/120/250 日涨跌方向的平均
  tsmom_12m        对照：过去 250 日涨跌方向
  adaptive_lowlag  挑战者：仿"霞羽均线"思路的自适应低延迟均线，长短周期共振
  repo_signal      对照：仓库里的 BOLL + KAMA + 波动率角度（需要 OHLC）

关于 adaptive_lowlag：作者描述了三步——KAMA 式的自适应平滑、多次递归平滑、相位补偿降低延迟。
前两步可以照做；第三步，因果滤波器不可能在不改变幅度响应的前提下减少延迟，所以这里用常见的
"零延迟"修正 2×MA1 − MA2（类似 DEMA）：延迟变小，代价是转折处会有过冲。这是同一思路的公开近似，
不是作者的原算法。
"""
import numpy as np
import pandas as pd

from trend_signals.indicators import kama


def tsmom_ensemble(prices: pd.DataFrame, lookbacks=(20, 60, 120, 250)) -> pd.DataFrame:
    parts = [np.sign(prices / prices.shift(lb) - 1) for lb in lookbacks]
    return sum(parts) / len(parts)


def tsmom_12m(prices: pd.DataFrame, lookback: int = 250) -> pd.DataFrame:
    return np.sign(prices / prices.shift(lookback) - 1)


def _adaptive_smooth(s: pd.Series, n: int, fast: int, slow: int) -> pd.Series:
    valid = s.dropna()
    if len(valid) <= n:
        return pd.Series(np.nan, index=s.index)
    return kama(valid, n, fast, slow).reindex(s.index)


def lowlag_ma(price: pd.Series, n: int = 10, fast: int = 2, slow: int = 30, comp: float = 1.0) -> pd.Series:
    """自适应低延迟均线：KAMA → 再做一次 KAMA 平滑 → 用 (1+comp)×MA1 − comp×MA2 补偿延迟。"""
    ma1 = _adaptive_smooth(price, n, fast, slow)
    ma2 = _adaptive_smooth(ma1, n, fast, slow)
    return (1 + comp) * ma1 - comp * ma2


def adaptive_lowlag(prices: pd.DataFrame, short_n: int = 10, long_n: int = 40, slope_days: int = 5,
                    comp: float = 1.0) -> pd.DataFrame:
    """长短周期共振：短均线斜率和长均线斜率同向时顺着做，方向不一致时空仓。"""
    out = {}
    for c in prices.columns:
        p = prices[c]
        short = lowlag_ma(p, short_n, 2, 30, comp)
        long = _adaptive_smooth(p, long_n, 2, 60)
        s_dir = np.sign(short - short.shift(slope_days))
        l_dir = np.sign(long - long.shift(slope_days))
        out[c] = s_dir.where(s_dir == l_dir, 0.0)
    return pd.DataFrame(out, index=prices.index)


def repo_signal(ohlc: dict) -> pd.DataFrame:
    """仓库 trend_signals 的持仓方向（1/0/−1）。ohlc: {品种: 含 open/high/low/close 的 DataFrame}。"""
    from trend_signals import generate_signals

    out = {}
    for v, df in ohlc.items():
        d = df[["open", "high", "low", "close"]].dropna()
        if len(d) < 60:
            continue
        out[v] = generate_signals(d)["position"].astype(float).reindex(df.index)
    return pd.DataFrame(out)


SIGNALS = {
    "v1：多周期动量": tsmom_ensemble,
    "对照：12个月动量": tsmom_12m,
    "挑战者：自适应低延迟均线（仿霞羽）": adaptive_lowlag,
}
