"""波动率估计与趋势预测值。

预测值的尺度统一为：长期平均 |预测值| ≈ 1，上下限 ±2。预测值 1 对应“平均仓位”，
即该品种按风险预算应占的风险；2 是最强趋势时的上限。

所有计算都基于价差（而不是收益率），所以对后复权（价差法）的连续价格成立，
价格整体平移不改变任何信号。
"""
import numpy as np
import pandas as pd

# EWMAC 缩放系数：Carver (2015, Systematic Trading) 用真实期货数据校准的数值（原书尺度为 10，这里除以 10）。
# 在零漂移随机游走上校准得到的系数比它高约 25%（0.65/0.47/0.33/0.23），差别来自真实市场里的趋势。
EWMAC_SCALAR = {(8, 32): 0.53, (16, 64): 0.375, (32, 128): 0.265, (64, 256): 0.187}
# 通道突破的原始值在 ±0.5 之间；随机游走上校准约为 3.7~4.0，按与 EWMAC 相同的比例折算为 3.1
BREAKOUT_SCALAR = 3.1


def price_vol(adj: pd.Series, span: int = 35, long_weight: float = 0.3, long_window: int = 2520) -> pd.Series:
    """每日价格变动的标准差（价格单位）。

    短期用 EWMA(span) 标准差，再和它自己过去 long_window 天的均值按 long_weight 混合：
    低波动时期不会因为波动率估计过低而把仓位放得过大，危机来临时也能较快反应。
    """
    short = adj.diff().ewm(span=span, min_periods=span).std()
    long = short.rolling(long_window, min_periods=1).mean()
    vol = (1 - long_weight) * short + long_weight * long
    return vol.where(vol > 0)


def ewmac(adj: pd.Series, vol: pd.Series, fast: int, slow: int) -> pd.Series:
    """均线交叉：(EMA_fast - EMA_slow) / 日波动，乘缩放系数。"""
    raw = adj.ewm(span=fast, adjust=False).mean() - adj.ewm(span=slow, adjust=False).mean()
    # 表外的周期（慢线 = 4 × 快线）按同一规律外推：系数与 sqrt(快线) 成反比，表内数值正好满足这个关系
    scalar = EWMAC_SCALAR.get((fast, slow), 0.53 * np.sqrt(8 / fast))
    return raw / vol * scalar


def breakout(adj: pd.Series, window: int) -> pd.Series:
    """通道突破：价格在过去 window 天高低点区间里的位置（-0.5 ~ +0.5），再做 window/4 的平滑。"""
    hi = adj.rolling(window, min_periods=window).max()
    lo = adj.rolling(window, min_periods=window).min()
    raw = (adj - (hi + lo) / 2) / (hi - lo).where(hi > lo)
    return raw.ewm(span=max(window // 4, 1), adjust=False).mean() * BREAKOUT_SCALAR


def rule_forecasts(adj: pd.Series, vol: pd.Series, cfg) -> pd.DataFrame:
    """每条规则的预测值（已截断在 ±cap）。列名如 ewmac16_64、breakout80。"""
    cols = {}
    for fast, slow in cfg.ewmac_speeds:
        cols[f"ewmac{fast}_{slow}"] = ewmac(adj, vol, fast, slow)
    for w in cfg.breakout_windows:
        cols[f"breakout{w}"] = breakout(adj, w)
    return pd.DataFrame(cols).clip(-cfg.forecast_cap, cfg.forecast_cap)


def rule_weights(cfg) -> pd.Series:
    """EWMAC 合计 cfg.ewmac_weight，通道突破合计其余部分；同一族内各周期等权。"""
    w = {}
    for fast, slow in cfg.ewmac_speeds:
        w[f"ewmac{fast}_{slow}"] = cfg.ewmac_weight / len(cfg.ewmac_speeds)
    for win in cfg.breakout_windows:
        w[f"breakout{win}"] = (1 - cfg.ewmac_weight) / len(cfg.breakout_windows)
    return pd.Series(w)


def combined_forecast(adj: pd.Series, vol: pd.Series, cfg) -> pd.Series:
    """各规则加权平均 × 预测分散乘数（FDM），再截断在 ±cap。

    规则之间不完全相关，加权平均后的平均绝对值会变小，FDM 把它拉回到 1 左右；
    默认的 1.2 是用默认规则组合的相关系数算出的 1/sqrt(w'Cw)。
    上市不久的品种，长周期突破规则还没有值，只用已有值的规则（权重按比例放大）。
    """
    f = rule_forecasts(adj, vol, cfg)
    w = rule_weights(cfg)
    f = f[w.index]
    combined = (f * w).sum(axis=1) / f.notna().mul(w).sum(axis=1)
    combined = combined.where(vol.notna())
    return (combined * cfg.fdm).clip(-cfg.forecast_cap, cfg.forecast_cap)
