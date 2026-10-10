"""组合构建：风险预算 → 品种权重 → IDM → 目标仓位 → 组合风控 → 缓冲区取整。

目标仓位（手）：

    N = F × 资金 × 目标波动率 × W × IDM / (合约乘数 × 日价格波动 × sqrt(244))

F 为合并预测值（平均 |F| ≈ 1），W 为品种权重（资产类别预算在同类可交易品种间平分），
IDM 为品种分散乘数。F = 1 时该品种贡献的年化风险 = 资金 × 目标波动率 × W × IDM。
"""
import numpy as np
import pandas as pd

from .config import TRADING_DAYS
from .instruments import UNIVERSE
from .signals import combined_forecast, price_vol


def build_panel(market: dict, cfg, universe=UNIVERSE) -> dict:
    """把各品种对齐到同一日期轴，算出波动率和预测值。每个值都是 日期 × 品种 的 DataFrame。"""
    codes = [c for c in market if c in universe]
    if not codes:
        raise ValueError("market 里没有 UNIVERSE 中的品种")
    dates = pd.DatetimeIndex(sorted(set().union(*(market[c].index for c in codes))))
    adj, price, roll, vol, fc = {}, {}, {}, {}, {}
    for c in codes:
        d = market[c].sort_index()
        v = price_vol(d["adj"], cfg.vol_span, cfg.vol_long_weight, cfg.vol_long_window)
        f = combined_forecast(d["adj"], v, cfg)
        adj[c], price[c], roll[c], vol[c], fc[c] = d["adj"], d["price"], d["roll"], v, f
    frame = lambda s: pd.DataFrame(s).reindex(dates).ffill(limit=10)  # noqa: E731  休市日沿用前值
    panel = {"adj": frame(adj), "price": frame(price), "vol": frame(vol), "forecast": frame(fc),
             "roll": pd.DataFrame(roll).reindex(dates).fillna(False).astype(bool)}
    panel["diff"] = panel["adj"].diff().fillna(0.0)
    panel["has_roll_data"] = pd.Series({c: bool(market[c]["roll"].any()) for c in codes})
    return panel


def instrument_weights(available: pd.DataFrame, budget: dict, universe=UNIVERSE) -> pd.DataFrame:
    """每天的品种权重：资产类别预算在当天可交易的同类品种之间平分；
    某个类别当天没有可交易品种时，它的预算按比例分给其他类别。"""
    cls = pd.Series({c: universe[c].asset_class for c in available.columns})
    w = pd.DataFrame(0.0, index=available.index, columns=available.columns)
    for k, b in budget.items():
        members = cls.index[cls == k]
        if len(members):
            n = available[members].sum(axis=1)
            w[members] = available[members].astype(float).mul(b / n.replace(0, np.nan), axis=0).fillna(0.0)
    total = w.sum(axis=1)
    return w.div(total.replace(0, np.nan), axis=0).fillna(0.0)


def _corr_estimates(panel: dict, weights: pd.DataFrame, cfg):
    """每月第一个交易日，用之前已经结束的最近 corr_window_weeks 个整周估计两组相关系数：
      子系统收益（预测值 × 风险标准化收益）的相关 → IDM
      风险标准化价格收益的相关 → 组合事前波动率
    更新日和所用数据都只取决于过去，截断数据不会改变之前的结果。"""
    z = panel["diff"] / panel["vol"].shift()
    unit = panel["forecast"].shift() * z
    weekly_z = z.resample("W").sum(min_count=1)
    weekly_u = unit.resample("W").sum(min_count=1)
    dates = panel["diff"].index
    month_starts = dates.to_series().groupby(dates.to_period("M")).min()

    idm = pd.Series(np.nan, index=dates)
    corr = {}
    for m in month_starts:
        wk = weekly_u.index[weekly_u.index < m][-cfg.corr_window_weeks:]
        corr[m] = _clean_corr(weekly_z.loc[wk])
        w = weights.loc[m].to_numpy()
        if len(wk) < cfg.corr_min_weeks or w.sum() == 0:
            idm[m] = 1.0
            continue
        c = np.clip(_clean_corr(weekly_u.loc[wk]), 0, None)
        idm[m] = min(cfg.idm_cap, 1 / np.sqrt(w @ c @ w))
    return idm.ffill().fillna(1.0), pd.Series(corr)


def _clean_corr(df: pd.DataFrame) -> np.ndarray:
    """成对相关系数；样本不足的品种对用其他品种对的平均相关系数代替。"""
    c = df.corr(min_periods=26).to_numpy()
    off = c[~np.eye(len(c), dtype=bool)]
    fill = np.nanmean(off) if np.isfinite(off).any() else 0.0
    c = np.where(np.isnan(c), fill, c)
    np.fill_diagonal(c, 1.0)
    return c


def target_positions(panel: dict, cfg, universe=UNIVERSE) -> dict:
    """风控后的目标仓位（小数手）、平均仓位（F=1 时的仓位，用来定缓冲区宽度）以及中间结果。"""
    codes = panel["forecast"].columns
    mult = pd.Series({c: universe[c].multiplier for c in codes})
    margin = pd.Series({c: universe[c].margin for c in codes})
    available = panel["forecast"].notna() & panel["vol"].notna()
    weights = instrument_weights(available, cfg.budget, universe)
    idm, corr = _corr_estimates(panel, weights, cfg)

    risk_per_lot = panel["vol"] * mult * np.sqrt(TRADING_DAYS)        # 每手年化风险（元）
    budget = weights.mul(idm, axis=0) * cfg.capital * cfg.target_vol   # 各品种 F=1 时的年化风险（元）
    avg_pos = (budget / risk_per_lot).where(available).fillna(0.0)
    raw = (panel["forecast"] * avg_pos).fillna(0.0)

    # 组合风控：事前波动率与保证金占用超限时，所有品种同比例缩仓
    month_of = pd.DatetimeIndex(corr.index).searchsorted(raw.index, side="right") - 1
    daily_risk = (raw * panel["vol"] * mult).fillna(0.0).to_numpy()  # 每品种每日风险（元）
    ex_ante = np.array([np.sqrt(max(r @ corr.iloc[k] @ r, 0.0)) if k >= 0 else 0.0
                        for r, k in zip(daily_risk, month_of)]) * np.sqrt(TRADING_DAYS) / cfg.capital
    vol_cap = cfg.target_vol * cfg.max_vol_mult
    vol_scale = np.where(ex_ante > vol_cap, vol_cap / np.where(ex_ante > 0, ex_ante, 1), 1.0)
    margin_used = (raw.abs() * panel["price"] * mult * margin).sum(axis=1).to_numpy() / cfg.capital
    margin_scale = np.where(margin_used > cfg.max_margin, cfg.max_margin / np.where(margin_used > 0, margin_used, 1), 1.0)
    scale = pd.Series(np.minimum(vol_scale, margin_scale), index=raw.index)

    return {
        "target": raw.mul(scale, axis=0),
        "avg_position": avg_pos,
        "weights": weights,
        "idm": idm,
        "scale": scale,
        "ex_ante_vol": pd.Series(ex_ante, index=raw.index),
        "available": available,
    }


def buffered(target, current, width, integer=True):
    """缓冲区调仓：当前仓位在 [目标 - 宽度, 目标 + 宽度] 内不动，否则只调到缓冲区边界。

    target、current、width 可以是标量或等长数组。width = 平均仓位 × cfg.buffer。
    """
    target, current, width = (np.asarray(x, dtype=float) for x in (target, current, width))
    lo, hi = target - width, target + width
    if integer:
        lo, hi = np.round(lo), np.round(hi)
    return np.where(current < lo, lo, np.where(current > hi, hi, current))


def held_positions(target: pd.DataFrame, avg_position: pd.DataFrame, cfg) -> pd.DataFrame:
    """逐日按缓冲区规则从目标仓位得到实际持仓。"""
    t = target.to_numpy()
    width = (avg_position.abs() * cfg.buffer).to_numpy()
    out = np.zeros_like(t)
    cur = np.zeros(t.shape[1])
    for i in range(len(t)):
        cur = buffered(t[i], cur, width[i], cfg.integer)
        out[i] = cur
    return pd.DataFrame(out, index=target.index, columns=target.columns)
