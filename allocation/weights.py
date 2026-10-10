"""权重计算：波动率倒数、两层风险预算、上限、波动率缩放。"""
import numpy as np
import pandas as pd

from .risk import TRADING_DAYS, ann_vol, ledoit_wolf, portfolio_vol


def inverse_vol(vols: pd.Series) -> pd.Series:
    inv = 1.0 / vols
    return inv / inv.sum()


def two_level_risk_parity(hist: pd.DataFrame, classes: dict, budgets: dict = None,
                          vol_floor: float = 0.02) -> pd.Series:
    """两层风险预算：大类内部按波动率倒数，大类之间按"风险预算 / 大类波动"分配，合计 100%。

    hist：参与配置的资产在估计窗口内的日收益（无缺失）。
    budgets：各大类的风险预算，默认等分；某类当期没有资产时，预算在其余大类间重新归一。
    """
    groups = {}
    for a in hist.columns:
        groups.setdefault(classes[a], []).append(a)
    if budgets is None:
        budgets = {c: 1.0 for c in groups}
    weights, class_w = {}, {}
    for c, assets in groups.items():
        inner = inverse_vol(ann_vol(hist[assets], vol_floor))
        sub = hist[assets] @ inner
        sigma_c = max(float(sub.std(ddof=1) * np.sqrt(TRADING_DAYS)), vol_floor)
        class_w[c] = budgets.get(c, 0.0) / sigma_c
        for a in assets:
            weights[a] = inner[a]
    total = sum(class_w.values())
    if total == 0:
        return pd.Series(0.0, index=hist.columns)
    return pd.Series({a: weights[a] * class_w[classes[a]] / total for a in hist.columns})


def cap_weights(w: pd.Series, cap: float) -> pd.Series:
    """单个资产不超过 cap，超出部分不再分配（留作现金）。"""
    return w.clip(upper=cap, lower=-cap)


def scale_to_vol(w: pd.Series, hist: pd.DataFrame, target: float, max_scale: float = np.inf,
                 max_gross: float = np.inf, only_down: bool = False) -> tuple:
    """按 Ledoit-Wolf 协方差估计组合波动，把权重缩放到目标波动。

    only_down=True 时只缩小不放大（不加杠杆的方案 A 用）；max_gross 是名义杠杆上限（Σ|w|）。
    返回 (缩放后的权重, 缩放后的事前波动)。事前波动和事后实际波动放在一起比较，能看出估计是否靠谱。
    """
    active = w[w != 0]
    if active.empty:
        return w, 0.0
    cov = ledoit_wolf(hist[active.index])
    vol = portfolio_vol(active, cov)
    if vol <= 0:
        return w, vol
    k = min(target / vol, max_scale)
    if only_down:
        k = min(k, 1.0)
    gross = active.abs().sum()
    if gross * k > max_gross:
        k = max_gross / gross
    return w * k, vol * k
