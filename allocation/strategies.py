"""方案 A（以及 B1）的目标权重函数。

基准：
  equal_weight     有数据的风险资产等权
  sixty_forty      沪深300 60% + 10年国债 40%
  risk_parity      两层风险平价（不加趋势过滤）= "静态风险平价"
方案 A v1：risk_parity + ma_mom 趋势过滤
挑战者：risk_parity + 负向趋势因子（作者思路的近似实现）

所有函数都只看 Context 里截至信号日的数据。
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .filters import ma_mom_exposure, negative_trend_exposure
from .weights import cap_weights, scale_to_vol, two_level_risk_parity


def _eligible(ctx, min_history: int) -> list:
    """信号日前至少有 min_history 个有效收益的资产。"""
    counts = ctx.total.iloc[-min_history:].notna().sum()
    return [a for a in ctx.total.columns if counts[a] >= min_history]


def equal_weight(min_history: int = 60):
    def fn(ctx):
        assets = _eligible(ctx, min_history)
        return pd.Series(1.0 / len(assets), index=assets) if assets else pd.Series(dtype=float)
    return fn


def sixty_forty(equity: str = "hs300", bond: str = "cgb10"):
    def fn(ctx):
        return pd.Series({equity: 0.6, bond: 0.4})
    return fn


@dataclass
class RPConfig:
    vol_window: int = 60          # 单资产波动率窗口
    cov_window: int = 120         # 组合波动估计窗口（Ledoit-Wolf）
    vol_floor: float = 0.02       # 波动率下限
    budgets: dict = None          # 大类风险预算，None 为等分
    target_vol: float = 0.06      # 组合波动目标 / 上限
    only_down: bool = True        # True：只缩不放（不加杠杆）
    max_gross: float = 1.0        # 名义杠杆上限
    asset_cap: float = 0.5        # 单资产上限
    min_history: int = 252        # 资产至少有这么多天数据才参与
    trend: str = None             # None / "ma_mom" / "neg_trend"
    ma: int = 200
    mom: int = 252
    neg_threshold: float = -1.0

    def to_dict(self):
        return asdict(self)


def risk_parity(classes: dict, cfg: RPConfig = RPConfig()):
    """两层风险平价 → 趋势过滤（可选）→ 单资产上限 → 波动率缩放。"""
    def fn(ctx):
        assets = _eligible(ctx, max(cfg.min_history, cfg.cov_window))
        if not assets:
            return pd.Series(dtype=float), np.nan
        hist = ctx.total[assets].iloc[-cfg.vol_window:].fillna(0.0)
        w = two_level_risk_parity(hist, {a: classes[a] for a in assets}, cfg.budgets, cfg.vol_floor)
        if cfg.trend:
            prices = ctx.prices()[assets]
            if cfg.trend == "ma_mom":
                e = ma_mom_exposure(prices, ctx.cash_index(), cfg.ma, cfg.mom)
            elif cfg.trend == "neg_trend":
                e = negative_trend_exposure(prices, ctx.extra.get("volume"), cfg.neg_threshold)
            else:
                raise ValueError(f"未知的趋势过滤 {cfg.trend}")
            w = w * e.reindex(w.index).fillna(0.0)
        w = cap_weights(w, cfg.asset_cap)
        cov_hist = ctx.total[assets].iloc[-cfg.cov_window:].fillna(0.0)
        return scale_to_vol(w, cov_hist, cfg.target_vol, max_gross=cfg.max_gross, only_down=cfg.only_down)
    return fn


def plan_a_strategies(classes: dict, assets: list) -> dict:
    """方案 A 要比较的全部策略：名称 -> (目标权重函数, 参数)。"""
    base = RPConfig()
    v1 = RPConfig(trend="ma_mom")
    neg = RPConfig(trend="neg_trend")
    out = {
        "等权": (equal_weight(), {"min_history": 60}),
        "静态风险平价": (risk_parity(classes, base), base.to_dict()),
        "方案A v1（均线+动量过滤）": (risk_parity(classes, v1), v1.to_dict()),
        "挑战者：负向趋势因子（近似）": (risk_parity(classes, neg), neg.to_dict()),
    }
    if {"hs300", "cgb10"} <= set(assets):
        out["股六债四"] = (sixty_forty(), {"equity": "hs300", "bond": "cgb10"})
    return out
