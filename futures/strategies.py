"""方案 B 的目标权重函数。

B2 趋势 CTA（docs/plan_A_B.md 3.4 节）：
  原始权重 = 信号 / 波动率 → 单板块风险不超过 sector_cap → 按 Ledoit-Wolf 协方差缩放到目标波动，
  名义杠杆不超过 max_gross。
B1 杠杆风险平价：复用 allocation.strategies.risk_parity，允许放大，目标波动 8%，名义杠杆 ≤ 2.5 倍。
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from allocation.risk import ann_vol
from allocation.strategies import RPConfig, risk_parity
from allocation.weights import scale_to_vol


@dataclass
class B2Config:
    vol_window: int = 60
    cov_window: int = 120
    vol_floor: float = 0.05
    target_vol: float = 0.10
    max_gross: float = 3.0
    sector_cap: float = 0.25
    min_days: int = 250
    min_volume: float = 20000

    def to_dict(self):
        return asdict(self)


def cap_sector_risk(raw: pd.Series, vols: pd.Series, sectors: dict, cap: float) -> pd.Series:
    """按单品种独立风险（|权重|×波动）计算板块占比，超过 cap 的板块缩小到正好 cap。

    设超限板块有 k 个，其余板块风险合计 U，缩小后总风险 T 满足 T = U + k×cap×T，即 T = U / (1 − k×cap)。
    缩小后其余板块的占比会上升，可能又有新的板块超限，所以重复几轮。
    板块太少、cap 无法满足（k×cap ≥ 1）时，各板块风险拉平。
    """
    w = raw.copy()
    sec_of = pd.Series({a: sectors.get(a, "其他") for a in w.index})
    for _ in range(10):
        risk = (w.abs() * vols).groupby(sec_of).sum()
        total = risk.sum()
        if total == 0:
            break
        over = risk[risk / total > cap + 1e-9]
        if over.empty:
            break
        under = total - over.sum()
        if len(over) * cap >= 1 or under == 0:
            target = total / len(risk)
            for sec, rs in risk.items():
                w[sec_of == sec] *= target / rs
            break
        new_total = under / (1 - len(over) * cap)
        for sec, rs in over.items():
            w[sec_of == sec] *= cap * new_total / rs
    return w


def b2_strategy(signal: pd.DataFrame, eligible: pd.DataFrame, sectors: dict, cfg: B2Config = B2Config()):
    def fn(ctx):
        d = ctx.date
        if d not in signal.index:
            return pd.Series(dtype=float), np.nan
        s = signal.loc[d]
        ok = eligible.loc[d] if d in eligible.index else pd.Series(False, index=signal.columns)
        hist_all = ctx.excess.iloc[-cfg.cov_window:]
        full = hist_all.notna().sum() >= int(cfg.cov_window * 0.9)
        assets = [a for a in ctx.excess.columns
                  if bool(ok.get(a, False)) and full.get(a, False) and pd.notna(s.get(a)) and s.get(a) != 0]
        if not assets:
            return pd.Series(dtype=float), 0.0
        hist = hist_all[assets].fillna(0.0)
        vols = ann_vol(hist.iloc[-cfg.vol_window:], cfg.vol_floor)
        raw = s[assets].astype(float) / vols
        raw = cap_sector_risk(raw, vols, sectors, cfg.sector_cap)
        return scale_to_vol(raw, hist, cfg.target_vol, max_gross=cfg.max_gross)
    return fn


def b1_configs() -> dict:
    """B1 的两个预先登记版本：带趋势过滤（v1）和不带（对照，因为 B2 已经提供趋势收益）。"""
    common = dict(target_vol=0.08, only_down=False, max_gross=2.5, asset_cap=1.0)
    return {
        "B1 v1（带趋势过滤）": RPConfig(trend="ma_mom", **common),
        "B1 对照（不带趋势过滤）": RPConfig(trend=None, **common),
    }


def b1_strategies(classes: dict) -> dict:
    return {name: (risk_parity(classes, cfg), cfg.to_dict()) for name, cfg in b1_configs().items()}
