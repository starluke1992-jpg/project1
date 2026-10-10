"""合成数据：只用来验证整条回测流程能跑通，不代表任何真实市场。

收益 = 缓慢切换的趋势段（上涨/横盘/下跌，每段几十到几百天）+ 随机噪声，同大类资产之间有相关性。
"""
import numpy as np
import pandas as pd

from allocation.data import Panel
from futures.data import FuturesPanel
from futures.sectors import sector_of

A_ASSETS = {  # 资产: (大类, 年化波动)
    "hs300": ("equity", 0.25), "zz500": ("equity", 0.28), "divlv": ("equity", 0.20),
    "hsi": ("equity", 0.22), "spx": ("equity", 0.18), "cgb10": ("bond", 0.04),
    "gold": ("gold", 0.15), "soymeal": ("commodity", 0.20), "metals": ("commodity", 0.20),
    "energy": ("commodity", 0.22),
}
B_VARIETIES = ["IF", "IC", "T", "AU", "AG", "CU", "AL", "ZN", "NI", "RB", "I", "J",
               "M", "Y", "P", "CF", "SR", "TA", "MA", "RU", "FG", "SC"]


def _regime_drift(rng, n, vol, sharpe=0.25, amp=0.8):
    drift = np.empty(n)
    i = 0
    while i < n:
        length = int(rng.integers(40, 260))
        state = rng.choice([-1, 0, 1], p=[0.3, 0.3, 0.4])
        drift[i:i + length] = (sharpe + amp * state) * vol / 252
        i += length
    return drift


def _returns(rng, n, vols, groups, corr_in=0.6):
    """同组资产共享一个因子，组内相关性约 corr_in。"""
    factors = {g: rng.standard_normal(n) for g in sorted(set(groups))}
    out = []
    for vol, g in zip(vols, groups):
        z = np.sqrt(corr_in) * factors[g] + np.sqrt(1 - corr_in) * rng.standard_normal(n)
        out.append(_regime_drift(rng, n, vol) + vol / np.sqrt(252) * z)
    return np.column_stack(out)


def synthetic_panel(seed: int = 7, start: str = "2008-01-01", end: str = "2025-12-31") -> Panel:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    names = list(A_ASSETS)
    r = _returns(rng, len(idx), [A_ASSETS[a][1] for a in names], [A_ASSETS[a][0] for a in names])
    total = pd.DataFrame(r, index=idx, columns=names)
    for a, first in {"divlv": "2010-06-01", "soymeal": "2012-01-01"}.items():  # 模拟晚上市的资产
        total.loc[:first, a] = np.nan
    cash = pd.Series(0.02 / 252, index=idx)
    return Panel(total=total + cash.to_numpy()[:, None], cash=cash,
                 classes={a: A_ASSETS[a][0] for a in names},
                 notes=["合成数据：只用来验证代码，不代表真实市场"])


def synthetic_futures(seed: int = 11, start: str = "2008-01-01", end: str = "2025-12-31") -> FuturesPanel:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    vols = rng.uniform(0.12, 0.35, len(B_VARIETIES))
    vols[B_VARIETIES.index("T")] = 0.04
    r = _returns(rng, len(idx), vols, [sector_of(v) for v in B_VARIETIES], corr_in=0.4)
    excess = pd.DataFrame(r, index=idx, columns=B_VARIETIES)
    for v in B_VARIETIES[::4]:  # 部分品种晚上市
        excess.loc[: idx[int(rng.integers(200, max(201, min(1500, len(idx) // 2))))], v] = np.nan
    prices = (1 + excess.fillna(0)).cumprod().where(excess.notna())
    volumes = pd.DataFrame(1e5, index=idx, columns=B_VARIETIES).where(excess.notna())
    rolls = pd.DataFrame(False, index=idx, columns=B_VARIETIES)
    rolls.iloc[::80] = True
    ohlc = {}
    for v in B_VARIETIES:
        c = prices[v]
        noise = np.abs(rng.standard_normal(len(idx))) * vols[B_VARIETIES.index(v)] / np.sqrt(252) * c
        ohlc[v] = pd.DataFrame({"open": c.shift().fillna(c), "high": c + noise, "low": c - noise, "close": c})
    return FuturesPanel(prices=prices, excess=excess, volumes=volumes, rolls=rolls, ohlc=ohlc)
