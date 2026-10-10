"""复权连续序列的生成、保存和读取。

生成：从 datasource 的逐合约缓存读数据 → continuous.build_all → 每个品种存一份
data/cache/futures_cont/<品种>.csv。方案 A 的商品代理和方案 B 都从这里读。
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from datasource.cache import DEFAULT_ROOT, SeriesCache

from .continuous import build_all

CONT_DIR = "futures_cont"


@dataclass
class FuturesPanel:
    prices: pd.DataFrame      # 复权收盘价
    excess: pd.DataFrame      # 日超额收益
    volumes: pd.DataFrame     # 全部合约合计成交量（手）
    rolls: pd.DataFrame       # 换月标记
    ohlc: dict                # 品种 -> 复权 OHLC


def build_and_save(markets, start, end=None, cache: SeriesCache = None, log=print) -> dict:
    cache = cache or SeriesCache(DEFAULT_ROOT)
    out_dir = cache.root / CONT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    built = {}
    for m in markets:
        contracts = cache.get_futures_contracts(m, start, end)
        if contracts.empty:
            log(f"{m}: 没有逐合约数据")
            continue
        conts = build_all(contracts)
        for variety, cont in conts.items():
            cont.to_csv(out_dir / f"{variety}.csv", date_format="%Y-%m-%d")
        built.update(conts)
        log(f"{m}: 生成 {len(conts)} 个品种")
    return built


def load_continuous(root=DEFAULT_ROOT) -> dict:
    d = Path(root) / CONT_DIR
    if not d.exists():
        return {}
    return {p.stem: pd.read_csv(p, index_col="date", parse_dates=["date"]) for p in sorted(d.glob("*.csv"))}


def to_panel(conts: dict) -> FuturesPanel:
    def col(name, fill=None):
        df = pd.DataFrame({v: c[name] for v, c in conts.items() if name in c.columns})
        return df.sort_index() if fill is None else df.sort_index().fillna(fill)

    ohlc = {v: c[[k for k in ("open", "high", "low", "close") if k in c.columns]] for v, c in conts.items()}
    rolls = col("roll").astype("boolean").fillna(False).astype(bool)
    return FuturesPanel(prices=col("close"), excess=col("ret"), volumes=col("volume") if any(
        "volume" in c.columns for c in conts.values()) else None, rolls=rolls, ohlc=ohlc)
