"""本地 CSV 缓存与增量更新。

缓存目录默认是 data/cache/（不进 git），每条日线序列一个文件：<source>/<symbol>.csv，
旁边的 .meta.json 记录请求过的起始日期和更新时间。

增量更新：从缓存最后一天往前回看 overlap_days 天重新拉取，和缓存的重叠部分对比。
数值一致就把新数据接上；不一致说明上游改写了历史（比如复权口径变了），整条重新拉取。

期货逐合约数据按"交易所/月份"缓存：已经结束的月份不会再变，只拉一次；当月每次都重新拉。
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import akshare_api as api

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "cache"


def with_retry(fn, *args, tries: int = 3, base_delay: float = 2.0, **kwargs):
    """失败后按 2s、4s、8s… 间隔重试，最后一次仍失败就抛出原异常。"""
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(base_delay * 2 ** i)


def _safe_name(symbol: str) -> str:
    return re.sub(r"[^\w\-]", "_", str(symbol))


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    df.index.name = "date"
    return df


class SeriesCache:
    def __init__(self, root=DEFAULT_ROOT, overlap_days: int = 10, tries: int = 3,
                 base_delay: float = 2.0, rtol: float = 1e-6):
        self.root = Path(root)
        self.overlap_days = overlap_days
        self.tries = tries
        self.base_delay = base_delay
        self.rtol = rtol

    def path(self, source: str, symbol: str) -> Path:
        return self.root / source / f"{_safe_name(symbol)}.csv"

    def _meta_path(self, source: str, symbol: str) -> Path:
        return self.path(source, symbol).with_suffix(".meta.json")

    def load(self, source: str, symbol: str):
        p = self.path(source, symbol)
        return _read_csv(p) if p.exists() else None

    def _save(self, source: str, symbol: str, df: pd.DataFrame, start) -> None:
        p = self.path(source, symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, date_format="%Y-%m-%d")
        meta = {"source": source, "symbol": symbol, "start": f"{pd.Timestamp(start):%Y-%m-%d}",
                "updated": f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}", "rows": len(df)}
        self._meta_path(source, symbol).write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    def _requested_start(self, source: str, symbol: str):
        p = self._meta_path(source, symbol)
        return pd.Timestamp(json.loads(p.read_text())["start"]) if p.exists() else None

    def _fetch(self, source, symbol, start, end) -> pd.DataFrame:
        return with_retry(api.fetch, source, symbol, start, end, tries=self.tries, base_delay=self.base_delay)

    def get(self, source: str, symbol: str, start="2010-01-01", end=None, refresh: bool = False) -> pd.DataFrame:
        """取 [start, end] 的日线；有缓存就增量更新，没有就整段拉取。end 默认今天。"""
        start = pd.Timestamp(start)
        end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
        cached = None if refresh else self.load(source, symbol)
        asked = self._requested_start(source, symbol)
        if cached is None or cached.empty or asked is None or start < asked:
            df = self._fetch(source, symbol, start, end)
            self._save(source, symbol, df, start)
            return df.loc[start:end]
        if end <= cached.index[-1]:  # 缓存已经覆盖，不用联网
            return cached.loc[start:end]

        new_from = cached.index[-1] - pd.Timedelta(days=self.overlap_days)
        new = self._fetch(source, symbol, new_from, end)
        if not self._consistent(cached, new):
            df = self._fetch(source, symbol, min(start, asked), end)
        elif new.empty:
            df = cached
        else:
            df = pd.concat([cached[cached.index < new.index[0]], new])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        self._save(source, symbol, df, min(start, asked))
        return df.loc[start:end]

    def _consistent(self, old: pd.DataFrame, new: pd.DataFrame) -> bool:
        common = old.index.intersection(new.index)
        cols = [c for c in old.columns if c in new.columns]
        if len(common) == 0 or not cols:
            return True
        a = old.loc[common, cols].to_numpy(dtype=float)
        b = new.loc[common, cols].to_numpy(dtype=float)
        return bool(np.allclose(a, b, rtol=self.rtol, atol=0, equal_nan=True))

    def get_futures_contracts(self, market: str, start, end=None, refresh: bool = False) -> pd.DataFrame:
        """交易所逐合约日线，按月缓存。已结束的月份读缓存，当月和缺失的月份去拉取。"""
        start = pd.Timestamp(start)
        end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
        today = pd.Timestamp.today().normalize()
        frames = []
        for month in pd.period_range(start, end, freq="M"):
            p = self.root / "futures_contracts" / market.upper() / f"{month}.csv"
            finished = month.end_time.normalize() < today
            if p.exists() and finished and not refresh:
                df = pd.read_csv(p, parse_dates=["date"])
            else:
                df = with_retry(api.futures_contracts_daily, market, month.start_time,
                                min(month.end_time.normalize(), today),
                                tries=self.tries, base_delay=self.base_delay)
                if finished:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(p, index=False, date_format="%Y-%m-%d")
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(columns=api._CONTRACT_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        return out[(out["date"] >= start) & (out["date"] <= end)].reset_index(drop=True)
