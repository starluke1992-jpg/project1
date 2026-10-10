"""用逐合约日线构造复权连续序列（超额收益）。

换月规则（不用未来数据）：
  - T 日收盘后决定第二天持有哪个合约，只看 T 日及以前的数据；
  - 可选合约：交割月的前一个月 15 日之前的合约（个人账户一般不能把仓位带进交割月）；
  - 当前合约不再可选，或者有交割月更晚的合约持仓量超过当前合约时，换到持仓量最大的那个更晚的合约；
    只往后换，不会换回更近的合约，避免两个合约持仓量接近时来回切换。
收益：持有合约当天收盘价 / 前一天收盘价 − 1。换月当天的收益算在旧合约上，新合约从第二天开始，
所以不存在主力连续那种换月跳空。期货不需要出资，这个收益就是超额收益。
同时输出复权后的 OHLC（按持有合约的价格和累计收益比例换算），供需要 K 线的信号使用。
"""
import re

import numpy as np
import pandas as pd

ROLL_DAY = 15  # 交割月前一个月的这一天之前完成换月


def delivery_month(symbol: str, ref: pd.Timestamp) -> pd.Timestamp:
    """从合约代码解析交割月。四位 YYMM（如 CU2402）直接读；郑商所三位 YMM（如 TA405）按参考日期推断年代。"""
    digits = re.search(r"(\d{3,4})$", symbol)
    if not digits:
        return pd.NaT
    d = digits.group(1)
    if len(d) == 4:
        return pd.Timestamp(year=2000 + int(d[:2]), month=int(d[2:]), day=1)
    year_digit, month = int(d[0]), int(d[1:])
    if not 1 <= month <= 12:
        return pd.NaT
    base = ref.year - ref.year % 10 + year_digit
    for y in (base - 10, base, base + 10):
        dm = pd.Timestamp(year=y, month=month, day=1)
        if ref - pd.DateOffset(years=2) <= dm <= ref + pd.DateOffset(years=3):
            return dm
    return pd.NaT


def last_hold_date(dm: pd.Timestamp) -> pd.Timestamp:
    """最晚可以持有到哪天：交割月前一个月的 ROLL_DAY 日（不含）。"""
    return dm - pd.DateOffset(months=1) + pd.Timedelta(days=ROLL_DAY - 1)


def build_continuous(contracts: pd.DataFrame) -> pd.DataFrame:
    """contracts：一个品种的逐合约长表（date, symbol, open, high, low, close, volume, open_interest...）。

    返回以日期为索引的 DataFrame：ret（当天超额收益）、contract（当天持有的合约）、roll（当天收盘是否换月）、
    open/high/low/close（复权价格）、volume、open_interest（全部合约合计）。
    """
    df = contracts.dropna(subset=["close"]).copy()
    df = df[df["close"] > 0]
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    close = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").reindex(dates)
    oi = df.pivot_table(index="date", columns="symbol", values="open_interest", aggfunc="last").reindex(dates)
    ohlc = {c: df.pivot_table(index="date", columns="symbol", values=c, aggfunc="last").reindex(dates)
            for c in ("open", "high", "low") if c in df.columns}
    vol_total = df.groupby("date")["volume"].sum().reindex(dates) if "volume" in df.columns else None
    oi_total = oi.sum(axis=1, min_count=1)

    symbols = list(close.columns)
    col = {s: j for j, s in enumerate(symbols)}
    c_arr, oi_arr = close.to_numpy(dtype=float), oi.reindex(columns=symbols).to_numpy(dtype=float)
    dm_cache = {}

    def dm_of(s, d):
        key = (s, d.year // 5)  # 郑商所三位代码的年代推断只和参考年份有关，按 5 年分桶缓存
        if key not in dm_cache:
            dm_cache[key] = delivery_month(s, d)
        return dm_cache[key]

    held, rets, rolls = [], [], []
    current = None
    for i, d in enumerate(dates):
        # 当天收益：昨天收盘后持有的合约
        if current is not None and i > 0:
            c0, c1 = c_arr[i - 1, col[current]], c_arr[i, col[current]]
            rets.append(c1 / c0 - 1 if np.isfinite(c0) and np.isfinite(c1) else 0.0)
        else:
            rets.append(np.nan)
        held.append(current)

        # 收盘后决定明天持有哪个合约（只用今天及以前的数据）
        live = [s for s in symbols if np.isfinite(oi_arr[i, col[s]]) and np.isfinite(c_arr[i, col[s]])]
        dms = {s: dm_of(s, d) for s in live}
        nxt = d + pd.Timedelta(days=1)
        ok = {s: dm for s, dm in dms.items() if pd.notna(dm) and nxt <= last_hold_date(dm)}
        new = current
        if ok:
            cur_dm = dms.get(current) if current is not None else None
            later = {s: dm for s, dm in ok.items() if cur_dm is None or dm > cur_dm}
            oi_of = lambda s: oi_arr[i, col[s]]  # noqa: E731
            if current not in ok:
                new = max(later or ok, key=oi_of)
            elif later:
                best = max(later, key=oi_of)
                if oi_of(best) > oi_of(current):
                    new = best
        rolls.append(current is not None and new != current)
        current = new

    out = pd.DataFrame({"ret": rets, "contract": held, "roll": rolls}, index=dates)
    out.index.name = "date"
    # roll 标记的是"当天收盘换月"，换月成本在当天产生
    nav = (1 + out["ret"].fillna(0)).cumprod()
    pick = lambda arr: np.array([arr[i, col[h]] if h else np.nan for i, h in enumerate(held)])  # noqa: E731
    factor = nav / pick(c_arr)
    out["close"] = nav.where(out["contract"].notna())
    for c, tbl in ohlc.items():
        out[c] = pick(tbl.reindex(columns=symbols).to_numpy(dtype=float)) * factor
    if vol_total is not None:
        out["volume"] = vol_total
    out["open_interest"] = oi_total
    return out.dropna(subset=["contract"])


def build_all(contracts: pd.DataFrame) -> dict:
    """按品种（variety 列）分别构造，返回 {品种: 连续序列}。"""
    out = {}
    for variety, grp in contracts.groupby("variety"):
        if len(grp["date"].unique()) >= 2:
            out[str(variety).upper()] = build_continuous(grp)
    return out
