"""行情数据：每个品种一个 DataFrame，日期作索引，三列：

  price  当前主力合约的真实价格，用来算名义价值、保证金和手续费
  adj    换月调整后的连续价格（后复权，价差法），用来算信号、波动率和盈亏
  roll   当天收盘是否换月

换月处理决定回测是否可信：未复权的主力连续在换月日会跳空，跳空会被当成真实盈亏，
也会污染趋势信号。按可靠程度，数据来源依次是：
  1. 逐合约日线（含持仓量）→ build_continuous() 自己拼接，换月当天用新旧两个合约的价格，误差为零；
  2. 主力连续 + 主力合约代码列（contract）→ 换月当天的价差置零，丢掉一天的真实涨跌；
  3. 数据商提供的复权连续价格（adj_close 列）→ 直接使用；
  4. 只有未复权的 close → 无法处理换月，只能做演示。
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .instruments import UNIVERSE

_ALIASES = {
    "date": "date", "datetime": "date", "trade_date": "date", "日期": "date", "时间": "date",
    "close": "close", "收盘": "close", "收盘价": "close",
    "adj_close": "adj_close", "复权收盘": "adj_close", "复权收盘价": "adj_close",
    "contract": "contract", "symbol": "contract", "合约": "contract", "主力合约": "contract",
    "open_interest": "open_interest", "oi": "open_interest", "持仓量": "open_interest",
    "volume": "volume", "成交量": "volume",
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: _ALIASES.get(str(c).strip().lower(), c) for c in df.columns})
    missing = {"date", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"缺少列: {sorted(missing)}，现有列: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"].astype(str))
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """主力连续行情 → price / adj / roll 三列。需要 date、close 列，可选 contract、adj_close 列。"""
    df = _normalize(df.copy())
    df = df.drop_duplicates("date", keep="last").set_index("date").sort_index()
    price = df["close"].astype(float)
    roll = pd.Series(False, index=df.index)
    if "adj_close" in df.columns:
        adj = df["adj_close"].astype(float)
    elif "contract" in df.columns:
        contract = df["contract"].astype(str)
        roll = (contract != contract.shift(-1)) & contract.shift(-1).notna()
        diff = price.diff()
        diff[roll.shift(fill_value=False)] = 0.0   # 换月后第一天的价差含跳空，置零
        adj = diff.fillna(0.0).cumsum()
        adj += price.iloc[-1] - adj.iloc[-1]
    else:
        adj = price
    return pd.DataFrame({"price": price, "adj": adj, "roll": roll})


def build_continuous(contracts: pd.DataFrame, by: str = "open_interest") -> pd.DataFrame:
    """用逐合约日线拼接主力连续（后复权，价差法）。

    contracts 需要 date、contract、close 列，以及 by 指定的列（默认持仓量）。
    主力 = 当天持仓量最大的合约；只向前换月（换出的合约不再换回），
    在新合约持仓量超过当前合约的那天收盘换月。换月当天的盈亏仍按旧合约计算，
    所以调整后的价格序列与“实际持有并按时移仓”的盈亏完全一致。
    """
    df = _normalize(contracts.copy())
    close = df.pivot_table(index="date", columns="contract", values="close", aggfunc="last").sort_index()
    size = df.pivot_table(index="date", columns="contract", values=by, aggfunc="last").reindex(close.index)

    current, retired = None, set()
    held, diffs, rolls = [], [], []
    prev = None
    for date in close.index:
        px = close.loc[date]
        avail = size.loc[date].dropna()
        avail = avail[[c for c in avail.index if c not in retired and pd.notna(px.get(c))]]
        diff = 0.0
        if current is not None and prev is not None:
            if pd.notna(px.get(current)) and pd.notna(prev.get(current)):
                diff = px[current] - prev[current]
        rolled = False
        if len(avail):
            best = avail.idxmax()
            gone = current is None or pd.isna(px.get(current))
            if current is not None and best != current and (gone or avail[best] > avail.get(current, -np.inf)):
                retired.add(current)
                current, rolled = best, True
            elif current is None:
                current = best
        held.append(current)
        diffs.append(diff)
        rolls.append(rolled)
        prev = px

    idx = close.index
    price = pd.Series([close.at[d, c] for d, c in zip(idx, held)], index=idx)
    adj = pd.Series(diffs, index=idx).cumsum()
    adj += price.iloc[-1] - adj.iloc[-1]
    return pd.DataFrame({"price": price, "adj": adj, "roll": rolls, "contract": held}, index=idx)


def _read_csv(path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def load_dir(path, codes=None) -> dict:
    """读取目录下的 <品种代码>.csv（如 IF.csv、RB.csv），返回 {代码: DataFrame}。

    只读取 UNIVERSE 里有的品种；codes 可以指定子集。
    """
    path = Path(path)
    out = {}
    for code in codes or UNIVERSE:
        f = path / f"{code}.csv"
        if f.exists():
            out[code] = prepare(_read_csv(f))
    if not out:
        raise FileNotFoundError(f"{path} 下没有找到品种数据文件（如 IF.csv）")
    return out


def fetch_akshare(codes=None, start="20100101", end="22220101", out_dir=None) -> dict:
    """用 akshare 拉新浪主力连续日线（未复权，没有合约代码列），只适合演示和粗略观察。"""
    from trend_signals.data import fetch_akshare as fetch_one

    out = {}
    for code in codes or UNIVERSE:
        df = fetch_one(f"{code}0", start, end)
        raw = pd.DataFrame({"date": df.index, "close": df["close"].to_numpy()})
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            raw.to_csv(Path(out_dir) / f"{code}.csv", index=False)
        out[code] = prepare(raw)
    return out


# 模拟行情参数（只用于演示和测试，不代表真实市场）：各类资产年化波动率、
# 危机期间的累计漂移（股市下跌、国债和贵金属走强、工业品下跌）、平时的年化漂移，以及各品种的大致价格水平
_SIM_VOL = {"equity": 0.22, "bond": 0.035, "precious": 0.15, "base_metal": 0.18,
            "ferrous": 0.25, "energy_chem": 0.22, "agri": 0.16}
_SIM_CRISIS_DRIFT = {"equity": -0.25, "bond": 0.02, "precious": 0.06, "base_metal": -0.15,
                     "ferrous": -0.15, "energy_chem": -0.18, "agri": -0.05}
_SIM_NORMAL_DRIFT = {"equity": 0.08}
_SIM_LEVEL = {
    "IF": 4000, "IH": 2700, "IC": 6000, "IM": 6500, "TS": 102, "TF": 106, "T": 108, "TL": 115,
    "AU": 600, "AG": 7500, "CU": 75000, "AL": 20000, "ZN": 23000, "NI": 125000,
    "RB": 3300, "HC": 3400, "I": 800, "J": 1800, "JM": 1200,
    "SC": 550, "FU": 3000, "RU": 15000, "TA": 5000, "MA": 2500, "PP": 7200, "EG": 4400, "SA": 1500, "FG": 1200,
    "M": 3000, "Y": 8000, "P": 8500, "C": 2300, "SR": 5800, "CF": 14000, "RM": 2500, "LH": 14000, "AP": 7500,
}


def synthetic_market(years: int = 12, crises=((750, 70), (1600, 120), (2400, 45)), codes=None,
                     seed: int = 7, days_per_year: int = 244) -> dict:
    """模拟一个多资产期货市场，用于测试和演示（不是真实行情）。

    平时是随机游走（含全市场、资产类别两层共同因子，股指有小幅正漂移）；crises 里每个
    (起始日, 天数) 是一段危机：股指持续下跌、国债和贵金属上涨、工业品下跌，波动率放大 1.6 倍。
    """
    rng = np.random.default_rng(seed)
    codes = list(codes or UNIVERSE)
    n = years * days_per_year
    idx = pd.bdate_range("2012-01-04", periods=n)
    classes = sorted({UNIVERSE[c].asset_class for c in codes})
    market_f = rng.standard_normal(n)
    class_f = {k: rng.standard_normal(n) for k in classes}
    crisis = np.zeros(n, dtype=bool)
    length = np.ones(n)
    for start, days in crises:
        crisis[start:start + days] = True
        length[start:start + days] = days
    out = {}
    for code in codes:
        k = UNIVERSE[code].asset_class
        sd = _SIM_VOL[k] / np.sqrt(days_per_year)
        beta = {"equity": 0.5, "bond": -0.2, "precious": -0.1}.get(k, 0.3)
        z = beta * market_f + 0.6 * class_f[k] + np.sqrt(1 - beta**2 - 0.36) * rng.standard_normal(n)
        r = np.where(crisis & (k != "bond"), sd * z * 1.6, sd * z)
        drift = np.where(crisis, np.log1p(_SIM_CRISIS_DRIFT[k]) / length,
                         _SIM_NORMAL_DRIFT.get(k, 0.0) / days_per_year)
        price = _SIM_LEVEL.get(code, 1000) * np.exp(np.cumsum(r + drift))
        out[code] = pd.DataFrame({"price": price, "adj": price, "roll": False}, index=idx)
    return out
