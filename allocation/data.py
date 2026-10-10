"""把缓存里的原始序列整理成回测用的日收益面板（方案 A 用，B1 也用）。

每个资产一列，按 A 股交易日对齐：
  1. ETF 上市后用 ETF 后复权价格算收益；
  2. 上市前用代理序列补齐，并按 ETF 费率每天扣减；
  3. 境外指数换算成人民币，并且晚一天对齐：美股、港股收盘晚于 A 股，A 股收盘时还看不到当天的收盘价；
  4. 国债 ETF 上市前，用 10 年国债收益率合成债券收益；
  5. 商品 ETF 上市前，优先用自建的复权连续期货（futures/ 生成），没有时才用未复权的主力连续；
  6. 现金：货币 ETF 上市后用它，之前按 7 天回购利率计息。
超额收益 = 资产收益 − 现金收益。

只读本地缓存，不联网，保证同一份缓存每次回测结果一样。缺数据的资产会被跳过并记在 notes 里。
"""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from datasource.cache import DEFAULT_ROOT, SeriesCache

DEFAULT_UNIVERSE = Path(__file__).resolve().parents[1] / "config" / "universe_A.csv"
LAGGED_SOURCES = {"us_index", "hk_index"}           # 收盘晚于 A 股，晚一天对齐
CALENDAR_SOURCES = {"etf", "csindex", "index_cn"}   # 用来确定 A 股交易日历
BOND_DURATION, BOND_CONVEXITY = 8.5, 85.0           # 10 年期国债的修正久期和凸性（近似）
TRADING_DAYS = 252


@dataclass
class Panel:
    total: pd.DataFrame                 # 各风险资产的日总收益（人民币）
    cash: pd.Series                     # 现金日收益
    classes: dict                       # 资产 -> 大类（equity/bond/gold/commodity）
    origin: pd.DataFrame = None         # 每天每个资产的数据来源：etf / proxy
    notes: list = field(default_factory=list)

    @property
    def excess(self) -> pd.DataFrame:
        return self.total.sub(self.cash, axis=0)

    @property
    def assets(self) -> list:
        return list(self.total.columns)


def load_universe(path=DEFAULT_UNIVERSE) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def bond_returns_from_yield(y: pd.Series, duration=BOND_DURATION, convexity=BOND_CONVEXITY) -> pd.Series:
    """y 是到期收益率（%）。日收益 ≈ 持有期利息 − 久期 × 收益率变化 + ½ × 凸性 × 变化²。"""
    y = y / 100
    dy = y.diff()
    days = y.index.to_series().diff().dt.days
    return y.shift() * days / 365 - duration * dy + 0.5 * convexity * dy ** 2


def rate_to_returns(rate: pd.Series) -> pd.Series:
    """年化利率（%）按自然日计息：前一天的利率 × 间隔天数 / 365。"""
    r = rate / 100
    days = r.index.to_series().diff().dt.days
    return r.shift() * days / 365


def _align(level: pd.Series, calendar: pd.DatetimeIndex, lag: bool = False) -> pd.Series:
    """把一条水平序列（价格、汇率、利率）对齐到交易日历：取每个交易日当天或之前最近的值。

    lag=True 时整体往后挪一天，即交易日 d 只能用到 d 之前的收盘价。
    """
    s = level.dropna()
    if lag:
        s.index = s.index + pd.Timedelta(days=1)
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(calendar.union(s.index)).ffill().reindex(calendar)


class PanelBuilder:
    def __init__(self, universe: pd.DataFrame = None, cache: SeriesCache = None):
        self.universe = load_universe() if universe is None else universe
        self.cache = cache if cache is not None else SeriesCache(DEFAULT_ROOT)
        self.notes = []

    def _load(self, source: str, symbol: str):
        df = self.cache.load(source, symbol)
        if df is None or df.empty or "close" not in df.columns:
            return None
        return df["close"].dropna()

    def _calendar(self) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex([])
        for row in self.universe.itertuples(index=False):
            if row.source in CALENDAR_SOURCES:
                s = self._load(row.source, row.symbol)
                if s is not None:
                    idx = idx.union(s.index)
        if len(idx) == 0:
            raise ValueError("缓存里没有任何 A 股 ETF 或指数数据，先运行 python -m datasource universe")
        return idx

    def _fx(self, calendar) -> dict:
        out = {"CNY": None}
        for row in self.universe[self.universe["role"] == "aux"].itertuples(index=False):
            s = self._load(row.source, row.symbol)
            if s is not None:
                out[row.symbol.upper()] = _align(s, calendar)
        return out

    def _futures_cont(self, symbol: str):
        """futures/ 生成的复权连续序列（超额收益），文件名用品种代码，如 M0 -> M。"""
        variety = symbol.rstrip("0123456789").upper()
        p = self.cache.root / "futures_cont" / f"{variety}.csv"
        if not p.exists():
            return None
        return pd.read_csv(p, index_col="date", parse_dates=["date"])["ret"]

    def _returns(self, row, calendar, fx, cash) -> pd.Series:
        """一条序列在交易日历上的日总收益（人民币）。"""
        if row.source == "futures_main":
            cont = self._futures_cont(row.symbol)
            if cont is not None:
                excess = cont.reindex(calendar)
                return excess + cash if cash is not None else excess
            self.notes.append(f"{row.id}: 没有自建复权连续，用了未复权主力连续，换月跳空会混进收益")
        s = self._load(row.source, row.symbol)
        if s is None:
            return None
        if row.source == "cn_bond_yield":
            return bond_returns_from_yield(_align(s, calendar))
        if row.source == "repo":
            return rate_to_returns(_align(s, calendar))
        level = _align(s, calendar, lag=row.source in LAGGED_SOURCES)
        cur = (row.currency or "CNY").upper()
        if cur != "CNY":
            if fx.get(cur) is None:
                self.notes.append(f"{row.id}: 缺少 {cur} 汇率，无法换算人民币，已跳过")
                return None
            level = level * fx[cur]
        r = level.pct_change(fill_method=None)
        if row.source == "futures_main" and cash is not None:
            r = r + cash  # 期货价格收益是超额收益，商品 ETF 还拿着现金的利息
        return r

    def build(self) -> Panel:
        self.notes = []
        u = self.universe
        calendar = self._calendar()
        fx = self._fx(calendar)

        # 先算现金，商品代理要用
        cash = self._splice("cash", calendar, fx, None)
        if cash is None:
            self.notes.append("缺少现金数据（货币 ETF 和回购利率都没有），现金收益按 0 计")
            cash_r = pd.Series(0.0, index=calendar)
        else:
            cash_r = cash[0].fillna(0.0)

        total, origin, classes = {}, {}, {}
        for asset in u[(u["role"] == "trade") & (u["asset_class"] != "cash")]["asset"]:
            res = self._splice(asset, calendar, fx, cash_r)
            if res is None:
                self.notes.append(f"{asset}: 没有任何数据，已跳过")
                continue
            total[asset], origin[asset] = res
            classes[asset] = u[(u["asset"] == asset) & (u["role"] == "trade")]["asset_class"].iloc[0]
        if not total:
            raise ValueError("没有可用的风险资产数据")
        total = pd.DataFrame(total)
        first = total.dropna(how="all").index[0]
        panel = Panel(total=total.loc[first:], cash=cash_r.loc[first:], classes=classes,
                      origin=pd.DataFrame(origin).loc[first:], notes=list(self.notes))
        return panel

    def _splice(self, asset, calendar, fx, cash):
        """返回 (日收益, 来源标记)；ETF 上市后用 ETF，之前用代理并扣 ETF 费率。"""
        rows = self.universe[self.universe["asset"] == asset]
        trade = rows[rows["role"] == "trade"]
        proxy = rows[rows["role"] == "proxy"]
        r_trade = self._returns(trade.iloc[0], calendar, fx, cash) if len(trade) else None
        r_proxy = self._returns(proxy.iloc[0], calendar, fx, cash) if len(proxy) else None
        if r_trade is None and r_proxy is None:
            return None
        if r_proxy is not None:
            fee = float(trade.iloc[0].fee or 0) if len(trade) else 0.0
            r_proxy = r_proxy - fee / TRADING_DAYS
        if r_trade is None:
            out = r_proxy
            tag = pd.Series("proxy", index=calendar).where(out.notna(), "")
        elif r_proxy is None:
            out = r_trade
            tag = pd.Series("etf", index=calendar).where(out.notna(), "")
        else:
            out = r_trade.combine_first(r_proxy)
            tag = pd.Series("", index=calendar)
            tag[r_proxy.notna()] = "proxy"
            tag[r_trade.notna()] = "etf"
        return out, tag


def build_panel(universe=None, cache=None) -> Panel:
    return PanelBuilder(universe, cache).build()
