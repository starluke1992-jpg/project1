"""AKShare 接口的薄封装：统一列名、日期索引和数值类型。

日线序列都返回以 date（DatetimeIndex）为索引、按日期升序的 DataFrame，至少有 close 一列：
  行情类    open/high/low/close，有的还有 volume/amount/open_interest/settle
  收益率类  close 是收益率或利率，单位为 %（如 1.85 表示 1.85%）
  汇率类    close 是 1 单位外币兑人民币
期货逐合约数据（futures_contracts_daily）是长表，每行一个合约一天，date 是普通列。

AKShare 靠抓取网页取数，网站一改版，字段就可能变。字段映射集中在这个文件里，
上游变了只需要改这里。
"""
import importlib
import time

import pandas as pd

# 逐月分段请求之间的停顿（秒），避免请求过密被网站限流；测试里设为 0
REQUEST_PAUSE = 0.5

_COLUMNS = {
    "日期": "date", "开盘": "open", "开盘价": "open", "收盘": "close", "收盘价": "close",
    "最高": "high", "最高价": "high", "最低": "low", "最低价": "low",
    "成交量": "volume", "成交额": "amount", "成交金额": "amount", "turnover": "amount",
    "持仓量": "open_interest", "动态结算价": "settle",
}
_BAR_COLUMNS = ["open", "high", "low", "close", "volume", "amount", "open_interest", "settle"]
_CONTRACT_COLUMNS = ["date", "symbol", "variety", "open", "high", "low", "close",
                     "volume", "open_interest", "amount", "settle", "pre_settle"]
_FX_NAMES = {"USD": "美元", "HKD": "港元", "EUR": "欧元", "JPY": "日元", "GBP": "英镑"}


def _ak():
    try:
        return importlib.import_module("akshare")
    except ImportError as e:
        raise ImportError("需要先安装 akshare：pip install -r requirements-data.txt") from e


def _ymd(d) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def _bars(raw: pd.DataFrame) -> pd.DataFrame:
    """中英文列名统一成 open/high/low/close…，日期作索引，去重并按日期排序。"""
    df = raw.rename(columns=_COLUMNS)
    if "date" not in df.columns:
        raise ValueError(f"上游数据没有日期列，现有列: {list(raw.columns)}")
    df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df[[c for c in _BAR_COLUMNS if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    return df[~df.index.duplicated(keep="last")]


def _clip(df: pd.DataFrame, start, end) -> pd.DataFrame:
    return df.loc[pd.Timestamp(start):pd.Timestamp(end)]


def _month_windows(start, end):
    """把 [start, end] 切成按自然月的小段，返回 [(段起, 段止), ...]。"""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    windows, s = [], start
    while s <= end:
        e = min(s + pd.offsets.MonthEnd(0), end)
        windows.append((s, e))
        s = e + pd.Timedelta(days=1)
    return windows


def etf_daily(code: str, start="19900101", end="22220101", adjust: str = "hfq") -> pd.DataFrame:
    """场内 ETF 日线（东方财富）。

    adjust 默认 "hfq" 后复权：分红再投资后的价格，算收益用它；而且后复权的历史价格不会因为
    新的分红而改变，可以安全地增量更新。"qfq" 前复权每次分红都会改写全部历史。
    """
    raw = _ak().fund_etf_hist_em(symbol=code, period="daily", start_date=_ymd(start),
                                 end_date=_ymd(end), adjust=adjust)
    return _bars(raw)


def index_cn_daily(code: str, start="19900101", end="22220101") -> pd.DataFrame:
    """A 股指数日线（东方财富），如 000300 沪深300、000905 中证500。价格指数，不含分红。"""
    raw = _ak().index_zh_a_hist(symbol=code, period="daily", start_date=_ymd(start), end_date=_ymd(end))
    return _bars(raw)


def csindex_daily(code: str, start="19900101", end="22220101") -> pd.DataFrame:
    """中证指数官网日线，可取全收益指数，如 H00300 沪深300全收益、H00905 中证500全收益。"""
    raw = _ak().stock_zh_index_hist_csindex(symbol=code, start_date=_ymd(start), end_date=_ymd(end))
    return _bars(raw)


def us_index_daily(symbol: str = ".INX", start="19900101", end="22220101") -> pd.DataFrame:
    """美股指数日线（新浪），.INX 标普500、.NDX 纳斯达克100、.DJI 道琼斯。美元计价，价格指数。"""
    return _clip(_bars(_ak().index_us_stock_sina(symbol=symbol)), start, end)


def hk_index_daily(symbol: str = "HSI", start="19900101", end="22220101") -> pd.DataFrame:
    """港股指数日线（新浪），HSI 恒生指数、HSTECH 恒生科技。港元计价，价格指数。"""
    return _clip(_bars(_ak().stock_hk_index_daily_sina(symbol=symbol)), start, end)


def gold_sge_daily(symbol: str = "Au99.99", start="19900101", end="22220101") -> pd.DataFrame:
    """上海黄金交易所现货日线，元/克。"""
    return _clip(_bars(_ak().spot_hist_sge(symbol=symbol)), start, end)


def cn_bond_yield_daily(tenor: str = "10", start="19900101", end="22220101") -> pd.DataFrame:
    """中国国债到期收益率（新浪），tenor 取 1/2/3/5/7/10/15/20/30（年）。close 单位 %。"""
    raw = _ak().bond_gb_zh_sina(symbol=f"中国{tenor}年期国债")
    return _clip(_bars(raw)[["close"]], start, end)


def fx_daily(currency: str = "USD", start="19900101", end="22220101") -> pd.DataFrame:
    """人民币汇率中间价（国家外汇管理局），close = 1 单位外币兑人民币。

    currency 可写 USD/HKD/EUR/JPY/GBP 或对应中文名。外管局表格按 100 外币报价，这里除以 100。
    （没有用新浪的中行牌价接口，因为它走 http，云端环境的代理只支持 https。）
    """
    name = _FX_NAMES.get(currency.upper(), currency)
    raw = _ak().currency_boc_safe()
    col = next((c for c in raw.columns if name in str(c)), None)
    if col is None:
        raise ValueError(f"外管局汇率表里没有 {name}，现有列: {list(raw.columns)}")
    df = pd.DataFrame({"date": raw["日期"], "close": pd.to_numeric(raw[col], errors="coerce") / 100})
    return _clip(_bars(df).dropna(), start, end)


def repo_daily(rate: str = "FR007", start="20100101", end="22220101") -> pd.DataFrame:
    """银行间回购定盘利率（外汇交易中心），rate 取 FR001/FR007/FR014 等，close 单位 %。

    接口每次最多查一个月，这里按月分段请求再拼起来。
    """
    ak, frames = _ak(), []
    for i, (s, e) in enumerate(_month_windows(start, end)):
        if i and REQUEST_PAUSE:
            time.sleep(REQUEST_PAUSE)
        part = ak.repo_rate_hist(start_date=_ymd(s), end_date=_ymd(e))
        if part is not None and not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["close"], index=pd.DatetimeIndex([], name="date"))
    raw = pd.concat(frames, ignore_index=True)
    df = pd.DataFrame({"date": raw["date"], "close": pd.to_numeric(raw[rate], errors="coerce")})
    return _bars(df).dropna()


def futures_main_daily(symbol: str = "RB0", start="19900101", end="22220101") -> pd.DataFrame:
    """期货主力连续日线（新浪），如 RB0、M0、CU0、IF0。

    注意：这是未复权的拼接序列，换月处有跳空，直接算收益会出错；只适合粗看走势。
    正式回测用 futures_contracts_daily 的逐合约数据自建复权连续价格。
    """
    raw = _ak().futures_main_sina(symbol=symbol, start_date=_ymd(start), end_date=_ymd(end))
    return _bars(raw)


def futures_contracts_daily(market: str, start, end) -> pd.DataFrame:
    """交易所逐合约日线（长表），market 取 CFFEX/SHFE/DCE/CZCE/INE/GFEX。

    AKShare 按天逐个请求交易所网站，取几年数据要很久；请通过 cache.SeriesCache 按月缓存使用。
    """
    raw = _ak().get_futures_daily(start_date=_ymd(start), end_date=_ymd(end), market=market.upper())
    if raw is None or raw.empty:
        return pd.DataFrame(columns=_CONTRACT_COLUMNS)
    df = raw.rename(columns={"turnover": "amount"})
    df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
    for c in ("open", "high", "low", "close", "volume", "open_interest", "amount", "settle", "pre_settle"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[[c for c in _CONTRACT_COLUMNS if c in df.columns]].dropna(subset=["date"])
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def futures_fees() -> pd.DataFrame:
    """各合约的手续费和保证金率（openctp 整理），用来生成合约规格表。"""
    return _ak().futures_fees_info()


# universe 配置文件里 source 列可用的取值 -> (函数, 序列类型)
# 序列类型决定数据检查方式：price 查跳变，rate 查取值范围
SOURCES = {
    "etf": (etf_daily, "price"),
    "index_cn": (index_cn_daily, "price"),
    "csindex": (csindex_daily, "price"),
    "us_index": (us_index_daily, "price"),
    "hk_index": (hk_index_daily, "price"),
    "gold_sge": (gold_sge_daily, "price"),
    "fx": (fx_daily, "price"),
    "futures_main": (futures_main_daily, "price"),
    "cn_bond_yield": (cn_bond_yield_daily, "rate"),
    "repo": (repo_daily, "rate"),
}


def fetch(source: str, symbol: str, start, end) -> pd.DataFrame:
    """按 source 名称调用对应接口。"""
    if source not in SOURCES:
        raise KeyError(f"未知数据源 {source}，可选: {sorted(SOURCES)}")
    return SOURCES[source][0](symbol, start, end)
