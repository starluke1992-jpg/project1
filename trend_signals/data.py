"""行情读取：本地 CSV，或用 akshare 拉取主力连续合约。"""
import pandas as pd

_ALIASES = {
    "date": "date", "datetime": "date", "日期": "date", "时间": "date",
    "open": "open", "开盘": "open", "开盘价": "open",
    "high": "high", "最高": "high", "最高价": "high",
    "low": "low", "最低": "low", "最低价": "low",
    "close": "close", "收盘": "close", "收盘价": "close",
    "volume": "volume", "成交量": "volume",
}


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名为 open/high/low/close，按日期升序，日期作索引。"""
    df = df.rename(columns={c: _ALIASES.get(str(c).strip().lower(), c) for c in df.columns})
    missing = {"date", "open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"缺少列: {sorted(missing)}，现有列: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df.set_index("date").sort_index()[keep].astype(float)
    return df[~df.index.duplicated(keep="last")]


def load_csv(path: str) -> pd.DataFrame:
    """UTF-8 或 GBK 编码都可以（行情软件导出、Excel 另存的 CSV 常是 GBK）。"""
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = pd.read_csv(path, encoding="gbk")
    return normalize(raw)


def fetch_akshare(symbol: str = "IF0", start: str = "19900101", end: str = "22220101") -> pd.DataFrame:
    """新浪主力连续合约日线，如 IF0（沪深300）、IC0、IM0、RB0。

    注意：新浪的主力连续是未复权拼接，换月处会有跳空；截图用的是前复权数据，
    所以价格水平和个别信号位置会与截图有出入。
    """
    try:
        import akshare as ak
    except ImportError as e:
        raise ImportError("需要先安装 akshare：pip install akshare") from e
    raw = ak.futures_main_sina(symbol=symbol,
                               start_date=start.replace("-", ""),
                               end_date=end.replace("-", ""))
    return normalize(raw)
