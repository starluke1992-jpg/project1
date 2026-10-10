"""datasource 的离线测试：用假的 akshare 模块按上游真实格式返回数据，不需要联网。"""
import types
import urllib.error

import pandas as pd
import pytest

from datasource import __main__ as cli
from datasource import akshare_api as api
from datasource import cache as cache_mod
from datasource import connectivity
from datasource.cache import SeriesCache
from datasource.checks import check_series


def _days(start, end):
    return pd.bdate_range(start, end)


def etf_raw(start, end, base=1.0):
    """fund_etf_hist_em 的返回格式：中文列名、字符串日期、按日期升序。"""
    d = _days(start, end)
    close = base + 0.001 * (d - pd.Timestamp("2020-01-01")).days.to_numpy()  # 只由日期决定，和请求区间无关
    return pd.DataFrame({"日期": d.strftime("%Y-%m-%d"), "开盘": close, "收盘": close, "最高": close,
                         "最低": close, "成交量": 100.0, "成交额": 1e4, "振幅": 0.0, "涨跌幅": 0.0,
                         "涨跌额": 0.0, "换手率": 0.0})


class FakeAk(types.SimpleNamespace):
    """记录每次调用，返回上游格式的数据。"""

    def __init__(self):
        super().__init__(calls=[])

    def fund_etf_hist_em(self, symbol, period, start_date, end_date, adjust):
        self.calls.append(("etf", symbol, start_date, end_date, adjust))
        return etf_raw(start_date, end_date)

    def futures_main_sina(self, symbol, start_date, end_date):
        d = _days(start_date, end_date)
        return pd.DataFrame({"日期": d.date, "开盘价": 3000.0, "最高价": 3010.0, "最低价": 2990.0,
                             "收盘价": 3005.0, "成交量": 1e5, "持仓量": 2e5, "动态结算价": 3001.0})

    def index_us_stock_sina(self, symbol):
        d = _days("2020-01-01", "2020-03-31")
        return pd.DataFrame({"date": d.date, "open": 3000.0, "high": 3000.0, "low": 3000.0,
                             "close": [3000.0 + i for i in range(len(d))], "volume": 1.0, "amount": 1.0})

    def currency_boc_safe(self):
        d = _days("2024-01-01", "2024-01-31")
        return pd.DataFrame({"日期": d.date, "美元": 710.0, "港元": 91.0, "日元": 4.8})

    def repo_rate_hist(self, start_date, end_date):
        self.calls.append(("repo", start_date, end_date))
        d = _days(start_date, end_date)
        return pd.DataFrame({"date": d.date, "FR001": 1.5, "FR007": 1.8, "FR014": 1.9,
                             "FDR001": 1.4, "FDR007": 1.7, "FDR014": 1.8})

    def get_futures_daily(self, start_date, end_date, market):
        self.calls.append(("contracts", market, start_date, end_date))
        d = _days(start_date, end_date)
        rows = [{"symbol": s, "date": day.strftime("%Y%m%d"), "open": 1.0, "high": 1.0, "low": 1.0,
                 "close": 1.0, "volume": 1, "open_interest": 1, "turnover": 1.0, "settle": 1.0,
                 "pre_settle": 1.0, "variety": "CU"} for day in d for s in ("CU2402", "CU2403")]
        return pd.DataFrame(rows)


@pytest.fixture
def fake(monkeypatch):
    ak = FakeAk()
    monkeypatch.setattr(api, "_ak", lambda: ak)
    monkeypatch.setattr(api, "REQUEST_PAUSE", 0)
    monkeypatch.setattr(cache_mod.time, "sleep", lambda s: None)
    return ak


# ---------- 字段统一 ----------

def test_etf_daily_normalized(fake):
    df = api.etf_daily("510300", "2024-01-01", "2024-01-31")
    assert isinstance(df.index, pd.DatetimeIndex) and df.index.name == "date"
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert df.index.is_monotonic_increasing and df["close"].dtype == float
    assert fake.calls[0] == ("etf", "510300", "20240101", "20240131", "hfq")


def test_futures_main_columns(fake):
    df = api.futures_main_daily("RB0", "2024-01-01", "2024-01-10")
    assert {"open", "high", "low", "close", "volume", "open_interest", "settle"} <= set(df.columns)


def test_full_history_source_is_clipped(fake):
    df = api.us_index_daily(".INX", "2020-02-01", "2020-02-29")
    assert df.index.min() >= pd.Timestamp("2020-02-01") and df.index.max() <= pd.Timestamp("2020-02-29")


def test_fx_converted_to_one_unit(fake):
    assert api.fx_daily("USD")["close"].iloc[0] == pytest.approx(7.10)
    assert api.fx_daily("HKD")["close"].iloc[0] == pytest.approx(0.91)
    with pytest.raises(ValueError):
        api.fx_daily("CHF")


def test_repo_requested_month_by_month(fake):
    df = api.repo_daily("FR007", "2024-01-15", "2024-03-10")
    windows = [(s, e) for kind, s, e in fake.calls if kind == "repo"]
    assert windows == [("20240115", "20240131"), ("20240201", "20240229"), ("20240301", "20240310")]
    assert list(df.columns) == ["close"] and (df["close"] == 1.8).all()


def test_futures_contracts_long_table(fake):
    df = api.futures_contracts_daily("shfe", "2024-01-02", "2024-01-05")
    assert "amount" in df.columns and "turnover" not in df.columns
    assert df["date"].dtype.kind == "M" and set(df["symbol"]) == {"CU2402", "CU2403"}
    assert fake.calls[0][1] == "SHFE"


def test_unknown_source():
    with pytest.raises(KeyError):
        api.fetch("nope", "x", "2024-01-01", "2024-01-02")


# ---------- 缓存 ----------

def test_cache_incremental_update(fake, tmp_path):
    c = SeriesCache(tmp_path, overlap_days=10)
    first = c.get("etf", "510300", "2024-01-01", "2024-03-29")
    assert c.path("etf", "510300").exists()
    second = c.get("etf", "510300", "2024-01-01", "2024-04-30")
    last_call = fake.calls[-1]
    assert last_call[2] == "20240319"  # 从缓存最后一天往前回看 10 天
    assert second.index[-1] == pd.Timestamp("2024-04-30")
    pd.testing.assert_frame_equal(second.loc[:"2024-03-29"], first, check_freq=False)


def test_cache_hit_needs_no_network(fake, tmp_path):
    c = SeriesCache(tmp_path)
    c.get("etf", "510300", "2024-01-01", "2024-03-29")
    n = len(fake.calls)
    df = c.get("etf", "510300", "2024-02-01", "2024-02-29")
    assert len(fake.calls) == n
    assert df.index[0] >= pd.Timestamp("2024-02-01") and df.index[-1] <= pd.Timestamp("2024-02-29")


def test_cache_refetch_when_earlier_start(fake, tmp_path):
    c = SeriesCache(tmp_path)
    c.get("etf", "510300", "2024-03-01", "2024-03-29")
    c.get("etf", "510300", "2024-01-01", "2024-03-29")
    assert fake.calls[-1][2] == "20240101"


def test_cache_refetch_when_history_revised(fake, tmp_path, monkeypatch):
    c = SeriesCache(tmp_path)
    c.get("etf", "510300", "2024-01-01", "2024-03-29")
    monkeypatch.setattr(fake, "fund_etf_hist_em",
                        lambda symbol, period, start_date, end_date, adjust:
                        (fake.calls.append(("etf", symbol, start_date, end_date, adjust)),
                         etf_raw(start_date, end_date, base=2.0))[1])
    df = c.get("etf", "510300", "2024-01-01", "2024-04-30")
    assert fake.calls[-1][2] == "20240101"  # 重叠部分对不上，整段重拉
    assert df["close"].iloc[0] > 2.0  # 用的是重拉后的新数据


def test_retry_then_succeed(monkeypatch):
    monkeypatch.setattr(cache_mod.time, "sleep", lambda s: None)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("boom")
        return "ok"

    assert cache_mod.with_retry(flaky, tries=3) == "ok" and len(attempts) == 3
    with pytest.raises(ConnectionError):
        cache_mod.with_retry(lambda: (_ for _ in ()).throw(ConnectionError("x")), tries=2)


def test_futures_contracts_monthly_cache(fake, tmp_path):
    c = SeriesCache(tmp_path)
    df = c.get_futures_contracts("SHFE", "2024-01-10", "2024-02-20")
    assert (tmp_path / "futures_contracts" / "SHFE" / "2024-01.csv").exists()
    assert df["date"].min() >= pd.Timestamp("2024-01-10") and df["date"].max() <= pd.Timestamp("2024-02-20")
    n = len(fake.calls)
    c.get_futures_contracts("SHFE", "2024-01-10", "2024-02-20")
    assert len(fake.calls) == n  # 已结束的月份直接读缓存


# ---------- 数据检查 ----------

def _series(values, start="2024-01-01"):
    return pd.DataFrame({"close": values}, index=pd.bdate_range(start, periods=len(values)))


def test_check_clean_series():
    assert check_series(_series([1.0 + 0.01 * i for i in range(50)])) == []


def test_check_jump_stale_nonpositive():
    issues = check_series(_series([1.0, 1.01, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, -1.0]))
    text = " ".join(issues)
    assert "单日涨跌" in text and "不变" in text and "非正价格" in text


def test_check_gap_and_rate_range():
    df = pd.DataFrame({"close": [2.0, 2.1, 25.0]},
                      index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-03-01"]))
    issues = check_series(df, kind="rate")
    assert any("断档" in i for i in issues) and any("超出" in i for i in issues)
    assert check_series(pd.DataFrame()) == ["没有数据"]


# ---------- 连通性 ----------

def test_probe_classification(monkeypatch):
    def http_error(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "nf", None, None)

    def blocked(req, timeout):
        raise urllib.error.URLError("Tunnel connection failed: 403 Forbidden")

    monkeypatch.setattr(connectivity.urllib.request, "urlopen", http_error)
    assert connectivity.probe("example.com") == (True, "HTTP 404")
    monkeypatch.setattr(connectivity.urllib.request, "urlopen", blocked)
    ok, detail = connectivity.probe("example.com")
    assert not ok and "403" in detail


# ---------- 命令行 ----------

def test_cli_universe(fake, tmp_path, capsys):
    cfg = tmp_path / "u.csv"
    cfg.write_text("id,name,asset_class,role,source,symbol,currency,note\n"
                   "hs300,沪深300ETF,equity,trade,etf,510300,CNY,\n"
                   "bad,坏数据源,equity,trade,nope,1,CNY,\n", encoding="utf-8")
    code = cli.main(["--cache-dir", str(tmp_path / "c"), "universe", "--config", str(cfg),
                     "--start", "2024-01-01", "--end", "2024-02-29"])
    out = capsys.readouterr().out
    assert code == 1  # 有一条失败
    assert "hs300" in out and "通过" in out and "失败" in out


def test_universe_config_sources_are_known():
    cfg = pd.read_csv(cli.DEFAULT_UNIVERSE, dtype=str, keep_default_na=False)
    assert set(cfg["source"]) <= set(api.SOURCES)
    assert cfg["id"].is_unique
