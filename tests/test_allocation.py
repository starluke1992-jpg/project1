"""方案 A 的数据面板、风险估计、权重、趋势过滤、回测引擎和指标。"""
import numpy as np
import pandas as pd
import pytest

from allocation.backtest import month_ends, run_backtest, week_ends
from allocation.data import PanelBuilder, bond_returns_from_yield, rate_to_returns
from allocation.filters import ma_mom_exposure, negative_trend_exposure
from allocation.risk import TRADING_DAYS, ledoit_wolf, portfolio_vol
from allocation.strategies import RPConfig, risk_parity
from allocation.weights import scale_to_vol, two_level_risk_parity
from datasource.cache import SeriesCache
from research import metrics
from research.__main__ import main as research_main
from research.synthetic import synthetic_panel
from research.trials import count_trials, log_trial


def rand_returns(n=500, vols=(0.2, 0.05, 0.15), seed=0, start="2015-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    data = {f"a{i}": rng.normal(0, v / np.sqrt(TRADING_DAYS), n) for i, v in enumerate(vols)}
    return pd.DataFrame(data, index=idx)


# ---------- 风险与权重 ----------

def test_ledoit_wolf_properties():
    r = rand_returns()
    cov = ledoit_wolf(r)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > 0
    sample = r.cov(ddof=0) * TRADING_DAYS
    off = ~np.eye(3, dtype=bool)
    assert np.abs(cov.to_numpy()[off]).sum() <= np.abs(sample.to_numpy()[off]).sum() + 1e-12
    one = ledoit_wolf(r[["a0"]])
    assert one.iloc[0, 0] == pytest.approx(r["a0"].var(ddof=0) * TRADING_DAYS)


def test_two_level_risk_parity_equal_class_risk():
    r = rand_returns(vols=(0.2, 0.05, 0.15))
    classes = {"a0": "equity", "a1": "bond", "a2": "gold"}
    w = two_level_risk_parity(r, classes)
    assert w.sum() == pytest.approx(1.0)
    risk = w * r.std(ddof=1) * np.sqrt(TRADING_DAYS)  # 每类只有一个资产时，各类独立风险应相等
    assert risk.max() / risk.min() == pytest.approx(1.0, rel=1e-6)
    assert w["a1"] > w["a0"]  # 低波动资产权重更大


def test_scale_to_vol_limits():
    r = rand_returns()
    w = pd.Series({"a0": 0.2, "a1": 0.2, "a2": 0.2})
    up, vol = scale_to_vol(w, r, target=0.10)
    assert vol == pytest.approx(0.10)
    assert portfolio_vol(up, ledoit_wolf(r)) == pytest.approx(0.10)
    down_only, _ = scale_to_vol(w, r, target=0.50, only_down=True)
    assert down_only.equals(w)
    capped, _ = scale_to_vol(w, r, target=1.0, max_gross=1.5)
    assert capped.abs().sum() == pytest.approx(1.5)


# ---------- 趋势过滤 ----------

def _prices(path):
    return pd.Series(path, index=pd.bdate_range("2015-01-01", periods=len(path)))


def test_ma_mom_exposure():
    n = 400
    up = _prices(np.linspace(1, 2, n))
    down = _prices(np.linspace(2, 1, n))
    # 先涨一年多再回落：过去一年仍跑赢现金，但已跌破 200 日均线
    hump = _prices(np.r_[np.linspace(1, 2, 300), np.linspace(2, 1.7, 100)])
    prices = pd.DataFrame({"up": up, "down": down, "hump": hump})
    cash = _prices(np.linspace(1, 1.02, n))
    e = ma_mom_exposure(prices, cash)
    assert e["up"] == 1.0 and e["down"] == 0.0 and e["hump"] == 0.5
    assert np.isnan(ma_mom_exposure(prices.iloc[:100], cash.iloc[:100])["up"])


def test_negative_trend_exposure():
    rng = np.random.default_rng(1)
    n = 300
    noise = rng.normal(0, 0.005, n).cumsum()
    prices = pd.DataFrame({"up": _prices(np.exp(np.linspace(0, 0.4, n) + noise)),
                           "down": _prices(np.exp(np.linspace(0, -0.4, n) + noise))})
    e = negative_trend_exposure(prices)
    assert e["up"] == 1.0 and e["down"] == 0.0


# ---------- 回测引擎 ----------

def test_engine_executes_next_day_and_charges_cost():
    idx = pd.bdate_range("2020-01-01", periods=6)
    excess = pd.DataFrame({"x": [0.0, 0.0, 0.0, 0.10, 0.0, 0.0]}, index=idx)
    cash = pd.Series(0.0, index=idx)
    res = run_backtest(excess, cash, lambda ctx: pd.Series({"x": 1.0}), [idx[1]], cost_bps=10)
    # 第 2 天收盘出信号，第 3 天收盘建仓并扣成本，第 4 天吃到 10% 收益
    assert res.weights["x"].iloc[1] == 0.0 and res.weights["x"].iloc[2] == pytest.approx(1.0)
    assert res.costs.iloc[2] == pytest.approx(0.001)
    assert res.returns.iloc[3] == pytest.approx(0.10)


def test_engine_funded_vs_unfunded_drift():
    idx = pd.bdate_range("2020-01-01", periods=4)
    excess = pd.DataFrame({"x": [0.0, 0.0, 0.0, 0.0]}, index=idx)
    cash = pd.Series(0.01, index=idx)
    fn = lambda ctx: pd.Series({"x": 0.5})  # noqa: E731
    funded = run_backtest(excess, cash, fn, [idx[0]], funded=True)
    unfunded = run_backtest(excess, cash, fn, [idx[0]], funded=False)
    assert funded.weights["x"].iloc[-1] == pytest.approx(0.5)            # 和组合一起按现金利率增长
    assert unfunded.weights["x"].iloc[-1] < 0.5                          # 期货名义金额不随现金增长
    assert funded.returns.iloc[-1] == pytest.approx(0.01)


def test_engine_band_skips_small_trades():
    idx = pd.bdate_range("2020-01-01", periods=5)
    excess = pd.DataFrame({"x": 0.0}, index=idx)
    cash = pd.Series(0.0, index=idx)
    targets = iter([0.50, 0.505])
    res = run_backtest(excess, cash, lambda ctx: pd.Series({"x": next(targets)}), [idx[0], idx[2]],
                       band_abs=0.01)
    assert res.weights["x"].iloc[-1] == pytest.approx(0.50)
    assert res.turnover.iloc[3] == 0.0


def test_engine_no_lookahead_truncation():
    panel = synthetic_panel(start="2010-01-01", end="2014-12-31")
    fn = risk_parity(panel.classes, RPConfig(trend="ma_mom"))
    full = run_backtest(panel.excess, panel.cash, fn, month_ends(panel.total.index))
    cut = "2013-06-28"
    part = run_backtest(panel.excess.loc[:cut], panel.cash.loc[:cut], fn, month_ends(panel.total.loc[:cut].index))
    pd.testing.assert_frame_equal(full.weights.loc[:cut], part.weights, check_freq=False)


def test_rebalance_calendars():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    assert list(month_ends(idx).strftime("%m-%d")) == ["01-31", "02-29", "03-29"]
    assert all(d.weekday() == 4 for d in week_ends(idx)[:-1])


# ---------- 数据面板 ----------

def _write(cache, source, symbol, s):
    p = cache.path(source, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    s.rename("close").to_frame().rename_axis("date").to_csv(p)


def test_panel_builder_splice_lag_fx_bond_cash(tmp_path):
    cache = SeriesCache(tmp_path)
    days = pd.bdate_range("2020-01-01", periods=30)
    idx_px = pd.Series(np.linspace(100, 129, 30), index=days)
    _write(cache, "csindex", "H00300", idx_px)
    _write(cache, "etf", "510300", pd.Series(np.linspace(1, 1.1, 15), index=days[15:]))
    us = pd.Series(1000.0, index=days)
    us.iloc[10:] = 1100.0                                 # 美股第 10 天涨 10%
    _write(cache, "us_index", ".INX", us)
    _write(cache, "fx", "USD", pd.Series(7.0, index=days))
    _write(cache, "cn_bond_yield", "10", pd.Series(3.0, index=days))
    _write(cache, "repo", "FR007", pd.Series(1.825, index=days))
    universe = pd.DataFrame([
        ["hs300", "hs300", "ETF", "equity", "trade", "etf", "510300", "CNY", "0.0252", ""],
        ["hs300_tr", "hs300", "idx", "equity", "proxy", "csindex", "H00300", "CNY", "", ""],
        ["spx_idx", "spx", "spx", "equity", "proxy", "us_index", ".INX", "USD", "", ""],
        ["cgb10_yield", "cgb10", "y", "bond", "proxy", "cn_bond_yield", "10", "CNY", "", ""],
        ["repo7d", "cash", "repo", "cash", "proxy", "repo", "FR007", "CNY", "", ""],
        ["usdcny", "usdcny", "fx", "fx", "aux", "fx", "USD", "CNY", "", ""],
    ], columns=["id", "asset", "name", "asset_class", "role", "source", "symbol", "currency", "fee", "note"])
    # 只有 trade 行的资产才进面板：给 spx、cgb10 补一个没有缓存的 trade 行
    universe = pd.concat([universe, pd.DataFrame([
        ["spx", "spx", "etf", "equity", "trade", "etf", "513500", "CNY", "0", ""],
        ["cgb10", "cgb10", "etf", "bond", "trade", "etf", "511260", "CNY", "0", ""],
    ], columns=universe.columns)], ignore_index=True)
    panel = PanelBuilder(universe, cache).build()

    hs = panel.total["hs300"]
    origin = panel.origin["hs300"]
    assert (origin.loc[:days[15]] == "proxy").all() and origin.loc[days[16]] == "etf"  # ETF 第一天还没有收益
    expected = idx_px.pct_change().loc[days[5]] - 0.0252 / 252   # 上市前：代理收益扣费率
    assert hs.loc[days[5]] == pytest.approx(expected)
    etf = np.linspace(1, 1.1, 15)
    assert hs.loc[days[20]] == pytest.approx(etf[5] / etf[4] - 1)
    spx = panel.total["spx"]
    assert spx.loc[days[10]] == pytest.approx(0.0) and spx.loc[days[11]] == pytest.approx(0.10)  # 晚一天
    assert panel.total["cgb10"].loc[days[2]] == pytest.approx(0.03 / 365)   # 收益率不变：只有利息
    assert panel.cash.loc[days[2]] == pytest.approx(0.01825 / 365)


def test_bond_and_rate_returns():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    r = bond_returns_from_yield(pd.Series([3.0, 3.1, 3.1], index=idx))
    assert r.iloc[1] == pytest.approx(0.03 / 365 - 8.5 * 0.001 + 0.5 * 85 * 0.001 ** 2)
    assert r.iloc[2] == pytest.approx(0.031 * 3 / 365)
    assert rate_to_returns(pd.Series([2.0, 2.0], index=idx[:2])).iloc[1] == pytest.approx(0.02 / 365)


# ---------- 指标与登记 ----------

def test_metrics_basic():
    idx = pd.bdate_range("2020-01-01", periods=252)
    r = pd.Series(0.0004, index=idx)
    r.iloc[100] = -0.10
    s = metrics.summary(r, pd.Series(0.0, index=idx))
    assert s["最大回撤"] == pytest.approx(-0.10, abs=1e-9)
    assert s["年化收益"] == pytest.approx((1.0004 ** 251 * 0.9) - 1, rel=1e-3)
    assert metrics.period_slice(pd.Series(1, index=pd.bdate_range("2018-12-25", "2019-01-10")), "dev").index[-1] \
        == pd.Timestamp("2018-12-31")


def test_trial_log(tmp_path):
    p = tmp_path / "t.csv"
    log_trial("A", "x", "dev", {"夏普": 1.0}, {"k": 1}, path=p)
    log_trial("B2", "y", "dev", {}, {}, path=p)
    assert count_trials("A", path=p) == 1 and count_trials(path=p) == 2


def test_holdout_guard(capsys):
    assert research_main(["run", "A", "--period", "holdout"]) == 2
    assert "--confirm-holdout" in capsys.readouterr().out
