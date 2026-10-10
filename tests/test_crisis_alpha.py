from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from crisis_alpha import (ASSET_CLASS_BUDGET, UNIVERSE, Config, benchmark_returns, build_continuous, prepare,
                          run_backtest, stats, synthetic_market)
from crisis_alpha.analysis import crisis_episodes, smile, up_down_beta
from crisis_alpha.config import TRADING_DAYS
from crisis_alpha.live import capital_check, daily_orders, split_order
from crisis_alpha.portfolio import buffered, build_panel, instrument_weights, target_positions
from crisis_alpha.signals import breakout, combined_forecast, ewmac, price_vol

CODES = ["IF", "IC", "T", "TF", "AU", "CU", "RB", "SC", "TA", "M", "SR"]


@pytest.fixture(scope="module")
def market():
    return synthetic_market(years=8, crises=((600, 80), (1300, 60)), codes=CODES)


@pytest.fixture(scope="module")
def result(market):
    return run_backtest(market, Config())


def test_universe_and_budget():
    assert sum(ASSET_CLASS_BUDGET.values()) == pytest.approx(1.0)
    assert {i.asset_class for i in UNIVERSE.values()} == set(ASSET_CLASS_BUDGET)
    assert UNIVERSE["IF"].multiplier == 300 and UNIVERSE["TS"].multiplier == 20000


def _trend(n=400, slope=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(1000 + slope * np.arange(n) + rng.normal(0, 3, n).cumsum(), index=pd.bdate_range("2020-01-01", periods=n))


def test_signals_follow_trend_direction():
    cfg = Config()
    for slope in (2.0, -2.0):
        adj = _trend(slope=slope)
        vol = price_vol(adj)
        f = combined_forecast(adj, vol, cfg)
        assert np.sign(f.iloc[-50:]).eq(np.sign(slope)).all()
        assert f.abs().max() <= cfg.forecast_cap


def test_signals_invariant_to_price_shift_and_scale():
    """价差法复权会把历史价格整体平移；信号必须不变。价格乘以常数（换计价单位）也不变。"""
    cfg = Config()
    adj = _trend(slope=0.3)
    base = combined_forecast(adj, price_vol(adj), cfg)
    for moved in (adj + 5000, adj * 7):
        pd.testing.assert_series_equal(base, combined_forecast(moved, price_vol(moved), cfg), check_exact=False)
    b = breakout(adj, 40).dropna()
    assert b.abs().max() <= 0.5 * 3.1 + 1e-9
    assert ewmac(adj, price_vol(adj), 16, 64).dropna().size > 0


def test_forecast_scaling_on_random_walk():
    """零漂移随机游走上平均 |预测值| 应接近 1（真实市场有趋势，会略高于这个值）。"""
    rng = np.random.default_rng(3)
    vals = []
    for _ in range(20):
        adj = pd.Series(rng.normal(0, 1, 3000).cumsum() + 500)
        vals.append(combined_forecast(adj, price_vol(adj), Config()).iloc[400:].abs().mean())
    assert 0.75 < np.mean(vals) < 1.1


def test_instrument_weights_follow_budget():
    idx = pd.bdate_range("2020-01-01", periods=3)
    avail = pd.DataFrame({"IF": [1, 1, 1], "IC": [0, 1, 1], "T": [1, 1, 0], "CU": [1, 1, 1]}, index=idx).astype(bool)
    budget = {"equity": 0.5, "bond": 0.3, "base_metal": 0.2}
    w = instrument_weights(avail, budget)
    assert w.sum(axis=1).tolist() == pytest.approx([1, 1, 1])
    assert w.loc[idx[0], "IF"] == pytest.approx(0.5) and w.loc[idx[0], "IC"] == 0
    assert w.loc[idx[1], ["IF", "IC"]].tolist() == pytest.approx([0.25, 0.25])
    # 第三天国债没有可交易品种，预算按比例分给股指和有色
    assert w.loc[idx[2], "T"] == 0
    assert w.loc[idx[2], "CU"] == pytest.approx(0.2 / 0.7)


def test_position_sizing_hits_risk_budget(market):
    """平均仓位（F=1）× 每手风险 = 资金 × 目标波动 × 权重 × IDM。"""
    cfg = Config()
    panel = build_panel(market, cfg)
    tp = target_positions(panel, cfg)
    d = panel["forecast"].index[-1]
    mult = pd.Series({c: UNIVERSE[c].multiplier for c in CODES})
    risk = tp["avg_position"].loc[d] * mult * panel["vol"].loc[d] * np.sqrt(TRADING_DAYS)
    expected = tp["weights"].loc[d] * tp["idm"].loc[d] * cfg.capital * cfg.target_vol
    np.testing.assert_allclose(risk.to_numpy(), expected.to_numpy(), rtol=1e-9)
    assert 1.0 <= tp["idm"].max() <= cfg.idm_cap


def test_risk_overlay_caps_ex_ante_vol_and_margin(market):
    cfg = replace(Config(), max_vol_mult=0.5, max_margin=0.05, integer=False)
    panel = build_panel(market, cfg)
    tp = target_positions(panel, cfg)
    assert (tp["scale"] < 1).any()
    mult = pd.Series({c: UNIVERSE[c].multiplier for c in CODES})
    margin = pd.Series({c: UNIVERSE[c].margin for c in CODES})
    used = (tp["target"].abs() * panel["price"] * mult * margin).sum(axis=1) / cfg.capital
    assert used.max() <= cfg.max_margin + 1e-9


def test_buffer_rule():
    assert buffered(5.0, 4.6, 0.5) == 4.6                  # 在缓冲区 [4, 6] 内不动
    assert buffered(5.0, 3.0, 0.5) == 4.0                  # 只调到缓冲区下沿 np.round(4.5) = 4
    assert buffered(-5.2, 0.0, 1.0) == -4.0
    assert buffered(5.0, 3.0, 0.5, integer=False) == 4.5
    np.testing.assert_array_equal(buffered([2, -2], [0, 0], [0.1, 0.1]), [2, -2])


def test_buffer_reduces_trading(market):
    loose = run_backtest(market, replace(Config(), buffer=0.0))
    tight = run_backtest(market, Config())
    assert tight.positions.diff().abs().sum().sum() < loose.positions.diff().abs().sum().sum()
    assert (tight.positions == tight.positions.round()).all().all()


def test_no_lookahead(market):
    """截断行情后算出的持仓，必须和完整行情算出的同一段持仓完全一致。"""
    full = run_backtest(market, Config())
    for cut in (500, 900, 1400):
        part = {c: d.iloc[:cut] for c, d in market.items()}
        res = run_backtest(part, Config())
        pd.testing.assert_frame_equal(res.positions, full.positions.iloc[:cut])
        pd.testing.assert_series_equal(res.returns, full.returns.iloc[:cut])


def test_execution_lag(market):
    a = run_backtest(market, replace(Config(), execution_lag=0))
    b = run_backtest(market, replace(Config(), execution_lag=2))
    pd.testing.assert_frame_equal(b.positions.iloc[2:], a.positions.shift(2).iloc[2:])


def test_pnl_accounting(result, market):
    mult = UNIVERSE["CU"].multiplier
    pos = result.positions["CU"]
    expected = pos.shift().fillna(0) * market["CU"]["adj"].diff().fillna(0) * mult
    pd.testing.assert_series_equal(result.pnl["CU"], expected, check_names=False)
    assert (result.costs >= 0).all().all()
    total = (result.pnl.sum(axis=1) - result.costs.sum(axis=1)) / result.cfg.capital
    pd.testing.assert_series_equal(result.returns, total)
    assert result.class_returns().sum(axis=1).sum() == pytest.approx(result.returns.sum())


def test_profits_in_synthetic_crises(result, market):
    """模拟危机（股指持续下跌、国债黄金上涨、商品下跌）期间组合应赚钱，危机结束时股指是空头。
    危机前股指若在上涨，多头要过一段时间才翻空，所以只检查危机末尾的仓位。"""
    idx = result.returns.index
    for start, days in ((600, 80), (1300, 60)):
        w = slice(idx[start], idx[start + days - 1])
        assert result.returns.loc[w].sum() > 0
        assert result.positions["IF"].iloc[start + days - 1] < 0
    rep = up_down_beta(result.returns, benchmark_returns(market["IF"]))
    assert rep["beta_down"] < 0 and rep["worst10_strategy_avg"] > 0


def test_realized_vol_near_target(result):
    s = stats(result.returns.iloc[300:])
    assert 0.05 < s["ann_vol"] < 0.25


def test_prepare_with_contract_column_removes_roll_gap():
    dates = pd.bdate_range("2021-01-04", periods=6)
    df = pd.DataFrame({"date": dates, "close": [100, 101, 102, 110, 111, 112],
                       "contract": ["A", "A", "A", "B", "B", "B"]})
    out = prepare(df)
    assert out["roll"].tolist() == [False, False, True, False, False, False]
    assert out["adj"].diff().iloc[3] == 0           # 换月跳空被去掉
    assert out["adj"].iloc[-1] == out["price"].iloc[-1] == 112
    assert out["adj"].diff().iloc[[1, 2, 4, 5]].tolist() == [1, 1, 1, 1]


def test_build_continuous_from_contracts():
    """两个合约：A 持仓量下降、B 上升，第 3 天 B 超过 A 后换月。换月当天的盈亏仍按 A 计。"""
    dates = pd.bdate_range("2021-01-04", periods=6)
    a = [100, 102, 101, 104, 105, np.nan]
    b = [110, 111, 112, 116, 115, 117]
    oi_a = [900, 800, 400, 300, 100, np.nan]
    oi_b = [100, 300, 600, 900, 1000, 1100]
    rows = []
    for i, d in enumerate(dates):
        if not np.isnan(a[i]):
            rows.append((d, "X2101", a[i], oi_a[i]))
        rows.append((d, "X2105", b[i], oi_b[i]))
    out = build_continuous(pd.DataFrame(rows, columns=["date", "contract", "close", "open_interest"]))
    assert out["contract"].tolist() == ["X2101", "X2101", "X2105", "X2105", "X2105", "X2105"]
    assert out["roll"].tolist() == [False, False, True, False, False, False]
    # 价差：A 的 +2、-1，换月后 B 的 +4、-1、+2
    assert out["adj"].diff().iloc[1:].tolist() == [2, -1, 4, -1, 2]
    assert out["adj"].iloc[-1] == out["price"].iloc[-1] == 117
    # A 后来持仓量再大也不会换回去
    rows.append((dates[-1] + pd.offsets.BDay(), "X2101", 120.0, 99999.0))
    rows.append((dates[-1] + pd.offsets.BDay(), "X2105", 118.0, 1.0))
    again = build_continuous(pd.DataFrame(rows, columns=["date", "contract", "close", "open_interest"]))
    assert again["contract"].iloc[-1] == "X2105"


def test_actual_roll_cost_charged_on_roll_days():
    m = synthetic_market(years=3, crises=((300, 60),), codes=["CU", "IF"])
    cu = m["CU"].copy()
    cu["roll"] = False
    cu.iloc[400::40, cu.columns.get_loc("roll")] = True
    m["CU"] = cu
    res = run_backtest(m, Config())
    roll_days = cu.index[cu["roll"]]
    pos = res.positions["CU"]
    held_through_roll = roll_days[(pos.loc[roll_days] != 0) & (pos.diff().loc[roll_days] == 0)]
    assert len(held_through_roll) and (res.costs.loc[held_through_roll, "CU"] > 0).all()
    quiet = cu.index[(~cu["roll"]) & (pos.diff() == 0)]
    assert (res.costs.loc[quiet, "CU"] == 0).all()


def test_split_order():
    assert split_order(0, 3) == ["买开 3"]
    assert split_order(2, -3) == ["卖平 2", "卖开 3"]
    assert split_order(-4, -1) == ["买平 3"]
    assert split_order(-1, 2) == ["买平 1", "买开 2"]
    assert split_order(5, 5) == []


def test_daily_orders_and_capital_check(market):
    table, summary = daily_orders(market, {"IF": -1, "CU": 2})
    assert set(table.index) == set(CODES)
    assert (table["trade"] == table["new"] - table["current"]).all()
    assert table.loc["IF", "current"] == -1
    assert summary["risk_scale"] <= 1
    with pytest.raises(ValueError):
        daily_orders(market, {"ZZ": 1})
    cap = capital_check(market)
    assert cap["capital_needed"].notna().all()
    assert cap["ok"].equals(cap["avg_position"] >= 1)


def test_crisis_episode_detection():
    idx = pd.bdate_range("2020-01-01", periods=8)
    bench = pd.Series([0.0, -0.1, -0.1, 0.05, 0.3, -0.05, -0.2, 0.0], index=idx)
    strat = pd.Series([0.0, 0.02, 0.03, 0.0, -0.01, 0.01, 0.04, 0.0], index=idx)
    ep = crisis_episodes(strat, bench, threshold=0.15)
    assert len(ep) == 2
    first = ep.iloc[0]
    assert (str(first["peak"]), str(first["trough"])) == ("2020-01-01", "2020-01-03")
    assert first["bench_return"] == pytest.approx(0.81 - 1)
    assert first["strategy_return"] == pytest.approx(0.05)
    assert ep.iloc[1]["recovered"] is None


def test_smile_and_betas_on_constructed_series():
    idx = pd.bdate_range("2015-01-01", periods=TRADING_DAYS * 6)
    rng = np.random.default_rng(5)
    bench = pd.Series(rng.normal(0, 0.012, len(idx)), index=idx)
    monthly = bench.groupby(idx.to_period("M")).transform("sum")
    strat = bench.abs() * 0.5 + np.where(monthly < 0, -bench, bench) * 0.5   # 理想的“跨式”形状
    sm = smile(strat, bench, "M")
    assert sm["strategy_avg"].iloc[0] > sm["strategy_avg"].iloc[2] < sm["strategy_avg"].iloc[4]
    b = up_down_beta(strat, bench, "M")
    assert b["beta_down"] < 0 < b["beta_up"]
