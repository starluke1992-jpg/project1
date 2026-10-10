"""方案 B：换月与复权连续、品种池、信号无未来函数、B2 仓位。"""
import numpy as np
import pandas as pd
import pytest

from futures.continuous import build_continuous, delivery_month
from futures.signals import adaptive_lowlag, tsmom_ensemble, tsmom_12m
from futures.strategies import B2Config, b2_strategy, cap_sector_risk
from futures.universe import eligibility
from research.synthetic import synthetic_futures


def test_delivery_month_parsing():
    ref = pd.Timestamp("2024-01-10")
    assert delivery_month("CU2402", ref) == pd.Timestamp("2024-02-01")
    assert delivery_month("TA405", ref) == pd.Timestamp("2024-05-01")
    assert delivery_month("TA405", pd.Timestamp("2014-01-10")) == pd.Timestamp("2014-05-01")
    assert delivery_month("TA501", pd.Timestamp("2024-11-01")) == pd.Timestamp("2025-01-01")
    assert pd.isna(delivery_month("XYZ", ref))


def two_contracts():
    """CU2403 和 CU2405：1 月中旬起 2405 的持仓量超过 2403；2405 价格比 2403 高 100。"""
    days = pd.bdate_range("2024-01-02", "2024-02-29")
    rows = []
    for i, d in enumerate(days):
        base = 70000 + 10 * i
        rows.append({"date": d, "symbol": "CU2403", "variety": "CU", "open": base, "high": base + 50,
                     "low": base - 50, "close": base, "volume": 1000, "open_interest": 1000 - 20 * i})
        rows.append({"date": d, "symbol": "CU2405", "variety": "CU", "open": base + 100, "high": base + 150,
                     "low": base + 50, "close": base + 100, "volume": 1000, "open_interest": 500 + 20 * i})
    return pd.DataFrame(rows)


def test_continuous_rolls_forward_without_gap():
    c = two_contracts()
    cont = build_continuous(c)
    assert cont["roll"].sum() == 1
    roll_day = cont.index[cont["roll"]][0]
    assert cont.loc[roll_day, "contract"] == "CU2403"          # 换月当天的收益算在旧合约上
    after = cont.index[cont.index.get_loc(roll_day) + 1]
    assert cont.loc[after, "contract"] == "CU2405"
    # 两个合约每天都涨 10 元，换月不应该产生 100 元的跳空
    jumps = cont["close"].pct_change().abs()
    assert jumps.max() < 0.001
    assert cont.loc[after, "ret"] == pytest.approx(10 / (70000 + 10 * c["date"].drop_duplicates()
                                                         .tolist().index(roll_day) + 100))


def test_continuous_respects_last_hold_date():
    c = two_contracts()
    c.loc[c["symbol"] == "CU2405", "open_interest"] = 1  # 2405 持仓一直很小
    cont = build_continuous(c)
    # 2403 的最后持有日是 2 月 14 日，之后必须换到 2405
    assert set(cont.loc["2024-02-16":, "contract"]) == {"CU2405"}


def test_continuous_no_lookahead():
    c = two_contracts()
    full = build_continuous(c)
    cut = pd.Timestamp("2024-01-31")
    part = build_continuous(c[c["date"] <= cut])
    pd.testing.assert_series_equal(full.loc[:cut, "contract"], part["contract"])
    pd.testing.assert_series_equal(full.loc[:cut, "ret"], part["ret"])


@pytest.mark.parametrize("fn", [tsmom_ensemble, tsmom_12m, adaptive_lowlag])
def test_signals_no_lookahead(fn):
    fut = synthetic_futures(start="2015-01-01", end="2019-12-31")
    prices = fut.prices[["CU", "RB", "M"]]
    full = fn(prices)
    cut = "2018-06-29"
    part = fn(prices.loc[:cut])
    pd.testing.assert_frame_equal(full.loc[:cut], part, check_freq=False)
    vals = full.stack().dropna()
    assert vals.between(-1, 1).all()


def test_eligibility_min_days():
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame({"a": 1.0, "b": np.nan}, index=idx)
    prices.loc[idx[100]:, "b"] = 1.0
    ok = eligibility(prices, min_days=250)
    assert not ok["a"].iloc[248] and ok["a"].iloc[249]
    assert not ok["b"].iloc[-1]


def test_cap_sector_risk():
    raw = pd.Series({"CU": 1.0, "AL": 1.0, "ZN": 1.0, "M": 1.0})
    vols = pd.Series(0.2, index=raw.index)
    sectors = {"CU": "有色", "AL": "有色", "ZN": "有色", "M": "农产品"}
    w = cap_sector_risk(raw, vols, sectors, cap=0.6)
    risk = (w.abs() * vols).groupby(lambda a: sectors[a]).sum()
    assert (risk / risk.sum()).max() <= 0.6 + 1e-6


def test_b2_targets_vol_and_gross():
    fut = synthetic_futures(start="2015-01-01", end="2018-12-31")
    sig = tsmom_ensemble(fut.prices)
    elig = eligibility(fut.prices, fut.volumes)
    cfg = B2Config()
    fn = b2_strategy(sig, elig, {v: "x" + str(i % 4) for i, v in enumerate(fut.prices.columns)}, cfg)

    class Ctx:
        date = pd.Timestamp("2018-06-29")
        excess = fut.excess.loc[:"2018-06-29"]

    w, vol = fn(Ctx)
    assert w.abs().sum() <= cfg.max_gross + 1e-9
    assert vol == pytest.approx(cfg.target_vol) or w.abs().sum() == pytest.approx(cfg.max_gross)
