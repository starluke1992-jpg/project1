from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trend_signals import Params, generate_signals, load_csv, summary, trades
from trend_signals.strategy import CLOSE_LONG, CLOSE_SHORT, OPEN_LONG, OPEN_SHORT

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "IF_main_digitized.csv"

# 截图上标注的信号（逐个读取箭头/文字位置得到）
CHART_SIGNALS = [
    ("2022-04-11", "开空"), ("2022-04-29", "平仓"), ("2022-05-31", "开多"), ("2022-07-08", "平仓"),
    ("2022-07-15", "开空"), ("2022-08-15", "平仓"), ("2022-08-24", "开空"), ("2022-10-13", "平仓"),
    ("2022-10-24", "开空"), ("2022-11-03", "平仓"), ("2022-11-08", "开多"), ("2022-12-20", "平仓"),
    ("2023-01-05", "开多"), ("2023-02-06", "平仓"), ("2023-02-17", "开空"), ("2023-03-24", "平仓"),
    ("2023-04-24", "开空"), ("2023-05-09", "平仓"), ("2023-05-12", "开空"), ("2023-06-15", "平仓"),
]
KIND = {OPEN_LONG: "开多", OPEN_SHORT: "开空", CLOSE_LONG: "平仓", CLOSE_SHORT: "平仓"}


@pytest.fixture(scope="module")
def sample():
    return load_csv(SAMPLE)


@pytest.fixture(scope="module")
def sig(sample):
    return generate_signals(sample)


def random_walk(n=600, seed=1):
    rng = np.random.default_rng(seed)
    close = 4000 + rng.normal(0, 30, n).cumsum()
    open_ = close + rng.normal(0, 10, n)
    high = np.maximum(open_, close) + rng.uniform(0, 25, n)
    low = np.minimum(open_, close) - rng.uniform(0, 25, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_reproduces_chart_signals(sample, sig):
    pos = {d: i for i, d in enumerate(sample.index.strftime("%Y-%m-%d"))}
    ours = [(i, KIND[s]) for i, s in enumerate(sig["signal"]) if s]
    used, hit = set(), 0
    for date, kind in CHART_SIGNALS:
        for j, (i, k) in enumerate(ours):
            if j not in used and k == kind and abs(i - pos[date]) <= 1:
                used.add(j)
                hit += 1
                break
    assert hit >= 16, f"只对上 {hit}/20 个信号"


def test_position_matches_chart_most_of_the_time(sample, sig):
    chart_pos = pd.Series(0, index=sample.index)
    side = 0
    for date, kind in CHART_SIGNALS:
        side = {"开多": 1, "开空": -1, "平仓": 0}[kind]
        chart_pos[date:] = side
    agree = (chart_pos == sig["position"]).loc["2022-04-01":"2023-06-15"].mean()
    assert agree >= 0.9


def test_signals_alternate_open_close():
    s = generate_signals(random_walk())["signal"]
    seq = [x for x in s if x]
    assert len(seq) > 10
    for a, b in zip(seq, seq[1:]):
        assert (a in (OPEN_LONG, OPEN_SHORT)) != (b in (OPEN_LONG, OPEN_SHORT))
    assert all(
        (a, b) in {(OPEN_LONG, CLOSE_LONG), (OPEN_SHORT, CLOSE_SHORT)}
        for a, b in zip(seq[::2], seq[1::2])
    )


def test_signal_bars_satisfy_their_conditions():
    s = generate_signals(random_walk())
    vol_up = (s["bb_width_angle"] > 0) & (s["atr_angle"] > 0)
    vol_down = (s["bb_width_angle"] < 0) & (s["atr_angle"] < 0)
    th = 0.2 * s["atr"]
    ol, os_ = s[s["signal"] == OPEN_LONG], s[s["signal"] == OPEN_SHORT]
    assert ((ol["close"] > ol["boll"]) & (ol["diff_mk"] > th[ol.index]) & vol_up[ol.index]).all()
    assert ((os_["close"] < os_["boll"]) & (os_["diff_mk"] < -th[os_.index]) & vol_up[os_.index]).all()
    # 平仓：核心条件，或最近 10 根最高价/最低价 ∓ 2.5*ATR 的移动止损
    hh = s["high"].rolling(10).max() - 2.5 * s["atr"]
    ll = s["low"].rolling(10).min() + 2.5 * s["atr"]
    cl, cs = s[s["signal"] == CLOSE_LONG], s[s["signal"] == CLOSE_SHORT]
    assert (((cl["close"] < cl["kama"]) & vol_down[cl.index]) | (cl["close"] < hh[cl.index])).all()
    assert (((cs["close"] > cs["kama"]) & vol_down[cs.index]) | (cs["close"] > ll[cs.index])).all()
    assert len(cl) + len(cs) > 5


def test_trailing_stop_can_be_disabled(sample):
    off = generate_signals(sample, Params(trail_atr_mult=0))
    on = generate_signals(sample)
    assert (on["signal"] != off["signal"]).any()
    # 关掉移动止损后，每个平仓都只能由核心条件触发
    vol_down = (off["bb_width_angle"] < 0) & (off["atr_angle"] < 0)
    cl = off[off["signal"] == CLOSE_LONG]
    cs = off[off["signal"] == CLOSE_SHORT]
    assert ((cl["close"] < cl["kama"]) & vol_down[cl.index]).all()
    assert ((cs["close"] > cs["kama"]) & vol_down[cs.index]).all()


def test_no_lookahead():
    """用截至某天的数据算出的信号，必须和用全部数据算出的该天之前的信号一致。"""
    df = random_walk()
    full = generate_signals(df)
    for cut in (150, 333, 480):
        part = generate_signals(df.iloc[:cut])
        pd.testing.assert_series_equal(part["signal"], full["signal"].iloc[:cut])


def test_trades_and_summary(sig):
    tr = trades(sig)
    assert tr["closed"].iloc[:-1].all()
    first = tr.iloc[0]
    assert (first["side"], str(first["entry_date"].date())) == ("short", "2022-04-11")
    assert first["points"] == pytest.approx(first["entry_price"] - first["exit_price"])
    s = summary(tr)
    assert s["trades"] == int(tr["closed"].sum())
    assert s["total_pnl_per_lot"] == pytest.approx(s["total_points"] * 300, abs=300)
