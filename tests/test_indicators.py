from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trend_signals import Params, compute_indicators, load_csv
from trend_signals.indicators import angle, atr, bollinger, kama

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "IF_main_digitized.csv"


def naive_kama(c, n=10, fast=2, slow=30):
    fsc, ssc = 2 / (fast + 1), 2 / (slow + 1)
    out = [np.nan] * len(c)
    out[n - 1] = c[n - 1]
    for i in range(n, len(c)):
        vol = sum(abs(c[j] - c[j - 1]) for j in range(i - n + 1, i + 1))
        er = abs(c[i] - c[i - n]) / vol if vol else 0.0
        sc = (er * (fsc - ssc) + ssc) ** 2
        out[i] = out[i - 1] + sc * (c[i] - out[i - 1])
    return out


def test_kama_matches_reference_formula():
    rng = np.random.default_rng(0)
    c = 4000 + rng.normal(0, 20, 200).cumsum()
    np.testing.assert_allclose(kama(pd.Series(c)).to_numpy(), naive_kama(list(c)), equal_nan=True)


def test_kama_on_straight_line_uses_fast_constant():
    c = pd.Series(np.arange(30, dtype=float))
    k = kama(c)
    # ER=1 时 SC=(2/3)^2
    assert k.iloc[10] == pytest.approx(9 + (2 / 3) ** 2 * (10 - 9))


def test_kama_flat_series_stays_flat():
    k = kama(pd.Series([100.0] * 40))
    assert (k.dropna() == 100.0).all()


def test_bollinger_uses_sample_std():
    c = pd.Series(np.arange(1, 21, dtype=float))
    mid, up, lo = bollinger(c, 20, 2)
    assert mid.iloc[-1] == pytest.approx(10.5)
    assert up.iloc[-1] - mid.iloc[-1] == pytest.approx(2 * np.std(np.arange(1, 21), ddof=1))
    assert mid.iloc[-1] - lo.iloc[-1] == pytest.approx(2 * np.std(np.arange(1, 21), ddof=1))


def test_atr_is_simple_average_of_true_range():
    high = pd.Series([10.0, 12, 11, 13])
    low = pd.Series([9.0, 10, 8, 12])
    close = pd.Series([9.5, 11, 9, 12.5])
    # TR: 1, max(2, 2.5, 0.5)=2.5, max(3, 0, 3)=3, max(1, 4, 3)=4
    assert atr(high, low, close, 2).tolist()[1:] == [1.75, 2.75, 3.5]


def test_angle_in_degrees():
    assert angle(pd.Series([0.0, 1.0, -1.0])).tolist() == pytest.approx([0, 45, -45])


def test_matches_values_shown_in_screenshot_header():
    """截图表头（2022-07-08 那根K线）显示的指标值。样本是从截图像素读出来的，有几点误差；
    KAMA 因为缺少 2022-03 之前的预热数据，误差略大。"""
    ind = compute_indicators(load_csv(SAMPLE), Params()).loc["2022-07-08"]
    assert ind["boll"] == pytest.approx(4168.7, abs=6)
    assert ind["upper"] == pytest.approx(4368.9, abs=6)
    assert ind["lower"] == pytest.approx(3968.6, abs=6)
    assert ind["mid"] == pytest.approx(4241.38, abs=6)
    assert ind["kama"] == pytest.approx(4191.93, abs=15)
    assert ind["bb_width_angle"] == pytest.approx(-75.29, abs=3)
    assert ind["atr_angle"] == pytest.approx(-69.71, abs=6)
