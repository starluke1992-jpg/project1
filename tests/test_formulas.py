"""用解释器运行 formulas/ 里的通达信、文华公式，检查与 Python 版逐K线一致。"""
from pathlib import Path

import numpy as np
import pytest

from formula_interpreter import Formula, sections
from test_strategy import random_walk
from trend_signals import Params, generate_signals, load_csv

ROOT = Path(__file__).resolve().parents[1]
FORMULAS = ROOT / "formulas"


def read(name):
    return (FORMULAS / name).read_text(encoding="utf-8")


DATASETS = {
    "IF样本": lambda: load_csv(ROOT / "data" / "IF_main_digitized.csv"),
    # 1500 根，超过通达信状态合成的 256 根窗口，覆盖“窗口外按空仓起算”的情形
    "随机1": lambda: random_walk(n=1500, seed=11),
    "随机2": lambda: random_walk(n=1500, seed=12),
    "随机3": lambda: random_walk(n=1500, seed=13),
}


@pytest.fixture(scope="module", params=list(DATASETS))
def case(request):
    df = DATASETS[request.param]()
    return df, generate_signals(df)


def test_tdx_main_positions_and_signals_match(case):
    df, sig = case
    f = Formula(df).run(read("tdx_main.txt"))
    np.testing.assert_array_equal(f.vars["HOLD"], sig["position"].to_numpy())
    marks = {"KD": "open_long", "KK": "open_short"}
    for var, kind in marks.items():
        drawn = np.nan_to_num(f.vars[var]) != 0
        np.testing.assert_array_equal(drawn, (sig["signal"] == kind).to_numpy(), err_msg=var)
    closed = np.nan_to_num(f.vars["PC"]) != 0
    np.testing.assert_array_equal(closed, sig["signal"].str.startswith("close").to_numpy())


def test_wenhua_model_positions_match(case):
    df, sig = case
    f = Formula(df).run(read("wenhua_model.txt"))
    assert f.autofilter and [a for _, a in f.signals] == ["BK", "SK", "SP", "BP"]
    np.testing.assert_array_equal(f.positions(), sig["position"].to_numpy())


@pytest.mark.parametrize("name", ["tdx_main.txt", "wenhua_model.txt"])
def test_main_chart_lines_match(case, name):
    df, sig = case
    f = Formula(df).run(read(name))
    for var, col in [("BOLL", "boll"), ("UPPER", "upper"), ("LOWER", "lower"), ("KAMA", "kama")]:
        ok = sig[col].notna().to_numpy()
        np.testing.assert_allclose(f.vars[var][ok], sig[col].to_numpy()[ok], atol=0.01, err_msg=var)


@pytest.mark.parametrize("name", ["tdx_sub.txt", "wenhua_sub.txt"])
def test_sub_chart_indicators_match(case, name):
    df, sig = case
    parts = sections(read(name))
    assert len(parts) == 4
    expect = [("KAMA", "kama"), ("MID", "mid"), ("BBWA", "bb_width_angle"),
              ("ATRA", "atr_angle"), ("DIFFMK", "diff_mk")]
    got = {}
    for text in parts:
        got.update(Formula(df).run(text).vars)
    for var, col in expect:
        ok = sig[col].notna().to_numpy()
        np.testing.assert_allclose(got[var][ok], sig[col].to_numpy()[ok], atol=0.01, err_msg=f"{name}:{var}")


def test_formula_parameters_match_python_defaults():
    p = Params()
    expected = {"BOLLN": p.boll_n, "BOLLK": p.boll_k, "ERN": p.kama_n, "FASTN": p.kama_fast,
                "SLOWN": p.kama_slow, "MIDN": p.mid_n, "ATRN": p.atr_n, "MOMK": p.momentum_atr_mult,
                "STOPK": p.trail_atr_mult, "STOPN": p.trail_lookback}
    for name in ("tdx_main.txt", "wenhua_model.txt"):
        f = Formula(random_walk(n=50)).run(read(name))
        assert {k: f.vars[k][0] for k in expected} == pytest.approx(expected), name
