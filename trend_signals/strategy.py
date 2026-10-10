"""开多 / 开空 / 平仓 信号。

开仓要求三类因子同时共振（空仓时才开）：
  趋势    收盘价 > BOLL 中轨（多） / < 中轨（空）
  动量    DIFF_MK > +阈值（多） / < -阈值（空），阈值 = momentum_atr_mult * ATR
  波动率  BB_Width_Angle > 0 且 ATR_Angle > 0（布林带张口、ATR 放大，即行情正在展开）

平仓（任一条件触发）：
  1. 波动率收敛（BB_Width_Angle < 0 且 ATR_Angle < 0）并且收盘价反向穿过 KAMA；
  2. 移动止损（吊灯止损）：多单收盘 < 最近 trail_lookback 根K线最高价 - trail_atr_mult * ATR，
     空单收盘 > 最近 trail_lookback 根K线最低价 + trail_atr_mult * ATR。trail_atr_mult 设为 0 可关闭。
     用固定回看窗口而不是“开仓以来”，是为了让平仓条件不依赖持仓状态，
     这样通达信公式（没有循环、不能记住持仓）也能算出完全相同的信号；两种写法在截图上的吻合度一样。

信号在K线收盘时确定，按收盘价成交（与截图标注一致）；实盘中应在下一根K线执行。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import compute_indicators

OPEN_LONG, OPEN_SHORT, CLOSE_LONG, CLOSE_SHORT = "open_long", "open_short", "close_long", "close_short"
LABELS = {OPEN_LONG: "开多", OPEN_SHORT: "开空", CLOSE_LONG: "平仓", CLOSE_SHORT: "平仓"}


@dataclass(frozen=True)
class Params:
    boll_n: int = 20
    boll_k: float = 2.0
    kama_n: int = 10
    kama_fast: int = 2
    kama_slow: int = 30
    mid_n: int = 5
    atr_n: int = 20
    # DIFF_MK 在零轴附近时不算有动量；截图里 |DIFF_MK| 很小时出现了四因子同向却未开仓的情况
    momentum_atr_mult: float = 0.2
    # 移动止损（原文：“以波动幅度为核心的跟踪最高价的止盈止损”）；trail_atr_mult=0 表示不用
    trail_atr_mult: float = 2.5
    trail_lookback: int = 10


def generate_signals(df: pd.DataFrame, params: Params = Params()) -> pd.DataFrame:
    """返回指标列 + signal（open_long/open_short/close_long/close_short 或空串）+ position（1/0/-1）。"""
    close = df["close"]
    ind = compute_indicators(df, params)

    vol_expanding = (ind["bb_width_angle"] > 0) & (ind["atr_angle"] > 0)
    vol_contracting = (ind["bb_width_angle"] < 0) & (ind["atr_angle"] < 0)
    threshold = params.momentum_atr_mult * ind["atr"]

    long_entry = ((close > ind["boll"]) & (ind["diff_mk"] > threshold) & vol_expanding).to_numpy()
    short_entry = ((close < ind["boll"]) & (ind["diff_mk"] < -threshold) & vol_expanding).to_numpy()
    long_exit = vol_contracting & (close < ind["kama"])
    short_exit = vol_contracting & (close > ind["kama"])
    if params.trail_atr_mult > 0:
        stop_dist = params.trail_atr_mult * ind["atr"]
        long_exit |= close < df["high"].rolling(params.trail_lookback).max() - stop_dist
        short_exit |= close > df["low"].rolling(params.trail_lookback).min() + stop_dist
    long_exit, short_exit = long_exit.to_numpy(), short_exit.to_numpy()

    signal = np.full(len(df), "", dtype=object)
    position = np.zeros(len(df), dtype=int)
    pos = 0
    for i in range(len(df)):
        if pos == 1:
            if long_exit[i]:
                pos, signal[i] = 0, CLOSE_LONG
        elif pos == -1:
            if short_exit[i]:
                pos, signal[i] = 0, CLOSE_SHORT
        elif long_entry[i]:
            pos, signal[i] = 1, OPEN_LONG
        elif short_entry[i]:
            pos, signal[i] = -1, OPEN_SHORT
        position[i] = pos

    out = df.join(ind)
    out["signal"] = signal
    out["position"] = position
    return out


def trades(sig: pd.DataFrame) -> pd.DataFrame:
    """把信号配对成交易。最后一笔若未平仓，按最后收盘价计算浮动盈亏，closed=False。"""
    rows, entry = [], None
    for date, row in sig[sig["signal"] != ""].iterrows():
        if row["signal"] in (OPEN_LONG, OPEN_SHORT):
            entry = (date, row["close"], 1 if row["signal"] == OPEN_LONG else -1)
        elif entry is not None:
            rows.append(_trade(entry, date, row["close"], True, sig))
            entry = None
    if entry is not None:
        rows.append(_trade(entry, sig.index[-1], sig["close"].iloc[-1], False, sig))
    cols = ["entry_date", "exit_date", "side", "entry_price", "exit_price", "points", "bars", "closed"]
    return pd.DataFrame(rows, columns=cols)


def _trade(entry, exit_date, exit_price, closed, sig):
    entry_date, entry_price, side = entry
    return {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "side": "long" if side == 1 else "short",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "points": side * (exit_price - entry_price),
        "bars": sig.index.get_loc(exit_date) - sig.index.get_loc(entry_date),
        "closed": closed,
    }


def summary(tr: pd.DataFrame, multiplier: float = 300) -> dict:
    """multiplier：合约乘数，沪深300股指期货 IF 为每点 300 元。"""
    done = tr[tr["closed"]]
    if done.empty:
        return {"trades": 0}
    return {
        "trades": len(done),
        "win_rate": round(float((done["points"] > 0).mean()), 3),
        "total_points": round(float(done["points"].sum()), 1),
        "avg_points": round(float(done["points"].mean()), 1),
        "total_pnl_per_lot": round(float(done["points"].sum() * multiplier)),
    }
