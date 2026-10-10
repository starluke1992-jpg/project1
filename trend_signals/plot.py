"""按截图样式画图：主图 BOLL + 信号，副图 KAMA/MID、BB_Width_Angle、ATR_Angle、DIFF_MK。"""
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .strategy import CLOSE_LONG, CLOSE_SHORT, LABELS, OPEN_LONG, OPEN_SHORT  # noqa: E402

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC", "Noto Sans CJK SC",
    "Source Han Sans SC", "WenQuanYi Zen Hei", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

BG, GRID, TEXT = "#13161a", "#37373c", "#c0c0c0"
UP, DOWN = "#ca0000", "#3acccc"
ORANGE, MAGENTA, WHITE, YELLOW = "#fe9b00", "#ff00ff", "#ffffff", "#ffff00"
SIGNAL_TEXT, SHORT_ARROW, LONG_ARROW = "#1582bb", "#428e6c", "#dc3023"


def _candles(ax, df, x):
    up = df["close"] >= df["open"]
    colors = np.where(up, UP, DOWN)
    ax.vlines(x, df["low"], df["high"], colors=colors, linewidth=0.8)
    body = (df["close"] - df["open"]).abs().clip(lower=(df["high"] - df["low"]).max() * 0.002)
    ax.bar(x, body, bottom=df[["open", "close"]].min(axis=1), width=0.6, color=colors, linewidth=0)


def _histogram(ax, x, values, title):
    v = values.fillna(0)
    ax.bar(x, v, width=0.35, color=np.where(v >= 0, "#ff0000", "#00ff00"))
    ax.plot(x, v, color=YELLOW, linewidth=0.8)
    ax.axhline(0, color=WHITE, linewidth=0.6)
    ax.set_ylim(-95, 95)
    _title(ax, f"{title}: {values.iloc[-1]:.2f}", YELLOW)


def _title(ax, text, color=TEXT):
    ax.text(0.003, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=9, color=color)


def _style(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linestyle=":", linewidth=0.6)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.yaxis.tick_right()
    for s in ax.spines.values():
        s.set_color(GRID)


def plot_signals(sig: pd.DataFrame, path: str, title: str = "") -> None:
    x = np.arange(len(sig))
    fig, axes = plt.subplots(5, 1, figsize=(19, 9), sharex=True, facecolor=BG,
                             gridspec_kw={"height_ratios": [4, 2, 1.5, 1.2, 1.6], "hspace": 0.06})
    for ax in axes:
        _style(ax)
    a1, a2, a3, a4, a5 = axes

    _candles(a1, sig, x)
    a1.plot(x, sig["boll"], color=WHITE, linewidth=0.9)
    a1.plot(x, sig["upper"], color=ORANGE, linewidth=0.9)
    a1.plot(x, sig["lower"], color=MAGENTA, linewidth=0.9)
    last = sig.iloc[-1]
    _title(a1, f"{title}  BOLL(20,2) BOLL: {last['boll']:.1f}  UPPER: {last['upper']:.1f}  LOWER: {last['lower']:.1f}")
    gap = _mark_signals(a1, sig, x)
    a1.set_ylim(sig[["low", "lower"]].min().min() - gap * 5, sig[["high", "upper"]].max().max() + gap * 9)

    _candles(a2, sig, x)
    a2.plot(x, sig["kama"], color=ORANGE, linewidth=0.9)
    a2.plot(x, sig["mid"], color=YELLOW, linewidth=0.9)
    _title(a2, f"KAMA: {last['kama']:.2f}  MID: {last['mid']:.2f}", ORANGE)

    _histogram(a3, x, sig["bb_width_angle"], "BB_Width_Angle")
    _histogram(a4, x, sig["atr_angle"], "ATR_Angle")

    a5.plot(x, sig["diff_mk"], color=YELLOW, linewidth=0.9)
    a5.axhline(0, color=ORANGE, linewidth=0.9)
    _title(a5, f"DIFF_MK: {last['diff_mk']:.2f}", YELLOW)

    months = pd.Series(sig.index.to_period("M"))
    ticks = np.flatnonzero(months.ne(months.shift()).to_numpy())
    a5.set_xticks(ticks)
    a5.set_xticklabels([sig.index[i].strftime("%y-%m") for i in ticks])
    a5.set_xlim(-1, len(sig))

    fig.savefig(path, dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def _mark_signals(ax, sig, x):
    """画信号，返回标注用的间距，便于调整纵轴范围。"""
    span = sig["high"].max() - sig["low"].min()
    gap = span * 0.03
    for i in np.flatnonzero(sig["signal"].to_numpy() != ""):
        s, row = sig["signal"].iloc[i], sig.iloc[i]
        if s == OPEN_SHORT:
            top = max(row["high"], row["upper"]) + gap
            ax.annotate("", xy=(x[i], top), xytext=(x[i], top + gap * 1.5),
                        arrowprops=dict(color=SHORT_ARROW, width=4, headwidth=10, headlength=6))
            ax.text(x[i], top + gap * 1.8, LABELS[s], color=SIGNAL_TEXT, ha="center", va="bottom", fontsize=9)
        elif s == OPEN_LONG:
            bottom = min(row["low"], row["lower"]) - gap
            ax.annotate("", xy=(x[i], bottom), xytext=(x[i], bottom - gap * 1.5),
                        arrowprops=dict(color=LONG_ARROW, width=4, headwidth=10, headlength=6))
            ax.text(x[i], bottom - gap * 1.8, LABELS[s], color=SIGNAL_TEXT, ha="center", va="top", fontsize=9)
        elif s in (CLOSE_LONG, CLOSE_SHORT):
            ax.text(x[i], row["low"] - gap * 0.5, LABELS[s], color=SIGNAL_TEXT, ha="center", va="top", fontsize=9)
    return gap
