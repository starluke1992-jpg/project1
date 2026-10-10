"""回测图：累计收益与基准（标出基准大幅回撤区间）、各资产类别贡献、季度“微笑”。"""
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import crisis_episodes, smile  # noqa: E402
from .instruments import ASSET_CLASSES  # noqa: E402

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC", "Noto Sans CJK SC",
    "Source Han Sans SC", "WenQuanYi Zen Hei", "Arial Unicode MS", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def plot_backtest(res, bench: pd.Series, path: str, title: str = "", threshold: float = 0.15) -> None:
    strategy, bench = res.returns.align(bench, join="inner")
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [3, 2, 2]})

    a1.plot(strategy.cumsum(), label="策略累计收益（固定资金）", color="#1f5fbf", linewidth=1.4)
    a1.plot((1 + bench).cumprod() - 1, label="基准", color="#888888", linewidth=1.0)
    for _, ep in crisis_episodes(strategy, bench, threshold).iterrows():
        a1.axvspan(pd.Timestamp(ep["peak"]), pd.Timestamp(ep["trough"]), color="#d9534f", alpha=0.12)
    a1.axhline(0, color="black", linewidth=0.6)
    a1.set_title(f"{title}  红色区间：基准回撤超过 {threshold:.0%}")
    a1.legend(loc="upper left")
    a1.grid(alpha=0.3)

    cls = res.class_returns().loc[strategy.index].cumsum()
    for k in cls.columns:
        a2.plot(cls[k], label=ASSET_CLASSES.get(k, k), linewidth=1.0)
    a2.axhline(0, color="black", linewidth=0.6)
    a2.set_title("各资产类别累计贡献（扣成本）")
    a2.legend(loc="upper left", ncol=4, fontsize=8)
    a2.grid(alpha=0.3)

    sm = smile(strategy, bench, "Q")
    labels = [f"第{i}组\n基准 {b:+.1%}" for i, b in zip(sm.index, sm["bench_avg"])]
    colors = ["#2e8b57" if v >= 0 else "#d9534f" for v in sm["strategy_avg"]]
    a3.bar(labels, sm["strategy_avg"], color=colors)
    a3.axhline(0, color="black", linewidth=0.6)
    a3.set_title("按基准季度收益分五组：策略平均季度收益（危机 Alpha 要求第 1 组为正）")
    a3.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
