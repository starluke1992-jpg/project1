"""把回测结果写成 Markdown 报告和净值/回撤图。"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import comparison_table, drawdown_table, period_slice, yearly_returns

# 分类色板（固定顺序，已通过色盲可分辨性校验），文字用中性色
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

PCT_COLS = {"年化收益", "年化波动", "最大回撤", "月胜率", "年成本", "事前波动(均值)"}


def _fmt(v, col):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if col in PCT_COLS:
        return f"{v:.2%}"
    return f"{v:.2f}"


def md_table(df: pd.DataFrame, index_name: str = "") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([index_name] + [str(c) for c in cols]) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [_fmt(row[c], c) for c in cols]) + " |")
    return "\n".join(lines)


def plot_results(results: list, period: str, path: Path, title: str) -> None:
    import logging

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "WenQuanYi Zen Hei Mono",
                                       "SimHei", "Microsoft YaHei", "PingFang SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]}, facecolor=SURFACE)
    for i, res in enumerate(results[: len(SERIES)]):
        r = period_slice(res.returns, period)
        nav = (1 + r).cumprod()
        dd = nav / nav.cummax() - 1
        ax1.plot(nav.index, nav, color=SERIES[i], lw=1.6, label=res.name)
        ax2.plot(dd.index, dd, color=SERIES[i], lw=1.2)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.set_ylabel("净值（对数刻度）", color=INK_2)
    ax2.set_ylabel("回撤", color=INK_2)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.tick_params(colors=INK_2, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    ax1.set_title(title, color=INK, fontsize=12, loc="left")
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def write_report(plan: str, results: list, period: str, out_dir: Path, synthetic: bool,
                 notes: list = None, extra_sections: list = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{plan}_{period}"
    png = out_dir / f"{stem}.png"
    title = f"方案 {plan} · {({'dev': '开发期', 'holdout': '留出期', 'full': '全样本'})[period]}"
    if synthetic:
        title += "（合成数据）"
    plot_results(results, period, png, title)

    table = comparison_table(results, period)
    years = pd.DataFrame({res.name: yearly_returns(period_slice(res.returns, period)) for res in results})
    years.index.name = "年份"
    lines = [f"# {title}", ""]
    if synthetic:
        lines += ["> **这是合成数据的运行结果，只用来验证代码能跑通，不代表任何真实市场表现，不能据此做任何判断。**", ""]
    lines += [f"生成时间：{datetime.now():%Y-%m-%d %H:%M}", "", f"![净值与回撤]({png.name})", "",
              "## 绩效对比", "", md_table(table, "策略"), "",
              "夏普 = 日超额收益（组合 − 现金）均值 / 日收益标准差 × √252。年换手、年成本按单边计。", "",
              "## 年度收益", "", md_table(years.T.rename(columns=str).map(lambda v: f"{v:.2%}" if pd.notna(v) else "—"), "策略"), ""]
    main = results[min(len(results) - 1, 2)] if plan == "A" else results[0]
    dd = drawdown_table(period_slice(main.returns, period))
    if len(dd):
        dd = dd.assign(**{c: dd[c].map(lambda d: f"{d:%Y-%m-%d}" if pd.notna(d) else "未恢复")
                          for c in ("开始", "谷底", "恢复")})
        dd["深度"] = dd["深度"].map(lambda v: f"{v:.2%}")
        lines += [f"## 最深的几次回撤：{main.name}", "", md_table(dd.set_index("开始"), "开始"), ""]
    for sec in extra_sections or []:
        lines += [sec, ""]
    if notes:
        lines += ["## 数据说明", ""] + [f"- {n}" for n in notes] + [""]
    path = out_dir / f"{stem}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
