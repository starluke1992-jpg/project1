"""回测入口。

  python -m research run A                    # 方案 A：三个基准 + v1 + 挑战者，开发期
  python -m research run B1                   # 杠杆风险平价（带/不带趋势过滤）
  python -m research run B2                   # 趋势 CTA：v1 + 对照 + 挑战者
  python -m research run B                    # B1 + B2 + 60/40 组合
  python -m research run A --synthetic        # 用合成数据验证代码能跑通（结果没有任何意义）
  python -m research run A --period holdout --confirm-holdout   # 留出期：规则冻结后只跑一次

真实数据来自 data/cache/（先运行 python -m datasource universe；B 还需要 python -m futures build）。
每次用真实数据运行都会登记到 research/trials.csv。报告写到 reports/（合成数据写到 reports/synthetic/）。
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from allocation.backtest import month_ends, run_backtest, week_ends
from allocation.data import build_panel
from allocation.strategies import plan_a_strategies
from datasource.cache import DEFAULT_ROOT, SeriesCache
from futures.data import load_continuous, to_panel
from futures.sectors import B1_CLASS, sector_of
from futures.signals import SIGNALS, repo_signal
from futures.strategies import B2Config, b1_strategies, b2_strategy
from futures.universe import eligibility

from .metrics import comparison_table, result_summary
from .report import md_table, write_report
from .trials import count_trials, log_trial

ROOT = Path(__file__).resolve().parents[1]
A_COSTS_BPS = {"equity": 3.5, "bond": 3.5, "gold": 3.5, "commodity": 6.5}   # 佣金万1.5 + 滑点
A_QDII = {"spx": 6.5}
B1_COMMODITIES = ["CU", "AL", "RB", "I", "M", "Y", "TA", "MA", "SC"]
B_SECTOR_COSTS = {"股指": 2.0, "利率": 1.0}                                 # 其余板块 4bp
B_DEFAULT_COST = 4.0


def _load_a(synthetic: bool):
    if synthetic:
        from .synthetic import synthetic_panel
        return synthetic_panel()
    return build_panel(cache=SeriesCache(DEFAULT_ROOT))


def _load_futures(synthetic: bool):
    if synthetic:
        from .synthetic import synthetic_futures
        return synthetic_futures()
    conts = load_continuous(DEFAULT_ROOT)
    return to_panel(conts) if conts else None


def run_a(synthetic: bool):
    panel = _load_a(synthetic)
    costs = pd.Series({a: A_QDII.get(a, A_COSTS_BPS.get(c, 3.5)) for a, c in panel.classes.items()})
    dates = month_ends(panel.total.index)
    results = []
    for name, (fn, params) in plan_a_strategies(panel.classes, panel.assets).items():
        results.append(run_backtest(panel.excess, panel.cash, fn, dates, name=name, cost_bps=costs,
                                    band_abs=0.01, funded=True, params=params))
    order = ["等权", "股六债四", "静态风险平价", "方案A v1（均线+动量过滤）", "挑战者：负向趋势因子（近似）"]
    results.sort(key=lambda r: order.index(r.name) if r.name in order else 99)
    notes = list(panel.notes)
    if panel.origin is not None:
        first_etf = {a: panel.origin.index[panel.origin[a] == "etf"].min() for a in panel.origin.columns}
        notes.append("各资产开始用 ETF 实际数据的日期（之前为代理）：" + "，".join(
            f"{a} {d:%Y-%m-%d}" if pd.notna(d) else f"{a} 全部为代理" for a, d in first_etf.items()))
    return results, notes


def _b1(panel, fut):
    """B1 的资产：股指（沪深300、中证500）、10 年国债、黄金来自方案 A 的面板，
    商品优先用复权连续期货，没有就用方案 A 的商品 ETF 序列。都按超额收益（期货口径）计。"""
    keep = {a: c for a, c in panel.classes.items() if a in ("hs300", "zz500", "cgb10", "gold")}
    excess = panel.excess[list(keep)]
    classes = dict(keep)
    notes = []
    comm = [v for v in B1_COMMODITIES if fut is not None and v in fut.excess.columns]
    if comm:
        excess = excess.join(fut.excess[comm].reindex(excess.index), how="left")
        classes.update({v: "commodity" for v in comm})
    else:
        a_comm = [a for a, c in panel.classes.items() if c == "commodity"]
        excess = excess.join(panel.excess[a_comm])
        classes.update({a: "commodity" for a in a_comm})
        notes.append("没有复权连续期货，B1 的商品部分用的是方案 A 的商品 ETF 序列")
    costs = pd.Series({a: {"equity": 2.0, "bond": 1.0, "gold": 2.0}.get(c, 4.0) for a, c in classes.items()})
    results = [run_backtest(excess, panel.cash, fn, week_ends(excess.index), name=name, cost_bps=costs,
                            band_abs=0.005, funded=False, params=params)
               for name, (fn, params) in b1_strategies(classes).items()]
    return results, notes


def _b2(fut, cash):
    cfg = B2Config()
    sectors = {v: sector_of(v) for v in fut.excess.columns}
    elig = eligibility(fut.prices, fut.volumes, cfg.min_days, cfg.min_volume)
    costs = pd.Series({v: B_SECTOR_COSTS.get(s, B_DEFAULT_COST) for v, s in sectors.items()})
    cash = cash.reindex(fut.excess.index).ffill().fillna(0.0)
    signals = {name: f(fut.prices) for name, f in SIGNALS.items()}
    sig = repo_signal(fut.ohlc)
    if not sig.empty:
        signals["对照：仓库 BOLL+KAMA 信号"] = sig.reindex(fut.prices.index)
    results = []
    for name, s in signals.items():
        fn = b2_strategy(s, elig, sectors, cfg)
        results.append(run_backtest(fut.excess, cash, fn, week_ends(fut.excess.index), name=name,
                                    cost_bps=costs, band_rel=0.1, funded=False, roll_flags=fut.rolls,
                                    params={**cfg.to_dict(), "signal": name}))
    return results


def run_b(which: str, synthetic: bool):
    panel = _load_a(synthetic)
    fut = _load_futures(synthetic)
    results, notes = [], list(panel.notes)
    if which in ("B1", "B"):
        r1, n1 = _b1(panel, fut)
        results += r1
        notes += n1
    if which in ("B2", "B"):
        if fut is None:
            raise SystemExit("没有复权连续期货数据：先运行 python -m futures build（逐合约数据需要在能访问交易所网站的地方取）")
        results += _b2(fut, panel.cash)
    extra = []
    if which == "B":
        b1 = next(r for r in results if r.name.startswith("B1 v1"))
        b2 = next(r for r in results if r.name.startswith("v1"))
        both = pd.concat([b1.returns, b2.returns], axis=1, join="inner").dropna()
        combo = both.iloc[:, 0] * 0.6 + both.iloc[:, 1] * 0.4
        from allocation.backtest import Result
        mix = lambda a, b: (a.reindex(combo.index) * 0.6 + b.reindex(combo.index) * 0.4)  # noqa: E731
        results.append(Result(name="B 组合（B1 60% + B2 40%）", returns=combo, cash=b1.cash.reindex(combo.index),
                              weights=pd.DataFrame(index=combo.index), targets=pd.DataFrame(),
                              turnover=mix(b1.turnover, b2.turnover), costs=mix(b1.costs, b2.costs),
                              exante_vol=pd.Series(dtype=float), params={"B1": b1.name, "B2": b2.name}))
        c = b1.cash.reindex(both.index)
        corr = both.sub(c, axis=0).corr().iloc[0, 1]
        extra.append(f"## B1 与 B2 的相关性\n\n全样本日超额收益相关系数：{corr:.2f}（验收标准 < 0.3）")
        notes.append("组合按每天固定 60/40 计算，是半年再平衡的近似")
    return results, notes, extra


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m research")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="运行回测并写报告")
    p.add_argument("plan", choices=["A", "B1", "B2", "B"])
    p.add_argument("--period", choices=["dev", "holdout", "full"], default="dev")
    p.add_argument("--confirm-holdout", action="store_true", help="确认规则已冻结，才允许跑留出期")
    p.add_argument("--synthetic", action="store_true", help="用合成数据验证代码（结果没有意义，不登记）")
    p.add_argument("--out", default=str(ROOT / "reports"))
    p.add_argument("--note", default="", help="写进试验登记的备注")
    args = ap.parse_args(argv)

    if args.period == "holdout" and not args.confirm_holdout and not args.synthetic:
        print("留出期只能在规则冻结后跑一次。确认要跑，请加 --confirm-holdout（会记入试验登记）。")
        return 2

    try:
        if args.plan == "A":
            results, notes = run_a(args.synthetic)
            extra = []
        else:
            results, notes, extra = run_b(args.plan, args.synthetic)
    except ValueError as e:
        print(f"数据不足：{e}")
        return 1

    table = comparison_table(results, args.period)
    print(md_table(table, "策略"))
    out = Path(args.out) / ("synthetic" if args.synthetic else "")
    path = write_report(args.plan, results, args.period, out, args.synthetic, notes, extra)
    print(f"\n报告：{path}")
    if args.synthetic:
        print("（合成数据，只验证代码能跑通，结果不代表任何真实表现，也不登记到试验记录）")
    else:
        for res in results:
            log_trial(args.plan, res.name, args.period, result_summary(res, args.period), res.params, args.note)
        print(f"已登记到 research/trials.csv，方案 {args.plan} 累计试验 {count_trials(args.plan)} 次")
    return 0


if __name__ == "__main__":
    sys.exit(main())
