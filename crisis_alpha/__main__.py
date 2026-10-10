"""命令行入口。

  python -m crisis_alpha backtest --demo --plot demo.png              # 模拟数据，检查程序
  python -m crisis_alpha backtest --data-dir data/futures --benchmark IF --start 2012-01-01 --plot bt.png
  python -m crisis_alpha positions --data-dir data/futures --holdings holdings.csv --out orders.csv
  python -m crisis_alpha capital --data-dir data/futures --capital 20000000
  python -m crisis_alpha fetch --out data/futures --start 20100101   # akshare，未复权，仅演示
"""
import argparse
from dataclasses import replace

import pandas as pd

from .analysis import benchmark_returns, crisis_report
from .backtest import run_backtest, stats
from .config import TRADING_DAYS, Config
from .data import fetch_akshare, load_dir, synthetic_market
from .instruments import ASSET_CLASSES
from .live import capital_check, daily_orders


def _config(args) -> Config:
    cfg = Config()
    if getattr(args, "capital", None):
        cfg = replace(cfg, capital=args.capital)
    if getattr(args, "target_vol", None):
        cfg = replace(cfg, target_vol=args.target_vol)
    if getattr(args, "fractional", False):
        cfg = replace(cfg, integer=False)
    return cfg


def _market(args):
    if getattr(args, "demo", False):
        return synthetic_market()
    if not args.data_dir:
        raise SystemExit("需要 --data-dir（或 backtest 用 --demo）")
    return load_dir(args.data_dir, args.codes)


def cmd_backtest(args):
    cfg, market = _config(args), _market(args)
    res = run_backtest(market, cfg, start=args.start)
    if args.demo:
        print("【模拟数据，只用于检查程序，结果不代表真实表现】\n")
    print("回测区间：", res.returns.index[0].date(), "~", res.returns.index[-1].date(), " 品种数：", len(res.pnl.columns))
    print("绩效：", stats(res.returns))
    years = len(res.returns) / TRADING_DAYS
    print("年化成本：", round(float(res.costs.sum().sum() / cfg.capital / years), 4),
          " 平均 IDM：", round(float(res.idm.mean()), 2),
          " 风控缩仓天数占比：", round(float((res.scale < 1).mean()), 3))
    contrib = (res.class_returns().sum() / years).rename(ASSET_CLASSES).round(4)
    print("\n各资产类别年化贡献：\n" + contrib.to_string())

    if args.benchmark not in market:
        print(f"\n没有基准 {args.benchmark} 的数据，跳过危机检验")
        return
    bench = benchmark_returns(market[args.benchmark])
    if args.start:
        bench = bench.loc[pd.Timestamp(args.start):]
    rep = crisis_report(res.returns, bench, args.threshold)
    print(f"\n基准 {args.benchmark} 回撤超过 {args.threshold:.0%} 的区间：")
    print(rep["episodes"].to_string(index=False) if len(rep["episodes"]) else "  无")
    print("\n按基准季度收益分组（1 = 最差）：\n" + rep["smile_quarterly"].to_string())
    print("\n上涨/下跌 beta：", rep["beta"])
    if args.plot:
        from .plot import plot_backtest

        plot_backtest(res, bench, args.plot, title="危机 Alpha CTA" + ("（模拟数据）" if args.demo else ""),
                      threshold=args.threshold)
        print(f"\n图已保存：{args.plot}")


def cmd_positions(args):
    cfg, market = _config(args), _market(args)
    holdings = {}
    if args.holdings:
        h = pd.read_csv(args.holdings)
        holdings = dict(zip(h.iloc[:, 0].astype(str).str.upper(), h.iloc[:, 1].astype(int)))
    table, summary = daily_orders(market, holdings, cfg)
    print("汇总：", summary)
    show = table[(table["trade"] != 0) | (table["current"] != 0) | (table["new"] != 0)]
    print(show.to_string())
    if args.out:
        table.to_csv(args.out, encoding="utf-8-sig")
        print(f"\n已保存：{args.out}")


def cmd_capital(args):
    cfg, market = _config(args), _market(args)
    out = capital_check(market, cfg, min_lots=args.min_lots)
    print(f"资金 {cfg.capital:,.0f} 元时各品种的平均仓位（手），以及平均仓位达到 {args.min_lots} 手需要的资金：")
    print(out.to_string())


def cmd_fetch(args):
    fetch_akshare(args.codes, args.start, args.end, args.out)
    print(f"已保存到 {args.out}。注意：新浪主力连续未复权，换月跳空会被当成盈亏，只适合演示。")


def main(argv=None):
    ap = argparse.ArgumentParser(description="危机 Alpha CTA：多资产、多周期趋势跟踪")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data-dir", help="每个品种一个 CSV：<代码>.csv，列 date, close[, contract | adj_close]")
        p.add_argument("--codes", nargs="*", help="只用这些品种，默认用目录里所有 UNIVERSE 品种")
        p.add_argument("--capital", type=float, help="资金（元）")
        p.add_argument("--target-vol", type=float, help="年化波动率目标，如 0.15")

    p = sub.add_parser("backtest", help="回测并做危机 Alpha 检验")
    common(p)
    p.add_argument("--demo", action="store_true", help="用模拟数据")
    p.add_argument("--benchmark", default="IF", help="基准品种代码，默认 IF")
    p.add_argument("--start", help="从这一天开始统计（之前用于预热）")
    p.add_argument("--threshold", type=float, default=0.15, help="基准回撤超过多少算危机区间")
    p.add_argument("--fractional", action="store_true", help="按小数手数回测（不取整）")
    p.add_argument("--plot", help="输出图片路径")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("positions", help="按最新收盘数据生成目标仓位和开平指令")
    common(p)
    p.add_argument("--holdings", help="当前持仓 CSV：两列，品种代码,手数（空头为负）")
    p.add_argument("--out", help="保存完整表格的 CSV 路径")
    p.set_defaults(func=cmd_positions)

    p = sub.add_parser("capital", help="检查资金规模是否足够交易每个品种")
    common(p)
    p.add_argument("--min-lots", type=float, default=1.0)
    p.set_defaults(func=cmd_capital)

    p = sub.add_parser("fetch", help="用 akshare 下载新浪主力连续（未复权）")
    p.add_argument("--out", required=True)
    p.add_argument("--codes", nargs="*")
    p.add_argument("--start", default="20100101")
    p.add_argument("--end", default="22220101")
    p.set_defaults(func=cmd_fetch)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
