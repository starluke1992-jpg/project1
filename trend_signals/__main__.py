"""命令行入口。

  python -m trend_signals --csv data/IF_main_digitized.csv --plot chart.png
  python -m trend_signals --akshare IF0 --start 2022-01-01 --end 2023-08-15 --plot chart.png
"""
import argparse

import pandas as pd

from .data import fetch_akshare, load_csv
from .strategy import LABELS, generate_signals, summary, trades


def main(argv=None):
    ap = argparse.ArgumentParser(description="趋势跟踪信号：BOLL + KAMA 动量 + 波动率角度")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="含 date/open/high/low/close 列的日线 CSV（中文列名也可）")
    src.add_argument("--akshare", metavar="SYMBOL", help="新浪主力连续合约代码，如 IF0、IC0、RB0")
    ap.add_argument("--start", default="19900101")
    ap.add_argument("--end", default="22220101")
    ap.add_argument("--plot", help="输出图片路径，如 chart.png")
    ap.add_argument("--from-date", help="只画/统计这一天之后的部分（之前的数据用于指标预热）")
    ap.add_argument("--multiplier", type=float, default=300, help="合约乘数，IF=300")
    args = ap.parse_args(argv)

    df = load_csv(args.csv) if args.csv else fetch_akshare(args.akshare, args.start, args.end)
    sig = generate_signals(df)
    if args.from_date:
        sig = sig[sig.index >= pd.Timestamp(args.from_date)]

    print("信号：")
    for date, row in sig[sig["signal"] != ""].iterrows():
        print(f"  {date:%Y-%m-%d}  {LABELS[row['signal']]}  {row['close']:.1f}")

    tr = trades(sig)
    if not tr.empty:
        print("\n交易：")
        print(tr.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\n汇总：", summary(tr, args.multiplier))

    if args.plot:
        from .plot import plot_signals

        name = args.akshare or (args.csv.rsplit("/", 1)[-1] if args.csv else "")
        plot_signals(sig, args.plot, title=name)
        print(f"\n图已保存：{args.plot}")


if __name__ == "__main__":
    main()
