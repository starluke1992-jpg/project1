"""命令行入口。

  python -m datasource check                            # 检查各数据网站能否连上
  python -m datasource universe                         # 按 config/universe_A.csv 拉取方案 A 的全部序列
  python -m datasource universe --only hs300,gold --start 2015-01-01
  python -m datasource main RB0 M0 CU0                  # 期货主力连续（未复权）
  python -m datasource contracts --market SHFE --start 2024-01-01 --end 2024-03-31
  python -m datasource fees                             # 手续费与保证金表

数据缓存在 data/cache/（不进 git），重复运行只增量更新。
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from . import akshare_api as api
from .cache import DEFAULT_ROOT, SeriesCache
from .checks import check_series
from .connectivity import check_all

DEFAULT_UNIVERSE = Path(__file__).resolve().parents[1] / "config" / "universe_A.csv"


def _short(e: Exception) -> str:
    msg = str(e).strip().splitlines()[0] if str(e).strip() else ""
    return f"{type(e).__name__}: {msg[:150]}"


def _report(label: str, df: pd.DataFrame, issues: list) -> str:
    if df is None or df.empty:
        return f"{label}  无数据"
    status = "通过" if not issues else "；".join(issues)
    return f"{label}  {len(df):>5} 行  {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}  {status}"


def cmd_check(args) -> int:
    rows = check_all(timeout=args.timeout)
    for purpose, host, scheme, ok, detail in rows:
        print(f"{'✓' if ok else '✗'} {purpose:<22} {scheme}://{host:<28} {detail}")
    blocked = [r for r in rows if not r[3]]
    print(f"\n连得上 {len(rows) - len(blocked)} 个，连不上 {len(blocked)} 个。")
    if blocked:
        https_hosts = [h for _, h, s, ok, _ in rows if not ok and s == "https"]
        if https_hosts:
            print("在云端环境里，可以在环境设置的 Network access → Allowed domains 中放行这些 https 域名：")
            print("  " + ", ".join(https_hosts))
        if any(not ok and s == "http" for _, _, s, ok, _ in rows):
            print("标为 http 的网站（大商所、郑商所、中金所、广期所）只能在本地运行时访问。")
    return 0 if not blocked else 1


def cmd_universe(args) -> int:
    cfg = pd.read_csv(args.config, dtype=str, keep_default_na=False)
    if args.only:
        wanted = set(args.only.split(","))
        cfg = cfg[cfg["id"].isin(wanted)]
    cache = SeriesCache(args.cache_dir)
    failed = 0
    for row in cfg.itertuples(index=False):
        label = f"{row.id:<12} {row.source}:{row.symbol:<8}"
        try:
            df = cache.get(row.source, row.symbol, args.start, args.end, refresh=args.refresh)
            print(_report(label, df, check_series(df, api.SOURCES[row.source][1])))
        except Exception as e:  # 单条失败不影响其他序列
            failed += 1
            print(f"{label}  失败  {_short(e)}")
    print(f"\n完成 {len(cfg) - failed}/{len(cfg)} 条，缓存目录：{cache.root}")
    return 0 if not failed else 1


def cmd_main(args) -> int:
    cache = SeriesCache(args.cache_dir)
    failed = 0
    for symbol in args.symbols:
        try:
            df = cache.get("futures_main", symbol, args.start, args.end, refresh=args.refresh)
            print(_report(f"{symbol:<8}", df, check_series(df)))
        except Exception as e:
            failed += 1
            print(f"{symbol:<8}  失败  {_short(e)}")
    return 0 if not failed else 1


def cmd_contracts(args) -> int:
    cache = SeriesCache(args.cache_dir)
    try:
        df = cache.get_futures_contracts(args.market, args.start, args.end, refresh=args.refresh)
    except Exception as e:
        print(f"{args.market}  失败  {_short(e)}")
        return 1
    if df.empty:
        print(f"{args.market}  无数据")
        return 1
    print(f"{args.market}  {len(df)} 行，{df['symbol'].nunique()} 个合约，"
          f"{df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}")
    if args.out:
        df.to_csv(args.out, index=False, date_format="%Y-%m-%d")
        print(f"已保存：{args.out}")
    return 0


def cmd_fees(args) -> int:
    try:
        df = api.futures_fees()
    except Exception as e:
        print(f"失败  {_short(e)}")
        return 1
    out = Path(args.cache_dir) / "futures_fees.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} 行，已保存：{out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m datasource", description="AKShare 数据接入与本地缓存")
    ap.add_argument("--cache-dir", default=str(DEFAULT_ROOT), help="缓存目录，默认 data/cache")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="检查各数据网站能否连上")
    p.add_argument("--timeout", type=float, default=8.0)
    p.set_defaults(func=cmd_check)

    def dates(p):
        p.add_argument("--start", default="2010-01-01")
        p.add_argument("--end", default=None, help="默认今天")
        p.add_argument("--refresh", action="store_true", help="忽略缓存，整段重新拉取")

    p = sub.add_parser("universe", help="按配置文件拉取一组序列")
    p.add_argument("--config", default=str(DEFAULT_UNIVERSE))
    p.add_argument("--only", help="只拉这些 id，逗号分隔")
    dates(p)
    p.set_defaults(func=cmd_universe)

    p = sub.add_parser("main", help="期货主力连续（新浪，未复权）")
    p.add_argument("symbols", nargs="+", help="如 RB0 M0 CU0 IF0")
    dates(p)
    p.set_defaults(func=cmd_main)

    p = sub.add_parser("contracts", help="交易所逐合约日线（按月缓存）")
    p.add_argument("--market", required=True, choices=["CFFEX", "SHFE", "DCE", "CZCE", "INE", "GFEX"])
    p.add_argument("--out", help="另存为 CSV")
    dates(p)
    p.set_defaults(func=cmd_contracts)

    p = sub.add_parser("fees", help="手续费与保证金表")
    p.set_defaults(func=cmd_fees)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
