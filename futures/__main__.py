"""生成复权连续序列。

  python -m futures build --markets SHFE,INE --start 2010-01-01
  python -m futures build --markets SHFE,INE,DCE,CZCE,CFFEX,GFEX --start 2010-01-01   # 本地运行

先用 datasource 按月拉逐合约数据（有缓存就直接读），再按 continuous.py 的规则换月、复权，
每个品种存到 data/cache/futures_cont/<品种>.csv。逐日请求交易所网站，第一次取十几年数据会很慢。
"""
import argparse
import sys

from datasource.cache import DEFAULT_ROOT, SeriesCache

from .data import build_and_save


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m futures")
    ap.add_argument("--cache-dir", default=str(DEFAULT_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="从逐合约数据生成复权连续序列")
    p.add_argument("--markets", default="SHFE,INE", help="逗号分隔：CFFEX,SHFE,DCE,CZCE,INE,GFEX")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    args = ap.parse_args(argv)
    built = build_and_save([m.strip().upper() for m in args.markets.split(",")], args.start, args.end,
                           SeriesCache(args.cache_dir))
    print(f"共生成 {len(built)} 个品种：{', '.join(sorted(built))}" if built else "没有生成任何品种")
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())
