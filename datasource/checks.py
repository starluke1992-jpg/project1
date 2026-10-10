"""数据质量检查：拿到数据先过一遍，再用来回测。

price（价格类）检查：空数据、非正价格、缺失值、长时间断档、单日涨跌过大、价格长时间不变。
rate（利率、收益率类）检查：空数据、缺失值、长时间断档、取值超出合理范围。
返回问题描述列表，空列表表示没发现问题。
"""
import pandas as pd


def check_series(df: pd.DataFrame, kind: str = "price", max_jump: float = 0.2,
                 max_gap_days: int = 15, stale_days: int = 5, rate_range=(-1.0, 20.0)) -> list:
    if df is None or df.empty or "close" not in df.columns:
        return ["没有数据"]
    close = df["close"]
    issues = []
    if close.isna().any():
        issues.append(f"收盘价缺失 {int(close.isna().sum())} 天")
    close = close.dropna()

    # 春节长假最多 9~10 个自然日，超过 max_gap_days 视为断档
    gaps = close.index.to_series().diff().dt.days
    big = gaps[gaps > max_gap_days]
    if len(big):
        examples = ", ".join(f"{d:%Y-%m-%d}({int(g)}天)" for d, g in big.head(3).items())
        issues.append(f"断档 {len(big)} 处: {examples}")

    if kind == "rate":
        lo, hi = rate_range
        bad = close[(close < lo) | (close > hi)]
        if len(bad):
            issues.append(f"取值超出 {lo}~{hi}（%）共 {len(bad)} 天，首个 {bad.index[0]:%Y-%m-%d}")
        return issues

    if (close <= 0).any():
        issues.append(f"非正价格 {int((close <= 0).sum())} 天")
    jumps = close.pct_change().abs()
    jumps = jumps[jumps > max_jump]
    if len(jumps):
        examples = ", ".join(f"{d:%Y-%m-%d}({v:+.0%})" for d, v in jumps.head(3).items())
        issues.append(f"单日涨跌超过 {max_jump:.0%} 共 {len(jumps)} 天: {examples}"
                      "（期货主力连续多为换月跳空；其他品种请核对是否拆分或数据错误）")
    run = (close.diff() == 0).astype(int)
    longest = int(run.groupby((run == 0).cumsum()).sum().max()) if len(run) else 0
    if longest >= stale_days:
        issues.append(f"价格连续 {longest} 天不变（停牌或数据未更新）")
    return issues
