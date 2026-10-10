"""实盘用：每日收盘后算目标仓位、生成开平指令，以及检查资金规模够不够交易每个品种。"""
import numpy as np
import pandas as pd

from .config import Config
from .instruments import ASSET_CLASSES, UNIVERSE
from .portfolio import build_panel, buffered, target_positions


def split_order(current: int, new: int) -> list:
    """把仓位变化拆成国内期货的开平指令，例如 +2 → -3 拆成 卖平 2、卖开 3。"""
    delta = new - current
    if delta == 0:
        return []
    side = "买" if delta > 0 else "卖"
    opposite = max(-current, 0) if delta > 0 else max(current, 0)   # 可平的反向持仓
    close = min(abs(delta), opposite)
    out = [f"{side}平 {close}"] if close else []
    if abs(delta) > close:
        out.append(f"{side}开 {abs(delta) - close}")
    return out


def daily_orders(market: dict, holdings: dict, cfg: Config = Config(), universe=UNIVERSE):
    """返回 (逐品种表, 组合汇总)。holdings: {品种代码: 当前持仓手数（空头为负）}。

    目标仓位用最新一天的收盘数据计算，按缓冲区规则和当前持仓比较后给出新持仓和开平指令，
    下一交易日执行。
    """
    panel = build_panel(market, cfg, universe)
    tp = target_positions(panel, cfg, universe)
    date = panel["forecast"].index[-1]
    codes = panel["forecast"].columns
    unknown = sorted(set(holdings) - set(codes))
    if unknown:
        raise ValueError(f"持仓里有没有行情数据的品种：{unknown}")

    target = tp["target"].loc[date]
    avg = tp["avg_position"].loc[date]
    current = pd.Series(holdings, dtype=float).reindex(codes).fillna(0.0)
    new = pd.Series(buffered(target, current, avg.abs() * cfg.buffer, cfg.integer), index=codes)
    mult = pd.Series({c: universe[c].multiplier for c in codes})
    margin = pd.Series({c: universe[c].margin for c in codes})
    price = panel["price"].loc[date]

    table = pd.DataFrame({
        "name": [universe[c].name for c in codes],
        "class": [ASSET_CLASSES[universe[c].asset_class] for c in codes],
        "forecast": panel["forecast"].loc[date].round(2),
        "avg_position": avg.round(2),
        "target": target.round(2),
        "current": current.astype(int),
        "new": new.astype(int),
        "trade": (new - current).astype(int),
        "orders": [" / ".join(split_order(int(c), int(n))) for c, n in zip(current, new)],
        "margin": (new.abs() * price * mult * margin).round(0),
    }, index=codes)
    summary = {
        "date": date.date(),
        "idm": round(float(tp["idm"].loc[date]), 2),
        "ex_ante_vol": round(float(tp["ex_ante_vol"].loc[date]), 4),
        "risk_scale": round(float(tp["scale"].loc[date]), 3),
        "margin_ratio": round(float(table["margin"].sum() / cfg.capital), 4),
    }
    return table, summary


def capital_check(market: dict, cfg: Config = Config(), universe=UNIVERSE, min_lots: float = 1.0) -> pd.DataFrame:
    """按最新数据，每个品种在当前资金下的平均仓位（F=1 时的手数），以及平均仓位达到 min_lots 手所需的资金。

    平均仓位不到 1 手的品种，取整误差会很大（信号再强也可能只有 0 或 1 手），
    应该提高资金、在同类里少选几个品种（集中预算），或者用小合约替代。
    """
    panel = build_panel(market, cfg, universe)
    tp = target_positions(panel, cfg, universe)
    date = panel["forecast"].index[-1]
    avg = tp["avg_position"].loc[date].abs()
    need = (cfg.capital * min_lots / avg.replace(0, np.nan)).round(-4)
    out = pd.DataFrame({
        "name": [universe[c].name for c in avg.index],
        "avg_position": avg.round(2),
        "capital_needed": need,
        "ok": avg >= min_lots,
    }, index=avg.index)
    return out.sort_values("capital_needed", ascending=False)
