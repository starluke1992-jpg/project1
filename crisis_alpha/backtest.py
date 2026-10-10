"""回测：固定资金（不复利），收盘出信号，滞后 execution_lag 天成交。

盈亏 = 持仓 × 合约乘数 × 复权价差；成本 = 调仓成本 + 移仓成本，
每手单边成本 = 滑点跳数 × 最小变动价位 × 乘数 + 手续费率 × 价格 × 乘数。
有换月标记（roll 列）的品种按实际换月日收移仓成本（平旧开新，两次单边），
没有的按 rolls_per_year 均摊到每天。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TRADING_DAYS, Config
from .instruments import UNIVERSE
from .portfolio import build_panel, held_positions, target_positions


@dataclass
class Result:
    returns: pd.Series        # 每日净收益率（占资金比例）
    pnl: pd.DataFrame         # 各品种每日盈亏（元，未扣成本）
    costs: pd.DataFrame       # 各品种每日成本（元）
    positions: pd.DataFrame   # 实际持仓（手，已含成交滞后）
    forecasts: pd.DataFrame
    targets: pd.DataFrame     # 风控后、取整前的目标仓位
    weights: pd.DataFrame
    idm: pd.Series
    scale: pd.Series          # 组合风控缩放系数（1 = 未缩仓）
    ex_ante_vol: pd.Series
    cfg: Config
    universe: dict

    def class_returns(self) -> pd.DataFrame:
        """按资产类别汇总的每日净收益率。"""
        net = self.pnl - self.costs
        cls = {c: self.universe[c].asset_class for c in net.columns}
        return net.T.groupby(cls).sum().T / self.cfg.capital


def run_backtest(market: dict, cfg: Config = Config(), universe=UNIVERSE, start=None) -> Result:
    """market: {品种代码: DataFrame(price, adj, roll)}，见 data.py。start 之前的数据只用于预热。"""
    panel = build_panel(market, cfg, universe)
    tp = target_positions(panel, cfg, universe)
    held = held_positions(tp["target"], tp["avg_position"], cfg)
    pos = held.shift(cfg.execution_lag).fillna(0.0)

    codes = pos.columns
    mult = pd.Series({c: universe[c].multiplier for c in codes})
    tick = pd.Series({c: universe[c].tick for c in codes})
    rolls = pd.Series({c: universe[c].rolls_per_year for c in codes})

    pnl = pos.shift().fillna(0.0) * panel["diff"] * mult
    per_lot = (cfg.slippage_ticks * tick + cfg.commission * panel["price"]) * mult
    trade_cost = pos.diff().fillna(pos).abs() * per_lot
    actual_roll = panel["roll"].astype(float) * 2 * pos.abs() * per_lot
    est_roll = pos.abs() * per_lot * 2 * rolls / TRADING_DAYS
    roll_cost = est_roll.copy()
    has = panel["has_roll_data"].reindex(codes, fill_value=False).astype(bool)
    roll_cost.loc[:, has] = actual_roll.loc[:, has]
    costs = (trade_cost + roll_cost).fillna(0.0)

    returns = (pnl.sum(axis=1) - costs.sum(axis=1)) / cfg.capital
    res = Result(returns, pnl, costs, pos, panel["forecast"], tp["target"], tp["weights"], tp["idm"],
                 tp["scale"], tp["ex_ante_vol"], cfg, universe)
    return _since(res, start) if start else res


def _since(res: Result, start) -> Result:
    t = pd.Timestamp(start)
    cut = lambda x: x.loc[t:]  # noqa: E731
    return Result(cut(res.returns), cut(res.pnl), cut(res.costs), cut(res.positions), cut(res.forecasts),
                  cut(res.targets), cut(res.weights), cut(res.idm), cut(res.scale), cut(res.ex_ante_vol),
                  res.cfg, res.universe)


def stats(returns: pd.Series) -> dict:
    """年化收益（算术，固定资金口径）、波动、夏普、最大回撤、卡玛、月度偏度、月度胜率。"""
    r = returns.dropna()
    if len(r) < 2:
        return {}
    equity = r.cumsum()
    dd = equity - equity.cummax()
    monthly = r.groupby(r.index.to_period("M")).sum()
    ann_ret = r.mean() * TRADING_DAYS
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    max_dd = -dd.min()
    return {
        "ann_return": round(float(ann_ret), 4),
        "ann_vol": round(float(ann_vol), 4),
        "sharpe": round(float(ann_ret / ann_vol), 2) if ann_vol > 0 else np.nan,
        "max_drawdown": round(float(max_dd), 4),
        "calmar": round(float(ann_ret / max_dd), 2) if max_dd > 0 else np.nan,
        "monthly_skew": round(float(monthly.skew()), 2),
        "monthly_hit_rate": round(float((monthly > 0).mean()), 3),
    }
