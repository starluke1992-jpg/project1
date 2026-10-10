"""通用回测引擎（方案 A、B1、B2 共用）。

输入是各资产的日超额收益（资产收益 − 现金收益）和现金日收益，组合日收益 =
  现金收益 + Σ 权重 × 超额收益
这个写法对两类资产都成立：
  ETF 等全额出资的资产：没投出去的钱拿现金利息，借钱（权重合计 > 1）按现金利率付息；
  期货：只交保证金，账户里的钱全部拿现金利息，期货价格变动本身就是超额收益。
区别只在权重随价格漂移的方式（funded 参数）。

时间顺序（不用未来数据）：
  T 日收盘后，用截至 T 日的数据算目标权重；
  T+lag 日收盘按目标权重调仓（默认 lag=1），扣交易成本；
  调仓前那一天的收益仍按旧权重计算。
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Context:
    """传给目标权重函数的信息，只包含截至信号日（含）的数据。"""
    date: pd.Timestamp
    excess: pd.DataFrame      # 各资产日超额收益
    total: pd.DataFrame       # 各资产日总收益
    cash: pd.Series           # 现金日收益
    weights: pd.Series        # 当前实际持仓权重
    extra: dict = field(default_factory=dict)  # 预先算好的信号等（必须只用到当日及以前的数据）

    def prices(self) -> pd.DataFrame:
        """由总收益累计得到的价格指数（起点为 1），用于均线、动量等信号。"""
        return (1 + self.total.fillna(0)).cumprod().where(self.total.notna().cummax())

    def cash_index(self) -> pd.Series:
        return (1 + self.cash.fillna(0)).cumprod()


@dataclass
class Result:
    name: str
    returns: pd.Series            # 组合日收益（已扣成本）
    cash: pd.Series               # 同期现金日收益
    weights: pd.DataFrame         # 每日收盘后的实际权重
    targets: pd.DataFrame         # 每个信号日的目标权重
    turnover: pd.Series           # 每日换手（Σ|调仓权重|）
    costs: pd.Series              # 每日交易成本（占净值比例）
    exante_vol: pd.Series         # 每个信号日缩放前的事前波动（策略提供时）
    params: dict = field(default_factory=dict)

    @property
    def nav(self) -> pd.Series:
        return (1 + self.returns).cumprod()


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每个月最后一个交易日。"""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().to_numpy())


def week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每周最后一个交易日。"""
    s = pd.Series(index, index=index)
    iso = index.isocalendar()
    return pd.DatetimeIndex(s.groupby([iso.year.to_numpy(), iso.week.to_numpy()]).max().sort_values().to_numpy())


def run_backtest(excess: pd.DataFrame, cash: pd.Series, target_fn, rebalance_dates, name: str = "",
                 cost_bps=0.0, band_abs: float = 0.0, band_rel: float = 0.0, lag: int = 1,
                 funded: bool = True, roll_flags: pd.DataFrame = None, extra: dict = None,
                 params: dict = None) -> Result:
    """target_fn(ctx) 返回目标权重 Series（不含现金；可返回 (权重, 事前波动)）。

    cost_bps：单边交易成本（基点），可以是数字或按资产的 Series。
    band_abs / band_rel：调整幅度小于净值的 band_abs，或小于目标权重的 band_rel 时，这个资产不调。
    roll_flags：期货换月标记（True 表示当天换月），换月当天按持仓 × 2 × 成本扣费。
    """
    assets = list(excess.columns)
    idx = excess.index
    cash = cash.reindex(idx).fillna(0.0)
    x = excess.fillna(0.0).to_numpy()
    total = excess.add(cash, axis=0)
    cost = (pd.Series(cost_bps, index=assets) if np.isscalar(cost_bps)
            else pd.Series(cost_bps).reindex(assets).fillna(0.0)).to_numpy() / 1e4
    rolls = roll_flags.reindex(index=idx, columns=assets).fillna(False).to_numpy() if roll_flags is not None else None
    signal_pos = set(idx.get_indexer(pd.DatetimeIndex(rebalance_dates).intersection(idx)))

    n, m = len(idx), len(assets)
    w = np.zeros(m)
    pending = {}  # 执行日位置 -> 目标权重
    rets, turns, costs = np.zeros(n), np.zeros(n), np.zeros(n)
    w_hist = np.zeros((n, m))
    targets, exante = {}, {}

    for t in range(n):
        # 1) 当天收益（按昨日收盘后的持仓）
        r_p = cash.iloc[t] + float(w @ x[t])
        growth = (1 + x[t] + (cash.iloc[t] if funded else 0.0))
        w = w * growth / (1 + r_p) if (1 + r_p) != 0 else w
        cost_t = 0.0
        if rolls is not None:
            cost_t += float(np.abs(w) @ (rolls[t] * 2 * cost))

        # 2) 到了执行日就调仓
        if t in pending:
            tgt = pending.pop(t)
            trade = tgt - w
            keep = np.abs(trade) < band_abs
            if band_rel > 0:
                keep |= np.abs(trade) < band_rel * np.abs(tgt)
            keep &= ~((tgt == 0) & (w != 0))  # 目标为 0 的仓位一定平掉
            trade[keep] = 0.0
            w = w + trade
            turns[t] = np.abs(trade).sum()
            cost_t += float(np.abs(trade) @ cost)

        rets[t] = r_p - cost_t
        costs[t] = cost_t
        w_hist[t] = w

        # 3) 信号日收盘后算目标
        if t in signal_pos and t + lag < n:
            ctx = Context(date=idx[t], excess=excess.iloc[: t + 1], total=total.iloc[: t + 1],
                          cash=cash.iloc[: t + 1], weights=pd.Series(w, index=assets), extra=extra or {})
            out = target_fn(ctx)
            tgt, vol = out if isinstance(out, tuple) else (out, np.nan)
            tgt = tgt.reindex(assets).fillna(0.0)
            targets[idx[t]] = tgt
            exante[idx[t]] = vol
            pending[t + lag] = tgt.to_numpy(dtype=float)

    return Result(name=name, returns=pd.Series(rets, index=idx), cash=cash,
                  weights=pd.DataFrame(w_hist, index=idx, columns=assets),
                  targets=pd.DataFrame(targets).T.reindex(columns=assets),
                  turnover=pd.Series(turns, index=idx), costs=pd.Series(costs, index=idx),
                  exante_vol=pd.Series(exante, dtype=float), params=params or {})
