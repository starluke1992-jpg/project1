"""绩效指标。口径统一（见 docs/plan_A_B.md 1.2 节）：

  年化收益   按净值几何年化
  年化波动   日收益标准差 × √252
  夏普       日超额收益（组合 − 现金）的均值 / 日收益标准差 × √252
  最大回撤   净值相对历史高点的最大跌幅
  卡玛       年化收益 / |最大回撤|
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEV = ("2011-01-01", "2018-12-31")      # 开发期：所有设计和调参只看这段
HOLDOUT = ("2019-01-01", None)          # 留出期：规则冻结后只跑一次


def period_slice(s, period: str):
    if period == "full":
        return s
    start, end = DEV if period == "dev" else HOLDOUT
    return s.loc[start:end] if end else s.loc[start:]


def max_drawdown(returns: pd.Series) -> float:
    nav = (1 + returns).cumprod()
    return float((nav / nav.cummax() - 1).min()) if len(nav) else np.nan


def drawdown_table(returns: pd.Series, top: int = 5) -> pd.DataFrame:
    """最深的几次回撤：开始、谷底、恢复日期和深度。"""
    nav = (1 + returns).cumprod()
    dd = nav / nav.cummax() - 1
    rows, in_dd, start = [], False, None
    for d, v in dd.items():
        if v < 0 and not in_dd:
            in_dd, start = True, d
        elif v == 0 and in_dd:
            seg = dd.loc[start:d]
            rows.append((start, seg.idxmin(), d, seg.min()))
            in_dd = False
    if in_dd:
        seg = dd.loc[start:]
        rows.append((start, seg.idxmin(), pd.NaT, seg.min()))
    out = pd.DataFrame(rows, columns=["开始", "谷底", "恢复", "深度"])
    return out.sort_values("深度").head(top).reset_index(drop=True)


def summary(returns: pd.Series, cash: pd.Series, turnover: pd.Series = None, costs: pd.Series = None) -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {}
    c = cash.reindex(r.index).fillna(0.0)
    years = len(r) / TRADING_DAYS
    nav_end = float((1 + r).prod())
    ann = nav_end ** (1 / years) - 1 if nav_end > 0 else -1.0
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ex = r - c
    sharpe = float(ex.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else np.nan
    mdd = max_drawdown(r)
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    out = {
        "起": f"{r.index[0]:%Y-%m-%d}", "止": f"{r.index[-1]:%Y-%m-%d}",
        "年化收益": ann, "年化波动": vol, "夏普": sharpe, "最大回撤": mdd,
        "卡玛": ann / abs(mdd) if mdd < 0 else np.nan,
        "月胜率": float((monthly > 0).mean()),
    }
    if turnover is not None:
        out["年换手"] = float(turnover.reindex(r.index).fillna(0).sum() / years)
    if costs is not None:
        out["年成本"] = float(costs.reindex(r.index).fillna(0).sum() / years)
    return out


def yearly_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).groupby(returns.index.year).prod() - 1


def result_summary(res, period: str = "dev") -> dict:
    r = period_slice(res.returns, period)
    s = summary(r, res.cash, res.turnover, res.costs)
    ev = period_slice(res.exante_vol.dropna(), period) if len(res.exante_vol.dropna()) else pd.Series(dtype=float)
    if len(ev):
        s["事前波动(均值)"] = float(ev.mean())
    return s


def comparison_table(results: list, period: str = "dev") -> pd.DataFrame:
    rows = {res.name: result_summary(res, period) for res in results}
    return pd.DataFrame(rows).T
