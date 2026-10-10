"""指标计算，与截图里的各个副图一一对应。

  主图    BOLL(20,2)                    —— 趋势因子：收盘价在中轨之上/之下
  副图1   KAMA(10,2,30) / MID = EMA(C,5)
  副图2   BB_Width_Angle                —— 波动率因子：布林带宽度的变化角度
  副图3   ATR_Angle                     —— 波动率因子：ATR 的变化角度
  副图4   DIFF_MK = MID - KAMA          —— 动量因子

公式是从截图上逐根K线读数反推出来的（2022-07-08 这根K线表头给出的
BOLL/KAMA/BB_Width_Angle/ATR_Angle 数值都能对上），参数可调。
"""
import numpy as np
import pandas as pd


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    """返回 (中轨, 上轨, 下轨)。STD 用样本标准差，与通达信/东方财富一致。"""
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=1)
    return mid, mid + k * std, mid - k * std


def kama(close: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman 自适应均线。

    ER = |C - C[n]| / sum(|C - C[1]|, n)
    SC = (ER * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ^ 2
    KAMA = KAMA[1] + SC * (C - KAMA[1])
    """
    c = close.to_numpy(dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return pd.Series(out, index=close.index)
    fsc, ssc = 2 / (fast + 1), 2 / (slow + 1)
    change = np.abs(c[n:] - c[:-n])
    volatility = pd.Series(np.abs(np.diff(c, prepend=np.nan))).rolling(n).sum().to_numpy()[n:]
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility > 0)
    sc = (er * (fsc - ssc) + ssc) ** 2
    out[n - 1] = c[n - 1]
    for i in range(n, len(c)):
        out[i] = out[i - 1] + sc[i - n] * (c[i] - out[i - 1])
    return pd.Series(out, index=close.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    """ATR = MA(TR, n)，简单平均（截图里的 ATR_Angle 用的是这个版本，不是 Wilder 平滑）。"""
    prev = close.shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def angle(x: pd.Series) -> pd.Series:
    """ATAN(x) 转角度，即通达信里常见的 ATAN(...)*180/PI。"""
    return pd.Series(np.degrees(np.arctan(x)), index=x.index)


def compute_indicators(df: pd.DataFrame, p) -> pd.DataFrame:
    """df 需含 open/high/low/close 列；p 为 strategy.Params。"""
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out["boll"], out["upper"], out["lower"] = bollinger(close, p.boll_n, p.boll_k)
    out["kama"] = kama(close, p.kama_n, p.kama_fast, p.kama_slow)
    out["mid"] = close.ewm(span=p.mid_n, adjust=False).mean()
    out["diff_mk"] = out["mid"] - out["kama"]

    # 带宽变化率(%)取角度：>0 带宽扩张，<0 带宽收缩
    width = (out["upper"] - out["lower"]) / out["boll"]
    out["bb_width_angle"] = angle((width / width.shift() - 1) * 100)

    # ATR 逐日变化量(点数)取角度：>0 波动放大，<0 波动收敛
    out["atr"] = atr(df["high"], df["low"], close, p.atr_n)
    out["atr_angle"] = angle(out["atr"].diff())
    return out
