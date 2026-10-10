"""全部参数集中在这里。默认值取自文献里的常用值，不在样本内寻优，见 docs/crisis_alpha.md 第 8 节。"""
from dataclasses import dataclass, field

from .instruments import ASSET_CLASS_BUDGET

TRADING_DAYS = 244  # 国内期货每年交易日约 242~245 天


@dataclass(frozen=True)
class Config:
    # 资金与风险目标
    capital: float = 20_000_000          # 用于计算仓位的资金（元）；实盘每月按账户权益更新
    target_vol: float = 0.15             # 组合年化波动率目标
    budget: dict = field(default_factory=lambda: dict(ASSET_CLASS_BUDGET))

    # 趋势信号
    ewmac_speeds: tuple = ((8, 32), (16, 64), (32, 128), (64, 256))
    breakout_windows: tuple = (40, 80, 160, 320)
    ewmac_weight: float = 0.6            # EWMAC 合计权重，其余给通道突破
    fdm: float = 1.2                     # 预测分散乘数
    forecast_cap: float = 2.0

    # 波动率估计
    vol_span: int = 35
    vol_long_weight: float = 0.3
    vol_long_window: int = 2520

    # 品种分散乘数（IDM）
    idm_cap: float = 2.5
    corr_window_weeks: int = 104         # 用最近两年的周度数据估计相关系数，每月末更新一次
    corr_min_weeks: int = 52             # 数据不足一年时 IDM 取 1

    # 组合风控
    max_vol_mult: float = 2.0            # 事前波动率超过 目标 × 该倍数 时整体缩仓
    max_margin: float = 0.40             # 保证金占用上限（占资金比例）

    # 交易
    buffer: float = 0.10                 # 缓冲区宽度 = 平均仓位 × buffer，目标仓位在缓冲区内不调仓
    integer: bool = True                 # 按整手交易；False 时按小数手数（研究用）
    execution_lag: int = 1               # 收盘出信号，滞后几天成交（1 = 下一交易日收盘前完成）
    slippage_ticks: float = 1.0          # 单边滑点（跳）
    commission: float = 0.5e-4           # 单边手续费（成交额比例）
