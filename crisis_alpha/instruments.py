"""品种池与资产类别风险预算。

合约乘数、最小变动价位按交易所规则填写；保证金率是“交易所 + 期货公司加收”的大致水平，
只用于保证金占用上限的估算，请按自己期货公司的实际数值修改。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    code: str            # 品种代码，也是数据文件名：<code>.csv
    name: str
    exchange: str
    asset_class: str     # 见 ASSET_CLASSES
    multiplier: float    # 合约乘数（每点价值）
    tick: float          # 最小变动价位
    margin: float        # 保证金率
    rolls_per_year: int  # 每年换月次数，没有换月数据时用来估算移仓成本


ASSET_CLASSES = {
    "equity": "股指",
    "bond": "国债",
    "precious": "贵金属",
    "base_metal": "有色",
    "ferrous": "黑色",
    "energy_chem": "能化",
    "agri": "农产品",
}

# 风险预算：股指、国债、贵金属这三类和股市危机关系最直接，合计 50%；
# 商品分成四个板块，各 12.5%，避免单一板块（如黑色系）主导组合。
ASSET_CLASS_BUDGET = {
    "equity": 0.20,
    "bond": 0.20,
    "precious": 0.10,
    "base_metal": 0.125,
    "ferrous": 0.125,
    "energy_chem": 0.125,
    "agri": 0.125,
}


def _i(code, name, exchange, asset_class, multiplier, tick, margin=0.13, rolls=3):
    return Instrument(code, name, exchange, asset_class, multiplier, tick, margin, rolls)


UNIVERSE = {i.code: i for i in [
    _i("IF", "沪深300", "CFFEX", "equity", 300, 0.2, 0.14, 12),
    _i("IH", "上证50", "CFFEX", "equity", 300, 0.2, 0.14, 12),
    _i("IC", "中证500", "CFFEX", "equity", 200, 0.2, 0.16, 12),
    _i("IM", "中证1000", "CFFEX", "equity", 200, 0.2, 0.16, 12),
    _i("TS", "2年国债", "CFFEX", "bond", 20000, 0.002, 0.01, 4),
    _i("TF", "5年国债", "CFFEX", "bond", 10000, 0.005, 0.022, 4),
    _i("T", "10年国债", "CFFEX", "bond", 10000, 0.005, 0.03, 4),
    _i("TL", "30年国债", "CFFEX", "bond", 10000, 0.01, 0.045, 4),
    _i("AU", "黄金", "SHFE", "precious", 1000, 0.02, 0.12, 2),
    _i("AG", "白银", "SHFE", "precious", 15, 1, 0.14, 2),
    _i("CU", "铜", "SHFE", "base_metal", 5, 10, rolls=12),
    _i("AL", "铝", "SHFE", "base_metal", 5, 5, rolls=12),
    _i("ZN", "锌", "SHFE", "base_metal", 5, 5, rolls=12),
    _i("NI", "镍", "SHFE", "base_metal", 1, 10, 0.16, 12),
    _i("RB", "螺纹钢", "SHFE", "ferrous", 10, 1),
    _i("HC", "热卷", "SHFE", "ferrous", 10, 1),
    _i("I", "铁矿石", "DCE", "ferrous", 100, 0.5, 0.15),
    _i("J", "焦炭", "DCE", "ferrous", 100, 0.5, 0.2),
    _i("JM", "焦煤", "DCE", "ferrous", 60, 0.5, 0.2),
    _i("SC", "原油", "INE", "energy_chem", 1000, 0.1, 0.15, 12),
    _i("FU", "燃料油", "SHFE", "energy_chem", 10, 1, 0.15),
    _i("RU", "橡胶", "SHFE", "energy_chem", 10, 5),
    _i("TA", "PTA", "CZCE", "energy_chem", 5, 2),
    _i("MA", "甲醇", "CZCE", "energy_chem", 10, 1),
    _i("PP", "聚丙烯", "DCE", "energy_chem", 5, 1),
    _i("EG", "乙二醇", "DCE", "energy_chem", 10, 1),
    _i("SA", "纯碱", "CZCE", "energy_chem", 20, 1, 0.15),
    _i("FG", "玻璃", "CZCE", "energy_chem", 20, 1, 0.15),
    _i("M", "豆粕", "DCE", "agri", 10, 1),
    _i("Y", "豆油", "DCE", "agri", 10, 2),
    _i("P", "棕榈油", "DCE", "agri", 10, 2),
    _i("C", "玉米", "DCE", "agri", 10, 1),
    _i("SR", "白糖", "CZCE", "agri", 10, 1),
    _i("CF", "棉花", "CZCE", "agri", 5, 5),
    _i("RM", "菜粕", "CZCE", "agri", 10, 1),
    _i("LH", "生猪", "DCE", "agri", 16, 5, 0.15),
    _i("AP", "苹果", "CZCE", "agri", 10, 1, 0.15),
]}
