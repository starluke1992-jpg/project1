"""国内期货品种 -> 板块。B2 用它做板块风险上限，B1 用它划分大类。"""

SECTORS = {
    "股指": ["IF", "IH", "IC", "IM"],
    "利率": ["T", "TF", "TS", "TL"],
    "贵金属": ["AU", "AG"],
    "有色": ["CU", "AL", "ZN", "PB", "NI", "SN", "SS", "AO", "BC", "SI", "LC", "PS"],
    "黑色": ["RB", "HC", "I", "J", "JM", "SF", "SM", "WR"],
    "能化": ["SC", "FU", "LU", "BU", "PG", "TA", "MA", "EG", "EB", "PP", "L", "V", "RU", "NR",
           "SA", "FG", "UR", "SP", "PF", "PX", "SH", "BR"],
    "农产品": ["M", "Y", "P", "A", "B", "C", "CS", "CF", "SR", "OI", "RM", "AP", "CJ", "JD", "LH",
            "PK", "CY", "RR", "WH", "PM", "RI", "JR", "LR", "RS"],
    "航运": ["EC"],
}

VARIETY_SECTOR = {v: s for s, vs in SECTORS.items() for v in vs}

# B1 的大类划分
B1_CLASS = {"股指": "equity", "利率": "bond", "贵金属": "gold"}


def sector_of(variety: str) -> str:
    return VARIETY_SECTOR.get(variety.upper(), "其他")
