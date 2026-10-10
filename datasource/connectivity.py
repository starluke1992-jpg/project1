"""网络连通性自检：逐个访问 AKShare 用到的数据网站，看当前环境能否连上。

判断方法：服务器有 HTTP 响应（哪怕是 404）就算连得上；连接被代理拒绝（403）、
域名解析失败、超时都算连不上。

云端环境的出网代理只支持 https，所以下表里标为 http 的网站在云端放行了域名也用不了，
需要在本地运行。
"""
import urllib.error
import urllib.request

# (用途, 域名, 协议)
HOSTS = [
    ("ETF、A股指数日线（东方财富）", "push2his.eastmoney.com", "https"),
    ("A股指数代码表（东方财富）", "80.push2.eastmoney.com", "https"),
    ("期货主力连续（新浪）", "stock2.finance.sina.com.cn", "https"),
    ("美股/港股指数（新浪）", "finance.sina.com.cn", "https"),
    ("国债收益率（新浪）", "bond.finance.sina.com.cn", "https"),
    ("全收益指数（中证指数）", "www.csindex.com.cn", "https"),
    ("黄金现货（上金所）", "www.sge.com.cn", "https"),
    ("人民币汇率中间价（外管局）", "www.safe.gov.cn", "https"),
    ("回购定盘利率（外汇交易中心）", "www.chinamoney.com.cn", "https"),
    ("逐合约日线：上期所", "www.shfe.com.cn", "https"),
    ("逐合约日线：上期能源", "www.ine.cn", "https"),
    ("逐合约日线：大商所", "www.dce.com.cn", "http"),
    ("逐合约日线：郑商所", "www.czce.com.cn", "http"),
    ("逐合约日线：中金所", "www.cffex.com.cn", "http"),
    ("逐合约日线：广期所", "www.gfex.com.cn", "http"),
    ("手续费与保证金（openctp）", "openctp.cn", "https"),
]

_UA = {"User-Agent": "Mozilla/5.0"}


def probe(host: str, scheme: str = "https", timeout: float = 8.0):
    """返回 (是否连得上, 说明)。"""
    req = urllib.request.Request(f"{scheme}://{host}/", method="HEAD", headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return True, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, str(e.reason)
    except OSError as e:
        return False, str(e)


def check_all(hosts=HOSTS, timeout: float = 8.0):
    """逐个探测，返回 [(用途, 域名, 协议, 是否连得上, 说明), ...]。"""
    return [(purpose, host, scheme, *probe(host, scheme, timeout)) for purpose, host, scheme in hosts]
