"""通达信 / 文华 公式子集的解释器，只用于测试 formulas/ 里的公式逻辑与 Python 版一致。

覆盖公式里用到的语法和函数；“无效值”按最严格的方式处理（参与比较、逻辑运算的结果仍是无效值），
所以只要公式在这里能和 Python 版对上，就说明预热期的保护条件写够了。
不能替代在行情软件里的实际编译运行。
"""
import re

import numpy as np
import pandas as pd

_TOKEN = re.compile(r"\s*(?:(\d+\.?\d*)|('[^']*')|([A-Za-z_一-鿿][A-Za-z0-9_一-鿿]*)"
                    r"|(<>|>=|<=|&&|\|\||!=|==|[-+*/()<>=,!]))")
_BINARY = {"OR": 1, "||": 1, "AND": 2, "&&": 2,
           "=": 3, "==": 3, "<>": 3, "!=": 3, ">": 3, "<": 3, ">=": 3, "<=": 3,
           "+": 4, "-": 4, "*": 5, "/": 5}
_DRAW = {"DRAWICON", "DRAWTEXT", "STICKLINE", "DRAWKLINE"}
_ACTIONS = {"BK", "SK", "SP", "BP"}


def _tokens(src):
    out, pos = [], 0
    src = src.strip()
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        if not m or m.end() == pos:
            raise SyntaxError(f"无法解析: {src[pos:pos + 20]!r}")
        num, string, name, op = m.groups()
        out.append(("num", float(num)) if num else ("str", string[1:-1]) if string
                   else ("name", name.upper()) if name else ("op", op.upper()))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, value=None):
        tok = self.peek()
        if value is not None and tok[1] != value:
            raise SyntaxError(f"期望 {value}，得到 {tok}")
        self.i += 1
        return tok

    def expr(self, min_prec=1):
        left = self.unary()
        while True:
            kind, val = self.peek()
            if kind not in ("op", "name") or val not in _BINARY or _BINARY[val] < min_prec:
                return left
            self.take()
            left = ("bin", val, left, self.expr(_BINARY[val] + 1))

    def unary(self):
        kind, val = self.peek()
        if val in ("-", "NOT", "!"):
            self.take()
            return ("neg" if val == "-" else "not", self.unary())
        return self.atom()

    def atom(self):
        kind, val = self.take()
        if kind in ("num", "str"):
            return (kind, val)
        if val == "(":
            e = self.expr()
            self.take(")")
            return e
        if kind == "name":
            if self.peek()[1] == "(":
                self.take("(")
                args = [self.expr()]
                while self.peek()[1] == ",":
                    self.take(",")
                    args.append(self.expr())
                self.take(")")
                return ("call", val, args)
            return ("var", val)
        raise SyntaxError(f"意外的符号 {val}")


def _split_top(stmt):
    """按最外层逗号拆分（括号、引号内的不拆）。"""
    parts, depth, quote, cur = [], 0, False, ""
    for ch in stmt:
        if ch == "'":
            quote = not quote
        elif not quote and ch == "(":
            depth += 1
        elif not quote and ch == ")":
            depth -= 1
        if ch == "," and depth == 0 and not quote:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return parts + [cur]


def _strict(fn, *xs):
    with np.errstate(all="ignore"):
        bad = np.zeros(len(xs[0]), bool)
        for x in xs:
            bad |= np.isnan(x)
        return np.where(bad, np.nan, fn(*xs).astype(float))


class Formula:
    """result.vars：变量名 -> 序列；result.signals：[(条件序列, 'BK'/'SK'/'SP'/'BP')]；result.autofilter。"""

    def __init__(self, df: pd.DataFrame):
        self.n = len(df)
        o, h, low, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        self.vars = {"O": o, "OPEN": o, "H": h, "HIGH": h, "L": low, "LOW": low, "C": c, "CLOSE": c,
                     "BARPOS": np.arange(1, self.n + 1, dtype=float)}
        self.signals, self.autofilter = [], False

    def run(self, text):
        text = re.sub(r"\{[^}]*\}", "", text)          # 通达信注释
        text = re.sub(r"//[^\n]*", "", text)            # 文华注释
        for stmt in text.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._statement(stmt)
        return self

    def _statement(self, stmt):
        if stmt.upper() == "AUTOFILTER":
            self.autofilter = True
            return
        head, *rest = _split_top(stmt)
        if rest and rest[-1].strip().upper() in _ACTIONS:
            self.signals.append((self._eval(head), rest[-1].strip().upper()))
            return
        m = re.match(r"^\s*([A-Za-z_一-鿿][\w一-鿿]*)\s*(:=|:)(.*)$", head, re.S)
        if m:
            self.vars[m.group(1).upper()] = self._eval(m.group(3))
        else:
            self._eval(head)  # 画图函数：只检查参数能否计算

    def _eval(self, src):
        p = _Parser(_tokens(src))
        node = p.expr()
        if p.i != len(p.t):
            raise SyntaxError(f"多余的内容: {src!r}")
        return self._node(node)

    def _series(self, v):
        return np.full(self.n, float(v)) if np.isscalar(v) else v

    def _node(self, node):
        kind = node[0]
        if kind == "num":
            return self._series(node[1])
        if kind == "str":
            return node[1]
        if kind == "var":
            if node[1].startswith("COLOR"):   # 颜色常量（文华 STICKLINE 的颜色参数）
                return self._series(0)
            if node[1] not in self.vars:
                raise NameError(f"未定义的变量 {node[1]}")
            return self.vars[node[1]]
        if kind == "neg":
            return -self._node(node[1])
        if kind == "not":
            return _strict(lambda a: a == 0, self._node(node[1]))
        if kind == "bin":
            return self._binary(node[1], self._node(node[2]), self._node(node[3]))
        return self._call(node[1], node[2])

    def _binary(self, op, a, b):
        with np.errstate(all="ignore"):
            if op in ("+", "-", "*"):
                return {"+": a + b, "-": a - b, "*": a * b}[op]
            if op == "/":
                r = a / b
                return np.where(np.isfinite(r), r, np.nan)
        fn = {"=": np.equal, "==": np.equal, "<>": np.not_equal, "!=": np.not_equal,
              ">": np.greater, "<": np.less, ">=": np.greater_equal, "<=": np.less_equal,
              "AND": lambda x, y: (x != 0) & (y != 0), "&&": lambda x, y: (x != 0) & (y != 0),
              "OR": lambda x, y: (x != 0) | (y != 0), "||": lambda x, y: (x != 0) | (y != 0)}[op]
        return _strict(fn, a, b)

    def _call(self, name, arg_nodes):
        args = [self._node(a) for a in arg_nodes]
        if name in _DRAW:
            return None
        s = lambda x: pd.Series(x)  # noqa: E731
        n = lambda x: int(x[0])     # noqa: E731  周期参数取常数
        if name == "MA":
            return s(args[0]).rolling(n(args[1]), min_periods=n(args[1])).mean().to_numpy()
        if name == "STD":
            return s(args[0]).rolling(n(args[1]), min_periods=n(args[1])).std(ddof=1).to_numpy()
        if name == "SUM":
            return s(args[0]).rolling(n(args[1]), min_periods=n(args[1])).sum().to_numpy()
        if name == "HHV":
            return s(args[0]).rolling(n(args[1]), min_periods=n(args[1])).max().to_numpy()
        if name == "LLV":
            return s(args[0]).rolling(n(args[1]), min_periods=n(args[1])).min().to_numpy()
        if name == "REF":
            return s(args[0]).shift(n(args[1])).to_numpy()
        if name == "ABS":
            return np.abs(args[0])
        if name == "ATAN":
            return np.arctan(args[0])
        if name in ("MAX", "MIN"):
            return _strict(np.maximum if name == "MAX" else np.minimum, args[0], args[1])
        if name in ("IF", "IFELSE"):
            cond, a, b = args
            return np.where(np.isnan(cond), np.nan, np.where(cond != 0, a, b))
        if name == "BARSCOUNT":
            valid = ~np.isnan(args[0])
            return np.where(valid, np.cumsum(valid), np.nan)
        if name == "EMA":
            return self._recursive(args[0], np.full(self.n, 2 / (n(args[1]) + 1)))
        if name == "DMA":
            return self._recursive(args[0], args[1])
        raise NameError(f"解释器不支持函数 {name}")

    def _recursive(self, x, a):
        """Y = A*X + (1-A)*Y'，首个有效值处 Y = X（通达信/文华 EMA、DMA 的算法）。"""
        out, y = np.full(self.n, np.nan), np.nan
        for i in range(self.n):
            if np.isnan(x[i]) or np.isnan(a[i]):
                continue
            y = x[i] if np.isnan(y) else a[i] * x[i] + (1 - a[i]) * y
            out[i] = y
        return out

    def positions(self):
        """按文华 AUTOFILTER 一开一平、每根K线最多一个信号、按代码顺序生效，模拟持仓。"""
        allowed = {"BK": (0, 1), "SK": (0, -1), "SP": (1, 0), "BP": (-1, 0)}
        pos, out = 0, np.zeros(self.n, dtype=int)
        for i in range(self.n):
            for cond, act in self.signals:
                frm, to = allowed[act]
                if not np.isnan(cond[i]) and cond[i] != 0 and (pos == frm or not self.autofilter):
                    pos = to
                    break
            out[i] = pos
        return out


def sections(text):
    """副图文件里有多个公式，按“===== 公式 N / 指标 N”分隔线拆开。"""
    parts = re.split(r"^.*=====\s*(?:公式|指标)\s*\d+.*$", text, flags=re.M)
    return [p for p in parts[1:] if p.strip()]
