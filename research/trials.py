"""试验登记：每次用真实数据回测都追加一行到 research/trials.csv，失败的试验也记。

最终版本要报告一共试了多少次（docs/plan_A_B.md 1.2 节）。合成数据的运行不登记。
"""
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

TRIALS = Path(__file__).resolve().parent / "trials.csv"
FIELDS = ["时间", "代码版本", "方案", "策略", "区间", "年化收益", "年化波动", "夏普", "最大回撤", "参数", "备注"]


def _commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=TRIALS.parent, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                               cwd=TRIALS.parent, timeout=5).stdout.strip()
        return out.stdout.strip() + ("+未提交改动" if dirty else "")
    except Exception:
        return ""


def log_trial(plan: str, strategy: str, period: str, stats: dict, params: dict, note: str = "",
              path: Path = TRIALS) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "时间": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "代码版本": _commit(), "方案": plan,
            "策略": strategy, "区间": period,
            "年化收益": f"{stats.get('年化收益', float('nan')):.4f}",
            "年化波动": f"{stats.get('年化波动', float('nan')):.4f}",
            "夏普": f"{stats.get('夏普', float('nan')):.3f}",
            "最大回撤": f"{stats.get('最大回撤', float('nan')):.4f}",
            "参数": json.dumps(params, ensure_ascii=False, default=str), "备注": note,
        })


def count_trials(plan: str = None, path: Path = TRIALS) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for row in csv.DictReader(f) if plan is None or row["方案"] == plan)
