#!/usr/bin/env python3
"""统一分析入口 —— 一条命令复现论文正文的全部数字（含记账）。

论文声称「正文每个率、区间、检验统计量与记账总数都由分析脚本从 per-episode
日志重新生成」。本脚本把这五个子分析串成一个入口，并补上论文 Accounting 句
（报告数据集 episode 总数 / no-verdict 率 / 各档排除率极差）：

1. ``paired_stats.py``      E1-E8 主分析（球速扫描 / 限位代价 / 标称四档 /
                            扰动网格 / 高功效角点 / 限速×机制）
2. ``aux_stats.py``         敏感性 sweep（exp18）+ TCP 超限诊断 + 豁免窗复核
3. ``extract_chat8_aux.py`` I(t,s) 交互 / both-succeed 子集 / hard-min 配对
4. ``check_shared_input.py``共享输入（ballstate）交叉验证
5. ``analyze_chat12_night.py`` 夜跑五实验（exp23-27：不冻结端到端 / 统一共享输入 /
                            候选选择基线 / 速度×TCP 网格 / 击球质量）

用法::

    python scripts/extract/analyze_all.py                  # 全部子分析 + 记账
    python scripts/extract/analyze_all.py --accounting-only  # 只打印记账

记账口径：只统计「论文正文引用了其数字」的数据集（README 证据表 E1-E4/E6-E9），
artifact-only 的感知 sweeps（exp17a_noise / exp17c_obsfreq）与研发期扫描
（exp13_arch / exp14_pd_v2）不计入。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
EXTRACT = PROJECT / "scripts" / "extract"
DATA = PROJECT / "experiment_data"

# 论文报告数据集（README 证据表）：目录名 -> 论文中的角色
REPORTED: dict[str, str] = {
    "exp15_speed_v2": "RQ0 球速能力包线",
    "exp16_limits_v2": "RQ0 限位代价（TCP1.8 vs 1.0）",
    "exp17b_perturb": "RQ1 扰动网格 full/point-target",
    "exp17d_mechanism": "RQ1 标称四档",
    "exp17e_perturb7": "RQ1 7 m/s 网格复证",
    "exp17f_mechanism_perturb": "RQ1 机制归因（扰动下）",
    "exp17g_spatial_power": "RQ2 高功效角点",
    "exp17h_extreme": "RQ2 极端条件角点",
    "exp17i_limits_ablation": "RQ3 限速×机制",
    "exp18_sensitivity": "敏感性 sweep（β/窗口/半径）",
    "exp19_hardmin": "hard-min 基线 @9 m/s",
    "exp21_hardmin_tcp10": "hard-min @TCP1.0",
    "exp22_shared_spatial": "共享输入交叉验证",
    "exp23_nofreeze": "RQ4 不冻结端到端",
    "exp24_shared_full": "RQ2 端到端共享输入",
    "exp25_select_one": "RQ1 候选选择基线",
    "exp26_tcp_grid": "RQ3 速度×TCP 网格",
    "exp27_ball_quality": "RQ1 击球质量",
}
SUBMODULES: tuple[str, ...] = (
    "paired_stats", "aux_stats", "extract_chat8_aux", "check_shared_input",
    "analyze_chat12_night",
)


def experiment_stat(name: str) -> dict[str, Any]:
    """统计一个实验目录：行数、error 行数、各档排除率极差（pp）。"""
    path = DATA / name / "results.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    per_tier: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    errors = 0
    for r in rows:
        tier = str(json.loads(r["config_json"]).get("--ablation", "(单档)"))
        per_tier[tier][0] += 1
        if (r.get("error") or "").strip():
            errors += 1
            per_tier[tier][1] += 1
    shares = [100 * e / n for n, e in per_tier.values() if n]
    return {
        "rows": len(rows),
        "errors": errors,
        "spread_pp": max(shares) - min(shares) if shares else 0.0,
        "tiers": len(per_tier),
    }


def print_accounting() -> None:
    """打印论文 Accounting 句的三个量（论文 results.tex Limitations 末段）。"""
    total = 0
    errs = 0
    spread = 0.0
    worst = "无"
    for name, role in REPORTED.items():
        st = experiment_stat(name)
        total += st["rows"]
        errs += st["errors"]
        if st["spread_pp"] > spread:
            spread, worst = st["spread_pp"], name
        print(f"  {name:24s} {st['rows']:6d} 行  error {st['errors']:4d}  "
              f"档间差 {st['spread_pp']:.2f} pp  [{role}]")
    print(f"\n  记账合计: {total} episodes, no-verdict {errs} = "
          f"{100 * errs / total:.2f}%  (论文写 1.3%)")
    print(f"  各档排除率最大极差: {spread:.2f} pp ({worst})  (论文写 at most 0.5 pp)")


def run_submodules() -> None:
    """依次运行五个子分析（各自打印结果并刷新其跟踪产物）。

    analyze_chat12_night 为顶层执行脚本（无 main()），用 runpy 以 __main__ 运行。
    """
    sys.path.insert(0, str(EXTRACT))
    for name in SUBMODULES:
        print("\n" + "═" * 78)
        print(f"【{name}】")
        print("═" * 78)
        if name == "analyze_chat12_night":
            import runpy
            runpy.run_path(str(EXTRACT / "analyze_chat12_night.py"),
                           run_name="__main__")
            continue
        mod = importlib.import_module(name)
        main = getattr(mod, "main", None)
        if main is None:
            print("  (无 main()，跳过)")
            continue
        main()


def main() -> None:
    """解析参数：默认全跑，--accounting-only 只打印记账。"""
    parser = argparse.ArgumentParser(description="论文数字统一复现入口")
    parser.add_argument("--accounting-only", action="store_true",
                        help="只打印记账（episode 总数 / no-verdict / 档间差）")
    args = parser.parse_args()
    if not args.accounting_only:
        run_submodules()
        print("\n" + "═" * 78)
    print("【记账 Accounting】")
    print("═" * 78)
    print_accounting()


if __name__ == "__main__":
    main()
