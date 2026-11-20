#!/usr/bin/env python3
"""统一分析入口 —— 一条命令复现论文正文的全部数字（含记账）。

论文声称「正文每个率、区间、检验统计量与记账总数都由分析脚本从 per-episode
日志重新生成」。本脚本把这五个子分析串成一个入口，并补上论文 Accounting 句
（报告数据集 episode 总数 / no-verdict 率 / 各档排除率极差）：

1. ``paired_stats.py``      E1-E8 主分析（球速扫描 / 限位代价 / 标称四档 /
                            扰动网格 / 高功效角点 / 限速×机制）+ E9 共享输入 + 表格数据
2. ``aux_stats.py``         敏感性 sweep（exp18）+ TCP 超限诊断 + 豁免窗复核
3. ``extract_chat8_aux.py`` I(t,s) 交互 / both-succeed 子集 / hard-min 配对
4. ``check_shared_input.py``共享输入（ballstate）交叉验证
5. ``analyze_chat12_night.py`` 夜跑五实验（exp23-27：不冻结端到端 / 统一共享输入 /
                            候选选择基线 / 速度×TCP 网格 / 击球质量）
6. ``analyze_exp28_collision.py`` 碰撞窗口验证（exp28 常开碰撞 vs 窗口口径裁决对照）

用法::

    python scripts/extract/analyze_all.py                       # 全部子分析 + 记账
    python scripts/extract/analyze_all.py --accounting-only     # 只打印记账
    python scripts/extract/analyze_all.py --data-root DIR --output DIR   # 指定根与输出

- `--data-root`：实验数据根（默认 `experiment_data/`，旧行为不变）。所有子分析
  都以**同一个** `--data-root` 调用，子模块不得回退默认根。
- `--output`：输出目录；透传给每个子模块（其 JSON 落在同一目录），并写
  `<output>/analyze_all.json` 记录记账与各子模块状态。缺省只打印。
- `--dry-run`：只打印，不写任何文件（也不让子模块写）。

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
DEFAULT_DATA_ROOT = PROJECT / "experiment_data"

sys.path.insert(0, str(EXTRACT))

from paired_stats import tier_of  # noqa: E402

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
    "exp28_always_collision": "碰撞窗口验证（常开对照 180 次重跑）",
}
SUBMODULES: tuple[str, ...] = (
    "paired_stats", "aux_stats", "extract_chat8_aux", "check_shared_input",
    "analyze_chat12_night", "analyze_exp28_collision",
)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（--data-root / --output / --dry-run / --accounting-only）。"""
    parser = argparse.ArgumentParser(
        description="论文数字统一复现入口",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=None,
                        help=f"实验数据根（默认 {DEFAULT_DATA_ROOT}）；"
                             "同一值透传给所有子分析")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出目录（透传子模块；写 <output>/analyze_all.json）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印，不写任何文件（子模块也不写）")
    parser.add_argument("--accounting-only", action="store_true",
                        help="只打印记账（episode 总数 / no-verdict / 档间差）")
    return parser


def experiment_stat(data_root: Path, name: str) -> dict[str, Any] | None:
    """统计一个实验目录：行数、error 行数、各档排除率极差（pp）；缺文件返回 None。"""
    path = data_root / name / "results.csv"
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    per_tier: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    errors = 0
    for r in rows:
        row = dict(r)
        row["config_json"] = r.get("config_json", "{}")
        tier = tier_of(row) or "(单档)"
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


def compute_accounting(data_root: Path) -> dict[str, Any]:
    """计算论文 Accounting 句的三个量；缺失实验记为 unavailable。"""
    total = 0
    errs = 0
    spread = 0.0
    worst = "无"
    missing: list[str] = []
    experiments: dict[str, Any] = {}
    for name, role in REPORTED.items():
        st = experiment_stat(data_root, name)
        if st is None:
            missing.append(name)
            experiments[name] = {"role": role, "unavailable_reason": "missing_csv"}
            continue
        experiments[name] = {"role": role, **st}
        total += st["rows"]
        errs += st["errors"]
        if st["spread_pp"] > spread:
            spread, worst = st["spread_pp"], name
    result: dict[str, Any] = {
        "data_root": str(data_root.resolve()),
        "experiments": experiments,
        "missing": missing,
        "totals": {"episodes": total, "no_verdict": errs,
                   "no_verdict_pct": (100 * errs / total) if total else None,
                   "max_tier_error_spread_pp": spread,
                   "worst_spread_experiment": worst if total else None},
    }
    if total == 0:
        result["unavailable_reason"] = "no_reported_experiment_csv"
    return result


def print_accounting(data: dict[str, Any]) -> None:
    """打印论文 Accounting 句的三个量（论文 results.tex Limitations 末段）。"""
    for name, role in REPORTED.items():
        st = data["experiments"].get(name, {})
        if "unavailable_reason" in st:
            print(f"  {name:24s} 不可用（{st['unavailable_reason']}）  [{role}]")
            continue
        print(f"  {name:24s} {st['rows']:6d} 行  error {st['errors']:4d}  "
              f"档间差 {st['spread_pp']:.2f} pp  [{role}]")
    totals = data["totals"]
    if totals["no_verdict_pct"] is None:
        print("\n  记账合计: 不可估计（无任何报告数据集 CSV）")
        return
    print(f"\n  记账合计: {totals['episodes']} episodes, "
          f"no-verdict {totals['no_verdict']} = {totals['no_verdict_pct']:.2f}%  "
          f"(论文写 1.3%)")
    print(f"  各档排除率最大极差: {totals['max_tier_error_spread_pp']:.2f} pp "
          f"({totals['worst_spread_experiment']})  (论文写 at most 0.5 pp)")


def _submodule_argv(data_root: Path, output: Path | None,
                    dry_run: bool) -> list[str]:
    """构造传给子模块的 argv：data-root 与 output 同源透传。"""
    argv = ["--data-root", str(data_root)]
    if output is not None:
        argv += ["--output", str(output)]
    if dry_run:
        argv.append("--dry-run")
    return argv


def run_submodules(data_root: Path, output: Path | None,
                   dry_run: bool) -> dict[str, Any]:
    """依次运行六个子分析（全部收到同一 --data-root / --output）。"""
    status: dict[str, Any] = {}
    shared_argv = _submodule_argv(data_root, output, dry_run)
    for name in SUBMODULES:
        print("\n" + "═" * 78)
        print(f"【{name}】  argv: {name}.py {' '.join(shared_argv)}")
        print("═" * 78)
        mod = importlib.import_module(name)
        main = getattr(mod, "main", None)
        if main is None:
            print("  (无 main()，跳过)")
            status[name] = {"status": "skipped", "reason": "no_main"}
            continue
        try:
            rc = main(list(shared_argv))
        except FileNotFoundError as exc:
            print(f"  不可估计：{exc}")
            status[name] = {"status": "unavailable", "reason": str(exc)}
            continue
        status[name] = {"status": "ok", "returncode": rc}
    return status


def main(argv: list[str] | None = None) -> int:
    """解析参数：默认全跑，--accounting-only 只打印记账。"""
    args = build_parser().parse_args(argv)
    data_root: Path = args.data_root if args.data_root is not None else DEFAULT_DATA_ROOT
    accounting = compute_accounting(data_root)
    status: dict[str, Any] = {}
    if not args.accounting_only:
        status = run_submodules(data_root, args.output, args.dry_run)
        print("\n" + "═" * 78)
    print("【记账 Accounting】")
    print("═" * 78)
    print_accounting(accounting)
    if args.output is not None:
        target = args.output / "analyze_all.json"
        payload = {"data_root": str(data_root.resolve()),
                   "accounting": accounting,
                   "submodules": status,
                   "argv": {"data_root": str(data_root), "output": str(args.output),
                            "dry_run": args.dry_run,
                            "accounting_only": args.accounting_only}}
        if args.dry_run:
            print(f"\n[analyze_all] dry-run：不写文件（目标 {target}）")
        else:
            print(f"\n[analyze_all] 写出目标 {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    elif not args.dry_run:
        print("\n[analyze_all] 未给 --output：只打印，不写文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
