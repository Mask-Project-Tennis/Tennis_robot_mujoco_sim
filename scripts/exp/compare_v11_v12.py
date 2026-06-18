#!/usr/bin/env python3
"""V11 vs V12 可靠性对比实验。

运行 100 seeds × 2 版本 × 2 模式 = 400 次仿真，
对比命中率、位置误差、求解时间。

用法:
    python scripts/exp/compare_v11_v12.py [--seeds 100] [--speed 7]

输出:
    experiment_data/v11_v12_comparison/results.csv
    experiment_data/v11_v12_comparison/summary.txt
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT / "experiment_data" / "v11_v12_comparison"
RESULT_FIELDS = [
    "pos_error", "vel_error", "min_dist", "ball_near_ms", "tube_ready_ms",
    "max_tcp", "max_qdot", "max_face", "hit_type", "hit_time_error_ms",
    "hit_pos_error", "v_racket_at_hit",
]

# __RESULT__ 解析正则
RESULT_RE = re.compile(r"__RESULT__: (.+)")
# REPLAN t=XXXms 解析（取首次规划 t= 或最大 t=）
REPLAN_RE = re.compile(r"REPLAN.*t=(\d+)ms")


def run_one(script: str, seed: int, extra_args: list[str], speed: int) -> dict:
    """运行一次仿真，返回解析后的指标 dict。

    Args:
        script: 脚本文件名（rm65_mpc_v11.py 或 rm65_mpc_v12.py）。
        seed: 随机种子。
        extra_args: 额外 CLI 参数（如 --position-mode）。
        speed: 球速 m/s。

    Returns:
        指标字典，失败时含 error 键。
    """
    cmd = [
        sys.executable, str(PROJECT / "scripts" / script),
        "--serve-box", "--ball-speed", str(speed),
        "--seed", str(seed), "--no-plot",
    ] + extra_args

    t_wall_start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT),
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "seed": seed}
    t_wall = time.perf_counter() - t_wall_start

    stdout = result.stdout + result.stderr

    # 解析 __RESULT__
    m = RESULT_RE.search(stdout)
    if not m:
        return {"error": "no_result", "seed": seed, "wall_time": t_wall}

    fields = dict(re.findall(r"(\S+)=(\S+)", m.group(1)))

    # 数值化
    for k in RESULT_FIELDS:
        if k in fields:
            try:
                fields[k] = float(fields[k])
            except ValueError:
                pass  # hit_type 保持字符串

    # 解析 REPLAN t=XXXms（首次规划 = 最大 t）
    replan_times = REPLAN_RE.findall(stdout)
    if replan_times:
        fields["first_plan_ms"] = max(int(t) for t in replan_times)
        fields["total_replan_ms"] = sum(int(t) for t in replan_times)
        fields["replan_count"] = len(replan_times)
    else:
        fields["first_plan_ms"] = 0
        fields["total_replan_ms"] = 0
        fields["replan_count"] = 0

    fields["wall_time"] = round(t_wall, 2)
    fields["seed"] = seed

    # hit 判定（pos_error < 0.153m 且 hit_type != miss）
    hit_type = fields.get("hit_type", "miss")
    pos_err = fields.get("pos_error", 999)
    fields["hit"] = (hit_type in ("active", "passive")) or (pos_err < 0.153)

    return fields


def main():
    import argparse

    parser = argparse.ArgumentParser(description="V11 vs V12 对比实验")
    parser.add_argument("--seeds", type=int, default=100, help="seed 数量（默认 100）")
    parser.add_argument("--speed", type=int, default=7, help="球速 m/s（默认 7）")
    parser.add_argument("--start-seed", type=int, default=1, help="起始 seed（默认 1）")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "results.csv"

    configs = [
        ("rm65_mpc_v11.py", "V11_torque", []),
        ("rm65_mpc_v12.py", "V12_torque", []),
        ("rm65_mpc_v11.py", "V11_position", ["--position-mode"]),
        ("rm65_mpc_v12.py", "V12_position", ["--position-mode"]),
    ]

    seeds = range(args.start_seed, args.start_seed + args.seeds)
    total = len(seeds) * len(configs)
    done = 0

    # 写 CSV header
    all_fields = RESULT_FIELDS + [
        "first_plan_ms", "total_replan_ms", "replan_count",
        "wall_time", "hit", "seed", "label",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()

        for seed in seeds:
            for script, label, extra in configs:
                done += 1
                r = run_one(script, seed, extra, args.speed)
                r["label"] = label

                hit = r.get("hit_type", "?")
                pos = r.get("pos_error", "?")
                print(
                    f"[{done}/{total}] seed={seed} {label}: "
                    f"hit={hit} pos_err={pos}",
                    flush=True,
                )

                writer.writerow(r)
                f.flush()

    print(f"\nCSV 已保存: {csv_path}")

    # 汇总统计
    summary_path = OUTPUT_DIR / "summary.txt"
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open(summary_path, "w") as s:
        s.write("V11 vs V12 可靠性对比实验汇总\n")
        s.write(f"Seeds: {args.seeds}, Speed: {args.speed} m/s\n")
        s.write("=" * 60 + "\n\n")

        for label in [c[1] for c in configs]:
            group = [r for r in rows if r.get("label") == label and "error" not in r]
            n = len(group)
            if n == 0:
                s.write(f"{label}: 无有效数据\n\n")
                continue

            hits = sum(1 for r in group if r.get("hit") == "True")
            pos_errors = [float(r["pos_error"]) for r in group if r.get("pos_error")]
            vel_errors = [float(r["vel_error"]) for r in group if r.get("vel_error")]
            plan_times = [float(r["first_plan_ms"]) for r in group if r.get("first_plan_ms")]
            wall_times = [float(r["wall_time"]) for r in group if r.get("wall_time")]

            def mean_std(vals):
                if not vals:
                    return 0, 0
                m = sum(vals) / len(vals)
                s_val = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5
                return m, s_val

            pos_m, pos_s = mean_std(pos_errors)
            vel_m, vel_s = mean_std(vel_errors)
            plan_m, plan_s = mean_std(plan_times)
            wall_m, wall_s = mean_std(wall_times)

            s.write(f"{label} (n={n}):\n")
            s.write(f"  命中率: {hits}/{n} = {hits/n*100:.1f}%\n")
            s.write(f"  pos_error: {pos_m:.4f} ± {pos_s:.4f} m\n")
            s.write(f"  vel_error: {vel_m:.4f} ± {vel_s:.4f} m/s\n")
            s.write(f"  首次规划: {plan_m:.0f} ± {plan_s:.0f} ms\n")
            s.write(f"  总耗时: {wall_m:.1f} ± {wall_s:.1f} s\n")
            s.write("\n")

    print(f"汇总已保存: {summary_path}")
    with open(summary_path, "r") as s:
        print(s.read())


if __name__ == "__main__":
    main()
