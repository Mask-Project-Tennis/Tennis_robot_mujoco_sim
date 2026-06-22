#!/usr/bin/env python3
"""exp15: V12 力矩模式多球速鲁棒性验证。

在 5/7/9/12 m/s 全球速范围内验证 V12 命中率稳定性，
生成球速-命中率曲线数据用于论文。

用法:
    python scripts/exp/run_exp15_speed_sweep.py [--seeds 50] [--start-seed 1]

输出:
    experiment_data/exp15_v12_speed_sweep/results.csv
    experiment_data/exp15_v12_speed_sweep/summary.txt

参数矩阵:
    ball_speed ∈ [5, 7, 9, 12] m/s × 50 seeds = 200 runs
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))
OUTPUT_DIR = PROJECT / "experiment_data" / "exp15_v12_speed_sweep"
SCRIPT = PROJECT / "scripts" / "rm65_mpc_v12.py"

RESULT_FIELDS = [
    "pos_error", "vel_error", "min_dist", "ball_near_ms", "tube_ready_ms",
    "max_tcp", "max_qdot", "max_face", "hit_type", "hit_time_error_ms",
    "hit_pos_error", "v_racket_at_hit",
]
RESULT_RE = re.compile(r"__RESULT__: (.+)")
SPEEDS = [7, 9, 12, 15]


def run_one(speed: int, seed: int) -> dict:
    """运行一次 V12 力矩模式仿真。

    Args:
        speed: 球速 m/s。
        seed: 随机种子。

    Returns:
        指标字典，失败时含 error 键。
    """
    cmd = [
        sys.executable, str(SCRIPT),
        "--serve-box", "--ball-speed", str(speed),
        "--seed", str(seed), "--no-plot",
    ]

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT), encoding="utf-8",
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "seed": seed, "speed": speed}
    wall = time.perf_counter() - t0

    stdout = result.stdout + result.stderr
    m = RESULT_RE.search(stdout)
    if not m:
        return {"error": "no_result", "seed": seed, "speed": speed,
                "wall_time": round(wall, 2)}

    fields = dict(re.findall(r"(\S+)=(\S+)", m.group(1)))
    for k in RESULT_FIELDS:
        if k in fields:
            try:
                fields[k] = float(fields[k])
            except ValueError:
                pass

    from src.sim.hit_detection import determine_hit_from_type
    hit_type = fields.get("hit_type", "miss")
    fields["hit"] = determine_hit_from_type(hit_type)
    fields["wall_time"] = round(wall, 2)
    fields["seed"] = seed
    fields["speed"] = speed
    return fields


def main():
    import argparse

    parser = argparse.ArgumentParser(description="exp15: V12 多球速鲁棒性")
    parser.add_argument("--seeds", type=int, default=50, help="每球速的 seed 数")
    parser.add_argument("--start-seed", type=int, default=1, help="起始 seed")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "results.csv"

    seeds = range(args.start_seed, args.start_seed + args.seeds)
    total = len(SPEEDS) * len(seeds)
    done = 0

    all_fields = RESULT_FIELDS + [
        "wall_time", "hit", "seed", "speed",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()

        for speed in SPEEDS:
            for seed in seeds:
                done += 1
                r = run_one(speed, seed)

                hit = r.get("hit_type", "?")
                pos = r.get("pos_error", "?")
                print(
                    f"[{done}/{total}] speed={speed} seed={seed}: "
                    f"hit={hit} pos_err={pos}",
                    flush=True,
                )

                writer.writerow(r)
                f.flush()

    print(f"\nCSV 已保存: {csv_path}")

    # 汇总统计
    summary_path = OUTPUT_DIR / "summary.txt"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open(summary_path, "w", encoding="utf-8") as s:
        s.write("exp15: V12 多球速鲁棒性验证\n")
        s.write(f"Seeds: {args.seeds}/speed\n")
        s.write("=" * 60 + "\n\n")
        s.write(f"{'Speed':>6} {'Hit%':>6} {'pos_err':>10} {'v_racket':>10} {'wall_s':>8} {'n':>4}\n")
        s.write("-" * 60 + "\n")

        for speed in SPEEDS:
            group = [r for r in rows
                     if int(r["speed"]) == speed and "error" not in r]
            n = len(group)
            if n == 0:
                s.write(f"{speed:>5}m/s   无数据\n")
                continue

            hits = sum(1 for r in group if r.get("hit") == "True")
            rate = hits / n * 100
            pos_errs = [float(r["pos_error"]) for r in group if r.get("pos_error")]
            v_rackets = [float(r["v_racket_at_hit"]) for r in group if r.get("v_racket_at_hit")]
            walls = [float(r["wall_time"]) for r in group if r.get("wall_time")]
            pos_m = sum(pos_errs) / len(pos_errs) if pos_errs else 0
            v_m = sum(v_rackets) / len(v_rackets) if v_rackets else 0
            wall_m = sum(walls) / len(walls) if walls else 0

            s.write(
                f"{speed:>5}m/s {rate:>5.1f}% {pos_m:>9.4f}m "
                f"{v_m:>8.2f}m/s {wall_m:>7.1f}s {n:>4}\n"
            )

    print(f"汇总已保存: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as s:
        print(s.read())


if __name__ == "__main__":
    main()
