#!/usr/bin/env python3
"""exp14: V12 位置模式 PD 增益扫描。

扫描 Kp_base × Kd_ratio 矩阵，寻找 V12 位置模式最优 PD 组合。
解决 exp13 暴露的位置模式 12-24% 命中率问题。

用法:
    python scripts/exp/run_exp14_v12_pd.py [--seeds 30] [--start-seed 1]

输出:
    experiment_data/exp14_v12_pd_scan/results.csv
    experiment_data/exp14_v12_pd_scan/summary.txt

参数矩阵:
    Kp_base ∈ [50, 100, 150, 200, 300, 500, 750, 1000]
    Kd_ratio ∈ [0.05, 0.08, 0.10, 0.15]
    32 组合 × 30 seeds = 960 runs
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
OUTPUT_DIR = PROJECT / "experiment_data" / "exp14_v12_pd_scan"
SCRIPT = PROJECT / "scripts" / "rm65_mpc_v12.py"

RESULT_FIELDS = [
    "pos_error", "vel_error", "min_dist", "ball_near_ms", "tube_ready_ms",
    "max_tcp", "max_qdot", "max_face", "hit_type", "hit_time_error_ms",
    "hit_pos_error", "v_racket_at_hit",
]
RESULT_RE = re.compile(r"__RESULT__: (.+)")

# PD 扫描矩阵
KP_BASES = [50, 100, 150, 200, 300, 500, 750, 1000]
KD_RATIOS = [0.05, 0.08, 0.10, 0.15]


def run_one(kp_base: int, kd_ratio: float, seed: int, speed: int = 7) -> dict:
    """运行一次 V12 位置模式仿真。

    Args:
        kp_base: Kp 基准值（所有 6 关节统一）。
        kd_ratio: Kd/Kp 比率。
        seed: 随机种子。
        speed: 球速 m/s。

    Returns:
        指标字典，失败时含 error 键。
    """
    kd_base = kp_base * kd_ratio
    cmd = [
        sys.executable, str(SCRIPT),
        "--serve-box", "--ball-speed", str(speed),
        "--seed", str(seed), "--no-plot",
        "--position-mode",
        "--kp", str(kp_base),
        "--kd", str(kd_base),
    ]

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT), encoding="utf-8",
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "seed": seed, "kp_base": kp_base, "kd_ratio": kd_ratio}
    wall = time.perf_counter() - t0

    stdout = result.stdout + result.stderr
    m = RESULT_RE.search(stdout)
    if not m:
        return {"error": "no_result", "seed": seed, "kp_base": kp_base,
                "kd_ratio": kd_ratio, "wall_time": round(wall, 2)}

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
    fields["kp_base"] = kp_base
    fields["kd_ratio"] = kd_ratio
    fields["kd_base"] = round(kd_base, 1)
    return fields


def main():
    import argparse

    parser = argparse.ArgumentParser(description="exp14: V12 位置模式 PD 扫描")
    parser.add_argument("--seeds", type=int, default=30, help="每组 Kp×Kd 的 seed 数")
    parser.add_argument("--start-seed", type=int, default=1, help="起始 seed")
    parser.add_argument("--speed", type=int, default=7, help="球速 m/s")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "results.csv"

    combos = [(kp, kr) for kp in KP_BASES for kr in KD_RATIOS]
    seeds = range(args.start_seed, args.start_seed + args.seeds)
    total = len(combos) * len(seeds)
    done = 0

    all_fields = RESULT_FIELDS + [
        "wall_time", "hit", "seed", "kp_base", "kd_ratio", "kd_base",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()

        for kp_base, kd_ratio in combos:
            for seed in seeds:
                done += 1
                r = run_one(kp_base, kd_ratio, seed, args.speed)

                hit = r.get("hit_type", "?")
                pos = r.get("pos_error", "?")
                print(
                    f"[{done}/{total}] kp={kp_base} kr={kd_ratio} seed={seed}: "
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
        s.write("exp14: V12 位置模式 PD 扫描汇总\n")
        s.write(f"Seeds: {args.seeds}/combo, Speed: {args.speed} m/s\n")
        s.write("=" * 70 + "\n\n")
        s.write(f"{'Kp':>6} {'Kr':>6} {'Hit%':>6} {'pos_err':>10} {'wall_s':>8} {'n':>4}\n")
        s.write("-" * 70 + "\n")

        best_combo = None
        best_rate = -1

        for kp_base in KP_BASES:
            for kd_ratio in KD_RATIOS:
                group = [r for r in rows
                         if int(r["kp_base"]) == kp_base
                         and float(r["kd_ratio"]) == kd_ratio
                         and "error" not in r]
                n = len(group)
                if n == 0:
                    s.write(f"{kp_base:>6} {kd_ratio:>6.2f}   无数据\n")
                    continue

                hits = sum(1 for r in group if r.get("hit") == "True")
                rate = hits / n * 100
                pos_errs = [float(r["pos_error"]) for r in group if r.get("pos_error")]
                walls = [float(r["wall_time"]) for r in group if r.get("wall_time")]
                pos_m = sum(pos_errs) / len(pos_errs) if pos_errs else 0
                wall_m = sum(walls) / len(walls) if walls else 0

                s.write(f"{kp_base:>6} {kd_ratio:>6.2f} {rate:>5.1f}% {pos_m:>9.4f}m {wall_m:>7.1f}s {n:>4}\n")

                if rate > best_rate:
                    best_rate = rate
                    best_combo = (kp_base, kd_ratio, hits, n)

        s.write("\n" + "=" * 70 + "\n")
        if best_combo:
            kp, kr, hits, n = best_combo
            s.write(f"最优组合: Kp={kp}, Kd_ratio={kr} → {hits}/{n} = {best_rate:.1f}%\n")

    print(f"汇总已保存: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as s:
        print(s.read())


if __name__ == "__main__":
    main()
