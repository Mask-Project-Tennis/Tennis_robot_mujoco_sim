#!/usr/bin/env python3
"""exp16: V12 真机限位 vs 仿真限位命中率对比。

验证默认真机限位（TCP 1.0 m/s, J6 ±180°）是否导致命中率显著回退。
两组配对对比（同 seeds），消除随机性差异。

用法:
    python scripts/exp/run_exp16_limits_comparison.py [--seeds 50] [--start-seed 1]

输出:
    experiment_data/exp16_v12_limits_comparison/results.csv
    experiment_data/exp16_v12_limits_comparison/summary.txt

参数矩阵:
    limits_mode ∈ [real, sim] × 50 seeds = 100 runs
    ball_speed = 7 m/s（固定）
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
OUTPUT_DIR = PROJECT / "experiment_data" / "exp16_v12_limits_comparison"
SCRIPT = PROJECT / "scripts" / "rm65_mpc_v12.py"

RESULT_FIELDS = [
    "pos_error", "vel_error", "min_dist", "ball_near_ms", "tube_ready_ms",
    "max_tcp", "max_qdot", "max_face", "hit_type", "hit_time_error_ms",
    "hit_pos_error", "v_racket_at_hit",
]
RESULT_RE = re.compile(r"__RESULT__: (.+)")

LIMITS_MODES = ["real", "sim"]
SPEED = 7


def run_one(limits_mode: str, seed: int) -> dict:
    """运行一次 V12 力矩模式仿真。

    Args:
        limits_mode: 限位模式（"real" 默认真机 / "sim" 仿真宽松）。
        seed: 随机种子。

    Returns:
        指标字典，失败时含 error 键。
    """
    cmd = [
        sys.executable, str(SCRIPT),
        "--serve-box", "--ball-speed", str(SPEED),
        "--seed", str(seed), "--no-plot",
    ]
    if limits_mode == "sim":
        cmd.append("--sim-limits")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT), encoding="utf-8",
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "seed": seed, "limits_mode": limits_mode}
    wall = time.perf_counter() - t0

    stdout = result.stdout + result.stderr
    m = RESULT_RE.search(stdout)
    if not m:
        return {"error": "no_result", "seed": seed, "limits_mode": limits_mode,
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
    fields["limits_mode"] = limits_mode
    return fields


def main():
    import argparse

    parser = argparse.ArgumentParser(description="exp16: V12 真机限位 vs 仿真限位对比")
    parser.add_argument("--seeds", type=int, default=50, help="每组的 seed 数")
    parser.add_argument("--start-seed", type=int, default=1, help="起始 seed")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "results.csv"

    seeds = range(args.start_seed, args.start_seed + args.seeds)
    total = len(LIMITS_MODES) * len(seeds)
    done = 0

    all_fields = RESULT_FIELDS + [
        "wall_time", "hit", "seed", "limits_mode",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()

        for limits_mode in LIMITS_MODES:
            for seed in seeds:
                done += 1
                r = run_one(limits_mode, seed)

                hit = r.get("hit_type", "?")
                pos = r.get("pos_error", "?")
                print(
                    f"[{done}/{total}] mode={limits_mode} seed={seed}: "
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

    stats: dict[str, dict] = {}

    for mode in LIMITS_MODES:
        group = [r for r in rows
                 if r.get("limits_mode") == mode and "error" not in r]
        n = len(group)
        if n == 0:
            stats[mode] = {"n": 0}
            continue

        hits = sum(1 for r in group if r.get("hit") == "True")
        rate = hits / n * 100
        pos_errs = [float(r["pos_error"]) for r in group if r.get("pos_error")]
        max_tcps = [float(r["max_tcp"]) for r in group if r.get("max_tcp")]
        v_rackets = [float(r["v_racket_at_hit"]) for r in group if r.get("v_racket_at_hit")]
        walls = [float(r["wall_time"]) for r in group if r.get("wall_time")]
        stats[mode] = {
            "n": n,
            "hits": hits,
            "rate": rate,
            "pos_mean": sum(pos_errs) / len(pos_errs) if pos_errs else 0,
            "tcp_mean": sum(max_tcps) / len(max_tcps) if max_tcps else 0,
            "tcp_max": max(max_tcps) if max_tcps else 0,
            "v_mean": sum(v_rackets) / len(v_rackets) if v_rackets else 0,
            "wall_mean": sum(walls) / len(walls) if walls else 0,
        }

    with open(summary_path, "w", encoding="utf-8") as s:
        s.write("exp16: V12 真机限位 vs 仿真限位命中率对比\n")
        s.write(f"Seeds: {args.seeds}/group, Speed: {SPEED} m/s\n")
        s.write("=" * 70 + "\n\n")
        s.write(f"{'Mode':>6} {'Hit%':>6} {'pos_err':>10} {'max_tcp':>10} {'tcp_peak':>10} {'v_racket':>10} {'wall_s':>8} {'n':>4}\n")
        s.write("-" * 70 + "\n")

        for mode in LIMITS_MODES:
            st = stats[mode]
            if st["n"] == 0:
                s.write(f"{mode:>6}   无数据\n")
                continue
            s.write(
                f"{mode:>6} {st['rate']:>5.1f}% {st['pos_mean']:>9.4f}m "
                f"{st['tcp_mean']:>8.2f}m/s {st['tcp_max']:>8.2f}m/s "
                f"{st['v_mean']:>8.2f}m/s {st['wall_mean']:>7.1f}s {st['n']:>4}\n"
            )

        # Δ 命中率差
        if all(stats[m].get("n", 0) > 0 for m in LIMITS_MODES):
            delta = stats["real"]["rate"] - stats["sim"]["rate"]
            s.write("-" * 70 + "\n")
            s.write(f"{'Δ':>6} {delta:>+5.1f}pp\n")
            s.write("\n决策依据:\n")
            if abs(delta) < 5:
                s.write("  Δ < 5pp → 真机限位安全，可使用默认\n")
            elif abs(delta) < 10:
                s.write("  5pp ≤ Δ < 10pp → 可接受，建议监控\n")
            else:
                s.write("  Δ ≥ 10pp → 需调整 terminal_exempt_steps 或默认限位\n")

    print(f"汇总已保存: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as s:
        print(s.read())


if __name__ == "__main__":
    main()
