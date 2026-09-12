#!/usr/bin/env python3
"""run_exemption_tcp —— chat2 审稿补跑：豁免窗口内 TCP 速度记录与分析。

背景（P0-4 修复证据）：论文原称 "strict TCP cap"，但安全滤波对击球前
terminal_exempt_steps=20 步（100 ms）跳过 TCP 检查。CSV 只有全局 max_tcp，
无法区分「豁免窗口内」与「窗口外」。本脚本用 --dump-trajectory 重录
4 档 × TCP1.0（real_robot.yaml）× 50 seeds 的逐步轨迹（tcp_pos 已记录），
离线计算：

- 窗口外（接触前 100 ms 之前）max TCP：验证滤波确实守住 1.0 m/s
- 豁免窗口内（接触前最后 20 步）max TCP：审稿人要求的「豁免区间内最大 TCP 速度」
- 全局 max TCP（与 CSV 口径一致，用于交叉校验）

输出：experiment_data/exp18_tcp_exempt/raw/<tier>_<seed>.npz + window_stats.csv

用法:
    python scripts/exp/run_exemption_tcp.py            # 全量（4×50=200 runs）
    python scripts/exp/run_exemption_tcp.py --seeds 5 --workers 8   # 冒烟
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

V12 = PROJECT / "scripts" / "rm65_mpc_v12.py"
OUT = PROJECT / "experiment_data" / "exp18_tcp_exempt"
RAW = OUT / "raw"

RESULT_RE = re.compile(r"__RESULT__: (.+)")
TIERS = ("full", "tube_only", "softmin_only", "none")
EXEMPT_STEPS = 20  # 与 robot_limits.py terminal_exempt_steps 一致


def run_one(tier: str, seed: int) -> dict[str, object]:
    """运行单 episode 并落盘轨迹 npz。"""
    dump = RAW / f"{tier}_seed{seed:03d}.npz"
    dump.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(V12),
        "--serve-box", "--ball-speed", "7", "--no-plot",
        "--ablation", tier,
        "--limits-config", "configs/real_robot.yaml",
        "--seed", str(seed),
        "--dump-trajectory", str(dump),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=240,
                            cwd=str(PROJECT), encoding="utf-8",
                            errors="replace")
    text = result.stdout + result.stderr
    hit_type = "error"
    for line in text.splitlines():
        m = RESULT_RE.search(line)
        if m:
            hit_type = m.group(1)
    return {"tier": tier, "seed": seed, "npz": str(dump), "result": hit_type}


def _parse_result_kv(text: str) -> dict[str, str]:
    """解析 __RESULT__ 行（复用 batch_runner 格式）。"""
    kv: dict[str, str] = {}
    for token in text.split():
        if "=" in token:
            k, v = token.split("=", 1)
            kv[k] = v
    return kv


def analyze() -> list[dict[str, float]]:
    """离线分析：逐 npz 计算窗口内/外 max TCP。"""
    rows: list[dict[str, float]] = []
    for npz_path in sorted(RAW.glob("*.npz")):
        d = np.load(npz_path, allow_pickle=True)
        tcp = d["tcp_pos"]
        dt = float(d["dt"])
        hit_step = int(d["hit_step"])
        speed = np.linalg.norm(np.gradient(tcp, axis=0), axis=1) / dt
        max_all = float(speed.max())
        # tier 名含下划线（tube_only/softmin_only），以 "_seed" 为界切分
        stem = npz_path.stem
        row: dict[str, float] = {
            "file": stem,
            "tier": stem.rsplit("_seed", 1)[0],
            "seed": int(stem.rsplit("seed", 1)[1]),
            "hit_step": hit_step,
            "max_tcp_all": max_all,
        }
        if hit_step > EXEMPT_STEPS:
            row["max_tcp_out_window"] = float(speed[:hit_step - EXEMPT_STEPS + 1].max())
            row["max_tcp_in_window"] = float(
                speed[max(0, hit_step - EXEMPT_STEPS):hit_step].max())
        else:
            row["max_tcp_out_window"] = float("nan")
            row["max_tcp_in_window"] = float("nan")
        rows.append(row)
    return rows


def main() -> None:
    """主流程：并行录制 + 离线分析 + 分档汇总。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = [(tier, seed) for tier in TIERS for seed in range(1, args.seeds + 1)]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, t, s): (t, s) for t, s in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} done", flush=True)

    rows = analyze()
    with open(OUT / "window_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 分档汇总（仅统计有击球的 episode：窗口才有定义）
    print("\n=== 豁免窗口 TCP 分档汇总（命中 episode） ===")
    from collections import defaultdict
    agg: dict[str, list[float]] = defaultdict(list)
    agg_all: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["hit_step"] > EXEMPT_STEPS:
            agg[r["tier"]].append(r["max_tcp_in_window"])
            agg_all[r["tier"]].append(r["max_tcp_all"])
    for tier in TIERS:
        if tier in agg:
            w = np.array(agg[tier])
            a = np.array(agg_all[tier])
            print(f"  {tier:12s} n={len(w):3d}  "
                  f"窗口内 max: mean={w.mean():.2f} p90={np.percentile(w, 90):.2f} "
                  f"max={w.max():.2f}  >1.0 占比 {(w > 1.0).mean() * 100:.0f}%  | "
                  f"全局 max: mean={a.mean():.2f} max={a.max():.2f}")
        else:
            print(f"  {tier:12s} （无命中 episode，无窗口统计）")


if __name__ == "__main__":
    main()
