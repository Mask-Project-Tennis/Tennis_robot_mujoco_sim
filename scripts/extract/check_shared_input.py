#!/usr/bin/env python3
"""shared-input 交叉验证：空间偏置「只偏点目标」vs「偏整个球状态」两口径对照。

论文 §V-B「Shared-input check」段的数据来源。两口径的代码分流见
`src/ilqt/replan_core.py`（`MPCConfig.spatial_perturb_target`）：

- ``hitpoint``（正文主口径）：偏置只叠加到预测击球点 `p_hit_new`，候选窗口与
  走廊轴仍由未偏置的球观测重建 → 走 cost 的那几项拿到无偏几何参照。
- ``ballstate``（交叉验证口径，exp22_shared_spatial，51,000 runs）：偏置注入喂给
  规划器的球位置 `ball_pos_goal`，点目标 / 候选集合 / 走廊轴由同一份有偏输入生成；
  时序与可达性仍用当前观测。

本脚本输出的三个量即论文中该段的三个数字：
  1) 两个高功效角点的 corridor-only 独立价值（两口径对照）；
  2) 9 m/s t=0 幅度扫描 s=0.2/0.3/0.4 的口径差；
  3) 20 格合并网格上 softmin-only vs point-target 的时间窗增益（两口径对照）。

用法::

    python scripts/extract/check_shared_input.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts" / "extract"))

from paired_stats import (
    HitPolicy,
    cfg_of,
    is_error,
    mcnemar_exact,
    paired_bootstrap,
)

DATA = PROJECT / "experiment_data"
TS: tuple[int, ...] = (0, 10, 25, 50, 100)
SS: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)
TIERS: tuple[str, ...] = ("full", "tube_only", "softmin_only", "none")

# 主口径（hitpoint）网格数据；s=0 列与角点的主口径来源
GRID_SRC: tuple[str, ...] = ("exp17b_perturb", "exp17f_mechanism_perturb")
CORNERS_SRC: dict[tuple[int, int, float], str] = {
    (9, 0, 0.2): "exp17g_spatial_power",
    (12, 0, 0.2): "exp17h_extreme",
    (9, 50, 0.2): "exp17g_spatial_power",
    (9, 0, 0.3): "exp17h_extreme",
    (9, 0, 0.4): "exp17h_extreme",
}
SHARED_SRC = "exp22_shared_spatial"


def load(name: str) -> list[dict[str, str]]:
    """读取一个实验目录的 results.csv。"""
    with open(DATA / name / "results.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


HIT = HitPolicy()


def cell_hits(
    rows: list[dict[str, str]], speed: int, t: int, s: float, tier: str
) -> dict[str, int]:
    """筛出一个格子的某档命中判定，返回 {seed: hit}。"""
    out: dict[str, int] = {}
    for r in rows:
        if is_error(r):
            continue
        c = cfg_of(r)
        if int(c.get("--ball-speed", 0)) != speed:
            continue
        if int(c.get("--time-perturb-ms", 0) or 0) != t:
            continue
        if abs(float(c.get("--space-perturb-m", 0.0) or 0.0) - s) > 1e-9:
            continue
        if str(c.get("--ablation", "")) != tier:
            continue
        out[str(r["seed"])] = HIT(r)
    return out


def paired(a: dict[str, int], b: dict[str, int]) -> dict[str, Any] | None:
    """同 seed 配对比较 a-b：命中率差（pp）+ 配对 bootstrap CI + McNemar 精确检验。"""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    ha = [a[k] for k in keys]
    hb = [b[k] for k in keys]
    n_a = sum(1 for x, y in zip(ha, hb) if x and not y)
    n_b = sum(1 for x, y in zip(ha, hb) if y and not x)
    lo, hi = paired_bootstrap(ha, hb)
    return {
        "n": len(keys),
        "diff": 100 * (sum(ha) - sum(hb)) / len(keys),
        "ci": (lo, hi),
        "p": mcnemar_exact(n_a, n_b),
        "rate_a": 100 * sum(ha) / len(keys),
        "rate_b": 100 * sum(hb) / len(keys),
    }


def fmt(d: dict[str, Any] | None) -> str:
    """把配对结果格式化成一行。"""
    if d is None:
        return "(无数据)"
    return (
        f"{d['diff']:+.2f} pp CI [{d['ci'][0]:+.1f},{d['ci'][1]:+.1f}] "
        f"p={d['p']:.2g} n={d['n']} ({d['rate_a']:.1f} vs {d['rate_b']:.1f})"
    )


def main() -> None:
    """打印三个核心量的两口径对照。"""
    shared = load(SHARED_SRC)
    grid_rows = [r for src in GRID_SRC for r in load(src)]
    old_rows = {src: load(src) for src in sorted(set(CORNERS_SRC.values()))}

    print(f"{SHARED_SRC}: {len(shared)} 行")

    # 1) 角点 corridor-only 独立价值（tube_only vs none）
    print("\n[1] 角点 corridor-only 独立价值（两口径，同 seed 配对）")
    for (speed, t, s), src in CORNERS_SRC.items():
        for label, rows in (("hitpoint ", old_rows[src]), ("ballstate", shared)):
            a = cell_hits(rows, speed, t, s, "tube_only")
            b = cell_hits(rows, speed, t, s, "none")
            print(f"  {speed:>2} m/s t={t:<3} s={s:.2f} [{label}]: {fmt(paired(a, b))}")

    # 2) 幅度扫描口径差（9 m/s, t=0）
    print("\n[2] 9 m/s t=0 幅度扫描：口径差 Δ = ballstate − hitpoint（tube_only vs none）")
    for s in (0.2, 0.3, 0.4):
        src = CORNERS_SRC[(9, 0, s)]
        d_old = paired(
            cell_hits(old_rows[src], 9, 0, s, "tube_only"),
            cell_hits(old_rows[src], 9, 0, s, "none"),
        )
        d_new = paired(
            cell_hits(shared, 9, 0, s, "tube_only"),
            cell_hits(shared, 9, 0, s, "none"),
        )
        assert d_old and d_new
        print(
            f"  s={s:.2f}: hitpoint {d_old['diff']:+.2f} pp → "
            f"ballstate {d_new['diff']:+.2f} pp  (Δ {d_new['diff'] - d_old['diff']:+.2f} pp, "
            f"p {d_old['p']:.2g} → {d_new['p']:.2g})"
        )

    # 3) 20 格合并网格的时间窗增益（s>0 格换用 ballstate，限定 seed<=100 保证配对）
    print("\n[3] 20 格合并网格 softmin-only vs point-target（时间窗增益）")
    for label, swap in (("hitpoint ", False), ("ballstate", True)):
        pool: dict[tuple[int, float, str, str], int] = {}
        for t in TS:
            for s in SS:
                for tier in TIERS:
                    if s == 0.0 or not swap:
                        src_rows: list[dict[str, str]] = list(grid_rows)
                    else:
                        src_rows = shared
                    hits = cell_hits(src_rows, 9, t, s, tier)
                    if swap and s > 0.0:
                        hits = {k: v for k, v in hits.items() if int(k) <= 100}
                    for seed, h in hits.items():
                        pool[(t, s, tier, seed)] = h
        idx = {(t, s, tier): {} for t in TS for s in SS for tier in TIERS}
        for (t, s, tier, seed), h in pool.items():
            idx[(t, s, tier)][seed] = h

        a: dict[str, int] = {}
        b: dict[str, int] = {}
        for t in TS:
            for s in SS:
                for seed, h in idx[(t, s, "softmin_only")].items():
                    a[f"{t}|{s}|{seed}"] = h
                for seed, h in idx[(t, s, "none")].items():
                    b[f"{t}|{s}|{seed}"] = h
        print(f"  {label}: {fmt(paired(a, b))}")

    # 附带：口径引起的逐档 seed 判定翻转数（说明点目标/走廊几乎不变，softmin 档重排）
    print("\n[附] 角点逐档 seed 判定翻转数（ballstate vs hitpoint）")
    for (speed, t, s), src in list(CORNERS_SRC.items())[:3]:
        flips = []
        for tier in TIERS:
            a = cell_hits(shared, speed, t, s, tier)
            b = cell_hits(old_rows[src], speed, t, s, tier)
            keys = sorted(set(a) & set(b))
            flips.append(f"{tier}={sum(1 for k in keys if a[k] != b[k])}/{len(keys)}")
        print(f"  {speed:>2} m/s t={t:<3} s={s:.2f}: " + " ".join(flips))

    # 论文数字核对（§V-B「Shared-input check」段引用的三个量）
    print(
        "\n论文数字核对：corridor-only 角点 +4.3 / +7.0 pp；幅度扫描 |Δ| ≤ 0.4 pp；"
        "合并网格 +8.9 → +10.2 pp"
    )


if __name__ == "__main__":
    main()
