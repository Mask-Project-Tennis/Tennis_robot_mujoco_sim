#!/usr/bin/env python3
"""提取 exp17i_limits_ablation 结果：限速层 × 消融档聚合 + 增益放大检验。

设计见 `docs/experiments/design/exp17i_limits_ablation.md`（配对设计）：
- TCP 1.0（real_robot.yaml）vs TCP 1.8（默认）× 4 档 × 400 seeds @ 7 m/s；
- 核心问题：限速越紧，tube/softmin 相对 none 的增益是否放大
  （Q5 相对版主张「鲁棒性预算与速度余量存在替代关系」）。

口径与 exp17g/17h 一致：剔除 error 非空行；同 seed 跨限速层 + 跨档对齐。

用法:
    python scripts/extract/extract_exp17i_results.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT / "experiment_data" / "exp17i_limits_ablation"
CSV_PATH = DATA_DIR / "results.csv"
SUMMARY_PATH = DATA_DIR / "extract_summary.json"

MODES = ("full", "tube_only", "softmin_only", "none")
LIMIT_LABELS = {True: "TCP1.0", False: "TCP1.8"}


def load_valid_rows(csv_path: Path) -> list[dict[str, str]]:
    """读取 results.csv 并剔除 error 非空行。"""
    with open(csv_path, "r", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if not (r.get("error") or "").strip()]


def row_key(row: dict[str, str]) -> tuple[str, str]:
    """解析 (限速层, 消融档)。有 --limits-config 即 TCP 1.0。"""
    cfg = json.loads(row["config_json"])
    return (LIMIT_LABELS["--limits-config" in cfg], str(cfg.get("--ablation", "")))


def rate_ci(hits: int, n: int) -> float:
    """命中率 95% 置信半宽（pp）。"""
    if n == 0:
        return 0.0
    p = hits / n
    return 1.96 * sqrt(p * (1 - p) / n) * 100


def prop_se(hits: int, n: int) -> float:
    """比例标准误。"""
    p = hits / n
    return sqrt(p * (1 - p) / n)


def two_sample_z(h1: int, n1: int, h2: int, n2: int) -> tuple[float, float, float]:
    """两样本比例 z 检验（双侧），返回 (diff_pp, z, p)。"""
    d, se = _diff_and_se(h1, n1, h2, n2)
    if se == 0:
        return d * 100, 0.0, 1.0
    z = d / se
    return d * 100, z, 2 * (1 - NormalDist().cdf(abs(z)))


def _diff_and_se(h1: int, n1: int, h2: int, n2: int) -> tuple[float, float]:
    """两独立比例之差及其标准误（合并口径，用于同层内对比）。"""
    p1, p2 = h1 / n1, h2 / n2
    p = (h1 + h2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return p1 - p2, se


def main() -> None:
    """聚合限速层 × 档位命中率、层内 z 检验、跨层增益放大 DiD 检验。"""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"缺少 {CSV_PATH}（实验未开始或未完成）")
    rows = load_valid_rows(CSV_PATH)
    raw_total = sum(1 for _ in csv.DictReader(open(CSV_PATH, encoding="utf-8")))

    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        counts[row_key(r)][1] += 1
        if r["hit"].strip().lower() in ("true", "1"):
            counts[row_key(r)][0] += 1

    print(f"exp17i_limits_ablation: 总行 {raw_total} | 有效 {len(rows)} | "
          f"error {raw_total - len(rows)}\n")

    summary: dict[str, Any] = {"total_rows": raw_total, "valid_rows": len(rows),
                               "cells": {}, "within_limit_tests": {},
                               "gain_amplification": {}}

    # ---- 层内聚合与检验 ----
    for limit in ("TCP1.0", "TCP1.8"):
        print(f"=== {limit}（7 m/s） ===")
        line = ""
        for mode in MODES:
            h, n = counts.get((limit, mode), [0, 0])
            if n:
                line += f"{mode}={100 * h / n:.2f}%±{rate_ci(h, n):.2f}(n={n}) "
                summary["cells"][f"{limit}|{mode}"] = {
                    "hits": h, "n": n, "rate": 100 * h / n}
        print(line)
        for a, b in (("tube_only", "none"), ("softmin_only", "none"),
                     ("full", "none"), ("full", "softmin_only")):
            ha, na = counts.get((limit, a), [0, 0])
            hb, nb = counts.get((limit, b), [0, 0])
            if na and nb:
                d, z, p = two_sample_z(ha, na, hb, nb)
                tag = "显著" if p < 0.05 else "ns"
                print(f"  {a} vs {b}: {d:+.2f}pp, z={z:+.2f}, p={p:.4f} [{tag}]")
                summary["within_limit_tests"][f"{limit}|{a}_vs_{b}"] = {
                    "diff_pp": d, "z": z, "p": p}
        print()

    # ---- 跨层增益放大检验（DiD：gain@1.0 vs gain@1.8）----
    print("=== 增益放大检验（Q5 相对版主张）===")
    for a in ("tube_only", "softmin_only", "full"):
        ha1, n1a = counts.get(("TCP1.0", a), [0, 0])
        hb1, n1b = counts.get(("TCP1.0", "none"), [0, 0])
        ha0, n0a = counts.get(("TCP1.8", a), [0, 0])
        hb0, n0b = counts.get(("TCP1.8", "none"), [0, 0])
        if not (n1a and n1b and n0a and n0b):
            continue
        g1 = ha1 / n1a - hb1 / n1b
        g0 = ha0 / n0a - hb0 / n0b
        # DiD 标准误：各层增益的两独立样本标准误之和（不合并池，保守口径）
        se1 = sqrt(prop_se(ha1, n1a) ** 2 + prop_se(hb1, n1b) ** 2)
        se0 = sqrt(prop_se(ha0, n0a) ** 2 + prop_se(hb0, n0b) ** 2)
        se_did = sqrt(se1 ** 2 + se0 ** 2)
        did = (g1 - g0) * 100
        z = did / (se_did * 100) if se_did > 0 else 0.0
        p = 2 * (1 - NormalDist().cdf(abs(z)))
        tag = "放大显著" if (p < 0.05 and did > 0) else ("ns" if p >= 0.05 else "反向")
        print(f"  {a} 增益: TCP1.8 {g0 * 100:+.2f}pp → TCP1.0 {g1 * 100:+.2f}pp | "
              f"DiD={did:+.2f}pp, z={z:+.2f}, p={p:.4f} [{tag}]")
        summary["gain_amplification"][a] = {
            "gain_tcp18_pp": g0 * 100, "gain_tcp10_pp": g1 * 100,
            "did_pp": did, "z": z, "p": p}

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {SUMMARY_PATH.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
