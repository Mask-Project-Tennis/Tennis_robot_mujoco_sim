#!/usr/bin/env python3
"""提取 exp17h_extreme 结果：分块聚合命中率 + 两样本比例 z 检验。

设计见 `docs/experiments/design/exp17h_extreme.md`：
- Block A（9 m/s，t=0，s∈{0.3,0.4}）：走廊独立增益随扰动幅度是否单调；
- Block B（12 m/s，t=0，s=0.2）：紧时间余量下走廊独立增益是否放大。

口径与 exp17g 分析一致：剔除 error 非空行；同 seed 四档对齐。

用法:
    python scripts/extract/extract_exp17h_results.py
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
DATA_DIR = PROJECT / "experiment_data" / "exp17h_extreme"
CSV_PATH = DATA_DIR / "results.csv"
SUMMARY_PATH = DATA_DIR / "extract_summary.json"

MODES = ("full", "tube_only", "softmin_only", "none")


def load_valid_rows(csv_path: Path) -> list[dict[str, str]]:
    """读取 results.csv 并剔除 error 非空行。

    Args:
        csv_path: batch_runner 产出的结果 CSV。

    Returns:
        有效行列表；文件不存在时抛 FileNotFoundError。
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if not (r.get("error") or "").strip()]


def cell_key(row: dict[str, str]) -> tuple[int, float, str]:
    """从一行结果解析 (球速, 空间扰动上限, 消融档)。"""
    cfg = json.loads(row["config_json"])
    return (
        int(cfg.get("--ball-speed", 0)),
        float(cfg.get("--space-perturb-m", 0.0) or 0.0),
        str(cfg.get("--ablation", "")),
    )


def z_test(h1: int, n1: int, h2: int, n2: int) -> tuple[float, float, float]:
    """两样本比例 z 检验（双侧）。

    Args:
        h1: 组 1 命中数。n1: 组 1 样本数。h2: 组 2 命中数。n2: 组 2 样本数。

    Returns:
        (差异 pp, z 值, p 值)。
    """
    p1, p2 = h1 / n1, h2 / n2
    p = (h1 + h2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 0.0, 1.0
    z = (p1 - p2) / se
    return (p1 - p2) * 100, z, 2 * (1 - NormalDist().cdf(abs(z)))


def rate_ci(hits: int, n: int) -> float:
    """命中率的 95% 置信半宽（pp）。"""
    if n == 0:
        return 0.0
    p = hits / n
    return 1.96 * sqrt(p * (1 - p) / n) * 100


def main() -> None:
    """聚合各块命中率、打印 z 检验表并写 extract_summary.json。"""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"缺少 {CSV_PATH}（实验未开始或未完成）")
    rows = load_valid_rows(CSV_PATH)
    raw_total = sum(1 for _ in csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    errors = raw_total - len(rows)

    counts: dict[tuple[int, float, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        key = cell_key(r)
        counts[key][1] += 1
        if r["hit"].strip().lower() in ("true", "1"):
            counts[key][0] += 1

    print(f"exp17h_extreme: 总行 {raw_total} | 有效 {len(rows)} | error {errors}\n")

    # ---- 分块打印（Block A: 9 m/s s=0.3/0.4；Block B: 12 m/s s=0.2）----
    summary: dict[str, Any] = {"total_rows": raw_total, "valid_rows": len(rows),
                               "errors": errors, "cells": {}, "tests": {}}
    for speed, s_max in sorted({(k[0], k[1]) for k in counts}):
        print(f"=== {speed} m/s, s_max={s_max} m ===")
        line = ""
        for mode in MODES:
            h, n = counts.get((speed, s_max, mode), [0, 0])
            if n:
                line += f"{mode}={100 * h / n:.2f}%±{rate_ci(h, n):.2f}(n={n}) "
                summary["cells"][f"{speed}|{s_max}|{mode}"] = {
                    "hits": h, "n": n, "rate": 100 * h / n}
        print(line)

        # 关键对比：tube_only vs none（走廊独立价值）；full vs softmin_only（覆盖性）
        for a, b in (("tube_only", "none"), ("full", "softmin_only"), ("full", "none"),
                     ("softmin_only", "none")):
            ha, na = counts.get((speed, s_max, a), [0, 0])
            hb, nb = counts.get((speed, s_max, b), [0, 0])
            if na and nb:
                d, z, p = z_test(ha, na, hb, nb)
                tag = "显著" if p < 0.05 else "ns"
                print(f"  {a} vs {b}: {d:+.2f}pp, z={z:+.2f}, p={p:.4f} [{tag}]")
                summary["tests"][f"{speed}|{s_max}|{a}_vs_{b}"] = {
                    "diff_pp": d, "z": z, "p": p}
        print()

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {SUMMARY_PATH.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
