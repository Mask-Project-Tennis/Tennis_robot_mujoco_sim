#!/usr/bin/env python3
"""overnight_summary —— 连夜重跑结果汇总: 新数据 vs 旧报告对照表。

读取 experiment_data/<exp>/results.csv, 按 config 聚合命中率与误差,
与 docs/experiments/reports/ 旧报告的关键数字对照, 输出 markdown 摘要。

输出:
- experiment_data/_driver_state/summary.md（写盘）
- stdout（同内容）

用法:
    python scripts/exp/overnight_summary.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "experiment_data"
STATE = DATA / "_driver_state"

EXPERIMENTS = [
    "exp13_arch", "exp15_speed_v2", "exp16_limits_v2", "exp14_pd_v2",
    "exp17a_noise", "exp17b_perturb", "exp17c_obsfreq",
]

# 旧报告关键数字（新数据应落在同量级; 大偏差需人工分析）
REPORT_REFS = {
    "exp13_arch": "报告 2026-06-18: V12 力矩 85.7% (n=98), 位置 23.7%",
    "exp15_speed_v2": "报告 2026-06-22: 7→78% 9→90% 12→86% 15→58% (n=50)",
    "exp16_limits_v2": "报告 2026-07-09: real(TCP1.0) 30%, sim(TCP1.8) 78%",
    "exp14_pd_v2": "报告 2026-06-22: 最优 Kp=500 Kr=0.15 → 80.0% (n=30)",
    "exp17a_noise": "新实验（V12 首测）; V11 旧值仅参考: σ0.02 → ~5%",
    "exp17b_perturb": "新实验（V12 首测）; 无旧对照",
    "exp17c_obsfreq": "新实验; V11 旧值仅参考: off 模式 200→10Hz 零退化",
}


def aggregate(csv_path: Path) -> dict[str, dict]:
    """按 config_id 聚合: n / 命中数 / 命中率 / hit 行平均 pos_error。

    Args:
        csv_path: results.csv 路径。

    Returns:
        {config_id: {"n": int, "hits": int, "errors": int,
                     "hit_pos_err_mean": float | None}}
    """
    agg: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "hits": 0, "errors": 0, "pos_errs": []})
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a = agg[row["config_id"]]
            if row.get("error"):
                a["errors"] += 1
                continue
            a["n"] += 1
            if row.get("hit") in ("True", "true", "1"):
                a["hits"] += 1
                try:
                    a["pos_errs"].append(float(row["pos_error"]))
                except (ValueError, TypeError):
                    pass
    out: dict[str, dict] = {}
    for cid, a in agg.items():
        out[cid] = {
            "n": a["n"], "hits": a["hits"], "errors": a["errors"],
            "hit_rate": round(a["hits"] / a["n"] * 100, 1) if a["n"] else 0.0,
            "hit_pos_err_mean": (
                round(sum(a["pos_errs"]) / len(a["pos_errs"]), 4)
                if a["pos_errs"] else None),
        }
    return out


def main() -> None:
    """生成全部实验的汇总 markdown。"""
    STATE.mkdir(parents=True, exist_ok=True)
    import time
    lines = ["# 连夜重跑汇总（新数据 vs 旧报告）",
             f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]

    overview: dict[str, dict] = {}
    for exp in EXPERIMENTS:
        csv_path = DATA / exp / "results.csv"
        lines += [f"## {exp}", f"> 对照: {REPORT_REFS.get(exp, '—')}", ""]
        if not csv_path.exists():
            lines += ["（无 results.csv —— 实验未跑或失败）", ""]
            continue
        agg = aggregate(csv_path)
        total_n = sum(a["n"] for a in agg.values())
        total_h = sum(a["hits"] for a in agg.values())
        total_e = sum(a["errors"] for a in agg.values())
        overview[exp] = {"configs": len(agg), "runs": total_n,
                         "hits": total_h, "errors": total_e,
                         "hit_rate": round(total_h / total_n * 100, 1)
                         if total_n else 0.0}
        lines += ["| config | n | hits | 命中率% | hit误差m | err |",
                  "|---|---|---|---|---|---|"]
        for cid in sorted(agg):
            a = agg[cid]
            pe = a["hit_pos_err_mean"] if a["hit_pos_err_mean"] is not None else "—"
            lines.append(f"| {cid} | {a['n']} | {a['hits']} | "
                         f"{a['hit_rate']} | {pe} | {a['errors']} |")
        lines.append(f"\n**小计**: {total_n} runs, {total_h} hits "
                     f"({overview[exp]['hit_rate']}%), {total_e} errors\n")

    (STATE / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (STATE / "summary.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n==> 已写入 {STATE / 'summary.md'}")


if __name__ == "__main__":
    main()
