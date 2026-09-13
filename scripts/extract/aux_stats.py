#!/usr/bin/env python3
"""辅助统计：敏感性 sweep（exp18）+ TCP 超限诊断（E2/E8 + 豁免窗口）。

正文中有两组数字不由 paired_stats.py 产出，本脚本把它们纳入统一复现链路：
- 9 m/s 下 β / 窗口半宽 / 走廊半径 的敏感性 sweep（exp18_sensitivity）；
- TCP 1.0 m/s 限速下的超限诊断（exp16_limits_v2 的 E2、exp17i_limits_ablation
  的 E8）与豁免窗复核（exp18_tcp_exempt）。

产物: experiment_data/aux_stats.json（跟踪入库，作为评审对照基准）。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "experiment_data"
OUT = DATA / "aux_stats.json"


def _rows(name: str) -> list[dict[str, str]]:
    """读取实验 results.csv。"""
    with open(DATA / name / "results.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cfg(row: dict[str, str]) -> dict[str, Any]:
    """解析 config_json。"""
    return json.loads(row["config_json"])


def _valid(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """剔除 error 行（口径与 paired_stats 一致）。"""
    return [r for r in rows if not (r.get("error") or "").strip()]


def _rate(rows: list[dict[str, str]]) -> tuple[float, int]:
    """active 命中率（%）与有效样本数。"""
    ok = _valid(rows)
    hits = sum(1 for r in ok if (r.get("hit_type") or "").strip() == "active")
    return 100.0 * hits / len(ok), len(ok)


def _tcp_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    """episode 峰值 TCP 的超限占比 / mean / p90。"""
    x = np.array([float(r["max_tcp"]) for r in _valid(rows) if r.get("max_tcp")])
    return {
        "n": int(x.size),
        "exceed_pct": round(float((x > 1.0).mean() * 100.0), 1),
        "mean": round(float(x.mean()), 3),
        "p90": round(float(np.percentile(x, 90)), 3),
    }


def _sensitivity() -> list[dict[str, Any]]:
    """exp18_sensitivity：按配置分组统计 active 命中率。"""
    groups: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for r in _rows("exp18_sensitivity"):
        c = _cfg(r)
        key = (c.get("--ablation"), c.get("--softmin-beta"),
               c.get("--window-ms"), c.get("--corridor-radius"))
        groups.setdefault(key, []).append(r)

    out: list[dict[str, Any]] = []
    for (tier, beta, win, rad), rs in sorted(groups.items(), key=str):
        rate, n = _rate(rs)
        if beta is not None:
            knob, value = "beta", beta
        elif win is not None:
            knob, value = "window_ms", win
        elif rad is not None:
            knob, value = "radius", rad
        else:
            knob, value = "nominal", None
        out.append({"tier": tier, "knob": knob, "value": value,
                    "rate": round(rate, 1), "n": n})
    return out


def _tcp_exceed() -> dict[str, Any]:
    """E2（exp16）+ E8（exp17i）在 TCP 1.0 层的超限诊断。"""
    e2_rows = [r for r in _rows("exp16_limits_v2") if _cfg(r).get("--limits-config")]
    e8_groups: dict[str, list[dict[str, str]]] = {}
    for r in _rows("exp17i_limits_ablation"):
        c = _cfg(r)
        if c.get("--limits-config"):
            e8_groups.setdefault(str(c.get("--ablation")), []).append(r)
    e8 = [{"tier": t, **_tcp_stats(rs)} for t, rs in sorted(e8_groups.items())]
    return {"E2": _tcp_stats(e2_rows), "E8": e8}


def _exempt_window() -> dict[str, Any]:
    """exp18_tcp_exempt：豁免窗内 per-step 峰值（复核 ≤ 1.0 m/s）。

    miss 行（hit_step=-1）没有豁免窗，对应 in-window 值为 NaN，需剔除。
    """
    with open(DATA / "exp18_tcp_exempt" / "window_stats.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    in_window = np.array([float(r["max_tcp_in_window"]) for r in rows])
    in_window = in_window[~np.isnan(in_window)]
    all_steps = np.array([float(r["max_tcp_all"]) for r in rows])
    return {"n": len(rows),
            "n_in_window": int(in_window.size),
            "max_in_window": round(float(in_window.max()), 4),
            "exceed_all_pct": round(float((all_steps > 1.0).mean() * 100.0), 1)}


def compute_aux() -> dict[str, Any]:
    """计算全部辅助统计。"""
    return {"sensitivity": _sensitivity(),
            "tcp_exceed": _tcp_exceed(),
            "exempt_window": _exempt_window()}


def main() -> None:
    """计算并写出 aux_stats.json，打印摘要。"""
    aux = compute_aux()
    OUT.write_text(json.dumps(aux, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {OUT.relative_to(PROJECT)}\n")
    print("敏感性 sweep（9 m/s, active-hit %）:")
    for r in aux["sensitivity"]:
        print(f"  {r['tier']:12s} {r['knob']:10s} {r['value']!s:7s} "
              f"-> {r['rate']:5.1f}% (n={r['n']})")
    e2 = aux["tcp_exceed"]["E2"]
    print(f"\nE2 TCP1.0 超限: {e2['exceed_pct']}%  mean={e2['mean']}  "
          f"p90={e2['p90']}  (n={e2['n']})")
    for r in aux["tcp_exceed"]["E8"]:
        print(f"  E8 {r['tier']:12s} 超限 {r['exceed_pct']}%  "
              f"p90={r['p90']}  (n={r['n']})")
    ew = aux["exempt_window"]
    print(f"\n豁免窗复核: max_tcp_in_window={ew['max_in_window']} (n={ew['n']})")


if __name__ == "__main__":
    main()
