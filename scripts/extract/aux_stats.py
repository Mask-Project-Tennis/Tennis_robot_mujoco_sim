#!/usr/bin/env python3
"""辅助统计：敏感性 sweep（exp18）+ TCP 超限诊断（E2/E8 + 豁免窗口）。

正文中有两组数字不由 paired_stats.py 产出，本脚本把它们纳入统一复现链路：
- 9 m/s 下 β / 窗口半宽 / 走廊半径 的敏感性 sweep（exp18_sensitivity）；
- TCP 1.0 m/s 限速下的超限诊断（exp16_limits_v2 的 E2、exp17i_limits_ablation
  的 E8）与豁免窗复核（exp18_tcp_exempt）。

产物（默认，保留旧行为）: ``<data_root>/aux_stats.json``；R1D 链路请显式传
``--data-root <run_root>/batch --output <output_dir>``，默认根不被触碰。
缺失实验/文件不再崩溃：对应小节标记 ``available=False`` 并给出 ``unavailable_reason``。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = PROJECT / "experiment_data"


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（--data-root / --output / --dry-run）。"""
    parser = argparse.ArgumentParser(description="辅助统计：敏感性 sweep + TCP 超限诊断 + 豁免窗复核")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"实验数据根（默认 {DEFAULT_DATA_ROOT}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出目录（给出时写 <output>/aux_stats.json；未给则只打印不写盘）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只计算并打印，不写任何文件")
    return parser


def default_output_path(data_root: Path) -> Path:
    """默认输出文件路径：``<data_root>/aux_stats.json``（旧行为）。"""
    return data_root / "aux_stats.json"


def _rows(data_root: Path, name: str) -> list[dict[str, str]]:
    """读取实验 results.csv；缺失返回空列表（由调用方标记 unavailable）。"""
    path = data_root / name / "results.csv"
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
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
    if not ok:
        return float("nan"), 0
    return 100.0 * hits / len(ok), len(ok)


def _tcp_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    """episode 峰值 TCP 的超限占比 / mean / p90。"""
    values = [float(r["max_tcp"]) for r in _valid(rows) if r.get("max_tcp")]
    if not values:
        return {"n": 0, "exceed_pct": None, "mean": None, "p90": None, "available": False}
    x = np.asarray(values, dtype=float)
    return {
        "n": int(x.size),
        "exceed_pct": round(float((x > 1.0).mean() * 100.0), 1),
        "mean": round(float(x.mean()), 3),
        "p90": round(float(np.percentile(x, 90)), 3),
        "available": True,
    }


def _sensitivity(data_root: Path) -> tuple[list[dict[str, Any]], str | None]:
    """exp18_sensitivity：按配置分组统计 active 命中率。"""
    rows = _rows(data_root, "exp18_sensitivity")
    if not rows:
        return [], "missing exp18_sensitivity/results.csv"
    groups: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for r in rows:
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
    return out, None


def _tcp_exceed(data_root: Path) -> dict[str, Any]:
    """E2（exp16）+ E8（exp17i）在 TCP 1.0 层的超限诊断。"""
    e2_rows = [r for r in _rows(data_root, "exp16_limits_v2") if _cfg(r).get("--limits-config")]
    e8_groups: dict[str, list[dict[str, str]]] = {}
    for r in _rows(data_root, "exp17i_limits_ablation"):
        c = _cfg(r)
        if c.get("--limits-config"):
            e8_groups.setdefault(str(c.get("--ablation")), []).append(r)
    e8 = [{"tier": t, **_tcp_stats(rs)} for t, rs in sorted(e8_groups.items())]
    return {"E2": _tcp_stats(e2_rows), "E8": e8}


def _exempt_window(data_root: Path) -> dict[str, Any]:
    """exp18_tcp_exempt：豁免窗内 per-step 峰值（复核 ≤ 1.0 m/s）。

    miss 行（hit_step=-1）没有豁免窗，对应 in-window 值为 NaN，需剔除。
    """
    path = data_root / "exp18_tcp_exempt" / "window_stats.csv"
    if not path.is_file():
        return {
            "n": 0,
            "n_in_window": 0,
            "max_in_window": None,
            "exceed_all_pct": None,
            "available": False,
            "unavailable_reason": "missing exp18_tcp_exempt/window_stats.csv",
        }
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    in_window = np.array([float(r["max_tcp_in_window"]) for r in rows])
    in_window = in_window[~np.isnan(in_window)]
    all_steps = np.array([float(r["max_tcp_all"]) for r in rows])
    return {"n": len(rows),
            "n_in_window": int(in_window.size),
            "max_in_window": round(float(in_window.max()), 4) if in_window.size else None,
            "exceed_all_pct": round(float((all_steps > 1.0).mean() * 100.0), 1),
            "available": True}


def compute_aux(data_root: Path | None = None) -> dict[str, Any]:
    """计算全部辅助统计（保持旧键 sensitivity/tcp_exceed/exempt_window）。"""
    root = DEFAULT_DATA_ROOT if data_root is None else data_root
    sensitivity, sensitivity_missing = _sensitivity(root)
    aux: dict[str, Any] = {
        "sensitivity": sensitivity,
        "tcp_exceed": _tcp_exceed(root),
        "exempt_window": _exempt_window(root),
    }
    unavailable: list[str] = []
    if sensitivity_missing:
        unavailable.append(sensitivity_missing)
    if not aux["tcp_exceed"]["E2"].get("available"):
        unavailable.append("missing exp16_limits_v2 TCP1.0 行")
    if not aux["tcp_exceed"]["E8"]:
        unavailable.append("missing exp17i_limits_ablation TCP1.0 行")
    if not aux["exempt_window"].get("available"):
        unavailable.append(aux["exempt_window"]["unavailable_reason"])
    aux["data_root"] = str(root.resolve())
    aux["unavailable"] = unavailable
    return aux


def main(argv: list[str] | None = None) -> int:
    """计算 aux 统计并打印摘要；仅在显式 --output 时写 aux_stats.json。"""
    args = build_parser().parse_args(argv)
    data_root: Path = args.data_root if args.data_root is not None else DEFAULT_DATA_ROOT
    aux = compute_aux(data_root)
    # 绝不在未显式指定 --output 时写默认根（避免误改 tracked 数据）
    out_path = (args.output / "aux_stats.json") if args.output is not None \
        else default_output_path(data_root)
    should_write = args.output is not None

    if args.dry_run:
        print(f"[aux_stats] dry-run：不写文件（目标 {out_path}）")
    elif should_write:
        print(f"[aux_stats] 写出目标 {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(aux, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print("[aux_stats] 未给 --output：只打印，不写文件")

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
    if aux["unavailable"]:
        print(f"[aux_stats] unavailable: {aux['unavailable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
