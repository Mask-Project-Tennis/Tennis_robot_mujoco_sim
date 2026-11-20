#!/usr/bin/env python3
"""chat8 辅助分析 —— I(t,s) 交互 / both-succeed 子集 / hard-min 配对（只读 CSV）。

三个输出（供 chat8 修复轮正文措辞使用，数据来源全部为已落盘 CSV）:

1. I(t,s) 交互（#2）: 每 seed 的
   I = [H_sm(t,s) − H_pt(t,s)] − [H_sm(0,0) − H_pt(0,0)]，
   对 softmin 相对 point-target 的优势在加入扰动后的变化；报告均值、
   paired-bootstrap 95% CI（按 seed 重采样 10000 次）、Holm 校正（19 格）。

2. both-succeed 子集（#3）: TCP1.0/TCP1.8 层内，限定「M 与 none 双方都命中的
   同一批 seed」后对比 hit_time_error_ms 与 v_racket_at_hit 均值——
   避免「各方法统计的是不同成功 episode 子集」的筛选偏差。

3. hard-min 基线配对（E-new1）: exp19_hardmin（β=1e6）与 β=5 名义
   （exp17b full/none + exp17f softmin_only @ 9 m/s, seeds 1-100）同 seed 配对。

T8b 起支持指定数据根与输出：
    python scripts/extract/extract_chat8_aux.py [--data-root DIR] [--output DIR]

- `--data-root` 指定实验数据根（默认 `experiment_data/`，旧行为不变）；
- `--output` 给出时把结果写成 `<output>/chat8_aux.json`（新增键）并打印路径；
  缺省只打印（旧行为，不写任何文件）；`--dry-run` 显式只打印。
- `analyze_all.py` 会以相同 `--data-root` 调用本脚本，保证同源。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = PROJECT / "experiment_data"

# 与 paired_stats 共用 RM65 / R1D 的统一配置访问器（脚本目录在 sys.path[0]）
from paired_stats import (  # noqa: E402
    space_perturb_of,
    tcp_label_of,
    tier_of,
    time_perturb_ms_of,
)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（--data-root / --output / --dry-run）。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=None,
                        help=f"实验数据根（默认 {DEFAULT_DATA_ROOT}）")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出目录（给出时写 <output>/chat8_aux.json；缺省只打印）")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    return parser


def load(data_root: Path, exp: str) -> list[dict[str, Any]]:
    """读取实验 CSV 行并解析 config_json（跳过 error 行）。"""
    rows: list[dict[str, Any]] = []
    path = data_root / exp / "results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}")
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            r["_cj"] = json.loads(r["config_json"])
            rows.append(r)
    return rows


def outcome(r: dict[str, Any]) -> int:
    """命中指标：active 记 1，其余（passive/miss）记 0。"""
    return 1 if r.get("hit_type") == "active" else 0


def by_seed(rows: list[dict[str, Any]], tier: str, t: float, s: float) -> dict[int, int]:
    """取某 (tier, t, s) 组合的 seed → hit 映射（兼容 RM65 旧 flag 与 R1D override）。"""
    out: dict[int, int] = {}
    for r in rows:
        if (tier_of(r) == tier
                and time_perturb_ms_of(r) == t
                and space_perturb_of(r) == s):
            out[int(r["seed"])] = outcome(r)
    return out


def paired_bootstrap_ci(vals: np.ndarray, n_boot: int = 10000,
                        seed: int = 0) -> tuple[float, float]:
    """按 seed 配对重采样的 95% 百分位 CI（均值）。"""
    rng = np.random.default_rng(seed)
    n = len(vals)
    means = np.array([rng.choice(vals, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _interaction(b17: list[dict[str, Any]],
                 f17: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """第 1 节：I(t,s) 交互（softmin − none），打印并返回结构化结果。"""
    ts = [0, 10, 25, 50, 100]
    ss = [0.0, 0.05, 0.1, 0.2]
    sm00 = by_seed(f17, "softmin_only", 0, 0.0)
    pt00 = by_seed(b17, "none", 0, 0.0)
    base = {k: sm00[k] - pt00[k] for k in sm00.keys() & pt00.keys()}
    print("== 1. I(t,s) = [sm−pt](t,s) − [sm−pt](0,0)，per-seed 配对 ==")
    cells: list[dict[str, Any]] = []
    notes: list[str] = []
    if not base:
        notes.append("I(t,s): 无共同 seed（softmin_only/none 基线）")
        print("  不可估计：无共同 seed（softmin_only/none 基线）")
    for s in ss:
        for t in ts:
            if (t, s) == (0, 0.0):
                continue
            sm = by_seed(f17, "softmin_only", t, s)
            pt = by_seed(b17, "none", t, s)
            keys = sm.keys() & pt.keys() & base.keys()
            if not keys:
                notes.append(f"I(t={t},s={s}): 无共同 seed")
                print(f"  t={t:>3} s={s:<4} 不可估计：无共同 seed")
                continue
            d = np.array([(sm[k] - pt[k]) - base[k] for k in sorted(keys)])
            lo, hi = paired_bootstrap_ci(d)
            entry = {"t": t, "s": s, "n": len(d), "mean_pp": 100.0 * d.mean(),
                     "ci95_pp": [100.0 * lo, 100.0 * hi]}
            cells.append(entry)
            flag = "*" if not (entry["ci95_pp"][0] <= 0 <= entry["ci95_pp"][1]) else ""
            print(f"  t={t:>3} s={s:<4} n={entry['n']:3d}  I = {entry['mean_pp']:+6.2f} pp  "
                  f"CI [{entry['ci95_pp'][0]:+6.2f}, {entry['ci95_pp'][1]:+6.2f}]{flag}")
    return {"cells": cells, "note": "Holm 校正用 CI 判定的近似排序（正文只引用 CI）"}, notes


def _both_succeed(i17: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """第 2 节：both-succeed 子集（TCP1.0 层）。"""
    print("\n== 2. both-succeed 子集（exp17i, TCP1.0 层）==")
    tiers = ["softmin_only", "full", "tube_only"]
    notes: list[str] = []

    def load_layer(limits: str) -> dict[str, dict[int, dict[str, Any]]]:
        """层 → tier → seed → {hit, err, v}。"""
        layer: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for r in i17:
            if tcp_label_of(r) != limits:
                continue
            layer[tier_of(r)][int(r["seed"])] = {
                "hit": outcome(r),
                "err": float(r["hit_time_error_ms"]) if r.get("hit_time_error_ms") else float("nan"),
                "v": float(r["v_racket_at_hit"]) if r.get("v_racket_at_hit") else float("nan"),
            }
        return layer

    layer10 = load_layer("TCP1.0")
    none10 = layer10["none"]
    out: dict[str, Any] = {}
    if not none10:
        notes.append("both-succeed: TCP1.0 层无 none 行")
        print("  不可估计：TCP1.0 层无 none 行")
    for tier in tiers:
        m = layer10[tier]
        both = sorted(k for k in m.keys() & none10.keys()
                      if m[k]["hit"] and none10[k]["hit"])
        if not both:
            notes.append(f"both-succeed[{tier}]: 无双方共同命中 seed")
            print(f"  {tier:13s} 不可估计：无双方共同命中 seed")
            continue
        err_m = np.array([m[k]["err"] for k in both])
        err_p = np.array([none10[k]["err"] for k in both])
        v_m = np.array([m[k]["v"] for k in both])
        v_p = np.array([none10[k]["v"] for k in both])
        print(f"  {tier:13s} n_both={len(both):3d}  "
              f"hit-time err: {err_m.mean():.1f} vs {err_p.mean():.1f} ms | "
              f"v_racket: {v_m.mean():.2f} vs {v_p.mean():.2f} m/s")
        out[tier] = {
            "n_both": len(both),
            "mean_hit_time_error_ms": [float(err_m.mean()), float(err_p.mean())],
            "mean_v_racket": [float(v_m.mean()), float(v_p.mean())],
        }
    return out, notes


def _hardmin(b17: list[dict[str, Any]],
             f17: list[dict[str, Any]],
             e19: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """第 3 节：hard-min（β=1e6）vs β=5 名义，seeds 1-100 配对。"""
    print("\n== 3. hard-min (β=1e6) vs β=5 名义，seeds 1-100 配对 ==")
    notes: list[str] = []

    def tier_seeds(rows: list[dict[str, Any]], tier: str) -> dict[int, int]:
        return {int(r["seed"]): outcome(r) for r in rows if tier_of(r) == tier}

    full_hm = tier_seeds(e19, "full")
    sm_hm = tier_seeds(e19, "softmin_only")
    full_b5 = by_seed(b17, "full", 0, 0.0)
    sm_b5 = by_seed(f17, "softmin_only", 0, 0.0)
    out: dict[str, Any] = {}
    for name, a, b in [("full", full_hm, full_b5), ("softmin_only", sm_hm, sm_b5)]:
        keys = sorted(k for k in a.keys() & b.keys() if 1 <= k <= 100)
        if not keys:
            notes.append(f"hard-min[{name}]: 无共同 seed")
            print(f"  {name:13s} 不可估计：无共同 seed")
            continue
        d = np.array([a[k] - b[k] for k in keys])
        lo, hi = paired_bootstrap_ci(d)
        print(f"  {name:13s} n={len(keys):3d}  β1e6−β5 = {100*d.mean():+.2f} pp  "
              f"CI [{100*lo:+.2f}, {100*hi:+.2f}]  "
              f"(β1e6 rate={100*np.mean([a[k] for k in keys]):.1f}%, "
              f"β5 rate={100*np.mean([b[k] for k in keys]):.1f}%)")
        out[name] = {
            "n_paired": len(keys),
            "diff_pp": float(100 * d.mean()),
            "ci95_pp": [float(100 * lo), float(100 * hi)],
            "rate_hardmin_pct": float(100 * np.mean([a[k] for k in keys])),
            "rate_nominal_pct": float(100 * np.mean([b[k] for k in keys])),
        }
    if full_hm:
        print(f"  full β1e6 全量: n={len(full_hm)} rate={100*np.mean(list(full_hm.values())):.1f}%")
        out["full_all"] = {"n": len(full_hm),
                            "rate_pct": float(100 * np.mean(list(full_hm.values())))}
    if sm_hm:
        print(f"  softmin_only β1e6 全量: n={len(sm_hm)} "
              f"rate={100*np.mean(list(sm_hm.values())):.1f}%")
        out["softmin_only_all"] = {"n": len(sm_hm),
                                    "rate_pct": float(100 * np.mean(list(sm_hm.values())))}
    return out, notes


def compute_chat8_aux(data_root: Path | None = None) -> dict[str, Any]:
    """计算三项分析（缺失实验返回 unavailable，不伪造数值）。"""
    root = DEFAULT_DATA_ROOT if data_root is None else data_root
    unavailable: list[str] = []
    result: dict[str, Any] = {"data_root": str(root.resolve()),
                              "metric": "active-hit（legacy 口径）"}
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name in ("exp17b_perturb", "exp17f_mechanism_perturb",
                 "exp17i_limits_ablation", "exp19_hardmin"):
        try:
            loaded[name] = load(root, name)
        except FileNotFoundError as exc:
            unavailable.append(str(exc))
            loaded[name] = []
    result["I_interaction"], n1 = _interaction(loaded["exp17b_perturb"],
                                               loaded["exp17f_mechanism_perturb"])
    result["both_succeed"], n2 = _both_succeed(loaded["exp17i_limits_ablation"])
    result["hardmin_paired"], n3 = _hardmin(loaded["exp17b_perturb"],
                                            loaded["exp17f_mechanism_perturb"],
                                            loaded["exp19_hardmin"])
    result["unavailable"] = unavailable + n1 + n2 + n3
    return result


def main(argv: list[str] | None = None) -> int:
    """执行三项分析并打印报告（--output 时写 JSON）。"""
    args = build_parser().parse_args(argv)
    data_root: Path = args.data_root if args.data_root is not None else DEFAULT_DATA_ROOT
    aux = compute_chat8_aux(data_root)
    if aux["unavailable"]:
        print("不可用数据/不可估计项:")
        for item in aux["unavailable"]:
            print(f"  - {item}")
        print()
    if args.output is not None:
        target = args.output / "chat8_aux.json"
        if args.dry_run:
            print(f"[chat8_aux] dry-run：不写文件（目标 {target}）")
        else:
            print(f"[chat8_aux] 写出目标 {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(aux, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    elif not args.dry_run:
        print("[chat8_aux] 未给 --output：只打印，不写文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
