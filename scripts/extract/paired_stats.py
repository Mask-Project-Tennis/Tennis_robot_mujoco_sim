#!/usr/bin/env python3
"""论文统计口径统一提取：active-hit 主指标 + same-seed 配对检验。

背景（2026-09-12 审稿意见核实后确立的口径）：
1. 主指标改为 **active-hit rate**：`hit_type == "active"`（击球瞬间球拍面速度 > 0.3 m/s）；
   被动接触（0 < v_r <= 0.3 m/s）单独计数、不进入主指标；
2. 全部实验为 same-seed 配对设计（同 seed 同发球同扰动交给各档），
   因此显著性检验改用 **配对 McNemar 精确检验** + **配对 bootstrap 95% CI**，
   取代此前提取脚本中的两独立样本比例 z 检验（忽略配对，口径偏保守）；
3. 排除行（`error` 非空，均为 `no_result`）逐档计数并报告，验证排除与方法无关；
4. 限速层 × 机制档的交互效应用 **配对 DiD**（每 seed 的增益差）报告。

输出（论文数字唯一事实源）：
    paper/planning/06-stats-active.json
    experiment_data/paper_stats_active.json

用法:
    python scripts/extract/paired_stats.py [--include-passive]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "experiment_data"
OUT_PAPER = PROJECT / "paper" / "planning" / "06-stats-active.json"
OUT_DATA = DATA / "paper_stats_active.json"

MODES = ("full", "tube_only", "softmin_only", "none")
REFERENCE = "none"  # 点目标基线（消融档名 none）


# ----------------------------------------------------------------------------
# 基础统计工具
# ----------------------------------------------------------------------------

def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% 置信区间（返回百分比 pp）。"""
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


def mcnemar_exact(b: int, c: int) -> float:
    """配对 McNemar 精确检验（双侧，二项分布）。

    Args:
        b: A 命中且 B 未命中的配对数。
        c: A 未命中且 B 命中的配对数。

    Returns:
        双侧 p 值。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(2 * tail, 1.0)


def paired_bootstrap(
    hits_a: list[int], hits_b: list[int], n_boot: int = 20000, seed: int = 0
) -> tuple[float, float]:
    """配对 bootstrap：对 seed 重采样，返回命中率差的 95% 分位区间（pp）。"""
    rng = random.Random(seed)
    n = len(hits_a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        pa = sum(hits_a[i] for i in idx)
        pb = sum(hits_b[i] for i in idx)
        diffs.append((pa - pb) / n * 100)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def sign_test(dids: list[int]) -> float:
    """非零配对差上的符号检验（DiD 用）。"""
    nz = [d for d in dids if d != 0]
    if not nz:
        return 1.0
    pos = sum(1 for d in nz if d > 0)
    k = min(pos, len(nz) - pos)
    tail = sum(math.comb(len(nz), i) for i in range(k + 1)) / 2 ** len(nz)
    return min(2 * tail, 1.0)


# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------

def load(name: str) -> list[dict[str, str]]:
    """读取实验 results.csv（rows 含 error 行，便于统计排除数）。"""
    path = DATA / name / "results.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cfg_of(row: dict[str, str]) -> dict[str, Any]:
    """解析 config_json。"""
    return json.loads(row["config_json"])


def is_error(row: dict[str, str]) -> bool:
    """该行是否被排除（error 非空，实测均为 no_result）。"""
    return bool((row.get("error") or "").strip())


class HitPolicy:
    """命中口径：默认只计 active，可选把 passive 一并计入。"""

    def __init__(self, include_passive: bool = False) -> None:
        self.include_passive = include_passive

    def __call__(self, row: dict[str, str]) -> int:
        if is_error(row):
            return -1
        ht = (row.get("hit_type") or "").strip()
        if ht == "active":
            return 1
        if ht == "passive":
            return 1 if self.include_passive else 0
        return 0


def cell_stats(rows: list[dict[str, str]], hitfn: HitPolicy) -> dict[str, Any]:
    """统计一格的样本构成与命中率（含 Wilson CI、passive 计数）。"""
    errs: Counter[str] = Counter()
    types: Counter[str] = Counter()
    for r in rows:
        if is_error(r):
            errs[(r.get("error") or "").strip()] += 1
        else:
            types[(r.get("hit_type") or "").strip()] += 1
    n_valid = sum(types.values())
    hits = sum(hitfn(r) for r in rows if not is_error(r))
    lo, hi = wilson(hits, n_valid)
    return {
        "rows": len(rows),
        "n_valid": n_valid,
        "hits": hits,
        "rate": 100 * hits / n_valid if n_valid else float("nan"),
        "ci95": [lo, hi],
        "active": types.get("active", 0),
        "passive": types.get("passive", 0),
        "miss": types.get("miss", 0),
        "passive_share": 100 * types.get("passive", 0) / n_valid if n_valid else 0.0,
        "errors": dict(errs),
    }


def paired_test(
    rows: list[dict[str, str]],
    hitfn: HitPolicy,
    tier_a: str,
    tier_b: str,
    keyfn: Callable[[dict[str, str]], str] = lambda r: r["seed"],
    filterfn: Callable[[dict[str, str]], bool] | None = None,
) -> dict[str, Any]:
    """same-seed 配对比较：McNemar 精确检验 + 配对 bootstrap 95% CI。

    Returns:
        含 n_paired / diff_pp / ci95 / p_mcnemar / 各档率的字典。
    """
    sel = [r for r in rows if (filterfn(r) if filterfn else True) and not is_error(r)]
    da: dict[str, int] = {}
    db: dict[str, int] = {}
    for r in sel:
        tier = str(cfg_of(r).get("--ablation", ""))
        if tier == tier_a:
            da[keyfn(r)] = hitfn(r)
        elif tier == tier_b:
            db[keyfn(r)] = hitfn(r)
    keys = sorted(set(da) & set(db))
    if not keys:
        return {"n_paired": 0}
    ha = [da[k] for k in keys]
    hb = [db[k] for k in keys]
    b = sum(1 for x, y in zip(ha, hb) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(ha, hb) if x == 0 and y == 1)
    lo, hi = paired_bootstrap(ha, hb)
    return {
        "n_paired": len(keys),
        "rate_a": 100 * sum(ha) / len(keys),
        "rate_b": 100 * sum(hb) / len(keys),
        "diff_pp": 100 * (sum(ha) - sum(hb)) / len(keys),
        "ci95": [lo, hi],
        "p_mcnemar": mcnemar_exact(b, c),
        "discordant": {"a_hit_b_miss": b, "a_miss_b_hit": c},
    }


def baseline_cell(row: dict[str, str]) -> bool:
    """exp17a 的「无噪声、无 KF」基线条件（空配置）。"""
    c = cfg_of(row)
    return (
        not c.get("--obs-noise-pos")
        and not c.get("--obs-noise-vel")
        and "--obs-use-kf" not in c
    )


# ----------------------------------------------------------------------------
# 各实验统计
# ----------------------------------------------------------------------------

def e1_speed_sweep(hitfn: HitPolicy) -> dict[str, Any]:
    """E1: 球速扫描（exp15，9 档 × 200 seeds，TCP 1.8 默认限位）。"""
    rows = load("exp15_speed_v2")
    by_speed: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_speed[int(cfg_of(r).get("--ball-speed", 0))].append(r)
    out = {}
    for sp in sorted(by_speed):
        out[str(sp)] = cell_stats(by_speed[sp], hitfn)
    return out


def e2_limit_cost(hitfn: HitPolicy) -> dict[str, Any]:
    """E2: 真机限位代价（exp16，7 m/s × 200 seeds，TCP 1.8 vs 1.0）。"""
    rows = load("exp16_limits_v2")
    by_lim: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        key = "TCP1.0" if "--limits-config" in cfg_of(r) else "TCP1.8"
        by_lim[key].append(r)
    out: dict[str, Any] = {}
    for key, sub in by_lim.items():
        st = cell_stats(sub, hitfn)
        speeds = [
            float(r["v_racket_at_hit"])
            for r in sub
            if not is_error(r) and hitfn(r) == 1 and (r.get("v_racket_at_hit") or "")
        ]
        st["mean_racket_speed_on_hit"] = mean(speeds) if speeds else float("nan")
        out[key] = st
    # 配对比较（同 seed 跨限速层）
    da, db = {}, {}
    for r in rows:
        if is_error(r):
            continue
        (da if "--limits-config" in cfg_of(r) else db)[r["seed"]] = hitfn(r)
    keys = sorted(set(da) & set(db))
    if keys:
        ha = [da[k] for k in keys]
        hb = [db[k] for k in keys]
        out["paired_TCP10_vs_TCP18"] = {
            "n_paired": len(keys),
            "diff_pp": 100 * (sum(ha) - sum(hb)) / len(keys),
            "ci95": list(paired_bootstrap(ha, hb)),
            "p_mcnemar": mcnemar_exact(
                sum(1 for x, y in zip(ha, hb) if x == 1 and y == 0),
                sum(1 for x, y in zip(ha, hb) if x == 0 and y == 1),
            ),
        }
    return out


def e3_nominal(hitfn: HitPolicy) -> dict[str, Any]:
    """E3: 标称四档（exp17a 基线格 full/none + exp17d tube_only/softmin_only）。"""
    rows_a, rows_d = load("exp17a_noise"), load("exp17d_mechanism")
    out: dict[str, Any] = {"cells": {}, "paired": {}}
    for sp in (7, 9, 12):
        for tier, rows in (("full", rows_a), ("none", rows_a),
                           ("tube_only", rows_d), ("softmin_only", rows_d)):
            sub = [r for r in rows
                   if baseline_cell(r) and int(cfg_of(r).get("--ball-speed", 0)) == sp
                   and str(cfg_of(r).get("--ablation", "")) == tier]
            out["cells"][f"{sp}|{tier}"] = cell_stats(sub, hitfn)
    combined = rows_a + rows_d

    def at_speed(sp: int):
        def f(r: dict[str, str]) -> bool:
            if not baseline_cell(r):
                return False
            return int(cfg_of(r).get("--ball-speed", 0)) == sp
        return f

    for sp in (7, 9, 12):
        for a, b in (("full", "none"), ("softmin_only", "none"),
                     ("tube_only", "none"), ("full", "softmin_only"),
                     ("tube_only", "softmin_only")):
            out["paired"][f"{sp}|{a}_vs_{b}"] = paired_test(
                combined, hitfn, a, b, filterfn=at_speed(sp))
    return out


def e4_grid(hitfn: HitPolicy) -> dict[str, Any]:
    """E4: 扰动网格（exp17b+17f，9 m/s，20 格合并，配对键 = (格, seed)）。"""
    rows = load("exp17b_perturb") + load("exp17f_mechanism_perturb")
    TS = {0, 10, 25, 50, 100}
    SS = {0.0, 0.05, 0.1, 0.2}

    def in_grid(r: dict[str, str]) -> bool:
        c = cfg_of(r)
        return (
            int(c.get("--ball-speed", 0)) == 9
            and int(c.get("--time-perturb-ms", 0) or 0) in TS
            and float(c.get("--space-perturb-m", 0.0) or 0.0) in SS
        )

    sub = [r for r in rows if in_grid(r)]
    out: dict[str, Any] = {"merged_20cell": {}, "paired_20cell": {}}
    for tier in MODES:
        out["merged_20cell"][tier] = cell_stats(
            [r for r in sub if str(cfg_of(r).get("--ablation", "")) == tier], hitfn)

    def grid_key(r: dict[str, str]) -> str:
        c = cfg_of(r)
        return (f"{int(c.get('--time-perturb-ms', 0) or 0)}|"
                f"{float(c.get('--space-perturb-m', 0.0) or 0.0)}|{r['seed']}")

    for a, b in (("full", "none"), ("softmin_only", "none"), ("tube_only", "none"),
                 ("full", "softmin_only"), ("full", "tube_only"),
                 ("tube_only", "softmin_only")):
        out["paired_20cell"][f"{a}_vs_{b}"] = paired_test(
            sub, hitfn, a, b, keyfn=grid_key)
    # 每格明细（供热图显著性标注）+ 每格配对检验
    per_cell: dict[str, Any] = {}
    per_cell_paired: dict[str, Any] = {}
    for t in sorted(TS):
        for s in sorted(SS):
            for tier in MODES:
                cellrows = [r for r in sub
                            if int(cfg_of(r).get("--time-perturb-ms", 0) or 0) == t
                            and float(cfg_of(r).get("--space-perturb-m", 0.0) or 0.0) == s
                            and str(cfg_of(r).get("--ablation", "")) == tier]
                per_cell[f"t{t}|s{s}|{tier}"] = cell_stats(cellrows, hitfn)

            def cell_filter(r: dict[str, str], t: int = t, s: float = s) -> bool:
                c = cfg_of(r)
                return (int(c.get("--time-perturb-ms", 0) or 0) == t
                        and float(c.get("--space-perturb-m", 0.0) or 0.0) == s)

            for a, b in (("full", "none"), ("tube_only", "none"),
                         ("softmin_only", "none"), ("full", "softmin_only")):
                per_cell_paired[f"t{t}|s{s}|{a}_vs_{b}"] = paired_test(
                    sub, hitfn, a, b, filterfn=cell_filter)
    out["per_cell"] = per_cell
    out["per_cell_paired"] = per_cell_paired
    return out


def e5_perception(hitfn: HitPolicy) -> dict[str, Any]:
    """E5: 噪声 × KF（exp17a）与观测频率（exp17c）。"""
    rows = load("exp17a_noise")
    out: dict[str, Any] = {"noise_x_kf": {}}
    for sp in (7, 9, 12):
        for pos, vel in sorted({(cfg_of(r).get("--obs-noise-pos"),
                                 cfg_of(r).get("--obs-noise-vel"))
                                for r in rows}, key=str):
            for kf in (False, True):
                sub = [r for r in rows
                       if int(cfg_of(r).get("--ball-speed", 0)) == sp
                       and cfg_of(r).get("--obs-noise-pos") == pos
                       and cfg_of(r).get("--obs-noise-vel") == vel
                       and ((("--obs-use-kf" in cfg_of(r)) == kf))]
                for tier in ("full", "none"):
                    cellrows = [r for r in sub
                                if str(cfg_of(r).get("--ablation", "")) == tier]
                    if cellrows:
                        out["noise_x_kf"][
                            f"{sp}|pos{pos}|vel{vel}|kf{int(kf)}|{tier}"
                        ] = cell_stats(cellrows, hitfn)
    rows_c = load("exp17c_obsfreq")
    by_rate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows_c:
        c = cfg_of(r)
        by_rate[str(c.get("--obs-freq", "?"))].append(r)
    out["obsfreq"] = {k: cell_stats(v, hitfn) for k, v in sorted(by_rate.items())}
    return out


def e7_corners(hitfn: HitPolicy) -> dict[str, Any]:
    """E7: 角点高功率格（exp17g / exp17h，预先设定的 4 个对比）。"""
    rows_g, rows_h = load("exp17g_spatial_power"), load("exp17h_extreme")
    out: dict[str, Any] = {"cells": {}, "paired": {}}
    specs = [("exp17g", rows_g, 9, 0.2, 0), ("exp17g", rows_g, 9, 0.2, 50),
             ("exp17h", rows_h, 12, 0.2, 0), ("exp17h", rows_h, 9, 0.3, 0),
             ("exp17h", rows_h, 9, 0.4, 0)]

    def filt(sp: int, s: float, t: int):
        def f(r: dict[str, str]) -> bool:
            c = cfg_of(r)
            return (int(c.get("--ball-speed", 0)) == sp
                    and float(c.get("--space-perturb-m", 0.0) or 0.0) == s
                    and int(c.get("--time-perturb-ms", 0) or 0) == t)
        return f

    for src, rows, sp, s, t in specs:
        tag = f"{sp}|s{s}|t{t}"
        for tier in MODES:
            out["cells"][f"{tag}|{tier}"] = cell_stats(
                [r for r in rows if filt(sp, s, t)(r)
                 and str(cfg_of(r).get("--ablation", "")) == tier], hitfn)
        out["paired"][f"{tag}|tube_only_vs_none"] = paired_test(
            rows, hitfn, "tube_only", "none", filterfn=filt(sp, s, t))
        out["paired"][f"{tag}|full_vs_softmin_only"] = paired_test(
            rows, hitfn, "full", "softmin_only", filterfn=filt(sp, s, t))
    return out


def e8_limit_mechanism(hitfn: HitPolicy) -> dict[str, Any]:
    """E8: 限速 × 机制配对（exp17i）+ 配对 DiD。"""
    rows = load("exp17i_limits_ablation")
    out: dict[str, Any] = {"cells": {}, "paired": {}, "did": {}, "diagnostics": {}}
    for lim in (False, True):
        label = "TCP1.0" if lim else "TCP1.8"
        for tier in MODES:
            sub = [r for r in rows
                   if ("--limits-config" in cfg_of(r)) == lim
                   and str(cfg_of(r).get("--ablation", "")) == tier]
            out["cells"][f"{label}|{tier}"] = cell_stats(sub, hitfn)
            # 命中行的诊断量：时间误差、击球瞬间拍速
            hitrows = [r for r in sub if not is_error(r) and hitfn(r) == 1]
            errs = [float(r["hit_time_error_ms"]) for r in hitrows
                    if (r.get("hit_time_error_ms") or "")]
            vels = [float(r["v_racket_at_hit"]) for r in hitrows
                    if (r.get("v_racket_at_hit") or "")]
            out["diagnostics"][f"{label}|{tier}"] = {
                "n_hit_rows": len(hitrows),
                "mean_hit_time_error_ms": mean(errs) if errs else float("nan"),
                "mean_v_racket": mean(vels) if vels else float("nan"),
            }
        for a, b in (("softmin_only", "none"), ("full", "none"),
                     ("tube_only", "none"), ("full", "softmin_only")):
            out["paired"][f"{label}|{a}_vs_{b}"] = paired_test(
                rows, hitfn, a, b,
                filterfn=lambda r, lim=lim: ("--limits-config" in cfg_of(r)) == lim)
    # 配对 DiD：每 seed 的 (档 - 基线) 增益在两层之差
    for a in ("softmin_only", "full", "tube_only"):
        d1: dict[str, dict[str, int]] = defaultdict(dict)
        d0: dict[str, dict[str, int]] = defaultdict(dict)
        for r in rows:
            if is_error(r):
                continue
            tier = str(cfg_of(r).get("--ablation", ""))
            if tier not in (a, "none"):
                continue
            (d1 if "--limits-config" in cfg_of(r) else d0)[r["seed"]][tier] = hitfn(r)
        dids = []
        for s in sorted(set(d1) & set(d0)):
            if a in d1[s] and "none" in d1[s] and a in d0[s] and "none" in d0[s]:
                dids.append((d1[s][a] - d1[s]["none"]) - (d0[s][a] - d0[s]["none"]))
        if dids:
            rng = random.Random(42)
            n = len(dids)
            boot = sorted(
                sum(dids[i] for i in [rng.randrange(n) for _ in range(n)]) / n
                for _ in range(20000))
            out["did"][a] = {
                "n_paired": n,
                "gain_TCP18_pp": 100 * (d1 and 0 or 0),  # 占位，下面覆盖
                "mean_did_pp": 100 * mean(dids),
                "ci95": [100 * boot[int(0.025 * 20000)], 100 * boot[int(0.975 * 20000)]],
                "p_sign": sign_test(dids),
            }
            # 两层各自的增益（未配对口径，仅作展示）
        g1 = out["paired"].get(f"TCP1.0|{a}_vs_none", {})
        g0 = out["paired"].get(f"TCP1.8|{a}_vs_none", {})
        if a in out["did"]:
            out["did"][a]["gain_TCP18_pp"] = g0.get("diff_pp", float("nan"))
            out["did"][a]["gain_TCP10_pp"] = g1.get("diff_pp", float("nan"))
    return out


def exclusion_report() -> dict[str, Any]:
    """排除行逐档计数（验证排除与方法无关）。"""
    out: dict[str, Any] = {}
    for name in ("exp15_speed_v2", "exp16_limits_v2", "exp17a_noise",
                 "exp17b_perturb", "exp17c_obsfreq", "exp17d_mechanism",
                 "exp17e_perturb7", "exp17f_mechanism_perturb",
                 "exp17g_spatial_power", "exp17h_extreme",
                 "exp17i_limits_ablation"):
        try:
            rows = load(name)
        except FileNotFoundError:
            continue
        by_tier: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [valid, error]
        causes: Counter[str] = Counter()
        for r in rows:
            tier = str(cfg_of(r).get("--ablation", "-"))
            if is_error(r):
                by_tier[tier][1] += 1
                causes[(r.get("error") or "").strip()] += 1
            else:
                by_tier[tier][0] += 1
        total_err = sum(causes.values())
        out[name] = {
            "rows": len(rows),
            "errors": total_err,
            "error_rate_pct": 100 * total_err / len(rows) if rows else 0.0,
            "causes": dict(causes),
            "per_tier": {k: {"valid": v[0], "error": v[1]} for k, v in by_tier.items()},
        }
    return out


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def main() -> None:
    """跑完全部统计并写出 JSON。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-passive", action="store_true",
                    help="把被动接触也计入命中（复现旧口径，用于对照）")
    args = ap.parse_args()
    hitfn = HitPolicy(include_passive=args.include_passive)
    metric = "hit(active+passive)" if args.include_passive else "active-hit"

    print(f"命中口径: {metric}\n")
    summary: dict[str, Any] = {"metric": metric, "include_passive": args.include_passive}

    summary["E1_speed_sweep"] = e1_speed_sweep(hitfn)
    summary["E2_limit_cost"] = e2_limit_cost(hitfn)
    summary["E3_nominal"] = e3_nominal(hitfn)
    summary["E4_grid"] = e4_grid(hitfn)
    summary["E5_perception"] = e5_perception(hitfn)
    summary["E7_corners"] = e7_corners(hitfn)
    summary["E8_limit_mechanism"] = e8_limit_mechanism(hitfn)
    summary["exclusion"] = exclusion_report()

    # ---- 打印可读报告 ----
    print("=== E1 球速扫描（active-hit %，含 95% CI 与 n）===")
    for sp, st in summary["E1_speed_sweep"].items():
        print(f"  {sp:>2} m/s: {st['rate']:.1f}% [{st['ci95'][0]:.1f},{st['ci95'][1]:.1f}] "
              f"n={st['n_valid']} passive={st['passive']} ({st['passive_share']:.1f}%)")

    print("\n=== E2 限位代价（exp16, 7 m/s）===")
    for key in ("TCP1.8", "TCP1.0"):
        st = summary["E2_limit_cost"].get(key, {})
        if st:
            print(f"  {key}: {st['rate']:.1f}% n={st['n_valid']} "
                  f"v_racket={st['mean_racket_speed_on_hit']:.2f} m/s")
    p = summary["E2_limit_cost"].get("paired_TCP10_vs_TCP18", {})
    if p:
        print(f"  配对差: {p['diff_pp']:+.1f}pp CI={p['ci95']} p={p['p_mcnemar']:.2e}")

    print("\n=== E3 标称四档（active-hit %）===")
    for sp in (7, 9, 12):
        vals = []
        for tier in MODES:
            st = summary["E3_nominal"]["cells"].get(f"{sp}|{tier}", {})
            vals.append(f"{tier}={st['rate']:.1f}%(n={st['n_valid']})")
        print(f"  {sp} m/s: " + " ".join(vals))
        for a, b in (("full", "none"), ("softmin_only", "none"),
                     ("tube_only", "none"), ("full", "softmin_only")):
            t = summary["E3_nominal"]["paired"].get(f"{sp}|{a}_vs_{b}", {})
            if t.get("n_paired"):
                print(f"    {a} vs {b}: {t['diff_pp']:+.2f}pp "
                      f"CI=[{t['ci95'][0]:+.2f},{t['ci95'][1]:+.2f}] p={t['p_mcnemar']:.4f}")

    print("\n=== E4 扰动网格（20 格合并, 9 m/s）===")
    for tier in MODES:
        st = summary["E4_grid"]["merged_20cell"][tier]
        print(f"  {tier}: {st['rate']:.1f}% [{st['ci95'][0]:.1f},{st['ci95'][1]:.1f}] n={st['n_valid']}")
    for comp, t in summary["E4_grid"]["paired_20cell"].items():
        if t.get("n_paired"):
            print(f"  {comp}: {t['diff_pp']:+.2f}pp "
                  f"CI=[{t['ci95'][0]:+.2f},{t['ci95'][1]:+.2f}] p={t['p_mcnemar']:.4f}")

    print("\n=== E7 角点（预先设定对比）===")
    for tag, t in summary["E7_corners"]["paired"].items():
        if t.get("n_paired"):
            print(f"  {tag}: {t['diff_pp']:+.2f}pp "
                  f"CI=[{t['ci95'][0]:+.2f},{t['ci95'][1]:+.2f}] p={t['p_mcnemar']:.4f} "
                  f"n={t['n_paired']}")

    print("\n=== E8 限速 × 机制（exp17i）===")
    for lim in ("TCP1.8", "TCP1.0"):
        vals = []
        for tier in MODES:
            st = summary["E8_limit_mechanism"]["cells"].get(f"{lim}|{tier}", {})
            vals.append(f"{tier}={st['rate']:.1f}%")
        print(f"  {lim}: " + " ".join(vals))
    for comp, t in summary["E8_limit_mechanism"]["paired"].items():
        if t.get("n_paired"):
            print(f"    {comp}: {t['diff_pp']:+.2f}pp "
                  f"CI=[{t['ci95'][0]:+.2f},{t['ci95'][1]:+.2f}] p={t['p_mcnemar']:.4f}")
    for a, d in summary["E8_limit_mechanism"]["did"].items():
        print(f"    DiD {a}: gain {d['gain_TCP18_pp']:+.2f} → {d['gain_TCP10_pp']:+.2f} pp, "
              f"mean DiD={d['mean_did_pp']:+.2f}pp CI={[round(x,2) for x in d['ci95']]} "
              f"sign p={d['p_sign']:.2e}")
    print("  诊断（命中行均值）:")
    for key, d in summary["E8_limit_mechanism"]["diagnostics"].items():
        print(f"    {key}: n_hit={d['n_hit_rows']} "
              f"t_err={d['mean_hit_time_error_ms']:.1f}ms v={d['mean_v_racket']:.2f}m/s")

    print("\n=== 排除行统计（error 均为 no_result）===")
    for name, st in summary["exclusion"].items():
        tiers = ", ".join(f"{k}:{v['error']}/{v['valid']+v['error']}"
                          for k, v in st["per_tier"].items())
        print(f"  {name}: 排除 {st['errors']}/{st['rows']} "
              f"({st['error_rate_pct']:.2f}%) [{tiers}]")

    OUT_PAPER.parent.mkdir(parents=True, exist_ok=True)
    OUT_PAPER.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    OUT_DATA.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n已写出:\n  {OUT_PAPER.relative_to(PROJECT)}\n  {OUT_DATA.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
