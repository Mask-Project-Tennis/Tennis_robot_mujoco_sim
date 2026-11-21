"""E1–E9 对比与 DiD：速度扫描、限速、机制消融、感知、空间与共享输入。

所有对比在 same-seed 配对口径下计算，返回可直接 JSON 序列化的字典；顶层键、
字段顺序与旧实现一致（追加式扩展）。
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable
from statistics import mean
from typing import Any

from .identity import (
    _obs_freq_of,
    _seed_value,
    space_perturb_of,
    speed_of,
    tcp_label_of,
    tier_of,
    time_perturb_ms_of,
)
from .metrics import (
    _physical_pair_summary,
    _unique_index,
    baseline_cell,
    cell_stats,
    mcnemar_exact,
    paired_bootstrap,
    paired_test,
    paired_test_physical,
    physical_of,
    physical_stats,
    sign_test,
)
from .sources import _load_rows
from .types import (
    MODES,
    NOFEEDFORWARD_SUFFIX,
    TIER_LABELS,
    HitPolicy,
    StatsContext,
    cfg_of,
    is_error,
)

# ----------------------------------------------------------------------------
# 各实验统计（每个函数返回 (legacy 段落, physical 段落)）
# ----------------------------------------------------------------------------

def e1_speed_sweep(hitfn: HitPolicy, ctx: StatsContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """E1: 球速扫描（exp15，9 档 × 200 seeds，TCP 1.8 默认限位）。"""
    rows = _load_rows(ctx, "exp15_speed_v2")
    by_speed: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_speed[speed_of(r)].append(r)
    legacy: dict[str, Any] = {}
    physical: dict[str, Any] = {}
    for sp in sorted(by_speed):
        legacy[str(sp)] = cell_stats(by_speed[sp], hitfn)
        physical[str(sp)] = physical_stats(by_speed[sp])
    return legacy, physical




def e2_limit_cost(hitfn: HitPolicy, ctx: StatsContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """E2: 真机限位代价（exp16，7 m/s × 200 seeds，TCP 1.8 vs 1.0）。"""
    rows = _load_rows(ctx, "exp16_limits_v2")
    by_lim: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_lim[tcp_label_of(r)].append(r)
    legacy: dict[str, Any] = {}
    physical: dict[str, Any] = {}
    for key, sub in by_lim.items():
        st = cell_stats(sub, hitfn)
        speeds = [
            float(r["v_racket_at_hit"])
            for r in sub
            if not is_error(r) and hitfn(r) == 1 and (r.get("v_racket_at_hit") or "")
        ]
        st["mean_racket_speed_on_hit"] = mean(speeds) if speeds else float("nan")
        legacy[key] = st
        physical[key] = physical_stats(sub)
    # 配对比较（同 seed 跨限速层）
    da, db = {}, {}
    pa, pb = {}, {}
    for r in rows:
        if is_error(r):
            continue
        is_tcp10 = tcp_label_of(r) == "TCP1.0"
        (da if is_tcp10 else db)[r["seed"]] = hitfn(r)
        verdict = physical_of(r)
        if verdict is not None:
            (pa if is_tcp10 else pb)[r["seed"]] = verdict
    keys = sorted(set(da) & set(db))
    if keys:
        ha = [da[k] for k in keys]
        hb = [db[k] for k in keys]
        legacy["paired_TCP10_vs_TCP18"] = {
            "n_paired": len(keys),
            "diff_pp": 100 * (sum(ha) - sum(hb)) / len(keys),
            "ci95": list(paired_bootstrap(ha, hb)),
            "p_mcnemar": mcnemar_exact(
                sum(1 for x, y in zip(ha, hb) if x == 1 and y == 0),
                sum(1 for x, y in zip(ha, hb) if x == 0 and y == 1),
            ),
        }
    pkeys = sorted(set(pa) & set(pb))
    if pkeys:
        physical["paired_TCP10_vs_TCP18"] = _physical_pair_summary(
            [pa[k] for k in pkeys], [pb[k] for k in pkeys])
    else:
        physical["paired_TCP10_vs_TCP18"] = {
            "n_paired": 0,
            "unavailable_reason": "no_physical_verdict",
        }
    return legacy, physical




def e3_nominal(hitfn: HitPolicy, ctx: StatsContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """E3: 标称四档（来源由 catalog 声明决定；缺 exp17a_noise 不阻断）。

    F06 根因：旧实现无条件先读 ``exp17a_noise``，R1D manifest 只声明
    ``source_sets.nominal_mechanism = exp17d_mechanism`` 时整节被记 unavailable。
    现在 catalog 声明了 mechanism 就只读它（四档都取自该来源）；无 catalog 的旧
    RM65 根仍走 exp17a（full/none）+ exp17d（tube_only/softmin_only）两段式来源。
    """
    declared = str(ctx.source_sets.get("nominal_mechanism") or "").strip()
    out: dict[str, Any] = {"cells": {}, "paired": {}, "data_sources": []}
    physical: dict[str, Any] = {"cells": {}, "paired": {}, "data_sources": []}
    if declared:
        rows = _load_rows(ctx, declared)
        tier_rows: dict[str, list[dict[str, str]]] = {tier: rows for tier in MODES}
        combined = rows
        out["data_sources"] = [declared]
    else:
        rows_a = _load_rows(ctx, "exp17a_noise")
        rows_d = _load_rows(ctx, "exp17d_mechanism")
        tier_rows = {"full": rows_a, "none": rows_a,
                     "tube_only": rows_d, "softmin_only": rows_d}
        combined = rows_a + rows_d
        out["data_sources"] = ["exp17a_noise", "exp17d_mechanism"]
    physical["data_sources"] = list(out["data_sources"])

    for sp in (7, 9, 12):
        for tier in MODES:
            sub = [r for r in tier_rows[tier]
                   if baseline_cell(r) and speed_of(r) == sp and tier_of(r) == tier]
            key = f"{sp}|{tier}"
            out["cells"][key] = cell_stats(sub, hitfn)
            physical["cells"][key] = physical_stats(sub)

    def at_speed(sp: int):
        def f(r: dict[str, str]) -> bool:
            if not baseline_cell(r):
                return False
            return speed_of(r) == sp
        return f

    for sp in (7, 9, 12):
        for a, b in (("full", "none"), ("softmin_only", "none"),
                     ("tube_only", "none"), ("full", "softmin_only"),
                     ("tube_only", "softmin_only")):
            key = f"{sp}|{a}_vs_{b}"
            out["paired"][key] = paired_test(combined, hitfn, a, b, filterfn=at_speed(sp))
            physical["paired"][key] = paired_test_physical(
                combined, a, b, filterfn=at_speed(sp))

    # M02：前馈关闭变体独立成节（不混入 nominal 四档，也不参与 nominal 配对）
    seen_variants: set[str] = set()
    for row in combined:
        tier = tier_of(row)
        if tier.endswith(NOFEEDFORWARD_SUFFIX):
            seen_variants.add(tier)
    variant_tiers = sorted(seen_variants)
    variants_legacy: dict[str, Any] = {
        "tiers": variant_tiers, "cells": {}, "paired": {},
    }
    variants_physical: dict[str, Any] = {
        "tiers": list(variant_tiers), "cells": {}, "paired": {},
    }
    for variant in variant_tiers:
        for sp in (7, 9, 12):
            sub = [r for r in combined
                   if baseline_cell(r) and speed_of(r) == sp and tier_of(r) == variant]
            cell_key = f"{sp}|{variant}"
            variants_legacy["cells"][cell_key] = cell_stats(sub, hitfn)
            variants_physical["cells"][cell_key] = physical_stats(sub)
            pair_key = f"{sp}|none_vs_{variant}"
            variants_legacy["paired"][pair_key] = paired_test(
                combined, hitfn, "none", variant, filterfn=at_speed(sp))
            variants_physical["paired"][pair_key] = paired_test_physical(
                combined, "none", variant, filterfn=at_speed(sp))
    out["variants"] = variants_legacy
    physical["variants"] = variants_physical
    return out, physical




def e4_grid(hitfn: HitPolicy, ctx: StatsContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """E4: 扰动网格（exp17b+17f，9 m/s，20 格合并，配对键 = (格, seed)）。"""
    rows = _load_rows(ctx, "exp17b_perturb") + _load_rows(ctx, "exp17f_mechanism_perturb")
    TS = {0, 10, 25, 50, 100}
    SS = {0.0, 0.05, 0.1, 0.2}

    def in_grid(r: dict[str, str]) -> bool:
        return (
            speed_of(r) == 9
            and time_perturb_ms_of(r) in TS
            and space_perturb_of(r) in SS
        )

    sub = [r for r in rows if in_grid(r)]
    out: dict[str, Any] = {"merged_20cell": {}, "paired_20cell": {}}
    physical: dict[str, Any] = {"merged_20cell": {}, "paired_20cell": {}}
    for tier in MODES:
        tier_rows = [r for r in sub if tier_of(r) == tier]
        out["merged_20cell"][tier] = cell_stats(tier_rows, hitfn)
        physical["merged_20cell"][tier] = physical_stats(tier_rows)

    def grid_key(r: dict[str, str]) -> str:
        return (f"{time_perturb_ms_of(r)}|"
                f"{space_perturb_of(r)}|{r['seed']}")

    for a, b in (("full", "none"), ("softmin_only", "none"), ("tube_only", "none"),
                 ("full", "softmin_only"), ("full", "tube_only"),
                 ("tube_only", "softmin_only")):
        key = f"{a}_vs_{b}"
        out["paired_20cell"][key] = paired_test(sub, hitfn, a, b, keyfn=grid_key)
        physical["paired_20cell"][key] = paired_test_physical(
            sub, a, b, keyfn=grid_key)
    # 每格明细（供热图显著性标注）+ 每格配对检验
    per_cell: dict[str, Any] = {}
    per_cell_paired: dict[str, Any] = {}
    per_cell_physical: dict[str, Any] = {}
    per_cell_paired_physical: dict[str, Any] = {}
    for t in sorted(TS):
        for s in sorted(SS):
            for tier in MODES:
                cellrows = [r for r in sub
                            if time_perturb_ms_of(r) == t
                            and space_perturb_of(r) == s
                            and tier_of(r) == tier]
                key = f"t{t}|s{s}|{tier}"
                per_cell[key] = cell_stats(cellrows, hitfn)
                per_cell_physical[key] = physical_stats(cellrows)

            def cell_filter(r: dict[str, str], t: int = t, s: float = s) -> bool:
                return time_perturb_ms_of(r) == t and space_perturb_of(r) == s

            for a, b in (("full", "none"), ("tube_only", "none"),
                         ("softmin_only", "none"), ("full", "softmin_only")):
                key = f"t{t}|s{s}|{a}_vs_{b}"
                per_cell_paired[key] = paired_test(
                    sub, hitfn, a, b, filterfn=cell_filter)
                per_cell_paired_physical[key] = paired_test_physical(
                    sub, a, b, filterfn=cell_filter)
    out["per_cell"] = per_cell
    out["per_cell_paired"] = per_cell_paired
    physical["per_cell"] = per_cell_physical
    physical["per_cell_paired"] = per_cell_paired_physical
    return out, physical




def e5_perception(hitfn: HitPolicy, ctx: StatsContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """E5: 噪声 × KF（exp17a）与观测频率（exp17c）。"""
    rows = _load_rows(ctx, "exp17a_noise")
    out: dict[str, Any] = {"noise_x_kf": {}}
    physical: dict[str, Any] = {"noise_x_kf": {}}
    for sp in (7, 9, 12):
        for pos, vel in sorted({(cfg_of(r).get("--obs-noise-pos"),
                                 cfg_of(r).get("--obs-noise-vel"))
                                for r in rows}, key=str):
            for kf in (False, True):
                sub = [r for r in rows
                       if speed_of(r) == sp
                       and cfg_of(r).get("--obs-noise-pos") == pos
                       and cfg_of(r).get("--obs-noise-vel") == vel
                       and (("--obs-use-kf" in cfg_of(r)) == kf)]
                for tier in ("full", "none"):
                    cellrows = [r for r in sub if tier_of(r) == tier]
                    if cellrows:
                        key = f"{sp}|pos{pos}|vel{vel}|kf{int(kf)}|{tier}"
                        out["noise_x_kf"][key] = cell_stats(cellrows, hitfn)
                        physical["noise_x_kf"][key] = physical_stats(cellrows)
    rows_c = _load_rows(ctx, "exp17c_obsfreq")
    by_rate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows_c:
        by_rate[_obs_freq_of(r)].append(r)
    out["obsfreq"] = {k: cell_stats(v, hitfn) for k, v in sorted(by_rate.items())}
    physical["obsfreq"] = {k: physical_stats(v) for k, v in sorted(by_rate.items())}
    return out, physical




def e7_corners(hitfn: HitPolicy, ctx: StatsContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """E7: 角点高功率格（exp17g / exp17h，预先设定的 4 个对比）。"""
    rows_g, rows_h = _load_rows(ctx, "exp17g_spatial_power"), _load_rows(ctx, "exp17h_extreme")
    out: dict[str, Any] = {"cells": {}, "paired": {}}
    physical: dict[str, Any] = {"cells": {}, "paired": {}}
    specs = [("exp17g", rows_g, 9, 0.2, 0), ("exp17g", rows_g, 9, 0.2, 50),
             ("exp17h", rows_h, 12, 0.2, 0), ("exp17h", rows_h, 9, 0.3, 0),
             ("exp17h", rows_h, 9, 0.4, 0)]

    def filt(sp: int, s: float, t: int):
        def f(r: dict[str, str]) -> bool:
            return (speed_of(r) == sp
                    and space_perturb_of(r) == s
                    and time_perturb_ms_of(r) == t)
        return f

    for src, rows, sp, s, t in specs:
        tag = f"{sp}|s{s}|t{t}"
        for tier in MODES:
            cellrows = [r for r in rows if filt(sp, s, t)(r) and tier_of(r) == tier]
            key = f"{tag}|{tier}"
            out["cells"][key] = cell_stats(cellrows, hitfn)
            physical["cells"][key] = physical_stats(cellrows)
        key = f"{tag}|tube_only_vs_none"
        out["paired"][key] = paired_test(rows, hitfn, "tube_only", "none",
                                         filterfn=filt(sp, s, t))
        physical["paired"][key] = paired_test_physical(
            rows, "tube_only", "none", filterfn=filt(sp, s, t))
        key = f"{tag}|full_vs_softmin_only"
        out["paired"][key] = paired_test(rows, hitfn, "full", "softmin_only",
                                         filterfn=filt(sp, s, t))
        physical["paired"][key] = paired_test_physical(
            rows, "full", "softmin_only", filterfn=filt(sp, s, t))
    return out, physical




def _did_reason(
    relevant: list[dict[str, str]],
    errors: list[dict[str, str]],
    verdicts: int,
    semantics: str,
) -> str:
    """DiD 不可估计原因（区分缺行、全 error、无判定、无公共 seed）。"""
    if not relevant:
        return "no_rows"
    if errors and len(errors) == len(relevant):
        return "all_error"
    if verdicts == 0:
        return "no_physical_verdict" if semantics == "physical" else "no_verdict"
    return "no_common_seed"




def _did(
    rows: list[dict[str, str]],
    value_fn: Callable[[dict[str, str]], int | None],
    baseline: str = "none",
    *,
    semantics: str = "legacy",
) -> dict[str, Any]:
    """配对 DiD：每 seed 的 (档 - 基线) 增益在 TCP1.0/TCP1.8 两层之差。

    R06/M04：DiD 的四个条件（档 × TCP 层 × 是否基线）各自建立唯一索引，
    同一 seed 在同一条件内取值相同则去重、取值不同则整键排除；只有四个索引
    都存在的 seed 才进入 DiD。因此结果与 CSV 行序无关，也不会出现“普通配对
    已判 duplicate_seed_conflict、DiD 却仍给出显著差异”的自相矛盾。

    Args:
        rows: 同一实验的记录行。
        value_fn: 命中判定函数（legacy ``HitPolicy`` 或 ``physical_of``）。
        baseline: 基线档名（默认 ``none``）。
        semantics: ``legacy`` / ``physical``；physical 全 miss 时保留样本数与观察差
            （可为 0.0），但 CI/p 推断字段置 None 并标 ``unavailable_reason="all_miss"``。

    Returns:
        档名 → DiD 结果（``n_pairs`` 恒在；有冲突键时附
        ``excluded_duplicate_keys``；无法估计时给 ``unavailable_reason``，
        若因重复 seed 冲突导致无样本则为 ``duplicate_seed_conflict``）。
    """
    out: dict[str, Any] = {}
    for a in ("softmin_only", "full", "tube_only"):
        relevant = [r for r in rows if tier_of(r) in (a, baseline)]
        errors = [r for r in relevant if is_error(r)]
        usable = [r for r in relevant if not is_error(r)]
        verdicts = sum(1 for r in usable if value_fn(r) is not None)
        # 四个条件各自去重取值：同值去重、异值整键排除（行序无关）
        indexes: dict[tuple[str, str], dict[Hashable, int]] = {}
        raw_keys: list[set[Hashable]] = []
        excluded: set[Hashable] = set()
        for tcp_label in ("TCP1.0", "TCP1.8"):
            for tier in (a, baseline):
                subset = [
                    r for r in usable
                    if tcp_label_of(r) == tcp_label and tier_of(r) == tier
                ]
                index, conflicts = _unique_index(subset, _seed_value, value_fn)
                indexes[(tcp_label, tier)] = index
                raw_keys.append(set(index) | set(conflicts))
                excluded.update(conflicts)
        common = set(indexes[("TCP1.0", a)])
        for index in indexes.values():
            common &= set(index)
        dids = [
            (indexes[("TCP1.0", a)][seed] - indexes[("TCP1.0", baseline)][seed])
            - (indexes[("TCP1.8", a)][seed] - indexes[("TCP1.8", baseline)][seed])
            for seed in sorted(common)
        ]
        excluded_keys = sorted(excluded)
        entry: dict[str, Any] = {"n_pairs": len(dids), "tier": a, "baseline": baseline}
        if semantics == "physical":
            entry["hit_semantics"] = "physical"
        if excluded_keys:
            entry["excluded_duplicate_keys"] = excluded_keys
        if not dids:
            entry["mean_did_pp"] = None
            entry["ci95"] = None
            entry["p_sign"] = None
            if excluded_keys and set.intersection(*raw_keys):
                # 四个条件本来有共同 seed，但冲突把它们全部排除：
                # 这与普通配对的 duplicate_seed_conflict 保持一致，不输出显著性
                reason = "duplicate_seed_conflict"
            else:
                reason = _did_reason(relevant, errors, verdicts, semantics)
            entry["unavailable_reason"] = reason
            out[a] = entry
            continue
        entry["mean_did_pp"] = 100 * mean(dids)
        total_hits = sum(
            1 for index in indexes.values() for value in index.values() if value == 1
        )
        if semantics == "physical" and total_hits == 0:
            # 全 miss 是测得的 0%，但不能据此声称"无差异"：不估计区间与符号检验。
            entry["ci95"] = None
            entry["p_sign"] = None
            entry["unavailable_reason"] = "all_miss"
            out[a] = entry
            continue
        rng = random.Random(42)
        n = len(dids)
        boot = sorted(
            sum(dids[i] for i in [rng.randrange(n) for _ in range(n)]) / n
            for _ in range(20000))
        entry["ci95"] = [100 * boot[int(0.025 * 20000)],
                         100 * boot[int(0.975 * 20000)]]
        entry["p_sign"] = sign_test(dids)
        out[a] = entry
    return out




def e8_limit_mechanism(
    hitfn: HitPolicy, ctx: StatsContext
) -> tuple[dict[str, Any], dict[str, Any]]:
    """E8: 限速 × 机制配对（exp17i）+ 配对 DiD。"""
    rows = _load_rows(ctx, "exp17i_limits_ablation")
    out: dict[str, Any] = {"cells": {}, "paired": {}, "did": {}, "diagnostics": {}}
    physical: dict[str, Any] = {"cells": {}, "paired": {}, "did": {}}
    for lim in ("TCP1.8", "TCP1.0"):
        for tier in MODES:
            sub = [r for r in rows
                   if tcp_label_of(r) == lim and tier_of(r) == tier]
            key = f"{lim}|{tier}"
            out["cells"][key] = cell_stats(sub, hitfn)
            physical["cells"][key] = physical_stats(sub)
            # 命中行的诊断量：时间误差、击球瞬间拍速
            hitrows = [r for r in sub if not is_error(r) and hitfn(r) == 1]
            errs = [float(r["hit_time_error_ms"]) for r in hitrows
                    if (r.get("hit_time_error_ms") or "")]
            vels = [float(r["v_racket_at_hit"]) for r in hitrows
                    if (r.get("v_racket_at_hit") or "")]
            out["diagnostics"][key] = {
                "n_hit_rows": len(hitrows),
                "mean_hit_time_error_ms": mean(errs) if errs else float("nan"),
                "mean_v_racket": mean(vels) if vels else float("nan"),
            }
        for a, b in (("softmin_only", "none"), ("full", "none"),
                     ("tube_only", "none"), ("full", "softmin_only")):
            key = f"{lim}|{a}_vs_{b}"
            filterfn = lambda r, lim=lim: tcp_label_of(r) == lim
            out["paired"][key] = paired_test(rows, hitfn, a, b, filterfn=filterfn)
            physical["paired"][key] = paired_test_physical(
                rows, a, b, filterfn=filterfn)
    # 配对 DiD：每 seed 的 (档 - 基线) 增益在两层之差
    legacy_did = _did(rows, hitfn, semantics="legacy")
    phys_did = _did(rows, physical_of, semantics="physical")
    for a, d in legacy_did.items():
        g1 = out["paired"].get(f"TCP1.0|{a}_vs_none", {})
        g0 = out["paired"].get(f"TCP1.8|{a}_vs_none", {})
        d["gain_TCP18_pp"] = g0.get("diff_pp", float("nan"))
        d["gain_TCP10_pp"] = g1.get("diff_pp", float("nan"))
    out["did"] = legacy_did
    for a, d in phys_did.items():
        g1 = physical["paired"].get(f"TCP1.0|{a}_vs_none", {})
        g0 = physical["paired"].get(f"TCP1.8|{a}_vs_none", {})
        d["gain_TCP18_pp"] = g0.get("diff_pp")
        d["gain_TCP10_pp"] = g1.get("diff_pp")
        d["hit_semantics"] = "physical"
    physical["did"] = phys_did
    return out, physical




def e9_shared_input(
    hitfn: HitPolicy, ctx: StatsContext
) -> tuple[dict[str, Any], dict[str, Any]]:
    """E9: shared-input 协议（exp24_shared_full；table3 数据源，旧 JSON 中无此节）。"""
    out: dict[str, Any] = {"cells": {}, "paired": {}, "data_sources": []}
    physical: dict[str, Any] = {"cells": {}, "paired": {}}
    rows: list[dict[str, str]] = []
    for name in ("exp24_shared_full", "exp22_shared_spatial"):
        try:
            loaded = _load_rows(ctx, name)
        except FileNotFoundError:
            continue
        out["data_sources"].append(name)
        rows.extend(loaded)
    if not rows:
        out["unavailable_reason"] = "missing exp24_shared_full / exp22_shared_spatial"
        physical["unavailable_reason"] = out["unavailable_reason"]
        return out, physical

    def cell_of(r: dict[str, str]) -> tuple[int, float]:
        return time_perturb_ms_of(r), space_perturb_of(r)

    def grid_key(r: dict[str, str]) -> str:
        t, s = cell_of(r)
        return f"{t}|{s}|{r['seed']}"

    for t, s in sorted({cell_of(r) for r in rows}):
        sub = [r for r in rows if cell_of(r) == (t, s)]
        for tier in MODES:
            tier_rows = [r for r in sub if tier_of(r) == tier]
            if not tier_rows:
                continue
            key = f"t{t}|s{s}|{tier}"
            out["cells"][key] = cell_stats(tier_rows, hitfn)
            physical["cells"][key] = physical_stats(tier_rows)
        for a, b in (("full", "none"), ("tube_only", "none"),
                     ("softmin_only", "none"), ("full", "softmin_only")):
            key = f"t{t}|s{s}|{a}_vs_{b}"
            out["paired"][key] = paired_test(sub, hitfn, a, b, keyfn=grid_key)
            physical["paired"][key] = paired_test_physical(
                sub, a, b, keyfn=grid_key)
    return out, physical




def exclusion_report(ctx: StatsContext) -> dict[str, Any]:
    """排除行逐档计数（验证排除与方法无关）。"""
    out: dict[str, Any] = {}
    for name in ("exp15_speed_v2", "exp16_limits_v2", "exp17a_noise",
                 "exp17b_perturb", "exp17c_obsfreq", "exp17d_mechanism",
                 "exp17e_perturb7", "exp17f_mechanism_perturb",
                 "exp17g_spatial_power", "exp17h_extreme",
                 "exp17i_limits_ablation"):
        try:
            rows = _load_rows(ctx, name)
        except FileNotFoundError:
            continue
        by_tier: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [valid, error]
        causes: Counter[str] = Counter()
        for r in rows:
            tier = tier_of(r) or "-"
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
# 表格数据（T8c 的 table1/2/3 数据源；只产数据不画图）
# ----------------------------------------------------------------------------

def _tier_cell(cells: dict[str, Any], key: str, tier: str) -> dict[str, Any]:
    """从 legacy cells 与 physical cells 组合出表格单元（含 missing/n 标记）。"""
    legacy_cell = cells.get(f"{key}|{tier}")
    if legacy_cell is None:
        return {"present": False, "tier": tier, "tier_label": TIER_LABELS[tier],
                "unavailable_reason": "missing_cell"}
    return legacy_cell


__all__ = [
    "_did",
    "_did_reason",
    "_tier_cell",
    "e1_speed_sweep",
    "e2_limit_cost",
    "e3_nominal",
    "e4_grid",
    "e5_perception",
    "e7_corners",
    "e8_limit_mechanism",
    "e9_shared_input",
    "exclusion_report",
]
