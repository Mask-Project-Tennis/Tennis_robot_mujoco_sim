"""汇总、表格与写盘：build_tables / compute_summary / 报告打印与 JSON 写出。

E1–E8 顶层键与字段保持旧定义不变；新内容只追加在 ``hit_semantics`` / ``physical``
/ ``E9_shared_input`` / ``tables`` / ``data_root`` / ``unavailable`` 下。
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .comparisons import (
    e1_speed_sweep,
    e2_limit_cost,
    e3_nominal,
    e4_grid,
    e5_perception,
    e7_corners,
    e8_limit_mechanism,
    e9_shared_input,
    exclusion_report,
)
from .dataset import _validate_dataset, resolve_mode
from .sources import _runs_csv_path, make_context
from .types import (
    HIT_SEMANTICS,
    MODES,
    MULTI_MODE_SCHEMA,
    SCHEMA,
    STATS_FILENAMES,
    TABLE3_CELLS,
    TIER_LABELS,
    HitPolicy,
    StatsContext,
)


def _fmt_optional(value: Any, fmt: str = "{:.2f}") -> str:
    """格式化可缺省数值：None/NaN 渲染为 ``--``（不伪造 0）。"""
    if value is None:
        return "--"
    if isinstance(value, float) and math.isnan(value):
        return "--"
    return fmt.format(value)




def _fmt_ci(ci: Any) -> str:
    """格式化配对 CI：None/不可解析 → ``不可估计``（R06：绝不写零宽区间）。"""
    if ci is None:
        return "不可估计"
    try:
        lo, hi = float(ci[0]), float(ci[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return "不可估计"
    return f"[{lo:+.2f},{hi:+.2f}]"




def _fmt_p(p_value: Any, fmt: str = "{:.4f}") -> str:
    """格式化配对 p 值：None/NaN → ``不可估计``（R06：绝不格式化成 0）。"""
    if p_value is None:
        return "不可估计"
    try:
        value = float(p_value)
    except (TypeError, ValueError):
        return "不可估计"
    if math.isnan(value):
        return "不可估计"
    return fmt.format(value)




def _fmt_pp(value: Any) -> str:
    """格式化 pp 差值/增益：None/NaN → ``不可估计``。"""
    if value is None:
        return "不可估计"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "不可估计"
    if math.isnan(v):
        return "不可估计"
    return f"{v:+.2f}"




def _reason_text(stats: dict[str, Any]) -> str:
    """从 cell 统计里取不可估计原因（缺失时给通用说明）。"""
    return str(stats.get("unavailable_reason") or "无有效判定（全 miss / 无 verdict / 缺数据）")




def build_tables(
    legacy: dict[str, Any], physical: dict[str, Any]
) -> dict[str, Any]:
    """构造 table1/2/3 的稳定数据键（列映射见 column_map；不画图）。"""
    t1_rows: list[dict[str, Any]] = []
    table1_specs = [
        ("nominal 7 m/s", "E3_nominal", "7"),
        ("nominal 9 m/s", "E3_nominal", "9"),
        ("nominal 12 m/s", "E3_nominal", "12"),
        ("perturbed grid (20 cells, 9 m/s)", "E4_grid", "merged"),
        ('corner 12 m/s t=0 s=0.2 m', "E7_corners", "12|s0.2|t0"),
    ]
    for label, section, prefix in table1_specs:
        cells_legacy = legacy.get(section) or {}
        cells_physical = physical.get(section) or {}
        row: dict[str, Any] = {"condition": label, "source": f"{section}:{prefix}", "cells": {}}
        n_values: list[int] = []
        for tier in MODES:
            if section == "E4_grid":
                leg = cells_legacy.get("merged_20cell", {}).get(tier, {})
                phy = cells_physical.get("merged_20cell", {}).get(tier, {})
            else:
                leg = cells_legacy.get("cells", {}).get(f"{prefix}|{tier}", {})
                phy = cells_physical.get("cells", {}).get(f"{prefix}|{tier}", {})
            present = bool(leg)
            if present:
                n_values.append(int(leg.get("n_valid") or 0))
            row["cells"][tier] = {
                "present": present,
                "tier_label": TIER_LABELS[tier],
                "legacy_rate": leg.get("rate"),
                "legacy_n": leg.get("n_valid"),
                "legacy_hits": leg.get("hits"),
                "physical_rate": phy.get("rate"),
                "physical_denominator": phy.get("denominator"),
                "physical_n_active": phy.get("n_active"),
                "unavailable_reason": phy.get("unavailable_reason"),
            }
        if n_values:
            row["n_range"] = [min(n_values), max(n_values)] if min(n_values) != max(n_values) \
                else [n_values[0], n_values[0]]
        else:
            row["n_range"] = None
            row["unavailable_reason"] = "missing_cell"
        t1_rows.append(row)

    t2_rows: list[dict[str, Any]] = []
    e8_legacy = legacy.get("E8_limit_mechanism") or {}
    e8_physical = physical.get("E8_limit_mechanism") or {}
    for tier in ("full", "tube_only", "softmin_only"):
        entry: dict[str, Any] = {"tier": tier, "tier_label": TIER_LABELS[tier]}
        for label in ("TCP1.8", "TCP1.0"):
            entry[f"gain_{label}_legacy"] = e8_legacy.get("paired", {}).get(
                f"{label}|{tier}_vs_none", {})
            entry[f"gain_{label}_physical"] = e8_physical.get("paired", {}).get(
                f"{label}|{tier}_vs_none", {})
        entry["did_legacy"] = e8_legacy.get("did", {}).get(tier, {})
        entry["did_physical"] = e8_physical.get("did", {}).get(tier, {})
        t2_rows.append(entry)

    t3_rows: list[dict[str, Any]] = []
    shared_legacy = legacy.get("E9_shared_input", {})
    shared_physical = physical.get("E9_shared_input", {})
    for t, s in TABLE3_CELLS:
        row = {"cell": f"t{t:g}|s{s:g}", "t": t, "s": s, "cells": {}, "missing": []}
        for tier in MODES:
            key = f"t{t:g}|s{s:g}|{tier}"
            leg = shared_legacy.get("cells", {}).get(key)
            phy = shared_physical.get("cells", {}).get(key)
            if leg is None:
                row["missing"].append(tier)
                row["cells"][tier] = {"present": False, "tier_label": TIER_LABELS[tier],
                                      "unavailable_reason": "missing_cell"}
                continue
            row["cells"][tier] = {
                "present": True,
                "tier_label": TIER_LABELS[tier],
                "legacy_rate": leg.get("rate"),
                "legacy_n": leg.get("n_valid"),
                "physical_rate": phy.get("rate") if phy else None,
                "physical_denominator": phy.get("denominator") if phy else 0,
                "unavailable_reason": phy.get("unavailable_reason") if phy else "no_physical_verdict",
            }
        p = shared_legacy.get("paired", {}).get(f"t{t:g}|s{s:g}|full_vs_none", {})
        row["p_raw_legacy"] = p.get("p_mcnemar")
        row["n_paired_legacy"] = p.get("n_paired")
        pp = shared_physical.get("paired", {}).get(f"t{t:g}|s{s:g}|full_vs_none", {})
        row["p_raw_physical"] = pp.get("p_mcnemar")
        row["n_paired_physical"] = pp.get("n_paired")
        t3_rows.append(row)

    return {
        "table1_comparison": {
            "description": "消融四档命中率（对应 paper/figures/table_data/table1_comparison.tex，"
                           "文件名与内容错位：该文件实为表 II ablation 表）",
            "columns": ["condition", "full", "tube-only", "candidate-set-only",
                        "point-target", "n", "design"],
            "rows": t1_rows,
        },
        "table2_limits": {
            "description": "限速层配对增益与 DiD（对应 table2_ablation.tex，实为表 I limits 表）",
            "columns": ["tier", "gain_TCP1.8", "gain_TCP1.0", "did"],
            "rows": t2_rows,
        },
        "table3_shared_input": {
            "description": "shared-input 协议（对应 table3_shared_input.tex，数据源 exp24_shared_full）",
            "columns": ["cell", "full", "tube-only", "candidate-set-only",
                        "point-target", "p_raw"],
            "rows": t3_rows,
        },
        "column_map": {
            "table1_comparison.tex": {
                "columns": {
                    "full": "tables.table1_comparison.rows[].cells.full.legacy_rate",
                    "tube-only": "tables.table1_comparison.rows[].cells.tube_only.legacy_rate",
                    "candidate-set-only": "tables.table1_comparison.rows[].cells.softmin_only.legacy_rate",
                    "point-target": "tables.table1_comparison.rows[].cells.none.legacy_rate",
                    "n": "tables.table1_comparison.rows[].n_range",
                },
                "note": "physical_rate 为 R1D 新口径（旧 RM65 根为 null + unavailable_reason）",
            },
            "table2_ablation.tex": {
                "columns": {
                    "Gain over point-target (pp)": "tables.table2_limits.rows[].gain_TCP1.8_legacy.diff_pp",
                    "p": "tables.table2_limits.rows[].gain_TCP1.8_legacy.p_mcnemar",
                    "DiD": "tables.table2_limits.rows[].did_legacy",
                },
                "note": "上块为 E8 paired legacy；下块为 E8 did legacy；physical 为并列新增",
            },
            "table3_shared_input.tex": {
                "columns": {
                    "$(t,s)$ cell": "tables.table3_shared_input.rows[].cell",
                    "full": "tables.table3_shared_input.rows[].cells.full.legacy_rate",
                    "tube-only": "tables.table3_shared_input.rows[].cells.tube_only.legacy_rate",
                    "candidate-set-only": "tables.table3_shared_input.rows[].cells.softmin_only.legacy_rate",
                    "point-target": "tables.table3_shared_input.rows[].cells.none.legacy_rate",
                    "$p_{raw}$": "tables.table3_shared_input.rows[].p_raw_legacy",
                },
                "note": "缺失 (t,s)/tier 在 rows[].missing / cells[*].present=False 中显式标注",
            },
        },
    }




# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def default_output_dir(data_root: Path) -> Path:
    """默认输出目录：数据根的兄弟目录 ``stats/``（绝不写回 data_root 或 paper/）。"""
    return data_root.parent / "stats"




def _catalog_metadata(ctx: StatsContext) -> dict[str, Any] | None:
    """summary 中的 catalog 元数据（无 catalog 时为 None，旧调用结构不变）。"""
    if ctx.catalog is None:
        return None
    return {
        "path": str(ctx.catalog_path) if ctx.catalog_path is not None else None,
        "schema_version": int(getattr(ctx.catalog, "schema_version", 0)),
        "robot_id": str(getattr(ctx.catalog, "robot_id", "")),
        "run_root": str(ctx.catalog.run_root),
        "data_root": str(ctx.catalog.data_root),
        "path_base": str(getattr(ctx.catalog, "path_base", "")),
        "source_sets": dict(ctx.source_sets),
        "synthetic": bool(ctx.synthetic),
    }




def compute_summary(
    data_root: Path | None = None,
    include_passive: bool = False,
    *,
    catalog_path: Path | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """计算完整统计 summary（不写盘；供主流程与测试复用）。

    Args:
        data_root: 实验数据根；None 时用 catalog 声明的 data_root，再退回默认根。
        include_passive: 是否把 passive 计入 legacy 命中（复现旧口径）。
        catalog_path: 可选 ``catalog.json``；显式坏 catalog 抛 StatsError（不回退）。
        mode: ``torque`` / ``position`` / None（自动推断；混合模式必须显式指定）。

    Returns:
        统计 summary。既有键结构不变，只新增 ``mode`` / ``catalog`` / ``synthetic`` /
        ``validation`` 元数据，以及 ``E3_nominal.data_sources``。

    Raises:
        StatsError: catalog 非法、模式混合未指定、记录重复或身份冲突。
    """
    ctx = make_context(data_root, catalog_path=catalog_path, mode=mode)
    resolved = resolve_mode(ctx)
    ctx = ctx.with_mode(resolved)
    validation = _validate_dataset(ctx)
    hitfn = HitPolicy(include_passive=include_passive)
    metric = "hit(active+passive)" if include_passive else "active-hit"

    summary: dict[str, Any] = {"metric": metric, "include_passive": include_passive}
    physical: dict[str, Any] = {}
    unavailable: list[str] = []
    for key, builder in (
        ("E1_speed_sweep", e1_speed_sweep),
        ("E2_limit_cost", e2_limit_cost),
        ("E3_nominal", e3_nominal),
        ("E4_grid", e4_grid),
        ("E5_perception", e5_perception),
        ("E7_corners", e7_corners),
        ("E8_limit_mechanism", e8_limit_mechanism),
    ):
        try:
            legacy_section, physical_section = builder(hitfn, ctx)
        except FileNotFoundError as exc:
            unavailable.append(f"{key}: {exc}")
            legacy_section, physical_section = {}, {}
        summary[key] = legacy_section
        physical[key] = physical_section
    summary["exclusion"] = exclusion_report(ctx)
    legacy_e9, physical_e9 = e9_shared_input(hitfn, ctx)
    summary["E9_shared_input"] = legacy_e9
    physical["E9_shared_input"] = physical_e9
    if legacy_e9.get("unavailable_reason"):
        unavailable.append(f"E9_shared_input: {legacy_e9['unavailable_reason']}")

    summary["mode"] = resolved
    summary["catalog"] = _catalog_metadata(ctx)
    summary["synthetic"] = bool(ctx.synthetic)
    summary["validation"] = validation
    summary["hit_semantics"] = HIT_SEMANTICS
    summary["physical"] = physical
    summary["tables"] = build_tables(summary, physical)
    summary["schema"] = SCHEMA
    summary["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary["data_root"] = str(ctx.data_root.resolve())
    summary["sources"] = _existing_sources(ctx)
    summary["unavailable"] = unavailable
    return summary




def _existing_sources(ctx: StatsContext) -> list[str]:
    """列出本次实际读取过的 CSV 路径（供 T8c 追溯数据来源）。"""
    paths: list[Path] = []
    runs_csv = _runs_csv_path(ctx)
    if runs_csv is not None and runs_csv.is_file():
        paths.append(runs_csv)
    paths.extend(sorted(ctx.data_root.glob("*/results.csv")))
    return [str(path) for path in paths]




def _print_report(summary: dict[str, Any]) -> None:
    """打印可读报告（保持旧输出小节；分模式时由 main 加标题）。"""
    print(f"命中口径: {summary['metric']}")
    print(f"数据根: {summary['data_root']}")
    if summary.get("mode"):
        print(f"执行模式: {summary['mode']}")
    catalog = summary.get("catalog") or {}
    if catalog:
        print(f"catalog: {catalog.get('path')}")
    if summary["unavailable"]:
        print(f"不可用数据: {summary['unavailable']}")
    print()

    # ---- 打印可读报告（保持旧输出小节） ----
    print("=== E1 球速扫描（active-hit %，含 95% CI 与 n）===")
    for sp, st in summary["E1_speed_sweep"].items():
        if st.get("rate") is None:
            print(f"  {sp:>2} m/s: 不可估计（{_reason_text(st)}）")
            continue
        print(f"  {sp:>2} m/s: {st['rate']:.1f}% [{st['ci95'][0]:.1f},{st['ci95'][1]:.1f}] "
              f"n={st['n_valid']} passive={st['passive']} ({st['passive_share']:.1f}%)")

    print("\n=== E2 限位代价（exp16, 7 m/s）===")
    for key in ("TCP1.8", "TCP1.0"):
        st = summary["E2_limit_cost"].get(key, {})
        if st:
            if st.get("rate") is None:
                print(f"  {key}: 不可估计（{_reason_text(st)}）")
                continue
            print(f"  {key}: {st['rate']:.1f}% n={st['n_valid']} "
                  f"v_racket={_fmt_optional(st.get('mean_racket_speed_on_hit'), '{:.2f}')} m/s")
    p = summary["E2_limit_cost"].get("paired_TCP10_vs_TCP18", {})
    if p.get("n_paired"):
        print(f"  配对差: {_fmt_optional(p.get('diff_pp'), '{:+.1f}')}pp "
              f"CI={_fmt_ci(p.get('ci95'))} p={_fmt_p(p.get('p_mcnemar'), '{:.2e}')}")

    print("\n=== E3 标称四档（active-hit %）===")
    e3_cells = summary["E3_nominal"].get("cells", {})
    for sp in (7, 9, 12):
        vals = []
        for tier in MODES:
            st = e3_cells.get(f"{sp}|{tier}", {})
            vals.append(f"{tier}={_fmt_optional(st.get('rate'), '{:.1f}')}%"
                        f"(n={st.get('n_valid', 0)})")
        print(f"  {sp} m/s: " + " ".join(vals))
        for a, b in (("full", "none"), ("softmin_only", "none"),
                     ("tube_only", "none"), ("full", "softmin_only")):
            t = summary["E3_nominal"].get("paired", {}).get(f"{sp}|{a}_vs_{b}", {})
            if t.get("n_paired"):
                print(f"    {a} vs {b}: {_fmt_optional(t.get('diff_pp'), '{:+.2f}')}pp "
                      f"CI={_fmt_ci(t.get('ci95'))} p={_fmt_p(t.get('p_mcnemar'))}")

    print("\n=== E4 扰动网格（20 格合并, 9 m/s）===")
    for tier in MODES:
        st = summary["E4_grid"].get("merged_20cell", {}).get(tier)
        if st:
            if st.get("rate") is None:
                print(f"  {tier}: 不可估计（{_reason_text(st)}）")
                continue
            print(f"  {tier}: {st['rate']:.1f}% [{st['ci95'][0]:.1f},{st['ci95'][1]:.1f}] "
                  f"n={st['n_valid']}")
    for comp, t in summary["E4_grid"].get("paired_20cell", {}).items():
        if t.get("n_paired"):
            print(f"  {comp}: {_fmt_optional(t.get('diff_pp'), '{:+.2f}')}pp "
                  f"CI={_fmt_ci(t.get('ci95'))} p={_fmt_p(t.get('p_mcnemar'))}")

    print("\n=== E7 角点（预先设定对比）===")
    for tag, t in summary["E7_corners"].get("paired", {}).items():
        if t.get("n_paired"):
            print(f"  {tag}: {_fmt_optional(t.get('diff_pp'), '{:+.2f}')}pp "
                  f"CI={_fmt_ci(t.get('ci95'))} p={_fmt_p(t.get('p_mcnemar'))} "
                  f"n={t.get('n_paired')}")

    print("\n=== E8 限速 × 机制（exp17i）===")
    e8_cells = summary["E8_limit_mechanism"].get("cells", {})
    for lim in ("TCP1.8", "TCP1.0"):
        vals = []
        for tier in MODES:
            st = e8_cells.get(f"{lim}|{tier}", {})
            vals.append(f"{tier}={_fmt_optional(st.get('rate'), '{:.1f}')}%")
        print(f"  {lim}: " + " ".join(vals))
    for comp, t in summary["E8_limit_mechanism"].get("paired", {}).items():
        if t.get("n_paired"):
            print(f"    {comp}: {_fmt_optional(t.get('diff_pp'), '{:+.2f}')}pp "
                  f"CI={_fmt_ci(t.get('ci95'))} p={_fmt_p(t.get('p_mcnemar'))}")
    for a, d in summary["E8_limit_mechanism"].get("did", {}).items():
        print(f"    DiD {a}: gain {_fmt_pp(d.get('gain_TCP18_pp'))} → "
              f"{_fmt_pp(d.get('gain_TCP10_pp'))} pp, "
              f"mean DiD={_fmt_pp(d.get('mean_did_pp'))}pp CI={_fmt_ci(d.get('ci95'))} "
              f"sign p={_fmt_p(d.get('p_sign'), '{:.2e}')}")
    print("  诊断（命中行均值）:")
    for key, d in summary["E8_limit_mechanism"].get("diagnostics", {}).items():
        print(f"    {key}: n_hit={d['n_hit_rows']} "
              f"t_err={_fmt_optional(d.get('mean_hit_time_error_ms'), '{:.1f}')}ms "
              f"v={_fmt_optional(d.get('mean_v_racket'), '{:.2f}')}m/s")

    print("\n=== 排除行统计（error 均为 no_result）===")
    for name, st in summary["exclusion"].items():
        tiers = ", ".join(f"{k}:{v['error']}/{v['valid']+v['error']}"
                          for k, v in st["per_tier"].items())
        print(f"  {name}: 排除 {st['errors']}/{st['rows']} "
              f"({st['error_rate_pct']:.2f}%) [{tiers}]")

    e9_cells = summary["E9_shared_input"].get("cells") or {}
    if e9_cells:
        print("\n=== E9 shared-input（table3 数据源）===")
        for key, st in sorted(e9_cells.items()):
            print(f"  {key}: {_fmt_optional(st.get('rate'), '{:.1f}')}% "
                  f"(n={st.get('n_valid', 0)})")



def _write_stats(out_dir: Path, summary: dict[str, Any]) -> list[Path]:
    """写出一份 summary 的两份同内容 JSON（06-stats-active / paper_stats_active）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    written: list[Path] = []
    for filename in STATS_FILENAMES:
        target = out_dir / filename
        target.write_text(payload, encoding="utf-8")
        written.append(target)
    return written




def _multi_mode_index(
    out_dir: Path, summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """构造 ``stats/modes.json`` 索引（只索引，不复制统计内容）。"""
    modes: dict[str, Any] = {}
    for mode, summary in summaries.items():
        validation = summary.get("validation") or {}
        unavailable = list(summary.get("unavailable") or [])
        records = int(validation.get("mode_records") or 0)
        modes[mode] = {
            "directory": mode,
            "files": [f"{mode}/{filename}" for filename in STATS_FILENAMES],
            # ok 只代表该模式的数据定位/唯一性通过，不要求所有实验节都有数据。
            "validation": {
                "ok": (
                    records > 0
                    and not validation.get("duplicates")
                    and not validation.get("physical_conflicts")
                ),
                "records": records,
                "results_records": int(validation.get("results_records") or 0),
                "runs_records": int(validation.get("runs_records") or 0),
                "manifest_missing": int(validation.get("manifest_missing") or 0),
                "duplicates": 0,
                "unavailable_sections": len(unavailable),
                "unavailable": unavailable,
            },
            "statistics": {
                "E3_cells": len((summary.get("E3_nominal") or {}).get("cells") or {}),
                "E8_did_tiers": sorted(
                    (summary.get("E8_limit_mechanism") or {}).get("did") or {}),
            },
        }
    first = next(iter(summaries.values())) if summaries else {}
    return {
        "schema": MULTI_MODE_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_root": first.get("data_root"),
        "catalog": first.get("catalog"),
        "modes": modes,
        "note": "本文件只是分模式产物的索引；统计内容以各模式目录下的 "
                f"{' / '.join(STATS_FILENAMES)} 为准",
    }




def _physical_conflict_summary(summaries: dict[str, dict[str, Any]]) -> tuple[int, list[str]]:
    """汇总各模式的富化冲突数与样例，供 CLI 以非零码明确拒绝。"""
    total = 0
    samples: list[str] = []
    for mode, summary in summaries.items():
        validation = summary.get("validation") or {}
        count = int(validation.get("physical_conflicts") or 0)
        if count <= 0:
            continue
        total += count
        for item in (validation.get("physical_conflict_samples") or [])[:3]:
            samples.append(
                f"{mode}:{item.get('experiment')}/{item.get('case')}/"
                f"{item.get('config_id')} seed={item.get('seed')} "
                f"fields={','.join(item.get('fields') or [])}"
            )
    return total, samples




def _reject_physical_conflicts(summaries: dict[str, dict[str, Any]]) -> int:
    """有富化冲突时打印明确错误并返回 2，避免冲突数据被当作正常结果。"""
    total, samples = _physical_conflict_summary(summaries)
    if total <= 0:
        return 0
    print(
        f"[paired_stats] 错误：runs.csv 与 results.csv 有 {total} 条身份匹配但字段冲突，"
        "已拒绝用 runs.csv 的 physical 判定富化统计。",
        file=sys.stderr,
    )
    for sample in samples:
        print(f"  冲突样例：{sample}", file=sys.stderr)
    print(
        "  请修复 runs.csv/config_json 中不一致的字段，或删除对应 runs 行后重跑。",
        file=sys.stderr,
    )
    return 2


__all__ = [
    "_catalog_metadata",
    "_existing_sources",
    "_fmt_ci",
    "_fmt_optional",
    "_fmt_p",
    "_fmt_pp",
    "_multi_mode_index",
    "_physical_conflict_summary",
    "_print_report",
    "_reason_text",
    "_reject_physical_conflicts",
    "_write_stats",
    "build_tables",
    "compute_summary",
    "default_output_dir",
]
