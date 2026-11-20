"""chat12 夜跑实验（exp23-27）统一分析：输出每个实验的关键结论数字。

用法（mujoco_tennis 环境）:
    python scripts/extract/analyze_chat12_night.py [--data-root DIR] [--output DIR]

- `--data-root` 指定实验数据根（默认 `experiment_data/`，旧行为不变）。
- `--output` 给出时写 `<output>/chat12_night.json`（新增结构化键）；缺省只打印。
- `--dry-run` 只打印，不写文件。

输出: stdout 分节打印 5 个实验的命中率/配对差值/指标分布。

T8b 起：
* 模块导入不再执行分析（`main()` + `if __name__ == "__main__"` 守卫）；
* 实验变量通过 paired_stats 的统一访问器解析（兼容 RM65 旧 flag 与 R1D
  `--task-overrides`），缺失实验/无共同 seed 时显式标注不可估计而不崩溃；
* 只依赖标准库 + numpy（本环境未安装 pandas），CSV 以 `csv.DictReader` 读入。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = REPO / "experiment_data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paired_stats import (  # noqa: E402
    overrides_of,
    space_perturb_of,
    speed_of,
    tcp_label_of,
    tier_of,
    time_perturb_ms_of,
)

METRIC = "active-hit（legacy 口径，passive 不算 active）"

Row = dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（--data-root / --output / --dry-run）。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=None,
                        help=f"实验数据根（默认 {DEFAULT_DATA_ROOT}）")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出目录（给出时写 <output>/chat12_night.json）")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    return parser


def _first_plan_iters(cfg: dict[str, Any]) -> int:
    """解析首规划迭代数（RM65 旧 flag 优先，其次 R1D override，缺省 30）。"""
    raw = cfg.get("--first-plan-iters")
    if raw is not None:
        return int(raw)
    ov = overrides_of(cfg).get("mpc.first_plan_iters")
    if ov is not None:
        return int(ov)
    return 30


def _expand(row: dict[str, Any]) -> Row:
    """把一行 CSV 扩展成带统一实验变量键的行（_tier/_speed/_tcp/_t_ms/_s_m/_cfg）。"""
    try:
        cfg = json.loads(row.get("config_json") or "{}")
    except json.JSONDecodeError:
        cfg = {}
    out: Row = dict(row)
    out["_cfg"] = cfg
    out["_tier"] = tier_of(row)
    out["_speed"] = speed_of(row)
    out["_tcp"] = tcp_label_of(row)
    out["_t_ms"] = time_perturb_ms_of(row)
    out["_s_m"] = space_perturb_of(row)
    return out


def load(data_root: Path, name: str) -> list[Row] | None:
    """读取实验 CSV 并展开变量列；缺文件返回 None（调用方标注不可用）。"""
    path = data_root / name / "results.csv"
    if not path.is_file():
        return None
    rows: list[Row] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if (raw.get("error") or "").strip():
                continue
            rows.append(_expand(raw))
    return rows


def hit_rate(rows: list[Row]) -> float:
    """legacy 命中率（active 占比，%）。"""
    if not rows:
        return float("nan")
    return 100.0 * sum(1 for r in rows if r.get("hit_type") == "active") / len(rows)


def _active_map(rows: list[Row], tier: str) -> dict[int, bool]:
    """某档的 seed → 是否 active。"""
    return {int(r["seed"]): r.get("hit_type") == "active"
            for r in rows if r["_tier"] == tier}


def _mcnemar_p(disc: int, disc2: int) -> float:
    """精确 McNemar p 值（优先 scipy；不可用时用二项尾概率等价实现）。"""
    total = disc + disc2
    if total == 0:
        return 1.0
    success = min(disc, disc2)
    try:
        from scipy.stats import binomtest
        return float(binomtest(success, total, 0.5).pvalue)
    except ImportError:
        from math import comb
        tail = sum(comb(total, k) for k in range(success + 1))
        return float(min(1.0, 2.0 * tail / (2.0 ** total)))


def paired_diff(sub: list[Row], a: str, b: str) -> tuple | None:
    """同 seed 配对 McNemar + 差值（a 组 vs b 组）。

    Returns:
        (n, rate_a, rate_b, diff_pp, p, disc_a_hit_b_miss, disc_a_miss_b_hit)
    """
    ga = _active_map(sub, a)
    gb = _active_map(sub, b)
    common = sorted(ga.keys() & gb.keys())
    if not common:
        print(f"  {a} vs {b}: 无共同 seed")
        return None
    n = len(common)
    ra = 100.0 * sum(ga[k] for k in common) / n
    rb = 100.0 * sum(gb[k] for k in common) / n
    disc = sum(1 for k in common if ga[k] and not gb[k])
    disc2 = sum(1 for k in common if not ga[k] and gb[k])
    p = _mcnemar_p(disc, disc2)
    print(f"  {a} vs {b}: {ra:.1f}% vs {rb:.1f}%  diff {ra-rb:+.1f} pp  "
          f"n={n}  不一致对 {disc}/{disc2}  McNemar p={p:.4f}")
    return n, ra, rb, ra - rb, p, disc, disc2


def section(title: str) -> None:
    """打印分节标题。"""
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def _numbers(rows: list[Row], key: str) -> list[float]:
    """取某数值列的有限值（跳过空字符串/非数）。"""
    vals: list[float] = []
    for r in rows:
        raw = (r.get(key) or "").strip()
        if not raw:
            continue
        try:
            vals.append(float(raw))
        except ValueError:
            continue
    return vals


def _sec_exp23(rows: list[Row] | None, out: dict[str, Any]) -> None:
    """exp23 不冻结时间：首次规划耗时与命中率。"""
    section("exp23_nofreeze — 首次规划期间球继续飞行（不冻结时间）")
    key = "exp23_nofreeze"
    if rows is None:
        print("  不可估计：缺少 exp23_nofreeze/results.csv")
        out[key] = {"unavailable_reason": "missing_csv"}
        return
    rec: dict[str, Any] = {"rows": len(rows), "metric": METRIC}
    print("总行数", len(rows))
    print("\n首次规划耗时 first_plan_ms（有效行）:")
    fp = _numbers(rows, "first_plan_ms")
    if not fp:
        print("  不可估计：无 first_plan_ms 数值")
        rec["first_plan_ms"] = {"n": 0, "unavailable_reason": "no_first_plan_ms"}
        out[key] = rec
        return
    rec["first_plan_ms"] = {"n": len(fp), "median": float(np.median(fp)),
                            "p90": float(np.quantile(fp, 0.9)), "max": float(max(fp))}
    print(f"  n={len(fp)}  median={np.median(fp):.0f} ms  "
          f"p90={np.quantile(fp, 0.9):.0f}  max={max(fp):.0f}")
    for r in rows:
        r["_iters"] = _first_plan_iters(r["_cfg"])
    by_iters: list[dict[str, Any]] = []
    iters_values = sorted({r["_iters"] for r in rows})
    for it in iters_values:
        group = [r for r in rows if r["_iters"] == it]
        vals = _numbers(group, "first_plan_ms")
        by_iters.append({"first_plan_iters": int(it), "n": len(group),
                         "median_ms": float(np.median(vals)) if vals else None,
                         "pooled_pct": float(hit_rate(group))})
        med = f"{np.median(vals):.0f}" if vals else "--"
        print(f"  first-plan-iters={it}: n={len(group)}  "
              f"median={med} ms  pooled={hit_rate(group):.2f}%")
    cells: list[dict[str, Any]] = []
    speeds = sorted({r["_speed"] for r in rows})
    for it in iters_values:
        for sp in speeds:
            for ab in ["full", "tube_only", "softmin_only", "none"]:
                sub = [r for r in rows if r["_iters"] == it
                       and r["_speed"] == sp and r["_tier"] == ab]
                if not sub:
                    continue
                cells.append({"first_plan_iters": int(it), "speed": int(sp),
                              "tier": ab, "n": len(sub),
                              "rate_pct": float(hit_rate(sub))})
                print(f"  iters={it:2d} {sp} m/s {ab:12s}: {hit_rate(sub):5.1f}%  "
                      f"(n={len(sub)})")
    rec["by_first_plan_iters"] = by_iters
    rec["cells"] = cells
    out[key] = rec


def _sec_exp24(rows: list[Row] | None, out: dict[str, Any]) -> None:
    """exp24 shared-input full：网格命中率、配对差与 Holm 校正。"""
    section("exp24_shared_full — 所有规划模块消费同一受扰球态")
    key = "exp24_shared_full"
    if rows is None:
        print("  不可估计：缺少 exp24_shared_full/results.csv")
        out[key] = {"unavailable_reason": "missing_csv"}
        return
    cellset = sorted({(float(r["_t_ms"]), float(r["_s_m"]), int(r["_speed"]))
                      for r in rows}, key=lambda x: (x[2], x[0]))
    rec: dict[str, Any] = {"metric": METRIC,
                           "cells": [{"t_ms": t, "s_m": s, "speed": sp}
                                     for t, s, sp in cellset]}
    print("网格:", cellset)
    corr_vs_pt: list[tuple] = []
    win_diffs: list[float] = []
    cell_records: list[dict[str, Any]] = []
    for t, s, sp in cellset:
        sub = [r for r in rows if r["_t_ms"] == t and r["_s_m"] == s
               and r["_speed"] == sp]
        print(f"\n[shared-full] t={t:g}ms s={s:g}m {sp}m/s (n={len(sub)//4} per tier):")
        cell_rec: dict[str, Any] = {"t_ms": t, "s_m": s, "speed": sp, "tiers": {}}
        for ab in ["full", "tube_only", "softmin_only", "none"]:
            g = [r for r in sub if r["_tier"] == ab]
            if not g:
                cell_rec["tiers"][ab] = {"unavailable_reason": "no_rows"}
                print(f"    {ab:12s}: 不可估计（无行）")
                continue
            cell_rec["tiers"][ab] = {"n": len(g), "rate_pct": float(hit_rate(g))}
            print(f"    {ab:12s}: {hit_rate(g):5.1f}%")
        r_tp = paired_diff(sub, "tube_only", "none")
        r_sp = paired_diff(sub, "softmin_only", "none")
        r_ft = paired_diff(sub, "full", "tube_only")
        paired_diff(sub, "full", "softmin_only")
        for name, r in (("tube_only_vs_none", r_tp), ("softmin_only_vs_none", r_sp),
                        ("full_vs_tube_only", r_ft)):
            cell_rec[name] = None if r is None else {
                "n": r[0], "rate_a": r[1], "rate_b": r[2], "diff_pp": r[3],
                "p_mcnemar": r[4], "discordant": {"a_hit_b_miss": r[5],
                                                  "a_miss_b_hit": r[6]}}
        cell_records.append(cell_rec)
        if r_tp is not None:
            corr_vs_pt.append((float(t), float(s), *r_tp))
        if r_sp is not None:
            win_diffs.append(r_sp[3])
    rec["cell_results"] = cell_records
    if win_diffs:
        print("\n候选窗口各格差值 (softmin_only - none):",
              [f"{d:+.1f}" for d in win_diffs],
              f"  合并 {sum(win_diffs)/len(win_diffs):+.2f} pp")
        rec["softmin_vs_none_merged_pp"] = float(sum(win_diffs) / len(win_diffs))
    print("\n走廊-vs-点目标 5 格 Holm 校正（检验族 = 5 个走廊独立值检验）:")
    ordered = sorted(corr_vs_pt, key=lambda x: x[6])
    m = len(ordered)
    adj: list[float] = []
    for rank, row in enumerate(ordered, 1):
        p_adj = min(row[6] * (m - rank + 1), 1.0)
        if adj:
            p_adj = max(p_adj, adj[-1])
        adj.append(p_adj)
    holm: list[dict[str, Any]] = []
    for row, p_adj in zip(ordered, adj):
        t, s, n, ra, rb, diff, p, d1, d2 = row
        holm.append({"t_ms": t, "s_m": s, "n": n, "rate_a": ra, "rate_b": rb,
                     "diff_pp": diff, "p_mcnemar": p, "p_holm": p_adj,
                     "discordant": {"a_hit_b_miss": d1, "a_miss_b_hit": d2}})
        print(f"  (t={t:g},s={s:g}) {ra:.1f}% vs {rb:.1f}%  diff {diff:+.1f} pp  "
              f"n={n}  不一致对 {d1}/{d2}  p={p:.4f}  Holm={p_adj:.4f}")
    rec["holm_corridor_vs_point"] = holm
    out[key] = rec


def _sec_exp25(rows: list[Row] | None, soft_rows: list[Row] | None,
               pt_rows: list[Row] | None, out: dict[str, Any]) -> None:
    """exp25 select_one 与名义档的配对对照。"""
    section("exp25_select_one — IK 裕度选单候选的 point-target 基线")
    key = "exp25_select_one"
    if rows is None:
        print("  不可估计：缺少 exp25_select_one/results.csv（R1D 无此消融开关）")
        out[key] = {"unavailable_reason": "missing_csv_or_unsupported"}
        return

    def nominal_series(src: list[Row] | None, tier: str, sp: float) -> dict[int, bool] | None:
        """名义档（无噪声/无扰动）同 seed active 映射；缺数据返回 None。"""
        if src is None:
            return None
        out_map: dict[int, bool] = {}
        for r in src:
            if r["_tier"] != tier or float(r["_speed"]) != float(sp):
                continue
            cfg = r["_cfg"]
            noisy = bool(cfg.get("--obs-noise-pos") or cfg.get("--obs-noise-vel")
                         or "--obs-use-kf" in cfg)
            if noisy:
                continue
            out_map[int(r["seed"])] = r.get("hit_type") == "active"
        return out_map

    rec: dict[str, Any] = {"metric": METRIC, "cells": []}
    for sp in sorted({r["_speed"] for r in rows}):
        sub = [r for r in rows if r["_speed"] == sp]
        print(f"\n  {sp} m/s select_one 整体: {hit_rate(sub):5.1f}% (n={len(sub)})")
        for tier, src, label in (("softmin_only", soft_rows, "softmin 名义"),
                                 ("none", pt_rows, "point-target 名义")):
            gb = nominal_series(src, tier, sp)
            if gb is None:
                print(f"    select_one vs {label} ({tier}): 不可估计（缺名义档数据）")
                rec["cells"].append({"speed": int(sp), "tier": tier,
                                     "unavailable_reason": "missing_nominal_rows"})
                continue
            ga = _active_map(sub, "select_one")
            if not ga:  # 兜底：无 select_one 档名时用全部行
                ga = {int(r["seed"]): r.get("hit_type") == "active" for r in sub}
            common = sorted(ga.keys() & gb.keys())
            if not common:
                print(f"    select_one vs {label} ({tier}): 无共同 seed")
                rec["cells"].append({"speed": int(sp), "tier": tier,
                                     "unavailable_reason": "no_common_seed"})
                continue
            n = len(common)
            ra = 100.0 * sum(ga[k] for k in common) / n
            rb = 100.0 * sum(gb[k] for k in common) / n
            disc = sum(1 for k in common if ga[k] and not gb[k])
            disc2 = sum(1 for k in common if not ga[k] and gb[k])
            p = _mcnemar_p(disc, disc2)
            print(f"    select_one vs {label} ({tier}): 配对 n={n}  "
                  f"select={ra:.1f}% vs {label}={rb:.1f}%  "
                  f"diff({label}−select) {rb-ra:+.1f} pp  "
                  f"不一致对 {disc}/{disc2}  McNemar p={p:.5f}")
            rec["cells"].append({"speed": int(sp), "tier": tier, "n": n,
                                 "select_pct": float(ra), "nominal_pct": float(rb),
                                 "diff_pp": float(rb - ra), "p_mcnemar": p,
                                 "discordant": {"a_hit_b_miss": disc,
                                                "a_miss_b_hit": disc2}})
    out[key] = rec


def _sec_exp26(rows: list[Row] | None, out: dict[str, Any]) -> None:
    """exp26 TCP 网格：球速 × TCP 限速二维命中率与配对 CI。"""
    section("exp26_tcp_grid — 球速 × TCP 限速二维网格")
    key = "exp26_tcp_grid"
    if rows is None:
        print("  不可估计：缺少 exp26_tcp_grid/results.csv")
        out[key] = {"unavailable_reason": "missing_csv"}
        return

    def paired_boot_ci(sub: list[Row], a: str, b: str,
                       n_boot: int = 20000) -> tuple[float, float] | None:
        """同 seed 配对 bootstrap 95% CI（a − b，pp）。"""
        ga = _active_map(sub, a)
        gb = _active_map(sub, b)
        common = sorted(ga.keys() & gb.keys())
        if not common:
            return None
        ha = np.array([int(ga[k]) for k in common])
        hb = np.array([int(gb[k]) for k in common])
        rng = np.random.default_rng(0)
        n = len(common)
        idx = rng.integers(0, n, size=(n_boot, n))
        diffs = np.sort((ha[idx] - hb[idx]).sum(axis=1)) * 100.0 / n
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return float(lo), float(hi)

    cells: list[dict[str, Any]] = []
    for sp in sorted({r["_speed"] for r in rows}):
        for tcp in sorted({r["_tcp"] for r in rows}):
            sub = [r for r in rows if r["_speed"] == sp and r["_tcp"] == tcp]
            if not sub:
                continue
            rates = {ab: hit_rate([r for r in sub if r["_tier"] == ab])
                     for ab in ["full", "tube_only", "softmin_only", "none"]}
            line = "  ".join(f"{ab}={v:.1f}" for ab, v in rates.items())
            pt = rates["none"]
            cis: dict[str, Any] = {}
            rec: dict[str, Any] = {"speed": int(sp), "tcp": tcp,
                                   "tiers": {k: float(v) for k, v in rates.items()},
                                   "vs_none": {}}
            for t in ("full", "tube_only", "softmin_only"):
                ci = paired_boot_ci(sub, t, "none")
                cis[t] = "(无共同 seed)" if ci is None else f"{ci[0]:+.1f}~{ci[1]:+.1f}"
                rec["vs_none"][t] = {"diff_pp": float(rates[t] - pt),
                                     "ci95_pp": None if ci is None else [ci[0], ci[1]]}
            print(f"  {sp} m/s TCP {tcp}: {line}  |  vs none: "
                  f"full {rates['full']-pt:+.1f} ({cis['full']}), "
                  f"corridor {rates['tube_only']-pt:+.1f} ({cis['tube_only']}), "
                  f"softmin {rates['softmin_only']-pt:+.1f} ({cis['softmin_only']})")
            cells.append(rec)
    out[key] = {"metric": METRIC, "cells": cells}


def _sec_exp27(rows: list[Row] | None, out: dict[str, Any]) -> None:
    """exp27 击球质量：出球速度、vy 方向占比。"""
    section("exp27_ball_quality — 出球速度与有效回击")
    key = "exp27_ball_quality"
    if rows is None:
        print("  不可估计：缺少 exp27_ball_quality/results.csv")
        out[key] = {"unavailable_reason": "missing_csv"}
        return
    hits = [r for r in rows if r.get("hit_type") == "active"]
    print(f"总 active hits: {len(hits)}/{len(rows)}")
    rec: dict[str, Any] = {"metric": METRIC, "total_rows": len(rows),
                           "total_active": len(hits), "cells": []}
    if not hits:
        print("  不可估计：无 active hit 行（无可统计质量指标）")
        rec["unavailable_reason"] = "all_miss"
        out[key] = rec
        return
    for sp in sorted({r["_speed"] for r in rows}):
        for ab in ["full", "tube_only", "softmin_only", "none"]:
            sub = [r for r in hits if r["_speed"] == sp and r["_tier"] == ab]
            if not sub:
                continue
            out_v = _numbers(sub, "ball_out_speed")
            out_vy = _numbers(sub, "ball_out_vy")
            if not out_v or not out_vy:
                print(f"  {sp} m/s {ab:12s}: n_hit={len(sub)}  不可估计（缺出球指标）")
                rec["cells"].append({"speed": int(sp), "tier": ab,
                                     "n_hit": len(sub),
                                     "unavailable_reason": "missing_ball_out_metrics"})
                continue
            valid = float(np.mean([(v > 1.0 and vy < 0) for v, vy in zip(out_v, out_vy)]))
            print(f"  {sp} m/s {ab:12s}: n_hit={len(sub)}  出球速度 "
                  f"median={np.median(out_v):.2f} p90={np.quantile(out_v, 0.9):.2f} m/s  "
                  f"vy<0 占比={100*valid:.0f}%")
            rec["cells"].append({"speed": int(sp), "tier": ab, "n_hit": len(sub),
                                 "median_ball_out_speed": float(np.median(out_v)),
                                 "p90_ball_out_speed": float(np.quantile(out_v, 0.9)),
                                 "vy_negative_share_pct": float(100 * valid)})
    out[key] = rec


def compute_chat12(data_root: Path) -> dict[str, Any]:
    """计算五个实验的分析结果（缺数据显式 unavailable）。"""
    unavailable: list[str] = []
    frames: dict[str, list[Row] | None] = {}
    for name in ("exp23_nofreeze", "exp24_shared_full", "exp25_select_one",
                 "exp17d_mechanism", "exp17a_noise", "exp26_tcp_grid",
                 "exp27_ball_quality"):
        frame = load(data_root, name)
        if frame is None:
            unavailable.append(f"缺少 {data_root / name / 'results.csv'}")
        frames[name] = frame
    out: dict[str, Any] = {"data_root": str(data_root.resolve()), "metric": METRIC}
    _sec_exp23(frames["exp23_nofreeze"], out)
    _sec_exp24(frames["exp24_shared_full"], out)
    _sec_exp25(frames["exp25_select_one"], frames["exp17d_mechanism"],
               frames["exp17a_noise"], out)
    _sec_exp26(frames["exp26_tcp_grid"], out)
    _sec_exp27(frames["exp27_ball_quality"], out)
    out["unavailable"] = unavailable
    return out


def main(argv: list[str] | None = None) -> int:
    """主入口：计算、打印并按需写 JSON。"""
    args = build_parser().parse_args(argv)
    data_root: Path = args.data_root if args.data_root is not None else DEFAULT_DATA_ROOT
    data = compute_chat12(data_root)
    if data["unavailable"]:
        print("\n不可用数据:")
        for item in data["unavailable"]:
            print(f"  - {item}")
    if args.output is not None:
        target = args.output / "chat12_night.json"
        if args.dry_run:
            print(f"[chat12] dry-run：不写文件（目标 {target}）")
        else:
            print(f"[chat12] 写出目标 {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    elif not args.dry_run:
        print("[chat12] 未给 --output：只打印，不写文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
