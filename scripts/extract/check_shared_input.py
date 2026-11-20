#!/usr/bin/env python3
"""shared-input 交叉验证：空间偏置「只偏点目标」vs「偏整个球状态」两口径对照。

论文 §V-B「Shared-input check」段的数据来源。两口径的代码分流见
`src/ilqt/replan_core.py`（`MPCConfig.spatial_perturb_target`）：

- ``hitpoint``（正文主口径）：偏置只叠加到预测击球点 `p_hit_new`，候选窗口与
  走廊轴仍由未偏置的球观测重建 → 走 cost 的那几项拿到无偏几何参照。
- ``ballstate``（交叉验证口径，exp22_shared_spatial，51,000 runs）：偏置注入喂给
  规划器的球位置 `ball_pos_goal`，点目标 / 候选集合 / 走廊轴由同一份有偏输入生成；
  时序与可达性仍用当前观测。

本脚本输出的四个量即论文中该段与 Table III 的数字：
  1) 两个高功效角点的 corridor-only 独立价值（两口径对照）；
  2) 9 m/s t=0 幅度扫描 s=0.2/0.3/0.4 的口径差；
  3) 20 格合并网格上 softmin-only vs point-target 的时间窗增益（两口径对照）；
  4) 20 格合并网格上 full vs softmin-only 的额外走廊增益（两口径对照，chat12 表）。

用法::

    python scripts/extract/check_shared_input.py [--data-root DIR] [--output DIR]

- `--data-root` 指定实验数据根（默认 `experiment_data/`，旧行为不变）。
- `--output` 给出时写 `<output>/shared_input.json`；缺省只打印（旧行为）。
- `--dry-run` 只打印，不写文件。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT / "experiment_data"
sys.path.insert(0, str(PROJECT / "scripts" / "extract"))

from paired_stats import (  # noqa: E402
    HitPolicy,
    is_error,
    mcnemar_exact,
    paired_bootstrap,
    space_perturb_of,
    speed_of,
    tier_of,
    time_perturb_ms_of,
)

TS: tuple[int, ...] = (0, 10, 25, 50, 100)
SS: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)
TIERS: tuple[str, ...] = ("full", "tube_only", "softmin_only", "none")

# 主口径（hitpoint）网格数据；s=0 列与角点的主口径来源
GRID_SRC: tuple[str, ...] = ("exp17b_perturb", "exp17f_mechanism_perturb")
CORNERS_SRC: dict[tuple[int, int, float], str] = {
    (9, 0, 0.2): "exp17g_spatial_power",
    (12, 0, 0.2): "exp17h_extreme",
    (9, 50, 0.2): "exp17g_spatial_power",
    (9, 0, 0.3): "exp17h_extreme",
    (9, 0, 0.4): "exp17h_extreme",
}
SHARED_SRC = "exp22_shared_spatial"
HIT = HitPolicy()


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（--data-root / --output / --dry-run）。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=None,
                        help=f"实验数据根（默认 {DEFAULT_DATA_ROOT}）")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出目录（给出时写 <output>/shared_input.json）")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    return parser


def load(data_root: Path, name: str) -> list[dict[str, str]]:
    """读取一个实验目录的 results.csv（缺文件时抛 FileNotFoundError）。"""
    path = data_root / name / "results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cell_hits(
    rows: list[dict[str, str]], speed: int, t: int, s: float, tier: str
) -> dict[str, int]:
    """筛出一个格子的某档命中判定，返回 {seed: hit}。

    通过 paired_stats 的统一访问器识别 RM65 旧 flag 与 R1D `--task-overrides`。
    """
    out: dict[str, int] = {}
    for r in rows:
        if is_error(r):
            continue
        if speed_of(r) != speed:
            continue
        if time_perturb_ms_of(r) != t:
            continue
        if abs(space_perturb_of(r) - s) > 1e-9:
            continue
        if tier_of(r) != tier:
            continue
        out[str(r["seed"])] = HIT(r)
    return out


def paired(a: dict[str, int], b: dict[str, int]) -> dict[str, Any] | None:
    """同 seed 配对比较 a-b：命中率差（pp）+ 配对 bootstrap CI + McNemar 精确检验。"""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    ha = [a[k] for k in keys]
    hb = [b[k] for k in keys]
    n_a = sum(1 for x, y in zip(ha, hb) if x and not y)
    n_b = sum(1 for x, y in zip(ha, hb) if y and not x)
    lo, hi = paired_bootstrap(ha, hb)
    return {
        "n": len(keys),
        "diff": 100 * (sum(ha) - sum(hb)) / len(keys),
        "ci": (lo, hi),
        "p": mcnemar_exact(n_a, n_b),
        "rate_a": 100 * sum(ha) / len(keys),
        "rate_b": 100 * sum(hb) / len(keys),
    }


def fmt(d: dict[str, Any] | None) -> str:
    """把配对结果格式化成一行。"""
    if d is None:
        return "(无数据)"
    return (
        f"{d['diff']:+.2f} pp CI [{d['ci'][0]:+.1f},{d['ci'][1]:+.1f}] "
        f"p={d['p']:.2g} n={d['n']} ({d['rate_a']:.1f} vs {d['rate_b']:.1f})"
    )


def _merged_grid_pair(
    shared: list[dict[str, str]],
    grid_rows: list[dict[str, str]],
    tier_a: str,
    tier_b: str,
    swap: bool,
) -> dict[str, Any] | None:
    """20 格合并网格上 tier_a vs tier_b 的同 seed 配对。

    swap=True 时 s>0 的格子换用 ballstate 口径（exp22），并限定 seed<=100
    与主口径共用种子以保证配对。
    """
    pool: dict[tuple[int, float, str, str], int] = {}
    for t in TS:
        for s in SS:
            for tier in TIERS:
                src_rows = list(shared if (swap and s > 0.0) else grid_rows)
                hits = cell_hits(src_rows, 9, t, s, tier)
                if swap and s > 0.0:
                    hits = {k: v for k, v in hits.items() if int(k) <= 100}
                for seed, h in hits.items():
                    pool[(t, s, tier, seed)] = h
    idx = {(t, s, tier): {} for t in TS for s in SS for tier in TIERS}
    for (t, s, tier, seed), h in pool.items():
        idx[(t, s, tier)][seed] = h

    a: dict[str, int] = {}
    b: dict[str, int] = {}
    for t in TS:
        for s in SS:
            for seed, h in idx[(t, s, tier_a)].items():
                a[f"{t}|{s}|{seed}"] = h
            for seed, h in idx[(t, s, tier_b)].items():
                b[f"{t}|{s}|{seed}"] = h
    return paired(a, b)


def compute_shared_input(data_root: Path) -> dict[str, Any]:
    """计算四节对照与翻转数；缺失数据记为 unavailable，不伪造数值。"""
    unavailable: list[str] = []
    loaded: dict[str, list[dict[str, str]]] = {}
    for src in sorted({SHARED_SRC, *GRID_SRC, *CORNERS_SRC.values()}):
        try:
            loaded[src] = load(data_root, src)
        except FileNotFoundError as exc:
            unavailable.append(str(exc))
            loaded[src] = []
    shared = loaded[SHARED_SRC]
    grid_rows = [r for src in GRID_SRC for r in loaded[src]]
    old_rows = {src: loaded[src] for src in sorted(set(CORNERS_SRC.values()))}

    out: dict[str, Any] = {
        "data_root": str(data_root.resolve()),
        "hit_semantics": "active-hit（legacy 口径，passive 不算 active）",
        "sources": {src: {"rows": len(rows), "path": str(data_root / src / "results.csv")}
                    for src, rows in loaded.items()},
        "unavailable": unavailable,
    }
    if not shared:
        out["shared_rows"] = 0
        out["unavailable_reason"] = "missing_shared_input_csv"

    # 1) 角点 corridor-only 独立价值（tube_only vs none）
    corners: list[dict[str, Any]] = []
    for (speed, t, s), src in CORNERS_SRC.items():
        entry: dict[str, Any] = {"speed": speed, "t": t, "s": s, "source": src}
        for label, rows in (("hitpoint", old_rows[src]), ("ballstate", shared)):
            d = paired(cell_hits(rows, speed, t, s, "tube_only"),
                       cell_hits(rows, speed, t, s, "none"))
            if d is None:
                entry[label] = {"unavailable_reason": "no_common_seed"}
            else:
                entry[label] = d
        corners.append(entry)
    out["corners"] = corners

    # 2) 幅度扫描口径差（9 m/s, t=0）
    scan: list[dict[str, Any]] = []
    for s in (0.2, 0.3, 0.4):
        src = CORNERS_SRC[(9, 0, s)]
        d_old = paired(cell_hits(old_rows[src], 9, 0, s, "tube_only"),
                       cell_hits(old_rows[src], 9, 0, s, "none"))
        d_new = paired(cell_hits(shared, 9, 0, s, "tube_only"),
                       cell_hits(shared, 9, 0, s, "none"))
        entry = {"s": s, "hitpoint": d_old, "ballstate": d_new}
        if d_old is None or d_new is None:
            entry["unavailable_reason"] = "no_common_seed_or_missing_csv"
        else:
            entry["delta_pp"] = d_new["diff"] - d_old["diff"]
        scan.append(entry)
    out["amplitude_scan"] = scan

    # 3) 20 格合并网格的时间窗增益；4) 额外走廊增益
    gains: dict[str, Any] = {}
    for key, ta, tb in (("softmin_gain", "softmin_only", "none"),
                        ("corridor_gain", "full", "softmin_only")):
        section: dict[str, Any] = {}
        for label, swap in (("hitpoint", False), ("ballstate", True)):
            d = _merged_grid_pair(shared, grid_rows, ta, tb, swap)
            section[label] = d if d is not None else {"unavailable_reason": "no_common_seed"}
        gains[key] = section
    out["merged_grid"] = gains

    # 附带：口径引起的逐档 seed 判定翻转数
    flips: list[dict[str, Any]] = []
    for (speed, t, s), src in list(CORNERS_SRC.items())[:3]:
        rec: dict[str, Any] = {"speed": speed, "t": t, "s": s, "per_tier": {}}
        for tier in TIERS:
            a = cell_hits(shared, speed, t, s, tier)
            b = cell_hits(old_rows[src], speed, t, s, tier)
            keys = sorted(set(a) & set(b))
            rec["per_tier"][tier] = {"flips": sum(1 for k in keys if a[k] != b[k]),
                                      "n": len(keys)}
        flips.append(rec)
    out["corner_verdict_flips"] = flips
    out["paper_checks_text"] = (
        "corridor-only 角点 +4.3 / +7.0 pp；幅度扫描 |Δ| ≤ 0.4 pp；"
        "合并网格 +8.9 → +10.2 pp；full − softmin_only 网格 −0.3 / 0.0 pp")
    return out


def report(data: dict[str, Any]) -> None:
    """打印四节对照（保持原输出段落与措辞）。"""
    print(f"{SHARED_SRC}: {data['sources'].get(SHARED_SRC, {}).get('rows', 0)} 行")
    print("\n[1] 角点 corridor-only 独立价值（两口径，同 seed 配对）")
    for entry in data["corners"]:
        for label, key in (("hitpoint ", "hitpoint"), ("ballstate", "ballstate")):
            d = entry[key]
            shown = None if "unavailable_reason" in d else d
            print(f"  {entry['speed']:>2} m/s t={entry['t']:<3} "
                  f"s={entry['s']:.2f} [{label}]: {fmt(shown)}")
    print("\n[2] 9 m/s t=0 幅度扫描：口径差 Δ = ballstate − hitpoint（tube_only vs none）")
    for entry in data["amplitude_scan"]:
        d_old, d_new = entry["hitpoint"], entry["ballstate"]
        if d_old is None or d_new is None:
            print(f"  s={entry['s']:.2f}: (无数据)")
            continue
        print(f"  s={entry['s']:.2f}: hitpoint {d_old['diff']:+.2f} pp → "
              f"ballstate {d_new['diff']:+.2f} pp  (Δ {entry['delta_pp']:+.2f} pp, "
              f"p {d_old['p']:.2g} → {d_new['p']:.2g})")
    print("\n[3] 20 格合并网格 softmin-only vs point-target（时间窗增益）")
    for label in ("hitpoint", "ballstate"):
        d = data["merged_grid"]["softmin_gain"][label]
        print(f"  {label:<9}: {fmt(None if 'unavailable_reason' in d else d)}")
    print("\n[4] 20 格合并网格 full vs softmin-only（额外走廊增益，chat12 表）")
    for label in ("hitpoint", "ballstate"):
        d = data["merged_grid"]["corridor_gain"][label]
        print(f"  {label:<9}: {fmt(None if 'unavailable_reason' in d else d)}")
    print("\n[附] 角点逐档 seed 判定翻转数（ballstate vs hitpoint）")
    for rec in data["corner_verdict_flips"]:
        parts = " ".join(
            f"{tier}={v['flips']}/{v['n']}" for tier, v in rec["per_tier"].items())
        print(f"  {rec['speed']:>2} m/s t={rec['t']:<3} s={rec['s']:.2f}: {parts}")
    print(f"\n论文数字核对：{data['paper_checks_text']}")


def main(argv: list[str] | None = None) -> int:
    """主入口：计算、打印并按需写 JSON。"""
    args = build_parser().parse_args(argv)
    data_root: Path = args.data_root if args.data_root is not None else DEFAULT_DATA_ROOT
    data = compute_shared_input(data_root)
    if data["unavailable"]:
        print("不可用数据源:")
        for item in data["unavailable"]:
            print(f"  - {item}")
        if data.get("unavailable_reason"):
            print(f"  不可估计：{data['unavailable_reason']}")
    else:
        report(data)
    if args.output is not None:
        target = args.output / "shared_input.json"
        if args.dry_run:
            print(f"[shared_input] dry-run：不写文件（目标 {target}）")
        else:
            print(f"[shared_input] 写出目标 {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    elif not args.dry_run:
        print("[shared_input] 未给 --output：只打印，不写文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
