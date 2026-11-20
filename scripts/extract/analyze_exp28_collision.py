#!/usr/bin/env python3
"""exp28 碰撞窗口验证分析：常开碰撞 vs 窗口碰撞的裁决对照。

读取 exp28_always_collision（--always-collision 常开）与 exp24_shared_full
（窗口启用，sim_component 的 k_hit/距离判定）两个 CSV，按 (tier, cell, seed)
配对比较 hit 裁决，输出翻转清单与一致性统计。

T8b 起支持指定数据根与输出：

    python scripts/extract/analyze_exp28_collision.py [--data-root DIR] [--output DIR]

- `--data-root` 指定实验数据根（默认 `experiment_data/`，旧行为不变）。
- `--output` 给出时写 `<output>/exp28_collision.json`（新增结构化键）；缺省只打印。
- `--dry-run` 只打印，不写文件。

R1D 数据根下没有 exp28_always_collision（常开碰撞是 RM65 专用开关），本脚本
退化为「物理接触口径 vs 窗口口径」的逐 seed 对照（数据源 exp24_shared_full
的 runs.csv 富化行），并在输出中标记 `mode="exp24_physical_vs_legacy"`。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT / "experiment_data"

# 与 paired_stats 共用 RM65 / R1D 的统一配置访问器
from paired_stats import (  # noqa: E402
    cfg_of,
    is_error,
    load as load_paired_rows,
    overrides_of,
    space_perturb_of,
    tier_of,
    time_perturb_ms_of,
)

# 名义组（无扰动）的对照来源：none→exp17a 无噪声档，softmin/tube→exp17d，full→exp15 默认管线
NOMINAL_SOURCES: dict[str, tuple[str, Callable[[dict[str, Any]], bool]]] = {
    "none": ("exp17a_noise", lambda cfg: (
        cfg.get("--ablation") == "none"
        and "--obs-noise-pos" not in cfg
        and cfg.get("--ball-speed") == 9)),
    "softmin_only": ("exp17d_mechanism", lambda cfg: (
        cfg.get("--ablation") == "softmin_only" and cfg.get("--ball-speed") == 9)),
    "tube_only": ("exp17d_mechanism", lambda cfg: (
        cfg.get("--ablation") == "tube_only" and cfg.get("--ball-speed") == 9)),
    "full": ("exp15_speed_v2", lambda cfg: cfg.get("--ball-speed") == 9),
}

# 网格定义（与 overnight_experiments.py 的 _exp28_seed_pool 一致）
CELLS = [(0, 0.1), (0, 0.2), (0, 0.3), (25, 0.2), (50, 0.2)]
TIERS = ["none", "tube_only", "corridor", "full"]


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（--data-root / --output / --dry-run）。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=None,
                        help=f"实验数据根（默认 {DEFAULT_DATA_ROOT}）")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出目录（给出时写 <output>/exp28_collision.json）")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    return parser


def _cell(row: dict[str, Any]) -> tuple[int, float] | None:
    """从行配置提取 (t_ms, s_m) 单元格；仅显式扰动配置返回 cell。

    判定“扰动配置存在”的依据（缺省 0.0 不能作为扰动证据）：
    RM65 旧口径要求 ``--random-perturb`` 出现；R1D 没有该 flag，改看配置是否
    显式含 ``--space-perturb-m``，或 ``--task-overrides`` 是否显式含
    ``mpc.space_perturb_m``。``space_perturb_of`` 无扰动时返回 0.0，若用
    ``is not None`` 判断会把所有名义对照行误记为 (0, 0.0) 扰动格。
    """
    cfg = cfg_of(row)
    explicit_space = (
        "--space-perturb-m" in cfg
        or "mpc.space_perturb_m" in overrides_of(cfg)
    )
    if "--random-perturb" in cfg or explicit_space:
        return int(time_perturb_ms_of(row)), float(space_perturb_of(row) or 0.0)
    return None


def load(path: Path) -> dict[tuple[str, tuple[int, float] | None, str], dict[str, Any]]:
    """返回 {(tier, cell, seed): row}，跳过 error 行。"""
    out: dict[tuple[str, tuple[int, float] | None, str], dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            cfg = json.loads(row["config_json"])
            row["_cj"] = cfg
            key = (tier_of(row), _cell(row), row["seed"])
            out[key] = row
    return out


def _load_nominal(data_root: Path, tier: str) -> dict[str, dict[str, Any]]:
    """加载名义组（无扰动）对照：返回 {seed: row}。

    未声明的档位（例如常量 ``TIERS`` 里保留、但 exp28 数据未出现的
    ``corridor``）返回空表，由 :func:`compute_exp28` 计入 ``no_control``，
    不伪造对照，也避免因新分类路径把名义行路由到此处而崩溃。
    """
    source = NOMINAL_SOURCES.get(tier)
    if source is None:
        return {}
    ds, match = source
    out: dict[str, dict[str, Any]] = {}
    path = data_root / ds / "results.csv"
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            cfg = json.loads(row["config_json"])
            if match(cfg):
                row["_cj"] = cfg
                out[row["seed"]] = row
    return out


def _verdicts(row: dict[str, Any]) -> dict[str, Any]:
    """提取一行的 legacy 与 physical 两种裁决（缺字段为 None）。"""
    return {"hit": row.get("hit"), "hit_type": row.get("hit_type"),
            "physical_hit": row.get("physical_hit"),
            "physical_hit_type": row.get("physical_hit_type")}


def _collect_legacy(data_root: Path) -> tuple[dict, dict]:
    """旧口径：读 exp28 与 exp24 两个 CSV。"""
    e28 = load(data_root / "exp28_always_collision" / "results.csv")
    e24 = load(data_root / "exp24_shared_full" / "results.csv")
    return e28, e24


def _collect_physical_vs_legacy(data_root: Path) -> dict[str, Any]:
    """R1D 退化口径：exp24 每行的 physical 裁决 vs legacy 窗口裁决。"""
    try:
        rows = load_paired_rows(data_root, "exp24_shared_full")
    except FileNotFoundError:
        rows = []
    pairs: list[dict[str, Any]] = []
    n_same = n_flip = n_missing = 0
    for row in rows:
        if is_error(row) or row.get("error"):
            continue
        legacy = row.get("hit_type")
        phys_raw = row.get("physical_hit")
        phys = None
        if str(phys_raw).strip().lower() in {"true", "1"}:
            phys = True
        elif str(phys_raw).strip().lower() in {"false", "0"}:
            phys = False
        if not legacy or phys is None:
            n_missing += 1
            continue
        legacy_hit = legacy == "active"
        cell = _cell(row)
        entry = {"tier": tier_of(row), "cell": cell, "seed": row.get("seed"),
                 "legacy_hit_type": legacy, "legacy_hit": legacy_hit,
                 "physical_hit": phys, "physical_hit_type": row.get("physical_hit_type")}
        if legacy_hit == phys:
            n_same += 1
        else:
            n_flip += 1
            entry["flip"] = True
        pairs.append(entry)
    return {"mode": "exp24_physical_vs_legacy", "pairs": pairs, "n_total": len(pairs),
            "n_same": n_same, "n_flip": n_flip, "n_missing": n_missing}


def compute_exp28(data_root: Path) -> dict[str, Any]:
    """计算 exp28 对照；缺数据时返回 unavailable 而非伪造数值。"""
    unavailable: list[str] = []
    e28 = load(data_root / "exp28_always_collision" / "results.csv")
    e24 = load(data_root / "exp24_shared_full" / "results.csv")
    if not e28:
        unavailable.append(f"缺少 {data_root / 'exp28_always_collision' / 'results.csv'}"
                           " 或为空（R1D 无 --always-collision 开关）")
    if not e24:
        unavailable.append(f"缺少 {data_root / 'exp24_shared_full' / 'results.csv'} 或为空")
    if not e28:
        result = _collect_physical_vs_legacy(data_root)
        result["data_root"] = str(data_root.resolve())
        result["legacy_mode"] = "exp28_vs_exp24_csv"
        result["unavailable"] = unavailable
        result["note"] = ("R1D 退化：比较同一 exp24 行的窗口裁决与物理接触裁决，"
                          "不等价于 RM65 的常开碰撞对照")
        return result
    # ---- 旧口径：exp28 常开 vs exp24 窗口 ----
    nominal_cache: dict[str, dict[str, dict[str, Any]]] = {}
    n_total = n_same = n_flip = n_only28 = 0
    flips: list[dict[str, Any]] = []
    cells: dict[str, dict[str, Any]] = {}
    for (tier, cell, seed), row28 in sorted(
        e28.items(), key=lambda kv: (kv[0][0], kv[0][1] or (-1, -1.0), kv[0][2])
    ):
        n_total += 1
        if cell is None:
            nominal = nominal_cache.setdefault(
                tier, _load_nominal(data_root, tier))
            row24 = nominal.get(seed)
            label = f"nominal[{tier}]"
        else:
            row24 = e24.get((tier, cell, seed))
            label = f"exp24{tier}"
        if row24 is None:
            n_only28 += 1
            continue
        if row28["hit"] == row24["hit"]:
            n_same += 1
        else:
            n_flip += 1
            flips.append({"tier": tier, "cell": cell, "seed": seed,
                          "source": label, "exp24": _verdicts(row24),
                          "exp28": _verdicts(row28)})
    for cell in CELLS:
        sub = [(r, e24.get(k)) for k, r in e28.items() if k[1] == cell]
        same = sum(1 for r, r24 in sub if r24 and r["hit"] == r24["hit"])
        flip = sum(1 for r, r24 in sub if r24 and r["hit"] != r24["hit"])
        no = sum(1 for r, r24 in sub if r24 is None)
        cells[f"t{cell[0]}|s{cell[1]}"] = {"n": len(sub), "same": same,
                                            "flip": flip, "no_control": no}
    nom = [(k, r) for k, r in e28.items() if k[1] is None]
    same = flip_n = no = 0
    for (tier, _, seed), r28 in nom:
        r24 = nominal_cache.get(tier, _load_nominal(data_root, tier)).get(seed)
        if r24 is None:
            no += 1
        elif r28["hit"] == r24["hit"]:
            same += 1
        else:
            flip_n += 1
    result = {"mode": "exp28_always_vs_exp24_window",
              "data_root": str(data_root.resolve()),
              "n_total": n_total, "n_same": n_same, "n_flip": n_flip,
              "n_no_control": n_only28, "flips": flips, "cells": cells,
              "nominal": {"n": len(nom), "same": same, "flip": flip_n,
                          "no_control": no},
              "unavailable": unavailable}
    return result


def report(data: dict[str, Any]) -> None:
    """打印人类可读摘要（保持原输出措辞）。"""
    if data["mode"] == "exp24_physical_vs_legacy":
        print(f"模式: {data['mode']}（R1D 退化口径）")
        print(f"总配对: {data['n_total']} | 裁决一致: {data['n_same']} | "
              f"翻转: {data['n_flip']} | 缺判定: {data['n_missing']}")
        for p in data["pairs"]:
            if p.get("flip"):
                print(f"  FLIP: tier={p['tier']} cell={p['cell']} seed={p['seed']}: "
                      f"legacy={p['legacy_hit_type']} -> physical={p['physical_hit_type']}")
        return
    print(f"总配对: {data['n_total']} | 裁决一致: {data['n_same']} | "
          f"翻转: {data['n_flip']} | 无对照: {data['n_no_control']}")
    for f in data["flips"]:
        print(f"  FLIP: tier={f['tier']} cell={f['cell']} seed={f['seed']}: "
              f"{f['source']}={f['exp24']['hit']}({f['exp24']['hit_type']}) -> "
              f"exp28={f['exp28']['hit']}({f['exp28']['hit_type']})")
    print("\n按单元格（tier 汇总）:")
    for cell in CELLS:
        sub = data["cells"].get(f"t{cell[0]}|s{cell[1]}")
        if sub is None:
            continue
        print(f"  cell={cell}: n={sub['n']} 一致={sub['same']} "
              f"翻转={sub['flip']} 无对照={sub['no_control']}")
    nom = data["nominal"]
    print("\n名义组（无扰动）:")
    print(f"  n={nom['n']} 一致={nom['same']} 翻转={nom['flip']} 无对照={nom['no_control']}")


def main(argv: list[str] | None = None) -> int:
    """主入口：计算、打印并按需写 JSON。"""
    args = build_parser().parse_args(argv)
    data_root: Path = args.data_root if args.data_root is not None else DEFAULT_DATA_ROOT
    data = compute_exp28(data_root)
    if data.get("unavailable"):
        print("不可用数据:")
        for item in data["unavailable"]:
            print(f"  - {item}")
        if data["mode"] == "exp24_physical_vs_legacy" and data["n_total"] == 0:
            print("  不可估计：没有任何带物理判定与窗口判定的行")
    else:
        report(data)
    if args.output is not None:
        target = args.output / "exp28_collision.json"
        if args.dry_run:
            print(f"[exp28] dry-run：不写文件（目标 {target}）")
        else:
            print(f"[exp28] 写出目标 {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    elif not args.dry_run:
        print("[exp28] 未给 --output：只打印，不写文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
