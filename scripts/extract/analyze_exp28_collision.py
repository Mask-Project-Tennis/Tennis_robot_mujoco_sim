#!/usr/bin/env python3
"""exp28 碰撞窗口验证分析：常开碰撞 vs 窗口碰撞的裁决对照。

读取 exp28_always_collision（--always-collision 常开）与 exp24_shared_full
（窗口启用，sim_component 的 k_hit/距离判定）两个 CSV，按 (tier, cell, seed)
配对比较 hit 裁决，输出翻转清单与一致性统计。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "experiment_data"
EXP28 = DATA / "exp28_always_collision" / "results.csv"
EXP24 = DATA / "exp24_shared_full" / "results.csv"

# 名义组（无扰动）的对照来源：none→exp17a 无噪声档，softmin/tube→exp17d，full→exp15 默认管线
NOMINAL_SOURCES = {
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


def _cell(cfg: dict) -> tuple[int, float] | None:
    """从 config_json 提取 (t_ms, s_m) 单元格；无扰动配置返回 None。"""
    if "--random-perturb" not in cfg:
        return None
    return int(cfg.get("--time-perturb-ms", 0)), float(cfg["--space-perturb-m"])


def load(path: Path) -> dict[tuple[str, tuple[int, float] | None, str], dict]:
    """返回 {(tier, cell, seed): row}，跳过 error 行。"""
    out: dict[tuple[str, tuple[int, float] | None, str], dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            cfg = json.loads(row["config_json"])
            tier = cfg["--ablation"]
            key = (tier, _cell(cfg), row["seed"])
            out[key] = row
    return out


def _load_nominal(tier: str) -> dict[str, dict]:
    """加载名义组（无扰动）对照：返回 {seed: row}。"""
    ds, match = NOMINAL_SOURCES[tier]
    out: dict[str, dict] = {}
    with (DATA / ds / "results.csv").open() as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            cfg = json.loads(row["config_json"])
            if match(cfg):
                out[row["seed"]] = row
    return out


def main() -> None:
    """主入口：配对比较并打印裁决一致性与翻转清单。"""
    e28 = load(EXP28)
    e24 = load(EXP24)
    nominal_cache: dict[str, dict[str, dict]] = {}
    n_total = n_same = n_flip = n_only28 = 0
    flips: list[str] = []
    for (tier, cell, seed), row28 in sorted(
        e28.items(), key=lambda kv: (kv[0][0], kv[0][1] or (-1, -1.0), kv[0][2])
    ):
        n_total += 1
        if cell is None:
            nominal = nominal_cache.setdefault(tier, _load_nominal(tier))
            row24 = nominal.get(seed)
            label = f"nominal[{tier}]"
        else:
            row24 = e24.get((tier, cell, seed))
            label = f"exp24{tier}"
        if row24 is None:
            n_only28 += 1
            print(f"[缺对照] tier={tier} cell={cell} seed={seed} hit28={row28['hit']}")
            continue
        if row28["hit"] == row24["hit"]:
            n_same += 1
        else:
            n_flip += 1
            flips.append(
                f"tier={tier} cell={cell} seed={seed}: "
                f"{label}={row24['hit']}({row24['hit_type']}) -> "
                f"exp28={row28['hit']}({row28['hit_type']})"
            )
    print(f"总配对: {n_total} | 裁决一致: {n_same} | 翻转: {n_flip} | 无对照: {n_only28}")
    for line in flips:
        print("  FLIP:", line)
    # 按单元格汇总
    print("\n按单元格（tier 汇总）:")
    for cell in CELLS:
        sub = [(r, e24.get(k)) for k, r in e28.items() if k[1] == cell]
        same = sum(1 for r, r24 in sub if r24 and r["hit"] == r24["hit"])
        flip = sum(1 for r, r24 in sub if r24 and r["hit"] != r24["hit"])
        no = sum(1 for r, r24 in sub if r24 is None)
        print(f"  cell={cell}: n={len(sub)} 一致={same} 翻转={flip} 无对照={no}")
    # 名义组汇总
    print("\n名义组（无扰动）:")
    nom = [(k, r) for k, r in e28.items() if k[1] is None]
    same = flip = no = 0
    for (tier, _, seed), r28 in nom:
        r24 = nominal_cache.get(tier, _load_nominal(tier)).get(seed)
        if r24 is None:
            no += 1
        elif r28["hit"] == r24["hit"]:
            same += 1
        else:
            flip += 1
    print(f"  n={len(nom)} 一致={same} 翻转={flip} 无对照={no}")


if __name__ == "__main__":
    main()
