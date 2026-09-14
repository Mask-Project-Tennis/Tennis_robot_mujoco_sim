"""chat12 夜跑实验（exp23-27）统一分析：输出每个实验的关键结论数字。

用法（mujoco_tennis 环境）:
    python scripts/extract/analyze_chat12_night.py
输出: stdout 分节打印 5 个实验的命中率/配对差值/指标分布。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

DATA = REPO / "experiment_data"


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / name / "results.csv")
    # 解析 config_json 中的实验变量
    cfgs = df["config_json"].apply(json.loads)
    for key in ("--ablation", "--ball-speed", "--max-tcp", "--shared-input-full",
                "--no-freeze-first-plan", "--first-plan-iters"):
        if cfgs.iloc[0] is not None and key in cfgs.iloc[0]:
            df[key] = cfgs.apply(lambda c: c.get(key))
    # 排除 error 行
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]
    return df


def hit_rate(df: pd.DataFrame) -> float:
    return 100.0 * (df["hit_type"] == "active").mean()


def paired_diff(sub: pd.DataFrame, a: str, b: str) -> None:
    """同 seed 配对 McNemar + 差值（a 组 vs b 组）。"""
    ga = sub[sub["--ablation"] == a].set_index("seed")["hit_type"]
    gb = sub[sub["--ablation"] == b].set_index("seed")["hit_type"]
    common = ga.index.intersection(gb.index)
    if len(common) == 0:
        print(f"  {a} vs {b}: 无共同 seed")
        return
    ga, gb = ga.loc[common], gb.loc[common]
    n = len(common)
    ra, rb = 100.0 * (ga == "active").mean(), 100.0 * (gb == "active").mean()
    disc = int(((ga == "active") & (gb != "active")).sum())
    disc2 = int(((ga != "active") & (gb == "active")).sum())
    from scipy.stats import binomtest
    p = binomtest(min(disc, disc2), disc + disc2, 0.5).pvalue if disc + disc2 else 1.0
    print(f"  {a} vs {b}: {ra:.1f}% vs {rb:.1f}%  diff {ra-rb:+.1f} pp  "
          f"n={n}  不一致对 {disc}/{disc2}  McNemar p={p:.4f}")


def section(title: str) -> None:
    print(f"\n{'='*72}\n{title}\n{'='*72}")


# ───────────────────────── exp23 不冻结时间 ─────────────────────────
section("exp23_nofreeze — 首次规划期间球继续飞行（不冻结时间）")
df = load("exp23_nofreeze")
print("变量:", [c for c in df.columns if c.startswith("--")], "总行数", len(df))
print("\n首次规划耗时 first_plan_ms（有效行）:")
fp = df["first_plan_ms"].dropna()
print(f"  n={len(fp)}  median={fp.median():.0f} ms  p90={fp.quantile(0.9):.0f}  max={fp.max():.0f}")
# 默认行无 --first-plan-iters 键（=30 次下限）；iters5 变体显式传 5
df["_iters"] = df["config_json"].apply(
    lambda c: json.loads(c).get("--first-plan-iters", 30))
for it in sorted(df["_iters"].unique()):
    g = df[df["_iters"] == it]
    print(f"  first-plan-iters={it}: n={len(g)}  median={g['first_plan_ms'].median():.0f} ms")
for it in sorted(df["_iters"].unique()):
    for sp in sorted(df["--ball-speed"].unique()):
        for ab in ["full", "tube_only", "softmin_only", "none"]:
            sub = df[(df["_iters"] == it) & (df["--ball-speed"] == sp)
                     & (df["--ablation"] == ab)]
            if len(sub) == 0:
                continue
            print(f"  iters={it:2d} {sp} m/s {ab:12s}: {hit_rate(sub):5.1f}%  (n={len(sub)})")

# ───────────────────────── exp24 shared-input full ─────────────────────────
section("exp24_shared_full — 所有规划模块消费同一受扰球态")
df = load("exp24_shared_full")
cells = df["config_json"].apply(json.loads)
cells = cells.apply(lambda c: (str(c.get("--time-perturb-ms")), str(c.get("--space-perturb-m")),
                               c.get("--ball-speed")))
cellset = sorted(set(map(tuple, cells)), key=lambda x: (x[2], x[0]))
print("网格:", cellset)
for cell in cellset:
    sub = df[cells == cell]
    t, s, sp = cell
    print(f"\n[shared-full] t={t}ms s={s}m {sp}m/s (n={len(sub)//4} per tier):")
    for ab in ["full", "tube_only", "softmin_only", "none"]:
        g = sub[sub["--ablation"] == ab]
        print(f"    {ab:12s}: {hit_rate(g):5.1f}%")
    paired_diff(sub, "tube_only", "none")
    paired_diff(sub, "softmin_only", "none")
    paired_diff(sub, "full", "softmin_only")

# ───────────────────────── exp25 select_one 基线 ─────────────────────────
section("exp25_select_one — IK 裕度选单候选的 point-target 基线")
df = load("exp25_select_one")
for sp in sorted(df["--ball-speed"].unique()):
    sub = df[df["--ball-speed"] == sp]
    print(f"  {sp} m/s select_one: {hit_rate(sub):5.1f}% (n={len(sub)})")

# ───────────────────────── exp26 TCP 网格 ─────────────────────────
section("exp26_tcp_grid — 球速 × TCP 限速二维网格")
df = load("exp26_tcp_grid")
for sp in sorted(df["--ball-speed"].unique()):
    for tcp in sorted(df["--max-tcp"].unique()):
        sub = df[(df["--ball-speed"] == sp) & (df["--max-tcp"] == tcp)]
        if len(sub) == 0:
            continue
        rates = {ab: hit_rate(sub[sub["--ablation"] == ab])
                 for ab in ["full", "tube_only", "softmin_only", "none"]}
        line = "  ".join(f"{ab}={v:.1f}" for ab, v in rates.items())
        # 与 point-target 差值
        pt = rates["none"]
        print(f"  {sp} m/s TCP {tcp}: {line}  |  vs none: "
              f"full {rates['full']-pt:+.1f}, corridor {rates['tube_only']-pt:+.1f}, "
              f"softmin {rates['softmin_only']-pt:+.1f}")

# ───────────────────────── exp27 击球质量 ─────────────────────────
section("exp27_ball_quality — 出球速度与有效回击")
df = load("exp27_ball_quality")
hits = df[df["hit_type"] == "active"]
print(f"总 active hits: {len(hits)}/{len(df)}")
for sp in sorted(df["--ball-speed"].unique()):
    for ab in ["full", "tube_only", "softmin_only", "none"]:
        sub = hits[(hits["--ball-speed"] == sp) & (hits["--ablation"] == ab)]
        if len(sub) == 0:
            continue
        out_v = sub["ball_out_speed"]
        out_vy = sub["ball_out_vy"]
        # 有效回击：出球速度 > 1 m/s 且 vy < 0（朝回场方向，来球沿 +y）
        valid = (out_v > 1.0) & (out_vy < 0)
        print(f"  {sp} m/s {ab:12s}: n_hit={len(sub)}  出球速度 "
              f"median={out_v.median():.2f} p90={out_v.quantile(0.9):.2f} m/s  "
              f"vy<0 占比={100*valid.mean():.0f}%")
