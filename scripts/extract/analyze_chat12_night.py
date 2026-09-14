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


def paired_diff(sub: pd.DataFrame, a: str, b: str) -> tuple | None:
    """同 seed 配对 McNemar + 差值（a 组 vs b 组）。

    Returns:
        (n, rate_a, rate_b, diff_pp, p, disc_a_hit_b_miss, disc_a_miss_b_hit)
    """
    ga = sub[sub["--ablation"] == a].set_index("seed")["hit_type"]
    gb = sub[sub["--ablation"] == b].set_index("seed")["hit_type"]
    common = ga.index.intersection(gb.index)
    if len(common) == 0:
        print(f"  {a} vs {b}: 无共同 seed")
        return None
    ga, gb = ga.loc[common], gb.loc[common]
    n = len(common)
    ra, rb = 100.0 * (ga == "active").mean(), 100.0 * (gb == "active").mean()
    disc = int(((ga == "active") & (gb != "active")).sum())
    disc2 = int(((ga != "active") & (gb == "active")).sum())
    from scipy.stats import binomtest
    p = binomtest(min(disc, disc2), disc + disc2, 0.5).pvalue if disc + disc2 else 1.0
    print(f"  {a} vs {b}: {ra:.1f}% vs {rb:.1f}%  diff {ra-rb:+.1f} pp  "
          f"n={n}  不一致对 {disc}/{disc2}  McNemar p={p:.4f}")
    return n, ra, rb, ra - rb, p, disc, disc2


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
    print(f"  first-plan-iters={it}: n={len(g)}  median={g['first_plan_ms'].median():.0f} ms  "
          f"pooled={hit_rate(g):.2f}%")
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
corr_vs_pt: list[tuple] = []   # 走廊独立值 5 格检验族（Holm 校正用）
win_diffs: list[float] = []    # 候选窗口 vs 点目标各格差值（+5.1 pp 口径核实）
for cell in cellset:
    sub = df[cells == cell]
    t, s, sp = cell
    print(f"\n[shared-full] t={t}ms s={s}m {sp}m/s (n={len(sub)//4} per tier):")
    for ab in ["full", "tube_only", "softmin_only", "none"]:
        g = sub[sub["--ablation"] == ab]
        print(f"    {ab:12s}: {hit_rate(g):5.1f}%")
    r_tp = paired_diff(sub, "tube_only", "none")      # 走廊独立值（正文 p=0.013/0.012）
    r_sp = paired_diff(sub, "softmin_only", "none")   # 候选窗口收益（正文 +5.1 pp 口径）
    r_ft = paired_diff(sub, "full", "tube_only")      # chat14 关键对照：full 是否保留走廊优势
    paired_diff(sub, "full", "softmin_only")
    if r_tp is not None:
        corr_vs_pt.append((0.0 if t == "None" else float(t), float(s), *r_tp))
    if r_sp is not None:
        win_diffs.append(r_sp[3])
print("\n候选窗口各格差值 (softmin_only - none):", [f"{d:+.1f}" for d in win_diffs],
      f"  合并 {sum(win_diffs)/len(win_diffs):+.2f} pp")
print("\n走廊-vs-点目标 5 格 Holm 校正（检验族 = 5 个走廊独立值检验）:")
ordered = sorted(corr_vs_pt, key=lambda x: x[6])
m = len(ordered)
adj: list[float] = []
for rank, row in enumerate(ordered, 1):
    p_adj = min(row[6] * (m - rank + 1), 1.0)
    if adj:
        p_adj = max(p_adj, adj[-1])   # Holm 单调性
    adj.append(p_adj)
for row, p_adj in zip(ordered, adj):
    t, s, n, ra, rb, diff, p, d1, d2 = row
    print(f"  (t={t:g},s={s:g}) {ra:.1f}% vs {rb:.1f}%  diff {diff:+.1f} pp  "
          f"n={n}  不一致对 {d1}/{d2}  p={p:.4f}  Holm={p_adj:.4f}")

# ───────────────────────── exp25 select_one 基线 ─────────────────────────
section("exp25_select_one — IK 裕度选单候选的 point-target 基线")
df = load("exp25_select_one")
# 配对侧：名义四档来自 exp17a_noise（full/none 基线格）+ exp17d_mechanism
# （tube_only/softmin_only），同 seed 可跨实验配对（serve-box RNG 同源）。
soft_rows = load("exp17d_mechanism")
pt_rows = load("exp17a_noise")


def nominal_series(rows: pd.DataFrame, tier: str, sp: float) -> pd.Series:
    """名义档（无噪声/无扰动）同 seed 命中序列。"""
    cfs = rows["config_json"].apply(json.loads)
    mask = (
        (rows["--ablation"] == tier)
        & (rows["--ball-speed"] == sp)
        & ~cfs.apply(lambda c: bool(c.get("--obs-noise-pos") or c.get("--obs-noise-vel")
                                    or c.get("--obs-use-kf")))
    )
    return rows.loc[mask].set_index("seed")["hit_type"]


for sp in sorted(df["--ball-speed"].unique()):
    sub = df[df["--ball-speed"] == sp]
    print(f"\n  {sp} m/s select_one 整体: {hit_rate(sub):5.1f}% (n={len(sub)})")
    for tier, src, label in (("softmin_only", soft_rows, "softmin 名义"),
                             ("none", pt_rows, "point-target 名义")):
        ga = sub.set_index("seed")["hit_type"]
        gb = nominal_series(src, tier, sp)
        common = ga.index.intersection(gb.index)
        n = len(common)
        ra = 100.0 * (ga.loc[common] == "active").mean()
        rb = 100.0 * (gb.loc[common] == "active").mean()
        disc = int(((ga.loc[common] == "active") & (gb.loc[common] != "active")).sum())
        disc2 = int(((ga.loc[common] != "active") & (gb.loc[common] == "active")).sum())
        from scipy.stats import binomtest
        p = binomtest(min(disc, disc2), disc + disc2, 0.5).pvalue if disc + disc2 else 1.0
        print(f"    select_one vs {label} ({tier}): 配对 n={n}  "
              f"select={ra:.1f}% vs {label}={rb:.1f}%  diff({label}−select) {rb-ra:+.1f} pp  "
              f"不一致对 {disc}/{disc2}  McNemar p={p:.5f}")

# ───────────────────────── exp26 TCP 网格 ─────────────────────────
section("exp26_tcp_grid — 球速 × TCP 限速二维网格")
df = load("exp26_tcp_grid")


def paired_boot_ci(sub: pd.DataFrame, a: str, b: str, n_boot: int = 20000) -> str:
    """同 seed 配对 bootstrap 95% CI（a − b，pp）。"""
    ga = sub[sub["--ablation"] == a].set_index("seed")["hit_type"]
    gb = sub[sub["--ablation"] == b].set_index("seed")["hit_type"]
    common = ga.index.intersection(gb.index)
    ha = (ga.loc[common] == "active").astype(int).to_numpy()
    hb = (gb.loc[common] == "active").astype(int).to_numpy()
    rng = np.random.default_rng(0)
    n = len(common)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = np.sort((ha[idx] - hb[idx]).sum(axis=1)) * 100.0 / n
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return f"{lo:+.1f}~{hi:+.1f}"


for sp in sorted(df["--ball-speed"].unique()):
    for tcp in sorted(df["--max-tcp"].unique()):
        sub = df[(df["--ball-speed"] == sp) & (df["--max-tcp"] == tcp)]
        if len(sub) == 0:
            continue
        rates = {ab: hit_rate(sub[sub["--ablation"] == ab])
                 for ab in ["full", "tube_only", "softmin_only", "none"]}
        line = "  ".join(f"{ab}={v:.1f}" for ab, v in rates.items())
        # 与 point-target 差值 + 配对 bootstrap CI（fig:speed (c) 用）
        pt = rates["none"]
        cis = {t: paired_boot_ci(sub, t, "none") for t in ("full", "tube_only", "softmin_only")}
        print(f"  {sp} m/s TCP {tcp}: {line}  |  vs none: "
              f"full {rates['full']-pt:+.1f} ({cis['full']}), "
              f"corridor {rates['tube_only']-pt:+.1f} ({cis['tube_only']}), "
              f"softmin {rates['softmin_only']-pt:+.1f} ({cis['softmin_only']})")

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
