#!/usr/bin/env python3
"""论文图表生成 —— ICRA 2027 冲刺版（fig3-8 + Table I/II）。

数据源（讨论留档 §6 图表-数据映射 v2，叙事定稿后不再返工）：
- fig3 走廊示意:        exp18 raw/a_hit_clean.npz
- fig4 关节轨迹+TCP:    exp18 raw/a（基准）+ b（空间扰动）
- fig5 命中率 vs 球速:  exp15_speed_v2（9 档 × 200）+ exp16_limits_v2（TCP 1.0 vs 1.8）
- fig6 鲁棒性核心:      exp17d/17a（标称四档）+ exp17b/17f（扰动网格）+ exp17g/17h（角点）
- fig7 实时性能:        exp18 timing.json（三段式 + 停顿对比）
- fig8 诊断:            exp18 raw/b（走廊场景）+ c（失败模式）+ d（噪声+KF）
- Table I:              四档 × 条件总表；Table II: 限速配对 + DiD

写法遵循 skills/figure_generation.md 的 IEEE 样式（Times、色盲友好、PDF 300dpi），
且遵守 §9 名实不符红线（走廊仅 hinge 位置偏离一项）与 W1-W8 写作约束。

用法:
    python scripts/plot/paper_figs.py --fig 3 4 5 6 7 8 table
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 3D 投影注册

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT / "experiment_data"
OUT = PROJECT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})

C = {
    "full": "#009E73",        # 绿
    "tube_only": "#0072B2",   # 蓝
    "softmin_only": "#E69F00",  # 琥珀
    "none": "#D55E00",        # 橙红
    "ball": "#D55E00",
    "racket": "#0072B2",
    "corridor": "#E69F00",
}
MODE_LABELS = {"full": "full", "tube_only": "tube only",
               "softmin_only": "softmin only", "none": "point target"}
MODE_ORDER = ["full", "tube_only", "softmin_only", "none"]


def load_csv(name: str) -> list[dict]:
    """读取 batch_runner 产出的 results.csv（剔除 error 行）。"""
    path = DATA / name / "results.csv"
    with open(path, "r", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if not (r.get("error") or "").strip()]


def cfg_of(row: dict) -> dict:
    """解析一行的 config_json。"""
    return json.loads(row["config_json"])


def is_hit(row: dict) -> bool:
    """命中判定（hit 列为 'True'/'False' 字符串）。"""
    return row["hit"].strip().lower() in ("true", "1")


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 置信区间（比例），返回 (p, 半宽)。"""
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center, half


def diff_ci(h1: int, n1: int, h2: int, n2: int) -> tuple[float, float]:
    """两比例之差的 95% CI（保守非合并口径），返回 (diff, 半宽)。"""
    p1, p2 = h1 / n1, h2 / n2
    se = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2) * 100, 1.96 * se * 100


def agg(rows: list[dict], key_fn) -> dict:
    """按 key_fn 聚合 (hits, n)。"""
    out: dict = defaultdict(lambda: [0, 0])
    for r in rows:
        key = key_fn(r)
        out[key][1] += 1
        if is_hit(r):
            out[key][0] += 1
    return out


def style_ax(ax) -> None:
    """统一坐标轴样式。"""
    ax.grid(True, alpha=0.3, linewidth=0.3)


# ==============================================================================
# fig3 走廊示意
# ==============================================================================

def fig3(npz_path: Path = DATA / "exp18_fig_assets/raw/a_hit_clean.npz") -> None:
    """Fig.3: 球轨迹 + 球拍轨迹 + 击球点 + 走廊横截面示意图。"""
    d = np.load(npz_path)
    ball, tcp = d["ball_pos"], d["tcp_pos"]
    hit = int(d["hit_step"])
    dt = float(d["dt"])

    fig = plt.figure(figsize=(7.16, 3.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ball[:, 0], ball[:, 1], ball[:, 2], color=C["ball"],
            lw=1.5, label="Ball trajectory")
    ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], color=C["racket"],
            lw=1.5, label="Racket TCP trajectory")
    # 击球时刻标记
    ax.scatter(*ball[hit], color="k", s=20, marker="*", label="Hit (t=%.1f s)" % (hit * dt))
    ax.scatter(*tcp[hit], color="k", s=20, marker="*")
    # 走廊横截面（半宽 = 拍半径 0.12 m，沿球轨迹在击球点前后各 2 处示意）
    r_half = 0.12
    for k in (max(0, hit - 15), hit, min(len(ball) - 1, hit + 15)):
        c = ball[k]
        u = ball[min(k + 1, len(ball) - 1)] - ball[max(k - 1, 0)]
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(u, [0, 0, 1.0])
        v = v / (np.linalg.norm(v) + 1e-12) if np.linalg.norm(v) > 1e-9 else [1, 0, 0]
        w = np.cross(u, v)
        th = np.linspace(0, 2 * np.pi, 24)
        ring = c + r_half * (np.outer(np.cos(th), v) + np.outer(np.sin(th), w))
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2],
                color=C["corridor"], lw=0.8, alpha=0.85)
    ax.plot([], [], color=C["corridor"], lw=1.2, label="Corridor (half-width 0.12 m)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=18, azim=-60)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2, linewidth=0.3)
    save(fig, "fig3_tube_corridor.pdf")


# ==============================================================================
# fig4 关节轨迹 + TCP 速度
# ==============================================================================

def fig4(npz_a: Path = DATA / "exp18_fig_assets/raw/a_hit_clean.npz",
         npz_b: Path = DATA / "exp18_fig_assets/raw/b_hit_space_perturb.npz") -> None:
    """Fig.4: (a) 关节角（期望 vs 实际）(b) TCP 速度 + 1.8 m/s 限位线。

    注：NPZ 未记录力矩（collect_fig_assets 只存 q/tcp/ball），
    按叙事改用 TCP 速度——安全限位叙事比力矩更贴合论文主线。
    """
    da, db = np.load(npz_a), np.load(npz_b)
    t_a = da["timestamps"] * 1000  # ms
    t_b = db["timestamps"] * 1000
    hit_a = int(da["hit_step"]) * float(da["dt"]) * 1000
    hit_b = int(db["hit_step"]) * float(db["dt"]) * 1000
    joint_names = ["J0", "J1", "J2", "J3", "J4", "J5"]
    colors = ["#0072B2", "#D55E00", "#009E73", "#56B4E9", "#E69F00", "#CC79A7"]

    fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1]})
    # (a) 关节角（注意单位：NPZ 中 q_actual 为弧度、q_desired 为度——
    #     TrajectoryRecorder 混用单位，此处分别归一化到「度」后绘制）
    ax = axes[0]
    for j in range(6):
        ax.plot(t_a, da["q_actual"][:, j] * 180 / np.pi, color=colors[j], lw=0.9,
                label=joint_names[j])
        ax.plot(t_a, da["q_desired"][:, j], color=colors[j],
                lw=0.4, ls="--", alpha=0.5)
    ax.axvline(hit_a, color="k", ls=":", lw=0.8)
    ax.text(hit_a + 5, ax.get_ylim()[1] * 0.9, "hit", fontsize=7)
    ax.set_ylabel("Joint angle (deg)")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.28), fontsize=6,
              columnspacing=0.8)
    ax.set_title("(a) Joint trajectories (solid: actual, dashed: desired)", fontsize=9)
    style_ax(ax)
    # (b) TCP 速度（两 run 对比 + 限位线）
    ax = axes[1]
    for d, t, hit_t, label, col in [
        (da, t_a, hit_a, "baseline", C["full"]),
        (db, t_b, hit_b, "space-perturbed (s=0.1 m)", C["none"]),
    ]:
        v = np.linalg.norm(np.gradient(d["tcp_pos"], axis=0), axis=1) / float(d["dt"])
        ax.plot(t, v, color=col, lw=0.9, label=label)
        ax.axvline(hit_t, color=col, ls=":", lw=0.7, alpha=0.7)
    ax.axhline(1.8, color="gray", ls="--", lw=0.8)
    ax.text(0.015, 0.95, "TCP limit 1.8 m/s", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5, color="dimgray",
            bbox=dict(fc="white", ec="none", alpha=0.75))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("TCP speed (m/s)")
    ax.legend(loc="upper right")
    ax.set_title("(b) TCP speed profiles", fontsize=9)
    style_ax(ax)
    save(fig, "fig4_joint_trajectory.pdf")


# ==============================================================================
# fig5 命中率 vs 球速
# ==============================================================================

def fig5() -> None:
    """Fig.5: (a) 命中率 vs 球速（仿真默认限位，9 档 × 200 seeds）
    (b) TCP 1.0 vs 1.8 对比（exp16，7 m/s）。"""
    rows15 = load_csv("exp15_speed_v2")
    rows16 = load_csv("exp16_limits_v2")

    by_speed = agg(rows15, lambda r: int(cfg_of(r).get("--ball-speed", 0)))
    speeds = sorted(by_speed)
    rates = [by_speed[s][0] / by_speed[s][1] * 100 for s in speeds]
    errs = [wilson(*by_speed[s])[1] * 100 for s in speeds]

    by_lim = agg(rows16, lambda r: ("TCP 1.0" if "--limits-config" in cfg_of(r)
                                    else "TCP 1.8"))
    lim_rates, lim_errs, lim_n = [], [], []
    for k in ["TCP 1.8", "TCP 1.0"]:
        h, n = by_lim.get(k, [0, 0])
        lim_rates.append(h / n * 100)
        lim_errs.append(wilson(h, n)[1] * 100)
        lim_n.append(n)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.errorbar(speeds, rates, yerr=errs, color=C["full"], marker="o", capsize=2,
                lw=1.0, label="Default limits (TCP 1.8 m/s)")
    ax.set_xlabel("Ball speed (m/s)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Hit rate vs. ball speed", fontsize=9)
    ax.legend(loc="lower left")
    style_ax(ax)

    ax = axes[1]
    x = np.arange(2)
    bars = ax.bar(x, lim_rates, yerr=lim_errs, capsize=3, width=0.5,
                  color=[C["full"], C["none"]], alpha=0.85)
    ax.set_xticks(x, ["TCP 1.8", "TCP 1.0"])
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(0, 105)
    for i, n in enumerate(lim_n):
        ax.text(x[i], lim_rates[i] + lim_errs[i] + 3, f"n={n}",
                ha="center", fontsize=6)
    ax.set_title("(b) Real-robot TCP limit (7 m/s)", fontsize=9)
    style_ax(ax)
    save(fig, "fig5_hit_rate_vs_speed.pdf")


# ==============================================================================
# fig6 鲁棒性核心图（四面板）
# ==============================================================================

def _nominal_modes() -> dict:
    """标称四档命中率（7/9/12 m/s）: full/none 取自 exp17a 无噪声格,
    tube/softmin 取自 exp17d。"""
    out: dict = defaultdict(dict)
    for r in load_csv("exp17a_noise"):
        cfg = cfg_of(r)
        if "--obs-noise-pos" in cfg or "--obs-use-kf" in cfg:
            continue  # 仅无噪声、无 KF 格
        speed, mode = int(cfg.get("--ball-speed", 0)), cfg.get("--ablation")
        if speed in (7, 9, 12) and mode in ("full", "none"):
            out[speed].setdefault(mode, [0, 0])
            out[speed][mode][1] += 1
            if is_hit(r):
                out[speed][mode][0] += 1
    for r in load_csv("exp17d_mechanism"):
        cfg = cfg_of(r)
        speed, mode = int(cfg.get("--ball-speed", 0)), cfg.get("--ablation")
        if speed in (7, 9, 12) and mode in ("tube_only", "softmin_only"):
            out[speed].setdefault(mode, [0, 0])
            out[speed][mode][1] += 1
            if is_hit(r):
                out[speed][mode][0] += 1
    return out


def _grid_gain() -> tuple[np.ndarray, list, list]:
    """扰动网格 full − none 增益（exp17b @9，5×4 格，n≈100/格）。"""
    cells = agg(load_csv("exp17b_perturb"),
                lambda r: (float(cfg_of(r).get("--time-perturb-ms", 0) or 0),
                           float(cfg_of(r).get("--space-perturb-m", 0) or 0),
                           cfg_of(r).get("--ablation")))
    t_ax = sorted({k[0] for k in cells})
    s_ax = sorted({k[1] for k in cells})
    gain = np.full((len(t_ax), len(s_ax)), np.nan)
    for i, t in enumerate(t_ax):
        for j, s in enumerate(s_ax):
            hf, nf = cells.get((t, s, "full"), [0, 0])
            hn, nn = cells.get((t, s, "none"), [0, 0])
            if nf and nn:
                gain[i, j] = (hf / nf - hn / nn) * 100
    return gain, t_ax, s_ax


def _corner_gain_vs_speed() -> tuple[list, list, list]:
    """角点 t=0/s=0.2 的 tube−none 增益 vs 球速（9 m/s exp17g, 12 m/s exp17h）。

    注意：exp17g 含 t=0 与 t=50 两个角点，此处必须过滤到 t=0
    （--time-perturb-ms 缺省或 0），否则 t=50 行稀释增益。
    """
    rows = load_csv("exp17g_spatial_power") + load_csv("exp17h_extreme")
    # t=0 且 s=0.2 的角点：exp17h 的 9 m/s 行含 s=0.3/0.4 角点，必须同时过滤 s
    rows = [r for r in rows
            if float(cfg_of(r).get("--time-perturb-ms", 0) or 0) == 0.0
            and float(cfg_of(r).get("--space-perturb-m", 0) or 0) == 0.2]
    cells = agg(rows, lambda r: (int(cfg_of(r).get("--ball-speed", 0)),
                                 cfg_of(r).get("--ablation")))
    speeds, gains, errs = [], [], []
    for sp in (9, 12):
        ht, nt = cells.get((sp, "tube_only"), [0, 0])
        hn, nn = cells.get((sp, "none"), [0, 0])
        d, ci = diff_ci(ht, nt, hn, nn)
        speeds.append(sp)
        gains.append(d)
        errs.append(ci)
    return speeds, gains, errs


def _corner_gain_vs_s() -> tuple[list, list, list]:
    """9 m/s t=0 下 tube−none 增益 vs s。

    仅保留高功效角点 s∈{0.2,0.3,0.4}（exp17g/17h，n≥1481/格）；
    s=0.05/0.1 仅 n=100/格（CI ±10pp），噪音淹没信号，不纳入论文图
    （与 discussion §10.1 表口径一致）。
    """
    rows_g = load_csv("exp17g_spatial_power")
    rows_h = load_csv("exp17h_extreme")

    def cell(rows, s, mode):
        return agg(rows, lambda r: (float(cfg_of(r).get("--space-perturb-m", 0) or 0),
                                    float(cfg_of(r).get("--time-perturb-ms", 0) or 0),
                                    cfg_of(r).get("--ablation"))).get((s, 0.0, mode),
                                                                      [0, 0])

    s_ax, gains, errs = [], [], []
    for s, rows in ((0.2, rows_g), (0.3, rows_h), (0.4, rows_h)):
        ht, nt = cell(rows, s, "tube_only")
        hn, nn = cell(rows, s, "none")
        d, ci = diff_ci(ht, nt, hn, nn)
        s_ax.append(s)
        gains.append(d)
        errs.append(ci)
    return s_ax, gains, errs


def fig6() -> None:
    """Fig.6: (a) 标称四档柱状 (b) 扰动网格增益热图
    (c) 走廊增益 vs 球速 (d) 走廊增益 vs 扰动幅度。"""
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.2))

    # (a) 标称四档
    ax = axes[0][0]
    nom = _nominal_modes()
    speeds = [7, 9, 12]
    width = 0.18
    for mi, mode in enumerate(MODE_ORDER):
        vals, errs = [], []
        for sp in speeds:
            h, n = nom[sp].get(mode, [0, 0])
            vals.append(h / n * 100 if n else 0)
            errs.append(wilson(h, n)[1] * 100 if n else 0)
        ax.bar(np.arange(3) + (mi - 1.5) * width, vals, width, yerr=errs,
               capsize=1.5, color=C[mode], label=MODE_LABELS[mode], alpha=0.9)
    ax.set_xticks(np.arange(3), [f"{s} m/s" for s in speeds])
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(0, 125)
    ax.set_title("(a) Nominal ablation", fontsize=9)
    ax.legend(ncol=2, fontsize=5.5, loc="upper left", framealpha=0.9)
    style_ax(ax)

    # (b) 扰动网格增益热图
    ax = axes[0][1]
    gain, t_ax, s_ax = _grid_gain()
    im = ax.imshow(gain, aspect="auto", cmap="RdBu_r", vmin=-25, vmax=25)
    ax.set_xticks(range(len(s_ax)), [f"{s:.2f}" for s in s_ax])
    ax.set_yticks(range(len(t_ax)), [f"{t:.0f}" for t in t_ax])
    ax.set_xlabel("Space perturb. max (m)")
    ax.set_ylabel("Time perturb. max (ms)")
    for i in range(gain.shape[0]):
        for j in range(gain.shape[1]):
            if not np.isnan(gain[i, j]):
                ax.text(j, i, f"{gain[i, j]:+.0f}", ha="center", va="center",
                        fontsize=6, color="k")
    ax.set_title("(b) full $-$ none gain (pp), 9 m/s", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="pp")

    # (c) 走廊增益 vs 球速
    ax = axes[1][0]
    nom_speeds, nom_gains = [], []
    for sp in (7, 9, 12):
        ht, nt = nom[sp].get("tube_only", [0, 0])
        hn, nn = nom[sp].get("none", [0, 0])
        d, _ = diff_ci(ht, nt, hn, nn)
        nom_speeds.append(sp)
        nom_gains.append(d)
    ax.plot(nom_speeds, nom_gains, marker="o", color=C["tube_only"],
            label="nominal (t=0, s=0)")
    sp2, g2, e2 = _corner_gain_vs_speed()
    ax.errorbar(sp2, g2, yerr=e2, marker="s", capsize=3, color=C["corridor"],
                label="corner (t=0, s=0.2 m)")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("Ball speed (m/s)")
    ax.set_ylabel("Corridor gain vs point target (pp)")
    ax.set_title("(c) Corridor value grows with speed", fontsize=9)
    ax.legend(loc="upper left")
    style_ax(ax)

    # (d) 走廊增益 vs s（仅高功效角点；显著性标注）
    ax = axes[1][1]
    s_ax, gains, errs = _corner_gain_vs_s()
    ax.errorbar(s_ax, gains, yerr=errs, marker="o", capsize=3,
                color=C["tube_only"])
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_ylim(-2.2, 9.0)
    ax.text(0.2, gains[0] + errs[0] + 0.7, "$*$, $p=0.0022$", ha="center",
            fontsize=6, color="k")
    ax.set_xlabel("Space perturb. max $s$ (m)")
    ax.set_ylabel("Corridor gain vs point target (pp)")
    ax.set_title("(d) Corridor value beyond $s{=}0.2$", fontsize=9)
    style_ax(ax)

    fig.tight_layout()
    save(fig, "fig6_tube_robustness.pdf")


# ==============================================================================
# fig7 实时性能
# ==============================================================================

def fig7(timing_path: Path = DATA / "exp18_fig_assets/timing.json") -> None:
    """Fig.7: (a) 三段式重规划耗时（对数轴点图）(b) 超周期步占比 (c) 最大单步延迟。"""
    t = json.load(open(timing_path))
    sync, async_ = t["sync"], t["async"]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3),
                             gridspec_kw={"width_ratios": [1.5, 1, 1]})

    # (a) 三段式耗时（点图：对数轴下柱状长度会失真，点图 + 数值标注更诚实）
    ax = axes[0]
    tiers = [("First plan (30 it)", sync["by_iters"]["30"]["mean_ms"], C["none"]),
             ("Steady far (0 it)", sync["by_iters"]["0"]["mean_ms"], C["tube_only"]),
             ("Steady near (5 it)", sync["by_iters"]["5"]["mean_ms"], C["full"])]
    ys = np.arange(len(tiers))[::-1]
    for y, (name, val, col) in zip(ys, tiers):
        ax.hlines(y, 1, val, color=col, lw=1.2)
        ax.plot(val, y, "o", color=col, ms=5)
        ax.text(val * 1.25, y, f"{val:.1f} ms", va="center", fontsize=6.5)
    ax.set_yticks(ys, [x[0] for x in tiers])
    ax.set_xscale("log")
    ax.set_xlim(1, 4000)
    ax.set_xlabel("Replan time (ms, log)")
    ax.set_title("(a) Sync replan time", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x", linewidth=0.3)

    # (b) 超周期步占比
    ax = axes[1]
    over = [sync["stall"]["over_ratio"] * 100, async_["stall"]["over_ratio"] * 100]
    ax.bar(["Sync", "Async"], over, color=[C["none"], C["tube_only"]],
           alpha=0.9, width=0.55)
    for i, v in enumerate(over):
        ax.text(i, v + 0.03, f"{v:.2f}%", ha="center", fontsize=6.5)
    ax.set_ylabel("Steps exceeding 5 ms period (%)")
    ax.set_ylim(0, 1.25)
    ax.set_title("(b) Stall frequency", fontsize=9)
    style_ax(ax)

    # (c) 最大单步延迟
    ax = axes[2]
    pmax = [sync["stall"]["max_ms_max"], async_["stall"]["max_ms_max"]]
    ax.bar(["Sync", "Async"], pmax, color=[C["none"], C["tube_only"]],
           alpha=0.9, width=0.55)
    for i, v in enumerate(pmax):
        ax.text(i, v + 1.0, f"{v:.1f} ms", ha="center", fontsize=6.5)
    ax.set_ylabel("Max per-step latency (ms)")
    ax.set_ylim(0, 52)
    ax.set_title("(c) Worst-case stall", fontsize=9)
    style_ax(ax)

    fig.tight_layout()
    save(fig, "fig7_realtime_performance.pdf")


# ==============================================================================
# fig8 诊断（三场景 × 双视图）
# ==============================================================================

def fig8() -> None:
    """Fig.8: 诊断六子图——三场景（走廊命中 / 失败 / 噪声+KF）× 双视图
    （3D 轨迹 + 球拍-球最近距离随时间）。"""
    specs = [
        ("b_hit_space_perturb", "Corridor hit (s=0.1 m)", C["full"]),
        ("c_miss_combined_perturb", "Miss (t=50 ms, s=0.2 m)", C["none"]),
        ("d_hit_noise_kf", "Hit with noise+KF", C["softmin_only"]),
    ]
    fig = plt.figure(figsize=(7.16, 6.0))
    for i, (name, label, col) in enumerate(specs):
        d = np.load(DATA / "exp18_fig_assets/raw" / f"{name}.npz")
        ball, tcp = d["ball_pos"], d["tcp_pos"]
        hit = int(d["hit_step"])
        t = d["timestamps"] * 1000
        # 左: 3D 轨迹
        ax = fig.add_subplot(3, 2, 2 * i + 1, projection="3d")
        ax.plot(ball[:, 0], ball[:, 1], ball[:, 2], color=C["ball"],
                lw=1.2, label="ball")
        ax.plot(tcp[:, 0], tcp[:, 1], tcp[:, 2], color=C["racket"],
                lw=1.2, label="racket")
        ax.scatter(*ball[hit], color="k", s=16, marker="*")
        # 3D 轴不重复标注轴名（刻度即米制；单位在图注中说明），避免与刻度文字碰撞
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_title(f"({chr(97 + 2 * i)}) {label}", fontsize=8)
        ax.view_init(elev=20, azim=-60)
        # 3D 刻度在透视缩角处易叠压：限制每轴刻度数量
        from matplotlib.ticker import MaxNLocator
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=5.5, pad=1)
        ax.legend(fontsize=6, loc="upper left")
        # 右: 最近距离曲线（对数轴）——阈值线不使用（球心-拍心距离的接触值依赖拍面几何，
        #     画经验阈值会把 miss 案例误读为「差一点碰到」）；改标注各案例最近距离
        ax = fig.add_subplot(3, 2, 2 * i + 2)
        dist = np.linalg.norm(ball - tcp, axis=1) * 1000  # mm
        ax.semilogy(t, dist, color=col, lw=1.0)
        ax.axvline(hit * float(d["dt"]) * 1000, color="k", ls=":", lw=0.7)
        d_min = float(dist.min())
        ax.axhline(d_min, color=col, ls="--", lw=0.6, alpha=0.7)
        ax.text(t[-1], d_min * 1.12, f"min {d_min:.0f} mm", ha="right",
                fontsize=5.5, color=col)
        ax.set_ylim(40, 10000)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Ball-racket dist. (mm, log)")
        ax.set_title(f"({chr(97 + 2 * i + 1)}) Distance profile", fontsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.3, which="both")
    fig.tight_layout(h_pad=2.0)
    save(fig, "fig8_tube_diagnostic.pdf")


# ==============================================================================
# Table I / II
# ==============================================================================

def tables() -> None:
    """Table I: 四档 × 条件总表；Table II: 限速配对 + DiD。"""
    out_dir = OUT / "table_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    nom = _nominal_modes()
    # 20 格合并
    merged = {}
    for name in ("exp17b_perturb", "exp17f_mechanism_perturb"):
        for r in load_csv(name):
            mode = cfg_of(r).get("--ablation")
            merged.setdefault(mode, [0, 0])
            merged[mode][1] += 1
            if is_hit(r):
                merged[mode][0] += 1
    # 角点 12 m/s
    corner12 = agg(load_csv("exp17h_extreme"),
                   lambda r: (int(cfg_of(r).get("--ball-speed", 0)),
                              cfg_of(r).get("--ablation")))
    # 限速配对
    lim = {}
    for r in load_csv("exp17i_limits_ablation"):
        cfg = cfg_of(r)
        key = ("TCP 1.0" if "--limits-config" in cfg else "TCP 1.8",
               cfg.get("--ablation"))
        lim.setdefault(key, [0, 0])
        lim[key][1] += 1
        if is_hit(r):
            lim[key][0] += 1

    def fmt(h, n):
        return f"{h / n * 100:.1f}"  # 原始比例（与正文/discussion 口径一致）

    lines = []
    lines.append(r"\begin{table*}[t]" + "\n" + r"\centering")
    lines.append(r"\caption{Hit rate (\%) of the four configurations across "
                 r"conditions.}" + "\n")
    lines.append(r"\label{tab:ablation}" + "\n")
    lines.append(r"\begin{tabular}{lcccc}" + "\n" + r"\toprule")
    lines.append("Condition & full & tube & softmin & point target \\\\\n" + r"\midrule")
    for sp in (7, 9, 12):
        row = [f"Nominal {sp} m/s"]
        for mode in MODE_ORDER:
            h, n = nom[sp].get(mode, [0, 0])
            row.append(fmt(h, n))
        lines.append(" & ".join(row) + " \\\\\n")
    row = ["Perturbed grid (20 cells, 9 m/s)"]
    for mode in MODE_ORDER:
        h, n = merged.get(mode, [0, 0])
        row.append(fmt(h, n))
    lines.append(" & ".join(row) + " \\\\\n")
    row = ["Corner 12 m/s, t=0, s=0.2 m"]
    for mode in MODE_ORDER:
        h, n = corner12.get((12, mode), [0, 0])
        row.append(fmt(h, n))
    lines.append(" & ".join(row) + " \\\\\n" + r"\bottomrule")
    lines.append(r"\end{tabular}" + "\n" + r"\end{table*}" + "\n")
    (out_dir / "table1_comparison.tex").write_text("".join(lines), encoding="utf-8")

    # Table II 限速配对 + DiD
    def gain(mode, lim_key):
        hm, nm = lim.get((lim_key, mode), [0, 0])
        hn, nn = lim.get((lim_key, "none"), [0, 0])
        return hm / nm - hn / nn

    lines = []
    lines.append(r"\begin{table}[t]" + "\n" + r"\centering")
    lines.append(r"\caption{Gain of each configuration over the point-target "
                 r"baseline under TCP speed limits (7 m/s, n$\approx$394).}"
                 + "\n")
    lines.append(r"\label{tab:limits}" + "\n")
    lines.append(r"\begin{tabular}{lccc}" + "\n" + r"\toprule")
    lines.append("Config. & TCP 1.8 gain (pp) & TCP 1.0 gain (pp) & DiD $p$ \\\\\n"
                 + r"\midrule")
    for mode in ("tube_only", "softmin_only", "full"):
        g18, g10 = gain(mode, "TCP 1.8") * 100, gain(mode, "TCP 1.0") * 100
        # 显著性：从提取脚本口径（保守非合并）取 p
        hm1, nm1 = lim.get(("TCP 1.0", mode), [0, 0])
        hn1, nn1 = lim.get(("TCP 1.0", "none"), [0, 0])
        hm0, nm0 = lim.get(("TCP 1.8", mode), [0, 0])
        hn0, nn0 = lim.get(("TCP 1.8", "none"), [0, 0])
        se1 = sqrt(hm1 / nm1 * (1 - hm1 / nm1) / nm1 + hn1 / nn1 * (1 - hn1 / nn1) / nn1)
        se0 = sqrt(hm0 / nm0 * (1 - hm0 / nm0) / nm0 + hn0 / nn0 * (1 - hn0 / nn0) / nn0)
        did = (g10 - g18) / 100
        z = did / sqrt(se1 ** 2 + se0 ** 2)
        from statistics import NormalDist
        p = 2 * (1 - NormalDist().cdf(abs(z)))
        lines.append(f"{MODE_LABELS[mode]} & {g18:+.1f} & {g10:+.1f} & {p:.4f} \\\\\n")
    lines.append(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n")
    (out_dir / "table2_ablation.tex").write_text("".join(lines), encoding="utf-8")
    print("Table I/II 已写入", out_dir)


# ==============================================================================

def save(fig, name: str) -> None:
    """保存到 paper/figures/。"""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 {path}")


def main() -> None:
    """按 --fig 参数生成指定图表。"""
    parser = argparse.ArgumentParser(description="ICRA 2027 论文图表生成")
    parser.add_argument("--fig", nargs="+", default=[],
                        choices=["3", "4", "5", "6", "7", "8", "table"],
                        help="要生成的图表（可多个）")
    args = parser.parse_args()
    targets = args.fig or ["3", "4", "5", "6", "7", "8", "table"]
    for f in targets:
        {"3": fig3, "4": fig4, "5": fig5, "6": fig6, "7": fig7, "8": fig8,
         "table": tables}[f]()


if __name__ == "__main__":
    main()
