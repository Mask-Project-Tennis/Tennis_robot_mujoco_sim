#!/usr/bin/env python3
"""结果段动画生成器：S5（标称增益图）/ S6（扰动网格）/ S7（TCP 限速）/ S8（实时时间轴）。

数据纪律（B5）：**所有屏显数字运行时从数据源读取/重算，不手抄**——
- ``paper/planning/06-stats-active.json``：E3 标称消融（S5）、E4 20 格扰动网格（S6）、
  E8 TCP 限速机制（S7）的率值/配对差/CI/p 值；
- ``experiment_data/exp18_fig_assets/timing.json``：首计划 928.3 ms、稳态 4 ms/p95 32 ms（S8）；
- ``experiment_data/exp23_nofreeze/results.csv``：不冻结端到端 999 ms→0% 与 224 ms→74.5%
  （S8，按 ``analyze_chat12_night.py`` 同口径重算）；
- 150 ms 预算为论文 results.tex RQ4 中声明的设计预算（常量，注明出处）。
显示舍入用 half-up（Decimal），与论文一致（如 +17.35 → +17.4、+32.65 → +32.7）。

用法：
    python scripts/video/anim_results.py                 # 渲染全部四段
    python scripts/video/anim_results.py --only s7       # 只渲染 S7
"""
from __future__ import annotations

import argparse
import json
import logging
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "paper" / "video" / "assets"
FPS_DEFAULT = 30
W, H = 128.0, 72.0
BUDGET_MS = 150.0  # 论文 results.tex RQ4 声明的重规划预算（设计常量，非实验量）

logger = logging.getLogger("anim_results")

# ── 配色（与 anim_method.py 一致）───────────────────────────────────────────
INK = (0.13, 0.14, 0.17)
INK_2 = (0.19, 0.20, 0.25)  # 次文字（用户两轮反馈逐次调深：0.42→0.27→0.19）
BG = (0.965, 0.961, 0.949)
GREY = (0.58, 0.58, 0.62)
ORANGE = (0.878, 0.482, 0.122)
BLUE = (0.184, 0.427, 0.710)
GREEN = (0.184, 0.620, 0.373)
RED = (0.765, 0.290, 0.212)
WHITE = (1.0, 1.0, 1.0)
STRIP = (0.137, 0.145, 0.165)


# ── 基础工具（与 anim_method.py 同构）───────────────────────────────────────
def seg(t: float, a: float, b: float) -> float:
    """阶段进度 0→1，smoothstep 缓动。"""
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0
    u = (t - a) / (b - a)
    return u * u * (3.0 - 2.0 * u)


def rgba(c: tuple, a: float) -> tuple:
    """颜色 + 透明度。"""
    return (c[0], c[1], c[2], max(0.0, min(1.0, a)))


def r1(v: float) -> str:
    """直接 half-up 保留 1 位小数（率值/毫秒口径，如 22.449→22.4）。"""
    return str(Decimal(str(float(v))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def r1_pp(v: float) -> str:
    """pp 增益口径：经 2 位小数中间步再取 1 位（与论文一致：17.347→17.35→+17.4）。"""
    q2 = Decimal(str(float(v))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return str(q2.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def r0(v: float) -> str:
    """half-up 取整。"""
    return str(Decimal(str(float(v))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def fmt_pp(v: float) -> str:
    """带符号 pp 值（Unicode 负号；用论文 pp 舍入管线）。"""
    s = r1_pp(abs(v))
    return ("+" if v >= 0 else "−") + s


def fmt_ci(ci: list) -> str:
    """CI 区间（直接舍入口径；Unicode 负号）。"""
    def one(v: float) -> str:
        return ("−" if v < 0 else "") + r1(abs(v))
    return f"[{one(ci[0])}, {one(ci[1])}]"


def fmt_p(p: float) -> str:
    """p 值显示：<0.001 或 3 位小数。"""
    if p < 1e-3:
        return "p<0.001"
    return "p=" + str(Decimal(str(float(p))).quantize(Decimal("0.001"),
                                                      rounding=ROUND_HALF_UP))


def new_scene(title: str, title_a: float, footer: str, tag: str):
    """建画布：底色 + 标题 + 底部条件条 + 右下数据来源角标。"""
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc=BG, ec="none"))
    ax.text(6, 66.2, title, fontsize=21, fontweight="bold", color=rgba(INK, title_a),
            ha="left", va="center")
    ax.add_patch(Rectangle((0, 0), W, 6.2, fc=STRIP, ec="none", alpha=0.96))
    ax.text(6, 3.1, footer, fontsize=14.5, color=(1, 1, 1, 0.94), ha="left", va="center")
    ax.text(W - 6, 3.1, tag, fontsize=12.5, color=(1, 1, 1, 0.55), ha="right",
            va="center")
    return fig, ax


def label(ax, x, y, s, a, size=13, color=INK, ha="left", va="center", weight="normal"):
    """带透明度文字。"""
    if a <= 0:
        return
    ax.text(x, y, s, fontsize=size, color=rgba(color, a), ha=ha, va=va,
            fontweight=weight)


def arrow(ax, p0, p1, a, color=INK_2, lw=2.2, style="-|>", ms=16, rad=0.0):
    """箭头（rad≠0 为弧线）。"""
    if a <= 0:
        return
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 color=rgba(color, a), lw=lw, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))


# ── 数据装载与 B5 审计 ──────────────────────────────────────────────────────
def load_data() -> dict:
    """装载全部数字源并打印 B5 审计表。"""
    stats = json.loads((REPO / "paper/planning/06-stats-active.json").read_text())
    timing = json.loads(
        (REPO / "experiment_data/exp18_fig_assets/timing.json").read_text())

    # exp23 不冻结端到端：与 scripts/extract/analyze_chat12_night.py 同口径重算
    import pandas as pd
    df = pd.read_csv(REPO / "experiment_data/exp23_nofreeze/results.csv")
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]
    df = df.assign(_iters=df["config_json"].apply(
        lambda c: json.loads(c).get("--first-plan-iters", 30)))
    nofreeze = {}
    for it, g in df.groupby("_iters"):
        nofreeze[int(it)] = {
            "median_ms": float(g["first_plan_ms"].median()),
            "rate": float(100.0 * (g["hit_type"] == "active").mean()),
            "n": len(g),
        }

    e3, e4, e8 = stats["E3_nominal"], stats["E4_grid"], stats["E8_limit_mechanism"]
    # S6 网格轴取值从 per_cell 键解析（t{ms}|s{m}|arm），不硬编码
    pc = e4["per_cell"]
    s6_axes = (sorted({int(k.split("|")[0][1:]) for k in pc}),
               sorted({float(k.split("|")[1][1:]) for k in pc}))
    assert s6_axes[0] and s6_axes[1] and len(s6_axes[0]) * len(s6_axes[1]) == 20
    d = {
        "s6_axes": s6_axes,
        "s5_cells": {k: e3["cells"][k] for k in
                     ("9|none", "9|softmin_only", "12|none", "12|softmin_only")},
        "s5_paired": {k: e3["paired"][k] for k in
                      ("9|softmin_only_vs_none", "12|softmin_only_vs_none")},
        "s6_cand": e4["merged_20cell"]["softmin_only"],
        "s6_pt": e4["merged_20cell"]["none"],
        "s6_paired": e4["paired_20cell"]["softmin_only_vs_none"],
        "s7_paired": {k: e8["paired"][k] for k in
                      ("TCP1.0|softmin_only_vs_none", "TCP1.0|full_vs_none",
                       "TCP1.0|tube_only_vs_none")},
        "s7_abs": {k: e8["cells"][k]["rate"] for k in
                   ("TCP1.0|none", "TCP1.0|softmin_only", "TCP1.0|full",
                    "TCP1.0|tube_only")},
        "s8_timing": {"first": timing["sync"]["first_plan"],
                      "steady": timing["sync"]["steady_state"]},
        "s8_nofreeze": nofreeze,
    }

    # B5 审计表（屏显数字与来源一一对应）
    p = d["s5_paired"]
    print("[B5 审计] S5 标称增益（E3_nominal）：")
    for sp in ("9", "12"):
        c = d["s5_cells"][f"{sp}|none"], d["s5_cells"][f"{sp}|softmin_only"]
        g = p[f"{sp}|softmin_only_vs_none"]
        print(f"  {sp} m/s: pt={r1(c[0]['rate'])} cand={r1(c[1]['rate'])}"
              f"  gain={fmt_pp(g['diff_pp'])} CI{fmt_ci(g['ci95'])}"
              f" {fmt_p(g['p_mcnemar'])} n={g['n_paired']}")
    g6 = d["s6_paired"]
    print(f"[B5 审计] S6 扰动网格（E4 merged_20cell）：cand={r1(d['s6_cand']['rate'])}"
          f" pt={r1(d['s6_pt']['rate'])} gain={fmt_pp(g6['diff_pp'])}"
          f" CI{fmt_ci(g6['ci95'])} n={g6['n_paired']}")
    print("[B5 审计] S7 TCP 限速（E8，serve 7 m/s · cap 1.0）：")
    names = {"TCP1.0|softmin_only_vs_none": "candidate set only",
             "TCP1.0|full_vs_none": "full",
             "TCP1.0|tube_only_vs_none": "tube only"}
    for k, g in d["s7_paired"].items():
        print(f"  {names[k]}: gain={fmt_pp(g['diff_pp'])} CI{fmt_ci(g['ci95'])}"
              f" {fmt_p(g['p_mcnemar'])} n={g['n_paired']}")
    print(f"[B5 审计] S8 实时（exp18 timing.json + exp23 重算）："
          f"first={r1(d['s8_timing']['first']['mean_ms'])} ms"
          f" steady median={r0(d['s8_timing']['steady']['median_ms'])} ms"
          f" p95={r0(d['s8_timing']['steady']['p95_ms'])} ms; nofreeze: "
          + ", ".join(f"{it}-iter {r0(v['median_ms'])} ms→{r1(v['rate'])}% (n={v['n']})"
                      for it, v in sorted(nofreeze.items())))
    return d


# ── S5：标称增益图（12 s）───────────────────────────────────────────────────
S5_DUR = 12.0


def scene_s5(ax, t: float, d: dict) -> None:
    """双组柱：9/12 m/s 的 pt vs candidate-set-only + 配对增益箭头。"""
    # y0/sc：基线与类目标签须让开成片字幕带（588–654 px），assemble.py 约定
    y0, sc = 17.0, 0.36
    a_ax = seg(t, 0.5, 1.5)
    if a_ax > 0:
        ax.plot([16, 110], [y0, y0], color=rgba(INK, a_ax), lw=1.6)
        for rv in (0, 50, 100):
            ax.plot([15, 16], [y0 + rv * sc, y0 + rv * sc], color=rgba(INK, a_ax), lw=1.4)
            label(ax, 14, y0 + rv * sc, str(rv), a_ax, size=10.5, color=INK_2, ha="right")
        label(ax, 16, 61.0, "active-hit (%)", a_ax, size=11, color=INK_2)
    # 图例
    for i, (col, name) in enumerate(((GREY, "point target"),
                                     (ORANGE, "candidate set only"))):
        a = seg(t, 1.5 + 0.3 * i, 2.3 + 0.3 * i)
        ax.add_patch(Circle((101.5, 60.5 - 3.0 * i), 1.0, fc=rgba(col, a), ec="none"))
        label(ax, 104, 60.5 - 3.0 * i, name, a, size=11.5)

    groups = [
        (26.0, "9 m/s serve", "9", 2.2, 6.0, 7.4),
        (74.0, "12 m/s serve", "12", 3.0, 6.8, 8.6),
    ]
    for gx, gname, sp, t_bar, t_val, t_gain in groups:
        pt = d["s5_cells"][f"{sp}|none"]
        cd = d["s5_cells"][f"{sp}|softmin_only"]
        pair = d["s5_paired"][f"{sp}|softmin_only_vs_none"]
        # 柱（生长）
        for i, (cell, col) in enumerate(((pt, GREY), (cd, ORANGE))):
            a = seg(t, t_bar + 0.3 * i, t_bar + 1.6 + 0.3 * i)
            if a <= 0:
                continue
            h = a * cell["rate"] * sc
            x = gx + 11.0 * i
            ax.add_patch(Rectangle((x, y0), 9.0, h, fc=rgba(col, 0.88), ec="none"))
        # CI + 数值标签
        a_v = seg(t, t_val, t_val + 1.2)
        if a_v > 0:
            for i, cell in enumerate((pt, cd)):
                x = gx + 11.0 * i + 4.5
                ytop = y0 + cell["rate"] * sc
                ci0, ci1 = cell["ci95"]
                ax.plot([x, x], [y0 + ci0 * sc, y0 + ci1 * sc],
                        color=rgba(INK, 0.65 * a_v), lw=1.4)
                for yy in (ci0, ci1):
                    ax.plot([x - 1.1, x + 1.1], [y0 + yy * sc, y0 + yy * sc],
                            color=rgba(INK, 0.65 * a_v), lw=1.4)
                label(ax, x, max(ytop, y0 + ci1 * sc) + 1.8, r1(cell["rate"]), a_v,
                      size=13, weight="bold", ha="center")
        label(ax, gx + 9.5, y0 - 2.2, gname, seg(t, t_bar, t_bar + 0.8), size=13.5,
              ha="center", weight="bold")
        # 配对增益箭头（标签放组右侧空白区，避免压柱）
        a_g = seg(t, t_gain, t_gain + 1.4)
        if a_g > 0:
            x0, y_top = gx + 9.0, y0 + pt["rate"] * sc + 1.0
            x1, y1 = gx + 11.0, y0 + cd["rate"] * sc + 1.0
            arrow(ax, (x0, y_top), (x1, y1), a_g, color=GREEN, lw=2.4, rad=-0.35, ms=15)
            y_mid = (y_top + y1) / 2
            label(ax, gx + 23.0, y_mid + 1.2, f"{fmt_pp(pair['diff_pp'])} pp",
                  a_g, size=15, color=GREEN, weight="bold")
            label(ax, gx + 23.0, y_mid - 1.4,
                  f"CI {fmt_ci(pair['ci95'])} · {fmt_p(pair['p_mcnemar'])}",
                  a_g, size=10, color=INK_2)


# ── S6：扰动网格（15 s）─────────────────────────────────────────────────────
S6_DUR = 15.0


def scene_s6(ax, t: float, d: dict) -> None:
    """左：共享偏差示意 + 20 格网格；右：cand 82.8% vs pt 73.9% + 增益。"""
    # 左上：真值弹道 vs 受扰预测（示意）
    a_tr = seg(t, 0.5, 2.5)
    if a_tr > 0:
        us = np.linspace(0, 1, 60)
        true = np.array([(10 + 46 * u, 42 + 16 * np.sin(np.pi * u) - 6 * u)
                         for u in us])
        ax.plot(true[:, 0], true[:, 1], color=rgba(GREY, a_tr), lw=2.2)
        bias = seg(t, 2.5, 4.0)
        if bias > 0:
            pred = true + np.array([4.5, 3.0])
            ax.plot(pred[:, 0], pred[:, 1], color=rgba(ORANGE, bias), lw=2.0,
                    ls=(0, (4, 3)))
            # 偏差箭头两端各加一个标注点（用户 2026-09-21：起点=真值，终点=受扰预测）
            u_a = 0.42
            pa = np.array([10 + 46 * u_a,
                           42 + 16 * np.sin(np.pi * u_a) - 6 * u_a])
            pb = pa + np.array([4.5, 3.0])
            arrow(ax, tuple(pa), tuple(pb), bias, color=ORANGE, lw=2.8, ms=20)
            ax.add_patch(Circle(pa, 0.9, fc=rgba(INK, bias), ec="none"))
            ax.add_patch(Circle(pb, 0.9, fc=rgba(ORANGE, bias), ec="none"))
            label(ax, pa[0] + 1.2, pa[1] - 3.0, "true", bias, size=12.5, color=INK_2,
                  ha="left")
            label(ax, pb[0] + 3.7, pb[1] + 1.9, "biased prediction", bias, size=12.5,
                  color=ORANGE)
            label(ax, 23.5, 49.5, "shared prediction bias", bias, size=14,
                  color=ORANGE, weight="bold")
    # 左下：20 格网格（列 = 时间偏差 t，行 = 空间偏差 s，轴值从 JSON 解析）
    # 格色按偏置强度橙色渐变（用户粗剪反馈：灰格辨识度低）；
    # gy0/行距/格高按成片字幕带上缘（616 px）收紧，assemble.py 约定
    a_grid = seg(t, 5.0, 6.8)
    if a_grid > 0:
        ts, ss = d["s6_axes"]
        gx0, gy0, cw, chh, px_, py_ = 10.0, 17.6, 8.4, 3.2, 9.2, 3.8
        n_cell = len(ts) * len(ss)
        t_max = max(abs(v) for v in ts)
        s_max = max(abs(v) for v in ss)
        for r in range(len(ss)):
            for c in range(len(ts)):
                a = seg(t, 5.0 + 0.05 * (r * len(ts) + c),
                        5.6 + 0.05 * (r * len(ts) + c))
                mag = 0.5 * (abs(ts[c]) / t_max + abs(ss[r]) / s_max)
                ax.add_patch(Rectangle((gx0 + c * px_, gy0 + r * py_), cw, chh,
                                       fc=rgba(ORANGE, (0.10 + 0.60 * mag) * a),
                                       ec=rgba(INK_2, 0.6 * a), lw=1.0))
        # 列刻度（t, ms）
        for c, tv in enumerate(ts):
            label(ax, gx0 + c * px_ + cw / 2, gy0 - 2.0, str(tv), a_grid, size=12,
                  color=INK_2, ha="center")
        # 行刻度（s, m）
        for r, sv in enumerate(ss):
            label(ax, gx0 - 1.7, gy0 + r * py_ + chh / 2, f"{sv:g}", a_grid,
                  size=12, color=INK_2, ha="right")
        # 轴题
        label(ax, gx0 + (len(ts) - 1) * px_ / 2 + cw / 2, gy0 - 4.4,
              "temporal bias t (ms)", a_grid, size=12.5, color=INK, ha="center")
        ax.text(3.0, gy0 + 5.8, "spatial bias s (m)", fontsize=12.5, rotation=90,
                color=rgba(INK, a_grid), ha="center", va="center")
        label(ax, 10, 34.8,
              f"merged {n_cell}-cell bias grid · 9 m/s · deeper = stronger bias",
              a_grid, size=12.5, color=INK_2)

    # 右：两根柱
    y0, sc = 16.0, 0.42
    cand, pt, pair = d["s6_cand"], d["s6_pt"], d["s6_paired"]
    a_ax = seg(t, 7.0, 8.0)
    if a_ax > 0:
        ax.plot([72, 120], [y0, y0], color=rgba(INK, a_ax), lw=1.6)
    for i, (cell, col, name) in enumerate(((pt, GREY, "point target"),
                                           (cand, ORANGE, "candidate set only"))):
        a = seg(t, 7.5 + 0.5 * i, 9.3 + 0.5 * i)
        if a <= 0:
            continue
        x = 78 + 14.0 * i
        ax.add_patch(Rectangle((x, y0), 10.0, a * cell["rate"] * sc,
                               fc=rgba(col, 0.88), ec="none"))
        a_v = seg(t, 10.0 + 0.4 * i, 11.0 + 0.4 * i)
        label(ax, x + 5, y0 + cell["rate"] * sc + 1.8, f"{r1(cell['rate'])}%", a_v,
              size=15.5, weight="bold", ha="center")
        label(ax, x + 5, y0 - 2.4, name, a, size=12.5, ha="center")
    a_g = seg(t, 11.6, 13.2)
    if a_g > 0:
        ytop = y0 + pt["rate"] * sc + 1.0
        y1 = y0 + cand["rate"] * sc + 1.0
        arrow(ax, (94.0, ytop), (100.0, y1), a_g, color=GREEN, lw=2.4, rad=-0.3, ms=15)
        label(ax, 105.0, 52.6, f"{fmt_pp(pair['diff_pp'])} pp", a_g, size=16.5,
              color=GREEN, weight="bold")
        label(ax, 105.0, 50.2, f"CI {fmt_ci(pair['ci95'])}", a_g, size=11.5,
              color=INK_2)
    label(ax, 72, 36.0, "the nominal gain partially persists", seg(t, 13.4, 14.6),
          size=15, weight="bold")


# ── S7：TCP 限速（27 s）─────────────────────────────────────────────────────
S7_DUR = 27.0
ZERO_X, PP_Scale = 78.0, 2.9


def scene_s7(ax, t: float, d: dict) -> None:
    """三行增益条（vs point target）：candidate −11.9 / full −12.7 / tube +5.8。"""
    a_ax = seg(t, 0.5, 2.0)
    if a_ax > 0:
        ax.plot([ZERO_X, ZERO_X], [21, 60], color=rgba(INK, 0.6 * a_ax), lw=1.8)
        label(ax, ZERO_X, 18.6, "0 pp", a_ax, size=12.5, color=INK_2, ha="center")
    rows = [
        # 行名前不放身份色块（用户 2026-09-21 反馈：色块与条形颜色无对应，反而误导）；
        # 条形颜色只编码增益符号（红负/绿正）
        ("TCP1.0|softmin_only_vs_none", "candidate set only",
         "TCP1.0|softmin_only", 2.5, 52.0),
        ("TCP1.0|full_vs_none", "full", "TCP1.0|full", 7.5, 40.0),
        ("TCP1.0|tube_only_vs_none", "tube only", "TCP1.0|tube_only", 12.5, 28.0),
    ]
    for key, name, abs_key, t0, y in rows:
        g = d["s7_paired"][key]
        a_name = seg(t, t0, t0 + 0.8)
        label(ax, 4.5, y, name, a_name, size=15, weight="bold")
        # 增益条（红负绿正）；数值标签放条形上方，避让 CI 横线
        a_bar = seg(t, t0 + 0.6, t0 + 2.0)
        if a_bar > 0:
            x_end = ZERO_X + g["diff_pp"] * PP_Scale * a_bar
            col = RED if g["diff_pp"] < 0 else GREEN
            ax.add_patch(Rectangle((min(ZERO_X, x_end), y - 1.7),
                                   abs(x_end - ZERO_X), 3.4, fc=rgba(col, 0.85),
                                   ec="none"))
            label(ax, x_end, y + 3.6, f"{fmt_pp(g['diff_pp'])} pp", a_bar, size=17,
                  color=col, ha="center", weight="bold")
        # CI + p
        a_ci = seg(t, t0 + 1.8, t0 + 3.0)
        if a_ci > 0:
            c0 = ZERO_X + g["ci95"][0] * PP_Scale
            c1 = ZERO_X + g["ci95"][1] * PP_Scale
            ax.plot([c0, c1], [y, y], color=rgba(INK, 0.7 * a_ci), lw=1.6)
            for cx in (c0, c1):
                ax.plot([cx, cx], [y - 1.0, y + 1.0], color=rgba(INK, 0.7 * a_ci),
                        lw=1.6)
            label(ax, (c0 + c1) / 2, y - 3.2,
                  f"CI {fmt_ci(g['ci95'])} · {fmt_p(g['p_mcnemar'])}", a_ci, size=12,
                  color=INK_2, ha="center")
        # 绝对率对照（右侧留白大，字号加大——用户 2026-09-21 反馈）
        a_abs = seg(t, t0 + 2.4, t0 + 3.4)
        label(ax, 124, y + 2.1, f"{r1(d['s7_abs'][abs_key])}%", a_abs, size=17,
              ha="right", weight="bold")
        label(ax, 124, y - 2.1,
              f"vs {r1(d['s7_abs']['TCP1.0|none'])}% pt", a_abs, size=13.5,
              color=INK_2, ha="right")
    label(ax, 124, 57.8, "rate vs pt", seg(t, 2.5, 3.5), size=13.5, color=INK_2,
          ha="right")
    # 底部结论行上移出字幕带（用户粗剪反馈：原 y=11.5 被成片字幕遮挡）
    label(ax, 64, 13.5, "under a tight speed cap, widening the target does not help",
          seg(t, 17.5, 19.5), size=17, weight="bold", ha="center")


# ── S8：实时时间轴（21 s）───────────────────────────────────────────────────
S8_DUR = 21.0


def scene_s8(ax, t: float, d: dict) -> None:
    """飞行时间轴 + 首计划冻结括号 + 预算量表 + 不冻结对照 chips。

    布局（用户粗剪反馈：整段放大一号；chips 底缘 ≤ 场景 y 10.8 = 成片 616 px
    安全线之上，避让单行字幕带 622–652 px）。
    """
    tim = d["s8_timing"]
    first_ms = tim["first"]["mean_ms"]
    med = tim["steady"]["median_ms"]
    p95 = tim["steady"]["p95_ms"]
    nf30 = d["s8_nofreeze"].get(30) or next(iter(d["s8_nofreeze"].values()))
    nf5 = d["s8_nofreeze"].get(5)

    # 飞行轴（示意 1.1 s @7 m/s）
    a_ax = seg(t, 0.5, 2.0)
    if a_ax > 0:
        ax.plot([8, 120], [50, 50], color=rgba(INK, a_ax), lw=2.0)
        for frac, lab in ((0.0, "0"), (0.5, "0.5 s"), (1.0, "1.0 s")):
            x = 8 + 112 * frac
            ax.plot([x, x], [49.0, 51.0], color=rgba(INK, a_ax), lw=1.8)
            label(ax, x, 45.0, lab, a_ax, size=12, color=INK_2, ha="center")
        label(ax, 8, 62.5, "serve flight (schematic, ≈1.1 s at 7 m/s)", a_ax,
              size=13, color=INK_2)
    # 首计划冻结括号
    a_fp = seg(t, 2.0, 4.0)
    if a_fp > 0:
        w_b = a_fp * (first_ms / 1100.0) * 112.0
        ax.add_patch(Rectangle((8, 52.4), w_b, 4.6, fc=rgba(BLUE, 0.16),
                               ec=rgba(BLUE, 0.8 * a_fp), lw=1.6, ls=(0, (4, 2))))
        if a_fp > 0.95:
            label(ax, 8 + w_b / 2, 54.7,
                  f"first plan · {r1(first_ms)} ms — simulated time frozen",
                  seg(t, 4.0, 5.2), size=15, color=BLUE, ha="center", weight="bold")
    # 稳态重规划刻度（贴飞行轴下缘）
    a_tick = seg(t, 6.0, 8.0)
    if a_tick > 0:
        n_tick = int(11 * a_tick) + 1
        for i in range(n_tick):
            x = 18.2 + i * 10.2
            if x > 120:
                break
            ax.plot([x, x], [47.6, 50.0], color=rgba(GREEN, 0.9), lw=3.0)
        label(ax, 64, 42.2,
              f"steady-state replans · {r0(med)} ms each (p95 {r0(p95)} ms)",
              seg(t, 7.0, 8.4), size=14.5, color=GREEN, ha="center", weight="bold")
    # 预算量表
    a_bg = seg(t, 10.0, 11.5)
    if a_bg > 0:
        ax.add_patch(Rectangle((8, 32.0), 100, 4.4, fc=rgba(WHITE, 0.9),
                               ec=rgba(INK, 0.5 * a_bg), lw=1.6))
        a_fill = seg(t, 11.5, 12.5)
        ax.add_patch(Rectangle((8, 32.0), a_fill * med / BUDGET_MS * 100, 4.4,
                               fc=rgba(GREEN, 0.9), ec="none"))
        label(ax, 13.5, 34.2, f"{r0(med)} ms", a_fill, size=14.5, color=GREEN,
              weight="bold")
        x95 = 8 + p95 / BUDGET_MS * 100
        a_p = seg(t, 12.5, 13.5)
        ax.plot([x95, x95], [30.6, 37.8], color=rgba(INK, 0.7 * a_p), lw=1.8)
        label(ax, x95, 28.9, f"p95 {r0(p95)} ms", a_p, size=12.5, color=INK_2,
              ha="center")
        ax.plot([108, 108], [30.8, 37.6], color=rgba(INK, a_bg), lw=2.0,
                ls=(0, (4, 2)))
        label(ax, 107, 38.9, f"budget {r0(BUDGET_MS)} ms", a_bg, size=13,
              ha="right")
    # 不冻结对照
    a_cap = seg(t, 14.0, 15.0)
    label(ax, 8, 25.6, "unfrozen end-to-end: the first plan competes with the swing",
          a_cap, size=14, color=INK_2)
    chips = [
        (8.0, RED, "no freeze · 30-iter first plan",
         f"{r0(nf30['median_ms'])} ms → {r1(nf30['rate'])}% hits",
         f"n = {nf30['n']:,} episodes", 15.5),
        (67.0, GREEN, "cap warm-up · 5-iter first plan",
         f"{r0(nf5['median_ms'])} ms → {r1(nf5['rate'])}% hits",
         f"n = {nf5['n']:,} (9/12 m/s · 4 tiers · 98 seeds)", 17.0),
    ]
    for cx, col, head, big, sub, t0 in chips:
        a = seg(t, t0, t0 + 1.4)
        if a <= 0:
            continue
        ax.add_patch(FancyBboxPatch((cx, 10.8), 53.0, 13.2,
                                    boxstyle="round,pad=0.5,rounding_size=1.0",
                                    fc=rgba(col, 0.07 * a), ec=rgba(col, 0.9 * a),
                                    lw=2.2))
        label(ax, cx + 3.0, 21.6, head, a, size=13.5, color=INK_2)
        label(ax, cx + 3.0, 17.2, big, a, size=18, color=col, weight="bold")
        label(ax, cx + 3.0, 13.3, sub, a, size=12, color=INK_2)


# ── 渲染管线 ────────────────────────────────────────────────────────────────
def build_clips(d: dict) -> dict:
    """各段（时长，场景函数，文件名，标题，脚注，角标，QC 帧时刻）。"""
    return {
        "s5": (S5_DUR, lambda ax, t: scene_s5(ax, t, d), "s5_chart",
               "The candidate window captures the nominal gain",
               ("9 & 12 m/s serve · paired seeds (n = 98/cell) · active-hit rate ·"
                " gain = paired diff over point target"),
               "data: exp17d/E3",
               [1.0, 3.6, 5.6, 7.0, 8.4, 11.6]),
        "s6": (S6_DUR, lambda ax, t: scene_s6(ax, t, d), "s6_perturb",
               "Under shared prediction bias",
               "shared prediction bias · merged 20-cell grid · 9 m/s · active-hit rate",
               "data: exp17b/E4",
               [1.6, 3.4, 6.0, 9.0, 12.4, 14.6]),
        "s7": (S7_DUR, lambda ax, t: scene_s7(ax, t, d), "s7_limit",
               "Under a tight speed cap (TCP 1.0 m/s)",
               ("serve 7 m/s · TCP cap 1.0 m/s · n = 394 paired ·"
                " gain over point target (pp)"),
               "data: exp17i/E8",
               [1.4, 4.2, 9.2, 14.2, 18.6, 26.6]),
        "s8": (S8_DUR, lambda ax, t: scene_s8(ax, t, d), "s8_timing",
               "Does replanning meet the runtime budget?",
               "real-time loop · replan budget 150 ms · active-hit rate",
               "data: exp18 + exp23",
               [1.4, 4.6, 7.6, 12.0, 16.4, 20.6]),
    }


def render_frame(scene, footer: str, title: str, tag: str, t: float) -> np.ndarray:
    """渲染单帧 → RGB uint8。"""
    fig, ax = new_scene(title, seg(t, 0.2, 1.4), footer, tag)
    scene(ax, t)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def render_clip(key: str, spec: tuple, fps: int) -> Path:
    """渲染一段 → mp4 + QC sheet。"""
    from PIL import Image

    dur, scene, name, title, footer, tag, times = spec
    out = OUT_DIR / f"{name}.mp4"
    n_frames = round(dur * fps)
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=None,
                            bitrate=None, macro_block_size=16,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "17",
                                           "-preset", "medium"]) as wr:
        for k in range(n_frames):
            wr.append_data(render_frame(scene, footer, title, tag, k / fps))
            if k % 150 == 0:
                logger.info("  %s: %d/%d 帧", key, k, n_frames)
    tiles = [render_frame(scene, footer, title, tag, min(tt, dur - 0.05))
             for tt in times]
    th, tw = tiles[0].shape[:2]
    sw, sh = tw // 3, th // 3
    sheet = Image.new("RGB", (sw * 3, sh * 2), (24, 24, 24))
    for i, tile in enumerate(tiles):
        img = Image.fromarray(tile).resize((sw, sh))
        sheet.paste(img, ((i % 3) * sw, (i // 3) * sh))
    sheet.save(OUT_DIR / f"{name}_sheet.png")
    logger.info("[done] %s → %s (%.1f s, %d 帧)", key, out.name, dur, n_frames)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["s5", "s6", "s7", "s8"], help="只渲染指定段")
    ap.add_argument("--fps", type=int, default=FPS_DEFAULT)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    d = load_data()
    clips = build_clips(d)
    keys = [args.only] if args.only else list(clips)
    for k in keys:
        render_clip(k, clips[k], args.fps)


if __name__ == "__main__":
    main()
