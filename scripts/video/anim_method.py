#!/usr/bin/env python3
"""方法段动画生成器：S2（MPC+iLQR 框架）/ S3（候选窗）/ S4（Tube 走廊）。

设计约束（paper/video/plan.md §3/§4）：
- 画布 1280x720 @ 30 fps，浅底工业风；配色 灰(基线)/橙(候选窗)/蓝(Tube)/绿(带内/收益)。
- 静态矢量逐项出现（reveal），**无追逐候选、无显式搜索触达时刻、无对勾红线**（B1 红线）。
- 屏显常量（±50 ms 窗、0.12 m 走廊半宽、softmin β）运行时从
  ``src.ilqt.tube_types.TubeConfig`` import，不手抄（B5 溯源）。
- 右下角 "Schematic" 角标：明示示意图非仿真画面（诚实呈现）。

用法：
    python scripts/video/anim_method.py                 # 渲染全部三段
    python scripts/video/anim_method.py --only s3       # 只渲染 S3
    python scripts/video/anim_method.py --fps 10        # 快速草稿
"""
from __future__ import annotations

import argparse
import logging
import sys
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
    Polygon,
    Rectangle,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.ilqt.tube_types import TubeConfig  # 屏显常量的唯一事实来源

OUT_DIR = REPO / "paper" / "video" / "assets"
FPS_DEFAULT = 30
W, H = 128.0, 72.0  # 画布逻辑坐标（1 单位 = 10 px）

logger = logging.getLogger("anim_method")

# ── 配色（0-1 RGB，与全片统一）───────────────────────────────────────────────
INK = (0.13, 0.14, 0.17)        # 近黑：主文字
INK_2 = (0.19, 0.20, 0.25)      # 次文字（用户两轮反馈逐次调深：0.42→0.27→0.19）
BG = (0.965, 0.961, 0.949)      # 暖白底
GREY = (0.58, 0.58, 0.62)       # 基线/点目标
ORANGE = (0.878, 0.482, 0.122)  # 候选窗
BLUE = (0.184, 0.427, 0.710)    # Tube
GREEN = (0.184, 0.620, 0.373)   # 带内 / 收益
RED = (0.765, 0.290, 0.212)     # 带外 / 受限
WHITE = (1.0, 1.0, 1.0)
STRIP = (0.137, 0.145, 0.165)   # 底部条带


# ── 基础工具 ────────────────────────────────────────────────────────────────
def seg(t: float, a: float, b: float) -> float:
    """阶段进度 0→1，smoothstep 缓动（t<a 为 0，t>b 为 1）。"""
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0
    u = (t - a) / (b - a)
    return u * u * (3.0 - 2.0 * u)


def rgba(c: tuple, a: float) -> tuple:
    """颜色 + 透明度合成 matplotlib rgba。"""
    return (c[0], c[1], c[2], max(0.0, min(1.0, a)))


def new_scene(title: str, title_a: float, footer: str, tag: str = "Schematic"):
    """建画布：底色 + 标题 + 底部条件条，返回 (fig, ax)。"""
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc=BG, ec="none"))
    ax.text(6, 66.2, title, fontsize=21, fontweight="bold", color=rgba(INK, title_a),
            ha="left", va="center")
    # 底部条件条
    ax.add_patch(Rectangle((0, 0), W, 6.2, fc=STRIP, ec="none", alpha=0.96))
    ax.text(6, 3.1, footer, fontsize=14.5, color=(1, 1, 1, 0.94), ha="left", va="center")
    ax.text(W - 6, 3.1, tag, fontsize=12.5, color=(1, 1, 1, 0.55), ha="right", va="center")
    return fig, ax


def block(ax, x, y, w, h, label_txt, sub, a, edge=GREY):
    """圆角流程块：白底 + 描边 + 主/副标签。"""
    if a <= 0:
        return
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.55,rounding_size=1.1",
                                fc=rgba(WHITE, 0.92 * a), ec=rgba(edge, a), lw=2.0))
    ax.text(x + w / 2, y + h * (0.66 if sub else 0.5), label_txt, fontsize=16.5,
            fontweight="bold", color=rgba(INK, a), ha="center", va="center")
    if sub:
        ax.text(x + w / 2, y + h * 0.30, sub, fontsize=12.5,
                color=rgba(INK_2, a), ha="center", va="center")


def arrow(ax, p0, p1, a, color=INK_2, lw=2.2, style="-|>", ms=16):
    """直线箭头。"""
    if a <= 0:
        return
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 color=rgba(color, a), lw=lw, shrinkA=0, shrinkB=0))


def label(ax, x, y, s, a, size=13, color=INK, ha="left", va="center", weight="normal"):
    """带透明度的文字。"""
    if a <= 0:
        return
    ax.text(x, y, s, fontsize=size, color=rgba(color, a), ha=ha, va=va,
            fontweight=weight)


# ── 来球抛物线（S3/S4 共用；物理小世界 → 画布映射）─────────────────────────
# 世界坐标：x∈[0.5, 7.3] m，y = 2.3 − 0.153·(x−4.2)²（顶点 (4.2, 2.3)，纯示意）
# 画布映射：X = 4 + 14.2·x，Y = 10 + 20·y（顶点 (63.6, 56)，触达点约 (102, 33)）
X0_W, X1_W = 0.5, 7.3
T_HIT = 0.945   # 触达点参数（x≈6.93 m）
CAND_DU = 0.024  # 候选点参数间隔（视觉口径，物理含义 = ±50 ms 窗）


def ball_pos(u: float) -> np.ndarray:
    """参数 u∈[0,1] → 画布坐标点。"""
    x = X0_W + u * (X1_W - X0_W)
    y = 2.3 - 0.153 * (x - 4.2) ** 2
    return np.array([4.0 + 14.2 * x, 10.0 + 20.0 * y])


def ball_tan(u: float) -> np.ndarray:
    """参数 u 处切向（画布系，数值微分，归一化）。"""
    e = 1e-3
    d = ball_pos(min(u + e, 1.0)) - ball_pos(max(u - e, 0.0))
    return d / np.linalg.norm(d)


def draw_trajectory(ax, u_max: float, a: float, n_dot: int = 26) -> None:
    """灰弹道曲线 + 采样点（u_max 控制画到多远，用于 draw-on 揭示）。"""
    if a <= 0:
        return
    us = np.linspace(0.0, 1.0, 121)
    pts = np.array([ball_pos(u) for u in us])
    m = us <= u_max
    ax.plot(pts[m, 0], pts[m, 1], color=rgba(GREY, 0.9 * a), lw=2.4, zorder=2)
    for ud in np.linspace(0.02, min(u_max, 0.99), n_dot):
        ax.add_patch(Circle(ball_pos(ud), 0.55, fc=rgba(GREY, 0.7 * a), ec="none",
                            zorder=2))
    arrow(ax, tuple(ball_pos(0.0)), tuple(ball_pos(0.0) + ball_tan(0.0) * 3.4), a,
          color=GREY, lw=2.2)


# ── S2：MPC + iLQR 框架（14 s）──────────────────────────────────────────────
S2_DUR = 14.0


def scene_s2(ax, t: float) -> None:
    """框架回路逐块出现 → 聚焦 iLQR → 两张代价卡 → 四档可切换 chips。"""
    blocks = [
        ("Perceive", "ball state", 8.0),
        ("Predict", "parabola", 36.0),
        ("Solve", "iLQR", 64.0),
        ("Execute", "safe filter", 92.0),
    ]
    bw, bh, by = 20.0, 10.5, 46.0
    for i, (name, sub, bx) in enumerate(blocks):
        a = seg(t, 0.8 + 0.6 * i, 2.0 + 0.6 * i)
        hot = (name == "Solve") and t > 5.4  # 聚焦阶段 Solve 描边转橙
        block(ax, bx, by, bw, bh, name, sub, a, edge=ORANGE if hot else GREY)
        arrow(ax, (bx + bw + 1.0, by + bh / 2), (bx + 27.0, by + bh / 2),
              seg(t, 2.6 + 0.4 * i, 3.4 + 0.4 * i))

    # 回环箭头（receding horizon，走顶部避免与下方卡片打架）
    a_loop = seg(t, 4.0, 5.2)
    if a_loop > 0:
        y_loop = 60.0
        ax.plot([104, 104, 18, 18], [by + bh + 1.0, y_loop, y_loop, by + bh + 1.0],
                color=rgba(INK_2, a_loop), lw=1.8, ls=(0, (5, 3)))
        arrow(ax, (18, y_loop), (18, by + bh + 1.2), a_loop)
        label(ax, 61, y_loop + 2.0, "receding horizon: replan, shift, repeat",
              a_loop, size=14, color=INK_2, ha="center")

    # 聚焦圈
    a_focus = seg(t, 5.4, 6.4)
    if a_focus > 0:
        ax.add_patch(Circle((74, by + bh / 2), 13.0, fill=False,
                            ec=rgba(ORANGE, 0.85 * a_focus), lw=2.0, ls=(0, (4, 3))))

    # 两张代价卡（我们改动的两项）
    cards = [
        (13.0, "① Terminal hit cost",
         "point target → candidate set + softmin", ORANGE, 6.2),
        (71.0, "② Running costs",
         "effort · smoothness · + tube hinge", BLUE, 7.4),
    ]
    for cx, head, sub, col, t0 in cards:
        a = seg(t, t0, t0 + 1.4)
        if a <= 0:
            continue
        rise = (1 - a) * 3.0
        ax.add_patch(FancyBboxPatch((cx, 22.5 - rise), 44.0, 11.5,
                                    boxstyle="round,pad=0.6,rounding_size=1.2",
                                    fc=rgba(WHITE, 0.95 * a), ec=rgba(col, a), lw=2.4))
        ax.add_patch(Circle((cx + 3.4, 30.6 - rise), 1.05, fc=rgba(col, a), ec="none"))
        label(ax, cx + 6.0, 30.6 - rise, head, a, size=16.5, weight="bold")
        label(ax, cx + 3.2, 25.9 - rise, sub, a, size=13.5, color=INK_2)
        label(ax, cx + 3.2, 23.6 - rise, "ours — switchable", a * 0.9, size=12,
              color=col)
    # 连线：Solve 块 → 两张卡
    a_link = seg(t, 7.0, 7.8)
    if a_link > 0:
        ax.plot([74, 74, 57.0], [by - 1.0, 36.0, 34.8], color=rgba(ORANGE, 0.5 * a_link),
                lw=1.5, ls=(0, (3, 3)))
        ax.plot([74, 74, 93.0], [by - 1.0, 36.0, 34.8], color=rgba(BLUE, 0.5 * a_link),
                lw=1.5, ls=(0, (3, 3)))

    # 四档消融 chips（each independently switchable）
    chips = [("point target", GREY), ("candidate set", ORANGE),
             ("tube", BLUE), ("full", GREEN)]
    cw, ch, cy = 24.0, 4.6, 11.0
    for i, (name, col) in enumerate(chips):
        a = seg(t, 9.2 + 0.35 * i, 10.2 + 0.35 * i)
        if a <= 0:
            continue
        x0 = 13.0 + i * 26.0
        ax.add_patch(FancyBboxPatch((x0, cy), cw, ch,
                                    boxstyle="round,pad=0.4,rounding_size=0.9",
                                    fc=rgba(WHITE, 0.9 * a), ec=rgba(col, a), lw=1.8))
        ax.add_patch(Circle((x0 + 2.6, cy + ch / 2), 0.95, fc=rgba(col, a), ec="none"))
        label(ax, x0 + 4.8, cy + ch / 2, name, a, size=13.5, weight="bold")
    label(ax, 13, 18.6, "each independently switchable — the four ablation tiers",
          seg(t, 9.2, 10.6), size=14, color=INK_2)


# ── S3：候选窗（20 s，末 3 s 末态驻留供阅读——用户粗剪二审反馈）──────────────
S3_DUR = 20.0


def scene_s3(ax, t: float) -> None:
    """点目标 → 展开候选窗 → softmin 汇聚单一终端状态。"""
    u_draw = 0.01 + seg(t, 0.5, 3.5) * 0.99
    draw_trajectory(ax, u_draw, min(1.0, seg(t, 0.5, 2.0) * 3))
    p_hit = ball_pos(T_HIT)
    tan = ball_tan(T_HIT)
    nrm = np.array([-tan[1], tan[0]])

    # 阶段 1：灰色点目标十字（候选窗展开时淡出——点目标被"放宽"而非并存）
    a_pt = seg(t, 3.6, 4.8) * (1 - seg(t, 6.0, 7.0))
    if a_pt > 0:
        r = 2.6
        for dx, dy in [(-r, 0), (r, 0), (0, -r), (0, r)]:
            ax.plot([p_hit[0], p_hit[0] + dx], [p_hit[1], p_hit[1] + dy],
                    color=rgba(INK, a_pt), lw=1.8, zorder=4)
        ax.add_patch(Circle(p_hit, 1.05, fill=False, ec=rgba(INK, a_pt), lw=1.8,
                            zorder=4))
        label(ax, 88, 21.5, "one predicted contact state", a_pt, size=13, ha="right")
        label(ax, 88, 19.0, "(point target)", a_pt, size=12, color=INK_2, ha="right")

    # 阶段 2：展开为橙色候选窗
    a_win = seg(t, 6.0, 7.6)
    cand_us = [T_HIT + k * CAND_DU for k in (-2, -1, 0, 1, 2)]
    if a_win > 0:
        half_len = a_win * (2.5 * 2 / 2 + 4.0)
        c = [p_hit + tan * half_len + nrm * 3.0,
             p_hit - tan * half_len + nrm * 3.0,
             p_hit - tan * half_len - nrm * 3.0,
             p_hit + tan * half_len - nrm * 3.0]
        ax.add_patch(Polygon(c, closed=True, fc=rgba(ORANGE, 0.16 * a_win),
                             ec=rgba(ORANGE, 0.85 * a_win), lw=2.0, zorder=3))
    for i, cu in enumerate(cand_us):
        a = seg(t, 6.6 + 0.18 * i, 7.4 + 0.18 * i)
        if a <= 0:
            continue
        ax.add_patch(Circle(ball_pos(cu), 0.95, fc=rgba(ORANGE, a), ec="none",
                            zorder=5))
    label(ax, 102.4, 24.6, f"candidate window  ±{TubeConfig.window_half_ms:.0f} ms",
          a_win, size=14, color=ORANGE, ha="center", weight="bold")
    # 小时间尺（y 上移避开成片字幕带 588–654 px，assemble.py 约定）
    a_rul = seg(t, 8.0, 9.2)
    if a_rul > 0:
        y0, x_c, half = 15.5, 102.4, 13.0 * a_rul
        ax.plot([x_c - half, x_c + half], [y0, y0], color=rgba(INK_2, a_rul), lw=1.6)
        for k, lab in ((-1, "−50"), (0, "0"), (1, "+50")):
            xk = x_c + k * half
            ax.plot([xk, xk], [y0 - 0.8, y0 + 0.8], color=rgba(INK_2, a_rul), lw=1.6)
            label(ax, xk, y0 - 2.4, lab, a_rul, size=10.5, color=INK_2, ha="center")
        label(ax, x_c, y0 + 2.6, "time-to-contact (ms)", a_rul, size=10.5,
              color=INK_2, ha="center")

    # 阶段 3：softmin 汇聚到绿色终端状态（中心候选箭头最粗 = 权重最大）
    a_dm = seg(t, 10.0, 11.0)
    d_ctr = p_hit + nrm * 9.2
    if a_dm > 0:
        s = 2.0 + (1 - a_dm) * 2.0
        ax.add_patch(Polygon([d_ctr + (0, s), d_ctr + (s * 0.7, 0),
                              d_ctr - (0, s), d_ctr - (s * 0.7, 0)],
                             closed=True, fc=rgba(GREEN, a_dm), ec="none", zorder=6))
        label(ax, d_ctr[0] + 3.4, d_ctr[1], "terminal state", a_dm, size=13,
              color=GREEN, weight="bold")
    a_ar = seg(t, 11.2, 13.0)
    if a_ar > 0:
        w_lw = [1.2, 2.0, 3.4, 2.0, 1.2]  # softmin 权重 → 箭头粗细
        for i, (cu, lwi) in enumerate(zip(cand_us, w_lw)):
            ai = seg(t, 11.2 + 0.25 * i, 12.0 + 0.25 * i)
            if ai <= 0:
                continue
            # 深墨色实线（橙底上的橙箭头不可辨，用户 2026-09-21 反馈）
            arrow(ax, tuple(ball_pos(cu) + nrm * 1.4),
                  tuple(d_ctr - nrm * 2.4), ai, color=INK, lw=lwi, ms=9 + 4 * lwi)
        label(ax, 12, 46.0, "softmin over candidates", a_ar, size=13.5, weight="bold")
        label(ax, 12, 43.6,
              f"(β = {TubeConfig.softmin_beta:.0f})  closest candidate dominates",
              a_ar, size=11.5, color=INK_2)
    label(ax, 6, 61.5, "Relax the point target → a window of candidates,"
          " one aggregated terminal goal", seg(t, 14.2, 15.6), size=13.5, color=INK)


# ── S4：Tube 走廊（13 s）────────────────────────────────────────────────────
S4_DUR = 13.0
HALF_W = 2.6  # 走廊视觉半宽（对应 corridor_radius = 0.12 m）


def scene_s4(ax, t: float) -> None:
    """蓝 Tube 沿弹道 → 带内绿/带外红探针 → hinge 内嵌小图。"""
    draw_trajectory(ax, 0.01 + seg(t, 0.3, 2.2) * 0.99,
                    min(1.0, seg(t, 0.3, 1.6) * 3))

    # 走廊带（沿法向偏移 ±HALF_W）
    a_band = seg(t, 1.6, 3.6)
    if a_band > 0:
        us = np.linspace(0.02, 0.99, 80)
        up, dn = [], []
        for u in us:
            nm = np.array([-ball_tan(u)[1], ball_tan(u)[0]])
            c = ball_pos(u)
            up.append(c + nm * HALF_W)
            dn.append(c - nm * HALF_W)
        up, dn = np.array(up), np.array(dn)
        ax.add_patch(Polygon(np.vstack([up, dn[::-1]]), closed=True,
                             fc=rgba(BLUE, 0.14 * a_band), ec="none", zorder=1))
        ax.plot(up[:, 0], up[:, 1], color=rgba(BLUE, 0.75 * a_band), lw=1.8)
        ax.plot(dn[:, 0], dn[:, 1], color=rgba(BLUE, 0.75 * a_band), lw=1.8)
        # 标签放弹道起点右下方空白区（原 (12, 15.5) 被走廊左下腿穿过——
        # 用户粗剪二审反馈；仍保持场景 y ≥ 11.55 在成片字幕带 622 px 之上）
        label(ax, 13.5, 12.0,
              f"tube · half-width {TubeConfig.corridor_radius:.2f} m (racket radius)",
              a_band, size=16.5, color=BLUE, weight="bold")

    # 探针点：带内绿（代价 0）、带外红（hinge 随偏离增长）
    # 带外偏移加大 + 箭头加粗（用户 2026-09-21 反馈：原箭头太短太细不明显）
    probes = [
        (0.30, 0.0, True, 4.6), (0.62, -1.3, True, 5.1), (0.86, 1.6, True, 5.6),
        (0.42, 6.2, False, 6.1), (0.74, -6.0, False, 6.6),
    ]
    for u, off, inside, t0 in probes:
        a = seg(t, t0, t0 + 0.9)
        if a <= 0:
            continue
        nm = np.array([-ball_tan(u)[1], ball_tan(u)[0]])
        p = ball_pos(u) + nm * off
        col = GREEN if inside else RED
        ax.add_patch(Circle(p, 1.0, fc=rgba(col, a), ec="none", zorder=5))
        if inside:
            if off > 0.5:  # 上内侧：标签放点上方（带外）
                label(ax, p[0] + 1.6, p[1] + 3.3, "cost 0", a, size=13, color=GREEN)
            else:
                # 中/下内侧：标签锚到走廊下缘外侧（斜坡段水平文字会蹭进带内，
                # 用户两轮反馈）+ 细引导线表明归属
                base = ball_pos(u) - nm * (HALF_W + 3.5)
                ax.plot([p[0], base[0]], [p[1] - 1.4, base[1] + 1.5],
                        color=rgba(GREEN, 0.4 * a), lw=1.1)
                label(ax, base[0], base[1], "cost 0", a, size=13, color=GREEN,
                      ha="center")
        else:
            edge = ball_pos(u) + nm * (HALF_W * np.sign(off))
            start = p - nm * (1.6 * np.sign(off))  # 从圆点边缘起笔，不压点
            arrow(ax, tuple(start), tuple(edge), a, color=RED, lw=2.8, ms=22)
            if off > 0:  # 上方探针标注放左侧，避开标题区
                label(ax, p[0] - 2.5, p[1] + 1.7, "hinge ↑", a, size=13,
                      color=RED, ha="right")
            else:
                label(ax, p[0] + 2.0, p[1] - 3.1, "hinge ↑", a, size=13, color=RED)

    # 沿线松弛注释（不绑定时间-空间对应；放左上空白区）
    a_sl = seg(t, 8.6, 10.0)
    label(ax, 12, 58.4, "along-track slack:", a_sl, size=15.5, weight="bold")
    label(ax, 12, 55.4, "early / late along the line is free", a_sl,
          size=14, color=INK_2)

    # 内嵌 hinge 小图（右下，避开弹道）
    a_in = seg(t, 6.6, 8.4)
    if a_in <= 0:
        return
    ins = ax.inset_axes([0.615, 0.235, 0.345, 0.30])
    ins.set_facecolor((*WHITE, 0.97 * a_in))
    for sp in ins.spines.values():
        sp.set_color(rgba(INK_2, 0.6 * a_in))
        sp.set_linewidth(1.0)
    d = np.linspace(0, 0.40, 200)
    pen = 0.5 * np.maximum(0.0, d - TubeConfig.corridor_radius) ** 2
    pen = pen / pen.max() * 0.82
    ins.axvspan(0, TubeConfig.corridor_radius, color=rgba(GREEN, 0.10 * a_in))
    ins.plot(d, pen, color=rgba(RED, a_in), lw=2.6)
    ins.axvline(TubeConfig.corridor_radius, color=rgba(BLUE, 0.8 * a_in), lw=1.4,
                ls=(0, (3, 2)))
    ins.set_xlim(0, 0.40)
    ins.set_ylim(0, 1.0)
    ins.set_xticks([0, 0.12, 0.40])
    ins.set_xticklabels(["0", f"{TubeConfig.corridor_radius:.2f}", "0.40 m"])
    ins.set_yticks([])
    for tl in ins.get_xticklabels():
        tl.set_color(rgba(INK, a_in))
        tl.set_fontsize(13)
    # 标题放小图内部顶部（外置会被走廊下缘穿过；底色 0.97 已压掉透底）
    ins.text(0.62, 0.96, "lateral penalty (hinge)", transform=ins.transAxes,
             fontsize=14, color=rgba(INK, a_in), ha="center", va="top")
    ins.text(TubeConfig.corridor_radius / 2, 0.86, "free", fontsize=13.5,
             color=rgba(GREEN, a_in), ha="center")


# ── 渲染管线 ────────────────────────────────────────────────────────────────
CLIPS = {
    "s2": (S2_DUR, scene_s2, "s2_method"),
    "s3": (S3_DUR, scene_s3, "s3_window"),
    "s4": (S4_DUR, scene_s4, "s4_tube"),
}
FOOTERS = {
    "s2": "MPC + iLQR — we modify exactly two cost terms, each independently switchable",
    "s3": f"Terminal relaxation: one point target → a ±{TubeConfig.window_half_ms:.0f} ms"
          " candidate window, softmin-aggregated",
    "s4": "Tube around the predicted trajectory: free inside, hinge outside, lateral only",
}
TITLES = {
    "s2": "The loop: MPC + iLQR",
    "s3": "Relaxing the point target",
    "s4": "The tube: free inside, costly outside",
}
# QC 用关键帧时刻（每段 6 帧，3x2 sheet）
SHEET_TIMES = {
    "s2": [1.6, 3.0, 5.0, 8.0, 11.0, 13.6],
    "s3": [2.0, 4.6, 7.2, 9.0, 13.4, 16.4],
    "s4": [1.2, 3.2, 5.8, 7.6, 9.6, 12.6],
}


def render_frame(scene, footer: str, title: str, t: float) -> np.ndarray:
    """渲染单帧 → RGB uint8。"""
    fig, ax = new_scene(title, seg(t, 0.2, 1.4), footer)
    scene(ax, t)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def render_clip(key: str, fps: int) -> Path:
    """渲染一段动画 → mp4 + QC sheet。"""
    from PIL import Image

    dur, scene, name = CLIPS[key]
    footer, title = FOOTERS[key], TITLES[key]
    out = OUT_DIR / f"{name}.mp4"
    n_frames = round(dur * fps)
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=None,
                            bitrate=None, macro_block_size=16,
                            ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "17",
                                           "-preset", "medium"]) as wr:
        for k in range(n_frames):
            wr.append_data(render_frame(scene, footer, title, k / fps))
            if k % 120 == 0:
                logger.info("  %s: %d/%d 帧", key, k, n_frames)
    # QC sheet
    times = SHEET_TIMES[key]
    tiles = [render_frame(scene, footer, title, min(tt, dur - 0.05)) for tt in times]
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
    ap.add_argument("--only", choices=list(CLIPS), help="只渲染指定段")
    ap.add_argument("--fps", type=int, default=FPS_DEFAULT)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 屏显常量溯源打印（B5：非手抄证据）
    print("[B5 溯源] TubeConfig 默认值（src/ilqt/tube_types.py）:")
    print(f"  window_half_ms = {TubeConfig.window_half_ms}")
    print(f"  corridor_radius = {TubeConfig.corridor_radius}")
    print(f"  softmin_beta    = {TubeConfig.softmin_beta}")

    keys = [args.only] if args.only else list(CLIPS)
    for k in keys:
        render_clip(k, args.fps)


if __name__ == "__main__":
    main()
