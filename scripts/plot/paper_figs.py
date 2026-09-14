#!/usr/bin/env python3
"""论文图表生成 —— ICRA 2027 冲刺版 v2（2026-09-13 图重制）。

本轮修订要点（对应审稿意见 + 视觉审计结论）：
1. **字号体系统一**：所有图按 figsize 原生尺寸出图（不再用 bbox_inches="tight"），
   插入时 width=\textwidth 即缩放 1.00，标签 9pt / 刻度 7.5pt / 图例 7.5pt 不再漂移；
   （旧版 fig3 因 tight bbox 裁到 2.33in 再放大 3.07 倍，字号达 27pt、占半页）
2. **投稿合规**：pdf.fonttype/ps.fonttype = 42（TrueType 嵌入），消除 Type 3 字体；
3. **统计口径**：所有统计量直接读 paper/planning/06-stats-active.json
   （active-hit 主指标 + 配对 McNemar/bootstrap），保证图与正文数字同源；
4. **术语**：tube-only → corridor-only（标题已去 "Tube MPC"）；
5. fig3 出两个版本：3D 修正版（正文用）+ 2D 双面板备选；
   fig4 改画相对初始位形的关节偏差 + 命中区放大；fig6 色条对称 ±16 并标显著格点；
   fig7 去除三点 log 图改为紧凑数值表；fig8 删除不可读的 3D 排，距离曲线事件对齐。

数据源：
- 统计量:   paper/planning/06-stats-active.json（由 scripts/extract/paired_stats.py 生成）
- 轨迹/诊断: experiment_data/exp18_fig_assets/raw/*.npz + timing.json

用法:
    python scripts/plot/paper_figs.py                      # 出全部
    python scripts/plot/paper_figs.py --fig 3 6 table      # 出指定项
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
# 允许 fig8 在函数内导入 src（与 rm65_mpc_v12.py 的 sys.path 处理一致）
sys.path.insert(0, str(PROJECT))
DATA = PROJECT / "experiment_data"
OUT = PROJECT / "paper" / "figures"
# 统计 JSON 由 resolve_stats_json() 解析：paper 工作区优先，主仓库内副本回退

# ---------------------------------------------------------------------------
# 全局样式（字号在 1.00 缩放下即最终字号）
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,     # TrueType 嵌入：PaperPlaza 不接受 Type 3
    "ps.fonttype": 42,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

C = {
    "full": "#009E73",        # 绿
    "tube_only": "#0072B2",   # 蓝（档位键名保留 tube_only，标签写 corridor-only）
    "softmin_only": "#E69F00",  # 琥珀
    "none": "#D55E00",        # 橙红
    "ball": "#D55E00",
    "racket": "#4D4D4D",      # 球拍中性深灰：不与走廊蓝/softmin 琥珀的机制语义冲突
    "corridor": "#0072B2",    # 走廊固定蓝色，与 tube_only (corridor-only) 一致
}
MODE_LABELS = {"full": "full", "tube_only": "corridor-only",
               "softmin_only": "candidate-set-only", "none": "point-target"}
MODE_ORDER = ["full", "tube_only", "softmin_only", "none"]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def resolve_stats_json(explicit: Path | None = None,
                       project: Path = PROJECT) -> Path:
    """解析统计 JSON：显式路径 > paper 工作区 > 主仓库内副本。"""
    if explicit is not None:
        return explicit
    for cand in (project / "paper" / "planning" / "06-stats-active.json",
                 project / "experiment_data" / "paper_stats_active.json"):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "未找到统计 JSON；请先运行 scripts/extract/paired_stats.py")


def load_stats(explicit: Path | None = None) -> dict:
    """读取配对统计 JSON（active-hit 口径，图与正文同源）。"""
    path = resolve_stats_json(explicit)
    return json.loads(path.read_text(encoding="utf-8"))


def style_ax(ax) -> None:
    """统一坐标轴样式。"""
    ax.grid(True, alpha=0.25, linewidth=0.4)


def ci_half(entry: dict) -> float:
    """从 JSON 的 ci95 取对称半宽（用于 errorbar）。"""
    lo, hi = entry["ci95"]
    return (hi - lo) / 2


def fmt_p(p: float) -> str:
    """p 值格式化（含关系符）：小于 0.001 写 "<0.001"，避免 "0.000" 被误读为精确零。"""
    return "<0.001" if p < 0.001 else f"={p:.3f}"


def fmt_pv(p: float) -> str:
    """p 值（仅数值列，列头已写 "p"）：0.171 / <0.001。"""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def save(fig, name: str) -> None:
    """保存图到输出目录（原生尺寸，无 tight bbox 以保持字号恒定）。"""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300)
    plt.close(fig)
    print(f"已保存 {OUT / name}")


# =============================================================================
# fig3 走廊示意（两个版本）
# =============================================================================

def _corridor_band(ball: np.ndarray, hit: int, r_half: float = 0.12):
    """沿球轨迹在窗口内构造走廊：返回 (方向, 采样点, 垂直基)。

    用于把走廊画成连续带状（而非几个孤立圆环）。
    """
    k0, k1 = max(0, hit - 25), min(len(ball) - 1, hit + 5)
    seg = ball[k0:k1 + 1]
    u = seg[-1] - seg[0]
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(u, [0, 0, 1.0])
    v = v / (np.linalg.norm(v) + 1e-12)
    w = np.cross(u, v)
    return u, seg, v, w


def fig1_hero(npz_path: Path = DATA / "exp18_fig_assets/raw/a_hit_clean.npz") -> None:
    """Fig.1（hero 概念示意图，顶视）: (a) 单点目标 vs (b) 候选窗口终端 + 走廊。

    chat10 审稿意见：坐标框边线像调试图、灰→橙轨迹有断口、空心圆不像球拍、
    黑色候选点没有时间层次、图内文字压线、没有机器人线索。现按「开放画布 +
    球拍轮廓 + 连续轨迹 + 橙/蓝配色」重做：
    - 两面板共享同一条连续球轨迹（窗口外浅灰、窗口内橙，重叠一步保证无断口）；
    - 球拍画成椭圆拍面 + 短柄，不用空心圆；横向半轴 ≈ 有效碰撞半径 0.12 m，
      具体数值只写在图注里；
    - 走廊 = 沿轨迹逐点 ±0.12 m 的浅蓝半透明带 + 蓝色虚线边界（与结果图同色）；
    - 图内只留三个标签（single target / candidate ball states / spatial corridor），
      面板编号 (a)/(b) 置于左上角；不显示刻度与坐标框。

    坐标约定：npz 的 ball_pos 列为 (X, Y, Z)，本图顶视绘制 (x=Y, y=X)；
    所有几何（切向、法向、拍面角度）都在绘图坐标下计算。画布高度按内容
    长宽比反推，保证内容填满面板、两面板完全一致。
    """
    from matplotlib.patches import Ellipse

    d = np.load(npz_path)
    ball = d["ball_pos"]
    tcp = d["tcp_pos"]
    hit = int(d["hit_step"])
    k0 = max(0, hit - 23)
    b = ball[k0:hit + 1]
    p_face = np.array([tcp[hit, 1], tcp[hit, 0]])       # 绘图坐标（x=Y, y=X）
    r_half = 0.12
    i_w0 = max(0, len(b) - 11)                          # 窗口起点（±50 ms ≈ 10 步）
    cw = np.stack([b[i_w0:, 1], b[i_w0:, 0]], axis=1)   # 窗口段（绘图坐标）
    tang = np.gradient(cw, axis=0)
    tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)   # 左法向（逐点）
    u = tang[-1]                                        # 末端切向（拍面朝向）

    # ---- 内容包围盒 → 画布几何（内容填满面板，两面板一致）----
    half_x = 0.17                                       # 拍面沿轨迹半长
    half_y = 0.15                                       # 拍面横向半轴 + 余量
    x_lo = float(b[:, 1].min()) - 0.03
    x_hi = float(p_face[0]) + max(0.36, half_x + 0.14)  # 含拍柄
    y_from_path = [float(b[:, 0].min()) - 0.14,         # 走廊下边界 + 标签
                   float(b[:, 0].max()) + 0.14]         # 走廊上边界
    y_lo = min(y_from_path[0], float(p_face[1]) - half_y - 0.03)
    y_hi = max(y_from_path[1], float(p_face[1]) + half_y + 0.03)
    xlim, ylim = (x_lo, x_hi), (y_lo, y_hi)

    fig_w = 3.40
    panel_w = fig_w * 0.976                             # subplots_adjust 后的面板宽
    aspect = (y_hi - y_lo) / (x_hi - x_lo)
    panel_h = aspect * panel_w
    fig_h = 2 * panel_h + 0.30                          # 两面板 + 行距/边距
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.995, bottom=0.005,
                        hspace=0.10)
    print(f"  [fig1] xspan={x_hi - x_lo:.3f} m, yspan={y_hi - y_lo:.3f} m, "
          f"figsize=({fig_w:.2f},{fig_h:.2f})")

    def canvas(ax) -> None:
        """开放画布：无坐标框、无刻度、等比例、两面板同视野。"""
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    def ball_path(ax) -> None:
        """整条球轨迹：窗口外浅灰、窗口内橙（窗口段向前多含一步，无断口）。"""
        ax.plot(b[:, 1], b[:, 0], color="#B8B8B8", lw=1.4, zorder=2,
                solid_capstyle="round")
        j0 = i_w0 - 1 if i_w0 > 0 else 0
        ax.plot(b[j0:, 1], b[j0:, 0], color=C["ball"], lw=1.8, zorder=3,
                solid_capstyle="round")

    def racket(ax) -> None:
        """球拍：椭圆拍面（横向半轴 ≈ r_half）+ 短柄（沿出球方向）。"""
        ang = float(np.degrees(np.arctan2(u[1], u[0])))
        ax.add_patch(Ellipse(p_face, width=2 * 0.14, height=2 * r_half,
                             angle=ang, facecolor="0.95",
                             edgecolor=C["racket"], lw=1.0, zorder=4))
        h0, h1 = p_face + 0.15 * u, p_face + 0.36 * u
        ax.plot([h0[0], h1[0]], [h0[1], h1[1]], color=C["racket"], lw=2.6,
                solid_capstyle="round", zorder=4)

    def corridor(ax) -> None:
        """走廊带：窗口段逐点 ±r_half 的浅蓝填充 + 蓝色虚线边界。"""
        up, dn = cw + r_half * nrm, cw - r_half * nrm
        ax.fill(np.concatenate([up[:, 0], dn[::-1, 0]]),
                np.concatenate([up[:, 1], dn[::-1, 1]]),
                color="#56B4E9", alpha=0.18, lw=0, zorder=1)
        for edge in (up, dn):
            ax.plot(edge[:, 0], edge[:, 1], color=C["corridor"], lw=0.9,
                    ls=(0, (4, 2)), zorder=2)

    # ---------------- (a) point-target：单一目标 ----------------
    ax = axes[0]
    canvas(ax)
    ball_path(ax)
    racket(ax)
    ax.scatter([b[-1, 1]], [b[-1, 0]], marker="*", s=75, color=C["ball"],
               edgecolor="white", lw=0.5, zorder=6)
    ax.annotate("single target", (b[-1, 1], b[-1, 0]), textcoords="offset points",
                xytext=(-13, 11), fontsize=8, ha="right", va="bottom",
                color=C["ball"], zorder=7,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C["ball"],
                                shrinkA=0, shrinkB=4))
    ax.text(0.005, 0.995, "(a) Point target", transform=ax.transAxes,
            fontsize=8.5, ha="left", va="top")

    # ---------------- (b) 候选窗口终端 + 走廊 ----------------
    ax = axes[1]
    canvas(ax)
    corridor(ax)
    ball_path(ax)
    racket(ax)
    # 候选球态（窗口内 5 个时刻，中心最深、两侧渐浅）
    ks = np.linspace(i_w0, i_w0 + 0.88 * (len(b) - 1 - i_w0), 5).round().astype(int)
    for k, al in zip(ks, (0.30, 0.55, 1.0, 0.55, 0.30)):
        ax.scatter([b[k, 1]], [b[k, 0]], s=30, color=C["ball"], alpha=al,
                   edgecolor="white", lw=0.4, zorder=6)
    # 时间顺序箭头（沿轨迹，贴走廊带上缘，轻量）
    j_ar = int(0.88 * (len(cw) - 1))
    m0 = cw[0] + 1.10 * r_half * nrm[0]
    m1 = cw[j_ar] + 1.10 * r_half * nrm[j_ar]
    ax.annotate("", xy=(m1[0], m1[1]), xytext=(m0[0], m0[1]),
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.45",
                                shrinkA=1, shrinkB=1), zorder=5)
    k_lab = int(ks[1])
    ax.annotate("candidate ball states", (b[k_lab, 1], b[k_lab, 0]),
                textcoords="offset points", xytext=(-5, 11), fontsize=8,
                ha="right", va="bottom", color=C["ball"], zorder=7,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C["ball"],
                                shrinkA=0, shrinkB=3))
    mid = cw[len(cw) // 2] - 1.35 * r_half * nrm[len(cw) // 2]
    ax.annotate("spatial corridor", (mid[0], mid[1]),
                textcoords="offset points", xytext=(0, -2), fontsize=8,
                ha="center", va="top", color=C["corridor"], zorder=7)
    ax.text(0.005, 0.995, "(b) Candidate-window terminal + corridor",
            transform=ax.transAxes, fontsize=8.5, ha="left", va="top")

    save(fig, "fig1_hero.pdf")


def fig3_3d(npz_path: Path = DATA / "exp18_fig_assets/raw/a_hit_clean.npz") -> None:
    """Fig.3（正文版）: 3D 球轨迹 + 球拍轨迹 + 连续走廊带 + 击球点。

    画布按**单栏**尺寸出图（3.40 in 宽，对应 \\columnwidth 3.5 in）：
    Axes3D 的投影盒受图高限制，若按 7.16 in 通栏铺开，墨迹只占画布 38%，
    两侧留出约 4.4 in 空白 —— 这正是旧版「页面正中一小块、大量浪费」的根因。
    """
    d = np.load(npz_path)
    ball, tcp = d["ball_pos"], d["tcp_pos"]
    hit = int(d["hit_step"])
    dt = float(d["dt"])

    # 画布 3.40x3.15 in：上方 2.66 in 给 3D 投影盒（盒宽高比 4:3，铺满画布宽度），
    # 下方 0.49 in 留给两栏图例，避免图例压住轨迹
    fig = plt.figure(figsize=(3.40, 3.15))
    # 投影盒留出四周余量：Axes3D 的 z 轴标签会伸出盒外，若铺满画布会被切到
    ax = fig.add_axes([0.055, 0.18, 0.87, 0.795], projection="3d")
    k0 = max(0, hit - 60)
    ax.plot(ball[k0:hit + 3, 0], ball[k0:hit + 3, 1], ball[k0:hit + 3, 2],
            color=C["ball"], lw=1.6, label="Ball trajectory")
    ax.plot(tcp[k0:hit + 8, 0], tcp[k0:hit + 8, 1], tcp[k0:hit + 8, 2],
            color=C["racket"], lw=1.6, label="Racket center trajectory")

    # 连续走廊管：沿球轨迹扫掠一个半宽 r_half 的管面（半透明）
    r_half = 0.12
    u, seg, v, w = _corridor_band(ball, hit, r_half)
    th = np.linspace(0, 2 * np.pi, 28)
    Xs = np.array([p[0] + r_half * (np.cos(a) * v[0] + np.sin(a) * w[0])
                   for p in seg for a in th]).reshape(len(seg), len(th))
    Ys = np.array([p[1] + r_half * (np.cos(a) * v[1] + np.sin(a) * w[1])
                   for p in seg for a in th]).reshape(len(seg), len(th))
    Zs = np.array([p[2] + r_half * (np.cos(a) * v[2] + np.sin(a) * w[2])
                   for p in seg for a in th]).reshape(len(seg), len(th))
    ax.plot_surface(Xs, Ys, Zs, color=C["corridor"], alpha=0.16, linewidth=0,
                    shade=False, label="Corridor (half-width 0.12 m)")
    # 击球点横截面圆环：标出走廊半径，避免整条管道难以判读尺度
    p_hit = ball[hit]
    ring = p_hit + r_half * (np.outer(np.cos(th), v) + np.outer(np.sin(th), w))
    ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=C["corridor"], lw=1.1,
            alpha=0.95, label="Cross-section at hit")
    ax.scatter(*ball[hit], color="k", s=32, marker="*", zorder=5,
               label=f"Hit ($t={hit * dt:.1f}$ s)")

    ax.set_xlabel("X (m)", labelpad=-1)
    ax.set_ylabel("Y (m)", labelpad=-1)
    ax.set_zlabel("Z (m)", labelpad=-1)
    ax.view_init(elev=22, azim=-58)
    ax.tick_params(labelsize=6.5, pad=0.3)
    # 图例移到画布底部两栏：单栏宽度下若放在盒内会压住球轨迹
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, fontsize=6.2, frameon=False, columnspacing=0.9,
               handlelength=1.5, handletextpad=0.4, labelspacing=0.25)
    save(fig, "fig3_tube_corridor.pdf")
    plt.close(fig)
    print("已保存 paper/figures/fig3_tube_corridor.pdf （3D 正文版）")


def fig3_alt_2d(npz_path: Path = DATA / "exp18_fig_assets/raw/d_hit_noise_kf.npz") -> None:
    """Fig.3: 走廊几何（放大到候选窗口）——(a) 侧视 Y–Z (b) 顶视 Y–X。

    chat9 审稿意见：旧版坐标仍覆盖整段飞行、走廊缩在右端；"±50 ms window"
    长引线横跨半张图；图内写 "hit" 而图例叫 "Reference point"（语义冲突）；
    图例约 6.6 pt。现改为：
    - 两面板放大到候选窗口邻域、共享 Y 轴；左上角保留小型全轨迹定位图；
    - 参考点（小圆点 = 预测最优候选步的球位置）与真实接触（星号）分开标注；
    - 窗口范围用两端短竖线 + 窗口内双向箭头表示，取消长引线；
    - 图例 8 pt，条目标注统一为 "Reference point (best candidate)" / "Contact"。
    """
    d = np.load(npz_path)
    ball, tcp = d["ball_pos"], d["tcp_pos"]
    k_contact = int(d["hit_step"])
    # 参考步 = 起始时刻预测的最优候选步（与 replan_core 同源函数）
    from src.robot.constants import SHOULDER_POS, WORKSPACE_RADIUS
    from src.sim.rm65_env import RM65Env
    from src.tennis.hitting import find_hitting_point_physics
    md = json.loads(d["metadata"].item())
    _env = RM65Env(PROJECT / "src" / "robot" / "rm65_model.xml")
    hi = find_hitting_point_physics(
        _env, np.array(md["p0"]), np.array(md["v0"]),
        SHOULDER_POS, WORKSPACE_RADIUS, len(d["timestamps"]),
    )
    k_ref = hi["k_hit"] if hi is not None else k_contact
    win = 10                                    # ±50 ms = ±10 步
    i_w0, i_w1 = k_ref - win, k_ref
    seg = ball[i_w0:i_w1 + 1]
    GRAY = "0.65"
    r_half = 0.12

    fig, axes = plt.subplots(2, 1, figsize=(3.40, 2.72), sharex=True)

    # 放大区间（两面板共享 Y 轴范围）
    y_lo = float(seg[:, 1].min()) - 0.16
    y_hi = float(seg[:, 1].max()) + 0.06

    def draw_window(ax, n_axis: np.ndarray) -> None:
        """两面板共用的窗口段绘制：走廊带 + 轴线 + 方向箭头 + 窗口端点竖线。"""
        p0, p1 = seg[0], seg[-1]
        u3 = p1 - p0
        u3 = u3 / (np.linalg.norm(u3) + 1e-12)

        def proj(v):                            # 投到本面板两维
            return np.array([v[1], v[2]]) if n_axis[0] == 2 else np.array([v[1], v[0]])

        a0, a1 = proj(p0), proj(p1)
        u2 = a1 - a0
        u2 = u2 / (np.linalg.norm(u2) + 1e-12)
        n2 = np.array([-u2[1], u2[0]])
        ax.fill([a0[0] + r_half * n2[0], a1[0] + r_half * n2[0],
                 a1[0] - r_half * n2[0], a0[0] - r_half * n2[0]],
                [a0[1] + r_half * n2[1], a1[1] + r_half * n2[1],
                 a1[1] - r_half * n2[1], a0[1] - r_half * n2[1]],
                color=C["corridor"], alpha=0.18, lw=0, zorder=1)
        for sgn in (+1, -1):
            ax.plot([a0[0] + sgn * r_half * n2[0], a1[0] + sgn * r_half * n2[0]],
                    [a0[1] + sgn * r_half * n2[1], a1[1] + sgn * r_half * n2[1]],
                    color=C["corridor"], lw=0.9, ls="--", zorder=2)
        # 轴线画细、压在球轨迹下方：窗口内球轨迹与轴线几乎重合，轴只作几何参照
        ax.plot([a0[0], a1[0]], [a0[1], a1[1]], color=C["corridor"], lw=0.9,
                alpha=0.65, zorder=2.5)
        # 轴线方向箭头（沿球前进方向）
        ax.annotate("", xy=(a1[0], a1[1]),
                    xytext=(a1[0] - 0.45 * np.linalg.norm(a1 - a0) * u2[0],
                            a1[1] - 0.45 * np.linalg.norm(a1 - a0) * u2[1]),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color=C["corridor"],
                                    shrinkA=0, shrinkB=0), zorder=4)
        # 窗口两端短竖线（垂直于轴线）——代替旧版长引线
        for pz in (p0, p1):
            c2 = proj(pz)
            ax.plot([c2[0] - 0.07 * n2[0], c2[0] + 0.07 * n2[0]],
                    [c2[1] - 0.07 * n2[1], c2[1] + 0.07 * n2[1]],
                    color=GRAY, lw=1.2, zorder=5)
        # 窗口内的双向箭头（表示 ±50 ms 候选窗口宽度）
        bp = [a0 + 0.35 * r_half * n2, a1 + 0.35 * r_half * n2]
        ax.annotate("", xy=(bp[1][0], bp[1][1]), xytext=(bp[0][0], bp[0][1]),
                    arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.35",
                                    shrinkA=0, shrinkB=0), zorder=5)

    # (a) 侧视：Y–Z（球前进方向 vs 高度）
    ax = axes[0]
    ax.plot(ball[:i_w0, 1], ball[:i_w0, 2], color=GRAY, lw=1.1, zorder=2)
    ax.plot(ball[i_w1:, 1], ball[i_w1:, 2], color=GRAY, lw=1.1, zorder=2)
    ax.plot(seg[:, 1], seg[:, 2], color=C["ball"], lw=1.6, zorder=3,
            label="Ball trajectory")
    draw_window(ax, np.array([2, 1]))
    ax.plot(tcp[:, 1], tcp[:, 2], color=C["racket"], lw=1.3, zorder=3,
            label="Racket center")
    ax.scatter([ball[k_ref, 1]], [ball[k_ref, 2]], color="w", edgecolor="k",
               marker="o", s=22, lw=0.9, zorder=7,
               label="Reference point (best candidate)")
    ax.scatter([ball[k_contact, 1]], [ball[k_contact, 2]], color="k",
               marker="*", s=45, zorder=8, label="Contact")
    ax.annotate("contact", (ball[k_contact, 1], ball[k_contact, 2]),
                textcoords="offset points", xytext=(-9, -15), fontsize=7,
                ha="right", va="top", color="k", zorder=9,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="k",
                                shrinkA=0, shrinkB=4))
    ax.annotate("corridor axis", (seg[len(seg) // 2, 1], seg[len(seg) // 2, 2]),
                textcoords="offset points", xytext=(-2, -20), fontsize=7,
                ha="right", va="top", color=C["corridor"])
    ax.set_ylabel("Z (m)")
    ax.set_ylim(0.68, 1.08)
    ax.set_title("(a) Side view", fontsize=8.5)
    style_ax(ax)

    # (b) 顶视：Y–X（球前进方向 vs 横向）
    ax = axes[1]
    ax.plot(ball[:i_w0, 1], ball[:i_w0, 0], color=GRAY, lw=1.1, zorder=2)
    ax.plot(ball[i_w1:, 1], ball[i_w1:, 0], color=GRAY, lw=1.1, zorder=2)
    ax.plot(seg[:, 1], seg[:, 0], color=C["ball"], lw=1.6, zorder=3)
    draw_window(ax, np.array([1, 0]))
    ax.plot(tcp[:, 1], tcp[:, 0], color=C["racket"], lw=1.3, zorder=3)
    ax.scatter([ball[k_ref, 1]], [ball[k_ref, 0]], color="w", edgecolor="k",
               marker="o", s=22, lw=0.9, zorder=7)
    ax.scatter([ball[k_contact, 1]], [ball[k_contact, 0]], color="k",
               marker="*", s=40, zorder=8)
    # chat10：窗口范围（±50 ms）改由图注说明，图内不再标注以去拥挤
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("X (m)")
    ax.set_ylim(-0.98, -0.60)
    ax.set_title("(b) Top view", fontsize=8.5)
    style_ax(ax)
    # 两面板共享放大的 Y 轴范围（sharex=True，设一次即可）
    axes[1].set_xlim(y_lo, y_hi)

    fig.tight_layout(pad=0.25, rect=(0, 0.15, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8.0,
               frameon=False, handlelength=1.6, columnspacing=1.4,
               handletextpad=0.5, labelspacing=0.3)
    save(fig, "fig3_alt_2d.pdf")
    plt.close(fig)
    print(f"已保存 {OUT / 'fig3_alt_2d.pdf'} （走廊几何放大版）")


def _fig3_alt_2d_old(npz_path: Path = DATA / "exp18_fig_assets/raw/a_hit_clean.npz") -> None:
    """Fig.3（备选版）: 2D 双面板 —— (a) 侧视 (b) 俯视，走廊画成半透明带。

    与技术选型无关，只与版式有关：画布宽度与 3D 正文版一致（3.40 in = 单栏），
    面板改为上下堆叠，这样两版可直接互换、对比时只差「呈现方式」这一个变量。

    chat8 审稿意见：running cost 只在候选窗口（±50 ms）内启用，旧图把走廊带
    画满 hit−60..hit+3 整段轨迹、且无参考点/方向标记，读者无法对照式(2)(3)。
    现改为：窗口外轨迹浅灰、窗口内橙色；直线带只覆盖窗口段（首尾连线 ± r_half）；
    标出窗口起止、走廊参考点（best 步球位置）与轴线方向箭头。
    """
    d = np.load(npz_path)
    ball, tcp = d["ball_pos"], d["tcp_pos"]
    hit = int(d["hit_step"])
    win = 10                                   # ±50 ms = ±10 步候选窗口半宽
    k0 = max(0, hit - 60)
    b, r = ball[k0:hit + 3], tcp[k0:hit + 8]
    i_hit = hit - k0                        # 命中步在 b 中的下标
    i_w0 = i_hit - win                      # 窗口首步在 b 中的下标（hit−10）
    i_w1 = i_hit                            # 窗口末步下标（命中步）
    GRAY = "0.65"

    fig, axes = plt.subplots(2, 1, figsize=(3.40, 2.85))
    r_half = 0.12

    # (a) 侧视：Y–Z 平面（球的前进方向 vs 高度）
    # chat7 审稿意见：式(2)(3)/代码的走廊轴是「窗口平均方向 + 固定投影矩阵」的
    # 直线轴（零代价集合为绕直线的圆柱），旧版沿抛物线画弯曲带、与公式不一致。
    # 现按公式画为：窗口首尾连线（窗口平均方向的平面投影）± r_half 的直线带；
    # 橙色曲线保留真实球轨迹，二者在窗口中段几乎相切、如实表达「局部直线近似」。
    ax = axes[0]
    # 窗口外轨迹浅灰、窗口内橙色（区分「空间轴延伸」与「代价启用时间范围」）
    ax.plot(b[:i_w0, 1], b[:i_w0, 2], color=GRAY, lw=1.4, zorder=2)
    ax.plot(b[i_w1:, 1], b[i_w1:, 2], color=GRAY, lw=1.4, zorder=2)
    ax.plot(b[i_w0:i_w1 + 1, 1], b[i_w0:i_w1 + 1, 2], color=C["ball"], lw=1.6,
            zorder=3, label="Ball trajectory")
    p0_2, p1_2 = b[i_w0, 1:].astype(float), b[i_w1, 1:].astype(float)
    u2 = p1_2 - p0_2
    L2 = float(np.linalg.norm(u2))
    u2 = u2 / (L2 + 1e-12)
    n2 = np.array([-u2[1], u2[0]])          # 带内法线（Y–Z 平面内垂直于轴）
    tau = np.array([0.0, L2])
    axis = p0_2[None, :] + tau[:, None] * u2[None, :]
    band_up = axis + r_half * n2
    band_dn = axis - r_half * n2
    ax.plot(axis[:, 0], axis[:, 1], color=C["corridor"], lw=1.0, alpha=0.9)
    for band in (band_up, band_dn):
        ax.plot(band[:, 0], band[:, 1], color=C["corridor"], lw=0.9, ls="--")
    poly = np.vstack([band_up, band_dn[::-1]])
    ax.fill(poly[:, 0], poly[:, 1], color=C["corridor"], alpha=0.18, lw=0,
            label="Corridor ($\\pm0.12$ m)")
    # 方向箭头：轴线末端（窗口末）画箭头，表示 d_ball 方向
    ax.annotate("", xy=axis[-1], xytext=axis[-1] - 0.55 * (L2 + 1e-12) * u2,
                arrowprops=dict(arrowstyle="->", lw=1.0, color=C["corridor"],
                                shrinkA=0, shrinkB=0))
    # 窗口起止标记：窗口两端画短竖线（灰色），标注 "±50 ms window"
    for ix in (i_w0, i_w1):
        pz = b[ix]
        ax.plot([pz[1], pz[1]], [pz[2] - 0.10, pz[2] + 0.10],
                color=GRAY, lw=1.0, zorder=4)
    ax.annotate("$\\pm 50$ ms window", (0.5 * (b[i_w0, 1] + b[i_w1, 1]),
                                        b[i_w0, 2] + r_half + 0.06),
                ha="center", va="bottom", fontsize=7, color="0.35")
    ax.plot(r[:, 1], r[:, 2], color=C["racket"], lw=1.4, label="Racket center")
    ax.scatter(b[i_hit, 1], b[i_hit, 2], color="k", marker="*", s=40, zorder=5,
               label="Reference point")
    # hit 标注置于命中点左下方（走廊带内空白区）：左上偏移会被球轨迹线穿过词面
    ax.annotate("hit", (b[i_hit, 1], b[i_hit, 2]), textcoords="offset points",
                xytext=(-8, -12), fontsize=8, ha="right", va="top")
    # 窗口段标注：锚定窗口起点，文字放左下空白区（贴带上缘会被边框截断）
    ax.annotate("$\\pm 50$ ms window", xy=(b[i_w0, 1], b[i_w0, 2]),
                xycoords="data", xytext=(0.04, 0.18),
                textcoords="axes fraction", fontsize=6.5, color="0.35",
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.35",
                                shrinkA=2, shrinkB=2))
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("(a) Side view: corridor along the ball line", fontsize=8)
    style_ax(ax)

    # (b) 俯视：X–Y 平面（横向走廊宽度）
    ax = axes[1]
    # 横向走廊：以窗口首尾连线为轴，两侧各 r_half 的平行带（只覆盖窗口段）
    u = b[i_w1, :2] - b[i_w0, :2]
    u = u / (np.linalg.norm(u) + 1e-12)
    nrm = np.array([-u[1], u[0]])
    off = r_half * nrm
    seg0, seg1 = b[i_w0, :2], b[i_w1, :2]
    poly_x = [seg0[0] + off[0], seg1[0] + off[0],
              seg1[0] - off[0], seg0[0] - off[0]]
    poly_y = [seg0[1] + off[1], seg1[1] + off[1],
              seg1[1] - off[1], seg0[1] - off[1]]
    # 绘制顺序 Ball→Corridor→Racket 与 (a) 一致；走廊填充压底不遮球线
    ax.plot(b[:i_w0, 0], b[:i_w0, 1], color=GRAY, lw=1.4, zorder=2)
    ax.plot(b[i_w1:, 0], b[i_w1:, 1], color=GRAY, lw=1.4, zorder=2)
    ax.plot(b[i_w0:i_w1 + 1, 0], b[i_w0:i_w1 + 1, 1], color=C["ball"], lw=1.6,
            zorder=3, label="Ball trajectory")
    ax.fill(poly_x, poly_y, color=C["corridor"], alpha=0.2, lw=0, zorder=1,
            label="Corridor ($\\pm0.12$ m)")
    ax.plot([seg0[0], seg1[0]], [seg0[1], seg1[1]], color=C["corridor"],
            lw=1.0, alpha=0.9, zorder=2)
    for sgn in (+1, -1):
        ax.plot([seg0[0] + sgn * off[0], seg1[0] + sgn * off[0]],
                [seg0[1] + sgn * off[1], seg1[1] + sgn * off[1]],
                color=C["corridor"], lw=0.9, ls="--", zorder=2)
    # 方向箭头（轴线末端）
    ax.annotate("", xy=seg1, xytext=seg1 - 0.55 * np.linalg.norm(seg1 - seg0) * u,
                arrowprops=dict(arrowstyle="->", lw=1.0, color=C["corridor"],
                                shrinkA=0, shrinkB=0))
    for ix in (i_w0, i_w1):
        px = b[ix]
        n_p = np.array([-u[1], u[0]])
        ax.plot([px[0] - 0.10 * n_p[0], px[0] + 0.10 * n_p[0]],
                [px[1] - 0.10 * n_p[1], px[1] + 0.10 * n_p[1]],
                color=GRAY, lw=1.0, zorder=4)
    ax.plot(r[:, 0], r[:, 1], color=C["racket"], lw=1.4, zorder=2,
            label="Racket center")
    ax.scatter(b[i_hit, 0], b[i_hit, 1], color="k", marker="*", s=40, zorder=5)
    # hit 标注移到锚点下方：俯视图中命中点贴近上边框（judge 意见）
    ax.annotate("hit", (b[i_hit, 0], b[i_hit, 1]), textcoords="offset points",
                xytext=(6, -7), fontsize=8, va="top")
    ax.annotate("$\\pm 50$ ms window", xy=(b[i_w0, 0], b[i_w0, 1]),
                xycoords="data", xytext=(0.55, 0.10),
                textcoords="axes fraction", fontsize=6.5, color="0.35",
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.35",
                                shrinkA=2, shrinkB=2))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("(b) Top view: lateral corridor width", fontsize=8)
    style_ax(ax)

    # 图例统一移到画布底部：面板内任意角落都会被走廊带/球轨迹遮挡（judge 意见）
    fig.tight_layout(pad=0.3, rect=(0, 0.13, 1, 1))
    handles, labels = axes[0].get_legend_handles_labels()
    # 4 条图例：2 列 2 行（3 列会把第 4 条 "Reference point" 裁出画布右缘）
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6.4,
               frameon=False, handlelength=1.5, columnspacing=1.6,
               bbox_to_anchor=(0.5, 0.004))
    save(fig, "fig3_alt_2d.pdf")
    plt.close(fig)
    print(f"已保存 {OUT / 'fig3_alt_2d.pdf'} （2D 备选版）")


# =============================================================================
# fig4 关节偏差 + TCP 速度（含命中区放大）
# =============================================================================

def fig4(npz_a: Path = DATA / "exp18_fig_assets/raw/a_hit_clean.npz",
         npz_b: Path = DATA / "exp18_fig_assets/raw/b_hit_space_perturb.npz") -> None:
    """Fig.4: (a) 关节角相对初始位形的偏差；(b) TCP 速度 + 限速带 + 命中区放大 inset。"""
    da, db = np.load(npz_a), np.load(npz_b)
    t_a = da["timestamps"] * 1000
    t_b = db["timestamps"] * 1000
    hit_a = int(da["hit_step"]) * float(da["dt"]) * 1000
    hit_b = int(db["hit_step"]) * float(db["dt"]) * 1000
    names = ["J0", "J1", "J2", "J3", "J4", "J5"]
    colors = ["#0072B2", "#D55E00", "#009E73", "#56B4E9", "#E69F00", "#CC79A7"]
    # 灰度打印下单靠颜色无法区分 6 条曲线（审稿人意见）：每条曲线给唯一线型，
    # J4/J5 再叠稀疏空心 marker，保证黑白下仍可逐条追踪
    styles = ["-", "--", "-.", ":", "-", "--"]
    markers = [None, None, None, None, ("o", 60), ("s", 60)]

    # 版式：(a)(b) 上下同宽共享时间轴（同列，竖直对照成立），(c) 占右列两行。
    # 旧版 (a) 通栏而 (b) 只有 55% 宽：上下同宽的时间轴被拉成不同比例，
    # 竖直对照失效；放大图早期是 (b) 的内嵌 inset，但 (b) 里没有足够大的
    # 空白矩形（任意位置都会压住 1.8 m/s 限速线），故升为右列独立面板。
    fig = plt.figure(figsize=(7.16, 1.88))
    # 显式给四周留白（不用 tight_layout）：默认 subplot 参数会让坐标轴只占 125%-90% 宽
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.55, 1],
                          left=0.055, right=0.995, top=0.87, bottom=0.185,
                          hspace=0.62, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    # (a) 关节利用率：|Δq| 占初始位形到最近限位余量的百分比（100% 线 = 关节限位）。
    # 绝对偏差（deg）各关节量级不同，同一尺度下读不出安全余量（审稿人意见）。
    import yaml
    q0 = da["q_actual"][0]
    lim = yaml.safe_load(
        (PROJECT / "configs" / "default.yaml").read_text(encoding="utf-8")
    )["robot_limits"]
    head = np.minimum(np.abs(np.radians(lim["q_min_deg"][:6]) - q0),
                      np.abs(np.radians(lim["q_max_deg"][:6]) - q0))
    util = 100.0 * np.abs(da["q_actual"] - q0) / head
    for j in range(6):
        mk = markers[j]
        kw = {}
        if mk is not None:
            kw = dict(marker=mk[0], markevery=mk[1], markersize=2.6,
                      markerfacecolor="none", markeredgewidth=0.7)
        ax.plot(t_a, util[:, j], color=colors[j], lw=1.0, ls=styles[j],
                label=names[j], **kw)
    ax.axhline(100, color="gray", ls="--", lw=0.8)
    ax.annotate("joint limit", xy=(0.985, 100), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=7, color="dimgray")
    ax.axvline(hit_a, color="k", ls=":", lw=0.8)
    ax.set_ylabel("Utilization (%)")
    ax.set_ylim(0, 130)
    # 图例放在曲线峰值（≈45%）与 100% 限位线之间的空白带内（图高压缩后贴顶会压标题）；
    # handlelength 加长以便虚/点线型在图例中可辨认（灰度可读性，审稿人意见）
    ax.legend(ncol=6, loc="lower left", bbox_to_anchor=(0.0, 0.55),
              fontsize=7, columnspacing=0.7, handlelength=1.5, labelspacing=0.2,
              borderpad=0.25, framealpha=0.9)
    ax.set_title("(a) Joint excursion vs. available limit headroom", fontsize=9)
    ax.tick_params(labelbottom=False)
    style_ax(ax)

    # (b) TCP 速度：两条曲线 + 限速带 + 命中时刻竖线
    ax = fig.add_subplot(gs[1, 0], sharex=ax)
    v_a = np.linalg.norm(np.gradient(da["tcp_pos"], axis=0), axis=1) / float(da["dt"])
    v_b = np.linalg.norm(np.gradient(db["tcp_pos"], axis=0), axis=1) / float(db["dt"])
    # 灰阶 = 实验条件（标称/扰动 run），机制配色只留给 cost 变体（figs 6/7）
    ax.plot(t_a, v_a, color="#4D4D4D", lw=1.1, label="nominal run")
    ax.plot(t_b, v_b, color="#999999", lw=1.1, ls="--",
            label="space-perturbed run ($s=0.1$ m)")
    ax.axhspan(1.8, 1.98, color="gray", alpha=0.25, lw=0)
    ax.axhline(1.8, color="gray", ls="--", lw=0.8)
    # 标签放在限速线下方（4% 处起步，避免贴左轴；无白底，不遮虚线）
    ax.annotate("TCP limit 1.8 m/s", xy=(0.04, 1.72),
                xycoords=("axes fraction", "data"), ha="left", va="top",
                fontsize=7, color="dimgray")
    # 命中时刻统一为黑色点线（与 (a) 一致；两条曲线的身份已由颜色区分）
    ax.axvline(hit_a, color="k", ls=":", lw=0.8)
    ax.axvline(hit_b, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("TCP speed (m/s)")
    ax.set_title("(b) TCP speed profiles", fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
    style_ax(ax)

    # (c) 命中邻域放大：两条曲线在此分离（整段画在一起时完全重合、读不出差异）
    # chat8 审稿意见：caption 给出 −5 ms 接触时间差，但旧版两曲线各自按自己的
    # 接触时刻对齐（差值被减掉），图中看不到两次接触事件。改为统一参考
    # t − hit_a，两次接触分别用与曲线同线型的竖线标出。
    axi = fig.add_subplot(gs[0:2, 1])
    m_a = (t_a > hit_a - 120) & (t_a < hit_a + 120)
    m_b = (t_b > hit_b - 120) & (t_b < hit_b + 120)
    shift_b = hit_b - hit_a
    axi.plot(t_a[m_a] - hit_a, v_a[m_a], color="#4D4D4D", lw=1.1,
             label="nominal")
    axi.plot(t_b[m_b] - hit_a, v_b[m_b], color="#999999", lw=1.1, ls="--",
             label="space-perturbed")
    axi.axhline(1.8, color="gray", ls="--", lw=0.8)
    # 两次接触时刻：线型与各自曲线一致（统一参考 t − hit_a）
    axi.axvline(0, color="#4D4D4D", ls=":", lw=0.9)
    axi.axvline(shift_b, color="#999999", ls=":", lw=0.9)
    axi.set_xlabel("$t-t_{hit}^{nom}$ (ms)")
    axi.set_ylabel("TCP speed (m/s)")
    axi.set_ylim(0, max(v_a[m_a].max(), v_b[m_b].max()) * 1.15)
    # chat9 审稿意见：−5 ms 接触偏移由 caption 承载，图内只留两条与曲线同线型的
    # 接触竖线（不再画多行长引线，也不再加图内文字）
    # 两条曲线的身份用小图例在左下角标出（审稿人意见：原内嵌标签与 Δv 文字
    # 在曲线峰值附近互相压字；左下 [−120,−40] ms 区全空，figure 内其余位置
    # 右下被回收段占据）。Δv_max / hit shift 数值移入 caption，此处 print 供核对。
    # 统一参考 t − hit_a（与绘图一致）
    xa, xb = t_a[m_a] - hit_a, t_b[m_b] - hit_a
    common = np.union1d(xa, xb)
    dv = np.abs(np.interp(common, xa, v_a[m_a])
                - np.interp(common, xb, v_b[m_b])).max()
    print(f"fig4(c): dv_max={dv:.2f} m/s, hit shift={hit_b - hit_a:+.0f} ms")
    axi.legend(loc="lower left", ncol=2, fontsize=6.5, framealpha=0.9,
               handlelength=1.3, columnspacing=0.9, borderpad=0.3,
               labelspacing=0.2)
    axi.set_title("(c) Zoom: hit neighbourhood", fontsize=9)
    style_ax(axi)

    # 不用 tight_layout：它会把上面 add_gridspec 设的 hspace/wspace 覆盖掉，
    # 导致 (a) 的标题与 (b)(c) 的标题挤在一起
    save(fig, "fig4_joint_trajectory.pdf")


# =============================================================================
# fig5 命中率 vs 球速（统计量读 JSON）
# =============================================================================

def fig5(stats: dict) -> None:
    """Fig.5: (a) active-hit vs 球速（E1，Wilson CI）(b) TCP 1.0 vs 1.8（E2）。"""
    e1 = stats["E1_speed_sweep"]
    speeds = sorted(int(k) for k in e1)
    rates = [e1[str(s)]["rate"] for s in speeds]
    errs = [ci_half(e1[str(s)]) for s in speeds]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.42),
                            gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    # 中性深灰：单序列实验条件不占机制配色（绿=full 档，见 figs 6/7）
    ax.errorbar(speeds, rates, yerr=errs, color="#4D4D4D", marker="o", capsize=2.5,
                lw=1.0, label="Default limits (TCP 1.8 m/s)")
    ax.set_xlabel("Ball speed (m/s)")
    ax.set_ylabel("Active-hit rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Hit rate vs. ball speed", fontsize=9)
    ax.legend(loc="lower left", framealpha=0.9)
    style_ax(ax)

    ax = axes[1]
    e2 = stats["E2_limit_cost"]
    keys = ["TCP1.8", "TCP1.0"]
    vals = [e2[k]["rate"] for k in keys]
    errs = [ci_half(e2[k]) for k in keys]
    ns = [e2[k]["n_valid"] for k in keys]
    # 中性灰阶：限速档不是方法档，不复用四档配色语义
    ax.bar(np.arange(2), vals, yerr=errs, capsize=3, width=0.5,
           color=["#6B6B6B", "#BFBFBF"], alpha=0.95)
    # 柱顶数值 + 配对差值（judge 意见：面板须自含结论，读者不看正文也能读）
    # chat9 审稿意见：柱顶百分比与顶部配对差值的间距需要拉开，故抬高 y 上限
    for i in range(2):
        ax.text(i, vals[i] + errs[i] + 2.5, f"{vals[i]:.1f}%", ha="center",
                fontsize=8)
    # n 并入类别刻度（保持柱顶只放数值）
    ax.set_xticks(np.arange(2),
                  [f"TCP 1.8\n$n$={ns[0]}", f"TCP 1.0\n$n$={ns[1]}"])
    # 配对差值（TCP1.0 − TCP1.8，与正文 −48.7 pp 同源）；缺失时退化为边际差
    paired = e2.get("paired_TCP10_vs_TCP18", {})
    d_pp = paired.get("diff_pp")
    if d_pp is None:
        d_pp = vals[1] - vals[0]
    ax.text(0.5, 0.995, f"paired diff ${d_pp:+.1f}$ pp", ha="center", va="top",
            transform=ax.transAxes, fontsize=7.5)
    ax.set_ylabel("Active-hit rate (%)")
    ax.set_ylim(0, 116)
    ax.set_title("(b) Robot-derived TCP constraint set (7 m/s)", fontsize=9)
    style_ax(ax)

    fig.tight_layout(pad=0.3)
    save(fig, "fig5_hit_rate_vs_speed.pdf")


# =============================================================================
# fig6 机制归因（四面板，统计量全部读 JSON）
# =============================================================================

def _holm_sig(pvals: dict[str, float]) -> dict[str, bool]:
    """Holm 逐步法多重校正（α=0.05）：p 升序，阈值 α/(m-i)，首个不显著即停。"""
    m = len(pvals)
    sig: dict[str, bool] = {k: False for k in pvals}
    for i, (k, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        if p <= 0.05 / (m - i):
            sig[k] = True
        else:
            break
    return sig


def fig6(stats: dict) -> None:
    """Fig.6: (a) 标称增益两系列 (b) candidate-set 相对点目标增益热图 (c) 走廊增量热图
    full−softmin (d) 走廊独立价值 vs 扰动幅度（E7 高功效角点）。

    chat2 第二轮改版：(b) 原为 full−pt（混合两机制），拆成两张热图后
    (c) 全格 ≈0 = 「走廊被时间窗覆盖」的视觉证明；(b)(c) 共享 ±20 pp 色阶
    （chat8：±16 容不下 (0,0) 格 +17.3，扩到 ±20；标题去掉机制预判）。
    星号全部 Holm-adjusted（m=20）：raw p<0.05 有 11 格，Holm 后仅 2 格。
    """
    e3, e4, e7 = stats["E3_nominal"], stats["E4_grid"], stats["E7_corners"]
    # chat9 审稿意见：这是核心结果图，应获得更多面积（旧版 2.25 in 四子图明显变扁），
    # 面板标题缩短、方法/速度/单位移入坐标轴与图注
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 2.48))

    def asym_err(v: dict) -> tuple[float, float]:
        """配对 bootstrap CI 的上下误差条（以 diff_pp 为基准）。"""
        lo, hi = v["ci95"]
        return v["diff_pp"] - lo, hi - v["diff_pp"]

    # (a) 标称增益两系列：不再画四档率柱（与 Table I 重复），改画对点目标的增益
    ax = axes[0][0]
    speeds = [7, 9, 12]
    width = 0.32
    for mi, (key, lab, col) in enumerate([
            ("softmin_only_vs_none", "candidate-set-only", C["softmin_only"]),
            ("tube_only_vs_none", "corridor-only", C["tube_only"])]):
        vals, errs = [], [[], []]
        for sp in speeds:
            p = e3["paired"].get(f"{sp}|{key}", {})
            vals.append(p.get("diff_pp", np.nan))
            lo, hi = asym_err(p) if p else (np.nan, np.nan)
            errs[0].append(lo)
            errs[1].append(hi)
        ax.bar(np.arange(3) + (mi - 0.5) * width, vals, width, yerr=errs,
               capsize=1.6, color=col, label=lab, alpha=0.9)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_xticks(np.arange(3), [f"{s}" for s in speeds])
    ax.set_xlabel("Ball speed (m/s)")
    ax.set_ylabel("Gain (pp)")
    ax.set_title("(a) Nominal gain", fontsize=9)
    ax.legend(ncol=1, loc="upper left", framealpha=0.9, fontsize=7,
              borderpad=0.3, labelspacing=0.3, handlelength=1.2)
    style_ax(ax)

    # (b)(c) 拆分热图：共享 ±20 pp 色阶（(c) 全格 ≈0 即视觉证明）
    ts = [0, 10, 25, 50, 100]
    ss = [0.0, 0.05, 0.1, 0.2]
    heat_specs = [
        (axes[0][1], "softmin_only_vs_none", "(b) Candidate-set gain"),
        (axes[1][0], "full_vs_softmin_only", "(c) Added corridor gain"),
    ]
    for ax, cmp, title in heat_specs:
        keys = [f"t{t}|s{s}" for t in ts for s in ss]
        pvals = {k: e4["per_cell_paired"][f"{k}|{cmp}"]["p_mcnemar"] for k in keys}
        sig = _holm_sig(pvals)
        gain = np.full((len(ts), len(ss)), np.nan)
        for i, t in enumerate(ts):
            for j, s in enumerate(ss):
                k = f"t{t}|s{s}"
                gain[i, j] = e4["per_cell_paired"][f"{k}|{cmp}"]["diff_pp"]
        im = ax.imshow(gain, aspect="auto", cmap="RdBu_r", vmin=-20, vmax=20)
        ax.set_xticks(range(len(ss)), [f"{s:.2f}" for s in ss])
        ax.set_yticks(range(len(ts)), [f"{t}" for t in ts])
        ax.set_xlabel("Spatial max $s$ (m), 9 m/s")
        ax.set_ylabel("Time max $t$ (ms)")
        for i in range(len(ts)):
            for j in range(len(ss)):
                k = f"t{ts[i]}|s{ss[j]}"
                # 深色格（|pp|>10）用白字，保证打印对比度（judge 意见）
                col_txt = "w" if abs(gain[i, j]) > 10 else "k"
                ax.text(j, i, f"{gain[i, j]:+.0f}{'*' if sig[k] else ''}",
                        ha="center", va="center", fontsize=7.5, color=col_txt)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85, label="pp", pad=0.02)

    # (d) 走廊独立价值 vs 扰动幅度（E7 高功效角点，9 m/s, t=0；不连线避免趋势暗示）
    ax = axes[1][1]
    s_list, g_list, e_list, p_list, n_list = [], [], [], [], []
    for s in (0.2, 0.3, 0.4):
        p = e7["paired"].get(f"9|s{s}|t0|tube_only_vs_none", {})
        if p.get("n_paired"):
            s_list.append(s)
            g_list.append(p["diff_pp"])
            e_list.append(asym_err(p))
            p_list.append(p["p_mcnemar"])
            n_list.append(p["n_paired"])
    for s, g, (lo, hi), pv in zip(s_list, g_list, e_list, p_list):
        ax.errorbar([s], [g], yerr=[[lo], [hi]], marker="o", capsize=2.5,
                    lw=0, color=C["tube_only"])
        ax.text(s, g + hi + 0.5, f"$p$${fmt_p(pv)}$", fontsize=7, ha="center")
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_ylim(-2.0, 9.0)
    ax.set_xlim(0.17, 0.44)
    # chat9 审稿意见：(d) 横轴只保留实际采样位置（去掉无用的 0.25/0.35 刻度）
    ax.set_xticks([0.2, 0.3, 0.4])
    ax.set_xlabel("Spatial max $s$ (m), 9 m/s, $t{=}0$")
    ax.set_ylabel("Corridor gain (pp)")
    ax.set_title("(d) Corridor under spatial offsets", fontsize=9)
    style_ax(ax)

    fig.tight_layout(pad=0.3, w_pad=2.2)
    save(fig, "fig6_tube_robustness.pdf")


# =============================================================================
# fig7 实时性能（三段耗时改紧凑数值表）
# =============================================================================

def fig_realtime(stats: dict, timing_path: Path = DATA / "exp18_fig_assets/timing.json") -> None:
    """打印 Fig.8（实时预算）: (a) 三段耗时数值表 (b) 逐步延迟 ECDF。"""
    t = json.loads(Path(timing_path).read_text(encoding="utf-8"))
    sync, async_ = t["sync"], t["async"]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.34),
                            gridspec_kw={"width_ratios": [1.3, 1.7]})

    # (a) 数值表：只有 3 个点，表格比 log 坐标轴更紧凑也不误导
    ax = axes[0]
    ax.axis("off")
    # 表值全部从 timing.json 读取（与正文/artifact 同源，杜绝硬编码漂移）；
    # first plan 只统计 step=0 且 iters>=30 的首次求解（不含 ~4ms JT 热身步）。
    ss = sync["steady_state"]
    rows = [
        ("First plan (30 it.), mean", sync["first_plan"]["mean_ms"]),
        ("Steady far (0 it.), mean", sync["by_iters"]["0"]["mean_ms"]),
        ("Steady near (5 it.), mean", sync["by_iters"]["5"]["mean_ms"]),
        ("Steady-state, median", ss["median_ms"]),
        ("Steady-state, p95", ss["p95_ms"]),
        ("Steady-state, max", ss["max_ms"]),
    ]
    # 标题已含 (a) 与单位，不再另起一行 "Replanning time (ms)"（与标题重复）；
    # 表头横线下移到标题降部以下（judge 意见：原 0.96 横线穿过标题括号/降部）
    ax.set_title("(a) Replanning time (ms)", fontsize=9)
    ax.plot([0.07, 0.95], [0.94, 0.94], transform=ax.transAxes, lw=0.7,
            color="k")
    for i, (name, val) in enumerate(rows):
        y = 0.86 - i * 0.145
        ax.text(0.07, y, name, fontsize=7.5, va="center", transform=ax.transAxes)
        ax.text(0.93, y, f"{val:.1f}", fontsize=7.5, va="center", ha="right",
                transform=ax.transAxes, color=C["none"] if val > 100 else "k")
    ax.text(0.07, -0.02, "synchronous mode, 30 episodes; budget 150 ms/replan",
            fontsize=6.8, va="top", transform=ax.transAxes, color="dimgray")

    # (b) 延迟超越概率 P(L > t)，对数纵轴（chat9 审稿意见：普通 CDF 把尾部压在
    # 最上方 0.04%，尾部放大 inset 又与图例重叠且字号只有 5.8 pt；改为
    # survival function + log y，尾部直接可见，不再需要 inset）
    ax = axes[1]
    for tag, col, ls in (("sync", "#4D4D4D", "-"), ("async", "#9E9E9E", "--")):
        pool = np.sort(np.asarray(t[tag]["stall"].get("raw_pool", []), dtype=float))
        if pool.size == 0:
            continue
        n = pool.size
        # P(L>t) = 1 − F(t)：在观测点右连续取值；去掉 P=0 的尾点以适配对数轴
        x = np.concatenate([[pool[0] * 0.7], pool])
        y = np.concatenate([[1.0], 1.0 - np.arange(1, n + 1) / n])
        m = y > 0
        ax.step(x[m], y[m], where="post", color=col, ls=ls, lw=1.1,
                label=f"{tag} ({n:,} steps)")
    # 只保留 5 ms 主控制阈值线；150 ms 预算在 (a) 已标注、caption 说明
    ax.axvline(5.0, color="k", ls=":", lw=0.9)
    # 5 ms 控制周期信息移入面板标题（图内左侧被 async 下降段扫过，任何位置都会压线）
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 400)
    ax.set_ylim(4e-4, 1.6)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1], ["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"])
    ax.set_xlabel("Per-step main-loop latency (ms)")
    ax.set_ylabel("$P(L>t)$")
    ax.set_title("(b) Latency exceedance (dotted: 5 ms period)", fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
    style_ax(ax)

    fig.tight_layout(pad=0.3)
    save(fig, "fig8_realtime_performance.pdf")


# =============================================================================
# fig8 诊断（仅距离曲线，事件对齐）
# =============================================================================

def fig_diagnostic() -> None:
    """Fig.8（诊断距离）: 两个配对面板 —— (a) 50 ms 时间扰动 (b) 0.2 m 空间扰动。

    chat9 审稿意见：四个窄面板里 "contact: min X mm" 文字越出坐标框、星号盖住
    最小距离圆点、「接触时刻」与「最小距离」被混成一个标签。现合并为两个配对
    面板（每对同 seed、同扰动，两条曲线放在同一坐标系里），曲线用不同颜色 +
    线型区分，接触（星号）与最小距离（圆点，各自曲线同色）分开标注，数值统一
    用短引线放在坐标框内，其余数字进图注。

    对齐基准 = 规划器在 episode 起始时刻预测的击球步：find_hitting_point_physics
    （MuJoCo 前向仿真，含地面反弹，与 do_replan 同一函数）。
    """
    from src.robot.constants import SHOULDER_POS, WORKSPACE_RADIUS
    from src.sim.rm65_env import RM65Env
    from src.tennis.hitting import find_hitting_point_physics

    env = RM65Env(PROJECT / "src" / "robot" / "rm65_model.xml")
    # 两个配对面板：每对同 seed、同扰动，仅方法档不同（跨面板只变扰动类型）
    # 基线用深灰（与机制配色分离，避免 point-target 橙与 softmin 琥珀在灰度下混淆）
    C_PT = "#333333"
    panels = [
        ("(a) 50 ms temporal perturbation", [
            ("pt_miss_t50", "point-target (miss)", C_PT, "-", False),
            ("sm_hit_t50", "candidate-set-only (hit)", C["softmin_only"], "--", True)]),
        ("(b) 0.2 m spatial perturbation", [
            ("pt_miss_s20_s77", "point-target (miss)", C_PT, "-", False),
            ("co_hit_s20_s77", "corridor-only (hit)", C["tube_only"], "--", True)]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.50), sharex=True, sharey=True)
    stats_txt: list[str] = []
    for i, (title, curves) in enumerate(panels):
        ax = axes[i]
        for name, lab, col, ls, is_hit in curves:
            d = np.load(DATA / "exp18_fig_assets/raw" / f"{name}.npz",
                        allow_pickle=True)
            md = json.loads(d["metadata"].item())
            # 重算起始时刻预测的击球步（与 replan_core.do_replan 同源函数）
            hi = find_hitting_point_physics(
                env, np.array(md["p0"]), np.array(md["v0"]),
                SHOULDER_POS, WORKSPACE_RADIUS, len(d["timestamps"]),
            )
            k_nom = hi["k_hit"] if hi is not None else int(d["hit_step"])
            tt = (d["timestamps"] - d["timestamps"][k_nom]) * 1000
            dist = np.linalg.norm(d["ball_pos"] - d["tcp_pos"], axis=1) * 1000
            ax.plot(tt, dist, color=col, lw=1.2, ls=ls, label=lab)
            dmin = float(dist.min())
            tmin = float(tt[int(np.argmin(dist))])
            # 最小距离 = 该曲线同色实心圆点（不再让黑点被星号遮挡）
            ax.plot([tmin], [dmin], marker="o", ms=3.6, color=col, zorder=5)
            # 数值标注用短引线：miss 放下方、hit 放上方，两行文字都在坐标框内
            off = (-6, -5) if not is_hit else (10, 6)
            align = "right" if not is_hit else "left"
            va = "top" if not is_hit else "bottom"
            ax.annotate(f"{dmin:.0f} mm", (tmin, dmin), textcoords="offset points",
                        xytext=off, fontsize=7, ha=align, va=va, color=col,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=col,
                                        shrinkA=0, shrinkB=2))
            if is_hit:
                hs = int(d["hit_step"])
                if 0 < hs < len(d["timestamps"]):
                    t_c = float((d["timestamps"][hs] - d["timestamps"][k_nom]) * 1000)
                    ax.plot([t_c], [dist[hs]], marker="*", ms=6.5, color="k",
                            zorder=6, mew=0.6)
                    # 接触时刻单独标注（与最小距离分离，短引线指向星号）
                    ax.annotate("contact", (t_c, dist[hs]), textcoords="offset points",
                                xytext=(16, 10), fontsize=7, ha="left", va="bottom",
                                arrowprops=dict(arrowstyle="->", lw=0.6, color="k",
                                                shrinkA=0, shrinkB=3))
            stats_txt.append(f"{lab.split(' (')[0]} {dmin:.0f}")
        ax.axhline(120, color="gray", ls="--", lw=0.9)
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(35, 2.0e4)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("$t - t_{hit}^{nom}$ (ms)")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=7,
                  handlelength=1.8, borderpad=0.3)
        style_ax(ax)
    axes[0].set_ylabel("Ball–racket center dist. (mm)")
    axes[0].annotate("racket radius 0.12 m", xy=(0.03, 0.05),
                     xycoords="axes fraction", fontsize=6.8, color="dimgray")
    print("fig7 最小距离:", "; ".join(stats_txt))

    fig.tight_layout(pad=0.3)
    save(fig, "fig7_diagnostic.pdf")


# =============================================================================
# Table I / II（统计量读 JSON，含 n 与配对 p）
# =============================================================================

def tables(stats: dict, out_dir: Path | None = None) -> None:
    """生成 Table I（含 n）与 Table II（含配对 p 与 DiD p）。"""
    e3, e4, e7, e8 = (stats["E3_nominal"], stats["E4_grid"],
                      stats["E7_corners"], stats["E8_limit_mechanism"])
    if out_dir is None:
        out_dir = OUT / "table_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    def cell(kind: str, key: str) -> float:
        src = {"E3": e3["cells"], "E4": e4["merged_20cell"],
               "E7": e7["cells"]}[kind]
        return src[key]["rate"]

    def n_of(kind: str, key: str) -> int:
        src = {"E3": e3["cells"], "E4": e4["merged_20cell"],
               "E7": e7["cells"]}[kind]
        return src[key]["n_valid"]

    rows12 = [
        ("Nominal 7 m/s", "E3", *[f"7|{m}" for m in MODE_ORDER]),
        ("Nominal 9 m/s", "E3", *[f"9|{m}" for m in MODE_ORDER]),
        ("Nominal 12 m/s", "E3", *[f"12|{m}" for m in MODE_ORDER]),
    ]
    rows_grid = [("Perturbed grid (20 cells, 9 m/s)", "E4",
                  *[m for m in MODE_ORDER])]
    rows_corner = [("Corner 12 m/s, $t{=}0$, $s{=}0.2$ m", "E7",
                    *[f"12|s0.2|t0|{m}" for m in MODE_ORDER])]
    rows_all = rows12 + rows_grid + rows_corner

    def ci_half_pp(kind: str, key: str) -> float:
        """Wilson 95% CI 半宽（pp）。"""
        src = {"E3": e3["cells"], "E4": e4["merged_20cell"],
               "E7": e7["cells"]}[kind]
        lo, hi = src[key]["ci95"]
        return (hi - lo) / 2.0

    # caption 的数字全部由 stats JSON 计算（与正文数字同源，避免手改失同步）
    nom_hw = [ci_half_pp(k, key) for _, k, *keys in rows12 for key in keys]
    gc_hw = [ci_half_pp(k, key)
             for _, k, *keys in rows_grid + rows_corner for key in keys]
    nom_n = [n_of(k, key) for _, k, *keys in rows12 for key in keys]

    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Active-hit rate (\%) of the four configurations across conditions;",
             r"$n$ = valid runs per configuration, given as the range across the four "
             r"configurations (the perturbation-grid row pools its $20$ cells). Wilson 95\% "
             r"intervals are "
             f"$\\pm{min(nom_hw):.1f}$--${max(nom_hw):.1f}$ pp for the nominal rows "
             f"($n{{=}}{min(nom_n)}$--${max(nom_n)}$) and "
             f"$\\pm{min(gc_hw):.1f}$--${max(gc_hw):.1f}$ pp for the grid and corner rows; exact intervals are in the "
             r"artifact.}",
             r"\label{tab:ablation}",
             r"\begin{tabular}{lccccrc}", r"\toprule",
             r"Condition & full & corridor-only & candidate-set-only & point-target & $n$ & Design \\",
             r"\midrule"]
    kind_label = {"E3": "nominal four-tier", "E4": "perturbation grid",
                  "E7": "high-power corners"}
    for label, kind, *keys in rows_all:
        vals = " & ".join(f"{cell(kind, k):.1f}" for k in keys)
        ns = [n_of(kind, k) for k in keys]
        n_str = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}--{max(ns)}"
        lines.append(f"{label} & {vals} & {n_str} & {kind_label[kind]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (out_dir / "table1_comparison.tex").write_text("\n".join(lines), encoding="utf-8")

    # Table II：TCP 限速层内配对 p + 跨层配对 DiD
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Gain (pp) of each configuration over the point-target "
             r"baseline under the two TCP speed caps ($7$ m/s, $n=394$ paired "
             r"seeds); $p$ from the exact paired McNemar test; DiD is the paired "
             r"per-seed difference in gain (mean with paired-bootstrap $95\%$ CI; "
             r"sign test on the per-seed direction); all $p$-values are unadjusted "
             r"tests for the pre-specified mechanism comparisons.}",
             r"\label{tab:limits}",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lcccr}", r"\toprule",
             r"Config. & \multicolumn{2}{c}{Gain over point-target (pp)} & "
             r"\multicolumn{2}{c}{$p$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r" & TCP 1.8 & TCP 1.0 & TCP 1.8 & TCP 1.0 \\", r"\midrule"]
    for mode in ("tube_only", "softmin_only", "full"):
        p18 = e8["paired"].get(f"TCP1.8|{mode}_vs_none", {})
        p10 = e8["paired"].get(f"TCP1.0|{mode}_vs_none", {})
        lines.append(
            f"{MODE_LABELS[mode]} & ${p18.get('diff_pp', 0):+.1f}$ & "
            f"${p10.get('diff_pp', 0):+.1f}$ & ${fmt_pv(p18.get('p_mcnemar', 1))}$ & "
            f"${fmt_pv(p10.get('p_mcnemar', 1))}$ \\\\")
    did = e8["did"]
    lines += [r"\midrule",
              r"\multicolumn{5}{l}{Paired DiD (gain shift from TCP 1.8 to "
              r"TCP 1.0):} \\"]
    for mode in ("tube_only", "softmin_only", "full"):
        d = did.get(mode, {})
        lo, hi = d.get("ci95", [np.nan, np.nan])
        lines.append(f"\\quad {MODE_LABELS[mode]} & \\multicolumn{{4}}{{l}}"
                     f"{{${d.get('mean_did_pp', 0):+.1f}$ pp "
                     f"(CI $[{lo:.1f},{hi:.1f}]$; sign test "
                     f"$p{fmt_p(d.get('p_sign', 1))}$)}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out_dir / "table2_ablation.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"已保存 {out_dir}/table1_comparison.tex + table2_ablation.tex")


# =============================================================================
# 入口
# =============================================================================

def main() -> None:
    """按 --fig 参数生成图表。"""
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fig", nargs="*", default=["3", "3alt", "4", "5", "6", "7", "8", "table"])
    ap.add_argument("--stats", type=Path, default=None,
                    help="统计 JSON 路径（默认自动解析 paper/planning 或 experiment_data）")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出目录（默认 paper/figures；表格写入 <out>/table_data）")
    args = ap.parse_args()
    if args.out is not None:
        OUT = args.out
    want = set(args.fig)
    stats = load_stats(args.stats)
    if "1hero" in want:
        fig1_hero()
    if "3" in want:
        fig3_3d()
    if "3alt" in want:
        fig3_alt_2d()
    if "4" in want:
        fig4()
    if "5" in want:
        fig5(stats)
    if "6" in want:
        fig6(stats)
    if "7" in want:
        fig_diagnostic()
    if "8" in want:
        fig_realtime(stats)
    if "table" in want:
        tables(stats)


if __name__ == "__main__":
    main()
