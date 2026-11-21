"""轨迹类图：fig1 hero、fig3 三维/2D 备选、fig4 关节偏差、fig3 实景 hero。

所有 matplotlib import 都在函数内（经 :func:`scripts.plot.paper.style.pyplot`）；
数据缺失时保持原行为：打印原因并跳过，不补数。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import data as _data
from .data import _data_path
from .style import C, pyplot, save, style_ax


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


def fig1_hero(npz_path: Path | None = None) -> None:
    """Fig.1（hero 概念示意图，顶视）: (a) 单点目标 vs (b) 候选窗口终端 + 走廊。

    chat10 审稿意见：坐标框边线像调试图、灰→橙轨迹有断口、空心圆不像球拍、
    黑色候选点没有时间层次、图内文字压线、没有机器人线索。现按「开放画布 +
    球拍轮廓 + 连续轨迹 + 橙/蓝配色」重做：
    - 两面板共享同一条连续球轨迹（窗口外浅灰、窗口内橙，重叠一步保证无断口）；
    - 球拍画成椭圆拍面 + 短柄，不用空心圆；横向半轴 ≈ 有效碰撞半径 0.12 m，
      具体数值只写在图注里；
    - 走廊 = 沿轨迹逐点 ±0.12 m 的浅蓝半透明带 + 蓝色虚线边界（与结果图同色）；
    - 图内只留三个标签（single target / candidate ball states / spatial tube），
      面板编号 (a)/(b) 置于左上角；不显示刻度与坐标框。

    坐标约定：npz 的 ball_pos 列为 (X, Y, Z)，本图顶视绘制 (x=Y, y=X)；
    所有几何（切向、法向、拍面角度）都在绘图坐标下计算。画布高度按内容
    长宽比反推，保证内容填满面板、两面板完全一致。
    """
    plt = pyplot()
    npz_path = (Path(npz_path) if npz_path is not None
                else _data_path("exp18_fig_assets/raw/a_hit_clean.npz"))
    from matplotlib.patches import Ellipse

    d = np.load(npz_path)
    ball = d["ball_pos"]
    tcp = d["tcp_pos"]
    hit = int(d["hit_step"])
    # chat12：显示段从 hit-23 延长到 hit-30（更扁的构图，同宽下留高更少，
    # 为 p1 的 Artifact 段腾出空间——见 introduction.tex 宽度 0.88 的取值）
    k0 = max(0, hit - 30)
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
    # chat12 审稿意见：裁掉图内空白——边距较原 0.14 收紧；上边界取 0.13
    # （> 走廊半宽 0.12，保证走廊带完整不被面板顶裁角——0.11 时实测裁掉约
    # 6 mm），下边界 0.115（"spatial tube" 标签所需空间较小）。
    half_x = 0.17                                       # 拍面沿轨迹半长
    half_y = 0.15                                       # 拍面横向半轴 + 余量
    x_lo = float(b[:, 1].min()) - 0.02
    # chat14：原 0.36 只够拍柄长度，椭圆拍面被右边界裁掉——加大右余量
    x_hi = float(p_face[0]) + max(0.34, half_x + 0.22)
    y_from_path = [float(b[:, 0].min()) - 0.115,        # 走廊下边界 + 标签
                   float(b[:, 0].max()) + 0.13]         # 走廊上边界（完整）
    y_lo = min(y_from_path[0], float(p_face[1]) - half_y - 0.02)
    y_hi = max(y_from_path[1], float(p_face[1]) + half_y + 0.02)
    xlim, ylim = (x_lo, x_hi), (y_lo, y_hi)

    fig_w = 3.40
    panel_w = fig_w * 0.976                             # subplots_adjust 后的面板宽
    aspect = (y_hi - y_lo) / (x_hi - x_lo)
    panel_h = aspect * panel_w
    fig_h = 2 * panel_h + 0.30                          # 两面板 + 行距/边距
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.995, bottom=0.005,
                        hspace=0.06)
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
        """球拍：椭圆拍面（横向半轴 ≈ r_half）+ 短柄（沿出球方向）。

        chat12 审稿意见：拍面改半透明填充——原不透明浅灰会把走廊边界与
        候选点盖住，削弱「一个目标变成一组目标」的视觉差异。
        """
        ang = float(np.degrees(np.arctan2(u[1], u[0])))
        ax.add_patch(Ellipse(p_face, width=2 * 0.14, height=2 * r_half,
                             angle=ang, facecolor="0.90", alpha=0.55,
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
                xytext=(-13, 11), fontsize=12.9, ha="right", va="bottom",
                color=C["ball"], zorder=7,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C["ball"],
                                shrinkA=0, shrinkB=4))
    ax.text(0.005, 0.995, "(a) Point target", transform=ax.transAxes,
            fontsize=12.6, ha="left", va="top")

    # ---------------- (b) 候选窗口终端 + 走廊 ----------------
    ax = axes[1]
    canvas(ax)
    corridor(ax)
    ball_path(ax)
    racket(ax)
    # 候选球态（窗口内 5 个时刻，中心最深、两侧渐浅）
    # chat12 审稿意见：端点透明度提高（原 0.30 在拍面处几乎不可见），
    # 保证「一组目标」在整条窗口上都读得出来
    ks = np.linspace(i_w0, i_w0 + 0.88 * (len(b) - 1 - i_w0), 5).round().astype(int)
    for k, al in zip(ks, (0.45, 0.70, 1.0, 0.70, 0.45)):
        ax.scatter([b[k, 1]], [b[k, 0]], s=34, color=C["ball"], alpha=al,
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
                textcoords="offset points", xytext=(-5, 11), fontsize=12.9,
                ha="right", va="bottom", color=C["ball"], zorder=7,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C["ball"],
                                shrinkA=0, shrinkB=3))
    mid = cw[len(cw) // 2] - 1.35 * r_half * nrm[len(cw) // 2]
    ax.annotate("spatial tube", (mid[0], mid[1]),
                textcoords="offset points", xytext=(0, -2), fontsize=12.9,
                ha="center", va="top", color=C["corridor"], zorder=7)
    # 标题字号放大后需缩短文案（原 "…terminal + tube" 会超出画布右界）
    ax.text(0.005, 0.995, "(b) Candidate window + tube",
            transform=ax.transAxes, fontsize=12.6, ha="left", va="top")

    save(fig, "fig1_hero.pdf")


def fig3_3d(npz_path: Path | None = None) -> None:
    """Fig.3（正文版）: 3D 球轨迹 + 球拍轨迹 + 连续走廊带 + 击球点。

    画布按**单栏**尺寸出图（3.40 in 宽，对应 \\columnwidth 3.5 in）：
    Axes3D 的投影盒受图高限制，若按 7.16 in 通栏铺开，墨迹只占画布 38%，
    两侧留出约 4.4 in 空白 —— 这正是旧版「页面正中一小块、大量浪费」的根因。
    """
    plt = pyplot()
    npz_path = (Path(npz_path) if npz_path is not None
                else _data_path("exp18_fig_assets/raw/a_hit_clean.npz"))
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
                    shade=False, label="Tube (half-width 0.12 m)")
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


def fig3_alt_2d(npz_path: Path | None = None) -> None:
    """Fig.3: 走廊几何（放大到候选窗口）——(a) 侧视 Y–Z (b) 顶视 Y–X。

    chat9 审稿意见：旧版坐标仍覆盖整段飞行、走廊缩在右端；"±50 ms window"
    长引线横跨半张图；图内写 "hit" 而图例叫 "Reference point"（语义冲突）；
    图例约 6.6 pt。现改为：
    - 两面板放大到候选窗口邻域、共享 Y 轴；左上角保留小型全轨迹定位图；
    - 参考点（小圆点 = 预测最优候选步的球位置）与真实接触（星号）分开标注；
    - 窗口范围用两端短竖线 + 窗口内双向箭头表示，取消长引线；
    - 图例 8 pt，条目标注统一为 "Reference point (best candidate)" / "Contact"。
    """
    from matplotlib.ticker import FormatStrFormatter
    plt = pyplot()
    npz_path = (Path(npz_path) if npz_path is not None
                else _data_path("exp18_fig_assets/raw/d_hit_noise_kf.npz"))
    d = np.load(npz_path)
    ball, tcp = d["ball_pos"], d["tcp_pos"]
    k_contact = int(d["hit_step"])
    # 参考步 = 起始时刻预测的最优候选步（与 replan_core 同源函数）
    # 仅绘图入口按需把仓库根加入 sys.path（不在 import 时产生副作用）
    if str(_data.PROJECT) not in sys.path:
        sys.path.insert(0, str(_data.PROJECT))
    from src.robot.constants import SHOULDER_POS, WORKSPACE_RADIUS
    from src.sim.rm65_env import RM65Env
    from src.tennis.hitting import find_hitting_point_physics
    md = json.loads(d["metadata"].item())
    _env = RM65Env(_data.PROJECT / "src" / "robot" / "rm65_model.xml")
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

    fig, axes = plt.subplots(2, 1, figsize=(3.40, 2.30), sharex=True)

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
    ax.annotate("tube axis", (seg[len(seg) // 2, 1], seg[len(seg) // 2, 2]),
                textcoords="offset points", xytext=(-2, -20), fontsize=7,
                ha="right", va="top", color=C["corridor"])
    ax.set_ylabel("Z (m)")
    ax.set_ylim(0.68, 1.08)
    # chat14 审稿意见：两面板 Y 刻度数与标签格式统一（0.70/1.00 与 -0.70/-0.95 同宽），
    # 面板刻度对齐后读者才能直接比较两视图的尺度
    ax.set_yticks([0.70, 0.80, 0.90, 1.00])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
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
    ax.set_yticks([-0.95, -0.85, -0.75, -0.65])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
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
    print(f"已保存 {_data.OUT / 'fig3_alt_2d.pdf'} （走廊几何放大版）")


def _fig3_alt_2d_old(npz_path: Path | None = None) -> None:
    """Fig.3（备选版）: 2D 双面板 —— (a) 侧视 (b) 俯视，走廊画成半透明带。

    与技术选型无关，只与版式有关：画布宽度与 3D 正文版一致（3.40 in = 单栏），
    面板改为上下堆叠，这样两版可直接互换、对比时只差「呈现方式」这一个变量。

    chat8 审稿意见：running cost 只在候选窗口（±50 ms）内启用，旧图把走廊带
    画满 hit−60..hit+3 整段轨迹、且无参考点/方向标记，读者无法对照式(2)(3)。
    现改为：窗口外轨迹浅灰、窗口内橙色；直线带只覆盖窗口段（首尾连线 ± r_half）；
    标出窗口起止、走廊参考点（best 步球位置）与轴线方向箭头。
    """
    plt = pyplot()
    npz_path = (Path(npz_path) if npz_path is not None
                else _data_path("exp18_fig_assets/raw/a_hit_clean.npz"))
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
    print(f"已保存 {_data.OUT / 'fig3_alt_2d.pdf'} （2D 备选版）")


def fig4(npz_a: Path | None = None,
         npz_b: Path | None = None) -> None:
    """Fig.4: (a) TCP 速度曲线（标称 vs 空间扰动 + 限速带）；(b) 命中邻域放大。

    chat12 审稿意见：原 (a) 关节余量面板相对贡献占幅偏大、信息量低
    （峰值 46% 限位余量一句话可概括），已整体移除；该数字并入图注文字，
    本图只保留与速度预算论点直接相关的 TCP 曲线。
    """
    plt = pyplot()
    npz_a = (Path(npz_a) if npz_a is not None
             else _data_path("exp18_fig_assets/raw/a_hit_clean.npz"))
    npz_b = (Path(npz_b) if npz_b is not None
             else _data_path("exp18_fig_assets/raw/b_hit_space_perturb.npz"))
    da, db = np.load(npz_a), np.load(npz_b)
    t_a = da["timestamps"] * 1000
    t_b = db["timestamps"] * 1000
    hit_a = int(da["hit_step"]) * float(da["dt"]) * 1000
    hit_b = int(db["hit_step"]) * float(db["dt"]) * 1000

    # 版式（chat12 审稿意见）：关节余量面板已移除（见 docstring），
    # 两面板并排：(a) 全程 TCP 速度、(b) 命中邻域放大。
    fig = plt.figure(figsize=(7.16, 1.30))
    # 显式给四周留白（不用 tight_layout）：默认 subplot 参数会让坐标轴只占 125%-90% 宽
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1],
                          left=0.055, right=0.995, top=0.87, bottom=0.28,
                          wspace=0.30)

    # (a) TCP 速度：两条曲线 + 限速带 + 命中时刻竖线
    ax = fig.add_subplot(gs[0, 0])
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
    ax.set_title("(a) TCP speed profiles", fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
    style_ax(ax)

    # (b) 命中邻域放大：两条曲线在此分离（整段画在一起时完全重合、读不出差异）
    # chat8 审稿意见：caption 给出 −5 ms 接触时间差，但旧版两曲线各自按自己的
    # 接触时刻对齐（差值被减掉），图中看不到两次接触事件。改为统一参考
    # t − hit_a，两次接触分别用与曲线同线型的竖线标出。
    axi = fig.add_subplot(gs[0, 1])
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
    print(f"fig4(b): dv_max={dv:.2f} m/s, hit shift={hit_b - hit_a:+.0f} ms")
    axi.legend(loc="lower left", ncol=2, fontsize=6.5, framealpha=0.9,
               handlelength=1.3, columnspacing=0.9, borderpad=0.3,
               labelspacing=0.2)
    axi.set_title("(b) Zoom: hit neighbourhood", fontsize=9)
    style_ax(axi)

    # 不用 tight_layout：它会把上面 add_gridspec 设的 hspace/wspace 覆盖掉，
    # 导致 (a) 的标题与 (b)(c) 的标题挤在一起
    save(fig, "fig4_joint_trajectory.pdf")


def fig3_hero(render_png: Path | None = None, meta_json: Path | None = None) -> None:
    """Fig.3: 一次挥拍的实景分解 —— 3D 实景 + 球轨迹残影 + 整臂多帧残影 + 走廊/窗口。

    chat27 审稿意见：数据图密度过高、缺少「球在空中走 → 手臂怎么移动 → 恰好击打」
    的那张图。本图顶替旧的 2D 走廊示意（`fig3_alt_2d.pdf`），并把旧图的定量信息
    一并带上：走廊 = 半径 0.12 m 的**三维容差管**（代价罚的是三维垂直距离
    ‖P⊥(p_ee − p_ref)‖，几何上是绕球线的圆柱而非平面带），管内候选窗口（±50 ms）
    用橙色球段标出，参考点（预测最优候选步的球位置）与真实接触分别用圆/星标出。

    实景图由 `scripts/tools/render_swing_hero.py` 渲染（输出 PNG + 关键点像素 JSON），
    本函数只做矢量标注与版面合成，图内文字因此与全文其余图同字体（STIX/Times）。
    """
    import matplotlib.patheffects as pe
    plt = pyplot()
    render_png = render_png or _data.OUT / "fig3_swing_hero_render.png"
    meta_json = meta_json or _data.OUT / "fig3_swing_hero_render.json"
    if not render_png.exists() or not meta_json.exists():
        print(f"跳过 Fig.3 hero：缺少渲染素材 {render_png.name} / {meta_json.name}"
              "（先跑 scripts/tools/render_swing_hero.py）")
        return
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    img = plt.imread(render_png)
    W, H = int(meta["width"]), int(meta["height"])
    px = {k: v for k, v in meta["px"].items() if v is not None}
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(3.387, 3.00))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(img, extent=(0, W, H, 0), interpolation="bilinear", zorder=0)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()

    def halo(txt: str, xy: tuple[float, float], *, color: str = "k",
             size: float = 7.0, ha: str = "left", va: str = "center",
             weight: str = "normal") -> None:
        """带白色描边的图内文字（压在渲染图上仍可读）。"""
        ax.text(*xy, txt, fontsize=size, ha=ha, va=va, color=color, zorder=6,
                weight=weight, path_effects=[
                    pe.withStroke(linewidth=2.0, foreground="white")])

    # 参考点（预测最优候选步的球位置 = 走廊轴过点）与真实接触
    ax.scatter(*px["reference_ball"], s=34, marker="o", facecolors="none",
               edgecolors="white", linewidths=1.5, zorder=5)
    ax.scatter(*px["reference_ball"], s=34, marker="o", facecolors="none",
               edgecolors="black", linewidths=0.7, zorder=5.1)
    ax.scatter(*px["contact_ball"], s=70, marker="*", facecolors="black",
               edgecolors="white", linewidths=0.7, zorder=6)
    halo("contact", (px["contact_ball"][0] + 18, px["contact_ball"][1] - 6))
    halo("reference", (px["reference_ball"][0] - 12, px["reference_ball"][1] + 34),
         ha="right")

    # 走廊管与候选窗口的引线标注：位置按画幅比例给（渲染分辨率可换而不必改代码；
    # 实测教训：写死像素锚点在换渲染尺寸后会把右端标签挤出画布被裁）
    ax.annotate("tube ($\\pm0.12$ m)", xy=(0.606 * W, 0.535 * H),
                xytext=(0.624 * W, 0.631 * H), fontsize=7.0, ha="center",
                va="center", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                arrowprops=dict(arrowstyle="-", lw=0.7, color="white",
                                shrinkA=2, shrinkB=3, alpha=0.9))
    ax.annotate("candidate window ($\\pm50$ ms)",
                xy=(0.695 * W, 0.503 * H), xytext=(0.981 * W, 0.556 * H),
                fontsize=7.0, ha="right", va="center", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                arrowprops=dict(arrowstyle="-", lw=0.7, color="white",
                                shrinkA=2, shrinkB=3, alpha=0.9))

    # 颜色图例（右下角空场地）：球轨迹的窗口内外配色 + 走廊管色。
    # chat27 跟进（用户意见）：窗口/走廊的语义原来只写在题注里，图内自解释性不足。
    # 参考点/接触已有图内标签，不进图例（保持图例小）；题注因此可精简。
    handles = [
        Line2D([], [], marker="o", ls="", ms=4.2, mfc="#F9F240", mec="0.35",
               mew=0.4, label="ball, outside the window"),
        Line2D([], [], marker="o", ls="", ms=4.2, mfc="#B84500", mec="0.35",
               mew=0.4, label="ball, in the $\\pm50$ ms window"),
        Patch(facecolor="#0066A3", alpha=0.35, edgecolor="0.4", lw=0.4,
              label="tube ($\\pm0.12$ m)"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.995, 0.012),
                    fontsize=6.5, framealpha=0.85, borderpad=0.35, labelspacing=0.28,
                    handlelength=1.0, handletextpad=0.45, edgecolor="0.6")
    leg.set_zorder(7)
    leg.get_frame().set_linewidth(0.4)

    save(fig, "fig3_swing_hero.pdf", dpi=600)
    print(f"已保存 {_data.OUT / 'fig3_swing_hero.pdf'} （实景 hero 版，{W}x{H} px 渲染，600 dpi 嵌入）")
