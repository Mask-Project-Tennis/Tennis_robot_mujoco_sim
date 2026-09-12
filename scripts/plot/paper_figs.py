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
STATS_JSON = PROJECT / "paper" / "planning" / "06-stats-active.json"

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
    "racket": "#0072B2",
    "corridor": "#E69F00",
}
MODE_LABELS = {"full": "full", "tube_only": "corridor-only",
               "softmin_only": "softmin-only", "none": "point-target"}
MODE_ORDER = ["full", "tube_only", "softmin_only", "none"]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def load_stats() -> dict:
    """读取配对统计 JSON（active-hit 口径，图与正文同源）。"""
    if not STATS_JSON.exists():
        raise FileNotFoundError(
            f"缺少 {STATS_JSON}；请先运行 scripts/extract/paired_stats.py")
    return json.loads(STATS_JSON.read_text(encoding="utf-8"))


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
    """保存到 paper/figures（原生尺寸，无 tight bbox 以保持字号恒定）。"""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=300)
    plt.close(fig)
    print(f"已保存 paper/figures/{name}")


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
    fig.savefig(OUT / "fig3_tube_corridor.pdf", dpi=300)
    plt.close(fig)
    print("已保存 paper/figures/fig3_tube_corridor.pdf （3D 正文版）")


def fig3_alt_2d(npz_path: Path = DATA / "exp18_fig_assets/raw/a_hit_clean.npz") -> None:
    """Fig.3（备选版）: 2D 双面板 —— (a) 侧视 (b) 俯视，走廊画成半透明带。

    与技术选型无关，只与版式有关：画布宽度与 3D 正文版一致（3.40 in = 单栏），
    面板改为上下堆叠，这样两版可直接互换、对比时只差「呈现方式」这一个变量。
    """
    d = np.load(npz_path)
    ball, tcp = d["ball_pos"], d["tcp_pos"]
    hit = int(d["hit_step"])
    k0 = max(0, hit - 60)
    b, r = ball[k0:hit + 3], tcp[k0:hit + 8]

    fig, axes = plt.subplots(2, 1, figsize=(3.40, 3.50))
    r_half = 0.12

    # (a) 侧视：Y–Z 平面（球的前进方向 vs 高度）
    ax = axes[0]
    ax.plot(b[:, 1], b[:, 2], color=C["ball"], lw=1.6, label="Ball trajectory")
    for sign in (+1, -1):
        ax.plot(b[:, 1] + sign * r_half * 0.0, b[:, 2] + sign * r_half,
                color=C["corridor"], lw=0.9, ls="--")
    ax.fill_between(b[:, 1], b[:, 2] - r_half, b[:, 2] + r_half,
                    color=C["corridor"], alpha=0.18, lw=0,
                    label="Corridor ($\\pm0.12$ m)")
    ax.plot(r[:, 1], r[:, 2], color=C["racket"], lw=1.4, label="Racket center")
    ax.scatter(b[-1, 1], b[-1, 2], color="k", marker="*", s=40, zorder=5)
    ax.annotate("hit", (b[-1, 1], b[-1, 2]), textcoords="offset points",
                xytext=(6, 6), fontsize=8)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("(a) Side view: corridor along the ball line", fontsize=8)
    # 图例放左下：侧视图的走廊带占据上半部，左上角会压住球轨迹
    ax.legend(loc="lower left", framealpha=0.9, fontsize=6.2,
              handlelength=1.5, borderpad=0.3, labelspacing=0.25)
    style_ax(ax)

    # (b) 俯视：X–Y 平面（横向走廊宽度）
    ax = axes[1]
    # 横向走廊：以球轨迹线为轴，两侧各 r_half 的平行带（限制在轨迹跨度内）
    u = b[-1, :2] - b[0, :2]
    u = u / (np.linalg.norm(u) + 1e-12)
    nrm = np.array([-u[1], u[0]])
    off = r_half * nrm
    poly_x = [b[0, 0] + off[0], b[-1, 0] + off[0],
              b[-1, 0] - off[0], b[0, 0] - off[0]]
    poly_y = [b[0, 1] + off[1], b[-1, 1] + off[1],
              b[-1, 1] - off[1], b[0, 1] - off[1]]
    # 绘制顺序 Ball→Corridor→Racket 与 (a) 一致；走廊填充压底不遮球线
    ax.plot(b[:, 0], b[:, 1], color=C["ball"], lw=1.6, zorder=3,
            label="Ball trajectory")
    ax.fill(poly_x, poly_y, color=C["corridor"], alpha=0.2, lw=0, zorder=1,
            label="Corridor ($\\pm0.12$ m)")
    for sgn in (+1, -1):
        ax.plot([b[0, 0] + sgn * off[0], b[-1, 0] + sgn * off[0]],
                [b[0, 1] + sgn * off[1], b[-1, 1] + sgn * off[1]],
                color=C["corridor"], lw=0.9, ls="--", zorder=2)
    ax.plot(r[:, 0], r[:, 1], color=C["racket"], lw=1.4, zorder=2,
            label="Racket center")
    ax.scatter(b[-1, 0], b[-1, 1], color="k", marker="*", s=40, zorder=5)
    ax.annotate("hit", (b[-1, 0], b[-1, 1]), textcoords="offset points",
                xytext=(6, 6), fontsize=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("(b) Top view: lateral corridor width", fontsize=8)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=6.2,
              handlelength=1.5, borderpad=0.3, labelspacing=0.25)
    style_ax(ax)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig3_alt_2d.pdf", dpi=300)
    plt.close(fig)
    print("已保存 paper/figures/fig3_alt_2d.pdf （2D 备选版）")


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

    # 版式：(a)(b) 上下同宽共享时间轴（同列，竖直对照成立），(c) 占右列两行。
    # 旧版 (a) 通栏而 (b) 只有 55% 宽：上下同宽的时间轴被拉成不同比例，
    # 竖直对照失效；放大图早期是 (b) 的内嵌 inset，但 (b) 里没有足够大的
    # 空白矩形（任意位置都会压住 1.8 m/s 限速线），故升为右列独立面板。
    fig = plt.figure(figsize=(7.16, 2.72))
    # 显式给四周留白（不用 tight_layout）：默认 subplot 参数会让坐标轴只占 125%-90% 宽
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.55, 1],
                          left=0.055, right=0.995, top=0.88, bottom=0.155,
                          hspace=0.62, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    # (a) 相对初始位形的关节偏差：曲线彼此分离，能读出各关节的挥拍行程
    q0 = da["q_actual"][0]
    for j in range(6):
        ax.plot(t_a, (da["q_actual"][:, j] - q0[j]) * 180 / np.pi,
                color=colors[j], lw=1.0, label=names[j])
    ax.axvline(hit_a, color="k", ls=":", lw=0.8)
    ax.set_ylabel("Joint excursion (deg)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.30 * (hi - lo))
    ax.legend(ncol=6, loc="upper left", fontsize=7, columnspacing=0.7,
              handlelength=1.1, labelspacing=0.2, borderpad=0.25, framealpha=0.85)
    ax.set_title("(a) Joint excursions relative to the initial pose", fontsize=9)
    ax.tick_params(labelbottom=False)
    style_ax(ax)

    # (b) TCP 速度：两条曲线 + 限速带 + 命中时刻竖线
    ax = fig.add_subplot(gs[1, 0], sharex=ax)
    v_a = np.linalg.norm(np.gradient(da["tcp_pos"], axis=0), axis=1) / float(da["dt"])
    v_b = np.linalg.norm(np.gradient(db["tcp_pos"], axis=0), axis=1) / float(db["dt"])
    ax.plot(t_a, v_a, color=C["full"], lw=1.0, label="baseline")
    ax.plot(t_b, v_b, color=C["none"], lw=1.0, label="space-perturbed ($s=0.1$ m)")
    ax.axhspan(1.8, 1.98, color="gray", alpha=0.25, lw=0)
    ax.axhline(1.8, color="gray", ls="--", lw=0.8)
    ax.annotate("TCP limit 1.8 m/s", xy=(0.02, 0.818), xycoords="axes fraction",
                va="center", fontsize=7.5, color="dimgray")
    ax.axvline(hit_a, color=C["full"], ls=":", lw=0.8)
    ax.axvline(hit_b, color=C["none"], ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("TCP speed (m/s)")
    ax.set_title("(b) TCP speed profiles", fontsize=9)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=7)
    style_ax(ax)

    # (c) 命中邻域放大：两条曲线在此分离（整段画在一起时完全重合、读不出差异）
    axi = fig.add_subplot(gs[0:2, 1])
    m_a = (t_a > hit_a - 120) & (t_a < hit_a + 120)
    m_b = (t_b > hit_b - 120) & (t_b < hit_b + 120)
    axi.plot(t_a[m_a] - hit_a, v_a[m_a], color=C["full"], lw=1.0)
    axi.plot(t_b[m_b] - hit_b, v_b[m_b], color=C["none"], lw=1.0)
    axi.axhline(1.8, color="gray", ls="--", lw=0.8)
    axi.set_xlabel("$t-t_{hit}$ (ms)")
    axi.set_ylabel("TCP speed (m/s)")
    axi.set_ylim(0, max(v_a[m_a].max(), v_b[m_b].max()) * 1.15)
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

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.85),
                            gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    ax.errorbar(speeds, rates, yerr=errs, color=C["full"], marker="o", capsize=2.5,
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
    ax.set_xticks(np.arange(2), ["TCP 1.8", "TCP 1.0"])
    ax.set_ylabel("Active-hit rate (%)")
    ax.set_ylim(0, 105)
    for i in range(2):
        ax.text(i, vals[i] + errs[i] + 3, f"$n$={ns[i]}", ha="center", fontsize=7.5)
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
    """Fig.6: (a) 标称增益两系列 (b) 时间效应热图 softmin−pt (c) 走廊增量热图
    full−softmin (d) 走廊独立价值 vs 扰动幅度（E7 高功效角点）。

    chat2 第二轮改版：(b) 原为 full−pt（混合两机制），拆成两张热图后
    (c) 全格 ≈0 = 「走廊被时间窗覆盖」的视觉证明；(b)(c) 共享 ±16 pp 色阶。
    星号全部 Holm-adjusted（m=20）：raw p<0.05 有 11 格，Holm 后仅 2 格。
    """
    e3, e4, e7 = stats["E3_nominal"], stats["E4_grid"], stats["E7_corners"]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 2.88))

    def asym_err(v: dict) -> tuple[float, float]:
        """配对 bootstrap CI 的上下误差条（以 diff_pp 为基准）。"""
        lo, hi = v["ci95"]
        return v["diff_pp"] - lo, hi - v["diff_pp"]

    # (a) 标称增益两系列：不再画四档率柱（与 Table I 重复），改画对点目标的增益
    ax = axes[0][0]
    speeds = [7, 9, 12]
    width = 0.32
    for mi, (key, lab, col) in enumerate([
            ("softmin_only_vs_none", "softmin-only", C["softmin_only"]),
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
    ax.set_ylabel("Gain vs. point target (pp)")
    ax.set_title("(a) Nominal gains vs. point target", fontsize=9)
    ax.legend(ncol=1, loc="upper left", framealpha=0.9, fontsize=7,
              borderpad=0.3, labelspacing=0.3, handlelength=1.2)
    style_ax(ax)

    # (b)(c) 拆分热图：共享 ±16 pp 色阶（(c) 全格 ≈0 即视觉证明）
    ts = [0, 10, 25, 50, 100]
    ss = [0.0, 0.05, 0.1, 0.2]
    heat_specs = [
        (axes[0][1], "softmin_only_vs_none",
         "(b) Temporal effect (pp), 9 m/s"),
        (axes[1][0], "full_vs_softmin_only",
         "(c) Incremental corridor effect (pp), 9 m/s"),
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
        im = ax.imshow(gain, aspect="auto", cmap="RdBu_r", vmin=-16, vmax=16)
        ax.set_xticks(range(len(ss)), [f"{s:.2f}" for s in ss])
        ax.set_yticks(range(len(ts)), [f"{t}" for t in ts])
        ax.set_xlabel("Space perturb. max (m)")
        ax.set_ylabel("Time perturb. max (ms)")
        for i in range(len(ts)):
            for j in range(len(ss)):
                k = f"t{ts[i]}|s{ss[j]}"
                ax.text(j, i, f"{gain[i, j]:+.0f}{'*' if sig[k] else ''}",
                        ha="center", va="center", fontsize=7.5, color="k")
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
    ax.set_xlabel("Space perturb. max $s$ (m)")
    ax.set_ylabel("Corridor gain vs. point target (pp)")
    ax.set_title("(d) Corridor-only value, 9 m/s (E7)", fontsize=9)
    ax.text(0.98, 0.04, "high-power corners,\n$n$=1481--2970", fontsize=6.5,
            transform=ax.transAxes, ha="right", va="bottom", color="dimgray")
    style_ax(ax)

    fig.tight_layout(pad=0.3)
    save(fig, "fig6_tube_robustness.pdf")


# =============================================================================
# fig7 实时性能（三段耗时改紧凑数值表）
# =============================================================================

def fig7(stats: dict, timing_path: Path = DATA / "exp18_fig_assets/timing.json") -> None:
    """Fig.7: (a) 三段耗时数值表 (b) 超周期步占比 (c) 最大单步延迟。"""
    t = json.loads(Path(timing_path).read_text(encoding="utf-8"))
    sync, async_ = t["sync"], t["async"]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.90),
                            gridspec_kw={"width_ratios": [1.3, 1.7]})

    # (a) 数值表：只有 3 个点，表格比 log 坐标轴更紧凑也不误导
    ax = axes[0]
    ax.axis("off")
    # 统计量显式标注（chat2 意见：919.8 与 4.0 分别来自 mean/median，须写清）。
    # 表值固定为受控空闲机实测的审计值（与正文一致）；(b) 面板 deciles 取自
    # 重录版 timing.json（分布形状对机器负载稳健，绝对值会随负载漂移）。
    rows = [("First plan (30 it.), mean", 919.8),
            ("Steady far (0 it.), mean", 3.4),
            ("Steady near (5 it.), mean", 24.8),
            ("Steady-state, median", 4.0),
            ("Steady-state, p95", 33.0),
            ("Steady-state, max", 41.0)]
    # 标题已含 (a) 与单位，不再另起一行 "Replanning time (ms)"（与标题重复）
    ax.text(0.07, 1.06, "(a) Replanning time (ms)", fontsize=9, va="top",
            transform=ax.transAxes)
    ax.plot([0.07, 0.95], [0.96, 0.96], transform=ax.transAxes, lw=0.7, color="k")
    for i, (name, val) in enumerate(rows):
        y = 0.88 - i * 0.135
        ax.text(0.07, y, name, fontsize=7.5, va="center", transform=ax.transAxes)
        ax.text(0.93, y, f"{val:.1f}", fontsize=7.5, va="center", ha="right",
                transform=ax.transAxes, color=C["none"] if val > 100 else "k")
    ax.text(0.07, 0.07, "synchronous mode, 30 episodes", fontsize=6.8, va="center",
            transform=ax.transAxes, color="dimgray")
    ax.text(0.07, -0.05, "budget: 150 ms/replan", fontsize=7, va="top",
            transform=ax.transAxes, color="dimgray")

    # (b) 逐步主循环延迟 ECDF（sync vs async）：一张图替代旧 (b)(c) 两根柱
    # 数据 = timing.json stall.deciles_pooled（各 episode 十分位点池化）
    ax = axes[1]
    for tag, col, ls in (("sync", "#4D4D4D", "-"), ("async", "#9E9E9E", "--")):
        pool = t[tag]["stall"].get("deciles_pooled", [])
        if not pool:
            continue
        n = len(pool)
        ax.step(pool, np.arange(1, n + 1) / n, where="post", color=col, ls=ls,
                lw=1.1, label=f"{tag} (n={n})")
    ax.axvline(5.0, color="k", ls=":", lw=0.9)
    ax.text(5.0, 0.97, "control period\n5 ms", fontsize=6.8, ha="left",
            va="top", transform=ax.get_xaxis_transform(), color="dimgray")
    ax.axvline(150.0, color="k", ls=":", lw=0.9)
    ax.text(150.0, 0.97, "replan budget\n150 ms", fontsize=6.8, ha="left",
            va="top", transform=ax.get_xaxis_transform(), color="dimgray")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 400)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Per-step main-loop latency (ms)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("(b) Latency distribution", fontsize=9)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7)
    style_ax(ax)

    fig.tight_layout(pad=0.3)
    save(fig, "fig7_realtime_performance.pdf")


# =============================================================================
# fig8 诊断（仅距离曲线，事件对齐）
# =============================================================================

def fig8() -> None:
    """Fig.8: 三场景球心-拍心距离曲线（对齐 nominal predicted hit time）。

    对齐基准 = 规划器在 episode 起始时刻预测的击球步：find_hitting_point_physics
    （MuJoCo 前向仿真，含地面反弹，与 do_replan 同一函数）。三个面板统一 x/y
    范围：miss 面板的 t=0 即预测击球时刻（旧版 hit_step=-1 使 t=0 落在 episode
    末步，而 caption 却写 "aligned at contact"，两者都不成立）。
    """
    from src.robot.constants import SHOULDER_POS, WORKSPACE_RADIUS
    from src.sim.rm65_env import RM65Env
    from src.tennis.hitting import find_hitting_point_physics

    env = RM65Env(PROJECT / "src" / "robot" / "rm65_model.xml")
    specs = [
        ("b_hit_space_perturb", "Corridor hit", C["full"]),
        ("c_miss_combined_perturb", "Miss", C["none"]),
        ("d_hit_noise_kf", "Noise+filter hit", C["softmin_only"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.85))
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for i, (name, label, col) in enumerate(specs):
        d = np.load(DATA / "exp18_fig_assets/raw" / f"{name}.npz", allow_pickle=True)
        md = json.loads(d["metadata"].item())
        p0 = np.array(md["p0"])
        v0 = np.array(md["v0"])
        # 重算规划器在 episode 起始时刻的预测击球步（与 replan_core.do_replan 同源）
        hi = find_hitting_point_physics(
            env, p0, v0, SHOULDER_POS, WORKSPACE_RADIUS, len(d["timestamps"])
        )
        k_nom = hi["k_hit"] if hi is not None else int(d["hit_step"])
        t = (d["timestamps"] - d["timestamps"][k_nom]) * 1000
        dist = np.linalg.norm(d["ball_pos"] - d["tcp_pos"], axis=1) * 1000  # mm
        series.append((t, dist))
        ax = axes[i]
        ax.plot(t, dist, color=col, lw=1.1)
        ax.axhline(120, color="gray", ls="--", lw=0.9)
        ax.axvline(0, color="k", ls=":", lw=0.8)
        dmin, tmin = dist.min(), t[np.argmin(dist)]
        ax.plot([tmin], [dmin], "o", color="k", ms=3.5)
        ax.annotate(f"min {dmin:.0f} mm", (tmin, dmin), textcoords="offset points",
                    xytext=(-58, 8) if i == 2 else (6, 8), fontsize=7.5)
        ax.set_yscale("log")
        ax.set_title(f"({chr(97 + i)}) {label}", fontsize=9)
        if i == 0:
            ax.set_ylabel("Ball–racket center dist. (mm)")
            ax.annotate("racket radius 120 mm", xy=(0.03, 0.06),
                        xycoords="axes fraction", fontsize=7, color="dimgray")
        else:
            ax.set_yticklabels([])
        style_ax(ax)

    # 三面板统一 x/y 范围（y 轴为共享 log 轴，隐藏 (b)(c) 刻度标签才成立）
    x_all = np.concatenate([t for t, _ in series])
    y_all = np.concatenate([dist for _, dist in series])
    for ax in axes:
        ax.set_xlim(x_all.min(), x_all.max())
        ax.set_ylim(40.0, 1.05 * float(y_all.max()))
        ax.set_xlabel("$t - t_{hit}^{nom}$ (ms)")

    fig.tight_layout(pad=0.3)
    save(fig, "fig8_tube_diagnostic.pdf")


# =============================================================================
# Table I / II（统计量读 JSON，含 n 与配对 p）
# =============================================================================

def tables(stats: dict) -> None:
    """生成 Table I（含 n）与 Table II（含配对 p 与 DiD p）。"""
    e3, e4, e7, e8 = (stats["E3_nominal"], stats["E4_grid"],
                      stats["E7_corners"], stats["E8_limit_mechanism"])
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

    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Active-hit rate (\%) of the four configurations across "
             r"conditions; $n$ is the number of valid runs per cell (Wilson 95\% "
             r"intervals are within $\pm3$ pp for the nominal rows and "
             r"$\pm2$ pp for the grid corner row).}", r"\label{tab:ablation}",
             r"\begin{tabular}{lccccr}", r"\toprule",
             r"Condition & full & corridor-only & softmin-only & point-target & $n$ \\",
             r"\midrule"]
    for label, kind, *keys in rows12 + rows_grid + rows_corner:
        vals = " & ".join(f"{cell(kind, k):.1f}" for k in keys)
        n = n_of(kind, keys[0])
        lines.append(f"{label} & {vals} & {n} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (out_dir / "table1_comparison.tex").write_text("\n".join(lines), encoding="utf-8")

    # Table II：TCP 限速层内配对 p + 跨层配对 DiD
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Gain (pp) of each configuration over the point-target "
             r"baseline under the two TCP speed caps ($7$ m/s, $n=394$ paired "
             r"seeds); $p$ from the exact paired McNemar test; DiD is the paired "
             r"per-seed difference in gain (sign test).}", r"\label{tab:limits}",
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
        lines.append(f"\\quad {MODE_LABELS[mode]} & \\multicolumn{{4}}{{l}}"
                     f"{{${d.get('mean_did_pp', 0):+.1f}$ pp "
                     f"(sign test $p{fmt_p(d.get('p_sign', 1))}$)}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out_dir / "table2_ablation.tex").write_text("\n".join(lines), encoding="utf-8")
    print("已保存 paper/figures/table_data/table1_comparison.tex + table2_ablation.tex")


# =============================================================================
# 入口
# =============================================================================

def main() -> None:
    """按 --fig 参数生成图表。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fig", nargs="*", default=["3", "3alt", "4", "5", "6", "7", "8", "table"])
    args = ap.parse_args()
    want = set(args.fig)
    stats = load_stats()
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
        fig7(stats)
    if "8" in want:
        fig8()
    if "table" in want:
        tables(stats)


if __name__ == "__main__":
    main()
