"""诊断图：fig7（配对距离曲线）、fig8（实时延迟预算）。

fig7 需要 MuJoCo 前向仿真（函数内才把仓库根加入 sys.path 并导入 src），
fig8 只读 timing.json；两者均在绘图入口延迟初始化 pyplot。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import data as _data
from .data import _data_path
from .style import C, pyplot, save, style_ax


def fig_realtime(stats: dict, timing_path: Path | None = None) -> None:
    """打印 Fig.8（实时预算）: (a) 三段耗时数值表 (b) 逐步延迟 ECDF。"""
    plt = pyplot()
    timing_path = (Path(timing_path) if timing_path is not None
                   else _data_path("exp18_fig_assets/timing.json"))
    t = json.loads(Path(timing_path).read_text(encoding="utf-8"))
    sync = t["sync"]

    # chat21 审稿意见：面板 (a) 的 6 行数值表上下压缩过度（行距 5.09 pt / 字号
    # 7.5 pt，相邻行墨迹相接）→ 画布高度 1.15 → 1.45 in，行距同步放开。
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.45),
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
    ax.plot([0.07, 0.95], [0.93, 0.93], transform=ax.transAxes, lw=0.7,
            color="k")
    # 行位随画布加高重排：首行离表头横线 0.10 轴高、行距 0.152 轴高
    # （实测 1.45 in 画布下行距 ≈ 8.8 pt，首行墨迹离横线 ≈ 2 pt）
    for i, (name, val) in enumerate(rows):
        y = 0.83 - i * 0.152
        ax.text(0.07, y, name, fontsize=7.5, va="center", transform=ax.transAxes)
        ax.text(0.93, y, f"{val:.1f}", fontsize=7.5, va="center", ha="right",
                transform=ax.transAxes, color=C["none"] if val > 100 else "k")
    ax.text(0.07, -0.12, "synchronous mode, 30 episodes; budget 150 ms/replan",
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
    plt = pyplot()
    # 仅绘图入口按需把仓库根加入 sys.path（不在 import 时产生副作用）
    if str(_data.PROJECT) not in sys.path:
        sys.path.insert(0, str(_data.PROJECT))
    from src.robot.constants import SHOULDER_POS, WORKSPACE_RADIUS
    from src.sim.rm65_env import RM65Env
    from src.tennis.hitting import find_hitting_point_physics

    env = RM65Env(_data.PROJECT / "src" / "robot" / "rm65_model.xml")
    # 两个配对面板：每对同 seed、同扰动，仅方法档不同（跨面板只变扰动类型）
    # 基线用深灰（与机制配色分离，避免 point-target 橙与 softmin 琥珀在灰度下混淆）
    C_PT = "#333333"
    panels = [
        ("(a) 50 ms temporal perturbation", [
            ("pt_miss_t50", "point-target (miss)", C_PT, "-", False),
            ("sm_hit_t50", "candidate-set-only (hit)", C["softmin_only"], "--", True)]),
        ("(b) 0.2 m spatial perturbation", [
            ("pt_miss_s20_s77", "point-target (miss)", C_PT, "-", False),
            ("co_hit_s20_s77", "tube-only (hit)", C["tube_only"], "--", True)]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.50), sharex=True, sharey=True)
    stats_txt: list[str] = []
    for i, (title, curves) in enumerate(panels):
        ax = axes[i]
        for name, lab, col, ls, is_hit in curves:
            d = np.load(_data_path(f"exp18_fig_assets/raw/{name}.npz"),
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
