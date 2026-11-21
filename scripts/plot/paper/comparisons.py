"""机制对比图：fig5（标称/球速）、fig5 交互备选、fig6（机制归因热图）。

exp26 增益 CSV 的读取与置信区间计算集中在本模块；绘图入口延迟初始化 pyplot。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import _data_path
from .style import C, ci_half, fmt_p, pyplot, save, style_ax


def _exp26_paired_gains(csv_path: Path) -> dict[tuple[float, float], dict[str, float]]:
    """exp26 速度×TCP 网格：每格每档相对点目标的配对增益（pp，同 seed 配对）。

    无裁决行（error 非空）剔除；配对差值取两档共同有效 seed 的命中差均值。
    """
    import csv as _csv
    rows: dict[tuple[str, float, float], dict[int, bool]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            if r.get("error"):
                continue
            cid = r["config_id"]
            abl = cid.split("__")[0].replace("ablation", "")
            spd = float(cid.split("ballspeed")[1].split("__")[0])
            tcp = float(cid.split("maxtcp")[1].split("__")[0])
            rows.setdefault((abl, spd, tcp), {})[int(r["seed"])] = (
                r["hit_type"].strip().lower() == "active")
    gains: dict[tuple[float, float], dict[str, float]] = {}
    for spd in (7.0, 9.0):
        for tcp in (1.0, 1.4, 1.8):
            base = rows[("none", spd, tcp)]
            cell: dict[str, float] = {}
            for t in ("full", "tube_only", "softmin_only"):
                trow = rows[(t, spd, tcp)]
                common = sorted(set(base) & set(trow))
                cell[t] = float(np.mean([trow[s] - base[s] for s in common]) * 100.0)
            gains[(spd, tcp)] = cell
    return gains


def _exp26_gains_ci(csv_path: Path, n_boot: int = 20000) -> dict:
    """exp26：每格每档相对点目标的配对增益 + 配对 bootstrap 95% CI（pp）。

    返回 {(spd, tcp): {tier: (point, lo, hi)}}；与 analyze_chat12_night.py 的
    paired_boot_ci 同口径（同 seed 配对、seed=0、两万次重抽），保证图与正文同源。
    """
    import csv as _csv
    rows: dict[tuple[str, float, float], dict[int, bool]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            if r.get("error"):
                continue
            cid = r["config_id"]
            abl = cid.split("__")[0].replace("ablation", "")
            spd = float(cid.split("ballspeed")[1].split("__")[0])
            tcp = float(cid.split("maxtcp")[1].split("__")[0])
            rows.setdefault((abl, spd, tcp), {})[int(r["seed"])] = (
                r["hit_type"].strip().lower() == "active")
    out: dict[tuple[float, float], dict[str, tuple[float, float, float]]] = {}
    rng = np.random.default_rng(0)
    for spd in (7.0, 9.0):
        for tcp in (1.0, 1.4, 1.8):
            base = rows[("none", spd, tcp)]
            cell: dict[str, tuple[float, float, float]] = {}
            for t in ("full", "tube_only", "softmin_only"):
                trow = rows[(t, spd, tcp)]
                common = sorted(set(base) & set(trow))
                d = np.array([trow[s] - base[s] for s in common], dtype=float) * 100.0
                n = len(d)
                idx = rng.integers(0, n, size=(n_boot, n))
                boot = np.sort(d[idx].mean(axis=1))
                lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
                cell[t] = (float(d.mean()), lo, hi)
            out[(spd, tcp)] = cell
    return out


def fig5(stats: dict) -> None:
    """Fig.5（能力与 TCP 限速代价）: (a) active-hit vs 球速（E1，Wilson CI）+ 限速档单点
    (b) 速度 × TCP 网格各档配对增益（exp26，vs 点目标）。

    chat25 审稿意见：第 5 页 Fig.4+Fig.5 两个通栏横条组合「过于细长」→ 单栏化。
    原 (c) 18 根柱横跨 7.16 in，3.39 in 下每柱仅 ~0.15 in 不可读 → 按球速拆成
    (c) 7 m/s / (d) 9 m/s 两个 9 柱面板；(a)(b) 单栏整宽竖排。
    画布 7.16×2.57 → 3.39×6.31 in（手动 add_axes 定位；数据/统计/配色零改动）。

    chat27 审稿意见（图太多太密、缺实景故事图）→ 本图瘦身 2.44 in 给新 Fig.3 实景图：
    旧的 4 面板 6.24 in → 2 面板 3.80 in——(b) 两根柱折进 (a)（同 x 上下两点 +
    配对差箭头），(c)(d) 合并为 9 槽配对点图（同一槽内 7 / 9 m/s 两枚标记相连）。
    18 个增益值、84.4/35.9/−48.7 等数字一个不少，配色与统计量（配对 bootstrap CI）不变。"""
    plt = pyplot()
    e1 = stats.get("E1_speed_sweep") or {}
    e2 = stats.get("E2_limit_cost") or {}
    gains_csv = _data_path("exp26_tcp_grid/results.csv")
    missing = []
    if not e1:
        missing.append("E1_speed_sweep")
    if not (e2.get("TCP1.8") and e2.get("TCP1.0")):
        missing.append("E2_limit_cost（TCP1.8/TCP1.0）")
    if not gains_csv.is_file():
        missing.append(str(gains_csv))
    if missing:
        print("[fig5] 跳过：数据不足，无法估计 Fig.5：" + "；".join(missing)
              + "（如实留空，不补数）")
        return
    speeds = sorted(int(k) for k in e1)
    rates = [e1[str(s)]["rate"] for s in speeds]
    errs = [ci_half(e1[str(s)]) for s in speeds]

    # chat27 审稿意见：数据图密度过高、缺少一张「有故事」的实景图 → 本图腾出空间
    # 给新的 Fig.3 实景 hero（省 ~176 pt）。两处改动：
    #   1) 旧的 (b)「84.4% vs 35.9%」两根柱折进 (a)：限速代价直接标在能力曲线同一
    #      坐标里（同 7 m/s 的上下两点 + 配对差箭头），既省一个面板又更直观；
    #   2) 旧的 (c)/(d)（7 m/s、9 m/s 各 9 柱）合并为一个 9 槽配对点图：同一槽内
    #      7 m/s（圆）与 9 m/s（三角）两枚标记用细线相连 → 18 个数值一个不少，
    #      而每槽宽 0.317 in 是旧版 18 柱横排（0.15 in/柱）的两倍。
    FW, FH = 3.39, 3.80
    fig = plt.figure(figsize=(FW, FH))

    def _box(y0_in: float, h_in: float) -> list[float]:
        """(x0, y0, w, h)：列向统一 0.44–3.29 in，纵向按英寸给。"""
        return [0.44 / FW, y0_in / FH, 2.85 / FW, h_in / FH]

    # 自下而上：底边距 0.45（放合并面板的 xlabel+刻度）→ 合并面板 1.30 →
    # 行间隙 0.67（实测：下方面板轴底装饰 29.6 pt + 上方标题 15.5 pt ≈ 45 pt）
    # → (a) 0.86 → 顶部 0.52（共享图例 + (a) 标题）
    ax_a = fig.add_axes(_box(2.42, 0.86))
    ax_b = fig.add_axes(_box(0.45, 1.30))

    # ── (a) 能力曲线 + 限速档单点（原 (b) 面板折入此处） ──
    ax = ax_a
    # 中性深灰：单序列实验条件不占机制配色（绿=full 档，见 figs 6/7）
    # chat12 审稿意见：图例补明曲线对应的配置（默认 sweep 跑的就是 full 档）
    # chat14：图例缩短——原 "Full configuration, default limits (TCP 1.8 m/s)"
    # 在 2.4 in 宽面板里越界压到 (b)（审稿意见 P7），配置全称移到图注
    ax.errorbar(speeds, rates, yerr=errs, color="#4D4D4D", marker="o", capsize=2.5,
                lw=1.0, label="full config., default limits")
    e2 = stats["E2_limit_cost"]
    v18, v10 = e2["TCP1.8"]["rate"], e2["TCP1.0"]["rate"]
    lo10, hi10 = e2["TCP1.0"]["ci95"]
    ax.errorbar([7], [v10], yerr=[[v10 - lo10], [hi10 - v10]], fmt="s", ms=4.0,
                mfc="none", mec="#6B6B6B", ecolor="#6B6B6B", capsize=2.5, lw=0.9,
                label="TCP 1.0 m/s cap")
    # 两个速率数值直接标在点旁（原 (b) 面板的柱顶百分比，去掉柱后仍需可读）
    ax.text(6.88, v18, f"{v18:.1f}", ha="right", va="center", fontsize=6.5,
            color="0.25")
    ax.text(6.88, v10, f"{v10:.1f}", ha="right", va="center", fontsize=6.5,
            color="0.25")
    paired = e2.get("paired_TCP10_vs_TCP18", {})
    d_pp = paired.get("diff_pp", v10 - v18)
    # 同 x 两点的配对差：竖直双箭头 + 数值（读者不必读正文即可取用这一结论）
    ax.annotate("", xy=(7.28, v10), xytext=(7.28, v18),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.35",
                                shrinkA=1, shrinkB=1))
    ax.text(7.45, 0.5 * (v10 + v18), f"${d_pp:+.1f}$ pp", fontsize=7,
            va="center", ha="left", color="0.2")
    ax.set_xlabel("Ball speed (m/s)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("(a) Hit rate vs. ball speed", fontsize=9)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=7.5)
    style_ax(ax)

    # ── (b) 速度 × cap 网格的配对增益（7 / 9 m/s 同一槽内相连） ──
    # chat14 P6：每格每档补配对 bootstrap 95% CI 误差棒（原只有点估计）
    # chat25：18 柱横排在 3.39 in 下不可读（0.15 in/柱）→ chat27 改为配对点图，
    # 槽宽回到 0.317 in；面板加高到 1.30 in 让两枚标记拉开（9~21 pp ≈ 10~22 pt）。
    gains = _exp26_gains_ci(gains_csv)
    tiers = [("full", "full"), ("tube_only", "tube-only"),
             ("softmin_only", "candidate-set-only")]
    tcps = (1.0, 1.4, 1.8)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax = ax_b
    for ci, tcp in enumerate(tcps):
        for ti, (key, _lab) in enumerate(tiers):
            xc = ci * 3 + ti
            v7, lo7, hi7 = gains[(7.0, tcp)][key]
            v9, lo9, hi9 = gains[(9.0, tcp)][key]
            ax.plot([xc - 0.17, xc + 0.17], [v7, v9], color="0.55", lw=0.8, zorder=2)
            ax.errorbar([xc - 0.17], [v7], yerr=[[v7 - lo7], [hi7 - v7]], fmt="o",
                        ms=3.4, color=C[key], capsize=1.2, elinewidth=0.7,
                        ecolor="0.25", zorder=3)
            ax.errorbar([xc + 0.17], [v9], yerr=[[v9 - lo9], [hi9 - v9]], fmt="^",
                        ms=3.8, color=C[key], capsize=1.2, elinewidth=0.7,
                        ecolor="0.25", zorder=3)
            # 数值标注：7 m/s 标在 CI 下端下方、9 m/s 标在上端上方 → 天然错开不相撞
            ax.text(xc - 0.17, lo7 - 1.1, f"{v7:+.0f}", ha="center", va="top",
                    fontsize=6.5)
            ax.text(xc + 0.17, hi9 + 1.1, f"{v9:+.0f}", ha="center", va="bottom",
                    fontsize=6.5)
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.set_xticks([ci * 3 + 1 for ci in range(3)],
                  [f"{tcp:.1f}" for tcp in tcps])
    ax.set_xlim(-0.75, 8.75)
    ax.set_xlabel("TCP cap (m/s)")
    ax.set_ylabel("Gain (pp)")
    # chat25：ylim 下沿 −48——负增益的数值标注写在 CI 下端下方，7×1.0 的 CI 下沿
    # 约 −33，标注到 −34 仍在轴内（渲染检测实证）。
    ax.set_ylim(-46, 40)
    # chat14 P6：面板名去「反转区域」的因果暗示，只陈述已测设置下的增益
    ax.set_title("(b) Gain vs. TCP cap", fontsize=9)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.4, color="0.3",
                              label="7 m/s"),
                       Line2D([], [], marker="^", ls="", ms=3.8, color="0.3",
                              label="9 m/s")],
              # 左上角会压住 1.0 组 9 m/s 的数值标注（"+15" 被图例框盖住，渲染实证），
              # 右下角在 1.8 组下方是空区（该组最低标注 −3 的 CI 下端 ≈ −14）
              loc="lower right", ncol=2, fontsize=7, framealpha=0.9, borderpad=0.25,
              handlelength=0.8, handletextpad=0.3, columnspacing=0.9)
    style_ax(ax)
    # chat25：三档图例提升为图顶共享，锚在顶部预留带内（0.993），避免压 (a) 标题
    fig.legend(handles=[Patch(facecolor=C[k], label=lab, alpha=0.9)
                        for k, lab in tiers],
               loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3,
               fontsize=7, framealpha=0.9, borderpad=0.3, labelspacing=0.25,
               handlelength=1.2, columnspacing=1.2)

    save(fig, "fig5_hit_rate_vs_speed.pdf")


def fig5_interaction(stats: dict) -> None:
    """Fig.5 单栏备选（chat25 变体 c2）：(c)(d) 改档位折线/点图
    （x=TCP 上限，配对 bootstrap CI 误差棒，按球速分面），预览对比用。"""
    plt = pyplot()
    e1 = stats.get("E1_speed_sweep") or {}
    e2 = stats.get("E2_limit_cost") or {}
    gains_csv = _data_path("exp26_tcp_grid/results.csv")
    if not e1 or not (e2.get("TCP1.8") and e2.get("TCP1.0")) or not gains_csv.is_file():
        print("[fig5alt] 跳过：数据不足（E1/E2 或 exp26 results.csv 缺失），不补数")
        return
    speeds = sorted(int(k) for k in e1)
    rates = [e1[str(s)]["rate"] for s in speeds]
    errs = [ci_half(e1[str(s)]) for s in speeds]

    # chat25：完全手动 axes 定位（理由与实测间隙口径见 fig5；画布 4.4→4.89 in）。
    FW, FH = 3.39, 4.89
    fig = plt.figure(figsize=(FW, FH))

    def _box(x0_in: float, y0_in: float, w_in: float, h_in: float) -> list[float]:
        """(x0, y0, w, h) 英寸 → 图坐标分数。"""
        return [x0_in / FW, y0_in / FH, w_in / FW, h_in / FH]

    ax = fig.add_axes(_box(0.44, 3.57, 2.85, 0.80))
    ax.errorbar(speeds, rates, yerr=errs, color="#4D4D4D", marker="o", capsize=2.5,
                lw=1.0, label="full config., default limits")
    ax.set_xlabel("Ball speed (m/s)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Hit rate vs. ball speed", fontsize=9)
    ax.legend(loc="lower left", framealpha=0.9)
    style_ax(ax)

    ax = fig.add_axes(_box(0.44, 2.15, 2.85, 0.75))
    e2 = stats["E2_limit_cost"]
    keys = ["TCP1.8", "TCP1.0"]
    vals = [e2[k]["rate"] for k in keys]
    errs2 = [ci_half(e2[k]) for k in keys]
    ns = [e2[k]["n_valid"] for k in keys]
    ax.bar(np.arange(2), vals, yerr=errs2, capsize=3, width=0.5,
           color=["#6B6B6B", "#BFBFBF"], alpha=0.95)
    for i in range(2):
        ax.text(i, vals[i] + errs2[i] + 2.5, f"{vals[i]:.1f}%", ha="center",
                fontsize=8)
    ax.set_xticks(np.arange(2),
                  [f"TCP 1.8 ($n$={ns[0]})", f"TCP 1.0 ($n$={ns[1]})"])
    paired = e2.get("paired_TCP10_vs_TCP18", {})
    d_pp = paired.get("diff_pp")
    if d_pp is None:
        d_pp = vals[1] - vals[0]
    ax.text(0.5, 0.90, f"paired diff ${d_pp:+.1f}$ pp", ha="center", va="top",
            transform=ax.transAxes, fontsize=7.5)
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(0, 116)
    ax.set_title("(b) Robot-derived TCP constraint set (7 m/s)", fontsize=9)
    style_ax(ax)

    gains = _exp26_gains_ci(gains_csv)
    tiers = [("full", "full"), ("tube_only", "tube-only"),
             ("softmin_only", "candidate-set-only")]
    from matplotlib.patches import Patch
    for spi, spd in enumerate((7.0, 9.0)):
        ax = fig.add_axes(_box(0.44 + spi * 1.55, 0.45, 1.30, 1.10))
        for key, lab in tiers:
            xs = [1.0, 1.4, 1.8]
            ys = [gains[(spd, tcp)][key][0] for tcp in xs]
            los = [gains[(spd, tcp)][key][0] - gains[(spd, tcp)][key][1]
                   for tcp in xs]
            his = [gains[(spd, tcp)][key][2] - gains[(spd, tcp)][key][0]
                   for tcp in xs]
            ax.errorbar(xs, ys, yerr=[los, his], marker="o", ms=3.5, lw=1.0,
                        capsize=2.0, color=C[key], alpha=0.9,
                        elinewidth=0.7, ecolor="0.35")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_xticks(xs)
        ax.set_xlabel("TCP cap (m/s)")
        ax.set_ylim(-34, 44)
        if spi == 0:
            ax.set_ylabel("Gain (pp)")
        ax.set_title(f"({('c', 'd')[spi]}) Gain at {spd:g} m/s", fontsize=9)
        style_ax(ax)
    # chat25：共享图例锚定在顶部预留带内（0.99）；bbox (0.5, 1.0) 的 "lower center"
    # 会让图例整体越出画布顶部被 PDF 裁剪（实测 ink y0<0）。
    fig.legend(handles=[Patch(facecolor=C[k], label=lab, alpha=0.9)
                        for k, lab in tiers],
               loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3,
               fontsize=7, framealpha=0.9, borderpad=0.3, labelspacing=0.25,
               handlelength=1.2, columnspacing=1.2)
    save(fig, "fig5_interaction_preview.pdf")


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
    """Fig.4（机制归因）: (a) 标称增益两系列 (b) candidate-set 相对点目标增益热图
    (c) 走廊增量热图 full−softmin (d) 走廊独立价值 vs 扰动幅度（E7 高功效角点）。

    chat2 第二轮改版：(b) 原为 full−pt（混合两机制），拆成两张热图后
    (c) 全格 ≈0 = 「走廊被时间窗覆盖」的视觉证明；(b)(c) 共享 ±20 pp 色阶
    （chat8：±16 容不下 (0,0) 格 +17.3，扩到 ±20；标题去掉机制预判）。
    星号全部 Holm-adjusted（m=20）：raw p<0.05 有 11 格，Holm 后仅 2 格。

    chat25 审稿意见：第 5 页 Fig.4+Fig.5 两个通栏横条组合「过于细长」→ 单栏化。
    原 2×2 每个面板宽 ≈3.4 in（本已 ≈ 单栏 \columnwidth 3.39 in），改为 4×1 竖排后
    各面板尺寸、热图格子与字号（9/7.5 pt）与旧版一致；画布 7.16×2.45 → 3.39×4.5 in。
    数据/统计/配色零改动（数字守恒）。

    chat25 跟进（第 5 页版式二次校平）：4.5 in 下四面板轴高实测仅 ≈35 pt，两张 5 行
    热图光栅 33.6 pt → 每行 6.7 pt 而格内标注 7.5 pt（字比行高、数字贴满格）= 审稿
    所指「过度压缩」。画布加高到 3.39×5.15 in（+0.65 in；空间由题注精简 + Fig.5
    收紧 + 末页余量腾出），tight_layout 按比例把新增空间分给四个轴 → 轴高 ≈46.7 pt、
    热图行高 6.7→8.9 pt。数据/统计/配色仍零改动。
    """
    plt = pyplot()
    _e3 = stats.get("E3_nominal") or {}
    _e4 = stats.get("E4_grid") or {}
    _e7 = stats.get("E7_corners") or {}
    if not (_e3.get("paired") and _e4.get("per_cell_paired") and _e7.get("paired")):
        print("[fig6] 跳过：数据不足（E3/E4/E7 配对统计缺失或为空），不补数")
        return
    e3, e4, e7 = stats["E3_nominal"], stats["E4_grid"], stats["E7_corners"]
    # chat9 审稿意见：这是核心结果图，应获得更多面积（旧版 2.25 in 四子图明显变扁），
    # 面板标题缩短、方法/速度/单位移入坐标轴与图注
    fig, axes = plt.subplots(4, 1, figsize=(3.39, 5.30))

    def asym_err(v: dict) -> tuple[float, float]:
        """配对 bootstrap CI 的上下误差条（以 diff_pp 为基准）。"""
        lo, hi = v["ci95"]
        return v["diff_pp"] - lo, hi - v["diff_pp"]

    # (a) 标称增益两系列：不再画四档率柱（与 Table I 重复），改画对点目标的增益
    ax = axes[0]
    speeds = [7, 9, 12]
    width = 0.32
    for mi, (key, lab, col) in enumerate([
            ("softmin_only_vs_none", "candidate-set-only", C["softmin_only"]),
            ("tube_only_vs_none", "tube-only", C["tube_only"])]):
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
        (axes[1], "softmin_only_vs_none", "(b) Candidate-set gain"),
        (axes[2], "full_vs_softmin_only", "(c) Added tube gain"),
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
    ax = axes[3]
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
    ax.set_ylabel("Tube gain (pp)")
    ax.set_title("(d) Tube under spatial offsets", fontsize=9)
    style_ax(ax)

    fig.tight_layout(pad=0.3, h_pad=0.9)
    save(fig, "fig6_tube_robustness.pdf")
