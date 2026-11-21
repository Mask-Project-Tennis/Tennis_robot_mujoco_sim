"""绘图样式层：matplotlib 延迟初始化、全局 rcParams、坐标轴样式与保存。

matplotlib 不在 import 时初始化：只有进入明确绘图入口（各 fig* 函数 /
``save`` / ``_table3_pdf``）才调用 :func:`pyplot`，因此 ``import paper_figs``
不会触发后端/字体/图形初始化，也不会弹窗。
"""

from __future__ import annotations

from . import data as _data

C = {
    "full": "#009E73",        # 绿
    "tube_only": "#0072B2",   # 蓝（档位键名 tube_only，标签同写 tube-only）
    "softmin_only": "#E69F00",  # 琥珀
    "none": "#D55E00",        # 橙红
    "ball": "#D55E00",
    "racket": "#4D4D4D",      # 球拍中性深灰：不与走廊蓝/softmin 琥珀的机制语义冲突
    "corridor": "#0072B2",    # tube 固定蓝色，与 tube_only (tube-only) 一致（键名保留历史写法）
}


_RC_PARAMS = {
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
}

_INITIALIZED = False


def apply_style() -> None:
    """初始化 matplotlib：Agg 后端 + 论文全局 rcParams；幂等，可重复调用。"""
    global _INITIALIZED
    if _INITIALIZED:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_RC_PARAMS)
    _INITIALIZED = True


def pyplot():
    """返回已应用全局样式的 ``matplotlib.pyplot`` 模块（函数内延迟导入）。"""
    apply_style()
    import matplotlib.pyplot as plt

    return plt


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



def save(fig, name: str, dpi: int = 300) -> None:
    """保存图到输出目录（原生尺寸，无 tight bbox 以保持字号恒定）。

    dpi 只影响位图类面板的嵌入分辨率（矢量文字/线条不受影响）：默认 300；
    实景 hero 这类「源渲染就是高分辨率」的图用 600，避免 matplotlib 把渲染图
    降采样到 300 dpi（渲染素材与画布同比例，1:1 嵌入不重采样）。
    """
    plt = pyplot()
    _data.OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(_data.OUT / name, dpi=dpi)
    plt.close(fig)
    print(f"已保存 {_data.OUT / name}")
