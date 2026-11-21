"""论文图表生成 —— ICRA 2027 冲刺版 v2（2026-09-13 图重制）。

本轮修订要点（对应审稿意见 + 视觉审计结论）：
1. **字号体系统一**：所有图按 figsize 原生尺寸出图（不再用 bbox_inches="tight"），
   插入时 width=\textwidth 即缩放 1.00，标签 9pt / 刻度 7.5pt / 图例 7.5pt 不再漂移；
   （旧版 fig3 因 tight bbox 裁到 2.33in 再放大 3.07 倍，字号达 27pt、占半页）
2. **投稿合规**：pdf.fonttype/ps.fonttype = 42（TrueType 嵌入），消除 Type 3 字体；
3. **统计口径**：所有统计量直接读 paper/planning/06-stats-active.json
   （active-hit 主指标 + 配对 McNemar/bootstrap），保证图与正文数字同源；
4. **术语**：全图标签用 tube / tube-only（chat29 随正文 corridor→tube 改名；代码内键名
   `C["corridor"]`、`_corridor_band`、`corridor(ax)` 保留历史写法，不随渲染文字变化）；
5. fig3 出两个版本：3D 修正版（正文用）+ 2D 双面板备选；
   fig4 改画相对初始位形的关节偏差 + 命中区放大；fig6 色条对称 ±16 并标显著格点；
   fig7 去除三点 log 图改为紧凑数值表；fig8 删除不可读的 3D 排，距离曲线事件对齐。
6. chat25（单栏化）：fig5/fig6 由通栏 7.16 in 改单栏 3.39 in 原生重排——
   fig6 2×2 → 4×1 竖排（面板尺寸与旧版一致）；fig5 (c) 18 柱按球速拆 (c)(d)
   两个 9 柱面板（备选 c2 交互点图：--fig 5alt）；插入 width=\columnwidth 缩放 1.00。

数据源：
- 统计量:   paper/planning/06-stats-active.json（由 scripts/extract/paired_stats.py 生成）
- 轨迹/诊断: experiment_data/exp18_fig_assets/raw/*.npz + timing.json

用法:
    python scripts/plot/paper_figs.py                      # 出全部
    python scripts/plot/paper_figs.py --fig 3 6 table      # 出指定项
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import data as _data
from .comparisons import fig5, fig5_interaction, fig6
from .data import load_stats, resolve_data_root
from .diagnostics import fig_diagnostic, fig_realtime
from .tables import table3_shared_input, tables
from .trajectory import fig1_hero, fig3_3d, fig3_alt_2d, fig3_hero, fig4


def build_parser() -> argparse.ArgumentParser:
    """构建论文图表命令行解析器（参数与历史脚本保持一致）。"""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fig", nargs="*",
                    default=["3", "3alt", "4", "5", "6", "7", "8", "table"])
    ap.add_argument("--data-root", type=Path, default=None,
                    help="数据根（默认 PROJECT/experiment_data）；exp18 npz / "
                         "exp26 csv / timing.json 与自动解析的统计 JSON 均从该根读取")
    ap.add_argument("--stats", type=Path, default=None,
                    help="统计 JSON 路径（默认从数据根自动解析）")
    ap.add_argument("--out", type=Path, default=None,
                    help="输出目录（默认 paper/figures；表格写入 <out>/table_data）")
    ap.add_argument("--hit-semantics", choices=("legacy", "physical"), default="legacy",
                    help="表格命中口径：legacy（默认，旧 RM65 输出）或 physical（读 stats['physical']）")
    return ap


def main() -> None:
    """生成论文图表（默认全出；数据缺失时逐项打印跳过，不补数）。"""
    args = build_parser().parse_args()

    data_root_given = args.data_root is not None
    if data_root_given:
        _data.DATA = resolve_data_root(args.data_root)
        print(f"数据根: {_data.DATA}")
    if args.out is not None:
        out = Path(args.out)
        _data.OUT = (out if out.is_absolute() else _data.PROJECT / out).resolve()
        print(f"输出目录: {_data.OUT}")
    want = set(args.fig)
    stats = load_stats(args.stats, data_root=_data.DATA if data_root_given else None)

    def _run(name: str, fn, *fn_args, **fn_kwargs) -> None:
        """执行单个图项；数据缺失时如实跳过（打印原因，不伪造结果）。"""
        try:
            fn(*fn_args, **fn_kwargs)
        except FileNotFoundError as exc:
            print(f"[{name}] 跳过：缺少数据文件（{exc}）；未验收，不补数")
        except KeyError as exc:
            print(f"[{name}] 跳过：统计 JSON 缺少键 {exc}；未验收，不补数")

    if "1hero" in want:
        _run("fig1", fig1_hero)
    if "3" in want:
        _run("fig3", fig3_3d)
    if "3alt" in want:
        _run("fig3alt", fig3_alt_2d)
    if "3hero" in want:
        _run("fig3hero", fig3_hero)
    if "4" in want:
        _run("fig4", fig4)
    if "5" in want:
        fig5(stats)
    if "5alt" in want:
        fig5_interaction(stats)
    if "6" in want:
        fig6(stats)
    if "7" in want:
        _run("fig7", fig_diagnostic)
    if "8" in want:
        _run("fig8", fig_realtime, stats)
    if "table" in want:
        _run("table", tables, stats, hit_semantics=args.hit_semantics)
    if "table3" in want:
        table3_shared_input(stats)


if __name__ == "__main__":
    main()
