"""论文图表子包：数据寻址、样式、轨迹/对比/诊断图、LaTeX 表格与 CLI。

分层与依赖方向（单向）::

    data.py          # PROJECT/DATA/OUT 与统计 JSON 读取（不 import matplotlib）
    style.py         # matplotlib 延迟初始化 / rcParams / 公共样式与保存
    trajectory.py    # fig1 hero、fig3 3D/2D/实景 hero、fig4
    comparisons.py   # fig5、fig5 交互备选、fig6
    diagnostics.py   # fig7 配对距离、fig8 实时延迟
    tables.py        # Table I/II 与 table3 共享输入预览
    cli.py           # build_parser() / main()

``PROJECT`` / ``DATA`` / ``OUT`` 是可变模块级状态，只由 :mod:`scripts.plot.paper.data`
持有（``paper.data.PROJECT`` 等）；绘图模块运行时通过 ``paper.data`` 读取，
``main()`` 的 ``--data-root`` / ``--out`` 因此能统一影响所有图与表。
本 ``__init__`` 只做重导出，不读文件、不初始化 matplotlib。
"""

from __future__ import annotations

from .cli import build_parser as build_parser
from .cli import main as main
from .comparisons import (
    fig5 as fig5,
)
from .comparisons import (
    fig5_interaction as fig5_interaction,
)
from .comparisons import (
    fig6 as fig6,
)
from .data import (
    load_stats as load_stats,
)
from .data import (
    resolve_data_root as resolve_data_root,
)
from .data import (
    resolve_stats_json as resolve_stats_json,
)
from .diagnostics import (
    fig_diagnostic as fig_diagnostic,
)
from .diagnostics import (
    fig_realtime as fig_realtime,
)
from .style import (
    C as C,
)
from .style import (
    apply_style as apply_style,
)
from .style import (
    ci_half as ci_half,
)
from .style import (
    fmt_p as fmt_p,
)
from .style import (
    fmt_pv as fmt_pv,
)
from .style import (
    pyplot as pyplot,
)
from .style import (
    save as save,
)
from .style import (
    style_ax as style_ax,
)
from .tables import (
    MODE_LABELS as MODE_LABELS,
)
from .tables import (
    MODE_ORDER as MODE_ORDER,
)
from .tables import (
    TABLE3_ANNOTATION as TABLE3_ANNOTATION,
)
from .tables import (
    TABLE3_ANNOTATION_EN as TABLE3_ANNOTATION_EN,
)
from .tables import (
    build_table3_preview as build_table3_preview,
)
from .tables import (
    table3_shared_input as table3_shared_input,
)
from .trajectory import (
    fig1_hero as fig1_hero,
)
from .trajectory import (
    fig3_3d as fig3_3d,
)
from .trajectory import (
    fig3_alt_2d as fig3_alt_2d,
)
from .trajectory import (
    fig3_hero as fig3_hero,
)
from .trajectory import (
    fig4 as fig4,
)
