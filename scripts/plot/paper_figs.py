# ruff: noqa: PLC0414
# facade 的显式重导出别名（``X as X``）是兼容契约要求：保证对象 is 相同、
# ``__module__`` 指向实现模块并避免误删。PLC0414 会把这类别名判为冗余，
# 故在本文件整体屏蔽该规则。

"""论文图表生成 —— 兼容 facade（实现已拆分到 :mod:`scripts.plot.paper`）。

本文件不再包含任何绘图/表格业务逻辑，也不 import matplotlib；所有 API 都从
子包重导出，``X as X`` 形式保证 ``__module__`` 指向真实实现模块、对象 is 不变。

``PROJECT`` / ``DATA`` / ``OUT`` 三个可变模块级全局通过 facade 的模块类转发到
:mod:`scripts.plot.paper.data`：``main()`` 的 ``--data-root`` / ``--out`` 与测试中的
``monkeypatch.setattr(paper_figs, "DATA", ...)`` 都作用于同一份状态。

用法保持不变::

    python scripts/plot/paper_figs.py --fig 3 4 5 --out /tmp/figs
    python scripts/plot/paper_figs.py --fig table3 --stats path/to/stats.json

也兼容 ``sys.path.insert(0, "scripts/plot"); import paper_figs`` 的旧导入方式。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

if not __package__:
    # 兼容顶层导入（``import paper_figs``）：补仓库根，使 ``scripts.plot.paper`` 可解析。
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

from scripts.plot.paper import data as _data
from scripts.plot.paper.cli import build_parser as build_parser
from scripts.plot.paper.cli import main as main
from scripts.plot.paper.comparisons import (
    _exp26_gains_ci as _exp26_gains_ci,
)
from scripts.plot.paper.comparisons import (
    _exp26_paired_gains as _exp26_paired_gains,
)
from scripts.plot.paper.comparisons import (
    _holm_sig as _holm_sig,
)
from scripts.plot.paper.comparisons import (
    fig5 as fig5,
)
from scripts.plot.paper.comparisons import (
    fig5_interaction as fig5_interaction,
)
from scripts.plot.paper.comparisons import (
    fig6 as fig6,
)
from scripts.plot.paper.data import (
    _data_path as _data_path,
)
from scripts.plot.paper.data import (
    load_stats as load_stats,
)
from scripts.plot.paper.data import (
    resolve_data_root as resolve_data_root,
)
from scripts.plot.paper.data import (
    resolve_stats_json as resolve_stats_json,
)
from scripts.plot.paper.diagnostics import (
    fig_diagnostic as fig_diagnostic,
)
from scripts.plot.paper.diagnostics import (
    fig_realtime as fig_realtime,
)
from scripts.plot.paper.style import (
    C as C,
)
from scripts.plot.paper.style import (
    apply_style as apply_style,
)
from scripts.plot.paper.style import (
    ci_half as ci_half,
)
from scripts.plot.paper.style import (
    fmt_p as fmt_p,
)
from scripts.plot.paper.style import (
    fmt_pv as fmt_pv,
)
from scripts.plot.paper.style import (
    pyplot as pyplot,
)
from scripts.plot.paper.style import (
    save as save,
)
from scripts.plot.paper.style import (
    style_ax as style_ax,
)
from scripts.plot.paper.tables import (
    _TABLE3_TIERS_FALLBACK as _TABLE3_TIERS_FALLBACK,
)
from scripts.plot.paper.tables import (
    MODE_LABELS as MODE_LABELS,
)
from scripts.plot.paper.tables import (
    MODE_ORDER as MODE_ORDER,
)
from scripts.plot.paper.tables import (
    TABLE3_ANNOTATION as TABLE3_ANNOTATION,
)
from scripts.plot.paper.tables import (
    TABLE3_ANNOTATION_EN as TABLE3_ANNOTATION_EN,
)
from scripts.plot.paper.tables import (
    _fmt_ci_dash as _fmt_ci_dash,
)
from scripts.plot.paper.tables import (
    _fmt_dash as _fmt_dash,
)
from scripts.plot.paper.tables import (
    _fmt_p_dash as _fmt_p_dash,
)
from scripts.plot.paper.tables import (
    _fmt_p_raw as _fmt_p_raw,
)
from scripts.plot.paper.tables import (
    _fmt_pct as _fmt_pct,
)
from scripts.plot.paper.tables import (
    _table3_pdf as _table3_pdf,
)
from scripts.plot.paper.tables import (
    _table3_tex as _table3_tex,
)
from scripts.plot.paper.tables import (
    _table_section as _table_section,
)
from scripts.plot.paper.tables import (
    _tex_escape as _tex_escape,
)
from scripts.plot.paper.tables import (
    build_table3_preview as build_table3_preview,
)
from scripts.plot.paper.tables import (
    table3_shared_input as table3_shared_input,
)
from scripts.plot.paper.tables import (
    tables as tables,
)
from scripts.plot.paper.trajectory import (
    _corridor_band as _corridor_band,
)
from scripts.plot.paper.trajectory import (
    _fig3_alt_2d_old as _fig3_alt_2d_old,
)
from scripts.plot.paper.trajectory import (
    fig1_hero as fig1_hero,
)
from scripts.plot.paper.trajectory import (
    fig3_3d as fig3_3d,
)
from scripts.plot.paper.trajectory import (
    fig3_alt_2d as fig3_alt_2d,
)
from scripts.plot.paper.trajectory import (
    fig3_hero as fig3_hero,
)
from scripts.plot.paper.trajectory import (
    fig4 as fig4,
)

#: 由 facade 动态转发到 :mod:`scripts.plot.paper.data` 的可变全局名。
_FORWARDED_GLOBALS = frozenset({"PROJECT", "DATA", "OUT"})

#: 通配符导入必须继续导出旧 facade 的全部公有名与三个可变全局。
__all__ = [
    "DATA",  # noqa: F822
    "MODE_LABELS",
    "MODE_ORDER",
    "OUT",  # noqa: F822
    "PROJECT",  # noqa: F822
    "TABLE3_ANNOTATION",
    "TABLE3_ANNOTATION_EN",
    "C",
    "apply_style",
    "build_parser",
    "build_table3_preview",
    "ci_half",
    "fig1_hero",
    "fig3_3d",
    "fig3_alt_2d",
    "fig3_hero",
    "fig4",
    "fig5",
    "fig5_interaction",
    "fig6",
    "fig_diagnostic",
    "fig_realtime",
    "fmt_p",
    "fmt_pv",
    "load_stats",
    "main",
    "pyplot",
    "resolve_data_root",
    "resolve_stats_json",
    "save",
    "style_ax",
    "table3_shared_input",
    "tables",
]


class _FacadeModule(ModuleType):
    """把 ``PROJECT`` / ``DATA`` / ``OUT`` 的读写转发给 data 模块。

    这样旧代码对 facade 属性的读取、以及 ``main()`` 对 data 模块的写入，
    看到的是同一份状态；其余属性行为与普通模块一致。
    """

    def __getattr__(self, name: str):
        if name in _FORWARDED_GLOBALS:
            return getattr(_data, name)
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        if name in _FORWARDED_GLOBALS:
            setattr(_data, name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _FacadeModule


if __name__ == "__main__":
    main()
