"""E1 拆分回归：旧 ``paper_figs`` facade 与 ``scripts.plot.paper`` 子包的契约。

覆盖范围：

* 重导出对象 ``is`` 相同、``__module__`` 指向新模块；
* ``import`` 不初始化 matplotlib（子进程验证）、``PROJECT`` 与 cwd 无关；
* ``resolve_stats_json`` 的显式/旧位置/数据根候选回退与错误文案；
* facade 上 ``PROJECT`` / ``DATA`` / ``OUT`` 的读写转发（含 monkeypatch 用法）；
* Table I/II 的空值 ``--``、正文无裸下划线、不可估计原因注释保留原值；
* ``build_parser`` 的默认参数；``build_table3_preview`` 的缺键报错；
* 源码结构（facade 行数、无反向 import、matplotlib 仅函数内 import）；
* 固定小样本无头渲染（Agg；PDF 落到 tmp_path，不触碰 ``paper/figures/``）。
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
PLOT_DIR = PROJECT / "scripts" / "plot"
PAPER_DIR = PLOT_DIR / "paper"
FACADE = PLOT_DIR / "paper_figs.py"
FIGURES_DIR = PROJECT / "paper" / "figures"

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

#: 旧 facade 必须保持可用的 API → 实现模块（用于同一性与 __module__ 断言）。
_EXPORTS: dict[str, tuple[str, ...]] = {
    "comparisons": (
        "_exp26_gains_ci",
        "_exp26_paired_gains",
        "_holm_sig",
        "fig5",
        "fig5_interaction",
        "fig6",
    ),
    "data": (
        "_data_path",
        "load_stats",
        "resolve_data_root",
        "resolve_stats_json",
    ),
    "diagnostics": ("fig_diagnostic", "fig_realtime"),
    "style": (
        "C",
        "apply_style",
        "ci_half",
        "fmt_p",
        "fmt_pv",
        "pyplot",
        "save",
        "style_ax",
    ),
    "tables": (
        "_TABLE3_TIERS_FALLBACK",
        "_fmt_ci_dash",
        "_fmt_dash",
        "_fmt_p_dash",
        "_fmt_p_raw",
        "_fmt_pct",
        "_table3_pdf",
        "_table3_tex",
        "_table_section",
        "_tex_escape",
        "MODE_LABELS",
        "MODE_ORDER",
        "TABLE3_ANNOTATION",
        "TABLE3_ANNOTATION_EN",
        "build_table3_preview",
        "table3_shared_input",
        "tables",
    ),
    "trajectory": (
        "_corridor_band",
        "_fig3_alt_2d_old",
        "fig1_hero",
        "fig3_3d",
        "fig3_alt_2d",
        "fig3_hero",
        "fig4",
    ),
    "cli": ("build_parser", "main"),
}

_EXPORT_CASES = [
    pytest.param(module_name, name, id=f"{module_name}.{name}")
    for module_name, names in _EXPORTS.items()
    for name in names
]


def _facade():
    """导入旧 facade（模块级缓存，不重复执行 import 副作用）。"""
    return importlib.import_module("scripts.plot.paper_figs")


def _run_python(code: str, *, cwd: Path, pythonpath: Path = PROJECT) -> str:
    """在子进程中执行代码，返回 stdout；用于验证 import 副作用与 cwd 相关性。"""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(PROJECT), str(pythonpath), existing or "") if p
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


# ---------------------------------------------------------------------------
# 1. 重导出契约
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("module_name", "name"), _EXPORT_CASES)
def test_facade_reexports_same_object(module_name: str, name: str) -> None:
    """facade 属性必须就是实现模块里的同一个对象，且 __module__ 指向新模块。"""
    facade = _facade()
    module = importlib.import_module(f"scripts.plot.paper.{module_name}")
    assert hasattr(module, name)
    assert getattr(facade, name) is getattr(module, name)
    obj = getattr(module, name)
    if callable(obj) and hasattr(obj, "__module__"):
        assert obj.__module__ == f"scripts.plot.paper.{module_name}"


def test_package_tables_name_is_module_not_function() -> None:
    """子包属性 ``paper.tables`` 是模块（函数是 ``paper.tables.tables``），避免遮蔽。"""
    package = importlib.import_module("scripts.plot.paper")
    submodule = importlib.import_module("scripts.plot.paper.tables")
    assert package.tables is submodule
    assert callable(submodule.tables)


def test_facade_keeps_main_entry() -> None:
    """旧文件末尾的 ``if __name__ == "__main__": main()`` 入口仍在。"""
    source = FACADE.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in source
    assert source.rstrip().endswith("main()")


# ---------------------------------------------------------------------------
# 2. import 副作用与 PROJECT 位置
# ---------------------------------------------------------------------------


def test_import_does_not_import_matplotlib() -> None:
    """导入子包与 facade 都不得触发 matplotlib 初始化。"""
    code = (
        "import sys;"
        "import scripts.plot.paper;"
        "import scripts.plot.paper_figs;"
        "print('matplotlib' in sys.modules)"
    )
    assert _run_python(code, cwd=PROJECT) == "False"


def test_import_package_from_foreign_cwd(tmp_path: Path) -> None:
    """在仓库外 cwd 以包方式导入：PROJECT 仍指向仓库根，且无 matplotlib 副作用。"""
    code = (
        "import sys;"
        "import scripts.plot.paper.data as d;"
        "print(d.PROJECT);"
        "print('matplotlib' in sys.modules)"
    )
    out = _run_python(code, cwd=tmp_path).splitlines()
    assert Path(out[0]) == PROJECT.resolve()
    assert out[1] == "False"


def test_legacy_path_import_from_foreign_cwd(tmp_path: Path) -> None:
    """``sys.path`` 上有 scripts/plot 时 ``import paper_figs`` 仍可用（旧用法）。"""
    code = (
        "import sys;"
        "import paper_figs as pf;"
        "print(pf.__name__);"
        "print(pf.PROJECT);"
        "print(pf.fig1_hero.__module__);"
        "print('matplotlib' in sys.modules)"
    )
    out = _run_python(code, cwd=tmp_path, pythonpath=PLOT_DIR).splitlines()
    assert out[0] == "paper_figs"
    assert Path(out[1]) == PROJECT.resolve()
    assert out[2] == "scripts.plot.paper.trajectory"
    assert out[3] == "False"


# ---------------------------------------------------------------------------
# 3. 统计 JSON 回退
# ---------------------------------------------------------------------------


def test_resolve_stats_json_prefers_explicit(tmp_path: Path) -> None:
    """显式 ``--stats`` 路径最优先。"""
    facade = _facade()
    explicit = tmp_path / "stats.json"
    explicit.write_text("{}", encoding="utf-8")
    assert facade.resolve_stats_json(explicit) == explicit


def test_resolve_stats_json_legacy_project_order(tmp_path: Path) -> None:
    """旧位置回退：paper/planning 优先，其次 experiment_data/paper_stats_active.json。"""
    facade = _facade()
    project = tmp_path / "repo"
    planning = project / "paper" / "planning" / "06-stats-active.json"
    planning.parent.mkdir(parents=True)
    planning.write_text("{}", encoding="utf-8")
    assert facade.resolve_stats_json(None, project=project) == planning

    active = project / "experiment_data" / "paper_stats_active.json"
    active.parent.mkdir(parents=True)
    planning.unlink()
    active.write_text("{}", encoding="utf-8")
    assert facade.resolve_stats_json(None, project=project) == active

    active.unlink()
    with pytest.raises(FileNotFoundError, match="paired_stats"):
        facade.resolve_stats_json(None, project=project)


def test_resolve_stats_json_data_root_candidates(tmp_path: Path) -> None:
    """给出 data_root 时按固定候选顺序查找，找不到报错且不回退旧根。"""
    facade = _facade()
    root = tmp_path / "root"
    root.mkdir()
    candidates = [
        root / "paper_stats_active.json",
        root / "06-stats-active.json",
        root / "stats" / "paper_stats_active.json",
        root / "stats" / "06-stats-active.json",
        root.parent / "stats" / "paper_stats_active.json",
        root.parent / "stats" / "06-stats-active.json",
    ]
    # 逐条验证：每个候选单独存在时必须被命中（且按顺序优先）。
    for index, candidate in enumerate(candidates):
        for other in candidates[index + 1:]:
            other.unlink(missing_ok=True)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("{}", encoding="utf-8")
        assert facade.resolve_stats_json(None, data_root=root) == candidate
        candidate.unlink()
    for other in candidates:
        other.unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError, match="paired_stats"):
        facade.resolve_stats_json(None, data_root=root)


# ---------------------------------------------------------------------------
# 4. facade 上的可变全局转发
# ---------------------------------------------------------------------------


def test_facade_star_import_exports_dynamic_globals() -> None:
    """``from paper_figs import *`` 必须继续导出 ``PROJECT``/``DATA``/``OUT``。"""
    facade = _facade()
    namespace: dict[str, object] = {}
    exec("from scripts.plot.paper_figs import *", namespace)  # noqa: S102

    for name in ("PROJECT", "DATA", "OUT"):
        assert name in facade.__all__
        assert name in namespace
        assert namespace[name] == getattr(facade, name)
    assert callable(namespace["main"])
    assert namespace["fig1_hero"] is facade.fig1_hero
    # 私有 helper 不应因通配符导入泄漏。
    assert not [name for name in namespace if name.startswith("_") and name != "__builtins__"]


def test_facade_data_global_forwarding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``monkeypatch.setattr(paper_figs, "DATA", ...)`` 必须影响 ``_data_path``。"""
    facade = _facade()
    data_module = importlib.import_module("scripts.plot.paper.data")
    monkeypatch.setattr(facade, "DATA", tmp_path / "root")
    assert facade.DATA == tmp_path / "root"
    assert data_module.DATA == tmp_path / "root"
    assert facade._data_path("exp18_fig_assets/timing.json") == (
        tmp_path / "root" / "exp18_fig_assets" / "timing.json"
    )


def test_facade_project_global_forwarding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``PROJECT`` 转发后 ``resolve_data_root(None)`` 走新的项目根。"""
    facade = _facade()
    monkeypatch.setattr(facade, "PROJECT", tmp_path)
    (tmp_path / "experiment_data").mkdir()
    assert facade.resolve_data_root(None) == (tmp_path / "experiment_data").resolve()


def test_facade_out_global_forwarding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``save()`` 读取转发后的 ``OUT``，且文件名/DPI 不变。"""
    facade = _facade()
    monkeypatch.setattr(facade, "OUT", tmp_path / "figs")
    plt = facade.pyplot()
    fig = plt.figure(figsize=(1.0, 1.0))
    fig.add_subplot(111).plot([0.0, 1.0], [0.0, 1.0])
    facade.save(fig, "smoke.pdf", dpi=150)
    target = tmp_path / "figs" / "smoke.pdf"
    assert target.is_file()
    assert target.read_bytes().startswith(b"%PDF")


# ---------------------------------------------------------------------------
# 5. Table I/II 空值与 TeX 语义
# ---------------------------------------------------------------------------


def _all_miss_stats(mode: str = "torque") -> dict:
    """构造「全部不可估计」的最小统计 JSON：缺失格一律 --，原因写入注释。"""

    def section() -> dict:
        paired: dict = {}
        for item_mode in ("tube_only", "softmin_only", "full"):
            for label in ("TCP1.8", "TCP1.0"):
                paired[f"{label}|{item_mode}_vs_none"] = {
                    "unavailable_reason": f"all_miss_{item_mode}_{label}".replace(".", "_"),
                    "n_paired": 0,
                    "diff_pp": None,
                    "ci95": None,
                    "p_mcnemar": None,
                }
        # 一条带换行的原因：注释行必须折叠成单行，不能把下一行泄漏进正文
        paired["TCP1.0|full_vs_none"]["unavailable_reason"] = "all_miss\nsecond_line"
        did: dict = {}
        for item_mode in ("tube_only", "softmin_only", "full"):
            did[item_mode] = {
                "unavailable_reason": f"no_pairs_{item_mode}",
                "n_pairs": 0,
                "mean_did_pp": None,
                "ci95": None,
                "p_sign": None,
            }
        return {
            "E3_nominal": {"cells": {}, "paired": {}},
            "E4_grid": {"merged_20cell": {}, "per_cell_paired": {}},
            "E7_corners": {"cells": {}, "paired": {}},
            "E8_limit_mechanism": {"paired": paired, "did": did},
        }

    stats = {"mode": mode, "physical": {"mode": mode, **section()}, **section()}
    return stats


def _body_lines(text: str) -> str:
    """去掉以 % 开头的 TeX 注释行，返回正文。"""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("%")
    )


def test_tables_all_miss_dashes_and_escaped_notes(tmp_path: Path) -> None:
    """Table I/II 缺档渲染 --；正文下划线全部转义；注释保留原始原因。"""
    facade = _facade()
    out_dir = tmp_path / "tables"
    facade.tables(_all_miss_stats(), out_dir)

    table1 = (out_dir / "table1_comparison.tex").read_text(encoding="utf-8")
    table2 = (out_dir / "table2_ablation.tex").read_text(encoding="utf-8")

    for name, text in (("table1", table1), ("table2", table2)):
        body = _body_lines(text)
        assert not re.search(r"(?<!\\)_", body), f"{name} 正文存在未转义的裸下划线"
        assert "--" in body

    notes = [line for line in table2.splitlines() if line.startswith("% unavailable: ")]
    assert len(notes) == 9  # 6 条配对 + 3 条 DiD
    assert all("\n" not in line for line in notes)
    assert any("all_miss_tube_only_TCP1_8" in line for line in notes)
    assert any("all_miss second_line" in line for line in notes)  # 换行原因被折叠

    # 表格内固定注释原样保留（chat22 版式说明）
    assert "% chat22：" in table2
    assert "% 验证：5 页以内 与V-E 两不超（超版）时约 ~35 pt 空白" in table2


def test_tables_physical_semantics_footnote(tmp_path: Path) -> None:
    """physical 口径：Table I 必须带口径脚注，且缺档仍是 --。"""
    facade = _facade()
    out_dir = tmp_path / "tables_physical"
    facade.tables(_all_miss_stats(), out_dir, hit_semantics="physical")
    table1 = (out_dir / "table1_comparison.tex").read_text(encoding="utf-8")
    assert "命中口径：physical" in table1
    assert "mode=torque" in table1


def test_build_table3_preview_requires_section() -> None:
    """缺少 tables.table3_shared_input 时必须 ValueError 且点名该节。"""
    facade = _facade()
    with pytest.raises(ValueError, match="table3_shared_input"):
        facade.build_table3_preview({})


# ---------------------------------------------------------------------------
# 6. CLI parser 与 main 签名
# ---------------------------------------------------------------------------


def test_build_parser_defaults() -> None:
    """``build_parser`` 保持原 argparse 参数与默认 --fig 列表。"""
    facade = _facade()
    parser = facade.build_parser()
    args = parser.parse_args([])
    assert args.fig == ["3", "3alt", "4", "5", "6", "7", "8", "table"]
    assert args.stats is None
    assert args.data_root is None
    assert args.out is None
    assert args.hit_semantics == "legacy"
    assert parser.parse_args(["--fig", "3"]).fig == ["3"]
    assert parser.parse_args(["--fig", "5alt", "table3"]).fig == ["5alt", "table3"]


def test_main_returns_none() -> None:
    """``main`` 契约签名是 ``main() -> None``。"""
    facade = _facade()
    assert facade.main.__annotations__.get("return") in (None, "None")


# ---------------------------------------------------------------------------
# 7. 源码结构
# ---------------------------------------------------------------------------


def _module_level_matplotlib_imports(tree: ast.Module) -> list[ast.AST]:
    """返回模块顶层（不在函数内）的 matplotlib import 节点。"""
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and _mentions_matplotlib(node)
    ]


def _mentions_matplotlib(node: ast.Import | ast.ImportFrom) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[0] == "matplotlib" for alias in node.names)
    return bool(node.module and node.module.split(".")[0] == "matplotlib")


def _all_matplotlib_imports(tree: ast.Module) -> list[ast.AST]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and _mentions_matplotlib(node)
    ]


def test_facade_line_budget_and_no_matplotlib() -> None:
    """facade 只做重导出：行数 < 250，且不 import matplotlib。"""
    source = FACADE.read_text(encoding="utf-8")
    assert len(source.splitlines()) < 250
    tree = ast.parse(source)
    assert _all_matplotlib_imports(tree) == []
    assert "from scripts.plot.paper" in source


def test_paper_modules_do_not_import_facade() -> None:
    """子包模块不得反向 import 旧 facade（依赖方向单向；docstring 提及不算）。"""
    for path in sorted(PAPER_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all("paper_figs" not in alias.name for alias in node.names), path.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "paper_figs" not in module, f"{path.name} 反向 import 旧 facade"


def test_data_and_tables_have_no_matplotlib() -> None:
    """data.py 不碰绘图；tables.py 只经 pyplot() 间接使用 matplotlib。"""
    for name in ("data.py", "tables.py"):
        tree = ast.parse((PAPER_DIR / name).read_text(encoding="utf-8"))
        assert _all_matplotlib_imports(tree) == [], f"{name} 出现 matplotlib import"


def test_matplotlib_imports_are_function_local() -> None:
    """除 data/tables 外，matplotlib import 也必须全部在函数内（延迟初始化）。"""
    for path in sorted(PAPER_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert _module_level_matplotlib_imports(tree) == [], f"{path.name} 顶层 import matplotlib"


def test_data_project_is_repo_root_via_parents3() -> None:
    """data.py 的 PROJECT 必须用 ``Path(__file__).resolve().parents[3]``。"""
    tree = ast.parse((PAPER_DIR / "data.py").read_text(encoding="utf-8"))
    found = False
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "PROJECT" for t in node.targets):
            continue
        sub = node.value
        assert isinstance(sub, ast.Subscript)
        parents = sub.value
        assert isinstance(parents, ast.Attribute) and parents.attr == "parents"
        resolve_call = parents.value
        assert isinstance(resolve_call, ast.Call)
        assert isinstance(resolve_call.func, ast.Attribute)
        assert resolve_call.func.attr == "resolve"
        depth = sub.slice
        assert isinstance(depth, ast.Constant) and depth.value == 3
        found = True
    assert found, "data.py 缺少 PROJECT 赋值"


# ---------------------------------------------------------------------------
# 8. 固定小样本无头渲染（Agg）
# ---------------------------------------------------------------------------


def _fig6_stats() -> dict:
    """fig6 所需最小 E3/E4/E7 配对统计（数值仅为链路 smoke）。"""
    paired: dict = {}
    for speed in (7, 9, 12):
        for comparison in ("softmin_only_vs_none", "tube_only_vs_none"):
            paired[f"{speed}|{comparison}"] = {
                "diff_pp": 1.5,
                "ci95": [0.2, 2.8],
                "p_mcnemar": 0.01,
            }
    per_cell: dict = {}
    for time_ms in (0, 10, 25, 50, 100):
        for space_m in (0.0, 0.05, 0.1, 0.2):
            for comparison in ("softmin_only_vs_none", "full_vs_softmin_only"):
                per_cell[f"t{time_ms}|s{space_m}|{comparison}"] = {
                    "diff_pp": 0.5,
                    "p_mcnemar": 0.2,
                }
    corners: dict = {}
    for space_m in (0.2, 0.3, 0.4):
        corners[f"9|s{space_m}|t0|tube_only_vs_none"] = {
            "diff_pp": 2.0,
            "ci95": [1.0, 3.0],
            "p_mcnemar": 0.02,
            "n_paired": 30,
        }
    return {
        "E3_nominal": {"paired": paired},
        "E4_grid": {"per_cell_paired": per_cell},
        "E7_corners": {"paired": corners},
    }


def _timing_payload() -> dict:
    """fig_realtime 所需最小 timing.json。"""
    return {
        "sync": {
            "steady_state": {"median_ms": 4.2, "p95_ms": 5.1, "max_ms": 9.0},
            "first_plan": {"mean_ms": 42.0},
            "by_iters": {"0": {"mean_ms": 3.1}, "5": {"mean_ms": 6.4}},
            "stall": {"raw_pool": [1.0, 1.2, 2.5, 4.0]},
        },
        "async": {"stall": {"raw_pool": [1.1, 1.5]}},
    }


def test_headless_render_writes_only_to_tmp_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fig6 + fig8 用 Agg 渲染到临时 OUT；paper/figures 不新增文件。"""
    import matplotlib

    matplotlib.use("Agg", force=True)

    before = {p.name for p in FIGURES_DIR.iterdir()} if FIGURES_DIR.is_dir() else set()

    facade = _facade()
    monkeypatch.setattr(facade, "OUT", tmp_path / "figs")
    monkeypatch.setattr(facade, "DATA", tmp_path / "data")
    timing_path = tmp_path / "timing.json"
    timing_path.write_text(json.dumps(_timing_payload()), encoding="utf-8")

    facade.fig6(_fig6_stats())
    facade.fig_realtime({}, timing_path=timing_path)

    fig6_pdf = tmp_path / "figs" / "fig6_tube_robustness.pdf"
    fig8_pdf = tmp_path / "figs" / "fig8_realtime_performance.pdf"
    for pdf in (fig6_pdf, fig8_pdf):
        assert pdf.is_file(), f"缺少输出 {pdf}"
        assert pdf.read_bytes().startswith(b"%PDF")
    assert matplotlib.get_backend().lower() == "agg"

    after = {p.name for p in FIGURES_DIR.iterdir()} if FIGURES_DIR.is_dir() else set()
    assert after == before, "渲染不得写入 paper/figures/"


def test_fig6_skips_without_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """E3/E4/E7 缺失时打印跳过、不补数、不产文件。"""
    facade = _facade()
    monkeypatch.setattr(facade, "OUT", tmp_path / "figs")
    facade.fig6({})
    out = capsys.readouterr().out
    assert "[fig6] 跳过" in out
    assert not (tmp_path / "figs" / "fig6_tube_robustness.pdf").exists()
