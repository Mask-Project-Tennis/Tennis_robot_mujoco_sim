"""C2 回归：``paired_stats`` 拆分为 ``paired_analysis`` 后的兼容契约。

覆盖范围（对应 ``briefs/C2.md`` 验收项）：

1. 旧 facade 与 ``paired_analysis`` 子模块的公有/私有名字是**同一个对象**（``is`` 相同），
   两种历史导入方式（``import paired_stats`` 与 ``from scripts.extract import
   paired_stats``）也指向同一批定义；
2. ``PROJECT`` 在任意 cwd 都解析到仓库根（旧实现依赖 ``__file__`` 的 ``parents[2]``，
   迁移后必须为 ``parents[3]``）；
3. CLI 契约不变：``--mode`` choices / ``--data-root`` / ``--output`` / ``--catalog`` /
   ``--dry-run``；``--help`` 退出 0、非法 mode 退出 2、显式坏 catalog 返回 2；
4. 最小 ``StatsContext`` / ``HitPolicy`` / ``RecordIdentity`` / ``PhysicalIndex``
   构造与字段默认值不变；
5. 子模块之间无循环 import（AST 依赖图 + 每个模块可独立冷启动导入）。

本测试只使用 ``tmp_path``，不读真实实验数据、不写 ``paper/`` 与 ``planning/``。
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTRACT_DIR = ROOT / "scripts" / "extract"
for _path in (ROOT, EXTRACT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import paired_stats  # 顶层导入（历史用法：sys.path.insert(extract) 后 import paired_stats）

from scripts.extract import paired_stats as facade_pkg  # 包路径导入（历史用法）
from scripts.extract.paired_analysis import (
    cli,
    comparisons,
    dataset,
    identity,
    metrics,
    report,
    sources,
)
from scripts.extract.paired_analysis import types as patypes

PYTHON = sys.executable

# p0-public-api.txt 中的 paired_stats 公有清单（41 项）。
P0_PUBLIC: tuple[str, ...] = (
    "DatasetConflictError",
    "DuplicateRecordError",
    "HitPolicy",
    "IdentityError",
    "ModeSelectionError",
    "PhysicalIndex",
    "RecordIdentity",
    "StatsContext",
    "StatsError",
    "baseline_cell",
    "build_parser",
    "build_tables",
    "case_of",
    "cell_stats",
    "cfg_of",
    "compute_summary",
    "default_output_dir",
    "e1_speed_sweep",
    "e2_limit_cost",
    "e3_nominal",
    "e4_grid",
    "e5_perception",
    "e7_corners",
    "e8_limit_mechanism",
    "e9_shared_input",
    "exclusion_report",
    "is_error",
    "load",
    "main",
    "make_context",
    "mcnemar_exact",
    "mode_of",
    "overrides_of",
    "paired_bootstrap",
    "paired_test",
    "paired_test_physical",
    "physical_of",
    "physical_stats",
    "record_identity",
    "resolve_mode",
    "shared_input_of",
    "sign_test",
    "space_perturb_of",
    "speed_of",
    "tcp_label_of",
    "tier_of",
    "time_perturb_ms_of",
    "wilson",
)

# 既有测试与 scripts/extract 兄弟脚本直接引用的私有兼容名。
FACADE_PRIVATE: tuple[str, ...] = (
    "_SHARED_RESULT_COLUMNS",
    "_did",
    "_enrichment_conflicts",
    "_enrichment_field_names",
    "_enrichment_signature",
    "_feedforward_of_cfg",
    "_load_rows",
    "_tcp_label",
)

# 每个名字的权威归属模块（cfg_of/is_error 为打破环前移到 types，metrics 仍重导出）。
OWNER = {
    **{name: patypes for name in (
        "DatasetConflictError", "DuplicateRecordError", "HitPolicy", "IdentityError",
        "ModeSelectionError", "PhysicalIndex", "RecordIdentity", "StatsContext",
        "StatsError", "cfg_of", "is_error",
    )},
    **{name: identity for name in (
        "case_of", "mode_of", "overrides_of", "record_identity", "shared_input_of",
        "speed_of", "space_perturb_of", "tcp_label_of", "tier_of", "time_perturb_ms_of",
        "_feedforward_of_cfg", "_tcp_label",
    )},
    **{name: sources for name in (
        "load", "make_context", "_SHARED_RESULT_COLUMNS", "_load_rows",
        "_enrichment_conflicts", "_enrichment_field_names", "_enrichment_signature",
    )},
    "resolve_mode": dataset,
    **{name: metrics for name in (
        "baseline_cell", "cell_stats", "mcnemar_exact", "paired_bootstrap",
        "paired_test", "paired_test_physical", "physical_of", "physical_stats",
        "sign_test", "wilson",
    )},
    **{name: comparisons for name in (
        "e1_speed_sweep", "e2_limit_cost", "e3_nominal", "e4_grid", "e5_perception",
        "e7_corners", "e8_limit_mechanism", "e9_shared_input", "exclusion_report", "_did",
    )},
    **{name: report for name in (
        "build_tables", "compute_summary", "default_output_dir",
    )},
    **{name: cli for name in ("build_parser", "main")},
}


def _run_module(statement: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """在指定 cwd 下跑一段导入语句（仓库根经 PYTHONPATH 注入，验证不依赖 cwd）。"""
    return _run_argv(["-c", statement], cwd)


def _run_argv(argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """在指定 cwd 下运行一个 python 子进程（PYTHONPATH 指向仓库根）。"""
    env = {
        "PYTHONPATH": str(ROOT),
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [PYTHON, *argv],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )


@pytest.mark.parametrize("name", P0_PUBLIC)
def test_public_api_is_same_object_everywhere(name: str) -> None:
    """facade 两种导入方式与权威子模块必须导出同一对象。"""
    assert hasattr(paired_stats, name)
    assert getattr(paired_stats, name) is getattr(facade_pkg, name)
    assert getattr(paired_stats, name) is getattr(OWNER[name], name)


@pytest.mark.parametrize("name", FACADE_PRIVATE)
def test_private_compat_names_are_same_object(name: str) -> None:
    """旧测试直接引用的私有名必须有显式 facade 出口，且不复制实现。"""
    assert hasattr(paired_stats, name)
    assert getattr(paired_stats, name) is getattr(OWNER[name], name)


def test_tcp_label_keeps_legacy_int_annotation() -> None:
    """`_tcp_label` 旧签名须保留 int 入参（--max-tcp 的 JSON 值可能为 int）。"""
    assert "float | int | str" in str(inspect.signature(paired_stats._tcp_label))
    assert paired_stats._tcp_label(7) == "TCP7.0"
    assert paired_stats._tcp_label("1.4") == "TCP1.4"


def test_all_lists_cover_compat_surface() -> None:
    """facade 与各子模块 __all__ 覆盖全部兼容名，且子模块 __all__ 可解析。"""
    facade_all = set(paired_stats.__all__)
    assert set(P0_PUBLIC) <= facade_all
    assert set(FACADE_PRIVATE) <= facade_all
    for module in (patypes, identity, sources, dataset, metrics, comparisons, report, cli):
        for name in module.__all__:
            assert hasattr(module, name), f"{module.__name__}.__all__ 含缺失名字 {name}"


def test_facade_keeps_no_business_logic() -> None:
    """旧文件只允许留 facade：无函数/类定义，且按职责委托到 8 个子模块。"""
    source = (EXTRACT_DIR / "paired_stats.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = [
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert defined == []
    assert len(source.splitlines()) < 400
    for module in ("types", "identity", "sources", "dataset", "metrics",
                   "comparisons", "report", "cli"):
        assert f"from scripts.extract.paired_analysis.{module} import" in source


def test_project_resolves_to_repo_root_from_any_cwd(tmp_path: Path) -> None:
    """PROJECT 不得依赖 cwd：任意目录导入都指向仓库根（迁移后 parents[3]）。"""
    assert cli.PROJECT == ROOT
    assert paired_stats.PROJECT == ROOT
    result = _run_module(
        "from scripts.extract import paired_stats as p\n"
        "from scripts.extract.paired_analysis import cli\n"
        "print(p.PROJECT)\n"
        "print(cli.PROJECT)\n"
        "print(p.DEFAULT_DATA_ROOT)\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.split()
    assert lines == [str(ROOT), str(ROOT), str(ROOT / "experiment_data")]


def test_old_data_root_name_kept() -> None:
    """兄弟脚本引用的 DATA 继续存在，并与默认数据根同值。"""
    assert paired_stats.DATA == ROOT / "experiment_data"
    assert paired_stats.DATA == paired_stats.DEFAULT_DATA_ROOT
    assert patypes.DATA is paired_stats.DATA


def test_facade_keeps_stdlib_module_attributes() -> None:
    """历史调用方会取 facade 上的 stdlib 模块属性（如 paired_stats.json）。"""
    assert paired_stats.json is json
    assert paired_stats.Path is Path
    assert paired_stats.defaultdict is defaultdict


def test_build_parser_contract() -> None:
    """CLI 参数集合与默认值保持不变。"""
    parser = cli.build_parser()
    actions = {action.dest: action for action in parser._actions}
    assert set(actions) >= {"include_passive", "data_root", "catalog", "mode", "output", "dry_run"}
    assert actions["mode"].choices == ("torque", "position", "both")
    assert actions["mode"].default is None
    assert actions["data_root"].default is None
    assert actions["catalog"].default is None
    assert actions["output"].default is None
    assert actions["dry_run"].default is False


def test_main_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """main(["--help"]) 退出 0，并且仍打印旧模块 docstring 的描述。"""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "active-hit rate" in out
    assert "--include-passive" in out


def test_main_rejects_unknown_mode() -> None:
    """非法 --mode 由 argparse 退出 2（与旧实现一致）。"""
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--mode", "bogus"])
    assert exc_info.value.code == 2


def test_main_returns_2_for_bad_catalog(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """显式坏 catalog 抛 StatsError → main 返回 2，且不写任何输出文件。"""
    data_root = tmp_path / "data"
    data_root.mkdir()
    out_dir = tmp_path / "out"
    code = cli.main([
        "--data-root", str(data_root),
        "--catalog", str(tmp_path / "missing_catalog.json"),
        "--output", str(out_dir),
    ])
    assert code == 2
    err = capsys.readouterr().err
    assert "[paired_stats] 错误：" in err
    assert not out_dir.exists()


def test_cli_module_has_main_guard() -> None:
    """cli.py 必须保留 ``if __name__ == "__main__"`` 守卫（C2 审查阻断 1 回归）。"""
    tree = ast.parse((EXTRACT_DIR / "paired_analysis" / "cli.py").read_text(encoding="utf-8"))
    guards = [
        node for node in tree.body
        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
        and "__main__" in ast.unparse(node.test)
    ]
    assert len(guards) == 1
    assert "SystemExit(main())" in ast.unparse(guards[0].body[0])


def test_cli_runs_as_module_with_help(tmp_path: Path) -> None:
    """``python -m scripts.extract.paired_analysis.cli --help`` 退出 0 且输出非空。"""
    result = _run_argv(
        ["-m", "scripts.extract.paired_analysis.cli", "--help"], tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "active-hit rate" in result.stdout
    assert result.stderr == ""


def test_minimal_dataclass_contract() -> None:
    """StatsContext/HitPolicy/RecordIdentity/PhysicalIndex 的字段顺序与默认值不变。"""
    policy = patypes.HitPolicy()
    assert policy.include_passive is False
    assert patypes.HitPolicy(include_passive=True).include_passive is True

    ctx = patypes.StatsContext(data_root=Path("d"), run_root=Path("r"))
    assert ctx.catalog is None
    assert ctx.catalog_path is None
    assert ctx.mode is None
    assert dict(ctx.source_sets) == {}
    assert ctx.synthetic is False
    assert ctx.manifest_path is None
    assert ctx.with_mode("torque") is not ctx
    assert ctx.with_mode("torque").mode == "torque"

    identity_fields = tuple(field.name for field in fields(patypes.RecordIdentity))
    assert identity_fields == (
        "experiment", "case", "config_id", "seed", "mode",
        "robot_id", "racket_id", "source", "tier", "design",
    )
    rid = patypes.RecordIdentity(
        experiment="e", case="c", config_id="cfg", seed=1,
        mode="torque", robot_id="r", racket_id="k", source="s",
    )
    assert rid.tier == "" and rid.design == ""
    assert rid.join_key() == ("e", "c", "cfg", 1, "torque", "r", "k")

    first, second = patypes.PhysicalIndex(), patypes.PhysicalIndex()
    assert first.by_identity == {} and first.by_artifact == {}
    assert first.path is None
    assert first.by_identity is not second.by_identity


def test_submodules_import_standalone(tmp_path: Path) -> None:
    """每个子模块都能在任意 cwd 冷启动导入（证明没有隐式初始化顺序依赖）。"""
    for module in ("types", "identity", "sources", "dataset", "metrics",
                   "comparisons", "report", "cli"):
        result = _run_module(
            f"import scripts.extract.paired_analysis.{module}", tmp_path,
        )
        assert result.returncode == 0, f"{module}: {result.stderr}"


def test_no_relative_import_cycles() -> None:
    """AST 依赖图必须无环（types ← … ← cli 的单向依赖）。"""
    package_dir = EXTRACT_DIR / "paired_analysis"
    graph: dict[str, set[str]] = {}
    for path in sorted(package_dir.glob("*.py")):
        module = path.stem
        if module.startswith("_"):
            continue
        edges: set[str] = set()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # 只看模块级相对导入；函数体内的延迟导入（sources → dataset）不构成模块级环。
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                edges.add(node.module.split(".")[0])
        graph[module] = edges

    visiting: set[str] = set()
    done: set[str] = set()

    def visit(name: str, stack: tuple[str, ...]) -> None:
        if name in done:
            return
        assert name not in visiting, f"循环 import: {' -> '.join((*stack, name))}"
        visiting.add(name)
        for dep in sorted(graph.get(name, ())):
            if dep in graph:
                visit(dep, (*stack, name))
        visiting.remove(name)
        done.add(name)

    for module in sorted(graph):
        visit(module, ())
    assert graph["types"] == set(), "types 必须是叶模块"
    assert graph["cli"], "cli 必须组合其它子模块"
    assert all(
        "cli" not in deps for name, deps in graph.items() if name != "cli"
    ), "cli 只能被 facade 使用，不能被其它子模块反向依赖"
