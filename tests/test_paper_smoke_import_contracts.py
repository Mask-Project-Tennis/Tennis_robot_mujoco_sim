"""C1 迁移 import 契约测试。

覆盖三件事：
1. 旧入口 facade 与 ``scripts.exp.paper_smoke`` 新模块导出的是同一批对象；
2. manifest 加载/校验的既有行为（默认路径、绝对路径、错误路径）不变；
3. 旧/新两个 CLI 入口都能 ``--help`` 退出，且导入包本身不启动子进程、不写文件。

测试只使用临时/合成 manifest，不跑真实仿真。
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _minimal_manifest() -> dict[str, Any]:
    """构造可通过 ``validate_manifest`` 的最小合成 manifest。

    配置路径引用仓库内真实存在的冻结文件（校验器会读取 runtime YAML 的
    override 键集合），但 experiments 只含一个合成单元，不触发任何仿真。
    """
    return {
        "schema": "r1d_paper_smoke_v1",
        "description": "C1 import 契约测试用合成 manifest",
        "seeds": [100, 101],
        "script": "scripts/rm65_mpc_v12.py",
        "configs": {
            "robot": "configs/robots/r1d_right.yaml",
            "runtime": "configs/robots/r1d_v12_runtime_right.yaml",
            "racket": "configs/rackets/blade100_v9_reference.yaml",
            "task_overlay": "configs/r1d_v12.yaml",
            "sim_task_config": None,
        },
        "identity_defaults": {
            "robot_id": "unitree_r1d",
            "racket_id": "blade100_v9_reference",
            "metric_schema": "v12_episode_v2_physical_hit",
        },
        "base_argv": {"--serve-box": None, "--no-plot": None},
        "ablation_profiles": {"full": {}},
        "case_defaults": {
            "source_paper_item": (
                "docs/experiments/design/r1d_v12_parity_matrix.md#{experiment_id}"
            ),
            "expected_outputs": ["v12_episode.json"],
            "extract_argv": ["{python}", "scripts/extract/extract_r1d_smoke_stats.py"],
            "plot_argv": ["{python}", "scripts/plot/plot_r1d_smoke_stats.py"],
            "timeout_s": 30,
        },
        "experiments": [
            {
                "id": "exp_c1_contract",
                "kind": "batch",
                "cases": [
                    {"id": "case_torque", "mode": "torque"},
                    {"id": "case_position", "mode": "position"},
                ],
            }
        ],
    }


def _write_manifest(tmp_path: Path, payload: dict[str, Any]) -> Path:
    """把合成 manifest 写入临时 YAML 文件并返回路径。"""
    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_facade_reexports_same_objects() -> None:
    """旧入口 facade 与新包必须暴露同一批函数/异常对象。"""
    from scripts.exp import run_r1d_paper_smoke as facade
    from scripts.exp.paper_smoke import cli
    from scripts.exp.paper_smoke import manifest as manifest_mod
    from scripts.exp.paper_smoke import tasks as tasks_mod

    assert facade.ManifestError is manifest_mod.ManifestError
    assert facade.load_manifest is manifest_mod.load_manifest
    assert facade.validate_manifest is manifest_mod.validate_manifest
    assert facade.resolve_manifest is tasks_mod.resolve_manifest
    assert facade.build_specs_from_manifest is tasks_mod.build_specs_from_manifest
    assert facade.main is cli.main
    # 既有测试还会直接调用这些私有函数，必须是同一对象而非副本。
    assert facade._format_override is manifest_mod._format_override
    assert facade._artifact_metrics.__module__ == "scripts.exp.paper_smoke.artifacts"
    assert facade._collect_runs.__module__ == "scripts.exp.paper_smoke.resume"
    assert facade._v12_config_issues is not None


def test_load_manifest_default_and_absolute(tmp_path: Path) -> None:
    """默认仓库 manifest 与临时绝对路径 manifest 都可加载。"""
    from scripts.exp.paper_smoke import manifest as manifest_mod

    default = manifest_mod.load_manifest()
    assert default["schema"] == "r1d_paper_smoke_v1"
    assert "experiments" in default

    payload = _minimal_manifest()
    path = _write_manifest(tmp_path, payload)
    loaded = manifest_mod.load_manifest(path)
    assert loaded["schema"] == payload["schema"]
    assert loaded["experiments"][0]["id"] == "exp_c1_contract"


def test_load_manifest_missing_raises(tmp_path: Path) -> None:
    """不存在的 manifest 必须抛 ManifestError。"""
    from scripts.exp.paper_smoke import manifest as manifest_mod

    with pytest.raises(manifest_mod.ManifestError, match="manifest 不存在"):
        manifest_mod.load_manifest(tmp_path / "nope.yaml")


def test_validate_manifest_normalizes_minimal(tmp_path: Path) -> None:
    """最小合成 manifest 校验后得到规范化结构（seeds/cases 字段齐全）。"""
    from scripts.exp.paper_smoke import manifest as manifest_mod

    path = _write_manifest(tmp_path, _minimal_manifest())
    normalized = manifest_mod.validate_manifest(manifest_mod.load_manifest(path))

    assert normalized["schema"] == "r1d_paper_smoke_v1"
    assert normalized["seeds"] == [100, 101]
    assert normalized["ablation_profiles"] == {"full": {}}
    cases = normalized["experiments"][0]["cases"]
    assert [case["id"] for case in cases] == ["case_torque", "case_position"]
    assert all(case["seeds"] == [100, 101] for case in cases)
    assert all(case["status"] == "ready" for case in cases)
    assert cases[0]["expected_outputs"] == ["v12_episode.json"]


def test_validate_manifest_missing_field_raises() -> None:
    """缺少 script 字段必须抛 ManifestError。"""
    from scripts.exp.paper_smoke import manifest as manifest_mod

    broken = _minimal_manifest()
    del broken["script"]
    with pytest.raises(manifest_mod.ManifestError, match="script"):
        manifest_mod.validate_manifest(broken)


def test_validate_manifest_bad_seeds_raises() -> None:
    """少于两个 seed 或负 seed 必须抛 ManifestError。"""
    from scripts.exp.paper_smoke import manifest as manifest_mod

    single = copy.deepcopy(_minimal_manifest())
    single["seeds"] = [100]
    with pytest.raises(manifest_mod.ManifestError, match="2 个共同 seed"):
        manifest_mod.validate_manifest(single)

    negative = copy.deepcopy(_minimal_manifest())
    negative["seeds"] = [100, -1]
    with pytest.raises(manifest_mod.ManifestError, match="不能为负"):
        manifest_mod.validate_manifest(negative)


def _run_help(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """以仓库根为 cwd 运行 CLI 帮助命令（不读取真实 manifest）。"""
    return subprocess.run(
        argv,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )


def test_legacy_and_module_cli_help_exit_zero() -> None:
    """旧脚本入口与 ``python -m`` 新入口的 --help 都必须退出 0。"""
    legacy = _run_help([sys.executable, "scripts/exp/run_r1d_paper_smoke.py", "--help"])
    assert legacy.returncode == 0, legacy.stderr
    assert "--manifest" in legacy.stdout
    assert "--data-root" in legacy.stdout

    module = _run_help([sys.executable, "-m", "scripts.exp.paper_smoke.cli", "--help"])
    assert module.returncode == 0, module.stderr
    assert "--manifest" in module.stdout
    assert "--mode" in module.stdout


def test_import_package_has_no_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """导入轻量包时不得启动子进程、也不得写文件。"""
    def _no_subprocess(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("import scripts.exp.paper_smoke 时不应启动子进程")

    def _no_write(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("import scripts.exp.paper_smoke 时不应写文件")

    monkeypatch.setattr(subprocess, "run", _no_subprocess)
    monkeypatch.setattr(subprocess, "Popen", _no_subprocess)
    monkeypatch.setattr(Path, "write_text", _no_write)
    monkeypatch.setattr(Path, "mkdir", _no_write)

    for name in list(sys.modules):
        if name == "scripts.exp.paper_smoke" or name.startswith(
            "scripts.exp.paper_smoke."
        ):
            del sys.modules[name]

    # 本测试需要在清空 sys.modules 后强制重新导入。
    import scripts.exp.paper_smoke as package

    assert package.__all__ == [
        "ManifestError",
        "build_specs_from_manifest",
        "load_manifest",
        "resolve_manifest",
        "validate_manifest",
    ]
    assert package.load_manifest()["schema"] == "r1d_paper_smoke_v1"
