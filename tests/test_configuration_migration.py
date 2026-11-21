"""B1 配置解析迁移回归测试。

锁定 ``src/sim/v12_config.py`` → ``src/sim/configuration/`` 迁移的对外契约：

- 旧 facade 与新模块导出同一对象（``is`` 相同），``__all__`` 保持不变；
- 仓库根常量指向工作树根（``parents[3]``），从任意 cwd 解析仍读仓库 ``configs/``；
- ``False`` / ``0`` / ``None`` 等显式值不被 truthiness 吞掉；
- R1D 注册表路径始终为仓库 ``configs/`` 下的绝对路径；
- 旧模块的属性 patch 仍可拦截 ``v12_config.resolve_v12_config`` 调用；
- 旧文件不再包含第二份解析实现，内部模块保持单向依赖。
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

import pytest

import src.sim.configuration.types as configuration_types
import src.sim.v12_config as legacy
from src.sim import configuration
from src.sim.configuration import resolution, validation

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
ROBOTS_DIR = CONFIGS_DIR / "robots"

# 迁移前后必须逐字保持的旧 facade 导出清单。
_LEGACY_EXPORTS = [
    "R1D_V12_REGISTRY",
    "R1DConfigRegistry",
    "ResolvedV12Config",
    "normalize_explicit_options",
    "resolve_v12_config",
]

# 每个公有名迁移后的真实定义模块。
_EXPECTED_MODULE = {
    "ResolvedV12Config": "src.sim.configuration.types",
    "R1DConfigRegistry": "src.sim.configuration.selection",
    "R1D_V12_REGISTRY": "src.sim.configuration.selection",
    "normalize_explicit_options": "src.sim.configuration.overrides",
    "resolve_v12_config": "src.sim.configuration.resolution",
}


def _args(**overrides: Any) -> argparse.Namespace:
    """构造与 V12 argparse 命名空间等价的最小对象。"""
    return argparse.Namespace(**overrides)


def _resolve_rm65(
    explicit: frozenset[str] | None = None,
    **overrides: Any,
) -> configuration.ResolvedV12Config:
    """用 RM65 旧路径解析一份配置快照。"""
    return configuration.resolve_v12_config(_args(**overrides), explicit)


def revision_paths() -> tuple[Path, Path]:
    """返回 registry 专用任务的右臂 profile 与 task 配置文件路径。"""
    from src.sim.configuration.selection import (
        _R1D_RIGHT_PROFILE_PATH,
        _R1D_TASK_CONFIG_PATH,
    )

    return _R1D_RIGHT_PROFILE_PATH.resolve(), _R1D_TASK_CONFIG_PATH.resolve()


@pytest.mark.parametrize("name", _LEGACY_EXPORTS)
def test_legacy_facade_reexports_identical_objects(name: str) -> None:
    """旧 facade 与新模块必须导出同一对象，且定义位置指向新模块。"""
    old = getattr(legacy, name)
    new = getattr(configuration, name)
    assert old is new
    if name == "R1D_V12_REGISTRY":
        # 注册表是 MappingProxyType 实例，没有 __module__；改为锁定元素类型。
        assert type(old).__name__ == "mappingproxy"
        assert all(
            isinstance(entry, configuration.R1DConfigRegistry)
            for entry in old.values()
        )
        return
    assert old.__module__ == _EXPECTED_MODULE[name]


def test_legacy_all_and_private_override_table_unchanged() -> None:
    """旧 ``__all__`` 顺序不变，私有 ``_OVERRIDE_VALIDATORS`` 仍可导入。"""
    assert legacy.__all__ == _LEGACY_EXPORTS
    assert legacy._OVERRIDE_VALIDATORS is validation._OVERRIDE_VALIDATORS
    # 旧调用点（scripts/exp/run_r1d_paper_smoke.py）依赖该私有名可导入。
    assert "mpc.total_horizon" in legacy._OVERRIDE_VALIDATORS
    assert legacy._ConfigDict is configuration_types._ConfigDict


def test_repo_root_constant_points_to_worktree_root() -> None:
    """仓库根常量来自 ``configuration/resolution.py`` 的 ``parents[3]``。"""
    resolution_file = Path(inspect.getsourcefile(resolution.resolve_v12_config)).resolve()
    assert resolution_file == (REPO_ROOT / "src" / "sim" / "configuration" / "resolution.py")
    assert resolution._REPO_ROOT == REPO_ROOT
    assert resolution._CONFIGS_DIR == CONFIGS_DIR
    assert resolution._CONFIGS_DIR.is_dir()


def test_exported_entry_defined_in_resolution_module() -> None:
    """导出入口必须唯一，旧文件不得保留第二份解析实现。"""
    assert configuration.resolve_v12_config is legacy.resolve_v12_config
    assert inspect.getsourcefile(legacy.resolve_v12_config) == str(
        REPO_ROOT / "src" / "sim" / "configuration" / "resolution.py"
    )
    facade_source = Path(legacy.__file__).read_text(encoding="utf-8")
    assert "def resolve_v12_config" not in facade_source
    assert "def _finalize" not in facade_source
    assert "def normalize_explicit_options" not in facade_source


def test_resolve_from_foreign_cwd_keeps_repo_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """离开仓库 cwd 后仍从仓库 ``configs/`` 读取，并输出到仓库 ``results/``。"""
    monkeypatch.chdir(tmp_path)
    assert Path.cwd() == tmp_path

    resolved = _resolve_rm65()

    assert resolved.sources["dt"] == "default.yaml"
    assert resolved.sources["cost.Q_p"] == "v5_active_hit.yaml"
    assert resolved.sources["hitting.racket_speed"] == "v5_active_hit.yaml"
    assert (CONFIGS_DIR / "default.yaml").is_file()
    assert resolved.output_dir == (REPO_ROOT / "results").resolve()
    assert resolved.output_dir != (tmp_path / "results").resolve()


def test_r1d_registry_paths_are_absolute_under_repo_configs() -> None:
    """注册表键与三个路径必须是仓库 ``configs/`` 下的绝对路径。"""
    right_profile = (ROBOTS_DIR / "r1d_right.yaml").resolve()
    assert right_profile in configuration.R1D_V12_REGISTRY
    # 注册表专用路径常量也必须落在仓库 configs/ 下。
    assert revision_paths() == (right_profile, CONFIGS_DIR / "r1d_v12.yaml")
    entry = configuration.R1D_V12_REGISTRY[right_profile]
    assert entry.side == "right"
    for path in (right_profile, entry.runtime_path, entry.task_path, entry.racket_path):
        assert path.is_absolute()
        assert path.is_file()
        assert path.is_relative_to(CONFIGS_DIR)


def test_r1d_selection_from_foreign_cwd_uses_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """离开仓库 cwd 后 R1D 仍按注册表的绝对路径完成选择。"""
    monkeypatch.chdir(tmp_path)
    resolved = configuration.resolve_v12_config(
        _args(robot_config=ROBOTS_DIR / "r1d_right.yaml"),
        frozenset({"--robot-config"}),
    )
    assert resolved.robot_id == "unitree_r1d"
    assert resolved.task_mode == "r1d_overlay"
    assert resolved.task_path == (CONFIGS_DIR / "r1d_v12.yaml").resolve()
    assert resolved.runtime_path == (
        CONFIGS_DIR / "robots" / "r1d_v12_runtime_right.yaml"
    ).resolve()
    assert resolved.sources["robot_profile_path"] == "cli:--robot-config"
    assert resolved.sources["target.center"] == "r1d_v12.yaml"


def test_explicit_false_flag_is_not_swallowed() -> None:
    """显式 ``--position-mode=False`` 不应被 truthiness 误判为启用。"""
    resolved = _resolve_rm65(
        frozenset({"--position-mode"}),
        position_mode=False,
    )
    assert resolved.actuator_mode == "torque"
    assert resolved.feedforward is None
    assert resolved.kp is None
    assert resolved.kd is None
    assert resolved.explicit_options == frozenset({"position_mode"})


def test_seed_zero_and_zero_weight_are_preserved() -> None:
    """显式 ``--seed 0`` 与 ``--Q-tcp-soft 0`` 必须保留为 0，而非 None。"""
    resolved = _resolve_rm65(
        frozenset({"--seed", "--Q-tcp-soft"}),
        seed=0,
        Q_tcp_soft=0.0,
    )
    assert resolved.seed == 0
    assert resolved.mpc_config["Q_tcp_soft"] == 0.0
    assert resolved.sources["seed"] == "cli:--seed"
    assert resolved.sources["mpc.Q_tcp_soft"] == "cli:--Q-tcp-soft"
    assert resolved.explicit_options == frozenset({"seed", "q_tcp_soft"})


def test_none_seed_is_preserved_but_tracked_as_explicit() -> None:
    """``None`` 与“未提供”在 explicit_options 中仍可区分。"""
    resolved = _resolve_rm65(frozenset({"--seed"}), seed=None)
    assert resolved.seed is None
    assert "--seed" in {f"--{name.replace('_', '-')}" for name in resolved.explicit_options}


def test_false_boolean_task_override_is_not_swallowed() -> None:
    """位置模式下任务覆盖 ``feedforward=False`` 必须写入并保留来源。"""
    resolved = _resolve_rm65(
        frozenset({"task_overrides", "--position-mode"}),
        position_mode=True,
        task_overrides={"actuator": {"feedforward": False}},
    )
    assert resolved.feedforward is False
    assert resolved.sources["actuator.feedforward"] == "task_overrides"


def test_false_use_r_decay_override_is_not_swallowed() -> None:
    """力矩模式下 ``mpc.use_r_decay=False`` 覆盖必须保留。"""
    resolved = _resolve_rm65(
        frozenset({"task_overrides"}),
        task_overrides={"mpc": {"use_r_decay": False}},
    )
    assert resolved.mpc_config["use_r_decay"] is False
    assert resolved.sources["mpc.use_r_decay"] == "task_overrides"


def test_normalize_explicit_options_handles_flags_and_actions() -> None:
    """字符串旗标与 argparse Action 都归一化为小写 dest。"""
    action = argparse.Action(option_strings=["--ball-speed"], dest="ball_speed")
    normalized = configuration.normalize_explicit_options(
        ["--serve-box", "Q_tcp_soft", action]
    )
    assert normalized == frozenset({"serve_box", "q_tcp_soft", "ball_speed"})


def test_legacy_module_patch_intercepts_attribute_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧模块属性 patch 仍可拦截 ``v12_config.resolve_v12_config`` 调用。

    注意：这只覆盖“通过模块属性调用”的消费点；迁移前就
    ``from src.sim.v12_config import resolve_v12_config`` 的调用方在 import
    时已绑定原函数，需在主代理的 patch 点清单里单独处理。
    """
    sentinel = object()

    def fake_resolve(
        args: argparse.Namespace | None = None,
        explicit_options: Any = None,
    ) -> object:
        return sentinel

    monkeypatch.setattr(legacy, "resolve_v12_config", fake_resolve)
    assert legacy.resolve_v12_config() is sentinel
    # 新入口不受旧 facade 属性 patch 影响。
    assert configuration.resolve_v12_config is not fake_resolve


def test_legacy_module_unittest_patch_intercepts_attribute_calls() -> None:
    """``unittest.mock.patch`` 仍能拦截旧模块属性入口。"""
    from unittest import mock

    sentinel = object()
    with mock.patch(
        "src.sim.v12_config.resolve_v12_config", return_value=sentinel
    ) as patched:
        assert legacy.resolve_v12_config() is sentinel
        patched.assert_called_once()
    # patch 退出后恢复原对象（仍是新模块入口）。
    assert legacy.resolve_v12_config is configuration.resolve_v12_config


# 每个模块允许依赖的同包模块之外的上层模块（其余同包 import 视为违规）。
_FORBIDDEN_DEPENDENCIES: dict[str, set[str]] = {
    "types": {"validation", "defaults", "selection", "overrides", "resolution"},
    "validation": {"defaults", "selection", "overrides", "resolution"},
    "defaults": {"overrides", "resolution"},
    "selection": {"overrides", "resolution"},
    "overrides": {"resolution"},
    "resolution": set(),
}


def _runtime_imports(source: str) -> set[str]:
    """收集模块运行期（排除 ``TYPE_CHECKING`` 分支）的同包依赖名。"""
    import ast

    tree = ast.parse(source)
    imported: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, ast.ImportFrom) and node.module:
            prefix = "src.sim.configuration."
            if node.module.startswith(prefix):
                imported.add(node.module[len(prefix) :])
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return imported


@pytest.mark.parametrize("module_name", sorted(_FORBIDDEN_DEPENDENCIES))
def test_internal_modules_have_single_direction_dependencies(module_name: str) -> None:
    """同包模块不得反向 import 上层模块（``TYPE_CHECKING`` 不算运行期依赖）。"""
    module = __import__(
        f"src.sim.configuration.{module_name}", fromlist=[module_name]
    )
    source = Path(module.__file__).read_text(encoding="utf-8")
    forbidden = _runtime_imports(source) & _FORBIDDEN_DEPENDENCIES[module_name]
    assert not forbidden, f"{module_name}.py 反向依赖: {sorted(forbidden)}"


def test_types_module_has_no_upward_dependency() -> None:
    """``types.py`` 不得 import resolution / defaults 等上层模块。"""
    source = Path(configuration_types.__file__).read_text(encoding="utf-8")
    assert "configuration.resolution" not in source
    assert "configuration.defaults" not in source
    assert "from src.sim.v12_config" not in source
    assert "import src.sim.v12_config" not in source
    assert not hasattr(configuration_types, "resolve_v12_config")
