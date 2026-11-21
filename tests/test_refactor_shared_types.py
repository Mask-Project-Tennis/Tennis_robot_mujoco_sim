"""P1 共享类型前置迁移回归测试。

覆盖 ``contracts.md`` §2 的四类共享类型迁移：

- 旧导入路径与新包导入返回同一对象（``is`` 相同），``__module__`` 指向新模块；
- 字段名、字段顺序、默认值与 ``frozen``/``eq``/``hash`` 语义保持不变；
- ``ResolvedV12Config.__post_init__`` 的数组冻结与容器深拷贝语义；
- ``V12EpisodeResult.model_dependency_manifest`` 是独立的 list 默认工厂；
- 新包可在干净子进程中独立导入，且不触发旧实现模块；
- 旧模块的 ``__all__`` 与显式重导出保持不变。
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.robot.model_building.types import RacketAssembly, RobotModelBundle
from src.sim.configuration.types import ResolvedV12Config
from src.sim.episode.types import V12EpisodeResult

REPO_ROOT = Path(__file__).resolve().parents[1]

# 契约 §2 规定的字段顺序（不可随意调整）。
_RACKET_FIELDS = (
    "racket_id",
    "source_path",
    "total_mass",
    "com",
    "inertia_diagonal",
    "face_center",
    "face_semi_axes",
    "mount_position",
    "mount_wxyz",
    "connector_mass",
)

_BUNDLE_FIELDS = (
    "profile",
    "parameters",
    "model",
    "binding",
    "racket",
    "dependency_manifest",
)

_CONFIG_FIELDS = (
    "robot_id",
    "task_mode",
    "robot_profile_path",
    "runtime_path",
    "racket_path",
    "task_path",
    "dt",
    "seed",
    "actuator_mode",
    "feedforward",
    "kp",
    "kd",
    "ablation_mode",
    "solver_backend",
    "solver_config",
    "mpc_config",
    "cost_config",
    "perception_config",
    "hitting_config",
    "scene_config",
    "serve_config",
    "target_center",
    "target_offset",
    "workspace_radius",
    "output_dir",
    "viewer",
    "no_plot",
    "sources",
    "explicit_options",
    "runtime_overrides",
)

_EPISODE_FIELDS = (
    "config",
    "robot_id",
    "mode",
    "joint_names",
    "dt",
    "racket_id",
    "X",
    "U",
    "ball_pos",
    "ball_vel",
    "time",
    "racket_pos",
    "hit_type",
    "hit_step",
    "active_contact",
    "passive_contact",
    "v_ee_at_hit",
    "p_ee_at_hit",
    "ball_pos_at_hit",
    "ball_out_speed",
    "ball_out_vy",
    "metrics",
    "model_dependency_manifest",
)


def _resolved_config(**overrides: Any) -> ResolvedV12Config:
    """构造仅用于字段/冻结语义测试的最小配置，不读取任何 YAML。"""
    from src.sim.configuration.types import _ConfigDict

    values: dict[str, Any] = {
        "robot_id": "rm65",
        "task_mode": "legacy_cli",
        "robot_profile_path": None,
        "runtime_path": None,
        "racket_path": None,
        "task_path": None,
        "dt": 0.005,
        "seed": 0,
        "actuator_mode": "torque",
        "feedforward": None,
        "kp": None,
        "kd": None,
        "ablation_mode": "none",
        "solver_backend": "cpp",
        "solver_config": _ConfigDict({"backend": "cpp", "tol": 1e-8}),
        "mpc_config": _ConfigDict({"horizon": 5}),
        "cost_config": {"Q_p": np.ones(3)},
        "perception_config": {},
        "hitting_config": {},
        "scene_config": {},
        "serve_config": {},
        "target_center": np.array([1.0, 2.0, 3.0]),
        "target_offset": 0.10,
        "workspace_radius": 0.80,
        "output_dir": Path("results/p1-test"),
        "viewer": False,
        "no_plot": True,
        "sources": {"dt": "default.yaml"},
        "explicit_options": frozenset({"--seed"}),
    }
    values.update(overrides)
    return ResolvedV12Config(**values)


def _episode_result() -> V12EpisodeResult:
    """构造仅用于默认工厂测试的最小 episode 结果。"""
    return V12EpisodeResult(
        config=_resolved_config(),
        robot_id="rm65",
        mode="torque",
        joint_names=("r_joint1", "r_joint2"),
        dt=0.005,
        racket_id=None,
        X=np.zeros((3, 4)),
        U=np.zeros((2, 2)),
        ball_pos=np.zeros((3, 3)),
        ball_vel=np.zeros((3, 3)),
        time=np.zeros(3),
        racket_pos=np.zeros((3, 3)),
        hit_type="miss",
        hit_step=-1,
        active_contact=False,
        passive_contact=False,
        v_ee_at_hit=None,
        p_ee_at_hit=None,
        ball_pos_at_hit=None,
        ball_out_speed=None,
        ball_out_vy=None,
    )


def test_racket_assembly_reexport_identity() -> None:
    """RacketAssembly 旧路径、新模块与新包导出必须是同一对象。"""
    from src.robot import model_building as package
    from src.robot import model_factory as legacy

    assert legacy.RacketAssembly is RacketAssembly
    assert package.RacketAssembly is RacketAssembly
    assert package.__all__ == ["RacketAssembly", "RobotModelBundle"]
    assert RacketAssembly.__module__ == "src.robot.model_building.types"


def test_robot_model_bundle_reexport_identity() -> None:
    """RobotModelBundle 旧路径、新模块与新包导出必须是同一对象。"""
    from src.robot import model_factory as legacy

    assert legacy.RobotModelBundle is RobotModelBundle
    assert RobotModelBundle.__module__ == "src.robot.model_building.types"


def test_racket_assembly_fields_and_dataclass_semantics() -> None:
    """RacketAssembly 字段顺序/默认值/frozen/eq/hash 语义保持不变。"""
    fields = tuple(field.name for field in dataclasses.fields(RacketAssembly))
    assert fields == _RACKET_FIELDS
    for field in dataclasses.fields(RacketAssembly):
        assert field.default is dataclasses.MISSING
        assert field.default_factory is dataclasses.MISSING

    params = RacketAssembly.__dataclass_params__
    assert params.frozen is True
    assert params.eq is True
    assert RacketAssembly.__hash__ is not None

    assembly = RacketAssembly(
        racket_id="rm65_legacy",
        source_path=Path("configs/rackets/rm65_legacy.yaml"),
        total_mass=0.35,
        com=np.zeros(3),
        inertia_diagonal=np.ones(3),
        face_center=np.zeros(3),
        face_semi_axes=np.ones(3),
        mount_position=np.zeros(3),
        mount_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        connector_mass=0.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        assembly.total_mass = 1.0  # type: ignore[misc]

    # ndarray 字段不可哈希：frozen dataclass 的默认 hash 会向上抛 TypeError。
    with pytest.raises(TypeError):
        hash(assembly)


def test_robot_model_bundle_fields_and_defaults() -> None:
    """RobotModelBundle 字段顺序与可选字段默认值保持不变。"""
    fields = {field.name: field for field in dataclasses.fields(RobotModelBundle)}
    assert tuple(fields) == _BUNDLE_FIELDS
    assert fields["racket"].default is None
    assert fields["dependency_manifest"].default == ()
    for name in ("profile", "parameters", "model", "binding"):
        assert fields[name].default is dataclasses.MISSING
        assert fields[name].default_factory is dataclasses.MISSING

    params = RobotModelBundle.__dataclass_params__
    assert params.frozen is True
    assert params.eq is True
    assert RobotModelBundle.__hash__ is not None


def test_resolved_v12_config_reexport_identity() -> None:
    """ResolvedV12Config 旧路径、新模块与新包导出必须是同一对象。"""
    from src.sim import configuration as package
    from src.sim import v12_config as legacy

    assert legacy.ResolvedV12Config is ResolvedV12Config
    assert package.ResolvedV12Config is ResolvedV12Config
    # B1 起新包对外导出五个公有名；旧 facade 的 __all__ 保持迁移前五项。
    expected_all = [
        "R1D_V12_REGISTRY",
        "R1DConfigRegistry",
        "ResolvedV12Config",
        "normalize_explicit_options",
        "resolve_v12_config",
    ]
    assert legacy.__all__ == expected_all
    assert package.__all__ == expected_all
    assert ResolvedV12Config.__module__ == "src.sim.configuration.types"


def test_resolved_v12_config_fields_and_defaults() -> None:
    """ResolvedV12Config 字段顺序与唯一默认工厂保持不变。"""
    fields = {field.name: field for field in dataclasses.fields(ResolvedV12Config)}
    assert tuple(fields) == _CONFIG_FIELDS
    for name in _CONFIG_FIELDS[:-1]:
        assert fields[name].default is dataclasses.MISSING
        assert fields[name].default_factory is dataclasses.MISSING
    assert fields["runtime_overrides"].default is dataclasses.MISSING
    assert fields["runtime_overrides"].default_factory is dict

    params = ResolvedV12Config.__dataclass_params__
    assert params.frozen is True
    assert params.eq is False
    assert ResolvedV12Config.__hash__ is None


def test_resolved_v12_config_freezes_arrays_and_copies_containers() -> None:
    """__post_init__ 冻结数组、深拷贝容器，但容器本身仍可写。"""
    from src.sim.configuration.types import _ConfigDict

    kp = np.array([1.0, 2.0, 3.0])
    kd = np.array([4.0, 5.0, 6.0])
    target_center = np.array([1.0, 2.0, 3.0])
    sources = {"dt": "default.yaml"}
    explicit_options = {"--seed"}
    runtime_overrides: dict[str, Any] = {
        "qdot_max": np.array([1.0, 2.0]),
        "nested": {"dq_max": np.array([0.1, 0.2])},
    }
    cost_config = {"Q_p": np.ones(3)}

    config = _resolved_config(
        kp=kp,
        kd=kd,
        target_center=target_center,
        sources=sources,
        explicit_options=explicit_options,
        runtime_overrides=runtime_overrides,
        cost_config=cost_config,
    )

    # 输入数组未被原地冻结；配置内部是独立只读副本。
    assert kp.flags.writeable is True
    assert kd.flags.writeable is True
    assert target_center.flags.writeable is True
    assert config.kp is not None and config.kp.flags.writeable is False
    assert config.kd is not None and config.kd.flags.writeable is False
    assert config.target_center.flags.writeable is False
    assert config.cost_config["Q_p"].flags.writeable is False
    assert config.runtime_overrides["qdot_max"].flags.writeable is False
    assert config.runtime_overrides["nested"]["dq_max"].flags.writeable is False
    assert isinstance(config.solver_config, _ConfigDict)
    assert config.solver_config["tol"] == 1e-8
    assert isinstance(config.explicit_options, frozenset)
    assert config.explicit_options == frozenset({"--seed"})

    # 修改输入不会反向污染配置副本。
    kp[0] = 99.0
    sources["dt"] = "mutated.yaml"
    assert config.kp[0] == 1.0
    assert config.sources["dt"] == "default.yaml"

    # 容器本身仍可写（既有语义，只影响该副本）。
    config.cost_config["extra"] = 1
    assert config.cost_config["extra"] == 1


def test_resolved_v12_config_equality_and_hash() -> None:
    """字段递归相等比较支持 numpy 数组，且实例不可哈希。"""
    first = _resolved_config()
    second = _resolved_config()
    assert first == second
    assert first != _resolved_config(seed=7)
    assert (first == object()) is False
    with pytest.raises(TypeError):
        hash(first)


def test_config_dict_equality_and_hash() -> None:
    """_ConfigDict 按 numpy 数组逐元素比较，且不可哈希。"""
    from src.sim.configuration.types import _ConfigDict

    left = _ConfigDict({"tol": 1e-8, "weights": np.array([1.0, 2.0])})
    right = _ConfigDict({"weights": np.array([1.0, 2.0]), "tol": 1e-8})
    assert left == right
    assert left != _ConfigDict({"tol": 1e-8})
    assert left != {"tol": 2e-8, "weights": np.array([1.0, 2.0])}
    assert left.__eq__(["not-a-mapping"]) is NotImplemented
    with pytest.raises(TypeError):
        hash(left)


def test_v12_episode_result_reexport_identity() -> None:
    """V12EpisodeResult 旧路径、新模块与新包导出必须是同一对象。"""
    from src.sim import episode as package
    from src.sim import v12_episode as legacy

    assert legacy.V12EpisodeResult is V12EpisodeResult
    assert package.V12EpisodeResult is V12EpisodeResult
    assert package.__all__ == ["V12EpisodeResult"]
    assert V12EpisodeResult.__module__ == "src.sim.episode.types"


def test_v12_episode_result_fields_and_default_factories() -> None:
    """V12EpisodeResult 字段顺序与两个可变默认工厂保持不变。"""
    fields = {field.name: field for field in dataclasses.fields(V12EpisodeResult)}
    assert tuple(fields) == _EPISODE_FIELDS
    for name in _EPISODE_FIELDS[:-2]:
        assert fields[name].default is dataclasses.MISSING
    assert fields["metrics"].default is dataclasses.MISSING
    assert fields["metrics"].default_factory is dict
    assert fields["model_dependency_manifest"].default is dataclasses.MISSING
    assert fields["model_dependency_manifest"].default_factory is list

    first = _episode_result()
    second = _episode_result()
    assert first.metrics == {}
    assert isinstance(first.metrics, dict)
    assert first.model_dependency_manifest == []
    assert isinstance(first.model_dependency_manifest, list)
    assert first.metrics is not second.metrics
    assert first.model_dependency_manifest is not second.model_dependency_manifest

    with pytest.raises(dataclasses.FrozenInstanceError):
        first.hit_step = 0  # type: ignore[misc]


def test_legacy_module_all_unchanged() -> None:
    """旧模块 __all__ 与重导出保持契约规定的形状。"""
    from src.robot import model_factory as legacy_factory
    from src.sim import v12_config as legacy_config
    from src.sim import v12_episode as legacy_episode

    assert legacy_config.__all__ == [
        "R1D_V12_REGISTRY",
        "R1DConfigRegistry",
        "ResolvedV12Config",
        "normalize_explicit_options",
        "resolve_v12_config",
    ]
    assert legacy_episode.__all__ == ["V12EpisodeResult", "run_v12_episode"]

    # model_factory 历史上没有 __all__，迁移后也不应新增。
    assert not hasattr(legacy_factory, "__all__")
    assert legacy_factory.RacketAssembly is RacketAssembly
    assert legacy_factory.RobotModelBundle is RobotModelBundle
    assert legacy_config.ResolvedV12Config is ResolvedV12Config
    assert legacy_episode.V12EpisodeResult is V12EpisodeResult


def test_new_packages_import_without_impl_modules() -> None:
    """干净子进程只导入三个新包时，不加载旧实现模块且退出码为 0。"""
    code = (
        "import sys\n"
        "import src.robot.model_building\n"
        "import src.sim.configuration\n"
        "import src.sim.episode\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name in {\n"
        "        'src.robot.model_factory',\n"
        "        'src.sim.v12_config',\n"
        "        'src.sim.v12_episode',\n"
        "    }\n"
        ")\n"
        "assert not loaded, loaded\n"
        "print('ok')\n"
    )
    env = {**os.environ, "PYTHONUTF8": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        env=env,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
