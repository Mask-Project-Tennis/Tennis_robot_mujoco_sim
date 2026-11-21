"""V12 配置层共享类型 — 解析结果与内部配置容器。

本模块是 :class:`ResolvedV12Config` 的**唯一类定义**所在；旧路径
``src.sim.v12_config`` 只做同一对象重导出（``is`` 相同）。``_ConfigDict``、
``_config_equal``、``_freeze_value`` 随之迁入，供配置解析层与 episode 层共用。

B1 迁移完成后，YAML 合并、默认值解析、机器人选择与注册表全部位于
``src/sim/configuration`` 的其他模块；本模块只保留类型与冻结语义。

设计约束：

* 不 import ``src.sim.v12_config`` / ``src.sim.v12_episode``（避免循环依赖）；
* 不做文件 IO、不读 YAML、不导入 MuJoCo；
* 冻结/深拷贝/相等比较语义与迁移前逐字一致。
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["ResolvedV12Config"]


# ---------------------------------------------------------------------------
# 配置字典：带 numpy 数组的相等比较（避免“数组真值歧义”）
# ---------------------------------------------------------------------------


class _ConfigDict(dict[str, Any]):
    """配置字典子类，比较时把 numpy 数组按逐元素相等处理。"""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return _config_equal(self, other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    __hash__ = None  # type: ignore[assignment]


def _config_equal(left: Any, right: Any) -> bool:
    """递归比较配置值；numpy 数组用 ``array_equal``，其余用普通相等。"""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_config_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_config_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    return bool(left == right)


@dataclass(frozen=True, eq=False)
class ResolvedV12Config:
    """V12 唯一任务配置解析结果（深拷贝隔离、数组只读、可追溯来源）。

    该结构是 T5 接线的唯一输入：``mpc_config`` 可近似直接作为
    ``MPCConfig(**...)`` 的字段来源，``solver_config`` 供 solver 构建，
    ``hitting_config`` / ``scene_config`` / ``serve_config`` 供环境与
    来球生成，``sources`` 记录每个字段的实际来源标签。

    不可变语义：numpy 数组复制后设为只读；dict/list 为深拷贝副本，
    **容器本身仍可写**（修改只影响该副本，不会污染其他解析结果）。
    相等性按字段递归比较（numpy 数组逐元素），避免默认 dataclass
    比较在数组上抛出“truth value ambiguous”。

    Attributes:
        robot_id: ``"rm65"`` 或 ``"unitree_r1d"``。
        task_mode: ``"legacy_cli"`` / ``"r1d_overlay"`` /
            ``"r1d_fixed_ball_compat"``。
        robot_profile_path: 显式机器人描述路径（旧 RM65 缺省为 None）。
        runtime_path: R1D runtime 路径（显式或注册表；RM65 为 None）。
        racket_path: 球拍配置路径（显式或注册表；可为 None）。
        task_path: R1D 任务配置路径（overlay 或固定球兼容文件）。
        dt: 仿真步长（来自 default.yaml）。
        seed: 随机种子。
        actuator_mode: ``"torque"`` / ``"position"``。
        feedforward: position 模式 PD 前馈开关；torque 模式为 None。
        kp/kd: position 模式 PD 增益（按主关节数解析）；torque 为 None。
        ablation_mode: 消融模式。
        solver_backend: 求解器后端。
        solver_config: default.yaml ilqt 节深拷贝 + backend。
        mpc_config: MPC 参数（键名接近 MPCConfig 字段）。
        cost_config: 代价参数（Q_p/Q_v/R/平滑/阶段权重/Tube）。
        perception_config: 观测门控与噪声配置。
        hitting_config: 击球参数（合并后的 hitting 节）。
        scene_config: 球/重力/发球反弹/肩点等场景参数。
        serve_config: 来球生成参数（含服务箱派生几何）。
        target_center: 目标区域中心。
        target_offset: 目标区域偏移。
        workspace_radius: 工作区半径。
        output_dir: 输出目录。
        viewer: 是否打开查看器。
        no_plot: 是否禁用绘图。
        sources: 点路径 → 来源标签。
        explicit_options: 规范化后的显式 CLI 选项集合。
        runtime_overrides: 本次运行实际生效的运行时覆盖（limits_config /
            max_tcp / terminal_exempt_steps / dq_max_fraction / qdot_max /
            dq_max 的规范化记录；解析阶段为空，由 ``run_v12_episode``
            用 ``dataclasses.replace`` 回填）。
    """

    robot_id: str
    task_mode: str
    robot_profile_path: Path | None
    runtime_path: Path | None
    racket_path: Path | None
    task_path: Path | None
    dt: float
    seed: int | None
    actuator_mode: str
    feedforward: bool | None
    kp: np.ndarray | None
    kd: np.ndarray | None
    ablation_mode: str
    solver_backend: str
    solver_config: _ConfigDict
    mpc_config: _ConfigDict
    cost_config: dict[str, Any]
    perception_config: dict[str, Any]
    hitting_config: dict[str, Any]
    scene_config: dict[str, Any]
    serve_config: dict[str, Any]
    target_center: np.ndarray
    target_offset: float
    workspace_radius: float
    output_dir: Path
    viewer: bool
    no_plot: bool
    sources: dict[str, str]
    explicit_options: frozenset[str]
    runtime_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """冻结数组并深拷贝所有可变容器，防止跨解析共享状态。"""
        dict_fields = (
            "solver_config",
            "mpc_config",
            "cost_config",
            "perception_config",
            "hitting_config",
            "scene_config",
            "serve_config",
        )
        for name in dict_fields:
            object.__setattr__(
                self, name, _freeze_value(getattr(self, name))
            )
        object.__setattr__(self, "sources", dict(self.sources))
        object.__setattr__(self, "explicit_options", frozenset(self.explicit_options))
        # R16：运行时覆盖同样深拷贝并递归 freeze（嵌套 dict/list 全部只留副本）
        object.__setattr__(
            self,
            "runtime_overrides",
            _freeze_value(dict(self.runtime_overrides)),
        )
        for name in ("target_center", "kp", "kd"):
            value = getattr(self, name)
            if value is not None:
                array = np.array(value, dtype=np.float64, copy=True)
                array.setflags(write=False)
                object.__setattr__(self, name, array)

    def __eq__(self, other: object) -> bool:
        """按字段递归比较；numpy 数组用 ``array_equal`` 避免歧义异常。"""
        if not isinstance(other, ResolvedV12Config):
            return NotImplemented
        return all(
            _config_equal(getattr(self, field.name), getattr(other, field.name))
            for field in fields(self)
        )

    __hash__ = None  # type: ignore[assignment]


def _freeze_value(value: Any) -> Any:
    """递归拷贝配置值；numpy 数组复制后设为只读，字典保持可变类型。"""
    if isinstance(value, np.ndarray):
        array = np.array(value, dtype=np.float64, copy=True)
        array.setflags(write=False)
        return array
    if isinstance(value, _ConfigDict):
        return _ConfigDict(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, Mapping):
        return {key: _freeze_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_freeze_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    return copy.deepcopy(value)
