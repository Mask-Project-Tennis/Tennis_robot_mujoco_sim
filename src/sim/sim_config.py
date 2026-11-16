"""R1D 仿真任务配置（SimTaskConfig）与严格仿真预检 — 阶段 3 Task 3.1。

职责：
- ``SimTaskConfig``：冻结 dataclass，描述一次仿真击球任务的几何名称、
  工作空间、球初态、步数上限、执行器模式与三个透传配置节
  （solver/mpc/cost，键由 3.5-3.8 消费方定义）。schema version=1；
  未知字段拒绝；数组复制为只读副本（不与输入别名）。
- ``load_sim_task_config``：从 YAML 加载并对照 ``RobotProfile`` 校验
  侧别（robot_id + joint_names 逐项同序），不靠文件名猜测侧别。
- ``validate_simulation_inputs``：严格仿真预检 — 先经 ``load_robot_model``
  从 ``model.jnt_range`` 回填 q_lower/q_upper（模型是位置限位唯一权威），
  再做 ``purpose="simulation"`` 完整性校验（未知/None 限值不得进入
  仿真执行器）；位置模式还要求 kp/kd 完整。

依赖方向：本模块（sim 层）依赖 robot 层（profile/runtime/model_factory，
后者引入 MuJoCo）；纯参数模块 ``runtime_parameters`` 不反向依赖本模块。
球的 free joint / geom 名称与保护 geom 对在此仅作字符串约定，存在性
校验由 3.2 的 ``build_tennis_scene`` 在带球场景中执行。

单位约定：m、s、rad、rad/s、rad/s²、N·m。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

import numpy as np
import yaml

from src.robot.model_factory import load_robot_model
from src.robot.robot_profile import RobotProfile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    validate_runtime_parameters,
)

# §共享接口契约中的数组类型别名
Array = np.ndarray

# 当前支持的仿真任务配置 schema 版本（必须为整数 1）
SUPPORTED_SCHEMA_VERSION = 1

# 合法执行器模式（位置模式为仿真 PD，不预设阶段 4 SDK 控制模式）
_ACTUATOR_MODES: tuple[str, ...] = ("torque", "position")

# 任务配置文件允许的顶层字段（未知字段一律拒绝，防拼写错误静默生效）
_TASK_FIELDS = (
    "schema_version",
    "robot_id",
    "joint_names",
    "ball_joint_name",
    "ball_geom_name",
    "racket_geom_names",
    "protected_geom_pairs",
    "workspace_center",
    "workspace_radius",
    "initial_ball_position",
    "initial_ball_velocity",
    "max_steps",
    "actuator_mode",
    "solver_config",
    "mpc_config",
    "cost_config",
)

# (3,) 只读向量字段
_VECTOR_FIELDS = (
    "workspace_center",
    "initial_ball_position",
    "initial_ball_velocity",
)

# 三个透传配置节（值结构由 3.5-3.8 消费方定义，此处只要求映射结构）
_CONFIG_SECTIONS = ("solver_config", "mpc_config", "cost_config")


@dataclass(frozen=True)
class SimTaskConfig:
    """仿真击球任务配置 — 几何名称/工作空间/球初态/执行器模式。

    Attributes:
        robot_id: 机器人身份标识，须与 profile.robot_id 一致。
        joint_names: 主动关节名，须与 profile.active_joint_names 逐项同序
            （task 配置以 robot_id+joint_names 校验侧别，不靠文件名猜测）。
        ball_joint_name: 网球 free joint 名（3.2 带球场景按名组装/解析）。
        ball_geom_name: 网球碰撞 geom 名。
        racket_geom_names: 球拍碰撞 geom 名元组（至少一个）。
        protected_geom_pairs: 禁止接触的 geom 名对（碰撞保护语义，
            由 3.2 解析为 geom id 对；球拍击球接触不在此列）。
        workspace_center: (3,) 击球工作区中心（世界系，m）。
        workspace_radius: 工作区半径（m），须为正。
        initial_ball_position: (3,) 球初始位置（世界系，m）。
        initial_ball_velocity: (3,) 球初始速度（世界系，m/s）。
        max_steps: 单次 episode 最大步数（正整数）。
        actuator_mode: 执行器模式，"torque" 或 "position"（仿真 PD）。
        solver_config: 求解器透传配置（3.5 消费）。
        mpc_config: MPC 透传配置（3.8 消费）。
        cost_config: 代价权重透传配置（3.6 消费）。
    """

    robot_id: str
    joint_names: tuple[str, ...]
    ball_joint_name: str
    ball_geom_name: str
    racket_geom_names: tuple[str, ...]
    protected_geom_pairs: tuple[tuple[str, str], ...]
    workspace_center: Array
    workspace_radius: float
    initial_ball_position: Array
    initial_ball_velocity: Array
    max_steps: int
    actuator_mode: Literal["torque", "position"]
    solver_config: dict[str, Any]
    mpc_config: dict[str, Any]
    cost_config: dict[str, Any]

    def __post_init__(self) -> None:
        """冻结数组为只读 float64 副本、dict 浅拷贝、校验标量与名称。"""
        context = f"robot_id: {self.robot_id}"
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(
            self, "racket_geom_names", tuple(self.racket_geom_names)
        )
        object.__setattr__(
            self,
            "protected_geom_pairs",
            tuple(
                (str(pair[0]), str(pair[1]))
                for pair in self.protected_geom_pairs
            ),
        )
        for field in _VECTOR_FIELDS:
            object.__setattr__(
                self, field, _readonly_vector(getattr(self, field), field, context)
            )
        object.__setattr__(
            self,
            "workspace_radius",
            _positive_float(self.workspace_radius, "workspace_radius", context),
        )
        object.__setattr__(
            self, "max_steps", _positive_int(self.max_steps, "max_steps", context)
        )
        if self.actuator_mode not in _ACTUATOR_MODES:
            _reject(
                "actuator_mode",
                f"{self.actuator_mode!r} 不在支持模式 {_ACTUATOR_MODES} 中",
                context,
            )
        for field in _CONFIG_SECTIONS:
            value = getattr(self, field)
            if not isinstance(value, dict):
                _reject(field, "必须是映射结构", context)
            object.__setattr__(self, field, dict(value))
        _check_names(self, context)


def load_sim_task_config(path: Path, profile: RobotProfile) -> SimTaskConfig:
    """从 YAML 文件加载仿真任务配置并对照 profile 校验侧别。

    Args:
        path: 任务配置 YAML 文件路径（如 configs/r1d_sim_left.yaml）。
        profile: 机器人描述，作为身份与关节顺序的校验基准。

    Returns:
        不可变的 SimTaskConfig 实例（数组为只读副本）。

    Raises:
        FileNotFoundError: 任务配置文件不存在或不是常规文件。
        ValueError: 结构错误、未知/缺失字段、schema 版本不支持、
            向量维数/NaN/非正标量、actuator_mode 非法，或
            robot_id/joint_names 与 profile 不一致（不靠文件名猜测侧别）。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"任务配置文件不存在或不是常规文件: {path}")

    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"任务配置文件必须是映射结构（文件: {path}）")

    # 未知字段检查（先于字段级校验，尽早暴露拼写错误）
    unknown = sorted(map(str, set(raw) - set(_TASK_FIELDS)))
    if unknown:
        raise ValueError(f"未知字段 {unknown}（文件: {path}）")
    for field in _TASK_FIELDS:
        if field not in raw:
            raise ValueError(f"缺少字段 {field}（文件: {path}）")

    _require_schema_version(raw, path)

    robot_id = _require_str(raw, "robot_id", path)
    if robot_id != profile.robot_id:
        _reject(
            "robot_id",
            f"'{robot_id}' 与 profile.robot_id '{profile.robot_id}' 不一致"
            f"（任务配置: {path}）",
            f"robot_id: {profile.robot_id}",
        )

    joint_names = _require_name_list(raw, "joint_names", path)
    if joint_names != tuple(profile.active_joint_names):
        _reject(
            "joint_names",
            f"{list(joint_names)} 与 profile.active_joint_names "
            f"{list(profile.active_joint_names)} 不一致（须逐项同序；"
            f"任务配置: {path}）",
            f"robot_id: {profile.robot_id}",
        )

    vectors = {
        field: _require_vector(raw, field, path) for field in _VECTOR_FIELDS
    }

    return SimTaskConfig(
        robot_id=robot_id,
        joint_names=joint_names,
        ball_joint_name=_require_str(raw, "ball_joint_name", path),
        ball_geom_name=_require_str(raw, "ball_geom_name", path),
        racket_geom_names=_require_name_list(
            raw, "racket_geom_names", path, allow_empty=False
        ),
        protected_geom_pairs=_require_pair_list(raw, path),
        workspace_center=vectors["workspace_center"],
        workspace_radius=_positive_float(
            raw["workspace_radius"], "workspace_radius", f"文件: {path}"
        ),
        initial_ball_position=vectors["initial_ball_position"],
        initial_ball_velocity=vectors["initial_ball_velocity"],
        max_steps=_positive_int(raw["max_steps"], "max_steps", f"文件: {path}"),
        actuator_mode=cast(
            Literal["torque", "position"], _require_actuator_mode(raw, path)
        ),
        solver_config=_require_mapping(raw, "solver_config", path),
        mpc_config=_require_mapping(raw, "mpc_config", path),
        cost_config=_require_mapping(raw, "cost_config", path),
    )


def validate_simulation_inputs(
    profile: RobotProfile,
    parameters: RobotRuntimeParameters,
    task: SimTaskConfig,
) -> None:
    """严格仿真预检：模型回填 q 限位后校验限值完整性与侧别一致性。

    流程：
    1. task 与 profile 的身份/关节序一致性（防御手构对象，与加载路径
       同规则）；
    2. ``load_robot_model`` 组装模型并从 ``model.jnt_range`` 回填
       q_lower/q_upper（模型是位置限位唯一权威；显式提供且不一致即拒绝）；
    3. ``validate_runtime_parameters(purpose="simulation")``：全部动态
       限值必须完整有限，未知（None）限值不得进入仿真执行器；
    4. ``actuator_mode == "position"`` 时 kp/kd 必须完整（仿真 PD 增益）。

    Args:
        profile: 机器人描述（robot_id 须为 unitree_r1d）。
        parameters: 仿真运行参数（q_lower/q_upper 通常为 None，由模型回填）。
        task: 仿真任务配置。

    Raises:
        ValueError: 身份/关节序不一致、模型组装或限位回填失败、
            simulation 完整性校验失败，或位置模式缺 kp/kd。
    """
    context = f"robot_id: {profile.robot_id}，来源: {parameters.source}"
    if task.robot_id != profile.robot_id:
        _reject(
            "robot_id",
            f"task.robot_id '{task.robot_id}' 与 profile.robot_id "
            f"'{profile.robot_id}' 不一致",
            context,
        )
    if tuple(task.joint_names) != tuple(profile.active_joint_names):
        _reject(
            "joint_names",
            f"task {list(task.joint_names)} 与 profile.active_joint_names "
            f"{list(profile.active_joint_names)} 不一致（须逐项同序）",
            context,
        )

    # 先模型回填 q 限位（jnt_range），再做 simulation 完整性预检；
    # load_robot_model 内部含模型形状/初态/显式限位一致性校验。
    bundle = load_robot_model(profile, parameters)
    validate_runtime_parameters(
        bundle.parameters, profile, purpose="simulation"
    )

    # 位置模式是仿真 PD：必须有 kp/kd 才能构成执行器闭环
    if task.actuator_mode == "position":
        missing = [
            field
            for field in ("kp", "kd")
            if getattr(bundle.parameters, field) is None
        ]
        if missing:
            _reject(
                "/".join(missing),
                "position 模式需要完整 kp/kd 仿真 PD 增益，"
                "当前为 None（不得隐式补默认值）",
                context,
            )


# ---------- 底层校验/转换辅助 ----------


def _reject(field: str, message: str, context: str) -> NoReturn:
    """抛出含字段名与上下文的 ValueError（统一错误格式）。

    Raises:
        ValueError: 始终抛出（NoReturn 供类型收窄）。
    """
    raise ValueError(f"字段 {field} {message}（{context}）")


def _readonly_vector(value: Any, field: str, context: str) -> Array:
    """把 (3,) 向量复制为只读 float64 副本并校验有限性。

    Raises:
        ValueError: 形状不是 (3,) 或含 NaN/Inf。
    """
    array = np.array(value, dtype=np.float64)
    if array.shape != (3,):
        _reject(field, f"形状 {array.shape} 应为 (3,)", context)
    if not bool(np.all(np.isfinite(array))):
        _reject(field, "含 NaN/Inf", context)
    array.setflags(write=False)
    return array


def _positive_float(value: Any, field: str, context: str) -> float:
    """把标量转为严格正的有限 float。

    Raises:
        ValueError: 值是 bool、非数值、NaN/Inf 或非正数。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject(field, "必须是数值标量（bool 不作为数值接受）", context)
    result = float(value)
    if not math.isfinite(result):
        _reject(field, "含 NaN/Inf", context)
    if not result > 0:
        _reject(field, f"{result} 必须为正数", context)
    return result


def _positive_int(value: Any, field: str, context: str) -> int:
    """把标量转为严格正整数（bool 拒绝）。

    Raises:
        ValueError: 值是 bool、非整数类型或非正数。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        _reject(field, "必须是整数（bool 不作为整数接受）", context)
    if value <= 0:
        _reject(field, f"{value} 必须为正整数", context)
    return value


def _check_names(task: SimTaskConfig, context: str) -> None:
    """校验名称类字段：非空、无重复、保护对结构完整。"""
    if not task.joint_names:
        _reject("joint_names", "不能为空", context)
    for field, names in (
        ("joint_names", task.joint_names),
        ("racket_geom_names", task.racket_geom_names),
    ):
        if any(not str(name).strip() for name in names):
            _reject(field, "含空字符串元素", context)
        duplicates = sorted(
            {name for name in names if names.count(name) > 1}
        )
        if duplicates:
            _reject(field, f"存在重复名称: {duplicates}", context)
    if not task.racket_geom_names:
        _reject("racket_geom_names", "不能为空（至少一个球拍 geom）", context)
    if not task.protected_geom_pairs:
        _reject("protected_geom_pairs", "不能为空（至少一对保护几何）", context)
    for pair in task.protected_geom_pairs:
        if len(pair) != 2:
            _reject(
                "protected_geom_pairs",
                f"{list(pair)} 必须是两个名称的 geom 对",
                context,
            )
        if any(not str(name).strip() for name in pair):
            _reject("protected_geom_pairs", "含空字符串名称", context)


# ---------- YAML 解析辅助（错误消息均含文件路径） ----------


def _require_schema_version(raw: dict[str, Any], path: Path) -> int:
    """校验 schema_version 字段（bool 不视为合法整数，必须为 1）。"""
    value = raw.get("schema_version")
    if isinstance(value, bool) or not isinstance(value, int):
        _reject("schema_version", "必须是整数", f"文件: {path}")
    if value != SUPPORTED_SCHEMA_VERSION:
        _reject(
            "schema_version",
            f"不支持: {value}（当前支持 {SUPPORTED_SCHEMA_VERSION}）",
            f"文件: {path}",
        )
    return value


def _require_str(raw: dict[str, Any], field: str, path: Path) -> str:
    """校验非空字符串字段。"""
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        _reject(field, "必须是非空字符串", f"文件: {path}")
    return value


def _require_name_list(
    raw: dict[str, Any], field: str, path: Path, *, allow_empty: bool = True
) -> tuple[str, ...]:
    """校验名称列表字段：字符串列表、元素非空且不重复。"""
    value = raw.get(field)
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        _reject(field, "必须是字符串列表", f"文件: {path}")
    if not allow_empty and not value:
        _reject(field, "不能为空列表", f"文件: {path}")
    names = tuple(value)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        _reject(field, f"存在重复名称: {duplicates}", f"文件: {path}")
    if any(not n.strip() for n in names):
        _reject(field, "含空字符串元素", f"文件: {path}")
    return names


def _require_pair_list(
    raw: dict[str, Any], path: Path
) -> tuple[tuple[str, str], ...]:
    """校验 protected_geom_pairs：非空、每项为两个非空字符串的列表。"""
    value = raw.get("protected_geom_pairs")
    if not isinstance(value, list) or not value:
        _reject("protected_geom_pairs", "必须是非空列表", f"文件: {path}")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(not isinstance(name, str) or not name.strip() for name in item)
        ):
            _reject(
                "protected_geom_pairs",
                f"{item} 必须是两个非空字符串名称的列表",
                f"文件: {path}",
            )
        pairs.append((item[0], item[1]))
    return tuple(pairs)


def _require_vector(raw: dict[str, Any], field: str, path: Path) -> Array:
    """校验 (3,) 数值向量字段（bool/字符串/NaN/Inf 拒绝）。"""
    value = raw.get(field)
    if not isinstance(value, list):
        _reject(field, "必须是数值列表", f"文件: {path}")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for item in value
    ):
        _reject(
            field, "必须是数值列表（bool/字符串不作为数值接受）", f"文件: {path}"
        )
    return _readonly_vector(value, field, f"文件: {path}")


def _require_actuator_mode(raw: dict[str, Any], path: Path) -> str:
    """校验 actuator_mode 字面量。"""
    value = raw.get("actuator_mode")
    if not isinstance(value, str) or value not in _ACTUATOR_MODES:
        _reject(
            "actuator_mode",
            f"{value!r} 不在支持模式 {_ACTUATOR_MODES} 中",
            f"文件: {path}",
        )
    return value


def _require_mapping(raw: dict[str, Any], field: str, path: Path) -> dict[str, Any]:
    """校验透传配置节为映射结构（键结构由 3.5-3.8 消费方定义）。"""
    value = raw.get(field)
    if not isinstance(value, dict):
        _reject(field, "必须是映射结构", f"文件: {path}")
    return dict(value)
