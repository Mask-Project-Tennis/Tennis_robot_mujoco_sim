"""V12 唯一任务配置解析层（R1D-V12 parity T3）。

本模块把 ``scripts/rm65_mpc_v12.py::main`` 里散落的配置合并、默认值、服务箱
自动参数、显式 CLI 覆盖与 R1D 专用覆盖集中到一处，产出深拷贝隔离的
:class:`ResolvedV12Config`（numpy 数组只读；dict/list 容器本身仍可变，
但修改只影响该副本）。设计约束（见 T3 brief）：

1. **RM65 旧路径逐字段保持**：配置来源顺序必须是
   ``configs/default.yaml`` → ``configs/v5_active_hit.yaml`` →
   ``configs/mpc.yaml`` → V12 字面量默认 → 服务箱自动值 → 显式 CLI 覆盖；
   仓库常量（``src/robot/constants.py``）不经 YAML 复制。
2. **R1D 从同一解析派生**：先得到与 RM65 完全相同的基线，再应用
   机器人专用覆盖（``configs/r1d_v12.yaml``）与显式任务覆盖，最后应用
   专用 CLI 旗标。禁止把 RM65 的六维数组补零后当成 R1D 配置。
3. **路径只来自显式配置或显式注册表**：不按文件名猜左/右侧，不读取
   未声明的 sidecar 文件；注册表未登记且未显式提供 runtime/task 时直接失败。
4. **运行前失败**：未知 robot、错配 runtime、非法任务覆盖、6/7 维向量
   冲突、固定球兼容模式与来球参数冲突等都在解析阶段抛出 ``ValueError``，
   不得接受后静默忽略。

本模块只依赖纯配置层（numpy/yaml + ``src.robot`` 的 profile/runtime 解析），
不导入 MuJoCo、仿真环境或球拍实现。
"""

from __future__ import annotations

import argparse
import copy
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
import yaml

from src.robot.constants import (
    BOUNCE_RESTITUTION,
    INIT_Q,
    INIT_Q_REAL,
    SHOULDER_POS,
    WORKSPACE_RADIUS,
)
from src.robot.robot_profile import RobotProfile, load_robot_profile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    load_runtime_parameters,
)
from src.utils.yaml_utils import load_config, merge_configs

__all__ = [
    "R1D_V12_REGISTRY",
    "R1DConfigRegistry",
    "ResolvedV12Config",
    "normalize_explicit_options",
    "resolve_v12_config",
]

# 仓库根目录与配置目录：本文件位于 src/sim/，向上两级即仓库根。
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO_ROOT / "configs"
_R1D_TASK_CONFIG_PATH = _CONFIGS_DIR / "r1d_v12.yaml"
_R1D_RIGHT_PROFILE_PATH = _CONFIGS_DIR / "robots" / "r1d_right.yaml"

# V12 脚本中的目标区域字面量（原 main() 内硬编码，保持逐位一致）。
_TARGET_CENTER = np.array([-0.82765693, -0.47411682, 0.86947444], dtype=np.float64)
_TARGET_OFFSET = 0.10
# 发球区字面量：中心 (0, -serve_distance, serve_height)，全尺寸 8 × 0.2 × 0.3。
_SERVE_SPEED_RANGE = (8.0, 18.0)
_HIT_TIME_UNIFORM = (0.3, 0.4)
_BALL_START_Y_RANGE = (-5.5, -4.5)
_BALL_START_Z_RANGE = (1.4, 1.8)
_BALL_DIRECTION = "y"


@dataclass(frozen=True)
class R1DConfigRegistry:
    """R1D 显式配置注册表条目。

    只登记“机器人描述 → runtime / 任务覆盖 / 球拍配置”的显式三元组，
    不做文件名推断。字段值均为绝对路径。

    Attributes:
        side: 侧别（``"right"`` / ``"left"``），用于校验关节名前缀。
        runtime_path: R1D V12 运行参数 YAML 路径。
        task_path: R1D V12 任务覆盖 YAML 路径。
        racket_path: 球拍配置 YAML 路径（由 T1 交付，本层只记录路径）。
    """

    side: str
    runtime_path: Path
    task_path: Path
    racket_path: Path


#: R1D 显式注册表：键为解析后的机器人描述绝对路径。
#:
#: 当前只登记右臂：右臂是 T5/T7 的验收臂；左臂尚未提供 V12 专用 runtime /
#: 球拍配置，因此不登记，必须由调用方显式传三个路径，避免“按文件名猜侧别”。
R1D_V12_REGISTRY: Mapping[Path, R1DConfigRegistry] = MappingProxyType(
    {
        _R1D_RIGHT_PROFILE_PATH.resolve(): R1DConfigRegistry(
            side="right",
            runtime_path=(
                _CONFIGS_DIR / "robots" / "r1d_v12_runtime_right.yaml"
            ).resolve(),
            task_path=_R1D_TASK_CONFIG_PATH.resolve(),
            racket_path=(
                _CONFIGS_DIR / "rackets" / "blade100_v9_reference.yaml"
            ).resolve(),
        )
    }
)


# ---------------------------------------------------------------------------
# CLI 字段缺省值（与 scripts/rm65_mpc_v12.py::_build_parser 保持一致）
# ---------------------------------------------------------------------------

_CLI_DEFAULTS: dict[str, Any] = {
    "viewer": False,
    "seed": None,
    "solver_backend": "auto",
    "horizon": None,
    "iter": None,
    "fix_joint5": False,
    "backswing": 0.6,
    "bs_ratio": 0.35,
    "use_backswing": False,
    "r_decay": 0.40,
    "no_r_decay": False,
    "hit_shift": 0.0,
    "ball_speed": None,
    "ball_distance": None,
    "approach_angle": 0.0,
    "serve_box": False,
    "no_bounce": False,
    "serve_distance": 8.0,
    "serve_height": 1.2,
    "serve_x_size": 8.0,
    "serve_y_size": 0.2,
    "serve_z_size": 0.3,
    "Q_tcp_soft": 0.0,
    "Q_qdot_limit": 0.0,
    "normal_weight": 500000.0,
    "normal_flip": False,
    "replan_interval": None,
    "near_iters": None,
    "window_ms": 50.0,
    "softmin_beta": 5.0,
    "corridor_radius": 0.12,
    "ablation": None,
    "no_softmin": False,
    "no_tube": False,
    "no_follow_through": False,
    "follow_trigger": "planned",
    "target_speed": 1.8,
    "no_plot": False,
    "async_replan": False,
    "time_perturb_ms": 0.0,
    "space_perturb_m": 0.0,
    "spatial_perturb_target": "hitpoint",
    "perturb_alpha_min": 0.0,
    "shared_input_full": False,
    "random_perturb": False,
    "perturb_sign": "random",
    "time_perturb_min_ms": 0.0,
    "space_perturb_min_m": 0.0,
    "obs_freq": 0.0,
    "obs_noise_pos": 0.0,
    "obs_noise_vel": 0.0,
    "obs_use_kf": False,
    "position_mode": False,
    "no_feedforward": False,
    "kp": None,
    "kd": None,
    "far_threshold": None,
    "near_threshold": None,
    "first_plan_iters": None,
    "Qp_scale_far": None,
    "Qv_scale_far": None,
    "Qp_scale_near": None,
    "Qv_scale_near": None,
    "follow_kp": None,
    "follow_kd": None,
    "hit_lock": None,
    "hard_margin": None,
    "warn_margin": None,
    "j1_warn": None,
    "refiner_window": None,
    "robot_config": None,
    "runtime_config": None,
    "sim_task_config": None,
    "racket_config": None,
    "task_overrides": None,
    "output": None,
    "limits_config": None,
}

#: 部分 CLI 旗标的展示名带大写或数字，来源标签按此映射还原。
_FLAG_DISPLAY: dict[str, str] = {
    "Q_tcp_soft": "--Q-tcp-soft",
    "Q_qdot_limit": "--Q-qdot-limit",
    "Qp_scale_far": "--Qp-scale-far",
    "Qv_scale_far": "--Qv-scale-far",
    "Qp_scale_near": "--Qp-scale-near",
    "Qv_scale_near": "--Qv-scale-near",
}


def _flag(name: str) -> str:
    """把 argparse dest 还原为命令行旗标文本（用于来源标签）。"""
    return _FLAG_DISPLAY.get(name, "--" + name.replace("_", "-"))


def normalize_explicit_options(
    options: Iterable[str | argparse.Action] | None,
) -> frozenset[str]:
    """把显式 CLI 选项规范化为小写 dest 集合。

    Args:
        options: 原始选项文本（可混用 ``--flag`` 与 argparse dest，如
            ``"ball_speed"``）、或 argparse Action 对象（T5 的
            ``_explicit_cli_actions()`` 正是返回 ``list[argparse.Action]``）。
            对 Action 会同时纳入其 ``dest`` 与全部 ``option_strings``。

    Returns:
        小写、去 ``--`` 前缀、连字符转下划线的 frozenset。
    """
    result: set[str] = set()
    for raw in options or ():
        if isinstance(raw, argparse.Action):
            names: list[str] = []
            if raw.dest:
                names.append(str(raw.dest))
            names.extend(str(option) for option in raw.option_strings)
        else:
            names = [str(raw)]
        for name in names:
            text = name.strip()
            if text.startswith("--"):
                text = text[2:]
            elif text.startswith("-"):
                text = text[1:]
            text = text.replace("-", "_").lower()
            if text:
                result.add(text)
    return frozenset(result)


def _arg(args: argparse.Namespace, name: str) -> Any:
    """读取 argparse 字段；字段缺失时回退到 V12 parser 缺省值。"""
    return getattr(args, name, _CLI_DEFAULTS.get(name))


def _is_explicit(
    args: argparse.Namespace,
    name: str,
    explicit: frozenset[str],
) -> bool:
    """判断某 CLI 字段是否由用户显式给出。

    语义与 V12 原脚本一致：

    * ``--flag`` 布尔：值为 True 即视为显式；
    * 可空字段（缺省 None）：值非 None 即视为显式；
    * 有非空数值/字符串缺省的字段：只有出现在显式选项集合中才算显式，
      以免缺省值覆盖 R1D 覆盖层。
    """
    if name.lower() in explicit:
        return True
    default = _CLI_DEFAULTS.get(name)
    value = _arg(args, name)
    if isinstance(default, bool):
        return bool(value)
    if default is None:
        return value is not None
    return False


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


# ---------------------------------------------------------------------------
# 任务覆盖白名单与类型校验
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float:
    """把值转换为有限 float；拒绝布尔与字符串（防止 YAML 隐式类型混淆）。"""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"必须是数值，实际为 {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("必须是有限数值")
    return result


def _v_num(value: Any) -> float:
    return _as_float(value)


def _v_pos_num(value: Any) -> float:
    number = _as_float(value)
    if number <= 0.0:
        raise ValueError("必须大于 0")
    return number


def _v_nonneg_num(value: Any) -> float:
    number = _as_float(value)
    if number < 0.0:
        raise ValueError("必须大于等于 0")
    return number


def _v_pos_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("必须是整数")
    number = int(value)
    if number <= 0:
        raise ValueError("必须大于 0")
    return number


def _v_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return _v_pos_int(value)


def _v_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("必须是布尔值")
    return value


def _v_str(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("必须是非空字符串")
    return value.strip()


def _choice_validator(*allowed: str) -> Callable[[Any], str]:
    """构造枚举字符串校验器。"""

    def validate(value: Any) -> str:
        text = _v_str(value)
        if text not in allowed:
            raise ValueError(f"必须是 {', '.join(allowed)} 之一")
        return text

    return validate


def _v_vec3(value: Any) -> np.ndarray:
    """校验并返回 3 维 float 向量。"""
    if isinstance(value, (str, bytes)) or not isinstance(
        value, (list, tuple, np.ndarray)
    ):
        raise ValueError("必须是长度为 3 的数值序列")
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 3:
        raise ValueError("必须是长度为 3 的数值序列")
    if not np.all(np.isfinite(array)):
        raise ValueError("必须全部为有限数值")
    return array


def _v_vec3_list(value: Any) -> list[float]:
    return [float(item) for item in _v_vec3(value)]


def _v_gain_vector(value: Any) -> list[float]:
    """校验 PD 增益序列（长度在解析完成后按主关节数检查）。"""
    if isinstance(value, (str, bytes)) or not isinstance(
        value, (list, tuple, np.ndarray)
    ):
        raise ValueError("必须是数值序列")
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("不能为空")
    if not np.all(np.isfinite(array)):
        raise ValueError("必须全部为有限数值")
    return [float(item) for item in array]


def _v_optional_vec3(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return _v_vec3(value)


_OVERRIDE_VALIDATORS: dict[str, Callable[[Any], Any]] = {}


def _register(path: str, validator: Callable[[Any], Any]) -> None:
    _OVERRIDE_VALIDATORS[path] = validator


# 顶层与执行器
_register("seed", _v_optional_int)
_register("actuator.mode", _choice_validator("torque", "position"))
_register("actuator.feedforward", _v_bool)
_register("actuator.kp", _v_gain_vector)
_register("actuator.kd", _v_gain_vector)

# MPC 时间/阈值/权重字面量
for _key in (
    "total_horizon",
    "fixed_horizon",
    "replan_interval",
    "max_iter_per_plan",
    "first_plan_iters",
    "near_plan_iters",
    "follow_through_steps",
    "hit_lock_threshold",
    "refiner_window_half",
):
    _register(f"mpc.{_key}", _v_pos_int)
for _key in ("near_threshold", "far_threshold"):
    _register(f"mpc.{_key}", _v_pos_int)
for _key in (
    "Q_p_scale_far",
    "Q_v_scale_far",
    "Q_p_scale_near",
    "Q_v_scale_near",
    "normal_weight",
    "racket_speed",
    "target_speed",
    "Q_tcp_soft",
    "Q_qdot_limit",
    "softmin_beta",
    "follow_through_length",
    "follow_through_v_terminal",
    "follow_kp",
    "follow_kd",
    "hard_margin_deg",
    "warn_margin_deg",
    "j1_warn_margin_deg",
    "backswing_offset",
    "backswing_ratio",
    "r_decay_ratio",
    "time_perturb_s",
    "space_perturb_m",
    "perturb_alpha_min",
):
    _register(f"mpc.{_key}", _v_num)
for _key in (
    "use_backswing",
    "use_r_decay",
    "async_mode",
    "normal_flip",
    "no_follow_through",
    "shared_input_full",
):
    _register(f"mpc.{_key}", _v_bool)
_register("mpc.follow_trigger", _choice_validator("planned", "contact"))
_register(
    "mpc.spatial_perturb_target",
    _choice_validator("hitpoint", "ballstate"),
)

# 求解器
_register("solver.backend", _choice_validator("auto", "python", "cpp"))
for _key in ("max_iter", "horizon"):
    _register(f"solver.{_key}", _v_pos_int)
for _key in ("tol", "mu_min", "mu_max", "mu_init", "delta_0", "lin_eps"):
    _register(f"solver.{_key}", _v_num)

# 代价
_register("cost.Q_p", _v_vec3_list)
_register("cost.Q_v", _v_vec3_list)
for _key in ("R", "Q_qdot", "Q_qddot", "Q_du"):
    _register(f"cost.{_key}", _v_nonneg_num)
for _stage in ("far", "mid", "near"):
    for _mult in ("Q_qdot_mult", "Q_qddot_mult", "Q_du_mult"):
        _register(f"cost.stage_weights.{_stage}.{_mult}", _v_num)
for _key in (
    "window_half_ms",
    "Q_p_tube",
    "Q_v_tube",
    "Q_n_tube",
    "tube_cost_ratio",
    "softmin_beta",
    "corridor_radius",
):
    _register(f"cost.tube.{_key}", _v_num)
_register("cost.tube.use_softmin_terminal", _v_bool)

# 感知
_register("perception.obs_freq", _v_nonneg_num)
_register("perception.obs_noise_pos", _v_nonneg_num)
_register("perception.obs_noise_vel", _v_nonneg_num)
_register("perception.obs_use_kf", _v_bool)

# 来球
_register("serve.mode", _choice_validator("serve_box", "target_box"))
_register("serve.ball_speed", _v_pos_num)
_register("serve.ball_distance", _v_pos_num)
_register("serve.approach_angle", _v_num)
for _key in (
    "serve_distance",
    "serve_height",
    "serve_x_size",
    "serve_y_size",
    "serve_z_size",
    "bounce_restitution",
):
    _register(f"serve.{_key}", _v_pos_num)
_register("serve.use_bounce", _v_bool)

# 目标区域
_register("target.center", _v_vec3)
_register("target.offset", _v_nonneg_num)
_register("target.workspace_radius", _v_pos_num)
_register("target.shoulder_pos", _v_optional_vec3)

# 击球参数（目前只开放方向）
_register("hitting.hit_direction", _v_vec3)


def _flatten_overrides(value: Any, prefix: str = "") -> dict[str, Any]:
    """把（可能嵌套的）任务覆盖映射展开为点路径扁平字典。"""
    if not isinstance(value, Mapping):
        raise ValueError("非法任务覆盖键：任务覆盖必须是映射")
    flat: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError("非法任务覆盖键：键必须是非空字符串")
        path = f"{prefix}.{raw_key.strip()}" if prefix else raw_key.strip()
        if isinstance(raw_value, Mapping):
            flat.update(_flatten_overrides(raw_value, path))
        else:
            flat[path] = raw_value
    return flat


def _validate_overrides(
    raw_overrides: Any,
    origin: str,
) -> dict[str, Any]:
    """展开并校验任务覆盖；未知键或类型错误立即失败。"""
    flat = _flatten_overrides(raw_overrides)
    validated: dict[str, Any] = {}
    for key, value in flat.items():
        validator = _OVERRIDE_VALIDATORS.get(key)
        if validator is None:
            raise ValueError(
                f"非法任务覆盖键: {key}（来源 {origin}，允许键见 "
                "src/sim/v12_config.py 的 _OVERRIDE_VALIDATORS 注册表）"
            )
        try:
            validated[key] = validator(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"任务覆盖值类型非法: {key}={value!r}（来源 {origin}）：{exc}"
            ) from exc
    return validated


# ---------------------------------------------------------------------------
# 机器人选择：RM65 旧路径 / R1D 显式路径
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RobotSelection:
    """机器人选择结果（内部结构，不对外暴露）。"""

    robot_id: str
    ndof: int
    side: str | None
    profile: RobotProfile | None
    profile_path: Path | None
    runtime_path: Path | None
    runtime_params: RobotRuntimeParameters | None
    task_path: Path | None
    racket_path: Path | None
    task_mode: str
    overlay: dict[str, Any]
    overlay_origin: str | None
    path_sources: dict[str, str]


def _resolve_explicit_path(raw: Any, label: str) -> Path:
    """解析并校验显式给出的文件路径（存在性检查）。"""
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} 文件不存在: {path}")
    return path


def _resolve_racket_path(raw: Any) -> Path | None:
    if raw is None:
        return None
    return _resolve_explicit_path(raw, "显式 --racket-config")


def _load_task_overlay(path: Path, profile: RobotProfile) -> dict[str, Any]:
    """加载并校验 R1D V12 任务覆盖 YAML。"""
    if not path.is_file():
        raise FileNotFoundError(f"R1D V12 任务覆盖文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"R1D V12 任务覆盖必须是映射: {path}")
    allowed = {"schema_version", "robot_id", "overrides"}
    unknown = set(raw.keys()) - allowed
    if unknown:
        raise ValueError(
            f"R1D V12 任务覆盖存在未知顶层键 {sorted(unknown)}（允许 {sorted(allowed)}）"
        )
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("R1D V12 任务覆盖 schema_version 必须为 1")
    if raw.get("robot_id") != profile.robot_id:
        raise ValueError(
            "R1D V12 任务覆盖 robot_id 与机器人描述不一致："
            f"{raw.get('robot_id')!r} != {profile.robot_id!r}"
        )
    overrides = raw.get("overrides") or {}
    return _validate_overrides(overrides, origin=path.name)


def _check_compat_task(path: Path, profile: RobotProfile) -> None:
    """校验旧固定球 sim-task YAML 的机器人身份（不导入 sim_config）。"""
    if not path.is_file():
        raise FileNotFoundError(f"显式 --sim-task-config 文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"旧 sim-task 配置必须是映射: {path}")
    if raw.get("robot_id") != profile.robot_id:
        raise ValueError(
            "旧 sim-task 配置 robot_id 与机器人描述不一致："
            f"{raw.get('robot_id')!r} != {profile.robot_id!r}"
        )
    joint_names = raw.get("joint_names")
    if not isinstance(joint_names, (list, tuple)):
        raise ValueError("旧 sim-task 配置缺少 joint_names 列表")
    if tuple(joint_names) != profile.active_joint_names:
        raise ValueError(
            "旧 sim-task 配置 joint_names 与机器人描述主动关节顺序不一致"
        )
    actuator_mode = raw.get("actuator_mode")
    if actuator_mode is not None and actuator_mode not in ("torque", "position"):
        raise ValueError(
            f"旧 sim-task 配置 actuator_mode 非法: {actuator_mode!r}"
        )


_FIXED_BALL_CONFLICT_FLAGS: tuple[tuple[str, str], ...] = (
    ("serve_box", "--serve-box"),
    ("ball_speed", "--ball-speed"),
    ("ball_distance", "--ball-distance"),
    ("approach_angle", "--approach-angle"),
    ("no_bounce", "--no-bounce"),
    ("serve_distance", "--serve-distance"),
    ("serve_height", "--serve-height"),
    ("serve_x_size", "--serve-x-size"),
    ("serve_y_size", "--serve-y-size"),
    ("serve_z_size", "--serve-z-size"),
)

_R1D_FORBIDDEN_FLAGS: tuple[tuple[str, str], ...] = (
    ("limits_config", "--limits-config"),
    ("fix_joint5", "--fix-joint5"),
)


def _reject_r1d_rm65_flags(
    args: argparse.Namespace,
    explicit: frozenset[str],
) -> None:
    """R1D 明确拒绝 RM65 专属旗标（不能接受后忽略）。"""
    for dest, flag in _R1D_FORBIDDEN_FLAGS:
        if _is_explicit(args, dest, explicit):
            raise ValueError(
                f"R1D 不支持 {flag}：该旗标属于 RM65 旧路径，不能接受后忽略"
            )


def _select_robot(
    args: argparse.Namespace,
    explicit: frozenset[str],
) -> _RobotSelection:
    """解析机器人身份并确定 runtime/任务/球拍路径。"""
    robot_raw = _arg(args, "robot_config")
    runtime_raw = _arg(args, "runtime_config")
    sim_task_raw = _arg(args, "sim_task_config")
    racket_raw = _arg(args, "racket_config")

    if robot_raw is None:
        if runtime_raw is not None or sim_task_raw is not None:
            raise ValueError(
                "RM65 旧路径不接受 --runtime-config/--sim-task-config："
                "这两个参数只用于 R1D 显式配置"
            )
        racket_path = _resolve_racket_path(racket_raw)
        path_sources: dict[str, str] = {}
        if racket_path is not None:
            path_sources["racket_path"] = "cli:--racket-config"
        return _RobotSelection(
            robot_id="rm65",
            ndof=6,
            side=None,
            profile=None,
            profile_path=None,
            runtime_path=None,
            runtime_params=None,
            task_path=None,
            racket_path=racket_path,
            task_mode="legacy_cli",
            overlay={},
            overlay_origin=None,
            path_sources=path_sources,
        )

    profile_path = _resolve_explicit_path(robot_raw, "显式 --robot-config")
    profile = load_robot_profile(profile_path)

    if profile.robot_id == "rm65":
        if runtime_raw is not None or sim_task_raw is not None:
            raise ValueError(
                "RM65 旧路径不接受 --runtime-config/--sim-task-config："
                "这两个参数只用于 R1D 显式配置"
            )
        if profile.ndof != 6:
            raise ValueError(
                f"rm65 机器人描述主动关节数应为 6，实际为 {profile.ndof}"
            )
        racket_path = _resolve_racket_path(racket_raw)
        path_sources = {"robot_profile_path": "cli:--robot-config"}
        if racket_path is not None:
            path_sources["racket_path"] = "cli:--racket-config"
        return _RobotSelection(
            robot_id="rm65",
            ndof=6,
            side=None,
            profile=profile,
            profile_path=profile_path,
            runtime_path=None,
            runtime_params=None,
            task_path=None,
            racket_path=racket_path,
            task_mode="legacy_cli",
            overlay={},
            overlay_origin=None,
            path_sources=path_sources,
        )

    if profile.robot_id != "unitree_r1d":
        raise ValueError(
            f"未知 robot_id={profile.robot_id!r}：当前只支持 'rm65' 与 "
            "'unitree_r1d'（R1D 需要显式注册表或显式 runtime/task 路径）"
        )

    return _select_r1d(
        profile=profile,
        profile_path=profile_path,
        runtime_raw=runtime_raw,
        sim_task_raw=sim_task_raw,
        racket_raw=racket_raw,
        args=args,
        explicit=explicit,
    )


def _select_r1d(
    profile: RobotProfile,
    profile_path: Path,
    runtime_raw: Any,
    sim_task_raw: Any,
    racket_raw: Any,
    args: argparse.Namespace,
    explicit: frozenset[str],
) -> _RobotSelection:
    """解析 R1D 选择：显式 runtime/task 优先，其次显式注册表。"""
    if profile.ndof != 7:
        raise ValueError(
            f"unitree_r1d 机器人描述主动关节数应为 7，实际为 {profile.ndof}"
        )
    _reject_r1d_rm65_flags(args, explicit)

    registry = R1D_V12_REGISTRY.get(profile_path)
    if registry is not None:
        prefix = f"{registry.side}_"
        if not all(
            name.startswith(prefix) for name in profile.active_joint_names
        ):
            raise ValueError(
                "R1D 注册表侧别与机器人描述主动关节名前缀不一致："
                f"注册表侧别 {registry.side!r}，关节名 {profile.active_joint_names}"
            )

    path_sources: dict[str, str] = {"robot_profile_path": "cli:--robot-config"}

    # runtime：显式优先；否则取注册表；注册表未登记则留空（后续失败）
    if runtime_raw is not None:
        runtime_path = _resolve_explicit_path(runtime_raw, "显式 runtime")
        try:
            runtime_params = load_runtime_parameters(runtime_path, profile)
        except (TypeError, ValueError, FileNotFoundError) as exc:
            raise ValueError(
                "显式 runtime 校验失败：robot_id 或关节顺序与机器人描述不匹配"
                f"（{runtime_path.name}）：{exc}"
            ) from exc
        runtime_source = "cli:--runtime-config"
    elif registry is not None:
        runtime_path = registry.runtime_path
        runtime_params = None
        runtime_source = f"registry:r1d_{registry.side}"
    else:
        runtime_path = None
        runtime_params = None
        runtime_source = ""
    if runtime_source:
        path_sources["runtime_path"] = runtime_source

    # 任务：显式 sim-task → 固定球兼容模式；否则注册表任务覆盖 → R1D overlay
    if sim_task_raw is not None:
        task_path = _resolve_explicit_path(sim_task_raw, "显式 --sim-task-config")
        _check_compat_task(task_path, profile)
        for dest, flag in _FIXED_BALL_CONFLICT_FLAGS:
            if _is_explicit(args, dest, explicit):
                raise ValueError(
                    f"固定球兼容模式不允许 {flag}：来球由旧 sim-task 文件决定，"
                    "不能同时给 V12 发球参数"
                )
        task_mode = "r1d_fixed_ball_compat"
        overlay: dict[str, Any] = {}
        overlay_origin: str | None = None
        path_sources["task_path"] = "cli:--sim-task-config"
    else:
        if registry is None:
            raise ValueError(
                "该 R1D 机器人描述未在 R1D V12 注册表中登记：请显式提供 "
                "--runtime-config 与 --sim-task-config，或使用已注册的 "
                "configs/robots/r1d_right.yaml"
            )
        task_path = registry.task_path
        overlay = _load_task_overlay(task_path, profile)
        overlay_origin = task_path.name
        task_mode = "r1d_overlay"
        path_sources["task_path"] = f"registry:r1d_{registry.side}"

    # 固定球兼容模式必须同时有 runtime：旧入口用 runtime 重建模型/限位，
    # 不能在 resolve 阶段放过、留到 T5 才失败（复审 Q1）。
    if task_mode == "r1d_fixed_ball_compat" and runtime_path is None:
        raise ValueError(
            "固定球兼容模式（--sim-task-config）必须同时提供 --runtime-config："
            "旧入口依赖 runtime 重建模型与限位，不能留到 T5 才失败"
        )

    # 球拍：显式优先，其次注册表（本层只记录路径，不要求文件已交付）
    if racket_raw is not None:
        racket_path = _resolve_explicit_path(racket_raw, "显式 --racket-config")
        path_sources["racket_path"] = "cli:--racket-config"
    elif registry is not None:
        racket_path = registry.racket_path
        path_sources["racket_path"] = f"registry:r1d_{registry.side}"
    else:
        racket_path = None

    return _RobotSelection(
        robot_id="unitree_r1d",
        ndof=7,
        side=registry.side if registry is not None else None,
        profile=profile,
        profile_path=profile_path,
        runtime_path=runtime_path,
        runtime_params=runtime_params,
        task_path=task_path,
        racket_path=racket_path,
        task_mode=task_mode,
        overlay=overlay,
        overlay_origin=overlay_origin,
        path_sources=path_sources,
    )


# ---------------------------------------------------------------------------
# 基线值：default → v5 → mpc → V12 字面量 → 服务箱自动值
# ---------------------------------------------------------------------------


def _collect_config_sources(
    mapping: Mapping[str, Any],
    prefix: str,
    label: str,
    out: dict[str, str],
) -> None:
    """把 YAML 叶子路径记录为来源标签（嵌套字典递归展开）。"""
    for key, value in mapping.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            _collect_config_sources(value, path, label, out)
        else:
            out[path] = label


def _load_config_with_sources() -> tuple[dict[str, Any], dict[str, str]]:
    """按 V12 顺序合并 default/v5/mpc 并记录每个叶子来源。"""
    merged = load_config(_CONFIGS_DIR / "default.yaml")
    sources: dict[str, str] = {}
    _collect_config_sources(merged, "", "default.yaml", sources)
    for filename in ("v5_active_hit.yaml", "mpc.yaml"):
        path = _CONFIGS_DIR / filename
        if not path.is_file():
            continue
        overlay = load_config(path)
        merged = merge_configs(merged, overlay)
        _collect_config_sources(overlay, "", filename, sources)
    return merged, sources


def _stage_weights(
    root: Mapping[str, Any],
    name: str,
    defaults: Mapping[str, float],
) -> dict[str, float]:
    """读取单个阶段权重（缺省时使用 V12 字面量）。"""
    raw = root.get(name, {}) or {}
    return {
        key: float(raw.get(key, default)) for key, default in defaults.items()
    }


def _build_base_values(
    selection: _RobotSelection,
    args: argparse.Namespace,
    explicit: frozenset[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """构建解析基线（不含 R1D 覆盖层与显式 CLI 覆盖）。"""
    cfg, cfg_src = _load_config_with_sources()
    dt = float(cfg["sim"]["dt"])

    if selection.runtime_params is not None and not math.isclose(
        float(selection.runtime_params.dt),
        dt,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"显式 runtime dt={selection.runtime_params.dt} 与 "
            f"default.yaml sim.dt={dt} 不一致"
        )

    ball_cfg = cfg.get("ball", {}) or {}
    actuator_cfg = cfg.get("actuator", {}) or {}
    hitting_cfg = copy.deepcopy(cfg.get("hitting", {}) or {})
    cost_cfg = cfg.get("cost", {}) or {}
    stage_cfg = cfg.get("stage_weights", {}) or {}
    ilqt_cfg = copy.deepcopy(cfg.get("ilqt", {}) or {})

    sources: dict[str, str] = {"dt": "default.yaml"}

    # 主关节数：RM65 为 6，R1D 为 7；PD 增益基线不跨机器人复用。
    if selection.robot_id == "rm65":
        shoulder_pos: np.ndarray | None = SHOULDER_POS.copy()
        shoulder_source = "src/robot/constants.py"
    else:
        shoulder_pos = None
        shoulder_source = "deferred(T5 从模型派生)"

    # U 与观测：均为显式 CLI 输入或任务覆盖，基线只放零值。
    perception = {
        "obs_freq": 0.0,
        "obs_freq_effective": 0.0,
        "obs_noise_pos": 0.0,
        "obs_noise_vel": 0.0,
        "obs_use_kf": False,
        "noise_enabled": False,
        "use_kf": False,
        "kf_enabled": False,
        "obs_gate_enabled": False,
    }

    far_stage = _stage_weights(
        stage_cfg,
        "far",
        {"Q_qdot_mult": 1.0, "Q_qddot_mult": 1.0, "Q_du_mult": 1.0},
    )
    mid_stage = _stage_weights(
        stage_cfg,
        "mid",
        {"Q_qdot_mult": 2.0, "Q_qddot_mult": 2.0, "Q_du_mult": 2.0},
    )
    near_stage = _stage_weights(
        stage_cfg,
        "near",
        {"Q_qdot_mult": 0.0, "Q_qddot_mult": 0.0, "Q_du_mult": 0.0},
    )

    cost = {
        "Q_p": [float(item) for item in cost_cfg["Q_p"]],
        "Q_v": [float(item) for item in cost_cfg["Q_v"]],
        "R": float(cost_cfg["R"]),
        "Q_qdot": float(cost_cfg.get("Q_qdot", 0.0)),
        "Q_qddot": float(cost_cfg.get("Q_qddot", 0.0)),
        "Q_du": float(cost_cfg.get("Q_du", 0.0)),
        "stage_weights": {
            "far": far_stage,
            "mid": mid_stage,
            "near": near_stage,
        },
        "tube": {
            "window_half_ms": 50.0,
            "Q_p_tube": 500.0,
            "Q_v_tube": 0.0,
            "Q_n_tube": 0.0,
            "tube_cost_ratio": 1.0,
            "softmin_beta": 5.0,
            "corridor_radius": 0.12,
            "use_softmin_terminal": True,
        },
    }
    for key in ("Q_p", "Q_v", "R", "Q_qdot", "Q_qddot", "Q_du"):
        sources[f"cost.{key}"] = cfg_src.get(f"cost.{key}", "default.yaml")
    for key in ("window_half_ms", "Q_p_tube", "Q_v_tube", "Q_n_tube",
                "tube_cost_ratio", "softmin_beta", "corridor_radius",
                "use_softmin_terminal"):
        sources[f"cost.tube.{key}"] = "V12 字面量"
    for stage, values in cost["stage_weights"].items():
        for key in values:
            sources[f"cost.stage_weights.{stage}.{key}"] = cfg_src.get(
                f"stage_weights.{stage}.{key}", "V12 字面量"
            )

    mpc: dict[str, Any] = {
        "version": "v12",
        "dt": dt,
        "is_position_mode": False,
        "ablation_mode": "full",
        "solver_backend": "auto",
        "total_horizon": 200,
        "fixed_horizon": 40,
        "replan_interval": 30,
        "max_iter_per_plan": 5,
        "first_plan_iters": 15,
        "near_plan_iters": 20,
        "near_threshold": None,
        "Q_p_scale_far": 5.0,
        "Q_v_scale_far": 3.0,
        "Q_p_scale_near": 8.0,
        "Q_v_scale_near": 120.0,
        "normal_weight": 500000.0,
        "racket_speed": float(hitting_cfg.get("racket_speed", 1.8)),
        "target_speed": 1.8,
        "Q_tcp_soft": 0.0,
        "Q_qdot_limit": 0.0,
        "softmin_beta": 5.0,
        "follow_through_length": 0.0,
        "follow_through_steps": int(
            hitting_cfg.get("follow_through_steps", 160)
        ),
        "follow_through_v_terminal": float(
            hitting_cfg.get("follow_through_v_terminal", 0.3)
        ),
        "follow_trigger": "planned",
        "no_follow_through": False,
        "use_backswing": False,
        "backswing_offset": -0.6,
        "backswing_ratio": 0.35,
        "time_perturb_s": 0.0,
        "space_perturb_m": 0.0,
        "perturb_alpha_min": 0.0,
        "spatial_perturb_target": "hitpoint",
        "shared_input_full": False,
        "use_r_decay": True,
        "r_decay_ratio": 0.40,
        "fix_joint5_angle": None,
        "normal_flip": False,
        "smooth_far": copy.deepcopy(far_stage),
        "smooth_mid": copy.deepcopy(mid_stage),
        "smooth_near": copy.deepcopy(near_stage),
        "async_mode": False,
        "far_threshold": 50,
        "follow_kp": 200.0,
        "follow_kd": 20.0,
        "hit_lock_threshold": 60,
        "hard_margin_deg": 2.0,
        "warn_margin_deg": 5.0,
        "j1_warn_margin_deg": 8.0,
        "refiner_window_half": 15,
        "Q_p_base": None,
        "Q_v_base": None,
        "Q_qdot_base": None,
        "Q_qddot_base": None,
        "Q_du_base": None,
    }
    for key in mpc:
        if key in ("dt", "smooth_far", "smooth_mid", "smooth_near"):
            continue
        sources[f"mpc.{key}"] = "V12 字面量"
    sources["mpc.dt"] = "default.yaml"
    sources["mpc.racket_speed"] = cfg_src.get(
        "hitting.racket_speed", "V12 字面量"
    )
    sources["mpc.follow_through_steps"] = cfg_src.get(
        "hitting.follow_through_steps", "V12 字面量"
    )
    sources["mpc.follow_through_v_terminal"] = cfg_src.get(
        "hitting.follow_through_v_terminal", "V12 字面量"
    )

    # 服务箱自动参数：只在对应 CLI 字段未显式给出时生效。
    if bool(_arg(args, "serve_box")):
        auto_set = (
            ("horizon", "fixed_horizon", 120),
            ("iter", "max_iter_per_plan", 10),
            ("replan_interval", "replan_interval", 20),
            ("near_iters", "near_plan_iters", 5),
            ("first_plan_iters", "first_plan_iters", 30),
        )
        for dest, key, value in auto_set:
            if _is_explicit(args, dest, explicit):
                continue
            mpc[key] = value
            sources[f"mpc.{key}"] = "serve_box_auto"
        mpc["total_horizon"] = max(mpc["total_horizon"], 250)
        sources["mpc.total_horizon"] = "serve_box_auto"

    serve = {
        "mode": "target_box",
        "ball_speed": None,
        "ball_distance": None,
        "approach_angle": 0.0,
        "serve_distance": 8.0,
        "serve_height": 1.2,
        "serve_x_size": 8.0,
        "serve_y_size": 0.2,
        "serve_z_size": 0.3,
        "serve_box_center": np.array([0.0, -8.0, 1.2], dtype=np.float64),
        "serve_box_halfsize": np.array([4.0, 0.1, 0.15], dtype=np.float64),
        "speed_range": _SERVE_SPEED_RANGE,
        "hit_time_uniform": _HIT_TIME_UNIFORM,
        "ball_start_y_range": _BALL_START_Y_RANGE,
        "ball_start_z_range": _BALL_START_Z_RANGE,
        "ball_direction": _BALL_DIRECTION,
    }
    for key in serve:
        sources[f"serve.{key}"] = "V12 字面量"
    sources["serve.ball_speed"] = "cli 缺省(None)"

    source_table: dict[str, Any] = {
        "seed": None,
        "dt": dt,
        "actuator_mode": str(actuator_cfg.get("mode", "torque")),
        "actuator_feedforward_cfg": bool(
            actuator_cfg.get("feedforward", True)
        ),
        "actuator_kp_cfg": np.asarray(
            actuator_cfg.get(
                "kp", [200.0, 200.0, 200.0, 50.0, 50.0, 20.0]
            ),
            dtype=np.float64,
        ),
        "actuator_kd_cfg": np.asarray(
            actuator_cfg.get("kd", [20.0, 20.0, 5.0, 5.0, 5.0, 2.0]),
            dtype=np.float64,
        ),
        "kp_override": None,
        "kd_override": None,
        "feedforward_override": None,
        "tube_terminal_override": None,
        "mpc": mpc,
        "cost": cost,
        "solver": {**ilqt_cfg, "backend": "auto"},
        "perception": perception,
        "serve": serve,
        "hitting_config": hitting_cfg,
        "scene": {
            "use_bounce": True,
            "bounce_restitution": float(BOUNCE_RESTITUTION),
            "ball_gravity": np.array(
                ball_cfg.get("gravity", [0.0, 0.0, -9.81]),
                dtype=np.float64,
            ),
            "ball_radius": float(ball_cfg.get("radius", 0.033)),
            "ball_mass": float(ball_cfg.get("mass", 0.058)),
            "shoulder_pos": shoulder_pos,
        },
        "target_center": _TARGET_CENTER.copy(),
        "target_offset": _TARGET_OFFSET,
        "workspace_radius": float(WORKSPACE_RADIUS),
        "shoulder_pos": shoulder_pos,
        "ablation_mode": "full",
        "output_dir": None,
        "viewer": False,
        "no_plot": False,
        "robot_id": selection.robot_id,
        "robot_profile_path": selection.profile_path,
        "runtime_path": selection.runtime_path,
        "racket_path": selection.racket_path,
        "task_path": selection.task_path,
        "task_mode": selection.task_mode,
        "kp": None,
        "kd": None,
        "feedforward": None,
    }
    sources.update(
        {
            "actuator.mode": cfg_src.get("actuator.mode", "default.yaml"),
            "actuator.feedforward": cfg_src.get(
                "actuator.feedforward", "default.yaml"
            ),
            "actuator.kp": cfg_src.get("actuator.kp", "default.yaml"),
            "actuator.kd": cfg_src.get("actuator.kd", "default.yaml"),
            "hitting.racket_speed": cfg_src.get(
                "hitting.racket_speed", "V12 字面量"
            ),
            "hitting.hit_direction": cfg_src.get(
                "hitting.hit_direction", "V12 字面量"
            ),
            "target.center": "V12 字面量",
            "target.offset": "V12 字面量",
            "target.workspace_radius": "src/robot/constants.py",
            "target.shoulder_pos": shoulder_source,
            "scene.ball_gravity": cfg_src.get("ball.gravity", "default.yaml"),
            "scene.ball_radius": cfg_src.get("ball.radius", "default.yaml"),
            "scene.ball_mass": cfg_src.get("ball.mass", "default.yaml"),
            "scene.bounce_restitution": "src/robot/constants.py",
            "scene.use_bounce": "V12 字面量",
            "solver.backend": "V12 字面量",
            "perception.obs_freq": "缺省(0)",
            "perception.obs_noise_pos": "缺省(0)",
            "perception.obs_noise_vel": "缺省(0)",
            "perception.obs_use_kf": "缺省(False)",
            "mpc.near_threshold": "deferred(k_hit)",
            "seed": "缺省(None)",
            "output_dir": "缺省(results)",
        }
    )
    # brief 第 31 行：新路径显式提供 RacketProfile 后，旧 runtime 的球拍几何字段
    # 不再参与计算；本层只在来源报告中标记为 legacy-unused（不 import T1 实现）。
    if selection.racket_path is not None and selection.runtime_path is not None:
        for _racket_field in (
            "racket_position",
            "racket_quaternion",
            "racket_radius",
            "racket_half_thickness",
            "racket_mass",
        ):
            sources[f"runtime.scene.{_racket_field}"] = "legacy-unused(racket_profile)"
    return source_table, sources


# ---------------------------------------------------------------------------
# 覆盖应用：R1D 任务覆盖 / 显式任务覆盖
# ---------------------------------------------------------------------------

_TARGET_ALIASES: dict[str, str] = {
    "center": "target_center",
    "offset": "target_offset",
    "workspace_radius": "workspace_radius",
    "shoulder_pos": "shoulder_pos",
}

# 来球覆盖里少数键的真实字段在 scene 中（球—拍反弹语义），必须显式别名，
# 否则覆盖会落到 serve 的 stray 键而不生效（复审 K2）。
_SERVE_SCENE_ALIASES: dict[str, str] = {
    "serve.use_bounce": "use_bounce",
    "serve.bounce_restitution": "bounce_restitution",
}


def _set_value(v: dict[str, Any], path: str, value: Any) -> None:
    """按点路径写入解析草稿；路径语义与任务覆盖白名单一致。"""
    if path == "seed":
        v["seed"] = value
        return
    if path == "actuator.mode":
        v["actuator_mode"] = value
        return
    if path == "actuator.feedforward":
        v["feedforward_override"] = value
        return
    if path == "actuator.kp":
        v["kp_override"] = value
        return
    if path == "actuator.kd":
        v["kd_override"] = value
        return
    if path in _SERVE_SCENE_ALIASES:
        # 球—拍反弹字段的真实归属是 scene；source 由 _apply_overrides 同步。
        scene_key = _SERVE_SCENE_ALIASES[path]
        if scene_key == "use_bounce":
            v["scene"][scene_key] = bool(value)
        else:
            v["scene"][scene_key] = float(value)
        return
    if path == "cost.tube.use_softmin_terminal":
        # 该键的真实生效点是消融派生之后，先暂存并记录来源，再在 _apply_cli 覆盖。
        v["tube_terminal_override"] = bool(value)
        return
    if path.startswith("target."):
        v[_TARGET_ALIASES[path.split(".", 1)[1]]] = value
        return
    if path.startswith("hitting."):
        v["hitting_config"][path.split(".", 1)[1]] = value
        return
    node: Any = v
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def _apply_overrides(
    v: dict[str, Any],
    sources: dict[str, str],
    overrides: Mapping[str, Any],
    origin: str,
) -> None:
    """应用一组已校验的点路径覆盖并记录来源标签。"""
    for path, value in overrides.items():
        _set_value(v, path, value)
        sources[path] = origin
        scene_key = _SERVE_SCENE_ALIASES.get(path)
        if scene_key is not None:
            sources[f"scene.{scene_key}"] = origin


# ---------------------------------------------------------------------------
# 显式 CLI 覆盖（最高优先，但不吞服务箱自动值）
# ---------------------------------------------------------------------------


def _apply_cli(
    v: dict[str, Any],
    sources: dict[str, str],
    args: argparse.Namespace,
    explicit: frozenset[str],
    selection: _RobotSelection,
) -> None:
    """把显式 CLI 旗标叠加到解析草稿（最高优先级）。"""
    mpc: dict[str, Any] = v["mpc"]
    cost: dict[str, Any] = v["cost"]
    serve: dict[str, Any] = v["serve"]
    scene: dict[str, Any] = v["scene"]
    perception: dict[str, Any] = v["perception"]

    seed = _arg(args, "seed")
    if _is_explicit(args, "seed", explicit) or seed is not None:
        v["seed"] = seed
        sources["seed"] = "cli:--seed"

    # 输出与可视化
    output = _arg(args, "output")
    if output is None:
        default_output = (
            (_REPO_ROOT / "results" / "r1d_sim").resolve()
            if selection.robot_id == "unitree_r1d"
            else (_REPO_ROOT / "results").resolve()
        )
        v["output_dir"] = default_output
        sources["output_dir"] = "缺省(results)"
    else:
        v["output_dir"] = Path(str(output)).expanduser().resolve()
        sources["output_dir"] = "cli:--output"
    v["viewer"] = bool(_arg(args, "viewer"))
    v["no_plot"] = bool(_arg(args, "no_plot"))

    # 消融模式
    ablation_raw = _arg(args, "ablation")
    if ablation_raw is not None:
        v["ablation_mode"] = str(ablation_raw)
        sources["ablation_mode"] = "cli:--ablation"
    else:
        no_tube = bool(_arg(args, "no_tube"))
        no_softmin = bool(_arg(args, "no_softmin"))
        if no_tube and no_softmin:
            v["ablation_mode"] = "none"
            sources["ablation_mode"] = "派生(--no-tube/--no-softmin)"
        elif no_tube:
            v["ablation_mode"] = "softmin_only"
            sources["ablation_mode"] = "派生(--no-tube)"
        elif no_softmin:
            v["ablation_mode"] = "tube_only"
            sources["ablation_mode"] = "派生(--no-softmin)"
    if v["ablation_mode"] not in (
        "full",
        "tube_only",
        "softmin_only",
        "none",
        "select_one",
    ):
        raise ValueError(f"消融模式非法: {v['ablation_mode']!r}")
    # 消融推导 use_softmin_terminal；显式任务覆盖必须在推导之后应用，
    # 否则会被无条件派生覆盖而静默失效（复审 K2）。
    terminal_override_source = sources.get("cost.tube.use_softmin_terminal")
    cost["tube"]["use_softmin_terminal"] = v["ablation_mode"] in (
        "full",
        "softmin_only",
    )
    sources["cost.tube.use_softmin_terminal"] = "派生(ablation_mode)"
    if v["tube_terminal_override"] is not None:
        cost["tube"]["use_softmin_terminal"] = bool(
            v["tube_terminal_override"]
        )
        sources["cost.tube.use_softmin_terminal"] = (
            terminal_override_source or "task_overrides"
        )

    # 求解器后端
    if _is_explicit(args, "solver_backend", explicit):
        v["solver"]["backend"] = str(_arg(args, "solver_backend"))
        sources["solver.backend"] = "cli:--solver-backend"

    # 时间参数（显式值不得被服务箱自动值吞掉）
    for dest, key in (
        ("horizon", "fixed_horizon"),
        ("iter", "max_iter_per_plan"),
        ("replan_interval", "replan_interval"),
        ("near_iters", "near_plan_iters"),
        ("first_plan_iters", "first_plan_iters"),
    ):
        if _is_explicit(args, dest, explicit):
            mpc[key] = int(_arg(args, dest))
            sources[f"mpc.{key}"] = f"cli:{_flag(dest)}"
    for dest, key in (
        ("Qp_scale_far", "Q_p_scale_far"),
        ("Qv_scale_far", "Q_v_scale_far"),
        ("Qp_scale_near", "Q_p_scale_near"),
        ("Qv_scale_near", "Q_v_scale_near"),
    ):
        if _is_explicit(args, dest, explicit):
            mpc[key] = float(_arg(args, dest))
            sources[f"mpc.{key}"] = f"cli:{_flag(dest)}"
    for dest, key in (
        ("near_threshold", "near_threshold"),
        ("far_threshold", "far_threshold"),
    ):
        if _is_explicit(args, dest, explicit):
            mpc[key] = int(_arg(args, dest))
            sources[f"mpc.{key}"] = f"cli:{_flag(dest)}"
    for dest, key in (
        ("normal_weight", "normal_weight"),
        ("Q_tcp_soft", "Q_tcp_soft"),
        ("Q_qdot_limit", "Q_qdot_limit"),
        ("target_speed", "target_speed"),
        ("follow_kp", "follow_kp"),
        ("follow_kd", "follow_kd"),
        ("hard_margin", "hard_margin_deg"),
        ("warn_margin", "warn_margin_deg"),
        ("j1_warn", "j1_warn_margin_deg"),
        ("hit_shift", "follow_through_length"),
    ):
        if _is_explicit(args, dest, explicit):
            mpc[key] = float(_arg(args, dest))
            sources[f"mpc.{key}"] = f"cli:{_flag(dest)}"
    if _is_explicit(args, "softmin_beta", explicit):
        # 唯一真实消费者是 TubeConfig.softmin_beta；写 cost.tube，mpc 字段在
        # _finalize 同步为同值镜像（复审 K1）。
        cost["tube"]["softmin_beta"] = float(_arg(args, "softmin_beta"))
        sources["cost.tube.softmin_beta"] = "cli:--softmin-beta"
    if _is_explicit(args, "hit_lock", explicit):
        mpc["hit_lock_threshold"] = int(_arg(args, "hit_lock"))
        sources["mpc.hit_lock_threshold"] = "cli:--hit-lock"
    if _is_explicit(args, "refiner_window", explicit):
        mpc["refiner_window_half"] = int(_arg(args, "refiner_window"))
        sources["mpc.refiner_window_half"] = "cli:--refiner-window"
    if _is_explicit(args, "window_ms", explicit):
        cost["tube"]["window_half_ms"] = float(_arg(args, "window_ms"))
        sources["cost.tube.window_half_ms"] = "cli:--window-ms"
    if _is_explicit(args, "corridor_radius", explicit):
        cost["tube"]["corridor_radius"] = float(_arg(args, "corridor_radius"))
        sources["cost.tube.corridor_radius"] = "cli:--corridor-radius"

    # 后摆 / R 退火 / 随挥 / 扰动
    if bool(_arg(args, "use_backswing")):
        mpc["use_backswing"] = True
        sources["mpc.use_backswing"] = "cli:--use-backswing"
    for dest, key in (
        ("backswing", "backswing_offset"),
        ("bs_ratio", "backswing_ratio"),
        ("r_decay", "r_decay_ratio"),
    ):
        if _is_explicit(args, dest, explicit):
            value = float(_arg(args, dest))
            mpc[key] = -abs(value) if key == "backswing_offset" else value
            sources[f"mpc.{key}"] = f"cli:{_flag(dest)}"
    if bool(_arg(args, "no_r_decay")):
        mpc["use_r_decay"] = False
        sources["mpc.use_r_decay"] = "cli:--no-r-decay"
    if bool(_arg(args, "no_follow_through")):
        mpc["no_follow_through"] = True
        mpc["follow_through_length"] = 0.0
        mpc["follow_through_steps"] = 0
        mpc["follow_through_v_terminal"] = 0.0
        v["hitting_config"]["follow_through_steps"] = 0
        v["hitting_config"]["follow_through_length"] = 0.0
        v["hitting_config"]["follow_through_v_terminal"] = 0.0
        for key in (
            "follow_through_length",
            "follow_through_steps",
            "follow_through_v_terminal",
        ):
            sources[f"mpc.{key}"] = "cli:--no-follow-through"
    if _is_explicit(args, "follow_trigger", explicit):
        mpc["follow_trigger"] = str(_arg(args, "follow_trigger"))
        sources["mpc.follow_trigger"] = "cli:--follow-trigger"
    if bool(_arg(args, "normal_flip")):
        mpc["normal_flip"] = True
        sources["mpc.normal_flip"] = "cli:--normal-flip"
    if bool(_arg(args, "shared_input_full")):
        mpc["shared_input_full"] = True
        sources["mpc.shared_input_full"] = "cli:--shared-input-full"

    # 扰动：显式或 random_perturb 时按 V12 原逻辑解析（含 seed+99999 子随机源）
    perturb_dests = (
        "time_perturb_ms",
        "space_perturb_m",
        "perturb_alpha_min",
        "time_perturb_min_ms",
        "space_perturb_min_m",
        "perturb_sign",
        "spatial_perturb_target",
    )
    perturb_explicit = bool(_arg(args, "random_perturb")) or any(
        _is_explicit(args, dest, explicit) for dest in perturb_dests
    )
    if perturb_explicit:
        time_perturb_s = float(_arg(args, "time_perturb_ms")) / 1000.0
        space_perturb_m = float(_arg(args, "space_perturb_m"))
        if bool(_arg(args, "random_perturb")):
            seed = _arg(args, "seed")
            rng = np.random.default_rng(
                None if seed is None else int(seed) + 99999
            )
            tp_max = abs(float(_arg(args, "time_perturb_ms")))
            tp_min = abs(float(_arg(args, "time_perturb_min_ms")))
            if tp_max > 1e-9:
                tp_abs = rng.uniform(tp_min, tp_max)
                tp_sign = {"positive": 1.0, "negative": -1.0}.get(
                    str(_arg(args, "perturb_sign")),
                    float(rng.choice([-1.0, 1.0])),
                )
                time_perturb_s = tp_sign * tp_abs / 1000.0
            sp_max = abs(float(_arg(args, "space_perturb_m")))
            sp_min = abs(float(_arg(args, "space_perturb_min_m")))
            if sp_max > 1e-9:
                sp_abs = rng.uniform(sp_min, sp_max)
                sp_sign = {"positive": 1.0, "negative": -1.0}.get(
                    str(_arg(args, "perturb_sign")),
                    float(rng.choice([-1.0, 1.0])),
                )
                space_perturb_m = sp_sign * sp_abs
            sources["mpc.time_perturb_s"] = "cli:--random-perturb"
            sources["mpc.space_perturb_m"] = "cli:--random-perturb"
        else:
            sources["mpc.time_perturb_s"] = "cli:--time-perturb-ms"
            sources["mpc.space_perturb_m"] = "cli:--space-perturb-m"
        mpc["time_perturb_s"] = time_perturb_s
        mpc["space_perturb_m"] = space_perturb_m
        mpc["perturb_alpha_min"] = float(_arg(args, "perturb_alpha_min"))
        sources["mpc.perturb_alpha_min"] = "cli:--perturb-alpha-min"
        mpc["spatial_perturb_target"] = str(
            _arg(args, "spatial_perturb_target")
        )
        sources["mpc.spatial_perturb_target"] = "cli:--spatial-perturb-target"

    # 执行器与增益
    if bool(_arg(args, "position_mode")):
        v["actuator_mode"] = "position"
        sources["actuator.mode"] = "cli:--position-mode"
    for dest, key in (("kp", "kp_override"), ("kd", "kd_override")):
        raw_gain = _arg(args, dest)
        if _is_explicit(args, dest, explicit) and raw_gain is not None:
            v[key] = [float(item) for item in raw_gain]
            sources[dest] = f"cli:{_flag(dest)}"
    if bool(_arg(args, "no_feedforward")):
        v["feedforward_override"] = False
        sources["actuator.feedforward"] = "cli:--no-feedforward"
    if selection.robot_id == "rm65" and bool(_arg(args, "fix_joint5")):
        init_q = (
            INIT_Q_REAL
            if _arg(args, "limits_config") is not None
            else INIT_Q
        )
        mpc["fix_joint5_angle"] = float(init_q[5])
        sources["mpc.fix_joint5_angle"] = "cli:--fix-joint5"
    if bool(_arg(args, "async_replan")):
        mpc["async_mode"] = True
        sources["mpc.async_mode"] = "cli:--async-replan"

    # 观测门控
    for dest, key in (
        ("obs_freq", "obs_freq"),
        ("obs_noise_pos", "obs_noise_pos"),
        ("obs_noise_vel", "obs_noise_vel"),
    ):
        if _is_explicit(args, dest, explicit):
            perception[key] = float(_arg(args, dest))
            sources[f"perception.{key}"] = f"cli:{_flag(dest)}"
    if bool(_arg(args, "obs_use_kf")):
        perception["obs_use_kf"] = True
        sources["perception.obs_use_kf"] = "cli:--obs-use-kf"

    # 来球
    if bool(_arg(args, "serve_box")):
        serve["mode"] = "serve_box"
        sources["serve.mode"] = "cli:--serve-box"
    for dest, key in (
        ("ball_speed", "ball_speed"),
        ("ball_distance", "ball_distance"),
        ("approach_angle", "approach_angle"),
        ("serve_distance", "serve_distance"),
        ("serve_height", "serve_height"),
        ("serve_x_size", "serve_x_size"),
        ("serve_y_size", "serve_y_size"),
        ("serve_z_size", "serve_z_size"),
    ):
        if not _is_explicit(args, dest, explicit):
            continue
        value = _arg(args, dest)
        if value is None:
            continue
        serve[key] = float(value)
        sources[f"serve.{key}"] = f"cli:{_flag(dest)}"
    if bool(_arg(args, "no_bounce")):
        scene["use_bounce"] = False
        sources["scene.use_bounce"] = "cli:--no-bounce"


# ---------------------------------------------------------------------------
# 收尾：代价基底、执行器解析、派生量
# ---------------------------------------------------------------------------


def _resolve_gain(
    name: str,
    base: Any,
    base_source: str,
    override: Any,
    ndof: int,
    sources: dict[str, str],
) -> np.ndarray | None:
    """解析 PD 增益：1 维广播到主关节数，长度冲突运行前失败。"""
    value = override if override is not None else base
    if value is None:
        sources[name] = base_source
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size not in (1, ndof):
        raise ValueError(
            f"{name} 长度={array.size} 非法：机器人主关节数为 {ndof}，"
            f"只接受 1（广播）或 {ndof}"
        )
    if array.size == 1:
        array = np.full(ndof, float(array[0]), dtype=np.float64)
    sources[name] = (
        sources.get(name, "cli:" + _flag(name))
        if override is not None
        else base_source
    )
    return array


def _finalize(
    v: dict[str, Any],
    sources: dict[str, str],
    selection: _RobotSelection,
) -> None:
    """计算派生量并解析执行器/增益/门控最终值。"""
    mpc: dict[str, Any] = v["mpc"]
    cost: dict[str, Any] = v["cost"]
    serve: dict[str, Any] = v["serve"]
    perception: dict[str, Any] = v["perception"]

    # 代价向量与基底（V12 语义：Q_p_base = 2 × Q_p，Q_v_base = 2 × Q_v）。
    cost["Q_p"] = [float(item) for item in cost["Q_p"]]
    cost["Q_v"] = [float(item) for item in cost["Q_v"]]
    if len(cost["Q_p"]) != 3 or len(cost["Q_v"]) != 3:
        raise ValueError("cost.Q_p/cost.Q_v 必须为 3 维")
    for key in ("R", "Q_qdot", "Q_qddot", "Q_du"):
        cost[key] = float(cost[key])
    mpc["Q_p_base"] = np.array(cost["Q_p"], dtype=np.float64) * 2.0
    sources["mpc.Q_p_base"] = sources.get("cost.Q_p", "default.yaml")
    mpc["Q_v_base"] = np.array(cost["Q_v"], dtype=np.float64) * 2.0
    sources["mpc.Q_v_base"] = sources.get("cost.Q_v", "default.yaml")
    mpc["R"] = float(cost["R"])
    sources["mpc.R"] = sources.get("cost.R", "default.yaml")
    mpc["Q_qdot_base"] = float(cost["Q_qdot"])
    mpc["Q_qddot_base"] = float(cost["Q_qddot"])
    mpc["Q_du_base"] = float(cost["Q_du"])
    sources["mpc.Q_qdot_base"] = sources.get("cost.Q_qdot", "default.yaml")
    sources["mpc.Q_qddot_base"] = sources.get("cost.Q_qddot", "default.yaml")
    sources["mpc.Q_du_base"] = sources.get("cost.Q_du", "default.yaml")
    mpc["smooth_far"] = copy.deepcopy(cost["stage_weights"]["far"])
    mpc["smooth_mid"] = copy.deepcopy(cost["stage_weights"]["mid"])
    mpc["smooth_near"] = copy.deepcopy(cost["stage_weights"]["near"])
    for stage in ("far", "mid", "near"):
        sources[f"mpc.smooth_{stage}"] = sources.get(
            f"cost.stage_weights.{stage}.Q_qdot_mult", "V12 字面量"
        )

    # softmin_beta 双源收敛：唯一真实消费者是 TubeConfig.softmin_beta
    # （cost_config["tube"]），mpc_config 的同名字段只是镜像，二者必须同值。
    tube_beta_source = sources.get("cost.tube.softmin_beta", "V12 字面量")
    mpc_beta_source = sources.get("mpc.softmin_beta", "V12 字面量")
    if tube_beta_source != "V12 字面量":
        softmin_beta = float(cost["tube"]["softmin_beta"])
        softmin_beta_source = tube_beta_source
    elif mpc_beta_source != "V12 字面量":
        softmin_beta = float(mpc["softmin_beta"])
        softmin_beta_source = mpc_beta_source
    else:
        softmin_beta = 5.0
        softmin_beta_source = "V12 字面量"
    cost["tube"]["softmin_beta"] = softmin_beta
    mpc["softmin_beta"] = softmin_beta
    sources["cost.tube.softmin_beta"] = softmin_beta_source
    sources["mpc.softmin_beta"] = softmin_beta_source

    # 时间步与来源
    mpc["dt"] = float(v["dt"])
    sources["mpc.dt"] = "default.yaml"

    # 执行器：torque 下不得把 RM65 六维 PD 增益泄漏给 R1D；position 下
    # 解析 kp/kd（R1D 优先 runtime，RM65 用 default.yaml），并强制关闭 R 退火。
    mode = str(v["actuator_mode"])
    if mode not in ("torque", "position"):
        raise ValueError(f"执行器模式非法: {mode!r}")
    ndof = selection.ndof
    if mode == "position":
        if selection.robot_id == "unitree_r1d":
            if selection.runtime_params is not None:
                base_kp = selection.runtime_params.kp
                base_kd = selection.runtime_params.kd
                base_source = f"runtime:{selection.runtime_path.name}"
            else:
                base_kp = None
                base_kd = None
                base_source = "deferred(T5 从 runtime 派生)"
        else:
            base_kp = v["actuator_kp_cfg"]
            base_kd = v["actuator_kd_cfg"]
            base_source = sources.get("actuator.kp", "default.yaml")
        v["kp"] = _resolve_gain(
            "kp", base_kp, base_source, v["kp_override"], ndof, sources
        )
        v["kd"] = _resolve_gain(
            "kd", base_kd, base_source, v["kd_override"], ndof, sources
        )
        feedforward = (
            v["feedforward_override"]
            if v["feedforward_override"] is not None
            else bool(v["actuator_feedforward_cfg"])
        )
        v["feedforward"] = bool(feedforward)
        sources.setdefault(
            "feedforward", sources.get("actuator.feedforward", "default.yaml")
        )
        mpc["use_r_decay"] = False
        sources["mpc.use_r_decay"] = "position_mode_rule"
        mpc["is_position_mode"] = True
    else:
        v["kp"] = None
        v["kd"] = None
        v["feedforward"] = None
        sources["kp"] = "not_consumed(torque)"
        sources["kd"] = "not_consumed(torque)"
        sources["feedforward"] = "not_consumed(torque)"
        mpc["is_position_mode"] = False

    # 消融/求解器/随挥零值等派生字段
    mpc["ablation_mode"] = str(v["ablation_mode"])
    mpc["solver_backend"] = str(v["solver"]["backend"])
    sources["mpc.ablation_mode"] = sources.get("ablation_mode", "派生")
    sources["mpc.solver_backend"] = sources.get(
        "solver.backend", "V12 字面量"
    )
    if mpc["near_threshold"] is None:
        sources["mpc.near_threshold"] = "deferred(k_hit)"
    if mpc["fix_joint5_angle"] is not None:
        mpc["fix_joint5_angle"] = float(mpc["fix_joint5_angle"])
    if bool(mpc["no_follow_through"]):
        mpc["follow_through_length"] = 0.0
        mpc["follow_through_steps"] = 0
        mpc["follow_through_v_terminal"] = 0.0

    # 击球字段双源收敛：权威来源是 mpc_config；hitting_config 只是镜像，
    # 防止 --hit-shift / --task-overrides mpc.* 只改一处而 T5 读到旧值。
    hitting = v["hitting_config"]
    hitting["racket_speed"] = float(mpc["racket_speed"])
    hitting["follow_through_length"] = float(mpc["follow_through_length"])
    hitting["follow_through_steps"] = int(mpc["follow_through_steps"])
    hitting["follow_through_v_terminal"] = float(
        mpc["follow_through_v_terminal"]
    )
    for key in (
        "racket_speed",
        "follow_through_length",
        "follow_through_steps",
        "follow_through_v_terminal",
    ):
        sources[f"hitting.{key}"] = sources.get(
            f"mpc.{key}", "V12 字面量"
        )

    # 发球箱派生几何（供 T5 直接传给 generate_ball_from_serve_box）。
    # 来源标签必须反映实际参与派生的分量来源，不能无条件写“V12 字面量”
    # （例如 --serve-distance 6 后 center 变了，标签也要随之改变）。
    serve_distance = float(serve["serve_distance"])
    serve_height = float(serve["serve_height"])
    serve["serve_box_center"] = np.array(
        [0.0, -serve_distance, serve_height], dtype=np.float64
    )
    serve["serve_box_halfsize"] = np.array(
        [
            float(serve["serve_x_size"]) / 2.0,
            float(serve["serve_y_size"]) / 2.0,
            float(serve["serve_z_size"]) / 2.0,
        ],
        dtype=np.float64,
    )
    center_sources = (
        sources.get("serve.serve_distance", "V12 字面量"),
        sources.get("serve.serve_height", "V12 字面量"),
    )
    halfsize_sources = (
        sources.get("serve.serve_x_size", "V12 字面量"),
        sources.get("serve.serve_y_size", "V12 字面量"),
        sources.get("serve.serve_z_size", "V12 字面量"),
    )
    sources["serve.serve_box_center"] = (
        "V12 字面量"
        if all(src == "V12 字面量" for src in center_sources)
        else (
            "派生(serve_distance="
            f"{center_sources[0]}, serve_height={center_sources[1]})"
        )
    )
    sources["serve.serve_box_halfsize"] = (
        "V12 字面量"
        if all(src == "V12 字面量" for src in halfsize_sources)
        else (
            "派生(serve_x_size="
            f"{halfsize_sources[0]}, serve_y_size={halfsize_sources[1]}, "
            f"serve_z_size={halfsize_sources[2]})"
        )
    )

    # 观测门控有效频率（V12 原逻辑）
    obs_freq = float(perception["obs_freq"])
    noise_pos = float(perception["obs_noise_pos"])
    noise_vel = float(perception["obs_noise_vel"])
    dt = float(v["dt"])
    effective = (
        obs_freq
        if obs_freq > 0.0
        else (1.0 / dt if (noise_pos > 0.0 or noise_vel > 0.0) else 0.0)
    )
    perception["obs_freq_effective"] = effective
    perception["noise_enabled"] = bool(noise_pos > 0.0 or noise_vel > 0.0)
    perception["use_kf"] = bool(perception["obs_use_kf"])
    perception["kf_enabled"] = bool(perception["obs_use_kf"] and effective > 0.0)
    perception["obs_gate_enabled"] = bool(effective > 0.0)

    # 目标区域与场景标量校验
    v["target_center"] = np.asarray(v["target_center"], dtype=np.float64)
    if v["target_center"].shape != (3,):
        raise ValueError("target.center 必须为 3 维")
    if v["shoulder_pos"] is not None:
        shoulder = np.asarray(v["shoulder_pos"], dtype=np.float64)
        if shoulder.shape != (3,):
            raise ValueError("target.shoulder_pos 必须为 3 维")
        v["shoulder_pos"] = shoulder
    v["target_offset"] = float(v["target_offset"])
    v["workspace_radius"] = float(v["workspace_radius"])
    if v["workspace_radius"] <= 0.0:
        raise ValueError("target.workspace_radius 必须大于 0")
    v["scene"]["shoulder_pos"] = v["shoulder_pos"]


# ---------------------------------------------------------------------------
# 公开数据结构
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 唯一入口
# ---------------------------------------------------------------------------


def resolve_v12_config(
    args: argparse.Namespace | None = None,
    explicit_options: Iterable[str] | None = None,
) -> ResolvedV12Config:
    """解析 V12 唯一任务配置。

    优先级（低 → 高）：

    1. ``default.yaml`` → ``v5_active_hit.yaml`` → ``mpc.yaml`` 合并；
    2. V12 脚本字面量默认（horizon/iter/阈值/目标区域等）；
    3. 服务箱自动值（仅未显式给出的时间字段）；
    4. R1D 机器人专用覆盖（``configs/r1d_v12.yaml``）；
    5. 显式任务覆盖（``--task-overrides`` 映射）；
    6. 显式 CLI 专用旗标（``--horizon`` / ``--position-mode`` 等）。

    Args:
        args: CLI 参数命名空间；缺失字段按 V12 parser 缺省处理。
        explicit_options: 用户显式给出的选项（``--flag`` 或 dest 均可；
            解析器会归一为小写 dest 集合）。

    Returns:
        深拷贝隔离的 :class:`ResolvedV12Config`（数组只读、dict 为副本）。

    Raises:
        ValueError: 未知 robot、错配 runtime、非法覆盖、6/7 维向量冲突、
            固定球兼容模式与来球参数冲突等。
        FileNotFoundError: 显式给出的配置/profile 文件不存在。
    """
    namespace = argparse.Namespace() if args is None else args
    explicit = normalize_explicit_options(explicit_options)

    selection = _select_robot(namespace, explicit)
    values, sources = _build_base_values(selection, namespace, explicit)
    sources.update(selection.path_sources)

    if selection.overlay:
        _apply_overrides(
            values,
            sources,
            selection.overlay,
            selection.overlay_origin or "r1d_v12.yaml",
        )

    raw_overrides = _arg(namespace, "task_overrides")
    overrides = (
        _validate_overrides(raw_overrides, origin="task_overrides")
        if raw_overrides is not None
        else {}
    )
    if overrides:
        if selection.task_mode == "r1d_fixed_ball_compat":
            for key in overrides:
                if key.startswith("serve.") or key.startswith("target."):
                    raise ValueError(
                        "固定球兼容模式不允许任务覆盖 "
                        f"{key}：来球/目标由旧 sim-task 文件决定"
                    )
        _apply_overrides(values, sources, overrides, "task_overrides")

    _apply_cli(values, sources, namespace, explicit, selection)
    _finalize(values, sources, selection)

    return ResolvedV12Config(
        robot_id=str(values["robot_id"]),
        task_mode=str(values["task_mode"]),
        robot_profile_path=selection.profile_path,
        runtime_path=selection.runtime_path,
        racket_path=selection.racket_path,
        task_path=selection.task_path,
        dt=float(values["dt"]),
        seed=values["seed"],
        actuator_mode=str(values["actuator_mode"]),
        feedforward=values["feedforward"],
        kp=values["kp"],
        kd=values["kd"],
        ablation_mode=str(values["ablation_mode"]),
        solver_backend=str(values["solver"]["backend"]),
        solver_config=_ConfigDict(values["solver"]),
        mpc_config=_ConfigDict(values["mpc"]),
        cost_config=values["cost"],
        perception_config=values["perception"],
        hitting_config=values["hitting_config"],
        scene_config=values["scene"],
        serve_config=values["serve"],
        target_center=values["target_center"],
        target_offset=float(values["target_offset"]),
        workspace_radius=float(values["workspace_radius"]),
        output_dir=Path(values["output_dir"]),
        viewer=bool(values["viewer"]),
        no_plot=bool(values["no_plot"]),
        sources=dict(sources),
        explicit_options=explicit,
    )
