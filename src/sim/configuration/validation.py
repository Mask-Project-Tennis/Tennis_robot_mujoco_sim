"""V12 配置解析：任务覆盖白名单与类型校验。

从 ``src/sim/v12_config.py`` 迁入的 ``_v_*`` 校验器、``_OVERRIDE_VALIDATORS``
注册表，以及任务覆盖映射的展开与校验。本模块只依赖标准库与 numpy，
不读取配置文件、不 import 上层解析模块（单向依赖的最低层之一）。

校验器沿用迁移前的 ``ValueError`` 口径（异常类型与消息逐字不变），
因此对 Ruff 的 TRY004 建议做局部抑制。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 任务覆盖白名单与类型校验
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float:
    """把值转换为有限 float；拒绝布尔与字符串（防止 YAML 隐式类型混淆）。"""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"必须是数值，实际为 {type(value).__name__}")  # noqa: TRY004
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
        raise ValueError("必须是整数")  # noqa: TRY004
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
        raise ValueError("必须是布尔值")  # noqa: TRY004
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
        raise ValueError("必须是长度为 3 的数值序列")  # noqa: TRY004
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
        raise ValueError("必须是数值序列")  # noqa: TRY004
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


def allowed_override_keys() -> frozenset[str]:
    """返回 V12 task override 白名单的键集合（只读副本）。"""
    return frozenset(_OVERRIDE_VALIDATORS)


def _flatten_overrides(value: Any, prefix: str = "") -> dict[str, Any]:
    """把（可能嵌套的）任务覆盖映射展开为点路径扁平字典。"""
    if not isinstance(value, Mapping):
        raise ValueError("非法任务覆盖键：任务覆盖必须是映射")  # noqa: TRY004
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
                "src/sim/configuration/validation.py::allowed_override_keys）"
            )
        try:
            validated[key] = validator(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"任务覆盖值类型非法: {key}={value!r}（来源 {origin}）：{exc}"
            ) from exc
    return validated
