"""V12 配置解析：CLI 缺省值、配置来源合并与基线值构建。

从 ``src/sim/v12_config.py`` 迁入的 V12 字面量默认、``default.yaml`` →
``v5_active_hit.yaml`` → ``mpc.yaml`` 来源合并、服务箱自动参数与任务覆盖
点路径写入。本模块不 import 上层 ``resolution`` 模块；配置目录与仓库根由
调用方显式传入，保证内部依赖单向。
"""

from __future__ import annotations

import argparse
import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.robot.constants import (
    BOUNCE_RESTITUTION,
    SHOULDER_POS,
    WORKSPACE_RADIUS,
)
from src.utils.yaml_utils import load_config, merge_configs

if TYPE_CHECKING:
    # 仅用于类型标注，避免 ``selection -> defaults`` 与 ``defaults -> selection``
    # 形成运行期循环 import。
    from src.sim.configuration.selection import _RobotSelection



# V12 脚本中的目标区域字面量（原 main() 内硬编码，保持逐位一致）。
_TARGET_CENTER = np.array([-0.82765693, -0.47411682, 0.86947444], dtype=np.float64)
_TARGET_OFFSET = 0.10
# 发球区字面量：中心 (0, -serve_distance, serve_height)，全尺寸 8 × 0.2 × 0.3。
_SERVE_SPEED_RANGE = (8.0, 18.0)
_HIT_TIME_UNIFORM = (0.3, 0.4)
_BALL_START_Y_RANGE = (-5.5, -4.5)
_BALL_START_Z_RANGE = (1.4, 1.8)
_BALL_DIRECTION = "y"

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


def _load_config_with_sources(
    configs_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """按 V12 顺序合并 default/v5/mpc 并记录每个叶子来源。"""
    merged = load_config(configs_dir / "default.yaml")
    sources: dict[str, str] = {}
    _collect_config_sources(merged, "", "default.yaml", sources)
    for filename in ("v5_active_hit.yaml", "mpc.yaml"):
        path = configs_dir / filename
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
    configs_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    """构建解析基线（不含 R1D 覆盖层与显式 CLI 覆盖）。"""
    cfg, cfg_src = _load_config_with_sources(configs_dir)
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
