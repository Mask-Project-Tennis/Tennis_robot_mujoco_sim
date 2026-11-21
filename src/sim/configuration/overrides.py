"""V12 配置解析：显式选项归一化与覆盖应用。

从 ``src/sim/v12_config.py`` 迁入的 ``normalize_explicit_options``、任务
覆盖写入、显式 CLI 覆盖与 PD 增益解析。任务覆盖的展开与校验位于
:mod:`src.sim.configuration.validation`，本模块只负责应用。
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.robot.constants import INIT_Q, INIT_Q_REAL
from src.sim.configuration.defaults import (
    _SERVE_SCENE_ALIASES,
    _arg,
    _flag,
    _is_explicit,
    _set_value,
)
from src.sim.configuration.selection import _RobotSelection


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


# ---------------------------------------------------------------------------
# 任务覆盖应用
# ---------------------------------------------------------------------------




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
    repo_root: Path,
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
            (repo_root / "results" / "r1d_sim").resolve()
            if selection.robot_id == "unitree_r1d"
            else (repo_root / "results").resolve()
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
# PD 增益解析
# ---------------------------------------------------------------------------


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
