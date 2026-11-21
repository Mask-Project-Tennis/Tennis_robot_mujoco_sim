"""V12 配置解析：收尾派生与唯一入口。

从 ``src/sim/v12_config.py`` 迁入的仓库根/配置目录常量、``_finalize`` 与
``resolve_v12_config``。本模块是配置解析层的最上层，可 import 同包内其余
解析模块；反向依赖禁止。
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from src.sim.configuration.defaults import _arg, _build_base_values
from src.sim.configuration.overrides import (
    _apply_cli,
    _apply_overrides,
    _resolve_gain,
    normalize_explicit_options,
)
from src.sim.configuration.selection import _RobotSelection, _select_robot
from src.sim.configuration.types import ResolvedV12Config, _ConfigDict
from src.sim.configuration.validation import _validate_overrides

# 仓库根目录与配置目录：本文件位于 ``src/sim/configuration/``，向上三级才是
# 仓库根（``configuration -> sim -> src -> repo``）。旧 ``v12_config.py`` 位于
# ``src/sim/``，当时用 ``parents[2]``；迁入本包后必须修正为 ``parents[3]``，
# 否则会错误地落到 ``src/`` 并导致 ``configs/`` 路径失效。
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIGS_DIR = _REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# 收尾：代价基底、执行器解析、派生量
# ---------------------------------------------------------------------------



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
    values, sources = _build_base_values(
        selection, namespace, explicit, _CONFIGS_DIR
    )
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
                if key.startswith(("serve.", "target.")):
                    raise ValueError(
                        "固定球兼容模式不允许任务覆盖 "
                        f"{key}：来球/目标由旧 sim-task 文件决定"
                    )
        _apply_overrides(values, sources, overrides, "task_overrides")

    _apply_cli(values, sources, namespace, explicit, selection, _REPO_ROOT)
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
