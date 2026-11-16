"""R1D 仿真回合运行器 — 配置入口组装 + 可追踪产物（阶段 3 · Task 3.10）。

职责（施工计划 §共享接口契约"运行结果"条目，逐字实现）：
- ``run_r1d_episode(profile_path, runtime_path, task_path, output_dir)``：
  加载三份 YAML（profile/runtime/task）→ 严格仿真预检 → 组装环境对
  （3.4）+ 限位/碰撞保护（3.7 推荐配方）+ MPC 控制器（3.8 组合方式）+
  共享执行管线（3.9），调用 ``EpisodeRunner`` 跑完整回合；
- 每次输出 JSON + NPZ 两份产物（见 ``_OUTPUT_SCHEMA``），标记
  **simulation-only**（不作可直接下发真机的轨迹承诺）；
- 返回可 JSON 序列化的指标 dict；**不连接 real 模块**（无 SDK/真机依赖）。

组合来源（上游冻结接口，消费为准）：
- 3.1 ``load_sim_task_config`` / ``validate_simulation_inputs``
- 3.4 ``build_r1d_environments`` → RobotEnvironments(execution/planning)
- 3.7 ``ForbiddenContactChecker.from_env`` + ``RobotLimits.from_runtime_parameters``
  + ``PredictiveSafetyFilter(enable_x_wall=False, contact_checker=...)``
- 3.8 ``MPCController``/``MPCConfig``（shoulder_pos 必须显式来自 task YAML —
  MPCConfig 缺省是 RM65 硬编码值，R1D 组装不得依赖缺省）
- 3.9 ``SimComponent``（按解析 geom ID 判定接触）+ ``SimPerception`` +
  ``classify_hit_result(robot_name="R1-A7-D")``

复用 ``EpisodeRunner``（4 组件 + 5 hook），不复制 V12 主循环。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.ilqt.components.predictive_safety import PredictiveSafetyFilter
from src.ilqt.components.sim_component import SimComponent
from src.ilqt.components.sim_perception import SimPerception
from src.ilqt.episode_runner import EpisodeRunner
from src.ilqt.mpc_controller import MPCConfig, MPCController
from src.ilqt.robot_limits import RobotLimits
from src.ilqt.solver import is_cpp_kernel_compatible
from src.ilqt.tube_types import TubeConfig
from src.robot.robot_profile import load_robot_profile
from src.robot.runtime_parameters import load_runtime_parameters
from src.sim.collision_checks import ForbiddenContactChecker
from src.sim.env_factory import build_r1d_environments
from src.sim.hit_detection import classify_hit_result
from src.sim.sim_config import load_sim_task_config, validate_simulation_inputs

logger = logging.getLogger(__name__)

# R1D 身份与显示名（3.9 classify_hit_result 的 robot_name 约定）
R1D_ROBOT_ID = "unitree_r1d"
R1D_DISPLAY_NAME = "R1-A7-D"

# 产物 schema 与文件名（新输出，独立于 RM65 --dump-trajectory 的 NPZ schema）
_OUTPUT_SCHEMA = "r1d_sim_episode_v1"
_JSON_NAME = "r1d_episode.json"
_NPZ_NAME = "r1d_episode.npz"

# mpc_config 允许键 = MPCConfig 数值调度/随挥类字段（task YAML → MPCConfig 透传）
_MPC_CONFIG_ALLOWED = frozenset({
    "dt", "total_horizon", "fixed_horizon", "replan_interval",
    "max_iter_per_plan", "first_plan_iters", "near_plan_iters",
    "near_threshold", "far_threshold",
    "Q_p_scale_far", "Q_v_scale_far", "Q_p_scale_near", "Q_v_scale_near",
    "R", "normal_weight", "racket_speed", "target_speed",
    "Q_tcp_soft", "Q_qdot_limit", "tcp_soft_ratio", "qdot_limit_ratio",
    "softmin_beta",
    "follow_through_length", "follow_through_steps",
    "follow_through_v_terminal", "follow_trigger", "no_follow_through",
    "use_r_decay", "r_decay_ratio",
})

# RM65 专属策略键 + runner 独占控制键（出现在 mpc_config 即拒绝，给出点名错误）
_MPC_CONFIG_FORBIDDEN = frozenset({
    "fix_joint5_angle",          # R1D 不使用 fix_joint5（3.8 策略决定）
    "use_backswing", "backswing_offset", "backswing_ratio",  # RM65 后摆
    "normal_flip",
    "async_mode",                # R1D 异步明确拒绝（3.8）
    "shoulder_pos", "workspace_radius",  # 红线：必须走 task 顶层 workspace_*
    "cost_config", "solver_backend", "solver_config", "tube_cfg",
    "version", "is_position_mode", "ablation_mode",
})


def run_r1d_episode(
    profile_path: Path,
    runtime_path: Path,
    task_path: Path,
    output_dir: Path,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """运行一个 R1D 仿真击球回合并写 JSON+NPZ 产物。

    流程：配置加载与预检（身份/侧别错误在此抛出，早于任何模型构造）→
    环境对 + 限位/碰撞保护 + MPC + 共享管线组装 → ``EpisodeRunner.run`` →
    汇总指标与接触判定 → 写 ``r1d_episode.json`` / ``r1d_episode.npz``。

    Args:
        profile_path: 机器人描述 YAML（robot_id 须为 unitree_r1d）。
        runtime_path: 仿真运行参数 YAML（完整显式，q 限位可 null 由模型回填）。
        task_path: 仿真任务 YAML（SimTaskConfig schema v1）。
        output_dir: 产物输出目录（不存在则创建；产物固定命名，重复运行覆盖）。
        seed: 可选随机种子（当前 R1D 仿真无随机源，仅记录进产物供追溯）。

    Returns:
        可 JSON 序列化的回合指标 dict（与 JSON 产物同构）。

    Raises:
        FileNotFoundError: 任一配置文件不存在。
        ValueError: robot_id 非 unitree_r1d、侧别错配（profile/runtime/task
            身份或关节序不一致）、mpc_config 含禁止键、dt 不一致，或任一
            上游预检失败 —— 全部在构造环境/求解器之前抛出。
    """
    # ── 1. 配置加载 + 严格预检（纯 YAML 阶段，无模型开销）──
    profile = load_robot_profile(Path(profile_path))
    if profile.robot_id != R1D_ROBOT_ID:
        raise ValueError(
            f"run_r1d_episode 仅支持 {R1D_ROBOT_ID}（收到 robot_id "
            f"'{profile.robot_id}'，来源 {profile_path}）；RM65 请走 V12 缺省路径"
        )
    parameters = load_runtime_parameters(Path(runtime_path), profile)
    task = load_sim_task_config(Path(task_path), profile)
    validate_simulation_inputs(profile, parameters, task)
    side = _resolve_side(task.joint_names)

    # ── 2. 环境对（带球执行 + 规划副本，模型/data 独立）──
    envs = build_r1d_environments(profile, parameters, task)
    execution = envs.execution
    planning = envs.planning
    scene = execution.scene
    filled = scene.robot.parameters  # 场景回填后的同源运行参数

    # dt 一致性：runtime 声明值 vs 模型编译值（不一致即配置漂移，快速失败）
    dt = float(planning.dt)
    if not math.isclose(dt, float(filled.dt), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"dt 不一致：模型 timestep={dt} vs runtime dt={filled.dt}"
            f"（robot_id: {profile.robot_id}）"
        )

    # ── 3. MPCConfig（红线：shoulder_pos/workspace_radius 显式来自 task）──
    mpc_config, backend_info = _build_mpc_config(task, dt)
    backend_info["effective"] = _describe_effective_backend(planning)

    # ── 4. 限位 + 禁止对碰撞检查 + 预测安全滤波（3.7 推荐配方逐字）──
    limits = RobotLimits.from_runtime_parameters(filled, execution.control_range)
    checker = ForbiddenContactChecker.from_env(
        execution, scene.protected_geom_pairs
    )
    safety = PredictiveSafetyFilter(
        execution, limits,
        is_position_mode=(task.actuator_mode == "position"),
        arm_nq=execution.NQ,
        enable_x_wall=False,       # R1D 禁用 RM65 世界 X 墙（True 构造即失败）
        contact_checker=checker,
    )

    # ── 5. MPC 控制器（planning 副本；绑定环境 auto → Python FD）──
    mpc = MPCController(planning, mpc_config, robot_limits=limits)

    # ── 6. 共享管线组件（3.9 组装模板）──
    init_q = np.asarray(filled.init_q, dtype=np.float64)
    is_position_mode = task.actuator_mode == "position"
    ball_vel_history: list[np.ndarray] = []

    def _record_ball_vel(ctx: object) -> None:
        """post_exec hook：逐步记录物理球速度（与 U 序列对齐）。"""
        ball_vel_history.append(execution.get_ball_vel().copy())

    sim_component = SimComponent(
        execution, mpc, limits, init_q, is_position_mode,
        hard_x_body_ids=[],              # R1D：跳过 X 墙（避免一逻辑步两次 step_full）
        initial_hitting_tube=None,       # 候选窗口由 do_replan 内部构建
        tube_cfg=mpc_config.tube_cfg,
        dt=dt,
        ball_geom_id=scene.ball_geom_id,
        racket_geom_ids=scene.racket_geom_ids,
    )
    perception = SimPerception(execution)

    # 初始 history（对齐 V12 约定：X_history[0]=x0，ball_pos[0]=发球初态）
    x0 = execution.get_arm_state()
    sim_component.X_history = [x0.copy()]
    sim_component.ball_pos_history = [execution.get_ball_pos().copy()]
    ball_vel_history.append(execution.get_ball_vel().copy())

    # 安全滤波提前停止的可观察标志（on_unsafe hook 置位）
    unsafe_stopped = {"flag": False}

    def _mark_unsafe(ctx: object) -> None:
        """on_unsafe hook：记录安全滤波拒绝导致的停止。"""
        unsafe_stopped["flag"] = True

    runner = EpisodeRunner(
        mpc=mpc,
        perception=perception,
        safety=safety,
        executor=sim_component,
        post_exec_hooks=[_record_ball_vel],
        on_unsafe_hooks=[_mark_unsafe],
    )

    # ── 7. 运行回合 ──
    logger.info(
        "R1D 回合开始: side=%s mode=%s backend=%s max_steps=%d dt=%.4f",
        side, task.actuator_mode, backend_info["effective"],
        int(task.max_steps), dt,
    )
    t_start = time.perf_counter()
    metrics = runner.run(max_steps=int(task.max_steps))
    wall_time_s = time.perf_counter() - t_start

    # ── 8. 汇总 + 产物 ──
    result = _summarize(
        task=task, side=side, dt=dt, seed=seed,
        filled=filled, backend_info=backend_info,
        profile_path=Path(profile_path), runtime_path=Path(runtime_path),
        task_path=Path(task_path), model_path=Path(profile.model_path),
        sim_component=sim_component, mpc=mpc,
        metrics=metrics, wall_time_s=wall_time_s,
        unsafe_stopped=unsafe_stopped["flag"],
    )
    _write_outputs(
        Path(output_dir), result, sim_component, ball_vel_history, dt
    )
    logger.info(
        "R1D 回合结束: steps=%s safe=%s hit=%s，产物已写入 %s",
        result["episode"]["total_steps"], result["episode"]["safe_steps"],
        result["hit"]["type"], Path(output_dir),
    )
    return result


# ============================================================================
#  配置 → MPCConfig 映射
# ============================================================================


def _build_mpc_config(
    task: Any, dt: float
) -> tuple[MPCConfig, dict[str, str]]:
    """把 task YAML 的 mpc_config/solver_config/cost_config 映射进 MPCConfig。

    Args:
        task: SimTaskConfig（mpc_config 键域受控；cost_config 透传 3.6；
            solver_config 拆出 backend 控制键，其余透传 build_solver）。
        dt: 生效仿真步长（runtime 与模型一致值；mpc_config.dt 须与之相等）。

    Returns:
        (MPCConfig 实例, backend 信息 dict)。

    Raises:
        ValueError: mpc_config 含 RM65 专属/未知键、solver backend 非法，
            或 mpc_config.dt 与生效 dt 不一致。
    """
    mpc_raw = dict(task.mpc_config)
    forbidden_hits = sorted(set(mpc_raw) & _MPC_CONFIG_FORBIDDEN)
    if forbidden_hits:
        raise ValueError(
            f"task mpc_config 含 RM65 专属或 runner 独占键 {forbidden_hits}"
            f"（fix_joint5/后摆/async 等对 R1D 明确拒绝；shoulder_pos/"
            f"workspace_radius 必须走 task 顶层 workspace_center/radius，"
            f"不得在 mpc_config 内另行给值；robot_id: {task.robot_id}）"
        )
    unknown = sorted(set(mpc_raw) - _MPC_CONFIG_ALLOWED)
    if unknown:
        raise ValueError(
            f"task mpc_config 含未知键 {unknown}"
            f"（允许键: {sorted(_MPC_CONFIG_ALLOWED)}；robot_id: {task.robot_id}）"
        )

    # dt 注入与一致性（缺省注入生效值；显式提供须相等）
    overrides = dict(mpc_raw)
    if "dt" in overrides:
        if not math.isclose(
            float(overrides["dt"]), dt, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"task mpc_config.dt={overrides['dt']} 与生效 dt={dt} 不一致"
                f"（runtime dt 与模型 timestep 已校验相等，三处必须一致）"
            )
    overrides["dt"] = dt

    # solver_config：拆出 backend 控制键，其余透传 build_solver 数值配置
    solver_cfg = dict(task.solver_config)
    backend = str(solver_cfg.pop("backend", "auto"))
    if backend not in ("auto", "python", "cpp"):
        raise ValueError(
            f"solver_config.backend={backend!r} 非法"
            f"（支持 'auto' / 'python' / 'cpp'；R1D 建议显式 'python'）"
        )

    # tube 子字典 → TubeConfig 实例（MPC 层 Tube 包装按 config.tube_cfg 组装，
    # 工厂不消费 tube 子字典 —— 3.6 单源纪律）
    tube_dict = task.cost_config.get("tube")
    if tube_dict is None:
        tube_cfg = TubeConfig()
    elif isinstance(tube_dict, dict):
        tube_cfg = TubeConfig(**tube_dict)
    else:
        raise ValueError(
            "cost_config.tube 必须是映射结构（TubeConfig 字段子字典）"
        )

    config = MPCConfig(
        version="r1d_sim_v1",
        is_position_mode=(task.actuator_mode == "position"),
        # 红线（3.8 冻结提醒）：显式当前侧工作空间，不得依赖 RM65 硬编码缺省
        shoulder_pos=np.array(task.workspace_center, dtype=np.float64),
        workspace_radius=float(task.workspace_radius),
        cost_config=dict(task.cost_config),
        solver_backend=backend,           # type: ignore[arg-type]
        solver_config=solver_cfg,
        async_mode=False,                 # R1D 本阶段同步（True 会被 3.8 拒绝）
        tube_cfg=tube_cfg,
        **overrides,
    )
    backend_info = {"requested": backend, "effective": "python-fd"}
    return config, backend_info


def _describe_effective_backend(planning: Any) -> str:
    """描述规划环境的实际生效求解后端（R1D 非 C++ 兼容 → Python FD）。"""
    try:
        compatible = is_cpp_kernel_compatible(planning)
    except Exception:  # pragma: no cover - 防御：描述性字段不阻断回合
        return "python"
    return "python-fd" if not compatible else "python-analytical"


# ============================================================================
#  侧别解析（由完整关节名校验得出，不靠文件名猜测）
# ============================================================================


def _resolve_side(joint_names: tuple[str, ...]) -> str:
    """由完整主动关节名的前缀一致性地解析侧别。

    Args:
        joint_names: 主动关节名元组（已与 profile 逐项同序校验）。

    Returns:
        "left" 或 "right"。

    Raises:
        ValueError: 前缀缺失或左右混杂（无法唯一确定侧别）。
    """
    sides: set[str] = set()
    for name in joint_names:
        if name.startswith("left_"):
            sides.add("left")
        elif name.startswith("right_"):
            sides.add("right")
    if sides == {"left"}:
        return "left"
    if sides == {"right"}:
        return "right"
    raise ValueError(
        f"无法由完整关节名唯一确定侧别（期望全体 left_* 或全体 right_*，"
        f"实际解析到 {sorted(sides) or '无前缀'}；joint_names={list(joint_names)}）"
    )


# ============================================================================
#  指标汇总与产物写出
# ============================================================================


def _summarize(
    *,
    task: Any,
    side: str,
    dt: float,
    seed: int | None,
    filled: Any,
    backend_info: dict[str, str],
    profile_path: Path,
    runtime_path: Path,
    task_path: Path,
    model_path: Path,
    sim_component: SimComponent,
    mpc: MPCController,
    metrics: dict,
    wall_time_s: float,
    unsafe_stopped: bool,
) -> dict[str, Any]:
    """汇总回合指标为可 JSON 序列化 dict（与 JSON 产物同构）。"""
    active = bool(sim_component.active_contact)
    passive = bool(sim_component.passive_contact)

    # 击球相关（未触球时如实为 None/-1，不伪造数值）
    hit_step = int(sim_component.hit_step)
    p_ee_at_hit = sim_component.p_ee_at_hit
    ball_pos_at_hit = sim_component.ball_pos_at_hit
    if p_ee_at_hit is not None and ball_pos_at_hit is not None:
        pos_error = float(np.linalg.norm(p_ee_at_hit - ball_pos_at_hit))
    else:
        d_arr = np.asarray(sim_component.distances_history, dtype=np.float64)
        pos_error = float(np.min(d_arr)) if d_arr.size else float("inf")
    hit_type = "active" if active else ("passive" if passive else "miss")

    # 失败原因（从可观察信号推断；正常走完为 None）
    total_steps = int(metrics.get("total_steps", 0))
    safe_steps = int(metrics.get("safe_steps", 0))
    mpc_done = bool(metrics.get("mpc_done", False))
    if metrics.get("error") is not None:
        failure_reason = str(metrics["error"])
    elif bool(getattr(mpc, "ball_unreachable", False)):
        failure_reason = "ball_unreachable"
    elif unsafe_stopped:
        failure_reason = "safety_filter_stop"
    elif safe_steps < total_steps:
        failure_reason = "ball_state_unavailable"
    elif not mpc_done and total_steps >= int(task.max_steps):
        failure_reason = None  # 步数上限内正常走完（未触发随挥结束）
    else:
        failure_reason = None

    scene_dict = asdict(filled.scene)

    return {
        "schema": _OUTPUT_SCHEMA,
        "simulation_only": True,
        "robot_id": task.robot_id,
        "side": side,
        "joint_names": list(task.joint_names),
        "dt": dt,
        "mode": task.actuator_mode,
        "backend": dict(backend_info),
        "input_sha256": {
            "profile_yaml": _sha256_file(profile_path),
            "runtime_yaml": _sha256_file(runtime_path),
            "task_yaml": _sha256_file(task_path),
            "model_template": _sha256_file(model_path),
        },
        "provisional_fields": list(filled.provisional_fields),
        "fixed_scene_params": _to_jsonable(scene_dict),
        "workspace": {
            "center": np.asarray(task.workspace_center, dtype=np.float64).tolist(),
            "radius": float(task.workspace_radius),
        },
        "ball_initial": {
            "position": np.asarray(
                task.initial_ball_position, dtype=np.float64
            ).tolist(),
            "velocity": np.asarray(
                task.initial_ball_velocity, dtype=np.float64
            ).tolist(),
        },
        "seed": seed,
        "episode": {
            "max_steps": int(task.max_steps),
            "total_steps": total_steps,
            "safe_steps": safe_steps,
            "mpc_done": mpc_done,
            "first_plan_ms": float(metrics.get("first_plan_ms", 0.0)),
            "wall_time_s": float(wall_time_s),   # 非确定字段（性能记录）
            "failure_reason": failure_reason,
        },
        "hit": {
            "type": hit_type,
            "step": hit_step,
            "time_s": float(hit_step * dt) if hit_step >= 0 else None,
            "racket_face_speed": (
                float(sim_component.v_ee_at_hit)
                if sim_component.v_ee_at_hit is not None else None
            ),
            "p_ee_at_hit": (
                np.asarray(p_ee_at_hit, dtype=np.float64).tolist()
                if p_ee_at_hit is not None else None
            ),
            "ball_pos_at_hit": (
                np.asarray(ball_pos_at_hit, dtype=np.float64).tolist()
                if ball_pos_at_hit is not None else None
            ),
            "ball_out_speed": (
                float(sim_component.ball_out_speed)
                if sim_component.ball_out_speed is not None else None
            ),
            "ball_out_vy": (
                float(sim_component.ball_out_vy)
                if sim_component.ball_out_vy is not None else None
            ),
            "message": classify_hit_result(
                active, passive, pos_error, robot_name=R1D_DISPLAY_NAME
            ),
        },
        "metrics": {
            "n_mpc_steps": len(sim_component.U_history),
            "min_dist": pos_error if math.isfinite(pos_error) else None,
            "max_qdot_ratio": float(sim_component.max_qdot_ratio),
            "max_tcp_speed": float(sim_component.max_tcp_speed),
            "max_racket_face_speed": float(sim_component.max_racket_face_speed),
            "active_contact": active,
            "passive_contact": passive,
        },
    }


def _write_outputs(
    output_dir: Path,
    result: dict[str, Any],
    sim_component: SimComponent,
    ball_vel_history: list[np.ndarray],
    dt: float,
) -> None:
    """写 JSON + NPZ 产物（simulation-only；固定命名，重复运行覆盖）。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / _JSON_NAME
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    X = np.asarray(sim_component.X_history, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)  # 防御：仅初始态（正常路径 X_history 非空）
    nu = X.shape[1] // 2
    U = np.asarray(sim_component.U_history, dtype=np.float64).reshape(-1, nu)
    ball_pos = np.asarray(
        sim_component.ball_pos_history, dtype=np.float64
    ).reshape(-1, 3)
    ball_vel = np.asarray(ball_vel_history, dtype=np.float64).reshape(-1, 3)
    t = np.arange(X.shape[0], dtype=np.float64) * dt

    np.savez(
        output_dir / _NPZ_NAME,
        X=X, U=U, ball_pos=ball_pos, ball_vel=ball_vel, t=t,
        simulation_only=np.array(True),
        schema=np.array(_OUTPUT_SCHEMA),
    )
    logger.debug(
        "产物写出: %s（X %s, U %s）", output_dir / _NPZ_NAME, X.shape, U.shape
    )


def _sha256_file(path: Path) -> str:
    """计算文件内容的 SHA256 十六进制摘要（可追溯性输入指纹）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_jsonable(value: Any) -> Any:
    """递归把 numpy 标量/数组转为 JSON 原生类型。"""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
