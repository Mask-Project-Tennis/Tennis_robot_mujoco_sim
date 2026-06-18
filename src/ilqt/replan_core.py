"""异步重规划核心编排函数 do_replan。

在后台线程中执行完整的 MPC+iLQR 重规划流程，使用独立的 env_plan（独立 MjData）。
依赖 tube_types、mpc_helpers、tube_builder、tube_cost 等模块，
供仿真主脚本（V11）与真机 runner 共享复用。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.ilqt.async_replanner import PlanRequest, PlanResult
from src.ilqt.cost import HittingCost
from src.ilqt.tube_types import ReplanState
from src.ilqt.mpc_helpers import (
    _jt_init_dispatch,
    _backswing_dispatch,
    _fix_joint5_dispatch,
    resample_control_sequence,
    compute_r_schedule,
)
from src.ilqt.tube_builder import (
    search_hit_window,
    build_hitting_tube,
)
from src.ilqt.tube_cost import (
    TubeHittingCostWrapper,
    TubeOnlyCost,
    SoftminOnlyCost,
)
from src.tennis.hitting import find_hitting_point_physics

if TYPE_CHECKING:
    from src.ilqt.replan_config import ReplanConfig
    from src.sim.rm65_env import RM65Env


logger = logging.getLogger(__name__)


def do_replan(
    request: PlanRequest,
    env_plan: "RM65Env",
    state: ReplanState,
    cfg: "dict | ReplanConfig",
) -> PlanResult:
    """在后台线程中执行完整的重规划流程。

    使用独立的 env_plan（独立 MjData），不与主线程共享可变 MuJoCo 状态。

    Args:
        request: 规划请求（球状态、臂状态等）。
        env_plan: 独立 MjData 的规划环境。
        state: 当前的重规划状态快照。
        cfg: 配置字典或 ReplanConfig（B2 兼容层自动转 dict），
            包含所有规划参数。

    Returns:
        PlanResult: 规划结果（新控制序列、击打点等）。
    """
    # B2 兼容层: ReplanConfig → dict（统一支持类型安全对象与旧 dict 接口）
    # isinstance 缩窄确保后续 cfg["..."] / cfg.get(...) 类型安全
    if not isinstance(cfg, dict):
        cfg = cfg.to_dict()

    result = PlanResult()
    x_current = request.x_current.copy()
    ball_pos = request.ball_pos.copy()
    ball_vel = request.ball_vel.copy()
    step = request.step
    k_hit_new = state.k_hit_new

    remaining_horizon = cfg["total_horizon"] - step

    # 1. 查找击打点
    env_plan.set_ball_state(ball_pos, ball_vel)
    hit_info_new = find_hitting_point_physics(
        env_plan, ball_pos, ball_vel, cfg["shoulder_pos"], cfg["workspace_radius"],
        remaining_horizon,
    )
    if hit_info_new is None:
        logger.warning(
            "ASYNC 步 %d: 球不可达, ball_pos=%s, ball_vel=%s, remaining_horizon=%d",
            step, ball_pos, ball_vel, remaining_horizon,
        )
        result.ball_unreachable = True
        return result

    k_hit_candidate = hit_info_new["k_hit"]
    if k_hit_candidate < max(10, k_hit_new // 4) and k_hit_new > 30:
        k_hit_candidate = max(1, k_hit_new - cfg["replan_interval"])

    # 衰减式扰动：计算衰减系数（模拟预测精度随观测逐步提高）
    decay_alpha = 1.0
    if abs(cfg.get("time_perturb_s", 0.0)) > 1e-6 or abs(cfg.get("space_perturb_m", 0.0)) > 1e-6:
        r_interval = cfg.get("replan_interval", 20)
        r_total = max(1, cfg.get("k_hit_total", 100) // r_interval)
        r_count = max(1, (cfg.get("k_hit_total", 100) - remaining_horizon) // r_interval + 1)
        decay_alpha = max(cfg.get("perturb_alpha_min", 0.0), 1.0 - r_count / r_total)

    # 时间扰动（衰减式）
    if abs(cfg.get("time_perturb_s", 0.0)) > 1e-6:
        effective_perturb = cfg["time_perturb_s"] * decay_alpha
        perturb_steps = int(round(effective_perturb / cfg["dt"]))
        if perturb_steps != 0:
            k_hit_candidate = k_hit_candidate - perturb_steps
            k_hit_candidate = max(5, min(k_hit_candidate, remaining_horizon - 1))

    p_hit_new = hit_info_new["p_hit"].copy()
    v_ball_hit_new = hit_info_new["v_ball_hit"].copy()
    k_hit_new = k_hit_candidate

    # 2. 击球点可执行性后过滤
    q_hit_feas = env_plan.solve_ik(p_hit_new, q_init=x_current[:env_plan.NQ], max_iter=50, eps=1e-2)
    env_plan.set_arm_state(np.concatenate([q_hit_feas, np.zeros(env_plan.NQ)]))
    J_p_feas = env_plan.get_ee_jacp()
    max_ee_v = float(np.linalg.norm(np.abs(J_p_feas) @ cfg["robot_limits"].qdot_max))
    ball_spd = float(np.linalg.norm(v_ball_hit_new))
    if ball_spd > max_ee_v * 2.0:
        logger.warning("ASYNC 步 %d: 球速 %.1fm/s 超过限速 %.1fm/s", step, ball_spd, max_ee_v)

    # 3. 空间偏移（衰减式）
    if abs(cfg.get("space_perturb_m", 0.0)) > 1e-6:
        effective_sp = cfg["space_perturb_m"] * decay_alpha
        if abs(effective_sp) > 1e-6:
            d_ball_hit = v_ball_hit_new / (np.linalg.norm(v_ball_hit_new) + 1e-8)
            lateral = np.cross(d_ball_hit, np.array([0.0, 0.0, 1.0]))
            lateral_norm = np.linalg.norm(lateral)
            if lateral_norm > 1e-6:
                lateral /= lateral_norm
            else:
                lateral = np.array([1.0, 0.0, 0.0])
            p_hit_new = p_hit_new + lateral * effective_sp

    # 4. 法向量和随挥目标
    n_des_new = -v_ball_hit_new / (np.linalg.norm(v_ball_hit_new) + 1e-8)
    if cfg.get("normal_flip", False):
        n_des_new = -n_des_new

    # v5: 终端目标 = 随挥终点（击球点前方 follow_through_length）
    p_follow_new = p_hit_new + cfg["hit_shift"] * cfg["d_hat"]

    # v6: horizon 仅到击球时刻（不加随挥段）
    horizon_full = k_hit_new
    horizon_plan = min(horizon_full, cfg["fixed_horizon"])

    # v6: 终端目标 = 击球点 + 偏移（arm 穿过击球点）
    p_terminal_v5 = p_hit_new + cfg["follow_through_length"] * cfg["d_hat"]
    v_terminal_v5 = cfg["v_hit_at_contact"]

    # v11: 远段 JT 控制（k_hit > far_threshold 且非首次规划时跳过 iLQR）
    # 对齐 V11 主循环 1679-1694 行：远段用雅可比转置初始控制，近段才用 iLQR
    far_threshold_jt = cfg.get("far_threshold", 50)
    if k_hit_new > far_threshold_jt and not request.is_first_plan:
        ball_pos_save_jt, ball_vel_save_jt = env_plan.get_ball_state()
        u_jt = _jt_init_dispatch(
            env_plan, x_current, p_follow_new, cfg["replan_interval"], gain=60.0,
            fix_joint5_angle=cfg.get("fix_joint5_angle"),
        )
        env_plan.set_ball_state(ball_pos_save_jt, ball_vel_save_jt)
        env_plan.set_arm_state(x_current)

        result.request_step = step
        result.k_hit_new = k_hit_new
        result.p_hit_new = p_hit_new.copy()
        result.v_ball_hit_new = v_ball_hit_new.copy()
        result.n_des_new = n_des_new.copy()
        result.solver_ok = True
        result.iters_plan = 0
        result.horizon_plan = 0
        result.fast_lin = False
        result.fp_limits_was_none = True
        result.U_mpc_full = u_jt.copy()
        result.U_buffer = u_jt[: cfg["replan_interval"]].copy()
        result.U_prev = (
            request.U_prev.copy()
            if len(request.U_prev) > 0
            else np.zeros((0, env_plan.NU))
        )
        logger.debug("FAR_JT step=%d k_hit=%d (JT gain=60, skip iLQR)", step, k_hit_new)
        return result

    # 6. 位置误差 → 权重调度
    env_plan.set_arm_state(x_current)
    env_plan.update_kinematics()
    pos_err_now = float(np.linalg.norm(env_plan.get_ee_pos() - p_hit_new))

    _tc = 0.10
    _tw = 0.03
    _s = 1.0 / (1.0 + np.exp(-(pos_err_now - _tc) / _tw))
    Q_p_scale = cfg["Q_p_scale_near"] + (cfg["Q_p_scale_far"] - cfg["Q_p_scale_near"]) * _s
    Q_v_scale = cfg["Q_v_scale_near"] + (cfg["Q_v_scale_far"] - cfg["Q_v_scale_near"]) * _s

    # 7. 迭代策略（与主循环同步策略保持一致）
    near_threshold = cfg["near_threshold"]
    iters_plan = cfg["max_iter_per_plan"]
    skip_ls = True
    fp_limits = cfg["robot_limits"]
    fast_lin = False

    if request.is_first_plan:
        iters_plan = cfg["first_plan_iters"]
        skip_ls = True
        fp_limits = None
    elif k_hit_new <= near_threshold:
        iters_plan = cfg["near_plan_iters"]
        if k_hit_new > 30:
            fast_lin = True
            fp_limits = None
        else:
            fast_lin = True
            fp_limits = None
    else:
        iters_plan = cfg["max_iter_per_plan"]
        fast_lin = True

    # 8. 计算 warm start
    U_prev = request.U_prev.copy()
    fix_joint5_angle = cfg.get("fix_joint5_angle")

    if not cfg.get("use_backswing", False):
        if len(U_prev) >= horizon_full // 3:
            U_warm = resample_control_sequence(U_prev, horizon_full)[:horizon_plan]
            if fix_joint5_angle is not None:
                U_warm = _fix_joint5_dispatch(U_warm, x_current, env_plan, fix_joint5_angle)
        else:
            U_warm = _jt_init_dispatch(
                env_plan, x_current, p_follow_new, horizon_full, gain=30.0,
                fix_joint5_angle=fix_joint5_angle,
            )[:horizon_plan]
    else:
        if len(U_prev) >= horizon_full // 3:
            U_warm = resample_control_sequence(U_prev, horizon_full)[:horizon_plan]
            if fix_joint5_angle is not None:
                U_warm = _fix_joint5_dispatch(U_warm, x_current, env_plan, fix_joint5_angle)
        else:
            U_warm_full, _ = _backswing_dispatch(
                env_plan, x_current, p_follow_new, cfg.get("v_hit_at_contact", cfg["v_hit_desired"]), horizon_full,
                backswing_offset=cfg.get("backswing_offset", 0.0),
                backswing_ratio=cfg.get("backswing_ratio", 0.3),
                fix_joint5_angle=fix_joint5_angle,
                n_des=n_des_new,
            )
            U_warm = U_warm_full[:horizon_plan]

    # 9. 构建 cost_fn（使用临时 cost_fn，env_plan 独立 MjData）
    # v12: 使用配置中的 Q_p_base/Q_v_base（对齐 V11 base_cost_fn 的权重基底）
    _q_p_base = cfg.get("Q_p_base", None)
    if _q_p_base is None:
        Q_p_mat = Q_p_scale * np.eye(3)
    elif _q_p_base.ndim == 1:
        Q_p_mat = Q_p_scale * np.diag(_q_p_base)
    else:
        Q_p_mat = Q_p_scale * _q_p_base
    _q_v_base = cfg.get("Q_v_base", None)
    if _q_v_base is None:
        Q_v_mat = Q_v_scale * np.eye(3)
    elif _q_v_base.ndim == 1:
        Q_v_mat = Q_v_scale * np.diag(_q_v_base)
    else:
        Q_v_mat = Q_v_scale * _q_v_base
    R_mat = cfg["R"] * np.eye(env_plan.NU)
    cost_fn_plan = HittingCost(
        env_plan, p_terminal_v5, v_terminal_v5, Q_p_mat, Q_v_mat, R_mat,
        Q_n=cfg.get("normal_weight", 500000.0),
        n_des=n_des_new,
        Q_qdot=cfg.get("Q_qdot_base", 0.0),
        Q_qddot=cfg.get("Q_qddot_base", 0.0),
        Q_du=cfg.get("Q_du_base", 0.0),
        actuator_mode=1 if cfg.get("is_position_mode", False) else 0,
    )

    if cfg.get("use_r_decay", False) and not cfg.get("is_position_mode", False):
        R_schedule = compute_r_schedule(
            horizon_full, cfg["R"],
            decay_ratio=cfg.get("r_decay_ratio", 0.3),
        )[:horizon_plan]
        cost_fn_plan.set_R_schedule(R_schedule)

    # v11: do_replan 中也构建 Tube/Softmin 代价（与主循环一致）
    ablation_mode_replan = cfg.get("ablation_mode", "full")
    need_candidates_replan = ablation_mode_replan in ("full", "tube_only", "softmin_only")
    if need_candidates_replan and horizon_full > 5:
        tube_cfg_replan = cfg.get("tube_cfg", None)
        if tube_cfg_replan is not None:
            hit_window_replan = search_hit_window(
                env_plan, ball_pos, ball_vel,
                cfg["shoulder_pos"], cfg["workspace_radius"],
                remaining_horizon, tube_cfg_replan,
                ball_direction="y",
                current_step=0,
                robot_limits=cfg["robot_limits"],
                init_q=x_current[:env_plan.NQ].copy(),
            )
            if hit_window_replan is not None:
                racket_speed_replan = float(cfg.get("racket_speed", 5.0))
                d_follow_replan = cfg.get("d_follow", cfg["d_hat"])
                hitting_tube_replan = build_hitting_tube(
                    hit_window_replan, racket_speed_replan, d_follow_replan, tube_cfg_replan,
                )
                if ablation_mode_replan == "full":
                    cost_fn_plan = TubeHittingCostWrapper(
                        env_plan, cost_fn_plan, hitting_tube_replan, k_hit_new, tube_cfg_replan,
                    )
                elif ablation_mode_replan == "tube_only":
                    cost_fn_plan = TubeOnlyCost(
                        env_plan, cost_fn_plan, hitting_tube_replan, k_hit_new, tube_cfg_replan,
                    )
                elif ablation_mode_replan == "softmin_only":
                    cost_fn_plan = SoftminOnlyCost(
                        env_plan, cost_fn_plan, hitting_tube_replan, k_hit_new, tube_cfg_replan,
                    )

    # v6: 不设 midpoint（与 v2 一致，避免梯度冲突）

    # 分阶段软平滑权重调度
    if hasattr(cost_fn_plan, 'set_smoothness_scale'):
        if k_hit_new > 50:
            sq = cfg.get("smooth_far", {"Q_qdot_mult": 0.01, "Q_qddot_mult": 0.01, "Q_du_mult": 0.1})
        elif k_hit_new > 20:
            sq = cfg.get("smooth_mid", {"Q_qdot_mult": 0.1,  "Q_qddot_mult": 0.1,  "Q_du_mult": 0.2})
        else:
            sq = cfg.get("smooth_near", {"Q_qdot_mult": 1.0,  "Q_qddot_mult": 1.0,  "Q_du_mult": 0.5})
        cost_fn_plan.set_smoothness_scale(
            float(sq["Q_qdot_mult"]), float(sq["Q_qddot_mult"]), float(sq["Q_du_mult"]),
        )

    # 10. 求解 iLQR
    ball_pos_save, ball_vel_save = env_plan.get_ball_state()

    X_mpc, U_mpc, iter_costs, solver_ok = cfg["solver"].solve_few_iters(
        env_plan, cost_fn_plan, x_current, U_warm,
        max_iter=iters_plan,
        skip_linesearch=skip_ls,
        limits=fp_limits,
        use_fast_lin=fast_lin,
    )

    env_plan.set_ball_state(ball_pos_save, ball_vel_save)
    env_plan.set_arm_state(x_current)

    # 11. 构建 PlanResult
    result.request_step = step
    result.k_hit_new = k_hit_new
    result.p_hit_new = p_hit_new
    result.v_ball_hit_new = v_ball_hit_new
    result.n_des_new = n_des_new
    result.solver_ok = solver_ok
    result.iters_plan = iters_plan
    result.horizon_plan = horizon_plan
    result.fast_lin = fast_lin
    result.fp_limits_was_none = (fp_limits is None)
    result.U_mpc_full = U_mpc.copy()

    if solver_ok:
        if fix_joint5_angle is not None:
            U_mpc = _fix_joint5_dispatch(U_mpc, x_current, env_plan, fix_joint5_angle)
            result.U_mpc_full = U_mpc.copy()

        # U_prev：保存规划尾部，用于下次 warm start
        if len(U_mpc) > cfg["replan_interval"]:
            result.U_prev = U_mpc[cfg["replan_interval"]:].copy()
        elif len(U_mpc) > 0:
            result.U_prev = U_mpc[1:].copy()

        # U_buffer：异步模式需更长 buffer 覆盖后台规划延迟（≈horizon×0.2ms/步）
        # 远段 horizon~100 → ~200ms ≈ 40步，buffer 取 60步（×1.5余量）
        buffer_interval = cfg["replan_interval"]
        if len(U_mpc) >= buffer_interval * 6:
            result.U_buffer = U_mpc[:buffer_interval * 6].copy()
        elif len(U_mpc) >= buffer_interval * 4:
            result.U_buffer = U_mpc[:buffer_interval * 4].copy()
        elif k_hit_new <= 30 and len(U_mpc) >= buffer_interval * 2:
            result.U_buffer = U_mpc[:buffer_interval * 2].copy()
        else:
            result.U_buffer = U_mpc[:min(len(U_mpc), buffer_interval)].copy()
    else:
        # fallback: JT 控制
        u_jt = _jt_init_dispatch(
            env_plan, x_current, p_follow_new, horizon=cfg["replan_interval"], gain=40.0,
        )
        result.U_buffer = u_jt[:cfg["replan_interval"]].copy()
        result.U_prev = np.zeros((0, env_plan.NU))

    return result
