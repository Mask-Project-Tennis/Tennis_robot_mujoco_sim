"""MPC 规划辅助工具函数集合。

包含 iLQR 初始控制 dispatch（力矩/位置双模式）、fix_joint5 力矩版工具、
后摆 warm-start 力矩版、控制序列重采样、R 退火调度等。
供仿真主脚本（V11）与真机 runner 共享复用。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.ilqt.jt_init import (
    compute_jacobian_init_control_position,
    generate_backswing_warm_start_position,
    fix_joint5_control_trajectory_position,
)

if TYPE_CHECKING:
    from src.sim.rm65_env import RM65Env


# 力矩增益→位置增益的缩放因子
# 力矩版 gain=50 对应力矩 ~50Nm，位置版 gain=0.5 对应弧度增量 ~0.5rad
_GAIN_TORQUE_TO_POSITION = 0.01


def _jt_init_dispatch(
    env: "RM65Env",
    x0: np.ndarray,
    p_hit: np.ndarray,
    horizon: int,
    gain: float = 50.0,
    fix_joint5_angle: float | None = None,
) -> np.ndarray:
    """根据 env.actuator_mode 选择力矩版或位置版 JT 初始控制。"""
    if getattr(env, "actuator_mode", 0) == 1:
        return compute_jacobian_init_control_position(
            env, x0, p_hit, horizon, gain=gain * _GAIN_TORQUE_TO_POSITION,
            fix_joint5_angle=fix_joint5_angle,
        )
    return compute_jacobian_init_control(
        env, x0, p_hit, horizon, gain=gain,
        fix_joint5_angle=fix_joint5_angle,
    )


def _backswing_dispatch(
    env: "RM65Env",
    x0: np.ndarray,
    p_hit: np.ndarray,
    v_hit_desired: np.ndarray,
    horizon: int,
    backswing_offset: float = 0,
    backswing_ratio: float = 0,
    fix_joint5_angle: float | None = None,
    n_des: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """根据 env.actuator_mode 选择力矩版或位置版后摆 warm-start。"""
    if getattr(env, "actuator_mode", 0) == 1:
        U, q_des = generate_backswing_warm_start_position(
            env, x0, p_hit, v_hit_desired, horizon,
            backswing_offset=backswing_offset,
            backswing_ratio=backswing_ratio,
            fix_joint5_angle=fix_joint5_angle,
            n_des=n_des,
        )
        return U, q_des
    U, q_des = generate_backswing_warm_start(
        env, x0, p_hit, v_hit_desired, horizon,
        backswing_offset=backswing_offset,
        backswing_ratio=backswing_ratio,
        fix_joint5_angle=fix_joint5_angle,
        n_des=n_des,
    )
    return U, q_des


def _fix_joint5_dispatch(
    U: np.ndarray,
    x0: np.ndarray,
    env: "RM65Env",
    fix_joint5_angle: float,
) -> np.ndarray:
    """根据 env.actuator_mode 选择力矩版或位置版 fix_joint5。"""
    if getattr(env, "actuator_mode", 0) == 1:
        return fix_joint5_control_trajectory_position(U, fix_joint5_angle)
    return fix_joint5_control_trajectory(U, x0, env, fix_joint5_angle)


def _fix_joint5_single_dispatch(
    u: np.ndarray,
    q_fixed: float,
    x_current: np.ndarray,
    nq: int,
    env: "RM65Env",
) -> np.ndarray:
    """单步 fix_joint5：力矩模式用 PD 保持，位置模式直接设目标角度。"""
    if getattr(env, "actuator_mode", 0) == 1:
        u = u.copy()
        u[5] = q_fixed
        return u
    return fix_joint5_control(u, q_fixed, x_current, nq)


# ==============================================================================
# 辅助函数（从 rm65_mpc_fast.py 复用）
# ==============================================================================

def fix_joint5_control(
    u: np.ndarray,
    q_fixed: float,
    x_current: np.ndarray,
    nq: int,
    kp: float = 300.0,
    kd: float = 30.0,
) -> np.ndarray:
    """将第 6 关节（索引 5）的控制力矩替换为 PD 保持力矩。"""
    u = u.copy()
    q5_err = q_fixed - x_current[:nq][5]
    q5dot_err = -x_current[nq:][5]
    tau5 = kp * q5_err + kd * q5dot_err
    if u.ndim == 1:
        u[5] = tau5
    else:
        u[:, 5] = tau5
    return u


def fix_joint5_control_trajectory(
    U: np.ndarray,
    x0: np.ndarray,
    env: "RM65Env",
    q_fixed: float,
    kp: float = 300.0,
    kd: float = 30.0,
) -> np.ndarray:
    """将整个控制序列的第 6 关节替换为 PD 保持力矩。"""
    U = U.copy()
    x = x0.copy()
    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)
    bq = env.BALL_QPOS_START
    bv = env.BALL_QVEL_START
    ball_qpos_save = env.data.qpos[bq:bq + 7].copy()
    ball_qvel_save = env.data.qvel[bv:bv + 6].copy()

    for k in range(len(U)):
        q5_err = q_fixed - x[:env.NQ][5]
        q5dot_err = -x[env.NQ:][5]
        U[k, 5] = kp * q5_err + kd * q5dot_err
        x = env.step_from_state(x, U[k])

    env.data.qpos[bq:bq + 7] = ball_qpos_save
    env.data.qvel[bv:bv + 6] = ball_qvel_save

    if has_collision_ctrl:
        env.set_arm_collision(True)
    return U


def compute_jacobian_init_control(
    env: "RM65Env",
    x0: np.ndarray,
    p_hit: np.ndarray,
    horizon: int,
    gain: float = 50.0,
    fix_joint5_angle: float | None = None,
) -> np.ndarray:
    """基于雅可比转置法计算初始控制序列。"""
    U = np.zeros((horizon, env.NU))
    x = x0.copy()
    ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1]

    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)

    for k in range(horizon):
        env.set_arm_state(x)
        p_ee = env.get_ee_pos()
        J_p = env.get_ee_jacp()

        err = p_hit - p_ee
        dist = np.linalg.norm(err)
        scale = gain * min(dist, 0.5)
        tau = J_p.T @ err * scale
        tau -= 2.0 * x[env.NQ:]
        tau = np.clip(tau, ctrl_lo, ctrl_hi)
        U[k] = tau

        if fix_joint5_angle is not None:
            U[k, 5] = 300.0 * (fix_joint5_angle - x[:env.NQ][5]) - 30.0 * x[env.NQ:][5]
        x = env.step_from_state(x, U[k])

    if has_collision_ctrl:
        env.set_arm_collision(True)
    return U


def compute_joint1_backswing_trajectory(
    q1_current: float,
    qdot1_current: float,
    q1_hit: float,
    qdot1_hit: float,
    horizon: int,
    backswing_offset: float = -0.6,
    backswing_ratio: float = 0.35,
) -> np.ndarray:
    """生成关节1的"后摆→前挥"五次多项式轨迹。"""
    if horizon <= 0:
        return np.zeros(0)

    T = float(horizon)
    alpha = float(np.clip(backswing_ratio, 0.05, 0.95))
    q_mid = q1_current + backswing_offset

    a0 = q1_current
    a1 = qdot1_current * T

    alpha2, alpha3, alpha4, alpha5 = alpha**2, alpha**3, alpha**4, alpha**5

    A = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 3.0, 4.0, 5.0],
        [alpha2, alpha3, alpha4, alpha5],
        [2 * alpha, 3 * alpha2, 4 * alpha3, 5 * alpha4],
    ])
    b = np.array([
        q1_hit - a0 - a1,
        qdot1_hit * T - a1,
        q_mid - a0 - a1 * alpha,
        -a1,
    ])
    coeffs_high = np.linalg.solve(A, b)
    a2, a3, a4, a5 = coeffs_high[0], coeffs_high[1], coeffs_high[2], coeffs_high[3]

    q1_traj = np.zeros(horizon)
    for k in range(horizon):
        tau = (k + 1) / T
        q1_traj[k] = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3 + a4 * tau**4 + a5 * tau**5
    return q1_traj


def generate_backswing_warm_start(
    env: "RM65Env",
    x0: np.ndarray,
    p_hit: np.ndarray,
    v_hit_desired: np.ndarray,
    horizon: int,
    backswing_offset: float = 0,
    backswing_ratio: float = 0,
    kp: float = 150.0,
    kd: float = 15.0,
    fix_joint5_angle: float | None = None,
    n_des: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """生成带后摆的关节空间轨迹 + PD 跟踪初始控制序列。"""
    NQ = env.NQ
    NU = env.NU
    ctrl_lo = env.model.actuator_ctrlrange[:NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:NU, 1]

    if horizon <= 0:
        return np.zeros((0, NU)), np.zeros((0, NQ))

    q_hit = env.solve_ik(p_hit, q_init=x0[:NQ], max_iter=200, eps=1e-3)
    if fix_joint5_angle is not None:
        q_hit[5] = fix_joint5_angle

    if n_des is not None:
        wrist_joints = [3, 4, 5]
        for _ in range(20):
            env.set_arm_state(np.concatenate([q_hit, np.zeros(NQ)]))
            n_cur = env.get_ee_normal()
            n_err = n_cur - n_des
            err_norm = np.linalg.norm(n_err)
            if err_norm < 0.01:
                break
            J_omega = env.get_ee_jacr()
            nx, ny, nz = -n_cur[0], -n_cur[1], -n_cur[2]
            skew = np.array([[0, -nz, ny], [nz, 0, -nx], [-ny, nx, 0]])
            J_n = skew @ J_omega
            J_n_wrist = J_n[:, wrist_joints]
            dq_wrist = -np.linalg.lstsq(J_n_wrist, n_err, rcond=None)[0]
            dq_wrist *= min(1.0, 0.02 / (np.linalg.norm(dq_wrist) + 1e-12))
            q_hit[wrist_joints] += dq_wrist

    env.set_arm_state(np.concatenate([q_hit, np.zeros(NQ)]))
    J_p_hit = env.get_ee_jacp()
    qdot_hit = np.linalg.lstsq(J_p_hit, v_hit_desired, rcond=None)[0]
    max_qdot = 3.0
    qdot_norm = np.linalg.norm(qdot_hit)
    if qdot_norm > max_qdot:
        qdot_hit *= max_qdot / qdot_norm

    q1_traj = compute_joint1_backswing_trajectory(
        x0[0], x0[NQ], q_hit[0], qdot_hit[0],
        horizon,
        backswing_offset=backswing_offset,
        backswing_ratio=backswing_ratio,
    )

    q_des_traj = np.zeros((horizon, NQ))
    for j in range(NQ):
        if j == 0:
            q_des_traj[:, j] = q1_traj
        else:
            q_des_traj[:, j] = np.linspace(x0[j], q_hit[j], horizon)

    U = np.zeros((horizon, NU))
    x = x0.copy()

    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)

    for k in range(horizon):
        q_des_k = q_des_traj[k]
        if k < horizon - 1:
            qdot_des_k = (q_des_traj[k + 1] - q_des_k) / env.dt
        else:
            qdot_des_k = qdot_hit

        tau = kp * (q_des_k - x[:NQ]) + kd * (qdot_des_k - x[NQ:])
        if fix_joint5_angle is not None:
            tau[5] = 300.0 * (fix_joint5_angle - x[:NQ][5]) - 30.0 * x[NQ:][5]
        tau = np.clip(tau, ctrl_lo, ctrl_hi)
        U[k] = tau
        x = env.step_from_state(x, U[k])

    if has_collision_ctrl:
        env.set_arm_collision(True)
    return U, q_des_traj


def resample_control_sequence(U_old: np.ndarray, new_horizon: int) -> np.ndarray:
    """将旧控制序列重采样到新 horizon（线性插值）。"""
    old_horizon = len(U_old)
    if old_horizon == new_horizon:
        return U_old.copy()
    if old_horizon == 0:
        return np.zeros((new_horizon, U_old.shape[1]))
    n_u = U_old.shape[1]
    U_new = np.zeros((new_horizon, n_u))
    for k in range(new_horizon):
        t_frac = k / max(new_horizon - 1, 1) * (old_horizon - 1)
        idx_lo = int(np.floor(t_frac))
        idx_hi = min(idx_lo + 1, old_horizon - 1)
        alpha = t_frac - idx_lo
        U_new[k] = (1.0 - alpha) * U_old[idx_lo] + alpha * U_old[idx_hi]
    return U_new


def compute_r_schedule(
    steps_remaining: int,
    base_R: float,
    decay_ratio: float = 0.30,
    joint1_extra_decay: float = 10.0,
) -> np.ndarray:
    """生成 R 退火调度。"""
    if steps_remaining <= 0:
        return np.zeros((0, 6))
    decay_ratio = float(np.clip(decay_ratio, 0.0, 1.0))
    R_schedule = np.full((steps_remaining, 6), base_R)
    decay_start = int(steps_remaining * (1.0 - decay_ratio))
    if decay_start < steps_remaining:
        decay_len = steps_remaining - decay_start
        R_other = base_R * (1.0 - np.linspace(0.0, 1.0, decay_len))
        R_joint1 = base_R * (1.0 - np.linspace(0.0, 1.0, decay_len)) ** joint1_extra_decay
        R_schedule[decay_start:, 0] = R_joint1
        for j in range(1, 6):
            R_schedule[decay_start:, j] = R_other
    return R_schedule
