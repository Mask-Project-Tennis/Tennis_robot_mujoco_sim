"""MPC 规划辅助工具函数集合。

包含 iLQR 初始控制 dispatch（力矩/位置双模式）、fix_joint5 力矩版工具、
后摆 warm-start 力矩版、控制序列重采样、R 退火调度等。
供仿真主脚本（V11）与真机 runner 共享复用。

阶段 3 Task 3.8：多机器人化——
- 控制界统一经 ``env.control_range``（RobotEnv 协议，按执行器绑定序，
  RM65 为前缀序、绑定环境按 binding.actuator_ids）取得；
- fix_joint5 / 后摆（关节 0 五次多项式）收敛为 RM65（NQ=6）专属，
  其他维数（R1D n=7）明确拒绝（ValueError），不静默落到错误关节索引；
- R 退火 ``compute_r_schedule`` 支持任意关节数与均匀模式（R1D 默认
  均匀关节 R 比例，不继承关节 1 特殊衰减）；
- ``validate_workspace_against_model``：工作空间中心/半径对模型校验
  （中心 IK 可达 + 限位内），中心/半径显式来自当前侧配置，不读 RM65
  SHOULDER_POS/WORKSPACE_RADIUS。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.ilqt.jt_init import (
    _is_bound_env,
    solve_hit_pose,
    solve_hit_velocity,
    compute_joint1_backswing_trajectory,
    compute_jacobian_init_control_position,
    generate_backswing_warm_start_position,
    fix_joint5_control_trajectory_position,
)

if TYPE_CHECKING:
    from src.ilqt.robot_env_protocol import RobotEnv
    from src.sim.rm65_env import RM65Env


# 力矩增益→位置增益的缩放因子
# 力矩版 gain=50 对应力矩 ~50Nm，位置版 gain=0.5 对应弧度增量 ~0.5rad
_GAIN_TORQUE_TO_POSITION = 0.01


def _reject_rm65_only_policy(env: "RobotEnv", policy_name: str) -> None:
    """RM65 专属策略在非 RM65 维数上的统一拒绝（Task 3.8 策略决定）。

    Args:
        env: 规划环境。
        policy_name: 策略名（用于错误消息）。

    Raises:
        ValueError: 绑定环境（BoundRobotEnv 族，R1D 路径）上使用了
            RM65 专属策略。
    """
    if _is_bound_env(env):
        raise ValueError(
            f"{policy_name} 是 RM65 专属策略，绑定环境（R1D，NQ={env.NQ}）"
            "不使用该策略（Task 3.8 策略决定：R1D 不使用 fix_joint5、"
            "不继承'索引 0 必是后摆轴'与关节 1 特殊 R 衰减）"
        )


def _jt_init_dispatch(
    env: "RM65Env",
    x0: np.ndarray,
    p_hit: np.ndarray,
    horizon: int,
    gain: float = 50.0,
    fix_joint5_angle: float | None = None,
) -> np.ndarray:
    """根据 env.actuator_mode 选择力矩版或位置版 JT 初始控制。"""
    if fix_joint5_angle is not None:
        _reject_rm65_only_policy(env, "fix_joint5")
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
    """根据 env.actuator_mode 选择力矩版或位置版后摆 warm-start。

    Raises:
        ValueError: 绑定环境（R1D）不继承关节 0 后摆策略。
    """
    _reject_rm65_only_policy(env, "后摆 warm-start（关节 0 五次多项式）")
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
    """根据 env.actuator_mode 选择力矩版或位置版 fix_joint5。

    Raises:
        ValueError: 绑定环境（R1D）不使用 fix_joint5。
    """
    _reject_rm65_only_policy(env, "fix_joint5")
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
    """单步 fix_joint5：力矩模式用 PD 保持，位置模式直接设目标角度。

    Raises:
        ValueError: 绑定环境（R1D）不使用 fix_joint5。
    """
    _reject_rm65_only_policy(env, "fix_joint5")
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
    """将第 6 关节（索引 5）的控制力矩替换为 PD 保持力矩。

    Raises:
        ValueError: 控制维数不是 6（fix_joint5 为 RM65 专属策略）。
    """
    if u.shape[-1] != 6:
        raise ValueError(
            f"fix_joint5 是 RM65（NU=6）专属策略，收到控制维数 "
            f"{u.shape[-1]}（Task 3.8 策略决定）"
        )
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
    """将整个控制序列的第 6 关节替换为 PD 保持力矩。

    Raises:
        ValueError: 控制维数不是 6（fix_joint5 为 RM65 专属策略）。
    """
    if U.shape[-1] != 6 or env.NQ != 6:
        raise ValueError(
            f"fix_joint5 是 RM65（NQ/NU=6）专属策略，收到环境 NQ={env.NQ}、"
            f"控制维数 {U.shape[-1]}（Task 3.8 策略决定）"
        )
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
    """基于雅可比转置法计算初始控制序列（维度无关）。

    控制界从环境 ``control_range`` 取得（RM65 前缀序 / 绑定环境按
    binding.actuator_ids 序），不读模型前缀切片。

    Raises:
        ValueError: fix_joint5_angle 非 None 但环境维数不是 6。
    """
    if fix_joint5_angle is not None and env.NQ != 6:
        raise ValueError(
            f"fix_joint5 是 RM65（NQ=6）专属策略，当前环境 NQ={env.NQ}，"
            "R1D 不使用 fix_joint5（Task 3.8 策略决定）"
        )
    U = np.zeros((horizon, env.NU))
    x = x0.copy()
    ctrl_range = np.asarray(env.control_range, dtype=np.float64)
    ctrl_lo = ctrl_range[:, 0]
    ctrl_hi = ctrl_range[:, 1]

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
    """生成带后摆的关节空间轨迹 + PD 跟踪初始控制序列。

    Raises:
        ValueError: 环境维数不是 6（关节 0 五次多项式后摆是 RM65 专属
            策略，不继承到其他机器人）。
    """
    if env.NQ != 6:
        raise ValueError(
            f"后摆 warm-start（关节 0 五次多项式）是 RM65（NQ=6）专属策略，"
            f"当前环境 NQ={env.NQ}；R1D 默认全关节 JT/IK warm-start"
            "（use_backswing=False，Task 3.8 策略决定）"
        )
    NQ = env.NQ
    NU = env.NU
    ctrl_range = np.asarray(env.control_range, dtype=np.float64)
    ctrl_lo = ctrl_range[:, 0]
    ctrl_hi = ctrl_range[:, 1]

    if horizon <= 0:
        return np.zeros((0, NU)), np.zeros((0, NQ))

    q_hit = solve_hit_pose(env, p_hit, x0[:NQ], fix_joint5_angle, n_des)
    qdot_hit = solve_hit_velocity(env, q_hit, v_hit_desired)

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
    n_joints: int = 6,
    uniform_decay: bool = False,
) -> np.ndarray:
    """生成 R 退火调度（维度无关）。

    RM65 缺省（n_joints=6, uniform_decay=False）与旧实现逐值一致：
    击球前 decay_ratio 比例的步段内控制代价衰减到 0，关节 1（索引 0）
    额外按 joint1_extra_decay 次幂加速衰减。

    R1D 口径（Task 3.8 策略决定）：uniform_decay=True 时所有关节均匀
    衰减（无关节 1 特殊衰减），n_joints 取环境 NU。

    Args:
        steps_remaining: 剩余步数。
        base_R: 基础控制代价。
        decay_ratio: 末段衰减步数占比（0~1）。
        joint1_extra_decay: RM65 关节 1 额外衰减幂次（uniform 模式无效）。
        n_joints: 关节数（默认 6 = RM65）。
        uniform_decay: True = 全关节均匀衰减（R1D 默认）。

    Returns:
        R_schedule，形状 (steps_remaining, n_joints)。
    """
    if steps_remaining <= 0:
        return np.zeros((0, n_joints))
    decay_ratio = float(np.clip(decay_ratio, 0.0, 1.0))
    R_schedule = np.full((steps_remaining, n_joints), base_R)
    decay_start = int(steps_remaining * (1.0 - decay_ratio))
    if decay_start < steps_remaining:
        decay_len = steps_remaining - decay_start
        ramp = 1.0 - np.linspace(0.0, 1.0, decay_len)
        R_other = base_R * ramp
        if uniform_decay:
            for j in range(n_joints):
                R_schedule[decay_start:, j] = R_other
        else:
            R_joint1 = base_R * ramp ** joint1_extra_decay
            R_schedule[decay_start:, 0] = R_joint1
            for j in range(1, n_joints):
                R_schedule[decay_start:, j] = R_other
    return R_schedule


def validate_workspace_against_model(
    env: "RobotEnv",
    workspace_center: np.ndarray,
    workspace_radius: float,
    robot_limits: object | None = None,
    ik_max_iter: int = 100,
    ik_eps: float = 5.0e-3,
    pos_tol: float = 0.05,
) -> None:
    """工作空间中心/半径对模型校验（阶段 3 Task 3.8）。

    中心/半径显式来自当前侧 task 配置（侧别经 profile/model_path 表达），
    本函数把两者对照模型可达性校验，防止"左配置配右臂模型"之类的
    静默错配——不能读 RM65 SHOULDER_POS/WORKSPACE_RADIUS 常量。

    校验内容：
    1. 半径为正有限值，中心为有限 (3,)；
    2. 中心点 IK 可达（末端误差 < pos_tol）；
    3. 提供限位时，IK 解在关节限位内。

    Args:
        env: 规划环境（以 reset 后的 init_q 为 IK 初值）。
        workspace_center: 工作区中心 (3,)，世界系。
        workspace_radius: 工作区半径（m）。
        robot_limits: 关节限位（q_lower/q_upper）；None 跳过限位校验。
        ik_max_iter: IK 最大迭代次数。
        ik_eps: IK 收敛阈值（m）。
        pos_tol: 允许的最大末端误差（m）。

    Raises:
        ValueError: 半径非正/中心非法/中心不可达/IK 解超限位。
    """
    center = np.asarray(workspace_center, dtype=np.float64)
    if center.shape != (3,) or not bool(np.all(np.isfinite(center))):
        raise ValueError(
            f"工作空间中心应为有限 (3,)，收到 {center.shape}："
            f"{np.round(center, 4).tolist()}"
        )
    radius = float(workspace_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            f"工作空间半径必须为正有限值，收到 {radius}（m）"
        )
    from src.ilqt.jt_init import solve_ik_checked

    x_restore = env.get_arm_state().copy()
    q_init = x_restore[: env.NQ].copy()
    try:
        solve_ik_checked(
            env, center, q_init=q_init, robot_limits=robot_limits,
            max_iter=ik_max_iter, eps=ik_eps, pos_tol=pos_tol,
            label="工作空间中心",
        )
    except ValueError as exc:
        raise ValueError(
            f"工作空间中心/半径对模型校验失败（中心 "
            f"{np.round(center, 4).tolist()}，半径 {radius} m）：{exc}"
        ) from exc
    finally:
        # 校验过程会 set_arm_state 到 IK 试解，恢复原状态
        env.set_arm_state(x_restore)
