"""位置模式下的雅可比转置初始控制序列生成。

力矩模式的 JT 初始控制（compute_jacobian_init_control）输出力矩 τ = J^T * err * scale。
位置模式需要输出目标角度 q_desired，本模块提供角度增量版本的 JT 初始控制。

阶段 3 Task 3.8：维度无关化 + 绑定环境（BoundRobotEnv 族）适配——
关节限位优先按 ``env.binding`` 的名称地址解析（支持 free joint 前置/
执行器乱序的模型），旧 RM65 环境保持前缀布局路径原样。
"""

from __future__ import annotations

import logging

import numpy as np
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.ilqt.robot_env_protocol import RobotEnv

logger = logging.getLogger(__name__)


def _is_bound_env(env: RobotEnv) -> bool:
    """判定是否为绑定环境（BoundRobotEnv 族，阶段 3 新路径）。

    绑定环境按 ``ModelBinding`` 名称地址访问状态/执行器（R1D n=7、
    free joint 前置等），RM65Env/PlanningEnv 无 ``binding`` 属性走旧
    前缀布局路径。本谓词是 3.8 的 R1D/RM65 策略分派依据（环境能力
    判定，非维数猜测）。
    """
    return hasattr(env, "binding")


def _get_jnt_range(env: RobotEnv) -> tuple[np.ndarray, np.ndarray]:
    """获取主动臂关节限位 (jnt_lo, jnt_hi)，各形状 (NU,)。

    绑定环境：按 ``binding.actuator_ids`` 解析执行器→关节地址（不限
    前缀布局），未声明限位的关节放宽为 ±inf（不硬裁剪）；
    旧 RM65 环境：保持前缀 ``actuator_trnid[:NU]`` 路径原样。
    """
    jnt_lo = np.zeros(env.NU)
    jnt_hi = np.zeros(env.NU)
    if _is_bound_env(env):
        binding = env.binding  # type: ignore[attr-defined]
        for i, act_id in enumerate(binding.actuator_ids):
            jnt_id = env.model.actuator_trnid[act_id, 0]
            if env.model.jnt_limited[jnt_id]:
                jnt_lo[i] = env.model.jnt_range[jnt_id, 0]
                jnt_hi[i] = env.model.jnt_range[jnt_id, 1]
            else:
                jnt_lo[i] = -np.inf
                jnt_hi[i] = np.inf
        return jnt_lo, jnt_hi
    for i in range(env.NU):
        jnt_id = env.model.actuator_trnid[i, 0]
        jnt_lo[i] = env.model.jnt_range[jnt_id, 0]
        jnt_hi[i] = env.model.jnt_range[jnt_id, 1]
    return jnt_lo, jnt_hi


def solve_ik_checked(
    env: RobotEnv,
    target_pos: np.ndarray,
    q_init: np.ndarray | None = None,
    robot_limits: Any = None,
    max_iter: int = 100,
    eps: float = 1e-3,
    pos_tol: float = 0.02,
    label: str = "",
) -> np.ndarray:
    """带误差与限位校验的逆运动学（阶段 3 Task 3.8）。

    与裸 ``env.solve_ik`` 的差异：求解后显式 ``set_arm_state`` 复核
    末端误差，并在提供 ``robot_limits`` 时校验关节限位——失败抛出
    ``ValueError``（含目标/误差/裕度报告），而不是把坏初值交给下游
    warm start / 控制指令继续。

    Args:
        env: 满足 RobotEnv 协议的环境。
        target_pos: 目标末端位置 (3,)。
        q_init: IK 初值 (NQ,)；None 用零位。
        robot_limits: 关节限位（q_lower/q_upper）；None 跳过限位校验。
        max_iter: IK 最大迭代次数。
        eps: IK 收敛阈值（m）。
        pos_tol: 求解后允许的最大末端误差（m）。
        label: 报告用上下文标签（如 "随挥" / "工作空间中心"）。

    Returns:
        校验通过的关节角 (NQ,)。

    Raises:
        ValueError: 末端误差超 pos_tol 或关节超出限位。
    """
    nq = env.NQ
    q = env.solve_ik(target_pos, q_init=q_init, max_iter=max_iter, eps=eps)
    env.set_arm_state(np.concatenate([q, np.zeros(nq)]))
    err = float(np.linalg.norm(env.get_ee_pos() - np.asarray(target_pos)))
    if not np.isfinite(err) or err > pos_tol:
        raise ValueError(
            f"{label}IK 未收敛：目标 {np.round(np.asarray(target_pos), 4).tolist()}，"
            f"末端误差 {err:.4f} m 超容差 {pos_tol:.4f} m"
            f"（q_init={'零位' if q_init is None else '给定初值'}，"
            f"max_iter={max_iter}）——拒绝带坏初值继续"
        )
    if robot_limits is not None:
        below = q < np.asarray(robot_limits.q_lower)
        above = q > np.asarray(robot_limits.q_upper)
        if np.any(below) or np.any(above):
            bad = np.flatnonzero(below | above).tolist()
            raise ValueError(
                f"{label}IK 解超关节限位：关节索引 {bad}，"
                f"q={np.round(q, 4).tolist()}，"
                f"下界={np.round(np.asarray(robot_limits.q_lower), 4).tolist()}，"
                f"上界={np.round(np.asarray(robot_limits.q_upper), 4).tolist()}"
                "——拒绝带坏初值继续"
            )
    return q


def compute_joint1_backswing_trajectory(
    q1_current: float,
    qdot1_current: float,
    q1_hit: float,
    qdot1_hit: float,
    horizon: int,
    backswing_offset: float = -0.6,
    backswing_ratio: float = 0.35,
) -> np.ndarray:
    """生成关节1的"后摆→前挥"五次多项式轨迹。

    共享实现，供位置模式（jt_init）和力矩模式（mpc_helpers）复用。

    Args:
        q1_current: 当前关节1角度。
        qdot1_current: 当前关节1角速度。
        q1_hit: 击球时刻关节1角度。
        qdot1_hit: 击球时刻关节1角速度。
        horizon: 轨迹步数。
        backswing_offset: 后摆偏移量（弧度）。
        backswing_ratio: 后摆占比（0~1）。

    Returns:
        关节1轨迹，形状 (horizon,)。
    """
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


def compute_jacobian_init_control_position(
    env: RobotEnv,
    x0: np.ndarray,
    p_hit: np.ndarray,
    horizon: int,
    gain: float = 0.5,
    fix_joint5_angle: float | None = None,
    damp: float = 1e-6,
) -> np.ndarray:
    """位置模式初始控制：JT → 角度增量序列。

    输出 q_desired[k]，使末端逐步趋近 p_hit。每步：
      1. set_arm_state([q, zeros]) → 计算 p_ee 和 J_p
      2. dq = J^T * solve(J*J^T + λI, err) * gain
      3. q_desired = q + dq，clip 到 jnt_range
      4. step_from_state 物理仿真更新状态
      5. q = 仿真后真实角度

    Args:
        env: RobotEnv 实例（需已配置位置模式）。
        x0: 初始右臂状态 [q, qdot]，形状 (2*NQ,)。
        p_hit: 目标击打点位置，形状 (3,)。
        horizon: 控制序列长度。
        gain: 角度增量步长（弧度），默认 0.5。
        fix_joint5_angle: 若非 None，第 6 关节（索引 5）固定为该角度。
            仅 RM65（NQ=6）可用；其他维数明确拒绝。
        damp: 阻尼因子，防止雅可比奇异。

    Returns:
        控制序列 U，形状 (horizon, NU)，每行为 q_desired。

    Raises:
        ValueError: fix_joint5_angle 非 None 但环境维数不是 6。
    """
    if fix_joint5_angle is not None and env.NQ != 6:
        raise ValueError(
            f"fix_joint5 是 RM65（NQ=6）专属策略，当前环境 NQ={env.NQ}，"
            "R1D 不使用 fix_joint5（Task 3.8 策略决定）"
        )
    NQ = env.NQ
    NU = env.NU
    U = np.zeros((horizon, NU))
    x = x0.copy()

    jnt_lo, jnt_hi = _get_jnt_range(env)

    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)  # type: ignore[attr-defined]

    for k in range(horizon):
        env.set_arm_state(x)
        p_ee = env.get_ee_pos()
        J_p = env.get_ee_jacp()

        err = p_hit - p_ee
        JJT = J_p @ J_p.T + damp * np.eye(3)
        dq = J_p.T @ np.linalg.solve(JJT, err) * gain

        q_current = x[:NQ]
        q_desired = q_current + dq
        q_desired = np.clip(q_desired, jnt_lo, jnt_hi)

        if fix_joint5_angle is not None:
            q_desired[5] = fix_joint5_angle

        U[k] = q_desired
        x = env.step_from_state(x, q_desired)

    if has_collision_ctrl:
        env.set_arm_collision(True)  # type: ignore[attr-defined]
    return U


def solve_hit_pose(
    env: RobotEnv,
    p_hit: np.ndarray,
    q_init: np.ndarray,
    fix_joint5_angle: float | None,
    n_des: np.ndarray | None,
) -> np.ndarray:
    """求解击球时刻的关节角度（IK + 可选腕部法向量对齐）。

    先用阻尼最小二乘 IK 把末端对准 p_hit；若提供 fix_joint5_angle 则强制
    覆盖第 5 关节；若提供 n_des 则通过 20 次迭代的腕部伪逆对齐拍面法向量。

    Args:
        env: 满足 RobotEnv 协议的环境（需提供 solve_ik/set_arm_state/
            get_ee_normal/get_ee_jacr）。
        p_hit: 目标击打点位置，形状 (3,)。
        q_init: IK 初值，形状 (NQ,)。
        fix_joint5_angle: 若非 None，强制第 5 关节为该角度。
        n_des: 期望拍面法向量，形状 (3,)；None 跳过法向量对齐。

    Returns:
        q_hit: 击球时刻关节角度，形状 (NQ,)。
    """
    NQ = env.NQ
    q_hit = env.solve_ik(p_hit, q_init=q_init, max_iter=200, eps=1e-3)
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

    return q_hit


def solve_hit_velocity(
    env: RobotEnv,
    q_hit: np.ndarray,
    v_hit_desired: np.ndarray,
    max_qdot: float = 3.0,
) -> np.ndarray:
    """通过雅可比伪逆求解击球时刻关节速度。

    在 q_hit 处计算位置雅可比 J_p，用最小二乘解 J_p @ qdot ≈ v_hit_desired，
    并将结果范数裁剪到 max_qdot 以内。

    Args:
        env: 满足 RobotEnv 协议的环境。
        q_hit: 击球时刻关节角度，形状 (NQ,)。
        v_hit_desired: 期望末端线速度，形状 (3,)。
        max_qdot: 关节速度范数上限（弧度/秒）。

    Returns:
        qdot_hit: 击球时刻关节速度，形状 (NQ,)。
    """
    NQ = env.NQ
    env.set_arm_state(np.concatenate([q_hit, np.zeros(NQ)]))
    J_p_hit = env.get_ee_jacp()
    qdot_hit = np.linalg.lstsq(J_p_hit, v_hit_desired, rcond=None)[0]
    qdot_norm = np.linalg.norm(qdot_hit)
    if qdot_norm > max_qdot:
        qdot_hit *= max_qdot / qdot_norm
    return qdot_hit


def generate_backswing_warm_start_position(
    env: RobotEnv,
    x0: np.ndarray,
    p_hit: np.ndarray,
    v_hit_desired: np.ndarray,
    horizon: int,
    backswing_offset: float = 0.0,
    backswing_ratio: float = 0.0,
    fix_joint5_angle: float | None = None,
    n_des: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """位置模式的后摆 warm-start：直接输出角度轨迹作为控制序列。

    与力矩版 generate_backswing_warm_start 对应，但输出 q_desired 而非 PD 力矩。
    位置模式下无需 PD 跟踪仿真，直接输出 q_des_traj 即可，
    因为 MuJoCo 内部 PD 执行器会自动跟踪。

    Args:
        env: RobotEnv 实例（需已配置位置模式）。
        x0: 初始右臂状态 [q, qdot]，形状 (12,)。
        p_hit: 目标击打点位置，形状 (3,)。
        v_hit_desired: 期望击球速度，形状 (3,)。
        horizon: 控制序列长度。
        backswing_offset: 关节 1 后摆偏移量（弧度）。
        backswing_ratio: 后摆占比（0~1）。
        fix_joint5_angle: 若非 None，第 5 关节固定为该角度。
        n_des: 期望法向量，形状 (3,)。

    Returns:
        (U, q_des_traj): U 形状 (horizon, NQ)，q_des_traj 形状 (horizon, NQ)。

    Raises:
        ValueError: 环境维数不是 6（关节 0 五次多项式后摆是 RM65 专属
            策略，"索引 0 必是后摆轴"不继承到其他机器人）。
    """
    if env.NQ != 6:
        raise ValueError(
            f"后摆 warm-start（关节 0 五次多项式）是 RM65（NQ=6）专属策略，"
            f"当前环境 NQ={env.NQ}；R1D 默认全关节 JT/IK warm-start"
            "（use_backswing=False，Task 3.8 策略决定）"
        )
    NQ = env.NQ
    NU = env.NU

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
    q_des_traj[:, 0] = q1_traj
    for j in range(1, NQ):
        q_des_traj[:, j] = np.linspace(x0[j], q_hit[j], horizon)

    if fix_joint5_angle is not None:
        q_des_traj[:, 5] = fix_joint5_angle

    jnt_lo, jnt_hi = _get_jnt_range(env)
    q_des_traj = np.clip(q_des_traj, jnt_lo, jnt_hi)

    return q_des_traj.copy(), q_des_traj


def fix_joint5_control_trajectory_position(
    U: np.ndarray,
    q_fixed: float,
) -> np.ndarray:
    """位置模式下将控制序列的第 5 关节替换为固定角度。

    力矩版用 PD 力矩保持，位置版直接设 q_desired。

    Args:
        U: 控制序列，形状 (horizon, 6)。
        q_fixed: 固定的关节 5 角度。

    Returns:
        修改后的 U（副本）。

    Raises:
        ValueError: U 维数不是 6（fix_joint5 为 RM65 专属策略）。
    """
    if U.shape[-1] != 6:
        raise ValueError(
            f"fix_joint5 是 RM65（NU=6）专属策略，收到控制维数 "
            f"{U.shape[-1]}（Task 3.8 策略决定）"
        )
    U = U.copy()
    U[:, 5] = q_fixed
    return U
