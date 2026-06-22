"""轨迹回放碰撞管理模块。

从 V11/V12 回放代码中提取，统一管理：
- 碰撞窗口判定（should_enable_collision）
- 弹性反弹物理（compute_rebound_velocity / apply_elastic_rebound）
- 完整回放管线（replay_trajectory）

消除 9 个脚本中重复的回放逻辑，防止 V12 类似的 incomplete copy-paste 回归。
"""
from __future__ import annotations

import numpy as np
import mujoco

from src.sim.rm65_env import RM65Env


# ============================================================================
# 数据结构
# ============================================================================


class ReplayResult:
    """轨迹回放结果。"""

    def __init__(
        self,
        X_replay: np.ndarray,
        ball_replay: np.ndarray,
        rebound_applied: bool,
        contact_step: int,
    ) -> None:
        """初始化回放结果。

        Args:
            X_replay: (N+1, 12) 关节状态历史 [q, qdot]。
            ball_replay: (N+1, 3) 球位置历史。
            rebound_applied: 是否触发了弹性反弹。
            contact_step: 接触检测到的步数（-1 = 无接触）。
        """
        self.X_replay = X_replay
        self.ball_replay = ball_replay
        self.rebound_applied = rebound_applied
        self.contact_step = contact_step


# ============================================================================
# Cycle 1: should_enable_collision — 碰撞窗口纯函数
# ============================================================================


def should_enable_collision(
    step: int,
    hit_step: int,
    dist: float,
    rebound_applied: bool,
) -> bool:
    """判断回放某步是否应启用球-拍碰撞。

    碰撞窗口策略（与 V6-V11 live MPC 一致）：
    - hit_step < 0（无击球计划）→ 永不启用
    - 已触发反弹 → 关闭（避免重复检测）
    - k_hit_remaining ≤ 10 → 启用（最终接近阶段，无论距离）
    - k_hit_remaining ≤ 30 且球-拍距离 < 0.35m → 启用（中距离接近）
    - 否则 → 不启用

    Args:
        step: 当前回放步索引。
        hit_step: MPC 检测到的击球步（-1 = 未击中）。
        dist: 当前球-拍距离（米）。
        rebound_applied: 是否已触发过弹性反弹。

    Returns:
        是否启用碰撞。
    """
    if hit_step < 0 or rebound_applied:
        return False
    k_hit_remaining = max(0, hit_step - step)
    if k_hit_remaining <= 10:
        return True
    if k_hit_remaining <= 30 and dist < 0.35:
        return True
    return False


# ============================================================================
# Cycle 2-4 stubs（后续 Cycle 实现）
# ============================================================================


def compute_rebound_velocity(
    v_ball_pre: np.ndarray,
    v_ee: np.ndarray,
    n_racket: np.ndarray,
    e: float = 0.8,
) -> np.ndarray:
    """计算弹性反弹速度。

    使用刚体碰撞恢复系数模型：
        v_ball_new = v_ball_pre - (1 + e) * v_rel_n * n_hat

    其中 v_rel_n 为球-拍相对速度沿球拍法线的分量。

    Args:
        v_ball_pre: 碰撞前球速度，形状 (3,)。
        v_ee: 碰撞瞬间末端执行器（球拍）速度，形状 (3,)。
        n_racket: 球拍法向量（无需归一化），形状 (3,)。
        e: 恢复系数（0=完全非弹性, 1=完全弹性，默认 0.8）。

    Returns:
        反弹后球速度，形状 (3,)。
    """
    n_hat = n_racket / (np.linalg.norm(n_racket) + 1e-8)
    v_rel_n = float(np.dot(v_ball_pre - v_ee, n_hat))
    return v_ball_pre - (1 + e) * v_rel_n * n_hat


def apply_elastic_rebound(
    env: RM65Env,
    ball_vel_pre: np.ndarray,
    e: float = 0.8,
) -> np.ndarray | None:
    """检测球-拍接触并计算弹性反弹速度。

    遍历 MuJoCo 当前接触列表，查找 ball-racket 接触对。
    找到则用 compute_rebound_velocity 计算反弹速度，否则返回 None。

    Args:
        env: MuJoCo 环境实例（需已调用 mj_forward 更新接触列表）。
        ball_vel_pre: 碰撞前球速度，形状 (3,)。
        e: 恢复系数（默认 0.8）。

    Returns:
        反弹后球速度（形状 (3,)），无接触时返回 None。
    """
    ncon = env.data.ncon
    for ci in range(ncon):
        c = env.data.contact[ci]
        g1 = env.model.geom(c.geom1).name
        g2 = env.model.geom(c.geom2).name
        if ("ball" in g1 or "ball" in g2) and ("racket" in g1 or "racket" in g2):
            n_racket = env.get_ee_normal()
            v_ee = env.get_ee_vel()
            return compute_rebound_velocity(ball_vel_pre, v_ee, n_racket, e)
    return None


def replay_trajectory(
    env: RM65Env,
    U_arr: np.ndarray,
    init_q: np.ndarray,
    init_q_left: np.ndarray,
    p0: np.ndarray,
    v0: np.ndarray,
    hit_step: int,
    e: float = 0.8,
) -> ReplayResult:
    """完整回放管线：reset → 逐步碰撞管理 + 接触检测 + 弹性反弹 → 返回结果。

    与 V6-V11 live MPC 执行完全相同的碰撞窗口策略，确保回放中
    球-拍接触物理与实际运行一致。

    Args:
        env: MuJoCo 环境实例。
        U_arr: 控制序列，形状 (N, NU)。
        init_q: 右臂初始关节角度，形状 (6,)。
        init_q_left: 左臂初始关节角度，形状 (6,)。
        p0: 球初始位置，形状 (3,)。
        v0: 球初始速度，形状 (3,)。
        hit_step: MPC 检测到的击球步（-1 = 未击中）。
        e: 恢复系数（默认 0.8）。

    Returns:
        ReplayResult 包含 X_replay、ball_replay、rebound_applied、contact_step。
    """
    # 重置到初始状态
    env.reset(init_q)
    env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
    env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
    env.update_kinematics()
    env.set_ball_state(p0, v0)

    # 初始化碰撞状态（禁用，同时初始化 _arm_contype_save）
    if hasattr(env, "set_arm_collision"):
        env.set_arm_collision(False)

    X_replay: list[np.ndarray] = [env.get_arm_state().copy()]
    ball_replay: list[np.ndarray] = [env.get_ball_pos().copy()]
    rebound_applied = False
    contact_step = -1

    for i, u_cmd in enumerate(U_arr):
        # 碰撞窗口管理
        ball_pos_rp = env.get_ball_pos()
        racket_pos_rp = env.get_ee_pos()
        dist_rp = float(np.linalg.norm(racket_pos_rp - ball_pos_rp))

        enable = should_enable_collision(i, hit_step, dist_rp, rebound_applied)
        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(enable)

        ball_vel_pre = env.get_ball_vel().copy() if enable else np.zeros(3)
        env.step(u_cmd)

        # 接触检测 + 弹性反弹
        if enable and not rebound_applied and hit_step >= 0:
            v_rebound = apply_elastic_rebound(env, ball_vel_pre, e)
            if v_rebound is not None:
                env.set_ball_vel(v_rebound)
                rebound_applied = True
                contact_step = i

        X_replay.append(env.get_arm_state().copy())
        ball_replay.append(env.get_ball_pos().copy())

    if hasattr(env, "set_arm_collision"):
        env.set_arm_collision(True)

    return ReplayResult(
        X_replay=np.array(X_replay),
        ball_replay=np.array(ball_replay),
        rebound_applied=rebound_applied,
        contact_step=contact_step,
    )
