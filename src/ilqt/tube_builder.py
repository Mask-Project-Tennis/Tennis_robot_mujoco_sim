"""Tube 构建函数：球轨迹管道、击球时间窗口搜索、击球管道构建。

依赖 tube_types 中的数据结构，供仿真主脚本（V11）与真机 runner 共享复用。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.ilqt.tube_types import (
    TubeConfig,
    BallTrajectoryTube,
    HitWindow,
    HittingTube,
)
from src.tennis.hitting import find_hitting_point_physics

if TYPE_CHECKING:
    from src.sim.rm65_env import RM65Env
    from src.ilqt.robot_limits import RobotLimits


def build_ball_trajectory_tube(
    ball_positions: np.ndarray,
    ball_velocities: np.ndarray,
    dt: float,
    config: TubeConfig,
) -> BallTrajectoryTube:
    """将确定球轨迹转换为管道（仅保留位置和速度信息）。"""
    N = len(ball_positions)
    times = np.arange(N) * dt
    return BallTrajectoryTube(
        positions=ball_positions.copy(),
        velocities=ball_velocities.copy(),
        times=times.copy(),
    )


def search_hit_window(
    env: "RM65Env",
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    shoulder_pos: np.ndarray,
    workspace_radius: float,
    horizon: int,
    config: TubeConfig,
    ball_direction: str = "y",
    current_step: int = 0,
    robot_limits: "RobotLimits | None" = None,
    init_q: "np.ndarray | None" = None,
) -> HitWindow | None:
    """搜索候选击球时间窗口。

    先通过 find_hitting_point_physics 得到 best_k，然后以 best_k 为中心
    扩展 candidate_range 步的候选窗口，过滤不满足条件的步。

    返回的 k_candidates 已减去 current_step，为 iLQR 规划地平线内的相对步索引。

    Args:
        env: RM-65 环境实例。
        ball_pos: 球当前位置。
        ball_vel: 球当前速度。
        shoulder_pos: 肩关节世界坐标。
        workspace_radius: 工作空间半径。
        horizon: 规划步数上限。
        config: Tube 配置。
        ball_direction: 球飞来方向。
        current_step: 当前 MPC 绝对仿真步（用于将绝对步转为 iLQR 相对步）。

    Returns:
        HitWindow 或 None（若最佳击打球步不在工作空间内）。
    """
    # 1. 先找最佳击球步
    hit_info = find_hitting_point_physics(
        env, ball_pos, ball_vel, shoulder_pos, workspace_radius, horizon
    )
    if hit_info is None:
        return None

    best_k_abs = hit_info["k_hit"]
    dt = env.dt

    # 2. 窗口扩展范围（步数）——基于绝对步
    window_half_steps = int(round(config.window_half_ms / 1000.0 / dt))
    k_min_abs = max(1, best_k_abs - window_half_steps)
    k_max_abs = min(horizon, best_k_abs + window_half_steps)

    # 3. 预测球轨迹（仅预测窗口范围内的）
    n_pred = k_max_abs + 5
    ball_positions, ball_velocities = env.predict_ball_trajectory(
        ball_pos, ball_vel, n_pred
    )

    # 4. 筛选候选时刻（绝对步）
    candidates_k_abs: list[int] = []
    candidates_p: list[np.ndarray] = []
    candidates_v: list[np.ndarray] = []

    for k in range(k_min_abs, k_max_abs + 1):
        if k < 1 or k > len(ball_positions):
            continue
        p_ball = ball_positions[k - 1]  # predict_ball_trajectory 从 k=0 开始
        v_ball = ball_velocities[k - 1]

        dist = np.linalg.norm(p_ball - shoulder_pos)
        dz = p_ball[2] - shoulder_pos[2]

        # 可达性检查
        if not (dist < workspace_radius and p_ball[2] > 0.3 and -0.60 < dz < 0.55):
            continue

        # ball_direction="y" 时，球从 -Y 飞来，Y 坐标应 < shoulder_pos[1]（前方）
        if ball_direction == "y":
            dy = p_ball[1] - shoulder_pos[1]
            if dy > 0.6:
                continue
        else:
            dx = p_ball[0] - shoulder_pos[0]
            if dx > 0.6:
                continue

        # IK 可达性过滤：排除关节限制超限的点
        if robot_limits is not None and init_q is not None:
            q_ik = env.solve_ik(p_ball, q_init=init_q, max_iter=30, eps=2e-2)
            m_low_deg = (q_ik - robot_limits.q_lower) * 180.0 / np.pi
            m_up_deg = (robot_limits.q_upper - q_ik) * 180.0 / np.pi
            min_margin_deg = float(np.min(np.minimum(m_low_deg, m_up_deg)))
            if min_margin_deg < 3.0:
                continue

        candidates_k_abs.append(k)
        candidates_p.append(p_ball.copy())
        candidates_v.append(v_ball.copy())

    if len(candidates_k_abs) == 0:
        # 回退：至少包含 best_k_abs
        k = best_k_abs
        if 1 <= k <= len(ball_positions):
            p_ball = ball_positions[k - 1]
            v_ball = ball_velocities[k - 1]
            candidates_k_abs.append(k)
            candidates_p.append(p_ball.copy())
            candidates_v.append(v_ball.copy())

    # 5. 计算高斯衰减权重：exp(-0.5 * ((k - best_k) / half_window)^2)
    half_ws = max(window_half_steps, 1)
    k_arr = np.array(candidates_k_abs, dtype=np.float64)
    weights = np.exp(-0.5 * ((k_arr - best_k_abs) / half_ws) ** 2)
    weights /= weights.sum()  # 归一化

    # 6. 将绝对步转为 iLQR 相对步（减去 current_step）
    best_k_rel = best_k_abs - current_step
    k_candidates_rel = np.array(candidates_k_abs, dtype=int) - current_step

    return HitWindow(
        best_k=best_k_rel,
        k_candidates=k_candidates_rel,
        p_ball_candidates=np.array(candidates_p),
        v_ball_candidates=np.array(candidates_v),
        weights=weights,
    )


def build_hitting_tube(
    hit_window: HitWindow,
    desired_speed: float,
    hit_direction: np.ndarray,
    config: TubeConfig,
) -> HittingTube:
    """为每个候选时刻生成期望球拍状态。

    对每个候选时刻 k：
      - p_racket_des[k] = p_ball[k] + contact_offset * target_direction
      - v_racket_des[k] = desired_speed * target_direction
      - n_racket_des[k] = -normalize(v_ball[k])  （拍面朝向来球方向）

    Args:
        hit_window: 候选击球窗口。
        desired_speed: 期望击球速度（标量）。
        hit_direction: 期望击球方向，形状 (3,)。
        config: Tube 配置。

    Returns:
        HittingTube 实例。
    """
    M = len(hit_window.k_candidates)
    d_hat = hit_direction / (np.linalg.norm(hit_direction) + 1e-8)

    p_racket_des = np.zeros((M, 3))
    v_racket_des = np.zeros((M, 3))
    n_racket_des = np.zeros((M, 3))

    for i in range(M):
        v_ball = hit_window.v_ball_candidates[i]
        v_ball_norm = np.linalg.norm(v_ball)
        if v_ball_norm > 1e-6:
            n_des = -v_ball / v_ball_norm
        else:
            n_des = d_hat

        n_racket_des[i] = n_des
        p_racket_des[i] = hit_window.p_ball_candidates[i] + config.contact_offset * d_hat
        v_racket_des[i] = desired_speed * d_hat

    return HittingTube(
        k_candidates=hit_window.k_candidates.copy(),
        p_racket_des=p_racket_des,
        v_racket_des=v_racket_des,
        n_racket_des=n_racket_des,
        p_ball=hit_window.p_ball_candidates.copy(),
        v_ball=hit_window.v_ball_candidates.copy(),
        weights=hit_window.weights.copy(),
        best_k=hit_window.best_k,
    )
