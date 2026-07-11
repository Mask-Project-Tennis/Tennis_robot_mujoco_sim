"""Warm-start 辅助函数测试。

覆盖：
- 五次多项式后摆轨迹的特征化行为（Phase A）
- jt_init 与 mpc_helpers 多项式实现的合并安全性证明（Phase A）
- _solve_hit_pose / _solve_hit_velocity 提取函数（Phase B）
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ilqt.mpc_helpers import compute_joint1_backswing_trajectory
from src.ilqt.planning_env import PlanningEnv
from src.robot.constants import INIT_Q

DT = 0.005
INIT_Q_RIGHT = np.asarray(INIT_Q)[:6]

PARAM_SETS = [
    dict(q1_current=0.5, qdot1_current=0.1, q1_hit=1.2, qdot1_hit=2.0, horizon=50),
    dict(q1_current=0.0, qdot1_current=0.0, q1_hit=0.8, qdot1_hit=1.5, horizon=30),
    dict(
        q1_current=1.0,
        qdot1_current=-0.2,
        q1_hit=0.5,
        qdot1_hit=3.0,
        horizon=100,
        backswing_offset=-0.4,
        backswing_ratio=0.25,
    ),
    dict(q1_current=0.3, qdot1_current=0.05, q1_hit=0.9, qdot1_hit=0.8, horizon=1),
]


@pytest.mark.parametrize("params", PARAM_SETS)
def test_quintic_trajectory_properties(params: dict) -> None:
    """五次多项式轨迹满足形状约束。"""
    traj = compute_joint1_backswing_trajectory(**params)
    assert traj.shape == (params["horizon"],)


def test_quintic_horizon0_returns_empty() -> None:
    """horizon=0 返回空数组。"""
    traj = compute_joint1_backswing_trajectory(0, 0, 1, 1, 0)
    assert traj.shape == (0,)


def test_quintic_endpoint_meets_q_hit() -> None:
    """轨迹末步接近 q1_hit。"""
    traj = compute_joint1_backswing_trajectory(0.5, 0.1, 1.2, 2.0, 50)
    assert abs(traj[-1] - 1.2) < 0.15


def test_quintic_backswing_ratio_edges() -> None:
    """backswing_ratio 在 clip 边界 (0.05, 0.95) 不 crash。"""
    for ratio in [0.05, 0.95]:
        traj = compute_joint1_backswing_trajectory(
            0.5, 0.1, 1.2, 2.0, 30, backswing_ratio=ratio
        )
        assert traj.shape == (30,)


def test_jt_init_equals_mpc_helpers_polynomial() -> None:
    """jt_init 和 mpc_helpers 的五次多项式实现输出一致 — 合并安全性证明。

    Phase C 后 mpc_helpers 直接 re-export jt_init 的实现（同一对象），
    此处仍用输出比对（atol=1e-12）保证行为等价的回归护栏。
    """
    from src.ilqt.jt_init import compute_joint1_backswing_trajectory as jt_impl

    params = dict(
        q1_current=0.5, qdot1_current=0.1, q1_hit=1.2, qdot1_hit=2.0, horizon=50
    )
    traj_jt = jt_impl(**params)
    traj_mpc = compute_joint1_backswing_trajectory(**params)
    np.testing.assert_allclose(traj_jt, traj_mpc, atol=1e-12)
    # 去重后应为同一对象（mpc_helpers 从 jt_init re-export）
    assert compute_joint1_backswing_trajectory is jt_impl


# ── Phase B: _solve_hit_pose / _solve_hit_velocity 测试 ──


@pytest.fixture
def planning_env() -> PlanningEnv:
    """力矩模式 PlanningEnv fixture。"""
    env = PlanningEnv(dt=DT)
    env.reset(INIT_Q_RIGHT)
    return env


def test_solve_hit_pose_fk_converges(planning_env: PlanningEnv) -> None:
    """_solve_hit_pose 返回的 q_hit 做 FK 接近 p_hit。"""
    from src.ilqt.jt_init import _solve_hit_pose

    p_hit = np.array([0.3, -0.5, 1.2])
    q_init = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45])
    q_hit = _solve_hit_pose(planning_env, p_hit, q_init, fix_joint5_angle=None, n_des=None)
    planning_env.set_arm_state(np.concatenate([q_hit, np.zeros(planning_env.NQ)]))
    p_actual = planning_env.get_ee_pos()
    assert np.linalg.norm(p_actual - p_hit) < 0.01


def test_solve_hit_velocity_jacobian_consistent(planning_env: PlanningEnv) -> None:
    """_solve_hit_velocity 返回的 qdot 满足 J@qdot ≈ v_desired。

    v_desired 选小量（|v|≈0.34）避免触发 max_qdot=3.0 的范数裁剪，
    从而隔离测试雅可比伪逆的映射一致性。
    """
    from src.ilqt.jt_init import _solve_hit_velocity

    q_hit = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45])
    v_desired = np.array([0.3, 0.15, 0.0])
    qdot = _solve_hit_velocity(planning_env, q_hit, v_desired)
    planning_env.set_arm_state(np.concatenate([q_hit, np.zeros(planning_env.NQ)]))
    J = planning_env.get_ee_jacp()
    v_actual = J @ qdot
    assert np.linalg.norm(v_actual - v_desired) < 0.1  # 无裁剪时伪逆近似精确
