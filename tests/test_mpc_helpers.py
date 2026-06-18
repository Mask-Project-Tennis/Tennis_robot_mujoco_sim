"""mpc_helpers dispatch 双分支测试。

覆盖 _jt_init_dispatch / _fix_joint5_dispatch 的力矩模式（actuator_mode == 0）
与位置模式（actuator_mode == 1）两条分支，确保双模式执行器 dispatch 正确。
"""

from __future__ import annotations

import numpy as np

from src.ilqt.mpc_helpers import _fix_joint5_dispatch, _jt_init_dispatch
from src.ilqt.planning_env import PlanningEnv

DT = 0.005
INIT_Q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45])


def _make_env_torque() -> PlanningEnv:
    """力矩模式 PlanningEnv。"""
    env = PlanningEnv(dt=DT)
    env.reset(INIT_Q)
    return env


def _make_env_position() -> PlanningEnv:
    """位置模式 PlanningEnv。"""
    env = PlanningEnv(dt=DT)
    env.configure_actuator_mode(
        "position",
        kp=np.array([200, 200, 100, 50, 50, 20], dtype=float),
        kd=np.array([20, 20, 10, 5, 5, 2], dtype=float),
    )
    env.configure_feedforward(True)
    env.reset(INIT_Q)
    return env


def test_jt_init_dispatch_torque_mode() -> None:
    """力矩模式 dispatch 返回非零力矩序列，shape=(horizon, 6)。"""
    env = _make_env_torque()
    x0 = env.get_arm_state()
    p_hit = np.array([0.3, -0.5, 1.2])
    horizon = 20
    U = _jt_init_dispatch(env, x0, p_hit, horizon, gain=50.0)
    assert U.shape == (horizon, 6)
    assert np.any(np.abs(U) > 1e-6), "力矩序列不应全零"


def test_jt_init_dispatch_position_mode() -> None:
    """位置模式 dispatch 返回期望角度序列，值在 [-π, π] 范围内。"""
    env = _make_env_position()
    x0 = env.get_arm_state()
    p_hit = np.array([0.3, -0.5, 1.2])
    horizon = 20
    U = _jt_init_dispatch(env, x0, p_hit, horizon, gain=50.0)
    assert U.shape == (horizon, 6)
    # 位置模式返回的是期望角度，应在关节范围内
    assert np.all(np.abs(U) < np.pi), "角度应在 [-π, π] 范围内"


def test_fix_joint5_dispatch_position_mode() -> None:
    """位置模式 fix_joint5 dispatch 返回的 U[:, 5] 全等于 fix_angle。"""
    env = _make_env_position()
    x0 = env.get_arm_state()
    fix_angle = 2.0
    U = np.zeros((10, 6))
    U_fixed = _fix_joint5_dispatch(U, x0, env, fix_angle)
    assert np.allclose(U_fixed[:, 5], fix_angle), "位置模式第6关节应全等于 fix_angle"
