"""PlanningEnv 球状态委托方法测试。

验证 PlanningEnv 通过 BallPredictor 委托实现的球状态接口，
使其鸭子兼容 RM65Env 的球接口（供 do_replan 复用）。
"""

import numpy as np

from src.ilqt.ball_predictor import BallPredictor
from src.ilqt.planning_env import PlanningEnv


def test_env_ball_state_roundtrip():
    """球状态写入后读出应保持一致。"""
    env = PlanningEnv()
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.5, -1.0, 2.0])
    env.set_ball_state(pos, vel)
    got_pos, got_vel = env.get_ball_state()
    np.testing.assert_array_equal(got_pos, pos)
    np.testing.assert_array_equal(got_vel, vel)


def test_env_predict_matches_predictor():
    """PlanningEnv.predict_ball_trajectory 应与独立 BallPredictor 完全一致。"""
    env = PlanningEnv()
    bp = BallPredictor(dt=env.dt)
    p0 = np.array([0.0, 0.0, 1.5])
    v0 = np.array([1.0, -0.5, 0.0])
    n = 20
    env_pos, env_vel = env.predict_ball_trajectory(p0, v0, n)
    bp_pos, bp_vel = bp.predict_from(p0, v0, n)
    np.testing.assert_allclose(env_pos, bp_pos)
    np.testing.assert_allclose(env_vel, bp_vel)

