"""runner_factory config 消费测试。

验证 build_robot_limits / build_replan_cfg 从 RealRobotConfig 读取参数，
而非使用硬编码值。YAML 为唯一真相源。
"""

from __future__ import annotations

import numpy as np

from src.ilqt.planning_env import PlanningEnv
from src.real.config import RealRobotConfig
from src.real.runner_factory import (
    INIT_Q,
    INIT_Q_LEFT,
    KD,
    KP,
    build_replan_cfg,
    build_robot_limits,
    build_solver,
)


def _build_env() -> PlanningEnv:
    """构建位置模式 PlanningEnv。"""
    from src.real.runner_factory import DT

    env = PlanningEnv(dt=DT)
    env.init_q_left = INIT_Q_LEFT.copy()
    env.configure_actuator_mode("position", kp=KP, kd=KD)
    env.configure_feedforward(True)
    env.reset(INIT_Q)
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left
    return env


class TestBuildRobotLimitsFromConfig:
    """build_robot_limits 应从 RealRobotConfig 读取限位参数。"""

    def test_custom_joint_limits_flow_through(self):
        """config 关节限位应反映到 RobotLimits（含 margin 内缩）。"""
        env = _build_env()
        cfg = RealRobotConfig()
        cfg.q_lower = np.radians([-90] * 6)
        cfg.q_upper = np.radians([90] * 6)

        rl = build_robot_limits(env, cfg)

        # q_margin_deg = [2, 1, 3, 3, 3, 3]
        # q_lower = q_min + margin, q_upper = q_max - margin
        expected_lower_deg = np.array([-88, -89, -87, -87, -87, -87])
        expected_upper_deg = np.array([88, 89, 87, 87, 87, 87])
        np.testing.assert_allclose(np.degrees(rl.q_lower), expected_lower_deg, atol=0.1)
        np.testing.assert_allclose(np.degrees(rl.q_upper), expected_upper_deg, atol=0.1)

    def test_custom_tcp_speed_flows_through(self):
        """config TCP 速度应反映到 RobotLimits。"""
        env = _build_env()
        cfg = RealRobotConfig()
        cfg.max_tcp_speed = 0.5

        rl = build_robot_limits(env, cfg)

        assert rl.max_tcp_speed == 0.5

    def test_custom_qdot_flows_through(self):
        """config 关节速度限制应反映到 RobotLimits。"""
        env = _build_env()
        cfg = RealRobotConfig()
        cfg.max_qdot = np.array([2.0] * 6)

        rl = build_robot_limits(env, cfg)

        np.testing.assert_allclose(rl.qdot_max, np.array([2.0] * 6), atol=0.01)

    def test_custom_dt_flows_through(self):
        """config dt 应反映到 RobotLimits 的 dq_max（dq_max = qdot_max * dt * fraction）。"""
        env = _build_env()
        cfg_default = RealRobotConfig()
        cfg_custom = RealRobotConfig()
        cfg_custom.dt = 0.01

        rl_default = build_robot_limits(env, cfg_default)
        rl_custom = build_robot_limits(env, cfg_custom)

        # dt 翻倍 → dq_max 翻倍
        np.testing.assert_allclose(rl_custom.dq_max, rl_default.dq_max * 2, rtol=0.01)
