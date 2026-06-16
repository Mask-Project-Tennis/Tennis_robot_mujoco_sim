"""TorqueToPositionIntegrator 单元测试。

测试力矩→位置积分器的物理正确性。
"""

import numpy as np
import pytest

from src.real.torque_to_position import TorqueToPositionIntegrator


class TestTorqueToPositionIntegrator:
    """力矩→位置积分器测试。"""

    def test_zero_torque_constant_velocity(self):
        """零力矩时，位置指令 = 当前位置 + 速度·dt（匀速运动）。"""
        dt = 0.01
        integrator = TorqueToPositionIntegrator(dt=dt)
        arm_state = np.array([
            0.5, -0.3, 0.8, 0.0, 0.1, -0.2,  # q
            0.1,  0.0, -0.5, 0.3, 0.0, 0.2,  # qdot
        ])
        u = np.zeros(6)
        q_desired = integrator.integrate(arm_state, u)
        expected = arm_state[:6] + arm_state[6:] * dt
        np.testing.assert_allclose(q_desired, expected, atol=1e-12)

    def test_nonzero_torque_acceleration(self):
        """非零力矩产生加速度，q_desired 含 ½·(τ/M)·dt² 项。"""
        dt = 0.01
        M_diag = np.array([5.0, 5.0, 3.0, 1.0, 1.0, 0.5])
        integrator = TorqueToPositionIntegrator(dt=dt, M_diag=M_diag)
        arm_state = np.zeros(12)
        arm_state[6:] = np.array([1.0, 0.0, -0.5, 0.3, 0.0, 0.2])
        u = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_desired = integrator.integrate(arm_state, u)
        # 关节 0：q=0, qdot=1.0, tau=10, M=5 → qddot=2.0
        expected_0 = 0.0 + 1.0 * dt + 0.5 * (10.0 / 5.0) * dt ** 2
        np.testing.assert_allclose(q_desired[0], expected_0, atol=1e-12)

    def test_M_diag_scales_acceleration(self):
        """大惯量 → 小加速度 → 小位移增量。"""
        arm_state = np.zeros(12)
        u = np.array([10.0, 0, 0, 0, 0, 0])
        small_M = TorqueToPositionIntegrator(dt=0.01, M_diag=np.full(6, 1.0))
        large_M = TorqueToPositionIntegrator(dt=0.01, M_diag=np.full(6, 100.0))
        q_small = small_M.integrate(arm_state, u)
        q_large = large_M.integrate(arm_state, u)
        assert abs(q_small[0]) > abs(q_large[0])
