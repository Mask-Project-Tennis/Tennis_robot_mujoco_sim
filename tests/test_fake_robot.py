"""FakeRobot Mock 单元测试 — 验证 RobotArmInterface 协议实现。"""

import numpy as np

from src.real.fake_robot import FakeRobot
from src.real.robot_arm_protocol import RobotArmInterface


def test_fake_robot_implements_protocol():
    """tracer 测试：FakeRobot 满足 RobotArmInterface 协议（runtime_checkable）。"""
    robot = FakeRobot(init_q=np.zeros(6))
    assert isinstance(robot, RobotArmInterface)


def test_fake_robot_send_then_get():
    """send 后 get 应返回最新 [q(6), qdot(6)]，qdot = (q_desired - q_old) / dt。"""
    robot = FakeRobot(init_q=np.zeros(6), dt=0.005)
    q_cmd = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    robot.send_joint_command(q_cmd)
    state = robot.get_arm_state()
    assert state.shape == (12,)
    np.testing.assert_array_almost_equal(state[:6], q_cmd)
    # qdot = (q_desired - q_old) / dt，q_old 为零
    expected_qdot = q_cmd / 0.005
    np.testing.assert_array_almost_equal(state[6:], expected_qdot)
    assert len(robot.command_history) == 1
