"""RobotInterface 单元测试。

使用 MockRoboticArm 模拟 Realman SDK，验证：
- 连接管理
- 关节角度读取（度→弧度转换）
- 关节速度数值微分
- 多模式控制（IP/CANFD）
- 急停/缓停
"""

import time
from typing import Any

import numpy as np
import pytest

from src.real.config import RealRobotConfig
from src.real.robot_interface import RobotInterface


class MockRoboticArm:
    """模拟 Realman SDK RoboticArm，记录所有调用供断言。"""

    def __init__(self) -> None:
        self._joint_deg: list[float] = [0.0] * 6
        self.calls: list[tuple[Any, ...]] = []

    def rm_create_robot_arm(self, ip: str, port: int) -> Any:
        self.calls.append(("create", ip, port))
        return type("Handle", (), {"id": 1})()

    def rm_get_joint_degree(self) -> tuple[int, list[float]]:
        return (0, self._joint_deg.copy())

    def rm_movej_follow(self, joint: list[float]) -> int:
        self.calls.append(("movej_follow", list(joint)))
        self._joint_deg = list(joint)
        return 0

    def rm_movej_canfd(
        self,
        joint: list[float],
        follow: bool,
        expand: float = 0,
        trajectory_mode: int = 0,
        radio: int = 0,
    ) -> int:
        self.calls.append(("movej_canfd", list(joint), follow,
                           trajectory_mode, radio))
        self._joint_deg = list(joint)
        return 0

    def rm_set_arm_stop(self) -> int:
        self.calls.append(("arm_stop",))
        return 0

    def rm_set_arm_slow_stop(self) -> int:
        self.calls.append(("slow_stop",))
        return 0

    def rm_delete_robot_arm(self) -> int:
        self.calls.append(("delete",))
        return 0


@pytest.fixture
def mock_arm() -> MockRoboticArm:
    """提供 MockRoboticArm 实例。"""
    return MockRoboticArm()


@pytest.fixture
def config() -> RealRobotConfig:
    """提供默认 RealRobotConfig。"""
    return RealRobotConfig()


class TestRobotInterfaceConnect:
    """连接管理测试。"""

    def test_connect_mock_returns_true(
        self, mock_arm: MockRoboticArm, config: RealRobotConfig
    ):
        """Mock SDK 连接成功，返回 True。"""
        ri = RobotInterface(config, arm=mock_arm)
        assert ri.connect() is True
        assert any(c[0] == "create" for c in mock_arm.calls)


class TestRobotInterfaceState:
    """状态读取测试。"""

    def test_get_arm_state_returns_radians(
        self, mock_arm: MockRoboticArm, config: RealRobotConfig
    ):
        """get_arm_state 返回弧度（SDK 返回度，边界转换）。"""
        mock_arm._joint_deg = [90.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ri = RobotInterface(config, arm=mock_arm)
        ri.connect()
        state = ri.get_arm_state()
        assert state.shape == (12,)
        np.testing.assert_allclose(state[0], np.pi / 2, atol=1e-10)

    def test_get_arm_state_raises_on_sdk_failure(
        self, mock_arm: MockRoboticArm, config: RealRobotConfig
    ):
        """SDK 返回错误码时，get_arm_state 抛出 RuntimeError（而非静默返回零）。"""

        def fail_get_joint_degree():
            return (-1, [0.0] * 6)

        mock_arm.rm_get_joint_degree = fail_get_joint_degree
        ri = RobotInterface(config, arm=mock_arm)
        ri.connect()
        with pytest.raises(RuntimeError, match="rm_get_joint_degree"):
            ri.get_arm_state()

    def test_velocity_numerical_diff(
        self, monkeypatch, mock_arm: MockRoboticArm, config: RealRobotConfig
    ):
        """连续两次读取，qdot ≈ Δq/Δt（数值微分）。"""
        time_values = iter([100.0, 100.01])
        monkeypatch.setattr(
            "src.real.robot_interface.time.perf_counter",
            lambda: next(time_values),
        )

        mock_arm._joint_deg = [0.0] * 6
        ri = RobotInterface(config, arm=mock_arm)
        ri.connect()
        state1 = ri.get_arm_state()
        np.testing.assert_allclose(state1[6:], np.zeros(6))  # 首次 qdot=0

        mock_arm._joint_deg = [1.0] * 6  # 关节变化 1°
        state2 = ri.get_arm_state()
        expected_qdot = np.radians(1.0) / 0.01  # dt=0.01s
        np.testing.assert_allclose(state2[6:], expected_qdot, rtol=0.01)


class TestRobotInterfaceControl:
    """多模式控制测试。"""

    def test_ip_mode_calls_movej_follow(
        self, mock_arm: MockRoboticArm, config: RealRobotConfig
    ):
        """IP 模式（默认）下发指令 → 调用 rm_movej_follow，参数为度。"""
        config.control_mode = "ip"
        ri = RobotInterface(config, arm=mock_arm)
        ri.connect()
        q_desired_rad = np.array([0.0, 0.5, -0.3, 0.0, 0.1, 0.0])
        ri.send_joint_command(q_desired_rad)

        movej_calls = [c for c in mock_arm.calls if c[0] == "movej_follow"]
        assert len(movej_calls) == 1
        np.testing.assert_allclose(
            movej_calls[0][1], np.degrees(q_desired_rad), atol=1e-6
        )

    def test_canfd_mode_calls_movej_canfd(
        self, mock_arm: MockRoboticArm, config: RealRobotConfig
    ):
        """CANFD 模式下发指令 → 调用 rm_movej_canfd，follow=True。"""
        config.control_mode = "canfd"
        config.canfd_trajectory_mode = 1
        config.canfd_smooth_radio = 50
        ri = RobotInterface(config, arm=mock_arm)
        ri.connect()
        q_desired_rad = np.array([0.0, 0.5, -0.3, 0.0, 0.1, 0.0])
        ri.send_joint_command(q_desired_rad)

        canfd_calls = [c for c in mock_arm.calls if c[0] == "movej_canfd"]
        assert len(canfd_calls) == 1
        assert canfd_calls[0][2] is True  # follow=True
        assert canfd_calls[0][3] == 1     # trajectory_mode
        assert canfd_calls[0][4] == 50    # radio


class TestRobotInterfaceSafety:
    """安全停止测试。"""

    def test_emergency_stop(
        self, mock_arm: MockRoboticArm, config: RealRobotConfig
    ):
        """急停 → 调用 rm_set_arm_stop。"""
        ri = RobotInterface(config, arm=mock_arm)
        ri.connect()
        ri.emergency_stop()
        assert any(c[0] == "arm_stop" for c in mock_arm.calls)

    def test_slow_stop(
        self, mock_arm: MockRoboticArm, config: RealRobotConfig
    ):
        """缓停 → 调用 rm_set_arm_slow_stop。"""
        ri = RobotInterface(config, arm=mock_arm)
        ri.connect()
        ri.slow_stop()
        assert any(c[0] == "slow_stop" for c in mock_arm.calls)
