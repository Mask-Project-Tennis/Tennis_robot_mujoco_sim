"""create_runner 工厂函数测试。

验证 mock=True/False 两种模式下 robot 类型的正确选择。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 将 scripts/ 加入 sys.path 以导入 create_runner
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

from run_real_robot import create_runner
from src.real.config import RealRobotConfig
from src.real.fake_robot import FakeRobot
from src.real.robot_interface import RobotInterface


class TestCreateRunnerRealMode:
    """mock=False 真机模式测试。"""

    def test_mock_false_creates_robot_interface(self) -> None:
        """mock=False → runner._robot 是 RobotInterface（不是 FakeRobot）。"""
        config = RealRobotConfig(robot_ip="fake-ip")
        runner = create_runner(config, mock=False)
        assert isinstance(runner._robot, RobotInterface)
        assert not isinstance(runner._robot, FakeRobot)


class TestCreateRunnerMockMode:
    """mock=True Mock 模式回归测试。"""

    def test_mock_true_creates_fake_robot(self) -> None:
        """mock=True → runner._robot 是 FakeRobot（回归保护）。"""
        config = RealRobotConfig(robot_ip="fake-ip")
        runner = create_runner(config, mock=True)
        assert isinstance(runner._robot, FakeRobot)


class TestCreateRunnerTimerConfig:
    """create_runner 的 AdaptiveTimer 配置测试。"""

    def test_timer_target_100hz(self) -> None:
        """create_runner 的 timer 目标频率为 100Hz（匹配 SDK 实测吞吐上限）。"""
        runner = create_runner(RealRobotConfig(), mock=True)
        assert runner._timer._target_dt == pytest.approx(1.0 / 100.0)
