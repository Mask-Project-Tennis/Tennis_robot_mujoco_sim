"""SafetyMonitor 单元测试。

测试真机安全监控的三重检查（位置/速度/TCP）。
"""

import numpy as np
import pytest

from src.real.config import RealRobotConfig
from src.real.safety_monitor import SafetyMonitor


@pytest.fixture
def config() -> RealRobotConfig:
    """提供限位缩小的测试配置。"""
    cfg = RealRobotConfig()
    cfg.q_lower = np.full(6, -1.0)
    cfg.q_upper = np.full(6, 1.0)
    cfg.max_qdot = np.full(6, 3.14)
    cfg.max_tcp_speed = 1.0
    return cfg


class TestSafetyMonitor:
    """安全监控测试。"""

    def test_unsafe_when_q_exceeds_limits(self, config: RealRobotConfig):
        """关节目标位置超出范围 → is_safe 返回 False。"""
        monitor = SafetyMonitor(config)
        arm_state = np.zeros(12)
        q_desired = np.array([2.0, 0, 0, 0, 0, 0])  # 超上限
        assert not monitor.is_safe(arm_state, q_desired)

    def test_unsafe_when_tcp_overspeed(self, config: RealRobotConfig):
        """TCP 速度超出限制 → is_safe 返回 False。"""
        monitor = SafetyMonitor(config)
        arm_state = np.zeros(12)
        q_desired = np.zeros(6)
        assert not monitor.is_safe(arm_state, q_desired, tcp_speed=1.5)

    def test_safe_when_all_pass(self, config: RealRobotConfig):
        """全部检查通过 → is_safe 返回 True。"""
        monitor = SafetyMonitor(config)
        arm_state = np.zeros(12)
        q_desired = np.zeros(6)
        assert monitor.is_safe(arm_state, q_desired, tcp_speed=0.5)
