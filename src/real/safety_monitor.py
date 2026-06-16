"""真机安全监控。

每个控制 tick 检查关节位置、关节速度、TCP 速度。
超限时委托 RobotInterface 执行急停/缓停。
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.real.config import RealRobotConfig

if TYPE_CHECKING:
    from src.real.robot_interface import RobotInterface

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """真机安全监控。

    每个控制 tick 检查：
    - 关节位置超限（q_desired 超出 q_lower/q_upper）
    - 关节速度超限（|qdot| 超出 max_qdot）
    - TCP 速度超限（tcp_speed 超出 max_tcp_speed）

    Args:
        config: 真机配置（含关节限位和安全参数）。
        robot: RobotInterface 实例，None 时不执行急停（测试用）。
    """

    def __init__(
        self,
        config: RealRobotConfig,
        robot: "RobotInterface | None" = None,
    ) -> None:
        self._config = config
        self._robot = robot

    def is_safe(
        self,
        arm_state: np.ndarray,
        q_desired: np.ndarray,
        tcp_speed: float = 0.0,
    ) -> bool:
        """检查当前状态是否安全。

        Args:
            arm_state: (12,) [q(6), qdot(6)]，弧度。
            q_desired: (6,) 目标关节角度，弧度。
            tcp_speed: 末端线速度 m/s。

        Returns:
            True 如果全部检查通过。
        """
        qdot = arm_state[6:]

        if np.any(q_desired < self._config.q_lower) or np.any(
            q_desired > self._config.q_upper
        ):
            logger.warning("关节位置超限")
            return False

        if np.any(np.abs(qdot) > self._config.max_qdot):
            logger.warning("关节速度超限")
            return False

        if tcp_speed > self._config.max_tcp_speed:
            logger.warning("TCP 速度超限: %.2f > %.2f", tcp_speed, self._config.max_tcp_speed)
            return False

        return True

    def emergency_stop(self) -> None:
        """急停（委托给 RobotInterface）。"""
        if self._robot:
            self._robot.emergency_stop()

    def slow_stop(self) -> None:
        """缓停（委托给 RobotInterface）。"""
        if self._robot:
            self._robot.slow_stop()
