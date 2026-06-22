"""RobotArmInterface → ExecutorComponent 适配器。

将真机/FakeRobot 的 RobotArmInterface 包装为 EpisodeRunner 所需的
ExecutorComponent Protocol 实现，统一仿真/真机编排。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.real.robot_arm_protocol import RobotArmInterface


class RobotExecutor:
    """RobotArmInterface → ExecutorComponent 适配器。

    包装 RobotInterface/FakeRobot 为 ExecutorComponent Protocol 实现，
    供 EpisodeRunner 调用 ``get_arm_state`` / ``execute``。

    Args:
        robot: 实现 RobotArmInterface 协议的机器人接口。
    """

    def __init__(self, robot: RobotArmInterface) -> None:
        """初始化适配器。

        Args:
            robot: 机器人接口（FakeRobot 或 RobotInterface）。
        """
        self._robot = robot

    def get_arm_state(self) -> NDArray[np.floating]:
        """返回当前臂状态 [q(6), qdot(6)]，形状 (12,)，弧度制。

        Returns:
            拼接的关节位置+速度数组。
        """
        return np.asarray(self._robot.get_arm_state(), dtype=np.float64)

    def execute(self, u_cmd: NDArray[np.floating]) -> None:
        """发送控制指令到机器人。

        Args:
            u_cmd: 控制指令 (6,)。位置模式下为期望关节角度，力矩模式下为力矩。
        """
        self._robot.send_joint_command(np.asarray(u_cmd, dtype=np.float64))

    def get_metrics(self) -> dict:
        """返回空指标字典（真机执行器不收集仿真诊断数据）。"""
        return {}
