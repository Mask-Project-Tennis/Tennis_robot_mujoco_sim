"""机械臂控制接口协议 — 真机和 Mock 共同实现。

RealRunner 依赖此 Protocol，不依赖具体的 RobotInterface 类。
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RobotArmInterface(Protocol):
    """机械臂控制接口协议。

    真机实现: RobotInterface（Realman SDK 封装）
    测试 Mock: FakeRobot（简单一阶动力学）
    """

    def connect(self) -> bool:
        """连接机械臂。返回是否成功。"""
        ...

    def disconnect(self) -> None:
        """断开连接。"""
        ...

    def get_arm_state(self) -> np.ndarray:
        """读取关节状态。

        Returns:
            (12,) [q(6), qdot(6)]，弧度。与 RM65Env/RobotInterface 一致。
        """
        ...

    def send_joint_command(self, q_desired: np.ndarray) -> int:
        """发送关节位置指令。

        Args:
            q_desired: (6,) 目标关节角度，弧度。

        Returns:
            状态码（0=成功）。
        """
        ...

    def emergency_stop(self) -> None:
        """紧急停止（不可恢复）。"""
        ...

    def slow_stop(self) -> None:
        """缓停（可恢复）。"""
        ...
