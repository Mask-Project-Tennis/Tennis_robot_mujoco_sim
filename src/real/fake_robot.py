"""假机器人 Mock 接口 — 用于离线测试 RealRunner。

简单一阶动力学: 收到 q_desired 后，
q 直接设为 q_desired，qdot = (q_desired - q_old) / dt。
"""

import numpy as np


class FakeRobot:
    """假机器人 — 实现 RobotArmInterface 协议。

    用于 RealRunner 离线测试，不连接任何真机硬件。
    维护 command_history 供测试验证。

    Args:
        init_q: 初始关节角度 (6,)，弧度。
        dt: 控制时间步长 (s)，用于计算 qdot。
    """

    def __init__(self, init_q: np.ndarray, dt: float = 0.005) -> None:
        self._q = np.asarray(init_q, dtype=float).copy()
        self._qdot = np.zeros(6)
        self._dt = dt
        self._connected = False
        self.command_history: list[np.ndarray] = []
        self.slow_stop_count = 0
        self.emergency_stop_count = 0

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def get_arm_state(self) -> np.ndarray:
        """返回 [q(6), qdot(6)] 拼接数组，形状 (12,)，弧度。"""
        return np.concatenate([self._q.copy(), self._qdot.copy()])

    def send_joint_command(self, q_desired: np.ndarray) -> int:
        q_desired = np.asarray(q_desired, dtype=float)
        self._qdot = (q_desired - self._q) / self._dt
        self._q = q_desired.copy()
        self.command_history.append(q_desired.copy())
        return 0

    def emergency_stop(self) -> None:
        self.emergency_stop_count += 1

    def slow_stop(self) -> None:
        self.slow_stop_count += 1
