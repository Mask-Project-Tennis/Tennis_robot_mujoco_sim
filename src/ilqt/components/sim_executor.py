"""仿真执行组件 — 调用 env.step_full 推进 MuJoCo 物理仿真。

实现 ``ExecutorComponent`` Protocol。封装 V11 主循环中的物理步进逻辑。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SimExecutor:
    """仿真执行组件 — 调用 ``env.step_full`` 推进物理仿真。

    实现 ``ExecutorComponent`` Protocol。

    每步 ``execute(u_cmd)`` 等价于 V11 主循环中的 ``env.step_full(u_final)``：
    推进臂物理 + 观测管线（更新球缓存），下一拍感知组件即可读到新球状态。

    Args:
        env: RM65Env 实例（须提供 ``step_full`` 与 ``get_arm_state`` 方法）。
    """

    def __init__(self, env: object) -> None:
        """初始化执行组件。

        Args:
            env: 含 ``step_full`` / ``get_arm_state`` 的仿真环境。
        """
        self._env = env

    def get_arm_state(self) -> NDArray[np.floating]:
        """返回当前右臂状态 [q(6), qdot(6)]，形状 (12,)，弧度制。"""
        return self._env.get_arm_state()  # type: ignore[attr-defined]

    def execute(self, u_cmd: NDArray[np.floating]) -> None:
        """执行控制指令（调用 ``env.step_full`` 推进物理 + 观测）。

        Args:
            u_cmd: 控制指令 (6,)。力矩模式下为力矩，位置模式下为期望角度。
        """
        # step_full 返回 (x_arm, ball_pos, ball_vel)，本组件仅需推进物理，
        # 球状态由感知组件在下一拍读取。
        self._env.step_full(u_cmd)  # type: ignore[attr-defined]
