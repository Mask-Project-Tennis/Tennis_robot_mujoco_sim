"""力矩→位置指令转换器。

MPC 输出力矩 u_k → 积分为关节位置 q_desired。
真机内部位置控制器跟踪 q_desired。
"""

import numpy as np


class TorqueToPositionIntegrator:
    """力矩 → 位置指令转换器。

    公式: q_desired = q + qdot·dt + ½·(τ/M_diag)·dt²

    Args:
        dt: 控制时间步长（秒）。
        M_diag: 对角惯量近似 (6,)，None 时使用默认值。
    """

    def __init__(
        self,
        dt: float,
        M_diag: np.ndarray | None = None,
    ) -> None:
        self._dt = dt
        self._M_diag = (
            M_diag
            if M_diag is not None
            else np.array([5.0, 5.0, 3.0, 1.0, 1.0, 0.5])
        )

    def integrate(self, arm_state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """力矩 → 位置指令。

        Args:
            arm_state: (12,) [q(6), qdot(6)]，弧度。
            u: (6,) 力矩。

        Returns:
            q_desired: (6,) 目标关节角度，弧度。
        """
        q = arm_state[:6]
        qdot = arm_state[6:]
        qddot = u / self._M_diag
        return q + qdot * self._dt + 0.5 * qddot * self._dt ** 2
