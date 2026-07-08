"""基础安全滤波 — 无预测，仅关节限位 + 速度限位检查。

实现 ``SafetyComponent`` Protocol。适用场景：

  - 计算资源受限、无法承担逐步预测开销；
  - 不信任规划模型，仅做硬限位兜底；
  - 作为 PredictiveSafetyFilter 的退化对照基线。

注意：本组件**不修改** ``u_cmd`` —— 通过时原样返回，违反时返回
``is_safe=False``（由上层 EpisodeRunner 决定是否中止 episode）。
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray


class BasicSafetyFilter:
    """基础安全滤波 — 不需要 env，仅检查关节位置/速度限位。

    实现 ``SafetyComponent`` Protocol。

    Args:
        q_lower: (NU,) 关节位置下限，弧度。
        q_upper: (NU,) 关节位置上限，弧度。
        max_qdot: (NU,) 关节速度上限，弧度/秒。
        max_tcp_speed: TCP 最大线速度 m/s（**设计上不检查** — 需要 env FK
            做 TCP 速度预测，本基础版无 env 依赖。如需 TCP 检查请用
            PredictiveSafetyFilter。参数保留仅为构造签名一致性）。
    """

    def __init__(
        self,
        q_lower: NDArray[np.floating],
        q_upper: NDArray[np.floating],
        max_qdot: NDArray[np.floating],
        max_tcp_speed: float = 1.0,
    ) -> None:
        """初始化基础安全滤波器。

        Args:
            q_lower: 关节位置下限 (NU,)。
            q_upper: 关节位置上限 (NU,)。
            max_qdot: 关节速度上限 (NU,)。
            max_tcp_speed: TCP 速度上限（预留，当前未检查）。
        """
        warnings.warn(
            "BasicSafetyFilter 缺少 TCP 速度检查，不应用于生产环境。"
            "请使用 PredictiveSafetyFilter。",
            RuntimeWarning,
            stacklevel=2,
        )
        self._q_lower = np.asarray(q_lower, dtype=np.float64)
        self._q_upper = np.asarray(q_upper, dtype=np.float64)
        self._max_qdot = np.asarray(max_qdot, dtype=np.float64)
        self._max_tcp_speed = max_tcp_speed
        self._nq = int(self._q_lower.shape[0])

    def filter(
        self, u_cmd: NDArray[np.floating], arm_state: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], bool]:
        """安全滤波 → (safe_u, is_safe)。

        位置模式下 ``u_cmd`` 视为 q_desired 做限位检查；
        力矩模式下本组件无法判定力矩限位（需 ctrlrange），仅检查当前 qdot。

        Args:
            u_cmd: 控制指令 (NU,)。
            arm_state: 臂状态 [q, qdot] (2*NU,)。

        Returns:
            (u_cmd 原样, is_safe)。通过时不修改 u_cmd。
        """
        qdot = arm_state[self._nq : self._nq * 2]

        # 位置限位检查（u_cmd 视为 q_desired）
        if np.any(u_cmd < self._q_lower) or np.any(u_cmd > self._q_upper):
            return np.asarray(u_cmd), False

        # 关节速度限位检查（当前已超速则判不安全）
        if np.any(np.abs(qdot) > self._max_qdot):
            return np.asarray(u_cmd), False

        # 设计决策：TCP 速度检查需要 env FK，本基础版不支持（见类 docstring）。
        return np.asarray(u_cmd), True
