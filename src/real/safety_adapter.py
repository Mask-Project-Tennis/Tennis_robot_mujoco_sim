"""SafetyMonitor → SafetyComponent 适配器。

将真机 SafetyMonitor 包装为 EpisodeRunner 所需的 SafetyComponent
Protocol 实现，统一仿真/真机编排。

SafetyMonitor 的接口是 ``is_safe(arm_state, q_desired, tcp_speed) -> bool``，
而 SafetyComponent 要求 ``filter(u_cmd, arm_state) -> (safe_u, is_safe)``。
本适配器桥接两者，并可选地用规划环境的雅可比精确计算 TCP 线速度
（匹配 RealRunner.step 的 TCP 计算路径）。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.real.safety_monitor import SafetyMonitor

logger = logging.getLogger(__name__)


class SafetyAdapter:
    """SafetyMonitor → SafetyComponent 适配器。

    包装 SafetyMonitor.is_safe 为 SafetyComponent.filter 接口。
    通过可选的 env 雅可比精确计算 TCP 线速度（匹配 RealRunner.step 行为），
    env 为 None 时跳过 TCP 检查（tcp_speed=0.0）。

    Args:
        safety: SafetyMonitor 实例（关节位置/速度/TCP 速度检查）。
        env: 可选的规划环境（用于雅可比 TCP 速度计算）。None 时 tcp_speed=0.0。
    """

    def __init__(
        self,
        safety: SafetyMonitor,
        env: Any | None = None,
    ) -> None:
        """初始化适配器。

        Args:
            safety: SafetyMonitor 实例。
            env: 可选规划环境（含 set_arm_state / get_ee_jacp / NQ 接口）。
        """
        self._safety = safety
        self._env = env

    def filter(
        self,
        u_cmd: NDArray[np.floating],
        arm_state: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], bool]:
        """安全滤波 → (safe_u, is_safe)。

        用 env 雅可比精确计算 TCP 线速度（与 RealRunner.step 一致），
        再委托 SafetyMonitor.is_safe 做关节位置/速度/TCP 三重检查。
        本适配器不修改 u_cmd（通过时原样返回，违反时由上层决定中止）。

        Args:
            u_cmd: 控制指令 (NU,)。
            arm_state: 臂状态 [q, qdot] (2*NU,)。

        Returns:
            (u_cmd 原样, is_safe)。
        """
        u_cmd = np.asarray(u_cmd, dtype=np.float64)
        tcp_speed = 0.0

        if self._env is not None:
            try:
                nq = int(getattr(self._env, "NQ", 6))
                self._env.set_arm_state(arm_state)
                jacp = self._env.get_ee_jacp()
                qdot = arm_state[nq:]
                tcp_speed = float(np.linalg.norm(jacp @ qdot))
            except Exception as e:
                logger.debug("SafetyAdapter: TCP 速度计算回退 (%s)", e)
                tcp_speed = 0.0

        is_safe = self._safety.is_safe(arm_state, u_cmd, tcp_speed=tcp_speed)
        return u_cmd, bool(is_safe)
