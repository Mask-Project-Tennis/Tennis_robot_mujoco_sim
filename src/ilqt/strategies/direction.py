"""方向策略 — 从球速度计算击球方向与期望末端速度。

默认实现为"来球反方向"：d_hat = -ball_vel / |ball_vel|（V11 行 819-838）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass
class DirectionResult:
    """方向计算结果。"""

    d_hat: NDArray[np.floating]            # 击球方向单位向量 (3,)
    d_follow: NDArray[np.floating]         # 随挥方向（默认 = d_hat）(3,)
    v_hit_desired: NDArray[np.floating]    # 期望击球时刻末端速度 (3,)


@runtime_checkable
class DirectionPolicy(Protocol):
    """方向策略接口 — 从球速度推导击球方向。"""

    def compute(self, ball_vel: NDArray[np.floating]) -> DirectionResult:
        """计算击球方向。

        Args:
            ball_vel: 球速度 (3,)。

        Returns:
            DirectionResult（d_hat, d_follow, v_hit_desired）。
        """
        ...


class ReflectDirection:
    """来球反方向策略 — 与 V11 `_compute_direction` 完全一致。

    d_hat = -ball_vel / |ball_vel|（球向哪里飞，球拍就向反方向击）。
    零速度时回落到默认方向 [0, 1, 0]。
    """

    def __init__(self, target_speed: float = 1.8) -> None:
        """初始化目标击球速度。

        Args:
            target_speed: 期望末端速度大小（m/s）。
        """
        self._target_speed: float = target_speed

    def compute(self, ball_vel: NDArray[np.floating]) -> DirectionResult:
        """计算来球反方向。

        Args:
            ball_vel: 球速度 (3,)。

        Returns:
            DirectionResult（d_hat = d_follow = 来球反方向，
            v_hit_desired = target_speed * d_hat）。
        """
        v_norm = float(np.linalg.norm(ball_vel))
        if v_norm > 1e-6:
            d_hat = -ball_vel / v_norm
        else:
            d_hat = np.array([0.0, 1.0, 0.0])
        v_hit_desired = self._target_speed * d_hat
        return DirectionResult(
            d_hat=d_hat,
            d_follow=d_hat.copy(),
            v_hit_desired=v_hit_desired,
        )
