"""BallPerceiver → PerceptionComponent 适配器。

将 BallPerceiver 包装为 EpisodeRunner 所需的 PerceptionComponent
Protocol 实现，统一仿真/真机编排。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.real.ball_perceiver import BallPerceiver


class PerceptionAdapter:
    """BallPerceiver → PerceptionComponent 适配器。

    包装 BallPerceiver 为 PerceptionComponent Protocol 实现。
    每次调用 ``get_ball_state`` 先触发一次 KF 更新，再返回最新滤波结果。

    Args:
        perceiver: 球感知器（sensor → 有限差分速度 → KF 滤波）。
    """

    def __init__(self, perceiver: BallPerceiver) -> None:
        """初始化适配器。

        Args:
            perceiver: BallPerceiver 实例。
        """
        self._perceiver = perceiver

    def get_ball_state(self) -> tuple[NDArray[np.floating], NDArray[np.floating]] | None:
        """更新并返回滤波后的球状态。

        先调用 ``perceiver.update()`` 触发一次观测+KF 更新，
        再返回最近一次滤波结果（不重复观测）。

        Returns:
            (filtered_pos(3,), filtered_vel(3,)) 或 None（无数据时）。
        """
        self._perceiver.update()
        return self._perceiver.get_latest_filtered()
