"""球感知抽象基类与仿真实现。

BallSensor(ABC) 定义球位置感知的统一接口。
SimulatedBallSensor 提供 push 模式用于测试和 MuJoCo 仿真。
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class BallSensor(ABC):
    """球感知抽象基类。

    真机实现：OptiTrackSensor / RealSenseSensor（待实施）。
    仿真实现：SimulatedBallSensor（测试用）。
    """

    @abstractmethod
    def start(self) -> None:
        """启动感知系统。"""
        ...

    @abstractmethod
    def stop(self) -> None:
        """停止感知系统。"""
        ...

    @abstractmethod
    def get_latest(self) -> tuple[NDArray[np.floating] | None, float | None]:
        """获取最新球位置和时间戳。

        Returns:
            pos: (3,) 球位置，或 None（无数据时）。
            timestamp: 时间戳（秒），或 None。
        """
        ...

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """感知系统是否正在运行。"""
        ...


class SimulatedBallSensor(BallSensor):
    """仿真球传感器（push 模式）。

    测试用：外部调用 push() 推入观测，get_latest() 返回最新值。
    模拟动捕/相机系统的异步数据回调。
    """

    def __init__(self) -> None:
        self._latest_pos: NDArray[np.floating] | None = None
        self._latest_ts: float | None = None
        self._running: bool = False

    def push(self, pos: NDArray[np.floating], timestamp: float) -> None:
        """推入新观测（模拟传感器回调）。

        Args:
            pos: (3,) 球位置。
            timestamp: 时间戳（秒）。
        """
        self._latest_pos = np.asarray(pos, dtype=np.float64).copy()
        self._latest_ts = float(timestamp)

    def start(self) -> None:
        self._running = True
        logger.debug("SimulatedBallSensor 已启动")

    def stop(self) -> None:
        self._running = False
        self._latest_pos = None
        self._latest_ts = None
        logger.debug("SimulatedBallSensor 已停止")

    def get_latest(self) -> tuple[NDArray[np.floating] | None, float | None]:
        if not self._running:
            return None, None
        return self._latest_pos, self._latest_ts

    @property
    def is_running(self) -> bool:
        return self._running
