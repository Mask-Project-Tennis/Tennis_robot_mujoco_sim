"""球感知器：BallSensor → 卡尔曼滤波 → 滤波输出。

封装 BallSensor（位置观测）+ BallEstimator（6D KF）。
通过有限差分从位置观测计算速度，喂给 KF。
"""

import logging
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.perception.ball_estimator import BallEstimator
from src.real.ball_sensor import BallSensor

logger = logging.getLogger(__name__)


class BallPerceiver:
    """球感知器 — 从 BallSensor 读取位置，KF 滤波，输出 (pos, vel)。

    Pipeline:
        sensor.get_latest() → 有限差分速度 → BallEstimator.update() → (filtered_pos, filtered_vel)

    三项关键修复：
        1. 过时数据短路：传感器时间戳未变时返回缓存值，避免零速度注入 KF。
        2. 挂钟时间同步：强制 KF 预测步用传感器 dt 而非挂钟时间（exp8 dt 陷阱修复）。
        3. 速度噪声缩放：有限差分放大位置噪声 σ_v = σ_p·√2/dt_obs，自动更新 R 矩阵。

    Attributes:
        sensor: BallSensor 实例（位置数据源）。
        estimator: BallEstimator 实例（6D KF）。
    """

    def __init__(
        self,
        sensor: BallSensor,
        estimator_config: dict[str, Any] | None = None,
        dt: float = 0.005,
    ) -> None:
        """初始化球感知器。

        Args:
            sensor: BallSensor 实例。
            estimator_config: BallEstimator 参数字典（pos_noise_std, vel_noise_std 等）。
            dt: 仿真/控制时间步长（秒）。
        """
        self._sensor = sensor
        self._dt = dt
        self._estimator = BallEstimator(dt=dt, **(estimator_config or {}))

        # 提取位置噪声 std，用于自动推导有限差分速度噪声
        cfg = estimator_config or {}
        self._pos_noise_std: float = cfg.get("pos_noise_std", 0.0)
        self._pos_noise_xyz: tuple[float, float, float] | None = cfg.get("pos_noise_xyz")

        self._last_pos: NDArray[np.floating] | None = None
        self._last_ts: float | None = None
        self._last_sensor_ts: float | None = None
        self._filtered: tuple[NDArray[np.floating], NDArray[np.floating]] | None = None

    def update(self) -> tuple[NDArray[np.floating], NDArray[np.floating]] | None:
        """从 sensor 读取最新数据，KF 更新。

        - 若 sensor 无数据（pos=None），返回 None。
        - 若传感器时间戳未变（过时数据），返回缓存值，不触发 KF 更新。
        - 有限差分速度 vel = (pos - last_pos) / dt_obs。

        Returns:
            (filtered_pos, filtered_vel) 或 None。
        """
        pos, ts = self._sensor.get_latest()
        if pos is None:
            return None

        # 修复 #1：过时数据短路 — 传感器时间戳未变时返回缓存，避免零速度注入 KF
        if (
            ts is not None
            and self._last_sensor_ts is not None
            and ts == self._last_sensor_ts
        ):
            return self._filtered

        self._last_sensor_ts = ts

        # 有限差分速度估计
        dt_obs = self._dt
        if (
            self._last_pos is not None
            and self._last_ts is not None
            and ts is not None
        ):
            dt_obs = ts - self._last_ts
            if dt_obs > 1e-6:
                vel = (pos - self._last_pos) / dt_obs
            else:
                vel = np.zeros(3)
        else:
            vel = np.zeros(3)

        self._last_pos = pos.copy()
        self._last_ts = ts

        # 修复 #3：有限差分放大位置噪声 → σ_v_fd = σ_p·√2/dt_obs
        # 自动缩放速度观测噪声 R，避免 KF 过度信任噪声速度
        if dt_obs > 1e-6:
            if self._pos_noise_xyz is not None:
                fd_vel_xyz = tuple(
                    s * np.sqrt(2.0) / dt_obs for s in self._pos_noise_xyz
                )
                self._estimator.update_noise_params(vel_noise_xyz=fd_vel_xyz)
            elif self._pos_noise_std > 0:
                fd_vel_std = self._pos_noise_std * np.sqrt(2.0) / dt_obs
                self._estimator.update_noise_params(vel_noise_std=fd_vel_std)

        # 修复 #2：挂钟时间同步 — 强制 KF 预测步用传感器 dt_obs 而非挂钟时间
        # 避免 exp8 dt 陷阱（仿真快于实时时挂钟 elapsed >> 物理 dt）
        self._estimator._last_update_time = time.perf_counter() - dt_obs

        # KF 更新
        filtered_pos, filtered_vel = self._estimator.update(pos, vel)
        self._filtered = (filtered_pos, filtered_vel)
        return self._filtered

    def get_latest_filtered(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]] | None:
        """获取最近一次滤波结果（不触发新观测）。

        Returns:
            (filtered_pos, filtered_vel) 或 None（尚未 update 过）。
        """
        return self._filtered

    def reset(self) -> None:
        """重置感知器（新 episode 时调用）。"""
        self._estimator.reset()
        self._last_pos = None
        self._last_ts = None
        self._last_sensor_ts = None
        self._filtered = None
        logger.debug("BallPerceiver 已重置")
