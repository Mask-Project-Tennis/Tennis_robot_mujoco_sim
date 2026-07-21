"""时序数据采集与持久化。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.joint_test.types import (
    BackendType,
    TrackingResult,
    WaveformConfig,
)


class TrackingRecorder:
    """采集实验过程中的时序数据。

    预分配数组，逐步填充，结束时返回 TrackingResult。

    Args:
        config: 波形配置（用于构造 TrackingResult）。
        dt: 时间步长 (s)。
        backend: 后端类型（记录在 TrackingResult 中）。
    """

    def __init__(
        self,
        config: WaveformConfig,
        dt: float,
        backend: BackendType,
    ) -> None:
        """初始化记录器，预分配数组。"""
        self._cfg = config
        self._dt = dt
        self._backend = backend
        self._n = int(config.duration_s / dt)
        # 预分配（不预填，避免误导）
        self._time = np.zeros(self._n)
        self._q_des = np.zeros((self._n, 6))
        self._q_act = np.zeros((self._n, 6))
        self._qdot_act = np.zeros((self._n, 6))
        self._k = 0  # 已记录的步数

    def record(
        self,
        t: float,
        q_des: np.ndarray,
        q_actual: np.ndarray,
        qdot_actual: np.ndarray,
    ) -> None:
        """记录单步数据。

        超容量时静默丢弃（不报错，不覆盖已有数据）。

        Args:
            t: 当前时间 (s)。
            q_des: 期望关节角度 (6,)。
            q_actual: 实际关节角度 (6,)。
            qdot_actual: 实际关节速度 (6,)。
        """
        if self._k >= self._n:
            return  # 超容量，静默丢弃
        self._time[self._k] = t
        self._q_des[self._k] = q_des
        self._q_act[self._k] = q_actual
        self._qdot_act[self._k] = qdot_actual
        self._k += 1

    def finalize(self) -> TrackingResult:
        """收尾，截断未填满的部分（如有），返回 TrackingResult。

        Returns:
            包含已记录数据的 TrackingResult。
        """
        n = self._k
        return TrackingResult(
            config=self._cfg,
            backend=self._backend,
            dt=self._dt,
            time=self._time[:n].copy(),
            q_desired=self._q_des[:n].copy(),
            q_actual=self._q_act[:n].copy(),
            qdot_actual=self._qdot_act[:n].copy(),
        )

    @staticmethod
    def save_npz(result: TrackingResult, path: Path) -> None:
        """保存为 NumPy 压缩格式 (.npz)。

        Args:
            result: 跟踪结果。
            path: 输出路径，父目录自动创建。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            time=result.time,
            q_desired=result.q_desired,
            q_actual=result.q_actual,
            qdot_actual=result.qdot_actual,
            tracking_error=result.tracking_error,
            dt=result.dt,
            joint_idx=result.config.joint_idx,
            waveform=result.config.waveform.value,
            frequency_hz=result.config.frequency_hz,
            amplitude_rad=result.config.amplitude_rad,
            backend=result.backend.value,
        )

    @staticmethod
    def save_csv(result: TrackingResult, path: Path) -> None:
        """保存为 CSV（时间 + 目标关节的 q_des/q_act/qdot/error）。

        Args:
            result: 跟踪结果。
            path: 输出路径，父目录自动创建。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        j = result.config.joint_idx
        data = np.column_stack([
            result.time,
            result.q_desired[:, j],
            result.q_actual[:, j],
            result.qdot_actual[:, j],
            result.tracking_error,
        ])
        header = "time_s,q_desired_rad,q_actual_rad,qdot_actual_rad,tracking_error_rad"
        np.savetxt(str(path), data, delimiter=",", header=header, comments="")
