"""5 种波形生成器：sine/triangle/square/chirp/step。"""
from __future__ import annotations

import numpy as np
from scipy.signal import sawtooth, square, chirp

from src.joint_test.types import WaveformConfig, WaveformType


class WaveformGenerator:
    """生成单个关节的 q_des 轨迹，其他关节保持 base_q。

    所有波形预生成为 (N, 6) 数组，向量化、可缓存。

    Args:
        config: 波形配置。
        base_q: 6 关节基础角度 (6,)，其他关节保持此值。
        dt: 时间步长 (s)。
    """

    def __init__(
        self,
        config: WaveformConfig,
        base_q: np.ndarray,
        dt: float,
    ) -> None:
        """初始化波形生成器。"""
        assert 0 <= config.joint_idx < 6, "joint_idx 必须在 [0, 6)"
        self._cfg = config
        self._base_q = np.asarray(base_q, dtype=float).copy()
        self._dt = dt
        self._n_steps = int(config.duration_s / dt)
        self._t = np.arange(self._n_steps) * dt

    def generate(self) -> np.ndarray:
        """生成 (N, 6) 期望轨迹数组。

        仅在 config.joint_idx 对应的列上叠加波形，
        其他列保持 base_q 不变。

        Returns:
            (N, 6) 关节角度轨迹，单位 rad。
        """
        traj = np.tile(self._base_q, (self._n_steps, 1))  # (N, 6) 初始为常量
        j = self._cfg.joint_idx
        offset = self._cfg.offset_rad
        A = self._cfg.amplitude_rad
        f = self._cfg.frequency_hz
        t = self._t

        if self._cfg.waveform == WaveformType.SINE:
            traj[:, j] = offset + A * np.sin(2 * np.pi * f * t)
        elif self._cfg.waveform == WaveformType.TRIANGLE:
            traj[:, j] = offset + A * sawtooth(2 * np.pi * f * t, width=0.5)
        elif self._cfg.waveform == WaveformType.SQUARE:
            traj[:, j] = offset + A * square(2 * np.pi * f * t)
        elif self._cfg.waveform == WaveformType.CHIRP:
            f1 = self._cfg.end_frequency_hz if self._cfg.end_frequency_hz is not None else (f * 10)
            traj[:, j] = offset + A * chirp(
                t, f0=f, f1=f1, t1=t[-1] if len(t) > 0 else 1.0, method="linear"
            )
        elif self._cfg.waveform == WaveformType.STEP:
            target = self._cfg.step_target_rad if self._cfg.step_target_rad is not None else offset + A
            # n_steps 较小时确保 hold 阶段至少 1 步（避免空 hold）
            step_k = max(self._n_steps // 10, 1) if self._n_steps > 0 else 0
            traj[:step_k, j] = offset
            traj[step_k:, j] = target
        else:
            raise ValueError(f"未知波形类型: {self._cfg.waveform}")

        return traj

    @property
    def time_array(self) -> np.ndarray:
        """时间序列 (N,)，单位 s。"""
        return self._t.copy()
