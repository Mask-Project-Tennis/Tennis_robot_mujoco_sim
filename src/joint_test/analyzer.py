"""跟踪性能指标计算。"""
from __future__ import annotations

import numpy as np

from src.joint_test.types import (
    Metrics,
    TrackingResult,
    WaveformType,
)


class MetricsAnalyzer:
    """从 TrackingResult 计算跟踪性能指标。

    根据波形类型选择不同的分析策略:
        - SINE/CHIRP: FFT 提取目标频率的幅值比和相位滞后
        - STEP: 上升时间、超调、稳态误差
        - TRIANGLE/SQUARE: 仅通用指标 (RMSE/max_error/mean_error)
    """

    def analyze(self, result: TrackingResult) -> Metrics:
        """分析跟踪结果，返回指标。

        Args:
            result: 跟踪实验结果。

        Returns:
            包含各项指标的 Metrics 对象。
        """
        j = result.config.joint_idx
        err = result.tracking_error
        wf = result.config.waveform

        rmse = float(np.sqrt(np.mean(err**2)))
        max_err = float(np.max(np.abs(err)))
        mean_err = float(np.mean(err))

        if wf in (WaveformType.SINE, WaveformType.CHIRP):
            # 退化检查：零幅值或常量信号
            q_des_ac = result.q_desired[:, j] - np.mean(result.q_desired[:, j])
            q_act_ac = result.q_actual[:, j] - np.mean(result.q_actual[:, j])
            des_energy = float(np.sum(q_des_ac**2))
            act_energy = float(np.sum(q_act_ac**2))
            if des_energy < 1e-12:
                # 期望信号本身为常量 → 频域指标 N/A
                return Metrics(
                    waveform=wf,
                    target_joint=j,
                    rmse_rad=rmse,
                    max_error_rad=max_err,
                    mean_error_rad=mean_err,
                    amplitude_ratio=None,
                    phase_lag_deg=None,
                )
            if act_energy < 1e-12:
                # 期望有内容但实际为常量 → 机器人卡死，幅值比显式置 0（区分 N/A）
                return Metrics(
                    waveform=wf,
                    target_joint=j,
                    rmse_rad=rmse,
                    max_error_rad=max_err,
                    mean_error_rad=mean_err,
                    amplitude_ratio=0.0,
                    phase_lag_deg=None,
                )
            a_ratio, phase_lag = self._fft_amplitude_phase(result)
            return Metrics(
                waveform=wf,
                target_joint=j,
                rmse_rad=rmse,
                max_error_rad=max_err,
                mean_error_rad=mean_err,
                amplitude_ratio=a_ratio,
                phase_lag_deg=phase_lag,
            )
        elif wf == WaveformType.STEP:
            step_metrics = self._step_metrics(result)
            return Metrics(
                waveform=wf,
                target_joint=j,
                rmse_rad=rmse,
                max_error_rad=max_err,
                mean_error_rad=mean_err,
                **step_metrics,
            )
        else:
            # triangle/square: 仅通用指标
            return Metrics(
                waveform=wf,
                target_joint=j,
                rmse_rad=rmse,
                max_error_rad=max_err,
                mean_error_rad=mean_err,
            )

    def _fft_amplitude_phase(
        self, result: TrackingResult
    ) -> tuple[float, float]:
        """FFT 提取目标频率的幅值比和相位滞后。

        Args:
            result: 跟踪结果。

        Returns:
            (amplitude_ratio, phase_lag_deg)。
            amplitude_ratio = A_actual / A_desired
            phase_lag_deg 归一化到 [-180, 180]
        """
        j = result.config.joint_idx
        dt = result.dt
        n = len(result.time)

        q_des = result.q_desired[:, j]
        q_act = result.q_actual[:, j]

        # 单边 FFT
        fft_des = np.fft.rfft(q_des)
        fft_act = np.fft.rfft(q_act)
        freqs = np.fft.rfftfreq(n, dt)

        # 目标频率附近的 bin
        target_f = result.config.frequency_hz
        f_idx = int(np.argmin(np.abs(freqs - target_f)))

        a_des = 2.0 * np.abs(fft_des[f_idx]) / n
        a_act = 2.0 * np.abs(fft_act[f_idx]) / n

        phase_des = np.angle(fft_des[f_idx])
        phase_act = np.angle(fft_act[f_idx])
        phase_lag_rad = phase_des - phase_act
        # 归一化到 [-π, π]
        phase_lag_rad = (phase_lag_rad + np.pi) % (2 * np.pi) - np.pi

        a_ratio = float(a_act / max(a_des, 1e-12))
        phase_lag_deg = float(np.degrees(phase_lag_rad))
        return a_ratio, phase_lag_deg

    def _step_metrics(self, result: TrackingResult) -> dict:
        """阶跃响应指标。

        计算上升时间 (10%-90%)、settling time (±2%)、超调量、稳态误差。

        Args:
            result: 跟踪结果。

        Returns:
            包含 step 指标的字典。
        """
        j = result.config.joint_idx
        q_act = result.q_actual[:, j]
        q_des = result.q_desired[:, j]
        target = q_des[-1]
        initial = q_des[0]
        step_size = target - initial

        if abs(step_size) < 1e-9:
            return {}  # 无阶跃

        # 10%-90% 上升时间
        norm = (q_act - initial) / step_size
        idx_10 = int(np.argmax(norm >= 0.1)) if np.any(norm >= 0.1) else 0
        idx_90 = int(np.argmax(norm >= 0.9)) if np.any(norm >= 0.9) else len(norm) - 1
        rise_time = float((idx_90 - idx_10) * result.dt)

        # 超调量
        if step_size > 0:
            peak = float(np.max(q_act))
            overshoot_pct = max(0.0, (peak - target) / step_size * 100.0)
        else:
            trough = float(np.min(q_act))
            overshoot_pct = max(0.0, (target - trough) / abs(step_size) * 100.0)

        # settling time (±2% 稳态带)
        band = 0.02 * abs(step_size)
        settled = np.abs(q_act - target) <= band
        settle_k = len(q_act)
        for k in range(len(q_act) - 1, -1, -1):
            if not settled[k]:
                settle_k = k + 1
                break
        settling_time = float(settle_k * result.dt)

        # 稳态误差（最后 10% 平均）
        n_tail = max(1, len(q_act) // 10)
        sse = float(np.mean(q_act[-n_tail:] - target))

        return dict(
            rise_time_s=rise_time,
            settling_time_s=settling_time,
            overshoot_pct=overshoot_pct,
            steady_state_error_rad=sse,
        )
