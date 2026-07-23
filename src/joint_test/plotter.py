"""跟踪性能可视化（matplotlib）。

提供单次跟踪曲线、Bode 频率响应图、终端指标打印三种输出。
实时弹窗模式（show_realtime）切换到交互 backend 后阻塞显示。
"""
from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.joint_test.types import (
    TrackingResult,
    Metrics,
    SweepResult,
)

logger = logging.getLogger(__name__)


class ResultPlotter:
    """绘图组件：单次曲线 + Bode 图 + 对比模式 + 终端打印。

    Args:
        output_dir: 输出目录，不存在自动创建。
        backend: matplotlib 后端（默认 TkAgg，用于实时弹窗）。
    """

    def __init__(
        self,
        output_dir: Path,
        backend: str = "TkAgg",
    ) -> None:
        """初始化绘图器。"""
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._backend = backend

    @property
    def output_dir(self) -> Path:
        """输出目录。"""
        return self._out

    def plot_single(
        self,
        result: TrackingResult,
        metrics: Metrics,
        fake_result: TrackingResult | None = None,
    ) -> Path:
        """绘制单次测试图：3 子图（指令vs实际、误差、Lissajous 相图）。

        Args:
            result: 跟踪结果。
            metrics: 指标。
            fake_result: 可选，FakeRobot 完美跟踪基准线。

        Returns:
            保存的 PNG 文件路径。
        """
        j = result.config.joint_idx
        wf = result.config.waveform

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # 子图 1: 指令 vs 实际
        ax1 = axes[0]
        ax1.plot(result.time, result.q_desired[:, j], "b-", lw=1.5, label="q_desired")
        ax1.plot(result.time, result.q_actual[:, j], "r-", lw=1.5, label="q_actual")
        if fake_result is not None:
            ax1.plot(
                fake_result.time, fake_result.q_actual[:, j],
                "g--", lw=1.0, label="FakeRobot (perfect)",
            )
        ax1.set_ylabel("Joint angle (rad)")
        ax1.set_title(
            f"{wf.value.upper()} @ {result.config.frequency_hz:.2f} Hz, "
            f"A={result.config.amplitude_rad:.3f} rad, joint={j}, "
            f"backend={result.backend.value}"
        )
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # 子图 2: 跟踪误差
        ax2 = axes[1]
        ax2.plot(result.time, result.tracking_error, "k-", lw=1.0, label="error")
        ax2.axhline(
            y=metrics.rmse_rad, color="orange", ls="--", lw=0.8,
            label=f"±RMSE={metrics.rmse_rad:.4f}",
        )
        ax2.axhline(y=-metrics.rmse_rad, color="orange", ls="--", lw=0.8)
        ax2.set_ylabel("Tracking error (rad)")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        # 子图 3: 相图 (Lissajous)
        ax3 = axes[2]
        ax3.plot(
            result.q_desired[:, j], result.q_actual[:, j],
            "purple", lw=1.0,
        )
        lim_lo = min(result.q_desired[:, j].min(), result.q_actual[:, j].min())
        lim_hi = max(result.q_desired[:, j].max(), result.q_actual[:, j].max())
        ax3.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.5, label="perfect")
        ax3.set_xlabel("q_desired (rad)")
        ax3.set_ylabel("q_actual (rad)")
        ax3.set_title("Phase plot (Lissajous)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect("equal")

        plt.tight_layout()

        # 保存
        fname = (
            f"tracking_{result.backend.value}_{wf.value}_j{j}_"
            f"f{result.config.frequency_hz:.2f}_"
            f"A{result.config.amplitude_rad:.3f}.png"
        )
        out_path = self._out / fname
        fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_bode(self, sweep: SweepResult) -> Path:
        """绘制频率响应图：幅频 + 相频。

        Args:
            sweep: 批量扫频结果。

        Returns:
            保存的 PNG 文件路径。
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # 幅频
        ax1.plot(sweep.frequencies_hz, sweep.amplitude_ratios, "bo-", lw=1.5)
        ax1.axhline(
            y=1 / np.sqrt(2), color="r", ls="--", lw=0.8,
            label="-3dB (0.707)",
        )
        ax1.set_ylabel("Amplitude ratio A_act/A_des")
        ax1.set_title(
            f"Frequency Response — joint={sweep.joint_idx}, "
            f"A={sweep.amplitude_rad:.3f} rad, {sweep.waveform.value}"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")

        # 相频
        ax2.plot(sweep.frequencies_hz, sweep.phase_lags_deg, "ro-", lw=1.5)
        ax2.axhline(y=-45, color="gray", ls="--", lw=0.8, label="-45°")
        ax2.axhline(y=-90, color="gray", ls=":", lw=0.8, label="-90°")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase lag (deg)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log")

        plt.tight_layout()
        out_path = self._out / f"bode_j{sweep.joint_idx}.png"
        fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def show_realtime(
        self,
        result: TrackingResult,
        metrics: Metrics,
    ) -> None:
        """实验后弹窗显示（阻塞）。

        切换到交互 backend 后调用 plt.show()。
        """
        # 切换 backend（如果当前是 Agg）
        if matplotlib.get_backend().lower() == "agg":
            try:
                plt.switch_backend(self._backend)
            except Exception as e:
                logger.warning("无法切换 matplotlib backend: %s", e)
        self.plot_single(result, metrics)
        plt.show()  # 阻塞直到用户关闭窗口

    @staticmethod
    def print_metrics(metrics: Metrics) -> None:
        """格式化打印指标到 stdout。

        参照 scan_joint_safety.py 的输出风格。
        """
        print(f"\n{'=' * 60}")
        print(
            f"  跟踪性能指标 — {metrics.waveform.value.upper()} "
            f"@ joint {metrics.target_joint}"
        )
        print(f"{'=' * 60}")
        print(
            f"  RMSE:              {metrics.rmse_rad:.6f} rad "
            f"({np.degrees(metrics.rmse_rad):.4f}°)"
        )
        print(
            f"  最大误差:          {metrics.max_error_rad:.6f} rad "
            f"({np.degrees(metrics.max_error_rad):.4f}°)"
        )
        print(f"  平均误差:          {metrics.mean_error_rad:.6f} rad")

        if metrics.amplitude_ratio is not None:
            print(f"  幅值比 A_act/A_des: {metrics.amplitude_ratio:.4f}")
        if metrics.phase_lag_deg is not None:
            print(f"  相位滞后:          {metrics.phase_lag_deg:.2f}°")
        if metrics.rise_time_s is not None:
            print(f"  上升时间 (10-90%): {metrics.rise_time_s:.4f} s")
        if metrics.settling_time_s is not None:
            print(f"  Settling time (±2%): {metrics.settling_time_s:.4f} s")
        if metrics.overshoot_pct is not None:
            print(f"  超调量:            {metrics.overshoot_pct:.2f} %")
        if metrics.steady_state_error_rad is not None:
            print(
                f"  稳态误差:          {metrics.steady_state_error_rad:.6f} rad"
            )
        print(f"{'=' * 60}\n")
