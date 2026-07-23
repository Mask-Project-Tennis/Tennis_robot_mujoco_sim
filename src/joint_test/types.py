"""单关节跟踪实验的数据结构定义。

包含波形/后端枚举与实验配置/结果 dataclass，所有后续组件
（waveform/safety/recorder/analyzer/plotter/experiment）均依赖本模块。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class WaveformType(Enum):
    """波形类型枚举。"""

    SINE = "sine"
    TRIANGLE = "triangle"
    SQUARE = "square"
    CHIRP = "chirp"
    STEP = "step"


class BackendType(Enum):
    """执行后端类型枚举。"""

    SIM = "sim"      # RM65Env (MuJoCo)
    FAKE = "fake"    # FakeRobot (完美跟踪 Mock)
    REAL = "real"    # RobotInterface (真机 SDK)


@dataclass(frozen=True)
class WaveformConfig:
    """波形生成参数（不可变）。

    Attributes:
        waveform: 波形类型。
        joint_idx: 目标关节索引（0-5）。
        frequency_hz: 基础频率（Hz）。
        amplitude_rad: 幅值（弧度）。
        offset_rad: 直流偏置（默认 INIT_Q[joint_idx]）。
        duration_s: 实验时长（秒）。
        end_frequency_hz: Chirp 终止频率（Hz），仅 CHIRP 使用。
        step_target_rad: Step 目标角度（弧度），仅 STEP 使用。
    """

    waveform: WaveformType
    joint_idx: int                    # 0-5
    frequency_hz: float
    amplitude_rad: float
    offset_rad: float                 # 直流偏置（默认 INIT_Q[joint_idx]）
    duration_s: float
    end_frequency_hz: float | None = None  # Chirp 终止频率
    step_target_rad: float | None = None   # Step 目标角度


@dataclass
class TrackingResult:
    """单次跟踪实验的完整时序数据。

    Attributes:
        config: 波形配置。
        backend: 执行后端。
        dt: 控制周期（秒）。
        time: 时间序列 (N,)，仿真存 k*dt（合成世界时间）。
        q_desired: 指令关节角 (N, 6)。
        q_actual: 实际关节角 (N, 6)。
        qdot_actual: 实际关节速度 (N, 6)。
        tracking_error: 目标关节跟踪误差 (N,)，由 __post_init__ 自动计算。
        wall_time: 墙钟时间戳 (N,)。仿真存 k*dt（合成世界时间），真机存 perf_counter()（观测时刻）。默认 None 保持向后兼容。
    """

    config: WaveformConfig
    backend: BackendType
    dt: float
    time: np.ndarray                  # (N,)
    q_desired: np.ndarray             # (N, 6)
    q_actual: np.ndarray              # (N, 6)
    qdot_actual: np.ndarray           # (N, 6)
    tracking_error: np.ndarray = field(init=False)  # (N,) 自动计算
    wall_time: np.ndarray | None = None  # (N,) 墙钟时间戳，向后兼容

    def __post_init__(self) -> None:
        """自动计算 tracking_error = q_desired[:, joint_idx] - q_actual[:, joint_idx]。"""
        j = self.config.joint_idx
        object.__setattr__(
            self,
            "tracking_error",
            self.q_desired[:, j] - self.q_actual[:, j],
        )


@dataclass
class Metrics:
    """跟踪性能指标。

    None 表示该指标对此波形不适用。

    Attributes:
        waveform: 波形类型。
        target_joint: 目标关节索引。
        rmse_rad: 均方根误差（弧度）。
        max_error_rad: 最大绝对误差（弧度）。
        mean_error_rad: 平均误差（弧度）。
        amplitude_ratio: 输出/输入幅值比（仅周期波形适用）。
        phase_lag_deg: 相位滞后（度，仅周期波形适用）。
        rise_time_s: 上升时间（秒，仅 STEP 适用）。
        settling_time_s: 稳定时间（秒，仅 STEP 适用）。
        overshoot_pct: 超调百分比（仅 STEP 适用）。
        steady_state_error_rad: 稳态误差（弧度，仅 STEP 适用）。
    """

    waveform: WaveformType
    target_joint: int
    rmse_rad: float
    max_error_rad: float
    mean_error_rad: float
    amplitude_ratio: float | None = None
    phase_lag_deg: float | None = None
    rise_time_s: float | None = None
    settling_time_s: float | None = None
    overshoot_pct: float | None = None
    steady_state_error_rad: float | None = None


@dataclass
class SweepResult:
    """批量扫频聚合结果（用于 Bode 图）。

    Attributes:
        joint_idx: 扫频目标关节索引。
        waveform: 波形类型。
        amplitude_rad: 扫频激励幅值（弧度）。
        frequencies_hz: 扫频频率序列 (M,)。
        amplitude_ratios: 各频率下的幅值比 (M,)。
        phase_lags_deg: 各频率下的相位滞后（度）(M,)。
        rmses_rad: 各频率下的 RMSE（弧度）(M,)。
        individual_metrics: 各频率点对应的 Metrics 列表。
    """

    joint_idx: int
    waveform: WaveformType
    amplitude_rad: float
    frequencies_hz: np.ndarray        # (M,)
    amplitude_ratios: np.ndarray      # (M,)
    phase_lags_deg: np.ndarray        # (M,)
    rmses_rad: np.ndarray             # (M,)
    individual_metrics: list[Metrics]


@dataclass
class TestConfig:
    """单次测试的完整配置（传给编排器）。

    注：速度比例（speed_ratio）属于实验级配置，在 ``TrackingExperiment.__init__``
    设置，不应放在此处；否则会被静默忽略。

    Attributes:
        waveform_cfg: 波形配置。
        backend: 执行后端。
        realtime_plot: 是否实时绘图。
        save_npz: 是否保存 NPZ 数据。
        save_csv: 是否保存 CSV 数据。
        save_png: 是否保存 PNG 图表。
        print_metrics: 是否打印指标摘要。
        compare_fake: 是否同时跑 FakeRobot 作对照。
    """

    __test__ = False

    waveform_cfg: WaveformConfig
    backend: BackendType
    realtime_plot: bool = False
    save_npz: bool = True
    save_csv: bool = False
    save_png: bool = True
    print_metrics: bool = True
    compare_fake: bool = False
