"""组合 5 组件，编排单次/批量跟踪实验。"""
from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np
from src.real.adaptive_timer import AdaptiveTimer

from src.joint_test.types import (
    BackendType,
    Metrics,
    SweepResult,
    TestConfig,
    TrackingResult,
    WaveformConfig,
    WaveformType,
)
from src.joint_test.waveform import WaveformGenerator
from src.joint_test.robot_adapter import RobotAdapter
from src.joint_test.recorder import TrackingRecorder
from src.joint_test.analyzer import MetricsAnalyzer
from src.joint_test.plotter import ResultPlotter

logger = logging.getLogger(__name__)


class TrackingExperiment:
    """编排器：依赖注入组件，运行单次或批量实验。

    通过 speed_ratio 控制实验节奏:
        - speed_ratio=1.0 (默认): Sim 无 pacing，Real 实时
        - speed_ratio<1.0: 慢放（用于观察）

    Args:
        robot_adapter: 机器人适配器。
        analyzer: 指标分析器。
        plotter: 结果绘图器。
        dt: 时间步长 (s)。
        base_q: 基础关节角度 (6,)，弧度。
        speed_ratio: 速度比例 (0, 1.0]，默认 1.0。
        backend: 后端类型（决定 pacing 策略）。
    """

    def __init__(
        self,
        robot_adapter: RobotAdapter,
        analyzer: MetricsAnalyzer,
        plotter: ResultPlotter,
        dt: float,
        base_q,
        speed_ratio: float = 1.0,
        backend: BackendType = BackendType.SIM,
    ) -> None:
        """初始化编排器，根据 backend 和 speed_ratio 配置 timer。"""
        if not 0.0 < speed_ratio <= 1.0:
            raise ValueError(
                f"speed_ratio 必须在 (0, 1.0]，得到 {speed_ratio}"
            )

        self._adapter = robot_adapter
        self._analyzer = analyzer
        self._plotter = plotter
        self._dt = dt
        self._base_q = np.asarray(base_q, dtype=float)
        self._speed_ratio = speed_ratio
        self._backend = backend
        # 根据 backend 和 speed_ratio 配置 AdaptiveTimer
        timer: AdaptiveTimer | None
        if backend == BackendType.REAL:
            # 真机：始终用 AdaptiveTimer（speed_ratio=1.0 即实时）
            target_hz = (1.0 / dt) * speed_ratio
            timer = AdaptiveTimer(target_hz=target_hz)
        elif speed_ratio < 1.0:
            # 仿真 + 慢放：用 AdaptiveTimer
            target_hz = (1.0 / dt) * speed_ratio
            timer = AdaptiveTimer(target_hz=target_hz)
        else:
            # 仿真 + 最大速度：不引入 pacing
            timer = None
        self._timer = timer

    def run_single(
        self, config: TestConfig,
    ) -> tuple[TrackingResult, Metrics]:
        """运行单次跟踪实验。

        Args:
            config: 测试配置。

        Returns:
            (TrackingResult, Metrics)。
        """
        wcfg = config.waveform_cfg

        # 1. 构造本次实验的组件实例（每次新建，状态隔离）
        wave_gen = WaveformGenerator(wcfg, self._base_q, self._dt)
        recorder = TrackingRecorder(wcfg, self._dt, config.backend)

        # 2. 生成期望轨迹
        q_traj = wave_gen.generate()
        logger.info(
            "启动跟踪实验: %s j=%d f=%.2fHz A=%.3frad backend=%s",
            wcfg.waveform.value, wcfg.joint_idx, wcfg.frequency_hz,
            wcfg.amplitude_rad, config.backend.value,
        )

        # 3. 主循环：逐步下发 + 记录 + 节奏控制
        # 真机模式必须捕获 KeyboardInterrupt，否则 Ctrl+C 会跳过急停直接退出，
        # 留下机器人继续执行最后一条指令，存在物理安全风险。
        try:
            for k in range(len(q_traj)):
                if self._timer is not None:
                    self._timer.tick_start()

                q_act, qdot_act, wall_ts = self._adapter.step(q_traj[k])
                recorder.record(k * self._dt, q_traj[k], q_act, qdot_act, wall_ts)

                # 运行时安全监控：每步检查实际 q/qdot 是否越限（真机必备）
                guard = self._adapter.safety_guard
                if guard is not None:
                    ok, reason = guard.check_runtime_state(q_act, qdot_act)
                    if not ok:
                        self._adapter.emergency_stop()
                        raise RuntimeError(f"运行时安全违规: {reason}")

                if self._timer is not None:
                    sleep_dt = self._timer.tick_end()
                    if sleep_dt > 0:
                        time.sleep(sleep_dt)
        except KeyboardInterrupt:
            # 用户中断（Ctrl+C）：立即急停机器人，避免危险动作残留
            self._adapter.emergency_stop()
            logger.warning("KeyboardInterrupt 已触发，机器人已急停")
            raise

        # 5. 收集结果 + 分析
        result = recorder.finalize()
        metrics = self._analyzer.analyze(result)

        # 6. 可选：对比模式（FakeRobot 完美跟踪基准）
        fake_result = None
        if config.compare_fake and config.backend != BackendType.FAKE:
            fake_result = self._run_fake_baseline(config)

        # 7. 输出
        self._output(result, metrics, config, fake_result)
        return result, metrics

    def _run_fake_baseline(self, config: TestConfig) -> TrackingResult:
        """用 FakeRobot 跑同一波形，作为完美跟踪基准。"""
        from src.real.fake_robot import FakeRobot

        fake = FakeRobot(self._base_q.copy(), dt=self._dt)
        adapter = RobotAdapter(
            fake, BackendType.FAKE, safety_guard=None,
        )

        wcfg = config.waveform_cfg
        wave_gen = WaveformGenerator(wcfg, self._base_q, self._dt)
        recorder = TrackingRecorder(wcfg, self._dt, BackendType.FAKE)

        q_traj = wave_gen.generate()
        if len(q_traj) > 0:
            adapter.reset(q_traj[0])
        for k in range(len(q_traj)):
            q, qd, wall_ts = adapter.step(q_traj[k])
            recorder.record(k * self._dt, q_traj[k], q, qd, wall_ts)

        return recorder.finalize()

    def _output(
        self,
        result: TrackingResult,
        metrics: Metrics,
        config: TestConfig,
        fake_result: TrackingResult | None,
    ) -> None:
        """根据 TestConfig 触发各类输出。"""
        wcfg = config.waveform_cfg
        tag = (
            f"{wcfg.waveform.value}_j{wcfg.joint_idx}_"
            f"f{wcfg.frequency_hz:.2f}"
        )

        if config.print_metrics:
            ResultPlotter.print_metrics(metrics)

        if config.save_npz:
            TrackingRecorder.save_npz(
                result, self._plotter.output_dir / f"{tag}.npz",
            )
        if config.save_csv:
            TrackingRecorder.save_csv(
                result, self._plotter.output_dir / f"{tag}.csv",
            )
        if config.save_png:
            self._plotter.plot_single(
                result, metrics, fake_result=fake_result,
            )
        if config.realtime_plot:
            self._plotter.show_realtime(result, metrics)

    def run_sweep(
        self,
        joint_idx: int,
        frequencies_hz: list[float],
        amplitude_rad: float,
        waveform: WaveformType = WaveformType.SINE,
        duration_s: float = 3.0,
        compare_fake: bool = False,
        save_npz: bool = False,
        print_metrics: bool = False,
        pre_sweep_home: Callable[[], None] | None = None,
    ) -> SweepResult:
        """批量扫频，生成 Bode 图数据。

        Args:
            joint_idx: 测试关节 (0-5)。
            frequencies_hz: 频率列表 (Hz)。
            amplitude_rad: 幅值 (rad)。
            waveform: 波形类型（默认 SINE）。
            duration_s: 单次持续时间 (s)。
            compare_fake: 是否附加 FakeRobot 基准。
            save_npz: 是否保存 NPZ 数据（扫频默认 False，避免重复写盘）。
            print_metrics: 是否打印单次指标（扫频默认 False，避免日志刷屏）。

        Returns:
            SweepResult 聚合结果。
        """
        individual_metrics: list[Metrics] = []
        amp_ratios: list[float] = []
        phase_lags: list[float] = []
        rmses: list[float] = []

        for f in frequencies_hz:
            wcfg = WaveformConfig(
                waveform=waveform,
                joint_idx=joint_idx,
                frequency_hz=f,
                amplitude_rad=amplitude_rad,
                offset_rad=float(self._base_q[joint_idx]),
                duration_s=duration_s,
            )
            cfg = TestConfig(
                waveform_cfg=wcfg,
                backend=self._backend,
                realtime_plot=False,
                save_npz=save_npz,
                save_png=False,  # 批量模式不画单次图
                print_metrics=print_metrics,
                compare_fake=compare_fake,
            )

            # 真机扫频：每频点前归位到 base_q，消除上一轮残余误差
            if self._backend == BackendType.REAL and pre_sweep_home is not None:
                pre_sweep_home()

            _, m = self.run_single(cfg)
            individual_metrics.append(m)
            amp_ratios.append(m.amplitude_ratio if m.amplitude_ratio is not None else 0.0)
            phase_lags.append(m.phase_lag_deg if m.phase_lag_deg is not None else 0.0)
            rmses.append(m.rmse_rad)

        sweep = SweepResult(
            joint_idx=joint_idx,
            waveform=waveform,
            amplitude_rad=amplitude_rad,
            frequencies_hz=np.array(frequencies_hz),
            amplitude_ratios=np.array(amp_ratios),
            phase_lags_deg=np.array(phase_lags),
            rmses_rad=np.array(rmses),
            individual_metrics=individual_metrics,
        )
        # 空频率列表跳过绘图（log-scale 无数据会报错）
        if len(frequencies_hz) > 0:
            self._plotter.plot_bode(sweep)
        return sweep

    @property
    def timer(self):
        """暴露 timer 用于测试验证（None 或 AdaptiveTimer）。

        Returns:
            AdaptiveTimer 实例或 None。
        """
        return self._timer
