"""单关节跟踪实验组件单元测试。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

from src.joint_test.analyzer import MetricsAnalyzer
from src.joint_test.experiment import TrackingExperiment
from src.joint_test.plotter import ResultPlotter
from src.joint_test.recorder import TrackingRecorder
from src.joint_test.robot_adapter import RobotAdapter
from src.joint_test.safety import JointSafetyGuard
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


class TestTypes:
    """数据结构测试。"""

    def test_waveform_type_enum_values(self):
        """WaveformType 包含 5 个正确枚举值。"""
        assert WaveformType.SINE.value == "sine"
        assert WaveformType.TRIANGLE.value == "triangle"
        assert WaveformType.SQUARE.value == "square"
        assert WaveformType.CHIRP.value == "chirp"
        assert WaveformType.STEP.value == "step"

    def test_tracking_result_computes_error(self):
        """TrackingResult 自动计算 tracking_error。"""
        cfg = WaveformConfig(
            waveform=WaveformType.SINE,
            joint_idx=2,
            frequency_hz=1.0,
            amplitude_rad=0.1,
            offset_rad=0.0,
            duration_s=1.0,
        )
        n = 10
        q_des = np.ones((n, 6))
        q_des[:, 2] = 0.5  # joint 2 的指令
        q_act = np.ones((n, 6))
        q_act[:, 2] = 0.4  # joint 2 的实际（有 0.1 误差）
        result = TrackingResult(
            config=cfg,
            backend=BackendType.SIM,
            dt=0.1,
            time=np.arange(n) * 0.1,
            q_desired=q_des,
            q_actual=q_act,
            qdot_actual=np.zeros((n, 6)),
        )
        assert result.tracking_error.shape == (n,)
        assert np.allclose(result.tracking_error, 0.1)

    def test_waveform_config_is_frozen(self):
        """WaveformConfig 不可变。"""
        cfg = WaveformConfig(
            waveform=WaveformType.SINE,
            joint_idx=0,
            frequency_hz=1.0,
            amplitude_rad=0.1,
            offset_rad=0.0,
            duration_s=1.0,
        )
        with pytest.raises((AttributeError, Exception)):
            cfg.frequency_hz = 2.0  # type: ignore[misc]

    def test_metrics_optional_fields_default_none(self):
        """Metrics 可选字段默认 None。"""
        m = Metrics(
            waveform=WaveformType.SINE,
            target_joint=0,
            rmse_rad=0.01,
            max_error_rad=0.02,
            mean_error_rad=0.005,
        )
        assert m.amplitude_ratio is None
        assert m.phase_lag_deg is None
        assert m.rise_time_s is None
        assert m.settling_time_s is None
        assert m.overshoot_pct is None
        assert m.steady_state_error_rad is None


class TestWaveformGenerator:
    """波形生成器测试。"""

    # ── 基础形状（5 tests）──

    def test_sine_correct_frequency_via_fft(self):
        """sine 波 FFT 主峰在指定频率。"""
        cfg = WaveformConfig(WaveformType.SINE, joint_idx=2, frequency_hz=1.0,
                             amplitude_rad=0.1, offset_rad=0.0, duration_s=2.0)
        gen = WaveformGenerator(cfg, np.zeros(6), dt=0.01)
        traj = gen.generate()
        assert traj.shape == (200, 6)
        # FFT 验证频率
        fft = np.fft.rfft(traj[:, 2])
        freqs = np.fft.rfftfreq(200, 0.01)
        peak_freq = freqs[np.argmax(np.abs(fft))]
        assert abs(peak_freq - 1.0) < 0.05

    def test_triangle_has_correct_peaks(self):
        """三角波峰谷值正确。"""
        cfg = WaveformConfig(WaveformType.TRIANGLE, joint_idx=0, frequency_hz=1.0,
                             amplitude_rad=0.5, offset_rad=0.0, duration_s=1.0)
        gen = WaveformGenerator(cfg, np.zeros(6), dt=0.001)
        traj = gen.generate()
        # 三角波峰值为 +A，谷值为 -A
        assert traj[:, 0].max() <= 0.5 + 1e-6
        assert traj[:, 0].min() >= -0.5 - 1e-6

    def test_square_amplitude_alternates(self):
        """方波在 ±A 之间切换。"""
        cfg = WaveformConfig(WaveformType.SQUARE, joint_idx=0, frequency_hz=1.0,
                             amplitude_rad=0.3, offset_rad=0.0, duration_s=2.0)
        gen = WaveformGenerator(cfg, np.zeros(6), dt=0.01)
        traj = gen.generate()
        # 方波只有两个值：+A 和 -A（加上 offset=0）
        unique_vals = np.unique(np.round(traj[:, 0], decimals=6))
        assert len(unique_vals) <= 2  # 允许数值误差
        # 注：pytest.approx 与 numpy array 的 __contains__ 不兼容，需转 list
        unique_list = unique_vals.tolist()
        assert pytest.approx(0.3) in unique_list or pytest.approx(-0.3) in unique_list

    def test_chirp_sweeps_frequency(self):
        """chirp 起始/终止频率正确。"""
        cfg = WaveformConfig(WaveformType.CHIRP, joint_idx=0, frequency_hz=1.0,
                             amplitude_rad=0.1, offset_rad=0.0, duration_s=2.0,
                             end_frequency_hz=10.0)
        gen = WaveformGenerator(cfg, np.zeros(6), dt=0.001)
        traj = gen.generate()
        # 起始段（前 100ms）频率接近 1Hz
        early = traj[:100, 0]
        # 末段（最后 100ms）频率接近 10Hz
        late = traj[-100:, 0]
        # 通过过零点数量验证频率增长
        early_zeros = np.sum(np.diff(np.sign(early)) != 0)
        late_zeros = np.sum(np.diff(np.sign(late)) != 0)
        assert late_zeros > early_zeros

    def test_step_transitions_at_10pct(self):
        """step 前 10% 保持，之后阶跃。"""
        cfg = WaveformConfig(WaveformType.STEP, joint_idx=0, frequency_hz=0,
                             amplitude_rad=0.5, offset_rad=0.0, duration_s=1.0,
                             step_target_rad=0.5)
        gen = WaveformGenerator(cfg, np.zeros(6), dt=0.01)
        traj = gen.generate()
        assert traj[0, 0] == 0.0       # 起始为 offset
        assert traj[-1, 0] == 0.5      # 终止为 target
        assert traj[5, 0] == 0.0       # 前 10% (100 步中的前 10 步) 保持 offset
        assert traj[15, 0] == 0.5      # 10% 后阶跃

    # ── 边界条件（9 tests）──

    def test_zero_amplitude_returns_constant(self):
        """A=0 → 全为 offset。"""
        cfg = WaveformConfig(WaveformType.SINE, joint_idx=0, frequency_hz=1.0,
                             amplitude_rad=0.0, offset_rad=0.7, duration_s=1.0)
        gen = WaveformGenerator(cfg, np.zeros(6), dt=0.01)
        traj = gen.generate()
        assert np.allclose(traj[:, 0], 0.7)

    def test_zero_frequency_sine_is_constant(self):
        """sine f=0 → 全为 offset（sin(0)=0）。"""
        cfg = WaveformConfig(WaveformType.SINE, joint_idx=0, frequency_hz=0.0,
                             amplitude_rad=0.1, offset_rad=0.3, duration_s=1.0)
        gen = WaveformGenerator(cfg, np.zeros(6), dt=0.01)
        traj = gen.generate()
        assert np.allclose(traj[:, 0], 0.3)

    def test_only_target_joint_moves(self):
        """其他关节保持 base_q。"""
        cfg = WaveformConfig(WaveformType.SINE, joint_idx=2, frequency_hz=1.0,
                             amplitude_rad=0.1, offset_rad=0.0, duration_s=1.0)
        base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        gen = WaveformGenerator(cfg, base, dt=0.01)
        traj = gen.generate()
        for j in [0, 1, 3, 4, 5]:
            assert np.allclose(traj[:, j], base[j])

    def test_invalid_joint_idx_minus1_raises(self):
        """joint_idx=-1 → ValueError 或 AssertionError。"""
        cfg = WaveformConfig(WaveformType.SINE, joint_idx=-1, frequency_hz=1.0,
                             amplitude_rad=0.1, offset_rad=0.0, duration_s=1.0)
        with pytest.raises((ValueError, AssertionError)):
            WaveformGenerator(cfg, np.zeros(6), dt=0.01)

    def test_invalid_joint_idx_6_raises(self):
        """joint_idx=6 → ValueError 或 AssertionError。"""
        cfg = WaveformConfig(WaveformType.SINE, joint_idx=6, frequency_hz=1.0,
                             amplitude_rad=0.1, offset_rad=0.0, duration_s=1.0)
        with pytest.raises((ValueError, AssertionError)):
            WaveformGenerator(cfg, np.zeros(6), dt=0.01)

    def test_chirp_without_end_freq_uses_default(self):
        """chirp 无 end_freq → 默认 10*f0。"""
        cfg = WaveformConfig(WaveformType.CHIRP, joint_idx=0, frequency_hz=2.0,
                             amplitude_rad=0.1, offset_rad=0.0, duration_s=1.0)
        gen = WaveformGenerator(cfg, np.zeros(6), dt=0.001)
        traj = gen.generate()  # 不应报错，使用默认 end_freq=20Hz
        assert traj.shape == (1000, 6)

    def test_step_without_target_uses_offset_plus_amp(self):
        """step 无 target → 默认 offset+A。"""
        cfg = WaveformConfig(WaveformType.STEP, joint_idx=0, frequency_hz=0,
                             amplitude_rad=0.4, offset_rad=0.1, duration_s=1.0)
        gen = WaveformGenerator(cfg, np.zeros(6), dt=0.01)
        traj = gen.generate()
        # 阶跃后应到 offset + A = 0.5
        assert traj[-1, 0] == pytest.approx(0.5)

    def test_step_target_equals_offset_flat_line(self):
        """step target==offset → 平直线。"""
        cfg = WaveformConfig(WaveformType.STEP, joint_idx=0, frequency_hz=0,
                             amplitude_rad=0.1, offset_rad=0.3, duration_s=1.0,
                             step_target_rad=0.3)
        gen = WaveformGenerator(cfg, np.zeros(6), dt=0.01)
        traj = gen.generate()
        assert np.allclose(traj[:, 0], 0.3)

    def test_duration_zero_returns_empty(self):
        """duration=0 → 空数组 (0, 6)。"""
        cfg = WaveformConfig(WaveformType.SINE, joint_idx=0, frequency_hz=1.0,
                             amplitude_rad=0.1, offset_rad=0.0, duration_s=0.0)
        gen = WaveformGenerator(cfg, np.zeros(6), dt=0.01)
        traj = gen.generate()
        assert traj.shape == (0, 6)

    # ── step_k 边界（M2, 1 test）──

    def test_step_waveform_short_trajectory_still_has_hold(self):
        """n_steps < 10 → 仍应有至少 1 步 hold 阶段（修复 M2）。"""
        # n_steps = 0.05 / 0.01 = 5（< 10），原代码 5//10=0 会丢失 hold
        cfg = WaveformConfig(WaveformType.STEP, joint_idx=0, frequency_hz=0,
                             amplitude_rad=0.5, offset_rad=0.0, duration_s=0.05,
                             step_target_rad=0.5)
        gen = WaveformGenerator(cfg, np.zeros(6), dt=0.01)
        traj = gen.generate()
        assert traj.shape == (5, 6)
        # 至少第一步应保持 offset=0.0（hold 阶段至少 1 步）
        assert traj[0, 0] == 0.0
        # 后续步应已阶跃到 target=0.5
        assert traj[1, 0] == 0.5
        assert traj[-1, 0] == 0.5


class TestSafetyGuard:
    """安全防护器测试。"""

    def _make_guard(
        self, max_A: float = 0.3, max_f: float = 2.0
    ) -> JointSafetyGuard:
        """构造测试用 guard（参数可调）。"""
        return JointSafetyGuard(
            q_lower=np.full(6, -1.0),
            q_upper=np.full(6, 1.0),
            qdot_max=np.full(6, 3.0),
            max_amplitude_rad=max_A,
            max_frequency_hz=max_f,
        )

    # ── 预检查（5 tests）──

    def test_safe_config_no_warnings(self):
        """安全参数 → 空警告列表。"""
        guard = self._make_guard(max_A=0.3, max_f=2.0)
        cfg = WaveformConfig(
            waveform=WaveformType.SINE, joint_idx=0,
            frequency_hz=1.0, amplitude_rad=0.2,
            offset_rad=0.0, duration_s=1.0,
        )
        assert guard.check_preconditions(cfg) == []

    def test_amplitude_exceeds_max_warns(self):
        """幅值超上限 → 警告包含"幅值"。"""
        guard = self._make_guard(max_A=0.3, max_f=2.0)
        cfg = WaveformConfig(
            waveform=WaveformType.SINE, joint_idx=0,
            frequency_hz=1.0, amplitude_rad=0.5,  # 超 0.3
            offset_rad=0.0, duration_s=1.0,
        )
        warnings = guard.check_preconditions(cfg)
        assert len(warnings) >= 1
        assert any("幅值" in w for w in warnings)

    def test_frequency_exceeds_max_warns(self):
        """频率超上限 → 警告包含"频率"。"""
        guard = self._make_guard(max_A=0.3, max_f=2.0)
        cfg = WaveformConfig(
            waveform=WaveformType.SINE, joint_idx=0,
            frequency_hz=5.0,  # 超 2.0
            amplitude_rad=0.1,
            offset_rad=0.0, duration_s=1.0,
        )
        warnings = guard.check_preconditions(cfg)
        assert len(warnings) >= 1
        assert any("频率" in w for w in warnings)

    def test_both_exceed_returns_two_warnings(self):
        """幅值+频率同时超 → 2 条警告。"""
        guard = self._make_guard(max_A=0.3, max_f=2.0)
        cfg = WaveformConfig(
            waveform=WaveformType.SINE, joint_idx=0,
            frequency_hz=5.0, amplitude_rad=0.5,  # 都超
            offset_rad=0.0, duration_s=1.0,
        )
        warnings = guard.check_preconditions(cfg)
        assert len(warnings) == 2

    def test_waveform_range_outside_joint_limits_warns(self):
        """波形范围超关节限位 → 警告包含"波形范围"。"""
        guard = self._make_guard(max_A=10.0, max_f=10.0)  # 故意放大 A/f 上限
        # 但关节限位是 ±1.0
        cfg = WaveformConfig(
            waveform=WaveformType.SINE, joint_idx=0,
            frequency_hz=1.0, amplitude_rad=0.5,
            offset_rad=0.8,  # offset+A = 1.3 超上限 1.0
            duration_s=1.0,
        )
        warnings = guard.check_preconditions(cfg)
        assert any("波形范围" in w for w in warnings)

    # ── 指令裁剪（5 tests）──

    def test_clip_command_in_range_unchanged(self):
        """范围内 → 原样返回。"""
        guard = self._make_guard()
        q_des = np.array([0.1, -0.2, 0.5, -0.3, 0.0, 0.4])
        q_cur = np.zeros(6)
        clipped = guard.clip_command(q_des, q_cur)
        assert np.allclose(clipped, q_des)

    def test_clip_command_upper_bound(self):
        """单个超上限 → 裁剪。"""
        guard = self._make_guard()
        q_des = np.array([0.1, 1.5, 0.5, -0.3, 0.0, 0.4])  # j=1 超 1.0
        q_cur = np.zeros(6)
        clipped = guard.clip_command(q_des, q_cur)
        assert clipped[1] == 1.0
        assert np.allclose(clipped[0], 0.1)

    def test_clip_command_lower_bound(self):
        """单个超下限 → 裁剪。"""
        guard = self._make_guard()
        q_des = np.array([0.1, -1.5, 0.5, -0.3, 0.0, 0.4])  # j=1 超 -1.0
        q_cur = np.zeros(6)
        clipped = guard.clip_command(q_des, q_cur)
        assert clipped[1] == -1.0

    def test_clip_command_nan_does_not_propagate(self):
        """NaN 输入 → 替换为 q_current，不传播。"""
        guard = self._make_guard()
        q_des = np.array([0.1, np.nan, 0.5, -0.3, 0.0, 0.4])
        q_cur = np.array([0.0, 0.7, 0.0, 0.0, 0.0, 0.0])
        clipped = guard.clip_command(q_des, q_cur)
        assert clipped[1] == 0.7  # NaN 被替换为 q_current[1]
        assert np.all(np.isfinite(clipped))

    def test_clip_command_at_boundary_passes(self):
        """恰好等于边界 → 通过（不裁剪）。"""
        guard = self._make_guard()
        q_des = np.array([1.0, -1.0, 0.5, -0.3, 0.0, 0.4])  # j=0,1 恰好边界
        q_cur = np.zeros(6)
        clipped = guard.clip_command(q_des, q_cur)
        assert clipped[0] == 1.0
        assert clipped[1] == -1.0

    # ── 真机默认参数（3 tests）──

    def test_default_max_amplitude_is_005(self):
        """默认 max_amplitude_rad = 0.05（保守）。"""
        guard = JointSafetyGuard(
            q_lower=np.full(6, -1.0), q_upper=np.full(6, 1.0),
            qdot_max=np.full(6, 3.0),
        )
        # 不传 max_amplitude_rad，应默认 0.05
        cfg = WaveformConfig(
            waveform=WaveformType.SINE, joint_idx=0,
            frequency_hz=0.5, amplitude_rad=0.06,  # 略超 0.05
            offset_rad=0.0, duration_s=1.0,
        )
        warnings = guard.check_preconditions(cfg)
        assert any("幅值" in w for w in warnings)

    def test_default_max_frequency_is_1hz(self):
        """默认 max_frequency_hz = 1.0（保守）。"""
        guard = JointSafetyGuard(
            q_lower=np.full(6, -1.0), q_upper=np.full(6, 1.0),
            qdot_max=np.full(6, 3.0),
        )
        cfg = WaveformConfig(
            waveform=WaveformType.SINE, joint_idx=0,
            frequency_hz=1.5,  # 略超 1.0
            amplitude_rad=0.02,
            offset_rad=0.0, duration_s=1.0,
        )
        warnings = guard.check_preconditions(cfg)
        assert any("频率" in w for w in warnings)

    def test_step_waveform_skips_frequency_check(self):
        """step 波形跳过频率检查（无频率概念）。"""
        guard = self._make_guard(max_A=0.3, max_f=0.001)  # f 上限极低
        cfg = WaveformConfig(
            waveform=WaveformType.STEP, joint_idx=0,
            frequency_hz=999.0,  # 故意超高，但 step 应跳过此检查
            amplitude_rad=0.1,
            offset_rad=0.0, duration_s=1.0,
            step_target_rad=0.1,
        )
        warnings = guard.check_preconditions(cfg)
        # 不应有频率警告
        assert not any("频率" in w for w in warnings)

    # ── 运行时状态检查 (C1, 3 tests) ──

    def test_runtime_state_all_safe(self):
        """所有关节在限位内 → (True, '')。"""
        guard = self._make_guard()  # q∈[-1,1], qdot_max=3.0
        q = np.array([0.1, -0.2, 0.5, -0.3, 0.0, 0.4])
        qd = np.array([0.5, -1.0, 2.0, -0.5, 0.0, 1.0])
        ok, reason = guard.check_runtime_state(q, qd)
        assert ok is True
        assert reason == ""

    def test_runtime_state_position_violation(self):
        """关节位置超上限 → (False, reason)。"""
        guard = self._make_guard()  # q∈[-1,1]
        q = np.array([0.1, 1.5, 0.5, -0.3, 0.0, 0.4])  # j=1 超 1.0
        qd = np.zeros(6)
        ok, reason = guard.check_runtime_state(q, qd)
        assert ok is False
        assert "关节 1" in reason
        assert "超上限" in reason

    def test_runtime_state_qdot_violation(self):
        """关节速度超限 → (False, reason)。"""
        guard = self._make_guard()  # qdot_max=3.0
        q = np.zeros(6)
        qd = np.array([0.5, 5.0, 0.0, 0.0, 0.0, 0.0])  # j=1 超 3.0
        ok, reason = guard.check_runtime_state(q, qd)
        assert ok is False
        assert "关节 1" in reason
        assert "速度" in reason

    def test_runtime_state_position_lower_violation(self):
        """关节位置超下限 → (False, reason)。"""
        guard = self._make_guard()  # q∈[-1,1]
        q = np.array([0.1, -1.5, 0.5, -0.3, 0.0, 0.4])  # j=1 超 -1.0
        qd = np.zeros(6)
        ok, reason = guard.check_runtime_state(q, qd)
        assert ok is False
        assert "关节 1" in reason
        assert "超下限" in reason

    def test_runtime_state_nan_qdot_violation(self):
        """qdot 含 NaN → (False, reason)。"""
        guard = self._make_guard()
        q = np.zeros(6)
        qd = np.array([0.0, np.nan, 0.0, 0.0, 0.0, 0.0])
        ok, reason = guard.check_runtime_state(q, qd)
        assert ok is False
        assert "NaN/Inf" in reason

    # ── q_current NaN 兜底 (M3, 1 test) ──

    def test_clip_command_q_current_nan_uses_zero(self):
        """q_current 含 NaN/Inf → 用零兜底，结果有限。"""
        guard = self._make_guard()
        q_des = np.array([0.1, np.nan, 0.5, -0.3, 0.0, 0.4])  # j=1 NaN
        q_cur = np.array([0.0, np.nan, 0.0, 0.0, 0.0, 0.0])   # j=1 NaN
        clipped = guard.clip_command(q_des, q_cur)
        # j=1: q_des NaN 降级为 q_current (NaN→0) → 0.0
        assert clipped[1] == 0.0
        assert np.all(np.isfinite(clipped))

    # ── Chirp 频率上限 (I4, 1 test) ──

    def test_chirp_end_frequency_bypasses_cap(self):
        """chirp: end_frequency 超 cap 时应触发警告（修复 I4）。"""
        guard = self._make_guard(max_A=0.3, max_f=2.0)
        cfg = WaveformConfig(
            waveform=WaveformType.CHIRP, joint_idx=0,
            frequency_hz=1.0,  # f0 未超 2.0
            amplitude_rad=0.1,
            offset_rad=0.0, duration_s=1.0,
            end_frequency_hz=10.0,  # 但 end_freq 超 2.0
        )
        warnings = guard.check_preconditions(cfg)
        assert any("频率" in w and "10.00" in w for w in warnings)

    def test_chirp_f0_within_cap_no_warning(self):
        """chirp: f0 和 end_freq 都在 cap 内 → 无频率警告。"""
        guard = self._make_guard(max_A=0.3, max_f=2.0)
        cfg = WaveformConfig(
            waveform=WaveformType.CHIRP, joint_idx=0,
            frequency_hz=0.5, amplitude_rad=0.1,
            offset_rad=0.0, duration_s=1.0,
            end_frequency_hz=1.5,
        )
        warnings = guard.check_preconditions(cfg)
        assert not any("频率" in w for w in warnings)


class TestTrackingRecorder:
    """数据记录器测试。"""

    def _make_recorder(self, duration_s: float = 1.0, dt: float = 0.1) -> TrackingRecorder:
        """构造测试用 recorder。"""
        cfg = WaveformConfig(
            waveform=WaveformType.SINE, joint_idx=2,
            frequency_hz=1.0, amplitude_rad=0.1,
            offset_rad=0.0, duration_s=duration_s,
        )
        return TrackingRecorder(cfg, dt=dt, backend=BackendType.SIM)

    def test_record_single_step(self):
        """单步记录数据正确。"""
        rec = self._make_recorder(duration_s=0.1, dt=0.1)  # 容量 1
        q_des = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        q_act = np.array([0.09, 0.2, 0.29, 0.4, 0.5, 0.6])
        qdot = np.zeros(6)
        rec.record(0.0, q_des, q_act, qdot)
        result = rec.finalize()
        assert result.time.shape == (1,)
        assert np.allclose(result.q_desired[0], q_des)
        assert np.allclose(result.q_actual[0], q_act)

    def test_record_full_capacity(self):
        """N 步填满。"""
        rec = self._make_recorder(duration_s=1.0, dt=0.1)  # 容量 10
        for k in range(10):
            rec.record(k * 0.1, np.zeros(6), np.zeros(6), np.zeros(6))
        result = rec.finalize()
        assert result.time.shape == (10,)
        assert result.q_desired.shape == (10, 6)

    def test_record_overflow_ignored(self):
        """超容量 → 静默丢弃，不报错。"""
        rec = self._make_recorder(duration_s=0.1, dt=0.1)  # 容量 1
        rec.record(0.0, np.zeros(6), np.zeros(6), np.zeros(6))
        rec.record(0.1, np.ones(6), np.ones(6), np.ones(6))  # 超容量
        result = rec.finalize()
        assert result.time.shape == (1,)  # 仅第 1 步
        assert np.allclose(result.q_desired[0], 0.0)  # 第 1 步数据

    def test_finalize_truncates_unfilled(self):
        """未填满 → 截断到实际记录数。"""
        rec = self._make_recorder(duration_s=1.0, dt=0.1)  # 容量 10
        rec.record(0.0, np.zeros(6), np.zeros(6), np.zeros(6))
        rec.record(0.1, np.zeros(6), np.zeros(6), np.zeros(6))
        result = rec.finalize()
        assert result.time.shape == (2,)  # 仅 2 步

    def test_save_npz_roundtrip(self, tmp_path: Path):
        """保存 NPZ 后加载，字段一致。"""
        rec = self._make_recorder(duration_s=0.2, dt=0.1)  # 容量 2
        rec.record(0.0, np.array([0.1]*6), np.array([0.05]*6), np.zeros(6))
        rec.record(0.1, np.array([0.2]*6), np.array([0.15]*6), np.zeros(6))
        result = rec.finalize()

        npz_path = tmp_path / "test.npz"
        TrackingRecorder.save_npz(result, npz_path)
        assert npz_path.exists()

        # 加载验证
        data = np.load(str(npz_path))
        assert np.allclose(data["time"], [0.0, 0.1])
        assert np.allclose(data["q_desired"], [[0.1]*6, [0.2]*6])
        assert np.allclose(data["q_actual"], [[0.05]*6, [0.15]*6])
        assert data["joint_idx"] == 2
        assert str(data["waveform"]) == "sine"

    def test_save_csv_header_and_data(self, tmp_path: Path):
        """CSV 表头+数据正确。"""
        rec = self._make_recorder(duration_s=0.2, dt=0.1)
        rec.record(0.0, np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
                   np.array([0.0, 0.0, 0.45, 0.0, 0.0, 0.0]), np.zeros(6))
        result = rec.finalize()

        csv_path = tmp_path / "test.csv"
        TrackingRecorder.save_csv(result, csv_path)
        assert csv_path.exists()

        # 加载验证（np.loadtxt 对单行 CSV 返回 1D，强制 2D）
        data = np.atleast_2d(np.loadtxt(str(csv_path), delimiter=",", skiprows=1))
        assert data.shape == (1, 5)  # 1 行 5 列
        assert np.isclose(data[0, 0], 0.0)  # time
        assert np.isclose(data[0, 1], 0.5)  # q_des joint 2
        assert np.isclose(data[0, 2], 0.45)  # q_act joint 2

    def test_tracking_error_auto_in_result(self):
        """TrackingResult 自动计算 tracking_error。"""
        rec = self._make_recorder(duration_s=0.1, dt=0.1)
        q_des = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        q_act = np.array([0.0, 0.0, 0.45, 0.0, 0.0, 0.0])
        rec.record(0.0, q_des, q_act, np.zeros(6))
        result = rec.finalize()
        # joint_idx=2, q_des[2]=0.5, q_act[2]=0.45 → error=0.05
        assert np.allclose(result.tracking_error, [0.05])


class TestMetricsAnalyzer:
    """指标分析器测试。"""

    def _make_result(
        self,
        q_des_1d: np.ndarray,
        q_act_1d: np.ndarray,
        joint_idx: int = 0,
        waveform: WaveformType = WaveformType.SINE,
        frequency_hz: float = 1.0,
        amplitude_rad: float = 0.5,
        dt: float = 0.001,
    ) -> TrackingResult:
        """构造测试用 TrackingResult（单关节）。"""
        n = len(q_des_1d)
        q_des = np.zeros((n, 6))
        q_act = np.zeros((n, 6))
        q_des[:, joint_idx] = q_des_1d
        q_act[:, joint_idx] = q_act_1d
        cfg = WaveformConfig(
            waveform=waveform, joint_idx=joint_idx,
            frequency_hz=frequency_hz, amplitude_rad=amplitude_rad,
            offset_rad=0.0, duration_s=n * dt,
        )
        return TrackingResult(
            config=cfg, backend=BackendType.SIM, dt=dt,
            time=np.arange(n) * dt,
            q_desired=q_des, q_actual=q_act,
            qdot_actual=np.zeros((n, 6)),
        )

    # ── 正弦/Chirp 分析（4 tests）──

    def test_perfect_tracking_zero_rmse(self):
        """完美跟踪 → RMSE=0。"""
        t = np.arange(0, 1.0, 0.001)
        q = 0.5 * np.sin(2 * np.pi * 2.0 * t)
        result = self._make_result(q, q.copy(), frequency_hz=2.0)
        m = MetricsAnalyzer().analyze(result)
        assert m.rmse_rad < 1e-9

    def test_known_amplitude_recovery(self):
        """已知 A_des=0.5, A_act=0.45 → 幅值比 ≈ 0.9。"""
        t = np.arange(0, 1.0, 0.001)
        q_des = 0.5 * np.sin(2 * np.pi * 2.0 * t)
        q_act = 0.45 * np.sin(2 * np.pi * 2.0 * t)
        result = self._make_result(q_des, q_act, frequency_hz=2.0)
        m = MetricsAnalyzer().analyze(result)
        assert abs(m.amplitude_ratio - 0.9) < 0.02

    def test_known_phase_lag_recovery(self):
        """已知相位滞后 45° → 反推误差 < 2°。"""
        dt = 0.001
        t = np.arange(0, 1.0, dt)
        f = 2.0
        q_des = 0.5 * np.sin(2 * np.pi * f * t)
        # 相位滞后 45° = π/4
        phase_lag = np.pi / 4
        q_act = 0.5 * np.sin(2 * np.pi * f * t - phase_lag)
        result = self._make_result(q_des, q_act, frequency_hz=f, dt=dt)
        m = MetricsAnalyzer().analyze(result)
        assert abs(m.phase_lag_deg - 45.0) < 2.0

    def test_amplitude_ratio_unit_for_equal_amp(self):
        """A_act=A_des → ratio=1.0。"""
        t = np.arange(0, 1.0, 0.001)
        q = 0.3 * np.sin(2 * np.pi * 5.0 * t)
        result = self._make_result(q, q.copy(), frequency_hz=5.0, amplitude_rad=0.3)
        m = MetricsAnalyzer().analyze(result)
        assert abs(m.amplitude_ratio - 1.0) < 0.01

    # ── Step 分析（3 tests）──

    def test_step_rise_time_correct(self):
        """合成阶跃 → rise_time 精确。"""
        dt = 0.01
        n = 200
        q_des = np.zeros(n)
        q_des[20:] = 1.0  # 第 20 步阶跃到 1.0
        # 实际：10 步内从 0 上升到 1.0（线性）
        q_act = np.zeros(n)
        for k in range(20, 30):
            q_act[k] = (k - 20) / 10.0  # 0.0 → 1.0
        q_act[30:] = 1.0
        result = self._make_result(
            q_des, q_act, waveform=WaveformType.STEP,
            frequency_hz=0, amplitude_rad=1.0, dt=dt,
        )
        m = MetricsAnalyzer().analyze(result)
        # 10%-90%: 从 0.1 到 0.9，即 q_act 从 k=21 到 k=29
        # rise_time = (29-21) * dt = 0.08s
        assert m.rise_time_s is not None
        assert abs(m.rise_time_s - 0.08) < 0.03  # 容忍 ±30ms

    def test_step_overshoot_zero_for_critical(self):
        """临界阻尼（无超调）→ overshoot=0。"""
        n = 200
        q_des = np.zeros(n)
        q_des[20:] = 1.0
        q_act = np.zeros(n)
        q_act[20:] = 1.0  # 完美跟踪，无超调
        result = self._make_result(
            q_des, q_act, waveform=WaveformType.STEP,
            frequency_hz=0, amplitude_rad=1.0,
        )
        m = MetricsAnalyzer().analyze(result)
        assert m.overshoot_pct == 0.0

    def test_step_no_settling_returns_full_duration(self):
        """不收敛 → settling_time = 全程。"""
        n = 100
        q_des = np.zeros(n)
        q_des[10:] = 1.0
        q_act = np.linspace(0, 0.5, n)  # 永远到不了 1.0
        result = self._make_result(
            q_des, q_act, waveform=WaveformType.STEP,
            frequency_hz=0, amplitude_rad=1.0, dt=0.01,
        )
        m = MetricsAnalyzer().analyze(result)
        # 最后仍偏离 target=1.0 超过 2%，所以 settle_k = len(q_act)
        assert m.settling_time_s is not None
        assert m.settling_time_s >= 0.99  # 接近全程

    # ── 退化情况（4 tests）──

    def test_zero_amplitude_signal_no_div_by_zero(self):
        """A=0 信号 → 不抛异常，返回 None 指标。"""
        n = 100
        q_flat = np.full(n, 0.3)  # 常量
        result = self._make_result(
            q_flat, q_flat, frequency_hz=1.0, amplitude_rad=0.0,
        )
        m = MetricsAnalyzer().analyze(result)
        # 常量信号 → amplitude_ratio=None
        assert m.amplitude_ratio is None
        assert m.rmse_rad == 0.0  # 完美跟踪常量

    def test_constant_signal_returns_none_amplitude(self):
        """常量信号 → amplitude_ratio=None。"""
        n = 100
        q_const = np.full(n, 0.5)
        result = self._make_result(q_const, q_const)
        m = MetricsAnalyzer().analyze(result)
        assert m.amplitude_ratio is None
        assert m.phase_lag_deg is None

    # ── 机器人卡死检测（I7, 1 test）──

    def test_robot_stuck_returns_zero_amplitude_ratio(self):
        """期望有内容但实际为常量 → amplitude_ratio=0.0（区分 N/A）。"""
        t = np.arange(0, 1.0, 0.01)
        q_des = 0.5 * np.sin(2 * np.pi * 2.0 * t)  # 期望正弦
        q_act = np.full_like(q_des, 0.0)  # 实际全 0（机器人卡死）
        result = self._make_result(q_des, q_act, frequency_hz=2.0)
        m = MetricsAnalyzer().analyze(result)
        # 修复 I7：以前返回 None（与 N/A 混淆），现在显式返回 0.0
        assert m.amplitude_ratio == 0.0
        assert m.phase_lag_deg is None  # 相位无法计算，仍为 None

    def test_short_signal_fft_handles_low_resolution(self):
        """N<10 短信号 → 不抛异常。"""
        q_des = np.array([0.0, 0.1, 0.2, 0.1, 0.0])
        q_act = np.array([0.0, 0.08, 0.18, 0.08, 0.0])
        result = self._make_result(q_des, q_act, dt=0.01)
        # 不应抛异常
        m = MetricsAnalyzer().analyze(result)
        assert m.rmse_rad >= 0.0  # 仅验证不崩

    def test_phase_wrapping_normalized(self):
        """相位差 >180° → 折回 [-180, 180]。"""
        dt = 0.001
        t = np.arange(0, 1.0, dt)
        f = 2.0
        q_des = 0.5 * np.sin(2 * np.pi * f * t)
        # 相位滞后 200°（应折回为 -160°）
        phase_lag = np.radians(200.0)
        q_act = 0.5 * np.sin(2 * np.pi * f * t - phase_lag)
        result = self._make_result(q_des, q_act, frequency_hz=f, dt=dt)
        m = MetricsAnalyzer().analyze(result)
        # 200° 折回为 -160°
        assert m.phase_lag_deg is not None
        assert -180.0 <= m.phase_lag_deg <= 180.0


class TestRobotAdapter:
    """机器人适配器测试。"""

    # ── 后端适配（4 tests）──

    def test_fake_robot_reset_sets_q(self):
        """FakeRobot reset 后 _q 正确。"""
        from src.joint_test.robot_adapter import RobotAdapter
        from src.real.fake_robot import FakeRobot

        fake = FakeRobot(np.zeros(6), dt=0.005)
        adapter = RobotAdapter(fake, backend=BackendType.FAKE)
        q0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        adapter.reset(q0)
        assert np.allclose(fake._q, q0)
        assert np.allclose(fake._qdot, 0.0)

    def test_fake_robot_step_returns_state(self):
        """FakeRobot step 后 get_q_qdot 返回正确状态。"""
        from src.joint_test.robot_adapter import RobotAdapter
        from src.real.fake_robot import FakeRobot

        fake = FakeRobot(np.zeros(6), dt=0.005)
        adapter = RobotAdapter(fake, backend=BackendType.FAKE)
        adapter.reset(np.zeros(6))
        q_des = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        adapter.step(q_des)
        q, qdot = adapter.get_q_qdot()
        assert np.allclose(q, q_des)
        # FakeRobot qdot = (q_des - q_old) / dt
        assert qdot.shape == (6,)

    def test_env_adapter_step_calls_mujoco(self):
        """RM65Env 适配器调用 step（集成测试）。"""
        try:
            from src.sim.rm65_env import RM65Env
        except Exception:
            pytest.skip("RM65Env 需要 MuJoCo，跳过")
        from pathlib import Path

        from src.joint_test.robot_adapter import RobotAdapter

        # RM65Env 需要 model_path 参数；用项目标准模型
        model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
        if not model_path.exists():
            pytest.skip(f"模型文件不存在: {model_path}")
        env = RM65Env(model_path, dt=0.005)
        adapter = RobotAdapter(env, backend=BackendType.SIM)
        q0 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        adapter.reset(q0)
        q, _ = adapter.get_q_qdot()
        assert np.allclose(q, q0, atol=1e-3)

    def test_backend_property_correct(self):
        """backend 属性返回正确枚举。"""
        from src.joint_test.robot_adapter import RobotAdapter
        from src.real.fake_robot import FakeRobot

        fake = FakeRobot(np.zeros(6), dt=0.005)
        adapter = RobotAdapter(fake, backend=BackendType.FAKE)
        assert adapter.backend == BackendType.FAKE

    # ── safety_guard 属性暴露（I5, 2 tests）──

    def test_safety_guard_property_returns_none_when_absent(self):
        """无 safety → safety_guard 属性为 None。"""
        from src.joint_test.robot_adapter import RobotAdapter
        from src.real.fake_robot import FakeRobot

        fake = FakeRobot(np.zeros(6), dt=0.005)
        adapter = RobotAdapter(fake, backend=BackendType.FAKE)
        assert adapter.safety_guard is None

    def test_safety_guard_property_returns_instance_when_set(self):
        """有 safety → safety_guard 属性返回同一实例。"""
        from src.joint_test.robot_adapter import RobotAdapter
        from src.real.fake_robot import FakeRobot

        fake = FakeRobot(np.zeros(6), dt=0.005)
        guard = JointSafetyGuard(
            q_lower=np.full(6, -0.1),
            q_upper=np.full(6, 0.1),
            qdot_max=np.full(6, 3.0),
        )
        adapter = RobotAdapter(
            fake, backend=BackendType.FAKE, safety_guard=guard,
        )
        assert adapter.safety_guard is guard

    # ── 安全集成（3 tests）──

    def test_adapter_with_safety_clips_command(self):
        """有 safety → q_des 被裁剪。"""
        from src.joint_test.robot_adapter import RobotAdapter
        from src.real.fake_robot import FakeRobot

        fake = FakeRobot(np.zeros(6), dt=0.005)
        guard = JointSafetyGuard(
            q_lower=np.full(6, -0.1),
            q_upper=np.full(6, 0.1),
            qdot_max=np.full(6, 3.0),
        )
        adapter = RobotAdapter(
            fake, backend=BackendType.FAKE, safety_guard=guard,
        )
        adapter.reset(np.zeros(6))
        # 下发超限指令
        q_des = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        adapter.step(q_des)
        q, _ = adapter.get_q_qdot()
        # 应被裁剪到 ±0.1
        assert np.all(q <= 0.1 + 1e-9)
        assert np.all(q >= -0.1 - 1e-9)

    def test_adapter_without_safety_no_clip(self):
        """无 safety → 直接下发。"""
        from src.joint_test.robot_adapter import RobotAdapter
        from src.real.fake_robot import FakeRobot

        fake = FakeRobot(np.zeros(6), dt=0.005)
        adapter = RobotAdapter(fake, backend=BackendType.FAKE)
        adapter.reset(np.zeros(6))
        q_des = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 超大值
        adapter.step(q_des)
        q, _ = adapter.get_q_qdot()
        assert np.allclose(q, q_des)  # 未裁剪

    def test_emergency_stop_propagates(self):
        """emergency_stop 调用底层。"""
        from src.joint_test.robot_adapter import RobotAdapter
        from src.real.fake_robot import FakeRobot

        fake = FakeRobot(np.zeros(6), dt=0.005)
        adapter = RobotAdapter(fake, backend=BackendType.FAKE)
        adapter.emergency_stop()
        assert fake.emergency_stop_count == 1

    # ── 异常处理（3 tests）──

    def test_step_calls_safety_before_send(self):
        """安全检查先于下发（通过观察裁剪效果验证）。"""
        from src.joint_test.robot_adapter import RobotAdapter
        from src.real.fake_robot import FakeRobot

        fake = FakeRobot(np.zeros(6), dt=0.005)
        guard = JointSafetyGuard(
            q_lower=np.full(6, -0.05),
            q_upper=np.full(6, 0.05),
            qdot_max=np.full(6, 3.0),
        )
        adapter = RobotAdapter(
            fake, backend=BackendType.FAKE, safety_guard=guard,
        )
        adapter.reset(np.zeros(6))
        adapter.step(np.array([1.0] * 6))  # 会被裁剪到 0.05
        # 第二步：当前已在 0.05，再下发 1.0 应再次裁剪
        adapter.step(np.array([1.0] * 6))
        q, _ = adapter.get_q_qdot()
        assert np.allclose(q, 0.05)

    def test_double_reset_does_not_break(self):
        """二次 reset 无副作用。"""
        from src.joint_test.robot_adapter import RobotAdapter
        from src.real.fake_robot import FakeRobot

        fake = FakeRobot(np.zeros(6), dt=0.005)
        adapter = RobotAdapter(fake, backend=BackendType.FAKE)
        q0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        adapter.reset(q0)
        adapter.reset(q0)  # 二次 reset
        q, _ = adapter.get_q_qdot()
        assert np.allclose(q, q0)

    def test_get_q_qdot_returns_copy(self):
        """返回数组是副本（防外部修改）。"""
        from src.joint_test.robot_adapter import RobotAdapter
        from src.real.fake_robot import FakeRobot

        fake = FakeRobot(np.zeros(6), dt=0.005)
        adapter = RobotAdapter(fake, backend=BackendType.FAKE)
        adapter.reset(np.array([0.1] * 6))
        q1, _ = adapter.get_q_qdot()
        q1[0] = 999.0  # 外部修改
        q2, _ = adapter.get_q_qdot()
        assert q2[0] != 999.0  # 内部状态未被影响

    # ── 通信失败处理 (C2, 1 test) ──

    def test_step_raises_on_comm_failure(self):
        """send_joint_command 返回非零 → step() 抛 RuntimeError + 急停。"""
        from src.joint_test.robot_adapter import RobotAdapter
        from src.real.fake_robot import FakeRobot

        fake = FakeRobot(np.zeros(6), dt=0.005)
        # 把 send_joint_command 改为返回错误码（模拟通信失败）
        original_send = fake.send_joint_command

        def failing_send(q_desired: np.ndarray) -> int:
            original_send(q_desired)
            return -1  # 非零错误码

        fake.send_joint_command = failing_send  # type: ignore[method-assign]
        adapter = RobotAdapter(fake, backend=BackendType.FAKE)
        adapter.reset(np.zeros(6))

        with pytest.raises(RuntimeError, match="机器人通信失败"):
            adapter.step(np.zeros(6))

        # 应同时触发急停
        assert fake.emergency_stop_count >= 1


class _CommFailStubRobot:
    """通信失败桩机器人（仅用于测试 C2）。

    实现 RobotArmInterface 协议所需的最小接口，send_joint_command 固定返回 -1。
    """

    def __init__(self) -> None:
        self._q = np.zeros(6)
        self._qdot = np.zeros(6)
        self.emergency_stop_count = 0

    def get_arm_state(self) -> np.ndarray:
        return np.concatenate([self._q.copy(), self._qdot.copy()])

    def send_joint_command(self, q_desired: np.ndarray) -> int:
        return -1  # 固定返回非零

    def emergency_stop(self) -> None:
        self.emergency_stop_count += 1


class TestCommFailureAdapter:
    """通信失败场景的适配器测试（C2）。"""

    def test_adapter_step_raises_runtime_error(self):
        """非 env 后端通信失败 → step() 抛 RuntimeError 并急停。"""
        from src.joint_test.robot_adapter import RobotAdapter

        robot = _CommFailStubRobot()
        adapter = RobotAdapter(robot, backend=BackendType.REAL)
        with pytest.raises(RuntimeError) as exc_info:
            adapter.step(np.zeros(6))
        assert "机器人通信失败" in str(exc_info.value)
        assert "错误码 -1" in str(exc_info.value)
        assert robot.emergency_stop_count == 1


class TestResultPlotter:
    """结果绘图器测试。"""

    def _make_result(
        self, n: int = 100, joint_idx: int = 2, dt: float = 0.01,
    ) -> TrackingResult:
        """构造测试用 TrackingResult。"""
        t = np.arange(n) * dt
        q_des = np.zeros((n, 6))
        q_act = np.zeros((n, 6))
        q_des[:, joint_idx] = 0.5 * np.sin(2 * np.pi * 2.0 * t)
        q_act[:, joint_idx] = 0.45 * np.sin(2 * np.pi * 2.0 * t - 0.3)
        cfg = WaveformConfig(
            waveform=WaveformType.SINE, joint_idx=joint_idx,
            frequency_hz=2.0, amplitude_rad=0.5,
            offset_rad=0.0, duration_s=n * dt,
        )
        return TrackingResult(
            config=cfg, backend=BackendType.SIM, dt=dt,
            time=t, q_desired=q_des, q_actual=q_act,
            qdot_actual=np.zeros((n, 6)),
        )

    def _make_metrics(self) -> Metrics:
        """构造测试用 Metrics。"""
        return Metrics(
            waveform=WaveformType.SINE, target_joint=2,
            rmse_rad=0.05, max_error_rad=0.1, mean_error_rad=0.04,
            amplitude_ratio=0.9, phase_lag_deg=17.2,
        )

    def test_plot_single_creates_png(self, tmp_path):
        """生成 PNG 文件存在。"""
        matplotlib.use("Agg")  # 避免弹窗
        plotter = ResultPlotter(tmp_path, backend="Agg")
        result = self._make_result()
        metrics = self._make_metrics()
        out_path = plotter.plot_single(result, metrics)
        assert out_path.exists()
        assert out_path.suffix == ".png"

    def test_plot_bode_creates_png(self, tmp_path):
        """Bode 图 PNG 存在。"""
        matplotlib.use("Agg")
        plotter = ResultPlotter(tmp_path, backend="Agg")
        sweep = SweepResult(
            joint_idx=2,
            waveform=WaveformType.SINE,
            amplitude_rad=0.1,
            frequencies_hz=np.array([0.1, 0.5, 1.0, 2.0, 5.0]),
            amplitude_ratios=np.array([1.0, 0.95, 0.85, 0.7, 0.4]),
            phase_lags_deg=np.array([-1, -5, -15, -35, -70]),
            rmses_rad=np.array([0.001, 0.005, 0.02, 0.05, 0.1]),
            individual_metrics=[],
        )
        out_path = plotter.plot_bode(sweep)
        assert out_path.exists()
        assert "bode" in out_path.name

    def test_plot_single_with_fake_overlay(self, tmp_path):
        """含 fake 基准线 → 不报错。"""
        matplotlib.use("Agg")
        plotter = ResultPlotter(tmp_path, backend="Agg")
        result = self._make_result()
        fake_result = self._make_result()  # 简化：用相同数据
        metrics = self._make_metrics()
        out_path = plotter.plot_single(result, metrics, fake_result=fake_result)
        assert out_path.exists()

    def test_print_metrics_contains_rmse(self, capsys):
        """stdout 包含 RMSE。"""
        metrics = self._make_metrics()
        ResultPlotter.print_metrics(metrics)
        captured = capsys.readouterr()
        assert "RMSE" in captured.out
        assert "幅值比" in captured.out
        assert "相位滞后" in captured.out

    def test_output_dir_auto_created(self, tmp_path):
        """不存在目录 → 自动创建。"""
        matplotlib.use("Agg")
        nested = tmp_path / "sub" / "dir"
        ResultPlotter(nested, backend="Agg")
        assert nested.exists()


class TestTrackingExperiment:
    """编排器测试。"""

    def _make_experiment(
        self,
        backend: BackendType = BackendType.FAKE,
        speed_ratio: float = 1.0,
        base_q=None,
    ) -> TrackingExperiment:
        """构造测试用 experiment（默认 FakeRobot 后端）。"""
        import numpy as np
        from src.real.fake_robot import FakeRobot
        import tempfile

        if base_q is None:
            base_q = np.zeros(6)
        fake = FakeRobot(np.asarray(base_q, dtype=float).copy(), dt=0.005)
        adapter = RobotAdapter(fake, backend=backend)
        analyzer = MetricsAnalyzer()
        plotter = ResultPlotter(Path(tempfile.mkdtemp()), backend="Agg")
        return TrackingExperiment(
            adapter, analyzer, plotter, dt=0.005,
            base_q=base_q, speed_ratio=speed_ratio, backend=backend,
        )

    def _make_test_config(
        self,
        waveform: WaveformType = WaveformType.SINE,
        frequency_hz: float = 1.0,
        amplitude_rad: float = 0.1,
        duration_s: float = 0.5,
        backend: BackendType = BackendType.FAKE,
        compare_fake: bool = False,
    ) -> TestConfig:
        """构造测试配置。"""
        wcfg = WaveformConfig(
            waveform=waveform, joint_idx=2,
            frequency_hz=frequency_hz, amplitude_rad=amplitude_rad,
            offset_rad=0.0, duration_s=duration_s,
        )
        return TestConfig(
            waveform_cfg=wcfg, backend=backend,
            save_npz=False, save_png=False, print_metrics=False,
            compare_fake=compare_fake,
        )

    # ── 端到端（8 tests）──

    def test_run_single_fake_zero_rmse(self):
        """FakeRobot 完美跟踪 → RMSE < 1e-3。"""
        exp = self._make_experiment()
        cfg = self._make_test_config()
        result, metrics = exp.run_single(cfg)
        assert metrics.rmse_rad < 1e-3

    def test_run_single_records_n_steps(self):
        """步数 = duration/dt。"""
        exp = self._make_experiment()
        cfg = self._make_test_config(duration_s=0.3)  # 0.3/0.005=60 步
        result, _ = exp.run_single(cfg)
        assert len(result.time) == 60

    def test_run_single_all_outputs_disabled(self):
        """全关 → 仅返回 result，不报错。"""
        exp = self._make_experiment()
        cfg = self._make_test_config()
        cfg.save_npz = False
        cfg.save_png = False
        cfg.print_metrics = False
        cfg.realtime_plot = False
        result, metrics = exp.run_single(cfg)
        assert result is not None
        assert metrics is not None

    def test_compare_fake_adds_baseline(self):
        """compare_fake=True + backend!=FAKE → _run_fake_baseline 被调用。"""
        exp = self._make_experiment(backend=BackendType.FAKE)
        # 用 monkey-patch spy 验证 _run_fake_baseline 确实被调用
        called = [False]
        original = exp._run_fake_baseline

        def spy(config: TestConfig) -> TrackingResult:
            called[0] = True
            return original(config)

        exp._run_fake_baseline = spy  # type: ignore[method-assign]
        # backend=SIM 触发 compare_fake 逻辑（即使内部 adapter 实为 FakeRobot）
        cfg = self._make_test_config(compare_fake=True, backend=BackendType.SIM)
        exp.run_single(cfg)
        assert called[0], "_run_fake_baseline 应被调用"

    def test_compare_fake_with_fake_backend_warns(self):
        """backend=FAKE + compare_fake → 不执行对比（无意义）。"""
        exp = self._make_experiment(backend=BackendType.FAKE)
        cfg = self._make_test_config(compare_fake=True, backend=BackendType.FAKE)
        # 应正常完成，不抛异常
        result, metrics = exp.run_single(cfg)
        assert metrics.rmse_rad < 1e-3  # FakeRobot 完美跟踪

    def test_run_sweep_empty_list_returns_empty(self):
        """空频率列表 → 空 SweepResult（长度 0）。"""
        exp = self._make_experiment()
        sweep = exp.run_sweep(
            joint_idx=2, frequencies_hz=[],
            amplitude_rad=0.1,
        )
        assert len(sweep.frequencies_hz) == 0
        assert len(sweep.amplitude_ratios) == 0

    def test_run_sweep_single_frequency(self):
        """单频率 → 长度 1。"""
        exp = self._make_experiment()
        sweep = exp.run_sweep(
            joint_idx=2, frequencies_hz=[1.0],
            amplitude_rad=0.1,
        )
        assert len(sweep.frequencies_hz) == 1
        assert sweep.frequencies_hz[0] == 1.0

    def test_run_sweep_multiple_frequencies(self):
        """多频率 → 长度正确。"""
        exp = self._make_experiment()
        sweep = exp.run_sweep(
            joint_idx=2,
            frequencies_hz=[0.5, 1.0, 2.0],
            amplitude_rad=0.1,
        )
        assert len(sweep.frequencies_hz) == 3

    def test_run_sweep_defaults_no_npz_no_metrics(self):
        """扫频默认 save_npz=False, print_metrics=False（修复 M5）。

        通过观察 plotter.output_dir 下不产生 .npz 文件验证 save_npz=False。
        """
        from pathlib import Path
        import tempfile

        # 显式构造一个独立 plotter，便于检查产物
        out_dir = Path(tempfile.mkdtemp())
        plotter = ResultPlotter(out_dir, backend="Agg")
        from src.real.fake_robot import FakeRobot
        fake = FakeRobot(np.zeros(6), dt=0.005)
        adapter = RobotAdapter(fake, backend=BackendType.FAKE)
        exp = TrackingExperiment(
            adapter, MetricsAnalyzer(), plotter, dt=0.005,
            base_q=np.zeros(6), speed_ratio=1.0, backend=BackendType.FAKE,
        )
        exp.run_sweep(
            joint_idx=2,
            frequencies_hz=[1.0],
            amplitude_rad=0.1,
            duration_s=0.1,
        )
        # 扫频默认 save_npz=False → 不应生成 .npz 文件（仅 bode.png）
        npz_files = list(out_dir.glob("*.npz"))
        assert len(npz_files) == 0, f"应默认不写 NPZ，但发现: {npz_files}"

    # ── speed_ratio 集成（3 tests）──

    def test_sim_speed_ratio_one_disables_pacing(self):
        """sim + ratio=1.0 → timer is None。"""
        exp = self._make_experiment(
            backend=BackendType.SIM, speed_ratio=1.0,
        )
        assert exp.timer is None

    def test_sim_speed_ratio_half_creates_timer(self):
        """sim + ratio=0.5 → timer 非 None（AdaptiveTimer 实例）。"""
        exp = self._make_experiment(
            backend=BackendType.SIM, speed_ratio=0.5,
        )
        assert exp.timer is not None

    def test_real_mode_always_creates_timer(self):
        """real + ratio=1.0 → timer 非 None（强制实时）。"""
        exp = self._make_experiment(
            backend=BackendType.REAL, speed_ratio=1.0,
        )
        assert exp.timer is not None


# ── subprocess + path 准备 ──
import subprocess  # noqa: E402
import sys  # noqa: E402


class TestCLIIntegration:
    """CLI 集成测试（subprocess 调用 scripts/tools/joint_tracking_test.py）。"""

    CLI_PATH = (
        Path(__file__).resolve().parent.parent
        / "scripts" / "tools" / "joint_tracking_test.py"
    )

    def _run_cli(self, *args: str) -> subprocess.CompletedProcess:
        """以子进程方式运行 CLI 命令。

        Args:
            *args: 命令行参数（不含 python 解释器与脚本路径）。

        Returns:
            subprocess.CompletedProcess 结果。
        """
        cmd = [sys.executable, str(self.CLI_PATH)] + list(args)
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
            cwd=str(Path(__file__).resolve().parent.parent),
        )

    def test_cli_fake_sine_runs(self):
        """fake + sine 短时长运行 → 退出码 0。"""
        result = self._run_cli(
            "--backend", "fake",
            "--waveform", "sine",
            "--joint", "2",
            "--freq", "1.0",
            "--amplitude", "0.1",
            "--duration", "0.5",
            "--no-plot",
            "--no-metrics",
            "--output-dir", "results/test_cli_fake",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_cli_real_without_flag_exits(self):
        """real 后端未加 --i-understand-real-risk → 非零退出且提示 flag。"""
        result = self._run_cli(
            "--backend", "real",
            "--waveform", "sine",
            "--joint", "2",
            "--no-plot",
        )
        assert result.returncode != 0
        assert (
            "i-understand-real-risk" in result.stderr
            or "i-understand-real-risk" in result.stdout
        ), f"stdout: {result.stdout}\nstderr: {result.stderr}"

    def test_cli_invalid_speed_rejected(self):
        """--speed 1.5 → argparse 拒绝。"""
        result = self._run_cli(
            "--backend", "fake",
            "--speed", "1.5",
            "--no-plot",
        )
        assert result.returncode != 0

    def test_cli_zero_speed_rejected(self):
        """--speed 0 → argparse 拒绝。"""
        result = self._run_cli(
            "--backend", "fake",
            "--speed", "0",
            "--no-plot",
        )
        assert result.returncode != 0

    def test_cli_negative_speed_rejected(self):
        """--speed -0.5 → argparse 拒绝。"""
        result = self._run_cli(
            "--backend", "fake",
            "--speed", "-0.5",
            "--no-plot",
        )
        assert result.returncode != 0

    # ── argparse BooleanOptionalAction（I2, 2 tests）──

    def test_cli_no_save_npz_flag_accepted(self):
        """--no-save-npz 应被接受（BooleanOptionalAction 修复 I2）。"""
        result = self._run_cli(
            "--backend", "fake",
            "--waveform", "sine",
            "--joint", "2",
            "--freq", "1.0",
            "--amplitude", "0.1",
            "--duration", "0.1",
            "--no-plot",
            "--no-metrics",
            "--no-save-npz",
            "--no-save-png",
            "--output-dir", "results/test_cli_no_npz",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # --no-save-npz 不应在 stderr 中产生 unrecognized arguments 错误
        assert "unrecognized arguments" not in result.stderr

    def test_cli_save_npz_default_true(self):
        """--save-npz 默认 True（BooleanOptionalAction 保持默认值）。"""
        result = self._run_cli(
            "--backend", "fake",
            "--waveform", "sine",
            "--joint", "2",
            "--duration", "0.1",
            "--no-plot",
            "--no-metrics",
            "--no-save-png",  # 关闭 PNG，但保留默认 NPZ
            "--output-dir", "results/test_cli_save_npz_default",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
