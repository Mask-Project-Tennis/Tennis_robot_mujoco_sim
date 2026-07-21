# Task 2 Brief: waveform.py（5 种波形 + 14 tests）

## 任务背景

这是单关节跟踪实验的第 2 个任务。Task 1 已完成 `src/joint_test/types.py`（WaveformConfig 等数据结构）。本任务实现波形生成器，产生单个关节的期望轨迹 q_des(t)。

完整 spec: `docs/superpowers/specs/2026-07-20-joint-tracking-test-design.md`

## 依赖

- `src/joint_test/types.py`（Task 1 已完成，可用 `WaveformConfig` 和 `WaveformType`）
- `numpy`, `scipy.signal`（sawtooth, square, chirp）

## 你要做的事

### 1. 创建 `src/joint_test/waveform.py`

实现 `WaveformGenerator` 类：

```python
"""5 种波形生成器：sine/triangle/square/chirp/step。"""
from __future__ import annotations
import numpy as np
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
            from scipy.signal import sawtooth
            traj[:, j] = offset + A * sawtooth(2 * np.pi * f * t, width=0.5)
        elif self._cfg.waveform == WaveformType.SQUARE:
            from scipy.signal import square
            traj[:, j] = offset + A * square(2 * np.pi * f * t)
        elif self._cfg.waveform == WaveformType.CHIRP:
            from scipy.signal import chirp
            f1 = self._cfg.end_frequency_hz if self._cfg.end_frequency_hz is not None else (f * 10)
            traj[:, j] = offset + A * chirp(t, f0=f, f1=f1, t1=t[-1] if len(t) > 0 else 1.0, method="linear")
        elif self._cfg.waveform == WaveformType.STEP:
            target = self._cfg.step_target_rad if self._cfg.step_target_rad is not None else offset + A
            step_k = self._n_steps // 10  # 前 10% 时间保持 offset
            traj[:step_k, j] = offset
            traj[step_k:, j] = target
        else:
            raise ValueError(f"未知波形类型: {self._cfg.waveform}")

        return traj

    @property
    def time_array(self) -> np.ndarray:
        """时间序列 (N,)，单位 s。"""
        return self._t.copy()
```

### 2. 在 `tests/test_joint_tracking.py` 中添加 TestWaveformGenerator 类

**14 个测试**（5 形状 + 9 边界）：

```python
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
        assert pytest.approx(0.3) in unique_vals or pytest.approx(-0.3) in unique_vals

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
```

**重要**: 在 `tests/test_joint_tracking.py` 中**追加** `TestWaveformGenerator` 类，保留 Task 1 的 `TestTypes` 类。需要 import `WaveformGenerator`。

## 全局约束

- **所有 docstring 使用中文**（Google 风格）
- 类型提示必须标注
- `from __future__ import annotations`
- 禁止对数组用原生 Python 循环（用 numpy 向量化）
- 测试用 pytest

## 验证

```bash
# TDD: 先写测试（RED）
pytest tests/test_joint_tracking.py::TestWaveformGenerator -v
# 应该全部失败（ImportError 或失败）

# 实现 waveform.py 后（GREEN）
pytest tests/test_joint_tracking.py::TestWaveformGenerator -v
# 应该 14/14 通过

# 全部测试（确保 Task 1 不被破坏）
pytest tests/test_joint_tracking.py -v

# Lint + 类型
ruff check src/joint_test/waveform.py tests/test_joint_tracking.py
mypy src/joint_test/waveform.py
```

## 不要做的事

- 不要修改 Task 1 的 types.py
- 不要实现其他组件（safety/recorder 等）
- 不要在 `__init__.py` 中导出 WaveformGenerator
- 不要添加 spec 未列出的波形类型或参数
