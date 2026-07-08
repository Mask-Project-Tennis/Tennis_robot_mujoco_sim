"""重采样策略单元测试。

测试 InterpolatingResampler 的协议符合性、基本行为、时间戳正确性、
插值精度、target_dt 控制、边界条件和 C¹ 连续性。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.real.resample_strategy import InterpolatingResampler, ResampleStrategy


def _joint_signal(joint: int, t: np.ndarray) -> np.ndarray:
    """生成测试用第 joint 个关节的真值信号。

    与 test_multi_joint_all_interpolated 中 qs 的各列定义保持一致，
    用于独立验证每个关节的插值精度。

    Args:
        joint: 关节索引（0~5）。
        t: 时间数组。

    Returns:
        该关节在 t 上的真值角度序列。
    """
    if joint == 0:
        return np.sin(2 * np.pi * 1.0 * t)
    if joint == 1:
        return np.sin(2 * np.pi * 2.0 * t)
    if joint == 2:
        return np.cos(2 * np.pi * 1.0 * t)
    if joint == 3:
        return np.cos(2 * np.pi * 2.0 * t)
    if joint == 4:
        return t * 2.0
    return np.sin(2 * np.pi * 1.0 * t + 0.5)


class TestResampleStrategyProtocol:
    """验证 InterpolatingResampler 满足 ResampleStrategy Protocol。"""

    def test_satisfies_protocol(self) -> None:
        """InterpolatingResampler 实例是 ResampleStrategy 的运行时实例。"""
        r = InterpolatingResampler()
        assert isinstance(r, ResampleStrategy)

    def test_has_resample_method(self) -> None:
        """InterpolatingResampler 暴露可调用的 resample 方法。"""
        assert callable(getattr(InterpolatingResampler(), "resample", None))


class TestInterpolatingResamplerBasic:
    """基本重采样行为测试。"""

    def test_speed_factor_1_no_change(self) -> None:
        """speed_factor=1.0 + 均匀时间戳 → 点数不变、值一致。"""
        N = 100
        dt = 0.005
        ts = np.arange(N, dtype=np.float64) * dt
        qs = np.sin(2 * np.pi * 1.0 * ts).reshape(-1, 1).repeat(6, axis=1)
        r = InterpolatingResampler()
        new_qs, new_ts = r.resample(qs, ts, speed_factor=1.0)
        assert new_qs.shape[0] == N
        assert new_ts.shape[0] == N
        np.testing.assert_allclose(new_qs, qs, atol=1e-10)

    def test_speed_factor_0_1_more_points(self) -> None:
        """speed_factor=0.1 → 约 10 倍点数。"""
        N = 100
        dt = 0.005
        ts = np.arange(N, dtype=np.float64) * dt
        qs = np.sin(2 * np.pi * 1.0 * ts).reshape(-1, 1).repeat(6, axis=1)
        r = InterpolatingResampler()
        new_qs, new_ts = r.resample(qs, ts, speed_factor=0.1)
        # M = 10*(N-1)+1 = 991
        assert new_qs.shape[0] == 10 * (N - 1) + 1
        assert new_ts.shape[0] == 10 * (N - 1) + 1

    def test_speed_factor_0_1_original_values_match(self) -> None:
        """speed_factor=0.1 时原始数据点在重采样结果中精确重现。"""
        N = 100
        dt = 0.005
        ts = np.arange(N, dtype=np.float64) * dt
        qs = np.sin(2 * np.pi * 1.0 * ts).reshape(-1, 1).repeat(6, axis=1)
        r = InterpolatingResampler()
        new_qs, _ = r.resample(qs, ts, speed_factor=0.1)
        # 每 10 个采样点对应一个原始点（sample_t[10*i] == ts[i]）
        for i in range(N):
            np.testing.assert_allclose(new_qs[10 * i], qs[i], atol=1e-9)

    def test_speed_factor_2_half_points(self) -> None:
        """speed_factor=2.0 → 约一半点数。"""
        N = 101  # 100 段
        dt = 0.005
        ts = np.arange(N, dtype=np.float64) * dt
        qs = np.sin(2 * np.pi * 1.0 * ts).reshape(-1, 1).repeat(6, axis=1)
        r = InterpolatingResampler()
        new_qs, new_ts = r.resample(qs, ts, speed_factor=2.0)
        # T = 100*dt, T_new = 50*dt, M = int(50)+1 = 51
        assert new_qs.shape[0] == 51
        assert new_ts.shape[0] == 51


class TestInterpolatingResamplerTimestamps:
    """重采样时间戳正确性。"""

    def test_new_ts_starts_at_ts0(self) -> None:
        """new_ts[0] == ts[0]（非零起点）。"""
        N = 50
        dt = 0.01
        ts = np.arange(N, dtype=np.float64) * dt + 1.5
        qs = np.random.RandomState(0).randn(N, 6)
        r = InterpolatingResampler()
        _, new_ts = r.resample(qs, ts, speed_factor=1.0)
        assert new_ts[0] == pytest.approx(ts[0])

    def test_new_ts_spacing_equal_target_dt(self) -> None:
        """new_ts 间隔恒等于 target_dt。"""
        N = 50
        dt = 0.01
        ts = np.arange(N, dtype=np.float64) * dt
        qs = np.random.RandomState(0).randn(N, 6)
        r = InterpolatingResampler()
        _, new_ts = r.resample(qs, ts, speed_factor=0.5, target_dt=0.005)
        diffs = np.diff(new_ts)
        np.testing.assert_allclose(diffs, 0.005, atol=1e-15)

    def test_new_ts_uses_original_dt_when_none(self) -> None:
        """target_dt=None 时用原始 dt（中位数）。"""
        N = 50
        dt = 0.01
        ts = np.arange(N, dtype=np.float64) * dt
        qs = np.random.RandomState(0).randn(N, 6)
        r = InterpolatingResampler()
        _, new_ts = r.resample(qs, ts, speed_factor=1.0)
        np.testing.assert_allclose(np.diff(new_ts), dt, atol=1e-15)

    def test_new_ts_covers_stretched_duration(self) -> None:
        """new_ts 覆盖拉伸后时长 T/speed_factor。"""
        N = 100
        dt = 0.01
        ts = np.arange(N, dtype=np.float64) * dt
        qs = np.random.RandomState(0).randn(N, 6)
        r = InterpolatingResampler()
        _, new_ts = r.resample(qs, ts, speed_factor=0.1, target_dt=dt)
        T = ts[-1] - ts[0]
        T_new = T / 0.1
        # int 截断：new_ts[-1] <= ts[0] + T_new
        assert new_ts[-1] <= ts[0] + T_new + 1e-12
        # 至少覆盖到 T_new - dt（最后一个采样点不远于一个 dt）
        assert new_ts[-1] > ts[0] + T_new - dt


class TestInterpolatingResamplerAccuracy:
    """插值精度测试。"""

    def test_sine_wave_accuracy(self) -> None:
        """正弦波插值误差极小（密采样下 CubicSpline 近似精确）。"""
        N = 200
        dt = 0.005
        ts = np.arange(N, dtype=np.float64) * dt
        f = 1.0
        qs = np.sin(2 * np.pi * f * ts).reshape(-1, 1)
        r = InterpolatingResampler()
        new_qs, new_ts = r.resample(qs, ts, speed_factor=1.0, target_dt=0.002)
        # speed_factor=1.0 → sample_t == new_ts
        expected = np.sin(2 * np.pi * f * new_ts).reshape(-1, 1)
        np.testing.assert_allclose(new_qs, expected, atol=1e-4)

    def test_linear_exact(self) -> None:
        """线性轨迹插值精确（CubicSpline 再现线性函数）。"""
        N = 50
        dt = 0.01
        ts = np.arange(N, dtype=np.float64) * dt
        a, b = 3.0, 1.5
        qs = (a * ts + b).reshape(-1, 1)
        r = InterpolatingResampler()
        new_qs, new_ts = r.resample(qs, ts, speed_factor=1.0, target_dt=0.003)
        expected = (a * new_ts + b).reshape(-1, 1)
        np.testing.assert_allclose(new_qs, expected, atol=1e-10)

    def test_endpoints_match_original(self) -> None:
        """重采样首末点与原始首末点一致（speed_factor=1.0 + 均匀时间戳）。"""
        N = 100
        dt = 0.005
        ts = np.arange(N, dtype=np.float64) * dt
        qs = np.sin(2 * np.pi * 1.0 * ts).reshape(-1, 1).repeat(6, axis=1)
        r = InterpolatingResampler()
        new_qs, _ = r.resample(qs, ts, speed_factor=1.0)
        np.testing.assert_allclose(new_qs[0], qs[0], atol=1e-12)
        np.testing.assert_allclose(new_qs[-1], qs[-1], atol=1e-12)


class TestInterpolatingResamplerTargetDt:
    """target_dt 控制测试。"""

    def test_target_dt_100hz_point_count(self) -> None:
        """target_dt=0.01 (100Hz) → 正确点数。"""
        N = 101
        dt = 0.005
        ts = np.arange(N, dtype=np.float64) * dt  # T = 0.5s
        qs = np.random.RandomState(0).randn(N, 6)
        r = InterpolatingResampler()
        new_qs, _ = r.resample(qs, ts, speed_factor=1.0, target_dt=0.01)
        # M = int(0.5/0.01)+1 = 51
        assert new_qs.shape[0] == 51

    def test_target_dt_different_from_original_spacing(self) -> None:
        """target_dt 与原始 dt 不同 → 间隔正确。"""
        N = 100
        dt = 0.005
        ts = np.arange(N, dtype=np.float64) * dt
        qs = np.random.RandomState(0).randn(N, 6)
        r = InterpolatingResampler()
        _, new_ts = r.resample(qs, ts, speed_factor=1.0, target_dt=0.007)
        np.testing.assert_allclose(np.diff(new_ts), 0.007, atol=1e-15)


class TestInterpolatingResamplerEdgeCases:
    """边界条件测试。"""

    def test_empty_input_raises(self) -> None:
        """N=0 空输入 → ValueError。"""
        r = InterpolatingResampler()
        with pytest.raises(ValueError):
            r.resample(np.zeros((0, 6)), np.zeros(0), speed_factor=1.0)

    def test_single_point_returns_as_is(self) -> None:
        """N=1 单点 → 原样返回（无法插值）。"""
        r = InterpolatingResampler()
        qs = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        ts = np.array([0.5])
        new_qs, new_ts = r.resample(qs, ts, speed_factor=0.1)
        assert new_qs.shape == (1, 6)
        assert new_ts.shape == (1,)
        np.testing.assert_allclose(new_qs, qs)
        np.testing.assert_allclose(new_ts, ts)

    def test_speed_factor_zero_raises(self) -> None:
        """speed_factor=0 → ValueError。"""
        N = 10
        ts = np.arange(N, dtype=np.float64) * 0.01
        qs = np.random.RandomState(0).randn(N, 6)
        r = InterpolatingResampler()
        with pytest.raises(ValueError):
            r.resample(qs, ts, speed_factor=0.0)

    def test_speed_factor_negative_raises(self) -> None:
        """speed_factor<0 → ValueError。"""
        N = 10
        ts = np.arange(N, dtype=np.float64) * 0.01
        qs = np.random.RandomState(0).randn(N, 6)
        r = InterpolatingResampler()
        with pytest.raises(ValueError):
            r.resample(qs, ts, speed_factor=-1.0)

    def test_multi_joint_all_interpolated(self) -> None:
        """6 自由度全部正确插值（各关节独立信号）。"""
        N = 100
        dt = 0.005
        ts = np.arange(N, dtype=np.float64) * dt
        qs = np.column_stack(
            [
                np.sin(2 * np.pi * 1.0 * ts),
                np.sin(2 * np.pi * 2.0 * ts),
                np.cos(2 * np.pi * 1.0 * ts),
                np.cos(2 * np.pi * 2.0 * ts),
                ts * 2.0,
                np.sin(2 * np.pi * 1.0 * ts + 0.5),
            ]
        )
        r = InterpolatingResampler()
        new_qs, new_ts = r.resample(qs, ts, speed_factor=1.0, target_dt=0.002)
        assert new_qs.shape[1] == 6
        for j in range(6):
            expected_j = _joint_signal(j, new_ts)
            np.testing.assert_allclose(new_qs[:, j], expected_j, atol=1e-3)


class TestInterpolatingResamplerContinuity:
    """C¹ 连续性测试。"""

    def test_c1_continuous_sine(self) -> None:
        """正弦波重采样结果 C¹ 连续（二阶差分有界、无尖峰）。"""
        N = 200
        dt = 0.005
        ts = np.arange(N, dtype=np.float64) * dt
        qs = np.sin(2 * np.pi * 1.0 * ts).reshape(-1, 1).repeat(6, axis=1)
        r = InterpolatingResampler()
        new_qs, _ = r.resample(qs, ts, speed_factor=1.0, target_dt=0.001)
        d2 = np.diff(new_qs, n=2, axis=0)
        assert np.all(np.isfinite(d2))
        # 正弦二阶导 ≈ (2πf)²≈39.5，步长 0.001 → d2 ≈ 39.5e-6；无尖峰应远小于 0.01
        assert np.max(np.abs(d2)) < 0.01

    def test_c1_continuous_multi_joint(self) -> None:
        """多关节重采样结果 C¹ 连续（二阶差分无离群尖峰）。"""
        N = 200
        dt = 0.005
        ts = np.arange(N, dtype=np.float64) * dt
        qs = np.column_stack(
            [
                np.sin(2 * np.pi * 1.0 * ts),
                np.sin(2 * np.pi * 2.0 * ts),
                np.cos(2 * np.pi * 1.0 * ts),
            ]
        )
        r = InterpolatingResampler()
        new_qs, _ = r.resample(qs, ts, speed_factor=0.5, target_dt=0.002)
        d2 = np.diff(new_qs, n=2, axis=0)
        assert np.all(np.isfinite(d2))
        # 无尖峰：最大二阶差分 < 10× 中位数 + 容差
        med = float(np.median(np.abs(d2)))
        if med > 1e-12:
            assert float(np.max(np.abs(d2))) < 10.0 * med + 1e-9
