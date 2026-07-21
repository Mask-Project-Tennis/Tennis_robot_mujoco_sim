"""math_utils 工具函数单元测试。

当前覆盖：
    - quintic_smoothstep: 五次多项式插值原语（M2 抽取自 replay_pipeline.py）
    - QUINTIC_SMOOTHSTEP_PEAK_FACTOR: 峰值速度因子常量
"""

from __future__ import annotations

from math import isclose, nextafter

import numpy as np
import pytest

from src.utils.math_utils import (
    QUINTIC_SMOOTHSTEP_PEAK_FACTOR,
    normalize,
    quintic_smoothstep,
)


class TestQuinticSmoothstep:
    """quintic_smoothstep 五次多项式平滑插值测试。"""

    def test_endpoints_zero_and_one(self) -> None:
        """t=0 → 0, t=1 → 1（标准 smoothstep 契约）。"""
        assert quintic_smoothstep(0.0) == 0.0
        assert quintic_smoothstep(1.0) == 1.0

    def test_midpoint_half(self) -> None:
        """t=0.5 → 0.5（对称性，s(1-t) = 1 - s(t)）。"""
        assert isclose(quintic_smoothstep(0.5), 0.5, abs_tol=1e-12)

    def test_monotone_increasing(self) -> None:
        """t∈[0, 1] 区间内单调递增（任意相邻样本后者 ≥ 前者）。"""
        ts = np.linspace(0.0, 1.0, 100)
        vals = quintic_smoothstep(ts)
        diffs = np.diff(vals)
        assert np.all(diffs >= 0), "smoothstep 在 [0,1] 应单调递增"

    def test_symmetry(self) -> None:
        """s(1-t) = 1 - s(t)：曲线关于 (0.5, 0.5) 中心对称。"""
        ts = np.linspace(0.0, 1.0, 50)
        forward = quintic_smoothstep(ts)
        mirror = quintic_smoothstep(1.0 - ts)
        np.testing.assert_allclose(mirror, 1.0 - forward, atol=1e-12)

    def test_zero_velocity_at_endpoints(self) -> None:
        """端点速度（解析导数）为零：s'(0)=s'(1)=0。

        s'(t) = 30t² - 60t³ + 30t⁴ = 30t²(1-t)²。
        """
        # 数值差分（ULP 级）验证导数为零
        eps = 1e-8
        v_left = (quintic_smoothstep(eps) - quintic_smoothstep(0.0)) / eps
        v_right = (quintic_smoothstep(1.0) - quintic_smoothstep(1.0 - eps)) / eps
        assert abs(v_left) < 1e-5, f"端点 s'(0+)≈{v_left} 应为零"
        assert abs(v_right) < 1e-5, f"端点 s'(1-)≈{v_right} 应为零"

    def test_zero_acceleration_at_endpoints(self) -> None:
        """端点加速度为零（C² 连续，无 jerk 突变）。

        s''(t) = 60t - 180t² + 120t³。
        """
        eps = 1e-4
        # 中心差分估计二阶导
        h = eps
        v_left = (
            quintic_smoothstep(h) - 2 * quintic_smoothstep(0.0) + quintic_smoothstep(-h)
        ) / (h * h)
        v_right = (
            quintic_smoothstep(1.0 + h)
            - 2 * quintic_smoothstep(1.0)
            + quintic_smoothstep(1.0 - h)
        ) / (h * h)
        # 多项式 s''(0) = 0, s''(1) = 60 - 180 + 120 = 0
        assert abs(v_left) < 1.0, f"端点 s''(0)≈{v_left} 应为零"
        assert abs(v_right) < 1.0, f"端点 s''(1)≈{v_right} 应为零"

    def test_peak_factor_constant_correct(self) -> None:
        """QUINTIC_SMOOTHSTEP_PEAK_FACTOR 常量 = 1.875（s'(0.5) 的精确值）。

        推导: s'(t) = 30t²(1-t)²
               s'(0.5) = 30 × 0.25 × 0.25 = 30/16 = 1.875
        """
        assert QUINTIC_SMOOTHSTEP_PEAK_FACTOR == 1.875

    def test_peak_velocity_at_midpoint(self) -> None:
        """s'(t) 的峰值在 t=0.5 处取到，值为 1.875。

        用数值差分验证峰值位置和峰值大小。
        """
        ts = np.linspace(0.0, 1.0, 1001)
        vals = quintic_smoothstep(ts)
        # 数值导数
        derivs = np.diff(vals) / np.diff(ts)
        peak_idx = int(np.argmax(derivs))
        # 峰值在中间附近（t≈0.5）
        assert abs(ts[peak_idx] - 0.5) < 0.01
        # 峰值大小 ≈ 1.875（数值差分有误差，放宽到 0.05）
        assert abs(derivs[peak_idx] - QUINTIC_SMOOTHSTEP_PEAK_FACTOR) < 0.05

    def test_accepts_ndarray(self) -> None:
        """支持 NDArray 输入（广播）。

        用途: replay_pipeline 多步插值、后摆批量计算等。
        """
        ts = np.linspace(0.0, 1.0, 11)
        vals = quintic_smoothstep(ts)
        assert isinstance(vals, np.ndarray)
        assert vals.shape == (11,)
        # 端点契约
        assert vals[0] == 0.0
        assert vals[-1] == 1.0

    def test_accepts_scalar(self) -> None:
        """标量输入返回标量（Python float）。"""
        v = quintic_smoothstep(0.3)
        assert isinstance(v, float)
        # 0 < t < 0.5 时，smoothstep < t（小值时多项式增长慢于线性）
        assert 0.0 < v < 0.3


class TestNormalizeRegression:
    """normalize 简单回归测试（保证 math_utils 现有 API 不破）。"""

    def test_unit_vector(self) -> None:
        """单位向量归一化后模为 1。"""
        v = np.array([3.0, 4.0])
        n = normalize(v)
        assert isclose(float(np.linalg.norm(n)), 1.0)

    def test_zero_vector(self) -> None:
        """零向量归一化返回零向量（不抛异常）。"""
        v = np.zeros(3)
        n = normalize(v)
        np.testing.assert_array_equal(n, np.zeros(3))
