"""inspect_trajectory.py 纯函数单元测试。

测试三大检查函数:
  - check_joint_limits: 关节限位检查
  - check_smoothness: 平滑性（突跳）检查
  - estimate_tcp_speed: TCP 速度估计
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 统一导入模式：sys.path.insert + bare name（与 test_replay_trajectory.py 一致，
# 不依赖 namespace package + rootdir 注入）
_TOOLS_DIR = str(Path(__file__).resolve().parent.parent / "scripts" / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from inspect_trajectory import (  # noqa: E402
    check_joint_limits,
    check_smoothness,
    check_tcp_speed,
    estimate_tcp_speed,
)


# ============================================================================
# check_joint_limits
# ============================================================================


class TestCheckJointLimits:
    """关节限位检查测试。"""

    def test_all_within_bounds(self) -> None:
        """所有关节在限位内且有足够裕度 → 无告警。"""
        q = np.zeros((50, 6))
        q_lower_deg = np.array([-175, -270, -130, -175, -115, -180])
        q_upper_deg = np.array([175, 90, 130, 175, 115, 180])
        warnings = check_joint_limits(q, q_lower_deg, q_upper_deg, margin_deg=10.0)
        assert warnings == []

    def test_close_to_boundary(self) -> None:
        """关节接近边界（裕度内）→ 产生告警。"""
        q = np.zeros((10, 6))
        q[5, 0] = np.radians(170)  # 距 upper 175 仅 5° < margin 10°
        q_lower_deg = np.array([-175, -270, -130, -175, -115, -180])
        q_upper_deg = np.array([175, 90, 130, 175, 115, 180])
        warnings = check_joint_limits(q, q_lower_deg, q_upper_deg, margin_deg=10.0)
        assert len(warnings) == 1
        assert "J0" in warnings[0] or "J1" in warnings[0]

    def test_out_of_bounds(self) -> None:
        """关节超出限位 → 产生告警。"""
        q = np.zeros((10, 6))
        q[5, 2] = np.radians(200)  # 远超 J3 upper 130°
        q_lower_deg = np.array([-175, -270, -130, -175, -115, -180])
        q_upper_deg = np.array([175, 90, 130, 175, 115, 180])
        warnings = check_joint_limits(q, q_lower_deg, q_upper_deg, margin_deg=10.0)
        assert len(warnings) >= 1

    def test_negative_boundary(self) -> None:
        """关节接近下边界 → 产生告警。"""
        q = np.zeros((10, 6))
        q[3, 1] = np.radians(-260)  # 距 lower -270 仅 10°
        q_lower_deg = np.array([-175, -270, -130, -175, -115, -180])
        q_upper_deg = np.array([175, 90, 130, 175, 115, 180])
        warnings = check_joint_limits(q, q_lower_deg, q_upper_deg, margin_deg=10.0)
        assert len(warnings) >= 1


# ============================================================================
# check_smoothness
# ============================================================================


class TestCheckSmoothness:
    """平滑性检查测试。"""

    def test_smooth_trajectory(self) -> None:
        """平滑轨迹（小步增量）→ 无告警。"""
        dt = 0.005
        t = np.arange(100) * dt
        q = np.zeros((100, 6))
        q[:, 0] = 0.5 * np.sin(2 * np.pi * t)  # 平滑正弦
        warnings = check_smoothness(q, dt, threshold_deg=30.0)
        assert warnings == []

    def test_detects_large_jump(self) -> None:
        """大跳变（>30°/step）→ 产生告警。"""
        dt = 0.005
        q = np.zeros((20, 6))
        q[10, 0] = np.radians(45)  # 单步 45° 跳变
        warnings = check_smoothness(q, dt, threshold_deg=30.0)
        assert len(warnings) >= 1
        assert "J0" in warnings[0] or "J1" in warnings[0]

    def test_small_jump_ok(self) -> None:
        """小跳变（<30°/step）→ 无告警。"""
        dt = 0.005
        q = np.zeros((20, 6))
        q[10, 0] = np.radians(15)  # 单步 15° < 阈值 30°
        warnings = check_smoothness(q, dt, threshold_deg=30.0)
        assert warnings == []


# ============================================================================
# estimate_tcp_speed
# ============================================================================


class TestEstimateTcpSpeed:
    """TCP 速度估计测试。"""

    def test_constant_speed(self) -> None:
        """匀速直线运动 → 速度恒定。"""
        dt = 0.005
        n = 50
        tcp_pos = np.zeros((n, 3))
        tcp_pos[:, 0] = np.arange(n) * 0.1  # 0.1m/step = 20 m/s
        speeds = estimate_tcp_speed(tcp_pos, dt)
        expected_speed = 0.1 / dt  # 20 m/s
        # 中间步（非边界）应接近 expected_speed
        mid = n // 2
        assert np.isclose(speeds[mid], expected_speed, rtol=1e-3)

    def test_stationary(self) -> None:
        """静止 → 速度 ~0。"""
        dt = 0.005
        tcp_pos = np.ones((20, 3)) * 0.5  # 固定位置
        speeds = estimate_tcp_speed(tcp_pos, dt)
        assert np.all(speeds < 1e-6)

    def test_output_shape(self) -> None:
        """输出形状 == 输入行数。"""
        dt = 0.005
        tcp_pos = np.random.randn(30, 3)
        speeds = estimate_tcp_speed(tcp_pos, dt)
        assert speeds.shape == (30,)


# ============================================================================
# check_tcp_speed
# ============================================================================


class TestCheckTcpSpeed:
    """TCP 速度阈值校验测试。"""

    def test_within_threshold(self) -> None:
        """峰值 < max_tcp → 无告警。"""
        speeds = np.array([0.5, 1.0, 1.5, 1.0, 0.5])
        warnings, rec_speed = check_tcp_speed(speeds, max_tcp=2.0, firmware_tcp=1.0)
        assert warnings == []
        # peak=1.5, firmware=1.0 → rec_speed = 1.0/1.5 ≈ 0.667
        assert np.isclose(rec_speed, 1.0 / 1.5)

    def test_exceeds_threshold(self) -> None:
        """峰值 ≥ max_tcp → 告警 + 推荐 speed 反推。"""
        # 峰值 3.0 m/s，阈值 2.0
        speeds = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        warnings, rec_speed = check_tcp_speed(speeds, max_tcp=2.0, firmware_tcp=1.0)
        assert len(warnings) == 1
        assert "超限" in warnings[0]
        # peak=3.0, firmware=1.0 → rec_speed = 1.0/3.0 ≈ 0.333
        assert np.isclose(rec_speed, 1.0 / 3.0)

    def test_empty_input(self) -> None:
        """空数组 → 无告警，rec_speed=1.0（不除零）。"""
        speeds = np.array([])
        warnings, rec_speed = check_tcp_speed(speeds, max_tcp=2.0, firmware_tcp=1.0)
        assert warnings == []
        assert rec_speed == 1.0

    def test_zero_peak(self) -> None:
        """全零速度（静止轨迹）→ 不除零，rec_speed=1.0。"""
        speeds = np.zeros(10)
        warnings, rec_speed = check_tcp_speed(speeds, max_tcp=2.0, firmware_tcp=1.0)
        assert warnings == []
        assert rec_speed == 1.0

    def test_custom_firmware_tcp(self) -> None:
        """自定义固件 TCP → rec_speed 按自定义值反推。"""
        speeds = np.array([2.0])  # peak=2.0
        _, rec_speed = check_tcp_speed(speeds, max_tcp=2.0, firmware_tcp=0.5)
        # firmware=0.5, peak=2.0 → rec_speed = 0.5/2.0 = 0.25
        assert np.isclose(rec_speed, 0.25)


# ============================================================================
# _load_limits 回归（防 RealRobotConfig 默认值漂移）
# ============================================================================


class TestLoadLimitsDefaults:
    """_load_limits 无 --config 时返回 RealRobotConfig 默认值（防硬编码漂移）。"""

    def test_defaults_match_known_values(self) -> None:
        """无 --config 返回的限位应与项目既定值一致（J2 upper=90° 等）。"""
        from inspect_trajectory import _load_limits
        q_lower_deg, q_upper_deg, firmware_tcp = _load_limits(None)
        # 这些值来自 RealRobotConfig dataclass 默认（src/real/config.py:46-55）
        # 与 configs/real_robot.yaml 一致。断言关键 J2 上限 90°。
        assert q_upper_deg[1] == 90.0  # J2 upper
        assert q_lower_deg[1] == -270.0  # J2 lower
        assert firmware_tcp == 1.0  # max_tcp_speed 默认
