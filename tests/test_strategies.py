"""策略模块测试 — 阶段调度 / 方向 / 随挥 / 击球点 refine / 重规划模式。

每个策略均为独立可测试单元，不依赖完整 MPCController。
"""

from __future__ import annotations

import numpy as np

from src.ilqt.strategies.direction import (
    DirectionResult,
    DirectionPolicy,
    ReflectDirection,
)
from src.ilqt.strategies.phase_schedule import (
    DefaultPhaseSchedule,
    PhaseSchedule,
)


# ──────────────────────────────────────────────────────────────────
# A4 轮 1: PhaseSchedule.classify
# ──────────────────────────────────────────────────────────────────


class TestPhaseClassify:
    """阶段分类策略测试（far/mid/near）。"""

    def test_phase_classify_far(self) -> None:
        """k_hit=100 → 'far'（远离击球时刻）。"""
        schedule = DefaultPhaseSchedule()
        assert schedule.classify(100) == "far"

    def test_phase_classify_mid(self) -> None:
        """k_hit=30 → 'mid'（接近击球时刻）。"""
        schedule = DefaultPhaseSchedule()
        assert schedule.classify(30) == "mid"

    def test_phase_classify_near(self) -> None:
        """k_hit=10 → 'near'（临近击球时刻）。"""
        schedule = DefaultPhaseSchedule()
        assert schedule.classify(10) == "near"

    def test_phase_classify_boundary_far_mid(self) -> None:
        """k_hit=51 → 'far'（far/mid 边界 +1）。"""
        schedule = DefaultPhaseSchedule()
        assert schedule.classify(51) == "far"

    def test_phase_classify_boundary_mid_near(self) -> None:
        """k_hit=21 → 'mid'（mid/near 边界 +1）。"""
        schedule = DefaultPhaseSchedule()
        assert schedule.classify(21) == "mid"

    def test_phase_schedule_is_protocol(self) -> None:
        """DefaultPhaseSchedule 实现 PhaseSchedule Protocol。"""
        schedule = DefaultPhaseSchedule()
        assert isinstance(schedule, PhaseSchedule)


# ──────────────────────────────────────────────────────────────────
# A4 轮 2: ReflectDirection.compute
# ──────────────────────────────────────────────────────────────────


class TestReflectDirection:
    """来球反方向策略测试。"""

    def test_reflect_direction_basic(self) -> None:
        """ball_vel=[0,2,0] → d_hat=[0,-1,0]（来球反方向）。"""
        policy = ReflectDirection(target_speed=1.8)
        result = policy.compute(ball_vel=np.array([0.0, 2.0, 0.0]))
        assert isinstance(result, DirectionResult)
        np.testing.assert_allclose(result.d_hat, [0.0, -1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(result.d_follow, [0.0, -1.0, 0.0], atol=1e-9)

    def test_reflect_direction_v_hit_desired(self) -> None:
        """v_hit_desired = target_speed * d_hat。"""
        policy = ReflectDirection(target_speed=1.8)
        result = policy.compute(ball_vel=np.array([0.0, 0.0, -3.0]))
        np.testing.assert_allclose(result.v_hit_desired, [0.0, 0.0, 1.8], atol=1e-9)

    def test_reflect_direction_zero_vel_fallback(self) -> None:
        """零速度时 d_hat 回落到默认 [0,1,0]。"""
        policy = ReflectDirection(target_speed=1.8)
        result = policy.compute(ball_vel=np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result.d_hat, [0.0, 1.0, 0.0], atol=1e-9)

    def test_reflect_direction_is_protocol(self) -> None:
        """ReflectDirection 实现 DirectionPolicy Protocol。"""
        policy = ReflectDirection(target_speed=1.8)
        assert isinstance(policy, DirectionPolicy)
