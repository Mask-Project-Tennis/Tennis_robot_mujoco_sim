"""策略注入容器 — 允许外部注入自定义策略实现。

所有字段默认 None，表示使用 MPCConfig 驱动的默认实现。
注入非 None 值时，MPCController 使用注入的策略实例。

用法示例:
    config = StrategyConfig(direction_policy=TopspinDirection(...))
    mpc = MPCController(env, mpc_config, robot_limits, strategies=config)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ilqt.strategies.follow_through import FollowThroughPolicy
    from src.ilqt.strategies.hit_point_refiner import HitPointRefiner
    from src.ilqt.strategies.phase_schedule import PhaseSchedule
    from src.ilqt.strategies.direction import DirectionPolicy


@dataclass
class StrategyConfig:
    """策略注入容器 — None 字段使用 MPCConfig 驱动的默认实现。

    Attributes:
        follow_through: 随挥策略。None → PlannedFollowThrough（从 MPCConfig 构建）。
        hit_point_refiner: 击球点过滤策略。None → HybridRefiner。
        phase_schedule: 阶段调度策略。None → DefaultPhaseSchedule。
        direction_policy: 方向策略。None → ReflectDirection。
    """

    follow_through: "FollowThroughPolicy | None" = None
    hit_point_refiner: "HitPointRefiner | None" = None
    phase_schedule: "PhaseSchedule | None" = None
    direction_policy: "DirectionPolicy | None" = None
