"""重演轨迹数据结构 — 仿真侧记录 + 真机侧消费的共享类型。

ReplayTrajectory: 完整轨迹（保存/加载/回放用）
StepState: 单步状态（在 Sink 间共享的轻量数据载体）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReplayTrajectory:
    """重演轨迹数据结构（仿真侧记录 + 真机侧消费）。

    Attributes:
        q_desired: (N, 6) 规划输出的目标关节角度（弧度）。
        q_actual: (N, 6) 实际执行的关节角度（弧度）。
        timestamps: (N,) 每步时间戳（秒，相对 episode 起点）。
        tcp_pos: (N, 3) 末端执行器位置。
        ball_pos: (N, 3) 球位置（空挥时为零）。
        init_q: (6,) 右臂初始关节角度。
        init_q_left: (6,) 左臂初始关节角度。
        dt: 仿真步长（秒）。
        hit_step: 击球步索引（-1=未击中）。
        metadata: 其他元信息（hit_type/pos_error/ball_speed 等）。
    """

    q_desired: np.ndarray
    q_actual: np.ndarray
    timestamps: np.ndarray
    tcp_pos: np.ndarray
    ball_pos: np.ndarray
    init_q: np.ndarray
    init_q_left: np.ndarray
    dt: float
    hit_step: int
    metadata: dict = field(default_factory=dict)


@dataclass
class StepState:
    """单步执行状态（在 Sink 间共享的轻量数据载体）。

    Attributes:
        q_desired: (6,) 当前步目标角度。
        timestamp: 当前步时间戳。
        arm_state: (12,) [q(6), qdot(6)]，由 RobotSink 填充。
        tcp_pos: (3,) 末端位置，由 RobotSink 填充。
    """

    q_desired: np.ndarray
    timestamp: float
    arm_state: np.ndarray | None = None
    tcp_pos: np.ndarray | None = None
