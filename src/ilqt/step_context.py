"""管线步骤上下文 — hook 回调的数据载体。

每个 hook 接收一个 StepContext 实例，可读取当前步的状态数据，
也可向 metrics 字典追加自定义指标。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class StepContext:
    """每步上下文 — hook 可读写。

    字段在管线各阶段逐步填充：
        初始: step_count, arm_state, ball_pos, ball_vel
        规划后: mpc_result
        安全后: u_cmd, is_safe
    """

    step_count: int
    arm_state: np.ndarray
    ball_pos: np.ndarray | None = None
    ball_vel: np.ndarray | None = None
    mpc_result: Any = None
    u_cmd: np.ndarray | None = None
    is_safe: bool = True
    metrics: dict = field(default_factory=dict)
