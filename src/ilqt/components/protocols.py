"""管线组件接口协议 — 仿真/真机组件的共同接口。

所有组件通过 Protocol 解耦，EpisodeRunner 依赖 Protocol 不依赖具体类。
组合优于继承：不同实现可以自由替换。
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
from numpy.typing import NDArray
import numpy as np


@runtime_checkable
class PerceptionComponent(Protocol):
    """感知组件接口 — 提供球状态。"""

    def get_ball_state(self) -> tuple[NDArray[np.floating], NDArray[np.floating]] | None:
        """获取最新球状态。

        Returns:
            (pos(3,), vel(3,)) 或 None（无数据时）。
        """
        ...


@runtime_checkable
class ExecutorComponent(Protocol):
    """执行组件接口 — 执行控制指令、提供臂状态和指标。"""

    def get_arm_state(self) -> NDArray[np.floating]:
        """读取当前臂状态 [q(6), qdot(6)]，弧度，形状 (12,)。"""
        ...

    def execute(self, u_cmd: NDArray[np.floating]) -> None:
        """执行控制指令（力矩或 q_desired）。"""
        ...

    def get_metrics(self) -> dict:
        """返回汇总指标（碰撞检测、history、安全统计等）。"""
        ...


@runtime_checkable
class SafetyComponent(Protocol):
    """安全组件接口 — 安全滤波。

    实现可以:
    - 仅做基础检查（无预测）：BasicSafetyFilter
    - 做预测性检查（需 RobotEnv）：PredictiveSafetyFilter
    """

    def filter(
        self, u_cmd: NDArray[np.floating], arm_state: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], bool]:
        """安全滤波 → (safe_u, is_safe)。"""
        ...
