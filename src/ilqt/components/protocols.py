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


# ── 代价模块协议 ──
# 拆分 Running + Terminal 匹配最优控制理论 J = Σ l_k(x,u) + l_N(x)。
# 13 个惩罚项无一跨界（全部纯运行或纯终端），证明拆分是自然的。


@runtime_checkable
class RunningCost(Protocol):
    """运行代价模块接口 — 每步惩罚项实现此接口。

    数学含义：运行代价 l_k(x, u)，依赖状态和控制。
    导数返回 5 元组 (l_x, l_u, l_xx, l_ux, l_uu)。
    """

    def running_cost(
        self, x: NDArray[np.floating], u: NDArray[np.floating], k: int | None = None
    ) -> float:
        """计算运行代价 l(x, u, k)。

        Args:
            x: 臂状态 [q(6), qdot(6)]，形状 (12,)。
            u: 控制量，形状 (6,)。
            k: 当前时间步索引。None 表示无时间依赖。

        Returns:
            运行代价值。
        """
        ...

    def running_derivatives(
        self, x: NDArray[np.floating], u: NDArray[np.floating], k: int | None = None
    ) -> tuple[NDArray[np.floating], ...]:
        """计算运行代价导数 (l_x, l_u, l_xx, l_ux, l_uu)。

        Returns:
            5 元组，形状分别为 (NX,), (NU,), (NX,NX), (NU,NX), (NU,NU)。
        """
        ...


@runtime_checkable
class TerminalCost(Protocol):
    """终端代价模块接口 — 最终步惩罚项实现此接口。

    数学含义：终端代价 l_N(x)，仅依赖状态（终端步无控制量）。
    导数返回 2 元组 (l_x, l_xx)。
    """

    def terminal_cost(self, x: NDArray[np.floating]) -> float:
        """计算终端代价 l_N(x)。

        Args:
            x: 臂状态，形状 (12,)。

        Returns:
            终端代价值。
        """
        ...

    def terminal_derivatives(
        self, x: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """计算终端代价导数 (l_x, l_xx)。

        Returns:
            2 元组，形状 (NX,), (NX,NX)。
        """
        ...


@runtime_checkable
class SmoothnessMixin(Protocol):
    """需要上一帧控制量的惩罚项接口。

    仅 SmoothnessTerm（Q_du 控制变化率项）需要此接口。
    solver 用 isinstance(cost, SmoothnessMixin) 替代 hasattr 守卫。
    """

    def set_u_prev(self, u_prev: NDArray[np.floating]) -> None:
        """设置上一帧控制量 u_{k-1}，用于计算 Δu = u_k - u_{k-1}。"""
        ...


# ── MPC 动态更新协议（CompositeCost 委托用）──


@runtime_checkable
class TargetUpdatable(Protocol):
    """终端目标可更新的代价项（TerminalHitTerm 实现）。"""

    def update_target(
        self,
        p_hit: NDArray[np.floating],
        v_hit: NDArray[np.floating],
        n_des: NDArray[np.floating] | None = None,
    ) -> None: ...


@runtime_checkable
class WeightUpdatable(Protocol):
    """权重可缩放的代价项（TerminalHitTerm 实现，MPC far/near 阶段调度）。"""

    def update_weights(self, Q_p_scale: float = 1.0, Q_v_scale: float = 1.0) -> None: ...


@runtime_checkable
class RScheduleUpdatable(Protocol):
    """R 调度可更新的代价项（ControlEffortTerm 实现，R 退火）。"""

    def set_R_schedule(self, R_schedule: NDArray[np.floating] | None) -> None: ...


# ── 活跃代码路径动态更新协议（R3 修复追加）──


@runtime_checkable
class JointTrackUpdatable(Protocol):
    """关节轨迹跟踪动态更新接口（JointTrackingTerm 实现）。

    活跃调用点：8+ 活跃脚本 MPC 循环（后摆关节跟踪）。
    """

    def set_q_des_traj(
        self,
        q_des_traj: NDArray[np.floating] | None,
        Q_joint: dict[int, float] | None = None,
    ) -> None: ...


@runtime_checkable
class SmoothnessScaleUpdatable(Protocol):
    """平滑度缩放动态更新接口（SmoothnessTerm 实现）。

    活跃调用点：replan_core.py:336 生产路径（分阶段平滑度调度）。
    注意：TubeHittingCostWrapper 未委托此方法，Tube 模式下 hasattr 跳过（已有行为）。
    """

    def set_smoothness_scale(
        self, qdot_scale: float, qddot_scale: float, du_scale: float,
    ) -> None: ...


@runtime_checkable
class MidpointUpdatable(Protocol):
    """中途目标动态更新接口（MidpointTerm 实现）。

    活跃调用点：run_20hits_video.py:633（Tube 包装前在 base_cost_fn 上调用）。
    """

    def set_midpoint_target(
        self,
        step: int | None,
        target: NDArray[np.floating] | None,
        Q_midpoint: NDArray[np.floating] | None = None,
        v_target: NDArray[np.floating] | None = None,
        Q_midpoint_v: NDArray[np.floating] | None = None,
    ) -> None: ...
