"""管线组件接口协议 — 仿真/真机组件的共同接口。

所有组件通过 Protocol 解耦，EpisodeRunner 依赖 Protocol 不依赖具体类。
组合优于继承：不同实现可以自由替换。
"""

from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
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
        """读取当前臂状态 [q(n), qdot(n)]，弧度，形状 (2n,)。

        n 为臂关节数（RM65=6 → (12,)；R1D=7 → (14,)），维度由
        注入的环境决定，组件不得假设固定宽度。
        """
        ...

    def execute(self, u_cmd: NDArray[np.floating]) -> None:
        """执行控制指令（力矩或 q_desired），形状 (n,)。"""
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


# ── 仿真执行环境协议（Task 3.9：SimComponent 按协议而非 RM65 类型消费）──


@runtime_checkable
class SimExecutionEnv(Protocol):
    """仿真执行环境接口 — SimComponent 消费的最小成员集合。

    RM65Env（旧 RM65 路径）与 ProfileSimEnv（R1D 路径，阶段 3 · 3.4）
    共同满足本协议；SimComponent 依赖本协议而非具体环境类型，
    执行/诊断逻辑对臂维数 n（6 或 7）通用。

    成员语义：
      - ``NQ``/``NU``：主动臂关节数/执行器数（实例属性或 property）；
      - ``dt``：物理步长（秒）；
      - ``model``/``data``：MuJoCo 模型与 live 数据（接触判定读
        ``data.ncon``/``data.contact``，geom 查询走 ``model``）；
      - ``control_range``：(n,2) 执行器控制范围副本（X 墙推回/
        advance_ball 的力矩裁剪用它，不假设执行器占据地址前缀）；
      - ``step_full(u)``：前进一步返回 (arm_state, ball_pos, ball_vel)；
      - ``set_arm_collision(enabled)``：碰撞窗口（RM65=臂 geom 掩码；
        ProfileSimEnv=仅球—拍 pair_margin 通道，机身保护常开）；
      - 球读写：``get_ball_pos``/``get_ball_vel``/``set_ball_vel``；
      - 末端查询：``get_ee_pos``/``get_ee_vel``/``get_ee_normal``/
        ``get_racket_face_speed``（主动命中拍速阈值判定）。
    """

    @property
    def NQ(self) -> int:
        """主动臂关节数 n。"""
        ...

    @property
    def NU(self) -> int:
        """执行器数 n。"""
        ...

    @property
    def dt(self) -> float:
        """物理步长（秒）。"""
        ...

    @property
    def model(self) -> Any:
        """MuJoCo MjModel（geom 查询与接触判定用）。"""
        ...

    @property
    def data(self) -> Any:
        """MuJoCo MjData（live 状态与接触数组）。"""
        ...

    @property
    def control_range(self) -> NDArray[np.floating]:
        """(n,2) 执行器控制范围副本。"""
        ...

    def get_arm_state(self) -> NDArray[np.floating]:
        """读取臂状态 (2n,)。"""
        ...

    def update_kinematics(self) -> None:
        """轻量运动学刷新。"""
        ...

    def step_full(
        self, u: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """前进一步 → (arm_state, ball_pos, ball_vel)。"""
        ...

    def set_arm_collision(self, enabled: bool) -> None:
        """碰撞窗口开关（语义见类 docstring）。"""
        ...

    def get_ball_pos(self) -> NDArray[np.floating]:
        """物理球位置 (3,)。"""
        ...

    def get_ball_vel(self) -> NDArray[np.floating]:
        """物理球线速度 (3,)。"""
        ...

    def set_ball_vel(self, vel: NDArray[np.floating]) -> None:
        """写入物理球速度（反弹）。"""
        ...

    def get_ee_pos(self) -> NDArray[np.floating]:
        """末端（拍面中心）位置 (3,)。"""
        ...

    def get_ee_vel(self) -> NDArray[np.floating]:
        """末端线速度 (3,)。"""
        ...

    def get_ee_normal(self) -> NDArray[np.floating]:
        """拍面法向 (3,)。"""
        ...

    def get_racket_face_speed(self) -> float:
        """拍面中心线速度标量 [m/s]。"""
        ...


# ── 代价模块协议 ──
# 拆分 Running + Terminal 匹配最优控制理论 J = Σ l_k(x,u) + l_N(x)。
# 13 个惩罚项无一跨界（全部纯运行或纯终端），证明拆分是自然的。


@runtime_checkable
class RunningCost(Protocol):
    """运行代价模块接口 — solver 调用此接口（CompositeCost 实现）。

    注意：独立惩罚项（ControlEffortTerm 等）的签名多一个 fk 参数
    （running_cost(self, x, u, k, fk)），不直接满足此 Protocol。
    CompositeCost 适配两层接口：solver 调 3-arg，内部委托 4-arg 给各惩罚项。

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
    """终端代价模块接口 — solver 调用此接口（CompositeCost 实现）。

    注意：独立惩罚项签名多一个 fk 参数（terminal_cost(self, x, fk)），
    由 CompositeCost 适配。此 Protocol 描述 solver-facing 接口。

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
class SmoothnessScaleUpdatable(Protocol):
    """平滑度缩放动态更新接口（SmoothnessTerm 实现）。

    活跃调用点：replan_core.py:336 生产路径（分阶段平滑度调度）。
    注意：TubeHittingCostWrapper 未委托此方法，Tube 模式下 hasattr 跳过（已有行为）。
    """

    def set_smoothness_scale(
        self, qdot_scale: float, qddot_scale: float, du_scale: float,
    ) -> None: ...
