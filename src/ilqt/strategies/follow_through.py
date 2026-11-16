"""随挥策略 — 击球后的随挥动作控制。

封装随挥触发条件（planned/contact/none）和控制计算（PD/IK）。
随挥状态（起始步、击球点末端位置、球位置）封装在策略实例内。

来源：V11 `_check_follow_through`（行 1418-1425, 2127-2143）+
`_compute_follow_through_control`（行 2149-2214）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class FollowContext:
    """随挥计算的每步输入上下文。

    Attributes:
        step_count: 当前 MPC 步索引。
        mpc_horizon: MPC 阶段总步数（超过即强制触发随挥）。
        k_hit: 剩余击球步数（planned 模式下 ≤1 触发）。
        arm_state: 臂状态 [q(n), qdot(n)] (2n,)。
        ball_pos: 球当前位置 (3,)（触发时记录为 ball_pos_at_hit）。
        d_follow: 随挥方向单位向量 (3,)。
        v_hit_desired: 期望击球时刻末端速度 (3,)。
        env: 规划环境（提供 FK/IK/Jacobian；RM65 PlanningEnv 或 R1D
            绑定环境族，鸭子类型：set_arm_state/get_ee_pos/get_ee_jacp/
            solve_ik/control_range）。
        robot_limits: 关节限位（可选；绑定环境位置模式随挥 IK 校验用，
            Task 3.8）。None 跳过限位校验（RM65 原行为）。
    """

    step_count: int
    mpc_horizon: int
    k_hit: int
    arm_state: NDArray[np.floating]
    ball_pos: NDArray[np.floating]
    d_follow: NDArray[np.floating]
    v_hit_desired: NDArray[np.floating]
    env: Any  # PlanningEnv（鸭子类型：set_arm_state/get_ee_pos/get_ee_jacp/solve_ik/model）
    robot_limits: Any = None  # 关节限位（R1D 随挥 IK 校验，Task 3.8）


@runtime_checkable
class FollowThroughPolicy(Protocol):
    """随挥策略接口 — 触发判定 + 控制计算 + 生命周期。"""

    @property
    def is_active(self) -> bool:
        """是否已触发随挥（随挥阶段进行中）。"""
        ...

    @property
    def follow_through_start(self) -> int:
        """随挥起始步索引（未触发时为 -1）。"""
        ...

    def should_trigger(self, ctx: FollowContext) -> bool:
        """判定本步是否触发随挥。

        Args:
            ctx: 随挥上下文。

        Returns:
            True 表示本步触发随挥（首次触发时记录末端/球位置）。
        """
        ...

    def compute_control(self, ctx: FollowContext, step_in_follow: int) -> NDArray[np.floating]:
        """计算随挥控制指令。

        Args:
            ctx: 随挥上下文。
            step_in_follow: 随挥内步数（0=触发当步）。

        Returns:
            控制指令 (6,)（力矩或 q_desired）。
        """
        ...

    def is_done(self, step_in_follow: int, ctx: FollowContext) -> bool:
        """判定随挥是否完成。

        Args:
            step_in_follow: 随挥内步数。
            ctx: 随挥上下文。

        Returns:
            True 表示随挥结束。
        """
        ...

    def reset(self) -> None:
        """重置随挥状态（清空起始步/记录位置）。"""
        ...


class PlannedFollowThrough:
    """计划触发随挥 — 与 V11 `_check_follow_through` + `_compute_follow_through_control` 一致。

    触发条件（任一满足）：
    - step_count >= mpc_horizon（MPC 阶段结束，强制触发）
    - follow_trigger == "planned" 且 k_hit <= 1（到达计划击打时刻）

    控制计算：
    - 位置模式：``env.solve_ik(p_des_follow)``
    - 力矩模式：``J_p.T @ (Kp*dp - Kd*v_ee)``
    """

    # PD 增益默认值（可通过构造函数 kp/kd 参数覆盖）

    def __init__(
        self,
        follow_through_steps: int,
        follow_trigger: str,
        dt: float,
        is_position_mode: bool,
        NQ: int,
        NU: int,
        kp: float = 200.0,
        kd: float = 20.0,
    ) -> None:
        """初始化随挥参数。

        Args:
            follow_through_steps: 随挥总步数（≤0 表示禁用）。
            follow_trigger: 触发模式 "planned" / "contact"。
            dt: 仿真步长（秒）。
            is_position_mode: 是否位置模式（True=IK，False=力矩 PD）。
            NQ: 关节数（位置向量维度）。
            NU: 控制维度（执行器数）。
            kp: 随挥 PD 比例增益（默认 200.0）。
            kd: 随挥 PD 微分增益（默认 20.0）。
        """
        self._follow_through_steps: int = follow_through_steps
        self._follow_trigger: str = follow_trigger
        self._dt: float = dt
        self._is_position_mode: bool = is_position_mode
        self._NQ: int = NQ
        self._NU: int = NU
        self._kp_follow: float = kp
        self._kd_follow: float = kd

        # ── 随挥状态（从 MPCController 移入）──
        self._follow_through_start: int = -1
        self._p_ee_at_hit: NDArray[np.floating] | None = None
        self._ball_pos_at_hit: NDArray[np.floating] | None = None

    @property
    def is_active(self) -> bool:
        """是否已触发随挥。"""
        return self._follow_through_start >= 0

    @property
    def follow_through_start(self) -> int:
        """随挥起始步索引（未触发时为 -1）。"""
        return self._follow_through_start

    def should_trigger(self, ctx: FollowContext) -> bool:
        """判定随挥触发（V11 行 1418-1425 + 2127-2143）。

        Args:
            ctx: 随挥上下文。

        Returns:
            True 表示本步触发（首次触发时记录末端/球位置）。
        """
        if self._follow_through_start >= 0:
            return False
        if self._follow_through_steps <= 0:
            return False

        triggered = False
        # 强制触发：MPC 阶段结束
        if ctx.step_count >= ctx.mpc_horizon:
            triggered = True
        # planned 模式：到达计划击打时刻
        elif self._follow_trigger == "planned" and ctx.k_hit <= 1:
            triggered = True

        if triggered:
            self._follow_through_start = ctx.step_count
            ctx.env.set_arm_state(ctx.arm_state)
            if self._p_ee_at_hit is None:
                self._p_ee_at_hit = ctx.env.get_ee_pos().copy()
            if self._ball_pos_at_hit is None:
                self._ball_pos_at_hit = ctx.ball_pos.copy()
            logger.info(
                "随挥触发: step=%d, steps=%d",
                ctx.step_count, self._follow_through_steps,
            )

        return triggered

    def compute_control(
        self, ctx: FollowContext, step_in_follow: int,
    ) -> NDArray[np.floating]:
        """计算随挥控制（V11 行 2149-2214）。

        沿 d_follow 方向匀减速直线延伸目标位置：
        ``p_des = p_ee_at_hit + d_follow * (v_max*t - 0.5*a*t^2)``

        Args:
            ctx: 随挥上下文。
            step_in_follow: 随挥内步数（0=触发当步）。

        Returns:
            控制指令 (6,)。
        """
        assert self._p_ee_at_hit is not None, "随挥未初始化 p_ee_at_hit"

        # 沿 d_follow 方向匀减速直线延伸（V11 行 2152-2158）
        v_max_follow = float(np.linalg.norm(ctx.v_hit_desired))
        T_follow = self._follow_through_steps * self._dt
        a_follow = v_max_follow / T_follow if T_follow > 0 else 0.0
        t_elapsed = step_in_follow * self._dt
        p_des_follow = self._p_ee_at_hit + ctx.d_follow * (
            v_max_follow * t_elapsed - 0.5 * a_follow * t_elapsed ** 2
        )

        ctx.env.set_arm_state(ctx.arm_state)
        if self._is_position_mode:
            u_follow = self._solve_follow_ik(ctx, p_des_follow)
        else:
            p_ee_cur = ctx.env.get_ee_pos()
            J_p = ctx.env.get_ee_jacp()
            dp = p_des_follow - p_ee_cur
            F_follow = self._kp_follow * dp - self._kd_follow * (
                J_p @ ctx.arm_state[self._NQ:]
            )
            u_follow = J_p.T @ F_follow

        # Task 3.8: 控制界经 env.control_range（RobotEnv 协议，按执行器
        # 绑定序；RM65 前缀序与旧 actuator_ctrlrange[:NU] 逐值一致）
        ctrl_range = np.asarray(ctx.env.control_range, dtype=np.float64)
        return np.clip(u_follow, ctrl_range[:, 0], ctrl_range[:, 1])

    def _solve_follow_ik(
        self, ctx: FollowContext, p_des_follow: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """位置模式随挥 IK（绑定环境带误差/限位校验，失败回退 hold）。

        RM65（无 binding 属性）保持原行为：裸 solve_ik，不校验。
        绑定环境（R1D，Task 3.8）：IK 结果必须检查误差和限位——失败
        告警并回退 hold（当前角度），不带坏指令继续。

        Args:
            ctx: 随挥上下文。
            p_des_follow: 随挥目标位置 (3,)。

        Returns:
            位置指令 (NU,)；IK 失败时为当前角度（hold 回退）。
        """
        if not hasattr(ctx.env, "binding"):
            # RM65 原路径
            return ctx.env.solve_ik(
                p_des_follow, q_init=ctx.arm_state[: self._NQ],
                max_iter=20, eps=1e-2,
            )
        from src.ilqt.jt_init import solve_ik_checked

        try:
            return solve_ik_checked(
                ctx.env, p_des_follow,
                q_init=ctx.arm_state[: self._NQ],
                robot_limits=ctx.robot_limits,
                max_iter=20, eps=1e-2, pos_tol=0.05,
                label="随挥 ",
            )
        except ValueError as exc:
            logger.warning(
                "随挥 IK 失败，回退 hold（保持当前角度）: %s", exc,
            )
            return ctx.arm_state[: self._NQ].copy()

    def is_done(self, step_in_follow: int, ctx: FollowContext) -> bool:
        """判定随挥是否完成。

        Args:
            step_in_follow: 随挥内步数。
            ctx: 随挥上下文（保留接口一致性）。

        Returns:
            step_in_follow > follow_through_steps 时 True。
        """
        return step_in_follow > self._follow_through_steps

    def reset(self) -> None:
        """重置随挥状态。"""
        self._follow_through_start = -1
        self._p_ee_at_hit = None
        self._ball_pos_at_hit = None
