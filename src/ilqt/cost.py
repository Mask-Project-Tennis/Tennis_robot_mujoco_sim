"""iLQT 代价函数：终端击打点代价 + 拍面法向量代价 + 运行代价 + 身体硬约束。

Phase 1 (C1): FKContext + RunningDerivatives + TerminalDerivatives 基础设施。
Phase 2 (C2): CompositeCost + 惩罚项组合化（替换旧 HittingCost）。
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
from src.ilqt.components.protocols import (
    SmoothnessMixin,
    TargetUpdatable,
    WeightUpdatable,
    RScheduleUpdatable,
    SmoothnessScaleUpdatable,
)


class RunningDerivatives:
    """惩罚项运行导数贡献（Flyweight 模式）。

    数组在惩罚项构造时预分配，每步 np.copyto 原地更新。
    返回持久引用——零分配/步。

    None 语义：此项数学上不依赖该变量（非缺失实现）。
    例如 ControlEffortTerm.l_x = None，因为 R·u² 不依赖状态 x。
    """

    __slots__ = ("l_x", "l_u", "l_xx", "l_ux", "l_uu")

    def __init__(
        self,
        l_x: np.ndarray | None = None,
        l_u: np.ndarray | None = None,
        l_xx: np.ndarray | None = None,
        l_ux: np.ndarray | None = None,
        l_uu: np.ndarray | None = None,
    ) -> None:
        """初始化导数容器。

        Args:
            l_x: 状态梯度 (NX,)。None=不依赖状态。
            l_u: 控制梯度 (NU,)。None=不依赖控制。
            l_xx: 状态 Hessian (NX,NX)。None=不依赖状态。
            l_ux: 控制-状态交叉 Hessian (NU,NX)。None=不贡献。
            l_uu: 控制 Hessian (NU,NU)。None=不依赖控制。
        """
        self.l_x = l_x
        self.l_u = l_u
        self.l_xx = l_xx
        self.l_ux = l_ux
        self.l_uu = l_uu


class TerminalDerivatives:
    """惩罚项终端导数贡献（Flyweight 模式）。

    同 RunningDerivatives，但终端代价仅依赖状态（无控制）。
    """

    __slots__ = ("l_x", "l_xx")

    def __init__(
        self,
        l_x: np.ndarray | None = None,
        l_xx: np.ndarray | None = None,
    ) -> None:
        """初始化终端导数容器。

        Args:
            l_x: 状态梯度 (NX,)。
            l_xx: 状态 Hessian (NX,NX)。
        """
        self.l_x = l_x
        self.l_xx = l_xx


class FKContext:
    """末端 FK 缓存上下文（维度无关：n=6/7 通吃，缓冲随 env.NQ/NX）。

    缓存一次 env.set_arm_state(x) 的全部末端 FK 结果（位置/速度/雅可比/法向量），
    供所有惩罚项共享读取。惩罚项不直接接触 env，通过此对象获取 FK 量，
    便于单元测试（MockFKContext 纯数据，不依赖 MuJoCo）。

    缓存失效语义（RM65 既有约定，保持不变）：缓存以传入状态 x 为键——
    相同状态重复调用 update 时跳过（避免冗余 mj_forward）；环境状态
    变化后传入新 x 即刷新。update 之外直接修改 env 状态不会使缓存
    失效，消费方应始终以最新 x 调用 update。
    """

    def __init__(self, env) -> None:
        """初始化 FK 上下文。

        Args:
            env: 满足 RobotEnv Protocol 的环境实例（NQ/NX 为实例属性或
                类属性，维数任意，如 RM65Env n=6 / BoundRobotEnv n=7）。
        """
        self._env = env
        self._x_cached: np.ndarray | None = None
        # 初始缓冲区（update 时由 env 返回值替换）
        # 优化点：当前 env.get_ee_*() 内部 .copy() 导致每步分配 8 个小数组（~816B）。
        # 可通过添加 get_ee_*_into(buf) API 实现零拷贝，但 profiling 显示分配仅占
        # 总 solve 的 0.2-0.5%（mj_forward 和 iLQR 迭代是瓶颈），当前不值得优化。
        self._p_ee: np.ndarray = np.zeros(3)
        self._v_ee: np.ndarray = np.zeros(3)
        self._J_p: np.ndarray = np.zeros((3, env.NQ))
        self._J_r: np.ndarray = np.zeros((3, env.NQ))
        self._n_rack: np.ndarray = np.zeros(3)
        self._J_n: np.ndarray = np.zeros((3, env.NX))
        self._skew: np.ndarray = np.zeros((3, 3))

    def update(self, x: np.ndarray) -> None:
        """设置机器人状态并缓存全部末端 FK 结果。

        相同状态跳过（避免冗余 mj_forward 调用）；新状态刷新全部缓存。

        Args:
            x: 臂状态 [q(NQ), qdot(NQ)]，形状 (NX,)。
        """
        if self._x_cached is not None and np.array_equal(x, self._x_cached):
            return
        self._env.set_arm_state(x)
        self._p_ee = self._env.get_ee_pos()
        self._v_ee = self._env.get_ee_vel()
        self._J_p = self._env.get_ee_jacp()
        self._J_r = self._env.get_ee_jacr()
        self._n_rack = self._env.get_ee_normal()
        # 法向量雅可比：J_n = skew(-n_rack) @ J_r，写入 (3, NX) 前 NQ 列
        nx_, ny_, nz_ = -self._n_rack[0], -self._n_rack[1], -self._n_rack[2]
        self._skew[0, 0] = 0.0;  self._skew[0, 1] = -nz_; self._skew[0, 2] = ny_
        self._skew[1, 0] = nz_;  self._skew[1, 1] = 0.0;  self._skew[1, 2] = -nx_
        self._skew[2, 0] = -ny_; self._skew[2, 1] = nx_;  self._skew[2, 2] = 0.0
        self._J_n.fill(0)
        self._J_n[:, : self._env.NQ] = self._skew @ self._J_r
        if self._x_cached is None:
            self._x_cached = x.copy()
        else:
            np.copyto(self._x_cached, x)

    @property
    def p_ee(self) -> np.ndarray:
        """末端位置 (3,)。"""
        return self._p_ee

    @property
    def v_ee(self) -> np.ndarray:
        """末端速度 (3,)。"""
        return self._v_ee

    @property
    def J_p(self) -> np.ndarray:
        """位置雅可比 (3, NQ)。"""
        return self._J_p

    @property
    def J_r(self) -> np.ndarray:
        """旋转雅可比 (3, NQ)。"""
        return self._J_r

    @property
    def n_rack(self) -> np.ndarray:
        """拍面法向量 (3,)，单位向量。"""
        return self._n_rack

    @property
    def J_n(self) -> np.ndarray:
        """法向量雅可比 (3, NX)。前 NQ 列 = skew(-n)@J_r，后 NQ 列为零。"""
        return self._J_n


class CompositeCost:
    """组合代价聚合器 — 统一管理多个惩罚项。

    实现 RunningCost + TerminalCost Protocol。
    核心职责：
    1. FKContext 统一管理（每步只调一次 set_arm_state，避免冗余 mj_forward）
    2. 预分配累加器（Flyweight，零分配/步）
    3. MPC 动态更新委托（update_target / update_weights / set_u_prev /
       set_R_schedule / set_q_des_traj / set_smoothness_scale / set_midpoint_target）

    组合优于继承：惩罚项通过列表组合，新增项只需追加到列表，无需修改本类。
    各惩罚项实现 RunningCost 或 TerminalCost Protocol（或两者），通过本类聚合。

    导数累加规则：惩罚项的 running_derivatives / terminal_derivatives 可能返回
    None（该项数学上不依赖该变量，如 ControlEffortTerm.l_x = None）。本类对每个
    字段单独做 None 守卫，跳过未贡献的项。

    R1 修复：通过 @property 透传终端项的 p_hit / v_hit / Q_p / Q_v / Q_n / n_des，
    TubeHittingCostWrapper 的 softmin 路径（tube_cost.py:195-204, 225-227）直接
    读取这些属性。
    """

    needs_fk: bool  # 由 running_terms 的 needs_fk 决定

    def __init__(self, env, running_terms: list, terminal_terms: list) -> None:
        """初始化组合代价。

        Args:
            env: 满足 RobotEnv Protocol 的环境实例（提供 NQ/NX/NU 维度与 FK 接口）。
            running_terms: 运行代价项列表（实现 RunningCost）。
            terminal_terms: 终端代价项列表（实现 TerminalCost）。
        """
        self._env = env
        self._logger = logging.getLogger(__name__)
        self.fk = FKContext(env)
        self.running_terms = running_terms
        self.terminal_terms = terminal_terms
        self._NX = env.NX
        self._NU = env.NU
        self._NQ = env.NQ
        # 判断是否需要 FK（任意运行项或终端项 needs_fk=True 即启用）
        self.needs_fk = any(getattr(t, "needs_fk", False) for t in running_terms)
        self._terminal_needs_fk = any(
            getattr(t, "needs_fk", False) for t in terminal_terms
        )
        # 预分配累加器（Flyweight：每步 fill(0) 原地清零，零分配）
        self._acc_l_x = np.zeros(env.NX)
        self._acc_l_u = np.zeros(env.NU)
        self._acc_l_xx = np.zeros((env.NX, env.NX))
        self._acc_l_ux = np.zeros((env.NU, env.NX))
        self._acc_l_uu = np.zeros((env.NU, env.NU))
        self._term_l_x = np.zeros(env.NX)
        self._term_l_xx = np.zeros((env.NX, env.NX))
        # R1 修复：缓存首个含 p_hit 属性的终端项引用
        # （Tube wrapper softmin 路径读 base_cost.Q_p / .Q_v / .Q_n / .p_hit / .v_hit / .n_des）
        self._terminal_hit = next(
            (t for t in terminal_terms if hasattr(t, "p_hit")), None
        )

    # ── R1 修复：TubeHittingCostWrapper softmin 兼容属性 ──
    # Tube wrapper softmin 路径直接读 base_cost.Q_p / .Q_v / .Q_n / .p_hit / .v_hit / .n_des
    # （tube_cost.py:195-204, 225-227），CompositeCost 通过 @property 透传到 TerminalHitTerm。

    @property
    def p_hit(self) -> np.ndarray:
        """终端击打位置（Tube wrapper softmin 路径读取）。"""
        if self._terminal_hit is None:
            raise AttributeError("CompositeCost 无含 p_hit 的终端代价项")
        return self._terminal_hit.p_hit

    @property
    def v_hit(self) -> np.ndarray:
        """终端击打速度（Tube wrapper softmin 路径读取）。"""
        if self._terminal_hit is None:
            raise AttributeError("CompositeCost 无含 v_hit 的终端代价项")
        return self._terminal_hit.v_hit

    @property
    def Q_p(self) -> np.ndarray:
        """位置权重矩阵（Tube wrapper softmin 路径读取）。"""
        if self._terminal_hit is None:
            raise AttributeError("CompositeCost 无含 Q_p 的终端代价项")
        return self._terminal_hit.Q_p

    @property
    def Q_v(self) -> np.ndarray:
        """速度权重矩阵（Tube wrapper softmin 路径读取）。"""
        if self._terminal_hit is None:
            raise AttributeError("CompositeCost 无含 Q_v 的终端代价项")
        return self._terminal_hit.Q_v

    @property
    def Q_n(self) -> float:
        """法向量权重（无终端项时返回 0.0）。"""
        return self._terminal_hit.Q_n if self._terminal_hit is not None else 0.0

    @property
    def n_des(self) -> np.ndarray | None:
        """期望法向量（无终端项时返回 None）。"""
        return self._terminal_hit.n_des if self._terminal_hit is not None else None

    def running_cost(self, x, u, k=None) -> float:
        """计算运行代价 = 各运行项之和。

        Args:
            x: 臂状态 [q(NQ), qdot(NQ)]，形状 (NX,)。
            u: 控制量，形状 (NU,)。
            k: 当前时间步索引。None 表示无时间依赖。

        Returns:
            运行代价值（标量）。
        """
        if self.needs_fk:
            self.fk.update(x)
        return sum(t.running_cost(x, u, k, self.fk) for t in self.running_terms)

    def terminal_cost(self, x) -> float:
        """计算终端代价 = 各终端项之和。

        Args:
            x: 臂状态，形状 (NX,)。

        Returns:
            终端代价值（标量）。
        """
        if self._terminal_needs_fk:
            self.fk.update(x)
        return sum(t.terminal_cost(x, self.fk) for t in self.terminal_terms)

    def running_derivatives(self, x, u, k=None):
        """计算运行导数 = 各运行项之和（预分配累加器，零分配/步）。

        对每个项返回的 RunningDerivatives 各字段做 None 守卫：该项数学上不依赖
        某变量时该字段为 None（如 ControlEffortTerm.l_x=None），累加时跳过。

        Args:
            x: 臂状态 [q(NQ), qdot(NQ)]，形状 (NX,)。
            u: 控制量，形状 (NU,)。
            k: 当前时间步索引。None 表示无时间依赖。

        Returns:
            5 元组 (l_x, l_u, l_xx, l_ux, l_uu)，形状分别为
            (NX,), (NU,), (NX,NX), (NU,NX), (NU,NU)。
            l_ux 恒为零（当前无项贡献交叉项）。
        """
        if self.needs_fk:
            self.fk.update(x)
        self._acc_l_x.fill(0)
        self._acc_l_u.fill(0)
        self._acc_l_xx.fill(0)
        self._acc_l_ux.fill(0)
        self._acc_l_uu.fill(0)
        for t in self.running_terms:
            d = t.running_derivatives(x, u, k, self.fk)
            if d.l_x is not None:
                self._acc_l_x += d.l_x
            if d.l_u is not None:
                self._acc_l_u += d.l_u
            if d.l_xx is not None:
                self._acc_l_xx += d.l_xx
            if d.l_ux is not None:
                self._acc_l_ux += d.l_ux
            if d.l_uu is not None:
                self._acc_l_uu += d.l_uu
        return (
            self._acc_l_x,
            self._acc_l_u,
            self._acc_l_xx,
            self._acc_l_ux,
            self._acc_l_uu,
        )

    def terminal_derivatives(self, x):
        """计算终端导数 = 各终端项之和（预分配累加器，零分配/步）。

        Args:
            x: 臂状态，形状 (NX,)。

        Returns:
            2 元组 (l_x, l_xx)，形状分别为 (NX,), (NX,NX)。
        """
        if self._terminal_needs_fk:
            self.fk.update(x)
        self._term_l_x.fill(0)
        self._term_l_xx.fill(0)
        for t in self.terminal_terms:
            d = t.terminal_derivatives(x, self.fk)
            if d.l_x is not None:
                self._term_l_x += d.l_x
            if d.l_xx is not None:
                self._term_l_xx += d.l_xx
        return self._term_l_x, self._term_l_xx

    # ── MPC 动态更新委托 ──
    # 通过 isinstance(t, XxxProtocol) 守卫：仅委托给实现了对应 Protocol 的项，
    # 无对应项时静默跳过（循环不迭代或 isinstance 返回 False）。

    def update_target(self, p_hit, v_hit, n_des=None):
        """委托终端目标更新（TerminalHitTerm 实现 TargetUpdatable）。

        Args:
            p_hit: 新的期望击打位置 (3,)。
            v_hit: 新的期望击打速度 (3,)。
            n_des: 新的期望拍面法向量 (3,)。None 表示保持不变。
        """
        for t in self.terminal_terms:
            if isinstance(t, TargetUpdatable):
                t.update_target(p_hit, v_hit, n_des)

    def update_weights(self, Q_p_scale=1.0, Q_v_scale=1.0):
        """委托权重缩放（TerminalHitTerm 实现 WeightUpdatable，MPC far/near 调度）。

        Args:
            Q_p_scale: 位置权重缩放因子。
            Q_v_scale: 速度权重缩放因子。
        """
        for t in self.terminal_terms:
            if isinstance(t, WeightUpdatable):
                t.update_weights(Q_p_scale, Q_v_scale)

    def set_u_prev(self, u_prev):
        """委托上一帧控制量设置（SmoothnessTerm 实现 SmoothnessMixin）。

        Args:
            u_prev: 上一帧控制量 u_{k-1}，形状 (NU,)。
        """
        for t in self.running_terms:
            if isinstance(t, SmoothnessMixin):
                t.set_u_prev(u_prev)

    def set_R_schedule(self, R_schedule):
        """委托 R 调度更新（ControlEffortTerm 实现 RScheduleUpdatable，R 退火）。

        Args:
            R_schedule: 时变 R 调度数组 (N,)，或 None 清除调度。
        """
        for t in self.running_terms:
            if isinstance(t, RScheduleUpdatable):
                t.set_R_schedule(R_schedule)

    # ── R3b 修复：活跃代码路径调用的额外方法（含 no-op 兼容存根） ──

    def set_q_des_traj(self, q_des_traj, Q_joint=None):
        """显式 no-op（G5 修复：JointTrackUpdatable Protocol 已删除）。

        历史上有 JointTrackingTerm 通过 JointTrackUpdatable Protocol 接收
        期望关节轨迹，当前无任何具体实现类。保留此方法为 no-op 以兼容
        8+ 活跃脚本的调用（rm65_mpc_ilqr_5_5.py / rm65_evaluate.py /
        run_20hits_video.py 等 MPC 循环中调用 set_q_des_traj 后摆关节跟踪）。

        Args:
            q_des_traj: 期望关节轨迹（忽略）。
            Q_joint: 关节级权重（忽略）。
        """
        return

    def set_smoothness_scale(self, qdot_scale, qddot_scale, du_scale):
        """委托平滑度缩放更新（SmoothnessTerm 实现 SmoothnessScaleUpdatable）。

        活跃调用点：replan_core.py 分阶段平滑度调度。
        注意：Tube wrapper 未委托此方法，Tube 模式下 hasattr 跳过（已有行为，保持一致）。

        Args:
            qdot_scale: Q_qdot 缩放因子。
            qddot_scale: Q_qddot 缩放因子。
            du_scale: Q_du 缩放因子。
        """
        for t in self.running_terms:
            if isinstance(t, SmoothnessScaleUpdatable):
                t.set_smoothness_scale(qdot_scale, qddot_scale, du_scale)

    def set_midpoint_target(self, step, target, **kwargs):
        """显式 no-op（G5 修复：MidpointUpdatable Protocol 已删除）。

        历史上有 MidpointTerm 通过 MidpointUpdatable Protocol 接收中途
        目标，当前无任何具体实现类。保留此方法为 no-op 以兼容
        run_20hits_video.py 等脚本在 Tube 包装前的 base_cost_fn 上调用
        set_midpoint_target(None, None) 清除目标。

        Args:
            step: 目标生效的时间步索引（忽略）。
            target: 期望中途目标位置（忽略）。
            **kwargs: 额外参数（忽略）。
        """
        return


def build_production_cost(
    env,
    config,
    robot_limits,
    p_hit: np.ndarray,
    v_hit: np.ndarray,
    n_des: np.ndarray | None,
    Q_p_mat: np.ndarray | None,
    Q_v_mat: np.ndarray | None,
    *,
    cost_config: Mapping[str, Any] | None = None,
) -> CompositeCost:
    """从 MPCConfig 构建生产路径代价（replan_core.py 调用，维度无关）。

    维度全部显式取自 env（NQ/NX/NU），不依赖各 cost term 的 RM65 缺省 6。
    R2 修复：Q_p_mat/Q_v_mat 由调用方预计算（replan_core.py:256-269 基于
    sigmoid 位置误差调度），工厂函数不负责权重调度计算。
    R_schedule 同样由调用方在工厂返回后设置（replan_core.py:286-291 不动）。

    机器人专属权重（阶段 3 Task 3.6）：``cost_config`` 接收 task YAML 的
    cost_config 字典（``SimTaskConfig.cost_config``，3.1 交付），键名仅认
    既有口径、不发明新键：

    - 工厂消费的 MPCConfig 字段名：``R``、``Q_qdot_base``、``Q_qddot_base``、
      ``Q_du_base``、``Q_tcp_soft``、``Q_qdot_limit``、``normal_weight``、
      ``tcp_soft_ratio``、``qdot_limit_ratio``——出现即覆盖 ``config``
      同名字段（None 缺省完全不覆盖，RM65 数值路径原样）；
    - 终端基准权重 ``Q_p`` / ``Q_v``（3.1 YAML 已有键）：标量 → ×I₃、
      (3,) → 对角、(3,3) → 原样；提供后可省 Q_p_mat/Q_v_mat（二者
      全缺时 ValueError，不静默零权重）。注意不收 ``Q_p_base``/``Q_v_base``
      （那是 replan_core 的 sigmoid 调度口径，避免歧义）；
    - ``tube`` 子字典：仅校验键名须为 TubeConfig 字段（拼写错误快速
      失败），本工厂不消费——Tube 包装由 MPC 层按 ``config.tube_cfg`` 组装。

    Args:
        env: 规划环境（PlanningEnv / RM65Env / BoundRobotEnv），满足
            RobotEnv Protocol，维数任意（RM65 n=6 / R1D n=7）。
        config: MPCConfig 实例（包含 R/Q_qdot/Q_tcp_soft 等参数）。
        robot_limits: RobotLimits 实例（包含 max_tcp_speed/qdot_max）。
        p_hit: 终端击打位置 (3,)。
        v_hit: 终端击打速度 (3,)。
        n_des: 期望拍面法向量 (3,) 或 None。
        Q_p_mat: 预计算的位置权重矩阵 (3,3)，由调用方从 config + sigmoid
            调度生成。None 时必须由 cost_config["Q_p"] 提供。
        Q_v_mat: 预计算的速度权重矩阵 (3,3)，同上。
        cost_config: task YAML 代价权重覆盖（keyword-only）。None=缺省，
            行为与旧版完全一致（RM65 路径不变）。

    Returns:
        CompositeCost 实例，组装了生产路径全部活跃惩罚项。

    Raises:
        ValueError: cost_config 含未知键、tube 子字典含非 TubeConfig 字段，
            或 Q_p/Q_v 两个来源（矩阵参数与 cost_config）均未提供。
    """
    from src.ilqt.cost_terms import (
        ControlEffortTerm,
        SmoothnessTerm,
        TcpSoftTerm,
        QdotLimitTerm,
        TerminalHitTerm,
    )

    overrides = _validate_task_cost_config(cost_config)

    NU = env.NU
    NQ = env.NQ
    NX = env.NX
    actuator_mode = 1 if config.is_position_mode else 0

    # 终端基准权重：cost_config 覆盖 > 显式矩阵 > 拒绝（不静默零权重）
    Q_p_final = _resolve_terminal_weight(
        Q_p_mat, overrides.get("Q_p"), "Q_p"
    )
    Q_v_final = _resolve_terminal_weight(
        Q_v_mat, overrides.get("Q_v"), "Q_v"
    )

    running = [
        ControlEffortTerm(
            R=float(overrides.get("R", config.R)),
            actuator_mode=actuator_mode,
            NU=NU,
        ),
        SmoothnessTerm(
            Q_qdot=float(overrides.get("Q_qdot_base", config.Q_qdot_base)),
            Q_qddot=float(overrides.get("Q_qddot_base", config.Q_qddot_base)),
            Q_du=float(overrides.get("Q_du_base", config.Q_du_base)),
            NQ=NQ, NX=NX, NU=NU, dt=env.dt,
        ),
        TcpSoftTerm(
            Q_tcp_soft=float(overrides.get("Q_tcp_soft", config.Q_tcp_soft)),
            tcp_threshold=float(overrides.get("tcp_soft_ratio", config.tcp_soft_ratio))
            * robot_limits.max_tcp_speed,
            NQ=NQ, NX=NX,
        ),
        QdotLimitTerm(
            Q_qdot_limit=float(
                overrides.get("Q_qdot_limit", config.Q_qdot_limit)
            ),
            qdot_limit_thresholds=float(
                overrides.get("qdot_limit_ratio", config.qdot_limit_ratio)
            )
            * robot_limits.qdot_max,
            NQ=NQ, NX=NX,
        ),
    ]

    terminal = [
        TerminalHitTerm(
            p_hit=p_hit, v_hit=v_hit,
            Q_p=Q_p_final, Q_v=Q_v_final,
            Q_n=float(overrides.get("normal_weight", config.normal_weight)),
            n_des=n_des,
            NX=NX, NQ=NQ,
        ),
    ]

    return CompositeCost(env, running, terminal)


# 工厂消费的 task cost_config 顶层键（MPCConfig 字段名 + 3.1 YAML 终端基准键）
_FACTORY_COST_KEYS: frozenset[str] = frozenset(
    {
        "R",
        "Q_qdot_base",
        "Q_qddot_base",
        "Q_du_base",
        "Q_tcp_soft",
        "Q_qdot_limit",
        "normal_weight",
        "tcp_soft_ratio",
        "qdot_limit_ratio",
        "Q_p",
        "Q_v",
        "tube",
    }
)


def _validate_task_cost_config(
    cost_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """校验 task cost_config 键名并返回普通 dict（None → 空 dict）。

    键名口径：工厂消费的 MPCConfig 字段名 + 3.1 YAML 终端基准键 Q_p/Q_v
    + tube 子字典（仅校验，不消费）。未知键（含杜撰键、Q_p_base 等
    调度口径别名）快速失败，不静默忽略。

    Args:
        cost_config: task YAML 的 cost_config 映射；None 表示缺省路径。

    Returns:
        可覆盖键值 dict（空 = 无覆盖）。

    Raises:
        ValueError: 未知顶层键，或 tube 子字典含非 TubeConfig 字段名。
    """
    if cost_config is None:
        return {}
    unknown = set(cost_config) - _FACTORY_COST_KEYS
    if unknown:
        raise ValueError(
            f"cost_config 含未识别键 {sorted(unknown)}；"
            f"可用键（MPCConfig 字段名 / 3.1 YAML 终端基准键）: "
            f"{sorted(_FACTORY_COST_KEYS)}。"
            "注：Q_p_base/Q_v_base 是 replan_core 的 sigmoid 调度口径，"
            "请经 MPCConfig 或 Q_p_mat/Q_v_mat 传入。"
        )
    tube_cfg = cost_config.get("tube")
    if tube_cfg is not None:
        from dataclasses import fields as dc_fields

        from src.ilqt.tube_types import TubeConfig

        if not isinstance(tube_cfg, Mapping):
            raise ValueError("cost_config['tube'] 必须是映射（TubeConfig 字段名口径）")
        valid_tube = {f.name for f in dc_fields(TubeConfig)}
        tube_unknown = set(tube_cfg) - valid_tube
        if tube_unknown:
            raise ValueError(
                f"cost_config['tube'] 含非 TubeConfig 字段 {sorted(tube_unknown)}；"
                f"可用字段: {sorted(valid_tube)}"
            )
    return dict(cost_config)


def _resolve_terminal_weight(
    mat: np.ndarray | None,
    override: Any,
    name: str,
) -> np.ndarray:
    """解析终端基准权重矩阵（cost_config 覆盖 > 显式矩阵 > 拒绝）。

    Args:
        mat: 调用方预计算的权重（(3,)/(3,3) 均可，交由 TerminalHitTerm
            统一化）或 None。
        override: cost_config 提供的标量 / (3,) / (3,3) 权重。
        name: 键名（用于报错信息）。

    Returns:
        (3,3) 或 (3,) 权重数组。

    Raises:
        ValueError: 两个来源均未提供（不静默零权重）。
    """
    if override is not None:
        arr = np.asarray(override, dtype=float)
        if arr.ndim == 0:
            return float(arr) * np.eye(3)
        return arr
    if mat is not None:
        return mat
    raise ValueError(
        f"终端权重 {name} 未提供：须传入 {name}_mat 或 cost_config['{name}']，"
        "不接受静默零权重"
    )
