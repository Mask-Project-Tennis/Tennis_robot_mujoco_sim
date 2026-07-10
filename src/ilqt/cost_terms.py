"""惩罚项独立模块 — 每个惩罚项是一个可独立测试的深模块。

组合优于继承：惩罚项通过 CompositeCost 组合，不通过继承扩展。
每个项实现 RunningCost 或 TerminalCost Protocol（或两者）。

Flyweight 模式：导数容器在构造时预分配，每步原地更新，返回持久引用。
"""

from __future__ import annotations

import numpy as np

from src.ilqt.cost import RunningDerivatives, TerminalDerivatives

__all__ = [
    "ControlEffortTerm",
    "SmoothnessTerm",
    "QdotLimitTerm",
    "TcpSoftTerm",
    "TerminalHitTerm",
    "JointLimitTerm",
    "BodyAvoidanceTerm",
    "XWallTerm",
]


class ControlEffortTerm:
    """控制力矩代价 l = ½ uᵀ R u。

    力矩模式（actuator_mode=0）：R·u² 全矩阵代价（含 R_joint_scale 关节级缩放）。
    位置模式（actuator_mode=1）：R=0，控制代价为零。
    R 调度：时变 R_schedule[k] 覆盖常数 R（退火策略）。

    needs_fk = False（纯控制空间，不需要末端 FK）。
    """

    needs_fk = False

    def __init__(
        self,
        R: float,
        R_joint_scale: dict[int, float] | None = None,
        R_schedule: np.ndarray | None = None,
        actuator_mode: int = 0,
        NU: int = 6,
    ) -> None:
        """初始化控制代价项。

        Args:
            R: 控制代价权重（标量）。
            R_joint_scale: 关节级缩放，{关节索引: 缩放因子}。
            R_schedule: 时变 R 值，形状 (N,)。None 表示常数 R。
            actuator_mode: 0=力矩模式, 1=位置模式（R=0）。
            NU: 控制维度。
        """
        self._actuator_mode = actuator_mode
        self._R_schedule = R_schedule
        # 构建基础 R 矩阵
        if actuator_mode == 1:
            self._R_mat = np.zeros((NU, NU))
        else:
            self._R_mat = R * np.eye(NU)
            for j_idx, scale in (R_joint_scale or {}).items():
                self._R_mat[j_idx, j_idx] *= scale
        self._eye_nu = np.eye(NU)
        # Flyweight：预分配导数容器
        self._derivs = RunningDerivatives(
            l_u=np.zeros(NU),
            l_uu=np.zeros((NU, NU)),
        )

    def running_cost(self, x, u, k, fk) -> float:
        """计算控制代价。"""
        if self._actuator_mode == 1:
            return 0.0
        if k is not None and self._R_schedule is not None and k < len(self._R_schedule):
            R_k = self._R_schedule[k]
            if np.ndim(R_k) == 0:
                return 0.5 * R_k * float(u @ u)
            return 0.5 * float(u @ (R_k * u))
        return 0.5 * float(u @ self._R_mat @ u)

    def running_derivatives(self, x, u, k, fk) -> RunningDerivatives:
        """计算控制代价导数。原地更新预分配容器，返回持久引用。"""
        if self._actuator_mode == 1:
            self._derivs.l_u.fill(0)
            self._derivs.l_uu.fill(0)
        elif k is not None and self._R_schedule is not None and k < len(self._R_schedule):
            R_k = self._R_schedule[k]
            if np.ndim(R_k) == 0:
                np.copyto(self._derivs.l_u, R_k * u)
                np.copyto(self._derivs.l_uu, R_k * self._eye_nu)
            else:
                np.copyto(self._derivs.l_u, R_k * u)
                np.copyto(self._derivs.l_uu, np.diag(R_k))
        else:
            np.copyto(self._derivs.l_u, self._R_mat @ u)
            np.copyto(self._derivs.l_uu, self._R_mat)
        return self._derivs

    def set_R_schedule(self, R_schedule: np.ndarray | None) -> None:
        """更新时变 R 调度（MPC 重规划时调用）。"""
        self._R_schedule = R_schedule


class SmoothnessTerm:
    """关节平滑代价 l = ½(Q_qdot·|qdot|² + Q_qddot·|qdot/dt|² + Q_du·|Δu|²)。

    Q_qdot：关节速度软平滑权重（惩罚速度幅值）。
    Q_qddot：关节加速度软平滑权重（θ̈ ≈ θ̇/dt 近似）。
    Q_du：控制变化率权重（||u_k - u_{k-1}||²），需要 set_u_prev。

    needs_fk = False（纯关节空间，不需要末端 FK）。
    实现 SmoothnessMixin（仅此项需要 set_u_prev）与 SmoothnessScaleUpdatable
    （MPC far/mid/near 阶段动态缩放）。

    移植来源：src/ilqt/cost.py:1075-1134（_add_smoothness_cost / _derivatives）。

    None 语义（对齐 ControlEffortTerm）：Q_qdot=Q_qddot=0 时 l_x=l_xx=None；
    Q_du=0 时 l_u=l_uu=None。基于基础权重判定（0*scale 恒为 0，稳定）。
    """

    needs_fk = False

    def __init__(
        self,
        Q_qdot: float,
        Q_qddot: float,
        Q_du: float,
        NQ: int = 6,
        NX: int = 12,
        NU: int = 6,
        dt: float = 0.005,
    ) -> None:
        """初始化平滑代价项。

        Args:
            Q_qdot: 关节速度平滑权重（0=禁用）。
            Q_qddot: 关节加速度平滑权重（0=禁用）。
            Q_du: 控制变化率权重（0=禁用）。
            NQ: 关节维度。
            NX: 状态维度。
            NU: 控制维度。
            dt: 时间步长（用于 qddot ≈ qdot/dt）。
        """
        self._Q_qdot = max(0.0, Q_qdot)
        self._Q_qddot = max(0.0, Q_qddot)
        self._Q_du = max(0.0, Q_du)
        self._dt = dt
        self._NQ = NQ
        self._NU = NU
        self._u_prev: np.ndarray | None = None
        # 动态缩放后的有效权重（MPC 分阶段调度用；0*scale 恒为 0）
        self._Q_qdot_eff = self._Q_qdot
        self._Q_qddot_eff = self._Q_qddot
        self._Q_du_eff = self._Q_du
        # 预分配单位矩阵
        self._eye_nq = np.eye(NQ)
        self._eye_nu = np.eye(NU)
        # None 语义：仅分配该项实际贡献的导数数组（基于基础权重，稳定）
        has_state = (self._Q_qdot > 0) or (self._Q_qddot > 0)
        has_control = self._Q_du > 0
        # Flyweight：预分配导数容器，每步原地更新
        self._derivs = RunningDerivatives(
            l_x=np.zeros(NX) if has_state else None,
            l_xx=np.zeros((NX, NX)) if has_state else None,
            l_u=np.zeros(NU) if has_control else None,
            l_uu=np.zeros((NU, NU)) if has_control else None,
        )

    def set_u_prev(self, u_prev: np.ndarray) -> None:
        """设置上一帧控制量（用于 Q_du 计算控制变化率）。

        实现 SmoothnessMixin Protocol。solver 用 isinstance(cost, SmoothnessMixin)
        替代 hasattr 守卫。

        Args:
            u_prev: 上一帧控制量 u_{k-1}，形状 (NU,)。
        """
        self._u_prev = u_prev.copy()

    def set_smoothness_scale(
        self, qdot_scale: float, qddot_scale: float, du_scale: float
    ) -> None:
        """动态调整软平滑项权重（MPC far/mid/near 阶段策略用）。

        实现 SmoothnessScaleUpdatable Protocol。
        活跃调用点：replan_core.py 分阶段平滑度调度。

        Args:
            qdot_scale: Q_qdot 缩放因子。
            qddot_scale: Q_qddot 缩放因子。
            du_scale: Q_du 缩放因子。
        """
        self._Q_qdot_eff = self._Q_qdot * qdot_scale
        self._Q_qddot_eff = self._Q_qddot * qddot_scale
        self._Q_du_eff = self._Q_du * du_scale

    def running_cost(self, x, u, k, fk) -> float:
        """计算平滑代价。

        Args:
            x: 臂状态 [q(NQ), qdot(NQ)]，形状 (NX,)。
            u: 控制量，形状 (NU,)。
            k: 当前时间步索引。None 表示无时间依赖。
            fk: 末端 FK 上下文（本项不使用，保持接口一致）。

        Returns:
            平滑代价值。
        """
        nq = self._NQ
        qdot = x[nq:]
        cost = 0.0
        if self._Q_qdot_eff > 0:
            cost += 0.5 * self._Q_qdot_eff * float(qdot @ qdot)
        if self._Q_qddot_eff > 0 and k is not None:
            qddot = qdot / self._dt
            cost += 0.5 * self._Q_qddot_eff * float(qddot @ qddot)
        if self._Q_du_eff > 0 and k is not None and k > 0 and self._u_prev is not None:
            du = u - self._u_prev
            cost += 0.5 * self._Q_du_eff * float(du @ du)
        return cost

    def running_derivatives(self, x, u, k, fk) -> RunningDerivatives:
        """计算平滑代价导数。原地更新预分配容器，返回持久引用。

        Args:
            x: 臂状态 [q(NQ), qdot(NQ)]，形状 (NX,)。
            u: 控制量，形状 (NU,)。
            k: 当前时间步索引。None 表示无时间依赖。
            fk: 末端 FK 上下文（本项不使用，保持接口一致）。

        Returns:
            RunningDerivatives，未贡献项为 None。
        """
        nq = self._NQ
        qdot = x[nq:]
        d = self._derivs
        # 读入局部变量以利类型收窄（l_x/l_xx 同生灭，l_u/l_uu 同生灭）
        l_x = d.l_x
        l_xx = d.l_xx
        l_u = d.l_u
        l_uu = d.l_uu
        # 原地清零（仅对已分配的数组）
        if l_x is not None:
            l_x.fill(0)
        if l_xx is not None:
            l_xx.fill(0)
        if l_u is not None:
            l_u.fill(0)
        if l_uu is not None:
            l_uu.fill(0)

        if self._Q_qdot_eff > 0 and l_x is not None and l_xx is not None:
            l_x[nq:] += self._Q_qdot_eff * qdot
            l_xx[nq:, nq:] += self._Q_qdot_eff * self._eye_nq
        if self._Q_qddot_eff > 0 and k is not None and l_x is not None and l_xx is not None:
            scale = self._Q_qddot_eff / (self._dt * self._dt)
            l_x[nq:] += scale * qdot
            l_xx[nq:, nq:] += scale * self._eye_nq
        if (
            self._Q_du_eff > 0
            and k is not None
            and k > 0
            and self._u_prev is not None
            and l_u is not None
            and l_uu is not None
        ):
            du = u - self._u_prev
            l_u += self._Q_du_eff * du
            l_uu += self._Q_du_eff * self._eye_nu
        return self._derivs


class QdotLimitTerm:
    """关节速度阈值软惩罚 l = ½·Q·Σⱼ max(0, |qdot_j| - threshold_j)²。

    与安全滤波的硬限位不同，这是代价函数中的软惩罚，鼓励求解器
    主动降低关节速度。Q=0 时禁用。

    needs_fk = False（纯关节空间速度，从状态 x[NQ:] 读取 qdot）。
    移植来源：src/ilqt/cost.py:574-579（cost）+ :748-756（derivatives）。
    """

    needs_fk = False

    def __init__(
        self,
        Q_qdot_limit: float,
        qdot_limit_thresholds: np.ndarray,
        NQ: int = 6,
        NX: int = 12,
    ) -> None:
        """初始化关节速度限制项。

        Args:
            Q_qdot_limit: 软惩罚权重（0=禁用）。
            qdot_limit_thresholds: 各关节速度阈值 (NQ,)。
            NQ: 关节维度。
            NX: 状态维度。
        """
        self._Q = max(0.0, Q_qdot_limit)
        self._thresholds = np.asarray(qdot_limit_thresholds, dtype=float).copy()
        self._NQ = NQ
        # Flyweight：预分配导数容器（此项仅依赖状态 x，不依赖控制 u）
        self._derivs = RunningDerivatives(
            l_x=np.zeros(NX),
            l_xx=np.zeros((NX, NX)),
        )

    def running_cost(self, x, u, k, fk) -> float:
        """计算关节速度超限代价。

        Args:
            x: 臂状态 [q(NQ), qdot(NQ)]，形状 (NX,)。
            u: 控制量（本项不使用，保持接口一致）。
            k: 当前时间步索引（本项不使用）。
            fk: 末端 FK 上下文（本项不使用）。

        Returns:
            关节速度超限代价值。
        """
        if self._Q <= 0:
            return 0.0
        qdot = x[self._NQ:]
        excess = np.maximum(0.0, np.abs(qdot) - self._thresholds)
        return 0.5 * self._Q * float(excess @ excess)

    def running_derivatives(self, x, u, k, fk) -> RunningDerivatives:
        """计算关节速度超限导数。原地更新预分配容器，返回持久引用。

        对每个超限关节 j：l_x[NQ+j] = Q·excess·sign(qdot_j)，l_xx[NQ+j,NQ+j] = Q。
        符号函数取 qdot>=0 → +1，qdot<0 → -1（与原 cost.py 一致）。

        Args:
            x: 臂状态 [q(NQ), qdot(NQ)]，形状 (NX,)。
            u: 控制量（本项不使用）。
            k: 当前时间步索引（本项不使用）。
            fk: 末端 FK 上下文（本项不使用）。

        Returns:
            RunningDerivatives，l_u/l_uu 为 None（不依赖控制）。
        """
        self._derivs.l_x.fill(0)
        self._derivs.l_xx.fill(0)
        if self._Q <= 0:
            return self._derivs
        qdot = x[self._NQ:]
        excess = np.maximum(0.0, np.abs(qdot) - self._thresholds)
        mask = excess > 0
        if np.any(mask):
            # sign(qdot)，qdot>=0 → +1（与原 cost.py :754 一致）
            sign = np.where(qdot >= 0, 1.0, -1.0)
            self._derivs.l_x[self._NQ:] = self._Q * excess * sign * mask
            # Hessian 对角块：超限关节对角元 = Q
            diag_idx = np.arange(self._NQ)
            self._derivs.l_xx[
                self._NQ + diag_idx, self._NQ + diag_idx
            ] = self._Q * mask
        return self._derivs


class TcpSoftTerm:
    """TCP 速度软惩罚 l = ½·Q·max(0, |v_tcp| - threshold)²。

    v_tcp = J_p @ qdot，末端线速度（位置雅可比作用于关节速度）。
    超过阈值时二次惩罚，鼓励求解器在代价层面约束 TCP 速度。
    Q=0 时禁用。

    needs_fk = True（需要 FK 位置雅可比 J_p）。
    移植来源：src/ilqt/cost.py:566-572（cost）+ :735-746（derivatives）。

    导数采用与 cost.py:744-746 一致的近似（非标准 Gauss-Newton）：
        ∂l/∂qdot = Q·excess·J_pᵀ·v̂
        ∂²l/∂qdot² = J_pᵀ·(Q·v̂v̂ᵀ/speed)·J_p   （无 excess 因子）
    其中 v̂ = v_tcp/|v_tcp|，仅在 excess>0 且 speed>1e-8 时贡献。
    """

    needs_fk = True

    def __init__(
        self,
        Q_tcp_soft: float,
        tcp_threshold: float,
        NQ: int = 6,
        NX: int = 12,
    ) -> None:
        """初始化 TCP 软惩罚项。

        Args:
            Q_tcp_soft: 软惩罚权重（0=禁用）。
            tcp_threshold: TCP 速度阈值（m/s）。
            NQ: 关节维度。
            NX: 状态维度。
        """
        self._Q = max(0.0, Q_tcp_soft)
        self._threshold = float(tcp_threshold)
        self._NQ = NQ
        # Flyweight：预分配导数容器（此项仅依赖状态 x，不依赖控制 u）
        self._derivs = RunningDerivatives(
            l_x=np.zeros(NX),
            l_xx=np.zeros((NX, NX)),
        )

    def running_cost(self, x, u, k, fk) -> float:
        """计算 TCP 速度超限代价。

        Args:
            x: 臂状态 [q(NQ), qdot(NQ)]，形状 (NX,)。
            u: 控制量（本项不使用，保持接口一致）。
            k: 当前时间步索引（本项不使用）。
            fk: 末端 FK 上下文，提供位置雅可比 J_p (3, NQ)。

        Returns:
            TCP 速度超限代价值。
        """
        if self._Q <= 0:
            return 0.0
        J_p = fk.J_p  # (3, NQ)
        qdot = x[self._NQ:]
        tcp_vel = J_p @ qdot
        tcp_speed = float(np.linalg.norm(tcp_vel))
        if tcp_speed <= self._threshold:
            return 0.0
        excess = tcp_speed - self._threshold
        return 0.5 * self._Q * excess * excess

    def running_derivatives(self, x, u, k, fk) -> RunningDerivatives:
        """计算 TCP 速度超限导数（与 cost.py:744-746 一致的近似）。原地更新，返回持久引用。

        Args:
            x: 臂状态 [q(NQ), qdot(NQ)]，形状 (NX,)。
            u: 控制量（本项不使用）。
            k: 当前时间步索引（本项不使用）。
            fk: 末端 FK 上下文，提供位置雅可比 J_p (3, NQ)。

        Returns:
            RunningDerivatives，l_u/l_uu 为 None（不依赖控制）。
        """
        self._derivs.l_x.fill(0)
        self._derivs.l_xx.fill(0)
        if self._Q <= 0:
            return self._derivs
        J_p = fk.J_p  # (3, NQ)
        qdot = x[self._NQ:]
        tcp_vel = J_p @ qdot
        tcp_speed = float(np.linalg.norm(tcp_vel))
        if tcp_speed <= self._threshold or tcp_speed < 1e-8:
            return self._derivs
        excess = tcp_speed - self._threshold
        # ∂l/∂qdot = Q·excess·J_pᵀ·v̂（与 cost.py:744 一致）
        v_hat = tcp_vel / tcp_speed
        self._derivs.l_x[self._NQ:] = self._Q * excess * (J_p.T @ v_hat)
        # Hessian：与 cost.py:745-746 一致（非标准 Gauss-Newton，无 excess 因子）
        #   H_tcp = Q·(v̂ v̂ᵀ)/speed（3×3），l_xx = J_pᵀ H_tcp J_p
        H_tcp = (self._Q / tcp_speed) * np.outer(v_hat, v_hat)
        self._derivs.l_xx[self._NQ:, self._NQ:] = J_p.T @ H_tcp @ J_p
        return self._derivs


class TerminalHitTerm:
    """终端击打代价 l_N = ½||p_ee-p_hit||²_Qp + ½||v_ee-v_hit||²_Qv + ½Q_n||n-n_des||²。

    位置代价：末端偏离期望击打点。
    速度代价：末端速度偏离期望击打速度。
    法向量代价：拍面法向量偏离期望方向（Q_n=0 时禁用）。

    needs_fk = True（需要末端位置/速度/法向量/雅可比）。
    实现 TargetUpdatable + WeightUpdatable（MPC 动态更新）。

    移植来源：src/ilqt/cost.py:603-648（terminal_cost）+ :850-865（terminal_derivatives）。
    Gauss-Newton 近似与原实现一致：
        J_h ≈ [J_p,  0 ]        h = [p_ee; v_ee]
              [ 0,  J_p]
    故 l_x[:NQ]=J_pᵀQ_p·dp，l_x[NQ:]=J_pᵀQ_v·dv，l_xx 对角块化。
    """

    needs_fk = True

    def __init__(
        self,
        p_hit: np.ndarray,
        v_hit: np.ndarray,
        Q_p: np.ndarray,
        Q_v: np.ndarray,
        Q_n: float = 0.0,
        n_des: np.ndarray | None = None,
        NX: int = 12,
        NQ: int = 6,
    ) -> None:
        """初始化终端击打代价项。

        Args:
            p_hit: 期望击打位置 (3,)。
            v_hit: 期望击打速度 (3,)。
            Q_p: 位置代价权重 (3,) 或 (3,3)。(3,) 视为对角线。
            Q_v: 速度代价权重 (3,) 或 (3,3)。(3,) 视为对角线。
            Q_n: 法向量代价权重（标量，0=禁用）。
            n_des: 期望拍面法向量 (3,)。None 且 Q_n>0 时不参与。
            NX: 状态维度。
            NQ: 关节维度。
        """
        self._NQ = NQ
        self._NX = NX
        self.p_hit = p_hit.copy()
        self.v_hit = v_hit.copy()
        # 接受 (3,) 或 (3,3)，统一化为矩阵形式
        self._Q_p_base = np.diag(Q_p) if Q_p.ndim == 1 else Q_p.copy()
        self._Q_v_base = np.diag(Q_v) if Q_v.ndim == 1 else Q_v.copy()
        self.Q_p = self._Q_p_base.copy()
        self.Q_v = self._Q_v_base.copy()
        self.Q_n = Q_n
        self.n_des = n_des.copy() if n_des is not None else None
        # Flyweight：预分配终端导数容器
        self._derivs = TerminalDerivatives(
            l_x=np.zeros(NX),
            l_xx=np.zeros((NX, NX)),
        )

    def terminal_cost(self, x, fk) -> float:
        """计算终端击打代价。

        Args:
            x: 臂状态 [q(NQ), qdot(NQ)]，形状 (NX,)（本项不直接读 x，全部从 fk 取）。
            fk: 末端 FK 上下文，提供 p_ee/v_ee/n_rack。

        Returns:
            终端代价值。
        """
        dp = fk.p_ee - self.p_hit
        cost = 0.5 * float(dp @ self.Q_p @ dp)
        dv = fk.v_ee - self.v_hit
        cost += 0.5 * float(dv @ self.Q_v @ dv)
        if self.n_des is not None and self.Q_n > 0:
            n_err = fk.n_rack - self.n_des
            cost += 0.5 * self.Q_n * float(n_err @ n_err)
        return cost

    def terminal_derivatives(self, x, fk) -> TerminalDerivatives:
        """计算终端击打代价导数（Gauss-Newton 近似）。原地更新预分配容器，返回持久引用。

        数学：
            l_x[:NQ]  = J_pᵀ Q_p dp           （位置部分对 q 求导）
            l_x[NQ:]  = J_pᵀ Q_v dv           （速度部分对 qdot 求导，v_ee=J_p·qdot）
            l_xx[:NQ,:NQ] = J_pᵀ Q_p J_p
            l_xx[NQ:,NQ:] = J_pᵀ Q_v J_p
            法向量贡献：l_x += Q_n·J_nᵀ·n_err，l_xx += Q_n·J_nᵀ·J_n
        其中 J_n (3, NX) = skew(-n)·J_ω（仅前 NQ 列非零，法向量不依赖 qdot）。

        Args:
            x: 臂状态 [q(NQ), qdot(NQ)]，形状 (NX,)（本项不直接读 x）。
            fk: 末端 FK 上下文，提供 J_p/J_n/n_rack/p_ee/v_ee。

        Returns:
            TerminalDerivatives，l_x (NX,) + l_xx (NX,NX)。
        """
        n_q = self._NQ
        J_p = fk.J_p  # (3, NQ)

        dp = fk.p_ee - self.p_hit
        dv = fk.v_ee - self.v_hit

        # 原地清零（Flyweight 模式）
        self._derivs.l_x.fill(0)
        self._derivs.l_xx.fill(0)

        # 位置部分：l_x[:n_q] = J_pᵀ Q_p dp，l_xx[:n_q,:n_q] = J_pᵀ Q_p J_p
        self._derivs.l_x[:n_q] = J_p.T @ self.Q_p @ dp
        self._derivs.l_xx[:n_q, :n_q] = J_p.T @ self.Q_p @ J_p

        # 速度部分：v_ee = J_p·qdot，l_x[n_q:] = J_pᵀ Q_v dv，l_xx[n_q:,n_q:] = J_pᵀ Q_v J_p
        self._derivs.l_x[n_q:] = J_p.T @ self.Q_v @ dv
        self._derivs.l_xx[n_q:, n_q:] = J_p.T @ self.Q_v @ J_p

        # 法向量部分（Q_n=0 或 n_des=None 时禁用）
        if self.n_des is not None and self.Q_n > 0:
            J_n = fk.J_n  # (3, NX)
            n_err = fk.n_rack - self.n_des
            self._derivs.l_x += self.Q_n * (J_n.T @ n_err)
            self._derivs.l_xx += self.Q_n * (J_n.T @ J_n)

        return self._derivs

    def update_target(
        self,
        p_hit: np.ndarray,
        v_hit: np.ndarray,
        n_des: np.ndarray | None = None,
    ) -> None:
        """更新击打目标（MPC 重规划时调用）。

        实现 TargetUpdatable Protocol。

        Args:
            p_hit: 新的期望击打位置 (3,)。
            v_hit: 新的期望击打速度 (3,)。
            n_des: 新的期望拍面法向量 (3,)。None 表示保持不变。
        """
        self.p_hit = p_hit.copy()
        self.v_hit = v_hit.copy()
        if n_des is not None:
            self.n_des = n_des.copy()

    def update_weights(self, Q_p_scale: float = 1.0, Q_v_scale: float = 1.0) -> None:
        """缩放位置/速度权重（MPC far/near 阶段调度）。

        实现 WeightUpdatable Protocol。基于构造时的基准权重缩放，
        保证可逆：scale=1.0 恢复基准。

        Args:
            Q_p_scale: 位置权重缩放因子。
            Q_v_scale: 速度权重缩放因子。
        """
        self.Q_p = self._Q_p_base * Q_p_scale
        self.Q_v = self._Q_v_base * Q_v_scale


# ── 可选惩罚项（旧脚本/实验用，V12 生产路径未启用）──


class BodyAvoidanceTerm:
    """身体碰撞规避代价 — 将躯干建模为竖直圆柱体，惩罚臂关键点进入半径内。

    移植自 HittingCost._add_body_avoidance_cost (cost.py:801-824)
    和 _add_body_avoidance_derivatives (cost.py:825-864)。

    needs_fk = True（需 body_pos_by_id，FKContext 需扩展或直接传 env）。
    """

    needs_fk = True

    def __init__(
        self,
        center_xy: np.ndarray,
        radius: float,
        Q_body: float,
        body_names: list[str],
        NX: int = 12,
        NQ: int = 6,
    ) -> None:
        """初始化身体规避项。

        Args:
            center_xy: 圆柱体中心 (2,)，XY 平面。
            radius: 圆柱体半径（m）。
            Q_body: 规避代价权重。
            body_names: 需检测的 body 名称列表。
            NX: 状态维度（占位，实现时使用）。
            NQ: 关节维度（占位，实现时使用）。
        """
        # TODO: 移植 _init_body_avoidance_ids + _add_body_avoidance_cost/derivatives
        # 注意：此项需要 env.get_body_pos_by_id，FKContext 需扩展
        raise NotImplementedError("BodyAvoidanceTerm — Phase 4 迁移旧脚本时实现")


class JointLimitTerm:
    """关节角度安全范围软惩罚 l = ½·Q·Σ hinge(q_j, lo_j, hi_j)。

    移植自 HittingCost.running_cost 的 joint_limits 分支 (cost.py:552-560)
    及 _running_cost_derivatives 的 joint_limits 分支 (cost.py:714-726)。

    hinge 形式（单侧二次惩罚）：
        超下界贡献 = ½·Q·max(0, lo - q_j)²
        超上界贡献 = ½·Q·max(0, q_j - hi)²

    needs_fk = False（纯关节空间，从状态 x[:NQ] 读取 q）。
    实现 RunningCost Protocol。
    """

    needs_fk = False

    def __init__(
        self,
        joint_limits: dict[int, tuple[float | None, float | None]],
        Q_joint_limit: float = 100000.0,
        NQ: int = 6,
        NX: int = 12,
    ) -> None:
        """初始化关节限位项。

        Args:
            joint_limits: {关节索引: (下界, 上界)}，None 表示该侧无边。
            Q_joint_limit: 限位惩罚权重。
            NQ: 关节维度。
            NX: 状态维度。
        """
        self._Q = Q_joint_limit
        self._NQ = NQ
        # 预计算向量化数组：将 dict 展开为 (NQ,) 数组
        # None 边界替换为 ±inf 使 max(0, ...) 自然为零
        self._lo = np.full(NQ, -np.inf)
        self._hi = np.full(NQ, np.inf)
        self._active_lo = np.zeros(NQ, dtype=bool)
        self._active_hi = np.zeros(NQ, dtype=bool)
        for j, (lo, hi) in joint_limits.items():
            if lo is not None:
                self._lo[j] = lo
                self._active_lo[j] = True
            if hi is not None:
                self._hi[j] = hi
                self._active_hi[j] = True
        # Flyweight：预分配导数容器
        self._derivs = RunningDerivatives(
            l_x=np.zeros(NX),
            l_xx=np.zeros((NX, NX)),
        )

    def running_cost(self, x, u, k, fk) -> float:
        """计算关节限位代价。

        Args:
            x: 臂状态 [q(NQ), qdot(NQ)]，形状 (NX,)。
            u: 控制量（本项不使用，保持接口一致）。
            k: 当前时间步索引（本项不使用）。
            fk: 末端 FK 上下文（本项不使用）。

        Returns:
            关节超限代价值。
        """
        q = x[:self._NQ]
        excess_lo = np.maximum(0.0, self._lo - q) * self._active_lo
        excess_hi = np.maximum(0.0, q - self._hi) * self._active_hi
        return 0.5 * self._Q * float(excess_lo @ excess_lo + excess_hi @ excess_hi)

    def running_derivatives(self, x, u, k, fk) -> RunningDerivatives:
        """计算关节限位导数。原地更新预分配容器，返回持久引用。

        超下界时：l_x[j] = -Q·margin，l_xx[j,j] = Q（margin = lo - q_j > 0）。
        超上界时：l_x[j] = +Q·margin，l_xx[j,j] = Q（margin = q_j - hi > 0）。

        Args:
            x: 臂状态 [q(NQ), qdot(NQ)]，形状 (NX,)。
            u: 控制量（本项不使用）。
            k: 当前时间步索引（本项不使用）。
            fk: 末端 FK 上下文（本项不使用）。

        Returns:
            RunningDerivatives，l_u/l_uu 为 None（不依赖控制）。
        """
        self._derivs.l_x.fill(0)
        self._derivs.l_xx.fill(0)
        q = x[:self._NQ]
        # 下界违反: margin > 0 时 l_x = -Q·margin, l_xx = Q
        excess_lo = np.maximum(0.0, self._lo - q) * self._active_lo
        mask_lo = excess_lo > 0
        if np.any(mask_lo):
            self._derivs.l_x[:self._NQ] -= np.where(mask_lo, self._Q * excess_lo, 0.0)
            diag_idx = np.where(mask_lo)[0]
            self._derivs.l_xx[diag_idx, diag_idx] = self._Q
        # 上界违反: margin > 0 时 l_x = +Q·margin, l_xx = Q
        excess_hi = np.maximum(0.0, q - self._hi) * self._active_hi
        mask_hi = excess_hi > 0
        if np.any(mask_hi):
            self._derivs.l_x[:self._NQ] += np.where(mask_hi, self._Q * excess_hi, 0.0)
            diag_idx = np.where(mask_hi)[0]
            self._derivs.l_xx[diag_idx, diag_idx] = self._Q
        return self._derivs


class XWallTerm:
    """X 平面墙约束 — 右臂 body 的 X 坐标必须 ≤ limit_x。

    移植自 HittingCost._add_x_limit_cost (cost.py:875-893)
    和 _add_x_limit_derivatives (cost.py:894-925)。

    needs_fk = True（需 body X 坐标）。
    """

    needs_fk = True

    def __init__(self, limit_x: float, Q_x: float, body_names: list[str]) -> None:
        """初始化 X 平面墙约束项。

        Args:
            limit_x: X 坐标上界（m），右臂 body 的 X 不得超过此值。
            Q_x: 越墙惩罚权重。
            body_names: 需检测的 body 名称列表。
        """
        # TODO: 移植 _init_x_limit_ids + _add_x_limit_cost/derivatives
        raise NotImplementedError("XWallTerm — Phase 4 迁移旧脚本时实现")
