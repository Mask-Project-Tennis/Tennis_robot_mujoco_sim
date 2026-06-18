"""Tube + Softmin 代价包装器。

包含 TubeHittingCostWrapper（空间走廊 + 多终端 softmin 代价）及其消融变体
（TubeOnlyCost / SoftminOnlyCost），与现有 iLQT solver 接口兼容。
供仿真主脚本（V11）与真机 runner 共享复用。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from src.ilqt.cost import HittingCost
from src.ilqt.tube_types import (
    TubeConfig,
    HittingTube,
)

if TYPE_CHECKING:
    from src.sim.rm65_env import RM65Env


logger = logging.getLogger(__name__)


class TubeHittingCostWrapper:
    """包装 HittingCost，在候选击球窗口内施加空间走廊式 tube 代价。

    空间走廊（hinge loss）：
      - 走廊半宽 = RACKET_RADIUS（固定）
      - hinge loss: margin = perp_dist - RACKET_RADIUS
      - 走廊内零代价，走廊外二次惩罚

    Softmin 多终端代价：
      - 终端代价在所有候选击球位置上取 softmin
      - 求解器不需要精确预测"何时"击球，只需到达某个候选位置即可获得低代价
      - 球早到/晚到时，对应的候选位置提供低代价路径

    原始设计（空间重合，而非时间追踪）：
    - 提取球在窗口内的轨迹线方向 d_ball，构建"空间走廊"
    - 在 tube 窗口内的每个 iLQR 步 k，注入三类代价：
      1. 垂直偏离代价（hinge loss）：不绑定时间-空间对应
      2. 速度方向代价：鼓励球拍沿球轨迹线方向运动
      3. 法向量代价：拍面朝向来球方向
    - 兼容 ILQTSolver 的接口
    """

    RACKET_RADIUS: float = 0.12

    def __init__(
        self,
        env: "RM65Env",
        base_cost: HittingCost,
        hitting_tube: HittingTube,
        horizon: int,
        config: TubeConfig,
    ) -> None:
        """初始化 Tube 代价包装器。

        Args:
            env: RM-65 环境实例。
            base_cost: 原始 HittingCost 实例（提供终端代价和基础运行代价）。
            hitting_tube: 击球管道。
            horizon: 规划地平线步数。
            config: Tube 配置。
        """
        self.env = env
        self.base_cost = base_cost
        self.hitting_tube = hitting_tube
        self.horizon = horizon
        self.config = config

        self._tube_ratio = config.tube_cost_ratio
        self._current_ratio = config.tube_cost_ratio
        self._anchor_alpha: float = 0.9
        self._Q_p_tube = config.Q_p_tube
        self._Q_v_tube = config.Q_v_tube
        self._Q_n_tube = config.Q_n_tube

        # P0-2: softmin 参数
        self._use_softmin = config.use_softmin_terminal
        self._softmin_beta = config.softmin_beta

        self._tube_steps: set[int] = set()
        self._tube_weight_scales: dict[int, float] = {}
        self._d_ball: np.ndarray = np.zeros(3)
        self._P_perp: np.ndarray = np.zeros((3, 3))
        self._p_ball_ref: np.ndarray = np.zeros(3)
        self._n_des_common: np.ndarray = np.zeros(3)
        # P0-2: 多终端候选信息（用于 softmin）
        self._p_ball_candidates: np.ndarray = np.zeros((0, 3))
        self._v_des_candidates: np.ndarray = np.zeros((0, 3))
        self._n_des_candidates: np.ndarray = np.zeros((0, 3))
        self._candidate_weights: np.ndarray = np.zeros(0)
        # 可解释性日志：最近一次 softmin 权重和候选代价
        self._last_softmin_alphas: np.ndarray = np.zeros(0)
        self._last_softmin_costs: np.ndarray = np.zeros(0)
        # 可解释性日志：最近一次 tube 走廊 margin
        self._last_tube_margins: dict[int, float] = {}
        self._rebuild_tube_maps(hitting_tube, horizon)

    def _rebuild_tube_maps(self, tube: HittingTube, horizon: int) -> None:
        """重建 tube 步集合、权重缓存和终端候选信息。"""
        self._tube_steps.clear()
        self._tube_weight_scales.clear()

        if len(tube.k_candidates) == 0:
            self._d_ball = np.zeros(3)
            self._P_perp = np.zeros((3, 3))
            self._p_ball_ref = np.zeros(3)
            self._n_des_common = np.zeros(3)
            self._p_ball_candidates = np.zeros((0, 3))
            self._v_des_candidates = np.zeros((0, 3))
            self._n_des_candidates = np.zeros((0, 3))
            self._candidate_weights = np.zeros(0)
            return

        for i in range(len(tube.k_candidates)):
            k = int(tube.k_candidates[i])
            if 0 <= k < horizon:
                self._tube_steps.add(k)
                self._tube_weight_scales[k] = float(tube.weights[i]) * self._current_ratio

        # 球轨迹线方向：用窗口内所有候选球速度的加权平均方向
        weights = tube.weights[:, np.newaxis]
        v_ball_mean = np.sum(weights * tube.v_ball, axis=0)
        v_norm = np.linalg.norm(v_ball_mean)
        if v_norm > 1e-6:
            self._d_ball = v_ball_mean / v_norm
        else:
            self._d_ball = np.array([0.0, -1.0, 0.0])

        # 垂直投影矩阵：P_perp = I - d_ball @ d_ball.T
        self._P_perp = np.eye(3) - np.outer(self._d_ball, self._d_ball)

        # 参考点：best_k 时刻的球位置（走廊中心线上的参考）
        best_idx = int(np.argmin(np.abs(tube.k_candidates - tube.best_k)))
        self._p_ball_ref = tube.p_ball[best_idx].copy()

        # 法向量：用 best_k 对应的拍面法向
        self._n_des_common = tube.n_racket_des[best_idx].copy()

        # P0-2: 保存所有候选位置用于多终端 softmin
        self._p_ball_candidates = tube.p_ball.copy()
        self._v_des_candidates = tube.v_racket_des.copy()
        self._n_des_candidates = tube.n_racket_des.copy()
        self._candidate_weights = tube.weights.copy()

        if len(self._candidate_weights) > 0:
            top3 = np.argsort(self._candidate_weights)[-min(3, len(self._candidate_weights)):][::-1]
            w_str = ", ".join(
                f"k={int(tube.k_candidates[i])}:w={self._candidate_weights[i]:.3f}"
                for i in top3
            )
            logger.info("[Softmin诊断] 候选权重TOP3: %s", w_str)

    def running_cost(self, x: np.ndarray, u: np.ndarray, k: int | None = None) -> float:
        """计算运行代价 = 原始运行代价 + tube 代价（若 k 在候选窗口内）。"""
        cost = self.base_cost.running_cost(x, u, k)
        if k is not None and k in self._tube_steps:
            cost += self._compute_tube_cost_at_k(x, k)
        return cost

    def terminal_cost(self, x: np.ndarray) -> float:
        """计算终端代价。

        V2 改进（P0-2）：
        若启用 softmin，终端代价在所有候选击球位置上取 softmin：
          cost = -log(Σ_i w_i * exp(-β * c_i)) / β
        其中 c_i = ||p_ee - p_ball[i]||²_Qp + ||v_ee - v_des[i]||²_Qv
        这允许求解器"选择"在任意候选时刻击球，容忍时间不确定性。

        若未启用 softmin，退化为原始单点终端代价。
        """
        self.env.set_arm_state(x)
        p_ee = self.env.get_ee_pos()
        v_ee = self.env.get_ee_vel()
        n_rack = self.env.get_ee_normal()

        if self._use_softmin and len(self._p_ball_candidates) > 1:
            return self._compute_softmin_terminal(p_ee, v_ee, n_rack)
        else:
            return self._compute_single_terminal(p_ee, v_ee, n_rack)

    def _compute_single_terminal(
        self, p_ee: np.ndarray, v_ee: np.ndarray, n_rack: np.ndarray
    ) -> float:
        """原始单点终端代价（best_k 处的精确击打约束）。

        v8 改进：终端代价不再乘以 (1-ratio)，始终完整保留。
        Tube 走廊代价作为辅助项叠加在运行代价上，不削弱终端目标。
        """
        dp = p_ee - self.base_cost.p_hit
        cost_p = 0.5 * float(dp @ self.base_cost.Q_p @ dp)

        dv = v_ee - self.base_cost.v_hit
        cost_v = 0.5 * float(dv @ self.base_cost.Q_v @ dv)

        cost = cost_p + cost_v
        if self.base_cost.n_des is not None and self.base_cost.Q_n > 0:
            n_err = n_rack - self.base_cost.n_des
            cost += 0.5 * self.base_cost.Q_n * float(n_err @ n_err)
        return cost

    def _compute_softmin_terminal(
        self, p_ee: np.ndarray, v_ee: np.ndarray, n_rack: np.ndarray
    ) -> float:
        """P0-2: 多终端 softmin 代价。

        对每个候选位置 i 计算：
          c_i = 0.5 * [ ||p_ee - p_ball_i||²_Qp + ||v_ee - v_des_i||²_Qv
                        + Q_n * ||n_rack - n_des_i||² ]
        然后取 softmin:
          cost = -log(Σ_i w_i * exp(-β * c_i)) / β

        softmin 的效果：只有代价最低的候选（最接近的候选位置）主导结果，
        但梯度从所有候选流向最优点附近的候选，保证光滑可微。

        Returns:
            终端代价值（已乘以 tube_ratio 缩放）。
        """
        M = len(self._p_ball_candidates)
        Q_p = self.base_cost.Q_p
        Q_v = self.base_cost.Q_v
        Q_n = self.base_cost.Q_n

        costs_i = np.zeros(M)
        for i in range(M):
            dp = p_ee - self._p_ball_candidates[i]
            costs_i[i] = 0.5 * float(dp @ Q_p @ dp)

            dv = v_ee - self._v_des_candidates[i]
            costs_i[i] += 0.5 * float(dv @ Q_v @ dv)

            if Q_n > 0:
                n_err = n_rack - self._n_des_candidates[i]
                costs_i[i] += 0.5 * Q_n * float(n_err @ n_err)

        # softmin: -log(Σ w_i * exp(-β * c_i)) / β
        # 数值稳定：减去最大值避免溢出
        beta = self._softmin_beta
        weighted_neg_costs = -beta * costs_i + np.log(self._candidate_weights + 1e-30)
        max_wnc = np.max(weighted_neg_costs)
        log_sum = max_wnc + np.log(np.sum(np.exp(weighted_neg_costs - max_wnc)))
        softmin_val = -log_sum / beta

        return softmin_val

    def running_derivatives(
        self, x: np.ndarray, u: np.ndarray, k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """计算运行代价导数 = 原始导数 + tube 导数（若 k 在候选窗口内）。"""
        l_x, l_u, l_xx, l_ux, l_uu = self.base_cost.running_derivatives(x, u, k)
        if k is not None and k in self._tube_steps:
            tl_x, tl_xx = self._compute_tube_derivatives_at_k(x, k)
            l_x += tl_x
            l_xx += tl_xx
        return l_x, l_u, l_xx, l_ux, l_uu

    def terminal_derivatives(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """终端代价导数。

        V2 改进（P0-2）：softmin 加权导数。
        softmin 的梯度 = Σ_i α_i * ∂c_i/∂x，其中 α_i 是 softmin 权重。
        """
        n_x = self.env.NX
        n_q = self.env.NQ

        self.env.set_arm_state(x)
        p_ee = self.env.get_ee_pos()
        v_ee = self.env.get_ee_vel()
        n_rack = self.env.get_ee_normal()
        J_p = self.env.get_ee_jacp()

        if self._use_softmin and len(self._p_ball_candidates) > 1:
            l_x, l_xx = self._compute_softmin_terminal_derivatives(
                p_ee, v_ee, n_rack, J_p, n_x, n_q
            )
        else:
            l_x, l_xx = self._compute_single_terminal_derivatives(
                p_ee, v_ee, n_rack, J_p, n_x, n_q
            )

        return l_x, l_xx

    def _compute_single_terminal_derivatives(
        self,
        p_ee: np.ndarray,
        v_ee: np.ndarray,
        n_rack: np.ndarray,
        J_p: np.ndarray,
        n_x: int,
        n_q: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """原始单点终端代价导数。"""
        K_p = self.base_cost.Q_p
        dp = p_ee - self.base_cost.p_hit

        l_x = np.zeros(n_x)
        l_xx = np.zeros((n_x, n_x))

        l_x[:n_q] = J_p.T @ K_p @ dp
        l_xx[:n_q, :n_q] = J_p.T @ K_p @ J_p

        dv = v_ee - self.base_cost.v_hit
        l_x[n_q:] = J_p.T @ self.base_cost.Q_v @ dv
        l_xx[n_q:, n_q:] = J_p.T @ self.base_cost.Q_v @ J_p

        if self.base_cost.n_des is not None and self.base_cost.Q_n > 0:
            J_omega = self.env.get_ee_jacr()
            nx, ny, nz = -n_rack[0], -n_rack[1], -n_rack[2]
            skew = np.array([
                [0, -nz, ny],
                [nz, 0, -nx],
                [-ny, nx, 0],
            ])
            J_n = np.zeros((3, n_x))
            J_n[:, :n_q] = skew @ J_omega
            n_err = n_rack - self.base_cost.n_des
            l_x += self.base_cost.Q_n * (J_n.T @ n_err)
            l_xx += self.base_cost.Q_n * (J_n.T @ J_n)

        return l_x, l_xx

    def _compute_softmin_terminal_derivatives(
        self,
        p_ee: np.ndarray,
        v_ee: np.ndarray,
        n_rack: np.ndarray,
        J_p: np.ndarray,
        n_x: int,
        n_q: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """P0-2: 多终端 softmin 代价的 Gauss-Newton 近似导数。

        softmin(c_1, ..., c_M) = -log(Σ w_i * exp(-β * c_i)) / β

        梯度：∂softmin/∂x = Σ_i α_i * ∂c_i/∂x
        Hessian 近似：Σ_i α_i * ∂²c_i/∂x²
        其中 α_i = w_i * exp(-β * c_i) / Σ_j w_j * exp(-β * c_j)  是 softmin 权重。

        Returns:
            (l_x, l_xx) — 终端代价对状态的一阶和二阶导数。
        """
        M = len(self._p_ball_candidates)
        Q_p = self.base_cost.Q_p
        Q_v = self.base_cost.Q_v
        Q_n = self.base_cost.Q_n
        beta = self._softmin_beta

        # 计算每个候选的代价
        costs_i = np.zeros(M)
        for i in range(M):
            dp = p_ee - self._p_ball_candidates[i]
            costs_i[i] = 0.5 * float(dp @ Q_p @ dp)
            dv = v_ee - self._v_des_candidates[i]
            costs_i[i] += 0.5 * float(dv @ Q_v @ dv)
            if Q_n > 0:
                n_err = n_rack - self._n_des_candidates[i]
                costs_i[i] += 0.5 * Q_n * float(n_err @ n_err)

        # 计算 softmin 权重 α_i
        weighted_neg_costs = -beta * costs_i + np.log(self._candidate_weights + 1e-30)
        max_wnc = np.max(weighted_neg_costs)
        exp_wnc = np.exp(weighted_neg_costs - max_wnc)
        alpha_i = exp_wnc / np.sum(exp_wnc)

        # 可解释性：缓存 softmin 权重和候选代价
        self._last_softmin_alphas = alpha_i.copy()
        self._last_softmin_costs = costs_i.copy()

        # 加权组合各候选的导数
        l_x = np.zeros(n_x)
        l_xx = np.zeros((n_x, n_x))

        # 法向量雅可比（对所有候选共用）
        J_omega = None
        J_n = None
        if Q_n > 0:
            J_omega = self.env.get_ee_jacr()
            nx, ny, nz = -n_rack[0], -n_rack[1], -n_rack[2]
            skew = np.array([
                [0, -nz, ny],
                [nz, 0, -nx],
                [-ny, nx, 0],
            ])
            J_n = np.zeros((3, n_x))
            J_n[:, :n_q] = skew @ J_omega

        for i in range(M):
            a_i = alpha_i[i]
            if a_i < 1e-12:
                continue

            # 位置导数
            dp = p_ee - self._p_ball_candidates[i]
            l_x[:n_q] += a_i * (J_p.T @ Q_p @ dp)
            l_xx[:n_q, :n_q] += a_i * (J_p.T @ Q_p @ J_p)

            # 速度导数
            dv = v_ee - self._v_des_candidates[i]
            l_x[n_q:] += a_i * (J_p.T @ Q_v @ dv)
            l_xx[n_q:, n_q:] += a_i * (J_p.T @ Q_v @ J_p)

            # 法向量导数
            if Q_n > 0 and J_n is not None:
                n_err = n_rack - self._n_des_candidates[i]
                l_x += a_i * Q_n * (J_n.T @ n_err)
                l_xx += a_i * Q_n * (J_n.T @ J_n)

        # Hessian 修正：加入 α_i 的一阶项（交叉项）
        # 对于 softmin，完整的 Hessian 包含 Σ_i α_i * (∂c_i/∂x)(∂c_i/∂x)^T
        # 减去 (Σ_i α_i * ∂c_i/∂x)(Σ_i α_i * ∂c_i/∂x)^T 乘以 β
        # 但这在 Gauss-Newton 近似中通常省略，因为 β 不太大时影响有限
        # 此处保留简化版本，仅用加权二阶项

        return l_x, l_xx

    def update_target(self, p_hit: np.ndarray, v_hit: np.ndarray, n_des: np.ndarray | None = None) -> None:
        """委托给 base_cost 更新终端目标。"""
        self.base_cost.update_target(p_hit, v_hit, n_des=n_des)

    def update_weights(self, Q_p_scale: float = 1.0, Q_v_scale: float = 1.0) -> None:
        """委托给 base_cost 更新权重。"""
        self.base_cost.update_weights(Q_p_scale, Q_v_scale)

    def set_q_des_traj(self, q_des_traj: np.ndarray | None, Q_joint: dict | None = None) -> None:
        """委托给 base_cost 设置关节轨迹。"""
        self.base_cost.set_q_des_traj(q_des_traj, Q_joint)

    def set_R_schedule(self, R_schedule: np.ndarray | None) -> None:
        """委托给 base_cost 设置 R 调度。"""
        self.base_cost.set_R_schedule(R_schedule)

    def update_hitting_tube(self, hitting_tube: HittingTube, horizon: int | None = None) -> None:
        """更新击球管道（用于 MPC 重规划）。

        Args:
            hitting_tube: 新的击球管道（k_candidates 应为 iLQR 相对步）。
            horizon: 新的规划地平线步数。None 表示保持原值。
        """
        self.hitting_tube = hitting_tube
        if horizon is not None:
            self.horizon = horizon
        self._rebuild_tube_maps(hitting_tube, self.horizon)

    def update_tube_params(self, ratio: float, anchor_alpha: float) -> None:
        """更新 tube 代价参数（用于渐进衰减策略）。

        Args:
            ratio: 新的有效 tube 代价比例 (0~1)。
            anchor_alpha: 终端锚定强度 (0~1)。0=全约束，1=沿d_ball完全自由。
        """
        self._current_ratio = max(0.0, min(1.0, ratio))
        self._anchor_alpha = max(0.0, min(1.0, anchor_alpha))
        # 用新比例重建 tube 权重
        self._rebuild_tube_maps(self.hitting_tube, self.horizon)

    def _compute_tube_cost_at_k(self, x: np.ndarray, k: int) -> float:
        """计算步骤 k 处的空间走廊 tube 代价值。

        v8 重新设计：轻量级走廊引导，仅保留 hinge loss 位置偏离代价。
        不再惩罚速度/法向量（这些由终端代价处理）。
        走廊半宽 = RACKET_RADIUS，走廊内零代价，走廊外温和二次惩罚。
        """
        self.env.set_arm_state(x)
        p_ee = self.env.get_ee_pos()

        dp = p_ee - self._p_ball_ref
        dp_perp = self._P_perp @ dp
        perp_dist = float(np.linalg.norm(dp_perp))
        margin = perp_dist - self.RACKET_RADIUS
        pos_err = max(0.0, margin)
        pos_cost = self._Q_p_tube * pos_err**2

        scale = self._tube_weight_scales.get(k, 1.0)
        return 0.5 * scale * pos_cost

    def _compute_tube_derivatives_at_k(
        self, x: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算步骤 k 处空间走廊 tube 代价的导数。

        v8 简化版：仅 hinge loss 位置偏离的梯度。
        margin > 0 时提供平滑梯度，走廊内梯度为零。
        """
        self.env.set_arm_state(x)
        n_x = self.env.NX
        n_q = self.env.NQ

        p_ee = self.env.get_ee_pos()
        J_p = self.env.get_ee_jacp()

        scale = self._tube_weight_scales.get(k, 1.0)

        l_x_tube = np.zeros(n_x)
        l_xx_tube = np.zeros((n_x, n_x))

        dp = p_ee - self._p_ball_ref
        dp_perp = self._P_perp @ dp
        perp_dist = float(np.linalg.norm(dp_perp))
        margin = perp_dist - self.RACKET_RADIUS

        if margin > 0.0 and perp_dist > 1e-8:
            dp_perp_hat = dp_perp / perp_dist
            grad_perp_q = dp_perp_hat @ self._P_perp @ J_p
            l_x_tube[:n_q] += self._Q_p_tube * margin * grad_perp_q
            l_xx_tube[:n_q, :n_q] += self._Q_p_tube * np.outer(grad_perp_q, grad_perp_q)

        self._last_tube_margins[k] = margin

        l_x_tube *= scale
        l_xx_tube *= scale

        return l_x_tube, l_xx_tube


class TubeOnlyCost(TubeHittingCostWrapper):
    """消融：仅 Tube 走廊运行代价 + 单点终端（无 softmin）。

    直接复用父类全部走廊逻辑，仅强制关闭 softmin 终端。
    终端代价退化为 base_cost 的单点精确击打目标。
    """

    def __init__(
        self,
        env: "RM65Env",
        base_cost: HittingCost,
        hitting_tube: HittingTube,
        horizon: int,
        config: TubeConfig,
    ) -> None:
        super().__init__(env, base_cost, hitting_tube, horizon, config)
        self._use_softmin = False


class SoftminOnlyCost(TubeHittingCostWrapper):
    """消融：仅 Softmin 多终端代价，无 Tube 走廊运行代价。

    继承父类的 softmin 终端逻辑（候选点搜索、加权代价、导数），
    但清空走廊步骤集合，使 running_cost / running_derivatives 不叠加走廊项。
    """

    def __init__(
        self,
        env: "RM65Env",
        base_cost: HittingCost,
        hitting_tube: HittingTube,
        horizon: int,
        config: TubeConfig,
    ) -> None:
        super().__init__(env, base_cost, hitting_tube, horizon, config)
        self._tube_steps = set()
        self._tube_weight_scales = {}

    def running_cost(self, x: np.ndarray, u: np.ndarray, k: int | None = None) -> float:
        return self.base_cost.running_cost(x, u, k)

    def running_derivatives(
        self, x: np.ndarray, u: np.ndarray, k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.base_cost.running_derivatives(x, u, k)
