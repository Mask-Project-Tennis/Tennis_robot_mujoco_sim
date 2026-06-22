"""击球点 refine 策略 — 可执行性后过滤。

在 tube 候选窗口内搜索关节裕度更充足的替代点，避免命中关节极限。
hysteresis 状态（hit_lock_active, last_p_hit）封装在策略实例内。

来源：V11 `_refine_hit_point`（行 1076-1236 嵌套闭包 → 方法 → 策略）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class RefineResult:
    """refine 结果。"""

    p_hit: NDArray[np.floating]   # 击球点（可能被替换）(3,)
    k_hit: int                     # 击球步数（可能被替换）
    log: str                       # 日志标签：feasible/locked/swapped/risk_kept/passthrough


@runtime_checkable
class HitPointRefiner(Protocol):
    """击球点 refine 接口 — 后过滤可执行性。"""

    def refine(
        self,
        p_hit: NDArray[np.floating],
        k_hit: int,
        remaining: int,
        env: Any,
        arm_state: NDArray[np.floating],
        robot_limits: Any,
    ) -> RefineResult:
        """对击球点做可执行性后过滤。

        Args:
            p_hit: 原始击球点 (3,)。
            k_hit: 原始击球步数。
            remaining: 剩余总步数。
            env: 规划环境（solve_ik/predict_ball_trajectory/get_ball_pos/get_ball_vel）。
            arm_state: 臂状态 [q(6), qdot(6)]（提供 IK 初始猜测）。
            robot_limits: 关节限位（q_lower/q_upper）。

        Returns:
            RefineResult（p_hit 可能被替换为更安全的候选）。
        """
        ...

    def reset(self) -> None:
        """重置 hysteresis 状态（hit_lock_active/last_p_hit）。"""
        ...


class HybridRefiner:
    """混合 refine 策略 — 与 V11 `_refine_hit_point` 完全一致。

    三阶段过滤：
    1. 防抖锁定：k_hit ≤ hit_lock_threshold 时不再换点
    2. 裕度检查：IK 求解当前点，计算关节双边裕度
       - 安全（margin_min ≥ hard_margin 且 j1 ≥ j1_warn）→ "feasible"
       - 危险 → 进入候选搜索
    3. 候选搜索（hysteresis：新点需显著优于旧点）：
       - 策略1：位置微调（j1 偏移，保持时间不变）
       - 策略2：tube 窗口搜索（改变时间点）
       - 找到更优候选 → "swapped"，否则 → "risk_kept"
    """

    def __init__(
        self,
        shoulder_pos: NDArray[np.floating],
        workspace_radius: float,
        hit_lock_threshold: int = 60,
        hard_margin_deg: float = 2.0,
        warn_margin_deg: float = 5.0,
        j1_warn_margin_deg: float = 8.0,
        window_half_steps: int = 15,
    ) -> None:
        """初始化 refine 参数。

        Args:
            shoulder_pos: 肩部位置 (3,)（工作空间中心参考）。
            workspace_radius: 工作空间半径（m）。
            hit_lock_threshold: 防抖锁定阈值（k_hit ≤ 此值时锁定）。
            hard_margin_deg: 硬裕度阈值（°），低于此值视为高风险。
            warn_margin_deg: 警告裕度阈值（°）。
            j1_warn_margin_deg: 关节1 警告裕度阈值（°）。
            window_half_steps: tube 窗口搜索半宽（步数）。
        """
        self._shoulder_pos: NDArray[np.floating] = shoulder_pos
        self._workspace_radius: float = workspace_radius
        self._hit_lock_threshold: int = hit_lock_threshold
        self._hard_margin_deg: float = hard_margin_deg
        self._warn_margin_deg: float = warn_margin_deg
        self._j1_warn_margin_deg: float = j1_warn_margin_deg
        self._window_half_steps: int = window_half_steps

        # ── hysteresis 状态（从 MPCController 移入）──
        self._hit_lock_active: bool = False
        self._last_p_hit: NDArray[np.floating] | None = None

    def refine(
        self,
        p_hit: NDArray[np.floating],
        k_hit: int,
        remaining: int,
        env: Any,
        arm_state: NDArray[np.floating],
        robot_limits: Any,
    ) -> RefineResult:
        """击球点可执行性后过滤（V11 行 1076-1236）。

        Args:
            p_hit: 原始击球点 (3,)。
            k_hit: 原始击球步数。
            remaining: 剩余步数。
            env: 规划环境。
            arm_state: 臂状态（IK 初始猜测）。
            robot_limits: 关节限位。

        Returns:
            RefineResult（p_hit/k_hit 可能被替换）。
        """
        # 防抖锁定：末段不再换点
        if k_hit <= self._hit_lock_threshold:
            if not self._hit_lock_active:
                self._hit_lock_active = True
                logger.info(
                    "[HIT_LOCK] k_hit=%d ≤ %d, 锁定击球点不再替换",
                    k_hit, self._hit_lock_threshold,
                )
            return RefineResult(p_hit=p_hit.copy(), k_hit=k_hit, log="locked")

        # 快速 IK 检查当前点的双边裕度
        NQ: int = env.NQ
        q_ik = env.solve_ik(
            p_hit, q_init=arm_state[:NQ], max_iter=50, eps=1e-2,
        )
        margin_lower_deg = (q_ik - robot_limits.q_lower) * 180.0 / np.pi
        margin_upper_deg = (robot_limits.q_upper - q_ik) * 180.0 / np.pi
        margin_min_deg = float(np.min(np.minimum(margin_lower_deg, margin_upper_deg)))
        margin_j1_deg = float(min(margin_lower_deg[1], margin_upper_deg[1]))

        high_risk = margin_min_deg < self._hard_margin_deg
        j1_near = margin_j1_deg < self._j1_warn_margin_deg
        medium_risk = margin_min_deg < self._warn_margin_deg

        if not high_risk and not j1_near:
            if medium_risk:
                logger.info(
                    "[HIT_KEEP] p=%s min_margin=%.1f° j1=%.1f° → feasible (medium risk)",
                    np.round(p_hit, 3), margin_min_deg, margin_j1_deg,
                )
            return RefineResult(p_hit=p_hit.copy(), k_hit=k_hit, log="feasible")

        logger.warning(
            "[HIT_RISK] p=%s k=%d min_margin=%.1f° j1=%.1f° → searching alternatives",
            np.round(p_hit, 3), k_hit, margin_min_deg, margin_j1_deg,
        )

        best_candidate: tuple | None = None
        best_score = -1e9

        # 策略1：微调位置偏移（保持时间不变）
        if j1_near:
            best_candidate, best_score = self._search_j1_offset(
                p_hit, k_hit, margin_min_deg, margin_j1_deg,
                margin_lower_deg, env, arm_state, robot_limits, best_score,
            )

        # 策略2：tube 窗口搜索（改变时间点）
        best_candidate, best_score = self._search_tube_window(
            p_hit, k_hit, remaining, margin_min_deg,
            env, arm_state, robot_limits, best_candidate, best_score,
        )

        # hysteresis：新点需显著优于旧点
        if best_candidate is not None:
            p_new, k_new, m_min_new, m_j1_new = best_candidate
            score_original = 2.0 * margin_min_deg + 3.0 * margin_j1_deg
            if best_score > score_original + 10.0:
                logger.warning(
                    "[HIT_SWAP] k %d→%d, min_margin %.1f°→%.1f°, j1 %.1f°→%.1f°",
                    k_hit, k_new, margin_min_deg, m_min_new, margin_j1_deg, m_j1_new,
                )
                self._last_p_hit = p_new.copy()
                return RefineResult(p_hit=p_new.copy(), k_hit=k_new, log="swapped")

        logger.warning(
            "[HIT_RISK] min_margin=%.1f° j1=%.1f°, no safer candidate found",
            margin_min_deg, margin_j1_deg,
        )
        return RefineResult(p_hit=p_hit.copy(), k_hit=k_hit, log="risk_kept")

    def _search_j1_offset(
        self,
        p_hit: NDArray[np.floating],
        k_hit: int,
        margin_min_deg: float,
        margin_j1_deg: float,
        margin_lower_deg: NDArray[np.floating],
        env: Any,
        arm_state: NDArray[np.floating],
        robot_limits: Any,
        best_score: float,
    ) -> tuple[tuple | None, float]:
        """策略1：j1 偏移搜索（V11 行 1225-1254）。

        Args:
            p_hit: 原始击球点。
            k_hit: 原始击球步数。
            margin_min_deg: 原始最小裕度。
            margin_j1_deg: 原始 j1 裕度。
            margin_lower_deg: 原始下边裕度向量。
            env: 规划环境。
            arm_state: 臂状态。
            robot_limits: 关节限位。
            best_score: 当前最佳分数。

        Returns:
            (best_candidate, best_score)。
        """
        NQ: int = env.NQ
        best_candidate: tuple | None = None
        j1_dir = 1.0 if margin_j1_deg == margin_lower_deg[1] else -1.0
        for offset_cm in [3, 5, 8, 12]:
            offset_m = offset_cm / 100.0
            p_shifted = p_hit.copy()
            p_shifted[1] += j1_dir * offset_m
            dist_s = float(np.linalg.norm(p_shifted - self._shoulder_pos))
            if dist_s > self._workspace_radius or p_shifted[2] < 0.3:
                continue
            q_s = env.solve_ik(
                p_shifted, q_init=arm_state[:NQ], max_iter=30, eps=2e-2,
            )
            m_low_s = (q_s - robot_limits.q_lower) * 180.0 / np.pi
            m_up_s = (robot_limits.q_upper - q_s) * 180.0 / np.pi
            m_min_s = float(np.min(np.minimum(m_low_s, m_up_s)))
            m_j1_s = float(min(m_low_s[1], m_up_s[1]))
            if m_min_s < margin_min_deg - 0.5:
                continue
            if m_j1_s < self._j1_warn_margin_deg:
                continue
            score_s = (
                2.0 * m_min_s
                + 3.0 * m_j1_s
                - 50.0 * float(np.linalg.norm(p_shifted - p_hit))
            )
            if score_s > best_score:
                best_score = score_s
                best_candidate = (p_shifted.copy(), k_hit, m_min_s, m_j1_s)
        return best_candidate, best_score

    def _search_tube_window(
        self,
        p_hit: NDArray[np.floating],
        k_hit: int,
        remaining: int,
        margin_min_deg: float,
        env: Any,
        arm_state: NDArray[np.floating],
        robot_limits: Any,
        best_candidate: tuple | None,
        best_score: float,
    ) -> tuple[tuple | None, float]:
        """策略2：tube 窗口搜索（V11 行 1256-1298）。

        Args:
            p_hit: 原始击球点。
            k_hit: 原始击球步数。
            remaining: 剩余步数。
            margin_min_deg: 原始最小裕度。
            env: 规划环境。
            arm_state: 臂状态。
            robot_limits: 关节限位。
            best_candidate: 当前最佳候选。
            best_score: 当前最佳分数。

        Returns:
            (best_candidate, best_score)。
        """
        NQ: int = env.NQ
        ball_positions_pred, _ = env.predict_ball_trajectory(
            env.get_ball_pos(), env.get_ball_vel(),
            min(remaining + 30, 300),
        )
        k_min = max(1, k_hit - self._window_half_steps)
        k_max = min(len(ball_positions_pred), k_hit + self._window_half_steps)

        for k_cand in range(k_min, k_max + 1):
            if k_cand == k_hit:
                continue
            p_cand = ball_positions_pred[k_cand - 1]
            dist_cand = float(np.linalg.norm(p_cand - self._shoulder_pos))
            if dist_cand > self._workspace_radius * 1.1 or p_cand[2] < 0.3:
                continue
            q_cand = env.solve_ik(
                p_cand, q_init=arm_state[:NQ], max_iter=30, eps=2e-2,
            )
            m_low = (q_cand - robot_limits.q_lower) * 180.0 / np.pi
            m_up = (robot_limits.q_upper - q_cand) * 180.0 / np.pi
            m_min = float(np.min(np.minimum(m_low, m_up)))
            m_j1 = float(min(m_low[1], m_up[1]))
            if m_min < margin_min_deg - 0.5:
                continue
            y_risk = max(0.0, (self._shoulder_pos[1] - 0.40) - p_cand[1])
            score = (
                2.0 * m_min
                + 3.0 * m_j1
                - 1.0 * abs(k_cand - k_hit)
                - 30.0 * float(np.linalg.norm(p_cand - p_hit))
                - 10.0 * y_risk
            )
            if score > best_score:
                best_score = score
                best_candidate = (p_cand.copy(), k_cand, m_min, m_j1)
        return best_candidate, best_score

    def reset(self) -> None:
        """重置 hysteresis 状态。"""
        self._hit_lock_active = False
        self._last_p_hit = None
