"""MPC 规划控制器 — 可组合的规划模块。

封装完整规划生命周期：
击球点搜索 → refine 后过滤 → Tube 构建 → iLQR 求解 → 阶段调度 → 随挥目标计算。

不含：球感知、物理步进、碰撞检测、可视化。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.ilqt.async_replanner import AsyncReplanner, PlanRequest
from src.ilqt.planning_env import PlanningEnv
from src.ilqt.replan_config import ReplanConfig
from src.ilqt.replan_core import do_replan
from src.ilqt.strategies.direction import DirectionPolicy, ReflectDirection
from src.ilqt.strategies.follow_through import (
    FollowContext,
    FollowThroughPolicy,
    PlannedFollowThrough,
)
from src.ilqt.strategies.hit_point_refiner import (
    HitPointRefiner,
    HybridRefiner,
)
from src.ilqt.strategies.phase_schedule import DefaultPhaseSchedule, PhaseSchedule
from src.ilqt.strategies.replan_mode import (
    AsyncReplanMode,
    ReplanMode,
    SyncReplanMode,
)
from src.ilqt.tube_types import ReplanState, TubeConfig
from src.ilqt.strategy_config import StrategyConfig
from src.real.runner_factory import build_robot_limits, build_solver

logger = logging.getLogger(__name__)


@dataclass
class MPCConfig:
    """MPC 控制器配置 — 控制规划行为，可版本化。"""

    # ── 版本与模式 ──
    version: str = "v11"
    is_position_mode: bool = False
    ablation_mode: str = "full"           # full / tube_only / softmin_only / none

    # ── 时间参数 ──
    dt: float = 0.005
    total_horizon: int = 250
    fixed_horizon: int = 120
    replan_interval: int = 20

    # ── 迭代参数 ──
    max_iter_per_plan: int = 10
    first_plan_iters: int = 30
    near_plan_iters: int = 5
    near_threshold: int = 80

    # ── 代价参数 ──
    Q_p_scale_far: float = 5.0
    Q_v_scale_far: float = 3.0
    Q_p_scale_near: float = 8.0
    Q_v_scale_near: float = 120.0
    R: float = 0.0001
    normal_weight: float = 500000.0
    racket_speed: float = 1.8
    target_speed: float = 1.8
    Q_tcp_soft: float = 5000.0
    Q_qdot_limit: float = 1000.0

    # ── Tube 参数 ──
    tube_cfg: TubeConfig = field(default_factory=TubeConfig)
    softmin_beta: float = 5.0

    # ── 随挥参数 ──
    follow_through_length: float = 0.0
    follow_through_steps: int = 0
    follow_through_v_terminal: float = 0.3
    follow_trigger: str = "planned"       # planned / contact
    no_follow_through: bool = False

    # ── 后摆参数 ──
    use_backswing: bool = False
    backswing_offset: float = 0.6
    backswing_ratio: float = 0.35

    # ── 安全参数 ──
    max_tcp_speed: float = 1.8
    terminal_exempt_steps: int = 20
    dq_max_fraction: float = 0.5

    # ── 几何参数 ──
    shoulder_pos: np.ndarray = field(
        default_factory=lambda: np.array([-0.1, -0.22693, 1.302645])
    )
    workspace_radius: float = 0.90

    # ── 扰动参数（实验用）──
    time_perturb_s: float = 0.0
    space_perturb_m: float = 0.0
    perturb_alpha_min: float = 0.0

    # ── R 退火 ──
    use_r_decay: bool = False
    r_decay_ratio: float = 0.40

    # ── joint5 固定 ──
    fix_joint5_angle: float | None = None
    normal_flip: bool = False

    # ── 阶段权重 ──
    smooth_far: dict = field(
        default_factory=lambda: {"Q_qdot_mult": 0.01, "Q_qddot_mult": 0.01, "Q_du_mult": 0.1}
    )
    smooth_mid: dict = field(
        default_factory=lambda: {"Q_qdot_mult": 0.1, "Q_qddot_mult": 0.1, "Q_du_mult": 0.2}
    )
    smooth_near: dict = field(
        default_factory=lambda: {"Q_qdot_mult": 1.0, "Q_qddot_mult": 1.0, "Q_du_mult": 0.5}
    )

    # ── 代价基底（v12: 对齐 V11 base_cost_fn 的 Q_p/Q_v基底；None 表示走 do_replan 默认 eye(3)）──
    Q_p_base: np.ndarray | None = field(
        default_factory=lambda: np.array([100000.0, 100000.0, 100000.0])
    )
    Q_v_base: np.ndarray | None = field(
        default_factory=lambda: np.array([400.0, 400.0, 400.0])
    )
    Q_qdot_base: float = 0.001
    Q_qddot_base: float = 0.0005
    Q_du_base: float = 0.001

    # ── 远段阈值（V11 far_threshold：k_hit > 此值时用 JT 控制）──
    far_threshold: int = 50

    # ── 策略内部参数（原硬编码在策略类中）──
    follow_kp: float = 200.0             # 随挥 PD 比例增益
    follow_kd: float = 20.0              # 随挥 PD 微分增益
    hit_lock_threshold: int = 60         # 击球点防抖锁定步数
    hard_margin_deg: float = 2.0         # IK 硬裕度（度）
    warn_margin_deg: float = 5.0         # IK 警告裕度（度）
    j1_warn_margin_deg: float = 8.0      # 关节1 警告裕度（度）
    refiner_window_half: int = 15        # Refiner 搜索窗口半宽（步）

    # ── 异步重规划 ──
    async_mode: bool = False


@dataclass
class MPCStepResult:
    """MPC 一步输出结果。"""

    u_cmd: np.ndarray              # (6,) 控制指令（力矩或 q_desired）
    phase: str                     # "far" / "mid" / "near" / "follow_through" / "done"
    k_hit: int                     # 距击球步数（剩余）
    ball_unreachable: bool = False # 球是否不可达
    replanned: bool = False        # 本步是否触发了重规划
    p_hit: np.ndarray | None = None       # (3,) 当前击球点
    n_des: np.ndarray | None = None       # (3,) 期望拍面法向量
    info: dict = field(default_factory=dict)  # 额外信息


class MPCController:
    """MPC 规划控制器 — 可组合的规划模块。

    封装完整规划生命周期：
    击球点搜索 → refine 后过滤 → Tube 构建 → iLQR 求解 → 阶段调度 → 随挥目标计算。

    不包含：球感知、物理步进、碰撞检测、可视化。

    设计要点：
    - 同步/异步统一调用 ``do_replan``（消除 V11 中 ~325 行同步路径重复）
    - refine_hit_point 从 V11 嵌套闭包提取为实例方法，闭包变量变实例属性
    - 随挥触发和控制计算在 ``step()`` 内部完成
    """

    def __init__(
        self,
        env: PlanningEnv,
        config: MPCConfig,
        strategies: "StrategyConfig | None" = None,
    ) -> None:
        """初始化 solver/limits/replanner/state 构建。

        Args:
            env: 规划环境（PlanningEnv 或 RM65Env），用于主线程规划计算。
            config: MPC 配置（控制规划行为）。
            strategies: 可选策略注入容器。None → 全部使用 MPCConfig 驱动的默认实现。
        """
        self._env = env
        self._config = config
        self._NU: int = env.NU
        self._NQ: int = env.NQ

        # ── 构建 RobotLimits + solver（复用 runner_factory）──
        self._robot_limits = build_robot_limits(env)
        self._solver: Any = build_solver()

        # ── AsyncReplanner（独立 env_plan）──
        model_path = Path(__file__).resolve().parent.parent / "robot" / "rm65_model.xml"
        self._replanner = AsyncReplanner(
            env, do_replan, config={}, state=None, model_path=model_path,
        )
        self._replan_cfg: ReplanConfig | None = None
        self._replan_state: ReplanState | None = None

        # ── 规划状态变量（V11 行 1047-1067）──
        self._step_count: int = 0
        self._U_buffer: np.ndarray = np.zeros((0, self._NU))
        self._buffer_idx: int = 0
        self._k_hit: int = 0
        self._p_hit: np.ndarray = np.zeros(3)
        self._v_ball_hit: np.ndarray = np.zeros(3)
        self._n_des: np.ndarray = np.zeros(3)
        self._d_hat: np.ndarray = np.zeros(3)
        self._v_hit_desired: np.ndarray = np.zeros(3)
        self._d_follow: np.ndarray = np.zeros(3)
        self._U_prev: np.ndarray = np.zeros((0, self._NU))
        self._x_current: np.ndarray = np.zeros(env.NX)

        # ── 随挥状态（V11 行 1054-1061）──
        self._ball_was_hit: bool = False

        # ── 结束标志 ──
        self._ball_unreachable: bool = False
        self._done: bool = False

        # ── 异步状态 ──
        self._mpc_horizon: int = config.total_horizon

        # ── 策略（A4/A1/A2 提取 — 支持外部注入）──
        strat = strategies  # 别名缩短
        self._phase_schedule: PhaseSchedule = (
            strat.phase_schedule if strat and strat.phase_schedule
            else DefaultPhaseSchedule(
                far_threshold=config.far_threshold,
                near_threshold=config.near_threshold,
            )
        )
        self._direction_policy: DirectionPolicy = (
            strat.direction_policy if strat and strat.direction_policy
            else ReflectDirection(target_speed=config.target_speed)
        )
        follow_steps = 0 if config.no_follow_through else config.follow_through_steps
        self._follow_through: FollowThroughPolicy = (
            strat.follow_through if strat and strat.follow_through
            else PlannedFollowThrough(
                follow_through_steps=follow_steps,
                follow_trigger=config.follow_trigger,
                dt=config.dt,
                is_position_mode=config.is_position_mode,
                NQ=self._NQ,
                NU=self._NU,
                kp=config.follow_kp,
                kd=config.follow_kd,
            )
        )
        self._refiner: HitPointRefiner = (
            strat.hit_point_refiner if strat and strat.hit_point_refiner
            else HybridRefiner(
                shoulder_pos=config.shoulder_pos,
                workspace_radius=config.workspace_radius,
                hit_lock_threshold=config.hit_lock_threshold,
                hard_margin_deg=config.hard_margin_deg,
                warn_margin_deg=config.warn_margin_deg,
                j1_warn_margin_deg=config.j1_warn_margin_deg,
                window_half_steps=config.refiner_window_half,
            )
        )

        # ── 重规划模式（A3 提取：同步/异步统一）──
        if config.async_mode:
            self._replan_mode: ReplanMode = AsyncReplanMode(self._replanner)
        else:
            self._replan_mode = SyncReplanMode(self._sync_replan_fn)

    # ──────────────────────────────────────────────────────────────────
    # 公有接口
    # ──────────────────────────────────────────────────────────────────

    def _sync_replan_fn(self, request: PlanRequest) -> Any:
        """同步重规划闭包 — 供 SyncReplanMode 调用。

        Args:
            request: 规划请求。

        Returns:
            PlanResult（do_replan 结果）。
        """
        assert self._replanner.env_plan is not None
        assert self._replan_cfg is not None and self._replan_state is not None
        return do_replan(
            request, self._replanner.env_plan,
            self._replan_state, self._replan_cfg,
        )

    def start(self, ball_pos: np.ndarray, ball_vel: np.ndarray,
              arm_state: np.ndarray) -> None:
        """首次同步规划。

        计算击球方向 → 构建 replan_cfg → 首次 do_replan → 初始化 buffer。

        Args:
            ball_pos: 球当前位置 (3,)。
            ball_vel: 球当前速度 (3,)。
            arm_state: 臂状态 [q(6), qdot(6)] (12,)。
        """
        self._x_current = arm_state.copy()

        # 1. 计算击球方向（V11 行 819-838）→ DirectionPolicy
        direction = self._direction_policy.compute(ball_vel)
        self._d_hat = direction.d_hat.copy()
        self._v_hit_desired = direction.v_hit_desired.copy()
        self._d_follow = direction.d_follow.copy()

        # 2. 构建 replan_cfg（B2: ReplanConfig.from_mpc_config 类型安全工厂）
        self._replan_cfg = self._build_replan_cfg(self._d_hat, self._v_hit_desired)

        # 3. 构建 ReplanState（V11 行 1254-1261）
        self._replan_state = ReplanState(
            k_hit_new=self._config.total_horizon,
            p_hit_new=ball_pos.copy(),
            v_ball_hit_new=ball_vel.copy(),
            current_n_des=self._d_hat.copy(),
            U_prev=np.zeros((0, self._NU)),
            is_first_plan=True,
        )
        self._replanner._state = self._replan_state
        self._replanner._config = self._replan_cfg

        # 4. 启动 AsyncReplanner + 确保 env_plan 就绪（V11 行 1315-1318）
        self._replanner.start()
        self._replanner._ensure_env_plan()

        # 5. 首次同步 do_replan（V11 行 1349-1364）
        first_request = PlanRequest(
            x_current=arm_state.copy(),
            ball_pos=ball_pos.copy(),
            ball_vel=ball_vel.copy(),
            step=0,
            k_hit_current=self._config.total_horizon,
            U_prev=np.zeros((0, self._NU)),
            p_hit_current=ball_pos.copy(),
            v_hit_desired=self._v_hit_desired.copy(),
            n_des_current=self._d_hat.copy(),
            is_first_plan=True,
        )
        assert self._replanner.env_plan is not None, "env_plan 未就绪"
        t_replan_start = time.perf_counter()
        result = do_replan(
            first_request, self._replanner.env_plan,
            self._replan_state, self._replan_cfg,
        )
        t_replan_ms = (time.perf_counter() - t_replan_start) * 1000
        logger.info(
            "REPLAN step=0 k_hit=%d iters=%d horizon=%d t=%.0fms",
            result.k_hit_new, result.iters_plan, result.horizon_plan, t_replan_ms,
        )

        # 6. 球不可达处理
        if result.ball_unreachable:
            self._ball_unreachable = True
            self._done = True
            logger.warning("MPCController.start: 球不可达, ball_pos=%s", ball_pos)
            return

        # 7. 更新规划状态（V11 行 1373-1392）
        self._replan_state.is_first_plan = False
        self._replan_state.k_hit_new = result.k_hit_new
        self._replan_state.p_hit_new = result.p_hit_new.copy()
        self._replan_state.v_ball_hit_new = result.v_ball_hit_new.copy()
        self._replan_state.current_n_des = result.n_des_new.copy()
        self._replan_state.U_prev = result.U_prev.copy()

        self._k_hit = result.k_hit_new
        self._p_hit = result.p_hit_new.copy()
        self._v_ball_hit = result.v_ball_hit_new.copy()
        self._n_des = result.n_des_new.copy()
        self._U_prev = result.U_prev.copy()

        # 8. refine_hit_point 初始过滤 → HitPointRefiner（V11 行 1590 调用点）
        self._env.set_ball_state(ball_pos.copy(), ball_vel.copy())
        refine_result = self._refiner.refine(
            p_hit=self._p_hit, k_hit=self._k_hit,
            remaining=self._config.total_horizon,
            env=self._env, arm_state=self._x_current,
            robot_limits=self._robot_limits,
        )
        self._p_hit = refine_result.p_hit
        self._k_hit = refine_result.k_hit
        self._replan_state.p_hit_new = self._p_hit.copy()
        self._replan_state.k_hit_new = self._k_hit
        if refine_result.log == "swapped":
            logger.info(
                "MPCController.start refine: p_hit 修正, k_hit=%d, refine=%s",
                self._k_hit, refine_result.log,
            )

        # 9. 初始化 buffer（V11 行 1381-1382）
        self._U_buffer = result.U_buffer.copy()
        self._buffer_idx = 0
        self._step_count = 0
        self._mpc_horizon = self._config.total_horizon

        # 10. 更新 replan_cfg 的 k_hit_total（B2: ReplanConfig 属性访问替代 dict 下标）
        assert self._replan_cfg is not None
        self._replan_cfg.k_hit_total = self._k_hit

        # 11. 异步模式首次提交 → ReplanMode.submit（V11 行 1394-1409）
        if self._config.async_mode:
            async_request = self._build_replan_request(
                ball_pos, ball_vel, arm_state, step=0,
            )
            self._replan_mode.submit(async_request)

        logger.info(
            "MPCController.start 完成: k_hit=%d, p_hit=%s, buffer=%d",
            self._k_hit, np.round(self._p_hit, 3), len(self._U_buffer),
        )

    def step(self, ball_pos: np.ndarray, ball_vel: np.ndarray,
             arm_state: np.ndarray) -> MPCStepResult:
        """一步 MPC 循环。

        1. 检查随挥触发（step_count >= mpc_horizon 或 k_hit <= 1）
        2. 随挥阶段 → 计算随挥控制（PD/IK），返回 phase="follow_through"
        3. MPC 阶段 → 异步/同步重规划 + buffer 提取
        4. 返回 MPCStepResult

        Args:
            ball_pos: 球当前位置 (3,)。
            ball_vel: 球当前速度 (3,)。
            arm_state: 臂状态 [q(6), qdot(6)] (12,)。

        Returns:
            MPCStepResult（含 u_cmd, phase, k_hit 等）。
        """
        sc = self._step_count
        self._step_count += 1
        self._x_current = arm_state.copy()

        # ── 已结束：返回 hold 控制 ──
        if self._done:
            u_hold = self._hold_control(arm_state)
            return MPCStepResult(
                u_cmd=u_hold, phase="done", k_hit=0,
                ball_unreachable=self._ball_unreachable,
            )

        # ── 1. 检查随挥触发 → FollowThroughPolicy ──
        follow_ctx = FollowContext(
            step_count=sc,
            mpc_horizon=self._mpc_horizon,
            k_hit=self._k_hit,
            arm_state=arm_state,
            ball_pos=ball_pos,
            d_follow=self._d_follow,
            v_hit_desired=self._v_hit_desired,
            env=self._env,
        )
        if not self._follow_through.is_active:
            if self._follow_through.should_trigger(follow_ctx):
                # 触发当步即进入随挥（step_in_follow=0）
                u_cmd = self._follow_through.compute_control(follow_ctx, 0)
                return MPCStepResult(
                    u_cmd=u_cmd, phase="follow_through", k_hit=0,
                    p_hit=self._p_hit.copy(), n_des=self._n_des.copy(),
                    info={"follow_step": 0},
                )
        else:
            # ── 已在随挥阶段 ──
            step_in_follow = sc - self._follow_through.follow_through_start
            if not self._follow_through.is_done(step_in_follow, follow_ctx):
                u_cmd = self._follow_through.compute_control(follow_ctx, step_in_follow)
                return MPCStepResult(
                    u_cmd=u_cmd, phase="follow_through", k_hit=0,
                    p_hit=self._p_hit.copy(), n_des=self._n_des.copy(),
                    info={"follow_step": step_in_follow},
                )
            else:
                # 随挥完成
                self._done = True
                u_hold = self._hold_control(arm_state)
                return MPCStepResult(
                    u_cmd=u_hold, phase="done", k_hit=0,
                )

        # ── 2. MPC 阶段（A3 统一：ReplanMode 调度）──
        need_replan = (
            (sc % self._config.replan_interval == 0)
            or (self._buffer_idx >= len(self._U_buffer))
        ) and sc < self._mpc_horizon

        replanned = False

        # 2a. 轮询异步结果（之前提交的规划完成）
        if self._config.async_mode and self._replan_mode.has_result():
            replanned = self._apply_async_result(
                self._replan_mode.get_result(), sc,
            )

        # 2b. 提交新请求（同步：立即获取结果；异步：提交到后台）
        can_submit = need_replan and not self._replan_mode.is_busy()
        if self._config.async_mode:
            can_submit = can_submit and sc > 0  # async: start() 已首次提交
        if can_submit:
            request = self._build_replan_request(ball_pos, ball_vel, arm_state, sc)
            t_replan_start = time.perf_counter()
            self._replan_mode.submit(request)
            t_replan_ms = (time.perf_counter() - t_replan_start) * 1000
            if not self._config.async_mode:
                # 同步：submit 后结果立即可用
                result = self._replan_mode.get_result()
                if result is not None:
                    logger.debug(
                        "REPLAN step=%d k_hit=%d iters=%d horizon=%d t=%.0fms",
                        sc, result.k_hit_new, result.iters_plan,
                        result.horizon_plan, t_replan_ms,
                    )
                unreachable = self._apply_sync_result(
                    result, sc, ball_pos, ball_vel,
                )
                if unreachable:
                    u_hold = self._hold_control(arm_state)
                    return MPCStepResult(
                        u_cmd=u_hold, phase="done", k_hit=0,
                        ball_unreachable=True,
                    )
                replanned = True

        # ── 3. 提取 U_buffer（V11 行 1889-1912）──
        if self._buffer_idx < len(self._U_buffer):
            u_cmd = self._U_buffer[self._buffer_idx].copy()
            self._buffer_idx += 1
        else:
            u_cmd = self._compute_buffer_fallback(arm_state)

        # ── 4. 确定阶段 → PhaseSchedule ──
        phase = self._phase_schedule.classify(self._k_hit)

        return MPCStepResult(
            u_cmd=u_cmd, phase=phase, k_hit=self._k_hit,
            replanned=replanned,
            p_hit=self._p_hit.copy(), n_des=self._n_des.copy(),
        )

    def stop(self) -> None:
        """停止 AsyncReplanner 后台线程。"""
        self._replanner.stop()

    @property
    def done(self) -> bool:
        """是否结束（球不可达/随挥完成）。"""
        return self._done

    @property
    def ball_unreachable(self) -> bool:
        """球是否不可达。"""
        return self._ball_unreachable

    # ──────────────────────────────────────────────────────────────────
    # 私有方法：配置
    # ──────────────────────────────────────────────────────────────────

    def _build_replan_cfg(
        self, d_hat: np.ndarray, v_hit_desired: np.ndarray,
    ) -> ReplanConfig:
        """构建 do_replan 所需类型安全配置。

        复用 ``ReplanConfig.from_mpc_config`` 工厂（消除 80 行字段翻译），
        替代旧的 runner_factory.build_replan_cfg() + MPCConfig.update() dict 路径。
        do_replan 入口兼容层会自动 to_dict() 以支持旧 cfg["..."] 访问。

        Args:
            d_hat: 击球方向单位向量。
            v_hit_desired: 期望末端速度。

        Returns:
            ReplanConfig（可直接传给 do_replan）。
        """
        return ReplanConfig.from_mpc_config(
            self._config,
            robot_limits=self._robot_limits,
            solver=self._solver,
            d_hat=d_hat,
            v_hit_desired=v_hit_desired,
        )

    # ──────────────────────────────────────────────────────────────────
    # 私有方法：重规划（A3 统一辅助）
    # ──────────────────────────────────────────────────────────────────

    def _build_replan_request(
        self, ball_pos: np.ndarray, ball_vel: np.ndarray,
        arm_state: np.ndarray, step: int,
        is_first_plan: bool = False,
    ) -> PlanRequest:
        """构建重规划请求（消除 start/sync/async 三处重复）。

        Args:
            ball_pos: 球当前位置 (3,)。
            ball_vel: 球当前速度 (3,)。
            arm_state: 臂状态 [q(6), qdot(6)]。
            step: 当前步索引。
            is_first_plan: 是否首次规划。

        Returns:
            PlanRequest（可直接传给 do_replan 或 ReplanMode.submit）。
        """
        return PlanRequest(
            x_current=arm_state.copy(),
            ball_pos=ball_pos.copy(),
            ball_vel=ball_vel.copy(),
            step=step,
            k_hit_current=self._k_hit,
            U_prev=self._U_prev.copy() if len(self._U_prev) > 0 else np.zeros((0, self._NU)),
            p_hit_current=self._p_hit.copy(),
            v_hit_desired=self._v_hit_desired.copy(),
            n_des_current=self._n_des.copy(),
            is_first_plan=is_first_plan,
        )

    def _apply_plan_fields(self, result: Any) -> None:
        """从 PlanResult 更新规划字段 + 同步 replan_state（共享后处理）。

        更新：_p_hit, _v_ball_hit, _n_des, _U_prev 及对应的 replan_state 字段。
        注意：_k_hit 由调用方设置（sync/async 计算方式不同）。

        Args:
            result: PlanResult（do_replan 或 AsyncReplanner 的结果）。
        """
        assert self._replan_state is not None
        self._p_hit = result.p_hit_new.copy()
        self._v_ball_hit = result.v_ball_hit_new.copy()
        self._n_des = result.n_des_new.copy()
        self._U_prev = result.U_prev.copy()
        self._replan_state.k_hit_new = self._k_hit
        self._replan_state.p_hit_new = self._p_hit.copy()
        self._replan_state.v_ball_hit_new = self._v_ball_hit.copy()
        self._replan_state.current_n_des = self._n_des.copy()
        self._replan_state.U_prev = self._U_prev.copy()

    def _apply_sync_result(
        self, result: Any, step: int,
        ball_pos: np.ndarray, ball_vel: np.ndarray,
    ) -> bool:
        """应用同步重规划结果（含 refine）。

        Args:
            result: PlanResult（SyncReplanMode.get_result）。
            step: 当前步索引。
            ball_pos: 球当前位置（refine 用）。
            ball_vel: 球当前速度（refine 用）。

        Returns:
            True 表示球不可达。
        """
        if result is None:
            logger.warning("MPCController.step %d: 同步重规划返回 None", step)
            return False
        if result.ball_unreachable:
            self._ball_unreachable = True
            self._done = True
            logger.warning("MPCController.step %d: 球不可达", step)
            return True

        self._k_hit = result.k_hit_new
        self._apply_plan_fields(result)

        # refine_hit_point 后过滤 → HitPointRefiner（V11 行 1590）
        self._env.set_ball_state(ball_pos.copy(), ball_vel.copy())
        remaining = max(self._config.total_horizon - step, 5)
        refine_result = self._refiner.refine(
            p_hit=self._p_hit, k_hit=self._k_hit, remaining=remaining,
            env=self._env, arm_state=self._x_current,
            robot_limits=self._robot_limits,
        )
        self._p_hit = refine_result.p_hit
        self._k_hit = refine_result.k_hit
        self._replan_state.k_hit_new = self._k_hit
        self._replan_state.p_hit_new = self._p_hit.copy()

        self._U_buffer = result.U_buffer.copy()
        self._buffer_idx = 0

        logger.debug(
            "MPCController.step %d: replan k_hit=%d refine=%s buffer=%d",
            step, self._k_hit, refine_result.log, len(self._U_buffer),
        )
        return False

    def _apply_async_result(self, result: Any, step: int) -> bool:
        """应用异步重规划结果（含 buffer 时间偏移，无 refine）。

        异步结果可能来自若干步前的请求，需按 elapsed 偏移 U_mpc_full。

        Args:
            result: PlanResult（AsyncReplanMode.get_result）。
            step: 当前步索引。

        Returns:
            True 表示成功应用了新规划（replanned）。
        """
        assert self._replan_state is not None
        if result is None or result.request_step < 0 or result.k_hit_new <= 0:
            return False

        elapsed = step - result.request_step
        if elapsed >= len(result.U_mpc_full) or elapsed >= result.k_hit_new:
            return False

        U_shifted = result.U_mpc_full[elapsed:]
        ri = self._config.replan_interval
        if len(U_shifted) < ri:
            return False

        # 选择 buffer 长度（V11 行 1487-1494）
        for mult in [6, 4, 2, 1]:
            if len(U_shifted) >= ri * mult:
                self._U_buffer = U_shifted[:ri * mult]
                break

        self._buffer_idx = 0
        self._k_hit = max(1, result.k_hit_new - elapsed)
        self._apply_plan_fields(result)
        logger.info(
            "ASYNC_APPLY step=%d k_hit=%d elapsed=%d",
            step, self._k_hit, elapsed,
        )
        return True

    # ──────────────────────────────────────────────────────────────────
    # 私有方法：辅助
    # ──────────────────────────────────────────────────────────────────

    def _compute_buffer_fallback(self, arm_state: np.ndarray) -> np.ndarray:
        """buffer 耗尽 fallback（V11 行 1892-1909）。

        位置模式：``q + clip(J_p.T @ err, ±0.05)``
        力矩模式：``clip(J_p.T @ err * 30 - 2*qdot, ctrlrange)``

        Args:
            arm_state: 臂状态 [q, qdot] (12,)。

        Returns:
            控制指令 (6,)。
        """
        if self._k_hit > 0:
            self._env.set_arm_state(arm_state)
            p_ee = self._env.get_ee_pos()
            J_p = self._env.get_ee_jacp()
            err = self._p_hit - p_ee
            if self._config.is_position_mode:
                dq_backup = J_p.T @ err
                dq_backup = np.clip(dq_backup, -0.05, 0.05)
                return arm_state[:self._NQ] + dq_backup
            else:
                tau_backup = J_p.T @ err * 30.0
                tau_backup -= 2.0 * arm_state[self._NQ:]
                ctrl_lo = self._env.model.actuator_ctrlrange[:self._NU, 0]
                ctrl_hi = self._env.model.actuator_ctrlrange[:self._NU, 1]
                return np.clip(tau_backup, ctrl_lo, ctrl_hi)
        return self._hold_control(arm_state)

    def _hold_control(self, arm_state: np.ndarray) -> np.ndarray:
        """保持当前角度（done/buffer 耗尽时的兜底）。

        Args:
            arm_state: 臂状态。

        Returns:
            控制指令 (6,)。
        """
        if self._config.is_position_mode:
            return arm_state[:self._NQ].copy()
        return np.zeros(self._NU)
