"""MPC 规划控制器 — 可组合的规划模块。

封装完整规划生命周期：
击球点搜索 → refine 后过滤 → Tube 构建 → iLQR 求解 → 阶段调度 → 随挥目标计算。

不含：球感知、物理步进、碰撞检测、可视化。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.ilqt.async_replanner import AsyncReplanner, PlanRequest
from src.ilqt.planning_env import PlanningEnv
from src.ilqt.replan_core import do_replan
from src.ilqt.tube_types import ReplanState, TubeConfig
from src.real.runner_factory import build_replan_cfg, build_robot_limits, build_solver

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

    # ── 代价基底（v12: 对齐 V11 base_cost_fn 的 Q_p/Q_v基底）──
    Q_p_base: np.ndarray = field(
        default_factory=lambda: np.array([100000.0, 100000.0, 100000.0])
    )
    Q_v_base: np.ndarray = field(
        default_factory=lambda: np.array([400.0, 400.0, 400.0])
    )
    Q_qdot_base: float = 0.001
    Q_qddot_base: float = 0.0005
    Q_du_base: float = 0.001

    # ── 远段阈值（V11 far_threshold：k_hit > 此值时用 JT 控制）──
    far_threshold: int = 50

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

    def __init__(self, env: PlanningEnv, config: MPCConfig) -> None:
        """初始化 solver/limits/replanner/state 构建。

        Args:
            env: 规划环境（PlanningEnv 或 RM65Env），用于主线程规划计算。
            config: MPC 配置（控制规划行为）。
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
        self._replan_cfg: dict | None = None
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
        self._follow_through_start: int = -1
        self._ball_was_hit: bool = False
        self._p_ee_at_hit: np.ndarray | None = None
        self._ball_pos_at_hit: np.ndarray | None = None
        self._k_hit_at_follow: int = 0

        # ── refine_hit_point 状态（V11 行 1072-1074）──
        self._hit_lock_active: bool = False
        self._last_p_hit: np.ndarray | None = None

        # ── 结束标志 ──
        self._ball_unreachable: bool = False
        self._done: bool = False

        # ── 异步状态 ──
        self._async_replan_submitted: bool = False
        self._mpc_horizon: int = config.total_horizon

    # ──────────────────────────────────────────────────────────────────
    # 公有接口
    # ──────────────────────────────────────────────────────────────────

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

        # 1. 计算击球方向（V11 行 819-838）
        self._d_hat, self._v_hit_desired = self._compute_direction(ball_vel)
        self._d_follow = self._d_hat.copy()

        # 2. 构建 replan_cfg（复用 runner_factory.build_replan_cfg + MPCConfig 覆盖）
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
        result = do_replan(
            first_request, self._replanner.env_plan,
            self._replan_state, self._replan_cfg,
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

        # 8. refine_hit_point 初始过滤（V11 行 1590 调用点）
        self._env.set_ball_state(ball_pos.copy(), ball_vel.copy())
        self._p_hit, self._k_hit, refine_log = self._refine_hit_point(
            self._p_hit, self._k_hit, self._config.total_horizon, self._env,
        )
        self._replan_state.p_hit_new = self._p_hit.copy()
        self._replan_state.k_hit_new = self._k_hit
        if refine_log == "swapped":
            logger.info(
                "MPCController.start refine: p_hit 修正, k_hit=%d, refine=%s",
                self._k_hit, refine_log,
            )

        # 9. 初始化 buffer（V11 行 1381-1382）
        self._U_buffer = result.U_buffer.copy()
        self._buffer_idx = 0
        self._step_count = 0
        self._mpc_horizon = self._config.total_horizon

        # 10. 更新 replan_cfg 的 k_hit_total
        self._replan_cfg["k_hit_total"] = self._k_hit

        # 11. 异步模式首次提交（V11 行 1394-1409）
        if self._config.async_mode:
            async_request = PlanRequest(
                x_current=arm_state.copy(),
                ball_pos=ball_pos.copy(),
                ball_vel=ball_vel.copy(),
                step=0,
                k_hit_current=self._k_hit,
                U_prev=self._U_prev.copy(),
                p_hit_current=self._p_hit.copy(),
                v_hit_desired=self._v_hit_desired.copy(),
                n_des_current=self._n_des.copy(),
                is_first_plan=False,
            )
            if self._replanner.submit(async_request):
                self._async_replan_submitted = True

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

        # ── 1. 检查随挥触发 ──
        if self._follow_through_start < 0:
            triggered = self._check_follow_through(sc, self._k_hit, arm_state, ball_pos)
            if triggered:
                # 触发当步即进入随挥（dt_follow=0）
                u_cmd = self._compute_follow_through_control(arm_state, dt_follow=0)
                return MPCStepResult(
                    u_cmd=u_cmd, phase="follow_through", k_hit=0,
                    p_hit=self._p_hit.copy(), n_des=self._n_des.copy(),
                    info={"follow_step": 0},
                )
        else:
            # ── 已在随挥阶段 ──
            dt_follow = sc - self._follow_through_start
            if dt_follow <= self._config.follow_through_steps:
                u_cmd = self._compute_follow_through_control(arm_state, dt_follow)
                return MPCStepResult(
                    u_cmd=u_cmd, phase="follow_through", k_hit=0,
                    p_hit=self._p_hit.copy(), n_des=self._n_des.copy(),
                    info={"follow_step": dt_follow},
                )
            else:
                # 随挥完成
                self._done = True
                u_hold = self._hold_control(arm_state)
                return MPCStepResult(
                    u_cmd=u_hold, phase="done", k_hit=0,
                )

        # ── 2. MPC 阶段 ──
        need_replan = (
            (sc % self._config.replan_interval == 0)
            or (self._buffer_idx >= len(self._U_buffer))
        ) and sc < self._mpc_horizon

        replanned = False
        if self._config.async_mode:
            replanned = self._async_step(ball_pos, ball_vel, arm_state, sc, need_replan)
        else:
            if need_replan:
                replanned = True
                unreachable = self._sync_replan(ball_pos, ball_vel, arm_state, sc)
                if unreachable:
                    u_hold = self._hold_control(arm_state)
                    return MPCStepResult(
                        u_cmd=u_hold, phase="done", k_hit=0,
                        ball_unreachable=True,
                    )

        # ── 3. 提取 U_buffer（V11 行 1889-1912）──
        if self._buffer_idx < len(self._U_buffer):
            u_cmd = self._U_buffer[self._buffer_idx].copy()
            self._buffer_idx += 1
        else:
            u_cmd = self._compute_buffer_fallback(arm_state)

        # ── 4. 确定阶段 ──
        phase = self._classify_phase(self._k_hit)

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
    # 私有方法：方向与配置
    # ──────────────────────────────────────────────────────────────────

    def _compute_direction(
        self, ball_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """从球速度计算击球方向（V11 行 819-838）。

        Args:
            ball_vel: 球速度 (3,)。

        Returns:
            (d_hat, v_hit_desired)：d_hat 为来球反方向单位向量，
            v_hit_desired 为期望击球时刻末端速度。
        """
        v_norm = float(np.linalg.norm(ball_vel))
        if v_norm > 1e-6:
            d_hat = -ball_vel / v_norm
        else:
            d_hat = np.array([0.0, 1.0, 0.0])
        v_hit_desired = self._config.target_speed * d_hat
        return d_hat, v_hit_desired

    def _build_replan_cfg(
        self, d_hat: np.ndarray, v_hit_desired: np.ndarray,
    ) -> dict:
        """构建 do_replan 所需配置字典。

        复用 ``runner_factory.build_replan_cfg`` 基础字典，用 MPCConfig 覆盖。

        Args:
            d_hat: 击球方向单位向量。
            v_hit_desired: 期望末端速度。

        Returns:
            replan_cfg dict（可直接传给 do_replan）。
        """
        cfg = build_replan_cfg(
            self._env, self._robot_limits, self._solver, d_hat, v_hit_desired,
        )
        c = self._config
        cfg.update({
            # 时间与迭代参数
            "dt": c.dt,
            "total_horizon": c.total_horizon,
            "fixed_horizon": c.fixed_horizon,
            "replan_interval": c.replan_interval,
            "max_iter_per_plan": c.max_iter_per_plan,
            "first_plan_iters": c.first_plan_iters,
            "near_plan_iters": c.near_plan_iters,
            "near_threshold": c.near_threshold,
            # 代价参数
            "R": c.R,
            "Q_p_scale_far": c.Q_p_scale_far,
            "Q_v_scale_far": c.Q_v_scale_far,
            "Q_p_scale_near": c.Q_p_scale_near,
            "Q_v_scale_near": c.Q_v_scale_near,
            "normal_weight": c.normal_weight,
            "racket_speed": c.racket_speed,
            "max_tcp_speed": c.max_tcp_speed,
            # 模式
            "is_position_mode": c.is_position_mode,
            "ablation_mode": c.ablation_mode,
            "use_backswing": c.use_backswing,
            "use_r_decay": c.use_r_decay,
            "r_decay_ratio": c.r_decay_ratio,
            "fix_joint5_angle": c.fix_joint5_angle,
            "backswing_offset": c.backswing_offset,
            "backswing_ratio": c.backswing_ratio,
            "normal_flip": c.normal_flip,
            # 几何
            "shoulder_pos": c.shoulder_pos,
            "workspace_radius": c.workspace_radius,
            # 方向（d_follow = d_hat）
            "d_hat": d_hat,
            "d_follow": d_hat,
            "v_hit_desired": v_hit_desired,
            "v_hit_at_contact": v_hit_desired,
            # 随挥
            "hit_shift": c.follow_through_length,
            "follow_through_length": c.follow_through_length,
            "follow_through_steps": c.follow_through_steps,
            "follow_through_v_terminal": c.follow_through_v_terminal,
            # Tube
            "tube_cfg": c.tube_cfg,
            "smooth_far": c.smooth_far,
            "smooth_mid": c.smooth_mid,
            "smooth_near": c.smooth_near,
            # 扰动
            "time_perturb_s": c.time_perturb_s,
            "space_perturb_m": c.space_perturb_m,
            "perturb_alpha_min": c.perturb_alpha_min,
            # v12: 代价基底 + 远段阈值（对齐 V11 base_cost_fn）
            "Q_p_base": c.Q_p_base,
            "Q_v_base": c.Q_v_base,
            "Q_qdot_base": c.Q_qdot_base,
            "Q_qddot_base": c.Q_qddot_base,
            "Q_du_base": c.Q_du_base,
            "far_threshold": c.far_threshold,
            # 初始 k_hit_total（首次规划后更新）
            "k_hit_total": c.total_horizon,
        })
        return cfg

    # ──────────────────────────────────────────────────────────────────
    # 私有方法：重规划
    # ──────────────────────────────────────────────────────────────────

    def _sync_replan(
        self, ball_pos: np.ndarray, ball_vel: np.ndarray,
        arm_state: np.ndarray, step: int,
    ) -> bool:
        """同步重规划 — 统一调用 do_replan（V11 行 1548-1872 简化）。

        Returns:
            True 表示球不可达。
        """
        assert self._replan_cfg is not None and self._replan_state is not None
        request = PlanRequest(
            x_current=arm_state.copy(),
            ball_pos=ball_pos.copy(),
            ball_vel=ball_vel.copy(),
            step=step,
            k_hit_current=self._k_hit,
            U_prev=self._U_prev.copy() if len(self._U_prev) > 0 else np.zeros((0, self._NU)),
            p_hit_current=self._p_hit.copy(),
            v_hit_desired=self._v_hit_desired.copy(),
            n_des_current=self._n_des.copy(),
            is_first_plan=False,
        )
        assert self._replanner.env_plan is not None
        result = do_replan(
            request, self._replanner.env_plan,
            self._replan_state, self._replan_cfg,
        )

        if result.ball_unreachable:
            self._ball_unreachable = True
            self._done = True
            logger.warning("MPCController.step %d: 球不可达", step)
            return True

        # 更新规划状态
        self._k_hit = result.k_hit_new
        self._p_hit = result.p_hit_new.copy()
        self._v_ball_hit = result.v_ball_hit_new.copy()
        self._n_des = result.n_des_new.copy()
        self._U_prev = result.U_prev.copy()

        # refine_hit_point 后过滤（V11 行 1590）
        self._env.set_ball_state(ball_pos.copy(), ball_vel.copy())
        remaining = max(self._config.total_horizon - step, 5)
        self._p_hit, self._k_hit, refine_log = self._refine_hit_point(
            self._p_hit, self._k_hit, remaining, self._env,
        )

        # 同步到 replan_state
        self._replan_state.k_hit_new = self._k_hit
        self._replan_state.p_hit_new = self._p_hit.copy()
        self._replan_state.v_ball_hit_new = self._v_ball_hit.copy()
        self._replan_state.current_n_des = self._n_des.copy()
        self._replan_state.U_prev = self._U_prev.copy()

        # 更新 buffer
        self._U_buffer = result.U_buffer.copy()
        self._buffer_idx = 0

        logger.info(
            "MPCController.step %d: replan k_hit=%d refine=%s buffer=%d",
            step, self._k_hit, refine_log, len(self._U_buffer),
        )
        return False

    def _async_step(
        self, ball_pos: np.ndarray, ball_vel: np.ndarray,
        arm_state: np.ndarray, step: int, need_replan: bool,
    ) -> bool:
        """异步重规划路径（V11 行 1475-1546）。

        Returns:
            True 表示本步应用了新规划（replanned）。
        """
        assert self._replan_state is not None
        replanned = False

        # 检查异步结果（V11 行 1478-1527）
        if self._replanner.has_new_plan():
            result = self._replanner.apply_new_plan()
            if result is not None and result.request_step >= 0 and result.k_hit_new > 0:
                self._async_replan_submitted = False
                elapsed = step - result.request_step
                if elapsed < len(result.U_mpc_full) and elapsed < result.k_hit_new:
                    U_shifted = result.U_mpc_full[elapsed:]
                    k_hit_adjusted = max(1, result.k_hit_new - elapsed)
                    ri = self._config.replan_interval
                    if len(U_shifted) >= ri:
                        # 选择 buffer 长度（V11 行 1487-1494）
                        if len(U_shifted) >= ri * 6:
                            self._U_buffer = U_shifted[:ri * 6]
                        elif len(U_shifted) >= ri * 4:
                            self._U_buffer = U_shifted[:ri * 4]
                        elif len(U_shifted) >= ri * 2:
                            self._U_buffer = U_shifted[:ri * 2]
                        else:
                            self._U_buffer = U_shifted[:ri]
                        self._buffer_idx = 0
                        self._k_hit = k_hit_adjusted
                        self._p_hit = result.p_hit_new.copy()
                        self._v_ball_hit = result.v_ball_hit_new.copy()
                        self._n_des = result.n_des_new.copy()
                        self._U_prev = result.U_prev.copy()
                        self._replan_state.k_hit_new = self._k_hit
                        self._replan_state.p_hit_new = self._p_hit.copy()
                        self._replan_state.v_ball_hit_new = self._v_ball_hit.copy()
                        self._replan_state.current_n_des = self._n_des.copy()
                        self._replan_state.U_prev = self._U_prev.copy()
                        replanned = True
                        logger.info(
                            "ASYNC_APPLY step=%d k_hit=%d elapsed=%d",
                            step, self._k_hit, elapsed,
                        )

        # 提交新请求（V11 行 1529-1546）
        can_submit = (
            not self._async_replan_submitted
            and not self._replanner.is_planning()
            and step > 0
        )
        if need_replan and can_submit:
            request = PlanRequest(
                x_current=arm_state.copy(),
                ball_pos=ball_pos.copy(),
                ball_vel=ball_vel.copy(),
                step=step,
                k_hit_current=self._k_hit,
                U_prev=self._U_prev.copy() if len(self._U_prev) > 0 else np.zeros((0, self._NU)),
                p_hit_current=self._p_hit.copy(),
                v_hit_desired=self._v_hit_desired.copy(),
                n_des_current=self._n_des.copy(),
                is_first_plan=False,
            )
            if self._replanner.submit(request):
                self._async_replan_submitted = True

        return replanned

    # ──────────────────────────────────────────────────────────────────
    # 私有方法：refine_hit_point（V11 行 1076-1236 闭包提取）
    # ──────────────────────────────────────────────────────────────────

    def _refine_hit_point(
        self,
        p_hit: np.ndarray,
        k_hit: int,
        remaining: int,
        env_local: PlanningEnv,
    ) -> tuple[np.ndarray, int, str]:
        """击球点可执行性后过滤（V11 行 1076-1236 嵌套闭包提取）。

        在 tube 候选窗口内搜索关节裕度更充足的替代点。
        闭包变量变为实例属性：self._hit_lock_active, self._last_p_hit, self._x_current。

        Args:
            p_hit: 原始击球点 (3,)。
            k_hit: 原始击球步数。
            remaining: 剩余步数。
            env_local: 用于 IK 和轨迹预测的环境。

        Returns:
            (p_hit_refined, k_hit_refined, log_message)。
        """
        hit_lock_threshold = 60
        hard_margin_deg = 2.0
        warn_margin_deg = 5.0
        j1_warn_margin_deg = 8.0
        window_half_steps = 15

        # 防抖锁定：末段不再换点
        if k_hit <= hit_lock_threshold:
            if not self._hit_lock_active:
                self._hit_lock_active = True
                logger.info(
                    "[HIT_LOCK] k_hit=%d ≤ %d, 锁定击球点不再替换",
                    k_hit, hit_lock_threshold,
                )
            return p_hit, k_hit, "locked"

        # 快速 IK 检查当前点的双边裕度
        q_ik = env_local.solve_ik(
            p_hit, q_init=self._x_current[:env_local.NQ], max_iter=50, eps=1e-2,
        )
        margin_lower_deg = (q_ik - self._robot_limits.q_lower) * 180.0 / np.pi
        margin_upper_deg = (self._robot_limits.q_upper - q_ik) * 180.0 / np.pi
        margin_min_deg = float(np.min(np.minimum(margin_lower_deg, margin_upper_deg)))
        margin_j1_deg = float(min(margin_lower_deg[1], margin_upper_deg[1]))

        high_risk = margin_min_deg < hard_margin_deg
        j1_near = margin_j1_deg < j1_warn_margin_deg
        medium_risk = margin_min_deg < warn_margin_deg

        if not high_risk and not j1_near:
            if medium_risk:
                logger.info(
                    "[HIT_KEEP] p=%s min_margin=%.1f° j1=%.1f° → feasible (medium risk)",
                    np.round(p_hit, 3), margin_min_deg, margin_j1_deg,
                )
            return p_hit, k_hit, "feasible"

        logger.warning(
            "[HIT_RISK] p=%s k=%d min_margin=%.1f° j1=%.1f° → searching alternatives",
            np.round(p_hit, 3), k_hit, margin_min_deg, margin_j1_deg,
        )

        shoulder_pos = self._config.shoulder_pos
        workspace_radius = self._config.workspace_radius
        best_candidate: tuple | None = None
        best_score = -1e9

        # 策略1：微调位置偏移（保持时间不变）
        if j1_near:
            j1_dir = 1.0 if margin_j1_deg == margin_lower_deg[1] else -1.0
            for offset_cm in [3, 5, 8, 12]:
                offset_m = offset_cm / 100.0
                p_shifted = p_hit.copy()
                p_shifted[1] += j1_dir * offset_m
                dist_s = float(np.linalg.norm(p_shifted - shoulder_pos))
                if dist_s > workspace_radius or p_shifted[2] < 0.3:
                    continue
                q_s = env_local.solve_ik(
                    p_shifted, q_init=self._x_current[:env_local.NQ],
                    max_iter=30, eps=2e-2,
                )
                m_low_s = (q_s - self._robot_limits.q_lower) * 180.0 / np.pi
                m_up_s = (self._robot_limits.q_upper - q_s) * 180.0 / np.pi
                m_min_s = float(np.min(np.minimum(m_low_s, m_up_s)))
                m_j1_s = float(min(m_low_s[1], m_up_s[1]))
                if m_min_s < margin_min_deg - 0.5:
                    continue
                if m_j1_s < j1_warn_margin_deg:
                    continue
                score_s = (
                    2.0 * m_min_s
                    + 3.0 * m_j1_s
                    - 50.0 * float(np.linalg.norm(p_shifted - p_hit))
                )
                if score_s > best_score:
                    best_score = score_s
                    best_candidate = (p_shifted.copy(), k_hit, m_min_s, m_j1_s)

        # 策略2：tube 窗口搜索（改变时间点）
        ball_positions_pred, _ = env_local.predict_ball_trajectory(
            env_local.get_ball_pos(), env_local.get_ball_vel(),
            min(remaining + 30, 300),
        )
        k_min = max(1, k_hit - window_half_steps)
        k_max = min(len(ball_positions_pred), k_hit + window_half_steps)

        for k_cand in range(k_min, k_max + 1):
            if k_cand == k_hit:
                continue
            p_cand = ball_positions_pred[k_cand - 1]
            dist_cand = float(np.linalg.norm(p_cand - shoulder_pos))
            if dist_cand > workspace_radius * 1.1 or p_cand[2] < 0.3:
                continue
            q_cand = env_local.solve_ik(
                p_cand, q_init=self._x_current[:env_local.NQ],
                max_iter=30, eps=2e-2,
            )
            m_low = (q_cand - self._robot_limits.q_lower) * 180.0 / np.pi
            m_up = (self._robot_limits.q_upper - q_cand) * 180.0 / np.pi
            m_min = float(np.min(np.minimum(m_low, m_up)))
            m_j1 = float(min(m_low[1], m_up[1]))
            if m_min < margin_min_deg - 0.5:
                continue
            y_risk = max(0.0, (shoulder_pos[1] - 0.40) - p_cand[1])
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
                return p_new, k_new, "swapped"

        logger.warning(
            "[HIT_RISK] min_margin=%.1f° j1=%.1f°, no safer candidate found",
            margin_min_deg, margin_j1_deg,
        )
        return p_hit, k_hit, "risk_kept"

    # ──────────────────────────────────────────────────────────────────
    # 私有方法：随挥
    # ──────────────────────────────────────────────────────────────────

    def _check_follow_through(
        self, step_count: int, k_hit: int,
        arm_state: np.ndarray, ball_pos: np.ndarray,
    ) -> bool:
        """随挥触发检测（V11 行 1418-1425 + 2127-2143）。

        Args:
            step_count: 当前步索引。
            k_hit: 剩余击球步数。
            arm_state: 臂状态。
            ball_pos: 球位置。

        Returns:
            True 表示触发了随挥。
        """
        if self._follow_through_start >= 0:
            return False
        if self._config.follow_through_steps <= 0:
            return False

        triggered = False
        # 强制触发：MPC 阶段结束（V11 行 1418-1425）
        if step_count >= self._mpc_horizon:
            triggered = True
        # planned 模式：到达计划击打时刻（V11 行 2133-2143）
        elif (self._config.follow_trigger == "planned" and k_hit <= 1):
            triggered = True

        if triggered:
            self._follow_through_start = step_count
            self._k_hit_at_follow = k_hit
            self._env.set_arm_state(arm_state)
            if self._p_ee_at_hit is None:
                self._p_ee_at_hit = self._env.get_ee_pos().copy()
            if self._ball_pos_at_hit is None:
                self._ball_pos_at_hit = ball_pos.copy()
            logger.info(
                "MPCController.step %d: 触发随挥 (%d 步)",
                step_count, self._config.follow_through_steps,
            )

        return triggered

    def _compute_follow_through_control(
        self, arm_state: np.ndarray, dt_follow: int,
    ) -> np.ndarray:
        """随挥 PD/IK 控制计算（V11 行 2149-2214）。

        位置模式：``env.solve_ik(p_des_follow)``
        力矩模式：``J_p.T @ (Kp*dp - Kd*v_ee)``

        Args:
            arm_state: 臂状态 [q, qdot] (12,)。
            dt_follow: 随挥内步数（0=触发当步）。

        Returns:
            控制指令 (6,)。
        """
        assert self._p_ee_at_hit is not None, "随挥未初始化 p_ee_at_hit"
        c = self._config

        # 沿 d_follow 方向匀减速直线延伸（V11 行 2152-2158）
        v_max_follow = float(np.linalg.norm(self._v_hit_desired))
        T_follow = c.follow_through_steps * c.dt
        a_follow = v_max_follow / T_follow if T_follow > 0 else 0.0
        t_elapsed = dt_follow * c.dt
        p_des_follow = self._p_ee_at_hit + self._d_follow * (
            v_max_follow * t_elapsed - 0.5 * a_follow * t_elapsed ** 2
        )

        self._env.set_arm_state(arm_state)
        if c.is_position_mode:
            u_follow = self._env.solve_ik(
                p_des_follow, q_init=arm_state[:self._NQ], max_iter=20, eps=1e-2,
            )
        else:
            p_ee_cur = self._env.get_ee_pos()
            J_p = self._env.get_ee_jacp()
            dp = p_des_follow - p_ee_cur
            Kp_follow = 200.0
            Kd_follow = 20.0
            F_follow = Kp_follow * dp - Kd_follow * (J_p @ arm_state[self._NQ:])
            u_follow = J_p.T @ F_follow

        ctrl_lo = self._env.model.actuator_ctrlrange[:self._NU, 0]
        ctrl_hi = self._env.model.actuator_ctrlrange[:self._NU, 1]
        return np.clip(u_follow, ctrl_lo, ctrl_hi)

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

    def _classify_phase(self, k_hit: int) -> str:
        """根据剩余击球步数判定阶段。

        Args:
            k_hit: 剩余击球步数。

        Returns:
            "far" / "mid" / "near"。
        """
        if k_hit > 50:
            return "far"
        elif k_hit > 20:
            return "mid"
        else:
            return "near"
