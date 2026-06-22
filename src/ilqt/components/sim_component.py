"""仿真执行+诊断一体化组件（共享模块）。

从 ``scripts/rm65_mpc_v12.py`` 的 ``_V12SimComponent`` 提取为共享模块，
供 V12 / V11 等脚本复用，实现 ``ExecutorComponent`` Protocol。
"""

from __future__ import annotations

import numpy as np

from src.sim.rm65_env import RM65Env
from src.ilqt.robot_limits import RobotLimits
from src.ilqt.tube_types import TubeConfig, HittingTube
from src.ilqt.mpc_controller import MPCController
from src.sim.replay import compute_rebound_velocity


class SimComponent:
    """仿真执行+诊断一体化组件（实现 Executor + Diagnostics Protocol）。

    封装 V11 主循环中 step_full 前后的所有仿真专属逻辑：

      - PRE-step 诊断（distances / normal_align / ball_near / tube_ready）
      - 碰撞窗口管理（set_arm_collision）
      - 前向物理步进（step_full）
      - X 平面墙 PD 推回
      - 接触检测 + 弹性反弹
      - 执行层指标累积（max_qdot / max_tcp / max_face）
      - history 记录（X / U / ball_pos）

    对齐 V11 ``rm65_mpc_v11.py`` 1428-2123 行的逐步仿真逻辑。
    """

    def __init__(
        self,
        env: RM65Env,
        mpc: MPCController,
        robot_limits: RobotLimits,
        init_q: np.ndarray,
        is_position_mode: bool,
        hard_x_body_ids: list[int],
        initial_hitting_tube: HittingTube | None,
        tube_cfg: TubeConfig,
        dt: float,
    ) -> None:
        """初始化仿真组件。

        Args:
            env: RM65Env 仿真环境。
            mpc: MPCController（读取 _k_hit / _p_hit / _n_des）。
            robot_limits: 关节约束参数。
            init_q: 初始关节角度（PD 推回用）。
            is_position_mode: 位置模式标志。
            hard_x_body_ids: X 平面墙刚体 IDs。
            initial_hitting_tube: 初始候选窗口（tube_ready 诊断用）。
            tube_cfg: Tube 配置。
            dt: 仿真时间步长。
        """
        self._env = env
        self._mpc = mpc
        self._limits = robot_limits
        self._init_q = init_q
        self._is_position_mode = is_position_mode
        self._hard_x_body_ids = hard_x_body_ids
        self._initial_tube = initial_hitting_tube
        self._tube_cfg = tube_cfg
        self._dt = dt
        self._nu = env.NU
        self._nq = env.NQ

        # history 列表
        self.X_history: list[np.ndarray] = []
        self.U_history: list[np.ndarray] = []
        self.ball_pos_history: list[np.ndarray] = []
        self.distances_history: list[float] = []
        self.normal_align_history: list[float] = []
        self.ball_near_history: list[bool] = []
        self.tube_ready_history: list[bool] = []
        self.pos_error_history: list[float] = []

        # 执行层指标
        self.max_qdot_ratio: float = 0.0
        self.max_tcp_speed: float = 0.0
        self.max_racket_face_speed: float = 0.0
        self.total_mpc_steps: int = 0

        # 击球追踪
        self.ball_was_hit: bool = False
        self.hit_step: int = -1
        self.p_ee_at_hit: np.ndarray | None = None
        self.ball_pos_at_hit: np.ndarray | None = None
        self.v_ee_at_hit: float | None = None
        self.active_contact: bool = False
        self.passive_contact: bool = False

        self._step_count: int = 0

    # ── ExecutorComponent Protocol ──────────────────────────────────────

    def get_arm_state(self) -> np.ndarray:
        """返回当前右臂状态 [q(6), qdot(6)]，形状 (12,)。"""
        return self._env.get_arm_state()

    def execute(self, u_cmd: np.ndarray) -> None:
        """执行控制指令 — 封装 V11 逐步仿真逻辑。

        Args:
            u_cmd: 安全滤波后的控制指令 (6,)。
        """
        env = self._env
        step = self._step_count
        k_hit = self._mpc._k_hit
        p_hit = self._mpc._p_hit.copy()

        # ---- PRE-step 诊断（V11 1428-1471）----
        env.update_kinematics()
        ball_pos = env.get_ball_pos()
        p_ee_cur = env.get_ee_pos()
        n_rack_cur = env.get_ee_normal()
        dist_cur = float(np.linalg.norm(p_ee_cur - ball_pos))
        self.distances_history.append(dist_cur)

        # normal_align
        if self._initial_tube is not None and len(self._initial_tube.n_racket_des) > 0:
            n_des_cur = self._initial_tube.n_racket_des[0]
        else:
            n_des_cur = self._mpc._n_des
        n_align = float(n_rack_cur @ n_des_cur)
        self.normal_align_history.append(n_align)

        # ball_near
        is_ball_near = (dist_cur < 0.033 + 0.12 + 0.03) and (abs(n_align) > 0.7)
        self.ball_near_history.append(is_ball_near)

        # tube_ready
        is_tube_ready = False
        if self._initial_tube is not None and len(self._initial_tube.p_ball) > 0:
            window_half_steps = int(round(self._tube_cfg.window_half_ms / 1000.0 / self._dt))
            tube_center = self._initial_tube.best_k
            if abs(step - tube_center) <= window_half_steps:
                v_ball_mean = np.mean(self._initial_tube.v_ball, axis=0)
                v_norm_ball = np.linalg.norm(v_ball_mean)
                d_ball = v_ball_mean / v_norm_ball if v_norm_ball > 1e-6 else np.array([0.0, -1.0, 0.0])
                P_perp = np.eye(3) - np.outer(d_ball, d_ball)
                best_idx = int(np.argmin(np.abs(self._initial_tube.k_candidates - tube_center)))
                p_ref = self._initial_tube.p_ball[best_idx]
                dp = p_ee_cur - p_ref
                perp_dist = float(np.linalg.norm(P_perp @ dp))
                if perp_dist < 0.15 and abs(n_align) > 0.7:
                    is_tube_ready = True
        self.tube_ready_history.append(is_tube_ready)

        # ---- 碰撞窗口管理（V11 1914-1923）----
        enable_collision = False
        if not self.ball_was_hit:
            if k_hit <= 30 and dist_cur < 0.35:
                enable_collision = True
            elif k_hit <= 10:
                enable_collision = True
        env.set_arm_collision(enable_collision)

        # 记录碰撞前的球速度（用于弹性反弹计算，get_ball_vel 已返回 copy）
        ball_vel_before_step = env.get_ball_vel()

        # ---- 前向物理步进 ----
        x_new, ball_pos_new, ball_vel_new = env.step_full(u_cmd)

        # ---- POST-step X 平面墙 PD 推回（V11 2011-2034）----
        env.update_kinematics()
        violated = [
            bid for bid in self._hard_x_body_ids
            if env.data.xpos[bid, 0] > -0.1
        ]
        if violated:
            q_now = x_new[: self._nq]
            qdot_now = x_new[self._nq:]
            if self._is_position_mode:
                u_push = self._init_q.copy()
            else:
                u_push = 300.0 * (self._init_q - q_now) - 20.0 * qdot_now
                ctrl_lo = env.model.actuator_ctrlrange[: self._nu, 0]
                ctrl_hi = env.model.actuator_ctrlrange[: self._nu, 1]
                u_push = np.clip(u_push, ctrl_lo, ctrl_hi)
            x_new, ball_pos_new, ball_vel_new = env.step_full(u_push)

        # ---- 接触检测（V11 2036-2062）----
        ball_racket_hit = False
        if enable_collision and not self.ball_was_hit:
            n_contacts = env.data.ncon
            if n_contacts > 0:
                for ci in range(n_contacts):
                    c = env.data.contact[ci]
                    g1 = env.model.geom(c.geom1).name
                    g2 = env.model.geom(c.geom2).name
                    if ("ball" in g1 or "ball" in g2) and ("racket" in g1 or "racket" in g2):
                        ball_racket_hit = True
                        ee_vel = env.get_ee_vel()
                        ee_speed = float(np.linalg.norm(ee_vel))
                        self.v_ee_at_hit = ee_speed
                        if ee_speed > 0.3:
                            self.active_contact = True
                        else:
                            self.passive_contact = True
                        break

        # ---- 碰撞恢复 ----
        env.set_arm_collision(True)

        # ---- history 记录（V11 2067-2069）----
        self.X_history.append(x_new.copy())
        self.U_history.append(u_cmd.copy())
        self.ball_pos_history.append(ball_pos_new.copy())

        # ---- pos_error 记录（V11 2071-2073）----
        env.update_kinematics()
        pos_err = float(np.linalg.norm(env.get_ee_pos() - p_hit))
        self.pos_error_history.append(pos_err)

        # ---- 执行层指标（V11 2090-2098）----
        qdot_cur = x_new[self._nq:]
        qdot_ratio = float(np.max(np.abs(qdot_cur) / np.maximum(self._limits.qdot_max, 1e-8)))
        racket_speed = float(np.linalg.norm(env.get_ee_vel()))
        face_speed = env.get_racket_face_speed()
        self.max_qdot_ratio = max(self.max_qdot_ratio, qdot_ratio)
        self.max_tcp_speed = max(self.max_tcp_speed, racket_speed)
        self.max_racket_face_speed = max(self.max_racket_face_speed, face_speed)
        self.total_mpc_steps += 1

        # ---- 击球追踪 + 弹性反弹（V11 2105-2123）----
        if ball_racket_hit and not self.ball_was_hit:
            self.ball_was_hit = True
            self.hit_step = step
            env.update_kinematics()
            self.p_ee_at_hit = env.get_ee_pos().copy()
            self.ball_pos_at_hit = ball_pos_new.copy()
            # 反弹物理委托到 replay.py 共享函数（消除重复公式）
            n_racket = env.get_ee_normal()
            v_ee = env.get_ee_vel()
            v_ball_rebound = compute_rebound_velocity(ball_vel_before_step, v_ee, n_racket, e=0.8)
            env.set_ball_vel(v_ball_rebound)

        self._step_count += 1

    # ── 指标汇总 ─────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """返回汇总指标字典。"""
        return {
            "X_history": self.X_history,
            "U_history": self.U_history,
            "ball_pos_history": self.ball_pos_history,
            "distances_history": self.distances_history,
            "normal_align_history": self.normal_align_history,
            "ball_near_history": self.ball_near_history,
            "tube_ready_history": self.tube_ready_history,
            "pos_error_history": self.pos_error_history,
            "max_qdot_ratio": self.max_qdot_ratio,
            "max_tcp_speed": self.max_tcp_speed,
            "max_racket_face_speed": self.max_racket_face_speed,
            "total_mpc_steps": self.total_mpc_steps,
            "ball_was_hit": self.ball_was_hit,
            "hit_step": self.hit_step,
            "p_ee_at_hit": self.p_ee_at_hit,
            "ball_pos_at_hit": self.ball_pos_at_hit,
            "v_ee_at_hit": self.v_ee_at_hit,
            "active_contact": self.active_contact,
            "passive_contact": self.passive_contact,
        }
