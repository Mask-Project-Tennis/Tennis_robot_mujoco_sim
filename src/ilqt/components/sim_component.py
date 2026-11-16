"""仿真执行+诊断一体化组件（共享模块）。

从 ``scripts/rm65_mpc_v12.py`` 的 ``_V12SimComponent`` 提取为共享模块，
供 V12 / V11 等脚本复用，实现 ``ExecutorComponent`` Protocol。

Task 3.9 起按 ``SimExecutionEnv`` 协议（见 protocols.py）而非 RM65Env
具体类型消费环境：RM65Env（旧路径）与 ProfileSimEnv（R1D 路径）走同一
执行组件。R1D 通过可选接触 ID 配置（``ball_geom_id`` +
``racket_geom_ids``，来自 3.2 场景解析）启用按 geom ID 的接触判定；
缺省不传则保持旧 geom 名子串路径，RM65 行为逐字不变。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.ilqt.robot_limits import RobotLimits
from src.ilqt.tube_types import TubeConfig, HittingTube
from src.ilqt.mpc_controller import MPCController
from src.sim.replay import compute_rebound_velocity

if TYPE_CHECKING:
    from src.ilqt.components.protocols import SimExecutionEnv

# 主动命中判定的拍面速度阈值 [m/s]（接触时拍速 > 该值为主动击球，
# ≤ 为被动接触；V11 约定，R1D 复用同一阈值）
ACTIVE_HIT_FACE_SPEED_THRESHOLD = 0.3


class SimComponent:
    """仿真执行+诊断一体化组件（实现 Executor + Diagnostics Protocol）。

    封装 V11 主循环中 step_full 前后的所有仿真专属逻辑：

      - PRE-step 诊断（distances / normal_align / ball_near / tube_ready）
      - 碰撞窗口管理（set_arm_collision）
      - 前向物理步进（step_full）
      - X 平面墙 PD 推回（仅 RM65：``hard_x_body_ids`` 非空时；
        R1D 传空列表即完全跳过，一个逻辑步只推进一次 dt）
      - 接触检测 + 弹性反弹
      - 执行层指标累积（max_qdot / max_tcp / max_face）
      - history 记录（X / U / ball_pos）

    对齐 V11 ``rm65_mpc_v11.py`` 1428-2123 行的逐步仿真逻辑。
    """

    def __init__(
        self,
        env: "SimExecutionEnv",
        mpc: MPCController,
        robot_limits: RobotLimits,
        init_q: np.ndarray,
        is_position_mode: bool,
        hard_x_body_ids: list[int],
        initial_hitting_tube: HittingTube | None,
        tube_cfg: TubeConfig,
        dt: float,
        always_collision: bool = False,
        ball_geom_id: int | None = None,
        racket_geom_ids: tuple[int, ...] | None = None,
    ) -> None:
        """初始化仿真组件。

        Args:
            env: 仿真执行环境（RM65Env / ProfileSimEnv，按
                ``SimExecutionEnv`` 协议消费）。
            mpc: MPCController（读取 _k_hit / _p_hit / _n_des）。
            robot_limits: 关节约束参数。
            init_q: 初始关节角度（PD 推回用）。
            is_position_mode: 位置模式标志。
            hard_x_body_ids: X 平面墙刚体 IDs（R1D 传空列表 → 跳过
                X 墙推回，避免一个逻辑步额外 step_full 推进两次）。
            initial_hitting_tube: 初始候选窗口（tube_ready 诊断用）。
            tube_cfg: Tube 配置。
            dt: 仿真时间步长。
            always_collision: 击球前球-拍碰撞常开（chat14 碰撞窗口验证，
                绕过 k_hit/距离窗口判定，仅批量验证实验用）。
            ball_geom_id: 球碰撞 geom id（R1D 按解析 ID 判定接触；
                须与 racket_geom_ids 同时提供或同时缺省）。
            racket_geom_ids: 拍碰撞 geom id 元组（3.2 场景解析结果）。

        Raises:
            ValueError: ball_geom_id 与 racket_geom_ids 只提供其一
                （接触 ID 配置不完整，不得静默退回子串猜测）。
        """
        if (ball_geom_id is None) != (racket_geom_ids is None):
            raise ValueError(
                "接触 ID 配置不完整：ball_geom_id 与 racket_geom_ids 必须"
                "同时提供（R1D 按解析 geom ID 判定）或同时缺省"
                "（旧 RM65 geom 名子串路径）"
            )
        self._env = env
        self._mpc = mpc
        self._limits = robot_limits
        self._init_q = init_q
        self._is_position_mode = is_position_mode
        self._hard_x_body_ids = hard_x_body_ids
        self._initial_tube = initial_hitting_tube
        self._tube_cfg = tube_cfg
        self._dt = dt
        self._always_collision = always_collision
        self._contact_geom_ids: tuple[int, tuple[int, ...]] | None = (
            None
            if ball_geom_id is None or racket_geom_ids is None
            else (int(ball_geom_id), tuple(int(g) for g in racket_geom_ids))
        )
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

        # 出球质量（chat12 审稿 C6：击球后球的反弹速度）
        self.ball_out_speed: float | None = None
        self.ball_out_vy: float | None = None

        self._step_count: int = 0

    # ── ExecutorComponent Protocol ──────────────────────────────────────

    def get_arm_state(self) -> np.ndarray:
        """返回当前臂状态 [q(n), qdot(n)]，形状 (2n,)（n=env.NQ）。"""
        return self._env.get_arm_state()

    def execute(self, u_cmd: np.ndarray) -> None:
        """执行控制指令 — 封装 V11 逐步仿真逻辑。

        Args:
            u_cmd: 安全滤波后的控制指令 (n,)。
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
        # --always-collision（chat14 验证）：击球前常开，绕过 k_hit/距离窗口
        # 判定，用于核对"窗口外本会发生的接触被屏蔽"是否改变裁决
        enable_collision = False
        if not self.ball_was_hit:
            if self._always_collision:
                enable_collision = True
            elif k_hit <= 30 and dist_cur < 0.35:
                enable_collision = True
            elif k_hit <= 10:
                enable_collision = True
        env.set_arm_collision(enable_collision)

        # 记录碰撞前的球速度（用于弹性反弹计算，get_ball_vel 已返回 copy）
        ball_vel_before_step = env.get_ball_vel()

        # ---- 前向物理步进 ----
        x_new, ball_pos_new, ball_vel_new = env.step_full(u_cmd)

        # ---- POST-step X 平面墙 PD 推回（V11 2011-2034；仅 RM65 配置）----
        # R1D 传空 hard_x_body_ids → 无越界 body，不触发额外 step_full
        #（一个逻辑步只推进一次 dt；机身保护由 3.7 禁止对检查器承担）
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
                ctrl_range = env.control_range
                u_push = np.clip(u_push, ctrl_range[:, 0], ctrl_range[:, 1])
            x_new, ball_pos_new, ball_vel_new = env.step_full(u_push)

        # ---- 接触检测（V11 2036-2062；R1D 按解析 geom ID + 真实接触过滤）----
        ball_racket_hit = False
        if enable_collision and not self.ball_was_hit:
            n_contacts = env.data.ncon
            if n_contacts > 0:
                for ci in range(n_contacts):
                    c = env.data.contact[ci]
                    if self._is_ball_racket_contact(c):
                        ball_racket_hit = True
                        if self._contact_geom_ids is not None:
                            # R1D：拍面中心线速度标量（ProfileSimEnv 绑定
                            # TCP site 即拍面中心，无连杆偏移）
                            contact_speed = env.get_racket_face_speed()
                        else:
                            # 旧 RM65 路径：末端线速度模（行为逐字保留）
                            ee_vel = env.get_ee_vel()
                            contact_speed = float(np.linalg.norm(ee_vel))
                        self.v_ee_at_hit = contact_speed
                        if contact_speed > ACTIVE_HIT_FACE_SPEED_THRESHOLD:
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
            # 出球质量记录：反弹速度模长 + Y 分量（来球自 -Y 飞来，出球应朝 +Y）
            self.ball_out_speed = float(np.linalg.norm(v_ball_rebound))
            self.ball_out_vy = float(v_ball_rebound[1])

        self._step_count += 1

    def _is_ball_racket_contact(self, contact: Any) -> bool:
        """判断一个 MuJoCo 接触是否为球—拍物理接触。

        - R1D（构造时注入接触 ID）：按解析出的 ball/racket geom ID
          匹配，并要求 ``contact.dist <= 0``（真实物理接触）。dist 过滤
          是防御纵深/语义自洽约束：MuJoCo 3.9 斜圆柱显式 pair 在
          pair_margin=-1 关闭态下可产生 dist>0 的幽灵接触点（本仓实测，
          见 tests/test_r1d_sim_pipeline.py 的合成幽灵接触单测），而
          "物理接触"语义要求 dist<=0 —— 防止未来调用方在窗口外/
          常开等绕过窗口管理的路径上复用本判定时把幽灵点误判命中。
          当前 execute() 调用图中窗口关闭即不扫描，该过滤不改变既有
          行为；球碰连杆/地面/机身天然不匹配球—拍 ID 对。
        - 缺省（旧 RM65 路径）：geom 名子串判定（"ball"/"racket"），
          行为与 V11 逐字一致（不追加 dist 过滤，RM65 回归红线）。

        Args:
            contact: MuJoCo 接触结构（data.contact[i]）。

        Returns:
            True 表示该接触为球—拍命中接触。
        """
        if self._contact_geom_ids is not None:
            ball_id, racket_ids = self._contact_geom_ids
            g1, g2 = int(contact.geom1), int(contact.geom2)
            is_ball_racket_pair = (
                (g1 == ball_id and g2 in racket_ids)
                or (g2 == ball_id and g1 in racket_ids)
            )
            return is_ball_racket_pair and float(contact.dist) <= 0.0
        g1 = self._env.model.geom(contact.geom1).name  # type: ignore[union-attr]
        g2 = self._env.model.geom(contact.geom2).name  # type: ignore[union-attr]
        return ("ball" in g1 or "ball" in g2) and ("racket" in g1 or "racket" in g2)

    def advance_ball(self, elapsed_s: float) -> int:
        """不冻结时间实验（chat12 审稿 C1）：首次规划期间推进球物理。

        用与击打后保持段相同的 PD 力矩保持当前臂位形，推进
        round(elapsed_s/dt) 步 env.step_full —— 球在首次规划墙钟时间内
        继续飞行，计算延迟被计入任务时间。history 同步记录，
        主评估段的球/臂轨迹保持完整。

        Args:
            elapsed_s: 首次规划实际墙钟耗时（秒）。

        Returns:
            实际推进的物理步数。
        """
        env = self._env
        n_steps = int(round(elapsed_s / self._dt))
        if n_steps <= 0:
            return 0
        x_current = env.get_arm_state()
        ctrl_range = env.control_range
        for _ in range(n_steps):
            q_hold = x_current[: self._nq].copy()
            if self._is_position_mode:
                u_hold = q_hold.copy()
            else:
                u_hold = 100.0 * (q_hold - x_current[: self._nq]) - 10.0 * x_current[self._nq:]
                u_hold = np.clip(u_hold, ctrl_range[:, 0], ctrl_range[:, 1])
            x_current, ball_pos_adv, _ = env.step_full(u_hold)
            self.X_history.append(x_current.copy())
            self.U_history.append(u_hold.copy())
            self.ball_pos_history.append(ball_pos_adv.copy())
        return n_steps

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
            "ball_out_speed": self.ball_out_speed,
            "ball_out_vy": self.ball_out_vy,
        }
