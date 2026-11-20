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

from dataclasses import dataclass
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

# 球-拍接近门控的 V12 缺省几何（RM65 旧路径逐字保持；R1D 由
# bundle.racket / 场景参数注入覆盖，见 Task 5 brief §4.5）
DEFAULT_BALL_RADIUS = 0.033
DEFAULT_RACKET_RADIUS = 0.12
DEFAULT_PROXIMITY_MARGIN = 0.03

# 命中口径规则标识（T6：写入指标/产物，便于跨机器人对照与审计）
LEGACY_HIT_RULE = "first_ball_racket_contact_in_collision_window"
PHYSICAL_HIT_RULE = "first_physical_face_contact_face_speed_0p3"


@dataclass(frozen=True)
class ContactEvent:
    """一个球—拍接触事件（同一物理步采集，含正距离接近事件）。

    字段与 T6 契约逐字一致：``step`` / ``time_s`` 为该物理步（``step *
    dt``）；``distance_m`` 直接取 MuJoCo ``contact.dist``（正值为接近
    margin，负值为真实穿透）；``physical_contact`` 仅在 ``dist <= 0`` 时
    为 True；``face_speed_m_s`` 取**同一物理步**的拍面/末端速度；命中
    分类按拍面 + 物理接触 + 速度阈值（0.3 m/s）判定。
    """

    step: int
    time_s: float
    ball_geom_name: str
    other_geom_name: str
    surface_kind: str
    distance_m: float
    face_speed_m_s: float
    physical_contact: bool
    hit_type: str

    def to_dict(self) -> dict[str, object]:
        """返回 JSON 可序列化的字段字典（键名与契约一致）。"""
        return {
            "step": int(self.step),
            "time_s": float(self.time_s),
            "ball_geom_name": self.ball_geom_name,
            "other_geom_name": self.other_geom_name,
            "surface_kind": self.surface_kind,
            "distance_m": float(self.distance_m),
            "face_speed_m_s": float(self.face_speed_m_s),
            "physical_contact": bool(self.physical_contact),
            "hit_type": self.hit_type,
        }


def classify_surface_kind(geom_name: str) -> str:
    """按拍 geom 名称判定接触面类型。

    Args:
        geom_name: 球拍侧 geom 名称（如 ``r_racket_face`` /
            ``racket_frame`` / ``racket_handle``）。

    Returns:
        ``"face"`` / ``"frame"`` / ``"handle"`` / ``"other"``。
    """
    name = str(geom_name).lower()
    if "face" in name:
        return "face"
    if "frame" in name or "throat" in name:
        return "frame"
    if "handle" in name:
        return "handle"
    return "other"


def classify_contact_hit_type(
    surface_kind: str,
    physical_contact: bool,
    face_speed_m_s: float,
) -> str:
    """按接触面/物理接触/拍速分类单个事件命中类型。

    Args:
        surface_kind: 接触面类型（见 :func:`classify_surface_kind`）。
        physical_contact: 是否为真实物理接触（``dist <= 0``）。
        face_speed_m_s: 同一物理步的拍面速度 [m/s]。

    Returns:
        ``"active"``（拍面真实接触且拍速 > 0.3）、``"passive"``（拍面
        真实接触但拍速 ≤ 0.3）或 ``"miss"``。
    """
    if surface_kind == "face" and physical_contact:
        if float(face_speed_m_s) > ACTIVE_HIT_FACE_SPEED_THRESHOLD:
            return "active"
        return "passive"
    return "miss"



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
        ball_radius: float = DEFAULT_BALL_RADIUS,
        racket_radius: float = DEFAULT_RACKET_RADIUS,
        proximity_margin: float = DEFAULT_PROXIMITY_MARGIN,
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
            ball_radius: 球半径 [m]（球-拍接近门控；缺省为 V12 RM65 旧值
                0.033，R1D 由场景参数提供）。
            racket_radius: 球拍等效半径 [m]（缺省 0.12 保持 RM65 旧判据；
                R1D 由 ``bundle.racket.face_semi_axes`` 提供）。
            proximity_margin: 接近门控余量 [m]（缺省 0.03 保持 RM65 旧判据）。

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
        # 球-拍接近门控几何（Task 5：参数化，旧 RM65 缺省值逐位不变）
        for _name, _value in (
            ("ball_radius", ball_radius),
            ("racket_radius", racket_radius),
            ("proximity_margin", proximity_margin),
        ):
            if not np.isfinite(float(_value)) or float(_value) < 0.0:
                raise ValueError(
                    f"{_name} 必须为非负有限值，收到 {_value!r}"
                )
        self.ball_radius = float(ball_radius)
        self.racket_radius = float(racket_radius)
        self.proximity_margin = float(proximity_margin)
        # 与旧表达式 (0.033 + 0.12 + 0.03) 相同的相加次序，保证 RM65
        # 缺省判据逐位不变。
        self._ball_near_threshold = (
            self.ball_radius + self.racket_radius + self.proximity_margin
        )
        self._nu = env.NU
        self._nq = env.NQ

        # history 列表（ball_vel_history 与 ball_pos_history 严格同帧对齐，
        # 见 R07：每逻辑步末追加该步最终球速度，含击球反弹后的速度）
        self.X_history: list[np.ndarray] = []
        self.U_history: list[np.ndarray] = []
        self.ball_pos_history: list[np.ndarray] = []
        self.ball_vel_history: list[np.ndarray] = []
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

        # T6：本次回合采集到的全部球—拍接触事件（同一步多事件全保留），
        # 以及独立于 legacy 口径的 physical 命中判定（首帧拍面真实接触）。
        self.contact_events: list[ContactEvent] = []
        self.physical_hit: bool = False
        self.physical_hit_type: str = "miss"
        self.physical_hit_step: int = -1

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

        # ball_near（Task 5：球/拍半径由构造参数提供，缺省值与旧判据逐位一致）
        is_ball_near = (dist_cur < self._ball_near_threshold) and (
            abs(n_align) > 0.7
        )
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
            # T6：先采集本物理步的全部球—拍接触事件（含正距离 margin），
            # 再执行下方 legacy 第一接触判定（判据/阈值/break 一字不改）。
            self._collect_contact_events(step)
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
            # 本逻辑步的最终球速度取反弹后的速度（与 set_ball_vel 写入的
            # env 状态一致，保证 ball_vel_history 记录的是该帧真实速度）
            ball_vel_new = v_ball_rebound
            # 出球质量记录：反弹速度模长 + Y 分量（来球自 -Y 飞来，出球应朝 +Y）
            self.ball_out_speed = float(np.linalg.norm(v_ball_rebound))
            self.ball_out_vy = float(v_ball_rebound[1])

        # ---- 球速度 history（R07：与 ball_pos 同帧，取本逻辑步最终速度）----
        self.ball_vel_history.append(
            np.asarray(ball_vel_new, dtype=np.float64).copy()
        )

        self._step_count += 1

    def _current_contact_speed(self) -> float:
        """取当前物理步的拍面速度标量 [m/s]。

        R1D（注入接触 ID）用 ``get_racket_face_speed()``（拍面中心线速度）；
        RM65 旧路径用末端线速度模（与 legacy 判定逐字一致）。
        """
        if self._contact_geom_ids is not None:
            return float(self._env.get_racket_face_speed())
        return float(np.linalg.norm(self._env.get_ee_vel()))

    def _is_ball_racket_pair(self, contact: Any) -> bool:
        """判断接触是否为球—拍对（**不做** dist 过滤，供事件采集用）。

        与 :meth:`_is_ball_racket_contact` 的区别：后者是 legacy 命中判据
        （R1D 要求 ``dist <= 0``），本方法只判断 geom 对，以便把正距离
        margin 接近也记录为 ``ContactEvent``（``physical_contact=False``）。
        """
        if self._contact_geom_ids is not None:
            ball_id, racket_ids = self._contact_geom_ids
            g1, g2 = int(contact.geom1), int(contact.geom2)
            return (g1 == ball_id and g2 in racket_ids) or (
                g2 == ball_id and g1 in racket_ids
            )
        g1 = self._env.model.geom(contact.geom1).name  # type: ignore[union-attr]
        g2 = self._env.model.geom(contact.geom2).name  # type: ignore[union-attr]
        return ("ball" in g1 or "ball" in g2) and (
            "racket" in g1 or "racket" in g2
        )

    def _contact_geom_names(self, contact: Any) -> tuple[str, str]:
        """返回 ``(球 geom 名, 拍 geom 名)``（顺序无关的接触对）。"""
        g1 = self._env.model.geom(contact.geom1).name  # type: ignore[union-attr]
        g2 = self._env.model.geom(contact.geom2).name  # type: ignore[union-attr]
        if self._contact_geom_ids is not None:
            ball_id = self._contact_geom_ids[0]
            if int(contact.geom1) == ball_id:
                return g1, g2
            return g2, g1
        if "ball" in g1:
            return g1, g2
        return g2, g1

    def _collect_contact_events(self, step: int) -> None:
        """采集当前物理步的全部球—拍接触事件（T6 契约）。

        扫描 ``env.data.ncon`` 全部接触，命中球—拍对的接触逐个记录为
        :class:`ContactEvent`（同一步多事件全保留）；同一物理步的拍速只
        取一次。首个 ``physical_contact and surface_kind == "face"`` 事件
        决定 ``physical_hit / physical_hit_type / physical_hit_step``。

        Args:
            step: 当前执行步编号（时间 = ``step * dt``）。
        """
        n_contacts = int(self._env.data.ncon)
        if n_contacts <= 0:
            return
        speed = self._current_contact_speed()
        for ci in range(n_contacts):
            c = self._env.data.contact[ci]
            if not self._is_ball_racket_pair(c):
                continue
            ball_name, racket_name = self._contact_geom_names(c)
            distance = float(c.dist)
            physical = distance <= 0.0
            surface_kind = classify_surface_kind(racket_name)
            event = ContactEvent(
                step=int(step),
                time_s=float(step) * float(self._dt),
                ball_geom_name=str(ball_name),
                other_geom_name=str(racket_name),
                surface_kind=surface_kind,
                distance_m=distance,
                face_speed_m_s=speed,
                physical_contact=physical,
                hit_type=classify_contact_hit_type(
                    surface_kind, physical, speed,
                ),
            )
            self.contact_events.append(event)
            if (
                not self.physical_hit
                and event.physical_contact
                and event.surface_kind == "face"
            ):
                self.physical_hit = True
                self.physical_hit_type = event.hit_type
                self.physical_hit_step = int(step)

    def contact_summary(self) -> dict:
        """汇总新旧命中口径（供 ``metrics``/产物写入）。

        Returns:
            扁平字典：``physical_hit`` / ``physical_hit_type`` /
            ``physical_hit_step`` / ``legacy_hit`` / ``legacy_hit_type`` /
            ``legacy_hit_rule`` / ``physical_hit_rule`` /
            ``legacy_vs_physical_divergence`` / ``contact_events``。
        """
        legacy_hit = bool(self.ball_was_hit)
        if self.active_contact:
            legacy_hit_type = "active"
        elif self.passive_contact:
            legacy_hit_type = "passive"
        else:
            legacy_hit_type = "miss"

        divergence: dict[str, object] | None = None
        if legacy_hit != self.physical_hit:
            divergence = {
                "legacy_hit": legacy_hit,
                "legacy_hit_type": legacy_hit_type,
                "physical_hit": self.physical_hit,
                "physical_hit_type": self.physical_hit_type,
                "reason": (
                    "legacy_hit_without_physical_face_contact"
                    if legacy_hit
                    else "physical_hit_without_legacy_hit"
                ),
            }
        elif self.physical_hit and legacy_hit_type != self.physical_hit_type:
            divergence = {
                "legacy_hit": legacy_hit,
                "legacy_hit_type": legacy_hit_type,
                "physical_hit": self.physical_hit,
                "physical_hit_type": self.physical_hit_type,
                "reason": (
                    "hit_type_mismatch_legacy_first_contact_vs_first_"
                    "physical_face"
                ),
            }

        return {
            "physical_hit": self.physical_hit,
            "physical_hit_type": self.physical_hit_type,
            "physical_hit_step": self.physical_hit_step,
            "legacy_hit": legacy_hit,
            "legacy_hit_type": legacy_hit_type,
            "legacy_hit_rule": LEGACY_HIT_RULE,
            "physical_hit_rule": PHYSICAL_HIT_RULE,
            "legacy_vs_physical_divergence": divergence,
            "contact_events": [event.to_dict() for event in self.contact_events],
        }

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
            # R07：每个物理子步都记录球速度（取 env 真实 qvel，与位置同帧）
            self.ball_vel_history.append(env.get_ball_vel().copy())
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
            # 球-拍接近门控几何事实（Task 5：随产物记录，便于跨机器人对照）
            "proximity": {
                "ball_radius": self.ball_radius,
                "racket_radius": self.racket_radius,
                "proximity_margin": self.proximity_margin,
                "ball_near_threshold": self._ball_near_threshold,
            },
            # T6：新旧命中口径独立汇总（含全部接触事件）
            **self.contact_summary(),
        }
