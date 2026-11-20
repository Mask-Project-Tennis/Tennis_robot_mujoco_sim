"""预测性安全滤波 — beta 递降 + X 平面墙预判 + 紧急制动 + 可选禁止对碰撞检查。

实现 ``SafetyComponent`` Protocol。对齐 V11 主循环（``rm65_mpc_v11.py``
1925-2004 行）的硬约束两层结构：

  层 1 — X 平面墙预判：机械臂连杆不越过身体中线 (X ≤ -0.1)，
          用 ``apply_control_beta`` 做 [1.0, 0.6, 0.3, 0.0] 递降，
          取第一个预测状态满足 X 墙的控制。
  层 2 — 关节约束安全滤波：``strict_braking_check`` 检查
          q 限位 / qdot 制动感知 / TCP 速度。失败时对层 1 结果做
          [0.8, 0.6, 0.4, 0.2, 0.0] 递降；全失败 → 紧急制动。

阶段 3 · Task 3.7 扩展（R1D 多机器人路径）：

  - **可选碰撞检查注入**（``contact_checker``）：注入后在独立数据上做
    显式禁止对的预测接触检查（3.7 ``ForbiddenContactChecker``），与
    层 2 复用同一 beta 递降策略 —— 力矩 beta 与位置插值的每个候选都
    经**实际预测终态**逐一验证；初态（当前位形）已穿透时立即判不安全。
    R1D 世界没有 RM65 的身体中线语义，应 ``enable_x_wall=False`` 并
    注入检查器（X 墙与禁止对检查正交，可同时启用）；
  - **"尝试了制动" ≠ "已验证安全"**：注入碰撞检查时，全部候选失败
    返回 ``is_safe=False``（EpisodeRunner 据此停止回合），不给未验证
    的紧急制动无条件放行；**无注入的旧调用保持 RM65 语义**（紧急制动
    视为 ``is_safe=True``，episode 不中止，回归见 test_sim_components）；
  - X 平面墙 body 名在启用时**全部必须存在**（构造即解析，缺失抛
    ValueError），不再静默丢弃未知名而得到空列表空真放行；
  - 控制裁剪范围优先取 ``env.control_range``（绑定序，n 维），无该
    属性的旧环境回退 ``model.actuator_ctrlrange[:NU]``（RM65 前缀布局）。

仿真 (RM65Env) 与真机 (PlanningEnv) 具有同等预测能力（都实现
``step_from_state``），故本组件两者通用。

能力声明（不宣称的部分）：本滤波是**离散一步**预测检查 —— 每个候选
只验证 ``step_from_state`` 的下一步终态，不保证步间连续轨迹无碰撞
（步内扫掠穿透可能不被观测）；它是仿真闭环中的保护性拒绝条件，
**不构成真机安全认证**。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import mujoco
import numpy as np
from numpy.typing import NDArray

from src.ilqt.robot_limits import RobotLimits, strict_braking_check
from src.ilqt.utils import apply_control_beta

if TYPE_CHECKING:
    # 仅类型标注：注入对象为 duck-type（须有 has_contact_at_q(q) -> bool），
    # 运行时不强制 isinstance，避免组件层对 src.sim 的硬依赖
    from src.sim.collision_checks import ForbiddenContactChecker

logger = logging.getLogger(__name__)

# X 平面墙：机械臂连杆不可越过的身体中线（世界坐标系 X 坐标上限）
_X_WALL_LIMIT: float = -0.1

# X 平面墙检查的刚体名（公共常量 — V12 和 PredictiveSafetyFilter 共用）
X_WALL_BODY_NAMES: tuple[str, ...] = (
    "r_link1",
    "r_link2",
    "r_link3",
    "r_link4",
    "r_link5",
    "r_link6",
    "r_flange",
    "r_racket_body",
)


class PredictiveSafetyFilter:
    """预测性安全滤波 — 使用 ``env.step_from_state`` 逐步预测。

    实现 ``SafetyComponent`` Protocol。

    beta 递降策略对齐 V11：

      - X 平面墙层: ``[1.0, 0.6, 0.3, 0.0]``。
      - 关节约束层: ``[0.8, 0.6, 0.4, 0.2, 0.0]``（不含 1.0，因层 1
        输出已在 beta=1.0 下做过关节检查）。
      - 力矩模式: ``u_try = beta * u``；位置模式: ``u_try = q + beta*(u-q)``。
      - 注入碰撞检查时每个候选额外验证预测终态无禁止对接触。

    全部 beta 失败时进入紧急制动（位置模式 hold 当前 q，力矩模式施加
    ``-20*qdot`` 阻尼力矩）：

      - **无注入（RM65 旧语义）**：制动视为 ``is_safe=True``（episode
        不中止，与 V11 一致）；
      - **注入 contact_checker（R1D）**：制动只是"尝试了制动"，未经
        验证 → 返回 ``is_safe=False``（停止回合），不给未验证制动
        无条件放行。

    Attributes:
        k_hit_remaining: 距击球剩余步数。由上层每拍更新，用于终段豁免
            (``k_hit ≤ terminal_exempt_steps`` 时跳过 qdot/TCP 检查)。
            默认 99 = 全程无豁免。
        emergency_stop_count: 累计紧急制动次数（供诊断读取）。

    Args:
        env: RobotEnv 实例（须有 ``step_from_state`` / ``set_arm_state`` /
            ``update_kinematics`` / ``get_ball_state`` / ``set_ball_state`` /
            ``dt`` / ``actuator_mode``）。
        robot_limits: RobotLimits 实例（维数 n 通用）。
        is_position_mode: 位置模式时用 q 插值，力矩模式用力矩缩放。
        arm_nq: 臂关节数（默认 6；R1D 为 7）。
        enable_x_wall: 是否启用 X 平面墙预判（默认 True，对齐 V11）。
            启用时 RM65 body 名必须全部存在于模型（缺失即构造失败）。
        contact_checker: 可选禁止对碰撞检查器（须有
            ``has_contact_at_q(q) -> bool``，如 3.7
            ``ForbiddenContactChecker.from_env(env, pairs)``）。None =
            不注入（旧 RM65 默认行为，语义不变）。

    Raises:
        ValueError: enable_x_wall=True 且模型缺少任一 X 墙 body 名
            （不再空列表空真放行）。
    """

    def __init__(
        self,
        env: object,
        robot_limits: RobotLimits,
        is_position_mode: bool = False,
        arm_nq: int = 6,
        enable_x_wall: bool = True,
        contact_checker: ForbiddenContactChecker | None = None,
        *,
        x_wall_body_names: tuple[str, ...] | None = None,
        x_wall_limit: float | None = None,
    ) -> None:
        """初始化预测性安全滤波器。

        Args:
            env: 规划/仿真环境。
            robot_limits: 关节约束参数。
            is_position_mode: 是否位置模式。
            arm_nq: 臂关节数。
            enable_x_wall: 是否启用 X 平面墙。
            contact_checker: 可选禁止对碰撞检查器（None = 旧默认）。
            x_wall_body_names: T4 新增 keyword-only。X 平面墙检查的刚体名
                元组；None 时保持 RM65 共享常量 ``X_WALL_BODY_NAMES``
                （旧行为逐位不变）。供 V12RobotBundle 的 ``x_wall`` 配置
                显式映射“世界 X 平面 + 检查刚体”，不在组件层硬编码单一
                机器人布局。
            x_wall_limit: T4 新增 keyword-only。世界坐标系 X 坐标上限
                （米）；None 时保持 RM65 常量 ``-0.1``。R1D 的机体坐标
                中线保护由 contact_checker 承担（``enable_x_wall=False``），
                此处只表达“世界 X 平面”这一语义，不照搬 RM65 数值。

        Raises:
            ValueError: enable_x_wall=True 且 X 墙 body 名缺失。
        """
        self._env = env
        self._limits = robot_limits
        self._is_position_mode = is_position_mode
        self._nq = arm_nq
        self._enable_x_wall = enable_x_wall
        self._contact_checker = contact_checker
        # X 墙语义参数（None → RM65 旧常量，T0 行为逐位不变）
        self._x_wall_body_names: tuple[str, ...] = (
            X_WALL_BODY_NAMES if x_wall_body_names is None else tuple(x_wall_body_names)
        )
        self._x_wall_limit: float = (
            _X_WALL_LIMIT if x_wall_limit is None else float(x_wall_limit)
        )

        # dt / NU 从环境读取（运行时固定）
        self._dt: float = float(env.dt)  # type: ignore[attr-defined]
        nu = getattr(env, "NU", arm_nq)
        self._nu: int = int(nu)

        # beta 递降表（对齐 V11）
        self._beta_list_x: list[float] = [1.0, 0.6, 0.3, 0.0]
        self._beta_list_safety: list[float] = [0.8, 0.6, 0.4, 0.2, 0.0]

        # 运行时状态（上层每拍更新）
        self.k_hit_remaining: int = 99
        self.emergency_stop_count: int = 0

        # X 平面墙 body IDs：启用时构造即解析（缺失即失败，不空真放行）
        self._x_wall_body_ids: Optional[list[int]] = None
        if self._enable_x_wall:
            self._x_wall_body_ids = self._resolve_x_wall_body_ids()

    # ── 内部辅助 ───────────────────────────────────────────────────────────

    def _resolve_x_wall_body_ids(self) -> list[int]:
        """解析 X 平面墙刚体 IDs（按名查找；任一缺失即失败）。

        旧实现对未知名静默丢弃可得空列表，``all()`` 空真放行 —— 已废弃：
        名单中任一 body 不存在说明模型不是 RM65 布局（或名单漂移），
        必须显式失败，不允许以空集冒充 X 墙保护。

        Raises:
            ValueError: 任一 X 墙 body 名在模型中不存在。
        """
        model = self._env.model  # type: ignore[attr-defined]
        ids: list[int] = []
        missing: list[str] = []
        for name in self._x_wall_body_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                missing.append(name)
            else:
                ids.append(int(bid))
        if missing:
            raise ValueError(
                f"X 平面墙 body 名缺失 {missing}（模型非 RM65 布局；"
                "缺失不得空真放行 —— R1D 等多机器人场景应 "
                "enable_x_wall=False 并注入 contact_checker 显式禁止对检查）"
            )
        return ids

    def _control_range(self) -> NDArray[np.floating]:
        """读取控制裁剪范围 (NU,2)：优先 env.control_range（绑定序）。"""
        rng = getattr(self._env, "control_range", None)
        if rng is not None:
            return np.asarray(rng, dtype=np.float64)
        model = self._env.model  # type: ignore[attr-defined]
        return np.asarray(
            model.actuator_ctrlrange[: self._nu], dtype=np.float64
        )

    def _braking_control(
        self, arm_state: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """紧急制动控制：位置模式 hold 当前 q；力矩模式 -20*qdot 阻尼。"""
        if self._is_position_mode:
            return np.array(arm_state[: self._nq], dtype=np.float64)
        return -20.0 * np.asarray(arm_state[self._nq :], dtype=np.float64)

    def _check_candidate(
        self,
        u_try: NDArray[np.floating],
        arm_state: NDArray[np.floating],
        ball_ref: tuple[NDArray[np.floating], NDArray[np.floating]],
    ) -> tuple[bool, str]:
        """预测并验证单个候选控制（关节约束 + 可选禁止对接触）。

        与旧 ``check_one_step_feasibility`` 路径逐步等价（step_from_state
        → 恢复球真值 → strict_braking_check），并在注入检查器时对**该
        候选的实际预测终态**追加禁止对接触验证 —— 力矩 beta 与位置插值
        候选均经此验证，不经验证的控制不放行。

        Args:
            u_try: 候选控制 (NU,)。
            arm_state: 当前臂状态 (2*NQ,)。
            ball_ref: 预测后用于恢复的球真值。

        Returns:
            (ok, reason)。
        """
        env = self._env
        # 对齐旧 _restore()：预测前复位臂/球，避免上次预测残留
        env.set_arm_state(arm_state)  # type: ignore[attr-defined]
        env.set_ball_state(*ball_ref)  # type: ignore[attr-defined]
        x_pred = env.step_from_state(arm_state, u_try)  # type: ignore[attr-defined]
        env.set_ball_state(*ball_ref)  # type: ignore[attr-defined]

        if self._contact_checker is not None:
            if self._contact_checker.has_contact_at_q(x_pred[: self._nq]):
                return False, "forbidden contact predicted"
        return strict_braking_check(
            arm_state,
            x_pred,
            u_try,
            self._limits,
            self._dt,
            k_hit_remaining=self.k_hit_remaining,
            env=env,  # type: ignore[arg-type]
            actuator_mode=getattr(env, "actuator_mode", 0),
        )

    def _check_x_wall(
        self,
        u_try: NDArray[np.floating],
        arm_state: NDArray[np.floating],
        ball_ref: tuple[NDArray[np.floating], NDArray[np.floating]],
    ) -> bool:
        """预测一步并检查所有 X 平面墙刚体是否满足 X ≤ -0.1。

        Args:
            u_try: 候选控制 (NU,)。
            arm_state: 当前臂状态 (2*NQ,)。
            ball_ref: 用于恢复的球状态。

        Returns:
            True 表示预测状态满足 X 平面墙。
        """
        assert self._x_wall_body_ids is not None  # 启用时构造即解析
        env = self._env
        # 预测下一步（先复位臂状态，避免上次预测残留）
        env.set_arm_state(arm_state)  # type: ignore[attr-defined]
        env.set_ball_state(*ball_ref)  # type: ignore[attr-defined]
        x_pred = env.step_from_state(arm_state, u_try)  # type: ignore[attr-defined]
        env.set_arm_state(x_pred)  # type: ignore[attr-defined]
        env.update_kinematics()  # type: ignore[attr-defined]
        data = env.data  # type: ignore[attr-defined]
        ok = all(
            data.xpos[bid, 0] <= self._x_wall_limit
            for bid in self._x_wall_body_ids
        )
        # 恢复
        env.set_arm_state(arm_state)  # type: ignore[attr-defined]
        env.set_ball_state(*ball_ref)  # type: ignore[attr-defined]
        return ok

    # ── Protocol 接口 ──────────────────────────────────────────────────────

    def filter(
        self, u_cmd: NDArray[np.floating], arm_state: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], bool]:
        """安全滤波 → (safe_u, is_safe)。

        流程对齐 V11 1925-2004 行（Task 3.7 扩展注入路径）：
          0. [注入] 初态已存在禁止对接触 → 立即不安全（任何候选都无从
             验证安全），返回制动控制 + is_safe=False；
          1. X 平面墙 beta 递降 → u_xsafe；
          2. 候选 u_xsafe 预测验证（关节约束 + [注入] 禁止对接触）
             → 通过则返回；
          3. 关节约束 beta 递降 → 每个候选实际预测验证，第一个通过者返回；
          4. 全失败 → 紧急制动：无注入 is_safe=True（旧 RM65 语义）；
             注入时 is_safe=False（尝试了制动 ≠ 已验证安全，停止回合）。

        预测过程会修改 env 内部状态，本方法在返回前完整恢复臂/球状态，
        保证调用对 env 无副作用；注入的碰撞检查器使用独立数据，不改
        live 球/状态/时钟。

        Args:
            u_cmd: 待滤波的控制指令 (NU,)。
            arm_state: 当前臂状态 [q, qdot] (2*NQ,)。

        Returns:
            (safe_u, is_safe)。无注入时紧急制动亦返回 is_safe=True
            （不中止 episode，旧语义）；注入时全候选失败/初态穿透返回
            is_safe=False（停止回合）。
        """
        env = self._env
        q = np.asarray(arm_state[: self._nq], dtype=np.float64)
        ctrl_range = self._control_range()
        ctrl_lo = ctrl_range[:, 0]
        ctrl_hi = ctrl_range[:, 1]

        # 保存 env 状态（预测会污染，结束前恢复）
        ball_save = env.get_ball_state()  # type: ignore[attr-defined]

        def _restore() -> None:
            env.set_arm_state(arm_state)  # type: ignore[attr-defined]
            env.set_ball_state(*ball_save)  # type: ignore[attr-defined]

        # ---- 层 0: [注入] 初态禁止对接触检查 ----
        if self._contact_checker is not None:
            if self._contact_checker.has_contact_at_q(q):
                self.emergency_stop_count += 1
                logger.warning(
                    "[FORBIDDEN_CONTACT] 初态已存在禁止对接触 → 不安全，"
                    "停止回合（任何候选控制都无法验证安全）"
                )
                _restore()
                return self._braking_control(arm_state), False

        # ---- 层 1: X 平面墙预判 ----
        u_xsafe = np.clip(
            apply_control_beta(u_cmd, q, 1.0, self._is_position_mode),
            ctrl_lo,
            ctrl_hi,
        )
        if self._enable_x_wall:
            for beta_x in self._beta_list_x:
                u_try = np.clip(
                    apply_control_beta(u_cmd, q, beta_x, self._is_position_mode),
                    ctrl_lo,
                    ctrl_hi,
                )
                if self._check_x_wall(u_try, arm_state, ball_save):
                    u_xsafe = u_try
                    break
            # 若所有 beta_x 都不满足 X 墙，u_xsafe 保持 beta=1.0 的裁剪值
            # （与 V11 一致：交由层 2 安全滤波继续处理）。

        # ---- 层 2: 候选逐一经实际预测验证（关节约束 + 可选接触） ----
        _restore()
        ok, reason = self._check_candidate(u_xsafe, arm_state, ball_save)
        if ok:
            _restore()
            return u_xsafe, True

        # beta 递降（从 0.8 开始，1.0 已在上方检查过）
        for beta_s in self._beta_list_safety:
            u_try = np.clip(
                apply_control_beta(u_xsafe, q, beta_s, self._is_position_mode),
                ctrl_lo,
                ctrl_hi,
            )
            _restore()
            ok_s, _ = self._check_candidate(u_try, arm_state, ball_save)
            if ok_s:
                logger.info("[SAFETY_FILTER] beta=%.1f: %s", beta_s, reason)
                _restore()
                return u_try, True

        # ---- 全部失败 → 紧急制动 ----
        self.emergency_stop_count += 1
        logger.warning(
            "[EMERGENCY_STOP] 安全滤波全 beta 失败: %s, %s",
            reason,
            "safe_hold 阻尼制动（注入模式：制动未经验证 → 不安全）"
            if self._contact_checker is not None
            else "safe_hold 阻尼制动",
        )
        _restore()
        if self._contact_checker is not None:
            # "尝试了制动" ≠ "已验证安全"：不安全的候选集合不因制动而
            # 变为安全 —— 返回不安全，由上层停止回合
            return self._braking_control(arm_state), False
        # 旧 RM65 默认语义：紧急制动视为安全（episode 不中止，与 V11 一致）
        return self._braking_control(arm_state), True
