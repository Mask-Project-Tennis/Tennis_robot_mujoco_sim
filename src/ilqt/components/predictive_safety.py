"""预测性安全滤波 — beta 递降 + X 平面墙预判 + 紧急制动。

实现 ``SafetyComponent`` Protocol。对齐 V11 主循环（``rm65_mpc_v11.py``
1925-2004 行）的硬约束两层结构：

  层 1 — X 平面墙预判：机械臂连杆不越过身体中线 (X ≤ -0.1)，
          用 ``apply_control_beta`` 做 [1.0, 0.6, 0.3, 0.0] 递降，
          取第一个预测状态满足 X 墙的控制。
  层 2 — 关节约束安全滤波：``check_one_step_feasibility`` 检查
          q 限位 / qdot 制动感知 / TCP 速度。失败时对层 1 结果做
          [0.8, 0.6, 0.4, 0.2, 0.0] 递降；全失败 → 紧急制动。

仿真 (RM65Env) 与真机 (PlanningEnv) 具有同等预测能力（都实现
``step_from_state``），故本组件两者通用。
"""

from __future__ import annotations

import logging
from typing import Optional

import mujoco
import numpy as np
from numpy.typing import NDArray

from src.ilqt.robot_limits import RobotLimits, check_one_step_feasibility
from src.ilqt.utils import apply_control_beta

logger = logging.getLogger(__name__)

# X 平面墙：机械臂连杆不可越过的身体中线（世界坐标系 X 坐标上限）
_X_WALL_LIMIT: float = -0.1

# X 平面墙检查的刚体名（对齐 V11 ``_hard_x_body_ids``）
_X_WALL_BODY_NAMES: tuple[str, ...] = (
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

    全部 beta 失败时进入紧急制动：位置模式 hold 当前 q，力矩模式施加
    ``-20*qdot`` 阻尼力矩 —— 二者均视为 ``is_safe=True``（episode 不中止，
    与 V11 一致）。

    Attributes:
        k_hit_remaining: 距击球剩余步数。由上层每拍更新，用于终段豁免
            (``k_hit ≤ terminal_exempt_steps`` 时跳过 qdot/TCP 检查)。
            默认 99 = 全程无豁免。
        emergency_stop_count: 累计紧急制动次数（供诊断读取）。

    Args:
        env: RobotEnv 实例（须有 ``step_from_state`` / ``set_arm_state`` /
            ``update_kinematics`` / ``get_ball_state`` / ``set_ball_state`` /
            ``dt`` / ``actuator_mode``）。
        robot_limits: RobotLimits 实例。
        is_position_mode: 位置模式时用 q 插值，力矩模式用力矩缩放。
        arm_nq: 臂关节数（默认 6）。
        enable_x_wall: 是否启用 X 平面墙预判（默认 True，对齐 V11）。
    """

    def __init__(
        self,
        env: object,
        robot_limits: RobotLimits,
        is_position_mode: bool = False,
        arm_nq: int = 6,
        enable_x_wall: bool = True,
    ) -> None:
        """初始化预测性安全滤波器。

        Args:
            env: 规划/仿真环境。
            robot_limits: 关节约束参数。
            is_position_mode: 是否位置模式。
            arm_nq: 臂关节数。
            enable_x_wall: 是否启用 X 平面墙。
        """
        self._env = env
        self._limits = robot_limits
        self._is_position_mode = is_position_mode
        self._nq = arm_nq
        self._enable_x_wall = enable_x_wall

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

        # 惰性计算 X 平面墙 body IDs
        self._x_wall_body_ids: Optional[list[int]] = None

    # ── 内部辅助 ───────────────────────────────────────────────────────────

    def _resolve_x_wall_body_ids(self) -> list[int]:
        """惰性解析 X 平面墙刚体 IDs（从模型按名查找）。"""
        if self._x_wall_body_ids is None:
            model = self._env.model  # type: ignore[attr-defined]
            ids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)  # type: ignore[attr-defined]
                for name in _X_WALL_BODY_NAMES
            ]
            self._x_wall_body_ids = [i for i in ids if i >= 0]
        return self._x_wall_body_ids

    def _make_step_predictor(self, ball_ref: tuple[NDArray[np.floating], NDArray[np.floating]]):
        """构建 check_one_step_feasibility 用的 step_predictor 闭包。

        预测后立即恢复虚拟/MuJoCo 球状态，避免逐步预测污染球真值
        （对齐 V11 ``_safety_step``）。
        """

        def _step_predictor(
            x: NDArray[np.floating], u_val: NDArray[np.floating]
        ) -> NDArray[np.floating]:
            x_next = self._env.step_from_state(x, u_val)  # type: ignore[attr-defined]
            self._env.set_ball_state(*ball_ref)  # type: ignore[attr-defined]
            return x_next

        return _step_predictor

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
        env = self._env
        # 预测下一步（先复位臂状态，避免上次预测残留）
        env.set_arm_state(arm_state)  # type: ignore[attr-defined]
        env.set_ball_state(*ball_ref)  # type: ignore[attr-defined]
        x_pred = env.step_from_state(arm_state, u_try)  # type: ignore[attr-defined]
        env.set_arm_state(x_pred)  # type: ignore[attr-defined]
        env.update_kinematics()  # type: ignore[attr-defined]
        data = env.data  # type: ignore[attr-defined]
        ok = all(data.xpos[bid, 0] <= _X_WALL_LIMIT for bid in self._resolve_x_wall_body_ids())
        # 恢复
        env.set_arm_state(arm_state)  # type: ignore[attr-defined]
        env.set_ball_state(*ball_ref)  # type: ignore[attr-defined]
        return ok

    # ── Protocol 接口 ──────────────────────────────────────────────────────

    def filter(
        self, u_cmd: NDArray[np.floating], arm_state: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], bool]:
        """安全滤波 → (safe_u, is_safe)。

        流程对齐 V11 1925-2004 行：
          1. X 平面墙 beta 递降 → u_xsafe
          2. check_one_step_feasibility(u_xsafe) → 通过则返回
          3. 关节约束 beta 递降 → 第一个通过者返回
          4. 全失败 → 紧急制动（is_safe=True）

        预测过程会修改 env 内部状态，本方法在返回前完整恢复臂/球状态，
        保证调用对 env 无副作用。

        Args:
            u_cmd: 待滤波的控制指令 (NU,)。
            arm_state: 当前臂状态 [q, qdot] (2*NQ,)。

        Returns:
            (safe_u, is_safe)。紧急制动亦返回 is_safe=True（不中止 episode）。
        """
        env = self._env
        q = arm_state[: self._nq]
        ctrl_range = env.model.actuator_ctrlrange[: self._nu]  # type: ignore[attr-defined]
        ctrl_lo = ctrl_range[:, 0]
        ctrl_hi = ctrl_range[:, 1]

        # 保存 env 状态（预测会污染，结束前恢复）
        ball_save = env.get_ball_state()  # type: ignore[attr-defined]

        def _restore() -> None:
            env.set_arm_state(arm_state)  # type: ignore[attr-defined]
            env.set_ball_state(*ball_save)  # type: ignore[attr-defined]

        step_predictor = self._make_step_predictor(ball_save)

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

        # ---- 层 2: 关节约束安全滤波 ----
        _restore()
        ok, reason = check_one_step_feasibility(
            arm_state,
            u_xsafe,
            self._limits,
            self._dt,
            step_predictor=step_predictor,
            k_hit_remaining=self.k_hit_remaining,
            env=env,  # type: ignore[arg-type]
        )
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
            ok_s, _ = check_one_step_feasibility(
                arm_state,
                u_try,
                self._limits,
                self._dt,
                step_predictor=step_predictor,
                k_hit_remaining=self.k_hit_remaining,
                env=env,  # type: ignore[arg-type]
            )
            if ok_s:
                logger.info("[SAFETY_FILTER] beta=%.1f: %s", beta_s, reason)
                _restore()
                return u_try, True

        # ---- 全部失败 → 紧急制动 ----
        self.emergency_stop_count += 1
        logger.warning(
            "[EMERGENCY_STOP] 安全滤波全 beta 失败: %s, safe_hold 阻尼制动", reason
        )
        _restore()
        if self._is_position_mode:
            # 位置模式：hold 当前 q（视为安全）
            return q.copy(), True
        # 力矩模式：阻尼力矩 -20*qdot（视为安全）
        return -20.0 * arm_state[self._nq :], True
