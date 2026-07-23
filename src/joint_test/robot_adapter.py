"""统一适配 RM65Env / FakeRobot / RobotInterface 三种后端。"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from src.joint_test.safety import JointSafetyGuard
from src.joint_test.types import BackendType

logger = logging.getLogger(__name__)


class RobotAdapter:
    """抹平不同后端的 API 差异，对外提供 reset/step/get_q_qdot/emergency_stop。

    通过 BackendType 枚举派发后端行为（避免循环导入):
        - SIM (RM65Env): step() 直接步进仿真
        - FAKE (FakeRobot): 直接写入内部状态 _q
        - REAL (RobotInterface): send_joint_command 下发给真机

    Args:
        robot: 后端机器人实例（RM65Env | FakeRobot | RobotInterface）。
        backend: 后端类型枚举。
        safety_guard: 安全防护器（真机模式必传，仿真可 None）。
    """

    def __init__(
        self,
        robot: Any,  # duck-typing: RM65Env | FakeRobot | RobotInterface（三后端无公共 Protocol）
        backend: BackendType,
        safety_guard: JointSafetyGuard | None = None,
    ) -> None:
        """初始化适配器，通过 BackendType 枚举派发后端行为。"""
        self._robot = robot
        self._backend = backend
        self._safety = safety_guard
        self._cached_q: np.ndarray | None = None

    def reset(self, q0: np.ndarray) -> None:
        """重置到初始位姿 q0。

        真机模式下不应调用 reset()（应使用 CLI 层 home_to_pose()），
        若误调用则打印警告并跳过。

        对于 RM65Env: 直接调 env.reset(q0)。
        对于 FakeRobot: 直接写入内部状态 _q（绕过 send_joint_command 的速度计算）。
        对于 RobotInterface: 多次下发 q0 让真机回到位。

        Args:
            q0: 目标初始关节角度 (6,)，弧度。
        """
        q0 = np.asarray(q0, dtype=float)

        # 清空缓存，确保下次 step 重新初始化
        self._cached_q = None

        # 真机模式 guard：真机不应通过 reset() 回零，应由 CLI 层 home_to_pose() 处理
        if self._backend == BackendType.REAL:
            logger.warning("真机模式下不应调用 reset()，请使用 CLI 层 home_to_pose()")
            return

        if self._backend == BackendType.SIM:
            self._robot.reset(q0)
        elif self._backend == BackendType.FAKE:
            # FakeRobot: 直接写内部状态
            self._robot._q = q0.copy()  # type: ignore[attr-defined]
            if hasattr(self._robot, "_qdot"):
                self._robot._qdot = np.zeros(6)  # type: ignore[attr-defined]
        else:
            # RobotInterface: 多次下发让真机回位
            for _ in range(20):
                self._robot.send_joint_command(q0)

    def step(self, q_des: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """下发指令并步进，返回 (q_actual, qdot_actual, wall_ts)。

        仿真模式: step() 后读状态，wall_ts 用 perf_counter()
        真机模式: send_joint_command() + get_arm_state()，perf_counter() 紧随读取捕获

        若有 safety_guard，先裁剪指令再下发。
        非仿真后端返回非零状态码时立即急停并抛 RuntimeError，
        避免通信失败被静默吞掉继续下发后续指令。

        Args:
            q_des: 期望关节角度 (6,)，弧度。

        Returns:
            (q (6,), qdot (6,), wall_ts (float)) — wall_ts 为 perf_counter() 时间戳(s)

        Raises:
            RuntimeError: 当 send_joint_command 返回非零状态码。
        """
        q_des = np.asarray(q_des, dtype=float)

        # 安全裁剪（用上一步缓存的状态，省去一次 get_arm_state 调用）
        if self._safety is not None:
            if self._cached_q is None:
                try:
                    state = self._robot.get_arm_state()
                    self._cached_q = state[:6].copy()
                except Exception:
                    try:
                        self.emergency_stop()
                    except Exception as estop_err:
                        logger.error("急停也失败: %s", estop_err)
                    raise RuntimeError("读取机器人状态失败") from None
            assert self._cached_q is not None  # 初始化或上次 step 已设置
            q_des = self._safety.clip_command(q_des, self._cached_q)

        if self._backend == BackendType.SIM:
            self._robot.step(q_des)
        else:
            ret = self._robot.send_joint_command(q_des)
            if ret != 0:
                # 通信失败：立即急停并抛错，避免静默失败导致后续步继续下发
                self.emergency_stop()
                raise RuntimeError(f"机器人通信失败，错误码 {ret}")

        q, qdot = self.get_q_qdot()
        self._cached_q = q.copy()
        wall_ts = time.perf_counter()
        return q, qdot, wall_ts

    def get_q_qdot(self) -> tuple[np.ndarray, np.ndarray]:
        """返回当前 (q, qdot)。

        Returns:
            (q (6,), qdot (6,))，均为副本（防止外部修改内部状态）。
        """
        state = self._robot.get_arm_state()  # (12,) = [q(6), qdot(6)]
        return state[:6].copy(), state[6:].copy()

    def emergency_stop(self) -> None:
        """紧急停止，传播到底层机器人。"""
        if hasattr(self._robot, "emergency_stop"):
            self._robot.emergency_stop()
        logger.warning("紧急停止已触发")

    @property
    def backend(self) -> BackendType:
        """后端类型。"""
        return self._backend

    @property
    def safety_guard(self) -> JointSafetyGuard | None:
        """暴露的安全防护器（供外部读取，不应直接访问 _safety）。"""
        return self._safety
