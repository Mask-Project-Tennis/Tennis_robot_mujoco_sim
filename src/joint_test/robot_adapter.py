"""统一适配 RM65Env / FakeRobot / RobotInterface 三种后端。"""
from __future__ import annotations

import logging

import numpy as np

from src.joint_test.safety import JointSafetyGuard
from src.joint_test.types import BackendType

logger = logging.getLogger(__name__)


class RobotAdapter:
    """抹平不同后端的 API 差异，对外提供 reset/step/get_q_qdot/emergency_stop。

    通过 hasattr 检测后端类型（避免循环导入):
        - RM65Env: 有 set_arm_state 和 step 方法 → 直接调用
        - FakeRobot/RobotInterface: 有 send_joint_command → 协议接口

    Args:
        robot: 后端机器人实例（RM65Env | FakeRobot | RobotInterface）。
        backend: 后端类型枚举。
        safety_guard: 安全防护器（真机模式必传，仿真可 None）。
    """

    def __init__(
        self,
        robot,  # RM65Env | FakeRobot | RobotInterface（避免循环导入，用 duck typing）
        backend: BackendType,
        safety_guard: JointSafetyGuard | None = None,
    ) -> None:
        """初始化适配器。"""
        self._robot = robot
        self._backend = backend
        self._safety = safety_guard
        # 通过属性检测后端类型（不用 isinstance 避免循环导入）
        self._is_env = (
            hasattr(robot, "set_arm_state")
            and hasattr(robot, "step")
            and hasattr(robot, "reset")
            and not hasattr(robot, "send_joint_command")
        )

    def reset(self, q0: np.ndarray) -> None:
        """重置到初始位姿 q0。

        对于 RM65Env: 直接调 env.reset(q0)。
        对于 FakeRobot: 直接写入内部状态 _q（绕过 send_joint_command 的速度计算）。
        对于 RobotInterface: 多次下发 q0 让真机回到位。

        Args:
            q0: 目标初始关节角度 (6,)，弧度。
        """
        q0 = np.asarray(q0, dtype=float)
        if self._is_env:
            self._robot.reset(q0)
        elif hasattr(self._robot, "_q"):
            # FakeRobot: 直接写内部状态
            self._robot._q = q0.copy()  # type: ignore[attr-defined]
            if hasattr(self._robot, "_qdot"):
                self._robot._qdot = np.zeros(6)  # type: ignore[attr-defined]
        else:
            # RobotInterface: 多次下发让真机回位
            for _ in range(20):
                self._robot.send_joint_command(q0)

    def step(self, q_des: np.ndarray) -> None:
        """下发指令并步进。

        若有 safety_guard，先裁剪指令再下发。
        非仿真后端返回非零状态码时立即急停并抛 RuntimeError，
        避免通信失败被静默吞掉继续下发后续指令。

        Args:
            q_des: 期望关节角度 (6,)，弧度。

        Raises:
            RuntimeError: 当 send_joint_command 返回非零状态码。
        """
        q_des = np.asarray(q_des, dtype=float)

        # 安全裁剪（在读取当前状态之前）
        if self._safety is not None:
            current_state = self._robot.get_arm_state()
            q_current = current_state[:6]
            q_des = self._safety.clip_command(q_des, q_current)

        if self._is_env:
            self._robot.step(q_des)
        else:
            ret = self._robot.send_joint_command(q_des)
            if ret != 0:
                # 通信失败：立即急停并抛错，避免静默失败导致后续步继续下发
                self.emergency_stop()
                raise RuntimeError(f"机器人通信失败，错误码 {ret}")

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
