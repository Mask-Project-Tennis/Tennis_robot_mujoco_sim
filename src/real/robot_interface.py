"""Realman RM-65B SDK 封装。

内部处理弧度↔角度转换，对外统一弧度制。
支持两种控制模式：
  - "ip"（默认）：rm_movej_follow，IP 通信关节跟随
  - "canfd"：rm_movej_canfd，CANFD 透传高跟随

SDK 条件导入：仿真环境无 SDK 时不报错。
"""

import logging
import time
from typing import Any

import numpy as np

from src.real.config import RealRobotConfig

logger = logging.getLogger(__name__)


class RobotInterface:
    """Realman RM-65B 机械臂接口封装。

    内部处理弧度↔角度转换，对外统一弧度制。
    通过依赖注入 SDK 实例支持测试（Mock 注入）。

    Args:
        config: 真机配置。
        arm: SDK RoboticArm 实例。None 时在 connect() 中创建。
             测试时注入 MockRoboticArm。
    """

    def __init__(
        self,
        config: RealRobotConfig,
        arm: Any = None,
    ) -> None:
        self._config = config
        self._arm = arm
        self._connected = False
        self._handle: Any = None
        self._last_q: np.ndarray | None = None
        self._last_time: float | None = None

    def connect(self) -> bool:
        """连接机械臂。

        Returns:
            连接是否成功。
        """
        if self._arm is None:
            try:
                from Robotic_Arm.rm_robot_interface import (
                    RoboticArm,
                    rm_thread_mode_e,
                )
            except ImportError:
                logger.error("Realman SDK 未安装，无法连接真机")
                return False
            self._arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

        self._handle = self._arm.rm_create_robot_arm(
            self._config.robot_ip, self._config.robot_port
        )
        if self._handle and self._handle.id > 0:
            self._connected = True
            logger.info(
                "机械臂连接成功 %s:%d (id=%d)",
                self._config.robot_ip,
                self._config.robot_port,
                self._handle.id,
            )
            return True
        logger.error("机械臂连接失败")
        return False

    def get_arm_state(self) -> np.ndarray:
        """读取右臂状态 [q(6), qdot(6)]，弧度制。

        关节角度从 SDK 读取（度），转换为弧度。
        关节速度通过数值微分计算（SDK 状态字典不可靠）。

        Returns:
            (12,) [q(6), qdot(6)]，弧度。
        """
        ret, joint_deg = self._arm.rm_get_joint_degree()
        if ret != 0:
            logger.error("rm_get_joint_degree 失败，错误码 %d", ret)
            return np.zeros(12)

        q_rad = np.radians(np.array(joint_deg[:6], dtype=np.float64))

        # 数值微分计算关节速度
        now = time.perf_counter()
        if self._last_q is not None and self._last_time is not None:
            dt = now - self._last_time
            qdot = (q_rad - self._last_q) / dt if dt > 1e-9 else np.zeros(6)
        else:
            qdot = np.zeros(6)

        self._last_q = q_rad.copy()
        self._last_time = now

        return np.concatenate([q_rad, qdot])

    def send_joint_command(self, q_desired: np.ndarray) -> int:
        """发送关节位置指令（弧度制）。

        内部转换为度，根据 control_mode 调用不同 SDK 接口：
          - "ip"：rm_movej_follow（默认）
          - "canfd"：rm_movej_canfd（高跟随透传）

        Args:
            q_desired: (6,) 目标关节角度，弧度。

        Returns:
            SDK 状态码（0=成功）。
        """
        q_deg = np.degrees(q_desired).tolist()

        if self._config.control_mode == "canfd":
            ret = self._arm.rm_movej_canfd(
                q_deg,
                follow=True,
                trajectory_mode=self._config.canfd_trajectory_mode,
                radio=self._config.canfd_smooth_radio,
            )
        else:
            ret = self._arm.rm_movej_follow(q_deg)

        if ret != 0:
            logger.warning("send_joint_command 返回错误码 %d", ret)
        return ret

    def emergency_stop(self) -> None:
        """急停 — 关节最快速度停止，轨迹不可恢复。"""
        self._arm.rm_set_arm_stop()
        logger.warning("急停已触发")

    def slow_stop(self) -> None:
        """缓停 — 在当前轨迹上平滑停止。"""
        self._arm.rm_set_arm_slow_stop()
        logger.info("缓停已触发")

    def disconnect(self) -> None:
        """断开机械臂连接。"""
        if self._arm is not None:
            self._arm.rm_delete_robot_arm()
            self._connected = False
            logger.info("机械臂连接已断开")
