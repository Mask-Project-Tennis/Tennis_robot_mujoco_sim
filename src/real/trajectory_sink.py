"""命令消费 Sink 链 — 消费 StepState，下发命令或记录数据。

设计模式: Composite（TeeSink）+ 责任链（顺序执行，短路失败）。
数据共享: StepState 是可变数据载体，RobotSink 填充 arm_state/tcp_pos，
          RecorderSink 消费（无需 fallback）。

典型链路:
    TeeSink([RobotSink, RecorderSink])
    → RobotSink 填充 state.arm_state / state.tcp_pos 并下发真机
    → RecorderSink 读取已填充的 state 记录轨迹
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from src.real.adaptive_timer import AdaptiveTimer
from src.real.robot_arm_protocol import RobotArmInterface
from src.real.safety_monitor import SafetyMonitor
from src.real.trajectory_recorder import TrajectoryRecorder
from src.real.trajectory_types import StepState

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.ilqt.robot_env_protocol import RobotEnv


@runtime_checkable
class CommandSink(Protocol):
    """命令消费接口：消费 StepState，下发命令。"""

    def send(self, state: StepState) -> bool:
        """处理一步命令。

        Args:
            state: 单步状态（含 q_desired, timestamp，arm_state/tcp_pos
                   由前序 Sink 填充）。

        Returns:
            True 表示成功，False 表示失败（安全检查不通过等）。
        """
        ...


class RobotSink:
    """下发到真机 + SafetyMonitor 检查 + AdaptiveTimer 节奏控制。

    职责:
        1. 读取机械臂状态并填充到 state.arm_state
        2. 计算 TCP 速度（jacp @ qdot）并填充 state.tcp_pos
        3. SafetyMonitor 安全检查
        4. 下发关节命令
        5. AdaptiveTimer 节奏控制
    """

    def __init__(
        self,
        robot: RobotArmInterface,
        safety: SafetyMonitor,
        timer: AdaptiveTimer,
        env: RobotEnv,
    ) -> None:
        """初始化 RobotSink。

        Args:
            robot: 机器人接口（RobotArmInterface Protocol 实现）。
            safety: 安全监控器。
            timer: 自适应计时器。
            env: 规划环境（用于 FK/Jacobian，需提供
                 set_arm_state/get_ee_jacp/get_ee_pos）。
        """
        self._robot = robot
        self._safety = safety
        self._timer = timer
        self._env = env

    def send(self, state: StepState) -> bool:
        """执行一步：读状态 → 填充 state → 安全检查 → 下发 → 节奏控制。

        流程:
            1. timer.tick_start()
            2. arm_state = robot.get_arm_state() → state.arm_state = arm_state
            3. env.set_arm_state(arm_state) → jacp = env.get_ee_jacp()
            4. tcp_speed = |jacp @ arm_state[6:]|（qdot = arm_state[6:]）
            5. state.tcp_pos = env.get_ee_pos()
            6. 安全检查失败 → robot.slow_stop(); return False
            7. robot.send_joint_command(state.q_desired)
               返回非 0 → robot.slow_stop(); return False（C1 修复）
            8. sleep_dt = timer.tick_end(); if > 0: time.sleep(sleep_dt)
            9. return True

        安全失败 / 通信异常 / SDK 非零返回码时不调 tick_end（立即返回，主循环 break）。
        通信异常时尝试缓停（best-effort），不重新抛出异常。

        Returns:
            True 表示成功下发，False 表示安全检查失败、通信异常或 SDK 返回非零
            （均已尝试缓停）。
        """
        self._timer.tick_start()
        try:
            arm_state = self._robot.get_arm_state()
            state.arm_state = arm_state

            self._env.set_arm_state(arm_state)
            jacp = self._env.get_ee_jacp()
            qdot = arm_state[6:]
            tcp_speed = float(np.linalg.norm(jacp @ qdot))

            state.tcp_pos = self._env.get_ee_pos()

            # NaN/Inf 安全检查（必须在 SafetyMonitor 之前：NaN 比较行为不确定）
            if not np.all(np.isfinite(state.q_desired)):
                logger.error("q_desired 含 NaN/Inf，拒绝下发，执行缓停")
                self._robot.slow_stop()
                return False

            if not self._safety.is_safe(
                arm_state, state.q_desired, tcp_speed=tcp_speed
            ):
                self._robot.slow_stop()
                return False

            # C1 修复：必须检查返回码（与 pre_motion 行为一致）
            # 原实现仅 Exception 触发 abort，SDK 通过非零返回码报错时被静默吞掉，
            # 100Hz 主循环会继续推进 setpoint → 控制器保持上一帧 + Sink 推进 → 物理风险
            ret = self._robot.send_joint_command(state.q_desired)
            if ret != 0:
                logger.error(
                    "send_joint_command 返回错误码 %d, 执行缓停", ret,
                )
                try:
                    self._robot.slow_stop()
                except Exception:
                    pass
                return False
        except Exception:
            # 机器人通信异常：尝试缓停，不调 tick_end（episode 已终止）
            try:
                self._robot.slow_stop()
            except Exception:
                pass  # 缓停也失败，无法做更多
            return False

        sleep_dt = self._timer.tick_end()
        if sleep_dt > 0:
            time.sleep(sleep_dt)

        return True


class RecorderSink:
    """记录到 TrajectoryRecorder（不实际运动）。

    从 StepState 读取数据，如果 arm_state/tcp_pos 为 None 则用 fallback。
    正常情况下 RobotSink 先执行并填充 state，RecorderSink 消费。
    """

    def __init__(
        self,
        recorder: TrajectoryRecorder,
        robot: RobotArmInterface | None = None,
        env: RobotEnv | None = None,
    ) -> None:
        """初始化 RecorderSink。

        Args:
            recorder: 轨迹记录器。
            robot: 可选，用于 fallback 读取 arm_state
                   （当 state.arm_state 为 None 时）。
            env: 可选，用于 fallback 读取 tcp_pos
                 （当 state.tcp_pos 为 None 时）。
        """
        self._recorder = recorder
        self._robot = robot
        self._env = env

    def send(self, state: StepState) -> bool:
        """记录一步数据。

        Fallback 逻辑:
            - arm_state: state.arm_state → robot.get_arm_state() → zeros(12)
            - tcp_pos: state.tcp_pos → env.get_ee_pos()（需先 set_arm_state）→ zeros(3)

        Fallback 读取异常时降级为 zeros，不抛出异常。

        始终返回 True（记录不会失败）。
        """
        if state.arm_state is not None:
            arm_state = state.arm_state
        elif self._robot is not None:
            try:
                arm_state = self._robot.get_arm_state()
            except Exception:
                arm_state = np.zeros(12)
        else:
            arm_state = np.zeros(12)

        if state.tcp_pos is not None:
            tcp_pos = state.tcp_pos
        elif self._env is not None and arm_state is not None:
            try:
                self._env.set_arm_state(arm_state)
                tcp_pos = self._env.get_ee_pos()
            except Exception:
                tcp_pos = np.zeros(3)
        else:
            tcp_pos = np.zeros(3)

        q_actual = arm_state[:6]
        self._recorder.record(
            state.q_desired, q_actual, state.timestamp, tcp_pos
        )
        return True


class TeeSink:
    """广播到多个 Sink（composite 模式），顺序执行。

    按列表顺序调用每个 Sink 的 send()。
    任一 Sink 返回 False 则短路（后续 Sink 不执行）。
    典型顺序: [RobotSink, RecorderSink] — RobotSink 填充 state，
              RecorderSink 消费。
    """

    def __init__(self, sinks: list[CommandSink]) -> None:
        """初始化 TeeSink。

        Args:
            sinks: Sink 列表，按顺序执行。
        """
        self._sinks = sinks

    def send(self, state: StepState) -> bool:
        """顺序调用每个 Sink。任一失败则短路返回 False。"""
        for sink in self._sinks:
            if not sink.send(state):
                return False
        return True
