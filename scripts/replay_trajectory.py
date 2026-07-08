#!/usr/bin/env python
"""真机轨迹重演入口脚本。

从 .npz/.pkl 文件加载仿真记录的轨迹，以可调慢速在真机上重演。
用于验证仿真算法的部署可行性。

用法:
    # 仿真生成轨迹
    python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --position-mode \
        --dump-trajectory results/traj.npz

    # 真机重演（1/10 速度）
    python scripts/replay_trajectory.py --trajectory results/traj.npz --speed 0.1

    # Mock 模式（无真机，用 FakeRobot 验证流程）
    python scripts/replay_trajectory.py --trajectory results/traj.npz --speed 0.1 --mock
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# 支持从 scripts/ 目录直接运行，确保 `from src.xxx` 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ilqt.planning_env import PlanningEnv
from src.real.adaptive_timer import AdaptiveTimer
from src.real.config import RealRobotConfig
from src.real.fake_robot import FakeRobot
from src.real.resample_strategy import InterpolatingResampler
from src.real.robot_arm_protocol import RobotArmInterface
from src.real.runner_factory import build_robot_limits
from src.real.safety_monitor import SafetyMonitor
from src.real.trajectory_recorder import TrajectoryRecorder
from src.real.trajectory_sink import CommandSink, RecorderSink, RobotSink, TeeSink
from src.real.trajectory_source import (
    FileSource,
    ResampledSource,
    TcpSpeedLimiter,
    TrajectorySource,
)
from src.real.trajectory_types import ReplayTrajectory, StepState

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "real_robot.yaml"


def pre_motion(
    robot: RobotArmInterface,
    traj: ReplayTrajectory,
    max_tcp_speed: float = 0.3,
) -> bool:
    """预运动：将机械臂移动到轨迹初始位置。

    流程:
        1. 读取当前关节角度
        2. 计算与 init_q 的差距
        3. 差距 < 0.5°: 跳过
        4. 差距 > 30°: 警告 + 要求确认
        5. 否则: 限制 TCP 速度 → send_joint_command(init_q) → 轮询到位

    Args:
        robot: 机器人接口（RobotArmInterface Protocol 实现）。
        traj: 重演轨迹数据（含 init_q）。
        max_tcp_speed: 预运动期间的 TCP 速度限制（m/s）。

    Returns:
        True 表示成功到达初始位置，False 表示用户拒绝或超时。
    """
    q_current = robot.get_arm_state()[:6]
    delta = float(np.max(np.abs(q_current - traj.init_q)))
    delta_deg = float(np.degrees(delta))

    if delta_deg < 0.5:
        logger.info("已在初始位置 (delta=%.2f°), 跳过预运动", delta_deg)
        return True

    if delta_deg > 30.0:
        logger.warning("当前位置与 init_q 差距过大 (delta=%.2f°)", delta_deg)
        try:
            response = input("是否继续? (y/N): ").strip().lower()
        except EOFError:
            response = ""
        if response != "y":
            return False

    # 限制 TCP 速度
    if hasattr(robot, "set_max_tcp_speed"):
        robot.set_max_tcp_speed(max_tcp_speed)

    robot.send_joint_command(traj.init_q)

    # 轮询到位
    tolerance = float(np.radians(0.5))  # 0.5°
    timeout = 15.0  # 15秒
    poll_interval = 0.05  # 50ms
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        q = robot.get_arm_state()[:6]
        if float(np.max(np.abs(q - traj.init_q))) < tolerance:
            logger.info("预运动到位 (%.2fs)", time.perf_counter() - t0)
            return True
        time.sleep(poll_interval)

    logger.error("预运动超时 (%.1fs)", timeout)
    return False


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="真机轨迹重演 — 从文件加载仿真轨迹，以可调慢速在真机上重演",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        required=True,
        help="轨迹文件路径 (.npz 或 .pkl)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.1,
        help="速度因子 (0.1 = 1/10 速度)",
    )
    parser.add_argument(
        "--max-tcp-speed",
        type=float,
        default=0.0,
        help="TCP 速度限制 (m/s), 0=不限制",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG_PATH),
        help="真机配置文件",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用 FakeRobot (无真机)",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="记录真机实际轨迹到文件",
    )
    parser.add_argument(
        "--target-dt",
        type=float,
        default=None,
        help="目标采样间隔 (秒), 默认用原始 dt",
    )
    parser.add_argument(
        "--force-mode",
        action="store_true",
        help="跳过控制模式安全检查（危险！仅用于确认旧文件是位置模式生成）",
    )
    return parser.parse_args()


def main() -> None:
    """入口主函数：配置日志 → 加载配置/轨迹 → 组装组件 → 预运动 → 主循环 → 清理。"""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    logger.info(
        "启动轨迹重演: trajectory=%s speed=%.2f mock=%s",
        args.trajectory, args.speed, args.mock,
    )

    # 加载配置
    config = RealRobotConfig.from_yaml(args.config)
    logger.info(
        "配置已加载: %s (dt=%.3f tcp=%.1f)",
        args.config, config.dt, config.max_tcp_speed,
    )

    # 加载轨迹
    traj = TrajectoryRecorder.load(Path(args.trajectory))
    logger.info(
        "轨迹已加载: %d 步, dt=%.4f, init_q=%s",
        len(traj.q_desired), traj.dt, traj.init_q,
    )

    # 安全校验：只支持位置模式轨迹（真机仅支持角度控制）
    # 力矩模式的 q_desired 实际是力矩值（N·m），直接下发会危及安全
    if not traj.metadata.get("is_position_mode", True):
        if args.force_mode:
            logger.warning("--force-mode: 跳过控制模式检查，请确认轨迹是位置模式生成！")
        else:
            logger.error("轨迹为力矩模式或来源不明，无法安全重演（真机仅支持位置模式）")
            logger.error("如确认是位置模式生成，加 --force-mode 跳过检查")
            return

    # 创建规划环境（位置模式）
    env = PlanningEnv(dt=config.dt)
    env.init_q_left = traj.init_q_left.copy()
    env.configure_actuator_mode("position", kp=config.kp, kd=config.kd)
    env.configure_feedforward(config.enable_feedforward)
    env.reset(traj.init_q)
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left

    # 创建机器人
    if args.mock:
        robot: RobotArmInterface = FakeRobot(init_q=traj.init_q, dt=config.dt)
    else:
        from src.real.robot_interface import RobotInterface

        robot = RobotInterface(config)
        logger.info("使用真机接口 RobotInterface (mock=False)")

    # 检查连接（connect 失败则无法继续）
    if not robot.connect():
        logger.error("机器人连接失败")
        return

    # 安全监控 — 限位与 RobotLimits 含裕度一致，避免边界抖动误判
    robot_limits = build_robot_limits(env, config)
    safety_cfg = RealRobotConfig()
    safety_cfg.q_lower = robot_limits.q_lower.copy()
    safety_cfg.q_upper = robot_limits.q_upper.copy()
    safety_cfg.max_qdot = robot_limits.qdot_max.copy()
    safety_cfg.max_tcp_speed = float(robot_limits.max_tcp_speed)
    safety = SafetyMonitor(safety_cfg, robot=robot)

    # 计时器（100Hz 目标频率）
    timer = AdaptiveTimer(target_hz=100.0)

    # 空轨迹告警
    if len(traj.q_desired) == 0:
        logger.warning("轨迹为空（0 步），无内容可重演")

    recorder: TrajectoryRecorder | None = None
    step_count = 0
    safety_failed = False
    try:
        # 预运动
        if not pre_motion(robot, traj, max_tcp_speed=0.3):
            logger.error("预运动失败，退出")
            return

        # 恢复预运动期间降低的 TCP 速度
        if hasattr(robot, "set_max_tcp_speed"):
            robot.set_max_tcp_speed(config.max_tcp_speed)

        # 构建 Source 链
        source: TrajectorySource = FileSource(args.trajectory)
        source = ResampledSource(
            source, InterpolatingResampler(), args.speed, args.target_dt
        )
        if args.max_tcp_speed > 0:
            source = TcpSpeedLimiter(
                source, robot, args.max_tcp_speed, restore_speed=config.max_tcp_speed
            )

        # 构建 Sink 链
        sinks: list[CommandSink] = [RobotSink(robot, safety, timer, env)]
        if args.record:
            recorder = TrajectoryRecorder(
                env, traj.init_q, traj.init_q_left, config.dt
            )
            sinks.append(RecorderSink(recorder, robot=robot, env=env))
        sink: CommandSink = TeeSink(sinks) if len(sinks) > 1 else sinks[0]

        # 主循环
        for q_desired, timestamp in source:
            state = StepState(q_desired=q_desired.copy(), timestamp=timestamp)
            if not sink.send(state):
                logger.error("安全检查失败或异常，停止重演")
                safety_failed = True
                break
            step_count += 1
    finally:
        robot.slow_stop()
        if args.record and recorder is not None:
            recorder.save(Path(args.record))
            logger.info("真机轨迹已保存至 %s", args.record)
        robot.disconnect()

    logger.info("重演完成: %d 步, 安全失败: %s", step_count, safety_failed)


if __name__ == "__main__":
    main()
