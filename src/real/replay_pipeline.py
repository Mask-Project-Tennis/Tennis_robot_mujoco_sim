"""真机轨迹重演核心管线。

从 scripts/replay_trajectory.py 抽取的可 import 模块，供 CLI 入口和
Trajectory Studio 共享。包含 pre_motion 预运动、run_replay 主循环、
ReplayConfig 参数容器。

典型用法:
    # CLI 薄壳
    cfg = ReplayConfig(trajectory_path=Path("results/traj.npz"), speed=0.1)
    steps = run_replay(cfg)

    # Studio 内调用
    cfg = ReplayConfig(..., mock=True)  # FakeRobot 验证
    steps = run_replay(cfg)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from src.ilqt.planning_env import PlanningEnv
from src.real.adaptive_timer import AdaptiveTimer
from src.real.config import RealRobotConfig
from src.real.fake_robot import FakeRobot
from src.real.resample_strategy import InterpolatingResampler
from src.real.robot_arm_protocol import RobotArmInterface
from src.real.runner_factory import build_robot_limits
from src.real.safety_monitor import SafetyMonitor
from src.real.trajectory_recorder import TrajectoryRecorder
from src.real.trajectory_safety import check_joint_limits
from src.real.trajectory_sink import CommandSink, RecorderSink, RobotSink, TeeSink
from src.real.trajectory_source import (
    FileSource,
    ResampledSource,
    TcpSpeedControllable,
    TcpSpeedLimiter,
    TrajectorySource,
)
from src.real.trajectory_types import ReplayTrajectory, StepState

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "real_robot.yaml"
)


@dataclass
class ReplayConfig:
    """重演参数容器（替代 argparse.Namespace，可被 Studio 直接构造）。

    Attributes:
        trajectory_path: 轨迹文件路径 (.npz 或 .pkl)。
        speed: 速度因子 (0.1 = 1/10 速度)，必须 ≤ 1.0。
        use_actual: 使用 q_actual（仿真实际角度）替代 q_desired。推荐启用。
        mock: 使用 FakeRobot（无真机）。
        record: 记录真机实际轨迹到文件，None 不记录。
        pre_motion_duration: 预运动过渡时长（秒）。
        max_tcp_speed: TCP 速度限制 (m/s)，0=不限制。
        target_dt: 目标采样间隔（秒），None 用原始 dt。
        force_mode: 跳过控制模式安全检查（危险）。
        config_path: 真机配置文件路径。
    """

    trajectory_path: Path
    speed: float = 0.1
    use_actual: bool = True
    mock: bool = False
    record: Path | None = None
    pre_motion_duration: float = 10.0
    max_tcp_speed: float = 0.0
    target_dt: float | None = None
    force_mode: bool = False
    config_path: Path = field(default_factory=lambda: _DEFAULT_CONFIG_PATH)


@dataclass
class ReplayResult:
    """run_replay 的返回值，包含执行统计与失败诊断。

    替代早期版本的 int 返回值（-1 表示失败）。提供结构化失败原因，便于
    CLI / Studio / 上层编排准确区分失败模式并给出针对性反馈。

    Attributes:
        steps: 成功执行的步数（不含触发失败的当前步）。失败前未执行则为 0。
        status: 终结状态分类，取值见类型注解。
            - "ok": 正常完成全部轨迹。
            - "empty": 轨迹为 0 步（非失败，但无内容可重演）。
            - "control_mode_mismatch": 轨迹非位置模式且未 --force-mode。
            - "empty_q_actual": --use-actual 但 q_actual 为空。
            - "connect_failed": robot.connect() 返回 False。
            - "pre_motion_aborted": 预运动被用户拒绝或到位失败。
            - "limit_violation": I3 预检：q 超关节硬限位且未 --force-mode。
            - "safety_abort": 主循环中 SafetyMonitor.is_safe() 返回 False。
        reason: 人类可读的失败描述（status="ok"/"empty" 时为空字符串）。
        peak_step: 失败发生时已成功的步数；表示"第 peak_step+1 步失败"。
            例如 safety_abort 时 peak_step=5 表示前 5 步成功、第 6 步触发失败。
            status="ok"/"empty" 时为 None。
    """
    steps: int
    status: Literal[
        "ok",
        "empty",
        "control_mode_mismatch",
        "empty_q_actual",
        "connect_failed",
        "pre_motion_aborted",
        "limit_violation",
        "safety_abort",
    ]
    reason: str = ""
    peak_step: int | None = None

    @property
    def success(self) -> bool:
        """是否成功完成（含空轨迹，status in {"ok", "empty"}）。"""
        return self.status in ("ok", "empty")


def pre_motion(
    robot: RobotArmInterface,
    traj: ReplayTrajectory,
    duration_s: float = 10.0,
    rate_hz: float = 100.0,
    max_qdot_deg_s: float = 180.0,
) -> bool:
    """预运动：将机械臂以多步五次多项式插值平滑移动到轨迹初始位置。

    rm_set_arm_max_line_speed 对单点 rm_movej_follow 无效（实测验证），
    因此改为手动多步插值：以 rate_hz 频率发送五次多项式中间点，
    显式控制过渡速度。五次多项式保证起点和终点速度为零。

    流程:
        1. 读取当前关节角度，计算与 init_q 的差距
        2. 差距 < 0.5°: 跳过
        3. 差距 > 30°: 警告 + 要求确认
        4. 计算最小安全 duration（峰值角速度 < max_qdot_deg_s）
        5. 多步五次多项式插值下发，逐次检查返回码
        6. 轮询验证到位

    Args:
        robot: 机器人接口（RobotArmInterface Protocol 实现）。
        traj: 重演轨迹数据（含 init_q）。
        duration_s: 过渡时长（秒），实际取 max(duration_s, 最小安全值)。
        rate_hz: 插值下发频率（Hz），默认 100。
        max_qdot_deg_s: 关节角速度安全限制（°/s），用于计算最小时长。

    Returns:
        True 表示成功到达初始位置，False 表示失败/用户拒绝/通信错误。
    """
    q_current = robot.get_arm_state()[:6]
    delta_rad = np.abs(traj.init_q - q_current)
    max_delta_deg = float(np.degrees(np.max(delta_rad)))

    if max_delta_deg < 0.5:
        logger.info("已在初始位置 (delta=%.2f°), 跳过预运动", max_delta_deg)
        return True

    if max_delta_deg > 30.0:
        logger.warning("当前位置与 init_q 差距过大 (delta=%.2f°)", max_delta_deg)
        try:
            response = input("是否继续? (y/N): ").strip().lower()
        except EOFError:
            response = ""
        if response != "y":
            return False

    # 最小安全 duration（五次多项式峰值速度因子 1.875，在 s=0.5 处取到）
    PEAK_FACTOR = 1.875
    min_duration = PEAK_FACTOR * max_delta_deg / max_qdot_deg_s
    effective_duration = max(duration_s, min_duration)
    if effective_duration > duration_s:
        logger.info(
            "预运动 duration 自动延长: %.1fs → %.1fs (delta=%.1f°, 限速=%.0f°/s)",
            duration_s, effective_duration, max_delta_deg, max_qdot_deg_s,
        )

    n_steps = max(1, int(effective_duration * rate_hz))
    dt_step = 1.0 / rate_hz

    logger.info(
        "预运动开始: %d 步 @ %.0fHz (%.1fs), delta=%.1f°",
        n_steps, rate_hz, effective_duration, max_delta_deg,
    )

    # 多步五次多项式插值: f(s) = 10s³ - 15s⁴ + 6s⁵
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        s = 10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5
        q_target = q_current + s * (traj.init_q - q_current)
        ret = robot.send_joint_command(q_target)
        if ret != 0:
            logger.error("预运动发送失败: step %d/%d, 错误码 %d", i, n_steps, ret)
            return False
        time.sleep(dt_step)

    # 轮询验证到位
    tolerance = float(np.radians(1.0))
    timeout = 3.0
    poll_interval = 0.05
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        q = robot.get_arm_state()[:6]
        if float(np.max(np.abs(q - traj.init_q))) < tolerance:
            logger.info(
                "预运动到位 (delta=%.1f° → 0°, 耗时 %.1fs)",
                max_delta_deg, effective_duration + time.perf_counter() - t0,
            )
            return True
        time.sleep(poll_interval)

    final_delta = float(np.degrees(np.max(np.abs(robot.get_arm_state()[:6] - traj.init_q))))
    logger.warning("预运动到位偏差: %.2f° (容差 1.0°)", final_delta)
    return final_delta < 2.0


def run_replay(cfg: ReplayConfig) -> ReplayResult:
    """执行真机重演完整管线。

    加载配置 → 加载轨迹 → 安全校验 → 创建 env/robot/safety →
    预运动 → Source/Sink 链 → 主循环 → 清理。

    Args:
        cfg: 重演参数容器。

    Returns:
        ReplayResult，含执行步数与失败诊断。result.success 判断成败，
        result.status 区分失败模式，result.reason 给出可读描述。
    """
    logger.info(
        "启动轨迹重演: trajectory=%s speed=%.2f mock=%s",
        cfg.trajectory_path, cfg.speed, cfg.mock,
    )

    # 加载配置
    config = RealRobotConfig.from_yaml(str(cfg.config_path))
    logger.info(
        "配置已加载: %s (dt=%.3f tcp=%.1f)",
        cfg.config_path, config.dt, config.max_tcp_speed,
    )

    # 加载轨迹
    traj = TrajectoryRecorder.load(cfg.trajectory_path)
    logger.info(
        "轨迹已加载: %d 步, dt=%.4f, init_q=%s",
        len(traj.q_desired), traj.dt, traj.init_q,
    )

    # 安全校验：只支持位置模式轨迹（真机仅支持角度控制）
    if not traj.metadata.get("is_position_mode", True):
        if cfg.force_mode:
            logger.warning("--force-mode: 跳过控制模式检查，请确认轨迹是位置模式生成！")
        else:
            logger.error("轨迹为力矩模式或来源不明，无法安全重演（真机仅支持位置模式）")
            logger.error("如确认是位置模式生成，加 --force-mode 跳过检查")
            return ReplayResult(
                steps=0,
                status="control_mode_mismatch",
                reason="轨迹非位置模式，加 --force-mode 跳过",
            )

    # --use-actual: 使用 q_actual 重演
    if cfg.use_actual:
        if len(traj.q_actual) == 0:
            logger.error(
                "--use-actual 但轨迹 q_actual 为空（旧格式或未记录），拒绝重演。"
                "去掉 --use-actual 用 q_desired，或重新生成轨迹。"
            )
            return ReplayResult(
                steps=0,
                status="empty_q_actual",
                reason="--use-actual 但 q_actual 为空",
            )
        old_init = np.degrees(traj.init_q).round(1)
        traj.init_q = traj.q_actual[0].copy()
        logger.info(
            "--use-actual: init_q 覆盖为 q_actual[0] (%s → %s)",
            old_init.tolist(), np.degrees(traj.init_q).round(1).tolist(),
        )

    # 空轨迹提前返回（非失败，但无内容可重演）
    if len(traj.q_desired) == 0:
        logger.warning("轨迹为空（0 步），无内容可重演")
        return ReplayResult(steps=0, status="empty", reason="轨迹 0 步")

    # I3: 关节限位预检（仅看硬超限；裕度告警留给 inspect_trajectory 工具）
    # 必须在 robot.connect / pre_motion 之前完成，避免对拒绝的轨迹做任何硬件动作
    q_check = traj.q_actual if (cfg.use_actual and len(traj.q_actual) > 0) else traj.q_desired
    violations = check_joint_limits(
        q_check,
        np.degrees(config.q_lower),
        np.degrees(config.q_upper),
        margin_deg=0.0,
    )
    if violations and not cfg.force_mode:
        logger.error("轨迹超关节硬限位，拒绝重演：%s", violations[:3])
        return ReplayResult(
            steps=0,
            status="limit_violation",
            reason="; ".join(v.strip() for v in violations[:3]),
        )

    # 创建规划环境（位置模式）
    env = PlanningEnv(dt=config.dt)
    env.init_q_left = traj.init_q_left.copy()
    env.configure_actuator_mode("position", kp=config.kp, kd=config.kd)
    env.configure_feedforward(config.enable_feedforward)
    env.reset(traj.init_q)
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left

    # 创建机器人
    if cfg.mock:
        robot: RobotArmInterface = FakeRobot(init_q=traj.init_q, dt=config.dt)
    else:
        from src.real.robot_interface import RobotInterface

        robot = RobotInterface(config)
        logger.info("使用真机接口 RobotInterface (mock=False)")

    if not robot.connect():
        logger.error("机器人连接失败")
        return ReplayResult(
            steps=0,
            status="connect_failed",
            reason="robot.connect() 返回 False",
        )

    # 安全监控
    robot_limits = build_robot_limits(env, config)
    safety_cfg = RealRobotConfig()
    safety_cfg.q_lower = robot_limits.q_lower.copy()
    safety_cfg.q_upper = robot_limits.q_upper.copy()
    safety_cfg.max_qdot = robot_limits.qdot_max.copy()
    safety_cfg.max_tcp_speed = float(robot_limits.max_tcp_speed)
    safety = SafetyMonitor(safety_cfg, robot=robot)

    timer = AdaptiveTimer(target_hz=100.0)

    recorder: TrajectoryRecorder | None = None
    step_count = 0
    safety_failed = False
    try:
        if not pre_motion(robot, traj, duration_s=cfg.pre_motion_duration):
            logger.error("预运动失败，退出")
            return ReplayResult(
                steps=0,
                status="pre_motion_aborted",
                reason="预运动被用户拒绝或到位失败",
            )

        # 设置主轨迹阶段的 TCP 速度限制
        if isinstance(robot, TcpSpeedControllable):
            robot.set_max_tcp_speed(config.max_tcp_speed)

        # 构建 Source 链
        source: TrajectorySource = FileSource(
            cfg.trajectory_path, use_actual=cfg.use_actual
        )
        source = ResampledSource(
            source, InterpolatingResampler(), cfg.speed, cfg.target_dt
        )
        if cfg.max_tcp_speed > 0:
            source = TcpSpeedLimiter(
                source, robot, cfg.max_tcp_speed,
                restore_speed=config.max_tcp_speed,
            )

        # 构建 Sink 链
        sinks: list[CommandSink] = [RobotSink(robot, safety, timer, env)]
        if cfg.record:
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
        if cfg.record and recorder is not None:
            recorder.save(Path(cfg.record))
            logger.info("真机轨迹已保存至 %s", cfg.record)
        robot.disconnect()

    # 主循环结束：根据 safety_failed 标志返回对应状态
    if safety_failed:
        logger.error(
            "重演中止: 已完成 %d 步，第 %d 步 SafetyMonitor 触发失败",
            step_count, step_count + 1,
        )
        return ReplayResult(
            steps=step_count,
            status="safety_abort",
            reason=f"SafetyMonitor 在第 {step_count + 1} 步触发 slow_stop",
            peak_step=step_count,
        )
    logger.info("重演完成: %d 步", step_count)
    return ReplayResult(steps=step_count, status="ok")
