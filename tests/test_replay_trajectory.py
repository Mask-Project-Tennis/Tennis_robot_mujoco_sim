"""真机轨迹重演入口脚本端到端测试。

使用 FakeRobot + 真实 PlanningEnv + 真实 SafetyMonitor 构建完整 Mock 闭环，
验证 pre_motion 预运动、Source/Sink 链、主循环、安全失败、TCP 限速等行为。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 添加 scripts/ 到路径，导入入口脚本中的 pre_motion
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from replay_trajectory import pre_motion  # noqa: E402

from src.ilqt.planning_env import PlanningEnv  # noqa: E402
from src.real.adaptive_timer import AdaptiveTimer  # noqa: E402
from src.real.config import RealRobotConfig  # noqa: E402
from src.real.fake_robot import FakeRobot  # noqa: E402
from src.real.runner_factory import (  # noqa: E402
    DT,
    INIT_Q,
    INIT_Q_LEFT,
    KD,
    KP,
    build_robot_limits,
)
from src.real.safety_monitor import SafetyMonitor  # noqa: E402
from src.real.trajectory_recorder import TrajectoryRecorder  # noqa: E402
from src.real.trajectory_sink import RecorderSink, RobotSink, TeeSink  # noqa: E402
from src.real.trajectory_source import (  # noqa: E402
    FileSource,
    ResampledSource,
    TcpSpeedLimiter,
)
from src.real.trajectory_types import ReplayTrajectory, StepState  # noqa: E402

_CFG = RealRobotConfig()


def _build_env() -> PlanningEnv:
    """构建测试用 PlanningEnv（位置模式）。

    Returns:
        已初始化的 PlanningEnv，右臂在 INIT_Q，左臂在 INIT_Q_LEFT。
    """
    env = PlanningEnv(dt=DT)
    env.init_q_left = INIT_Q_LEFT.copy()
    env.configure_actuator_mode("position", kp=KP, kd=KD)
    env.configure_feedforward(True)
    env.reset(INIT_Q)
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left
    return env


def _build_safety_cfg(env: PlanningEnv) -> RealRobotConfig:
    """构建测试用安全配置（限位与 RobotLimits 含裕度一致）。

    Args:
        env: 规划环境，用于读取 actuator_ctrlrange。

    Returns:
        安全配置，q_lower/q_upper/max_qdot/max_tcp_speed 与 RobotLimits 对齐。
    """
    robot_limits = build_robot_limits(env, _CFG)
    safety_cfg = RealRobotConfig()
    safety_cfg.q_lower = robot_limits.q_lower.copy()
    safety_cfg.q_upper = robot_limits.q_upper.copy()
    safety_cfg.max_qdot = robot_limits.qdot_max.copy()
    safety_cfg.max_tcp_speed = float(robot_limits.max_tcp_speed)
    return safety_cfg


def _make_traj_obj(n_steps: int = 20) -> ReplayTrajectory:
    """构建测试用 ReplayTrajectory 对象（不写入文件）。

    轨迹在 INIT_Q 附近做小幅度正弦运动（关节0），确保安全通过。

    Args:
        n_steps: 轨迹步数。

    Returns:
        ReplayTrajectory 对象，init_q=INIT_Q，init_q_left=INIT_Q_LEFT。
    """
    q_desired = np.zeros((n_steps, 6))
    for i in range(n_steps):
        q_desired[i] = INIT_Q
        q_desired[i, 0] += 0.01 * np.sin(i * 0.5)
    timestamps = np.arange(n_steps, dtype=float) * DT
    return ReplayTrajectory(
        q_desired=q_desired,
        q_actual=q_desired.copy(),
        timestamps=timestamps,
        tcp_pos=np.zeros((n_steps, 3)),
        ball_pos=np.zeros((n_steps, 3)),
        init_q=INIT_Q.copy(),
        init_q_left=INIT_Q_LEFT.copy(),
        dt=DT,
        hit_step=-1,
    )


def _save_test_trajectory(tmp_path: Path, n_steps: int = 20) -> Path:
    """保存测试轨迹到 .npz 文件。

    Args:
        tmp_path: pytest 临时目录。
        n_steps: 轨迹步数。

    Returns:
        轨迹文件路径。
    """
    env = _build_env()
    recorder = TrajectoryRecorder(env, INIT_Q, INIT_Q_LEFT, DT)
    for i in range(n_steps):
        q = INIT_Q.copy()
        q[0] += 0.01 * np.sin(i * 0.5)
        recorder.record(q, q, i * DT, np.zeros(3))
    path = tmp_path / "test_traj.npz"
    recorder.save(path)
    return path


class _NoSleepTimer(AdaptiveTimer):
    """不 sleep 的计时器（测试用，避免 AdaptiveTimer 的 time.sleep 拖慢测试）。"""

    def tick_end(self) -> float:
        return 0.0


class _LinearResampler:
    """线性插值重采样策略（测试用，避免 CubicSpline DLL 崩溃）。

    实现 ResampleStrategy Protocol，用 np.interp 替代 scipy CubicSpline。
    时间轴计算逻辑与 InterpolatingResampler 完全一致，仅插值方法不同。
    """

    def resample(
        self,
        qs: np.ndarray,
        ts: np.ndarray,
        speed_factor: float,
        target_dt: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """线性插值重采样。

        Args:
            qs: (N, 6) 原始关节角度序列。
            ts: (N,) 原始时间戳（秒）。
            speed_factor: 速度因子（0.1 = 1/10 速度）。
            target_dt: 目标采样间隔（秒），None 用原始 dt。

        Returns:
            (new_qs, new_ts) 重采样后的关节角度和时间戳。
        """
        n = qs.shape[0]
        if n == 0 or ts.shape[0] == 0:
            raise ValueError("输入序列为空，无法重采样")
        if speed_factor <= 0:
            raise ValueError(f"speed_factor 必须 > 0，得到 {speed_factor}")
        if n == 1 or ts.shape[0] == 1:
            return (
                np.asarray(qs, dtype=np.float64).copy(),
                np.asarray(ts, dtype=np.float64).copy(),
            )

        raw_dt = float(np.median(np.diff(ts)))
        dt = float(target_dt) if target_dt is not None else raw_dt
        if dt <= 0:
            raise ValueError(f"target_dt 必须 > 0，得到 {dt}")

        t0 = float(ts[0])
        t_end = float(ts[-1])
        T = t_end - t0
        T_new = T / speed_factor
        m = int(T_new / dt + 1e-9) + 1
        new_ts = t0 + np.arange(m, dtype=np.float64) * dt

        sample_t = t0 + (new_ts - t0) * speed_factor
        sample_t = np.clip(sample_t, t0, t_end)

        new_qs = np.zeros((m, qs.shape[1]), dtype=np.float64)
        for j in range(qs.shape[1]):
            new_qs[:, j] = np.interp(sample_t, ts, qs[:, j])

        return new_qs, new_ts


class _UnsafeSafety:
    """始终返回不安全的安全监控 Mock（手写，不使用 MagicMock）。"""

    def is_safe(
        self,
        arm_state: np.ndarray,
        q_desired: np.ndarray,
        tcp_speed: float = 0.0,
    ) -> bool:
        return False


class TestPreMotion:
    """预运动测试。"""

    def test_already_at_init(self) -> None:
        """已在初始位置 → delta<0.5° → 跳过预运动，无命令发送。"""
        traj = _make_traj_obj()
        robot = FakeRobot(init_q=traj.init_q, dt=DT)
        robot.connect()
        result = pre_motion(robot, traj, max_tcp_speed=0.3)
        assert result is True
        assert len(robot.command_history) == 0
        assert len(robot.set_max_tcp_speed_calls) == 0

    def test_moves_to_init(self) -> None:
        """需移动（0.1rad≈5.7°）→ 发送命令 → 到位。"""
        traj = _make_traj_obj()
        different_q = INIT_Q + np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot = FakeRobot(init_q=different_q, dt=DT)
        robot.connect()
        result = pre_motion(robot, traj, max_tcp_speed=0.3)
        assert result is True
        assert len(robot.command_history) >= 1
        assert 0.3 in robot.set_max_tcp_speed_calls
        q = robot.get_arm_state()[:6]
        assert np.allclose(q, traj.init_q, atol=np.radians(0.5))

    def test_large_delta_decline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """差距过大（1.0rad≈57°>30°）→ 用户拒绝 → 返回 False，无命令。"""
        traj = _make_traj_obj()
        far_q = INIT_Q + np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot = FakeRobot(init_q=far_q, dt=DT)
        robot.connect()
        monkeypatch.setattr("builtins.input", lambda prompt="": "n")
        result = pre_motion(robot, traj, max_tcp_speed=0.3)
        assert result is False
        assert len(robot.command_history) == 0

    def test_large_delta_accept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """差距过大（1.0rad≈57°>30°）→ 用户确认 → 移动 → 到位。"""
        traj = _make_traj_obj()
        far_q = INIT_Q + np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot = FakeRobot(init_q=far_q, dt=DT)
        robot.connect()
        monkeypatch.setattr("builtins.input", lambda prompt="": "y")
        result = pre_motion(robot, traj, max_tcp_speed=0.3)
        assert result is True
        assert len(robot.command_history) >= 1
        assert 0.3 in robot.set_max_tcp_speed_calls
        q = robot.get_arm_state()[:6]
        assert np.allclose(q, traj.init_q, atol=np.radians(0.5))


class TestReplayTrajectoryMock:
    """完整重演管线测试（FakeRobot）。"""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        """完整管线：加载轨迹 → 预运动 → Source 链 → Sink 链 → 主循环。

        验证:
            - step_count 等于轨迹步数
            - command_history 匹配重采样轨迹
            - 安全检查全部通过（无 slow_stop）
        """
        n_steps = 20
        traj_path = _save_test_trajectory(tmp_path, n_steps=n_steps)

        env = _build_env()
        traj = TrajectoryRecorder.load(traj_path)
        robot = FakeRobot(init_q=traj.init_q, dt=DT)
        robot.connect()
        safety_cfg = _build_safety_cfg(env)
        safety = SafetyMonitor(safety_cfg, robot=robot)
        timer = _NoSleepTimer(target_hz=100.0)

        # 预运动（已在 init_q，跳过）
        assert pre_motion(robot, traj) is True

        # Source 链
        source = FileSource(traj_path)
        source = ResampledSource(source, _LinearResampler(), speed_factor=1.0)

        # Sink 链
        sink = RobotSink(robot, safety, timer, env)

        # 主循环
        step_count = 0
        for q_desired, timestamp in source:
            state = StepState(q_desired=q_desired.copy(), timestamp=timestamp)
            if not sink.send(state):
                break
            step_count += 1

        # 验证步数
        assert step_count == n_steps
        assert len(robot.command_history) == n_steps

        # 验证命令匹配重采样轨迹（speed=1.0 → 值应一致）
        resampler = _LinearResampler()
        expected_qs, _ = resampler.resample(
            traj.q_desired, traj.timestamps, speed_factor=1.0, target_dt=None
        )
        for i, cmd in enumerate(robot.command_history):
            assert np.allclose(cmd, expected_qs[i], atol=1e-6), (
                f"步 {i} 命令不匹配: cmd={cmd}, expected={expected_qs[i]}"
            )

        # 验证安全通过（无 slow_stop）
        assert robot.slow_stop_count == 0

    def test_record_output(self, tmp_path: Path) -> None:
        """--record 生成输出文件，文件含与步数等长的轨迹数据。"""
        n_steps = 10
        traj_path = _save_test_trajectory(tmp_path, n_steps=n_steps)

        env = _build_env()
        traj = TrajectoryRecorder.load(traj_path)
        robot = FakeRobot(init_q=traj.init_q, dt=DT)
        robot.connect()
        safety_cfg = _build_safety_cfg(env)
        safety = SafetyMonitor(safety_cfg, robot=robot)
        timer = _NoSleepTimer(target_hz=100.0)

        pre_motion(robot, traj)

        source = FileSource(traj_path)
        source = ResampledSource(source, _LinearResampler(), speed_factor=1.0)

        recorder = TrajectoryRecorder(env, traj.init_q, traj.init_q_left, DT)
        sinks = [
            RobotSink(robot, safety, timer, env),
            RecorderSink(recorder, robot=robot, env=env),
        ]
        sink = TeeSink(sinks)

        step_count = 0
        for q_desired, timestamp in source:
            state = StepState(q_desired=q_desired.copy(), timestamp=timestamp)
            if not sink.send(state):
                break
            step_count += 1

        record_path = tmp_path / "recorded.npz"
        recorder.save(record_path)
        assert record_path.exists()

        recorded = TrajectoryRecorder.load(record_path)
        assert len(recorded.q_desired) == step_count
        assert step_count == n_steps


class TestReplayTrajectorySafetyFailure:
    """安全失败测试。"""

    def test_safety_failure_stops_loop(self, tmp_path: Path) -> None:
        """安全检查失败 → 主循环第一步中断 → slow_stop 被调用。"""
        n_steps = 20
        traj_path = _save_test_trajectory(tmp_path, n_steps=n_steps)

        env = _build_env()
        traj = TrajectoryRecorder.load(traj_path)
        robot = FakeRobot(init_q=traj.init_q, dt=DT)
        robot.connect()
        unsafe_safety = _UnsafeSafety()
        timer = _NoSleepTimer(target_hz=100.0)

        pre_motion(robot, traj)

        source = FileSource(traj_path)
        source = ResampledSource(source, _LinearResampler(), speed_factor=1.0)
        sink = RobotSink(robot, unsafe_safety, timer, env)

        step_count = 0
        for q_desired, timestamp in source:
            state = StepState(q_desired=q_desired.copy(), timestamp=timestamp)
            if not sink.send(state):
                break
            step_count += 1

        # 第一步安全检查即失败
        assert step_count == 0
        assert robot.slow_stop_count >= 1


class TestReplayTrajectoryTcpSpeedLimit:
    """TCP 速度限制测试。"""

    def test_tcp_speed_limit_set(self, tmp_path: Path) -> None:
        """--max-tcp-speed > 0 → TcpSpeedLimiter 调用 set_max_tcp_speed。

        验证:
            - 迭代前设置 0.2 m/s
            - 迭代后恢复 1.0 m/s
            - 预运动未调用 set_max_tcp_speed（已在 init_q）
        """
        n_steps = 10
        traj_path = _save_test_trajectory(tmp_path, n_steps=n_steps)

        env = _build_env()
        traj = TrajectoryRecorder.load(traj_path)
        robot = FakeRobot(init_q=traj.init_q, dt=DT)
        robot.connect()
        safety_cfg = _build_safety_cfg(env)
        safety = SafetyMonitor(safety_cfg, robot=robot)
        timer = _NoSleepTimer(target_hz=100.0)

        # 预运动（已在 init_q，不调用 set_max_tcp_speed）
        pre_motion(robot, traj)
        assert len(robot.set_max_tcp_speed_calls) == 0

        # Source 链含 TcpSpeedLimiter
        source = FileSource(traj_path)
        source = ResampledSource(source, _LinearResampler(), speed_factor=1.0)
        source = TcpSpeedLimiter(
            source, robot, max_tcp_speed=0.2, restore_speed=1.0
        )
        sink = RobotSink(robot, safety, timer, env)

        for q_desired, timestamp in source:
            state = StepState(q_desired=q_desired.copy(), timestamp=timestamp)
            if not sink.send(state):
                break

        # TcpSpeedLimiter 在迭代前设置 0.2，迭代后恢复 1.0
        assert 0.2 in robot.set_max_tcp_speed_calls
        assert 1.0 in robot.set_max_tcp_speed_calls
        assert len(robot.set_max_tcp_speed_calls) >= 2
