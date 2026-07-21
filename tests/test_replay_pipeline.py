"""replay_pipeline 模块测试。

验证 ReplayConfig 参数映射、run_replay mock 端到端管线、pre_motion 行为。
pre_motion 的详细测试见 test_replay_trajectory.py（已迁移 import）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.real.replay_pipeline import ReplayConfig, run_replay
from src.real.trajectory_recorder import TrajectoryRecorder
from src.real.runner_factory import DT, INIT_Q, INIT_Q_LEFT


def _save_test_trajectory(tmp_path: Path, n_steps: int = 10) -> Path:
    """保存测试轨迹到 .npz 文件。"""
    from src.ilqt.planning_env import PlanningEnv
    from src.real.runner_factory import KD, KP

    env = PlanningEnv(dt=DT)
    env.init_q_left = INIT_Q_LEFT.copy()
    env.configure_actuator_mode("position", kp=KP, kd=KD)
    env.configure_feedforward(True)
    env.reset(INIT_Q)
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left

    recorder = TrajectoryRecorder(env, INIT_Q, INIT_Q_LEFT, DT)
    for i in range(n_steps):
        q = INIT_Q.copy()
        q[0] += 0.01 * np.sin(i * 0.5)
        tcp = env.get_ee_pos()
        recorder.record(q, q, i * DT, tcp)
        env.data.qpos[:6] = q
        env.update_kinematics()

    path = tmp_path / "pipeline_test.npz"
    recorder.save(path)
    return path


class TestReplayConfig:
    """ReplayConfig 参数容器测试。"""

    def test_defaults(self) -> None:
        """默认值正确。"""
        cfg = ReplayConfig(trajectory_path=Path("test.npz"))
        assert cfg.speed == 0.1
        assert cfg.use_actual is True
        assert cfg.mock is False
        assert cfg.record is None
        assert cfg.pre_motion_duration == 10.0
        assert cfg.max_tcp_speed == 0.0
        assert cfg.target_dt is None
        assert cfg.force_mode is False

    def test_custom_values(self) -> None:
        """自定义值正确映射。"""
        cfg = ReplayConfig(
            trajectory_path=Path("x.npz"),
            speed=0.5,
            use_actual=False,
            mock=True,
            record=Path("out.npz"),
            max_tcp_speed=0.3,
        )
        assert cfg.speed == 0.5
        assert cfg.use_actual is False
        assert cfg.mock is True
        assert cfg.record == Path("out.npz")
        assert cfg.max_tcp_speed == 0.3


class TestRunReplayMock:
    """run_replay mock 端到端测试。"""

    def test_mock_replay_success(self, tmp_path: Path) -> None:
        """FakeRobot mock 模式完整重演成功。"""
        traj_path = _save_test_trajectory(tmp_path, n_steps=10)

        cfg = ReplayConfig(
            trajectory_path=traj_path,
            speed=1.0,
            use_actual=True,
            mock=True,
        )
        result = run_replay(cfg)
        assert result.steps == 10
        assert result.status == "ok"
        assert result.success

    def test_mock_replay_with_record(self, tmp_path: Path) -> None:
        """mock 模式 + record 输出文件。"""
        traj_path = _save_test_trajectory(tmp_path, n_steps=5)
        record_path = tmp_path / "recorded.npz"

        cfg = ReplayConfig(
            trajectory_path=traj_path,
            speed=1.0,
            use_actual=True,
            mock=True,
            record=record_path,
        )
        result = run_replay(cfg)
        assert result.steps == 5
        assert result.status == "ok"
        assert record_path.exists()

        recorded = TrajectoryRecorder.load(record_path)
        assert len(recorded.q_desired) == 5

    def test_empty_trajectory(self, tmp_path: Path) -> None:
        """空轨迹（0步）不崩溃，返回 0。"""
        from src.ilqt.planning_env import PlanningEnv
        from src.real.runner_factory import KP, KD

        env = PlanningEnv(dt=DT)
        env.init_q_left = INIT_Q_LEFT.copy()
        env.configure_actuator_mode("position", kp=KP, kd=KD)
        env.configure_feedforward(True)
        env.reset(INIT_Q)

        recorder = TrajectoryRecorder(env, INIT_Q, INIT_Q_LEFT, DT)
        traj_path = tmp_path / "empty.npz"
        recorder.save(traj_path)

        cfg = ReplayConfig(
            trajectory_path=traj_path,
            speed=1.0,
            use_actual=False,  # 空轨迹没有 q_actual
            mock=True,
        )
        result = run_replay(cfg)
        assert result.steps == 0
        assert result.status == "empty"
        assert result.success  # 空轨迹视为成功（无失败，只是无内容）


def _save_out_of_limit_trajectory(
    tmp_path: Path,
    j2_offset_rad: float = 1.0,
    name: str = "limit_violation.npz",
) -> Path:
    """保存 J2 超上限的轨迹（用于 I3 limit_violation 预检测试）。

    INIT_Q[1] = 1.4 rad (≈80°)，q_upper[1] = 90° = 1.5708 rad。
    j2_offset_rad=1.0 → J2 ≈ 80°+57°=137° > 90°，硬超限。

    Args:
        tmp_path: pytest 临时目录。
        j2_offset_rad: J2 偏移量（弧度），正值=超上限。
        name: 文件名。

    Returns:
        保存的文件路径。
    """
    import json

    n = 10
    q = np.tile(INIT_Q, (n, 1))  # (N, 6)
    q[:, 1] += j2_offset_rad  # J2 偏移 → 超限

    init_q = INIT_Q.copy()
    init_q[1] += j2_offset_rad  # init_q 也对应偏移，避免 pre_motion 触发 large_delta

    path = tmp_path / name
    np.savez(
        path,
        q_desired=q.copy(),
        q_actual=q.copy(),
        timestamps=np.arange(n) * DT,
        tcp_pos=np.zeros((n, 3)),
        ball_pos=np.zeros((n, 3)),
        init_q=init_q,
        init_q_left=INIT_Q_LEFT.copy(),
        dt=DT,
        hit_step=-1,
        metadata=json.dumps({"is_position_mode": True}),
    )
    return path


class TestRunReplayFailures:
    """run_replay 失败路径测试（C1 + I3 回归）。

    C1: safety_abort 必须返回 status="safety_abort"（早期 bug 是返回正步数）。
    I3: limit_violation 必须在 robot.connect / pre_motion 前拒绝。
    """

    def test_safety_abort_returns_failed_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SafetyMonitor 触发 → status='safety_abort', peak_step=0（第一步即失败）。

        回归 C1：原本 safety_failed=True 时仍返回正 step_count，消费方误判为成功。
        """
        from src.real import replay_pipeline

        traj_path = _save_test_trajectory(tmp_path, n_steps=10)

        class _FailSafety:
            """永远返回不安全的 SafetyMonitor 替身。"""

            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def is_safe(self, *args: object, **kwargs: object) -> bool:
                return False

        monkeypatch.setattr(replay_pipeline, "SafetyMonitor", _FailSafety)

        cfg = ReplayConfig(
            trajectory_path=traj_path,
            speed=1.0,
            use_actual=True,
            mock=True,
        )
        result = run_replay(cfg)

        assert result.status == "safety_abort"
        assert result.steps == 0  # 第一步即失败
        assert result.peak_step == 0
        assert not result.success

    def test_limit_violation_rejected_pre_flight(self, tmp_path: Path) -> None:
        """q_desired 超关节硬限位 → status='limit_violation', steps=0。

        回归 I3：超限轨迹必须在 robot.connect / pre_motion 前被拒绝。
        """
        traj_path = _save_out_of_limit_trajectory(tmp_path, j2_offset_rad=1.0)

        cfg = ReplayConfig(
            trajectory_path=traj_path,
            speed=1.0,
            use_actual=True,
            mock=True,
        )
        result = run_replay(cfg)

        assert result.status == "limit_violation"
        assert result.steps == 0
        assert not result.success
        # reason 应包含超限关节标识（J2 或"肩俯仰"）
        assert "J2" in result.reason or "肩俯仰" in result.reason
        assert "[超限]" in result.reason

    def test_force_mode_bypasses_limit_check(self, tmp_path: Path) -> None:
        """--force-mode 跳过静态限位预检（但运行时 SafetyMonitor 仍生效）。

        验证 force_mode 作为统一的安全绕过开关覆盖 limit_violation 静态检查。
        注意：force_mode 只跳过 *预检*，运行时 SafetyMonitor 仍会拦截真超限轨迹
        （由 test_safety_abort_returns_failed_status 单独覆盖运行时安全闭环）。
        因此本测试断言 status != "limit_violation"（即预检被绕过）。
        """
        traj_path = _save_out_of_limit_trajectory(tmp_path, j2_offset_rad=1.0)

        cfg = ReplayConfig(
            trajectory_path=traj_path,
            speed=1.0,
            use_actual=True,
            mock=True,
            force_mode=True,  # 关键：启用绕过
        )
        result = run_replay(cfg)

        # 预检被绕过：不应是 limit_violation
        assert result.status != "limit_violation"
        # 运行时 SafetyMonitor 仍会拦截超限（这是设计上的深度防御）
        assert result.status == "safety_abort"
        assert not result.success


def _save_torque_mode_trajectory(
    tmp_path: Path, n_steps: int = 5
) -> Path:
    """保存力矩模式轨迹（is_position_mode=False），用于 control_mode_mismatch 测试。"""
    import json

    q = np.tile(INIT_Q, (n_steps, 1))
    path = tmp_path / "torque_mode.npz"
    np.savez(
        path,
        q_desired=q.copy(),
        q_actual=q.copy(),
        timestamps=np.arange(n_steps) * DT,
        tcp_pos=np.zeros((n_steps, 3)),
        ball_pos=np.zeros((n_steps, 3)),
        init_q=INIT_Q.copy(),
        init_q_left=INIT_Q_LEFT.copy(),
        dt=DT,
        hit_step=-1,
        metadata=json.dumps({"is_position_mode": False}),  # 标记为力矩模式
    )
    return path


def _save_trajectory_without_q_actual(tmp_path: Path, n_steps: int = 5) -> Path:
    """保存 q_actual 为空的旧格式轨迹（空数组），用于 empty_q_actual 测试。

    TrajectoryRecorder.load 要求 q_actual 字段存在，所以保存 shape (0, 6) 的空数组。
    """
    import json

    q = np.tile(INIT_Q, (n_steps, 1))
    path = tmp_path / "no_q_actual.npz"
    np.savez(
        path,
        q_desired=q.copy(),
        q_actual=np.zeros((0, 6)),  # 空 q_actual（旧格式未记录）
        timestamps=np.arange(n_steps) * DT,
        tcp_pos=np.zeros((n_steps, 3)),
        ball_pos=np.zeros((n_steps, 3)),
        init_q=INIT_Q.copy(),
        init_q_left=INIT_Q_LEFT.copy(),
        dt=DT,
        hit_step=-1,
        metadata=json.dumps({"is_position_mode": True}),
    )
    return path


class TestRunReplayRemainingStatuses:
    """I2 回归测试：补全 ReplayResult 8 status 的剩余 4 个未测路径。

    已测：ok / empty / safety_abort / limit_violation(+force_mode)
    待测：control_mode_mismatch / empty_q_actual / connect_failed / pre_motion_aborted
    """

    def test_control_mode_mismatch_rejected(self, tmp_path: Path) -> None:
        """力矩模式轨迹（is_position_mode=False）且未 --force-mode → 拒绝。

        回归保护：防止误把力矩模式轨迹下发到仅支持位置控制的真机。
        """
        traj_path = _save_torque_mode_trajectory(tmp_path)

        cfg = ReplayConfig(
            trajectory_path=traj_path,
            speed=1.0,
            use_actual=False,
            mock=True,
        )
        result = run_replay(cfg)

        assert result.status == "control_mode_mismatch"
        assert result.steps == 0
        assert not result.success
        assert "--force-mode" in result.reason

    def test_empty_q_actual_with_use_actual(self, tmp_path: Path) -> None:
        """--use-actual 但 q_actual 为空（旧格式）→ 拒绝。

        回归保护：避免对未记录 q_actual 的旧轨迹误用空数组。
        """
        traj_path = _save_trajectory_without_q_actual(tmp_path)

        cfg = ReplayConfig(
            trajectory_path=traj_path,
            speed=1.0,
            use_actual=True,  # 关键：要求 q_actual
            mock=True,
        )
        result = run_replay(cfg)

        assert result.status == "empty_q_actual"
        assert result.steps == 0
        assert not result.success
        assert "q_actual" in result.reason

    def test_connect_failed_returns_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """robot.connect() 返回 False → status='connect_failed'。

        回归保护：避免连接失败时仍尝试下发轨迹。
        """
        from src.real import replay_pipeline

        traj_path = _save_test_trajectory(tmp_path, n_steps=5)

        # 包装 FakeRobot：connect 返回 False
        from src.real.fake_robot import FakeRobot

        def fake_connect(self) -> bool:
            return False

        monkeypatch.setattr(FakeRobot, "connect", fake_connect)

        cfg = ReplayConfig(
            trajectory_path=traj_path,
            speed=1.0,
            use_actual=True,
            mock=True,
        )
        result = run_replay(cfg)

        assert result.status == "connect_failed"
        assert result.steps == 0
        assert not result.success

    def test_pre_motion_declined_returns_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pre_motion 返回 False（用户拒绝 / 到位失败）→ status='pre_motion_aborted'。

        回归保护：pre_motion 失败时必须正确返回 pre_motion_aborted，不能误判为成功。

        注：mock 模式下 FakeRobot 初始姿态即 traj.init_q，pre_motion 自然跳过
        （delta < 0.5°）。为触发 pre_motion 失败路径，直接 monkeypatch 返回 False，
        测试 run_replay 的状态映射（不测 pre_motion 内部逻辑——那是 TestPreMotionWrapAround 的职责）。
        """
        from src.real import replay_pipeline

        traj_path = _save_test_trajectory(tmp_path, n_steps=5)

        # 模拟 pre_motion 返回 False（用户拒绝 / 到位失败）
        monkeypatch.setattr(replay_pipeline, "pre_motion", lambda *a, **kw: False)

        cfg = ReplayConfig(
            trajectory_path=traj_path,
            speed=1.0,
            use_actual=True,
            mock=True,
        )
        result = run_replay(cfg)

        assert result.status == "pre_motion_aborted"
        assert result.steps == 0
        assert not result.success
        assert "拒绝" in result.reason or "到位" in result.reason


class TestSinkReturnCodeSafety:
    """C1 回归测试：RobotSink.send 必须检查 send_joint_command 返回码。

    背景：原实现 `self._robot.send_joint_command(state.q_desired)` 丢弃返回值，
    SDK 通过非零返回码报错时（控制器拒指令、通信超时等）被静默吞掉，
    100Hz 主循环继续推进 setpoint → 控制器保持上一帧 + Sink 推进 → 物理风险。
    修复：检查 ret != 0 时 slow_stop + return False，与 pre_motion 行为一致。
    """

    def test_nonzero_return_code_aborts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """send_joint_command 返回 -1 中流 → safety_abort + peak_step 正确。

        关键断言：
        - status == 'safety_abort'（不是 'ok' 误判）
        - peak_step == 3（前 3 步成功，第 4 步失败）
        - steps == 3
        """
        traj_path = _save_test_trajectory(tmp_path, n_steps=10)

        # 包装 FakeRobot.send_joint_command：第 4 步起返回 -1
        from src.real.fake_robot import FakeRobot

        # 保存原实现以复用（在 monkeypatch 之前保存）
        original_send_impl = FakeRobot.send_joint_command
        call_count = {"n": 0}

        def patched_send(self, q_desired):
            call_count["n"] += 1
            if call_count["n"] >= 4:
                return -1  # 模拟 SDK 拒指令
            # 复用原逻辑保持 arm_state 正确
            return original_send_impl(self, q_desired)

        monkeypatch.setattr(FakeRobot, "send_joint_command", patched_send)

        cfg = ReplayConfig(
            trajectory_path=traj_path,
            speed=1.0,
            use_actual=True,
            mock=True,
        )
        result = run_replay(cfg)

        assert result.status == "safety_abort", (
            f"期望 safety_abort（C1 修复后），实际 {result.status}"
        )
        assert result.steps == 3, f"期望前 3 步成功，实际 {result.steps}"
        assert result.peak_step == 3
        assert not result.success


class TestPreMotionWrapAround:
    """I1 回归测试：pre_motion J6 wrap-around 处理。

    背景：原实现 `q_current + s * (init_q - q_current)` 在 J6 等边界 wrap 关节
    （真机范围 [-180°, 180°]）上会走长弧。例如 q_current[J6]=170°,
    init_q[J6]=-170° 时，线性插值走 340° 而非短弧 20°。
    手动 jog 后真机重演可能触发 280°+ 大回旋 → 物理风险。

    修复：用 arctan2(sin, cos) 折回到 [-π, π]，对非 wrap 关节 idempotent。
    """

    def test_j6_takes_short_path(self) -> None:
        """J6 从 +170° 到 -170° 应走 20° 短弧，而非 340° 长弧。

        端到端验证 pre_motion 内部插值路径。使用 FakeRobot 记录命令历史，
        断言所有中间 J6 角度都走短弧（不经过 ±π 之外）。
        """
        from src.real.fake_robot import FakeRobot
        from src.real.replay_pipeline import pre_motion
        from src.real.trajectory_types import ReplayTrajectory

        # 构造 FakeRobot，初始 J6 = +170° (≈2.967 rad)
        q_current = np.zeros(6)
        q_current[5] = np.radians(170.0)
        robot = FakeRobot(init_q=q_current, dt=0.005)
        robot.connect()

        # 目标 J6 = -170° (≈-2.967 rad)，short path = 20°
        init_q = np.zeros(6)
        init_q[5] = np.radians(-170.0)

        traj = ReplayTrajectory(
            q_desired=init_q.reshape(1, 6).copy(),
            q_actual=init_q.reshape(1, 6).copy(),
            timestamps=np.array([0.0]),
            tcp_pos=np.zeros((1, 3)),
            ball_pos=np.zeros((1, 3)),
            init_q=init_q,
            init_q_left=np.zeros(6),
            dt=0.005,
            hit_step=-1,
            metadata={"is_position_mode": True},
        )

        ok = pre_motion(robot, traj, duration_s=1.0, rate_hz=50)
        assert ok

        # 检查所有命令的 J6 走短弧：从 +170° 穿过 180° 到 -170°
        # （数学上 +190° ≡ -170°，这是连续最短路径，不是长弧）
        commands = robot.command_history
        assert len(commands) > 0
        j6_degrees = np.degrees([c[5] for c in commands])

        # 关键断言：相邻命令间 J6 步进的总和 ≈ 20°（短弧），而非 340°（长弧）
        # 短弧总长度 = |+170° - (-170°)| 经最短路径 = 20°
        # 长弧总长度 ≈ 340°（如果走"线性数值差"会经过 0°）
        step_increments = np.abs(np.diff(j6_degrees))
        # 单步可能跨越 ±180° 边界（如 +179° → +181°），实际等价于 -179°
        # 所以每个增量需要折回 [-180°, 180°]
        step_increments_wrapped = np.array([
            ((inc + 180.0) % 360.0) - 180.0 if inc > 180.0 else inc
            for inc in step_increments
        ])
        step_increments_wrapped = np.abs(step_increments_wrapped)
        total_path = float(np.sum(step_increments_wrapped))
        assert total_path < 50.0, (
            f"J6 路径总长 {total_path:.1f}° > 50°，疑似走了长弧（应 ≈ 20°）"
        )

        # 第一帧接近 +170°
        first = j6_degrees[0]
        first_wrapped = ((first + 180.0) % 360.0) - 180.0
        assert abs(first_wrapped - 170.0) < 5.0
        # 最后一帧接近 -170°（或等价的 +190°）
        last = j6_degrees[-1]
        last_wrapped = ((last + 180.0) % 360.0) - 180.0
        assert abs(last_wrapped - (-170.0)) < 5.0

    def test_non_wrap_joints_idempotent(self) -> None:
        """arctan2 对非 wrap 关节（J2 ∈ [-270°, 90°]）等价于原 delta。

        验证：对 J2 偏移 30° 的场景，wrap-around 修复前后行为一致。
        """
        # arctan2(sin(x), cos(x)) 对 x ∈ [-π, π] 恒等
        deltas = np.array([np.radians(30), np.radians(-30), np.radians(0)])
        wrapped = np.arctan2(np.sin(deltas), np.cos(deltas))
        np.testing.assert_allclose(wrapped, deltas, atol=1e-12)
