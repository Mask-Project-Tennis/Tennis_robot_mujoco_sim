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
