"""RealRunner 真机部署主循环编排器测试。

使用 FakeRobot + SimulatedBallSensor + PlanningEnv 构建完整 Mock 闭环，
验证 start/step/stop 分步接口的端到端行为。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from src.ilqt.async_replanner import AsyncReplanner
from src.ilqt.planning_env import PlanningEnv
from src.ilqt.replan_core import do_replan
from src.ilqt.tube_types import ReplanState
from src.real.adaptive_timer import AdaptiveTimer
from src.real.ball_perceiver import BallPerceiver
from src.real.ball_sensor import SimulatedBallSensor
from src.real.config import RealRobotConfig
from src.real.fake_robot import FakeRobot
from src.real.real_runner import RealRunner
from src.real.runner_factory import (
    DT,
    INIT_Q,
    INIT_Q_LEFT,
    KD,
    KP,
    _build_real_robot_mpc_config,
    build_robot_limits,
    build_solver,
)
from src.real.safety_monitor import SafetyMonitor

logger = logging.getLogger(__name__)

_CFG = RealRobotConfig()
_CFG.max_tcp_speed = 10.0  # 测试用宽松 TCP（安全测试走 _force_unsafe）


def _build_test_runner(
    ball_pos: np.ndarray | None = None,
    ball_vel: np.ndarray | None = None,
) -> RealRunner:
    """构建完整 Mock 测试 runner。

    组装 PlanningEnv（位置模式）+ FakeRobot + SimulatedBallSensor +
    BallPerceiver + SafetyMonitor + AsyncReplanner + AdaptiveTimer。

    Args:
        ball_pos: 初始球位置，默认 cand1（可达，k_hit≈97）。
        ball_vel: 初始球速度。

    Returns:
        已组装但尚未 start 的 RealRunner。
    """
    if ball_pos is None:
        ball_pos = np.array([0.0, -1.5, 1.8], dtype=np.float64)
    if ball_vel is None:
        ball_vel = np.array([0.0, 2.0, 1.0], dtype=np.float64)

    # 1. 规划环境（位置模式）
    env = PlanningEnv(dt=DT)
    env.init_q_left = INIT_Q_LEFT.copy()  # 维持左臂位姿（C++ 线性化需要）
    env.configure_actuator_mode("position", kp=KP, kd=KD)
    env.configure_feedforward(True)
    env.reset(INIT_Q)
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left

    # 2. 假机器人
    robot = FakeRobot(init_q=INIT_Q, dt=DT)

    # 3. 球传感器 + 推入初始观测
    sensor = SimulatedBallSensor()
    sensor.start()

    # 4. 球感知器（bootstrap：推两次观测 + 两次 update，使有限差分速度可用）
    perceiver = BallPerceiver(sensor, estimator_config=None, dt=DT)
    sensor.push(ball_pos, 0.0)
    perceiver.update()  # 第一次：_last_pos=None → vel=0，存储基准点
    t_obs = 0.02  # 模拟传感器 50Hz 间隔
    p_obs = ball_pos + ball_vel * t_obs
    sensor.push(p_obs, t_obs)
    perceiver.update()  # 第二次：有限差分得真实速度，KF 滤波

    # 5. 安全监控（限位与规划器 RobotLimits 含裕度一致，避免边界抖动误判）
    robot_limits_pre = build_robot_limits(env, _CFG)
    safety_cfg = RealRobotConfig()
    safety_cfg.q_lower = robot_limits_pre.q_lower.copy()
    safety_cfg.q_upper = robot_limits_pre.q_upper.copy()
    safety_cfg.max_qdot = robot_limits_pre.qdot_max.copy()
    safety_cfg.max_tcp_speed = float(robot_limits_pre.max_tcp_speed)
    safety = SafetyMonitor(safety_cfg, robot=robot)

    # 6. 规划方向（来球反方向，用于 replan_state.current_n_des）
    ball_vel_norm = float(np.linalg.norm(ball_vel))
    if ball_vel_norm > 1e-6:
        d_hat = -ball_vel / ball_vel_norm
    else:
        d_hat = np.array([0.0, 1.0, 0.0])

    # 7. 规划配置：构建 MPCConfig + 直接传 RealRunner（无翻译层）
    robot_limits = robot_limits_pre
    solver = build_solver()
    mpc_config = _build_real_robot_mpc_config(_CFG)

    # 8. 重规划状态
    replan_state = ReplanState(
        k_hit_new=mpc_config.total_horizon,
        p_hit_new=ball_pos.copy(),
        v_ball_hit_new=ball_vel.copy(),
        current_n_des=d_hat.copy(),
        U_prev=np.zeros((0, env.NU)),
        is_first_plan=True,
    )

    # 9. 异步重规划器（静态注入 config/robot_limits/solver）
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
    replanner = AsyncReplanner(
        env, do_replan, mpc_config, robot_limits, solver,
        state=replan_state, model_path=model_path,
    )

    # 10. 自适应定时器
    timer = AdaptiveTimer(target_hz=100.0)

    return RealRunner(
        env=env,
        robot=robot,
        ball_perceiver=perceiver,
        safety=safety,
        replanner=replanner,
        config=mpc_config,
        robot_limits=robot_limits,
        solver=solver,
        timer=timer,
        replan_state=replan_state,
    )


class TestRealRunnerSmoke:
    """轮 1：start → step → stop 烟雾测试。"""

    def test_start_step_stop_smoke(self) -> None:
        """start → step × 1 → stop 不崩溃，返回 dict。"""
        runner = _build_test_runner()
        runner.start()
        info = runner.step()
        assert isinstance(info, dict)
        assert "u_cmd" in info
        assert "safe" in info
        metrics = runner.stop()
        assert isinstance(metrics, dict)
        assert "total_steps" in metrics


class TestRealRunnerCommandLimits:
    """轮 2：step 返回的 u_cmd 在合理关节范围内。"""

    def test_step_command_within_limits(self) -> None:
        """连续 5 步 u_cmd 均在 [-π, π] 范围内（dq_max 限幅生效）。"""
        runner = _build_test_runner()
        runner.start()
        for _ in range(5):
            info = runner.step()
            assert np.all(np.abs(info["u_cmd"]) < np.pi)
        assert runner._safe_step_count == 5
        runner.stop()


class TestRealRunnerSafetyFailure:
    """轮 3：安全检查失败时 step 返回 safe=False, done 变 True。"""

    def test_safety_failure_stops_episode(self) -> None:
        """注入 _force_unsafe=True → step 返回 safe=False 且 runner.done。"""
        runner = _build_test_runner()
        runner.start()
        runner._force_unsafe = True
        info = runner.step()
        assert not info["safe"]
        assert info["done"]
        assert runner.done
        assert runner.safety_failed
        metrics = runner.stop()
        assert metrics["safety_failed"] is True
        assert metrics["safe_steps"] == 0


class TestRealRunnerBallUnreachable:
    """轮 4：球飞出工作空间 → 优雅退出。"""

    def test_ball_unreachable_graceful_exit(self) -> None:
        """球在 [10,10,10]（远离工作空间）→ start 即判不可达 → stop 优雅退出。"""
        far_pos = np.array([10.0, 10.0, 10.0])
        far_vel = np.array([0.0, -1.0, 0.0])
        runner = _build_test_runner(ball_pos=far_pos, ball_vel=far_vel)
        runner.start()
        # 首次规划即判不可达，done 应为 True
        assert runner.ball_unreachable
        assert runner.done
        # 循环 step 不崩溃，立即退出
        for _ in range(3):
            runner.step()
            if runner.done:
                break
        metrics = runner.stop()
        assert metrics["ball_unreachable"] is True
        assert metrics["safe_steps"] == 0


class TestRealRunnerEpisodeMode:
    """轮 5：run_episode 模式 — 内部用 EpisodeRunner 编排。"""

    def test_runner_episode_mode(self) -> None:
        """run_episode(10) 返回 dict，含 total_steps/safe_steps 键（EpisodeRunner 编排）。"""
        runner = _build_test_runner()
        metrics = runner.run_episode(10)
        assert isinstance(metrics, dict)
        assert "total_steps" in metrics
        assert "safe_steps" in metrics


class TestRealRunnerTimerPacing:
    """轮 6：step() 通过 AdaptiveTimer 控制节奏。"""

    def test_step_calls_timer_tick_start_and_end(self) -> None:
        """step() 调用 timer.tick_start() 和 tick_end() 各一次。"""
        runner = _build_test_runner()
        runner.start()

        calls: list[str] = []
        runner._timer.tick_start = lambda: calls.append("start")
        runner._timer.tick_end = lambda: (calls.append("end"), 0.0)[1]

        runner.step()

        assert calls == ["start", "end"]
        runner.stop()

    def test_step_sleeps_when_timer_returns_positive(
        self, monkeypatch
    ) -> None:
        """tick_end 返回正值 → time.sleep 被调用一次。"""
        runner = _build_test_runner()
        runner.start()
        runner._timer.tick_end = lambda: 0.002

        sleeps: list[float] = []
        monkeypatch.setattr(
            "src.real.real_runner.time.sleep",
            lambda dt: sleeps.append(dt),
        )
        runner.step()

        assert len(sleeps) == 1
        assert sleeps[0] == pytest.approx(0.002)
        runner.stop()

    def test_step_no_sleep_when_timer_returns_zero(
        self, monkeypatch
    ) -> None:
        """tick_end 返回 0.0 → time.sleep 不被调用。"""
        runner = _build_test_runner()
        runner.start()
        runner._timer.tick_end = lambda: 0.0

        sleeps: list[float] = []
        monkeypatch.setattr(
            "src.real.real_runner.time.sleep",
            lambda dt: sleeps.append(dt),
        )
        runner.step()

        assert len(sleeps) == 0
        runner.stop()
