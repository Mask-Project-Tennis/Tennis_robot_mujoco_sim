"""RealRunner 真机部署主循环编排器测试。

使用 FakeRobot + SimulatedBallSensor + PlanningEnv 构建完整 Mock 闭环，
验证 start/step/stop 分步接口的端到端行为。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.ilqt.async_replanner import AsyncReplanner
from src.ilqt.planning_env import PlanningEnv
from src.ilqt.replan_core import do_replan
from src.ilqt.robot_limits import RobotLimits
from src.ilqt.tube_types import ReplanState, TubeConfig
from src.real.adaptive_timer import AdaptiveTimer
from src.real.ball_perceiver import BallPerceiver
from src.real.ball_sensor import SimulatedBallSensor
from src.real.config import RealRobotConfig
from src.real.fake_robot import FakeRobot
from src.real.real_runner import RealRunner
from src.real.safety_monitor import SafetyMonitor

logger = logging.getLogger(__name__)

# ── 共享常量（对齐 V11 真机配置）──
DT: float = 0.005
INIT_Q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
INIT_Q_LEFT = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45], dtype=np.float64)
SHOULDER_POS = np.array([-0.1, -0.22693, 1.302645], dtype=np.float64)
WORKSPACE_RADIUS = 0.90
KP = np.array([200.0, 200.0, 100.0, 50.0, 50.0, 20.0], dtype=np.float64)
KD = np.array([20.0, 20.0, 10.0, 5.0, 5.0, 2.0], dtype=np.float64)


def _build_robot_limits(env: PlanningEnv) -> RobotLimits:
    """从 default.yaml 风格配置构建 RobotLimits。"""
    rl_cfg = {
        "q_min_deg": [-178, -130, -135, -178, -128, -360],
        "q_max_deg": [178, 130, 135, 178, 128, 360],
        "q_margin_deg": [2, 1, 3, 3, 3, 3],
        "qdot_max_deg_s": [180, 180, 225, 225, 225, 225],
        "qdot_scale": 1.0,
        "qddot_max_deg_s2": [400, 400, 500, 500, 500, 500],
        "qddot_scale": 0.85,
        "max_tcp_speed": 1.8,
        "terminal_exempt_steps": 20,
        "dq_max_fraction": 0.5,
    }
    return RobotLimits.from_config(
        rl_cfg, dt=DT, ctrlrange=env.model.actuator_ctrlrange[: env.NU]
    )


def _build_solver():
    """构建 ILQTSolver（优先 C++ 加速版）。"""
    try:
        from src.cpp.solver_cpp import ILQTSolver
    except ImportError:
        from src.ilqt.solver import ILQTSolver
    return ILQTSolver(
        {
            "max_iter": 10,
            "tol": 1e-4,
            "horizon": 60,
            "mu_min": 1e-6,
            "mu_max": 1e10,
            "mu_init": 0.01,
            "delta_0": 1.6,
            "alpha_list": [1.0, 0.5, 0.25, 0.1, 0.05, 0.01],
            "lin_eps": 1e-6,
        }
    )


def _build_replan_cfg(
    env: PlanningEnv,
    robot_limits: RobotLimits,
    solver,
    d_hat: np.ndarray,
    v_hit_desired: np.ndarray,
) -> dict:
    """构建 do_replan 所需的完整配置字典。"""
    total_horizon = 200
    return {
        # ── 必需键 ──
        "dt": DT,
        "shoulder_pos": SHOULDER_POS,
        "workspace_radius": WORKSPACE_RADIUS,
        "total_horizon": total_horizon,
        "fixed_horizon": 60,
        "replan_interval": 20,
        "max_iter_per_plan": 3,
        "first_plan_iters": 5,
        "near_plan_iters": 2,
        "near_threshold": 80,
        "R": 0.0001,
        "Q_p_scale_far": 5.0,
        "Q_v_scale_far": 3.0,
        "Q_p_scale_near": 8.0,
        "Q_v_scale_near": 120.0,
        "robot_limits": robot_limits,
        "solver": solver,
        "d_hat": d_hat,
        "v_hit_desired": v_hit_desired,
        "v_hit_at_contact": v_hit_desired,
        "hit_shift": 0.0,
        "follow_through_length": 0.0,
        "time_perturb_s": 0.0,
        "space_perturb_m": 0.0,
        # ── 可选键（真机位置模式）──
        "ablation_mode": "full",
        "is_position_mode": True,
        "use_backswing": False,
        "use_r_decay": False,
        "fix_joint5_angle": None,
        "backswing_offset": 0.0,
        "backswing_ratio": 0.3,
        "r_decay_ratio": 0.3,
        "racket_speed": 5.0,
        "normal_weight": 500000.0,
        "normal_flip": False,
        "max_tcp_speed": 1.8,
        "perturb_alpha_min": 0.0,
        "k_hit_total": total_horizon,
        "tube_cfg": TubeConfig(),
        "smooth_far": {"Q_qdot_mult": 1.0, "Q_qddot_mult": 1.0, "Q_du_mult": 1.0},
        "smooth_mid": {"Q_qdot_mult": 2.0, "Q_qddot_mult": 2.0, "Q_du_mult": 2.0},
        "smooth_near": {"Q_qdot_mult": 2.0, "Q_qddot_mult": 2.0, "Q_du_mult": 3.0},
    }


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
    robot_limits_pre = _build_robot_limits(env)
    safety_cfg = RealRobotConfig()
    safety_cfg.q_lower = robot_limits_pre.q_lower.copy()
    safety_cfg.q_upper = robot_limits_pre.q_upper.copy()
    safety_cfg.max_qdot = robot_limits_pre.qdot_max.copy()
    safety_cfg.max_tcp_speed = float(robot_limits_pre.max_tcp_speed)
    safety = SafetyMonitor(safety_cfg, robot=robot)

    # 6. 规划方向（来球反方向）
    ball_vel_norm = float(np.linalg.norm(ball_vel))
    if ball_vel_norm > 1e-6:
        d_hat = -ball_vel / ball_vel_norm
    else:
        d_hat = np.array([0.0, 1.0, 0.0])
    v_hit_desired = 1.8 * d_hat

    # 7. 规划配置
    robot_limits = robot_limits_pre
    solver = _build_solver()
    replan_cfg = _build_replan_cfg(env, robot_limits, solver, d_hat, v_hit_desired)

    # 8. 重规划状态
    replan_state = ReplanState(
        k_hit_new=replan_cfg["k_hit_total"],
        p_hit_new=ball_pos.copy(),
        v_ball_hit_new=ball_vel.copy(),
        current_n_des=d_hat.copy(),
        U_prev=np.zeros((0, env.NU)),
        is_first_plan=True,
    )

    # 9. 异步重规划器
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
    replanner = AsyncReplanner(
        env, do_replan, replan_cfg, state=replan_state, model_path=model_path
    )

    # 10. 自适应定时器
    timer = AdaptiveTimer(target_hz=200.0)

    return RealRunner(
        env=env,
        robot=robot,
        ball_perceiver=perceiver,
        safety=safety,
        replanner=replanner,
        replan_cfg=replan_cfg,
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
