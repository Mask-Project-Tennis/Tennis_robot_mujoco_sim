"""do_replan 直接单测。

绕过 AsyncReplanner 后台线程，直接调用 do_replan 验证：
- 可达球 → solver_ok=True, U_buffer 非空, k_hit_new > 0
- 不可达球 → ball_unreachable=True

工厂函数与共享常量复用自 src/real/runner_factory.py。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ilqt.async_replanner import AsyncReplanner, PlanRequest
from src.ilqt.planning_env import PlanningEnv
from src.ilqt.replan_core import do_replan
from src.ilqt.tube_types import ReplanState
from src.real.config import RealRobotConfig
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

_CFG = RealRobotConfig()


def _build_env() -> PlanningEnv:
    """构建位置模式 PlanningEnv（左臂位姿已设置）。"""
    env = PlanningEnv(dt=DT)
    env.init_q_left = INIT_Q_LEFT.copy()
    env.configure_actuator_mode("position", kp=KP, kd=KD)
    env.configure_feedforward(True)
    env.reset(INIT_Q)
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left
    return env


def _build_planning_setup(
    ball_pos: np.ndarray, ball_vel: np.ndarray
) -> tuple[PlanningEnv, PlanningEnv, AsyncReplanner, np.ndarray, np.ndarray]:
    """构建 do_replan 直接调用所需的 env / config / env_plan / replanner。

    Args:
        ball_pos: 球初始位置。
        ball_vel: 球初始速度。

    Returns:
        (env, env_plan, replanner, d_hat, v_hit_desired)。
    """
    env = _build_env()
    ball_vel_norm = float(np.linalg.norm(ball_vel))
    d_hat = (
        -ball_vel / ball_vel_norm
        if ball_vel_norm > 1e-6
        else np.array([0.0, 1.0, 0.0])
    )
    v_hit_desired = 1.8 * d_hat

    robot_limits = build_robot_limits(env, _CFG)
    solver = build_solver()
    config = _build_real_robot_mpc_config(_CFG)

    # 创建 env_plan（独立 MjData，由 AsyncReplanner 延迟构建）
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
    replanner = AsyncReplanner(
        env, do_replan, config, robot_limits, solver,
        state=ReplanState(), model_path=model_path,
    )
    env_plan = replanner._ensure_env_plan()
    env_plan.init_q_left = INIT_Q_LEFT.copy()
    env_plan.configure_actuator_mode("position", kp=KP, kd=KD)
    env_plan.configure_feedforward(True)

    return env, env_plan, replanner, d_hat, v_hit_desired


def test_replan_reachable_ball() -> None:
    """可达球 → solver_ok=True, U_buffer 非空, k_hit_new > 0。"""
    ball_pos = np.array([0.0, -1.5, 1.8])
    ball_vel = np.array([0.0, 2.0, 1.0])
    env, env_plan, replanner, d_hat, v_hit_desired = _build_planning_setup(ball_pos, ball_vel)

    robot_limits = build_robot_limits(env, _CFG)
    solver = build_solver()
    config = _build_real_robot_mpc_config(_CFG)

    arm_state = env.get_arm_state()
    request = PlanRequest(
        x_current=arm_state,
        ball_pos=ball_pos,
        ball_vel=ball_vel,
        step=0,
        k_hit_current=0,
        U_prev=np.zeros((0, 6)),
        p_hit_current=ball_pos,
        v_hit_desired=v_hit_desired,
        n_des_current=d_hat,
        d_hat=d_hat,
        is_first_plan=True,
    )

    state = ReplanState()
    result = do_replan(request, env_plan, state, config, robot_limits, solver)

    assert result.solver_ok, "solver 应成功"
    assert len(result.U_buffer) > 0, "U_buffer 应非空"
    assert result.k_hit_new > 0, "k_hit_new 应 > 0"

    replanner.stop()


def test_replan_unreachable_ball() -> None:
    """远处球 → ball_unreachable=True。"""
    ball_pos = np.array([10.0, 10.0, 10.0])
    ball_vel = np.array([0.0, 0.0, 0.0])
    env, env_plan, replanner, d_hat, v_hit_desired = _build_planning_setup(ball_pos, ball_vel)

    robot_limits = build_robot_limits(env, _CFG)
    solver = build_solver()
    config = _build_real_robot_mpc_config(_CFG)

    arm_state = env.get_arm_state()
    request = PlanRequest(
        x_current=arm_state,
        ball_pos=ball_pos,
        ball_vel=ball_vel,
        step=0,
        k_hit_current=0,
        U_prev=np.zeros((0, 6)),
        p_hit_current=ball_pos,
        v_hit_desired=v_hit_desired,
        n_des_current=d_hat,
        d_hat=d_hat,
        is_first_plan=True,
    )

    state = ReplanState()
    result = do_replan(request, env_plan, state, config, robot_limits, solver)

    assert result.ball_unreachable, "远处球应标记不可达"

    replanner.stop()
