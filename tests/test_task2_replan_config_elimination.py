"""Task 2 RED 测试：验证 ReplanConfig 消除 + Q_tcp_soft 管道接通。

本测试文件先于实现编写（TDD RED 阶段），验证 5 项契约：
- RED 4: PlanRequest 包含 d_hat/d_follow 字段
- RED 5: do_replan 接受 MPCConfig 而非 dict
- RED 6: Q_tcp_soft 从 MPCConfig 到达 HittingCost
- RED 7: MPCConfig 不含 max_tcp_speed 死字段
- RED 8: ReplanConfig 模块已删除
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ilqt.async_replanner import PlanRequest
from src.ilqt.planning_env import PlanningEnv
from src.ilqt.tube_types import ReplanState
from src.real.config import RealRobotConfig
from src.real.runner_factory import (
    DT,
    INIT_Q,
    INIT_Q_LEFT,
    KD,
    KP,
    build_robot_limits,
    build_solver,
)


# ── RED 4: PlanRequest 包含 d_hat/d_follow ──────────────────────────────


def test_plan_request_has_d_hat() -> None:
    """PlanRequest 包含 d_hat 字段（替代 cfg['d_hat']）。"""
    req = PlanRequest(
        x_current=np.zeros(12),
        ball_pos=np.zeros(3),
        ball_vel=np.zeros(3),
        step=0,
        k_hit_current=100,
        U_prev=np.zeros((0, 6)),
        p_hit_current=np.zeros(3),
        v_hit_desired=np.zeros(3),
        n_des_current=np.zeros(3),
        d_hat=np.array([0.0, 1.0, 0.0]),
    )
    assert np.allclose(req.d_hat, [0, 1, 0])
    assert req.d_follow is None  # 默认 None


# ── RED 7: MPCConfig 无 max_tcp_speed 死字段 ────────────────────────────


def test_mpc_config_no_max_tcp_speed() -> None:
    """MPCConfig 不应包含 max_tcp_speed（死代码已删除）。"""
    import dataclasses

    from src.ilqt.mpc_controller import MPCConfig

    names = {f.name for f in dataclasses.fields(MPCConfig)}
    assert "max_tcp_speed" not in names


# ── RED 8: ReplanConfig 模块已删除 ──────────────────────────────────────


def test_replan_config_deleted() -> None:
    """ReplanConfig 模块已删除（浅层翻译层消除）。"""
    with pytest.raises(ImportError):
        from src.ilqt.replan_config import ReplanConfig  # noqa: F401


# ── RED 5 & 6: do_replan 接受 MPCConfig + Q_tcp_soft 到达 HittingCost ──
# 这两个测试需要完整的 env_plan / state / config / robot_limits / solver，
# 共享构建逻辑集中在此。


def _build_plan_test_setup() -> tuple:
    """构建 RED 5/6 所需的 do_replan 输入（env / state / config / limits / solver）。

    Returns:
        (env_plan, state, config, robot_limits, solver, request)
    """
    from src.ilqt.mpc_controller import MPCConfig
    from src.ilqt.replan_core import do_replan  # noqa: F401  验证可导入

    env = PlanningEnv(dt=DT)
    env.init_q_left = INIT_Q_LEFT.copy()
    env.configure_actuator_mode("position", kp=KP, kd=KD)
    env.configure_feedforward(True)
    env.reset(INIT_Q)
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left

    cfg = RealRobotConfig()
    robot_limits = build_robot_limits(env, cfg)
    solver = build_solver()

    config = MPCConfig(
        is_position_mode=True,
        dt=DT,
        total_horizon=200,
        fixed_horizon=60,
        replan_interval=20,
        max_iter_per_plan=3,
        first_plan_iters=5,
        near_plan_iters=2,
        near_threshold=80,
        Q_tcp_soft=5000.0,
    )

    ball_pos = np.array([0.0, -1.5, 1.8])
    ball_vel = np.array([0.0, 2.0, 1.0])
    ball_vel_norm = float(np.linalg.norm(ball_vel))
    d_hat = -ball_vel / ball_vel_norm if ball_vel_norm > 1e-6 else np.array([0.0, 1.0, 0.0])
    v_hit_desired = 1.8 * d_hat

    arm_state = env.get_arm_state()
    request = PlanRequest(
        x_current=arm_state,
        ball_pos=ball_pos,
        ball_vel=ball_vel,
        step=0,
        k_hit_current=200,
        U_prev=np.zeros((0, 6)),
        p_hit_current=ball_pos,
        v_hit_desired=v_hit_desired,
        n_des_current=d_hat,
        is_first_plan=True,
        d_hat=d_hat,
    )
    state = ReplanState()
    return env, state, config, robot_limits, solver, request


def test_do_replan_accepts_mpc_config() -> None:
    """do_replan 接受 MPCConfig + 类型化参数，而非 cfg dict。"""
    from src.ilqt.async_replanner import PlanResult
    from src.ilqt.replan_core import do_replan

    env_plan, state, config, robot_limits, solver, request = _build_plan_test_setup()

    result = do_replan(
        request,
        env_plan,
        state,
        config,  # MPCConfig 直接传入（新签名）
        robot_limits,
        solver,
    )
    assert isinstance(result, PlanResult)


def test_q_tcp_soft_flows_to_hitting_cost(monkeypatch) -> None:
    """MPCConfig.Q_tcp_soft 通过 do_replan 到达 HittingCost。"""
    from src.ilqt.cost import HittingCost
    from src.ilqt.replan_core import do_replan

    captured: dict = {}
    orig_init = HittingCost.__init__

    def spy(self, *a, **kw):
        captured["Q_tcp_soft"] = kw.get("Q_tcp_soft", 0.0)
        captured["tcp_threshold"] = kw.get("tcp_threshold", 1.44)
        captured["Q_qdot_limit"] = kw.get("Q_qdot_limit", 0.0)
        captured["qdot_limit_thresholds"] = kw.get("qdot_limit_thresholds", None)
        orig_init(self, *a, **kw)

    monkeypatch.setattr(HittingCost, "__init__", spy)

    env_plan, state, config, robot_limits, solver, request = _build_plan_test_setup()
    assert config.Q_tcp_soft == 5000.0  # 测试前置条件

    do_replan(request, env_plan, state, config, robot_limits, solver)

    assert captured["Q_tcp_soft"] == 5000.0
    assert captured["tcp_threshold"] > 0  # 从 robot_limits.max_tcp_speed 派生
    assert captured["tcp_threshold"] == pytest.approx(0.8 * robot_limits.max_tcp_speed)
    # qdot_limit_thresholds 从 robot_limits.qdot_max 派生
    assert captured["qdot_limit_thresholds"] is not None
