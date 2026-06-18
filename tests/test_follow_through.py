"""随挥策略测试 — FollowThroughPolicy。

测试随挥触发条件（planned/contact/none）和控制计算。
"""

from __future__ import annotations

import numpy as np

from src.ilqt.planning_env import PlanningEnv
from src.ilqt.strategies.follow_through import (
    FollowContext,
    FollowThroughPolicy,
    NoFollowThrough,
    PlannedFollowThrough,
)
from src.real.runner_factory import DT, INIT_Q, INIT_Q_LEFT, KD, KP


def _build_env() -> PlanningEnv:
    """构建位置模式 PlanningEnv（左臂维持零位）。"""
    env = PlanningEnv(dt=DT)
    env.init_q_left = INIT_Q_LEFT.copy()
    env.configure_actuator_mode("position", kp=KP, kd=KD)
    env.configure_feedforward(True)
    env.reset(INIT_Q)
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left
    env.update_kinematics()
    return env


def _make_ctx(
    env: PlanningEnv,
    step_count: int = 50,
    mpc_horizon: int = 250,
    k_hit: int = 1,
    arm_state: np.ndarray | None = None,
    ball_pos: np.ndarray | None = None,
    d_follow: np.ndarray | None = None,
    v_hit_desired: np.ndarray | None = None,
) -> FollowContext:
    """构建测试用 FollowContext。"""
    if arm_state is None:
        arm_state = env.get_arm_state()
    if ball_pos is None:
        ball_pos = np.array([0.0, -0.6, 1.4])
    if d_follow is None:
        d_follow = np.array([0.0, 1.0, 0.0])
    if v_hit_desired is None:
        v_hit_desired = np.array([0.0, 1.8, 0.0])
    return FollowContext(
        step_count=step_count,
        mpc_horizon=mpc_horizon,
        k_hit=k_hit,
        arm_state=arm_state,
        ball_pos=ball_pos,
        d_follow=d_follow,
        v_hit_desired=v_hit_desired,
        env=env,
    )


# ──────────────────────────────────────────────────────────────────
# A1 轮 1: PlannedFollowThrough.should_trigger
# ──────────────────────────────────────────────────────────────────


class TestPlannedFollowThroughTrigger:
    """随挥触发条件测试。"""

    def test_planned_trigger_k_hit_le_1(self) -> None:
        """follow_trigger='planned', k_hit=1 → 触发（到达计划击打时刻）。"""
        env = _build_env()
        policy = PlannedFollowThrough(
            follow_through_steps=60, follow_trigger="planned",
            dt=DT, is_position_mode=True, NQ=6, NU=6,
        )
        ctx = _make_ctx(env, step_count=50, mpc_horizon=250, k_hit=1)
        assert policy.should_trigger(ctx) is True

    def test_planned_trigger_mpc_end(self) -> None:
        """step_count >= mpc_horizon → 强制触发（MPC 阶段结束）。"""
        env = _build_env()
        policy = PlannedFollowThrough(
            follow_through_steps=60, follow_trigger="planned",
            dt=DT, is_position_mode=True, NQ=6, NU=6,
        )
        ctx = _make_ctx(env, step_count=250, mpc_horizon=250, k_hit=50)
        assert policy.should_trigger(ctx) is True

    def test_no_trigger_when_follow_steps_zero(self) -> None:
        """follow_through_steps=0 → 不触发（随挥禁用）。"""
        env = _build_env()
        policy = PlannedFollowThrough(
            follow_through_steps=0, follow_trigger="planned",
            dt=DT, is_position_mode=True, NQ=6, NU=6,
        )
        ctx = _make_ctx(env, step_count=250, mpc_horizon=250, k_hit=1)
        assert policy.should_trigger(ctx) is False

    def test_no_trigger_when_already_started(self) -> None:
        """已触发后再次调用 → False（防重入）。"""
        env = _build_env()
        policy = PlannedFollowThrough(
            follow_through_steps=60, follow_trigger="planned",
            dt=DT, is_position_mode=True, NQ=6, NU=6,
        )
        ctx = _make_ctx(env, step_count=50, mpc_horizon=250, k_hit=1)
        assert policy.should_trigger(ctx) is True   # 首次触发
        assert policy.should_trigger(ctx) is False  # 已触发

    def test_no_trigger_k_hit_large(self) -> None:
        """k_hit=50 且未到 mpc_horizon → 不触发。"""
        env = _build_env()
        policy = PlannedFollowThrough(
            follow_through_steps=60, follow_trigger="planned",
            dt=DT, is_position_mode=True, NQ=6, NU=6,
        )
        ctx = _make_ctx(env, step_count=50, mpc_horizon=250, k_hit=50)
        assert policy.should_trigger(ctx) is False


# ──────────────────────────────────────────────────────────────────
# A1 轮 2: PlannedFollowThrough.compute_control
# ──────────────────────────────────────────────────────────────────


class TestPlannedFollowThroughControl:
    """随挥控制计算测试。"""

    def test_planned_control_position_mode(self) -> None:
        """位置模式：触发后 compute_control 返回合理 q_desired (6,)。"""
        env = _build_env()
        policy = PlannedFollowThrough(
            follow_through_steps=60, follow_trigger="planned",
            dt=DT, is_position_mode=True, NQ=6, NU=6,
        )
        ctx = _make_ctx(env, step_count=50, mpc_horizon=250, k_hit=1)
        policy.should_trigger(ctx)  # 触发并记录 p_ee_at_hit
        u_cmd = policy.compute_control(ctx, step_in_follow=0)
        assert u_cmd.shape == (6,)
        assert np.all(np.isfinite(u_cmd))

    def test_planned_control_advances_along_d_follow(self) -> None:
        """随挥沿 d_follow 方向推进（dt_follow 增大 → 末端目标前移）。"""
        env = _build_env()
        policy = PlannedFollowThrough(
            follow_through_steps=60, follow_trigger="planned",
            dt=DT, is_position_mode=True, NQ=6, NU=6,
        )
        ctx = _make_ctx(
            env, step_count=50, mpc_horizon=250, k_hit=1,
            d_follow=np.array([0.0, 1.0, 0.0]),
            v_hit_desired=np.array([0.0, 1.8, 0.0]),
        )
        policy.should_trigger(ctx)
        u0 = policy.compute_control(ctx, step_in_follow=0)
        u1 = policy.compute_control(ctx, step_in_follow=10)
        # 两个控制应不同（目标随时间前移）
        assert not np.allclose(u0, u1)

    def test_planned_control_torque_mode(self) -> None:
        """力矩模式：compute_control 返回力矩 (6,)。"""
        env = _build_env()
        policy = PlannedFollowThrough(
            follow_through_steps=60, follow_trigger="planned",
            dt=DT, is_position_mode=False, NQ=6, NU=6,
        )
        ctx = _make_ctx(env, step_count=50, mpc_horizon=250, k_hit=1)
        policy.should_trigger(ctx)
        u_cmd = policy.compute_control(ctx, step_in_follow=0)
        assert u_cmd.shape == (6,)
        assert np.all(np.isfinite(u_cmd))


# ──────────────────────────────────────────────────────────────────
# A1 轮 3: NoFollowThrough
# ──────────────────────────────────────────────────────────────────


class TestNoFollowThrough:
    """禁用随挥策略测试。"""

    def test_no_follow_never_triggers(self) -> None:
        """NoFollowThrough.should_trigger 始终 False。"""
        env = _build_env()
        policy = NoFollowThrough()
        ctx = _make_ctx(env, step_count=999, mpc_horizon=100, k_hit=0)
        assert policy.should_trigger(ctx) is False

    def test_no_follow_is_protocol(self) -> None:
        """NoFollowThrough 实现 FollowThroughPolicy Protocol。"""
        policy = NoFollowThrough()
        assert isinstance(policy, FollowThroughPolicy)


# ──────────────────────────────────────────────────────────────────
# A1 轮 4: is_done / reset / Protocol 一致性
# ──────────────────────────────────────────────────────────────────


class TestFollowThroughLifecycle:
    """随挥生命周期测试。"""

    def test_is_done_after_follow_steps(self) -> None:
        """step_in_follow > follow_through_steps → is_done=True。"""
        env = _build_env()
        policy = PlannedFollowThrough(
            follow_through_steps=60, follow_trigger="planned",
            dt=DT, is_position_mode=True, NQ=6, NU=6,
        )
        ctx = _make_ctx(env, step_count=50, mpc_horizon=250, k_hit=1)
        policy.should_trigger(ctx)
        assert policy.is_done(60, ctx) is False
        assert policy.is_done(61, ctx) is True

    def test_reset_clears_state(self) -> None:
        """reset 后可重新触发。"""
        env = _build_env()
        policy = PlannedFollowThrough(
            follow_through_steps=60, follow_trigger="planned",
            dt=DT, is_position_mode=True, NQ=6, NU=6,
        )
        ctx = _make_ctx(env, step_count=50, mpc_horizon=250, k_hit=1)
        policy.should_trigger(ctx)
        assert policy.should_trigger(ctx) is False  # 已触发
        policy.reset()
        assert policy.should_trigger(ctx) is True   # reset 后可重新触发

    def test_planned_is_protocol(self) -> None:
        """PlannedFollowThrough 实现 FollowThroughPolicy Protocol。"""
        policy = PlannedFollowThrough(
            follow_through_steps=60, follow_trigger="planned",
            dt=DT, is_position_mode=True, NQ=6, NU=6,
        )
        assert isinstance(policy, FollowThroughPolicy)
