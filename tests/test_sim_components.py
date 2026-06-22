"""仿真管线组件测试 — SimPerception/PredictiveSafety/BasicSafety。

覆盖组件的核心行为与 Protocol 契约：
  - SimPerception: 从环境读球状态（含 obs_gate 回调路径）
  - PredictiveSafetyFilter: beta 递降 + X 平面墙 + 紧急制动 fallback
  - BasicSafetyFilter: 无预测的纯限位检查
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ilqt.planning_env import PlanningEnv
from src.real.runner_factory import (
    DT,
    INIT_Q,
    INIT_Q_LEFT,
    KD,
    KP,
    build_robot_limits,
)
from src.ilqt.components.sim_perception import SimPerception
from src.ilqt.components.predictive_safety import PredictiveSafetyFilter
from src.ilqt.components.basic_safety import BasicSafetyFilter
from src.ilqt.components.protocols import (
    PerceptionComponent,
    SafetyComponent,
)


def _build_planning_env() -> PlanningEnv:
    """构建位置模式 PlanningEnv（INIT_Q 位姿满足 X 平面墙，左臂位姿已设置）。"""
    env = PlanningEnv(dt=DT)
    env.init_q_left = INIT_Q_LEFT.copy()
    env.configure_actuator_mode("position", kp=KP, kd=KD)
    env.configure_feedforward(True)
    env.reset(INIT_Q)
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left
    return env


# ── Test 1: SimPerception ──────────────────────────────────────────────────


def test_sim_perception_reads_ball() -> None:
    """SimPerception 无 obs_gate 时直接从环境读球状态真值。"""
    env = PlanningEnv(dt=DT)
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.0, 0.0, -1.0])
    env.set_ball_state(pos, vel)
    perc = SimPerception(env)
    ball = perc.get_ball_state()
    assert ball is not None
    got_pos, got_vel = ball
    np.testing.assert_array_almost_equal(got_pos, pos)
    np.testing.assert_array_almost_equal(got_vel, vel)


def test_sim_perception_obs_gate() -> None:
    """obs_gate 回调对原始 (pos, vel) 做变换后再返回。"""
    env = PlanningEnv(dt=DT)
    env.set_ball_state(np.zeros(3), np.array([1.0, -1.0, 0.0]))

    def gate(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return pos + 1.0, vel * 2.0

    perc = SimPerception(env, obs_gate=gate)
    ball = perc.get_ball_state()
    assert ball is not None
    got_pos, got_vel = ball
    np.testing.assert_array_almost_equal(got_pos, np.ones(3))
    np.testing.assert_array_almost_equal(got_vel, np.array([2.0, -2.0, 0.0]))


# ── Test 2: PredictiveSafetyFilter ─────────────────────────────────────────


def test_predictive_safety_beta_descent() -> None:
    """INIT_Q 位姿 + 限位内 u → 通过安全检查（无需 beta 递降）。"""
    env = _build_planning_env()
    limits = build_robot_limits(env)
    safety = PredictiveSafetyFilter(env, limits, is_position_mode=True)
    safety.k_hit_remaining = 99  # 无终段豁免，全程施加 qdot/TCP 检查

    u_cmd = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    arm_state = env.get_arm_state()
    safe_u, is_safe = safety.filter(u_cmd, arm_state)

    assert is_safe
    # 预测应在 env 状态被恢复后不改变读数
    np.testing.assert_array_almost_equal(env.get_arm_state(), arm_state)


def test_predictive_safety_emergency_fallback() -> None:
    """当前位姿即违反 q 上限 → 全部 beta 失败 → 紧急 hold q（位置模式，视为安全）。

    紧急制动是最后一道防线：当所有 beta（含 beta=0 保持）预测都违反硬约束时触发。
    本测试人为收紧 q_upper 至当前位姿之下，使任何预测都超上限。
    """
    env = _build_planning_env()
    limits = build_robot_limits(env)
    # 收紧 q 上限：使 INIT_Q 各关节 q 均超出上限 → 任何预测都违反 q upper
    limits.q_upper = np.array([-10.0] * 6)
    safety = PredictiveSafetyFilter(env, limits, is_position_mode=True)
    safety.k_hit_remaining = 99

    arm_state = env.get_arm_state()
    u_cmd = np.array([0.1] * 6)
    safe_u, is_safe = safety.filter(u_cmd, arm_state)

    assert is_safe  # 紧急保持视为安全（episode 不中止）
    assert safety.emergency_stop_count == 1
    # 位置模式紧急制动 → 保持当前 q
    np.testing.assert_array_almost_equal(safe_u, arm_state[:6])
    # env 状态应被恢复
    np.testing.assert_array_almost_equal(env.get_arm_state(), arm_state)


# ── Test 3: BasicSafetyFilter ──────────────────────────────────────────────


def test_basic_safety_limit_check() -> None:
    """超限 q_desired → is_safe=False；限位内 → True。"""
    safety = BasicSafetyFilter(
        q_lower=-np.ones(6),
        q_upper=np.ones(6),
        max_qdot=np.ones(6) * 10.0,
    )
    arm = np.zeros(12)  # q=0, qdot=0

    u_over = np.array([5.0] * 6)  # 超出 [-1, 1]
    _, is_safe = safety.filter(u_over, arm)
    assert not is_safe

    u_ok = np.array([0.5] * 6)  # 在限位内
    safe_u, is_safe = safety.filter(u_ok, arm)
    assert is_safe
    # 通过时 u 不被修改
    np.testing.assert_array_almost_equal(safe_u, u_ok)


def test_basic_safety_qdot_violation() -> None:
    """关节速度超限 → is_safe=False。"""
    safety = BasicSafetyFilter(
        q_lower=-np.ones(6),
        q_upper=np.ones(6),
        max_qdot=np.ones(6) * 1.0,
    )
    arm = np.zeros(12)
    arm[6:] = 5.0  # qdot 超限
    _, is_safe = safety.filter(np.zeros(6), arm)
    assert not is_safe


# ── Protocol 契约 ───────────────────────────────────────────────────────────


def test_components_satisfy_protocols() -> None:
    """所有组件满足对应的 @runtime_checkable Protocol。"""
    env = _build_planning_env()
    limits = build_robot_limits(env)

    assert isinstance(SimPerception(env), PerceptionComponent)
    assert isinstance(
        PredictiveSafetyFilter(env, limits, is_position_mode=True), SafetyComponent
    )
    assert isinstance(
        BasicSafetyFilter(-np.ones(6), np.ones(6), np.ones(6) * 10.0), SafetyComponent
    )


# ── BasicSafetyFilter 构造警告 ──────────────────────────────────────────────


class TestBasicSafetyFilterWarning:
    """BasicSafetyFilter 构造时应发出 RuntimeWarning（标记为不完整）。"""

    def test_construction_emits_warning(self):
        """构造 BasicSafetyFilter 时触发 RuntimeWarning，警告缺少 TCP 检查。"""
        import warnings
        import numpy as np
        from src.ilqt.components.basic_safety import BasicSafetyFilter
        with pytest.warns(RuntimeWarning, match="TCP"):
            BasicSafetyFilter(
                q_lower=-np.ones(6), q_upper=np.ones(6),
                max_qdot=np.ones(6) * 10,
            )
