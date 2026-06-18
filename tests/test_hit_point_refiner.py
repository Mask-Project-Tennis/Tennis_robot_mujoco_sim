"""击球点 refine 策略测试 — HitPointRefiner。

测试击球点可执行性后过滤（关节裕度检查 + tube 窗口搜索 + hysteresis 防抖）。
"""

from __future__ import annotations

import numpy as np

from src.ilqt.planning_env import PlanningEnv
from src.ilqt.robot_limits import RobotLimits
from src.ilqt.strategies.hit_point_refiner import (
    HitPointRefiner,
    HybridRefiner,
    NoRefinement,
    RefineResult,
)
from src.real.runner_factory import DT, INIT_Q, INIT_Q_LEFT, KD, KP, build_robot_limits


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


def _build_limits(env: PlanningEnv) -> RobotLimits:
    """构建 RobotLimits。"""
    return build_robot_limits(env)


# ──────────────────────────────────────────────────────────────────
# A2 轮 1: NoRefinement 直通
# ──────────────────────────────────────────────────────────────────


class TestNoRefinement:
    """NoRefinement 直通策略测试。"""

    def test_no_refinement_passthrough(self) -> None:
        """NoRefinement 返回原始 p_hit/k_hit，log='passthrough'。"""
        env = _build_env()
        limits = _build_limits(env)
        refiner = NoRefinement()
        p_hit = np.array([0.3, -0.5, 1.2])
        arm_state = env.get_arm_state()
        result = refiner.refine(
            p_hit=p_hit, k_hit=100, remaining=150,
            env=env, arm_state=arm_state, robot_limits=limits,
        )
        assert isinstance(result, RefineResult)
        np.testing.assert_allclose(result.p_hit, p_hit)
        assert result.k_hit == 100
        assert result.log == "passthrough"

    def test_no_refinement_reset(self) -> None:
        """NoRefinement.reset() 无状态，不报错。"""
        refiner = NoRefinement()
        refiner.reset()  # 不应抛异常

    def test_no_refinement_is_protocol(self) -> None:
        """NoRefinement 实现 HitPointRefiner Protocol。"""
        refiner = NoRefinement()
        assert isinstance(refiner, HitPointRefiner)


# ──────────────────────────────────────────────────────────────────
# A2 轮 2: HybridRefiner 安全点不变
# ──────────────────────────────────────────────────────────────────


class TestHybridRefinerSafePoint:
    """HybridRefiner 安全点测试 — 裕度充足时不变。"""

    def test_safe_point_feasible(self) -> None:
        """安全点（关节裕度充足）→ log='feasible'，p_hit 不变。"""
        env = _build_env()
        limits = _build_limits(env)
        refiner = HybridRefiner(
            shoulder_pos=np.array([-0.1, -0.22693, 1.302645]),
            workspace_radius=0.90,
        )
        # 用默认 INIT_Q 对应的末端位置作为安全点（裕度充足）
        arm_state = env.get_arm_state()
        env.set_arm_state(arm_state)
        p_hit = env.get_ee_pos().copy()
        result = refiner.refine(
            p_hit=p_hit, k_hit=200, remaining=200,
            env=env, arm_state=arm_state, robot_limits=limits,
        )
        assert result.log == "feasible"
        np.testing.assert_allclose(result.p_hit, p_hit, atol=1e-9)
        assert result.k_hit == 200

    def test_hybrid_lock_when_k_hit_small(self) -> None:
        """k_hit <= 60（防抖锁定阈值）→ log='locked'。"""
        env = _build_env()
        limits = _build_limits(env)
        refiner = HybridRefiner(
            shoulder_pos=np.array([-0.1, -0.22693, 1.302645]),
            workspace_radius=0.90,
        )
        p_hit = np.array([0.3, -0.5, 1.2])
        arm_state = env.get_arm_state()
        result = refiner.refine(
            p_hit=p_hit, k_hit=60, remaining=100,
            env=env, arm_state=arm_state, robot_limits=limits,
        )
        assert result.log == "locked"
        np.testing.assert_allclose(result.p_hit, p_hit)
        assert result.k_hit == 60

    def test_hybrid_is_protocol(self) -> None:
        """HybridRefiner 实现 HitPointRefiner Protocol。"""
        refiner = HybridRefiner(
            shoulder_pos=np.array([-0.1, -0.22693, 1.302645]),
            workspace_radius=0.90,
        )
        assert isinstance(refiner, HitPointRefiner)


# ──────────────────────────────────────────────────────────────────
# A2 轮 3: HybridRefiner reset / hysteresis 状态
# ──────────────────────────────────────────────────────────────────


class TestHybridRefinerState:
    """HybridRefiner hysteresis 状态测试。"""

    def test_reset_clears_lock(self) -> None:
        """reset 清空 hit_lock_active，允许重新 refine。"""
        env = _build_env()
        limits = _build_limits(env)
        refiner = HybridRefiner(
            shoulder_pos=np.array([-0.1, -0.22693, 1.302645]),
            workspace_radius=0.90,
        )
        p_hit = np.array([0.3, -0.5, 1.2])
        arm_state = env.get_arm_state()
        # 首次小 k_hit → 锁定
        r1 = refiner.refine(p_hit, k_hit=50, remaining=100, env=env,
                            arm_state=arm_state, robot_limits=limits)
        assert r1.log == "locked"
        # reset
        refiner.reset()
        # 再次大 k_hit 应进入正常 refine（不再锁定）
        r2 = refiner.refine(p_hit, k_hit=200, remaining=200, env=env,
                            arm_state=arm_state, robot_limits=limits)
        assert r2.log != "locked"
