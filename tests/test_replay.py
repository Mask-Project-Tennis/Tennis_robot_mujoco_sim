"""测试 src/sim/replay.py — 轨迹回放碰撞管理模块。

TDD 顺序：Cycle 1-4，每个 Cycle 严格 RED → GREEN → REFACTOR。
"""
import numpy as np
import pytest
import mujoco
from pathlib import Path

from src.sim.replay import (
    should_enable_collision,
    compute_rebound_velocity,
    apply_elastic_rebound,
    replay_trajectory,
    ReplayResult,
)

# ============================================================================
# 辅助 fixture
# ============================================================================

MODEL_PATH = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"


@pytest.fixture
def make_env():
    """创建 RM65Env 实例（延迟导入避免无 MuJoCo 环境时 collect 错误）。"""
    from src.sim.rm65_env import RM65Env
    env = RM65Env(MODEL_PATH)
    return env


# ============================================================================
# Cycle 1: should_enable_collision — 纯函数
# ============================================================================

class TestShouldEnableCollision:
    """测试碰撞窗口判定逻辑。"""

    def test_hit_step_negative_returns_false(self):
        """无击球计划（hit_step=-1）时永不启用碰撞。"""
        assert should_enable_collision(step=0, hit_step=-1, dist=0.1, rebound_applied=False) is False

    def test_rebound_already_applied_returns_false(self):
        """已触发反弹后关闭碰撞。"""
        assert should_enable_collision(step=10, hit_step=20, dist=0.1, rebound_applied=True) is False

    def test_k_hit_le_10_returns_true_regardless_of_dist(self):
        """k_hit_remaining ≤ 10 时启用碰撞，即使球-拍距离远。"""
        # hit_step=50, step=45 → k_hit_remaining=5
        assert should_enable_collision(step=45, hit_step=50, dist=2.0, rebound_applied=False) is True

    def test_k_hit_le_30_and_dist_lt_threshold_returns_true(self):
        """k_hit_remaining ≤ 30 且距离 < 0.35m 时启用碰撞。"""
        # hit_step=50, step=30 → k_hit_remaining=20
        assert should_enable_collision(step=30, hit_step=50, dist=0.20, rebound_applied=False) is True

    def test_k_hit_le_30_but_dist_ge_threshold_returns_false(self):
        """k_hit_remaining ≤ 30 但距离 ≥ 0.35m 时不启用碰撞。"""
        assert should_enable_collision(step=30, hit_step=50, dist=0.50, rebound_applied=False) is False

    def test_k_hit_gt_30_returns_false(self):
        """k_hit_remaining > 30 时不启用碰撞。"""
        # hit_step=50, step=0 → k_hit_remaining=50
        assert should_enable_collision(step=0, hit_step=50, dist=0.10, rebound_applied=False) is False


# ============================================================================
# Cycle 2: compute_rebound_velocity — 弹性反弹纯物理
# ============================================================================


class TestComputeReboundVelocity:
    """测试弹性反弹速度计算（纯物理公式）。"""

    def test_head_on_collision_reverses_normal_component(self):
        """球沿法线方向撞向静止球拍 → 法线分量反转 × e。"""
        n = np.array([0.0, 0.0, 1.0])       # 球拍法线朝 +Z
        v_ball = np.array([0.0, 0.0, -5.0])  # 球向 -Z 运动（朝向球拍）
        v_ee = np.zeros(3)                    # 球拍静止
        # v_rel_n = dot(v_ball - v_ee, n) = -5.0
        # v_new = v_ball - (1+e) * v_rel_n * n = [0,0,-5] - 1.8*(-5)*[0,0,1] = [0,0,4]
        result = compute_rebound_velocity(v_ball, v_ee, n, e=0.8)
        assert np.allclose(result, [0.0, 0.0, 4.0])

    def test_tangential_component_preserved(self):
        """切向速度分量不受反弹影响。"""
        n = np.array([0.0, 0.0, 1.0])
        v_ball = np.array([3.0, 0.0, -5.0])  # 有切向分量
        v_ee = np.zeros(3)
        result = compute_rebound_velocity(v_ball, v_ee, n, e=0.8)
        assert np.isclose(result[0], 3.0)    # 切向不变
        assert np.isclose(result[2], 4.0)    # 法线反弹

    def test_racket_moving_toward_ball_increases_rebound(self):
        """球拍向球运动 → 反弹速度更大。"""
        n = np.array([0.0, 0.0, 1.0])
        v_ball = np.array([0.0, 0.0, -5.0])
        v_ee = np.array([0.0, 0.0, 2.0])     # 球拍向 +Z（迎着球）运动
        # v_rel_n = dot(v_ball - v_ee, n) = dot([0,0,-7], [0,0,1]) = -7
        # v_new = [0,0,-5] - 1.8*(-7)*[0,0,1] = [0,0,-5+12.6] = [0,0,7.6]
        result = compute_rebound_velocity(v_ball, v_ee, n, e=0.8)
        assert np.allclose(result, [0.0, 0.0, 7.6])


# ============================================================================
# Cycle 3: apply_elastic_rebound — 接触检测 + 反弹（环境集成）
# ============================================================================


class TestApplyElasticRebound:
    """测试接触检测 + 反弹速度（需要 MuJoCo 环境）。"""

    def test_returns_rebound_when_ball_at_racket(self, make_env):
        """球放置在球拍位置 + 碰撞启用 → 返回反弹速度（法线分量反转）。"""
        env = make_env
        env.reset(np.zeros(6))
        racket_pos = env.get_ee_pos()
        n_racket = env.get_ee_normal()
        n_hat = n_racket / (np.linalg.norm(n_racket) + 1e-8)
        # 球放在球拍面上，沿法线反方向运动（朝向球拍）
        env.set_ball_state(racket_pos + 0.05 * n_hat, -3.0 * n_hat)
        env.set_arm_collision(False)  # 先禁用以初始化 save 属性
        env.set_arm_collision(True)
        mujoco.mj_forward(env.model, env.data)  # 触发碰撞检测
        ball_vel_pre = (-3.0 * n_hat).copy()
        result = apply_elastic_rebound(env, ball_vel_pre, e=0.8)
        assert result is not None
        # 法线分量应反转（从 -3 变为 +2.4）
        assert np.dot(result, n_hat) > 0

    def test_returns_none_when_ball_far(self, make_env):
        """球远离球拍 → 返回 None。"""
        env = make_env
        env.reset(np.zeros(6))
        env.set_ball_state(np.array([5.0, 0.0, 2.0]), np.zeros(3))  # 初始位置
        env.set_arm_collision(False)
        env.set_arm_collision(True)
        mujoco.mj_forward(env.model, env.data)
        result = apply_elastic_rebound(env, np.zeros(3), e=0.8)
        assert result is None


# ============================================================================
# Cycle 4: replay_trajectory — 完整回放管线集成
# ============================================================================


class TestReplayTrajectory:
    """测试完整回放管线。"""

    def test_replay_applies_rebound_when_ball_at_racket(self, make_env):
        """球在球拍位置 + hit_step 在轨迹中部 → 回放触发反弹。"""
        env = make_env
        init_q = np.zeros(6)
        init_q_left = np.zeros(6)
        env.reset(init_q)
        racket_pos = env.get_ee_pos()
        n_racket = env.get_ee_normal()
        n_hat = n_racket / (np.linalg.norm(n_racket) + 1e-8)
        p0 = racket_pos + 0.05 * n_hat
        v0 = -0.5 * n_hat  # 球缓慢沿法线方向移向球拍
        U_arr = np.zeros((20, env.NU))  # 保持位置
        hit_step = 10
        result = replay_trajectory(env, U_arr, init_q, init_q_left, p0, v0, hit_step)
        assert result.rebound_applied is True
        assert result.contact_step >= 0

    def test_replay_no_rebound_when_hit_step_negative(self, make_env):
        """hit_step=-1（无击球）→ 回放不触发反弹。"""
        env = make_env
        init_q = np.zeros(6)
        U_arr = np.zeros((20, env.NU))
        p0 = np.array([5.0, 0.0, 2.0])
        v0 = np.zeros(3)
        result = replay_trajectory(env, U_arr, init_q, np.zeros(6), p0, v0, hit_step=-1)
        assert result.rebound_applied is False
        assert result.contact_step == -1

    def test_replay_returns_correct_array_lengths(self, make_env):
        """回放结果数组长度 = N+1（含初始状态）。"""
        env = make_env
        N = 15
        U_arr = np.zeros((N, env.NU))
        result = replay_trajectory(
            env, U_arr, np.zeros(6), np.zeros(6),
            np.array([5.0, 0.0, 2.0]), np.zeros(3), hit_step=-1,
        )
        assert len(result.X_replay) == N + 1
        assert len(result.ball_replay) == N + 1
