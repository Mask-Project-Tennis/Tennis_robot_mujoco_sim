"""雅可比缓存回归测试（P2）。

验证 rm65_env 的雅可比缓存机制：
- 缓存后 get_ee_jacp/jacr/vel/angular_vel 结果与直接 mj_jacSite 一致
- set_arm_state / reset / step 后缓存正确失效
"""

import numpy as np
import mujoco
import pytest
from pathlib import Path

from src.sim.rm65_env import RM65Env


def _make_env() -> RM65Env:
    """创建测试用 RM65Env 实例。"""
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
    return RM65Env(model_path)


@pytest.fixture
def env_with_known_state() -> RM65Env:
    """创建一个设置了已知状态的 env。"""
    env = _make_env()
    env.reset(q0=np.array([0.1, -0.3, 0.5, 0.0, 0.0, 0.0]))
    return env


class TestJacobianCache:
    """雅可比缓存正确性回归测试。"""

    def test_jacp_matches_uncached(self, env_with_known_state: RM65Env) -> None:
        """缓存后 get_ee_jacp() 与直接 mj_jacSite 结果一致。"""
        env = env_with_known_state
        jacp_ref = np.zeros((3, env.model.nv))
        jacr_ref = np.zeros((3, env.model.nv))
        mujoco.mj_jacSite(env.model, env.data, jacp_ref, jacr_ref, env.racket_center_id)
        expected = jacp_ref[:, :env.NQ]
        result = env.get_ee_jacp()
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_ee_vel_uses_cached_jacp(self, env_with_known_state: RM65Env) -> None:
        """get_ee_vel() 使用缓存雅可比，结果与 jacp @ qdot 一致。"""
        env = env_with_known_state
        jacp = env.get_ee_jacp()
        expected = jacp @ env.data.qvel[:env.NQ]
        result = env.get_ee_vel()
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_jacr_matches_uncached(self, env_with_known_state: RM65Env) -> None:
        """缓存后 get_ee_jacr() 与直接 mj_jacSite 结果一致。"""
        env = env_with_known_state
        jacp_ref = np.zeros((3, env.model.nv))
        jacr_ref = np.zeros((3, env.model.nv))
        mujoco.mj_jacSite(env.model, env.data, jacp_ref, jacr_ref, env.racket_center_id)
        expected = jacr_ref[:, :env.NQ]
        result = env.get_ee_jacr()
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_state_change_invalidates_cache(self, env_with_known_state: RM65Env) -> None:
        """set_arm_state(x2) 后雅可比反映新状态。"""
        env = env_with_known_state
        jacp1 = env.get_ee_jacp()
        x2 = np.zeros(12)
        x2[1] = 0.5
        env.set_arm_state(x2)
        jacp2 = env.get_ee_jacp()
        assert not np.allclose(jacp1, jacp2), "状态变化后雅可比应不同"

    def test_reset_invalidates_cache(self, env_with_known_state: RM65Env) -> None:
        """reset() 后缓存失效，雅可比反映重置后状态。"""
        env = env_with_known_state
        # 触发缓存
        jacp_before = env.get_ee_jacp()
        # reset 到不同状态
        env.reset(q0=np.array([0.3, -0.5, 0.8, 0.0, 0.0, 0.0]))
        jacp_after = env.get_ee_jacp()
        # 验证雅可比确实反映了新状态
        jacp_ref = np.zeros((3, env.model.nv))
        jacr_ref = np.zeros((3, env.model.nv))
        mujoco.mj_jacSite(env.model, env.data, jacp_ref, jacr_ref, env.racket_center_id)
        expected = jacp_ref[:, :env.NQ]
        np.testing.assert_allclose(jacp_after, expected, atol=1e-14)
        assert not np.allclose(jacp_before, jacp_after), "reset 后雅可比应不同"

    def test_step_invalidates_cache(self, env_with_known_state: RM65Env) -> None:
        """step() 后缓存失效（发现 #1 安全保障）。"""
        env = env_with_known_state
        env.get_ee_jacp()
        assert env._jacp_cache is not None, "调用后缓存应已填充"
        # step 改变状态
        u = np.array([10.0, -10.0, 5.0, 0.0, 0.0, 0.0])
        for _ in range(20):
            env.step(u)
        assert env._jacp_cache is None, "step 后缓存应已失效"
        # 大力矩多步后雅可比应有可观差异
        jacp1 = np.zeros((3, env.NQ))
        jacp1[:] = env.get_ee_jacp()
        env.reset(q0=np.array([0.1, -0.3, 0.5, 0.0, 0.0, 0.0]))
        jacp2 = env.get_ee_jacp()
        assert not np.allclose(jacp1, jacp2, atol=1e-6), "大力矩多步后雅可比应有可观差异"

    def test_angular_vel_uses_cached_jacr(self, env_with_known_state: RM65Env) -> None:
        """get_ee_angular_vel() 使用缓存 jacr，结果与 jacr @ qdot 一致。"""
        env = env_with_known_state
        jacr = env.get_ee_jacr()
        expected = jacr @ env.data.qvel[:env.NQ]
        result = env.get_ee_angular_vel()
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_shared_cache_across_calls(self, env_with_known_state: RM65Env) -> None:
        """同一状态下连续调用多个 get_ee_* 方法共享同一缓存。"""
        env = env_with_known_state
        # 第一次调用触发缓存
        jacp = env.get_ee_jacp()
        # 后续调用不重新计算
        jacr = env.get_ee_jacr()
        vel = env.get_ee_vel()
        ang_vel = env.get_ee_angular_vel()
        # 验证一致性
        np.testing.assert_allclose(vel, jacp @ env.data.qvel[:env.NQ], atol=1e-14)
        np.testing.assert_allclose(ang_vel, jacr @ env.data.qvel[:env.NQ], atol=1e-14)
