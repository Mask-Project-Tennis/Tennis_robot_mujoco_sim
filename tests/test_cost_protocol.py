"""代价 Protocol 接口测试。"""

import numpy as np
from src.ilqt.components.protocols import (
    RunningCost, TerminalCost, SmoothnessMixin,
    TargetUpdatable, WeightUpdatable, RScheduleUpdatable,  # noqa: F401  验证可导入
    SmoothnessScaleUpdatable,
)


class _FakeRunningCost:
    """满足 RunningCost Protocol 的最小实现。"""
    def running_cost(self, x, u, k=None): return 0.0
    def running_derivatives(self, x, u, k=None):
        return np.zeros(12), np.zeros(6), np.zeros((12,12)), np.zeros((6,12)), np.zeros((6,6))


class _FakeTerminalCost:
    """满足 TerminalCost Protocol 的最小实现。"""
    def terminal_cost(self, x): return 0.0
    def terminal_derivatives(self, x):
        return np.zeros(12), np.zeros((12,12))


class _FakeSmoothness:
    """满足 SmoothnessMixin 的最小实现。"""
    def set_u_prev(self, u_prev): pass


def test_running_cost_protocol_satisfied():
    """满足 running_cost + running_derivatives 的类通过 RunningCost isinstance 检查。"""
    obj = _FakeRunningCost()
    assert isinstance(obj, RunningCost)


def test_terminal_cost_protocol_satisfied():
    """满足 terminal_cost + terminal_derivatives 的类通过 TerminalCost isinstance 检查。"""
    obj = _FakeTerminalCost()
    assert isinstance(obj, TerminalCost)


def test_smoothness_mixin_protocol_satisfied():
    """满足 set_u_prev 的类通过 SmoothnessMixin isinstance 检查。"""
    obj = _FakeSmoothness()
    assert isinstance(obj, SmoothnessMixin)


def test_running_cost_not_terminal():
    """仅实现 RunningCost 的类不满足 TerminalCost。"""
    obj = _FakeRunningCost()
    assert isinstance(obj, RunningCost)
    assert not isinstance(obj, TerminalCost)


class _FakeSmoothnessScale:
    """满足 SmoothnessScaleUpdatable 的最小实现。"""
    def set_smoothness_scale(self, qdot_scale, qddot_scale, du_scale): pass


def test_smoothness_scale_protocol_satisfied():
    """满足 set_smoothness_scale 的类通过 SmoothnessScaleUpdatable isinstance 检查。"""
    assert isinstance(_FakeSmoothnessScale(), SmoothnessScaleUpdatable)


# ── FKContext 测试 ──

from src.ilqt.cost import FKContext, RunningDerivatives, TerminalDerivatives  # noqa: E402


class _MockEnv:
    """不依赖 MuJoCo 的假 env，用于 FKContext 测试。"""
    NQ = 6
    NX = 12
    NU = 6

    def __init__(self):
        self._call_count = 0

    def set_arm_state(self, x):
        self._call_count += 1
        self._x = x

    def get_ee_pos(self):
        return np.array([0.5, -0.3, 1.0])

    def get_ee_vel(self):
        return np.array([0.1, -0.2, 0.0])

    def get_ee_jacp(self):
        return np.eye(3, 6)

    def get_ee_jacr(self):
        return np.zeros((3, 6))

    def get_ee_normal(self):
        return np.array([1.0, 0.0, 0.0])


def test_fkcontext_caches_fk_results():
    """FKContext.update 后属性返回 env 的 FK 结果。"""
    env = _MockEnv()
    fk = FKContext(env)
    x = np.zeros(12)
    fk.update(x)
    assert np.allclose(fk.p_ee, [0.5, -0.3, 1.0])
    assert np.allclose(fk.v_ee, [0.1, -0.2, 0.0])
    assert fk.J_p.shape == (3, 6)
    assert fk.J_r.shape == (3, 6)
    assert np.allclose(fk.n_rack, [1.0, 0.0, 0.0])


def test_fkcontext_skips_redundant_update():
    """相同状态重复 update 不触发 env.set_arm_state。"""
    env = _MockEnv()
    fk = FKContext(env)
    x = np.zeros(12)
    fk.update(x)
    assert env._call_count == 1
    fk.update(x)  # 相同状态
    assert env._call_count == 1  # 不重复调用


def test_fkcontext_different_state_triggers_update():
    """不同状态触发新的 set_arm_state。"""
    env = _MockEnv()
    fk = FKContext(env)
    fk.update(np.zeros(12))
    assert env._call_count == 1
    fk.update(np.ones(12))
    assert env._call_count == 2


def test_fkcontext_j_n_shape():
    """法向量雅可比 J_n 形状为 (3, NX)。"""
    env = _MockEnv()
    fk = FKContext(env)
    fk.update(np.zeros(12))
    assert fk.J_n.shape == (3, 12)


def test_running_derivatives_slots():
    """RunningDerivatives 使用 __slots__，字段默认 None。"""
    d = RunningDerivatives()
    assert d.l_x is None
    assert d.l_u is None
    assert d.l_xx is None
    assert d.l_ux is None
    assert d.l_uu is None


def test_running_derivatives_with_arrays():
    """RunningDerivatives 可持有预分配数组。"""
    nx, nu = 12, 6
    d = RunningDerivatives(
        l_x=np.zeros(nx),
        l_u=np.zeros(nu),
        l_xx=np.zeros((nx, nx)),
    )
    assert d.l_x.shape == (12,)
    assert d.l_u.shape == (6,)
    assert d.l_xx.shape == (12, 12)
    assert d.l_ux is None  # 未提供
    assert d.l_uu is None


def test_terminal_derivatives_slots():
    """TerminalDerivatives 使用 __slots__，字段默认 None。"""
    d = TerminalDerivatives()
    assert d.l_x is None
    assert d.l_xx is None
