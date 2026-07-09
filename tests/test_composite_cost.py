"""CompositeCost 集成测试。

验证 CompositeCost 作为组合代价聚合器：
1. 运行/终端代价 = 各项之和
2. 导数形状正确 + None 贡献跳过
3. MPC 动态更新委托（update_target / set_u_prev / update_weights 等）
4. R1 修复：6 个 @property 透传终端项属性（Tube wrapper softmin 兼容）
5. R3b 修复：活跃代码路径额外委托（set_smoothness_scale 等）
6. Protocol 满足性
"""

import numpy as np
import pytest
from src.ilqt.cost import CompositeCost
from src.ilqt.cost_terms import ControlEffortTerm, SmoothnessTerm, TerminalHitTerm


class _MockEnv:
    """最小 env 替身 — 满足 FKContext 所需接口。"""

    NQ = 6
    NX = 12
    NU = 6

    def set_arm_state(self, x):
        pass

    def get_ee_pos(self):
        return np.array([0.5, -0.3, 1.0])

    def get_ee_vel(self):
        return np.array([0.0, -3.0, 0.0])

    def get_ee_jacp(self):
        return np.eye(3, 6)

    def get_ee_jacr(self):
        return np.zeros((3, 6))

    def get_ee_normal(self):
        return np.array([1.0, 0.0, 0.0])


def test_composite_cost_running_sum():
    """CompositeCost.running_cost = 各运行项之和。"""
    env = _MockEnv()
    ctrl = ControlEffortTerm(R=2.0, NU=6)
    smooth = SmoothnessTerm(Q_qdot=1.0, Q_qddot=0.0, Q_du=0.0)
    composite = CompositeCost(env, running_terms=[ctrl, smooth], terminal_terms=[])
    x = np.zeros(12)
    x[6:] = 1.0
    u = np.ones(6)
    cost = composite.running_cost(x, u, k=0)
    # ctrl: 0.5 * 2.0 * 6 = 6.0
    # smooth: 0.5 * 1.0 * 6 = 3.0
    assert cost == pytest.approx(9.0)


def test_composite_cost_terminal():
    """CompositeCost.terminal_cost = 各终端项之和。"""
    env = _MockEnv()
    p_hit = np.array([0.5, -0.3, 1.0])
    v_hit = np.array([0.0, -3.0, 0.0])
    term = TerminalHitTerm(p_hit, v_hit, np.eye(3) * 100, np.eye(3) * 10, Q_n=0.0)
    composite = CompositeCost(env, running_terms=[], terminal_terms=[term])
    # 末端恰好在击打点
    cost = composite.terminal_cost(np.zeros(12))
    assert cost == pytest.approx(0.0)


def test_composite_cost_derivatives_shape():
    """CompositeCost.running_derivatives 返回正确形状的 5 元组。"""
    env = _MockEnv()
    ctrl = ControlEffortTerm(R=1.0, NU=6)
    composite = CompositeCost(env, running_terms=[ctrl], terminal_terms=[])
    l_x, l_u, l_xx, l_ux, l_uu = composite.running_derivatives(
        np.zeros(12), np.ones(6), k=0
    )
    assert l_x.shape == (12,)
    assert l_u.shape == (6,)
    assert l_xx.shape == (12, 12)
    assert l_ux.shape == (6, 12)
    assert l_uu.shape == (6, 6)


def test_composite_cost_delegates_update_target():
    """CompositeCost.update_target 委托给实现了 TargetUpdatable 的项。"""
    env = _MockEnv()
    term = TerminalHitTerm(
        np.zeros(3), np.zeros(3), np.eye(3), np.eye(3), Q_n=0.0
    )
    composite = CompositeCost(env, running_terms=[], terminal_terms=[term])
    new_p = np.array([1.0, 0.0, 0.0])
    new_v = np.array([0.0, 1.0, 0.0])
    composite.update_target(new_p, new_v)
    assert np.allclose(term.p_hit, new_p)
    assert np.allclose(term.v_hit, new_v)


def test_composite_cost_delegates_set_u_prev():
    """CompositeCost.set_u_prev 委托给 SmoothnessMixin 项。"""
    env = _MockEnv()
    smooth = SmoothnessTerm(Q_qdot=0.0, Q_qddot=0.0, Q_du=1.0)
    composite = CompositeCost(env, running_terms=[smooth], terminal_terms=[])
    u_prev = np.ones(6)
    composite.set_u_prev(u_prev)
    assert smooth._u_prev is not None
    assert np.allclose(smooth._u_prev, u_prev)


def test_composite_cost_protocol_satisfied():
    """CompositeCost 满足 RunningCost + TerminalCost Protocol。"""
    from src.ilqt.components.protocols import RunningCost, TerminalCost
    env = _MockEnv()
    composite = CompositeCost(env, running_terms=[], terminal_terms=[])
    assert isinstance(composite, RunningCost)
    assert isinstance(composite, TerminalCost)


# ── R1 修复测试：Tube wrapper softmin 兼容属性 ──

def test_composite_cost_exposes_terminal_attributes():
    """CompositeCost 通过 @property 暴露终端项属性（Tube wrapper 兼容）。"""
    env = _MockEnv()
    p_hit = np.array([0.5, -0.3, 1.0])
    v_hit = np.array([0.0, -3.0, 0.0])
    Q_p = np.eye(3) * 100.0
    Q_v = np.eye(3) * 10.0
    term = TerminalHitTerm(p_hit, v_hit, Q_p, Q_v, Q_n=5.0, n_des=np.array([0, 0, 1.0]))
    composite = CompositeCost(env, running_terms=[], terminal_terms=[term])
    # 全部属性可读且值正确
    assert np.allclose(composite.p_hit, p_hit)
    assert np.allclose(composite.v_hit, v_hit)
    assert np.allclose(composite.Q_p, Q_p)
    assert np.allclose(composite.Q_v, Q_v)
    assert composite.Q_n == 5.0
    assert np.allclose(composite.n_des, [0, 0, 1.0])


def test_composite_cost_attributes_reflect_weight_update():
    """update_weights 后 @property 返回更新后的 Q_p/Q_v。"""
    env = _MockEnv()
    Q_p = np.eye(3) * 100.0
    term = TerminalHitTerm(np.zeros(3), np.zeros(3), Q_p, np.eye(3), Q_n=0.0)
    composite = CompositeCost(env, running_terms=[], terminal_terms=[term])
    assert np.allclose(composite.Q_p, Q_p)
    composite.update_weights(Q_p_scale=2.0, Q_v_scale=1.0)
    assert np.allclose(composite.Q_p, Q_p * 2.0)


# ── R3b 修复测试：额外委托方法 ──

def test_composite_cost_delegates_set_smoothness_scale():
    """CompositeCost.set_smoothness_scale 委托给 SmoothnessTerm。

    注意：SmoothnessTerm 的有效权重属性名为 _Q_qdot_eff / _Q_qddot_eff / _Q_du_eff
    （见 cost_terms.py:143-145），非 _effective 后缀。
    """
    env = _MockEnv()
    smooth = SmoothnessTerm(Q_qdot=1.0, Q_qddot=1.0, Q_du=1.0)
    composite = CompositeCost(env, running_terms=[smooth], terminal_terms=[])
    composite.set_smoothness_scale(2.0, 3.0, 0.5)
    assert smooth._Q_qdot_eff == pytest.approx(2.0)
    assert smooth._Q_qddot_eff == pytest.approx(3.0)
    assert smooth._Q_du_eff == pytest.approx(0.5)


def test_composite_cost_no_op_when_term_absent():
    """委托方法在无对应 term 时静默跳过（不崩溃）。"""
    env = _MockEnv()
    composite = CompositeCost(env, running_terms=[], terminal_terms=[])
    # 全部不崩溃
    composite.set_q_des_traj(None, None)
    composite.set_smoothness_scale(1.0, 1.0, 1.0)
    composite.set_midpoint_target(None, None)
