"""惩罚项独立测试 — 使用 MockFKContext，不依赖 MuJoCo。"""

import numpy as np
import pytest
from src.ilqt.cost_terms import (
    BodyAvoidanceTerm,
    ControlEffortTerm,
    JointLimitTerm,
    QdotLimitTerm,
    SmoothnessTerm,
    TerminalHitTerm,
    TcpSoftTerm,
    XWallTerm,
)


class MockFKContext:
    """测试用 FKContext 替身，纯数据不依赖 MuJoCo。"""

    def __init__(self, p_ee=None, v_ee=None, J_p=None, J_r=None,
                 n_rack=None, J_n=None, NQ=6, NX=12):
        self._p_ee = p_ee if p_ee is not None else np.zeros(3)
        self._v_ee = v_ee if v_ee is not None else np.zeros(3)
        self._J_p = J_p if J_p is not None else np.zeros((3, NQ))
        self._J_r = J_r if J_r is not None else np.zeros((3, NQ))
        self._n_rack = n_rack if n_rack is not None else np.array([1.0, 0.0, 0.0])
        self._J_n = J_n if J_n is not None else np.zeros((3, NX))

    def update(self, x):
        pass  # 测试中不需要真实 FK

    @property
    def p_ee(self):
        return self._p_ee

    @property
    def v_ee(self):
        return self._v_ee

    @property
    def J_p(self):
        return self._J_p

    @property
    def J_r(self):
        return self._J_r

    @property
    def n_rack(self):
        return self._n_rack

    @property
    def J_n(self):
        return self._J_n


def test_control_effort_torque_mode():
    """力矩模式：l = ½ R uᵀu。"""
    term = ControlEffortTerm(R=2.0, NU=6)
    u = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fk = MockFKContext()
    cost = term.running_cost(np.zeros(12), u, k=0, fk=fk)
    assert cost == pytest.approx(1.0)  # 0.5 * 2.0 * 1.0


def test_control_effort_position_mode():
    """位置模式：控制代价为零。"""
    term = ControlEffortTerm(R=2.0, actuator_mode=1, NU=6)
    u = np.ones(6)
    fk = MockFKContext()
    cost = term.running_cost(np.zeros(12), u, k=0, fk=fk)
    assert cost == 0.0


def test_control_effort_derivatives_torque():
    """力矩模式导数：l_u = R·u, l_uu = R。"""
    term = ControlEffortTerm(R=3.0, NU=6)
    u = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    fk = MockFKContext()
    d = term.running_derivatives(np.zeros(12), u, k=0, fk=fk)
    assert np.allclose(d.l_u, 3.0 * u)
    assert np.allclose(d.l_uu, 3.0 * np.eye(6))
    assert d.l_x is None  # 不依赖状态
    assert d.l_xx is None


def test_control_effort_r_schedule():
    """R 调度：时变 R_k 覆盖常数 R。"""
    schedule = np.array([1.0, 2.0, 3.0])
    term = ControlEffortTerm(R=0.0, R_schedule=schedule, NU=6)
    term.set_R_schedule(schedule)
    u = np.ones(6)
    fk = MockFKContext()
    # k=0: R=1.0, cost = 0.5 * 1.0 * 6 = 3.0
    assert term.running_cost(np.zeros(12), u, k=0, fk=fk) == pytest.approx(3.0)
    # k=1: R=2.0, cost = 0.5 * 2.0 * 6 = 6.0
    assert term.running_cost(np.zeros(12), u, k=1, fk=fk) == pytest.approx(6.0)


def test_control_effort_r_joint_scale():
    """关节级缩放：R_mat[j,j] *= scale。"""
    term = ControlEffortTerm(R=1.0, R_joint_scale={0: 0.5}, NU=6)
    u = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fk = MockFKContext()
    # R_mat[0,0] = 1.0 * 0.5 = 0.5, cost = 0.5 * 0.5 * 4 = 1.0
    assert term.running_cost(np.zeros(12), u, k=None, fk=fk) == pytest.approx(1.0)


# ── SmoothnessTerm 测试 ──


def test_smoothness_qdot_cost():
    """Q_qdot 惩罚关节速度幅值。"""
    term = SmoothnessTerm(Q_qdot=2.0, Q_qddot=0.0, Q_du=0.0)
    x = np.zeros(12)
    x[6:] = 1.0  # qdot = [1,1,1,1,1,1]
    fk = MockFKContext()
    cost = term.running_cost(x, np.zeros(6), k=0, fk=fk)
    assert cost == pytest.approx(0.5 * 2.0 * 6.0)  # 0.5*2*|qdot|^2 = 6.0


def test_smoothness_qdot_derivatives():
    """Q_qdot 导数：l_x[nq:] = Q·qdot, l_xx[nq:,nq:] = Q·I。"""
    term = SmoothnessTerm(Q_qdot=3.0, Q_qddot=0.0, Q_du=0.0)
    x = np.zeros(12)
    x[6:] = 2.0
    fk = MockFKContext()
    d = term.running_derivatives(x, np.zeros(6), k=0, fk=fk)
    assert np.allclose(d.l_x[6:], 3.0 * 2.0)
    assert np.allclose(d.l_x[:6], 0.0)  # 位置部分为零
    assert np.allclose(d.l_xx[6:, 6:], 3.0 * np.eye(6))


def test_smoothness_du_requires_u_prev():
    """Q_du 在 set_u_prev 前不产生代价。"""
    term = SmoothnessTerm(Q_qdot=0.0, Q_qddot=0.0, Q_du=5.0)
    u = np.ones(6)
    fk = MockFKContext()
    # 未设置 u_prev
    assert term.running_cost(np.zeros(12), u, k=0, fk=fk) == 0.0
    # 设置后产生代价
    term.set_u_prev(np.zeros(6))
    cost = term.running_cost(np.zeros(12), u, k=0, fk=fk)
    assert cost == pytest.approx(0.5 * 5.0 * 6.0)


def test_smoothness_scale():
    """set_smoothness_scale 动态调整权重。"""
    term = SmoothnessTerm(Q_qdot=2.0, Q_qddot=0.0, Q_du=0.0)
    x = np.zeros(12)
    x[6:] = 1.0
    fk = MockFKContext()
    cost_base = term.running_cost(x, np.zeros(6), k=0, fk=fk)
    term.set_smoothness_scale(qdot_scale=0.5, qddot_scale=1.0, du_scale=1.0)
    cost_scaled = term.running_cost(x, np.zeros(6), k=0, fk=fk)
    assert cost_scaled == pytest.approx(cost_base * 0.5)


def test_smoothness_none_semantics():
    """None 语义：Q_qdot=Q_qddot=0 时 l_x=None；Q_du=0 时 l_u=None。

    对齐 ControlEffortTerm 与 RunningDerivatives None 约定——
    此项数学上不依赖该变量时返回 None（非零数组），便于 CompositeCost 跳过聚合。
    """
    # 全零权重：无状态依赖、无控制依赖
    term_zero = SmoothnessTerm(Q_qdot=0.0, Q_qddot=0.0, Q_du=0.0)
    fk = MockFKContext()
    d_zero = term_zero.running_derivatives(np.zeros(12), np.zeros(6), k=0, fk=fk)
    assert d_zero.l_x is None
    assert d_zero.l_xx is None
    assert d_zero.l_u is None
    assert d_zero.l_uu is None

    # 仅 qdot：有状态依赖、无控制依赖
    term_qdot = SmoothnessTerm(Q_qdot=1.0, Q_qddot=0.0, Q_du=0.0)
    d_qdot = term_qdot.running_derivatives(np.zeros(12), np.zeros(6), k=0, fk=MockFKContext())
    assert d_qdot.l_x is not None
    assert d_qdot.l_u is None

    # 仅 du：无状态依赖、有控制依赖
    term_du = SmoothnessTerm(Q_qdot=0.0, Q_qddot=0.0, Q_du=1.0)
    term_du.set_u_prev(np.zeros(6))
    d_du = term_du.running_derivatives(np.zeros(12), np.zeros(6), k=0, fk=MockFKContext())
    assert d_du.l_x is None
    assert d_du.l_u is not None


def test_smoothness_qddot_cost():
    """Q_qddot 用 qdot/dt 近似加速度：cost = 0.5*Q_qddot*|qdot/dt|^2。"""
    dt = 0.005
    term = SmoothnessTerm(Q_qdot=0.0, Q_qddot=4.0, Q_du=0.0, dt=dt)
    x = np.zeros(12)
    x[6:] = 0.01  # qdot = 0.01
    fk = MockFKContext()
    cost = term.running_cost(x, np.zeros(6), k=0, fk=fk)
    qddot = 0.01 / dt
    assert cost == pytest.approx(0.5 * 4.0 * 6.0 * qddot * qddot)


# ── QdotLimitTerm 测试 ──


def test_qdot_limit_below_threshold():
    """关节速度低于阈值时代价为 0。"""
    thresholds = np.full(6, 2.0)
    term = QdotLimitTerm(Q_qdot_limit=100.0, qdot_limit_thresholds=thresholds)
    x = np.zeros(12)
    x[6:] = 1.0  # 全部低于 2.0
    fk = MockFKContext()
    assert term.running_cost(x, np.zeros(6), k=0, fk=fk) == 0.0


def test_qdot_limit_above_threshold():
    """关节速度超过阈值时产生 hinge 代价。"""
    thresholds = np.full(6, 1.0)
    term = QdotLimitTerm(Q_qdot_limit=100.0, qdot_limit_thresholds=thresholds)
    x = np.zeros(12)
    x[6:] = 2.0  # excess = 1.0 per joint
    fk = MockFKContext()
    cost = term.running_cost(x, np.zeros(6), k=0, fk=fk)
    # 6 关节 × 0.5 × 100 × 1.0^2 = 300.0
    assert cost == pytest.approx(300.0)


def test_qdot_limit_disabled():
    """Q=0 时禁用：代价恒为 0，导数恒为零。"""
    thresholds = np.full(6, 1.0)
    term = QdotLimitTerm(Q_qdot_limit=0.0, qdot_limit_thresholds=thresholds)
    x = np.zeros(12)
    x[6:] = 5.0  # 远超阈值
    fk = MockFKContext()
    assert term.running_cost(x, np.zeros(6), k=0, fk=fk) == 0.0
    d = term.running_derivatives(x, np.zeros(6), k=0, fk=fk)
    assert np.allclose(d.l_x, 0.0)
    assert np.allclose(d.l_xx, 0.0)


def test_qdot_limit_derivatives():
    """导数：l_x[nq+j] = Q·excess·sign(qdot_j), l_xx 对角 = Q。"""
    thresholds = np.full(6, 1.0)
    term = QdotLimitTerm(Q_qdot_limit=100.0, qdot_limit_thresholds=thresholds)
    x = np.zeros(12)
    x[6:] = 2.0  # excess = 1.0 per joint
    fk = MockFKContext()
    d = term.running_derivatives(x, np.zeros(6), k=0, fk=fk)
    # 全部超限：l_x[6:] = 100 * 1.0 * sign(2.0) = 100
    assert np.allclose(d.l_x[6:], 100.0)
    assert np.allclose(d.l_x[:6], 0.0)  # 位置部分为零
    assert np.allclose(d.l_xx[6:, 6:], 100.0 * np.eye(6))
    assert d.l_u is None  # 不依赖控制
    assert d.l_uu is None


def test_qdot_limit_mixed():
    """部分超限部分未超限：仅超限关节贡献代价。"""
    thresholds = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    term = QdotLimitTerm(Q_qdot_limit=100.0, qdot_limit_thresholds=thresholds)
    x = np.zeros(12)
    x[6:] = [0.5, 2.0, -3.0, 0.0, 1.5, -0.5]  # 关节 1,2,4 超限
    fk = MockFKContext()
    cost = term.running_cost(x, np.zeros(6), k=0, fk=fk)
    # excess: [0, 1.0, 2.0, 0, 0.5, 0] → 0.5*100*(1+4+0.25) = 262.5
    expected = 0.5 * 100.0 * (1.0**2 + 2.0**2 + 0.5**2)
    assert cost == pytest.approx(expected)
    d = term.running_derivatives(x, np.zeros(6), k=0, fk=fk)
    # l_x[7]=100*1*1=100, l_x[8]=100*2*(-1)=-200, l_x[10]=100*0.5*1=50
    assert np.isclose(d.l_x[7], 100.0)
    assert np.isclose(d.l_x[8], -200.0)
    assert np.isclose(d.l_x[10], 50.0)
    assert np.isclose(d.l_x[6], 0.0)  # 未超限
    assert np.isclose(d.l_x[9], 0.0)  # 未超限
    assert np.isclose(d.l_x[11], 0.0)  # 未超限


# ── TcpSoftTerm 测试 ──


def test_tcp_soft_below_threshold():
    """TCP 速度低于阈值时代价为 0。"""
    # J_p = identity → tcp_vel = qdot, tcp_speed = |qdot|
    J_p = np.eye(3, 6)
    fk = MockFKContext(J_p=J_p)
    term = TcpSoftTerm(Q_tcp_soft=5000.0, tcp_threshold=10.0)
    x = np.zeros(12)
    x[6:] = 1.0  # tcp_speed = sqrt(3) ≈ 1.73 < 10
    assert term.running_cost(x, np.zeros(6), k=0, fk=fk) == 0.0


def test_tcp_soft_above_threshold():
    """TCP 速度超过阈值时产生二次惩罚。"""
    J_p = np.eye(3, 6)
    fk = MockFKContext(J_p=J_p)
    term = TcpSoftTerm(Q_tcp_soft=5000.0, tcp_threshold=1.0)
    x = np.zeros(12)
    x[6:] = 1.0  # tcp_speed = sqrt(3) ≈ 1.732
    cost = term.running_cost(x, np.zeros(6), k=0, fk=fk)
    excess = np.sqrt(3) - 1.0
    expected = 0.5 * 5000.0 * excess ** 2
    assert cost == pytest.approx(expected, rel=1e-4)


def test_tcp_soft_disabled():
    """Q=0 时禁用：代价恒为 0。"""
    J_p = np.eye(3, 6)
    fk = MockFKContext(J_p=J_p)
    term = TcpSoftTerm(Q_tcp_soft=0.0, tcp_threshold=1.0)
    x = np.zeros(12)
    x[6:] = 10.0
    assert term.running_cost(x, np.zeros(6), k=0, fk=fk) == 0.0
    d = term.running_derivatives(x, np.zeros(6), k=0, fk=fk)
    assert np.allclose(d.l_x, 0.0)
    assert d.l_u is None


def test_tcp_soft_derivatives_finite_difference():
    """导数与有限差分一致（Gauss-Newton 近似）。"""
    J_p = np.eye(3, 6)
    fk = MockFKContext(J_p=J_p)
    term = TcpSoftTerm(Q_tcp_soft=5000.0, tcp_threshold=0.5)
    x = np.zeros(12)
    x[6:] = [1.0, 0.5, -0.3, 0.0, 0.0, 0.0]
    d = term.running_derivatives(x, np.zeros(6), k=0, fk=fk)
    # 有限差分验证 l_x（仅 qdot 部分）
    eps = 1e-6
    grad_fd = np.zeros(12)
    for i in range(6):
        xp = x.copy()
        xp[6 + i] += eps
        xm = x.copy()
        xm[6 + i] -= eps
        cp = term.running_cost(xp, np.zeros(6), k=0, fk=fk)
        cm = term.running_cost(xm, np.zeros(6), k=0, fk=fk)
        grad_fd[6 + i] = (cp - cm) / (2 * eps)
    assert np.allclose(d.l_x, grad_fd, atol=1e-3)


# ── TerminalHitTerm 测试 ──


def test_terminal_hit_cost_zero_at_target():
    """末端恰好在击打点时代价为 0。"""
    p_hit = np.array([0.5, -0.3, 1.0])
    v_hit = np.array([0.0, -3.0, 0.0])
    Q_p = np.diag([100.0, 100.0, 100.0])
    Q_v = np.diag([10.0, 10.0, 10.0])
    term = TerminalHitTerm(p_hit, v_hit, Q_p, Q_v, Q_n=0.0)
    fk = MockFKContext(p_ee=p_hit, v_ee=v_hit)
    assert term.terminal_cost(np.zeros(12), fk) == pytest.approx(0.0)


def test_terminal_hit_cost_position_error():
    """末端偏离击打点产生位置代价。"""
    p_hit = np.array([0.5, -0.3, 1.0])
    v_hit = np.array([0.0, 0.0, 0.0])
    Q_p = np.diag([100.0, 100.0, 100.0])
    Q_v = np.diag([0.0, 0.0, 0.0])  # 禁用速度代价
    term = TerminalHitTerm(p_hit, v_hit, Q_p, Q_v, Q_n=0.0)
    p_ee = p_hit + np.array([0.1, 0.0, 0.0])  # 偏移 0.1m
    fk = MockFKContext(p_ee=p_ee, v_ee=v_hit)
    cost = term.terminal_cost(np.zeros(12), fk)
    expected = 0.5 * 100.0 * 0.1 ** 2
    assert cost == pytest.approx(expected)


def test_terminal_hit_cost_velocity_error():
    """末端速度偏离期望速度产生速度代价。"""
    p_hit = np.array([0.0, 0.0, 0.0])
    v_hit = np.array([0.0, -3.0, 0.0])
    Q_p = np.diag([0.0, 0.0, 0.0])
    Q_v = np.diag([10.0, 10.0, 10.0])
    term = TerminalHitTerm(p_hit, v_hit, Q_p, Q_v, Q_n=0.0)
    v_ee = np.array([0.0, 0.0, 0.0])  # 速度偏差 3.0
    fk = MockFKContext(p_ee=p_hit, v_ee=v_ee)
    cost = term.terminal_cost(np.zeros(12), fk)
    expected = 0.5 * 10.0 * 3.0 ** 2
    assert cost == pytest.approx(expected)


def test_terminal_hit_derivatives_shape():
    """终端导数形状正确。"""
    p_hit = np.array([0.5, -0.3, 1.0])
    v_hit = np.array([0.0, -3.0, 0.0])
    Q_p = np.eye(3) * 100.0
    Q_v = np.eye(3) * 10.0
    J_p = np.eye(3, 6)
    term = TerminalHitTerm(p_hit, v_hit, Q_p, Q_v, Q_n=0.0)
    fk = MockFKContext(p_ee=np.zeros(3), v_ee=np.zeros(3), J_p=J_p)
    d = term.terminal_derivatives(np.zeros(12), fk)
    assert d.l_x.shape == (12,)
    assert d.l_xx.shape == (12, 12)


def test_terminal_hit_update_target():
    """update_target 更新击打目标。"""
    term = TerminalHitTerm(
        np.zeros(3), np.zeros(3), np.eye(3), np.eye(3), Q_n=0.0
    )
    new_p = np.array([1.0, 0.0, 0.0])
    new_v = np.array([0.0, 1.0, 0.0])
    term.update_target(new_p, new_v)
    fk = MockFKContext(p_ee=new_p, v_ee=new_v)
    assert term.terminal_cost(np.zeros(12), fk) == pytest.approx(0.0)


def test_terminal_hit_update_weights():
    """update_weights 缩放 Q_p/Q_v。"""
    p_hit = np.array([1.0, 0.0, 0.0])
    term = TerminalHitTerm(p_hit, np.zeros(3), np.eye(3), np.eye(3), Q_n=0.0)
    fk = MockFKContext(p_ee=np.zeros(3), v_ee=np.zeros(3))
    cost_base = term.terminal_cost(np.zeros(12), fk)
    term.update_weights(Q_p_scale=2.0, Q_v_scale=1.0)
    cost_scaled = term.terminal_cost(np.zeros(12), fk)
    # Q_p 翻倍 → 位置代价翻倍（速度代价不变，但 v_ee=v_hit=0 所以速度代价=0）
    assert cost_scaled == pytest.approx(cost_base * 2.0)


# ── JointLimitTerm 测试 ──


def test_joint_limit_within_bounds():
    """关节角度在限位内时代价为 0。"""
    term = JointLimitTerm({0: (-1.0, 1.0)}, Q_joint_limit=100.0)
    x = np.zeros(12)
    x[0] = 0.5  # 在 (-1, 1) 内
    fk = MockFKContext()
    assert term.running_cost(x, np.zeros(6), k=0, fk=fk) == 0.0


def test_joint_limit_above_upper():
    """关节角度超上界产生代价。"""
    term = JointLimitTerm({0: (-1.0, 1.0)}, Q_joint_limit=100.0)
    x = np.zeros(12)
    x[0] = 1.5  # excess = 0.5
    fk = MockFKContext()
    cost = term.running_cost(x, np.zeros(6), k=0, fk=fk)
    assert cost == pytest.approx(0.5 * 100.0 * 0.5 ** 2)


def test_joint_limit_below_lower():
    """关节角度低于下界产生代价（对称于上界）。"""
    term = JointLimitTerm({0: (-1.0, 1.0)}, Q_joint_limit=100.0)
    x = np.zeros(12)
    x[0] = -1.5  # excess = 0.5
    fk = MockFKContext()
    cost = term.running_cost(x, np.zeros(6), k=0, fk=fk)
    assert cost == pytest.approx(0.5 * 100.0 * 0.5 ** 2)


def test_joint_limit_derivatives_above():
    """超上界导数：l_x[j] = Q·margin, l_xx[j,j] = Q。"""
    term = JointLimitTerm({0: (-1.0, 1.0)}, Q_joint_limit=100.0)
    x = np.zeros(12)
    x[0] = 1.5  # margin = 0.5
    fk = MockFKContext()
    d = term.running_derivatives(x, np.zeros(6), k=0, fk=fk)
    assert np.isclose(d.l_x[0], 100.0 * 0.5)
    assert np.isclose(d.l_xx[0, 0], 100.0)
    assert d.l_u is None  # 不依赖控制
    assert d.l_uu is None


def test_joint_limit_derivatives_below():
    """超下界导数：l_x[j] = -Q·margin, l_xx[j,j] = Q。"""
    term = JointLimitTerm({0: (-1.0, 1.0)}, Q_joint_limit=100.0)
    x = np.zeros(12)
    x[0] = -1.5  # margin = 0.5
    fk = MockFKContext()
    d = term.running_derivatives(x, np.zeros(6), k=0, fk=fk)
    assert np.isclose(d.l_x[0], -100.0 * 0.5)
    assert np.isclose(d.l_xx[0, 0], 100.0)


def test_joint_limit_mixed_multi_joint():
    """多关节混合：部分超上界、部分超下界、部分在范围内。"""
    term = JointLimitTerm(
        {0: (-1.0, 1.0), 1: (-1.0, 1.0), 2: (-1.0, 1.0)},
        Q_joint_limit=100.0,
    )
    x = np.zeros(12)
    x[0] = 1.5  # 超上界 margin=0.5
    x[1] = -2.0  # 超下界 margin=1.0
    x[2] = 0.5  # 在范围内
    fk = MockFKContext()
    cost = term.running_cost(x, np.zeros(6), k=0, fk=fk)
    # 0.5*100*(0.5^2 + 1.0^2) = 0.5*100*1.25 = 62.5
    assert cost == pytest.approx(62.5)
    d = term.running_derivatives(x, np.zeros(6), k=0, fk=fk)
    assert np.isclose(d.l_x[0], 100.0 * 0.5)  # 超上界 +
    assert np.isclose(d.l_x[1], -100.0 * 1.0)  # 超下界 -
    assert np.isclose(d.l_x[2], 0.0)  # 在范围内
    assert np.isclose(d.l_xx[0, 0], 100.0)
    assert np.isclose(d.l_xx[1, 1], 100.0)
    assert np.isclose(d.l_xx[2, 2], 0.0)


def test_joint_limit_none_bound():
    """某侧界为 None 时不惩罚该侧。"""
    term = JointLimitTerm({0: (None, 1.0)}, Q_joint_limit=100.0)
    x = np.zeros(12)
    x[0] = -10.0  # 无下界，不惩罚
    fk = MockFKContext()
    assert term.running_cost(x, np.zeros(6), k=0, fk=fk) == 0.0
    d = term.running_derivatives(x, np.zeros(6), k=0, fk=fk)
    assert np.allclose(d.l_x, 0.0)


def test_joint_limit_finite_difference():
    """导数与有限差分一致（数值梯度验证）。"""
    term = JointLimitTerm({0: (-1.0, 1.0), 1: (-0.5, 0.5)}, Q_joint_limit=1000.0)
    x = np.zeros(12)
    x[0] = 1.3  # 超上界
    x[1] = -0.8  # 超下界
    fk = MockFKContext()
    d = term.running_derivatives(x, np.zeros(6), k=0, fk=fk)
    eps = 1e-6
    grad_fd = np.zeros(12)
    for i in range(2):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        cp = term.running_cost(xp, np.zeros(6), k=0, fk=fk)
        cm = term.running_cost(xm, np.zeros(6), k=0, fk=fk)
        grad_fd[i] = (cp - cm) / (2 * eps)
    assert np.allclose(d.l_x, grad_fd, atol=1e-3)


def test_joint_limit_flyweight_persistent_ref():
    """Flyweight：running_derivatives 返回持久引用（多次调用同一对象）。"""
    term = JointLimitTerm({0: (-1.0, 1.0)}, Q_joint_limit=100.0)
    fk = MockFKContext()
    x1 = np.zeros(12)
    x1[0] = 1.5
    d1 = term.running_derivatives(x1, np.zeros(6), k=0, fk=fk)
    x2 = np.zeros(12)
    x2[0] = 0.0  # 在范围内
    d2 = term.running_derivatives(x2, np.zeros(6), k=0, fk=fk)
    assert d1 is d2  # 同一对象
    # 第二次调用后应清零（在范围内无贡献）
    assert np.isclose(d2.l_x[0], 0.0)


def test_body_avoidance_stub_raises():
    """BodyAvoidanceTerm 构造时抛 NotImplementedError（Phase 4 占位）。"""
    with pytest.raises(NotImplementedError):
        BodyAvoidanceTerm(
            center_xy=np.zeros(2),
            radius=0.2,
            Q_body=1.0,
            body_names=["r_link3"],
        )


def test_xwall_stub_raises():
    """XWallTerm 构造时抛 NotImplementedError（Phase 4 占位）。"""
    with pytest.raises(NotImplementedError):
        XWallTerm(limit_x=-0.1, Q_x=1.0, body_names=["r_link3"])
