"""代价函数数值回归测试 — 确保 CompositeCost 在已知输入下产生精确的已知输出。

此测试作为数值守卫：组合化重构（Phase 4）后，任何对代价计算的改动
若改变了已知状态/控制下的数值输出，将立即被此测试捕获。

预期值全部由数学公式手算，独立于实现代码。
"""

import numpy as np
import pytest
from src.ilqt.cost import CompositeCost
from src.ilqt.cost_terms import ControlEffortTerm, SmoothnessTerm, TerminalHitTerm


class _KnownEnv:
    """已知 FK 输出的 env 替身 — 回归测试的数值锚点。

    所有 FK 返回固定常量，确保预期值可手算且与实现无关。
    J_p = eye(3,6) 使得雅可比运算可手算（前三列为单位阵，后三列为零）。
    """

    NQ = 6
    NX = 12
    NU = 6

    def set_arm_state(self, x: np.ndarray) -> None:
        pass

    def get_ee_pos(self) -> np.ndarray:
        return np.array([0.5, -0.3, 1.0])

    def get_ee_vel(self) -> np.ndarray:
        return np.array([0.0, -3.0, 0.0])

    def get_ee_jacp(self) -> np.ndarray:
        return np.eye(3, 6)

    def get_ee_jacr(self) -> np.ndarray:
        return np.zeros((3, 6))

    def get_ee_normal(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])


# ── 测试配置常量（手算预期值的锚点）──
P_HIT = np.array([0.4, -0.2, 0.9])
V_HIT = np.array([0.0, -2.0, 1.0])
Q_P = np.diag([100.0, 100.0, 100.0])
Q_V = np.diag([10.0, 10.0, 10.0])

# FK 输出（来自 _KnownEnv，固定常量）
P_EE = np.array([0.5, -0.3, 1.0])
V_EE = np.array([0.0, -3.0, 0.0])

# 测试状态和控制
#   q    = [0, 0, 0, 0, 0, 0]
#   qdot = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]
X_TEST = np.array([0, 0, 0, 0, 0, 0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
U_TEST = np.array([0.5, -0.3, 0.0, 0.0, 0.0, 0.0])


def _build_composite() -> CompositeCost:
    """构建标准 CompositeCost（ControlEffort + Smoothness + TerminalHit）。"""
    env = _KnownEnv()
    return CompositeCost(
        env,
        running_terms=[
            ControlEffortTerm(R=2.0, NU=6),
            SmoothnessTerm(Q_qdot=3.0, Q_qddot=0.0, Q_du=0.0),
        ],
        terminal_terms=[
            TerminalHitTerm(P_HIT, V_HIT, Q_P, Q_V, Q_n=0.0),
        ],
    )


def test_running_cost_known_value() -> None:
    """运行代价 = ½R|u|² + ½Q_qdot|qdot|²。

    手算：
      ctrl:    ½ · 2.0 · (0.5² + 0.3²) = ½ · 2.0 · 0.34 = 0.34
      smooth:  ½ · 3.0 · (1.0² + 2.0²) = ½ · 3.0 · 5.0  = 7.5
      总计 = 7.84
    """
    composite = _build_composite()
    cost = composite.running_cost(X_TEST, U_TEST, k=0)
    assert cost == pytest.approx(7.84)


def test_terminal_cost_known_value() -> None:
    """终端代价 = ½||p_ee-p_hit||²_Qp + ½||v_ee-v_hit||²_Qv。

    手算：
      dp = p_ee - p_hit = [0.1, -0.1, 0.1]
      dv = v_ee - v_hit = [0.0, -1.0, -1.0]
      位置: ½ · 100 · (0.01 + 0.01 + 0.01) = 1.5
      速度: ½ · 10  · (0 + 1 + 1)          = 10.0
      总计 = 11.5
    """
    composite = _build_composite()
    cost = composite.terminal_cost(X_TEST)
    assert cost == pytest.approx(11.5)


def test_running_derivatives_known_values() -> None:
    """运行导数各字段与手算一致。

    手算：
      l_x:   SmoothnessTerm 贡献 l_x[NQ:] = Q_qdot·qdot = [3, 6, 0, 0, 0, 0]
             其余为零（ControlEffortTerm.l_x = None，被累加器跳过）
      l_u:   ControlEffortTerm 贡献 l_u = R·u = [1.0, -0.6, 0, 0, 0, 0]
      l_uu:  ControlEffortTerm 贡献 l_uu = R·I₆ = 2.0·I₆
      l_xx:  SmoothnessTerm 贡献 l_xx[NQ:, NQ:] = Q_qdot·I₆ = 3.0·I₆
      l_ux:  全零（当前无项贡献交叉项）
    """
    composite = _build_composite()
    l_x, l_u, l_xx, l_ux, l_uu = composite.running_derivatives(
        X_TEST, U_TEST, k=0
    )

    # 形状验证
    assert l_x.shape == (12,)
    assert l_u.shape == (6,)
    assert l_xx.shape == (12, 12)
    assert l_ux.shape == (6, 12)
    assert l_uu.shape == (6, 6)

    # l_x: 仅 SmoothnessTerm 贡献下半块
    expected_l_x = np.zeros(12)
    expected_l_x[6:] = [3.0, 6.0, 0.0, 0.0, 0.0, 0.0]
    assert np.allclose(l_x, expected_l_x)

    # l_u: 仅 ControlEffortTerm 贡献
    expected_l_u = np.array([1.0, -0.6, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(l_u, expected_l_u)

    # l_uu: 仅 ControlEffortTerm 贡献 = 2.0·I₆
    assert np.allclose(l_uu, 2.0 * np.eye(6))

    # l_xx: 仅 SmoothnessTerm 贡献右下 6×6 块 = 3.0·I₆
    expected_l_xx = np.zeros((12, 12))
    expected_l_xx[6:, 6:] = 3.0 * np.eye(6)
    assert np.allclose(l_xx, expected_l_xx)

    # l_ux: 全零
    assert np.allclose(l_ux, 0.0)


def test_terminal_derivatives_known_values() -> None:
    """终端导数各字段与手算一致。

    手算（J_p = eye(3,6)，前三列为单位阵，后三列为零）：
      dp = [0.1, -0.1, 0.1],  Q_p·dp = [10, -10, 10]
      dv = [0.0, -1.0, -1.0], Q_v·dv = [0, -10, -10]

      l_x[:6]  = J_pᵀ·Q_p·dp = [10, -10, 10, 0, 0, 0]
      l_x[6:]  = J_pᵀ·Q_v·dv = [0, -10, -10, 0, 0, 0]

      l_xx[0:6, 0:6]    = J_pᵀ·Q_p·J_p = diag(100, 100, 100, 0, 0, 0)
      l_xx[6:12, 6:12]  = J_pᵀ·Q_v·J_p = diag(10, 10, 10, 0, 0, 0)
    """
    composite = _build_composite()
    l_x, l_xx = composite.terminal_derivatives(X_TEST)

    # 形状验证
    assert l_x.shape == (12,)
    assert l_xx.shape == (12, 12)

    # l_x 手算值
    expected_l_x = np.array(
        [10.0, -10.0, 10.0, 0.0, 0.0, 0.0,
         0.0, -10.0, -10.0, 0.0, 0.0, 0.0]
    )
    assert np.allclose(l_x, expected_l_x)

    # l_xx 手算值（仅对角线非零）
    expected_l_xx = np.zeros((12, 12))
    expected_l_xx[0, 0] = 100.0
    expected_l_xx[1, 1] = 100.0
    expected_l_xx[2, 2] = 100.0
    expected_l_xx[6, 6] = 10.0
    expected_l_xx[7, 7] = 10.0
    expected_l_xx[8, 8] = 10.0
    assert np.allclose(l_xx, expected_l_xx)


def test_terminal_cost_with_normal_vector() -> None:
    """终端法向量代价 = ½Q_n||n_rack - n_des||²。

    手算：
      n_rack = [1, 0, 0], n_des = [0, 1, 0]
      n_err  = [1, -1, 0], |n_err|² = 2.0
      法向量代价 = ½ · 5.0 · 2.0 = 5.0
      总计 = 11.5（位置+速度）+ 5.0（法向量）= 16.5
    """
    env = _KnownEnv()
    composite = CompositeCost(
        env,
        running_terms=[],
        terminal_terms=[
            TerminalHitTerm(
                P_HIT, V_HIT, Q_P, Q_V,
                Q_n=5.0,
                n_des=np.array([0.0, 1.0, 0.0]),
            ),
        ],
    )
    cost = composite.terminal_cost(X_TEST)
    assert cost == pytest.approx(16.5)


def test_running_cost_zero_input() -> None:
    """零状态+零控制下运行代价恰为零（边界守卫）。"""
    composite = _build_composite()
    cost = composite.running_cost(np.zeros(12), np.zeros(6), k=0)
    assert cost == pytest.approx(0.0)


def test_terminal_cost_zero_offset() -> None:
    """末端恰在击打点时终端代价为零（边界守卫）。

    p_hit/v_hit 设为 FK 输出值，dp=dv=0 → 代价=0。
    """
    env = _KnownEnv()
    composite = CompositeCost(
        env,
        running_terms=[],
        terminal_terms=[
            TerminalHitTerm(P_EE, V_EE, Q_P, Q_V, Q_n=0.0),
        ],
    )
    cost = composite.terminal_cost(X_TEST)
    assert cost == pytest.approx(0.0)
