"""C++ 后向传递等效性测试（选项 C）。

验证 C++ backward_pass 与 Python numpy Riccati 递推数值等价。
所有输入通过真实 solver pipeline 生成（_linearize + _running_cost_derivatives），
确保测试覆盖实际运行时的矩阵尺度与数值范围。
"""

import numpy as np
import pytest
from pathlib import Path

from src.sim.rm65_env import RM65Env
from src.ilqt.cost import CompositeCost
from src.ilqt.cost_terms import ControlEffortTerm, SmoothnessTerm, TerminalHitTerm
from src.cpp.solver_cpp import ILQTSolver, _backward_pass_numpy
from src.cpp.iLQR_Core import backward_pass as cpp_backward_pass


def _make_env() -> RM65Env:
    """创建测试用 RM65Env（力矩模式）。"""
    return RM65Env(Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml")


def _make_solver(N: int = 60) -> ILQTSolver:
    """创建求解器实例。"""
    cfg = {
        "max_iter": 300, "tol": 1e-4, "horizon": N,
        "mu_min": 1e-6, "mu_max": 1e10, "mu_init": 0.01,
        "delta_0": 1.6, "alpha_list": [1.0, 0.5, 0.25, 0.1, 0.05, 0.01],
        "lin_eps": 1e-6,
    }
    return ILQTSolver(cfg, use_analytical=True)


def _make_cost(env: RM65Env, N: int) -> CompositeCost:
    """创建 V11 默认风格代价函数（CompositeCost 组装）。

    关节跟踪（q_des_traj + Q_joint）当前无对应具体项类（G5 修复后
    JointTrackUpdatable Protocol 已删除），省略该惩罚（不影响 C++ vs
    Python 后向传递数值等价验证）。
    """
    return CompositeCost(
        env,
        running_terms=[
            ControlEffortTerm(
                R=0.0001,
                R_schedule=0.0001 * (0.4 ** (np.arange(N) / N)),
                actuator_mode=0,
                NU=env.NU,
            ),
            SmoothnessTerm(
                Q_qdot=0.001, Q_qddot=0.0005, Q_du=0.001,
                NQ=env.NQ, NX=env.NX, NU=env.NU, dt=env.dt,
            ),
        ],
        terminal_terms=[
            TerminalHitTerm(
                np.array([0.5, -0.5, 1.2]), np.array([0.0, -3.0, 1.0]),
                np.array([50000.0] * 3), np.array([200.0] * 3),
                NX=env.NX, NQ=env.NQ,
            ),
        ],
    )


def _build_backward_inputs(N: int, mu: float = 0.01):
    """通过真实 solver pipeline 构建后向传递输入。

    Returns:
        (solver, As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu)
    """
    env = _make_env()
    solver = _make_solver(N)
    cost_fn = _make_cost(env, N)
    init_q = np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0])
    x0 = np.concatenate([init_q, np.zeros(6)])
    rng = np.random.default_rng(123)
    U = rng.standard_normal((N, 6)) * 3.0
    X = np.zeros((N + 1, 12))
    X[0] = x0
    for k in range(N):
        X[k + 1] = env.step_from_state(X[k], U[k])
    As, Bs, _ = solver._linearize_fast(env, X, U)
    l_xs, l_us, l_xxs, l_uxs, l_uus = solver._running_cost_derivatives(cost_fn, X[:N], U)
    l_x_N, l_xx_N = cost_fn.terminal_derivatives(X[-1])
    return solver, As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu


def _call_cpp_backward(As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu):
    """调用 C++ backward_pass，输入 list→连续数组。"""
    ok, Ks_arr, ks_arr = cpp_backward_pass(
        np.stack(As), np.stack(Bs),
        np.stack(l_xs), np.stack(l_us),
        np.stack(l_xxs), np.stack(l_uxs), np.stack(l_uus),
        l_x_N, l_xx_N, mu,
    )
    return ok, Ks_arr, ks_arr


# ============================================================================
# Slice 1 (tracer): N=5 数值等价
# ============================================================================

class TestBackwardEquivalence:
    """C++ backward_pass 与 Python numpy 参考数值等价。"""

    def test_n5_matches_python(self):
        """N=5 轨迹: C++ Ks/ks 与 Python numpy atol=1e-10。"""
        _, As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu = \
            _build_backward_inputs(N=5)
        Ks_py, ks_py = _backward_pass_numpy(
            As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu)
        ok, Ks_cpp, ks_cpp = _call_cpp_backward(
            As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu)
        assert ok, "C++ backward 返回 ok=False（不应奇异）"
        assert Ks_cpp.shape == (5, 6, 12)
        assert ks_cpp.shape == (5, 6)
        for k in range(5):
            np.testing.assert_allclose(Ks_cpp[k], Ks_py[k], atol=1e-10,
                err_msg=f"Ks[{k}] 偏差")
            np.testing.assert_allclose(ks_cpp[k], ks_py[k], atol=1e-10,
                err_msg=f"ks[{k}] 偏差")

    def test_n60_matches_python(self):
        """N=60 全尺寸轨迹: C++ 与 Python atol=1e-10（验证真实规划地平线）。"""
        _, As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu = \
            _build_backward_inputs(N=60)
        Ks_py, ks_py = _backward_pass_numpy(
            As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu)
        ok, Ks_cpp, ks_cpp = _call_cpp_backward(
            As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu)
        assert ok
        np.testing.assert_allclose(Ks_cpp, np.stack(Ks_py), atol=1e-10,
            err_msg="Ks (N=60) 整体偏差")
        np.testing.assert_allclose(ks_cpp, np.stack(ks_py), atol=1e-10,
            err_msg="ks (N=60) 整体偏差")


# ============================================================================
# Slice 3: 奇异检测
# ============================================================================

class TestBackwardSingularity:
    """Q_uu_reg 奇异时 C++ 与 Python 行为一致（报告失败）。"""

    def test_singular_returns_false(self):
        """Q_uu_reg 奇异（全零 + mu=0）: C++ ok=False，Python None。"""
        N = 3
        As = [np.eye(12) for _ in range(N)]
        Bs = [np.zeros((12, 6)) for _ in range(N)]      # B=0 → Q_uu = l_uu
        l_xs = [np.zeros(12) for _ in range(N)]
        l_us = [np.zeros(6) for _ in range(N)]
        l_xxs = [np.zeros((12, 12)) for _ in range(N)]
        l_uxs = [np.zeros((6, 12)) for _ in range(N)]
        l_uus = [np.zeros((6, 6)) for _ in range(N)]    # l_uu=0 → Q_uu_reg=0 奇异
        l_x_N = np.zeros(12)
        l_xx_N = np.zeros((12, 12))
        mu = 0.0
        py_result = _backward_pass_numpy(
            As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu)
        assert py_result is None, "Python numpy 应检测奇异返回 None"
        ok, _, _ = _call_cpp_backward(
            As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu)
        assert not ok, "C++ 应报告奇异 ok=False"


# ============================================================================
# Slice 4: 端到端集成
# ============================================================================

class TestBackwardIntegration:
    """solve_few_iters 用 C++ backward 与强制 Python backward 产出一致轨迹。"""

    def test_solve_few_iters_matches(self, monkeypatch):
        """5 迭代 solve: C++ backward vs Python backward 轨迹 atol=1e-8。

        注意：两次 solve 使用独立的 cost_fn 实例，避免 SmoothnessTerm._u_prev
        状态泄漏（防御性隔离，当前 k>0 守卫已阻止 k=0 使用 stale _u_prev，
        但独立实例可防止未来守卫被移除时测试以困惑方式失败）。
        """
        env = _make_env()
        N = 30
        solver = _make_solver(N)
        init_q = np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0])
        x0 = np.concatenate([init_q, np.zeros(6)])
        U_init = np.random.default_rng(99).standard_normal((N, 6)) * 2.0

        # C++ backward（默认）
        cost_fn_cpp = _make_cost(env, N)
        X_cpp, U_cpp, _, ok_cpp = solver.solve_few_iters(
            env, cost_fn_cpp, x0, U_init, max_iter=5,
            skip_linesearch=True, use_fast_lin=True)
        # 强制 Python backward（独立 cost_fn 避免状态泄漏）
        monkeypatch.setattr(
            solver, "_backward_pass",
            lambda As, Bs, lx, lu, lxx, lux, luu, lxN, lxxN, mu:
                _backward_pass_numpy(As, Bs, lx, lu, lxx, lux, luu, lxN, lxxN, mu))
        cost_fn_py = _make_cost(env, N)
        X_py, U_py, _, ok_py = solver.solve_few_iters(
            env, cost_fn_py, x0, U_init, max_iter=5,
            skip_linesearch=True, use_fast_lin=True)
        assert ok_cpp and ok_py
        np.testing.assert_allclose(X_cpp, X_py, atol=1e-8, err_msg="轨迹偏差")
        np.testing.assert_allclose(U_cpp, U_py, atol=1e-8, err_msg="控制偏差")
