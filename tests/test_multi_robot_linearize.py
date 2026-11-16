"""多机器人有限差分线性化与七维求解闭环测试（阶段 3 · Task 3.5）。

核心断言组（brief + 施工计划 Task 3.5）：
1. ``test_fd_shapes_*`` / ``test_fd_directional_*``：真实 R1D 双侧裸模型
   （n=7，经 ``load_robot_model``）+ 左侧带球模型（free joint 前置的非
   前缀布局）——FD 的 A/B 形状 ``(14,14)/(14,7)``；在限位/饱和之外的
   平滑姿态，独立方向扰动验证 ``f(x+dx,u+du)-f(x,u)`` 与 ``A@dx+B@du``
   一致（rtol=1e-3/atol=1e-6）；
2. ``test_seventh_input_*``：非零第七输入必须影响动力学（B 第 7 列非零
   + 行为级 step 差异），且全部 7 个输入列均非零；
3. ``test_fd_isolation_*``：FD 线性化不扰动 live 数据（隔离契约在求解
   器消费侧复核）；
4. ``test_reordered_*``：六维地址重排合成模型上的 FD 方向导数对照——
   FD 按绑定地址生效，而非前缀假设；
5. ``test_lq_*``：七维正定线性二次系统闭环（LinearTestEnv，无 MuJoCo、
   A/B 精确已知）——成本下降并收敛到解析 LQR 最优值；后向传递增益
   K ``(N,7,14)`` / k ``(N,7)`` 与独立 Riccati 递推一致（增益合理性，
   不把"形状正确"当唯一验收）；前向线搜索按 ``control_range`` 裁剪；
   ``solve``/``solve_few_iters`` 维持旧返回结构。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from src.dynamics.linearize import linearize_dynamics, linearize_trajectory
from src.ilqt.solver import build_solver
from src.robot.model_factory import load_robot_model
from src.robot.robot_profile import load_robot_profile
from src.robot.runtime_parameters import load_runtime_parameters
from src.sim.bound_robot_env import BoundRobotEnv
from src.sim.sim_config import load_sim_task_config
from src.sim.tennis_scene import build_tennis_scene

from test_solver_backend_selection import (
    QuadraticCost,
    build_reordered_six_dof_env,
    make_quadratic_cost,
)

# 仓库根目录（tests/ 的父目录）
REPO_ROOT = Path(__file__).resolve().parent.parent

# 随机种子（固定，保证失败可复现）
_RNG_SEED = 20261116

# 方向导数验收容差（brief 逐字）
_RTOL = 1e-3
_ATOL = 1e-6


# ---------- 环境用例 ----------


@lru_cache(maxsize=4)
def _r1d_profile(side: str):
    """加载并缓存该侧 RobotProfile。"""
    return load_robot_profile(REPO_ROOT / "configs" / "robots" / f"r1d_{side}.yaml")


@lru_cache(maxsize=4)
def _r1d_params(side: str):
    """加载该侧仿真 runtime 参数。"""
    return load_runtime_parameters(
        REPO_ROOT / "configs" / "robots" / f"r1d_sim_runtime_{side}.yaml",
        _r1d_profile(side),
    )


@lru_cache(maxsize=4)
def _r1d_env(case: str) -> BoundRobotEnv:
    """构建测试环境：双侧裸模型 + 左侧带球模型（非前缀布局）。"""
    if case == "r1d_tennis_left":
        task = load_sim_task_config(
            REPO_ROOT / "configs" / "r1d_sim_left.yaml", _r1d_profile("left")
        )
        scene = build_tennis_scene(_r1d_profile("left"), _r1d_params("left"), task)
        return BoundRobotEnv(scene.robot)
    side = case.split("_")[1]
    return BoundRobotEnv(load_robot_model(_r1d_profile(side), _r1d_params(side)))


_CASES = ("r1d_left", "r1d_right", "r1d_tennis_left")


def _smooth_operating_point(env: BoundRobotEnv, case: str) -> tuple[np.ndarray, np.ndarray]:
    """构造限位/饱和之外的平滑姿态与控制。

    q 取关节范围中部 40%~60%（远离限位），qdot ∈ (-0.3, 0.3) rad/s，
    u = 0.15·torque_max·(±1)（远离执行器饱和）。
    """
    rng = np.random.default_rng(_RNG_SEED + hash(case) % 1000)
    joint_ids = np.asarray(env.binding.joint_ids, dtype=int)
    lower = env.model.jnt_range[joint_ids, 0]
    upper = env.model.jnt_range[joint_ids, 1]
    q = lower + (upper - lower) * (0.4 + 0.2 * rng.random(env.NQ))
    qdot = rng.uniform(-0.3, 0.3, env.NQ)
    x = np.concatenate([q, qdot])
    signs = np.where(rng.random(env.NU) < 0.5, -1.0, 1.0)
    u = 0.15 * env.torque_max * signs
    # 前提自检：姿态在限位内部、控制未饱和
    assert bool(np.all(q > lower) and np.all(q < upper))
    assert bool(np.all(np.abs(u) < 0.5 * env.torque_max))
    return x, u


# ---------- 1. FD 形状与方向导数 ----------


@pytest.mark.parametrize("case", _CASES)
def test_fd_shapes_seven_dof(case: str) -> None:
    """FD 的 A/B/x_next 形状 (14,14)/(14,7)/(14,)。"""
    env = _r1d_env(case)
    assert (env.NQ, env.NX, env.NU) == (7, 14, 7)
    x, u = _smooth_operating_point(env, case)
    A, B, x_next = linearize_dynamics(env, x, u, eps=1e-5)
    assert A.shape == (14, 14)
    assert B.shape == (14, 7)
    assert x_next.shape == (14,)


@pytest.mark.parametrize("case", _CASES)
def test_fd_directional_derivative_seven_dof(case: str) -> None:
    """独立方向扰动：f(x+dx,u+du)-f(x,u) ≈ A@dx+B@du（rtol 1e-3/atol 1e-6）。"""
    env = _r1d_env(case)
    x, u = _smooth_operating_point(env, case)
    A, B, _ = linearize_dynamics(env, x, u, eps=1e-5)

    rng = np.random.default_rng(_RNG_SEED + 7)
    for trial in range(3):
        # 扰动尺度 5e-5：二阶截断误差 ~4e-7（atol=1e-6 的 2.5 倍裕度内，
        # 经 20 组随机方向数值核对），信号仍高于 atol 一个量级
        dx = np.zeros(14)
        dx[:7] = 5e-5 * rng.uniform(-1.0, 1.0, 7)   # 关节角扰动
        dx[7:] = 5e-5 * rng.uniform(-1.0, 1.0, 7)   # 关节速度扰动
        du = 5e-5 * env.torque_max * rng.uniform(-1.0, 1.0, 7)
        actual = env.step_from_state(x + dx, u + du) - env.step_from_state(x, u)
        predicted = A @ dx + B @ du
        np.testing.assert_allclose(
            actual, predicted, rtol=_RTOL, atol=_ATOL,
            err_msg=f"{case} 第 {trial} 个方向导数失配",
        )


@pytest.mark.parametrize("case", _CASES)
def test_fd_repeatable(case: str) -> None:
    """FD 线性化重复调用结果相同（确定性，无隐藏状态污染）。"""
    env = _r1d_env(case)
    x, u = _smooth_operating_point(env, case)
    A1, B1, f1 = linearize_dynamics(env, x, u, eps=1e-5)
    A2, B2, f2 = linearize_dynamics(env, x, u, eps=1e-5)
    assert np.array_equal(A1, A2)
    assert np.array_equal(B1, B2)
    assert np.array_equal(f1, f2)


# ---------- 2. 第七输入敏感性 ----------


@pytest.mark.parametrize("case", _CASES)
def test_seventh_input_affects_dynamics(case: str) -> None:
    """非零第七输入必须影响动力学：B 第 7 列非零 + 行为级 step 差异。"""
    env = _r1d_env(case)
    x, u = _smooth_operating_point(env, case)
    A, B, _ = linearize_dynamics(env, x, u, eps=1e-5)

    # B 的第 7 列（索引 6）显著非零
    assert float(np.linalg.norm(B[:, 6])) > 1e-6, "第七输入对动力学无影响"

    # 全部 7 个输入列均非零（任意输入维度不被漏掉）
    for j in range(7):
        assert float(np.linalg.norm(B[:, j])) > 1e-6, f"输入 {j} 对动力学无影响"

    # 行为级：仅改变第 7 输入（非饱和幅度），下一步状态必须不同
    u2 = u.copy()
    u2[6] = 0.5 * env.torque_max[6]
    x_next1 = env.step_from_state(x, u)
    x_next2 = env.step_from_state(x, u2)
    assert float(np.linalg.norm(x_next2 - x_next1)) > 1e-6


# ---------- 3. FD 不扰动 live 数据 ----------


@pytest.mark.parametrize("case", _CASES)
def test_fd_does_not_disturb_live_data(case: str) -> None:
    """整条轨迹 FD 线性化后 live qpos/qvel/ctrl/time 逐项不变。"""
    env = _r1d_env(case)
    x, u = _smooth_operating_point(env, case)
    env.set_arm_state(x)
    U = np.tile(u, (3, 1))
    X = np.tile(x, (4, 1))

    qpos0 = env.data.qpos.copy()
    qvel0 = env.data.qvel.copy()
    ctrl0 = env.data.ctrl.copy()
    time0 = float(env.data.time)

    linearize_trajectory(env, X, U, eps=1e-5)

    assert np.array_equal(env.data.qpos, qpos0)
    assert np.array_equal(env.data.qvel, qvel0)
    assert np.array_equal(env.data.ctrl, ctrl0)
    assert float(env.data.time) == time0


# ---------- 4. 六维地址重排模型上的 FD ----------


def test_fd_directional_derivative_reordered_six_dof() -> None:
    """六维地址重排模型：FD 按绑定地址生效（非前缀布局方向导数对照）。"""
    env = build_reordered_six_dof_env()
    assert (env.NQ, env.NX, env.NU) == (6, 12, 6)

    rng = np.random.default_rng(_RNG_SEED + 13)
    joint_ids = np.asarray(env.binding.joint_ids, dtype=int)
    lower = env.model.jnt_range[joint_ids, 0]
    upper = env.model.jnt_range[joint_ids, 1]
    q = lower + (upper - lower) * (0.4 + 0.2 * rng.random(6))
    x = np.concatenate([q, rng.uniform(-0.3, 0.3, 6)])
    u = 0.15 * env.torque_max * np.where(rng.random(6) < 0.5, -1.0, 1.0)

    A, B, _ = linearize_dynamics(env, x, u, eps=1e-5)
    assert A.shape == (12, 12)
    assert B.shape == (12, 6)

    for trial in range(3):
        dx = 5e-5 * rng.uniform(-1.0, 1.0, 12)
        du = 5e-5 * env.torque_max * rng.uniform(-1.0, 1.0, 6)
        actual = env.step_from_state(x + dx, u + du) - env.step_from_state(x, u)
        predicted = A @ dx + B @ du
        np.testing.assert_allclose(actual, predicted, rtol=_RTOL, atol=_ATOL)


# ---------- 5. 七维正定线性二次系统闭环 ----------


class LinearTestEnv:
    """七维线性系统测试环境（无 MuJoCo，A/B 精确已知）。

    x_{k+1} = A @ x + B @ u（双积分器 + 速度阻尼 0.98）。
    刻意**不提供** ``model`` 属性：前向/线搜索路径必须改用
    ``control_range``（Task 3.5），旧前缀 ``model.actuator_ctrlrange``
    实现在此环境上直接 AttributeError。
    """

    def __init__(
        self,
        control_range: np.ndarray,
        dt: float = 0.05,
        damping: float = 0.95,
        input_gain: float = 3.0,
    ) -> None:
        """初始化线性系统与控制范围。

        Args:
            control_range: (7,2) 控制范围。
            dt: 时间步长（s）。
            damping: 速度衰减系数。
            input_gain: 输入增益（B 下块 = input_gain·dt·I，保证控制
                有足够权威使 LQR 最优成本显著低于零控制成本）。
        """
        n = 7
        self.NQ = n
        self.NX = 2 * n
        self.NU = n
        self._dt = dt
        A = np.zeros((2 * n, 2 * n))
        A[:n, :n] = np.eye(n)
        A[:n, n:] = dt * np.eye(n)
        A[n:, n:] = damping * np.eye(n)
        self.A = A
        B = np.zeros((2 * n, n))
        B[n:, :] = input_gain * dt * np.eye(n)
        self.B = B
        self._control_range = np.array(control_range, dtype=np.float64)
        self._x = np.zeros(2 * n)

    @property
    def dt(self) -> float:
        """时间步长（s）。"""
        return self._dt

    @property
    def control_range(self) -> np.ndarray:
        """(7,2) 控制范围副本。"""
        return self._control_range.copy()

    def get_arm_state(self) -> np.ndarray:
        """读取当前状态。"""
        return self._x.copy()

    def set_arm_state(self, x: np.ndarray) -> None:
        """写入当前状态。"""
        self._x = np.array(x, dtype=np.float64)

    def step(self, u: np.ndarray) -> np.ndarray:
        """live 前进一步。"""
        self._x = self.A @ self._x + self.B @ np.asarray(u, dtype=np.float64)
        return self._x.copy()

    def step_from_state(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """从指定状态前进一步（纯函数）。"""
        return self.A @ np.asarray(x, dtype=np.float64) + self.B @ np.asarray(
            u, dtype=np.float64
        )


def _lqr_reference(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, Qf: np.ndarray, N: int
) -> tuple[list[np.ndarray], np.ndarray]:
    """独立离散 LQR Riccati 递推（测试对照，不依赖求解器实现）。"""
    P = Qf.copy()
    Ks: list[np.ndarray] = []
    for _ in range(N):
        BtP = B.T @ P
        K = -np.linalg.solve(R + BtP @ B, BtP @ A)
        P = Q + A.T @ P @ (A + B @ K)
        P = 0.5 * (P + P.T)
        Ks.append(K)
    return Ks[::-1], P


def _lqr_optimal_cost(
    env: LinearTestEnv,
    cost: QuadraticCost,
    x0: np.ndarray,
    Ks: list[np.ndarray],
) -> float:
    """沿 LQR 增益回放生成最优轨迹并用与求解器相同的代价函数计值。"""
    x = x0.copy()
    total = 0.0
    for k in range(len(Ks)):
        u = Ks[k] @ x
        total += cost.running_cost(x, u, k)
        x = env.step_from_state(x, u)
    total += cost.terminal_cost(x)
    return total


def _make_linear_case() -> tuple[LinearTestEnv, QuadraticCost, np.ndarray, int]:
    """构造七维 LQ 用例：环境、代价、初始状态、地平线。

    参数经数值核对：零控制成本 J0≈9.45，解析 LQR 最优 J*≈3.51
    （比值≈0.37，控制权威充分）；最优控制幅值 ≈1.18，远在 ±10
    控制范围内（主用例不受约束影响，裁剪行为由窄范围用例单独覆盖）。
    """
    env = LinearTestEnv(control_range=np.tile(np.array([-10.0, 10.0]), (7, 1)))
    Q = np.diag(np.concatenate([np.ones(7), 0.1 * np.ones(7)]))
    cost = QuadraticCost(Q=Q, R=0.05 * np.eye(7), Qf=10.0 * Q)
    rng = np.random.default_rng(_RNG_SEED + 21)
    x0 = np.concatenate([0.3 * np.where(rng.random(7) < 0.5, -1.0, 1.0), np.zeros(7)])
    return env, cost, x0, 20


def test_lq_cost_decreases_to_analytic_optimum() -> None:
    """七维 LQ 系统：成本下降并收敛到解析 LQR 最优值（1e-3 相对带内）。"""
    env, cost, x0, N = _make_linear_case()
    # 非兼容环境（无 init_q_left）→ auto 自动选择 Python FD
    solver = build_solver(
        env=env, backend="auto", config={"horizon": N, "max_iter": 40, "tol": 1e-11}
    )
    assert solver.use_analytical is False

    X, U, cost_history = solver.solve(env, cost, x0)
    assert X.shape == (N + 1, 14)
    assert U.shape == (N, 7)

    # 成本下降：历史非增，且显著低于零控制初值
    diffs = np.diff(np.asarray(cost_history))
    assert bool(np.all(diffs <= 1e-12)), f"代价历史非单调递减: {cost_history}"
    assert cost_history[-1] < 0.5 * cost_history[0]

    # 收敛到解析最优：与独立 Riccati 参考一致（不把"下降"当唯一验收）
    Ks_ref, _ = _lqr_reference(env.A, env.B, cost.Q, cost.R, cost.Qf, N)
    j_opt = _lqr_optimal_cost(env, cost, x0, Ks_ref)
    assert j_opt < cost_history[0], "LQR 参考解应优于零控制"
    assert abs(cost_history[-1] - j_opt) <= 1e-3 * max(1.0, j_opt), (
        f"未收敛到解析最优: solver={cost_history[-1]:.6f}, LQR={j_opt:.6f}"
    )


def test_lq_backward_pass_gains_match_riccati() -> None:
    """后向传递 K (7,14)/k (7,) 与独立 Riccati 增益一致（形状+数值）。"""
    env, cost, x0, N = _make_linear_case()
    solver = build_solver(env=env, backend="python", config={"horizon": N})

    # 名义轨迹（零控制）与精确 A/B（线性系统解析已知）
    X = np.zeros((N + 1, 14))
    X[0] = x0
    U = np.zeros((N, 7))
    for k in range(N):
        X[k + 1] = env.step_from_state(X[k], U[k])
    As = [env.A] * N
    Bs = [env.B] * N
    l_xs, l_us, l_xxs, l_uxs, l_uus = [], [], [], [], []
    for k in range(N):
        lx, lu, lxx, lux, luu = cost.running_derivatives(X[k], U[k], k)
        l_xs.append(lx)
        l_us.append(lu)
        l_xxs.append(lxx)
        l_uxs.append(lux)
        l_uus.append(luu)
    l_x_N, l_xx_N = cost.terminal_derivatives(X[-1])

    Ks, ks = solver._backward_pass(
        As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu=1e-9
    )

    # 形状验收：K (N,7,14)、k (N,7)
    assert len(Ks) == N and len(ks) == N
    for K_k, k_k in zip(Ks, ks):
        assert K_k.shape == (7, 14)
        assert k_k.shape == (7,)

    # 数值验收：与独立 Riccati 增益一致（LQ 问题后向传递精确）
    Ks_ref, _ = _lqr_reference(env.A, env.B, cost.Q, cost.R, cost.Qf, N)
    for k, (K_solver, K_ref) in enumerate(zip(Ks, Ks_ref)):
        np.testing.assert_allclose(
            K_solver, K_ref, rtol=1e-5, atol=1e-7,
            err_msg=f"增益 K[{k}] 与 Riccati 参考不一致",
        )
    # 增益合理性：有限且幅值有界
    for K_k in Ks:
        assert bool(np.all(np.isfinite(K_k)))
        assert float(np.max(np.abs(K_k))) < 1e3


def test_lq_forward_pass_respects_control_range() -> None:
    """前向传递按 env.control_range 裁剪（环境无 model 属性）。

    旧实现读 ``env.model.actuator_ctrlrange[:NU]``——本环境刻意不提供
    ``model`` 属性，前缀实现直接 AttributeError；新实现必须消费
    ``control_range``（n,2）。用 ``forward_pass_single``（无代价接受
    耦合）单测裁剪行为本身。
    """
    from src.ilqt.utils import forward_pass_single

    # 窄范围：[-0.05, 0.05]
    env = LinearTestEnv(control_range=np.tile(np.array([-0.05, 0.05]), (7, 1)))
    cost = make_quadratic_cost(7)
    N = 6
    x0 = np.concatenate([np.ones(7), np.zeros(7)])
    X = np.zeros((N + 1, 14))
    X[0] = x0
    U = np.zeros((N, 7))
    for k in range(N):
        X[k + 1] = env.step_from_state(X[k], U[k])
    # 零反馈 + 单位前馈：名义控制 0 + 0.5·1.0 = 0.5，超出 ±0.05
    Ks = [np.zeros((7, 14)) for _ in range(N)]
    ks = [np.ones(7) for _ in range(N)]

    X_new, U_new, _cost, reject_reason = forward_pass_single(
        env, cost, X, U, Ks, ks, alpha=0.5
    )
    assert X_new is not None and U_new is not None, f"前向传递被拒: {reject_reason}"
    lo, hi = env.control_range[:, 0], env.control_range[:, 1]
    assert bool(np.all(U_new >= lo - 1e-12) and np.all(U_new <= hi + 1e-12))
    # 名义控制 0 + 0.5·k 被裁到上界 0.05
    np.testing.assert_allclose(U_new, np.tile(hi, (N, 1)), atol=1e-12)


def test_lq_solve_few_iters_return_structure() -> None:
    """solve_few_iters 维持旧返回结构 (X, U, cost_history, success)。"""
    env, cost, x0, N = _make_linear_case()
    solver = build_solver(
        env=env, backend="python", config={"horizon": N, "max_iter": 3}
    )
    U0 = np.zeros((N, 7))
    X, U, history, success = solver.solve_few_iters(
        env, cost, x0, U0, max_iter=2, skip_linesearch=False
    )
    assert isinstance(X, np.ndarray) and X.shape == (N + 1, 14)
    assert isinstance(U, np.ndarray) and U.shape == (N, 7)
    assert isinstance(history, list) and len(history) >= 1
    assert isinstance(success, bool) and success is True
