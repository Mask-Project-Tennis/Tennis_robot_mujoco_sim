"""C++ 前向传递等效性测试（P1）。

验证 C++ sim_step / forward_pass / check_step / rollout 与 Python 参考实现数值等价。
所有测试通过公共 pybind11 接口验证，不触碰 C++ 内部结构。
"""

import numpy as np
import mujoco
import pytest
from pathlib import Path
from collections import deque

from src.sim.rm65_env import RM65Env
from src.ilqt.utils import forward_pass_single as py_forward_pass_single
from src.ilqt.robot_limits import RobotLimits, build_qdot_history, compute_qddot_filtered

# 尝试加载 C++ 模块
try:
    from src.cpp.iLQR_Core import (
        sim_step as cpp_sim_step_raw,
    )
    _CPP_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _CPP_AVAILABLE = False

# C++ 模块不可用时跳过所有测试
pytestmark = pytest.mark.skipif(not _CPP_AVAILABLE, reason="C++ 模块未编译")


def _make_env() -> RM65Env:
    """创建测试用 RM65Env 实例（力矩模式）。"""
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
    return RM65Env(model_path)


def _make_env_position(kp: float = 25.0, kd: float = 2.0, ff: bool = True) -> RM65Env:
    """创建位置模式 env。"""
    env = _make_env()
    env.configure_actuator_mode("position", kp=np.full(6, kp), kd=np.full(6, kd))
    env.configure_feedforward(ff)
    return env


def _get_ptrs(env: RM65Env) -> tuple[int, int]:
    """获取 model/data 指针。"""
    return env.model._address, env.data._address


def _call_cpp_sim_step(env: RM65Env, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """调用 C++ sim_step，自动从 env 提取 actuator 参数。"""
    actuator_mode = getattr(env, 'actuator_mode', 0)
    kp = getattr(env, 'kp', None)
    kd = getattr(env, 'kd', None)
    use_ff = getattr(env, 'use_feedforward', False)
    # torque_max 用原始力矩限制（configure_actuator_mode 前保存的值），
    # 位置模式下 actuator_ctrlrange 已被改为 jnt_range
    torque_max = env._torque_ctrlrange[:, 1].copy()
    ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0].copy()
    ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1].copy()
    return cpp_sim_step_raw(
        x, u, *_get_ptrs(env), env.init_q_left,
        ctrl_lo, ctrl_hi,
        actuator_mode, kp, kd, use_ff, torque_max,
    )


# ============================================================================
# Phase A: sim_step 等效性
# ============================================================================

class TestSimStepTorqueMode:
    """A1-A4: C++ sim_step 与 Python step_from_state 数值等价。"""

    def test_torque_mode_matches_python(self) -> None:
        """A1[tracer]: 力矩模式 C++ sim_step 与 Python step_from_state 一致。"""
        env = _make_env()
        env.reset(q0=np.array([0.1, -0.3, 0.5, 0, 0, 0]))
        x = np.array([0.1, -0.3, 0.5, 0, 0, 0, 0.1, -0.2, 0.05, 0, 0, 0])
        u = np.array([5.0, -3.0, 2.0, 0.0, 0.0, 0.0])

        x_next_py = env.step_from_state(x.copy(), u.copy())

        actuator_mode = getattr(env, 'actuator_mode', 0)
        kp = getattr(env, 'kp', None)
        kd = getattr(env, 'kd', None)
        use_ff = getattr(env, 'use_feedforward', False)
        torque_max = env.model.actuator_ctrlrange[:env.NU, 1].copy()
        ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0].copy()
        ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1].copy()

        x_next_cpp = cpp_sim_step_raw(
            x, u,
            *_get_ptrs(env),
            env.init_q_left,
            ctrl_lo, ctrl_hi,
            actuator_mode, kp, kd, use_ff, torque_max,
        )
        np.testing.assert_allclose(x_next_cpp, x_next_py, atol=1e-12,
                                   err_msg="力矩模式 sim_step 不一致")

    def test_position_mode_matches_python(self) -> None:
        """A2: 位置模式 C++ sim_step 与 Python step 一致。"""
        env = _make_env_position(kp=25.0, kd=2.0, ff=False)
        env.reset(q0=np.array([0.1, -0.3, 0.5, 0, 0, 0]))
        x = np.array([0.1, -0.3, 0.5, 0, 0, 0, 0.1, -0.2, 0.05, 0, 0, 0])
        u = np.array([0.2, -0.1, 0.3, 0.1, 0, 0])

        x_next_py = env.step_from_state(x.copy(), u.copy())
        x_next_cpp = _call_cpp_sim_step(env, x, u)
        np.testing.assert_allclose(x_next_cpp, x_next_py, atol=1e-12,
                                   err_msg="位置模式 sim_step 不一致")

    def test_position_ff_matches_python(self) -> None:
        """A3: 位置模式+前馈 C++ sim_step 与 Python step(FF) 一致。"""
        env = _make_env_position(kp=25.0, kd=2.0, ff=True)
        env.reset(q0=np.array([0.1, -0.3, 0.5, 0, 0, 0]))
        x = np.array([0.1, -0.3, 0.5, 0, 0, 0, 0.1, -0.2, 0.05, 0, 0, 0])
        u = np.array([0.2, -0.1, 0.3, 0.1, 0, 0])

        x_next_py = env.step_from_state(x.copy(), u.copy())
        x_next_cpp = _call_cpp_sim_step(env, x, u)
        np.testing.assert_allclose(x_next_cpp, x_next_py, atol=1e-12,
                                   err_msg="位置模式+FF sim_step 不一致")

    def test_position_noff_matches_python(self) -> None:
        """A4: 位置模式无前馈 C++ sim_step 与 Python step(无FF) 一致。"""
        env = _make_env_position(kp=25.0, kd=2.0, ff=False)
        env.reset(q0=np.array([0.1, -0.3, 0.5, 0, 0, 0]))
        x = np.array([0.1, -0.3, 0.5, 0, 0, 0, 0.1, -0.2, 0.05, 0, 0, 0])
        u = np.array([0.2, -0.1, 0.3, 0.1, 0, 0])

        x_next_py = env.step_from_state(x.copy(), u.copy())
        x_next_cpp = _call_cpp_sim_step(env, x, u)
        np.testing.assert_allclose(x_next_cpp, x_next_py, atol=1e-12,
                                    err_msg="位置模式无FF sim_step 不一致")


# ============================================================================
# Phase B: forward_pass / rollout 等效性
# ============================================================================

# 尝试加载 forward_pass C++ 绑定
try:
    from src.cpp.iLQR_Core import forward_pass_single as cpp_fp_single_raw
    _FP_CPP_AVAILABLE = True
except (ImportError, AttributeError):
    _FP_CPP_AVAILABLE = False


def _make_simple_trajectory(env: RM65Env, N: int = 20):
    """生成简单轨迹 + 随机增益。"""
    rng = np.random.default_rng(42)
    X = np.zeros((N + 1, 12))
    U = rng.normal(0, 0.1, (N, 6))
    env.reset(q0=np.zeros(6))
    X[0] = np.zeros(12)
    env.set_arm_collision(False)
    for k in range(N):
        X[k + 1] = env.step_from_state(X[k], U[k])
    env.set_arm_collision(True)
    Ks = [np.zeros((6, 12)) for _ in range(N)]
    ks = rng.normal(0, 0.01, (N, 6))
    return X, U, Ks, ks


def _call_cpp_forward_pass(env, X, U, Ks, ks, alpha=0.5):
    """调用 C++ forward_pass_single。"""
    N = len(U)
    Ks_flat = np.array(Ks).reshape(N, 72) if not isinstance(Ks, np.ndarray) else Ks
    ks_arr = np.array(ks).reshape(N, 6) if not isinstance(ks, np.ndarray) else ks
    X_new = np.zeros_like(X)
    U_new = np.zeros_like(U)
    actuator_mode = getattr(env, 'actuator_mode', 0)
    kp = getattr(env, 'kp', None)
    kd = getattr(env, 'kd', None)
    use_ff = getattr(env, 'use_feedforward', False)
    torque_max = env._torque_ctrlrange[:, 1].copy()
    ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0].copy()
    ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1].copy()
    ball_geom_start = env.model.body("ball").geomadr[0]

    ok = cpp_fp_single_raw(
        X_new, U_new, X, U, Ks_flat, ks_arr,
        env.model._address, env.data._address,
        env.init_q_left,
        ctrl_lo, ctrl_hi, alpha,
        actuator_mode, kp, kd, use_ff, torque_max,
        ball_geom_start, True,  # disable_collision=True
    )
    return X_new, U_new, ok


@pytest.mark.skipif(not _FP_CPP_AVAILABLE, reason="C++ forward_pass 未更新")
class TestForwardPassEquivalence:
    """B1-B3: C++ forward_pass 与 Python forward_pass_single 等价。"""

    def test_forward_pass_no_limits_torque(self) -> None:
        """B1: 力矩模式无 limits，C++ forward_pass 与 Python 一致。"""
        env = _make_env()
        X, U, Ks, ks = _make_simple_trajectory(env, N=20)
        # Python 参考
        X_py, U_py, _, _ = py_forward_pass_single(
            env, None, X.copy(), U.copy(), Ks, ks, alpha=0.5)
        # C++
        X_cpp, U_cpp, ok = _call_cpp_forward_pass(env, X, U, Ks, ks, alpha=0.5)
        assert ok, "C++ forward_pass 应成功"
        np.testing.assert_allclose(X_cpp, X_py, atol=1e-10,
                                   err_msg="forward_pass X 不一致")
        np.testing.assert_allclose(U_cpp, U_py, atol=1e-10,
                                   err_msg="forward_pass U 不一致")


# ============================================================================
# Phase C: check_step 约束检查
# ============================================================================

try:
    from src.cpp.iLQR_Core import check_step as cpp_check_step_raw
    _CHECK_CPP_AVAILABLE = True
except (ImportError, AttributeError):
    _CHECK_CPP_AVAILABLE = False


def _make_check_params(
    q_lo=-3.0*np.ones(6), q_hi=3.0*np.ones(6),
    qd_max=3.0*np.ones(6),
    u_lo=-60.0*np.ones(6), u_hi=60.0*np.ones(6),
    qdd_max=10.0*np.ones(6),
    margin=1.5, fp_q_tol=0.0,
    actuator_mode=0, qdd_window=5,
    dt=0.005, qdd_hard_reject=False,
) -> dict:
    """构造 check_step 参数字典。"""
    return {
        "q_lo": np.asarray(q_lo, dtype=np.float64),
        "q_hi": np.asarray(q_hi, dtype=np.float64),
        "qd_max": np.asarray(qd_max, dtype=np.float64),
        "u_lo": np.asarray(u_lo, dtype=np.float64),
        "u_hi": np.asarray(u_hi, dtype=np.float64),
        "qdd_max": np.asarray(qdd_max, dtype=np.float64),
        "margin": margin, "fp_q_tol": fp_q_tol,
        "actuator_mode": actuator_mode,
        "qdd_window": qdd_window, "dt": dt,
        "qdd_hard_reject": qdd_hard_reject,
    }


@pytest.mark.skipif(not _CHECK_CPP_AVAILABLE, reason="C++ check_step 未编译")
class TestCheckStep:
    """C1-C4: C++ check_step 约束检查。"""

    def test_feasible_step(self) -> None:
        """C1[tracer]: 满足所有约束的步返回 feasible=True。"""
        x_prev = np.zeros(12)
        x_next = np.zeros(12)
        u = np.zeros(6)
        qdot_hist = np.zeros((6, 6))
        params = _make_check_params()
        feasible, reason = cpp_check_step_raw(
            x_prev, x_next, u, qdot_hist.flatten(), 1, params)
        assert feasible, f"应可行但被拒: {reason}"

    def test_q_violation(self) -> None:
        """C2: q 超界拒绝。"""
        x_next = np.zeros(12)
        x_next[0] = 10.0  # 远超 q_hi=3.0
        params = _make_check_params()
        feasible, reason = cpp_check_step_raw(
            np.zeros(12), x_next, np.zeros(6),
            np.zeros(6), 0, params)
        assert not feasible
        assert "q" in reason.lower()

    def test_qdot_braking_decel_passes(self) -> None:
        """C3: qdot 超限但减速时通过（制动方向）。"""
        x_prev = np.zeros(12); x_prev[6] = 6.0
        x_next = np.zeros(12); x_next[6] = 5.0  # 减速但超限 (>3.0*1.5=4.5)
        params = _make_check_params()
        feasible, _ = cpp_check_step_raw(
            x_prev, x_next, np.zeros(6),
            np.zeros(6), 0, params)
        assert feasible, "制动中应允许超限"

    def test_qdot_accelerating_fails(self) -> None:
        """C3: qdot 超限且加速时拒绝。"""
        x_prev = np.zeros(12); x_prev[6] = 4.0
        x_next = np.zeros(12); x_next[6] = 5.0  # 加速且超限
        params = _make_check_params()
        feasible, reason = cpp_check_step_raw(
            x_prev, x_next, np.zeros(6),
            np.zeros(6), 0, params)
        assert not feasible
        assert "qdot" in reason.lower()

    def test_u_torque_mode_rejected(self) -> None:
        """C4: 力矩模式 u 超界拒绝。"""
        u = np.array([100.0, 0, 0, 0, 0, 0])
        params = _make_check_params(actuator_mode=0)
        feasible, reason = cpp_check_step_raw(
            np.zeros(12), np.zeros(12), u,
            np.zeros(6), 0, params)
        assert not feasible
        assert "u" in reason.lower()

    def test_u_position_mode_skipped(self) -> None:
        """C4: 位置模式跳过 u 检查。"""
        u = np.array([100.0, 0, 0, 0, 0, 0])
        params = _make_check_params(actuator_mode=1)
        feasible, _ = cpp_check_step_raw(
            np.zeros(12), np.zeros(12), u,
            np.zeros(6), 0, params)
        assert feasible, "位置模式应跳过 u 检查"

    def test_qddot_hard_reject(self) -> None:
        """C4: qddot 滑窗超限 + hard_reject 时拒绝。"""
        # 构造历史使 qddot 很大：qdot 从 0 跳到 10 in 1 step
        qdot_hist = np.array([
            [0, 0, 0, 0, 0, 0],
            [10, 0, 0, 0, 0, 0],
        ])  # hist_len=2
        params = _make_check_params(qdd_max=10.0, qdd_hard_reject=True)
        # x_next 的 qdot 也需要匹配历史最后一项
        x_next = np.zeros(12); x_next[6] = 10.0
        x_prev = np.zeros(12); x_prev[6] = 0.0
        feasible, reason = cpp_check_step_raw(
            x_prev, x_next, np.zeros(6),
            qdot_hist.flatten(), 2, params)
        assert not feasible


# ============================================================================
# Phase D: forward_pass + limits 集成
# ============================================================================

@pytest.mark.skipif(not _FP_CPP_AVAILABLE, reason="C++ forward_pass 未更新")
class TestForwardPassWithLimits:
    """D1-D2: C++ forward_pass + limits 检查集成。"""

    def test_forward_pass_feasible_limits(self) -> None:
        """D1: 有 limits + 可行轨迹，结果与无 limits 一致。"""
        env = _make_env()
        X, U, Ks, ks = _make_simple_trajectory(env, N=20)
        ctrl_lo = env.model.actuator_ctrlrange[:6, 0].copy()
        ctrl_hi = env.model.actuator_ctrlrange[:6, 1].copy()
        torque_max = env._torque_ctrlrange[:, 1].copy()
        ball_geom_start = env.model.body("ball").geomadr[0]
        Ks_flat = np.array(Ks).reshape(20, 72)

        # 无 limits
        X_nolim = np.zeros_like(X)
        U_nolim = np.zeros_like(U)
        ok1 = cpp_fp_single_raw(
            X_nolim, U_nolim, X, U, Ks_flat, np.array(ks).reshape(20,6),
            env.model._address, env.data._address, env.init_q_left,
            ctrl_lo, ctrl_hi, 0.5, 0, None, None, False, None,
            ball_geom_start, True)

        # 有 limits（宽松到不会触发）
        params = _make_check_params(
            q_lo=-3*np.ones(6), q_hi=3*np.ones(6),
            qd_max=100*np.ones(6),
            qdd_max=1000*np.ones(6),
        )
        X_lim = np.zeros_like(X)
        U_lim = np.zeros_like(U)
        ok2 = cpp_fp_single_raw(
            X_lim, U_lim, X, U, Ks_flat, np.array(ks).reshape(20,6),
            env.model._address, env.data._address, env.init_q_left,
            ctrl_lo, ctrl_hi, 0.5, 0, None, None, False, None,
            ball_geom_start, True, params)

        assert ok1 and ok2
        np.testing.assert_allclose(X_lim, X_nolim, atol=1e-10)

    def test_forward_pass_infeasible_rejects(self) -> None:
        """D2: qdot 极低限制导致拒绝。"""
        env = _make_env()
        X, U, Ks, ks = _make_simple_trajectory(env, N=20)
        ctrl_lo = env.model.actuator_ctrlrange[:6, 0].copy()
        ctrl_hi = env.model.actuator_ctrlrange[:6, 1].copy()
        torque_max = env._torque_ctrlrange[:, 1].copy()
        ball_geom_start = env.model.body("ball").geomadr[0]
        Ks_flat = np.array(Ks).reshape(20, 72)

        # 极严格速度限制
        params = _make_check_params(qd_max=0.01*np.ones(6))
        X_new = np.zeros_like(X)
        U_new = np.zeros_like(U)
        contype_before = env.model.geom_contype.copy()
        ok = cpp_fp_single_raw(
            X_new, U_new, X, U, Ks_flat, np.array(ks).reshape(20,6),
            env.model._address, env.data._address, env.init_q_left,
            ctrl_lo, ctrl_hi, 0.5, 0, None, None, False, None,
            ball_geom_start, True, params)
        assert not ok, "极严格限制应被拒绝"
        # 碰撞设置恢复
        np.testing.assert_array_equal(env.model.geom_contype, contype_before)
