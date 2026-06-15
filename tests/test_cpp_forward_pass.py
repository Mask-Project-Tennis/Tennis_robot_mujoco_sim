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
