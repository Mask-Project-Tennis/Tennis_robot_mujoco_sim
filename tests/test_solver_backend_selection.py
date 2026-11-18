"""求解器后端选择测试（阶段 3 · Task 3.5）。

核心断言组（brief + 施工计划 §共享接口契约 · 求解器）：
1. ``test_default_*``：``build_solver()`` 无参保持旧结果——C++ 入口可导入
   时优先 C++ 类，不可导入时回退 Python 类，缺省超参数不变；
2. ``test_r1d_*``：R1D（n=7）在 ``auto``/``python`` 后端下均落到
   ``src.ilqt.solver.ILQTSolver`` 且 ``use_analytical=False``（Python FD），
   **即使 C++ 扩展可导入也不调用 C++**（模拟可导入条件 + 内核哨兵）；
   强制 ``cpp`` → 明确 ``ValueError``；
3. ``test_reordered_*``：六维但地址重排的绑定环境同样不能进入旧六维
   内核——``auto`` → Python FD，``cpp`` → ``ValueError``；
4. ``test_compatible_*``：RM65 兼容环境（PlanningEnv/RM65Env 前缀布局）
   ``auto`` 保持旧偏好（可导入 → C++ 类；不可导入 → Python 解析线性化）；
5. ``test_cpp_entry_guard_*``：直接构造 ``src.cpp.solver_cpp.ILQTSolver``
   的 Python 入口对非兼容环境能力拒绝（``ValueError``，内核函数零调用）；
6. 配置合并/非法后端拒绝等工厂行为。

模拟 C++ 可导入的方式：向 ``sys.modules`` 注入伪造 ``src.cpp.iLQR_Core``
（三个内核函数为哨兵——一旦被真实调用即 ``AssertionError``），随后
reload ``src.cpp.solver_cpp`` 使其 ``_CPP_AVAILABLE=True``；测试结束后
移除伪造模块并再次 reload 还原。即使本机已编译真实 C++ 扩展，也先
暂存并替换为伪造模块，确保测试只验证后端选择、不调用真实六维内核。
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from src.ilqt.solver import ILQTSolver as PythonILQTSolver
from src.ilqt.solver import build_solver
from src.ilqt.solver import is_cpp_kernel_compatible
from src.robot.model_binding import bind_robot_model
from src.robot.model_factory import RobotModelBundle
from src.robot.robot_profile import RobotProfile
from src.robot.runtime_parameters import RobotRuntimeParameters
from src.sim.bound_robot_env import BoundRobotEnv
from src.utils.mujoco_loader import load_mujoco_model

# 仓库根目录（tests/ 的父目录）
REPO_ROOT = Path(__file__).resolve().parent.parent
SIDES = ("left", "right")


# ---------- 共享测试件：维度无关正定二次代价 ----------


class QuadraticCost:
    """维度无关正定二次代价（供行为级验证：solver 真跑且不碰 C++）。

    运行代价 l(x,u) = ½(xᵀQx + uᵀRu)，终端代价 l_N(x) = ½xᵀQ_f x。
    满足 RunningCost/TerminalCost 协议（3-arg running + 2 元组终端导数）。
    """

    def __init__(self, Q: np.ndarray, R: np.ndarray, Qf: np.ndarray) -> None:
        """初始化权重矩阵（正定对称）。"""
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.Qf = np.asarray(Qf, dtype=np.float64)

    def running_cost(self, x: np.ndarray, u: np.ndarray, k: int | None = None) -> float:
        """运行代价值。"""
        return 0.5 * (float(x @ self.Q @ x) + float(u @ self.R @ u))

    def running_derivatives(
        self, x: np.ndarray, u: np.ndarray, k: int | None = None
    ) -> tuple[np.ndarray, ...]:
        """运行代价导数 (l_x, l_u, l_xx, l_ux, l_uu)。"""
        n_u = self.R.shape[0]
        return (
            self.Q @ x,
            self.R @ u,
            self.Q,
            np.zeros((n_u, self.Q.shape[0])),
            self.R,
        )

    def terminal_cost(self, x: np.ndarray) -> float:
        """终端代价值。"""
        return 0.5 * float(x @ self.Qf @ x)

    def terminal_derivatives(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """终端代价导数 (l_x, l_xx)。"""
        return self.Qf @ x, self.Qf


def make_quadratic_cost(n: int) -> QuadraticCost:
    """构造 n 维正定二次代价（位置权重 1、速度权重 0.1、控制权重 0.1）。"""
    Q = np.diag(np.concatenate([np.ones(n), 0.1 * np.ones(n)]))
    Qf = 10.0 * Q
    R = 0.1 * np.eye(n)
    return QuadraticCost(Q, R, Qf)


# ---------- 真实 R1D 环境（双侧裸模型经 load_robot_model） ----------


@lru_cache(maxsize=2)
def _r1d_profile(side: str) -> RobotProfile:
    """加载并缓存该侧 RobotProfile。"""
    from src.robot.robot_profile import load_robot_profile

    return load_robot_profile(REPO_ROOT / "configs" / "robots" / f"r1d_{side}.yaml")


@lru_cache(maxsize=2)
def _r1d_env(side: str) -> BoundRobotEnv:
    """构建该侧真实 R1D 裸模型绑定环境（n=7）。"""
    from src.robot.model_factory import load_robot_model
    from src.robot.runtime_parameters import load_runtime_parameters

    profile = _r1d_profile(side)
    params = load_runtime_parameters(
        REPO_ROOT / "configs" / "robots" / f"r1d_sim_runtime_{side}.yaml", profile
    )
    return BoundRobotEnv(load_robot_model(profile, params))


# ---------- 六维但地址重排的合成绑定环境 ----------

# 树序（XML 深度序）与 profile 序不同；执行器 XML 序乱序 → 地址非前缀
_SYNTH_TREE_ORDER = ("j2", "j5", "j1", "j6", "j3", "j4")
_SYNTH_PROFILE_ORDER = ("j1", "j2", "j3", "j4", "j5", "j6")
_SYNTH_ACTUATOR_XML_ORDER = ("m_j4", "m_j1", "m_j6", "m_j2", "m_j5", "m_j3")
_SYNTH_TORQUE_MAX = np.array([30.0, 25.0, 20.0, 15.0, 10.0, 8.0])


def _build_reordered_xml() -> str:
    """生成 n=6 地址重排合成模型 XML（树序乱序 + 执行器 XML 序乱序）。"""

    def _link_body(joint: str, child: str | None) -> str:
        body = f'      <body name="link_{joint}" pos="0 0 -0.08">\n'
        body += (
            f'        <joint name="{joint}" type="hinge" axis="0 1 0"'
            ' range="-170 170" />\n'
            f'        <geom name="geom_{joint}" type="capsule"'
            ' fromto="0 0 0 0 0 -0.08" size="0.012" mass="0.5"'
            ' contype="0" conaffinity="0" />\n'
        )
        body += child or ""
        body += "      </body>\n"
        return body

    innermost = '        <site name="tool_site" pos="0 0 -0.06" size="0.005" />\n'
    chain = innermost
    for joint in reversed(_SYNTH_TREE_ORDER):
        chain = _link_body(joint, chain)

    actuators = "\n".join(
        f'    <motor name="{name}" joint="{name[2:]}" gear="1" />'
        for name in _SYNTH_ACTUATOR_XML_ORDER
    )

    return f"""<mujoco model="reordered_six_dof">
  <compiler angle="degree" />
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" />
  <worldbody>
    <body name="mount" pos="0 0 0.6">
{chain}    </body>
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


@lru_cache(maxsize=1)
def build_reordered_six_dof_env() -> BoundRobotEnv:
    """构建六维地址重排绑定环境（qpos/qvel/执行器地址均非前缀）。

    Returns:
        (env, 前提校验通过的 BoundRobotEnv)；地址重排由调用方断言。
    """
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="backend_synth_"))
    model_path = tmp_dir / "reordered_six_dof.xml"
    model_path.write_text(_build_reordered_xml(), encoding="utf-8")

    profile = RobotProfile(
        schema_version=1,
        robot_id="synthetic_reordered",
        model_path=model_path,
        active_joint_names=_SYNTH_PROFILE_ORDER,
        actuator_names=tuple(f"m_{j}" for j in _SYNTH_PROFILE_ORDER),
        held_joint_names=(),
        tcp_site_name="tool_site",
    )
    parameters = RobotRuntimeParameters(
        robot_id="synthetic_reordered",
        joint_names=_SYNTH_PROFILE_ORDER,
        dt=0.005,
        init_q=np.zeros(6),
        held_q=np.zeros(0),  # 无保持关节：held_q 为必填字段，空数组合法
        kp=None,
        kd=None,
        q_lower=None,
        q_upper=None,
        qdot_max=None,
        qddot_max=None,
        torque_max=_SYNTH_TORQUE_MAX.copy(),
        max_tcp_speed=None,
        scene=None,
        provisional_fields=(),
        source="synthetic-test",
    )
    model = load_mujoco_model(model_path)
    binding = bind_robot_model(profile, model)
    bundle = RobotModelBundle(
        profile=profile, parameters=parameters, model=model, binding=binding
    )
    return BoundRobotEnv(bundle)


# ---------- 模拟"C++ 可导入"条件 ----------


def _make_fake_core_module() -> types.ModuleType:
    """构造伪造 src.cpp.iLQR_Core 模块（内核函数为哨兵）。"""
    fake = types.ModuleType("src.cpp.iLQR_Core")

    def _forbidden(*args: object, **kwargs: object) -> None:
        """内核哨兵：模拟条件下 C++ 内核被真实调用即失败。"""
        raise AssertionError("C++ 内核函数被调用——后端选择未生效")

    fake.linearize_analytical_batch = _forbidden  # type: ignore[attr-defined]
    fake.forward_pass_single = _forbidden  # type: ignore[attr-defined]
    fake.backward_pass = _forbidden  # type: ignore[attr-defined]
    return fake


@contextlib.contextmanager
def cpp_importable() -> Iterator[types.ModuleType]:
    """模拟 C++ 扩展可导入：替换为伪造内核模块并 reload solver_cpp。

    若本机已编译真实扩展，``src.cpp.solver_cpp`` 在导入时就会注册
    ``src.cpp.iLQR_Core``；本上下文先暂存并移除真实模块（含包属性），
    再注入伪造模块，退出时恢复真实模块并 reload。这样无论扩展是否
    编译，测试都只走伪造哨兵、不会调用真实六维内核。

    Yields:
        reload 后的 ``src.cpp.solver_cpp`` 模块（``_CPP_AVAILABLE=True``）。
    """
    import src.cpp as cpp_pkg

    core_name = "src.cpp.iLQR_Core"
    prev_core = sys.modules.pop(core_name, None)
    prev_attr = getattr(cpp_pkg, "iLQR_Core", None)
    if prev_attr is not None:
        delattr(cpp_pkg, "iLQR_Core")

    sys.modules[core_name] = _make_fake_core_module()
    try:
        import src.cpp.solver_cpp as solver_cpp

        importlib.reload(solver_cpp)
        assert solver_cpp._CPP_AVAILABLE is True, "模拟注入后应判定 C++ 可用"
        yield solver_cpp
    finally:
        sys.modules.pop(core_name, None)
        if prev_core is not None:
            sys.modules[core_name] = prev_core
            setattr(cpp_pkg, "iLQR_Core", prev_core)
        import src.cpp.solver_cpp as solver_cpp_restore

        importlib.reload(solver_cpp_restore)


@contextlib.contextmanager
def cpp_unavailable() -> Iterator[None]:
    """模拟 C++ 扩展不可导入：移除真实内核模块并 reload solver_cpp。

    本机编译 C++ 扩展后，``solver_cpp`` 在导入时已置
    ``_CPP_AVAILABLE=True``，回退分支无法直接观测；本上下文暂存并移除
    真实 ``src.cpp.iLQR_Core``（含包属性），reload 后强制走 ImportError
    分支，退出时恢复真实模块并再次 reload。
    """
    import src.cpp as cpp_pkg

    core_name = "src.cpp.iLQR_Core"
    prev_core = sys.modules.pop(core_name, None)
    prev_attr = getattr(cpp_pkg, "iLQR_Core", None)
    if prev_attr is not None:
        delattr(cpp_pkg, "iLQR_Core")
    # None 是 Python 导入系统的哨兵：命中时 import 直接抛 ImportError，
    # 避免 reload 时又从磁盘把真实 .pyd 重新导入。
    sys.modules[core_name] = None

    try:
        import src.cpp.solver_cpp as solver_cpp

        importlib.reload(solver_cpp)
        assert solver_cpp._CPP_AVAILABLE is False, "模拟移除后应判定 C++ 不可用"
        yield
    finally:
        if prev_core is not None:
            sys.modules[core_name] = prev_core
            setattr(cpp_pkg, "iLQR_Core", prev_core)
        else:
            sys.modules.pop(core_name, None)
        import src.cpp.solver_cpp as solver_cpp_restore

        importlib.reload(solver_cpp_restore)


# ---------- 1. build_solver() 无参缺省行为保持 ----------


def test_default_no_args_matches_legacy_entry() -> None:
    """无参 build_solver() 与旧入口语义一致：返回 solver_cpp.ILQTSolver。"""
    import src.cpp.solver_cpp as solver_cpp

    solver = build_solver()
    assert type(solver) is solver_cpp.ILQTSolver
    # 缺省超参数保持
    assert solver.max_iter == 10
    assert solver.horizon == 60
    assert solver.mu_min == 1e-6
    assert solver.alpha_list == [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]


def test_default_no_args_prefers_cpp_when_importable() -> None:
    """模拟可导入时，无参 build_solver() 返回 C++ 加速类（旧优先级保持）。"""
    with cpp_importable() as solver_cpp:
        solver = build_solver()
        # C++ 分支类：独有 _linearize_cpp；绝非纯 Python 类
        assert type(solver) is solver_cpp.ILQTSolver
        assert hasattr(solver, "_linearize_cpp")
        assert not isinstance(solver, PythonILQTSolver)


# ---------- 2. R1D（n=7）：auto/python 均不碰 C++，强制 cpp 拒绝 ----------


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("backend", ["auto", "python"])
def test_r1d_never_enters_cpp_kernel_even_when_importable(
    side: str, backend: str
) -> None:
    """模拟可导入条件下，R1D auto/python 均得到 Python FD 求解器。"""
    env = _r1d_env(side)
    assert (env.NQ, env.NX, env.NU) == (7, 14, 7)
    with cpp_importable():
        solver = build_solver(env=env, backend=backend)  # type: ignore[arg-type]
        assert type(solver) is PythonILQTSolver
        assert solver.use_analytical is False
        # 行为级验证：真跑 1 次 solve_few_iters，内核哨兵未被触发即通过
        cost = make_quadratic_cost(7)
        x0 = env.get_arm_state()
        U0 = np.zeros((4, 7))
        X, U, history, success = solver.solve_few_iters(
            env, cost, x0, U0, max_iter=1
        )
        assert X.shape == (5, 14)
        assert U.shape == (4, 7)
        assert len(history) >= 1
        assert success is True


@pytest.mark.parametrize("side", SIDES)
def test_r1d_python_backend_without_simulation(side: str) -> None:
    """不注入伪造模块时，python 后端同样给出 FD 求解器。"""
    solver = build_solver(env=_r1d_env(side), backend="python")
    assert type(solver) is PythonILQTSolver
    assert solver.use_analytical is False


@pytest.mark.parametrize("side", SIDES)
def test_r1d_forced_cpp_rejected_without_simulation(side: str) -> None:
    """强制 cpp + R1D：不兼容判定先于导入尝试，任何机器上均 ValueError。"""
    with pytest.raises(ValueError, match="C\\+\\+"):
        build_solver(env=_r1d_env(side), backend="cpp")


@pytest.mark.parametrize("side", SIDES)
def test_r1d_forced_cpp_rejected_even_when_importable(side: str) -> None:
    """模拟可导入条件下，强制 cpp + R1D 仍明确 ValueError。"""
    with cpp_importable():
        with pytest.raises(ValueError, match="不兼容"):
            build_solver(env=_r1d_env(side), backend="cpp")


# ---------- 3. 六维但地址重排：同样不能进入旧内核 ----------


def test_reordered_env_addresses_are_non_prefix() -> None:
    """测试前提自检：合成环境的绑定地址确实非前缀。"""
    env = build_reordered_six_dof_env()
    prefix = np.arange(6)
    binding = env.binding
    assert (env.NQ, env.NX, env.NU) == (6, 12, 6)
    assert not np.array_equal(np.asarray(binding.qpos_indices), prefix)
    assert not np.array_equal(np.asarray(binding.actuator_ids), prefix)


def test_reordered_env_is_not_cpp_compatible() -> None:
    """兼容性谓词：六维重排环境 False，RM65 PlanningEnv True，R1D False。"""
    from src.ilqt.planning_env import PlanningEnv

    assert is_cpp_kernel_compatible(build_reordered_six_dof_env()) is False
    assert is_cpp_kernel_compatible(PlanningEnv()) is True
    assert is_cpp_kernel_compatible(_r1d_env("left")) is False


def test_reordered_env_auto_never_cpp_when_importable() -> None:
    """模拟可导入时，六维重排环境 auto → Python FD（不进旧内核）。"""
    env = build_reordered_six_dof_env()
    with cpp_importable():
        solver = build_solver(env=env, backend="auto")
        assert type(solver) is PythonILQTSolver
        assert solver.use_analytical is False


def test_reordered_env_forced_cpp_rejected() -> None:
    """六维重排环境强制 cpp → ValueError。"""
    with cpp_importable():
        with pytest.raises(ValueError, match="不兼容"):
            build_solver(env=build_reordered_six_dof_env(), backend="cpp")


# ---------- 4. RM65 兼容环境：auto 保持旧偏好 ----------


def test_compatible_env_auto_prefers_cpp_when_importable() -> None:
    """模拟可导入时，RM65 兼容环境（PlanningEnv）auto → C++ 类。"""
    from src.ilqt.planning_env import PlanningEnv

    with cpp_importable() as solver_cpp:
        solver = build_solver(env=PlanningEnv(), backend="auto")
        assert type(solver) is solver_cpp.ILQTSolver
        assert hasattr(solver, "_linearize_cpp")


def test_compatible_env_auto_falls_back_to_python_analytical() -> None:
    """C++ 不可导入时，兼容环境 auto → Python 解析线性化。"""
    from src.ilqt.planning_env import PlanningEnv

    with cpp_unavailable():
        solver = build_solver(env=PlanningEnv(), backend="auto")
        assert type(solver) is PythonILQTSolver
        # 旧 Python 缺省：use_analytical=True
        assert solver.use_analytical is True


def test_compatible_env_explicit_cpp_unavailable_raises() -> None:
    """显式 cpp 但扩展不可导入 → ValueError（不静默回退）。"""
    from src.ilqt.planning_env import PlanningEnv

    with cpp_unavailable():
        with pytest.raises(ValueError, match="不可用"):
            build_solver(env=PlanningEnv(), backend="cpp")


# ---------- 5. C++ 求解器 Python 入口能力拒绝 ----------


@pytest.mark.parametrize("side", SIDES)
def test_cpp_entry_guard_rejects_seven_dof(side: str) -> None:
    """直接构造 C++ 类 + R1D 环境：入口 ValueError，内核零调用。"""
    env = _r1d_env(side)
    with cpp_importable() as solver_cpp:
        solver = solver_cpp.ILQTSolver(
            {
                "max_iter": 2,
                "tol": 1e-4,
                "horizon": 4,
                "mu_min": 1e-6,
                "mu_max": 1e10,
                "mu_init": 0.01,
                "delta_0": 1.6,
                "alpha_list": [0.5],
                "lin_eps": 1e-6,
            }
        )
        cost = make_quadratic_cost(7)
        with pytest.raises(ValueError, match="六维"):
            solver.solve_few_iters(
                env, cost, env.get_arm_state(), np.zeros((4, 7)), max_iter=1
            )


def test_cpp_entry_guard_rejects_reordered_six_dof() -> None:
    """直接构造 C++ 类 + 六维重排环境：入口同样 ValueError。"""
    env = build_reordered_six_dof_env()
    with cpp_importable() as solver_cpp:
        solver = solver_cpp.ILQTSolver(
            {
                "max_iter": 2,
                "tol": 1e-4,
                "horizon": 4,
                "mu_min": 1e-6,
                "mu_max": 1e10,
                "mu_init": 0.01,
                "delta_0": 1.6,
                "alpha_list": [0.5],
                "lin_eps": 1e-6,
            }
        )
        cost = make_quadratic_cost(6)
        with pytest.raises(ValueError, match="六维"):
            solver.solve_few_iters(
                env, cost, env.get_arm_state(), np.zeros((4, 6)), max_iter=1
            )


# ---------- 6. 工厂行为：配置合并与非法后端 ----------


def test_config_merge_over_defaults() -> None:
    """config 覆盖指定键，未指定键保持缺省。"""
    solver = build_solver(
        env=_r1d_env("left"),
        backend="python",
        config={"horizon": 11, "max_iter": 2},
    )
    assert solver.horizon == 11
    assert solver.max_iter == 2
    assert solver.mu_min == 1e-6  # 未覆盖键保持缺省
    assert solver.use_analytical is False


def test_invalid_backend_raises() -> None:
    """非法 backend 字符串 → ValueError。"""
    with pytest.raises(ValueError, match="backend"):
        build_solver(env=_r1d_env("left"), backend="gpu")  # type: ignore[arg-type]


def test_legacy_positional_call_still_works() -> None:
    """旧调用形态 build_solver()（零参）在签名扩展后不受影响。"""
    solver = build_solver()
    assert solver.horizon == 60
