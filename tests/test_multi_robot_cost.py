"""多维机器人成本/Tube 导数契约测试（阶段 3 · Task 3.6）。

核心断言组（brief + 施工计划 Task 3.6）：
1. ``test_terminal_*``：n=6（合成链）/ n=7（真实 R1D 双侧）终端位置/速度/
   法向成本的方向导数与中心有限差分对照（rtol=1e-3）；第七 q/qdot 变化
   不能被忽略（lx[6]/lx[n+6] 非零且与 FD 一致）；lx (2n,)、lxx (2n,2n)
   形状与 Gauss-Newton 约定（只验对称/维数/约定，不伪称精确 Hessian）。
2. ``test_running_*``：生产四运行项（ControlEffort/Smoothness/TcpSoft/
   QdotLimit）的 lx (2n,)/lu (n,)/lxx (2n,2n)/luu (n,n)/l_ux (n,2n) 形状
   与数值；平滑点（远离 hinge 转折）一阶导数 FD 对照；精确二次项的
   二阶导数精确对照；交叉项约定。
3. ``test_factory_*``：build_production_cost 维度全部取自 env（不依赖
   cost term 默认 6）；新增 keyword-only cost_config 消费 task YAML
   权重（键名以 MPCConfig/TubeConfig 字段与 3.1 YAML 已有键为准），
   RM65 缺省路径（cost_config=None）行为不变。
4. ``test_tube_*`` / ``test_softmin_*``：Tube 走廊 hinge（远离转折处）
   与 softmin 终端聚合的一阶导数 FD 对照（rtol=1e-3/atol=1e-5 缩放）；
   Gauss-Newton 二阶只验对称/维数/约定。
5. ``test_fkcontext_*``：FKContext n 维缓冲与"环境状态变化失效"缓存
   语义（相同 x 跳过、新 x 刷新）。

环境构造（3.3 交付 BoundRobotEnv）：
- n=7 真实 R1D 双侧裸模型（load_robot_model + RobotModelBundle）；
- n=6 自包含合成铰链链（本文件私有 XML，经 load_mujoco_model +
  bind_robot_model 组装，不走 RM65Env）。
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from src.ilqt.cost import CompositeCost, FKContext, build_production_cost
from src.ilqt.cost_terms import (
    ControlEffortTerm,
    QdotLimitTerm,
    SmoothnessTerm,
    TcpSoftTerm,
    TerminalHitTerm,
)
from src.ilqt.mpc_controller import MPCConfig
from src.ilqt.robot_limits import RobotLimits
from src.ilqt.tube_cost import TubeHittingCostWrapper
from src.ilqt.tube_types import HittingTube, TubeConfig
from src.robot.model_binding import bind_robot_model
from src.robot.model_factory import RobotModelBundle, load_robot_model
from src.robot.robot_profile import RobotProfile, load_robot_profile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    load_runtime_parameters,
)
from src.sim.bound_robot_env import BoundRobotEnv
from src.utils.mujoco_loader import load_mujoco_model

# 仓库根目录（tests/ 的父目录）
REPO_ROOT = Path(__file__).resolve().parent.parent
# 固定随机种子（失败可复现）
_RNG_SEED = 202611163
# FD 方向导数默认步长（rad；中心差分）
_FD_EPS = 1e-6


# ---------- 用例载体 ----------


@dataclass
class CostCase:
    """单个测试用例的环境与姿态元数据。

    Attributes:
        name: 用例名（synthetic6 / r1d_left / r1d_right）。
        env: BoundRobotEnv 实例（n=6 或 7）。
        x: 测试姿态状态 [q, qdot]（雅可比非退化的固定姿态）。
    """

    name: str
    env: BoundRobotEnv
    x: np.ndarray

    @property
    def n(self) -> int:
        """主动关节自由度数。"""
        return self.env.NQ


# ---------- n=7 真实 R1D 用例（双侧裸模型） ----------


@lru_cache(maxsize=4)
def _r1d_profile(side: str) -> RobotProfile:
    """加载并缓存该侧 RobotProfile。"""
    return load_robot_profile(REPO_ROOT / "configs" / "robots" / f"r1d_{side}.yaml")


@lru_cache(maxsize=4)
def _r1d_parameters(side: str) -> RobotRuntimeParameters:
    """加载该侧仿真 runtime 参数（含有限 torque_max）。"""
    return load_runtime_parameters(
        REPO_ROOT / "configs" / "robots" / f"r1d_sim_runtime_{side}.yaml",
        _r1d_profile(side),
    )


# n=7 测试姿态：肘部弯曲 + 腕部三轴偏转（init_q 基础上扰动），
# 保证 J_p 全列非零（实测 min 列和 ≈ 0.076）且法向量远离坐标轴。
_R1D_Q = np.array([0.15, 0.45, -0.25, -0.95, 0.3, 0.5, 0.4])
_R1D_QDOT = np.array([0.4, -0.3, 0.6, 0.2, -0.5, 0.3, 0.7])


@lru_cache(maxsize=2)
def _r1d_case(side: str) -> CostCase:
    """构建真实 R1D 单侧用例（n=7）。"""
    bundle = load_robot_model(_r1d_profile(side), _r1d_parameters(side))
    env = BoundRobotEnv(bundle)
    x = np.concatenate([_R1D_Q, _R1D_QDOT])
    return CostCase(name=f"r1d_{side}", env=env, x=x)


# ---------- n=6 合成铰链链用例（自包含，不经 RM65Env） ----------

# 合成临时目录（模块级共享；进程退出时清理）
_SYNTH_DIR = Path(tempfile.mkdtemp(prefix="multi_robot_cost_synth_"))
_SYNTH_TCP_SITE = "tcp_site"


def _build_synth6_xml() -> str:
    """生成 n=6 合成铰链链 XML（各关节轴交错，保证雅可比列满秩）。"""
    # 由内向外折叠：j6 最内层（带 TCP site），j1 最外层挂在世界
    axes = ["0 1 0", "1 0 0", "0 1 0", "0 0 1", "1 0 0", "0 1 0"]
    chain = f'        <site name="{_SYNTH_TCP_SITE}" pos="0 0 -0.05" size="0.005" />\n'
    for i in reversed(range(6)):
        pos = "0 0 0.6" if i == 0 else "0 0 -0.12"
        chain = (
            f'      <body name="link{i + 1}" pos="{pos}">\n'
            f'        <joint name="j{i + 1}" type="hinge" axis="{axes[i]}"'
            ' range="-3 3" />\n'
            f'        <geom name="geom{i + 1}" type="capsule"'
            ' fromto="0 0 0 0 0 -0.12" size="0.015" mass="0.5"'
            ' contype="0" conaffinity="0" />\n'
            + chain
            + "      </body>\n"
        )
    actuators = "\n".join(
        f'    <motor name="m{j}" joint="j{j}" gear="1" />' for j in range(1, 7)
    )
    return (
        '<mujoco model="synthetic_cost6">\n'
        '  <compiler angle="radian" />\n'
        '  <option timestep="0.005" gravity="0 0 -9.81"'
        ' integrator="implicitfast" />\n'
        "  <worldbody>\n"
        f"{chain}"
        "  </worldbody>\n"
        "  <actuator>\n"
        f"{actuators}\n"
        "  </actuator>\n"
        "</mujoco>\n"
    )


@lru_cache(maxsize=1)
def _synth6_case() -> CostCase:
    """构建并缓存 n=6 合成链用例。"""
    model_path = _SYNTH_DIR / "synthetic_cost6.xml"
    model_path.write_text(_build_synth6_xml(), encoding="utf-8")
    joints = tuple(f"j{i}" for i in range(1, 7))
    profile = RobotProfile(
        schema_version=1,
        robot_id="synthetic_cost6",
        model_path=model_path,
        active_joint_names=joints,
        actuator_names=tuple(f"m{i}" for i in range(1, 7)),
        held_joint_names=(),
        tcp_site_name=_SYNTH_TCP_SITE,
    )
    parameters = RobotRuntimeParameters(
        robot_id="synthetic_cost6",
        joint_names=joints,
        dt=0.005,
        init_q=np.zeros(6),
        held_q=np.zeros(0),  # 无保持关节 → 空数组（None 会被规范化成长度 1）
        kp=None,
        kd=None,
        q_lower=None,
        q_upper=None,
        qdot_max=None,
        qddot_max=None,
        torque_max=np.full(6, 20.0),
        max_tcp_speed=None,
        scene=None,
        provisional_fields=(),
        source="synthetic-test",
    )
    model = load_mujoco_model(model_path)
    bundle = RobotModelBundle(
        profile=profile,
        parameters=parameters,
        model=model,
        binding=bind_robot_model(profile, model),
    )
    env = BoundRobotEnv(bundle)
    q = np.array([0.2, -0.35, 0.5, -0.4, 0.3, 0.6])
    qdot = np.array([0.1, 0.4, -0.2, 0.5, -0.3, 0.2])
    return CostCase(name="synthetic6", env=env, x=np.concatenate([q, qdot]))


def _cleanup_synth_dir() -> None:
    """进程退出时清理合成模型临时目录。"""
    shutil.rmtree(_SYNTH_DIR, ignore_errors=True)


atexit.register(_cleanup_synth_dir)


def _case(name: str) -> CostCase:
    """按名取用例（合成 n=6 或真实 R1D n=7）。"""
    if name == "synthetic6":
        return _synth6_case()
    side = name.removeprefix("r1d_")
    return _r1d_case(side)


_ALL_CASES = ("synthetic6", "r1d_left", "r1d_right")


# ---------- 通用测试辅助 ----------


def _random_direction(dim: int, seed: int) -> np.ndarray:
    """生成固定种子的单位随机方向 (dim,)。"""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(dim)
    return d / np.linalg.norm(d)


def _fd_directional(
    fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    d: np.ndarray,
    eps: float = _FD_EPS,
) -> float:
    """中心有限差分方向导数 (f(x+εd) - f(x-εd)) / 2ε。"""
    return (fn(x + eps * d) - fn(x - eps * d)) / (2.0 * eps)


def _fd_gradient(
    fn: Callable[[np.ndarray], float], x: np.ndarray, eps: float = _FD_EPS
) -> np.ndarray:
    """中心有限差分全梯度（逐坐标）。"""
    grad = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = eps
        grad[i] = (fn(x + e) - fn(x - e)) / (2.0 * eps)
    return grad


def _assert_directional_close(analytic: float, fd: float, label: str) -> None:
    """按 brief 口径断言方向导数一致（rtol=1e-3，atol 按量级缩放 1e-5）。"""
    tol = 1e-3 * abs(fd) + 1e-5 * max(1.0, abs(fd))
    assert abs(analytic - fd) <= tol, (
        f"{label}: analytic={analytic:.6e} fd={fd:.6e} tol={tol:.3e}"
    )


def _terminal_composite(
    case: CostCase,
    q_p: bool = True,
    q_v: bool = True,
    q_n: float = 0.0,
) -> CompositeCost:
    """构建仅含 TerminalHitTerm 的组合代价（终端导数隔离测试）。

    Gauss-Newton 约定下三项导数的精确子空间不同：位置/法向对 q 精确、
    速度对 qdot 精确；速度项经 J_p(q) 对 q 的耦合被 GN 显式丢弃（RM65
    同一约定）。FD 对照按项隔离权重，混合权重仅验约定（见
    ``test_terminal_derivatives_shapes_and_convention``）。

    Args:
        case: 测试用例。
        q_p: 是否启用位置项（Q_p 对角 [500,400,300]）。
        q_v: 是否启用速度项（Q_v 对角 [50,40,30]）。
        q_n: 法向项权重（0=禁用）。
    """
    env = case.env
    env.set_arm_state(case.x)
    p_ee = env.get_ee_pos().copy()
    v_ee = env.get_ee_vel().copy()
    p_hit = p_ee + np.array([0.05, -0.03, 0.02])
    v_hit = v_ee + np.array([0.2, -0.4, 0.1])
    n_des = np.array([0.3, 0.5, 0.8])
    n_des = n_des / np.linalg.norm(n_des)
    term = TerminalHitTerm(
        p_hit=p_hit,
        v_hit=v_hit,
        Q_p=np.diag([500.0, 400.0, 300.0]) if q_p else np.zeros((3, 3)),
        Q_v=np.diag([50.0, 40.0, 30.0]) if q_v else np.zeros((3, 3)),
        Q_n=q_n,
        n_des=n_des if q_n > 0 else None,
        NX=env.NX,
        NQ=env.NQ,
    )
    return CompositeCost(env, running_terms=[], terminal_terms=[term])


def _make_limits(n: int, max_tcp_speed: float) -> RobotLimits:
    """构造 n 维试验限位（阈值来源，不做执行约束）。"""
    return RobotLimits(
        q_lower=np.full(n, -3.0),
        q_upper=np.full(n, 3.0),
        qdot_max=np.full(n, 3.0),
        qddot_max=np.full(n, 50.0),
        u_min=np.full(n, -20.0),
        u_max=np.full(n, 20.0),
        max_tcp_speed=max_tcp_speed,
    )


# ---------- 1. 终端位置/速度/法向方向导数（n=6/7） ----------
#
# Gauss-Newton 约定的精确子空间（RM65 既有约定，n 维化后保持）：
# - 位置项 l = ½‖p_ee(q)-p_hit‖²_Qp：对 q 精确（FK 一阶 = J_p），
#   不依赖 qdot → qdot 块两侧均为零；
# - 速度项 l = ½‖J_p(q)·qdot - v_hit‖²_Qv：对 qdot 精确（线性）。
#   其对 q 的真实依赖（经 J_p(q) 耦合）被 GN 显式丢弃——解析 q 块为
#   零而 FD 非零属约定行为，FD 只在 qdot 子空间对照；
# - 法向项 l = ½Q_n‖n(q)-n_des‖²：对 q 精确（dn = skew(-n)·J_r 解析）。
# FD 对照按项隔离权重；混合权重仅验约定（见下方 convention 测试）。


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_terminal_position_directional_derivative_fd(case_name: str) -> None:
    """终端位置成本（仅 Q_p）全梯度与 FD 逐坐标对照。"""
    case = _case(case_name)
    cost = _terminal_composite(case, q_p=True, q_v=False, q_n=0.0)
    lx, _ = cost.terminal_derivatives(case.x)
    grad_fd = _fd_gradient(lambda x_: cost.terminal_cost(x_), case.x)
    np.testing.assert_allclose(lx, grad_fd, rtol=1e-3, atol=1e-3)
    # 位置项不依赖 qdot → qdot 块恰为零
    np.testing.assert_allclose(lx[case.n :], 0.0, atol=1e-12)


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_terminal_velocity_directional_derivative_fd(case_name: str) -> None:
    """终端速度成本（仅 Q_v）qdot 子空间梯度与 FD 逐坐标对照。"""
    case = _case(case_name)
    cost = _terminal_composite(case, q_p=False, q_v=True, q_n=0.0)
    lx, _ = cost.terminal_derivatives(case.x)
    grad_fd = _fd_gradient(lambda x_: cost.terminal_cost(x_), case.x)
    # qdot 块（v_ee = J_p·qdot 线性 → 解析精确）
    np.testing.assert_allclose(
        lx[case.n :], grad_fd[case.n :], rtol=1e-3, atol=1e-3
    )
    # GN 约定：q 块解析为零（J_p(q) 耦合显式丢弃，FD 非零不属缺陷）
    np.testing.assert_allclose(lx[: case.n], 0.0, atol=1e-12)


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_terminal_normal_directional_derivative_fd(case_name: str) -> None:
    """终端法向成本（仅 Q_n）全梯度与 FD 逐坐标对照（非退化法向姿态）。"""
    case = _case(case_name)
    cost = _terminal_composite(case, q_p=False, q_v=False, q_n=250.0)
    lx, _ = cost.terminal_derivatives(case.x)
    grad_fd = _fd_gradient(lambda x_: cost.terminal_cost(x_), case.x)
    np.testing.assert_allclose(lx, grad_fd, rtol=1e-3, atol=1e-3)
    # 法向不依赖 qdot → qdot 块恰为零
    np.testing.assert_allclose(lx[case.n :], 0.0, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
def test_terminal_seventh_joint_not_ignored(side: str) -> None:
    """第七 q/qdot 变化不能被忽略：lx[6]/lx[13] 非零且与 FD 逐项一致。

    分别在位置项（q 子空间）与速度项（qdot 子空间）的精确子空间内
    对照第七关节分量；混合权重的 GN 约定见 convention 测试。
    """
    case = _r1d_case(side)
    n = case.n
    assert n == 7

    cost_pos = _terminal_composite(case, q_p=True, q_v=False, q_n=0.0)
    lx_pos, _ = cost_pos.terminal_derivatives(case.x)
    assert abs(lx_pos[6]) > 1e-6, "第七关节 q 的终端位置梯度不能恒为零"
    grad_fd = _fd_gradient(lambda x_: cost_pos.terminal_cost(x_), case.x)
    np.testing.assert_allclose(lx_pos, grad_fd, rtol=1e-3, atol=1e-3)

    cost_vel = _terminal_composite(case, q_p=False, q_v=True, q_n=0.0)
    lx_vel, _ = cost_vel.terminal_derivatives(case.x)
    assert abs(lx_vel[n + 6]) > 1e-6, "第七关节 qdot 的终端速度梯度不能恒为零"
    grad_fd_v = _fd_gradient(lambda x_: cost_vel.terminal_cost(x_), case.x)
    # qdot 子空间逐项对照（GN 约定：q 块解析为零，见头部注释）
    np.testing.assert_allclose(
        lx_vel[n:], grad_fd_v[n:], rtol=1e-3, atol=1e-3
    )

    # 第七 q/qdot 扰动确实改变终端代价值（不能被静默吞掉）
    c0 = cost_vel.terminal_cost(case.x)
    x7 = case.x.copy()
    x7[6] += 1e-3
    x7[n + 6] += 1e-3
    assert cost_vel.terminal_cost(x7) != pytest.approx(c0, rel=1e-12)


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_terminal_derivatives_shapes_and_convention(case_name: str) -> None:
    """终端导数形状 (2n,)/(2n,2n)；Gauss-Newton 一/二阶约定逐块对照。

    约定（TerminalHitTerm docstring，RM65 既有算法含义，n 维化不变）：
    - lx[:n] = J_pᵀ Q_p dp + Q_n·J_nᵀ·n_err；lx[n:] = J_pᵀ Q_v dv
      （速度项对 q 的 J_p 耦合被 GN 丢弃）；
    - l_xx 位置块 = J_pᵀ Q_p J_p + Q_n·J_nᵀ J_n、速度块 = J_pᵀ Q_v J_p
      （对角块化，v_ee = J_p·qdot）。
    二阶为 Gauss-Newton 近似：只验对称/维数/约定，不伪称精确 Hessian
    （GN 丢弃残差二阶项）。
    """
    case = _case(case_name)
    n = case.n
    cost = _terminal_composite(case, q_p=True, q_v=True, q_n=250.0)
    lx, lxx = cost.terminal_derivatives(case.x)
    assert lx.shape == (2 * n,)
    assert lxx.shape == (2 * n, 2 * n)
    np.testing.assert_allclose(lxx, lxx.T, atol=1e-9)

    env = case.env
    env.set_arm_state(case.x)
    J_p = env.get_ee_jacp()
    term = cost.terminal_terms[0]
    # 一阶约定（法向贡献仅在 q 块：J_n 前 NQ 列非零）
    np.testing.assert_allclose(
        lx[:n],
        J_p.T @ term.Q_p @ (cost.fk.p_ee - term.p_hit)
        + term.Q_n * (cost.fk.J_n.T @ (cost.fk.n_rack - term.n_des))[:n],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        lx[n:], J_p.T @ term.Q_v @ (cost.fk.v_ee - term.v_hit), rtol=1e-6
    )
    # 二阶约定（Gauss-Newton，只验约定不伪称精确 Hessian）
    np.testing.assert_allclose(
        lxx[:n, :n] - term.Q_n * (cost.fk.J_n.T @ cost.fk.J_n)[:n, :n],
        J_p.T @ term.Q_p @ J_p,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        lxx[n:, n:], J_p.T @ term.Q_v @ J_p, rtol=1e-6
    )
    # GN 约定：法向不依赖 qdot → 交叉块为零
    np.testing.assert_allclose(lxx[:n, n:], 0.0, atol=1e-9)


# ---------- 2. 运行项导数（形状/数值/FD） ----------


def _production_composite(
    case: CostCase,
    *,
    q_tcp_soft: float = 0.0,
    q_qdot_limit: float = 0.0,
) -> CompositeCost:
    """按生产四项组装运行代价（软惩罚可开关以便隔离 FD）。"""
    env = case.env
    n = case.n
    running: list = [
        ControlEffortTerm(R=0.02, actuator_mode=0, NU=env.NU),
        SmoothnessTerm(
            Q_qdot=0.05,
            Q_qddot=0.01,
            Q_du=0.2,
            NQ=env.NQ,
            NX=env.NX,
            NU=env.NU,
            dt=env.dt,
        ),
    ]
    if q_tcp_soft > 0:
        running.append(
            TcpSoftTerm(
                Q_tcp_soft=q_tcp_soft,
                tcp_threshold=0.05,  # 远低于测试姿态 TCP 速度（远离 hinge 转折）
                NQ=env.NQ,
                NX=env.NX,
            )
        )
    if q_qdot_limit > 0:
        running.append(
            QdotLimitTerm(
                Q_qdot_limit=q_qdot_limit,
                qdot_limit_thresholds=np.full(n, 0.05),  # 全部超限（远离转折）
                NQ=env.NQ,
                NX=env.NX,
            )
        )
    return CompositeCost(env, running_terms=running, terminal_terms=[])


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_running_derivatives_shapes(case_name: str) -> None:
    """运行导数形状：lx (2n,)/lu (n,)/lxx (2n,2n)/luu (n,n)/l_ux (n,2n)。"""
    case = _case(case_name)
    n = case.n
    cost = _production_composite(case)
    u = np.full(n, 0.3)
    cost.set_u_prev(np.zeros(n))
    lx, lu, lxx, lux, luu = cost.running_derivatives(case.x, u, k=5)
    assert lx.shape == (2 * n,)
    assert lu.shape == (n,)
    assert lxx.shape == (2 * n, 2 * n)
    assert lux.shape == (n, 2 * n)
    assert luu.shape == (n, n)
    # 约定：当前无运行项贡献控制-状态交叉项 → l_ux 恒零
    np.testing.assert_allclose(lux, 0.0, atol=1e-12)


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_running_gradient_fd_smooth_terms(case_name: str) -> None:
    """平滑运行项（软惩罚关闭）lx/lu 与 FD 对照（精确二次项，一阶导精确）。"""
    case = _case(case_name)
    n = case.n
    cost = _production_composite(case)
    u = np.linspace(0.1, 0.5, n)
    cost.set_u_prev(np.linspace(-0.2, 0.1, n))
    lx, lu, _, _, _ = cost.running_derivatives(case.x, u, k=5)
    dx = _random_direction(2 * n, _RNG_SEED + 3)
    du = _random_direction(n, _RNG_SEED + 4)
    fd_x = _fd_directional(
        lambda x_: cost.running_cost(x_, u, k=5), case.x, dx
    )
    fd_u = _fd_directional(
        lambda u_: cost.running_cost(case.x, u_, k=5), u, du
    )
    _assert_directional_close(float(lx @ dx), fd_x, f"{case_name} 运行 lx")
    _assert_directional_close(float(lu @ du), fd_u, f"{case_name} 运行 lu")


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_running_gradient_fd_hinge_terms_away_from_kink(case_name: str) -> None:
    """hinge 项在远离转折处一阶导数与 FD 对照（qdot 精确子空间）。

    TcpSoft 的 v_tcp = J_p(q)·qdot 对 q 的耦合同终端速度项一样被近似
    丢弃（RM65 既有约定），故 FD 方向限制在 qdot 子空间；q 块按约定为零。
    QdotLimit 纯 qdot 依赖，两个方向均精确。
    """
    case = _case(case_name)
    n = case.n
    cost = _production_composite(case, q_tcp_soft=300.0, q_qdot_limit=150.0)
    u = np.full(n, 0.2)
    lx, lu, _, _, _ = cost.running_derivatives(case.x, u, k=5)
    # qdot 子空间方向（TcpSoft/QdotLimit 的精确子空间）
    dq = _random_direction(n, _RNG_SEED + 5)
    dx = np.concatenate([np.zeros(n), dq])
    fd_x = _fd_directional(
        lambda x_: cost.running_cost(x_, u, k=5), case.x, dx
    )
    _assert_directional_close(float(lx @ dx), fd_x, f"{case.name} hinge lx")
    # 约定：TcpSoft/QdotLimit 均不产生 q 块（Smoothness 也无 q 块）
    np.testing.assert_allclose(lx[:n], 0.0, atol=1e-9)
    # hinge 项不依赖控制：lu 与 FD 一致
    du = _random_direction(n, _RNG_SEED + 6)
    fd_u = _fd_directional(
        lambda u_: cost.running_cost(case.x, u_, k=5), u, du
    )
    _assert_directional_close(float(lu @ du), fd_u, f"{case.name} hinge lu")


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_running_hessian_exact_quadratic_terms(case_name: str) -> None:
    """精确二次项二阶导数逐块对照：ControlEffort l_uu=R·I、Smoothness
    l_xx[n:,n:]=（Q_qdot+Q_qddot/dt²)·I、l_uu=Q_du·I；交叉项为零。"""
    case = _case(case_name)
    n = case.n
    cost = _production_composite(case)
    u = np.full(n, 0.3)
    cost.set_u_prev(np.zeros(n))
    lx, lu, lxx, lux, luu = cost.running_derivatives(case.x, u, k=5)
    np.testing.assert_allclose(luu, 0.02 * np.eye(n) + 0.2 * np.eye(n), rtol=1e-12)
    scale = 0.05 + 0.01 / (case.env.dt**2)
    np.testing.assert_allclose(lxx[n:, n:], scale * np.eye(n), rtol=1e-12)
    np.testing.assert_allclose(lxx[:n, :n], 0.0, atol=1e-12)
    np.testing.assert_allclose(lux, 0.0, atol=1e-12)


# ---------- 3. 工厂维度与 task cost_config 消费 ----------


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_factory_dims_come_from_env(case_name: str) -> None:
    """工厂显式传 NQ/NX/NU：终端/运行导数缓冲维数全部随 env（n=7 非 6）。"""
    case = _case(case_name)
    n = case.n
    env = case.env
    env.set_arm_state(case.x)
    p_hit = env.get_ee_pos() + np.array([0.02, 0.01, -0.01])
    cost = build_production_cost(
        env,
        MPCConfig(),
        _make_limits(n, 10.0),
        p_hit=p_hit,
        v_hit=np.array([0.0, -1.0, 0.0]),
        n_des=np.array([0.0, 0.0, 1.0]),
        Q_p_mat=np.eye(3) * 100.0,
        Q_v_mat=np.eye(3) * 10.0,
    )
    u = np.full(n, 0.1)
    lx, lu, lxx, lux, luu = cost.running_derivatives(case.x, u, k=3)
    tlx, tlxx = cost.terminal_derivatives(case.x)
    assert lx.shape == (2 * n,) and lu.shape == (n,)
    assert lxx.shape == (2 * n, 2 * n) and luu.shape == (n, n)
    assert lux.shape == (n, 2 * n)
    assert tlx.shape == (2 * n,) and tlxx.shape == (2 * n, 2 * n)
    # 各运行项导数缓冲维数与 env 一致（异常时上方求和即广播报错，此为显式断言）
    for t in cost.running_terms:
        d = t.running_derivatives(case.x, u, 3, cost.fk)
        if d.l_x is not None:
            assert d.l_x.shape == (2 * n,)
        if d.l_u is not None:
            assert d.l_u.shape == (n,)


def test_factory_rm65_default_unchanged_without_cost_config() -> None:
    """cost_config=None（缺省）：权重全部取自 config，RM65 数值路径不变。"""
    case = _synth6_case()
    env = case.env
    env.set_arm_state(case.x)
    config = MPCConfig(
        R=0.03, Q_qdot_base=0.02, Q_qddot_base=0.01,
        Q_du_base=0.1, Q_tcp_soft=0.0, Q_qdot_limit=0.0,
        normal_weight=123.0,
    )
    cost = build_production_cost(
        env,
        config,
        _make_limits(6, 10.0),
        p_hit=env.get_ee_pos() + np.array([0.05, 0.0, 0.0]),
        v_hit=np.zeros(3),
        n_des=np.array([0.0, 0.0, 1.0]),
        Q_p_mat=np.eye(3) * 7.0,
        Q_v_mat=np.eye(3) * 3.0,
    )
    term = cost.terminal_terms[0]
    np.testing.assert_allclose(term.Q_p, 7.0 * np.eye(3))
    np.testing.assert_allclose(term.Q_v, 3.0 * np.eye(3))
    assert term.Q_n == 123.0
    ctrl = cost.running_terms[0]
    u = np.full(6, 0.5)
    d = ctrl.running_derivatives(case.x, u, 0, cost.fk)
    np.testing.assert_allclose(d.l_u, 0.03 * u)
    np.testing.assert_allclose(d.l_uu, 0.03 * np.eye(6))


def test_factory_cost_config_overrides_weights() -> None:
    """cost_config 以 MPCConfig 字段名覆盖机器人专属权重（task YAML 口径）。"""
    case = _synth6_case()
    env = case.env
    env.set_arm_state(case.x)
    config = MPCConfig()  # RM65 缺省值
    cost = build_production_cost(
        env,
        config,
        _make_limits(6, 10.0),
        p_hit=env.get_ee_pos(),
        v_hit=np.zeros(3),
        n_des=None,
        Q_p_mat=None,
        Q_v_mat=None,
        cost_config={
            "Q_p": 2000.0,            # 3.1 YAML cost_config 终端位置基准（标量）
            "Q_v": np.array([9.0, 8.0, 7.0]),  # (3,) 对角
            "normal_weight": 2.0e5,   # MPCConfig 字段名
            "R": 0.5,
            "Q_qdot_base": 0.1,
            "Q_du_base": 0.2,
            "tube": {"softmin_beta": 5.0, "Q_p_tube": 50000.0},  # TubeConfig 字段名
        },
    )
    term = cost.terminal_terms[0]
    np.testing.assert_allclose(term.Q_p, 2000.0 * np.eye(3))
    np.testing.assert_allclose(term.Q_v, np.diag([9.0, 8.0, 7.0]))
    assert term.Q_n == 2.0e5
    ctrl = cost.running_terms[0]
    u = np.full(6, 0.5)
    d = ctrl.running_derivatives(case.x, u, 0, cost.fk)
    np.testing.assert_allclose(d.l_u, 0.5 * u)


def test_factory_cost_config_rejects_unknown_keys() -> None:
    """cost_config 未知键（含杜撰键/不支持的别名）快速失败，不静默忽略。"""
    case = _synth6_case()
    env = case.env
    env.set_arm_state(case.x)
    with pytest.raises(ValueError, match="Q_p_base"):
        # Q_p_base 是 replan_core 调度口径，不在此消费（避免与 Q_p_mat 流歧义）
        build_production_cost(
            env, MPCConfig(), _make_limits(6, 10.0),
            p_hit=env.get_ee_pos(), v_hit=np.zeros(3), n_des=None,
            Q_p_mat=np.eye(3), Q_v_mat=np.eye(3),
            cost_config={"Q_p_base": 100.0},
        )
    with pytest.raises(ValueError, match="terminal_position_weight"):
        build_production_cost(
            env, MPCConfig(), _make_limits(6, 10.0),
            p_hit=env.get_ee_pos(), v_hit=np.zeros(3), n_des=None,
            Q_p_mat=np.eye(3), Q_v_mat=np.eye(3),
            cost_config={"terminal_position_weight": 100.0},  # 杜撰键
        )
    with pytest.raises(ValueError, match="tube.*Q_p_tub"):
        build_production_cost(
            env, MPCConfig(), _make_limits(6, 10.0),
            p_hit=env.get_ee_pos(), v_hit=np.zeros(3), n_des=None,
            Q_p_mat=np.eye(3), Q_v_mat=np.eye(3),
            cost_config={"tube": {"Q_p_tub": 1.0}},  # TubeConfig 无此字段（拼写错）
        )


def test_factory_requires_terminal_weights_either_way() -> None:
    """Q_p_mat=None 且 cost_config 无 Q_p/Q_v 时拒绝构造（不静默零权重）。"""
    case = _synth6_case()
    env = case.env
    env.set_arm_state(case.x)
    with pytest.raises(ValueError, match="Q_p"):
        build_production_cost(
            env, MPCConfig(), _make_limits(6, 10.0),
            p_hit=env.get_ee_pos(), v_hit=np.zeros(3), n_des=None,
            Q_p_mat=None, Q_v_mat=np.eye(3),
        )
    with pytest.raises(ValueError, match="Q_v"):
        build_production_cost(
            env, MPCConfig(), _make_limits(6, 10.0),
            p_hit=env.get_ee_pos(), v_hit=np.zeros(3), n_des=None,
            Q_p_mat=np.eye(3), Q_v_mat=None,
        )


def test_cost_term_dimension_params_are_keyword_only() -> None:
    """NQ/NX/NU 为 keyword-only 参数（RM65 默认值 6/12/6 不变）。

    防止调用方按位置把 7 传进 n_des 之后的槽位——维度必须显式命名传递。
    """
    p = np.zeros(3)
    with pytest.raises(TypeError):
        TerminalHitTerm(p, p, np.eye(3), np.eye(3), 0.0, None, 14, 7)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ControlEffortTerm(1.0, None, None, 0, 7)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        SmoothnessTerm(1.0, 1.0, 1.0, 7, 14, 7, 0.005)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        TcpSoftTerm(1.0, 1.0, 7, 14)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        QdotLimitTerm(1.0, np.ones(7), 7, 14)  # type: ignore[call-arg]
    # 缺省构造仍为 RM65 维度
    assert TerminalHitTerm(p, p, np.eye(3), np.eye(3))._derivs.l_x.shape == (12,)  # type: ignore[attr-defined]
    assert ControlEffortTerm(R=1.0)._derivs.l_u.shape == (6,)  # type: ignore[attr-defined]


# ---------- 4. Tube 走廊与 softmin 导数（远离转折/退化处） ----------


def _tube_base_cost(case: CostCase, q_v: bool = True, q_n: float = 250.0) -> CompositeCost:
    """构建运行项为空的 base_cost（隔离 Tube 走廊项的 FD）。

    Args:
        case: 测试用例。
        q_v: 是否含速度项（softmin β 聚合的 q 子空间 FD 需置 False 以
            排除 GN 丢弃的 J_p(q) 耦合）。
        q_n: 法向项权重。
    """
    env = case.env
    env.set_arm_state(case.x)
    term = TerminalHitTerm(
        p_hit=env.get_ee_pos() + np.array([0.05, -0.03, 0.02]),
        v_hit=env.get_ee_vel() + np.array([0.2, -0.4, 0.1]),
        Q_p=np.diag([500.0, 400.0, 300.0]),
        Q_v=np.diag([50.0, 40.0, 30.0]) if q_v else np.zeros((3, 3)),
        Q_n=q_n,
        n_des=np.array([0.3, 0.5, 0.8018]),
        NX=env.NX,
        NQ=env.NQ,
    )
    return CompositeCost(env, running_terms=[], terminal_terms=[term])


def _make_tube(case: CostCase, p_ball: np.ndarray, k: int = 12) -> HittingTube:
    """构造单候选 HittingTube（走廊轴向 = 加权球速度方向 d_ball）。"""
    env = case.env
    env.set_arm_state(case.x)
    return HittingTube(
        k_candidates=np.array([k]),
        p_racket_des=p_ball.reshape(1, 3),
        v_racket_des=np.array([[0.0, -2.0, 0.0]]),
        n_racket_des=np.array([[0.0, 0.4, 0.9165]]),
        p_ball=p_ball.reshape(1, 3),
        v_ball=np.array([[0.2, 2.0, 0.5]]),  # 加权平均方向 → d_ball
        weights=np.array([1.0]),
        best_k=k,
    )


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_tube_corridor_gradient_fd_outside(case_name: str) -> None:
    """走廊外（margin≫0，远离 hinge 转折）一阶导数与 FD 对照 rtol=1e-3。"""
    case = _case(case_name)
    env = case.env
    env.set_arm_state(case.x)
    p_ee = env.get_ee_pos().copy()
    # 参考点沿近似垂直于 d_ball 的方向偏移 0.35m → margin = 0.35-0.12 = 0.23
    perp = np.array([0.9, -0.09, -0.04])  # 与 d_ball≈[0.097,0.970,0.242] 近似垂直
    tube = _make_tube(case, p_ee + 0.35 * perp)
    wrapper = TubeHittingCostWrapper(
        env, _tube_base_cost(case), tube, horizon=40,
        config=TubeConfig(Q_p_tube=800.0, corridor_radius=0.12),
    )
    u = np.zeros(case.n)
    lx, _, _, _, _ = wrapper.running_derivatives(case.x, u, k=12)
    # margin 为正（远离转折）的前置校验
    assert wrapper._last_tube_margins[12] > 0.15  # type: ignore[attr-defined]
    dx = _random_direction(2 * case.n, _RNG_SEED + 7)
    fd = _fd_directional(
        lambda x_: wrapper.running_cost(x_, u, k=12), case.x, dx
    )
    _assert_directional_close(float(lx @ dx), fd, f"{case_name} 走廊方向导数")


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_tube_corridor_zero_inside_and_shape_convention(case_name: str) -> None:
    """走廊内（perp_dist < 半宽，远离转折）零代价零梯度；二阶仅对称/维数。"""
    case = _case(case_name)
    env = case.env
    env.set_arm_state(case.x)
    p_ee = env.get_ee_pos().copy()
    perp = np.array([0.9, -0.09, -0.04])
    tube = _make_tube(case, p_ee + 0.05 * perp)  # perp_dist≈0.05 < 0.12
    wrapper = TubeHittingCostWrapper(
        env, _tube_base_cost(case), tube, horizon=40,
        config=TubeConfig(Q_p_tube=800.0),
    )
    u = np.zeros(case.n)
    k_in = 12
    assert wrapper.running_cost(case.x, u, k=k_in) == pytest.approx(0.0, abs=1e-12)
    lx, _, lxx, _, _ = wrapper.running_derivatives(case.x, u, k=k_in)
    np.testing.assert_allclose(lx, 0.0, atol=1e-12)
    np.testing.assert_allclose(lxx, 0.0, atol=1e-12)
    # 窗口外步不叠加走廊项
    assert wrapper.running_cost(case.x, u, k=3) == pytest.approx(0.0, abs=1e-12)
    # 走廊外形状/对称（Gauss-Newton 约定，不伪称精确 Hessian）
    tube_out = _make_tube(case, p_ee + 0.35 * perp)
    wrapper_out = TubeHittingCostWrapper(
        env, _tube_base_cost(case), tube_out, horizon=40,
        config=TubeConfig(Q_p_tube=800.0),
    )
    lx2, _, lxx2, _, _ = wrapper_out.running_derivatives(case.x, u, k=k_in)
    assert lx2.shape == (2 * case.n,)
    assert lxx2.shape == (2 * case.n, 2 * case.n)
    np.testing.assert_allclose(lxx2, lxx2.T, atol=1e-9)
    # 走廊不惩罚沿轴向超前/滞后：qdot 块为零（位置 hinge 仅依赖 q）
    np.testing.assert_allclose(lx2[case.n :], 0.0, atol=1e-12)


@pytest.mark.parametrize("side", ["left", "right"])
def test_tube_seventh_joint_sensitivity(side: str) -> None:
    """走廊梯度第七关节分量非零且与 FD 一致（第七 q 变化不能被忽略）。"""
    case = _r1d_case(side)
    env = case.env
    env.set_arm_state(case.x)
    p_ee = env.get_ee_pos().copy()
    perp = np.array([0.9, -0.09, -0.04])
    tube = _make_tube(case, p_ee + 0.35 * perp)
    wrapper = TubeHittingCostWrapper(
        env, _tube_base_cost(case), tube, horizon=40,
        config=TubeConfig(Q_p_tube=800.0),
    )
    u = np.zeros(case.n)
    lx, _, _, _, _ = wrapper.running_derivatives(case.x, u, k=12)
    assert abs(lx[6]) > 1e-9, "走廊梯度第七关节分量不能恒为零"
    e7 = np.zeros(2 * case.n)
    e7[6] = 1.0
    fd = _fd_directional(
        lambda x_: wrapper.running_cost(x_, u, k=12), case.x, e7
    )
    _assert_directional_close(float(lx[6]), fd, "走廊第七关节 FD")


def _multi_candidate_tube(case: CostCase) -> HittingTube:
    """构造 M=3 候选 tube（softmin 用）。

    候选围绕该 case 实际 FK 状态生成（位置偏移 1-2.5cm、速度偏移小量、
    法向 = 实际 n_rack 加小扰动后单位化），保证 β=1 下 softmin 权重
    α 非退化（既非单点独占也非对称均匀），且法向候选互相分离非退化。
    """
    env = case.env
    env.set_arm_state(case.x)
    p_ee = env.get_ee_pos().copy()
    v_ee = env.get_ee_vel().copy()
    n_rack = env.get_ee_normal().copy()
    p_ball = p_ee + np.array(
        [[0.0, 0.0, 0.0], [0.02, 0.0, 0.01], [-0.01, 0.015, -0.008]]
    )
    v_des = v_ee + np.array(
        [[0.0, -0.05, 0.0], [0.02, -0.08, 0.01], [-0.01, -0.02, -0.005]]
    )
    n_raw = n_rack + np.array(
        [[0.0, 0.0, 0.0], [0.08, -0.05, 0.04], [-0.05, 0.06, -0.03]]
    )
    n_des = n_raw / np.linalg.norm(n_raw, axis=1, keepdims=True)
    v_ball = np.tile(np.array([0.2, 2.0, 0.5]), (3, 1))
    return HittingTube(
        k_candidates=np.array([11, 12, 13]),
        p_racket_des=p_ball,
        v_racket_des=v_des,
        n_racket_des=n_des,
        p_ball=p_ball,
        v_ball=v_ball,
        weights=np.array([0.3, 0.4, 0.3]),
        best_k=12,
    )


# softmin 测试统一锐度（候选密近 + β=1 → α 非退化非单点）
_SOFTMIN_BETA = 1.0


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_softmin_terminal_gradient_fd(case_name: str) -> None:
    """softmin 终端一阶导数与 FD 对照（q/qdot 两个精确子空间，rtol=1e-3）。

    q 子空间：base 置 Q_v=0（速度项经 J_p(q) 的耦合被 GN 丢弃），
    softmin 加权梯度 Σ α_i ∂c_i/∂x 在该子空间精确；
    qdot 子空间：全权重（v_ee = J_p·qdot 线性，softmin 权重恒等式保证精确）。
    """
    case = _case(case_name)
    env = case.env
    tube = _multi_candidate_tube(case)

    # q 子空间（Q_v=0 隔离）
    wrapper_q = TubeHittingCostWrapper(
        env, _tube_base_cost(case, q_v=False), tube, horizon=40,
        config=TubeConfig(use_softmin_terminal=True, softmin_beta=_SOFTMIN_BETA),
    )
    lx_q, _ = wrapper_q.terminal_derivatives(case.x)
    alphas = wrapper_q._last_softmin_alphas  # type: ignore[attr-defined]
    assert 1e-3 < alphas.max() < 1.0 - 1e-3, f"softmin α 退化: {alphas}"
    dq = _random_direction(case.n, _RNG_SEED + 8)
    dx_q = np.concatenate([dq, np.zeros(case.n)])
    fd_q = _fd_directional(lambda x_: wrapper_q.terminal_cost(x_), case.x, dx_q)
    _assert_directional_close(
        float(lx_q @ dx_q), fd_q, f"{case_name} softmin q 子空间方向导数"
    )
    np.testing.assert_allclose(lx_q[case.n :], 0.0, atol=1e-12)

    # qdot 子空间（全权重）
    wrapper_v = TubeHittingCostWrapper(
        env, _tube_base_cost(case, q_v=True), tube, horizon=40,
        config=TubeConfig(use_softmin_terminal=True, softmin_beta=_SOFTMIN_BETA),
    )
    lx_v, lxx_v = wrapper_v.terminal_derivatives(case.x)
    dv = _random_direction(case.n, _RNG_SEED + 9)
    dx_v = np.concatenate([np.zeros(case.n), dv])
    fd_v = _fd_directional(lambda x_: wrapper_v.terminal_cost(x_), case.x, dx_v)
    _assert_directional_close(
        float(lx_v @ dx_v), fd_v, f"{case_name} softmin qdot 子空间方向导数"
    )
    # 形状/对称（Gauss-Newton 加权近似，只验约定不伪称精确 Hessian）
    assert lx_v.shape == (2 * case.n,)
    assert lxx_v.shape == (2 * case.n, 2 * case.n)
    np.testing.assert_allclose(lxx_v, lxx_v.T, atol=1e-9)


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_softmin_terminal_value_matches_formula(case_name: str) -> None:
    """softmin 终端值 = -log(Σ w_i exp(-βc_i))/β（与逐候选公式独立重算对照）。"""
    case = _case(case_name)
    env = case.env
    tube = _multi_candidate_tube(case)
    base = _tube_base_cost(case)
    wrapper = TubeHittingCostWrapper(
        env, base, tube, horizon=40,
        config=TubeConfig(use_softmin_terminal=True, softmin_beta=_SOFTMIN_BETA),
    )
    env.set_arm_state(case.x)
    p_ee, v_ee, n_rack = env.get_ee_pos(), env.get_ee_vel(), env.get_ee_normal()
    Q_p, Q_v, Q_n = base.Q_p, base.Q_v, base.Q_n
    c = np.zeros(3)
    for i in range(3):
        dp = p_ee - tube.p_ball[i]
        dv = v_ee - tube.v_racket_des[i]
        ne = n_rack - tube.n_racket_des[i]
        c[i] = (
            0.5 * float(dp @ Q_p @ dp)
            + 0.5 * float(dv @ Q_v @ dv)
            + 0.5 * Q_n * float(ne @ ne)
        )
    beta = _SOFTMIN_BETA
    expected = -np.log(np.sum(tube.weights * np.exp(-beta * c))) / beta
    assert wrapper.terminal_cost(case.x) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("side", ["left", "right"])
def test_softmin_seventh_joint_sensitivity(side: str) -> None:
    """softmin 终端梯度第七 q/qdot 分量非零且与 FD 一致（精确子空间）。"""
    case = _r1d_case(side)
    env = case.env
    n = case.n
    tube = _multi_candidate_tube(case)
    wrapper_q = TubeHittingCostWrapper(
        env, _tube_base_cost(case, q_v=False), tube, horizon=40,
        config=TubeConfig(use_softmin_terminal=True, softmin_beta=_SOFTMIN_BETA),
    )
    lx_q, _ = wrapper_q.terminal_derivatives(case.x)
    assert abs(lx_q[6]) > 1e-9, "softmin 梯度第七关节 q 分量不能恒为零"
    e7 = np.zeros(2 * n)
    e7[6] = 1.0
    fd_q = _fd_directional(lambda x_: wrapper_q.terminal_cost(x_), case.x, e7)
    _assert_directional_close(float(lx_q[6]), fd_q, "softmin 第七 q FD")

    wrapper_v = TubeHittingCostWrapper(
        env, _tube_base_cost(case, q_v=True), tube, horizon=40,
        config=TubeConfig(use_softmin_terminal=True, softmin_beta=_SOFTMIN_BETA),
    )
    lx_v, _ = wrapper_v.terminal_derivatives(case.x)
    assert abs(lx_v[n + 6]) > 1e-9, "softmin 梯度第七关节 qdot 分量不能恒为零"
    e14 = np.zeros(2 * n)
    e14[n + 6] = 1.0
    fd_v = _fd_directional(lambda x_: wrapper_v.terminal_cost(x_), case.x, e14)
    _assert_directional_close(float(lx_v[n + 6]), fd_v, "softmin 第七 qdot FD")


# ---------- 5. FKContext n 维缓冲与缓存语义 ----------


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_fkcontext_n_dim_buffers_and_cache(case_name: str) -> None:
    """FKContext 缓冲随 env 维数（J_p (3,n)/J_n (3,2n)）；缓存按状态失效。"""
    case = _case(case_name)
    env = case.env
    n = case.n
    fk = FKContext(env)

    class _CountingEnv:
        """包裹 env 统计 set_arm_state 调用次数的替身。"""

        def __init__(self, inner: BoundRobotEnv) -> None:
            self._inner = inner
            self.set_calls = 0

        def __getattr__(self, name: str):  # 透传其余属性/方法
            return getattr(self._inner, name)

        @property
        def NQ(self) -> int:
            return self._inner.NQ

        @property
        def NX(self) -> int:
            return self._inner.NX

        def set_arm_state(self, x: np.ndarray) -> None:
            self.set_calls += 1
            self._inner.set_arm_state(x)

    counted = _CountingEnv(env)
    fk2 = FKContext(counted)

    x1 = case.x.copy()
    x2 = case.x.copy()
    x2[0] += 0.05  # 环境状态变化（不同 x）
    fk2.update(x1)
    calls_after_first = counted.set_calls
    fk2.update(x1)  # 相同 x → 跳过
    assert counted.set_calls == calls_after_first
    fk2.update(x2)  # 新 x → 刷新
    assert counted.set_calls == calls_after_first + 1

    # 缓冲形状与数值一致性
    env.set_arm_state(case.x)
    fk.update(case.x)
    assert fk.J_p.shape == (3, n)
    assert fk.J_r.shape == (3, n)
    assert fk.J_n.shape == (3, 2 * n)
    np.testing.assert_allclose(fk.J_n[:, n:], 0.0, atol=1e-12)
    np.testing.assert_allclose(fk.p_ee, env.get_ee_pos(), atol=1e-12)
    np.testing.assert_allclose(fk.v_ee, env.get_ee_vel(), atol=1e-10)
    # 法向雅可比前 n 列 = skew(-n) @ J_r（解析约定）
    skew = np.array(
        [
            [0.0, -fk.n_rack[2], fk.n_rack[1]],
            [fk.n_rack[2], 0.0, -fk.n_rack[0]],
            [-fk.n_rack[1], fk.n_rack[0], 0.0],
        ]
    )
    np.testing.assert_allclose(fk.J_n[:, :n], -skew @ fk.J_r, atol=1e-10)
