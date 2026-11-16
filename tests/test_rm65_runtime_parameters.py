"""RM65 旧参数兼容适配器测试（阶段 2 Task 2.3）。

覆盖六个验收点：
  1. test_sim_parameters_match_current_sources — 仿真适配器逐项对齐原来源
     （constants.INIT_Q/INIT_Q_LEFT、config["sim"]/["actuator"]/["robot_limits"]、
     RobotLimits.from_config、模型 actuator ctrlrange）。
  2. test_real_parameters_match_factory — 真机适配器逐项对齐
     build_robot_limits(env, config) 结果与 RealRobotConfig/INIT_Q_REAL。
  3. test_custom_config_overrides_survive — 自定义 dt=0.01、kp=[17]*6、
     kd=[3]*6、max_tcp_speed=0.7 覆盖默认值。
  4. test_no_new_import_side_effects — 导入不创建环境/线程/SDK/viewer。
  5. test_rejects_non_rm65 — 非 RM65 描述/模型被明确拒绝。
  6. test_rejects_asymmetric_torque_bounds — 力矩上下限不对称拒绝折算，
     不取较大值放宽限值。

共同断言：两套初态保持区别（仿真 INIT_Q[1]=1.57 vs 真机 INIT_Q_REAL[1]=1.40）、
只读数组不与来源共享可写内存、scene=None、provisional_fields=()、
产物可通过 validate_runtime_parameters(purpose="kinematics")。
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from src.ilqt.planning_env import PlanningEnv
from src.ilqt.robot_limits import RobotLimits
from src.ilqt.robot_parameters import (
    resolve_rm65_sim_parameters,
    symmetric_torque_max,
)
from src.real.config import RealRobotConfig
from src.real.runner_factory import build_robot_limits, resolve_rm65_real_parameters
from src.robot.constants import INIT_Q, INIT_Q_LEFT, INIT_Q_REAL
from src.robot.robot_profile import RobotProfile, load_robot_profile
from src.robot.runtime_parameters import validate_runtime_parameters
from src.utils.mujoco_loader import load_mujoco_model
from src.utils.yaml_utils import load_config

# 仓库根目录（tests/ 的父目录）
REPO_ROOT = Path(__file__).resolve().parent.parent

# RM65 描述与两份旧参数来源文件
RM65_PROFILE_YAML = REPO_ROOT / "configs" / "robots" / "rm65.yaml"
RM65_MODEL_XML = REPO_ROOT / "src" / "robot" / "rm65_model.xml"
DEFAULT_YAML = REPO_ROOT / "configs" / "default.yaml"
REAL_YAML = REPO_ROOT / "configs" / "real_robot.yaml"

# RM65 右臂六个主动关节名（与 configs/robots/rm65.yaml 一致）
RM65_ACTIVE_JOINTS = tuple(f"r_joint{i}" for i in range(1, 7))

# 仿真入口位置模式 PD 缺省回退（与 scripts/rm65_mpc_v12.py 一致）
_SIM_KP_FALLBACK = [200.0, 200.0, 200.0, 50.0, 50.0, 20.0]
_SIM_KD_FALLBACK = [20.0, 20.0, 5.0, 5.0, 5.0, 2.0]


# ---------- 共享 fixture（模块级复用，避免重复加载 MuJoCo 模型） ----------


@pytest.fixture(scope="module")
def rm65_profile() -> RobotProfile:
    """加载仓库默认 RM65 描述。"""
    return load_robot_profile(RM65_PROFILE_YAML)


@pytest.fixture(scope="module")
def rm65_model() -> mujoco.MjModel:
    """加载 RM65 双臂 MuJoCo 模型（只读使用）。"""
    return load_mujoco_model(RM65_MODEL_XML)


@pytest.fixture(scope="module")
def sim_config() -> dict[str, Any]:
    """加载 configs/default.yaml（仿真入口当前消费的同一份配置）。"""
    return load_config(DEFAULT_YAML)


@pytest.fixture(scope="module")
def planning_env() -> PlanningEnv:
    """默认 RM65 规划环境（真机适配器的 env 参数）。"""
    return PlanningEnv()


@pytest.fixture(scope="module")
def real_config() -> RealRobotConfig:
    """从 configs/real_robot.yaml 加载真机配置（不连接硬件）。"""
    return RealRobotConfig.from_yaml(REAL_YAML)


# ---------- 1. 仿真适配器逐项对齐原来源 ----------


def test_sim_parameters_match_current_sources(
    rm65_profile: RobotProfile,
    rm65_model: mujoco.MjModel,
    sim_config: dict[str, Any],
) -> None:
    """仿真适配器输出与原来源（constants/config/RobotLimits/ctrlrange）逐项一致。"""
    params = resolve_rm65_sim_parameters(rm65_profile, rm65_model, sim_config)
    validate_runtime_parameters(params, rm65_profile, purpose="kinematics")

    # 身份与关节顺序来自 profile（RM65 右臂 6 关节，不截断不广播）
    assert params.robot_id == "rm65"
    assert params.joint_names == rm65_profile.active_joint_names

    # 初态来源 constants：仿真 INIT_Q（J2=1.57，与真机 1.40 保持区别）
    np.testing.assert_allclose(params.init_q, INIT_Q)
    assert params.init_q[1] == pytest.approx(1.57)
    assert params.init_q[1] != pytest.approx(INIT_Q_REAL[1])
    np.testing.assert_allclose(params.held_q, INIT_Q_LEFT)

    # dt 来源 config["sim"]["dt"]
    assert params.dt == pytest.approx(sim_config["sim"]["dt"])
    assert params.dt == pytest.approx(0.005)

    # default.yaml actuator.mode="torque" → 未配置位置控制，kp/kd 整组 None
    assert sim_config["actuator"]["mode"] == "torque"
    assert params.kp is None
    assert params.kd is None

    # 限位复用 RobotLimits.from_config（与当前入口同参复算对照）
    limits = RobotLimits.from_config(
        sim_config.get("robot_limits", {}),
        dt=float(sim_config["sim"]["dt"]),
        ctrlrange=rm65_model.actuator_ctrlrange[:6],
    )
    np.testing.assert_allclose(params.q_lower, limits.q_lower)
    np.testing.assert_allclose(params.q_upper, limits.q_upper)
    np.testing.assert_allclose(params.qdot_max, limits.qdot_max)
    np.testing.assert_allclose(params.qddot_max, limits.qddot_max)
    assert params.max_tcp_speed == pytest.approx(limits.max_tcp_speed)
    assert params.max_tcp_speed == pytest.approx(1.8)

    # 力矩对称折算 = ctrlrange 上限（[-60,60]×2、[-30,30]、[-10,10]×3）
    np.testing.assert_allclose(params.torque_max, limits.u_max)
    np.testing.assert_allclose(params.torque_max, [60.0, 60.0, 30.0, 10.0, 10.0, 10.0])

    # RM65 scene=None（不覆写旧几何）、无临时字段标记
    assert params.scene is None
    assert params.provisional_fields == ()

    # 只读数组且不与 constants 共享内存
    assert params.init_q.flags.writeable is False
    assert params.held_q.flags.writeable is False
    assert not np.shares_memory(params.init_q, INIT_Q)
    assert not np.shares_memory(params.held_q, INIT_Q_LEFT)

    # 位置模式缺省回退：actuator 有 mode=position 但缺 kp/kd → 入口同款回退值；
    # robot_limits 整节缺失 → RobotLimits 默认 TCP=inf（无限制）映射为 None
    fallback_config: dict[str, Any] = {
        "sim": {"dt": 0.005},
        "actuator": {"mode": "position"},
    }
    fallback_params = resolve_rm65_sim_parameters(
        rm65_profile, rm65_model, fallback_config
    )
    np.testing.assert_allclose(fallback_params.kp, _SIM_KP_FALLBACK)
    np.testing.assert_allclose(fallback_params.kd, _SIM_KD_FALLBACK)
    assert fallback_params.max_tcp_speed is None


# ---------- 2. 真机适配器逐项对齐原工厂 ----------


def test_real_parameters_match_factory(
    planning_env: PlanningEnv,
    real_config: RealRobotConfig,
    rm65_profile: RobotProfile,
) -> None:
    """真机适配器输出与 build_robot_limits/RealRobotConfig/INIT_Q_REAL 逐项一致。"""
    params = resolve_rm65_real_parameters(planning_env, real_config)
    validate_runtime_parameters(params, rm65_profile, purpose="kinematics")

    # 对照基准：原工厂构建的 RobotLimits（唯一安全规则来源）
    limits = build_robot_limits(planning_env, real_config)

    assert params.robot_id == "rm65"
    assert params.joint_names == RM65_ACTIVE_JOINTS

    # 初态取 INIT_Q_REAL（J2=1.40，保持与仿真 1.57 的区别）
    np.testing.assert_allclose(params.init_q, INIT_Q_REAL)
    assert params.init_q[1] == pytest.approx(1.40)
    assert INIT_Q[1] == pytest.approx(1.57)
    np.testing.assert_allclose(params.held_q, INIT_Q_LEFT)

    # dt/kp/kd 取传入 config
    assert params.dt == pytest.approx(real_config.dt)
    np.testing.assert_allclose(params.kp, real_config.kp)
    np.testing.assert_allclose(params.kd, real_config.kd)

    # 位置/速度/加速度/TCP 界限取原 build_robot_limits 结果
    np.testing.assert_allclose(params.q_lower, limits.q_lower)
    np.testing.assert_allclose(params.q_upper, limits.q_upper)
    np.testing.assert_allclose(params.qdot_max, limits.qdot_max)
    np.testing.assert_allclose(params.qddot_max, limits.qddot_max)
    assert params.max_tcp_speed == pytest.approx(limits.max_tcp_speed)
    assert params.max_tcp_speed == pytest.approx(real_config.max_tcp_speed)

    # 力矩界限来自规划 u 限（ctrlrange 对称折算），不是控制器 torque_limit
    assert np.array_equal(limits.u_min, -limits.u_max)  # 前提：RM65 ctrlrange 对称
    np.testing.assert_allclose(params.torque_max, limits.u_max)
    np.testing.assert_allclose(
        params.torque_max, planning_env.model.actuator_ctrlrange[:6, 1]
    )
    # config.torque_limit 是控制器固件安全设置（Layer 1），不得混入规划力矩界限
    assert not np.allclose(params.torque_max, np.asarray(real_config.torque_limit))

    assert params.scene is None
    assert params.provisional_fields == ()
    assert params.init_q.flags.writeable is False
    assert not np.shares_memory(params.init_q, INIT_Q_REAL)


# ---------- 3. 自定义配置覆盖 ----------


def test_custom_config_overrides_survive(
    rm65_profile: RobotProfile,
    rm65_model: mujoco.MjModel,
    planning_env: PlanningEnv,
) -> None:
    """自定义 dt=0.01、kp=[17]*6、kd=[3]*6、max_tcp_speed=0.7 覆盖默认值。"""
    # 仿真侧：位置模式 + 自定义 PD + 自定义 dt/TCP 限速
    custom_sim: dict[str, Any] = {
        "sim": {"dt": 0.01},
        "actuator": {"mode": "position", "kp": [17.0] * 6, "kd": [3.0] * 6},
        "robot_limits": {"max_tcp_speed": 0.7},
    }
    sim_params = resolve_rm65_sim_parameters(rm65_profile, rm65_model, custom_sim)
    assert sim_params.dt == pytest.approx(0.01)
    assert sim_params.kp is not None
    assert sim_params.kd is not None
    np.testing.assert_allclose(sim_params.kp, [17.0] * 6)
    np.testing.assert_allclose(sim_params.kd, [3.0] * 6)
    assert sim_params.max_tcp_speed == pytest.approx(0.7)
    # 未覆盖项仍取原来源（初态不受配置影响）
    np.testing.assert_allclose(sim_params.init_q, INIT_Q)

    # 真机侧：自定义 dt/PD/TCP 限速同样生效
    custom_real = RealRobotConfig(
        dt=0.01, kp=[17.0] * 6, kd=[3.0] * 6, max_tcp_speed=0.7
    )
    real_params = resolve_rm65_real_parameters(planning_env, custom_real)
    assert real_params.dt == pytest.approx(0.01)
    assert real_params.kp is not None
    assert real_params.kd is not None
    np.testing.assert_allclose(real_params.kp, [17.0] * 6)
    np.testing.assert_allclose(real_params.kd, [3.0] * 6)
    assert real_params.max_tcp_speed == pytest.approx(0.7)
    np.testing.assert_allclose(real_params.init_q, INIT_Q_REAL)


# ---------- 4. 导入纯度 ----------


def test_no_new_import_side_effects() -> None:
    """导入适配器模块不创建环境/线程/SDK 连接/viewer。"""
    code = (
        "import os, sys, threading\n"
        "sys.path.insert(0, os.environ['STAGE2_REPO_ROOT'])\n"
        # ilqt 层适配器不得拉入 real/sim 层
        "import src.ilqt.robot_parameters\n"
        "banned_ilqt = ('src.real', 'src.sim')\n"
        "bad = [m for m in sys.modules\n"
        "       if any(m == b or m.startswith(b + '.') for b in banned_ilqt)]\n"
        "assert not bad, f'ilqt 适配器拉入越层模块: {bad}'\n"
        # real 层工厂：不连 SDK、不开 viewer、不起线程
        "import src.real.runner_factory\n"
        "assert 'rmcl' not in sys.modules, 'SDK 在导入期被拉入'\n"
        "assert 'mujoco.viewer' not in sys.modules, 'viewer 在导入期被拉入'\n"
        "count = threading.active_count()\n"
        "assert count == 1, f'导入产生额外线程: {count}'\n"
    )
    env = {**os.environ, "PYTHONUTF8": "1", "STAGE2_REPO_ROOT": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"意外副作用: {result.stdout}\n{result.stderr}"
    )


# ---------- 5. 非 RM65 拒绝 ----------


def test_rejects_non_rm65(
    rm65_profile: RobotProfile,
    rm65_model: mujoco.MjModel,
    sim_config: dict[str, Any],
    tmp_path: Path,
) -> None:
    """非 RM65 描述（仿真侧）与非 RM65 模型（真机侧）均被明确拒绝。"""
    # 仿真侧：robot_id 非 rm65
    other_id = replace(rm65_profile, robot_id="unitree_d2")
    with pytest.raises(ValueError) as exc_info:
        resolve_rm65_sim_parameters(other_id, rm65_model, sim_config)
    assert "rm65" in str(exc_info.value)

    # 仿真侧：robot_id 冒用 rm65 但主动关节名不符
    renamed = replace(
        rm65_profile, active_joint_names=tuple(f"j{i}" for i in range(1, 7))
    )
    with pytest.raises(ValueError) as exc_info:
        resolve_rm65_sim_parameters(renamed, rm65_model, sim_config)
    assert "active_joint_names" in str(exc_info.value)

    # 真机侧：env 的模型不是 RM65（缺少 r_joint/torque_r_joint/racket_center）
    not_rm65_xml = tmp_path / "not_rm65.xml"
    not_rm65_xml.write_text(
        "<mujoco model=\"not_rm65\">\n"
        "  <option timestep=\"0.005\"/>\n"
        "  <worldbody>\n"
        "    <body name=\"pod\" pos=\"0 0 0\">\n"
        "      <joint name=\"j1\" type=\"hinge\" axis=\"0 0 1\"/>\n"
        "      <geom name=\"pod_geom\" type=\"capsule\""
        " fromto=\"0 0 0 0 0 0.2\" size=\"0.05\"/>\n"
        "    </body>\n"
        "  </worldbody>\n"
        "  <actuator>\n"
        "    <motor name=\"torque_j1\" joint=\"j1\" ctrlrange=\"-10 10\"/>\n"
        "  </actuator>\n"
        "</mujoco>\n",
        encoding="utf-8",
    )
    env = PlanningEnv(not_rm65_xml)
    with pytest.raises(ValueError) as exc_info:
        resolve_rm65_real_parameters(env, RealRobotConfig())
    assert "r_joint1" in str(exc_info.value)


# ---------- 6. 不对称力矩界限拒绝 ----------


def test_rejects_asymmetric_torque_bounds(
    rm65_profile: RobotProfile,
    rm65_model: mujoco.MjModel,
    planning_env: PlanningEnv,
) -> None:
    """力矩上下限不对称时明确拒绝折算，不取较大值放宽限值。"""
    # 折算函数本身：不对称 → 拒绝
    with pytest.raises(ValueError) as exc_info:
        symmetric_torque_max(np.array([-60.0, -60.0]), np.array([60.0, 45.0]))
    assert "不对称" in str(exc_info.value)

    # 仿真适配器层：robot_limits 显式 u_min/u_max 不对称 → 拒绝
    asymmetric_config: dict[str, Any] = {
        "sim": {"dt": 0.005},
        "robot_limits": {
            "u_min": [-60.0, -60.0, -30.0, -10.0, -10.0, -12.0],
            "u_max": [60.0, 60.0, 30.0, 10.0, 10.0, 10.0],
        },
    }
    with pytest.raises(ValueError) as exc_info:
        resolve_rm65_sim_parameters(rm65_profile, rm65_model, asymmetric_config)
    message = str(exc_info.value)
    assert "torque_max" in message
    assert "不对称" in message

    # 真机适配器层：ctrlrange 被改成不对称 → 同样拒绝（改完恢复共享模型）
    original_ctrlrange = planning_env.model.actuator_ctrlrange[:6].copy()
    try:
        planning_env.model.actuator_ctrlrange[5] = [-12.0, 10.0]
        with pytest.raises(ValueError, match="torque_max"):
            resolve_rm65_real_parameters(planning_env, RealRobotConfig())
    finally:
        planning_env.model.actuator_ctrlrange[:6] = original_ctrlrange
