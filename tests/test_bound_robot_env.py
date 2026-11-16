"""BoundRobotEnv 契约测试（阶段 3 · Task 3.3）。

核心断言组（brief + 施工计划 Task 3.3）：
1. ``test_dims_*``/``test_state_roundtrip_*``：n=6/7 通吃 —— 真实 R1D 双侧
   裸模型（n=7 前缀布局）、R1D 带球模型（free joint 位于主动关节之前）、
   乱序合成模型（n=6：树序/执行器序真实打乱 + 2 个保持关节 + 旁观
   free joint + 1 个未绑定执行器）；NQ/NX/NU 为实例属性；状态往返、
   Jacobian (3,n)、速度 = J@qdot、输入数组不被修改、保持关节 held_q/零速。
2. ``test_step_from_state_*``：预测隔离 —— 重复输入输出相同；live
   qpos/qvel/ctrl/time/qfrc_applied/qacc_warmstart 逐项不变；与独立
   MjData 按文档基态配方单步对照 atol=1e-10。
3. ``test_construction_enforces_*``/``test_torque_saturation_*``/
   ``test_position_mode_*``：执行器仅按 actuator_ids 配置；力矩模式实际
   裁剪 torque_max（物理等价验证）；位置模式实际施加非均匀 PD 增益与
   force limit（第七轴/末轴饱和验证，非仅形状）；mode 往返恢复原配置；
   异常输入在步进前拒绝（live data 逐项不变）。
4. ``test_fk_normal_jacobian_match_*``：随机姿态 FK/法向/雅可比/速度与
   ModelKinematics 对照 atol=1e-8（不依赖其私有缓存）；雅可比缓存失效。
5. ``test_control_range_*``：control_range (n,2) 副本、随模式切换、
   RM65Env/PlanningEnv 只读兼容属性。
6. 协议成员：BoundRobotEnv 满足 RobotEnv；RM65Env/PlanningEnv 在协议
   变更（移除 LEFT_ARM_NQ/init_q_left、新增 control_range）后仍满足。
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.ilqt.robot_env_protocol import RobotEnv
from src.robot.model_binding import bind_robot_model
from src.robot.model_factory import RobotModelBundle, load_robot_model
from src.robot.model_kinematics import ModelKinematics
from src.robot.robot_profile import RobotProfile, load_robot_profile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    load_runtime_parameters,
)
from src.sim.bound_robot_env import BoundRobotEnv
from src.sim.sim_config import load_sim_task_config
from src.sim.tennis_scene import build_tennis_scene
from src.utils.mujoco_loader import load_mujoco_model

# 仓库根目录（tests/ 的父目录）
REPO_ROOT = Path(__file__).resolve().parent.parent
SIDES = ("left", "right")

# 双侧描述/仿真 runtime/任务配置（Task 3.1 交付的真实文件）
_PROFILE_YAML = {
    side: REPO_ROOT / "configs" / "robots" / f"r1d_{side}.yaml" for side in SIDES
}
_SIM_RUNTIME_YAML = {
    side: REPO_ROOT / "configs" / "robots" / f"r1d_sim_runtime_{side}.yaml"
    for side in SIDES
}
_TASK_YAML = {
    side: REPO_ROOT / "configs" / f"r1d_sim_{side}.yaml" for side in SIDES
}

# 随机姿态种子（固定，保证失败可复现）
_RNG_SEED = 20261116


# ---------- 用例载体 ----------


@dataclass
class CaseInfo:
    """单个测试用例的环境与元数据。

    Attributes:
        name: 用例名（r1d_left / r1d_right / r1d_tennis_left / synthetic6）。
        env: 被测 BoundRobotEnv 实例。
        bundle: 环境构造用的同源组装结果。
        torque_max: (n,) 力矩上限（与构造参数一致）。
        kp: 位置模式非均匀试验增益（n 维，各轴不同）。
        kd: 位置模式非均匀试验阻尼（n 维）。
        extra_actuator_id: 未绑定执行器 id（仅合成模型有，其他 None）。
        bystander_qpos_slice: 旁观 free joint 的 qpos 切片 (start, end)，
            无球模型为 None。
    """

    name: str
    env: BoundRobotEnv
    bundle: RobotModelBundle
    torque_max: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    extra_actuator_id: int | None
    bystander_qpos_slice: tuple[int, int] | None

    @property
    def n(self) -> int:
        """主动关节自由度数。"""
        return self.env.NQ

    @property
    def binding(self):
        """名称 → 地址绑定。"""
        return self.bundle.binding

    @property
    def model(self) -> mujoco.MjModel:
        """MuJoCo 模型。"""
        return self.bundle.model

    @property
    def parameters(self) -> RobotRuntimeParameters:
        """运行参数。"""
        return self.bundle.parameters


# ---------- 真实 R1D 用例（双侧裸模型 + 左侧带球模型） ----------


@lru_cache(maxsize=4)
def _profile(side: str) -> RobotProfile:
    """加载并缓存该侧 RobotProfile。"""
    return load_robot_profile(_PROFILE_YAML[side])


@lru_cache(maxsize=4)
def _parameters(side: str) -> RobotRuntimeParameters:
    """加载该侧仿真 runtime 参数（含有限 torque_max/kp/kd）。"""
    return load_runtime_parameters(_SIM_RUNTIME_YAML[side], _profile(side))


def _r1d_position_gains(n: int) -> tuple[np.ndarray, np.ndarray]:
    """构造 n 维非均匀试验增益（各轴互异，驱动按 actuator_ids 落位）。"""
    if n == 7:
        # 前 6 维取 r1d_sim_runtime yaml 值，第 7 维改小以制造饱和差异
        kp = np.array([400.0, 400.0, 250.0, 120.0, 60.0, 40.0, 30.0])
        kd = np.array([12.0, 12.0, 8.0, 5.0, 2.5, 1.5, 1.2])
    else:
        kp = np.array([120.0, 100.0, 80.0, 60.0, 40.0, 20.0])
        kd = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    return kp, kd


@lru_cache(maxsize=4)
def _r1d_case(name: str) -> CaseInfo:
    """构建真实 R1D 用例（r1d_left / r1d_right / r1d_tennis_left）。"""
    if name == "r1d_tennis_left":
        scene = build_tennis_scene(
            _profile("left"), _parameters("left"), _task("left")
        )
        bundle = scene.robot
        bystander = (scene.ball_qpos_start, scene.ball_qpos_start + 7)
    else:
        side = name.split("_")[1]
        bundle = load_robot_model(_profile(side), _parameters(side))
        bystander = None
    n = bundle.profile.ndof
    kp, kd = _r1d_position_gains(n)
    torque_max = np.asarray(bundle.parameters.torque_max, dtype=np.float64)
    assert torque_max.shape == (n,), "测试前提：仿真 runtime 必须提供有限 torque_max"
    return CaseInfo(
        name=name,
        env=BoundRobotEnv(bundle),
        bundle=bundle,
        torque_max=torque_max,
        kp=kp,
        kd=kd,
        extra_actuator_id=None,
        bystander_qpos_slice=bystander,
    )


@lru_cache(maxsize=2)
def _task(side: str):
    """加载并缓存该侧 SimTaskConfig。"""
    return load_sim_task_config(_TASK_YAML[side], _profile(side))


# ---------- 乱序合成模型用例（n=6） ----------

# 树序（XML 深度序）与 profile 序不同；执行器 XML 序乱序
_SYNTH_TREE_ORDER = ("j2", "j5", "j1", "j6", "j3", "j4")
_SYNTH_PROFILE_ORDER = ("j1", "j2", "j3", "j4", "j5", "j6")
_SYNTH_ACTUATOR_XML_ORDER = ("m_j4", "m_j1", "m_j6", "m_j2", "m_j5", "m_j3")
_SYNTH_HELD_JOINTS = ("h1", "h2")
_SYNTH_HELD_Q = np.array([0.15, -0.25])
_SYNTH_TORQUE_MAX = np.array([30.0, 25.0, 20.0, 15.0, 10.0, 8.0])
_SYNTH_TCP_SITE = "tool_site"

# 合成模型临时目录（模块级共享同一份 XML；进程退出时清理）
_SYNTH_DIR = Path(tempfile.mkdtemp(prefix="bound_env_synth_"))


def _build_synth_xml() -> str:
    """生成 n=6 乱序合成模型 XML（球 free joint 在最前 + 保持关节 + 未绑定执行器）。"""

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

    # 最内层 link_j4 带 TCP site（局部 +Z 为法向约定）
    innermost = (
        f'        <site name="{_SYNTH_TCP_SITE}" pos="0 0 -0.06" size="0.005" />\n'
    )
    # 由内向外折叠臂链：树序 j2,j5,j1,j6,j3,j4（j4 最内层）
    chain = innermost
    for joint in reversed(_SYNTH_TREE_ORDER):
        chain = _link_body(joint, chain)

    actuators = "\n".join(
        f'    <motor name="{name}" joint="{name[2:]}" gear="1" />'
        for name in _SYNTH_ACTUATOR_XML_ORDER
    )
    # 未绑定执行器：驱动保持关节 h1，显式 ctrlrange（配置切换不得触碰）
    actuators += '\n    <motor name="m_extra" joint="h1" gear="1" ctrlrange="-5 5" />'

    return f"""<mujoco model="synthetic_bound_env">
  <compiler angle="degree" />
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" />
  <worldbody>
    <body name="ball" pos="0.3 0.2 1.0">
      <freejoint name="ball_joint" />
      <geom name="ball_geom" type="sphere" size="0.02" mass="0.05"
       contype="0" conaffinity="0" />
    </body>
    <body name="mount" pos="0 0 0.6">
{chain}      <body name="link_h1" pos="0.05 0 0">
        <joint name="h1" type="hinge" axis="1 0 0" range="-90 90" />
        <geom name="geom_h1" type="capsule" fromto="0 0 0 0.06 0 0"
         size="0.01" mass="0.3" contype="0" conaffinity="0" />
      </body>
      <body name="link_h2" pos="-0.05 0 0">
        <joint name="h2" type="hinge" axis="0 0 1" range="-90 90" />
        <geom name="geom_h2" type="capsule" fromto="0 0 0 -0.06 0 0"
         size="0.01" mass="0.3" contype="0" conaffinity="0" />
      </body>
    </body>
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


@lru_cache(maxsize=1)
def _synth_case() -> CaseInfo:
    """构建并缓存乱序合成用例（n=6，树序/执行器序乱序 + 保持关节 + 旁观球）。"""
    model_path = _SYNTH_DIR / "synthetic_bound_env.xml"
    model_path.write_text(_build_synth_xml(), encoding="utf-8")

    profile = RobotProfile(
        schema_version=1,
        robot_id="synthetic_arm",
        model_path=model_path,
        active_joint_names=_SYNTH_PROFILE_ORDER,
        actuator_names=tuple(f"m_{j}" for j in _SYNTH_PROFILE_ORDER),
        held_joint_names=_SYNTH_HELD_JOINTS,
        tcp_site_name=_SYNTH_TCP_SITE,
    )
    parameters = RobotRuntimeParameters(
        robot_id="synthetic_arm",
        joint_names=_SYNTH_PROFILE_ORDER,
        dt=0.005,
        init_q=np.zeros(6),
        held_q=_SYNTH_HELD_Q.copy(),
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
    kp, kd = _r1d_position_gains(6)
    return CaseInfo(
        name="synthetic6",
        env=BoundRobotEnv(bundle),
        bundle=bundle,
        torque_max=_SYNTH_TORQUE_MAX.copy(),
        kp=kp,
        kd=kd,
        extra_actuator_id=int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_extra")
        ),
        bystander_qpos_slice=(0, 7),  # 球 free joint 位于 qpos 最前
    )


def _case(name: str) -> CaseInfo:
    """按名取用例（真实 R1D 或合成）。"""
    if name == "synthetic6":
        return _synth_case()
    return _r1d_case(name)


_ALL_CASES = ("r1d_left", "r1d_right", "r1d_tennis_left", "synthetic6")


def _cleanup_synth_dir() -> None:
    """进程退出时清理合成模型临时目录。"""
    shutil.rmtree(_SYNTH_DIR, ignore_errors=True)


atexit.register(_cleanup_synth_dir)


# ---------- 通用测试辅助 ----------


def _random_state(case: CaseInfo, rng: np.random.Generator) -> np.ndarray:
    """在模型 jnt_range 内部（15%~85%）生成随机状态 x=[q,qdot]。"""
    joint_ids = np.asarray(case.binding.joint_ids, dtype=int)
    lower = case.model.jnt_range[joint_ids, 0]
    upper = case.model.jnt_range[joint_ids, 1]
    span = upper - lower
    q = lower + span * (0.15 + 0.70 * rng.random(case.n))
    qdot = rng.uniform(-1.0, 1.0, case.n)
    return np.concatenate([q, qdot])


def _actuator_rows(model: mujoco.MjModel, ids) -> dict[str, np.ndarray]:
    """快照指定执行器行的全部配置数组（模式切换/拒绝断言用）。"""
    ids = np.asarray(list(ids), dtype=int)
    return {
        "ctrlrange": model.actuator_ctrlrange[ids].copy(),
        "ctrllimited": model.actuator_ctrllimited[ids].copy(),
        "gainprm": model.actuator_gainprm[ids].copy(),
        "biasprm": model.actuator_biasprm[ids].copy(),
        "biastype": model.actuator_biastype[ids].copy(),
        "forcerange": model.actuator_forcerange[ids].copy(),
        "forcelimited": model.actuator_forcelimited[ids].copy(),
    }


def _assert_rows_equal(
    before: dict[str, np.ndarray], after: dict[str, np.ndarray]
) -> None:
    """逐项断言执行器配置行完全一致。"""
    for key in before:
        np.testing.assert_array_equal(
            before[key], after[key], err_msg=f"执行器配置 {key} 不一致"
        )


def _live_snapshot(env: BoundRobotEnv) -> dict[str, object]:
    """快照 live data 的全部契约保护字段。"""
    data = env.data
    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "ctrl": data.ctrl.copy(),
        "time": float(data.time),
        "qfrc_applied": data.qfrc_applied.copy(),
        "qacc_warmstart": data.qacc_warmstart.copy(),
    }


def _assert_live_unchanged(before: dict[str, object], env: BoundRobotEnv) -> None:
    """断言 live data 与快照逐项一致（数组逐元素、标量精确）。"""
    after = _live_snapshot(env)
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(
                value, after[key], err_msg=f"live {key} 被修改"
            )
        else:
            assert value == after[key], f"live {key} 被修改"


def _reference_step(case: CaseInfo, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """独立 MjData 按文档基态配方单步（力矩模式对照基准，atol=1e-10）。

    配方与 BoundRobotEnv.step_from_state 文档一致：
    mj_resetData → 保持关节 held_q/零速 → 主动关节 ← x →
    ctrl[actuator_ids] ← clip(u, ±torque_max) → mj_step → 保持关节再钉回。
    """
    model, binding = case.model, case.binding
    n = case.n
    qidx = list(binding.qpos_indices)
    vidx = list(binding.qvel_indices)
    hq = list(binding.held_qpos_indices)
    hv = list(binding.held_qvel_indices)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if hq:
        data.qpos[hq] = case.parameters.held_q
        data.qvel[hv] = 0.0
    data.qpos[qidx] = x[:n]
    data.qvel[vidx] = x[n:]
    data.ctrl[list(binding.actuator_ids)] = np.clip(
        u, -case.torque_max, case.torque_max
    )
    mujoco.mj_step(model, data)
    if hq:
        data.qpos[hq] = case.parameters.held_q
        data.qvel[hv] = 0.0
    return np.concatenate([data.qpos[qidx].copy(), data.qvel[vidx].copy()])


# ---------- 1. 维度契约与状态往返 ----------


def test_synthetic_layout_actually_scrambled() -> None:
    """合成模型布局压力前提：绑定地址/执行器序真实乱序（非前缀假设）。"""
    case = _synth_case()
    qpos = list(case.binding.qpos_indices)
    acts = list(case.binding.actuator_ids)
    assert qpos != sorted(qpos), "合成模型 qpos 地址应乱序（树序打乱）"
    assert acts != sorted(acts), "合成模型执行器 id 应乱序（XML 序打乱）"
    assert min(qpos) >= 7, "球 free joint 应位于主动关节之前"
    # 对照：真实 R1D 裸模型是前缀布局（两种布局同一套代码通吃）
    bare = _r1d_case("r1d_left")
    assert list(bare.binding.qpos_indices) == list(range(bare.n))
    assert list(bare.binding.actuator_ids) == list(range(bare.n))


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_dims_and_instance_attributes(case_name: str) -> None:
    """NQ/NX/NU 为实例属性且等于绑定维数（n=6/7 通吃）。"""
    case = _case(case_name)
    n = len(case.binding.qpos_indices)
    # 实例属性（而非仅类变量）：值等于绑定维数
    assert case.env.NQ == n
    assert case.env.NX == 2 * n
    assert case.env.NU == n
    assert case.env.dt == pytest.approx(case.parameters.dt)


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_state_roundtrip_and_inputs_unmodified(case_name: str) -> None:
    """set→get 状态往返精确一致；输入数组不被修改。"""
    case = _case(case_name)
    rng = np.random.default_rng(_RNG_SEED)
    for _ in range(3):
        x = _random_state(case, rng)
        x_before = x.copy()
        case.env.set_arm_state(x)
        np.testing.assert_array_equal(case.env.get_arm_state(), x)
        np.testing.assert_array_equal(x, x_before, err_msg="set_arm_state 修改了输入")


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_jacobian_shape_and_velocity(case_name: str) -> None:
    """Jacobian 形状 (3,n)；末端速度 = J@qdot（独立 mj_jacSite 对照）。"""
    case = _case(case_name)
    rng = np.random.default_rng(_RNG_SEED + 1)
    x = _random_state(case, rng)
    case.env.set_arm_state(x)
    jacp = case.env.get_ee_jacp()
    jacr = case.env.get_ee_jacr()
    n = case.n
    assert jacp.shape == (3, n)
    assert jacr.shape == (3, n)

    # 独立 mj_jacSite（列按 binding.qvel_indices 收取）
    kin = ModelKinematics(case.bundle)
    kin.set_state(x)
    ref_jacp, ref_jacr = kin.tcp_jacobian()
    np.testing.assert_allclose(jacp, ref_jacp, atol=1e-8)
    np.testing.assert_allclose(jacr, ref_jacr, atol=1e-8)
    np.testing.assert_allclose(
        case.env.get_ee_vel(), ref_jacp @ x[n:], atol=1e-8, err_msg="get_ee_vel ≠ J@qdot"
    )


def test_held_joints_pinned() -> None:
    """合成模型：保持关节在 set_arm_state 与 step 后保持 held_q/零速。"""
    case = _synth_case()
    env = case.env
    hq = list(case.binding.held_qpos_indices)
    hv = list(case.binding.held_qvel_indices)
    rng = np.random.default_rng(_RNG_SEED + 2)
    x = _random_state(case, rng)
    env.set_arm_state(x)
    np.testing.assert_allclose(env.data.qpos[hq], case.parameters.held_q, atol=1e-15)
    np.testing.assert_array_equal(env.data.qvel[hv], 0.0)

    env.set_arm_state(x)
    env.step(np.zeros(case.n))
    np.testing.assert_allclose(env.data.qpos[hq], case.parameters.held_q, atol=1e-15)
    np.testing.assert_array_equal(env.data.qvel[hv], 0.0)


@pytest.mark.parametrize("case_name", ["r1d_tennis_left", "synthetic6"])
def test_bystander_joints_untouched(case_name: str) -> None:
    """旁观 free joint（球）：set_arm_state/step_from_state 不触碰其 qpos。"""
    case = _case(case_name)
    env = case.env
    start, end = case.bystander_qpos_slice
    assert start is not None and end is not None
    custom = np.array([0.42, -0.13, 0.87, 1.0, 0.0, 0.0, 0.0])
    env.data.qpos[start:end] = custom

    rng = np.random.default_rng(_RNG_SEED + 3)
    x = _random_state(case, rng)
    env.set_arm_state(x)
    np.testing.assert_array_equal(env.data.qpos[start:end], custom)

    u = np.full(case.n, 0.3)
    env.step_from_state(x, u)
    np.testing.assert_array_equal(env.data.qpos[start:end], custom)


# ---------- 2. step_from_state 隔离 ----------


@pytest.mark.parametrize("case_name", ["r1d_left", "synthetic6"])
def test_step_from_state_deterministic_and_live_unchanged(case_name: str) -> None:
    """重复输入输出相同；live qpos/qvel/ctrl/time/qfrc/warmstart 逐项不变。"""
    case = _case(case_name)
    env = case.env
    # 先推进 live 若干步，使 live 状态非平凡（含非零 ctrl/warmstart）
    rng = np.random.default_rng(_RNG_SEED + 4)
    env.reset()
    for _ in range(3):
        env.step(rng.uniform(-0.5, 0.5, case.n) * case.torque_max)
    snapshot = _live_snapshot(env)

    x = _random_state(case, rng)
    u = rng.uniform(-1.0, 1.0, case.n) * case.torque_max
    x_before, u_before = x.copy(), u.copy()
    out1 = env.step_from_state(x, u)
    out2 = env.step_from_state(x, u)
    np.testing.assert_array_equal(out1, out2, err_msg="重复输入输出不同")
    np.testing.assert_array_equal(x, x_before, err_msg="step_from_state 修改了 x")
    np.testing.assert_array_equal(u, u_before, err_msg="step_from_state 修改了 u")
    _assert_live_unchanged(snapshot, env)


@pytest.mark.parametrize("case_name", ["r1d_left", "synthetic6"])
def test_step_from_state_matches_independent_mjdata(case_name: str) -> None:
    """与独立 MjData 按文档基态配方单步对照 atol=1e-10。"""
    case = _case(case_name)
    env = case.env
    rng = np.random.default_rng(_RNG_SEED + 5)
    for _ in range(3):
        x = _random_state(case, rng)
        u = rng.uniform(-1.0, 1.0, case.n) * case.torque_max
        expected = _reference_step(case, x, u)
        actual = env.step_from_state(x, u)
        np.testing.assert_allclose(
            actual, expected, atol=1e-10, err_msg="step_from_state 与独立单步不一致"
        )


# ---------- 3. 执行器配置与物理应用 ----------


@pytest.mark.parametrize("case_name", ["r1d_left", "synthetic6"])
def test_construction_enforces_torque_limits(case_name: str) -> None:
    """构造即应用力矩限界（ctrl/force 双通道）；未绑定执行器不被触碰。"""
    case = _case(case_name)
    model, ids = case.model, list(case.binding.actuator_ids)
    assert np.all(model.actuator_ctrllimited[ids] == 1), "ctrllimited 未置位"
    np.testing.assert_allclose(model.actuator_ctrlrange[ids][:, 0], -case.torque_max)
    np.testing.assert_allclose(model.actuator_ctrlrange[ids][:, 1], case.torque_max)
    assert np.all(model.actuator_forcelimited[ids] == 1), "forcelimited 未置位"
    np.testing.assert_allclose(model.actuator_forcerange[ids][:, 0], -case.torque_max)
    np.testing.assert_allclose(model.actuator_forcerange[ids][:, 1], case.torque_max)
    if case.extra_actuator_id is not None:
        aid = case.extra_actuator_id
        assert int(model.actuator_ctrllimited[aid]) == 1, "未绑定执行器被触碰"
        np.testing.assert_allclose(model.actuator_ctrlrange[aid], (-5.0, 5.0))


@pytest.mark.parametrize("case_name", ["r1d_left", "synthetic6"])
def test_torque_saturation_physics(case_name: str) -> None:
    """力矩模式实际裁剪：超限控制与限界控制产生完全相同的单步结果。"""
    case = _case(case_name)
    env = case.env
    env.configure_actuator_mode("torque")
    rng = np.random.default_rng(_RNG_SEED + 6)
    x = _random_state(case, rng)
    # 符号随机、幅值严格在界内（未饱和）；u_limit 为对应的饱和限界
    u = rng.uniform(0.2, 0.9, case.n) * case.torque_max
    u = u * np.where(rng.random(case.n) < 0.5, 1.0, -1.0)
    u_limit = np.sign(u) * case.torque_max

    np.testing.assert_array_equal(
        env.step_from_state(x, 5.0 * u),
        env.step_from_state(x, u_limit),
        err_msg="正向超限控制未被裁剪到 torque_max",
    )
    np.testing.assert_array_equal(
        env.step_from_state(x, -5.0 * u),
        env.step_from_state(x, -u_limit),
        err_msg="负向超限控制未被裁剪到 -torque_max",
    )
    # 未饱和输入与饱和限界结果应不同（证明限界真实生效而非恒等）
    assert not np.array_equal(
        env.step_from_state(x, u), env.step_from_state(x, u_limit)
    ), "限界内控制与限界控制结果相同，裁剪未真实生效"


@pytest.mark.parametrize("case_name", ["r1d_left", "synthetic6"])
def test_position_mode_configures_pd_by_actuator_ids(case_name: str) -> None:
    """位置模式按 actuator_ids 逐执行器写入 PD/ctrlrange/forcerange；乱序落位正确。"""
    case = _case(case_name)
    model = case.model
    env = case.env
    env.configure_actuator_mode("position", kp=case.kp, kd=case.kd)
    assert env.actuator_mode == 1
    np.testing.assert_array_equal(env.kp, case.kp)
    np.testing.assert_array_equal(env.kd, case.kd)

    for i, aid in enumerate(case.binding.actuator_ids):
        jid = int(model.actuator_trnid[aid, 0])
        assert int(model.actuator_biastype[aid]) == int(
            mujoco.mjtBias.mjBIAS_AFFINE
        ), f"执行器 {aid} biastype 未设为 AFFINE"
        np.testing.assert_allclose(
            model.actuator_gainprm[aid, 0],
            case.kp[i],
            atol=1e-12,
            err_msg=f"执行器 {aid}（绑定序 {i}）kp 落位错误",
        )
        np.testing.assert_allclose(
            model.actuator_biasprm[aid, :3],
            (0.0, -case.kp[i], -case.kd[i]),
            atol=1e-12,
        )
        # ctrlrange = 所驱动关节的 jnt_range（位置指令按关节限位裁剪）
        np.testing.assert_allclose(
            model.actuator_ctrlrange[aid], model.jnt_range[jid], atol=1e-12
        )
        assert int(model.actuator_ctrllimited[aid]) == 1
        np.testing.assert_allclose(
            model.actuator_forcerange[aid], (-case.torque_max[i], case.torque_max[i])
        )
        assert int(model.actuator_forcelimited[aid]) == 1

    if case.extra_actuator_id is not None:
        aid = case.extra_actuator_id
        assert int(model.actuator_biastype[aid]) == int(
            mujoco.mjtBias.mjBIAS_NONE
        ), "未绑定执行器被位置模式触碰"
        np.testing.assert_allclose(model.actuator_ctrlrange[aid], (-5.0, 5.0))


@pytest.mark.parametrize("case_name", ["r1d_left", "synthetic6"])
def test_position_mode_pd_and_saturation_physics(case_name: str) -> None:
    """位置模式实际施加非均匀 PD 与 force limit：末轴饱和、首轴不饱和。"""
    case = _case(case_name)
    env = case.env
    n = case.n
    env.configure_actuator_mode("position", kp=case.kp, kd=case.kd)
    env.configure_feedforward(False)

    q0 = np.asarray(case.parameters.init_q, dtype=np.float64)
    x0 = np.concatenate([q0, np.zeros(n)])
    env.set_arm_state(x0)

    # 期望力矩（qdot=0 → kd 项为零）：
    # 首轴误差小（未饱和）→ kp*delta；末轴误差大（饱和）→ ±torque_max
    delta = np.zeros(n)
    delta[0] = 0.01
    delta[n - 1] = 1.0
    expected = case.kp * np.clip(delta, -case.torque_max / case.kp, case.torque_max / case.kp)

    env.step(q0 + delta)
    vidx = list(case.binding.qvel_indices)
    actual = env.data.qfrc_actuator[vidx].copy()
    np.testing.assert_allclose(
        actual, expected, atol=1e-6, err_msg="位置模式 PD 力矩与 kp/kd/饱和不符"
    )
    # 末轴确实饱和在 torque_max（而非 kp*delta）且首轴确实不饱和
    assert abs(actual[n - 1]) == pytest.approx(case.torque_max[n - 1], abs=1e-6)
    assert abs(actual[0]) < case.torque_max[0]
    # 非均匀增益证据：末轴与首轴每单位误差的力矩差异即 kp 差异
    assert case.kp[n - 1] != case.kp[0]


@pytest.mark.parametrize("case_name", ["r1d_left", "synthetic6"])
def test_feedforward_total_torque_capped_live(case_name: str) -> None:
    """位置模式前馈：PD+前馈总驱动力矩逐轴 ≤ torque_max（live 路径）。

    审查 P2：模型 forcerange 只裁剪 PD 执行器分量，qfrc_applied 中的
    偏置补偿不受限——总主动力矩曾可超上限（R1D 左臂第一轴实测
    −61.95 Nm vs 上限 60 Nm）；修复后限幅在前馈写入时统一施加。
    """
    case = _case(case_name)
    env = case.env
    n = case.n
    env.configure_actuator_mode("position", kp=case.kp, kd=case.kd)
    rng = np.random.default_rng(_RNG_SEED + 40)
    vidx = list(case.binding.qvel_indices)
    saw_nonzero_ff = False
    for _ in range(8):
        x = _random_state(case, rng)
        # 目标角带大误差 → PD 分量易饱和，压缩前馈余量（限幅压力场景）
        u = x[:n] + rng.uniform(-1.5, 1.5, size=n)
        env.set_arm_state(x)
        env.step(u)
        # qfrc_actuator/qfrc_applied 均为本步步前状态的力（可直接相加）
        total = env.data.qfrc_actuator[vidx] + env.data.qfrc_applied[vidx]
        assert np.all(np.abs(total) <= case.torque_max + 1e-9), (
            f"总驱动力矩超限: {total} vs ±{case.torque_max}"
        )
        saw_nonzero_ff = saw_nonzero_ff or bool(
            np.any(np.abs(env.data.qfrc_applied[vidx]) > 1e-9)
        )
    # 前馈路径确实激活（限幅未把偏置补偿整体削成零）
    assert saw_nonzero_ff, "前馈偏置力恒为零——限幅或前馈路径未生效"


@pytest.mark.parametrize("case_name", ["r1d_left", "synthetic6"])
def test_feedforward_total_torque_capped_scratch(case_name: str) -> None:
    """scratch 路径（step_from_state）共享同一前馈总力矩限界（白盒）。

    _apply_step_forces 同时服务 live/scratch 两条路径；此处按
    step_from_state 的基态配方手动铺 scratch 后直接断言同一契约。
    """
    case = _case(case_name)
    env = case.env
    n = case.n
    env.configure_actuator_mode("position", kp=case.kp, kd=case.kd)
    rng = np.random.default_rng(_RNG_SEED + 41)
    vidx = list(case.binding.qvel_indices)
    scratch = env._scratch  # noqa: SLF001 — 白盒：复刻 step_from_state 铺态
    for _ in range(8):
        x = _random_state(case, rng)
        u = x[:n] + rng.uniform(-1.5, 1.5, size=n)
        mujoco.mj_resetData(env.model, scratch)
        env._pin_held_joints(scratch)  # noqa: SLF001 — 白盒同上
        scratch.qpos[env._qpos_indices] = x[:n]
        scratch.qvel[env._qvel_indices] = x[n:]
        scratch.ctrl[env._actuator_ids] = env._compute_ctrl(u, x[:n].copy())
        env._apply_step_forces(scratch)
        # 内部 mj_forward 已算出 qfrc_actuator（步前状态，与预测一致）
        total = scratch.qfrc_actuator[vidx] + scratch.qfrc_applied[vidx]
        assert np.all(np.abs(total) <= case.torque_max + 1e-9), (
            f"scratch 总驱动力矩超限: {total} vs ±{case.torque_max}"
        )


@pytest.mark.parametrize("case_name", ["r1d_left", "synthetic6"])
def test_mode_roundtrip_restores_config(case_name: str) -> None:
    """position → torque 往返恢复构造时（力矩模式）的执行器配置。"""
    case = _case(case_name)
    env = case.env
    env.configure_actuator_mode("torque")
    ids = list(case.binding.actuator_ids)
    snapshot = _actuator_rows(case.model, ids)

    env.configure_actuator_mode("position", kp=case.kp, kd=case.kd)
    env.configure_actuator_mode("torque")
    _assert_rows_equal(snapshot, _actuator_rows(case.model, ids))

    assert env.actuator_mode == 0
    assert env.kp is None and env.kd is None
    assert env.use_feedforward is False


def test_configure_rejects_invalid_before_mutation() -> None:
    """异常配置在写入模型前拒绝（模型行保持不变）。"""
    case = _r1d_case("r1d_left")
    env = case.env
    env.configure_actuator_mode("torque")
    ids = list(case.binding.actuator_ids)
    snapshot = _actuator_rows(case.model, ids)
    n = case.n

    bad_calls: list[tuple[dict, str]] = [
        (dict(mode="position"), "kp"),  # 缺 kp/kd
        (dict(mode="position", kp=case.kp, kd=None), "kd"),
        (dict(mode="position", kp=case.kp[: n - 1], kd=case.kd), "kp"),
        (dict(mode="position", kp=np.full(n, np.nan), kd=case.kd), "kp"),
        (dict(mode="position", kp=-case.kp, kd=case.kd), "kp"),
        (dict(mode="position", kp=case.kp, kd=-case.kd), "kd"),
        (dict(mode="velocity"), "模式"),
    ]
    for kwargs, match in bad_calls:
        with pytest.raises(ValueError, match=match):
            env.configure_actuator_mode(**kwargs)
    _assert_rows_equal(snapshot, _actuator_rows(case.model, ids))


def test_step_rejects_invalid_input_before_write() -> None:
    """异常步进输入在写入前拒绝（live data 逐项不变）。"""
    case = _r1d_case("r1d_left")
    env = case.env
    env.configure_actuator_mode("torque")
    rng = np.random.default_rng(_RNG_SEED + 7)
    env.reset()
    env.step(rng.uniform(-0.5, 0.5, case.n) * case.torque_max)
    snapshot = _live_snapshot(env)
    x = env.get_arm_state()

    bad_us = [np.zeros(case.n + 1), np.zeros(case.n - 1), np.full(case.n, np.nan)]
    for bad_u in bad_us:
        with pytest.raises(ValueError):
            env.step(bad_u)
        _assert_live_unchanged(snapshot, env)
    with pytest.raises(ValueError):
        env.step_from_state(x, np.full(case.n, np.inf))
    with pytest.raises(ValueError):
        env.step_from_state(np.zeros(2 * case.n + 1), np.zeros(case.n))
    with pytest.raises(ValueError):
        env.set_arm_state(np.full(2 * case.n, np.nan))
    _assert_live_unchanged(snapshot, env)


# ---------- 4. FK/法向/雅可比/IK 与缓存 ----------


@pytest.mark.parametrize("case_name", ["r1d_left", "r1d_right", "synthetic6"])
def test_fk_normal_jacobian_match_model_kinematics(case_name: str) -> None:
    """随机姿态 FK/法向/雅可比与 ModelKinematics 对照 atol=1e-8。"""
    case = _case(case_name)
    env, kin = case.env, ModelKinematics(case.bundle)
    rng = np.random.default_rng(_RNG_SEED + 8)
    for _ in range(4):
        x = _random_state(case, rng)
        env.set_arm_state(x)
        kin.set_state(x)
        pos, rot = kin.tcp_pose()
        np.testing.assert_allclose(env.get_ee_pos(), pos, atol=1e-8)
        # 法向约定：TCP site 局部 +Z（对照 ModelKinematics 旋转第 3 列）
        np.testing.assert_allclose(
            env.get_ee_normal(), rot[:, 2] / np.linalg.norm(rot[:, 2]), atol=1e-8
        )
        jacp, jacr = kin.tcp_jacobian()
        np.testing.assert_allclose(env.get_ee_jacp(), jacp, atol=1e-8)
        np.testing.assert_allclose(env.get_ee_jacr(), jacr, atol=1e-8)
        np.testing.assert_allclose(env.get_ee_vel(), jacp @ x[case.n :], atol=1e-8)


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_jacobian_cache_invalidation(case_name: str) -> None:
    """状态变化后雅可比缓存失效（旧缓存不得泄漏到新状态的速度计算）。"""
    case = _case(case_name)
    env = case.env
    rng = np.random.default_rng(_RNG_SEED + 9)
    xa = _random_state(case, rng)
    xb = _random_state(case, rng)
    if np.allclose(xa[: case.n], xb[: case.n]):
        xb[: case.n] += 0.1  # 保证两姿态不同

    env.set_arm_state(xa)
    ja = env.get_ee_jacp()  # 触发缓存
    env.set_arm_state(xb)
    jb = env.get_ee_jacp()
    assert not np.allclose(ja, jb), "状态变化后雅可比未刷新"
    np.testing.assert_allclose(env.get_ee_vel(), jb @ xb[case.n :], atol=1e-12)

    # step 后缓存失效：直接改 qpos（不经 set_arm_state）再步进，step 内部
    # forward 使运动学落在步前位形 xb；若缓存未失效将返回旧 J(xa)。
    # （沿旧环境约定：mj_step 后运动学数组停留在步前位形，精确 FK 查询
    # 前应先 set_arm_state/update_kinematics。）
    env.set_arm_state(xa)
    env.get_ee_jacp()  # 重建缓存 J(xa)
    env.data.qpos[list(case.binding.qpos_indices)] = xb[: case.n]
    env.data.qvel[list(case.binding.qvel_indices)] = xb[case.n :]
    env.step(np.zeros(case.n))
    kin = ModelKinematics(case.bundle)
    kin.set_state(np.concatenate([xb[: case.n], np.zeros(case.n)]))
    ref_jacp, _ = kin.tcp_jacobian()
    np.testing.assert_allclose(
        env.get_ee_jacp(), ref_jacp, atol=1e-8, err_msg="step 后缓存未失效"
    )


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_update_kinematics_refreshes_fk(case_name: str) -> None:
    """update_kinematics 刷新 FK 且使缓存失效。"""
    case = _case(case_name)
    env = case.env
    rng = np.random.default_rng(_RNG_SEED + 10)
    xa = _random_state(case, rng)
    xb = _random_state(case, rng)

    env.set_arm_state(xa)
    env.get_ee_jacp()  # 建缓存
    # 直接改 qpos 后 update_kinematics：FK 应反映新姿态
    env.data.qpos[list(case.binding.qpos_indices)] = xb[: case.n]
    env.update_kinematics()
    kin = ModelKinematics(case.bundle)
    kin.set_state(np.concatenate([xb[: case.n], np.zeros(case.n)]))
    ref_pos, _ = kin.tcp_pose()
    np.testing.assert_allclose(env.get_ee_pos(), ref_pos, atol=1e-8)


@pytest.mark.parametrize("case_name", ["r1d_left", "synthetic6"])
def test_solve_ik_reaches_target(case_name: str) -> None:
    """IK：FK 目标可达（维数无关，返回 (n,)）。"""
    case = _case(case_name)
    env = case.env
    rng = np.random.default_rng(_RNG_SEED + 11)
    for _ in range(3):
        x = _random_state(case, rng)
        env.set_arm_state(x)
        target = env.get_ee_pos()
        q = env.solve_ik(target, q_init=np.asarray(case.parameters.init_q))
        assert q.shape == (case.n,)
        env.set_arm_state(np.concatenate([q, np.zeros(case.n)]))
        err = np.linalg.norm(env.get_ee_pos() - target)
        assert err < 1e-2, f"IK 未收敛到目标（误差 {err:.3e} m）"


# ---------- 5. control_range ----------


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_control_range_property(case_name: str) -> None:
    """control_range (n,2) 副本：力矩模式 ±torque_max、位置模式关节限位。"""
    case = _case(case_name)
    env = case.env
    env.configure_actuator_mode("torque")
    cr = env.control_range
    assert cr.shape == (case.n, 2)
    np.testing.assert_allclose(cr[:, 0], -case.torque_max)
    np.testing.assert_allclose(cr[:, 1], case.torque_max)

    env.configure_actuator_mode("position", kp=case.kp, kd=case.kd)
    cr = env.control_range
    for i, aid in enumerate(case.binding.actuator_ids):
        jid = int(case.model.actuator_trnid[aid, 0])
        np.testing.assert_allclose(
            cr[i], case.model.jnt_range[jid], atol=1e-12, err_msg=f"绑定序 {i} 行不符"
        )

    # 副本语义：改返回值不影响环境
    env.configure_actuator_mode("torque")
    cr = env.control_range
    cr[:] = 0.0
    np.testing.assert_allclose(env.control_range[:, 1], case.torque_max)
    env.configure_actuator_mode("torque")


def test_rm65_and_planning_env_control_range_compat() -> None:
    """RM65Env/PlanningEnv 只读 control_range 兼容属性（协议新增成员）。"""
    from src.ilqt.planning_env import PlanningEnv
    from src.sim.rm65_env import RM65Env

    model_path = REPO_ROOT / "src" / "robot" / "rm65_model.xml"
    rm65 = RM65Env(model_path)
    np.testing.assert_allclose(
        rm65.control_range, rm65.model.actuator_ctrlrange[: rm65.NU]
    )
    planning = PlanningEnv(model_path)
    np.testing.assert_allclose(
        planning.control_range, planning.model.actuator_ctrlrange[: planning.NU]
    )
    # 只读：赋值应抛 AttributeError
    with pytest.raises(AttributeError):
        rm65.control_range = np.zeros((6, 2))  # type: ignore[misc]


# ---------- 6. reset 与协议成员 ----------


@pytest.mark.parametrize("case_name", ["r1d_left", "synthetic6"])
def test_reset_returns_init_state_and_keeps_config(case_name: str) -> None:
    """reset 返回 [init_q, 0]；保持关节回 held_q；执行器配置不被重置。"""
    case = _case(case_name)
    env = case.env
    env.configure_actuator_mode("torque")
    ids = list(case.binding.actuator_ids)
    snapshot = _actuator_rows(case.model, ids)

    rng = np.random.default_rng(_RNG_SEED + 12)
    env.step(rng.uniform(-0.5, 0.5, case.n) * case.torque_max)
    x0 = env.reset()
    expected = np.concatenate(
        [np.asarray(case.parameters.init_q), np.zeros(case.n)]
    )
    np.testing.assert_allclose(x0, expected, atol=1e-15)
    if case.binding.held_qpos_indices:
        np.testing.assert_allclose(
            env.data.qpos[list(case.binding.held_qpos_indices)],
            case.parameters.held_q,
            atol=1e-15,
        )
    _assert_rows_equal(snapshot, _actuator_rows(case.model, ids))


@pytest.mark.parametrize("case_name", _ALL_CASES)
def test_bound_env_satisfies_protocol(case_name: str) -> None:
    """BoundRobotEnv 满足 RobotEnv 协议（isinstance 成立）。"""
    assert isinstance(_case(case_name).env, RobotEnv)


def test_old_envs_still_satisfy_protocol() -> None:
    """协议变更后 RM65Env/PlanningEnv 仍满足 RobotEnv（含 control_range）。"""
    from src.ilqt.planning_env import PlanningEnv
    from src.sim.rm65_env import RM65Env

    model_path = REPO_ROOT / "src" / "robot" / "rm65_model.xml"
    assert isinstance(RM65Env(model_path), RobotEnv)
    assert isinstance(PlanningEnv(model_path), RobotEnv)
