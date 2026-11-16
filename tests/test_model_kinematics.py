"""ModelKinematics 运动学探针测试（阶段 2 · Task 2.6）。

测试组（任务卡 6 项，左右双侧参数化）：
1. ``test_state_roundtrip_uses_binding``：set_state/get_state 经
   ModelBinding 地址读写；写入结果与"手动按地址组装的 MjData"FK 一致。
2. ``test_invalid_state_rejected_before_write``：shape/finite/关节位置
   界限校验；非法输入在写入 MjData 前抛错，不产生部分写入。
3. ``test_fk_matches_independent_urdf``：核心断言——MuJoCo tcp_pose 与
   tests/r1d_reference.py 的**独立整机 URDF** FK（numpy+stdlib 实现，
   不 import src/mujoco）对照 zero/init_q/seed=42×8 姿态，双侧
   位置/旋转 atol=1e-8。拦截"镜像复制冒充双侧导出"与"肩部 ±0.26187
   安装角遗漏"。
4. ``test_translation_and_rotation_jacobians``：tcp_jacobian 的解析雅可比
   与独立 URDF FK 的中心差分（ε=1e-6 rad，姿态离上下限 ≥1e-3）对照；
   位置 atol=1e-6/rtol=1e-5；角速度用世界系 ``(dR/dq) @ R.T`` 反对称
   向量同容差。
5. ``test_synthetic_disordered_model``：qpos 前带 free joint、执行器乱序
   的合成 bundle——验证按 ModelBinding 地址读写（不只测地址恰为 0..6
   的 R1D 情形）。
6. ``test_separate_data_instances``：每实例自有 MjData，互不串扰。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

import tests.r1d_reference as r1d_reference
from src.robot.model_binding import bind_robot_model
from src.robot.model_factory import RobotModelBundle, load_robot_model
from src.robot.robot_profile import RobotProfile, load_robot_profile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    load_runtime_parameters,
)
from src.utils.mujoco_loader import load_mujoco_model

REPO_ROOT = Path(__file__).resolve().parent.parent
SIDES = ("left", "right")
PROFILE_PATHS = {
    side: REPO_ROOT / "configs" / "robots" / f"r1d_{side}.yaml" for side in SIDES
}
RUNTIME_PATHS = {
    side: REPO_ROOT / "configs" / "robots" / f"r1d_runtime_{side}.yaml"
    for side in SIDES
}

# FK 对照姿态数（zero + init_q 之外再取 seed=42 的 8 个随机姿态）
FK_RANDOM_POSTURES = 8
# 雅可比对照姿态数（init_q 之外再取 4 个随机姿态）
JAC_RANDOM_POSTURES = 4
# 随机姿态采样时距上下限的最小行程比例（雅可比姿态须 ≥1e-3 rad）
FK_MARGIN_RATIO = 0.05
JAC_MARGIN_RATIO = 0.01


def _kinematics_class() -> Any:
    """惰性导入 ModelKinematics；缺失即判功能缺失（保持红灯为逐测试失败）。"""
    try:
        from src.robot import model_kinematics
    except ImportError as exc:
        pytest.fail(f"model_kinematics 模块缺失: {exc}")
    return model_kinematics.ModelKinematics


def _default_bundle(side: str) -> RobotModelBundle:
    """以该侧默认描述 + 默认运行参数组装 bundle（2.5b 接收口）。"""
    profile = load_robot_profile(PROFILE_PATHS[side])
    params = load_runtime_parameters(RUNTIME_PATHS[side], profile)
    return load_robot_model(profile, params)


def _random_postures(
    bundle: RobotModelBundle, count: int, seed: int, margin_ratio: float
) -> list[np.ndarray]:
    """在 [lower+margin·range, upper-margin·range] 内均匀采样 count 个姿态。"""
    rng = np.random.default_rng(seed)
    lower = np.asarray(bundle.parameters.q_lower, dtype=np.float64)
    upper = np.asarray(bundle.parameters.q_upper, dtype=np.float64)
    span = upper - lower
    return [
        rng.uniform(lower + margin_ratio * span, upper - margin_ratio * span)
        for _ in range(count)
    ]


def _make_state(q: np.ndarray, qdot: np.ndarray | None = None) -> np.ndarray:
    """拼 [q, qdot] 状态向量（qdot 缺省为零）。"""
    n = len(q)
    if qdot is None:
        qdot = np.zeros(n)
    return np.concatenate([np.asarray(q, dtype=np.float64), qdot])


def _manual_fk(
    bundle: RobotModelBundle, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """独立于 ModelKinematics 的手动 FK：新 MjData 按绑定地址写入后 mj_forward。"""
    data = mujoco.MjData(bundle.model)
    data.qpos[list(bundle.binding.qpos_indices)] = q
    mujoco.mj_forward(bundle.model, data)
    sid = bundle.binding.tcp_site_id
    return (
        data.site_xpos[sid].copy(),
        data.site_xmat[sid].reshape(3, 3).copy(),
    )


# ---------------------------------------------------------------- 状态读写


@pytest.mark.parametrize("side", SIDES)
def test_state_roundtrip_uses_binding(side: str) -> None:
    """set_state/get_state 按 ModelBinding 地址读写，FK 与手动按地址组装一致。"""
    bundle = _default_bundle(side)
    kin = _kinematics_class()(bundle)
    n = bundle.profile.ndof
    assert n == 7

    q = _random_postures(bundle, 1, seed=7, margin_ratio=FK_MARGIN_RATIO)[0]
    qdot = np.linspace(-0.3, 0.4, n)
    x = _make_state(q, qdot)

    kin.set_state(x)
    np.testing.assert_allclose(kin.get_state(), x, atol=0.0, rtol=0.0)

    # 写入确实落在绑定地址：与手动按地址组装的 MjData FK 逐位一致
    pos, rot = kin.tcp_pose()
    manual_pos, manual_rot = _manual_fk(bundle, q)
    np.testing.assert_allclose(pos, manual_pos, atol=1e-12)
    np.testing.assert_allclose(rot, manual_rot, atol=1e-12)

    # 再写一次不同状态，get_state 跟随更新
    q2 = np.asarray(bundle.parameters.init_q, dtype=np.float64)
    kin.set_state(_make_state(q2))
    np.testing.assert_allclose(kin.get_state()[:n], q2, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(kin.get_state()[n:], np.zeros(n), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("side", SIDES)
def test_invalid_state_rejected_before_write(side: str) -> None:
    """非法输入在写入 MjData 前抛错——校验 shape/finite/界限，且不部分写入。"""
    bundle = _default_bundle(side)
    kin = _kinematics_class()(bundle)
    n = bundle.profile.ndof
    lower = np.asarray(bundle.parameters.q_lower, dtype=np.float64)
    upper = np.asarray(bundle.parameters.q_upper, dtype=np.float64)

    good_q = np.asarray(bundle.parameters.init_q, dtype=np.float64)
    good = _make_state(good_q, np.full(n, 0.1))
    kin.set_state(good)

    bad_nan_q = good.copy()
    bad_nan_q[2] = np.nan
    bad_inf_qdot = good.copy()
    bad_inf_qdot[n + 4] = np.inf
    bad_below = good.copy()
    bad_below[3] = lower[3] - 1e-6
    bad_above = good.copy()
    bad_above[1] = upper[1] + 1e-6

    bad_inputs = [
        good[: 2 * n - 1],  # shape (13,)
        np.concatenate([good, [0.0]]),  # shape (15,)
        good_q,  # shape (7,)
        good.reshape(2 * n, 1),  # 2D 形状
        good.tolist() + [0.0] * 3,  # list 且长度错
        bad_nan_q,
        bad_inf_qdot,
        bad_below,
        bad_above,
    ]
    for bad in bad_inputs:
        with pytest.raises(ValueError):
            kin.set_state(bad)
        # 不部分写入：MjData 保持上一个合法状态
        np.testing.assert_allclose(kin.get_state(), good, atol=0.0, rtol=0.0)

    # 边界值本身合法（闭区间）
    kin.set_state(_make_state(lower.copy()))
    np.testing.assert_allclose(kin.get_state()[:n], lower, atol=0.0, rtol=0.0)
    kin.set_state(_make_state(upper.copy()))
    np.testing.assert_allclose(kin.get_state()[:n], upper, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------- FK / 雅可比


@pytest.mark.parametrize("side", SIDES)
def test_fk_matches_independent_urdf(side: str) -> None:
    """核心断言：双侧 TCP 位姿与独立整机 URDF FK 对照（atol=1e-8）。

    对照器 tests/r1d_reference.py 只用 numpy+stdlib 逐字解析
    assets/unitree_r1d/r1d.urdf（世界→mount→升降两级→腰→该侧 7 关节
    →r1d_flange→球拍），环境链与场景数值独立累乘。若模型为单侧生成后
    镜像改名、或遗漏肩部 ±0.26187 安装角，本测试必然失配。
    """
    bundle = _default_bundle(side)
    kin = _kinematics_class()(bundle)
    fk = r1d_reference.R1DUrdfFk(side)

    postures = [
        np.zeros(bundle.profile.ndof),
        np.asarray(bundle.parameters.init_q, dtype=np.float64),
        *_random_postures(bundle, FK_RANDOM_POSTURES, seed=42, margin_ratio=FK_MARGIN_RATIO),
    ]
    assert len(postures) == 2 + FK_RANDOM_POSTURES

    for q in postures:
        kin.set_state(_make_state(q))
        pos, rot = kin.tcp_pose()
        ref_pos, ref_rot = fk.tcp_pose(q)
        np.testing.assert_allclose(pos, ref_pos, atol=1e-8)
        np.testing.assert_allclose(rot, ref_rot, atol=1e-8)


@pytest.mark.parametrize("side", SIDES)
def test_translation_and_rotation_jacobians(side: str) -> None:
    """解析雅可比与独立 URDF FK 中心差分对照（ε=1e-6 rad，离限位 ≥1e-3）。"""
    bundle = _default_bundle(side)
    kin = _kinematics_class()(bundle)
    fk = r1d_reference.R1DUrdfFk(side)
    lower = np.asarray(bundle.parameters.q_lower, dtype=np.float64)
    upper = np.asarray(bundle.parameters.q_upper, dtype=np.float64)

    postures = [
        np.asarray(bundle.parameters.init_q, dtype=np.float64),
        *_random_postures(
            bundle, JAC_RANDOM_POSTURES, seed=42, margin_ratio=JAC_MARGIN_RATIO
        ),
    ]
    for q in postures:
        # 前置条件：姿态离上下限 ≥1e-3（保证 q±ε 的对照 FK 数值稳定）
        assert np.min(np.minimum(q - lower, upper - q)) >= 1e-3

        kin.set_state(_make_state(q))
        jacp, jacr = kin.tcp_jacobian()
        assert jacp.shape == (3, 7)
        assert jacr.shape == (3, 7)
        ref_jacp, ref_jacr = fk.tcp_jacobian(q, eps=1e-6)
        np.testing.assert_allclose(jacp, ref_jacp, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(jacr, ref_jacr, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------- 合成乱序模型


_SYNTHETIC_XML = """<mujoco model="synthetic_disordered">
  <compiler angle="radian" />
  <worldbody>
    <body name="carrier" pos="0.1 0.2 0.3">
      <freejoint />
      <geom name="carrier_geom" type="box" size="0.05 0.05 0.05" contype="0" conaffinity="0" />
      <body name="c_body" pos="0.3 0 0">
        <joint name="joint_C" type="hinge" axis="0 0 1" range="-1.0 1.0" />
        <geom name="c_geom" type="sphere" size="0.02" contype="0" conaffinity="0" />
        <body name="a_body" pos="0.2 0.1 0">
          <joint name="joint_A" type="hinge" axis="0 1 0" range="-1.5 1.5" />
          <geom name="a_geom" type="sphere" size="0.02" contype="0" conaffinity="0" />
          <site name="probe_tcp" pos="0.1 0 0.05" size="0.005" />
        </body>
      </body>
    </body>
    <body name="slider" pos="0 0 0.5">
      <joint name="joint_B" type="slide" axis="1 0 0" range="-0.5 0.5" />
      <geom name="slider_geom" type="box" size="0.02 0.02 0.02" contype="0" conaffinity="0" />
    </body>
  </worldbody>
  <actuator>
    <motor name="act_joint_B" joint="joint_B" gear="1" />
    <motor name="act_joint_C" joint="joint_C" gear="1" />
    <motor name="act_joint_A" joint="joint_A" gear="1" />
  </actuator>
</mujoco>
"""


def _synthetic_bundle(tmp_path: Path, *, held_q: float = 0.0) -> RobotModelBundle:
    """构造「free joint 前置 + 执行器乱序」的合成 bundle（保持关节 joint_B）。

    模型 qpos 布局：free joint 占 0..6，joint_C=7，joint_A=8，joint_B=9；
    主动序 (joint_A, joint_C) 的绑定地址为 (8, 7)——非 0..n-1 连续段。
    """
    xml_path = tmp_path / "synthetic_disordered.xml"
    xml_path.write_text(_SYNTHETIC_XML, encoding="utf-8")
    model = load_mujoco_model(xml_path)
    assert model.nq == 10 and model.nv == 9 and model.nu == 3

    profile = RobotProfile(
        schema_version=1,
        robot_id="synthetic_probe",
        model_path=xml_path,
        active_joint_names=("joint_A", "joint_C"),
        actuator_names=("act_joint_A", "act_joint_C"),
        held_joint_names=("joint_B",),
        tcp_site_name="probe_tcp",
    )
    binding = bind_robot_model(profile, model)
    # 布局前置断言：主动地址不是 0..n-1；执行器 id 与声明序不一致（乱序）
    assert binding.qpos_indices == (8, 7)
    assert binding.qvel_indices == (7, 6)
    assert binding.actuator_ids != tuple(sorted(binding.actuator_ids))

    joint_ids = list(binding.joint_ids)
    params = RobotRuntimeParameters(
        robot_id="synthetic_probe",
        joint_names=("joint_A", "joint_C"),
        dt=0.005,
        init_q=np.array([0.0, 0.0]),
        held_q=np.array([held_q]),
        kp=None,
        kd=None,
        q_lower=model.jnt_range[joint_ids, 0].copy(),
        q_upper=model.jnt_range[joint_ids, 1].copy(),
        qdot_max=None,
        qddot_max=None,
        torque_max=None,
        max_tcp_speed=None,
        scene=None,
        provisional_fields=(),
        source="tests/test_model_kinematics.py 合成模型",
    )
    return RobotModelBundle(
        profile=profile, parameters=params, model=model, binding=binding
    )


def test_synthetic_disordered_model(tmp_path: Path) -> None:
    """free joint 前置 + 执行器乱序的合成 bundle：按 binding 地址读写。

    主动序 (joint_A, joint_C) 的绑定地址为 (8, 7)——非 0..n-1 连续段，
    验证 ModelKinematics 不依赖"地址恰为 0..6"的 R1D 特殊布局。
    """
    bundle = _synthetic_bundle(tmp_path)

    kin = _kinematics_class()(bundle)
    q = np.array([0.4, -0.7])
    qdot = np.array([0.25, -0.15])
    kin.set_state(_make_state(q, qdot))

    # 内部状态验证（测试特许访问 _data）：写入落在绑定地址，其余地址不被触碰
    data = kin._data
    np.testing.assert_allclose(data.qpos[8], q[0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.qpos[7], q[1], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.qvel[7], qdot[0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(data.qvel[6], qdot[1], atol=0.0, rtol=0.0)
    assert data.qpos[3] == 1.0  # free joint 四元数保持默认 w=1
    assert data.qpos[9] == 0.0  # 保持关节 joint_B（slide）未被写入
    assert data.qvel[8] == 0.0

    # 公共行为：roundtrip + FK 与手动按地址组装一致 + 越界拒绝
    np.testing.assert_allclose(kin.get_state(), _make_state(q, qdot), atol=0.0, rtol=0.0)
    pos, rot = kin.tcp_pose()
    manual_pos, manual_rot = _manual_fk(bundle, q)
    np.testing.assert_allclose(pos, manual_pos, atol=1e-12)
    np.testing.assert_allclose(rot, manual_rot, atol=1e-12)
    with pytest.raises(ValueError):
        kin.set_state(_make_state(np.array([2.0, 0.0])))  # joint_A 超 1.5 上界
    np.testing.assert_allclose(kin.get_state()[:2], q, atol=0.0, rtol=0.0)


def test_held_joint_initialized_from_held_q(tmp_path: Path) -> None:
    """构造时把 held_q 写进保持关节、速度置零；set_state 不改保持关节。

    任务卡要求「初始化保持关节为 held_q、保持速度为零」。R1D 双侧
    held_joint_names 为空，这条只有合成模型能覆盖。
    """
    bundle_held = _synthetic_bundle(tmp_path, held_q=0.35)
    kin = _kinematics_class()(bundle_held)
    data = kin._data
    np.testing.assert_allclose(data.qpos[9], 0.35, atol=0.0, rtol=0.0)
    assert data.qvel[8] == 0.0
    # 主动关节仍是零初态，get_state 只含主动关节
    np.testing.assert_allclose(kin.get_state(), np.zeros(4), atol=0.0, rtol=0.0)

    # set_state 不触碰保持关节（写主动关节前后保持关节值不变）
    q = np.array([0.4, -0.7])
    kin.set_state(_make_state(q, np.array([0.25, -0.15])))
    np.testing.assert_allclose(data.qpos[9], 0.35, atol=0.0, rtol=0.0)
    assert data.qvel[8] == 0.0

    # 保持关节确实进入运动学：slider body 的世界位置随 held_q 变化
    kin_zero = _kinematics_class()(_synthetic_bundle(tmp_path, held_q=0.0))
    slider_id = mujoco.mj_name2id(
        bundle_held.model, mujoco.mjtObj.mjOBJ_BODY, "slider"
    )
    assert slider_id >= 0
    assert (
        abs(float(kin._data.xpos[slider_id][0]) - float(kin_zero._data.xpos[slider_id][0]))
        > 1e-6
    )


# ---------------------------------------------------------------- 实例隔离


@pytest.mark.parametrize("side", SIDES)
def test_separate_data_instances(side: str) -> None:
    """每实例自有 MjData：另一实例 set_state 不影响本实例的状态与 FK。"""
    bundle = _default_bundle(side)
    kin_cls = _kinematics_class()
    kin1 = kin_cls(bundle)
    kin2 = kin_cls(bundle)

    q_a = np.asarray(bundle.parameters.init_q, dtype=np.float64)
    q_b = _random_postures(bundle, 1, seed=123, margin_ratio=FK_MARGIN_RATIO)[0]

    kin1.set_state(_make_state(q_a))
    pos_a, rot_a = kin1.tcp_pose()
    kin2.set_state(_make_state(q_b))
    pos_b, _ = kin2.tcp_pose()

    # 两实例结果不同（姿态确实不同），且 kin1 不受 kin2 写入影响
    assert not np.allclose(pos_a, pos_b, atol=1e-6)
    np.testing.assert_allclose(kin1.get_state()[:7], q_a, atol=0.0, rtol=0.0)
    pos_a2, rot_a2 = kin1.tcp_pose()
    np.testing.assert_allclose(pos_a2, pos_a, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(rot_a2, rot_a, atol=0.0, rtol=0.0)
