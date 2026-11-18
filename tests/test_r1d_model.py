"""R1D 单臂场景模型工厂交付物测试（阶段 2 · Task 2.5b）。

核心断言组（任务卡 7 组 + 防漂移三件套，左右双侧参数化）：
1. ``test_r1d_model_contract``：load_robot_model 产出的 model/binding/
   parameters 同源；nq=nv=nu=7；TCP 可解析；执行器仅名称绑定；
   不支持的 robot_id 与缺失 scene 明确拒绝。
2. ``test_base_translation_and_rotation``：基座 z +0.10 m 时 TCP 增量
   精确等于 [0,0,0.10] 且姿态不变；基座绕 Z 旋转 90° 按刚体变换。
3. ``test_racket_transform_and_inertia``：拍面偏移变化符合 r1d_flange
   旋转；圆盘惯量 Izz=0.5mr²、Ixx=Iyy=m(3r²+(2h)²)/12；单一 geom。
4. ``test_parameter_change_does_not_mutate_other_bundle``：xml_text
   参数化每次独立实例，磁盘模板与其他 bundle 不被改写。
5. ``test_model_range_populates_parameters``：q_lower/q_upper 由模型
   jnt_range 填充且与审计报告 §2 表 URDF 限位逐项一致（含右腕俯仰
   +0.5326 不对称）；输入对象不被改写。
6. ``test_initial_pose_has_no_nonadjacent_penetration``：独立诊断模型
   编译期打开环境链/臂/球拍/torso 接触掩码（仅排除相邻/连接与结构
   紧凑单元对），初态无非相邻穿透；偏心人为重叠负例证明检测有效。
7. ``test_side_specific_geometry``：左右 bundle 的肩挂接点 y 坐标符号
   相反、幅值相等（0.085688 m），证明两侧模板确从各自 URDF 链导出。

防漂移三件套：重建字节一致（双侧）、直接加载磁盘模板可绑定 TCP、
模板默认值=runtime YAML。
"""

from __future__ import annotations

import dataclasses
import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
import pytest

from src.robot.model_binding import ModelBinding, bind_robot_model
from src.robot.robot_profile import RobotProfile, load_robot_profile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    load_runtime_parameters,
)
from src.utils.mujoco_loader import load_mujoco_model

if TYPE_CHECKING:
    # 仅类型标注使用：模块本身属被测交付物，运行期经 _model_factory 惰性导入
    from src.robot.model_factory import RobotModelBundle

# 仓库根目录与双侧关键交付物路径
REPO_ROOT = Path(__file__).resolve().parent.parent
SIDES = ("left", "right")
PROFILE_PATHS = {
    side: REPO_ROOT / "configs" / "robots" / f"r1d_{side}.yaml" for side in SIDES
}
RUNTIME_PATHS = {
    side: REPO_ROOT / "configs" / "robots" / f"r1d_runtime_{side}.yaml"
    for side in SIDES
}
TEMPLATE_PATHS = {
    side: REPO_ROOT / "src" / "robot" / f"r1d_model_{side}.xml" for side in SIDES
}
ARM_FRAGMENTS = {
    side: REPO_ROOT / "src" / "robot" / f"r1d_arm_{side}.xml" for side in SIDES
}
ENV_FRAGMENT = REPO_ROOT / "src" / "robot" / "r1d_env.xml"
ASSET_FRAGMENT = REPO_ROOT / "src" / "robot" / "r1d_assets.xml"
URDF_PATH = REPO_ROOT / "assets" / "unitree_r1d" / "r1d.urdf"

# 审计报告 §2：该侧 7 个 URDF 关节原名与语义顺序
# （肩俯仰/肩侧摆/肩偏航/肘/腕滚转/腕俯仰/腕偏航）
JOINT_NAMES: dict[str, tuple[str, ...]] = {
    "left": (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ),
    "right": (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
}

# 审计报告 §2 的 URDF 限位表（rad，逐字抄录；★右腕俯仰上界 +0.5326 不对称，
# 官方疑似笔误不修正；右肩侧摆下界 -2.47849/上界 0.2268 均逐字）
AUDIT_LIMITS: dict[str, tuple[tuple[float, float], ...]] = {
    "left": (
        (-3.1416, 2.0944),
        (-0.22689, 2.4784),
        (-1.9199, 1.9199),
        (-0.97564, 2.1852),
        (-1.9199, 1.9199),
        (-0.5236, 0.5236),
        (-1.5708, 1.5708),
    ),
    "right": (
        (-3.1416, 2.0944),
        (-2.47849, 0.2268),
        (-1.9199, 1.9199),
        (-0.97564, 2.1852),
        (-1.9199, 1.9199),
        (-0.5236, 0.5326),
        (-1.5708, 1.5708),
    ),
}

# 审计报告 §2/§3：r1d_flange = 该侧 dex1_base_joint origin（wrist_yaw_link 局部系）
FLANGE_POS: dict[str, np.ndarray] = {
    "left": np.array([0.0485, -0.003, 0.02]),
    "right": np.array([0.0485, 0.003, 0.02]),
}

# 审计报告 §4：肩挂接点 y 分量（waist_yaw_link 系，左右符号相反）
SHOULDER_MOUNT_Y = 0.085688

# 环境冻结链 body（相邻排除之外的诊断对象）
ENV_CHAIN_BODIES = (
    "base_link",
    "lift_mount_link",
    "lift_stage1_link",
    "lift_stage2_link",
    "pelvis_link",
    "waist_yaw_link",
)


def _arm_chain_bodies(side: str) -> tuple[str, ...]:
    """该侧臂链 7 个连杆 body 名（URDF 子 link 原名，注意首级大写 L）。"""
    return (
        f"{side}_shoulder_pitch_Link",
        f"{side}_shoulder_roll_link",
        f"{side}_shoulder_yaw_link",
        f"{side}_elbow_link",
        f"{side}_wrist_roll_link",
        f"{side}_wrist_pitch_link",
        f"{side}_wrist_yaw_link",
    )


def _diagnostic_bodies(side: str) -> tuple[str, ...]:
    """独立诊断模型中打开接触掩码的 body（地面保持关闭，不参与诊断）。"""
    return (*ENV_CHAIN_BODIES, *_arm_chain_bodies(side), "r1d_racket", "r1d_torso")


# 绕 Z / 绕 X 的 90° 旋转矩阵（刚体变换断言用）
_RZ90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_RX90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def _model_factory() -> Any:
    """惰性导入模型工厂；缺失即判功能缺失（保持红灯为逐测试失败）。"""
    try:
        from src.robot import model_factory
    except ImportError as exc:
        pytest.fail(f"模型工厂模块缺失: {exc}")
    return model_factory


def _load_profile(side: str) -> RobotProfile:
    """加载该侧描述（configs/robots/r1d_{side}.yaml）。"""
    return load_robot_profile(PROFILE_PATHS[side])


def _load_parameters(side: str) -> RobotRuntimeParameters:
    """加载该侧运行参数（configs/robots/r1d_runtime_{side}.yaml）。"""
    return load_runtime_parameters(RUNTIME_PATHS[side], _load_profile(side))


def _default_bundle(side: str) -> RobotModelBundle:
    """以该侧默认描述 + 默认运行参数组装 bundle。"""
    factory = _model_factory()
    profile = _load_profile(side)
    return factory.load_robot_model(profile, _load_parameters(side))


def _bundle_with_scene(side: str, **overrides: Any) -> RobotModelBundle:
    """以该侧默认参数为基础覆盖 scene 字段并组装 bundle。"""
    factory = _model_factory()
    profile = _load_profile(side)
    params = _load_parameters(side)
    assert params.scene is not None
    patched = dataclasses.replace(
        params, scene=dataclasses.replace(params.scene, **overrides)
    )
    return factory.load_robot_model(profile, patched)


def _fk_site(
    model: mujoco.MjModel, binding: ModelBinding, qpos: np.ndarray, site_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """以给定关节角做前向运动学，返回 site 的世界位置与旋转矩阵。"""
    data = mujoco.MjData(model)
    data.qpos[list(binding.qpos_indices)] = qpos
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    assert sid != -1, f"缺少 site: {site_name}"
    return (
        data.site_xpos[sid].copy(),
        data.site_xmat[sid].reshape(3, 3).copy(),
    )


def _find_body(root: ET.Element, name: str) -> ET.Element:
    """在模板中按名称查找 body 元素，找不到即判定失败。"""
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    pytest.fail(f"模板中缺少 body: {name}")


def _vec(text: str) -> np.ndarray:
    """把空格分隔的数值串解析为 float 数组。"""
    return np.array([float(token) for token in text.split()])


def _penetration_pairs(side: str, bundle: RobotModelBundle) -> list[tuple[float, str, str]]:
    """独立诊断模型：打开环境/臂/球拍/torso 接触掩码后的非排除接触列表。

    生产模型遵守自碰撞关闭约定（全部 geom contype=0 conaffinity=0）。
    此处在**独立诊断模型**（模板 XML 副本上编译期注入接触掩码）中做
    诊断：MuJoCo 3.9 实测表明运行期修改 MjModel 的 contype/conaffinity
    不会重建碰撞对，必须经 XML 文本在编译期打开；诊断模型经 xml_text
    通道加载，磁盘模板与其他 bundle 不受影响。场景参数（含负例的
    torso 移位）经工厂 _apply_scene_parameters 同一写入入口注入，
    与生产组装共用一套数值防漂移。相邻/连接/结构紧凑单元对已由模板
    <contact><exclude> 排除，出现的接触均为非排除对。
    """
    factory = _model_factory()
    root = ET.fromstring(bundle.profile.model_path.read_bytes())
    factory._apply_scene_parameters(root, bundle.parameters)
    masked = 0
    for body in root.iter("body"):
        if body.attrib.get("name") in _diagnostic_bodies(side):
            for geom in body.findall("geom"):
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
                masked += 1
    assert masked > 0, "诊断掩码未注入任何 geom"
    model = load_mujoco_model(
        bundle.profile.model_path, xml_text=ET.tostring(root, encoding="unicode")
    )
    data = mujoco.MjData(model)
    data.qpos[list(bundle.binding.qpos_indices)] = bundle.parameters.init_q
    mujoco.mj_forward(model, data)
    pairs: list[tuple[float, str, str]] = []
    for i in range(data.ncon):
        contact = data.contact[i]
        b1 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[int(contact.geom[0])])
        )
        b2 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[int(contact.geom[1])])
        )
        pairs.append((float(contact.dist), str(b1), str(b2)))
    return pairs


# ---------------------------------------------------------------- 任务卡 7 组


@pytest.mark.parametrize("side", SIDES)
def test_r1d_model_contract(side: str) -> None:
    """同源契约：model/binding/parameters 同属一次配置，维数与命名正确。"""
    profile = _load_profile(side)
    params = _load_parameters(side)
    bundle = _model_factory().load_robot_model(profile, params)
    model, binding = bundle.model, bundle.binding

    # 维数契约：qpos=7、qvel=7、actuator=7；无 freejoint/球/活动夹爪/环境关节
    assert model.nq == 7
    assert model.nv == 7
    assert model.nu == 7
    assert model.njnt == 7

    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
    ]
    assert tuple(joint_names) == JOINT_NAMES[side]
    assert all(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in range(model.njnt))

    # 描述与参数同源：身份、关节序、步长一致
    assert profile.robot_id == "unitree_r1d" == bundle.parameters.robot_id
    assert tuple(bundle.parameters.joint_names) == profile.active_joint_names
    assert model.opt.timestep == pytest.approx(params.dt)

    # 绑定地址：主动关节顺序覆盖全部 qpos/qvel，held 为空，TCP 可解析
    assert binding.qpos_indices == tuple(range(7))
    assert binding.qvel_indices == tuple(range(7))
    assert binding.held_qpos_indices == ()
    assert binding.held_qvel_indices == ()
    assert (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, binding.tcp_site_id)
        == "racket_center"
    )

    # r1d_flange 挂在该侧 dex1_base_joint 的官方 origin（wrist_yaw_link 局部系）
    flange_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r1d_flange")
    assert flange_id != -1
    np.testing.assert_allclose(model.site_pos[flange_id], FLANGE_POS[side], atol=1e-12)

    # 环境冻结链存在且不含关节（升降 q_L 已 bake，无运行时等式约束）
    for env_body in ENV_CHAIN_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, env_body)
        assert bid != -1, f"缺少环境 body: {env_body}"
        assert int(model.body_jntnum[bid]) == 0, f"{env_body} 不应含关节"

    # 执行器仅名称绑定：unit-gear、无限值（R1D URDF 能力值未经实机验证，
    # 不配置 ctrlrange/forcerange 为安全限值，N0-9）
    for joint_name, aid in zip(JOINT_NAMES[side], binding.actuator_ids):
        assert (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            == f"torque_{joint_name}"
        )
        assert model.actuator_gear[aid, 0] == pytest.approx(1.0)
        assert int(model.actuator_ctrllimited[aid]) == 0
        assert int(model.actuator_forcelimited[aid]) == 0

    # 场景元素：地面 / 固定基座挂载 / 环境链根 / 球拍 / 简化机身存在
    for name, obj_type in (
        ("ground", mujoco.mjtObj.mjOBJ_GEOM),
        ("r1d_mount", mujoco.mjtObj.mjOBJ_BODY),
        ("r1d_racket", mujoco.mjtObj.mjOBJ_BODY),
        ("r1d_torso", mujoco.mjtObj.mjOBJ_BODY),
    ):
        assert mujoco.mj_name2id(model, obj_type, name) != -1, f"缺少 {name}"

    # 非法输入：其他 robot_id 明确不支持；scene 缺失明确报错
    other_profile = dataclasses.replace(profile, robot_id="rm65")
    with pytest.raises(ValueError, match="unitree_r1d"):
        _model_factory().load_robot_model(other_profile, params)
    scene_missing = dataclasses.replace(params, scene=None)
    with pytest.raises(ValueError, match="scene"):
        _model_factory().load_robot_model(profile, scene_missing)


@pytest.mark.parametrize("side", SIDES)
def test_base_translation_and_rotation(side: str) -> None:
    """基座平移只平移 TCP；基座旋转按刚体变换作用于位置与姿态。"""
    params = _load_parameters(side)
    default = _default_bundle(side)
    tcp0, rot0 = _fk_site(default.model, default.binding, params.init_q, "racket_center")

    # 高度 +0.10 m：TCP 增量精确等于 [0,0,0.10]，姿态不变
    lifted = _bundle_with_scene(side, base_position=np.array([0.0, 0.0, 1.1]))
    tcp1, rot1 = _fk_site(lifted.model, lifted.binding, params.init_q, "racket_center")
    np.testing.assert_allclose(tcp1 - tcp0, [0.0, 0.0, 0.10], atol=1e-9)
    np.testing.assert_allclose(rot1, rot0, atol=1e-9)

    # 绕 Z 旋转 90°（基座位置不变）：FK 按刚体变换变化
    rotated = _bundle_with_scene(
        side, base_quaternion=np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    )
    tcp2, rot2 = _fk_site(rotated.model, rotated.binding, params.init_q, "racket_center")
    scene = params.scene
    assert scene is not None
    base = scene.base_position
    np.testing.assert_allclose(tcp2, base + _RZ90 @ (tcp0 - base), atol=1e-9)
    np.testing.assert_allclose(rot2, _RZ90 @ rot0, atol=1e-9)


@pytest.mark.parametrize("side", SIDES)
def test_racket_transform_and_inertia(side: str) -> None:
    """拍面偏移符合 flange 旋转；圆盘惯量与质量符合闭式公式。"""
    params = _load_parameters(side)
    scene = params.scene
    assert scene is not None

    default = _default_bundle(side)
    tcp0, _ = _fk_site(default.model, default.binding, params.init_q, "racket_center")
    _, flange_rot = _fk_site(default.model, default.binding, params.init_q, "r1d_flange")

    # 拍面偏移变化符合 flange 旋转：ΔTCP = R_flange @ Δracket_position
    shifted = _bundle_with_scene(
        side, racket_position=scene.racket_position + np.array([0.05, 0.0, 0.0])
    )
    tcp1, _ = _fk_site(shifted.model, shifted.binding, params.init_q, "racket_center")
    np.testing.assert_allclose(
        tcp1 - tcp0, flange_rot @ np.array([0.05, 0.0, 0.0]), atol=1e-9
    )

    # 拍面姿态变化 = flange 姿态 ∘ 拍面局部旋转（绕局部 X 90°）
    tilted = _bundle_with_scene(
        side, racket_quaternion=np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])
    )
    _, rot2 = _fk_site(tilted.model, tilted.binding, params.init_q, "racket_center")
    np.testing.assert_allclose(rot2, flange_rot @ _RX90, atol=1e-9)

    # 圆盘惯量闭式公式（默认与自定义参数均须满足）；显式 inertial 不双重计质量。
    # 球拍现含 3 个 geom：racket_face 碰撞代理 + 视觉拍面 + 拍柄（后两者 contype=0）
    bundles = [
        default,
        _bundle_with_scene(
            side, racket_mass=0.5, racket_radius=0.15, racket_half_thickness=0.01
        ),
    ]
    for bundle in bundles:
        sc = bundle.parameters.scene
        assert sc is not None
        m, r, h = sc.racket_mass, sc.racket_radius, sc.racket_half_thickness
        rid = mujoco.mj_name2id(bundle.model, mujoco.mjtObj.mjOBJ_BODY, "r1d_racket")
        assert rid != -1
        np.testing.assert_allclose(bundle.model.body_mass[rid], m, atol=1e-12)
        ixx = m * (3.0 * r**2 + (2.0 * h) ** 2) / 12.0
        izz = 0.5 * m * r**2
        np.testing.assert_allclose(
            bundle.model.body_inertia[rid], [ixx, ixx, izz], atol=1e-12
        )
        assert int(bundle.model.body_geomnum[rid]) == 3


@pytest.mark.parametrize("side", SIDES)
def test_parameter_change_does_not_mutate_other_bundle(side: str) -> None:
    """参数化加载互不影响：其他 bundle、输入参数与磁盘模板均不被改写。"""
    template_before = TEMPLATE_PATHS[side].read_bytes()
    params = _load_parameters(side)
    scene = params.scene
    assert scene is not None

    base_a = _default_bundle(side)
    tcp_a, _ = _fk_site(base_a.model, base_a.binding, params.init_q, "racket_center")
    racket_a = mujoco.mj_name2id(base_a.model, mujoco.mjtObj.mjOBJ_BODY, "r1d_racket")
    mass_a = float(base_a.model.body_mass[racket_a])
    q_lower_a = np.array(base_a.parameters.q_lower)

    # 以另一组参数构建 bundle B（基座抬升 + 球拍加重 + torso 移位）
    modified = _bundle_with_scene(
        side,
        base_position=scene.base_position + np.array([0.0, 0.0, 0.10]),
        racket_mass=0.5,
        racket_radius=0.15,
        torso_position=np.array([0.0, 0.0, 0.0]),
    )
    racket_b = mujoco.mj_name2id(modified.model, mujoco.mjtObj.mjOBJ_BODY, "r1d_racket")
    assert float(modified.model.body_mass[racket_b]) == pytest.approx(0.5)

    # bundle A 不受 B 构建影响（模型实例独立、参数对象独立）
    tcp_a2, _ = _fk_site(base_a.model, base_a.binding, params.init_q, "racket_center")
    np.testing.assert_array_equal(tcp_a2, tcp_a)
    assert float(base_a.model.body_mass[racket_a]) == mass_a
    np.testing.assert_array_equal(base_a.parameters.q_lower, q_lower_a)

    # 输入参数对象不被填充改写；磁盘模板字节不变
    assert params.q_lower is None
    assert TEMPLATE_PATHS[side].read_bytes() == template_before


@pytest.mark.parametrize("side", SIDES)
def test_model_range_populates_parameters(side: str) -> None:
    """q_lower/q_upper 从模型 jnt_range 填充，与审计 §2 表逐项一致。"""
    profile = _load_profile(side)
    params = _load_parameters(side)
    assert params.q_lower is None and params.q_upper is None

    bundle = _model_factory().load_robot_model(profile, params)
    filled = bundle.parameters
    assert filled is not params

    # 与模型 jnt_range 一致（按绑定关节顺序逐行读取）
    joint_ids = list(bundle.binding.joint_ids)
    np.testing.assert_allclose(
        filled.q_lower, bundle.model.jnt_range[joint_ids, 0], atol=1e-12
    )
    np.testing.assert_allclose(
        filled.q_upper, bundle.model.jnt_range[joint_ids, 1], atol=1e-12
    )

    # 与审计报告 §2 的 URDF 限位逐项对照（含右腕俯仰 +0.5326 不对称）
    for k, (lower, upper) in enumerate(AUDIT_LIMITS[side]):
        np.testing.assert_allclose(filled.q_lower[k], lower, atol=1e-8)
        np.testing.assert_allclose(filled.q_upper[k], upper, atol=1e-8)

    # 与原始 URDF 独立交叉验证（同串解析须精确相等）
    urdf_root = ET.parse(URDF_PATH).getroot()
    urdf_joints = {
        joint.attrib["name"]: joint.find("limit")
        for joint in urdf_root.findall("joint")
    }
    for k, name in enumerate(profile.active_joint_names):
        limit = urdf_joints[name]
        assert limit is not None
        np.testing.assert_allclose(
            filled.q_lower[k], float(limit.attrib["lower"]), atol=1e-8
        )
        np.testing.assert_allclose(
            filled.q_upper[k], float(limit.attrib["upper"]), atol=1e-8
        )

    # 其余字段保留：身份、来源、临时字段标记；数组只读
    assert filled.source == params.source
    assert filled.provisional_fields == params.provisional_fields
    with pytest.raises(ValueError):
        filled.q_lower[0] = 0.0


@pytest.mark.parametrize("side", SIDES)
def test_explicit_joint_limits_must_match_model(side: str) -> None:
    """显式 q_lower/q_upper 必须与模型一致：一致接受，收紧/放宽均拒绝。

    模型是位置限位的唯一权威。曾经的实现无条件覆盖入参，使调用方
    在 YAML 里收紧的安全限位在校验通过后被静默放宽回 URDF 全范围。
    """
    factory = _model_factory()
    profile = _load_profile(side)
    base = _load_parameters(side)
    reference = factory.load_robot_model(profile, base)
    lower = np.array(reference.parameters.q_lower)
    upper = np.array(reference.parameters.q_upper)

    # 显式给出的值与模型一致：接受，且数值不变
    same = dataclasses.replace(base, q_lower=lower.copy(), q_upper=upper.copy())
    accepted = factory.load_robot_model(profile, same)
    np.testing.assert_allclose(accepted.parameters.q_lower, lower, atol=1e-12)
    np.testing.assert_allclose(accepted.parameters.q_upper, upper, atol=1e-12)

    # 收紧（下界上移 1e-3）：拒绝，不得静默放宽
    tightened_lower = lower.copy()
    tightened_lower[0] += 1e-3
    tightened = dataclasses.replace(
        base, q_lower=tightened_lower, q_upper=upper.copy()
    )
    with pytest.raises(ValueError, match="q_lower"):
        factory.load_robot_model(profile, tightened)

    # 放宽（上界上移 1e-3）：拒绝，不得静默接受越权值
    widened_upper = upper.copy()
    widened_upper[0] += 1e-3
    widened = dataclasses.replace(
        base, q_lower=lower.copy(), q_upper=widened_upper
    )
    with pytest.raises(ValueError, match="q_upper"):
        factory.load_robot_model(profile, widened)


@pytest.mark.parametrize("side", SIDES)
def test_initial_pose_has_no_nonadjacent_penetration(side: str) -> None:
    """独立诊断模型：初态无非相邻穿透；人为重叠负例证明检测有效。"""
    bundle = _default_bundle(side)
    contacts = _penetration_pairs(side, bundle)
    for dist, b1, b2 in contacts:
        assert dist >= -1e-4, f"非相邻部件穿透: {b1} <-> {b2}, dist={dist}"

    # 负例：把简化 torso 移到球拍中心附近（偏心重叠，避免完全同心——
    # 同心重叠是凸-凸碰撞检测的数值退化配置，MuJoCo 不产出接触）
    scene = bundle.parameters.scene
    assert scene is not None
    tcp, _ = _fk_site(
        bundle.model, bundle.binding, bundle.parameters.init_q, "racket_center"
    )
    overlapped = _bundle_with_scene(
        side, torso_position=tcp + np.array([0.10, 0.0, 0.0]) - scene.base_position
    )
    bad = _penetration_pairs(side, overlapped)
    assert any(dist < -1e-4 for dist, _, _ in bad), f"人为重叠未检出: {bad}"


def test_side_specific_geometry() -> None:
    """左右肩挂接点 y 坐标符号相反、幅值相等，证明双侧独立导出非镜像改名。"""
    shoulder_y: dict[str, float] = {}
    for side in SIDES:
        bundle = _default_bundle(side)
        data = mujoco.MjData(bundle.model)
        data.qpos[list(bundle.binding.qpos_indices)] = bundle.parameters.init_q
        mujoco.mj_forward(bundle.model, data)
        bid = mujoco.mj_name2id(
            bundle.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_pitch_Link"
        )
        assert bid != -1
        shoulder_y[side] = float(data.xpos[bid][1])

    np.testing.assert_allclose(shoulder_y["left"], SHOULDER_MOUNT_Y, atol=1e-9)
    np.testing.assert_allclose(shoulder_y["right"], -SHOULDER_MOUNT_Y, atol=1e-9)


# ---------------------------------------------------------------- 防漂移 3 测试


@pytest.mark.parametrize("side", SIDES)
def test_template_rebuild_is_byte_deterministic(side: str, tmp_path: Path) -> None:
    """相同输入两次生成字节一致且与交付物一致；跨目录输出重定位 mesh 路径。"""
    factory = _model_factory()
    params = _load_parameters(side)

    # 同目录重建（mesh 相对路径与仓库模板一致）→ 字节一致
    tmp_scene = REPO_ROOT / "src" / "robot" / f"_tmp_r1d_model_{side}_rebuild.xml"
    try:
        factory.write_r1d_model_template(
            params,
            tmp_scene,
            side=side,
            arm_path=ARM_FRAGMENTS[side],
            env_path=ENV_FRAGMENT,
            assets_path=ASSET_FRAGMENT,
        )
        first = tmp_scene.read_bytes()
        factory.write_r1d_model_template(
            params,
            tmp_scene,
            side=side,
            arm_path=ARM_FRAGMENTS[side],
            env_path=ENV_FRAGMENT,
            assets_path=ASSET_FRAGMENT,
        )
        second = tmp_scene.read_bytes()
    finally:
        tmp_scene.unlink(missing_ok=True)
    assert first == second
    assert first == TEMPLATE_PATHS[side].read_bytes()

    # 跨目录输出：mesh 引用重定位后仍指向同一资产内容——同盘符为相对
    # 回引（解析回仓库资产文件），跨盘符为重定位副本（导出目录 meshes/
    # 下按名逐字节等于片段资产）
    relocated = tmp_path / f"r1d_model_{side}.xml"
    factory.write_r1d_model_template(
        params,
        relocated,
        side=side,
        arm_path=ARM_FRAGMENTS[side],
        env_path=ENV_FRAGMENT,
        assets_path=ASSET_FRAGMENT,
    )
    rebuilt_refs = {
        mesh.attrib["file"]: (relocated.parent / mesh.attrib["file"]).resolve()
        for mesh in ET.fromstring(relocated.read_bytes()).iter("mesh")
    }
    fragment_refs = {
        mesh.attrib["file"]: (ASSET_FRAGMENT.parent / mesh.attrib["file"]).resolve()
        for mesh in ET.fromstring(ASSET_FRAGMENT.read_bytes()).iter("mesh")
    }
    fragment_by_name = {p.name: p for p in fragment_refs.values()}
    assert len(rebuilt_refs) == len(fragment_refs) == 20
    for target in rebuilt_refs.values():
        assert target.is_file(), f"重定位后的 mesh 引用应存在: {target}"
        assert target.read_bytes() == fragment_by_name[target.name].read_bytes(), (
            f"mesh 内容应与片段资产一致: {target.name}"
        )


@pytest.mark.parametrize("side", SIDES)
def test_export_relocates_meshes_when_relative_unavailable(
    side: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """相对化不可用（如跨盘符）时 mesh 重定位到导出目录且模板可实际加载。

    os.path.relpath 跨盘符抛 ValueError，此处确定性注入该失败模拟任意
    平台的跨盘导出。退回仓库绝对路径（含非 ASCII）会被 MuJoCo C 层
    拒绝打开（Error opening file），导出的模板不可加载。
    """

    def _cross_drive_relpath(*_args: object, **_kwargs: object) -> str:
        raise ValueError("path is on mount 'C:', start on mount 'F:'")

    monkeypatch.setattr(os.path, "relpath", _cross_drive_relpath)
    factory = _model_factory()
    params = _load_parameters(side)
    out = tmp_path / f"r1d_model_{side}.xml"
    factory.write_r1d_model_template(
        params,
        out,
        side=side,
        arm_path=ARM_FRAGMENTS[side],
        env_path=ENV_FRAGMENT,
        assets_path=ASSET_FRAGMENT,
    )

    root = ET.fromstring(out.read_bytes())
    files = [mesh.attrib["file"] for mesh in root.iter("mesh")]
    assert len(files) == 20, "资产片段应携带 20 个 mesh 声明"
    assert all(f.startswith("meshes/") for f in files), "应改写为重定位子目录的相对引用"
    exported = {p.name for p in (tmp_path / "meshes").iterdir()}
    assert exported == {Path(f).name for f in files}, "mesh 文件应全部重定位到导出目录"

    # 导出后实际加载：模板与重定位 mesh 自包含，MuJoCo 可编译出 7 维模型
    model = load_mujoco_model(out)
    assert model.nq == 7 and model.nv == 7 and model.nu == 7
    assert model.nmesh == 20


@pytest.mark.skipif(
    Path(tempfile.gettempdir()).drive == REPO_ROOT.resolve().drive,
    reason="需要系统临时目录与仓库位于不同盘符（Windows 多盘环境）",
)
@pytest.mark.parametrize("side", SIDES)
def test_cross_drive_export_loads_directly(side: str) -> None:
    """真跨盘符导出端到端：导出物自包含，不经加载器可直接编译。"""
    factory = _model_factory()
    params = _load_parameters(side)
    out_dir = Path(tempfile.mkdtemp(prefix="r1d_xdrive_"))
    try:
        out = out_dir / f"r1d_model_{side}.xml"
        factory.write_r1d_model_template(
            params,
            out,
            side=side,
            arm_path=ARM_FRAGMENTS[side],
            env_path=ENV_FRAGMENT,
            assets_path=ASSET_FRAGMENT,
        )
        model = mujoco.MjModel.from_xml_path(str(out))
        assert model.nq == 7 and model.nv == 7 and model.nu == 7
        assert model.nmesh == 20
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


@pytest.mark.parametrize("side", SIDES)
def test_template_loads_directly_and_binds_tcp(side: str) -> None:
    """直接加载磁盘模板（不经 xml_text）也能编译并绑定 TCP。"""
    profile = _load_profile(side)
    model = load_mujoco_model(profile.model_path)
    assert model.nq == 7 and model.nv == 7 and model.nu == 7
    binding = bind_robot_model(profile, model)
    assert (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, binding.tcp_site_id)
        == "racket_center"
    )

    # 工厂产物与直接加载结构一致（防裸模板与工厂结果漂移）
    bundle = _default_bundle(side)
    assert bundle.model.nbody == model.nbody
    assert bundle.model.nmesh == model.nmesh
    all_joints = list(range(model.njnt))
    np.testing.assert_array_equal(
        bundle.model.jnt_range[all_joints], model.jnt_range[all_joints]
    )


@pytest.mark.parametrize("side", SIDES)
def test_template_defaults_match_runtime_yaml(side: str) -> None:
    """模板默认值与 r1d_runtime_{side}.yaml 数值真值一致（防裸模板漂移）。"""
    params = _load_parameters(side)
    scene = params.scene
    assert scene is not None
    root = ET.fromstring(TEMPLATE_PATHS[side].read_bytes())

    # 唯一 compiler 且角度单位为 degree（来自 2.5a 片段，模板不另加 compiler）
    compilers = [el for el in root if el.tag == "compiler"]
    assert len(compilers) == 1
    assert compilers[0].attrib.get("angle") == "degree"

    option = root.find("option")
    assert option is not None
    assert float(option.attrib["timestep"]) == pytest.approx(params.dt, abs=1e-15)

    # 固定基座挂载：r1d_mount 承载环境链 + 双侧同位姿
    mount = _find_body(root, "r1d_mount")
    np.testing.assert_allclose(_vec(mount.attrib["pos"]), scene.base_position, atol=1e-12)
    np.testing.assert_allclose(_vec(mount.attrib["quat"]), scene.base_quaternion, atol=1e-12)

    # r1d_flange site：来自 2.5a 臂片段（该侧 dex1_base_joint origin）
    flange = next(s for s in root.iter("site") if s.attrib.get("name") == "r1d_flange")
    np.testing.assert_allclose(_vec(flange.attrib["pos"]), FLANGE_POS[side], atol=1e-12)

    # 球拍：body 挂接 = 法兰 + 拍面中心偏移；惯量/尺寸/质量与参数一致
    racket = _find_body(root, "r1d_racket")
    np.testing.assert_allclose(
        _vec(racket.attrib["pos"]), FLANGE_POS[side] + scene.racket_position, atol=1e-12
    )
    np.testing.assert_allclose(
        _vec(racket.attrib["quat"]), scene.racket_quaternion, atol=1e-12
    )
    racket_geom = racket.find("geom")
    assert racket_geom is not None
    np.testing.assert_allclose(
        _vec(racket_geom.attrib["size"]),
        [scene.racket_radius, scene.racket_half_thickness],
        atol=1e-12,
    )
    inertial = racket.find("inertial")
    assert inertial is not None
    m, r, h = scene.racket_mass, scene.racket_radius, scene.racket_half_thickness
    assert float(inertial.attrib["mass"]) == pytest.approx(m, abs=1e-12)
    ixx = m * (3.0 * r**2 + (2.0 * h) ** 2) / 12.0
    izz = 0.5 * m * r**2
    np.testing.assert_allclose(
        _vec(inertial.attrib["diaginertia"]), [ixx, ixx, izz], atol=1e-12
    )

    # 简化机身：相对固定基座定位
    torso = _find_body(root, "r1d_torso")
    np.testing.assert_allclose(_vec(torso.attrib["pos"]), scene.torso_position, atol=1e-12)
    torso_geom = torso.find("geom")
    assert torso_geom is not None
    np.testing.assert_allclose(
        _vec(torso_geom.attrib["size"]), scene.torso_half_size, atol=1e-12
    )

    # 执行器命名契约：七个 motor 仅 gear=1，无 ctrlrange/forcerange
    actuator = root.find("actuator")
    assert actuator is not None
    motors = list(actuator)
    assert [motor.attrib["name"] for motor in motors] == [
        f"torque_{name}" for name in JOINT_NAMES[side]
    ]
    for joint_name, motor in zip(JOINT_NAMES[side], motors):
        assert motor.attrib.get("joint") == joint_name
        assert motor.attrib.get("gear") == "1"
        assert "ctrlrange" not in motor.attrib
        assert "forcerange" not in motor.attrib
