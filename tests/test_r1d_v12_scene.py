"""T2 验收：R1D V12 完整场景、RacketProfile 参数化与旧 RM65 场地一致性。

本文件是 Task 2（R1D V12 parity 施工 · 场景/装配/旧场地恢复）的正式验收
测试，断言的是**正式契约**而不是实现细节：

- 完整整机外观（头/双臂/底座/轮脚轮/升降）与右臂恰 7 主动 DOF；
- 左臂为静态刚体（无关节、``mj_step`` 后不漂）；
- 底座落地高度由轮/脚轮几何推导（0.155 m）、无穿地支撑柱、无初始穿透；
- ``SceneParameters`` 显式升降位移 + 固定关节姿态：内存重烘、连续加载
  不串值、不重写版本控制中的默认模板；
- 旧 RM65 场地（材质/纹理/地面/10 条白线/单灯）与球物理（dt/重力/球
  质量半径/接触参数）与 ``src/robot/rm65_model.xml`` 真实 MJCF 属性一致；
  ``legacy_v1`` profile（默认、旧 runtime 不带 court_profile）下仍不得出现
  ITF 标准球场/球网/四灯/baseline 元素；ITF 12 条标准线走
  ``itf_2026_v1`` 显式 profile，由 ``tests/test_court_scene_integration.py``
  验收（本文件只锁 legacy 默认口径）；
- 两套球拍档案（``rm65_legacy`` 与 ``blade100_v9_reference``）独立核对
  TCP、拍面法向（局部 +X）、碰撞中心、法兰安装刚体变换、总质量/COM/
  惯量与连接件只计一次；框/柄/面 geom 标签互不相同；
- 改球拍长度或安装姿态后，裸模型、带球执行场景、执行环境与规划环境
  的 TCP/碰撞中心/法向同步；
- ``load_robot_model`` 省略 ``racket_profile``（None）保持旧 API 语义。

独立对照来源：``tests/r1d_reference.py`` 的 R1D 整机 URDF FK（只解析官方
URDF + 输入位姿，不复用被测生成器）；球拍期望值由 ``RacketProfile`` 档案
数据 + 本文档内的四元数/平行轴公式独立推导。
"""

from __future__ import annotations

import dataclasses
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
import pytest

import tests.r1d_reference as r1d_reference
from src.ilqt.profile_planning_env import ProfilePlanningEnv
from src.robot.model_factory import load_robot_model
from src.robot.racket_profile import (
    LEGACY_RACKET_PROFILE_PATH,
    REFERENCE_RACKET_PROFILE_PATH,
    RacketProfile,
    load_racket_profile,
)
from src.robot.robot_profile import RobotProfile, load_robot_profile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    load_runtime_parameters,
)
from src.sim.bound_robot_env import BoundRobotEnv
from src.sim.profile_sim_env import ProfileSimEnv
from src.sim.sim_config import SimTaskConfig, load_sim_task_config
from src.sim.tennis_scene import TennisSceneBundle, build_tennis_scene
from src.utils.mujoco_loader import load_mujoco_model

# 仓库根目录（tests/ 的父目录）
REPO_ROOT = Path(__file__).resolve().parent.parent

# 本任务验收侧别：右臂为主动臂，左臂为静态渲染臂
SIDE = "right"
PASSIVE_SIDE = "left"

# 真实配置路径（白名单内的正式文件）
_PROFILE_YAML = REPO_ROOT / "configs" / "robots" / f"r1d_{SIDE}.yaml"
_V12_RUNTIME_YAML = REPO_ROOT / "configs" / "robots" / f"r1d_v12_runtime_{SIDE}.yaml"
_LEGACY_RUNTIME_YAML = (
    REPO_ROOT / "configs" / "robots" / f"r1d_sim_runtime_{SIDE}.yaml"
)
_TASK_YAML = REPO_ROOT / "configs" / f"r1d_sim_{SIDE}.yaml"
_TEMPLATE_PATH = REPO_ROOT / "src" / "robot" / f"r1d_model_{SIDE}.xml"
_RM65_PATH = REPO_ROOT / "src" / "robot" / "rm65_model.xml"

# 球拍档案（T1 交付，本任务只消费）
_BLADE_PROFILE_PATH = REFERENCE_RACKET_PROFILE_PATH
_RM65_RACKET_PROFILE_PATH = LEGACY_RACKET_PROFILE_PATH

# ---------------------------------------------------------------- 期望常量
# 底座落地高度：轮（半径 0.0625，中心局部 z=-0.0925）触地推导
_BASE_LANDING_HEIGHT = 0.155
# 双级升降模板 bake 常数与单级行程上界（r1d_env.xml 契约）
_ENV_BAKED_LIFT = 0.4225
_LIFT_STAGE_RANGE = 0.425
# 环境链 body（base_link → waist_yaw_link）与 head 链接
_ENV_CHAIN_BODIES = (
    "base_link",
    "lift_mount_link",
    "lift_stage1_link",
    "lift_stage2_link",
    "pelvis_link",
    "waist_yaw_link",
)
_HEAD_BODIES = ("head_pitch_Link", "head_yaw_Link")
# 主动/被动臂 7 连杆（URDF 命名：仅 shoulder_pitch 含大写 Link）
_ARM_LINK_SUFFIXES = (
    "shoulder_pitch_Link",
    "shoulder_roll_link",
    "shoulder_yaw_link",
    "elbow_link",
    "wrist_roll_link",
    "wrist_pitch_link",
    "wrist_yaw_link",
)
_ACTIVE_ARM_BODIES = tuple(f"{SIDE}_{suffix}" for suffix in _ARM_LINK_SUFFIXES)
_PASSIVE_ARM_BODIES = tuple(
    f"{PASSIVE_SIDE}_{suffix}" for suffix in _ARM_LINK_SUFFIXES
)
# 轮/脚轮视觉 geom 名与几何（与 RM65 口径无关，R1D 自身落地件）
_WHEEL_GEOMS = ("r1d_wheel_left", "r1d_wheel_right")
_CASTER_GEOMS = tuple(f"r1d_caster_{index}" for index in range(4))
_WHEEL_RADIUS = 0.0625
_CASTER_RADIUS = 0.03245
# 期望网格数：20 个 R1D 整机 mesh + 2 个 head mesh
_EXPECTED_NMESH = 22

# 旧 RM65 场地期望属性（逐字来自 src/robot/rm65_model.xml）
_COURT_GROUND_SIZE = (25.0, 25.0, 0.1)
_COURT_GROUND_RGBA = (0.28, 0.42, 0.22, 1.0)
_COURT_MATERIAL_RGBA = (0.28, 0.42, 0.22, 1.0)
_COURT_TEXTURE_RGB1 = (0.28, 0.42, 0.22)
_COURT_TEXTURE_RGB2 = (0.24, 0.38, 0.20)
_COURT_TEXREPEAT = (8.0, 8.0)
_COURT_REFLECTANCE = 0.15
_COURT_LINES = (
    ((12.0, 0.05, 0.001), (0.0, 0.0, 0.001)),
    ((0.05, 12.0, 0.001), (0.0, 0.0, 0.001)),
    ((0.05, 6.0, 0.001), (-6.0, 0.0, 0.001)),
    ((0.05, 6.0, 0.001), (6.0, 0.0, 0.001)),
    ((0.05, 5.0, 0.001), (-3.0, 0.0, 0.001)),
    ((0.05, 5.0, 0.001), (3.0, 0.0, 0.001)),
    ((3.0, 0.05, 0.001), (-4.5, 0.0, 0.001)),
    ((3.0, 0.05, 0.001), (4.5, 0.0, 0.001)),
    ((0.05, 12.0, 0.001), (0.0, -5.0, 0.001)),
    ((0.05, 12.0, 0.001), (0.0, 5.0, 0.001)),
)
# 恢复的 pre-attempt 旧灯（单灯；R6 要求移除用户尝试新增的四灯）
_LIGHT_POS = (0.0, 0.0, 8.0)
_LIGHT_DIR = (0.0, 0.0, -1.0)
_LIGHT_DIFFUSE = (1.2, 1.25, 1.3)
# 球物理（rm65_model.xml 逐字）
_BALL_RADIUS = 0.033
_BALL_MASS = 0.058
_BALL_RGBA = (0.95, 0.95, 0.1, 1.0)
_BALL_CONTYPE = 16
_BALL_CONAFFINITY = 10
_PAIR_CONDIM = 1
_PAIR_SOLREF = (-5000.0, 0.0)
_PAIR_SOLIMP = (0.9, 0.99, 0.001, 0.5, 2.0)
_OPTION_TIMESTEP = 0.005
_OPTION_GRAVITY = (0.0, 0.0, -9.81)


# ---------------------------------------------------------------- 加载 helper
@lru_cache(maxsize=1)
def _profile() -> RobotProfile:
    """加载并缓存右侧官方机器人描述。"""
    return load_robot_profile(_PROFILE_YAML)


@lru_cache(maxsize=1)
def _v12_parameters() -> RobotRuntimeParameters:
    """加载并缓存 T2 新增的 V12 仿真运行参数（无 legacy racket_* 字段）。"""
    return load_runtime_parameters(_V12_RUNTIME_YAML, _profile())


@lru_cache(maxsize=1)
def _legacy_parameters() -> RobotRuntimeParameters:
    """加载并缓存阶段 3 旧仿真参数（scene 带 legacy racket_* 字段）。"""
    return load_runtime_parameters(_LEGACY_RUNTIME_YAML, _profile())


@lru_cache(maxsize=1)
def _task() -> SimTaskConfig:
    """加载并缓存右侧仿真任务配置。"""
    return load_sim_task_config(_TASK_YAML, _profile())


@lru_cache(maxsize=1)
def _blade_profile() -> RacketProfile:
    """加载并缓存成人参考球拍档案（Blade 100 V9）。"""
    return load_racket_profile(_BLADE_PROFILE_PATH)


@lru_cache(maxsize=1)
def _legacy_racket_profile() -> RacketProfile:
    """加载并缓存 RM65 旧拍档案（忠实导出旧模型）。"""
    return load_racket_profile(_RM65_RACKET_PROFILE_PATH)


@lru_cache(maxsize=1)
def _rm65_model() -> mujoco.MjModel:
    """编译旧 RM65 模型（场地/球物理对照真值来源）。"""
    return load_mujoco_model(_RM65_PATH)


def _parameters() -> RobotRuntimeParameters:
    """V12 参数（可变访问入口，供 dataclasses.replace 派生用例）。"""
    return _v12_parameters()


def _with_scene(
    parameters: RobotRuntimeParameters, **overrides: Any
) -> RobotRuntimeParameters:
    """派生一份 scene 字段被覆盖的运行参数（不修改入参）。"""
    scene = dataclasses.replace(parameters.scene, **overrides)
    return dataclasses.replace(parameters, scene=scene)


def _bundle(racket: RacketProfile | None = None) -> Any:
    """用 V12 参数组装该侧裸模型（可选传入球拍档案）。"""
    return load_robot_model(
        _profile(), _parameters(), racket_profile=racket
    )


def _scene(racket: RacketProfile | None = None) -> TennisSceneBundle:
    """用 V12 参数组装带球执行场景（可选传入球拍档案）。"""
    return build_tennis_scene(
        _profile(), _parameters(), _task(), racket_profile=racket
    )


def _template_bytes() -> bytes:
    """读取版本控制中的 R1D 场景模板字节（用于"未改写模板"断言）。"""
    return _TEMPLATE_PATH.read_bytes()


# ---------------------------------------------------------------- MuJoCo helper
def _obj_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    """按名查 MuJoCo 对象 id，缺失即判定失败。"""
    obj_id = int(mujoco.mj_name2id(model, obj_type, name))
    assert obj_id >= 0, f"模型中缺少 {obj_type.name} '{name}'"
    return obj_id


def _body_id(model: mujoco.MjModel, name: str) -> int:
    """按名查 body id（缺失失败）。"""
    return _obj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    """按名查 geom id（缺失失败）。"""
    return _obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _site_id(model: mujoco.MjModel, name: str) -> int:
    """按名查 site id（缺失失败）。"""
    return _obj_id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def _quat_matrix(quat: Sequence[float]) -> np.ndarray:
    """wxyz 单位四元数 → 3×3 旋转矩阵（测试内独立实现）。"""
    w, x, y, z = (float(value) for value in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _body_xmat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """该 body 的世界旋转矩阵 (3,3)。"""
    return data.xmat[_body_id(model, name)].reshape(3, 3).copy()


def _forward(
    model: mujoco.MjModel,
    binding: Any,
    parameters: RobotRuntimeParameters,
) -> mujoco.MjData:
    """按绑定写入初始关节角并做一次 ``mj_forward``。"""
    data = mujoco.MjData(model)
    data.qpos[list(binding.qpos_indices)] = np.asarray(
        parameters.init_q, dtype=np.float64
    )
    mujoco.mj_forward(model, data)
    return data


def _racket_geom_ids(model: mujoco.MjModel) -> dict[str, int]:
    """球拍 body 子树下所有 geom 名 → id。"""
    racket_body = _body_id(model, "r1d_racket")
    result: dict[str, int] = {}
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != racket_body:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        assert name is not None
        result[str(name)] = geom_id
    return result


def _plane_geoms(model: mujoco.MjModel) -> list[int]:
    """所有 plane 类型 geom 的 id 列表。"""
    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    return [gid for gid in range(model.ngeom) if int(model.geom_type[gid]) == plane]


def _ground_geom_id(model: mujoco.MjModel) -> int:
    """按尺寸识别 ground（RM65 中 ground geom 无名；R1D 中名为 ground）。"""
    named = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground"))
    if named >= 0:
        return named
    for geom_id in _plane_geoms(model):
        if np.allclose(model.geom_size[geom_id], _COURT_GROUND_SIZE, atol=1e-12):
            return geom_id
    raise AssertionError("未找到 ground plane geom")


def _line_signatures(model: mujoco.MjModel) -> list[tuple[tuple[float, ...], ...]]:
    """白线 geom 的 (size, pos) 签名（排除 ground），按值排序。"""
    ground = _ground_geom_id(model)
    signatures = [
        (
            tuple(np.round(model.geom_size[geom_id], 10)),
            tuple(np.round(model.geom_pos[geom_id], 10)),
        )
        for geom_id in _plane_geoms(model)
        if geom_id != ground
    ]
    return sorted(signatures)


def _racket_expected_assembly(
    racket: RacketProfile,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """独立推导球拍装配期望（质量、COM、绕 COM 惯量、法兰挂载点）。

    公式：总质量 = 拍质量 + 连接件质量（如有）；COM 与绕 COM 惯量用
    平行轴定理逐项合并（球拍与连接件的 com/inertia 均相对各自质心）。

    Returns:
        (total_mass, com_local (3,), inertia_about_com (3,3), mount_position)
    """
    mass = float(racket.mass)
    com = np.asarray(racket.com, dtype=np.float64).copy()
    inertia = np.asarray(racket.inertia.as_matrix(), dtype=np.float64).copy()
    connector = racket.connector
    if connector is not None:
        connector_mass = float(connector.mass)
        connector_com = np.asarray(connector.com, dtype=np.float64)
        connector_inertia = np.asarray(
            connector.inertia.as_matrix(), dtype=np.float64
        )
        total = mass + connector_mass
        combined_com = (mass * com + connector_mass * connector_com) / total
        shift_racket = com - combined_com
        shift_connector = connector_com - combined_com
        inertia = (
            inertia
            + mass
            * (
                float(shift_racket @ shift_racket) * np.eye(3)
                - np.outer(shift_racket, shift_racket)
            )
            + connector_inertia
            + connector_mass
            * (
                float(shift_connector @ shift_connector) * np.eye(3)
                - np.outer(shift_connector, shift_connector)
            )
        )
        com = combined_com
        mass = total
    return (
        mass,
        com,
        inertia,
        np.asarray(racket.mount.position, dtype=np.float64).copy(),
    )


def _expected_racket_tcp(
    racket: RacketProfile,
    flange_pos: np.ndarray,
    flange_rot: np.ndarray,
) -> np.ndarray:
    """安装契约：``p_flange = R(wxyz) @ p_local + mount.position``。"""
    mount_rot = _quat_matrix(racket.mount.wxyz)
    local = np.asarray(racket.mount.position, dtype=np.float64) + (
        mount_rot @ np.asarray(racket.geometry.face.center, dtype=np.float64)
    )
    return flange_pos + flange_rot @ local


def _template_body_quat(name: str) -> np.ndarray:
    """从版本控制模板 XML 读取该 body 的 quat（缺省为单位四元数）。"""
    root = ET.fromstring(_template_bytes())
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            text = body.attrib.get("quat")
            if text is None:
                return np.array([1.0, 0.0, 0.0, 0.0])
            return np.array([float(token) for token in text.split()])
    raise AssertionError(f"模板中缺少 body '{name}'")


def _template_exclude_pairs() -> set[tuple[str, str]]:
    """从版本控制模板 XML 读取 contact exclude 的 body 名对（有序无关）。"""
    root = ET.fromstring(_template_bytes())
    contact = root.find("contact")
    assert contact is not None, "模板缺少 <contact> 节"
    pairs: set[tuple[str, str]] = set()
    for exclude in contact.findall("exclude"):
        body1 = exclude.attrib.get("body1")
        body2 = exclude.attrib.get("body2")
        assert body1 and body2
        pairs.add(tuple(sorted((body1, body2))))
    return pairs


# ================================================================ 完整外观
class TestCompleteAppearance:
    """A3：完整整机外观、7 主动 DOF、左臂冻结、落地与无穿透。"""

    def test_all_body_parts_present(self) -> None:
        """头/双臂/底座/轮/脚轮/升降/球拍/躯干全部存在。"""
        bundle = _bundle()
        model = bundle.model
        expected_bodies = (
            *(_ENV_CHAIN_BODIES),
            *_HEAD_BODIES,
            "r1d_mount",
            "r1d_base",
            "r1d_base_left_passive",
            *_ACTIVE_ARM_BODIES,
            *_PASSIVE_ARM_BODIES,
            "r1d_racket",
            "r1d_torso",
        )
        for name in expected_bodies:
            assert _body_id(model, name) > 0, f"缺少 body {name}"

        base_link = _body_id(model, "base_link")
        for name in (*_WHEEL_GEOMS, *_CASTER_GEOMS):
            geom_id = _geom_id(model, name)
            assert int(model.geom_bodyid[geom_id]) == base_link, (
                f"{name} 应挂在 base_link 下（落地视觉）"
            )

    def test_head_and_both_arms_have_visual_meshes(self) -> None:
        """头部 2 个 mesh 与双臂各 7 个视觉 mesh 编译进模型。"""
        model = _bundle().model
        assert model.nmesh == _EXPECTED_NMESH
        mesh_names = {
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, index))
            for index in range(model.nmesh)
        }
        assert set(_HEAD_BODIES) <= mesh_names
        for body_name in (*_ACTIVE_ARM_BODIES, *_PASSIVE_ARM_BODIES):
            body_id = _body_id(model, body_name)
            geom_ids = [
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) == body_id
            ]
            assert geom_ids, f"{body_name} 缺少视觉 geom"
        for body_name in _HEAD_BODIES:
            body_id = _body_id(model, body_name)
            geom_ids = [
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) == body_id
            ]
            assert geom_ids, f"{body_name} 缺少视觉 geom"

    def test_active_dofs_are_exactly_seven(self) -> None:
        """右臂恰 7 个主动 hinge；环境链/头/被动臂 0 关节。"""
        bundle = _bundle()
        model = bundle.model
        assert (model.nq, model.nv, model.nu, model.njnt) == (7, 7, 7, 7)
        joint_names = tuple(
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt))
            for jnt in range(model.njnt)
        )
        assert joint_names == tuple(_profile().active_joint_names)
        hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
        assert np.all(model.jnt_type == hinge)
        for body_name in (
            *_ENV_CHAIN_BODIES,
            *_HEAD_BODIES,
            "r1d_base_left_passive",
            *_PASSIVE_ARM_BODIES,
        ):
            body_id = _body_id(model, body_name)
            assert int(model.body_jntnum[body_id]) == 0, f"{body_name} 不应含关节"

    def test_passive_arm_does_not_drift_after_steps(self) -> None:
        """左臂为静态刚体：施加零力矩步进后位姿逐位不变。"""
        bundle = _bundle()
        model = bundle.model
        data = _forward(model, bundle.binding, bundle.parameters)
        watched = ("r1d_base_left_passive", *_PASSIVE_ARM_BODIES)
        before_pos = {name: data.xpos[_body_id(model, name)].copy() for name in watched}
        before_mat = {name: _body_xmat(model, data, name) for name in watched}
        data.ctrl[:] = 0.0
        for _ in range(300):
            mujoco.mj_step(model, data)
        for name in watched:
            assert np.array_equal(
                data.xpos[_body_id(model, name)], before_pos[name]
            ), f"{name} 位置漂移（左臂应固定）"
            assert np.array_equal(
                _body_xmat(model, data, name), before_mat[name]
            ), f"{name} 姿态漂移（左臂应固定）"

    def test_ground_landing_geometry(self) -> None:
        """底座落在 0.155 m：轮/脚轮最低点触地且无 geom 穿地。"""
        bundle = _bundle()
        data = _forward(bundle.model, bundle.binding, bundle.parameters)
        model = bundle.model
        base_z = data.xpos[_body_id(model, "base_link")][2]
        assert base_z == pytest.approx(_BASE_LANDING_HEIGHT, abs=1e-12)
        bottoms: list[float] = []
        for name in _WHEEL_GEOMS:
            geom_id = _geom_id(model, name)
            bottoms.append(float(data.geom_xpos[geom_id][2]) - _WHEEL_RADIUS)
        for name in _CASTER_GEOMS:
            geom_id = _geom_id(model, name)
            bottoms.append(float(data.geom_xpos[geom_id][2]) - _CASTER_RADIUS)
        assert min(bottoms) >= -1e-9, f"落地件穿地: {bottoms}"
        assert min(bottoms) == pytest.approx(0.0, abs=1e-4)

    def test_no_ground_support_column_or_pedestal(self) -> None:
        """R1D 不添加 RM65 落地支撑柱：worldbody 只有单个 mount body。"""
        model = _bundle().model
        for body_id in range(model.nbody):
            name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
            lowered = name.lower()
            assert "pedestal" not in lowered
            assert "support" not in lowered
            assert "column" not in lowered
        world_children = [
            body_id
            for body_id in range(model.nbody)
            if int(model.body_parentid[body_id]) == 0 and body_id != 0
        ]
        assert len(world_children) == 1

    def test_no_initial_penetration_in_tennis_scene(self) -> None:
        """带球执行场景 mj_forward 后无初始穿透（dist < 0）。"""
        scene = _scene()
        robot = scene.robot
        model = robot.model
        data = _forward(model, robot.binding, robot.parameters)
        penetrating: list[tuple[str, str, float]] = []
        for contact in data.contact[: data.ncon]:
            if contact.dist >= -1e-6:
                continue
            name1 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
            )
            name2 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
            )
            penetrating.append((str(name1), str(name2), float(contact.dist)))
        assert penetrating == [], f"初始穿透: {penetrating}"


# ================================================================ 升降重烘
class TestSceneLift:
    """A2：显式升降位移的内存重烘、隔离与校验。"""

    def test_default_lift_matches_template_bake(self) -> None:
        """默认 lift=0.4225 与 r1d_env.xml 的 bake 完全一致（未改写模板）。"""
        template_before = _template_bytes()
        bundle = _bundle()
        data = _forward(bundle.model, bundle.binding, bundle.parameters)
        waist_z = data.xpos[_body_id(bundle.model, "waist_yaw_link")][2]
        expected = (
            _BASE_LANDING_HEIGHT + 0.3288 + (0.0427 + _ENV_BAKED_LIFT)
            + (-0.2115 + _ENV_BAKED_LIFT)
        )
        assert waist_z == pytest.approx(expected, abs=1e-12)
        assert _template_bytes() == template_before

    def test_lift_rebake_matches_independent_fk(self) -> None:
        """改升降后法兰位姿与独立 URDF FK（同一 lift）一致。"""
        parameters = _with_scene(_parameters(), lift_displacement=0.2)
        bundle = load_robot_model(_profile(), parameters)
        data = _forward(bundle.model, bundle.binding, bundle.parameters)
        model = bundle.model
        flange_site = _site_id(model, "r1d_flange")
        waist_z = data.xpos[_body_id(model, "waist_yaw_link")][2]
        expected_waist = (
            _BASE_LANDING_HEIGHT + 0.3288 + (0.0427 + 0.2) + (-0.2115 + 0.2)
        )
        assert waist_z == pytest.approx(expected_waist, abs=1e-12)
        fk = r1d_reference.R1DUrdfFk(
            SIDE,
            base_position=parameters.scene.base_position,
            lift_ql=0.2,
        )
        expected_pos, expected_rot = fk.flange_pose(parameters.init_q)
        assert np.allclose(data.site_xpos[flange_site], expected_pos, atol=1e-9)
        assert np.allclose(
            data.site_xmat[flange_site].reshape(3, 3), expected_rot, atol=1e-9
        )

    def test_consecutive_loads_are_isolated(self) -> None:
        """默认 → lift=0.1 → 默认：首末完全一致，模板字节不被改写。"""
        template_before = _template_bytes()
        first = _bundle()
        first_data = _forward(first.model, first.binding, first.parameters)
        first_waist = first_data.xpos[_body_id(first.model, "waist_yaw_link")].copy()

        variant = load_robot_model(
            _profile(), _with_scene(_parameters(), lift_displacement=0.1)
        )
        variant_data = _forward(variant.model, variant.binding, variant.parameters)
        variant_waist = variant_data.xpos[
            _body_id(variant.model, "waist_yaw_link")
        ].copy()
        assert variant_waist[2] == pytest.approx(
            first_waist[2] - 2.0 * (_ENV_BAKED_LIFT - 0.1), abs=1e-12
        )

        again = _bundle()
        again_data = _forward(again.model, again.binding, again.parameters)
        assert np.array_equal(
            again_data.xpos[_body_id(again.model, "waist_yaw_link")], first_waist
        )
        assert _template_bytes() == template_before

    @pytest.mark.parametrize("bad_value", [-0.01, _LIFT_STAGE_RANGE + 0.001, "abc"])
    def test_invalid_lift_rejected_from_yaml(
        self, tmp_path: Path, bad_value: object
    ) -> None:
        """YAML 中的非法升降位移被拒绝（范围/类型校验）。"""
        text = _V12_RUNTIME_YAML.read_text(encoding="utf-8")
        assert "lift_displacement: 0.4225" in text
        patched = tmp_path / "runtime.yaml"
        patched.write_text(
            text.replace(
                "lift_displacement: 0.4225", f"lift_displacement: {bad_value}"
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="lift_displacement"):
            load_runtime_parameters(patched, _profile())

    def test_out_of_range_lift_rejected_at_model_load(self) -> None:
        """手工构造的越界 lift 在模型组装时被拒绝（防悬浮/穿地）。"""
        parameters = _with_scene(_parameters(), lift_displacement=0.5)
        with pytest.raises(ValueError, match="lift_displacement"):
            load_robot_model(_profile(), parameters)


# ================================================================ 固定关节姿态
class TestFixedJointPoses:
    """A2：固定关节姿态的内存重烘、校验与隔离。"""

    def test_default_pose_matches_template(self) -> None:
        """未配置固定姿态时沿用模板烘焙的默认被动臂姿态。"""
        bundle = _bundle()
        for name in _PASSIVE_ARM_BODIES:
            body_id = _body_id(bundle.model, name)
            assert np.allclose(
                _quat_matrix(bundle.model.body_quat[body_id]),
                _quat_matrix(_template_body_quat(name)),
                atol=1e-12,
            )

    def test_fixed_pose_overrides_passive_arm_only(self) -> None:
        """配置左肘 1.5 rad：左肘及其子链改变，左肩链/右臂/机身不变。"""
        default = _bundle()
        default_data = _forward(default.model, default.binding, default.parameters)
        parameters = _with_scene(
            _parameters(), fixed_joint_poses=(("left_elbow_joint", 1.5),)
        )
        variant = load_robot_model(_profile(), parameters)
        variant_data = _forward(variant.model, variant.binding, variant.parameters)

        default_model, variant_model = default.model, variant.model
        # 左肩三节（肘的上游）不变
        for name in _PASSIVE_ARM_BODIES[:3]:
            assert np.allclose(
                _body_xmat(default_model, default_data, name),
                _body_xmat(variant_model, variant_data, name),
                atol=1e-12,
            ), f"{name} 不应受固定姿态影响"
        # 左肘与下游改变
        for name in _PASSIVE_ARM_BODIES[3:]:
            assert not np.allclose(
                _body_xmat(default_model, default_data, name),
                _body_xmat(variant_model, variant_data, name),
                atol=1e-9,
            ), f"{name} 应随固定姿态改变"
        # 右臂/机身不变
        for name in (*_ACTIVE_ARM_BODIES, *_ENV_CHAIN_BODIES, *_HEAD_BODIES):
            assert np.array_equal(
                default_data.xpos[_body_id(default_model, name)],
                variant_data.xpos[_body_id(variant_model, name)],
            ), f"{name} 不应受被动臂固定姿态影响"

    def test_fixed_pose_consecutive_isolation(self) -> None:
        """连续加载不同固定姿态：默认姿态可完全复原。"""
        first = _bundle()
        first_data = _forward(first.model, first.binding, first.parameters)
        elbow_first = first_data.xmat[_body_id(first.model, "left_elbow_link")].copy()

        load_robot_model(
            _profile(),
            _with_scene(_parameters(), fixed_joint_poses=(("left_elbow_joint", 1.2),)),
        )
        again = _bundle()
        again_data = _forward(again.model, again.binding, again.parameters)
        assert np.array_equal(
            again_data.xmat[_body_id(again.model, "left_elbow_link")], elbow_first
        )

    @pytest.mark.parametrize(
        ("bad_poses", "match"),
        [
            ((("right_elbow_joint", 0.3),), "主动关节"),
            ((("left_bogus_joint", 0.3),), "left_bogus_joint"),
            ((("left_elbow_joint", 9.0),), "left_elbow_joint"),
        ],
    )
    def test_invalid_fixed_pose_rejected(
        self, bad_poses: tuple[tuple[str, float], ...], match: str
    ) -> None:
        """非法关节名/主动关节/超限角度被拒绝。"""
        parameters = _with_scene(_parameters(), fixed_joint_poses=bad_poses)
        with pytest.raises(ValueError, match=match):
            load_robot_model(_profile(), parameters)


# ================================================================ 旧场地恢复
class TestRestoredOldScene:
    """A4/R6：恢复旧 RM65 场地元素与球物理，删除 ITF 球场/球网/四灯。"""

    def test_option_and_ground_match_rm65(self) -> None:
        """dt/重力/积分器/地面尺寸与材质与 rm65_model.xml 一致。"""
        rm65 = _rm65_model()
        model = _scene().robot.model
        assert model.opt.timestep == pytest.approx(_OPTION_TIMESTEP, abs=1e-12)
        assert rm65.opt.timestep == pytest.approx(_OPTION_TIMESTEP, abs=1e-12)
        assert np.allclose(model.opt.gravity, _OPTION_GRAVITY, atol=1e-12)
        assert np.allclose(rm65.opt.gravity, _OPTION_GRAVITY, atol=1e-12)
        assert int(model.opt.integrator) == int(rm65.opt.integrator)

        ground = _ground_geom_id(model)
        rm65_ground = _ground_geom_id(rm65)
        assert np.allclose(
            model.geom_size[ground], _COURT_GROUND_SIZE, atol=1e-12
        )
        assert np.allclose(
            model.geom_size[rm65_ground], _COURT_GROUND_SIZE, atol=1e-12
        )
        assert np.allclose(model.geom_rgba[ground], _COURT_GROUND_RGBA, atol=1e-12)
        assert np.allclose(
            rm65.geom_rgba[rm65_ground], _COURT_GROUND_RGBA, atol=1e-12
        )

        material = _obj_id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "court_floor")
        rm65_material = _obj_id(
            rm65, mujoco.mjtObj.mjOBJ_MATERIAL, "court_floor"
        )
        assert np.allclose(
            model.mat_rgba[material], _COURT_MATERIAL_RGBA, atol=1e-12
        )
        assert np.allclose(
            rm65.mat_rgba[rm65_material], _COURT_MATERIAL_RGBA, atol=1e-12
        )
        assert np.allclose(
            model.mat_texrepeat[material], _COURT_TEXREPEAT, atol=1e-12
        )
        assert np.allclose(
            model.mat_reflectance[material], _COURT_REFLECTANCE, atol=1e-12
        )

        texture = _obj_id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "court_grid")
        rm65_texture = _obj_id(
            rm65, mujoco.mjtObj.mjOBJ_TEXTURE, "court_grid"
        )
        # MuJoCo 3.9 不再暴露 tex_rgb1/tex_rgb2：内置 checker 纹理在编译期
        # 烘焙进 tex_data。逐字节比较编译结果 = 纹理属性（rgb1/rgb2/
        # mark/markrgb/尺寸）与 RM65 完全一致的最强断言。
        assert model.ntex == rm65.ntex
        assert model.tex_width[texture] == rm65.tex_width[rm65_texture]
        assert model.tex_height[texture] == rm65.tex_height[rm65_texture]
        assert model.tex_nchannel[texture] == rm65.tex_nchannel[rm65_texture]
        assert int(model.tex_type[texture]) == int(rm65.tex_type[rm65_texture])
        assert np.array_equal(
            model.tex_data.reshape(-1)[
                model.tex_adr[texture] : model.tex_adr[texture]
                + model.tex_width[texture] * model.tex_height[texture]
                * model.tex_nchannel[texture]
            ],
            rm65.tex_data.reshape(-1)[
                rm65.tex_adr[rm65_texture] : rm65.tex_adr[rm65_texture]
                + rm65.tex_width[rm65_texture]
                * rm65.tex_height[rm65_texture]
                * rm65.tex_nchannel[rm65_texture]
            ],
        )

    def test_court_lines_match_rm65(self) -> None:
        """10 条旧白线的尺寸/位置与 rm65_model.xml 逐条一致。"""
        model = _scene().robot.model
        rm65 = _rm65_model()
        expected = sorted(
            (tuple(float(v) for v in size), tuple(float(v) for v in pos))
            for size, pos in _COURT_LINES
        )
        assert _line_signatures(model) == expected
        assert _line_signatures(rm65) == expected

    def test_no_itf_scene_remnants(self) -> None:
        """用户尝试中的 ITF 标准球场/球网痕迹（net_*/蓝色硬地/11 线）不存在。"""
        model = _scene().robot.model
        names: list[str] = []
        for obj_type in (
            mujoco.mjtObj.mjOBJ_BODY,
            mujoco.mjtObj.mjOBJ_GEOM,
            mujoco.mjtObj.mjOBJ_MATERIAL,
            mujoco.mjtObj.mjOBJ_TEXTURE,
        ):
            for index in range(
                {
                    mujoco.mjtObj.mjOBJ_BODY: model.nbody,
                    mujoco.mjtObj.mjOBJ_GEOM: model.ngeom,
                    mujoco.mjtObj.mjOBJ_MATERIAL: model.nmat,
                    mujoco.mjtObj.mjOBJ_TEXTURE: model.ntex,
                }[obj_type]
            ):
                name = mujoco.mj_id2name(model, obj_type, index)
                if name is not None:
                    names.append(str(name))
        assert not [name for name in names if "net" in name.lower()]
        # 地面 plane + 10 条白线 = 11 个 plane geom（ITF 版为地面 + 11 线）
        assert len(_plane_geoms(model)) == 11
        for material in range(model.nmat):
            assert not np.allclose(
                model.mat_rgba[material],
                np.array([0.10, 0.32, 0.52, 1.0]),
                atol=1e-12,
            ), "不应保留用户尝试的蓝色硬地材质"

    def test_single_restored_light(self) -> None:
        """恢复 pre-attempt 单灯（R6 要求移除用户尝试新增的四灯）。"""
        model = _scene().robot.model
        assert model.nlight == 1
        assert np.allclose(model.light_pos[0], _LIGHT_POS, atol=1e-12)
        assert np.allclose(model.light_dir[0], _LIGHT_DIR, atol=1e-12)
        assert np.allclose(model.light_diffuse[0], _LIGHT_DIFFUSE, atol=1e-12)

    def test_ball_physics_matches_rm65(self) -> None:
        """球半径/质量/颜色/掩码/初态与 rm65_model.xml 一致。"""
        scene = _scene()
        model = scene.robot.model
        rm65 = _rm65_model()
        ball = _geom_id(model, "ball")
        rm65_ball = _geom_id(rm65, "ball")
        assert np.allclose(model.geom_size[ball], [ _BALL_RADIUS, 0, 0], atol=1e-12)
        assert np.allclose(
            rm65.geom_size[rm65_ball], [_BALL_RADIUS, 0, 0], atol=1e-12
        )
        assert model.body_mass[_body_id(model, "ball")] == pytest.approx(
            _BALL_MASS, abs=1e-12
        )
        assert rm65.body_mass[_body_id(rm65, "ball")] == pytest.approx(
            _BALL_MASS, abs=1e-12
        )
        assert np.allclose(model.geom_rgba[ball], _BALL_RGBA, atol=1e-12)
        assert int(model.geom_contype[ball]) == _BALL_CONTYPE
        assert int(model.geom_conaffinity[ball]) == _BALL_CONAFFINITY
        assert int(rm65.geom_contype[rm65_ball]) == _BALL_CONTYPE
        assert int(rm65.geom_conaffinity[rm65_ball]) == _BALL_CONAFFINITY
        free = int(mujoco.mjtJoint.mjJNT_FREE)
        assert int(model.jnt_type[scene.ball_joint_id]) == free

    def test_ball_racket_contact_pair_matches_rm65(self) -> None:
        """球—拍接触 pair 参数（condim/solref/solimp）与旧模型一致。"""
        scene = _scene()
        model = scene.robot.model
        rm65 = _rm65_model()
        ball = _geom_id(model, "ball")
        rm65_ball = _geom_id(rm65, "ball")
        assert np.allclose(model.geom_size[scene.ball_geom_id], [ _BALL_RADIUS, 0, 0], atol=1e-12)
        assert np.allclose(
            model.geom_rgba[scene.ball_geom_id], _BALL_RGBA, atol=1e-12
        )
        pair_index = next(
            index
            for index in range(model.npair)
            if ball in (int(model.pair_geom1[index]), int(model.pair_geom2[index]))
        )
        rm65_pair = next(
            index
            for index in range(rm65.npair)
            if rm65_ball
            in (int(rm65.pair_geom1[index]), int(rm65.pair_geom2[index]))
        )
        assert int(model.pair_dim[pair_index]) == _PAIR_CONDIM
        assert int(rm65.pair_dim[rm65_pair]) == _PAIR_CONDIM
        assert np.allclose(model.pair_solref[pair_index], _PAIR_SOLREF, atol=1e-12)
        assert np.allclose(rm65.pair_solref[rm65_pair], _PAIR_SOLREF, atol=1e-12)
        assert np.allclose(
            model.pair_solimp[pair_index], _PAIR_SOLIMP, atol=1e-12
        )
        assert np.allclose(
            rm65.pair_solimp[rm65_pair], _PAIR_SOLIMP, atol=1e-12
        )


# ================================================================ 球拍档案装配
class TestRacketProfileAssembly:
    """B1/B2：两套球拍档案的独立核对（不复用被测生成函数）。"""

    @pytest.mark.parametrize(
        "racket_factory",
        [_legacy_racket_profile, _blade_profile],
        ids=["rm65_legacy", "blade100_v9"],
    )
    def test_tcp_matches_profile_contract(
        self, racket_factory: Any
    ) -> None:
        """TCP site = 法兰位姿 ∘ (mount.position + R(wxyz)·face.center)。"""
        racket = racket_factory()
        bundle = _bundle(racket)
        data = _forward(bundle.model, bundle.binding, bundle.parameters)
        model = bundle.model
        fk = r1d_reference.R1DUrdfFk(
            SIDE,
            base_position=bundle.parameters.scene.base_position,
            lift_ql=bundle.parameters.scene.lift_displacement,
        )
        flange_pos, flange_rot = fk.flange_pose(bundle.parameters.init_q)
        expected = _expected_racket_tcp(racket, flange_pos, flange_rot)
        tcp_site = _site_id(model, "racket_center")
        assert np.allclose(data.site_xpos[tcp_site], expected, atol=1e-9)
        # TCP 不得由 scenario runtime 重复给出
        assert bundle.parameters.scene.racket_position is None

    @pytest.mark.parametrize(
        "racket_factory",
        [_legacy_racket_profile, _blade_profile],
        ids=["rm65_legacy", "blade100_v9"],
    )
    def test_face_collision_center_and_normal(self, racket_factory: Any) -> None:
        """碰撞面中心=face.center、半轴=semi_axes、法向=桨体局部 +X。"""
        racket = racket_factory()
        bundle = _bundle(racket)
        data = _forward(bundle.model, bundle.binding, bundle.parameters)
        model = bundle.model
        face = _geom_id(model, "racket_face")
        assert int(model.geom_type[face]) == int(
            mujoco.mjtGeom.mjGEOM_ELLIPSOID
        )
        assert np.allclose(
            model.geom_pos[face], racket.geometry.face.center, atol=1e-12
        )
        assert np.allclose(
            model.geom_size[face], racket.geometry.face.semi_axes, atol=1e-12
        )
        body = _body_id(model, "r1d_racket")
        body_xmat = data.xmat[body].reshape(3, 3)
        expected_normal = body_xmat @ np.array([1.0, 0.0, 0.0])
        actual_normal = data.geom_xmat[face].reshape(3, 3)[:, 0]
        assert np.allclose(actual_normal, expected_normal, atol=1e-12)
        # T4 契约：get_ee_normal = TCP site 局部 +X = 拍面法向
        tcp_site = _site_id(model, "racket_center")
        site_x = data.site_xmat[tcp_site].reshape(3, 3)[:, 0]
        assert np.allclose(site_x, expected_normal, atol=1e-12)
        # 半轴第一项为最薄轴（沿局部 +X）
        assert racket.geometry.face.semi_axes[0] == min(
            racket.geometry.face.semi_axes
        )

    @pytest.mark.parametrize(
        "racket_factory",
        [_legacy_racket_profile, _blade_profile],
        ids=["rm65_legacy", "blade100_v9"],
    )
    def test_flange_mount_rigid_transform(self, racket_factory: Any) -> None:
        """拍体原点=法兰挂接点+mount.position、朝向=R(mount.wxyz)（刚体安装变换）。

        契约（T1）：``p_flange = R(mount.wxyz) @ p_local + mount.position``，
        即球拍局部系原点（柄底）在法兰系中的位置为 mount.position。
        ``r1d_racket`` 的父 body 是 ``right_wrist_yaw_link``，法兰 site
        ``r1d_flange`` 与父 body 同向（site 无 quat），因此：
        body_pos = site_pos[flange] + mount.position；body_quat = mount.wxyz。
        """
        racket = racket_factory()
        model = _bundle(racket).model
        body = _body_id(model, "r1d_racket")
        flange_site = _obj_id(
            model, mujoco.mjtObj.mjOBJ_SITE, "r1d_flange"
        )
        assert np.allclose(
            model.body_pos[body],
            model.site_pos[flange_site] + np.asarray(racket.mount.position),
            atol=1e-12,
        )
        assert np.allclose(
            _quat_matrix(model.body_quat[body]),
            _quat_matrix(racket.mount.wxyz),
            atol=1e-12,
        )
        # 世界系独立复核：拍体原点 = 法兰世界位姿 ∘ mount 变换
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        flange_pos = data.site_xpos[flange_site]
        flange_rot = data.site_xmat[flange_site].reshape(3, 3)
        expected_world = (
            flange_pos + flange_rot @ np.asarray(racket.mount.position)
        )
        assert np.allclose(data.xpos[body], expected_world, atol=1e-9)
        assert np.allclose(
            data.xmat[body].reshape(3, 3),
            flange_rot @ _quat_matrix(racket.mount.wxyz),
            atol=1e-9,
        )

    @pytest.mark.parametrize(
        "racket_factory",
        [_legacy_racket_profile, _blade_profile],
        ids=["rm65_legacy", "blade100_v9"],
    )
    def test_total_mass_com_and_inertia(self, racket_factory: Any) -> None:
        """拍体总质量/COM/绕 COM 惯量 = 拍 + 连接件（平行轴定理）。"""
        racket = racket_factory()
        model = _bundle(racket).model
        body = _body_id(model, "r1d_racket")
        expected_mass, expected_com, expected_inertia, _ = (
            _racket_expected_assembly(racket)
        )
        assert model.body_mass[body] == pytest.approx(expected_mass, abs=1e-12)
        assert np.allclose(model.body_ipos[body], expected_com, atol=1e-12)
        expected_moments = np.sort(np.diag(expected_inertia))
        actual_moments = np.sort(model.body_inertia[body])
        assert np.allclose(actual_moments, expected_moments, atol=1e-9)

    @pytest.mark.parametrize(
        "racket_factory",
        [_legacy_racket_profile, _blade_profile],
        ids=["rm65_legacy", "blade100_v9"],
    )
    def test_connector_mass_counted_once(self, racket_factory: Any) -> None:
        """连接件质量只进入拍体一次：其余 body 质量与 legacy 模型逐位相同。"""
        racket = racket_factory()
        profile_mode = _bundle(racket).model
        legacy = _bundle().model
        racket_body = _body_id(profile_mode, "r1d_racket")
        assert profile_mode.nbody == legacy.nbody
        for body in range(profile_mode.nbody):
            if body == racket_body:
                continue
            assert profile_mode.body_mass[body] == pytest.approx(
                legacy.body_mass[body], abs=1e-12
            )
        expected_mass, _, _, _ = _racket_expected_assembly(racket)
        assert profile_mode.body_mass[racket_body] == pytest.approx(
            expected_mass, abs=1e-12
        )

    @pytest.mark.parametrize(
        "racket_factory",
        [_legacy_racket_profile, _blade_profile],
        ids=["rm65_legacy", "blade100_v9"],
    )
    def test_geom_labels_distinct_and_counts(self, racket_factory: Any) -> None:
        """面/柄/框/喉/连接件 geom 标签互不相同且数量与档案一致。"""
        racket = racket_factory()
        model = _bundle(racket).model
        geoms = _racket_geom_ids(model)
        labels = set(geoms)
        assert "racket_face" in labels
        assert "racket_face_visual" in labels
        assert "racket_handle" in labels
        circles = {
            name
            for name in labels
            if name.startswith("racket_throat_") or name.startswith("racket_frame_")
        }
        assert len(circles) == len(racket.geometry.throat) + len(
            racket.geometry.frame
        )
        assert not (
            {"racket_face", "racket_face_visual", "racket_handle"} & circles
        ), "面/柄标签不得与框/喉标签冲突"
        for index in range(len(racket.geometry.throat)):
            assert f"racket_throat_{index}" in labels
        for index in range(len(racket.geometry.frame)):
            assert f"racket_frame_{index}" in labels
        if racket.connector is None:
            assert "racket_connector" not in labels
        else:
            assert "racket_connector" in labels
            connector_label = _geom_id(model, "racket_connector")
            assert np.allclose(
                model.geom_pos[connector_label],
                racket.connector.center,
                atol=1e-12,
            )
        # 渲染几何全部为视觉（碰撞槽在带球场景中按名开启）
        for geom_id in geoms.values():
            assert int(model.geom_contype[geom_id]) == 0
            assert int(model.geom_conaffinity[geom_id]) == 0

    @pytest.mark.parametrize(
        "racket_factory",
        [_legacy_racket_profile, _blade_profile],
        ids=["rm65_legacy", "blade100_v9"],
    )
    def test_scene_enables_face_collision_only(self, racket_factory: Any) -> None:
        """带球场景只把 racket_face 打开为球拍碰撞体（含球 pair）。"""
        scene = _scene(racket_factory())
        model = scene.robot.model
        face = _geom_id(model, "racket_face")
        for geom_id in scene.racket_geom_ids:
            assert geom_id == face
        assert int(model.geom_contype[face]) == 8
        assert int(model.geom_conaffinity[face]) == 16
        for name, geom_id in _racket_geom_ids(model).items():
            if name == "racket_face":
                continue
            assert int(model.geom_contype[geom_id]) == 0, name
        assert scene.racket_geom_ids == (face,)

    def test_forbidden_contact_pairs_documented(self) -> None:
        """完整机身禁止接触对是显式设计集合（19 条，含腕↔拍、mount↔躯干）。"""
        model = _bundle().model
        pairs = _template_exclude_pairs()
        expected = {
            tuple(sorted(("r1d_mount", "base_link"))),
            tuple(sorted(("base_link", "lift_mount_link"))),
            tuple(sorted(("lift_mount_link", "lift_stage1_link"))),
            tuple(sorted(("lift_stage1_link", "lift_stage2_link"))),
            tuple(sorted(("lift_stage2_link", "pelvis_link"))),
            tuple(sorted(("pelvis_link", "waist_yaw_link"))),
            tuple(sorted(("waist_yaw_link", "r1d_base"))),
            tuple(sorted(("r1d_base", f"{SIDE}_shoulder_pitch_Link"))),
            tuple(sorted(("waist_yaw_link", f"{SIDE}_shoulder_pitch_Link"))),
            tuple(sorted(("waist_yaw_link", f"{SIDE}_shoulder_roll_link"))),
            tuple(sorted(("r1d_mount", "r1d_torso"))),
            tuple(
                sorted(
                    (f"{SIDE}_wrist_yaw_link", "r1d_racket")
                )
            ),
        }
        for upper, lower in zip(
            _ACTIVE_ARM_BODIES,
            _ACTIVE_ARM_BODIES[1:],
        ):
            expected.add(tuple(sorted((upper, lower))))
        expected.add(
            tuple(sorted((f"{SIDE}_wrist_roll_link", f"{SIDE}_wrist_yaw_link")))
        )
        assert pairs == expected, f"禁止接触对漂移: {pairs ^ expected}"
        assert model.nexclude == len(expected)


# ================================================================ 几何同步
def _mount_variant(racket: RacketProfile) -> RacketProfile:
    """安装姿态变体：法兰挂载点平移 (+0.02, 0, +0.05)。"""
    mount = dataclasses.replace(
        racket.mount,
        position=np.asarray(racket.mount.position) + np.array([0.02, 0.0, 0.05]),
    )
    return dataclasses.replace(racket, mount=mount)


def _length_variant(racket: RacketProfile) -> RacketProfile:
    """球拍长度变体：拍面中心与 total length 各延长 0.05 m。"""
    geometry = racket.geometry
    face = dataclasses.replace(
        geometry.face,
        center=np.asarray(geometry.face.center) + np.array([0.0, 0.0, 0.05]),
    )
    extended = dataclasses.replace(
        geometry,
        face=face,
        tcp=face.center,
        length=geometry.length + 0.05,
    )
    return dataclasses.replace(racket, geometry=extended)


_VARIANTS = [
    ("mount_shift", _mount_variant),
    ("length_extended", _length_variant),
]


class TestGeometrySync:
    """A5：改变球拍长度/安装后，裸模型/执行场景/执行与规划环境同步。"""

    @pytest.mark.parametrize(("variant_name", "variant_factory"), _VARIANTS)
    def test_bare_and_scene_track_profile(
        self, variant_name: str, variant_factory: Any
    ) -> None:
        """裸模型与带球场景的 TCP/碰撞中心/法向一致，且随档案变化。"""
        base_racket = _blade_profile()
        base_bundle = _bundle(base_racket)
        base_data = _forward(
            base_bundle.model, base_bundle.binding, base_bundle.parameters
        )
        base_tcp = base_data.site_xpos[_site_id(base_bundle.model, "racket_center")]

        racket = variant_factory(base_racket)
        bundle = _bundle(racket)
        scene = _scene(racket)
        bare_data = _forward(bundle.model, bundle.binding, bundle.parameters)
        scene_data = _forward(
            scene.robot.model, scene.robot.binding, scene.robot.parameters
        )
        model = bundle.model
        scene_model = scene.robot.model
        fk = r1d_reference.R1DUrdfFk(
            SIDE,
            base_position=bundle.parameters.scene.base_position,
            lift_ql=bundle.parameters.scene.lift_displacement,
        )
        flange_pos, flange_rot = fk.flange_pose(bundle.parameters.init_q)
        expected_tcp = _expected_racket_tcp(racket, flange_pos, flange_rot)
        bare_tcp = bare_data.site_xpos[_site_id(model, "racket_center")]
        scene_tcp = scene_data.site_xpos[_site_id(scene_model, "racket_center")]
        assert np.allclose(bare_tcp, expected_tcp, atol=1e-9)
        assert np.allclose(scene_tcp, expected_tcp, atol=1e-9)
        if variant_name == "mount_shift":
            assert not np.allclose(bare_tcp, base_tcp, atol=1e-6)
        else:
            assert not np.allclose(bare_tcp, base_tcp, atol=1e-6)

        face = _geom_id(model, "racket_face")
        scene_face = _geom_id(scene_model, "racket_face")
        assert np.allclose(
            model.geom_pos[face], racket.geometry.face.center, atol=1e-12
        )
        assert np.allclose(
            scene_model.geom_pos[scene_face], racket.geometry.face.center, atol=1e-12
        )
        assert np.allclose(
            model.geom_size[face], racket.geometry.face.semi_axes, atol=1e-12
        )

    @pytest.mark.parametrize(("variant_name", "variant_factory"), _VARIANTS)
    def test_execution_and_planning_envs_track_profile(
        self, variant_name: str, variant_factory: Any
    ) -> None:
        """执行环境（ProfileSimEnv）与规划环境（ProfilePlanningEnv）TCP/法向同步。"""
        base_racket = _blade_profile()
        base_bundle = _bundle(base_racket)
        base_env = BoundRobotEnv(base_bundle)
        base_pos = base_env.get_ee_pos()

        racket = variant_factory(base_racket)
        bundle = _bundle(racket)
        scene = _scene(racket)
        execution = ProfileSimEnv(scene)
        planning = ProfilePlanningEnv(bundle)

        fk = r1d_reference.R1DUrdfFk(
            SIDE,
            base_position=bundle.parameters.scene.base_position,
            lift_ql=bundle.parameters.scene.lift_displacement,
        )
        flange_pos, flange_rot = fk.flange_pose(bundle.parameters.init_q)
        expected_tcp = _expected_racket_tcp(racket, flange_pos, flange_rot)
        assert np.allclose(execution.get_ee_pos(), expected_tcp, atol=1e-9)
        assert np.allclose(planning.get_ee_pos(), expected_tcp, atol=1e-9)
        assert not np.allclose(execution.get_ee_pos(), base_pos, atol=1e-6)

        expected_normal = (
            flange_rot
            @ _quat_matrix(racket.mount.wxyz)
            @ np.array([1.0, 0.0, 0.0])
        )
        assert np.allclose(execution.get_ee_normal(), expected_normal, atol=1e-9)
        assert np.allclose(planning.get_ee_normal(), expected_normal, atol=1e-9)


# ================================================================ legacy 模式
class TestLegacyRacketMode:
    """A1：省略 racket_profile（None）保持旧 API 语义。"""

    def test_none_keeps_legacy_disc_racket(self) -> None:
        """legacy 场景参数仍装配旧圆盘拍（mass/diaginertia/geom 数/位置）。"""
        bundle = load_robot_model(_profile(), _legacy_parameters())
        assert bundle.racket is None
        model = bundle.model
        body = _body_id(model, "r1d_racket")
        scene = bundle.parameters.scene
        assert model.body_mass[body] == pytest.approx(0.30, abs=1e-12)
        assert np.allclose(
            model.body_inertia[body], [0.0018, 0.0040, 0.0040], atol=1e-12
        )
        assert np.allclose(
            np.sort(model.body_inertia[body]),
            np.sort([0.0018, 0.0040, 0.0040]),
            atol=1e-12,
        )
        flange_site = _site_id(model, "r1d_flange")
        assert np.allclose(
            model.body_pos[body],
            model.site_pos[flange_site] + scene.racket_position,
            atol=1e-12,
        )
        geoms = _racket_geom_ids(model)
        assert len(geoms) == 9
        face = _geom_id(model, "racket_face")
        assert int(model.geom_type[face]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        assert np.allclose(
            model.geom_size[face][:2],
            [scene.racket_radius, scene.racket_half_thickness],
            atol=1e-12,
        )
        # 旧路径同样满足 +X 法向契约（碰撞面细轴 = 拍体局部 +X）
        data = _forward(model, bundle.binding, bundle.parameters)
        body_xmat = data.xmat[body].reshape(3, 3)
        face_xmat = data.geom_xmat[face].reshape(3, 3)
        assert np.allclose(
            face_xmat[:, 2], body_xmat[:, 0], atol=1e-12
        )

    def test_omitting_kwarg_equals_none(self) -> None:
        """省略 racket_profile 与显式传 None 完全等价。"""
        implicit = load_robot_model(_profile(), _parameters())
        explicit = load_robot_model(
            _profile(), _parameters(), racket_profile=None
        )
        assert implicit.model.nq == explicit.model.nq
        assert np.array_equal(implicit.model.body_pos, explicit.model.body_pos)
        assert np.array_equal(implicit.model.body_mass, explicit.model.body_mass)
        assert implicit.racket is None and explicit.racket is None

    def test_profile_mode_rejects_duplicated_racket_scene_fields(self) -> None:
        """提供档案时 scene.racket_* 必须为空（球拍物理唯一来源是档案）。"""
        with pytest.raises(ValueError, match="scene.racket_"):
            load_robot_model(
                _profile(), _legacy_parameters(), racket_profile=_blade_profile()
            )


# ================================================================ 新运行配置
class TestV12RuntimeConfig:
    """新增 V12 仿真运行配置：显式升降/固定姿态、无 legacy racket 字段。"""

    def test_v12_runtime_config_contract(self) -> None:
        """配置字段与加载契约：无 racket_*、显式 lift、固定姿态为空。"""
        parameters = _v12_parameters()
        scene = parameters.scene
        assert parameters.robot_id == "unitree_r1d"
        assert parameters.joint_names == tuple(_profile().active_joint_names)
        assert scene.base_position.tolist() == [-12.5, -0.8, _BASE_LANDING_HEIGHT]
        assert float(scene.lift_displacement) == pytest.approx(
            _ENV_BAKED_LIFT, abs=1e-12
        )
        assert dict(scene.fixed_joint_poses) == {}
        for field in (
            "racket_position",
            "racket_quaternion",
            "racket_radius",
            "racket_half_thickness",
            "racket_mass",
        ):
            assert getattr(scene, field) is None, f"{field} 不应在 V12 配置中重复给出"

    def test_v12_runtime_uses_profile_racket(self) -> None:
        """V12 参数 + 成人拍档案：7 主动 DOF、TCP 与档案契约一致。"""
        racket = _blade_profile()
        bundle = _bundle(racket)
        assert bundle.racket is not None
        assert bundle.racket.racket_id == racket.racket_id
        assert bundle.model.nu == 7
        assert bundle.racket.total_mass == pytest.approx(
            racket.mass + (racket.connector.mass if racket.connector else 0.0),
            abs=1e-12,
        )
