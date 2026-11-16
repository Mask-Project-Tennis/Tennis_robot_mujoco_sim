"""R1D 带球场景与碰撞集合测试（阶段 3 · Task 3.2）。

核心断言组（brief + 施工计划 Task 3.2）：
1. ``test_scene_shape_and_named_addresses``：双侧带球模型主动关节仍 7、
   受控状态仍 14；典型 nq=14/nv=13/nu=7/njnt=8，但全部地址按名称解析
   （关节/执行器/球地址逐名对照 jnt_qposadr/jnt_dofadr）；裸模型无球，
   场景 id 不复用另一模型（按名回查一致）。
2. ``test_ball_free_joint_precedes_active_joints``：球 free joint 位于
   主动关节之前（qpos[0:7]=球，臂在 7..13）；主动绑定不包含球
   quaternion（qpos[3:7] 与主动地址不相交）。
3. ``test_ball_physics_match_rm65_source``：球尺寸/质量/掩码/初位姿与
   rm65_model.xml 逐字来源一致，且球初态写入 qpos0。
4. ``test_collision_mask_semantics``：碰撞掩码表逐 geom 断言 + 派生
   允许/禁止对表（球—拍、臂/拍—机身、球—地面允许；臂—臂、臂—球、
   臂—拍、环境链—任意禁止）。
5. ``test_racket_ball_pair_params_and_toggle_channel``：球—拍显式
   <pair> 弹性参数（condim=1/solref=-5000 0/solimp，rm65 逐字）实际
   生效（接触 dim==1），pair_margin 为单独控制通道（关→无接触，开→恢复）。
6. ``test_initial_state_has_no_forbidden_contact``：init_q 下全部保护
   对无接触（初态无禁止接触）。
7. ``test_real_penetration_negative_observable``：真实穿透负例可被观测
   ——球塞入机身、臂/拍扫掠出与机身 torso_box 的保护对接触。
8. ``test_geometry_and_inertia_match_stage2_model``：带球模型与同参数
   裸模型（阶段 2）逐名比较 body 质量/惯量/位姿、geom 类型/尺寸/位姿、
   site 位置、jnt_range 完全一致。
9. ``test_template_and_asset_bytes_unchanged``：双侧模板 + 四份片段
   字节与 git HEAD 一致（组装只动内存副本）。
10. 负例：球 free joint 名称与既有 hinge 冲突（类型错误）、racket/
    保护对 geom 名称缺失、保护对为空、掩码不可观测保护对、task 与
    profile 身份不一致。
11. ``test_synthetic_scene_scrambled_layout``：free joint 位于主动关节
    之前 + 关节树序/执行器序乱序的合成模型，地址全部按名称解析。
"""

from __future__ import annotations

import dataclasses
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from src.robot.model_factory import load_robot_model
from src.robot.robot_profile import RobotProfile, load_robot_profile
from src.robot.runtime_parameters import (
    SceneParameters,
    RobotRuntimeParameters,
    load_runtime_parameters,
)
from src.sim.sim_config import SimTaskConfig, load_sim_task_config
from src.sim.tennis_scene import TennisSceneBundle, build_tennis_scene

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
_TEMPLATE_PATHS = {
    side: REPO_ROOT / "src" / "robot" / f"r1d_model_{side}.xml" for side in SIDES
}

# 字节未变校验的资产清单（模板 + 四份只读片段）
_ASSET_REL_PATHS = (
    "src/robot/r1d_arm_left.xml",
    "src/robot/r1d_arm_right.xml",
    "src/robot/r1d_env.xml",
    "src/robot/r1d_assets.xml",
    "src/robot/r1d_model_left.xml",
    "src/robot/r1d_model_right.xml",
)

# 球名称约定（与 r1d_sim_{side}.yaml 一致，沿 rm65_model.xml 命名先例）
_BALL_JOINT_NAME = "ball_freejoint"
_BALL_GEOM_NAME = "ball"

# 模板固定几何名（阶段 2 契约）
_TORSO_GEOM = "torso_box"
_GROUND_GEOM = "ground"
_RACKET_GEOM = "racket_face"
_ENV_CHAIN_GEOM = "base_link_visual"  # 环境链 mesh（保持视觉掩码 0/0）

# 该侧 7 个臂连杆 geom 名（模板实际名：仅 shoulder_pitch 连杆名含大写
# Link —— URDF mesh 文件名如此，其余为小写 link）
_ARM_GEOMS = {
    side: tuple(
        f"{side}_{link}_visual"
        for link in (
            "shoulder_pitch_Link",
            "shoulder_roll_link",
            "shoulder_yaw_link",
            "elbow_link",
            "wrist_roll_link",
            "wrist_pitch_link",
            "wrist_yaw_link",
        )
    )
    for side in SIDES
}

# 球物理参数来源（rm65_model.xml 逐字）：L304/L305/L308 球段、L38 class ball
_BALL_RADIUS = 0.033
_BALL_MASS = 0.058
_BALL_CONTYPE = 16
_BALL_CONAFFINITY = 10


# ---------- 加载 helper（不可变对象，进程内缓存避免重复编译） ----------


@lru_cache(maxsize=2)
def _profile(side: str) -> RobotProfile:
    """加载并缓存该侧 RobotProfile。"""
    return load_robot_profile(_PROFILE_YAML[side])


@lru_cache(maxsize=2)
def _parameters(side: str) -> RobotRuntimeParameters:
    """加载该侧仿真 runtime 参数。"""
    return load_runtime_parameters(_SIM_RUNTIME_YAML[side], _profile(side))


@lru_cache(maxsize=2)
def _task(side: str) -> SimTaskConfig:
    """加载并缓存该侧 SimTaskConfig。"""
    return load_sim_task_config(_TASK_YAML[side], _profile(side))


@lru_cache(maxsize=2)
def _scene(side: str) -> TennisSceneBundle:
    """构建并缓存该侧带球场景（双侧 happy path 复用同一实例）。"""
    return build_tennis_scene(_profile(side), _parameters(side), _task(side))


@lru_cache(maxsize=2)
def _bare_bundle(side: str) -> Any:
    """构建并缓存同参数裸模型（阶段 2 load_robot_model，几何/惯量对照基准）。"""
    return load_robot_model(_profile(side), _parameters(side))


def _name2id(
    model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str
) -> int:
    """按名称查 id（测试内不期待缺失，-1 直接返回由断言暴露）。"""
    return int(mujoco.mj_name2id(model, obj_type, name))


def _contact_geom_pairs(data: mujoco.MjData) -> set[tuple[int, int]]:
    """收集当前接触的 geom id 对（无序对，小 id 在前）。"""
    return {
        (min(int(c.geom1), int(c.geom2)), max(int(c.geom1), int(c.geom2)))
        for i in range(data.ncon)
        for c in (data.contact[i],)
    }


def _masks_allow(
    model: mujoco.MjModel, geom1: int, geom2: int
) -> bool:
    """按 MuJoCo 位掩码规则判断两 geom 是否允许动态碰撞。"""
    ct1, ca1 = int(model.geom_contype[geom1]), int(model.geom_conaffinity[geom1])
    ct2, ca2 = int(model.geom_contype[geom2]), int(model.geom_conaffinity[geom2])
    return (ct1 & ca2) != 0 or (ct2 & ca1) != 0


# ---------- 1. 维数与按名地址 ----------


@pytest.mark.parametrize("side", SIDES)
def test_scene_shape_and_named_addresses(side: str) -> None:
    """带球模型主动关节仍 7、受控状态仍 14；nq/nv/nu 典型值且地址按名称解析。"""
    scene = _scene(side)
    profile, model, binding = scene.robot.profile, scene.robot.model, scene.robot.binding

    # 主动关节 7、受控状态 14（n + n）
    assert profile.ndof == 7
    assert len(binding.joint_ids) == 7
    assert len(binding.qpos_indices) == 7
    assert len(binding.qvel_indices) == 7
    assert len(binding.actuator_ids) == 7
    assert len(binding.qpos_indices) + len(binding.qvel_indices) == 2 * profile.ndof

    # 典型带球维数（球 free joint: +7 qpos / +6 dof）
    assert model.nq == 14
    assert model.nv == 13
    assert model.nu == 7
    assert model.njnt == 8

    # 关节/执行器地址逐名对照模型 jnt_qposadr/jnt_dofadr（不是位置假设）
    for joint_name, jid, qidx, vidx in zip(
        profile.active_joint_names,
        binding.joint_ids,
        binding.qpos_indices,
        binding.qvel_indices,
    ):
        expected = _name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert jid == expected
        assert int(model.jnt_qposadr[jid]) == qidx
        assert int(model.jnt_dofadr[jid]) == vidx

    # 球地址按名解析（free joint 的 qpos 起点/速度起点）
    ball_id = _name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _BALL_JOINT_NAME)
    assert scene.ball_joint_id == ball_id
    assert scene.ball_qpos_start == int(model.jnt_qposadr[ball_id])
    assert scene.ball_qvel_start == int(model.jnt_dofadr[ball_id])

    # 裸模型（阶段 2）无球；场景 id 不复用另一模型（按名回查一致）
    bare = _bare_bundle(side)
    assert bare.model.nq == 7
    assert _name2id(bare.model, mujoco.mjtObj.mjOBJ_JOINT, _BALL_JOINT_NAME) == -1
    assert (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, scene.ball_geom_id)
        == _BALL_GEOM_NAME
    )
    for racket_id in scene.racket_geom_ids:
        assert (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, racket_id)
            == _RACKET_GEOM
        )
    # 保护对 id 按名回查与任务配置一致
    task = _task(side)
    assert len(scene.protected_geom_pairs) == len(task.protected_geom_pairs)
    for (name1, name2), (g1, g2) in zip(
        task.protected_geom_pairs, scene.protected_geom_pairs
    ):
        assert _name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name1) == g1
        assert _name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name2) == g2


# ---------- 2. 球在前 + quaternion 不入主动绑定 ----------


@pytest.mark.parametrize("side", SIDES)
def test_ball_free_joint_precedes_active_joints(side: str) -> None:
    """球 free joint 位于主动关节之前；主动绑定不包含球 quaternion。"""
    scene = _scene(side)
    model, binding = scene.robot.model, scene.robot.binding

    # 球关节为 free 类型
    assert (
        int(model.jnt_type[scene.ball_joint_id])
        == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    # 布局：球 qpos 占据 [0,7)，主动关节全部位于其后
    assert scene.ball_qpos_start == 0
    assert min(binding.qpos_indices) >= 7
    assert sorted(binding.qpos_indices) == list(range(7, 14))
    # 球 quaternion（wxyz 在球 qpos 段后 4 项 = 索引 3..6）不得进入主动绑定
    ball_quat_indices = {
        scene.ball_qpos_start + 3 + k for k in range(4)
    }
    assert ball_quat_indices.isdisjoint(set(binding.qpos_indices))


# ---------- 3. 球物理参数与初位姿（rm65 来源） ----------


@pytest.mark.parametrize("side", SIDES)
def test_ball_physics_match_rm65_source(side: str) -> None:
    """球尺寸/质量/掩码/rgba 与 rm65_model.xml 逐字来源一致；初位姿写入 qpos0。"""
    scene = _scene(side)
    model, task = scene.robot.model, _task(side)

    ball_geom = scene.ball_geom_id
    assert int(model.geom_type[ball_geom]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    assert float(model.geom_size[ball_geom, 0]) == pytest.approx(_BALL_RADIUS)
    # 质量：球 body 单 geom 显式 mass → body_mass
    ball_body = int(model.geom_bodyid[ball_geom])
    assert float(model.body_mass[ball_body]) == pytest.approx(_BALL_MASS)
    # 掩码：rm65 class="ball" 逐字（contype=16, conaffinity=10）
    assert int(model.geom_contype[ball_geom]) == _BALL_CONTYPE
    assert int(model.geom_conaffinity[ball_geom]) == _BALL_CONAFFINITY
    # 初位姿进入 qpos0（球 body pos = task.initial_ball_position，四元数单位）
    assert np.array_equal(
        model.qpos0[scene.ball_qpos_start : scene.ball_qpos_start + 3],
        task.initial_ball_position,
    )
    assert np.array_equal(
        model.qpos0[scene.ball_qpos_start + 3 : scene.ball_qpos_start + 7],
        np.array([1.0, 0.0, 0.0, 0.0]),
    )


# ---------- 4. 碰撞掩码语义 ----------


@pytest.mark.parametrize("side", SIDES)
def test_collision_mask_semantics(side: str) -> None:
    """逐 geom 掩码断言 + 派生允许/禁止接触表（碰撞语义绑定）。"""
    scene = _scene(side)
    model = scene.robot.model

    geom_id = {name: _name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in (
        _GROUND_GEOM, _TORSO_GEOM, _RACKET_GEOM, _ENV_CHAIN_GEOM,
        _BALL_GEOM_NAME, *_ARM_GEOMS[side],
    )}
    assert all(gid >= 0 for gid in geom_id.values())

    # 逐 geom 掩码（位分配沿 rm65 方案：地面1/机身2/臂4/拍8/球16）
    expected_masks = {
        _GROUND_GEOM: (1, 31),        # 地面万能碰撞
        _TORSO_GEOM: (2, 29),         # 机身：地面|臂|拍|球
        _RACKET_GEOM: (8, 16),        # 拍：球
        _BALL_GEOM_NAME: (16, 10),    # 球：机身|拍（沿 rm65 class ball）
        _ENV_CHAIN_GEOM: (0, 0),      # 环境链 mesh 保持视觉掩码（不宣称整机完备）
        **{name: (4, 2) for name in _ARM_GEOMS[side]},  # 臂：仅机身（保护对可观测）
    }
    for name, (ct, ca) in expected_masks.items():
        gid = geom_id[name]
        assert int(model.geom_contype[gid]) == ct, f"{name} contype 应为 {ct}"
        assert int(model.geom_conaffinity[gid]) == ca, f"{name} conaffinity 应为 {ca}"

    # 派生允许/禁止表（掩码判定，实际接触由物理测试另行验证）
    should_allow = (
        (_ARM_GEOMS[side][0], _TORSO_GEOM),     # 臂—机身（保护对，必须可观测）
        (_ARM_GEOMS[side][3], _TORSO_GEOM),     # 肘—机身
        (_RACKET_GEOM, _TORSO_GEOM),            # 拍—机身（保护对）
        (_BALL_GEOM_NAME, _TORSO_GEOM),         # 球—机身
        (_BALL_GEOM_NAME, _RACKET_GEOM),        # 球—拍（击球接触）
        (_BALL_GEOM_NAME, _GROUND_GEOM),        # 球—地面（落地反弹）
        (_ARM_GEOMS[side][0], _GROUND_GEOM),    # 臂—地面（沿 rm65 地面万能）
        (_RACKET_GEOM, _GROUND_GEOM),           # 拍—地面
    )
    should_deny = (
        (_ARM_GEOMS[side][0], _ARM_GEOMS[side][6]),  # 臂—臂（非相邻段不开启）
        (_ARM_GEOMS[side][0], _BALL_GEOM_NAME),      # 臂—球（沿 rm65：球不与臂碰）
        (_ARM_GEOMS[side][0], _RACKET_GEOM),         # 臂—拍（相邻挂接）
        (_ENV_CHAIN_GEOM, _BALL_GEOM_NAME),          # 环境链—球
        (_ENV_CHAIN_GEOM, _ARM_GEOMS[side][0]),      # 环境链—臂
        (_ENV_CHAIN_GEOM, _TORSO_GEOM),              # 环境链—机身
    )
    for name1, name2 in should_allow:
        assert _masks_allow(model, geom_id[name1], geom_id[name2]), (
            f"{name1} 与 {name2} 掩码应允许接触"
        )
    for name1, name2 in should_deny:
        assert not _masks_allow(model, geom_id[name1], geom_id[name2]), (
            f"{name1} 与 {name2} 掩码应禁止接触"
        )


# ---------- 5. 球—拍弹性 pair 与单独控制通道 ----------


@pytest.mark.parametrize("side", SIDES)
def test_racket_ball_pair_params_and_toggle_channel(side: str) -> None:
    """球—拍显式 pair 参数（rm65 逐字）实际生效；pair_margin 是单独控制通道。"""
    scene = _scene(side)
    model = scene.robot.model

    # 找到球—拍 pair（每个拍 geom 一条）
    pair_indices = [
        i
        for i in range(model.npair)
        if {int(model.pair_geom1[i]), int(model.pair_geom2[i])}
        == {scene.ball_geom_id, scene.racket_geom_ids[0]}
    ]
    assert pair_indices, "模型中应存在球—拍显式 pair"
    pi = pair_indices[0]
    # rm65_model.xml L75 逐字：condim=1（无摩擦法向），solref=-5000 0（零阻尼弹性）；
    # solimp 编译后为 5 参数（XML 3 参数 + 默认 midpoint/power 0.5/2）
    assert int(model.pair_dim[pi]) == 1
    assert np.allclose(model.pair_solref[pi], (-5000.0, 0.0))
    assert np.allclose(model.pair_solimp[pi][:3], (0.9, 0.99, 0.001))
    assert np.allclose(model.pair_solimp[pi][3:], (0.5, 2.0))

    # 物理验证：把球放到拍面上方轻微穿透 → 生成 dim=1 的弹性接触
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    racket_center = data.site_xpos[scene.robot.binding.tcp_site_id].copy()
    normal = data.geom_xmat[scene.racket_geom_ids[0]].reshape(3, 3)[:, 2].copy()
    data.qpos[scene.ball_qpos_start : scene.ball_qpos_start + 3] = (
        racket_center + normal * 0.01
    )
    data.qpos[scene.ball_qpos_start + 3 : scene.ball_qpos_start + 7] = (1.0, 0, 0, 0)
    mujoco.mj_forward(model, data)
    racket_ball = {
        (min(int(c.geom1), int(c.geom2)), max(int(c.geom1), int(c.geom2)))
        for i in range(data.ncon)
        for c in (data.contact[i],)
        if {int(c.geom1), int(c.geom2)}
        == {scene.ball_geom_id, scene.racket_geom_ids[0]}
    }
    assert racket_ball, "球—拍穿透应生成接触"
    contact_dim = next(
        int(c.dim)
        for i in range(data.ncon)
        for c in (data.contact[i],)
        if {int(c.geom1), int(c.geom2)}
        == {scene.ball_geom_id, scene.racket_geom_ids[0]}
    )
    assert contact_dim == 1, "球—拍接触应使用 pair 的 condim=1（无摩擦法向）"

    # 单独控制通道：pair_margin 置负 → 仅球—拍接触消失（掩码无关）；
    # 恢复 → 接触回来。该通道供 3.4 set_arm_collision 使用。
    original_margin = float(model.pair_margin[pi])
    model.pair_margin[pi] = -1.0
    mujoco.mj_forward(model, data)
    assert not {
        i
        for i in range(data.ncon)
        if {int(data.contact[i].geom1), int(data.contact[i].geom2)}
        == {scene.ball_geom_id, scene.racket_geom_ids[0]}
    }, "pair_margin 关闭后球—拍接触应消失"
    model.pair_margin[pi] = original_margin
    mujoco.mj_forward(model, data)
    assert {
        i
        for i in range(data.ncon)
        if {int(data.contact[i].geom1), int(data.contact[i].geom2)}
        == {scene.ball_geom_id, scene.racket_geom_ids[0]}
    }, "pair_margin 恢复后球—拍接触应回来"


# ---------- 6. 初态无禁止接触 ----------


@pytest.mark.parametrize("side", SIDES)
def test_initial_state_has_no_forbidden_contact(side: str) -> None:
    """init_q 下（球在任务初位姿）全部保护对无接触。"""
    scene = _scene(side)
    model, binding = scene.robot.model, scene.robot.binding

    data = mujoco.MjData(model)
    data.qpos[list(binding.qpos_indices)] = scene.robot.parameters.init_q
    mujoco.mj_forward(model, data)
    pairs = _contact_geom_pairs(data)
    for g1, g2 in scene.protected_geom_pairs:
        assert (min(g1, g2), max(g1, g2)) not in pairs, (
            f"初态出现禁止接触: geom {g1} ↔ {g2}"
        )


# ---------- 7. 真实穿透负例可被观测 ----------


@pytest.mark.parametrize("side", SIDES)
def test_real_penetration_negative_observable(side: str) -> None:
    """负例可观测：球塞入机身、臂/拍扫掠出与机身 torso_box 的保护对接触。"""
    scene = _scene(side)
    model, binding = scene.robot.model, scene.robot.binding
    torso_geom = _name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _TORSO_GEOM)

    # 负例 A：球直接放到机身 box 中心 → 球—机身接触出现
    data = mujoco.MjData(model)
    data.qpos[list(binding.qpos_indices)] = scene.robot.parameters.init_q
    mujoco.mj_forward(model, data)
    torso_center = data.xpos[int(model.geom_bodyid[torso_geom])].copy()
    data.qpos[scene.ball_qpos_start : scene.ball_qpos_start + 3] = torso_center
    mujoco.mj_forward(model, data)
    assert (min(torso_geom, scene.ball_geom_id), max(torso_geom, scene.ball_geom_id)) in (
        _contact_geom_pairs(data)
    ), "球穿入机身应产生可观测接触"

    # 负例 B：肩俯仰 × 肘 网格扫描，必存在臂 geom/拍与机身 torso_box 的
    # 保护对接触（真实穿透可被 3.7 has_forbidden_contact 观测的前提）
    arm_geoms = {
        _name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in _ARM_GEOMS[side]
    } | set(scene.racket_geom_ids)
    found: list[tuple[float, float, int]] = []
    q_backup = data.qpos.copy()
    for pitch in np.arange(0.4, 2.05, 0.15):
        for elbow in np.arange(-0.95, 2.15, 0.15):
            data.qpos[:] = q_backup
            data.qpos[binding.qpos_indices[0]] = float(pitch)
            data.qpos[binding.qpos_indices[3]] = float(elbow)
            mujoco.mj_forward(model, data)
            for g1, g2 in _contact_geom_pairs(data):
                if torso_geom in (g1, g2) and (g1 if g2 == torso_geom else g2) in arm_geoms:
                    found.append((float(pitch), float(elbow), int(g1)))
    assert found, "肩俯仰×肘扫描应找到臂/拍与机身的保护对穿透配置"
    # 扫描到的构型必须在关节限位内（真实可达，不是靠越界 qpos 制造的假象）
    lower, upper = scene.robot.parameters.q_lower, scene.robot.parameters.q_upper
    for pitch, elbow, _ in found[:5]:
        assert lower[0] <= pitch <= upper[0]
        assert lower[3] <= elbow <= upper[3]


# ---------- 8. 几何与惯量对照阶段 2 同参数模型 ----------


@pytest.mark.parametrize("side", SIDES)
def test_geometry_and_inertia_match_stage2_model(side: str) -> None:
    """带球模型与阶段 2 同参数裸模型逐名比较几何/惯量/限位完全一致。"""
    scene = _scene(side)
    bare = _bare_bundle(side)
    model, bare_model = scene.robot.model, bare.model

    # 裸模型全部名称应存在于带球模型（仅多出球 body/geom/joint）
    def _names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, size: int) -> list[str]:
        return [
            mujoco.mj_id2name(model, obj_type, i) for i in range(size)
        ]

    for obj_type, size_m, size_b, label in (
        (mujoco.mjtObj.mjOBJ_BODY, model.nbody, bare_model.nbody, "body"),
        (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom, bare_model.ngeom, "geom"),
        (mujoco.mjtObj.mjOBJ_SITE, model.nsite, bare_model.nsite, "site"),
        (mujoco.mjtObj.mjOBJ_JOINT, model.njnt, bare_model.njnt, "joint"),
    ):
        scene_names = set(_names(model, obj_type, size_m))
        bare_names = set(_names(bare_model, obj_type, size_b))
        assert bare_names <= scene_names, f"裸模型 {label} 名称应全部保留"

    # body：质量/惯量/质心/位姿逐名一致
    for bid_b in range(bare_model.nbody):
        name = mujoco.mj_id2name(bare_model, mujoco.mjtObj.mjOBJ_BODY, bid_b)
        bid_s = _name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
        for field in ("body_mass", "body_inertia", "body_ipos", "body_pos", "body_quat"):
            assert np.array_equal(
                getattr(model, field)[bid_s], getattr(bare_model, field)[bid_b]
            ), f"body {name} 的 {field} 与阶段 2 模型不一致"

    # geom：类型/尺寸/位姿逐名一致（掩码差异另由掩码测试断言，此处只查几何）
    for gid_b in range(bare_model.ngeom):
        name = mujoco.mj_id2name(bare_model, mujoco.mjtObj.mjOBJ_GEOM, gid_b)
        gid_s = _name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(name))
        for field in ("geom_type", "geom_size", "geom_pos", "geom_quat"):
            assert np.array_equal(
                getattr(model, field)[gid_s], getattr(bare_model, field)[gid_b]
            ), f"geom {name} 的 {field} 与阶段 2 模型不一致"

    # site：位置逐名一致（racket_center/r1d_flange）
    for sid_b in range(bare_model.nsite):
        name = mujoco.mj_id2name(bare_model, mujoco.mjtObj.mjOBJ_SITE, sid_b)
        sid_s = _name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(name))
        assert np.array_equal(model.site_pos[sid_s], bare_model.site_pos[sid_b])

    # 主动关节限位逐名一致；回填后的参数限位也一致
    for jid_b in range(bare_model.njnt):
        name = mujoco.mj_id2name(bare_model, mujoco.mjtObj.mjOBJ_JOINT, jid_b)
        jid_s = _name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        assert np.array_equal(model.jnt_range[jid_s], bare_model.jnt_range[jid_b])
    assert np.array_equal(
        scene.robot.parameters.q_lower, bare.parameters.q_lower
    )
    assert np.array_equal(
        scene.robot.parameters.q_upper, bare.parameters.q_upper
    )


# ---------- 9. 模板/资产字节未变 ----------


def test_template_and_asset_bytes_unchanged() -> None:
    """组装只动内存副本：模板与片段字节与 git HEAD 一致。"""
    # 先构建双侧场景（触发组装路径），再核对磁盘字节
    _scene("left")
    _scene("right")
    for rel in _ASSET_REL_PATHS:
        current = (REPO_ROOT / rel).read_bytes()
        committed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"HEAD:{rel}"],
            capture_output=True,
            check=True,
        ).stdout
        assert current == committed, f"{rel} 字节与 git HEAD 不一致"


# ---------- 10. 负例：类型/名称/身份失败 ----------


def test_ball_joint_name_type_conflict_rejected(tmp_path: Path) -> None:
    """模板中已存在同名 hinge 关节（free joint 类型错误场景）→ 构建失败。"""
    template = _TEMPLATE_PATHS["left"].read_text(encoding="utf-8")
    mutated = template.replace(
        'name="left_elbow_joint"', f'name="{_BALL_JOINT_NAME}"'
    )
    assert mutated != template, "测试前提：模板中应存在该关节名"
    tmp_model = tmp_path / "synthetic_conflict.xml"
    tmp_model.write_text(mutated, encoding="utf-8")
    profile = dataclasses.replace(_profile("left"), model_path=tmp_model)
    with pytest.raises(ValueError, match=_BALL_JOINT_NAME):
        build_tennis_scene(profile, _parameters("left"), _task("left"))


@pytest.mark.parametrize("side", SIDES)
def test_missing_geom_names_rejected(side: str) -> None:
    """racket geom / 保护对 geom 名称缺失、球 geom 名与模板冲突 → 构建失败。"""
    profile, parameters = _profile(side), _parameters(side)
    # racket geom 名不存在（拼写错误）
    task_typo = dataclasses.replace(_task(side), racket_geom_names=("racket_fac",))
    with pytest.raises(ValueError, match="racket_fac"):
        build_tennis_scene(profile, parameters, task_typo)
    # 保护对引用不存在的 geom
    arm_geom = _ARM_GEOMS[side][3]
    task_pair = dataclasses.replace(
        _task(side),
        protected_geom_pairs=((arm_geom, "torso_baxe"),),
    )
    with pytest.raises(ValueError, match="torso_baxe"):
        build_tennis_scene(profile, parameters, task_pair)
    # 球 geom 名与模板既有 geom 冲突（ground）
    task_ball = dataclasses.replace(_task(side), ball_geom_name=_GROUND_GEOM)
    with pytest.raises(ValueError, match=_GROUND_GEOM):
        build_tennis_scene(profile, parameters, task_ball)


@pytest.mark.parametrize("side", SIDES)
def test_empty_protected_pairs_rejected(side: str) -> None:
    """保护对为空 → 构造即拒绝（场景链路同样不得放行）。"""
    with pytest.raises(ValueError, match="protected_geom_pairs"):
        dataclasses.replace(_task(side), protected_geom_pairs=())


@pytest.mark.parametrize("side", SIDES)
def test_unobservable_protected_pair_rejected(side: str) -> None:
    """掩码不相容（臂—臂对）的保护对不可观测 → 构建失败（防空列表式放行）。"""
    arm1, arm2 = _ARM_GEOMS[side][1], _ARM_GEOMS[side][5]
    task = dataclasses.replace(
        _task(side), protected_geom_pairs=((arm1, arm2),)
    )
    with pytest.raises(ValueError, match="掩码"):
        build_tennis_scene(_profile(side), _parameters(side), task)


@pytest.mark.parametrize("side", SIDES)
def test_task_profile_identity_mismatch_rejected(side: str) -> None:
    """task 与 profile 身份/关节序不一致 → 构建失败（防御手构对象）。"""
    profile, parameters = _profile(side), _parameters(side)
    wrong_id = dataclasses.replace(_task(side), robot_id="somebody_else")
    with pytest.raises(ValueError, match="robot_id"):
        build_tennis_scene(profile, parameters, wrong_id)
    flipped = _task(side).joint_names[::-1]
    wrong_order = dataclasses.replace(_task(side), joint_names=flipped)
    with pytest.raises(ValueError, match="joint_names"):
        build_tennis_scene(profile, parameters, wrong_order)


# ---------- 11. 合成模型：球在前 + 关节/执行器乱序 ----------


def test_synthetic_scene_scrambled_layout(tmp_path: Path) -> None:
    """free joint 位于主动关节之前 + 关节树序/执行器乱序 → 全部按名称解析。

    合成模板契约与 R1D 模板一致（r1d_mount/r1d_racket/r1d_torso/
    racket_face/torso_box/ground/r1d_flange），但：
    - 臂链 7 个 hinge 的**树序**为 j2,j5,j1,j7,j3,j6,j4（与 profile 序不同）；
    - actuator 段乱序（m_j4,m_j1,m_j7,m_j3,m_j6,m_j2,m_j5）。
    """
    # 树序（XML 深度序）与 profile 序
    tree_order = ("j2", "j5", "j1", "j7", "j3", "j6", "j4")
    profile_order = ("j1", "j2", "j3", "j4", "j5", "j6", "j7")
    actuator_xml_order = ("m_j4", "m_j1", "m_j7", "m_j3", "m_j6", "m_j2", "m_j5")

    def _link_body(joint: str, child: str | None) -> str:
        body = f'      <body name="link_{joint}" pos="0 0 -0.08">\n'
        body += (
            f'        <joint name="{joint}" type="hinge" axis="0 1 0"'
            ' range="-170 170" />\n'
            f'        <geom name="geom_{joint}" type="capsule"'
            ' fromto="0 0 0 0 0 -0.08" size="0.01" contype="0" conaffinity="0" />\n'
        )
        body += child or ""
        body += "      </body>\n"
        return body

    # 由内向外构建臂链（j4 最内层带法兰与球拍）
    innermost = (
        '        <site name="r1d_flange" pos="0 0 -0.05" size="0.005" />\n'
        '        <body name="r1d_racket" pos="0 0 -0.05" quat="1 0 0 0">\n'
        '          <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.002" />\n'
        '          <geom name="racket_face" type="cylinder" size="0 0"'
        ' contype="0" conaffinity="0" />\n'
        '          <site name="racket_center" pos="0 0 0" size="0.005" />\n'
        "        </body>\n"
    )
    # 由内向外折叠臂链：innermost（法兰+球拍）属于 link_j4，逐层包到 link_j2
    chain = innermost
    for joint in reversed(tree_order):
        chain = _link_body(joint, chain)

    actuators = "\n".join(
        f'    <motor name="{name}" joint="{name[2:]}" gear="1" />'
        for name in actuator_xml_order
    )
    xml = f"""<mujoco model="synthetic_r1d_tennis">
  <compiler angle="degree" />
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" />
  <worldbody>
    <geom name="ground" type="plane" size="2 2 0.1" contype="0" conaffinity="0" />
    <body name="r1d_mount" pos="0 0 0.5" quat="1 0 0 0">
{chain}      <body name="r1d_torso" pos="0 0 0">
        <geom name="torso_box" type="box" size="0 0" contype="0" conaffinity="0" />
      </body>
    </body>
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""

    model_path = tmp_path / "synthetic_scrambled.xml"
    model_path.write_text(xml, encoding="utf-8")

    profile = RobotProfile(
        schema_version=1,
        robot_id="unitree_r1d",
        model_path=model_path,
        active_joint_names=profile_order,
        actuator_names=tuple(f"m_{j}" for j in profile_order),
        held_joint_names=(),
        tcp_site_name="racket_center",
    )
    parameters = RobotRuntimeParameters(
        robot_id="unitree_r1d",
        joint_names=profile_order,
        dt=0.005,
        init_q=np.zeros(7),
        held_q=np.array([]),
        kp=None,
        kd=None,
        q_lower=None,
        q_upper=None,
        qdot_max=None,
        qddot_max=None,
        torque_max=None,
        max_tcp_speed=None,
        scene=SceneParameters(
            base_position=np.zeros(3),
            base_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            racket_position=np.array([0.0, 0.0, 0.1]),
            racket_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            racket_radius=0.05,
            racket_half_thickness=0.005,
            racket_mass=0.1,
            torso_position=np.array([-0.2, 0.0, -0.1]),
            torso_half_size=np.array([0.1, 0.1, 0.2]),
        ),
        provisional_fields=(),
        source="synthetic-test",
    )
    task = SimTaskConfig(
        robot_id="unitree_r1d",
        joint_names=profile_order,
        ball_joint_name=_BALL_JOINT_NAME,
        ball_geom_name=_BALL_GEOM_NAME,
        racket_geom_names=(_RACKET_GEOM,),
        protected_geom_pairs=(("geom_j3", _TORSO_GEOM), (_RACKET_GEOM, _TORSO_GEOM)),
        workspace_center=np.zeros(3),
        workspace_radius=1.0,
        initial_ball_position=np.array([0.5, 0.0, 0.5]),
        initial_ball_velocity=np.zeros(3),
        max_steps=10,
        actuator_mode="torque",
        solver_config={},
        mpc_config={},
        cost_config={},
    )

    scene = build_tennis_scene(profile, parameters, task)
    model, binding = scene.robot.model, scene.robot.binding

    # 典型维数 + 球在前
    assert model.nq == 14 and model.nv == 13 and model.nu == 7 and model.njnt == 8
    assert scene.ball_qpos_start == 0
    assert min(binding.qpos_indices) >= 7

    # 关节地址按名解析：树序 j2,j5,j1,j7,j3,j6,j4 → qpos 地址 7..13 依树序
    expected_qpos = {
        joint: 7 + rank for rank, joint in enumerate(tree_order)
    }
    # 主动绑定地址 = 每个名字在模型中的真实地址（非 0..6 前缀假设）
    assert binding.qpos_indices == tuple(
        expected_qpos[name] for name in profile_order
    )
    assert sorted(binding.qpos_indices) == list(range(7, 14))
    # 绑定序不是单调递增（证明乱序被按名消化而非碰巧对齐）
    assert list(binding.qpos_indices) != sorted(binding.qpos_indices)

    # 执行器乱序：actuator id 按 profile 名解析（非 0..6 顺序）
    expected_actuators = {
        name: rank for rank, name in enumerate(actuator_xml_order)
    }
    assert binding.actuator_ids == tuple(
        expected_actuators[f"m_{name}"] for name in profile_order
    )
    assert list(binding.actuator_ids) != sorted(binding.actuator_ids)

    # 球为 free 类型；主动绑定不含球 quaternion
    assert (
        int(model.jnt_type[scene.ball_joint_id])
        == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    quat_indices = {3, 4, 5, 6}
    assert quat_indices.isdisjoint(set(binding.qpos_indices))

    # 保护对与拍 geom 按名解析成功
    torso_gid = _name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _TORSO_GEOM)
    j3_gid = _name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "geom_j3")
    assert scene.protected_geom_pairs[0] == (j3_gid, torso_gid)
    assert scene.racket_geom_ids == (
        _name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _RACKET_GEOM),
    )
