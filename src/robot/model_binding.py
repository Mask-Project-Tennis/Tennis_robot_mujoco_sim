"""MuJoCo 模型名称绑定器 — 把 RobotProfile 的语义名称解析为模型地址。

只读解析：按名称查关节/执行器/TCP 的 id 与 qpos/qvel 地址，
不修改、不重排 MjModel/MjData。仅接受标量关节（hinge/slide），
free/ball 关节不能当作一维读取。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from src.robot.robot_profile import RobotProfile

# 可作为主动/保持关节的标量关节类型（free/ball 的 qpos/qvel 维度非 1）
_SCALAR_JOINT_TYPES = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)


@dataclass(frozen=True)
class ModelBinding:
    """模型名称 → 地址的绑定结果。

    Attributes:
        joint_ids: 主动关节 id（按 profile.active_joint_names 顺序）。
        qpos_indices: 主动关节 qpos 地址。
        qvel_indices: 主动关节 qvel（dof）地址。
        actuator_ids: 执行器 id（按 profile.actuator_names 顺序，
            与主动关节一一对应且已校验传动目标）。
        held_qpos_indices: 保持关节 qpos 地址。
        held_qvel_indices: 保持关节 qvel 地址。
        tcp_site_id: TCP site id。
    """

    joint_ids: tuple[int, ...]
    qpos_indices: tuple[int, ...]
    qvel_indices: tuple[int, ...]
    actuator_ids: tuple[int, ...]
    held_qpos_indices: tuple[int, ...]
    held_qvel_indices: tuple[int, ...]
    tcp_site_id: int


def bind_robot_model(profile: RobotProfile, model: mujoco.MjModel) -> ModelBinding:
    """把机器人描述绑定到 MuJoCo 模型，解析全部名称地址。

    Args:
        profile: 机器人描述（语义名称）。
        model: 已加载的 MuJoCo 模型（只读，不修改）。

    Returns:
        不可变的 ModelBinding 实例。

    Raises:
        ValueError: 任一名称在模型中不存在、执行器传动目标与期望关节不符、
            或主动/保持关节为非标量（free/ball）类型。
    """
    joint_ids, qpos_indices, qvel_indices = _resolve_joints(
        profile, model, profile.active_joint_names
    )
    actuator_ids = _resolve_actuators(profile, model, joint_ids)
    _, held_qpos, held_qvel = _resolve_joints(
        profile, model, profile.held_joint_names
    )
    tcp_site_id = _lookup(model, mujoco.mjtObj.mjOBJ_SITE, profile.tcp_site_name, profile)

    return ModelBinding(
        joint_ids=joint_ids,
        qpos_indices=qpos_indices,
        qvel_indices=qvel_indices,
        actuator_ids=actuator_ids,
        held_qpos_indices=held_qpos,
        held_qvel_indices=held_qvel,
        tcp_site_id=tcp_site_id,
    )


def _lookup(
    model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str, profile: RobotProfile
) -> int:
    """按名称查 id，查不到即报错（含名称与 robot_id）。"""
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        kind = {
            mujoco.mjtObj.mjOBJ_JOINT: "关节",
            mujoco.mjtObj.mjOBJ_ACTUATOR: "执行器",
            mujoco.mjtObj.mjOBJ_SITE: "site",
        }[obj_type]
        raise ValueError(
            f"模型中不存在{kind} '{name}'（robot_id: {profile.robot_id}，"
            f"模型: {profile.model_path}）"
        )
    return int(obj_id)


def _resolve_joints(
    profile: RobotProfile, model: mujoco.MjModel, names: tuple[str, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """解析一组关节名的 id 与 qpos/qvel 地址，校验标量类型。"""
    joint_ids: list[int] = []
    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    for name in names:
        jid = _lookup(model, mujoco.mjtObj.mjOBJ_JOINT, name, profile)
        jnt_type = int(model.jnt_type[jid])
        if jnt_type not in (int(t) for t in _SCALAR_JOINT_TYPES):
            raise ValueError(
                f"关节 '{name}' 类型非标量（free/ball 不能按一维读取，"
                f"robot_id: {profile.robot_id}，模型: {profile.model_path}）"
            )
        joint_ids.append(jid)
        qpos_indices.append(int(model.jnt_qposadr[jid]))
        qvel_indices.append(int(model.jnt_dofadr[jid]))
    return tuple(joint_ids), tuple(qpos_indices), tuple(qvel_indices)


def _resolve_actuators(
    profile: RobotProfile, model: mujoco.MjModel, joint_ids: tuple[int, ...]
) -> tuple[int, ...]:
    """解析执行器 id 并校验每个执行器的传动目标正是对应主动关节。"""
    actuator_ids: list[int] = []
    for name, expected_joint_id in zip(profile.actuator_names, joint_ids):
        aid = _lookup(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name, profile)
        trn_type = int(model.actuator_trntype[aid])
        target_joint = int(model.actuator_trnid[aid, 0])
        if trn_type != int(mujoco.mjtTrn.mjTRN_JOINT) or target_joint != expected_joint_id:
            raise ValueError(
                f"执行器 '{name}' 未驱动期望的关节（期望关节 id "
                f"{expected_joint_id}，实际传动目标 id {target_joint}，"
                f"robot_id: {profile.robot_id}，模型: {profile.model_path}）"
            )
        actuator_ids.append(aid)
    return tuple(actuator_ids)
