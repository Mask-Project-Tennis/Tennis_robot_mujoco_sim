"""R1D 场景数值写入与装配后校验。

负责球场地面/白线口径、场景尺寸、升降位移重烘、固定关节姿态注入、关节
限位填充以及模型形状/初始状态校验。

分层：本模块不 import ``loading``；``_apply_scene_parameters`` 在调用期
延迟 import ``_apply_fixed_joint_poses``（固定姿态重烘位于加载层）。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from src.robot.court_geometry import COURT_LINE_Z_M, COURT_PROFILES, build_court_lines
from src.robot.model_binding import ModelBinding
from src.robot.model_building.racket import (
    _apply_legacy_racket,
    _apply_racket_profile,
    _has_legacy_racket,
    _racket_assembly,
    _require_no_legacy_racket,
)
from src.robot.model_building.template import (
    _BASE_LANDING_HEIGHT_M,
    _COURT_GROUND_RGBA,
    _COURT_MATERIAL,
    _COURT_TEXTURE,
    _ENV_BAKED_LIFT_M,
    _MOUNT_BODY,
    _TORSO_BODY,
    _find_body,
    _find_body_optional,
    _flange_position,
    _fmt_float,
    _fmt_vec,
    _model_side,
    _require_scene,
)
from src.robot.model_building.types import RacketAssembly
from src.robot.racket_profile import RacketProfile
from src.robot.robot_profile import RobotProfile
from src.robot.runtime_parameters import RobotRuntimeParameters


def _court_profile_of(parameters: RobotRuntimeParameters) -> str:
    """返回场景白线 profile（无 scene 的旧风格按 legacy_v1，保持旧渲染）。

    Args:
        parameters: R1D 运行参数（scene 可为 None，如历史 RM65 风格输入）。

    Returns:
        ``legacy_v1`` 或 ``itf_2026_v1``；未知值立即抛 ValueError。
    """
    scene = parameters.scene
    profile = "legacy_v1" if scene is None else str(scene.court_profile)
    if profile not in COURT_PROFILES:
        raise ValueError(
            f"未知球场 profile {profile!r}（合法: {COURT_PROFILES}）"
        )
    return profile


def _neutralize_court_ground(root: ET.Element) -> None:
    """把旧 RM65 的棋盘/白色网格纹理地面改为中性纯色（ITF profile 用）。

    旧 ``court_grid`` 纹理带 ``mark="edge"``，在俯视图上会形成球场外的
    浅色网格伪线。标准 profile 只保留中性绿色地面，因此移除棋盘边缘
    标记与地面材质的纹理引用；白线本身另由 ``court_geometry`` 生成。
    只改视觉资产，不改任何碰撞/物理属性。
    """
    ground_rgb = " ".join(_COURT_GROUND_RGBA.split()[:3])
    for texture in root.findall("asset/texture"):
        if texture.get("name") != _COURT_TEXTURE:
            continue
        texture.set("rgb1", ground_rgb)
        texture.set("rgb2", ground_rgb)
        texture.attrib.pop("mark", None)
        texture.attrib.pop("markrgb", None)
    for material in root.findall("asset/material"):
        if material.get("name") != _COURT_MATERIAL:
            continue
        material.attrib.pop("texture", None)
        material.attrib.pop("texrepeat", None)


def _is_court_line_geom(element: ET.Element) -> bool:
    """判断 worldbody 下的 geom 是否为球场白线（视觉 plane，贴地、无材质）。"""
    if element.tag != "geom" or element.get("type") != "plane":
        return False
    if element.get("material"):
        return False
    size = element.get("size", "").split()
    if len(size) != 3 or size[0] == "25":
        return False
    try:
        z = float(element.get("pos", "0 0 0").split()[2])
    except (IndexError, ValueError):
        return False
    return (
        abs(z - COURT_LINE_Z_M) <= 1e-12
        and element.get("rgba", "1 1 1 1") == "1 1 1 1"
        and element.get("contype", "0") == "0"
        and element.get("conaffinity", "0") == "0"
    )


def _apply_court_profile(root: ET.Element, parameters: RobotRuntimeParameters) -> None:
    """按 ``scene.court_profile`` 在模板副本上重建白线（磁盘模板不变）。

    模板磁盘字节固定以 legacy 10 条白线生成；若运行时要求 ITF 标准
    12 条，则在同一位置替换白线 geom（全部 contype=conaffinity=0，
    纯视觉）。legacy 目标且模板已是旧 10 条时直接返回，保持零改动。

    Raises:
        ValueError: 模板缺少 worldbody 或白线替换后仍不符合目标 profile。
    """
    profile = _court_profile_of(parameters)
    if profile == "itf_2026_v1":
        # 标准 profile 同时要求中性地面（旧棋盘纹理的浅色边缘会被误读为线）。
        _neutralize_court_ground(root)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("R1D 模板缺少 <worldbody>，无法按 court_profile 装配白线")
    victims = [el for el in list(worldbody) if _is_court_line_geom(el)]
    already_legacy = bool(victims) and all(
        el.get("name") is None for el in victims
    ) and len(victims) == 10
    already_itf = bool(victims) and all(
        el.get("name") is not None for el in victims
    ) and len(victims) == 12
    if (profile == "legacy_v1" and already_legacy) or (
        profile == "itf_2026_v1" and already_itf
    ):
        return
    anchor = list(worldbody).index(victims[0]) if victims else len(worldbody)
    for element in victims:
        worldbody.remove(element)
    for offset, line in enumerate(build_court_lines(profile)):
        attrs = line.geom_attributes()
        if profile == "legacy_v1":
            attrs = {
                "type": "plane",
                "size": attrs["size"],
                "rgba": attrs["rgba"],
                "contype": "0",
                "conaffinity": "0",
                "pos": attrs["pos"],
            }
        element = ET.Element("geom", attrs)
        element.tail = "\n    "
        worldbody.insert(anchor + offset, element)


def _require_profile(
    profile: RobotProfile | None, poses: Any
) -> RobotProfile:
    """返回重烘固定姿态所需的机器人描述（缺失时明确报错）。

    Raises:
        ValueError: 未提供 profile，无法推导场景侧别。
    """
    if profile is None:
        raise ValueError(
            f"固定关节姿态 {sorted(poses)} 需要机器人描述来推导对侧臂侧别，"
            "但当前未提供 profile"
        )
    return profile


def _check_scene_landing(root: ET.Element, scene: Any) -> None:
    """校验 R1D 底座落在轮/脚轮上（防支撑柱移除后悬浮或穿地）。

    只有模板含轮/脚轮视觉时才检查；合成夹具（无落地视觉）跳过。

    Raises:
        ValueError: 底座高度与轮/脚轮推导的落地高度不一致。
    """
    base = _find_body_optional(root, "base_link")
    if base is None:
        # 合成夹具/外部模板没有本仓库环境链根 body：跳过落地校验
        return
    landing_geoms = [
        geom
        for geom in base.iter("geom")
        if geom.attrib.get("name", "").startswith(("r1d_wheel_", "r1d_caster_"))
    ]
    if not landing_geoms:
        return
    actual = float(scene.base_position[2])
    if abs(actual - _BASE_LANDING_HEIGHT_M) > 1e-6:
        raise ValueError(
            f"scene.base_position z={actual} 与落地高度 "
            f"{_BASE_LANDING_HEIGHT_M} 不一致（R1D 底座必须落在轮/脚轮上，"
            f"不允许悬浮或穿地；落地高度由轮/脚轮几何推导）"
        )


def _rebase_lift(root: ET.Element, lift_displacement: float) -> None:
    """把模板中已 bake 的升降高度改写为配置的显式升降位移。

    模板 r1d_env.xml 以 ``_ENV_BAKED_LIFT_M`` 烘焙两级 prismatic 位移；
    装配时按差值在内存中重烘，磁盘模板保持不变。

    Raises:
        ValueError: 模板 bake 高度与 ``_ENV_BAKED_LIFT_M`` 不一致（模板漂移）。
    """
    base_offsets = {"lift_stage1_link": 0.0427, "lift_stage2_link": -0.2115}
    bodies: list[tuple[ET.Element, float]] = []
    for body_name, base_z in base_offsets.items():
        body = _find_body_optional(root, body_name)
        if body is None:
            # 合成夹具/外部模板没有本仓库升降链：无 bake 可重烘，跳过
            return
        bodies.append((body, base_z))
    for body, base_z in bodies:
        pos = body.attrib.get("pos")
        if pos is None:
            raise ValueError(
                f"模板 body {body.attrib.get('name')} 缺少 pos（模板漂移）"
            )
        current_z = float(pos.split()[2])
        if abs(current_z - (base_z + _ENV_BAKED_LIFT_M)) > 1e-6:
            raise ValueError(
                f"模板 body {body.attrib.get('name')} 的 z={current_z} 与预期 "
                f"bake 高度 {base_z + _ENV_BAKED_LIFT_M} 不一致（模板漂移："
                f"r1d_env.xml 未按 {_ENV_BAKED_LIFT_M} 烘焙）"
            )
        body.set(
            "pos", _fmt_vec((float(pos.split()[0]), float(pos.split()[1]),
                             base_z + lift_displacement))
        )


def _apply_scene_parameters(
    root: ET.Element,
    parameters: RobotRuntimeParameters,
    *,
    side: str | None = None,
    profile: RobotProfile | None = None,
    racket_profile: RacketProfile | None = None,
    template_mode: bool = False,
    fragments: list[Path] | None = None,
) -> RacketAssembly | None:
    """把运行参数写入模板元素（option/mount/lift/固定姿态/球拍/torso）。

    Args:
        root: 模板 XML 根元素（原地修改）。
        parameters: 运行参数。
        side: 主动臂侧别；None 时按固定姿态需要从描述推导。
        profile: 机器人描述。
        racket_profile: 可选球拍档案。
        template_mode: 模板生成模式（跳过升降重烘）。
        fragments: 可选输出列表；记录实际读取的对侧臂片段（F08 闭包）。

    球拍挂接点 = r1d_flange（从树内读取，该侧值已随片段固化）：
    - 档案模式：body pos = flange + mount.position、body quat = mount.wxyz，
      几何/质量/惯量/连接件全部来自 RacketProfile；
    - 旧 API 模式（无档案）：保留 T0 圆盘拍语义（scene.racket_* 参数化），
      未提供旧字段时保持模板默认值（V12 默认配置路径）。
    升降位移在内存中重烘（磁盘模板不变）；模板生成时保留 r1d_env.xml 的
    默认 bake。侧别仅在重烘对侧固定姿态时需要，未提供时可从描述推导。

    Raises:
        ValueError: scene 缺失、升降/落地/固定姿态非法或模板元素漂移。
    """
    _require_scene(parameters, "_apply_scene_parameters")
    scene = parameters.scene
    assert scene is not None

    option = root.find("option")
    if option is None:
        raise ValueError("模板缺少 <option> 元素（模板漂移）")
    option.set("timestep", _fmt_float(parameters.dt))

    mount = _find_body(root, _MOUNT_BODY)
    mount.set("pos", _fmt_vec(scene.base_position))
    mount.set("quat", _fmt_vec(scene.base_quaternion))

    flange_pos = _flange_position(root)
    torso = _find_body(root, _TORSO_BODY)
    torso_geom = torso.find("geom")
    if torso_geom is None:
        raise ValueError("模板机身 body 缺少 geom（模板漂移）")
    torso.set("pos", _fmt_vec(scene.torso_position))
    torso_geom.set("size", _fmt_vec(scene.torso_half_size))

    # 落地高度与升降位移由同一物理配置推导；模板生成时跳过升降重烘
    # （模板恒定 bake 默认 q_L，运行时按配置在内存中重烘）。
    _check_scene_landing(root, scene)
    if not template_mode:
        _rebase_lift(root, float(scene.lift_displacement))
    if scene.fixed_joint_poses:
        resolved_side = side if side is not None else _model_side(
            _require_profile(profile, scene.fixed_joint_poses)
        )
        # 延迟导入：固定姿态重烘与臂片段定位在 loading 层，而 loading 依赖
        # 本模块的场景/校验入口；调用期导入避免 import 期循环。
        from src.robot.model_building.loading import _apply_fixed_joint_poses

        _apply_fixed_joint_poses(
            root, parameters, resolved_side, profile, fragments=fragments
        )

    if racket_profile is not None:
        _require_no_legacy_racket(scene)
        _apply_racket_profile(root, racket_profile, flange_pos)
        return _racket_assembly(racket_profile)

    if template_mode or _has_legacy_racket(scene):
        _apply_legacy_racket(root, parameters, flange_pos)
    return None


def _populate_joint_limits(
    parameters: RobotRuntimeParameters,
    model: mujoco.MjModel,
    binding: ModelBinding,
) -> RobotRuntimeParameters:
    """从 model.jnt_range 读取主动关节位置限位，返回填充后的新参数对象。

    模型是位置限位的唯一权威：入参 ``q_lower``/``q_upper`` 为 None 时直接
    填充；若调用方**显式提供**，则必须与模型 jnt_range 一致——不一致立即
    拒绝。否则"调用方收紧的安全限位"会在校验通过后被静默放宽回 URDF
    全范围（校验先用调用方限值检查 init_q，再在这里覆盖）。

    不重复抄 URDF 数字；输入参数对象不被改写。

    Raises:
        ValueError: 任一主动关节在模型中未声明限位（jnt_limited=0），或
            显式提供的 q_lower/q_upper 与模型 jnt_range 不一致。
    """
    joint_ids = np.asarray(binding.joint_ids, dtype=int)
    if not bool(np.all(model.jnt_limited[joint_ids] == 1)):
        unlimited = [
            name
            for name, jid in zip(parameters.joint_names, binding.joint_ids)
            if int(model.jnt_limited[jid]) != 1
        ]
        raise ValueError(
            f"模型中存在未声明限位的主动关节 {unlimited}，无法填充"
            f"q_lower/q_upper（robot_id: {parameters.robot_id}，"
            f"来源: {parameters.source}）"
        )
    ranges = model.jnt_range[joint_ids]
    lower = np.array(ranges[:, 0], dtype=np.float64)
    upper = np.array(ranges[:, 1], dtype=np.float64)
    # 显式入参必须与模型一致：不一致即拒绝，不做静默取交集/覆盖
    for field, expected in (("q_lower", lower), ("q_upper", upper)):
        provided = getattr(parameters, field)
        if provided is None:
            continue
        provided_arr = np.asarray(provided, dtype=np.float64)
        if provided_arr.shape != expected.shape:
            raise ValueError(
                f"显式提供的 {field} 维数 {provided_arr.shape} 与模型关节数 "
                f"{expected.shape} 不符（robot_id: {parameters.robot_id}，"
                f"来源: {parameters.source}）"
            )
        mismatch = ~np.isclose(provided_arr, expected, rtol=0.0, atol=1e-12)
        if bool(np.any(mismatch)):
            names = [
                name
                for name, bad in zip(parameters.joint_names, mismatch.tolist())
                if bad
            ]
            raise ValueError(
                f"显式提供的 {field} 与模型 jnt_range 不一致（模型为位置限位的"
                f"唯一权威，不一致即拒绝而非放宽；关节 {names}，"
                f"robot_id: {parameters.robot_id}，来源: {parameters.source}）"
            )
    return replace(
        parameters,
        q_lower=lower,
        q_upper=upper,
    )


def _validate_model_shape(model: mujoco.MjModel, profile: RobotProfile) -> None:
    """校验模型维数与描述一致，且全部关节为铰链（无 freejoint 等）。

    Raises:
        ValueError: 维数不符或存在非铰链关节。
    """
    n = profile.ndof
    if model.nq != n or model.nv != n or model.nu != n or model.njnt != n:
        raise ValueError(
            f"模型维数与描述不符: nq={model.nq}, nv={model.nv}, nu={model.nu},"
            f" njnt={model.njnt}，期望均为 {n}"
            f"（robot_id: {profile.robot_id}，模型: {profile.model_path}）"
        )
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            raise ValueError(
                f"模型存在非铰链关节 '{name}'（R1D 场景不应有 freejoint/球）"
                f"（robot_id: {profile.robot_id}，模型: {profile.model_path}）"
            )


def _check_initial_state(
    model: mujoco.MjModel,
    binding: ModelBinding,
    parameters: RobotRuntimeParameters,
) -> None:
    """校验 init_q 位于模型 jnt_range 界内，且 mj_forward 初态有限。

    Raises:
        ValueError: 初始关节角超限或前向计算产生非有限状态。
    """
    joint_ids = np.asarray(binding.joint_ids, dtype=int)
    lower = model.jnt_range[joint_ids, 0]
    upper = model.jnt_range[joint_ids, 1]
    out_of_range = (parameters.init_q < lower) | (parameters.init_q > upper)
    if bool(np.any(out_of_range)):
        indices = np.where(out_of_range)[0].tolist()
        raise ValueError(
            f"字段 init_q 初始关节角超出模型 jnt_range（超限关节索引 {indices}，"
            f"robot_id: {parameters.robot_id}，来源: {parameters.source}）"
        )
    data = mujoco.MjData(model)
    data.qpos[np.asarray(binding.qpos_indices, dtype=int)] = parameters.init_q
    mujoco.mj_forward(model, data)
    if not (
        bool(np.all(np.isfinite(data.qpos))) and bool(np.all(np.isfinite(data.qvel)))
    ):
        raise ValueError(
            f"初态 mj_forward 产生非有限状态"
            f"（robot_id: {parameters.robot_id}，来源: {parameters.source}）"
        )
