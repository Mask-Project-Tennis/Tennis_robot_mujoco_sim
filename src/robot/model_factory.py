"""R1D 场景模型工厂 — 裸臂/环境/资产三片段 + 场景参数合成为可加载模型。

职责（阶段 2 共享约束 N1/Task 2.5b）：
- ``write_r1d_model_template``：用 ElementTree 合并 Task 2.5a 的三份
  ``<mujocoinclude>`` 片段（r1d_arm_{side} 裸臂 + r1d_env 冻结环境链 +
  r1d_assets 资产），生成完整可加载的 r1d_model_{side}.xml：
  ground + ``r1d_mount``（固定基座，pos/quat 来自 scene.base_*）→
  环境链（含已 bake 的 q_L，0 关节）→ 选定侧裸臂（r1d_base 锚在
  waist_yaw_link）→ ``r1d_flange`` 上挂球拍（racket body/geom/site=
  ``racket_center``，单一实心圆盘显式惯量）+ 简化 torso box +
  7 个 unit-gear motor（名称与 profile.actuator_names 一一对应）。
- ``load_robot_model``：读取磁盘模板 → 仅在副本上应用本次参数（经
  ``load_mujoco_model`` 的 xml_text 通道，不改磁盘模板）→ 绑定名称 →
  校验维数与初态 → 从 ``model.jnt_range`` 填充 q_lower/q_upper →
  返回 RobotModelBundle。组装管线提取为私有共享函数
  ``_assemble_bound_model``（阶段 3 Task 3.2 的 build_tennis_scene 复用，
  不改变本函数的模型形状与验收契约）。

约定：
- 支持集 robot_id = ``unitree_r1d``（无 unitree_d2 分支；原 D2 场景组装
  已被否决）；RM65 继续旧模型入口与兼容适配器。
- 生产模板遵守 AGENTS.md 自碰撞关闭约定（全部 geom contype=0
  conaffinity=0）；碰撞诊断在独立诊断模型（掩码打开的加载实例）中进行，
  不修改生产模板，也不在本模块内接入任何仿真/真机入口。
- motor 仅供名称绑定（gear=1）：R1D URDF 的 effort/velocity 声明值
  未经实机验证（N0-9），不配置 ctrlrange/forcerange 为安全限值。
- 官方臂/环境层级与惯量来自 2.5a 片段逐字保留（含右腕俯仰上界
  +0.5326 的官方不对称）；临时球拍/torso 属性只写入本工厂创建的元素，
  绝不回写官方臂惯量。r1d_flange 位置从臂片段按名读取，不重复抄数值。
- 升降高度只由 r1d_env 的 q_L bake 表达，本工厂不提供第二套升降偏移。
- mesh 路径先按片段父目录解析为绝对路径，再重写为相对 output_path.parent
  的路径；跨盘符等无法相对化时把 mesh 文件重定位复制到导出目录
  meshes/ 子目录并改写为相对引用（绝对路径会被 MuJoCo C 层拒绝打开）。
"""

from __future__ import annotations

import copy
import logging
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

import mujoco
import numpy as np

from src.robot.model_binding import ModelBinding, bind_robot_model
from src.robot.robot_profile import RobotProfile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    validate_runtime_parameters,
)
from src.utils.mujoco_loader import load_mujoco_model

logger = logging.getLogger(__name__)

# 本工厂支持场景组装的 robot_id；其他 robot_id 明确报不支持（RM65 走既有入口）
SUPPORTED_ROBOT_ID = "unitree_r1d"

# 合法的侧别（无隐式默认侧，side 必须显式传入）
SIDES: tuple[str, ...] = ("left", "right")

# 2.5a 片段的结构契约名称（工厂按名定位，缺失即视为模板漂移）
_ARM_ROOT_BODY = "r1d_base"
_ENV_ROOT_BODY = "base_link"
_ENV_TIP_BODY = "waist_yaw_link"
_FLANGE_SITE = "r1d_flange"

# 模板中的固定元素名称（工厂按名注入参数，缺失即视为模板漂移）
_MOUNT_BODY = "r1d_mount"
_RACKET_BODY = "r1d_racket"
_RACKET_GEOM = "racket_face"
_TORSO_BODY = "r1d_torso"
_TORSO_GEOM = "torso_box"
_TCP_SITE = "racket_center"
_GROUND_GEOM = "ground"

# 冻结环境链的 body 序列（相邻对排除表静态部分，顺序=父子链）
_ENV_CHAIN_BODIES: tuple[str, ...] = (
    "base_link",
    "lift_mount_link",
    "lift_stage1_link",
    "lift_stage2_link",
    "pelvis_link",
    "waist_yaw_link",
)

# 跨盘符导出时 mesh 资源的重定位子目录（相对 output_path.parent）
_EXPORT_MESH_DIRNAME = "meshes"


def _side_chain_bodies(side: str) -> tuple[str, ...]:
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


def _connected_excludes(side: str) -> tuple[tuple[str, str], ...]:
    """相邻/连接部件对（按侧生成）。

    生产模型自碰撞关闭（掩码全 0，此表不生效）；独立诊断模型打开
    接触掩码后仅排除这些对，其余组合出现穿透即判失败。包含：
    挂载-环境链根、环境链 5 个相邻对、躯干顶-虚拟安装点-肩部、
    臂链 6 个相邻对、末端连杆-球拍、挂载-简化 torso。

    另含三组**结构紧凑单元**对（官方 URDF 视觉网格=碰撞网格的固有
    交叠，与关节姿态无关，调准备姿态无法消除，实测记录见
    task-2.5b 报告）：
    - 躯干顶-肩俯仰/肩侧摆连杆：肩关节经无质量虚拟安装点 r1d_base
      直接锚在 waist_yaw_link 上（安装面交叠约 -0.0054~-0.0093 m）；
    - 腕滚转-腕偏航连杆（隔级）：腕三连杆为同一紧凑腕单元，官方
      网格相互包裹（交叠约 -0.0325 m，双侧同值）。
    """
    chain = _side_chain_bodies(side)
    arm_pairs = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    return (
        (_MOUNT_BODY, _ENV_ROOT_BODY),
        *((a, b) for a, b in zip(_ENV_CHAIN_BODIES, _ENV_CHAIN_BODIES[1:])),
        (_ENV_TIP_BODY, _ARM_ROOT_BODY),
        (_ARM_ROOT_BODY, chain[0]),
        # 肩部安装：躯干顶 ↔ 肩俯仰/肩侧摆（经虚拟安装点连接的结构对）
        (_ENV_TIP_BODY, chain[0]),
        (_ENV_TIP_BODY, chain[1]),
        *arm_pairs,
        # 腕紧凑单元隔级对：腕滚转 ↔ 腕偏航（官方网格固有交叠）
        (chain[4], chain[6]),
        (chain[-1], _RACKET_BODY),
        (_MOUNT_BODY, _TORSO_BODY),
    )


@dataclass(frozen=True)
class RobotModelBundle:
    """同源组装结果 — 描述、参数、模型与绑定出自同一次配置。

    Attributes:
        profile: 机器人描述（身份与语义名称）。
        parameters: 运行参数；q_lower/q_upper 已从模型 jnt_range 填充。
        model: 按 parameters 组装并加载的 MuJoCo 模型（每次调用独立实例）。
        binding: model 的名称 → 地址绑定结果。
    """

    profile: RobotProfile
    parameters: RobotRuntimeParameters
    model: mujoco.MjModel
    binding: ModelBinding


def write_r1d_model_template(
    parameters: RobotRuntimeParameters,
    output_path: Path,
    *,
    side: Literal["left", "right"],
    arm_path: Path,
    env_path: Path,
    assets_path: Path,
) -> None:
    """合并 2.5a 三份片段生成完整 R1D 单臂场景模板（字节确定输出）。

    mesh 引用重写为相对 output_path.parent 的路径；相对化失败（跨盘符）
    时把全部 mesh 文件重定位复制到 output_path.parent/meshes/，导出的
    模板与其 mesh 依赖自包含、可被 MuJoCo 直接加载。

    Args:
        parameters: 运行参数；scene 提供 r1d_mount/球拍/torso 的默认
            数值真值（与 r1d_runtime_{side}.yaml 一致），dt 写入
            <option timestep>。
        output_path: 模板输出路径（如 src/robot/r1d_model_left.xml）。
        side: 侧别（"left"/"right"），显式表达，无隐式默认侧。
        arm_path: 该侧裸臂片段路径（src/robot/r1d_arm_{side}.xml，只读）。
        env_path: 冻结环境片段路径（src/robot/r1d_env.xml，只读；
            升降高度只由其中的 q_L bake 表达）。
        assets_path: 资产片段路径（src/robot/r1d_assets.xml，只读）。

    Raises:
        ValueError: robot_id 不支持、side 非法、scene 缺失或片段结构
            不符合契约。
        FileNotFoundError: 片段引用的 mesh 文件在磁盘上不存在。
    """
    _require_r1d(parameters, "write_r1d_model_template")
    _require_side(side, "write_r1d_model_template")
    output_path = Path(output_path)
    root = _build_template_element(
        parameters,
        side=side,
        arm_path=Path(arm_path),
        env_path=Path(env_path),
        assets_path=Path(assets_path),
    )
    _retarget_mesh_paths(root, assets_path=Path(assets_path), output_path=output_path)
    ET.indent(root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    logger.info("R1D %s 侧场景模板已生成: %s", side, output_path)


def load_robot_model(
    profile: RobotProfile, parameters: RobotRuntimeParameters
) -> RobotModelBundle:
    """组装并加载 R1D 单臂场景模型，返回同源的 RobotModelBundle。

    流程：校验身份/维数/初态（复用 2.2 的 kinematics 级校验）→ 读取磁盘
    模板 → 仅在模板副本（xml_text）上应用本次参数 → 经 load_mujoco_model
    加载 → bind_robot_model 绑定 → 校验模型维数与初态在限位内 → 从
    model.jnt_range 填充 q_lower/q_upper（不重复抄 URDF 数字）。

    Args:
        profile: 机器人描述（当前仅支持 robot_id="unitree_r1d"；
            model_path 指向对应侧模板，侧别由此显式表达）。
        parameters: 运行参数（须含 scene；q_lower/q_upper 通常为 None，
            由模型 jnt_range 回填；若显式提供则必须与模型一致，不一致
            即拒绝，不接受"收紧/放宽"覆盖）。

    Returns:
        同源组装结果；parameters 为填充关节限位后的新对象（输入不被改写）。

    Raises:
        ValueError: robot_id 不支持、scene 缺失、校验失败、模板结构漂移、
            模型维数与描述不符、初态超出模型限位，或显式 q_lower/q_upper
            与模型 jnt_range 不一致。
    """
    if profile.robot_id != SUPPORTED_ROBOT_ID:
        raise ValueError(
            f"load_robot_model 当前仅支持 {SUPPORTED_ROBOT_ID!r} 的场景组装，"
            f"收到 robot_id={profile.robot_id!r}；RM65 等继续使用既有模型入口"
            f"与兼容适配器"
        )
    _require_scene(parameters, "load_robot_model")
    # 身份/关节序/维数/初态（kinematics 级）统一走 2.2 校验
    validate_runtime_parameters(parameters, profile, purpose="kinematics")

    bundle = _assemble_bound_model(
        profile, parameters, validate_shape=_validate_model_shape
    )
    logger.info(
        "R1D 模型组装完成: nq=%d nv=%d nu=%d TCP site=%s（参数来源: %s）",
        bundle.model.nq,
        bundle.model.nv,
        bundle.model.nu,
        profile.tcp_site_name,
        bundle.parameters.source,
    )
    return bundle


def _assemble_bound_model(
    profile: RobotProfile,
    parameters: RobotRuntimeParameters,
    *,
    validate_shape: Callable[[mujoco.MjModel, RobotProfile], None],
    mutate: Callable[[ET.Element], None] | None = None,
) -> RobotModelBundle:
    """共享组装管线（load_robot_model 与 3.2 带球场景 build_tennis_scene 共用）。

    流程：读取磁盘模板 → 仅在模板副本（xml_text）上应用本次参数 →
    可选 ``mutate`` 追加组装（如 3.2 的球/掩码）→ 经 load_mujoco_model
    编译 → ``validate_shape`` 校验模型形状（裸模型与带球模型维数契约
    不同，由调用方注入）→ bind_robot_model 按名绑定 → 校验初态 → 从
    model.jnt_range 填充 q_lower/q_upper 并复核。磁盘模板字节不变。

    Args:
        profile: 机器人描述（model_path 指向模板）。
        parameters: 运行参数（须含 scene；仅做 kinematics 级校验，
            完整仿真预检由 3.1 validate_simulation_inputs 负责）。
        validate_shape: 模型形状校验回调（裸/带球维数契约不同）。
        mutate: 模板副本的追加组装回调（元素级修改，须保持可编译）。

    Returns:
        同源组装结果；parameters 为填充关节限位后的新对象（输入不改写）。

    Raises:
        ValueError: scene 缺失、模板结构漂移、形状校验失败、名称绑定
            失败、初态超限或显式限位与模型不一致。
    """
    _require_scene(parameters, "_assemble_bound_model")

    # 仅在模板副本上应用参数：磁盘模板只读，数值经 xml_text 编译
    template_root = ET.fromstring(profile.model_path.read_bytes())
    _apply_scene_parameters(template_root, parameters)
    if mutate is not None:
        mutate(template_root)
    xml_text = ET.tostring(template_root, encoding="unicode")
    model = load_mujoco_model(profile.model_path, xml_text=xml_text)

    validate_shape(model, profile)
    binding = bind_robot_model(profile, model)
    _check_initial_state(model, binding, parameters)

    filled = _populate_joint_limits(parameters, model, binding)
    # 填充限位后复核：init_q 必须落在模型 jnt_range 界内
    validate_runtime_parameters(filled, profile, purpose="kinematics")
    return RobotModelBundle(
        profile=profile, parameters=filled, model=model, binding=binding
    )


# ---------------------------------------------------------------- 组装辅助


def _require_r1d(parameters: RobotRuntimeParameters, entry: str) -> None:
    """校验参数身份属于 unitree_r1d（模板生成/加载共用）。"""
    if parameters.robot_id != SUPPORTED_ROBOT_ID:
        raise ValueError(
            f"{entry} 仅支持 robot_id={SUPPORTED_ROBOT_ID!r}，"
            f"收到 {parameters.robot_id!r}（来源: {parameters.source}）"
        )
    _require_scene(parameters, entry)


def _require_scene(parameters: RobotRuntimeParameters, entry: str) -> None:
    """校验 scene 场景参数存在（R1D 场景组装必需）。"""
    if parameters.scene is None:
        raise ValueError(
            f"字段 scene 缺失：{entry} 需要 SceneParameters"
            f"（robot_id: {parameters.robot_id}，来源: {parameters.source}）"
        )


def _require_side(side: str, entry: str) -> None:
    """校验侧别显式且合法（无隐式默认侧）。"""
    if side not in SIDES:
        raise ValueError(
            f"{entry} 的 side 必须是 {list(SIDES)} 之一，收到 {side!r}"
            f"（侧别只经显式文件路径/参数表达，无隐式默认侧）"
        )


def _parse_fragment(path: Path, expected_root: str) -> ET.Element:
    """解析 <mujocoinclude> 片段并校验根标签。

    Raises:
        ValueError: 文件根标签不是 <mujocoinclude>。
    """
    root = ET.parse(path).getroot()
    if root.tag != expected_root:
        raise ValueError(
            f"片段根标签应为 <{expected_root}>，实际 <{root.tag}>（片段: {path}）"
        )
    return root


def _fragment_worldbody(root: ET.Element, path: Path) -> ET.Element:
    """取片段的唯一 worldbody 元素。

    Raises:
        ValueError: 片段缺少 worldbody。
    """
    worldbody = next((el for el in root if el.tag == "worldbody"), None)
    if worldbody is None:
        raise ValueError(f"片段缺少 <worldbody>（片段: {path}）")
    return worldbody


def _read_arm_chain(root_body: ET.Element, side: str, arm_path: Path) -> tuple[str, ...]:
    """沿裸臂根 body 逐级下钻，按序读取该侧 7 个关节名（用于 motor 绑定）。

    Raises:
        ValueError: 臂链层级与该侧 7 关节契约不符。
    """
    joint_names: list[str] = []
    body = root_body
    for expected_body in _side_chain_bodies(side):
        children = [el for el in body if el.tag == "body"]
        if len(children) != 1 or children[0].attrib.get("name") != expected_body:
            raise ValueError(
                f"裸臂片段在 {expected_body} 父级应有唯一下一级 body"
                f"（片段: {arm_path}）"
            )
        body = children[0]
        joint = body.find("joint")
        if joint is None:
            raise ValueError(f"臂链 body {expected_body} 缺少 joint（片段: {arm_path}）")
        joint_names.append(joint.attrib["name"])
    return tuple(joint_names)


def _build_template_element(
    parameters: RobotRuntimeParameters,
    *,
    side: str,
    arm_path: Path,
    env_path: Path,
    assets_path: Path,
) -> ET.Element:
    """从 2.5a 三份片段合成完整 <mujoco> 场景元素树（参数数值随后统一写入）。

    结构：ground/light + ``r1d_mount``（scene.base_* 位姿）→ 环境链
    （base_link → … → waist_yaw_link，0 关节，q_L 已 bake）→ 该侧裸臂
    （r1d_base 锚在 waist_yaw_link）→ 末端连杆上 ``r1d_flange`` site +
    球拍 body；简化 torso 挂 r1d_mount；7 个 unit-gear motor。

    Raises:
        ValueError: 片段缺少 compiler/worldbody/根 body/asset 或结构
            不符契约。
    """
    arm_root = _parse_fragment(arm_path, "mujocoinclude")
    env_root = _parse_fragment(env_path, "mujocoinclude")
    assets_root = _parse_fragment(assets_path, "mujocoinclude")

    compiler = next((el for el in arm_root if el.tag == "compiler"), None)
    if compiler is None:
        raise ValueError(f"裸臂片段缺少 <compiler>（片段: {arm_path}）")
    arm_worldbody = _fragment_worldbody(arm_root, arm_path)
    arm_root_body = next(
        (el for el in arm_worldbody if el.tag == "body"), None
    )
    if arm_root_body is None or arm_root_body.attrib.get("name") != _ARM_ROOT_BODY:
        raise ValueError(
            f"裸臂片段 worldbody 下唯一根 body 应为 {_ARM_ROOT_BODY}"
            f"（片段: {arm_path}）"
        )
    env_worldbody = _fragment_worldbody(env_root, env_path)
    env_root_body = next(
        (el for el in env_worldbody if el.tag == "body"), None
    )
    if env_root_body is None or env_root_body.attrib.get("name") != _ENV_ROOT_BODY:
        raise ValueError(
            f"环境片段 worldbody 下唯一根 body 应为 {_ENV_ROOT_BODY}"
            f"（片段: {env_path}）"
        )
    if list(env_root.iter("joint")):
        raise ValueError(
            f"环境片段必须无任何关节（升降 q_L 应为 bake 固定偏移）"
            f"（片段: {env_path}）"
        )
    asset_src = next((el for el in assets_root if el.tag == "asset"), None)
    if asset_src is None or not asset_src.findall("mesh"):
        raise ValueError(f"资产片段缺少含 mesh 的 <asset>（片段: {assets_path}）")

    # 关节名按片段层级逐级读取（motor 命名不硬编码，防与片段漂移）
    joint_names = _read_arm_chain(arm_root_body, side, arm_path)

    root = ET.Element("mujoco", {"model": f"r1d_{side}_tennis_scene"})
    root.append(ET.Comment(_template_comment(side)))
    # compiler 仅来自裸臂片段（angle="degree"，限位单位契约），不再添加第二个
    root.append(copy.deepcopy(compiler))
    ET.SubElement(
        root, "option", timestep="0.005", gravity="0 0 -9.81", integrator="implicitfast"
    )
    root.append(copy.deepcopy(asset_src))

    contact = ET.SubElement(root, "contact")
    for body1, body2 in _connected_excludes(side):
        ET.SubElement(contact, "exclude", body1=body1, body2=body2)

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody, "light", pos="0 0 8", dir="0 0 -1", diffuse="1.2 1.25 1.3"
    )
    ET.SubElement(
        worldbody,
        "geom",
        name=_GROUND_GEOM,
        type="plane",
        size="4 4 0.1",
        rgba="0.45 0.5 0.45 1",
        contype="0",
        conaffinity="0",
    )
    mount = ET.SubElement(
        worldbody, "body", name=_MOUNT_BODY, pos="0 0 0", quat="1 0 0 0"
    )
    # 环境链挂在固定基座下（含已 bake 的 q_L；升降高度唯一表达处）
    mount.append(copy.deepcopy(env_root_body))
    # 选定侧裸臂锚在环境链顶部 waist_yaw_link（虚拟固定安装点）
    env_tip = _find_body(mount, _ENV_TIP_BODY)
    arm_copy = copy.deepcopy(arm_root_body)
    env_tip.append(arm_copy)

    # 末端挂接：球拍 body 挂在该侧 wrist_yaw_link（r1d_flange site 之上的
    # 官方工具安装参考；flange 偏移从臂片段按名读取，不重复抄数值）
    arm_tip_name = _side_chain_bodies(side)[-1]
    arm_tip = _find_body(arm_copy, arm_tip_name)
    flange_pos = _flange_position(arm_tip)
    racket = ET.SubElement(
        arm_tip, "body", name=_RACKET_BODY, pos=_fmt_vec(flange_pos), quat="1 0 0 0"
    )
    # 单一几何（视觉即碰撞形状）+ 显式惯量：避免视觉/碰撞双重计质量
    ET.SubElement(
        racket, "inertial", pos="0 0 0", mass="0.3", diaginertia="0 0 0"
    )
    ET.SubElement(
        racket,
        "geom",
        name=_RACKET_GEOM,
        type="cylinder",
        size="0 0",
        rgba="0.9 0.9 0.25 1",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(racket, "site", name=_TCP_SITE, pos="0 0 0", size="0.005")

    # 简化机身 box：挂在固定基座（r1d_mount）下，相对基座定位
    torso = ET.SubElement(mount, "body", name=_TORSO_BODY, pos="0 0 0")
    ET.SubElement(
        torso,
        "geom",
        name=_TORSO_GEOM,
        type="box",
        size="0 0 0",
        rgba="0.55 0.57 0.6 1",
        contype="0",
        conaffinity="0",
    )

    actuator = ET.SubElement(root, "actuator")
    for joint_name in joint_names:
        ET.SubElement(
            actuator,
            "motor",
            name=f"torque_{joint_name}",
            joint=joint_name,
            gear="1",
        )

    # 参数数值真值统一写入（生成与加载共用同一入口，防两套代码漂移）
    _apply_scene_parameters(root, parameters)
    return root


def _template_comment(side: str) -> str:
    """模板文件头注释（说明来源与重建方式，随模板交付）。"""
    return (
        "自动生成文件，勿手改。重建：调用 src/robot/model_factory.py"
        f"::write_r1d_model_template（输入：configs/robots/r1d_runtime_{side}.yaml + "
        f"src/robot/r1d_arm_{side}.xml + src/robot/r1d_env.xml + "
        "src/robot/r1d_assets.xml）。官方臂/环境层级与惯量来自 Task 2.5a "
        "片段逐字保留（含右腕俯仰上界 +0.5326 的官方不对称）；升降高度只由 "
        "r1d_env.xml 的 q_L bake 表达（当前 q_L=0）；场景/球拍/torso 数值"
        f"真值为 r1d_runtime_{side}.yaml。"
    )


def _flange_position(parent: ET.Element) -> np.ndarray:
    """从元素树中读取 r1d_flange site 的 pos（该侧 dex1_base_joint origin）。

    Raises:
        ValueError: 树中不存在 r1d_flange site 或缺少 pos 属性。
    """
    for site in parent.iter("site"):
        if site.attrib.get("name") == _FLANGE_SITE:
            pos = site.attrib.get("pos")
            if pos is None:
                raise ValueError(f"site {_FLANGE_SITE} 缺少 pos 属性（模板漂移）")
            return np.array([float(token) for token in pos.split()], dtype=np.float64)
    raise ValueError(f"模板中缺少 site {_FLANGE_SITE}（模板漂移）")


def _apply_scene_parameters(
    root: ET.Element, parameters: RobotRuntimeParameters
) -> None:
    """把运行参数写入模板元素（option/mount/racket/torso 属性）。

    球拍挂接点 = r1d_flange（从树内读取，该侧值已随片段固化）+ 拍面中心
    偏移（racket_position 语义为相对 r1d_flange）。

    Raises:
        ValueError: scene 缺失或模板缺少期望的元素（模板漂移）。
    """
    _require_scene(parameters, "_apply_scene_parameters")
    scene = parameters.scene

    option = root.find("option")
    if option is None:
        raise ValueError("模板缺少 <option> 元素（模板漂移）")
    option.set("timestep", _fmt_float(parameters.dt))

    mount = _find_body(root, _MOUNT_BODY)
    mount.set("pos", _fmt_vec(scene.base_position))
    mount.set("quat", _fmt_vec(scene.base_quaternion))

    # 球拍挂接点 = 法兰安装参考 + 拍面中心偏移（均在末端连杆参考系）
    racket = _find_body(root, _RACKET_BODY)
    racket_pos = _flange_position(root) + scene.racket_position
    racket.set("pos", _fmt_vec(racket_pos))
    racket.set("quat", _fmt_vec(scene.racket_quaternion))
    inertial = racket.find("inertial")
    racket_geom = racket.find("geom")
    if inertial is None or racket_geom is None:
        raise ValueError("模板球拍 body 缺少 inertial/geom（模板漂移）")
    inertial.set("mass", _fmt_float(scene.racket_mass))
    inertial.set(
        "diaginertia",
        _fmt_vec(
            _disk_inertia(scene.racket_mass, scene.racket_radius, scene.racket_half_thickness)
        ),
    )
    racket_geom.set(
        "size", _fmt_vec((scene.racket_radius, scene.racket_half_thickness))
    )

    torso = _find_body(root, _TORSO_BODY)
    torso_geom = torso.find("geom")
    if torso_geom is None:
        raise ValueError("模板机身 body 缺少 geom（模板漂移）")
    torso.set("pos", _fmt_vec(scene.torso_position))
    torso_geom.set("size", _fmt_vec(scene.torso_half_size))


def _retarget_mesh_paths(
    root: ET.Element, *, assets_path: Path, output_path: Path
) -> None:
    """把资产片段的 mesh 路径重写为相对 output_path.parent 的路径。

    片段内 file 属性按片段父目录解析为绝对路径；再求相对 output_path.parent
    的路径。相对化失败（Windows 跨盘符等）时**重定位依赖资源**：把 mesh
    文件复制到 output_path.parent/meshes/ 并改写为正斜杠相对引用——
    退回绝对路径会让模板回引仓库目录（本仓库路径含非 ASCII），MuJoCo
    C 层报 Error opening file，导出的模板不可加载。资产片段的 mesh 文件
    全部平铺同名（meshes/*.STL），按 basename 落位无碰撞；复制用
    copyfile 不保留源文件只读属性，保证导出目录可被后续覆盖/清理。

    Raises:
        FileNotFoundError: 片段引用的 mesh 文件不存在。
    """
    mesh_dir = output_path.parent / _EXPORT_MESH_DIRNAME
    for mesh in root.iter("mesh"):
        file_attr = mesh.get("file")
        if file_attr is None:
            raise ValueError(f"资产片段存在无 file 属性的 mesh（片段: {assets_path}）")
        resolved = (assets_path.parent / file_attr).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"片段引用的 mesh 文件不存在: {resolved}")
        try:
            relative = os.path.relpath(resolved, output_path.parent.resolve())
        except ValueError:
            # 跨盘符无法相对化：mesh 重定位复制到导出目录，保持相对引用
            mesh_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolved, mesh_dir / resolved.name)
            relative = f"{_EXPORT_MESH_DIRNAME}/{resolved.name}"
        mesh.set("file", Path(relative).as_posix())


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


# ---------------------------------------------------------------- 数值辅助


def _disk_inertia(
    mass: float, radius: float, half_thickness: float
) -> tuple[float, float, float]:
    """实心圆盘主惯量（近似惯量，不宣称实物球拍）：返回 (Ixx, Iyy, Izz)。

    Izz = 0.5*m*r²；Ixx = Iyy = m*(3r² + (2h)²)/12，其中 2h 为全厚。
    """
    ixx = mass * (3.0 * radius**2 + (2.0 * half_thickness) ** 2) / 12.0
    izz = 0.5 * mass * radius**2
    return (ixx, ixx, izz)


def _fmt_float(value: float) -> str:
    """把浮点数格式化为 12 位有效数字字符串（往返精度高于测试容差）。"""
    return f"{float(value):.12g}"


def _fmt_vec(values: Any) -> str:
    """把数值序列格式化为空格分隔字符串（逐项 12 位有效数字）。"""
    return " ".join(_fmt_float(item) for item in values)


def _find_body(root: ET.Element, name: str) -> ET.Element:
    """在元素树中按名称查找 body，找不到即报模板漂移错误。

    Raises:
        ValueError: 树中不存在该名称的 body。
    """
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    raise ValueError(f"模板中缺少 body '{name}'（模板漂移）")
