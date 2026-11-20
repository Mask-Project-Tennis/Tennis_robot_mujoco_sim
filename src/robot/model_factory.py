"""R1D 场景模型工厂 — 裸臂/环境/资产三片段 + 场景参数合成为可加载模型。

职责（阶段 2 共享约束 N1/Task 2.5b）：
- ``write_r1d_model_template``：用 ElementTree 合并 Task 2.5a 的三份
  ``<mujocoinclude>`` 片段（r1d_arm_{side} 裸臂 + r1d_env 冻结环境链 +
  r1d_assets 资产），生成完整可加载的 r1d_model_{side}.xml：
  ground + ``r1d_mount``（固定基座，pos/quat 来自 scene.base_*）→
  环境链（含已 bake 的 q_L，0 关节）→ 选定侧裸臂（r1d_base 锚在
  waist_yaw_link）→ ``r1d_flange`` 上挂球拍（模板默认 T0 兼容圆盘拍；
  传 ``racket_profile`` 时按档案重建 racket body/geom/site，site=
  ``racket_center``）+ 简化 torso box +
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
  +0.5326 的官方不对称）；球拍物理（几何/质量/惯量/安装/连接件）唯一
  来源是 ``RacketProfile``，本工厂只把装配事实写入模型；旧 API 模式
  （未传档案且 scene 带 racket_* 字段）保留 T0 圆盘拍兼容语义。
  r1d_flange 位置从臂片段按名读取，不重复抄数值。
- 升降位移双轨：r1d_env 片段恒定 bake 默认 q_L=0.4225（模板磁盘字节
  不变）；运行时按 ``scene.lift_displacement`` 在模板副本内存中重烘两级
  lift body 的 z。底座落地高度（0.155 m，由轮/脚轮几何推导）与升降位移
  来自同一物理配置，工厂校验二者不矛盾（不允许悬浮/穿地）。
- 头部两节（head_pitch_Link/head_yaw_Link）与对侧被动臂都是无关节静态
  链，只提供完整外观；头部网格声明随 r1d_env 片段自带 ``<asset>`` 合并
  进模板唯一 ``<asset>`` 节（r1d_assets.xml 不重复声明），避免重复 mesh。
- mesh 路径先按片段父目录解析为绝对路径，再重写为相对 output_path.parent
  的路径；跨盘符等无法相对化时把 mesh 文件重定位复制到导出目录
  meshes/ 子目录并改写为相对引用（绝对路径会被 MuJoCo C 层拒绝打开）。
"""

from __future__ import annotations

import base64
import copy
import hashlib
import logging
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

import mujoco
import numpy as np

from src.robot.court_geometry import (
    COURT_LINE_Z_M,
    COURT_PROFILES,
    build_court_lines,
)
from src.robot.model_binding import ModelBinding, bind_robot_model
from src.robot.racket_profile import RacketProfile
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

# 旧 RM65 场地口径（R6）：地面材质/纹理/尺寸与 10 条白线逐字复制自
# src/robot/rm65_model.xml；不使用用户尝试中的标准 ITF 球场/球网/四灯。
_COURT_TEXTURE = "court_grid"
_COURT_TEXTURE_ATTRS = {
    "name": _COURT_TEXTURE,
    "type": "2d",
    "builtin": "checker",
    "rgb1": "0.28 0.42 0.22",
    "rgb2": "0.24 0.38 0.20",
    "width": "512",
    "height": "512",
    "mark": "edge",
    "markrgb": "0.9 0.95 0.85",
}
_COURT_MATERIAL = "court_floor"
_COURT_MATERIAL_ATTRS = {
    "name": _COURT_MATERIAL,
    "texture": _COURT_TEXTURE,
    "texrepeat": "8 8",
    "reflectance": "0.15",
    "rgba": "0.28 0.42 0.22 1",
}
_COURT_GROUND_SIZE = "25 25 0.1"
_COURT_GROUND_RGBA = "0.28 0.42 0.22 1"
# 球场白线几何唯一定义在 ``src/robot/court_geometry.py``（V01/I2）：
# - legacy_v1：旧 RM65 的 10 条白线（0.10 m 线宽，逐字保留）；
# - itf_2026_v1：ITF 2026 第 1 条的 12 条标准线（0.05 m 线宽）。
# 工厂只按 ``parameters.scene.court_profile`` 选择，不在此重复抄几何数值。
# 旧场地光源（pre-attempt 单灯，R6 要求删除用户尝试新增的四灯）
_LIGHT_ATTRS = {"pos": "0 0 8", "dir": "0 0 -1", "diffuse": "1.2 1.25 1.3"}

# 落地/升降物理契约（T2 A2）：轮半径 0.0625、轮心局部 z=-0.0925，故底座
# 落地高度 = 0.0625 - (-0.0925) = 0.155 m；双级升降单级 bake 位移默认
# 0.4225 m（r1d_env.xml 已按此烘焙），单级行程 [0, 0.425]。
_BASE_LANDING_HEIGHT_M = 0.155
_ENV_BAKED_LIFT_M = 0.4225
_LIFT_STAGE_RANGE_M = (0.0, 0.425)
_LIFT_STAGE_BODIES: tuple[str, ...] = ("lift_stage1_link", "lift_stage2_link")

# 落地视觉：基座真实落地高度 0.155 m（轮/脚轮触地），原先用于把基座
# 托到 z=1.0 的立柱已删除；这里补轮/脚轮的低成本图元，避免底盘悬空 35 mm。
_WHEEL_RADIUS = 0.0625
_WHEEL_HALF_WIDTH = 0.035
_WHEEL_X = -0.037284
_WHEEL_Y = 0.1968
_WHEEL_Z = -0.0925
_CASTER_RADIUS = 0.03245
_CASTER_Z = -0.1225
_CASTER_XY = (
    (0.17272, 0.13024),
    (0.17272, -0.13024),
    (-0.24953, 0.1495),
    (-0.24953, -0.1495),
)

# 球拍视觉：保留 racket_face 作为唯一碰撞几何；另加拍面椭球、拍框、
# 拍喉与拍柄，按成人拍“柄沿 Z、拍面法向 X”的 RM65 拓扑生成。
_RACKET_FACE_VISUAL_GEOM = "racket_face_visual"
_RACKET_HANDLE_GEOM = "racket_handle"
_RACKET_THROAT_GEOMS = ("racket_throat_left", "racket_throat_right")
_RACKET_FRAME_GEOMS = (
    "racket_frame_top",
    "racket_frame_bottom",
    "racket_frame_left",
    "racket_frame_right",
)
_RACKET_HANDLE_RADIUS = 0.014
_RACKET_HANDLE_LENGTH = 0.19
_RACKET_FACE_HALF_LENGTH = 0.165
_RACKET_FACE_HALF_WIDTH = 0.12
_RACKET_FACE_HALF_THICKNESS = 0.011
_RACKET_THROAT_RADIUS = 0.009
_RACKET_FRAME_RADIUS = 0.007
# RacketProfile 装配路径的 geom 命名与配色（面/框/柄标签必须互不相同，
# 供 tennis_scene 按名开启碰撞与测试独立核对）
_RACKET_CONNECTOR_GEOM = "racket_connector"
_RACKET_THROAT_GEOM_FORMAT = "racket_throat_{index}"
_RACKET_FRAME_GEOM_FORMAT = "racket_frame_{index}"
_RACKET_FACE_VISUAL_RGBA = "0.92 0.92 0.84 0.30"
_RACKET_HANDLE_RGBA = "0.16 0.17 0.20 1"
_RACKET_THROAT_RGBA = "0.10 0.11 0.13 1"
_RACKET_FRAME_RGBA = "0.08 0.09 0.11 1"
_RACKET_CONNECTOR_RGBA = "0.30 0.31 0.34 1"
# 被动臂的静态“待机姿态”（度）：不引入关节，只把角度烘进 body quat，
# 避免对侧臂零位时折进躯干、在查看器里几乎看不见。
_PASSIVE_ARM_POSE_DEG = {
    "shoulder_pitch": 10.0,
    "shoulder_roll": 30.0,
    "shoulder_yaw": 0.0,
    "elbow": 50.0,
    "wrist_roll": 0.0,
    "wrist_pitch": -10.0,
    "wrist_yaw": 0.0,
}

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

# ---- 装配依赖闭包（F08）-------------------------------------------------
# 清单版本：产物快照据此区分"记录了闭包"与"旧 artifact 未记录"。
DEPENDENCY_MANIFEST_VERSION = 1

# 依赖清单中的角色（写进快照条目，回放按角色决定物化方式）
DEPENDENCY_ROLE_MODEL = "model"
DEPENDENCY_ROLE_INCLUDE = "include"
DEPENDENCY_ROLE_ARM_FRAGMENT = "arm_fragment"
DEPENDENCY_ROLE_MESH = "mesh"
DEPENDENCY_ROLE_TEXTURE = "texture"

# 需要把内容写进快照的 XML 片段角色（网格/纹理体积大，只记哈希）
DEPENDENCY_XML_ROLES = frozenset(
    {
        DEPENDENCY_ROLE_MODEL,
        DEPENDENCY_ROLE_INCLUDE,
        DEPENDENCY_ROLE_ARM_FRAGMENT,
    }
)

# 资产目录属性：``<compiler meshdir>`` 管 mesh，``<compiler texturedir>`` 管 texture
_ASSET_DIR_ATTRS: dict[str, str] = {
    DEPENDENCY_ROLE_MESH: "meshdir",
    DEPENDENCY_ROLE_TEXTURE: "texturedir",
}

# MJCF 依赖解析用的正则（与 src/sim/episode_artifacts.py 的资产扫描同源：
# 只按真实解析路径收集，不做无边界递归扫描）。用一个"多标签"正则是为了
# 拿到**真正的文档顺序**——compiler 的全局覆盖语义依赖这个顺序。
_MJCF_TAG_PATTERN = re.compile(
    r"<(?P<tag>compiler|include|mesh|texture)\b(?P<attrs>[^>]*?)/?>",
    re.IGNORECASE,
)

# XML 注释（MuJoCo 解析器会整体丢弃；正则扫描前必须先掩码，见 R03）
_XML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)

# include 展开的最大深度（防止环状 include 无界递归）
_MAX_INCLUDE_DEPTH = 8


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


def _passive_side(side: str) -> str:
    """返回对侧名称（用于把对侧臂作为静态视觉挂件）。"""
    return "right" if side == "left" else "left"


def _quat_to_matrix(quat: Any) -> np.ndarray:
    """把 wxyz 四元数转换为旋转矩阵（仅用于视觉几何定位）。

    Args:
        quat: 长度为 4 的 wxyz 四元数。

    Returns:
        3×3 旋转矩阵；输入会先归一化（零四元数退化为单位阵）。
    """
    w, x, y, z = (float(value) for value in quat)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_mul(left: Any, right: Any) -> tuple[float, float, float, float]:
    """按 Hamilton 约定相乘两个 wxyz 四元数。"""
    w1, x1, y1, z1 = (float(value) for value in left)
    w2, x2, y2, z2 = (float(value) for value in right)
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _axis_angle_quat(axis: Any, angle_rad: float) -> tuple[float, float, float, float]:
    """把轴角转换为 wxyz 四元数（轴自动归一化）。"""
    ax, ay, az = (float(value) for value in axis)
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    half = 0.5 * angle_rad
    scale = math.sin(half) / norm
    return (math.cos(half), ax * scale, ay * scale, az * scale)


def _passive_joint_angle(joint_name: str, side: str) -> float:
    """按关节名返回被动臂烘入姿态角（弧度）；右侧 roll 镜像。"""
    for suffix, degrees in _PASSIVE_ARM_POSE_DEG.items():
        if suffix in joint_name:
            if suffix == "shoulder_roll" and side == "right":
                degrees = -degrees
            return math.radians(degrees)
    return 0.0


def _static_arm_copy(
    root_body: ET.Element,
    side: str,
    pose_overrides: dict[str, float] | None = None,
) -> ET.Element:
    """把对侧裸臂复制成无关节静态视觉链。

    仅用于“双臂一起渲染、只控制右臂”：先把待机姿态角烘进各级 body 的
    ``quat``，再删除全部 joint/freejoint 与法兰 site，根 body 改唯一名；
    保留官方 mesh/惯量，不添加 actuator。因无关节，模型的
    njnt/nq/nv/nu 不变，场景形状校验仍按单臂通过。

    Args:
        root_body: 对侧 ``r1d_arm_{side}.xml`` 的根 body 元素。
        side: 该被动臂的侧别（用于唯一命名与诊断）。
        pose_overrides: 可选的固定关节姿态覆盖 ``{关节名: 角度 rad}``；
            键必须出现在该臂片段中（否则报错），未覆盖的关节沿用
            默认待机姿态。

    Returns:
        可直接挂到 ``waist_yaw_link`` 下的静态 body 元素。

    Raises:
        ValueError: 覆盖字典含片段中不存在的关节名。
    """
    overrides = dict(pose_overrides or {})
    fragment_joint_names = {
        joint.attrib.get("name", "")
        for body in root_body.iter("body")
        for joint in body
        if joint.tag == "joint"
    }
    unknown = sorted(set(overrides) - fragment_joint_names)
    if unknown:
        raise ValueError(
            f"固定关节姿态含被动臂片段中不存在的关节 {unknown}"
            f"（side={side}，片段合法关节 {sorted(fragment_joint_names)}）"
        )
    passive = copy.deepcopy(root_body)
    passive.set("name", f"{_ARM_ROOT_BODY}_{side}_passive")
    # 先把每个关节的待机角（或显式固定姿态）烘进 body quat；关节轴在 body 局部系。
    for body in passive.iter("body"):
        joint = next((child for child in body if child.tag == "joint"), None)
        if joint is None:
            continue
        joint_name = joint.attrib.get("name", "")
        axis = joint.attrib.get("axis", "0 0 1").split()
        if joint_name in overrides:
            angle = float(overrides[joint_name])
            range_attr = joint.attrib.get("range")
            if range_attr is not None:
                lower_deg, upper_deg = (
                    float(token) for token in range_attr.split()[:2]
                )
                lower, upper = math.radians(lower_deg), math.radians(upper_deg)
                if angle < lower - 1e-9 or angle > upper + 1e-9:
                    raise ValueError(
                        f"固定关节姿态 {joint_name}={angle} rad 超出关节限位 "
                        f"[{lower}, {upper}] rad（片段角度制，"
                        f"range={range_attr!r}）"
                    )
        else:
            angle = _passive_joint_angle(joint_name, side)
        old_quat = body.attrib.get("quat", "1 0 0 0").split()
        new_quat = _quat_mul(old_quat, _axis_angle_quat(axis, angle))
        body.set("quat", " ".join(f"{value:.12g}" for value in new_quat))
    # 删除所有关节（含 freejoint）；静态视觉链不参与动力学
    for parent in list(passive.iter()):
        for child in list(parent):
            if child.tag in ("joint", "freejoint"):
                parent.remove(child)
    # 对侧法兰 site 与主动臂同名，必须删除以免 MuJoCo 重名
    for parent in list(passive.iter()):
        for child in list(parent):
            if child.tag == "site" and child.attrib.get("name") == _FLANGE_SITE:
                parent.remove(child)
    return passive


def _find_geom(root: ET.Element, name: str) -> ET.Element:
    """按名称查找 geom；缺失即视为模板漂移。"""
    for geom in root.iter("geom"):
        if geom.attrib.get("name") == name:
            return geom
    raise ValueError(f"模板中缺少 geom {name}（模板漂移）")


def _add_landing_visuals(root: ET.Element) -> None:
    """在 base_link 下补轮/脚轮图元（纯视觉，不参与碰撞）。

    基座真实落地高度按轮/脚轮触地取 0.155 m；底盘网格本身最低点只到
    -0.120 m，若没有轮子视觉会看起来仍悬空 35 mm。这里用两个驱动轮
    圆柱 + 四个脚轮球体近似，几何位置逐值取自 URDF 的轮/脚轮 origin。
    """
    base = _find_body(root, "base_link")
    for label, y in (("left", _WHEEL_Y), ("right", -_WHEEL_Y)):
        ET.SubElement(
            base,
            "geom",
            name=f"r1d_wheel_{label}",
            type="cylinder",
            size=f"{_WHEEL_RADIUS} {_WHEEL_HALF_WIDTH}",
            pos=f"{_WHEEL_X} {y} {_WHEEL_Z}",
            quat="0.7071067811865476 0.7071067811865476 0 0",
            rgba="0.08 0.09 0.11 1",
            contype="0",
            conaffinity="0",
        )
    for index, (x, y) in enumerate(_CASTER_XY):
        ET.SubElement(
            base,
            "geom",
            name=f"r1d_caster_{index}",
            type="sphere",
            size=f"{_CASTER_RADIUS}",
            pos=f"{x} {y} {_CASTER_Z}",
            rgba="0.12 0.13 0.16 1",
            contype="0",
            conaffinity="0",
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
class RacketAssembly:
    """球拍装配事实（RacketProfile → 模型 body 后的独立核对记录）。

    由工厂在装配时按档案推导，供 inspect 报告与上层（T4/T5）消费；字段
    全部为只读 float64 数组，不随模型对象生命周期变化。

    Attributes:
        racket_id: 档案标识（如 ``rm65_legacy`` / ``blade100_v9_reference``）。
        source_path: 档案文件路径。
        total_mass: 拍体总质量（kg，= 球拍质量 + 连接件质量，连接件只计一次）。
        com: 拍体质心在拍体局部系中的位置（m）。
        inertia_diagonal: 绕质心的主惯量（kg·m²，升序）。
        face_center: 拍面碰撞中心（拍体局部系，m）。
        face_semi_axes: 拍面碰撞椭球半轴（m，第一项为法向半厚）。
        mount_position: 法兰系中柄底位置（m）。
        mount_wxyz: 拍体局部系在法兰系中的姿态（wxyz 单位四元数）。
        connector_mass: 连接件质量（kg；无连接件为 0）。
    """

    racket_id: str
    source_path: Path
    total_mass: float
    com: np.ndarray
    inertia_diagonal: np.ndarray
    face_center: np.ndarray
    face_semi_axes: np.ndarray
    mount_position: np.ndarray
    mount_wxyz: np.ndarray
    connector_mass: float


@dataclass(frozen=True)
class RobotModelBundle:
    """同源组装结果 — 描述、参数、模型与绑定出自同一次配置。

    Attributes:
        profile: 机器人描述（身份与语义名称）。
        parameters: 运行参数；q_lower/q_upper 已从模型 jnt_range 填充。
        model: 按 parameters 组装并加载的 MuJoCo 模型（每次调用独立实例）。
        binding: model 的名称 → 地址绑定结果。
        racket: 球拍档案装配事实；旧 API（未提供档案）为 None。
        dependency_manifest: 本次装配**实际读取**的文件闭包（模型 XML、
            ``<include>`` 片段、对侧臂片段、mesh/texture）。每条含
            ``role/relative_path/source/sha256/size``，XML 片段另含
            ``content_base64``；由 :func:`collect_model_dependencies` 生成，
            供产物快照完整记录装配依赖（F08）。空元组表示未记录。
    """

    profile: RobotProfile
    parameters: RobotRuntimeParameters
    model: mujoco.MjModel
    binding: ModelBinding
    racket: RacketAssembly | None = None
    dependency_manifest: tuple[dict[str, Any], ...] = ()


def write_r1d_model_template(
    parameters: RobotRuntimeParameters,
    output_path: Path,
    *,
    side: Literal["left", "right"],
    arm_path: Path,
    env_path: Path,
    assets_path: Path,
) -> None:
    """合并 2.5a 三份片段生成 R1D 单侧主动臂 + 对侧静态渲染臂模板。

    对侧臂只保留 mesh/惯量、删除全部关节与法兰 site，因此模板的
    njnt/nq/nv/nu 仍为单臂 7 维；它仅用于查看器里“两条手臂一起渲染”。

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
    profile: RobotProfile,
    parameters: RobotRuntimeParameters,
    *,
    racket_profile: RacketProfile | None = None,
) -> RobotModelBundle:
    """组装并加载 R1D 单侧主动臂 + 对侧静态渲染臂模型。

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
        racket_profile: 可选球拍档案（``RacketProfile``）。提供时球拍几何/
            质量/惯量/安装变换全部来自档案，且 ``parameters.scene`` 不得
            再给 legacy ``racket_*`` 字段；省略（None）时保持旧 API 语义：
            球拍几何由 scene 的 legacy 字段或模板默认值决定。

    Returns:
        同源组装结果；parameters 为填充关节限位后的新对象（输入不被改写）。

    Raises:
        ValueError: robot_id 不支持、scene 缺失、校验失败、模板结构漂移、
            模型维数与描述不符、初态超出模型限位，或显式 q_lower/q_upper
            与模型 jnt_range 不一致；档案模式与 scene legacy 球拍字段同时
            出现也在此拒绝（球拍物理唯一来源是档案）。
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
        profile,
        parameters,
        validate_shape=_validate_model_shape,
        racket_profile=racket_profile,
    )
    logger.info(
        "R1D 模型组装完成: nq=%d nv=%d nu=%d TCP site=%s 球拍=%s（参数来源: %s）",
        bundle.model.nq,
        bundle.model.nv,
        bundle.model.nu,
        profile.tcp_site_name,
        bundle.racket.racket_id if bundle.racket is not None else "legacy",
        bundle.parameters.source,
    )
    return bundle


def _assemble_bound_model(
    profile: RobotProfile,
    parameters: RobotRuntimeParameters,
    *,
    validate_shape: Callable[[mujoco.MjModel, RobotProfile], None],
    mutate: Callable[[ET.Element], None] | None = None,
    racket_profile: RacketProfile | None = None,
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
        racket_profile: 可选球拍档案；None 保持旧 API 的模板/legacy 语义。

    Returns:
        同源组装结果；parameters 为填充关节限位后的新对象（输入不改写）。

    Raises:
        ValueError: scene 缺失、模板结构漂移、形状校验失败、名称绑定
            失败、初态超限或显式限位与模型不一致。
    """
    _require_scene(parameters, "_assemble_bound_model")

    # 仅在模板副本上应用参数：磁盘模板只读，数值经 xml_text 编译
    template_root = ET.fromstring(profile.model_path.read_bytes())
    # 白线口径来自 scene.court_profile（V01/I2）：磁盘模板固定 legacy，
    # 运行时按需在内存副本替换为 ITF 标准 12 条（纯视觉，不参与碰撞）。
    _apply_court_profile(template_root, parameters)
    # 侧别只在需要重烘对侧固定姿态时推导：合成夹具/旧调用不传 fixed poses，
    # 仍须能完成装配（不因关节命名不是 left_/right_ 而拒绝）
    scene = parameters.scene
    side = (
        _model_side(profile)
        if scene is not None and scene.fixed_joint_poses
        else None
    )
    # F08：装配过程中真实读取的兄弟片段（仅 fixed_joint_poses 非空的分支
    # 会读取对侧臂片段），由 _apply_fixed_joint_poses 在读到文件时追加。
    arm_fragments: list[Path] = []
    racket = _apply_scene_parameters(
        template_root,
        parameters,
        side=side,
        profile=profile,
        racket_profile=racket_profile,
        template_mode=False,
        fragments=arm_fragments,
    )
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
    # F08：记录本次装配实际读取的文件闭包（含按真实分支读到的对侧臂片段），
    # 供产物快照完整保存装配依赖；不改变模型/绑定/参数任何既有语义。
    dependency_manifest = collect_model_dependencies(
        profile, filled, arm_fragments=arm_fragments
    )
    return RobotModelBundle(
        profile=profile,
        parameters=filled,
        model=model,
        binding=binding,
        racket=racket,
        dependency_manifest=dependency_manifest,
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
    # 旧 RM65 场地材质/纹理（R6）：地面为 court_grid 棋盘纹理 + court_floor 材质
    ET.SubElement(asset_src, "texture", _COURT_TEXTURE_ATTRS)
    ET.SubElement(asset_src, "material", _COURT_MATERIAL_ATTRS)

    # 环境片段可自带 <asset>（头部网格属于机体，但 r1d_assets.xml 不在 T2
    # 白名单内；因此头部 mesh 声明随 r1d_env.xml 交付，这里合并进模板的唯一
    # <asset> 节。mesh 名必须全局唯一，重复即视为模板漂移）。
    env_asset = next((el for el in env_root if el.tag == "asset"), None)
    env_asset_children: list[ET.Element] = []
    if env_asset is not None:
        existing_mesh_names = {
            mesh.attrib.get("name") for mesh in asset_src.findall("mesh")
        }
        for child in env_asset:
            if child.tag == "mesh":
                name = child.attrib.get("name")
                if name in existing_mesh_names:
                    raise ValueError(
                        f"环境片段自带 mesh '{name}' 与资产片段重名"
                        f"（片段: {env_path}）"
                    )
                existing_mesh_names.add(name)
            env_asset_children.append(child)

    # 关节名按片段层级逐级读取（motor 命名不硬编码，防与片段漂移）
    joint_names = _read_arm_chain(arm_root_body, side, arm_path)

    root = ET.Element("mujoco", {"model": f"r1d_{side}_tennis_scene"})
    root.append(ET.Comment(_template_comment(side)))
    # compiler 仅来自裸臂片段（angle="degree"，限位单位契约），不再添加第二个
    root.append(copy.deepcopy(compiler))
    ET.SubElement(
        root, "option", timestep="0.005", gravity="0 0 -9.81", integrator="implicitfast"
    )
    asset_el = copy.deepcopy(asset_src)
    for child in env_asset_children:
        asset_el.append(copy.deepcopy(child))
    root.append(asset_el)

    contact = ET.SubElement(root, "contact")
    for body1, body2 in _connected_excludes(side):
        ET.SubElement(contact, "exclude", body1=body1, body2=body2)

    worldbody = ET.SubElement(root, "worldbody")
    # 旧场地光源（pre-attempt 单灯，R6 要求删除用户尝试新增的四灯）
    ET.SubElement(worldbody, "light", _LIGHT_ATTRS)
    # 地面：RM65 逐字口径（court_grid 棋盘纹理 + court_floor 材质，25×25 m）
    ET.SubElement(
        worldbody,
        "geom",
        name=_GROUND_GEOM,
        type="plane",
        size=_COURT_GROUND_SIZE,
        rgba=_COURT_GROUND_RGBA,
        material=_COURT_MATERIAL,
        contype="0",
        conaffinity="0",
    )
    # 球场白线：几何来自 court_geometry，profile 由 runtime.scene 决定。
    court_profile = _court_profile_of(parameters)
    if court_profile == "itf_2026_v1":
        _neutralize_court_ground(root)
    for line in build_court_lines(court_profile):
        if court_profile == "legacy_v1":
            # 旧输出没有 name，且属性顺序为 type/size/rgba/contype/
            # conaffinity[/pos]：逐字保持生成模板字节不变。
            attrs = {
                "type": "plane",
                "size": line.geom_attributes()["size"],
                "rgba": line.geom_attributes()["rgba"],
                "contype": "0",
                "conaffinity": "0",
            }
            attrs["pos"] = line.geom_attributes()["pos"]
        else:
            attrs = line.geom_attributes()
        ET.SubElement(worldbody, "geom", attrs)
    mount = ET.SubElement(
        worldbody, "body", name=_MOUNT_BODY, pos="0 0 0", quat="1 0 0 0"
    )
    # 环境链挂在固定基座下（含已 bake 的 q_L；升降高度唯一表达处）
    mount.append(copy.deepcopy(env_root_body))
    # 落地视觉：基座真实落地后，轮/脚轮触地；不参与碰撞/规划
    _add_landing_visuals(mount)
    # 选定侧裸臂锚在环境链顶部 waist_yaw_link（虚拟固定安装点）
    env_tip = _find_body(mount, _ENV_TIP_BODY)
    # 对侧臂作为无关节静态视觉挂件：只渲染、不参与控制/物理。
    # 这样 njnt/nq/nv/nu 仍为 7/7/7/7，带球场景形状校验保持单臂契约。
    passive_side = _passive_side(side)
    passive_path = arm_path.parent / f"r1d_arm_{passive_side}.xml"
    if not passive_path.is_file():
        raise FileNotFoundError(
            f"对侧臂片段缺失，无法组装双臂渲染场景: {passive_path}"
        )
    passive_root = _parse_fragment(passive_path, "mujocoinclude")
    passive_worldbody = _fragment_worldbody(passive_root, passive_path)
    passive_root_body = next(
        (el for el in passive_worldbody if el.tag == "body"), None
    )
    if (
        passive_root_body is None
        or passive_root_body.attrib.get("name") != _ARM_ROOT_BODY
    ):
        raise ValueError(
            f"对侧裸臂片段 worldbody 下唯一根 body 应为 {_ARM_ROOT_BODY}"
            f"（片段: {passive_path}）"
        )
    env_tip.append(_static_arm_copy(passive_root_body, passive_side))
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
    # 拍面碰撞几何保持第一个 geom（测试契约：racket.find("geom") 是拍面）
    ET.SubElement(
        racket, "inertial", pos="0 0 0", mass="0.3", diaginertia="0 0 0"
    )
    ET.SubElement(
        racket,
        "geom",
        name=_RACKET_GEOM,
        type="cylinder",
        size="0 0",
        # 圆盘细轴 = 拍体局部 +X（T4 get_ee_normal 契约；绕 Y 旋转 90°）
        quat="0.7071067811865476 0 0.7071067811865476 0",
        rgba="0.9 0.9 0.25 1",
        contype="0",
        conaffinity="0",
    )
    # 视觉拍面：椭球面，不参与碰撞（碰撞仍由 racket_face 代理）
    ET.SubElement(
        racket,
        "geom",
        name=_RACKET_FACE_VISUAL_GEOM,
        type="ellipsoid",
        size="0 0 0",
        rgba="0.92 0.88 0.30 1",
        contype="0",
        conaffinity="0",
    )
    # 拍柄：从腕部指向拍面，补上 RM65 的“拍柄 + 拍面 + TCP”组合
    ET.SubElement(
        racket,
        "geom",
        name=_RACKET_HANDLE_GEOM,
        type="capsule",
        fromto="0 0 0 0 0 0",
        size="0.012",
        rgba="0.16 0.17 0.20 1",
        contype="0",
        conaffinity="0",
    )
    # 拍喉：两根斜杆把拍柄上端连到拍框下缘
    for name in _RACKET_THROAT_GEOMS:
        ET.SubElement(
            racket,
            "geom",
            name=name,
            type="capsule",
            fromto="0 0 0 0 0 0",
            size=_fmt_float(_RACKET_THROAT_RADIUS),
            rgba="0.10 0.11 0.13 1",
            contype="0",
            conaffinity="0",
        )
    # 拍框：上下横杆 + 左右竖杆，包住半透明拍面
    for name in _RACKET_FRAME_GEOMS:
        ET.SubElement(
            racket,
            "geom",
            name=name,
            type="capsule",
            fromto="0 0 0 0 0 0",
            size=_fmt_float(_RACKET_FRAME_RADIUS),
            rgba="0.08 0.09 0.11 1",
            contype="0",
            conaffinity="0",
        )
    ET.SubElement(racket, "site", name=_TCP_SITE, pos="0 0 0", size="0.005")

    # 简化机身 box：保留为保护几何，但视觉透明，露出真实 waist 网格
    torso = ET.SubElement(mount, "body", name=_TORSO_BODY, pos="0 0 0")
    ET.SubElement(
        torso,
        "geom",
        name=_TORSO_GEOM,
        type="box",
        size="0 0 0",
        rgba="0.55 0.57 0.6 0",
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

    # 参数数值真值统一写入（生成与加载共用同一入口，防两套代码漂移）；
    # 模板生成即默认 scene（默认升降/默认固定姿态/旧 API 球拍），
    # 因此 template_mode=True 时保留 r1d_env.xml 的 bake 高度不重烘。
    _apply_scene_parameters(root, parameters, side=side, template_mode=True)
    return root


def _template_comment(side: str) -> str:
    """模板文件头注释（说明来源与重建方式，随模板交付）。"""
    return (
        "自动生成文件，勿手改。重建：调用 src/robot/model_factory.py"
        f"::write_r1d_model_template（输入：configs/robots/r1d_runtime_{side}.yaml + "
        f"src/robot/r1d_arm_{side}.xml + src/robot/r1d_env.xml + "
        "src/robot/r1d_assets.xml）。官方臂/环境层级与惯量来自 Task 2.5a "
        "片段逐字保留（含右腕俯仰上界 +0.5326 的官方不对称）；头部网格随 "
        "r1d_env.xml 交付（环境片段自带 asset，合并进唯一 <asset> 节）；"
        f"升降高度由 r1d_env.xml 以 q_L=0.4225 bake 表达，运行时按 "
        "scene.lift_displacement 在内存中重烘；场地恢复旧 RM65 口径"
        "（court_grid 纹理 + court_floor 材质 + 10 条白线 + 单灯，无球网）。"
        f"场景/torso 数值真值为 r1d_runtime_{side}.yaml；球拍几何/质量/"
        "惯量/安装的唯一物理来源是 configs/rackets/*.yaml（RacketProfile），"
        "旧 API 模式下模板默认值为 scene.racket_*。"
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


def _model_side(profile: RobotProfile) -> str:
    """从主动关节名前缀推导场景侧别（left/right）。

    Raises:
        ValueError: 主动关节名前缀不是 left/right（无法确定场景侧别）。
    """
    names = profile.active_joint_names
    if not names:
        raise ValueError(f"机器人描述 {profile.robot_id} 缺少主动关节名")
    first = str(names[0])
    prefix = first.split("_", 1)[0]
    if prefix not in SIDES:
        raise ValueError(
            f"主动关节 {first} 的侧向前缀 {prefix!r} 不是 left/right，"
            f"无法确定场景侧别（robot_id: {profile.robot_id}）"
        )
    return prefix


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


def _has_legacy_racket(scene: Any) -> bool:
    """判断 scene 是否提供了旧 API 球拍字段（5 项全有才算提供）。"""
    fields = (
        scene.racket_position,
        scene.racket_quaternion,
        scene.racket_radius,
        scene.racket_half_thickness,
        scene.racket_mass,
    )
    return all(value is not None for value in fields)


def _require_no_legacy_racket(scene: Any) -> None:
    """RacketProfile 模式下禁止 scene 再给球拍几何/质量（单一物理来源）。

    Raises:
        ValueError: scene 仍带任一 racket_* 字段。
    """
    leftovers = [
        name
        for name in (
            "racket_position",
            "racket_quaternion",
            "racket_radius",
            "racket_half_thickness",
            "racket_mass",
        )
        if getattr(scene, name) is not None
    ]
    if leftovers:
        raise ValueError(
            f"提供 racket_profile 时 scene.racket_* 必须全部为 None"
            f"（球拍几何/质量/安装的唯一物理来源是 RacketProfile；"
            f"冲突字段: {leftovers}）"
        )


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


def _find_body_with_parent(
    root: ET.Element, name: str
) -> tuple[ET.Element, ET.Element]:
    """按名称查找 body 及其父元素（用于原位替换）。

    Raises:
        ValueError: 树中不存在该名称的 body。
    """
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "body" and child.attrib.get("name") == name:
                return parent, child
    raise ValueError(f"模板中缺少 body '{name}'（模板漂移）")


def _apply_fixed_joint_poses(
    root: ET.Element,
    parameters: RobotRuntimeParameters,
    side: str,
    profile: RobotProfile | None = None,
    *,
    fragments: list[Path] | None = None,
) -> None:
    """按配置重新烘焙对侧固定臂姿态（替换模板中的静态臂 body）。

    片段从模型模板同目录的 ``r1d_arm_{passive}.xml`` 读取；只重建被动臂
    子树，主动臂/环境链/头部不受影响，磁盘模板与片段均不被改写。

    Args:
        root: 模板 XML 根元素（原地修改）。
        parameters: 运行参数（``scene.fixed_joint_poses`` 非空才生效）。
        side: 主动臂侧别（``left`` / ``right``）。
        profile: 机器人描述（决定片段的首选目录）。
        fragments: 可选输出列表；本函数实际读到片段文件时把路径追加进去，
            供 :func:`collect_model_dependencies` 记录装配闭包（F08）。

    Raises:
        ValueError: 覆盖关节不在片段中或角度超出该关节限位。
        FileNotFoundError: 对侧臂片段缺失。
    """
    scene = parameters.scene
    if scene is None or not scene.fixed_joint_poses:
        return
    passive_side = _passive_side(side)
    fragment_path = _arm_fragment_path(profile, side)
    # F08：只有这里会读取兄弟片段——就地登记，避免调用方猜测
    if fragments is not None:
        fragments.append(Path(fragment_path).resolve())
    passive_root = _parse_fragment(fragment_path, "mujocoinclude")
    passive_worldbody = _fragment_worldbody(passive_root, fragment_path)
    passive_root_body = next(
        (el for el in passive_worldbody if el.tag == "body"), None
    )
    if passive_root_body is None:
        raise ValueError(f"对侧裸臂片段缺少根 body（片段: {fragment_path}）")
    new_body = _static_arm_copy(
        passive_root_body, passive_side, dict(scene.fixed_joint_poses)
    )
    body_name = new_body.attrib.get("name")
    parent, old_body = _find_body_with_parent(root, str(body_name))
    index = list(parent).index(old_body)
    parent.remove(old_body)
    parent.insert(index, new_body)


def _arm_fragment_path(profile: RobotProfile | None, side: str) -> Path:
    """定位对侧臂 MJCF 片段路径（优先模型模板同目录，回退仓库 src/robot）。

    Raises:
        FileNotFoundError: 两种路径都找不到片段。
    """
    passive_side = _passive_side(side)
    name = f"r1d_arm_{passive_side}.xml"
    candidates: list[Path] = []
    if profile is not None:
        candidates.append(Path(profile.model_path).parent / name)
    candidates.append(Path(__file__).resolve().parent / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"对侧臂片段缺失，无法重烘固定关节姿态（查找: "
        f"{[str(c) for c in candidates]}）"
    )


def _dependency_repo_root() -> Path:
    """仓库根目录（``src/robot/model_factory.py`` 的上两级）。"""
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    """按块读取文件并返回 SHA256 十六进制摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_relative_path(source: Path) -> str:
    """依赖文件的稳定相对路径（仓库内保留相对路径，仓库外 ``external/<sha12>/``）。"""
    resolved = Path(source).resolve()
    try:
        return resolved.relative_to(_dependency_repo_root()).as_posix()
    except ValueError:
        digest = _sha256_file(resolved)[:12] if resolved.is_file() else "missing"
        return f"external/{digest}/{resolved.name}"


def _dependency_entry(
    role: str,
    path: Path,
    *,
    reference: str = "",
) -> dict[str, Any]:
    """构造一条依赖条目（XML 片段附带 base64 内容，网格只记哈希）。

    Args:
        role: 依赖角色（``model`` / ``include`` / ``arm_fragment`` /
            ``mesh`` / ``texture``）。
        path: 依赖文件绝对路径（须存在）。
        reference: 模型 XML 中引用该文件的原始字符串（网格/includes 用；
            回放据此把 ``file=`` 属性绑定到已校验条目，而非按路径猜）。

    Returns:
        ``role/relative_path/source/sha256/size``（XML 片段另含
        ``content_base64``；``reference`` 非空时一并记录）。
    """
    resolved = Path(path).resolve()
    data = resolved.read_bytes()
    entry: dict[str, Any] = {
        "role": role,
        "relative_path": _dependency_relative_path(resolved),
        "source": str(resolved),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }
    if reference:
        entry["reference"] = reference
    if role in DEPENDENCY_XML_ROLES:
        # F08：XML 片段必须把内容写进快照，回放才能在外部来源缺失时
        # 重建同一装配输入（网格体积大，仍按哈希从仓库/外部来源解析）。
        entry["content_base64"] = base64.b64encode(data).decode("ascii")
    return entry


def mask_xml_comments(text: str) -> str:
    """把 XML 注释覆盖为等长空白掩码（保留字符位置与换行）。

    MuJoCo 的 MJCF 解析器会**整体丢弃** XML 注释，但直接对原文做正则扫描会
    把注释里的 ``<compiler>``/``<include>``/``<mesh>`` 当成真实标签（R03
    根因：注释里的诱饵 ``meshdir`` 被合并，回放网格被放大）。本函数用空白
    覆盖注释区间，使掩码文本与原文**逐字符等长**：``finditer`` 得到的 span
    可直接映射回原文定位/替换，注释内容在原文中保持不动。

    局限：按 XML 词法假定注释不会出现在属性值内部（``file="a<!--b-->c"``
    这类病态写法不在支持范围）。

    Args:
        text: 原始 XML 文本。

    Returns:
        与 ``text`` 等长的掩码文本（仅注释区间变成空白，换行保留）。
    """

    def _blank(match: re.Match[str]) -> str:
        """把一段注释替换为同长度的空白（换行不吞掉）。"""
        return "".join("\n" if char == "\n" else " " for char in match.group(0))

    return _XML_COMMENT_PATTERN.sub(_blank, text)


def _attribute_value(attrs: str, name: str) -> str | None:
    """取出 XML 标签属性值（单/双引号均可），未声明时返回 ``None``。

    Args:
        attrs: 标签内的属性串（``<tag`` 与 ``/>`` 之间的部分）。
        name: 属性名（大小写不敏感）。

    Returns:
        去掉引号后的原始取值；属性不存在时返回 ``None``——与"显式声明空
        值"（返回空串）区分，后者在 compiler 覆盖语义里是有效重置。
    """
    match = re.search(
        rf"""\b{name}\s*=\s*(?:"([^"]*)"|'([^']*)')""",
        attrs,
        re.IGNORECASE,
    )
    if match is None:
        return None
    value = match.group(1)
    if value is None:
        value = match.group(2) or ""
    return value


def _resolve_reference_path(base_dir: Path, dir_prefix: str, raw: str) -> Path:
    """把 ``file=`` 引用解析为绝对路径（相对基准 = ``base_dir`` 叠加前缀目录）。

    绝对路径直接使用；相对路径先拼目录前缀（``meshdir``/``texturedir``）。

    Args:
        base_dir: 相对基准目录。
        dir_prefix: 目录前缀（空串表示无前缀；本身可为绝对路径）。
        raw: 原始引用字符串。

    Returns:
        解析后的绝对路径（不保证文件存在）。
    """
    reference = Path(raw.strip().replace("\\", "/"))
    if reference.is_absolute():
        return reference.resolve()
    base = (base_dir / dir_prefix).resolve() if dir_prefix else base_dir
    return (base / reference).resolve()


@dataclass(slots=True)
class _AssetReference:
    """一条待解析的 mesh/texture 引用（第一阶段收集，第二阶段统一解析）。

    Attributes:
        kind: 引用标签（``mesh`` 或 ``texture``）。
        raw: ``file=`` 原始字符串（已去首尾空白）。
        declaring_file: 直接写该引用的 XML 文件绝对路径（候选 2 的基准目录）。
    """

    kind: str
    raw: str
    declaring_file: Path


def _asset_candidate_paths(
    reference: _AssetReference,
    *,
    model_dir: Path,
    asset_dirs: Mapping[str, str],
) -> tuple[Path, Path]:
    """按 MuJoCo 语义给出资产的两个候选路径（顺序即查找顺序）。

    候选 1 = ``<顶层模型目录>/<全局 meshdir|texturedir>/<file>``（目录前缀
    为绝对路径时直接生效）；候选 2 = ``<声明该引用的 XML 文件目录>/<file>``
    （**不再拼目录前缀**，与 MuJoCo 的兜底查找一致）。绝对路径引用两个候选
    相同。

    Args:
        reference: 待解析的资产引用。
        model_dir: 顶层模型文件所在目录（全局目录前缀的基准，不是声明目录）。
        asset_dirs: 最终全局 ``{kind: 目录前缀}`` 表。

    Returns:
        ``(第一候选, 第二候选)``。
    """
    prefix = asset_dirs.get(reference.kind, "")
    first = _resolve_reference_path(model_dir, prefix, reference.raw)
    second = _resolve_reference_path(reference.declaring_file.parent, "", reference.raw)
    return first, second


def _include_candidate_paths(
    raw: str,
    *,
    model_dir: Path,
    declaring_dir: Path,
) -> tuple[Path, Path]:
    """按 MuJoCo 语义给出 ``<include file>`` 的两个候选路径。

    实测（3.9.0）：include 先按**顶层模型目录**解析，不存在时才回退到
    **声明该 include 的文件目录**——与资产的两候选顺序一致。绝对路径引用
    两个候选相同。

    Args:
        raw: ``file=`` 原始字符串。
        model_dir: 顶层模型文件所在目录（第一候选基准）。
        declaring_dir: 声明该 include 的 XML 文件目录（第二候选基准）。

    Returns:
        ``(顶层候选, 声明目录候选)``。
    """
    first = _resolve_reference_path(model_dir, "", raw)
    second = _resolve_reference_path(declaring_dir, "", raw)
    return first, second


def _walk_model_file(
    path: Path,
    *,
    role: str,
    reference: str,
    model_dir: Path,
    entries: list[dict[str, Any]],
    asset_refs: list[_AssetReference],
    asset_dirs: dict[str, str],
    missing: list[str],
    seen: set[Path],
    depth: int,
) -> None:
    """第一阶段：按文档顺序展开 MJCF，收集文件条目、资产引用与全局编译目录。

    与旧实现的关键差异：

    * 扫描前先用 :func:`mask_xml_comments` 掩码注释——注释里的标签不生效；
    * ``<compiler>`` 在这里**只记录不解析**——它是全局设置，include 片段里
      的声明同样生效，同名属性按文档顺序后者覆盖前者；``assetdir`` 是
      ``meshdir``/``texturedir`` 的别名（同一标签内显式属性覆盖它）；
    * ``<include>`` 先按顶层模型目录解析，不存在再回退声明目录；
    * 资产路径一律留给第二阶段用最终目录表统一解析。MuJoCo 同样是在解析完
      整个文档后才加载资产，因此这个顺序与真实加载行为一致。

    Args:
        path: 待展开的 XML 文件。
        role: 该文件的角色（首层 ``model``，嵌套层 ``include``）。
        reference: 上层引用该文件的原始字符串（首层为空）。
        model_dir: 顶层模型文件所在目录（include 第一候选的基准，递归不变）。
        entries: 输出文件条目列表（原地追加，按首次访问去重）。
        asset_refs: 输出资产引用列表（原地追加，保持文档顺序）。
        asset_dirs: 全局 ``{kind: 目录前缀}`` 表（原地更新，后者覆盖）。
        missing: 输出缺失引用列表（原地追加；含 include 与 XML 文件缺失）。
        seen: 已访问文件集合（防重复/环）。
        depth: 当前 include 深度（超过 :data:`_MAX_INCLUDE_DEPTH` 直接报错）。
    """
    resolved = Path(path).resolve()
    if resolved in seen:
        return
    if depth > _MAX_INCLUDE_DEPTH:
        raise ValueError(
            f"模型 include 嵌套超过 {_MAX_INCLUDE_DEPTH} 层，拒绝继续展开"
            f"（可能是环状 include）：{resolved}"
        )
    seen.add(resolved)
    if not resolved.is_file():
        missing.append(str(resolved))
        return
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError:
        missing.append(str(resolved))
        return
    entries.append(_dependency_entry(role, resolved, reference=reference))
    # 注释整体忽略：注释里的 compiler/include/mesh 都不参与解析（R03）
    masked = mask_xml_comments(text)
    for match in _MJCF_TAG_PATTERN.finditer(masked):
        tag = str(match.group("tag")).lower()
        attrs = match.group("attrs")
        if tag == "compiler":
            # 全局编译设置：逐属性合并，未声明的属性保留上一条取值。
            # assetdir 同时设置 meshdir/texturedir，同一标签内的显式属性
            # 覆盖它（与属性书写顺序无关）；跨标签按文档顺序后者覆盖前者。
            assetdir = _attribute_value(attrs, "assetdir")
            if assetdir is not None:
                for kind in _ASSET_DIR_ATTRS:
                    asset_dirs[kind] = assetdir.strip()
            for kind, attr_name in _ASSET_DIR_ATTRS.items():
                value = _attribute_value(attrs, attr_name)
                if value is not None:
                    asset_dirs[kind] = value.strip()
            continue
        raw = (_attribute_value(attrs, "file") or "").strip()
        if not raw:
            continue
        if tag == "include":
            # include：顶层模型目录优先，声明目录兜底（R04，实测 3.9.0）
            first, second = _include_candidate_paths(
                raw,
                model_dir=model_dir,
                declaring_dir=resolved.parent,
            )
            target = first if first.is_file() else second
            _walk_model_file(
                target,
                role=DEPENDENCY_ROLE_INCLUDE,
                reference=raw,
                model_dir=model_dir,
                entries=entries,
                asset_refs=asset_refs,
                asset_dirs=asset_dirs,
                missing=missing,
                seen=seen,
                depth=depth + 1,
            )
            continue
        asset_refs.append(
            _AssetReference(kind=tag, raw=raw, declaring_file=resolved)
        )


def collect_model_file_dependencies(
    model_path: Path,
    *,
    asset_missing: list[str] | None = None,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    """收集一个 MJCF 模型的**文件级**依赖闭包（模型/嵌套 include/网格纹理）。

    两阶段实现（对齐 MuJoCo 语义）：

    1. 按文档顺序展开 include 树，收集文件条目、每条 ``<mesh>/<texture>``
       引用（含声明目录）与**最终** ``<compiler>`` 目录表——compiler 是
       全局设置，include 片段里的声明同样生效，同名属性后者覆盖前者，
       ``assetdir`` 同时设置 meshdir/texturedir；注释整体忽略。include
       自身先按顶层模型目录解析、再回退声明目录。
    2. 用最终目录表统一解析资产：先 ``<顶层目录>/<全局 dir>/<file>``，不
       存在再 ``<声明目录>/<file>``；绝对路径直接使用。缺失时报告第一候选。

    模型条目会附带可选 ``asset_dirs``（最终 meshdir/texturedir），供回放
    物化时按同一语义重写资产引用（旧产物无该字段时回退旧解析）。

    Args:
        model_path: 模型 XML 路径。
        asset_missing: 可选的输出列表；传入时把**资产引用**的缺失路径追加
            进去（``missing`` 返回值仍混合 include 缺失，供整体闭包报告）。

    Returns:
        ``(entries, missing)``：``entries`` 为按
        ``(role, relative_path)`` 排序的依赖条目元组；``missing`` 为无法
        读取的引用路径元组（不抛异常——缺失由产物 ``assets.missing`` 与
        回放校验统一报告）。

    Raises:
        ValueError: include 嵌套超过 :data:`_MAX_INCLUDE_DEPTH` 层。
    """
    model_file = Path(model_path).resolve()
    entries: list[dict[str, Any]] = []
    asset_refs: list[_AssetReference] = []
    asset_dirs: dict[str, str] = {}
    missing: list[str] = []
    seen: set[Path] = set()
    _walk_model_file(
        model_file,
        role=DEPENDENCY_ROLE_MODEL,
        reference="",
        model_dir=model_file.parent,
        entries=entries,
        asset_refs=asset_refs,
        asset_dirs=asset_dirs,
        missing=missing,
        seen=seen,
        depth=0,
    )
    seen_assets: set[str] = set()
    for reference in asset_refs:
        first, second = _asset_candidate_paths(
            reference,
            model_dir=model_file.parent,
            asset_dirs=asset_dirs,
        )
        target = first if first.is_file() else second
        if not target.is_file():
            # 缺失时报告第一候选路径（与 MuJoCo 的报错口径一致）
            if str(first) not in seen_assets:
                seen_assets.add(str(first))
                missing.append(str(first))
                if asset_missing is not None:
                    asset_missing.append(str(first))
            continue
        key = str(target)
        if key in seen_assets:
            continue
        seen_assets.add(key)
        entries.append(_dependency_entry(reference.kind, target, reference=reference.raw))
    if asset_dirs:
        for entry in entries:
            if entry["role"] == DEPENDENCY_ROLE_MODEL:
                entry["asset_dirs"] = dict(asset_dirs)
                break
    ordered = tuple(sorted(entries, key=lambda item: (item["role"], item["relative_path"])))
    return ordered, tuple(missing)


def collect_model_dependencies(
    profile: RobotProfile,
    parameters: RobotRuntimeParameters,
    *,
    arm_fragments: Sequence[Path] = (),
) -> tuple[dict[str, Any], ...]:
    """汇总一次装配实际读取的完整依赖闭包。

    文件级闭包来自 :func:`collect_model_file_dependencies`（模型 +
    ``<include>`` + 网格纹理）；对侧臂片段由调用方按**真实装配分支**传入
    ``arm_fragments``（只有 ``scene.fixed_joint_poses`` 非空时工厂才会读取
    ``r1d_arm_{passive}.xml``，F08 根因即该片段此前未进快照）。

    Args:
        profile: 机器人描述（``model_path`` 为实际加载的模板）。
        parameters: 运行参数（当前不参与闭包判定，保留以表达"同一次装配"）。
        arm_fragments: 装配过程中真实读取的对侧臂片段路径。

    Returns:
        依赖条目元组（按 ``role/relative_path`` 排序，按绝对路径去重）。
    """
    entries, _ = collect_model_file_dependencies(profile.model_path)
    merged: list[dict[str, Any]] = list(entries)
    for fragment in arm_fragments:
        merged.append(
            _dependency_entry(DEPENDENCY_ROLE_ARM_FRAGMENT, Path(fragment).resolve())
        )
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in merged:
        deduped.setdefault((str(item["role"]), str(item["source"])), item)
    return tuple(
        sorted(deduped.values(), key=lambda item: (item["role"], item["relative_path"]))
    )


def _racket_inertial(
    profile: RacketProfile,
) -> tuple[float, np.ndarray, np.ndarray]:
    """合并球拍与连接件质量/COM/惯量（相对合并 COM，球拍局部系）。

    连接件只在安装时计入一次；连接件惯量按其自身质心给出，合并时用
    平行轴定理搬到合并质心。
    """
    mass = float(profile.mass)
    com = np.asarray(profile.com, dtype=np.float64)
    inertia = np.asarray(profile.inertia.as_matrix(), dtype=np.float64)
    connector = profile.connector
    if connector is None:
        return mass, com, inertia
    connector_mass = float(connector.mass)
    connector_com = np.asarray(connector.com, dtype=np.float64)
    connector_inertia = np.asarray(connector.inertia.as_matrix(), dtype=np.float64)
    total = mass + connector_mass
    merged_com = (mass * com + connector_mass * connector_com) / total

    def _shift(
        local_inertia: np.ndarray,
        local_mass: float,
        local_com: np.ndarray,
        target_com: np.ndarray,
    ) -> np.ndarray:
        offset = local_com - target_com
        return local_inertia + local_mass * (
            float(np.dot(offset, offset)) * np.eye(3) - np.outer(offset, offset)
        )

    merged_inertia = _shift(
        inertia, mass, com, merged_com
    ) + _shift(connector_inertia, connector_mass, connector_com, merged_com)
    return total, merged_com, merged_inertia


def _racket_assembly(profile: RacketProfile) -> RacketAssembly:
    """把 RacketProfile 的装配事实固化为 bundle 暴露的只读记录。"""
    total_mass, com, inertia = _racket_inertial(profile)
    connector_mass = 0.0 if profile.connector is None else float(
        profile.connector.mass
    )
    geometry = profile.geometry
    return RacketAssembly(
        racket_id=str(profile.racket_id),
        source_path=Path(profile.source_path),
        total_mass=total_mass,
        com=np.array(com, dtype=np.float64, copy=True),
        inertia_diagonal=np.array(np.diag(inertia), dtype=np.float64, copy=True),
        face_center=np.array(geometry.face.center, dtype=np.float64, copy=True),
        face_semi_axes=np.array(geometry.face.semi_axes, dtype=np.float64, copy=True),
        mount_position=np.array(
            profile.mount.position, dtype=np.float64, copy=True
        ),
        mount_wxyz=np.array(profile.mount.wxyz, dtype=np.float64, copy=True),
        connector_mass=connector_mass,
    )


def _quat_aligning_z(axis: Any) -> np.ndarray:
    """求把局部 +Z 轴对齐到给定方向的单位四元数（wxyz）。

    Raises:
        ValueError: 轴向为零向量。
    """
    direction = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("连接件安装轴为零向量（RacketProfile 非法）")
    direction = direction / norm
    dot = float(direction[2])
    if dot > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -1.0 + 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    cross = np.cross(np.array([0.0, 0.0, 1.0]), direction)
    quat = np.array([1.0 + dot, cross[0], cross[1], cross[2]], dtype=np.float64)
    return quat / float(np.linalg.norm(quat))


def _append_visual_geom(
    body: ET.Element,
    *,
    name: str,
    geom_type: str,
    size: str,
    rgba: str,
    pos: str | None = None,
    fromto: str | None = None,
    quat: str | None = None,
) -> ET.Element:
    """追加一个不参与碰撞（contype=0/conaffinity=0）的视觉 geom。"""
    attrs: dict[str, str] = {
        "name": name,
        "type": geom_type,
        "size": size,
        "rgba": rgba,
        "contype": "0",
        "conaffinity": "0",
    }
    if pos is not None:
        attrs["pos"] = pos
    if fromto is not None:
        attrs["fromto"] = fromto
    if quat is not None:
        attrs["quat"] = quat
    return ET.SubElement(body, "geom", attrs)


def _apply_racket_profile(
    root: ET.Element, profile: RacketProfile, flange_pos: np.ndarray
) -> None:
    """按 RacketProfile 重建球拍 body 子树（几何/质量/惯量/安装单一来源）。

    视觉/碰撞/TCP 同源：拍面碰撞椭球与视觉椭球都以 face.center 为中心、
    semi_axes 为半轴，细轴（半厚）沿球拍局部 +X，即 ``get_ee_normal`` 的
    拍面法向契约；TCP site 位于 face.center。

    Raises:
        ValueError: 模板缺少球拍 body 或 profile 质心/惯量非法。
    """
    _parent, body = _find_body_with_parent(root, _RACKET_BODY)
    for child in list(body):
        body.remove(child)
    mount = profile.mount
    body.set("pos", _fmt_vec(np.asarray(flange_pos) + np.asarray(mount.position)))
    body.set("quat", _fmt_vec(mount.wxyz))

    total_mass, com, inertia = _racket_inertial(profile)
    diagonal = np.diag(inertia)
    off_diagonal = inertia - np.diag(diagonal)
    inertial_attrs: dict[str, str] = {
        "pos": _fmt_vec(com),
        "mass": _fmt_float(total_mass),
    }
    if float(np.max(np.abs(off_diagonal))) < 1e-12:
        inertial_attrs["diaginertia"] = _fmt_vec(diagonal)
    else:
        inertial_attrs["fullinertia"] = _fmt_vec(
            (
                inertia[0, 0],
                inertia[1, 1],
                inertia[2, 2],
                inertia[0, 1],
                inertia[0, 2],
                inertia[1, 2],
            )
        )
    ET.SubElement(body, "inertial", inertial_attrs)

    geometry = profile.geometry
    face = geometry.face
    # 拍面碰撞几何：椭球，细轴（半厚）沿局部 +X ⇒ 法向 = 局部 +X
    ET.SubElement(
        body,
        "geom",
        {
            "name": _RACKET_GEOM,
            "type": "ellipsoid",
            "pos": _fmt_vec(face.center),
            "size": _fmt_vec(face.semi_axes),
            "rgba": "0.9 0.9 0.25 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    _append_visual_geom(
        body,
        name=_RACKET_FACE_VISUAL_GEOM,
        geom_type="ellipsoid",
        pos=_fmt_vec(face.center),
        size=_fmt_vec(face.semi_axes),
        rgba=_RACKET_FACE_VISUAL_RGBA,
    )
    handle = geometry.handle
    _append_visual_geom(
        body,
        name=_RACKET_HANDLE_GEOM,
        geom_type="capsule",
        fromto=_fmt_vec((*handle.start, *handle.end)),
        size=_fmt_float(handle.radius),
        rgba=_RACKET_HANDLE_RGBA,
    )
    for index, capsule in enumerate(geometry.throat):
        _append_visual_geom(
            body,
            name=_RACKET_THROAT_GEOM_FORMAT.format(index=index),
            geom_type="capsule",
            fromto=_fmt_vec((*capsule.start, *capsule.end)),
            size=_fmt_float(capsule.radius),
            rgba=_RACKET_THROAT_RGBA,
        )
    for index, capsule in enumerate(geometry.frame):
        _append_visual_geom(
            body,
            name=_RACKET_FRAME_GEOM_FORMAT.format(index=index),
            geom_type="capsule",
            fromto=_fmt_vec((*capsule.start, *capsule.end)),
            size=_fmt_float(capsule.radius),
            rgba=_RACKET_FRAME_RGBA,
        )
    connector = profile.connector
    if connector is not None:
        _append_visual_geom(
            body,
            name=_RACKET_CONNECTOR_GEOM,
            geom_type="cylinder",
            pos=_fmt_vec(connector.center),
            quat=_fmt_vec(_quat_aligning_z(connector.axis)),
            size=_fmt_vec((connector.radius, connector.length / 2.0)),
            rgba=_RACKET_CONNECTOR_RGBA,
        )
    ET.SubElement(
        body, "site", name=_TCP_SITE, pos=_fmt_vec(face.center), size="0.005"
    )


def _apply_legacy_racket(
    root: ET.Element, parameters: RobotRuntimeParameters, flange_pos: np.ndarray
) -> None:
    """旧 API 球拍参数化（无 RacketProfile 时的 T0 语义，保持向后兼容）。

    圆盘碰撞面 + 0.30 kg 缩放惯量；拍面细轴（局部 Z）经 quat 对齐到
    body 局部 +X（``get_ee_normal`` 契约）。

    Raises:
        ValueError: scene 未提供旧球拍字段或模板缺少期望元素。
    """
    scene = parameters.scene
    if scene is None or not _has_legacy_racket(scene):
        raise ValueError("旧 API 球拍参数缺失：需要 scene 提供完整 racket_* 字段")
    racket = _find_body(root, _RACKET_BODY)
    racket.set("pos", _fmt_vec(flange_pos + np.asarray(scene.racket_position)))
    racket.set("quat", _fmt_vec(scene.racket_quaternion))
    inertial = racket.find("inertial")
    racket_geom = racket.find("geom")
    if inertial is None or racket_geom is None:
        raise ValueError("模板球拍 body 缺少 inertial/geom（模板漂移）")
    inertial.set("mass", _fmt_float(float(scene.racket_mass)))
    # 复合体惯量（拍头+拍柄）：轴向 X 约 0.0018、横向 Y/Z 约 0.004
    # （0.30 kg 成人拍的迷你化近似；避免腕部动力学被放大 10 倍）。
    scale = float(scene.racket_mass) / 0.30
    inertial.set(
        "diaginertia",
        _fmt_vec((0.0018 * scale, 0.0040 * scale, 0.0040 * scale)),
    )
    racket_geom.set(
        "size",
        _fmt_vec((float(scene.racket_radius), float(scene.racket_half_thickness))),
    )

    face_visual = next(
        (
            geom
            for geom in root.iter("geom")
            if geom.get("name") == _RACKET_FACE_VISUAL_GEOM
        ),
        None,
    )
    if face_visual is not None:
        face_visual.set(
            "size",
            _fmt_vec(
                (
                    _RACKET_FACE_HALF_THICKNESS,
                    _RACKET_FACE_HALF_WIDTH,
                    _RACKET_FACE_HALF_LENGTH,
                )
            ),
        )
        face_visual.set("rgba", "0.92 0.92 0.84 0.30")

    handle = next(
        (
            geom
            for geom in root.iter("geom")
            if geom.get("name") == _RACKET_HANDLE_GEOM
        ),
        None,
    )
    if handle is not None:
        handle_start_z = -float(scene.racket_position[2]) + 0.02
        handle_end_z = handle_start_z + _RACKET_HANDLE_LENGTH
        handle.set(
            "fromto",
            _fmt_vec((0.0, 0.0, handle_start_z, 0.0, 0.0, handle_end_z)),
        )
        handle.set("size", _fmt_float(_RACKET_HANDLE_RADIUS))

        throat_starts = (
            (0.0, 0.035, handle_end_z, 0.0, 0.085, -_RACKET_FACE_HALF_LENGTH),
            (0.0, -0.035, handle_end_z, 0.0, -0.085, -_RACKET_FACE_HALF_LENGTH),
        )
        for name, values in zip(_RACKET_THROAT_GEOMS, throat_starts):
            _find_geom(root, name).set("fromto", _fmt_vec(values))

        frame_values = (
            (0.0, -_RACKET_FACE_HALF_WIDTH, _RACKET_FACE_HALF_LENGTH,
             0.0, _RACKET_FACE_HALF_WIDTH, _RACKET_FACE_HALF_LENGTH),
            (0.0, -_RACKET_FACE_HALF_WIDTH, -_RACKET_FACE_HALF_LENGTH,
             0.0, _RACKET_FACE_HALF_WIDTH, -_RACKET_FACE_HALF_LENGTH),
            (0.0, _RACKET_FACE_HALF_WIDTH, -_RACKET_FACE_HALF_LENGTH,
             0.0, _RACKET_FACE_HALF_WIDTH, _RACKET_FACE_HALF_LENGTH),
            (0.0, -_RACKET_FACE_HALF_WIDTH, -_RACKET_FACE_HALF_LENGTH,
             0.0, -_RACKET_FACE_HALF_WIDTH, _RACKET_FACE_HALF_LENGTH),
        )
        for name, values in zip(_RACKET_FRAME_GEOMS, frame_values):
            _find_geom(root, name).set("fromto", _fmt_vec(values))


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


def _fmt_float(value: float) -> str:
    """把浮点数格式化为 12 位有效数字字符串（往返精度高于测试容差）。"""
    return f"{float(value):.12g}"


def _fmt_vec(values: Any) -> str:
    """把数值序列格式化为空格分隔字符串（逐项 12 位有效数字）。"""
    return " ".join(_fmt_float(item) for item in values)


def _find_body_optional(root: ET.Element, name: str) -> ET.Element | None:
    """按名称查找 body，不存在时返回 None（用于可选结构/合成夹具）。"""
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    return None


def _find_body(root: ET.Element, name: str) -> ET.Element:
    """在元素树中按名称查找 body，找不到即报模板漂移错误。

    Raises:
        ValueError: 树中不存在该名称的 body。
    """
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    raise ValueError(f"模板中缺少 body '{name}'（模板漂移）")
