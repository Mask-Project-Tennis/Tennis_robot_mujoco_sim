"""R1D 模型模板生成（MJCF 组装骨架）。

本模块负责把 R1D 机器人 XML 片段（臂、环境、资产）合并为 MuJoCo 模板：
读取片段、重写 mesh 路径、写入球场/场景数值与球拍视觉。模板注释中的
``src/robot/model_factory.py::write_r1d_model_template`` 是历史公开入口，
保持逐字不变以避免模板字节漂移（旧 facade 仍导出同名函数）。

分层：本模块不 import ``scene``；``_build_template_element`` 在调用期延迟
import 场景模块，以避免 import 期循环。
"""

from __future__ import annotations

import copy
import logging
import math
import os
import shutil
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.robot.court_geometry import build_court_lines
from src.robot.robot_profile import RobotProfile
from src.robot.runtime_parameters import RobotRuntimeParameters

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

# A1：依赖角色常量、XML 扫描常量与实现已迁至 ``src.robot.model_assets``；
# 本模块在顶部显式重导出，旧导入路径继续返回同一对象（is 相同）。


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
        *pairwise(_ENV_CHAIN_BODIES),
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
    # 延迟导入：模板生成需要写入场景数值与球场白线口径，而 scene 模块
    # 依赖本模块的结构与常量；调用期导入可保持 import 期单向、无循环。
    from src.robot.model_building import scene as _scene

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
    court_profile = _scene._court_profile_of(parameters)
    if court_profile == "itf_2026_v1":
        _scene._neutralize_court_ground(root)
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
    _scene._apply_scene_parameters(root, parameters, side=side, template_mode=True)
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
