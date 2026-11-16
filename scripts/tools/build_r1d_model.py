"""R1D 双侧裸臂与环境 MJCF 片段生成器(阶段 2 · Task 2.5a)。

从固定 commit(157ff950351e5f535e03c5d85a16b7d594b3dbcc)的宇树 R1D
整机官方 URDF(unitree_ros/robots/r1d_description/r1d.urdf)导出四份
``<mujocoinclude>`` 片段:

- ``src/robot/r1d_arm_left.xml`` / ``src/robot/r1d_arm_right.xml``:
  选定侧 7 关节裸臂链。根 body 为 ``r1d_base``——waist_yaw_link 之上
  的虚拟固定安装点(无质量/无几何,场景组装时挂位姿);关节/连杆层级、
  惯量与网格引用逐字来自 URDF(含肩部 ±0.26187 rad 安装 rpy 的 Z 形
  偏置);wrist_yaw_link 下建 ``r1d_flange`` site,位置 = 该侧
  dex1_base_joint origin(工具安装参考,不声称等于实物安装面)。
  dex1 夹爪(prismatic ×2)与腕部相机固定子树移除并记入 manifest。
- ``src/robot/r1d_env.xml``:冻结环境几何,无任何关节。链为
  base_link → lift_mount →(stage1@q_L, stage2@q_L)→ pelvis →
  waist_yaw@0;两级 prismatic 按 mimic multiplier=1 的同一 q_L bake
  为固定偏移(默认 0,行程 [0, 0.425] m,总行程 2×0.425)。轮/脚轮/
  头/对侧臂不进入;waist_yaw_link 几何保留(臂挂接点)。
- ``src/robot/r1d_assets.xml``:双侧臂 + 环境引用的全部 mesh 声明
  (共 20 个,支持任一侧场景组装)。

两份臂片段由同一工具对同一 URDF 仅以 side 参数区分生成(独立导出,
非单侧生成后镜像改名)。生成过程完全确定:相同输入字节一致。

转换忠实性约定:

1. 保留 URDF 关节 origin/axis、限位(逐字 URDF 值,含右腕俯仰上界
   +0.5326 的官方不对称,不得"修正")、连杆质量/COM/完整惯量张量
   (fullinertia),不做 balanceinertia 等任何修补;惯量不满足编译
   要求时直接报错并给出证据。
2. 关节限位由 rad 转为 degree 写出(片段声明 ``compiler angle="degree"``),
   测试在编译后对照 rad 值(tests/test_r1d_assets.py)。
3. URDF 的 effort/velocity 非 0(官方声明值,未经实机验证,按 N0-9
   不作为安全限值):不生成 ctrlrange/forcerange/速度限值。
4. 仅生成视觉 geom(contype=0/conaffinity=0,自碰撞关闭约定);
   URDF collision 与 visual 引用同网格(逐连杆校验),碰撞几何由
   后续场景/诊断模型决定。
5. URDF 未提供关节阻尼/armature,不虚构数值。

用法::

    python scripts/tools/build_r1d_model.py            # 按默认路径重建交付物
    python scripts/tools/build_r1d_model.py --qL 0.2125  # 指定升降 bake 值
"""

from __future__ import annotations

import argparse
import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

# Task 2.0 固定的官方来源(与审计报告 §1 一致)
PINNED_REPO_URL = "https://github.com/unitreerobotics/unitree_ros"
PINNED_COMMIT = "157ff950351e5f535e03c5d85a16b7d594b3dbcc"
PINNED_URDF_SHA256 = (
    "39ad1a206c34c3deecd33745ccda0c0c94ece4195f674dfbaf950f567f08809c"
)

# 片段内网格文件的固定相对前缀:自片段部署位置 src/robot/ 指向
# assets/unitree_r1d/meshes/,沿用 RM65/D2 模型的相对路径参照模式。
# 前缀与输出位置无关,保证重建字节确定。
MESH_DIR_RELATIVE_TO_XML = "../../assets/unitree_r1d/meshes"

# 派生模型契约:臂片段根 body(waist_yaw_link 之上的虚拟固定安装点)
ROOT_BODY_NAME = "r1d_base"
FLANGE_SITE_NAME = "r1d_flange"

# 双级升降 bake 参数(审计报告 §4):单级行程 [0, 0.425] m,
# mimic multiplier=1 → 两级同值 bake,总行程 2×0.425 m
LIFT_TRAVEL_RANGE_M = (0.0, 0.425)

# 环境冻结链(joint 名, 子 link 名),自 base_link 至 waist_yaw_link
_ENV_CHAIN: tuple[tuple[str, str], ...] = (
    ("lift_mount_joint", "lift_mount_link"),
    ("lift_stage1_joint", "lift_stage1_link"),
    ("lift_stage2_joint", "lift_stage2_link"),
    ("upper_body_mount_joint", "pelvis_link"),
    ("waist_yaw_joint", "waist_yaw_link"),
)
_ENV_ROOT_LINK = "base_link"
_ENV_PRISMATIC_JOINTS = ("lift_stage1_joint", "lift_stage2_joint")

# 臂链语义顺序(审计报告 §2):肩俯仰/肩侧摆/肩偏航/肘/腕滚转/腕俯仰/腕偏航
_ARM_SEMANTIC_SUFFIXES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)

# 预期 URDF 结构(面向固定版本的严格断言):40 link + 39 joint
_EXPECTED_LINKS = frozenset(
    {
        "base_link",
        "Left_Wheel_link",
        "Right_Wheel_link",
        "lift_mount_link",
        "lift_stage1_link",
        "lift_stage2_link",
        "pelvis_link",
        "waist_yaw_link",
        "head_pitch_Link",
        "head_yaw_Link",
    }
    | {
        f"{corner}_{side}_caster_{axis}_link"
        for corner in ("front", "rear")
        for side in ("left", "right")
        for axis in ("Yaw", "Roll")
    }
    | {
        f"{side}_{suffix}"
        for side in ("left", "right")
        for suffix in (
            "shoulder_pitch_Link",
            "shoulder_roll_link",
            "shoulder_yaw_link",
            "elbow_link",
            "wrist_roll_link",
            "wrist_pitch_link",
            "wrist_yaw_link",
            "dex1_base_link",
            "dex1_finger_link_1",
            "dex1_finger_link_2",
            "dex1_camera_link",
        )
    }
)
_EXPECTED_JOINTS = frozenset(
    {
        "Left_Wheel_joint",
        "Right_Wheel_joint",
        "lift_mount_joint",
        "lift_stage1_joint",
        "lift_stage2_joint",
        "upper_body_mount_joint",
        "waist_yaw_joint",
        "head_pitch_joint",
        "head_yaw_joint",
    }
    | {
        f"{corner}_{side}_caster_{axis}_joint"
        for corner in ("front", "rear")
        for side in ("left", "right")
        for axis in ("Yaw", "Roll")
    }
    | {
        f"{side}_{suffix}"
        for side in ("left", "right")
        for suffix in (
            "shoulder_pitch_joint",
            "shoulder_roll_joint",
            "shoulder_yaw_joint",
            "elbow_joint",
            "wrist_roll_joint",
            "wrist_pitch_joint",
            "wrist_yaw_joint",
            "dex1_base_joint",
            "dex1_finger_joint_1",
            "dex1_finger_joint_2",
            "dex1_camera_joint",
        )
    }
)


def _sha256_of(path: Path) -> str:
    """计算文件内容的 SHA256 十六进制摘要。

    Args:
        path: 目标文件路径。

    Returns:
        64 位十六进制摘要字符串。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _floats(text: str) -> list[float]:
    """把空格分隔的数值串解析为 float 列表。

    Args:
        text: URDF 属性中的数值串(允许多余空格)。

    Returns:
        解析后的 float 列表。
    """
    return [float(token) for token in text.split()]


def _fmt(value: float) -> str:
    """以 10 位有效数字格式化数值。

    用于 rad→degree 转换与 z bake 加法结果的写出;10 位有效数字使
    往返误差 < 1e-9 rad / 1e-12 m。

    Args:
        value: 待格式化的数值。

    Returns:
        格式化后的字符串。
    """
    return format(value, ".10g")


def _rpy_to_quat_wxyz(rpy: tuple[float, float, float]) -> tuple[float, ...]:
    """URDF 固定轴 rpy(弧度,绕父系 X-Y-Z 旋转)转 MuJoCo wxyz 四元数。

    Args:
        rpy: (roll, pitch, yaw),单位 rad。

    Returns:
        (w, x, y, z) 四元数分量。
    """
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _origin_attrs(origin_el: ET.Element | None) -> dict[str, str]:
    """URDF <origin> 转 MJCF pos/quat 属性(xyz 数值串逐字保留)。

    零平移/零旋转省略属性;rpy 非零时换算为 wxyz 四元数。

    Args:
        origin_el: URDF origin 元素(可为 None,视为恒等变换)。

    Returns:
        属性字典(可能为空)。
    """
    attrs: dict[str, str] = {}
    if origin_el is None:
        return attrs
    xyz_tokens = origin_el.attrib.get("xyz", "0 0 0").split()
    rpy = tuple(_floats(origin_el.attrib.get("rpy", "0 0 0")))
    if any(token not in ("0", "0.0", "-0", "-0.0") for token in xyz_tokens):
        attrs["pos"] = " ".join(xyz_tokens)
    if any(value != 0.0 for value in rpy):
        quat = _rpy_to_quat_wxyz(rpy)
        attrs["quat"] = " ".join(_fmt(v) for v in quat)
    return attrs


def _origin_attrs_with_z_bake(
    origin_el: ET.Element | None, q_L: float
) -> dict[str, str]:
    """URDF <origin> 转 MJCF pos/quat 属性,xyz 的 z 分量叠加升降 bake 值。

    用于两级 prismatic 冻结:pos z = origin z + q_L(两级同值 bake,
    对应 mimic multiplier=1)。其余分量与 rpy 处理同 ``_origin_attrs``。

    Args:
        origin_el: URDF origin 元素(可为 None)。
        q_L: 升降 bake 值 q_L (m)。

    Returns:
        属性字典(可能为空)。
    """
    attrs: dict[str, str] = {}
    if origin_el is None:
        xyz_tokens = ["0", "0", "0"]
    else:
        xyz_tokens = origin_el.attrib.get("xyz", "0 0 0").split()
    baked = list(xyz_tokens)
    baked[2] = _fmt(float(xyz_tokens[2]) + q_L)
    if any(token not in ("0", "0.0", "-0", "-0.0") for token in baked):
        attrs["pos"] = " ".join(baked)
    if origin_el is not None:
        rpy = tuple(_floats(origin_el.attrib.get("rpy", "0 0 0")))
        if any(value != 0.0 for value in rpy):
            quat = _rpy_to_quat_wxyz(rpy)
            attrs["quat"] = " ".join(_fmt(v) for v in quat)
    return attrs


def _parse_pinned_urdf(urdf_path: Path) -> ET.Element:
    """校验并解析固定 commit 的官方 URDF。

    Args:
        urdf_path: URDF 文件路径(内容须与 PINNED_URDF_SHA256 逐字节一致)。

    Returns:
        URDF 根元素(<robot>)。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 内容哈希与固定 commit 不符(转换器仅面向该版本)。
    """
    digest = _sha256_of(urdf_path)
    if digest != PINNED_URDF_SHA256:
        raise ValueError(
            f"URDF 哈希与固定 commit {PINNED_COMMIT[:12]} 不符: {urdf_path} "
            f"(实际 {digest}, 期望 {PINNED_URDF_SHA256})"
        )
    return ET.parse(urdf_path).getroot()


def _collect_structure(
    robot_el: ET.Element,
) -> tuple[dict[str, ET.Element], dict[str, ET.Element], dict[str, list[tuple[str, str]]]]:
    """收集并严格校验整机 URDF 运动链结构。

    面向固定 URDF:恰好 40 个 link、39 个 joint、唯一根 base_link。
    任何偏离立即报错,杜绝静默容错。

    Args:
        robot_el: URDF 根元素。

    Returns:
        (link 名 → link 元素, joint 名 → joint 元素,
         父 link 名 → [(joint 名, 子 link 名)], 根 link 名仅 base_link)。

    Raises:
        ValueError: 结构与固定版本不符。
    """
    links = {el.attrib["name"]: el for el in robot_el.findall("link")}
    joints = {el.attrib["name"]: el for el in robot_el.findall("joint")}
    if set(links) != _EXPECTED_LINKS:
        missing = sorted(_EXPECTED_LINKS - set(links))
        extra = sorted(set(links) - _EXPECTED_LINKS)
        raise ValueError(f"连杆集合与固定 URDF 不符: 缺 {missing}, 多 {extra}")
    if set(joints) != _EXPECTED_JOINTS:
        missing = sorted(_EXPECTED_JOINTS - set(joints))
        extra = sorted(set(joints) - _EXPECTED_JOINTS)
        raise ValueError(f"关节集合与固定 URDF 不符: 缺 {missing}, 多 {extra}")

    children: dict[str, list[tuple[str, str]]] = {}
    child_links: set[str] = set()
    for name, joint_el in joints.items():
        parent = joint_el.find("parent").attrib["link"]
        child = joint_el.find("child").attrib["link"]
        children.setdefault(parent, []).append((name, child))
        child_links.add(child)
    roots = [name for name in links if name not in child_links]
    if roots != [_ENV_ROOT_LINK]:
        raise ValueError(f"根连杆应为唯一的 {_ENV_ROOT_LINK}: {roots}")
    return links, joints, children


def _resolve_mesh(filename: str, urdf_path: Path) -> tuple[str, str]:
    """解析 URDF 相对网格引用,返回 (mesh 名, 片段内相对路径)。

    R1D URDF 使用相对路径引用(``meshes/*.STL``,非 package:// URI),
    网格位于 URDF 同目录的 meshes/ 子目录,与上游仓库布局一致;
    逐文件校验存在性(任一官方网格取不到即报错,不得用占位替代)。

    Args:
        filename: URDF mesh filename 属性(相对路径)。
        urdf_path: URDF 文件路径。

    Returns:
        (mesh 名(STL 主文件名), 片段内相对文件路径)。

    Raises:
        FileNotFoundError: 本地资产目录中缺少网格文件。
        ValueError: 非 meshes/ 相对引用。
    """
    rel = filename.replace("\\", "/")
    parts = rel.split("/")
    if len(parts) != 2 or parts[0] != "meshes":
        raise ValueError(f"非 meshes/ 相对网格引用: {filename}")
    local_file = urdf_path.parent / "meshes" / parts[1]
    if not local_file.is_file():
        raise FileNotFoundError(f"官方网格缺失: {local_file} (来自 {filename})")
    return Path(parts[1]).stem, f"{MESH_DIR_RELATIVE_TO_XML}/{parts[1]}"


def _build_inertial(link_name: str, link_el: ET.Element) -> ET.Element:
    """URDF <inertial> 转 MJCF <inertial>(质量/COM/完整惯量逐字保留)。

    Args:
        link_name: 连杆名(报错定位用)。
        link_el: URDF link 元素。

    Returns:
        MJCF inertial 元素。

    Raises:
        ValueError: 连杆缺少惯量定义。
    """
    inertial_urdf = link_el.find("inertial")
    if inertial_urdf is None:
        raise ValueError(f"连杆缺少惯量: {link_name}")
    attrs = _origin_attrs(inertial_urdf.find("origin"))
    mass_el = inertial_urdf.find("mass")
    inertia_el = inertial_urdf.find("inertia")
    if mass_el is None or inertia_el is None:
        raise ValueError(f"连杆惯量字段不完整: {link_name}")
    attrs["mass"] = mass_el.attrib["value"].strip()
    # MuJoCo fullinertia 顺序:ixx iyy izz ixy ixz iyz(kg·m²,逐字保留)
    attrs["fullinertia"] = " ".join(
        inertia_el.attrib[key].strip()
        for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
    )
    return ET.Element("inertial", attrs)


def _check_collision_matches_visual(link_name: str, link_el: ET.Element) -> None:
    """校验连杆 collision 与 visual 引用同一网格(审计声明,不生成碰撞 geom)。

    Args:
        link_name: 连杆名(报错定位用)。
        link_el: URDF link 元素。

    Raises:
        ValueError: collision 与 visual 网格不一致(固定 URDF 不应出现)。
    """
    visual_meshes = [
        el.attrib["filename"] for el in link_el.findall("visual/geometry/mesh")
    ]
    collision_meshes = [
        el.attrib["filename"] for el in link_el.findall("collision/geometry/mesh")
    ]
    if visual_meshes != collision_meshes:
        raise ValueError(
            f"连杆 {link_name} 的 collision 与 visual 网格不一致: "
            f"{collision_meshes} vs {visual_meshes}"
        )


def _append_visual_geoms(
    body_el: ET.Element,
    link_name: str,
    link_el: ET.Element,
    urdf_path: Path,
    used_meshes: dict[str, str],
) -> None:
    """把 URDF visual 转 MJCF 视觉 geom 并登记 mesh(仅渲染,无碰撞)。

    Args:
        body_el: 目标 MJCF body 元素。
        link_name: 连杆名。
        link_el: URDF link 元素。
        urdf_path: URDF 文件路径。
        used_meshes: mesh 名 → 片段内相对路径(跨连杆去重登记)。

    Raises:
        ValueError: visual 缺少 mesh 定义。
    """
    for visual_el in link_el.findall("visual"):
        mesh_el = visual_el.find("geometry/mesh")
        if mesh_el is None:
            raise ValueError(f"连杆 {link_name} 的 visual 缺少 mesh")
        mesh_name, mesh_file = _resolve_mesh(mesh_el.attrib["filename"], urdf_path)
        used_meshes[mesh_name] = mesh_file
        attrs: dict[str, str] = {
            "name": f"{link_name}_visual",
            "type": "mesh",
            "mesh": mesh_name,
            "contype": "0",
            "conaffinity": "0",
        }
        attrs.update(_origin_attrs(visual_el.find("origin")))
        color_el = visual_el.find("material/color")
        if color_el is not None:
            attrs["rgba"] = " ".join(color_el.attrib["rgba"].split())
        body_el.append(ET.Element("geom", attrs))


def _arm_joint_names(side: Literal["left", "right"]) -> list[str]:
    """取选定侧 7 个臂关节名(审计 §2 语义顺序)。

    Args:
        side: "left" 或 "right"。

    Returns:
        关节名列表(父到子)。
    """
    return [f"{side}_{suffix}_joint" for suffix in _ARM_SEMANTIC_SUFFIXES]


def _child_link_of(joint_el: ET.Element) -> str:
    """取 URDF joint 的 child link 名。

    Args:
        joint_el: URDF joint 元素。

    Returns:
        child link 名。
    """
    return joint_el.find("child").attrib["link"]


def _validate_arm_chain(
    side: Literal["left", "right"],
    links: dict[str, ET.Element],
    joints: dict[str, ET.Element],
    children: dict[str, list[tuple[str, str]]],
) -> None:
    """严格校验选定侧 7 关节链与移除边界(面向固定 URDF)。

    校验内容:7 个关节均为 revolute 且带 limit/axis;链式父子关系自
    waist_yaw_link 起;每个链上 link 除下一关节外无其他子关节;
    末端 wrist_yaw_link 的子关节恰为该侧 dex1_base_joint(fixed,
    其下夹爪/相机子树整体移除)。

    Args:
        side: "left" 或 "right"。
        links: link 名 → link 元素。
        joints: joint 名 → joint 元素。
        children: 父 link 名 → [(joint 名, 子 link 名)]。

    Raises:
        ValueError: 链结构与固定版本不符。
    """
    names = _arm_joint_names(side)
    for name in names:
        if joints.get(name, ET.Element("joint")).attrib.get("type") != "revolute":
            raise ValueError(f"臂关节 {name} 缺失或非 revolute")
        if joints[name].find("limit") is None or joints[name].find("axis") is None:
            raise ValueError(f"臂关节 {name} 缺少 limit/axis")
    # 链式父子关系:waist_yaw_link → j0 → link0 → j1 → ...
    parent_link = "waist_yaw_link"
    for idx, name in enumerate(names):
        joint_el = joints[name]
        if joint_el.find("parent").attrib["link"] != parent_link:
            raise ValueError(
                f"臂关节 {name} 的父 link 应为 {parent_link}: "
                f"实际 {joint_el.find('parent').attrib['link']}"
            )
        child = _child_link_of(joint_el)
        # 链上每个 link 除链下一关节外不得有其他子关节(固定结构)
        child_joint_names = [j for j, _c in children.get(child, [])]
        if idx + 1 < len(names):
            next_joint = names[idx + 1]
            if child_joint_names != [next_joint]:
                raise ValueError(
                    f"链上连杆 {child} 的子关节应为 [{next_joint}]: "
                    f"实际 {child_joint_names}"
                )
        parent_link = child
    # 末端 wrist_yaw_link:子关节恰为该侧 dex1_base_joint(fixed),
    # 其下夹爪(prismatic ×2)与相机(fixed)子树整体移除
    wrist_yaw_link = parent_link
    wrist_children = sorted(children.get(wrist_yaw_link, []))
    expected_dex1 = f"{side}_dex1_base_joint"
    if [j for j, _c in wrist_children] != [expected_dex1]:
        raise ValueError(
            f"{wrist_yaw_link} 的子关节应为 [{expected_dex1}]: 实际 {wrist_children}"
        )
    if joints[expected_dex1].attrib.get("type") != "fixed":
        raise ValueError(f"{expected_dex1} 应为 fixed")
    # 移除子树的关节类型校验(记录于 manifest 的移除项)
    dex1_base = _child_link_of(joints[expected_dex1])
    for j, _c in children.get(dex1_base, []):
        if joints[j].attrib.get("type") not in ("prismatic", "fixed"):
            raise ValueError(f"dex1 子树出现未预期关节类型: {j}")
    # 安装 site 数据源:dex1_base_joint origin
    if joints[expected_dex1].find("origin") is None:
        raise ValueError(f"{expected_dex1} 缺少 origin(工具安装参考)")
    if wrist_yaw_link not in links:
        raise ValueError(f"链末端连杆缺失: {wrist_yaw_link}")


def _arm_header_comment(side: Literal["left", "right"]) -> ET.Comment:
    """构造裸臂片段的头注释(来源与契约说明,供人工审计)。"""
    text = (
        f"自动生成文件,勿手改。重建:python scripts/tools/build_r1d_model.py\n"
        f"来源:{PINNED_REPO_URL} commit {PINNED_COMMIT}\n"
        f"      robots/r1d_description/r1d.urdf(SHA256 {PINNED_URDF_SHA256})\n"
        f"侧别:side={side}(同一工具对同一 URDF 仅以 side 参数区分生成,\n"
        f"      独立导出,非单侧生成后镜像改名)\n"
        "契约(阶段 2 共享约束 N0):\n"
        "  1) <mujocoinclude> 片段:仅 compiler + worldbody(根 body=r1d_base),\n"
        "     无地面/球拍/执行器/世界位姿;场景组装归 Task 2.5b。\n"
        f"  2) {ROOT_BODY_NAME} 为 waist_yaw_link 之上的虚拟固定安装点\n"
        "     (无质量/无几何);臂片段内第一级 body 的 pos/quat 即 URDF\n"
        "     该侧 shoulder_pitch_joint origin(含肩部 ±0.26187 rad 安装 rpy,\n"
        "     零位为 Z 形偏置,非垂直下垂)。\n"
        "  3) 移除项:dex1 夹爪(prismatic ×2)与腕部相机固定子树,\n"
        f"     记录见 assets/unitree_r1d/source_manifest.yaml;\n"
        f"     {FLANGE_SITE_NAME} site 位置 = 该侧 dex1_base_joint origin\n"
        "     (工具安装参考,不声称等于实物安装面)。\n"
        "  4) 限位单位:compiler angle=degree,URDF rad 已换算写出;\n"
        "     逐字采用 URDF 值(含右腕俯仰上界 +0.5326 的官方不对称,\n"
        "     不得修正);tests/test_r1d_assets.py 编译后对照 rad 值。\n"
        "  5) URDF effort/velocity 非 0(官方声明,未经实机验证):\n"
        "     不设 ctrlrange/forcerange/速度限值,不作安全限值(N0-9)。\n"
        "  6) 惯量逐字保留 URDF 质量/COM/完整张量,无 balanceinertia 修补。\n"
        "  7) 几何仅视觉 geom(contype=0 conaffinity=0,自碰撞关闭约定);\n"
        "     URDF collision 与 visual 同网格,碰撞几何由后续场景模型决定。\n"
        "  8) URDF 未提供关节阻尼/armature,不虚构数值。"
    )
    return ET.Comment(text)


def _env_header_comment(q_L: float) -> ET.Comment:
    """构造环境片段的头注释(冻结链与 bake 说明,供人工审计)。"""
    text = (
        f"自动生成文件,勿手改。重建:python scripts/tools/build_r1d_model.py\n"
        f"来源:{PINNED_REPO_URL} commit {PINNED_COMMIT}\n"
        f"      robots/r1d_description/r1d.urdf(SHA256 {PINNED_URDF_SHA256})\n"
        "契约(阶段 2 共享约束 N0-3(b)):\n"
        "  1) 冻结环境链:base_link → lift_mount →(stage1@q_L, stage2@q_L)\n"
        "     → pelvis → waist_yaw@0;无任何关节(升降/腰冻结,场景 nq 不受影响)。\n"
        "  2) 双级升降按 mimic multiplier=1 的同一 q_L bake 为固定偏移:\n"
        f"     当前 q_L = {_fmt(q_L)} m(单级行程 [0, 0.425],总行程 2×0.425);\n"
        "     MuJoCo 的 URDF 导入不保留 mimic,升降冻结故无运行时等式约束。\n"
        "  3) waist_yaw_link 几何保留(臂挂接点);轮/脚轮/头/对侧臂\n"
        "     不进入本片段(排除清单见 source_manifest.yaml)。\n"
        "  4) 惯量/网格逐字来自 URDF;仅视觉 geom(contype=0 conaffinity=0)。\n"
        "  5) compiler angle=degree(本片段无角度敏感内容,声明与臂片段一致)。"
    )
    return ET.Comment(text)


def _assets_header_comment() -> ET.Comment:
    """构造资产片段的头注释(mesh 映射说明,供人工审计)。"""
    text = (
        "自动生成文件,勿手改。重建:python scripts/tools/build_r1d_model.py\n"
        "mesh 声明:URDF 相对引用 meshes/*.STL 映射到本地\n"
        f"{MESH_DIR_RELATIVE_TO_XML}/*.STL(相对片段部署位置 src/robot/,\n"
        "沿用 RM65/D2 模型相对路径参照模式)。\n"
        "覆盖:双侧臂(7×2)+ 环境(6)共 20 个 mesh,支持任一侧场景组装;\n"
        "其余官方网格(夹爪/相机/轮/脚轮/头)未在派生模型中引用,\n"
        "仅作资产留存(见 assets/unitree_r1d/source_manifest.yaml)。\n"
        "本片段仅含 <asset>,无 compiler(角度敏感内容在臂/环境片段)。"
    )
    return ET.Comment(text)


def _write_fragment(root: ET.Element, path: Path) -> None:
    """缩进并写出 <mujocoinclude> 片段(utf-8、LF 行尾、字节确定)。

    以字节方式写出,绕过文本模式的平台换行转换(Windows 下 ET.write
    会把 LF 改写为 CRLF),保证任意平台生成字节一致。

    Args:
        root: 片段根元素(<mujocoinclude>)。
        path: 输出文件路径(父目录自动创建)。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    path.write_bytes(xml_bytes + b"\n")


def build_r1d_arm(
    urdf_path: Path,
    side: Literal["left", "right"],
    output_path: Path,
) -> None:
    """从固定 commit 的 R1D 整机 URDF 导出选定侧裸臂 MJCF 片段。

    提取 waist_yaw_link 之下该侧 7 个 revolute 关节链,产出
    ``<mujocoinclude>`` 片段:根 body ``r1d_base``(虚拟固定安装点)、
    完整关节/连杆层级与惯量/网格引用(逐字来自 URDF);wrist_yaw_link
    下建 ``r1d_flange`` site(位置 = 该侧 dex1_base_joint origin)。
    生成过程完全确定:相同输入产生字节一致的输出。

    Args:
        urdf_path: 固定 commit 的官方 URDF 路径(内容哈希须匹配)。
        side: "left" 或 "right"。
        output_path: 裸臂片段输出路径(仓库交付物为
            src/robot/r1d_arm_{side}.xml)。

    Raises:
        FileNotFoundError: URDF 或其引用的网格文件缺失。
        ValueError: URDF 哈希/链结构与固定版本不符。
    """
    if side not in ("left", "right"):
        raise ValueError(f"side 必须为 'left' 或 'right': {side}")
    robot_el = _parse_pinned_urdf(urdf_path)
    links, joints, children = _collect_structure(robot_el)
    _validate_arm_chain(side, links, joints, children)
    used_meshes: dict[str, str] = {}

    arm_root = ET.Element("mujocoinclude")
    arm_root.append(_arm_header_comment(side))
    arm_root.append(ET.Element("compiler", {"angle": "degree"}))
    worldbody = ET.SubElement(arm_root, "worldbody")
    root_body = ET.Element("body", {"name": ROOT_BODY_NAME})

    parent_el = root_body
    for joint_name in _arm_joint_names(side):
        joint_el = joints[joint_name]
        child_link = _child_link_of(joint_el)
        limit_el = joint_el.find("limit")
        axis_el = joint_el.find("axis")
        body_attrs: dict[str, str] = {"name": child_link}
        body_attrs.update(_origin_attrs(joint_el.find("origin")))
        body_el = ET.SubElement(parent_el, "body", body_attrs)
        # URDF rad 限位 → degree 写出(compiler angle="degree")
        lower_deg = math.degrees(float(limit_el.attrib["lower"]))
        upper_deg = math.degrees(float(limit_el.attrib["upper"]))
        ET.SubElement(
            body_el,
            "joint",
            {
                "name": joint_name,
                "type": "hinge",
                "axis": " ".join(axis_el.attrib["xyz"].split()),
                "range": f"{_fmt(lower_deg)} {_fmt(upper_deg)}",
            },
        )
        child_link_el = links[child_link]
        body_el.append(_build_inertial(child_link, child_link_el))
        _check_collision_matches_visual(child_link, child_link_el)
        _append_visual_geoms(body_el, child_link, child_link_el, urdf_path, used_meshes)
        parent_el = body_el

    # 末端 wrist_yaw_link 下:r1d_flange site = 该侧 dex1_base_joint origin
    dex1_origin = joints[f"{side}_dex1_base_joint"].find("origin")
    flange_xyz = " ".join(dex1_origin.attrib["xyz"].split())
    ET.SubElement(
        parent_el,
        "site",
        {"name": FLANGE_SITE_NAME, "pos": flange_xyz, "size": "0.005"},
    )
    worldbody.append(root_body)
    _write_fragment(arm_root, output_path)


def _validate_env_chain(
    joints: dict[str, ET.Element], children: dict[str, list[tuple[str, str]]]
) -> None:
    """严格校验环境冻结链结构与 mimic 关系(面向固定 URDF)。

    校验内容:链上关节类型(fixed/prismatic/revolute)、两级 prismatic
    的 z 轴与 [0, 0.425] 行程、stage2 的 mimic multiplier=1 offset=0、
    waist_yaw_joint 的 z 轴与零 origin;base_link 的其余子关节均为
    continuous(轮/脚轮);waist_yaw_link 的其余子关节为对侧肩俯仰与
    头部俯仰(均不进入环境片段)。

    Args:
        joints: joint 名 → joint 元素。
        children: 父 link 名 → [(joint 名, 子 link 名)]。

    Raises:
        ValueError: 链结构与固定版本不符。
    """
    expected_types = {
        "lift_mount_joint": "fixed",
        "lift_stage1_joint": "prismatic",
        "lift_stage2_joint": "prismatic",
        "upper_body_mount_joint": "fixed",
        "waist_yaw_joint": "revolute",
    }
    for name, jtype in expected_types.items():
        if joints.get(name, ET.Element("joint")).attrib.get("type") != jtype:
            raise ValueError(f"环境链关节 {name} 缺失或非 {jtype}")
    for name in _ENV_PRISMATIC_JOINTS:
        joint_el = joints[name]
        axis = _floats(joint_el.find("axis").attrib["xyz"])
        if axis != [0.0, 0.0, 1.0]:
            raise ValueError(f"{name} 轴向应为 z: {axis}")
        limit = joint_el.find("limit")
        if _floats(limit.attrib["lower"])[0] != LIFT_TRAVEL_RANGE_M[0] or _floats(
            limit.attrib["upper"]
        )[0] != LIFT_TRAVEL_RANGE_M[1]:
            raise ValueError(f"{name} 行程应为 {LIFT_TRAVEL_RANGE_M} m")
    mimic = joints["lift_stage2_joint"].find("mimic")
    if mimic is None:
        raise ValueError("lift_stage2_joint 缺少 mimic(固定 URDF 应有)")
    if (
        mimic.attrib.get("joint") != "lift_stage1_joint"
        or mimic.attrib.get("multiplier") != "1"
        or mimic.attrib.get("offset") != "0"
    ):
        raise ValueError(
            f"lift_stage2_joint mimic 应为 stage1 multiplier=1 offset=0: "
            f"{mimic.attrib}"
        )
    waist = joints["waist_yaw_joint"]
    if _floats(waist.find("axis").attrib["xyz"]) != [0.0, 0.0, 1.0]:
        raise ValueError("waist_yaw_joint 轴向应为 z")
    if _floats(waist.find("origin").attrib["xyz"]) != [0.0, 0.0, 0.0]:
        raise ValueError("waist_yaw_joint origin 应为 0(冻结 0 的前提)")
    # 链式父子关系
    parent_link = _ENV_ROOT_LINK
    for name, child in _ENV_CHAIN:
        if joints[name].find("parent").attrib["link"] != parent_link:
            raise ValueError(f"环境链断裂: {name} 的父 link 应为 {parent_link}")
        if _child_link_of(joints[name]) != child:
            raise ValueError(f"环境链断裂: {name} 的子 link 应为 {child}")
        parent_link = child
    # base_link 其余子关节均为 continuous(轮/脚轮,不进环境片段)
    for name, _c in children.get(_ENV_ROOT_LINK, []):
        if name != "lift_mount_joint" and joints[name].attrib.get("type") != "continuous":
            raise ValueError(f"base_link 出现未预期子关节类型: {name}")
    # waist_yaw_link 其余子关节 = 对侧/同侧肩俯仰 + head_pitch(不进环境片段)
    waist_children = {j for j, _c in children.get("waist_yaw_link", [])}
    expected_extra = {
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "head_pitch_joint",
    }
    if waist_children != expected_extra:
        raise ValueError(
            f"waist_yaw_link 子关节应为 {sorted(expected_extra)}: "
            f"实际 {sorted(waist_children)}"
        )


def build_r1d_env(
    urdf_path: Path, output_path: Path, q_L: float = 0.0
) -> None:
    """从固定 commit 的 R1D 整机 URDF 导出冻结环境 MJCF 片段。

    导出 base_link → lift_mount →(stage1@q_L, stage2@q_L)→ pelvis →
    waist_yaw@0 的冻结链,无任何关节:两级 prismatic 按同一 q_L bake 为
    固定 pos 偏移(mimic multiplier=1)。生成过程完全确定。

    Args:
        urdf_path: 固定 commit 的官方 URDF 路径(内容哈希须匹配)。
        output_path: 环境片段输出路径(仓库交付物为 src/robot/r1d_env.xml)。
        q_L: 升降 bake 值 q_L (m),默认 0;须在 [0, 0.425] 内。

    Raises:
        FileNotFoundError: URDF 或其引用的网格文件缺失。
        ValueError: URDF 哈希/链结构不符,或 q_L 超出行程范围。
    """
    if not (LIFT_TRAVEL_RANGE_M[0] <= q_L <= LIFT_TRAVEL_RANGE_M[1]):
        raise ValueError(
            f"q_L 超出单级行程范围 {LIFT_TRAVEL_RANGE_M}: {q_L}"
        )
    robot_el = _parse_pinned_urdf(urdf_path)
    links, joints, children = _collect_structure(robot_el)
    _validate_env_chain(joints, children)
    used_meshes: dict[str, str] = {}

    env_root = ET.Element("mujocoinclude")
    env_root.append(_env_header_comment(q_L))
    env_root.append(ET.Element("compiler", {"angle": "degree"}))
    worldbody = ET.SubElement(env_root, "worldbody")

    def _append_link_body(parent_el: ET.Element, link_name: str, joint_name: str) -> ET.Element:
        """追加一个冻结链 body:pos 来自 joint origin(z 已 bake),无关节。"""
        joint_el = joints[joint_name]
        attrs: dict[str, str] = {"name": link_name}
        if joint_el.attrib.get("type") == "prismatic":
            attrs.update(_origin_attrs_with_z_bake(joint_el.find("origin"), q_L))
        else:
            attrs.update(_origin_attrs(joint_el.find("origin")))
        body_el = ET.SubElement(parent_el, "body", attrs)
        link_el = links[link_name]
        body_el.append(_build_inertial(link_name, link_el))
        _check_collision_matches_visual(link_name, link_el)
        _append_visual_geoms(body_el, link_name, link_el, urdf_path, used_meshes)
        return body_el

    base_el = ET.SubElement(worldbody, "body", {"name": _ENV_ROOT_LINK})
    base_link_el = links[_ENV_ROOT_LINK]
    base_el.append(_build_inertial(_ENV_ROOT_LINK, base_link_el))
    _check_collision_matches_visual(_ENV_ROOT_LINK, base_link_el)
    _append_visual_geoms(base_el, _ENV_ROOT_LINK, base_link_el, urdf_path, used_meshes)

    parent_el = base_el
    for joint_name, child_link in _ENV_CHAIN:
        parent_el = _append_link_body(parent_el, child_link, joint_name)

    _write_fragment(env_root, output_path)


def build_r1d_assets(urdf_path: Path, output_path: Path) -> None:
    """生成双侧臂 + 环境引用的全部 mesh 声明片段。

    覆盖左右臂各 7 个 + 环境 6 个共 20 个 mesh(支持任一侧场景组装),
    mesh 名按字节序排列保证输出确定;逐文件校验本地存在性。

    Args:
        urdf_path: 固定 commit 的官方 URDF 路径(内容哈希须匹配)。
        output_path: 资产片段输出路径(仓库交付物为 src/robot/r1d_assets.xml)。

    Raises:
        FileNotFoundError: URDF 或任一引用网格缺失。
        ValueError: URDF 哈希/结构与固定版本不符。
    """
    robot_el = _parse_pinned_urdf(urdf_path)
    links, joints, children = _collect_structure(robot_el)
    # 双侧臂链与环境链的引用几何(与片段生成同一遍历逻辑,防口径漂移)。
    # 传入临时 body 容器仅为复用 _append_visual_geoms 的 mesh 登记逻辑,
    # 容器本身丢弃。
    used_meshes: dict[str, str] = {}
    for side in ("left", "right"):
        _validate_arm_chain(side, links, joints, children)
        for joint_name in _arm_joint_names(side):
            child_link = _child_link_of(joints[joint_name])
            _append_visual_geoms(
                ET.Element("body"),
                child_link,
                links[child_link],
                urdf_path,
                used_meshes,
            )
    _validate_env_chain(joints, children)
    for link_name in (
        _ENV_ROOT_LINK,
        "lift_mount_link",
        "lift_stage1_link",
        "lift_stage2_link",
        "pelvis_link",
        "waist_yaw_link",
    ):
        _append_visual_geoms(
            ET.Element("body"), link_name, links[link_name], urdf_path, used_meshes
        )

    assets_root = ET.Element("mujocoinclude")
    assets_root.append(_assets_header_comment())
    asset_el = ET.SubElement(assets_root, "asset")
    for mesh_name in sorted(used_meshes):
        asset_el.append(ET.Comment(f"meshes/{mesh_name}.STL"))
        asset_el.append(
            ET.Element("mesh", {"name": mesh_name, "file": used_meshes[mesh_name]})
        )
    _write_fragment(assets_root, output_path)


def _repo_root() -> Path:
    """取仓库根目录(本脚本位于 scripts/tools/,上溯两级)。

    Returns:
        仓库根目录路径。
    """
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """命令行入口:按参数(或默认仓库路径)重建四份片段。

    Args:
        argv: 命令行参数(None 时取 sys.argv[1:])。

    Returns:
        进程退出码(0 成功)。
    """
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="从固定 commit 的 R1D 整机 URDF 导出裸臂/环境/资产 MJCF 片段"
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=root / "assets" / "unitree_r1d" / "r1d.urdf",
        help="固定 commit 的官方 URDF 路径(默认仓库交付物)",
    )
    parser.add_argument(
        "--arm-left",
        type=Path,
        default=root / "src" / "robot" / "r1d_arm_left.xml",
        help="左臂片段输出路径",
    )
    parser.add_argument(
        "--arm-right",
        type=Path,
        default=root / "src" / "robot" / "r1d_arm_right.xml",
        help="右臂片段输出路径",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=root / "src" / "robot" / "r1d_env.xml",
        help="环境片段输出路径",
    )
    parser.add_argument(
        "--assets-out",
        type=Path,
        default=root / "src" / "robot" / "r1d_assets.xml",
        help="资产片段输出路径",
    )
    parser.add_argument(
        "--qL",
        type=float,
        default=0.0,
        help="升降 bake 值 q_L (m),默认 0(范围 [0, 0.425])",
    )
    args = parser.parse_args(argv)
    build_r1d_arm(args.urdf, "left", args.arm_left)
    build_r1d_arm(args.urdf, "right", args.arm_right)
    build_r1d_env(args.urdf, args.env, q_L=args.qL)
    build_r1d_assets(args.urdf, args.assets_out)
    print(f"已生成左臂片段: {args.arm_left}")
    print(f"已生成右臂片段: {args.arm_right}")
    print(f"已生成环境片段: {args.env} (q_L={args.qL})")
    print(f"已生成资产片段: {args.assets_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
