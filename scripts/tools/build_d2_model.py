"""D2 裸臂 MJCF 片段生成器（阶段 2 · Task 2.1）。

从固定 commit（8d505a8c511f2a8dd401bd4c410e37121b1bfd17）的宇树 D2-Arm
官方 URDF 生成两份 ``<mujocoinclude>`` 片段：

- ``src/robot/d2_arm.xml``：完整关节/连杆层级。根 body 固定为 ``d2_base``
  （对应 URDF ``base_link``，按阶段 2 共享契约重命名），七个铰链关节
  ``joint_1..joint_7``；joint_8/link_8 固定末端附件按契约移除。
- ``src/robot/d2_assets.xml``：仅含 ``<asset>`` 视觉 mesh 声明。

两份片段均不含地面、球拍、执行器与世界位姿；场景组装归 Task 2.5
（write_d2_model_template 用 ElementTree 合并本产物）。

转换忠实性约定：

1. 保留 URDF 关节 origin/axis、连杆质量、COM 与完整惯量张量（fullinertia），
   不做 balanceinertia 等任何修补；惯量不满足编译要求时直接报错并给出证据。
2. 关节限位由 rad 转为 degree 写出（片段声明 ``compiler angle="degree"``），
   测试在编译后对照 rad 值（tests/test_d2_assets.py）。
3. URDF 的 velocity/effort 全为 0，表示官方未提供可用能力值：不生成
   ctrlrange/forcerange/速度限值，未知能力记录于 source_manifest.yaml。
4. 仅生成视觉 geom（contype=0/conaffinity=0，遵循 AGENTS.md 自碰撞关闭
   约定）；URDF collision 与 visual 引用同网格，碰撞几何由后续场景/
   诊断模型决定。
5. URDF 未提供关节阻尼/armature，不虚构数值。

用法::

    python scripts/tools/build_d2_model.py                # 按默认路径重建交付物
    python scripts/tools/build_d2_model.py --urdf X --output Y --assets-out Z
"""

from __future__ import annotations

import argparse
import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

# 阶段 0 固定的官方来源（与 stage0-evidence 交叉验证一致）
PINNED_REPO_URL = "https://github.com/unitreerobotics/D2-Arm"
PINNED_COMMIT = "8d505a8c511f2a8dd401bd4c410e37121b1bfd17"
PINNED_URDF_SHA256 = (
    "6f508a49719d8f7f02af284f7ecf4d6312c9cca17fd8251951a8bba232cbec73"
)

# 片段内网格文件的固定相对前缀：自片段部署位置 src/robot/ 指向
# assets/unitree_d2/meshes/，沿用 RM65 模型（file="../../assets/rm_65/
# urdf/meshes/*.STL"）的参照模式。前缀与输出位置无关，保证重建字节确定。
MESH_DIR_RELATIVE_TO_XML = "../../assets/unitree_d2/meshes"

# 派生模型契约：根 body 名称（对应 URDF base_link，按共享契约重命名）
ROOT_BODY_NAME = "d2_base"
ROOT_LINK_NAME = "base_link"

# 派生模型移除的固定关节及其子链（原始 URDF 保留，清单记录）
REMOVED_FIXED_JOINT = "joint_8"
REMOVED_LINK = "link_8"

# URDF 包 URI 前缀（面向固定 URDF，仅接受 D2_URDF 包）
_PACKAGE_PREFIX = "package://D2_URDF/"

# 预期 URDF 结构（面向固定版本的严格断言）
_EXPECTED_REVOLUTE_JOINTS = tuple(f"joint_{i}" for i in range(1, 8))
_EXPECTED_LINKS = frozenset({ROOT_LINK_NAME} | {f"link_{i}" for i in range(1, 9)})


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
        text: URDF 属性中的数值串（如 "0 0 0.071"）。

    Returns:
        解析后的 float 列表。
    """
    return [float(token) for token in text.split()]


def _fmt(value: float) -> str:
    """以 10 位有效数字格式化数值。

    用于 rad→degree 转换结果的写出；10 位有效数字使往返误差 < 1e-9 rad。

    Args:
        value: 待格式化的数值。

    Returns:
        格式化后的字符串。
    """
    return format(value, ".10g")


def _rpy_to_quat_wxyz(rpy: tuple[float, float, float]) -> tuple[float, ...]:
    """URDF 固定轴 rpy（弧度，绕父系 X-Y-Z 旋转）转 MuJoCo wxyz 四元数。

    Args:
        rpy: (roll, pitch, yaw)，单位 rad。

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
    """URDF <origin> 转 MJCF pos/quat 属性（xyz 数值串逐字保留）。

    零平移/零旋转省略属性；rpy 非零时换算为 wxyz 四元数。

    Args:
        origin_el: URDF origin 元素（可为 None，视为恒等变换）。

    Returns:
        属性字典（可能为空）。
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


def _parse_pinned_urdf(urdf_path: Path) -> ET.Element:
    """校验并解析固定 commit 的官方 URDF。

    Args:
        urdf_path: URDF 文件路径（内容须与 PINNED_URDF_SHA256 逐字节一致）。

    Returns:
        URDF 根元素（<robot>）。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 内容哈希与固定 commit 不符（转换器仅面向该版本）。
    """
    digest = _sha256_of(urdf_path)
    if digest != PINNED_URDF_SHA256:
        raise ValueError(
            f"URDF 哈希与固定 commit {PINNED_COMMIT[:12]} 不符: {urdf_path} "
            f"(实际 {digest}, 期望 {PINNED_URDF_SHA256})"
        )
    return ET.parse(urdf_path).getroot()


def _collect_chain(robot_el: ET.Element) -> tuple[dict[str, ET.Element], dict[str, ET.Element], dict[str, list[tuple[str, str]]], str]:
    """收集并严格校验 URDF 运动链结构。

    面向固定 URDF：恰好七个 revolute 关节 joint_1..joint_7、一个固定关节
    joint_8（子链 link_8 待移除）、连杆集合 base_link + link_1..link_8、
    唯一根 base_link。任何偏离立即报错，杜绝静默容错。

    Args:
        robot_el: URDF 根元素。

    Returns:
        (link 名 → link 元素, joint 名 → joint 元素,
         父 link 名 → [(joint 名, 子 link 名)], 根 link 名)。

    Raises:
        ValueError: 结构与固定版本不符。
    """
    links = {el.attrib["name"]: el for el in robot_el.findall("link")}
    joints = {el.attrib["name"]: el for el in robot_el.findall("joint")}
    if set(links) != set(_EXPECTED_LINKS):
        raise ValueError(f"连杆集合与固定 URDF 不符: {sorted(links)}")
    for name in _EXPECTED_REVOLUTE_JOINTS:
        if joints.get(name, ET.Element("joint")).attrib.get("type") != "revolute":
            raise ValueError(f"关节 {name} 缺失或非 revolute")
    if joints[REMOVED_FIXED_JOINT].attrib.get("type") != "fixed":
        raise ValueError(f"关节 {REMOVED_FIXED_JOINT} 应为 fixed")
    if set(joints) != set(_EXPECTED_REVOLUTE_JOINTS) | {REMOVED_FIXED_JOINT}:
        raise ValueError(f"关节集合与固定 URDF 不符: {sorted(joints)}")

    children: dict[str, list[tuple[str, str]]] = {}
    child_links: set[str] = set()
    for name, joint_el in joints.items():
        parent = joint_el.find("parent").attrib["link"]
        child = joint_el.find("child").attrib["link"]
        children.setdefault(parent, []).append((name, child))
        child_links.add(child)
    roots = [name for name in links if name not in child_links]
    if roots != [ROOT_LINK_NAME]:
        raise ValueError(f"根连杆应为唯一的 {ROOT_LINK_NAME}: {roots}")
    return links, joints, children, roots[0]


def _resolve_mesh(filename: str, urdf_path: Path) -> tuple[str, str]:
    """解析 package:// URI 到本地网格，返回 (mesh 名, 片段内相对路径)。

    官方包根即 URDF 所在目录（assets/unitree_d2/），网格位于 meshes/
    子目录，与上游仓库布局一致；逐文件校验存在性。

    Args:
        filename: URDF mesh filename 属性（package:// URI）。
        urdf_path: URDF 文件路径。

    Returns:
        (mesh 名（STL 主文件名）, 片段内相对文件路径)。

    Raises:
        FileNotFoundError: 本地资产目录中缺少网格文件。
        ValueError: 非 D2_URDF 包 URI。
    """
    if not filename.startswith(_PACKAGE_PREFIX):
        raise ValueError(f"非 D2_URDF 包 URI: {filename}")
    rel = filename[len(_PACKAGE_PREFIX) :]  # 形如 meshes/link_N.STL
    local_file = urdf_path.parent.joinpath(*rel.split("/"))
    if not local_file.is_file():
        raise FileNotFoundError(f"官方网格缺失: {local_file} (来自 {filename})")
    mesh_name = Path(rel).stem
    # URI 的 meshes/ 组件即本地资产 meshes/ 目录，避免路径重复拼接
    return mesh_name, f"{MESH_DIR_RELATIVE_TO_XML}/{Path(rel).name}"


def _build_inertial(link_name: str, link_el: ET.Element) -> ET.Element:
    """URDF <inertial> 转 MJCF <inertial>（质量/COM/完整惯量逐字保留）。

    Args:
        link_name: 连杆名（报错定位用）。
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
    # MuJoCo fullinertia 顺序：ixx iyy izz ixy ixz iyz（kg·m²，逐字保留）
    attrs["fullinertia"] = " ".join(
        inertia_el.attrib[key].strip()
        for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
    )
    return ET.Element("inertial", attrs)


def _check_collision_matches_visual(link_name: str, link_el: ET.Element) -> None:
    """校验连杆 collision 与 visual 引用同一网格（审计声明，不生成碰撞 geom）。

    Args:
        link_name: 连杆名（报错定位用）。
        link_el: URDF link 元素。

    Raises:
        ValueError: collision 与 visual 网格不一致（固定 URDF 不应出现）。
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
    """把 URDF visual 转 MJCF 视觉 geom 并登记 mesh（仅渲染，无碰撞）。

    Args:
        body_el: 目标 MJCF body 元素。
        link_name: 连杆名。
        link_el: URDF link 元素。
        urdf_path: URDF 文件路径。
        used_meshes: mesh 名 → 片段内相对路径（跨连杆去重登记）。

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


def _append_child_bodies(
    parent_el: ET.Element,
    link_name: str,
    links: dict[str, ET.Element],
    joints: dict[str, ET.Element],
    children: dict[str, list[tuple[str, str]]],
    urdf_path: Path,
    used_meshes: dict[str, str],
) -> None:
    """递归追加子 body：revolute 关节生成铰链，固定子链按契约移除。

    Args:
        parent_el: 父 MJCF body 元素。
        link_name: 当前 URDF 连杆名。
        links: link 名 → link 元素映射。
        joints: joint 名 → joint 元素映射。
        children: 父 link 名 → [(joint 名, 子 link 名)] 映射。
        urdf_path: URDF 文件路径。
        used_meshes: mesh 名 → 片段内相对路径（跨连杆去重登记）。

    Raises:
        ValueError: 出现契约外的关节类型或限位缺失。
    """
    for joint_name, child_link in children.get(link_name, []):
        joint_el = joints[joint_name]
        if joint_el.attrib.get("type") != "revolute":
            # 契约移除项：joint_8 固定关节及其子链（link_8 末端附件）
            if joint_name != REMOVED_FIXED_JOINT or child_link != REMOVED_LINK:
                raise ValueError(f"未预期的非 revolute 关节: {joint_name}")
            continue
        limit_el = joint_el.find("limit")
        axis_el = joint_el.find("axis")
        if limit_el is None or axis_el is None:
            raise ValueError(f"关节 {joint_name} 缺少 limit/axis")
        body_attrs = {"name": child_link}
        body_attrs.update(_origin_attrs(joint_el.find("origin")))
        body_el = ET.SubElement(parent_el, "body", body_attrs)
        # URDF rad 限位 → degree 写出（compiler angle="degree"）
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
        _append_child_bodies(
            body_el, child_link, links, joints, children, urdf_path, used_meshes
        )


def _arm_header_comment() -> ET.Comment:
    """构造裸臂片段的头注释（来源与契约说明，供人工审计）。"""
    text = (
        f"自动生成文件，勿手改。重建：python scripts/tools/build_d2_model.py\n"
        f"来源：{PINNED_REPO_URL} commit {PINNED_COMMIT}\n"
        f"      urdf/urdf/D2_URDF.urdf（SHA256 {PINNED_URDF_SHA256}）\n"
        "契约（阶段 2 共享约束）：\n"
        "  1) <mujocoinclude> 片段：仅 compiler + worldbody（根 body=d2_base），\n"
        "     无地面/球拍/执行器/世界位姿；场景组装归 Task 2.5。\n"
        f"  2) {ROOT_BODY_NAME} 对应 URDF {ROOT_LINK_NAME}（按契约重命名的固定根 body）。\n"
        f"  3) {REMOVED_FIXED_JOINT}（固定）与 {REMOVED_LINK}（末端附件，0.01 kg）已移除，\n"
        "     原始 URDF 保留该段；工具安装面为 joint_8 的 origin\n"
        "     （link_7 系 xyz=0.1888 0 0.024），d2_flange site 由场景组装添加。\n"
        "  4) 限位单位：compiler angle=degree，URDF rad 已换算写出；\n"
        "     tests/test_d2_assets.py 在编译后对照 rad 值。\n"
        "  5) URDF velocity/effort=0 表示官方未提供能力值：本片段及其派生\n"
        "     场景不设 ctrlrange/forcerange/速度限值（未知能力见 source_manifest）。\n"
        "  6) 惯量逐字保留 URDF 质量/COM/完整张量，无 balanceinertia 修补。\n"
        "  7) 几何仅视觉 geom（contype=0 conaffinity=0，自碰撞关闭约定）；\n"
        "     URDF collision 与 visual 同网格，碰撞几何由后续场景模型决定。\n"
        "  8) URDF 未提供关节阻尼/armature，不虚构数值。"
    )
    return ET.Comment(text)


def _assets_header_comment() -> ET.Comment:
    """构造资产片段的头注释（mesh 映射说明，供人工审计）。"""
    text = (
        "自动生成文件，勿手改。重建：python scripts/tools/build_d2_model.py\n"
        "视觉 mesh 声明：package://D2_URDF/meshes/*.STL 映射到本地\n"
        f"{MESH_DIR_RELATIVE_TO_XML}/*.STL（相对片段部署位置 src/robot/，\n"
        "沿用 RM65 模型相对路径参照模式）。\n"
        "本片段仅含 <asset>，无 compiler（角度敏感内容在 d2_arm.xml）；\n"
        f"{REMOVED_LINK} 网格未在派生模型中引用，仅作资产留存（见清单）。"
    )
    return ET.Comment(text)


def _write_fragment(root: ET.Element, path: Path) -> None:
    """缩进并写出 <mujocoinclude> 片段（utf-8、LF 行尾、字节确定）。

    以字节方式写出，绕过文本模式的平台换行转换（Windows 下 ET.write
    会把 LF 改写为 CRLF），保证任意平台生成字节一致。

    Args:
        root: 片段根元素（<mujocoinclude>）。
        path: 输出文件路径（父目录自动创建）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    path.write_bytes(xml_bytes + b"\n")


def build_d2_arm(urdf_path: Path, output_path: Path, assets_path: Path) -> None:
    """从固定 commit 的 D2 官方 URDF 生成裸臂 MJCF 片段。

    生成两份 ``<mujocoinclude>`` 片段：output_path（关节/连杆层级，根 body
    为 d2_base）与 assets_path（视觉 mesh 资产声明）。生成过程完全确定：
    相同输入产生字节一致的输出（见 test_rebuild_is_deterministic）。

    Args:
        urdf_path: 固定 commit 的官方 URDF 路径（内容哈希须匹配）。
        output_path: 裸臂片段输出路径（仓库交付物为 src/robot/d2_arm.xml）。
        assets_path: 视觉资产片段输出路径（仓库交付物为 src/robot/d2_assets.xml）。

    Raises:
        FileNotFoundError: URDF 或其引用的网格文件缺失。
        ValueError: URDF 哈希/结构与固定版本不符。
    """
    robot_el = _parse_pinned_urdf(urdf_path)
    links, joints, children, root_link = _collect_chain(robot_el)
    used_meshes: dict[str, str] = {}

    # 裸臂片段：compiler（degree）+ worldbody（根 body d2_base + 完整层级）
    arm_root = ET.Element("mujocoinclude")
    arm_root.append(_arm_header_comment())
    arm_root.append(ET.Element("compiler", {"angle": "degree"}))
    worldbody = ET.SubElement(arm_root, "worldbody")
    root_body = ET.Element("body", {"name": ROOT_BODY_NAME})
    root_link_el = links[root_link]
    root_body.append(_build_inertial(root_link, root_link_el))
    _check_collision_matches_visual(root_link, root_link_el)
    _append_visual_geoms(root_body, root_link, root_link_el, urdf_path, used_meshes)
    _append_child_bodies(
        root_body, root_link, links, joints, children, urdf_path, used_meshes
    )
    worldbody.append(root_body)

    # 资产片段：仅 <asset>（mesh 名按字节序排列，保证输出确定）
    assets_root = ET.Element("mujocoinclude")
    assets_root.append(_assets_header_comment())
    asset_el = ET.SubElement(assets_root, "asset")
    for mesh_name in sorted(used_meshes):
        asset_el.append(
            ET.Comment(f"package://D2_URDF/meshes/{mesh_name}.STL")
        )
        asset_el.append(
            ET.Element("mesh", {"name": mesh_name, "file": used_meshes[mesh_name]})
        )

    _write_fragment(arm_root, output_path)
    _write_fragment(assets_root, assets_path)


def _repo_root() -> Path:
    """取仓库根目录（本脚本位于 scripts/tools/，上溯两级）。

    Returns:
        仓库根目录路径。
    """
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    """命令行入口：按参数（或默认仓库路径）重建两份片段。

    Args:
        argv: 命令行参数（None 时取 sys.argv[1:]）。

    Returns:
        进程退出码（0 成功）。
    """
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="从固定 commit 的官方 URDF 生成 D2 裸臂 MJCF 片段"
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=root / "assets" / "unitree_d2" / "D2_URDF.urdf",
        help="固定 commit 的官方 URDF 路径（默认仓库交付物）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "src" / "robot" / "d2_arm.xml",
        help="裸臂片段输出路径",
    )
    parser.add_argument(
        "--assets-out",
        type=Path,
        default=root / "src" / "robot" / "d2_assets.xml",
        help="视觉资产片段输出路径",
    )
    args = parser.parse_args(argv)
    build_d2_arm(args.urdf, args.output, args.assets_out)
    print(f"已生成裸臂片段: {args.output}")
    print(f"已生成资产片段: {args.assets_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
