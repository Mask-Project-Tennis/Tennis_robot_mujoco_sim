"""D2 官方资产与裸臂 MJCF 交付物测试（阶段 2 · Task 2.1）。

三组核心断言：
1. ``test_joint_contract_and_limit_units``：裸臂片段在临时场景（七个 unit-gear
   motor + d2_flange site）中可经安全加载器编译；关节契约（joint_1..7 铰链）、
   限位单位（片段声明 degree，编译后的 rad 值与原始 URDF 逐项一致）、惯量正定
   与移除记录（无 link_8 / joint_8）全部成立。
2. ``test_source_manifest_hashes``：source_manifest.yaml 登记的每个交付资源哈希
   正确，目录内无未登记文件，且 URDF 与固定 commit 的 blob 逐字节一致。
3. ``test_rebuild_is_deterministic``：相同输入两次生成字节一致，且与仓库内
   交付物一致（可复建、无漂移）。
"""

from __future__ import annotations

import hashlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from src.utils.mujoco_loader import load_mujoco_model

# 仓库根目录与关键交付物路径
REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "unitree_d2"
URDF_PATH = ASSETS_DIR / "D2_URDF.urdf"
MANIFEST_PATH = ASSETS_DIR / "source_manifest.yaml"
ARM_FRAGMENT = REPO_ROOT / "src" / "robot" / "d2_arm.xml"
ASSET_FRAGMENT = REPO_ROOT / "src" / "robot" / "d2_assets.xml"

# 阶段 0 固定的官方提交与 URDF 内容哈希（与 stage0-evidence 交叉验证一致）
PINNED_COMMIT = "8d505a8c511f2a8dd401bd4c410e37121b1bfd17"
PINNED_URDF_SHA256 = (
    "6f508a49719d8f7f02af284f7ecf4d6312c9cca17fd8251951a8bba232cbec73"
)

# 临时场景必须位于仓库二级目录（src/robot/），使片段内网格相对路径
# "../../assets/unitree_d2/meshes/*.STL" 在加载器临时目录结构中可解析。
SCENE_PATH = REPO_ROOT / "src" / "robot" / "_tmp_d2_assets_test_scene.xml"

# 转换器模块位于 scripts/tools/（无 __init__.py，按项目测试惯例加入 sys.path）
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))


def _parse_urdf() -> tuple[dict[str, ET.Element], dict[str, ET.Element]]:
    """解析交付的原始 URDF。

    Returns:
        (link 名 → link 元素, joint 名 → joint 元素) 两个映射。
    """
    root = ET.parse(URDF_PATH).getroot()
    links = {el.attrib["name"]: el for el in root.findall("link")}
    joints = {el.attrib["name"]: el for el in root.findall("joint")}
    return links, joints


def _floats(text: str) -> list[float]:
    """把 URDF 空格分隔的数值串解析为 float 列表。"""
    return [float(token) for token in text.split()]


def _find_body(root: ET.Element, name: str) -> ET.Element:
    """在片段中按名称查找 body 元素，找不到即判定失败。"""
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    pytest.fail(f"片段中缺少 body: {name}")


def _build_test_scene(scene_path: Path) -> np.ndarray:
    """合成临时测试场景并写入磁盘。

    场景为完整 ``<mujoco>`` 文档：裸臂片段的 compiler/worldbody + 资产片段的
    asset + 七个 unit-gear motor（不设 ctrlrange，官方 effort 未知）；并在
    link_7 下加入 d2_flange site（位置取自被移除 joint_8 的 origin，即官方
    工具安装面）。

    Args:
        scene_path: 场景文件写入路径（须位于仓库二级目录，见 SCENE_PATH 注释）。

    Returns:
        d2_flange 的期望位置（link_7 局部系，米）。
    """
    arm_root = ET.parse(ARM_FRAGMENT).getroot()
    assets_root = ET.parse(ASSET_FRAGMENT).getroot()
    _, urdf_joints = _parse_urdf()
    origin_xyz_text = urdf_joints["joint_8"].find("origin").attrib["xyz"]
    flange_xyz = np.array(_floats(origin_xyz_text))

    scene = ET.Element("mujoco", {"model": "d2_bare_arm_test"})
    for section in arm_root:
        if section.tag == "compiler":
            # 携带片段声明的 angle="degree"（限位单位契约）
            scene.append(section)

    worldbody = ET.SubElement(scene, "worldbody")
    arm_worldbody = next((s for s in arm_root if s.tag == "worldbody"), None)
    assert arm_worldbody is not None, "裸臂片段缺少 worldbody"
    root_body = next((c for c in arm_worldbody if c.tag == "body"), None)
    assert root_body is not None, "裸臂片段缺少根 body"
    assert root_body.attrib.get("name") == "d2_base"
    link7 = _find_body(root_body, "link_7")
    ET.SubElement(
        link7,
        "site",
        {"name": "d2_flange", "pos": origin_xyz_text, "size": "0.005"},
    )
    worldbody.append(root_body)

    for section in assets_root:
        scene.append(section)

    actuator = ET.SubElement(scene, "actuator")
    for i in range(1, 8):
        ET.SubElement(
            actuator,
            "motor",
            {"name": f"d2_motor_joint_{i}", "joint": f"joint_{i}", "gear": "1"},
        )

    ET.indent(scene, space="  ")
    ET.ElementTree(scene).write(
        scene_path, encoding="utf-8", xml_declaration=True
    )
    return flange_xyz


def test_joint_contract_and_limit_units() -> None:
    """裸臂关节契约、限位单位与惯量正定（编译后 rad 对照原始 URDF）。"""
    try:
        flange_xyz = _build_test_scene(SCENE_PATH)
        model = load_mujoco_model(SCENE_PATH)
    finally:
        SCENE_PATH.unlink(missing_ok=True)

    # 维数契约：7 个铰链关节、7 个 unit-gear motor
    assert model.nq == 7
    assert model.nv == 7
    assert model.nu == 7

    # 关节契约：七个 revolute（hinge）关节名为 joint_1..joint_7
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
    ]
    assert joint_names == [f"joint_{i}" for i in range(1, 8)]

    _, urdf_joints = _parse_urdf()
    for j, name in enumerate(joint_names):
        assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        limit = urdf_joints[name].find("limit")
        assert limit is not None
        expected = (
            _floats(limit.attrib["lower"])[0],
            _floats(limit.attrib["upper"])[0],
        )
        # 片段以 degree 声明限位，编译后应还原 URDF 的 rad 值（单位往返无损）
        np.testing.assert_allclose(model.jnt_range[j], expected, atol=1e-8)
        assert model.jnt_limited[j] == 1

    # joint_8（固定关节）已在派生模型中移除
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_8") == -1

    # body 契约：固定根 body d2_base（对应 URDF base_link）+ link_1..link_7
    expected_bodies = ["d2_base"] + [f"link_{i}" for i in range(1, 8)]
    for name in expected_bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid != -1, f"缺少 body: {name}"
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_8") == -1
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "d2_base")
    assert int(model.body_parentid[base_id]) == 0  # 直接挂在 worldbody（固定基座）
    assert int(model.body_jntnum[base_id]) == 0

    # 惯量正定：每个连杆质量 > 0 且主惯量全部为正（三角不等式由编译强制）
    for name in expected_bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert model.body_mass[bid] > 0.0
        assert np.all(model.body_inertia[bid] > 0.0)

    # 视觉网格：base_link + link_1..link_7 共 8 个（link_8 已移除）
    assert model.nmesh == 8

    # d2_flange site 位于被移除 joint_8 的 origin（官方工具安装面）
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "d2_flange")
    assert sid != -1
    np.testing.assert_allclose(model.site_pos[sid], flange_xyz, atol=1e-12)


def test_source_manifest_hashes() -> None:
    """清单哈希审计：每个交付资源哈希正确、目录无未登记文件。"""
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))

    source = manifest["source"]
    assert source["commit"] == PINNED_COMMIT

    entries = manifest["files"]
    assert isinstance(entries, list) and entries

    listed: set[str] = set()
    for entry in entries:
        rel_path = entry["path"]
        file_path = ASSETS_DIR / rel_path
        assert file_path.is_file(), f"清单登记的文件缺失: {rel_path}"
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"哈希不符: {rel_path}"
        assert entry.get("upstream_path"), f"缺少上游路径: {rel_path}"
        listed.add(rel_path.replace("\\", "/"))

    # URDF 必须与固定 commit 的 blob 逐字节一致（stage0-evidence 同源）
    urdf_entry = next(e for e in entries if e["path"] == "D2_URDF.urdf")
    assert urdf_entry["sha256"] == PINNED_URDF_SHA256

    # 完整性：目录内除 README 与清单自身外，所有文件均已登记
    actual = {
        str(p.relative_to(ASSETS_DIR)).replace("\\", "/")
        for p in ASSETS_DIR.rglob("*")
        if p.is_file()
    }
    assert listed <= actual
    unlisted = actual - listed - {"source_manifest.yaml", "README.md"}
    assert not unlisted, f"存在未登记文件: {sorted(unlisted)}"

    # 派生记录：joint_8/link_8 移除；速度/力矩按未知记录
    removed = {
        item["name"] for item in manifest["derived_model"]["removed_from_urdf"]
    }
    assert removed == {"joint_8", "link_8"}
    unknown = manifest["unknown_capabilities"]
    assert "joint_velocity_limits" in unknown
    assert "joint_effort_limits" in unknown
    # 许可状态必须显式记录（未确认前为 unverified）
    assert manifest["license"]["status"]


def test_rebuild_is_deterministic(tmp_path: Path) -> None:
    """相同输入两次生成字节一致，且与仓库内交付物一致（可复建、无漂移）。"""
    try:
        from build_d2_model import build_d2_arm
    except ImportError as exc:  # 转换器模块即交付功能，缺失属于功能缺失
        pytest.fail(f"转换器模块缺失，裸臂不可复建: {exc}")

    outputs = []
    for run in ("run1", "run2"):
        out_dir = tmp_path / run
        out_dir.mkdir()
        arm_out = out_dir / "d2_arm.xml"
        assets_out = out_dir / "d2_assets.xml"
        build_d2_arm(URDF_PATH, arm_out, assets_out)
        outputs.append((arm_out.read_bytes(), assets_out.read_bytes()))

    # 两次生成字节一致
    assert outputs[0] == outputs[1]

    # 与仓库内交付物一致（重建命令：python scripts/tools/build_d2_model.py）
    assert outputs[0][0] == ARM_FRAGMENT.read_bytes()
    assert outputs[0][1] == ASSET_FRAGMENT.read_bytes()
