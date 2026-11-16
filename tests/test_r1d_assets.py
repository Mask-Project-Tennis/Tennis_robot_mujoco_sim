"""R1D 官方资产与整机导出片段交付物测试(阶段 2 · Task 2.5a)。

五组核心断言(全部以审计报告 2026-11-15-r1a7d-model-interface-audit.md
的固定事实表为唯一期望来源,URDF 原文作交叉验证):

1. ``test_source_manifest_hashes``:manifest 覆盖 37 个官方交付文件
   (r1d.urdf + 36 STL)与 LICENSE 拷贝,逐文件 SHA256 与审计报告 §1
   一致;license 节为 bsd-3-clause(仓库根)记录;移除项(dex1 手指/
   腕部相机)与环境排除项有记录。
2. ``test_joint_contract_matches_audit``:左右裸臂片段各断言 7 个 hinge
   的名称、语义顺序(肩俯仰/肩侧摆/肩偏航/肘/腕滚转/腕俯仰/腕偏航)、
   axis、origin xyz/rpy、限位 rad 逐字等于审计 §2 表(URDF 值);
   ``r1d_flange`` 位置 = 该侧 dex1_base_joint origin。
3. ``test_env_chain_frozen``:环境片段 0 个关节;q_L=0 与 q_L=0.2125
   两次构建的 waist 挂接点 z 差 = 2×0.2125 m(mimic multiplier=1 的
   两级同值 bake 验证,atol=1e-12)。
4. ``test_rebuild_is_deterministic``:相同输入两次生成字节一致,且与
   仓库内成品一致(可复建、无漂移)。
5. ``test_loadable_via_safe_loader``:临时场景(臂片段 + 资产片段 +
   7 个 unit-gear motor)经 load_mujoco_model 编译:nq=nv=nu=7、
   连杆惯量正定(编译一次通过,禁 balanceinertia)、编译后 range 的
   rad 值与 URDF 对照(生成器负责 rad→degree,MJCF 声明 degree)。
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
ASSETS_DIR = REPO_ROOT / "assets" / "unitree_r1d"
URDF_PATH = ASSETS_DIR / "r1d.urdf"
MANIFEST_PATH = ASSETS_DIR / "source_manifest.yaml"
ARM_FRAGMENTS = {
    "left": REPO_ROOT / "src" / "robot" / "r1d_arm_left.xml",
    "right": REPO_ROOT / "src" / "robot" / "r1d_arm_right.xml",
}
ENV_FRAGMENT = REPO_ROOT / "src" / "robot" / "r1d_env.xml"
ASSET_FRAGMENT = REPO_ROOT / "src" / "robot" / "r1d_assets.xml"

# Task 2.0 固定的官方提交与内容哈希(审计报告 §1)
PINNED_COMMIT = "157ff950351e5f535e03c5d85a16b7d594b3dbcc"
PINNED_URDF_SHA256 = (
    "39ad1a206c34c3deecd33745ccda0c0c94ece4195f674dfbaf950f567f08809c"
)
PINNED_LICENSE_SHA256 = (
    "84aac59fd3246e3ddc49d1387644e8fcf43b0def4f0b9fe687f372e90446df2d"
)

# 审计报告 §1 的 38 个交付文件哈希(37 个官方文件 + LICENSE 拷贝,逐字抄录)
AUDIT_FILE_SHA256: dict[str, str] = {
    "r1d.urdf": "39ad1a206c34c3deecd33745ccda0c0c94ece4195f674dfbaf950f567f08809c",
    "LICENSE": "84aac59fd3246e3ddc49d1387644e8fcf43b0def4f0b9fe687f372e90446df2d",
    "meshes/base_link.STL": "a9e6b7fd0a8c2c852ca3552bfa68097d46311f32896a9c7e2d17761467d9b98e",
    "meshes/Camera.STL": "9ac57a9bc1fe9f8a1aff2b679626f61bbb3a8e7c789cb330a07ae000be8a66b3",
    "meshes/Dex1_base_link.STL": "a4e95f91e1b3a349580080bf01d34307221c6be1462c01900392e677dbf668e5",
    "meshes/Dex1_finger_link_1.STL": "f34639c388f4a0b14670b7076307c76aae927dd076b426887d4677adf82790a8",
    "meshes/Dex1_finger_link_2.STL": "ee154ad8f5209d23387d85bed5622319151615e31bec0bd12fc44eccfa29603d",
    "meshes/front_left_caster_Roll_link.STL": "8b6d7fd800039387dadbbe905f21b856a014641e304269d352531177e68e46d7",
    "meshes/front_left_caster_Yaw_link.STL": "91c3190b59e945a40b0cddbbf721b6162d7443ae6b8b023a2c913fdc8f845661",
    "meshes/front_right_caster_Roll_link.STL": "241853e788c8cfb09c6e02fc06b90c26caa000d734342f1b7715f8ad15e4f43b",
    "meshes/front_right_caster_Yaw_link.STL": "e64c58331b721062fe119ef622ae975fe1b23e150d35b1f02fda114e752280c3",
    "meshes/head_pitch_Link.STL": "f2c64723273d93c2da9d5e204f27cd569f8fda0477dc0595760ce09d9d7f9db9",
    "meshes/head_yaw_Link.STL": "2ba30deb440e2584b7595d619ed3829c4ec52ae5c78a2e929b8be70577ceadf3",
    "meshes/left_elbow_link.STL": "e6b6584de6ec6c629da24fbebdc7b88cccd7ed24ad0d7547fa2d871fce4faa7c",
    "meshes/left_shoulder_pitch_Link.STL": "df713bd42d5349868befd618a7e571815677c7f39d026f716a56c366a9521a00",
    "meshes/left_shoulder_roll_link.STL": "7fcfd330aa78b58a9f2dcbf6f87d62b6314eae3678d36abbbc1632a29cf9d7b9",
    "meshes/left_shoulder_yaw_link.STL": "6b09a1c14f9ef1a38fe4d658b4e51586f972b0bace28cd90caada7d80db50fb0",
    "meshes/left_wrist_pitch_link.STL": "7b3c0568ebd0c864e2401d80436cb6ea0fdc4b83ac0892311d734e33b35a7bff",
    "meshes/left_wrist_roll_link.STL": "1cab8b3cf32fd7423b663d600dbc2666f5bcb3c78998c701335111bc8b96848b",
    "meshes/left_wrist_yaw_link.STL": "e490c5848bfab23e4780789cb65e98fa1788f104e759c5647caa0b2eee10f8e2",
    "meshes/Left_Wheel_link.STL": "09c88e244da0d5087d3313d5ed7f5b06162c98126a0383cc5e8e6d34579b4386",
    "meshes/lift_mount_link.STL": "1bd33bf5a12005f2c7f4f0e580fc40b0ac6242898077afa8621429bce34b512c",
    "meshes/lift_stage1_link.STL": "63368dd170c97a4a4c8ee6fbd331352687aee04e129ca49e5bf187ddf82729bd",
    "meshes/lift_stage2_link.STL": "dccf3ebd7390c90548cdb07e1acb0b3198a67f6351e3b3432a7b8f6971f3cd7e",
    "meshes/pelvis_link.STL": "17486b5bb0ee7b24608a87983a45bd690b87515345ebd20161a8d0287b45bfe6",
    "meshes/rear_left_caster_Roll_link.STL": "75c1d1bb63ebc022a1f9f96ddf639d9773d7d731f8720880887dd0e5011dcc16",
    "meshes/rear_left_caster_Yaw_link.STL": "04b8d6ae703f0bd3a051b99fd498e0350f2ac17367f9b1f3dc50ae68e9438dc6",
    "meshes/rear_right_caster_Roll_link.STL": "40cc63c5af5a8b0c28c269c2fec5da81353e846edbf48b0f17b2b072a827f427",
    "meshes/rear_right_caster_Yaw_link.STL": "9450b99657a58470d2989921c79fc21f098e457e492c37fa11d4f32063fb67d1",
    "meshes/right_elbow_link.STL": "1bd3a60789fe4b35ce399d6134e53fa84010f2c6c3e3d888c5cfd91b095ae37e",
    "meshes/right_shoulder_pitch_Link.STL": "beba627f582135b63c05d71e410d6bd5510f313d6fa0229aca67d06c05b92216",
    "meshes/right_shoulder_roll_link.STL": "97e41a1b567ffda5019cdefe9bfe6b0c637b589d07ccaaece5aa4e6ed2b31d15",
    "meshes/right_shoulder_yaw_link.STL": "37f814647983716e1049c18815a6d6f45eb616b9db1958469b4eb21914fc7e05",
    "meshes/right_wrist_pitch_link.STL": "74974a903dd795cf43d19dd67d0e8b455433d38eeeab016b2873da35882160a9",
    "meshes/right_wrist_roll_link.STL": "c4f86872bfa9f72a19e331b7c3b2fb5bf1282e2b6b88b60642abfe3cb8ae4a7d",
    "meshes/right_wrist_yaw_link.STL": "87f1fcfa6bfbffb8478b7164652f6c9067f56ea47b52859b718d48b59b214050",
    "meshes/Right_Wheel_link.STL": "3341865c7d2bc7f494b1037631443b7ffe550e49ec0abad5348b292d29c17ccb",
    "meshes/waist_yaw_link.STL": "54b7f815c79e0b085e783286fdaa64796f0e0ce00e13410b100921390b4c1135",
}

# 审计报告 §2 关节契约(逐字取 URDF 值):
# (joint 名, 子 link 名, axis, origin xyz, origin rpy, 限位 (lower, upper) rad)
# 顺序 = 肩俯仰/肩侧摆/肩偏航/肘/腕滚转/腕俯仰/腕偏航。
# 注意右腕俯仰上界 +0.5326 的官方不对称与右肩侧摆上界 0.2268,均不得"修正"。
AUDIT_JOINTS: dict[str, list[tuple[str, str, tuple[float, float, float],
                                   tuple[float, float, float],
                                   tuple[float, float, float],
                                   tuple[float, float]]]] = {
    "left": [
        ("left_shoulder_pitch_joint", "left_shoulder_pitch_Link",
         (0.0, 1.0, 0.0), (0.0, 0.085688, 0.19749), (0.26187, 0.0, 0.0),
         (-3.1416, 2.0944)),
        ("left_shoulder_roll_joint", "left_shoulder_roll_link",
         (1.0, 0.0, 0.0), (0.03445, 0.047132, -0.025693), (-0.26187, 0.0, 0.0),
         (-0.22689, 2.4784)),
        ("left_shoulder_yaw_joint", "left_shoulder_yaw_link",
         (0.0, 0.0, 1.0), (-0.03445, 0.0043, -0.10835), (0.0, 0.0, 0.0),
         (-1.9199, 1.9199)),
        ("left_elbow_joint", "left_elbow_link",
         (0.0, 1.0, 0.0), (0.016191, 0.026461, -0.082858), (0.0, 0.0, 0.0),
         (-0.97564, 2.1852)),
        ("left_wrist_roll_joint", "left_wrist_roll_link",
         (1.0, 0.0, 0.0), (0.11218, -0.03002, -0.011702), (0.0, 0.0, 0.0),
         (-1.9199, 1.9199)),
        ("left_wrist_pitch_joint", "left_wrist_pitch_link",
         (0.0, 1.0, 0.0), (0.1057, -0.028757, -0.0037491), (0.0, 0.0, 0.0),
         (-0.5236, 0.5236)),
        ("left_wrist_yaw_joint", "left_wrist_yaw_link",
         (0.0, 0.0, 1.0), (0.009848, 0.031494, -0.017245), (0.0, 0.0, 0.0),
         (-1.5708, 1.5708)),
    ],
    "right": [
        ("right_shoulder_pitch_joint", "right_shoulder_pitch_Link",
         (0.0, 1.0, 0.0), (0.0, -0.085688, 0.19749), (-0.26187, 0.0, 0.0),
         (-3.1416, 2.0944)),
        ("right_shoulder_roll_joint", "right_shoulder_roll_link",
         (1.0, 0.0, 0.0), (0.03445, -0.047132, -0.025693), (0.26187, 0.0, 0.0),
         (-2.47849, 0.2268)),
        ("right_shoulder_yaw_joint", "right_shoulder_yaw_link",
         (0.0, 0.0, 1.0), (-0.03445, -0.0043, -0.10835), (0.0, 0.0, 0.0),
         (-1.9199, 1.9199)),
        ("right_elbow_joint", "right_elbow_link",
         (0.0, 1.0, 0.0), (0.016191, -0.026461, -0.082858), (0.0, 0.0, 0.0),
         (-0.97564, 2.1852)),
        ("right_wrist_roll_joint", "right_wrist_roll_link",
         (1.0, 0.0, 0.0), (0.11218, 0.03002, -0.011702), (0.0, 0.0, 0.0),
         (-1.9199, 1.9199)),
        ("right_wrist_pitch_joint", "right_wrist_pitch_link",
         (0.0, 1.0, 0.0), (0.1057, 0.028757, -0.0037491), (0.0, 0.0, 0.0),
         (-0.5236, 0.5326)),
        ("right_wrist_yaw_joint", "right_wrist_yaw_link",
         (0.0, 0.0, 1.0), (0.009848, -0.031494, -0.017245), (0.0, 0.0, 0.0),
         (-1.5708, 1.5708)),
    ],
}

# r1d_flange site 位置 = 该侧 dex1_base_joint origin(审计报告 §3 变换链)
AUDIT_FLANGE_POS: dict[str, tuple[float, float, float]] = {
    "left": (0.0485, -0.003, 0.02),
    "right": (0.0485, 0.003, 0.02),
}

# 环境冻结链(审计报告 §4):base_link → lift_mount → stage1 → stage2
# → pelvis → waist_yaw;两级 prismatic 的 origin z 与行程
ENV_CHAIN_LINKS = (
    "base_link",
    "lift_mount_link",
    "lift_stage1_link",
    "lift_stage2_link",
    "pelvis_link",
    "waist_yaw_link",
)
ENV_LIFT_STAGE1_ORIGIN_Z = 0.0427
ENV_LIFT_STAGE2_ORIGIN_Z = -0.2115
ENV_LIFT_TRAVEL = 0.425  # 单级行程上限 (m);两级同值 bake,总行程 2×0.425

# 转换器模块位于 scripts/tools/(无 __init__.py,按项目测试惯例加入 sys.path)
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))


def _parse_urdf() -> tuple[dict[str, ET.Element], dict[str, ET.Element]]:
    """解析交付的原始 URDF(交叉验证期望值来源)。

    Returns:
        (link 名 → link 元素, joint 名 → joint 元素) 两个映射。
    """
    root = ET.parse(URDF_PATH).getroot()
    links = {el.attrib["name"]: el for el in root.findall("link")}
    joints = {el.attrib["name"]: el for el in root.findall("joint")}
    return links, joints


def _floats(text: str) -> list[float]:
    """把 URDF/MJCF 空格分隔的数值串解析为 float 列表。

    Args:
        text: 数值串(允许多余空格,如 URDF 原文的双空格)。

    Returns:
        解析后的 float 列表。
    """
    return [float(token) for token in text.split()]


def _rpy_to_mat(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF 固定轴 rpy 转旋转矩阵 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。

    Args:
        rpy: (roll, pitch, yaw),单位 rad。

    Returns:
        3×3 旋转矩阵。
    """
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
    """MuJoCo wxyz 四元数转 3×3 旋转矩阵。

    Args:
        quat: (4,) 四元数 (w, x, y, z)。

    Returns:
        3×3 旋转矩阵。
    """
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _build_arm_scene(side: str) -> str:
    """合成单侧臂测试场景 XML 文本(臂片段 + 资产片段 + 7 个 unit-gear motor)。

    经 load_mujoco_model 的 xml_text 参数在片段部署目录(src/robot/ 的
    加载器临时副本)中编译,include 与网格相对路径与交付布局一致。

    Args:
        side: "left" 或 "right"。

    Returns:
        完整 <mujoco> 场景 XML 文本。
    """
    scene = ET.Element("mujoco", {"model": f"r1d_{side}_arm_test"})
    scene.append(ET.Element("include", {"file": f"r1d_arm_{side}.xml"}))
    scene.append(ET.Element("include", {"file": "r1d_assets.xml"}))
    actuator = ET.SubElement(scene, "actuator")
    for i, (joint_name, *_rest) in enumerate(AUDIT_JOINTS[side], start=1):
        ET.SubElement(
            actuator,
            "motor",
            {"name": f"r1d_motor_{i}", "joint": joint_name, "gear": "1"},
        )
    ET.indent(scene, space="  ")
    return ET.tostring(scene, encoding="unicode")


def _build_env_scene() -> str:
    """合成环境测试场景 XML 文本(环境片段 + 资产片段,无执行器)。

    Returns:
        完整 <mujoco> 场景 XML 文本。
    """
    scene = ET.Element("mujoco", {"model": "r1d_env_test"})
    scene.append(ET.Element("include", {"file": "r1d_env.xml"}))
    scene.append(ET.Element("include", {"file": "r1d_assets.xml"}))
    ET.indent(scene, space="  ")
    return ET.tostring(scene, encoding="unicode")


def _load_arm_model(side: str) -> mujoco.MjModel:
    """经安全加载器编译单侧臂测试场景。

    Args:
        side: "left" 或 "right"。

    Returns:
        编译完成的 MjModel。
    """
    return load_mujoco_model(ARM_FRAGMENTS[side], xml_text=_build_arm_scene(side))


def _load_env_model() -> mujoco.MjModel:
    """经安全加载器编译环境测试场景。

    Returns:
        编译完成的 MjModel。
    """
    return load_mujoco_model(ENV_FRAGMENT, xml_text=_build_env_scene())


def _find_body(root: ET.Element, name: str) -> ET.Element:
    """在片段中按名称查找 body 元素,找不到即判定失败。"""
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    pytest.fail(f"片段中缺少 body: {name}")


def _accumulate_env_chain_z(fragment_path: Path) -> float:
    """沿环境冻结链累加 body pos 的 z 分量(链上 rpy 全 0,纯平移叠加)。

    Args:
        fragment_path: 环境片段路径。

    Returns:
        waist_yaw_link 原点相对 base_link 原点的 z 坐标 (m)。
    """
    root = ET.parse(fragment_path).getroot()
    z = 0.0
    for link_name in ENV_CHAIN_LINKS:
        body = _find_body(root, link_name)
        assert "quat" not in body.attrib, f"环境链 {link_name} 不应有旋转"
        z += _floats(body.attrib.get("pos", "0 0 0"))[2]
    return z


def test_source_manifest_hashes() -> None:
    """清单哈希审计:38 个交付文件逐项与审计报告 §1 一致,目录无未登记文件。"""
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))

    source = manifest["source"]
    assert source["commit"] == PINNED_COMMIT

    entries = manifest["files"]
    assert isinstance(entries, list) and entries
    listed = {entry["path"] for entry in entries}
    # manifest 必须覆盖 37 个官方交付文件 + LICENSE 拷贝,与审计 §1 一致
    assert listed == set(AUDIT_FILE_SHA256), (
        f"清单覆盖范围与审计 §1 不符: 缺 {set(AUDIT_FILE_SHA256) - listed},"
        f" 多 {listed - set(AUDIT_FILE_SHA256)}"
    )
    for entry in entries:
        rel_path = entry["path"]
        file_path = ASSETS_DIR / rel_path
        assert file_path.is_file(), f"清单登记的文件缺失: {rel_path}"
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"哈希不符: {rel_path}"
        assert digest == AUDIT_FILE_SHA256[rel_path], (
            f"与审计报告 §1 不一致: {rel_path}"
        )
        assert entry.get("upstream_path"), f"缺少上游路径: {rel_path}"

    # URDF 与固定 commit 逐字节一致
    urdf_entry = next(e for e in entries if e["path"] == "r1d.urdf")
    assert urdf_entry["sha256"] == PINNED_URDF_SHA256
    # LICENSE 拷贝与仓库根 LICENSE 逐字节一致
    license_entry = next(e for e in entries if e["path"] == "LICENSE")
    assert license_entry["sha256"] == PINNED_LICENSE_SHA256

    # 许可状态:bsd-3-clause(仓库根)记录 + 再分发前复核提示
    license_sec = manifest["license"]
    assert "bsd-3-clause" in license_sec["status"]
    assert license_sec["declaration"]["file"]  # 声明文件有记录
    assert license_sec.get("notes")

    # 派生记录:移除项(dex1 手指 prismatic + 腕部相机固定子树)
    removed_names = {
        item["name"] for item in manifest["derived_model"]["removed_from_urdf"]
    }
    for side in ("left", "right"):
        assert f"{side}_dex1_finger_joint_1" in removed_names
        assert f"{side}_dex1_finger_joint_2" in removed_names
        assert f"{side}_dex1_camera_joint" in removed_names

    # 派生记录:环境排除项(轮/脚轮/头/对侧臂)与简化替代记录
    derived = manifest["derived_model"]
    assert derived.get("env_excluded"), "缺少环境排除项记录"
    assert derived.get("simplifications") is not None, "缺少简化替代记录"

    # 完整性:目录内除 README 与清单自身外,所有文件均已登记
    actual = {
        str(p.relative_to(ASSETS_DIR)).replace("\\", "/")
        for p in ASSETS_DIR.rglob("*")
        if p.is_file()
    }
    unlisted = actual - listed - {"source_manifest.yaml", "README.md"}
    assert not unlisted, f"存在未登记文件: {sorted(unlisted)}"


def test_joint_contract_matches_audit() -> None:
    """左右裸臂片段关节契约逐字对照审计 §2 表(静态 XML 级)。"""
    _links, joints = _parse_urdf()

    for side in ("left", "right"):
        root = ET.parse(ARM_FRAGMENTS[side]).getroot()
        worldbody = next((s for s in root if s.tag == "worldbody"), None)
        assert worldbody is not None, f"{side} 片段缺少 worldbody"
        root_body = next((c for c in worldbody if c.tag == "body"), None)
        assert root_body is not None and root_body.attrib.get("name") == "r1d_base"

        expected = AUDIT_JOINTS[side]
        # 逐级下钻:7 个关节 body 的层级顺序 = 审计语义顺序
        parent_body = root_body
        for joint_name, child_link, axis, xyz, rpy, (lower, upper) in expected:
            bodies = [c for c in parent_body if c.tag == "body"]
            assert len(bodies) == 1, (
                f"{side} 链在 {joint_name} 父级应只有唯一的下一个 body"
            )
            body = bodies[0]
            assert body.attrib["name"] == child_link
            # origin xyz 逐字
            np.testing.assert_allclose(
                _floats(body.attrib.get("pos", "0 0 0")), xyz, atol=1e-12
            )
            # origin rpy 经 quat 表达:rpy→矩阵 与 quat→矩阵 一致
            if any(v != 0.0 for v in rpy):
                assert "quat" in body.attrib, f"{joint_name} 缺少安装 rpy 的 quat"
                np.testing.assert_allclose(
                    _quat_wxyz_to_mat(np.array(_floats(body.attrib["quat"]))),
                    _rpy_to_mat(rpy),
                    atol=1e-9,
                )
            else:
                assert "quat" not in body.attrib, f"{joint_name} 不应有旋转"
            # 关节元素:name/hinge/axis/限位(deg→rad 与审计表对照)
            joint_el = body.find("joint")
            assert joint_el is not None, f"{joint_name} 缺少 joint 元素"
            assert joint_el.attrib["name"] == joint_name
            assert joint_el.attrib["type"] == "hinge"
            np.testing.assert_allclose(
                _floats(joint_el.attrib["axis"]), axis, atol=1e-12
            )
            lower_deg, upper_deg = _floats(joint_el.attrib["range"])
            np.testing.assert_allclose(
                (np.radians(lower_deg), np.radians(upper_deg)),
                (lower, upper),
                atol=1e-8,
                err_msg=f"{joint_name} 限位(deg→rad)与审计表不符",
            )
            # URDF 原文交叉验证(审计表与交付 URDF 互验,同串解析须精确相等)
            urdf_joint = joints[joint_name]
            np.testing.assert_array_equal(
                np.array(_floats(urdf_joint.find("origin").attrib["xyz"])),
                np.array(xyz),
            )
            np.testing.assert_allclose(
                _floats(urdf_joint.find("limit").attrib["lower"]), lower,
                rtol=0.0, atol=0.0,
            )
            np.testing.assert_allclose(
                _floats(urdf_joint.find("limit").attrib["upper"]), upper,
                rtol=0.0, atol=0.0,
            )
            parent_body = body

        # 末端 wrist_yaw_link 下:r1d_flange site = 该侧 dex1_base_joint origin
        wrist_yaw_body = parent_body
        assert wrist_yaw_body.attrib["name"] == expected[-1][1]
        site = wrist_yaw_body.find("site")
        assert site is not None, f"{side} 片段缺少 r1d_flange site"
        assert site.attrib["name"] == "r1d_flange"
        np.testing.assert_allclose(
            _floats(site.attrib["pos"]), AUDIT_FLANGE_POS[side], atol=1e-12
        )
        dex1_origin = _floats(
            joints[f"{side}_dex1_base_joint"].find("origin").attrib["xyz"]
        )
        np.testing.assert_array_equal(
            np.array(_floats(site.attrib["pos"])), np.array(dex1_origin)
        )
        # 移除项不再出现:dex1 夹爪/相机子树
        all_names = {el.attrib.get("name") for el in root.iter()}
        for gone in (
            f"{side}_dex1_base_link",
            f"{side}_dex1_finger_link_1",
            f"{side}_dex1_camera_link",
        ):
            assert gone not in all_names, f"{gone} 应从臂片段移除"


def test_env_chain_frozen(tmp_path: Path) -> None:
    """环境片段 0 关节;两级 q_L 同值 bake 的 z 位移关系 = 2×Δq_L。"""
    # 仓库交付的环境片段:不含任何 <joint> 元素
    env_root = ET.parse(ENV_FRAGMENT).getroot()
    assert not list(env_root.iter("joint")), "环境片段必须无任何关节"

    # 编译验证:njnt == 0 且冻结链 body 存在、位置正确
    model = _load_env_model()
    assert model.njnt == 0
    assert model.nq == 0 and model.nv == 0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for link_name in ENV_CHAIN_LINKS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        assert bid != -1, f"环境片段缺少 body: {link_name}"
    waist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "waist_yaw_link")
    # q_L=0:base_link 原点起 0.3288 + 0.0427 - 0.2115 = 0.1600 m
    np.testing.assert_allclose(
        data.xpos[waist_id][2],
        0.3288 + ENV_LIFT_STAGE1_ORIGIN_Z + ENV_LIFT_STAGE2_ORIGIN_Z,
        atol=1e-12,
    )

    # 两次构建(q_L=0 与 q_L=0.2125)的 waist 挂接点 z 差 = 2×0.2125
    try:
        from build_r1d_model import build_r1d_env
    except ImportError as exc:  # 转换器模块即交付功能,缺失属于功能缺失
        pytest.fail(f"转换器模块缺失,环境片段不可复建: {exc}")

    env_a = tmp_path / "env_q0.xml"
    env_b = tmp_path / "env_q_half.xml"
    build_r1d_env(URDF_PATH, env_a, q_L=0.0)
    build_r1d_env(URDF_PATH, env_b, q_L=0.2125)
    z_a = _accumulate_env_chain_z(env_a)
    z_b = _accumulate_env_chain_z(env_b)
    np.testing.assert_allclose(z_b - z_a, 2 * 0.2125, atol=1e-12)


def test_rebuild_is_deterministic(tmp_path: Path) -> None:
    """相同输入两次生成字节一致,且与仓库内成品一致(可复建、无漂移)。"""
    try:
        from build_r1d_model import (
            build_r1d_arm,
            build_r1d_assets,
            build_r1d_env,
        )
    except ImportError as exc:  # 转换器模块即交付功能,缺失属于功能缺失
        pytest.fail(f"转换器模块缺失,片段不可复建: {exc}")

    outputs: list[tuple[bytes, ...]] = []
    for run in ("run1", "run2"):
        out_dir = tmp_path / run
        out_dir.mkdir()
        paths = (
            out_dir / "r1d_arm_left.xml",
            out_dir / "r1d_arm_right.xml",
            out_dir / "r1d_env.xml",
            out_dir / "r1d_assets.xml",
        )
        build_r1d_arm(URDF_PATH, "left", paths[0])
        build_r1d_arm(URDF_PATH, "right", paths[1])
        build_r1d_env(URDF_PATH, paths[2])
        build_r1d_assets(URDF_PATH, paths[3])
        outputs.append(tuple(p.read_bytes() for p in paths))

    # 两次生成字节一致
    assert outputs[0] == outputs[1]

    # 与仓库内交付物一致(重建命令:python scripts/tools/build_r1d_model.py)
    repo_outputs = (
        ARM_FRAGMENTS["left"].read_bytes(),
        ARM_FRAGMENTS["right"].read_bytes(),
        ENV_FRAGMENT.read_bytes(),
        ASSET_FRAGMENT.read_bytes(),
    )
    assert outputs[0] == repo_outputs, "仓库成品与重建结果不一致(漂移)"


def test_loadable_via_safe_loader() -> None:
    """臂场景经安全加载器编译:nq=nv=nu=7、惯量正定、编译后限位与 URDF 一致。"""
    _links, joints = _parse_urdf()

    for side in ("left", "right"):
        model = _load_arm_model(side)

        # 维数契约:仅选定臂 7 个铰链,完整模型维数不被环境/夹爪污染
        assert model.nq == 7, f"{side}: nq={model.nq} != 7"
        assert model.nv == 7, f"{side}: nv={model.nv} != 7"
        assert model.nu == 7, f"{side}: nu={model.nu} != 7"

        expected = AUDIT_JOINTS[side]
        joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(model.njnt)
        ]
        assert joint_names == [row[0] for row in expected]
        for j, (joint_name, _child_link, *_rest) in enumerate(expected):
            assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            limit = joints[joint_name].find("limit")
            assert limit is not None
            expected_range = (
                _floats(limit.attrib["lower"])[0],
                _floats(limit.attrib["upper"])[0],
            )
            # 片段以 degree 声明限位,编译后应还原 URDF 的 rad 值(单位往返无损)
            np.testing.assert_allclose(
                model.jnt_range[j], expected_range, atol=1e-8
            )
            assert model.jnt_limited[j] == 1

        # 根 body r1d_base:虚拟固定安装点(无关节、挂 worldbody)
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r1d_base")
        assert base_id != -1
        assert int(model.body_parentid[base_id]) == 0
        assert int(model.body_jntnum[base_id]) == 0

        # 惯量正定:7 个连杆质量 > 0 且主惯量全正
        # (编译一次通过本身即官方惯量满足要求,无 balanceinertia 修补)
        for _joint_name, child_link, *_rest in expected:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, child_link)
            assert bid != -1, f"缺少 body: {child_link}"
            assert model.body_mass[bid] > 0.0, f"{child_link} 质量非正"
            assert np.all(model.body_inertia[bid] > 0.0), f"{child_link} 主惯量非正"

        # 移除项不进模型:dex1 夹爪/相机
        assert (
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_dex1_base_link"
            )
            == -1
        )
        assert (
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_dex1_finger_joint_1"
            )
            == -1
        )
