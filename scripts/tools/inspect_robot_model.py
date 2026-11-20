"""R1D 模型预览与可追溯参数导出工具（阶段 2 · Task 2.7）。

无头检查 + JSON 报告 + 可选三视图渲染/交互 viewer。后续执行者凭本工具
从命令复建相同模型并看到参数、侧别、冻结与冲突状态——不依赖个人目录、
SDK 或 MPC。

数据来源契约：
- 模型/绑定/参数全部来自 Task 2.5b/2.6 交接收口（只读消费）：
  ``load_robot_profile`` → ``load_runtime_parameters`` → ``load_robot_model``
  → ``ModelKinematics``；
- 可选的 ``--racket-config`` 提供球拍档案（``load_racket_profile`` →
  ``load_robot_model(..., racket_profile=...)``）：几何/质量/惯量/安装/
  连接件的唯一物理来源，报告新增 ``racket`` 节记录配置哈希与装配事实；
  不传时保持旧 API 兼容语义（模板默认球拍，``racket.mode = "legacy-template"``）；
- 冻结部件、SDK 整机索引、数据冲突四项引用自审计报告
  ``docs/experiments/reports/2026-11-15-r1a7d-model-interface-audit.md``
  （§2/§4/§5，只记录不裁断）；SDK 索引是 DDS 通信地址，不进模型、不进
  ModelBinding（共享约束 N0-8），JSON 中仅作只读对照字段；
- 升降 bake 值 q_L 由已加载模型的 ``body_pos`` 推导
  （z(stage1)+z(stage2) = 2·q_L − 0.1688，审计 §4 的 mimic 位移关系），
  不硬抄配置注释。

退出码：0 成功；2 异常配置（缺文件/坏 YAML/身份不符等）；3 渲染或
viewer 失败（此时 JSON 已写出并保留）。

用法（从仓库根目录）::

    python scripts/tools/inspect_robot_model.py \\
        --robot-config configs/robots/r1d_right.yaml \\
        --runtime-config configs/robots/r1d_v12_runtime_right.yaml \\
        --racket-config configs/rackets/blade100_v9_reference.yaml \\
        --output-dir results/r1d_v12_parity/t2_preview --render
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# 仓库根加入 sys.path（脚本在 scripts/tools/ 下，需上溯三层；先例
# view_tennis_rbt.py）——直接以 `python scripts/tools/inspect_robot_model.py`
# 运行时 src 包可导入；经 pytest 导入时该路径已在 sys.path，无副作用
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from src.robot.model_factory import RobotModelBundle, load_robot_model  # noqa: E402
from src.robot.model_kinematics import ModelKinematics  # noqa: E402
from src.robot.racket_profile import (  # noqa: E402
    RacketProfile,
    load_racket_profile,
)
from src.robot.robot_profile import RobotProfile, load_robot_profile  # noqa: E402
from src.robot.runtime_parameters import load_runtime_parameters  # noqa: E402

logger = logging.getLogger(__name__)

# 退出码契约（任务卡：异常配置 2；渲染失败非零且保留 JSON）
EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_RENDER = 3

# JSON 报告结构标识（字段集演进时递增，供下游消费方判定）
REPORT_SCHEMA = "r1d_inspect_report/v1"

# 审计报告固定引用（数据冲突与 SDK 索引的唯一数据源）
AUDIT_REPORT_REL = "docs/experiments/reports/2026-11-15-r1a7d-model-interface-audit.md"

# 官方来源固定信息（审计 §1：unitree_ros @ 157ff950）
PINNED_COMMIT = "157ff950351e5f535e03c5d85a16b7d594b3dbcc"
PINNED_URDF_SHA256 = (
    "39ad1a206c34c3deecd33745ccda0c0c94ece4195f674dfbaf950f567f08809c"
)

# 环境链 body 名（q_L 推导按名定位，缺失即视为模板漂移）
_LIFT_STAGE1_BODY = "lift_stage1_link"
_LIFT_STAGE2_BODY = "lift_stage2_link"

# 双级升降 mimic 位移关系（审计 §4）：z(stage2 相对 mount) = 2·q_L − 0.1688，
# 其中 −0.1688 = stage1 origin z(+0.0427) + stage2 origin z(−0.2115)。
_LIFT_FORMULA_OFFSET_M = 0.1688
_LIFT_TRAVEL_RANGE_M = (0.0, 0.425)

# 三视图相机参数（MuJoCo 球坐标：lookat/azimuth/elevation/distance）
# front=从机器人前方(+x)看，side=从侧面看，oblique=斜上方 3/4 视角
# （oblique 俯角取 28° 并略拉远，保证画面下缘包含地面，便于目视核对
# 基座与地面接触）
_RENDER_VIEWS: tuple[tuple[str, tuple[float, float, float], float, float], ...] = (
    ("front", (0.10, 0.0, 1.05), 180.0, 5.0),
    ("side", (0.10, 0.0, 1.05), 90.0, 5.0),
    ("oblique", (0.10, 0.0, 1.05), 135.0, 28.0),
)
_RENDER_SIZE_PX = (800, 600)  # (width, height)

# SDK 整机索引对照（审计 §2 表逐字；左 15–21 / 右 22–28）
# 注意：这是宇树 DDS LowState motor_state[] 通信地址，不是 MuJoCo 状态地址
_SDK_SIDE_JOINTS: dict[str, list[dict[str, Any]]] = {
    "left": [
        {"semantic": "肩俯仰", "sdk_name": "L_SHOULDER_PITCH", "sdk_index": 15},
        {"semantic": "肩侧摆", "sdk_name": "L_SHOULDER_ROLL", "sdk_index": 16},
        {"semantic": "肩偏航", "sdk_name": "L_SHOULDER_YAW", "sdk_index": 17},
        {"semantic": "肘", "sdk_name": "L_ELBOW", "sdk_index": 18},
        {"semantic": "腕滚转", "sdk_name": "L_WRIST_ROLL", "sdk_index": 19},
        {"semantic": "腕俯仰", "sdk_name": "L_WRIST_PITCH", "sdk_index": 20},
        {"semantic": "腕偏航", "sdk_name": "L_WRIST_YAW", "sdk_index": 21},
    ],
    "right": [
        {"semantic": "肩俯仰", "sdk_name": "R_SHOULDER_PITCH", "sdk_index": 22},
        {"semantic": "肩侧摆", "sdk_name": "R_SHOULDER_ROLL", "sdk_index": 23},
        {"semantic": "肩偏航", "sdk_name": "R_SHOULDER_YAW", "sdk_index": 24},
        {"semantic": "肘", "sdk_name": "R_ELBOW", "sdk_index": 25},
        {"semantic": "腕滚转", "sdk_name": "R_WRIST_ROLL", "sdk_index": 26},
        {"semantic": "腕俯仰", "sdk_name": "R_WRIST_PITCH", "sdk_index": 27},
        {"semantic": "腕偏航", "sdk_name": "R_WRIST_YAW", "sdk_index": 28},
    ],
}

# SDK 其他部件整机索引（审计 §2“其他部件 SDK 索引”节）
_SDK_OTHER_COMPONENTS: dict[str, int] = {
    "L_CHASSIS_WHEEL": 0,
    "R_CHASSIS_WHEEL": 1,
    "LIFTING_COLUMN": 12,
    "WAIST_YAW": 13,
    "HEAD_PITCH": 29,
    "HEAD_YAW": 30,
}

# 审计报告 §5 冲突清单四项（逐字引用，只记录不裁断）
_DATA_CONFLICTS: dict[str, Any] = {
    "source": f"{AUDIT_REPORT_REL} §5（只记录，不裁断）",
    "policy": (
        "阶段 2 派生模型与测试逐字采用 URDF 数值（含右腕俯仰上界 +0.5326 "
        "的官方不对称，不修正疑似笔误）；指南数值仅存档对照；真机接入"
        "（阶段 4）以实机回读限位为准并回报修订"
    ),
    "items": [
        {
            "id": 1,
            "component": "腰 WAIST_YAW（索引 13）",
            "guide_value_rad": "±0.6109（=±35°，与规格表“腰关节 Yaw±35°”一致）",
            "urdf_value_rad": "±2.618（=±150°）",
            "remark": (
                "差 4.3 倍。可能分别是控制安全限位与机械极限，官方无说明。"
                "阶段 2 冻结腰为 0（两集合公共值），不选边"
            ),
        },
        {
            "id": 2,
            "component": "左腕俯仰 L_WRIST_PITCH（20）",
            "guide_value_rad": "±0.7854",
            "urdf_value_rad": "[-0.5236, 0.5236]",
            "remark": "指南宽于 URDF；派生模型采用 URDF 值",
        },
        {
            "id": 3,
            "component": "右腕俯仰 R_WRIST_PITCH（27）",
            "guide_value_rad": "±0.7854",
            "urdf_value_rad": "[-0.5236, 0.5326]",
            "remark": (
                "同上；且 URDF 右侧自身不对称（+0.5326 vs 左 +0.5236），"
                "疑似官方笔误，不修正"
            ),
        },
        {
            "id": 4,
            "component": "左/右腕滚转 WRIST_ROLL（19/26）",
            "guide_value_rad": "±2.0944（=±120°）",
            "urdf_value_rad": "±1.9199（=±110°）",
            "remark": "指南宽于 URDF 10°；派生模型采用 URDF 值",
        },
    ],
}


# ---------------------------------------------------------------- 工具函数


def _repo_root() -> Path:
    """取仓库根目录（本脚本位于 scripts/tools/，上溯两级）。"""
    return Path(__file__).resolve().parents[2]


def _sha256_of(path: Path) -> str:
    """计算文件内容的 SHA256 十六进制摘要。

    Args:
        path: 目标文件路径。

    Returns:
        64 位十六进制摘要字符串。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_key(path: Path, repo_root: Path) -> str:
    """资源哈希表的显示键：优先仓库相对路径（正斜杠），跨盘符退回绝对。"""
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(path.resolve())


def _floats(values: Any) -> list[float]:
    """把数值序列转为可 JSON 序列化的 Python float 列表。"""
    return [float(v) for v in np.asarray(values).ravel()]


def _sdk_entry_for_joint(side: str, joint_name: str) -> dict[str, Any]:
    """按完整关节名查该侧 SDK 对照条目（按名匹配，不按列表位置配对）。

    关节名形如 ``left_shoulder_pitch_joint``：去侧别前缀与 ``_joint``
    后缀得到语义干（``shoulder_pitch``），与 ``sdk_name`` 去 ``L_``/``R_``
    前缀后的小写形式精确匹配。绑定地址本身按名解析，profile 关节顺序
    合法重排时按位置配对会把 SDK 索引错标到别的关节上（对照字段与
    joint_name 同行自相矛盾），故一律按名查询。

    Args:
        side: 侧别（left/right）。
        joint_name: 完整关节名（URDF 原名）。

    Returns:
        SDK 对照条目（semantic/sdk_name/sdk_index）。

    Raises:
        ValueError: 关节名不在该侧 SDK 对照表中（命名契约破裂，
            由 main 映射为退出码 2 的配置错误类）。
    """
    stem_table = {
        entry["sdk_name"].split("_", 1)[1].lower(): entry
        for entry in _SDK_SIDE_JOINTS[side]
    }
    prefix, suffix = f"{side}_", "_joint"
    if not joint_name.startswith(prefix) or not joint_name.endswith(suffix):
        raise ValueError(
            f"关节名 {joint_name!r} 应为 {prefix}…{suffix} 形态"
            f"（{side} 侧 SDK 对照命名契约，审计 §2）"
        )
    stem = joint_name[len(prefix) : -len(suffix)]
    entry = stem_table.get(stem)
    if entry is None:
        raise ValueError(
            f"关节 {joint_name!r} 无法对应到 {side} 侧 SDK 对照表"
            f"（语义干 {stem!r} 不在 {sorted(stem_table)}，审计 §2 命名契约）"
        )
    return entry


def parse_side(profile: RobotProfile) -> str:
    """从 profile 主动关节名前缀解析侧别（left/right）。

    Args:
        profile: 机器人描述（关节名为该侧 URDF 原名，形如
            ``left_shoulder_pitch_joint``）。

    Returns:
        "left" 或 "right"。

    Raises:
        ValueError: 前缀缺失、混合或不在 {left, right} 内。
    """
    prefixes = set()
    for name in profile.active_joint_names:
        tokens = name.split("_", 1)
        if len(tokens) != 2:
            raise ValueError(f"关节名无法解析出侧别前缀: {name!r}")
        prefixes.add(tokens[0])
    if prefixes == {"left"}:
        return "left"
    if prefixes == {"right"}:
        return "right"
    raise ValueError(
        f"无法从 active_joint_names 前缀解析唯一侧别: {sorted(prefixes)}"
    )


def _rotation_to_quat_and_euler(
    rot: np.ndarray,
) -> tuple[list[float], list[float]]:
    """3×3 旋转矩阵转 (wxyz 四元数, rpy 欧拉)。

    欧拉约定与项目一致：R = Rz(yaw)·Ry(pitch)·Rx(roll)（固定轴）。

    Args:
        rot: 3×3 世界旋转矩阵。

    Returns:
        ((w,x,y,z) 四元数列表, (roll, pitch, yaw) 弧度列表)。
    """
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=np.float64).flatten())
    w, x, y, z = (float(v) for v in quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [w, x, y, z], [roll, pitch, yaw]


def _derive_lift_ql(model: mujoco.MjModel) -> float:
    """从已加载模型推导升降 bake 值 q_L（米）。

    环境片段把两级 prismatic 按同一 q_L bake 为固定偏移
    （mimic multiplier=1，审计 §4）：
    ``z(stage1 相对父) + z(stage2 相对父) = 2·q_L − 0.1688``。
    两级各推一次并互检（不一致即模板漂移）。

    Args:
        model: 已加载的 R1D 场景模型。

    Returns:
        q_L（m）。

    Raises:
        ValueError: 模型缺少升降链 body，或两级推导值不一致。
    """
    z_values = {}
    for body_name in (_LIFT_STAGE1_BODY, _LIFT_STAGE2_BODY):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"模型缺少环境链 body '{body_name}'（模板漂移）")
        z_values[body_name] = float(model.body_pos[body_id][2])
    # stage1 origin z = +0.0427，stage2 origin z = −0.2115（审计 §4 逐字）
    ql_from_stage1 = z_values[_LIFT_STAGE1_BODY] - 0.0427
    ql_from_stage2 = z_values[_LIFT_STAGE2_BODY] + 0.2115
    if not math.isclose(ql_from_stage1, ql_from_stage2, abs_tol=1e-9):
        raise ValueError(
            f"两级升降 bake 值推导不一致: stage1→{ql_from_stage1}, "
            f"stage2→{ql_from_stage2}（模板漂移）"
        )
    return 0.5 * (ql_from_stage1 + ql_from_stage2)


# ---------------------------------------------------------------- 报告构建


def _resource_hashes(
    profile: RobotProfile,
    robot_config: Path,
    runtime_config: Path,
    repo_root: Path,
    racket_config: Path | None = None,
) -> dict[str, Any]:
    """收集配置与模型资源 SHA256（复现本次检查的完整输入集）。

    覆盖：robot/runtime/racket YAML、场景模板、arm/env/assets 三片段、
    官方 URDF、模板引用的全部 mesh（按 mesh file 属性解析）。

    Args:
        profile: 机器人描述（model_path 指向场景模板）。
        robot_config: 描述 YAML 路径。
        runtime_config: 运行参数 YAML 路径。
        repo_root: 仓库根目录（显示键用）。
        racket_config: 可选的球拍档案 YAML 路径（唯一物理来源）。

    Returns:
        {"files": {显示键: sha256}, "template_meshes": {mesh 名: sha256}}。
    """
    files: dict[str, str] = {}
    # 对应侧 arm 片段与模板同目录同名推断
    side = parse_side(profile)
    candidates = [
        robot_config,
        runtime_config,
        profile.model_path,
        profile.model_path.parent / f"r1d_arm_{side}.xml",
        repo_root / "src" / "robot" / "r1d_env.xml",
        repo_root / "src" / "robot" / "r1d_assets.xml",
        repo_root / "assets" / "unitree_r1d" / "r1d.urdf",
    ]
    if racket_config is not None:
        candidates.insert(2, Path(racket_config))
    for path in candidates:
        if path.is_file():
            files[_display_key(path, repo_root)] = _sha256_of(path)
        else:
            logger.warning("资源哈希跳过缺失文件: %s", path)

    # 模板引用的 mesh（file 相对模板父目录解析）
    meshes: dict[str, str] = {}
    root = ET.fromstring(profile.model_path.read_bytes())
    for mesh in root.iter("mesh"):
        name = mesh.attrib.get("name", "")
        file_attr = mesh.attrib.get("file")
        if file_attr is None:
            continue
        mesh_path = (profile.model_path.parent / file_attr).resolve()
        if mesh_path.is_file():
            meshes[name] = _sha256_of(mesh_path)
        else:
            logger.warning("模板 mesh 缺失，跳过哈希: %s", mesh_path)
    return {"files": files, "template_meshes": meshes}


def _racket_report_section(
    bundle: RobotModelBundle,
    *,
    racket_config: Path | None,
    racket_profile: RacketProfile | None,
) -> dict[str, Any]:
    """组装报告 racket 节：档案哈希 + 装配事实 + TCP site 局部位置。

    档案模式的数值全部来自 ``bundle.racket``（RacketProfile 装配事实，
    与模型同源）；旧 API 兼容模式只记录模板默认拍在模型中的质量/惯量，
    并在 note 中说明物理来源缺失。
    """
    repo_root = _repo_root()
    model = bundle.model
    tcp_site_id = int(bundle.binding.tcp_site_id)
    section: dict[str, Any] = {
        "mode": "profile" if racket_profile is not None else "legacy-template",
        "note": (
            "档案模式：几何/质量/惯量/安装/连接件唯一物理来源是 RacketProfile，"
            "本节的 total_mass/com/face/mount/connector 为该档案装配事实"
            if racket_profile is not None
            else "旧 API 兼容模式：未提供 racket-config，球拍为模板默认/legacy "
            "scene.racket_* 参数，无档案装配事实"
        ),
        "config_path": None,
        "config_sha256": None,
        "racket_id": None,
        "source_path": None,
        "model_path": str(bundle.profile.model_path),
        "active_joint_count": len(bundle.profile.active_joint_names),
        "tcp_site": bundle.profile.tcp_site_name,
        "tcp_local_position": _floats(model.site_pos[tcp_site_id]),
        "total_mass": None,
        "com": None,
        "inertia_diagonal": None,
        "face_center": None,
        "face_semi_axes": None,
        "mount_position": None,
        "mount_wxyz": None,
        "connector_mass": None,
    }
    if racket_config is not None:
        config_path = Path(racket_config)
        section["config_path"] = _display_key(config_path, repo_root)
        section["config_sha256"] = _sha256_of(config_path)
    if racket_profile is not None:
        section["racket_id"] = racket_profile.racket_id
        section["source_path"] = str(racket_profile.source_path)
    racket = bundle.racket
    if racket is not None:
        section.update(
            {
                "total_mass": float(racket.total_mass),
                "com": _floats(racket.com),
                "inertia_diagonal": _floats(racket.inertia_diagonal),
                "face_center": _floats(racket.face_center),
                "face_semi_axes": _floats(racket.face_semi_axes),
                "mount_position": _floats(racket.mount_position),
                "mount_wxyz": _floats(racket.mount_wxyz),
                "connector_mass": float(racket.connector_mass),
            }
        )
    else:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r1d_racket")
        if body_id != -1:
            section["legacy_body_mass"] = float(model.body_mass[body_id])
            section["legacy_body_inertia"] = _floats(model.body_inertia[body_id])
    return section


def _build_report(
    profile: RobotProfile,
    bundle: RobotModelBundle,
    robot_config: Path,
    runtime_config: Path,
    *,
    render_requested: bool,
    view_requested: bool,
    racket_config: Path | None = None,
    racket_profile: RacketProfile | None = None,
) -> dict[str, Any]:
    """组装完整 JSON 报告（数组可序列化、未知 null、禁 NaN/Infinity）。

    Args:
        profile: 机器人描述。
        bundle: 同源组装结果（model/binding/parameters/racket）。
        robot_config: 描述 YAML 路径（复现命令用）。
        runtime_config: 运行参数 YAML 路径。
        render_requested: 是否请求了三视图渲染。
        view_requested: 是否请求了交互 viewer。
        racket_config: 可选球拍档案 YAML 路径（传入时进入 racket 节与哈希）。
        racket_profile: 可选球拍档案对象（唯一物理来源；None 为旧兼容模式）。

    Returns:
        报告字典（可直接 json.dumps(allow_nan=False)）。
    """
    parameters = bundle.parameters
    side = parse_side(profile)
    repo_root = _repo_root()
    model = bundle.model
    binding = bundle.binding

    # 初态 TCP 位姿（qdot=0）
    kin = ModelKinematics(bundle)
    n = model.nq
    kin.set_state(np.concatenate([parameters.init_q, np.zeros(n)]))
    tcp_pos, tcp_rot = kin.tcp_pose()
    quat, euler = _rotation_to_quat_and_euler(tcp_rot)
    q_L = _derive_lift_ql(model)

    # SDK 对照按完整关节名查询（防 profile 关节顺序与 SDK 表错位）
    sdk_by_name = {
        name: _sdk_entry_for_joint(side, name) for name in profile.active_joint_names
    }

    per_joint = [
        {
            "name": name,
            "joint_id": int(jid),
            "qpos_index": int(qp),
            "qvel_index": int(qv),
            "actuator_id": int(aid),
            "actuator_name": actuator,
            "sdk_reference_index": sdk_by_name[name]["sdk_index"],
        }
        for name, jid, qp, qv, aid, actuator in zip(
            profile.active_joint_names,
            binding.joint_ids,
            binding.qpos_indices,
            binding.qvel_indices,
            binding.actuator_ids,
            profile.actuator_names,
        )
    ]

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "robot_id": profile.robot_id,
        "side": side,
        "invocation": {
            "robot_config": str(robot_config),
            "runtime_config": str(runtime_config),
            "render": bool(render_requested),
            "view": bool(view_requested),
            "reproduce_hint": (
                "python scripts/tools/inspect_robot_model.py --robot-config "
                f"{robot_config} --runtime-config {runtime_config} "
                + (
                    f"--racket-config {racket_config} "
                    if racket_config is not None
                    else ""
                )
                + "--output-dir <目录>"
            ),
        },
        "profile": {
            "schema_version": profile.schema_version,
            "active_joint_names": list(profile.active_joint_names),
            "held_joint_names": list(profile.held_joint_names),
            "actuator_names": list(profile.actuator_names),
            "tcp_site_name": profile.tcp_site_name,
            "model_path": str(profile.model_path),
        },
        "racket": _racket_report_section(
            bundle, racket_config=racket_config, racket_profile=racket_profile
        ),
        "model": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "njnt": int(model.njnt),
            "timestep_s": float(parameters.dt),
            "actuator_binding_note": (
                "motor 仅名称绑定（gear=1），无 ctrlrange/forcerange——"
                "URDF effort/velocity 声明值未经实机验证，不作安全限值（N0-9）"
            ),
        },
        "binding": {
            "joint_ids": [int(v) for v in binding.joint_ids],
            "qpos_indices": [int(v) for v in binding.qpos_indices],
            "qvel_indices": [int(v) for v in binding.qvel_indices],
            "actuator_ids": [int(v) for v in binding.actuator_ids],
            "held_qpos_indices": [int(v) for v in binding.held_qpos_indices],
            "held_qvel_indices": [int(v) for v in binding.held_qvel_indices],
            "tcp_site_id": int(binding.tcp_site_id),
            "per_joint": per_joint,
        },
        "initial_state": {
            "init_q": _floats(parameters.init_q),
            "held_q": _floats(parameters.held_q),
            "qdot_assumed_rad_s": [0.0] * n,
            "dt": float(parameters.dt),
        },
        "tcp_pose": {
            "site": profile.tcp_site_name,
            "state": "init_q（qdot=0）下的世界位姿",
            "position_xyz_m": _floats(tcp_pos),
            "quaternion_wxyz": quat,
            "euler_rpy_rad": euler,
        },
        "joint_limits": {
            "source": "model.jnt_range（load_robot_model 回填至 parameters）",
            "q_lower": _floats(parameters.q_lower),
            "q_upper": _floats(parameters.q_upper),
            "unknown_limits": {
                "kp": None if parameters.kp is None else _floats(parameters.kp),
                "kd": None if parameters.kd is None else _floats(parameters.kd),
                "qdot_max": (
                    None if parameters.qdot_max is None
                    else _floats(parameters.qdot_max)
                ),
                "qddot_max": (
                    None if parameters.qddot_max is None
                    else _floats(parameters.qddot_max)
                ),
                "torque_max": (
                    None if parameters.torque_max is None
                    else _floats(parameters.torque_max)
                ),
                "max_tcp_speed": (
                    None if parameters.max_tcp_speed is None
                    else float(parameters.max_tcp_speed)
                ),
            },
            "unknown_limits_note": (
                "整组未知（null）：不从 RM65 补值；URDF effort/velocity 声明值"
                "未经实机验证，不自动成为运行限值（N0-9），安全限值留待阶段 3/4"
            ),
        },
        "provisional_fields": list(parameters.provisional_fields),
        "frozen_components": {
            "wheels": {
                "handling": "landing_visual",
                "description": (
                    "轮×2（Left/Right_Wheel_joint，continuous）作为底座落地视觉"
                    "geom 写入 base_link（r1d_wheel_left/right，contype=0），"
                    "不进入主动关节链"
                ),
            },
            "casters": {
                "handling": "landing_visual",
                "description": (
                    "4 组被动脚轮（r1d_caster_0..3，sphere 视觉）与轮共同决定"
                    "底座落地高度 0.155 m；无 joint/无碰撞（contype=0）"
                ),
            },
            "lift": {
                "handling": "baked_default_plus_runtime_offset",
                "q_L_m": float(q_L),
                "derivation": (
                    "r1d_env.xml 恒定 bake 默认 q_L=0.4225（两级 prismatic 同值）；"
                    "运行时按 scene.lift_displacement 在模板副本内存中重烘，"
                    "磁盘模板字节不变。q_L 由模型 body_pos 推导："
                    "z(stage1)+z(stage2)=2·q_L−0.1688（审计 §4）"
                ),
                "scene_lift_displacement_m": float(
                    bundle.parameters.scene.lift_displacement
                )
                if bundle.parameters.scene is not None
                else None,
                "travel_range_m_per_stage": list(_LIFT_TRAVEL_RANGE_M),
                "total_travel_m": 2.0 * (_LIFT_TRAVEL_RANGE_M[1] - _LIFT_TRAVEL_RANGE_M[0]),
            },
            "waist": {
                "handling": "frozen_at_0",
                "value_rad": 0.0,
                "description": (
                    "waist_yaw_joint 冻结 0；指南 ±0.6109 vs URDF ±2.618 冲突"
                    "见 data_conflicts 第 1 项（两集合公共值，不选边）"
                ),
            },
            "head": {
                "handling": "frozen_static_in_env_fragment",
                "description": (
                    "head_pitch_Link/head_yaw_Link 作为无关节静态链进入 "
                    "r1d_env.xml（冻结 0，仅完整外观；网格随片段自带 <asset>）"
                ),
            },
            "opposite_arm": {
                "handling": "frozen_static_body",
                "description": (
                    "对侧臂按 r1d_arm_{passive}.xml 烘焙固定姿态后删除全部关节"
                    "/法兰 site，作为静态 body 挂在 waist_yaw_link 下（仅渲染；"
                    "scene.fixed_joint_poses 可重烘，缺省为 passive 默认姿态）"
                ),
            },
        },
        "data_conflicts": _DATA_CONFLICTS,
        "sdk_index_reference": {
            "annotation": (
                "通信地址非 MuJoCo 状态地址：以下为宇树整机 DDS LowState "
                "motor_state[] 通信地址（仅作只读对照），不进入 MuJoCo 模型、"
                "不进入 ModelBinding（共享约束 N0-8）"
            ),
            "source": f"{AUDIT_REPORT_REL} §2",
            "selected_side_joints": [
                {
                    "semantic": sdk_by_name[name]["semantic"],
                    "joint_name": name,
                    "sdk_name": sdk_by_name[name]["sdk_name"],
                    "sdk_index": sdk_by_name[name]["sdk_index"],
                }
                for name in profile.active_joint_names
            ],
            "other_components": dict(sorted(_SDK_OTHER_COMPONENTS.items())),
            "unaddressed_note": "索引 2–11 空（被动脚轮未编址）、14 空（审计 §2）",
            "sdk_enum_warning": (
                "unitree_sdk2 defines.h 的 JointIndex 枚举是带双腿的 R1 布局，"
                "缺 A7 特有的 20/21/27/28 且 12 号索引含义不同；底盘/腰/升降段"
                "索引不能从 defines.h 推导（审计 §2 SDK 枚举警告）"
            ),
        },
        "resource_hashes": _resource_hashes(
            profile,
            robot_config,
            runtime_config,
            repo_root,
            racket_config=racket_config,
        ),
        "official_source": {
            "repo_url": "https://github.com/unitreerobotics/unitree_ros",
            "commit": PINNED_COMMIT,
            "urdf_sha256": PINNED_URDF_SHA256,
        },
        "versions": {
            "mujoco": mujoco.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    return report


def run_inspection(
    robot_config: Path,
    runtime_config: Path,
    output_dir: Path,
    *,
    render: bool = False,
    view: bool = False,
    racket_config: Path | None = None,
) -> tuple[dict[str, Any], Path, RobotModelBundle]:
    """执行无头检查并写出 JSON 报告（渲染/viewer 由 main 单独编排）。

    Args:
        robot_config: 描述 YAML 路径（侧别由此显式表达）。
        runtime_config: 运行参数 YAML 路径。
        output_dir: 输出目录（不存在则创建）。
        render: 是否请求三视图（仅记录进 invocation，渲染在 JSON 写出后）。
        view: 是否请求交互 viewer（同上）。
        racket_config: 可选球拍档案 YAML（传入时球拍几何/质量/惯量/安装
            全部来自 RacketProfile；None 保持旧 API 兼容语义）。

    Returns:
        (报告字典, JSON 路径, 同源 bundle)。

    Raises:
        FileNotFoundError / ValueError / yaml.YAMLError: 配置异常
            （由 main 统一映射为退出码 2）。
    """
    profile = load_robot_profile(Path(robot_config))
    parameters = load_runtime_parameters(Path(runtime_config), profile)
    racket_profile = (
        load_racket_profile(Path(racket_config))
        if racket_config is not None
        else None
    )
    bundle = load_robot_model(profile, parameters, racket_profile=racket_profile)
    side = parse_side(profile)

    report = _build_report(
        profile,
        bundle,
        Path(robot_config),
        Path(runtime_config),
        render_requested=render,
        view_requested=view,
        racket_config=Path(racket_config) if racket_config is not None else None,
        racket_profile=racket_profile,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"inspect_r1d_{side}.json"
    # allow_nan=False：任何 NaN/Infinity 在此处抛 ValueError，不落盘脏数据
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    json_path.write_text(text + "\n", encoding="utf-8")
    logger.info("JSON 报告已写出: %s", json_path)
    return report, json_path, bundle


# ---------------------------------------------------------------- 渲染


def render_views(bundle: RobotModelBundle, side: str, output_dir: Path) -> list[Path]:
    """渲染 front/side/oblique 三视图 PNG（MuJoCo offscreen，无平台 GL 调用）。

    只用 ``mujoco.Renderer`` 与 PIL 写文件；不 import OpenGL/osmesa 等
    平台特定模块（GL 后端由 MuJoCo 自行选择）。

    Args:
        bundle: 同源组装结果（模型按 parameters 组装）。
        side: 侧别（输出文件名用）。
        output_dir: 输出目录。

    Returns:
        三张 PNG 路径列表（front/side/oblique 顺序）。
    """
    model = bundle.model
    data = mujoco.MjData(model)
    qpos_indices = np.asarray(bundle.binding.qpos_indices, dtype=int)
    data.qpos[qpos_indices] = bundle.parameters.init_q
    mujoco.mj_forward(model, data)

    width, height = _RENDER_SIZE_PX
    # 扩大本运行实例的 offscreen framebuffer（默认 640×480 不够 800×600）。
    # 只改内存中的模型对象，不改磁盘模板。
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    renderer = mujoco.Renderer(model, height=height, width=width)
    written: list[Path] = []
    try:
        for view_name, lookat, azimuth, elevation in _RENDER_VIEWS:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = lookat
            camera.azimuth = azimuth
            camera.elevation = elevation
            camera.distance = 3.0 if view_name == "oblique" else 2.4
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            png_path = output_dir / f"inspect_r1d_{side}_{view_name}.png"
            Image.fromarray(pixels).save(png_path)
            written.append(png_path)
            logger.info("三视图已写出: %s", png_path)
    finally:
        renderer.close()
    return written


def launch_viewer(bundle: RobotModelBundle) -> None:
    """启动交互 viewer（阻塞直至用户关闭窗口）。

    ``mujoco.viewer`` 延迟导入：无头缺省路径不加载 viewer 模块。

    Args:
        bundle: 同源组装结果（按 init_q 展示准备姿态）。
    """
    import mujoco.viewer  # 延迟导入：仅 --view 时加载

    model = bundle.model
    data = mujoco.MjData(model)
    qpos_indices = np.asarray(bundle.binding.qpos_indices, dtype=int)
    data.qpos[qpos_indices] = bundle.parameters.init_q
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 30.0)


# ---------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（三个必传参数 + 两个可选开关）。"""
    parser = argparse.ArgumentParser(
        description=(
            "R1D 模型预览与可追溯参数导出：无头检查输出 JSON 报告"
            "（含侧别/绑定地址/冻结部件/冲突清单/SDK 对照/资源哈希），"
            "可选三视图 PNG 与交互 viewer"
        )
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        required=True,
        help=(
            "机器人描述 YAML（configs/robots/r1d_left.yaml 或 r1d_right.yaml，"
            "必传——侧别由此显式表达，无默认侧）"
        ),
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        required=True,
        help="运行参数 YAML（对应侧 r1d_runtime_*.yaml，必传）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录（JSON 报告与三视图 PNG，必传）",
    )
    parser.add_argument(
        "--racket-config",
        type=Path,
        default=None,
        help=(
            "可选球拍档案 YAML（configs/rackets/*.yaml，如 "
            "blade100_v9_reference.yaml）；提供时球拍几何/质量/惯量/安装/"
            "连接件全部来自档案，报告含 racket 节；缺省保持旧 API 兼容语义"
        ),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="渲染 front/side/oblique 三视图 PNG（MuJoCo offscreen）",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="启动交互 viewer（缺省无头检查，不启动 viewer）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 命令行参数（None 时取 sys.argv[1:]）。

    Returns:
        进程退出码：0 成功；2 异常配置；3 渲染/viewer 失败（JSON 保留）。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    # ---- 阶段 1：无头检查 + JSON 写出（失败 → 退出码 2，不产出 JSON）----
    try:
        report, json_path, bundle = run_inspection(
            args.robot_config,
            args.runtime_config,
            args.output_dir,
            render=args.render,
            view=args.view,
            racket_config=args.racket_config,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logger.error("配置错误（退出码 2）: %s", exc)
        return EXIT_CONFIG

    # CLI 工具面向操作员的摘要输出（AGENTS.md 对 scripts/ 工具豁免 print）
    print(f"robot_id={report['robot_id']} side={report['side']}")
    print(
        f"nq={report['model']['nq']} nv={report['model']['nv']} "
        f"nu={report['model']['nu']}"
    )
    print(f"TCP({report['tcp_pose']['site']}) 位置: {report['tcp_pose']['position_xyz_m']}")
    print(f"冻结: 升降 q_L={report['frozen_components']['lift']['q_L_m']} m, 腰=0, 头=0")
    racket = report["racket"]
    if racket["mode"] == "profile":
        print(
            f"球拍档案: {racket['racket_id']} "
            f"(总质量 {racket['total_mass']} kg, "
            f"tcp_local={racket['tcp_local_position']}, "
            f"sha256={str(racket['config_sha256'])[:12]}…)"
        )
    else:
        print("球拍: 旧 API 兼容模式（未传 --racket-config）")
    print(f"数据冲突: {len(report['data_conflicts']['items'])} 项（审计 §5 引用）")
    print(f"JSON 报告: {json_path}")

    # ---- 阶段 2：渲染/viewer（JSON 已落盘，失败 → 非零退出但保留 JSON）----
    exit_code = EXIT_OK
    side = report["side"]
    if args.render:
        try:
            pngs = render_views(bundle, side, args.output_dir)
            for png in pngs:
                print(f"三视图: {png}")
        except Exception as exc:  # noqa: BLE001 — 任意 GL/渲染失败统一降级
            logger.error("渲染失败（退出码 %d，JSON 已保留: %s）: %s", EXIT_RENDER, json_path, exc)
            exit_code = EXIT_RENDER
    if args.view:
        try:
            launch_viewer(bundle)
        except Exception as exc:  # noqa: BLE001 — viewer 失败统一降级
            logger.error("viewer 启动失败（退出码 %d，JSON 已保留: %s）: %s", EXIT_RENDER, json_path, exc)
            exit_code = EXIT_RENDER
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
