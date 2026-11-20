"""独立 URDF FK 对照器（Task 2.6 专用测试基准，不进运行代码）。

独立性契约（评审重点核查项）：
- 本模块**不得** import ``src/`` 下任何模块、不得 import ``mujoco``、
  不得 import ``model_factory``；只允许 numpy 与 Python 标准库
  （``xml.etree``/``pathlib``/``re`` 等）。因此其 FK 数值是与被测
  MuJoCo 模型完全独立的"整机 URDF 数值证据"。
- 关节链逐字解析自 ``assets/unitree_r1d/r1d.urdf``（固定 commit
  ``157ff950`` 的官方原文），按 URDF 变换约定累乘：
  ``T_parent_child = Origin(xyz, rpy) @ JointMotion(axis, q)``，
  其中 RPY 固定轴约定为 ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``。
- 冻结策略与派生模型一致（审计报告 §4）：
  * 轮/脚轮/头/对侧臂不进入链；
  * 双级升降 prismatic 按同一 ``LIFT_QL`` bake（当前 ``r1d_env.xml``
    bake q_L=0.4225，对应 base_z=0.155 m 的落地高度，waist 世界 z 仍为
    1.16 m）；
  * 腰 ``waist_yaw_joint`` 冻结 0；
  * 仅该侧 7 个臂关节取输入 q。
- 场景数值（r1d_mount 的 ``base_position``/``base_quaternion`` 与
  球拍 ``racket_position``/``racket_quaternion`` 相对 r1d_flange）从
  ``configs/robots/r1d_runtime_{side}.yaml`` 的 scene 节读取——scene
  是输入不是被测对象；用行级极简解析避免引入 PyYAML 依赖。
- 末端 TCP = wrist_yaw_link → ``{side}_dex1_base_joint`` origin
  （= r1d_flange site）→ 平移 racket_position → 旋转 racket_quaternion，
  与 2.5b 模板挂接结构对应；MuJoCo 侧用 tcp_pose()（racket_center site）。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import numpy as np

# 仓库根（tests/ 上一级）
REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = REPO_ROOT / "assets" / "unitree_r1d" / "r1d.urdf"
RUNTIME_PATHS: dict[str, Path] = {
    side: REPO_ROOT / "configs" / "robots" / f"r1d_runtime_{side}.yaml"
    for side in ("left", "right")
}

# 双级升降的 bake 值（m）。落地方案 base_z=0.155 m 时，取 q_L=0.4225
# 使 waist 世界高度仍为 1.16 m（原 base_z=1.0、q_L=0 的肩高不变）。
LIFT_QL = 0.4225

# 腰关节冻结值（rad，审计 §4 冻结策略：两限位集合公共值 0）
WAIST_QL = 0.0

# 从 base_link 到该侧 r1d_flange 的关节链（按父子顺序逐项查 URDF 原文；
# {side} 占位在构造时展开）。臂段 7 关节即审计 §2 表顺序。
_CHAIN_JOINTS: tuple[str, ...] = (
    "lift_mount_joint",
    "lift_stage1_joint",
    "lift_stage2_joint",
    "upper_body_mount_joint",
    "waist_yaw_joint",
    "{side}_shoulder_pitch_joint",
    "{side}_shoulder_roll_joint",
    "{side}_shoulder_yaw_joint",
    "{side}_elbow_joint",
    "{side}_wrist_roll_joint",
    "{side}_wrist_pitch_joint",
    "{side}_wrist_yaw_joint",
    "{side}_dex1_base_joint",
)

# scene 节中需要读取的向量字段（本对照器只用到位姿四项）
_SCENE_VECTOR_FIELDS: tuple[str, ...] = (
    "base_position",
    "base_quaternion",
    "racket_position",
    "racket_quaternion",
)

# 行级 YAML 解析用：匹配 "key: [a, b, c]"（值区允许科学计数法）
_SCENE_ITEM_RE = re.compile(
    r"^\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
    r"\[(?P<values>[^\]]*)\]\s*$"
)


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF 固定轴 RPY 旋转矩阵：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。

    Args:
        rpy: (3,) roll/pitch/yaw（rad）。

    Returns:
        3×3 旋转矩阵。
    """
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _axis_angle_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rodrigues 公式的轴角旋转矩阵（axis 在 URDF 中为单位标准基向量）。"""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    k = np.array(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _quaternion_matrix(quat: Sequence[float]) -> np.ndarray:
    """wxyz 单位四元数 → 3×3 旋转矩阵。"""
    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _vec3(text: str | None, default: Sequence[float]) -> np.ndarray:
    """把空格分隔的三元数值串解析为 (3,) 数组，缺省用 default。"""
    if text is None:
        return np.array(default, dtype=np.float64)
    values = np.array([float(tok) for tok in text.split()], dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"期望 3 个数值，实际 {values.shape}: {text!r}")
    return values


def _make_transform(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """构造 4×4 齐次变换。"""
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


def _apply_transform(t: np.ndarray, rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """齐次变换复合：t ← t @ [rot, pos]。"""
    new_rot = t[:3, :3] @ rot
    new_pos = t[:3, :3] @ pos + t[:3, 3]
    return _make_transform(new_rot, new_pos)


def _parse_scene_vectors(path: Path) -> dict[str, np.ndarray]:
    """行级极简解析 runtime YAML 的 scene 节向量字段。

    只识别 ``key: [a, b, c]`` 行（值区去掉 ``#`` 注释后解析），
    避免引入 PyYAML 依赖以维持本模块"仅标准库 + numpy"的独立性契约。

    Raises:
        ValueError: scene 节缺失或所需字段不全。
    """
    values: dict[str, np.ndarray] = {}
    in_scene = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            in_scene = line.strip() == "scene:"
            continue
        if not in_scene:
            continue
        match = _SCENE_ITEM_RE.match(line)
        if match is None or match.group("key") not in _SCENE_VECTOR_FIELDS:
            continue
        tokens = match.group("values").replace(",", " ").split()
        values[match.group("key")] = np.array(
            [float(tok) for tok in tokens], dtype=np.float64
        )
    missing = [key for key in _SCENE_VECTOR_FIELDS if key not in values]
    if missing:
        raise ValueError(f"runtime YAML scene 节缺少字段 {missing}（文件: {path}）")
    return values


class R1DUrdfFk:
    """从官方整机 URDF 独立计算 R1D 该侧 TCP 位姿的对照器。

    用法::

        fk = R1DUrdfFk("left")
        pos, rot = fk.tcp_pose(q)            # (3,) 位置 + 3×3 旋转（世界系）
        jacp, jacr = fk.tcp_jacobian(q)      # 中心差分雅可比（世界系）

    数值链：world→r1d_mount(scene YAML)→base_link→lift 两级(q_L bake)
    →pelvis→waist_yaw(0)→该侧 7 关节→dex1_base(r1d_flange)→球拍偏移
    （scene YAML 的 racket_position/racket_quaternion）。
    """

    def __init__(
        self,
        side: str,
        urdf_path: Path = URDF_PATH,
        *,
        base_position: Sequence[float] | None = None,
        base_quaternion: Sequence[float] | None = None,
        lift_ql: float = LIFT_QL,
    ) -> None:
        """解析 URDF 与该侧 runtime YAML scene 数值。

        Args:
            side: "left" 或 "right"。
            urdf_path: 官方整机 URDF 路径（默认 assets/unitree_r1d/r1d.urdf）。
            base_position: 可选底座世界位置覆盖 (3,)（默认取 runtime YAML）。
            base_quaternion: 可选底座姿态覆盖 wxyz (4,)（默认取 runtime YAML）。
            lift_ql: 双级升降单级 bake 值（m，默认 ``LIFT_QL``）；用于对照
                场景层显式升降位移重烘是否一致。

        Raises:
            ValueError: side 非法、URDF 关节链与预期不符（名称/父子关系
                缺失或断裂）。
        """
        if side not in RUNTIME_PATHS:
            raise ValueError(f"side 必须是 'left'/'right'，收到 {side!r}")
        self._side = side

        root = ET.parse(urdf_path).getroot()
        joints: dict[str, ET.Element] = {}
        for joint in root.findall("joint"):
            name = joint.attrib.get("name")
            if name is None:
                raise ValueError(f"URDF 存在无 name 的 joint（文件: {urdf_path}）")
            joints[name] = joint
        self._joints = joints

        # 逐字展开该侧链并校验父子连续性（从 base_link 到该侧 dex1_base）
        self._chain: list[ET.Element] = []
        expected_parent = "base_link"
        for template in _CHAIN_JOINTS:
            name = template.format(side=side)
            joint = joints.get(name)
            if joint is None:
                raise ValueError(f"URDF 缺少关节 '{name}'（文件: {urdf_path}）")
            parent = joint.find("parent")
            if parent is None or parent.attrib.get("link") != expected_parent:
                actual = None if parent is None else parent.attrib.get("link")
                raise ValueError(
                    f"关节 '{name}' 的父 link 应为 {expected_parent}，"
                    f"实际 {actual}（链断裂，文件: {urdf_path}）"
                )
            child = joint.find("child")
            if child is None or "link" not in child.attrib:
                raise ValueError(f"关节 '{name}' 缺少 child link（文件: {urdf_path}）")
            expected_parent = child.attrib["link"]
            self._chain.append(joint)

        # 该侧 7 个臂关节在链中的序号（q 向量 → 关节名映射）
        self._arm_joint_names = tuple(
            f"{side}_{suffix}" for suffix in (
                "shoulder_pitch_joint",
                "shoulder_roll_joint",
                "shoulder_yaw_joint",
                "elbow_joint",
                "wrist_roll_joint",
                "wrist_pitch_joint",
                "wrist_yaw_joint",
            )
        )
        for name in self._arm_joint_names:
            if name not in joints:
                raise ValueError(f"URDF 缺少臂关节 '{name}'（文件: {urdf_path}）")

        # scene 数值（base 挂载 + 球拍）——输入而非被测对象
        scene = _parse_scene_vectors(RUNTIME_PATHS[side])
        self._base_position = (
            scene["base_position"]
            if base_position is None
            else np.asarray(base_position, dtype=np.float64)
        )
        self._base_rotation = _quaternion_matrix(
            scene["base_quaternion"] if base_quaternion is None else base_quaternion
        )
        self._racket_position = scene["racket_position"]
        self._racket_rotation = _quaternion_matrix(scene["racket_quaternion"])
        self._lift_ql = float(lift_ql)

    def _joint_transform(self, joint: ET.Element, value: float) -> np.ndarray:
        """单个 URDF 关节的 4×4 变换：Origin(xyz, rpy) @ Motion(axis, q)。

        fixed 关节运动为恒等；revolute/continuous 为轴角旋转；
        prismatic 为沿 axis 的平移。
        """
        origin = joint.find("origin")
        xyz = _vec3(
            origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)
        )
        rpy = _vec3(
            origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)
        )
        jtype = joint.attrib.get("type", "")
        if jtype not in ("fixed", "revolute", "continuous", "prismatic"):
            raise ValueError(f"关节 {joint.attrib.get('name')} 类型未支持: {jtype}")
        axis_el = joint.find("axis")
        axis = _vec3(
            axis_el.attrib.get("xyz") if axis_el is not None else None,
            (1.0, 0.0, 0.0),
        )
        # URDF 约定：T_parent_child = Origin(xyz, rpy) @ Motion(axis, value)，
        # 即先平移 xyz、旋转 RPY，再在关节系内做轴角旋转/沿 axis 平移
        origin_rot = _rpy_matrix(rpy)
        if jtype == "fixed":
            # fixed 关节无运动（URDF 中其 axis 常为占位 "0 0 0"，不可归一化）
            rot, pos = origin_rot, xyz
        elif jtype == "prismatic":
            # 平移关节：沿关节系 axis 平移 value（升降两级 bake 为同一 q_L）
            rot, pos = origin_rot, xyz + origin_rot @ (value * axis)
        else:
            # 旋转关节（revolute/continuous）：origin 后接 axis-angle
            rot, pos = origin_rot @ _axis_angle_matrix(axis, value), xyz
        return _make_transform(rot, pos)

    def _flange_transform(self, q: Sequence[float]) -> np.ndarray:
        """世界系 → 该侧 r1d_flange（dex1_base_joint child 原点）的 4×4 变换。"""
        q_map = dict(zip(self._arm_joint_names, (float(v) for v in q)))
        frozen = {
            "lift_stage1_joint": self._lift_ql,
            "lift_stage2_joint": self._lift_ql,
            "waist_yaw_joint": WAIST_QL,
        }
        t = _make_transform(self._base_rotation, self._base_position)
        for joint in self._chain:
            name = joint.attrib["name"]
            value = q_map.get(name, frozen.get(name, 0.0))
            motion = self._joint_transform(joint, value)
            t = _apply_transform(t, motion[:3, :3], motion[:3, 3])
        return t

    def flange_pose(self, q: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """该侧法兰（``r1d_flange`` site 对应点）的世界位姿（独立 URDF FK）。

        Args:
            q: (7,) 该侧臂关节角（rad），顺序同 :meth:`tcp_pose`。

        Returns:
            (pos, rot)：(3,) 世界位置与 3×3 世界旋转矩阵。
        """
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape != (len(self._arm_joint_names),):
            raise ValueError(f"q 应为 ({len(self._arm_joint_names)},)，实际 {q_arr.shape}")
        t = self._flange_transform(q_arr)
        return t[:3, 3].copy(), t[:3, :3].copy()

    def tcp_pose(self, q: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """该侧 TCP（= MuJoCo racket_center site 对应点）的世界位姿。

        Args:
            q: (7,) 该侧臂关节角（rad），顺序为审计 §2 表
                （肩俯仰/肩侧摆/肩偏航/肘/腕滚转/腕俯仰/腕偏航）。

        Returns:
            (pos, rot)：(3,) 世界位置与 3×3 世界旋转矩阵。
        """
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape != (len(self._arm_joint_names),):
            raise ValueError(f"q 应为 ({len(self._arm_joint_names)},)，实际 {q_arr.shape}")
        t = self._flange_transform(q_arr)
        # r1d_flange → 拍面中心：平移 racket_position + 旋转 racket_quaternion
        tcp_pos = t[:3, :3] @ self._racket_position + t[:3, 3]
        tcp_rot = t[:3, :3] @ self._racket_rotation
        return tcp_pos, tcp_rot

    def tcp_jacobian(
        self, q: Sequence[float], eps: float = 1e-6
    ) -> tuple[np.ndarray, np.ndarray]:
        """TCP 雅可比的中心差分近似（独立于 MuJoCo 的解析雅可比）。

        Args:
            q: (7,) 该侧臂关节角（rad）。
            eps: 中心差分步长（rad），默认 1e-6。

        Returns:
            (jacp, jacr)：3×7 平移雅可比与 3×7 角速度雅可比（均世界系；
            jacr 列 = vee(dR/dq_i @ R^T) 反对称向量）。
        """
        n = len(self._arm_joint_names)
        q_arr = np.asarray(q, dtype=np.float64)
        pos0, rot0 = self.tcp_pose(q_arr)
        jacp = np.zeros((3, n))
        jacr = np.zeros((3, n))
        for i in range(n):
            q_plus = q_arr.copy()
            q_minus = q_arr.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            pos_p, rot_p = self.tcp_pose(q_plus)
            pos_m, rot_m = self.tcp_pose(q_minus)
            jacp[:, i] = (pos_p - pos_m) / (2.0 * eps)
            skew = ((rot_p - rot_m) / (2.0 * eps)) @ rot0.T
            jacr[:, i] = np.array(
                [skew[2, 1] - skew[1, 2], skew[0, 2] - skew[2, 0], skew[1, 0] - skew[0, 1]]
            ) * 0.5
        return jacp, jacr
