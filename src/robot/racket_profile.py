"""球拍参数档案（RacketProfile）的加载与校验。

本模块是 R1D-V12 parity 中球拍物理参数的**唯一事实来源**：视觉几何、碰撞几何、
TCP（拍面几何中心）与刚体动力学参数都由同一份档案驱动，避免 T0 阶段
"XML 几何一套、硬编码惯量另一套"的不一致。

坐标与单位契约（SI）
--------------------
球拍局部系（``RacketProfile`` 内部所有位置/惯量均基于此系）：

* 原点：柄底（手柄末端，即球拍最靠近手腕的一端）；
* ``+Z``：长轴，指向拍头；
* ``+X``：拍面法向；
* ``+Y``：由右手系确定（``+Z × +X = +Y``）；
* TCP：拍面几何中心，``tcp == geometry.face.center``；
* 惯量：相对质心（COM）表达、轴与局部系对齐的 3×3 矩阵，单位 ``kg·m^2``。

安装变换 ``mount`` 描述法兰 → 柄底的刚体变换：
``p_flange = R(wxyz) @ p_local + position``，其中 ``wxyz`` 为从球拍局部系到法兰系的
单位四元数（``w`` 在前）。``connector`` 描述法兰与柄底之间的独立连接件刚体
（可缺省），其质量**不计入**球拍质量 ``mass``。

来源标注
--------
``provenance`` 把每个物理字段映射到 :class:`Provenance`，``source`` 取值：

* ``official``：厂商/官方规格给出；
* ``estimated``：由几何/物理近似估计，必须给出 ``method``；
* ``measured``：实测（本任务未使用）；
* ``legacy_model``：从既有 MuJoCo 模型 ``src/robot/rm65_model.xml`` 导出，
  必须给出 ``reference``。

加载器对非法输入（未知字段、缺字段、非正质量、非正定或违反三角不等式的惯量、
非单位四元数、拍面中心越界、TCP 与拍面中心不一致等）一律抛 :class:`ValueError`，
**不会静默重置为默认值**。仅在浮点舍入容差内（如 ``0.707107`` 四元数）做归一化。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, TypeAlias

import numpy as np
import yaml

# --------------------------------------------------------------------------------------
# 类型别名与模块级常量
# --------------------------------------------------------------------------------------

Vec3: TypeAlias = tuple[float, float, float]
Vec4: TypeAlias = tuple[float, float, float, float]
Matrix3: TypeAlias = tuple[Vec3, Vec3, Vec3]

#: 球拍配置目录（本模块按仓库布局定位，避免测试/脚本重复拼路径）
RACKET_CONFIG_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "configs" / "rackets"
#: 旧拍档案：由 ``src/robot/rm65_model.xml`` 的几何/惯量独立导出
LEGACY_RACKET_PROFILE_PATH: Final[Path] = RACKET_CONFIG_DIR / "rm65_legacy.yaml"
#: 参考成人拍档案：Wilson Blade 100 V9，几何/惯量为合理刚体近似
REFERENCE_RACKET_PROFILE_PATH: Final[Path] = RACKET_CONFIG_DIR / "blade100_v9_reference.yaml"

#: 支持的档案 schema 版本
SUPPORTED_SCHEMA_VERSION: Final[int] = 1
#: 四元数范数与 1 的允许误差（MuJoCo XML 只给 6 位小数，存在 ~3e-7 舍入误差）
QUAT_NORM_TOLERANCE: Final[float] = 1e-6
#: 拍面面积与 ``π·半宽·半长`` 的相对误差容限（参考拍面积由官方规格取整）
FACE_AREA_REL_TOLERANCE: Final[float] = 0.02
#: 长度/几何自洽性检查的绝对误差容限（米）
GEOMETRY_ABS_TOLERANCE: Final[float] = 1e-9
#: 旋转/对称性检查的绝对误差容限
SYMMETRY_ABS_TOLERANCE: Final[float] = 1e-9
#: 单位向量（连接件轴向）检查容限
UNIT_VECTOR_TOLERANCE: Final[float] = 1e-6

#: 允许的来源枚举
_ALLOWED_SOURCES: Final[frozenset[str]] = frozenset(
    {"official", "estimated", "measured", "legacy_model"}
)
#: 允许的拍面形状（椭圆薄板 / 实心椭球近似）
_ALLOWED_FACE_SHAPES: Final[frozenset[str]] = frozenset({"ellipse", "ellipsoid"})

_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "racket_id",
    "description",
    "geometry",
    "mass",
    "com",
    "inertia",
    "mount",
    "connector",
    "provenance",
)
_GEOMETRY_FIELDS: Final[tuple[str, ...]] = (
    "length",
    "tcp",
    "face_area",
    "face",
    "handle",
    "throat",
    "frame",
)
_FACE_FIELDS: Final[tuple[str, ...]] = ("shape", "center", "semi_axes")
_CAPSULE_FIELDS: Final[tuple[str, ...]] = ("shape", "start", "end", "radius")
_INERTIA_FIELDS: Final[tuple[str, ...]] = ("matrix",)
_MOUNT_FIELDS: Final[tuple[str, ...]] = ("position", "wxyz")
_CONNECTOR_FIELDS: Final[tuple[str, ...]] = (
    "shape",
    "center",
    "axis",
    "length",
    "radius",
    "mass",
    "com",
    "inertia",
)
_PROVENANCE_ENTRY_FIELDS: Final[tuple[str, ...]] = ("source", "units", "reference", "method")

#: 基础 provenance 键（不含空喉部/框体与连接件）
_BASE_PROVENANCE_KEYS: Final[tuple[str, ...]] = (
    "geometry.length",
    "geometry.tcp",
    "geometry.face",
    "geometry.face_area",
    "geometry.handle",
    "mass",
    "com",
    "inertia",
    "mount.position",
    "mount.wxyz",
)
#: 连接件存在时必需的 provenance 键
_CONNECTOR_PROVENANCE_KEYS: Final[tuple[str, ...]] = (
    "connector.geometry",
    "connector.mass",
    "connector.com",
    "connector.inertia",
)
#: 所有允许出现的 provenance 键（喉部/框体仅在非空时必需，但始终允许出现）
_ALLOWED_PROVENANCE_KEYS: Final[frozenset[str]] = frozenset(
    _BASE_PROVENANCE_KEYS
    + ("geometry.throat", "geometry.frame")
    + _CONNECTOR_PROVENANCE_KEYS
)
#: 各 provenance 键的 SI 单位（用于校验单位标注未被写错）
_EXPECTED_UNITS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "geometry.length": "m",
        "geometry.tcp": "m",
        "geometry.face": "m",
        "geometry.face_area": "m^2",
        "geometry.handle": "m",
        "geometry.throat": "m",
        "geometry.frame": "m",
        "mass": "kg",
        "com": "m",
        "inertia": "kg·m^2",
        "mount.position": "m",
        "mount.wxyz": "1",
        "connector.geometry": "m",
        "connector.mass": "kg",
        "connector.com": "m",
        "connector.inertia": "kg·m^2",
    }
)


# --------------------------------------------------------------------------------------
# 数据类（全部冻结，向量/矩阵用 tuple 表达以避免可变的 numpy 数组）
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """单个物理字段的来源标注。

    Attributes:
        source: 来源枚举，取值见模块 docstring。
        units: 该字段的 SI 单位字符串（如 ``"m"``、``"kg·m^2"``）。
        reference: 出处（URL/文件路径/规格页说明）；``estimated`` 时可为空。
        method: 估计方法；仅有 ``estimated`` 时强制非空。
    """

    source: str
    units: str
    reference: str | None
    method: str | None


@dataclass(frozen=True)
class Capsule:
    """胶囊体（手柄/喉部/框体线段的统一表示）。

    Attributes:
        start: 胶囊轴线起点（局部系，米）。
        end: 胶囊轴线终点（局部系，米）。
        radius: 胶囊半径（米），必须为正。
    """

    start: Vec3
    end: Vec3
    radius: float


@dataclass(frozen=True)
class FaceGeometry:
    """拍面几何（椭圆薄板或实心椭球近似，视觉/碰撞/TCP 同源）。

    Attributes:
        shape: ``"ellipse"``（薄板近似）或 ``"ellipsoid"``（实心椭球近似）。
        center: 拍面几何中心（= TCP），必须位于局部 +Z 轴上。
        semi_axes: 半轴 ``(半厚, 半宽, 半长)``，单位米。
    """

    shape: str
    center: Vec3
    semi_axes: Vec3


@dataclass(frozen=True)
class RacketGeometry:
    """球拍几何（全部基于柄底原点的局部系）。

    Attributes:
        length: 柄底到拍头顶端的全长（米）。
        tcp: 拍面几何中心（= ``face.center``），米。
        face: 拍面几何。
        face_area: 拍面面积（米²），必须与拍面半轴自洽。
        handle: 手柄胶囊（下端 tip 落在局部系原点）。
        throat: 喉部线段列表（可为空元组）。
        frame: 框体线段列表（可为空元组）。
    """

    length: float
    tcp: Vec3
    face: FaceGeometry
    face_area: float
    handle: Capsule
    throat: tuple[Capsule, ...]
    frame: tuple[Capsule, ...]


@dataclass(frozen=True)
class Inertia:
    """相对质心的 3×3 转动惯量矩阵（kg·m²，轴与球拍局部系对齐）。

    Attributes:
        matrix: 3×3 对称正定矩阵，行优先 tuple 表示。
    """

    matrix: Matrix3

    def as_matrix(self) -> np.ndarray:
        """返回惯量矩阵的 numpy 视图（3×3、float64）。

        Returns:
            与 ``matrix`` 等值的 ``numpy.ndarray``。
        """
        return np.asarray(self.matrix, dtype=np.float64)

    def principal_moments(self) -> Vec3:
        """返回按升序排列的主惯量（特征值）。

        Returns:
            主惯量 ``(λ0, λ1, λ2)``，单位 kg·m²。
        """
        eigenvalues = np.linalg.eigvalsh(self.as_matrix())
        return (float(eigenvalues[0]), float(eigenvalues[1]), float(eigenvalues[2]))


@dataclass(frozen=True)
class Connector:
    """法兰与柄底之间的独立连接件刚体参数（质量不计入球拍质量）。

    Attributes:
        shape: 形状标识，当前仅支持 ``"cylinder"``。
        center: 几何中心（局部系，米）。
        axis: 圆柱轴向单位向量（局部系）。
        length: 圆柱长度（米）。
        radius: 圆柱半径（米）。
        mass: 连接件质量（千克）。
        com: 连接件质心（局部系，米）。
        inertia: 连接件相对自身质心的惯量矩阵。
    """

    shape: str
    center: Vec3
    axis: Vec3
    length: float
    radius: float
    mass: float
    com: Vec3
    inertia: Inertia


@dataclass(frozen=True)
class Mount:
    """法兰 → 柄底的安装变换。

    Attributes:
        position: 柄底在法兰系中的位置（米）。
        wxyz: 从球拍局部系到法兰系的单位四元数（``w`` 在前）。
    """

    position: Vec3
    wxyz: Vec4

    def rotation_matrix(self) -> np.ndarray:
        """把安装四元数转为 3×3 旋转矩阵。

        Returns:
            ``R``，满足 ``p_flange = R @ p_local + position``。
        """
        return quat_wxyz_to_rotmat(self.wxyz)


@dataclass(frozen=True)
class RacketProfile:
    """一份完整的球拍参数档案（不可变）。

    Attributes:
        schema_version: 档案 schema 版本（当前为 1）。
        racket_id: 球拍标识（非空字符串）。
        description: 人类可读描述。
        geometry: 几何参数。
        mass: 球拍质量（千克，穿线状态）。
        com: 质心位置（局部系，米）。
        inertia: 相对质心的惯量矩阵。
        mount: 法兰 → 柄底安装变换。
        connector: 独立连接件参数；无连接件时为 ``None``。
        provenance: 字段路径 → 来源标注（只读映射）。
        source_path: 档案来源文件绝对路径。
    """

    schema_version: int
    racket_id: str
    description: str
    geometry: RacketGeometry
    mass: float
    com: Vec3
    inertia: Inertia
    mount: Mount
    connector: Connector | None
    provenance: Mapping[str, Provenance]
    source_path: Path


# --------------------------------------------------------------------------------------
# 公共工具函数
# --------------------------------------------------------------------------------------


def quat_wxyz_to_rotmat(wxyz: Sequence[float]) -> np.ndarray:
    """把四元数 ``(w, x, y, z)`` 转为 3×3 旋转矩阵。

    采用右乘列向量约定：``p_rotated = R @ p``。输入会先归一化，
    因此可接受 MuJoCo XML 中 6 位小数的近似单位四元数。

    Args:
        wxyz: 长度为 4 的四元数序列。

    Returns:
        3×3 正交旋转矩阵（行列式 +1，``float64``）。

    Raises:
        ValueError: 序列长度不为 4，或四元数范数为 0。
    """
    values = [float(value) for value in wxyz]
    if len(values) != 4:
        raise ValueError(f"四元数必须包含 4 个分量 (w, x, y, z)，实际为 {len(values)} 个")
    w, x, y, z = values
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("四元数范数不能为 0（无法构造旋转矩阵）")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_racket_profile(path: Path) -> RacketProfile:
    """加载并严格校验一份球拍参数档案（YAML）。

    Args:
        path: 球拍档案 YAML 路径。

    Returns:
        校验通过的不可变 :class:`RacketProfile`；``source_path`` 为绝对路径。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: YAML 无法解析，或任一字段违反 schema/物理/单位契约。
    """
    profile_path = Path(path)
    if not profile_path.is_file():
        raise FileNotFoundError(f"球拍档案不存在：{profile_path}")

    where = f"球拍档案 {profile_path.name}"
    try:
        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        # 统一契约：YAML 语法错误也必须以含路径的 ValueError 暴露，
        # 避免调用方意外收到 yaml 库的 ParserError/ScannerError。
        raise ValueError(f"{where} 不是合法的 YAML：{exc}") from exc
    data = _require_mapping(raw, where)
    _require_fields(data, _TOP_LEVEL_FIELDS, _TOP_LEVEL_FIELDS, where)

    schema_version = _require_int(data["schema_version"], f"{where}.schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"{where}.schema_version 必须为 {SUPPORTED_SCHEMA_VERSION}，实际为 {schema_version}"
        )
    racket_id = _require_text(data["racket_id"], f"{where}.racket_id")
    description = _require_text(data["description"], f"{where}.description")

    geometry = _require_geometry(data["geometry"], f"{where}.geometry")
    mass = _require_positive(data["mass"], f"{where}.mass")
    com = _require_vec3(data["com"], f"{where}.com")
    inertia = _require_inertia(data["inertia"], f"{where}.inertia")
    mount = _require_mount(data["mount"], f"{where}.mount")
    connector = _require_connector(data["connector"], f"{where}.connector")

    required_provenance_keys = list(_BASE_PROVENANCE_KEYS)
    if geometry.throat:
        required_provenance_keys.append("geometry.throat")
    if geometry.frame:
        required_provenance_keys.append("geometry.frame")
    if connector is not None:
        required_provenance_keys.extend(_CONNECTOR_PROVENANCE_KEYS)
    provenance = _require_provenance(
        data["provenance"],
        f"{where}.provenance",
        required_keys=tuple(required_provenance_keys),
    )

    return RacketProfile(
        schema_version=schema_version,
        racket_id=racket_id,
        description=description,
        geometry=geometry,
        mass=mass,
        com=com,
        inertia=inertia,
        mount=mount,
        connector=connector,
        provenance=provenance,
        source_path=profile_path.resolve(),
    )


# --------------------------------------------------------------------------------------
# 基础校验工具
# --------------------------------------------------------------------------------------


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    """断言值是 YAML 映射。"""
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} 必须是映射（mapping），实际类型为 {type(value).__name__}")
    return value


def _require_fields(
    mapping: Mapping[str, Any],
    allowed: Sequence[str],
    required: Sequence[str],
    where: str,
) -> None:
    """断言映射只含允许字段且包含全部必需字段。"""
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ValueError(f"{where} 存在未知字段：{', '.join(unknown)}")
    missing = sorted(set(required) - set(mapping))
    if missing:
        raise ValueError(f"{where} 缺少必需字段：{', '.join(missing)}")


def _require_number(value: Any, where: str) -> float:
    """断言值是有限实数（拒绝 bool/字符串/NaN/Inf）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} 必须是数值，实际为 {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where} 必须是有限数，实际为 {number!r}")
    return number


def _require_positive(value: Any, where: str) -> float:
    """断言值是正的有限实数。"""
    number = _require_number(value, where)
    if number <= 0.0:
        raise ValueError(f"{where} 必须为正数，实际为 {number!r}")
    return number


def _require_int(value: Any, where: str) -> int:
    """断言值是整数（拒绝 bool/浮点）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} 必须是整数，实际为 {value!r}")
    return int(value)


def _require_text(value: Any, where: str) -> str:
    """断言值是非空字符串。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} 必须是非空字符串，实际为 {value!r}")
    return value


def _require_optional_text(value: Any, where: str) -> str | None:
    """断言值是非空字符串或 ``None``。"""
    if value is None:
        return None
    return _require_text(value, where)


def _require_vec3(value: Any, where: str) -> Vec3:
    """断言值是长度 3 的有限实数序列。"""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{where} 必须是长度 3 的数值列表，实际为 {value!r}")
    return (
        _require_number(value[0], f"{where}[0]"),
        _require_number(value[1], f"{where}[1]"),
        _require_number(value[2], f"{where}[2]"),
    )


def _require_vec4(value: Any, where: str) -> Vec4:
    """断言值是长度 4 的有限实数序列。"""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{where} 必须是长度 4 的数值列表，实际为 {value!r}")
    return (
        _require_number(value[0], f"{where}[0]"),
        _require_number(value[1], f"{where}[1]"),
        _require_number(value[2], f"{where}[2]"),
        _require_number(value[3], f"{where}[3]"),
    )


def _repeat(value: Any, where: str) -> Vec3:
    """断言值是长度 3 的有限实数序列并原样返回（语义化别名，便于阅读）。"""
    return _require_vec3(value, where)


# --------------------------------------------------------------------------------------
# 结构校验：胶囊 / 拍面 / 惯量 / 安装 / 连接件
# --------------------------------------------------------------------------------------


def _require_capsule(value: Any, where: str) -> Capsule:
    """解析并校验一个胶囊体（位置单位米，半径必须为正）。"""
    data = _require_mapping(value, where)
    _require_fields(data, _CAPSULE_FIELDS, _CAPSULE_FIELDS, where)
    shape = _require_text(data["shape"], f"{where}.shape")
    if shape != "capsule":
        raise ValueError(f"{where}.shape 必须为 'capsule'，实际为 {shape!r}")
    start = _require_vec3(data["start"], f"{where}.start")
    end = _require_vec3(data["end"], f"{where}.end")
    radius = _require_positive(data["radius"], f"{where}.radius")
    axis_length = math.dist(start, end)
    if axis_length <= 1e-12:
        raise ValueError(f"{where} 的 start 与 end 不能重合（胶囊轴线长度必须为正）")
    return Capsule(start=start, end=end, radius=radius)


def _require_face(value: Any, where: str) -> FaceGeometry:
    """解析并校验拍面几何（中心必须在 +Z 轴上，半轴必须为正）。"""
    data = _require_mapping(value, where)
    _require_fields(data, _FACE_FIELDS, _FACE_FIELDS, where)
    shape = _require_text(data["shape"], f"{where}.shape")
    if shape not in _ALLOWED_FACE_SHAPES:
        raise ValueError(
            f"{where}.shape 必须为 {sorted(_ALLOWED_FACE_SHAPES)} 之一，实际为 {shape!r}"
        )
    center = _repeat(data["center"], f"{where}.center")
    if abs(center[0]) > GEOMETRY_ABS_TOLERANCE or abs(center[1]) > GEOMETRY_ABS_TOLERANCE:
        raise ValueError(
            f"{where}.center 必须位于球拍 +Z 长轴上（x = y = 0），实际为 {center!r}"
        )
    semi_axes = _repeat(data["semi_axes"], f"{where}.semi_axes")
    if any(axis <= 0.0 for axis in semi_axes):
        raise ValueError(f"{where}.semi_axes 三个半轴都必须为正，实际为 {semi_axes!r}")
    return FaceGeometry(shape=shape, center=center, semi_axes=semi_axes)


def _require_handle(value: Any, where: str) -> Capsule:
    """解析手柄，并强制"柄底 = 局部系原点、长轴朝 +Z"的坐标契约。"""
    handle = _require_capsule(value, where)
    direction = np.asarray(handle.end, dtype=np.float64) - np.asarray(handle.start, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    if direction[2] <= 0.0:
        raise ValueError(f"{where}（手柄）长轴必须朝 +Z（指向拍头），实际方向为 {direction.tolist()}")
    butt_tip = np.asarray(handle.start, dtype=np.float64) - handle.radius * direction
    if np.linalg.norm(butt_tip) > GEOMETRY_ABS_TOLERANCE:
        raise ValueError(
            f"{where}（手柄）必须使柄底（胶囊下端 tip）落在球拍局部系原点，"
            f"实际 tip = {butt_tip.tolist()}"
        )
    return handle


def _require_capsule_sequence(
    value: Any, where: str, length: float
) -> tuple[Capsule, ...]:
    """解析喉部/框体线段序列，并校验端点落在 ``[0, length]`` 内。"""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{where} 必须是列表（可为空），实际为 {value!r}")
    segments: list[Capsule] = []
    for index, item in enumerate(value):
        segment_where = f"{where}[{index}]"
        segment = _require_capsule(item, segment_where)
        for label, point in (("start", segment.start), ("end", segment.end)):
            if not (-GEOMETRY_ABS_TOLERANCE <= point[2] <= length + GEOMETRY_ABS_TOLERANCE):
                raise ValueError(
                    f"{segment_where}.{label} 的 z 必须落在 [0, length={length}] 内，"
                    f"实际为 {point[2]}"
                )
        segments.append(segment)
    return tuple(segments)


def _require_inertia(value: Any, where: str) -> Inertia:
    """解析并校验 3×3 转动惯量矩阵（对称、正定、满足主惯量三角不等式）。"""
    data = _require_mapping(value, where)
    _require_fields(data, _INERTIA_FIELDS, _INERTIA_FIELDS, where)
    raw = data["matrix"]
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{where}.matrix 必须是 3×3 数值矩阵，实际为 {raw!r}")
    rows: list[Vec3] = []
    for row_index, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(
                f"{where}.matrix[{row_index}] 必须是长度 3 的数值列表，实际为 {row!r}"
            )
        rows.append(
            (
                _require_number(row[0], f"{where}.matrix[{row_index}][0]"),
                _require_number(row[1], f"{where}.matrix[{row_index}][1]"),
                _require_number(row[2], f"{where}.matrix[{row_index}][2]"),
            )
        )
    matrix: Matrix3 = (rows[0], rows[1], rows[2])
    array = np.asarray(matrix, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(array))))
    if float(np.max(np.abs(array - array.T))) > SYMMETRY_ABS_TOLERANCE * scale:
        raise ValueError(f"{where}.matrix 必须对称（转动惯量矩阵对称），实际为 {matrix!r}")
    eigenvalues = np.linalg.eigvalsh(array)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError(
            f"{where}.matrix 必须正定（主惯量全为正），实际特征值为 {eigenvalues.tolist()}"
        )
    if float(eigenvalues[2]) > float(eigenvalues[0] + eigenvalues[1]) + SYMMETRY_ABS_TOLERANCE * scale:
        raise ValueError(
            f"{where}.matrix 违反主惯量三角不等式（λ₂ ≤ λ₀ + λ₁），"
            f"实际特征值为 {eigenvalues.tolist()}"
        )
    return Inertia(matrix=matrix)


def _require_mount(value: Any, where: str) -> Mount:
    """解析并校验法兰 → 柄底安装变换（四元数容差内归一化）。"""
    data = _require_mapping(value, where)
    _require_fields(data, _MOUNT_FIELDS, _MOUNT_FIELDS, where)
    position = _require_vec3(data["position"], f"{where}.position")
    wxyz = _require_vec4(data["wxyz"], f"{where}.wxyz")
    norm = math.sqrt(sum(component * component for component in wxyz))
    if norm <= 1e-12:
        raise ValueError(f"{where}.wxyz 必须是单位四元数，实际范数为 0")
    if abs(norm - 1.0) > QUAT_NORM_TOLERANCE:
        raise ValueError(
            f"{where}.wxyz 必须是单位四元数（容差 {QUAT_NORM_TOLERANCE}），实际范数为 {norm}"
        )
    normalized: Vec4 = (
        wxyz[0] / norm,
        wxyz[1] / norm,
        wxyz[2] / norm,
        wxyz[3] / norm,
    )
    return Mount(position=position, wxyz=normalized)


def _require_connector(value: Any, where: str) -> Connector | None:
    """解析独立连接件刚体参数；``None`` 表示无连接件。"""
    if value is None:
        return None
    data = _require_mapping(value, where)
    _require_fields(data, _CONNECTOR_FIELDS, _CONNECTOR_FIELDS, where)
    shape = _require_text(data["shape"], f"{where}.shape")
    if shape != "cylinder":
        raise ValueError(f"{where}.shape 必须为 'cylinder'，实际为 {shape!r}")
    center = _repeat(data["center"], f"{where}.center")
    axis = _repeat(data["axis"], f"{where}.axis")
    axis_norm = math.sqrt(sum(component * component for component in axis))
    if abs(axis_norm - 1.0) > UNIT_VECTOR_TOLERANCE:
        raise ValueError(f"{where}.axis 必须是单位向量，实际范数为 {axis_norm}")
    length = _require_positive(data["length"], f"{where}.length")
    radius = _require_positive(data["radius"], f"{where}.radius")
    mass = _require_positive(data["mass"], f"{where}.mass")
    com = _repeat(data["com"], f"{where}.com")
    inertia = _require_inertia(data["inertia"], f"{where}.inertia")
    return Connector(
        shape=shape,
        center=center,
        axis=axis,
        length=length,
        radius=radius,
        mass=mass,
        com=com,
        inertia=inertia,
    )


# --------------------------------------------------------------------------------------
# 结构校验：几何整体 / 来源标注
# --------------------------------------------------------------------------------------


def _require_geometry(value: Any, where: str) -> RacketGeometry:
    """解析并校验球拍几何，并检查长度/TCP/拍面/面积/手柄的相互自洽。"""
    data = _require_mapping(value, where)
    _require_fields(data, _GEOMETRY_FIELDS, _GEOMETRY_FIELDS, where)
    length = _require_positive(data["length"], f"{where}.length")
    tcp = _require_vec3(data["tcp"], f"{where}.tcp")
    face = _require_face(data["face"], f"{where}.face")
    if any(
        abs(tcp[index] - face.center[index]) > GEOMETRY_ABS_TOLERANCE for index in range(3)
    ):
        raise ValueError(
            f"{where}.tcp 必须与 face.center 完全一致（TCP = 拍面几何中心），"
            f"tcp = {tcp!r}，face.center = {face.center!r}"
        )
    if abs(tcp[0]) > GEOMETRY_ABS_TOLERANCE or abs(tcp[1]) > GEOMETRY_ABS_TOLERANCE:
        raise ValueError(f"{where}.tcp 必须位于 +Z 长轴上（x = y = 0），实际为 {tcp!r}")
    half_length = face.semi_axes[2]
    if abs(length - (tcp[2] + half_length)) > GEOMETRY_ABS_TOLERANCE:
        raise ValueError(
            f"{where}.length 必须等于 face.center.z + face.semi_axes.z"
            f"（柄底到拍头顶端），实际 length = {length}，"
            f"计算值为 {tcp[2] + half_length}"
        )
    if tcp[2] - half_length < -GEOMETRY_ABS_TOLERANCE:
        raise ValueError(
            f"{where}.face.center 的 z 过小：拍面下端 (z = {tcp[2] - half_length}) 超出柄底以下"
        )
    face_area = _require_positive(data["face_area"], f"{where}.face_area")
    half_width = face.semi_axes[1]
    expected_area = math.pi * half_width * half_length
    relative_error = abs(face_area - expected_area) / expected_area
    if relative_error > FACE_AREA_REL_TOLERANCE:
        raise ValueError(
            f"{where}.face_area 与 face 半轴不自洽（π·半宽·半长 = {expected_area}），"
            f"标注面积为 {face_area}，相对误差 {relative_error:.4%} 超过 "
            f"{FACE_AREA_REL_TOLERANCE:.0%}"
        )
    handle = _require_handle(data["handle"], f"{where}.handle")
    throat = _require_capsule_sequence(data["throat"], f"{where}.throat", length)
    frame = _require_capsule_sequence(data["frame"], f"{where}.frame", length)
    return RacketGeometry(
        length=length,
        tcp=tcp,
        face=face,
        face_area=face_area,
        handle=handle,
        throat=throat,
        frame=frame,
    )


def _require_provenance(
    value: Any, where: str, required_keys: Sequence[str]
) -> Mapping[str, Provenance]:
    """解析来源标注，并校验来源/单位/method/reference 的强制关系。"""
    data = _require_mapping(value, where)
    _require_fields(data, sorted(_ALLOWED_PROVENANCE_KEYS), required_keys, where)
    result: dict[str, Provenance] = {}
    for key in sorted(data):
        entry_where = f"{where}.{key}"
        entry = _require_mapping(data[key], entry_where)
        _require_fields(
            entry, _PROVENANCE_ENTRY_FIELDS, _PROVENANCE_ENTRY_FIELDS, entry_where
        )
        source = _require_text(entry["source"], f"{entry_where}.source")
        if source not in _ALLOWED_SOURCES:
            raise ValueError(
                f"{entry_where}.source 必须为 {sorted(_ALLOWED_SOURCES)} 之一，"
                f"实际为 {source!r}"
            )
        units = _require_text(entry["units"], f"{entry_where}.units")
        expected_units = _EXPECTED_UNITS[key]
        if units != expected_units:
            raise ValueError(
                f"{entry_where}.units 必须为 {expected_units!r}（SI 单位契约），"
                f"实际为 {units!r}"
            )
        reference = _require_optional_text(entry["reference"], f"{entry_where}.reference")
        method = _require_optional_text(entry["method"], f"{entry_where}.method")
        if source == "estimated" and not method:
            raise ValueError(
                f"{entry_where}.method 不能为空（estimated 必须给出估计方法）"
            )
        if source != "estimated" and not reference:
            raise ValueError(
                f"{entry_where}.reference 不能为空（{source} 必须给出处）"
            )
        result[key] = Provenance(
            source=source, units=units, reference=reference, method=method
        )
    return MappingProxyType(result)
