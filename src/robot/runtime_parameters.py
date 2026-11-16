"""机器人运行参数（RobotRuntimeParameters）— 数值配置的纯参数层。

负责类型定义、YAML 严格解析与维数/单位/能力校验。不导入 real、sim、
ilqt、MuJoCo 或真机 SDK；可调数值全部来自 configs/robots/*_runtime.yaml
或手构对象，二者共用同一套校验规则（validate_runtime_parameters）。

与 RobotProfile 的分工：profile 只管身份与语义名称；本模块管 dt、初始
状态、PD 增益与各项限值的显式数值。所有可空限值按**整组未知**表达
（None），不从 RM65 或模型默认值隐式补齐；D2 的关节位置限位由
model_factory 绑定模型后从 jnt_range 解析填回。

单位约定：m、s、rad、rad/s、rad/s²、N·m；四元数 wxyz。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn

import numpy as np
import yaml

from src.robot.robot_profile import RobotProfile

# §2.2 契约中的数组类型别名
Array = np.ndarray

# 当前支持的运行参数 schema 版本（必须为整数 1）
SUPPORTED_SCHEMA_VERSION = 1

# 四元数单位长度允许的最大范数误差（超过即拒绝，不做默默归一化）
_QUATERNION_NORM_TOLERANCE = 1e-8

# 运行参数文件允许的顶层字段（未知字段一律拒绝；source 不由 YAML 自报）
_RUNTIME_FIELDS = (
    "schema_version",
    "robot_id",
    "joint_names",
    "dt",
    "init_q",
    "held_q",
    "kp",
    "kd",
    "q_lower",
    "q_upper",
    "qdot_max",
    "qddot_max",
    "torque_max",
    "max_tcp_speed",
    "scene",
    "provisional_fields",
)

# 可空限值字段（整组未知：成对/成组提供或保持 None）
_NULLABLE_ARRAY_FIELDS = (
    "kp",
    "kd",
    "q_lower",
    "q_upper",
    "qdot_max",
    "qddot_max",
    "torque_max",
)

# scene 允许的键（与 SceneParameters 字段同名）
_SCENE_FIELDS = (
    "base_position",
    "base_quaternion",
    "racket_position",
    "racket_quaternion",
    "racket_radius",
    "racket_half_thickness",
    "racket_mass",
    "torso_position",
    "torso_half_size",
)

# provisional_fields 允许标记的字段名（数值字段 + scene 字段）
_PROVISIONAL_ALLOWED_NAMES = frozenset(
    {
        "dt",
        "init_q",
        "held_q",
        *_NULLABLE_ARRAY_FIELDS,
        "max_tcp_speed",
        *_SCENE_FIELDS,
    }
)

# simulation 预检要求完整（非 None）的限值字段
_SIMULATION_REQUIRED_FIELDS = (
    "q_lower",
    "q_upper",
    "qdot_max",
    "qddot_max",
    "torque_max",
    "max_tcp_speed",
)


@dataclass(frozen=True)
class SceneParameters:
    """场景几何参数 — 固定基座/球拍/机身的位置与形状描述。

    D2 场景组装使用；RM65 运行参数 scene 为 None，不覆写旧几何。

    Attributes:
        base_position: (3,) 世界系中的固定基座位置（m）。
        base_quaternion: (4,) 基座姿态，wxyz，须单位长度，零四元数拒绝。
        racket_position: (3,) 相对 d2_flange（link_7 末端法兰）的拍面中心（m）。
        racket_quaternion: (4,) 拍面姿态，wxyz，拍面局部 +Z 为法向。
        racket_radius: 球拍半径（m）。
        racket_half_thickness: 球拍半厚（m）。
        racket_mass: 球拍质量（kg），实心圆盘近似，不宣称实物惯量。
        torso_position: (3,) 简化机身 box 中心，相对固定基座（m）。
        torso_half_size: (3,) 简化机身 box 半尺寸（m）。
    """

    base_position: Array
    base_quaternion: Array
    racket_position: Array
    racket_quaternion: Array
    racket_radius: float
    racket_half_thickness: float
    racket_mass: float
    torso_position: Array
    torso_half_size: Array

    def __post_init__(self) -> None:
        """冻结数组为只读 float64 副本、标量转有限 float（防别名写入）。"""
        for name in (
            "base_position",
            "base_quaternion",
            "racket_position",
            "racket_quaternion",
            "torso_position",
            "torso_half_size",
        ):
            object.__setattr__(
                self, name, _readonly_float64(getattr(self, name), name)
            )
        for name in ("racket_radius", "racket_half_thickness", "racket_mass"):
            object.__setattr__(
                self, name, _finite_float(getattr(self, name), name)
            )


@dataclass(frozen=True)
class RobotRuntimeParameters:
    """机器人运行参数 — 显式数值配置，未知限值保持 None（整组未知）。

    Attributes:
        robot_id: 机器人身份标识，须与 profile.robot_id 一致。
        joint_names: 关节名，须与 profile.active_joint_names 逐项同序一致。
        dt: 计算时间步长（s），正数。
        init_q: (n,) 初始关节角（rad）。
        held_q: (len(held_joint_names),) 保持关节角（rad）。
        kp: (n,) 位置增益，未配置位置控制时 None。
        kd: (n,) 阻尼增益，与 kp 成对提供。
        q_lower: (n,) 关节位置下界；成对提供，D2 从模型解析前为 None。
        q_upper: (n,) 关节位置上界；成对提供，D2 从模型解析前为 None。
        qdot_max: (n,) 关节速度上限（rad/s）。
        qddot_max: (n,) 关节加速度上限（rad/s²）。
        torque_max: (n,) 力矩上限（N·m，对称幅值）。
        max_tcp_speed: TCP 速度上限（m/s）。
        scene: 场景几何参数；RM65 为 None，不覆写旧几何。
        provisional_fields: 标记为临时（非官方）取值的字段名元组。
        source: 参数文件路径或旧来源说明；不作为安全授权。
    """

    robot_id: str
    joint_names: tuple[str, ...]
    dt: float
    init_q: Array
    held_q: Array
    kp: Array | None
    kd: Array | None
    q_lower: Array | None
    q_upper: Array | None
    qdot_max: Array | None
    qddot_max: Array | None
    torque_max: Array | None
    max_tcp_speed: float | None
    scene: SceneParameters | None
    provisional_fields: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        """冻结数组为只读 float64 副本、标量转有限 float（防别名写入）。"""
        context = f"robot_id: {self.robot_id}"
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(
            self, "provisional_fields", tuple(self.provisional_fields)
        )
        object.__setattr__(self, "dt", _finite_float(self.dt, "dt", context))
        for name in ("init_q", "held_q"):
            object.__setattr__(
                self, name, _readonly_float64(getattr(self, name), name, context)
            )
        for name in _NULLABLE_ARRAY_FIELDS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, _readonly_float64(value, name, context)
                )
        if self.max_tcp_speed is not None:
            object.__setattr__(
                self,
                "max_tcp_speed",
                _finite_float(self.max_tcp_speed, "max_tcp_speed", context),
            )


def load_runtime_parameters(
    path: Path, profile: RobotProfile
) -> RobotRuntimeParameters:
    """从 YAML 文件加载运行参数并执行 kinematics 级校验。

    Args:
        path: 运行参数 YAML 文件路径（如 configs/robots/d2_runtime.yaml）。
        profile: 机器人描述，作为身份与关节顺序的校验基准。

    Returns:
        不可变的 RobotRuntimeParameters 实例，source 为该文件路径。

    Raises:
        FileNotFoundError: 参数文件不存在或不是常规文件。
        ValueError: 结构错误、未知/缺失字段、schema 版本不支持、
            类型不符或 kinematics 级校验失败；消息含字段名与文件路径。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"运行参数文件不存在或不是常规文件: {path}")

    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"运行参数文件必须是映射结构（文件: {path}）")

    # 未知字段检查（先于字段级校验，尽早暴露拼写错误；source 自报亦在此拒绝）。
    # 键统一转 str 再排序：YAML 顶层出现 int/str 等混合键类型时不抛 TypeError。
    unknown = sorted(map(str, set(raw) - set(_RUNTIME_FIELDS)))
    if unknown:
        raise ValueError(f"未知字段 {unknown}（文件: {path}）")
    for field in _RUNTIME_FIELDS:
        if field not in raw:
            raise ValueError(f"缺少字段 {field}（文件: {path}）")

    _require_schema_version(raw, path)

    params = RobotRuntimeParameters(
        robot_id=_require_str(raw, "robot_id", path),
        joint_names=_require_joint_names(raw, path),
        dt=_finite_float(raw["dt"], "dt", f"文件: {path}"),
        init_q=_require_float_array(raw, "init_q", path),
        held_q=_require_float_array(raw, "held_q", path),
        kp=_require_optional_float_array(raw, "kp", path),
        kd=_require_optional_float_array(raw, "kd", path),
        q_lower=_require_optional_float_array(raw, "q_lower", path),
        q_upper=_require_optional_float_array(raw, "q_upper", path),
        qdot_max=_require_optional_float_array(raw, "qdot_max", path),
        qddot_max=_require_optional_float_array(raw, "qddot_max", path),
        torque_max=_require_optional_float_array(raw, "torque_max", path),
        max_tcp_speed=_require_optional_scalar(raw, "max_tcp_speed", path),
        scene=_require_scene(raw, path),
        provisional_fields=_require_provisional_fields(raw, path),
        source=str(path),
    )
    # YAML 加载与手构对象共用同一套校验规则
    validate_runtime_parameters(params, profile, purpose="kinematics")
    return params


def validate_runtime_parameters(
    params: RobotRuntimeParameters,
    profile: RobotProfile,
    *,
    purpose: Literal["kinematics", "simulation"],
) -> None:
    """校验运行参数（手构对象与 YAML 加载共用同一规则）。

    Args:
        params: 待校验的运行参数。
        profile: 机器人描述，作为身份与关节顺序的校验基准。
        purpose: 校验目的。"kinematics" 允许速度/力矩/PD 未知，但检查
            维数、有限性、身份、关节顺序、初态与已提供限值；
            "simulation" 在此之上要求全部动态限值完整（仅完整性预检，
            通过不代表控制模式或驱动可用）。

    Raises:
        ValueError: 任一校验失败，消息含字段名、robot_id 与来源说明。
    """
    if purpose not in ("kinematics", "simulation"):
        raise ValueError(
            f"未知 purpose: {purpose}（robot_id: {profile.robot_id}，"
            f"仅支持 kinematics/simulation，没有 real 授权开关）"
        )
    context = f"robot_id: {profile.robot_id}，来源: {params.source}"

    # 身份必须与 profile 一致
    if params.robot_id != profile.robot_id:
        _reject(
            "robot_id",
            f"'{params.robot_id}' 与 profile.robot_id "
            f"'{profile.robot_id}' 不一致",
            context,
        )
    # 关节名必须与 profile 逐项同序一致（不广播、不截断）
    if tuple(params.joint_names) != tuple(profile.active_joint_names):
        _reject(
            "joint_names",
            f"{list(params.joint_names)} 与 profile.active_joint_names "
            f"{list(profile.active_joint_names)} 不一致（须逐项同序）",
            context,
        )
    # 时间步长必须为正
    if not params.dt > 0:
        _reject("dt", f"{params.dt} 必须为正数", context)

    n = profile.ndof
    _check_array(params.init_q, (n,), "init_q", context)
    _check_array(params.held_q, (len(profile.held_joint_names),), "held_q", context)
    for field in _NULLABLE_ARRAY_FIELDS:
        array = getattr(params, field)
        if array is not None:
            _check_array(array, (n,), field, context)

    # 增益与限值的符号约束
    if params.kp is not None and bool(np.any(params.kp <= 0)):
        _reject("kp", "必须逐元素为正数", context)
    if params.kd is not None and bool(np.any(params.kd < 0)):
        _reject("kd", "不允许负值", context)
    for field in ("qdot_max", "qddot_max", "torque_max"):
        array = getattr(params, field)
        if array is not None and bool(np.any(array <= 0)):
            _reject(field, "必须逐元素为正数", context)

    # 可空限值必须成对提供（整组未知或整组已知）
    _check_paired(params.kp is None, params.kd is None, "kp", "kd", context)
    _check_paired(
        params.q_lower is None, params.q_upper is None, "q_lower", "q_upper", context
    )

    # 已知位置限位：下界严格小于上界，且初态在界内
    if params.q_lower is not None and params.q_upper is not None:
        if bool(np.any(params.q_lower >= params.q_upper)):
            _reject("q_lower", "存在 q_lower >= q_upper 的关节", context)
        if bool(np.any(params.init_q < params.q_lower)) or bool(
            np.any(params.init_q > params.q_upper)
        ):
            _reject("init_q", "初始关节角超出 [q_lower, q_upper] 界限", context)

    if params.max_tcp_speed is not None and not params.max_tcp_speed > 0:
        _reject("max_tcp_speed", f"{params.max_tcp_speed} 必须为正数", context)

    _validate_scene(params.scene, context)

    # provisional_fields 只允许标记已知数值字段（防拼写错误静默生效）
    unknown_provisional = sorted(
        set(params.provisional_fields) - _PROVISIONAL_ALLOWED_NAMES
    )
    if unknown_provisional:
        _reject(
            "provisional_fields", f"含未知字段名 {unknown_provisional}", context
        )

    # simulation 完整性预检：全部动态限值必须有效
    if purpose == "simulation":
        missing = [
            field
            for field in _SIMULATION_REQUIRED_FIELDS
            if getattr(params, field) is None
        ]
        if missing:
            raise ValueError(
                f"simulation 预检失败：限值不完整，缺少 {missing}（{context}）；"
                f"未知限值不得进入仿真用途"
            )


def _validate_scene(scene: SceneParameters | None, context: str) -> None:
    """校验场景几何参数的形状、有限性、四元数单位性与正数项。

    Args:
        scene: 场景参数；None（RM65 风格）直接通过。
        context: 错误消息附带的身份与来源上下文。
    """
    if scene is None:
        return
    for field, shape in (
        ("base_position", (3,)),
        ("base_quaternion", (4,)),
        ("racket_position", (3,)),
        ("racket_quaternion", (4,)),
        ("torso_position", (3,)),
        ("torso_half_size", (3,)),
    ):
        _check_array(getattr(scene, field), shape, f"scene.{field}", context)
    _check_unit_quaternion(scene.base_quaternion, "scene.base_quaternion", context)
    _check_unit_quaternion(
        scene.racket_quaternion, "scene.racket_quaternion", context
    )
    for field in ("racket_radius", "racket_half_thickness", "racket_mass"):
        if not getattr(scene, field) > 0:
            _reject(f"scene.{field}", "必须为正数", context)
    if bool(np.any(scene.torso_half_size <= 0)):
        _reject("scene.torso_half_size", "必须逐元素为正数", context)


# ---------- 底层校验/转换辅助 ----------


def _reject(field: str, message: str, context: str) -> NoReturn:
    """抛出含字段名与上下文的 ValueError（统一错误格式）。

    Raises:
        ValueError: 始终抛出。
    """
    raise ValueError(f"字段 {field} {message}（{context}）")


def _finite_float(value: Any, field: str, context: str = "构造输入") -> float:
    """把标量转为有限 float；bool/非数值/NaN/Inf 一律拒绝。

    Args:
        value: 待转换的标量。
        field: 字段名（用于错误消息）。
        context: 错误消息附带的上下文（文件路径或 robot_id）。

    Returns:
        转换后的有限 float。

    Raises:
        ValueError: 值是 bool、非数值类型或 NaN/Inf。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject(field, "必须是数值标量（bool 不作为数值接受）", context)
    result = float(value)
    if not math.isfinite(result):
        _reject(field, "含 NaN/Inf", context)
    return result


def _readonly_float64(
    value: Any, field: str, context: str = "构造输入"
) -> Array:
    """把数组类输入复制为只读 float64 ndarray（不与输入别名）。

    Args:
        value: 待转换的 array-like。
        field: 字段名（用于错误消息）。
        context: 错误消息附带的上下文（文件路径或 robot_id）。

    Returns:
        独立复制的 float64 数组，writeable=False。

    Raises:
        ValueError: 输入无法转换为数值数组。
    """
    try:
        array = np.array(value, dtype=np.float64)  # np.array 总是复制
    except (TypeError, ValueError) as exc:
        _reject(field, f"无法转换为数值数组: {exc}", context)
    array.setflags(write=False)
    return array


def _check_array(
    array: Array, shape: tuple[int, ...], field: str, context: str
) -> None:
    """校验数组形状与元素有限性。

    Raises:
        ValueError: 形状不符或含 NaN/Inf。
    """
    if array.shape != shape:
        _reject(field, f"形状 {array.shape} 应为 {shape}", context)
    if not bool(np.all(np.isfinite(array))):
        _reject(field, "含 NaN/Inf", context)


def _check_paired(
    left_none: bool, right_none: bool, left: str, right: str, context: str
) -> None:
    """校验可空限值成对提供（同时有值或同时 None）。"""
    if left_none != right_none:
        _reject(
            f"{left}/{right}",
            "必须成对提供（同时有值或同时为 None，整组未知）",
            context,
        )


def _check_unit_quaternion(
    quaternion: Array, field: str, context: str
) -> None:
    """校验四元数（wxyz）为单位长度，误差超过容差即拒绝。

    Raises:
        ValueError: 范数偏离 1 超过 1e-8（含零四元数），不默默归一化。
    """
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > _QUATERNION_NORM_TOLERANCE:
        _reject(
            field,
            f"四元数范数 {norm} 偏离 1 超过 {_QUATERNION_NORM_TOLERANCE}"
            f"（拒绝而非默默归一化）",
            context,
        )


# ---------- YAML 解析辅助（错误消息均含文件路径） ----------


def _require_str(raw: dict[str, Any], field: str, path: Path) -> str:
    """校验非空字符串字段。"""
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        _reject(field, "必须是非空字符串", f"文件: {path}")
    return value


def _require_schema_version(raw: dict[str, Any], path: Path) -> int:
    """校验 schema_version 字段（bool 不视为合法整数，必须为 1）。"""
    value = raw.get("schema_version")
    if isinstance(value, bool) or not isinstance(value, int):
        _reject("schema_version", "必须是整数", f"文件: {path}")
    if value != SUPPORTED_SCHEMA_VERSION:
        _reject(
            "schema_version",
            f"不支持: {value}（当前支持 {SUPPORTED_SCHEMA_VERSION}）",
            f"文件: {path}",
        )
    return value


def _require_joint_names(raw: dict[str, Any], path: Path) -> tuple[str, ...]:
    """校验 joint_names：非空字符串列表、元素非空且不重复。"""
    value = raw.get("joint_names")
    if not isinstance(value, list) or any(
        not isinstance(v, str) for v in value
    ):
        _reject("joint_names", "必须是字符串列表", f"文件: {path}")
    if not value:
        _reject("joint_names", "不能为空列表", f"文件: {path}")
    names = tuple(value)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        _reject("joint_names", f"存在重复名称: {duplicates}", f"文件: {path}")
    if any(not n.strip() for n in names):
        _reject("joint_names", "含空字符串元素", f"文件: {path}")
    return names


def _require_float_array(raw: dict[str, Any], field: str, path: Path) -> Array:
    """校验必填数值数组字段（元素 bool/非数值/NaN/Inf 拒绝）。"""
    value = raw.get(field)
    if not isinstance(value, list):
        _reject(field, "必须是数值列表", f"文件: {path}")
    assert isinstance(value, list)  # 供类型收窄
    _check_yaml_float_elements(value, field, path)
    return _readonly_float64(value, field, f"文件: {path}")


def _require_optional_float_array(
    raw: dict[str, Any], field: str, path: Path
) -> Array | None:
    """校验可空数值数组字段（null → None，整组未知）。"""
    value = raw.get(field)
    if value is None:
        return None
    return _require_float_array(raw, field, path)


def _require_optional_scalar(
    raw: dict[str, Any], field: str, path: Path
) -> float | None:
    """校验可空数值标量字段（null → None）。"""
    value = raw.get(field)
    if value is None:
        return None
    return _finite_float(value, field, f"文件: {path}")


def _check_yaml_float_elements(
    value: list[Any], field: str, path: Path
) -> None:
    """校验 YAML 数组元素均为数值（bool 拒绝）且有限。"""
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        _reject(field, "必须是数值列表（bool/字符串不作为数值接受）", f"文件: {path}")
    if not all(math.isfinite(float(v)) for v in value):
        _reject(field, "含 NaN/Inf", f"文件: {path}")


def _require_scene(raw: dict[str, Any], path: Path) -> SceneParameters | None:
    """解析 scene 节：null → None；映射 → 严格按 SceneParameters 字段解析。"""
    value = raw.get("scene")
    if value is None:
        return None
    if not isinstance(value, dict):
        _reject("scene", "必须是映射结构或 null", f"文件: {path}")
    unknown = sorted(map(str, set(value) - set(_SCENE_FIELDS)))
    if unknown:
        _reject("scene", f"含未知键 {unknown}", f"文件: {path}")
    for field in _SCENE_FIELDS:
        if field not in value:
            _reject("scene", f"缺少键 {field}", f"文件: {path}")

    arrays: dict[str, Array] = {}
    for field in (
        "base_position",
        "base_quaternion",
        "racket_position",
        "racket_quaternion",
        "torso_position",
        "torso_half_size",
    ):
        element = value[field]
        if not isinstance(element, list):
            _reject(
                f"scene.{field}", "必须是数值列表", f"文件: {path}"
            )
        assert isinstance(element, list)  # 供类型收窄
        _check_yaml_float_elements(element, f"scene.{field}", path)
        arrays[field] = _readonly_float64(element, f"scene.{field}", f"文件: {path}")
    scalars = {
        field: _finite_float(value[field], f"scene.{field}", f"文件: {path}")
        for field in ("racket_radius", "racket_half_thickness", "racket_mass")
    }
    return SceneParameters(
        base_position=arrays["base_position"],
        base_quaternion=arrays["base_quaternion"],
        racket_position=arrays["racket_position"],
        racket_quaternion=arrays["racket_quaternion"],
        torso_position=arrays["torso_position"],
        torso_half_size=arrays["torso_half_size"],
        racket_radius=scalars["racket_radius"],
        racket_half_thickness=scalars["racket_half_thickness"],
        racket_mass=scalars["racket_mass"],
    )


def _require_provisional_fields(
    raw: dict[str, Any], path: Path
) -> tuple[str, ...]:
    """校验 provisional_fields：字符串列表、不重复、名字须为已知数值字段。"""
    value = raw.get("provisional_fields")
    if not isinstance(value, list) or any(
        not isinstance(v, str) for v in value
    ):
        _reject(
            "provisional_fields", "必须是字符串列表", f"文件: {path}"
        )
    names = tuple(value)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        _reject(
            "provisional_fields", f"存在重复名称: {duplicates}", f"文件: {path}"
        )
    unknown = sorted(set(names) - _PROVISIONAL_ALLOWED_NAMES)
    if unknown:
        _reject(
            "provisional_fields",
            f"含未知字段名 {unknown}（仅允许标记数值/场景字段）",
            f"文件: {path}",
        )
    return names
