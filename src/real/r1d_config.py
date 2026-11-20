"""R1-A7-D 硬件配置 — 加载、结构校验与 purpose 门槛（阶段 4 Task 4.2）。

纯数据层：只依赖 YAML/NumPy 与 RobotProfile 类型，不导入 SDK/MuJoCo、
不启动线程。实物校准（sign/offset）与实机限值在模板中初始 null，
null/provisional（未核验）一律不得通过 motion 门槛——未知校准不能启用运动。

字段与协议事实来源：configs/robots/r1d_protocol_manifest.yaml（Task 4.1 冻结）。
purpose 校验梯度：
- offline：结构可解析即可（模板即满足）；
- read_only：需有效通信参数（network_interface 非 null）与已核验布局；
- motion：在 read_only 之上要求全部硬件门槛（校准/限值已核验且非 null、
  控制周期/状态超时/诊断时长均为正有限值）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np
import yaml

from src.real.r1d_types import R1D_SLOT_COUNT
from src.robot.robot_profile import RobotProfile

# 本文件 schema 版本
R1D_HARDWARE_SCHEMA_VERSION = 1
# R1-A7-D 单臂关节数（七关节）
R1D_ARM_JOINT_COUNT = 7
# 合法侧别
R1D_SIDES = ("left", "right")
# 合法 purpose
R1D_PURPOSES = ("offline", "read_only", "motion")

# 顶层与各节允许字段（未知字段一律拒绝，防拼写错误静默生效）
_TOP_FIELDS = (
    "schema_version",
    "robot_id",
    "side",
    "communication",
    "layout",
    "calibration",
    "limits",
    "timing",
)
_COMMUNICATION_FIELDS = ("dds_domain", "network_interface", "lowcmd_topic", "lowstate_topic")
_LAYOUT_FIELDS = ("layout_verified", "joint_slots")
_CALIBRATION_FIELDS = ("verified", "joint_signs", "joint_offsets_rad")
_LIMIT_FIELDS = (
    "verified",
    "q_lower_rad",
    "q_upper_rad",
    "dq_limit_rad_s",
    "tau_limit_nm",
    "kp_nm_per_rad",
    "kd_nms_per_rad",
)
_TIMING_FIELDS = ("control_period_s", "state_timeout_s", "diagnostic_duration_s")


# ---------------------------------------------------------------------------
# 配置数据类（按名映射存 MappingProxyType，防调用方修改）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class R1DCommunicationConfig:
    """DDS 通信参数。

    Attributes:
        dds_domain: DDS domain 编号（示例固定 0）。
        network_interface: 网卡名；模板初始 null（待现场确认）。
        lowcmd_topic: 命令话题名（协议冻结 rt/lowcmd）。
        lowstate_topic: 状态话题名（协议冻结 rt/lowstate）。
    """

    dds_domain: int
    network_interface: str | None
    lowcmd_topic: str
    lowstate_topic: str


@dataclass(frozen=True)
class R1DLayoutConfig:
    """整机槽位布局（模型关节名 → SDK 槽位）。

    Attributes:
        layout_verified: 是否经现场核验（4.9 只读核对后置 true）。只读
            采集允许 false（首轮原始采集正是核对的数据来源，§3.1 勘误
            评审 2026-11-18）；motion（接管/运动）门槛强制 true。
        joint_slots: 关节名到 35 槽槽位的只读映射（按名称寻址，
            不依赖前缀或顺序猜测）。
    """

    layout_verified: bool
    joint_slots: Mapping[str, int]


@dataclass(frozen=True)
class R1DCalibrationConfig:
    """实物校准（模型 ↔ SDK 坐标换算）。

    Attributes:
        verified: 是否经实测/核验（provisional 示例值不得置 true）。
        joint_signs: 每关节方向符号（仅 ±1）；未知为 None。
        joint_offsets_rad: 每关节零位偏移（rad，SDK 侧）；未知为 None。
    """

    verified: bool
    joint_signs: Mapping[str, int] | None
    joint_offsets_rad: Mapping[str, float] | None


@dataclass(frozen=True)
class R1DLimitConfig:
    """实机限值（全部待实测来源，模板初始 null）。

    Attributes:
        verified: 是否经现场核验。
        q_lower_rad: 每关节位置下限（rad）。
        q_upper_rad: 每关节位置上限（rad）。
        dq_limit_rad_s: 每关节速度限值（rad/s）。
        tau_limit_nm: 每关节力矩限值（N·m）。
        kp_nm_per_rad: 每关节位置增益（N·m/rad）。
        kd_nms_per_rad: 每关节速度增益（N·m·s/rad）。
    """

    verified: bool
    q_lower_rad: Mapping[str, float] | None
    q_upper_rad: Mapping[str, float] | None
    dq_limit_rad_s: Mapping[str, float] | None
    tau_limit_nm: Mapping[str, float] | None
    kp_nm_per_rad: Mapping[str, float] | None
    kd_nms_per_rad: Mapping[str, float] | None


@dataclass(frozen=True)
class R1DTimingConfig:
    """时间参数（只接受正有限值或 null 待确认）。

    Attributes:
        control_period_s: 控制周期（秒）；官方示例 0.002（500 Hz）仅为
            示例事实，不是本项目频率承诺。
        state_timeout_s: 状态新鲜度超时（秒）。
        diagnostic_duration_s: 诊断时长（秒）。
    """

    control_period_s: float | None
    state_timeout_s: float | None
    diagnostic_duration_s: float | None


@dataclass(frozen=True)
class R1DHardwareConfig:
    """R1-A7-D 硬件配置顶层容器。

    Attributes:
        schema_version: schema 版本（当前 1）。
        robot_id: 机器人身份标识（须与 RobotProfile 一致）。
        side: 侧别（left/right，经文件路径显式表达，无隐式默认侧）。
        communication: DDS 通信参数。
        layout: 整机槽位布局。
        calibration: 实物校准。
        limits: 实机限值。
        timing: 时间参数。
    """

    schema_version: int
    robot_id: str
    side: str
    communication: R1DCommunicationConfig
    layout: R1DLayoutConfig
    calibration: R1DCalibrationConfig
    limits: R1DLimitConfig
    timing: R1DTimingConfig


# ---------------------------------------------------------------------------
# 加载与结构校验
# ---------------------------------------------------------------------------


def load_r1d_hardware_config(path: Path) -> R1DHardwareConfig:
    """从 YAML 文件加载 R1D 硬件配置并做结构校验。

    Args:
        path: 硬件配置 YAML 路径（如 configs/robots/r1d_hardware_left.yaml）。

    Returns:
        不可变的 R1DHardwareConfig（按名映射为 MappingProxyType 只读视图）。

    Raises:
        FileNotFoundError: 文件不存在或不是常规文件。
        ValueError: 未知字段、字段缺失/类型不符、槽位重复或越界、
            数量不是 7、sign 非 ±1、NaN/Inf、限值为负、q_lower>q_upper、
            周期非正有限、键集不一致等结构错误（消息含字段名与文件路径）。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"硬件配置文件不存在或不是常规文件: {path}")

    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"硬件配置必须是映射结构（文件: {path}）")

    unknown = sorted(map(str, set(raw) - set(_TOP_FIELDS)))
    if unknown:
        raise ValueError(f"未知顶层字段 {unknown}（文件: {path}）")

    schema_version = _require_int(raw, "schema_version", path)
    if schema_version != R1D_HARDWARE_SCHEMA_VERSION:
        raise ValueError(
            f"字段 schema_version 不支持: {schema_version}"
            f"（当前支持 {R1D_HARDWARE_SCHEMA_VERSION}，文件: {path}）"
        )
    robot_id = _require_nonempty_str(raw, "robot_id", path)

    side = raw.get("side")
    if not isinstance(side, str) or side not in R1D_SIDES:
        raise ValueError(
            f"字段 side 必须是 {list(R1D_SIDES)} 之一（文件: {path}）"
        )

    communication = _parse_communication(raw, path)
    layout = _parse_layout(raw, path)
    calibration = _parse_calibration(raw, path)
    limits = _parse_limits(raw, path)
    timing = _parse_timing(raw, path)

    # 文件内部一致性：所有非 null 按名映射的键集合必须与 joint_slots 一致
    slot_keys = set(layout.joint_slots)
    named_maps: list[tuple[str, Mapping[str, Any] | None]] = []
    if calibration.joint_signs is not None:
        named_maps.append(("calibration.joint_signs", calibration.joint_signs))
    if calibration.joint_offsets_rad is not None:
        named_maps.append(("calibration.joint_offsets_rad", calibration.joint_offsets_rad))
    for limit_field in _LIMIT_FIELDS[1:]:
        value = getattr(limits, limit_field)
        if value is not None:
            named_maps.append((f"limits.{limit_field}", value))
    for field_name, mapping in named_maps:
        if set(mapping) != slot_keys:  # type: ignore[arg-type]
            raise ValueError(
                f"字段 {field_name} 的键集合与 layout.joint_slots 不一致"
                f"（差异: {sorted(set(mapping) ^ slot_keys)[:4]}，文件: {path}）"  # type: ignore[arg-type]
            )

    return R1DHardwareConfig(
        schema_version=schema_version,
        robot_id=robot_id,
        side=side,
        communication=communication,
        layout=layout,
        calibration=calibration,
        limits=limits,
        timing=timing,
    )


def _require_section(raw: dict[str, Any], name: str, fields: tuple[str, ...], path: Path) -> dict[str, Any]:
    """校验必需节存在且无未知字段，返回节内容。"""
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"缺少节 {name} 或其不是映射（文件: {path}）")
    unknown = sorted(map(str, set(section) - set(fields)))
    if unknown:
        raise ValueError(f"节 {name} 内未知字段 {unknown}（文件: {path}）")
    return section


def _require_nonempty_str(raw: dict[str, Any], field: str, path: Path) -> str:
    """校验非空字符串字段。"""
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"字段 {field} 必须是非空字符串（文件: {path}）")
    return value


def _require_int(raw: dict[str, Any], field: str, path: Path) -> int:
    """校验整数字段（bool 视为非法）。"""
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"字段 {field} 必须是整数（文件: {path}）")
    return value


def _require_bool(section: dict[str, Any], field: str, path: Path) -> bool:
    """校验严格布尔字段（整数 0/1 拒绝）。"""
    value = section.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"字段 {field} 必须是布尔值 true/false（文件: {path}）")
    return value


def _parse_communication(raw: dict[str, Any], path: Path) -> R1DCommunicationConfig:
    """解析 communication 节。"""
    section = _require_section(raw, "communication", _COMMUNICATION_FIELDS, path)
    dds_domain = _require_int(section, "dds_domain", path)
    if dds_domain < 0:
        raise ValueError(f"字段 dds_domain 必须非负（文件: {path}）")
    network_interface = section.get("network_interface")
    if network_interface is not None and (
        not isinstance(network_interface, str) or not network_interface.strip()
    ):
        raise ValueError(
            f"字段 network_interface 必须是非空字符串或 null（文件: {path}）"
        )
    # 话题名为协议冻结值，但显式写入文件以便与 manifest 对照
    lowcmd_topic = _require_nonempty_str(section, "lowcmd_topic", path)
    lowstate_topic = _require_nonempty_str(section, "lowstate_topic", path)
    return R1DCommunicationConfig(
        dds_domain=dds_domain,
        network_interface=network_interface,
        lowcmd_topic=lowcmd_topic,
        lowstate_topic=lowstate_topic,
    )


def _parse_named_int_map(
    section: dict[str, Any], field: str, path: Path
) -> Mapping[str, int]:
    """解析按名整数映射：键非空字符串、值 int（bool 非法）。"""
    value = section.get(field)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"字段 {field} 必须是非空映射（文件: {path}）")
    result: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"字段 {field} 含非字符串或空键（文件: {path}）")
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"字段 {field}[{key}] 必须是整数（文件: {path}）"
            )
        result[key] = item
    return MappingProxyType(result)


def _parse_named_float_map(
    section: dict[str, Any], field: str, path: Path
) -> Mapping[str, float]:
    """解析按名浮点映射：键非空字符串、值有限（NaN/Inf 拒绝）。"""
    value = section.get(field)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"字段 {field} 必须是非空映射（文件: {path}）")
    result: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"字段 {field} 含非字符串或空键（文件: {path}）")
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"字段 {field}[{key}] 必须是数值（文件: {path}）")
        number = float(item)
        if not bool(np.isfinite(number)):
            raise ValueError(
                f"字段 {field}[{key}] 含 NaN 或 Inf，必须是有限值（文件: {path}）"
            )
        result[key] = number
    return MappingProxyType(result)


def _parse_layout(raw: dict[str, Any], path: Path) -> R1DLayoutConfig:
    """解析 layout 节：槽位数量/范围/唯一性校验。"""
    section = _require_section(raw, "layout", _LAYOUT_FIELDS, path)
    layout_verified = _require_bool(section, "layout_verified", path)
    joint_slots = _parse_named_int_map(section, "joint_slots", path)
    if len(joint_slots) != R1D_ARM_JOINT_COUNT:
        raise ValueError(
            f"字段 joint_slots 必须恰好 {R1D_ARM_JOINT_COUNT} 项"
            f"（漏/多关节），实际 {len(joint_slots)}（文件: {path}）"
        )
    seen: dict[int, str] = {}
    for name, slot in joint_slots.items():
        if not (0 <= slot < R1D_SLOT_COUNT):
            raise ValueError(
                f"关节 {name} 槽位 {slot} 越界（整机 {R1D_SLOT_COUNT} 槽，"
                f"合法范围 0..{R1D_SLOT_COUNT - 1}，文件: {path}）"
            )
        if slot in seen:
            raise ValueError(
                f"槽位 {slot} 被重复映射（{seen[slot]} 与 {name}，文件: {path}）"
            )
        seen[slot] = name
    return R1DLayoutConfig(layout_verified=layout_verified, joint_slots=joint_slots)


def _parse_calibration(raw: dict[str, Any], path: Path) -> R1DCalibrationConfig:
    """解析 calibration 节：sign ±1、数量 7、成对提供校验。"""
    section = _require_section(raw, "calibration", _CALIBRATION_FIELDS, path)
    verified = _require_bool(section, "verified", path)
    signs_raw = section.get("joint_signs")
    offsets_raw = section.get("joint_offsets_rad")
    if (signs_raw is None) != (offsets_raw is None):
        raise ValueError(
            "字段 joint_signs 与 joint_offsets_rad 必须同时提供或同时为 null"
            f"（未知校准不能启用运动，文件: {path}）"
        )
    if signs_raw is None or offsets_raw is None:
        return R1DCalibrationConfig(verified=verified, joint_signs=None, joint_offsets_rad=None)
    signs = _parse_named_int_map(section, "joint_signs", path)
    offsets = _parse_named_float_map(section, "joint_offsets_rad", path)
    if len(signs) != R1D_ARM_JOINT_COUNT:
        raise ValueError(
            f"字段 joint_signs 必须恰好 {R1D_ARM_JOINT_COUNT} 项，"
            f"实际 {len(signs)}（文件: {path}）"
        )
    if len(offsets) != R1D_ARM_JOINT_COUNT:
        raise ValueError(
            f"字段 joint_offsets_rad 必须恰好 {R1D_ARM_JOINT_COUNT} 项，"
            f"实际 {len(offsets)}（文件: {path}）"
        )
    for name, sign in signs.items():
        if sign not in (1, -1):
            raise ValueError(
                f"字段 joint_signs[{name}] 只能取 ±1，实际 {sign}（文件: {path}）"
            )
    return R1DCalibrationConfig(
        verified=verified, joint_signs=signs, joint_offsets_rad=offsets
    )


def _parse_limits(raw: dict[str, Any], path: Path) -> R1DLimitConfig:
    """解析 limits 节：数量 7、有限、物理量非负、q_lower<=q_upper。"""
    section = _require_section(raw, "limits", _LIMIT_FIELDS, path)
    verified = _require_bool(section, "verified", path)
    maps: dict[str, Mapping[str, float] | None] = {}
    for field in _LIMIT_FIELDS[1:]:
        value = section.get(field)
        if value is None:
            maps[field] = None
            continue
        parsed = _parse_named_float_map(section, field, path)
        if len(parsed) != R1D_ARM_JOINT_COUNT:
            raise ValueError(
                f"字段 {field} 必须恰好 {R1D_ARM_JOINT_COUNT} 项，"
                f"实际 {len(parsed)}（文件: {path}）"
            )
        maps[field] = parsed
    # 物理量非负（位置上下限可为负）
    for field in ("dq_limit_rad_s", "tau_limit_nm", "kp_nm_per_rad", "kd_nms_per_rad"):
        mapping = maps[field]
        if mapping is None:
            continue
        for name, value in mapping.items():
            if value < 0.0:
                raise ValueError(
                    f"字段 {field}[{name}] 必须非负，实际 {value}（文件: {path}）"
                )
    # 位置上下限成对提供时校验次序
    lower = maps["q_lower_rad"]
    upper = maps["q_upper_rad"]
    if lower is not None and upper is not None:
        for name in lower:
            if lower[name] > upper[name]:  # type: ignore[index]
                raise ValueError(
                    f"关节 {name} q_lower_rad {lower[name]} > q_upper_rad "
                    f"{upper[name]}（文件: {path}）"  # type: ignore[index]
                )
    return R1DLimitConfig(
        verified=verified,
        q_lower_rad=lower,
        q_upper_rad=upper,
        dq_limit_rad_s=maps["dq_limit_rad_s"],
        tau_limit_nm=maps["tau_limit_nm"],
        kp_nm_per_rad=maps["kp_nm_per_rad"],
        kd_nms_per_rad=maps["kd_nms_per_rad"],
    )


def _parse_timing(raw: dict[str, Any], path: Path) -> R1DTimingConfig:
    """解析 timing 节：非 null 时只接受正有限值。"""
    section = _require_section(raw, "timing", _TIMING_FIELDS, path)

    def _parse_optional_positive(field: str) -> float | None:
        """正有限可选浮点：null 合法（待确认），0/负/NaN/Inf 拒绝。"""
        value = section.get(field)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"字段 {field} 必须是数值或 null（文件: {path}）")
        number = float(value)
        if not bool(np.isfinite(number)) or number <= 0.0:
            raise ValueError(
                f"字段 {field} 只接受正有限值，实际 {value}（文件: {path}）"
            )
        return number

    return R1DTimingConfig(
        control_period_s=_parse_optional_positive("control_period_s"),
        state_timeout_s=_parse_optional_positive("state_timeout_s"),
        diagnostic_duration_s=_parse_optional_positive("diagnostic_duration_s"),
    )


# ---------------------------------------------------------------------------
# purpose 门槛校验
# ---------------------------------------------------------------------------


def validate_r1d_hardware_config(
    config: R1DHardwareConfig,
    profile: RobotProfile,
    *,
    purpose: Literal["offline", "read_only", "motion"],
) -> None:
    """按用途梯度校验硬件配置与机器人描述的一致性。

    Args:
        config: 已加载的硬件配置。
        profile: 机器人描述（须为 R1D 单侧 7 关节描述）。
        purpose: 用途门槛：
            offline（结构可解析即可）/ read_only（有效通信 + 已核验布局 +
            状态超时，I-07：connect 依赖状态缓存）/ motion（全部硬件门槛，
            null/provisional 一律拒绝）。

    Raises:
        ValueError: purpose 非法、robot_id 不一致、关节数不是 7、
            side 与 profile 关节名冲突、joint_slots 与 profile 名称不符，
            或对应 purpose 门槛未满足（消息指明缺失字段）。
    """
    if purpose not in R1D_PURPOSES:
        raise ValueError(f"未知 purpose {purpose!r}，支持 {list(R1D_PURPOSES)}")

    # —— 基础一致性（offline 级）——
    if config.robot_id != profile.robot_id:
        raise ValueError(
            f"配置 robot_id={config.robot_id!r} 与 profile "
            f"robot_id={profile.robot_id!r} 不一致"
        )
    if profile.ndof != R1D_ARM_JOINT_COUNT:
        raise ValueError(
            f"R1D 硬件配置要求 {R1D_ARM_JOINT_COUNT} 关节 profile，"
            f"实际 ndof={profile.ndof}"
        )
    prefix = f"{config.side}_"
    conflicting = [n for n in profile.active_joint_names if not n.startswith(prefix)]
    if conflicting:
        raise ValueError(
            f"side={config.side!r} 与 profile 关节名冲突（不以 "
            f"{prefix!r} 开头）: {conflicting[:3]}"
        )
    profile_names = set(profile.active_joint_names)
    slot_names = set(config.layout.joint_slots)
    if profile_names != slot_names:
        raise ValueError(
            "layout.joint_slots 与 profile.active_joint_names 不一致"
            f"（差异: {sorted(profile_names ^ slot_names)[:4]}）"
        )

    if purpose == "offline":
        return

    # —— read_only 门槛：有效通信参数 + 状态超时（motion 亦须满足）——
    if config.communication.network_interface is None:
        raise ValueError(
            f"purpose={purpose} 需要有效通信参数：network_interface 为 null（待现场确认）"
        )
    if purpose == "read_only":
        # §3.1 勘误（评审 2026-11-18）：read_only 不再强制 layout_verified
        # ——只读采集不发送任何命令，恰是 4.9 首轮逐关节核对/布局核验的
        # 数据来源；强制已核验会让"首轮原始采集"无可执行路径，逼操作员
        # 虚标 verified。joint_slots 名称级一致性已在本函数上方对全部
        # purpose 校验；物理逐关节标定（校准节）仍由 motion 门槛强制。
        # layout_verified 的强制档位= motion（见下）。
        # I-07 修复（评审 2026-11-17）：只读会话同样依赖状态缓存做新鲜度
        # 判定——connect() 在 timing.state_timeout_s 为 null（缓存禁用）时
        # 直接拒绝。预检即显式拒绝，避免按规程准备的配置到 connect 才
        # 隐式失败（只读诊断/状态读取均不可用）。
        if config.timing.state_timeout_s is None:
            raise ValueError(
                "purpose=read_only 需要 timing.state_timeout_s（状态缓存/"
                "新鲜度判定依赖；未配置时 connect() 将拒绝连接）"
            )
        state_timeout = config.timing.state_timeout_s
        if not bool(np.isfinite(state_timeout)) or state_timeout <= 0.0:
            raise ValueError(
                f"purpose=read_only 的 timing.state_timeout_s 必须为正有限值，"
                f"实际 {state_timeout}"
            )
        return

    # —— motion 门槛：全部硬件门槛，null/provisional 一律拒绝 ——
    # 布局核验强制档位从 read_only 移到 motion（§3.1 勘误，评审
    # 2026-11-18）：接管/运动前必须完成 4.9 布局核对（joint_slots 名称
    # 级映射已核验），只读采集阶段允许未核验
    if not config.layout.layout_verified:
        raise ValueError(
            "purpose=motion 需要已核验布局：layout_verified=false"
            "（4.9 只读核对后置 true；只读采集允许 false）"
        )
    if not config.calibration.verified:
        raise ValueError(
            "purpose=motion 拒绝 provisional 校准：calibration.verified=false"
        )
    if (
        config.calibration.joint_signs is None
        or config.calibration.joint_offsets_rad is None
    ):
        raise ValueError(
            "purpose=motion 需要非 null 校准（joint_signs/joint_offsets_rad，"
            "未知校准不能启用运动）"
        )
    if not config.limits.verified:
        raise ValueError("purpose=motion 拒绝 provisional 限值：limits.verified=false")
    for field in _LIMIT_FIELDS[1:]:
        if getattr(config.limits, field) is None:
            raise ValueError(f"purpose=motion 需要非 null 限值：limits.{field}")
    # timing 复查正有限：绕过 load 直接构造 config 时（frozen dataclass
    # 不做运行时校验），仅查非 null 会放行 0/负/NaN 周期，故双层都查
    for field in _TIMING_FIELDS:
        value = getattr(config.timing, field)
        if value is None:
            raise ValueError(f"purpose=motion 需要 {field}（正有限值，待现场冻结）")
        if not bool(np.isfinite(value)) or value <= 0.0:
            raise ValueError(
                f"purpose=motion 的 {field} 必须为正有限值，实际 {value}"
            )
