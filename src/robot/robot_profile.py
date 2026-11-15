"""机器人描述（RobotProfile）— 身份与模型语义名称的纯配置层。

仅负责解析 configs/robots/*.yaml 中的机器人身份、MuJoCo 模型路径与
关节/执行器/TCP 的语义名称。不加载 MuJoCo 模型，不导入真机 SDK，
不持有任何数值参数（dt/PD/限位等继续来自 constants.py 与 real_robot.yaml）。

描述能力与运行能力分开：本模块接受任何合法描述（含未知 robot_id），
运行入口是否支持由 runner_factory 的能力校验负责。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# 当前支持的描述文件 schema 版本
SUPPORTED_SCHEMA_VERSION = 1

# 描述文件允许出现的字段（未知字段一律拒绝，防止拼写错误静默生效）
_PROFILE_FIELDS = (
    "schema_version",
    "robot_id",
    "model_path",
    "active_joint_names",
    "actuator_names",
    "held_joint_names",
    "tcp_site_name",
)


@dataclass(frozen=True)
class RobotProfile:
    """机器人语义描述 — 仅身份与名称，不含数值参数。

    Attributes:
        schema_version: 描述文件 schema 版本（当前为 1）。
        robot_id: 机器人身份标识（如 "rm65"）。
        model_path: MuJoCo 模型绝对路径（相对 YAML 父目录解析）。
        active_joint_names: 主动（驱动）关节名，按控制向量顺序。
        actuator_names: 执行器名，与主动关节一一对应。
        held_joint_names: 保持（不驱动）关节名。
        tcp_site_name: 末端执行器（TCP）site 名。
    """

    schema_version: int
    robot_id: str
    model_path: Path
    active_joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    held_joint_names: tuple[str, ...]
    tcp_site_name: str

    @property
    def ndof(self) -> int:
        """主动关节自由度数。"""
        return len(self.active_joint_names)


def load_robot_profile(path: Path) -> RobotProfile:
    """从 YAML 文件加载机器人描述。

    Args:
        path: 描述 YAML 文件路径。

    Returns:
        不可变的 RobotProfile 实例。

    Raises:
        FileNotFoundError: 描述文件不存在，或 model_path 指向的
            模型文件不存在/不是常规文件（消息含两个路径）。
        ValueError: 字段缺失/为空/重复、版本不支持或存在未知字段，
            错误消息包含字段名与文件路径。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"机器人描述文件不存在或不是常规文件: {path}")

    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"描述文件必须是映射结构（文件: {path}）")

    # 未知字段检查（先于字段级校验，尽早暴露拼写错误）。
    # 键统一转 str 再排序：YAML 顶层出现 int/date 等混合键类型时不抛 TypeError。
    unknown = sorted(map(str, set(raw) - set(_PROFILE_FIELDS)))
    if unknown:
        raise ValueError(f"未知字段 {unknown}（文件: {path}）")

    schema_version = _require_schema_version(raw, path)
    robot_id = _require_str(raw, "robot_id", path)
    model_path_str = _require_str(raw, "model_path", path)
    active = _require_name_tuple(raw, "active_joint_names", path, allow_empty=False)
    actuators = _require_name_tuple(raw, "actuator_names", path, allow_empty=False)
    held = _require_name_tuple(raw, "held_joint_names", path, allow_empty=True)
    tcp_site_name = _require_str(raw, "tcp_site_name", path)

    # 执行器与主动关节一一对应
    if len(actuators) != len(active):
        raise ValueError(
            f"字段 actuator_names 长度 {len(actuators)} 与 active_joint_names "
            f"长度 {len(active)} 不匹配（文件: {path}）"
        )
    # 主动与保持关节不得重叠
    overlap = sorted(set(active) & set(held))
    if overlap:
        raise ValueError(
            f"字段 held_joint_names 与 active_joint_names 重叠: {overlap}（文件: {path}）"
        )

    # model_path 相对 YAML 父目录解析为绝对路径
    model_path = Path(model_path_str)
    if not model_path.is_absolute():
        model_path = path.resolve().parent / model_path
    model_path = model_path.resolve()

    # 存在性校验（位于字段级校验之后，不影响字段诊断优先）：
    # 让"模型缺失"稳定报 FileNotFoundError 并给出两个路径，
    # 而不是把错误推迟到 MuJoCo 加载层的 ParseXML/临时目录错误。
    if not model_path.is_file():
        raise FileNotFoundError(
            f"model_path 指向的模型文件不存在或不是常规文件: "
            f"{model_path}（描述文件: {path}）"
        )

    return RobotProfile(
        schema_version=schema_version,
        robot_id=robot_id,
        model_path=model_path,
        active_joint_names=active,
        actuator_names=actuators,
        held_joint_names=held,
        tcp_site_name=tcp_site_name,
    )


def default_rm65_profile_path() -> Path:
    """返回仓库默认 RM65 描述路径（由模块文件位置定位，不依赖 cwd）。

    Returns:
        configs/robots/rm65.yaml 的绝对路径。
    """
    return (
        Path(__file__).resolve().parent.parent.parent
        / "configs"
        / "robots"
        / "rm65.yaml"
    )


def _require_schema_version(raw: dict[str, Any], path: Path) -> int:
    """校验 schema_version 字段（bool 不视为合法整数）。"""
    value = raw.get("schema_version")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"字段 schema_version 必须是整数（文件: {path}）")
    if value != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"字段 schema_version 不支持: {value}（当前支持 "
            f"{SUPPORTED_SCHEMA_VERSION}，文件: {path}）"
        )
    return value


def _require_str(raw: dict[str, Any], field: str, path: Path) -> str:
    """校验非空字符串字段。"""
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"字段 {field} 必须是非空字符串（文件: {path}）")
    return value


def _require_name_tuple(
    raw: dict[str, Any], field: str, path: Path, *, allow_empty: bool
) -> tuple[str, ...]:
    """校验名称列表字段：元素非空、不重复，返回不可变 tuple。"""
    value = raw.get(field)
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise ValueError(f"字段 {field} 必须是字符串列表（文件: {path}）")
    if not allow_empty and not value:
        raise ValueError(f"字段 {field} 不能为空列表（文件: {path}）")
    names = tuple(value)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"字段 {field} 存在重复名称: {duplicates}（文件: {path}）")
    if any(not n.strip() for n in names):
        raise ValueError(f"字段 {field} 含空字符串元素（文件: {path}）")
    return names
