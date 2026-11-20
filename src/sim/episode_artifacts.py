"""V12 回合产物（v2 schema）写入、读取、快照与回放。

本模块是 T6 的唯一产物 writer：把 :class:`V12EpisodeResult` 落盘为
``v12_episode.json``（schema ``v12_episode_v2``）与 ``v12_episode.npz``，
同时把本次回合依赖的小体积配置源文件复制到 ``output_dir/snapshot/``，
写 ``manifest.json``（逐文件 sha256）、``resolved_config.json`` 与
``scene.json``。回放只读产物目录，按关节名重建 MuJoCo 状态，**绝不**
改写已保存的 JSON/NPZ（不合成反弹、不重新判定命中）。

设计约束（T6 brief §3/§4）：

- 产物必须自描述：``schema`` / ``simulation_only`` / ``joint_names`` /
  ``units`` / ``input_sha256`` / ``scene_snapshot`` 全部写全；
- ``torque`` 与 ``position`` 模式的 ``U`` 语义不同，按 ``mode`` 记录单位，
  不写 ``q_desired`` 等混合字段；
- 失败回合（含 ``ball_unreachable``）同样落盘，``X`` 至少 1 帧、
  ``U`` 允许 0 行；
- 快照文件缺失时回退外部源文件，但必须逐个校验哈希，不一致即拒绝；
- 回放优先加载哈希校验通过的 ``model`` 快照：快照 XML 中 mesh/texture
  的相对路径按写入时的原始位置解析，物化为临时 XML（Windows 非 ASCII
  路径映射到 ASCII 副本）并用原 profile 文本生成临时 profile；快照存在
  但哈希不符视为篡改，即使外部模型匹配也拒绝；
- 依赖资产哈希（T6 追加，设计 §6）：模型 XML 引用的 ``<mesh>`` /
  ``<texture>`` 文件不复制（体积大），但逐文件记入 manifest 的
  ``assets`` 条目并聚合为 ``input_sha256["assets"]``；回放时校验这些资产
  的哈希，mesh 被改会明确拒绝，避免“模型哈希不变但几何悄悄变化”；
  旧产物缺 ``assets`` 键时按“无资产哈希”兼容（仅警告）；
- 回放默认 1.0 倍速，慢放通过显式 ``speed`` 参数。

注：日志统一走 ``logging``（本模块产出诊断/警告），不使用 ``print``。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import re
import shutil
import sys
import tempfile
import time
import weakref
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from src.sim.v12_episode import V12EpisodeResult
    from src.sim.v12_robot_adapter import V12RobotBundle

_LOGGER = logging.getLogger(__name__)

#: 产物 schema 版本（T6 契约）
EPISODE_SCHEMA = "v12_episode_v2"
ARTIFACT_JSON_NAME = "v12_episode.json"
ARTIFACT_NPZ_NAME = "v12_episode.npz"

#: 快照目录结构：``snapshot/files/<相对仓库路径>`` + 三个元数据文件
SNAPSHOT_DIR_NAME = "snapshot"
SNAPSHOT_FILES_DIR_NAME = "files"
SNAPSHOT_MANIFEST_NAME = "manifest.json"
SNAPSHOT_CONFIG_NAME = "resolved_config.json"
SNAPSHOT_SCENE_NAME = "scene.json"

#: 单位表（写入 payload["units"]；键名固定，供论文/工具消费）
UNITS: dict[str, str] = {
    "position": "m",
    "linear_velocity": "m/s",
    "velocity": "m/s",
    "angle": "rad",
    "angular_velocity": "rad/s",
    "time": "s",
    "distance": "m",
    "control_torque": "N*m",
    "control_position": "rad",
}

#: 快照需要复制的配置源角色（顺序稳定，便于 manifest 比对）
SNAPSHOT_ROLES: tuple[str, ...] = (
    "robot_profile",
    "runtime",
    "task",
    "racket",
    "model",
)

#: 需要参与 ``input_sha256`` 的角色（scene_snapshot 由 canonical JSON 计算，
#: assets 由依赖资产（mesh/texture）的 相对路径+sha256 列表聚合计算）
INPUT_SHA_ROLES: tuple[str, ...] = (
    "robot_profile",
    "runtime",
    "task",
    "racket",
    "model",
    "assets",
    "scene_snapshot",
)

#: 模型 XML 中纳入依赖资产哈希的标签（T6 追加，设计 §6 依赖资产哈希快照）
ASSET_XML_TAGS: tuple[str, ...] = ("mesh", "texture")
_ASSET_TAG_PATTERN = re.compile(
    r"<(mesh|texture)\b([^>]*?)/?>", re.IGNORECASE | re.DOTALL
)
_ASSET_FILE_ATTR_PATTERN = re.compile(
    r"""\bfile\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE
)

#: compiler/include 联合扫描：按**文档顺序**处理全局目录声明与 include 展开
#: （``<compiler>`` 是全局设置，include 片段里的声明同样生效，后者覆盖前者）
_COMPILER_OR_INCLUDE_PATTERN = re.compile(
    r"<(?P<tag>compiler|include)\b(?P<attrs>[^>]*?)/?>",
    re.IGNORECASE,
)
#: 资产种类 → compiler 目录属性（与 src/robot/model_factory.py 同源语义）
_ASSET_DIR_ATTRS: dict[str, str] = {
    "mesh": "meshdir",
    "texture": "texturedir",
}

#: 装配依赖闭包清单版本（F08）。旧 artifact 无该键 → legacy 分支（警告，
#: 不假装已验证闭包）。
DEPENDENCY_MANIFEST_VERSION = 1

#: 依赖条目角色（与 src/robot/model_factory.py 的同名常量保持一致）
DEPENDENCY_ROLE_MODEL = "model"
DEPENDENCY_ROLE_INCLUDE = "include"
DEPENDENCY_ROLE_ARM_FRAGMENT = "arm_fragment"
DEPENDENCY_ROLE_MESH = "mesh"
DEPENDENCY_ROLE_TEXTURE = "texture"

#: 需要把内容写进快照的 XML 角色（网格/纹理只记哈希，体积大不复制）
DEPENDENCY_XML_ROLES = frozenset(
    {
        DEPENDENCY_ROLE_MODEL,
        DEPENDENCY_ROLE_INCLUDE,
        DEPENDENCY_ROLE_ARM_FRAGMENT,
    }
)
#: 网格/纹理角色（回放改写 ``<mesh>/<texture>`` 的 ``file`` 引用）
DEPENDENCY_ASSET_ROLES = frozenset(
    {DEPENDENCY_ROLE_MESH, DEPENDENCY_ROLE_TEXTURE}
)
#: 快照内依赖资产副本目录（相对 ``snapshot/``）
_DEPENDENCY_COPY_DIRNAME = "assets"
#: MJCF ``<include>`` 标签（只按真实解析路径展开，不做无边界扫描）
_INCLUDE_TAG_PATTERN = re.compile(r"<include\b[^>]*?/?>", re.IGNORECASE)
#: 单个模型文件 include 展开的最大深度（防止环状 include 无界递归）
_MAX_INCLUDE_DEPTH = 8
#: 是否强制要求 R1D 回合结果自带装配依赖清单。
#
# ``V12EpisodeResult.model_dependency_manifest`` 字段已由 A 落地，但
# ``run_v12_episode`` 的接线（I1）尚未完成，因此该开关默认关闭：此时写入
# 会退化为"按配置推导 + ``dependency_manifest_source="derived"`` + 警告"，
# 产物明确自曝弱证据，不假装已验证闭包。**I1 接线完成后必须置为 True**，
# 届时 R1D 结果清单为空会直接拒绝写入（见 ``_build_dependency_entries``）。
REQUIRE_RESULT_DEPENDENCY_MANIFEST: bool = True

_CONFIG_PATH_FIELDS = (
    "robot_profile_path",
    "runtime_path",
    "racket_path",
    "task_path",
    "output_dir",
)
_CONFIG_ARRAY_FIELDS = ("target_center", "kp", "kd")


# ---------------------------------------------------------------------------
# 通用 JSON / 哈希工具
# ---------------------------------------------------------------------------
def _repo_root() -> Path:
    """返回仓库根目录（本文件为 ``<root>/src/sim/episode_artifacts.py``）。"""
    return Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    """计算文件的 SHA-256 十六进制摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """递归把 numpy/Path/Mapping 等转成 JSON 可序列化对象。

    非有限浮点（``nan`` / ``inf``）统一转成 ``None``：失败回合里
    ``min_dist`` / ``pos_error`` 等可能为 ``inf``，直接序列化会触发
    ``json.dumps(..., allow_nan=False)`` 的 ``ValueError``。
    """
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    return value


def _canonical_json(value: Any) -> str:
    """生成稳定的（排序键、紧凑、禁用 NaN）JSON 文本，用于哈希。"""
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def config_to_dict(config: Any) -> dict[str, Any]:
    """把 ``ResolvedV12Config`` 序列化为 JSON 可序列化字典。"""
    payload: dict[str, Any] = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if field.name in _CONFIG_PATH_FIELDS:
            payload[field.name] = None if value is None else str(Path(value))
        elif field.name in _CONFIG_ARRAY_FIELDS:
            payload[field.name] = None if value is None else _jsonable(value)
        elif field.name == "sources":
            payload[field.name] = {str(k): str(v) for k, v in value.items()}
        elif field.name == "explicit_options":
            payload[field.name] = sorted(str(v) for v in value)
        else:
            payload[field.name] = _jsonable(value)
    return payload


def config_from_dict(payload: Mapping[str, Any]) -> Any:
    """由 :func:`config_to_dict` 的字典重建 ``ResolvedV12Config``。"""
    from src.sim.v12_config import ResolvedV12Config  # 延迟导入避免环

    kwargs: dict[str, Any] = {}
    for field in fields(ResolvedV12Config):
        if field.name not in payload:
            continue
        value = payload[field.name]
        if field.name in _CONFIG_PATH_FIELDS:
            kwargs[field.name] = None if value is None else Path(value)
        elif field.name in _CONFIG_ARRAY_FIELDS:
            kwargs[field.name] = None if value is None else np.asarray(value, dtype=float)
        elif field.name == "explicit_options":
            kwargs[field.name] = frozenset(value)
        else:
            kwargs[field.name] = value
    return ResolvedV12Config(**kwargs)


# ---------------------------------------------------------------------------
# 配置源文件角色解析
# ---------------------------------------------------------------------------
def _profile_path_for_config(config: Any) -> Path | None:
    """返回机器人描述文件路径（legacy RM65 缺省回退到 rm65.yaml）。"""
    path = getattr(config, "robot_profile_path", None)
    if path is not None:
        return Path(path)
    from src.sim.v12_robot_adapter import default_rm65_profile_path

    return Path(default_rm65_profile_path())


def _model_path_for_profile(profile_path: Path) -> Path:
    """解析机器人描述里 ``model_path`` 的绝对路径。"""
    from src.robot.robot_profile import load_robot_profile

    return Path(load_robot_profile(profile_path).model_path)


def _role_paths(config: Any) -> dict[str, Path | None]:
    """按快照角色解析配置源文件路径（未配置的角色为 None）。"""
    profile_path = _profile_path_for_config(config)
    try:
        model_path: Path | None = (
            _model_path_for_profile(profile_path) if profile_path else None
        )
    except (OSError, ValueError) as exc:  # pragma: no cover - 配置损坏时防御
        _LOGGER.warning("解析机器人模型路径失败：%s", exc)
        model_path = None
    return {
        "robot_profile": profile_path,
        "runtime": getattr(config, "runtime_path", None),
        "task": getattr(config, "task_path", None),
        "racket": getattr(config, "racket_path", None),
        "model": model_path,
    }


def _role_paths_from_payload(payload: Mapping[str, Any]) -> dict[str, Path | None]:
    """从 payload 的 config 快照重建角色路径（用于快照/回放校验）。"""
    from src.sim.v12_config import ResolvedV12Config  # noqa: F401  (类型校验)

    config = config_from_dict(payload["config"])
    return _role_paths(config)


# ---------------------------------------------------------------------------
# 快照
# ---------------------------------------------------------------------------
def _snapshot_relative_path(source: Path) -> Path:
    """计算快照内的相对路径（仓库内保留相对路径，仓库外放 external/）。"""
    resolved = Path(source).resolve()
    try:
        return resolved.relative_to(_repo_root())
    except ValueError:
        return Path("external") / resolved.name


def _tag_attribute(attrs: str, name: str) -> str | None:
    """取出 XML 标签属性值（单/双引号均可），未声明时返回 ``None``。

    Args:
        attrs: 标签内的属性串。
        name: 属性名（大小写不敏感）。

    Returns:
        去掉引号的原始取值；属性不存在时返回 ``None``（与显式空值区分）。
    """
    match = re.search(
        rf"""\b{name}\s*=\s*(?:"([^"]*)"|'([^']*)')""",
        attrs,
        re.IGNORECASE,
    )
    if match is None:
        return None
    value = match.group(1)
    if value is None:
        value = match.group(2) or ""
    return value


def _replace_file_attribute(tag: str, value: str) -> str:
    """只替换标签内的 ``file="..."`` 属性值，其余字符原样保留。

    注释掩码保持字符位置，因此按**原文**标签做局部替换不会破坏注释内容与
    标签格式（R03 要求注释原样保留）。

    Args:
        tag: 原文中的完整标签字符串（``<mesh .../>`` 等）。
        value: 新的 ``file`` 属性值（通常为临时目录内的绝对路径）。

    Returns:
        替换后的标签；标签没有 ``file=`` 属性时原样返回。
    """
    match = _ASSET_FILE_ATTR_PATTERN.search(tag)
    if match is None:
        return tag
    return f'{tag[: match.start()]}file="{value}"{tag[match.end() :]}'


def _masked_xml(text: str) -> str:
    """复用模型工厂的 XML 注释掩码助手（延迟导入避免模块级 mujoco 依赖）。

    注释在 MJCF 解析时被整体丢弃；回放侧的正则改写也必须先掩码，否则注释
    里的 ``<compiler>/<include>/<mesh>`` 会被误当成真实标签（R03）。
    """
    from src.robot.model_factory import mask_xml_comments

    return mask_xml_comments(text)


def _resolve_xml_reference(base: Path, raw: str) -> Path:
    """复用模型工厂的引用解析语义（绝对路径直用、相对路径按 ``base``）。"""
    from src.robot.model_factory import _resolve_reference_path

    return _resolve_reference_path(base, "", raw)


def _model_asset_entries(model_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """解析模型（含嵌套 include）的 ``<mesh>`` / ``<texture>`` 引用并算资产哈希。

    路径解析统一交给
    :func:`src.robot.model_factory.collect_model_file_dependencies`（两阶段、
    全局 ``<compiler>`` 语义、顶层目录优先），避免产物记录与真实加载行为
    分叉；只读取文件、不复制（mesh/texture 体积大，回放直接使用仓库现有
    文件）。

    Args:
        model_path: 机器人模型 XML 路径。

    Returns:
        ``(entries, missing)``：``entries`` 为存在资产的 manifest 条目
        （role/kind/source/relative_path/sha256/size，按 relative_path
        排序）；``missing`` 为解析失败或不存在的**资产引用**路径字符串
        列表（include 缺失只影响依赖闭包，不混进资产缺失）。
    """
    # 延迟 import：src.robot.model_factory 依赖 mujoco，而本模块在纯产物
    # 校验/回放路径上不应被迫加载仿真依赖
    from src.robot.model_factory import (
        DEPENDENCY_ROLE_MESH,
        DEPENDENCY_ROLE_TEXTURE,
        collect_model_file_dependencies,
    )

    resolved_model = Path(model_path).resolve()
    if not resolved_model.is_file():
        return [], [str(resolved_model)]
    asset_missing: list[str] = []
    try:
        dependencies, _ = collect_model_file_dependencies(
            resolved_model,
            asset_missing=asset_missing,
        )
    except (OSError, ValueError) as exc:
        _LOGGER.warning(
            "解析模型资产依赖失败，按缺失处理：%s（%s）", resolved_model, exc
        )
        return [], [str(resolved_model)]

    asset_roles = {DEPENDENCY_ROLE_MESH, DEPENDENCY_ROLE_TEXTURE}
    entries: list[dict[str, Any]] = []
    for dependency in dependencies:
        if dependency["role"] not in asset_roles:
            continue
        source = Path(str(dependency["source"])).resolve()
        entries.append(
            {
                "role": "assets",
                "kind": str(dependency["role"]),
                "source": str(source),
                "relative_path": _snapshot_relative_path(source).as_posix(),
                "sha256": str(dependency["sha256"]),
                "size": int(dependency["size"]),
            }
        )
    entries.sort(key=lambda item: str(item["relative_path"]))
    return entries, asset_missing


def _assets_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    """把依赖资产（相对路径 + sha256，排序）聚合为 canonical JSON 的 sha256。"""
    canonical = [
        {
            "relative_path": str(entry["relative_path"]),
            "sha256": str(entry["sha256"]),
        }
        for entry in sorted(entries, key=lambda item: str(item["relative_path"]))
    ]
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def _assets_section(role_paths: Mapping[str, Path | None]) -> dict[str, Any]:
    """构造 payload 的 ``assets`` 段（依赖资产逐文件条目 + 聚合状态）。

    Args:
        role_paths: :func:`_role_paths` 解析结果（取 ``model`` 角色）。

    Returns:
        ``{status, kind, count, digest, files, missing}``：``status`` 为
        ``ok``（全部存在）/ ``missing``（有引用但文件缺失或被引用模型
        缺失）/ ``empty``（模型 XML 未引用 mesh/texture）。
    """
    model_path = role_paths.get("model")
    if model_path is None or not Path(model_path).is_file():
        missing = [str(model_path)] if model_path is not None else ["<model>"]
        return {
            "status": "missing",
            "kind": "model_asset_dependencies",
            "count": 0,
            "digest": _assets_digest([]),
            "files": [],
            "missing": missing,
        }
    files, missing = _model_asset_entries(Path(model_path))
    if missing:
        status = "missing"
    elif files:
        status = "ok"
    else:
        status = "empty"
    return {
        "status": status,
        "kind": "model_asset_dependencies",
        "count": len(files),
        "digest": _assets_digest(files),
        "files": files,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# 装配依赖闭包（F08）：清单构造、校验、稳定逻辑 ID 解析
# ---------------------------------------------------------------------------
def _asset_id(role: str, digest: str, name: str) -> str:
    """生成稳定逻辑 ID：``<role>-<sha12>-<安全化文件名>``。

    逻辑 ID 只由角色、内容哈希与文件名决定，与产物写入时所在绝对路径
    无关——因此旧检出/新检出的同一内容映射到同一 ID（F02 的"稳定逻辑
    asset_id"要求）。

    Args:
        role: 依赖角色（mesh/texture/include/arm_fragment/model）。
        digest: 文件 SHA256（用于前缀，避免同名不同内容冲突）。
        name: 文件名（会被安全化为 ASCII 标识符）。

    Returns:
        ``mesh-a1b2c3d4e5f6-base_link_underpan.STL`` 形式的 ID。
    """
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("._") or "asset"
    prefix = str(digest)[:12] or "nosha"
    return f"{role}-{prefix}-{safe}"


def _snapshot_root_for(path: Path | None) -> Path | None:
    """从任意快照内文件向上定位 ``snapshot/`` 目录（含 manifest）。

    物化函数只拿到模型文件路径时，用它反查快照根目录，避免依赖调用方
    额外传参；找不到（外部模型回退）时返回 None。
    """
    if path is None:
        return None
    for parent in Path(path).resolve().parents:
        if (
            (parent / SNAPSHOT_MANIFEST_NAME).is_file()
            and (parent / SNAPSHOT_FILES_DIR_NAME).is_dir()
        ):
            return parent
    return None


def _dependency_section(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """读取 payload 的 ``dependencies`` 段（新 artifact），否则 None。"""
    section = payload.get("dependencies")
    if isinstance(section, Mapping):
        return dict(section)
    return None


def _dependency_entries(payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """返回 payload 的依赖条目与 legacy 标记。

    Returns:
        ``(entries, legacy)``：新产物按 ``dependencies.files`` 返回（legacy
        为 False）；旧产物没有该段时按旧 ``assets.files`` 合成只含
        mesh/texture 的条目（legacy 为 True）。两者都缺则返回空列表。
    """
    section = _dependency_section(payload)
    if section is not None and int(section.get("version") or 0) >= 1:
        files = [
            dict(item) for item in section.get("files") or [] if isinstance(item, Mapping)
        ]
        return _normalize_dependency_entries(files), False
    legacy_files: list[dict[str, Any]] = []
    assets = payload.get("assets") or {}
    for item in assets.get("files") or []:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("kind") or DEPENDENCY_ROLE_MESH)
        if kind not in DEPENDENCY_ASSET_ROLES:
            kind = DEPENDENCY_ROLE_MESH
        legacy_files.append(
            {
                "role": kind,
                "relative_path": str(item.get("relative_path") or ""),
                "source": str(item.get("source") or ""),
                "sha256": str(item.get("sha256") or ""),
                "size": int(item.get("size") or 0),
            }
        )
    return _normalize_dependency_entries(legacy_files), True


def _normalize_dependency_entries(
    raw: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """规范化依赖条目（补 sha256/size/content_base64/asset_id）并排序去重。

    Args:
        raw: 原始条目序列（来自 ``RobotModelBundle.dependency_manifest``
            或推导结果）。

    Returns:
        新条目列表（键：role/relative_path/source/sha256/size/asset_id，
        XML 角色另含 content_base64，网格含 reference）。

    Raises:
        ValueError: 角色未知，或缺少可计算哈希的来源文件。
    """
    known = {
        DEPENDENCY_ROLE_MODEL,
        DEPENDENCY_ROLE_INCLUDE,
        DEPENDENCY_ROLE_ARM_FRAGMENT,
        DEPENDENCY_ROLE_MESH,
        DEPENDENCY_ROLE_TEXTURE,
    }
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in raw:
        role = str(item.get("role") or "")
        if role not in known:
            raise ValueError(f"装配依赖条目角色未知：{role!r}（条目 {dict(item)}）")
        source = str(item.get("source") or "")
        digest = str(item.get("sha256") or "")
        relative = str(item.get("relative_path") or "")
        source_path = Path(source) if source else None
        if not digest:
            if source_path is None or not source_path.is_file():
                raise ValueError(
                    f"装配依赖条目缺少 sha256 且来源不可读：role={role} source={source or '<空>'}"
                )
            digest = _sha256_file(source_path)
        if not relative:
            if source_path is None:
                raise ValueError(f"装配依赖条目缺少 relative_path：role={role}")
            relative = _snapshot_relative_path(source_path).as_posix()
        entry: dict[str, Any] = {
            "role": role,
            "relative_path": relative,
            "source": source,
            "sha256": digest,
            "size": int(
                item.get("size")
                or (source_path.stat().st_size if source_path and source_path.is_file() else 0)
            ),
        }
        reference = str(item.get("reference") or "")
        if reference:
            entry["reference"] = reference
        content = item.get("content_base64")
        if content:
            entry["content_base64"] = str(content)
        elif role in DEPENDENCY_XML_ROLES and source_path is not None and source_path.is_file():
            entry["content_base64"] = base64.b64encode(source_path.read_bytes()).decode("ascii")
        entry["asset_id"] = str(
            item.get("asset_id") or _asset_id(role, digest, Path(relative or source).name)
        )
        snapshot_path = item.get("snapshot_path")
        if snapshot_path:
            # 快照内副本（写入时复制）：解析时必须优先尝试，不能因规范化丢失
            entry["snapshot_path"] = str(snapshot_path)
        asset_dirs = item.get("asset_dirs")
        if role == DEPENDENCY_ROLE_MODEL and isinstance(asset_dirs, Mapping):
            # 模型条目的最终全局 meshdir/texturedir（旧产物没有该字段，
            # 回放侧按旧语义从 XML 文本重建）：只保留已知资产种类
            cleaned = {
                str(kind): str(value)
                for kind, value in asset_dirs.items()
                if str(kind) in ASSET_XML_TAGS
            }
            if cleaned:
                entry["asset_dirs"] = cleaned
        key = (role, digest, relative)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(entry)
    normalized.sort(key=lambda entry: (str(entry["role"]), str(entry["relative_path"])))
    return normalized


def _runtime_arm_fragment(payload: Mapping[str, Any], model_path: Path) -> Path | None:
    """按 runtime 的 ``scene.fixed_joint_poses`` 推导实际会被读取的兄弟片段。

    过渡推导专用（``V12EpisodeResult.model_dependency_manifest`` 尚未被
    ``run_v12_episode`` 填充时）：只有固定姿态非空时工厂才会读取
    ``r1d_arm_{passive}.xml``。片段定位复用
    :func:`src.robot.model_factory._arm_fragment_path`，与真实装配同源，
    避免"猜路径"。

    Returns:
        片段路径；固定姿态为空或 runtime 不可读时 None。
    """
    config = config_from_dict(payload["config"])
    runtime_path = getattr(config, "runtime_path", None)
    if runtime_path is None or not Path(runtime_path).is_file():
        return None
    import yaml  # 延迟导入：仅在过渡推导分支需要

    data = yaml.safe_load(Path(runtime_path).read_text(encoding="utf-8")) or {}
    poses = ((data.get("scene") or {}).get("fixed_joint_poses")) or {}
    if not poses:
        return None
    profile_path = getattr(config, "robot_profile_path", None)
    if profile_path is None or not Path(profile_path).is_file():
        return None
    from src.robot.model_factory import _arm_fragment_path, _model_side  # 过渡推导
    from src.robot.robot_profile import load_robot_profile

    profile = load_robot_profile(Path(profile_path))
    return Path(_arm_fragment_path(profile, _model_side(profile))).resolve()


def _derive_dependency_entries(
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """从产物配置推导装配依赖闭包（结果对象尚未携带清单时的过渡路径）。

    推导 = 模型文件闭包（模型 + 嵌套 ``<include>`` + mesh/texture，按真实
    解析路径展开）+ 按 runtime 固定姿态实际会读取的兄弟片段。调用方必须
    记录 ``dependency_manifest_source="derived"`` 并告警：推导不保证覆盖
    运行期所有分支，真正的闭包证据应来自装配过程本身（``RobotModelBundle``）。

    Returns:
        ``(entries, missing)``：``missing`` 为无法解析的引用路径（写入时
        记录，回放时按"写入时依赖缺失"拒绝），与 ``assets.missing`` 同源。
    """
    config = config_from_dict(payload["config"])
    model_path = _role_paths(config).get("model")
    if model_path is None or not Path(model_path).is_file():
        raise ValueError("无法推导装配依赖清单：产物配置未指向可读的模型文件")
    from src.robot.model_factory import (  # 延迟导入：避免无 mujoco 场景的依赖
        _dependency_entry,
        collect_model_file_dependencies,
    )

    entries, missing = collect_model_file_dependencies(Path(model_path))
    merged = [dict(item) for item in entries]
    fragment = _runtime_arm_fragment(payload, Path(model_path))
    if fragment is not None:
        merged.append(_dependency_entry(DEPENDENCY_ROLE_ARM_FRAGMENT, fragment))
    return _normalize_dependency_entries(merged), [str(item) for item in missing]


def _result_dependency_manifest(result: Any) -> list[dict[str, Any]] | None:
    """读取回合结果携带的装配依赖清单（字段未落地时返回 None）。

    Returns:
        规范化条目列表（可能为空）；``None`` 表示该结果类**没有**该字段
        （旧结果/A 的字段尚未落地），调用方应走过渡推导并告警。
    """
    field_names = {item.name for item in fields(type(result))}
    if "model_dependency_manifest" not in field_names:
        return None
    raw = getattr(result, "model_dependency_manifest", None) or ()
    return _normalize_dependency_entries(
        [item for item in raw if isinstance(item, Mapping)]
    )


def _build_dependency_entries(
    result: Any,
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """构造产物要记录的装配依赖清单，返回 ``(entries, source, missing)``。

    ``source`` 取值：``result``（装配过程记录，强证据）或 ``derived``
    （过渡推导，弱证据，写警告）。R1D 生产结果若**已有**清单字段却为空，
    在 :data:`REQUIRE_RESULT_DEPENDENCY_MANIFEST` 打开时直接拒绝写入——不能
    把"未记录闭包"的结果伪装成已验证产物。

    Raises:
        ValueError: 严格模式下 R1D 新结果缺少依赖清单，或推导失败。
    """
    from_manifest = _result_dependency_manifest(result)
    if from_manifest is not None:
        if from_manifest:
            return from_manifest, "result", []
        if REQUIRE_RESULT_DEPENDENCY_MANIFEST and str(
            getattr(result, "robot_id", "")
        ) == "unitree_r1d":
            raise ValueError(
                "R1D 生产结果缺少装配依赖清单（model_dependency_manifest 为空），"
                "拒绝写入无法证明装配闭包的产物："
                f"{getattr(result, 'robot_id', '')}"
            )
    _LOGGER.warning(
        "回合结果未携带装配依赖清单（robot=%s），退化为按配置推导并在产物中标记 "
        "dependency_manifest_source=derived；该清单不保证覆盖运行期所有装配分支。",
        getattr(result, "robot_id", ""),
    )
    entries, missing = _derive_dependency_entries(payload)
    return entries, "derived", missing


def _copy_verified(
    source: Path,
    destination: Path,
    expected_sha256: str,
    asset_id: str,
) -> None:
    """复制文件并对**复制出的字节**重新校验哈希（防"复制前篡改"）。

    Raises:
        ValueError: 复制后的字节哈希与记录不一致（错误信息带逻辑 asset_id）。
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(source), destination)
    actual = _sha256_file(destination)
    if actual != str(expected_sha256):
        raise ValueError(
            "装配依赖资产在复制前被改动（拒绝使用）："
            f"asset_id={asset_id} 源={source} 期望={expected_sha256} 实际={actual}"
        )


def _write_dependency_copies(
    output_dir: Path,
    entries: Sequence[dict[str, Any]],
    role_entries: Mapping[str, Mapping[str, Any]],
) -> None:
    """把需要自包含的依赖复制进快照，并写入 ``snapshot_path``。

    XML 片段（模型/include/兄弟片段）体积小且是装配输入，全部复制；仓库外
    资产复制（外部目录被移动/删除后仍可回放）；仓库内网格只记相对路径与
    哈希（体积大，不重复存储）。

    Raises:
        ValueError: 复制后哈希不一致（源文件在写入过程中被改动）。
    """
    snapshot_dir = Path(output_dir) / SNAPSHOT_DIR_NAME
    copies_dir = snapshot_dir / SNAPSHOT_FILES_DIR_NAME / _DEPENDENCY_COPY_DIRNAME
    for entry in entries:
        role = str(entry["role"])
        relative = str(entry["relative_path"])
        if role == DEPENDENCY_ROLE_MODEL:
            role_entry = role_entries.get("model")
            if role_entry is not None and str(role_entry.get("sha256")) == str(
                entry["sha256"]
            ):
                entry["snapshot_path"] = str(role_entry["relative_path"])
                continue
        need_copy = role in DEPENDENCY_XML_ROLES or relative.startswith("external/")
        if not need_copy:
            continue
        source = Path(str(entry["source"]))
        asset_id = str(entry["asset_id"])
        destination = copies_dir / asset_id
        _copy_verified(source, destination, str(entry["sha256"]), asset_id)
        entry["snapshot_path"] = (
            f"{SNAPSHOT_FILES_DIR_NAME}/{_DEPENDENCY_COPY_DIRNAME}/{asset_id}"
        )


def _collect_verified_assets(
    snapshot: Mapping[str, Any],
    project_root: Path,
) -> tuple[dict[str, Path], list[str]]:
    """把依赖条目解析为 ``asset_id -> 已验证路径``，并收集问题（不抛异常）。

    解析顺序（每个条目独立）：快照内副本 → 仓库相对路径 → 记录的外部
    绝对来源。只有 SHA256 与条目一致的文件才会被选中；因此"旧检出的同名
    文件"即使被替换/放大也会被拒绝，而不会被误当成已验证资产。

    Args:
        snapshot: 依赖视图（含 ``dependencies.files`` 或 legacy ``assets``
            条目；可带 ``snapshot_dir`` 指向快照根目录）。
        project_root: 仓库根目录（解析仓库内相对路径）。

    Returns:
        ``(resolved, problems)``：``resolved`` 为 ``asset_id -> 绝对路径``；
        ``problems`` 为按逻辑 ID 描述的问题列表（空表示全部通过）。
    """
    entries, _legacy = _dependency_entries(snapshot)
    snapshot_dir_value = snapshot.get("snapshot_dir")
    snapshot_dir = Path(str(snapshot_dir_value)) if snapshot_dir_value else None
    resolved: dict[str, Path] = {}
    problems: list[str] = []
    for entry in entries:
        role = str(entry.get("role") or DEPENDENCY_ROLE_MESH)
        digest = str(entry.get("sha256") or "")
        relative = str(entry.get("relative_path") or "")
        source = str(entry.get("source") or "")
        name = Path(relative or source).name or "asset"
        asset_id = str(entry.get("asset_id") or _asset_id(role, digest, name))
        if not digest:
            problems.append(f"{asset_id}（role={role}）缺少 sha256，无法校验")
            continue
        candidates: list[Path] = []
        snapshot_path = entry.get("snapshot_path")
        if snapshot_path and snapshot_dir is not None:
            candidates.append(snapshot_dir / str(snapshot_path))
        if relative and not relative.startswith("external/"):
            candidates.append(Path(project_root) / relative)
        if source:
            candidates.append(Path(source))
        found: Path | None = None
        for candidate in candidates:
            try:
                if candidate.is_file() and _sha256_file(candidate) == digest:
                    found = candidate.resolve()
                    break
            except OSError:  # pragma: no cover - 权限/临时消失
                continue
        if found is None:
            problems.append(
                f"{asset_id}（role={role}，relative_path={relative or '<空>'}，"
                f"候选 {len(candidates)} 个均缺失或哈希不一致）"
            )
            continue
        resolved[asset_id] = found
    return resolved, problems


def resolve_verified_assets(
    snapshot: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Path]:
    """把快照依赖清单解析为 ``稳定逻辑 asset_id -> 已验证文件路径``（冻结接口）。

    这是 F02 的修复核心：**校验与消费共用同一份映射**。每个条目按
    "快照副本 → 仓库相对路径 → 记录来源"的顺序找**哈希匹配**的文件，
    返回的路径就是回放必须加载的文件；不再有任何"按记录来源重新挑选"
    的第二套逻辑。

    Args:
        snapshot: 依赖视图。可以是快照 ``manifest.json``（可带
            ``snapshot_dir`` 指向快照根目录），也可以是产物 payload；
            旧 artifact 没有 ``dependencies`` 段时按 ``assets.files``
            合成条目。
        project_root: 仓库根目录（仓库内 ``relative_path`` 的解析基准）。

    Returns:
        ``asset_id -> 绝对路径``（只含哈希匹配的文件）。

    Raises:
        ValueError: 任一依赖缺失/哈希不一致；错误信息按逻辑 asset_id 列出
            全部问题（便于定位到具体依赖，而非某次绝对路径）。
    """
    resolved, problems = _collect_verified_assets(snapshot, Path(project_root))
    if problems:
        raise ValueError(
            "装配依赖资产未通过哈希校验（拒绝回放）：" + "；".join(problems)
        )
    return resolved


def _write_snapshot(
    output_dir: Path,
    config: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """复制配置源文件到快照目录，返回 (manifest 条目表, input_sha256)。

    Returns:
        ``(entries_by_role, input_sha256)``：条目表含 role/source/
        relative_path/sha256/size；``input_sha256`` 只含 INPUT_SHA_ROLES 中
        实际存在的角色（缺失角色为空串），并额外包含 scene_snapshot
        （由调用方在写入前补齐）。
    """
    snapshot_dir = output_dir / SNAPSHOT_DIR_NAME
    files_dir = snapshot_dir / SNAPSHOT_FILES_DIR_NAME
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    files_dir.mkdir(parents=True, exist_ok=True)

    entries_by_role: dict[str, dict[str, Any]] = {}
    input_sha256: dict[str, str] = {}
    for role in SNAPSHOT_ROLES:
        source = _role_paths(config).get(role)
        if source is None:
            continue
        source = Path(source)
        if not source.is_file():
            _LOGGER.warning("快照源文件不存在，跳过角色 %s：%s", role, source)
            input_sha256[role] = ""
            continue
        relative = Path(SNAPSHOT_FILES_DIR_NAME) / _snapshot_relative_path(source)
        dest = snapshot_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        digest = _sha256_file(dest)
        entries_by_role[role] = {
            "role": role,
            "source": str(source.resolve()),
            "relative_path": relative.as_posix(),
            "sha256": digest,
            "size": int(dest.stat().st_size),
        }
        input_sha256[role] = digest
    return entries_by_role, input_sha256


def _write_snapshot_metadata(
    output_dir: Path,
    entries: Mapping[str, Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> Path:
    """写快照的三个元数据文件并返回快照目录。"""
    snapshot_dir = output_dir / SNAPSHOT_DIR_NAME
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    assets_section = payload.get("assets") or {}
    manifest = {
        "schema": EPISODE_SCHEMA,
        "episode_schema": payload.get("schema"),
        "robot_id": payload.get("robot_id"),
        "files": [dict(entries[role]) for role in SNAPSHOT_ROLES if role in entries],
        # 装配依赖闭包清单（F08）：与 payload["dependencies"] 同源，回放按
        # asset_id 解析哈希匹配文件；旧 artifact 无此段 → legacy 分支。
        "dependency_manifest_version": payload.get("dependency_manifest_version"),
        "dependency_manifest_source": payload.get("dependency_manifest_source"),
        "dependencies": payload.get("dependencies"),
        # 依赖资产（mesh/texture）逐文件条目：不复制大文件，只记录路径与
        # 哈希；回放时按仓库现有文件重新校验（T6 追加，设计 §6）。
        "assets": {
            "status": assets_section.get("status"),
            "count": assets_section.get("count"),
            "digest": assets_section.get("digest"),
            "missing": [str(item) for item in assets_section.get("missing") or []],
            "files": [dict(item) for item in assets_section.get("files") or []],
        },
    }
    (snapshot_dir / SNAPSHOT_MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (snapshot_dir / SNAPSHOT_CONFIG_NAME).write_text(
        json.dumps(
            payload.get("config", {}), ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    (snapshot_dir / SNAPSHOT_SCENE_NAME).write_text(
        json.dumps(
            payload.get("scene_snapshot", {}),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return snapshot_dir


# ---------------------------------------------------------------------------
# 数组与 payload 构造
# ---------------------------------------------------------------------------
def _arrays_from_result(result: V12EpisodeResult) -> dict[str, np.ndarray]:
    """从回合结果提取并校验产物的全部数组（形状/对齐）。"""
    joint_names = tuple(str(name) for name in result.joint_names)
    n = len(joint_names)
    X = np.asarray(result.X, dtype=float)
    U = np.asarray(result.U, dtype=float)
    ball_pos = np.asarray(result.ball_pos, dtype=float)
    ball_vel = np.asarray(result.ball_vel, dtype=float)
    time_arr = np.asarray(result.time, dtype=float)
    racket_pos = np.asarray(result.racket_pos, dtype=float)

    if X.ndim != 2 or X.shape[1] != 2 * n:
        raise ValueError(
            f"X 形状 {X.shape} 与 joint_names({n}) 不匹配，期望 (N+1, {2 * n})"
        )
    if U.ndim != 2 or U.shape[1] != n:
        raise ValueError(
            f"U 形状 {U.shape} 与 joint_names({n}) 不匹配，期望 (N, {n})"
        )
    n_steps = int(U.shape[0])
    if int(X.shape[0]) != n_steps + 1:
        raise ValueError(
            f"X 行数({X.shape[0]}) 必须等于 U 行数({n_steps}) + 1"
        )
    for name, arr in (
        ("ball_pos", ball_pos),
        ("ball_vel", ball_vel),
        ("time", time_arr),
        ("racket_pos", racket_pos),
    ):
        expected_rows = n_steps + 1
        if arr.shape[0] != expected_rows:
            raise ValueError(
                f"{name} 行数({arr.shape[0]}) 必须等于 N+1({expected_rows})"
            )
    if ball_pos.shape[1:] != (3,):
        raise ValueError(f"ball_pos 形状 {ball_pos.shape} 期望 (N+1, 3)")
    if ball_vel.shape[1:] != (3,):
        raise ValueError(f"ball_vel 形状 {ball_vel.shape} 期望 (N+1, 3)")
    if racket_pos.shape[1:] != (3,):
        raise ValueError(f"racket_pos 形状 {racket_pos.shape} 期望 (N+1, 3)")
    for name, arr in (
        ("X", X),
        ("U", U),
        ("ball_pos", ball_pos),
        ("ball_vel", ball_vel),
        ("time", time_arr),
        ("racket_pos", racket_pos),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} 含非有限值（NaN/Inf），拒绝写入产物")
    return {
        "X": X,
        "U": U,
        "ball_pos": ball_pos,
        "ball_vel": ball_vel,
        "time": time_arr,
        "racket_pos": racket_pos,
    }


def _scene_snapshot(result: V12EpisodeResult) -> dict[str, Any]:
    """构造统一场景快照（R1D 取 r1d_artifact.fixed_scene_params）。"""
    config = result.config
    metrics = result.metrics or {}
    r1d_artifact = metrics.get("r1d_artifact")
    source = "metrics.r1d_artifact.fixed_scene_params"
    if isinstance(r1d_artifact, Mapping) and "fixed_scene_params" in r1d_artifact:
        parameters = _jsonable(r1d_artifact["fixed_scene_params"])
    else:
        source = "config.scene_config"
        parameters = _jsonable(getattr(config, "scene_config", {}) or {})
    return {
        "source": source,
        "parameters": parameters,
        "target_center": _jsonable(getattr(config, "target_center", None)),
        "target_offset": _jsonable(getattr(config, "target_offset", None)),
        "workspace_radius": _jsonable(getattr(config, "workspace_radius", None)),
        "ball_initial": {
            "position": _jsonable(np.asarray(result.ball_pos[0], dtype=float)),
            "velocity": _jsonable(np.asarray(result.ball_vel[0], dtype=float)),
        },
        "shoulder_pos": _jsonable(metrics.get("shoulder_pos")),
        "shoulder_pos_source": metrics.get("shoulder_pos_source"),
        "proximity": _jsonable(metrics.get("proximity", {})),
        "dt": float(getattr(config, "dt", 0.0)),
        "seed": getattr(config, "seed", None),
    }


#: robot_id → 人类可读显示名（旧 R1D v1 文案使用 "R1-A7-D"；RM65 默认不变）
_ROBOT_DISPLAY_NAMES: dict[str, str] = {
    "rm65": "RM-65",
    "unitree_r1d": "R1-A7-D",
}


def _robot_display_name(robot_id: Any) -> str:
    """返回产物 ``hit.message`` 使用的机器人显示名（评审 N1 修复）。"""
    key = str(robot_id or "").strip()
    if key in _ROBOT_DISPLAY_NAMES:
        return _ROBOT_DISPLAY_NAMES[key]
    return key or "RM-65"


def _hit_section(result: V12EpisodeResult) -> dict[str, Any]:
    """构造 ``payload["hit"]``（legacy 与 physical 口径独立）。"""
    from src.sim.hit_detection import classify_hit_result

    metrics = result.metrics or {}
    legacy_hit = bool(metrics.get("legacy_hit", result.hit_type in ("active", "passive")))
    contact_events = _jsonable(metrics.get("contact_events", []))
    divergence = metrics.get("legacy_vs_physical_divergence")
    hit_step = result.hit_step
    if hit_step is None or int(hit_step) < 0:
        time_s = None
    else:
        time_s = float(int(hit_step)) * float(getattr(result.config, "dt", 0.0))
    pos_error = metrics.get("pos_error")
    message = metrics.get("hit_message")
    if message is None:
        try:
            message = classify_hit_result(
                bool(result.active_contact),
                bool(result.passive_contact),
                float(pos_error) if pos_error is not None else 0.0,
                robot_name=_robot_display_name(result.robot_id),
            )
        except (TypeError, ValueError):  # pragma: no cover - 防御异常值
            message = None
    return {
        "legacy_hit": legacy_hit,
        "legacy_hit_type": result.hit_type,
        "legacy_hit_rule": metrics.get(
            "legacy_hit_rule", "first_ball_racket_contact_in_collision_window"
        ),
        "physical_hit": bool(metrics.get("physical_hit", False)),
        "physical_hit_type": metrics.get("physical_hit_type", "miss"),
        "physical_hit_rule": metrics.get(
            "physical_hit_rule", "first_physical_face_contact_face_speed_0p3"
        ),
        "physical_hit_step": metrics.get("physical_hit_step", -1),
        "legacy_vs_physical_divergence": _jsonable(divergence),
        "contact_events": contact_events,
        "step": None if hit_step is None else int(hit_step),
        "time_s": time_s,
        "pos_error": _jsonable(metrics.get("pos_error")),
        "min_dist": _jsonable(metrics.get("min_dist")),
        "v_racket_at_hit": _jsonable(result.v_ee_at_hit),
        "ball_out_speed": _jsonable(result.ball_out_speed),
        "ball_out_vy": _jsonable(result.ball_out_vy),
        "message": message,
    }


def build_episode_payload(result: V12EpisodeResult) -> dict[str, Any]:
    """把回合结果构造成完整、JSON 可序列化的 v2 payload（不落盘）。

    Args:
        result: 任意来源的 :class:`V12EpisodeResult`（真实运行或测试构造）。

    Returns:
        ``v12_episode_v2`` payload 字典（含 ``input_sha256`` 与
        ``scene_snapshot``）。
    """
    metrics = result.metrics or {}
    config = result.config
    joint_names = [str(name) for name in result.joint_names]
    mode = str(result.mode)
    scene_snapshot = _scene_snapshot(result)
    role_paths = _role_paths(config)
    input_sha256 = {role: "" for role in INPUT_SHA_ROLES}
    for role in ("robot_profile", "runtime", "task", "racket", "model"):
        path = role_paths.get(role)
        if path is not None and Path(path).is_file():
            input_sha256[role] = _sha256_file(Path(path))
    input_sha256["scene_snapshot"] = hashlib.sha256(
        _canonical_json(scene_snapshot).encode("utf-8")
    ).hexdigest()
    # 依赖资产（模型 XML 引用的 mesh/texture）逐文件哈希 + 聚合哈希；
    # 不复制资产文件，回放时按仓库路径重新校验，避免 mesh 被改而模型
    # 哈希不变导致回放几何悄悄变化（T6 追加，设计 §6）。
    assets = _assets_section(role_paths)
    input_sha256["assets"] = str(assets["digest"])

    backend = metrics.get("backend")
    if isinstance(backend, Mapping) and isinstance(backend.get("effective"), str):
        effective = str(backend["effective"])
        finite_difference = effective.endswith("fd")
    else:
        effective = None
        finite_difference = None

    return {
        "schema": EPISODE_SCHEMA,
        "simulation_only": True,
        "robot_id": result.robot_id,
        "mode": mode,
        "racket_id": result.racket_id,
        "joint_names": joint_names,
        "dt": float(getattr(config, "dt", 0.0)),
        "seed": getattr(config, "seed", None),
        "config": config_to_dict(config),
        "input_sha256": input_sha256,
        "scene_snapshot": scene_snapshot,
        "assets": assets,
        "target": _jsonable(getattr(config, "target_center", None)),
        "workspace_radius": _jsonable(getattr(config, "workspace_radius", None)),
        "ball_initial": {
            "position": _jsonable(np.asarray(result.ball_pos[0], dtype=float)),
            "velocity": _jsonable(np.asarray(result.ball_vel[0], dtype=float)),
        },
        "backend": {
            "requested": backend.get("requested") if isinstance(backend, Mapping) else None,
            "effective": effective,
            "solver_backend": getattr(config, "solver_backend", None),
            "solver_config": _jsonable(getattr(config, "solver_config", None)),
            "finite_difference": finite_difference,
        },
        "phase_timings": _jsonable(metrics.get("phase_timings", {})),
        "wall_time_s": _jsonable(metrics.get("wall_time_s")),
        "failure_reason": metrics.get("failure_reason"),
        "episode": {
            "total_steps": metrics.get("total_steps"),
            "safe_steps": metrics.get("safe_steps"),
            "mpc_done": metrics.get("mpc_done"),
            "first_plan_ms": metrics.get("first_plan_ms"),
            "wall_time_s": metrics.get("wall_time_s"),
            "failure_reason": metrics.get("failure_reason"),
        },
        "hit": _hit_section(result),
        "units": dict(UNITS),
        "control": {
            "mode": mode,
            "units": UNITS["control_torque"] if mode == "torque" else UNITS["control_position"],
        },
        "proximity": _jsonable(metrics.get("proximity", {})),
        "note": (
            "hit.legacy_* 为旧版第一接触口径（V11/RM65 行为保持）；hit.physical_* "
            "为 T6 新增的口径（首帧拍面真实接触 + 0.3 m/s 拍速阈值）。"
        ),
    }


# ---------------------------------------------------------------------------
# 写入 / 读取
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EpisodeArtifacts:
    """已落盘的 v2 产物（payload + 数组 + 目录）。"""

    payload: dict[str, Any]
    arrays: dict[str, np.ndarray]
    directory: Path

    #: 便捷访问（与 NPZ 键一一对应）
    @property
    def X(self) -> np.ndarray:
        return self.arrays["X"]

    @property
    def U(self) -> np.ndarray:
        return self.arrays["U"]

    @property
    def ball_pos(self) -> np.ndarray:
        return self.arrays["ball_pos"]

    @property
    def ball_vel(self) -> np.ndarray:
        return self.arrays["ball_vel"]

    @property
    def time(self) -> np.ndarray:
        return self.arrays["time"]

    @property
    def racket_pos(self) -> np.ndarray:
        return self.arrays["racket_pos"]


def write_episode_artifacts(result: V12EpisodeResult, output_dir: Path) -> dict[str, Any]:
    """把回合结果写为 ``v12_episode_v2`` JSON + NPZ + 快照。

    Args:
        result: 回合结果（成功或失败均可）。
        output_dir: 输出目录（不存在则创建）。

    Returns:
        写入的 payload 字典。

    Raises:
        ValueError: 数组形状/对齐/joint_names 不一致，或含非有限值。
        OSError: 目录/文件写入失败。
    """
    from src.sim.v12_episode import V12EpisodeResult  # 延迟导入避免环

    if not isinstance(result, V12EpisodeResult):
        raise TypeError(
            f"write_episode_artifacts 只接受 V12EpisodeResult，收到 {type(result)!r}"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    arrays = _arrays_from_result(result)
    payload = build_episode_payload(result)

    entries, snapshot_sha = _write_snapshot(output, result.config)
    for role in ("robot_profile", "runtime", "task", "racket", "model"):
        if role in snapshot_sha and snapshot_sha[role]:
            payload["input_sha256"][role] = snapshot_sha[role]
    # F08：装配依赖闭包（模板/include/兄弟片段/网格/纹理）。条目来自装配
    # 过程本身（result.model_dependency_manifest）；字段尚未落地时按配置
    # 推导并显式标记 derived（弱证据，不假装已验证闭包）。
    dependency_entries, dependency_source, dependency_missing = (
        _build_dependency_entries(result, payload)
    )
    _write_dependency_copies(output, dependency_entries, entries)
    payload["dependency_manifest_version"] = DEPENDENCY_MANIFEST_VERSION
    payload["dependency_manifest_source"] = dependency_source
    payload["dependencies"] = {
        "version": DEPENDENCY_MANIFEST_VERSION,
        "source": dependency_source,
        "files": [dict(entry) for entry in dependency_entries],
        "missing": list(dependency_missing),
    }
    _write_snapshot_metadata(output, entries, payload)

    json_path = output / ARTIFACT_JSON_NAME
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.savez(
        output / ARTIFACT_NPZ_NAME,
        schema=np.array(EPISODE_SCHEMA),
        simulation_only=np.array(True),
        **arrays,
    )
    _LOGGER.info(
        "v2 产物已写入 %s（robot=%s, hit=%s, physical=%s）",
        output,
        payload["robot_id"],
        payload["hit"]["legacy_hit_type"],
        payload["hit"]["physical_hit_type"],
    )
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 文件并断言顶层为对象。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return data


def load_episode_artifacts(output_dir: Path) -> EpisodeArtifacts:
    """读取 ``v12_episode_v2`` 产物（JSON + NPZ）。

    Raises:
        FileNotFoundError: JSON 或 NPZ 缺失。
        ValueError: schema 不是 ``v12_episode_v2`` 或 NPZ 缺键。
    """
    directory = Path(output_dir)
    json_path = directory / ARTIFACT_JSON_NAME
    npz_path = directory / ARTIFACT_NPZ_NAME
    if not json_path.is_file():
        raise FileNotFoundError(f"缺少产物 JSON：{json_path}")
    if not npz_path.is_file():
        raise FileNotFoundError(f"缺少产物 NPZ：{npz_path}")
    payload = _read_json(json_path)
    if payload.get("schema") != EPISODE_SCHEMA:
        raise ValueError(
            f"产物 schema 不是 {EPISODE_SCHEMA}：{payload.get('schema')!r}"
        )
    keys = ("X", "U", "ball_pos", "ball_vel", "time", "racket_pos")
    arrays: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [key for key in keys if key not in data.files]
        if missing:
            raise ValueError(f"产物 NPZ 缺少数组：{missing}")
        for key in keys:
            arrays[key] = np.asarray(data[key], dtype=float)
    return EpisodeArtifacts(payload=payload, arrays=arrays, directory=directory)


# ---------------------------------------------------------------------------
# 快照校验与回放
# ---------------------------------------------------------------------------
def _snapshot_entries(output_dir: Path) -> dict[str, dict[str, Any]]:
    """读取快照 manifest，返回 ``role -> entry`` 映射（缺失则空）。"""
    manifest_path = Path(output_dir) / SNAPSHOT_DIR_NAME / SNAPSHOT_MANIFEST_NAME
    if not manifest_path.is_file():
        return {}
    manifest = _read_json(manifest_path)
    entries: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("files", []):
        if isinstance(entry, Mapping) and entry.get("role"):
            entries[str(entry["role"])] = dict(entry)
    return entries


def _external_source_for_role(
    payload: Mapping[str, Any], role: str
) -> Path | None:
    """按 payload 的 config 快照解析角色的外部源文件路径。"""
    config = config_from_dict(payload["config"])
    return _role_paths(config).get(role)


def _verify_model_assets(
    payload: Mapping[str, Any],
    output_dir: Path,
) -> list[str]:
    """校验依赖资产（mesh/texture）哈希，返回问题列表（空表示通过）。

    新产物（``input_sha256`` 含 ``assets`` 键）必须逐个校验 payload 记录的
    资产文件并核对聚合哈希；仅当 manifest 确实带有 ``assets.files`` 时才
    额外比对 manifest 与 payload 的条目数与逐项哈希。快照整体缺失/旧产物
    没有该段时跳过数量比较（改按 payload 摘要验证外部文件），不再误报
    “0 vs N”（评审 R12）。

    候选顺序（M05）：仓库内路径 → 产物记录的来源 → **快照副本**。前两类
    只要有一个**存在**就以它们的字节为准——存在但哈希不符视为篡改并拒绝
    （不能用快照副本掩盖）；仅当两类来源都不存在（换机、清理旧检出）时才
    回退到快照副本，且仍按同一 sha256 校验，校验强度不变。

    Args:
        payload: ``v12_episode_v2`` 产物字典。
        output_dir: 产物目录（快照位于其下的 ``snapshot/``）。

    Returns:
        问题列表；空列表表示资产字节与写入时一致。
    """
    sha_table = payload.get("input_sha256") or {}
    if "assets" not in sha_table:
        _LOGGER.warning(
            "产物缺少依赖资产哈希（旧 artifact），跳过 mesh/texture 校验：%s",
            output_dir,
        )
        return []
    expected_digest = str(sha_table.get("assets") or "")
    section = payload.get("assets") or {}
    files = [dict(item) for item in section.get("files") or [] if isinstance(item, Mapping)]
    missing_at_write = [str(item) for item in section.get("missing") or []]

    problems: list[str] = []
    if missing_at_write:
        problems.append(
            "写入时依赖资产缺失（回放无法保证几何一致）："
            + ", ".join(missing_at_write)
        )

    dependency_section = _dependency_section(payload)
    dependency_files = [
        dict(item)
        for item in (dependency_section or {}).get("files") or []
        if isinstance(item, Mapping)
    ]
    snapshot_dir = Path(output_dir) / SNAPSHOT_DIR_NAME

    def _snapshot_candidates_for(item: Mapping[str, Any]) -> list[Path]:
        """该资产在快照内的副本候选（自身字段优先，再按依赖清单同源条目）。"""
        candidates: list[Path] = []
        own = item.get("snapshot_path")
        if own:
            candidates.append(snapshot_dir / str(own))
        expected_sha = str(item.get("sha256") or "")
        raw_source = str(item.get("source") or "")
        source_path = Path(raw_source).resolve() if raw_source else None
        for dependency in dependency_files:
            if str(dependency.get("role")) not in ASSET_XML_TAGS:
                continue
            if str(dependency.get("sha256") or "") != expected_sha:
                continue
            recorded = str(dependency.get("source") or "")
            if source_path is not None and recorded:
                if Path(recorded).resolve() != source_path:
                    continue
            snapshot_path = dependency.get("snapshot_path")
            if not snapshot_path:
                continue
            candidate = snapshot_dir / str(snapshot_path)
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    manifest_path = Path(output_dir) / SNAPSHOT_DIR_NAME / SNAPSHOT_MANIFEST_NAME
    manifest: dict[str, Any] | None = None
    manifest_files: list[dict[str, Any]] | None = None
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        manifest_assets = manifest.get("assets")
        # 仅当 manifest 的 assets 段确实带 files 键时才逐项比对：快照整体
        # 缺失或旧产物没有该段时，不能凭空要求条目数相等（评审 R12）。
        if isinstance(manifest_assets, Mapping) and "files" in manifest_assets:
            manifest_files = [
                dict(item)
                for item in manifest_assets.get("files") or []
                if isinstance(item, Mapping)
            ]
    if manifest_files is not None:
        if len(manifest_files) != len(files):
            problems.append(
                "manifest 依赖资产条目数"
                f"({len(manifest_files)})与 payload({len(files)})不一致"
            )
        else:
            manifest_sha = {
                str(item.get("relative_path")): str(item.get("sha256"))
                for item in manifest_files
            }
            for item in files:
                relative = str(item.get("relative_path"))
                if manifest_sha.get(relative) != str(item.get("sha256")):
                    problems.append(f"manifest 依赖资产哈希被篡改/不一致：{relative}")

    verified: list[dict[str, Any]] = []
    for item in files:
        relative = str(item.get("relative_path") or "")
        expected_sha = str(item.get("sha256") or "")
        source = str(item.get("source") or "")
        # 第一优先级：仓库内路径与产物记录的来源；只要文件存在就以它们的
        # 字节为准（存在但哈希不符 = 篡改，必须拒绝，不能用快照掩盖）
        direct: list[Path] = []
        if relative and not relative.startswith("external/"):
            direct.append(_repo_root() / relative)
        if source:
            direct.append(Path(source))
        found: Path | None = None
        for candidate in direct:
            if candidate.is_file() and _sha256_file(candidate) == expected_sha:
                found = candidate
                break
        if found is None and not any(candidate.is_file() for candidate in direct):
            # 来源整体消失（换机/清理旧检出）：回退快照副本（M05），仍按同一
            # sha256 校验字节，校验强度不变
            for candidate in _snapshot_candidates_for(item):
                if candidate.is_file() and _sha256_file(candidate) == expected_sha:
                    found = candidate
                    break
        if found is None:
            problems.append(f"依赖资产缺失或哈希不一致：{relative}（source={source}）")
        else:
            verified.append({"relative_path": relative, "sha256": expected_sha})
    if expected_digest and _assets_digest(verified) != expected_digest:
        problems.append("依赖资产聚合哈希不一致（input_sha256['assets']）")

    # F08：装配依赖闭包逐条目校验（按稳定逻辑 asset_id 报错），并核对
    # 快照 manifest 与 payload 的依赖清单一致——防止清单被改小/替换后
    # 仍宣称"闭包已验证"。旧 artifact 没有该段时跳过（legacy 已在上层告警）。
    if (
        dependency_section is not None
        and int(dependency_section.get("version") or 0) >= 1
    ):
        view: dict[str, Any] = {
            "dependencies": dependency_section,
            "snapshot_dir": str(Path(output_dir) / SNAPSHOT_DIR_NAME),
        }
        _resolved, dependency_problems = _collect_verified_assets(view, _repo_root())
        problems.extend(dependency_problems)
        missing_dependencies = [
            str(item) for item in dependency_section.get("missing") or []
        ]
        if missing_dependencies:
            problems.append(
                "写入时装配依赖缺失（回放无法保证装配闭包）："
                + ", ".join(missing_dependencies)
            )
        payload_dependencies = [
            dict(item)
            for item in dependency_section.get("files") or []
            if isinstance(item, Mapping)
        ]
        if manifest is not None and isinstance(manifest.get("dependencies"), Mapping):
            manifest_dependencies = [
                dict(item)
                for item in (manifest.get("dependencies") or {}).get("files") or []
                if isinstance(item, Mapping)
            ]
            if len(manifest_dependencies) != len(payload_dependencies):
                problems.append(
                    "manifest 装配依赖条目数"
                    f"({len(manifest_dependencies)})与 payload"
                    f"({len(payload_dependencies)})不一致"
                )
            else:
                manifest_sha = {
                    str(item.get("asset_id")): str(item.get("sha256"))
                    for item in manifest_dependencies
                }
                for item in payload_dependencies:
                    asset_id = str(item.get("asset_id"))
                    if manifest_sha.get(asset_id) != str(item.get("sha256")):
                        problems.append(
                            f"manifest 装配依赖条目被篡改/不一致：{asset_id}"
                        )
    return problems


def _verify_and_resolve_sources(
    payload: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """校验快照/外部源文件哈希，返回可用于回放的 ``role -> Path``。

    快照优先、外部源兜底：同一角色只要有一个候选文件存在且哈希匹配即
    通过，成功候选会清除该角色的缺失/不一致信息；只有**所有**候选都不
    可用时才拒绝回放，错误信息按角色列出各候选的失败原因（评审 R12）。

    例外：``model`` 角色的快照副本存在但哈希不一致时视为快照被篡改，
    即使外部模型碰巧匹配也拒绝（评审 R11 的“快照优先”契约）。

    Args:
        payload: v2 产物 payload。
        output_dir: 产物目录（快照根目录）。

    Returns:
        ``role -> Path``（仅包含通过校验的角色）。

    Raises:
        ValueError: 某角色所有候选均不可用、``model`` 快照哈希不一致，
            或依赖资产校验失败。
    """
    expected: dict[str, str] = {
        role: str(payload.get("input_sha256", {}).get(role, "") or "")
        for role in INPUT_SHA_ROLES
        if role not in {"scene_snapshot", "assets"}
    }
    entries = _snapshot_entries(output_dir)
    resolved: dict[str, Path] = {}
    problems: list[str] = []
    role_failures: dict[str, list[str]] = {}
    tampered_snapshots: dict[str, list[str]] = {}
    for role, digest in expected.items():
        if not digest:
            continue
        candidates: list[tuple[str, Path]] = []
        # 机器人描述优先用外部原文件：其相对 model_path 需要按原始目录
        # 解析（快照副本会指向快照内未复制的 mesh 资产，无法编译）；
        # runtime/task/racket/model 优先用快照副本，保证外部修改不影响回放。
        external = _external_source_for_role(payload, role)
        entry = entries.get(role)
        snapshot_path: Path | None = None
        if entry is not None and entry.get("relative_path"):
            snapshot_path = (
                Path(output_dir)
                / SNAPSHOT_DIR_NAME
                / str(entry["relative_path"])
            )
        if role == "robot_profile":
            if external is not None:
                candidates.append(("external", Path(external)))
            if snapshot_path is not None:
                candidates.append(("snapshot", snapshot_path))
        else:
            if snapshot_path is not None:
                candidates.append(("snapshot", snapshot_path))
            if external is not None:
                candidates.append(("external", Path(external)))
        if not candidates:
            role_failures[role] = ["无可用候选来源（快照与外部配置均缺失）"]
            continue
        chosen: Path | None = None
        failures: list[str] = []
        snapshot_mismatch = False
        for origin, candidate_path in candidates:
            if not candidate_path.is_file():
                failures.append(f"{origin} 文件缺失：{candidate_path}")
                continue
            try:
                actual = _sha256_file(candidate_path)
            except OSError as exc:  # pragma: no cover - 并发删除/权限问题
                failures.append(f"{origin} 文件不可读：{candidate_path}（{exc}）")
                continue
            if actual == digest:
                chosen = candidate_path
                break
            failures.append(f"{origin} 哈希不一致：{candidate_path}")
            if origin == "snapshot":
                snapshot_mismatch = True
        if chosen is not None:
            resolved[role] = chosen
        if snapshot_mismatch and role == "model":
            # R11：模型快照存在但内容与产物记录不符 → 快照被篡改/损坏，硬拒绝，
            # 不允许“外部文件碰巧匹配”掩盖快照损坏。其他角色按 R12 语义：
            # 匹配的外部候选即可清除该角色的失败（成功候选优先）。
            tampered_snapshots[role] = failures
        elif chosen is None:
            role_failures[role] = failures
    for role, failures in tampered_snapshots.items():
        problems.append(
            f"{role} 快照副本哈希与产物记录不一致（快照被篡改/损坏，"
            f"拒绝回放）：{'; '.join(failures)}"
        )
    for role, failures in role_failures.items():
        problems.append(f"{role} 所有候选来源均不可用：{'; '.join(failures)}")
    problems.extend(_verify_model_assets(payload, output_dir))
    if problems:
        raise ValueError(
            "回放来源不可信：快照缺失/损坏且外部文件哈希不一致，拒绝回放。"
            f" 详情：{'; '.join(problems)}"
        )
    return resolved


def _replay_config(payload: Mapping[str, Any], sources: Mapping[str, Path]) -> Any:
    """由 payload 与校验后的来源重建回放用 ``ResolvedV12Config``。"""
    config = config_from_dict(payload["config"])
    replacements = {
        "robot_profile_path": sources.get("robot_profile"),
        "runtime_path": sources.get("runtime"),
        "task_path": sources.get("task"),
        "racket_path": sources.get("racket"),
    }
    updates = {
        name: path
        for name, path in replacements.items()
        if path is not None
    }
    if not updates:
        return config
    from dataclasses import replace

    return replace(config, **updates)


def _resolved_model_path(payload: Mapping[str, Any], sources: Mapping[str, Path]) -> Path | None:
    """解析回放要加载的机器人模型文件路径（快照优先，回退 profile）。

    评审 R11：:func:`_verify_and_resolve_sources` 选出的
    ``sources["model"]`` 才是通过哈希校验的模型快照，回放必须加载它；
    只有快照缺失（``sources`` 无 ``model``）时才从 robot profile 解析外部
    模型，避免快照完好却仍使用被改动/删除的外部模型。
    """
    snapshot_model = sources.get("model")
    if snapshot_model is not None:
        return Path(snapshot_model)
    profile_path = sources.get("robot_profile")
    if profile_path is None:
        profile_path = _external_source_for_role(payload, "robot_profile")
    if profile_path is None:
        return None
    try:
        return Path(_model_path_for_profile(Path(profile_path)))
    except (OSError, ValueError):
        return None


def _verify_loaded_model(
    payload: Mapping[str, Any],
    sources: Mapping[str, Path],
) -> Path | None:
    """校验回放**实际加载**的模型文件哈希（评审 B1 修复）。

    ``_verify_and_resolve_sources`` 已校验快照/外部候选副本，这里对最终
    交给 ``build_v12_robot`` 的模型文件（``sources["model"]`` 快照，或
    快照缺失时 profile 解析出的外部模型）再比对一次
    ``input_sha256["model"]``，保证“校验的文件 = 消费的文件”。

    Args:
        payload: v2 产物 payload。
        sources: :func:`_verify_and_resolve_sources` 返回的已校验来源。

    Returns:
        实际将加载的模型路径（无 profile 时为 None）。

    Raises:
        FileNotFoundError: 模型文件不存在。
        ValueError: 实际加载模型哈希与 ``input_sha256["model"]`` 不一致。
    """
    model_path = _resolved_model_path(payload, sources)
    if model_path is None:
        return None
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"回放模型文件缺失：{model_path}")
    expected = str(payload.get("input_sha256", {}).get("model", "") or "")
    if not expected:
        _LOGGER.warning(
            "产物缺少模型文件哈希（旧 artifact），跳过回放模型文件校验：%s",
            model_path,
        )
        return model_path
    actual = _sha256_file(model_path)
    if actual != expected:
        raise ValueError(
            "回放模型文件哈希不一致，拒绝回放："
            f"{model_path}（期望 {expected}，实际 {actual}）"
        )
    return model_path


#: 物化模型快照的临时目录前缀（系统临时目录为 ASCII，规避中文路径）
_MATERIALIZED_MODEL_PREFIX = "v12_replay_model_"


def _is_ascii_text(text: str) -> bool:
    """判断字符串能否用 ASCII 编码（Windows MuJoCo C 层的路径限制）。"""
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _loadable_asset_path(
    path: Path,
    staging_dir: Path,
    index: int,
    *,
    expected_sha256: str = "",
    asset_id: str = "",
) -> Path:
    """把资产路径映射为当前平台可被 MuJoCo 打开的绝对路径。

    Windows 上 ``mj_loadXML`` 无法打开含非 ASCII 字符的路径（实测仓库
    路径 ``F:/刀客塔/...`` 直接报 ``Error opening file``）。仓库内资产
    复用 :mod:`src.utils.mujoco_loader` 的 ASCII 副本（与正常加载模型时
    同一份）；仓库外的非 ASCII 资产复制进本次物化的临时目录。

    Args:
        path: 资产原路径（调用方已确认存在）。
        staging_dir: 本次物化的临时目录。
        index: 资产序号（避免同名覆盖）。
        expected_sha256: 记录的文件哈希；非空时校验最终可加载路径的字节
            （ASCII 副本可能早于本次校验建立，必须复核，防"复制前篡改"）。
        asset_id: 稳定逻辑 ID（仅用于错误信息定位）。

    Returns:
        可加载的绝对路径（ASCII 路径原样返回）。

    Raises:
        ValueError: 仓库资产无法映射到 ASCII 副本，或最终字节哈希不一致。
    """
    resolved = Path(path).resolve()

    def _verified(candidate: Path) -> Path:
        """复核候选路径的字节哈希（未记录哈希时直接返回）。"""
        if not expected_sha256:
            return candidate
        actual = _sha256_file(candidate)
        if actual != str(expected_sha256):
            raise ValueError(
                "装配依赖资产在复制前被改动（拒绝使用）："
                f"asset_id={asset_id or '<未记录>'} 文件={candidate} "
                f"期望={expected_sha256} 实际={actual}"
            )
        return candidate

    if sys.platform != "win32" or _is_ascii_text(str(resolved)):
        return _verified(resolved)
    try:
        relative = resolved.relative_to(_repo_root())
    except ValueError:
        relative = None
    if relative is not None:
        from src.utils import mujoco_loader  # 延迟导入，避免引入 mujoco 依赖

        mirror = Path(mujoco_loader._setup_temp_copy(_repo_root()))
        mirrored = mirror / relative
        if mirrored.is_file():
            return _verified(mirrored)
        raise ValueError(
            "模型快照引用的仓库资产无法映射到 ASCII 副本："
            f"{resolved}（副本 {mirrored}）"
        )
    destination = staging_dir / "external_assets" / f"{index:03d}_{resolved.name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved, destination)
    return _verified(destination)



def _model_origin_dir(payload: Mapping[str, Any], model_path: Path) -> Path:
    """返回模型 XML 原始位置目录（解析相对 mesh/texture 路径的基准）。

    优先读取快照 manifest 中 ``model`` 条目的 ``source``（产物写入时记录
    的原始绝对路径）；其次回退 payload config 解析出的外部模型父目录；
    最后回退待物化模型自身所在目录。
    """
    for parent in Path(model_path).resolve().parents:
        manifest_path = parent / SNAPSHOT_MANIFEST_NAME
        if not manifest_path.is_file():
            continue
        try:
            manifest = _read_json(manifest_path)
        except (OSError, ValueError):  # pragma: no cover - 破损 manifest
            break
        for item in manifest.get("files") or []:
            if not isinstance(item, Mapping) or item.get("role") != "model":
                continue
            source = item.get("source")
            if source:
                return Path(str(source)).parent
        break
    external = _external_source_for_role(payload, "model")
    if external is not None:
        external_path = Path(external)
        if not external_path.is_absolute():
            external_path = _repo_root() / external_path
        return external_path.parent
    return Path(model_path).resolve().parent


class _DependencyStager:
    """把依赖闭包物化进临时目录，并把 MJCF 内部引用改写为可加载绝对路径。

    只消费 :func:`resolve_verified_assets` 返回的映射（``asset_id -> 路径``），
    不再按"记录来源"二次挑选文件；每个 XML 片段复制后复核字节哈希。

    Attributes:
        staging_dir: 本次物化的临时目录。
        logical_base: 顶层模型 XML 的**原始**路径（解析 ``<include>`` 与
            ``meshdir`` 相对引用；可以是写入时记录、当前已不存在的路径）。
    """

    def __init__(
        self,
        *,
        entries: Sequence[Mapping[str, Any]],
        resolved: Mapping[str, Path],
        staging_dir: Path,
        logical_base: Path,
    ) -> None:
        self.staging_dir = Path(staging_dir)
        self.logical_base = Path(logical_base)
        self._entries = [dict(entry) for entry in entries]
        self._resolved = dict(resolved)
        self._by_source: dict[Path, dict[str, Any]] = {}
        self._by_reference: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for entry in self._entries:
            source = str(entry.get("source") or "")
            if source:
                self._by_source.setdefault(Path(source).resolve(), entry)
            reference = str(entry.get("reference") or "")
            if reference:
                self._by_reference.setdefault(
                    (str(entry.get("role")), reference), []
                ).append(entry)
        self._staged: dict[str, Path] = {}
        self._counter = 0
        self._model_entry: dict[str, Any] | None = None
        for entry in self._entries:
            if str(entry.get("role")) == DEPENDENCY_ROLE_MODEL:
                self._model_entry = entry
                break
        #: 最终全局 ``{kind: meshdir/texturedir}``（stage_model 时确定性推导）
        self._asset_dirs: dict[str, str] = {}
        #: 顶层模型目录（include 第一候选与全局目录的固定基准）
        self._top_dir: Path = Path(self.logical_base).parent

    # -- 对外入口 ---------------------------------------------------------
    def stage_model(self, text: str) -> str:
        """改写顶层模型文本（include 递归物化 + mesh/texture 绝对化）。

        资产目录必须在改写**前**由原始文本与依赖清单确定性推导：``<compiler>``
        是全局设置（include 片段里的声明同样生效、后者覆盖前者），改写后
        文本里的 include 已换成临时副本，不能再据此重建全局目录。

        Args:
            text: 顶层模型 XML 原始文本。

        Returns:
            改写后的 MJCF 文本（所有引用均指向临时目录内已验证副本）。
        """
        self._asset_dirs = self._global_asset_dirs(text)
        return self._rewrite(text, logical_source=self.logical_base)

    def stage_sibling_fragments(self) -> list[Path]:
        """把兄弟臂片段写到临时模型**同目录**，供工厂优先命中（F08）。

        Returns:
            已物化的片段路径列表（按相对路径排序，顺序稳定）。
        """
        staged: list[Path] = []
        for entry in self._entries:
            if str(entry.get("role")) != DEPENDENCY_ROLE_ARM_FRAGMENT:
                continue
            staged.append(self._stage_xml_entry(entry))
        return staged

    # -- 全局资产目录 ------------------------------------------------------
    def _global_asset_dirs(self, model_text: str) -> dict[str, str]:
        """推导最终全局 ``{kind: meshdir/texturedir}``。

        优先使用模型条目记录的 ``asset_dirs``（写入时按 MuJoCo 语义算出的
        最终值）；旧产物没有该字段时，回退为按文档顺序读取顶层文本与 include
        条目（从已验证快照副本读取）里的 ``<compiler>`` 重建。

        Args:
            model_text: 顶层模型 XML 原始文本。

        Returns:
            只含已声明种类的目录前缀表（空表表示无目录前缀）。
        """
        recorded = (self._model_entry or {}).get("asset_dirs")
        if isinstance(recorded, Mapping) and recorded:
            return {
                str(kind): str(value)
                for kind, value in recorded.items()
                if str(kind) in ASSET_XML_TAGS
            }
        dirs: dict[str, str] = {}
        self._collect_compiler_dirs(
            model_text,
            base=self._top_dir,
            dirs=dirs,
            depth=0,
        )
        return dirs

    def _include_candidates(self, raw: str, *, base: Path) -> tuple[Path, Path]:
        """给出 include 的两个候选路径（顶层模型目录优先、声明目录兜底）。

        Args:
            raw: ``file=`` 原始字符串。
            base: 声明该 include 的 XML 文件所在目录（第二候选基准）。

        Returns:
            ``(顶层候选, 声明目录候选)``。
        """
        return (
            _resolve_xml_reference(self._top_dir, raw),
            _resolve_xml_reference(base, raw),
        )

    def _include_entry(self, raw: str, *, base: Path) -> dict[str, Any] | None:
        """按候选顺序定位依赖清单里已记录的 include 条目。

        顶层候选优先（新产物的记录口径）；未记录时回退声明目录候选
        （旧产物按声明目录记录，兼容回放）。绝不猜测未记录的文件。

        Args:
            raw: ``file=`` 原始字符串。
            base: 声明该 include 的 XML 文件所在目录。

        Returns:
            命中的 include 依赖条目；两个候选都未记录时返回 ``None``。
        """
        checked: set[Path] = set()
        for target in self._include_candidates(raw, base=base):
            if target in checked:
                continue
            checked.add(target)
            entry = self._by_source.get(target)
            if entry is not None and str(entry.get("role")) == DEPENDENCY_ROLE_INCLUDE:
                return entry
        return None

    def _collect_compiler_dirs(
        self,
        text: str,
        *,
        base: Path,
        dirs: dict[str, str],
        depth: int,
    ) -> None:
        """按文档顺序收集一段 MJCF 及其 include 链的 compiler 目录声明。

        语义与 MuJoCo 一致：注释整体忽略；全局生效、后者覆盖前者；
        ``assetdir`` 同时设置 meshdir/texturedir，同一标签内的显式属性覆盖
        它（与书写顺序无关）；只合并**实际声明**的属性（未声明不覆盖上层
        取值）。include 先按顶层模型目录解析、再回退声明目录，目标从依赖
        清单条目读取已验证快照文本（旧产物回放时原始文件可能已不存在）。

        Args:
            text: 当前片段文本。
            base: 该片段的原始文件所在目录（include 第二候选基准）。
            dirs: 输出目录表（原地更新）。
            depth: include 嵌套深度（超过上限直接停止，不猜测）。
        """
        if depth > _MAX_INCLUDE_DEPTH:
            return
        for match in _COMPILER_OR_INCLUDE_PATTERN.finditer(_masked_xml(text)):
            tag = str(match.group("tag")).lower()
            attrs = match.group("attrs")
            if tag == "compiler":
                assetdir = _tag_attribute(attrs, "assetdir")
                if assetdir is not None:
                    for kind in _ASSET_DIR_ATTRS:
                        dirs[kind] = assetdir.strip()
                for kind, attr_name in _ASSET_DIR_ATTRS.items():
                    value = _tag_attribute(attrs, attr_name)
                    if value is not None:
                        dirs[kind] = value.strip()
                continue
            raw = (_tag_attribute(attrs, "file") or "").strip()
            if not raw:
                continue
            entry = self._include_entry(raw, base=base)
            if entry is None:
                continue
            verified = self._resolved.get(str(entry.get("asset_id")))
            if verified is None or not verified.is_file():
                continue
            try:
                child_text = verified.read_text(encoding="utf-8", errors="replace")
            except OSError:  # pragma: no cover - 已验证副本理应可读
                continue
            child_base = Path(str(entry.get("source") or self._top_dir / raw)).parent
            self._collect_compiler_dirs(
                child_text,
                base=child_base,
                dirs=dirs,
                depth=depth + 1,
            )

    def _asset_candidate_paths(
        self,
        kind: str,
        raw: str,
        declaring_dir: Path,
    ) -> list[Path]:
        """按 MuJoCo 语义给出资产候选路径（顺序即查找顺序）。

        候选 1 = ``<顶层模型目录>/<全局 meshdir|texturedir>/<file>``（目录
        前缀为绝对路径时直接生效）；候选 2 = ``<声明该引用的 XML 目录>/
        <file>``（不再拼目录前缀）。绝对引用只有一个候选。

        Args:
            kind: 资产种类（``mesh`` 或 ``texture``）。
            raw: ``file=`` 原始字符串。
            declaring_dir: 声明该引用的 XML 文件所在目录。

        Returns:
            去重后的候选路径列表。
        """
        reference = Path(raw.replace("\\", "/"))
        if reference.is_absolute():
            return [reference.resolve()]
        top_dir = self._top_dir
        dir_prefix = self._asset_dirs.get(kind, "")
        base = (top_dir / dir_prefix).resolve() if dir_prefix else top_dir
        candidates = [
            (base / reference).resolve(),
            (Path(declaring_dir) / reference).resolve(),
        ]
        unique: list[Path] = []
        for candidate in candidates:
            if candidate not in unique:
                unique.append(candidate)
        return unique

    # -- 内部实现 ---------------------------------------------------------
    def _next_index(self) -> int:
        """递增资产序号（避免同名资产在临时目录内互相覆盖）。"""
        value = self._counter
        self._counter += 1
        return value

    def _stage_xml_entry(self, entry: Mapping[str, Any]) -> Path:
        """复制单个 XML 依赖到临时目录（已复制过则直接复用）。"""
        asset_id = str(entry["asset_id"])
        if asset_id in self._staged:
            return self._staged[asset_id]
        source = self._resolved.get(asset_id)
        if source is None:  # pragma: no cover - resolve_verified_assets 已抛错
            raise ValueError(f"装配依赖资产未通过哈希校验：asset_id={asset_id}")
        role = str(entry["role"])
        name = Path(str(entry.get("relative_path") or entry.get("source"))).name
        if role == DEPENDENCY_ROLE_INCLUDE:
            destination = self.staging_dir / "includes" / f"{self._next_index():03d}_{name}"
        else:
            # 兄弟片段/模型沿用原名：工厂 _arm_fragment_path 按名在同目录查找
            destination = self.staging_dir / name
        _copy_verified(source, destination, str(entry["sha256"]), asset_id)
        self._staged[asset_id] = destination
        if role in DEPENDENCY_XML_ROLES:
            try:
                text = destination.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"装配依赖 XML 片段不是 UTF-8，无法物化：asset_id={asset_id}"
                ) from exc
            logical_source = Path(str(entry.get("source") or destination))
            rewritten = self._rewrite(text, logical_source=logical_source)
            destination.write_text(rewritten, encoding="utf-8", newline="\n")
        return destination

    def _rewrite(self, text: str, *, logical_source: Path, depth: int = 0) -> str:
        """改写一段 MJCF 文本中的 ``<include>`` 与 ``<mesh>/<texture>``。

        先用 :func:`_masked_xml` 得到与原文等长的注释掩码文本，在掩码上
        ``finditer`` 定位标签、按**原文 span** 只替换 ``file=`` 属性——注释
        里的标签不参与改写，注释内容原样保留（R03）。include 先按顶层模型
        目录解析、再回退声明目录（R04）；资产路径在 ``stage_model`` 阶段已由
        最终全局目录表确定，这里不再重建（include 已被替换为临时副本，其
        内部 compiler 声明不可见）。

        Args:
            text: 原始 MJCF 文本。
            logical_source: 该文本的**原始**文件路径（相对引用基准）。
            depth: include 嵌套深度（防御环状引用）。

        Raises:
            ValueError: include 未被依赖清单记录、深度超限，或资产未记录。
        """
        if depth > _MAX_INCLUDE_DEPTH:
            raise ValueError(
                "模型快照的 <include> 嵌套超过上限"
                f"（{_MAX_INCLUDE_DEPTH} 层），拒绝静默错误解析：{logical_source}"
            )
        base = Path(logical_source).parent

        def file_reference(tag: str) -> str:
            """取出标签 ``file=`` 的原始取值（无则该空串）。"""
            match = _ASSET_FILE_ATTR_PATTERN.search(tag)
            if match is None:
                return ""
            value = match.group(1)
            if value is None:
                value = match.group(2) or ""
            return value.strip()

        # 第一遍：include（先在掩码文本定位，再按原文替换 file 属性）
        pieces: list[str] = []
        last = 0
        for match in _INCLUDE_TAG_PATTERN.finditer(_masked_xml(text)):
            tag = text[match.start() : match.end()]
            pieces.append(text[last : match.start()])
            last = match.end()
            raw = file_reference(tag)
            if not raw:
                pieces.append(tag)
                continue
            entry = self._include_entry(raw, base=base)
            if entry is None:
                raise ValueError(
                    "模型快照包含未被依赖清单记录的 <include>，拒绝静默错误解析："
                    f"{raw}（引用自 {logical_source}）"
                )
            staged = self._stage_xml_entry(entry)
            pieces.append(_replace_file_attribute(tag, staged.as_posix()))
        pieces.append(text[last:])
        rewritten = "".join(pieces)

        # 第二遍：mesh/texture（掩码基于第一遍改写后的文本，位置仍对应原文）
        pieces = []
        last = 0
        for match in _ASSET_TAG_PATTERN.finditer(_masked_xml(rewritten)):
            tag = rewritten[match.start() : match.end()]
            pieces.append(rewritten[last : match.start()])
            last = match.end()
            raw = file_reference(tag)
            if not raw:
                pieces.append(tag)
                continue
            kind = str(match.group(1)).lower()
            entry = self._asset_entry(kind, raw, base)
            asset_id = str(entry["asset_id"])
            resolved = self._resolved.get(asset_id)
            if resolved is None:  # pragma: no cover - 上游已抛错
                raise ValueError(f"装配依赖资产未通过哈希校验：asset_id={asset_id}")
            staged = _loadable_asset_path(
                resolved,
                self.staging_dir,
                self._next_index(),
                expected_sha256=str(entry["sha256"]),
                asset_id=asset_id,
            )
            pieces.append(_replace_file_attribute(tag, staged.as_posix()))
        pieces.append(rewritten[last:])
        return "".join(pieces)

    def _asset_entry(
        self,
        kind: str,
        raw: str,
        declaring_dir: Path,
    ) -> dict[str, Any]:
        """按引用字符串（并辅以 MuJoCo 真实路径）定位依赖条目。

        唯一引用直接命中；同一 ``file=`` 出现多次时，按候选路径（顶层目录 +
        全局目录 → 声明目录）与条目 ``source`` 精确匹配——绝不按文件名猜测。

        Args:
            kind: 资产种类（``mesh`` 或 ``texture``）。
            raw: ``file=`` 原始字符串。
            declaring_dir: 声明该引用的 XML 文件所在目录。

        Returns:
            命中的依赖条目。

        Raises:
            ValueError: 条目未被清单记录（不得猜测/回落同名文件）。
        """
        candidates = self._by_reference.get((kind, raw), [])
        if len(candidates) == 1:
            return candidates[0]
        target_paths = self._asset_candidate_paths(kind, raw, declaring_dir)
        if len(candidates) > 1:
            for candidate in candidates:
                source = Path(str(candidate.get("source") or "")).resolve()
                if any(source == target for target in target_paths):
                    return candidate
        for target in target_paths:
            entry = self._by_source.get(target)
            if entry is not None and str(entry.get("role")) == kind:
                return entry
        described = " / ".join(str(target) for target in target_paths)
        raise ValueError(
            "物化模型快照引用的资产未记录在依赖清单中（拒绝加载未校验文件）："
            f"{raw} → {described}（引用自 {declaring_dir}）"
        )


def _materialize_profile(
    payload: Mapping[str, Any],
    sources: Mapping[str, Path],
    staging_dir: Path,
    model_xml: Path,
) -> Path:
    """以原 robot profile 文本为模板，仅改 ``model_path`` 指向物化 XML。"""
    profile_path = sources.get("robot_profile")
    if profile_path is None:
        profile_path = _external_source_for_role(payload, "robot_profile")
    if profile_path is None:
        raise ValueError("无法物化模型快照：产物缺少 robot profile 来源")
    try:
        profile_text = Path(profile_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"读取 robot profile 失败，无法物化模型：{profile_path}（{exc}）"
        ) from exc
    materialized, count = re.subn(
        r"(?m)^(\s*model_path\s*:\s*).*$",
        lambda match: f"{match.group(1)}'{model_xml.as_posix()}'",
        profile_text,
        count=1,
    )
    if count == 0:
        raise ValueError(
            f"robot profile 缺少 model_path 字段，无法物化模型：{profile_path}"
        )
    temp_profile = staging_dir / "profile.yaml"
    temp_profile.write_text(materialized, encoding="utf-8", newline="\n")
    return temp_profile


def _materialize_snapshot_model(
    payload: Mapping[str, Any],
    sources: Mapping[str, Path],
    model_path: Path,
    *,
    output_dir: Path | None = None,
) -> tuple[Path, Callable[[], None]]:
    """把校验通过的模型快照物化为可加载的临时 XML + 临时 profile。

    快照里 mesh/texture 的路径仍相对原仓库（如 ``../../assets/...``），
    把快照 XML 直接交给 MuJoCo 会解析到快照目录内不存在的资产。这里读入
    **已校验**的快照字节（再次用 ``input_sha256["model"]`` 比对，保证
    "校验的文件 = 消费的文件"），用依赖闭包清单解析出的**同一份**已验证
    路径映射改写每个 ``<mesh>`` / ``<texture>`` 的 ``file``（F02：不再按
    记录来源重新挑选旧文件）；``<include>`` 递归物化；兄弟臂片段写到临时
    模型同目录，使工厂 ``_arm_fragment_path`` 优先命中快照内容（F08）。
    Windows 非 ASCII 路径映射到 ASCII 副本。

    Args:
        payload: v2 产物 payload（提供模型哈希与依赖清单）。
        sources: :func:`_verify_and_resolve_sources` 的校验结果。
        model_path: 将被加载的模型文件（快照或外部回退）。
        output_dir: 产物目录（用于定位快照副本；缺省时从 ``model_path``
            向上查找 ``snapshot/``，仍找不到则只用仓库/记录来源候选）。

    Returns:
        ``(temp_profile_path, cleanup)``；``cleanup`` 为无参可调用对象，
        删除本次物化的临时目录（构建成功/失败都必须调用）。

    Raises:
        ValueError: 模型字节哈希与记录不一致、XML 非 UTF-8、依赖缺失/
            哈希不一致、``<include>`` 未被清单记录，或 profile 无
            ``model_path``。
    """
    resolved_model = Path(model_path).resolve()
    try:
        raw = resolved_model.read_bytes()
    except OSError as exc:
        raise ValueError(f"读取待物化模型失败：{resolved_model}（{exc}）") from exc
    expected = str(payload.get("input_sha256", {}).get("model", "") or "")
    actual = hashlib.sha256(raw).hexdigest()
    if expected and actual != expected:
        raise ValueError(
            "回放模型文件哈希不一致，拒绝物化："
            f"{resolved_model}（期望 {expected}，实际 {actual}）"
        )
    try:
        model_text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"模型快照不是 UTF-8 XML，无法物化：{resolved_model}"
        ) from exc

    entries, legacy = _dependency_entries(payload)
    if not entries:
        raise ValueError(
            "产物缺少装配依赖条目（既无 dependencies 也无 assets.files），"
            f"无法证明装配闭包，拒绝物化：{resolved_model}"
        )
    snapshot_dir = _snapshot_root_for(resolved_model)
    if snapshot_dir is None:
        snapshot_dir = _snapshot_root_for(sources.get("model"))
    if snapshot_dir is None and output_dir is not None:
        candidate_dir = Path(output_dir) / SNAPSHOT_DIR_NAME
        if (candidate_dir / SNAPSHOT_MANIFEST_NAME).is_file():
            snapshot_dir = candidate_dir
    view: dict[str, Any] = {
        "dependencies": {"version": DEPENDENCY_MANIFEST_VERSION, "files": entries},
        "snapshot_dir": str(snapshot_dir) if snapshot_dir is not None else "",
    }
    # F02：校验与消费共用这一份映射；任何缺失/哈希不符都会带 asset_id 抛错
    resolved = resolve_verified_assets(view, _repo_root())

    model_entry = next(
        (
            entry
            for entry in entries
            if str(entry.get("role")) == DEPENDENCY_ROLE_MODEL
            and str(entry.get("sha256")) == (expected or actual)
        ),
        None,
    )
    if model_entry is not None and model_entry.get("source"):
        logical_base = Path(str(model_entry["source"]))
    else:
        logical_base = _model_origin_dir(payload, resolved_model) / resolved_model.name
        if not legacy:
            _LOGGER.warning(
                "依赖清单的模型条目与产物模型哈希不一致（快照模型可能被改动），"
                "按写入时记录的模型位置解析相对引用：%s",
                resolved_model,
            )
    if legacy:
        _LOGGER.warning(
            "旧 artifact 缺少装配依赖清单（legacy 分支）：仅校验角色文件与网格哈希，"
            "不保证装配闭包（兄弟片段可能回落到仓库同名文件）：%s",
            output_dir if output_dir is not None else resolved_model,
        )

    staging_dir = Path(tempfile.mkdtemp(prefix=_MATERIALIZED_MODEL_PREFIX))

    def cleanup() -> None:
        """删除本次物化的临时目录（幂等）。"""
        shutil.rmtree(staging_dir, ignore_errors=True)

    try:
        stager = _DependencyStager(
            entries=entries,
            resolved=resolved,
            staging_dir=staging_dir,
            logical_base=logical_base,
        )
        materialized_text = stager.stage_model(model_text)
        # F08：兄弟片段与模型同目录，工厂优先命中快照内容而非仓库同名文件
        stager.stage_sibling_fragments()
        # 沿用原模型文件名：R1D 组装按 model_path 名/内容推断侧别，避免
        # 无谓改变原有语义。
        xml_path = staging_dir / resolved_model.name
        xml_path.write_text(materialized_text, encoding="utf-8", newline="\n")
        profile_path = _materialize_profile(payload, sources, staging_dir, xml_path)
    except BaseException:
        cleanup()
        raise
    return profile_path, cleanup


def _prepare_replay_config(
    payload: Mapping[str, Any],
    sources: Mapping[str, Path],
    config: Any,
    model_path: Path | None,
    *,
    output_dir: Path | None = None,
) -> tuple[Any, Callable[[], None] | None]:
    """按实际加载模型修正回放配置（快照模型需物化为临时 profile）。

    只要 ``sources["model"]`` 存在（选中了模型快照）就**无条件物化**：
    即使快照 profile 里的相对 ``model_path`` 恰好解析回快照内的模型文件
    （``snapshot/files/src/robot/...``），快照目录也没有复制 mesh/texture
    资产，直接加载会因资产缺失而失败。只有外部回退（``sources`` 无
    ``model``）才保持原配置。

    Returns:
        ``(config, cleanup)``；未物化时 ``cleanup`` 为 None。
    """
    if model_path is None or "model" not in sources:
        return config, None
    temp_profile, cleanup = _materialize_snapshot_model(
        payload, sources, Path(model_path), output_dir=output_dir
    )
    from dataclasses import replace

    return replace(config, robot_profile_path=temp_profile), cleanup


def _build_replay_bundle_with_snapshot(
    artifacts: EpisodeArtifacts,
    sources: Mapping[str, Path],
    config: Any,
    model_path: Path | None,
) -> V12RobotBundle:
    """物化（必要时）并构建回放 bundle，临时目录生命周期绑定到 bundle。"""
    replay_config, cleanup = _prepare_replay_config(
        artifacts.payload,
        sources,
        config,
        model_path,
        output_dir=artifacts.directory,
    )
    from src.sim.v12_robot_adapter import build_v12_robot  # 延迟导入

    try:
        # F01：回放与真实执行消费同一份已保存 runtime_overrides（limits/
        # 碰撞/来球开关），而不是重新按默认值构建 bundle。
        from src.sim.episode_runtime import (  # 延迟导入避免循环依赖
            apply_runtime_limits,
            resolve_runtime_options,
        )

        replay_config = resolve_runtime_options(replay_config, {})
        bundle = build_v12_robot(replay_config)
        apply_runtime_limits(replay_config, bundle)
    except BaseException:
        if cleanup is not None:
            cleanup()
        raise
    if cleanup is not None:
        # planning_env_factory 在首次规划时才按 profile.model_path 建规划
        # 环境，临时目录必须活到 bundle 生命周期结束（弱引用终结器兜底）。
        weakref.finalize(bundle, cleanup)
    return bundle


def build_replay_bundle(output_dir: Path) -> V12RobotBundle:
    """读取产物并构建回放用机器人 bundle（不打开 viewer）。

    快照模型优先：``_resolved_model_path`` 选中模型快照时，先把快照 XML
    物化为临时 XML/profile 再构建（快照目录不含 mesh 资产），保证加载的
    几何来自快照而不是被改动/删除的外部模型（评审 R11）。只有快照模型
    缺失、回退到外部模型时才直接使用原配置。

    Raises:
        ValueError: 快照/外部来源哈希不一致、实际加载模型被篡改，或
            快照模型物化失败。
    """
    artifacts = load_episode_artifacts(output_dir)
    sources = _verify_and_resolve_sources(artifacts.payload, Path(output_dir))
    config = _replay_config(artifacts.payload, sources)
    model_path = _verify_loaded_model(artifacts.payload, sources)
    return _build_replay_bundle_with_snapshot(
        artifacts, sources, config, model_path
    )


@dataclass(frozen=True)
class EpisodeReplayResult:
    """回放结果（按名重建状态；不修改产物文件）。"""

    payload: dict[str, Any]
    ball_positions: np.ndarray
    racket_positions: np.ndarray
    speed: float
    viewer_opened: bool
    directory: Path


def _replay_frames(
    execution: Any,
    bundle: V12RobotBundle,
    artifacts: EpisodeArtifacts,
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """逐帧按名写入状态并 FK，返回 (ball_positions, racket_positions)。"""
    from src.sim.v12_robot_adapter import ReplayState  # 延迟导入

    n = len(joint_names)
    X = artifacts.X
    ball_pos = artifacts.ball_pos
    ball_vel = artifacts.ball_vel
    frames = int(X.shape[0])
    racket_positions = np.zeros((frames, 3), dtype=float)
    for frame in range(frames):
        state = ReplayState(
            joint_positions={
                name: float(X[frame, index]) for index, name in enumerate(joint_names)
            },
            joint_velocities={
                name: float(X[frame, n + index])
                for index, name in enumerate(joint_names)
            },
            ball_position=np.asarray(ball_pos[frame], dtype=float),
            ball_velocity=np.asarray(ball_vel[frame], dtype=float),
        )
        bundle.set_replay_state(state)
        execution.update_kinematics()
        racket_positions[frame] = np.asarray(execution.get_ee_pos(), dtype=float)
    return np.asarray(ball_pos, dtype=float), racket_positions


def _visualize_replay(
    execution: Any,
    bundle: V12RobotBundle,
    artifacts: EpisodeArtifacts,
    joint_names: list[str],
    speed: float,
    dt: float,
) -> None:
    """用 passive viewer 按 ``speed`` 倍速循环播放已保存的轨迹。"""
    import mujoco
    import mujoco.viewer

    X = artifacts.X
    frames = int(X.shape[0])
    if frames <= 0:
        return
    playback = float(speed) if speed and speed > 0 else 1.0
    with mujoco.viewer.launch_passive(
        execution.model, execution.data
    ) as viewer:
        viewer.cam.distance = 2.8
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth = 135.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
        start = time.perf_counter()
        frame = 0
        while viewer.is_running():
            elapsed = time.perf_counter() - start
            idx = int(elapsed * playback / max(float(dt), 1e-9))
            if idx >= frames:
                start = time.perf_counter()
                idx = 0
            if idx != frame:
                frame = idx
                from src.sim.v12_robot_adapter import ReplayState

                n = len(joint_names)
                state = ReplayState(
                    joint_positions={
                        name: float(X[frame, i])
                        for i, name in enumerate(joint_names)
                    },
                    joint_velocities={
                        name: float(X[frame, n + i])
                        for i, name in enumerate(joint_names)
                    },
                    ball_position=np.asarray(artifacts.ball_pos[frame], dtype=float),
                    ball_velocity=np.asarray(artifacts.ball_vel[frame], dtype=float),
                )
                bundle.set_replay_state(state)
                execution.update_kinematics()
                mujoco.mj_forward(execution.model, execution.data)
                viewer.sync()
            time.sleep(min(float(dt) / playback, 0.02))


def replay_episode_artifacts(
    output_dir: Path,
    *,
    speed: float = 1.0,
    viewer: bool = True,
    bundle: V12RobotBundle | None = None,
) -> EpisodeReplayResult:
    """只读回放一个已保存的 v2 回合产物。

    回放按关节名写入状态（不依赖 qpos 布局），默认 1.0 倍速；慢放必须
    显式传入 ``speed < 1``。回放**不会**改写 JSON/NPZ、也不会重新判定
    命中或合成反弹。

    Args:
        output_dir: 产物目录。
        speed: 播放倍速（默认 1.0；仅 viewer=True 时生效）。
        viewer: 是否打开被动查看器（False 时纯计算）。
        bundle: 可选的已构建 bundle（避免重复建模）。

    Returns:
        :class:`EpisodeReplayResult`（含 payload、球/拍位置历史与倍速）。

    Raises:
        ValueError: 快照/外部来源哈希不一致，或缺 ``input_sha256`` 校验项。
    """
    directory = Path(output_dir)
    artifacts = load_episode_artifacts(directory)
    payload = artifacts.payload
    sources = _verify_and_resolve_sources(payload, directory)
    config = _replay_config(payload, sources)
    model_path = _verify_loaded_model(payload, sources)
    if bundle is None:
        bundle = _build_replay_bundle_with_snapshot(
            artifacts, sources, config, model_path
        )
    execution = bundle.execution
    joint_names = [str(name) for name in payload["joint_names"]]
    ball_positions, racket_positions = _replay_frames(
        execution, bundle, artifacts, joint_names
    )
    viewer_opened = False
    if viewer:
        try:
            _visualize_replay(
                execution,
                bundle,
                artifacts,
                joint_names,
                speed=speed,
                dt=float(payload.get("dt", 0.005)),
            )
            viewer_opened = True
        except Exception as exc:  # pragma: no cover - 无显示环境时降级
            _LOGGER.warning("回放 viewer 不可用（降级为纯计算）：%s", exc)
    return EpisodeReplayResult(
        payload=payload,
        ball_positions=ball_positions,
        racket_positions=racket_positions,
        speed=float(speed),
        viewer_opened=viewer_opened,
        directory=directory,
    )
