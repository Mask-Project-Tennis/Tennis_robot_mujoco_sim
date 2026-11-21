# ruff: noqa: I001, PLC0414, UP035
# D1 契约 §6 要求显式 `import X as X` 重导出；生成器保留原文件 typing
# 导入与分组风格。BLE001/TRY004/SIM102 为拆分前旧代码既有告警，
# 行为不变，仅在此保留（记入报告）。

"""资产/XML 依赖闭包扫描、依赖清单与装配物化暂存（D1）。

负责 MJCF 资产扫描、依赖条目归一化/派生、逐文件哈希校验与复制，
以及 :class:`_DependencyStager`（把依赖闭包物化进临时目录并改写 XML
内部引用）。依赖图位置：``types ← serialization ← dependencies``。

``_DependencyStager`` 逐阶段迁移：``stage_model``（收集 include 链）→
``stage_sibling_fragments``（兄弟片段）→ ``_rewrite``（改写资产引用）→
``_asset_entry``（清单条目），清理由 :mod:`materialization` 的
``cleanup`` 在异常路径统一执行。算法未重写。
"""

from __future__ import annotations

import logging
import base64
import re
import shutil
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.sim.artifacts.types import (
    ASSET_XML_TAGS as ASSET_XML_TAGS,
    DEPENDENCY_ASSET_ROLES as DEPENDENCY_ASSET_ROLES,
    DEPENDENCY_ROLE_ARM_FRAGMENT as DEPENDENCY_ROLE_ARM_FRAGMENT,
    DEPENDENCY_ROLE_INCLUDE as DEPENDENCY_ROLE_INCLUDE,
    DEPENDENCY_ROLE_MESH as DEPENDENCY_ROLE_MESH,
    DEPENDENCY_ROLE_MODEL as DEPENDENCY_ROLE_MODEL,
    DEPENDENCY_ROLE_TEXTURE as DEPENDENCY_ROLE_TEXTURE,
    DEPENDENCY_XML_ROLES as DEPENDENCY_XML_ROLES,
    REQUIRE_RESULT_DEPENDENCY_MANIFEST as REQUIRE_RESULT_DEPENDENCY_MANIFEST,
    SNAPSHOT_DIR_NAME as SNAPSHOT_DIR_NAME,
    SNAPSHOT_FILES_DIR_NAME as SNAPSHOT_FILES_DIR_NAME,
    SNAPSHOT_MANIFEST_NAME as SNAPSHOT_MANIFEST_NAME,
    _ASSET_DIR_ATTRS as _ASSET_DIR_ATTRS,
    _ASSET_FILE_ATTR_PATTERN as _ASSET_FILE_ATTR_PATTERN,
    _ASSET_TAG_PATTERN as _ASSET_TAG_PATTERN,
    _COMPILER_OR_INCLUDE_PATTERN as _COMPILER_OR_INCLUDE_PATTERN,
    _DEPENDENCY_COPY_DIRNAME as _DEPENDENCY_COPY_DIRNAME,
    _INCLUDE_TAG_PATTERN as _INCLUDE_TAG_PATTERN,
    _MAX_INCLUDE_DEPTH as _MAX_INCLUDE_DEPTH,
)
from src.sim.artifacts.serialization import (
    _assets_digest as _assets_digest,
    _repo_root as _repo_root,
    _role_paths as _role_paths,
    _sha256_file as _sha256_file,
    config_from_dict as config_from_dict,
)

_LOGGER = logging.getLogger(__name__)

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
    from src.robot.model_assets.references import mask_xml_comments

    return mask_xml_comments(text)
def _resolve_xml_reference(base: Path, raw: str) -> Path:
    """复用模型工厂的引用解析语义（绝对路径直用、相对路径按 ``base``）。"""
    from src.robot.model_assets.references import _resolve_reference_path

    return _resolve_reference_path(base, "", raw)
def _model_asset_entries(model_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """解析模型（含嵌套 include）的 ``<mesh>`` / ``<texture>`` 引用并算资产哈希。

    路径解析统一交给
    :func:`src.robot.model_assets.dependencies.collect_model_file_dependencies`（两阶段、
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
    # 延迟 import：model_assets 不加载 mujoco；本模块在纯产物校验/回放路径上
    # 不应被迫加载仿真依赖
    from src.robot.model_assets.dependencies import collect_model_file_dependencies
    from src.robot.model_assets.references import (
        DEPENDENCY_ROLE_MESH,
        DEPENDENCY_ROLE_TEXTURE,
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
    :func:`src.robot.model_building.loading._arm_fragment_path`，与真实装配同源，
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
    from src.robot.model_building.loading import _arm_fragment_path
    from src.robot.model_building.template import _model_side
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
    from src.robot.model_assets.dependencies import (  # 延迟导入：避免无 mujoco 场景的依赖
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
