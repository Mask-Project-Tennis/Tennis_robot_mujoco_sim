# ruff: noqa: I001, PLC0414, UP035, SIM102
# D1 契约 §6 要求显式 `import X as X` 重导出；生成器保留原文件 typing
# 导入与分组风格。BLE001/TRY004/SIM102 为拆分前旧代码既有告警，
# 行为不变，仅在此保留（记入报告）。

"""产物读取与来源/哈希校验（D1）。

负责 ``load_episode_artifacts``、快照 manifest 解析、模型资产校验、
来源优先级解析（结果内 manifest > 当前 profile/配置 > snapshot）以及
回放前实际加载模型的哈希复核。依赖图位置：``types ← serialization ←
dependencies ← validation``。

``_resolved_model_path`` 语义上属于先解析来源再交给物化，定义在本模块；
:mod:`materialization` 以 ``as`` 重导出同一对象（旧 facade 路径兼容）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.sim.artifacts.types import (
    ARTIFACT_JSON_NAME as ARTIFACT_JSON_NAME,
    ARTIFACT_NPZ_NAME as ARTIFACT_NPZ_NAME,
    ASSET_XML_TAGS as ASSET_XML_TAGS,
    EPISODE_SCHEMA as EPISODE_SCHEMA,
    EpisodeArtifacts as EpisodeArtifacts,
    INPUT_SHA_ROLES as INPUT_SHA_ROLES,
    SNAPSHOT_DIR_NAME as SNAPSHOT_DIR_NAME,
    SNAPSHOT_MANIFEST_NAME as SNAPSHOT_MANIFEST_NAME,
)
from src.sim.artifacts.serialization import (
    _assets_digest as _assets_digest,
    _model_path_for_profile as _model_path_for_profile,
    _read_json as _read_json,
    _repo_root as _repo_root,
    _role_file_digest as _role_file_digest,
    _role_paths as _role_paths,
    _sha256_file as _sha256_file,
    config_from_dict as config_from_dict,
)
from src.sim.artifacts.dependencies import (
    _collect_verified_assets as _collect_verified_assets,
    _dependency_section as _dependency_section,
    _is_ascii_text as _is_ascii_text,
    _loadable_asset_path as _loadable_asset_path,
)

_LOGGER = logging.getLogger(__name__)

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
                actual = _role_file_digest(candidate_path, role)
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
