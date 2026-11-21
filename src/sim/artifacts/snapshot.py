# ruff: noqa: I001, PLC0414, UP035
# D1 契约 §6 要求显式 `import X as X` 重导出；生成器保留原文件 typing
# 导入与分组风格。BLE001/TRY004/SIM102 为拆分前旧代码既有告警，
# 行为不变，仅在此保留（记入报告）。

"""产物写入与快照元数据（D1）。

组合 :mod:`serialization`（payload/数组）与 :mod:`dependencies`（依赖
闭包）落盘 ``v12_episode.json`` / ``v12_episode.npz`` / ``snapshot/``。
依赖图位置：``types ← serialization ← dependencies ← snapshot``。
"""

from __future__ import annotations

import logging
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from src.sim.episode.types import V12EpisodeResult

from src.sim.artifacts.types import (
    ARTIFACT_JSON_NAME as ARTIFACT_JSON_NAME,
    ARTIFACT_NPZ_NAME as ARTIFACT_NPZ_NAME,
    DEPENDENCY_MANIFEST_VERSION as DEPENDENCY_MANIFEST_VERSION,
    EPISODE_SCHEMA as EPISODE_SCHEMA,
    SNAPSHOT_CONFIG_NAME as SNAPSHOT_CONFIG_NAME,
    SNAPSHOT_DIR_NAME as SNAPSHOT_DIR_NAME,
    SNAPSHOT_FILES_DIR_NAME as SNAPSHOT_FILES_DIR_NAME,
    SNAPSHOT_MANIFEST_NAME as SNAPSHOT_MANIFEST_NAME,
    SNAPSHOT_ROLES as SNAPSHOT_ROLES,
    SNAPSHOT_SCENE_NAME as SNAPSHOT_SCENE_NAME,
)
from src.sim.artifacts.serialization import (
    _arrays_from_result as _arrays_from_result,
    _is_eol_normalized_role as _is_eol_normalized_role,
    _read_normalized_text as _read_normalized_text,
    _role_paths as _role_paths,
    _sha256_bytes as _sha256_bytes,
    _sha256_file as _sha256_file,
    build_episode_payload as build_episode_payload,
)
from src.sim.artifacts.dependencies import (
    _build_dependency_entries as _build_dependency_entries,
    _snapshot_relative_path as _snapshot_relative_path,
    _write_dependency_copies as _write_dependency_copies,
)

_LOGGER = logging.getLogger(__name__)

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
        if _is_eol_normalized_role(role):
            # 文本配置按 LF 归一化写快照：快照字节与 input_sha256 同源，
            # 换机/换行尾设置后仍能逐字节复现。
            dest.write_bytes(_read_normalized_text(source))
            digest = _sha256_bytes(dest.read_bytes())
            size = int(dest.stat().st_size)
        else:
            shutil.copy2(source, dest)
            digest = _sha256_file(dest)
            size = int(dest.stat().st_size)
        entries_by_role[role] = {
            "role": role,
            "source": str(source.resolve()),
            "relative_path": relative.as_posix(),
            "sha256": digest,
            "size": size,
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
    from src.sim.episode.types import V12EpisodeResult  # 延迟导入避免环

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
        if snapshot_sha.get(role):
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
