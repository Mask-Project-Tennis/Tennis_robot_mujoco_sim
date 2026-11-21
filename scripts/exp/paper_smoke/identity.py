"""运行身份与配置哈希：config_hash、runtime 覆盖快照、config_id 与续跑身份键。

身份键顺序必须与 ``batch_runner`` 一致，否则续跑会错配历史 CSV。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from scripts.exp.batch_runner import compute_config_hash

from .manifest import (
    MANIFEST_SCHEMA,
    RUNTIME_SNAPSHOT_DIR,
    ManifestError,
    _display_path,
    _file_sha256,
    _resolve_project_path,
)


def _resolved_configs(manifest: Mapping[str, Any]) -> dict[str, str]:
    """返回 manifest 冻结配置文件（角色 → 项目内相对路径）。"""
    configs = manifest["configs"]
    return {
        "robot": str(configs["robot"]),
        "runtime": str(configs["runtime"]),
        "racket": str(configs["racket"]),
        "task": str(configs["task_overlay"]),
    }


def _config_hash(manifest: Mapping[str, Any]) -> tuple[str, dict[str, dict[str, str]]]:
    """计算冻结配置的 config_hash，并返回每个角色文件的 sha256。

    Raises:
        ManifestError: 任一冻结配置文件缺失（T7 未冻结/路径错误）。
    """
    role_paths = _resolved_configs(manifest)
    details: dict[str, dict[str, str]] = {}
    for role, rel in role_paths.items():
        path = _resolve_project_path(rel)
        if not path.is_file():
            raise ManifestError(
                f"冻结配置文件缺失（T7 未冻结或路径错误）: {role} -> {path}")
        details[role] = {"path": rel, "sha256": _file_sha256(path)}
    try:
        digest = compute_config_hash(role_paths, extra={"manifest_schema": MANIFEST_SCHEMA})
    except FileNotFoundError as exc:  # pragma: no cover - 上面已显式检查
        raise ManifestError(str(exc)) from exc
    return digest, details


def _runtime_effective_config(
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    case: Mapping[str, Any],
    run_root: Path,
) -> tuple[str, dict[str, Any]]:
    """解析 case 级 runtime 覆盖，返回 (``--runtime-config`` 值, 快照信息)。

    R1D 的 max_tcp_speed 只在 ``--runtime-config`` 指向的 YAML 里；不能用
    ``--max-tcp``（R1D 共享面拒绝）。因此按 base runtime + case 覆盖生成
    快照文件，并在 resolved/CSV 记录 base 路径、覆盖字典与快照 sha256。

    Args:
        manifest: 规范化 manifest（含 configs.runtime）。
        unit: 实验/依赖单元。
        case: 规范化 case（含 runtime_overrides）。
        run_root: 本次运行根目录（快照写在 run_root/runtime_overrides/ 下）。

    Returns:
        (传给 --runtime-config 的路径字符串, runtime 信息字典)。
    """
    base_rel = str(manifest["configs"]["runtime"])
    overrides = dict(case.get("runtime_overrides") or {})
    base_path = _resolve_project_path(base_rel)
    base_sha = _file_sha256(base_path)
    if not overrides:
        return base_rel, {
            "base": base_rel,
            "base_sha256": base_sha,
            "overrides": {},
            "snapshot": None,
            "sha256": base_sha,
        }
    payload = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ManifestError(f"runtime 配置必须是映射，无法应用覆盖: {base_path}")
    effective = dict(payload)
    effective.update(overrides)
    snapshot_dir = run_root / RUNTIME_SNAPSHOT_DIR / str(unit["id"])
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{case['id']}.yaml"
    snapshot_path.write_text(
        yaml.safe_dump(dict(effective), sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    rel = _display_path(snapshot_path)
    return rel, {
        "base": base_rel,
        "base_sha256": base_sha,
        "overrides": dict(overrides),
        "snapshot": rel,
        "sha256": _file_sha256(snapshot_path),
    }


def _stable_config_id_argv(
    argv: Mapping[str, Any], runtime_info: Mapping[str, Any],
) -> dict[str, Any]:
    """生成用于 config_id 的稳定 argv（N15）。

    runtime 覆盖任务的 ``--runtime-config`` 指向本次 run_root 下的快照文件，
    直接进入 config_id 会让同一条任务在换 ``--output`` 后身份变化。这里用
    ``<base_stem>#<runtime_sha8>`` 稳定 token 取代快照路径；``--output`` 与
    ``--seed`` 已由 ``make_config_id`` 忽略。
    """
    stable = dict(argv)
    snapshot = runtime_info.get("snapshot")
    if snapshot:
        stem = Path(str(runtime_info.get("base", ""))).stem
        sha8 = str(runtime_info.get("sha256", ""))[:8]
        stable["--runtime-config"] = f"<{stem}#{sha8}>"
    return stable


def _record_key(row: Mapping[str, Any]) -> tuple[str, str, int] | None:
    """从 runs.csv 行提取 ``(experiment, config_id, seed)`` 续跑键。

    Args:
        row: CSV 行（字段可能缺失或为空）。

    Returns:
        完整键；缺 experiment/config_id 或 seed 非法时为 None。
    """
    experiment = str(row.get("experiment") or "")
    config_id = str(row.get("config_id") or "")
    try:
        seed = int(row.get("seed"))
    except (TypeError, ValueError):
        return None
    if not experiment or not config_id:
        return None
    return (experiment, config_id, seed)


def _records_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    """按续跑键索引 runs.csv 记录（后写覆盖先写）。

    Args:
        records: ``_collect_runs`` 返回的原始记录（绝对路径）。

    Returns:
        续跑键到 CSV 行的映射。
    """
    index: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in records:
        key = _record_key(row)
        if key is not None:
            index[key] = row
    return index


def _resolved_task_index(
    resolved: Mapping[str, Any],
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    """按续跑键索引 manifest 任务。

    Args:
        resolved: ``resolve_manifest`` 的结果。

    Returns:
        续跑键到任务字典的映射。
    """
    from .tasks import _iter_resolved_tasks  # 延迟导入：避免 identity ↔ tasks 循环依赖
    index: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for task in _iter_resolved_tasks(resolved):
        key = _record_key(task)
        if key is not None:
            index[key] = task
    return index
