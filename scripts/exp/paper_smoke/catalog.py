"""catalog.json 生成：声明本次 smoke 的数据来源集合与各单元计划。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import SOURCE_SETS
from .tasks import _run_root_relative


def _catalog_payload(
    resolved: Mapping[str, Any], run_root: Path,
) -> dict[str, Any]:
    """构造 output 根的 ``catalog.json`` 声明（schema_version=1）。

    路径相对 catalog 文件本身声明，使整棵 run_root 可移动；``data_root``
    位于 run_root 内时写相对路径、外部时写绝对路径（不搬旧数据）。
    """
    source_manifest = resolved.get("source_manifest") or {}
    data_root = Path(str(resolved["data_root"]))
    identity = resolved.get("identity") or {}
    return {
        "schema_version": 1,
        "synthetic": False,
        "robot_id": str(identity.get("robot_id", "")),
        "run_root": ".",
        "data_root": _run_root_relative(run_root, data_root),
        "runs_csv": "runs.csv",
        "manifest_path": "manifest_resolved.json",
        "path_base": "run_root",
        "source_sets": dict(SOURCE_SETS),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_manifest": {
            "path": str(source_manifest.get("path", "")),
            "sha256": str(source_manifest.get("sha256", "")),
        },
        "config_hash": str(resolved.get("config_hash", "")),
        "seeds": [int(seed) for seed in resolved.get("seeds", [])],
    }


def _write_catalog(run_root: Path, resolved: Mapping[str, Any]) -> Path:
    """在 output 根写 ``catalog.json``，返回其路径。"""
    from .cli import _write_json  # 延迟导入：避免 catalog ↔ cli 循环依赖
    path = Path(run_root) / "catalog.json"
    _write_json(path, _catalog_payload(resolved, Path(run_root)))
    return path
