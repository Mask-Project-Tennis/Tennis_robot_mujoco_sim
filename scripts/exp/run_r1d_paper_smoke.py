"""R1D 论文 smoke 旧入口（兼容 facade）。

实现已按 C1 拆分迁入 ``scripts.exp.paper_smoke``；本文件只保留脚本启动所需的
``sys.path`` 注入与符号重导出，保证既有调用方/测试从原模块名拿到的仍是同一对象。
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.exp import batch_runner
from scripts.exp.batch_runner import (
    CSV_FIELDS,
    ExperimentSpec,
    RunIdentity,
    compute_config_hash,
    make_config_id,
)
from scripts.exp.paper_smoke.artifacts import (
    _artifact_metrics,
    _config_json_with_output,
    _is_absolute_path_text,
    _load_npz,
    _npz_scalar_text,
    _portable_config_json,
    _portable_run_row,
    _resolve_run_artifact_dir,
    _resolve_run_log_path,
)
from scripts.exp.paper_smoke.catalog import _catalog_payload, _write_catalog
from scripts.exp.paper_smoke.cli import (
    _check_tool_outputs,
    _print_resolved_summary,
    _run_tool_dependency,
    _substitute,
    _write_json,
    main,
)
from scripts.exp.paper_smoke.identity import (
    _config_hash,
    _record_key,
    _records_index,
    _resolved_configs,
    _resolved_task_index,
    _runtime_effective_config,
    _stable_config_id_argv,
)
from scripts.exp.paper_smoke.manifest import (
    ALLOWED_CASE_ARGV,
    ARTIFACT_FIELDS,
    CASE_DEFAULT_KEYS,
    CASE_SPEC_FIELDS,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RUN_ROOT,
    FORBIDDEN_CASE_ARGV,
    MANIFEST_SCHEMA,
    PROJECT,
    REQUIRED_WORKERS,
    RESOLVED_SCHEMA,
    RUNS_HIT_SEMANTICS,
    RUNTIME_SNAPSHOT_DIR,
    SOURCE_SETS,
    V12_EPISODE_NAME,
    V12_EPISODE_NPZ_NAME,
    V12_NPZ_KEYS,
    VALID_CASE_STATUS,
    VALID_MODE_FILTERS,
    VALID_MODES,
    VALID_UNIT_KINDS,
    ManifestError,
    _allowed_override_keys,
    _case_token_list,
    _check_dependency_refs,
    _check_seed_rule,
    _display_path,
    _file_sha256,
    _format_override,
    _load_base_runtime_keys,
    _normalize_manifest,
    _resolve_project_path,
    _validate_argv,
    _validate_case_defaults,
    _validate_case_plan_fields,
    _validate_cases,
    _validate_override_mapping,
    _validate_runtime_overrides,
    _validate_seeds,
    _validate_units,
    load_manifest,
    validate_manifest,
)
from scripts.exp.paper_smoke.resume import (
    _MISSING,
    _OVERRIDE_SECTION_FIELDS,
    _SERVE_SCENE_ALIASES,
    _TARGET_FIELD_ALIASES,
    _apply_actual_paths,
    _artifact_issues,
    _collect_runs,
    _expected_output_matches,
    _is_completed_record,
    _log_issues,
    _lookup_config_path,
    _numeric_value_matches,
    _override_config_path,
    _override_value_matches,
    _preserve_previous_metrics,
    _print_artifact_problems,
    _r1d_npz_issues,
    _read_previous_runs,
    _row_path,
    _v12_config_issues,
    _v12_identity_issues,
    _v12_npz_issues,
    _v12_payload_issues,
    _validate_completed_artifacts,
)
from scripts.exp.paper_smoke.tasks import (
    PORTABLE_TASK_PATH_FIELDS,
    _build_task,
    _filter_tasks_by_mode,
    _iter_resolved_tasks,
    _portable_resolved,
    _run_root_relative,
    _task_log_path,
    build_specs_from_manifest,
    resolve_manifest,
)


def __getattr__(name: str) -> Any:
    """兼容旧模块的可变缓存属性 ``_OVERRIDE_KEYS_CACHE``。

    override 白名单缓存实际存放在 ``paper_smoke.manifest``；facade 不复制值，
    避免出现两份不同步的缓存。
    """
    if name == "_OVERRIDE_KEYS_CACHE":
        from scripts.exp.paper_smoke import manifest as _manifest

        return _manifest._OVERRIDE_KEYS_CACHE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ALLOWED_CASE_ARGV",
    "ARTIFACT_FIELDS",
    "CASE_DEFAULT_KEYS",
    "CASE_SPEC_FIELDS",
    "CSV_FIELDS",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_RUN_ROOT",
    "FORBIDDEN_CASE_ARGV",
    "MANIFEST_SCHEMA",
    "PORTABLE_TASK_PATH_FIELDS",
    "PROJECT",
    "REQUIRED_WORKERS",
    "RESOLVED_SCHEMA",
    "RUNS_HIT_SEMANTICS",
    "RUNTIME_SNAPSHOT_DIR",
    "SOURCE_SETS",
    "V12_EPISODE_NAME",
    "V12_EPISODE_NPZ_NAME",
    "V12_NPZ_KEYS",
    "VALID_CASE_STATUS",
    "VALID_MODES",
    "VALID_MODE_FILTERS",
    "VALID_UNIT_KINDS",
    "_MISSING",
    "_OVERRIDE_SECTION_FIELDS",
    "_ROOT",
    "_SERVE_SCENE_ALIASES",
    "_TARGET_FIELD_ALIASES",
    "ExperimentSpec",
    "ManifestError",
    "Path",
    "RunIdentity",
    "_allowed_override_keys",
    "_apply_actual_paths",
    "_artifact_issues",
    "_artifact_metrics",
    "_build_task",
    "_case_token_list",
    "_catalog_payload",
    "_check_dependency_refs",
    "_check_seed_rule",
    "_check_tool_outputs",
    "_collect_runs",
    "_config_hash",
    "_config_json_with_output",
    "_display_path",
    "_expected_output_matches",
    "_file_sha256",
    "_filter_tasks_by_mode",
    "_format_override",
    "_is_absolute_path_text",
    "_is_completed_record",
    "_iter_resolved_tasks",
    "_load_base_runtime_keys",
    "_load_npz",
    "_log_issues",
    "_lookup_config_path",
    "_normalize_manifest",
    "_npz_scalar_text",
    "_numeric_value_matches",
    "_override_config_path",
    "_override_value_matches",
    "_portable_config_json",
    "_portable_resolved",
    "_portable_run_row",
    "_preserve_previous_metrics",
    "_print_artifact_problems",
    "_print_resolved_summary",
    "_r1d_npz_issues",
    "_read_previous_runs",
    "_record_key",
    "_records_index",
    "_resolve_project_path",
    "_resolve_run_artifact_dir",
    "_resolve_run_log_path",
    "_resolved_configs",
    "_resolved_task_index",
    "_row_path",
    "_run_root_relative",
    "_run_tool_dependency",
    "_runtime_effective_config",
    "_stable_config_id_argv",
    "_substitute",
    "_task_log_path",
    "_v12_config_issues",
    "_v12_identity_issues",
    "_v12_npz_issues",
    "_v12_payload_issues",
    "_validate_argv",
    "_validate_case_defaults",
    "_validate_case_plan_fields",
    "_validate_cases",
    "_validate_completed_artifacts",
    "_validate_override_mapping",
    "_validate_runtime_overrides",
    "_validate_seeds",
    "_validate_units",
    "_write_catalog",
    "_write_json",
    "argparse",
    "batch_runner",
    "build_specs_from_manifest",
    "compute_config_hash",
    "copy",
    "csv",
    "datetime",
    "hashlib",
    "json",
    "load_manifest",
    "main",
    "make_config_id",
    "math",
    "np",
    "resolve_manifest",
    "subprocess",
    "sys",
    "time",
    "timezone",
    "validate_manifest",
    "yaml",
    "zipfile",
]


if __name__ == "__main__":
    raise SystemExit(main())
