# ruff: noqa: PLC0414
"""论文统计口径统一提取（兼容 facade）。

原 2682 行实现已按职责拆分到 :mod:`scripts.extract.paired_analysis`：
``types`` / ``identity`` / ``sources`` / ``dataset`` / ``metrics`` /
``comparisons`` / ``report`` / ``cli``。本文件只做三件事：

1. 脚本模式运行时把仓库根加入 ``sys.path``（``python scripts/extract/paired_stats.py``）；
2. 从子模块显式重导出旧公开/私有名字（``X as X``，保证 ``is`` 相同、不复制实现）；
3. 保留 ``if __name__ == "__main__": raise SystemExit(main())``。

统计口径与 CLI 行为均以 ``paired_analysis`` 子模块为准。
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # paired_stats.py -> extract -> scripts -> 仓库根
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from scripts.extract.paired_analysis.cli import PROJECT as PROJECT
from scripts.extract.paired_analysis.cli import build_parser as build_parser
from scripts.extract.paired_analysis.cli import main as main
from scripts.extract.paired_analysis.comparisons import _did as _did
from scripts.extract.paired_analysis.comparisons import _did_reason as _did_reason
from scripts.extract.paired_analysis.comparisons import _tier_cell as _tier_cell
from scripts.extract.paired_analysis.comparisons import e1_speed_sweep as e1_speed_sweep
from scripts.extract.paired_analysis.comparisons import e2_limit_cost as e2_limit_cost
from scripts.extract.paired_analysis.comparisons import e3_nominal as e3_nominal
from scripts.extract.paired_analysis.comparisons import e4_grid as e4_grid
from scripts.extract.paired_analysis.comparisons import e5_perception as e5_perception
from scripts.extract.paired_analysis.comparisons import e7_corners as e7_corners
from scripts.extract.paired_analysis.comparisons import (
    e8_limit_mechanism as e8_limit_mechanism,
)
from scripts.extract.paired_analysis.comparisons import (
    e9_shared_input as e9_shared_input,
)
from scripts.extract.paired_analysis.comparisons import (
    exclusion_report as exclusion_report,
)
from scripts.extract.paired_analysis.dataset import _check_attempts as _check_attempts
from scripts.extract.paired_analysis.dataset import _discover_modes as _discover_modes
from scripts.extract.paired_analysis.dataset import (
    _final_attempt_rows as _final_attempt_rows,
)
from scripts.extract.paired_analysis.dataset import _group_attempts as _group_attempts
from scripts.extract.paired_analysis.dataset import (
    _is_completed_row as _is_completed_row,
)
from scripts.extract.paired_analysis.dataset import (
    _validate_dataset as _validate_dataset,
)
from scripts.extract.paired_analysis.dataset import resolve_mode as resolve_mode
from scripts.extract.paired_analysis.identity import (
    _PATH_CONFIG_KEYS as _PATH_CONFIG_KEYS,
)
from scripts.extract.paired_analysis.identity import (
    _RUNTIME_CONFIG_KEYS as _RUNTIME_CONFIG_KEYS,
)
from scripts.extract.paired_analysis.identity import (
    _base_tier_of_cfg as _base_tier_of_cfg,
)
from scripts.extract.paired_analysis.identity import (
    _design_fingerprint as _design_fingerprint,
)
from scripts.extract.paired_analysis.identity import _feedforward_of as _feedforward_of
from scripts.extract.paired_analysis.identity import (
    _feedforward_of_cfg as _feedforward_of_cfg,
)
from scripts.extract.paired_analysis.identity import _is_r1d_row as _is_r1d_row
from scripts.extract.paired_analysis.identity import (
    _normalize_config_path as _normalize_config_path,
)
from scripts.extract.paired_analysis.identity import _obs_freq_of as _obs_freq_of
from scripts.extract.paired_analysis.identity import _ov as _ov
from scripts.extract.paired_analysis.identity import (
    _parse_override_value as _parse_override_value,
)
from scripts.extract.paired_analysis.identity import (
    _row_artifact_value as _row_artifact_value,
)
from scripts.extract.paired_analysis.identity import _seed_value as _seed_value
from scripts.extract.paired_analysis.identity import _tcp_label as _tcp_label
from scripts.extract.paired_analysis.identity import case_of as case_of
from scripts.extract.paired_analysis.identity import mode_of as mode_of
from scripts.extract.paired_analysis.identity import overrides_of as overrides_of
from scripts.extract.paired_analysis.identity import record_identity as record_identity
from scripts.extract.paired_analysis.identity import shared_input_of as shared_input_of
from scripts.extract.paired_analysis.identity import (
    space_perturb_of as space_perturb_of,
)
from scripts.extract.paired_analysis.identity import speed_of as speed_of
from scripts.extract.paired_analysis.identity import tcp_label_of as tcp_label_of
from scripts.extract.paired_analysis.identity import tier_of as tier_of
from scripts.extract.paired_analysis.identity import (
    time_perturb_ms_of as time_perturb_ms_of,
)
from scripts.extract.paired_analysis.metrics import (
    _physical_pair_summary as _physical_pair_summary,
)
from scripts.extract.paired_analysis.metrics import _unique_index as _unique_index
from scripts.extract.paired_analysis.metrics import baseline_cell as baseline_cell
from scripts.extract.paired_analysis.metrics import cell_stats as cell_stats
from scripts.extract.paired_analysis.metrics import mcnemar_exact as mcnemar_exact
from scripts.extract.paired_analysis.metrics import paired_bootstrap as paired_bootstrap
from scripts.extract.paired_analysis.metrics import paired_test as paired_test
from scripts.extract.paired_analysis.metrics import (
    paired_test_physical as paired_test_physical,
)
from scripts.extract.paired_analysis.metrics import physical_of as physical_of
from scripts.extract.paired_analysis.metrics import physical_stats as physical_stats
from scripts.extract.paired_analysis.metrics import sign_test as sign_test
from scripts.extract.paired_analysis.metrics import wilson as wilson
from scripts.extract.paired_analysis.report import (
    _catalog_metadata as _catalog_metadata,
)
from scripts.extract.paired_analysis.report import (
    _existing_sources as _existing_sources,
)
from scripts.extract.paired_analysis.report import _fmt_ci as _fmt_ci
from scripts.extract.paired_analysis.report import _fmt_optional as _fmt_optional
from scripts.extract.paired_analysis.report import _fmt_p as _fmt_p
from scripts.extract.paired_analysis.report import _fmt_pp as _fmt_pp
from scripts.extract.paired_analysis.report import (
    _multi_mode_index as _multi_mode_index,
)
from scripts.extract.paired_analysis.report import (
    _physical_conflict_summary as _physical_conflict_summary,
)
from scripts.extract.paired_analysis.report import _print_report as _print_report
from scripts.extract.paired_analysis.report import _reason_text as _reason_text
from scripts.extract.paired_analysis.report import (
    _reject_physical_conflicts as _reject_physical_conflicts,
)
from scripts.extract.paired_analysis.report import _write_stats as _write_stats
from scripts.extract.paired_analysis.report import build_tables as build_tables
from scripts.extract.paired_analysis.report import compute_summary as compute_summary
from scripts.extract.paired_analysis.report import (
    default_output_dir as default_output_dir,
)
from scripts.extract.paired_analysis.sources import (
    _PHYSICAL_COLUMNS as _PHYSICAL_COLUMNS,
)
from scripts.extract.paired_analysis.sources import (
    _SHARED_RESULT_COLUMNS as _SHARED_RESULT_COLUMNS,
)
from scripts.extract.paired_analysis.sources import (
    _STATE_RESULT_COLUMNS as _STATE_RESULT_COLUMNS,
)
from scripts.extract.paired_analysis.sources import _artifact_key as _artifact_key
from scripts.extract.paired_analysis.sources import (
    _candidate_conflict_fields as _candidate_conflict_fields,
)
from scripts.extract.paired_analysis.sources import _catalog_module as _catalog_module
from scripts.extract.paired_analysis.sources import _dedupe_sources as _dedupe_sources
from scripts.extract.paired_analysis.sources import (
    _enrichment_conflicts as _enrichment_conflicts,
)
from scripts.extract.paired_analysis.sources import (
    _enrichment_field_names as _enrichment_field_names,
)
from scripts.extract.paired_analysis.sources import (
    _enrichment_signature as _enrichment_signature,
)
from scripts.extract.paired_analysis.sources import _identity_text as _identity_text
from scripts.extract.paired_analysis.sources import _load_rows as _load_rows
from scripts.extract.paired_analysis.sources import _manifest_tasks as _manifest_tasks
from scripts.extract.paired_analysis.sources import _normalize_path as _normalize_path
from scripts.extract.paired_analysis.sources import (
    _normalized_column as _normalized_column,
)
from scripts.extract.paired_analysis.sources import _physical_index as _physical_index
from scripts.extract.paired_analysis.sources import _read_csv_rows as _read_csv_rows
from scripts.extract.paired_analysis.sources import (
    _resolve_physical_source as _resolve_physical_source,
)
from scripts.extract.paired_analysis.sources import _runs_csv_path as _runs_csv_path
from scripts.extract.paired_analysis.sources import _runs_csv_paths as _runs_csv_paths
from scripts.extract.paired_analysis.sources import _sample_keys as _sample_keys
from scripts.extract.paired_analysis.sources import load as load
from scripts.extract.paired_analysis.sources import make_context as make_context
from scripts.extract.paired_analysis.types import DATA as DATA
from scripts.extract.paired_analysis.types import DEFAULT_DATA_ROOT as DEFAULT_DATA_ROOT
from scripts.extract.paired_analysis.types import HIT_SEMANTICS as HIT_SEMANTICS
from scripts.extract.paired_analysis.types import MODE_CHOICES as MODE_CHOICES
from scripts.extract.paired_analysis.types import MODES as MODES
from scripts.extract.paired_analysis.types import (
    MULTI_MODE_FILENAME as MULTI_MODE_FILENAME,
)
from scripts.extract.paired_analysis.types import MULTI_MODE_SCHEMA as MULTI_MODE_SCHEMA
from scripts.extract.paired_analysis.types import (
    NOFEEDFORWARD_SUFFIX as NOFEEDFORWARD_SUFFIX,
)
from scripts.extract.paired_analysis.types import REFERENCE as REFERENCE
from scripts.extract.paired_analysis.types import SCHEMA as SCHEMA
from scripts.extract.paired_analysis.types import SEED_DIR_RE as SEED_DIR_RE
from scripts.extract.paired_analysis.types import STATS_FILENAMES as STATS_FILENAMES
from scripts.extract.paired_analysis.types import TABLE3_CELLS as TABLE3_CELLS
from scripts.extract.paired_analysis.types import TIER_LABELS as TIER_LABELS
from scripts.extract.paired_analysis.types import VALID_MODES as VALID_MODES
from scripts.extract.paired_analysis.types import (
    DatasetConflictError as DatasetConflictError,
)
from scripts.extract.paired_analysis.types import (
    DuplicateRecordError as DuplicateRecordError,
)
from scripts.extract.paired_analysis.types import HitPolicy as HitPolicy
from scripts.extract.paired_analysis.types import IdentityError as IdentityError
from scripts.extract.paired_analysis.types import (
    ModeSelectionError as ModeSelectionError,
)
from scripts.extract.paired_analysis.types import PhysicalIndex as PhysicalIndex
from scripts.extract.paired_analysis.types import RecordIdentity as RecordIdentity
from scripts.extract.paired_analysis.types import StatsContext as StatsContext
from scripts.extract.paired_analysis.types import StatsError as StatsError
from scripts.extract.paired_analysis.types import _repo_root as _repo_root
from scripts.extract.paired_analysis.types import _to_bool_or_none as _to_bool_or_none
from scripts.extract.paired_analysis.types import cfg_of as cfg_of
from scripts.extract.paired_analysis.types import is_error as is_error

__all__ = [
    "DATA",
    "DEFAULT_DATA_ROOT",
    "HIT_SEMANTICS",
    "MODES",
    "MODE_CHOICES",
    "MULTI_MODE_FILENAME",
    "MULTI_MODE_SCHEMA",
    "NOFEEDFORWARD_SUFFIX",
    "PROJECT",
    "REFERENCE",
    "SCHEMA",
    "SEED_DIR_RE",
    "STATS_FILENAMES",
    "TABLE3_CELLS",
    "TIER_LABELS",
    "VALID_MODES",
    "_PATH_CONFIG_KEYS",
    "_PHYSICAL_COLUMNS",
    "_RUNTIME_CONFIG_KEYS",
    "_SHARED_RESULT_COLUMNS",
    "_STATE_RESULT_COLUMNS",
    "Any",
    "Callable",
    "Counter",
    "DatasetConflictError",
    "DuplicateRecordError",
    "Hashable",
    "HitPolicy",
    "IdentityError",
    "Mapping",
    "ModeSelectionError",
    "Path",
    "PhysicalIndex",
    "RecordIdentity",
    "StatsContext",
    "StatsError",
    "_artifact_key",
    "_base_tier_of_cfg",
    "_candidate_conflict_fields",
    "_catalog_metadata",
    "_catalog_module",
    "_check_attempts",
    "_dedupe_sources",
    "_design_fingerprint",
    "_did",
    "_did_reason",
    "_discover_modes",
    "_enrichment_conflicts",
    "_enrichment_field_names",
    "_enrichment_signature",
    "_existing_sources",
    "_feedforward_of",
    "_feedforward_of_cfg",
    "_final_attempt_rows",
    "_fmt_ci",
    "_fmt_optional",
    "_fmt_p",
    "_fmt_pp",
    "_group_attempts",
    "_identity_text",
    "_is_completed_row",
    "_is_r1d_row",
    "_load_rows",
    "_manifest_tasks",
    "_multi_mode_index",
    "_normalize_config_path",
    "_normalize_path",
    "_normalized_column",
    "_obs_freq_of",
    "_ov",
    "_parse_override_value",
    "_physical_conflict_summary",
    "_physical_index",
    "_physical_pair_summary",
    "_print_report",
    "_read_csv_rows",
    "_reason_text",
    "_reject_physical_conflicts",
    "_repo_root",
    "_resolve_physical_source",
    "_row_artifact_value",
    "_runs_csv_path",
    "_runs_csv_paths",
    "_sample_keys",
    "_seed_value",
    "_tcp_label",
    "_tier_cell",
    "_to_bool_or_none",
    "_unique_index",
    "_validate_dataset",
    "_write_stats",
    "argparse",
    "baseline_cell",
    "build_parser",
    "build_tables",
    "case_of",
    "cell_stats",
    "cfg_of",
    "compute_summary",
    "csv",
    "dataclass",
    "datetime",
    "default_output_dir",
    "defaultdict",
    "e1_speed_sweep",
    "e2_limit_cost",
    "e3_nominal",
    "e4_grid",
    "e5_perception",
    "e7_corners",
    "e8_limit_mechanism",
    "e9_shared_input",
    "exclusion_report",
    "field",
    "is_error",
    "json",
    "load",
    "main",
    "make_context",
    "math",
    "mcnemar_exact",
    "mean",
    "mode_of",
    "os",
    "overrides_of",
    "paired_bootstrap",
    "paired_test",
    "paired_test_physical",
    "physical_of",
    "physical_stats",
    "random",
    "re",
    "record_identity",
    "replace",
    "resolve_mode",
    "shared_input_of",
    "sign_test",
    "space_perturb_of",
    "speed_of",
    "sys",
    "tcp_label_of",
    "tier_of",
    "time_perturb_ms_of",
    "timezone",
    "wilson",
]

if __name__ == "__main__":
    raise SystemExit(main())
