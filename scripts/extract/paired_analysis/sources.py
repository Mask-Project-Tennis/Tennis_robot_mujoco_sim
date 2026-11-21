"""数据来源读取：results.csv/runs.csv、catalog、physical 富化与来源冲突。

本模块负责把磁盘上的原始行变成带 ``_physical_status``/``_physical_source`` 的行，
并按完整身份（必要时退回唯一 artifact 路径）做跨视图连接；不在此做 attempt 收敛
的最终选择（见 :mod:`paired_analysis.dataset`）。
"""
from __future__ import annotations

import csv
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .identity import _design_fingerprint, _row_artifact_value, mode_of, record_identity
from .types import (
    DEFAULT_DATA_ROOT,
    VALID_MODES,
    DatasetConflictError,
    PhysicalIndex,
    StatsContext,
    StatsError,
    _repo_root,
)

# ----------------------------------------------------------------------------
# 数据加载（含 R1D runs.csv 的 physical 列富化）
# ----------------------------------------------------------------------------

def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读 CSV 为 dict 列表（缺失返回空）。"""
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))




def _catalog_module() -> Any:
    """延迟导入 ``scripts.exp.run_catalog``（catalog 契约的唯一事实来源）。"""
    if str(_repo_root()) not in sys.path:
        sys.path.insert(0, str(_repo_root()))
    from scripts.exp import run_catalog

    return run_catalog




def make_context(
    data_root: Path | None = None,
    *,
    catalog_path: Path | None = None,
    mode: str | None = None,
) -> StatsContext:
    """构造统计上下文。

    Args:
        data_root: 实验数据根；为 None 时用 catalog 声明的 data_root，再退回默认根。
        catalog_path: 可选 ``catalog.json``；显式给出且加载失败时直接抛 StatsError。
        mode: ``torque`` / ``position`` / None（自动推断）。

    Returns:
        不可变 StatsContext。

    Raises:
        StatsError: catalog 非法或 mode 取值未知。
    """
    if mode is not None and mode not in VALID_MODES:
        raise StatsError(f"未知执行模式 {mode!r}：只接受 {' / '.join(VALID_MODES)}")
    catalog = None
    if catalog_path is not None:
        api = _catalog_module()
        try:
            catalog = api.load_catalog(Path(catalog_path))
        except api.CatalogError as exc:
            raise StatsError(f"catalog 加载失败（{catalog_path}）：{exc}") from exc
    if catalog is not None:
        root = Path(data_root) if data_root is not None else catalog.data_root
        return StatsContext(
            data_root=Path(root),
            run_root=Path(catalog.run_root),
            catalog=catalog,
            catalog_path=Path(catalog.catalog_path),
            mode=mode,
            source_sets=dict(catalog.source_sets),
            synthetic=bool(catalog.synthetic),
            manifest_path=Path(catalog.manifest_path),
        )
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    return StatsContext(data_root=root, run_root=root.parent, mode=mode)




def _normalize_path(value: str, base: Path | None) -> str:
    """把产物路径归一化为可比较键（相对 base 解析、绝对原样、大小写归一）。"""
    if not value:
        return ""
    path = Path(value)
    if base is not None and not path.is_absolute():
        path = base / path
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover - 极端路径错误
        resolved = path
    return os.path.normcase(str(resolved))




def _artifact_key(row: dict[str, str], ctx: StatsContext | None = None) -> str:
    """artifact 身份键：跨 output / data-root 两个根都能对上同一条记录。"""
    base = None if ctx is None else ctx.run_root
    return _normalize_path(_row_artifact_value(row), base)




def _runs_csv_paths(ctx: StatsContext) -> list[Path]:
    """候选 runs.csv：catalog 声明优先，否则数据根下 / 数据根同级。"""
    if ctx.catalog is not None:
        return [Path(ctx.catalog.runs_csv)]
    return [ctx.data_root / "runs.csv", ctx.data_root.parent / "runs.csv"]




def _runs_csv_path(ctx: StatsContext) -> Path | None:
    """实际存在的 runs.csv（缺失返回 None）。"""
    for path in _runs_csv_paths(ctx):
        if path.is_file():
            return path
    return None




# 富化列：results 视图缺失时由 runs 视图补齐（顺序即写回顺序）
_PHYSICAL_COLUMNS = (
    "physical_hit",
    "physical_hit_type",
    "physical_hit_step",
    "contact_event_count",
    "failure_reason",
    "backend",
)



# 共有结果列（R05）：producer 会把 results.csv 的 legacy 结果、完成状态与错误字段
# 原样复制进 runs.csv，因此同一运行的两份视图里这些同名列必须一致。它们只参与
# 一致性比较（富化签名/冲突报告），**不参与富化填空**——填空仍只认
# ``_PHYSICAL_COLUMNS``。这里只比较“同名对同名”，不跨越 legacy 与 physical 口径。
_SHARED_RESULT_COLUMNS = (
    "hit",
    "hit_type",
    "physical_hit",
    "physical_hit_type",
    "physical_hit_step",
    "contact_event_count",
    "failure_reason",
    "backend",
    "status",
    "error",
    "timed_out",
    "exit_code",
)



# 完成状态列（R05）：这些列描述“这次运行是否成功”，空值本身也是信息。
# 例如 results.csv 的 error 为空、runs.csv 却记为非空错误，说明同一身份的两份
# 视图对完成状态说法相反，必须判冲突，不能按“空值=未声明”放过。
_STATE_RESULT_COLUMNS = ("status", "error", "timed_out", "exit_code")




def _normalized_column(row: Mapping[str, Any], column: str) -> str:
    """富化列的可比取值：去空白，并把布尔字面量统一成小写。"""
    raw = row.get(column)
    if raw is None:
        return ""
    text = str(raw).strip()
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered
    return text




def _identity_text(row: dict[str, str], column: str) -> str:
    """身份列的可比取值（``mode`` 走 ``mode_of`` 推断，其余直接读列）。"""
    if column == "mode":
        return mode_of(row) or ""
    return str(row.get(column) or "").strip()




def _enrichment_signature(row: dict[str, str]) -> tuple[str, ...]:
    """富化行归一化签名：设计指纹 + 身份列 + 共有结果列。

    ``_SHARED_RESULT_COLUMNS`` 已包含 ``_PHYSICAL_COLUMNS``，故签名同时覆盖
    physical 丰富列与 legacy 共有列；两份视图同名列不一致即视为不同签名。
    """
    return (
        _design_fingerprint(row),
        _identity_text(row, "mode"),
        _identity_text(row, "robot_id"),
        _identity_text(row, "racket_id"),
        _identity_text(row, "config_hash"),
        *(_normalized_column(row, column) for column in _SHARED_RESULT_COLUMNS),
    )




def _enrichment_field_names() -> tuple[str, ...]:
    """``_enrichment_signature`` 各分量的字段名（错误信息用）。"""
    return (
        "design",
        "mode",
        "robot_id",
        "racket_id",
        "config_hash",
        *_SHARED_RESULT_COLUMNS,
    )




def _enrichment_conflicts(a: dict[str, str], b: dict[str, str]) -> list[str]:
    """比较两行富化相关字段；结果列空值视为未声明，完成状态列不适用该规则。

    比较范围 = 设计指纹 + 身份列 + ``_SHARED_RESULT_COLUMNS``（同名对同名）。
    ``hit``/``hit_type`` 等结果列必须与 physical 列分开看：这里只要求同一
    运行的两份视图对同名列说法一致，不要求 legacy 命中与 physical 命中相等。

    ``_STATE_RESULT_COLUMNS`` 是例外：``results.csv`` 的 error/exit_code 为空、
    ``runs.csv`` 却记为失败时，两份视图对“是否完成”说法相反，空值也必须判冲突
    （R05：单独改完成状态或错误不得被静默放过）。
    """
    fields: list[str] = []
    if _design_fingerprint(a) != _design_fingerprint(b):
        fields.append("design")
    for column in ("mode", "robot_id", "racket_id", "config_hash"):
        left, right = _identity_text(a, column), _identity_text(b, column)
        if left and right and left != right:
            fields.append(column)
    for column in _SHARED_RESULT_COLUMNS:
        left = _normalized_column(a, column)
        right = _normalized_column(b, column)
        if column in _STATE_RESULT_COLUMNS:
            if left != right:
                fields.append(column)
        elif left and right and left != right:
            fields.append(column)
    return fields




def _dedupe_sources(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    """按归一化签名去重（保留首次出现），只消除完全相同的富化行。"""
    seen: set[tuple[str, ...]] = set()
    unique: list[dict[str, str]] = []
    for row in candidates:
        signature = _enrichment_signature(row)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(row)
    return unique




def _candidate_conflict_fields(candidates: list[dict[str, str]]) -> list[str]:
    """多个候选富化行之间不一致的字段名（供 ``DatasetConflictError`` 报告）。"""
    signatures = [_enrichment_signature(row) for row in candidates]
    fields: list[str] = []
    for index, name in enumerate(_enrichment_field_names()):
        if len({signature[index] for signature in signatures}) > 1:
            fields.append(name)
    return fields




def _physical_index(ctx: StatsContext) -> PhysicalIndex:
    """建立 runs.csv 富化行索引（M03：按完整身份，不按 artifact 路径串接）。

    续跑恢复后同一身份可能有多条执行尝试；索引只收最终尝试行（完成行优先，
    否则最后一条失败行），避免把早期失败行的空 physical 列当成第二条富化记录。

    Args:
        ctx: 统计上下文。

    Returns:
        ``PhysicalIndex``；没有带 physical 列的 runs.csv 时返回空索引。
    """
    # 延迟导入：dataset 依赖 sources，此处反向使用以避免模块级循环导入。
    from .dataset import _final_attempt_rows
    for path in _runs_csv_paths(ctx):
        rows = _read_csv_rows(path)
        if not rows:
            continue
        if "physical_hit" not in rows[0] and "physical_hit_type" not in rows[0]:
            continue
        rows, _ = _final_attempt_rows(rows, path)
        by_identity: dict[tuple[Any, ...], list[dict[str, str]]] = {}
        by_artifact: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_identity.setdefault(record_identity(row).join_key(), []).append(row)
            key = _artifact_key(row, ctx)
            if key:
                by_artifact.setdefault(key, []).append(row)
        return PhysicalIndex(by_identity=by_identity, by_artifact=by_artifact, path=path)
    return PhysicalIndex()




def _resolve_physical_source(
    results_row: dict[str, str],
    index: PhysicalIndex,
    ctx: StatsContext,
    *,
    results_path: Path | None = None,
) -> tuple[dict[str, str] | None, str, list[str]]:
    """为一行 results 记录选择可安全回填的 runs 富化行（M03 唯一连接入口）。

    连接顺序：先按 ``RecordIdentity.join_key()``（完整身份）命中；只有该行缺少身份
    组件（legacy 数据）时才退回 artifact 路径通道，且要求路径唯一、两侧可用字段一致。

    Args:
        results_row: results.csv 的一行。
        index: runs.csv 富化行索引。
        ctx: 统计上下文（artifact 相对路径归一化用）。
        results_path: 该行所属 results 文件（错误信息用）。

    Returns:
        ``(source, status, conflict_fields)``；``source`` 为 None 表示不回填，
        ``status`` ∈ ``matched`` / ``conflict`` / ``missing``。

    Raises:
        DatasetConflictError: 同一身份键命中多行且归一化后不一致。
    """
    identity = record_identity(results_row)
    has_identity = bool(
        identity.experiment and identity.case and identity.config_id and identity.seed >= 0
    )
    candidates = index.by_identity.get(identity.join_key(), []) if has_identity else []
    if candidates:
        unique = _dedupe_sources(candidates)
        if len(unique) > 1:
            raise DatasetConflictError(
                "runs.csv 富化行冲突："
                f"join_key={identity.join_key()}，results 文件={results_path or '?'}，"
                f"runs 文件={index.path or '?'}，"
                f"冲突字段={_candidate_conflict_fields(unique)}；"
                "同一身份下禁止静默择一，请先修正数据"
            )
        conflicts = _enrichment_conflicts(unique[0], results_row)
        if conflicts:
            return None, "conflict", conflicts
        return unique[0], "matched", []
    if not has_identity:
        path_key = _artifact_key(results_row, ctx)
        path_candidates = index.by_artifact.get(path_key, []) if path_key else []
        unique = _dedupe_sources(path_candidates)
        if len(unique) > 1:
            raise DatasetConflictError(
                "runs.csv 富化行冲突（legacy 路径通道）："
                f"artifact={path_key}，results 文件={results_path or '?'}，"
                f"runs 文件={index.path or '?'}，"
                f"冲突字段={_candidate_conflict_fields(unique)}；"
                "路径下存在多条不一致富化记录，请先修正数据"
            )
        if unique:
            conflicts = _enrichment_conflicts(unique[0], results_row)
            if conflicts:
                return None, "conflict", conflicts
            return unique[0], "matched", []
    return None, "missing", []




def _load_rows(ctx: StatsContext, name: str) -> list[dict[str, str]]:
    """读取实验 results.csv，按执行模式过滤，并按完整身份富化 physical 列。

    M03：富化只走 ``RecordIdentity.join_key()``（legacy 缺身份组件时退回唯一 artifact
    路径），不再以 artifact 路径作为跨 mode/身份串接的通道。

    续跑恢复（P2）：同一身份的多条执行尝试先按 :func:`_final_attempt_rows` 收敛为
    一条统计样本（完成行优先，否则最后一条失败行），失败尝试仍保留在 CSV 原文件。
    """
    # 延迟导入：dataset 依赖 sources，此处反向使用以避免模块级循环导入。
    from .dataset import _final_attempt_rows
    path = ctx.data_root / name / "results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}")
    rows = _read_csv_rows(path)
    rows, _ = _final_attempt_rows(rows, path)
    index = _physical_index(ctx)
    kept: list[dict[str, str]] = []
    for row in rows:
        if ctx.mode is not None and mode_of(row) not in (None, ctx.mode):
            continue
        source, status, fields = _resolve_physical_source(
            row, index, ctx, results_path=path
        )
        if source is not None:
            for column in _PHYSICAL_COLUMNS:
                if not (row.get(column) or "").strip() and (source.get(column) or "").strip():
                    row[column] = source[column]
            row["_physical_source"] = str(index.path)
        row["_physical_status"] = status
        if fields:
            row["_physical_conflicts"] = ",".join(fields)
        kept.append(row)
    return kept




def load(data_root: Path, name: str) -> list[dict[str, str]]:
    """旧签名包装（无 catalog、无模式过滤），保持既有调用方可用。"""
    return _load_rows(make_context(data_root), name)




def _manifest_tasks(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """兼容 manifest 的 ``tasks`` 列表与 ``experiments/cases/seeds`` 结构。"""
    tasks = payload.get("tasks")
    if isinstance(tasks, list):
        return [task for task in tasks if isinstance(task, dict)]
    out: list[dict[str, Any]] = []
    for experiment in payload.get("experiments") or []:
        if not isinstance(experiment, dict):
            continue
        exp_id = str(experiment.get("experiment") or experiment.get("experiment_id") or "")
        for case in experiment.get("cases") or []:
            if not isinstance(case, dict):
                continue
            case_id = str(case.get("case_id") or case.get("case") or "")
            mode = str(case.get("mode") or "")
            for seed in case.get("seeds") or []:
                out.append({"experiment": exp_id, "case_id": case_id,
                            "seed": seed, "mode": mode})
    return out




def _sample_keys(keys: set[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """把身份键集合渲染成最多 5 条可读样例（报告 missing 用）。"""
    samples: list[dict[str, Any]] = []
    for key in sorted(keys)[:5]:
        samples.append({
            "experiment": key[0] if len(key) > 0 else "",
            "case": key[1] if len(key) > 1 else "",
            "config_id": key[2] if len(key) > 2 else "",
            "seed": key[3] if len(key) > 3 else -1,
            "mode": key[4] if len(key) > 4 else "",
        })
    return samples


__all__ = [
    "_PHYSICAL_COLUMNS",
    "_SHARED_RESULT_COLUMNS",
    "_STATE_RESULT_COLUMNS",
    "_artifact_key",
    "_candidate_conflict_fields",
    "_catalog_module",
    "_dedupe_sources",
    "_enrichment_conflicts",
    "_enrichment_field_names",
    "_enrichment_signature",
    "_identity_text",
    "_load_rows",
    "_manifest_tasks",
    "_normalize_path",
    "_normalized_column",
    "_physical_index",
    "_read_csv_rows",
    "_resolve_physical_source",
    "_runs_csv_path",
    "_runs_csv_paths",
    "_sample_keys",
    "load",
    "make_context",
]
