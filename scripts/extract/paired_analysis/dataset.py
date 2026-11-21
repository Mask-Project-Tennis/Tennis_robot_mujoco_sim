"""数据集校验与执行尝试收敛：final attempt、重复记录、mode 隔离。

同一身份允许多条失败尝试 + 至多一条完成尝试；统计只计最终尝试，但 CSV 原文件
不被改写。模式（torque/position）必须单一，混合数据要求显式模式选择。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .identity import _seed_value, case_of, mode_of, record_identity
from .sources import (
    _manifest_tasks,
    _physical_index,
    _read_csv_rows,
    _resolve_physical_source,
    _runs_csv_path,
    _sample_keys,
)
from .types import (
    VALID_MODES,
    DatasetConflictError,
    DuplicateRecordError,
    IdentityError,
    ModeSelectionError,
    StatsContext,
    _to_bool_or_none,
)


def _is_completed_row(row: Mapping[str, Any]) -> bool:
    """该行是否代表一次已完成的执行尝试。

    续跑语义（P2）：batch_runner 失败后允许 ``--resume``，原失败行保留、
    成功行追加，两行共享同一 ``RecordIdentity.unique_key()``。因此同一身份
    允许保留多条失败尝试（``error`` 非空或 ``timed_out``），但完成尝试至多
    一条。``timed_out`` 为真时即使 ``error`` 为空也不算完成（生产超时终止）。

    Args:
        row: results.csv / runs.csv 的一行。

    Returns:
        True 表示该行是一次完成尝试。
    """
    if str(row.get("error") or "").strip():
        return False
    return _to_bool_or_none(row.get("timed_out")) is not True




def _group_attempts(
    rows: list[dict[str, str]], path: Path
) -> dict[tuple[Any, ...], list[tuple[int, dict[str, str]]]]:
    """按单一来源内的完整身份归并执行尝试。

    Args:
        rows: 同一 CSV 的全部数据行。
        path: 行所属文件（参与身份构造，与既有查重口径一致）。

    Returns:
        身份键 → ``[(行号, 行)]``；行号从 2 起，与 CSV 表头占第 1 行对齐。
    """
    groups: dict[tuple[Any, ...], list[tuple[int, dict[str, str]]]] = {}
    for line, row in enumerate(rows, start=2):
        key = record_identity(row, source=str(path)).unique_key()
        groups.setdefault(key, []).append((line, row))
    return groups




def _check_attempts(rows: list[dict[str, str]], path: Path) -> None:
    """校验执行尝试契约：同一身份至多一条完成尝试，失败尝试可多条。

    Args:
        rows: 同一 CSV 的全部数据行。
        path: 行所属文件（错误信息用）。

    Raises:
        DuplicateRecordError: 同一身份出现两条及以上完成尝试。续跑恢复只允许
            一条完成行；两条完成行仍是真正的重复记录，按既有契约拒绝。
    """
    for key, attempts in _group_attempts(rows, path).items():
        completed = [line for line, row in attempts if _is_completed_row(row)]
        if len(completed) > 1:
            raise DuplicateRecordError(
                f"{path.name} 内出现同身份重复完成记录（第 {completed[0]} 行与"
                f"第 {completed[1]} 行）：{key}；同一身份允许多条失败尝试但至多"
                "一条完成尝试，按契约禁止静默去重，请先修正数据或拆分数据根"
            )




def _final_attempt_rows(
    rows: list[dict[str, str]], path: Path
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """为每个身份选择最终执行尝试行，作为统计样本。

    选择规则：有完成尝试则取该完成行；没有完成尝试则保留**最后一条失败行**
    （供 exclusion 统计与“无对照”报告使用）。其余早期尝试行只从统计视图剔除，
    仍保留在 CSV 原文件中，并在此处以 ``ignored`` 摘要返回，禁止静默丢弃文件行。

    Args:
        rows: 同一 CSV 的全部数据行。
        path: 行所属文件（身份构造与错误信息用）。

    Returns:
        ``(selected, ignored)``：``selected`` 保持文件行序；``ignored`` 每项含
        ``line`` / ``experiment`` / ``case`` / ``config_id`` / ``seed`` /
        ``mode`` / ``reason``，按行号升序。

    Raises:
        DuplicateRecordError: 同一身份出现两条及以上完成尝试（经
            :func:`_check_attempts` 判定）。
    """
    _check_attempts(rows, path)
    selected_lines: set[int] = set()
    ignored: list[dict[str, Any]] = []
    for attempts in _group_attempts(rows, path).values():
        completed = [line for line, row in attempts if _is_completed_row(row)]
        chosen_line = completed[0] if completed else attempts[-1][0]
        selected_lines.add(chosen_line)
        for line, row in attempts:
            if line == chosen_line:
                continue
            ignored.append({
                "line": line,
                "experiment": (row.get("experiment") or "").strip(),
                "case": case_of(row),
                "config_id": (row.get("config_id") or "").strip(),
                "seed": _seed_value(row),
                "mode": mode_of(row) or "",
                "reason": (
                    "superseded_completed_attempt"
                    if _is_completed_row(row)
                    else "superseded_failed_attempt"
                ),
            })
    selected = [row for line, row in enumerate(rows, start=2) if line in selected_lines]
    ignored.sort(key=lambda item: int(item["line"]))
    return selected, ignored




def _validate_dataset(ctx: StatsContext) -> dict[str, Any]:
    """校验唯一性/身份/两视图一对一连接，返回 validation 摘要。

    两视图（results.csv 基础字段、runs.csv 富化字段）**不纵向拼接**：各自先按
    执行尝试收敛（同一身份多条失败尝试 + 至多一条完成尝试，见
    :func:`_final_attempt_rows`），再按完整身份一对一连接；缺富化行只报告，
    不伪造 physical 判定。

    计数口径：``results_records`` / ``runs_records`` 是文件行数（含被忽略的早期
    尝试），``mode_records`` 是最终尝试样本数；被忽略的文件行数与样例见
    ``attempts_ignored`` / ``attempts_ignored_samples``，不静默丢弃。
    """
    results_paths = sorted(ctx.data_root.glob("*/results.csv"))
    results_records = 0
    results_rows: list[dict[str, str]] = []
    ignored_attempts: list[dict[str, Any]] = []
    for path in results_paths:
        rows = _read_csv_rows(path)
        selected, ignored = _final_attempt_rows(rows, path)
        results_records += len(rows)
        results_rows.extend(selected)
        ignored_attempts.extend({**item, "view": "results"} for item in ignored)

    runs_path = _runs_csv_path(ctx)
    runs_records = 0
    runs_rows: list[dict[str, str]] = []
    if runs_path is not None:
        rows = _read_csv_rows(runs_path)
        runs_records = len(rows)
        runs_rows, ignored = _final_attempt_rows(rows, runs_path)
        ignored_attempts.extend({**item, "view": "runs"} for item in ignored)

    robot_ids = Counter(
        (row.get("robot_id") or "").strip()
        for row in results_rows
        if (row.get("robot_id") or "").strip()
    )
    if ctx.catalog is not None and robot_ids and set(robot_ids) != {ctx.catalog.robot_id}:
        raise IdentityError(
            f"robot_id 与 catalog 声明不一致：数据 {sorted(robot_ids)} vs "
            f"catalog {ctx.catalog.robot_id!r}；禁止跨机器人配对"
        )
    rackets: dict[str, set[str]] = defaultdict(set)
    for row in results_rows:
        racket = (row.get("racket_id") or "").strip()
        if racket:
            rackets[mode_of(row) or "unknown"].add(racket)
    for mode, values in rackets.items():
        if len(values) > 1:
            raise IdentityError(
                f"mode={mode} 内出现多个 racket_id {sorted(values)}："
                "不同球拍/来源不得配对，请拆分数据根或显式指定身份"
            )

    # 校验与 CLI 拒绝都以“当前执行模式”为作用域：position 的冲突不能让
    # torque 统计失败（反之亦然）；--mode both 时每个模式各自看自己的记录。
    def in_scope(row: dict[str, str]) -> bool:
        return ctx.mode is None or mode_of(row) == ctx.mode

    scoped_results = [row for row in results_rows if in_scope(row)]
    scoped_runs = [row for row in runs_rows if in_scope(row)]

    results_keys = {record_identity(row).join_key() for row in scoped_results}
    runs_keys = {record_identity(row).join_key() for row in scoped_runs}
    missing_enrichment = results_keys - runs_keys
    runs_only = runs_keys - results_keys

    # M03：富化必须按完整身份命中；同路径异身份/异设计只报告，绝不静默回填
    physical_index = _physical_index(ctx)
    enrichment_status: Counter[str] = Counter()
    physical_conflicts: list[dict[str, Any]] = []
    for row in scoped_results:
        try:
            _, status, fields = _resolve_physical_source(row, physical_index, ctx)
        except DatasetConflictError as exc:
            status, fields = "conflict", [str(exc)]
        enrichment_status[status] += 1
        if status == "conflict":
            physical_conflicts.append({
                "experiment": (row.get("experiment") or "").strip(),
                "case": case_of(row),
                "config_id": (row.get("config_id") or "").strip(),
                "seed": _seed_value(row),
                "mode": mode_of(row) or "",
                "fields": fields,
            })

    manifest_path = ctx.manifest_path or (ctx.run_root / "manifest_resolved.json")
    expected: set[tuple[str, str, int, str]] = set()
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        for task in _manifest_tasks(payload):
            task_mode = str(task.get("mode") or "")
            if ctx.mode is not None and task_mode != ctx.mode:
                continue
            expected.add((
                str(task.get("experiment") or ""),
                str(task.get("case_id") or ""),
                _seed_value({"seed": task.get("seed")}),
                task_mode,
            ))
    actual = {(key[0], key[1], key[3], key[4]) for key in runs_keys}
    manifest_missing = expected - actual

    mode_counts: Counter[str] = Counter()
    for row in results_rows:
        mode_counts[mode_of(row) or "unknown"] += 1
    mode_records = (
        sum(1 for row in results_rows if mode_of(row) == ctx.mode)
        if ctx.mode is not None else len(results_rows)
    )
    return {
        "mode": ctx.mode,
        "modes_found": dict(mode_counts),
        "mode_records": mode_records,
        "results_files": [str(path) for path in results_paths],
        "runs_csv": str(runs_path) if runs_path is not None else None,
        "results_records": results_records,
        "runs_records": runs_records,
        "attempts_ignored": len(ignored_attempts),
        "attempts_ignored_samples": ignored_attempts[:5],
        "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
        "manifest_expected": len(expected),
        "manifest_missing": len(manifest_missing),
        "manifest_missing_samples": _sample_keys(manifest_missing),
        "missing_enrichment": len(missing_enrichment),
        "missing_enrichment_samples": _sample_keys(missing_enrichment),
        "runs_only": len(runs_only),
        "runs_only_samples": _sample_keys(runs_only),
        "enrichment_status": dict(enrichment_status),
        "physical_conflicts": len(physical_conflicts),
        "physical_conflict_samples": physical_conflicts[:5],
        "robot_ids": sorted(robot_ids),
        "mode_rackets": {mode: sorted(values) for mode, values in rackets.items()},
        "synthetic": bool(ctx.synthetic),
    }




def _discover_modes(ctx: StatsContext) -> dict[str, int]:
    """扫描数据根内的执行模式分布（results.csv 优先，空时退回 runs.csv）。"""
    counts: Counter[str] = Counter()
    for path in sorted(ctx.data_root.glob("*/results.csv")):
        for row in _read_csv_rows(path):
            counts[mode_of(row) or "unknown"] += 1
    if not counts:
        runs_path = _runs_csv_path(ctx)
        if runs_path is not None:
            for row in _read_csv_rows(runs_path):
                counts[mode_of(row) or "unknown"] += 1
    return dict(counts)




def resolve_mode(ctx: StatsContext) -> str | None:
    """确定本次统计的执行模式；混合模式必须显式选择（F04）。

    Args:
        ctx: 统计上下文（``ctx.mode`` 为显式选择或 None）。

    Returns:
        解析后的模式（``torque`` / ``position`` / None）。

    Raises:
        ModeSelectionError: 数据含多种模式或存在无法判定的模式。
    """
    if ctx.mode is not None:
        return ctx.mode
    counts = _discover_modes(ctx)
    known = {mode: counts[mode] for mode in VALID_MODES if counts.get(mode)}
    if len(known) > 1:
        detail = "、".join(f"{mode}={count}" for mode, count in sorted(known.items()))
        raise ModeSelectionError(
            f"检测到混合执行模式（{detail}）：必须显式选择 --mode torque / "
            "--mode position / --mode both，禁止把两种模式合成一个样本池"
        )
    if len(known) == 1:
        return next(iter(known))
    if counts.get("unknown"):
        raise ModeSelectionError(
            f"有 {counts['unknown']} 条记录无法确定执行模式（缺 mode 列，且无 "
            "--position-mode/--actuator-mode 证据）：请显式 --mode torque|position"
        )
    return None


__all__ = [
    "_check_attempts",
    "_discover_modes",
    "_final_attempt_rows",
    "_group_attempts",
    "_is_completed_row",
    "_validate_dataset",
    "resolve_mode",
]
