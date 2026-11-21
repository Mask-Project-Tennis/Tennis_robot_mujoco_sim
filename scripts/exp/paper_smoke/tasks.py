"""resolved manifest 构建与任务规格展开。

包含任务日志路径、可搬移 resolved 副本、单任务 argv/身份构造、
mode 过滤，以及公开入口 ``resolve_manifest`` / ``build_specs_from_manifest``。
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.exp import batch_runner
from scripts.exp.batch_runner import (
    ExperimentSpec,
    RunIdentity,
    compute_config_hash,
    make_config_id,
)

from .identity import _config_hash, _runtime_effective_config, _stable_config_id_argv
from .manifest import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RUN_ROOT,
    MANIFEST_SCHEMA,
    REQUIRED_WORKERS,
    RESOLVED_SCHEMA,
    VALID_MODE_FILTERS,
    ManifestError,
    _display_path,
    _file_sha256,
    _format_override,
    _resolve_project_path,
    _validate_seeds,
    load_manifest,
    validate_manifest,
)

PORTABLE_TASK_PATH_FIELDS: tuple[str, ...] = (
    "artifact_dir",
    "log_path",
    "planned_artifact_dir",
    "planned_log_path",
)


def _task_log_path(
    run_root: Path, experiment: str, config_id: str, seed: int,
) -> Path:
    """构造任务的计划日志路径（与 batch 实际写日志共用同一函数，N01）。"""
    return batch_runner.task_log_path(
        Path(run_root) / "logs", experiment, config_id, seed)


def _run_root_relative(run_root: Path, value: str | Path) -> str:
    """把路径改写为相对 ``run_root`` 的 posix 字符串（越界则保留绝对路径）。

    与 ``_display_path`` 不同：后者相对 ``PROJECT``，无法在换 output 后由
    ``run_root`` 正确解析，故 manifest 内路径一律用本函数。

    Args:
        run_root: 本次 output 根目录。
        value: 已解析的绝对路径字符串（或相对路径，原样返回）。

    Returns:
        位于 run_root 内→相对 posix 路径；否则→绝对路径字符串。
    """
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(Path(run_root).resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _iter_resolved_tasks(resolved: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """遍历 resolved 的顶层 ``tasks`` 与各 unit 的 ``tasks``（去重）。

    两者通常指向同一批任务对象；用 ``id`` 去重避免同一路径被重复处理，
    也兼容未来只在一处保存任务的实现。

    Args:
        resolved: ``resolve_manifest`` 的结果。

    Yields:
        任务字典（可变，供就地改写路径）。
    """
    seen: set[int] = set()
    groups: list[Any] = [resolved.get("tasks") or []]
    groups.extend(unit.get("tasks") or [] for unit in resolved.get("units") or []
                  if isinstance(unit, Mapping))
    for group in groups:
        for task in group:
            if not isinstance(task, dict) or id(task) in seen:
                continue
            seen.add(id(task))
            yield task


def _portable_resolved(
    resolved: Mapping[str, Any], run_root: Path,
) -> dict[str, Any]:
    """生成可整树搬移的 ``manifest_resolved.json`` 载荷（M06）。

    落盘版本把 ``run_root`` 写成 ``"."``，``data_root`` 与任务的
    ``artifact_dir``/``log_path``/``planned_*`` 在 run_root 内改写成相对
    posix 路径（run_root 外保持绝对路径），因此整棵 output 目录移动后
    catalog/统计消费端仍能解析。内存中的 ``resolved`` 继续保留绝对路径，
    producer 执行任务与回填不受影响；``schema``/``task.argv``（真实绝对
    命令）按冻结约定不改写。

    Args:
        resolved: ``resolve_manifest``（可经 ``_apply_actual_paths`` 回填）。
        run_root: 本次 output 根目录（绝对路径）。

    Returns:
        深拷贝后的可序列化字典，可直接交给 :func:`_write_json`。
    """
    root = Path(run_root).resolve()
    portable = copy.deepcopy(dict(resolved))
    portable["run_root"] = "."
    portable["data_root"] = _run_root_relative(root, str(resolved["data_root"]))
    for task in _iter_resolved_tasks(portable):
        for field in PORTABLE_TASK_PATH_FIELDS:
            value = task.get(field)
            if value is None or not str(value).strip():
                continue
            task[field] = _run_root_relative(root, str(value))
    return portable


def _build_task(
    *,
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    case: Mapping[str, Any],
    seed: int,
    identity: RunIdentity,
    run_root: Path,
    data_root: Path,
    profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """构造一条 ready/adapted 任务（argv、command、身份、artifact 目录）。

    per-task ``config_hash`` 由有效配置（含 case 级 runtime/task 覆盖）计算，
    保证同一实验内 TCP 1.0/1.8 等变体在续跑时不会混身份。
    """
    from .cli import _substitute  # 延迟导入：避免 tasks ↔ cli 循环依赖
    overrides: dict[str, Any] = dict(profiles.get(case.get("tier") or "", {}))
    overrides.update(case.get("task_overrides", {}))
    runtime_rel, runtime_info = _runtime_effective_config(
        manifest, unit, case, run_root)
    argv: dict[str, Any] = dict(manifest.get("base_argv", {}))
    argv.update(case.get("argv") or {})
    argv.update({
        "--robot-config": str(manifest["configs"]["robot"]),
        "--runtime-config": runtime_rel,
        "--racket-config": str(manifest["configs"]["racket"]),
    })
    artifact_dir = run_root / "artifacts" / str(unit["id"]) / str(case["id"]) / f"seed{seed}"
    argv["--output"] = str(artifact_dir)
    argv["--seed"] = seed
    if case["mode"] == "position":
        argv["--position-mode"] = None
    if overrides:
        argv["--task-overrides"] = [
            f"{key}={_format_override(value)}" for key, value in sorted(overrides.items())]
    # config_id 用稳定 token（不含 run_root 快照路径），保证换 --output 后仍可续跑（N15）。
    config_id = make_config_id(_stable_config_id_argv(argv, runtime_info))
    plan_tokens = {
        "{experiment_id}": str(unit["id"]),
        "{case_id}": str(case["id"]),
        "{seed}": str(seed),
        # 混合实验（同一 case 两种执行模式）必须显式传 mode，
        # 否则 extractor 无法区分样本池；plot 也按 mode 分层读写。
        "{mode}": str(case["mode"]),
    }
    extract_argv = _substitute(
        case.get("extract_argv") or [], run_root, data_root, extra=plan_tokens)
    plot_argv = _substitute(
        case.get("plot_argv") or [], run_root, data_root, extra=plan_tokens)
    metric_schema = str(manifest["identity_defaults"]["metric_schema"])
    runtime_for_hash = (runtime_info["snapshot"]
                        if runtime_info["snapshot"] else str(runtime_info["base"]))
    task_hash = compute_config_hash(
        {
            "robot": str(manifest["configs"]["robot"]),
            "runtime": str(runtime_for_hash),
            "racket": str(manifest["configs"]["racket"]),
            "task": str(manifest["configs"]["task_overlay"]),
        },
        extra={
            "manifest_schema": MANIFEST_SCHEMA,
            "metric_schema": metric_schema,
            "runtime_overrides": dict(runtime_info["overrides"]),
            "task_overrides": overrides,
        },
    )
    task_identity = RunIdentity(
        robot_id=str(identity.robot_id),
        mode=str(case["mode"]),
        racket_id=str(identity.racket_id),
        config_hash=task_hash,
        metric_schema=metric_schema,
    )
    return {
        "experiment": str(unit["id"]),
        "case_id": str(case["id"]),
        "seed": seed,
        "mode": case["mode"],
        "case_status": case["status"],
        "tier": case.get("tier"),
        "config_id": config_id,
        "config_hash": task_hash,
        "runtime": runtime_info,
        "identity": task_identity.as_dict(),
        # N14：每个任务携带可追溯的论文计划字段（T8b/T8c 直接消费）。
        "experiment_id": str(case.get("experiment_id", unit["id"])),
        "source_paper_item": str(case.get("source_paper_item", "")),
        "extract_argv": extract_argv,
        "plot_argv": plot_argv,
        "expected_outputs": [str(x) for x in case.get("expected_outputs") or []],
        "timeout_s": float(case.get("timeout_s") or manifest.get("_timeout_s_default", 900.0)),
        "argv": argv,
        "command": batch_runner.build_command(str(manifest["script"]), argv),
        # 计划路径（绝对）：`artifact_dir`/`log_path` 可能在续跑回填后被替换为
        # 旧 output 的真实路径，`planned_*` 始终保留本次 output 的计划位置（N01）。
        "artifact_dir": str(artifact_dir),
        "log_path": str(_task_log_path(run_root, str(unit["id"]), config_id, seed)),
        "planned_artifact_dir": str(artifact_dir),
        "planned_log_path": str(
            _task_log_path(run_root, str(unit["id"]), config_id, seed)),
    }


def _filter_tasks_by_mode(
    tasks: list[dict[str, Any]], mode_filter: str) -> list[dict[str, Any]]:
    """按 --mode 过滤任务（torque/position/both）。"""
    if mode_filter == "both":
        return tasks
    return [task for task in tasks if task["mode"] == mode_filter]


def resolve_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
    data_root: str | Path | None = None,
    output: str | Path | None = None,
    seeds: Sequence[int] | None = None,
    mode: str = "torque",
    limit: int | None = None,
    timeout_s: float | None = None,
    allow_single_seed: bool = False,
) -> dict[str, Any]:
    """把 manifest 解析为本次运行的完整任务清单（不执行子进程）。

    Args:
        manifest: 已加载的 manifest 字典。
        manifest_path: 原始 manifest 路径（写入 resolved JSON 的 sha256）。
        data_root: batch CSV 数据根（默认 ``<run_root>/batch``）。
        output: 本次运行根目录（默认 manifest.run_root_default）。
        seeds: 覆盖 manifest seeds（仍受小样本规则约束，≥2）。
        mode: torque/position/both 过滤。
        limit: 只保留前 N 条任务（跨实验统一截断）。
        timeout_s: 覆盖 manifest 默认单 run 超时（秒）。
        allow_single_seed: 允许 --limit 截断出单 seed cell（仅调试，
            默认 False 时任何 <2 seed 的比较 cell 都会被拒绝）。

    Returns:
        可直接 JSON 序列化的 resolved 字典（含 units / tasks / blocked / counts）。

    Raises:
        ManifestError: manifest 不合法、配置缺失、或工具有未声明引用。
    """
    manifest = validate_manifest(manifest)
    if mode not in VALID_MODE_FILTERS:
        raise ManifestError(f"mode 必须是 {VALID_MODE_FILTERS}，得到 {mode!r}")
    manifest_file = _resolve_project_path(
        manifest_path or DEFAULT_MANIFEST_PATH)
    run_root = _resolve_project_path(output) if output is not None else _resolve_project_path(
        manifest.get("run_root_default", str(DEFAULT_RUN_ROOT)))
    batch_root = (_resolve_project_path(data_root) if data_root is not None
                  else run_root / "batch")

    manifest_seeds = _validate_seeds(seeds, "--seeds") if seeds is not None else list(
        _validate_seeds(manifest.get("seeds"), "manifest"))
    config_hash, config_details = _config_hash(manifest)
    identity = RunIdentity(
        robot_id=str(manifest["identity_defaults"]["robot_id"]),
        mode="",
        racket_id=str(manifest["identity_defaults"]["racket_id"]),
        config_hash=config_hash,
        metric_schema=str(manifest["identity_defaults"]["metric_schema"]),
    )
    profiles: dict[str, dict[str, Any]] = {
        str(name): dict(overrides or {})
        for name, overrides in manifest.get("ablation_profiles", {}).items()}
    timeout_default = float(
        timeout_s if timeout_s is not None
        else manifest.get("timeout_s_default", 900.0))
    manifest_for_tasks = dict(manifest)
    manifest_for_tasks["_timeout_s_default"] = timeout_default

    units: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []
    experiments = manifest.get("experiments", [])
    dependencies = manifest.get("extra_dependencies", [])
    for unit in list(experiments) + list(dependencies):
        unit_id = str(unit["id"])
        kind = str(unit.get("kind", "batch"))
        entry: dict[str, Any] = {
            "id": unit_id,
            "kind": kind,
            "title": str(unit.get("title", "")),
            "paper_ref": unit.get("paper_ref", {}) or {},
            "matrix_ref": str(unit.get("matrix_ref", "")),
            "depends_on": [str(x) for x in unit.get("depends_on", [])],
            "expected_fields": [str(x) for x in unit.get("expected_fields", [])],
            "t8b_consumer": str(unit.get("t8b_consumer", "")),
            "notes": str(unit.get("notes", "")),
            "status": "blocked" if str(unit.get("status", "")) == "blocked" else "ready",
            "cases": [],
            "task_count": 0,
        }
        if kind == "derived":
            entry["derived_from"] = str(unit.get("derived_from", ""))
            entry["status"] = "derived"
            units.append(entry)
            continue
        if kind == "tool":
            entry["expected_outputs"] = [str(x) for x in unit.get("expected_outputs", [])]
            entry["command"] = [str(tok) for tok in unit.get("command", [])]
            if entry["status"] == "blocked":
                entry["blocked_reason"] = str(unit.get("blocked_reason", ""))
            units.append(entry)
            continue
        unit_tasks: list[dict[str, Any]] = []
        active_case_count = 0
        for case in unit.get("cases", []):
            blocked_case = case["status"] == "blocked"
            case_entry = {
                "id": str(case["id"]),
                "mode": str(case["mode"]),
                "tier": case.get("tier"),
                "status": str(case["status"]),
                "runtime_overrides": dict(case.get("runtime_overrides") or {}),
                "seeds": list(case["seeds"]),
                # N14：blocked case 也显式给出计划字段（不适用处为 null）。
                "experiment_id": case.get("experiment_id") or str(unit_id),
                "source_paper_item": case.get("source_paper_item"),
                "argv": (None if blocked_case
                         else dict(case.get("argv") or {})),
                "extract_argv": (None if blocked_case
                                 else list(case.get("extract_argv") or [])),
                "plot_argv": (None if blocked_case
                              else list(case.get("plot_argv") or [])),
                "expected_outputs": (None if blocked_case
                                     else [str(x) for x in case.get("expected_outputs") or []]),
                "seed": (None if blocked_case else list(case["seeds"])),
                "timeout_s": (None if blocked_case else case.get("timeout_s")),
            }
            entry["cases"].append(case_entry)
            if case["status"] == "blocked":
                blocked.append({
                    "unit": unit_id,
                    "case_id": str(case["id"]),
                    "reason": str(case.get("blocked_reason", "")),
                })
                case_entry["blocked_reason"] = str(case.get("blocked_reason", ""))
                continue
            active_case_count += 1
            case_seeds = (list(manifest_seeds) if seeds is not None
                          else list(case["seeds"]))
            for seed in case_seeds:
                unit_tasks.append(_build_task(
                    manifest=manifest_for_tasks, unit=unit, case=case, seed=seed,
                    identity=identity, run_root=run_root, data_root=batch_root,
                    profiles=profiles))
        unit_tasks = _filter_tasks_by_mode(unit_tasks, mode)
        entry["task_count"] = len(unit_tasks)
        if unit_tasks:
            entry["status"] = "ready"
        elif active_case_count == 0:
            entry["status"] = "blocked"
        else:
            # 有可执行 case 但被 --mode/--limit 过滤掉：显式标记，避免误读为 blocked
            entry["status"] = "filtered"
        entry["tasks"] = unit_tasks
        tasks.extend(unit_tasks)
        units.append(entry)

    if limit is not None:
        if limit < 0:
            raise ManifestError(f"limit 不能为负: {limit}")
        tasks = tasks[:limit]
        kept: dict[str, int] = {}
        for task in tasks:
            kept[task["experiment"]] = kept.get(task["experiment"], 0) + 1
        for unit in units:
            unit["tasks"] = [t for t in unit.get("tasks", [])
                             if t["experiment"] in kept][: kept.get(unit["id"], 0)]
            unit["task_count"] = len(unit["tasks"])
        if not allow_single_seed:
            by_cell: dict[tuple[str, str], set[int]] = {}
            for task in tasks:
                by_cell.setdefault(
                    (str(task["experiment"]), str(task["case_id"])), set()).add(
                        int(task["seed"]))
            short = [
                f"{unit_id}/{case_id}: {sorted(seeds_in_cell)}"
                for (unit_id, case_id), seeds_in_cell in sorted(by_cell.items())
                if len(seeds_in_cell) < 2]
            if short:
                raise ManifestError(
                    "--limit 截断后出现单 seed 比较 cell（小样本规则要求每 cell ≥2 "
                    f"共同 seed）: {'; '.join(short)}。请增大 --limit；仅在调试时"
                    "显式加 --allow-single-seed。")

    ready_tasks = [t for t in tasks]
    resolved = {
        "schema": RESOLVED_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_manifest": {
            "path": _display_path(manifest_file),
            "sha256": _file_sha256(manifest_file) if manifest_file.is_file() else "",
        },
        "script": str(manifest["script"]),
        "run_root": str(run_root),
        "data_root": str(batch_root),
        "configs": config_details,
        "config_hash": config_hash,
        "identity": identity.as_dict(),
        "seeds": manifest_seeds,
        "timeout_s_default": timeout_default,
        "mode_filter": mode,
        "limit": limit,
        "allow_single_seed": bool(allow_single_seed),
        "units": units,
        "tasks": ready_tasks,
        "blocked": blocked,
        "counts": {
            "experiments": len(experiments),
            "dependencies": len(dependencies),
            "batch_tasks": len(ready_tasks),
            "torque_tasks": sum(1 for t in ready_tasks if t["mode"] == "torque"),
            "position_tasks": sum(1 for t in ready_tasks if t["mode"] == "position"),
            "blocked_cases": len(blocked),
            "derived_units": sum(1 for u in units if u["kind"] == "derived"),
            "tool_units": sum(1 for u in units if u["kind"] == "tool"),
        },
    }
    return resolved


def build_specs_from_manifest(
    manifest_path: str | Path | None = None,
    *,
    data_root: str | Path | None = None,
    output: str | Path | None = None,
    seeds: Sequence[int] | None = None,
    mode: str = "torque",
    limit: int | None = None,
    timeout_s: float | None = None,
    workers: int = REQUIRED_WORKERS,
    allow_single_seed: bool = False,
) -> list[ExperimentSpec]:
    """构建 R1D 论文 smoke 的 ExperimentSpec 列表（供 overnight 工厂复用）。

    每个 manifest 实验（及 batch 型额外依赖）生成一个 spec，grid 为解析后的
    任务参数；``_timeout`` meta 写入任务级超时。R1D 强制 workers=1。

    Args:
        manifest_path: manifest 路径（默认 configs/experiments/r1d_paper_smoke.yaml）。
        data_root: batch CSV 数据根（默认 ``<output>/batch``）。
        output: 本次运行根目录（默认 manifest.run_root_default）。
        seeds: 覆盖 manifest seeds。
        mode: torque/position/both。
        limit: 只保留前 N 条任务。
        timeout_s: 覆盖默认超时。
        workers: 预期并行度（R1D 必须为 1）。
        allow_single_seed: 允许 --limit 截断出单 seed cell（仅调试）。

    Returns:
        ExperimentSpec 列表（name=实验/依赖 id；identity 携带 config_hash）。

    Raises:
        ManifestError: manifest 不合法或配置缺失。
        ValueError: workers != 1。
    """
    if workers != REQUIRED_WORKERS:
        raise ValueError(
            f"R1D 论文 smoke 要求 workers={REQUIRED_WORKERS}（串行，避免多进程 MuJoCo 问题），"
            f"得到 workers={workers}")
    manifest = load_manifest(manifest_path)
    resolved = resolve_manifest(
        manifest, manifest_path=manifest_path, data_root=data_root, output=output,
        seeds=seeds, mode=mode, limit=limit, timeout_s=timeout_s,
        allow_single_seed=allow_single_seed)
    identity = RunIdentity(
        robot_id=str(resolved["identity"]["robot_id"]),
        mode="",
        racket_id=str(resolved["identity"]["racket_id"]),
        config_hash=str(resolved["identity"]["config_hash"]),
        metric_schema=str(resolved["identity"]["metric_schema"]),
    )
    specs: list[ExperimentSpec] = []
    for unit in resolved["units"]:
        if unit["kind"] != "batch" or not unit.get("tasks"):
            continue
        grid: list[dict[str, Any]] = []
        for task in unit["tasks"]:
            params = dict(task["argv"])
            params["_timeout"] = task["timeout_s"]
            # per-task 有效配置 hash（含 runtime/task 覆盖）：batch_runner 用它
            # 覆盖身份，保证 TCP 1.0/1.8 或 ablation 档不会混续跑。
            params["_config_hash"] = str(task["config_hash"])
            # 保留 manifest 已算好的稳定 config_id（F05）：batch 不得按含绝对
            # --output/快照路径的 argv 重算，否则换 output 后 resume 失效。
            params["_config_id"] = str(task["config_id"])
            grid.append(params)
        specs.append(ExperimentSpec(
            name=str(unit["id"]),
            script=str(resolved["script"]),
            grid=grid,
            report_ref=f"paper_smoke:{unit['id']}",
            identity=identity,
        ))
    return specs
