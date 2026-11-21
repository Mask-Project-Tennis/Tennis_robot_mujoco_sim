"""CLI 入口与工具依赖执行：argv 模板替换、工具输出检查、
``main`` 参数解析/执行/退出码。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.exp import batch_runner

from .catalog import _write_catalog
from .manifest import (
    DEFAULT_MANIFEST_PATH,
    PROJECT,
    REQUIRED_WORKERS,
    VALID_MODE_FILTERS,
    ManifestError,
    _display_path,
    load_manifest,
)
from .resume import (
    _apply_actual_paths,
    _collect_runs,
    _print_artifact_problems,
    _validate_completed_artifacts,
)
from .tasks import _portable_resolved, build_specs_from_manifest, resolve_manifest


def _patched(name: str, default: Any) -> Any:
    """优先取 facade 模块上被测试替换的同名实现。

    历史测试通过 ``monkeypatch.setattr(run_r1d_paper_smoke, "_collect_runs", ...)``
    注入假实现；迁移后 ``main`` 实际位于本模块，需要在调用点动态读取
    facade 属性，否则 patch 失效。

    Args:
        name: 需要查找的属性名。
        default: 本模块内的默认实现。

    Returns:
        facade 上存在的替换实现；未替换时返回 ``default``。
    """
    facade = sys.modules.get("scripts.exp.run_r1d_paper_smoke")
    if facade is not None:
        candidate = getattr(facade, name, None)
        if candidate is not None and candidate is not default:
            return candidate
    return default


def _substitute(
    tokens: Sequence[str],
    run_root: Path,
    data_root: Path,
    extra: Mapping[str, str] | None = None,
) -> list[str]:
    """替换工具依赖/计划命令模板中的占位符。

    Args:
        tokens: 命令 token 列表（str/Path 等可转字符串值）。
        run_root: 本次运行根目录，替换 ``{run_root}``。
        data_root: batch 数据根，替换 ``{data_root}``。
        extra: 额外占位符映射（如 ``{experiment_id}``/``{case_id}``/``{seed}``）。

    Returns:
        替换后的字符串列表（顺序与输入一致）。
    """
    mapping = {
        "{python}": sys.executable,
        "{run_root}": str(run_root),
        "{data_root}": str(data_root),
        "{project}": str(PROJECT),
        # D 线（提取/绘图）消费端通过 --catalog 解析路径，不再猜父目录（F03）。
        "{catalog}": str(Path(run_root) / "catalog.json"),
    }
    if extra:
        mapping.update({str(key): str(value) for key, value in extra.items()})
    out: list[str] = []
    for token in tokens:
        value = str(token)
        for key, replacement in mapping.items():
            value = value.replace(key, replacement)
        out.append(value)
    return out


def _check_tool_outputs(
    patterns: Sequence[str], run_root: Path,
) -> list[str]:
    """校验工具依赖声明的 expected_outputs 是否真的生成。

    支持目录、文件与 ``*``/``?``/``[`` 通配；相对路径按 run_root 解析。
    返回缺失的模式列表（空列表表示全部存在）。
    """
    missing: list[str] = []
    for pattern in patterns:
        rel = str(pattern)
        target = Path(rel)
        if not target.is_absolute():
            target = run_root / rel
        if any(ch in rel for ch in "*?["):
            parent = target.parent
            match = None
            if parent.is_dir():
                match = next(parent.glob(target.name), None)
            if match is None:
                missing.append(rel)
        elif not target.exists():
            missing.append(rel)
    return missing


def _run_tool_dependency(
    unit: Mapping[str, Any], run_root: Path, data_root: Path,
) -> dict[str, Any]:
    """执行单个 ready 的工具型额外依赖，返回状态摘要。

    退出码 0 之外还校验 ``expected_outputs``：工具脚本可能“成功退出但
    没产出”（如 collect_fig_assets 跳过某目标），必须显式失败（T8a 评审 N5）。
    """
    unit_id = str(unit["id"])
    if unit.get("status") == "blocked":
        return {"id": unit_id, "status": "blocked",
                "blocked_reason": str(unit.get("blocked_reason", ""))}
    command = _substitute(unit.get("command", []), run_root, data_root)
    log_dir = run_root / "logs" / "_dependencies"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{unit_id}.log"
    timeout = float(unit.get("timeout_s", 3600.0))
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT), encoding="utf-8", errors="replace",
            check=False)
        text = "# command: " + " ".join(command) + "\n\n" + result.stdout + result.stderr
        log_path.write_text(text, encoding="utf-8", errors="replace")
        status = "ok" if result.returncode == 0 else "error"
        payload: dict[str, Any] = {
            "id": unit_id,
            "status": status,
            "returncode": result.returncode,
            "command": command,
            "log_path": _display_path(log_path),
            "elapsed_s": round(time.perf_counter() - started, 2),
        }
        if status == "ok":
            missing = _check_tool_outputs(
                [str(p) for p in unit.get("expected_outputs", [])], run_root)
            if missing:
                payload["status"] = "invalid_output"
                payload["missing_outputs"] = missing
        return payload
    except subprocess.TimeoutExpired:
        log_path.write_text(
            "# command: " + " ".join(command) + f"\n\nTIMEOUT after {timeout}s\n",
            encoding="utf-8")
        return {
            "id": unit_id,
            "status": "timeout",
            "returncode": -1,
            "command": command,
            "log_path": _display_path(log_path),
            "elapsed_s": round(time.perf_counter() - started, 2),
        }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """写契约 JSON（UTF-8、缩进 2、不转义中文）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_resolved_summary(resolved: Mapping[str, Any], *, dry_run: bool) -> None:
    """打印解析结果摘要（dry-run 时打印完整任务命令）。"""
    counts = resolved["counts"]
    print(f"[r1d-smoke] manifest={resolved['source_manifest']['path']} "
          f"manifest_sha256={resolved['source_manifest']['sha256'][:12]}…")
    print(f"[r1d-smoke] run_root={_display_path(resolved['run_root'])} "
          f"data_root={_display_path(resolved['data_root'])}")
    print(f"[r1d-smoke] config_hash={resolved['config_hash']} "
          f"identity={resolved['identity']}")
    print(f"[r1d-smoke] seeds={resolved['seeds']} mode={resolved['mode_filter']} "
          f"timeout_s={resolved['timeout_s_default']:g}")
    print(f"[r1d-smoke] 实验 {counts['experiments']} 个 + 额外依赖 "
          f"{counts['dependencies']} 个；batch 任务 {counts['batch_tasks']} 条"
          f"（torque {counts['torque_tasks']} / position {counts['position_tasks']}）；"
          f"blocked case {counts['blocked_cases']} 个；derived {counts['derived_units']}；"
          f"tool {counts['tool_units']}")
    for unit in resolved["units"]:
        kind = unit["kind"]
        if kind == "batch":
            print(f"  [{unit['status']:8s}] {unit['id']:34s} "
                  f"cases={len(unit['cases']):2d} tasks={unit['task_count']:3d}")
        elif kind == "derived":
            print(f"  [derived ] {unit['id']:34s} derived_from={unit.get('derived_from', '')}")
        else:
            print(f"  [{unit['status']:8s}] {unit['id']:34s} tool")
    for blocked in resolved["blocked"]:
        print(f"  [blocked] {blocked['unit']}/{blocked['case_id']}: {blocked['reason']}")
    if dry_run:
        print("[r1d-smoke] case 计划字段（N14：每 case 可核对 experiment_id/"
              "source_paper_item/argv/extract_argv/plot_argv/expected_outputs/"
              "mode/seed/timeout_s；blocked case 显式 null）：")

        def _show(value: Any) -> str:
            """把 case 计划字段渲染为单行 JSON（None 显式输出 null）。"""
            return "null" if value is None else json.dumps(
                value, ensure_ascii=False, sort_keys=True)

        for unit in resolved["units"]:
            if unit["kind"] != "batch":
                continue
            for case in unit["cases"]:
                print(f"  CASE {unit['id']}/{case['id']}: "
                      f"status={case['status']} "
                      f"experiment_id={case.get('experiment_id')} "
                      f"source_paper_item={case.get('source_paper_item')} "
                      f"mode={case['mode']} seed={_show(case.get('seed'))} "
                      f"timeout_s={case.get('timeout_s')} "
                      f"argv={_show(case.get('argv'))} "
                      f"expected_outputs={_show(case.get('expected_outputs'))} "
                      f"extract_argv={_show(case.get('extract_argv'))} "
                      f"plot_argv={_show(case.get('plot_argv'))}")
        for task in resolved["tasks"]:
            print(f"  DRY-RUN [{task['experiment']}/{task['case_id']}] "
                  f"seed={task['seed']} mode={task['mode']}: "
                  + " ".join(str(x) for x in task["command"]))
            print("           extract_argv: "
                  + " ".join(str(x) for x in task["extract_argv"]))
            print("           plot_argv: "
                  + " ".join(str(x) for x in task["plot_argv"]))
        for unit in resolved["units"]:
            if unit["kind"] == "tool" and unit["status"] == "ready":
                command = _substitute(unit["command"], Path(resolved["run_root"]),
                                      Path(resolved["data_root"]))
                print(f"  DRY-RUN [tool {unit['id']}]: " + " ".join(command))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口：校验并（可选）执行 R1D 论文 smoke。

    Args:
        argv: 参数列表（None 时用 ``sys.argv[1:]``）。

    Returns:
        进程退出码：0 成功（含 dry-run）；1 存在 batch error/timeout、工具失败
        或续跑产物校验失败；2 manifest/参数错误。
    """
    parser = argparse.ArgumentParser(
        description="R1D 论文 smoke 运行入口（manifest 驱动，T8a）")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH,
                        help="smoke manifest 路径")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="batch CSV 数据根（默认 <output>/batch）")
    parser.add_argument("--output", type=Path, default=None,
                        help="本次运行根目录（默认 experiment_data/r1d_v12_parity/paper_smoke）")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行进程数；R1D 论文 smoke 强制为 1")
    parser.add_argument("--timeout", type=float, default=None,
                        help="覆盖 manifest 默认单 run 超时（秒）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只校验/解析并打印完整任务清单，不执行子进程")
    parser.add_argument("--limit", type=int, default=None,
                        help="只保留前 N 条任务（调试用；若截断出单 seed cell 会失败，"
                             "除非显式加 --allow-single-seed）")
    parser.add_argument("--allow-single-seed", action="store_true",
                        help="仅调试：允许 --limit 截断出单 seed 的比较 cell（默认拒绝）")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="按身份命名空间续跑（默认）")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="忽略已完成的 runs 重新执行")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="覆盖 manifest seeds（小样本规则要求 ≥2 个共同 seed）")
    parser.add_argument("--mode", choices=VALID_MODE_FILTERS, default="torque",
                        help="只跑 torque / position / both（默认 torque）")
    parser.add_argument("--legacy-resume", action="store_true",
                        help="允许续跑旧 21 列 CSV（仅旧 RM65 身份；R1D 不应使用）")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        workers = (args.workers if args.workers is not None
                   else int(manifest.get("workers_default", REQUIRED_WORKERS)))
        if workers != REQUIRED_WORKERS:
            parser.error(
                f"R1D 论文 smoke 必须 --workers {REQUIRED_WORKERS}"
                f"（串行执行，避免多进程 MuJoCo 段错误），得到 {workers}")
        resolved = resolve_manifest(
            manifest, manifest_path=args.manifest, data_root=args.data_root,
            output=args.output, seeds=args.seeds, mode=args.mode,
            limit=args.limit, timeout_s=args.timeout,
            allow_single_seed=args.allow_single_seed)
    except ManifestError as exc:
        print(f"[r1d-smoke] manifest 校验失败: {exc}", file=sys.stderr)
        return 2

    run_root = Path(resolved["run_root"])
    resolved_path = run_root / "manifest_resolved.json"
    # M06：manifest 落盘用可搬移版本（run_root="."、内部路径相对），
    # 内存 resolved 继续保留绝对路径供执行与回填。
    _write_json(resolved_path, _portable_resolved(resolved, run_root))
    print(f"[r1d-smoke] 已写 {_display_path(resolved_path)}")
    _print_resolved_summary(resolved, dry_run=args.dry_run)
    if args.dry_run:
        print("[r1d-smoke] dry-run 完成：未执行任何子进程、未写 artifacts"
              "（仅写 manifest_resolved.json 与 runtime 覆盖快照）")
        return 0

    batch_root = Path(resolved["data_root"])
    # 兼容历史测试对 facade 私有函数的 monkeypatch（必须在调用点动态解析）。
    collect_runs = _patched("_collect_runs", _collect_runs)
    tool_dependency = _patched("_run_tool_dependency", _run_tool_dependency)
    specs = build_specs_from_manifest(
        args.manifest, data_root=args.data_root, output=args.output,
        seeds=args.seeds, mode=args.mode, limit=args.limit,
        timeout_s=args.timeout, workers=workers,
        allow_single_seed=args.allow_single_seed)
    spec_by_name = {spec.name: spec for spec in specs}
    summaries: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for unit in resolved["units"]:
        if unit["kind"] == "tool":
            result = tool_dependency(unit, run_root, batch_root)
            tool_results.append(result)
            print(f"[r1d-smoke] 工具依赖 {unit['id']}: {result['status']} "
                  f"log={result.get('log_path', '')}")
            continue
        if unit["kind"] != "batch" or unit["task_count"] == 0:
            continue
        spec = spec_by_name.get(str(unit["id"]))
        if spec is None:  # pragma: no cover - 解析与构建不一致属内部错误
            raise RuntimeError(f"内部错误：任务清单与 spec 不一致: {unit['id']}")
        try:
            summary = batch_runner.run_batch(
                spec,
                workers=workers,
                resume=args.resume,
                limit=None,
                # R10：只传 data_root（--data-root 优先，缺省 <run_root>/batch）。
                # 不再传优先级更高的 output=，否则 --data-root 会被静默覆盖，
                # 依赖/resume/_collect_runs 也会落到不同根。
                data_root=batch_root,
                timeout_s=resolved["timeout_s_default"],
                dry_run=False,
                allow_legacy_resume=args.legacy_resume,
                log_dir=run_root / "logs",
            )
        except ValueError as exc:
            # 身份/CSV 隔离契约拒绝混写：非零退出，不写 catalog/summary 掩盖冲突。
            print(f"[r1d-smoke] batch 拒绝执行 {unit['id']}: {exc}",
                  file=sys.stderr)
            return 2
        summaries.append(summary)
        print(f"[r1d-smoke] {spec.name}: launched={summary['launched']} "
              f"done_before={summary['done_before']} hits={summary['hits_this_run']} "
              f"errors={summary['errors']} timeouts={summary['timeouts']} "
              f"elapsed={summary['elapsed_s']}s")
    runs = collect_runs(resolved, batch_root, summaries)
    # N01：换 output 续跑（launched=0）时，把 manifest 的计划路径替换为旧 CSV
    # 里真实存在的 artifact/log；本次计划位置保留在 planned_* 字段。
    updated = _apply_actual_paths(resolved, runs.get("records") or [], run_root)
    # M09：续跑跳过的已完成行必须产物真实完好，否则静默 0 退出会掩盖数据缺失。
    artifact_problems = _validate_completed_artifacts(
        resolved, runs.get("records") or [], run_root)
    if updated:
        _write_json(resolved_path, _portable_resolved(resolved, run_root))
        print(f"[r1d-smoke] 已按旧 CSV 实际路径回填 {updated} 条任务并重写 "
              f"{_display_path(resolved_path)}")
    _print_artifact_problems(artifact_problems)
    catalog_path = _write_catalog(run_root, resolved)
    print(f"[r1d-smoke] 已写 {_display_path(catalog_path)}")
    totals = {
        "hits": sum(int(s.get("hits_this_run", 0)) for s in summaries),
        "errors": sum(int(s.get("errors", 0)) for s in summaries),
        "timeouts": sum(int(s.get("timeouts", 0)) for s in summaries),
        "tool_failures": sum(1 for r in tool_results
                             if r["status"] not in ("ok", "blocked")),
        # M09：续跑跳过但产物缺失/损坏的任务数（与 batch errors 分开统计）。
        "artifact_errors": len(artifact_problems),
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    summary_payload = {
        "schema": "r1d_paper_smoke_summary_v1",
        "run_root": resolved["run_root"],
        "config_hash": resolved["config_hash"],
        "seeds": resolved["seeds"],
        "mode_filter": resolved["mode_filter"],
        "batch": {"runs_csv": runs["runs_csv"], "runs": runs["runs"],
                  "hit_semantics": runs["hit_semantics"]},
        # F03：output 根的 catalog 声明（提取/绘图用 --catalog 消费）。
        "catalog": _display_path(catalog_path),
        "experiments": summaries,
        "tool_dependencies": tool_results,
        "totals": totals,
    }
    _write_json(run_root / "run_summary.json", summary_payload)
    print(f"[r1d-smoke] runs.csv={runs['runs_csv']} rows={runs['runs']}；"
          f"run_summary.json 已写")
    print(f"[r1d-smoke] 合计 hits={totals['hits']} errors={totals['errors']} "
          f"timeouts={totals['timeouts']} tool_failures={totals['tool_failures']} "
          f"artifact_errors={totals['artifact_errors']} "
          f"elapsed={totals['elapsed_s']}s")
    return 1 if (totals["errors"] or totals["timeouts"]
                 or totals["tool_failures"]
                 or totals["artifact_errors"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
