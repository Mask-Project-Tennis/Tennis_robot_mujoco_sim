#!/usr/bin/env python3
"""batch_runner —— 通用并行批量实验执行器。

功能:
- ProcessPoolExecutor 并行执行参数网格（默认 16 workers，实测 24 核甜点位）
- --resume: 从已有 results.csv 跳过已完成的 (config_id, seed)，支持断点续跑
- CSV 增量写入: 每个 run 完成后立即落盘，中断不丢已完成数据
- 单 run 失败记 error 行，不阻塞整体
- OMP_NUM_THREADS=1 防止 numpy/MuJoCo 内部线程与 worker 数超订

实验矩阵定义见 overnight_experiments.py（本文件只负责执行）。

用法:
    python scripts/exp/batch_runner.py exp15_speed --workers 16
    python scripts/exp/batch_runner.py exp17a_noise --limit 3   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 防 numpy/MuJoCo 内部多线程与 worker 数超订（必须在 import numpy 前设置）
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

RESULT_RE = re.compile(r"__RESULT__: (.+)")
RUN_TIMEOUT_S = 60

CSV_FIELDS = [
    "experiment", "config_id", "config_json", "seed", "hit", "hit_type",
    "pos_error", "vel_error", "min_dist", "ball_near_ms", "tube_ready_ms",
    "max_tcp", "max_qdot", "max_face", "hit_time_error_ms", "hit_pos_error",
    "v_racket_at_hit", "wall_time", "first_plan_ms", "ball_out_speed",
    "ball_out_vy", "error",
]
NUMERIC_FIELDS = {
    "pos_error", "vel_error", "min_dist", "ball_near_ms", "tube_ready_ms",
    "max_tcp", "max_qdot", "max_face", "hit_time_error_ms", "hit_pos_error",
    "v_racket_at_hit", "wall_time", "first_plan_ms", "ball_out_speed",
    "ball_out_vy",
}


@dataclass
class ExperimentSpec:
    """单个批量实验的完整定义。

    Attributes:
        name: 实验名（CLI 参数 & CSV experiment 列）。
        script: 相对项目根的仿真脚本路径。
        grid: 参数网格，每个元素是一次 run 的 CLI 参数字典
              （{"--flag": value, "--开关": None}，seed 由 batch_runner 注入）。
        report_ref: 旧报告对照摘要（写进 config.yaml 供 sanity check）。
    """

    name: str
    script: str
    grid: list[dict[str, Any]] = field(default_factory=list)
    report_ref: str = ""


def make_config_id(params: dict[str, Any]) -> str:
    """从非 seed 的 CLI 参数生成可读且稳定的 config_id。

    排序保证同一参数集合恒定映射到同一 id（resume 依赖此性质）。

    Args:
        params: 单次 run 的 CLI 参数字典。

    Returns:
        形如 "v12__ball-speed9__ablationfull" 的可读字符串。
    """
    parts: list[str] = []
    for key in sorted(params):
        if key == "--seed":
            continue
        val = params[key]
        flag = key.lstrip("-").replace("-", "")
        parts.append(f"{flag}{'' if val is None else val}")
    # 多值参数（如 --kp "500 500 ..."）会含空格，压成 -
    joined = "__".join(parts).replace(" ", "-").replace("/", "_")
    return joined or "default"


def parse_result_line(stdout: str) -> dict[str, Any]:
    """从仿真脚本的 stdout 中解析 __RESULT__ 行。

    Args:
        stdout: 子进程完整输出（stdout + stderr）。

    Returns:
        指标字典（数值字段已转 float）；无 __RESULT__ 行时返回 {"error": "no_result"}。
    """

    m = RESULT_RE.search(stdout)
    if not m:
        return {"error": "no_result"}
    fields = dict(re.findall(r"(\S+)=(\S+)", m.group(1)))
    for k in NUMERIC_FIELDS:
        if k in fields:
            try:
                fields[k] = float(fields[k])
            except ValueError:
                pass
    return fields


def load_done_keys(csv_path: Path) -> set[tuple[str, int]]:
    """读取已有 results.csv，返回已完成的 (config_id, seed) 集合。

    Args:
        csv_path: results.csv 路径（可不存在，返回空集）。

    Returns:
        已完成 run 的键集合；无 error 键的行才计入完成。
    """
    done: set[tuple[str, int]] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            try:
                done.add((row["config_id"], int(row["seed"])))
            except (KeyError, ValueError):
                continue
    return done


def run_one(experiment: str, script: str, params: dict[str, Any]) -> dict[str, Any]:
    """执行单次仿真 run（在 worker 进程内运行）。

    Args:
        experiment: 实验名（透传进 CSV）。
        script: 仿真脚本路径。
        params: CLI 参数字典，"--开关": None 表示布尔开关。

    Returns:
        一行完整的 CSV 记录（dict），失败时含 error 键。
    """
    from src.sim.hit_detection import determine_hit_from_type

    # "_script" 为参数级脚本覆盖键（exp13 需在同一实验里混跑 V11/V12），
    # 其余 "_" 开头的键为 meta，不传给子进程
    script_path = params.get("_script", script)
    cmd = [sys.executable, str(PROJECT / script_path)]
    for key, val in params.items():
        if key.startswith("_"):
            continue
        cmd.append(key)
        if val is not None:
            cmd.append(str(val))

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT_S,
            cwd=str(PROJECT), encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return _record(experiment, params, {"error": "timeout"})
    wall = round(time.perf_counter() - t0, 2)

    fields = parse_result_line(result.stdout + result.stderr)
    if "error" in fields:
        fields["wall_time"] = wall
        return _record(experiment, params, fields)

    hit_type = str(fields.get("hit_type", "miss"))
    fields["hit"] = determine_hit_from_type(hit_type)
    fields["wall_time"] = wall
    return _record(experiment, params, fields)


def _record(experiment: str, params: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    """把指标字典补全为一行完整 CSV 记录。"""
    record: dict[str, Any] = {
        "experiment": experiment,
        "config_id": make_config_id(params),
        "config_json": json.dumps(
            {k: v for k, v in params.items() if k != "--seed"},
            ensure_ascii=False, sort_keys=True),
        "seed": params.get("--seed"),
        "error": fields.get("error", ""),
    }
    for k in CSV_FIELDS:
        if k in fields:
            record[k] = fields[k]
        elif k not in record:
            record[k] = ""
    record["hit_type"] = fields.get("hit_type", "")
    return record


def run_batch(spec: ExperimentSpec, workers: int = 16,
              resume: bool = True, limit: int | None = None) -> dict[str, Any]:
    """并行执行一个实验的完整网格并增量写 CSV。

    Args:
        spec: 实验定义（含网格）。
        workers: 并行进程数。
        resume: True 时跳过 results.csv 中已完成的 (config_id, seed)。
        limit: 只跑前 N 个待完成任务（smoke test 用）。

    Returns:
        汇总字典: total / done_before / launched / hits / errors / elapsed_s / csv。
    """
    out_dir = PROJECT / "experiment_data" / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"

    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    done = load_done_keys(csv_path) if resume else set()
    tasks = [p for p in spec.grid
             if (make_config_id(p), p.get("--seed")) not in done]
    if limit is not None:
        tasks = tasks[:limit]

    print(f"[{spec.name}] 网格 {len(spec.grid)} | 已完成 {len(done)} | "
          f"待跑 {len(tasks)} | workers={workers}", flush=True)

    hits = errors = 0
    t0 = time.perf_counter()
    with open(csv_path, "a", newline="", encoding="utf-8") as f, \
            ProcessPoolExecutor(max_workers=workers) as pool:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        futures = {pool.submit(run_one, spec.name, spec.script, p): p
                   for p in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            record = fut.result()
            writer.writerow(record)
            f.flush()
            if record.get("error"):
                errors += 1
            elif record.get("hit"):
                hits += 1
            if i % 50 == 0 or i == len(tasks):
                rate = i / (time.perf_counter() - t0)
                print(f"[{spec.name}] {i}/{len(tasks)} "
                      f"({rate:.1f} runs/s, err={errors})", flush=True)

    summary = {
        "experiment": spec.name, "total": len(spec.grid),
        "done_before": len(done), "launched": len(tasks),
        "hits_this_run": hits, "errors": errors,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "csv": str(csv_path.relative_to(PROJECT)),
    }
    # 实验配置快照（覆盖写入，便于 git 归档实验定义）
    (out_dir / "config.yaml").write_text(
        f"# 由 batch_runner 自动生成\n# experiment: {spec.name}\n"
        f"# grid_size: {len(spec.grid)}\n# report_ref: {spec.report_ref}\n"
        f"# config_ids: {sorted({make_config_id(p) for p in spec.grid})}\n",
        encoding="utf-8")
    print(f"[{spec.name}] 完成: {json.dumps(summary, ensure_ascii=False)}",
          flush=True)
    return summary


def main() -> None:
    """CLI 入口: 按实验名执行（矩阵定义见 overnight_experiments.py）。"""
    from scripts.exp.overnight_experiments import EXPERIMENTS

    parser = argparse.ArgumentParser(description="通用并行批量实验执行器")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS),
                        help="实验名（见 overnight_experiments.py）")
    parser.add_argument("--workers", type=int, default=16, help="并行进程数")
    parser.add_argument("--no-resume", action="store_true",
                        help="忽略已有 results.csv 重跑全部")
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 个待完成任务（smoke test 用）")
    args = parser.parse_args()

    run_batch(EXPERIMENTS[args.experiment], workers=args.workers,
              resume=not args.no_resume, limit=args.limit)


if __name__ == "__main__":
    main()
