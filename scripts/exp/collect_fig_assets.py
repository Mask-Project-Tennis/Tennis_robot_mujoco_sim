#!/usr/bin/env python3
"""collect_fig_assets —— 冲刺补跑 Phase 1：重规划计时统计 + 图资产 NPZ 扫描。

两部分均为前台快速任务，产物落盘后立即可用（后续批量实验在 tmux 后台继续跑，
互不阻塞）:

1. 计时（fig7 数据源）: 两条序列
   - sync（默认配置, 与 exp13-17 全部实验一致）: DEBUG 级 ``REPLAN step=N ... t=XXms``
   - async（--async-replan, 架构能力）: INFO 级 ``ASYNC_PLAN done: ... t=XXms``
   输出 experiment_data/exp18_fig_assets/timing.json（含 by_iters 分解与首次规划）
2. NPZ 扫描（fig3/4/8 + 视频素材）: 4 类目标 run
   （干净命中/空间扰动命中/组合扰动 miss/噪声+KF 命中）逐 seed 扫描，
   命中目标类别且 episode 时长达标（wall >= min_wall_s）即保存 NPZ
   → experiment_data/exp18_fig_assets/raw/*.npz

用法:
    python scripts/exp/collect_fig_assets.py                        # 全量
    python scripts/exp/collect_fig_assets.py --timing-episodes 1 \
        --async-episodes 1 --scan-cap 2                             # 冒烟
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.exp.batch_runner import parse_result_line
from src.sim.hit_detection import determine_hit_from_type

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("collect_fig_assets")

V12 = PROJECT / "scripts" / "rm65_mpc_v12.py"
OUT_DIR = PROJECT / "experiment_data" / "exp18_fig_assets"
RAW_DIR = OUT_DIR / "raw"
TIMING_LOG_DIR = RAW_DIR / "timing_logs"

# 同步重规划（step=0 为 INFO, 稳态为 DEBUG）与异步求解日志
SYNC_RE = re.compile(
    r"REPLAN step=(\d+) k_hit=(-?\d+) iters=(\d+) horizon=(\d+) t=(\d+)ms")
ASYNC_RE = re.compile(
    r"ASYNC_PLAN done: step=(\d+) k_hit=(-?\d+) iters=(\d+) "
    r"horizon=(\d+) t=(\d+)ms")
RESULT_RE = re.compile(r"__RESULT__: (.+)")
STEP_RE = re.compile(r"__STEP_TIMING__: (.+)")
LAT_RE = re.compile(r"__STEP_LATENCIES__: (.+)")
RUN_TIMEOUT_S = 180

BASE_9MS: dict[str, Any] = {"--serve-box": None, "--ball-speed": 9, "--no-plot": None}

# 图资产四类目标 run；min_wall_s 过滤早退 episode（噪声大时 0.2s 空采样早退）
TARGETS: list[dict[str, Any]] = [
    {
        "name": "a_hit_clean",
        "want": "hit", "min_wall_s": 1.0,
        "desc": "基线 9 m/s 干净命中（fig3 走廊示意 + fig4 基准 + 视频）",
        "params": dict(BASE_9MS),
    },
    {
        "name": "b_hit_space_perturb",
        "want": "hit", "min_wall_s": 1.0,
        "desc": "空间扰动 0.1m 命中（fig4 对照 + fig8 走廊激活）",
        "params": {**BASE_9MS, "--random-perturb": None, "--perturb-sign": "random",
                   "--space-perturb-m": 0.1, "--space-perturb-min-m": 0.0},
    },
    {
        "name": "c_miss_combined_perturb",
        "want": "miss", "min_wall_s": 1.0,
        "desc": "t=50ms + s=0.2m 组合扰动 miss（fig8 失败模式 + 视频）",
        "params": {**BASE_9MS, "--random-perturb": None, "--perturb-sign": "random",
                   "--time-perturb-ms": 50, "--time-perturb-min-ms": 0,
                   "--space-perturb-m": 0.2, "--space-perturb-min-m": 0.0},
    },
    {
        "name": "d_hit_noise_kf",
        "want": "hit", "min_wall_s": 1.0,
        "desc": "噪声 σp=0.02 + KF 命中（fig8 感知面板备选）",
        "params": {**BASE_9MS, "--obs-noise-pos": 0.02, "--obs-noise-vel": 0.2,
                   "--obs-use-kf": None},
    },
]


def _ensure_mujoco_libpath() -> str:
    """定位 mujoco 动态库目录并注入 LD_LIBRARY_PATH（不 import mujoco）。

    Returns:
        定位到的库目录字符串（未找到时为空串）。
    """
    try:
        out = subprocess.run(
            [sys.executable, "-c",
             "import site; print(site.getsitepackages()[0])"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        libdir = Path(out.stdout.strip()) / "mujoco"
    except (OSError, subprocess.SubprocessError, IndexError):
        return ""
    if not any(libdir.glob("libmujoco.so*")):
        return ""
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if str(libdir) not in cur.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{libdir}:{cur}" if cur else str(libdir))
    return str(libdir)


def run_episode(params: dict[str, Any], dump_path: str | None = None
                ) -> tuple[str, dict[str, Any], float]:
    """运行单个 V12 episode，返回（完整输出, 指标字段, 墙钟秒）。

    Args:
        params: CLI 参数字典（{"--flag": value, "--开关": None}）。
        dump_path: 非 None 时传给 --dump-trajectory。

    Returns:
        (stdout+stderr 合并文本, 解析后的指标字段, 墙钟秒)。
    """
    cmd = [sys.executable, str(V12)]
    for key, val in params.items():
        cmd.append(key)
        if val is not None:
            cmd.append(str(val))
    if dump_path:
        cmd += ["--dump-trajectory", dump_path]
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT_S,
        cwd=str(PROJECT), encoding="utf-8", errors="replace", check=False,
    )
    wall = time.perf_counter() - t0
    text = result.stdout + result.stderr
    fields = parse_result_line(text)
    return text, fields, wall


def _stats(ms: list[int]) -> dict[str, Any]:
    """耗时样本的统计摘要（ms）。"""
    if not ms:
        return {"n": 0}
    ordered = sorted(ms)
    return {
        "n": len(ms),
        "mean_ms": round(statistics.fmean(ms), 1),
        "median_ms": round(statistics.median(ms), 1),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max_ms": max(ms),
        "min_ms": min(ms),
    }


def collect_series(episodes: int, async_mode: bool, tag: str) -> dict[str, Any]:
    """运行 N 个 episode，聚合重规划求解耗时（sync 或 async 序列）。

    Args:
        episodes: episode 数（seed 1..episodes）。
        async_mode: True 时加 --async-replan, 解析 ASYNC_PLAN 行。
        tag: 日志文件名前缀（sync/async）。

    Returns:
        timing.json 中的一条序列记录。
    """
    params: dict[str, Any] = {**BASE_9MS, "--log-level": "DEBUG",
                             "--dump-step-timing": None,
                             "--dump-step-latencies": None}
    if async_mode:
        params["--async-replan"] = None
    pattern = ASYNC_RE if async_mode else SYNC_RE
    records: list[dict[str, int]] = []
    stall_rows: list[dict[str, float]] = []
    raw_pool: list[float] = []
    per_episode: list[dict[str, Any]] = []
    for seed in range(1, episodes + 1):
        text, fields, wall = run_episode({**params, "--seed": seed})
        hits = [dict(zip(("step", "k_hit", "iters", "horizon", "t_ms"),
                         (int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]))))
                for g in pattern.findall(text)]
        records.extend(hits)
        m_step = STEP_RE.search(text)
        if m_step:
            kv = dict(re.findall(r"(\S+)=(\S+)", m_step.group(1)))
            row: dict[str, Any] = {}
            for k, v in kv.items():
                if k == "n":
                    row[k] = float(v)
                elif k == "deciles":
                    row["deciles"] = [float(x) for x in v.split(",")]
                else:
                    row[k] = float(v)
            stall_rows.append(row)
        # 逐步原始耗时（真 per-step ECDF 数据源，--dump-step-latencies 输出）
        m_lat = LAT_RE.search(text)
        if m_lat:
            raw_pool.extend(float(x) for x in m_lat.group(1).split(",") if x)
        per_episode.append({
            "seed": seed, "n_replans": len(hits), "wall_s": round(wall, 2),
            "mean_ms": (round(statistics.fmean([h["t_ms"] for h in hits]), 1)
                        if hits else None),
            "hit_type": fields.get("hit_type", fields.get("error", "?")),
        })
        (TIMING_LOG_DIR / f"{tag}_seed{seed:02d}.log").write_text(
            text, encoding="utf-8")
        logger.info("[timing:%s] seed=%d replans=%d hit=%s wall=%.1fs",
                    tag, seed, len(hits), fields.get("hit_type", "?"), wall)

    by_iters: dict[str, Any] = {}
    for rec in records:
        by_iters.setdefault(str(rec["iters"]), []).append(rec["t_ms"])
    # 首次规划只统计真实首解（iters>=30），不含 step=0 的 4ms JT 热身步
    n_first = sum(1 for r in records if r["step"] == 0 and r["iters"] >= 30)
    # 主循环停顿（每步墙钟）：同步 replan 阻塞步会拉长间隔, 异步保持平稳
    total_steps = int(sum(r["n"] for r in stall_rows))
    total_over = int(sum(r.get("n_over_period", 0) for r in stall_rows))
    stall: dict[str, Any] = {}
    if stall_rows:
        # 各 episode 的逐步耗时分位点池化 → 近似全样本 ECDF（实时图 (b) 面板数据源）
        pooled: list[float] = []
        for r in stall_rows:
            pooled.extend(r.get("deciles", []))
        stall = {
            "episodes": len(stall_rows),
            "total_steps": total_steps,
            "total_over_period": total_over,
            "over_ratio": round(total_over / total_steps, 5) if total_steps else 0,
            "p50_ms_mean": round(statistics.fmean(
                [r["p50"] for r in stall_rows]), 2),
            "p95_ms_mean": round(statistics.fmean(
                [r["p95"] for r in stall_rows]), 2),
            "p99_ms_max": max(r["p99"] for r in stall_rows),
            "max_ms_max": max(r["max"] for r in stall_rows),
            "deciles_pooled": sorted(pooled),
            # 逐步原始耗时池（fig8(b) 真 per-step ECDF；与 total_steps 同口径）
            "raw_pool": sorted(raw_pool),
        }
    return {
        "episodes": episodes,
        "async_mode": async_mode,
        "n_replans": len(records),
        "n_first_plan": n_first,
        "all": _stats([r["t_ms"] for r in records]),
        "first_plan": _stats([r["t_ms"] for r in records
                              if r["step"] == 0 and r["iters"] >= 30]),
        "steady_state": _stats([r["t_ms"] for r in records if r["step"] > 0]),
        "by_iters": {k: _stats(v) for k, v in sorted(
            by_iters.items(), key=lambda kv: int(kv[0]))},
        "stall": stall,
        "per_episode": per_episode,
    }


def scan_npz(scan_cap: int, seed_start: int) -> list[dict[str, Any]]:
    """逐 seed 扫描四类目标 run，达标即保存 NPZ。

    Args:
        scan_cap: 每个目标最多尝试的 seed 数。
        seed_start: 起始 seed。

    Returns:
        manifest 条目列表（未命中时 seed 为 None）。
    """
    manifest: list[dict[str, Any]] = []
    for target in TARGETS:
        found: dict[str, Any] | None = None
        for seed in range(seed_start, seed_start + scan_cap):
            tmp = OUT_DIR / ".tmp" / f"{target['name']}_{seed}.npz"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            _text, fields, wall = run_episode(
                {**target["params"], "--seed": seed}, dump_path=str(tmp))
            if "error" in fields:
                logger.warning("[npz] %s seed=%d 运行异常: %s",
                               target["name"], seed, fields["error"])
                tmp.unlink(missing_ok=True)
                continue
            hit = determine_hit_from_type(str(fields.get("hit_type", "miss")))
            got = "hit" if hit else "miss"
            ok_time = wall >= target["min_wall_s"]
            logger.info("[npz] %s seed=%d -> %s wall=%.1fs%s",
                        target["name"], seed, got, wall,
                        "" if ok_time else " (时长不足, 跳过)")
            if got == target["want"] and ok_time:
                dst = RAW_DIR / f"{target['name']}.npz"
                shutil.move(str(tmp), dst)
                found = {"name": target["name"], "desc": target["desc"],
                         "seed": seed, "outcome": got, "npz": dst.name,
                         "wall_s": round(wall, 2),
                         "hit_type": fields.get("hit_type"),
                         "pos_error": fields.get("pos_error")}
                break
            tmp.unlink(missing_ok=True)
        if found is None:
            logger.warning("[npz] %s: %d 个 seed 内未找到达标 %s",
                           target["name"], scan_cap, target["want"])
            manifest.append({"name": target["name"], "desc": target["desc"],
                             "seed": None, "outcome": None, "npz": None,
                             "scanned": scan_cap})
        else:
            manifest.append(found)
    return manifest


def main() -> int:
    """入口: 双序列计时 + NPZ 扫描，产物写入 exp18_fig_assets。"""
    parser = argparse.ArgumentParser(description="冲刺补跑 Phase 1 资产收集")
    parser.add_argument("--timing-episodes", type=int, default=30,
                        help="sync 计时 episode 数（默认 30）")
    parser.add_argument("--async-episodes", type=int, default=10,
                        help="async 计时 episode 数（默认 10）")
    parser.add_argument("--scan-cap", type=int, default=40,
                        help="NPZ 扫描每目标最大 seed 数（默认 40）")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--skip-npz", action="store_true",
                        help="跳过 NPZ 扫描（只重跑计时/停顿序列时用）")
    args = parser.parse_args()

    libdir = _ensure_mujoco_libpath()
    logger.info("mujoco 库目录: %s", libdir or "未找到（依赖调用方环境）")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TIMING_LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("==== Part 1 计时: sync %d ep + async %d ep @ 9 m/s ====",
                args.timing_episodes, args.async_episodes)
    sync = collect_series(args.timing_episodes, False, "sync")
    logger.info("sync: %s", {k: v for k, v in sync.items()
                             if k in ("n_replans", "all", "first_plan",
                                      "steady_state", "by_iters", "stall")})
    async_series: dict[str, Any] = {}
    if args.async_episodes > 0:
        async_series = collect_series(args.async_episodes, True, "async")
        logger.info("async: %s", {k: v for k, v in async_series.items()
                                  if k in ("n_replans", "all", "first_plan",
                                           "steady_state", "by_iters",
                                           "stall")})
    timing = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
              "config": BASE_9MS, "sync": sync, "async": async_series}
    (OUT_DIR / "timing.json").write_text(
        json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.skip_npz:
        logger.info("--skip-npz: 跳过 NPZ 扫描")
        return 0

    logger.info("==== Part 2 NPZ 扫描: 4 类目标, cap=%d ====", args.scan_cap)
    manifest = scan_npz(args.scan_cap, args.seed_start)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "targets": manifest},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    for item in manifest:
        logger.info("manifest: %s seed=%s outcome=%s",
                    item["name"], item.get("seed"), item.get("outcome"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
