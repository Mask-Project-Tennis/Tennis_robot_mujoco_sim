#!/usr/bin/env python3
"""collect_fig_assets —— 图资产收集（重规划计时统计 + 图资产 NPZ 扫描）。

T8a 扩展：数据根/输出目录/机器人配置/超时可注入，R1D 与旧 RM65 共用同一入口。

两部分均为前台快速任务，产物落盘后立即可用:

1. 计时（fig7 数据源）: 两条序列
   - sync（默认配置）: DEBUG 级 ``REPLAN step=N ... t=XXms``
   - async（--async-replan, 仅 RM65）: INFO 级 ``ASYNC_PLAN done: ... t=XXms``
   输出 ``<output>/timing.json``（含 by_iters 分解与首次规划）
2. NPZ 扫描（fig3/4/8 + 视频素材）: 4 类目标 run
   - RM65 路径：``--dump-trajectory`` 落盘 NPZ
   - R1D 路径：``--output <dir>`` 跑完后从 ``v12_episode.json`` 判定物理命中，
     复制该目录下的 episode NPZ 作为图资产
   命中目标类别且 episode 时长达标（wall >= min_wall_s）即保存
   → ``<output>/raw/*.npz``

默认输出 ``experiment_data/exp18_fig_assets``（旧 RM65 默认不变）；
R1D 论文 smoke 建议 ``--output <run_root>/exp18_fig_assets``。

用法:
    python scripts/exp/collect_fig_assets.py                        # 全量（旧默认根）
    python scripts/exp/collect_fig_assets.py --timing-episodes 1 \
        --async-episodes 1 --scan-cap 2                             # 冒烟
    python scripts/exp/collect_fig_assets.py \
        --data-root experiment_data/r1d_v12_parity/paper_smoke/batch \
        --output experiment_data/r1d_v12_parity/paper_smoke/exp18_fig_assets \
        --robot-config configs/robots/r1d_right.yaml \
        --runtime-config configs/robots/r1d_v12_runtime_right.yaml \
        --racket-config configs/rackets/blade100_v9_reference.yaml
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.exp.batch_runner import parse_result_line  # noqa: E402
from src.sim.hit_detection import determine_hit_from_type  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("collect_fig_assets")

V12 = PROJECT / "scripts" / "rm65_mpc_v12.py"
DEFAULT_DATA_ROOT = PROJECT / "experiment_data"
DEFAULT_OUTPUT_NAME = "exp18_fig_assets"

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
# 固定任务模式（显式 --sim-task-config）下 R1D 拒绝这些旧 CLI 参数
FIXED_TASK_CONFLICT_KEYS = ("--serve-box", "--ball-speed", "--no-bounce",
                            "--position-mode")

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

# R1D 版四类目标：R1D 共享 CLI 面不接受 RM65 专属 flag，
# 扰动/噪声改用 --task-overrides（固定幅度，与 RM65 random 幅度为近似等价）
_TARGETS_R1D: list[dict[str, Any]] = [
    {
        "name": "a_hit_clean",
        "want": "hit", "min_wall_s": 1.0,
        "desc": "R1D 基线 9 m/s 干净命中（fig3/fig4 + 视频）",
        "params": dict(BASE_9MS),
    },
    {
        "name": "b_hit_space_perturb",
        "want": "hit", "min_wall_s": 1.0,
        "desc": "R1D 空间扰动 0.1m 命中（fig4 对照 + fig8 走廊激活；固定幅度近似）",
        "params": {**BASE_9MS, "--task-overrides": [
            "mpc.space_perturb_m=0.1",
            "mpc.spatial_perturb_target=hitpoint",
        ]},
    },
    {
        "name": "c_miss_combined_perturb",
        "want": "miss", "min_wall_s": 1.0,
        "desc": "R1D t=50ms + s=0.2m 组合扰动 miss（fig8 失败模式）",
        "params": {**BASE_9MS, "--task-overrides": [
            "mpc.time_perturb_s=0.05",
            "mpc.space_perturb_m=0.2",
        ]},
    },
    {
        "name": "d_hit_noise_kf",
        "want": "hit", "min_wall_s": 1.0,
        "desc": "R1D 噪声 σp=0.02 + KF 命中（fig8 感知面板备选）",
        "params": {**BASE_9MS, "--task-overrides": [
            "perception.obs_noise_pos=0.02",
            "perception.obs_noise_vel=0.2",
            "perception.obs_use_kf=true",
        ]},
    },
]


@dataclass(frozen=True)
class FigAssetPaths:
    """图资产目录与超时（数据根/输出可注入，R1D 输出本次根）。"""

    out_dir: Path
    raw_dir: Path
    timing_log_dir: Path
    run_timeout_s: float = RUN_TIMEOUT_S


def resolve_paths(
    data_root: Path | str | None = None,
    output: Path | str | None = None,
    timeout_s: float | None = None,
) -> FigAssetPaths:
    """解析图资产目录：默认保持旧 RM65 根 ``experiment_data/exp18_fig_assets``。

    Args:
        data_root: 数据根（默认 <repo>/experiment_data）。
        output: 直接指定图资产输出目录（优先于 data_root/exp18_fig_assets）。
        timeout_s: 单 episode 超时秒数（默认 180）。

    Returns:
        目录与超时配置。
    """
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    out_dir = (Path(output) if output is not None
               else root / DEFAULT_OUTPUT_NAME)
    raw_dir = out_dir / "raw"
    return FigAssetPaths(
        out_dir=out_dir,
        raw_dir=raw_dir,
        timing_log_dir=raw_dir / "timing_logs",
        run_timeout_s=float(timeout_s if timeout_s is not None else RUN_TIMEOUT_S),
    )


def build_robot_args(
    robot_config: str | Path | None = None,
    runtime_config: str | Path | None = None,
    racket_config: str | Path | None = None,
    position_mode: bool = False,
) -> list[str]:
    """把机器人配置透传参数拼成 argv 片段（R1D/RM65 共用）。"""
    args: list[str] = []
    for flag, value in (("--robot-config", robot_config),
                        ("--runtime-config", runtime_config),
                        ("--racket-config", racket_config)):
        if value:
            args += [flag, str(value)]
    if position_mode:
        args.append("--position-mode")
    return args


def is_r1d_robot(robot_config: str | Path | None) -> bool:
    """按 robot-config 文件名判定是否 R1D 入口（r1d_*.yaml）。"""
    return bool(robot_config) and Path(str(robot_config)).name.startswith("r1d")


def _append_params(cmd: list[str], params: dict[str, Any]) -> None:
    """把 CLI 参数字典追加到 argv；列表值展开为重复 flag。"""
    for key, val in params.items():
        if isinstance(val, (list, tuple)):
            for item in val:
                cmd += [str(key), str(item)]
            continue
        cmd.append(str(key))
        if val is not None:
            cmd.append(str(val))


def build_episode_cmd(
    params: dict[str, Any],
    *,
    dump_path: str | Path | None = None,
    robot_args: Sequence[str] = (),
    is_r1d: bool = False,
    output_dir: str | Path | None = None,
    fixed_task: bool = False,
) -> list[str]:
    """构造单个 V12 episode 的命令（集中一处，便于测试）。

    Args:
        params: 条件 CLI 参数（{"--flag": value, "--开关": None}）。
        dump_path: RM65 的 ``--dump-trajectory`` 输出路径（R1D 忽略）。
        robot_args: 机器人配置透传参数（见 build_robot_args）。
        is_r1d: True 时使用 R1D 产物路径（``--output``），不传 dump 参数。
        output_dir: R1D 单 episode 输出目录。
        fixed_task: 显式 ``--sim-task-config`` 固定任务模式，去掉冲突参数。

    Returns:
        完整 argv（首元素为当前 Python 解释器）。
    """
    if fixed_task:
        params = {k: v for k, v in params.items()
                  if k not in FIXED_TASK_CONFLICT_KEYS}
    cmd = [sys.executable, str(V12), *[str(a) for a in robot_args]]
    _append_params(cmd, params)
    if is_r1d:
        if output_dir is not None:
            cmd += ["--output", str(output_dir)]
    elif dump_path is not None:
        cmd += ["--dump-trajectory", str(dump_path)]
    return cmd


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


def run_episode(
    params: dict[str, Any],
    dump_path: str | Path | None = None,
    *,
    paths: FigAssetPaths | None = None,
    robot_args: Sequence[str] = (),
    is_r1d: bool = False,
    output_dir: str | Path | None = None,
    fixed_task: bool = False,
) -> tuple[str, dict[str, Any], float]:
    """运行单个 V12 episode，返回（完整输出, 指标字段, 墙钟秒）。

    Args:
        params: CLI 参数字典（{"--flag": value, "--开关": None}）。
        dump_path: 非 None 时传给 RM65 的 --dump-trajectory。
        paths: 目录/超时配置（None 用旧默认根）。
        robot_args: 机器人配置透传参数。
        is_r1d: True 时走 R1D 产物路径（--output）。
        output_dir: R1D 单 episode 输出目录。
        fixed_task: 固定任务模式（去掉冲突参数）。

    Returns:
        (stdout+stderr 合并文本, 解析后的指标字段, 墙钟秒)。
    """
    cfg = paths if paths is not None else resolve_paths()
    cmd = build_episode_cmd(
        params, dump_path=dump_path, robot_args=robot_args,
        is_r1d=is_r1d, output_dir=output_dir, fixed_task=fixed_task)
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=cfg.run_timeout_s,
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


def collect_series(
    episodes: int,
    async_mode: bool,
    tag: str,
    *,
    paths: FigAssetPaths | None = None,
    robot_args: Sequence[str] = (),
    is_r1d: bool = False,
    fixed_task: bool = False,
) -> dict[str, Any]:
    """运行 N 个 episode，聚合重规划求解耗时（sync 或 async 序列）。

    Args:
        episodes: episode 数（seed 1..episodes）。
        async_mode: True 时加 --async-replan（仅 RM65）。
        tag: 日志文件名前缀（sync/async）。
        paths: 目录/超时配置。
        robot_args: 机器人配置透传参数。
        is_r1d: True 时用 R1D 共享 CLI 面（去掉 dump-step-* 与 async）。
        fixed_task: 固定任务模式。

    Returns:
        timing.json 中的一条序列记录。
    """
    cfg = paths if paths is not None else resolve_paths()
    params: dict[str, Any] = {**BASE_9MS, "--log-level": "DEBUG"}
    if not is_r1d:
        params["--dump-step-timing"] = None
        params["--dump-step-latencies"] = None
        if async_mode:
            params["--async-replan"] = None
    pattern = ASYNC_RE if async_mode and not is_r1d else SYNC_RE
    records: list[dict[str, int]] = []
    stall_rows: list[dict[str, Any]] = []
    raw_pool: list[float] = []
    per_episode: list[dict[str, Any]] = []
    for seed in range(1, episodes + 1):
        run_params = {**params, "--seed": seed}
        out_dir = None
        if is_r1d:
            out_dir = cfg.out_dir / ".tmp" / "timing" / f"{tag}_seed{seed:02d}"
        _text, fields, wall = run_episode(
            run_params,
            paths=cfg, robot_args=robot_args, is_r1d=is_r1d,
            output_dir=out_dir, fixed_task=fixed_task)
        hits = [dict(zip(("step", "k_hit", "iters", "horizon", "t_ms"),
                         (int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]))))
                for g in pattern.findall(_text)]
        records.extend(hits)
        m_step = STEP_RE.search(_text)
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
        m_lat = LAT_RE.search(_text)
        if m_lat:
            raw_pool.extend(float(x) for x in m_lat.group(1).split(",") if x)
        per_episode.append({
            "seed": seed, "n_replans": len(hits), "wall_s": round(wall, 2),
            "mean_ms": (round(statistics.fmean([h["t_ms"] for h in hits]), 1)
                        if hits else None),
            "hit_type": fields.get("hit_type", fields.get("error", "?")),
        })
        cfg.timing_log_dir.mkdir(parents=True, exist_ok=True)
        (cfg.timing_log_dir / f"{tag}_seed{seed:02d}.log").write_text(
            _text, encoding="utf-8")
        logger.info("[timing:%s] seed=%d replans=%d hit=%s wall=%.1fs",
                    tag, seed, len(hits), fields.get("hit_type", "?"), wall)
        # R1D 计时 episode 只需日志/指标，产物目录用完即删，避免论文 smoke 输出膨胀
        if out_dir is not None:
            shutil.rmtree(out_dir, ignore_errors=True)

    if is_r1d:
        shutil.rmtree(cfg.out_dir / ".tmp" / "timing", ignore_errors=True)

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


def _r1d_artifact_hit(output_dir: Path) -> tuple[str, dict[str, Any]]:
    """读取 R1D 产物 v12_episode.json，返回 (hit|miss, 命中段)。"""
    artifact = output_dir / "v12_episode.json"
    if not artifact.is_file():
        return "", {}
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", {}
    hit = payload.get("hit") or {}
    if payload.get("hit", {}).get("physical_hit"):
        return "hit", hit
    return "miss", hit


def _r1d_artifact_npz(output_dir: Path) -> Path | None:
    """在 R1D 产物目录中查找 episode NPZ（优先 r1d_episode.npz）。"""
    preferred = output_dir / "r1d_episode.npz"
    if preferred.is_file():
        return preferred
    candidates = sorted(output_dir.glob("*.npz"))
    return candidates[0] if candidates else None


def scan_npz(
    scan_cap: int,
    seed_start: int,
    *,
    paths: FigAssetPaths | None = None,
    robot_args: Sequence[str] = (),
    is_r1d: bool = False,
    fixed_task: bool = False,
) -> list[dict[str, Any]]:
    """逐 seed 扫描四类目标 run，达标即保存 NPZ。

    Args:
        scan_cap: 每个目标最多尝试的 seed 数。
        seed_start: 起始 seed。
        paths: 目录/超时配置。
        robot_args: 机器人配置透传参数。
        is_r1d: True 时用 R1D 产物路径（v12_episode.json + episode NPZ）。
        fixed_task: 固定任务模式。

    Returns:
        manifest 条目列表（未命中时 seed 为 None）。
    """
    cfg = paths if paths is not None else resolve_paths()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    targets = _TARGETS_R1D if is_r1d else TARGETS
    manifest: list[dict[str, Any]] = []
    for target in targets:
        found: dict[str, Any] | None = None
        for seed in range(seed_start, seed_start + scan_cap):
            run_params = {**target["params"], "--seed": seed}
            if is_r1d:
                tmp_dir = cfg.out_dir / ".tmp" / f"{target['name']}_{seed}"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                _text, fields, wall = run_episode(
                    run_params, paths=cfg, robot_args=robot_args,
                    is_r1d=True, output_dir=tmp_dir, fixed_task=fixed_task)
                if "error" in fields:
                    logger.warning("[npz] %s seed=%d 运行异常: %s",
                                   target["name"], seed, fields["error"])
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    continue
                got, hit = _r1d_artifact_hit(tmp_dir)
                npz_src = _r1d_artifact_npz(tmp_dir)
                ok_time = wall >= target["min_wall_s"]
                logger.info("[npz] %s seed=%d -> %s wall=%.1fs%s",
                            target["name"], seed, got or "?", wall,
                            "" if ok_time else " (时长不足, 跳过)")
                if got == target["want"] and ok_time and npz_src is not None:
                    dst = cfg.raw_dir / f"{target['name']}.npz"
                    shutil.copyfile(npz_src, dst)
                    found = {"name": target["name"], "desc": target["desc"],
                             "seed": seed, "outcome": got, "npz": dst.name,
                             "wall_s": round(wall, 2),
                             "hit_type": hit.get("physical_hit_type"),
                             "pos_error": hit.get("pos_error")}
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    break
                shutil.rmtree(tmp_dir, ignore_errors=True)
                if got == target["want"] and ok_time:
                    logger.warning("[npz] %s seed=%d 命中但未找到 NPZ 产物",
                                   target["name"], seed)
                continue
            tmp = cfg.out_dir / ".tmp" / f"{target['name']}_{seed}.npz"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            _text, fields, wall = run_episode(
                run_params, dump_path=str(tmp), paths=cfg,
                robot_args=robot_args, fixed_task=fixed_task)
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
                dst = cfg.raw_dir / f"{target['name']}.npz"
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


def main(argv: Sequence[str] | None = None) -> int:
    """入口: 双序列计时 + NPZ 扫描，产物写入 <output>（默认旧 exp18_fig_assets）。"""
    parser = argparse.ArgumentParser(description="图资产收集（计时 + NPZ 扫描）")
    parser.add_argument("--timing-episodes", type=int, default=30,
                        help="sync 计时 episode 数（默认 30）")
    parser.add_argument("--async-episodes", type=int, default=10,
                        help="async 计时 episode 数（默认 10；R1D 自动跳过）")
    parser.add_argument("--scan-cap", type=int, default=40,
                        help="NPZ 扫描每目标最大 seed 数（默认 40）")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--skip-npz", action="store_true",
                        help="跳过 NPZ 扫描（只重跑计时/停顿序列时用）")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="数据根目录（默认 <repo>/experiment_data）")
    parser.add_argument("--output", type=Path, default=None,
                        help="图资产输出目录（默认 <data-root>/exp18_fig_assets）")
    parser.add_argument("--timeout", type=float, default=None,
                        help=f"单 episode 超时秒数（默认 {RUN_TIMEOUT_S}）")
    parser.add_argument("--robot-config", default=None, help="机器人描述透传")
    parser.add_argument("--runtime-config", default=None, help="运行参数透传")
    parser.add_argument("--racket-config", default=None, help="球拍配置透传")
    parser.add_argument("--sim-task-config", default=None,
                        help="固定任务配置透传（R1D 固定来球模式）")
    parser.add_argument("--position-mode", action="store_true",
                        help="位置执行模式")
    args = parser.parse_args(argv)

    is_r1d = is_r1d_robot(args.robot_config)
    fixed_task = bool(args.sim_task_config)
    robot_args = build_robot_args(
        args.robot_config, args.runtime_config, args.racket_config,
        args.position_mode)
    if fixed_task:
        robot_args += ["--sim-task-config", str(args.sim_task_config)]
    paths = resolve_paths(args.data_root, args.output, args.timeout)

    libdir = _ensure_mujoco_libpath()
    logger.info("mujoco 库目录: %s", libdir or "未找到（依赖调用方环境）")
    logger.info("图资产输出: %s | timeout=%.0fs | is_r1d=%s",
                paths.out_dir, paths.run_timeout_s, is_r1d)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.timing_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==== Part 1 计时: sync %d ep + async %d ep @ 9 m/s ====",
                args.timing_episodes, args.async_episodes)
    sync = collect_series(args.timing_episodes, False, "sync", paths=paths,
                          robot_args=robot_args, is_r1d=is_r1d,
                          fixed_task=fixed_task)
    logger.info("sync: %s", {k: v for k, v in sync.items()
                             if k in ("n_replans", "all", "first_plan",
                                      "steady_state", "by_iters", "stall")})
    async_series: dict[str, Any] = {}
    if args.async_episodes > 0 and not is_r1d:
        async_series = collect_series(args.async_episodes, True, "async",
                                      paths=paths, robot_args=robot_args,
                                      is_r1d=is_r1d,
                                      fixed_task=fixed_task)
        logger.info("async: %s", {k: v for k, v in async_series.items()
                                  if k in ("n_replans", "all", "first_plan",
                                           "steady_state", "by_iters",
                                           "stall")})
    elif args.async_episodes > 0:
        logger.warning("R1D 共享 CLI 面不接受 --async-replan，跳过 async 序列")
    timing = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
              "config": BASE_9MS, "robot_args": list(robot_args),
              "is_r1d": is_r1d, "sync": sync, "async": async_series}
    (paths.out_dir / "timing.json").write_text(
        json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.skip_npz:
        logger.info("--skip-npz: 跳过 NPZ 扫描")
        return 0

    logger.info("==== Part 2 NPZ 扫描: 4 类目标, cap=%d ====", args.scan_cap)
    manifest = scan_npz(args.scan_cap, args.seed_start, paths=paths,
                        robot_args=robot_args, is_r1d=is_r1d,
                        fixed_task=fixed_task)
    (paths.out_dir / "manifest.json").write_text(
        json.dumps({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "robot_args": list(robot_args), "is_r1d": is_r1d,
                    "targets": manifest},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    for item in manifest:
        logger.info("manifest: %s seed=%s outcome=%s",
                    item["name"], item.get("seed"), item.get("outcome"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
