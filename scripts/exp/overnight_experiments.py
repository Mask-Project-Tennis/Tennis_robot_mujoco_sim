#!/usr/bin/env python3
"""overnight_experiments —— 连夜重跑实验矩阵定义（数据丢失后重建）。

背景: 系统重装丢失 experiment_data 全部结果（报告保留在 docs/experiments/reports/）。
本文件定义 7 个实验的参数网格，由 batch_runner.py 并行执行:

P1 主线（V12 三剑客 + exp14, 加菜 C: 200 seeds / exp14 50 seeds）:
- exp13_arch          V11 vs V12 × 力矩/位置 × 200 seeds（对照报告 85.7%/42.9%）
- exp15_speed_v2      球速 7-15 共 9 档 × 200 seeds（对照报告 4 档 50 seeds）
- exp16_limits_v2     真机限位(real_robot.yaml) vs 仿真默认 × 200 seeds
- exp14_pd_v2         位置模式 Kp×Kd 扫描 32 组合 × 50 seeds

P2 核心改进（论文缺失的鲁棒性数据, 全部 V12 架构）:
- exp17a_noise        观测噪声扫描 σp × ablation × KF × 球速
- exp17b_perturb      时间×空间扰动网格（--random-perturb, RNG 按 seed 可复现）
- exp17c_obsfreq      观测频率退化（论文真机叙事, 对照 V11 旧版 exp9）

V11 时代旧实验(exp1-12)不重跑: 研发迭代记录, 论文不引用其数据。

用法:
    python scripts/exp/batch_runner.py <实验名>            # 见 EXPERIMENTS 键
    python scripts/exp/overnight_experiments.py            # 仅打印矩阵清单
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 独立运行（打印矩阵清单）时保证可 import scripts.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.exp.batch_runner import ExperimentSpec, RunIdentity

SEEDS_P1 = 200   # P1 主线统一 200 seeds（加菜 C）
SEEDS_P2 = 100   # P2 网格实验 100 seeds
SEEDS_EXP14 = 50 # exp14 32 组合, 50 seeds 足够
MODES_MECHANISM = ("full", "tube_only", "softmin_only", "none")  # 四档机制消融

V12 = "scripts/rm65_mpc_v12.py"


def _grid(base: dict[str, Any], seeds: int, start: int = 1) -> list[dict[str, Any]]:
    """给一组公共参数展开 seed 维度。

    Args:
        base: 公共 CLI 参数（不含 --seed）。
        seeds: seed 数量。
        start: 起始 seed（历史实验从 1 或 0 开始, 保持与旧报告一致用 1）。

    Returns:
        参数字典列表，每项含 --seed。
    """
    return [{**base, "--seed": s} for s in range(start, start + seeds)]


def _noise_grid() -> list[dict[str, Any]]:
    """exp17a: 噪声 × ablation × KF × 球速。

    σv = 10·σp 沿用 exp7 档位比例; σp=0 时 KF 无输入差异,
    只跑 kf=off（exp9 已验证 off 模式 nokf=kf 逐位相同）。
    """
    grid: list[dict[str, Any]] = []
    for sig in [0.0, 0.005, 0.01, 0.02, 0.05]:
        for ablation in ["full", "none"]:
            kf_options = [False] if sig == 0.0 else [False, True]
            for use_kf in kf_options:
                for speed in [7, 9, 12]:
                    base: dict[str, Any] = {
                        "--serve-box": None, "--ball-speed": speed,
                        "--ablation": ablation, "--no-plot": None,
                    }
                    if sig > 0:
                        base["--obs-noise-pos"] = sig
                        base["--obs-noise-vel"] = sig * 10
                    if use_kf:
                        base["--obs-use-kf"] = None
                    grid.extend(_grid(base, SEEDS_P2))
    return grid


def _perturb_grid(speed: int = 9,
                  modes: tuple[str, ...] = ("full", "none")) -> list[dict[str, Any]]:
    """exp17b/17e/17f: 时间×空间扰动二维网格 × 消融档。

    --random-perturb + sign=random: 幅度 uniform[0,max]、符号随机,
    RNG 由 seed+99999 播种（可复现）。max=0 的轴不加扰动参数。

    Args:
        speed: 球速（m/s）；exp17e 复证用 7。
        modes: 消融档（默认 full/none；exp17f 机制归因用 tube_only/softmin_only）。

    Returns:
        参数字典列表。
    """
    grid: list[dict[str, Any]] = []
    for t_max_ms in [0, 10, 25, 50, 100]:
        for s_max_m in [0.0, 0.05, 0.1, 0.2]:
            for ablation in modes:
                base: dict[str, Any] = {
                    "--serve-box": None, "--ball-speed": speed,
                    "--ablation": ablation, "--no-plot": None,
                }
                if t_max_ms > 0 or s_max_m > 0:
                    base["--random-perturb"] = None
                    base["--perturb-sign"] = "random"
                if t_max_ms > 0:
                    base["--time-perturb-ms"] = t_max_ms
                    base["--time-perturb-min-ms"] = 0
                if s_max_m > 0:
                    base["--space-perturb-m"] = s_max_m
                    base["--space-perturb-min-m"] = 0.0
                grid.extend(_grid(base, SEEDS_P2))
    return grid


def _obsfreq_grid() -> list[dict[str, Any]]:
    """exp17c: 观测频率退化 × 噪声 × KF × ablation（球速固定 9 最优档）。

    噪声 lo 档取 17a 的温和档 σp=0.01/σv=0.1; 无噪声时只跑 kf=off。
    """
    grid: list[dict[str, Any]] = []
    for freq in [200, 60, 30, 15, 10]:
        for noisy in [False, True]:
            kf_options = [False] if not noisy else [False, True]
            for use_kf in kf_options:
                for ablation in ["full", "none"]:
                    base: dict[str, Any] = {
                        "--serve-box": None, "--ball-speed": 9,
                        "--obs-freq": freq,
                        "--ablation": ablation, "--no-plot": None,
                    }
                    if noisy:
                        base["--obs-noise-pos"] = 0.01
                        base["--obs-noise-vel"] = 0.1
                    if use_kf:
                        base["--obs-use-kf"] = None
                    grid.extend(_grid(base, SEEDS_P2))
    return grid


def _perturb_cells(speed: int, cells: list[tuple[int, float]],
                   modes: tuple[str, ...], seeds: int,
                   start: int = 1,
                   spatial_target: str = "hitpoint") -> list[dict[str, Any]]:
    """指定 (t_max_ms, s_max_m) 角点的高 seeds 复测网格（exp17g 用）。

    Args:
        speed: 球速（m/s）。
        cells: 角点列表 [(t_max_ms, s_max_m), ...]。
        modes: 消融档。
        seeds: 每格 seed 数。
        start: 起始 seed。
        spatial_target: 空间扰动注入口径（hitpoint / ballstate）。

    Returns:
        参数字典列表。
    """
    grid: list[dict[str, Any]] = []
    for t_max_ms, s_max_m in cells:
        for mode in modes:
            base: dict[str, Any] = {
                "--serve-box": None, "--ball-speed": speed,
                "--ablation": mode, "--no-plot": None,
                "--random-perturb": None, "--perturb-sign": "random",
            }
            if t_max_ms > 0:
                base["--time-perturb-ms"] = t_max_ms
                base["--time-perturb-min-ms"] = 0
            if s_max_m > 0:
                base["--space-perturb-m"] = s_max_m
                base["--space-perturb-min-m"] = 0.0
                base["--spatial-perturb-target"] = spatial_target
            grid.extend({**base, "--seed": s}
                        for s in range(start, start + seeds))
    return grid


def _sensitivity_grid() -> list[dict[str, Any]]:
    """exp18_sensitivity: β/窗口/半径单因素敏感性（chat2 审稿补跑）。

    参数只在起作用的档位上扫（paired 对照 none 只跑一次，seed 对齐可配对）：
    - softmin 参数（β、窗口）→ softmin_only
    - 走廊半径 → tube_only
    - full × 全部 6 个非标称组合（全系统对参数选择的鲁棒性）
    """
    combos: list[tuple[dict[str, Any], tuple[str, ...]]] = [
        ({"--softmin-beta": 1.0}, ("full", "softmin_only")),
        ({"--softmin-beta": 10.0}, ("full", "softmin_only")),
        ({"--window-ms": 25.0}, ("full", "softmin_only")),
        ({"--window-ms": 75.0}, ("full", "softmin_only")),
        ({"--corridor-radius": 0.08}, ("full", "tube_only")),
        ({"--corridor-radius": 0.16}, ("full", "tube_only")),
    ]
    grid: list[dict[str, Any]] = []
    for combo, tiers in combos:
        for tier in tiers:
            base = {"--serve-box": None, "--ball-speed": 9, "--no-plot": None,
                    "--ablation": tier, **combo}
            grid.extend({**base, "--seed": s}
                        for s in range(1, SEEDS_P1 + 1))
    # 配对基准：none 档跑一次（标称参数，seed 与上方对齐）
    base_none = {"--serve-box": None, "--ball-speed": 9, "--no-plot": None,
                 "--ablation": "none"}
    grid.extend({**base_none, "--seed": s} for s in range(1, SEEDS_P1 + 1))
    return grid


def _resolve_data_root(data_root: Path | str | None) -> Path:
    """解析数据根目录（默认 <repo>/experiment_data）；不做旧路径回退。"""
    return (Path(data_root) if data_root is not None
            else Path(__file__).resolve().parent.parent.parent / "experiment_data")


def _exp28_seed_pool(
    max_disc: int = 4,
    data_root: Path | str | None = None,
) -> list[tuple[tuple[float, float] | None, list[int]]]:
    """exp28（chat14 碰撞常开验证）代表 seed 选择：从当前 data-root 的 exp24 CSV 按格挑选。

    延迟读取：调用时才访问 ``<data_root>/exp24_shared_full/results.csv``，
    不退回仓库旧 ``experiment_data`` 固定路径（T8a 矩阵要求）。

    每格优先级：tube 命中/full 未命中的不一致 seed（走廊>full 点估计的根源样本）、
    point-target 命中、full 命中；另加一个名义（无扰动，9 m/s）组。

    Args:
        max_disc: 每格最多挑选的不一致 seed 数。
        data_root: 数据根目录（默认 <repo>/experiment_data）。

    Returns:
        [(扰动格 (time_ms, space_m) 或 None, seed 列表), ...]（6 项，末项为名义组）。

    Raises:
        FileNotFoundError: 当前数据根下缺少 exp24_shared_full/results.csv。
        ValueError: CSV 存在但没有可用行 / 所有扰动格都选不出 seed（数据不足）。
    """
    import csv
    import json as _json

    root = _resolve_data_root(data_root)
    csv_path = root / "exp24_shared_full" / "results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"exp28 seed 池需要 exp24 结果：{csv_path} 不存在。"
            f"请先用 --data-root 指向含 exp24_shared_full/results.csv 的数据根，"
            f"或先在当前数据根跑 exp24_shared_full。")

    rows: dict[tuple[tuple[float, float], str], dict[int, bool]] = {}
    n_rows = 0
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            try:
                cfg = _json.loads(r["config_json"])
                cell = (float(cfg.get("--time-perturb-ms") or 0.0),
                        float(cfg["--space-perturb-m"]))
                seed = int(r["seed"])
                ablation = str(cfg["--ablation"])
            except (KeyError, TypeError, ValueError, _json.JSONDecodeError):
                continue
            rows.setdefault((cell, ablation), {})[seed] = (
                (r.get("hit_type") or "").strip() == "active")
            n_rows += 1
    if n_rows == 0:
        raise ValueError(
            f"exp28 seed 池数据不足：{csv_path} 没有解析出任何可用行"
            f"（需要 config_json 含 --space-perturb-m/--ablation 且 error 为空）。")

    cells = [(0.0, 0.1), (0.0, 0.2), (0.0, 0.3), (25.0, 0.2), (50.0, 0.2)]
    pool: list[tuple[tuple[float, float] | None, list[int]]] = []
    for cell in cells:
        tube = rows.get((cell, "tube_only"), {})
        full = rows.get((cell, "full"), {})
        none_ = rows.get((cell, "none"), {})
        seeds = sorted(set(tube) & set(full) & set(none_))
        picked: list[int] = []
        for s in seeds:                      # 1. 不一致 seed（走廊命中/full 未命中）
            if tube[s] and not full[s] and len(picked) < max_disc:
                picked.append(s)
        for key in (none_, full):            # 2/3. pt 命中、full 命中
            for s in seeds:
                if key[s] and s not in picked and len(picked) < max_disc + 4:
                    picked.append(s)
        if not picked:                       # 兜底：任何 seed 都没有就取前 6 个
            picked = seeds[:6]
        pool.append((cell, picked))

    if not any(picked for _cell, picked in pool):
        raise ValueError(
            f"exp28 seed 池数据不足：{csv_path} 中五类扰动格均无共同 seed"
            f"（tube_only/full/none 三档交集为空）。请先补跑 exp24_shared_full。")
    pool.append((None, list(range(1, 6))))   # 名义组（无扰动）
    return pool


def _exp28_grid(data_root: Path | str | None = None) -> list[dict[str, Any]]:
    """按当前数据根构造 exp28 参数网格（四档机制 × 代表 seed）。"""
    return [
        {**base, "--seed": seed}
        for cell, seeds in _exp28_seed_pool(data_root=data_root)
        for mode in MODES_MECHANISM
        for base in [({
            "--serve-box": None, "--ball-speed": 9,
            "--ablation": mode, "--no-plot": None,
            "--always-collision": None,
        } if cell is None else {
            "--serve-box": None, "--ball-speed": 9,
            "--ablation": mode, "--no-plot": None,
            "--always-collision": None,
            "--random-perturb": None, "--perturb-sign": "random",
            "--space-perturb-m": cell[1], "--space-perturb-min-m": 0.0,
            "--spatial-perturb-target": "ballstate",
            "--shared-input-full": None,
            **({"--time-perturb-ms": cell[0], "--time-perturb-min-ms": 0}
               if cell[0] > 0 else {}),
        })]
        for seed in seeds
    ]


def make_exp28_spec(data_root: Path | str | None = None) -> "ExperimentSpec":
    """构造 exp28 实验定义（延迟读取 exp24 CSV，import 阶段不读盘）。

    Args:
        data_root: 数据根目录（默认 <repo>/experiment_data）。

    Returns:
        ExperimentSpec（grid_factory 在 execute 时按 data_root 解析）。
    """
    return ExperimentSpec(
        name="exp28_always_collision",
        script=V12,
        report_ref="chat14 审稿 P4（最建议）: 碰撞常开验证 —— --always-collision 使"
                   "击球前球-拍碰撞常开（绕过 k_hit/距离窗口判定），对 exp24 五格与"
                   "名义条件的代表 seed 重跑四档，核对裁决是否与窗口口径一致。"
                   "每格 ≤10 seeds（不一致/pt 命中/full 命中优先）× 4 档 + 名义 5 seeds。",
        identity=RunIdentity(robot_id="rm65", racket_id="rm65_legacy",
                             config_hash="legacy", metric_schema="v12_result_v1"),
        # 延迟工厂在运行期接收 run_batch 解析好的 data-root；若调用方
        # 显式给 data_root 则用它，否则回退到构造时参数（默认 None →
        # 由 _exp28_grid 解析 <repo>/experiment_data）。
        grid_factory=lambda root: _exp28_grid(root if root is not None else data_root),
    )


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "exp13_arch": ExperimentSpec(
        name="exp13_arch",
        script=V12,
        report_ref="2026-06-18 报告: V12 力矩 85.7% / 位置 23.7%; V11 侧引用报告存档"
                   "（旧 V11 已重构为 V12 薄壳, stale cost 行为不可复现, 不重跑）",
        # 只跑 V12 力矩 + V12 位置（当前 HEAD 的标称性能, 旧 V11 数字引用报告）
        grid=(
            _grid({"--serve-box": None, "--ball-speed": 7, "--no-plot": None},
                  SEEDS_P1)
            + _grid({"--serve-box": None, "--ball-speed": 7, "--no-plot": None,
                     "--position-mode": None}, SEEDS_P1)
        ),
    ),
    "exp15_speed_v2": ExperimentSpec(
        name="exp15_speed_v2",
        script=V12,
        report_ref="2026-06-22 报告: 7/9/12/15 m/s → 78/90/86/58%, 50 seeds",
        grid=[
            {**base, "--seed": seed}
            for speed in [7, 8, 9, 10, 11, 12, 13, 14, 15]
            for base in [{"--serve-box": None, "--ball-speed": speed,
                          "--no-plot": None}]
            for seed in range(1, SEEDS_P1 + 1)
        ],
    ),
    "exp16_limits_v2": ExperimentSpec(
        name="exp16_limits_v2",
        script=V12,
        report_ref="2026-07-09 报告: real(TCP1.0) 30% vs sim(TCP1.8) 78%; "
                   "注意旧 --sim-limits 已废弃, 现默认即 sim 行为, real 走 --limits-config",
        grid=(
            _grid({"--serve-box": None, "--ball-speed": 7, "--no-plot": None,
                   "--limits-config": "configs/real_robot.yaml"}, SEEDS_P1)  # real
            + _grid({"--serve-box": None, "--ball-speed": 7, "--no-plot": None},
                    SEEDS_P1)                                                # sim=默认
        ),
    ),
    "exp14_pd_v2": ExperimentSpec(
        name="exp14_pd_v2",
        script=V12,
        report_ref="2026-06-22 报告: Kp=500 Kr=0.15 → 80.0%（旧 32 组合 × 30 seeds）",
        grid=[
            {"--serve-box": None, "--ball-speed": 7, "--no-plot": None,
             "--position-mode": None, "--kp": kp, "--kd": kp * kr,
             "--seed": seed}
            for kp in [50, 100, 150, 200, 300, 500, 750, 1000]
            for kr in [0.05, 0.08, 0.10, 0.15]
            for seed in range(1, SEEDS_EXP14 + 1)
        ],
    ),
    "exp17a_noise": ExperimentSpec(
        name="exp17a_noise",
        script=V12,
        report_ref="新实验: V12 版噪声扫描（旧 exp7 为 V11+stale cost bug, 结论不作数）",
        grid=_noise_grid(),
    ),
    "exp17b_perturb": ExperimentSpec(
        name="exp17b_perturb",
        script=V12,
        report_ref="新实验: 时间×空间扰动网格, 论文核心鲁棒性图（旧 v8/v9 为旧架构）",
        grid=_perturb_grid(),
    ),
    "exp17c_obsfreq": ExperimentSpec(
        name="exp17c_obsfreq",
        script=V12,
        report_ref="新实验: V12 版观测频率退化（旧 exp9 为 V11, 且当时含 stale cost bug）",
        grid=_obsfreq_grid(),
    ),
    "exp17d_mechanism": ExperimentSpec(
        name="exp17d_mechanism",
        script=V12,
        report_ref="四档消融补齐: tube_only/softmin_only × 3 球速 × 100 seeds; "
                   "full/none 复用 exp17b 基线格, 合并供 fig6 机制归因",
        grid=[
            {**base, "--seed": seed}
            for speed in [7, 9, 12]
            for mode in ["tube_only", "softmin_only"]
            for base in [{"--serve-box": None, "--ball-speed": speed,
                          "--ablation": mode, "--no-plot": None}]
            for seed in range(1, SEEDS_P2 + 1)
        ],
    ),
    "exp17e_perturb7": ExperimentSpec(
        name="exp17e_perturb7",
        script=V12,
        report_ref="exp17b 扰动网格 @ 7 m/s 复证（鲁棒性结论的球速泛化）",
        grid=_perturb_grid(speed=7),
    ),
    "exp17f_mechanism_perturb": ExperimentSpec(
        name="exp17f_mechanism_perturb",
        script=V12,
        report_ref="机制归因补测: tube_only/softmin_only × 扰动网格 @9 m/s; "
                   "标称四档（exp17d）显示 softmin 主导, 需在扰动下分离走廊贡献",
        grid=_perturb_grid(speed=9, modes=("tube_only", "softmin_only")),
    ),
    "exp17g_spatial_power": ExperimentSpec(
        name="exp17g_spatial_power",
        script=V12,
        report_ref="强化版机制归因: s=0.2 角点高 seeds（t=0/s=0.2 × 3000, "
                   "t=50/s=0.2 × 1500, 各 4 档 = 18000 runs）—— 目标: 把走廊 vs "
                   "时间窗的空间鲁棒差异（~3pp）做到统计显著（n≥3000/格, 功效 ~85%）",
        grid=(
            _perturb_cells(9, [(0, 0.2)], MODES_MECHANISM, 3000)
            + _perturb_cells(9, [(50, 0.2)], MODES_MECHANISM, 1500)
        ),
    ),
    "exp17h_extreme": ExperimentSpec(
        name="exp17h_extreme",
        script=V12,
        report_ref="极端条件补强（设计见 paper/planning/03-exp17h-extreme-conditions-design.md）: "
                   "Block A 9 m/s s∈{0.3,0.4} × 4 档 × 1500（走廊增益是否随扰动幅度单调）"
                   "+ Block B 12 m/s t=0/s=0.2 × 4 档 × 3000（紧时间余量下走廊是否放大）"
                   "= 24000 runs",
        grid=(
            _perturb_cells(9, [(0, 0.3)], MODES_MECHANISM, 1500)
            + _perturb_cells(9, [(0, 0.4)], MODES_MECHANISM, 1500)
            + _perturb_cells(12, [(0, 0.2)], MODES_MECHANISM, 3000)
        ),
    ),
    "exp17i_limits_ablation": ExperimentSpec(
        name="exp17i_limits_ablation",
        script=V12,
        report_ref="限速×机制消融配对实验（设计见 docs/experiments/design/exp17i_limits_ablation.md）: "
                   "TCP 1.0(real_robot.yaml) vs 1.8(默认) × 4 档 × 400 seeds @7 m/s = 3200 runs "
                   "—— 验证「限速越紧, 鲁棒层相对增益越大」（Q5 相对版主张, discussion §8#6）",
        grid=[
            {**base, "--ablation": mode, "--seed": s}
            for base in [
                # TCP 1.0（真机限位）
                {"--serve-box": None, "--ball-speed": 7, "--no-plot": None,
                 "--limits-config": "configs/real_robot.yaml"},
                # TCP 1.8（默认仿真限位）
                {"--serve-box": None, "--ball-speed": 7, "--no-plot": None},
            ]
            for mode in MODES_MECHANISM
            for s in range(1, 401)
        ],
    ),
    "exp18_sensitivity": ExperimentSpec(
        name="exp18_sensitivity",
        script=V12,
        report_ref="chat2 第二轮审稿补跑: β/窗口/半径单因素敏感性（标称 β5/w50/r0.12 "
                   "已由 E3 覆盖，不复跑）。9 m/s × 200 seeds，配对基准 none 跑一次："
                   "full×6 + softmin_only×{β1,β10,w25,w75} + tube_only×{r0.08,r0.16} "
                   "+ none×1 = 13 组合 × 200 = 2600 runs",
        grid=(
            _sensitivity_grid()
        ),
    ),
    "exp19_hardmin": ExperimentSpec(
        name="exp19_hardmin",
        script=V12,
        report_ref="chat8 审稿补跑: hard-min 终端基线。β=1e6 ≈ 数值 hard-min（softmin "
                   "对数-和-指数实现数值稳定），区分「多候选目标集合」与「softmin 平滑聚合」"
                   "两种解释。9 m/s 标称 × full/softmin_only × 200 seeds = 400 runs；"
                   "β=5 标称与 point-target 基线复用 exp17b/exp17d（同 seeds 1-200）。",
        grid=[
            {**base, "--ablation": mode, "--seed": s}
            for base in [
                {"--serve-box": None, "--ball-speed": 9, "--no-plot": None,
                 "--softmin-beta": 1e6},
            ]
            for mode in ("full", "softmin_only")
            for s in range(1, 201)
        ],
    ),
    "exp20_shared_pilot": ExperimentSpec(
        name="exp20_shared_pilot",
        script=V12,
        report_ref="chat9 审稿 pilot: ballstate 口径空间扰动（先偏移喂给规划器的球状态估计，"
                   "再生成击打点/候选集合/走廊轴/warm start —— 各代价项面对同一份有偏预测）。"
                   "9 m/s × s∈{0.1,0.2,0.3} × 四档 × 100 seeds = 1200 runs；"
                   "与 exp17b/17f 的 hitpoint 口径同 seed 对照，用于决定是否改用该协议"
                   "重跑全部空间侧实验（chat9 意见一）。",
        grid=_perturb_cells(9, [(0, 0.1), (0, 0.2), (0, 0.3)], MODES_MECHANISM,
                            SEEDS_P2, spatial_target="ballstate"),
    ),
    "exp22_shared_spatial": ExperimentSpec(
        name="exp22_shared_spatial",
        script=V12,
        report_ref="chat9 意见一 §A 方案: 把空间侧实验全部改为「共享预测输入」口径 "
                   "(--spatial-perturb-target ballstate)：候选集合/走廊轴/warm start 与单点目标"
                   "由同一份有偏球态生成，任何代价项都拿不到无偏几何参照。"
                   "覆盖 s>0 的全部格子：9 m/s 网格 15 格 × 4 档 × 100 seeds（6000）；"
                   "7 m/s 复证网格 15 格 × full/none × 100（3000）；"
                   "高功效角点 exp17g/17h 原地重跑（42000）。合计 51000 runs。"
                   "s=0 列与标称格不受协议影响，沿用 exp17b/17d/17f。",
        grid=(
            _perturb_cells(9, [(t, s) for t in (0, 10, 25, 50, 100)
                               for s in (0.05, 0.1, 0.2)],
                           MODES_MECHANISM, SEEDS_P2, spatial_target="ballstate")
            + _perturb_cells(7, [(t, s) for t in (0, 10, 25, 50, 100)
                                 for s in (0.05, 0.1, 0.2)],
                             ("full", "none"), SEEDS_P2, spatial_target="ballstate")
            + _perturb_cells(9, [(0, 0.2)], MODES_MECHANISM, 3000, spatial_target="ballstate")
            + _perturb_cells(9, [(50, 0.2)], MODES_MECHANISM, 1500, spatial_target="ballstate")
            + _perturb_cells(9, [(0, 0.3)], MODES_MECHANISM, 1500, spatial_target="ballstate")
            + _perturb_cells(9, [(0, 0.4)], MODES_MECHANISM, 1500, spatial_target="ballstate")
            + _perturb_cells(12, [(0, 0.2)], MODES_MECHANISM, 3000, spatial_target="ballstate")
        ),
    ),
    "exp23_nofreeze": ExperimentSpec(
        name="exp23_nofreeze",
        script=V12,
        report_ref="chat12 审稿 C1（必须项）: 不冻结时间端到端实验。"
                   "--no-freeze-first-plan 使首次规划（~928ms）期间球继续飞行，"
                   "计算延迟计入任务时间 —— 验证真实来球条件下的可执行性。"
                   "对照组 A = 冻结口径既有标称数据（exp15/17a/17d 同 seeds）；"
                   "对照组 B = 首规划加速版（--first-plan-iters 5，约 1/6 延迟），"
                   "给出延迟-成功率权衡曲线。"
                   "{7,9,12} m/s × 4 档 × 200 seeds + iters5 版 {9,12} × 4 档 × 100"
                   " = 3200 runs；__RESULT__ 新增 first_plan_ms 与 ball_out_*（C6）。",
        grid=[
            {**base, "--seed": seed}
            for speed in [7, 9, 12]
            for mode in MODES_MECHANISM
            for base in [{"--serve-box": None, "--ball-speed": speed,
                          "--ablation": mode, "--no-plot": None,
                          "--no-freeze-first-plan": None}]
            for seed in range(1, SEEDS_P1 + 1)
        ] + [
            # 首规划加速变体：30→5 次迭代，首次可执行输出延迟 ~1/6
            {**base, "--seed": seed}
            for speed in [9, 12]
            for mode in MODES_MECHANISM
            for base in [{"--serve-box": None, "--ball-speed": speed,
                          "--ablation": mode, "--no-plot": None,
                          "--no-freeze-first-plan": None,
                          "--first-plan-iters": 5}]
            for seed in range(1, SEEDS_P2 + 1)
        ],
    ),
    "exp24_shared_full": ExperimentSpec(
        name="exp24_shared_full",
        script=V12,
        report_ref="chat12 审稿 C2（必须项）: 统一观测扰动 —— 在 ballstate 口径上"
                   "加 --shared-input-full，使到达时刻预测与可达性门限也消费同一份"
                   "受扰球态（所有规划模块共用同一观测，端到端预测偏差实验）。"
                   "对照 = exp22 同 seed 同格子的 ballstate（时序通道无偏）口径。"
                   "9 m/s × 5 格 × 4 档 × 100 seeds = 2000 runs。",
        grid=[
            # _perturb_cells 不支持透传 --shared-input-full，手动展开
            {**base, "--seed": seed}
            for (t_ms, s_m) in [(0, 0.1), (0, 0.2), (0, 0.3), (25, 0.2), (50, 0.2)]
            for mode in MODES_MECHANISM
            for base in [{
                "--serve-box": None, "--ball-speed": 9,
                "--ablation": mode, "--no-plot": None,
                "--random-perturb": None, "--perturb-sign": "random",
                "--space-perturb-m": s_m, "--space-perturb-min-m": 0.0,
                "--spatial-perturb-target": "ballstate",
                "--shared-input-full": None,
                **({"--time-perturb-ms": t_ms, "--time-perturb-min-ms": 0}
                   if t_ms > 0 else {}),
            }]
            for seed in range(1, SEEDS_P2 + 1)
        ],
    ),
    "exp25_select_one": ExperimentSpec(
        name="exp25_select_one",
        script=V12,
        report_ref="chat12 审稿 C3（高度建议）: 候选选择基线 —— 从候选窗口按 IK 关节裕度"
                   "硬选单一目标，再跑标准 point-target MPC（--ablation select_one）。"
                   "回答「是否必须把候选集合嵌入 softmin，还是目标选择本身就足够」。"
                   "对照 = exp17a/17d 标称四档（同 seeds）。{7,9,12} × 200 seeds = 600 runs。",
        grid=[
            {**base, "--seed": seed}
            for speed in [7, 9, 12]
            for base in [{"--serve-box": None, "--ball-speed": speed,
                          "--ablation": "select_one", "--no-plot": None}]
            for seed in range(1, SEEDS_P1 + 1)
        ],
    ),
    "exp26_tcp_grid": ExperimentSpec(
        name="exp26_tcp_grid",
        script=V12,
        report_ref="chat12 审稿 C4（高度建议）: 来球速度 × TCP cap 二维扫描，"
                   "定位候选集合收益的反转区域（而非只比较两个离散设置）。"
                   "--max-tcp 统一口径（仅改 TCP，其余仿真默认；与 Table I 的"
                   "real_robot.yaml 全套限位口径不同，文中说明）。"
                   "{7,9} m/s × {1.0,1.4,1.8} × 4 档 × 100 seeds = 2400 runs。",
        grid=[
            {**base, "--seed": seed}
            for speed in [7, 9]
            for tcp in [1.0, 1.4, 1.8]
            for mode in MODES_MECHANISM
            for base in [{"--serve-box": None, "--ball-speed": speed,
                          "--ablation": mode, "--no-plot": None,
                          "--max-tcp": tcp}]
            for seed in range(1, SEEDS_P2 + 1)
        ],
    ),
    "exp27_ball_quality": ExperimentSpec(
        name="exp27_ball_quality",
        script=V12,
        report_ref="chat12 审稿 C6（高度建议）: 标称条件重跑收集出球质量指标"
                   "（ball_out_speed / ball_out_vy，弹性反弹模型的出球速度与 Y 分量）。"
                   "与 Table II nominal 同条件（9/12 m/s × 4 档 × 100 seeds = 800 runs），"
                   "回答「active contact 是否产生有意义的回击」。",
        grid=[
            {**base, "--seed": seed}
            for speed in [9, 12]
            for mode in MODES_MECHANISM
            for base in [{"--serve-box": None, "--ball-speed": speed,
                          "--ablation": mode, "--no-plot": None}]
            for seed in range(1, SEEDS_P2 + 1)
        ],
    ),
    "exp28_always_collision": make_exp28_spec(),
}


# ---------------------------------------------------------------------------
# R1D 论文 smoke（T8a）：仅暴露延迟工厂，不并入 EXPERIMENTS
# ---------------------------------------------------------------------------
R1D_SMOKE_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "configs" / "experiments" / "r1d_paper_smoke.yaml"
)


def make_r1d_paper_smoke_specs(
    manifest_path: Path | str | None = None,
    *,
    data_root: Path | str | None = None,
    output: Path | str | None = None,
    seeds: list[int] | None = None,
    mode: str = "torque",
    limit: int | None = None,
    timeout_s: float | None = None,
    workers: int = 1,
) -> list[ExperimentSpec]:
    """延迟构建 R1D 论文 smoke 的 ExperimentSpec 列表（T8a/T8c 共享工厂）。

    不并入旧 ``EXPERIMENTS``（默认 22 个键与旧 RM65 行为保持不变）；
    仅由 ``run_r1d_paper_smoke.py`` 或显式调用方使用。

    Args:
        manifest_path: smoke manifest 路径（默认 configs/experiments/r1d_paper_smoke.yaml）。
        data_root: batch CSV 数据根（默认 <output>/batch）。
        output: 本次运行根目录（默认 experiment_data/r1d_v12_parity/paper_smoke）。
        seeds: 只跑指定 seed 列表（None 用 manifest seeds）。
        mode: torque/position/both 过滤。
        limit: 只保留前 N 个 ready 任务。
        timeout_s: 覆盖 manifest 默认超时（秒）。
        workers: 记录到 spec 的预期 workers（供调用方校验）。

    Returns:
        ExperimentSpec 列表（每个 manifest 实验一个）。
    """
    from scripts.exp.run_r1d_paper_smoke import build_specs_from_manifest

    path = Path(manifest_path) if manifest_path is not None else R1D_SMOKE_MANIFEST_PATH
    return build_specs_from_manifest(
        path,
        data_root=data_root,
        output=output,
        seeds=seeds,
        mode=mode,
        limit=limit,
        timeout_s=timeout_s,
        workers=workers,
    )


if __name__ == "__main__":
    # 显式解析当前数据根并传给延迟工厂，保证 --data-root 语义与 import
    # 期打印一致（默认旧根，不读取不存在的目录）。
    default_data_root = _resolve_data_root(None)
    total = 0
    for name, spec in EXPERIMENTS.items():
        try:
            n = len(spec.materialize_grid(default_data_root))
        except FileNotFoundError as exc:
            print(f"{name:28s} {'lazy':>6} runs  [{exc}]")
            continue
        total += n
        print(f"{name:28s} {n:>6} runs  [{spec.report_ref[:40]}]")
    print(f"{'合计':28s} {total:>6} runs")
    print(f"\nR1D 论文 smoke 工厂: {R1D_SMOKE_MANIFEST_PATH}")
    print("  python scripts/exp/run_r1d_paper_smoke.py --dry-run")
