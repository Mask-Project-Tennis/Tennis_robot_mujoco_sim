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

from scripts.exp.batch_runner import ExperimentSpec

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
                   start: int = 1) -> list[dict[str, Any]]:
    """指定 (t_max_ms, s_max_m) 角点的高 seeds 复测网格（exp17g 用）。

    Args:
        speed: 球速（m/s）。
        cells: 角点列表 [(t_max_ms, s_max_m), ...]。
        modes: 消融档。
        seeds: 每格 seed 数。
        start: 起始 seed。

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
}


if __name__ == "__main__":
    total = 0
    for name, spec in EXPERIMENTS.items():
        n = len(spec.grid)
        total += n
        print(f"{name:20s} {n:>6} runs  [{spec.report_ref[:50]}]")
    print(f"{'合计':20s} {total:>6} runs")
