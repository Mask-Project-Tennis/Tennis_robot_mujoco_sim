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

from typing import Any

from scripts.exp.batch_runner import ExperimentSpec

SEEDS_P1 = 200   # P1 主线统一 200 seeds（加菜 C）
SEEDS_P2 = 100   # P2 网格实验 100 seeds
SEEDS_EXP14 = 50 # exp14 32 组合, 50 seeds 足够

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


def _perturb_grid() -> list[dict[str, Any]]:
    """exp17b: 时间×空间扰动二维网格 × ablation。

    --random-perturb + sign=random: 幅度 uniform[0,max]、符号随机,
    RNG 由 seed+99999 播种（可复现）。max=0 的轴不加扰动参数。
    """
    grid: list[dict[str, Any]] = []
    for t_max_ms in [0, 10, 25, 50, 100]:
        for s_max_m in [0.0, 0.05, 0.1, 0.2]:
            for ablation in ["full", "none"]:
                base: dict[str, Any] = {
                    "--serve-box": None, "--ball-speed": 9,
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
}


if __name__ == "__main__":
    total = 0
    for name, spec in EXPERIMENTS.items():
        n = len(spec.grid)
        total += n
        print(f"{name:20s} {n:>6} runs  [{spec.report_ref[:50]}]")
    print(f"{'合计':20s} {total:>6} runs")
