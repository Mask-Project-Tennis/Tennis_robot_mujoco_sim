"""exp17: 单关节跟踪带宽表征 — 6 关节 × 8 频率正弦扫频。

调用 TrackingExperiment API 逐关节运行扫频，收集 SweepResult 指标，
写 results.csv + 生成对比汇总图（6 关节叠加 Bode + 截止频率柱状图）。
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# sys.path 注入（脚本位于 scripts/exp/）
_PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT))

from src.joint_test.types import SweepResult, WaveformType, BackendType  # noqa: E402
from src.joint_test.robot_adapter import RobotAdapter  # noqa: E402
from src.joint_test.experiment import TrackingExperiment  # noqa: E402
from src.joint_test.analyzer import MetricsAnalyzer  # noqa: E402
from src.joint_test.plotter import ResultPlotter  # noqa: E402
from src.robot.constants import INIT_Q, DT, KP, KD  # noqa: E402
from src.sim.rm65_env import RM65Env  # noqa: E402

logger = logging.getLogger(__name__)

# ─── 常量 ──────────────────────────────────────────────────
OUTPUT_DIR = _PROJECT / "experiment_data" / "exp17_joint_bandwidth"
JOINTS = [0, 1, 2, 3, 4, 5]
JOINT_NAMES = ["J0 肩偏航", "J1 肩俯仰", "J2 肘", "J3 腕1", "J4 腕2", "J5 腕3"]
FREQUENCIES_HZ = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
AMPLITUDE_RAD = 0.1
DURATION_S = 3.0

CSV_COLS = ["joint", "joint_name", "freq_hz", "rmse_rad", "max_error_rad",
            "mean_error_rad", "amplitude_ratio", "phase_lag_deg"]


def _find_cutoff(freqs: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    """线性插值找到 value 首次穿过 threshold 的频率。

    Args:
        freqs: 频率序列 (M,)，升序。
        values: 对应值序列 (M,)。
        threshold: 目标阈值。

    Returns:
        插值截止频率 (Hz)，若全程未穿过则返回 None。
    """
    if len(freqs) < 2:
        return None
    for i in range(len(freqs) - 1):
        a, b = values[i], values[i + 1]
        # 只找单调下降穿过 threshold 的点
        if (a >= threshold >= b) or (a <= threshold <= b):
            t = (threshold - a) / (b - a) if abs(b - a) > 1e-12 else 0.0
            return float(freqs[i] + t * (freqs[i + 1] - freqs[i]))
    return None


def _plot_all_joints_bode(sweeps: list[SweepResult], out_path: Path) -> None:
    """6 关节幅频 + 相频叠加对比图。

    Args:
        sweeps: 各关节的 SweepResult 列表（顺序对应 JOINTS）。
        out_path: 保存路径。
    """
    colors = plt.colormaps["tab10"](np.linspace(0, 1, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for i, sweep in enumerate(sweeps):
        color = colors[i]
        label = f"J{i} {JOINT_NAMES[i].split()[1]}" if i < len(JOINT_NAMES) else f"J{i}"

        ax1.plot(sweep.frequencies_hz, sweep.amplitude_ratios,
                 "o-", color=color, lw=1.5, label=label)
        ax2.plot(sweep.frequencies_hz, sweep.phase_lags_deg,
                 "o-", color=color, lw=1.5, label=label)

    ax1.axhline(y=1 / np.sqrt(2), color="r", ls="--", lw=0.8, label="-3dB (0.707)")
    ax1.set_ylabel("Amplitude Ratio A_act/A_des")
    ax1.set_title(f"Frequency Response — All Joints, sine A={AMPLITUDE_RAD:.2f} rad")
    ax1.legend(ncol=3, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")

    ax2.axhline(y=-45, color="gray", ls="--", lw=0.8, label="-45°")
    ax2.axhline(y=-90, color="gray", ls=":", lw=0.8, label="-90°")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase Lag (deg)")
    ax2.legend(ncol=3, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("汇总 Bode 图已保存: %s", out_path)


def _plot_bandwidth_summary(
    sweeps: list[SweepResult], out_path: Path,
) -> None:
    """截止频率柱状图：-3dB 幅值截止 + -45° 相位截止。

    Args:
        sweeps: 各关节 SweepResult 列表。
        out_path: 保存路径。
    """
    joints_label = [f"J{j}" for j in JOINTS]
    amp_cutoffs = []
    phase_cutoffs = []

    for sweep in sweeps:
        amp_cutoffs.append(
            _find_cutoff(sweep.frequencies_hz, sweep.amplitude_ratios, 1 / np.sqrt(2))
        )
        phase_cutoffs.append(
            _find_cutoff(sweep.frequencies_hz, sweep.phase_lags_deg, 45)
        )

    x = np.arange(len(JOINTS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, [c or 0 for c in amp_cutoffs], width,
                   label="-3dB Amp Cutoff", color="#4472C4")
    bars2 = ax.bar(x + width / 2, [c or 0 for c in phase_cutoffs], width,
                   label="-45 deg Phase Cutoff", color="#ED7D31")

    for bar in bars1:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(joints_label)
    ax.set_ylabel("Cutoff Frequency (Hz)")
    ax.set_title(f"Joint Tracking Bandwidth Comparison — sine A={AMPLITUDE_RAD:.2f} rad")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("带宽柱状图已保存: %s", out_path)


def main() -> None:
    """主入口：逐关节扫频 → 写 CSV → 绘汇总图。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 1. 创建 sim 后端
    model_path = _PROJECT / "src" / "robot" / "rm65_model.xml"
    env = RM65Env(
        model_path=model_path,
        dt=DT,
    )
    env.init_q_left = INIT_Q.copy()
    # 仅右臂（前 6 关节）为测试对象；左臂保持零位
    env.configure_actuator_mode("position", kp=KP, kd=KD)
    env.configure_feedforward(True)
    env.reset(INIT_Q)
    logger.info("RM65Env 初始化完成（位置模式）")

    adapter = RobotAdapter(env, BackendType.SIM)
    analyzer = MetricsAnalyzer()
    # Plotter 指向 raw/ 作为主输出目录（NPZ 数据保存处）
    plotter = ResultPlotter(
        OUTPUT_DIR / "raw",
        backend="Agg",
    )

    # 确保 figures 目录存在（Bode 图手动保存到 figures/）
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    # 2. 逐关节扫频
    sweeps: list[SweepResult] = []
    all_rows: list[dict] = []

    for j_idx, j_name in zip(JOINTS, JOINT_NAMES):
        logger.info("━━ 扫频 J%d (%s) ━━", j_idx, j_name)

        experiment = TrackingExperiment(
            adapter, analyzer, plotter, DT,
            base_q=INIT_Q.copy(),
            speed_ratio=1.0,
            backend=BackendType.SIM,
        )

        sweep = experiment.run_sweep(
            joint_idx=j_idx,
            frequencies_hz=FREQUENCIES_HZ,
            amplitude_rad=AMPLITUDE_RAD,
            waveform=WaveformType.SINE,
            duration_s=DURATION_S,
            save_npz=True,    # 扫频实验需保存原始数据
            print_metrics=False,
        )
        sweeps.append(sweep)

        # 将 sweep 自动生成的 Bode 图从 raw/ 移到 figures/
        bode_src = OUTPUT_DIR / "raw" / f"bode_j{j_idx}.png"
        if bode_src.exists():
            bode_dst = OUTPUT_DIR / "figures" / f"bode_j{j_idx}.png"
            bode_dst.unlink(missing_ok=True)
            bode_src.rename(bode_dst)

        # 收集指标行
        for freq, m in zip(sweep.frequencies_hz, sweep.individual_metrics):
            all_rows.append({
                "joint": j_idx,
                "joint_name": j_name,
                "freq_hz": f"{freq:.3f}",
                "rmse_rad": f"{m.rmse_rad:.6f}",
                "max_error_rad": f"{m.max_error_rad:.6f}",
                "mean_error_rad": f"{m.mean_error_rad:.6f}",
                "amplitude_ratio": f"{m.amplitude_ratio:.4f}" if m.amplitude_ratio is not None else "",
                "phase_lag_deg": f"{m.phase_lag_deg:.2f}" if m.phase_lag_deg is not None else "",
            })

    # 3. 写 CSV
    csv_path = OUTPUT_DIR / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info("results.csv 已写入 (%d 行)", len(all_rows))

    # 4. 汇总图
    logger.info("生成汇总图表...")
    _plot_all_joints_bode(sweeps, OUTPUT_DIR / "figures" / "bode_all_joints.png")
    _plot_bandwidth_summary(sweeps, OUTPUT_DIR / "figures" / "bandwidth_summary.png")

    # 5. 打印截止频率表
    print("\n" + "=" * 60)
    print("  各关节 -3dB / -45° 截止频率")
    print("=" * 60)
    print(f"{'关节':<6} {'名称':<10} {'-3dB Amp':>10} {'-45° Phase':>12}")
    print("-" * 48)
    for j_idx, j_name, sweep in zip(JOINTS, JOINT_NAMES, sweeps):
        amp_c = _find_cutoff(sweep.frequencies_hz, sweep.amplitude_ratios, 1 / np.sqrt(2))
        phase_c = _find_cutoff(sweep.frequencies_hz, sweep.phase_lags_deg, 45)
        amp_s = f"{amp_c:.2f} Hz" if amp_c is not None else ">12 Hz"
        phase_s = f"{phase_c:.2f} Hz" if phase_c is not None else ">12 Hz"
        print(f"  J{j_idx:<5} {j_name:<10} {amp_s:>10} {phase_s:>12}")
    print("=" * 60)

    # 6. 关闭 env（RM65Env 无 close 方法；让 GC 自动回收）
    logger.info("exp17 完成。")


if __name__ == "__main__":
    main()
