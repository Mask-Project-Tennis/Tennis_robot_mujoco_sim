"""单关节跟踪实验 CLI 入口。

组装 src/joint_test/ 下已有的 6 个组件（types/waveform/recorder/safety/
analyzer/plotter/robot_adapter/experiment），按命令行参数构造对应后端，
运行单次或扫频跟踪实验，输出 NPZ/CSV/PNG 与终端指标摘要。

用法示例:
  # 仿真单次正弦测试
  python scripts/tools/joint_tracking_test.py --backend sim \
      --waveform sine --joint 2 --freq 1.0 --amplitude 0.2 --duration 5.0

  # 仿真 + FakeRobot 对比
  python scripts/tools/joint_tracking_test.py --backend sim --compare-fake \
      --waveform chirp --joint 2 --freq 0.1 --end-freq 5.0 \
      --amplitude 0.2 --duration 10.0

  # 批量扫频（生成 Bode 图）
  python scripts/tools/joint_tracking_test.py --backend sim --sweep \
      --joint 2 --sweep-freqs 0.1,0.2,0.5,1.0,2.0,3.0,5.0 --amplitude 0.1

  # 慢动作（便于观察）
  python scripts/tools/joint_tracking_test.py --backend sim --speed 0.5 \
      --waveform sine --joint 2 --freq 1.0 --amplitude 0.2 --duration 3.0

  # 真机测试（强制安全参数，交互式确认）
  python scripts/tools/joint_tracking_test.py --backend real \
      --i-understand-real-risk \
      --waveform sine --joint 2 --freq 0.5 --amplitude 0.05 --duration 3.0
"""
from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

# sys.path 注入（参考 rm65_joint_viewer.py / scan_joint_safety.py 风格）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.joint_test.types import (  # noqa: E402
    WaveformConfig,
    WaveformType,
    BackendType,
    TestConfig,
)
from src.joint_test.analyzer import MetricsAnalyzer  # noqa: E402
from src.joint_test.plotter import ResultPlotter  # noqa: E402
from src.joint_test.experiment import TrackingExperiment  # noqa: E402
from src.joint_test.robot_adapter import RobotAdapter  # noqa: E402
from src.joint_test.safety import JointSafetyGuard  # noqa: E402
from src.robot.constants import INIT_Q, INIT_Q_REAL, DT, KP, KD  # noqa: E402

logger = logging.getLogger(__name__)


def speed_type(x: str) -> float:
    """argparse 类型校验函数：--speed 必须落在 (0, 1.0] 区间。

    Args:
        x: 命令行原始字符串。

    Returns:
        解析后的浮点速度比例。

    Raises:
        argparse.ArgumentTypeError: 当值不在 (0, 1.0] 时。
    """
    v = float(x)
    if not 0.0 < v <= 1.0:
        raise argparse.ArgumentTypeError(
            f"--speed 必须在 (0, 1.0]，得到 {v}"
        )
    return v


def build_sim_adapter(dt: float) -> RobotAdapter:
    """构造仿真 adapter（RM65Env，位置模式）。

    Args:
        dt: 控制时间步长 (s)。

    Returns:
        RobotAdapter 包装的 MuJoCo 仿真环境。
    """
    from src.sim.rm65_env import RM65Env

    model_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src" / "robot" / "rm65_model.xml"
    )
    env = RM65Env(model_path=model_path, dt=dt)
    try:
        env.configure_actuator_mode("position", kp=KP, kd=KD)
    except Exception as e:  # pragma: no cover - 兜底
        logger.warning("configure_actuator_mode 失败（可能已默认位置模式）: %s", e)
    return RobotAdapter(env, BackendType.SIM, safety_guard=None)


def build_fake_adapter(dt: float) -> RobotAdapter:
    """构造 FakeRobot adapter（完美跟踪 Mock，零硬件依赖）。

    Args:
        dt: 控制时间步长 (s)。

    Returns:
        RobotAdapter 包装的 FakeRobot 实例。
    """
    from src.real.fake_robot import FakeRobot

    fake_robot = FakeRobot(INIT_Q.copy(), dt=dt)
    return RobotAdapter(fake_robot, BackendType.FAKE, safety_guard=None)


def _run_real_backend(args: argparse.Namespace) -> None:
    """真机后端编排：连接 → 预检 → 归位 → 运行实验。

    整个真机操作从连接后到断开前被 try/finally 保护，
    pre_motion_check 的 SystemExit 和 Ctrl+C 都能触发安全断开。

    Args:
        args: 解析后的命令行参数。
    """
    if not args.i_understand_real_risk:
        raise SystemExit(
            "真机模式必须加 --i-understand-real-risk 表示已理解风险"
        )

    from scripts.tools.test_real_robot._connect import (
        load_and_connect, init_algo, home_to_pose, safe_disconnect,
    )
    from src.real.safety_monitor import SafetyMonitor

    # ① 连接
    ri, cfg = load_and_connect(args.config)
    try:
        monitor = SafetyMonitor(cfg, ri)
        algo = None if args.no_algo_check else init_algo()

        # ② 显示当前角度
        state = ri.get_arm_state()
        print(f"\n当前关节角度 (度): {np.degrees(state[:6]).round(2)}")

        # ③ YES 确认（归位）
        confirm = input(
            "\n即将归位到 INIT_Q_REAL 并运行跟踪实验，输入 YES 确认: "
        )
        if confirm.strip().upper() != "YES":
            return

        # ④ 归位
        base_q = INIT_Q_REAL.copy()
        home_to_pose(ri, monitor, algo, base_q, duration=1.0)

        # ⑤ 确认到位
        final_state = ri.get_arm_state()
        error_deg = np.degrees(final_state[:6] - base_q)
        print(f"\n到位误差 (度): {error_deg.round(2)}")

        # ⑥ 非扫频模式加二次确认
        if not args.sweep:
            offset_val = (
                args.offset if args.offset is not None
                else float(base_q[args.joint])
            )
            confirm2 = input(
                f"\n即将运行 {args.waveform} J{args.joint} 跟踪, "
                f"频率 {args.freq} Hz, 幅值 {args.amplitude} rad, "
                f"偏置 {offset_val:.3f} rad, "
                f"持续 {args.duration}s. 输入 YES 确认: "
            )
            if confirm2.strip().upper() != "YES":
                return

        # ⑦ 构造组件
        guard = JointSafetyGuard(
            q_lower=cfg.q_lower,
            q_upper=cfg.q_upper,
            qdot_max=cfg.max_qdot,
            max_amplitude_rad=args.max_real_amplitude,
            max_frequency_hz=args.max_real_freq,
        )
        adapter = RobotAdapter(ri, BackendType.REAL, safety_guard=guard)

        # ⑧ 构造 experiment
        dt = DT
        analyzer = MetricsAnalyzer()
        plotter = ResultPlotter(
            Path(args.output_dir),
            backend="Agg" if args.no_plot else "TkAgg",
        )
        experiment = TrackingExperiment(
            adapter, analyzer, plotter, dt, base_q,
            speed_ratio=args.speed, backend=BackendType.REAL,
        )

        # ⑨ 运行
        if args.sweep:
            freqs = [float(x.strip()) for x in args.sweep_freqs.split(",")]
            offset = float(base_q[args.joint])

            # 扫频预检查：对每个频率做安全校验
            for f in freqs:
                wcfg_check = WaveformConfig(
                    waveform=WaveformType(args.waveform),
                    joint_idx=args.joint,
                    frequency_hz=f,
                    amplitude_rad=args.amplitude,
                    offset_rad=offset,
                    duration_s=args.duration,
                )
                warnings = guard.check_preconditions(wcfg_check)
                if warnings:
                    print(f"\n⚠️  频率 {f} Hz 安全预检查未通过:")
                    for w in warnings:
                        print(f"   - {w}")
                    raise SystemExit("拒绝运行（安全预检查失败）")

            sweep = experiment.run_sweep(
                joint_idx=args.joint,
                frequencies_hz=freqs,
                amplitude_rad=args.amplitude,
                waveform=WaveformType(args.waveform),
                duration_s=args.duration,
                compare_fake=args.compare_fake,
                pre_sweep_home=lambda: home_to_pose(ri, monitor, algo, base_q),
            )
            print(f"\n扫频完成: {len(freqs)} 个频率点，Bode 图已保存")
            for f, rmse in zip(sweep.frequencies_hz, sweep.rmses_rad):
                print(f"  f={f:.3f} Hz  RMSE={rmse:.6f} rad")
        else:
            offset = (
                args.offset if args.offset is not None
                else float(base_q[args.joint])
            )
            wcfg = WaveformConfig(
                waveform=WaveformType(args.waveform),
                joint_idx=args.joint,
                frequency_hz=args.freq,
                amplitude_rad=args.amplitude,
                offset_rad=offset,
                duration_s=args.duration,
                end_frequency_hz=args.end_freq,
                step_target_rad=args.step_target,
            )
            # 单次模式预检查
            warnings = guard.check_preconditions(wcfg)
            if warnings:
                print("\n⚠️  真机安全预检查未通过:")
                for w in warnings:
                    print(f"   - {w}")
                raise SystemExit("拒绝运行（安全预检查失败）")

            cfg_test = TestConfig(
                waveform_cfg=wcfg,
                backend=BackendType.REAL,
                realtime_plot=args.realtime and not args.no_plot,
                save_npz=args.save_npz,
                save_csv=args.save_csv,
                save_png=args.save_png,
                print_metrics=not args.no_metrics,
                compare_fake=args.compare_fake,
            )
            result, metrics = experiment.run_single(cfg_test)
            print(f"\n实验完成: RMSE={metrics.rmse_rad:.6f} rad")
    finally:
        safe_disconnect(ri)


def _run_sim_backend(args: argparse.Namespace) -> None:
    """sim/fake 后端编排：构造适配器 → 运行实验。

    Args:
        args: 解析后的命令行参数。
    """
    dt = DT
    backend = BackendType(args.backend)
    analyzer = MetricsAnalyzer()
    plotter = ResultPlotter(
        Path(args.output_dir),
        backend="Agg" if args.no_plot else "TkAgg",
    )

    adapter = (
        build_sim_adapter(dt) if backend == BackendType.SIM
        else build_fake_adapter(dt)
    )
    base_q = INIT_Q.copy()

    experiment = TrackingExperiment(
        adapter, analyzer, plotter, dt,
        base_q=base_q,
        speed_ratio=args.speed,
        backend=backend,
    )

    if args.sweep:
        freqs = [float(x) for x in args.sweep_freqs.split(",")]
        sweep = experiment.run_sweep(
            joint_idx=args.joint,
            frequencies_hz=freqs,
            amplitude_rad=args.amplitude,
            waveform=WaveformType(args.waveform),
            duration_s=args.duration,
            compare_fake=args.compare_fake,
        )
        print(f"\n扫频完成: {len(freqs)} 个频率点，Bode 图已保存")
        for f, rmse in zip(sweep.frequencies_hz, sweep.rmses_rad):
            print(f"  f={f:.3f} Hz  RMSE={rmse:.6f} rad")
    else:
        offset = (
            args.offset if args.offset is not None
            else float(base_q[args.joint])
        )
        wcfg = WaveformConfig(
            waveform=WaveformType(args.waveform),
            joint_idx=args.joint,
            frequency_hz=args.freq,
            amplitude_rad=args.amplitude,
            offset_rad=offset,
            duration_s=args.duration,
            end_frequency_hz=args.end_freq,
            step_target_rad=args.step_target,
        )
        cfg_test = TestConfig(
            waveform_cfg=wcfg,
            backend=backend,
            realtime_plot=args.realtime and not args.no_plot,
            save_npz=args.save_npz,
            save_csv=args.save_csv,
            save_png=args.save_png,
            print_metrics=not args.no_metrics,
            compare_fake=args.compare_fake,
        )
        result, metrics = experiment.run_single(cfg_test)
        print(f"\n实验完成: RMSE={metrics.rmse_rad:.6f} rad")


def main() -> None:
    """CLI 主入口：解析参数 → 构造组件 → 运行实验。"""
    from scripts.tools.test_real_robot import _connect

    parser = argparse.ArgumentParser(
        description="单关节跟踪实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 后端选择 ──
    parser.add_argument(
        "--backend", choices=["sim", "fake", "real"], default="sim",
        help="后端类型（默认 sim）",
    )

    # ── 波形参数 ──
    parser.add_argument(
        "--waveform",
        choices=[w.value for w in WaveformType],
        default="sine",
        help="波形类型（默认 sine）",
    )
    parser.add_argument(
        "--joint", type=int, default=2, choices=range(6),
        help="测试关节编号 0-5（默认 2）",
    )
    parser.add_argument(
        "--freq", type=float, default=1.0,
        help="波形频率 Hz（默认 1.0）",
    )
    parser.add_argument(
        "--amplitude", type=float, default=0.2,
        help="波形幅值 rad（默认 0.2）",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="持续时间 s（默认 5.0）",
    )
    parser.add_argument(
        "--end-freq", type=float, default=None,
        help="Chirp 终止频率 Hz",
    )
    parser.add_argument(
        "--step-target", type=float, default=None,
        help="Step 目标角度 rad",
    )
    parser.add_argument(
        "--offset", type=float, default=None,
        help="直流偏置 rad（默认 base_q[joint]）",
    )

    # ── 模式 ──
    parser.add_argument(
        "--sweep", action="store_true",
        help="批量扫频模式（生成 Bode 图）",
    )
    parser.add_argument(
        "--sweep-freqs", type=str,
        default="0.1,0.2,0.5,1.0,2.0,3.0,5.0",
        help="扫频频率列表，逗号分隔",
    )

    # ── 输出 ──
    parser.add_argument(
        "--output-dir", type=str,
        default="results/joint_tracking",
        help="输出目录（默认 results/joint_tracking）",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="不弹窗（CI 友好，强制 Agg backend）",
    )
    parser.add_argument(
        "--realtime", action="store_true",
        help="实验后弹窗显示",
    )
    parser.add_argument(
        "--save-npz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保存 NPZ（默认开启，可用 --no-save-npz 关闭）",
    )
    parser.add_argument(
        "--save-csv", action="store_true",
        help="保存 CSV（默认关闭）",
    )
    parser.add_argument(
        "--save-png",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保存 PNG（默认开启，可用 --no-save-png 关闭）",
    )
    parser.add_argument(
        "--no-metrics", action="store_true",
        help="不打印指标到终端",
    )
    parser.add_argument(
        "--compare-fake", action="store_true",
        help="附加 FakeRobot 完美跟踪基准线",
    )

    # ── 速度控制 ──
    parser.add_argument(
        "--speed", type=speed_type, default=1.0,
        help="速度比例 (0, 1.0]，1.0=最大速度，0.5=半速（默认 1.0）",
    )

    # ── 真机专用 ──
    parser.add_argument(
        "--i-understand-real-risk", action="store_true",
        help="真机模式必须加此 flag 表示已理解风险",
    )
    parser.add_argument(
        "--max-real-amplitude", type=float, default=0.05,
        help="真机模式幅值硬上限 rad（默认 0.05）",
    )
    parser.add_argument(
        "--max-real-freq", type=float, default=1.0,
        help="真机模式频率硬上限 Hz（默认 1.0）",
    )
    _connect.add_config_arg(parser)
    parser.add_argument(
        "--no-algo-check", action="store_true", default=False,
        help="跳过 SDK Algo 自碰撞/奇异性检查",
    )

    # ── 其他 ──
    parser.add_argument(
        "--log-level", default="INFO",
        help="日志级别（默认 INFO）",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.backend == "real":
        _run_real_backend(args)
    else:
        _run_sim_backend(args)


if __name__ == "__main__":
    main()
