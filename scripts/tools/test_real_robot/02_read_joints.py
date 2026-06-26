#!/usr/bin/env python3
"""02_read_joints.py — 持续读关节角度和速度。

实时表格显示 6 个关节的角度（度/弧度）和速度（rad/s）。
数值微分计算速度（SDK 状态字典实测不可靠）。

用法:
    python 02_read_joints.py
    python 02_read_joints.py --hz 50          # 刷新频率
    python 02_read_joints.py --benchmark 500  # 压测 500 次查询极限频率

零风险: 只读，不发送任何指令。Ctrl+C 退出。
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _connect import add_config_arg, load_and_connect, safe_disconnect


def run_benchmark(ri, n: int) -> None:
    """紧密循环 N 次 get_arm_state，统计延迟分布。

    无 sleep、无终端输出（仅进度提示），测量纯 SDK 查询极限频率。

    Args:
        ri: RobotInterface 实例。
        n: 查询次数。
    """
    print(f"\n压测模式：{n} 次无间隔查询\n")

    # 预热 1 次（建立 TCP 稳定连接）
    ri.get_arm_state()

    latencies = np.empty(n, dtype=np.float64)
    for i in range(n):
        t0 = time.perf_counter()
        ri.get_arm_state()
        latencies[i] = time.perf_counter() - t0
        if (i + 1) % 100 == 0:
            print(f"\r进度: {i+1}/{n}", end="", flush=True)
    print()

    ms = latencies * 1000.0
    total_ms = ms.sum()
    throughput = n / (latencies.sum())

    print(f"\n总耗时: {total_ms:.0f} ms")
    print(f"吞吐:   {throughput:.0f} Hz")
    print(f"\n延迟分布 (ms):")
    print(f"  min:    {ms.min():.2f}")
    print(f"  median: {np.median(ms):.2f}")
    print(f"  mean:   {ms.mean():.2f}")
    print(f"  p95:    {np.percentile(ms, 95):.2f}")
    print(f"  p99:    {np.percentile(ms, 99):.2f}")
    print(f"  max:    {ms.max():.2f}")
    print(f"  std:    {ms.std():.2f}")


def main():
    """主函数：benchmark 模式或交互式读取模式。"""
    import argparse

    parser = argparse.ArgumentParser(description="RM-65B 关节角度持续读取（零风险）")
    add_config_arg(parser)
    parser.add_argument("--hz", type=float, default=20.0, help="刷新频率 Hz（默认 20）")
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="压测模式：连续 N 次无间隔查询，统计延迟分布",
    )
    args = parser.parse_args()

    dt = 1.0 / args.hz
    ri, config = load_and_connect(args.config)

    # benchmark 模式：跑完即退出
    if args.benchmark > 0:
        run_benchmark(ri, args.benchmark)
        safe_disconnect(ri)
        return

    print(f"\n持续读取关节角度（{args.hz:.0f}Hz），按 Ctrl+C 停止\n")

    # 表头
    header = (
        f"{'关节':>4s} | {'角度(°)':>10s} | {'角度(rad)':>10s} | {'速度(rad/s)':>12s}"
    )
    separator = "-" * len(header)

    prev_loop = 0.0
    try:
        while True:
            loop_start = time.perf_counter()
            state = ri.get_arm_state()
            q_rad = state[:6]
            qdot = state[6:]
            q_deg = np.degrees(q_rad)

            # 清屏 + 打印表格
            print(f"\033[2J\033[H", end="")  # ANSI 清屏
            print(f"[02_read_joints.py] {time.strftime('%H:%M:%S')}  Ctrl+C 停止\n")
            print(header)
            print(separator)
            for i in range(6):
                print(
                    f"  J{i+1} | {q_deg[i]:10.2f} | {q_rad[i]:10.4f} | {qdot[i]:12.4f}"
                )
            print(separator)

            # 实际通信频率（两次循环间隔的倒数）
            if prev_loop > 0:
                actual_hz = 1.0 / (loop_start - prev_loop)
                print(f"实际频率: {actual_hz:.1f} Hz (目标 {args.hz:.0f} Hz)")
            else:
                print(f"(目标 {args.hz:.0f} Hz)")
            prev_loop = loop_start

            # 补偿循环体耗时：sleep 剩余时间
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(dt - elapsed, 0))
    except KeyboardInterrupt:
        print("\n\n已停止")
    finally:
        safe_disconnect(ri)


if __name__ == "__main__":
    main()
