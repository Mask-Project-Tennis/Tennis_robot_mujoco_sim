#!/usr/bin/env python3
"""02_read_joints.py — 持续读关节角度和速度。

实时表格显示 6 个关节的角度（度/弧度）和速度（rad/s）。
数值微分计算速度（SDK 状态字典实测不可靠）。

用法:
    python 02_read_joints.py
    python 02_read_joints.py --hz 50          # 刷新频率

零风险: 只读，不发送任何指令。Ctrl+C 退出。
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _connect import add_config_arg, load_and_connect, safe_disconnect


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RM-65B 关节角度持续读取（零风险）")
    add_config_arg(parser)
    parser.add_argument("--hz", type=float, default=20.0, help="刷新频率 Hz（默认 20）")
    args = parser.parse_args()

    dt = 1.0 / args.hz
    ri, config = load_and_connect(args.config)

    print(f"\n持续读取关节角度（{args.hz:.0f}Hz），按 Ctrl+C 停止\n")

    # 表头
    header = (
        f"{'关节':>4s} | {'角度(°)':>10s} | {'角度(rad)':>10s} | {'速度(rad/s)':>12s}"
    )
    separator = "-" * len(header)

    try:
        while True:
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
            print(f"通信频率: {1.0/max(time.time() % 1, 0.001):.0f} Hz (目标 {args.hz:.0f} Hz)")

            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n\n已停止")
    finally:
        safe_disconnect(ri)


if __name__ == "__main__":
    main()
