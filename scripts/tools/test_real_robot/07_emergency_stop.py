#!/usr/bin/env python3
"""07_emergency_stop.py — 测试缓停和急停。

流程:
  1. 连接 + 回零位（小幅运动到测试起点）
  2. 发送缓慢运动（关节1 匀速旋转）
  3. 运动中按 Ctrl+C 触发缓停 → 验证机械臂平滑停止
  4. 重新连接后测试急停（rm_set_arm_stop，不可恢复）

⚠️ 急停测试后机械臂需要手动复位（重新上电或软件复位）。

用法:
    python 07_emergency_stop.py                    # 仅缓停测试（推荐首次）
    python 07_emergency_stop.py --test-estop       # 包含急停测试（需确认）

中风险: 有实际运动。Ctrl+C 缓停。
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _connect import (
    add_config_arg,
    load_and_connect,
    safe_disconnect,
)


def test_slow_stop(ri):
    """测试缓停：缓慢运动中触发 slow_stop。"""
    print("\n" + "=" * 50)
    print("测试 1: 缓停 (rm_set_arm_slow_stop)")
    print("=" * 50)
    print("\n机械臂将以缓慢速度运动，5秒后自动缓停。")
    print("也可以随时按 Ctrl+C 提前缓停。")

    confirm = input("\n输入 YES 开始缓停测试: ")
    if confirm != "YES":
        print("已跳过缓停测试")
        return

    # 记录起始位置
    state = ri.get_arm_state()
    q_start = state[:6].copy()

    # 缓慢运动: 关节1 在 ±5° 范围内匀速摆动
    dt = 0.01
    t_max = 5.0
    t0 = time.time()

    print("\n运动中...")
    try:
        while time.time() - t0 < t_max:
            t = time.time() - t0
            q = q_start.copy()
            q[0] += np.radians(5.0) * np.sin(2 * np.pi * 0.2 * t)
            ri.send_joint_command(q)
            time.sleep(dt)

        # 5秒后自动缓停
        print("5秒到达，自动缓停...")
        ri.slow_stop()
        time.sleep(0.5)
        print("✅ 缓停测试完成 — 机械臂应已平滑停止")

    except KeyboardInterrupt:
        print("\n⚠️ 手动 Ctrl+C — 正在缓停...")
        ri.slow_stop()
        time.sleep(0.5)
        print("✅ 缓停成功")

    # 回到起始位置
    print("\n回到测试起点...")
    n_steps = 50
    state = ri.get_arm_state()
    q_now = state[:6].copy()
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        q = q_now * (1 - alpha) + q_start * alpha
        ri.send_joint_command(q)
        time.sleep(0.01)


def test_emergency_stop(ri, config):
    """测试急停：rm_set_arm_stop。

    ⚠️ 急停后机械臂不可软件恢复，需要重新连接或手动复位。
    """
    print("\n" + "=" * 50)
    print("测试 2: 急停 (rm_set_arm_stop)")
    print("=" * 50)
    print("\n⚠️ 警告: 急停后机械臂不可软件恢复!")
    print("  - rm_set_arm_stop 会立即停止所有关节")
    print("  - 测试后需要重新连接或手动复位")
    print("  - 确保物理急停按钮在手边")

    confirm = input("\n⚠️ 确认要测试急停？输入 YES: ")
    if confirm != "YES":
        print("已跳过急停测试")
        return

    # 缓慢运动
    state = ri.get_arm_state()
    q_start = state[:6].copy()
    dt = 0.01
    t0 = time.time()

    print("\n运动中，3秒后急停...")
    try:
        while time.time() - t0 < 3.0:
            t = time.time() - t0
            q = q_start.copy()
            q[0] += np.radians(5.0) * np.sin(2 * np.pi * 0.2 * t)
            ri.send_joint_command(q)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass

    # 急停
    print("\n触发急停!")
    ri.emergency_stop()
    time.sleep(0.5)
    print("✅ 急停已执行 — 机械臂应已立即停止")
    print("⚠️ 后续需要重新连接或手动复位才能继续运动")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RM-65B 缓停/急停测试（中风险）")
    add_config_arg(parser)
    parser.add_argument("--test-estop", action="store_true",
                        help="包含急停测试（rm_set_arm_stop，不可恢复）")
    args = parser.parse_args()

    ri, config = load_and_connect(args.config)

    # 缓停测试
    test_slow_stop(ri)

    # 急停测试（可选）
    if args.test_estop:
        test_emergency_stop(ri, config)

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)

    safe_disconnect(ri)


if __name__ == "__main__":
    main()
