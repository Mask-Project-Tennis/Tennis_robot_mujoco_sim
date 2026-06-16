#!/usr/bin/env python3
"""01_connect_disconnect.py — 连接测试。

验证流程: 连接 → 自动配置安全参数 → 读当前关节角度 → 断开。

用法:
    python 01_connect_disconnect.py
    python 01_connect_disconnect.py --config configs/real_robot.yaml

零风险: 不发送任何运动指令。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _connect import add_config_arg, load_and_connect, safe_disconnect


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RM-65B 连接测试（零风险）")
    add_config_arg(parser)
    args = parser.parse_args()

    # 1. 连接（自动配置安全参数）
    ri, config = load_and_connect(args.config)

    # 2. 读当前关节角度（验证通信）
    try:
        state = ri.get_arm_state()
        q_deg = __import__("numpy").degrees(state[:6])
        print(f"\n当前关节角度（度）: {q_deg.round(2)}")
        print("✅ 通信正常")
    except Exception as e:
        print(f"❌ 读取关节角度失败: {e}")

    # 3. 读固件信息（验证 SDK 扩展接口）
    try:
        ret, info = ri._arm.rm_get_arm_software_info()
        if ret == 0:
            print(f"固件信息: {info.get('product_version', 'N/A')}")
    except Exception:
        pass

    # 4. 安全断开
    safe_disconnect(ri)
    print("✅ 测试完成")


if __name__ == "__main__":
    main()
