#!/usr/bin/env python3
"""03_read_temperature.py — 持续读关节温度/电压/电流。

实时表格显示 6 个关节的温度（°C）、电压（V）、电流（mA）。

用法:
    python 03_read_temperature.py
    python 03_read_temperature.py --hz 5           # 低频即可

零风险: 只读，不发送任何指令。Ctrl+C 退出。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _connect import add_config_arg, load_and_connect, safe_disconnect


def main():
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="RM-65B 关节温度/电压/电流持续读取（零风险）")
    add_config_arg(parser)
    parser.add_argument("--hz", type=float, default=2.0, help="刷新频率 Hz（默认 2，温度变化慢）")
    args = parser.parse_args()

    dt = 1.0 / args.hz
    ri, config = load_and_connect(args.config)

    print(f"\n持续读取关节状态（{args.hz:.0f}Hz），按 Ctrl+C 停止\n")

    header = f"{'关节':>4s} | {'温度(°C)':>8s} | {'电压(V)':>8s} | {'电流(mA)':>10s}"
    separator = "-" * len(header)

    try:
        while True:
            temps = [0.0] * 6
            volts = [0.0] * 6
            currs = [0.0] * 6
            errors = []

            # 温度
            try:
                ret, data = ri._arm.rm_get_current_joint_temperature()
                if ret == 0:
                    temps = data[:6]
                else:
                    errors.append(f"温度错误码 {ret}")
            except Exception as e:
                errors.append(f"温度异常: {e}")

            # 电压
            try:
                ret, data = ri._arm.rm_get_current_joint_voltage()
                if ret == 0:
                    volts = data[:6]
                else:
                    errors.append(f"电压错误码 {ret}")
            except Exception as e:
                errors.append(f"电压异常: {e}")

            # 电流
            try:
                ret, data = ri._arm.rm_get_current_joint_current()
                if ret == 0:
                    currs = data[:6]
                else:
                    errors.append(f"电流错误码 {ret}")
            except Exception as e:
                errors.append(f"电流异常: {e}")

            print(f"\033[2J\033[H", end="")
            print(f"[03_read_temperature.py] {time.strftime('%H:%M:%S')}  Ctrl+C 停止\n")
            print(header)
            print(separator)
            for i in range(6):
                temp_warn = " ⚠️" if temps[i] > 60 else ""
                print(f"  J{i+1} | {temps[i]:8.1f} | {volts[i]:8.1f} | {currs[i]:10.0f}{temp_warn}")
            print(separator)
            if errors:
                print(f"⚠️ 警告: {'; '.join(errors)}")
            else:
                print("✅ 全部正常")

            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n\n已停止")
    finally:
        safe_disconnect(ri)


if __name__ == "__main__":
    main()
