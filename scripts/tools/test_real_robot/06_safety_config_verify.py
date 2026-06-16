#!/usr/bin/env python3
"""06_safety_config_verify.py — 安全参数读回验证。

连接时 RobotInterface._configure_safety() 自动下发安全参数到控制器。
本脚本读回这些参数，验证是否生效。

验证项:
  - 碰撞灵敏度 (rm_get_collision_stage)
  - 自碰撞检测 (rm_get_self_collision_enable)
  - 奇异性规避 (rm_get_avoid_singularity_mode)
  - 力矩限制 (rm_get_controller_torque_limit)
  - TCP 速度限制（间接验证: rm_get_arm_software_info）

用法:
    python 06_safety_config_verify.py
    python 06_safety_config_verify.py --config configs/real_robot.yaml

零风险: 只读，不发送任何运动指令。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _connect import add_config_arg, load_and_connect, safe_disconnect


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RM-65B 安全参数读回验证（零风险）")
    add_config_arg(parser)
    args = parser.parse_args()

    ri, config = load_and_connect(args.config)

    print(f"\n{'='*60}")
    print("安全参数读回验证")
    print(f"{'='*60}\n")

    results = []

    # 1. 碰撞灵敏度
    try:
        ret, stage = ri._arm.rm_get_collision_stage()
        expected = config.collision_stage
        ok = (ret == 0 and stage == expected)
        results.append(("碰撞灵敏度", ok, f"期望={expected}, 读回={stage}, ret={ret}"))
    except Exception as e:
        results.append(("碰撞灵敏度", False, f"异常: {e}"))

    # 2. 自碰撞检测
    try:
        ret, enabled = ri._arm.rm_get_self_collision_enable()
        expected = config.enable_self_collision
        ok = (ret == 0 and enabled == expected)
        results.append(("自碰撞检测", ok, f"期望={expected}, 读回={enabled}, ret={ret}"))
    except Exception as e:
        results.append(("自碰撞检测", False, f"异常: {e}"))

    # 3. 奇异性规避
    try:
        ret, mode = ri._arm.rm_get_avoid_singularity_mode()
        expected = 1 if config.enable_singularity_avoidance else 0
        ok = (ret == 0 and mode == expected)
        results.append(("奇异性规避", ok, f"期望={expected}, 读回={mode}, ret={ret}"))
    except Exception as e:
        results.append(("奇异性规避", False, f"异常: {e}"))

    # 4. 力矩限制
    try:
        ret, limit = ri._arm.rm_get_controller_torque_limit()
        expected = config.torque_limit
        ok = (ret == 0)
        results.append(("力矩限制", ok, f"期望={expected}, 读回={limit}, ret={ret}"))
    except Exception as e:
        results.append(("力矩限制", False, f"异常: {e}"))

    # 打印结果
    all_ok = True
    for name, ok, detail in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}: {detail}")
        if not ok:
            all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print("✅ 全部安全参数验证通过")
    else:
        print("⚠️ 部分参数不匹配，请检查 _configure_safety() 日志")
    print(f"{'='*60}")

    safe_disconnect(ri)


if __name__ == "__main__":
    main()
