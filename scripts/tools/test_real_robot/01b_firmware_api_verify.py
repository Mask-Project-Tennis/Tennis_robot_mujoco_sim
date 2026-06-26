#!/usr/bin/env python3
"""01b_firmware_api_verify.py — 固件版本 + 查询 API 可靠性验证。

固件升级后第一步：验证版本已更新 + 之前失败的 API 现在可用。
应在 01_connect_disconnect（连接成功）之后、02+（其他测试）之前运行。

验证项:
  - 固件版本读取（product/ctrl/algorithm/plan/dynamic 五字段）
  - 与升级前快照（V1.6.4）比对，标注哪些已升级
  - 10 个查询 API 可靠性测试（含之前失败的 2 个重点 API）

用法:
    python 01b_firmware_api_verify.py
    python 01b_firmware_api_verify.py --config configs/real_robot.yaml

零风险: 只读，不发送任何运动指令。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _connect import add_config_arg, load_and_connect, safe_disconnect


# 升级前快照（V1.6.4，2024-10-15 记录自 192.168.1.19 右臂）
PRE_UPGRADE_SNAPSHOT = {
    "product_version": "RM65-BI",
    "ctrl_version": "V1.6.4",
    "algorithm_version": "1.4.8",
    "plan_version": "V1.6.4",
    "dynamic_model": 2,
}

# 升级前失败的 API 及其错误码（用于重点标注是否已修复）
PRE_UPGRADE_FAILURES = {
    "rm_get_current_arm_state": "163/165",
    "rm_get_arm_all_state": "-3",
}


def _extract_version_fields(info: dict) -> dict:
    """从 rm_get_arm_software_info 返回的 dict 中提取 5 个版本字段。

    Args:
        info: rm_get_arm_software_info()[1] 返回的字典。

    Returns:
        包含 product_version / ctrl_version / algorithm_version / plan_version / dynamic_model 的字典。
    """
    return {
        "product_version": info.get("product_version", "N/A"),
        "ctrl_version": info.get("ctrl_info", {}).get("version", "N/A"),
        "algorithm_version": info.get("algorithm_info", {}).get("version", "N/A"),
        "plan_version": info.get("plan_info", {}).get("version", "N/A"),
        "dynamic_model": info.get("dynamic_info", {}).get("model_version", "N/A"),
    }


def check_firmware_version(arm) -> bool:
    """读取固件版本并与升级前快照比对。

    Args:
        arm: RoboticArm 实例（ri._arm）。

    Returns:
        版本读取是否成功。
    """
    print(f"\n{'='*60}")
    print("第一部分：固件版本读取 + 快照比对")
    print(f"{'='*60}\n")

    try:
        ret, info = arm.rm_get_arm_software_info()
    except Exception as e:
        print(f"  ❌ rm_get_arm_software_info 异常: {e}")
        return False

    if ret != 0:
        print(f"  ❌ rm_get_arm_software_info 失败, ret={ret}")
        return False

    current = _extract_version_fields(info)

    labels = [
        ("product_version", "产品型号"),
        ("ctrl_version", "控制器版本"),
        ("algorithm_version", "算法库版本"),
        ("plan_version", "规划层版本"),
        ("dynamic_model", "动力学模型"),
    ]
    for key, label in labels:
        old = PRE_UPGRADE_SNAPSHOT.get(key, "?")
        new = current[key]
        changed = str(old) != str(new)
        marker = "⬆️ 已升级" if changed else "=  未变"
        print(f"  {label}: {old} → {new}  {marker}")

    print(f"\n  ✅ 固件版本读取成功")
    return True


def check_query_apis(arm) -> bool:
    """测试 10 个查询 API 的可靠性。

    使用 getattr 容错：SDK 缺失的函数标记为 ⊘ 跳过。
    之前失败的 API（PRE_UPGRADE_FAILURES）会额外标注修复状态。

    Args:
        arm: RoboticArm 实例（ri._arm）。

    Returns:
        是否全部成功（跳过不算失败）。
    """
    print(f"\n{'='*60}")
    print("第二部分：查询 API 可靠性验证")
    print(f"{'='*60}\n")

    apis = [
        ("rm_get_arm_software_info", "软件版本信息"),
        ("rm_get_current_arm_state", "机械臂状态（关节+位姿+错误码）"),
        ("rm_get_arm_all_state", "全部状态（温度/电流/电压/错误码）"),
        ("rm_get_joint_degree", "关节角度"),
        ("rm_get_current_joint_temperature", "关节温度"),
        ("rm_get_current_joint_current", "关节电流"),
        ("rm_get_current_joint_voltage", "关节电压"),
        ("rm_get_init_pose", "初始位姿角度"),
        ("rm_get_install_pose", "安装角度"),
        ("rm_get_arm_plan_num", "轨迹规划计数"),
    ]

    results = []
    for func_name, desc in apis:
        func = getattr(arm, func_name, None)
        if func is None:
            print(f"  ⊘ {func_name} — 函数不存在于当前 SDK 版本")
            results.append((func_name, "skip"))
            time.sleep(0.2)
            continue

        try:
            t0 = time.perf_counter()
            result = func()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # 兼容 tuple (ret_code, data) 和 dict 两种返回形式
            if isinstance(result, tuple) and len(result) >= 2:
                ret_code = result[0]
            elif isinstance(result, dict):
                ret_code = result.get("return_code", result.get("code", -999))
            else:
                ret_code = -999

            was_broken = func_name in PRE_UPGRADE_FAILURES
            if ret_code == 0:
                tag = " ← 之前失败，现已修复 ✅" if was_broken else ""
                print(f"  ✅ {func_name}  ({elapsed_ms:.0f}ms){tag}")
                results.append((func_name, "ok"))
            else:
                old_code = PRE_UPGRADE_FAILURES.get(func_name, "")
                tag = f" ← 之前 {old_code}，仍然失败" if was_broken else ""
                print(f"  ❌ {func_name}  ret={ret_code}{tag}")
                results.append((func_name, "fail"))

        except Exception as e:
            print(f"  ❌ {func_name}  异常: {type(e).__name__}: {e}")
            results.append((func_name, "fail"))

        time.sleep(0.2)

    ok_count = sum(1 for _, s in results if s == "ok")
    fail_count = sum(1 for _, s in results if s == "fail")
    skip_count = sum(1 for _, s in results if s == "skip")
    print(f"\n  汇总: {ok_count} 成功 / {fail_count} 失败 / {skip_count} 跳过")
    return fail_count == 0


def main():
    """主函数：连接机械臂 → 检查固件版本 → 测试查询 API → 汇总判定。"""
    import argparse

    parser = argparse.ArgumentParser(description="RM-65B 固件版本 + API 可靠性验证（零风险）")
    add_config_arg(parser)
    args = parser.parse_args()

    ri, config = load_and_connect(args.config)

    version_ok = check_firmware_version(ri._arm)
    api_ok = check_query_apis(ri._arm)

    print(f"\n{'='*60}")
    if version_ok and api_ok:
        print("✅ 全部通过：固件版本读取成功 + 所有查询 API 正常")
    elif version_ok and not api_ok:
        print("⚠️ 固件版本读取成功，但有查询 API 失败（见上方详情）")
    else:
        print("❌ 固件版本读取失败，请检查 SDK/连接")
    print(f"{'='*60}")

    safe_disconnect(ri)


if __name__ == "__main__":
    main()
