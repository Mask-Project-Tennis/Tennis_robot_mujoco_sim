#!/usr/bin/env python3
"""12_system_info_query.py — 系统信息查询。

读取运行模式、控制器状态、电源、机器人信息、安装姿态、软件版本等。
零风险：只读查询。

用法:
    python 12_system_info_query.py
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Windows GBK 控制台兼容：强制 stdout/stderr 用 UTF-8，避免 emoji 输出报错
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

# 确保能导入项目模块（_connect、src.*）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _connect import add_config_arg, load_and_connect, safe_disconnect  # noqa: E402


# ──────────────────────────────────────────────────────────────
# 查询工具函数
# ──────────────────────────────────────────────────────────────


def _query(arm: Any, name: str) -> tuple[int, Any]:
    """统一调用 SDK 查询接口，返回 (ret, payload)。

    适配两种返回形态：
      - tuple[int, list/dict]：直接拆分为 (ret, payload)
      - dict（内含 return_code）：读 return_code 作为 ret，dict 本身作为 payload
    异常时返回 (-1, {"exception": msg})，避免单接口失败拖垮整脚本。

    Args:
        arm: RoboticArm 实例（ri._arm）。
        name: SDK 方法名。

    Returns:
        (ret, payload) — ret==0 表示成功。
    """
    try:
        result = getattr(arm, name)()
    except Exception as e:  # noqa: BLE001
        return -1, {"exception": str(e)}
    if isinstance(result, tuple) and len(result) == 2:
        ret, payload = result
        return int(ret), payload
    if isinstance(result, dict):
        return int(result.get("return_code", 0)), result
    return 0, result


def _format_payload(payload: Any) -> str:
    """把 payload 格式化为可读字符串。

    dict 截断超长值，避免刷屏；其他类型直接 repr。

    Args:
        payload: 任意返回值。

    Returns:
        可读字符串。
    """
    if isinstance(payload, dict):
        items = []
        for k, v in payload.items():
            s = repr(v)
            if len(s) > 60:
                s = s[:57] + "..."
            items.append(f"{k}={s}")
        return "{" + ", ".join(items) + "}"
    return repr(payload)


# ──────────────────────────────────────────────────────────────
# 系统信息查询
# ──────────────────────────────────────────────────────────────


def query_system_info(arm: Any) -> dict[str, Any]:
    """查询全部系统信息接口（E1-E9），返回结果字典。

    对每个接口统一调用 _query，dict 返回值先保留原始结构。
    可选接口（工具端版本/RM Plus）失败时记录为 N/A，不算异常。

    Args:
        arm: RoboticArm 实例（ri._arm）。

    Returns:
        字典，键为 E1-E9 标识，值为 (ret, payload) 或异常信息。
    """
    info: dict[str, Any] = {}

    # E1 运行模式 tuple[int, int]
    info["E1_run_mode"] = _query(arm, "rm_get_arm_run_mode")

    # E2 控制器状态 dict
    info["E2_controller_state"] = _query(arm, "rm_get_controller_state")

    # E3 电源状态 tuple[int, int]
    info["E3_power_state"] = _query(arm, "rm_get_arm_power_state")

    # E4 机器人信息 tuple[int, dict]
    info["E4_robot_info"] = _query(arm, "rm_get_robot_info")

    # E5 安装姿态 dict
    info["E5_install_pose"] = _query(arm, "rm_get_install_pose")

    # E6 关节软件版本 tuple[int, dict]
    info["E6_joint_version"] = _query(arm, "rm_get_joint_software_version")

    # E7 工具端软件版本 tuple[int, dict]（可选）
    info["E7_tool_version"] = _query(arm, "rm_get_tool_software_version")

    # E8 RM Plus 基础信息 tuple[int, dict]（可选）
    info["E8_rm_plus_base"] = _query(arm, "rm_get_rm_plus_base_info")

    # E9 RM Plus 状态信息 tuple[int, dict]（可选）
    info["E9_rm_plus_state"] = _query(arm, "rm_get_rm_plus_state_info")

    return info


def print_system_info(info: dict[str, Any]) -> None:
    """打印系统信息快照。

    逐节打印 E1-E9 结果。dict 返回值先全量打印确认键名，再用 .get 提取
    常见字段。可选接口失败时标记 N/A。

    Args:
        info: query_system_info 返回的结果字典。
    """
    anomalies: list[str] = []

    # ── 运行状态（E1/E2/E3）──
    print("\n=== 运行状态 ===")

    ret, mode = info["E1_run_mode"]
    if ret == 0:
        ok = mode == 0
        print(f"  E1 运行模式: {mode} ({'空闲' if mode == 0 else '非空闲'}) "
              f"{'✅' if ok else '⚠️'}")
        if not ok:
            anomalies.append(f"运行模式非空闲（={mode}）")
    else:
        print(f"  E1 运行模式: 查询失败 ret={ret} payload={_format_payload(mode)}")
        anomalies.append(f"E1 运行模式查询失败 ret={ret}")

    ret, power = info["E3_power_state"]
    if ret == 0:
        ok = power == 1
        print(f"  E3 电源状态: {power} ({'上电' if power == 1 else '未上电'}) "
              f"{'✅' if ok else '⚠️'}")
        if not ok:
            anomalies.append(f"电源未上电（={power}）")
    else:
        print(f"  E3 电源状态: 查询失败 ret={ret} payload={_format_payload(power)}")
        anomalies.append(f"E3 电源状态查询失败 ret={ret}")

    ret, ctrl = info["E2_controller_state"]
    print(f"  E2 控制器状态 原始返回: {_format_payload(ctrl)}")
    if isinstance(ctrl, dict) and "exception" not in ctrl:
        cs = ctrl.get("controller_state", "N/A")
        cs_ok = cs == 0 if isinstance(cs, int) else False
        print(f"     controller_state = {cs} "
              f"({'正常' if cs_ok else '异常/未知'} {'✅' if cs_ok else '⚠️'})")
        if not cs_ok:
            anomalies.append(f"控制器状态异常（={cs}）")
    elif ret != 0:
        print(f"     查询失败 ret={ret}")
        anomalies.append(f"E2 控制器状态查询失败 ret={ret}")

    # ── 机器人信息（E4）──
    print("\n=== 机器人信息 ===")
    ret, robot = info["E4_robot_info"]
    if ret == 0 and isinstance(robot, dict):
        print(f"  E4 原始返回: {_format_payload(robot)}")
        # 尝试提取常见字段（键名未知，用 .get 兜底）
        model = robot.get("model", robot.get("arm_model", robot.get("name", "N/A")))
        dof = robot.get("dof", robot.get("arm_dof", "N/A"))
        reach = robot.get("reach", robot.get("arm_reach", "N/A"))
        payload_kg = robot.get("payload", robot.get("load", "N/A"))
        print(f"     机型   : {model}")
        print(f"     自由度 : {dof}")
        print(f"     臂展   : {reach}")
        print(f"     负载   : {payload_kg}")
        if isinstance(model, str) and "65" not in model.lower():
            anomalies.append(f"机型非 RM65（={model}）")
        if isinstance(dof, int) and dof != 6:
            anomalies.append(f"自由度非 6（={dof}）")
    else:
        print(f"  E4 机器人信息: 查询失败 ret={ret} payload={_format_payload(robot)}")
        anomalies.append(f"E4 机器人信息查询失败 ret={ret}")

    # ── 安装姿态（E5）──
    print("\n=== 安装姿态 ===")
    ret, pose = info["E5_install_pose"]
    print(f"  E5 原始返回: {_format_payload(pose)}")
    if isinstance(pose, dict) and "exception" not in pose:
        x = pose.get("x", "N/A")
        y = pose.get("y", "N/A")
        z = pose.get("z", "N/A")
        print(f"     x={x}°, y={y}°, z={z}°")
        # 期望竖直正装 (0,0,0)
        upright = (
            isinstance(x, (int, float)) and isinstance(y, (int, float))
            and isinstance(z, (int, float)) and x == 0 and y == 0 and z == 0
        )
        print(f"     安装方式: {'竖直正装 ✅' if upright else '非标准安装 ⚠️（确认物理方向）'}")
        if not upright:
            anomalies.append(
                f"安装姿态非 (0,0,0)（x={x},y={y},z={z}），需确认并更新 config"
            )
    elif ret != 0:
        print(f"     查询失败 ret={ret}")
        anomalies.append(f"E5 安装姿态查询失败 ret={ret}")

    # ── 软件版本（E6/E7）──
    print("\n=== 软件版本 ===")

    ret, jver = info["E6_joint_version"]
    if ret == 0:
        print(f"  E6 关节软件版本 原始返回: {_format_payload(jver)}")
        if isinstance(jver, dict) and "exception" not in jver:
            # 检查各关节版本是否一致（扫描列表型值）
            versions_set: set = set()
            for _k, v in jver.items():
                if isinstance(v, (list, tuple)):
                    versions_set.update(tuple(v))
            if len(versions_set) > 1:
                anomalies.append("各关节软件版本不一致，建议统一升级")
                print(f"     ⚠️ 关节版本不一致: {versions_set}")
            elif versions_set:
                print(f"     各关节版本一致 ✅ ({next(iter(versions_set))})")
    else:
        print(f"  E6 关节软件版本: 查询失败 ret={ret} payload={_format_payload(jver)}")
        anomalies.append(f"E6 关节软件版本查询失败 ret={ret}")

    ret, tver = info["E7_tool_version"]
    if ret == 0:
        print(f"  E7 工具端软件版本: {_format_payload(tver)}")
    else:
        # 工具端控制器可选，失败属正常
        print(f"  E7 工具端软件版本: N/A（ret={ret}，无工具端控制器属正常）")

    # ── RM Plus（E8/E9，可选模块）──
    print("\n=== RM Plus ===")

    ret, base = info["E8_rm_plus_base"]
    if ret == 0:
        print(f"  E8 RM Plus 基础信息: {_format_payload(base)}")
    else:
        print(f"  E8 RM Plus 基础信息: N/A（ret={ret}，未安装 RM Plus 属正常）")

    ret, state = info["E9_rm_plus_state"]
    if ret == 0:
        print(f"  E9 RM Plus 状态信息: {_format_payload(state)}")
    else:
        print(f"  E9 RM Plus 状态信息: N/A（ret={ret}，未安装 RM Plus 属正常）")

    # ── 异常汇总 ──
    print("\n=== 异常项 ===")
    if anomalies:
        for item in anomalies:
            print(f"  ⚠️ {item}")
    else:
        print("  无异常 ✅")


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────


def main() -> None:
    """主函数：连接 → 查询 E1-E9 → 打印系统信息快照 → 安全断开。"""
    parser = argparse.ArgumentParser(
        description="RM-65B 系统信息查询（零风险，只读）"
    )
    add_config_arg(parser)
    args = parser.parse_args()

    print("=" * 64)
    print("12_system_info_query — 系统信息查询")
    print("=" * 64)

    ri, _config = load_and_connect(args.config)
    arm = ri._arm

    try:
        print("\n[1/1] 查询系统信息（E1-E9）")
        info = query_system_info(arm)
        print_system_info(info)
    finally:
        safe_disconnect(ri)


if __name__ == "__main__":
    main()
