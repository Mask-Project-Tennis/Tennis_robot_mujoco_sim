"""_connect.py — 真机测试公共模块。

所有测试脚本 import 此模块获取连接和安全预检功能。
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.real.config import RealRobotConfig
from src.real.robot_interface import RobotInterface
from src.real.safety_monitor import SafetyMonitor

# 测试默认配置路径（保守参数）
_TEST_CONFIG = _PROJECT_ROOT / "configs" / "real_robot_test.yaml"
# 生产配置路径（回退用）
_PROD_CONFIG = _PROJECT_ROOT / "configs" / "real_robot.yaml"


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    """为 argparse 添加 --config 参数。

    默认路径: configs/real_robot_test.yaml（测试保守参数）。
    若测试配置不存在则回退到 configs/real_robot.yaml。

    Args:
        parser: argparse.ArgumentParser 实例。
    """
    default_path = str(_TEST_CONFIG) if _TEST_CONFIG.exists() else str(_PROD_CONFIG)
    parser.add_argument(
        "--config",
        type=str,
        default=default_path,
        help=(
            "真机配置 YAML 路径（默认: configs/real_robot_test.yaml，"
            "不存在时回退 real_robot.yaml）"
        ),
    )


def add_algo_check_arg(parser: argparse.ArgumentParser) -> None:
    """为 argparse 添加 --no-algo-check 参数。

    Args:
        parser: argparse.ArgumentParser 实例。
    """
    parser.add_argument(
        "--no-algo-check",
        action="store_true",
        default=False,
        help="跳过 SDK Algo 类自碰撞/奇异性检查（仅保留 SafetyMonitor 限位检查）",
    )


def load_config(config_path: str) -> RealRobotConfig:
    """加载 YAML 配置。

    Args:
        config_path: YAML 文件路径。

    Returns:
        RealRobotConfig 实例。
    """
    path = Path(config_path)
    if not path.exists():
        print(f"❌ 配置文件不存在: {path}")
        sys.exit(1)
    return RealRobotConfig.from_yaml(path)


def load_and_connect(config_path: str = None) -> tuple[RobotInterface, RealRobotConfig]:
    """加载配置，连接真机，返回 (RobotInterface, config)。

    连接失败时打印错误并退出（不返回 None）。

    Args:
        config_path: YAML 路径。None 时用默认测试配置（回退生产配置）。

    Returns:
        (RobotInterface, RealRobotConfig)
    """
    if config_path is None:
        if _TEST_CONFIG.exists():
            config_path = str(_TEST_CONFIG)
        else:
            config_path = str(_PROD_CONFIG)

    config = load_config(config_path)
    print(f"正在连接 {config.robot_ip}:{config.robot_port} ...")

    ri = RobotInterface(config)
    if not ri.connect():
        print("❌ 连接失败，请检查:")
        print(f"  - 机械臂 IP 地址: {config.robot_ip}")
        print("  - 网线是否连接")
        print("  - Realman SDK 是否安装: pip install Robotic_Arm")
        sys.exit(1)

    print(f"✅ 连接成功 {config.robot_ip}:{config.robot_port}")
    return ri, config


def init_algo():
    """初始化 SDK Algo 类（用于自碰撞/奇异性预检）。

    配置球拍工具包络球（粗略估计，后续精调）。

    Returns:
        Algo 实例，或 None（SDK 不可用时）。
    """
    try:
        from Robotic_Arm.rm_robot_interface import (
            rm_robot_arm_model_e,
            rm_force_type_e,
        )

        # 尝试导入 Algo 类（不同 SDK 版本路径可能不同）
        try:
            from Robotic_Arm.rm_robot_interface import Algo
        except ImportError:
            return None

        algo = Algo(rm_robot_arm_model_e.RM_MODEL_RM_65_E, rm_force_type_e.RM_MODEL_RM_B_E)

        # 配置球拍工具包络球
        # 球拍: 手柄长 25cm + 拍面 12cm，沿 Z 轴
        # 球 0: 法兰中心，球 1: 手柄中部，球 2: 拍面中心
        # ponytail: radius 0.07 ≈ 拍面半宽，包络略保守但不至于 2x 过大导致误报
        try:
            from Robotic_Arm.rm_robot_interface import rm_tool_sphere_t
            spheres = [
                rm_tool_sphere_t(centrePoint=(0, 0, 0.0), radius=0.03),
                rm_tool_sphere_t(centrePoint=(0, 0, 0.12), radius=0.02),
                rm_tool_sphere_t(centrePoint=(0, 0, 0.25), radius=0.07),
            ]
            for i, s in enumerate(spheres):
                algo.rm_algo_set_tool_envelope(i, s)
        except Exception as e:
            logging.warning("包络球配置失败（自碰撞检测将仅保护法兰区域）: %s", e)

        return algo
    except ImportError:
        return None


def pre_motion_check(
    ri: RobotInterface,
    monitor: SafetyMonitor,
    q_desired: np.ndarray,
    arm_state: np.ndarray = None,
    algo=None,
    n_path_samples: int = 15,
    check_singularity: bool = False,
) -> tuple[bool, str]:
    """运动前安全预检。

    检查项（按顺序）:
      1. SafetyMonitor 限位检查（关节位置 + 关节速度）
      2. SDK Algo 自碰撞检测（沿插值路径采样 n_path_samples 个点）
      3. SDK Algo 奇异性检测（仅 check_singularity=True 时）

    采样策略：从当前位姿到目标位姿均匀插值（含两端），
    任一采样点触发碰撞即拒绝。避免只查终点导致误判。

    奇异性默认关闭的原因：测试脚本使用 rm_movej_follow（关节空间运动），
    不经过逆运动学，奇异性不影响控制。控制器固件 Layer 1 的奇异性规避
    （rm_set_avoid_singularity_mode）始终独立生效，与预检无关。

    TCP 速度限制由控制器固件 Layer 1（rm_set_arm_max_line_speed）强制执行，
    不在预检中重复检查。

    Args:
        ri: RobotInterface 实例。
        monitor: SafetyMonitor 实例。
        q_desired: (6,) 目标关节角度，弧度。
        arm_state: (12,) 当前臂状态 [q, qdot]。None 时从 ri 读取。
        algo: Algo 实例。None 时跳过碰撞/奇异性检查。
        n_path_samples: 路径采样点数（含两端，默认 15）。
        check_singularity: 是否检测奇异性（默认 False，仅笛卡尔空间运动需要）。

    Returns:
        (is_safe, message) — 是否通过，原因说明。
    """
    if arm_state is None:
        arm_state = ri.get_arm_state()

    # 1. SafetyMonitor 限位检查（关节位置 + 关节速度）
    if not monitor.is_safe(arm_state, q_desired):
        return False, "❌ 限位检查未通过（关节位置/速度超限）"

    # 2. SDK Algo 自碰撞检测（路径采样）；奇异性仅按需检查
    if algo is not None:
        q_current = arm_state[:6]
        for alpha in np.linspace(0.0, 1.0, n_path_samples):
            q_sample = q_current * (1 - alpha) + q_desired * alpha
            q_deg = np.degrees(q_sample).tolist()

            # 自碰撞检测（始终执行）
            try:
                ret = algo.rm_algo_safety_robot_self_collision_detection(q_deg)
                if ret == 1:
                    pct = alpha * 100
                    return False, f"❌ 自碰撞风险（路径 {pct:.0f}% 处）"
            except Exception:
                pass

            # 奇异性检测（仅笛卡尔空间运动需要）
            if check_singularity:
                try:
                    ret, dist = algo.rm_algo_kin_robot_singularity_analyse(q_deg)
                    if ret != 0:
                        codes = {0: "正常", -1: "肩部奇异", -2: "肘部奇异", -3: "腕部奇异"}
                        pct = alpha * 100
                        return False, f"❌ 奇异性风险 ({codes.get(ret, f'code={ret}')}, 路径 {pct:.0f}% 处)"
                except Exception:
                    pass

    return True, "✅ 预检通过"


def home_to_pose(
    ri: RobotInterface,
    monitor: SafetyMonitor,
    algo: Any,
    q_target: np.ndarray,
    duration: float = 1.0,
    hz: float = 100.0,
    n_path_samples: int = 15,
) -> None:
    """流式线性插值回到目标位姿。

    从当前关节角度流式插值到目标角度，内置安全预检。
    不通过预检时抛 SystemExit。

    Args:
        ri: RobotInterface 实例。
        monitor: SafetyMonitor 实例。
        algo: Algo 实例或 None。
        q_target: (6,) 目标关节角度，弧度。
        duration: 插值总时长（秒），默认 1.0。
        hz: 发送频率（Hz），默认 100.0。
        n_path_samples: 预检路径采样点数，默认 15。

    Raises:
        SystemExit: 预检不通过时。
    """
    arm_state = ri.get_arm_state()
    q_current = arm_state[:6].copy()

    ok, msg = pre_motion_check(ri, monitor, q_target, arm_state, algo, n_path_samples)
    if not ok:
        raise SystemExit(f"归位预检失败: {msg}")

    dt = 1.0 / hz
    n_steps = int(duration * hz)

    try:
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            q = q_current * (1 - alpha) + q_target * alpha
            ri.send_joint_command(q)
            time.sleep(dt)

        ri.send_joint_command(q_target)
        time.sleep(0.2)
    except KeyboardInterrupt:
        ri.slow_stop()
        raise

    final_state = ri.get_arm_state()
    q_final = final_state[:6]
    error_deg = np.degrees(q_final - q_target)
    print(f"最终角度（度）: {np.degrees(q_final).round(2)}")
    print(f"跟踪误差（度）: {error_deg.round(2)}")
    print(f"最大误差: {np.max(np.abs(error_deg)):.2f}°")


def safe_disconnect(ri: RobotInterface) -> None:
    """安全断开连接（先缓停再断开）。"""
    try:
        ri.slow_stop()
    except Exception:
        pass
    ri.disconnect()
    print("已安全断开连接")
