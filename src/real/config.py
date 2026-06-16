"""真机部署配置。

从 YAML 加载或直接构造，包含机器人连接、控制模式、安全参数等。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RealRobotConfig:
    """真机部署配置（从 YAML 加载或直接构造）。

    Attributes:
        robot_ip: 机械臂 IP 地址。
        robot_port: 机械臂通信端口。
        control_mode: 控制模式，"ip"（默认，rm_movej_follow）或 "canfd"（rm_movej_canfd）。
        dt: MPC 规划步长（秒）。
        M_diag: 对角惯量近似 (6,)，用于力矩→位置转换。
        max_tcp_speed: 末端最大线速度（m/s），比仿真更保守。
        max_qdot: 关节最大角速度 (6,)（rad/s）。
        canfd_trajectory_mode: CANFD 轨迹模式（0=透传 1=曲线拟合 2=滤波）。
        canfd_smooth_radio: CANFD 平滑系数（0-100）。
        joint_zero_offset: 仿真 vs 真实关节零位偏移 (6,)（rad）。
    """

    # 机器人连接
    robot_ip: str = "192.168.1.18"
    robot_port: int = 8080

    # 控制参数
    control_mode: str = "ip"
    dt: float = 0.005

    # 力矩→位置转换器
    M_diag: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 5.0, 3.0, 1.0, 1.0, 0.5])
    )

    # 安全参数（比仿真更保守）
    max_tcp_speed: float = 1.0
    max_qdot: np.ndarray = field(
        default_factory=lambda: np.full(6, 3.14)
    )

    # CANFD 模式参数（control_mode="canfd" 时生效）
    canfd_trajectory_mode: int = 1
    canfd_smooth_radio: int = 50

    # 关节零位偏移（仿真 vs 真实）
    joint_zero_offset: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "RealRobotConfig":
        """从 YAML 文件加载配置。

        Args:
            path: YAML 文件路径。

        Returns:
            RealRobotConfig 实例。
        """
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)

        return cls(**cls._flatten(data))

    @staticmethod
    def _flatten(data: dict[str, Any]) -> dict[str, Any]:
        """将嵌套 YAML 展平为 dataclass kwargs。

        Args:
            data: 原始 YAML 字典（含 robot/control/safety 等嵌套节）。

        Returns:
            展平后的 kwargs 字典。
        """
        array_keys = {"max_qdot", "M_diag", "joint_zero_offset"}
        kwargs: dict[str, Any] = {}
        for section in data.values():
            if isinstance(section, dict):
                for k, v in section.items():
                    if k in array_keys and isinstance(v, list):
                        kwargs[k] = np.array(v)
                    else:
                        kwargs[k] = v
        return kwargs
