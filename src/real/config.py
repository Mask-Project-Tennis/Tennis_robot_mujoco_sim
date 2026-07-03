"""真机部署配置。

从 YAML 加载或直接构造，包含机器人连接、控制模式、安全参数等。
YAML 中关节角度用「度」（与 SDK 一致），config.py 自动转弧度。
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RealRobotConfig:
    """真机部署配置（从 YAML 加载或直接构造）。

    YAML 中 q_lower/q_lower 单位为度，加载时自动转为弧度。
    其余角度参数（如 max_qdot）已标注单位，不自动转换。
    """

    # ── [1] 机器人连接 ──
    robot_ip: str = "192.168.1.18"
    robot_port: int = 8080

    # ── [2] 控制参数 ──
    dt: float = 0.005
    target_hit_speed: float = 1.8

    # ── [3] 控制器安全配置（Layer 1，连接时下发控制器固件）──
    collision_stage: int = 5
    enable_self_collision: bool = True
    enable_singularity_avoidance: bool = True
    torque_limit: list[float] = field(
        default_factory=lambda: [50.0, 50.0, 50.0, 30.0, 30.0, 30.0]
    )
    max_tcp_speed: float = 1.0
    max_line_acc: float = 0.5
    max_angular_speed: float = 0.2
    max_angular_acc: float = 1.0

    # ── SafetyMonitor 软件检查（Layer 2）──
    max_qdot: np.ndarray = field(
        default_factory=lambda: np.full(6, 3.14)
    )
    q_lower: np.ndarray = field(
        default_factory=lambda: np.radians(
            np.array([-175.0, -270.0, -130.0, -175.0, -115.0, -180.0])
        )
    )
    q_upper: np.ndarray = field(
        default_factory=lambda: np.radians(
            np.array([175.0, 90.0, 130.0, 175.0, 115.0, 180.0])
        )
    )

    # ── [4] 位置模式 PD 参数 ──
    kp: list[float] = field(
        default_factory=lambda: [200.0, 200.0, 100.0, 50.0, 50.0, 20.0]
    )
    kd: list[float] = field(
        default_factory=lambda: [20.0, 20.0, 10.0, 5.0, 5.0, 2.0]
    )
    enable_feedforward: bool = True

    # ── [5] 力矩→位置转换器 ──
    M_diag: np.ndarray = field(
        default_factory=lambda: np.array([5.0, 5.0, 3.0, 1.0, 1.0, 0.5])
    )

    # ── [6] 关节零位偏移 ──
    joint_zero_offset: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )

    # ── [7] 感知配置 ──
    sensor_type: str = "simulated"
    pos_noise_std: float = 0.005
    vel_noise_std: float = 0.1

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

    # YAML section 内短键名 → dataclass 字段名的映射
    _YAML_ALIASES = {"ip": "robot_ip", "port": "robot_port"}

    @staticmethod
    def _flatten(data: dict[str, Any]) -> dict[str, Any]:
        """将嵌套 YAML 展平为 dataclass kwargs。

        处理三种 YAML 结构:
        1. 嵌套 section（dict）: 内层 key 经 _YAML_ALIASES 映射后匹配字段名
        2. 顶层标量/列表: 若 key 本身（或经别名映射）匹配字段名则直接采用
        3. 忽略所有不匹配的键

        q_lower/q_upper 在 YAML 中为度，此处自动转弧度。

        Args:
            data: 原始 YAML 字典（含 robot/control/safety 等嵌套节）。

        Returns:
            展平后的 kwargs 字典。
        """
        valid_names = {f.name for f in fields(RealRobotConfig)}
        ndarray_keys = {"max_qdot", "M_diag", "joint_zero_offset"}
        degree_keys = {"q_lower", "q_upper"}

        def _map_key(k: str) -> str:
            """YAML 短键名 → dataclass 字段名（无别名则原样返回）。"""
            return RealRobotConfig._YAML_ALIASES.get(k, k)

        def _store(kwargs: dict[str, Any], raw_key: str, value: Any) -> None:
            """尝试将 (raw_key, value) 存入 kwargs，处理别名/类型转换。"""
            k = _map_key(raw_key)
            if k not in valid_names:
                return
            if k in degree_keys and isinstance(value, list):
                kwargs[k] = np.radians(np.array(value, dtype=np.float64))
            elif k in ndarray_keys and isinstance(value, list):
                kwargs[k] = np.array(value, dtype=np.float64)
            else:
                kwargs[k] = value

        kwargs: dict[str, Any] = {}
        for top_key, top_value in data.items():
            if isinstance(top_value, dict):
                # 嵌套 section: 遍历内层 key-value
                for k, v in top_value.items():
                    _store(kwargs, k, v)
            else:
                # 顶层标量/列表: key 本身可能匹配字段名
                _store(kwargs, top_key, top_value)
        return kwargs
