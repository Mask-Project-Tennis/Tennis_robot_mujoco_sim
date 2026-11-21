"""机器人模型构建层（P1：仅共享类型）。

当前只导出 :class:`RacketAssembly` 与 :class:`RobotModelBundle` 的唯一类定义；
模型加载等实现仍在 ``src.robot.model_factory``，后续 A1/A2 迁入本包。

本包初始化不加载 MuJoCo 模型、不读配置文件、不 import 旧 facade。
"""

from src.robot.model_building.types import RacketAssembly, RobotModelBundle

__all__ = ["RacketAssembly", "RobotModelBundle"]
