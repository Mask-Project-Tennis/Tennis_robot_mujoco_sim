"""src.robot 包公共入口 — 纯类型与解析函数的重导出（阶段 2 Task 2.8）。

导出策略（计划 L497-498 约定）：

- 只重导出**纯类型/解析函数**：RobotProfile / runtime_parameters 两模块
  不依赖 MuJoCo、ilqt、real、sim，直接（eager）重导出——因此
  ``import src.robot`` 不会隐式加载任何重型依赖；
- **模型工具保持显式模块导入，不进本 ``__init__``**：
  ``load_robot_model`` / ``RobotModelBundle``（src.robot.model_factory）、
  ``ModelKinematics``（src.robot.model_kinematics）仍需
  ``from src.robot.model_factory import load_robot_model`` 等显式路径；
- ``ModelBinding`` / ``bind_robot_model`` 所在的 model_binding 模块顶层
  导入 mujoco（标量关节类型枚举），故按 PEP 562 以模块级 ``__getattr__``
  延迟导出：仅在显式访问属性（如 ``from src.robot import ModelBinding``）
  时才加载 mujoco，``import src.robot`` 本身保持纯净。
"""

from __future__ import annotations

import importlib

from src.robot.robot_profile import (
    RobotProfile,
    default_rm65_profile_path,
    load_robot_profile,
)
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    SceneParameters,
    load_runtime_parameters,
    validate_runtime_parameters,
)

# 延迟导出表：属性名 → 所在子模块（子模块自身依赖 mujoco，按需加载）
_LAZY_EXPORTS: dict[str, str] = {
    "ModelBinding": "src.robot.model_binding",
    "bind_robot_model": "src.robot.model_binding",
}

# 显式导出的完整清单（eager 纯导出 + lazy 绑定导出）
__all__ = [
    "RobotProfile",
    "default_rm65_profile_path",
    "load_robot_profile",
    "RobotRuntimeParameters",
    "SceneParameters",
    "load_runtime_parameters",
    "validate_runtime_parameters",
    "ModelBinding",
    "bind_robot_model",
]


def __getattr__(name: str) -> object:
    """按需加载依赖 mujoco 的绑定层导出（PEP 562 模块级 __getattr__）。

    Args:
        name: 被访问的属性名。

    Returns:
        子模块中的同名对象（并缓存进全局命名空间，后续访问零开销）。

    Raises:
        AttributeError: 属性不在延迟导出表中（保持普通模块语义）。
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attr = getattr(importlib.import_module(module_name), name)
    globals()[name] = attr
    return attr
