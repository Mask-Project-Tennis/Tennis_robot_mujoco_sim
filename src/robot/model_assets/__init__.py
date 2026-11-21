"""模型资产层 — MJCF 依赖解析的稳定导入入口。

只重导出公有 API 与依赖角色常量：不导入 MuJoCo/NumPy、不加载模型、不做
文件 IO 与依赖扫描。私有 helper 由 ``references`` / ``dependencies`` 模块
直接提供；旧路径 ``src.robot.model_factory`` 继续可用（对象相同）。
"""

from src.robot.model_assets.dependencies import (
    collect_model_file_dependencies,
)
from src.robot.model_assets.references import (
    DEPENDENCY_ROLE_ARM_FRAGMENT,
    DEPENDENCY_ROLE_INCLUDE,
    DEPENDENCY_ROLE_MESH,
    DEPENDENCY_ROLE_MODEL,
    DEPENDENCY_ROLE_TEXTURE,
    DEPENDENCY_XML_ROLES,
    mask_xml_comments,
)

__all__ = [
    "DEPENDENCY_ROLE_ARM_FRAGMENT",
    "DEPENDENCY_ROLE_INCLUDE",
    "DEPENDENCY_ROLE_MESH",
    "DEPENDENCY_ROLE_MODEL",
    "DEPENDENCY_ROLE_TEXTURE",
    "DEPENDENCY_XML_ROLES",
    "collect_model_file_dependencies",
    "mask_xml_comments",
]
