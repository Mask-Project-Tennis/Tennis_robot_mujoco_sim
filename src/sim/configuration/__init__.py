"""V12 配置解析层（B1 迁移后）。

对外稳定导出五个公有名：``ResolvedV12Config``、``R1DConfigRegistry``、
``R1D_V12_REGISTRY``、``normalize_explicit_options``、``resolve_v12_config``。

本包初始化只 import 纯配置模块：不导入 MuJoCo、不读取 YAML、
不 import 旧 facade ``src.sim.v12_config``（避免循环依赖）。
"""

from src.sim.configuration.overrides import normalize_explicit_options
from src.sim.configuration.resolution import resolve_v12_config
from src.sim.configuration.selection import R1D_V12_REGISTRY, R1DConfigRegistry
from src.sim.configuration.types import ResolvedV12Config

__all__ = [
    "R1D_V12_REGISTRY",
    "R1DConfigRegistry",
    "ResolvedV12Config",
    "normalize_explicit_options",
    "resolve_v12_config",
]
