"""V12 唯一任务配置解析层（兼容 facade，B1 迁移后）。

解析实现已按职责迁移到 :mod:`src.sim.configuration`：

- :mod:`src.sim.configuration.types`：``ResolvedV12Config`` 共享类型；
- :mod:`src.sim.configuration.validation`：任务覆盖白名单与类型校验；
- :mod:`src.sim.configuration.defaults`：CLI 缺省值、配置来源合并与基线值；
- :mod:`src.sim.configuration.selection`：机器人选择与 R1D 显式注册表；
- :mod:`src.sim.configuration.overrides`：显式选项归一化与覆盖应用；
- :mod:`src.sim.configuration.resolution`：仓库根常量、收尾派生与唯一入口。

本模块只做同名对象重导出，保证旧导入路径不变：``__all__`` 与迁移前一致，
``is`` 相同，``__module__`` 指向新模块。私有名 ``_OVERRIDE_VALIDATORS``
仍在此重导出，供 ``scripts/exp/run_r1d_paper_smoke.py`` 等旧调用点使用。
"""

from __future__ import annotations

from src.sim.configuration.overrides import (
    normalize_explicit_options as normalize_explicit_options,  # noqa: PLC0414
)
from src.sim.configuration.resolution import (
    resolve_v12_config as resolve_v12_config,  # noqa: PLC0414
)
from src.sim.configuration.selection import (
    R1D_V12_REGISTRY as R1D_V12_REGISTRY,  # noqa: PLC0414
)
from src.sim.configuration.selection import (
    R1DConfigRegistry as R1DConfigRegistry,  # noqa: PLC0414
)
from src.sim.configuration.types import (
    ResolvedV12Config as ResolvedV12Config,  # noqa: PLC0414
)
from src.sim.configuration.types import _ConfigDict as _ConfigDict  # noqa: PLC0414
from src.sim.configuration.validation import (
    _OVERRIDE_VALIDATORS as _OVERRIDE_VALIDATORS,  # noqa: PLC0414
)

__all__ = [
    "R1D_V12_REGISTRY",
    "R1DConfigRegistry",
    "ResolvedV12Config",
    "normalize_explicit_options",
    "resolve_v12_config",
]
