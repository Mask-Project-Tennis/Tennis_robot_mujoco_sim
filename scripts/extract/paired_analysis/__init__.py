"""论文配对统计子包（C2）。

轻量导出只覆盖 ``types`` 的共享类型/常量；数据读取、指标与报告模块按需从子模块导入，
避免 ``import scripts.extract.paired_analysis`` 时拉起 pandas 或读取数据。
旧入口 ``scripts.extract.paired_stats`` 仍是完整兼容 facade。
"""
from __future__ import annotations

from .types import *
from .types import __all__ as _types_all

__all__ = list(_types_all)
