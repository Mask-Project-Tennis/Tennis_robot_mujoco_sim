"""R1D 论文 smoke 实验的 manifest/任务/续跑实现包。

轻量导出仅覆盖最常用的数据契约入口；CLI 与产物校验等重模块请按需从子模块导入，
避免 ``import scripts.exp.paper_smoke`` 时拉起 subprocess 或解析大文件。
"""

from __future__ import annotations

from .manifest import ManifestError, load_manifest, validate_manifest
from .tasks import build_specs_from_manifest, resolve_manifest

__all__ = [
    "ManifestError",
    "build_specs_from_manifest",
    "load_manifest",
    "resolve_manifest",
    "validate_manifest",
]
