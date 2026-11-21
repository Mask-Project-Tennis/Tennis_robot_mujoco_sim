"""MJCF 依赖闭包 — 稳定相对路径、SHA256 与依赖清单条目构造。

承接 :mod:`src.robot.model_assets.references` 的引用发现结果，解析资产
候选并生成产物快照所需条目；同时提供模型文件级闭包入口
:func:`collect_model_file_dependencies`。本模块不加载 MuJoCo 模型、不装配
模型，依赖方向仅 ``references -> dependencies -> __init__``。

A1 迁移说明：实现自 ``src/robot/model_factory.py`` 原样迁出。仓库根改为由
本文件位置上溯四级（``dependencies.py -> model_assets -> robot -> src ->
仓库根``），修复原先按 ``model_factory.py`` 上溯两级写死的路径深度。
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

from src.robot.model_assets.references import (
    DEPENDENCY_ROLE_MODEL,
    DEPENDENCY_XML_ROLES,
    _asset_candidate_paths,
    _AssetReference,
    _walk_model_file,
)


def _dependency_repo_root() -> Path:
    """仓库根目录（``src/robot/model_assets/dependencies.py`` 的上四级）。"""
    return Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    """按块读取文件并返回 SHA256 十六进制摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_relative_path(source: Path) -> str:
    """依赖文件的稳定相对路径（仓库内保留相对路径，仓库外 ``external/<sha12>/``）。"""
    resolved = Path(source).resolve()
    try:
        return resolved.relative_to(_dependency_repo_root()).as_posix()
    except ValueError:
        digest = _sha256_file(resolved)[:12] if resolved.is_file() else "missing"
        return f"external/{digest}/{resolved.name}"


def _dependency_entry(
    role: str,
    path: Path,
    *,
    reference: str = "",
) -> dict[str, Any]:
    """构造一条依赖条目（XML 片段附带 base64 内容，网格只记哈希）。

    Args:
        role: 依赖角色（``model`` / ``include`` / ``arm_fragment`` /
            ``mesh`` / ``texture``）。
        path: 依赖文件绝对路径（须存在）。
        reference: 模型 XML 中引用该文件的原始字符串（网格/includes 用；
            回放据此把 ``file=`` 属性绑定到已校验条目，而非按路径猜）。

    Returns:
        ``role/relative_path/source/sha256/size``（XML 片段另含
        ``content_base64``；``reference`` 非空时一并记录）。
    """
    resolved = Path(path).resolve()
    data = resolved.read_bytes()
    entry: dict[str, Any] = {
        "role": role,
        "relative_path": _dependency_relative_path(resolved),
        "source": str(resolved),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }
    if reference:
        entry["reference"] = reference
    if role in DEPENDENCY_XML_ROLES:
        # F08：XML 片段必须把内容写进快照，回放才能在外部来源缺失时
        # 重建同一装配输入（网格体积大，仍按哈希从仓库/外部来源解析）。
        entry["content_base64"] = base64.b64encode(data).decode("ascii")
    return entry


def collect_model_file_dependencies(
    model_path: Path,
    *,
    asset_missing: list[str] | None = None,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    """收集一个 MJCF 模型的**文件级**依赖闭包（模型/嵌套 include/网格纹理）。

    两阶段实现（对齐 MuJoCo 语义）：

    1. 按文档顺序展开 include 树，收集文件条目、每条 ``<mesh>/<texture>``
       引用（含声明目录）与**最终** ``<compiler>`` 目录表——compiler 是
       全局设置，include 片段里的声明同样生效，同名属性后者覆盖前者，
       ``assetdir`` 同时设置 meshdir/texturedir；注释整体忽略。include
       自身先按顶层模型目录解析、再回退声明目录。
    2. 用最终目录表统一解析资产：先 ``<顶层目录>/<全局 dir>/<file>``，不
       存在再 ``<声明目录>/<file>``；绝对路径直接使用。缺失时报告第一候选。

    模型条目会附带可选 ``asset_dirs``（最终 meshdir/texturedir），供回放
    物化时按同一语义重写资产引用（旧产物无该字段时回退旧解析）。

    Args:
        model_path: 模型 XML 路径。
        asset_missing: 可选的输出列表；传入时把**资产引用**的缺失路径追加
            进去（``missing`` 返回值仍混合 include 缺失，供整体闭包报告）。

    Returns:
        ``(entries, missing)``：``entries`` 为按
        ``(role, relative_path)`` 排序的依赖条目元组；``missing`` 为无法
        读取的引用路径元组（不抛异常——缺失由产物 ``assets.missing`` 与
        回放校验统一报告）。

    Raises:
        ValueError: include 嵌套超过最大深度（防环状 include，见 ``references._MAX_INCLUDE_DEPTH``）。
    """
    model_file = Path(model_path).resolve()
    entries: list[dict[str, Any]] = []
    asset_refs: list[_AssetReference] = []
    asset_dirs: dict[str, str] = {}
    missing: list[str] = []
    seen: set[Path] = set()
    _walk_model_file(
        model_file,
        role=DEPENDENCY_ROLE_MODEL,
        reference="",
        model_dir=model_file.parent,
        entries=entries,
        asset_refs=asset_refs,
        asset_dirs=asset_dirs,
        missing=missing,
        seen=seen,
        depth=0,
        build_entry=_dependency_entry,
    )
    seen_assets: set[str] = set()
    for reference in asset_refs:
        first, second = _asset_candidate_paths(
            reference,
            model_dir=model_file.parent,
            asset_dirs=asset_dirs,
        )
        target = first if first.is_file() else second
        if not target.is_file():
            # 缺失时报告第一候选路径（与 MuJoCo 的报错口径一致）
            if str(first) not in seen_assets:
                seen_assets.add(str(first))
                missing.append(str(first))
                if asset_missing is not None:
                    asset_missing.append(str(first))
            continue
        key = str(target)
        if key in seen_assets:
            continue
        seen_assets.add(key)
        entries.append(_dependency_entry(reference.kind, target, reference=reference.raw))
    if asset_dirs:
        for entry in entries:
            if entry["role"] == DEPENDENCY_ROLE_MODEL:
                entry["asset_dirs"] = dict(asset_dirs)
                break
    ordered = tuple(sorted(entries, key=lambda item: (item["role"], item["relative_path"])))
    return ordered, tuple(missing)
