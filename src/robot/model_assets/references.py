"""MJCF 依赖引用发现 — XML 注释掩码、路径候选与两阶段引用收集。

本模块只做**文件级引用发现**：按 MuJoCo 语义展开 ``<include>`` 树，收集
模型/include 文件条目、``<mesh>/<texture>`` 资产引用与 ``<compiler>``
目录表。不读取或校验 MuJoCo 模型、不计算哈希/快照条目（由
:mod:`src.robot.model_assets.dependencies` 负责），仅依赖标准库。

A1 迁移说明：实现自 ``src/robot/model_factory.py`` 原样迁出；旧导入路径
继续由 ``model_factory`` 显式重导出，``__module__`` 指向本模块。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# 依赖清单中的角色（写进快照条目，回放按角色决定物化方式）
DEPENDENCY_ROLE_MODEL = "model"
DEPENDENCY_ROLE_INCLUDE = "include"
DEPENDENCY_ROLE_ARM_FRAGMENT = "arm_fragment"
DEPENDENCY_ROLE_MESH = "mesh"
DEPENDENCY_ROLE_TEXTURE = "texture"

# 需要把内容写进快照的 XML 片段角色（网格/纹理体积大，只记哈希）
DEPENDENCY_XML_ROLES = frozenset(
    {
        DEPENDENCY_ROLE_MODEL,
        DEPENDENCY_ROLE_INCLUDE,
        DEPENDENCY_ROLE_ARM_FRAGMENT,
    }
)

# 资产目录属性：``<compiler meshdir>`` 管 mesh，``<compiler texturedir>`` 管 texture
_ASSET_DIR_ATTRS: dict[str, str] = {
    DEPENDENCY_ROLE_MESH: "meshdir",
    DEPENDENCY_ROLE_TEXTURE: "texturedir",
}

# MJCF 依赖解析用的正则（与 src/sim/episode_artifacts.py 的资产扫描同源：
# 只按真实解析路径收集，不做无边界递归扫描）。用一个"多标签"正则是为了
# 拿到**真正的文档顺序**——compiler 的全局覆盖语义依赖这个顺序。
_MJCF_TAG_PATTERN = re.compile(
    r"<(?P<tag>compiler|include|mesh|texture)\b(?P<attrs>[^>]*?)/?>",
    re.IGNORECASE,
)

# XML 注释（MuJoCo 解析器会整体丢弃；正则扫描前必须先掩码，见 R03）
_XML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)

# include 展开的最大深度（防止环状 include 无界递归）
_MAX_INCLUDE_DEPTH = 8


class _EntryBuilder(Protocol):
    """构造一条依赖条目的可调用对象（依赖方向注入，避免循环导入）。

    ``references`` 只负责"发现了哪些引用"，条目中的哈希/相对路径等字段由
    ``dependencies`` 层决定，因此两阶段扫描通过本回调产出条目。
    """

    def __call__(
        self,
        role: str,
        path: Path,
        *,
        reference: str = "",
    ) -> dict[str, Any]: ...


def mask_xml_comments(text: str) -> str:
    """把 XML 注释覆盖为等长空白掩码（保留字符位置与换行）。

    MuJoCo 的 MJCF 解析器会**整体丢弃** XML 注释，但直接对原文做正则扫描会
    把注释里的 ``<compiler>``/``<include>``/``<mesh>`` 当成真实标签（R03
    根因：注释里的诱饵 ``meshdir`` 被合并，回放网格被放大）。本函数用空白
    覆盖注释区间，使掩码文本与原文**逐字符等长**：``finditer`` 得到的 span
    可直接映射回原文定位/替换，注释内容在原文中保持不动。

    局限：按 XML 词法假定注释不会出现在属性值内部（``file="a<!--b-->c"``
    这类病态写法不在支持范围）。

    Args:
        text: 原始 XML 文本。

    Returns:
        与 ``text`` 等长的掩码文本（仅注释区间变成空白，换行保留）。
    """

    def _blank(match: re.Match[str]) -> str:
        """把一段注释替换为同长度的空白（换行不吞掉）。"""
        return "".join("\n" if char == "\n" else " " for char in match.group(0))

    return _XML_COMMENT_PATTERN.sub(_blank, text)


def _attribute_value(attrs: str, name: str) -> str | None:
    """取出 XML 标签属性值（单/双引号均可），未声明时返回 ``None``。

    Args:
        attrs: 标签内的属性串（``<tag`` 与 ``/>`` 之间的部分）。
        name: 属性名（大小写不敏感）。

    Returns:
        去掉引号后的原始取值；属性不存在时返回 ``None``——与"显式声明空
        值"（返回空串）区分，后者在 compiler 覆盖语义里是有效重置。
    """
    match = re.search(
        rf"""\b{name}\s*=\s*(?:"([^"]*)"|'([^']*)')""",
        attrs,
        re.IGNORECASE,
    )
    if match is None:
        return None
    value = match.group(1)
    if value is None:
        value = match.group(2) or ""
    return value


def _resolve_reference_path(base_dir: Path, dir_prefix: str, raw: str) -> Path:
    """把 ``file=`` 引用解析为绝对路径（相对基准 = ``base_dir`` 叠加前缀目录）。

    绝对路径直接使用；相对路径先拼目录前缀（``meshdir``/``texturedir``）。

    Args:
        base_dir: 相对基准目录。
        dir_prefix: 目录前缀（空串表示无前缀；本身可为绝对路径）。
        raw: 原始引用字符串。

    Returns:
        解析后的绝对路径（不保证文件存在）。
    """
    reference = Path(raw.strip().replace("\\", "/"))
    if reference.is_absolute():
        return reference.resolve()
    base = (base_dir / dir_prefix).resolve() if dir_prefix else base_dir
    return (base / reference).resolve()


@dataclass(slots=True)
class _AssetReference:
    """一条待解析的 mesh/texture 引用（第一阶段收集，第二阶段统一解析）。

    Attributes:
        kind: 引用标签（``mesh`` 或 ``texture``）。
        raw: ``file=`` 原始字符串（已去首尾空白）。
        declaring_file: 直接写该引用的 XML 文件绝对路径（候选 2 的基准目录）。
    """

    kind: str
    raw: str
    declaring_file: Path


def _asset_candidate_paths(
    reference: _AssetReference,
    *,
    model_dir: Path,
    asset_dirs: Mapping[str, str],
) -> tuple[Path, Path]:
    """按 MuJoCo 语义给出资产的两个候选路径（顺序即查找顺序）。

    候选 1 = ``<顶层模型目录>/<全局 meshdir|texturedir>/<file>``（目录前缀
    为绝对路径时直接生效）；候选 2 = ``<声明该引用的 XML 文件目录>/<file>``
    （**不再拼目录前缀**，与 MuJoCo 的兜底查找一致）。绝对路径引用两个候选
    相同。

    Args:
        reference: 待解析的资产引用。
        model_dir: 顶层模型文件所在目录（全局目录前缀的基准，不是声明目录）。
        asset_dirs: 最终全局 ``{kind: 目录前缀}`` 表。

    Returns:
        ``(第一候选, 第二候选)``。
    """
    prefix = asset_dirs.get(reference.kind, "")
    first = _resolve_reference_path(model_dir, prefix, reference.raw)
    second = _resolve_reference_path(reference.declaring_file.parent, "", reference.raw)
    return first, second


def _include_candidate_paths(
    raw: str,
    *,
    model_dir: Path,
    declaring_dir: Path,
) -> tuple[Path, Path]:
    """按 MuJoCo 语义给出 ``<include file>`` 的两个候选路径。

    实测（3.9.0）：include 先按**顶层模型目录**解析，不存在时才回退到
    **声明该 include 的文件目录**——与资产的两候选顺序一致。绝对路径引用
    两个候选相同。

    Args:
        raw: ``file=`` 原始字符串。
        model_dir: 顶层模型文件所在目录（第一候选基准）。
        declaring_dir: 声明该 include 的 XML 文件目录（第二候选基准）。

    Returns:
        ``(顶层候选, 声明目录候选)``。
    """
    first = _resolve_reference_path(model_dir, "", raw)
    second = _resolve_reference_path(declaring_dir, "", raw)
    return first, second


def _walk_model_file(
    path: Path,
    *,
    role: str,
    reference: str,
    model_dir: Path,
    entries: list[dict[str, Any]],
    asset_refs: list[_AssetReference],
    asset_dirs: dict[str, str],
    missing: list[str],
    seen: set[Path],
    depth: int,
    build_entry: _EntryBuilder,
) -> None:
    """第一阶段：按文档顺序展开 MJCF，收集文件条目、资产引用与全局编译目录。

    与旧实现的关键差异：

    * 扫描前先用 :func:`mask_xml_comments` 掩码注释——注释里的标签不生效；
    * ``<compiler>`` 在这里**只记录不解析**——它是全局设置，include 片段里
      的声明同样生效，同名属性按文档顺序后者覆盖前者；``assetdir`` 是
      ``meshdir``/``texturedir`` 的别名（同一标签内显式属性覆盖它）；
    * ``<include>`` 先按顶层模型目录解析，不存在再回退声明目录；
    * 资产路径一律留给第二阶段用最终目录表统一解析。MuJoCo 同样是在解析完
      整个文档后才加载资产，因此这个顺序与真实加载行为一致。

    Args:
        path: 待展开的 XML 文件。
        role: 该文件的角色（首层 ``model``，嵌套层 ``include``）。
        reference: 上层引用该文件的原始字符串（首层为空）。
        model_dir: 顶层模型文件所在目录（include 第一候选的基准，递归不变）。
        entries: 输出文件条目列表（原地追加，按首次访问去重）。
        asset_refs: 输出资产引用列表（原地追加，保持文档顺序）。
        asset_dirs: 全局 ``{kind: 目录前缀}`` 表（原地更新，后者覆盖）。
        missing: 输出缺失引用列表（原地追加；含 include 与 XML 文件缺失）。
        seen: 已访问文件集合（防重复/环）。
        depth: 当前 include 深度（超过 :data:`_MAX_INCLUDE_DEPTH` 直接报错）。
        build_entry: 构造依赖条目字典的回调（由 ``dependencies.py`` 注入，
            避免本层反向依赖哈希/相对路径实现）。
    """
    resolved = Path(path).resolve()
    if resolved in seen:
        return
    if depth > _MAX_INCLUDE_DEPTH:
        raise ValueError(
            f"模型 include 嵌套超过 {_MAX_INCLUDE_DEPTH} 层，拒绝继续展开"
            f"（可能是环状 include）：{resolved}"
        )
    seen.add(resolved)
    if not resolved.is_file():
        missing.append(str(resolved))
        return
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError:
        missing.append(str(resolved))
        return
    entries.append(build_entry(role, resolved, reference=reference))
    # 注释整体忽略：注释里的 compiler/include/mesh 都不参与解析（R03）
    masked = mask_xml_comments(text)
    for match in _MJCF_TAG_PATTERN.finditer(masked):
        tag = str(match.group("tag")).lower()
        attrs = match.group("attrs")
        if tag == "compiler":
            # 全局编译设置：逐属性合并，未声明的属性保留上一条取值。
            # assetdir 同时设置 meshdir/texturedir，同一标签内的显式属性
            # 覆盖它（与属性书写顺序无关）；跨标签按文档顺序后者覆盖前者。
            assetdir = _attribute_value(attrs, "assetdir")
            if assetdir is not None:
                for kind in _ASSET_DIR_ATTRS:
                    asset_dirs[kind] = assetdir.strip()
            for kind, attr_name in _ASSET_DIR_ATTRS.items():
                value = _attribute_value(attrs, attr_name)
                if value is not None:
                    asset_dirs[kind] = value.strip()
            continue
        raw = (_attribute_value(attrs, "file") or "").strip()
        if not raw:
            continue
        if tag == "include":
            # include：顶层模型目录优先，声明目录兜底（R04，实测 3.9.0）
            first, second = _include_candidate_paths(
                raw,
                model_dir=model_dir,
                declaring_dir=resolved.parent,
            )
            target = first if first.is_file() else second
            _walk_model_file(
                target,
                role=DEPENDENCY_ROLE_INCLUDE,
                reference=raw,
                model_dir=model_dir,
                entries=entries,
                asset_refs=asset_refs,
                asset_dirs=asset_dirs,
                missing=missing,
                seen=seen,
                depth=depth + 1,
                build_entry=build_entry,
            )
            continue
        asset_refs.append(
            _AssetReference(kind=tag, raw=raw, declaring_file=resolved)
        )
