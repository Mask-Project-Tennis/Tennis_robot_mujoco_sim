"""M01/M05 回归：嵌套 include 的全局 meshdir 与快照资产校验。

本文件覆盖并行修复计划 §2 的两处问题：

* **M01**：``<compiler>`` 的 ``meshdir``/``texturedir`` 是全局设置，
  ``<include>`` 片段里的声明同样生效且基准是**顶层模型目录**。旧实现按
  "每个文件自己的目录 + 本文件第一个 ``<compiler>``" 解析资产，导致嵌套
  ``include`` 优先命中 include 目录下的同名诱饵文件，回放网格被放大 1.25×；
* **M05**：旧 ``assets.files`` 摘要校验只查仓库路径与记录的 ``source``，
  忽略写入时复制到快照内的副本；换机或清理旧检出后即使快照自包含也会
  被拒绝回放。

用例覆盖两种布局（同名错误文件存在 / 只有正确 ``meshdir`` 文件存在）、
缺资产时的第一候选报告、外部 STL 快照后删除来源、篡改快照仍拒绝、旧产物
没有 ``asset_dirs`` 时的回退解析，以及 ``assets.files`` 字段与聚合摘要不
回归。全部夹具都在 ``tmp_path`` 的 ASCII 路径下构造，不写
``experiment_data/``，也不修改任何既有测试文件。

另按第二轮复审 P2 补充 R03/R04 语义（以 MuJoCo 3.9.0 实测为准）：

* **R03**：``<!-- ... -->`` 内的 ``<compiler>``/``<mesh>``/``<include>``
  完全忽略——既不参与全局目录合并，也不被收集、不在快照改写时碰触；
* **R04**：``<include>`` 先按**顶层模型目录**解析，找不到才回退声明文件
  目录；``<compiler assetdir>`` 同时设置 mesh/texture 目录，同标签显式
  ``meshdir``/``texturedir`` 覆盖它，跨标签按文档顺序后者生效。
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for _path in (ROOT, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

import test_episode_asset_closure as closure  # noqa: E402
from src.robot.model_factory import collect_model_file_dependencies  # noqa: E402
from src.sim import episode_artifacts as artifacts  # noqa: E402
from src.utils.mujoco_loader import load_mujoco_model  # noqa: E402

# 与 RM65 真实模型一致的一条网格：移进 include 片段后由全局 meshdir 定位
_MESH_NAME = "base_link_underpan"
_MESH_FILE = "base_link_underpan.STL"
_MESHDIR = "meshes"
# 评审复现的放大倍数：诱饵文件（同名、位于 include 目录）是正确网格的 1.25×
_DECOY_SCALE = 1.25
# assets.files 条目字段（并行修复计划 §0 冻结，不得增删）
_ASSET_FILE_FIELDS = {
    "role",
    "kind",
    "source",
    "relative_path",
    "sha256",
    "size",
}


@dataclass(slots=True)
class _MeshdirCase:
    """嵌套 include + 全局 ``meshdir`` 的 RM65 夹具。

    Attributes:
        model_path: 顶层模型 XML（``<asset>`` 内只剩一条 include）。
        correct_mesh: ``<顶层目录>/<meshdir>/<file>`` 正确网格。
        decoy_mesh: ``<include 目录>/<file>`` 同名诱饵（可选）；旧实现会选它。
        external_dir: 外部资产目录（仅 ``external`` 布局使用）。
        output_dir: 产物目录。
    """

    model_path: Path
    correct_mesh: Path
    decoy_mesh: Path | None
    external_dir: Path | None
    output_dir: Path


def _mirrored_model_text() -> str:
    """真实 RM65 模型文本，绝对资产改指 ASCII 镜像（Windows 平台限制）。"""
    text = closure._rm65_model_text()
    return text.replace(
        f'file="{(ROOT / "assets").as_posix()}/',
        f'file="{closure._mirror_prefix()}',
    )


def _nested_include_model_text(meshdir: str) -> str:
    """把一条 ``<mesh>`` 换成嵌套 include，并加全局 ``<compiler meshdir>``。

    Args:
        meshdir: 写入顶层 ``<compiler>`` 的 ``meshdir`` 取值。

    Returns:
        顶层模型 XML 文本。
    """
    text = _mirrored_model_text()
    text, count = re.subn(
        rf'^[ \t]*<mesh name="{_MESH_NAME}"[^>]*/>[ \t]*\n',
        '<include file="sub/parts.xml"/>\n',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    assert count == 1, f"RM65 模型缺少待替换的网格行：{_MESH_NAME}"
    text, count = re.subn(
        r"(<mujoco\b[^>]*>)",
        lambda match: (
            f'{match.group(1)}\n'
            f'  <compiler angle="degree" meshdir="{meshdir}"/>'
        ),
        text,
        count=1,
    )
    assert count == 1, "RM65 模型缺少 <mujoco> 根标签"
    return text


def _write_include_fragment(directory: Path) -> Path:
    """写声明该网格的 include 片段（``file`` 仅为裸文件名）。"""
    include_path = directory / "parts.xml"
    include_path.write_text(
        "<mujocoinclude>\n"
        f'  <mesh name="{_MESH_NAME}" file="{_MESH_FILE}"/>\n'
        "</mujocoinclude>\n",
        encoding="utf-8",
    )
    return include_path


def _resolve_config(model_dir: Path, model_path: Path) -> Any:
    """以真实 profile 模板 + CLI 解析器构造 ``ResolvedV12Config``。"""
    profile_path = closure._write_profile(
        model_dir,
        model_path,
        template=ROOT / "configs" / "robots" / "rm65.yaml",
    )
    return closure._parse_and_resolve(
        ["--robot-config", str(profile_path), *closure._RM65_SHORT_ARGV]
    )


def _write_meshdir_case(
    tmp_path: Path,
    *,
    with_decoy: bool,
    name: str = "meshdir_case",
    external: bool = False,
) -> _MeshdirCase:
    """构造"顶层编译目录 + 嵌套 include"用例并写入 v2 产物。

    Args:
        tmp_path: 用例根目录（ASCII 路径）。
        with_decoy: 是否在 include 目录写同名诱饵（1.25× 放大）。
        name: 用例子目录名。
        external: 为 ``True`` 时不改 ``meshdir``，改为把该网格指向外部目录
            的绝对路径（M05 场景）。

    Returns:
        夹具描述。
    """
    base = tmp_path / name
    meshes_dir = base / _MESHDIR
    include_dir = base / "sub"
    meshes_dir.mkdir(parents=True)
    include_dir.mkdir(parents=True)
    source_mesh = ROOT / str(closure._UNDERPAN_RELATIVE)
    correct_mesh = meshes_dir / _MESH_FILE
    shutil.copy2(source_mesh, correct_mesh)
    decoy_mesh: Path | None = None
    if with_decoy:
        decoy_mesh = include_dir / _MESH_FILE
        closure._scale_binary_stl(source_mesh, decoy_mesh, _DECOY_SCALE)
    external_dir: Path | None = None
    if external:
        external_dir = base / "external_assets"
        external_dir.mkdir()
        external_mesh = external_dir / _MESH_FILE
        shutil.copy2(source_mesh, external_mesh)
        text = _mirrored_model_text()
        text, count = re.subn(
            rf'<mesh name="{_MESH_NAME}"[^>]*/>',
            f'<mesh name="{_MESH_NAME}" file="{external_mesh.as_posix()}"/>',
            text,
            count=1,
        )
        assert count == 1, "RM65 模型缺少待替换的网格行"
        # 该布局不写 include 片段：M05 只考察"顶层绝对引用外部 STL"
        model_path = base / "nested_external.xml"
        model_path.write_text(text, encoding="utf-8")
        correct_mesh = external_mesh
    else:
        _write_include_fragment(include_dir)
        model_path = base / "nested_meshdir.xml"
        model_path.write_text(_nested_include_model_text(_MESHDIR), encoding="utf-8")
    config = _resolve_config(base, model_path)
    # 用**与装配同源**的依赖闭包作为结果清单（R1D 由 load_robot_model 内部
    # 记录同一份；RM65 的 V12 bundle 不暴露该属性）：旧实现会在这里把
    # include 目录下的 1.25× 诱饵记成来源，回放网格随之放大
    manifest, closed_missing = collect_model_file_dependencies(model_path)
    assert closed_missing == ()
    result = closure._fabricate_rm65(config)
    if any(
        field.name == "model_dependency_manifest"
        for field in dataclasses.fields(type(result))
    ):
        result = dataclasses.replace(
            result, model_dependency_manifest=tuple(manifest)
        )
    output_dir = base / "episode"
    artifacts.write_episode_artifacts(result, output_dir)
    return _MeshdirCase(
        model_path=model_path,
        correct_mesh=correct_mesh,
        decoy_mesh=decoy_mesh,
        external_dir=external_dir,
        output_dir=output_dir,
    )


def _mesh_vertices(model: mujoco.MjModel, name: str = _MESH_NAME) -> np.ndarray:
    """按名称取已编译网格的顶点副本（用于逐位比较）。"""
    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, name)
    assert mesh_id >= 0, f"加载模型缺少网格：{name}"
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    return np.array(
        model.mesh_vert[start : start + count], dtype=np.float64, copy=True
    )


def _stl_extent(path: Path) -> np.ndarray:
    """读二进制 STL 的三轴包围盒（校验诱饵确实被等比放大）。"""
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    assert len(raw) == 84 + 50 * count, "仅支持二进制 STL"
    coords = np.array(
        [
            struct.unpack_from("<3f", raw, 84 + index * 50 + 12 + vertex * 12)
            for index in range(count)
            for vertex in range(3)
        ],
        dtype=np.float64,
    )
    return np.ptp(coords, axis=0)


def _mesh_asset_entry(payload: dict[str, Any], mesh: Path) -> dict[str, Any]:
    """从 ``assets.files`` 取出该网格条目。"""
    for entry in payload["assets"]["files"]:
        if Path(str(entry["source"])).name == mesh.name:
            return dict(entry)
    raise AssertionError(f"assets.files 未记录该网格：{mesh}")


def _strip_asset_dirs(output_dir: Path) -> int:
    """删掉 payload/manifest 依赖清单里的 ``asset_dirs``（模拟旧产物）。"""
    payload = closure._load_payload(output_dir)
    manifest = closure._load_manifest(output_dir)
    removed = 0
    for section in (payload.get("dependencies"), manifest.get("dependencies")):
        for entry in (section or {}).get("files", []):
            if isinstance(entry, dict) and entry.pop("asset_dirs", None) is not None:
                removed += 1
    closure._write_payload(output_dir, payload)
    closure._write_manifest(output_dir, manifest)
    return removed


@dataclass(slots=True)
class _VariantCase:
    """携带真实装配清单的 RM65 变体夹具。

    Attributes:
        base: 用例根目录。
        model_path: 顶层模型 XML。
        output_dir: 写入的 v2 产物目录。
    """

    base: Path
    model_path: Path
    output_dir: Path


def _compile_model(path: Path) -> mujoco.MjModel:
    """编译模型：ASCII 路径走原生 ``from_xml_path``，否则用跨平台加载器。"""
    raw = str(path)
    if raw.isascii():
        return mujoco.MjModel.from_xml_path(raw)
    return load_mujoco_model(path)


def _replace_underpan(text: str, replacement: str) -> str:
    """把顶层模型的 underpan ``<mesh>`` 行替换为给定片段（只替换一次）。"""
    text, count = re.subn(
        rf'^[ \t]*<mesh name="{_MESH_NAME}"[^>]*/>[ \t]*\n',
        replacement.rstrip("\n") + "\n",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    assert count == 1, f"RM65 模型缺少待替换的网格行：{_MESH_NAME}"
    return text


def _insert_after_root(text: str, *tags: str) -> str:
    """在 ``<mujoco>`` 根标签后按给定顺序插入标签（允许注释标签）。"""
    payload = "\n".join(f"  {tag}" for tag in tags)
    text, count = re.subn(
        r"(<mujoco\b[^>]*>)",
        lambda match: f"{match.group(1)}\n{payload}",
        text,
        count=1,
    )
    assert count == 1, "RM65 模型缺少 <mujoco> 根标签"
    return text


def _insert_into_asset(text: str, fragment: str) -> str:
    """在 ``<asset>`` 开始标签后插入一段内容（可含注释）。"""
    text, count = re.subn(
        r"(?m)^([ \t]*)<asset>[ \t]*$",
        lambda match: f"{match.group(0)}\n{match.group(1)}  {fragment}",
        text,
        count=1,
    )
    assert count == 1, "RM65 模型缺少 <asset> 标签"
    return text


def _write_variant_case(base: Path, text: str) -> _VariantCase:
    """写变体模型 + 真实装配清单 + v2 产物。"""
    base.mkdir(parents=True, exist_ok=True)
    model_path = base / "variant_model.xml"
    model_path.write_text(text, encoding="utf-8")
    config = _resolve_config(base, model_path)
    manifest, closed_missing = collect_model_file_dependencies(model_path)
    assert closed_missing == ()
    result = closure._fabricate_rm65(config)
    if any(
        field.name == "model_dependency_manifest"
        for field in dataclasses.fields(type(result))
    ):
        result = dataclasses.replace(
            result, model_dependency_manifest=tuple(manifest)
        )
    output_dir = base / "episode"
    artifacts.write_episode_artifacts(result, output_dir)
    return _VariantCase(base=base, model_path=model_path, output_dir=output_dir)


def _model_entry(model_path: Path) -> dict[str, Any]:
    """返回收集器记录的唯一 model 条目（断言无缺失）。"""
    dependencies, missing = collect_model_file_dependencies(model_path)
    assert missing == ()
    return dict(next(item for item in dependencies if item["role"] == "model"))


def _recorded_mesh_sources(
    model_path: Path, filename: str = _MESH_FILE
) -> list[str]:
    """收集器记录的、文件名等于 ``filename`` 的网格来源。"""
    dependencies, missing = collect_model_file_dependencies(model_path)
    assert missing == ()
    return [
        str(item["source"])
        for item in dependencies
        if item["role"] == "mesh" and Path(str(item["source"])).name == filename
    ]


def _standalone_mesh_vertices(mesh_path: Path) -> np.ndarray:
    """用真实 MuJoCo 单独编译一个 STL，返回其顶点（证明选中的文件）。"""
    probe_name = "__mesh_probe"
    probe_path = mesh_path.parent / f"{probe_name}.xml"
    probe_path.write_text(
        "<mujoco>\n"
        f'  <asset><mesh name="{probe_name}" file="{mesh_path.as_posix()}"/></asset>\n'
        f'  <worldbody><geom type="mesh" mesh="{probe_name}"/></worldbody>\n'
        "</mujoco>\n",
        encoding="utf-8",
    )
    return _mesh_vertices(_compile_model(probe_path), probe_name)


def _stage_model_text(model_path: Path, staging_dir: Path) -> str:
    """按收集清单直接物化一次模型文本（用于检查注释是否被改写）。"""
    entries, missing = collect_model_file_dependencies(model_path)
    assert missing == ()
    normalized = artifacts._normalize_dependency_entries(entries)
    resolved = {
        str(entry["asset_id"]): Path(str(entry["source"])) for entry in normalized
    }
    staging_dir.mkdir(parents=True, exist_ok=True)
    stager = artifacts._DependencyStager(
        entries=normalized,
        resolved=resolved,
        staging_dir=staging_dir,
        logical_base=model_path,
    )
    return stager.stage_model(model_path.read_text(encoding="utf-8"))


def _assert_replay_vertices_match(
    case: _VariantCase, *, mesh_name: str = _MESH_NAME
) -> np.ndarray:
    """编译原始/回放模型并逐顶点比较，返回原始顶点。"""
    reference = _mesh_vertices(_compile_model(case.model_path), mesh_name)
    replayed = _mesh_vertices(
        artifacts.build_replay_bundle(case.output_dir).execution.model, mesh_name
    )
    np.testing.assert_allclose(replayed, reference, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.ptp(replayed, axis=0), np.ptp(reference, axis=0), rtol=0.0, atol=1e-12
    )
    return reference


# ── M01：全局 meshdir 语义 ─────────────────────────────────────────────


def test_nested_include_prefers_global_meshdir_over_same_name_decoy(
    tmp_path: Path,
) -> None:
    """同名诱饵存在时，收集与回放都必须用顶层 meshdir 下的正确网格。"""
    case = _write_meshdir_case(tmp_path, with_decoy=True)
    assert case.decoy_mesh is not None and case.decoy_mesh.is_file()
    # 诱饵确实与正确网格不同（1.25×），否则本用例无法证明选择正确
    np.testing.assert_allclose(
        _stl_extent(case.decoy_mesh),
        _stl_extent(case.correct_mesh) * _DECOY_SCALE,
        rtol=1e-6,
    )

    dependencies, missing = collect_model_file_dependencies(case.model_path)
    assert missing == ()
    recorded = [
        str(item["source"])
        for item in dependencies
        if item["role"] == "mesh" and Path(str(item["source"])).name == _MESH_FILE
    ]
    # 旧实现按"include 文件目录 + 本文件 compiler"解析，这里会是诱饵路径
    assert recorded == [str(case.correct_mesh.resolve())]
    model_entry = next(item for item in dependencies if item["role"] == "model")
    assert model_entry["asset_dirs"] == {"mesh": _MESHDIR}

    payload = closure._load_payload(case.output_dir)
    # 夹具必须走"真实装配清单"（评审探针同链路），而不是推导降级分支
    assert payload["dependency_manifest_source"] == "result"
    assert _mesh_asset_entry(payload, case.correct_mesh)["source"] == str(
        case.correct_mesh.resolve()
    )
    assert artifacts._verify_model_assets(payload, case.output_dir) == []

    reference = _mesh_vertices(load_mujoco_model(case.model_path))
    replayed = _mesh_vertices(
        artifacts.build_replay_bundle(case.output_dir).execution.model
    )
    np.testing.assert_allclose(replayed, reference, rtol=0.0, atol=1e-12)
    reference_extent = np.ptp(reference, axis=0)
    np.testing.assert_allclose(
        np.ptp(replayed, axis=0), reference_extent, rtol=0.0, atol=1e-12
    )

    # 自证症状量级：把全局 meshdir 指向 include 目录（旧实现的解析基准）时，
    # 同一份模型加载出来的网格正是评审记录的 1.25×——本用例能抓住该回归
    variant_path = case.model_path.parent / "decoy_variant.xml"
    variant_path.write_text(_nested_include_model_text("sub"), encoding="utf-8")
    decoy_vertices = _mesh_vertices(load_mujoco_model(variant_path))
    np.testing.assert_allclose(
        np.ptp(decoy_vertices, axis=0),
        reference_extent * _DECOY_SCALE,
        rtol=1e-5,
    )
    assert not np.allclose(np.ptp(replayed, axis=0), np.ptp(decoy_vertices, axis=0))


def test_nested_include_falls_back_when_only_global_meshdir_file_exists(
    tmp_path: Path,
) -> None:
    """只有顶层 meshdir 文件存在时（旧实现报缺失）仍能收集并回放。"""
    case = _write_meshdir_case(tmp_path, with_decoy=False, name="meshdir_only")
    assert not (case.model_path.parent / "sub" / _MESH_FILE).exists()

    dependencies, missing = collect_model_file_dependencies(case.model_path)
    assert missing == ()
    recorded = [
        str(item["source"])
        for item in dependencies
        if item["role"] == "mesh" and Path(str(item["source"])).name == _MESH_FILE
    ]
    assert recorded == [str(case.correct_mesh.resolve())]

    payload = closure._load_payload(case.output_dir)
    assert artifacts._verify_model_assets(payload, case.output_dir) == []
    reference = _mesh_vertices(load_mujoco_model(case.model_path))
    replayed = _mesh_vertices(
        artifacts.build_replay_bundle(case.output_dir).execution.model
    )
    np.testing.assert_allclose(replayed, reference, rtol=0.0, atol=1e-12)


def test_missing_asset_reports_top_level_meshdir_candidate(tmp_path: Path) -> None:
    """两个候选都不存在时，缺失报告第一候选（顶层 meshdir 路径）。"""
    case = _write_meshdir_case(tmp_path, with_decoy=False, name="meshdir_missing")
    case.correct_mesh.unlink()

    dependencies, missing = collect_model_file_dependencies(case.model_path)
    expected_first = (
        case.model_path.parent / _MESHDIR / _MESH_FILE
    ).resolve()
    assert missing == (str(expected_first),)
    assert not [
        item
        for item in dependencies
        if item["role"] == "mesh" and Path(str(item["source"])).name == _MESH_FILE
    ]


# ── M05：快照副本校验 ─────────────────────────────────────────────────


def test_external_mesh_snapshot_survives_source_deletion(tmp_path: Path) -> None:
    """外部 STL 写入快照后删除来源：摘要校验与回放都必须通过。"""
    case = _write_meshdir_case(
        tmp_path, with_decoy=False, name="external_case", external=True
    )
    assert case.external_dir is not None
    payload = closure._load_payload(case.output_dir)
    before = _mesh_vertices(
        artifacts.build_replay_bundle(case.output_dir).execution.model
    )
    entry = next(
        item
        for item in payload["dependencies"]["files"]
        if item["role"] == "mesh"
        and item["source"] == str(case.correct_mesh.resolve())
    )
    snapshot_path = entry.get("snapshot_path")
    assert snapshot_path, "外部网格写入时必须复制快照副本"
    snapshot_file = case.output_dir / artifacts.SNAPSHOT_DIR_NAME / str(snapshot_path)
    assert snapshot_file.is_file()
    assert closure._sha256(snapshot_file) == entry["sha256"]

    shutil.rmtree(case.external_dir)
    assert not case.correct_mesh.exists()

    # M05 核心：旧实现只查仓库/来源，删除后误报"缺失 + 聚合哈希不一致"
    assert artifacts._verify_model_assets(payload, case.output_dir) == []
    assert payload["assets"]["digest"] == artifacts._assets_digest(
        payload["assets"]["files"]
    )
    assert payload["input_sha256"]["assets"] == payload["assets"]["digest"]

    bundle = artifacts.build_replay_bundle(case.output_dir)
    np.testing.assert_array_equal(
        _mesh_vertices(bundle.execution.model),
        before,
    )
    np.testing.assert_allclose(
        closure._loaded_site_pos(bundle), closure._SNAPSHOT_SITE_POS, atol=1e-12
    )


def test_tampered_snapshot_copy_is_still_rejected(tmp_path: Path) -> None:
    """来源删除后篡改快照副本：校验仍按 sha256 拒绝（强度不降）。"""
    case = _write_meshdir_case(
        tmp_path, with_decoy=False, name="external_tampered", external=True
    )
    assert case.external_dir is not None
    payload = closure._load_payload(case.output_dir)
    entry = next(
        item
        for item in payload["dependencies"]["files"]
        if item["role"] == "mesh"
        and item["source"] == str(case.correct_mesh.resolve())
    )
    snapshot_file = (
        case.output_dir / artifacts.SNAPSHOT_DIR_NAME / str(entry["snapshot_path"])
    )
    shutil.rmtree(case.external_dir)
    assert not case.correct_mesh.exists()

    snapshot_file.write_bytes(b"solid tampered\nendsolid tampered\n")
    assert artifacts._verify_model_assets(payload, case.output_dir)
    with pytest.raises(ValueError, match="依赖资产|哈希|不一致"):
        artifacts.build_replay_bundle(case.output_dir)


# ── 兼容性：旧产物 / 字段契约 ─────────────────────────────────────────


def test_legacy_artifact_without_asset_dirs_replays(tmp_path: Path) -> None:
    """旧产物没有 ``asset_dirs``：回放侧从 XML 文本重建全局目录仍正确。"""
    case = _write_meshdir_case(tmp_path, with_decoy=True, name="legacy_case")
    removed = _strip_asset_dirs(case.output_dir)
    assert removed >= 1, "夹具应产生 asset_dirs，否则本用例无法覆盖旧产物分支"

    payload = closure._load_payload(case.output_dir)
    assert all(
        "asset_dirs" not in item for item in payload["dependencies"]["files"]
    )
    reference = _mesh_vertices(load_mujoco_model(case.model_path))
    bundle = artifacts.build_replay_bundle(case.output_dir)
    replayed = _mesh_vertices(bundle.execution.model)
    np.testing.assert_allclose(replayed, reference, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.ptp(replayed, axis=0), np.ptp(reference, axis=0), rtol=0.0, atol=1e-12
    )


def test_asset_files_schema_and_digest_unchanged(tmp_path: Path) -> None:
    """``assets.files`` 字段与聚合摘要保持既有契约（不新增字段）。"""
    case = _write_meshdir_case(tmp_path, with_decoy=True, name="schema_case")
    payload = closure._load_payload(case.output_dir)
    section = payload["assets"]
    files = [dict(item) for item in section["files"]]
    assert section["status"] == "ok"
    assert section["count"] == len(files) == 22
    for entry in files:
        assert set(entry) == _ASSET_FILE_FIELDS, entry
        assert entry["role"] == "assets"
        assert entry["kind"] in {"mesh", "texture"}
        source = Path(str(entry["source"]))
        assert source.is_file()
        assert entry["sha256"] == closure._sha256(source)
        assert entry["size"] == source.stat().st_size
    assert section["digest"] == artifacts._assets_digest(files)
    assert payload["input_sha256"]["assets"] == section["digest"]
    manifest = closure._load_manifest(case.output_dir)
    assert manifest["assets"]["files"] == section["files"]
    # include 片段里声明的网格必须进入资产闭包（旧实现只扫顶层文本）
    assert any(
        Path(str(entry["source"])).name == _MESH_FILE for entry in files
    )


def test_top_level_assets_count_matches_real_r1d_model() -> None:
    """真实 R1D 模型（网格全在顶层）的资产集合与文本声明逐一致（不回归）。"""
    model_path = ROOT / "src" / "robot" / "r1d_model_right.xml"
    text = model_path.read_text(encoding="utf-8")
    assert "<include" not in text
    declared = {
        str((model_path.parent / raw).resolve())
        for raw in re.findall(r'<mesh\b[^>]*\bfile\s*=\s*"([^"]+)"', text)
    }
    entries, missing = artifacts._model_asset_entries(model_path)
    assert missing == []
    recorded = {
        str(Path(str(entry["source"])).resolve())
        for entry in entries
        if entry["kind"] == "mesh"
    }
    assert recorded == declared
    assert len(entries) == len(declared)


# ── R03：XML 注释完全忽略 ─────────────────────────────────────────────


def _write_commented_case(base: Path) -> _VariantCase:
    """构造"注释里含 compiler/mesh/include"的变体用例。"""
    actual_dir = base / "actual"
    decoy_dir = base / "decoy"
    actual_dir.mkdir(parents=True)
    decoy_dir.mkdir(parents=True)
    source_stl = ROOT / str(closure._UNDERPAN_RELATIVE)
    shutil.copy2(source_stl, actual_dir / _MESH_FILE)
    closure._scale_binary_stl(source_stl, decoy_dir / _MESH_FILE, _DECOY_SCALE)

    text = _mirrored_model_text()
    text = _replace_underpan(
        text, f'<mesh name="{_MESH_NAME}" file="{_MESH_FILE}"/>'
    )
    text = _insert_after_root(
        text,
        f'<compiler angle="degree" meshdir="{actual_dir.as_posix()}"/>',
        f'<!-- <compiler meshdir="{decoy_dir.as_posix()}"/> -->',
    )
    text = _insert_into_asset(
        text,
        '<!-- <mesh name="ghost" file="ghost.STL"/>'
        ' <include file="sub/ghost.xml"/> -->',
    )
    return _write_variant_case(base, text)


def test_commented_compiler_mesh_and_include_are_ignored(tmp_path: Path) -> None:
    """注释里的 compiler/mesh/include 不参与目录、收集与快照改写。"""
    case = _write_commented_case(tmp_path / "commented")
    actual_dir = case.base / "actual"

    # 注释里的 meshdir 不合并：只记录显式生效的 actual 目录
    assert _model_entry(case.model_path)["asset_dirs"] == {
        "mesh": actual_dir.as_posix()
    }
    assert _recorded_mesh_sources(case.model_path) == [
        str((actual_dir / _MESH_FILE).resolve())
    ]
    dependencies, missing = collect_model_file_dependencies(case.model_path)
    assert missing == ()
    assert not [item for item in dependencies if item["role"] == "include"]
    assert not [
        item for item in dependencies if "ghost" in str(item["source"])
    ]

    payload = closure._load_payload(case.output_dir)
    assert artifacts._verify_model_assets(payload, case.output_dir) == []
    assert len(payload["assets"]["files"]) == 22

    reference = _assert_replay_vertices_match(case)
    np.testing.assert_allclose(
        reference,
        _standalone_mesh_vertices(actual_dir / _MESH_FILE),
        rtol=0.0,
        atol=1e-12,
    )

    # 注释内容在快照改写中原样保留：既不改写、也不引入 staging 路径
    model_text = case.model_path.read_text(encoding="utf-8")
    staged = _stage_model_text(case.model_path, case.base / "staging_probe")
    comments = re.findall(r"<!--.*?-->", model_text, flags=re.DOTALL)
    assert len(comments) >= 2, "夹具应包含 compiler 与 mesh/include 两段注释"
    for comment in comments:
        assert comment in staged
    assert 'file="ghost.STL"' in staged  # 注释内的引用未被改写


# ── R04：include 先顶层、后声明目录 ───────────────────────────────────


def _write_mesh_fragment(path: Path, name: str, mesh: Path) -> None:
    """写一个只声明单条网格的 include 片段。"""
    path.write_text(
        "<mujocoinclude>\n"
        f'  <mesh name="{name}" file="{mesh.as_posix()}"/>\n'
        "</mujocoinclude>\n",
        encoding="utf-8",
    )


def _write_include_priority_case(base: Path, *, keep_top_level: bool) -> _VariantCase:
    """构造根目录与 ``sub/`` 同名 ``mesh.xml`` 的嵌套 include 用例。

    Args:
        base: 用例根目录。
        keep_top_level: 为 ``False`` 时删除顶层 ``mesh.xml``，只留 ``sub/``。

    Returns:
        变体夹具；``base/actual`` 与 ``base/decoy`` 下各有一份 STL。
    """
    source_stl = ROOT / str(closure._UNDERPAN_RELATIVE)
    actual = base / "actual" / _MESH_FILE
    decoy = base / "decoy" / _MESH_FILE
    actual.parent.mkdir(parents=True)
    decoy.parent.mkdir(parents=True)
    shutil.copy2(source_stl, actual)
    closure._scale_binary_stl(source_stl, decoy, _DECOY_SCALE)
    sub = base / "sub"
    sub.mkdir(parents=True)
    _write_mesh_fragment(base / "mesh.xml", _MESH_NAME, actual)
    _write_mesh_fragment(sub / "mesh.xml", _MESH_NAME, decoy)
    (sub / "entry.xml").write_text(
        '<mujocoinclude>\n  <include file="mesh.xml"/>\n</mujocoinclude>\n',
        encoding="utf-8",
    )
    if not keep_top_level:
        (base / "mesh.xml").unlink()
    text = _replace_underpan(
        _mirrored_model_text(), '<include file="sub/entry.xml"/>'
    )
    return _write_variant_case(base, text)


def test_include_prefers_top_level_directory_over_declaring_directory(
    tmp_path: Path,
) -> None:
    """根目录与 ``sub/`` 同名 ``mesh.xml`` 并存时，回放必须选根目录。"""
    case = _write_include_priority_case(
        tmp_path / "root_wins", keep_top_level=True
    )
    assert (case.base / "mesh.xml").is_file()
    assert (case.base / "sub" / "mesh.xml").is_file()

    dependencies, missing = collect_model_file_dependencies(case.model_path)
    assert missing == ()
    includes = {
        str(item["source"])
        for item in dependencies
        if item["role"] == "include"
    }
    assert includes == {
        str((case.base / "sub" / "entry.xml").resolve()),
        str((case.base / "mesh.xml").resolve()),
    }
    assert _recorded_mesh_sources(case.model_path) == [
        str((case.base / "actual" / _MESH_FILE).resolve())
    ]
    assert _model_entry(case.model_path).get("asset_dirs") is None

    payload = closure._load_payload(case.output_dir)
    assert artifacts._verify_model_assets(payload, case.output_dir) == []
    assert len(payload["assets"]["files"]) == 22
    reference = _assert_replay_vertices_match(case)
    expected = _standalone_mesh_vertices(case.base / "actual" / _MESH_FILE)
    np.testing.assert_allclose(reference, expected, rtol=0.0, atol=1e-12)
    decoy = _standalone_mesh_vertices(case.base / "decoy" / _MESH_FILE)
    np.testing.assert_allclose(
        np.ptp(decoy, axis=0),
        np.ptp(expected, axis=0) * _DECOY_SCALE,
        rtol=1e-5,
    )


def test_include_falls_back_to_declaring_directory_when_top_level_missing(
    tmp_path: Path,
) -> None:
    """顶层 ``mesh.xml`` 不存在时，回退声明文件目录（``sub/``）。"""
    case = _write_include_priority_case(
        tmp_path / "sub_fallback", keep_top_level=False
    )
    assert not (case.base / "mesh.xml").exists()
    dependencies, missing = collect_model_file_dependencies(case.model_path)
    assert missing == ()
    includes = {
        str(item["source"])
        for item in dependencies
        if item["role"] == "include"
    }
    assert includes == {
        str((case.base / "sub" / "entry.xml").resolve()),
        str((case.base / "sub" / "mesh.xml").resolve()),
    }
    assert _recorded_mesh_sources(case.model_path) == [
        str((case.base / "decoy" / _MESH_FILE).resolve())
    ]

    payload = closure._load_payload(case.output_dir)
    assert artifacts._verify_model_assets(payload, case.output_dir) == []
    reference = _assert_replay_vertices_match(case)
    fallback_vertices = _standalone_mesh_vertices(
        case.base / "decoy" / _MESH_FILE
    )
    np.testing.assert_allclose(
        reference, fallback_vertices, rtol=0.0, atol=1e-12
    )
    top_vertices = _standalone_mesh_vertices(case.base / "actual" / _MESH_FILE)
    np.testing.assert_allclose(
        np.ptp(reference, axis=0),
        np.ptp(top_vertices, axis=0) * _DECOY_SCALE,
        rtol=1e-5,
    )


# ── R04：compiler assetdir 语义 ───────────────────────────────────────


def _write_assetdir_case(
    base: Path, compilers: tuple[str, ...]
) -> tuple[_VariantCase, Path, Path]:
    """构造 assetdir 用例：``assets/`` 为正确网格、``other/`` 为 1.25× 诱饵。"""
    source_stl = ROOT / str(closure._UNDERPAN_RELATIVE)
    assets_mesh = base / "assets" / _MESH_FILE
    other_mesh = base / "other" / _MESH_FILE
    assets_mesh.parent.mkdir(parents=True)
    other_mesh.parent.mkdir(parents=True)
    shutil.copy2(source_stl, assets_mesh)
    closure._scale_binary_stl(source_stl, other_mesh, _DECOY_SCALE)
    text = _replace_underpan(
        _mirrored_model_text(), f'<mesh name="{_MESH_NAME}" file="{_MESH_FILE}"/>'
    )
    text = _insert_after_root(text, *compilers)
    return _write_variant_case(base, text), assets_mesh, other_mesh


@pytest.mark.parametrize(
    ("compilers", "expected_dir", "expected_dirs"),
    [
        pytest.param(
            ('<compiler angle="degree" assetdir="assets"/>',),
            "assets",
            {"mesh": "assets", "texture": "assets"},
            id="assetdir-only",
        ),
        pytest.param(
            (
                '<compiler angle="degree" assetdir="assets" meshdir="other"/>',
            ),
            "other",
            {"mesh": "other", "texture": "assets"},
            id="assetdir-then-explicit-meshdir",
        ),
        pytest.param(
            (
                '<compiler angle="degree" meshdir="other" assetdir="assets"/>',
            ),
            "other",
            {"mesh": "other", "texture": "assets"},
            id="explicit-meshdir-then-assetdir",
        ),
        pytest.param(
            (
                '<compiler angle="degree" assetdir="assets"/>',
                '<compiler meshdir="other"/>',
            ),
            "other",
            {"mesh": "other", "texture": "assets"},
            id="later-explicit-meshdir",
        ),
        pytest.param(
            (
                '<compiler angle="degree" meshdir="other"/>',
                '<compiler angle="degree" assetdir="assets"/>',
            ),
            "assets",
            {"mesh": "assets", "texture": "assets"},
            id="later-assetdir-overrides-both",
        ),
    ],
)
def test_assetdir_compiler_semantics(
    tmp_path: Path,
    compilers: tuple[str, ...],
    expected_dir: str,
    expected_dirs: dict[str, str],
) -> None:
    """assetdir 同标签/跨标签覆盖顺序与 MuJoCo 实测语义一致。"""
    case, assets_mesh, other_mesh = _write_assetdir_case(
        tmp_path / "assetdir_case", compilers
    )
    expected_mesh = case.base / expected_dir / _MESH_FILE
    assert _model_entry(case.model_path)["asset_dirs"] == expected_dirs
    assert _recorded_mesh_sources(case.model_path) == [
        str(expected_mesh.resolve())
    ]

    payload = closure._load_payload(case.output_dir)
    assert artifacts._verify_model_assets(payload, case.output_dir) == []
    reference = _assert_replay_vertices_match(case)
    np.testing.assert_allclose(
        reference,
        _standalone_mesh_vertices(expected_mesh),
        rtol=0.0,
        atol=1e-12,
    )
    unexpected = other_mesh if expected_dir == "assets" else assets_mesh
    unexpected_vertices = _standalone_mesh_vertices(unexpected)
    ratio = np.ptp(unexpected_vertices, axis=0) / np.ptp(reference, axis=0)
    np.testing.assert_allclose(
        ratio,
        np.full(3, 1.0 / _DECOY_SCALE if expected_dir == "other" else _DECOY_SCALE),
        rtol=1e-5,
    )


# ── 兼容性：旧产物在 R03/R04 语义下重建 ───────────────────────────────


def test_legacy_artifact_without_asset_dirs_ignores_commented_compiler(
    tmp_path: Path,
) -> None:
    """旧产物删掉 ``asset_dirs`` 后，注释里的 compiler 仍不得参与重建。"""
    case = _write_commented_case(tmp_path / "legacy_commented")
    removed = _strip_asset_dirs(case.output_dir)
    assert removed >= 1, "夹具应产生 asset_dirs，否则无法覆盖旧产物分支"
    payload = closure._load_payload(case.output_dir)
    assert all(
        "asset_dirs" not in item for item in payload["dependencies"]["files"]
    )

    reference = _assert_replay_vertices_match(case)
    np.testing.assert_allclose(
        reference,
        _standalone_mesh_vertices(case.base / "actual" / _MESH_FILE),
        rtol=0.0,
        atol=1e-12,
    )


def test_legacy_artifact_without_asset_dirs_rebuilds_assetdir(
    tmp_path: Path,
) -> None:
    """旧产物删掉 ``asset_dirs`` 后，回放侧仍能从文本重建 ``assetdir``。"""
    case, _assets_mesh, _other_mesh = _write_assetdir_case(
        tmp_path / "legacy_assetdir",
        ('<compiler angle="degree" assetdir="assets"/>',),
    )
    removed = _strip_asset_dirs(case.output_dir)
    assert removed >= 1, "夹具应产生 asset_dirs，否则无法覆盖旧产物分支"

    reference = _assert_replay_vertices_match(case)
    np.testing.assert_allclose(
        reference,
        _standalone_mesh_vertices(case.base / "assets" / _MESH_FILE),
        rtol=0.0,
        atol=1e-12,
    )
