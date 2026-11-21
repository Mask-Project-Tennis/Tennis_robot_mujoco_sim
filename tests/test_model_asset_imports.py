"""A1 回归：``src.robot.model_assets`` 拆分后的导入契约与依赖闭包。

覆盖三类保证：

1. 旧路径（``src.robot.model_factory``）与新包重导出的是**同一对象**
   （``is`` 相同、``__module__`` 指向新模块），冻结签名不变，且包导入
   不触发工厂/MuJoCo；
2. 最小 MJCF（嵌套 include、注释诱饵、缺失资产/缺失 include）上的依赖
   清单字段、SHA256、内容 base64 与缺失分区保持不变；
3. 非仓库 cwd 下仓库根仍由 ``dependencies.py`` 文件位置解析，仓库内模型
   保持仓库相对路径。
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.robot import model_assets
from src.robot import model_factory as legacy
from src.robot.model_assets import dependencies, references

_ROLE_CONSTANTS = (
    "DEPENDENCY_ROLE_MODEL",
    "DEPENDENCY_ROLE_INCLUDE",
    "DEPENDENCY_ROLE_ARM_FRAGMENT",
    "DEPENDENCY_ROLE_MESH",
    "DEPENDENCY_ROLE_TEXTURE",
)

_REFERENCE_HELPERS = (
    "_ASSET_DIR_ATTRS",
    "_MAX_INCLUDE_DEPTH",
    "_MJCF_TAG_PATTERN",
    "_XML_COMMENT_PATTERN",
    "_AssetReference",
    "_asset_candidate_paths",
    "_attribute_value",
    "_include_candidate_paths",
    "_resolve_reference_path",
    "_walk_model_file",
)

_DEPENDENCY_HELPERS = (
    "_dependency_entry",
    "_dependency_relative_path",
    "_dependency_repo_root",
    "_sha256_file",
)

_ENTRY_FIELDS_BASE = {"role", "relative_path", "source", "sha256", "size"}


def _write(path: Path, text: str) -> Path:
    """写入文本文件（自动建父目录），返回路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    """独立计算文件 SHA256，用于交叉校验依赖条目。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def mjcf_tree(tmp_path: Path) -> dict[str, Path]:
    """构造最小 MJCF 依赖树：嵌套 include、全局目录覆盖、注释诱饵与两类缺失。"""
    models = tmp_path / "models"
    main = _write(
        models / "main.xml",
        """<mujoco model="fixture">
  <!-- 诱饵注释：<mesh file="decoy.STL"/> 与 <include file="decoy.xml"/> -->
  <compiler meshdir="assets/meshes"/>
  <include file="sub/parts.xml"/>
  <include file="nope.xml"/>
  <asset>
    <mesh name="box" file="box.STL"/>
    <mesh name="missing" file="missing.STL"/>
    <texture name="tex" type="2d" file="tex.png"/>
  </asset>
</mujoco>
""",
    )
    parts = _write(
        models / "sub/parts.xml",
        """<mujoco>
  <!-- <mesh file="decoy-in-parts.STL"/> -->
  <compiler texturedir="textures"/>
  <include file="deep/leaf.xml"/>
  <include file="missing_include.xml"/>
  <asset>
    <mesh name="part" file="part.STL"/>
    <mesh name="box_dup" file="box.STL"/>
  </asset>
</mujoco>
""",
    )
    leaf = _write(
        models / "sub/deep/leaf.xml",
        """<mujoco>
  <asset>
    <mesh name="leaf" file="leaf.STL"/>
  </asset>
</mujoco>
""",
    )
    box = _write(models / "assets/meshes/box.STL", "global-meshdir-box")
    part = _write(models / "assets/meshes/part.STL", "global-meshdir-part")
    leaf_mesh = _write(models / "assets/meshes/leaf.STL", "global-meshdir-leaf")
    texture = _write(models / "textures/tex.png", "texture-bytes")
    # 同名诱饵：只在声明目录存在，用于验证"顶层模型目录 + 全局 meshdir"优先。
    decoy = _write(models / "sub/box.STL", "declaring-dir-decoy")
    return {
        "main": main,
        "parts": parts,
        "leaf": leaf,
        "box": box,
        "part": part,
        "leaf_mesh": leaf_mesh,
        "texture": texture,
        "decoy": decoy,
        "missing_mesh": models / "assets/meshes/missing.STL",
        "missing_include": models / "nope.xml",
        "missing_nested_include": models / "sub/missing_include.xml",
    }


def test_public_api_is_reexported_identically() -> None:
    """旧工厂、新包与实现模块必须暴露同一批对象（is 相同）。"""
    assert (
        legacy.collect_model_file_dependencies
        is model_assets.collect_model_file_dependencies
    )
    assert (
        model_assets.collect_model_file_dependencies
        is dependencies.collect_model_file_dependencies
    )
    assert legacy.mask_xml_comments is model_assets.mask_xml_comments
    assert model_assets.mask_xml_comments is references.mask_xml_comments

    for name in (*_ROLE_CONSTANTS, "DEPENDENCY_XML_ROLES"):
        assert getattr(legacy, name) is getattr(model_assets, name)
        assert getattr(legacy, name) is getattr(references, name)

    for name in _REFERENCE_HELPERS:
        assert getattr(legacy, name) is getattr(references, name)
    for name in _DEPENDENCY_HELPERS:
        assert getattr(legacy, name) is getattr(dependencies, name)

    assert set(model_assets.__all__) == {
        *_ROLE_CONSTANTS,
        "DEPENDENCY_XML_ROLES",
        "collect_model_file_dependencies",
        "mask_xml_comments",
    }


def test_frozen_signatures_and_module_paths() -> None:
    """冻结签名（参数名/kind/默认值/返回注解）与对象所属模块不变。"""
    signature = inspect.signature(legacy.collect_model_file_dependencies)
    assert list(signature.parameters) == ["model_path", "asset_missing"]
    assert (
        signature.parameters["model_path"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert (
        signature.parameters["asset_missing"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert signature.parameters["asset_missing"].default is None
    assert signature.return_annotation == (
        "tuple[tuple[dict[str, Any], ...], tuple[str, ...]]"
    )
    assert (
        legacy.collect_model_file_dependencies.__module__
        == "src.robot.model_assets.dependencies"
    )
    assert legacy.mask_xml_comments.__module__ == "src.robot.model_assets.references"

    collect_dependencies = inspect.signature(legacy.collect_model_dependencies)
    assert list(collect_dependencies.parameters) == [
        "profile",
        "parameters",
        "arm_fragments",
    ]
    arm_fragments = collect_dependencies.parameters["arm_fragments"]
    assert arm_fragments.kind is inspect.Parameter.KEYWORD_ONLY
    assert arm_fragments.default == ()


def test_minimal_mjcf_closure_fields_and_hashes(
    mjcf_tree: dict[str, Path],
) -> None:
    """最小 MJCF 的条目字段、排序、哈希与全局目录语义保持不变。"""
    asset_missing: list[str] = []
    entries, missing = legacy.collect_model_file_dependencies(
        mjcf_tree["main"], asset_missing=asset_missing
    )

    assert isinstance(entries, tuple)
    assert isinstance(missing, tuple)
    assert all(isinstance(entry, dict) for entry in entries)
    assert entries == tuple(
        sorted(entries, key=lambda item: (item["role"], item["relative_path"]))
    )
    assert all(entry["relative_path"].startswith("external/") for entry in entries)
    assert missing == tuple(dict.fromkeys(missing))  # 缺失列表已去重

    by_source = {Path(entry["source"]): entry for entry in entries}
    model_entry = by_source[mjcf_tree["main"]]
    assert set(model_entry) == _ENTRY_FIELDS_BASE | {"content_base64", "asset_dirs"}
    assert model_entry["asset_dirs"] == {"mesh": "assets/meshes", "texture": "textures"}
    assert model_entry["role"] == legacy.DEPENDENCY_ROLE_MODEL

    include_entries = [
        entry for entry in entries if entry["role"] == legacy.DEPENDENCY_ROLE_INCLUDE
    ]
    assert {Path(entry["source"]) for entry in include_entries} == {
        mjcf_tree["parts"],
        mjcf_tree["leaf"],
    }
    assert {entry["reference"] for entry in include_entries} == {
        "sub/parts.xml",
        "deep/leaf.xml",
    }
    for entry in include_entries:
        assert set(entry) == _ENTRY_FIELDS_BASE | {"reference", "content_base64"}

    mesh_entries = [
        entry for entry in entries if entry["role"] == legacy.DEPENDENCY_ROLE_MESH
    ]
    assert {Path(entry["source"]) for entry in mesh_entries} == {
        mjcf_tree["box"],
        mjcf_tree["part"],
        mjcf_tree["leaf_mesh"],
    }
    for entry in mesh_entries:
        assert set(entry) == _ENTRY_FIELDS_BASE | {"reference"}

    texture_entries = [
        entry for entry in entries if entry["role"] == legacy.DEPENDENCY_ROLE_TEXTURE
    ]
    assert [Path(entry["source"]) for entry in texture_entries] == [
        mjcf_tree["texture"]
    ]
    assert set(texture_entries[0]) == _ENTRY_FIELDS_BASE | {"reference"}

    for entry in entries:
        source = Path(entry["source"])
        data = source.read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["size"] == len(data)
        if entry["role"] in legacy.DEPENDENCY_XML_ROLES:
            assert entry["content_base64"] == base64.b64encode(data).decode("ascii")
        else:
            assert "content_base64" not in entry

    # 全局 meshdir 优先于声明目录里的同名文件；注释诱饵不参与解析。
    assert by_source[mjcf_tree["box"]]["sha256"] == _sha256(mjcf_tree["box"])
    assert mjcf_tree["decoy"].read_text(encoding="utf-8") != mjcf_tree["box"].read_text(
        encoding="utf-8"
    )
    assert mjcf_tree["decoy"] not in by_source
    assert all("decoy" not in item for item in missing)


def test_asset_missing_only_contains_asset_failures(
    mjcf_tree: dict[str, Path],
) -> None:
    """``asset_missing`` 只收资产缺失；``missing`` 仍混合 include 缺失。"""
    asset_missing: list[str] = []
    _, missing = legacy.collect_model_file_dependencies(
        mjcf_tree["main"], asset_missing=asset_missing
    )

    assert asset_missing == [str(mjcf_tree["missing_mesh"])]
    assert set(missing) == {
        str(mjcf_tree["missing_mesh"]),
        str(mjcf_tree["missing_include"]),
        str(mjcf_tree["missing_nested_include"]),
    }
    assert str(mjcf_tree["missing_include"]) not in asset_missing
    assert str(mjcf_tree["missing_nested_include"]) not in asset_missing


def test_repo_root_is_resolved_from_file_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """非仓库 cwd 下，仓库根仍指向仓库；仓库内模型保持仓库相对路径。"""
    monkeypatch.chdir(tmp_path)
    assert dependencies._dependency_repo_root() == ROOT

    model = ROOT / "src/robot/r1d_model_right.xml"
    entries, missing = legacy.collect_model_file_dependencies(model)
    assert missing == ()
    assert len(entries) > 1

    model_entry = next(
        entry
        for entry in entries
        if entry["role"] == legacy.DEPENDENCY_ROLE_MODEL
    )
    assert model_entry["relative_path"] == "src/robot/r1d_model_right.xml"
    assert all(not Path(entry["relative_path"]).is_absolute() for entry in entries)


def test_collect_model_dependencies_scans_files_once(
    mjcf_tree: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``collect_model_dependencies`` 只调用一次文件级扫描并合并臂片段。"""
    calls: list[Path] = []
    original = legacy.collect_model_file_dependencies

    def counting_scan(
        model_path: Path, *, asset_missing: list[str] | None = None
    ) -> tuple[tuple[dict[str, object], ...], tuple[str, ...]]:
        calls.append(Path(model_path))
        return original(model_path, asset_missing=asset_missing)

    monkeypatch.setattr(legacy, "collect_model_file_dependencies", counting_scan)
    fragment = _write(tmp_path / "fragments/r1d_arm_left.xml", "<mujoco/>")
    profile = SimpleNamespace(model_path=mjcf_tree["main"])

    entries = legacy.collect_model_dependencies(
        profile, None, arm_fragments=[fragment, fragment]
    )

    assert calls == [mjcf_tree["main"]]
    assert entries == tuple(
        sorted(entries, key=lambda item: (item["role"], item["relative_path"]))
    )
    arm_entries = [
        entry
        for entry in entries
        if entry["role"] == legacy.DEPENDENCY_ROLE_ARM_FRAGMENT
    ]
    assert len(arm_entries) == 1
    assert Path(arm_entries[0]["source"]) == fragment.resolve()
    assert arm_entries[0]["sha256"] == _sha256(fragment)

    file_entries = original(mjcf_tree["main"])[0]
    assert {
        entry["source"]
        for entry in entries
        if entry["role"] != legacy.DEPENDENCY_ROLE_ARM_FRAGMENT
    } == {entry["source"] for entry in file_entries}


def test_package_import_stays_lightweight_in_subprocess(tmp_path: Path) -> None:
    """子进程从仓库外 cwd 导入包：不加载工厂/MuJoCo/仿真层。"""
    script = """
import sys
before = set(sys.modules)
import src.robot.model_assets
added = sorted(set(sys.modules) - before)
forbidden = [
    name
    for name in added
    if name in {"mujoco", "src.robot.model_factory", "src.robot.model_binding"}
    or name.startswith(("mujoco.", "src.sim.", "src.ilqt."))
]
print("MODULES=" + ",".join(added))
print("FORBIDDEN=" + ",".join(forbidden))
"""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    forbidden_line = next(line for line in lines if line.startswith("FORBIDDEN="))
    assert forbidden_line == "FORBIDDEN="
    modules_line = next(line for line in lines if line.startswith("MODULES="))
    assert "src.robot.model_assets.dependencies" in modules_line
    assert "src.robot.model_assets.references" in modules_line
