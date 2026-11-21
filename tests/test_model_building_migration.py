"""A2 模型装配迁移回归测试。

覆盖 A2 brief 的 5 条验收：

1. 旧 facade 与新模块重导出**同一对象**（``is`` 相同，``__module__``
   指向新模块），保证既有调用点与类型检查不漂移；
2. 从非仓库 cwd 调用 ``load_robot_model`` 仍可加载；``_arm_fragment_path``
   的回退目录来自 ``__file__``（``src/robot``）而非进程 CWD；
   ``collect_model_dependencies`` 的 ``relative_path`` 仍以仓库相对路径开头；
3. R1D 固定关节姿态场景把对侧臂片段纳入依赖清单，且文件级扫描只发生一次
   （不产生第二份扫描实现）；
4. 旧 facade 的公开入口可导入；``__all__`` 情况与既有契约一致——旧模块
   历史上没有 ``__all__``，``tests/test_refactor_shared_types.py`` 明确断言
   ``not hasattr(legacy_factory, "__all__")``，故 brief 中"仍含"的表述按
   既有测试执行（不新增 ``__all__``）；
5. 迁移不改写任何 R1D XML：六个 MJCF 文件的 sha256 与迁移前 golden 相同。
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from src.robot import model_factory as legacy
from src.robot.model_assets import DEPENDENCY_ROLE_ARM_FRAGMENT
from src.robot.model_building import loading, racket, scene, template
from src.robot.model_building import types as building_types
from src.robot.robot_profile import RobotProfile, load_robot_profile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    load_runtime_parameters,
)

# 仓库根目录与双侧交付物路径（与 tests/test_r1d_model.py 同源）
REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOT_DIR = REPO_ROOT / "src" / "robot"
SIDES = ("left", "right")
PROFILE_PATHS = {
    side: REPO_ROOT / "configs" / "robots" / f"r1d_{side}.yaml" for side in SIDES
}
RUNTIME_PATHS = {
    side: REPO_ROOT / "configs" / "robots" / f"r1d_runtime_{side}.yaml"
    for side in SIDES
}

# 迁移前（G1_SHA）记录的 XML 哈希；迁移只允许移动代码，不允许改字节
XML_GOLDEN_SHA256 = {
    "src/robot/r1d_model_left.xml": (
        "240ffb4eefc2862d4dc099ceaa8ddf4e8361a0f9b0e676238a7d528319b34250"
    ),
    "src/robot/r1d_model_right.xml": (
        "7a9116cfb1fbaf7863771d7ecd5e9f4219dfc67d7207321c829d95cd2171cec5"
    ),
    "src/robot/r1d_arm_left.xml": (
        "6ef05cf974ac1053f3f0da23d026ac3eb59010c2acd35ec34ad0579b34368919"
    ),
    "src/robot/r1d_arm_right.xml": (
        "fcaeb323d0ae67ed6f5b80b0ed8ef25bfc166fea11c5523a9e18dc44684799b1"
    ),
    "src/robot/r1d_env.xml": (
        "a3b44d8789c91c0a9f3d6f8f2c74348b85cc97fc18a6cc6122260cb3a1b9139f"
    ),
    "src/robot/r1d_assets.xml": (
        "b03be72a2c23cf195ed59d09ce8e32947e5d639f2d86c4a40b4095e984d372c3"
    ),
}


def _load_profile(side: str) -> RobotProfile:
    """加载该侧机器人描述（``configs/robots/r1d_{side}.yaml``）。"""
    return load_robot_profile(PROFILE_PATHS[side])


def _load_parameters(side: str) -> RobotRuntimeParameters:
    """加载该侧运行参数（``configs/robots/r1d_runtime_{side}.yaml``）。"""
    return load_runtime_parameters(RUNTIME_PATHS[side], _load_profile(side))


def _sha256(path: Path) -> str:
    """返回文件字节的 sha256 十六进制摘要。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_facade_and_new_modules_share_same_objects() -> None:
    """旧 facade 与新模块重导出同一对象，``__module__`` 指向新模块。"""
    assert legacy.load_robot_model is loading.load_robot_model
    assert legacy.collect_model_dependencies is loading.collect_model_dependencies
    assert legacy.write_r1d_model_template is template.write_r1d_model_template
    assert legacy.RacketAssembly is building_types.RacketAssembly
    assert legacy.RobotModelBundle is building_types.RobotModelBundle

    # 集外调用点仍经旧 facade 访问的私有装配入口也必须同一对象
    assert legacy._assemble_bound_model is loading._assemble_bound_model
    assert legacy._arm_fragment_path is loading._arm_fragment_path
    assert legacy._apply_scene_parameters is scene._apply_scene_parameters
    assert legacy._apply_racket_profile is racket._apply_racket_profile
    assert legacy._side_chain_bodies is template._side_chain_bodies

    assert legacy.load_robot_model.__module__ == "src.robot.model_building.loading"
    assert (
        legacy.collect_model_dependencies.__module__
        == "src.robot.model_building.loading"
    )
    assert (
        legacy.write_r1d_model_template.__module__
        == "src.robot.model_building.template"
    )
    assert legacy.RacketAssembly.__module__ == "src.robot.model_building.types"
    assert legacy.RobotModelBundle.__module__ == "src.robot.model_building.types"


def test_loading_and_dependency_paths_ignore_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """非仓库 cwd 下加载成功；片段回退与依赖相对路径均以仓库为根。"""
    profile = _load_profile("left")
    parameters = _load_parameters("left")

    monkeypatch.chdir(tmp_path)
    bundle = legacy.load_robot_model(profile, parameters)
    assert bundle.model.nq == len(profile.active_joint_names)

    manifest = legacy.collect_model_dependencies(profile, parameters)
    assert manifest
    for entry in manifest:
        relative = Path(str(entry["relative_path"]))
        assert not relative.is_absolute()
    assert any(
        entry["relative_path"] == "src/robot/r1d_model_left.xml"
        for entry in manifest
    )

    # 模板同目录无片段时，回退必须指向仓库 src/robot，而不是当前 cwd
    displaced = dataclasses.replace(
        profile, model_path=tmp_path / "elsewhere" / "r1d_model_left.xml"
    )
    fragment = legacy._arm_fragment_path(displaced, "left")
    assert fragment == ROBOT_DIR / "r1d_arm_right.xml"


def test_fixed_joint_poses_add_arm_fragment_with_single_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """固定对侧关节姿态时，片段进依赖清单且文件级扫描只发生一次。"""
    profile = _load_profile("left")
    parameters = _load_parameters("left")
    assert parameters.scene is not None
    fixed_parameters = dataclasses.replace(
        parameters,
        scene=dataclasses.replace(
            parameters.scene,
            fixed_joint_poses={"right_shoulder_pitch_joint": 0.05},
        ),
    )

    calls: list[Path] = []
    original_scan = loading.collect_model_file_dependencies

    def counting_scan(
        model_path: Path, *, asset_missing: list[str] | None = None
    ) -> tuple[tuple[dict[str, object], ...], tuple[str, ...]]:
        calls.append(Path(model_path))
        return original_scan(model_path, asset_missing=asset_missing)

    monkeypatch.setattr(loading, "collect_model_file_dependencies", counting_scan)
    bundle = legacy.load_robot_model(profile, fixed_parameters)

    arm_entries = [
        entry
        for entry in bundle.dependency_manifest
        if entry["role"] == DEPENDENCY_ROLE_ARM_FRAGMENT
    ]
    assert len(arm_entries) == 1
    arm_entry = arm_entries[0]
    assert Path(str(arm_entry["source"])) == (
        ROBOT_DIR / "r1d_arm_right.xml"
    ).resolve()
    assert arm_entry["relative_path"] == "src/robot/r1d_arm_right.xml"
    # 单次 get 装配只允许一次文件级扫描：片段来自装配分支，而非二次扫描
    assert calls == [profile.model_path]


def test_legacy_scanner_patch_still_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """契约 §6：旧 facade 上的扫描 monkeypatch 仍影响新模块的依赖收集。"""
    profile = _load_profile("right")
    parameters = _load_parameters("right")

    calls: list[Path] = []
    original_scan = legacy.collect_model_file_dependencies

    def counting_scan(
        model_path: Path, *, asset_missing: list[str] | None = None
    ) -> tuple[tuple[dict[str, object], ...], tuple[str, ...]]:
        calls.append(Path(model_path))
        return original_scan(model_path, asset_missing=asset_missing)

    monkeypatch.setattr(legacy, "collect_model_file_dependencies", counting_scan)
    entries = legacy.collect_model_dependencies(profile, parameters)

    assert calls == [profile.model_path]
    assert entries


def test_facade_public_names_and_no_module_all() -> None:
    """旧 facade 公开入口可导入，且不新增 ``__all__``（既有契约）。"""
    for name in (
        "load_robot_model",
        "collect_model_dependencies",
        "write_r1d_model_template",
        "RacketAssembly",
        "RobotModelBundle",
    ):
        assert hasattr(legacy, name)
    assert not hasattr(legacy, "__all__")


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    sorted(XML_GOLDEN_SHA256.items()),
)
def test_r1d_xml_files_unchanged(relative_path: str, expected: str) -> None:
    """迁移前后 XML 字节完全一致（只读哈希，不写文件）。"""
    assert _sha256(REPO_ROOT / relative_path) == expected
