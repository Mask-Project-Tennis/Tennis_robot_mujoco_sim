"""球场白线场景接线测试（V01/I2）。

验证：
- 新 ITF runtime 与既有 R1D runtime 除 ``scene.court_profile`` 外逐字段一致；
- legacy 模板加载仍为旧 10 条白线，ITF profile 在内存副本替换为 12 条
  标准线（命名、尺寸、位置、无碰撞）；
- 白线不参与接触；
- 同控制序列/来球/初态下，legacy 与 ITF 两 profile 的整回合轨迹、
  控制与球状态逐位一致（白线纯视觉，不改物理）。
"""

from __future__ import annotations

import copy
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src.robot.court_geometry import build_court_lines
from src.robot.model_factory import load_robot_model
from src.robot.robot_profile import load_robot_profile
from src.robot.runtime_parameters import load_runtime_parameters
from src.sim.v12_episode import V12EpisodeResult, run_v12_episode

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rm65_mpc_v12 as entry  # noqa: E402

PROFILE = ROOT / "configs" / "robots" / "r1d_right.yaml"
LEGACY_RUNTIME = ROOT / "configs" / "robots" / "r1d_v12_runtime_right.yaml"
ITF_RUNTIME = ROOT / "configs" / "robots" / "r1d_v12_runtime_right_itf.yaml"
SCENARIO = yaml.safe_load(
    (ROOT / "configs" / "scenarios" / "r1d_v12_active.yaml").read_text(
        encoding="utf-8"
    )
)


def _override_value(value: Any) -> str:
    """把 YAML 标量格式化为 ``--task-overrides`` 可解析的字符串。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 映射（用于逐字段对照两份 runtime）。"""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@lru_cache(maxsize=4)
def _model_for(runtime: Path) -> Any:
    """按指定 runtime 组装 R1D 模型 bundle（读取磁盘 legacy 模板，按路径缓存）。"""
    profile = load_robot_profile(PROFILE)
    parameters = load_runtime_parameters(runtime, profile)
    return load_robot_model(profile, parameters)


def _plane_geoms(model: Any) -> list[int]:
    """所有 plane geom 的 id（含 ground）。"""
    return [gid for gid in range(model.ngeom) if int(model.geom_type[gid]) == 0]


def _config_for(runtime: Path, mode: str = "torque") -> Any:
    """按冻结案例构造 V12 配置，仅替换 runtime 路径（其余物理完全相同）。"""
    common = SCENARIO["common_physical"]
    mode_cfg = SCENARIO["modes"][mode]
    argv = [
        "--robot-config",
        str(ROOT / common["robot_profile"]),
        "--runtime-config",
        str(runtime),
        "--seed",
        str(int(mode_cfg["expected"]["seed"])),
        "--no-plot",
        "--serve-box",
        "--ball-speed",
        str(common["serve"]["ball_speed"]),
    ]
    if mode_cfg.get("actuator_mode") == "position":
        argv.append("--position-mode")
    parser = entry._build_parser()
    args = parser.parse_args(argv)
    overrides = entry._parse_task_overrides(
        [
            f"{key}={_override_value(value)}"
            for key, value in mode_cfg["overrides"].items()
        ]
    )
    assert overrides is not None
    args.task_overrides = overrides
    explicit = entry._explicit_cli_actions(parser, argv)
    return entry.resolve_v12_config(args, explicit)


class TestRuntimeConfigDiff:
    def test_itf_runtime_only_adds_court_profile(self) -> None:
        legacy = _load_yaml(LEGACY_RUNTIME)
        itf = _load_yaml(ITF_RUNTIME)
        assert itf["scene"]["court_profile"] == "itf_2026_v1"
        stripped = copy.deepcopy(itf)
        del stripped["scene"]["court_profile"]
        assert stripped == legacy

    def test_legacy_runtime_defaults_to_legacy_profile(self) -> None:
        profile = load_robot_profile(PROFILE)
        params = load_runtime_parameters(LEGACY_RUNTIME, profile)
        assert params.scene is not None
        assert params.scene.court_profile == "legacy_v1"


class TestModelGeometry:
    def test_legacy_model_keeps_ten_unnamed_lines(self) -> None:
        import mujoco

        model = _model_for(LEGACY_RUNTIME).model
        planes = _plane_geoms(model)
        # ground + 旧 10 条白线；旧模板白线没有 name。
        assert len(planes) == 11
        assert all(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) is None
            for gid in planes[1:]
        )
        expected = sorted(
            (tuple(round(v, 6) for v in line.half_size),
             tuple(round(v, 6) for v in line.center))
            for line in build_court_lines("legacy_v1")
        )
        actual = sorted(
            (tuple(round(float(v), 6) for v in model.geom_size[gid]),
             tuple(round(float(v), 6) for v in model.geom_pos[gid]))
            for gid in planes[1:]
        )
        assert actual == expected

    def test_itf_ground_is_neutral_without_checker_edge(self) -> None:
        """标准 profile 移除棋盘白边伪线：纹理退化为纯中性色。"""
        legacy_model = _model_for(LEGACY_RUNTIME).model
        itf_model = _model_for(ITF_RUNTIME).model
        legacy_texture = legacy_model.texture("court_grid")
        itf_texture = itf_model.texture("court_grid")
        legacy_rgb = np.asarray(legacy_texture.data, dtype=np.uint8).reshape(-1, 3)
        itf_rgb = np.asarray(itf_texture.data, dtype=np.uint8).reshape(-1, 3)
        # 旧棋盘：深浅绿 + 浅色边缘 → 多于一种颜色；新地面：纯中性色。
        assert len(np.unique(legacy_rgb, axis=0)) > 1
        assert len(np.unique(itf_rgb, axis=0)) == 1

    def test_itf_model_has_twelve_named_standard_lines(self) -> None:
        import mujoco

        model = _model_for(ITF_RUNTIME).model
        planes = _plane_geoms(model)
        assert len(planes) == 13
        names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            for gid in planes[1:]
        ]
        expected_names = [line.name for line in build_court_lines("itf_2026_v1")]
        assert names == expected_names
        for gid, line in zip(planes[1:], build_court_lines("itf_2026_v1")):
            np.testing.assert_allclose(model.geom_size[gid], line.half_size, atol=1e-12)
            np.testing.assert_allclose(model.geom_pos[gid], line.center, atol=1e-12)
            np.testing.assert_allclose(model.geom_rgba[gid], line.rgba, atol=1e-12)
            assert int(model.geom_contype[gid]) == 0
            assert int(model.geom_conaffinity[gid]) == 0

    def test_court_lines_never_contact(self) -> None:
        import mujoco

        bundle = _model_for(ITF_RUNTIME)
        model = bundle.model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        line_ids = set(_plane_geoms(model)[1:])
        touching = {
            int(contact.geom1) for contact in data.contact
        } | {int(contact.geom2) for contact in data.contact}
        assert not (line_ids & touching)

    def test_mount_base_position_unchanged(self) -> None:
        import mujoco

        model = _model_for(ITF_RUNTIME).model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "r1d_mount"
        )
        assert body_id != -1
        np.testing.assert_allclose(
            data.xpos[body_id], [-12.5, -0.8, 0.155], atol=1e-9
        )


class TestEpisodeParity:
    def test_legacy_and_itf_episodes_are_bit_identical(self) -> None:
        legacy_config = _config_for(LEGACY_RUNTIME)
        itf_config = _config_for(ITF_RUNTIME)
        legacy = run_v12_episode(legacy_config)
        itf = run_v12_episode(itf_config)
        assert isinstance(legacy, V12EpisodeResult)
        assert isinstance(itf, V12EpisodeResult)
        # 防止“未命中提前返回”让对比变得空洞：固定案例必须真实主动击球。
        assert legacy.hit_type == "active"
        assert legacy.active_contact is True
        assert len(legacy.U) > 0
        for field in ("X", "U", "ball_pos", "ball_vel", "time", "racket_pos"):
            np.testing.assert_array_equal(
                np.asarray(getattr(itf, field)),
                np.asarray(getattr(legacy, field)),
                err_msg=f"{field} 在 legacy/ITF profile 间不一致",
            )
        assert itf.hit_type == legacy.hit_type
        assert itf.hit_step == legacy.hit_step
        assert itf.active_contact == legacy.active_contact
        assert itf.passive_contact == legacy.passive_contact
        assert itf.v_ee_at_hit == pytest.approx(legacy.v_ee_at_hit)
        assert itf.ball_out_speed == pytest.approx(legacy.ball_out_speed)
        assert itf.metrics.get("failure_reason") == legacy.metrics.get("failure_reason")
        assert itf.metrics.get("hit_position_error") == legacy.metrics.get(
            "hit_position_error"
        )
        # I1：真实执行结果必须携带装配依赖清单（writer 严格模式依赖它）。
        assert legacy.model_dependency_manifest
        assert itf.model_dependency_manifest == legacy.model_dependency_manifest
        legacy_scene = legacy.metrics["r1d_artifact"]["fixed_scene_params"]
        itf_scene = itf.metrics["r1d_artifact"]["fixed_scene_params"]
        # legacy_v1 是既有默认：为保持 T7 冻结 scene_snapshot 哈希，默认口径
        # 不写 profile/几何版本键（读侧缺字段即 legacy）；ITF 必须显式记录。
        assert "court_profile" not in legacy_scene
        assert "court_line_count" not in legacy_scene
        assert itf_scene["court_profile"] == "itf_2026_v1"
        assert itf_scene["court_line_count"] == 12
        assert itf_scene["court_geometry_version"] == 1
