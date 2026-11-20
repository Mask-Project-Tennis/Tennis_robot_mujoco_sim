"""I1 跨线集成：产物依赖清单、严格写入与回放限位恢复。

本文件只验证**跨工作线接缝**（A 的 runtime restore × B 的资产/回放闭包）：

- ``run_v12_episode`` 生成的 R1D 结果携带装配依赖清单，writer 以
  ``dependency_manifest_source="result"`` 记录强证据；
- 严格模式下缺失清单拒绝写入（R1D 生产路径）；
- ``build_replay_bundle`` 消费产物里已保存的 runtime_overrides：回放
  bundle 的 ``max_tcp_speed`` / ``terminal_exempt_steps`` / ``dq_max`` 与
  保存的有效值一致，而不是重新落回默认值（F01 replay 缺口）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src.sim.episode_artifacts import (
    build_replay_bundle,
    load_episode_artifacts,
    write_episode_artifacts,
)
from src.sim.episode_runtime import apply_runtime_limits, resolve_runtime_options
from src.sim.v12_episode import V12EpisodeResult
from src.sim.v12_robot_adapter import build_v12_robot

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rm65_mpc_v12 as entry  # noqa: E402

PROFILE = ROOT / "configs" / "robots" / "r1d_right.yaml"
ITF_RUNTIME = ROOT / "configs" / "robots" / "r1d_v12_runtime_right_itf.yaml"
SCENARIO = yaml.safe_load(
    (ROOT / "configs" / "scenarios" / "r1d_v12_active.yaml").read_text(
        encoding="utf-8"
    )
)
# 与 F01 复现参数一致的非默认有效值。
SAVED_OPTIONS: dict[str, Any] = {
    "limits_config": None,
    "max_tcp": 0.45,
    "terminal_exempt_steps": 4,
    "dq_max_fraction": 0.2,
    "ball_speed_perturb_pct": 10.0,
    "always_collision": True,
    "first_plan_flight": False,
    "fixed_direction": False,
    "hit_direction": None,
}


def _override_value(value: Any) -> str:
    """把 YAML 标量格式化为 ``--task-overrides`` 可解析的字符串。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _config() -> Any:
    """按冻结案例构造 R1D 配置（ITF 球场，其余物理与验收一致）。"""
    mode_cfg = SCENARIO["modes"]["torque"]
    argv = [
        "--robot-config",
        str(PROFILE),
        "--runtime-config",
        str(ITF_RUNTIME),
        "--seed",
        str(int(mode_cfg["expected"]["seed"])),
        "--no-plot",
        "--serve-box",
        "--ball-speed",
        str(SCENARIO["common_physical"]["serve"]["ball_speed"]),
    ]
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


def _synthetic_result(config: Any, bundle: Any) -> V12EpisodeResult:
    """构造可被 writer 接受的最小结果（数组为零，仅验证装配/清单接缝）。"""
    nq = len(bundle.joint_names) + 7  # 主动关节 + 自由球（v2 产物 X 含球）
    nu = int(bundle.execution.NU)
    return V12EpisodeResult(
        config=config,
        robot_id=str(config.robot_id),
        mode=str(config.actuator_mode),
        joint_names=tuple(str(name) for name in bundle.joint_names),
        dt=float(config.dt),
        racket_id=(None if bundle.racket is None else str(bundle.racket.racket_id)),
        X=np.zeros((3, nq), dtype=np.float64),
        U=np.zeros((2, nu), dtype=np.float64),
        ball_pos=np.zeros((3, 3), dtype=np.float64),
        ball_vel=np.zeros((3, 3), dtype=np.float64),
        time=np.zeros(3, dtype=np.float64),
        racket_pos=np.zeros((3, 3), dtype=np.float64),
        hit_type="miss",
        hit_step=-1,
        active_contact=False,
        passive_contact=False,
        v_ee_at_hit=None,
        p_ee_at_hit=None,
        ball_pos_at_hit=None,
        ball_out_speed=None,
        ball_out_vy=None,
        metrics={"failure_reason": "synthetic"},
        model_dependency_manifest=[
            dict(item)
            for item in (bundle.robot_model_bundle.dependency_manifest or ())
        ],
    )


def test_artifact_records_result_dependency_manifest(tmp_path: Path) -> None:
    """writer 消费结果里的装配清单，而不是退化为按配置推导。"""
    config = resolve_runtime_options(_config(), SAVED_OPTIONS)
    bundle = build_v12_robot(config)
    config = apply_runtime_limits(config, bundle)
    manifest = bundle.robot_model_bundle
    assert manifest is not None
    assert manifest.dependency_manifest, "R1D 装配必须产生依赖清单"

    result = _synthetic_result(config, bundle)
    write_episode_artifacts(result, tmp_path)

    artifacts = load_episode_artifacts(tmp_path)
    payload = artifacts.payload
    assert payload["dependency_manifest_source"] == "result"
    assert payload["dependency_manifest_version"] == 1
    files = payload["dependencies"]["files"]
    assert files
    assert any(item["role"] == "model" for item in files)
    assert payload["dependencies"]["missing"] == []


def test_replay_bundle_restores_saved_limits(tmp_path: Path) -> None:
    """回放不能落回默认限制：F01 的 replay 侧必须消费保存值。"""
    config = resolve_runtime_options(_config(), SAVED_OPTIONS)
    bundle = build_v12_robot(config)
    config = apply_runtime_limits(config, bundle)
    saved = config.runtime_overrides
    assert saved["max_tcp"]["effective"] == pytest.approx(0.45)
    assert saved["terminal_exempt_steps"]["effective"] == 4
    write_episode_artifacts(_synthetic_result(config, bundle), tmp_path)

    replay = build_replay_bundle(tmp_path)
    assert replay.robot_limits.max_tcp_speed == pytest.approx(0.45)
    assert replay.robot_limits.terminal_exempt_steps == 4
    expected_dq = np.asarray(saved["dq_max"], dtype=np.float64)
    assert replay.robot_limits.dq_max is not None
    np.testing.assert_allclose(replay.robot_limits.dq_max, expected_dq)


def test_strict_mode_rejects_r1d_result_without_manifest(tmp_path: Path) -> None:
    """严格模式下 R1D 结果缺清单必须拒绝，不能写出弱证据产物。"""
    import src.sim.episode_artifacts as artifacts_module

    config = resolve_runtime_options(_config(), SAVED_OPTIONS)
    bundle = build_v12_robot(config)
    config = apply_runtime_limits(config, bundle)
    result = _synthetic_result(config, bundle)
    object.__setattr__(result, "model_dependency_manifest", [])
    assert artifacts_module.REQUIRE_RESULT_DEPENDENCY_MANIFEST is True
    with pytest.raises(ValueError, match="缺少装配依赖清单"):
        write_episode_artifacts(result, tmp_path)
