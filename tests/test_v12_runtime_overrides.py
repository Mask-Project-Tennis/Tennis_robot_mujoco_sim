"""R16 运行时覆盖规范化与配置序列化回归测试。

覆盖三点：

- ``ResolvedV12Config.runtime_overrides`` 必须有默认值、深拷贝、可参与
  ``dataclasses.replace``/相等比较，并能经 ``config_to_dict`` /
  ``config_from_dict`` 无损 round-trip（序列化在 T6 产物层，不在本任务
  允许修改的文件内，故只做黑盒验证）。
- ``_apply_limits_overrides`` 返回的规范化 dict 必须记录实际生效值：
  ``max_tcp=0`` → ``effective=None``、``limits_config`` 含绝对路径 +
  sha256 + 解析内容、``dq_max_fraction`` 由实际 ``dq_max`` 反推。
- 快照 ``resolved_config.json`` 与 ``config_to_dict`` 同源，可用
  ``config_from_dict`` 还原为同一份配置。

全部测试不跑整回合、不加载 MuJoCo 模型（远低于验收测试成本）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.ilqt.robot_limits import RobotLimits
from src.robot.constants import INIT_Q_REAL
from src.sim.episode_artifacts import (
    SNAPSHOT_CONFIG_NAME,
    _write_snapshot_metadata,
    config_from_dict,
    config_to_dict,
)
from src.sim.v12_config import ResolvedV12Config, resolve_v12_config
from src.sim.v12_episode import _apply_limits_overrides, _runtime_option_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]
LIMITS_CONFIG = REPO_ROOT / "configs" / "real_robot.yaml"

# ---------------------------------------------------------------------------
# 轻量配置 fixture（纯 Python，不加载模型）
# ---------------------------------------------------------------------------


def _args(**overrides: object) -> argparse.Namespace:
    """构造只含所需字段的局部 argparse.Namespace（其余走 resolver 缺省）。"""
    return argparse.Namespace(**overrides)


def _resolve_rm65_snapshot() -> ResolvedV12Config:
    """解析 T0 基线命令对应的 RM65 配置（与 test_v12_config 同口径）。"""
    return resolve_v12_config(
        _args(serve_box=True, ball_speed=7.0, seed=42, no_plot=True),
        frozenset({"--serve-box", "--ball-speed", "--seed", "--no-plot"}),
    )


def _fake_bundle(nu: int = 6) -> SimpleNamespace:
    """构造 ``_apply_limits_overrides`` 所需的最小 bundle。

    ``limits_config`` 分支只读取 ``execution.NU`` /
    ``execution.model.actuator_ctrlrange``，其余字段仅被赋值。
    """
    limits = RobotLimits(
        q_lower=np.full(nu, -1.0),
        q_upper=np.full(nu, 1.0),
        qdot_max=np.linspace(1.0, 2.0, nu),
        qddot_max=np.full(nu, 10.0),
        u_min=np.full(nu, -50.0),
        u_max=np.full(nu, 50.0),
    )
    execution = SimpleNamespace(
        NU=nu,
        model=SimpleNamespace(
            actuator_ctrlrange=np.tile(
                np.array([[-50.0, 50.0]], dtype=np.float64), (nu, 1)
            )
        ),
    )
    return SimpleNamespace(
        robot_limits=limits,
        execution=execution,
        init_q=np.zeros(nu, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# ResolvedV12Config.runtime_overrides：默认值 / 深拷贝 / 相等 / 序列化
# ---------------------------------------------------------------------------


def test_runtime_overrides_defaults_to_empty_dict() -> None:
    """新字段有默认值：旧构造点（不传该字段）保持兼容。"""
    cfg = _resolve_rm65_snapshot()
    assert cfg.runtime_overrides == {}
    payload = config_to_dict(cfg)
    assert payload["runtime_overrides"] == {}
    assert config_from_dict(payload) == cfg


def test_runtime_overrides_are_deep_copied_and_immutable_field() -> None:
    """runtime_overrides 深拷贝输入；frozen dataclass 拒绝整字段赋值。"""
    source: dict[str, Any] = {
        "max_tcp": {"requested": 1.8, "effective": 1.8, "note": None},
        "qdot_max": [1.0, 2.0],
    }
    cfg = replace(_resolve_rm65_snapshot(), runtime_overrides=source)
    # 篡改源 dict 不得影响已构造配置
    source["max_tcp"]["requested"] = 99.0
    source["qdot_max"].append(3.0)
    source["extra"] = True
    assert cfg.runtime_overrides == {
        "max_tcp": {"requested": 1.8, "effective": 1.8, "note": None},
        "qdot_max": [1.0, 2.0],
    }
    with pytest.raises(FrozenInstanceError):
        cfg.runtime_overrides = {}  # type: ignore[misc]


def test_config_roundtrip_preserves_runtime_overrides() -> None:
    """config_to_dict → config_from_dict 无损还原（含嵌套 dict/list）。"""
    overrides: dict[str, Any] = {
        "limits_config": {
            "path": "C:/tmp/real_robot.yaml",
            "sha256": "ab" * 32,
            "content": {"max_tcp_speed": 1.0, "max_qdot": [3.14] * 6},
        },
        "max_tcp": {"requested": 0.0, "effective": None, "note": "不限速"},
        "terminal_exempt_steps": {"requested": 5, "effective": 5},
        "dq_max_fraction": {"requested": 0.2, "effective": 0.2},
        "qdot_max": [1.0, 2.0, 3.0],
        "dq_max": None,
        "nested": {"flag": True, "value": 0.5},
    }
    cfg = replace(_resolve_rm65_snapshot(), runtime_overrides=overrides)
    payload = config_to_dict(cfg)
    assert payload["runtime_overrides"] == overrides

    restored = config_from_dict(payload)
    assert isinstance(restored, ResolvedV12Config)
    assert restored == cfg
    assert restored.runtime_overrides == overrides
    # 二次序列化稳定（round-trip 幂等）
    assert config_to_dict(restored) == payload


def test_config_roundtrip_serializes_nested_arrays() -> None:
    """runtime_overrides 内的 ndarray 由产物层转 list，还原后仍相等。"""
    overrides = {"qdot_max": np.array([1.0, 2.0, 3.0])}
    cfg = replace(_resolve_rm65_snapshot(), runtime_overrides=overrides)
    payload = config_to_dict(cfg)
    assert payload["runtime_overrides"]["qdot_max"] == [1.0, 2.0, 3.0]
    restored = config_from_dict(payload)
    assert restored == cfg


# ---------------------------------------------------------------------------
# _apply_limits_overrides：实际生效值规范化
# ---------------------------------------------------------------------------


def test_apply_limits_overrides_records_nondefault_values() -> None:
    """非默认 max_tcp/terminal_exempt_steps/dq_max_fraction 全部落盘。"""
    bundle = _fake_bundle(nu=6)
    qdot_before = np.asarray(bundle.robot_limits.qdot_max, dtype=np.float64).copy()

    result = _apply_limits_overrides(
        bundle,  # type: ignore[arg-type]
        limits_config=None,
        max_tcp=1.25,
        terminal_exempt_steps=5,
        dq_max_fraction=0.2,
        dt=0.005,
    )

    assert result["limits_config"] is None
    assert result["max_tcp"] == {
        "requested": 1.25,
        "effective": 1.25,
        "note": None,
    }
    assert result["terminal_exempt_steps"] == {"requested": 5, "effective": 5}
    assert result["dq_max_fraction"]["requested"] == pytest.approx(0.2)
    assert result["dq_max_fraction"]["effective"] == pytest.approx(0.2)
    assert result["qdot_max"] == [float(x) for x in qdot_before]
    expected_dq = qdot_before * 0.005 * 0.2
    assert result["dq_max"] == pytest.approx([float(x) for x in expected_dq])
    # 未切 limits_config 时 init_q 保持 bundle 原值（这里是全零假值）
    assert result["init_q"] == [0.0] * 6
    # 限位对象本身也被改写
    assert bundle.robot_limits.max_tcp_speed == pytest.approx(1.25)
    assert bundle.robot_limits.terminal_exempt_steps == 5
    assert bundle.robot_limits.dq_max == pytest.approx(expected_dq)
    # 规范化 dict 必须是 JSON 原生类型（允许直接落盘）
    json.dumps(result)


def test_apply_limits_overrides_max_tcp_zero_means_unlimited() -> None:
    """max_tcp=0 → effective=None（不得把 inf 写进 JSON），并给出说明。"""
    bundle = _fake_bundle(nu=6)
    result = _apply_limits_overrides(
        bundle,  # type: ignore[arg-type]
        limits_config=None,
        max_tcp=0.0,
        terminal_exempt_steps=None,
        dq_max_fraction=None,
        dt=0.005,
    )

    assert result["max_tcp"]["requested"] == 0.0
    assert result["max_tcp"]["effective"] is None
    assert result["max_tcp"]["note"] == "max_tcp=0 → 不限速（effective=null）"
    assert np.isinf(bundle.robot_limits.max_tcp_speed)
    # dq_max 未被触碰、requested/effective 均为 None
    assert result["dq_max"] is None
    assert result["dq_max_fraction"] == {"requested": None, "effective": None}
    assert result["terminal_exempt_steps"] == {
        "requested": None,
        "effective": 20,
    }
    json.dumps(result, allow_nan=False)


def test_apply_limits_overrides_records_limits_config_snapshot() -> None:
    """limits_config → 绝对路径 + sha256 + 解析内容，并采用真机初值。"""
    bundle = _fake_bundle(nu=6)
    result = _apply_limits_overrides(
        bundle,  # type: ignore[arg-type]
        limits_config=LIMITS_CONFIG,
        max_tcp=1.8,
        terminal_exempt_steps=None,
        dq_max_fraction=None,
        dt=0.005,
    )

    info = result["limits_config"]
    assert info is not None
    assert info["path"] == str(LIMITS_CONFIG.resolve())
    expected_sha = hashlib.sha256(LIMITS_CONFIG.read_bytes()).hexdigest()
    assert info["sha256"] == expected_sha
    content = info["content"]
    assert content["robot_ip"] == "192.168.1.19"
    assert content["max_tcp_speed"] == pytest.approx(1.0)
    assert content["max_qdot"] == pytest.approx([3.14] * 6)
    assert len(content) >= 20  # 全字段快照，不是仅挑几项

    # limits_config 构建后 max_tcp 覆盖仍生效
    assert result["max_tcp"]["requested"] == pytest.approx(1.8)
    assert result["max_tcp"]["effective"] == pytest.approx(1.8)
    # build_robot_limits 的缺省 dq_max_fraction=0.5 被反推出来
    assert result["dq_max_fraction"]["requested"] is None
    assert result["dq_max_fraction"]["effective"] == pytest.approx(0.5)
    # 真机路径把 init_q 切换为 INIT_Q_REAL
    np.testing.assert_array_equal(bundle.init_q, INIT_Q_REAL)
    assert result["init_q"] == pytest.approx(
        [float(x) for x in np.asarray(INIT_Q_REAL).ravel()]
    )
    json.dumps(result)


def test_runtime_option_overrides_are_json_native() -> None:
    """运行选项键名与参数同名；hit_direction None 保留 null、tuple 转 list。"""
    overrides = _runtime_option_overrides(
        ball_speed_perturb_pct=10.0,
        always_collision=True,
        first_plan_flight=False,
        fixed_direction=True,
        hit_direction=(1.0, 0.0, 0.0),
    )
    assert overrides == {
        "ball_speed_perturb_pct": 10.0,
        "always_collision": True,
        "first_plan_flight": False,
        "fixed_direction": True,
        "hit_direction": [1.0, 0.0, 0.0],
    }
    json.dumps(overrides, allow_nan=False)

    default = _runtime_option_overrides(
        ball_speed_perturb_pct=0.0,
        always_collision=False,
        first_plan_flight=False,
        fixed_direction=False,
        hit_direction=None,
    )
    assert default["hit_direction"] is None


# ---------------------------------------------------------------------------
# 快照同源：resolved_config.json == config_to_dict(config)
# ---------------------------------------------------------------------------


def test_snapshot_resolved_config_matches_config_payload(tmp_path: Path) -> None:
    """T6 快照写盘内容与 config_to_dict 完全一致，并可还原配置。"""
    cfg = replace(
        _resolve_rm65_snapshot(),
        runtime_overrides={
            "max_tcp": {"requested": 1.8, "effective": 1.8, "note": None},
            "qdot_max": [3.0] * 7,
        },
    )
    payload = {
        "schema": "v12_episode_v2",
        "config": config_to_dict(cfg),
        "assets": {
            "status": "ok",
            "count": 0,
            "digest": "test",
            "missing": [],
            "files": [],
        },
        "scene_snapshot": {},
    }

    snapshot_dir = _write_snapshot_metadata(tmp_path, {}, payload)
    on_disk = json.loads(
        (snapshot_dir / SNAPSHOT_CONFIG_NAME).read_text(encoding="utf-8")
    )

    assert on_disk == payload["config"]
    assert on_disk["runtime_overrides"] == cfg.runtime_overrides
    assert config_from_dict(on_disk) == cfg
