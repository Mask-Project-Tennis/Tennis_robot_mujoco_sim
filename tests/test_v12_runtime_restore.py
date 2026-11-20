"""V12 运行参数恢复回归测试（F01）。

覆盖 brief 的三条验收线：

* 真实执行路径：显式覆盖跑一次 RM65 → 序列化 ``result.config`` →
  ``config_from_dict`` → 不传任何覆盖再跑一次，两次「实际消费」的限位、
  碰撞开关、来球扰动与初值必须逐项一致（不靠早退或只比 JSON）；
* 优先级：显式 ``False`` / ``0`` 覆盖已保存值，``UNSET`` 继承已保存值，
  可空字段显式 ``None`` 清除覆盖并回落当前默认；
* limits 快照：删除原 limits 文件后仍从保存内容恢复；显式给新文件时
  完全按新文件重算，不叠加旧 effective。

所有真实回合使用 ``--horizon 60 --iter 2`` 的 RM65 短配置，避免测试时长
失控；断言只依赖「实际消费值」与几何量，不依赖是否命中。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("mujoco")

import rm65_mpc_v12 as v12  # noqa: E402
from src.ilqt.robot_limits import RobotLimits  # noqa: E402
from src.sim import (  # noqa: E402
    episode_runtime,
    v12_episode as v12_episode_module,
    v12_robot_adapter as adapter,
)
from src.sim.episode_artifacts import (  # noqa: E402
    build_replay_bundle,
    config_from_dict,
    config_to_dict,
    load_episode_artifacts,
    write_episode_artifacts,
)
from src.sim.episode_runtime import (  # noqa: E402
    apply_runtime_limits,
    apply_runtime_overrides,
    resolve_runtime_options,
    resolve_runtime_overrides,
)
from src.sim.v12_episode import run_v12_episode  # noqa: E402
from src.robot.constants import INIT_Q_REAL  # noqa: E402

_RM65_SHORT_ARGV = [
    "--serve-box",
    "--ball-speed", "7",
    "--seed", "42",
    "--no-plot",
    "--horizon", "60",
    "--iter", "2",
    "--near-iters", "2",
    "--first-plan-iters", "2",
    "--near-threshold", "30",
    "--replan-interval", "20",
]

# 与 brief F01 现象一致的一组覆盖：TCP 0.45、终段豁免 4、dq 0.2、
# 来球扰动 10%、碰撞常开。
_F01_OVERRIDES: dict[str, Any] = {
    "max_tcp": 0.45,
    "terminal_exempt_steps": 4,
    "dq_max_fraction": 0.2,
    "ball_speed_perturb_pct": 10.0,
    "always_collision": True,
}


def _resolve_short_config() -> Any:
    """用真实 CLI 解析器 + T3 解析层构造 RM65 短配置。"""
    parser = v12._build_parser()
    args = parser.parse_args(_RM65_SHORT_ARGV)
    explicit = v12._explicit_cli_actions(parser, _RM65_SHORT_ARGV)
    return v12.resolve_v12_config(args, explicit)


def _fake_bundle() -> SimpleNamespace:
    """构造可被 ``build_robot_limits``/``apply_runtime_limits`` 消费的 bundle。"""
    qdot = np.linspace(1.0, 2.0, 6)
    limits = RobotLimits(
        q_lower=np.full(6, -1.0),
        q_upper=np.full(6, 1.0),
        qdot_max=qdot.copy(),
        qddot_max=np.full(6, 10.0),
        u_min=np.full(6, -50.0),
        u_max=np.full(6, 50.0),
        max_tcp_speed=1.8,
        terminal_exempt_steps=20,
        dq_max=qdot * 0.005 * 0.5,
    )
    execution = SimpleNamespace(
        NU=6,
        model=SimpleNamespace(
            actuator_ctrlrange=np.tile(np.array([[-50.0, 50.0]]), (6, 1)),
        ),
    )
    return SimpleNamespace(
        robot_limits=limits,
        execution=execution,
        init_q=np.linspace(0.1, 0.6, 6),
    )


class _Captures:
    """记录一次真实回合里各组件「实际消费」的运行时参数。"""

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    def start_run(self) -> dict[str, Any]:
        record: dict[str, Any] = {}
        self.runs.append(record)
        return record

    @property
    def last(self) -> dict[str, Any]:
        return self.runs[-1]


def _install_captures(monkeypatch: pytest.MonkeyPatch) -> _Captures:
    """用 monkeypatch 在真实管线的组件构造点捕获实际消费值。"""
    captures = _Captures()

    real_build = v12_episode_module.build_v12_robot
    real_mpc = v12_episode_module.MPCController
    real_sim = v12_episode_module.SimComponent
    real_run = v12_episode_module.EpisodeRunner.run

    def _build_bundle(config: Any) -> Any:
        record = captures.start_run()
        bundle = real_build(config)
        record["bundle"] = bundle
        return bundle

    class _CapturingMPC(real_mpc):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            record = captures.last
            record["mpc_limits"] = kwargs.get(
                "robot_limits", args[2] if len(args) > 2 else None
            )
            super().__init__(*args, **kwargs)

    class _CapturingSimComponent(real_sim):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            record = captures.last
            record["sim_limits"] = self._limits
            record["sim_init_q"] = np.asarray(self._init_q, dtype=np.float64).copy()
            record["always_collision"] = bool(self._always_collision)

    def _capture_run(
        self: Any, max_steps: int = 500, first_plan_flight: bool = False
    ) -> dict[str, Any]:
        captures.last["first_plan_flight"] = bool(first_plan_flight)
        return real_run(
            self, max_steps=max_steps, first_plan_flight=first_plan_flight
        )

    monkeypatch.setattr(v12_episode_module, "build_v12_robot", _build_bundle)
    monkeypatch.setattr(v12_episode_module, "MPCController", _CapturingMPC)
    monkeypatch.setattr(v12_episode_module, "SimComponent", _CapturingSimComponent)
    monkeypatch.setattr(v12_episode_module.EpisodeRunner, "run", _capture_run)
    return captures


def _assert_consumed_values(
    record: dict[str, Any],
    *,
    max_tcp: float,
    terminal_exempt_steps: int,
    dq_fraction: float,
    dt: float,
) -> None:
    """断言组件实际拿到的限位对象与初值（而非只比较配置字典）。"""
    limits = record["mpc_limits"]
    assert limits is not None
    assert limits is record["bundle"].robot_limits
    assert record["sim_limits"] is limits
    assert float(limits.max_tcp_speed) == pytest.approx(max_tcp)
    assert int(limits.terminal_exempt_steps) == terminal_exempt_steps
    qdot = np.asarray(limits.qdot_max, dtype=np.float64)
    dq = np.asarray(limits.dq_max, dtype=np.float64)
    assert np.allclose(dq, qdot * dt * dq_fraction, rtol=0.0, atol=1e-12)
    assert np.asarray(record["sim_init_q"]).shape == np.asarray(
        record["bundle"].init_q
    ).shape
    assert np.allclose(record["sim_init_q"], record["bundle"].init_q)


# ── 1. 真实执行路径：保存 → 反序列化 → 不传覆盖再跑 ────────────────────


def test_saved_runtime_overrides_are_consumed_by_real_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一份 config 反序列化再执行时，限位/碰撞/来球参数逐项保持。"""
    captures = _install_captures(monkeypatch)
    config = _resolve_short_config()
    dt = float(config.dt)

    first = run_v12_episode(config, **_F01_OVERRIDES)
    first_record = captures.last
    assert len(captures.runs) == 1
    assert first.X.shape[0] > 1
    assert first.metrics["total_steps"] > 0
    _assert_consumed_values(
        first_record,
        max_tcp=0.45,
        terminal_exempt_steps=4,
        dq_fraction=0.2,
        dt=dt,
    )
    assert first_record["always_collision"] is True
    assert first_record["first_plan_flight"] is False

    # 来球扰动的实际消费值：首帧球位置相对 serve.p0 偏移 0.2 m。
    serve_p0 = np.asarray(first.metrics["serve"]["p0"], dtype=np.float64)
    ball0 = np.asarray(first.ball_pos[0], dtype=np.float64)
    serve_v0 = np.asarray(first.metrics["serve"]["v0"], dtype=np.float64)
    v_dir = serve_v0 / (np.linalg.norm(serve_v0) + 1e-12)
    offset = ball0 - serve_p0
    assert float(np.linalg.norm(offset)) == pytest.approx(0.2, abs=1e-9)
    assert float(np.dot(offset, v_dir)) == pytest.approx(0.2, abs=1e-9)

    payload = config_to_dict(first.config)
    saved = payload["runtime_overrides"]
    assert saved["max_tcp"]["effective"] == pytest.approx(0.45)
    assert saved["terminal_exempt_steps"]["effective"] == 4
    assert saved["dq_max_fraction"]["effective"] == pytest.approx(0.2)
    assert saved["ball_speed_perturb_pct"] == pytest.approx(10.0)
    assert saved["always_collision"] is True

    restored = config_from_dict(payload)
    second = run_v12_episode(restored)
    second_record = captures.last
    assert len(captures.runs) == 2
    assert second.X.shape[0] > 1
    assert second.metrics["total_steps"] > 0
    _assert_consumed_values(
        second_record,
        max_tcp=0.45,
        terminal_exempt_steps=4,
        dq_fraction=0.2,
        dt=dt,
    )
    # M04：恢复执行必须逐位复用保存的实际生效向量，而不是重算/落回默认。
    np.testing.assert_array_equal(
        second_record["mpc_limits"].qdot_max, saved["qdot_max"]
    )
    np.testing.assert_array_equal(
        second_record["mpc_limits"].dq_max, saved["dq_max"]
    )
    assert second_record["always_collision"] is True
    assert second_record["first_plan_flight"] is False
    assert np.allclose(
        second_record["sim_init_q"], first_record["sim_init_q"], rtol=0.0, atol=0.0
    )

    serve_p0_2 = np.asarray(second.metrics["serve"]["p0"], dtype=np.float64)
    ball0_2 = np.asarray(second.ball_pos[0], dtype=np.float64)
    assert np.allclose(ball0_2 - serve_p0_2, offset, rtol=0.0, atol=1e-12)


# ── 2. 显式 False / 0 / None 的优先级 ──────────────────────────────────


def test_explicit_false_zero_and_none_override_saved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式 False/0 覆盖已保存的 True/非零，显式 None 清除回落默认。"""
    captures = _install_captures(monkeypatch)
    config = _resolve_short_config()
    first = run_v12_episode(config, **_F01_OVERRIDES)
    restored = config_from_dict(config_to_dict(first.config))
    dt = float(restored.dt)

    third = run_v12_episode(
        restored,
        always_collision=False,
        max_tcp=0.0,
        terminal_exempt_steps=None,
        dq_max_fraction=None,
    )
    record = captures.last
    assert len(captures.runs) == 2
    assert third.X.shape[0] > 1
    assert record["always_collision"] is False
    limits = record["mpc_limits"]
    assert float(limits.max_tcp_speed) == float("inf")
    assert int(limits.terminal_exempt_steps) == 20
    qdot = np.asarray(limits.qdot_max, dtype=np.float64)
    assert np.allclose(
        np.asarray(limits.dq_max, dtype=np.float64),
        qdot * dt * 0.5,
        rtol=0.0,
        atol=1e-12,
    )
    assert record["first_plan_flight"] is False

    runtime = third.config.runtime_overrides
    assert runtime["max_tcp"]["requested"] == 0.0
    assert runtime["max_tcp"]["effective"] is None
    assert runtime["terminal_exempt_steps"] == {
        "requested": None, "effective": 20,
    }
    assert runtime["dq_max_fraction"]["requested"] is None
    assert runtime["dq_max_fraction"]["effective"] == pytest.approx(0.5)
    assert runtime["always_collision"] is False
    assert runtime["ball_speed_perturb_pct"] == pytest.approx(10.0)


def test_unset_inherits_saved_and_none_clears_clearable_fields() -> None:
    """``UNSET`` 继承已保存值；五个可空字段显式 None 清除并重算默认。"""
    config = _resolve_short_config()
    resolved = resolve_runtime_options(config, dict(_F01_OVERRIDES))
    saved = config_from_dict(config_to_dict(resolved))

    inherited = resolve_runtime_options(saved, {})
    inherited_bundle = _fake_bundle()
    inherited_applied = apply_runtime_limits(inherited, inherited_bundle)
    inherited_runtime = inherited_applied.runtime_overrides
    assert inherited_runtime["max_tcp"]["effective"] == pytest.approx(0.45)
    assert inherited_runtime["terminal_exempt_steps"]["effective"] == 4
    assert inherited_runtime["dq_max_fraction"]["effective"] == pytest.approx(0.2)
    assert inherited_runtime["ball_speed_perturb_pct"] == pytest.approx(10.0)
    assert inherited_runtime["always_collision"] is True
    assert float(inherited_bundle.robot_limits.max_tcp_speed) == pytest.approx(0.45)
    assert int(inherited_bundle.robot_limits.terminal_exempt_steps) == 4

    cleared = resolve_runtime_options(
        saved,
        {
            "hit_direction": None,
            "limits_config": None,
            "max_tcp": None,
            "terminal_exempt_steps": None,
            "dq_max_fraction": None,
        },
    )
    cleared_bundle = _fake_bundle()
    cleared_applied = apply_runtime_limits(cleared, cleared_bundle)
    cleared_runtime = cleared_applied.runtime_overrides
    assert cleared_runtime["limits_config"] is None
    assert cleared_runtime["hit_direction"] is None
    assert cleared_runtime["max_tcp"] == {
        "requested": None, "effective": 1.8, "note": None,
    }
    assert cleared_runtime["terminal_exempt_steps"] == {
        "requested": None, "effective": 20,
    }
    assert cleared_runtime["dq_max_fraction"]["requested"] is None
    assert cleared_runtime["dq_max_fraction"]["effective"] == pytest.approx(0.5)
    assert float(cleared_bundle.robot_limits.max_tcp_speed) == pytest.approx(1.8)
    assert int(cleared_bundle.robot_limits.terminal_exempt_steps) == 20


# ── 3. 已保存 limits 内容：原文件删除后仍可恢复 ─────────────────────────


def test_limits_snapshot_restores_after_file_deleted(tmp_path: Path) -> None:
    """恢复只消费 ``runtime_overrides['limits_config']['content']``。"""
    source = ROOT / "configs" / "real_robot.yaml"
    limits_file = tmp_path / "limits.yaml"
    limits_file.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    config = _resolve_short_config()
    resolved = resolve_runtime_options(config, {"limits_config": limits_file})
    payload = config_to_dict(resolved)
    limits_file.unlink()
    assert not limits_file.exists()

    restored = config_from_dict(payload)
    bundle = _fake_bundle()
    applied = apply_runtime_limits(restored, bundle)

    assert not limits_file.exists()
    assert float(bundle.robot_limits.max_tcp_speed) == pytest.approx(1.0)
    assert int(bundle.robot_limits.terminal_exempt_steps) == 20
    assert np.allclose(
        np.asarray(bundle.robot_limits.qdot_max, dtype=np.float64),
        np.full(6, 3.14),
        rtol=0.0,
        atol=1e-9,
    )
    assert np.allclose(bundle.init_q, INIT_Q_REAL, rtol=0.0, atol=0.0)
    layer = applied.runtime_overrides["limits_config"]
    assert layer["path"] == str(limits_file.resolve())
    assert layer["content"]["robot_ip"] == "192.168.1.19"
    assert layer["content"]["max_qdot"] == pytest.approx([3.14] * 6)
    assert applied.runtime_overrides["max_tcp"]["effective"] == pytest.approx(1.0)
    assert applied.runtime_overrides["dq_max_fraction"]["effective"] == pytest.approx(
        0.5
    )


# ── 4. CLI：只转发 argv 显式出现的键 ──────────────────────────────────


def test_cli_forwards_only_explicit_argv_keys() -> None:
    """旧调用/未显式传参不得凭空注入覆盖（distinguish 未传与显式 False/0）。"""
    parser = v12._build_parser()
    base_args = parser.parse_args(_RM65_SHORT_ARGV)
    assert v12._runtime_explicit_options(parser, base_args, _RM65_SHORT_ARGV) == {}

    argv = [
        *_RM65_SHORT_ARGV,
        "--max-tcp", "0.45",
        "--terminal-exempt-steps", "4",
        "--dq-max-fraction", "0.2",
        "--always-collision",
        "--ball-speed-perturb-pct", "10",
        "--no-freeze-first-plan",
    ]
    args = parser.parse_args(argv)
    options = v12._runtime_explicit_options(parser, args, argv)
    assert options == {
        "first_plan_flight": True,
        "always_collision": True,
        "ball_speed_perturb_pct": 10.0,
        "max_tcp": 0.45,
        "terminal_exempt_steps": 4,
        "dq_max_fraction": 0.2,
    }

    limits_argv = [*_RM65_SHORT_ARGV, "--limits-config", "configs/real_robot.yaml"]
    limits_args = parser.parse_args(limits_argv)
    limits_options = v12._runtime_explicit_options(parser, limits_args, limits_argv)
    assert limits_options == {"limits_config": Path("configs/real_robot.yaml")}


# ── 5. 显式新 limits 文件：不得叠加旧 effective ────────────────────────
def _write_limits_variant(path: Path, *, max_tcp_speed: float, max_qdot: float) -> Path:
    """基于真机配置写一份只改 TCP/qdot 的临时 limits 文件。"""
    raw = yaml.safe_load((ROOT / "configs" / "real_robot.yaml").read_text("utf-8"))
    # 真实 YAML 把安全参数放在 safety 节下；RealRobotConfig._flatten 只按
    # 内层键名映射，故必须改嵌套节而不是新增顶层同名键。
    raw["safety"]["max_tcp_speed"] = float(max_tcp_speed)
    raw["safety"]["max_qdot"] = [float(max_qdot)] * 6
    path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def test_explicit_new_limits_file_recomputes_without_old_effective(
    tmp_path: Path,
) -> None:
    """显式新 limits 文件必须完全重算，旧文件派生的 effective/hint 全部丢弃。"""
    first_file = _write_limits_variant(
        tmp_path / "limits_a.yaml", max_tcp_speed=0.9, max_qdot=2.5
    )
    second_file = _write_limits_variant(
        tmp_path / "limits_b.yaml", max_tcp_speed=0.6, max_qdot=3.5
    )

    config = _resolve_short_config()
    first_resolved = resolve_runtime_options(config, {"limits_config": first_file})
    first_bundle = _fake_bundle()
    first_applied = apply_runtime_limits(first_resolved, first_bundle)
    first_runtime = first_applied.runtime_overrides
    assert first_runtime["max_tcp"]["effective"] == pytest.approx(0.9)
    assert first_runtime["qdot_max"] == pytest.approx([2.5] * 6)
    assert np.allclose(first_bundle.init_q, INIT_Q_REAL, rtol=0.0, atol=0.0)

    restored = config_from_dict(config_to_dict(first_applied))
    second_resolved = resolve_runtime_options(restored, {"limits_config": second_file})
    second_runtime = second_resolved.runtime_overrides
    # 显式新文件：旧 effective 被清空，且不再携带旧 qdot/dq hint。
    assert second_runtime["max_tcp"]["effective"] is None
    assert "qdot_max" not in second_runtime
    assert "dq_max" not in second_runtime

    second_bundle = _fake_bundle()
    second_applied = apply_runtime_limits(second_resolved, second_bundle)
    applied_runtime = second_applied.runtime_overrides
    assert applied_runtime["limits_config"]["path"] == str(second_file.resolve())
    assert applied_runtime["max_tcp"] == {
        "requested": None, "effective": 0.6, "note": None,
    }
    assert applied_runtime["qdot_max"] == pytest.approx([3.5] * 6)
    assert np.allclose(
        np.asarray(second_bundle.robot_limits.qdot_max, dtype=np.float64),
        np.full(6, 3.5),
        rtol=0.0,
        atol=1e-12,
    )
    assert np.allclose(second_bundle.init_q, INIT_Q_REAL, rtol=0.0, atol=0.0)
    # 关键：不得保留第一次文件派生的 0.9 / 2.5。
    assert float(applied_runtime["max_tcp"]["effective"]) != pytest.approx(0.9)
    assert applied_runtime["qdot_max"] != pytest.approx([2.5] * 6)

    # 同一份 restored config 直接 apply（未经显式新文件）时，仍按旧内容恢复。
    inherited = apply_runtime_limits(restored, _fake_bundle())
    assert inherited.runtime_overrides["max_tcp"]["effective"] == pytest.approx(0.9)
    assert inherited.runtime_overrides["qdot_max"] == pytest.approx([2.5] * 6)


# ── 6. M04：无 limits 文件时也恢复保存的 qdot/dq（默认配置变化不漂移） ─────


def _write_changed_default_config(tmp_path: Path) -> Path:
    """写一份只改 qdot 上限与 dq 分数的默认配置（模拟默认值后来变化）。"""
    raw = yaml.safe_load(
        (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    )
    raw["robot_limits"]["qdot_max_deg_s"] = [90.0] * 6
    raw["robot_limits"]["dq_max_fraction"] = 0.9
    path = tmp_path / "changed_default.yaml"
    path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def test_no_limits_file_restores_saved_joint_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M04：``limits_config=None`` 的产物在默认配置变化后仍恢复 qdot/dq。

    普通仿真产物没有 limits 文件，但 ``runtime_overrides`` 同样保存了实际
    生效的 ``qdot_max``/``dq_max``。回放与恢复执行都必须逐位复用它们，
    不能因为当前默认配置变化而悄悄改成新默认（本例 qdot 减半）。
    """
    captures = _install_captures(monkeypatch)
    config = _resolve_short_config()
    dt = float(config.dt)

    first = run_v12_episode(config, **_F01_OVERRIDES)
    assert len(captures.runs) == 1
    saved = dict(first.config.runtime_overrides)
    assert saved["limits_config"] is None
    saved_qdot = np.asarray(saved["qdot_max"], dtype=np.float64)
    saved_dq = np.asarray(saved["dq_max"], dtype=np.float64)
    assert saved_qdot.shape == (6,)
    assert saved_dq.shape == (6,)
    # 保存值来自写产物时的默认配置（180/225 deg/s 与 0.2 步长分数）。
    np.testing.assert_allclose(saved_dq, saved_qdot * dt * 0.2, rtol=0.0, atol=0.0)

    directory = tmp_path / "episode"
    write_episode_artifacts(first, directory)
    artifacts = load_episode_artifacts(directory)
    artifact_overrides = artifacts.payload["config"]["runtime_overrides"]
    assert artifact_overrides["limits_config"] is None
    assert artifact_overrides["qdot_max"] == saved_qdot.tolist()
    assert artifact_overrides["dq_max"] == saved_dq.tolist()

    # 只改「当前默认配置」（进程内），产物文件与快照保持原样。
    changed_path = _write_changed_default_config(tmp_path)
    monkeypatch.setattr(adapter, "_DEFAULT_CONFIG_PATH", changed_path)
    changed_limits = RobotLimits.from_config(
        yaml.safe_load(changed_path.read_text(encoding="utf-8"))["robot_limits"],
        dt=dt,
        ctrlrange=np.tile(np.array([[-50.0, 50.0]]), (6, 1)),
    )
    # 对照：变化后的默认值本身会把 qdot[0] 减半（修复前回放/执行漂移来源）。
    assert not np.array_equal(changed_limits.qdot_max, saved_qdot)
    assert float(changed_limits.qdot_max[0]) == pytest.approx(
        float(saved_qdot[0]) / 2.0
    )
    assert changed_limits.dq_max is not None
    assert not np.array_equal(changed_limits.dq_max, saved_dq)

    # 路径 1：回放 bundle 必须逐位恢复保存值（不依赖 limits 文件内容层）。
    replay = build_replay_bundle(directory)
    np.testing.assert_array_equal(replay.robot_limits.qdot_max, saved_qdot)
    np.testing.assert_array_equal(replay.robot_limits.dq_max, saved_dq)
    assert float(replay.robot_limits.max_tcp_speed) == pytest.approx(0.45)
    assert int(replay.robot_limits.terminal_exempt_steps) == 4

    # 路径 2：恢复执行（同一产物 → 反序列化 → 真实跑一遍）消费同样的值。
    restored = config_from_dict(artifacts.payload["config"])
    second = run_v12_episode(restored)
    assert len(captures.runs) == 2
    assert second.X.shape[0] > 1
    assert second.metrics["total_steps"] > 0
    record = captures.last
    limits = record["mpc_limits"]
    assert limits is record["bundle"].robot_limits
    np.testing.assert_array_equal(limits.qdot_max, saved_qdot)
    np.testing.assert_array_equal(limits.dq_max, saved_dq)
    assert float(limits.max_tcp_speed) == pytest.approx(0.45)
    assert int(limits.terminal_exempt_steps) == 4
    runtime = second.config.runtime_overrides
    assert runtime["qdot_max"] == saved_qdot.tolist()
    assert runtime["dq_max"] == saved_dq.tolist()
    assert runtime["dq_max_fraction"]["effective"] == pytest.approx(0.2)


# ── 7. hints 与显式请求的优先级：显式 > 已保存 > 当前默认 ────────────────


def test_saved_hint_priority_without_limits_file() -> None:
    """无 limits 内容时的合并优先级（纯函数，覆盖显式请求的各分支）。"""
    saved: dict[str, Any] = {
        "limits_config": None,
        "max_tcp": {"requested": 0.45, "effective": 0.45, "note": None},
        "terminal_exempt_steps": {"requested": 4, "effective": 4},
        "dq_max_fraction": {"requested": 0.2, "effective": 0.2},
        "qdot_max": [3.0, 3.0, 3.5, 3.5, 3.5, 3.5],
        "dq_max": [0.003, 0.003, 0.0035, 0.0035, 0.0035, 0.0035],
        "init_q": [0.2] * 6,
        "ball_speed_perturb_pct": 10.0,
        "always_collision": True,
        "first_plan_flight": False,
        "fixed_direction": False,
        "hit_direction": None,
    }
    dt = 0.005

    # 未显式请求：qdot/dq/init_q 全部按保存值恢复。
    inherited = resolve_runtime_overrides({}, saved)
    assert inherited["limits_config"] is None
    assert inherited["qdot_max"] == saved["qdot_max"]
    assert inherited["dq_max"] == saved["dq_max"]
    assert inherited["init_q"] == saved["init_q"]
    inherited_bundle = _fake_bundle()
    applied = apply_runtime_overrides(inherited_bundle, inherited, dt=dt)
    np.testing.assert_array_equal(
        inherited_bundle.robot_limits.qdot_max, saved["qdot_max"]
    )
    np.testing.assert_array_equal(
        inherited_bundle.robot_limits.dq_max, saved["dq_max"]
    )
    np.testing.assert_array_equal(inherited_bundle.init_q, saved["init_q"])
    assert float(inherited_bundle.robot_limits.max_tcp_speed) == pytest.approx(0.45)
    assert int(inherited_bundle.robot_limits.terminal_exempt_steps) == 4
    assert applied["dq_max_fraction"]["effective"] == pytest.approx(0.2)

    # 显式 dq_max_fraction：丢弃 saved dq hint，按恢复后的 qdot 重算。
    fraction_resolved = resolve_runtime_overrides({"dq_max_fraction": 0.35}, saved)
    assert "dq_max" not in fraction_resolved
    assert fraction_resolved["qdot_max"] == saved["qdot_max"]
    fraction_bundle = _fake_bundle()
    fraction_applied = apply_runtime_overrides(
        fraction_bundle, fraction_resolved, dt=dt
    )
    np.testing.assert_array_equal(
        fraction_bundle.robot_limits.qdot_max, saved["qdot_max"]
    )
    np.testing.assert_array_equal(
        fraction_bundle.robot_limits.dq_max,
        np.asarray(saved["qdot_max"], dtype=np.float64) * dt * 0.35,
    )
    assert fraction_applied["dq_max_fraction"]["effective"] == pytest.approx(0.35)

    # 显式清除 limits_config（现有语义）：旧 effective/hint 全部丢弃。
    cleared = resolve_runtime_overrides({"limits_config": None}, saved)
    assert "qdot_max" not in cleared
    assert "dq_max" not in cleared
    assert "init_q" not in cleared
    cleared_bundle = _fake_bundle()
    apply_runtime_overrides(cleared_bundle, cleared, dt=dt)
    np.testing.assert_array_equal(
        cleared_bundle.robot_limits.qdot_max, np.linspace(1.0, 2.0, 6)
    )

    # 显式 qdot_max 不是公开请求键：必须报错，不能静默被 saved hint 覆盖。
    with pytest.raises(ValueError, match="未知的运行时覆盖键"):
        resolve_runtime_overrides({"qdot_max": [1.0] * 6}, saved)
    # 应用层已拿到的显式向量直接生效（优先级高于任何 saved 值）。
    explicit_bundle = _fake_bundle()
    apply_runtime_overrides(
        explicit_bundle,
        {"qdot_max": [1.0] * 6, "dq_max": [0.002] * 6},
        dt=dt,
    )
    np.testing.assert_array_equal(
        explicit_bundle.robot_limits.qdot_max, np.full(6, 1.0)
    )
    np.testing.assert_array_equal(
        explicit_bundle.robot_limits.dq_max, np.full(6, 0.002)
    )


def test_episode_runtime_does_not_import_episode_artifacts() -> None:
    """R0：episode_runtime 不得 import episode_artifacts（避免循环依赖）。"""
    source = Path(episode_runtime.__file__).read_text(encoding="utf-8")
    forbidden = (
        "import episode_artifacts",
        "from src.sim.episode_artifacts",
        "from .episode_artifacts",
    )
    assert not any(token in source for token in forbidden)
