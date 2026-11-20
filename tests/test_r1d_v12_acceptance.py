"""R1D 右臂 V12 parity 冻结固定案例验收测试（T7 交付）。

读取 `configs/scenarios/r1d_v12_active.yaml` 冻结的物理配置与两模式 task_overrides，
在进程内以真实发球（serve-box、7 m/s、弹跳 restitution 0.75）+ ProfileSimEnv 解析弹跳
+ R1D V12 估计关节阻尼/armature 跑固定 seed 42 的 torque / position 各一回合。

验收判据（T6 `v12_episode_v2` 产物，不用旧 hit_type / pos_error 代替）：

* `hit.physical_hit is True` 且 `hit.physical_hit_type == "active"`；
* `hit.contact_events` 中存在 `surface_kind == "face"` 且 `physical_contact is True`
  且 `distance_m <= 0` 的真实拍面接触事件；
* 同一物理步 `face_speed_m_s > 0.3`；
* `episode.failure_reason is None`（触前无 safety_filter_stop / ball_unreachable）；
* 固定案例确定论：接触步、拍速、距离、回合步数与冻结期望逐位一致；
* 产物 `input_sha256` 与冻结场景文件记录的哈希一致（证明用的就是冻结物理配置）。

本测试只跑固定 seed 42 两回合（约 1.5–2 分钟）；20 回合留出集是
`experiment_data/r1d_v12_parity/validation/` 的运行器职责，不在普通单测中执行。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
from src.sim.episode_artifacts import write_episode_artifacts  # noqa: E402
from src.sim.v12_episode import run_v12_episode  # noqa: E402

SCENARIO_PATH = ROOT / "configs" / "scenarios" / "r1d_v12_active.yaml"
SCENARIO: dict[str, Any] = yaml.safe_load(SCENARIO_PATH.read_text(encoding="utf-8"))

# 验收阈值（与 src/ilqt/components/sim_component.py 的 ACTIVE 阈值一致，不得放宽）
_MIN_ACTIVE_FACE_SPEED_M_S = 0.3


def _override_value(value: Any) -> str:
    """把 YAML 标量格式化为 `--task-overrides` 可解析的字符串。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _resolve_mode_config(mode: str) -> Any:
    """按冻结场景文件构造 ResolvedV12Config（与 CLI 缺省/显式行为等价）。"""
    common = SCENARIO["common_physical"]
    mode_cfg = SCENARIO["modes"][mode]
    seed = int(mode_cfg["expected"]["seed"])

    argv = [
        "--robot-config", str(ROOT / common["robot_profile"]),
        "--seed", str(seed),
        "--no-plot",
        "--serve-box",
        "--ball-speed", str(common["serve"]["ball_speed"]),
    ]
    if mode_cfg.get("actuator_mode") == "position":
        argv.append("--position-mode")

    overrides = v12._parse_task_overrides(
        [f"{key}={_override_value(value)}" for key, value in mode_cfg["overrides"].items()]
    )
    assert overrides is not None

    parser = v12._build_parser()
    args = parser.parse_args(argv)
    args.task_overrides = overrides
    explicit = v12._explicit_cli_actions(parser, argv)
    return v12.resolve_v12_config(args, explicit)


def _face_contact_events(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """取产物中真实拍面接触事件（T6 contact_events）。"""
    hit = payload.get("hit") or {}
    events = hit.get("contact_events") or []
    return [
        dict(event)
        for event in events
        if str(event.get("surface_kind")) == "face"
        and bool(event.get("physical_contact"))
    ]


def _run_and_write(mode: str, tmp_path: Path) -> dict[str, Any]:
    """跑固定案例一回合并写 T6 v2 产物，返回 JSON payload。"""
    config = _resolve_mode_config(mode)
    result = run_v12_episode(config)
    payload = write_episode_artifacts(result, tmp_path / mode)
    # 产物必须能落盘为 JSON（allow_nan=False 已在 writer 内校验）。
    assert (tmp_path / mode / "v12_episode.json").is_file()
    return payload


@pytest.mark.parametrize("mode", ["torque", "position"])
def test_frozen_seed42_real_active_face_contact(mode: str, tmp_path: Path) -> None:
    """两模式固定 seed42：真实拍面物理接触 + 同物理步拍速 > 0.3 + 无安全拒绝。"""
    payload = _run_and_write(mode, tmp_path)
    expected = SCENARIO["modes"][mode]["expected"]
    hit = payload["hit"]
    episode = payload["episode"]

    # ① T6 产物判定：physical_hit / active
    assert payload["schema"] == "v12_episode_v2"
    assert payload["robot_id"] == "unitree_r1d"
    assert payload["mode"] == mode
    assert hit["physical_hit"] is True
    assert hit["physical_hit_type"] == "active"
    assert hit["physical_hit_rule"] == (
        "first_physical_face_contact_face_speed_0p3"
    )

    # ② contact_events 真实拍面接触（正距 margin 不算碰触）
    events = _face_contact_events(payload)
    assert events, "冻结案例必须记录真实 face 物理接触事件"
    event = events[0]
    assert event["other_geom_name"] == expected["contact_other_geom_name"]
    assert float(event["distance_m"]) <= 0.0
    assert int(event["step"]) == int(expected["physical_hit_step"])
    # ③ 同一物理步拍速严格大于阈值
    assert float(event["face_speed_m_s"]) > _MIN_ACTIVE_FACE_SPEED_M_S
    assert float(event["face_speed_m_s"]) == pytest.approx(
        float(expected["face_speed_m_s"]), rel=0.0, abs=1e-12,
    )
    assert float(event["distance_m"]) == pytest.approx(
        float(expected["contact_distance_m"]), rel=0.0, abs=1e-12,
    )
    assert int(hit["physical_hit_step"]) == int(expected["physical_hit_step"])

    # ④ 回合本身无安全拒绝/不可达（触前无 safety_filter_stop）
    assert episode["failure_reason"] is None
    assert payload["failure_reason"] is None
    assert int(episode["total_steps"]) == int(expected["total_steps"])
    assert int(episode["safe_steps"]) == int(episode["total_steps"])
    # 旧 hit 字段同时保留（分类标准不退化）
    assert hit["legacy_hit"] is True

    # ⑤ 冻结物理配置哈希一致（runtime/task/racket/model/assets）
    for role, digest in SCENARIO["input_sha256"].items():
        assert payload["input_sha256"][role] == digest, (
            f"input_sha256[{role}] 与冻结场景不一致："
            f"{payload['input_sha256'][role]!r} != {digest!r}"
        )


def test_frozen_scenario_metadata_consistent() -> None:
    """冻结场景文件自洽性：两模式 expected seed 相同、dev=42、留出集不重叠。"""
    assert SCENARIO["dev_seed"] == 42
    assert SCENARIO["holdout_seeds"] == list(range(10))
    assert SCENARIO["dev_seed"] not in SCENARIO["holdout_seeds"]
    for mode in ("torque", "position"):
        expected = SCENARIO["modes"][mode]["expected"]
        assert int(expected["seed"]) == SCENARIO["dev_seed"]
        assert expected["physical_hit"] is True
        assert expected["physical_hit_type"] == "active"
        assert float(expected["face_speed_m_s"]) > _MIN_ACTIVE_FACE_SPEED_M_S
        assert float(expected["contact_distance_m"]) <= 0.0
        assert expected["failure_reason"] is None
    # 两模式共用同一物理配置文件（不得按模式/seed 换拍或目标区）
    common = SCENARIO["common_physical"]
    assert common["robot_profile"] == "configs/robots/r1d_right.yaml"
    assert common["runtime_config"] == "configs/robots/r1d_v12_runtime_right.yaml"
    assert common["task_config"] == "configs/r1d_v12.yaml"
    assert common["racket_config"] == "configs/rackets/blade100_v9_reference.yaml"
    assert set(SCENARIO["input_sha256"]) >= {
        "runtime", "task", "racket", "model", "assets",
    }
