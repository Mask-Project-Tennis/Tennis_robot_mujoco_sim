"""T6 多机器人回放 + viewer 数据切分测试（先红后绿）。

覆盖 brief §6.4-6.5 与计划 T6 行 154-158：

* ``src/sim/viewer.py`` 的播放布局按 ``env.NQ`` / 球自由关节地址解析，
  不再假设"臂 qpos 前 NQ 槽 + 球在 NQ"（RM65 双臂 12+7、R1D 单臂 7+7）；
* ``replay_episode_artifacts`` 按**关节名**写入状态（``set_replay_state``），
  默认 1.0 速度、慢放为显式参数；回放不改写已保存 JSON/NPZ 与命中字段，
  不做合成反弹；
* 快照回放：外部 runtime/racket 被修改后仍用快照副本重建，复现相同 TCP；
  快照缺失且外部哈希不一致时显式拒绝；
* 旧 RM65 ``replay_trajectory`` 兼容（由 ``tests/test_replay.py`` 继续覆盖，
  本文件只做新 API 的真实验证）。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("mujoco")

import yaml  # noqa: E402

import rm65_mpc_v12 as v12  # noqa: E402
from src.robot.robot_profile import load_robot_profile  # noqa: E402
from src.robot.runtime_parameters import load_runtime_parameters  # noqa: E402
from src.sim import episode_artifacts as artifacts  # noqa: E402
from src.sim.rm65_env import RM65Env  # noqa: E402
from src.sim.v12_episode import V12EpisodeResult, run_v12_episode  # noqa: E402
from src.sim.viewer import (  # noqa: E402
    PlaybackLayout,
    arm_state_from_frame,
    control_plot_labels,
    resolve_playback_layout,
    write_playback_state,
)

_RM65_MODEL_PATH = ROOT / "src" / "robot" / "rm65_model.xml"
_JOINT_NAMES_RM65 = (
    "r_joint1", "r_joint2", "r_joint3",
    "r_joint4", "r_joint5", "r_joint6",
)
_R1D_PROFILE = ROOT / "configs" / "robots" / "r1d_right.yaml"
_R1D_RUNTIME = ROOT / "configs" / "robots" / "r1d_v12_runtime_right.yaml"
_R1D_RACKET = ROOT / "configs" / "rackets" / "blade100_v9_reference.yaml"

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


# ── 1. viewer 播放布局（headless 数据切分）────────────────────────────


def test_playback_layout_rm65_real_env_splits_arm_and_ball() -> None:
    """RM65：臂 qpos 0-11、球 qpos 12-18；不得把球切到 NQ 槽。"""
    env = RM65Env(_RM65_MODEL_PATH)
    layout = resolve_playback_layout(env, 12)
    assert isinstance(layout, PlaybackLayout)
    assert layout.joint_count == 6
    assert layout.arm_qpos_indices == (0, 1, 2, 3, 4, 5)
    assert layout.arm_qvel_indices == (0, 1, 2, 3, 4, 5)
    assert layout.ball_qpos_start == 12

    frame = np.arange(12, dtype=np.float64)
    ball = np.array([0.3, -0.6, 2.2])
    # 左臂槽位哨兵值：播放不应改动
    env.data.qpos[6:12] = 0.5
    arm_state_from_frame(frame, layout)
    write_playback_state(env.data, layout, frame[:6], frame[6:], ball)
    np.testing.assert_allclose(env.data.qpos[0:6], frame[:6])
    np.testing.assert_allclose(env.data.qvel[0:6], frame[6:])
    np.testing.assert_allclose(env.data.qpos[6:12], np.full(6, 0.5))
    np.testing.assert_allclose(env.data.qpos[12:15], ball)
    np.testing.assert_allclose(env.data.qpos[15:19], [1.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(("nq", "ball_start"), [(6, 6), (7, 7)])
def test_playback_layout_r1d_six_or_seven_joints(nq: int, ball_start: int) -> None:
    """R1D nq=6/7：臂占前 nq 槽，球自由关节紧随其后，滑条数=臂关节数。"""
    env = type(
        "FakeProfileEnv",
        (),
        {
            "NQ": nq,
            "NV": nq,
            "_ball_qpos_start": ball_start,
            "_ball_qvel_start": ball_start,
        },
    )()
    layout = resolve_playback_layout(env, 2 * nq)
    assert layout.joint_count == nq
    assert layout.arm_qpos_indices == tuple(range(nq))
    assert layout.ball_qpos_start == ball_start

    data = type(
        "FakeData",
        (),
        {
            "qpos": np.zeros(ball_start + 7, dtype=np.float64),
            "qvel": np.zeros(ball_start + 6, dtype=np.float64),
        },
    )()
    frame = np.arange(2 * nq, dtype=np.float64)
    ball = np.array([0.1, 0.2, 0.3])
    write_playback_state(data, layout, frame[:nq], frame[nq:], ball)
    np.testing.assert_allclose(data.qpos[0:nq], frame[:nq])
    np.testing.assert_allclose(data.qvel[0:nq], frame[nq:])
    np.testing.assert_allclose(data.qpos[ball_start:ball_start + 3], ball)
    np.testing.assert_allclose(
        data.qpos[ball_start + 3:ball_start + 7], [1.0, 0.0, 0.0, 0.0],
    )


# ── 2. 真实回放：按名重建、不改写产物 ────────────────────────────────


def _make_absolute_rm65_profile(tmp_dir: Path) -> Path:
    """复制 RM65 描述并把 model_path 改为绝对路径（快照回放测试用）。"""
    source = ROOT / "configs" / "robots" / "rm65.yaml"
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        "model_path: ../../src/robot/rm65_model.xml",
        f"model_path: {_RM65_MODEL_PATH.as_posix()}",
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)
    profile = tmp_dir / "rm65_abs_profile.yaml"
    profile.write_text(text, encoding="utf-8")
    return profile


def _parse_and_resolve(argv: list[str]):
    """真实 CLI 解析 + 配置解析。"""
    parser = v12._build_parser()
    args = parser.parse_args(argv)
    explicit = v12._explicit_cli_actions(parser, argv)
    return v12.resolve_v12_config(args, explicit)


@pytest.fixture(scope="module")
def rm65_episode(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, object, Path]:
    """真实 RM65 短回合 + v2 产物（含绝对 model_path 的临时机器人描述）。"""
    base = tmp_path_factory.mktemp("t6_replay_rm65")
    profile = _make_absolute_rm65_profile(base / "profiles")
    config = _parse_and_resolve(
        ["--robot-config", str(profile), *_RM65_SHORT_ARGV]
    )
    result = run_v12_episode(config)
    out = base / "episode"
    artifacts.write_episode_artifacts(result, out)
    return out, result, profile


def test_replay_by_name_reproduces_racket_positions(
    rm65_episode: tuple[Path, object, Path],
) -> None:
    """回放按关节名重建状态，TCP 逐帧复现；球位来自保存的 NPZ。"""
    out, result, _ = rm65_episode
    replay = artifacts.replay_episode_artifacts(out, viewer=False)

    assert replay.speed == pytest.approx(1.0)
    assert replay.payload["schema"] == "v12_episode_v2"
    assert replay.viewer_opened is False
    np.testing.assert_allclose(
        replay.racket_positions, result.racket_pos, atol=1e-8,
    )
    with np.load(out / artifacts.ARTIFACT_NPZ_NAME) as npz:
        np.testing.assert_allclose(replay.ball_positions, npz["ball_pos"])
        assert replay.racket_positions.shape == npz["racket_pos"].shape


def test_replay_does_not_rewrite_saved_artifacts(
    rm65_episode: tuple[Path, object, Path],
) -> None:
    """回放只读产物：JSON/NPZ 字节与命中字段在回放前后完全一致。"""
    out, _, _ = rm65_episode
    json_path = out / artifacts.ARTIFACT_JSON_NAME
    npz_path = out / artifacts.ARTIFACT_NPZ_NAME
    before_json = json_path.read_bytes()
    before_npz = npz_path.read_bytes()
    before_payload = json.loads(before_json.decode("utf-8"))

    artifacts.replay_episode_artifacts(out, viewer=False, speed=3.0)

    assert json_path.read_bytes() == before_json
    assert npz_path.read_bytes() == before_npz
    after_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert after_payload["hit"] == before_payload["hit"]


def test_replay_missing_snapshot_with_mismatched_hash_is_rejected(
    rm65_episode: tuple[Path, object, Path],
    tmp_path: Path,
) -> None:
    """删掉快照后，外部机器人描述哈希不符 → 明确拒绝回放。"""
    out, _, profile = rm65_episode
    copied = tmp_path / "episode_without_snapshot"
    shutil.copytree(out, copied)
    shutil.rmtree(copied / artifacts.SNAPSHOT_DIR_NAME)
    # 外部机器人描述被篡改（model_path 指向不存在的文件 → 哈希改变）
    tampered = profile.read_text(encoding="utf-8").replace(
        f"model_path: {_RM65_MODEL_PATH.as_posix()}",
        "model_path: /nonexistent/tampered_model.xml",
    )
    profile.write_text(tampered, encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="哈希|不一致|快照"):
            artifacts.replay_episode_artifacts(copied, viewer=False)
    finally:
        # 还原共享 fixture 的原始描述，避免影响同模块后续测试
        original = tampered.replace(
            "model_path: /nonexistent/tampered_model.xml",
            f"model_path: {_RM65_MODEL_PATH.as_posix()}",
        )
        profile.write_text(original, encoding="utf-8")


def test_control_plot_labels_keep_rm65_legacy_defaults() -> None:
    """评审 N2：n=6 保持旧 RM65 图注（Control Torques/tau_），n=7 用动态标签。"""
    title, titles, ylabel = control_plot_labels(
        6, list(_JOINT_NAMES_RM65), "Torque (N*m)"
    )
    assert title == "Control Torques"
    assert titles[0] == "tau_r_joint1"
    assert ylabel == "Torque (N*m)"

    title, titles, ylabel = control_plot_labels(
        7, [f"right_joint_{index}" for index in range(7)], "rad"
    )
    assert title == "Control Inputs"
    assert titles[0] == "u_right_joint_0"
    assert ylabel == "rad"


def _mirror_profile_with_tampered_model(
    base: Path,
    profile_source: Path,
    model_source: Path,
    *,
    old: str,
    new: str,
) -> Path:
    """复制 robot profile（逐字节相同）+ 篡改其相对 model_path 指向的模型。

    复刻评审 B1 探针：不修改仓库文件，通过把 payload 的
    ``config.robot_profile_path`` 指向 mirror 目录，让回放**实际加载**
    被篡改的模型，同时 profile 本身的 sha256 与产物记录一致（从而通过
    快照/来源校验，只有“实际加载模型哈希”能拦住）。
    """
    mirror = base / "mirror"
    profile_dir = mirror / "configs" / "robots"
    profile_dir.mkdir(parents=True)
    model_dir = mirror / "src" / "robot"
    model_dir.mkdir(parents=True)
    mirror_profile = profile_dir / profile_source.name
    shutil.copy2(profile_source, mirror_profile)
    model_text = model_source.read_text(encoding="utf-8")
    assert old in model_text, "探针锚点不存在，无法构造 B1 场景"
    (model_dir / model_source.name).write_text(
        model_text.replace(old, new, 1), encoding="utf-8"
    )
    return mirror_profile


def _patch_payload_profile(
    out: Path, profile: Path, *, record_profile_sha: bool = False
) -> None:
    """把产物 payload 的 robot_profile_path 指向 mirror profile（测试态）。

    ``record_profile_sha=True`` 时同步把 ``input_sha256["robot_profile"]``
    改写为 mirror profile 的哈希（模拟“产物本就记录了该相对 profile”的
    legacy RM65 场景），使来源校验通过、只留“实际加载模型”这一处缺口。
    """
    json_path = out / artifacts.ARTIFACT_JSON_NAME
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["config"]["robot_profile_path"] = str(profile)
    if record_profile_sha:
        payload["input_sha256"]["robot_profile"] = hashlib.sha256(
            profile.read_bytes()
        ).hexdigest()
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def test_replay_rejects_tampered_loaded_model_rm65(
    rm65_episode: tuple[Path, object, Path],
    tmp_path: Path,
) -> None:
    """评审 B1/R11：无快照可回退时，实际加载的外部模型被篡改 → 必须拒绝（RM65）。

    R11 修复后快照模型优先；有快照时外部模型被改不再影响回放（见
    ``test_episode_replay_sources.py``）。本测试删除快照，强制回放走外部
    来源，继续验证“校验文件 = 消费文件”的哈希拒绝仍然有效。
    """
    out, result, _ = rm65_episode
    copied = tmp_path / "episode_tampered_model"
    shutil.copytree(out, copied)
    shutil.rmtree(copied / artifacts.SNAPSHOT_DIR_NAME)
    # 用仓库原始相对 profile（legacy RM65 产物实际记录的就是它：sha 与产物一致），
    # 仅把其相对 model_path 指向的 mirror 模型改掉，profile 本身逐字节不变。
    mirror_profile = _mirror_profile_with_tampered_model(
        tmp_path,
        ROOT / "configs" / "robots" / "rm65.yaml",
        _RM65_MODEL_PATH,
        old='name="racket_center" pos="0 0 0.25"',
        new='name="racket_center" pos="0 0 0.35"',
    )
    _patch_payload_profile(copied, mirror_profile, record_profile_sha=True)
    with pytest.raises(ValueError, match="哈希|来源"):
        artifacts.replay_episode_artifacts(copied, viewer=False)

    # 未篡改的原产物仍可正常回放，且复现 racket_pos
    replay = artifacts.replay_episode_artifacts(out, viewer=False)
    np.testing.assert_allclose(
        replay.racket_positions, result.racket_pos, atol=1e-8,
    )


def _fabricated_r1d_result(tmp_path: Path) -> tuple[Path, V12EpisodeResult]:
    """用真实 R1D 配置构造最小 v2 产物（不跑整回合，仅测回放模型校验）。"""
    parser = v12._build_parser()
    argv = [
        "--robot-config", str(_R1D_PROFILE),
        "--seed", "42",
        "--no-plot",
        "--output", str(tmp_path / "r1d_min"),
    ]
    args = parser.parse_args(argv)
    explicit = v12._explicit_cli_actions(parser, argv)
    config = v12.resolve_v12_config(args, explicit)
    runtime = load_runtime_parameters(
        config.runtime_path, load_robot_profile(config.robot_profile_path)
    )
    # I1 后 R1D 生产结果必须携带装配依赖清单（严格写入）；测试用真实
    # 模型工厂生成同一份清单，避免手工伪造闭包证据。
    from src.robot.model_factory import load_robot_model

    model_bundle = load_robot_model(
        load_robot_profile(config.robot_profile_path), runtime
    )
    joint_names = tuple(str(name) for name in runtime.joint_names)
    n = len(joint_names)
    frames = 2
    dt = float(config.dt)
    X = np.zeros((frames, 2 * n), dtype=np.float64)
    U = np.zeros((frames - 1, n), dtype=np.float64)
    ball_pos = np.zeros((frames, 3), dtype=np.float64)
    ball_vel = np.zeros((frames, 3), dtype=np.float64)
    time_arr = np.arange(frames, dtype=np.float64) * dt
    result = V12EpisodeResult(
        config=config,
        robot_id=str(config.robot_id),
        mode=str(config.actuator_mode),
        joint_names=joint_names,
        dt=dt,
        racket_id=None,
        X=X,
        U=U,
        ball_pos=ball_pos,
        ball_vel=ball_vel,
        time=time_arr,
        racket_pos=np.zeros((frames, 3), dtype=np.float64),
        hit_type="miss",
        hit_step=-1,
        active_contact=False,
        passive_contact=False,
        v_ee_at_hit=None,
        p_ee_at_hit=None,
        ball_pos_at_hit=None,
        ball_out_speed=None,
        ball_out_vy=None,
        metrics={"failure_reason": "fabricated_for_replay_model_check"},
        model_dependency_manifest=[
            dict(item) for item in (model_bundle.dependency_manifest or ())
        ],
    )
    out = tmp_path / "r1d_min_episode"
    artifacts.write_episode_artifacts(result, out)
    return out, result


def test_replay_rejects_tampered_loaded_model_r1d(tmp_path: Path) -> None:
    """评审 B1/R11：无快照可回退时，R1D 外部模型被篡改 → 回放必须拒绝。"""
    out, _ = _fabricated_r1d_result(tmp_path)
    # 基线：未篡改产物可回放（真实加载 R1D 模型 + mesh）
    replay = artifacts.replay_episode_artifacts(out, viewer=False)
    assert replay.racket_positions.shape == (2, 3)

    profile = _R1D_PROFILE
    model_path = Path(load_robot_profile(profile).model_path)
    copied = tmp_path / "r1d_min_tampered"
    shutil.copytree(out, copied)
    # R11 后快照模型优先；删掉快照才能强制回放消费外部（被篡改）模型。
    shutil.rmtree(copied / artifacts.SNAPSHOT_DIR_NAME)
    mirror_profile = _mirror_profile_with_tampered_model(
        tmp_path,
        profile,
        model_path,
        old='name="racket_center" pos="0 0 0"',
        new='name="racket_center" pos="0 0 0.05"',
    )
    _patch_payload_profile(copied, mirror_profile)
    with pytest.raises(ValueError, match="哈希|来源"):
        artifacts.replay_episode_artifacts(copied, viewer=False)


# ── 3. R1D 快照隔离：外部 runtime/racket 改动不影响回放 ──────────────


def _tamper_yaml(path: Path, mutate) -> None:
    """按回调修改 YAML 文件（保持文件可加载）。"""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def test_r1d_replay_uses_snapshot_when_runtime_and_racket_change(
    tmp_path: Path,
) -> None:
    """R1D：外部 runtime/racket 语义被改后，回放仍用快照副本复现相同 TCP。"""
    runtime_copy = tmp_path / "runtime_right.yaml"
    racket_copy = tmp_path / "racket.yaml"
    shutil.copy2(_R1D_RUNTIME, runtime_copy)
    shutil.copy2(_R1D_RACKET, racket_copy)

    overrides = v12._parse_task_overrides(
        [
            "mpc.total_horizon=80",
            "mpc.fixed_horizon=40",
            "mpc.first_plan_iters=2",
            "mpc.max_iter_per_plan=2",
            "mpc.near_plan_iters=2",
        ]
    )
    parser = v12._build_parser()
    argv = [
        "--robot-config", str(_R1D_PROFILE),
        "--seed", "42",
        "--no-plot",
        "--output", str(tmp_path / "r1d_episode"),
    ]
    args = parser.parse_args(argv)
    args.task_overrides = overrides
    explicit = v12._explicit_cli_actions(parser, argv)
    config = v12.resolve_v12_config(args, explicit)
    config = replace(
        config, runtime_path=runtime_copy, racket_path=racket_copy,
    )

    result = run_v12_episode(config)
    out = tmp_path / "r1d_episode"
    payload = artifacts.write_episode_artifacts(result, out)

    # 快照必须逐文件记录 runtime/racket 的 sha256（快照自包含前提）
    manifest = json.loads(
        (out / artifacts.SNAPSHOT_DIR_NAME
         / artifacts.SNAPSHOT_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    roles = {entry["role"] for entry in manifest["files"]}
    assert {"robot_profile", "runtime", "racket"} <= roles
    for entry in manifest["files"]:
        copied = out / artifacts.SNAPSHOT_DIR_NAME / entry["relative_path"]
        assert copied.is_file()
        assert payload["input_sha256"][entry["role"]] == entry["sha256"]

    # 外部 runtime/racket 语义改动（若回放误用外部文件，TCP 会变）
    _tamper_yaml(
        runtime_copy,
        lambda data: data["scene"].__setitem__("lift_displacement", 0.30),
    )
    _tamper_yaml(
        racket_copy,
        lambda data: data["mount"].__setitem__("position", [0.0, 0.0, 0.05]),
    )

    replay = artifacts.replay_episode_artifacts(out, viewer=False)
    assert replay.payload["robot_id"] == "unitree_r1d"
    assert replay.racket_positions.shape == result.racket_pos.shape
    np.testing.assert_allclose(
        replay.racket_positions, result.racket_pos, atol=1e-6,
    )

    # 快照缺失且外部哈希不符 → 拒绝，而不是用被改文件静默回放
    copied = tmp_path / "r1d_without_snapshot"
    shutil.copytree(out, copied)
    shutil.rmtree(copied / artifacts.SNAPSHOT_DIR_NAME)
    with pytest.raises(ValueError, match="哈希|不一致|快照"):
        artifacts.replay_episode_artifacts(copied, viewer=False)


def test_replay_warns_when_payload_lacks_model_hash(
    tmp_path: Path,
    rm65_episode: tuple[Path, object, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """旧 payload 缺 ``input_sha256["model"]`` 时：仅 warning 兼容，不拒绝回放。"""
    out, result, _ = rm65_episode
    copied = tmp_path / "episode_no_model_hash"
    shutil.copytree(out, copied)
    json_path = copied / artifacts.ARTIFACT_JSON_NAME
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["input_sha256"].pop("model", None)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with caplog.at_level("WARNING"):
        replay = artifacts.replay_episode_artifacts(copied, viewer=False)

    assert replay.racket_positions.shape == result.racket_pos.shape
    np.testing.assert_allclose(
        replay.racket_positions, result.racket_pos, atol=1e-8,
    )
    assert any(
        "缺少模型文件哈希" in record.message for record in caplog.records
    ), [record.message for record in caplog.records]
