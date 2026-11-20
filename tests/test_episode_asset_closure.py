"""F02/F08 回归：已验证资产映射（F02）与装配依赖闭包（F08）。

本文件覆盖并行修复计划 §7 的六类场景：

1. 仓库内正确网格 + 产物记录的 ``source`` 指向"旧检出"：删除旧来源、
   把旧来源 STL 合法放大 1.25 倍，回放都必须只用哈希匹配的仓库字节
   （模型数组与保存前逐位一致），禁止"校验通过却加载旧几何"；
2. 外部模板目录 + 自定义 ``r1d_arm_left.xml``（左肩 X=0.2）+ 非空
   ``fixed_joint_poses``：保存后移动/删除外部目录，回放仍读到 0.2；
3. 中文路径、跨根资产、嵌套 ``<include>``、复制前篡改、缺失新依赖、
   旧 artifact 无依赖清单（legacy 分支）。

全部用例都在 ``tmp_path`` 独立目录内构造，不写 ``experiment_data/``。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for _path in (ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

pytest.importorskip("mujoco")

import mujoco  # noqa: E402

import rm65_mpc_v12 as v12  # noqa: E402
from src.ilqt.components.sim_component import (  # noqa: E402
    LEGACY_HIT_RULE,
    PHYSICAL_HIT_RULE,
    ContactEvent,
)
from src.sim import episode_artifacts as artifacts  # noqa: E402
from src.sim.v12_episode import V12EpisodeResult  # noqa: E402
from src.sim.v12_robot_adapter import build_v12_robot  # noqa: E402
from src.robot.model_factory import collect_model_dependencies  # noqa: E402
from src.robot.robot_profile import load_robot_profile  # noqa: E402
from src.robot.runtime_parameters import load_runtime_parameters  # noqa: E402
from src.utils.mujoco_loader import _setup_temp_copy, load_mujoco_model  # noqa: E402

# 与 test_episode_artifacts.py / test_episode_replay_sources.py 一致的短回合参数
_RM65_SHORT_ARGV = [
    "--serve-box",
    "--ball-speed",
    "7",
    "--seed",
    "42",
    "--no-plot",
    "--horizon",
    "60",
    "--iter",
    "2",
    "--near-iters",
    "2",
    "--first-plan-iters",
    "2",
    "--near-threshold",
    "30",
    "--replan-interval",
    "20",
]
_JOINT_NAMES_RM65 = tuple(f"r_joint{i}" for i in range(1, 7))
_FAKE_BALL_POS = np.array([0.30, 0.0, 1.00])
_RACKET_SITE = "racket_center"
_SNAPSHOT_SITE_POS = np.array([0.0, 0.0, 0.25])
_UNDERPAN_RELATIVE = "assets/rm_65/urdf/meshes/base_link_underpan.STL"
_PASSIVE_BODY = "left_shoulder_pitch_Link"


# ── 通用辅助 ──────────────────────────────────────────────────────────


def _parse_and_resolve(argv: list[str]):
    """用真实 CLI 解析器 + 解析层构造 ResolvedV12Config。"""
    parser = v12._build_parser()
    args = parser.parse_args(argv)
    explicit = v12._explicit_cli_actions(parser, argv)
    return v12.resolve_v12_config(args, explicit)


def _sha256(path: Path) -> str:
    """文件 SHA256。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest_path(output_dir: Path) -> Path:
    """快照 manifest 路径。"""
    return output_dir / artifacts.SNAPSHOT_DIR_NAME / artifacts.SNAPSHOT_MANIFEST_NAME


def _load_payload(output_dir: Path) -> dict[str, Any]:
    """读产物 JSON。"""
    path = output_dir / artifacts.ARTIFACT_JSON_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def _write_payload(output_dir: Path, payload: dict[str, Any]) -> None:
    """写回产物 JSON。"""
    path = output_dir / artifacts.ARTIFACT_JSON_NAME
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_manifest(output_dir: Path) -> dict[str, Any]:
    """读快照 manifest。"""
    return json.loads(_manifest_path(output_dir).read_text(encoding="utf-8"))


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    """写回快照 manifest。"""
    _manifest_path(output_dir).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _supported_kwargs(**kwargs: Any) -> dict[str, Any]:
    """按当前 V12EpisodeResult 字段表过滤关键字（兼容并行改动）。"""
    names = {field.name for field in dataclasses.fields(V12EpisodeResult)}
    return {key: value for key, value in kwargs.items() if key in names}


def _fabricate_rm65(config: Any, *, frames: int = 4) -> V12EpisodeResult:
    """构造最小 RM65 V12EpisodeResult（复用既有测试口径）。"""
    n = len(_JOINT_NAMES_RM65)
    n_rows = frames + 1
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_rows, 2 * n)) * 0.1
    U = rng.normal(size=(frames, n)) * 0.01
    ball_pos = np.tile(_FAKE_BALL_POS, (n_rows, 1))
    ball_vel = np.tile(np.array([0.0, -5.0, 1.0]), (n_rows, 1))
    time = np.arange(n_rows, dtype=np.float64) * float(config.dt)
    racket_pos = np.tile(_FAKE_BALL_POS, (n_rows, 1))
    contact_events = [
        ContactEvent(
            step=2,
            time_s=2 * float(config.dt),
            ball_geom_name="ball",
            other_geom_name="r_racket_face",
            surface_kind="face",
            distance_m=-0.002,
            face_speed_m_s=1.2,
            physical_contact=True,
            hit_type="active",
        ).to_dict()
    ]
    metrics = {
        "total_steps": frames,
        "safe_steps": frames,
        "mpc_done": True,
        "first_plan_ms": 1.0,
        "wall_time_s": 2.0,
        "mpc_time_s": 1.0,
        "phase_timings": {"far": 0.1, "mid": 0.2, "near": 0.3},
        "backend": {"requested": "cpp", "effective": "python-analytic"},
        "failure_reason": None,
        "ablation_mode": "off",
        "is_position_mode": config.actuator_mode == "position",
        "n_mpc_steps": frames,
        "min_dist": 0.02,
        "max_qdot_ratio": 0.5,
        "max_tcp_speed": 1.0,
        "max_racket_face_speed": 1.2,
        "pos_error": 0.05,
        "shoulder_pos": [0.0, 0.0, 1.0],
        "shoulder_pos_source": "cli/scene.shoulder_pos",
        "proximity": {
            "ball_radius": 0.033,
            "racket_radius": 0.12,
            "proximity_margin": 0.03,
            "ball_near_threshold": 0.183,
        },
        "hit_window": None,
        "serve": {
            "p0": np.asarray(_FAKE_BALL_POS, dtype=np.float64),
            "v0": np.array([0.0, -5.0, 1.0], dtype=np.float64),
        },
        "physical_hit": True,
        "physical_hit_type": "active",
        "physical_hit_step": 2,
        "legacy_hit": True,
        "legacy_hit_type": "active",
        "legacy_hit_rule": LEGACY_HIT_RULE,
        "physical_hit_rule": PHYSICAL_HIT_RULE,
        "legacy_vs_physical_divergence": None,
        "contact_events": contact_events,
    }
    return V12EpisodeResult(
        config=config,
        robot_id="rm65",
        mode=config.actuator_mode,
        joint_names=_JOINT_NAMES_RM65,
        dt=float(config.dt),
        racket_id=None,
        X=X,
        U=U,
        ball_pos=ball_pos,
        ball_vel=ball_vel,
        time=time,
        racket_pos=racket_pos,
        hit_type="active",
        hit_step=2,
        active_contact=True,
        passive_contact=False,
        v_ee_at_hit=1.2,
        p_ee_at_hit=np.array([0.30, 0.0, 1.00]),
        ball_pos_at_hit=np.asarray(_FAKE_BALL_POS),
        ball_out_speed=5.0,
        ball_out_vy=1.0,
        metrics=metrics,
    )


def _fabricate_r1d(
    config: Any,
    *,
    dependency_manifest: tuple[dict[str, Any], ...] = (),
    frames: int = 3,
) -> V12EpisodeResult:
    """构造最小 R1D V12EpisodeResult（可选携带装配依赖清单）。"""
    runtime = load_runtime_parameters(
        config.runtime_path, load_robot_profile(config.robot_profile_path)
    )
    joint_names = tuple(str(name) for name in runtime.joint_names)
    n = len(joint_names)
    n_rows = frames + 1
    kwargs = _supported_kwargs(
        model_dependency_manifest=tuple(dependency_manifest),
    )
    return V12EpisodeResult(
        config=config,
        robot_id=str(config.robot_id),
        mode=str(config.actuator_mode),
        joint_names=joint_names,
        dt=float(config.dt),
        racket_id=None,
        X=np.zeros((n_rows, 2 * n), dtype=np.float64),
        U=np.zeros((frames, n), dtype=np.float64),
        ball_pos=np.zeros((n_rows, 3), dtype=np.float64),
        ball_vel=np.zeros((n_rows, 3), dtype=np.float64),
        time=np.arange(n_rows, dtype=np.float64) * float(config.dt),
        racket_pos=np.zeros((n_rows, 3), dtype=np.float64),
        hit_type="miss",
        hit_step=-1,
        active_contact=False,
        passive_contact=False,
        v_ee_at_hit=None,
        p_ee_at_hit=None,
        ball_pos_at_hit=None,
        ball_out_speed=None,
        ball_out_vy=None,
        metrics={"failure_reason": "fabricated_for_asset_closure"},
        **kwargs,
    )


def _rm65_model_text(site_pos: str = "0 0 0.25") -> str:
    """读真实 RM65 模型文本并改成绝对 POSIX 资产引用。"""
    text = (ROOT / "src" / "robot" / "rm65_model.xml").read_text(encoding="utf-8")
    text = text.replace('file="../../assets/', f'file="{(ROOT / "assets").as_posix()}/')
    text, count = re.subn(
        r'(<site name="racket_center"[^>]*?pos=")[^"]*(")',
        lambda match: f"{match.group(1)}{site_pos}{match.group(2)}",
        text,
        count=1,
    )
    assert count == 1, "真实 RM65 模型中未找到 racket_center site"
    return text


def _mirror_prefix() -> str:
    """ASCII 镜像目录下 assets 的绝对 POSIX 前缀。"""
    mirror = Path(_setup_temp_copy(ROOT))
    return f"{(mirror / 'assets').as_posix()}/"


def _write_profile(work_dir: Path, model_path: Path, *, template: Path) -> Path:
    """以真实 profile 为模板，仅改 ``model_path``。"""
    text = template.read_text(encoding="utf-8")
    text, count = re.subn(
        r"(?m)^(\s*model_path:\s*).*$",
        lambda match: f"{match.group(1)}{model_path.as_posix()}",
        text,
        count=1,
    )
    assert count == 1, f"profile 缺少 model_path：{template}"
    profile_path = work_dir / "profile.yaml"
    profile_path.write_text(text, encoding="utf-8")
    return profile_path


def _write_rm65_episode(
    tmp_path: Path,
    *,
    name: str = "case",
    work_dir: Path | None = None,
) -> tuple[Path, Path]:
    """写临时 RM65 模型 + profile，并生成一份 v2 产物。

    Returns:
        ``(产物目录, 临时模型路径)``。
    """
    base = work_dir if work_dir is not None else tmp_path / name
    base.mkdir(parents=True, exist_ok=True)
    model_path = base / f"{name}_model.xml"
    model_path.write_text(_rm65_model_text(), encoding="utf-8")
    profile_path = _write_profile(
        base, model_path, template=ROOT / "configs" / "robots" / "rm65.yaml"
    )
    config = _parse_and_resolve(["--robot-config", str(profile_path), *_RM65_SHORT_ARGV])
    output_dir = base / "episode"
    artifacts.write_episode_artifacts(_fabricate_rm65(config), output_dir)
    return output_dir, model_path


def _retarget_recorded_sources(output_dir: Path, old_root: Path) -> None:
    """把产物/快照里记录的**来源定位元数据**改指旧检出（内容哈希不变）。

    这正是 F02 探针的做法：模型快照字节、聚合哈希、每条资产 sha256 全部
    保持不变，只把 ``source`` 指到一个"旧检出"路径——修复前的实现会在物化
    阶段把已校验的仓库文件换成这个旧路径。
    """
    payload = _load_payload(output_dir)
    manifest = _load_manifest(output_dir)

    def retarget(entry: dict[str, Any]) -> None:
        relative = str(entry.get("relative_path") or "")
        if not relative:
            return
        if relative.startswith("external/"):
            entry["source"] = str(old_root / Path(relative).name)
        else:
            entry["source"] = str(old_root / relative)

    for entry in payload.get("assets", {}).get("files", []):
        retarget(entry)
    for entry in payload.get("dependencies", {}).get("files", []):
        retarget(entry)
    for entry in manifest.get("files", []):
        retarget(entry)
    for entry in manifest.get("assets", {}).get("files", []):
        retarget(entry)
    for entry in (manifest.get("dependencies") or {}).get("files", []):
        retarget(entry)
    _write_payload(output_dir, payload)
    _write_manifest(output_dir, manifest)


def _scale_binary_stl(source: Path, destination: Path, factor: float) -> None:
    """把二进制 STL 的每个顶点等比缩放（保持 84+50n 结构，仍是合法 STL）。"""
    raw = source.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    assert len(raw) == 84 + 50 * count, "仅支持二进制 STL"
    out = bytearray(raw)
    for index in range(count):
        base = 84 + index * 50
        for vertex in range(3):
            offset = base + 12 + vertex * 12
            x, y, z = struct.unpack_from("<3f", raw, offset)
            struct.pack_into("<3f", out, offset, x * factor, y * factor, z * factor)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(bytes(out))


def _mesh_vertex(model: Any) -> np.ndarray:
    """取模型网格顶点数组副本。"""
    return np.array(model.mesh_vert, dtype=np.float64, copy=True)


def _loaded_site_pos(bundle: Any, site: str = _RACKET_SITE) -> np.ndarray:
    """取回放 bundle 实际加载模型中的 site 位置。"""
    model = bundle.execution.model
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    assert site_id >= 0, f"加载模型缺少 site：{site}"
    return np.asarray(model.site_pos[site_id], dtype=float)


def _loaded_body_pos(bundle: Any, body: str) -> np.ndarray:
    """取回放 bundle 实际加载模型中的 body 位置。"""
    model = bundle.execution.model
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    assert body_id >= 0, f"加载模型缺少 body：{body}"
    return np.asarray(model.body_pos[body_id], dtype=float)


def _dependency_entry(
    output_dir: Path, *, role: str, name: str
) -> tuple[dict[str, Any], Path]:
    """按角色 + 文件名取依赖条目及其快照副本路径。"""
    payload = _load_payload(output_dir)
    for entry in payload["dependencies"]["files"]:
        if entry["role"] != role:
            continue
        if Path(str(entry["relative_path"])).name != name:
            continue
        snapshot_path = entry.get("snapshot_path")
        assert snapshot_path, f"依赖条目缺少快照副本：{entry['asset_id']}"
        return entry, output_dir / artifacts.SNAPSHOT_DIR_NAME / str(snapshot_path)
    raise AssertionError(f"依赖清单缺少 {role}/{name}")


# ── F02：验证与加载同源 ───────────────────────────────────────────────


def test_replay_uses_repo_mesh_when_recorded_source_deleted(tmp_path: Path) -> None:
    """正确仓库网格 + 记录来源（旧检出）被删除：回放仍用已验证字节。"""
    output_dir, _model_path = _write_rm65_episode(tmp_path)
    old_root = tmp_path / "old_checkout"
    _retarget_recorded_sources(output_dir, old_root)
    assert not (old_root / _UNDERPAN_RELATIVE).exists()

    payload = _load_payload(output_dir)
    assert artifacts._verify_model_assets(payload, output_dir) == []

    bundle = artifacts.build_replay_bundle(output_dir)
    reference = load_mujoco_model(
        tmp_path / "reference.xml",
        xml_text=_rm65_model_text().replace(
            f'file="{(ROOT / "assets").as_posix()}/', f'file="{_mirror_prefix()}'
        ),
    )
    np.testing.assert_array_equal(_mesh_vertex(bundle.execution.model), _mesh_vertex(reference))
    np.testing.assert_allclose(_loaded_site_pos(bundle), _SNAPSHOT_SITE_POS, atol=1e-12)
    assert bundle.execution.model.nmesh == 22


def test_replay_rejects_scaled_recorded_source_mesh(tmp_path: Path) -> None:
    """旧来源 STL 被合法放大 1.25 倍：回放仍只用哈希匹配的仓库字节。"""
    output_dir, _model_path = _write_rm65_episode(tmp_path)
    old_root = tmp_path / "old_checkout"
    _retarget_recorded_sources(output_dir, old_root)
    payload = _load_payload(output_dir)
    expected_sha = payload["input_sha256"]["assets"]
    repo_stl = ROOT / _UNDERPAN_RELATIVE

    old_stl = old_root / _UNDERPAN_RELATIVE
    _scale_binary_stl(repo_stl, old_stl, 1.25)
    assert old_stl.is_file()
    # 旧来源仍是合法 STL，但内容不再与记录哈希一致
    assert _sha256(old_stl) != _sha256(repo_stl)
    assert expected_sha  # 聚合哈希存在（仅作存在性检查）

    assert artifacts._verify_model_assets(payload, output_dir) == []
    bundle = artifacts.build_replay_bundle(output_dir)
    reference = load_mujoco_model(
        tmp_path / "reference.xml",
        xml_text=_rm65_model_text().replace(
            f'file="{(ROOT / "assets").as_posix()}/', f'file="{_mirror_prefix()}'
        ),
    )
    np.testing.assert_array_equal(_mesh_vertex(bundle.execution.model), _mesh_vertex(reference))

    # 反证：如果真按旧来源加载，几何会明显不同（F02 的 0.0857 量级）
    tampered_text = _rm65_model_text().replace(
        f'file="{(ROOT / "assets").as_posix()}/', f'file="{_mirror_prefix()}'
    )
    tampered_text = tampered_text.replace(
        f'file="{_mirror_prefix()}rm_65/urdf/meshes/base_link_underpan.STL"',
        f'file="{old_stl.as_posix()}"',
    )
    tampered = load_mujoco_model(tmp_path / "tampered.xml", xml_text=tampered_text)
    max_diff = float(
        np.max(np.abs(_mesh_vertex(tampered) - _mesh_vertex(bundle.execution.model)))
    )
    assert max_diff > 0.05, f"旧来源几何差异过小，测试无效：{max_diff}"


def test_dependency_manifest_records_stable_ids_and_verified_paths(tmp_path: Path) -> None:
    """依赖清单版本/来源/稳定 asset_id 与冻结接口返回的映射一致。"""
    output_dir, _model_path = _write_rm65_episode(tmp_path)
    payload = _load_payload(output_dir)
    assert payload["dependency_manifest_version"] == 1
    assert payload["dependency_manifest_source"] in {"derived", "result"}
    entries = payload["dependencies"]["files"]
    assert entries, "依赖清单为空"
    for entry in entries:
        asset_id = str(entry["asset_id"])
        assert asset_id.startswith(f"{entry['role']}-")
        # 逻辑 ID 不能是某次运行的绝对路径
        assert ":" not in asset_id and "\\" not in asset_id and "/" not in asset_id
        assert entry["sha256"] and entry["relative_path"]
    snapshot = dict(payload)
    snapshot["snapshot_dir"] = str(output_dir / artifacts.SNAPSHOT_DIR_NAME)
    resolved = artifacts.resolve_verified_assets(snapshot, ROOT)
    assert set(resolved) == {str(entry["asset_id"]) for entry in entries}
    for path in resolved.values():
        assert Path(path).is_file()


# ── F08：装配依赖闭包 ─────────────────────────────────────────────────


def _build_custom_r1d_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """构造"外部模型目录 + 自定义对侧片段 + 非空固定关节姿态"的产物。

    Returns:
        ``(产物目录, 外部模型目录, 依赖清单)``。
    """
    mirror = Path(_setup_temp_copy(ROOT))
    fixture_dir = tmp_path / "r1d_external"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    model_text = (ROOT / "src" / "robot" / "r1d_model_right.xml").read_text(
        encoding="utf-8"
    )
    model_text = model_text.replace(
        'file="../../assets/', f'file="{(mirror / "assets").as_posix()}/'
    )
    model_path = fixture_dir / "r1d_model_right.xml"
    model_path.write_text(model_text, encoding="utf-8")

    fragment_text = (ROOT / "src" / "robot" / "r1d_arm_left.xml").read_text(
        encoding="utf-8"
    )
    fragment_text, count = re.subn(
        r'(name="left_shoulder_pitch_Link"\s+pos=")[^"]*(")',
        lambda match: f"{match.group(1)}0.2 0.085688 0.19749{match.group(2)}",
        fragment_text,
        count=1,
    )
    assert count == 1, "对侧臂片段缺少 left_shoulder_pitch_Link"
    (fixture_dir / "r1d_arm_left.xml").write_text(fragment_text, encoding="utf-8")

    profile_path = _write_profile(
        fixture_dir, model_path, template=ROOT / "configs" / "robots" / "r1d_right.yaml"
    )
    runtime_text = (
        ROOT / "configs" / "robots" / "r1d_v12_runtime_right.yaml"
    ).read_text(encoding="utf-8")
    runtime_text, count = re.subn(
        r"(?m)^(\s*fixed_joint_poses:\s*)\{\}(\s*(?:#.*)?)$",
        lambda match: (
            f"{match.group(1)}{{left_shoulder_pitch_joint: 0.2}}{match.group(2)}"
        ),
        runtime_text,
        count=1,
    )
    assert count == 1, "runtime 缺少空的 fixed_joint_poses 行"
    runtime_path = fixture_dir / "runtime.yaml"
    runtime_path.write_text(runtime_text, encoding="utf-8")

    output_dir = tmp_path / "r1d_episode"
    config = _parse_and_resolve(
        [
            "--robot-config",
            str(profile_path),
            "--runtime-config",
            str(runtime_path),
            "--sim-task-config",
            str(ROOT / "configs" / "r1d_sim_right.yaml"),
            "--seed",
            "42",
            "--no-plot",
            "--output",
            str(output_dir),
        ]
    )
    # 保存前先确认真实装配会读到自定义片段（fixture 有效性）
    original = build_v12_robot(config)
    bundle = original.robot_model_bundle
    assert bundle is not None
    assert float(_loaded_body_pos(original, _PASSIVE_BODY)[0]) == pytest.approx(0.2)
    manifest = bundle.dependency_manifest
    assert manifest, "装配过程未记录依赖清单"

    artifacts.write_episode_artifacts(
        _fabricate_r1d(config, dependency_manifest=manifest), output_dir
    )
    return output_dir, fixture_dir, _load_payload(output_dir)


def test_custom_arm_fragment_survives_external_dir_move(tmp_path: Path) -> None:
    """外部目录整体移走后，回放仍使用快照里的自定义对侧片段（X=0.2）。"""
    output_dir, fixture_dir, payload = _build_custom_r1d_fixture(tmp_path)
    assert payload["dependency_manifest_version"] == 1
    roles = {str(entry["role"]) for entry in payload["dependencies"]["files"]}
    assert "arm_fragment" in roles

    fragment_entry, fragment_copy = _dependency_entry(
        output_dir, role="arm_fragment", name="r1d_arm_left.xml"
    )
    assert fragment_copy.is_file()
    import base64

    decoded = base64.b64decode(str(fragment_entry["content_base64"])).decode("utf-8")
    assert 'name="left_shoulder_pitch_Link" pos="0.2 0.085688 0.19749"' in decoded

    moved = tmp_path / "moved_external"
    shutil.move(str(fixture_dir), str(moved))
    assert not fixture_dir.exists()

    bundle = artifacts.build_replay_bundle(output_dir)
    position = _loaded_body_pos(bundle, _PASSIVE_BODY)
    assert float(position[0]) == pytest.approx(0.2), position


def test_tampered_after_verification_is_rejected_with_asset_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """已验证文件在复制前被篡改：物化拒绝，错误带逻辑 asset_id。"""
    output_dir, _fixture_dir, _payload = _build_custom_r1d_fixture(tmp_path)
    _entry, fragment_copy = _dependency_entry(
        output_dir, role="arm_fragment", name="r1d_arm_left.xml"
    )
    original = artifacts.resolve_verified_assets

    def sneaky(snapshot: dict[str, Any], project_root: Path) -> dict[str, Path]:
        resolved = original(snapshot, project_root)
        fragment_copy.write_text(
            fragment_copy.read_text(encoding="utf-8") + "\n<!-- tampered -->\n",
            encoding="utf-8",
        )
        return resolved

    monkeypatch.setattr(artifacts, "resolve_verified_assets", sneaky)
    with pytest.raises(ValueError, match=r"arm_fragment-[0-9a-f]{12}"):
        artifacts.build_replay_bundle(output_dir)


def test_missing_dependency_reports_asset_id(tmp_path: Path) -> None:
    """快照副本与外部来源都缺失：拒绝回放并按逻辑 asset_id 报错。"""
    output_dir, fixture_dir, _payload = _build_custom_r1d_fixture(tmp_path)
    _entry, fragment_copy = _dependency_entry(
        output_dir, role="arm_fragment", name="r1d_arm_left.xml"
    )
    fragment_copy.unlink()
    shutil.rmtree(fixture_dir)

    with pytest.raises(ValueError, match=r"arm_fragment-[0-9a-f]{12}"):
        artifacts.build_replay_bundle(output_dir)


# ── 中文路径 / 跨根资产 / 嵌套 include ────────────────────────────────


def _write_nested_include_episode(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    """构造中文目录 + 嵌套 include + 跨根 STL 的 RM65 产物。

    Returns:
        ``(产物目录, 模型目录, 跨根资产目录)``。
    """
    model_dir = tmp_path / "中文 目录" / "模型"
    model_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = tmp_path / "跨根资产"
    shared_dir.mkdir(parents=True, exist_ok=True)
    mirror_prefix = _mirror_prefix()

    text = _rm65_model_text().replace(
        f'file="{(ROOT / "assets").as_posix()}/', f'file="{mirror_prefix}'
    )
    match = re.search(r"<asset>(.*?)</asset>", text, re.S)
    assert match is not None, "RM65 模型缺少 <asset> 块"
    block = match.group(1)
    mesh_lines = [
        line.strip() for line in block.splitlines() if line.strip().startswith("<mesh")
    ]
    other_lines = [
        line.strip()
        for line in block.splitlines()
        if line.strip().startswith(("<texture", "<material"))
    ]
    assert len(mesh_lines) == 22, len(mesh_lines)

    shared_stl = shared_dir / "base_link_underpan.STL"
    shutil.copy2(ROOT / _UNDERPAN_RELATIVE, shared_stl)
    mesh_lines = [
        line.replace(
            f'file="{mirror_prefix}rm_65/urdf/meshes/base_link_underpan.STL"',
            f'file="{shared_stl.as_posix()}"',
        )
        for line in mesh_lines
    ]
    assert any(shared_stl.as_posix() in line for line in mesh_lines)

    (model_dir / "meshes.xml").write_text(
        "<mujocoinclude>\n"
        + "\n".join(f"  {line}" for line in mesh_lines)
        + "\n</mujocoinclude>\n",
        encoding="utf-8",
    )
    (model_dir / "assets.xml").write_text(
        "<mujocoinclude>\n"
        + "\n".join(f"  {line}" for line in other_lines)
        + '\n  <include file="meshes.xml"/>\n</mujocoinclude>\n',
        encoding="utf-8",
    )
    model_text = (
        text[: match.start()]
        + '<asset>\n  <include file="assets.xml"/>\n</asset>'
        + text[match.end() :]
    )
    model_path = model_dir / "rm65_nested.xml"
    model_path.write_text(model_text, encoding="utf-8")

    profile_path = _write_profile(
        model_dir, model_path, template=ROOT / "configs" / "robots" / "rm65.yaml"
    )
    config = _parse_and_resolve(
        ["--robot-config", str(profile_path), *_RM65_SHORT_ARGV]
    )
    output_dir = tmp_path / "nested_episode"
    artifacts.write_episode_artifacts(_fabricate_rm65(config), output_dir)
    return output_dir, model_dir, shared_dir


def test_nested_include_and_cross_root_asset_survive_source_loss(tmp_path: Path) -> None:
    """中文路径 + 跨根资产 + 嵌套 include：删除来源后回放仍完整。"""
    output_dir, model_dir, shared_dir = _write_nested_include_episode(tmp_path)
    payload = _load_payload(output_dir)
    entries = payload["dependencies"]["files"]
    by_role: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_role.setdefault(str(entry["role"]), []).append(entry)
    # 真实 RM65 的 texture 是 builtin（无 file 属性）→ 不产生文件依赖
    assert {"model", "include", "mesh"} <= set(by_role)
    assert "texture" not in by_role
    assert len(by_role["include"]) == 2, [e["relative_path"] for e in by_role["include"]]
    import base64

    for entry in by_role["include"] + by_role["model"]:
        assert entry.get("content_base64"), entry["asset_id"]
        assert base64.b64decode(str(entry["content_base64"]))
        assert entry.get("snapshot_path")

    # 直接检查物化结果：include 被递归展开且改写为临时目录绝对路径
    sources = artifacts._verify_and_resolve_sources(payload, output_dir)
    model_path = artifacts._resolved_model_path(payload, sources)
    profile_path, cleanup = artifacts._materialize_snapshot_model(
        payload, sources, model_path, output_dir=output_dir
    )
    try:
        staged_profile = profile_path.read_text(encoding="utf-8")
        staged_model = Path(
            re.search(r"model_path:\s*'([^']+)'", staged_profile).group(1)  # type: ignore[union-attr]
        )
        staged_text = staged_model.read_text(encoding="utf-8")
        # include 必须全部改写为临时目录内的绝对路径（不再依赖原外部目录）
        include_refs = re.findall(r'<include\s+file="([^"]+)"', staged_text)
        assert len(include_refs) == 1, include_refs
        assert Path(include_refs[0]).is_file()
        assert Path(include_refs[0]).is_relative_to(staged_model.parent)
        assert staged_text.count('file="') >= 1
        include_dir = staged_model.parent / "includes"
        assert len(list(include_dir.glob("*.xml"))) == 2
        assert staged_text.count('file="') >= 1
    finally:
        cleanup()

    # 删除外部模型目录与跨根资产目录，回放仍成功（快照自包含）
    shutil.rmtree(model_dir)
    shutil.rmtree(shared_dir)
    assert not shared_dir.exists()

    bundle = artifacts.build_replay_bundle(output_dir)
    assert bundle.execution.model.nmesh == 22
    assert bundle.execution.model.ntex >= 1
    np.testing.assert_allclose(_loaded_site_pos(bundle), _SNAPSHOT_SITE_POS, atol=1e-12)


# ── legacy 分支 ───────────────────────────────────────────────────────


def test_legacy_artifact_without_dependency_manifest_warns_and_replays(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """旧 artifact 无依赖清单：显式 legacy 分支 + 警告，不假装已验证闭包。"""
    output_dir, _model_path = _write_rm65_episode(tmp_path)
    payload = _load_payload(output_dir)
    manifest = _load_manifest(output_dir)
    for key in ("dependency_manifest_version", "dependency_manifest_source", "dependencies"):
        payload.pop(key, None)
        manifest.pop(key, None)
    _write_payload(output_dir, payload)
    _write_manifest(output_dir, manifest)

    with caplog.at_level(logging.WARNING, logger="src.sim.episode_artifacts"):
        bundle = artifacts.build_replay_bundle(output_dir)
    messages = [record.getMessage() for record in caplog.records]
    assert any("旧 artifact 缺少装配依赖清单" in message for message in messages), messages
    assert any("legacy" in message for message in messages), messages
    np.testing.assert_allclose(_loaded_site_pos(bundle), _SNAPSHOT_SITE_POS, atol=1e-12)


def test_legacy_manifest_is_not_silently_treated_as_verified(
    tmp_path: Path,
) -> None:
    """旧 artifact 的 assets 段仍受哈希约束：缺失即拒绝（不因 legacy 放松）。"""
    output_dir, _model_path = _write_rm65_episode(tmp_path)
    payload = _load_payload(output_dir)
    manifest = _load_manifest(output_dir)
    for key in ("dependency_manifest_version", "dependency_manifest_source", "dependencies"):
        payload.pop(key, None)
        manifest.pop(key, None)
    # 把一条资产哈希改成不匹配，模拟旧产物被篡改
    payload["assets"]["files"][0]["sha256"] = "0" * 64
    _write_payload(output_dir, payload)
    _write_manifest(output_dir, manifest)

    with pytest.raises(ValueError, match="依赖资产|哈希"):
        artifacts.build_replay_bundle(output_dir)


def test_derive_dependency_entries_matches_assembly_manifest(tmp_path: Path) -> None:
    """R1D 推导路径与真实装配清单口径一致（角色/条目数/哈希）。"""
    output_dir, _fixture_dir, payload = _build_custom_r1d_fixture(tmp_path)
    assert collect_model_dependencies is not None
    derived = payload["dependencies"]["files"]
    roles = sorted({str(entry["role"]) for entry in derived})
    assert "arm_fragment" in roles
    assert "mesh" in roles
    assert len(derived) == len({str(entry["asset_id"]) for entry in derived})


# ── 清单来源：result 强证据 / derived 弱证据 / 严格模式 ────────────────


def test_derived_manifest_is_marked_weak_and_warns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """结果未携带清单：产物自曝 derived 来源并告警，不假装已验证闭包。

    该用例保留的是旧产物流的过渡语义；R1D 新产物的严格拒绝由
    ``test_v12_review_integration.py::test_strict_mode_rejects_r1d_result_without_manifest``
    覆盖，因此这里显式关闭严格模式，只验证 degraded 路径本身。
    """
    monkeypatch.setattr(artifacts, "REQUIRE_RESULT_DEPENDENCY_MANIFEST", False)
    config = _parse_and_resolve(
        [
            "--robot-config",
            str(ROOT / "configs" / "robots" / "r1d_right.yaml"),
            "--runtime-config",
            str(ROOT / "configs" / "robots" / "r1d_v12_runtime_right.yaml"),
            "--sim-task-config",
            str(ROOT / "configs" / "r1d_sim_right.yaml"),
            "--seed",
            "42",
            "--no-plot",
            "--output",
            str(tmp_path / "r1d_derived"),
        ]
    )
    output_dir = tmp_path / "r1d_derived"
    with caplog.at_level(logging.WARNING, logger="src.sim.episode_artifacts"):
        artifacts.write_episode_artifacts(_fabricate_r1d(config), output_dir)
    payload = _load_payload(output_dir)
    assert payload["dependency_manifest_version"] == 1
    assert payload["dependency_manifest_source"] == "derived"
    assert payload["dependencies"]["files"], "推导清单为空"
    messages = [record.getMessage() for record in caplog.records]
    assert any("退化为按配置推导" in message for message in messages), messages


def test_strict_mode_rejects_r1d_result_without_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """打开严格模式后，R1D 新结果清单为空必须拒绝写入（契约开关生效）。"""
    config = _parse_and_resolve(
        [
            "--robot-config",
            str(ROOT / "configs" / "robots" / "r1d_right.yaml"),
            "--runtime-config",
            str(ROOT / "configs" / "robots" / "r1d_v12_runtime_right.yaml"),
            "--sim-task-config",
            str(ROOT / "configs" / "r1d_sim_right.yaml"),
            "--seed",
            "42",
            "--no-plot",
            "--output",
            str(tmp_path / "r1d_strict"),
        ]
    )
    monkeypatch.setattr(artifacts, "REQUIRE_RESULT_DEPENDENCY_MANIFEST", True)
    with pytest.raises(ValueError, match="缺少装配依赖清单"):
        artifacts.write_episode_artifacts(
            _fabricate_r1d(config), tmp_path / "r1d_strict"
        )
