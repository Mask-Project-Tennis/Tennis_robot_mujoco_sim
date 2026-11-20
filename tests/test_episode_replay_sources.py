"""R11/R12 回放来源测试：快照模型优先与缺失候选回退。

独立审查 R11 指出：模型快照哈希校验通过后，回放仍从原 profile 解析外部
模型，外部模型被改动时快照完好的产物被拒绝；R12 指出：某个候选快照
缺失/不匹配时立即追加全局问题，后续匹配的外部来源无法清除，且整个
snapshot 缺失时会误报“manifest 依赖资产条目数 0 与 payload 不一致”。

本文件用真实 RM65 模型（mesh 引用改为绝对 POSIX 路径）构造 V12 产物，验证：

- 外部模型被改/删除时回放使用快照模型（``racket_center`` site 来自快照）；
- 快照模型被篡改时仍然拒绝（即使外部模型匹配）；
- 仅缺快照 model.xml 或整个 snapshot 时按外部匹配来源回放；
- 物化 XML 的 mesh/texture 路径全部可解析，临时目录在 build 后仍存活。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

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

# 与 test_episode_artifacts.py 一致的短回合 CLI 参数
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
# 真实 rm65_model.xml 中 racket_center site 的 z 坐标
_SNAPSHOT_SITE_POS = np.array([0.0, 0.0, 0.25])
# 篡改外部模型时使用的 site 位置（几何明显不同）
_TAMPERED_SITE_POS = "0 0 0.9"


# ── 构造辅助（复用 tests/test_episode_artifacts.py 的轻量方式）────────


def _parse_and_resolve(argv: list[str]):
    """用真实 CLI 解析器 + 解析层构造 ResolvedV12Config。"""
    parser = v12._build_parser()
    args = parser.parse_args(argv)
    explicit = v12._explicit_cli_actions(parser, argv)
    return v12.resolve_v12_config(args, explicit)


def _fabricated_result(config: object, *, frames: int = 4) -> V12EpisodeResult:
    """构造最小 V12EpisodeResult（合成产物用）。"""
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


def _sha256(path: Path) -> str:
    """文件 SHA256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rm65_model_text(site_pos: str = "0 0 0.25") -> str:
    """读真实 RM65 模型文本并改成绝对 POSIX 资产引用。

    Args:
        site_pos: 写入 ``racket_center`` site 的 pos 属性（MuJoCo 单位）。

    Returns:
        可直接写入临时文件并编译的模型 XML 文本。
    """
    text = (ROOT / "src" / "robot" / "rm65_model.xml").read_text(encoding="utf-8")
    # 相对资产路径 → 仓库 assets 的绝对 POSIX 路径（MuJoCo 接受正斜杠）
    assets_prefix = f"file=\"{ (ROOT / 'assets').as_posix() }/"
    text = text.replace('file="../../assets/', assets_prefix)
    text, count = re.subn(
        r'(<site name="racket_center"[^>]*?pos=")[^"]*(")',
        lambda match: f"{match.group(1)}{site_pos}{match.group(2)}",
        text,
        count=1,
    )
    assert count == 1, "真实 RM65 模型中未找到 racket_center site"
    return text


def _write_rm65_episode(
    tmp_path: Path,
    *,
    site_pos: str = "0 0 0.25",
    name: str = "case",
) -> tuple[Path, Path, Path]:
    """写临时 RM65 模型与 profile，并生成一份 V12 产物。

    Args:
        tmp_path: pytest 临时目录。
        site_pos: ``racket_center`` site 位置。
        name: 子目录/文件名前缀。

    Returns:
        ``(产物目录, 临时模型路径, 临时 profile 路径)``。
    """
    work_dir = tmp_path / name
    work_dir.mkdir(parents=True, exist_ok=True)
    model_path = work_dir / f"{name}_model.xml"
    model_path.write_text(_rm65_model_text(site_pos), encoding="utf-8")
    template = (ROOT / "configs" / "robots" / "rm65.yaml").read_text(encoding="utf-8")
    profile_text = re.sub(
        r"(?m)^\s*model_path:.*$",
        f"model_path: {model_path.as_posix()}",
        template,
        count=1,
    )
    profile_path = work_dir / f"{name}_profile.yaml"
    profile_path.write_text(profile_text, encoding="utf-8")
    config = _parse_and_resolve(["--robot-config", str(profile_path), *_RM65_SHORT_ARGV])
    output_dir = work_dir / "episode"
    artifacts.write_episode_artifacts(_fabricated_result(config), output_dir)
    return output_dir, model_path, profile_path


def _snapshot_role_path(output_dir: Path, role: str) -> Path:
    """按 manifest 取某个 role 的快照文件路径。"""
    manifest_path = (
        output_dir / artifacts.SNAPSHOT_DIR_NAME / artifacts.SNAPSHOT_MANIFEST_NAME
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["role"] == role:
            return output_dir / artifacts.SNAPSHOT_DIR_NAME / entry["relative_path"]
    raise AssertionError(f"快照 manifest 缺少角色：{role}")


def _load_payload(output_dir: Path) -> dict:
    """读产物 JSON。"""
    path = output_dir / artifacts.ARTIFACT_JSON_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def _write_payload(output_dir: Path, payload: dict) -> None:
    """写回产物 JSON。"""
    path = output_dir / artifacts.ARTIFACT_JSON_NAME
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _loaded_site_pos(bundle: object, site: str = _RACKET_SITE) -> np.ndarray:
    """取回放 bundle 实际加载模型中的 site 位置。"""
    model = bundle.execution.model
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    assert site_id >= 0, f"加载模型缺少 site：{site}"
    return np.asarray(model.site_pos[site_id], dtype=float)


# ── R11：快照模型优先 ────────────────────────────────────────────────


def test_replay_uses_snapshot_model_when_external_model_changed(tmp_path: Path) -> None:
    """外部模型被改而快照完好：回放成功且几何来自快照。"""
    output_dir, model_path, _ = _write_rm65_episode(tmp_path)
    snapshot_model = _snapshot_role_path(output_dir, "model")
    payload = _load_payload(output_dir)
    assert _sha256(snapshot_model) == payload["input_sha256"]["model"]

    model_path.write_text(_rm65_model_text(_TAMPERED_SITE_POS), encoding="utf-8")
    assert _sha256(model_path) != payload["input_sha256"]["model"]

    bundle = artifacts.build_replay_bundle(output_dir)
    np.testing.assert_allclose(
        _loaded_site_pos(bundle), _SNAPSHOT_SITE_POS, atol=1e-12
    )
    assert bundle.execution.model.nmesh == 22


def test_replay_uses_snapshot_model_when_external_model_deleted(tmp_path: Path) -> None:
    """外部模型被删而快照完好：回放仍使用快照模型。"""
    output_dir, model_path, _ = _write_rm65_episode(tmp_path)
    model_path.unlink()

    bundle = artifacts.build_replay_bundle(output_dir)
    np.testing.assert_allclose(
        _loaded_site_pos(bundle), _SNAPSHOT_SITE_POS, atol=1e-12
    )


def test_replay_materializes_when_external_profile_missing(tmp_path: Path) -> None:
    """外部 profile 缺失（选中快照 profile）时仍无条件物化快照模型。"""
    output_dir, _, profile_path = _write_rm65_episode(tmp_path)
    profile_path.unlink()

    bundle = artifacts.build_replay_bundle(output_dir)
    np.testing.assert_allclose(
        _loaded_site_pos(bundle), _SNAPSHOT_SITE_POS, atol=1e-12
    )


def test_replay_materializes_when_snapshot_profile_points_at_snapshot_model(
    tmp_path: Path,
) -> None:
    """快照 profile 的相对 model_path 解析回快照模型本身时仍需物化。

    这是 Lead 指出的边界：默认仓库 profile 快照为
    ``snapshot/files/configs/robots/rm65.yaml``，其 ``model_path:
    ../../src/robot/rm65_model.xml`` 恰好解析到
    ``snapshot/files/src/robot/rm65_model.xml``（即模型快照），但快照目录
    没有 mesh 资产；若按“解析结果相同”跳过物化，构建会因资产缺失失败。
    """
    work_dir = tmp_path / "repo_relative"
    work_dir.mkdir(parents=True, exist_ok=True)
    # 用默认（仓库相对）profile 写产物，使快照 profile 保留相对 model_path
    config = _parse_and_resolve([*_RM65_SHORT_ARGV])
    output_dir = work_dir / "episode"
    artifacts.write_episode_artifacts(_fabricated_result(config), output_dir)

    # 把 payload 的外部 profile 指向不存在的路径 → robot_profile 回退到快照
    payload = _load_payload(output_dir)
    payload["config"]["robot_profile_path"] = str(work_dir / "missing_profile.yaml")
    _write_payload(output_dir, payload)

    bundle = artifacts.build_replay_bundle(output_dir)
    np.testing.assert_allclose(
        _loaded_site_pos(bundle), _SNAPSHOT_SITE_POS, atol=1e-12
    )


def test_replay_rejects_tampered_snapshot_model_even_when_external_matches(
    tmp_path: Path,
) -> None:
    """快照模型哈希不符：即使外部模型匹配也拒绝回放。"""
    output_dir, _, _ = _write_rm65_episode(tmp_path)
    snapshot_model = _snapshot_role_path(output_dir, "model")
    snapshot_model.write_text(_rm65_model_text(_TAMPERED_SITE_POS), encoding="utf-8")

    with pytest.raises(ValueError, match="快照|哈希"):
        artifacts.build_replay_bundle(output_dir)


def test_replay_falls_back_to_external_model_when_snapshot_model_missing(
    tmp_path: Path,
) -> None:
    """仅缺快照 model.xml：外部模型哈希匹配时允许回放。"""
    output_dir, model_path, _ = _write_rm65_episode(tmp_path)
    _snapshot_role_path(output_dir, "model").unlink()

    payload = _load_payload(output_dir)
    sources = artifacts._verify_and_resolve_sources(payload, output_dir)
    resolved = artifacts._resolved_model_path(payload, sources)
    assert resolved is not None
    assert Path(resolved).resolve() == model_path.resolve()

    bundle = artifacts.build_replay_bundle(output_dir)
    np.testing.assert_allclose(
        _loaded_site_pos(bundle), _SNAPSHOT_SITE_POS, atol=1e-12
    )


# ── R12：快照整体缺失时按外部来源验证 ────────────────────────────────


def test_replay_without_snapshot_uses_matching_external_sources(
    tmp_path: Path,
) -> None:
    """整个 snapshot 缺失：按 payload 资产摘要验证外部文件并允许回放。"""
    output_dir, _, _ = _write_rm65_episode(tmp_path)
    shutil.rmtree(output_dir / artifacts.SNAPSHOT_DIR_NAME)

    payload = _load_payload(output_dir)
    # R12：manifest 缺失时不得再报“依赖资产条目数 0 与 payload 不一致”
    assert artifacts._verify_model_assets(payload, output_dir) == []
    sources = artifacts._verify_and_resolve_sources(payload, output_dir)
    assert "model" in sources
    assert "robot_profile" in sources

    bundle = artifacts.build_replay_bundle(output_dir)
    np.testing.assert_allclose(
        _loaded_site_pos(bundle), _SNAPSHOT_SITE_POS, atol=1e-12
    )


def test_replay_rejects_tampered_external_model_without_snapshot(
    tmp_path: Path,
) -> None:
    """无快照且外部模型被改：仍按哈希拒绝回放。"""
    output_dir, model_path, _ = _write_rm65_episode(tmp_path)
    shutil.rmtree(output_dir / artifacts.SNAPSHOT_DIR_NAME)
    model_path.write_text(_rm65_model_text(_TAMPERED_SITE_POS), encoding="utf-8")

    with pytest.raises(ValueError, match="哈希|不一致"):
        artifacts.build_replay_bundle(output_dir)


def test_replay_falls_back_to_external_profile_when_snapshot_role_missing(
    tmp_path: Path,
) -> None:
    """仅缺快照 profile：外部 profile 匹配时清除该 role 失败并回放。"""
    output_dir, _, _ = _write_rm65_episode(tmp_path)
    _snapshot_role_path(output_dir, "robot_profile").unlink()

    bundle = artifacts.build_replay_bundle(output_dir)
    np.testing.assert_allclose(
        _loaded_site_pos(bundle), _SNAPSHOT_SITE_POS, atol=1e-12
    )


# ── 物化正确性 ───────────────────────────────────────────────────────


def test_materialized_snapshot_xml_assets_exist_and_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """物化 XML 的 mesh/texture 引用全部为可解析的绝对 POSIX 路径。"""
    output_dir, _, _ = _write_rm65_episode(tmp_path)
    created: list[Path] = []
    real_mkdtemp = artifacts.tempfile.mkdtemp

    def _recording_mkdtemp(*args: object, **kwargs: object) -> str:
        path = Path(real_mkdtemp(*args, **kwargs))
        created.append(path)
        return str(path)

    monkeypatch.setattr(artifacts.tempfile, "mkdtemp", _recording_mkdtemp)
    bundle = artifacts.build_replay_bundle(output_dir)

    staging_dirs = [
        path
        for path in created
        if path.name.startswith(artifacts._MATERIALIZED_MODEL_PREFIX)
    ]
    assert len(staging_dirs) == 1, created
    staging_dir = staging_dirs[0]
    xml_files = list(staging_dir.glob("*.xml"))
    assert len(xml_files) == 1, list(staging_dir.iterdir())
    text = xml_files[0].read_text(encoding="utf-8")
    refs = re.findall(
        r"<(?:mesh|texture)\b[^>]*?\bfile=\"([^\"]+)\"", text, re.IGNORECASE
    )
    assert len(refs) == 22, refs
    for ref in refs:
        assert re.match(r"^(?:[A-Za-z]:/|/)", ref), f"非绝对 POSIX 路径：{ref}"
        assert "\\" not in ref, f"路径含反斜杠：{ref}"
        assert Path(ref).is_file(), f"资产不可解析：{ref}"
    assert bundle.execution.model.nmesh == 22
    # 规划环境延迟加载 profile.model_path，临时目录必须在 build 后仍存活
    assert staging_dir.is_dir()
    np.testing.assert_allclose(
        _loaded_site_pos(bundle), _SNAPSHOT_SITE_POS, atol=1e-12
    )


def test_materialize_snapshot_model_cleanup_removes_temp_dir(tmp_path: Path) -> None:
    """物化返回的 cleanup 必须删除临时目录（异常/生命周期收尾共用）。"""
    output_dir, _, _ = _write_rm65_episode(tmp_path)
    payload = _load_payload(output_dir)
    sources = artifacts._verify_and_resolve_sources(payload, output_dir)
    model_path = artifacts._resolved_model_path(payload, sources)
    assert model_path is not None

    profile_path, cleanup = artifacts._materialize_snapshot_model(
        payload, sources, Path(model_path)
    )
    match = re.search(r"(?m)^\s*model_path:\s*'?([^'\n]+)'?\s*$", profile_path.read_text(encoding="utf-8"))
    assert match is not None
    xml_path = Path(match.group(1))
    staging_dir = xml_path.parent
    try:
        assert staging_dir.is_dir()
        assert xml_path.is_file()
        assert "<include" not in xml_path.read_text(encoding="utf-8").lower()
    finally:
        cleanup()
    assert not staging_dir.exists()


def test_replay_rejects_snapshot_model_containing_include(tmp_path: Path) -> None:
    """含 ``<include>`` 的快照模型必须明确拒绝，不得静默解析错误。"""
    output_dir, _, _ = _write_rm65_episode(tmp_path)
    snapshot_model = _snapshot_role_path(output_dir, "model")
    include_text = (
        '<mujoco model="with_include">\n'
        '  <include file="other.xml"/>\n'
        "</mujoco>\n"
    )
    snapshot_model.write_text(include_text, encoding="utf-8")
    payload = _load_payload(output_dir)
    payload["input_sha256"]["model"] = _sha256(snapshot_model)
    _write_payload(output_dir, payload)

    with pytest.raises(ValueError, match="include"):
        artifacts.build_replay_bundle(output_dir)
