"""D1 拆分导入契约与行为回归测试。

覆盖 brief §测试要求：

* 旧 facade 与新子包的对象身份（``is``）、``__module__``、签名不变；
* 配置 JSON 往返与 ``SNAPSHOT_CONFIG_NAME`` 常量；
* 从非仓库 cwd 写入产物到 tmp 目录并读回（不跑完整仿真）；
* ``load_episode_artifacts`` 缺文件/坏 JSON/坏 schema 的错误类型与文案；
* ``build_replay_bundle`` / ``replay_episode_artifacts`` 的 ``viewer=False``；
* 源码结构：facade < 250 行且无函数/类定义，子包不反向 import 旧 facade
  或 ``episode.execution``，``V12EpisodeResult`` 只从 ``episode.types`` 导入。

测试只使用合成的最小 ``V12EpisodeResult``，不启动仿真回合。
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.sim.artifacts as artifacts_pkg
import src.sim.artifacts.dependencies as artifacts_dependencies
import src.sim.artifacts.materialization as artifacts_materialization
import src.sim.artifacts.serialization as artifacts_serialization
import src.sim.artifacts.validation as artifacts_validation
from src.sim import episode_artifacts as legacy
from src.sim.configuration.resolution import resolve_v12_config
from src.sim.episode.types import V12EpisodeResult

_JOINT_NAMES_RM65 = (
    "r_joint1",
    "r_joint2",
    "r_joint3",
    "r_joint4",
    "r_joint5",
    "r_joint6",
)
_FAKE_BALL_POS = np.array([0.30, 0.0, 1.00])

# 既有测试直接访问的旧私有名 → 新定义模块（facade 必须显式重导出）。
_PRIVATE_EXPORTS: tuple[tuple[Any, tuple[str, ...]], ...] = (
    (
        artifacts_serialization,
        ("_sha256_bytes", "_sha256_file", "_assets_digest", "_hit_section"),
    ),
    (
        artifacts_dependencies,
        ("_model_asset_entries", "_normalize_dependency_entries", "_DependencyStager"),
    ),
    (artifacts_validation, ("_verify_model_assets", "_verify_and_resolve_sources")),
    (
        artifacts_materialization,
        ("_resolved_model_path", "_materialize_snapshot_model", "_MATERIALIZED_MODEL_PREFIX"),
    ),
)

# p0-signatures.txt 冻结的公开签名（拆分前逐字记录）。
_EXPECTED_SIGNATURES = {
    "EpisodeArtifacts": (
        "(payload: 'dict[str, Any]', arrays: 'dict[str, np.ndarray]', "
        "directory: 'Path') -> None"
    ),
    "EpisodeReplayResult": (
        "(payload: 'dict[str, Any]', ball_positions: 'np.ndarray', "
        "racket_positions: 'np.ndarray', speed: 'float', viewer_opened: 'bool', "
        "directory: 'Path') -> None"
    ),
    "write_episode_artifacts": (
        "(result: 'V12EpisodeResult', output_dir: 'Path') -> 'dict[str, Any]'"
    ),
    "load_episode_artifacts": "(output_dir: 'Path') -> 'EpisodeArtifacts'",
    "build_replay_bundle": "(output_dir: 'Path') -> 'V12RobotBundle'",
    "replay_episode_artifacts": (
        "(output_dir: 'Path', *, speed: 'float' = 1.0, viewer: 'bool' = True, "
        "bundle: 'V12RobotBundle | None' = None) -> 'EpisodeReplayResult'"
    ),
}


def _fabricated_result(config: Any, *, frames: int = 4) -> V12EpisodeResult:
    """构造最小 ``V12EpisodeResult``（与既有产物测试同构，不跑仿真）。"""
    n = len(_JOINT_NAMES_RM65)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(frames + 1, 2 * n)) * 0.1
    U = rng.normal(size=(frames, n)) * 0.01
    ball_pos = np.tile(_FAKE_BALL_POS, (frames + 1, 1))
    ball_vel = np.tile(np.array([0.0, -5.0, 1.0]), (frames + 1, 1))
    time = np.arange(frames + 1, dtype=np.float64) * float(config.dt)
    racket_pos = np.tile(np.array([0.30, 0.0, 1.00]), (frames + 1, 1))
    metrics = {
        "total_steps": frames,
        "safe_steps": frames,
        "mpc_done": True,
        "first_plan_ms": 1.0,
        "wall_time_s": 2.0,
        "phase_timings": {"far": 0.1, "mid": 0.2, "near": 0.3},
        "backend": {"requested": "cpp", "effective": "python-analytic"},
        "failure_reason": None,
        "pos_error": 0.05,
        "min_dist": 0.02,
        "proximity": {
            "ball_radius": 0.033,
            "racket_radius": 0.12,
            "proximity_margin": 0.03,
            "ball_near_threshold": 0.183,
        },
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


@pytest.fixture(scope="module")
def resolved_config() -> Any:
    """默认 RM65 配置（读取 configs/*.yaml；路径字段为 None 走 profile 回退）。"""
    return resolve_v12_config()


@pytest.fixture(scope="module")
def written_artifacts(tmp_path_factory: pytest.TempPathFactory, resolved_config: Any):
    """写一次产物，供 load/replay 测试复用（避免重复复制模型资产）。"""
    out = tmp_path_factory.mktemp("d1_written_artifacts")
    result = _fabricated_result(resolved_config)
    payload = legacy.write_episode_artifacts(result, out)
    return out, result, payload


def test_facade_reexports_same_objects() -> None:
    """旧 facade 与子包必须共享同一批对象。"""
    for name in (
        "EpisodeArtifacts",
        "EpisodeReplayResult",
        "write_episode_artifacts",
        "load_episode_artifacts",
        "build_replay_bundle",
        "replay_episode_artifacts",
        "config_to_dict",
        "config_from_dict",
        "SNAPSHOT_CONFIG_NAME",
        "ARTIFACT_NPZ_NAME",
        "EPISODE_SCHEMA",
    ):
        assert getattr(legacy, name) is getattr(artifacts_pkg, name), name

    for module, names in _PRIVATE_EXPORTS:
        for name in names:
            assert getattr(legacy, name) is getattr(module, name), name

    # 定义归属落在新模块，而不是 facade。
    assert legacy.EpisodeArtifacts.__module__ == "src.sim.artifacts.types"
    assert legacy.EpisodeReplayResult.__module__ == "src.sim.artifacts.types"
    assert legacy.write_episode_artifacts.__module__ == "src.sim.artifacts.snapshot"
    assert legacy.load_episode_artifacts.__module__ == "src.sim.artifacts.validation"
    assert legacy.build_replay_bundle.__module__ == "src.sim.artifacts.replay"
    # 原模块没有 __all__，facade 也不应新增。
    assert not hasattr(legacy, "__all__")
    # 仓库根：artifacts/serialization.py 必须用 parents[3]。
    assert legacy._repo_root() == ROOT


def test_config_roundtrip_and_constants() -> None:
    """配置字典往返逐字段一致，快照常量可导入。"""
    config = resolve_v12_config()
    dumped = legacy.config_to_dict(config)
    restored = legacy.config_from_dict(dumped)
    assert legacy.config_to_dict(restored) == dumped
    assert restored.dt == config.dt
    assert restored.robot_id == config.robot_id
    assert legacy.SNAPSHOT_CONFIG_NAME == "resolved_config.json"
    assert legacy.ARTIFACT_NPZ_NAME == "v12_episode.npz"


def test_write_from_non_repo_cwd_and_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolved_config: Any
) -> None:
    """在仓库之外的 cwd 写产物，目录布局/格式/NPZ 契约不变且可读回。"""
    result = _fabricated_result(resolved_config)
    out = tmp_path / "artifacts"
    monkeypatch.chdir(tmp_path)  # 非仓库 cwd

    payload = legacy.write_episode_artifacts(result, out)

    for relative in (
        "v12_episode.json",
        "v12_episode.npz",
        "snapshot/manifest.json",
        "snapshot/resolved_config.json",
        "snapshot/scene.json",
    ):
        assert (out / relative).is_file(), relative

    assert payload["schema"] == legacy.EPISODE_SCHEMA == "v12_episode_v2"
    assert payload["input_sha256"]["assets"]
    assert payload["dependency_manifest_version"] == 1
    assert payload["dependency_manifest_source"] == "derived"

    raw = (out / "v12_episode.json").read_text(encoding="utf-8")
    assert raw.endswith("\n")
    # ensure_ascii=False：payload 中文说明原样落盘。
    assert "旧版第一接触口径" in raw
    assert json.loads(raw)["schema"] == legacy.EPISODE_SCHEMA

    with np.load(out / "v12_episode.npz", allow_pickle=False) as data:
        assert set(data.files) == {
            "schema",
            "simulation_only",
            "X",
            "U",
            "ball_pos",
            "ball_vel",
            "time",
            "racket_pos",
        }
        assert str(data["schema"]) == "v12_episode_v2"
        assert bool(data["simulation_only"]) is True
        for key in ("X", "U", "ball_pos", "ball_vel", "time", "racket_pos"):
            assert data[key].dtype == np.float64, key

    manifest = json.loads((out / "snapshot/manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == legacy.EPISODE_SCHEMA
    assert manifest["assets"]["digest"] == payload["input_sha256"]["assets"]
    assert manifest["dependency_manifest_source"] == "derived"

    loaded = legacy.load_episode_artifacts(out)
    assert loaded.directory == out
    np.testing.assert_allclose(loaded.X, result.X)
    np.testing.assert_allclose(loaded.U, result.U)
    np.testing.assert_allclose(loaded.ball_pos, result.ball_pos)
    np.testing.assert_allclose(loaded.racket_pos, result.racket_pos)


def test_load_error_contract(tmp_path: Path) -> None:
    """load 的错误类型与文案保持拆分前契约。"""
    with pytest.raises(FileNotFoundError, match="缺少产物 JSON"):
        legacy.load_episode_artifacts(tmp_path)

    (tmp_path / legacy.ARTIFACT_JSON_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="缺少产物 NPZ"):
        legacy.load_episode_artifacts(tmp_path)

    npz_path = tmp_path / legacy.ARTIFACT_NPZ_NAME
    np.savez(
        npz_path,
        X=np.zeros((2, 12)),
        U=np.zeros((1, 6)),
        ball_pos=np.zeros((2, 3)),
        ball_vel=np.zeros((2, 3)),
        time=np.zeros(2),
        racket_pos=np.zeros((2, 3)),
    )
    (tmp_path / legacy.ARTIFACT_JSON_NAME).write_text(
        json.dumps({"schema": "v1"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="产物 schema 不是 v12_episode_v2"):
        legacy.load_episode_artifacts(tmp_path)

    (tmp_path / legacy.ARTIFACT_JSON_NAME).write_text(
        json.dumps({"schema": legacy.EPISODE_SCHEMA}), encoding="utf-8"
    )
    np.savez(npz_path, X=np.zeros((2, 12)))
    with pytest.raises(ValueError, match="产物 NPZ 缺少数组"):
        legacy.load_episode_artifacts(tmp_path)

    (tmp_path / legacy.ARTIFACT_JSON_NAME).write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON 顶层必须是对象"):
        legacy.load_episode_artifacts(tmp_path)

    (tmp_path / legacy.ARTIFACT_JSON_NAME).write_text("{oops", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        legacy.load_episode_artifacts(tmp_path)


def test_public_signatures_frozen() -> None:
    """四个公开入口与两个结果类型的签名与 p0 冻结值逐字一致。"""
    for name, expected in _EXPECTED_SIGNATURES.items():
        assert str(inspect.signature(getattr(legacy, name))) == expected, name


def test_replay_viewer_false(written_artifacts: tuple[Path, Any, Any]) -> None:
    """回放入口在无 viewer 模式下可调用，参数/结果字段不变。"""
    out, result, _payload = written_artifacts
    bundle = legacy.build_replay_bundle(out)
    replay = legacy.replay_episode_artifacts(
        out, speed=2.0, viewer=False, bundle=bundle
    )
    assert replay.payload["schema"] == legacy.EPISODE_SCHEMA
    assert replay.speed == 2.0
    assert replay.viewer_opened is False
    assert replay.directory == out
    assert replay.ball_positions.shape == (result.ball_pos.shape[0], 3)
    assert replay.racket_positions.shape == (result.racket_pos.shape[0], 3)


def test_package_import_does_not_pull_legacy_or_execution() -> None:
    """新子包运行时不得把旧 facade 或 episode.execution 拉进 sys.modules。"""
    code = (
        "import sys; import src.sim.artifacts; "
        "bad = [m for m in ('src.sim.episode_artifacts', "
        "'src.sim.episode.execution') if m in sys.modules]; "
        "assert not bad, bad"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_source_structure() -> None:
    """facade 只做重导出；artifacts 包不反向依赖旧模块/执行层。"""
    facade_path = ROOT / "src" / "sim" / "episode_artifacts.py"
    facade_source = facade_path.read_text(encoding="utf-8")
    assert len(facade_source.splitlines()) < 250
    facade_tree = ast.parse(facade_source)
    # facade 只能保留模块级重导出与一个属性转发垫片类，不得承载业务函数。
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in facade_tree.body
    ), "facade 不得定义模块级函数"
    facade_classes = {
        node.name for node in facade_tree.body if isinstance(node, ast.ClassDef)
    }
    assert facade_classes <= {"_FacadeModule"}, facade_classes
    for node in ast.walk(facade_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.startswith("src.sim.artifacts") or alias.name in {
                    "__future__",
                    "logging",
                    "tempfile",
                    "sys",
                    "types",
                }, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module.startswith("src.sim.artifacts") or module in {
                "__future__",
            }, module

    for path in sorted((ROOT / "src" / "sim" / "artifacts").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "src.sim.episode_artifacts"
                    assert not alias.name.startswith("src.sim.episode_artifacts.")
                    assert alias.name != "src.sim.episode.execution"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module != "src.sim.episode_artifacts"
                assert not module.startswith("src.sim.episode_artifacts.")
                assert module != "src.sim.episode.execution"
                if module in {"src.sim", "src.sim.episode"}:
                    assert "execution" not in {
                        alias.name for alias in node.names
                    }, (path.name, node.lineno)
                if any(alias.name == "V12EpisodeResult" for alias in node.names):
                    assert module == "src.sim.episode.types", (
                        path.name,
                        node.lineno,
                        module,
                    )
