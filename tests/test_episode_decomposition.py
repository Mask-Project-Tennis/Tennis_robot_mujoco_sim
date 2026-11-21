"""B2 阶段拆分契约测试（facade / 阶段顺序 / 早退 / 结构防回退）。

这些测试**不启动真实仿真**：全部通过 monkeypatch 阶段入口验证编排关系。真实
闭环行为由既有 ``test_v12_*`` 测试覆盖，避免重复长跑。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.ilqt.components.sim_component import SimComponent
from src.ilqt.episode_runner import EpisodeRunner
from src.ilqt.mpc_controller import MPCController
from src.sim import v12_episode
from src.sim.configuration.types import ResolvedV12Config
from src.sim.episode import (
    V12EpisodeResult as PackageResult,
)
from src.sim.episode import (
    assembly,
    execution,
    hooks,
    post_hit,
    preparation,
    results,
)
from src.sim.episode.types import V12EpisodeResult as TypesResult
from src.sim.v12_robot_adapter import build_v12_robot

REPO_ROOT = Path(__file__).resolve().parents[1]
FACADE_PATH = REPO_ROOT / "src" / "sim" / "v12_episode.py"
EXECUTION_PATH = REPO_ROOT / "src" / "sim" / "episode" / "execution.py"

# 旧路径必须继续可见的私有名 → 现在所属阶段模块
_PRIVATE_EXPORTS = [
    (preparation, "_DEFAULT_HIT_DIRECTION"),
    (preparation, "_effective_backend"),
    (preparation, "_model_shoulder_pos"),
    (preparation, "_mpc_workspace_center_pos"),
    (preparation, "_normalize_hit_direction"),
    (preparation, "_predict_effective_backend"),
    (preparation, "_resolve_hit_direction"),
    (preparation, "_resolve_serve_box_center"),
    (preparation, "_resolve_serve_box_halfsize"),
    (preparation, "_resolve_shoulder_pos"),
    (preparation, "_select_follow_direction"),
    (preparation, "_set_rm65_left_arm"),
    (preparation, "_shoulder_body_name"),
    (preparation, "_build_tube_config"),
    (assembly, "_build_mpc_config"),
    (assembly, "_build_obs_gate"),
    (assembly, "_racket_radius_from_bundle"),
    (assembly, "_resolve_body_ids"),
    (assembly, "_resolve_geom_ids"),
    (assembly, "_resolve_geom_pairs"),
    (post_hit, "_POST_HIT_ALLOWED_STOP_REASONS"),
    (post_hit, "_POST_HIT_STEPS"),
    (post_hit, "_apply_post_hit_phase"),
    (post_hit, "_run_post_hit_hold"),
    (hooks, "_seed_ball_history"),
    (results, "_apply_limits_overrides"),
    (results, "_explicit_runtime_options"),
    (results, "_model_dependency_manifest"),
    (results, "_r1d_artifact_metadata"),
    (results, "_real_config_snapshot"),
    (results, "_resolve_failure_reason"),
    (results, "_runtime_option_overrides"),
    (results, "_scene_to_jsonable"),
    (results, "_serialize_hit_window"),
    (results, "_sha256_file"),
]


def _resolved_config() -> ResolvedV12Config:
    """构造最小合法 ``ResolvedV12Config``（仅供编排测试，不参与真实解析）。"""
    return ResolvedV12Config(
        robot_id="rm65",
        task_mode="legacy_cli",
        robot_profile_path=None,
        runtime_path=None,
        racket_path=None,
        task_path=None,
        dt=0.005,
        seed=0,
        actuator_mode="torque",
        feedforward=None,
        kp=None,
        kd=None,
        ablation_mode="full",
        solver_backend="cpp",
        solver_config={"backend": "cpp"},
        mpc_config={"total_horizon": 10},
        cost_config={},
        perception_config={},
        hitting_config={},
        scene_config={},
        serve_config={"mode": "serve_box", "ball_speed": 7.0},
        target_center=np.zeros(3),
        target_offset=0.1,
        workspace_radius=0.8,
        output_dir=Path("results/test"),
        viewer=False,
        no_plot=True,
        sources={},
        explicit_options=frozenset(),
    )


def test_result_type_is_single_definition() -> None:
    """旧路径、包入口与 types 模块必须是同一个 ``V12EpisodeResult`` 类。"""
    assert v12_episode.V12EpisodeResult is TypesResult
    assert PackageResult is TypesResult
    assert TypesResult.__module__ == "src.sim.episode.types"


def test_facade_keeps_public_exports() -> None:
    """facade 的 ``__all__`` 与公有重导出保持旧契约。"""
    assert v12_episode.__all__ == ["V12EpisodeResult", "run_v12_episode"]
    assert v12_episode.build_v12_robot is build_v12_robot
    assert v12_episode.MPCController is MPCController
    assert v12_episode.SimComponent is SimComponent
    assert v12_episode.EpisodeRunner is EpisodeRunner


@pytest.mark.parametrize(("module", "name"), _PRIVATE_EXPORTS)
def test_private_compat_names_are_same_objects(
    module: object, name: str
) -> None:
    """旧路径私有名必须存在且与阶段模块内是同一对象（含 monkeypatch 语义）。"""
    assert hasattr(v12_episode, name), name
    assert getattr(v12_episode, name) is getattr(module, name), name


def test_facade_signature_unchanged() -> None:
    """facade 签名逐字保留（含 9 个 UNSET 覆盖参数与顺序）。"""
    import inspect

    from src.sim.episode_runtime import UNSET

    params = list(inspect.signature(v12_episode.run_v12_episode).parameters)
    assert params == [
        "config",
        "dump_trajectory",
        "first_plan_flight",
        "always_collision",
        "ball_speed_perturb_pct",
        "fixed_direction",
        "hit_direction",
        "post_exec_hooks",
        "limits_config",
        "max_tcp",
        "terminal_exempt_steps",
        "dq_max_fraction",
    ]
    defaults = inspect.signature(v12_episode.run_v12_episode).parameters
    for name in (
        "first_plan_flight",
        "always_collision",
        "ball_speed_perturb_pct",
        "fixed_direction",
        "hit_direction",
        "limits_config",
        "max_tcp",
        "terminal_exempt_steps",
        "dq_max_fraction",
    ):
        assert defaults[name].default is UNSET, name
    assert defaults["dump_trajectory"].default is None
    assert defaults["post_exec_hooks"].default == ()


def test_facade_reads_patched_symbols_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """facade 调用时读取本模块三个可变属性并注入 impl（monkeypatch 契约）。"""
    captured: dict[str, object] = {}

    def spy_impl(config: object, **kwargs: object) -> object:
        captured["config"] = config
        captured.update(kwargs)
        return "impl-sentinel"

    class _FakeMPC:
        pass

    class _FakeSim:
        pass

    fake_builder = object()
    monkeypatch.setattr(v12_episode, "run_v12_episode_impl", spy_impl)
    monkeypatch.setattr(v12_episode, "build_v12_robot", fake_builder)
    monkeypatch.setattr(v12_episode, "MPCController", _FakeMPC)
    monkeypatch.setattr(v12_episode, "SimComponent", _FakeSim)

    config = object()  # 类型校验在 impl 内；spy 不校验
    result = v12_episode.run_v12_episode(
        config,  # type: ignore[arg-type]
        first_plan_flight=False,
        hit_direction=(1.0, 0.0, 0.0),
    )

    assert result == "impl-sentinel"
    assert captured["config"] is config
    assert captured["robot_builder"] is fake_builder
    assert captured["mpc_controller_cls"] is _FakeMPC
    assert captured["sim_component_cls"] is _FakeSim
    assert captured["first_plan_flight"] is False
    assert captured["hit_direction"] == (1.0, 0.0, 0.0)


def test_impl_forwards_robot_builder_to_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    """impl 必须把 facade 注入的 ``robot_builder`` 透传给准备阶段。"""
    captured: dict[str, object] = {}
    fake_builder = object()

    def fake_prepare(config: object, **kwargs: object) -> SimpleNamespace:
        captured["config"] = config
        captured.update(kwargs)
        return SimpleNamespace(hit_info=None)

    monkeypatch.setattr(execution, "prepare_episode", fake_prepare)
    monkeypatch.setattr(
        execution, "build_unreachable_result", lambda prep: "unreachable-sentinel"
    )

    result = execution.run_v12_episode_impl(
        _resolved_config(), robot_builder=fake_builder,  # type: ignore[arg-type]
    )

    assert result == "unreachable-sentinel"
    assert captured["robot_builder"] is fake_builder


def test_stage_order_and_injected_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    """编排顺序固定为 prepare → assemble → runner.run → post_hit → result。"""
    sequence: list[str] = []
    captured: dict[str, object] = {}

    class _FakeMPC:
        pass

    class _FakeSim:
        pass

    def fake_run(**kwargs: object) -> dict[str, object]:
        sequence.append("run")
        captured["run_kwargs"] = kwargs
        return {"stop_reason": "mpc_done", "safe_steps": 3}

    fake_prep = SimpleNamespace(
        hit_info=object(),
        total_steps=7,
        first_plan_flight_value=False,
        execution=SimpleNamespace(
            get_arm_state=lambda: np.zeros(12, dtype=np.float64)
        ),
    )
    fake_assembly = SimpleNamespace(runner=SimpleNamespace(run=fake_run))

    def fake_prepare(*args: object, **kwargs: object) -> SimpleNamespace:
        sequence.append("prepare")
        return fake_prep

    def fake_assemble(*args: object, **kwargs: object) -> SimpleNamespace:
        sequence.append("assemble")
        captured["assemble_kwargs"] = kwargs
        return fake_assembly

    def fake_post_hit(**kwargs: object) -> tuple[int, None]:
        sequence.append("post_hit")
        return 0, None

    def fake_build_result(**kwargs: object) -> str:
        sequence.append("result")
        return "result-sentinel"

    monkeypatch.setattr(execution, "prepare_episode", fake_prepare)
    monkeypatch.setattr(execution, "assemble_episode", fake_assemble)
    monkeypatch.setattr(execution, "run_post_hit", fake_post_hit)
    monkeypatch.setattr(execution, "build_result", fake_build_result)

    result = execution.run_v12_episode_impl(
        _resolved_config(),
        mpc_controller_cls=_FakeMPC,  # type: ignore[arg-type]
        sim_component_cls=_FakeSim,  # type: ignore[arg-type]
    )

    assert result == "result-sentinel"
    assert sequence == ["prepare", "assemble", "run", "post_hit", "result"]
    assert captured["run_kwargs"] == {
        "max_steps": 7,
        "first_plan_flight": False,
    }
    assemble_kwargs = captured["assemble_kwargs"]
    assert isinstance(assemble_kwargs, dict)
    assert assemble_kwargs["mpc_controller_cls"] is _FakeMPC
    assert assemble_kwargs["sim_component_cls"] is _FakeSim


def test_unreachable_early_return_skips_assembly(monkeypatch: pytest.MonkeyPatch) -> None:
    """``hit_info is None`` 时只构造早退结果，不进入组装/运行。"""
    bundle = SimpleNamespace(
        robot_id="rm65",
        init_q=np.zeros(6, dtype=np.float64),
        joint_names=tuple(f"r_joint{i}" for i in range(1, 7)),
        racket=None,
        robot_model_bundle=None,
    )
    fake_prep = SimpleNamespace(
        hit_info=None,
        config=_resolved_config(),
        bundle=bundle,
        execution=SimpleNamespace(NQ=6, NX=12, NU=6),
        backend={"requested": "cpp", "effective": "cpp"},
        p0=np.array([1.0, 2.0, 3.0]),
        v0=np.array([0.0, -7.0, 0.0]),
        rng=np.random.default_rng(0),
        target_center=np.zeros(3),
        workspace_radius=0.8,
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("早退路径不得进入组装/运行/击打后阶段")

    monkeypatch.setattr(execution, "prepare_episode", lambda *a, **k: fake_prep)
    monkeypatch.setattr(execution, "assemble_episode", forbidden)
    monkeypatch.setattr(execution, "run_post_hit", forbidden)

    result = execution.run_v12_episode_impl(_resolved_config())

    assert result.hit_type == "miss"
    assert result.hit_step == -1
    assert result.metrics["failure_reason"] == "ball_unreachable"
    assert result.metrics["post_hit_skipped_reason"] == "ball_unreachable"
    assert result.metrics["unreachable"] is True
    assert result.U.shape == (0, 6)
    assert result.X.shape == (1, 12)


def test_facade_stays_thin() -> None:
    """结构防回退：facade 行数 < 250，``run_v12_episode`` 体只剩一次委托。"""
    source = FACADE_PATH.read_text(encoding="utf-8")
    assert len(source.splitlines()) < 250

    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_v12_episode"
    )
    body = [
        node
        for node in function.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    assert len(body) == 1
    assert isinstance(body[0], ast.Return)


def test_execution_orchestrator_stays_small() -> None:
    """结构防回退：``run_v12_episode_impl`` 行数 < 250（只编排、不复制阶段）。"""
    tree = ast.parse(EXECUTION_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_v12_episode_impl"
    )
    assert function.end_lineno is not None
    assert function.lineno is not None
    assert function.end_lineno - function.lineno + 1 < 250

    source = EXECUTION_PATH.read_text(encoding="utf-8")
    for stage_call in (
        "prepare_episode(",
        "assemble_episode(",
        "run_post_hit(",
        "build_result(",
    ):
        assert stage_call in source


def test_stage_modules_do_not_import_facade() -> None:
    """阶段模块不得反向 import facade（防循环依赖）。"""
    for module in (preparation, assembly, hooks, post_hit, results, execution):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith("src.sim.v12_episode")
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("src.sim.v12_episode")
                    for alias in node.names
                )
