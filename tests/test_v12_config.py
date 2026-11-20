"""R1D-V12 parity T3：V12 唯一任务配置解析层测试。

测试对象：`src/sim/v12_config.py` 的 `resolve_v12_config`。
设计目标（与 T3 brief 对齐）：

1. RM65 旧路径数值快照必须与 T0 冻结基线逐字段一致（default→v5→mpc 合并
   + 显式 CLI/服务箱自动值 + `src/robot/constants.py` 仓库常量）。
2. 显式 CLI 覆盖不得被服务箱自动值吞掉。
3. R1D 只经显式注册表解析 runtime/task/racket 路径，不按文件名猜左右侧。
4. 未知 robot、错配 runtime、非法任务覆盖、6/7 维向量冲突必须在运行前失败。
5. 两种机器人依次解析互不污染，返回的字典/数组为独立深拷贝。

测试全部纯 Python（不加载 MuJoCo），使用 `argparse.Namespace` 局部字段 +
resolver 内建缺省，避免复制整份 parser。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.ilqt.tube_types import TubeConfig
from src.robot.robot_profile import load_robot_profile
from src.sim.v12_config import (
    R1D_V12_REGISTRY,
    R1DConfigRegistry,
    ResolvedV12Config,
    normalize_explicit_options,
    resolve_v12_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOTS_DIR = REPO_ROOT / "configs" / "robots"


def _args(**overrides: object) -> argparse.Namespace:
    """构造只含所需字段的局部 argparse.Namespace（其余走 resolver 缺省）。"""
    return argparse.Namespace(**overrides)


# ---------------------------------------------------------------------------
# RM65 基线快照
# ---------------------------------------------------------------------------


def _resolve_rm65_snapshot() -> ResolvedV12Config:
    """解析 T0 基线命令对应的配置（serve-box / 7 m/s / seed 42 / torque）。"""
    return resolve_v12_config(
        _args(serve_box=True, ball_speed=7.0, seed=42, no_plot=True),
        frozenset({"--serve-box", "--ball-speed", "--seed", "--no-plot"}),
    )


def test_rm65_serve_box_snapshot_matches_t0() -> None:
    """RM65 serve-box 7 m/s 快照与 T0 冻结基线一致。"""
    r = _resolve_rm65_snapshot()
    assert r.robot_id == "rm65"
    assert r.task_mode == "legacy_cli"
    assert r.robot_profile_path is None
    assert r.runtime_path is None
    assert r.task_path is None
    assert r.racket_path is None

    assert r.dt == pytest.approx(0.005)
    assert r.seed == 42
    assert r.actuator_mode == "torque"
    assert r.feedforward is None
    assert r.kp is None and r.kd is None
    assert r.ablation_mode == "full"
    assert r.solver_backend == "auto"

    m = r.mpc_config
    assert m["total_horizon"] == 250
    assert m["fixed_horizon"] == 120
    assert m["replan_interval"] == 20
    assert m["max_iter_per_plan"] == 10
    assert m["first_plan_iters"] == 30
    assert m["near_plan_iters"] == 5
    assert m["near_threshold"] is None  # 运行时由 k_hit 推导
    assert m["Q_p_scale_far"] == pytest.approx(5.0)
    assert m["Q_v_scale_far"] == pytest.approx(3.0)
    assert m["Q_p_scale_near"] == pytest.approx(8.0)
    assert m["Q_v_scale_near"] == pytest.approx(120.0)
    assert m["normal_weight"] == pytest.approx(5.0e5)
    assert m["racket_speed"] == pytest.approx(1.8)
    assert m["target_speed"] == pytest.approx(1.8)
    assert m["follow_through_length"] == pytest.approx(0.0)
    assert m["follow_through_steps"] == 60
    assert m["follow_through_v_terminal"] == pytest.approx(0.3)
    assert m["use_r_decay"] is True
    assert m["r_decay_ratio"] == pytest.approx(0.40)
    assert m["far_threshold"] == 50
    assert m["follow_kp"] == pytest.approx(200.0)
    assert m["follow_kd"] == pytest.approx(20.0)
    assert m["hit_lock_threshold"] == 60
    assert m["hard_margin_deg"] == pytest.approx(2.0)
    assert m["warn_margin_deg"] == pytest.approx(5.0)
    assert m["j1_warn_margin_deg"] == pytest.approx(8.0)
    assert m["refiner_window_half"] == 15
    assert m["fix_joint5_angle"] is None

    assert list(r.cost_config["Q_p"]) == [50000.0, 50000.0, 50000.0]
    assert list(r.cost_config["Q_v"]) == [200.0, 200.0, 200.0]
    assert list(m["Q_p_base"]) == [100000.0, 100000.0, 100000.0]
    assert list(m["Q_v_base"]) == [400.0, 400.0, 400.0]
    assert r.cost_config["R"] == pytest.approx(1e-4)
    assert r.cost_config["Q_qdot"] == pytest.approx(0.001)
    assert r.cost_config["Q_qddot"] == pytest.approx(0.0005)
    assert r.cost_config["Q_du"] == pytest.approx(0.001)

    tube = r.cost_config["tube"]
    assert tube["window_half_ms"] == pytest.approx(50.0)
    assert tube["Q_p_tube"] == pytest.approx(500.0)
    assert tube["Q_v_tube"] == pytest.approx(0.0)
    assert tube["Q_n_tube"] == pytest.approx(0.0)
    assert tube["tube_cost_ratio"] == pytest.approx(1.0)
    assert tube["softmin_beta"] == pytest.approx(5.0)
    assert tube["corridor_radius"] == pytest.approx(0.12)
    assert tube["use_softmin_terminal"] is True

    assert r.cost_config["stage_weights"]["far"] == {
        "Q_qdot_mult": 1.0,
        "Q_qddot_mult": 1.0,
        "Q_du_mult": 1.0,
    }
    assert r.cost_config["stage_weights"]["mid"] == {
        "Q_qdot_mult": 2.0,
        "Q_qddot_mult": 2.0,
        "Q_du_mult": 2.0,
    }
    assert r.cost_config["stage_weights"]["near"] == {
        "Q_qdot_mult": 0.0,
        "Q_qddot_mult": 0.0,
        "Q_du_mult": 0.0,
    }

    assert np.allclose(r.target_center, [-0.82765693, -0.47411682, 0.86947444])
    assert r.target_offset == pytest.approx(0.10)
    assert r.workspace_radius == pytest.approx(0.90)
    assert np.allclose(r.scene_config["shoulder_pos"], [-0.1, -0.22693, 1.302645])
    assert r.scene_config["use_bounce"] is True
    assert r.scene_config["bounce_restitution"] == pytest.approx(0.75)
    assert np.allclose(r.scene_config["ball_gravity"], [0.0, 0.0, -9.81])
    assert r.scene_config["ball_radius"] == pytest.approx(0.033)
    assert r.scene_config["ball_mass"] == pytest.approx(0.058)

    assert r.serve_config["mode"] == "serve_box"
    assert r.serve_config["ball_speed"] == pytest.approx(7.0)
    assert tuple(r.serve_config["serve_box_center"]) == (0.0, -8.0, 1.2)
    assert tuple(r.serve_config["serve_box_halfsize"]) == (4.0, 0.1, 0.15)
    assert r.serve_config["speed_range"] == (8.0, 18.0)

    assert np.allclose(r.hitting_config["hit_direction"], [0.0, -1.0, 0.3])
    assert r.hitting_config["racket_speed"] == pytest.approx(1.8)

    assert r.output_dir == (REPO_ROOT / "results").resolve()
    assert r.viewer is False
    assert r.no_plot is True


def test_rm65_snapshot_sources() -> None:
    """来源表记录文件/常量/自动值/CLI 的可追溯标签。"""
    r = _resolve_rm65_snapshot()
    s = r.sources
    assert s["dt"] == "default.yaml"
    assert s["cost.Q_p"] == "v5_active_hit.yaml"
    assert s["hitting.racket_speed"] == "v5_active_hit.yaml"
    assert s["mpc.total_horizon"] == "serve_box_auto"
    assert s["mpc.fixed_horizon"] == "serve_box_auto"
    assert s["mpc.first_plan_iters"] == "serve_box_auto"
    assert s["mpc.Q_p_base"] == "v5_active_hit.yaml"
    assert s["target.center"] == "V12 字面量"
    assert s["target.shoulder_pos"] == "src/robot/constants.py"
    assert s["serve.mode"] == "cli:--serve-box"
    assert s["serve.ball_speed"] == "cli:--ball-speed"
    assert s["seed"] == "cli:--seed"
    assert s["mpc.near_threshold"] == "deferred(k_hit)"


def test_serve_box_auto_does_not_swallow_explicit_cli() -> None:
    """显式 CLI 覆盖不得被 serve-box 自动参数吞掉。"""
    r = resolve_v12_config(
        _args(
            serve_box=True,
            horizon=90,
            iter=7,
            replan_interval=11,
            near_iters=3,
            first_plan_iters=5,
        ),
        frozenset(
            {
                "--serve-box",
                "--horizon",
                "--iter",
                "--replan-interval",
                "--near-iters",
                "--first-plan-iters",
            }
        ),
    )
    m = r.mpc_config
    assert m["fixed_horizon"] == 90
    assert m["max_iter_per_plan"] == 7
    assert m["replan_interval"] == 11
    assert m["near_plan_iters"] == 3
    assert m["first_plan_iters"] == 5
    assert m["total_horizon"] == 250
    assert r.sources["mpc.fixed_horizon"] == "cli:--horizon"
    assert r.sources["mpc.max_iter_per_plan"] == "cli:--iter"


def test_target_box_mode_defaults() -> None:
    """非 serve-box 模式沿用 V12 字面量默认时间参数。"""
    r = resolve_v12_config(_args(), frozenset())
    m = r.mpc_config
    assert m["total_horizon"] == 200
    assert m["fixed_horizon"] == 40
    assert m["replan_interval"] == 30
    assert m["max_iter_per_plan"] == 5
    assert m["first_plan_iters"] == 15
    assert m["near_plan_iters"] == 20
    assert r.serve_config["mode"] == "target_box"
    assert r.serve_config["ball_start_y_range"] == (-5.5, -4.5)
    assert r.serve_config["ball_start_z_range"] == (1.4, 1.8)
    assert r.serve_config["ball_direction"] == "y"
    assert r.serve_config["hit_time_uniform"] == (0.3, 0.4)


# ---------------------------------------------------------------------------
# 执行器 / 观测 / 消融 / 扰动
# ---------------------------------------------------------------------------


def test_position_mode_rm65_uses_config_gains_and_disables_r_decay() -> None:
    r = resolve_v12_config(
        _args(position_mode=True),
        frozenset({"--position-mode"}),
    )
    assert r.actuator_mode == "position"
    assert r.feedforward is True
    assert np.allclose(r.kp, [25.0] * 6)
    assert np.allclose(r.kd, [2.0] * 6)
    assert r.mpc_config["use_r_decay"] is False
    assert r.sources["mpc.use_r_decay"] == "position_mode_rule"


def test_position_mode_no_feedforward_and_gain_broadcast() -> None:
    r = resolve_v12_config(
        _args(position_mode=True, no_feedforward=True, kp=[3.5], kd=[0.5]),
        frozenset({"--position-mode", "--no-feedforward", "--kp", "--kd"}),
    )
    assert r.feedforward is False
    assert np.allclose(r.kp, [3.5] * 6)
    assert np.allclose(r.kd, [0.5] * 6)
    assert r.sources["kp"] == "cli:--kp"


def test_ablation_and_observation_passthrough() -> None:
    r = resolve_v12_config(
        _args(
            no_tube=True,
            obs_noise_pos=0.01,
            obs_noise_vel=0.1,
            obs_use_kf=True,
        ),
        frozenset({"--no-tube", "--obs-noise-pos", "--obs-noise-vel", "--obs-use-kf"}),
    )
    assert r.ablation_mode == "softmin_only"
    assert r.cost_config["tube"]["use_softmin_terminal"] is True
    p = r.perception_config
    assert p["obs_freq_effective"] == pytest.approx(200.0)
    assert p["obs_use_kf"] is True
    assert p["use_kf"] is True
    assert p["noise_enabled"] is True

    r2 = resolve_v12_config(
        _args(no_tube=True, no_softmin=True),
        frozenset({"--no-tube", "--no-softmin"}),
    )
    assert r2.ablation_mode == "none"
    assert r2.cost_config["tube"]["use_softmin_terminal"] is False


def test_perturbation_passthrough() -> None:
    r = resolve_v12_config(
        _args(time_perturb_ms=10.0, space_perturb_m=0.02, perturb_alpha_min=0.5),
        frozenset({"--time-perturb-ms", "--space-perturb-m", "--perturb-alpha-min"}),
    )
    m = r.mpc_config
    assert m["time_perturb_s"] == pytest.approx(0.01)
    assert m["space_perturb_m"] == pytest.approx(0.02)
    assert m["perturb_alpha_min"] == pytest.approx(0.5)
    assert m["spatial_perturb_target"] == "hitpoint"


def test_follow_through_disable_zeroes_all_fields() -> None:
    r = resolve_v12_config(
        _args(no_follow_through=True),
        frozenset({"--no-follow-through"}),
    )
    m = r.mpc_config
    assert m["follow_through_length"] == 0.0
    assert m["follow_through_steps"] == 0
    assert m["follow_through_v_terminal"] == 0.0
    assert r.hitting_config["follow_through_steps"] == 0


# ---------------------------------------------------------------------------
# R1D 注册表
# ---------------------------------------------------------------------------


def test_r1d_registry_resolves_three_paths_without_sidecar_guess() -> None:
    """仅传 r1d_right profile 时由显式注册表给出 runtime/task/racket。"""
    r = resolve_v12_config(
        _args(robot_config=ROBOTS_DIR / "r1d_right.yaml"),
        frozenset({"--robot-config"}),
    )
    assert r.robot_id == "unitree_r1d"
    assert r.task_mode == "r1d_overlay"
    assert r.robot_profile_path == (ROBOTS_DIR / "r1d_right.yaml").resolve()
    assert r.runtime_path == (
        REPO_ROOT / "configs" / "robots" / "r1d_v12_runtime_right.yaml"
    ).resolve()
    assert r.task_path == (REPO_ROOT / "configs" / "r1d_v12.yaml").resolve()
    assert r.racket_path == (
        REPO_ROOT / "configs" / "rackets" / "blade100_v9_reference.yaml"
    ).resolve()
    assert r.sources["runtime_path"] == "registry:r1d_right"
    assert r.sources["racket_path"] == "registry:r1d_right"

    assert r.actuator_mode == "torque"
    assert r.kp is None and r.kd is None
    assert r.scene_config["shoulder_pos"] is None
    assert r.workspace_radius == pytest.approx(0.80)
    assert np.allclose(r.target_center, [-11.75, -0.9, 1.25])
    assert r.target_offset == pytest.approx(0.10)
    assert r.sources["target.center"] == "r1d_v12.yaml"
    assert r.sources["target.workspace_radius"] == "r1d_v12.yaml"
    # 未显式加载 runtime 前，6 维 RM65 PD 增益不得泄漏给 7 维 R1D
    assert r.kp is None
    # 新路径以 RacketProfile 为球拍几何唯一来源：旧 runtime 的球拍字段标记 legacy-unused
    assert (
        r.sources["runtime.scene.racket_position"]
        == "legacy-unused(racket_profile)"
    )
    assert (
        r.sources["runtime.scene.racket_radius"]
        == "legacy-unused(racket_profile)"
    )


def test_r1d_registry_metadata_is_complete() -> None:
    """注册表条目必须给全 runtime/task/racket 三路径与侧别。"""
    right_key = (ROBOTS_DIR / "r1d_right.yaml").resolve()
    entry = R1D_V12_REGISTRY[right_key]
    assert isinstance(entry, R1DConfigRegistry)
    assert entry.side == "right"
    assert entry.runtime_path.is_absolute()
    assert entry.task_path.is_absolute()
    assert entry.racket_path.is_absolute()
    left_key = (ROBOTS_DIR / "r1d_left.yaml").resolve()
    assert left_key not in R1D_V12_REGISTRY


def test_r1d_explicit_runtime_wins_and_runtime_vectors_used() -> None:
    """显式 runtime 优先于注册表缺省，7 维 PD 增益取自 runtime。"""
    runtime_path = (ROBOTS_DIR / "r1d_sim_runtime_right.yaml").resolve()
    r = resolve_v12_config(
        _args(
            robot_config=ROBOTS_DIR / "r1d_right.yaml",
            runtime_config=runtime_path,
            position_mode=True,
        ),
        frozenset({"--robot-config", "--runtime-config", "--position-mode"}),
    )
    assert r.runtime_path == runtime_path
    assert r.sources["runtime_path"] == "cli:--runtime-config"
    assert r.actuator_mode == "position"
    assert r.kp is not None and r.kp.shape == (7,)
    assert np.allclose(r.kp, [400.0, 400.0, 250.0, 120.0, 60.0, 40.0, 40.0])
    assert np.allclose(r.kd, [12.0, 12.0, 8.0, 5.0, 2.5, 1.5, 1.5])
    # position 模式对 R1D 同样强制禁用 R 退火
    assert r.mpc_config["use_r_decay"] is False


def test_r1d_position_mode_kp_length_conflict_fails() -> None:
    """R1D 七维下给 6 元 kp 必须运行前失败（不得静默按 6 维使用）。"""
    runtime_path = (ROBOTS_DIR / "r1d_sim_runtime_right.yaml").resolve()
    with pytest.raises(ValueError, match="kp 长度"):
        resolve_v12_config(
            _args(
                robot_config=ROBOTS_DIR / "r1d_right.yaml",
                runtime_config=runtime_path,
                position_mode=True,
                kp=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
            frozenset({"--robot-config", "--runtime-config", "--position-mode", "--kp"}),
        )


def test_rm65_kp_length_seven_fails() -> None:
    """RM65 六维下给 7 元 kp 必须运行前失败。"""
    with pytest.raises(ValueError, match="kp 长度"):
        resolve_v12_config(
            _args(position_mode=True, kp=[1.0] * 7),
            frozenset({"--position-mode", "--kp"}),
        )


def test_r1d_runtime_side_mismatch_fails() -> None:
    """右臂 profile 配左臂 runtime 必须运行前失败。"""
    left_runtime = (ROBOTS_DIR / "r1d_sim_runtime_left.yaml").resolve()
    with pytest.raises(ValueError, match="显式 runtime"):
        resolve_v12_config(
            _args(
                robot_config=ROBOTS_DIR / "r1d_right.yaml",
                runtime_config=left_runtime,
            ),
            frozenset({"--robot-config", "--runtime-config"}),
        )


def test_r1d_unregistered_profile_requires_explicit_paths() -> None:
    """未注册的 R1D profile（左臂）不得偷偷找 sidecar。"""
    with pytest.raises(ValueError, match="注册表"):
        resolve_v12_config(
            _args(robot_config=ROBOTS_DIR / "r1d_left.yaml"),
            frozenset({"--robot-config"}),
        )


def test_r1d_forbidden_rm65_flags_fail() -> None:
    """R1D 明确拒绝 RM65 专属旗标，不能接受后忽略。"""
    for kwargs, flag in (
        ({"limits_config": Path("configs/real_robot.yaml")}, "--limits-config"),
        ({"fix_joint5": True}, "--fix-joint5"),
    ):
        with pytest.raises(ValueError, match=flag):
            resolve_v12_config(
                _args(robot_config=ROBOTS_DIR / "r1d_right.yaml", **kwargs),
                frozenset({"--robot-config", flag}),
            )


def test_r1d_registry_accepts_async_replan() -> None:
    """R1D 异步重规划（计时实验必需）由 T4/T5 消费，T3 只解析不拒绝。"""
    r = resolve_v12_config(
        _args(robot_config=ROBOTS_DIR / "r1d_right.yaml", async_replan=True),
        frozenset({"--robot-config", "--async-replan"}),
    )
    assert r.task_mode == "r1d_overlay"
    assert r.mpc_config["async_mode"] is True
    assert r.sources["mpc.async_mode"] == "cli:--async-replan"


# ---------------------------------------------------------------------------
# 固定球旧任务兼容模式
# ---------------------------------------------------------------------------


def _write_compat_task(
    tmp_path: Path,
    profile_path: Path = ROBOTS_DIR / "r1d_right.yaml",
) -> Path:
    """写最小 R1D 旧 sim-task YAML（resolver 只校验身份/关节序）。"""
    profile = load_robot_profile(profile_path)
    payload = {
        "schema_version": 1,
        "robot_id": profile.robot_id,
        "joint_names": list(profile.active_joint_names),
    }
    path = tmp_path / f"{profile_path.stem}_compat_task.yaml"
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8"
    )
    return path


def test_fixed_ball_compat_mode_happy_path(tmp_path: Path) -> None:
    task_path = _write_compat_task(tmp_path)
    runtime_path = (ROBOTS_DIR / "r1d_sim_runtime_right.yaml").resolve()
    r = resolve_v12_config(
        _args(
            robot_config=ROBOTS_DIR / "r1d_right.yaml",
            runtime_config=runtime_path,
            sim_task_config=task_path,
        ),
        frozenset({"--robot-config", "--runtime-config", "--sim-task-config"}),
    )
    assert r.task_mode == "r1d_fixed_ball_compat"
    assert r.task_path == task_path.resolve()
    assert r.runtime_path == runtime_path


def test_fixed_ball_compat_conflicts_with_serve_flags(tmp_path: Path) -> None:
    task_path = _write_compat_task(tmp_path)
    runtime_path = (ROBOTS_DIR / "r1d_sim_runtime_right.yaml").resolve()
    for kwargs, flag in (
        ({"serve_box": True}, "--serve-box"),
        ({"ball_speed": 7.0}, "--ball-speed"),
        ({"no_bounce": True}, "--no-bounce"),
        ({"serve_distance": 6.0}, "--serve-distance"),
    ):
        with pytest.raises(ValueError, match="固定球兼容模式"):
            resolve_v12_config(
                _args(
                    robot_config=ROBOTS_DIR / "r1d_right.yaml",
                    runtime_config=runtime_path,
                    sim_task_config=task_path,
                    **kwargs,
                ),
                frozenset(
                    {"--robot-config", "--runtime-config", "--sim-task-config", flag}
                ),
            )


def test_rm65_rejects_r1d_only_configs(tmp_path: Path) -> None:
    task_path = _write_compat_task(tmp_path)
    with pytest.raises(ValueError, match="RM65"):
        resolve_v12_config(
            _args(runtime_config=Path("configs/robots/r1d_sim_runtime_right.yaml")),
            frozenset({"--runtime-config"}),
        )
    with pytest.raises(ValueError, match="RM65"):
        resolve_v12_config(
            _args(sim_task_config=task_path),
            frozenset({"--sim-task-config"}),
        )


# ---------------------------------------------------------------------------
# 非法 robot / 覆盖
# ---------------------------------------------------------------------------


def _write_unknown_profile(tmp_path: Path) -> Path:
    """写一个 robot_id 未知但结构合法的临时 profile。"""
    model = tmp_path / "dummy_model.xml"
    model.write_text("<mujoco/>", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "robot_id": "ghost_robot",
        "model_path": "dummy_model.xml",
        "active_joint_names": ["joint1"],
        "actuator_names": ["actuator1"],
        "held_joint_names": [],
        "tcp_site_name": "tcp",
    }
    path = tmp_path / "ghost.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


def test_unknown_robot_id_fails_before_runtime(tmp_path: Path) -> None:
    profile_path = _write_unknown_profile(tmp_path)
    with pytest.raises(ValueError, match="未知"):
        resolve_v12_config(
            _args(robot_config=profile_path),
            frozenset({"--robot-config"}),
        )


def test_illegal_task_overrides_fail() -> None:
    with pytest.raises(ValueError, match="非法任务覆盖键"):
        resolve_v12_config(
            _args(task_overrides={"mpc.not_a_real_key": 1}),
            frozenset({"--task-overrides"}),
        )
    with pytest.raises(ValueError, match="任务覆盖值类型"):
        resolve_v12_config(
            _args(task_overrides={"serve.ball_speed": "fast"}),
            frozenset({"--task-overrides"}),
        )


def test_task_overrides_apply_with_cli_highest_precedence() -> None:
    """task_overrides 生效，但专用 CLI 旗标仍最高优先。"""
    r = resolve_v12_config(
        _args(
            serve_box=True,
            horizon=77,
            task_overrides={
                "mpc.total_horizon": 333,
                "mpc.fixed_horizon": 44,
                "cost.Q_p": [1000.0, 2000.0, 3000.0],
                "target.center": [1.0, 2.0, 3.0],
            },
        ),
        frozenset({"--serve-box", "--horizon", "--task-overrides"}),
    )
    assert r.mpc_config["total_horizon"] == 333
    assert r.mpc_config["fixed_horizon"] == 77  # 专用 CLI 旗标优先
    assert list(r.mpc_config["Q_p_base"]) == [2000.0, 4000.0, 6000.0]
    assert np.allclose(r.target_center, [1.0, 2.0, 3.0])
    assert r.sources["mpc.total_horizon"] == "task_overrides"
    assert r.sources["mpc.fixed_horizon"] == "cli:--horizon"


def test_explicit_rm65_profile_keeps_legacy_values() -> None:
    """显式 rm65.yaml 等价旧缺省路径（数值快照不变）。"""
    baseline = _resolve_rm65_snapshot()
    explicit = resolve_v12_config(
        _args(
            robot_config=ROBOTS_DIR / "rm65.yaml",
            serve_box=True,
            ball_speed=7.0,
            seed=42,
            no_plot=True,
        ),
        frozenset({"--robot-config", "--serve-box", "--ball-speed", "--seed", "--no-plot"}),
    )
    assert explicit.robot_id == "rm65"
    assert explicit.robot_profile_path == (ROBOTS_DIR / "rm65.yaml").resolve()
    assert explicit.mpc_config == baseline.mpc_config
    assert explicit.serve_config["ball_speed"] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# 隔离与深拷贝
# ---------------------------------------------------------------------------


def test_resolution_is_isolated_and_deep_copied() -> None:
    r1 = _resolve_rm65_snapshot()
    r1.cost_config["Q_p"][0] = 123.0
    r1.sources["dt"] = "tampered"
    r1.mpc_config["version"] = "tampered"
    r1.solver_config["backend"] = "tampered"
    r2 = _resolve_rm65_snapshot()
    assert r2.cost_config["Q_p"][0] == pytest.approx(50000.0)
    assert r2.sources["dt"] == "default.yaml"
    # dict 容器本身可变，但每个解析结果持有独立副本（不是全局单例）
    assert r2.mpc_config["version"] == "v12"
    assert r2.solver_config["backend"] == "auto"
    with pytest.raises(ValueError):
        r2.mpc_config["Q_p_base"][0] = 1.0  # numpy 数组只读


def test_rm65_and_r1d_resolutions_do_not_cross_contaminate() -> None:
    before = _resolve_rm65_snapshot()
    resolve_v12_config(
        _args(robot_config=ROBOTS_DIR / "r1d_right.yaml"),
        frozenset({"--robot-config"}),
    )
    after = _resolve_rm65_snapshot()
    assert after.mpc_config == before.mpc_config
    assert after.cost_config == before.cost_config
    assert np.allclose(after.target_center, before.target_center)


def test_normalize_explicit_options_accepts_flags_and_dests() -> None:
    normalized = normalize_explicit_options(
        ["--serve-box", "ball_speed", "--Q-tcp-soft", "--no-follow-through"]
    )
    assert normalized == frozenset(
        {"serve_box", "ball_speed", "q_tcp_soft", "no_follow_through"}
    )


def test_normalize_explicit_options_accepts_argparse_actions() -> None:
    """T5 的 _explicit_cli_actions() 返回 Action 列表；归一化必须兼容。"""
    action = argparse.Action(
        option_strings=["--Q-tcp-soft"], dest="q_tcp_soft"
    )
    normalized = normalize_explicit_options([action, "--serve-box"])
    assert normalized == frozenset({"q_tcp_soft", "serve_box"})


# ---------------------------------------------------------------------------
# 复审修正：双源收敛 / 覆盖别名 / 相等性 / 固定球 runtime
# ---------------------------------------------------------------------------


def test_softmin_beta_syncs_tube_single_consumer() -> None:
    """softmin_beta 唯一消费者是 TubeConfig；CLI 后 tube 与 mpc 必须同值。"""
    r = resolve_v12_config(
        _args(serve_box=True, softmin_beta=7.5),
        frozenset({"--serve-box", "--softmin-beta"}),
    )
    assert r.cost_config["tube"]["softmin_beta"] == pytest.approx(7.5)
    assert r.mpc_config["softmin_beta"] == pytest.approx(7.5)
    assert TubeConfig(**r.cost_config["tube"]).softmin_beta == pytest.approx(7.5)
    assert r.sources["cost.tube.softmin_beta"] == "cli:--softmin-beta"
    assert r.sources["mpc.softmin_beta"] == "cli:--softmin-beta"

    overridden = resolve_v12_config(
        _args(serve_box=True, task_overrides={"cost.tube.softmin_beta": 3.0}),
        frozenset({"--serve-box", "--task-overrides"}),
    )
    assert overridden.cost_config["tube"]["softmin_beta"] == pytest.approx(3.0)
    assert overridden.mpc_config["softmin_beta"] == pytest.approx(3.0)
    assert overridden.sources["cost.tube.softmin_beta"] == "task_overrides"

    # 只覆盖 mpc.softmin_beta 时也要收敛到 tube（避免 mpc 独有值被丢弃）
    mpc_only = resolve_v12_config(
        _args(serve_box=True, task_overrides={"mpc.softmin_beta": 4.0}),
        frozenset({"--serve-box", "--task-overrides"}),
    )
    assert mpc_only.cost_config["tube"]["softmin_beta"] == pytest.approx(4.0)
    assert mpc_only.mpc_config["softmin_beta"] == pytest.approx(4.0)
    assert mpc_only.sources["cost.tube.softmin_beta"] == "task_overrides"


def test_serve_bounce_overrides_alias_to_scene() -> None:
    """serve.use_bounce / serve.bounce_restitution 必须落到 scene 真实字段。"""
    r = resolve_v12_config(
        _args(
            task_overrides={
                "serve.use_bounce": False,
                "serve.bounce_restitution": 0.5,
            }
        ),
        frozenset({"--task-overrides"}),
    )
    assert r.scene_config["use_bounce"] is False
    assert r.scene_config["bounce_restitution"] == pytest.approx(0.5)
    # 不得留下永远不生效的 stray serve 键
    assert "use_bounce" not in r.serve_config
    assert "bounce_restitution" not in r.serve_config
    assert r.sources["scene.use_bounce"] == "task_overrides"
    assert r.sources["scene.bounce_restitution"] == "task_overrides"

    with pytest.raises(ValueError, match="任务覆盖值类型"):
        resolve_v12_config(
            _args(task_overrides={"serve.use_bounce": "yes"}),
            frozenset({"--task-overrides"}),
        )


def test_tube_softmin_terminal_override_applied_after_ablation() -> None:
    """cost.tube.use_softmin_terminal 覆盖必须晚于消融推导才不会被吞掉。"""
    r = resolve_v12_config(
        _args(task_overrides={"cost.tube.use_softmin_terminal": False}),
        frozenset({"--task-overrides"}),
    )
    assert r.cost_config["tube"]["use_softmin_terminal"] is False
    assert r.sources["cost.tube.use_softmin_terminal"] == "task_overrides"

    r2 = resolve_v12_config(
        _args(
            no_tube=True,
            task_overrides={"cost.tube.use_softmin_terminal": False},
        ),
        frozenset({"--no-tube", "--task-overrides"}),
    )
    # --no-tube 派生 softmin_only 会启用 softmin，但显式覆盖更高优先
    assert r2.ablation_mode == "softmin_only"
    assert r2.cost_config["tube"]["use_softmin_terminal"] is False
    assert r2.sources["cost.tube.use_softmin_terminal"] == "task_overrides"

    baseline = resolve_v12_config(
        _args(serve_box=True), frozenset({"--serve-box"})
    )
    assert baseline.cost_config["tube"]["use_softmin_terminal"] is True
    assert (
        baseline.sources["cost.tube.use_softmin_terminal"]
        == "派生(ablation_mode)"
    )


def test_fixed_ball_compat_requires_runtime(tmp_path: Path) -> None:
    """固定球兼容模式缺 runtime 必须在 resolve 阶段失败（复审 Q1）。"""
    task_path = _write_compat_task(tmp_path, ROBOTS_DIR / "r1d_left.yaml")
    with pytest.raises(ValueError, match="runtime"):
        resolve_v12_config(
            _args(
                robot_config=ROBOTS_DIR / "r1d_left.yaml",
                sim_task_config=task_path,
            ),
            frozenset({"--robot-config", "--sim-task-config"}),
        )


def test_resolved_config_equality_handles_arrays() -> None:
    """ResolvedV12Config 相等性按字段递归比较，不因 ndarray 抛歧义异常。"""
    first = _resolve_rm65_snapshot()
    second = _resolve_rm65_snapshot()
    assert first == second
    different_seed = resolve_v12_config(
        _args(seed=7), frozenset({"--seed"})
    )
    assert first != different_seed
    assert (first == object()) is False


def test_hitting_config_mirrors_mpc_authoritative_fields() -> None:
    """mpc_config 是击球权威来源；hitting_config 必须同步为同值镜像。"""
    r = resolve_v12_config(
        _args(
            serve_box=True,
            hit_shift=0.2,
            task_overrides={
                "mpc.racket_speed": 2.5,
                "mpc.follow_through_steps": 12,
            },
        ),
        frozenset({"--serve-box", "--hit-shift", "--task-overrides"}),
    )
    assert r.mpc_config["follow_through_length"] == pytest.approx(0.2)
    assert r.mpc_config["racket_speed"] == pytest.approx(2.5)
    assert r.hitting_config["racket_speed"] == pytest.approx(2.5)
    assert r.hitting_config["follow_through_length"] == pytest.approx(0.2)
    assert r.hitting_config["follow_through_steps"] == 12
    assert r.sources["hitting.racket_speed"] == "task_overrides"
    assert r.sources["hitting.follow_through_length"] == "cli:--hit-shift"


def test_serve_box_derived_sources_track_cli_overrides() -> None:
    """--serve-distance 改变 center 时，派生来源标签也要跟着变。"""
    r = resolve_v12_config(
        _args(serve_box=True, serve_distance=6.0),
        frozenset({"--serve-box", "--serve-distance"}),
    )
    assert np.allclose(r.serve_config["serve_box_center"], [0.0, -6.0, 1.2])
    assert "cli:--serve-distance" in r.sources["serve.serve_box_center"]
    assert r.sources["serve.serve_box_halfsize"] == "V12 字面量"

    baseline = resolve_v12_config(
        _args(serve_box=True), frozenset({"--serve-box"})
    )
    assert baseline.sources["serve.serve_box_center"] == "V12 字面量"
