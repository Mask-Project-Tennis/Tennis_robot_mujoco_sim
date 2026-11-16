"""R1D 阶段 3 Task 3.11：确定性击打闭环验收测试。

验收内容（对应施工计划 Task 3.11 与验收报告
docs/experiments/reports/2026-11-16-unitree-stage3.md）：

1. 固定案例契约：``configs/scenarios/r1d_stage3_reachable.yaml`` 冻结
   左臂可达来球 + 固定 seed 42，且案例引用的生产 task 保留 3.11 调参值。
2. 左固定案例闭环：用真实 ``run_r1d_episode`` 驱动生产调用链（解析输入 →
   构建带球执行/规划环境 → MPCController（iLQR/Tube）→ PredictiveSafetyFilter
   → SimComponent 接触判定），断言 active 命中、真实球—拍接触、接触拍速
   > `ACTIVE_HIT_FACE_SPEED_THRESHOLD`（0.3 m/s）、X/U/ball 序列有限且形状正确、
   无禁止机身接触；同 seed/同配置
   连续 3 次逐位一致。
3. 右固定环境短回合：右生产配置真实跑通（环境/求解器/安全链被调用），
   允许未命中，不宣称右侧命中。
4. RM65 seed 42 兼容回归：与 3.0 基线 NPZ 逐键/逐形状/逐 dtype/逐位对照
   （基线不存在时 skip 并注明原因——results/ 为 git-ignored 产物目录）。

本测试不做任何 mock 替身：不使用手写挥拍轨迹，不 teleport 球，不修改
接触判定阈值。active 结论只能来自生产 ``classify_hit_result`` 的物理接触。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src.ilqt.components.sim_component import ACTIVE_HIT_FACE_SPEED_THRESHOLD
from src.robot.robot_profile import load_robot_profile
from src.robot.runtime_parameters import load_runtime_parameters
from src.sim.collision_checks import ForbiddenContactChecker
from src.sim.env_factory import build_r1d_environments
from src.sim.r1d_runner import run_r1d_episode
from src.sim.sim_config import load_sim_task_config

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "configs" / "scenarios" / "r1d_stage3_reachable.yaml"

# 右臂生产配置（固定案例只覆盖左侧；右侧做同等级环境/求解器/短回合验证）
RIGHT_PROFILE_PATH = ROOT / "configs" / "robots" / "r1d_right.yaml"
RIGHT_RUNTIME_PATH = ROOT / "configs" / "robots" / "r1d_sim_runtime_right.yaml"
RIGHT_TASK_PATH = ROOT / "configs" / "r1d_sim_right.yaml"

# RM65 seed 42 兼容回归（3.0 基线与 3.11 重跑产物，均位于 git-ignored results/）
RM65_BASELINE_PATH = ROOT / "results" / "unitree_stage3" / "rm65_seed42.npz"

# 闭环拍速断言直接引用生产主动接触判定常量（单一来源，测试不自带硬编码阈值）


# ---------------------------------------------------------------------------
# 场景/配置读取
# ---------------------------------------------------------------------------


def _load_scenario() -> dict[str, Any]:
    """读取固定案例 scenario，并把相对路径解析为绝对路径。

    Returns:
        含 schema_version / profile_path / runtime_path / task_path / seed /
        expected_hit_type 的字典；路径为相对 scenario 文件解析后的 Path。

    Raises:
        AssertionError: 字段缺失或路径不存在（固定案例必须是可执行的冻结输入）。
    """
    assert SCENARIO_PATH.is_file(), f"固定案例缺失: {SCENARIO_PATH}"
    raw = yaml.safe_load(SCENARIO_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), "scenario 应为 YAML 映射"
    expected_keys = {
        "schema_version",
        "profile_path",
        "runtime_path",
        "task_path",
        "seed",
        "expected_hit_type",
    }
    assert set(raw) == expected_keys, (
        f"scenario 字段必须恰为 6 键（不多不少），实际: {sorted(raw)}"
    )
    scenario = {
        "schema_version": int(raw["schema_version"]),
        "profile_path": (SCENARIO_PATH.parent / raw["profile_path"]).resolve(),
        "runtime_path": (SCENARIO_PATH.parent / raw["runtime_path"]).resolve(),
        "task_path": (SCENARIO_PATH.parent / raw["task_path"]).resolve(),
        "seed": int(raw["seed"]),
        "expected_hit_type": str(raw["expected_hit_type"]),
    }
    for key in ("profile_path", "runtime_path", "task_path"):
        assert scenario[key].is_file(), f"scenario 引用的 {key} 不存在: {scenario[key]}"
    return scenario


def _run_episode(
    profile_path: Path,
    runtime_path: Path,
    task_path: Path,
    output_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """用生产入口跑一个回合，并读回 NPZ 轨迹数组。

    Args:
        profile_path: 机器人描述 YAML 路径。
        runtime_path: 运行参数 YAML 路径。
        task_path: 仿真任务 YAML 路径。
        output_dir: 产物目录（JSON + NPZ）。
        seed: 固定随机种子。

    Returns:
        (回合摘要 dict, {键: 轨迹数组 dict})。
    """
    summary = run_r1d_episode(
        profile_path, runtime_path, task_path, output_dir, seed=seed
    )
    json_path = output_dir / "r1d_episode.json"
    npz_path = output_dir / "r1d_episode.npz"
    assert json_path.is_file(), "缺少产物 r1d_episode.json"
    assert npz_path.is_file(), "缺少产物 r1d_episode.npz"
    with json_path.open("r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["schema"] == "r1d_sim_episode_v1"
    assert on_disk["hit"]["type"] == summary["hit"]["type"]
    with np.load(npz_path) as data:
        arrays = {
            key: np.asarray(data[key])
            for key in ("X", "U", "ball_pos", "ball_vel", "t")
        }
    return summary, arrays


def _assert_arrays_valid(summary: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    """断言轨迹数组形状/有限性正确（与 nq=14, nu=7 契约一致）。

    Args:
        summary: 回合摘要（用于读取 total_steps）。
        arrays: NPZ 数组字典。

    Raises:
        AssertionError: 形状/有限性不符合契约。
    """
    total_steps = int(summary["episode"]["total_steps"])
    assert total_steps > 0, "回合未执行任何控制步"
    X = arrays["X"]
    U = arrays["U"]
    ball_pos = arrays["ball_pos"]
    ball_vel = arrays["ball_vel"]
    t = arrays["t"]
    assert X.ndim == 2 and X.shape == (total_steps + 1, 14), (
        f"X 形状应为 (N+1, 14)，实际 {X.shape}"
    )
    assert U.ndim == 2 and U.shape == (total_steps, 7), (
        f"U 形状应为 (N, 7)，实际 {U.shape}"
    )
    assert ball_pos.shape == (total_steps + 1, 3), f"ball_pos 形状异常: {ball_pos.shape}"
    assert ball_vel.shape == (total_steps + 1, 3), f"ball_vel 形状异常: {ball_vel.shape}"
    assert t.shape == (total_steps + 1,), f"t 形状异常: {t.shape}"
    for name, arr in (
        ("X", X),
        ("U", U),
        ("ball_pos", ball_pos),
        ("ball_vel", ball_vel),
        ("t", t),
    ):
        assert bool(np.all(np.isfinite(arr))), f"{name} 含非有限值"
    assert float(summary["dt"]) == pytest.approx(0.005, abs=1e-12)


def _assert_no_forbidden_contact(
    profile_path: Path,
    runtime_path: Path,
    task_path: Path,
    npz_path: Path,
) -> int:
    """用生产场景的 ForbiddenContactChecker 逐帧检查禁止机身接触。

    检查器在独立 MjData 上按臂关节角重建位形（live 环境不受影响），
    被查对象是回合产物 NPZ 的 X 逐帧主动臂 7 维关节角。

    Args:
        profile_path: 机器人描述 YAML 路径。
        runtime_path: 运行参数 YAML 路径。
        task_path: 仿真任务 YAML 路径。
        npz_path: 回合 NPZ 产物路径。

    Returns:
        禁止接触对数量（供断言场景保护配置非空）。

    Raises:
        AssertionError: 任一帧出现禁止对接触，或保护对为空。
    """
    profile = load_robot_profile(profile_path)
    parameters = load_runtime_parameters(runtime_path, profile)
    task = load_sim_task_config(task_path, profile)
    envs = build_r1d_environments(profile, parameters, task)
    execution = envs.execution
    scene = execution.scene
    n_arm = int(execution.NQ)
    assert len(scene.protected_geom_pairs) > 0, "保护对为空不得冒充机身保护"
    checker = ForbiddenContactChecker.from_env(execution, scene.protected_geom_pairs)
    with np.load(npz_path) as data:
        X = np.asarray(data["X"], dtype=np.float64)
    bad_frames = [
        step
        for step in range(X.shape[0])
        if checker.has_contact_at_q(np.asarray(X[step, :n_arm], dtype=np.float64))
    ]
    assert not bad_frames, f"出现禁止机身接触的帧: {bad_frames[:5]}"
    return len(scene.protected_geom_pairs)


# ---------------------------------------------------------------------------
# 1. 固定案例契约
# ---------------------------------------------------------------------------


def test_fixed_scenario_contract() -> None:
    """固定案例冻结正确：schema/seed/期望命中类型/生产配置调参值。

    Raises:
        AssertionError: 案例字段或生产 task 配置与 3.11 冻结值不一致。
    """
    scenario = _load_scenario()
    raw = yaml.safe_load(SCENARIO_PATH.read_text(encoding="utf-8"))
    assert set(raw) == {
        "schema_version",
        "profile_path",
        "runtime_path",
        "task_path",
        "seed",
        "expected_hit_type",
    }, f"固定案例字段必须恰为 6 键（不多不少），实际: {sorted(raw)}"
    assert scenario["schema_version"] == 1
    assert scenario["seed"] == 42, "固定案例必须固定 seed=42（不扫描/不挑选）"
    assert scenario["expected_hit_type"] == "active"

    task_raw = yaml.safe_load(scenario["task_path"].read_text(encoding="utf-8"))
    mpc_cfg = task_raw["mpc_config"]
    cost_cfg = task_raw["cost_config"]
    # 冻结 3.11 调参结果（详见 task YAML 文件头调参记录与验收报告 §调参）
    assert int(mpc_cfg["total_horizon"]) == 200, "3.11 调参：total_horizon=200"
    assert int(mpc_cfg["follow_through_steps"]) == 20, "3.11 调参：随挥 20 步"
    assert float(cost_cfg["Q_p"]) == 50000.0, "3.11 调参：Q_p=50000"
    assert float(cost_cfg["Q_v"]) == 200.0, "3.11 调参：Q_v=200"

    # 来球几何与固定重力抛物线一致：t=0.45s 到达 [0.25, 0.315, 1.357] m
    p0 = np.asarray(task_raw["initial_ball_position"], dtype=np.float64)
    v0 = np.asarray(task_raw["initial_ball_velocity"], dtype=np.float64)
    center = np.asarray(task_raw["workspace_center"], dtype=np.float64)
    radius = float(task_raw["workspace_radius"])
    t_hit = 0.45
    p_analytic = np.array(
        [
            p0[0] + v0[0] * t_hit,
            p0[1] + v0[1] * t_hit,
            p0[2] + v0[2] * t_hit - 0.5 * 9.81 * t_hit * t_hit,
        ]
    )
    assert np.allclose(p_analytic, [0.25, 0.315, 1.357], atol=1e-3), (
        f"固定案例来球在 t=0.45s 应到达 [0.25,0.315,1.357]±1mm，实际 {p_analytic}"
    )
    assert float(np.linalg.norm(p_analytic - center)) < radius, (
        "固定案例的击球点必须落在 task 声明的击球工作区内"
    )
    # 肩—击球点距离 < R1D 臂展 0.35m（阶段 2 FK 实测左肩 [0.059,0.086,1.358]）
    shoulder = np.array([0.059, 0.086, 1.358])
    assert float(np.linalg.norm(p_analytic - shoulder)) < 0.35


# ---------------------------------------------------------------------------
# 2. 左固定案例：active 命中 + 连续 3 次逐位一致
# ---------------------------------------------------------------------------


def test_left_fixed_case_active_hit_and_deterministic(tmp_path: Path) -> None:
    """左固定案例闭环：真实调用链 active 命中，且 3 次逐位一致。

    链路证据：``backend.effective == "python-fd"`` 说明规划由真实 FD
    iLQR 求解器产生；``hit.step == 90 > replan_interval(20)`` 说明命中前
    至少发生过多次重规划（预测—重规划链被真实调用）；``active_contact``
    只能由接触窗口内的真实球—拍 geom 接触（dist<=0）置位。

    Args:
        tmp_path: pytest 提供的临时产物目录（不提交任何结果文件）。
    """
    scenario = _load_scenario()
    runs = []
    for index in range(3):
        output_dir = tmp_path / f"left_run{index + 1}"
        summary, arrays = _run_episode(
            scenario["profile_path"],
            scenario["runtime_path"],
            scenario["task_path"],
            output_dir,
            scenario["seed"],
        )
        _assert_arrays_valid(summary, arrays)
        assert summary["side"] == "left"
        assert summary["robot_id"] == "unitree_r1d"
        assert summary["simulation_only"] is True
        assert summary["backend"]["effective"] == "python-fd", (
            "固定案例必须走真实 Python FD iLQR 求解器（非 mock/非 C++ 兼容路径）"
        )
        assert summary["seed"] == scenario["seed"]

        hit = summary["hit"]
        metrics = summary["metrics"]
        episode = summary["episode"]
        assert hit["type"] == scenario["expected_hit_type"], (
            f"固定案例应命中 {scenario['expected_hit_type']}，实际 {hit}"
        )
        assert metrics["active_contact"] is True, "active 命中必须来自真实球—拍接触"
        assert metrics["passive_contact"] is False
        assert 0 <= int(hit["step"]) < int(episode["total_steps"])
        assert float(hit["racket_face_speed"]) > ACTIVE_HIT_FACE_SPEED_THRESHOLD, (
            "接触瞬间拍速 "
            f"{hit['racket_face_speed']} 未超过生产主动接触阈值 "
            f"{ACTIVE_HIT_FACE_SPEED_THRESHOLD}"
        )
        assert np.isfinite(float(hit["racket_face_speed"]))
        assert epilogue_ok(episode), (
            f"回合未正常收尾: failure={episode['failure_reason']}, "
            f"mpc_done={episode['mpc_done']}"
        )
        # 重规划链证据：命中步远大于重规划间隔，说明预测—重规划被多次调用
        assert int(episode["total_steps"]) > int(
            yaml.safe_load(scenario["task_path"].read_text(encoding="utf-8"))[
                "mpc_config"
            ]["replan_interval"]
        )

        # 禁止机身接触：用生产场景检查器逐帧复核（真实检查，非形状断言）
        pairs = _assert_no_forbidden_contact(
            scenario["profile_path"],
            scenario["runtime_path"],
            scenario["task_path"],
            output_dir / "r1d_episode.npz",
        )
        assert pairs == 8, f"左生产场景保护对应为 8，实际 {pairs}"
        runs.append((summary, arrays))

    # 同一固定案例 + 同 seed 的确定性：轨迹逐位一致、命中结论一致
    key_fields = ("type", "step", "racket_face_speed", "p_ee_at_hit", "ball_pos_at_hit")
    for summary, _arrays in runs[1:]:
        for field in key_fields:
            assert summary["hit"][field] == runs[0][0]["hit"][field], (
                f"多次重跑 hit.{field} 不一致（非确定性）"
            )
        assert summary["metrics"] == runs[0][0]["metrics"]
        assert summary["episode"]["total_steps"] == runs[0][0]["episode"]["total_steps"]
    for key in ("X", "U", "ball_pos", "ball_vel", "t"):
        reference = runs[0][1][key]
        for _summary, arrays in runs[1:]:
            assert np.array_equal(reference, arrays[key]), (
                f"多次重跑轨迹数组 {key} 未逐位一致（非确定性）"
            )


def epilogue_ok(episode: dict[str, Any]) -> bool:
    """判断回合是否按计划收尾（未命中安全停止/未提前失败）。

    Args:
        episode: 回合摘要的 ``episode`` 段。

    Returns:
        True 表示 mpc_done 且 failure_reason 为空。
    """
    return bool(episode["mpc_done"]) and episode["failure_reason"] is None


# ---------------------------------------------------------------------------
# 3. 右侧生产配置：同等级环境/求解器/短回合（允许 miss，不宣称命中）
# ---------------------------------------------------------------------------


def test_right_production_short_episode(tmp_path: Path) -> None:
    """右生产配置真实跑通：环境构建、iLQR 规划、安全滤波、接触判定均被调用。

    右侧不宣称命中：右臂肩部侧摆限位较窄（3.10 已记录镜像击球点 refiner
    的 j1 裕度为负），本测试只要求链路真实运行并产出合法产物（允许 miss /
    safety_filter_stop）。

    Args:
        tmp_path: pytest 提供的临时产物目录。
    """
    assert RIGHT_PROFILE_PATH.is_file()
    assert RIGHT_RUNTIME_PATH.is_file()
    assert RIGHT_TASK_PATH.is_file()
    output_dir = tmp_path / "right_run"
    summary, arrays = _run_episode(
        RIGHT_PROFILE_PATH,
        RIGHT_RUNTIME_PATH,
        RIGHT_TASK_PATH,
        output_dir,
        seed=42,
    )
    _assert_arrays_valid(summary, arrays)
    assert summary["side"] == "right"
    assert summary["backend"]["effective"] == "python-fd"
    assert int(summary["episode"]["total_steps"]) > 0
    assert int(summary["metrics"]["n_mpc_steps"]) > 0
    assert summary["hit"]["type"] in ("active", "passive", "miss")
    # 右侧链路的合法终止原因（不允许异常/NaN 逃逸）
    assert summary["episode"]["failure_reason"] in (
        None,
        "safety_filter_stop",
        "ball_unreachable",
    ), f"右侧回合意外终止原因: {summary['episode']['failure_reason']}"
    _assert_no_forbidden_contact(
        RIGHT_PROFILE_PATH,
        RIGHT_RUNTIME_PATH,
        RIGHT_TASK_PATH,
        output_dir / "r1d_episode.npz",
    )


# ---------------------------------------------------------------------------
# 4. RM65 seed 42 兼容回归（基线在 results/，git-ignored → 缺失即 skip）
# ---------------------------------------------------------------------------


def _compare_npz_bitwise(baseline_path: Path, candidate_path: Path) -> None:
    """逐键/逐形状/逐 dtype/逐位对照两个 NPZ（不做 zip 文件级 SHA 比较）。

    Args:
        baseline_path: 3.0 基线 NPZ 路径。
        candidate_path: 3.11 重跑 NPZ 路径。

    Raises:
        AssertionError: 键集合、形状、dtype 或数值不一致（附首个差异）。
    """
    with np.load(baseline_path) as base, np.load(candidate_path) as cand:
        assert set(base.files) == set(cand.files), (
            f"NPZ 键集合漂移: 仅基线={sorted(set(base.files) - set(cand.files))}, "
            f"仅新跑={sorted(set(cand.files) - set(base.files))}"
        )
        for key in sorted(base.files):
            a = np.asarray(base[key])
            b = np.asarray(cand[key])
            assert a.shape == b.shape, f"{key}: 形状 {a.shape} != {b.shape}"
            assert a.dtype == b.dtype, f"{key}: dtype {a.dtype} != {b.dtype}"
            if a.dtype.kind in "fc":
                assert np.array_equal(a, b), (
                    f"{key}: 数值未逐位一致，max|Δ|="
                    f"{float(np.max(np.abs(a.astype(np.float64) - b)))!r}"
                )
            else:
                assert np.array_equal(a, b), f"{key}: 值不一致"


def test_rm65_seed42_regression(tmp_path: Path) -> None:
    """RM65 seed 42 与 3.0 基线逐位一致（默认权重/模型/时间步无授权变化）。

    results/ 为 git-ignored 产物目录：基线缺失时跳过并注明原因，而不是让
    测试静默通过。

    Args:
        tmp_path: pytest 提供的临时产物目录。
    """
    if not RM65_BASELINE_PATH.is_file():
        pytest.skip(
            "RM65 3.0 基线缺失（results/ 为 git-ignored 产物目录）: "
            f"{RM65_BASELINE_PATH}；本机存在基线时才执行逐字段回归"
        )
    candidate_path = tmp_path / "rm65_seed42_stage3.npz"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "rm65_mpc_v12.py"),
        "--serve-box",
        "--ball-speed",
        "7",
        "--seed",
        "42",
        "--no-plot",
        "--dump-trajectory",
        str(candidate_path),
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert completed.returncode == 0, (
        f"RM65 seed42 重跑失败（rc={completed.returncode}）:\n"
        f"stdout 尾部:\n{completed.stdout[-2000:]}\n"
        f"stderr 尾部:\n{completed.stderr[-2000:]}"
    )
    assert candidate_path.is_file(), "RM65 重跑未产出轨迹 NPZ"
    _compare_npz_bitwise(RM65_BASELINE_PATH, candidate_path)
