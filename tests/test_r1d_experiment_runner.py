"""T8a 运行层测试：manifest / 身份续跑 / timeout / dry-run / exp28 延迟 / 图资产命令。

覆盖矩阵要求的运行层改动点（`docs/experiments/design/r1d_v12_parity_matrix.md`）：

- batch_runner：身份列、resume 身份隔离、旧 CSV 兼容、timeout 行保留、
  data-root/output 贯穿、dry-run 不执行；
- overnight_experiments：exp28 延迟读取当前 data-root、数据缺失显式报错；
- collect_fig_assets：R1D 命令构造与目录注入、RM65 旧默认不变；
- run_r1d_paper_smoke + manifest：19 组覆盖、每比较 cell ≥2 共同 seed、
  override 白名单、R1D 共享 CLI 面、resolved 产物。

全部测试使用临时目录与哑脚本，不跑长 MuJoCo 仿真。
"""

from __future__ import annotations

import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exp import batch_runner as br  # noqa: E402
from scripts.exp import collect_fig_assets as fig_assets  # noqa: E402
from scripts.exp import overnight_experiments as overnight  # noqa: E402
from scripts.exp import run_r1d_paper_smoke as smoke  # noqa: E402

EXPECTED_EXPERIMENTS = {
    "exp15_speed_v2",
    "exp16_limits_v2",
    "exp17b_perturb",
    "exp17d_mechanism",
    "exp17e_perturb7",
    "exp17f_mechanism_perturb",
    "exp17g_spatial_power",
    "exp17h_extreme",
    "exp17i_limits_ablation",
    "exp18_sensitivity",
    "exp19_hardmin",
    "exp21_hardmin_tcp10",
    "exp22_shared_spatial",
    "exp23_nofreeze",
    "exp24_shared_full",
    "exp25_select_one",
    "exp26_tcp_grid",
    "exp27_ball_quality",
    "exp28_always_collision",
}

RESULT_LINE = "__RESULT__: hit_type=active pos_error=0.010000 min_dist=0.001000\n"


def _cli_env() -> dict[str, str]:
    """子进程固定 UTF-8 环境（N13）：避免父进程未设 PYTHONUTF8 时 GBK 解码失败。"""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# ---------------------------------------------------------------------------
# 测试夹具：临时项目 + 哑仿真脚本
# ---------------------------------------------------------------------------


def _write_dummy(tmp_path: Path, body: str | None = None) -> str:
    """在临时项目根写入哑仿真脚本，返回相对脚本路径。"""
    (tmp_path / "dummy_sim.py").write_text(
        body or f"print({RESULT_LINE!r})\n", encoding="utf-8")
    return "dummy_sim.py"


def _identity(**overrides: str) -> br.RunIdentity:
    """构造 R1D 测试身份（config_hash 固定为 16 hex，便于断言）。"""
    base = {
        "robot_id": "unitree_r1d",
        "mode": "",
        "racket_id": "blade100_v9_reference",
        "config_hash": "0123456789abcdef",
        "metric_schema": "v12_episode_v2_physical_hit",
    }
    base.update(overrides)
    return br.RunIdentity(**base)


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    """读取 results.csv 全部行。"""
    with csv_path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def manifest() -> dict:
    """真实 smoke manifest（模块级只读，测试内用 deepcopy 变异）。"""
    return smoke.load_manifest(smoke.DEFAULT_MANIFEST_PATH)


# ---------------------------------------------------------------------------
# manifest 校验
# ---------------------------------------------------------------------------


def test_manifest_valid_and_covers_nineteen_experiments(manifest: dict) -> None:
    """manifest 合法，且覆盖矩阵 19 组正文实验。"""
    smoke.validate_manifest(manifest)
    ids = [str(exp["id"]) for exp in manifest["experiments"]]
    assert len(ids) == 19
    assert set(ids) == EXPECTED_EXPERIMENTS
    assert len(ids) == len(set(ids))
    # 额外依赖三项必须存在（矩阵要求）
    dep_ids = {str(dep["id"]) for dep in manifest["extra_dependencies"]}
    assert dep_ids == {
        "exp18_fig_assets",
        "exp18_tcp_exempt_window_stats",
        "exp17a_noise_none_control",
    }
    # 每条实验都有论文引用/矩阵入口/T8b 消费者
    for exp in manifest["experiments"]:
        assert exp.get("paper_ref")
        assert exp.get("matrix_ref")
        assert exp.get("t8b_consumer")


def test_manifest_rejects_unknown_override_key(manifest: dict) -> None:
    """task_overrides 键必须在 V12 overlay 白名单内。"""
    mutated = copy.deepcopy(manifest)
    mutated["experiments"][0]["cases"][0]["task_overrides"] = {"nope.key": 1}
    with pytest.raises(smoke.ManifestError, match="白名单"):
        smoke.validate_manifest(mutated)


def test_manifest_rejects_rm65_only_argv(manifest: dict) -> None:
    """RM65 专属开关（如 --always-collision）不得出现在 R1D case argv。"""
    mutated = copy.deepcopy(manifest)
    mutated["experiments"][0]["cases"][0]["argv"] = {"--always-collision": None}
    with pytest.raises(smoke.ManifestError, match="RM65 专属"):
        smoke.validate_manifest(mutated)


def test_manifest_rejects_blocked_without_reason(manifest: dict) -> None:
    """blocked case 必须给出 blocked_reason，禁止静默缺条件。"""
    mutated = copy.deepcopy(manifest)
    case = mutated["experiments"][0]["cases"][0]
    case["status"] = "blocked"
    case.pop("blocked_reason", None)
    with pytest.raises(smoke.ManifestError, match="blocked_reason"):
        smoke.validate_manifest(mutated)


def test_manifest_rejects_sim_task_config(manifest: dict) -> None:
    """R1D overlay 模式不允许显式 sim-task-config（会切到固定球兼容模式）。"""
    mutated = copy.deepcopy(manifest)
    mutated["configs"]["sim_task_config"] = "configs/scenarios/r1d_stage3_reachable.yaml"
    with pytest.raises(smoke.ManifestError, match="sim_task_config"):
        smoke.validate_manifest(mutated)


def test_manifest_rejects_unknown_dependency(manifest: dict) -> None:
    """depends_on 引用未知 id 必须报错。"""
    mutated = copy.deepcopy(manifest)
    mutated["experiments"][0]["depends_on"] = ["exp_not_exist"]
    with pytest.raises(smoke.ManifestError, match="未知 id"):
        smoke.validate_manifest(mutated)


def test_manifest_rejects_unknown_derived_from(manifest: dict) -> None:
    """derived 实验的 derived_from 必须指向已知实验。"""
    mutated = copy.deepcopy(manifest)
    for exp in mutated["experiments"]:
        if exp["id"] == "exp28_always_collision":
            exp["derived_from"] = "exp_not_exist"
    with pytest.raises(smoke.ManifestError, match="derived_from"):
        smoke.validate_manifest(mutated)


def test_manifest_rejects_single_seed(manifest: dict, tmp_path: Path) -> None:
    """小样本规则：每比较 cell ≥2 共同 seed，单 seed 选择必须报错。"""
    with pytest.raises(smoke.ManifestError, match="共同 seed"):
        smoke.resolve_manifest(manifest, output=tmp_path / "run", seeds=[100])


def test_manifest_missing_frozen_config(manifest: dict, tmp_path: Path) -> None:
    """T7 冻结配置缺失时 resolve 必须报错（禁止伪造 config_hash）。"""
    mutated = copy.deepcopy(manifest)
    mutated["configs"]["robot"] = "configs/robots/not_frozen_yet.yaml"
    with pytest.raises(smoke.ManifestError, match="冻结配置文件缺失"):
        smoke.resolve_manifest(mutated, output=tmp_path / "run")


# ---------------------------------------------------------------------------
# resolved 清单：身份 / 任务数 / 每 cell 双 seed
# ---------------------------------------------------------------------------


def test_resolve_manifest_identity_and_task_counts(
    manifest: dict, tmp_path: Path,
) -> None:
    """resolved 清单含 config_hash/身份，任务数符合 manifest 条件矩阵。"""
    resolved = smoke.resolve_manifest(manifest, output=tmp_path / "run")
    assert resolved["schema"] == smoke.RESOLVED_SCHEMA
    assert resolved["config_hash"] == smoke._config_hash(manifest)[0]
    assert len(resolved["config_hash"]) == 16
    assert resolved["identity"]["robot_id"] == "unitree_r1d"
    assert resolved["identity"]["racket_id"] == "blade100_v9_reference"
    assert resolved["identity"]["mode"] == ""
    assert resolved["seeds"] == [100, 101]
    # 默认 torque 过滤：124 条；both 时 134 条（19 组实验 132 + exp17a 2）
    assert resolved["counts"]["batch_tasks"] == 126
    assert resolved["counts"]["torque_tasks"] == 126
    assert resolved["counts"]["position_tasks"] == 0
    assert resolved["counts"]["experiments"] == 19
    assert resolved["counts"]["blocked_cases"] == 1
    assert resolved["counts"]["derived_units"] == 1
    assert resolved["counts"]["tool_units"] == 2
    both = smoke.resolve_manifest(manifest, output=tmp_path / "run", mode="both")
    assert both["counts"]["batch_tasks"] == 136
    assert both["counts"]["position_tasks"] == 10
    position_only = smoke.resolve_manifest(
        manifest, output=tmp_path / "run", mode="position")
    assert position_only["counts"]["batch_tasks"] == 10
    filtered = {u["id"]: u["status"] for u in position_only["units"]
                if u["kind"] == "batch"}
    assert filtered["exp17b_perturb"] == "filtered"


def test_resolved_tasks_have_two_common_seeds_per_case(
    manifest: dict, tmp_path: Path,
) -> None:
    """每个 ready/adapted case 在 resolved 中都恰好有 2 条任务（seed 100/101）。"""
    resolved = smoke.resolve_manifest(
        manifest, output=tmp_path / "run", mode="both")
    grouped: dict[tuple[str, str], list[int]] = {}
    for task in resolved["tasks"]:
        grouped.setdefault(
            (task["experiment"], task["case_id"]), []).append(task["seed"])
    assert grouped
    for key, seeds in grouped.items():
        assert seeds == [100, 101], f"{key} 共享 seed 不是 [100,101]: {seeds}"
        assert len(seeds) == 2


def test_resolved_tasks_carry_config_hash_identity(
    manifest: dict, tmp_path: Path,
) -> None:
    """每条任务的身份都带运行时 config_hash 与正确 mode。"""
    resolved = smoke.resolve_manifest(
        manifest, output=tmp_path / "run", mode="both")
    for task in resolved["tasks"]:
        assert len(task["identity"]["config_hash"]) == 16
        assert task["identity"]["robot_id"] == "unitree_r1d"
        assert task["identity"]["mode"] == task["mode"]
        assert task["command"][0] == sys.executable
        assert "--output" in task["argv"]
        assert str(task["argv"]["--output"]).startswith(str(tmp_path / "run" / "artifacts"))


def test_build_specs_from_manifest(
    manifest: dict, tmp_path: Path,
) -> None:
    """spec 工厂：每个有任务的 batch 单元一个 spec，grid 汇总任务数一致。"""
    specs = smoke.build_specs_from_manifest(
        smoke.DEFAULT_MANIFEST_PATH, output=tmp_path / "run", mode="both")
    assert len(specs) == 18  # 17 组有任务的 batch 实验 + exp17a（blocked-only exp25 与 derived exp28 除外）
    names = {spec.name for spec in specs}
    assert "exp24_shared_full" in names
    assert "exp17a_noise_none_control" in names
    assert "exp28_always_collision" not in names
    assert "exp25_select_one" not in names
    assert sum(len(spec.grid) for spec in specs) == 136
    for spec in specs:
        assert spec.identity.robot_id == "unitree_r1d"
        assert len(spec.identity.config_hash) == 16
        for params in spec.grid:
            assert params["--seed"] in (100, 101)
            assert "_timeout" in params
            assert "--robot-config" in params
    with pytest.raises(ValueError, match="workers=1"):
        smoke.build_specs_from_manifest(
            smoke.DEFAULT_MANIFEST_PATH, output=tmp_path / "run", workers=2)


def test_main_dry_run_writes_resolved_only(tmp_path: Path) -> None:
    """CLI dry-run：写 manifest_resolved.json、打印任务、不建 artifacts、退出码 0。"""
    run_root = tmp_path / "run"
    code = smoke.main([
        "--manifest", str(smoke.DEFAULT_MANIFEST_PATH),
        "--output", str(run_root), "--dry-run",
    ])
    assert code == 0
    resolved = json.loads((run_root / "manifest_resolved.json").read_text(encoding="utf-8"))
    assert resolved["schema"] == smoke.RESOLVED_SCHEMA
    assert resolved["counts"]["batch_tasks"] == 126
    assert len(resolved["tasks"]) == 126
    assert not (run_root / "artifacts").exists()
    assert not (run_root / "runs.csv").exists()


def test_main_rejects_parallel_workers(tmp_path: Path) -> None:
    """R1D 论文 smoke 强制 workers=1（多进程 MuJoCo 风险），>1 退出码 2。"""
    with pytest.raises(SystemExit) as exc:
        smoke.main([
            "--manifest", str(smoke.DEFAULT_MANIFEST_PATH),
            "--output", str(tmp_path / "run"), "--workers", "2", "--dry-run",
        ])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# batch_runner：身份 / resume / timeout / data-root / dry-run
# ---------------------------------------------------------------------------


def test_resume_skips_completed_and_isolates_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同 CSV 续跑跳过已完成；config_hash 变化后禁止混写并报错。"""
    monkeypatch.setattr(br, "PROJECT", tmp_path)
    _write_dummy(tmp_path)
    out_dir = tmp_path / "batch" / "exp_ident"
    grid = [{"--seed": 100, "--ball-speed": 7},
            {"--seed": 101, "--ball-speed": 7}]
    spec = br.ExperimentSpec(
        name="exp_ident", script="dummy_sim.py", grid=grid, identity=_identity())
    first = br.run_batch(spec, workers=1, output=out_dir)
    assert first["launched"] == 2
    assert first["errors"] == 0
    assert first["hits_this_run"] == 2
    rows = _read_rows(out_dir / "results.csv")
    assert len(rows) == 2
    assert {row["robot_id"] for row in rows} == {"unitree_r1d"}
    assert {row["config_hash"] for row in rows} == {"0123456789abcdef"}
    assert {row["mode"] for row in rows} == {"torque"}
    assert br.CSV_FIELDS[:len(br.LEGACY_CSV_FIELDS)] == br.LEGACY_CSV_FIELDS

    second = br.run_batch(spec, workers=1, output=out_dir)
    assert second["done_before"] == 2
    assert second["launched"] == 0

    other = br.ExperimentSpec(
        name="exp_ident", script="dummy_sim.py", grid=grid,
        identity=_identity(config_hash="ffffffffffffffff"))
    with pytest.raises(ValueError, match="不同身份"):
        br.run_batch(other, workers=1, output=out_dir)


def test_legacy_csv_resume_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧 21 列 CSV：legacy 身份默认可续跑；新身份需 --legacy-resume。"""
    monkeypatch.setattr(br, "PROJECT", tmp_path)
    _write_dummy(tmp_path)
    out_dir = tmp_path / "batch" / "exp_legacy"
    out_dir.mkdir(parents=True)
    params = {"--seed": 100, "--ball-speed": 7}
    row = {key: "" for key in br.LEGACY_CSV_FIELDS}
    row.update({
        "experiment": "exp_legacy",
        "config_id": br.make_config_id(params),
        "seed": "100",
        "error": "",
        "hit_type": "active",
        "hit": "True",
    })
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=br.LEGACY_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    legacy_spec = br.ExperimentSpec(
        name="exp_legacy", script="dummy_sim.py", grid=[params],
        identity=br.LEGACY_IDENTITY)
    summary = br.run_batch(legacy_spec, workers=1, output=out_dir)
    assert summary["done_before"] == 1
    assert summary["launched"] == 0

    r1d_spec = br.ExperimentSpec(
        name="exp_legacy", script="dummy_sim.py", grid=[params], identity=_identity())
    with pytest.raises(ValueError, match="旧格式"):
        br.run_batch(r1d_spec, workers=1, output=out_dir)
    summary = br.run_batch(
        r1d_spec, workers=1, output=out_dir, allow_legacy_resume=True)
    assert summary["done_before"] == 1


def test_timeout_row_recorded_and_not_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """超时行写入 timed_out=1/error=timeout，且 resume 视为已完成不重跑。"""
    monkeypatch.setattr(br, "PROJECT", tmp_path)
    _write_dummy(tmp_path, "import time\ntime.sleep(5)\n")
    out_dir = tmp_path / "batch" / "exp_timeout"
    spec = br.ExperimentSpec(
        name="exp_timeout", script="dummy_sim.py",
        grid=[{"--seed": 100, "--ball-speed": 7, "_timeout": 0.2}],
        identity=_identity())
    summary = br.run_batch(spec, workers=1, output=out_dir, timeout_s=0.2)
    assert summary["launched"] == 1
    assert summary["timeouts"] == 1
    rows = _read_rows(out_dir / "results.csv")
    assert len(rows) == 1
    assert rows[0]["error"] == "timeout"
    assert rows[0]["timed_out"] == "1"
    assert rows[0]["exit_code"] == "-1"

    again = br.run_batch(spec, workers=1, output=out_dir)
    assert again["done_before"] == 1
    assert again["launched"] == 0


def test_data_root_and_output_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """data_root 与 output 贯穿 CSV 路径（不写死 experiment_data）。"""
    monkeypatch.setattr(br, "PROJECT", tmp_path)
    _write_dummy(tmp_path)
    grid = [{"--seed": 100, "--ball-speed": 7}]
    spec = br.ExperimentSpec(
        name="exp_route", script="dummy_sim.py", grid=grid, identity=_identity())
    by_root = br.run_batch(spec, workers=1, data_root=tmp_path / "root")
    root_csv = (br.PROJECT / by_root["csv"]).resolve()
    assert root_csv == (tmp_path / "root" / "exp_route" / "results.csv").resolve()
    assert root_csv.is_file()
    # 已完成后换 output：新目录重新跑而不是续跑旧根
    by_output = br.run_batch(spec, workers=1, output=tmp_path / "other")
    output_csv = (br.PROJECT / by_output["csv"]).resolve()
    assert output_csv == (tmp_path / "other" / "results.csv").resolve()
    assert by_output["launched"] == 1


def test_dry_run_does_not_execute_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dry-run 只返回计划任务：不调用 run_one、不建目录、不写 CSV。"""
    monkeypatch.setattr(br, "PROJECT", tmp_path)
    _write_dummy(tmp_path)

    def _boom(*_args, **_kwargs):  # pragma: no cover - 被调用即失败
        raise AssertionError("dry-run 不应执行子进程")

    monkeypatch.setattr(br, "run_one", _boom)
    out_dir = tmp_path / "dry"
    spec = br.ExperimentSpec(
        name="exp_dry", script="dummy_sim.py",
        grid=[{"--seed": 100, "--ball-speed": 7},
              {"--seed": 101, "--ball-speed": 7}],
        identity=_identity())
    summary = br.run_batch(spec, workers=1, dry_run=True, output=out_dir)
    assert summary["dry_run"] is True
    assert summary["tasks"]
    assert len(summary["tasks"]) == 2
    assert summary["tasks"][0]["command"][0] == sys.executable
    assert not out_dir.exists()
    assert not (out_dir / "results.csv").exists()


def test_config_id_skips_seed_and_output() -> None:
    """config_id 只由配置身份决定：seed/output/meta 不参与。"""
    a = br.make_config_id({"--ball-speed": 9, "--seed": 100, "--no-plot": None})
    b = br.make_config_id({"--no-plot": None, "--ball-speed": 9, "--seed": 101,
                           "--output": "/tmp/x", "_timeout": 5})
    assert a == b == "ballspeed9__noplot"


def test_build_command_expands_list_values() -> None:
    """列表值展开为重复 flag（--task-overrides 每项一个 argv），不能传 list 字面量。"""
    cmd = br.build_command(
        "scripts/rm65_mpc_v12.py",
        {"--seed": 100, "--task-overrides": ["a=1", "b=2"], "_timeout": 9},
    )
    assert cmd == [
        sys.executable,
        str(br.PROJECT / "scripts/rm65_mpc_v12.py"),
        "--seed", "100",
        "--task-overrides", "a=1",
        "--task-overrides", "b=2",
    ]
    assert "_timeout" not in " ".join(cmd)


def test_resolve_identity_derives_mode() -> None:
    """身份 mode 留空时按 --position-mode 推导 torque/position。"""
    assert br.resolve_identity(_identity(), {"--seed": 1}).mode == "torque"
    assert br.resolve_identity(
        _identity(), {"--seed": 1, "--position-mode": None}).mode == "position"
    assert br.resolve_identity(
        _identity(mode="torque"), {"--position-mode": None}).mode == "torque"


def test_compute_config_hash_real_files(tmp_path: Path) -> None:
    """config_hash 由真实文件内容决定；缺文件必须报错，禁止假哈希。"""
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("robot_id: unitree_r1d\n", encoding="utf-8")
    b.write_text("dt: 0.005\n", encoding="utf-8")
    first = br.compute_config_hash({"robot": a, "runtime": b})
    second = br.compute_config_hash({"runtime": b, "robot": a})
    assert first == second
    assert len(first) == 16
    b.write_text("dt: 0.006\n", encoding="utf-8")
    assert br.compute_config_hash({"robot": a, "runtime": b}) != first
    with pytest.raises(FileNotFoundError, match="config_hash 无法计算"):
        br.compute_config_hash({"robot": tmp_path / "missing.yaml"})


def test_parse_result_line_supports_rm65_and_r1d() -> None:
    """结果行解析同时支持 RM65 __RESULT__ 与 R1D __R1D_RESULT__。"""
    rm65 = br.parse_result_line(
        "noise\n" + "".join(RESULT_LINE))
    assert rm65["result_line_kind"] == "rm65"
    assert rm65["hit_type"] == "active"
    assert rm65["pos_error"] == pytest.approx(0.01)
    r1d = br.parse_result_line(
        "__R1D_RESULT__: side=right hit_type=active hit_step=89 total_steps=111 "
        "safe_steps=110 failure_reason=None first_plan_ms=6500 output=/tmp/out\n")
    assert r1d["result_line_kind"] == "r1d"
    assert r1d["total_steps"] == 111
    assert r1d["hit_step"] == 89
    assert r1d["failure_reason"] == ""
    assert r1d["artifact_dir"] == "/tmp/out"
    assert br.parse_result_line("no result") == {"error": "no_result"}


# ---------------------------------------------------------------------------
# overnight：exp28 延迟读取 + 数据不足显式报错
# ---------------------------------------------------------------------------


def _write_exp24_csv(root: Path) -> None:
    """写一份最小 exp24 results.csv（3 个扰动格 × 6 seed × 3 档）。"""
    csv_path = root / "exp24_shared_full" / "results.csv"
    csv_path.parent.mkdir(parents=True)
    fieldnames = ["experiment", "config_id", "config_json", "seed", "hit",
                  "hit_type", "error"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for time_ms, space_m in ((0.0, 0.2), (25.0, 0.2), (50.0, 0.2)):
            for seed in range(1, 7):
                for ablation in ("tube_only", "full", "none"):
                    active = ((ablation == "tube_only" and seed == 1)
                              or (ablation == "full" and seed == 2))
                    writer.writerow({
                        "experiment": "exp24_shared_full",
                        "config_id": f"cell{time_ms}-{space_m}-{ablation}",
                        "config_json": json.dumps({
                            "--time-perturb-ms": time_ms,
                            "--space-perturb-m": space_m,
                            "--ablation": ablation,
                        }),
                        "seed": seed,
                        "hit": "True" if active else "False",
                        "hit_type": "active" if active else "miss",
                        "error": "",
                    })


def test_exp28_import_is_lazy() -> None:
    """exp28 注册为延迟工厂：import 阶段不读旧 exp24 CSV。"""
    spec = overnight.EXPERIMENTS["exp28_always_collision"]
    assert spec.grid == []
    assert spec.grid_factory is not None
    # 旧 22 个实验键仍存在且可用（不破坏旧默认行为）
    assert len(overnight.EXPERIMENTS) == 23
    for name, other in overnight.EXPERIMENTS.items():
        if name != "exp28_always_collision":
            assert not other.grid_factory


def test_exp28_seed_pool_missing_data_root(tmp_path: Path) -> None:
    """当前 data-root 缺 exp24 CSV 时报缺，不退回仓库旧根。"""
    with pytest.raises(FileNotFoundError, match="exp24"):
        overnight._exp28_seed_pool(data_root=tmp_path / "empty")


def test_exp28_seed_pool_insufficient_rows(tmp_path: Path) -> None:
    """CSV 存在但没有可用行 → ValueError（数据不足），不静默降级。"""
    csv_path = tmp_path / "exp24_shared_full" / "results.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(
        "experiment,config_id,config_json,seed,hit,hit_type,error\n", encoding="utf-8")
    with pytest.raises(ValueError, match="数据不足"):
        overnight._exp28_seed_pool(data_root=tmp_path)


def test_exp28_seed_pool_and_grid_from_current_root(tmp_path: Path) -> None:
    """有效 CSV 时按格挑选 seed，并构造含 --always-collision 的网格。"""
    _write_exp24_csv(tmp_path)
    pool = overnight._exp28_seed_pool(data_root=tmp_path)
    assert len(pool) == 6
    assert pool[-1] == (None, [1, 2, 3, 4, 5])
    by_cell = {cell: seeds for cell, seeds in pool[:-1] if cell is not None}
    assert by_cell[(0.0, 0.2)] == [1, 2]
    assert by_cell[(25.0, 0.2)] == [1, 2]
    assert by_cell[(50.0, 0.2)] == [1, 2]
    assert by_cell[(0.0, 0.1)] == []
    assert by_cell[(0.0, 0.3)] == []

    spec = overnight.make_exp28_spec(data_root=tmp_path)
    grid = spec.materialize_grid()
    expected = sum(len(seeds) for _, seeds in pool) * 4
    assert len(grid) == expected
    assert all("--always-collision" in row for row in grid)
    assert all(row["--ball-speed"] == 9 for row in grid)


# ---------------------------------------------------------------------------
# collect_fig_assets：目录注入 + R1D 命令构造（不改变 RM65 旧默认）
# ---------------------------------------------------------------------------


def test_fig_assets_default_paths_unchanged() -> None:
    """RM65 旧默认仍是 experiment_data/exp18_fig_assets，超时 180s。"""
    paths = fig_assets.resolve_paths()
    assert paths.out_dir == fig_assets.PROJECT / "experiment_data" / "exp18_fig_assets"
    assert paths.raw_dir == paths.out_dir / "raw"
    assert paths.run_timeout_s == 180
    injected = fig_assets.resolve_paths(
        data_root="/tmp/data", output="/tmp/out", timeout_s=42)
    assert injected.out_dir == Path("/tmp/out")
    assert injected.raw_dir == Path("/tmp/out") / "raw"
    assert injected.run_timeout_s == 42


def test_fig_assets_r1d_command_uses_output_not_dump(tmp_path: Path) -> None:
    """R1D episode 命令：带机器人配置 + --output，不传 RM65 的 --dump-trajectory。"""
    robot_args = fig_assets.build_robot_args(
        "configs/robots/r1d_right.yaml",
        "configs/robots/r1d_v12_runtime_right.yaml",
        "configs/rackets/blade100_v9_reference.yaml")
    assert fig_assets.is_r1d_robot("configs/robots/r1d_right.yaml")
    assert not fig_assets.is_r1d_robot("configs/robots/rm65.yaml")
    cmd = fig_assets.build_episode_cmd(
        dict(fig_assets.BASE_9MS), robot_args=robot_args, is_r1d=True,
        output_dir=tmp_path / "out")
    assert "--robot-config" in cmd
    assert "--runtime-config" in cmd
    assert "--racket-config" in cmd
    assert "--output" in cmd
    assert "--dump-trajectory" not in cmd
    assert "--serve-box" in cmd


def test_fig_assets_rm65_command_keeps_dump(tmp_path: Path) -> None:
    """RM65 命令保持旧行为：--dump-trajectory 可见；fixed_task 时剔除冲突参数。"""
    cmd = fig_assets.build_episode_cmd(
        dict(fig_assets.BASE_9MS), dump_path=tmp_path / "traj.npz")
    assert "--dump-trajectory" in cmd
    assert "--robot-config" not in cmd
    fixed = fig_assets.build_episode_cmd(
        dict(fig_assets.BASE_9MS), dump_path=tmp_path / "traj.npz", fixed_task=True)
    assert "--ball-speed" not in fixed
    assert "--serve-box" not in fixed


# ---------------------------------------------------------------------------
# 产物提取（runs.csv 追加列）
# ---------------------------------------------------------------------------


def test_artifact_metrics_extract_physical_hit(tmp_path: Path) -> None:
    """从 v12_episode.json 提取物理命中/出球质量列；缺失产物显式标记。"""
    artifact_dir = tmp_path / "art"
    artifact_dir.mkdir()
    payload = {
        "hit": {
            "physical_hit": True,
            "physical_hit_type": "active",
            "physical_hit_step": 89,
            "v_racket_at_hit": 0.574,
            "ball_out_speed": 8.1,
            "ball_out_vy": -2.3,
            "contact_events": [{"face": "racket", "distance": 0.0}],
        },
        "episode": {"total_steps": 111},
    }
    (artifact_dir / "v12_episode.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    metrics = smoke._artifact_metrics(artifact_dir)
    assert metrics["physical_hit"] is True
    assert metrics["physical_hit_type"] == "active"
    assert metrics["physical_hit_step"] == 89
    assert metrics["contact_event_count"] == 1
    assert metrics["artifact_error"] == ""
    missing = smoke._artifact_metrics(tmp_path / "nope")
    assert missing["artifact_error"] == "missing_v12_episode"


def test_format_override_values() -> None:
    """override 值格式化为 V12 可解析的 YAML 标量。"""
    assert smoke._format_override(1000000.0) == "1000000.0"
    assert smoke._format_override(True) == "true"
    assert smoke._format_override(False) == "false"
    assert smoke._format_override("ballstate") == "ballstate"
    assert smoke._format_override(0) == "0"


# ---------------------------------------------------------------------------
# 评审修复轮：B1 data-root 贯穿 / B3 旧 CSV CLI 续跑 / B2 runtime 快照 /
# N3 单 seed 防呆 / N5 工具产物 / N6 前馈代表 case / N10 hit_semantics
# ---------------------------------------------------------------------------


def test_exp28_grid_uses_runtime_data_root(tmp_path: Path) -> None:
    """B1：run_batch 传入的 --data-root 必须贯穿到 exp28 seed 池构造。"""
    data_root = tmp_path / "runtime_root"
    _write_exp24_csv(data_root)
    spec = overnight.make_exp28_spec()
    summary = br.run_batch(
        spec, workers=1, data_root=data_root, output=tmp_path / "out", dry_run=True)
    assert summary["dry_run"] is True
    assert summary["total"] > 0
    assert all("--always-collision" in task["command"] for task in summary["tasks"])


def test_cli_batch_runner_reports_missing_data_root(tmp_path: Path) -> None:
    """B1：CLI 指定不存在的 data-root 时必须非 0 退出且报出缺失路径。"""
    missing = tmp_path / "no_such_root"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "exp" / "batch_runner.py"),
         "exp28_always_collision", "--data-root", str(missing), "--dry-run"],
        capture_output=True, text=True, cwd=ROOT, env=_cli_env(),
        encoding="utf-8", errors="replace")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "exp24_shared_full" in proc.stderr
    assert "results.csv" in proc.stderr


def test_cli_batch_runner_resumes_legacy_csv(tmp_path: Path) -> None:
    """B3：CLI 直接运行时旧 21 列 CSV 默认续跑（不再因两个 RunIdentity 类而报错）。"""
    data_root = tmp_path / "data"
    exp_dir = data_root / "exp16_limits_v2"
    exp_dir.mkdir(parents=True)
    csv_path = exp_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=br.LEGACY_CSV_FIELDS)
        writer.writeheader()
        writer.writerow({"experiment": "exp16_limits_v2", "config_id": "old",
                         "seed": 1, "error": ""})
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "exp" / "batch_runner.py"),
         "exp16_limits_v2", "--data-root", str(data_root),
         "--dry-run", "--limit", "0"],
        capture_output=True, text=True, cwd=ROOT, env=_cli_env(),
        encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "旧格式 CSV" in proc.stdout


def test_runtime_overrides_snapshot_and_identity(
    manifest: dict, tmp_path: Path,
) -> None:
    """B2：运行时覆盖生成快照、影响 --runtime-config 与 per-task config_hash。"""
    resolved = smoke.resolve_manifest(manifest, output=tmp_path / "run")
    tasks = {(task["experiment"], task["case_id"]): task
             for task in resolved["tasks"]}
    sim = tasks[("exp16_limits_v2", "sim_tcp18_speed7")]
    real = tasks[("exp16_limits_v2", "real_tcp10_speed7")]
    assert sim["runtime"]["snapshot"] is None
    assert real["runtime"]["overrides"] == {"max_tcp_speed": 1.0}
    assert real["identity"]["config_hash"] != sim["identity"]["config_hash"]
    snapshot = Path(str(real["runtime"]["snapshot"]))
    assert snapshot.is_file()
    payload = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    assert payload["max_tcp_speed"] == 1.0
    assert str(real["argv"]["--runtime-config"]) == str(real["runtime"]["snapshot"])
    # exp26 三档 TCP 的 per-task 身份互不相同
    tcp_hashes = {
        task["case_id"]: task["identity"]["config_hash"]
        for task in resolved["tasks"] if task["experiment"] == "exp26_tcp_grid"}
    assert len(set(tcp_hashes.values())) == 3


def test_limit_rejects_single_seed_cell(
    manifest: dict, tmp_path: Path,
) -> None:
    """N3：--limit 截断出单 seed cell 时拒绝，显式 allow_single_seed 才放行。"""
    with pytest.raises(smoke.ManifestError, match="allow-single-seed"):
        smoke.resolve_manifest(manifest, output=tmp_path / "run", limit=1)
    resolved = smoke.resolve_manifest(
        manifest, output=tmp_path / "run", limit=1, allow_single_seed=True)
    assert resolved["counts"]["batch_tasks"] == 1
    assert resolved["allow_single_seed"] is True


def test_tool_outputs_patterns(tmp_path: Path) -> None:
    """N5：expected_outputs 支持目录/文件/glob，缺失显式列出。"""
    run_root = tmp_path / "run"
    (run_root / "tool" / "raw").mkdir(parents=True)
    (run_root / "tool" / "raw" / "a.npz").write_text("x", encoding="utf-8")
    assert smoke._check_tool_outputs(
        ["tool/raw", "tool/raw/a.npz", "tool/raw/*.npz"], run_root) == []
    assert smoke._check_tool_outputs(
        ["tool/raw/*.log", "tool/x.json"], run_root) == [
            "tool/raw/*.log", "tool/x.json"]


def test_tool_dependency_flags_missing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N5：工具退出码 0 但没产出时必须标 invalid_output。"""

    class _OkResult:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **k: _OkResult())
    unit = {
        "id": "tool1", "kind": "tool", "status": "ready",
        "command": ["{python}", "-c", "pass"],
        "expected_outputs": ["tool1/out.json"], "timeout_s": 1,
    }
    payload = smoke._run_tool_dependency(
        unit, tmp_path / "run", tmp_path / "data")
    assert payload["status"] == "invalid_output"
    assert payload["missing_outputs"] == ["tool1/out.json"]


def test_position_no_feedforward_case(manifest: dict, tmp_path: Path) -> None:
    """N6：position 代表链包含前馈关闭对照（actuator.feedforward=false）。"""
    resolved = smoke.resolve_manifest(
        manifest, output=tmp_path / "run", mode="both")
    tasks = [task for task in resolved["tasks"]
             if task["case_id"] == "full_position_nofeedforward"]
    assert len(tasks) == 2
    for task in tasks:
        assert task["mode"] == "position"
        assert task["argv"].get("--position-mode") is None
        overrides = task["argv"]["--task-overrides"]
        assert "actuator.feedforward=false" in overrides


def test_collect_runs_adds_hit_semantics(tmp_path: Path) -> None:
    """N10：runs.csv/run_summary 显式标记 hit 为 legacy 口径。"""
    batch_dir = tmp_path / "batch" / "exp_x"
    batch_dir.mkdir(parents=True)
    csv_path = batch_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=br.CSV_FIELDS)
        writer.writeheader()
        writer.writerow({"experiment": "exp_x", "config_id": "c", "seed": 100,
                         "hit": "True", "hit_type": "active", "error": ""})
    resolved = {"run_root": str(tmp_path / "run")}
    payload = smoke._collect_runs(
        resolved, tmp_path / "batch", [{"csv": str(csv_path)}])
    assert payload["hit_semantics"] == "legacy"
    rows = _read_rows(tmp_path / "run" / "runs.csv")
    assert len(rows) == 1
    assert rows[0]["hit_semantics"] == "legacy"
    assert rows[0]["hit"] == "True"


# ---------------------------------------------------------------------------
# 第三轮评审：N14 case 计划字段 / N15 稳定 config_id / N13 CLI UTF-8
# ---------------------------------------------------------------------------

PLAN_FIELDS = (
    "id",
    "source_paper_item",
    "experiment_id",
    "argv",
    "extract_argv",
    "plot_argv",
    "expected_outputs",
    "mode",
    "seed",
    "timeout_s",
)


def test_manifest_cases_have_all_plan_fields(manifest: dict) -> None:
    """N14：每个 case 都带齐 10 个计划字段；blocked case 不适用处显式 null。"""
    normalized = smoke.validate_manifest(copy.deepcopy(manifest))
    total_cases = 0
    blocked_cases = 0
    for unit in normalized["experiments"]:
        if unit["kind"] != "batch":
            continue
        for case in unit["cases"]:
            total_cases += 1
            for field in PLAN_FIELDS:
                assert field in case, (unit["id"], case["id"], field)
            assert case["experiment_id"] == unit["id"]
            assert case["source_paper_item"], (unit["id"], case["id"])
            if case["status"] == "blocked":
                blocked_cases += 1
                for field in ("argv", "extract_argv", "plot_argv",
                              "expected_outputs", "seed", "timeout_s"):
                    assert case[field] is None, (case["id"], field)
            else:
                assert case["seeds"] == [100, 101]
                assert case["seed"] == [100, 101]
                assert case["timeout_s"] == 900.0
                assert case["expected_outputs"]
                assert case["extract_argv"]
                assert case["plot_argv"]
    assert total_cases == 68
    assert blocked_cases == 1


def test_dry_run_prints_case_plan_fields(
    manifest: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """N14：--dry-run 逐 case 打印 9 个计划字段，T8c 可直接核对。"""
    code = smoke.main([
        "--manifest", str(ROOT / "configs" / "experiments" / "r1d_paper_smoke.yaml"),
        "--dry-run", "--output", str(tmp_path / "run")])
    assert code == 0
    out = capsys.readouterr().out
    case_lines = [line for line in out.splitlines() if line.startswith("  CASE ")]
    # 68 个正文实验 case + 1 个 batch 额外依赖（exp17a_noise_none_control）
    assert len(case_lines) == 69
    assert any("CASE exp17a_noise_none_control/" in line for line in case_lines)
    assert "CASE exp25_select_one/select_one_full_speed9" in out
    blocked_line = next(line for line in case_lines
                        if "exp25_select_one/select_one_full_speed9" in line)
    assert "seed=null" in blocked_line
    assert "argv=null" in blocked_line
    assert "extract_argv=null" in blocked_line
    assert "plot_argv=null" in blocked_line
    assert "expected_outputs=null" in blocked_line
    ready_line = next(line for line in case_lines if "exp15_speed_v2/speed7" in line)
    assert "extract_argv=" in ready_line
    assert "plot_argv=" in ready_line
    assert "expected_outputs=" in ready_line
    assert "[100, 101]" in ready_line or "[100,101]" in ready_line


def test_config_id_stable_across_output_roots(manifest: dict, tmp_path: Path) -> None:
    """N15：runtime 覆盖任务的 config_id 与 run_root 无关，且快照路径仍不同。"""
    first = smoke.resolve_manifest(copy.deepcopy(manifest), output=tmp_path / "root_a")
    second = smoke.resolve_manifest(copy.deepcopy(manifest), output=tmp_path / "root_b")

    def _pick(resolved: dict, experiment: str, case_id: str) -> dict:
        return next(task for task in resolved["tasks"]
                    if task["experiment"] == experiment
                    and task["case_id"] == case_id and task["seed"] == 100)

    for experiment, case_id in (
        ("exp16_limits_v2", "real_tcp10_speed7"),
        ("exp26_tcp_grid", "tcp10_full_speed7"),
    ):
        task_a = _pick(first, experiment, case_id)
        task_b = _pick(second, experiment, case_id)
        assert task_a["config_id"] == task_b["config_id"]
        assert task_a["config_hash"] == task_b["config_hash"]
        assert "#" in task_a["config_id"]
        assert (task_a["argv"]["--runtime-config"]
                != task_b["argv"]["--runtime-config"])
