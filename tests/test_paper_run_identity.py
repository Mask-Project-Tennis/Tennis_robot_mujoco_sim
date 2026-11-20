"""R1D×V12 论文 smoke 的任务身份、日志与 catalog 契约测试（工作线 C）。

覆盖 ``docs/r1d_v12_parallel_repair_plan.md`` §8：

1. 同一 manifest/data_root 下换 output 续跑：行数不变、config_id 稳定；
2. 参数内容变化 → 身份变化 → 既有隔离契约拒绝混写；
3. ``_config_id`` 保留元数据：不进 CLI、不被显示路径重算、旧调用兼容、
   同 seed 不同 mode/robot/racket 不误去重；
4. 日志：``task_log_path`` 与实际写日志一致、Windows 合法/唯一/≤96、
   换 output 后 manifest/CSV 回填旧目录真实文件；
5. catalog：相对解析、整棵 run_root 移动后可解析、跨盘绝对路径、坏 catalog 非零退出。

真实 ``run_batch`` 一律用 ``sys.executable`` + ``tmp_path`` 下的极简假脚本
（只输出一行 ``__R1D_RESULT__``），不加载 MuJoCo、不写 ``experiment_data/``。
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.exp import batch_runner, run_catalog
from scripts.exp import run_r1d_paper_smoke as smoke
from scripts.exp.batch_runner import ExperimentSpec, RunIdentity

PROJECT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "r1d_review_contract"
ILLEGAL_FILENAME_CHARS = set('<>:"/\\|?*')
RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

#: 极简假仿真入口：解析参数、写最小 ``v12_episode_v2`` 产物并输出结果行。
#: M09 起 producer 会校验已完成任务的产物/身份，假入口必须写真实最小产物，
#: 否则既有续跑测试会按新契约失败（不放宽校验）。
FAKE_ENTRY_SOURCE = '''\
"""极简假仿真入口：写最小但合法的 v12_episode_v2 产物并输出 __R1D_RESULT__。

R02 起 producer 会校验 ``payload["config"]`` 与 ``--ball-speed``/mode 对齐，
假入口必须写出同样的最小 config（本文件 manifest 不产生 task-overrides）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ball-speed", type=float, default=7.0)
    args, unknown = parser.parse_known_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mode = "position" if "--position-mode" in unknown else "torque"
    result = {
        "hit_type": "miss",
        "hit_step": None,
        "total_steps": 5,
        "safe_steps": 5,
        "failure_reason": None,
        "first_plan_ms": 1.0,
    }
    payload = {
        "schema": "v12_episode_v2",
        "robot_id": "unitree_r1d",
        "mode": mode,
        "racket_id": "blade100_v9_reference",
        "joint_names": [f"r_joint{i}" for i in range(1, 7)],
        "seed": args.seed,
        "config": {
            "robot_id": "unitree_r1d",
            "actuator_mode": mode,
            "feedforward": (False if "--no-feedforward" in unknown else True)
            if mode == "position" else None,
            "seed": args.seed,
            "serve_config": {"ball_speed": args.ball_speed},
            "mpc_config": {},
            "solver_config": {},
            "cost_config": {"tube": {}},
            "perception_config": {},
            "hitting_config": {},
            "scene_config": {},
        },
        "hit": {"physical_hit": False, "physical_hit_type": "miss"},
        "episode": {"total_steps": 5, "safe_steps": 5},
    }
    (output / "v12_episode.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print("__R1D_RESULT__: " + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------


def _write_fake_entry(directory: Path) -> Path:
    """在 ``directory`` 写一个只输出 ``__R1D_RESULT__`` 的假脚本。"""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "fake_r1d_entry.py"
    script.write_text(FAKE_ENTRY_SOURCE, encoding="utf-8")
    return script


def _write_smoke_manifest(
    tmp_path: Path,
    script: Path,
    *,
    max_tcp_speed: float = 1.0,
) -> Path:
    """写一个最小但完整合法的 smoke manifest（真实配置角色 + 假仿真脚本）。"""
    payload: dict[str, Any] = {
        "schema": "r1d_paper_smoke_v1",
        "description": "工作线 C 身份/日志/catalog 测试用最小 manifest",
        "generated_by": "test_paper_run_identity",
        "seeds": [100, 101],
        "workers_default": 1,
        "timeout_s_default": 60,
        "run_root_default": str(tmp_path / "default_run"),
        "data_root_name": "batch",
        "script": str(script),
        "configs": {
            "robot": "configs/robots/r1d_right.yaml",
            "runtime": "configs/robots/r1d_v12_runtime_right.yaml",
            "racket": "configs/rackets/blade100_v9_reference.yaml",
            "task_overlay": "configs/r1d_v12.yaml",
            "sim_task_config": None,
        },
        "identity_defaults": {
            "robot_id": "unitree_r1d",
            "racket_id": "blade100_v9_reference",
            "metric_schema": "v12_episode_v2_physical_hit",
        },
        "base_argv": {"--serve-box": None, "--no-plot": None},
        "ablation_profiles": {
            "full": {},
            "none": {"cost.tube.use_softmin_terminal": False},
        },
        "case_defaults": {
            "experiment_id": "{experiment_id}",
            "source_paper_item": (
                "docs/experiments/design/r1d_v12_parity_matrix.md#{experiment_id}"),
            "seed": [100, 101],
            "timeout_s": 60,
            "expected_outputs": ["v12_episode.json"],
            "extract_argv": [
                "{python}", "scripts/extract/extract_r1d_smoke_stats.py",
                "--experiment", "{experiment_id}",
                "--case", "{case_id}",
                "--data-root", "{data_root}",
                "--catalog", "{catalog}",
                "--mode", "{mode}",
                "--output", "{run_root}/stats",
            ],
            # M07：plot 只读 <stats-root>/<experiment>/<case>.json，
            # 不再接受 --catalog；两模式按 <mode> 分层避免互相覆盖。
            "plot_argv": [
                "{python}", "scripts/plot/plot_r1d_smoke_stats.py",
                "--stats-root", "{run_root}/stats/{mode}",
                "--experiment", "{experiment_id}",
                "--case", "{case_id}",
                "--output", "{run_root}/figures/{mode}",
            ],
        },
        "experiments": [
            {
                "id": "exp16_limits_v2",
                "kind": "batch",
                "title": "TCP 1.0 对照",
                "cases": [
                    {
                        "id": "sim_tcp18_speed7",
                        "mode": "torque",
                        "status": "ready",
                        "argv": {"--ball-speed": 7},
                    },
                    {
                        "id": "real_tcp10_speed7",
                        "mode": "torque",
                        "status": "ready",
                        "argv": {"--ball-speed": 7},
                        "runtime_overrides": {"max_tcp_speed": max_tcp_speed},
                    },
                ],
            },
        ],
        "extra_dependencies": [],
    }
    manifest = tmp_path / "smoke_manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return manifest


def _run_smoke(manifest: Path, output: Path, data_root: Path) -> int:
    """以 tmp 输出根执行 smoke main（真实 run_batch + 假仿真脚本）。"""
    return smoke.main([
        "--manifest", str(manifest),
        "--output", str(output),
        "--data-root", str(data_root),
        "--mode", "torque",
        "--timeout", "60",
    ])


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 为字典列表（保持文件编码/换行语义）。"""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(__import__("csv").DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_against(base: Path, value: str | Path) -> Path:
    """把可能相对的路径按 ``base`` 解析为绝对路径（不要求存在）。"""
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def _assert_windows_safe_filename(name: str) -> None:
    """断言文件名 Windows 合法且总长 ≤ 96（含 ``.log``）。"""
    assert name.endswith(".log"), name
    assert len(name) <= batch_runner.LOG_FILENAME_MAX_CHARS, (len(name), name)
    assert name == name.strip(), repr(name)
    assert not (set(name) & ILLEGAL_FILENAME_CHARS), name
    assert re.fullmatch(r"[A-Za-z0-9._-]+\.log", name), name
    stem = name[:-4]
    assert stem and not stem.endswith("."), name
    for part in stem.split("."):
        assert part.upper() not in RESERVED_NAMES, name


def _fixture_tasks() -> list[dict[str, Any]]:
    """读取 R0 冻结的 4 条合成任务（``tasks.sample.json``）。"""
    payload = _read_json(FIXTURE_DIR / "tasks.sample.json")
    return list(payload["tasks"])


# ---------------------------------------------------------------------------
# §8-1 / §8-2 身份稳定与隔离
# ---------------------------------------------------------------------------


class TestStableIdentityAcrossOutputs:
    """F05：换 output 续跑不得因显示路径重算身份而重复执行。"""

    def test_switch_output_resume_keeps_rows_and_config_ids(
            self, tmp_path: Path) -> None:
        """plan_a 首次 4 行；换 plan_b 续跑 launched=0 且仍是 4 行。"""
        script = _write_fake_entry(tmp_path)
        manifest = _write_smoke_manifest(tmp_path, script)
        data_root = tmp_path / "csv_root"
        plan_a, plan_b = tmp_path / "plan_a", tmp_path / "plan_b"

        assert _run_smoke(manifest, plan_a, data_root) == 0
        results = data_root / "exp16_limits_v2" / "results.csv"
        rows_a = _read_csv(results)
        assert len(rows_a) == 4, "2 case × 2 seed 应写 4 行"
        ids_a = {row["config_id"] for row in rows_a}
        assert len(ids_a) == 2, "每个 case 一个稳定 config_id"
        assert len({(row["config_id"], row["seed"]) for row in rows_a}) == 4

        resolved_a = _read_json(plan_a / "manifest_resolved.json")
        assert {task["config_id"] for task in resolved_a["tasks"]} == ids_a

        assert _run_smoke(manifest, plan_b, data_root) == 0
        rows_b = _read_csv(results)
        assert len(rows_b) == 4, "换 output 续跑不得新增重复行"
        assert {row["config_id"] for row in rows_b} == ids_a
        assert len({(row["config_id"], row["seed"]) for row in rows_b}) == 4

        summary_b = _read_json(plan_b / "run_summary.json")
        assert summary_b["experiments"], "至少应有一个 batch 单元汇总"
        for unit_summary in summary_b["experiments"]:
            assert unit_summary["launched"] == 0
            assert unit_summary["done_before"] == 4

        resolved_b = _read_json(plan_b / "manifest_resolved.json")
        assert {task["config_id"] for task in resolved_b["tasks"]} == ids_a
        for task in resolved_b["tasks"]:
            assert str(plan_b) in str(task["argv"]["--output"])
            assert task["config_id"] in ids_a


class TestIdentityIsolation:
    """§8-2：参数内容变化必须改变身份，并按既有契约拒绝混写。"""

    def test_changed_parameter_changes_identity_and_is_rejected(
            self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """只改 runtime 覆盖值 → config_hash/config_id 变化 → 拒绝追加。"""
        script = _write_fake_entry(tmp_path)
        manifest = _write_smoke_manifest(tmp_path, script, max_tcp_speed=1.0)
        data_root = tmp_path / "csv_root"
        plan_a, plan_b = tmp_path / "plan_a", tmp_path / "plan_b"

        assert _run_smoke(manifest, plan_a, data_root) == 0
        results = data_root / "exp16_limits_v2" / "results.csv"
        rows_before = _read_csv(results)
        assert len(rows_before) == 4

        # 同一 manifest 路径重写：物理条件/seed 不变，只改 runtime 覆盖内容。
        _write_smoke_manifest(tmp_path, script, max_tcp_speed=1.4)
        code = _run_smoke(manifest, plan_b, data_root)
        captured = capsys.readouterr()
        assert code == 2, f"身份冲突必须非零退出；stdout={captured.out[-2000:]}"
        assert "混入不同身份" in captured.err

        rows_after = _read_csv(results)
        assert len(rows_after) == 4, "被拒绝的新身份不得写入任何行"
        assert [row["config_id"] for row in rows_after] == [
            row["config_id"] for row in rows_before]


class TestConfigIdMetadata:
    """§8-3：``_config_id`` 的保留元数据语义。"""

    def _resolved_pair(self, tmp_path: Path) -> tuple[Path, dict[str, Any],
                                                     dict[str, Any]]:
        """在同一 manifest 上解析 plan_a / plan_b 两份任务清单。"""
        script = _write_fake_entry(tmp_path)
        manifest = _write_smoke_manifest(tmp_path, script)
        payload = smoke.load_manifest(manifest)
        plan_a, plan_b = tmp_path / "plan_a", tmp_path / "plan_b"
        data = tmp_path / "csv_root"
        resolved_a = smoke.resolve_manifest(
            payload, manifest_path=manifest, output=plan_a, data_root=data,
            mode="both")
        resolved_b = smoke.resolve_manifest(
            payload, manifest_path=manifest, output=plan_b, data_root=data,
            mode="both")
        return manifest, resolved_a, resolved_b

    def test_specs_carry_config_id_and_cli_excludes_meta(
            self, tmp_path: Path) -> None:
        """spec.grid 带 ``_config_id``；它不出现在 argv/命令行里。"""
        manifest, resolved_a, _resolved_b = self._resolved_pair(tmp_path)
        specs = smoke.build_specs_from_manifest(
            manifest, output=tmp_path / "plan_a", data_root=tmp_path / "csv_root",
            mode="both")
        spec = {item.name: item for item in specs}["exp16_limits_v2"]
        expected = [task["config_id"] for task in resolved_a["tasks"]]
        assert [params["_config_id"] for params in spec.grid] == expected

        for params in spec.grid:
            command = batch_runner.build_command("scripts/rm65_mpc_v12.py", params)
            # 保留元数据不得变成一个 CLI token（--seed 等可见参数仍在）
            assert not any(token.startswith("_") for token in command)
            assert "--seed" in command
            for key in ("--output", "--seed"):
                assert key in params
        tcp = next(task for task in resolved_a["tasks"]
                   if task["case_id"] == "real_tcp10_speed7")
        assert batch_runner.make_config_id(tcp["argv"]) != tcp["config_id"]
        assert "#" in tcp["config_id"]

    def test_config_id_stable_but_display_paths_differ(
            self, tmp_path: Path) -> None:
        """同任务换 output 后 config_id 不变、--output 变；快照路径不参与身份。"""
        _manifest, resolved_a, resolved_b = self._resolved_pair(tmp_path)
        pair = {(task["case_id"], task["seed"]): task
                for task in resolved_a["tasks"]}
        for task_b in resolved_b["tasks"]:
            task_a = pair[(task_b["case_id"], task_b["seed"])]
            assert task_a["config_id"] == task_b["config_id"]
            assert task_a["argv"]["--output"] != task_b["argv"]["--output"]
            assert task_a["config_hash"] == task_b["config_hash"]

    def test_legacy_grid_without_config_id_uses_params_id(
            self, tmp_path: Path) -> None:
        """旧调用（无 ``_config_id``）继续用 make_config_id，日志路径一致。"""
        script = _write_fake_entry(tmp_path)
        spec = ExperimentSpec(
            name="legacy_exp",
            script=str(script),
            grid=[{"--ball-speed": 7, "--seed": 100},
                  {"--ball-speed": 7, "--seed": 101}],
        )
        log_dir = tmp_path / "legacy_logs"
        summary = batch_runner.run_batch(
            spec, workers=1, data_root=tmp_path / "legacy_csv", timeout_s=60,
            log_dir=log_dir)
        rows = _read_csv(Path(summary["csv"]))
        assert [row["config_id"] for row in rows] == [
            batch_runner.make_config_id(params) for params in spec.grid]
        for params, row in zip(spec.grid, rows):
            planned = batch_runner.task_log_path(
                log_dir, "legacy_exp", batch_runner.make_config_id(params),
                params["--seed"])
            assert Path(row["log_path"]) == planned
            assert planned.is_file()

    def test_same_seed_different_mode_not_deduplicated(
            self, tmp_path: Path) -> None:
        """同 seed 的 torque/position 是两条任务，不得被误去重。"""
        script = _write_fake_entry(tmp_path)
        spec = ExperimentSpec(
            name="mode_exp",
            script=str(script),
            grid=[
                {"--ball-speed": 7, "--seed": 100},
                {"--ball-speed": 7, "--seed": 100, "--position-mode": None},
            ],
            identity=RunIdentity(
                robot_id="unitree_r1d", mode="",
                racket_id="blade100_v9_reference", config_hash="abc123",
                metric_schema="v12_episode_v2_physical_hit"),
        )
        summary = batch_runner.run_batch(
            spec, workers=1, data_root=tmp_path / "mode_csv", timeout_s=60)
        assert summary["launched"] == 2, "不同 mode 不得去重"
        rows = _read_csv(Path(summary["csv"]))
        assert len(rows) == 2
        assert {row["mode"] for row in rows} == {"torque", "position"}
        assert len({row["config_id"] for row in rows}) == 2

    def test_resume_key_distinguishes_robot_and_racket(self) -> None:
        """同 config_id/seed 下，robot/racket 不同也不得碰撞。"""
        params = {"--ball-speed": 7, "--seed": 100,
                  "_config_id": "ballspeed7"}
        variants = [
            ("rm65", "torque", "rm65_legacy"),
            ("unitree_r1d", "torque", "rm65_legacy"),
            ("unitree_r1d", "position", "rm65_legacy"),
            ("unitree_r1d", "torque", "blade100_v9_reference"),
        ]
        keys = {
            batch_runner.make_resume_key(
                "exp", params,
                RunIdentity(robot_id=robot, mode=mode, racket_id=racket,
                            config_hash="h", metric_schema="s"))
            for robot, mode, racket in variants
        }
        assert len(keys) == 4
        assert len({key.config_id for key in keys}) == 1


# ---------------------------------------------------------------------------
# §8-4 日志计划路径与实际写日志一致（N01）
# ---------------------------------------------------------------------------


class TestTaskLogPath:
    """N01：manifest 计划日志路径必须等于实际落盘文件名。"""

    def test_matches_written_log_and_header(self, tmp_path: Path) -> None:
        """``task_log_path`` 与 ``_write_run_log`` 解析到同一实际文件。"""
        log_dir = tmp_path / "logs"
        for task in _fixture_tasks():
            params = {**task["argv"], "_config_id": task["config_id"]}
            planned = batch_runner.task_log_path(
                log_dir, task["experiment"], task["config_id"], task["seed"])
            written = Path(batch_runner._write_run_log(
                log_dir, task["experiment"], params, ["python", "x.py"],
                "stdout\n"))
            assert written == planned
            assert planned.is_file()
            assert planned.read_text(encoding="utf-8").startswith(
                f"# config_id: {task['config_id']}\n")

    def test_windows_safe_unique_and_bounded(self, tmp_path: Path) -> None:
        """Windows 合法、总长 ≤96、同配置不同 seed/实验/超长 id 唯一。"""
        log_dir = tmp_path / "logs"
        tasks = _fixture_tasks()
        paths = [batch_runner.task_log_path(
            log_dir, task["experiment"], task["config_id"], task["seed"])
            for task in tasks]
        paths.append(batch_runner.task_log_path(
            log_dir, "exp17d_mechanism", tasks[0]["config_id"], 101))
        paths.append(batch_runner.task_log_path(
            log_dir, "exp16_limits_v2", "x" * 400, 100))
        for path in paths:
            _assert_windows_safe_filename(path.name)
        assert len(set(paths)) == len(paths)

    def test_backfill_after_output_switch_points_to_real_files(
            self, tmp_path: Path) -> None:
        """换 output 续跑后：manifest/CSV 回填旧目录真实文件，仅计划信息指向新根。"""
        script = _write_fake_entry(tmp_path)
        manifest = _write_smoke_manifest(tmp_path, script)
        data_root = tmp_path / "csv_root"
        plan_a, plan_b = tmp_path / "plan_a", tmp_path / "plan_b"
        assert _run_smoke(manifest, plan_a, data_root) == 0
        assert _run_smoke(manifest, plan_b, data_root) == 0

        resolved_b = _read_json(plan_b / "manifest_resolved.json")
        summary_b = _read_json(plan_b / "run_summary.json")
        rows = _read_csv(Path(summary_b["batch"]["runs_csv"]))
        assert len(rows) == 4
        csv_by_key = {(row["experiment"], row["config_id"], int(row["seed"])): row
                      for row in rows}

        for task in resolved_b["tasks"]:
            key = (task["experiment"], task["config_id"], int(task["seed"]))
            row = csv_by_key[key]
            log_value = task["log_path"]
            actual_log = _resolve_against(plan_b, log_value)
            assert actual_log.is_file()
            assert actual_log.resolve() == Path(row["log_path"]).resolve()
            assert plan_a.resolve() in Path(row["log_path"]).resolve().parents
            # 旧文件在新 run_root 之外 → 保留绝对引用（不搬旧数据）
            assert Path(log_value).is_absolute()
            assert plan_b.resolve() not in Path(log_value).resolve().parents

            actual_artifact = _resolve_against(plan_b, task["artifact_dir"])
            assert actual_artifact.is_dir()
            assert actual_artifact.resolve() == Path(
                row["artifact_dir"]).resolve()

            planned_log = _resolve_against(plan_b, task["planned_log_path"])
            # M06：run_root 内的计划路径写相对值（整树可搬移），
            # 但解析后必须落在本次 output 根内，且续跑不生成该计划日志。
            assert not Path(task["planned_log_path"]).is_absolute()
            assert plan_b.resolve() in planned_log.resolve().parents
            assert not planned_log.exists(), "续跑不应在新根生成计划日志"
            assert planned_log == batch_runner.task_log_path(
                plan_b / "logs", task["experiment"], task["config_id"],
                int(task["seed"]))
            planned_artifact = _resolve_against(
                plan_b, task["planned_artifact_dir"])
            assert not Path(task["planned_artifact_dir"]).is_absolute()
            assert plan_b.resolve() in planned_artifact.resolve().parents

        # plan_a 自己（未切换）的 manifest 也必须是相对引用且文件存在
        resolved_a = _read_json(plan_a / "manifest_resolved.json")
        for task in resolved_a["tasks"]:
            assert not Path(task["log_path"]).is_absolute()
            assert _resolve_against(plan_a, task["log_path"]).is_file()


# ---------------------------------------------------------------------------
# §8-5 catalog（F03 生产端）
# ---------------------------------------------------------------------------


def _copy_contract(tmp_path: Path) -> Path:
    """把 R0 冻结夹具复制到 tmp（相对路径需要整棵目录一起移动/复制）。"""
    destination = tmp_path / "contract"
    shutil.copytree(FIXTURE_DIR, destination)
    return destination


class TestRunCatalog:
    """F03：RunCatalog 加载、路径基准与显式 catalog 的错误退出。"""

    def test_sample_catalog_relative_paths(self, tmp_path: Path) -> None:
        """``catalog.sample.json`` 的相对路径按 catalog 文件/run_root 解析。"""
        contract = _copy_contract(tmp_path)
        catalog = run_catalog.load_catalog(contract / "catalog.sample.json")

        assert catalog.schema_version == 1
        assert catalog.catalog_path == (contract / "catalog.sample.json").resolve()
        assert catalog.robot_id == "unitree_r1d"
        assert catalog.run_root == (contract / "run").resolve()
        assert catalog.data_root == (contract / "run" / "data").resolve()
        assert catalog.runs_csv == (contract / "run" / "data" / "runs.csv").resolve()
        assert catalog.manifest_path == (
            contract / "run" / "manifest_resolved.json").resolve()
        assert catalog.source_sets["nominal_mechanism"] == "exp17d_mechanism"
        assert catalog.path_base == "run_root"

        # JSON 路径：相对 catalog 文件解析，绝对路径原样接受
        assert run_catalog.resolve_catalog_path(
            catalog, "run/data") == (contract / "run" / "data").resolve()
        absolute = tmp_path / "elsewhere" / "runs.csv"
        assert run_catalog.resolve_catalog_path(catalog, str(absolute)) == absolute
        # runs.csv 内的相对路径以 run_root 为基准
        relative = "artifacts/exp16_limits_v2/l1_tcp10/seed100"
        assert run_catalog.resolve_run_path(catalog, relative) == (
            contract / "run" / relative).resolve()
        assert run_catalog.resolve_catalog_path(
            catalog, relative, base="run_root") == (
                contract / "run" / relative).resolve()

    def test_moved_run_root_still_resolves(self, tmp_path: Path) -> None:
        """整棵 run_root（含 catalog）移动后仍按新位置解析。"""
        before = tmp_path / "before"
        shutil.copytree(FIXTURE_DIR, before)
        loaded_before = run_catalog.load_catalog(before / "catalog.sample.json")
        after = tmp_path / "after"
        before.rename(after)

        moved = run_catalog.load_catalog(after / "catalog.sample.json")
        relative = "artifacts/exp16_limits_v2/l1_tcp10/seed100"
        resolved = run_catalog.resolve_run_path(moved, relative)
        assert moved.run_root == (after / "run").resolve()
        assert resolved == (after / "run" / relative).resolve()
        assert not resolved.is_relative_to(loaded_before.run_root)

    @pytest.mark.skipif(os.name != "nt", reason="Windows 盘符绝对路径")
    def test_cross_drive_absolute_path_passthrough(self, tmp_path: Path) -> None:
        """跨盘绝对路径不得被拼到 run_root 下。"""
        contract = _copy_contract(tmp_path)
        catalog = run_catalog.load_catalog(contract / "catalog.sample.json")
        other_drive = Path(r"D:\other_root\artifacts\x")
        assert run_catalog.resolve_catalog_path(
            catalog, str(other_drive)) == other_drive
        assert run_catalog.resolve_run_path(catalog, str(other_drive)) == other_drive

    def test_missing_catalog_exits_nonzero(self, tmp_path: Path) -> None:
        """显式 catalog 不存在：非零退出，不回退猜父目录。"""
        missing = tmp_path / "not_exists" / "catalog.json"
        assert run_catalog.main([str(missing)]) == 2
        with pytest.raises(run_catalog.CatalogError):
            run_catalog.load_catalog(missing)

    def test_conflicting_declaration_exits_nonzero(self, tmp_path: Path) -> None:
        """相对声明越出 run_root / 声明与 manifest 根冲突：非零退出。"""
        contract = _copy_contract(tmp_path)
        sample = contract / "catalog.sample.json"
        payload = json.loads(sample.read_text(encoding="utf-8"))

        escaping = dict(payload)
        escaping["runs_csv"] = "../outside/runs.csv"
        (contract / "escaping.json").write_text(
            json.dumps(escaping), encoding="utf-8")
        assert run_catalog.main([str(contract / "escaping.json")]) == 2

        # manifest_resolved.json 声明了另一个 run_root → 冲突
        run_dir = contract / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest_resolved.json").write_text(
            json.dumps({"run_root": str(tmp_path / "other_root")}),
            encoding="utf-8")
        assert run_catalog.main([str(sample)]) == 2

    def test_identity_mismatch_exits_nonzero(self, tmp_path: Path) -> None:
        """catalog.robot_id 与 runs.csv/manifest 身份不一致：非零退出。"""
        contract = _copy_contract(tmp_path)
        data_dir = contract / "run" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "runs.csv").write_text(
            "experiment,config_id,seed,robot_id\n"
            "exp16_limits_v2,cfg,100,rm65\n", encoding="utf-8")
        assert run_catalog.main([str(contract / "catalog.sample.json")]) == 2

    def test_producer_writes_loadable_catalog(self, tmp_path: Path) -> None:
        """producer 在 output 根写 catalog.json，可被 load_catalog 消费。"""
        script = _write_fake_entry(tmp_path)
        manifest = _write_smoke_manifest(tmp_path, script)
        data_root = tmp_path / "csv_root"
        plan_a = tmp_path / "plan_a"
        assert _run_smoke(manifest, plan_a, data_root) == 0

        catalog_path = plan_a / "catalog.json"
        assert catalog_path.is_file()
        catalog = run_catalog.load_catalog(catalog_path)
        assert catalog.run_root == plan_a.resolve()
        assert catalog.data_root == data_root.resolve()
        assert catalog.runs_csv == (plan_a / "runs.csv").resolve()
        assert catalog.manifest_path == (plan_a / "manifest_resolved.json").resolve()
        assert catalog.robot_id == "unitree_r1d"
        assert catalog.source_sets == {"nominal_mechanism": "exp17d_mechanism"}
        assert run_catalog.main([str(catalog_path)]) == 0

        rows = _read_csv(catalog.runs_csv)
        assert len(rows) == 4
        for row in rows:
            resolved = run_catalog.resolve_run_path(catalog, row["log_path"])
            assert resolved.is_file()
            assert run_catalog.resolve_run_path(
                catalog, row["artifact_dir"]).is_dir()
        # 新 output 的续跑 catalog 仍指向旧实际文件（绝对路径原样保留）
        plan_b = tmp_path / "plan_b"
        assert _run_smoke(manifest, plan_b, data_root) == 0
        catalog_b = run_catalog.load_catalog(plan_b / "catalog.json")
        assert catalog_b.run_root == plan_b.resolve()
        rows_b = _read_csv(catalog_b.runs_csv)
        assert len(rows_b) == 4
        for row in rows_b:
            resolved = run_catalog.resolve_run_path(catalog_b, row["log_path"])
            assert resolved.is_file()
            assert plan_a.resolve() in resolved.resolve().parents


# ---------------------------------------------------------------------------
# 端到端最小回归：extract 仍消费 catalog，plot 只读 <stats>/<mode>
# ---------------------------------------------------------------------------


def test_resolved_commands_carry_catalog_token(tmp_path: Path) -> None:
    """extract 带 `{catalog}` 占位；plot 不再接受 --catalog（M07）。"""
    script = _write_fake_entry(tmp_path)
    manifest = _write_smoke_manifest(tmp_path, script)
    plan = tmp_path / "plan_a"
    resolved = smoke.resolve_manifest(
        smoke.load_manifest(manifest), manifest_path=manifest, output=plan,
        data_root=tmp_path / "csv_root", mode="torque")
    catalog_token = str((plan / "catalog.json").resolve())
    for task in resolved["tasks"]:
        mode = task["mode"]
        assert task["extract_argv"][task["extract_argv"].index("--mode") + 1] == mode
        assert "--catalog" in task["extract_argv"]
        assert catalog_token in task["extract_argv"], task["extract_argv"]
        # M07：plot 的命令行不再有 --catalog，输入根按模式隔离。
        plot_joined = " ".join(task["plot_argv"])
        assert "--catalog" not in task["plot_argv"], task["plot_argv"]
        assert f"/stats/{mode}" in plot_joined, task["plot_argv"]
        assert f"/figures/{mode}" in plot_joined, task["plot_argv"]
        assert "{mode}" not in plot_joined
