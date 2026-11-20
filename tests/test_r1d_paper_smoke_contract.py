"""R1D 论文 smoke 的命令契约、可搬移性与续跑产物校验（M06–M09 验收）。

覆盖 ``results/r1d_v12_parity/reviews/merge_ba8acfd_2026-11-20/fix_plan.md``
§2 的 M06/M07/M08/M09 与 §5 验收矩阵：

1. 真实 manifest（``configs/experiments/r1d_paper_smoke.yaml``）生成的 136 条
   extract/plot 命令全部能被各自 argparse 接受；136 条 plot 不再带
   ``--catalog``，32 条混合实验 extract 带各自 ``--mode``。
2. 真实跑一组 extract→plot（torque 与 position 各一条），产物分别落在
   ``<run_root>/stats/<mode>/...`` 与 ``<run_root>/figures/<mode>/...``，互不覆盖。
3. producer 产物整树移动后 ``run_catalog.load_catalog`` 与统计 consumer
   （extract 读 catalog + runs.csv）仍正常。
4. 已完成任务产物缺失/损坏/身份不符/日志缺失时续跑明确失败（exit 1）并打印
   恢复指引；恢复后 exit 0，旧 runs.csv 行保留。

真实 ``run_batch`` 一律用 ``sys.executable`` + ``tmp_path`` 下的假脚本
（写最小 ``v12_episode_v2`` JSON），不加载 MuJoCo、不写 ``experiment_data/``。
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import yaml

from scripts.exp import run_catalog
from scripts.exp import run_r1d_paper_smoke as smoke
from scripts.extract import extract_r1d_smoke_stats as extractor
from scripts.plot import plot_r1d_smoke_stats as plotter

PROJECT = Path(__file__).resolve().parents[1]
REAL_MANIFEST = PROJECT / "configs" / "experiments" / "r1d_paper_smoke.yaml"
#: review 认定的混合模式实验（同一实验内同时存在 torque/position case）。
MIXED_EXPERIMENTS = ("exp15_speed_v2", "exp17d_mechanism", "exp27_ball_quality")
FAKE_ENTRY_SOURCE = '''\
"""假仿真入口：写 schema 合法的 v12/r1d JSON+NPZ，不加载 MuJoCo。

R02 后 smoke 会实际解包 NPZ 并核对 payload["config"] 与命令行/override
一致，因此这里必须写出与 ``write_v12_r1d_artifacts`` 同构的最小产物；
``--ball-speed`` 为 7 时物理命中（active），其余为 miss，便于区分 case。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

JOINT_NAMES = [f"r_joint{i}" for i in range(1, 7)]
OVERRIDE_SECTIONS = {
    "mpc": "mpc_config",
    "solver": "solver_config",
    "cost": "cost_config",
    "perception": "perception_config",
    "hitting": "hitting_config",
    "serve": "serve_config",
}


def _coerce(text: str):
    """把 override 文本还原为 bool/int/float/str（与 _format_override 对称）。"""
    token = text.strip()
    lowered = token.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _apply_override(config: dict, key: str, text: str) -> None:
    """按 smoke/v12_config 的点路径语义写 config（与写端 _set_value 对齐）。"""
    if key == "seed":
        config["seed"] = int(_coerce(text))
        return
    if key == "actuator.mode":
        config["actuator_mode"] = text.strip()
        return
    if key == "actuator.feedforward":
        config["feedforward"] = bool(_coerce(text))
        return
    if key in ("actuator.kp", "actuator.kd"):
        config[key.rsplit(".", 1)[1]] = float(_coerce(text))
        return
    section, _, rest = key.partition(".")
    if section == "target":
        aliases = {"center": "target_center", "offset": "target_offset"}
        config[aliases.get(rest, rest)] = _coerce(text)
        return
    if section == "serve" and rest in ("use_bounce", "bounce_restitution"):
        config["scene_config"][rest] = bool(_coerce(text))
        return
    node = config[OVERRIDE_SECTIONS[section]]
    parts = rest.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = _coerce(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ball-speed", type=float, default=7.0)
    parser.add_argument("--task-overrides", action="append", default=[])
    args, unknown = parser.parse_known_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mode = "position" if "--position-mode" in unknown else "torque"
    config = {
        "robot_id": "unitree_r1d",
        "actuator_mode": mode,
        "feedforward": (False if "--no-feedforward" in unknown else True)
        if mode == "position" else None,
        "seed": args.seed,
        "serve_config": {"ball_speed": args.ball_speed},
        "mpc_config": {"softmin_beta": 5.0, "space_perturb_m": 0.0,
                       "time_perturb_s": 0.0, "first_plan_iters": 10},
        "solver_config": {"backend": "python"},
        "cost_config": {"tube": {"use_softmin_terminal": True,
                                 "Q_n_tube": 0.0, "corridor_radius": 0.12,
                                 "window_half_ms": 50.0}},
        "perception_config": {"enabled": False},
        "hitting_config": {"hit_direction": "reflect"},
        "scene_config": {"use_bounce": True, "bounce_restitution": 0.75,
                         "shoulder_pos": [0.0, 0.0, 0.0]},
        "target_center": [0.55, 0.0, 1.1],
        "target_offset": [0.0, 0.0, 0.0],
        "workspace_radius": 0.85,
        "kp": 100.0,
        "kd": 2.0,
    }
    for item in args.task_overrides:
        key, _, text = str(item).partition("=")
        _apply_override(config, key, text)
    active = abs(float(config["serve_config"]["ball_speed"]) - 7.0) < 1e-9
    hit_type = "active" if active else "miss"
    result = {
        "hit_type": hit_type,
        "hit_step": 2 if active else None,
        "total_steps": 5,
        "safe_steps": 5,
        "failure_reason": None,
        "first_plan_ms": 1.0,
    }
    payload = {
        "schema": "v12_episode_v2",
        "synthetic": True,
        "robot_id": "unitree_r1d",
        "mode": mode,
        "racket_id": "blade100_v9_reference",
        "joint_names": JOINT_NAMES,
        "seed": args.seed,
        "config": config,
        "hit": {"physical_hit": active, "physical_hit_type": hit_type,
                "legacy_hit": active, "legacy_hit_type": hit_type,
                "physical_hit_step": 2 if active else None,
                "v_racket_at_hit": 1.0 if active else 0.1,
                "ball_out_speed": 8.0 if active else 0.0,
                "ball_out_vy": -1.0 if active else 0.0,
                "contact_events": []},
        "episode": {"total_steps": 5, "safe_steps": 5},
    }
    (output / "v12_episode.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (output / "r1d_episode.json").write_text(
        json.dumps({"schema": "r1d_sim_episode_v1", "synthetic": True,
                    "robot_id": "unitree_r1d", "mode": mode,
                    "seed": args.seed, "hit": {"type": hit_type}},
                   ensure_ascii=False), encoding="utf-8")
    rows_x, rows_u = 6, 5
    base = {
        "X": np.zeros((rows_x, 2 * len(JOINT_NAMES))),
        "U": np.zeros((rows_u, len(JOINT_NAMES))),
        "ball_pos": np.zeros((rows_x, 3)),
        "ball_vel": np.zeros((rows_x, 3)),
    }
    np.savez_compressed(
        output / "v12_episode.npz",
        schema=np.array("v12_episode_v2"),
        time=np.arange(rows_x) * 0.005,
        racket_pos=np.zeros((rows_x, 3)),
        **base)
    np.savez_compressed(
        output / "r1d_episode.npz",
        schema=np.array("r1d_sim_episode_v1"),
        t=np.arange(rows_x) * 0.005,
        **base)
    print("__R1D_RESULT__: " + json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------


def _write_fake_entry(directory: Path) -> Path:
    """写一个只写最小契约产物的假仿真入口（v12 JSON + 兼容占位文件）。"""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "fake_r1d_entry.py"
    script.write_text(FAKE_ENTRY_SOURCE, encoding="utf-8")
    return script


def _write_manifest(
    tmp_path: Path,
    script: Path,
    *,
    cases: list[dict[str, Any]] | None = None,
) -> Path:
    """写最小合法 manifest（默认 exp17d_mechanism 的 torque/position 双模式）。"""
    payload: dict[str, Any] = {
        "schema": "r1d_paper_smoke_v1",
        "description": "M06–M09 验收用最小 manifest",
        "generated_by": "test_r1d_paper_smoke_contract",
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
        "ablation_profiles": {"full": {}, "none": {"cost.tube.use_softmin_terminal": False}},
        "case_defaults": {
            "experiment_id": "{experiment_id}",
            "source_paper_item": (
                "docs/experiments/design/r1d_v12_parity_matrix.md#{experiment_id}"),
            "seed": [100, 101],
            "timeout_s": 60,
            "expected_outputs": [
                "v12_episode.json",
                "r1d_episode.json",
                "r1d_episode.npz",
                "v12_episode.npz",
            ],
            "extract_argv": [
                "{python}", "scripts/extract/extract_r1d_smoke_stats.py",
                "--experiment", "{experiment_id}",
                "--case", "{case_id}",
                "--data-root", "{data_root}",
                "--catalog", "{catalog}",
                "--mode", "{mode}",
                "--output", "{run_root}/stats",
            ],
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
                "id": "exp17d_mechanism",
                "kind": "batch",
                "title": "四档机制消融（双模式最小集）",
                "cases": cases if cases is not None else [
                    {"id": "full", "mode": "torque", "tier": "full",
                     "argv": {"--ball-speed": 9}},
                    {"id": "full_position", "mode": "position", "tier": "full",
                     "argv": {"--ball-speed": 9}},
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


def _run_smoke(
    manifest: Path,
    output: Path,
    data_root: Path,
    *,
    mode: str = "both",
    extra_args: list[str] | None = None,
) -> int:
    """以 tmp 输出根执行 smoke main（真实 run_batch + 假仿真脚本）。"""
    argv = [
        "--manifest", str(manifest),
        "--output", str(output),
        "--data-root", str(data_root),
        "--mode", mode,
        "--timeout", "60",
    ]
    argv.extend(extra_args or [])
    return smoke.main(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 为字典列表。"""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """写 UTF-8 JSON（测试内构造损坏/身份不符产物用）。"""
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _run_argv(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """在仓库根执行 manifest 生成的命令。"""
    return subprocess.run(
        argv, cwd=PROJECT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False)


def _first_task(resolved: dict[str, Any], mode: str) -> dict[str, Any]:
    """取指定模式的第一条任务。"""
    for task in resolved["tasks"]:
        if task["mode"] == mode:
            return task
    raise AssertionError(f"resolved 中没有 mode={mode} 的任务")


# ---------------------------------------------------------------------------
# M07 / M08：真实 manifest 命令契约
# ---------------------------------------------------------------------------


def test_real_manifest_commands_parse_and_match_mode_contract(
        tmp_path: Path) -> None:
    """136 条 extract/plot 全部可解析；plot 无 --catalog；混合实验带 mode。"""
    resolved = smoke.resolve_manifest(
        smoke.load_manifest(REAL_MANIFEST),
        manifest_path=REAL_MANIFEST,
        output=tmp_path / "run",
        data_root=tmp_path / "data",
        mode="both",
    )
    tasks = resolved["tasks"]
    assert len(tasks) == 136, "真实 manifest --mode both 应为 136 条任务"
    assert sum(1 for task in tasks if task["mode"] == "torque") == 126
    assert sum(1 for task in tasks if task["mode"] == "position") == 10

    extract_parser = extractor.build_parser()
    plot_parser = plotter.build_parser()
    run_root = Path(resolved["run_root"])
    plot_total = 0
    mixed_total = 0
    for task in tasks:
        # extract：逐条用真实 argparse 解析（首元素是 python 解释器）。
        extract_args = task["extract_argv"][2:]
        parsed = extract_parser.parse_args(extract_args)
        assert parsed.experiment == task["experiment"]
        assert parsed.case == task["case_id"]
        assert parsed.mode == task["mode"], task["extract_argv"]
        assert parsed.output == run_root / "stats"
        # plot：逐条解析，且命令行不得再出现 --catalog（M07）。
        plot_args = task["plot_argv"][2:]
        plot_parsed = plot_parser.parse_args(plot_args)
        assert plot_parsed.stats_root == run_root / "stats" / task["mode"]
        assert plot_parsed.output == run_root / "figures" / task["mode"]
        assert "--catalog" not in task["plot_argv"], task["plot_argv"]
        assert "{mode}" not in " ".join(task["plot_argv"])
        plot_total += 1
        if task["experiment"] in MIXED_EXPERIMENTS:
            mixed_total += 1
            # M08：混合实验必须显式带各自 mode，禁止合成样本池。
            assert parsed.mode == task["mode"]
    assert plot_total == 136
    assert mixed_total == 32, (
        f"三个混合实验应共 32 条 extract，实际 {mixed_total}")


def test_real_manifest_plot_rejects_catalog_flag(tmp_path: Path) -> None:
    """plot 的 argparse 明确拒绝 --catalog（M07 根因回归）。"""
    parser = plotter.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([
            "--stats-root", str(tmp_path / "stats"),
            "--experiment", "exp17d_mechanism",
            "--case", "full",
            "--output", str(tmp_path / "figures"),
            "--catalog", str(tmp_path / "catalog.json"),
        ])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# M07 / M08：真实 extract → plot（两模式不覆盖）
# ---------------------------------------------------------------------------


def test_extract_then_plot_keeps_modes_separate(tmp_path: Path) -> None:
    """torque/position 各跑一次 extract→plot，输出分居 <mode> 子目录。"""
    script = _write_fake_entry(tmp_path)
    manifest = _write_manifest(tmp_path, script)
    data_root = tmp_path / "csv_root"
    run_root = tmp_path / "run"
    assert _run_smoke(manifest, run_root, data_root) == 0

    resolved = _read_json(run_root / "manifest_resolved.json")
    produced: dict[str, tuple[Path, Path]] = {}
    for mode in ("torque", "position"):
        task = _first_task(resolved, mode)
        extract_proc = _run_argv(task["extract_argv"])
        assert extract_proc.returncode == 0, extract_proc.stderr
        stats_path = (run_root / "stats" / mode / task["experiment"]
                      / f"{task['case_id']}.json")
        assert stats_path.is_file(), extract_proc.stdout + extract_proc.stderr
        assert _read_json(stats_path)["mode"] == mode

        plot_proc = _run_argv(task["plot_argv"])
        assert plot_proc.returncode == 0, plot_proc.stderr
        figure_path = (run_root / "figures" / mode
                       / f"{task['experiment']}__{task['case_id']}.png")
        assert figure_path.is_file(), plot_proc.stdout + plot_proc.stderr
        produced[mode] = (stats_path, figure_path)

    # 两模式的统计与图各自独立存在，互不覆盖。
    assert produced["torque"][0].parent != produced["position"][0].parent
    assert produced["torque"][1].parent != produced["position"][1].parent
    assert all(path.is_file() for pair in produced.values() for path in pair)
    # plot 若被强行追加 --catalog 必须报错（旧 manifest 的回归点）。
    rejected = _run_argv([
        *_first_task(resolved, "torque")["plot_argv"],
        "--catalog", str(run_root / "catalog.json"),
    ])
    assert rejected.returncode == 2
    assert "unrecognized arguments: --catalog" in rejected.stderr


# ---------------------------------------------------------------------------
# M06：整树移动后 catalog 与统计 consumer
# ---------------------------------------------------------------------------


def test_moved_run_root_catalog_and_stats_consumer(tmp_path: Path) -> None:
    """producer 整树移动后 load_catalog 可解析，extract 仍能读 catalog 出统计。"""
    script = _write_fake_entry(tmp_path)
    manifest = _write_manifest(tmp_path, script)
    data_root = tmp_path / "csv_root"
    original = tmp_path / "original_run"
    assert _run_smoke(manifest, original, data_root) == 0

    moved = tmp_path / "moved_run"
    original.rename(moved)
    assert not original.exists()
    assert (moved / "catalog.json").is_file()

    catalog = run_catalog.load_catalog(moved / "catalog.json")
    assert catalog.run_root == moved.resolve()
    rows = _read_csv(moved / "runs.csv")
    assert len(rows) == 4
    for row in rows:
        # runs.csv 内路径相对 run_root；搬移后仍指向新根内的真实文件。
        assert run_catalog.resolve_run_path(catalog, row["log_path"]).is_file()
        assert run_catalog.resolve_run_path(
            catalog, row["artifact_dir"]).is_dir()
    # 兼容旧行的绝对路径也不能被拼到新根下。
    assert run_catalog.main([str(moved / "catalog.json")]) == 0

    # 统计 consumer：用搬移后的 catalog 重新提取一条任务，产物写新根。
    resolved = _read_json(moved / "manifest_resolved.json")
    task = _first_task(resolved, "torque")
    consumer_argv = [
        task["extract_argv"][0],
        "scripts/extract/extract_r1d_smoke_stats.py",
        "--experiment", task["experiment"],
        "--case", task["case_id"],
        "--data-root", str(data_root),
        "--catalog", str(moved / "catalog.json"),
        "--mode", "torque",
        "--output", str(moved / "stats_after_move"),
    ]
    proc = _run_argv(consumer_argv)
    assert proc.returncode == 0, proc.stderr
    target = (moved / "stats_after_move" / "torque" / task["experiment"]
              / f"{task['case_id']}.json")
    assert target.is_file(), proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# R01：搬移后再次 resume 必须按任务身份重解析产物路径并保留旧指标
# ---------------------------------------------------------------------------


def _seven_speed_cases() -> list[dict[str, Any]]:
    """4 个 case × 2 seed：speed7 两模式 + tier=none + speed9（共 8 任务）。"""
    return [
        {"id": "speed7", "mode": "torque", "tier": "full",
         "argv": {"--ball-speed": 7}},
        {"id": "speed7_position", "mode": "position", "tier": "full",
         "argv": {"--ball-speed": 7}},
        {"id": "none_speed7", "mode": "torque", "tier": "none",
         "argv": {"--ball-speed": 7}},
        {"id": "speed9", "mode": "torque", "tier": "full",
         "argv": {"--ball-speed": 9}},
    ]


#: 搬移/续跑回归只比较这些统计字段（不含生成时间戳等易变字段）。
_METRIC_KEYS = (
    "n_valid",
    "n_attempted",
    "n_error",
    "physical_active_rate",
    "physical_active_denominator",
    "physical_hit_semantics_available",
)


def _metric_snapshot(run_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """提取 speed7 两模式的统计关键字段（模拟下游 consumer）。"""
    snapshot: dict[tuple[str, str], dict[str, Any]] = {}
    for case, mode in (("speed7", "torque"), ("speed7_position", "position")):
        stats = extractor.collect_case_stats(
            experiment="exp17d_mechanism", case=case,
            catalog_path=run_root / "catalog.json", mode=mode)
        snapshot[(case, mode)] = {key: stats.get(key) for key in _METRIC_KEYS}
    return snapshot


def _row_for(rows: list[dict[str, str]], case_id: str, seed: str) -> dict[str, str]:
    """按 artifact_dir 尾部 ``<case>/seed<N>`` 取 runs.csv 行。

    ``runs.csv`` 的字段是 batch_runner 的固定 CSV_FIELDS（无 ``case_id``），
    任务身份由 ``config_id``/artifact 目录体现。
    """
    suffix = f"/{case_id}/seed{seed}"
    for row in rows:
        normalized = row["artifact_dir"].replace("\\", "/").rstrip("/")
        if normalized.endswith(suffix):
            return row
    raise AssertionError(f"未找到 case={case_id} seed={seed} 的 runs.csv 行")


def test_moved_resume_reparses_artifact_paths_and_keeps_metrics(
    tmp_path: Path,
) -> None:
    """整树搬移后再次 resume：路径按当前 run_root 解析，physical 指标不清空。"""
    script = _write_fake_entry(tmp_path)
    manifest = _write_manifest(tmp_path, script, cases=_seven_speed_cases())
    data_root = tmp_path / "csv_root"
    original = tmp_path / "original_run"
    assert _run_smoke(manifest, original, data_root) == 0

    before = _metric_snapshot(original)
    assert before[("speed7", "torque")]["n_valid"] == 2
    assert before[("speed7", "torque")]["physical_active_rate"] == 100.0
    identity_before = {row["config_id"]
                       for row in _read_csv(data_root / "exp17d_mechanism"
                                            / "results.csv")}

    moved = tmp_path / "moved_run"
    original.rename(moved)
    # 搬移后、续跑前，统计 consumer 仍能按 catalog 读到同一批指标。
    assert _metric_snapshot(moved) == before

    # 再次 resume：launched=0，但必须重解析产物路径（行内旧绝对路径不可用）。
    assert _run_smoke(manifest, moved, data_root) == 0
    assert _metric_snapshot(moved) == before

    rows = _read_csv(moved / "runs.csv")
    assert len(rows) == 8
    for row in rows:
        artifact_dir = Path(row["artifact_dir"])
        log_path = Path(row["log_path"])
        assert not artifact_dir.is_absolute(), row["artifact_dir"]
        assert not log_path.is_absolute(), row["log_path"]
        assert (moved / artifact_dir).is_dir(), row["artifact_dir"]
        assert (moved / log_path).is_file(), row["log_path"]
        assert row["artifact_error"] == ""
        # config_json 的 --output 也必须指向搬移后的真实产物目录。
        command = json.loads(row["config_json"])
        assert not Path(command["--output"]).is_absolute()
        assert (moved / command["--output"]).is_dir()

    # runs.csv 不得再残留搬移前的绝对 artifact 路径（JSON 会转义反斜杠，
    # 比较前先还原，避免断言因转义而假通过）。
    text = (moved / "runs.csv").read_text(encoding="utf-8")
    normalized = text.replace("\\\\", "\\")
    assert str(original) not in normalized
    assert str(original.resolve()) not in normalized
    identity_after = {row["config_id"]
                      for row in _read_csv(data_root / "exp17d_mechanism"
                                           / "results.csv")}
    assert identity_after == identity_before


def test_moved_resume_missing_artifact_keeps_previous_row(
    tmp_path: Path,
) -> None:
    """搬移后产物真的缺失：保留旧行指标但 exit 1，恢复后 exit 0。"""
    script = _write_fake_entry(tmp_path)
    manifest = _write_manifest(tmp_path, script, cases=_seven_speed_cases())
    data_root = tmp_path / "csv_root"
    moved = tmp_path / "moved_run"
    assert _run_smoke(manifest, moved, data_root) == 0
    before = _metric_snapshot(moved)

    target = (moved / "artifacts" / "exp17d_mechanism" / "speed7"
              / "seed100")
    backup = tmp_path / "backup_speed7_seed100"
    shutil.move(str(target), str(backup))

    failed = _run_smoke(manifest, moved, data_root)
    assert failed == 1
    summary = _read_json(moved / "run_summary.json")
    assert summary["totals"]["artifact_errors"] >= 1

    rows = _read_csv(moved / "runs.csv")
    assert len(rows) == 8
    retained = _row_for(rows, "speed7", "100")
    assert retained["physical_hit"] == "True"
    assert retained["physical_hit_type"] == "active"
    assert retained["artifact_error"] == ""
    # 下游统计不得因产物缺失把已有效的 physical 行从分母里丢掉。
    assert _metric_snapshot(moved) == before

    shutil.move(str(backup), str(target))
    assert _run_smoke(manifest, moved, data_root) == 0
    assert _metric_snapshot(moved) == before
    assert _row_for(_read_csv(moved / "runs.csv"), "speed7",
                    "100")["artifact_error"] == ""


# ---------------------------------------------------------------------------
# M09：已完成任务的产物/日志校验
# ---------------------------------------------------------------------------


@pytest.fixture()
def produced_run(tmp_path: Path) -> Iterator[dict[str, Any]]:
    """跑一次双模式 producer，返回复现 M09 所需路径。"""
    script = _write_fake_entry(tmp_path)
    manifest = _write_manifest(tmp_path, script)
    data_root = tmp_path / "csv_root"
    run_root = tmp_path / "run"
    assert _run_smoke(manifest, run_root, data_root) == 0
    rows = _read_csv(data_root / "exp17d_mechanism" / "results.csv")
    assert len(rows) == 4
    v12_path = (run_root / "artifacts" / "exp17d_mechanism" / "full"
                / "seed100" / "v12_episode.json")
    assert v12_path.is_file()
    log_path = Path(rows[0]["log_path"])
    assert log_path.is_file()
    yield {
        "manifest": manifest,
        "run_root": run_root,
        "data_root": data_root,
        "results_csv": data_root / "exp17d_mechanism" / "results.csv",
        "v12_path": v12_path,
        "log_path": log_path,
        "log_config_id": rows[0]["config_id"],
    }


def _break_and_restore(
    produced: dict[str, Any],
    tmp_path: Path,
    kind: str,
) -> tuple[Callable[[], None], str]:
    """制造一类产物问题并返回恢复函数与期望的 stderr 关键词。"""
    v12_path: Path = produced["v12_path"]
    backup_dir = tmp_path / f"backup_{kind}"
    backup_dir.mkdir()
    if kind == "missing":
        shutil.move(str(v12_path), str(backup_dir / v12_path.name))
        return (lambda: shutil.move(str(backup_dir / v12_path.name), str(v12_path)),
                "缺少产物 v12_episode.json")
    if kind == "corrupt":
        shutil.copy2(v12_path, backup_dir / v12_path.name)
        v12_path.write_text("{not a json", encoding="utf-8")
        return (lambda: shutil.copy2(backup_dir / v12_path.name, v12_path),
                "v12_episode.json 无法解析")
    if kind == "identity":
        payload = _read_json(v12_path)
        shutil.copy2(v12_path, backup_dir / v12_path.name)
        payload["seed"] = 999
        _write_json(v12_path, payload)
        return (lambda: shutil.copy2(backup_dir / v12_path.name, v12_path),
                "v12_episode.json seed 不符")
    if kind == "config_seed":
        # R02：顶层 seed 正确但产物内 config.seed 与任务不一致，同样必须失败。
        payload = _read_json(v12_path)
        shutil.copy2(v12_path, backup_dir / v12_path.name)
        payload["config"]["seed"] = 999
        _write_json(v12_path, payload)
        return (lambda: shutil.copy2(backup_dir / v12_path.name, v12_path),
                "config.seed 不符")
    if kind == "log":
        log_path: Path = produced["log_path"]
        shutil.move(str(log_path), str(backup_dir / log_path.name))
        return (lambda: shutil.move(str(backup_dir / log_path.name), str(log_path)),
                "日志不存在")
    if kind == "r1d_missing":
        # expected_outputs 中的 R1D 兼容产物缺失同样必须失败（M09 契约）。
        r1d_path = v12_path.with_name("r1d_episode.npz")
        shutil.move(str(r1d_path), str(backup_dir / r1d_path.name))
        return (lambda: shutil.move(str(backup_dir / r1d_path.name), str(r1d_path)),
                "缺少产物 r1d_episode.npz")
    if kind == "npz_truncated":
        # R02：截断 NPZ 必须真正 numpy.load 才会暴露为 BadZipFile。
        npz_path = v12_path.with_name("v12_episode.npz")
        shutil.copy2(npz_path, backup_dir / npz_path.name)
        npz_path.write_bytes(npz_path.read_bytes()[:24])
        return (lambda: shutil.copy2(backup_dir / npz_path.name, npz_path),
                "v12_episode.npz 无法加载")
    if kind == "log_empty":
        # R02：日志存在但 0 字节同样视为未完成。
        log_path = produced["log_path"]
        shutil.copy2(log_path, backup_dir / log_path.name)
        log_path.write_bytes(b"")
        return (lambda: shutil.copy2(backup_dir / log_path.name, log_path),
                "日志为空")
    if kind == "log_config_id":
        # R02：日志头部 config_id 必须与 manifest 任务一致，防止错配日志。
        log_path = produced["log_path"]
        shutil.copy2(log_path, backup_dir / log_path.name)
        text = log_path.read_text(encoding="utf-8")
        expected = f"# config_id: {produced['log_config_id']}"
        assert expected in text, text[:200]
        log_path.write_text(
            text.replace(expected, "# config_id: stale#config#id"),
            encoding="utf-8")
        return (lambda: shutil.copy2(backup_dir / log_path.name, log_path),
                "日志缺少 config_id 头部")
    if kind == "r1d_npz_corrupt":
        # R02：R1D 兼容 NPZ 截断时必须报错（不能只查文件存在）。
        r1d_path = v12_path.with_name("r1d_episode.npz")
        shutil.copy2(r1d_path, backup_dir / r1d_path.name)
        r1d_path.write_bytes(b"PK\x03\x04truncated")
        return (lambda: shutil.copy2(backup_dir / r1d_path.name, r1d_path),
                "r1d_episode.npz 无法加载")
    raise AssertionError(f"未知问题类型: {kind}")


@pytest.mark.parametrize(
    "kind", [
        "missing", "corrupt", "identity", "config_seed", "log", "r1d_missing",
        "npz_truncated", "log_empty", "log_config_id", "r1d_npz_corrupt",
    ])
def test_resume_detects_broken_completed_artifacts(
    produced_run: dict[str, Any],
    tmp_path: Path,
    kind: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """产物缺失/损坏/身份不符/日志缺失 → exit 1 + 恢复指引；修复后 exit 0。"""
    produced = produced_run
    restore, keyword = _break_and_restore(produced, tmp_path, kind)
    capsys.readouterr()

    failed = _run_smoke(
        produced["manifest"], produced["run_root"], produced["data_root"],
        mode="both")
    captured = capsys.readouterr()
    assert failed == 1, captured.out + captured.err
    assert "续跑产物校验失败" in captured.err
    assert "恢复指引" in captured.err
    assert keyword in captured.err, captured.err
    assert "launched=0" in captured.out or "launched=0" in captured.err
    # 失败会计入 run_summary.totals.artifact_errors（供 CI/批次门禁用）。
    failed_summary = _read_json(produced["run_root"] / "run_summary.json")
    assert failed_summary["totals"]["artifact_errors"] >= 1
    # 失败不删行：旧 runs.csv 行全部保留。
    assert len(_read_csv(produced["run_root"] / "runs.csv")) == 4

    restore()
    recovered = _run_smoke(
        produced["manifest"], produced["run_root"], produced["data_root"],
        mode="both")
    recovered_out = capsys.readouterr()
    assert recovered == 0, recovered_out.out + recovered_out.err
    recovered_summary = _read_json(produced["run_root"] / "run_summary.json")
    assert recovered_summary["totals"]["artifact_errors"] == 0
    assert len(_read_csv(produced["run_root"] / "runs.csv")) == 4


def test_resume_valid_artifacts_is_idempotent(
    produced_run: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """产物齐全时正常 resume：不再启动任务，行数/指标不变，无 artifact 错误。"""
    produced = produced_run
    before = _read_csv(produced["run_root"] / "runs.csv")
    capsys.readouterr()
    assert _run_smoke(
        produced["manifest"], produced["run_root"], produced["data_root"],
        mode="both") == 0
    captured = capsys.readouterr()
    assert "launched=0" in captured.out or "launched=0" in captured.err
    summary = _read_json(produced["run_root"] / "run_summary.json")
    assert summary["totals"]["artifact_errors"] == 0
    after = _read_csv(produced["run_root"] / "runs.csv")
    assert len(after) == len(before) == 4
    assert [row["config_id"] for row in after] == [
        row["config_id"] for row in before]
    assert all(row["artifact_error"] == "" for row in after)


def test_speed9_json_in_speed7_directory_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """同 robot/mode/seed 但 ball_speed=9 的 JSON 放入 speed7 目录必须失败。"""
    cases = [
        {"id": "speed7", "mode": "torque", "tier": "full",
         "argv": {"--ball-speed": 7}},
        {"id": "speed9", "mode": "torque", "tier": "full",
         "argv": {"--ball-speed": 9}},
    ]
    script = _write_fake_entry(tmp_path)
    manifest = _write_manifest(tmp_path, script, cases=cases)
    data_root = tmp_path / "csv_root"
    run_root = tmp_path / "run"
    assert _run_smoke(manifest, run_root, data_root) == 0

    def speed7_stats() -> dict[str, Any]:
        return extractor.collect_case_stats(
            experiment="exp17d_mechanism", case="speed7",
            catalog_path=run_root / "catalog.json", mode="torque")

    def stable(stats: dict[str, Any]) -> dict[str, Any]:
        # generated_utc 每次统计都会刷新，不属于指标本体。
        return {key: value for key, value in stats.items()
                if key != "generated_utc"}

    stats_before = speed7_stats()
    assert stats_before["n_valid"] == 2
    assert stats_before["physical_active_rate"] == 100.0

    speed7_json = (run_root / "artifacts" / "exp17d_mechanism" / "speed7"
                   / "seed100" / "v12_episode.json")
    speed9_json = (run_root / "artifacts" / "exp17d_mechanism" / "speed9"
                   / "seed100" / "v12_episode.json")
    backup = tmp_path / "speed7_v12_episode.json"
    shutil.copy2(speed7_json, backup)
    # 只换 JSON：robot/mode/seed 全同，唯一差异是 serve_config.ball_speed。
    shutil.copy2(speed9_json, speed7_json)

    capsys.readouterr()
    assert _run_smoke(manifest, run_root, data_root) == 1
    captured = capsys.readouterr()
    assert "config.serve_config.ball_speed 不符" in captured.err, captured.err
    summary = _read_json(run_root / "run_summary.json")
    assert summary["totals"]["artifact_errors"] >= 1

    rows = _read_csv(run_root / "runs.csv")
    assert len(rows) == 4
    retained = _row_for(rows, "speed7", "100")
    assert retained["physical_hit"] == "True"
    assert retained["physical_hit_type"] == "active"
    assert retained["artifact_error"] == ""
    # 统计 consumer 会优先读产物 JSON 的判决（此处是错配的 miss），但错配行
    # 不得因 artifact_error 被丢弃：分母仍为 2、行仍有效。
    stats_after = speed7_stats()
    assert stats_after["n_valid"] == 2
    assert stats_after["physical_active_denominator"] == 2

    shutil.copy2(backup, speed7_json)
    assert _run_smoke(manifest, run_root, data_root) == 0
    assert stable(speed7_stats()) == stable(stats_before)


def test_v12_config_issues_maps_override_paths() -> None:
    """R02：覆盖项按点路径映射到 config，任一值不符都要报问题。"""
    payload = {"config": {
        "actuator_mode": "position",
        "feedforward": False,
        "serve_config": {"ball_speed": 7.0},
        "scene_config": {"use_bounce": True},
        "mpc_config": {"space_perturb_m": 0.1},
        "cost_config": {"tube": {"use_softmin_terminal": False}},
    }}
    task: dict[str, Any] = {
        "mode": "position",
        "argv": {
            "--ball-speed": 7,
            "--position-mode": None,
            "--no-feedforward": None,
            "--task-overrides": [
                "actuator.feedforward=false",
                "mpc.space_perturb_m=0.1",
                "cost.tube.use_softmin_terminal=false",
                "serve.use_bounce=true",
            ],
        },
    }
    assert smoke._v12_config_issues(payload, task) == []

    wrong_value = {**task, "argv": {
        **task["argv"],
        "--task-overrides": ["mpc.space_perturb_m=0.2"],
    }}
    issues = smoke._v12_config_issues(payload, wrong_value)
    assert any("mpc_config.space_perturb_m" in issue for issue in issues), issues

    wrong_mode = {
        "mode": "position",
        "argv": {"--position-mode": None},
    }
    issues = smoke._v12_config_issues(
        {"config": {**payload["config"], "actuator_mode": "torque"}}, wrong_mode)
    assert any("config.actuator_mode 不符" in issue for issue in issues), issues
