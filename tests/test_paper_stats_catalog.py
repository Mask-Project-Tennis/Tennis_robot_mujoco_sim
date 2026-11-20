"""D 线回归：统计/提取链路的 catalog 定位、模式分池、E3 来源与 E8 不可估计。

覆盖 ``docs/r1d_v12_parallel_repair_plan.md`` §9 的 D 任务：

1. **catalog 消费端（F03）**：``--catalog`` 能定位 output 与 data-root 分离、跨目录
   （含绝对路径在 run_root 之外）的产物；整棵数据目录移动后仍可解析；显式坏 catalog
   必须非零退出且不回退旧逻辑。
2. **模式混淆（F04）**：torque 2 active / position 2 miss 必须分池得到 100% 与 0%；
   ``--mode both`` 写两个模式目录与 ``modes.json`` 索引；混合模式未指定 ``--mode``
   非零退出；单一来源内重复记录报错；``runs.csv`` 富化视图与 ``results.csv`` 基础视图
   按身份一对一连接、不重复计数。
3. **E3 来源（F06）**：只有 ``source_sets.nominal_mechanism`` 声明的 ``exp17d_mechanism``
   四档也能产出完整 E3；缺 ``exp17a_noise`` 不阻断；无 catalog 的 RM65 旧来源保留。
4. **E8 全 miss（F07）**：physical DiD 全 miss 时保留样本数与观察差（可 0），推断字段为
   ``null`` 并给 ``unavailable_reason="all_miss"``；混合 miss/hit 正常估计；
   ``paper_figs.py --hit-semantics physical`` 在 TeX 里显示 ``--`` 且保留原因。

所有合成数据都写进 ``pytest`` 的临时目录，并显式标 ``synthetic``；不读写
``experiment_data/`` 下的真实数据。
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
EXTRACT_DIR = ROOT / "scripts" / "extract"
sys.path.insert(0, str(EXTRACT_DIR))

import paired_stats  # noqa: E402  （被测模块，直接测内部语义）

PYTHON = sys.executable

ROBOT_ID = "unitree_r1d"
RACKET_ID = "blade100_v9_reference"
METRIC_SCHEMA = "v12_episode_v2_physical_hit"
GENERATED_UTC = "2026-11-20T00:00:00+00:00"

# 与真实 ``experiment_data/r1d_v12_parity/paper_smoke/runs.csv`` 一致的表头顺序。
RUNS_FIELDS: tuple[str, ...] = (
    "experiment", "config_id", "config_json", "seed", "hit", "hit_type",
    "pos_error", "vel_error", "min_dist", "ball_near_ms", "tube_ready_ms",
    "max_tcp", "max_qdot", "max_face", "hit_time_error_ms", "hit_pos_error",
    "v_racket_at_hit", "wall_time", "first_plan_ms", "ball_out_speed",
    "ball_out_vy", "error", "robot_id", "mode", "racket_id", "config_hash",
    "metric_schema", "timeout_s", "timed_out", "exit_code", "failure_reason",
    "total_steps", "safe_steps", "hit_step", "artifact_dir", "log_path",
    "physical_hit", "physical_hit_type", "physical_hit_step",
    "contact_event_count", "artifact_error", "hit_semantics",
)
# ``results.csv`` 只有基础字段（physical 列在 runs.csv 富化视图里）。
RESULT_FIELDS: tuple[str, ...] = tuple(
    field for field in RUNS_FIELDS
    if field not in {
        "physical_hit", "physical_hit_type", "physical_hit_step",
        "contact_event_count", "artifact_error", "hit_semantics",
    }
)

# R1D 消融档 → ``--task-overrides``（与 manifest 的 ablation_profiles 一致）。
TIER_OVERRIDES: dict[str, list[str] | None] = {
    "full": None,
    "none": [
        "cost.tube.Q_n_tube=0.0",
        "cost.tube.Q_p_tube=0.0",
        "cost.tube.Q_v_tube=0.0",
        "cost.tube.use_softmin_terminal=false",
    ],
    "tube_only": ["cost.tube.use_softmin_terminal=false"],
    "softmin_only": [
        "cost.tube.Q_n_tube=0.0",
        "cost.tube.Q_p_tube=0.0",
        "cost.tube.Q_v_tube=0.0",
    ],
}


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _run_script(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """用当前解释器跑真实 CLI，返回完整进程结果（stdout/stderr/returncode）。"""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [PYTHON, *args],
        cwd=str(cwd or ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Iterable[Mapping[str, str]], fields: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return path


def _append_csv_rows(path: Path, rows: Iterable[Mapping[str, str]], fields: Sequence[str]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        for row in rows:
            writer.writerow(dict(row))


def _rel(path: Path, root: Path) -> str:
    """把 path 表示为相对 root 的 POSIX 路径（跨目录时用 ``../``）。"""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return Path(os.path.relpath(path, root)).as_posix()


def _paired_stats(*args: str) -> subprocess.CompletedProcess[str]:
    return _run_script("scripts/extract/paired_stats.py", *args)


def _extract_case(*args: str) -> subprocess.CompletedProcess[str]:
    return _run_script("scripts/extract/extract_r1d_smoke_stats.py", *args)


def _paper_figs(*args: str) -> subprocess.CompletedProcess[str]:
    return _run_script("scripts/plot/paper_figs.py", *args)


def _stats_payload(out_dir: Path, mode: str | None = None) -> dict[str, Any]:
    """读取 paired_stats 写出的统计 JSON（可分模式目录）。"""
    base = out_dir / mode if mode else out_dir
    return _read_json(base / "paper_stats_active.json")


def _e3_cell(stats: Mapping[str, Any], key: str, *, physical: bool = False) -> Mapping[str, Any]:
    section = stats["physical"]["E3_nominal"] if physical else stats["E3_nominal"]
    return section["cells"][key]


def _e8_section(stats: Mapping[str, Any], *, physical: bool = False) -> Mapping[str, Any]:
    return stats["physical"]["E8_limit_mechanism"] if physical else stats["E8_limit_mechanism"]


# ---------------------------------------------------------------------------
# 合成数据根（run_root / data_root 双根，相对路径 + catalog）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """一个 (experiment, case, mode, tier, seed) 任务的合成描述。"""

    experiment: str
    case: str
    mode: str = "torque"
    tier: str = "full"
    seeds: tuple[int, ...] = (100, 101)
    physical: str | None = "active"
    legacy: str | None = "active"
    speed: int = 7
    tcp: str = "TCP1.8"
    error: str = ""

    @property
    def config_id(self) -> str:
        return f"{self.case}__{self.mode}__{self.tier}__s{self.speed}__{self.tcp.lower()}"


def _cfg(spec: Spec, artifact: Path) -> dict[str, Any]:
    """按真实 R1D argv 结构造 config_json（含 identity 之外的运行期键）。"""
    cfg: dict[str, Any] = {
        "--ball-speed": spec.speed,
        "--no-plot": None,
        "--racket-config": "configs/rackets/blade100_v9_reference.yaml",
        "--robot-config": "configs/robots/r1d_right.yaml",
        "--runtime-config": "configs/robots/r1d_v12_runtime_right.yaml",
        "--serve-box": None,
        "--output": str(artifact),
        "_config_hash": "0123456789abcdef",
        "_timeout": 900.0,
    }
    if spec.mode == "position":
        cfg["--position-mode"] = None
    overrides = TIER_OVERRIDES[spec.tier]
    if overrides:
        cfg["--task-overrides"] = list(overrides)
    if spec.tcp == "TCP1.0":
        cfg["--limits-config"] = "configs/real_robot.yaml"
    return cfg


def _argv(spec: Spec, artifact: Path, seed: int) -> list[str]:
    argv = [
        "scripts/rm65_mpc_v12.py",
        "--robot-config", "configs/robots/r1d_right.yaml",
        "--runtime-config", "configs/robots/r1d_v12_runtime_right.yaml",
        "--serve-box", "--no-plot",
        "--ball-speed", str(spec.speed),
        "--output", str(artifact),
        "--seed", str(seed),
    ]
    if spec.mode == "position":
        argv.append("--position-mode")
    if spec.tcp == "TCP1.0":
        argv.extend(["--limits-config", "configs/real_robot.yaml"])
    return argv


def _v12_payload(spec: Spec, seed: int) -> dict[str, Any]:
    """v12_episode_v2 最小产物：physical/legacy 判定与身份字段齐全。"""
    hit: dict[str, Any] = {
        "legacy_hit": spec.legacy in {"active", "passive"},
        "legacy_hit_type": spec.legacy or "",
        "pos_error": 0.012,
        "v_racket_at_hit": 0.71,
        "contact_events": [],
    }
    if spec.physical is not None:
        hit["physical_hit"] = spec.physical in {"active", "passive"}
        hit["physical_hit_type"] = spec.physical
        hit["physical_hit_step"] = 88 if spec.physical == "active" else -1
        if spec.physical == "active":
            hit["contact_events"] = [{"distance_m": -0.0001, "step": 88, "kind": "physical"}]
    payload: dict[str, Any] = {
        "schema": "v12_episode_v2",
        "generated_utc": GENERATED_UTC,
        "robot_id": ROBOT_ID,
        "mode": spec.mode,
        "racket_id": RACKET_ID,
        "seed": seed,
        "backend": "python",
        "failure_reason": "" if not spec.error else "safety_filter_stop",
        "metric_schema": METRIC_SCHEMA,
        "episode": {"total_steps": 90, "safe_steps": 90},
        "hit": hit,
    }
    return payload


def _row_pair(
    spec: Spec,
    seed: int,
    artifact: Path,
    run_root: Path,
    *,
    absolute_artifact: bool,
) -> tuple[dict[str, str], dict[str, str]]:
    """同一记录的两份视图：results.csv 基础行与 runs.csv 富化行。"""
    artifact_text = str(artifact) if absolute_artifact else _rel(artifact, run_root)
    log_text = (
        str(artifact.parent.parent.parent / "logs" / f"{seed}.log")
        if absolute_artifact
        else f"logs/{spec.experiment}/{spec.case}/seed{seed}.log"
    )
    common = {
        "experiment": spec.experiment,
        "config_id": spec.config_id,
        "config_json": json.dumps(_cfg(spec, artifact), ensure_ascii=False, sort_keys=True),
        "seed": str(seed),
        "hit": "True" if spec.legacy in {"active", "passive"} else "False",
        "hit_type": spec.legacy or "",
        "pos_error": "0.012",
        "vel_error": "0.21",
        "min_dist": "0.02",
        "ball_near_ms": "12",
        "tube_ready_ms": "20",
        "max_tcp": "1.2",
        "max_qdot": "3.0",
        "max_face": "0.7",
        "hit_time_error_ms": "1.5",
        "hit_pos_error": "0.008",
        "v_racket_at_hit": "0.71",
        "wall_time": "11.5",
        "first_plan_ms": "120.0",
        "ball_out_speed": "5.5",
        "ball_out_vy": "1.2",
        "error": spec.error,
        "robot_id": ROBOT_ID,
        "mode": spec.mode,
        "racket_id": RACKET_ID,
        "config_hash": "0123456789abcdef",
        "metric_schema": METRIC_SCHEMA,
        "timeout_s": "900",
        "timed_out": "False",
        "exit_code": "0" if not spec.error else "3",
        "failure_reason": "safety_filter_stop" if spec.physical == "miss" else "",
        "total_steps": "90",
        "safe_steps": "90",
        "hit_step": "88",
        "artifact_dir": artifact_text,
        "log_path": log_text,
    }
    runs = dict(common)
    runs.update(
        {
            "physical_hit": (
                "" if spec.physical is None
                else "True" if spec.physical in {"active", "passive"} else "False"
            ),
            "physical_hit_type": spec.physical or "",
            "physical_hit_step": (
                "" if spec.physical is None
                else "88" if spec.physical == "active" else "-1"
            ),
            "contact_event_count": (
                "" if spec.physical is None else "1" if spec.physical == "active" else "0"
            ),
            "artifact_error": "",
            "hit_semantics": "physical",
        }
    )
    return common, runs


class Dataset:
    """最小 R1D 数据根：``<root>/run``（artifact/runs/manifest/catalog）+ ``<root>/data/batch``。"""

    def __init__(
        self,
        root: Path,
        *,
        source_sets: Mapping[str, str] | None = None,
        synthetic: bool = True,
    ) -> None:
        self.root = root
        self.run_root = root / "run"
        self.data_root = root / "data" / "batch"
        self.source_sets: dict[str, str] = dict(
            source_sets
            or {"nominal_mechanism": "exp17d_mechanism", "noise": "exp17a_noise"}
        )
        self.synthetic = synthetic
        self.specs: list[Spec] = []
        self.results_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.runs_rows: list[dict[str, str]] = []
        self.tasks: list[dict[str, Any]] = []
        self.catalog_path = self.run_root / "catalog.json"
        self.manifest_path = self.run_root / "manifest_resolved.json"

    # -- 构建 ---------------------------------------------------------------
    def add(self, spec: Spec) -> "Dataset":
        self.specs.append(spec)
        return self

    def artifact_dir(self, spec: Spec, seed: int, artifact_root: Path | None = None) -> Path:
        base = artifact_root or self.run_root / "artifacts"
        return base / spec.experiment / spec.case / f"seed{seed}"

    def write(self, *, artifact_root: Path | None = None, write_catalog: bool = True) -> "Dataset":
        absolute_artifact = artifact_root is not None
        for spec in self.specs:
            for seed in spec.seeds:
                artifact = self.artifact_dir(spec, seed, artifact_root)
                _write_json(artifact / "v12_episode.json", _v12_payload(spec, seed))
                result_row, runs_row = _row_pair(
                    spec, seed, artifact, self.run_root, absolute_artifact=absolute_artifact,
                )
                self.results_rows[spec.experiment].append(result_row)
                self.runs_rows.append(runs_row)
                self.tasks.append(
                    self._task(spec, seed, artifact, absolute_artifact=absolute_artifact)
                )
        for experiment, rows in self.results_rows.items():
            _write_csv(self.data_root / experiment / "results.csv", rows, RESULT_FIELDS)
        _write_csv(self.run_root / "runs.csv", self.runs_rows, RUNS_FIELDS)
        _write_json(self.manifest_path, self._manifest())
        if write_catalog:
            _write_json(self.catalog_path, self._catalog())
        return self

    def _task(
        self, spec: Spec, seed: int, artifact: Path, *, absolute_artifact: bool,
    ) -> dict[str, Any]:
        artifact_text = str(artifact) if absolute_artifact else _rel(artifact, self.run_root)
        return {
            "experiment": spec.experiment,
            "experiment_id": spec.experiment,
            "case_id": spec.case,
            "config_id": spec.config_id,
            "seed": seed,
            "mode": spec.mode,
            "tier": spec.tier,
            "identity": {
                "robot_id": ROBOT_ID,
                "racket_id": RACKET_ID,
                "config_hash": "0123456789abcdef",
                "metric_schema": METRIC_SCHEMA,
            },
            "config_hash": "0123456789abcdef",
            "artifact_dir": artifact_text,
            "log_path": (
                str(artifact.parent.parent.parent / "logs" / f"{seed}.log")
                if absolute_artifact else f"logs/{spec.experiment}/{spec.case}/seed{seed}.log"
            ),
            "case_status": "ok",
            "timeout_s": 900,
            "argv": _argv(spec, artifact, seed),
            "extract_argv": [
                "scripts/extract/extract_r1d_smoke_stats.py",
                "--experiment", spec.experiment,
                "--case", spec.case,
                "--data-root", str(self.data_root),
                "--catalog", str(self.catalog_path),
                "--mode", spec.mode,
                "--output", str(self.run_root / "stats"),
            ],
            "plot_argv": [],
            "expected_outputs": [],
            "runtime": {},
            "source_paper_item": "",
        }

    def _manifest(self) -> dict[str, Any]:
        return {
            "schema": "r1d_paper_smoke_resolved_v1",
            "generated_utc": GENERATED_UTC,
            "source_manifest": {"path": "configs/experiments/r1d_paper_smoke.yaml"},
            "script": "scripts/exp/run_r1d_paper_smoke.py",
            "run_root": ".",
            "data_root": _rel(self.data_root, self.run_root),
            "configs": {"robot": "configs/robots/r1d_right.yaml"},
            "config_hash": "0123456789abcdef",
            "identity": {
                "robot_id": ROBOT_ID,
                "mode": "",
                "racket_id": RACKET_ID,
                "config_hash": "0123456789abcdef",
                "metric_schema": METRIC_SCHEMA,
            },
            "seeds": [100, 101],
            "timeout_s_default": 900,
            "mode_filter": "both",
            "limit": None,
            "allow_single_seed": False,
            "units": {"experiment": "experiment", "case": "case", "seed": "seed"},
            "tasks": self.tasks,
            "blocked": [],
            "counts": {"tasks": len(self.tasks)},
        }

    def _catalog(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "synthetic": self.synthetic,
            "robot_id": ROBOT_ID,
            "run_root": ".",
            "data_root": _rel(self.data_root, self.run_root),
            "runs_csv": "runs.csv",
            "manifest_path": "manifest_resolved.json",
            "path_base": "run_root",
            "source_sets": dict(self.source_sets),
            "generated_utc": GENERATED_UTC,
        }


# ---------------------------------------------------------------------------
# F03：catalog 消费端
# ---------------------------------------------------------------------------


class TestCatalogConsumer:
    """两个提取入口读 ``--catalog``：分离根、移动目录、坏 catalog 不回退。"""

    def test_extract_per_case_with_catalog_and_separated_roots(self, tmp_path: Path) -> None:
        dataset = Dataset(tmp_path / "root").add(
            Spec("exp15_speed_v2", "speed7", mode="torque", physical="active", legacy="active")
        ).write()
        out = tmp_path / "out"
        proc = _extract_case(
            "--experiment", "exp15_speed_v2",
            "--case", "speed7",
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--mode", "torque",
            "--output", str(out),
        )
        assert proc.returncode == 0, proc.stderr
        payload = _read_json(out / "torque" / "exp15_speed_v2" / "speed7.json")
        summary = payload["summary"]
        assert payload["synthetic"] is True
        assert payload["mode"] == "torque"
        assert summary["physical_active_denominator"] == 2
        assert summary["physical_active_rate"] == 100.0
        assert summary["n_missing"] == 0
        assert not summary["unavailable_reason"]
        for entry in payload["per_seed"]:
            assert entry["status"] == "ok"
            assert entry["artifact_files"], entry
            assert entry["source"] == "artifact"

    def test_extract_per_case_keeps_legacy_layout_without_explicit_mode(
        self, tmp_path: Path,
    ) -> None:
        dataset = Dataset(tmp_path / "root").add(
            Spec("exp15_speed_v2", "speed7", mode="torque")
        ).write()
        out = tmp_path / "out"
        proc = _extract_case(
            "--experiment", "exp15_speed_v2",
            "--case", "speed7",
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--output", str(out),
        )
        assert proc.returncode == 0, proc.stderr
        assert (out / "exp15_speed_v2" / "speed7.json").is_file()
        assert not (out / "torque").exists()
        payload = _read_json(out / "exp15_speed_v2" / "speed7.json")
        assert payload["mode"] == "torque"

    def test_extract_per_case_mixed_modes_requires_mode(self, tmp_path: Path) -> None:
        dataset = Dataset(tmp_path / "root")
        dataset.add(Spec("exp15_speed_v2", "speed7", mode="torque"))
        dataset.add(Spec("exp15_speed_v2", "speed7_position", mode="position"))
        dataset.write()
        out = tmp_path / "out"
        proc = _extract_case(
            "--experiment", "exp15_speed_v2",
            "--case", "speed7_position",
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--output", str(out),
        )
        assert proc.returncode != 0
        stderr = proc.stderr
        assert "--mode" in stderr
        assert "torque" in stderr and "position" in stderr
        assert not (out / "exp15_speed_v2" / "speed7_position.json").exists()

    def test_extract_bad_catalog_exits_without_fallback(self, tmp_path: Path) -> None:
        dataset = Dataset(tmp_path / "root").add(
            Spec("exp15_speed_v2", "speed7", mode="torque")
        ).write()
        broken = dataset.catalog_path
        payload = _read_json(broken)
        payload["source_sets"] = {}
        _write_json(broken, payload)
        out = tmp_path / "out"
        proc = _extract_case(
            "--experiment", "exp15_speed_v2",
            "--case", "speed7",
            "--data-root", str(dataset.data_root),
            "--catalog", str(broken),
            "--mode", "torque",
            "--output", str(out),
        )
        assert proc.returncode != 0
        assert "catalog" in proc.stderr.lower() or "source_sets" in proc.stderr
        assert not (out / "torque").exists()

    def test_extract_survives_moving_whole_dataset_tree(self, tmp_path: Path) -> None:
        built = Dataset(tmp_path / "first" / "root").add(
            Spec("exp15_speed_v2", "speed7", mode="torque")
        ).write()
        moved_root = tmp_path / "second" / "root"
        moved_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(built.root), str(moved_root))
        out = tmp_path / "out"
        proc = _extract_case(
            "--experiment", "exp15_speed_v2",
            "--case", "speed7",
            "--catalog", str(moved_root / "run" / "catalog.json"),
            "--mode", "torque",
            "--output", str(out),
        )
        assert proc.returncode == 0, proc.stderr
        payload = _read_json(out / "torque" / "exp15_speed_v2" / "speed7.json")
        assert payload["summary"]["physical_active_rate"] == 100.0
        assert payload["data_root"] == str(moved_root / "data" / "batch")

    def test_extract_resolves_absolute_artifact_outside_run_root(self, tmp_path: Path) -> None:
        dataset = Dataset(tmp_path / "root").add(
            Spec("exp15_speed_v2", "speed7", mode="torque")
        ).write(artifact_root=tmp_path / "external_artifacts")
        out = tmp_path / "out"
        proc = _extract_case(
            "--experiment", "exp15_speed_v2",
            "--case", "speed7",
            "--catalog", str(dataset.catalog_path),
            "--mode", "torque",
            "--output", str(out),
        )
        assert proc.returncode == 0, proc.stderr
        payload = _read_json(out / "torque" / "exp15_speed_v2" / "speed7.json")
        assert payload["summary"]["physical_active_denominator"] == 2
        assert payload["summary"]["physical_active_rate"] == 100.0

    def test_paired_stats_consumes_catalog_and_reports_validation(self, tmp_path: Path) -> None:
        dataset = Dataset(tmp_path / "root").add(
            Spec("exp17d_mechanism", "speed7", mode="torque", tier="full")
        ).write()
        out = tmp_path / "stats"
        proc = _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--output", str(out),
        )
        assert proc.returncode == 0, proc.stderr
        stats = _stats_payload(out)
        assert stats["mode"] == "torque"
        assert Path(stats["catalog"]["path"]) == dataset.catalog_path.resolve()
        assert stats["synthetic"] is True
        assert _e3_cell(stats, "7|full")["n_valid"] == 2
        assert _e3_cell(stats, "7|full", physical=True)["denominator"] == 2
        assert _e3_cell(stats, "7|full", physical=True)["rate"] == 100.0
        validation = stats["validation"]
        assert validation["manifest_expected"] == 2
        assert validation["manifest_missing"] == 0
        assert validation["runs_records"] == 2

    def test_paired_stats_reports_manifest_missing_enrichment(self, tmp_path: Path) -> None:
        dataset = Dataset(tmp_path / "root").add(
            Spec("exp17d_mechanism", "speed7", mode="torque", tier="full")
        ).write()
        _write_csv(dataset.run_root / "runs.csv", dataset.runs_rows[:1], RUNS_FIELDS)
        out = tmp_path / "stats"
        proc = _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--output", str(out),
        )
        assert proc.returncode == 0, proc.stderr
        stats = _stats_payload(out)
        validation = stats["validation"]
        assert validation["manifest_missing"] == 1
        assert validation["manifest_missing_samples"]
        assert "101" in json.dumps(validation["manifest_missing_samples"])
        assert _e3_cell(stats, "7|full", physical=True)["denominator"] == 1
        assert _e3_cell(stats, "7|full")["n_valid"] == 2

    def test_paired_stats_bad_catalog_exits_nonzero(self, tmp_path: Path) -> None:
        dataset = Dataset(tmp_path / "root").add(
            Spec("exp17d_mechanism", "speed7", mode="torque")
        ).write()
        broken = dataset.catalog_path
        payload = _read_json(broken)
        payload["schema_version"] = 99
        _write_json(broken, payload)
        out = tmp_path / "stats"
        proc = _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(broken),
            "--output", str(out),
        )
        assert proc.returncode != 0
        assert "catalog" in proc.stderr.lower()
        assert not (out / "06-stats-active.json").exists()
        assert not (out / "paper_stats_active.json").exists()


# ---------------------------------------------------------------------------
# F04：模式分池 / 唯一性 / 一对一连接
# ---------------------------------------------------------------------------


def _mixed_mode_dataset(tmp: Path) -> Dataset:
    """同一 speed/seed 下 torque 2 active + position 2 miss（F04 的原始反例）。"""
    dataset = Dataset(tmp / "root")
    dataset.add(
        Spec("exp17d_mechanism", "speed7", mode="torque", tier="full",
             physical="active", legacy="active")
    )
    dataset.add(
        Spec("exp17d_mechanism", "speed7_position", mode="position", tier="full",
             physical="miss", legacy="miss")
    )
    return dataset.write()


class TestModeSeparation:
    """torque/position 不得合成一个样本池。"""

    def test_mixed_modes_without_mode_exits_nonzero(self, tmp_path: Path) -> None:
        dataset = _mixed_mode_dataset(tmp_path)
        out = tmp_path / "stats"
        proc = _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--output", str(out),
        )
        assert proc.returncode != 0
        assert "--mode" in proc.stderr
        assert "torque" in proc.stderr and "position" in proc.stderr
        assert not (out / "paper_stats_active.json").exists()

    def test_separates_torque_and_position_rates(self, tmp_path: Path) -> None:
        dataset = _mixed_mode_dataset(tmp_path)
        torque_out = tmp_path / "torque_stats"
        position_out = tmp_path / "position_stats"
        for mode, target in (("torque", torque_out), ("position", position_out)):
            proc = _paired_stats(
                "--data-root", str(dataset.data_root),
                "--catalog", str(dataset.catalog_path),
                "--mode", mode,
                "--output", str(target),
            )
            assert proc.returncode == 0, proc.stderr
        torque = _stats_payload(torque_out)
        position = _stats_payload(position_out)
        assert torque["mode"] == "torque"
        assert position["mode"] == "position"
        assert _e3_cell(torque, "7|full")["n_valid"] == 2
        assert _e3_cell(torque, "7|full")["rate"] == 100.0
        assert _e3_cell(torque, "7|full", physical=True)["rate"] == 100.0
        assert _e3_cell(position, "7|full")["n_valid"] == 2
        assert _e3_cell(position, "7|full")["rate"] == 0.0
        assert _e3_cell(position, "7|full", physical=True)["rate"] == 0.0
        assert _e3_cell(position, "7|full", physical=True)["denominator"] == 2

    def test_both_writes_mode_dirs_and_index(self, tmp_path: Path) -> None:
        dataset = _mixed_mode_dataset(tmp_path)
        out = tmp_path / "stats"
        proc = _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--mode", "both",
            "--output", str(out),
        )
        assert proc.returncode == 0, proc.stderr
        for mode in ("torque", "position"):
            for filename in paired_stats.STATS_FILENAMES:
                assert (out / mode / filename).is_file(), (mode, filename)
        index = _read_json(out / "modes.json")
        assert index["schema"] == "paired_stats_multi_mode_v1"
        assert set(index["modes"]) == {"torque", "position"}
        for mode, entry in index["modes"].items():
            assert entry["directory"] == mode
            assert entry["validation"]["ok"] is True
            assert entry["validation"]["records"] == 2
        assert _stats_payload(out, "torque")["mode"] == "torque"
        assert _stats_payload(out, "position")["mode"] == "position"
        assert _e3_cell(_stats_payload(out, "torque"), "7|full")["rate"] == 100.0
        assert _e3_cell(_stats_payload(out, "position"), "7|full")["rate"] == 0.0

    def test_single_mode_keeps_existing_structure(self, tmp_path: Path) -> None:
        dataset = _mixed_mode_dataset(tmp_path)
        out = tmp_path / "stats"
        proc = _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--mode", "torque",
            "--output", str(out),
        )
        assert proc.returncode == 0, proc.stderr
        stats = _stats_payload(out)
        for key in (
            "metric", "include_passive", "E1_speed_sweep", "E2_limit_cost",
            "E3_nominal", "E4_grid", "E5_perception", "E7_corners",
            "E8_limit_mechanism", "exclusion", "data_root", "sources",
            "E9_shared_input", "tables",
        ):
            assert key in stats, key
        assert "cells" in stats["E3_nominal"] and "paired" in stats["E3_nominal"]
        cell = _e3_cell(stats, "7|full")
        for key in ("rows", "n_valid", "hits", "rate", "ci95", "active",
                    "passive", "miss", "passive_share", "errors"):
            assert key in cell, key

    def test_duplicate_results_record_raises(self, tmp_path: Path) -> None:
        dataset = Dataset(tmp_path / "root").add(
            Spec("exp17d_mechanism", "speed7", mode="torque", tier="full")
        ).write()
        rows_path = dataset.data_root / "exp17d_mechanism" / "results.csv"
        _append_csv_rows(rows_path, dataset.results_rows["exp17d_mechanism"][:1], RESULT_FIELDS)
        out = tmp_path / "stats"
        proc = _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--mode", "torque",
            "--output", str(out),
        )
        assert proc.returncode != 0
        assert "重复" in proc.stderr
        assert "results.csv" in proc.stderr

    def test_duplicate_runs_record_raises(self, tmp_path: Path) -> None:
        dataset = Dataset(tmp_path / "root").add(
            Spec("exp17d_mechanism", "speed7", mode="torque", tier="full")
        ).write()
        _append_csv_rows(dataset.run_root / "runs.csv", dataset.runs_rows[:1], RUNS_FIELDS)
        out = tmp_path / "stats"
        proc = _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--mode", "torque",
            "--output", str(out),
        )
        assert proc.returncode != 0
        assert "重复" in proc.stderr
        assert "runs.csv" in proc.stderr

    def test_joins_views_without_double_counting(self, tmp_path: Path) -> None:
        dataset = Dataset(tmp_path / "root").add(
            Spec("exp17d_mechanism", "speed7", mode="torque", tier="full")
        ).write()
        orphan = dict(dataset.runs_rows[0])
        orphan["seed"] = "999"
        orphan["artifact_dir"] = "artifacts/exp17d_mechanism/speed7/seed999"
        _append_csv_rows(dataset.run_root / "runs.csv", [orphan], RUNS_FIELDS)
        out = tmp_path / "stats"
        proc = _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--mode", "torque",
            "--output", str(out),
        )
        assert proc.returncode == 0, proc.stderr
        stats = _stats_payload(out)
        cell = _e3_cell(stats, "7|full")
        assert cell["rows"] == 2
        assert cell["n_valid"] == 2
        assert cell["hits"] == 2
        assert _e3_cell(stats, "7|full", physical=True)["denominator"] == 2
        assert stats["validation"]["runs_only"] == 1


# ---------------------------------------------------------------------------
# F06：E3 来源取 catalog/source_sets 实际声明
# ---------------------------------------------------------------------------


class TestNominalSource:
    """E3 先找 exp17a_noise 会整节 unavailable；应按声明来源取数。"""

    def test_e3_uses_declared_nominal_mechanism_only(self, tmp_path: Path) -> None:
        dataset = Dataset(
            tmp_path / "root",
            source_sets={"nominal_mechanism": "exp17d_mechanism"},
        )
        for tier in ("full", "none", "tube_only", "softmin_only"):
            for speed in (7, 9, 12):
                dataset.add(
                    Spec("exp17d_mechanism", f"speed{speed}", mode="torque", tier=tier,
                         speed=speed)
                )
        dataset.write()
        out = tmp_path / "stats"
        proc = _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--mode", "torque",
            "--output", str(out),
        )
        assert proc.returncode == 0, proc.stderr
        stats = _stats_payload(out)
        assert not any(item.startswith("E3_nominal") for item in stats["unavailable"])
        for speed in (7, 9, 12):
            for tier in ("full", "none", "tube_only", "softmin_only"):
                cell = _e3_cell(stats, f"{speed}|{tier}")
                assert cell["n_valid"] == 2, (speed, tier, cell)
        assert "exp17d_mechanism" in json.dumps(stats["E3_nominal"].get("data_sources", []))
        assert "exp17a_noise" not in json.dumps(stats["sources"])

    def test_e3_legacy_sources_without_catalog(self, tmp_path: Path) -> None:
        """无 catalog 的 RM65 旧根仍走 exp17a_noise + exp17d_mechanism。"""
        data_root = tmp_path / "legacy_data"
        rows = []
        for tier in ("full", "none"):
            for speed in (7, 9, 12):
                for seed in (100, 101):
                    rows.append(_legacy_row("exp17a_noise", tier, speed, seed))
        _write_csv(data_root / "exp17a_noise" / "results.csv", rows, RESULT_FIELDS)
        rows = []
        for tier in ("tube_only", "softmin_only"):
            for speed in (7, 9):
                for seed in (100, 101):
                    rows.append(_legacy_row("exp17d_mechanism", tier, speed, seed))
        _write_csv(data_root / "exp17d_mechanism" / "results.csv", rows, RESULT_FIELDS)
        out = tmp_path / "stats"
        proc = _paired_stats("--data-root", str(data_root), "--output", str(out))
        assert proc.returncode == 0, proc.stderr
        stats = _stats_payload(out)
        for key in ("7|full", "7|none", "7|tube_only", "7|softmin_only"):
            assert _e3_cell(stats, key)["n_valid"] == 2, key
        assert _e3_cell(stats, "12|tube_only")["n_valid"] == 0
        assert stats["catalog"] is None


def _legacy_row(experiment: str, tier: str, speed: int, seed: int) -> dict[str, str]:
    """旧 RM65 行：无 mode 列值、无 physical 列、tier 由 ``--ablation`` 表达。"""
    cfg = {"--ball-speed": speed, "--no-plot": None, "--ablation": tier}
    return {
        "experiment": experiment,
        "config_id": f"{experiment}__{tier}__{speed}__{seed}",
        "config_json": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
        "seed": str(seed),
        "hit": "True",
        "hit_type": "active",
        "pos_error": "0.01",
        "error": "",
        "mode": "",
        "robot_id": "",
        "racket_id": "",
    }


# ---------------------------------------------------------------------------
# F07：E8 DiD 全 miss 的不可估计语义
# ---------------------------------------------------------------------------


def _e8_dataset(tmp: Path, *, hit_map: Mapping[tuple[str, str, int], str]) -> Dataset:
    """E8 数据根：tier ∈ {full, none} × TCP 层 × 两个 seed。

    ``hit_map`` 的键为 ``(tier, tcp, seed)``，值为 ``active`` / ``miss``；
    未给出的单元格默认 ``miss``。legacy 判定统一 ``active``（只让 physical 全 miss）。
    """
    dataset = Dataset(tmp / "root")
    for tier in ("full", "none"):
        for tcp in ("TCP1.8", "TCP1.0"):
            for seed in (100, 101):
                physical = hit_map.get((tier, tcp, seed), "miss")
                dataset.add(
                    Spec(
                        "exp17i_limits_ablation",
                        f"{tier}_{tcp.lower()}",
                        mode="torque",
                        tier=tier,
                        seeds=(seed,),
                        physical=physical,
                        legacy="active",
                        tcp=tcp,
                    )
                )
    return dataset.write()


class TestDidUnavailable:
    """全 miss 不得输出 [0,0] 与 p=1 的“无差异”假象。"""

    def test_e8_all_miss_physical_did_is_unavailable(self, tmp_path: Path) -> None:
        dataset = _e8_dataset(tmp_path, hit_map={})
        out = tmp_path / "stats"
        proc = _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--mode", "torque",
            "--output", str(out),
        )
        assert proc.returncode == 0, proc.stderr
        stats = _stats_payload(out)
        physical_did = _e8_section(stats, physical=True)["did"]["full"]
        assert physical_did["n_pairs"] == 2
        assert physical_did["mean_did_pp"] == 0.0
        assert physical_did["ci95"] is None
        assert physical_did["p_sign"] is None
        assert physical_did["unavailable_reason"] == "all_miss"
        assert physical_did["hit_semantics"] == "physical"
        pair = _e8_section(stats, physical=True)["paired"]["TCP1.8|full_vs_none"]
        assert pair["n_paired"] == 2
        assert pair["diff_pp"] == 0.0
        assert pair["ci95"] is None and pair["p_mcnemar"] is None
        assert pair["unavailable_reason"] == "all_miss"
        # legacy 行为保持不变（有真实 hit，数值可估计）
        legacy_did = _e8_section(stats)["did"]["full"]
        assert isinstance(legacy_did["ci95"], list)
        assert legacy_did["p_sign"] is not None

    def test_e8_mixed_miss_hit_physical_did_is_estimated(self, tmp_path: Path) -> None:
        dataset = _e8_dataset(
            tmp_path,
            hit_map={
                ("full", "TCP1.8", 100): "miss",
                ("full", "TCP1.8", 101): "miss",
                ("full", "TCP1.0", 100): "active",
                ("full", "TCP1.0", 101): "miss",
            },
        )
        out = tmp_path / "stats"
        proc = _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--mode", "torque",
            "--output", str(out),
        )
        assert proc.returncode == 0, proc.stderr
        stats = _stats_payload(out)
        did = _e8_section(stats, physical=True)["did"]["full"]
        assert did["n_pairs"] == 2
        assert isinstance(did["ci95"], list)
        assert isinstance(did["p_sign"], float)
        assert "unavailable_reason" not in did

    def test_did_reason_missing_verdict_error_and_no_common_seed(self, tmp_path: Path) -> None:
        rows = _did_rows(
            tier="full", tcp="TCP1.8", seeds=(100, 101), physical_type="", physical_hit="",
        ) + _did_rows(
            tier="none", tcp="TCP1.8", seeds=(100, 101), physical_type="", physical_hit="",
        )
        result = paired_stats._did(rows, paired_stats.physical_of, semantics="physical")
        assert result["full"]["n_pairs"] == 0
        assert result["full"]["unavailable_reason"] == "no_physical_verdict"

        errored = _did_rows(
            tier="full", tcp="TCP1.8", seeds=(100,), physical_type="miss", error="boom",
        ) + _did_rows(
            tier="none", tcp="TCP1.8", seeds=(100,), physical_type="miss", error="boom",
        )
        assert paired_stats._did(
            errored, paired_stats.physical_of, semantics="physical",
        )["full"]["unavailable_reason"] == "all_error"

        disjoint = _did_rows(
            tier="full", tcp="TCP1.0", seeds=(201,), physical_type="miss",
        ) + _did_rows(
            tier="none", tcp="TCP1.0", seeds=(201,), physical_type="miss",
        ) + _did_rows(
            tier="full", tcp="TCP1.8", seeds=(100,), physical_type="miss",
        ) + _did_rows(
            tier="none", tcp="TCP1.8", seeds=(100,), physical_type="miss",
        )
        assert paired_stats._did(
            disjoint, paired_stats.physical_of, semantics="physical",
        )["full"]["unavailable_reason"] == "no_common_seed"

    def test_paper_figs_table2_physical_all_miss_renders_dashes(self, tmp_path: Path) -> None:
        dataset = _e8_dataset(tmp_path, hit_map={})
        stats_dir = tmp_path / "stats"
        proc = _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--mode", "torque",
            "--output", str(stats_dir),
        )
        assert proc.returncode == 0, proc.stderr
        out = tmp_path / "tex"
        proc = _paper_figs(
            "--stats", str(stats_dir / "paper_stats_active.json"),
            "--fig", "table",
            "--out", str(out),
            "--hit-semantics", "physical",
        )
        assert proc.returncode == 0, proc.stderr
        tex = (out / "table_data" / "table2_ablation.tex").read_text(encoding="utf-8")
        assert "--" in tex
        assert "all_miss" in tex
        assert "physical" in tex
        assert "nan" not in tex.lower()
        assert "$[0.0,0.0]$" not in tex

    def test_paper_figs_table2_legacy_stays_numeric(self, tmp_path: Path) -> None:
        dataset = _e8_dataset(tmp_path, hit_map={})
        stats_dir = tmp_path / "stats"
        assert _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--mode", "torque",
            "--output", str(stats_dir),
        ).returncode == 0
        out = tmp_path / "tex"
        proc = _paper_figs(
            "--stats", str(stats_dir / "paper_stats_active.json"),
            "--fig", "table",
            "--out", str(out),
        )
        assert proc.returncode == 0, proc.stderr
        tex = (out / "table_data" / "table2_ablation.tex").read_text(encoding="utf-8")
        assert "$[0.0,0.0]$" in tex
        assert "all_miss" not in tex

    def test_paper_figs_table1_physical_uses_physical_section_and_mode(
        self, tmp_path: Path,
    ) -> None:
        dataset = _mixed_mode_dataset(tmp_path)
        stats_dir = tmp_path / "stats"
        assert _paired_stats(
            "--data-root", str(dataset.data_root),
            "--catalog", str(dataset.catalog_path),
            "--mode", "torque",
            "--output", str(stats_dir),
        ).returncode == 0
        out = tmp_path / "tex"
        proc = _paper_figs(
            "--stats", str(stats_dir / "paper_stats_active.json"),
            "--fig", "table",
            "--out", str(out),
            "--hit-semantics", "physical",
        )
        assert proc.returncode == 0, proc.stderr
        tex = (out / "table_data" / "table1_comparison.tex").read_text(encoding="utf-8")
        assert "100.0" in tex
        assert "mode=torque" in tex
        assert "physical" in tex
        assert "--" in tex

    def test_paper_figs_physical_without_section_skips_cleanly(self, tmp_path: Path) -> None:
        stats_path = _write_json(
            tmp_path / "legacy_stats.json",
            {"E3_nominal": {"cells": {}}, "E4_grid": {}, "E7_corners": {}, "E8_limit_mechanism": {}},
        )
        out = tmp_path / "tex"
        proc = _paper_figs(
            "--stats", str(stats_path),
            "--fig", "table",
            "--out", str(out),
            "--hit-semantics", "physical",
        )
        assert proc.returncode == 0
        assert "跳过" in proc.stdout
        assert not (out / "table_data" / "table1_comparison.tex").exists()


def _did_rows(
    *,
    tier: str,
    tcp: str,
    seeds: Sequence[int],
    physical_type: str,
    physical_hit: str | None = None,
    error: str = "",
) -> list[dict[str, str]]:
    """构造 ``_did`` 直接消费的伪行（只带必要列）。"""
    cfg: dict[str, Any] = {
        "--ball-speed": 7,
        "--no-plot": None,
        "--robot-config": "configs/robots/r1d_right.yaml",
    }
    if tier != "full":
        cfg["--task-overrides"] = list(TIER_OVERRIDES[tier] or [])
    if tcp == "TCP1.0":
        cfg["--limits-config"] = "configs/real_robot.yaml"
    rows: list[dict[str, str]] = []
    for seed in seeds:
        flags = physical_hit
        if flags is None:
            flags = "True" if physical_type in {"active", "passive"} else "False"
        rows.append(
            {
                "experiment": "exp17i_limits_ablation",
                "seed": str(seed),
                "config_json": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
                "error": error,
                "mode": "torque",
                "robot_id": ROBOT_ID,
                "racket_id": RACKET_ID,
                "physical_hit": flags,
                "physical_hit_type": physical_type,
            }
        )
    return rows




# ---------------------------------------------------------------------------
# 复审 R06：同配置复运行 case（不同 case_id、同 tier/tcp/seed）的 DiD 冲突
# ---------------------------------------------------------------------------


def _e8_repeat_dataset(tmp: Path, *, reverse: bool = False) -> Dataset:
    """E8 数据根 + 一个同配置复运行 case（TCP1.0/full 每 seed 一命中一未命中）。

    ``reverse=True`` 时把 spec 顺序整体反转，用于验证统计结果与 CSV 行序无关。
    """
    dataset = Dataset(tmp / "root")
    for tier in ("full", "none"):
        for tcp in ("TCP1.8", "TCP1.0"):
            for seed in (100, 101):
                hit = tier == "full"
                dataset.add(
                    Spec(
                        "exp17i_limits_ablation",
                        f"{tier}_{tcp.lower()}",
                        mode="torque",
                        tier=tier,
                        seeds=(seed,),
                        physical="active" if hit else "miss",
                        legacy="active" if hit else "miss",
                        tcp=tcp,
                    )
                )
    # 复运行：同配置、不同 case ID；TCP1.0/full 的复运行全部 miss
    for seed in (100, 101):
        dataset.add(
            Spec(
                "exp17i_limits_ablation",
                "full_tcp1.0_repeat",
                mode="torque",
                tier="full",
                seeds=(seed,),
                physical="miss",
                legacy="miss",
                tcp="TCP1.0",
            )
        )
    if reverse:
        dataset.specs.reverse()
    return dataset.write()


class TestE8RepeatedCaseDid:
    """复审 R06: DiD 与普通配对共用冲突排除规则，且行序无关。"""

    def _stats(self, tmp_path: Path, *, reverse: bool) -> dict[str, Any]:
        """以真实 CLI 跑 E8 统计（复运行行由 producer 原样复制，应 exit 0）。"""
        dataset = _e8_repeat_dataset(
            tmp_path / ("rev" if reverse else "fwd"), reverse=reverse,
        )
        out = tmp_path / ("stats_rev" if reverse else "stats_fwd")
        proc = _paired_stats(
            "--catalog", str(dataset.catalog_path), "--mode", "torque",
            "--output", str(out),
        )
        assert proc.returncode == 0, proc.stderr
        return _stats_payload(out)

    def test_legacy_did_flags_duplicate_seed_conflict(self, tmp_path: Path) -> None:
        """TCP1.0/full 每 seed 一命中一未命中：DiD 不得给出 0/100pp。"""
        stats = self._stats(tmp_path, reverse=False)
        did = _e8_section(stats)["did"]["full"]
        assert did["n_pairs"] == 0
        assert did["unavailable_reason"] == "duplicate_seed_conflict"
        assert did["excluded_duplicate_keys"] == [100, 101]
        assert did["mean_did_pp"] is None
        assert did["ci95"] is None and did["p_sign"] is None
        pair = _e8_section(stats)["paired"]["TCP1.0|full_vs_none"]
        assert pair["n_paired"] == 0
        assert pair["unavailable_reason"] == "duplicate_seed_conflict"

    def test_physical_did_flags_same_conflict(self, tmp_path: Path) -> None:
        """physical 口径同样排除冲突 seed，不输出显著性。"""
        stats = self._stats(tmp_path, reverse=False)
        did = _e8_section(stats, physical=True)["did"]["full"]
        assert did["n_pairs"] == 0
        assert did["unavailable_reason"] == "duplicate_seed_conflict"
        assert did["hit_semantics"] == "physical"

    def test_row_order_reversal_matches(self, tmp_path: Path) -> None:
        """正序 / 反序 CSV 的 DiD 与配对结果逐字段一致。"""
        forward = self._stats(tmp_path, reverse=False)
        reverse = self._stats(tmp_path, reverse=True)
        assert _e8_section(forward)["did"] == _e8_section(reverse)["did"]
        assert _e8_section(forward)["paired"] == _e8_section(reverse)["paired"]
        assert (
            _e8_section(forward, physical=True)["did"]
            == _e8_section(reverse, physical=True)["did"]
        )


# ---------------------------------------------------------------------------
# M03：physical 富化按身份 join（同 artifact 路径但身份不同必须拒绝）
# ---------------------------------------------------------------------------


def _rewrite_runs(
    dataset: Dataset, mutate: Callable[[dict[str, str]], dict[str, str]],
) -> Path:
    """按 ``mutate`` 改写 runs.csv（保留列顺序与其余字段），返回文件路径。"""
    path = dataset.run_root / "runs.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [mutate(dict(row)) for row in csv.DictReader(handle)]
    _write_csv(path, rows, RUNS_FIELDS)
    return path


def _cross_view_mutation(row: dict[str, str]) -> dict[str, str]:
    """同身份但不同配置/判定的跨视图行（复审 cross_view_conflict 场景）。"""
    cfg = json.loads(row["config_json"])
    cfg["--ball-speed"] = 12
    row["config_json"] = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
    row["config_hash"] = "different-source-hash"
    row["hit"] = row["physical_hit"] = "True"
    row["hit_type"] = row["physical_hit_type"] = "active"
    return row


def _legacy_result_mutation(row: dict[str, str]) -> dict[str, str]:
    """只改 runs 的 legacy 结果列，配置/身份/哈希与 physical 列全部不动。

    复审 R05 场景：producer 本应把 results.csv 的 ``hit``/``hit_type`` 原样复制
    进 runs.csv，这里故意只改这两列，检验富化冲突检测覆盖共有结果列。
    """
    row["hit"] = "False"
    row["hit_type"] = "miss"
    return row


def _foreign_mode_mutation(row: dict[str, str]) -> dict[str, str]:
    """把 runs 行改成另一执行模式（position）的伪判定（复审 foreign_mode 场景）。"""
    row["mode"] = "position"
    row["physical_hit"] = "True"
    row["physical_hit_type"] = "active"
    row["hit_type"] = "active"
    return row


class TestPhysicalEnrichmentIdentity:
    """M03: 富化按 ``RecordIdentity.join_key()`` 命中，artifact 路径不再兜底串接。"""

    def _dataset(self, tmp_path: Path) -> Dataset:
        """单个 torque/full/speed9 数据根（2 seeds）。"""
        return Dataset(tmp_path / "ds").add(
            Spec("exp17d_mechanism", "speed9", mode="torque", tier="full", speed=9)
        )

    def _stats(
        self, tmp_path: Path, dataset: Dataset, expected_returncode: int = 0
    ) -> dict[str, Any]:
        """以真实 CLI 生成统计 JSON（torque 口径；冲突场景期望 exit 2）。"""
        out = tmp_path / "stats"
        proc = _paired_stats(
            "--catalog", str(dataset.catalog_path), "--mode", "torque",
            "--output", str(out),
        )
        assert proc.returncode == expected_returncode, proc.stderr
        return _stats_payload(out)

    def test_cross_view_conflict_is_not_enriched(self, tmp_path: Path) -> None:
        """同 artifact 路径但 config/判定不同：不富化，物理判定保留空值并 exit 2。"""
        dataset = self._dataset(tmp_path).write()
        _rewrite_runs(dataset, _cross_view_mutation)
        stats = self._stats(tmp_path, dataset, expected_returncode=2)
        # legacy 视图来自 results.csv，不受 runs 篡改影响
        assert _e3_cell(stats, "9|full")["n_valid"] == 2
        assert _e3_cell(stats, "9|full", physical=True)["denominator"] == 0
        validation = stats["validation"]
        assert validation["physical_conflicts"] == 2
        fields = {
            field
            for sample in validation["physical_conflict_samples"]
            for field in sample["fields"]
        }
        assert any("design" in field for field in fields), fields

    def test_legacy_result_conflict_is_rejected(self, tmp_path: Path) -> None:
        """只改 runs 的 hit/hit_type（配置与哈希不变）也必须冲突且不回填。"""
        dataset = self._dataset(tmp_path).write()
        _rewrite_runs(dataset, _legacy_result_mutation)
        stats = self._stats(tmp_path, dataset, expected_returncode=2)
        # legacy 视图来自 results.csv：冲突不改变其命中结论
        cell = _e3_cell(stats, "9|full")
        assert cell["n_valid"] == 2
        assert cell["hits"] == 2
        # 不回填：冲突行不得拿到 runs 的 physical 判定
        assert _e3_cell(stats, "9|full", physical=True)["denominator"] == 0
        validation = stats["validation"]
        assert validation["physical_conflicts"] == 2
        fields = {
            field
            for sample in validation["physical_conflict_samples"]
            for field in sample["fields"]
        }
        assert {"hit", "hit_type"} <= fields, fields

    def test_completion_state_conflict_is_rejected(self, tmp_path: Path) -> None:
        """单独改 error/exit_code/timed_out 也必须判冲突且不回填（R05）。"""
        for column, value in (
            ("error", "boom"),
            ("exit_code", "1"),
            ("timed_out", "1"),
        ):
            case_dir = tmp_path / column
            dataset = self._dataset(case_dir).write()

            def mutate(row: dict[str, str], column: str = column, value: str = value) -> dict[str, str]:
                """只改完成状态列，配置/身份/命中列全部保持原样。"""
                row[column] = value
                return row

            _rewrite_runs(dataset, mutate)
            stats = self._stats(case_dir, dataset, expected_returncode=2)
            assert _e3_cell(stats, "9|full", physical=True)["denominator"] == 0
            validation = stats["validation"]
            assert validation["physical_conflicts"] == 2
            fields = {
                field
                for sample in validation["physical_conflict_samples"]
                for field in sample["fields"]
            }
            assert column in fields, (column, fields)

    def test_foreign_mode_runs_are_not_joined(self, tmp_path: Path) -> None:
        """runs 行属于 position 模式：torque 统计不得接入该判定。"""
        dataset = self._dataset(tmp_path).write()
        _rewrite_runs(dataset, _foreign_mode_mutation)
        stats = self._stats(tmp_path, dataset)
        assert _e3_cell(stats, "9|full", physical=True)["denominator"] == 0
        validation = stats["validation"]
        assert validation["enrichment_status"]["missing"] == 2
        assert validation["physical_conflicts"] == 0

    def test_normal_dataset_still_enriched(self, tmp_path: Path) -> None:
        """正常数据仍按身份一对一富化（不引入回归）。"""
        dataset = self._dataset(tmp_path).write()
        stats = self._stats(tmp_path, dataset)
        assert _e3_cell(stats, "9|full", physical=True)["denominator"] == 2
        validation = stats["validation"]
        assert validation["enrichment_status"]["matched"] == 2
        assert validation["physical_conflicts"] == 0

    def test_design_fingerprint_ignores_config_path_spelling(self) -> None:
        """M06 相对化后配置路径写法不同，但同一配置不得被误判为设计冲突。

        results.csv 保留 run_root 内的绝对配置路径，runs.csv 的
        ``config_json`` 已被 M06 改写成相对路径；两者只有路径拼写不同，
        设计指纹必须一致，否则真实数据的物理判定会被整批拒绝富化。
        """
        results_row = {
            "config_json": json.dumps({
                "--runtime-config": "F:/run/runtime/r1d_v12_runtime_right.yaml",
                "--output": "F:/run/artifacts/exp/case/seed100",
                "--ball-speed": 9,
            })
        }
        runs_row = {
            "config_json": json.dumps({
                "--runtime-config": "runtime/r1d_v12_runtime_right.yaml",
                "--output": "artifacts/exp/case/seed100",
                "--ball-speed": 9,
            })
        }
        assert (
            paired_stats._design_fingerprint(results_row)
            == paired_stats._design_fingerprint(runs_row)
        )

    def test_design_fingerprint_keeps_distinct_config_basenames(self) -> None:
        """路径归一化只忽略目录，不同配置文件名仍必须区分。"""
        left = {"config_json": json.dumps({
            "--runtime-config": "F:/run/runtime/left.yaml"})}
        right = {"config_json": json.dumps({
            "--runtime-config": "runtime/right.yaml"})}
        assert (
            paired_stats._design_fingerprint(left)
            != paired_stats._design_fingerprint(right)
        )
