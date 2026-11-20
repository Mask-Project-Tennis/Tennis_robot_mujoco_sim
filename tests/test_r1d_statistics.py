"""T8b 论文统计链路测试：R1D 数据根 + 显式物理/legacy 口径。

覆盖点（对应 task-10-brief §7）：
1. ``extract_r1d_smoke_stats`` 六类缺口 fixture（完整/缺数据/全 miss/无共同 seed/
   超时/no-verdict）+ legacy-only/physical-only，断言 rate、分母、unavailable_reason；
2. 七个既有 extract 入口接受 ``--data-root`` / ``--output``，子模块调用与顶层同源；
3. ``analyze_chat12_night`` 导入零副作用（不再 import 即执行、零打印）；
4. ``paired_stats`` 默认不再硬写 ``paper/planning``，旧 E1–E8 键仍在（追加式扩展）；
5. 旧 RM65 口径（legacy config_json flag）与新 R1D 口径（``--task-overrides``）
   不得混算同一条指标。
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTRACT_DIR = ROOT / "scripts" / "extract"
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

PYTHON = sys.executable

# 旧 RM65 results.csv 的列定义（不含 T8a 新增身份/状态列）
LEGACY_CSV_FIELDS = [
    "experiment",
    "config_id",
    "config_json",
    "seed",
    "hit",
    "hit_type",
    "pos_error",
    "vel_error",
    "min_dist",
    "ball_near_ms",
    "tube_ready_ms",
    "max_tcp",
    "max_qdot",
    "max_face",
    "hit_time_error_ms",
    "hit_pos_error",
    "v_racket_at_hit",
    "wall_time",
    "first_plan_ms",
    "ball_out_speed",
    "ball_out_vy",
    "error",
    "robot_id",
    "mode",
    "racket_id",
    "config_hash",
    "metric_schema",
    "timeout_s",
    "timed_out",
    "exit_code",
    "failure_reason",
    "total_steps",
    "safe_steps",
    "hit_step",
    "artifact_dir",
    "log_path",
    "physical_hit",
    "physical_hit_type",
    "physical_hit_step",
    "contact_event_count",
    "artifact_error",
]


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _run_script(*args: str, cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    """用当前解释器真实运行入口脚本（UTF-8 输出，避免 Windows GBK 崩）。"""
    return subprocess.run(
        [PYTHON, *args],
        cwd=str(cwd or ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env={
            **__import__("os").environ,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        },
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _cfg(**kwargs: object) -> str:
    return json.dumps(kwargs, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# R1D smoke 产物 fixture
# ---------------------------------------------------------------------------


def _v12_hit(
    *,
    physical: bool | None = True,
    physical_type: str | None = "active",
    legacy: bool | None = True,
    legacy_type: str | None = "active",
    physical_step: int = 12,
    pos_error: float = 0.01,
    contact_events: int = 1,
) -> dict[str, object]:
    """构造 ``v12_episode.json`` 的 ``hit`` 段（v12_episode_v2 语义）。"""
    hit: dict[str, object] = {
        "legacy_hit": legacy,
        "legacy_hit_type": legacy_type,
        "legacy_hit_rule": "legacy_first_window_contact",
        "physical_hit_rule": "first_physical_face_contact_face_speed_0p3",
        "step": physical_step,
        "time_s": 0.5,
        "pos_error": pos_error,
        "min_dist": 0.05,
        "v_racket_at_hit": 0.6,
        "ball_out_speed": 6.1,
        "ball_out_vy": 2.2,
        "contact_events": [{"step": physical_step, "face_speed": 0.6}] * contact_events,
        "message": "",
    }
    if physical is not None:
        hit["physical_hit"] = physical
    if physical_type is not None:
        hit["physical_hit_type"] = physical_type
    if physical is not None or physical_type is not None:
        hit["physical_hit_step"] = physical_step if (physical or physical_type in {"active", "passive"}) else -1
    return hit


def _write_seed(
    run_root: Path,
    experiment: str,
    case: str,
    seed: int,
    *,
    v12_hit: dict[str, object] | None = None,
    v12_extra: dict[str, object] | None = None,
    r1d_hit_type: str | None = None,
    write_v12: bool = True,
    write_r1d: bool = True,
    backend: str = "auto",
) -> Path:
    """在 ``<run_root>/artifacts/<exp>/<case>/seed<seed>/`` 写最小产物。"""
    artifact_dir = run_root / "artifacts" / experiment / case / f"seed{seed}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if write_v12:
        payload: dict[str, object] = {
            "schema": "v12_episode_v2",
            "robot_id": "unitree_r1d",
            "mode": "torque",
            "racket_id": "blade100_v9_reference",
            "backend": backend,
            "failure_reason": None,
            "episode": {"total_steps": 12},
            "hit": v12_hit if v12_hit is not None else _v12_hit(),
        }
        if v12_extra:
            payload.update(v12_extra)
        _write_json(artifact_dir / "v12_episode.json", payload)
    if write_r1d:
        _write_json(
            artifact_dir / "r1d_episode.json",
            {
                "schema": "r1d_sim_episode_v1",
                "robot_id": "unitree_r1d",
                "mode": "torque",
                "backend": backend,
                "hit": {"type": r1d_hit_type or "active", "step": 12, "message": ""},
                "metrics": {"n_mpc_steps": 12},
            },
        )
    return artifact_dir


def _smoke_row(
    *,
    experiment: str,
    case: str,
    seed: int,
    recorded_seed: int | None = None,
    hit_type: str = "active",
    hit: str | None = None,
    physical_hit: str = "True",
    physical_hit_type: str = "active",
    failure_reason: str = "",
    timed_out: str = "0",
    error: str = "",
    pos_error: str = "0.01",
    artifact_dir: Path | None = None,
    mode: str = "torque",
) -> dict[str, str]:
    """构造 T8a ``runs.csv`` / 每实验 results.csv 的一行（含身份列）。"""
    used_seed = seed if recorded_seed is None else recorded_seed
    return {
        "experiment": experiment,
        "config_id": f"{case}__seed",
        "config_json": _cfg(
            **{
                "--ball-speed": "7",
                **({"--output": str(artifact_dir)} if artifact_dir else {}),
            }
        ),
        "seed": str(used_seed),
        "hit": ("True" if hit_type in {"active", "passive"} else "False") if hit is None else hit,
        "hit_type": hit_type,
        "pos_error": pos_error,
        "error": error,
        "robot_id": "unitree_r1d",
        "mode": mode,
        "racket_id": "blade100_v9_reference",
        "config_hash": "deadbeefdeadbeef",
        "metric_schema": "v12_episode_v2_physical_hit",
        "timeout_s": "900",
        "timed_out": timed_out,
        "exit_code": "0" if not error else "3",
        "failure_reason": failure_reason,
        "artifact_dir": str(artifact_dir) if artifact_dir else "",
        "physical_hit": physical_hit,
        "physical_hit_type": physical_hit_type,
        "physical_hit_step": "12" if physical_hit == "True" else "-1",
        "contact_event_count": "1",
    }


def _build_smoke_root(
    tmp_path: Path,
    *,
    seeds_artifact: list[int],
    seeds_csv: list[int] | None = None,
    v12_hit_factory=None,
    row_factory=None,
    experiment: str = "exp15_speed_v2",
    case: str = "speed7",
    write_r1d: bool = True,
    r1d_hit_type: str = "active",
) -> tuple[Path, Path, Path]:
    """构造 R1D smoke 数据根，返回 (data_root, run_root, artifact_root)。"""
    run_root = tmp_path / "run"
    data_root = run_root / "batch"
    artifact_root = run_root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    row_factory = row_factory or (lambda seed, artifact_dir: _smoke_row(experiment=experiment, case=case, seed=seed, artifact_dir=artifact_dir))
    for seed in seeds_artifact:
        factory = v12_hit_factory or (lambda seed: _v12_hit())
        hit = factory(seed)
        _write_seed(
            run_root,
            experiment,
            case,
            seed,
            v12_hit=hit,
            write_r1d=write_r1d,
            r1d_hit_type=r1d_hit_type,
        )
    rows: list[dict[str, str]] = []
    for seed in seeds_artifact if seeds_csv is None else seeds_csv:
        artifact_dir = artifact_root / experiment / case / f"seed{seed}"
        rows.append(row_factory(seed, artifact_dir))
    _write_csv(run_root / "runs.csv", rows, LEGACY_CSV_FIELDS)
    _write_csv(data_root / experiment / "results.csv", rows, LEGACY_CSV_FIELDS)
    (data_root / experiment / "config.yaml").write_text("experiment: fixture\n", encoding="utf-8")
    return data_root, run_root, artifact_root


@pytest.fixture()
def smoke_extract():
    """导入被测模块；未实现时应明确失败（TDD 红灯）。"""
    import extract_r1d_smoke_stats as module

    return module


def _extract(tmp_path: Path, data_root: Path, case: str = "speed7", experiment: str = "exp15_speed_v2", extra: list[str] | None = None) -> dict:
    out = tmp_path / "out"
    proc = _run_script(
        str(EXTRACT_DIR / "extract_r1d_smoke_stats.py"),
        "--experiment",
        experiment,
        "--case",
        case,
        "--data-root",
        str(data_root),
        "--output",
        str(out),
        *(extra or []),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads((out / experiment / f"{case}.json").read_text(encoding="utf-8"))
    return payload


# ---------------------------------------------------------------------------
# extract_r1d_smoke_stats：完整与口径分离
# ---------------------------------------------------------------------------


def test_complete_case_reports_physical_and_legacy_rates(tmp_path: Path) -> None:
    """完整 3 seed：physical/legacy 两条口径都给出 rate 与分母，且口径字段不混。"""
    data_root, _, _ = _build_smoke_root(tmp_path, seeds_artifact=[100, 101, 102])
    payload = _extract(tmp_path, data_root)
    assert payload["schema"] == "r1d_smoke_case_stats_v1"
    assert payload["experiment_id"] == "exp15_speed_v2"
    assert payload["case_id"] == "speed7"
    assert payload["mode"] == "torque"
    assert payload["racket_id"] == "blade100_v9_reference"
    assert payload["robot_id"] == "unitree_r1d"
    assert payload["n_attempted"] == 3
    assert payload["n_valid"] == 3
    assert payload["n_physical_active"] == 3
    assert payload["n_legacy_active"] == 3
    assert payload["physical_active_rate"] == pytest.approx(100.0)
    assert payload["legacy_active_rate"] == pytest.approx(100.0)
    assert payload["physical_active_denominator"] == 3
    assert payload["legacy_active_denominator"] == 3
    assert payload["unavailable_reason"] is None
    assert len(payload["per_seed"]) == 3
    seed_entry = payload["per_seed"][0]
    for key in (
        "seed",
        "status",
        "physical_hit",
        "physical_hit_type",
        "legacy_hit",
        "legacy_hit_type",
        "failure_reason",
        "backend",
    ):
        assert key in seed_entry, key
    assert payload["hit_semantics"]["physical"].startswith("v12_episode_v2")
    assert "hit_type" in payload["hit_semantics"]["legacy"]


def test_empty_root_is_missing_data_not_zero_rate(tmp_path: Path) -> None:
    """完全缺数据：rate 必须是 null + unavailable_reason，不得伪造 0.0。"""
    run_root = tmp_path / "empty_run"
    data_root = run_root / "batch"
    data_root.mkdir(parents=True, exist_ok=True)
    payload = _extract(tmp_path, data_root)
    assert payload["n_attempted"] == 0
    assert payload["physical_active_rate"] is None
    assert payload["legacy_active_rate"] is None
    assert payload["unavailable_reason"]
    assert "missing_data" in payload["unavailable_reason"]
    assert payload["per_seed"] == []


def test_all_miss_keeps_denominator_and_nulls_rate(tmp_path: Path) -> None:
    """全 miss：分母保留、rate=null、reason=all_miss，不许把 0 当有效命中率。"""
    data_root, _, _ = _build_smoke_root(
        tmp_path,
        seeds_artifact=[100, 101],
        v12_hit_factory=lambda seed: _v12_hit(
            physical=False,
            physical_type="miss",
            legacy=False,
            legacy_type="miss",
            physical_step=-1,
            pos_error=0.4,
        ),
    )
    payload = _extract(tmp_path, data_root)
    assert payload["n_valid"] == 2
    assert payload["n_miss"] == 2
    assert payload["physical_active_denominator"] == 2
    assert payload["physical_active_rate"] is None
    assert payload["legacy_active_rate"] is None
    assert "all_miss" in payload["unavailable_reason"]


def test_no_common_seed_between_artifacts_and_csv(tmp_path: Path) -> None:
    """产物 seed 与 CSV seed 完全不相交：明确 no_common_seed，不得借用他根数据。"""
    data_root, _, _ = _build_smoke_root(
        tmp_path,
        seeds_artifact=[200, 201],
        seeds_csv=[100, 101],
    )
    payload = _extract(tmp_path, data_root)
    assert payload["n_attempted"] >= 2
    assert payload["physical_active_rate"] is None
    assert "no_common_seed" in payload["unavailable_reason"]
    assert payload["per_seed"], "失败/缺数据行不得被删除"


def test_timeout_row_is_kept_and_not_valid(tmp_path: Path) -> None:
    """超时行：保留在 per_seed、计入 n_timeout，不进命中率分母。"""
    run_root = tmp_path / "run"
    data_root = run_root / "batch"
    rows = [
        _smoke_row(
            experiment="exp15_speed_v2",
            case="speed7",
            seed=100,
            timed_out="1",
            hit="",
            hit_type="",
            physical_hit="",
            physical_hit_type="",
            failure_reason="timeout",
            artifact_dir=run_root / "artifacts" / "exp15_speed_v2" / "speed7" / "seed100",
        ),
        _smoke_row(
            experiment="exp15_speed_v2",
            case="speed7",
            seed=101,
            artifact_dir=run_root / "artifacts" / "exp15_speed_v2" / "speed7" / "seed101",
        ),
    ]
    artifact_dir = _write_seed(run_root, "exp15_speed_v2", "speed7", 101)
    rows[1]["artifact_dir"] = str(artifact_dir)
    _write_csv(run_root / "runs.csv", rows, LEGACY_CSV_FIELDS)
    _write_csv(data_root / "exp15_speed_v2" / "results.csv", rows, LEGACY_CSV_FIELDS)
    payload = _extract(tmp_path, data_root)
    timeout_seed = [s for s in payload["per_seed"] if s["seed"] == 100][0]
    assert timeout_seed["status"] == "timeout"
    assert payload["n_timeout"] == 1
    assert payload["n_valid"] == 1
    assert payload["physical_active_denominator"] == 1
    assert len(payload["per_seed"]) == len(rows)


def test_error_row_is_kept_and_not_valid(tmp_path: Path) -> None:
    """失败（error 非空）行：保留、计 n_error，不进入分母。"""
    run_root = tmp_path / "run"
    data_root = run_root / "batch"
    rows = [
        _smoke_row(
            experiment="exp15_speed_v2",
            case="speed7",
            seed=100,
            error="mujoco: bad model",
            hit="",
            hit_type="",
            physical_hit="",
            physical_hit_type="",
            artifact_dir=run_root / "artifacts" / "exp15_speed_v2" / "speed7" / "seed100",
        ),
        _smoke_row(experiment="exp15_speed_v2", case="speed7", seed=101, artifact_dir=run_root / "artifacts" / "exp15_speed_v2" / "speed7" / "seed101"),
    ]
    artifact_dir = _write_seed(run_root, "exp15_speed_v2", "speed7", 101)
    rows[1]["artifact_dir"] = str(artifact_dir)
    _write_csv(run_root / "runs.csv", rows, LEGACY_CSV_FIELDS)
    _write_csv(data_root / "exp15_speed_v2" / "results.csv", rows, LEGACY_CSV_FIELDS)
    payload = _extract(tmp_path, data_root)
    entry = [s for s in payload["per_seed"] if s["seed"] == 100][0]
    assert entry["status"] == "error"
    assert payload["n_error"] == 1
    assert payload["n_valid"] == 1


def test_no_verdict_when_artifact_dir_has_no_episode_json(tmp_path: Path) -> None:
    """产物目录存在但没有可判读 JSON：no_verdict，rate=null。"""
    data_root, run_root, _ = _build_smoke_root(tmp_path, seeds_artifact=[])
    (run_root / "artifacts" / "exp15_speed_v2" / "speed7" / "seed100").mkdir(parents=True)
    rows = [
        _smoke_row(
            experiment="exp15_speed_v2",
            case="speed7",
            seed=100,
            hit="",
            hit_type="",
            physical_hit="",
            physical_hit_type="",
            artifact_dir=run_root / "artifacts" / "exp15_speed_v2" / "speed7" / "seed100",
        )
    ]
    _write_csv(run_root / "runs.csv", rows, LEGACY_CSV_FIELDS)
    _write_csv(data_root / "exp15_speed_v2" / "results.csv", rows, LEGACY_CSV_FIELDS)
    payload = _extract(tmp_path, data_root)
    assert payload["n_valid"] == 0
    assert payload["n_no_verdict"] == 1
    assert payload["physical_active_rate"] is None
    assert "no_verdict" in payload["unavailable_reason"]
    assert payload["per_seed"][0]["status"] == "no_verdict"


def test_physical_only_reports_legacy_unavailable(tmp_path: Path) -> None:
    """只有 physical 口径（旧 legacy 字段缺失）：legacy rate=null + 显式原因。"""
    data_root, _, _ = _build_smoke_root(
        tmp_path,
        seeds_artifact=[100, 101],
        v12_hit_factory=lambda seed: _v12_hit(legacy=None, legacy_type=None),
        write_r1d=False,
        row_factory=lambda seed, artifact_dir: _smoke_row(
            experiment="exp15_speed_v2",
            case="speed7",
            seed=seed,
            hit="",
            hit_type="",
            physical_hit="True",
            physical_hit_type="active",
            artifact_dir=artifact_dir,
        ),
    )
    payload = _extract(tmp_path, data_root)
    assert payload["physical_active_rate"] == pytest.approx(100.0)
    assert payload["legacy_active_rate"] is None
    assert "legacy" in payload["unavailable_reasons"]["legacy"]
    assert payload["physical_hit_semantics_available"] is True


def test_legacy_only_reports_physical_unavailable(tmp_path: Path) -> None:
    """只有 legacy 口径（v2 物理字段缺失）：physical rate=null + 显式原因。"""
    data_root, _, _ = _build_smoke_root(
        tmp_path,
        seeds_artifact=[100, 101],
        v12_hit_factory=lambda seed: _v12_hit(physical=None, physical_type=None),
        row_factory=lambda seed, artifact_dir: _smoke_row(
            experiment="exp15_speed_v2",
            case="speed7",
            seed=seed,
            hit="True",
            hit_type="active",
            physical_hit="",
            physical_hit_type="",
            artifact_dir=artifact_dir,
        ),
    )
    payload = _extract(tmp_path, data_root)
    assert payload["legacy_active_rate"] == pytest.approx(100.0)
    assert payload["physical_active_rate"] is None
    assert "physical" in payload["unavailable_reasons"]["physical"]
    assert payload["legacy_hit_semantics_available"] is True


def test_passive_contact_is_not_active(tmp_path: Path) -> None:
    """passive 接触不是 active：physical_hit=True 但类型 passive 不计入 active。"""
    data_root, _, _ = _build_smoke_root(
        tmp_path,
        seeds_artifact=[100],
        v12_hit_factory=lambda seed: _v12_hit(physical=True, physical_type="passive", legacy=True, legacy_type="passive"),
        r1d_hit_type="passive",
    )
    payload = _extract(tmp_path, data_root)
    assert payload["n_physical_active"] == 0
    assert payload["n_physical_passive"] == 1
    assert payload["physical_active_rate"] is None


def test_csv_only_fallback_when_artifacts_absent(tmp_path: Path) -> None:
    """产物 JSON 缺失但 runs.csv 带 physical 列：按 CSV 口径回退并标注来源。"""
    run_root = tmp_path / "run"
    data_root = run_root / "batch"
    rows = [
        _smoke_row(experiment="exp15_speed_v2", case="speed7", seed=100, artifact_dir=run_root / "artifacts" / "exp15_speed_v2" / "speed7" / "seed100"),
    ]
    _write_csv(run_root / "runs.csv", rows, LEGACY_CSV_FIELDS)
    _write_csv(data_root / "exp15_speed_v2" / "results.csv", rows, LEGACY_CSV_FIELDS)
    payload = _extract(tmp_path, data_root)
    assert payload["n_valid"] == 1
    assert payload["physical_active_rate"] == pytest.approx(100.0)
    assert payload["per_seed"][0]["source"] == "runs_csv"


def test_pos_error_without_verdict_is_not_a_hit(tmp_path: Path) -> None:
    """pos_error 再小也不是命中：无 physical verdict 时不得把 pos_error 当 physical_hit。"""
    data_root, _, _ = _build_smoke_root(
        tmp_path,
        seeds_artifact=[100],
        v12_hit_factory=lambda seed: {
            "pos_error": 0.0,
            "min_dist": 0.0,
            "contact_events": [],
            "message": "",
        },
        write_r1d=False,
        row_factory=lambda seed, artifact_dir: _smoke_row(
            experiment="exp15_speed_v2",
            case="speed7",
            seed=seed,
            hit="",
            hit_type="",
            physical_hit="",
            physical_hit_type="",
            artifact_dir=artifact_dir,
        ),
    )
    payload = _extract(tmp_path, data_root)
    assert payload["n_physical_active"] == 0
    assert payload["physical_active_denominator"] == 0
    assert payload["physical_active_rate"] is None


def test_manifest_expected_seeds_mark_missing_runs(tmp_path: Path) -> None:
    """--manifest 给出预期 seed：缺跑的 seed 记为 missing，不丢失。"""
    data_root, run_root, _ = _build_smoke_root(tmp_path, seeds_artifact=[100])
    manifest = {
        "schema": "r1d_paper_smoke_resolved_v1",
        "tasks": [
            {
                "experiment": "exp15_speed_v2",
                "case_id": "speed7",
                "seed": 100,
                "mode": "torque",
                "identity": {"robot_id": "unitree_r1d", "racket_id": "blade100_v9_reference", "config_hash": "abc"},
            },
            {
                "experiment": "exp15_speed_v2",
                "case_id": "speed7",
                "seed": 101,
                "mode": "torque",
                "identity": {"robot_id": "unitree_r1d", "racket_id": "blade100_v9_reference", "config_hash": "abc"},
            },
        ],
    }
    manifest_path = tmp_path / "manifest_resolved.json"
    _write_json(manifest_path, manifest)
    payload = _extract(tmp_path, data_root, extra=["--manifest", str(manifest_path)])
    assert payload["n_attempted"] == 2
    statuses = {entry["seed"]: entry["status"] for entry in payload["per_seed"]}
    assert statuses[100] == "ok"
    assert statuses[101] == "missing"
    assert payload["physical_active_denominator"] == 1


def test_output_filename_is_stable_and_in_output_dir(tmp_path: Path) -> None:
    """输出文件名/目录稳定（T8c 可预测）：<output>/<experiment>/<case>.json。"""
    data_root, _, _ = _build_smoke_root(tmp_path, seeds_artifact=[100])
    out = tmp_path / "stats"
    proc = _run_script(
        str(EXTRACT_DIR / "extract_r1d_smoke_stats.py"),
        "--experiment",
        "exp15_speed_v2",
        "--case",
        "speed7",
        "--data-root",
        str(data_root),
        "--output",
        str(out),
    )
    assert proc.returncode == 0, proc.stderr
    target = out / "exp15_speed_v2" / "speed7.json"
    assert target.is_file()
    assert "物理" in proc.stdout or "physical" in proc.stdout.lower()


# ---------------------------------------------------------------------------
# 七个既有入口：参数、子调用同源、import 零副作用
# ---------------------------------------------------------------------------


ALL_ENTRY_SCRIPTS = [
    "analyze_all.py",
    "paired_stats.py",
    "aux_stats.py",
    "extract_chat8_aux.py",
    "check_shared_input.py",
    "analyze_chat12_night.py",
    "analyze_exp28_collision.py",
    "extract_r1d_smoke_stats.py",
]


@pytest.mark.parametrize("script", ALL_ENTRY_SCRIPTS)
def test_entry_help_advertises_data_root_and_output(script: str) -> None:
    """每个入口 --help 都必须暴露 --data-root 与 --output。"""
    proc = _run_script(str(EXTRACT_DIR / script), "--help")
    assert proc.returncode == 0, proc.stderr
    assert "--data-root" in proc.stdout
    assert "--output" in proc.stdout


def test_analyze_chat12_night_import_has_no_side_effect(capsys) -> None:
    """import analyze_chat12_night 不得执行分析、不得打印、不得写文件。"""
    import analyze_chat12_night

    assert hasattr(analyze_chat12_night, "main")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ---------------------------------------------------------------------------
# 旧 RM65 口径 fixture（paired_stats 结构兼容）
# ---------------------------------------------------------------------------

LEGACY_ROWS: dict[str, list[dict[str, str]]] = {}


def _legacy_row(experiment: str, cfg: dict[str, object], seed: int, *, hit_type: str = "active", error: str = "") -> dict[str, str]:
    return {
        "experiment": experiment,
        "config_id": f"{experiment}__{seed}",
        "config_json": _cfg(**cfg),
        "seed": str(seed),
        "hit": "True" if hit_type in {"active", "passive"} else "False",
        "hit_type": hit_type,
        "pos_error": "0.01",
        "v_racket_at_hit": "1.1",
        "hit_time_error_ms": "3.0",
        "wall_time": "20.0",
        "error": error,
    }


def _build_legacy_root(tmp_path: Path) -> Path:
    """构造最小旧 RM65 数据根，覆盖 E1/E2/E3/E4/E5/E7/E8 所需 flag。"""
    data_root = tmp_path / "experiment_data"
    experiments: dict[str, list[dict[str, str]]] = {}

    rows_15: list[dict[str, str]] = []
    for speed in (7, 9):
        for seed in (0, 1):
            rows_15.append(_legacy_row("exp15_speed_v2", {"--ball-speed": str(speed)}, seed, hit_type="active" if seed == 0 else "miss"))
    experiments["exp15_speed_v2"] = rows_15

    rows_16: list[dict[str, str]] = []
    for limits, seed in (("configs/real_robot.yaml", 0), (None, 1)):
        cfg = {"--ball-speed": "7"}
        if limits:
            cfg["--limits-config"] = limits
        rows_16.append(_legacy_row("exp16_limits_v2", cfg, seed, hit_type="active" if seed else "miss"))
    experiments["exp16_limits_v2"] = rows_16

    rows_17a: list[dict[str, str]] = []
    for speed, tier, seed in ((7, "full", 0), (7, "none", 0), (9, "full", 0), (9, "none", 0), (12, "full", 0), (12, "none", 0)):
        rows_17a.append(_legacy_row("exp17a_noise", {"--ball-speed": str(speed), "--ablation": tier}, seed))
    experiments["exp17a_noise"] = rows_17a

    rows_17d: list[dict[str, str]] = []
    for speed, tier in ((7, "tube_only"), (7, "softmin_only"), (9, "tube_only"), (9, "softmin_only")):
        rows_17d.append(_legacy_row("exp17d_mechanism", {"--ball-speed": str(speed), "--ablation": tier}, 0))
    experiments["exp17d_mechanism"] = rows_17d

    rows_17b: list[dict[str, str]] = []
    for t_ms, s_m, seed in ((0, 0.1, 0), (25, 0.2, 0)):
        rows_17b.append(
            _legacy_row(
                "exp17b_perturb",
                {"--ball-speed": "9", "--ablation": "full", "--time-perturb-ms": str(t_ms), "--space-perturb-m": str(s_m)},
                seed,
            )
        )
    experiments["exp17b_perturb"] = rows_17b

    rows_17f: list[dict[str, str]] = []
    for t_ms, s_m in ((0, 0.2), (50, 0.2)):
        rows_17f.append(
            _legacy_row(
                "exp17f_mechanism_perturb",
                {"--ball-speed": "9", "--ablation": "tube_only", "--time-perturb-ms": str(t_ms), "--space-perturb-m": str(s_m)},
                0,
            )
        )
    experiments["exp17f_mechanism_perturb"] = rows_17f

    rows_17c: list[dict[str, str]] = []
    rows_17c.append(_legacy_row("exp17c_obsfreq", {"--ball-speed": "9", "--ablation": "full", "--obs-freq": "30"}, 0))
    experiments["exp17c_obsfreq"] = rows_17c

    rows_17g: list[dict[str, str]] = [
        _legacy_row("exp17g_spatial_power", {"--ball-speed": "9", "--ablation": "full", "--space-perturb-m": "0.2"}, 0)
    ]
    experiments["exp17g_spatial_power"] = rows_17g

    rows_17h: list[dict[str, str]] = [
        _legacy_row("exp17h_extreme", {"--ball-speed": "12", "--ablation": "full", "--space-perturb-m": "0.2"}, 0)
    ]
    experiments["exp17h_extreme"] = rows_17h

    rows_17i: list[dict[str, str]] = []
    for limits, tier, seed in (("configs/real_robot.yaml", "full", 0), (None, "full", 1), ("configs/real_robot.yaml", "none", 0), (None, "none", 1)):
        cfg = {"--ball-speed": "7", "--ablation": tier}
        if limits:
            cfg["--limits-config"] = limits
        rows_17i.append(_legacy_row("exp17i_limits_ablation", cfg, seed))
    experiments["exp17i_limits_ablation"] = rows_17i

    rows_24: list[dict[str, str]] = []
    for t_ms, s_m, tier, seed in ((0, 0.2, "full", 0), (0, 0.2, "none", 1), (25, 0.2, "full", 0), (25, 0.2, "none", 1)):
        rows_24.append(
            _legacy_row(
                "exp24_shared_full",
                {
                    "--ball-speed": "9",
                    "--ablation": tier,
                    "--shared-input-full": None,
                    "--time-perturb-ms": str(t_ms),
                    "--space-perturb-m": str(s_m),
                },
                seed,
            )
        )
    experiments["exp24_shared_full"] = rows_24

    for name, rows in experiments.items():
        _write_csv(data_root / name / "results.csv", rows, LEGACY_CSV_FIELDS)
        (data_root / name / "config.yaml").write_text("fixture: true\n", encoding="utf-8")
    return data_root


def test_paired_stats_keeps_legacy_keys_and_is_append_only(tmp_path: Path) -> None:
    """paired_stats 在旧口径根上仍产 E1–E8 + exclusion 顶层键（追加式扩展）。"""
    data_root = _build_legacy_root(tmp_path)
    out = tmp_path / "stats_out"
    proc = _run_script(
        str(EXTRACT_DIR / "paired_stats.py"),
        "--data-root",
        str(data_root),
        "--output",
        str(out),
    )
    assert proc.returncode == 0, proc.stderr
    stats_path = out / "06-stats-active.json"
    assert stats_path.is_file()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for key in (
        "metric",
        "include_passive",
        "E1_speed_sweep",
        "E2_limit_cost",
        "E3_nominal",
        "E4_grid",
        "E5_perception",
        "E7_corners",
        "E8_limit_mechanism",
        "exclusion",
    ):
        assert key in stats, key
    assert "7" in stats["E1_speed_sweep"]
    assert "9" in stats["E1_speed_sweep"]
    assert "TCP1.8" in stats["E2_limit_cost"]
    assert "TCP1.0" in stats["E2_limit_cost"]
    assert "cells" in stats["E3_nominal"] and "paired" in stats["E3_nominal"]
    assert "merged_20cell" in stats["E4_grid"]
    assert "noise_x_kf" in stats["E5_perception"] and "obsfreq" in stats["E5_perception"]
    assert "cells" in stats["E7_corners"]
    assert "did" in stats["E8_limit_mechanism"]
    # 追加的新键（旧键不动）
    assert "data_root" in stats
    assert "sources" in stats
    assert "E9_shared_input" in stats
    assert "tables" in stats
    # 旧口径 cell 结构保留
    cell = next(iter(stats["E1_speed_sweep"].values()))
    for key in ("rows", "n_valid", "hits", "rate", "ci95", "active", "passive", "miss", "passive_share", "errors"):
        assert key in cell, key
    # 不得写回旧 paper 目录（cwd 隔离后再验证真实 paper 未被触碰见集成测试）
    assert not (tmp_path / "paper").exists()


def test_paired_stats_does_not_touch_repo_paper_stats(tmp_path: Path) -> None:
    """即使 --output 指向临时目录，也不得改写仓库 paper/planning 下的旧统计 JSON。"""
    paper_stats = ROOT / "paper" / "planning" / "06-stats-active.json"
    if not paper_stats.is_file():
        pytest.skip("仓库内无旧 paper 统计 JSON")
    before = _sha256(paper_stats)
    data_root = _build_legacy_root(tmp_path)
    proc = _run_script(
        str(EXTRACT_DIR / "paired_stats.py"),
        "--data-root",
        str(data_root),
        "--output",
        str(tmp_path / "out"),
    )
    assert proc.returncode == 0, proc.stderr
    assert _sha256(paper_stats) == before


def test_analyze_all_passes_same_roots_to_submodules(tmp_path: Path) -> None:
    """analyze_all 把同一 --data-root/--output 传给子模块：全部产物落在同一输出目录。"""
    data_root = _build_legacy_root(tmp_path)
    out = tmp_path / "all_out"
    proc = _run_script(
        str(EXTRACT_DIR / "analyze_all.py"),
        "--data-root",
        str(data_root),
        "--output",
        str(out),
    )
    assert proc.returncode == 0, proc.stderr
    for name in (
        "06-stats-active.json",
        "paper_stats_active.json",
        "aux_stats.json",
        "chat8_aux.json",
        "shared_input.json",
        "chat12_night.json",
        "exp28_collision.json",
    ):
        assert (out / name).is_file(), f"子模块未把产物写进同一 --output: {name}"
    assert not (tmp_path / "paper").exists()


def test_submodule_json_missing_data_is_explicit(tmp_path: Path) -> None:
    """子模块遇缺数据：JSON 里必须显式 unavailable，不得静默少一节或造 0。"""
    data_root = tmp_path / "empty_experiment_data"
    data_root.mkdir(parents=True)
    out = tmp_path / "out"
    proc = _run_script(
        str(EXTRACT_DIR / "analyze_all.py"),
        "--data-root",
        str(data_root),
        "--output",
        str(out),
    )
    assert proc.returncode == 0, proc.stderr
    stats = json.loads((out / "06-stats-active.json").read_text(encoding="utf-8"))
    assert stats.get("unavailable"), "缺数据时必须在 stats JSON 里列出 unavailable 项"
    for name in ("aux_stats.json", "chat8_aux.json", "shared_input.json", "chat12_night.json", "exp28_collision.json"):
        payload = json.loads((out / name).read_text(encoding="utf-8"))
        assert isinstance(payload, dict) and payload, name
        assert payload.get("data_root") == str(data_root), name


def test_missing_output_never_writes_default_root(tmp_path: Path) -> None:
    """未传 --output 时只打印：不得写仓库 experiment_data/、paper/ 或相邻 stats/。"""
    repo_aux = ROOT / "experiment_data" / "aux_stats.json"
    repo_paper = ROOT / "paper" / "planning" / "06-stats-active.json"
    before_aux = _sha256(repo_aux) if repo_aux.is_file() else None
    before_paper = _sha256(repo_paper) if repo_paper.is_file() else None
    data_root = tmp_path / "experiment_data"
    data_root.mkdir(parents=True)
    for script in ("aux_stats.py", "analyze_chat12_night.py", "extract_chat8_aux.py"):
        proc = _run_script(str(EXTRACT_DIR / script), "--data-root", str(data_root))
        assert proc.returncode == 0, (script, proc.stderr)
        assert "未给 --output：只打印，不写文件" in proc.stdout, script
    # 非默认数据根下：不得在数据根或兄弟 stats/ 目录落任何产物
    assert not (data_root / "aux_stats.json").exists()
    assert not (tmp_path / "stats").exists()
    assert not (data_root / "chat12_night.json").exists()
    assert not (data_root / "chat8_aux.json").exists()
    # 仓库默认根与 paper/planning 完全未被触碰
    assert (before_aux is None) == (not repo_aux.is_file())
    if before_aux is not None:
        assert _sha256(repo_aux) == before_aux
    assert (before_paper is None) == (not repo_paper.is_file())
    if before_paper is not None:
        assert _sha256(repo_paper) == before_paper
