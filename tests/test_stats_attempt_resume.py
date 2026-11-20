"""续跑执行尝试统计回归 + exp28 扰动格判定单元测试。

覆盖第三轮复审两项 P2：

1. **失败任务续跑后无法统计（P2-3）**：``batch_runner --resume`` 保留原失败行、
   追加成功行，两行共享同一 ``RecordIdentity.unique_key()``。``paired_stats``
   必须把“执行尝试”与“统计样本”分开：同一身份允许多条失败尝试 + 至多一条
   完成尝试，统计只计最终尝试（完成优先，否则最后一条失败行），但 CSV 文件行
   不得被重写/静默丢弃；两条完成行仍按重复记录拒绝。
2. **exp28 无扰动对照被错误分类（P2-4）**：``analyze_exp28_collision._cell``
   只能凭显式扰动配置判定扰动格；``space_perturb_of`` 的缺省 0.0 不是扰动
   证据，否则 20 条名义对照会被误记为 (0, 0.0) 并跳过 exp15/exp17 名义来源。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTRACT_DIR = ROOT / "scripts" / "extract"
sys.path.insert(0, str(EXTRACT_DIR))

import analyze_exp28_collision  # noqa: E402  （被测模块）
import paired_stats  # noqa: E402  （被测模块）
from test_paper_stats_catalog import (  # noqa: E402  （复用合成数据根工具）
    RESULT_FIELDS,
    RUNS_FIELDS,
    Dataset,
    Spec,
    _e3_cell,
    _append_csv_rows,
    _paired_stats,
    _stats_payload,
    _write_csv,
)

# ---------------------------------------------------------------------------
# 续跑恢复：同一身份的失败行 + 完成行
# ---------------------------------------------------------------------------


def _resume_case(tmp_path: Path) -> tuple[Dataset, Path, Path]:
    """构造“原失败行在前、成功行追加”的续跑结果（两视图同身份、同 config_json）。

    Returns:
        ``(dataset, results_path, runs_path)``；两个 CSV 都保留失败尝试行。
    """
    dataset = Dataset(tmp_path / "root").add(
        Spec("exp17d_mechanism", "speed7", mode="torque", tier="full", seeds=(100,))
    ).write()
    results_path = dataset.data_root / "exp17d_mechanism" / "results.csv"
    success = dict(dataset.results_rows["exp17d_mechanism"][0])
    failed = dict(success)
    failed.update({
        "error": "no_result",
        "exit_code": "3",
        "timed_out": "0",
        "hit": "False",
        "hit_type": "",
    })
    _write_csv(results_path, [failed, success], RESULT_FIELDS)

    runs_path = dataset.run_root / "runs.csv"
    runs_success = dict(dataset.runs_rows[0])
    runs_failed = dict(runs_success)
    runs_failed.update({
        "error": "no_result",
        "exit_code": "3",
        "timed_out": "0",
        "hit": "False",
        "hit_type": "",
        "physical_hit": "",
        "physical_hit_type": "",
        "physical_hit_step": "",
        "contact_event_count": "",
    })
    _write_csv(runs_path, [runs_failed, runs_success], RUNS_FIELDS)
    return dataset, results_path, runs_path


def _run_stats(dataset: Dataset, out: Path) -> Any:
    """以 catalog + torque 模式跑真实统计 CLI。"""
    return _paired_stats(
        "--data-root", str(dataset.data_root),
        "--catalog", str(dataset.catalog_path),
        "--mode", "torque",
        "--output", str(out),
    )


def test_resume_failed_then_completed_counts_one_sample(tmp_path: Path) -> None:
    """失败尝试 + 完成尝试同身份：rc0、只计 1 个样本、失败行仍在 CSV。"""
    dataset, results_path, runs_path = _resume_case(tmp_path)
    out = tmp_path / "stats"
    proc = _run_stats(dataset, out)
    assert proc.returncode == 0, proc.stderr

    stats = _stats_payload(out)
    cell = _e3_cell(stats, "7|full")
    assert (cell["rows"], cell["n_valid"], cell["hits"]) == (1, 1, 1)
    assert cell["errors"] == {}
    assert _e3_cell(stats, "7|full", physical=True)["denominator"] == 1

    validation = stats["validation"]
    assert validation["results_records"] == 2   # 文件行全部保留
    assert validation["runs_records"] == 2
    assert validation["mode_records"] == 1      # 统计样本只计最终完成尝试
    assert validation["attempts_ignored"] == 2  # results + runs 各 1 条失败尝试
    samples = validation["attempts_ignored_samples"]
    assert [sample["view"] for sample in samples] == ["results", "runs"]
    assert all(sample["line"] == 2 for sample in samples)
    assert all(
        sample["reason"] == "superseded_failed_attempt" for sample in samples
    )

    # 统计只读：两个视图的失败行都没有被重写或删除
    for path in (results_path, runs_path):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert rows[0]["error"] == "no_result"
        assert rows[1]["error"] == ""


def test_two_completed_attempts_same_identity_still_raise(tmp_path: Path) -> None:
    """失败行之后出现两条完成行：仍是重复记录，CLI 必须非零退出。"""
    dataset, results_path, _ = _resume_case(tmp_path)
    _append_csv_rows(
        results_path,
        dataset.results_rows["exp17d_mechanism"][:1],
        RESULT_FIELDS,
    )
    proc = _run_stats(dataset, tmp_path / "stats")
    assert proc.returncode != 0
    assert "重复" in proc.stderr
    assert "results.csv" in proc.stderr


# ---------------------------------------------------------------------------
# 最终尝试选择（内部语义）
# ---------------------------------------------------------------------------


def _attempt_row(*, error: str = "", timed_out: str = "False") -> dict[str, str]:
    """一条最小可识别记录；同一身份由相同实验/config_json/seed/产物路径决定。"""
    return {
        "experiment": "exp17d_mechanism",
        "config_id": "speed7__torque__full__s7__tcp1.8",
        "config_json": json.dumps({"--ball-speed": 7, "--ablation": "full"}),
        "seed": "100",
        "mode": "torque",
        "robot_id": "r1d_right",
        "racket_id": "blade100_v9_reference",
        "artifact_dir": "artifacts/exp17d_mechanism/speed7/seed100",
        "hit": "True",
        "error": error,
        "timed_out": timed_out,
    }


CSV_PATH = Path("results.csv")


def test_final_attempt_prefers_completed_over_failed() -> None:
    failed = _attempt_row(error="no_result")
    success = _attempt_row()
    selected, ignored = paired_stats._final_attempt_rows([failed, success], CSV_PATH)
    assert selected == [success]
    assert [item["line"] for item in ignored] == [2]
    assert ignored[0]["reason"] == "superseded_failed_attempt"


def test_final_attempt_keeps_last_failed_when_never_completed() -> None:
    first = _attempt_row(error="no_result")
    second = _attempt_row(error="timeout", timed_out="1")
    selected, ignored = paired_stats._final_attempt_rows([first, second], CSV_PATH)
    assert selected == [second]
    assert [item["line"] for item in ignored] == [2]


def test_final_attempt_completed_wins_even_before_later_failure() -> None:
    success = _attempt_row()
    late_failure = _attempt_row(error="no_result")
    selected, ignored = paired_stats._final_attempt_rows(
        [success, late_failure], CSV_PATH
    )
    assert selected == [success]
    assert [item["line"] for item in ignored] == [3]


def test_two_completed_attempts_raise_duplicate() -> None:
    with pytest.raises(paired_stats.DuplicateRecordError, match="重复完成记录"):
        paired_stats._final_attempt_rows(
            [_attempt_row(), _attempt_row()], CSV_PATH
        )


def test_timed_out_row_is_not_completed() -> None:
    """timeout 行即使 error 为空也不算完成尝试。"""
    timed_out = _attempt_row(timed_out="1")
    success = _attempt_row()
    selected, _ = paired_stats._final_attempt_rows([timed_out, success], CSV_PATH)
    assert selected == [success]


# ---------------------------------------------------------------------------
# exp28 扰动格判定
# ---------------------------------------------------------------------------


def _cell_row(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"config_json": json.dumps(cfg, ensure_ascii=False)}


def test_cell_nominal_row_default_zero_is_not_perturbation() -> None:
    """缺省 space_perturb_of=0.0 不能证明存在扰动（修复前的误分类根因）。"""
    row = _cell_row({"--ball-speed": 9, "--ablation": "none"})
    assert paired_stats.space_perturb_of(row) == 0.0
    assert analyze_exp28_collision._cell(row) is None


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        # exp28 名义对照行：无任何扰动 flag
        ({"--serve-box": None, "--ball-speed": 9, "--ablation": "full"}, None),
        # RM65：--random-perturb 出现即扰动格，缺省空间扰动记 0
        ({"--random-perturb": None, "--time-perturb-ms": 25}, (25, 0.0)),
        # RM65：显式空间/时间扰动
        ({"--time-perturb-ms": 25, "--space-perturb-m": 0.2}, (25, 0.2)),
        # 显式 0.0 仍是显式扰动配置（时间扰动格 (0, 0.0)）
        ({"--space-perturb-m": 0.0}, (0, 0.0)),
        # R1D：--task-overrides 显式含 mpc.space_perturb_m
        (
            {"--task-overrides": ["mpc.space_perturb_m=0.1"],
             "--time-perturb-ms": 10},
            (10, 0.1),
        ),
        # R1D：时间扰动经 override 换算为 ms
        (
            {"--task-overrides": ["mpc.space_perturb_m=0.2",
                                  "mpc.time_perturb_s=0.05"]},
            (50, 0.2),
        ),
        # R1D 其他 override（显式消融）不是扰动证据
        ({"--task-overrides": ["cost.tube.Q_p_tube=0.0"]}, None),
    ],
)
def test_cell_requires_explicit_perturbation_config(
    cfg: dict[str, Any], expected: tuple[int, float] | None
) -> None:
    assert analyze_exp28_collision._cell(_cell_row(cfg)) == expected


def test_load_nominal_unknown_tier_returns_empty(tmp_path: Path) -> None:
    """新分类路径可能把 corridor 名义行路由到 _load_nominal；缺来源时不得崩溃。"""
    assert analyze_exp28_collision._load_nominal(tmp_path, "corridor") == {}
