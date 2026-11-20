"""R1D smoke 单 case 统计绘图测试（T8c 后续集成）。

覆盖：
1. T8b 统计 schema 的加载/校验（缺文件、schema 不符、缺字段）；
2. physical/legacy 双口径摘要：可估计时保留百分比，不可估计时 rate 必须为 None
   （绝不补 0）；
3. PNG + JSON 摘要落盘，JSON 中不可估计字段保持 null；
4. 未给 ``--output`` 时只打印、不写盘。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts" / "plot"))

import plot_r1d_smoke_stats as plot_mod  # noqa: E402


def _stats_payload(*, physical_rate: float | None, legacy_rate: float | None) -> dict:
    """构造最小可加载的 T8b 单 case 统计字典。"""
    physical_denom = 2
    legacy_denom = 2
    physical_reason = None if physical_rate is not None else "all_miss"
    legacy_reason = None if legacy_rate is not None else "all_miss"
    return {
        "schema": "r1d_smoke_case_stats_v1",
        "experiment_id": "exp15_speed_v2",
        "case_id": "speed7",
        "mode": "torque",
        "robot_id": "unitree_r1d",
        "metric_schema": "v12_episode_v2_physical_hit",
        "n_attempted": 2,
        "n_valid": 2,
        "n_error": 0,
        "n_timeout": 0,
        "n_no_verdict": 0,
        "n_missing": 0,
        "n_physical_active": 1 if physical_rate is not None else 0,
        "n_physical_passive": 0,
        "n_physical_miss": 1 if physical_rate is not None else 2,
        "n_legacy_active": 1 if legacy_rate is not None else 0,
        "n_legacy_passive": 0,
        "n_legacy_miss": 1 if legacy_rate is not None else 2,
        "physical_active_rate": physical_rate,
        "physical_active_denominator": physical_denom,
        "physical_ci95": [0.0, 90.0] if physical_rate is not None else [0.0, 65.76],
        "legacy_active_rate": legacy_rate,
        "legacy_active_denominator": legacy_denom,
        "legacy_ci95": [0.0, 90.0] if legacy_rate is not None else [0.0, 65.76],
        "unavailable_reasons": {
            "physical": physical_reason,
            "legacy": legacy_reason,
        },
        "unavailable_reason": "all_miss" if physical_rate is None and legacy_rate is None else None,
        "per_seed": [
            {"seed": 100, "status": "ok"},
            {"seed": 101, "status": "ok"},
        ],
    }


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    """缺文件必须报 FileNotFoundError 并给出生成命令提示。"""
    with pytest.raises(FileNotFoundError, match="extract_r1d_smoke_stats"):
        plot_mod.load_case_stats(tmp_path, "exp15_speed_v2", "speed7")


def test_load_rejects_wrong_schema(tmp_path: Path) -> None:
    """schema 不符必须 ValueError。"""
    target = tmp_path / "exp15_speed_v2" / "speed7.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema 不符"):
        plot_mod.load_case_stats(tmp_path, "exp15_speed_v2", "speed7")


def test_rates_available() -> None:
    """可估计时保留百分比与真实分母。"""
    payload = _stats_payload(physical_rate=50.0, legacy_rate=50.0)
    physical, legacy = plot_mod.build_case_rates(payload)
    assert physical.available is True
    assert physical.rate == pytest.approx(50.0)
    assert physical.denominator == 2
    assert legacy.available is True
    assert legacy.rate == pytest.approx(50.0)


def test_rates_unavailable_keeps_none() -> None:
    """不可估计时 rate 必须 None，reason 保留，分母仍记录。"""
    payload = _stats_payload(physical_rate=None, legacy_rate=None)
    physical, legacy = plot_mod.build_case_rates(payload)
    assert physical.available is False
    assert physical.rate is None
    assert physical.unavailable_reason == "all_miss"
    assert physical.denominator == 2
    assert legacy.available is False
    assert legacy.rate is None
    assert legacy.unavailable_reason == "all_miss"


def test_render_writes_png_and_json_with_nulls(tmp_path: Path) -> None:
    """渲染落盘；JSON 中不可估计字段保持 null，不写 0。"""
    payload = _stats_payload(physical_rate=None, legacy_rate=None)
    physical, legacy = plot_mod.build_case_rates(payload)
    png_path, json_path = plot_mod.render_case_figure(payload, physical, legacy, tmp_path)
    assert png_path.is_file()
    assert png_path.stat().st_size > 0
    assert json_path.is_file()
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["schema"] == plot_mod.PLOT_SCHEMA
    assert summary["physical"]["active_rate"] is None
    assert summary["legacy"]["active_rate"] is None
    assert summary["physical"]["unavailable_reason"] == "all_miss"
    assert summary["legacy"]["unavailable_reason"] == "all_miss"


def test_main_without_output_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """未给 --output 时只打印、不创建输出目录。"""
    stats_root = tmp_path / "stats"
    case_dir = stats_root / "exp15_speed_v2"
    case_dir.mkdir(parents=True)
    (case_dir / "speed7.json").write_text(
        json.dumps(_stats_payload(physical_rate=50.0, legacy_rate=50.0)),
        encoding="utf-8",
    )
    rc = plot_mod.main([
        "--stats-root", str(stats_root),
        "--experiment", "exp15_speed_v2",
        "--case", "speed7",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "physical=50.0%" in captured.out
    assert "未给 --output" in captured.out
    assert not (tmp_path / "out").exists()
