"""paper_figs 统计 JSON 路径解析测试（回退顺序与显式优先）。"""
from pathlib import Path

import pytest

from scripts.plot.paper_figs import resolve_stats_json


def test_explicit_path_wins(tmp_path: Path) -> None:
    """显式路径优先于任何回退候选。"""
    p = tmp_path / "stats.json"
    p.write_text("{}", encoding="utf-8")
    assert resolve_stats_json(p, project=tmp_path) == p


def test_prefers_paper_planning_copy(tmp_path: Path) -> None:
    """paper/planning 与 experiment_data 同时存在时，优先 paper 侧。"""
    paper = tmp_path / "paper" / "planning"
    paper.mkdir(parents=True)
    (paper / "06-stats-active.json").write_text("{}", encoding="utf-8")
    data = tmp_path / "experiment_data"
    data.mkdir()
    (data / "paper_stats_active.json").write_text("{}", encoding="utf-8")
    assert resolve_stats_json(None, project=tmp_path) == paper / "06-stats-active.json"


def test_falls_back_to_experiment_data(tmp_path: Path) -> None:
    """paper 侧缺失时回退到 experiment_data/paper_stats_active.json。"""
    data = tmp_path / "experiment_data"
    data.mkdir()
    (data / "paper_stats_active.json").write_text("{}", encoding="utf-8")
    assert resolve_stats_json(None, project=tmp_path) == data / "paper_stats_active.json"


def test_raises_with_hint_when_missing(tmp_path: Path) -> None:
    """两者都缺失时报错并提示先运行 paired_stats.py。"""
    with pytest.raises(FileNotFoundError, match="paired_stats"):
        resolve_stats_json(None, project=tmp_path)
