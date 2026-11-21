"""论文图表数据层：仓库/数据根/输出目录寻址与统计 JSON 读取。

本模块只做路径解析与 JSON 读取，不做任何绘图，也不 import matplotlib；
``PROJECT`` / ``DATA`` / ``OUT`` 是模块级可变状态（``main()`` 的
``--data-root`` / ``--out`` 会改写后两者），绘图模块通过
``from . import data as _data`` 在运行时读取。
"""

from __future__ import annotations

import json
from pathlib import Path

# 本文件位于 <repo>/scripts/plot/paper/data.py，向上 4 层才是仓库根
PROJECT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "experiment_data"
OUT = PROJECT / "paper" / "figures"


def resolve_data_root(path: Path | None, project: Path | None = None) -> Path:
    """解析数据根：None → `<project>/experiment_data`；相对路径相对 `project` 解析。

    Args:
        path: 命令行 --data-root（None 表示沿用旧默认根）。
        project: 仓库根目录（None → 模块级 PROJECT；测试可注入临时目录）。

    Returns:
        绝对路径形式的数据根。

    Raises:
        FileNotFoundError: 解析后的数据根不存在（显式报错，不静默回退旧根）。
    """
    project = PROJECT if project is None else Path(project)
    if path is None:
        root = project / "experiment_data"
    else:
        root = Path(path)
        if not root.is_absolute():
            root = project / root
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"数据根不存在: {root}")
    return root


def _data_path(rel: str | Path) -> Path:
    """固定数据资产路径（exp18 npz / exp26 csv / timing.json）统一走当前 DATA 根。

    Args:
        rel: 相对数据根的资产路径。

    Returns:
        数据根下的绝对路径（不检查存在性，由调用方决定缺失时的语义）。
    """
    return Path(DATA) / rel


def resolve_stats_json(explicit: Path | None = None,
                       project: Path | None = None,
                       data_root: Path | None = None) -> Path:
    """解析统计 JSON：显式路径 > 数据根候选 > 旧 paper 工作区/主仓库副本。

    Args:
        explicit: `--stats` 显式路径（最优先）。
        project: 仓库根（旧默认语义用）。
        data_root: `--data-root` 解析后的数据根；给出时只在根内候选查找，
            找不到即显式报错并列出候选路径（不得静默回退旧根）。

    Returns:
        统计 JSON 路径。

    Raises:
        FileNotFoundError: 显式路径不存在，或数据根/旧位置均无统计 JSON。
    """
    if explicit is not None:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"统计 JSON 不存在: {p}")
        return p
    project = PROJECT if project is None else Path(project)
    if data_root is not None:
        root = Path(data_root)
        candidates = [root / "paper_stats_active.json",
                      root / "06-stats-active.json",
                      root / "stats" / "paper_stats_active.json",
                      root / "stats" / "06-stats-active.json",
                      root.parent / "stats" / "paper_stats_active.json",
                      root.parent / "stats" / "06-stats-active.json"]
        hint = "在数据根内未找到统计 JSON；候选路径:\n  " + "\n  ".join(
            str(c) for c in candidates)
    else:
        candidates = [project / "paper" / "planning" / "06-stats-active.json",
                      project / "experiment_data" / "paper_stats_active.json"]
        hint = "未找到统计 JSON；候选路径:\n  " + "\n  ".join(
            str(c) for c in candidates)
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(hint + "\n请先运行 scripts/extract/paired_stats.py")


def load_stats(explicit: Path | None = None,
               data_root: Path | None = None) -> dict:
    """读取配对统计 JSON（active-hit 口径，图与正文同源）。"""
    path = resolve_stats_json(explicit, data_root=data_root)
    print(f"统计 JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
