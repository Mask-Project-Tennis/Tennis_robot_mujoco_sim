"""T8c：R1D 论文链路（paper_figs 数据根/table3 + render_swing_hero 模型选择）测试。

设计原则：
- **先红后绿**：本文件先于实现落地，用于锁定 T8c 要求的可验证行为；
- **不依赖 GL**：渲染相关只测模型选择/路径解析/NPZ 归一化逻辑，不碰 mujoco.Renderer；
- **不补数**：缺 cell 一律 `--`/空串，断言绝不出现 0 伪造值。

测试组织：
1. paper_figs 数据根解析（`resolve_data_root` / `_data_path` / `resolve_stats_json`）；
2. `--out` 语义（输出进临时目录、不写 `paper/`）；
3. table3 隔离预览（合成 stats：present / 缺 cell / physical 缺失 / 缺键报错 / 输出 tex+csv+pdf）；
4. render_swing_hero 模型选择（默认 RM65 / R1D metadata / --robot-config / 非法参数）；
5. hero NPZ 归一化（RM65 exp18 格式 / R1D v12 格式、k_ref fallback、缺字段清晰报错）。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts" / "plot"))
sys.path.insert(0, str(PROJECT / "scripts" / "tools"))

import paper_figs  # noqa: E402
import render_swing_hero as hero  # noqa: E402

TABLE3_CELLS = [(0.0, 0.1), (0.0, 0.2), (0.0, 0.3), (25.0, 0.2), (50.0, 0.2)]
TIER_LABELS = {
    "full": "full",
    "tube_only": "tube-only",
    "softmin_only": "candidate-set-only",
    "none": "point-target",
}


# ---------------------------------------------------------------------------
# 合成 T8b 统计（结构与 paired_stats.build_tables 的 table3_shared_input 一致）
# ---------------------------------------------------------------------------

def _table3_stats(present: dict[tuple[float, float, str], dict] | None = None) -> dict:
    """构造只含 tables.table3_shared_input + column_map 的 T8b 统计。

    Args:
        present: (t, s, tier) → cell 字典（present=True 的格）；未给出的格一律 missing_cell。
    """
    present = present or {}
    rows = []
    for t, s in TABLE3_CELLS:
        cells: dict[str, dict] = {}
        missing: list[str] = []
        for tier in TIER_LABELS:
            if (t, s, tier) in present:
                cells[tier] = {"present": True, "tier_label": TIER_LABELS[tier],
                               **present[(t, s, tier)]}
            else:
                cells[tier] = {"present": False, "tier_label": TIER_LABELS[tier],
                               "unavailable_reason": "missing_cell"}
                missing.append(tier)
        rows.append({"cell": f"t{t:g}|s{s:g}", "t": t, "s": s, "cells": cells,
                     "missing": missing, "p_raw_legacy": None, "n_paired_legacy": None,
                     "p_raw_physical": None, "n_paired_physical": None})
    column_map = {
        "table3_shared_input.tex": {
            "columns": {
                "$(t,s)$ cell": "tables.table3_shared_input.rows[].cell",
                "full": "tables.table3_shared_input.rows[].cells.full.legacy_rate",
                "tube-only": "tables.table3_shared_input.rows[].cells.tube_only.legacy_rate",
                "candidate-set-only":
                    "tables.table3_shared_input.rows[].cells.softmin_only.legacy_rate",
                "point-target": "tables.table3_shared_input.rows[].cells.none.legacy_rate",
                "$p_{raw}$": "tables.table3_shared_input.rows[].p_raw_legacy",
            },
            "note": "缺失 (t,s)/tier 在 rows[].missing / cells[*].present=False 中显式标注",
        }
    }
    return {"tables": {"table3_shared_input": {
        "description": "shared-input 协议（数据源 exp24_shared_full）",
        "columns": ["cell", "full", "tube-only", "candidate-set-only",
                    "point-target", "p_raw"],
        "rows": rows,
    }, "column_map": column_map}}


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _synthetic_npz(path: Path, **arrays) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


# ---------------------------------------------------------------------------
# 1. paper_figs 数据根解析
# ---------------------------------------------------------------------------

def test_resolve_data_root_default_and_relative(tmp_path, monkeypatch):
    """缺省 → PROJECT/experiment_data；相对路径 → 相对 PROJECT 解析。"""
    monkeypatch.setattr(paper_figs, "PROJECT", tmp_path)
    (tmp_path / "experiment_data").mkdir(parents=True, exist_ok=True)
    assert paper_figs.resolve_data_root(None, project=tmp_path) == (tmp_path / "experiment_data").resolve()
    (tmp_path / "experiment_data" / "r1d_v12_parity").mkdir(parents=True)
    got = paper_figs.resolve_data_root(Path("experiment_data/r1d_v12_parity"),
                                       project=tmp_path)
    assert got == (tmp_path / "experiment_data" / "r1d_v12_parity").resolve()


def test_resolve_data_root_missing_raises(tmp_path):
    """数据根不存在必须显式报错（不得静默回退旧根）。"""
    with pytest.raises(FileNotFoundError, match="数据根不存在"):
        paper_figs.resolve_data_root(tmp_path / "nope", project=tmp_path)


def test_data_paths_all_under_data_root(tmp_path, monkeypatch):
    """exp18 npz / exp26 csv / timing.json 三个固定 DATA 读取全部走数据根。"""
    monkeypatch.setattr(paper_figs, "DATA", tmp_path / "root")
    for rel in ("exp18_fig_assets/raw/a_hit_clean.npz",
                "exp26_tcp_grid/results.csv",
                "exp18_fig_assets/timing.json"):
        assert paper_figs._data_path(rel) == (tmp_path / "root" / rel)


def test_resolve_stats_explicit_wins(tmp_path):
    """--stats 显式路径优先于数据根内候选。"""
    explicit = _write_json(tmp_path / "explicit.json", {"E1_speed_sweep": {}})
    _write_json(tmp_path / "root" / "paper_stats_active.json", {"E1_speed_sweep": {"7": {}}})
    assert paper_figs.resolve_stats_json(explicit, data_root=tmp_path / "root") == explicit


def test_resolve_stats_from_data_root_and_missing_lists_candidates(tmp_path):
    """数据根内解析统计 JSON；缺失时错误信息列出候选路径。"""
    root = tmp_path / "root"
    target = _write_json(root / "paper_stats_active.json", {"tables": {}})
    assert paper_figs.resolve_stats_json(None, data_root=root) == target
    with pytest.raises(FileNotFoundError) as excinfo:
        paper_figs.resolve_stats_json(None, data_root=tmp_path / "empty")
    msg = str(excinfo.value)
    assert "paper_stats_active.json" in msg and "06-stats-active.json" in msg


# ---------------------------------------------------------------------------
# 2. --out 语义：写到临时目录，不写 paper/
# ---------------------------------------------------------------------------

def test_main_table3_writes_to_out_and_not_paper(tmp_path, monkeypatch):
    """`--fig table3 --out <tmp>`：产物全在 tmp 下，paper/ 不新增文件。"""
    stats = _write_json(tmp_path / "stats.json", _table3_stats(
        present={(0.0, 0.1, "full"): {"legacy_rate": 42.0, "legacy_n": 10,
                                      "physical_rate": 50.0,
                                      "physical_denominator": 10,
                                      "unavailable_reason": None}}))
    out = tmp_path / "figures"
    paper_before = {p for p in (PROJECT / "paper").rglob("*") if p.is_file()}
    monkeypatch.setattr(sys, "argv", ["paper_figs.py", "--fig", "table3",
                                      "--stats", str(stats), "--out", str(out)])
    paper_figs.main()
    produced = {p for p in out.rglob("*") if p.is_file()}
    assert produced, "table3 预览未产生任何文件"
    paper_after = {p for p in (PROJECT / "paper").rglob("*") if p.is_file()}
    assert paper_after == paper_before, "paper/ 下出现了新增文件（--out 必须控制所有输出）"


# ---------------------------------------------------------------------------
# 3. table3 隔离预览
# ---------------------------------------------------------------------------

def test_table3_preview_legacy_physical_separated():
    """present 格的 legacy/physical 各自成表，physical 缺失不得拿 legacy 顶替。"""
    stats = _table3_stats(present={
        (0.0, 0.1, "full"): {"legacy_rate": 42.0, "legacy_n": 10,
                             "physical_rate": 50.0, "physical_denominator": 10,
                             "unavailable_reason": None},
        (0.0, 0.2, "full"): {"legacy_rate": 33.0, "legacy_n": 9,
                             "physical_rate": None, "physical_denominator": 0,
                             "unavailable_reason": "no_physical_verdict"},
    })
    preview = paper_figs.build_table3_preview(stats)
    rows = {r["cell"]: r for r in preview["rows"]}
    assert rows["t0|s0.1"]["legacy"]["full"] == "42.0"
    assert rows["t0|s0.1"]["physical"]["full"] == "50.0"
    # physical 缺失：physical 列 `--`，legacy 列仍是真实值（互不顶替）
    assert rows["t0|s0.2"]["legacy"]["full"] == "33.0"
    assert rows["t0|s0.2"]["physical"]["full"] == "--"


def test_table3_missing_and_nan_cells_never_render_zero():
    """缺 cell（present=False）与 NaN 率一律 `--`，绝不画 0。"""
    stats = _table3_stats(present={
        (0.0, 0.3, "none"): {"legacy_rate": float("nan"), "legacy_n": 0,
                             "physical_rate": None, "physical_denominator": 0,
                             "unavailable_reason": "no_valid_rows"},
    })
    preview = paper_figs.build_table3_preview(stats)
    rows = {r["cell"]: r for r in preview["rows"]}
    missing = rows["t50|s0.2"]
    assert missing["legacy"]["full"] == "--" and missing["physical"]["full"] == "--"
    nan_row = rows["t0|s0.3"]
    assert nan_row["legacy"]["none"] == "--"
    for row in preview["rows"]:
        for panel in ("legacy", "physical"):
            for value in row[panel].values():
                assert value != "0", f"{row['cell']} {panel} 出现伪造 0"


def test_table3_missing_key_clear_error():
    """缺 tables.table3_shared_input 时给清晰错误（不是裸 KeyError 的 traceback 语义）。"""
    with pytest.raises(ValueError, match="table3_shared_input"):
        paper_figs.build_table3_preview({"tables": {"column_map": {}}})


def test_table3_independent_of_legacy_stats_keys():
    """table3 只依赖 tables 键，不触碰 E1–E8（隔离版）。"""
    stats = _table3_stats(present={(0.0, 0.1, "full"): {
        "legacy_rate": 10.0, "legacy_n": 1, "physical_rate": 10.0,
        "physical_denominator": 1, "unavailable_reason": None}})
    assert set(stats) == {"tables"}
    preview = paper_figs.build_table3_preview(stats)
    assert preview["rows"]


def test_table3_writes_tex_csv_pdf(tmp_path):
    """产出 tex + csv + pdf 三件套，含“小样本、仅链路预览”标注。"""
    stats = _table3_stats(present={(0.0, 0.1, "full"): {
        "legacy_rate": 42.0, "legacy_n": 10, "physical_rate": 50.0,
        "physical_denominator": 10, "unavailable_reason": None}})
    paths = paper_figs.table3_shared_input(stats, out_dir=tmp_path)
    assert set(paths) == {"tex", "csv", "pdf"}
    for p in paths.values():
        assert Path(p).is_file() and Path(p).stat().st_size > 0
    tex = paths["tex"].read_text(encoding="utf-8")
    assert "仅链路预览" in tex and "不作论文统计结论" in tex
    with paths["csv"].open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(TABLE3_CELLS)
    missing_row = next(r for r in rows if r["cell"] == "t50|s0.2")
    assert missing_row["full_legacy"] == "" and missing_row["full_physical"] == ""
    assert paths["pdf"].read_bytes()[:4] == b"%PDF"


# ---------------------------------------------------------------------------
# 4. render_swing_hero 模型选择
# ---------------------------------------------------------------------------

def test_hero_default_model_is_rm65():
    """缺省（无 metadata / 无参数）＝原 RM65 行为。"""
    sel = hero.resolve_model_selection(model_xml=None, robot_config=None, metadata=None)
    assert sel.model_xml == (PROJECT / "src" / "robot" / "rm65_model.xml")
    assert sel.robot_id == "rm65"


def test_hero_model_selection_from_r1d_metadata():
    """metadata.robot_id=unitree_r1d → r1d_model_right.xml（7 关节模型）。"""
    sel = hero.resolve_model_selection(
        model_xml=None, robot_config=None,
        metadata={"robot_id": "unitree_r1d", "joint_names": ["right_wrist_yaw_joint"]})
    assert sel.robot_id == "unitree_r1d"
    assert sel.model_xml.name == "r1d_model_right.xml"
    assert sel.model_xml.is_file()


def test_hero_model_selection_from_robot_config(tmp_path):
    """--robot-config 读 model_path（相对配置文件解析）并识别左/右侧。"""
    cfg = tmp_path / "r1d_left.yaml"
    cfg.write_text("robot_id: unitree_r1d\n"
                   "model_path: ../../src/robot/r1d_model_left.xml\n",
                   encoding="utf-8")
    sel = hero.resolve_model_selection(model_xml=None, robot_config=cfg, metadata=None)
    assert sel.robot_id == "unitree_r1d"
    assert sel.model_xml.name == "r1d_model_left.xml"
    assert sel.model_xml.is_file()


def test_hero_explicit_model_xml_wins_and_missing_raises(tmp_path):
    """--model-xml 显式优先；文件不存在必须清晰报错。"""
    xml = tmp_path / "custom.xml"
    xml.write_text("<mujoco/>", encoding="utf-8")
    sel = hero.resolve_model_selection(model_xml=xml, robot_config=None, metadata=None)
    assert sel.model_xml == xml
    with pytest.raises(FileNotFoundError, match="模型 XML 不存在"):
        hero.resolve_model_selection(model_xml=tmp_path / "nope.xml",
                                     robot_config=None, metadata=None)


def test_hero_unknown_robot_id_clear_error():
    """不支持的 robot_id 给清晰错误（不得静默套用 RM65 7→6 关节）。"""
    with pytest.raises(ValueError, match="robot_id"):
        hero.resolve_model_selection(model_xml=None, robot_config=None,
                                     metadata={"robot_id": "d2_unknown"})


def test_hero_output_path_outside_paper(tmp_path):
    """--out 可到任意目录；指向 paper/ 时拒绝写入。"""
    assert hero.resolve_output_path(tmp_path / "hero.png") == tmp_path / "hero.png"
    rel = hero.resolve_output_path("results/r1d_v12_parity/paper_smoke/hero/hero.png")
    assert rel == PROJECT / "results/r1d_v12_parity/paper_smoke/hero/hero.png"
    with pytest.raises(ValueError, match="paper"):
        hero.resolve_output_path(PROJECT / "paper" / "figures" / "hero.png")


# ---------------------------------------------------------------------------
# 5. hero NPZ 归一化（不依赖 GL）
# ---------------------------------------------------------------------------

def _rm65_episode(path: Path) -> Path:
    """exp18 风格 NPZ：q_actual/tcp_pos/ball_pos/dt/hit_step/metadata。"""
    n, hit = 40, 20
    q = np.tile(np.linspace(-0.2, 0.2, 6), (n, 1))
    ball = np.linspace([-1.0, -4.0, 1.2], [-0.6, -0.5, 1.0], n)
    tcp = ball + np.array([0.0, 0.0, 0.05])
    return _synthetic_npz(
        path, q_actual=q, tcp_pos=tcp, ball_pos=ball, init_q_left=np.zeros(6),
        dt=np.array(0.005), hit_step=np.array(hit),
        metadata=np.array(json.dumps({"ball_speed": 9.0, "hit_type": "active"})))


def _r1d_episode(path: Path) -> Path:
    """R1D v12 风格 NPZ：X(N,14)/U(N-1,7)/ball_pos/ball_vel/time/racket_pos。"""
    n, hit = 40, 20
    q = np.tile(np.linspace(-0.3, 0.3, 7), (n, 1))
    X = np.concatenate([q, np.zeros((n, 7))], axis=1)
    ball = np.linspace([-1.0, -4.0, 1.2], [-0.6, -0.5, 1.0], n)
    racket = ball + np.array([0.0, 0.0, 0.05])
    racket[hit] = ball[hit]            # 第 hit 步拍心与球心重合 → 可反推 hit_step
    return _synthetic_npz(
        path, X=X, U=np.zeros((n - 1, 7)), ball_pos=ball,
        ball_vel=np.tile(np.array([0.0, 8.0, 0.0]), (n, 1)),
        time=np.arange(n) * 0.005, racket_pos=racket)


def test_hero_load_rm65_episode(tmp_path):
    """exp18 格式：6 关节、metadata/hit_step 直读，k_ref 缺 metadata 时明确 fallback。"""
    ep = hero.load_episode(_rm65_episode(tmp_path / "a_hit.npz"))
    assert ep.n_joints == 6
    assert ep.hit_step == 20
    assert ep.q_actual.shape == (40, 6)
    assert ep.robot_id == "rm65"
    assert ep.dt == pytest.approx(0.005)
    k_ref, source = hero.resolve_k_ref(ep)
    assert k_ref == 20 and "fallback" in source.lower()


def test_hero_load_r1d_episode_7joints(tmp_path):
    """R1D 格式：X[:, :7] 为 7 关节，hit_step 由球-拍最近距离反推，k_ref fallback。"""
    ep = hero.load_episode(_r1d_episode(tmp_path / "v12_episode.npz"),
                           robot_config=PROJECT / "configs" / "robots" / "r1d_right.yaml")
    assert ep.n_joints == 7
    assert ep.q_actual.shape == (40, 7)
    assert ep.robot_id == "unitree_r1d"
    assert ep.hit_step == 20
    k_ref, source = hero.resolve_k_ref(ep)
    assert k_ref == 20 and "fallback" in source.lower()


def test_hero_r1d_missing_racket_pos_clear_error(tmp_path):
    """R1D NPZ 缺 racket_pos 时清晰报错（提示用 v12_episode.npz）。"""
    p = _r1d_episode(tmp_path / "bad.npz")
    z = dict(np.load(p))
    z.pop("racket_pos")
    bad = _synthetic_npz(tmp_path / "v12_no_racket.npz", **{
        k: v for k, v in z.items()})
    with pytest.raises(ValueError, match="racket_pos"):
        hero.load_episode(bad, robot_config=PROJECT / "configs" / "robots" / "r1d_right.yaml")


def test_hero_rejects_miss_episode_clear_error(tmp_path):
    """v12 sidecar 记录 miss（hit_step=-1）时应给出清晰错误，而不是走廊 NaN 崩溃。"""
    p = _r1d_episode(tmp_path / "v12_episode.npz")
    p.with_suffix(".json").write_text(
        json.dumps({"robot_id": "unitree_r1d", "dt": 0.005,
                    "hit": {"physical_hit": False, "legacy_hit": False,
                             "physical_hit_step": -1, "legacy_hit_step": -1}}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="miss|无命中"):
        hero.load_episode(p, robot_config=PROJECT / "configs" / "robots" / "r1d_right.yaml")
