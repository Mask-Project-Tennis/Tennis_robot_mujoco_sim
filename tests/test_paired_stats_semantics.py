"""paired_stats 语义回归测试（R03/R04/R05/R06）。

R03：physical 口径以 ``physical_hit_type`` 为准，被动接触不再算主动命中。
R04：无 ``--task-overrides`` 的 R1D 行识别为默认 ``full`` 档（RM65 仍为空档）。
R05：TCP 标签固定 1 位小数（1.0 → ``TCP1.0``，不再被 ``:g`` 压成 ``TCP1``）。
R06：两档共同 seed 全 miss 时 CI/p 值不可估计（None），不输出零宽区间。
M02：前馈关闭（``actuator.feedforward=false``）是独立档位，不与 ``full`` 混算；
同 seed 重复且取值冲突的配对整键排除，结果与 CSV 行序无关。
第二轮复审：R05 富化冲突覆盖 producer 原样复制的共有结果列；R06 DiD 四个条件
共用唯一索引（冲突 seed 整键排除、行序无关）；R07 ``--no-feedforward`` 覆盖
``--task-overrides``（与真实 V12 配置解析优先级一致）。
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from scripts.extract.paired_stats import (
    HIT_SEMANTICS,
    MODES,
    StatsContext,
    _SHARED_RESULT_COLUMNS,
    _did,
    _enrichment_conflicts,
    _enrichment_field_names,
    _enrichment_signature,
    _feedforward_of_cfg,
    _load_rows,
    _tcp_label,
    e3_nominal,
    make_context,
    paired_test,
    paired_test_physical,
    physical_of,
    physical_stats,
    tcp_label_of,
    tier_of,
)


def _row(cfg: dict[str, Any], seed: int = 100, *,
         physical_hit: bool | None = None,
         physical_hit_type: str | None = None,
         error: str = "") -> dict[str, str]:
    """构造一行仅含 paired_stats 所需列的伪 CSV 记录。"""
    row = {
        "config_json": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
        "seed": str(seed),
        "error": error,
    }
    if physical_hit is not None:
        row["physical_hit"] = "True" if physical_hit else "False"
    if physical_hit_type is not None:
        row["physical_hit_type"] = physical_hit_type
    return row


#: exp17d/exp17i/exp27 共用的 R1D 行参数（无 --task-overrides，即默认 full 档）
_R1D_FULL_CFG: dict[str, Any] = {
    "--ball-speed": 9,
    "--no-plot": None,
    "--racket-config": "configs/rackets/blade100_v9_reference.yaml",
    "--robot-config": "configs/robots/r1d_right.yaml",
    "--runtime-config": "configs/robots/r1d_v12_runtime_right.yaml",
    "--serve-box": None,
}


class TestPhysicalVerdict:
    """R03: physical_of / physical_stats 的主动命中口径。"""

    def test_passive_contact_does_not_count_as_active(self) -> None:
        """physical_hit=True + type=passive 必须判 0（被动接触非主动命中）。"""
        row = _row({}, physical_hit=True, physical_hit_type="passive")
        assert physical_of(row) == 0

    def test_active_contact_counts(self) -> None:
        """type=active 恒判 1（即使布尔列缺失）。"""
        assert physical_of(_row({}, physical_hit_type="active")) == 1
        assert physical_of(_row({}, physical_hit=True,
                                physical_hit_type="active")) == 1

    def test_miss_type_counts_zero(self) -> None:
        """type=miss 判 0。"""
        assert physical_of(_row({}, physical_hit_type="miss")) == 0

    def test_missing_ptype_falls_back_to_bool(self) -> None:
        """ptype 缺失时回退布尔列（旧数据兼容）。"""
        assert physical_of(_row({}, physical_hit=True)) == 1
        assert physical_of(_row({}, physical_hit=False)) == 0

    def test_unknown_ptype_falls_back_to_bool(self) -> None:
        """未知 ptype 也回退布尔列，而不是当作命中。"""
        assert physical_of(_row({}, physical_hit=True,
                                physical_hit_type="weird")) == 1
        assert physical_of(_row({}, physical_hit=False,
                                physical_hit_type="weird")) == 0

    def test_no_verdict_returns_none(self) -> None:
        """ptype 与布尔列都缺失时返回 None（不臆造 0）。"""
        assert physical_of(_row({})) is None

    def test_stats_passive_row_is_not_active_hit(self) -> None:
        """单行 True+passive：n_active=0、n_passive=1、rate=0（R03 回归）。"""
        stats = physical_stats([_row({}, physical_hit=True,
                                     physical_hit_type="passive")])
        assert stats["n_active"] == 0
        assert stats["n_passive"] == 1
        assert stats["n_miss"] == 0
        assert stats["denominator"] == 1
        assert stats["unavailable_reason"] == "all_miss"
        # fix_plan R03 回归：0/1 是测得的 0%，rate 必须为 0.0；CI 不给零命中区间。
        assert stats["rate"] == 0.0
        assert stats["ci95"] is None

    def test_stats_without_any_verdict_keeps_rate_null(self) -> None:
        """无任何 physical 判定（旧 RM65 无该列）时 rate=null，不伪造 0.0。"""
        stats = physical_stats([_row({}, 100), _row({}, 101)])
        assert stats["denominator"] == 0
        assert stats["rate"] is None
        assert stats["ci95"] is None
        assert stats["unavailable_reason"] == "no_physical_verdict"

    def test_stats_mixed_verdicts(self) -> None:
        """active + passive + miss：命中率只按 active 计。"""
        rows = [
            _row({}, 100, physical_hit_type="active"),
            _row({}, 101, physical_hit=True, physical_hit_type="passive"),
            _row({}, 102, physical_hit_type="miss"),
        ]
        stats = physical_stats(rows)
        assert stats["n_active"] == 1
        assert stats["n_passive"] == 1
        assert stats["n_miss"] == 1
        assert stats["denominator"] == 3
        assert stats["rate"] == 100.0 / 3
        assert stats["hit_semantics"] == "physical"

    def test_hit_semantics_documents_active_rule(self) -> None:
        """HIT_SEMANTICS 文案与实现一致（physical_hit_type=active 口径）。"""
        rule = HIT_SEMANTICS["physical"]["rule"]
        assert rule.startswith("v12_episode_v2")
        assert "physical_hit_type=active" in rule


class TestTierOf:
    """R04: R1D 默认 full 档识别。"""

    def test_exp17d_full_without_overrides(self) -> None:
        """exp17d_mechanism/full：无 overrides 的 R1D 行 → full。"""
        assert tier_of(_row(_R1D_FULL_CFG)) == "full"

    def test_exp17i_tcp18_full_without_overrides(self) -> None:
        """exp17i tcp18_full_speed7：无 overrides 的 R1D 行 → full。"""
        cfg = {**_R1D_FULL_CFG, "--ball-speed": 7}
        assert tier_of(_row(cfg)) == "full"

    def test_exp27_speed9_full_without_overrides(self) -> None:
        """exp27 speed9_full：只带 robot/runtime 的 R1D 行 → full。"""
        cfg = {"--ball-speed": 9,
               "--robot-config": "configs/robots/r1d_right.yaml",
               "--runtime-config": "configs/robots/r1d_v12_runtime_right.yaml",
               "--serve-box": None, "--no-plot": None}
        assert tier_of(_row(cfg)) == "full"

    def test_explicit_sim_task_config_is_r1d(self) -> None:
        """仅显式 --sim-task-config 的 R1D 行也识别为 full。"""
        cfg = {"--sim-task-config": "configs/r1d_sim_right.yaml",
               "--serve-box": None}
        assert tier_of(_row(cfg)) == "full"

    def test_rm65_without_overrides_stays_empty(self) -> None:
        """RM65 行无 overrides 仍返回空串（不臆造档位）。"""
        assert tier_of(_row({"--ball-speed": 7, "--serve-box": None})) == ""

    def test_explicit_ablation_wins(self) -> None:
        """显式 --ablation 优先于 R1D 默认 full 推导。"""
        cfg = {**_R1D_FULL_CFG, "--ablation": "none"}
        assert tier_of(_row(cfg)) == "none"

    def test_r1d_override_profiles_still_derived(self) -> None:
        """带 overrides 时仍按 tube 开关推导四档（不被默认 full 覆盖）。"""
        none_cfg = {**_R1D_FULL_CFG, "--task-overrides": [
            "cost.tube.Q_n_tube=0.0",
            "cost.tube.Q_p_tube=0.0",
            "cost.tube.Q_v_tube=0.0",
            "cost.tube.use_softmin_terminal=false",
        ]}
        assert tier_of(_row(none_cfg)) == "none"
        softmin_only = {**_R1D_FULL_CFG, "--task-overrides": [
            "cost.tube.Q_n_tube=0.0",
            "cost.tube.Q_p_tube=0.0",
            "cost.tube.Q_v_tube=0.0",
        ]}
        assert tier_of(_row(softmin_only)) == "softmin_only"


class TestTcpLabel:
    """R05: TCP 标签固定 1 位小数。"""

    def test_fixed_one_decimal(self) -> None:
        """__tcp_label 对 1.0/1.4/1.8 都保留一位小数。"""
        assert _tcp_label(1.0) == "TCP1.0"
        assert _tcp_label(1.4) == "TCP1.4"
        assert _tcp_label(1.8) == "TCP1.8"

    def test_limits_config_means_tcp10(self) -> None:
        """--limits-config（真机限位）→ TCP1.0（旧实现给 TCP1）。"""
        row = _row({"--limits-config": "configs/real_robot.yaml"})
        assert tcp_label_of(row) == "TCP1.0"

    def test_max_tcp_values(self) -> None:
        """--max-tcp 的 1.0/1.4/1.8 均保留一位小数。"""
        assert tcp_label_of(_row({"--max-tcp": 1.0})) == "TCP1.0"
        assert tcp_label_of(_row({"--max-tcp": "1.0"})) == "TCP1.0"
        assert tcp_label_of(_row({"--max-tcp": 1.4})) == "TCP1.4"
        assert tcp_label_of(_row({"--max-tcp": 1.8})) == "TCP1.8"

    def test_runtime_config_filename_suffix(self) -> None:
        """R1D runtime 文件名 tcp10/tcp14/tcp18 换算为 TCP1.0/1.4/1.8。"""
        mk = lambda suffix: _row(  # noqa: E731 - 测试内联构造
            {"--runtime-config": f"configs/robots/r1d_runtime_{suffix}.yaml"})
        assert tcp_label_of(mk("tcp10")) == "TCP1.0"
        assert tcp_label_of(mk("tcp14")) == "TCP1.4"
        assert tcp_label_of(mk("tcp18")) == "TCP1.8"

    def test_default_is_tcp18(self) -> None:
        """无任何限速信息时默认 TCP1.8。"""
        assert tcp_label_of(_row({"--serve-box": None})) == "TCP1.8"


def _tier_cfg(tier: str) -> dict[str, Any]:
    """按消融档构造 R1D 行参数（none 与 full 两档）。"""
    cfg = dict(_R1D_FULL_CFG)
    if tier == "none":
        cfg["--task-overrides"] = [
            "cost.tube.Q_n_tube=0.0",
            "cost.tube.Q_p_tube=0.0",
            "cost.tube.Q_v_tube=0.0",
            "cost.tube.use_softmin_terminal=false",
        ]
    return cfg


def _pair_rows(a_hits: list[int], b_hits: list[int], *,
               seeds: list[int] | None = None) -> list[dict[str, str]]:
    """构造 tier_a=none / tier_b=full 的同 seed 配对行。"""
    seed_list = seeds if seeds is not None else list(range(100, 100 + len(a_hits)))
    rows: list[dict[str, str]] = []
    for seed, hit in zip(seed_list, a_hits):
        rows.append(_row(_tier_cfg("none"), seed,
                         physical_hit=bool(hit), physical_hit_type="active" if hit else "miss"))
    for seed, hit in zip(seed_list, b_hits):
        rows.append(_row(_tier_cfg("full"), seed,
                         physical_hit=bool(hit), physical_hit_type="active" if hit else "miss"))
    return rows


class TestPairedAllMiss:
    """R06: 全 miss 配对不输出零宽 CI / p=1。"""

    def test_both_tiers_all_miss(self) -> None:
        """两档共同 seed 全 0：保留构成字段，CI/p 不可估计。"""
        result = paired_test_physical(_pair_rows([0, 0], [0, 0]), "none", "full")
        assert result["n_paired"] == 2
        assert result["rate_a"] == 0.0
        assert result["rate_b"] == 0.0
        assert result["diff_pp"] == 0.0
        assert result["discordant"] == {"a_hit_b_miss": 0, "a_miss_b_hit": 0}
        assert result["hit_semantics"] == "physical"
        assert result["unavailable_reason"] == "all_miss"
        assert result["ci95"] is None
        assert result["p_mcnemar"] is None

    def test_one_tier_all_miss_keeps_estimate(self) -> None:
        """仅一档全 miss 时仍可估计（CI/p 为数值，不标 all_miss）。"""
        result = paired_test_physical(_pair_rows([0, 0], [0, 1]), "none", "full")
        assert result["n_paired"] == 2
        assert result["rate_a"] == 0.0
        assert result["rate_b"] == 50.0
        assert result["diff_pp"] == -50.0
        assert "unavailable_reason" not in result
        assert isinstance(result["ci95"], list) and len(result["ci95"]) == 2
        assert isinstance(result["p_mcnemar"], float)

    def test_normal_mixed_pair(self) -> None:
        """正常混合：速率、差值、不一致计数与 CI/p 均为数值。"""
        result = paired_test_physical(_pair_rows([1, 0], [0, 0]), "none", "full")
        assert result["n_paired"] == 2
        assert result["rate_a"] == 50.0
        assert result["rate_b"] == 0.0
        assert result["diff_pp"] == 50.0
        assert result["discordant"] == {"a_hit_b_miss": 1, "a_miss_b_hit": 0}
        assert isinstance(result["ci95"], list)
        assert isinstance(result["p_mcnemar"], float)
        assert "unavailable_reason" not in result

    def test_no_common_seed(self) -> None:
        """只有一档数据时仍明确返回 no_common_seed。"""
        rows = [_row(_tier_cfg("none"), 100,
                     physical_hit_type="active")]
        result = paired_test_physical(rows, "none", "full")
        assert result["n_paired"] == 0
        assert result["unavailable_reason"] == "no_common_seed"

    def test_no_physical_verdict(self) -> None:
        """两档都无 physical 判定时返回 no_physical_verdict。"""
        rows = [_row(_tier_cfg("none"), 100), _row(_tier_cfg("full"), 100)]
        result = paired_test_physical(rows, "none", "full")
        assert result["n_paired"] == 0
        assert result["unavailable_reason"] == "no_physical_verdict"

    def test_error_rows_are_excluded(self) -> None:
        """error 行不参与配对（但其余 seed 仍正常统计）。"""
        rows = _pair_rows([0, 0], [0, 0]) + [
            _row(_tier_cfg("none"), 999, physical_hit_type="active",
                 error="no_result")]
        result = paired_test_physical(rows, "none", "full")
        assert result["n_paired"] == 2
        assert result["rate_a"] == 0.0




# ---------------------------------------------------------------------------
# M02：前馈关闭变体、E3 变体分节、重复 seed 冲突（行序无关）
# ---------------------------------------------------------------------------


def _feedforward_cfg(value: str = "false") -> dict[str, Any]:
    """前馈关闭档：``--task-overrides`` 声明 ``actuator.feedforward=<value>``。"""
    return {**_R1D_FULL_CFG, "--task-overrides": [f"actuator.feedforward={value}"]}


def _hit_of(row: dict[str, str]) -> int:
    """legacy 命中口径：``hit_type=active`` 记 1（error 行由调用方过滤）。"""
    return 1 if (row.get("hit_type") or "").strip() == "active" else 0


def _pair_row(tier: str, seed: int, hit: bool) -> dict[str, str]:
    """构造同时具备 legacy 与 physical 判定的配对行。"""
    row = _row(
        _tier_cfg(tier),
        seed,
        physical_hit=hit,
        physical_hit_type="active" if hit else "miss",
    )
    row["hit"] = "True" if hit else "False"
    row["hit_type"] = "active" if hit else "miss"
    return row


def _e3_row(tier: str, speed: int, seed: int, hit: bool) -> dict[str, str]:
    """构造一行 ``exp17d_mechanism`` 的 position 记录（四档 + 前馈关闭变体）。"""
    cfg: dict[str, Any] = {
        **_R1D_FULL_CFG,
        "--ball-speed": speed,
        "--position-mode": None,
    }
    if tier == "none":
        cfg["--task-overrides"] = list(_tier_cfg("none")["--task-overrides"])
    elif tier == "full_nofeedforward":
        cfg["--task-overrides"] = ["actuator.feedforward=false"]
    elif tier != "full":
        raise AssertionError(f"未支持的测试档位：{tier}")
    physical_type = "active" if hit else "miss"
    return {
        "experiment": "exp17d_mechanism",
        "config_id": f"{tier}__position__s{speed}",
        "config_json": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
        "seed": str(seed),
        "hit": "True" if hit else "False",
        "hit_type": physical_type,
        "mode": "position",
        "robot_id": "unitree_r1d",
        "racket_id": "blade100_v9_reference",
        "error": "",
        "physical_hit": "True" if hit else "False",
        "physical_hit_type": physical_type,
    }


def _write_e3_dataset(root: Path, rows: list[dict[str, str]]) -> Path:
    """把伪行写成 ``<root>/exp17d_mechanism/results.csv`` 并返回 data root。"""
    exp_dir = root / "exp17d_mechanism"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with (exp_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


#: 每档两 seed 的命中真值：full 全中、变体 1/2、none 全 miss。
#: 旧实现把变体行后写进 full 档，正序得 0 pp、反序得 100 pp（复审 M02 证据）。
_E3_HITS: dict[str, tuple[bool, bool]] = {
    "full": (True, True),
    "full_nofeedforward": (True, False),
    "none": (False, False),
}


def _e3_rows(order: str = "forward") -> list[dict[str, str]]:
    """三档 × 两 seed 的 E3 数据；``reversed`` 用于验证行序无关。"""
    rows: list[dict[str, str]] = []
    for tier in ("full", "full_nofeedforward", "none"):
        for seed, hit in ((100, _E3_HITS[tier][0]), (101, _E3_HITS[tier][1])):
            rows.append(_e3_row(tier, 9, seed, hit))
    if order == "reversed":
        rows.reverse()
    return rows


def _e3_ctx(data_root: Path) -> StatsContext:
    """构造只声明 ``exp17d_mechanism`` 的统计上下文。"""
    ctx = make_context(data_root, mode="position")
    return replace(ctx, source_sets={"nominal_mechanism": "exp17d_mechanism"})


class TestTierOfFeedforward:
    """M02: 前馈关闭必须单独成档（同 seed 不再与 full 混算）。"""

    def test_same_seed_distinct_tiers(self) -> None:
        """同 seed 的 full 与前馈关闭行落在不同档位。"""
        assert tier_of(_row(_R1D_FULL_CFG, 100)) == "full"
        assert tier_of(_row(_feedforward_cfg(), 100)) == "full_nofeedforward"

    def test_task_overrides_zero_and_false_string(self) -> None:
        """``0`` / ``"False"`` 等写法同样识别为关闭。"""
        assert tier_of(_row(_feedforward_cfg("0"))) == "full_nofeedforward"
        assert tier_of(_row(_feedforward_cfg("False"))) == "full_nofeedforward"

    def test_task_overrides_true_stays_full(self) -> None:
        """显式开启前馈仍为普通 full 档。"""
        assert tier_of(_row(_feedforward_cfg("true"))) == "full"

    def test_cli_no_feedforward_marks_variant(self) -> None:
        """命令行 ``--no-feedforward`` 同样识别为关闭。"""
        cfg = {**_R1D_FULL_CFG, "--no-feedforward": None}
        assert tier_of(_row(cfg)) == "full_nofeedforward"

    def test_explicit_full_ablation_with_nofeedforward(self) -> None:
        """显式 ``--ablation full`` + 关闭前馈仍组合成变体档。"""
        cfg = {**_R1D_FULL_CFG, "--ablation": "full",
               "--task-overrides": ["actuator.feedforward=false"]}
        assert tier_of(_row(cfg)) == "full_nofeedforward"


class TestE3FeedforwardVariant:
    """M02: 变体独立成节，nominal 四档与配对不被污染。"""

    def test_nominal_cells_only_four_modes(self, tmp_path: Path) -> None:
        """nominal cells 的键集合恰为 3 速度 × 4 档，不含变体。"""
        legacy, physical = e3_nominal(
            _hit_of, _e3_ctx(_write_e3_dataset(tmp_path, _e3_rows())))
        nominal = {f"{sp}|{tier}" for sp in (7, 9, 12) for tier in MODES}
        assert set(legacy["cells"]) == nominal
        assert set(physical["cells"]) == nominal
        assert "9|full_nofeedforward" not in legacy["cells"]

    def test_full_cell_excludes_variant_rows(self, tmp_path: Path) -> None:
        """同 seed 的变体行不再计入 full（旧实现会串成 4 行 / 2 命中）。"""
        legacy, physical = e3_nominal(
            _hit_of, _e3_ctx(_write_e3_dataset(tmp_path, _e3_rows())))
        cell = legacy["cells"]["9|full"]
        assert cell["rows"] == 2
        assert cell["n_valid"] == 2
        assert cell["hits"] == 2
        assert physical["cells"]["9|full"]["denominator"] == 2

    def test_variants_section_reports_cells_and_pairs(self, tmp_path: Path) -> None:
        """变体节给出自己的 cells 与 none_vs_变体 配对。"""
        legacy, physical = e3_nominal(
            _hit_of, _e3_ctx(_write_e3_dataset(tmp_path, _e3_rows())))
        variants = legacy["variants"]
        assert variants["tiers"] == ["full_nofeedforward"]
        assert variants["cells"]["9|full_nofeedforward"]["n_valid"] == 2
        assert variants["cells"]["9|full_nofeedforward"]["hits"] == 1
        pair = variants["paired"]["9|none_vs_full_nofeedforward"]
        assert pair["n_paired"] == 2
        assert pair["diff_pp"] == -50.0
        assert physical["variants"]["tiers"] == ["full_nofeedforward"]
        assert physical["variants"]["cells"]["9|full_nofeedforward"]["denominator"] == 2

    def test_paired_result_independent_of_row_order(self, tmp_path: Path) -> None:
        """正序 / 反序 CSV 的 E3 配对结果逐字段一致（M02 验收）。"""
        fwd_legacy, fwd_physical = e3_nominal(
            _hit_of, _e3_ctx(_write_e3_dataset(tmp_path / "fwd", _e3_rows("forward"))))
        rev_legacy, rev_physical = e3_nominal(
            _hit_of, _e3_ctx(_write_e3_dataset(tmp_path / "rev", _e3_rows("reversed"))))
        assert fwd_legacy["paired"] == rev_legacy["paired"]
        assert fwd_physical["paired"] == rev_physical["paired"]
        for section in (fwd_legacy, rev_legacy):
            pair = section["paired"]["9|full_vs_none"]
            assert pair["n_paired"] == 2
            assert pair["diff_pp"] == 100.0
            assert "excluded_duplicate_keys" not in pair
        fwd_cells = {k: v for k, v in fwd_legacy["variants"]["cells"].items()
                     if k.startswith("9|")}
        rev_cells = {k: v for k, v in rev_legacy["variants"]["cells"].items()
                     if k.startswith("9|")}
        assert fwd_cells == rev_cells


class TestDuplicateSeedConflict:
    """M02: 同 seed 重复且取值冲突 → 整键排除并显式报告。"""

    def test_conflicting_seed_is_excluded(self) -> None:
        """full 档同 seed 一命中一未命中：该 seed 整键排除。"""
        rows = [
            _pair_row("full", 100, True),
            _pair_row("full", 100, False),
            _pair_row("none", 100, True),
        ]
        result = paired_test(rows, _hit_of, "full", "none")
        assert result["n_paired"] == 0
        assert result["unavailable_reason"] == "duplicate_seed_conflict"
        assert result["excluded_duplicate_keys"] == ["100"]
        assert paired_test(list(reversed(rows)), _hit_of, "full", "none") == result

    def test_clean_seed_still_pairs_with_conflict_reported(self) -> None:
        """冲突键被排除，但其余共同 seed 仍正常配对。"""
        rows = [
            _pair_row("full", 100, True),
            _pair_row("full", 100, False),
            _pair_row("none", 100, True),
            _pair_row("full", 101, True),
            _pair_row("none", 101, False),
        ]
        result = paired_test(rows, _hit_of, "full", "none")
        assert result["n_paired"] == 1
        assert result["diff_pp"] == 100.0
        assert result["excluded_duplicate_keys"] == ["100"]
        assert paired_test(list(reversed(rows)), _hit_of, "full", "none") == result

    def test_same_value_duplicates_are_deduped(self) -> None:
        """同 seed 重复但取值相同：去重后正常配对，不报冲突。"""
        rows = [
            _pair_row("full", 100, True),
            _pair_row("full", 100, True),
            _pair_row("none", 100, False),
        ]
        result = paired_test(rows, _hit_of, "full", "none")
        assert result["n_paired"] == 1
        assert result["diff_pp"] == 100.0
        assert "excluded_duplicate_keys" not in result

    def test_physical_conflict_is_reported(self) -> None:
        """physical 口径的重复冲突同样整键排除并报告。"""
        rows = [
            _row(_tier_cfg("full"), 100, physical_hit=True, physical_hit_type="active"),
            _row(_tier_cfg("full"), 100, physical_hit=False, physical_hit_type="miss"),
            _row(_tier_cfg("none"), 100, physical_hit_type="active"),
        ]
        result = paired_test_physical(rows, "full", "none")
        assert result["n_paired"] == 0
        assert result["unavailable_reason"] == "duplicate_seed_conflict"
        assert result["excluded_duplicate_keys"] == ["100"]


# ---------------------------------------------------------------------------
# 第二轮复审 R05：富化冲突覆盖 producer 原样复制的共有结果列
# ---------------------------------------------------------------------------

#: 两份视图共有的身份/结果列（构造伪行时一次性给全，便于只改一处）
_SHARED_VIEW_FIELDS = (
    "experiment", "config_id", "config_json", "seed", "mode", "robot_id",
    "racket_id", "config_hash", "hit", "hit_type", "physical_hit",
    "physical_hit_type", "physical_hit_step", "contact_event_count",
    "failure_reason", "backend", "status", "error", "artifact_dir",
)


def _shared_view_row(seed: int, *, hit: str = "False", hit_type: str = "miss",
                     physical_hit: str = "", physical_hit_type: str = "") -> dict[str, str]:
    """构造一行 runs/results 视图记录（身份与设计指纹固定，只改结果列）。"""
    cfg = {
        "--ball-speed": 9,
        "--robot-config": "configs/robots/r1d_right.yaml",
        "--serve-box": None,
    }
    return {
        "experiment": "exp17i_limits_ablation",
        "config_id": "full__torque__s9__tcp1.8",
        "config_json": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
        "seed": str(seed),
        "mode": "torque",
        "robot_id": "unitree_r1d",
        "racket_id": "blade100_v9_reference",
        "config_hash": "0123456789abcdef",
        "hit": hit,
        "hit_type": hit_type,
        "physical_hit": physical_hit,
        "physical_hit_type": physical_hit_type,
        "physical_hit_step": "",
        "contact_event_count": "",
        "failure_reason": "",
        "backend": "",
        "status": "",
        "error": "",
        "artifact_dir": f"artifacts/exp17i_limits_ablation/caseA/seed{seed}",
    }


class TestSharedResultColumns:
    """复审 R05: 同一运行的两份视图必须对同名共有列说法一致。"""

    def test_legacy_hit_conflict_is_detected(self) -> None:
        """只改 hit/hit_type（配置与哈希不变）也必须判冲突。"""
        results_view = _shared_view_row(100)
        runs_view = _shared_view_row(100, hit="True", hit_type="active")
        assert _enrichment_conflicts(results_view, runs_view) == ["hit", "hit_type"]

    def test_physical_column_conflict_is_detected(self) -> None:
        """physical_hit_type 同名不一致同样冲突，字段名就是列名。"""
        left = _shared_view_row(100, physical_hit="True", physical_hit_type="active")
        right = _shared_view_row(100, physical_hit="True", physical_hit_type="passive")
        assert _enrichment_conflicts(left, right) == ["physical_hit_type"]

    def test_completion_and_error_columns_are_compared(self) -> None:
        """failure_reason / backend / status / error 纳入同一套比较。"""
        left = _shared_view_row(100)
        left.update(failure_reason="timeout", backend="python",
                    status="ok", error="timeout")
        right = _shared_view_row(100)
        right.update(failure_reason="safety_filter_stop", backend="cpp",
                     status="error", error="no_result")
        assert set(_enrichment_conflicts(left, right)) == {
            "failure_reason", "backend", "status", "error",
        }

    def test_empty_values_are_not_conflicts(self) -> None:
        """空值视为未声明：单侧有值不算冲突（空值不回填、不误报）。"""
        left = _shared_view_row(100)
        right = _shared_view_row(100, physical_hit="True",
                                 physical_hit_type="active")
        assert _enrichment_conflicts(left, right) == []
        assert _enrichment_conflicts(right, left) == []

    def test_signature_and_field_names_cover_shared_columns(self) -> None:
        """签名随共有列变化，字段名与 ``_SHARED_RESULT_COLUMNS`` 对齐。"""
        assert _enrichment_signature(_shared_view_row(100)) != _enrichment_signature(
            _shared_view_row(100, hit="True", hit_type="active")
        )
        names = _enrichment_field_names()
        for column in _SHARED_RESULT_COLUMNS:
            assert names.count(column) == 1

    def test_shared_result_columns_constant_contains_required_columns(self) -> None:
        """复审要求的最小列集合必须全部在常量里。"""
        required = {
            "hit", "hit_type", "physical_hit", "physical_hit_type",
            "physical_hit_step", "contact_event_count", "failure_reason",
            "backend", "status", "error",
        }
        assert required <= set(_SHARED_RESULT_COLUMNS)


class TestSharedResultColumnsNoBackfill:
    """复审 R05: 共有列冲突时 results 视图不得被富化，且保留原值。"""

    def test_conflicting_views_block_enrichment(self, tmp_path: Path) -> None:
        """runs 的 hit/hit_type 与 results 矛盾时不回填 physical 列。"""
        results_rows = [_shared_view_row(100), _shared_view_row(101)]
        runs_rows = [
            _shared_view_row(100, hit="True", hit_type="active",
                             physical_hit="True", physical_hit_type="active"),
            _shared_view_row(101, hit="True", hit_type="active",
                             physical_hit="True", physical_hit_type="active"),
        ]
        data_root = _write_two_view_dataset(tmp_path / "batch", results_rows, runs_rows)
        rows = _load_rows(make_context(data_root, mode="torque"), "exp17i_limits_ablation")
        assert [row["_physical_status"] for row in rows] == ["conflict", "conflict"]
        assert rows[0]["_physical_conflicts"] == "hit,hit_type"
        # 不回填：physical 列保持空，legacy 结论仍是 results 的 miss
        assert not (rows[0].get("physical_hit") or "").strip()
        assert rows[0]["hit"] == "False" and rows[0]["hit_type"] == "miss"


def _write_two_view_dataset(root: Path, results_rows: list[dict[str, str]],
                            runs_rows: list[dict[str, str]]) -> Path:
    """写 results.csv（数据根下实验目录）与 runs.csv（数据根），返回数据根。"""
    fields: list[str] = []
    for row in [*results_rows, *runs_rows]:
        for key in row:
            if key not in fields:
                fields.append(key)
    exp_dir = root / "exp17i_limits_ablation"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for path, rows in ((exp_dir / "results.csv", results_rows),
                       (root / "runs.csv", runs_rows)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return root


# ---------------------------------------------------------------------------
# 第二轮复审 R06：DiD 四个条件各自唯一索引，冲突 seed 整键排除且行序无关
# ---------------------------------------------------------------------------


def _did_row(tier: str, tcp: str, seed: int, hit: bool) -> dict[str, str]:
    """构造一行 DiD 输入（档位 + TCP 层 + seed + legacy/physical 判定）。"""
    cfg = dict(_tier_cfg(tier))
    if tcp == "TCP1.0":
        cfg["--limits-config"] = "configs/real_robot.yaml"
    row = _pair_row(tier, seed, hit)
    row["config_json"] = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
    return row


def _did_repeat_rows(*, reverse: bool = False) -> list[dict[str, str]]:
    """复现 e8 复运行场景：TCP1.0/full 每 seed 同时有一命中一未命中。"""
    rows = [
        _did_row("full", "TCP1.8", 100, True),
        _did_row("full", "TCP1.8", 101, True),
        _did_row("none", "TCP1.8", 100, False),
        _did_row("none", "TCP1.8", 101, False),
        _did_row("full", "TCP1.0", 100, True),
        _did_row("full", "TCP1.0", 100, False),
        _did_row("full", "TCP1.0", 101, True),
        _did_row("full", "TCP1.0", 101, False),
        _did_row("none", "TCP1.0", 100, False),
        _did_row("none", "TCP1.0", 101, False),
    ]
    if reverse:
        rows.reverse()
    return rows


class TestDidDuplicateSeedConflict:
    """复审 R06: DiD 不得用后写覆盖掩盖同 seed 冲突。"""

    def test_legacy_did_reports_duplicate_conflict(self) -> None:
        """四个条件交叉后无可用 seed：n_pairs=0 且不给显著性结果。"""
        entry = _did(_did_repeat_rows(), _hit_of)["full"]
        assert entry["n_pairs"] == 0
        assert entry["unavailable_reason"] == "duplicate_seed_conflict"
        assert entry["excluded_duplicate_keys"] == [100, 101]
        assert entry["mean_did_pp"] is None
        assert entry["ci95"] is None and entry["p_sign"] is None

    def test_physical_did_reports_same_conflict(self) -> None:
        """physical 口径同样整键排除，并保留 hit_semantics。"""
        entry = _did(_did_repeat_rows(), physical_of, semantics="physical")["full"]
        assert entry["n_pairs"] == 0
        assert entry["unavailable_reason"] == "duplicate_seed_conflict"
        assert entry["excluded_duplicate_keys"] == [100, 101]
        assert entry["hit_semantics"] == "physical"

    def test_row_order_reversal_is_identical(self) -> None:
        """正序 / 反序行输入给出逐字段一致的结果（复审 R06 反例）。"""
        forward = _did(_did_repeat_rows(), _hit_of)
        reverse = _did(_did_repeat_rows(reverse=True), _hit_of)
        assert forward == reverse

    def test_clean_seed_still_estimated_with_conflict_reported(self) -> None:
        """只冲突 seed 100 时，seed 101 仍正常估计，并报告被排除的键。"""
        rows = [
            _did_row("full", "TCP1.8", 100, True),
            _did_row("full", "TCP1.8", 101, True),
            _did_row("none", "TCP1.8", 100, False),
            _did_row("none", "TCP1.8", 101, False),
            # 只有 seed 100 在 TCP1.0/full 出现一命中一未命中 → 整键排除
            _did_row("full", "TCP1.0", 100, True),
            _did_row("full", "TCP1.0", 100, False),
            _did_row("full", "TCP1.0", 101, True),
            _did_row("none", "TCP1.0", 100, False),
            _did_row("none", "TCP1.0", 101, False),
        ]
        entry = _did(rows, _hit_of)["full"]
        assert entry["n_pairs"] == 1
        assert entry["excluded_duplicate_keys"] == [100]
        assert entry["mean_did_pp"] == 0.0
        assert _did(list(reversed(rows)), _hit_of) == _did(rows, _hit_of)


# ---------------------------------------------------------------------------
# 第二轮复审 R07：--no-feedforward 覆盖 --task-overrides
# ---------------------------------------------------------------------------


class TestFeedforwardPrecedence:
    """复审 R07: 统计分类必须与 V12 配置解析的优先级一致。"""

    def test_cli_no_feedforward_beats_task_override(self) -> None:
        """``--no-feedforward``（store_true 缺值）优先于 override=true。"""
        cfg = {
            **_R1D_FULL_CFG,
            "--ablation": "full",
            "--no-feedforward": None,
            "--task-overrides": ["actuator.feedforward=true"],
        }
        assert _feedforward_of_cfg(cfg) is False
        assert tier_of(_row(cfg)) == "full_nofeedforward"

    def test_cli_no_feedforward_true_beats_override(self) -> None:
        """显式 True 与缺值同义（argparse store_true）。"""
        cfg = {
            **_R1D_FULL_CFG,
            "--ablation": "full",
            "--no-feedforward": True,
            "--task-overrides": ["actuator.feedforward=false"],
        }
        assert _feedforward_of_cfg(cfg) is False

    def test_explicit_false_flag_is_not_disabled(self) -> None:
        """显式 false/0 视为未关闭：退回 override（此处 override=true）。"""
        cfg = {
            **_R1D_FULL_CFG,
            "--ablation": "full",
            "--no-feedforward": False,
            "--task-overrides": ["actuator.feedforward=true"],
        }
        assert _feedforward_of_cfg(cfg) is True
        assert tier_of(_row(cfg)) == "full"

    def test_override_alone_still_works(self) -> None:
        """没有 CLI 旗标时 override 仍是唯一依据（不回归 M02 行为）。"""
        assert _feedforward_of_cfg(_feedforward_cfg("false")) is False
        assert _feedforward_of_cfg(_feedforward_cfg("true")) is True
        assert _feedforward_of_cfg(_R1D_FULL_CFG) is None
