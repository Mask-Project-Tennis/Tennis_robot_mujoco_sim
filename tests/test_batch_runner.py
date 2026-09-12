"""batch_runner + overnight_experiments 单元测试。

不跑真实仿真（run_one 的 subprocess 路径由 smoke test 覆盖），
只测纯函数: config_id 生成 / 结果解析 / resume 读取 / 网格构造。
"""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.exp.batch_runner import (
    CSV_FIELDS,
    _record,
    load_done_keys,
    make_config_id,
    parse_result_line,
)
from scripts.exp.overnight_experiments import (
    EXPERIMENTS,
    _noise_grid,
    _obsfreq_grid,
    _perturb_grid,
)


class TestMakeConfigId:
    """config_id 生成: 稳定性与可读性（resume 正确性依赖稳定性）。"""

    def test_same_params_same_id(self) -> None:
        """参数字典键序不同但内容相同时 id 必须一致。"""
        a = {"--ball-speed": 9, "--seed": 5, "--no-plot": None}
        b = {"--no-plot": None, "--seed": 5, "--ball-speed": 9}
        assert make_config_id(a) == make_config_id(b)

    def test_seed_excluded(self) -> None:
        """不同 seed 的同配置必须有相同 id（resume 按 (id, seed) 去重）。"""
        assert make_config_id({"--ball-speed": 9, "--seed": 1}) == \
            make_config_id({"--ball-speed": 9, "--seed": 99})

    def test_distinct_configs_distinct_id(self) -> None:
        """不同配置必须产生不同 id。"""
        assert make_config_id({"--ablation": "full"}) != \
            make_config_id({"--ablation": "none"})

    def test_empty_params(self) -> None:
        """空参数（除 seed 外）给出 default。"""
        assert make_config_id({"--seed": 1}) == "default"

    def test_multivalue_sanitized(self) -> None:
        """多值参数（含空格）被压成 -，避免非法文件/字段字符。"""
        cid = make_config_id({"--kp": "500 500 500"})
        assert " " not in cid


class TestParseResultLine:
    """__RESULT__ 行解析。"""

    def test_normal_line(self) -> None:
        """数值字段转 float，字符串字段保持 str。"""
        line = ("一些输出\n__RESULT__: pos_error=0.0668 hit_type=active "
                "max_tcp=1.81 hit_time_error_ms=15.0\n")
        fields = parse_result_line(line)
        assert fields["pos_error"] == 0.0668
        assert fields["hit_type"] == "active"
        assert fields["max_tcp"] == 1.81

    def test_no_result(self) -> None:
        """无 __RESULT__ 行返回 no_result 错误。"""
        assert parse_result_line("崩溃了 Traceback...")["error"] == "no_result"


class TestLoadDoneKeys:
    """resume: 已完成 (config_id, seed) 集合读取。"""

    def test_missing_file(self, tmp_path: Path) -> None:
        """CSV 不存在时返回空集。"""
        assert load_done_keys(tmp_path / "nope.csv") == set()

    def test_reads_and_skips_errors(self, tmp_path: Path) -> None:
        """正常行计入、error 行不计入（error 行需要重跑）。"""
        csv_path = tmp_path / "results.csv"
        rows = [
            {"config_id": "cfgA", "seed": "1", "error": ""},
            {"config_id": "cfgA", "seed": "2", "error": "timeout"},
            {"config_id": "cfgB", "seed": "1", "error": ""},
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["config_id", "seed", "error"])
            w.writeheader()
            w.writerows(rows)
        assert load_done_keys(csv_path) == {("cfgA", 1), ("cfgB", 1)}


class TestRecord:
    """单行 CSV 记录补全。"""

    def test_full_fields(self) -> None:
        """记录包含全部 CSV 列（缺省补空），meta 键进 config_json。"""
        rec = _record("exp", {"--ball-speed": 9, "--seed": 3},
                      {"pos_error": 0.1, "hit_type": "active", "hit": True})
        assert set(rec.keys()) == set(CSV_FIELDS)
        assert rec["experiment"] == "exp"
        assert rec["seed"] == 3
        assert rec["hit"] is True
        assert '"--ball-speed": 9' in rec["config_json"]
        assert "--seed" not in rec["config_json"]


class TestGrids:
    """七个实验矩阵的结构性校验。"""

    def test_all_rows_have_seed(self) -> None:
        """所有网格行必须含 --seed（batch_runner 依赖它做 resume 键）。"""
        for name, spec in EXPERIMENTS.items():
            for p in spec.grid:
                assert "--seed" in p, f"{name} 缺 --seed: {p}"

    def test_expected_sizes(self) -> None:
        """网格规模与设计一致（改动矩阵时此测试强制更新预期）。"""
        expected = {
            "exp13_arch": 400,        # V12 力矩/位置 × 200（V11 引用报告）
            "exp15_speed_v2": 9 * 200,
            "exp16_limits_v2": 2 * 200,
            "exp14_pd_v2": 8 * 4 * 50,
            "exp17a_noise": (1 * 2 * 1 + 4 * 2 * 2) * 3 * 100,
            "exp17b_perturb": 5 * 4 * 2 * 100,
            "exp17c_obsfreq": (5 * 2 * 1 + 5 * 2 * 2) * 100,
            "exp17h_extreme": 2 * 4 * 1500 + 4 * 3000,  # Block A + Block B
            "exp17i_limits_ablation": 2 * 4 * 400,     # 限速层 × 4 档（配对设计）
        }
        for name, n in expected.items():
            assert len(EXPERIMENTS[name].grid) == n, \
                f"{name}: {len(EXPERIMENTS[name].grid)} != {n}"

    def test_noise_grid_zero_sigma_single_kf(self) -> None:
        """σp=0 时只跑 kf=off（KF 无噪声输入差异, exp9 已验证）。"""
        for p in _noise_grid():
            if p.get("--obs-noise-pos") is None:
                assert "--obs-use-kf" not in p, f"σp=0 不应有 kf: {p}"
            else:
                assert p["--obs-noise-vel"] == p["--obs-noise-pos"] * 10

    def test_perturb_grid_origin_cell_clean(self) -> None:
        """(0,0) 原点格不加任何扰动 flag（对照=纯标称）。"""
        for p in _perturb_grid():
            if "time-perturb" in str(p) or "space-perturb" in str(p):
                keys = {k for k in p if "perturb" in k}
                has_time = any("time" in k for k in keys)
                has_space = any("space" in k for k in keys)
                # 任一轴 max>0 才允许出现扰动参数
                assert has_time or has_space or "--random-perturb" not in p

    def test_obsfreq_grid_no_noise_single_kf(self) -> None:
        """无噪声行只跑 kf=off。"""
        for p in _obsfreq_grid():
            if p.get("--obs-noise-pos") is None:
                assert "--obs-use-kf" not in p

    def test_config_ids_unique_per_config(self) -> None:
        """同一网格内不同配置必须产生不同 config_id（collision=数据污染）。"""
        for name, spec in EXPERIMENTS.items():
            ids = {make_config_id(p) for p in spec.grid}
            n_configs = len({
                make_config_id({k: v for k, v in p.items() if k != "--seed"})
                for p in spec.grid
            })
            assert len(ids) == n_configs, f"{name} config_id 碰撞"
