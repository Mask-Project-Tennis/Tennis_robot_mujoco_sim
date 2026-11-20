"""batch_runner 日志文件名安全化（R02）与 CSV 身份校验（R09）单元测试。

R02：日志文件名必须 Windows 合法且长度受控。真实 R1D 用例的 config_id
可达 300+ 字符（exp17d_mechanism/none）并含 Windows 绝对路径（冒号/反斜杠），
旧实现直接当文件名会抛 ``OSError: [Errno 22]`` / ``WinError 123``。

R09：``--no-resume`` 只应关闭"跳过已完成"，不能绕过已有 CSV 的 schema/身份
校验；身份不匹配一律 ValueError，且不得改动旧文件字节。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from scripts.exp import batch_runner
from scripts.exp.batch_runner import (
    CSV_FIELDS,
    LEGACY_CSV_FIELDS,
    LOG_FILENAME_MAX_CHARS,
    ExperimentSpec,
    RunIdentity,
    _record,
    _safe_log_stem,
    _write_run_log,
    make_config_id,
)

#: Windows 文件名禁用字符（控制字符由字符白名单排除，不会出现在生成名里）
_ILLEGAL_CHARS = set('<>:"/\\|?*')
#: Windows 保留设备名（大小写不敏感，含扩展名前的任意后缀）
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

#: 真实 exp17d_mechanism/none 参数（config_id 300+ 字符，含 --task-overrides 列表）
_EXP17D_NONE_PARAMS: dict[str, Any] = {
    "--ball-speed": "9",
    "--no-plot": None,
    "--output": ("experiment_data/r1d_v12_parity/paper_smoke/artifacts/"
                 "exp17d_mechanism/none/seed100"),
    "--racket-config": "configs/rackets/blade100_v9_reference.yaml",
    "--robot-config": "configs/robots/r1d_right.yaml",
    "--runtime-config": "configs/robots/r1d_v12_runtime_right.yaml",
    "--seed": 100,
    "--serve-box": None,
    "--task-overrides": [
        "cost.tube.Q_n_tube=0.0",
        "cost.tube.Q_p_tube=0.0",
        "cost.tube.Q_v_tube=0.0",
        "cost.tube.use_softmin_terminal=false",
    ],
}

#: 真实 exp16_limits_v2/real_tcp10_speed7 参数（runtime 为含中文的绝对 Windows 路径）
_EXP16_TCP10_PARAMS: dict[str, Any] = {
    "--ball-speed": "7",
    "--no-plot": None,
    "--robot-config": "configs/robots/r1d_right.yaml",
    "--runtime-config": (
        r"F:\刀客塔\研究\网球机器人\工程\Code\Sim\experiment_data"
        r"\r1d_v12_parity\paper_smoke\runtime_overrides"
        r"\exp16_limits_v2\real_tcp10_speed7.yaml"),
    "--seed": 100,
    "--serve-box": None,
}


def _assert_windows_safe(filename: str) -> None:
    """断言文件名为 Windows 合法且长度受控（只允许白名单 ASCII 字符）。"""
    assert filename.endswith(".log"), filename
    assert len(filename) <= LOG_FILENAME_MAX_CHARS, (len(filename), filename)
    assert filename == filename.strip(), repr(filename)
    assert not (set(filename) & _ILLEGAL_CHARS), filename
    assert re.fullmatch(r"[A-Za-z0-9._-]+\.log", filename), filename
    stem = filename[:-4]
    assert stem and not stem.endswith("."), filename
    for part in stem.split("."):
        assert part.upper() not in _RESERVED_NAMES, filename


class TestSafeLogStem:
    """R02: 日志名主干生成（slug + seed + digest）。"""

    def test_real_exp17d_none_config_id_is_long_but_stem_is_short(self) -> None:
        """真实 300+ 字符 config_id 必须被压到 ≤96 的合法文件名。"""
        config_id = make_config_id(_EXP17D_NONE_PARAMS)
        # 前置条件：确实是旧实现会炸的长名字（review 记 328，实测 315）
        assert len(config_id) > LOG_FILENAME_MAX_CHARS
        stem = _safe_log_stem(_EXP17D_NONE_PARAMS, 100)
        _assert_windows_safe(stem + ".log")
        # 三段式（slug 内部可能含 __，从右切）：slug__s<seed>__<10 位 digest>
        slug, seed_part, digest = stem.rsplit("__", 2)
        assert slug and len(slug) <= 48
        assert seed_part == "s100"
        assert re.fullmatch(r"[0-9a-f]{10}", digest), digest

    def test_real_exp16_windows_absolute_path_is_sanitized(self) -> None:
        """绝对 Windows runtime 路径（冒号/反斜杠/中文）不得进入文件名。"""
        config_id = make_config_id(_EXP16_TCP10_PARAMS)
        # 前置条件：config_id 里确实带冒号与反斜杠（旧实现非法文件名的来源）
        assert ":" in config_id and "\\" in config_id
        stem = _safe_log_stem(_EXP16_TCP10_PARAMS, 100)
        _assert_windows_safe(stem + ".log")
        assert ":" not in stem and "\\" not in stem

    def test_illegal_characters_in_values_are_sanitized(self) -> None:
        """值里出现 /:*?"<>| 时必须全部清洗掉。"""
        params = {"--tag": 'a/b:c*d?e"f<g>h|i\\j', "--seed": 7}
        stem = _safe_log_stem(params, 7)
        _assert_windows_safe(stem + ".log")

    def test_pure_chinese_and_empty_params(self) -> None:
        """纯中文 / 空参数退化为 ``case`` 前缀，仍必须合法。"""
        # 键与值都非 ASCII：slug 清洗后为空 → case
        chinese = _safe_log_stem({"中文键": "中文值"}, None)
        assert chinese.startswith("case__sNA__"), chinese
        _assert_windows_safe(chinese + ".log")
        # 空参数：make_config_id 给出 "default"，沿用为 slug
        empty = _safe_log_stem({}, None)
        assert empty.startswith("default__sNA__"), empty
        _assert_windows_safe(empty + ".log")

    @pytest.mark.parametrize(("seed", "expected_part"), [
        ("12/34*", "s12-34"),
        ("..", "sNA"),
        ("a b", "sa-b"),
        (100, "s100"),
    ])
    def test_illegal_seed_is_sanitized(self, seed: Any,
                                       expected_part: str) -> None:
        """非法 seed 字符同样清洗；只剩点时退化为 NA。"""
        stem = _safe_log_stem({"--serve-box": None, "--seed": seed}, seed)
        assert f"__{expected_part}__" in stem, stem
        _assert_windows_safe(stem + ".log")

    def test_distinct_configs_with_same_prefix_do_not_collide(self) -> None:
        """共享前 48 字符 slug 的不同配置靠 digest 区分（不碰撞）。"""
        prefix = "x" * 80
        a = _safe_log_stem({"--tag": prefix + "a"}, 100)
        b = _safe_log_stem({"--tag": prefix + "b"}, 100)
        assert a != b
        assert a.split("__")[0] == b.split("__")[0]  # slug 相同，digest 不同
        assert a.split("__")[-1] != b.split("__")[-1]

    def test_same_params_different_seed_do_not_collide(self) -> None:
        """同配置不同 seed 必须是不同文件。"""
        a = _safe_log_stem(_EXP17D_NONE_PARAMS, 100)
        b = _safe_log_stem(_EXP17D_NONE_PARAMS, 101)
        assert a != b

    def test_very_long_value_stays_within_length_limit(self) -> None:
        """超长单值也不得突破 96 字符上限。"""
        stem = _safe_log_stem({"--tag": "z" * 500}, 100)
        _assert_windows_safe(stem + ".log")
        assert len(stem.split("__")[0]) == 48


class TestWriteRunLog:
    """R02: 落盘文件名与正文可追溯性（完整 config_id / params / cmd）。"""

    def test_header_keeps_full_config_id_params_and_command(self,
                                                            tmp_path: Path) -> None:
        """日志正文首部必须保留完整可追溯信息（文件名已截断）。"""
        params = _EXP17D_NONE_PARAMS
        cmd = ["python", "scripts/rm65_mpc_v12.py", "--serve-box",
               "--seed", "100"]
        path = Path(_write_run_log(tmp_path, "exp17d_mechanism", params, cmd,
                                   "运行输出\n__RESULT__ hit_type=active\n"))
        assert path.is_file()
        _assert_windows_safe(path.name)
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# config_id: " + make_config_id(params) + "\n")
        assert "# params: " + json.dumps(params, sort_keys=True,
                                         ensure_ascii=False) + "\n" in text
        assert "# command: " + " ".join(cmd) + "\n" in text
        assert text.endswith("运行输出\n__RESULT__ hit_type=active\n")

    def test_distinct_seeds_write_two_files(self, tmp_path: Path) -> None:
        """同配置不同 seed 落盘为两个文件（不互相覆盖）。"""
        p100 = _EXP17D_NONE_PARAMS
        p101 = {**p100, "--seed": 101}
        path100 = Path(_write_run_log(tmp_path, "exp", p100, ["python"], "a"))
        path101 = Path(_write_run_log(tmp_path, "exp", p101, ["python"], "b"))
        assert path100 != path101
        assert path100.is_file() and path101.is_file()
        _assert_windows_safe(path100.name)
        _assert_windows_safe(path101.name)

    def test_log_dir_none_returns_empty(self) -> None:
        """未配置日志目录时不落盘（保持旧行为）。"""
        assert _write_run_log(None, "exp", {}, [], "x") == ""


def _install_fake_run_one(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """用假 run_one 替换 subprocess 执行，记录调用并返回合法 CSV 记录。"""
    calls: list[dict[str, Any]] = []

    def fake_run_one(experiment: str, script: str, params: dict[str, Any],
                     identity: RunIdentity = batch_runner.LEGACY_IDENTITY,
                     default_timeout_s: float = 60.0,
                     log_dir: Any = None) -> dict[str, Any]:
        calls.append({"params": params, "identity": identity,
                      "script": script, "log_dir": log_dir})
        return _record(
            experiment, params, identity,
            {"error": "", "hit": 1, "hit_type": "active", "wall_time": 0.0},
            timeout_s=default_timeout_s, exit_code=0, log_path="")

    monkeypatch.setattr(batch_runner, "run_one", fake_run_one)
    return calls


#: 旧格式 CSV（无身份列）：真实历史 RM65 数据实测 20 列，缺两个 ball_out 列
_LEGACY_HEADER_20 = [key for key in LEGACY_CSV_FIELDS
                     if key not in ("ball_out_speed", "ball_out_vy")]


def _write_legacy_csv(path: Path, *, config_id: str = "servebox",
                      seed: int = 100) -> None:
    """写一个旧格式 results.csv（20 列 + 一行已完成记录）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv

    row = {key: "" for key in _LEGACY_HEADER_20}
    row.update({"experiment": path.parent.name, "config_id": config_id,
                "seed": str(seed), "error": "", "hit": "1",
                "hit_type": "active"})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=_LEGACY_HEADER_20)
        writer.writeheader()
        writer.writerow(row)


def _write_new_csv(path: Path, *, identity: dict[str, str]) -> None:
    """写一个新格式 results.csv（36 列 + 指定身份的一行）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv

    row = {key: "" for key in CSV_FIELDS}
    row.update({"experiment": path.parent.name, "config_id": "cfg",
                "seed": "100", "error": "", "hit": "1",
                "hit_type": "active"})
    row.update(identity)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def _r1d_spec(name: str, *, seed: int = 100) -> ExperimentSpec:
    """构造一个 R1D/position 身份的实验定义（与旧 RM65 CSV 不同身份）。"""
    return ExperimentSpec(
        name=name,
        script="scripts/rm65_mpc_v12.py",
        grid=[{"--robot-config": "configs/robots/r1d_right.yaml",
               "--position-mode": None, "--serve-box": None,
               "--seed": seed}],
        identity=RunIdentity(
            robot_id="unitree_r1d", mode="", racket_id="blade100_v9_reference",
            config_hash="abc1234567",
            metric_schema="v12_episode_v2_physical_hit"),
    )


class TestCsvIdentityGuard:
    """R09: CSV 存在即校验，resume 只控制是否跳过已完成任务。"""

    def test_no_resume_still_rejects_legacy_csv_other_identity(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--no-resume 也不能把 R1D 行追加进旧 RM65 身份 CSV，且旧文件不变。"""
        name = "exp17d_mechanism"
        csv_path = tmp_path / name / "results.csv"
        _write_legacy_csv(csv_path)
        before = csv_path.read_bytes()
        calls = _install_fake_run_one(monkeypatch)

        with pytest.raises(ValueError, match="旧格式"):
            batch_runner.run_batch(_r1d_spec(name), workers=1, resume=False,
                                   data_root=tmp_path)
        assert csv_path.read_bytes() == before
        assert calls == []  # 校验必须先于任何 run

    def test_no_resume_rejects_new_csv_with_foreign_identity(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """新格式 CSV 混入其他身份的行：--no-resume 同样 ValueError。"""
        name = "exp17d_mechanism"
        csv_path = tmp_path / name / "results.csv"
        _write_new_csv(csv_path, identity={
            "robot_id": "rm65", "mode": "torque",
            "racket_id": "rm65_legacy", "config_hash": "legacy",
            "metric_schema": "v12_result_v1"})
        before = csv_path.read_bytes()
        calls = _install_fake_run_one(monkeypatch)

        with pytest.raises(ValueError, match="混入不同身份"):
            batch_runner.run_batch(_r1d_spec(name), workers=1, resume=False,
                                   data_root=tmp_path)
        assert csv_path.read_bytes() == before
        assert calls == []

    def test_same_identity_no_resume_appends(self, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
        """同身份 + --no-resume：正常重跑并追加一行，表头沿用旧 20 列。"""
        name = "exp_rm65_legacy"
        csv_path = tmp_path / name / "results.csv"
        _write_legacy_csv(csv_path)
        before_header = csv_path.read_text(
            encoding="utf-8").splitlines()[0]
        calls = _install_fake_run_one(monkeypatch)

        spec = ExperimentSpec(
            name=name, script="scripts/rm65_mpc_v12.py",
            grid=[{"--serve-box": None, "--seed": 100}])
        summary = batch_runner.run_batch(spec, workers=1, resume=False,
                                         data_root=tmp_path)
        assert summary["launched"] == 1
        assert len(calls) == 1
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == before_header  # 表头不被新 schema 覆盖
        assert len(lines) == 3  # 表头 + 旧行 + 新行

    def test_allow_legacy_resume_accepts_r1d_identity(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
            capsys: pytest.CaptureFixture[str]) -> None:
        """显式 --legacy-resume 时允许 R1D 身份延续旧 CSV（兼容路径仍可用）。"""
        name = "exp17d_mechanism"
        csv_path = tmp_path / name / "results.csv"
        _write_legacy_csv(csv_path)
        calls = _install_fake_run_one(monkeypatch)

        summary = batch_runner.run_batch(
            _r1d_spec(name), workers=1, resume=False, data_root=tmp_path,
            allow_legacy_resume=True)
        assert summary["launched"] == 1
        assert len(calls) == 1
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert "旧格式 CSV" in capsys.readouterr().out

    def test_empty_csv_is_rejected(self, tmp_path: Path) -> None:
        """已存在但无表头的 CSV 无法判定身份，必须拒绝而非盲目追加。"""
        csv_path = tmp_path / "exp" / "results.csv"
        csv_path.parent.mkdir(parents=True)
        csv_path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="表头为空"):
            batch_runner.run_batch(_r1d_spec("exp"), workers=1, resume=False,
                                   data_root=tmp_path)

    def test_missing_core_columns_are_rejected(self, tmp_path: Path) -> None:
        """缺核心列的 CSV 不是本 runner 产物，必须拒绝。"""
        csv_path = tmp_path / "exp" / "results.csv"
        csv_path.parent.mkdir(parents=True)
        csv_path.write_text("experiment,config_id\nx,y\n", encoding="utf-8")
        with pytest.raises(ValueError, match="缺少核心列"):
            batch_runner.run_batch(_r1d_spec("exp"), workers=1, resume=False,
                                   data_root=tmp_path)
