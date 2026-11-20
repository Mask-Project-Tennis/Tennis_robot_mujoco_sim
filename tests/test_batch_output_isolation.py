"""batch_runner 产物隔离测试（并行 RM65 任务不得共享 ``results/``）。

只 monkeypatch ``subprocess.run``，不启动真实仿真、不触碰仓库 ``results/``。
覆盖两件事：

1. RM65 V12 入口在批量执行时被注入独立 ``--output``，且 dry-run 打印的命令、
   真实执行的命令与 CSV ``artifact_dir`` 三者一致（不同 seed/config 不共享目录，
   同一任务重复 prepare 稳定、resume 身份不变）；
2. 已显式给 ``--output`` 的任务（R1D 论文 smoke 等）与非 V12 脚本不会被二次注入。
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts.exp import batch_runner as br

RM65_SCRIPT = "scripts/rm65_mpc_v12.py"
# 任意非 V12 入口：产物路径由 manifest/catalog 决定，批量层不得介入。
OTHER_SCRIPT = "scripts/exp/run_r1d_paper_smoke.py"
RESULT_LINE = "__RESULT__: hit=1 hit_type=active pos_error=0.1 ball_out_speed=3.0\n"


def _rm65_spec(*seeds: int, extra: Mapping[str, Any] | None = None) -> br.ExperimentSpec:
    """构造最小 RM65 批量 spec（每个 seed 一个任务，无 --output）。"""
    base: dict[str, Any] = {"--serve-box": None, "--ball-speed": 7, "--no-plot": None}
    if extra:
        base.update(extra)
    grid = [{**base, "--seed": seed} for seed in seeds]
    return br.ExperimentSpec(name="exp_test", script=RM65_SCRIPT, grid=grid)


def _install_fake_subprocess(
    monkeypatch: pytest.MonkeyPatch, calls: list[list[str]]) -> None:
    """用假 subprocess.run 替换真实子进程，记录 argv 并返回合法结果行。"""

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(
            cmd, 0, stdout=RESULT_LINE, stderr="")

    monkeypatch.setattr(br.subprocess, "run", fake_run)


def _argv_value(cmd: list[str], flag: str) -> str:
    """取 argv 中某个带值参数的值（缺失或多个时直接失败）。"""
    assert cmd.count(flag) == 1, f"{flag} 出现 {cmd.count(flag)} 次: {cmd}"
    return cmd[cmd.index(flag) + 1]


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    """读 results.csv 为字典列表。"""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    """本实验输出目录（注入路径必须以它为根）。"""
    return tmp_path / "exp_test"


class TestRm65ArtifactInjection:
    """RM65 V12 任务自动获得独立产物目录。"""

    def test_seeds_get_distinct_dirs_under_out_dir(
        self, monkeypatch: pytest.MonkeyPatch, out_dir: Path
    ) -> None:
        """2 个 seed 的命令各有不同 --output，均位于 <out_dir>/artifacts/... 下。"""
        calls: list[list[str]] = []
        _install_fake_subprocess(monkeypatch, calls)
        spec = _rm65_spec(100, 101)

        br.run_batch(spec, workers=1, resume=False, data_root=out_dir.parent)

        assert len(calls) == 2
        outputs = [_argv_value(cmd, "--output") for cmd in calls]
        assert len(set(outputs)) == 2, f"seed 之间共享了产物目录: {outputs}"
        for value in outputs:
            path = Path(value)
            assert path.is_relative_to(out_dir / "artifacts")
            assert path.name.startswith("seed")
        # 同一配置的 2 个 seed 共用配置级目录，只有 seed 层级不同
        parents = {Path(value).parent for value in outputs}
        assert len(parents) == 1
        assert {Path(value).name for value in outputs} == {"seed100", "seed101"}

    def test_csv_artifact_dir_matches_command(
        self, monkeypatch: pytest.MonkeyPatch, out_dir: Path
    ) -> None:
        """CSV artifact_dir 与真实命令的 --output 逐 seed 一致。"""
        calls: list[list[str]] = []
        _install_fake_subprocess(monkeypatch, calls)
        spec = _rm65_spec(100, 101)

        br.run_batch(spec, workers=1, resume=False, data_root=out_dir.parent)

        by_seed = {_argv_value(cmd, "--seed"): _argv_value(cmd, "--output")
                   for cmd in calls}
        rows = _read_rows(out_dir / "results.csv")
        assert len(rows) == 2
        for row in rows:
            assert row["artifact_dir"] == by_seed[row["seed"]]
            assert Path(row["artifact_dir"]).is_relative_to(out_dir / "artifacts")

    def test_manifest_config_id_keeps_config_id(
        self, monkeypatch: pytest.MonkeyPatch, out_dir: Path
    ) -> None:
        """注入 --output 不改变 config_id（resume/配对身份稳定）。"""
        calls: list[list[str]] = []
        _install_fake_subprocess(monkeypatch, calls)
        params = {"--serve-box": None, "--ball-speed": 7, "--no-plot": None,
                  "--seed": 100}

        prepared = br._prepare_task_params(RM65_SCRIPT, params, out_dir)

        assert prepared["--output"].startswith(str(out_dir / "artifacts"))
        assert br._effective_config_id(prepared) == br.make_config_id(params)
        assert br._effective_config_id(prepared) == br._effective_config_id(params)
        # 原始网格字典不被改写（同一 spec 可在不同 out_dir 下重复准备）
        assert "--output" not in params

    def test_same_task_prepare_is_stable(
        self, monkeypatch: pytest.MonkeyPatch, out_dir: Path
    ) -> None:
        """同一任务重复 prepare / 重复 dry-run 输出相同路径。"""
        calls: list[list[str]] = []
        _install_fake_subprocess(monkeypatch, calls)
        spec = _rm65_spec(100)
        params = spec.grid[0]

        first = br._prepare_task_params(RM65_SCRIPT, params, out_dir)
        second = br._prepare_task_params(RM65_SCRIPT, params, out_dir)
        assert first["--output"] == second["--output"]
        assert br.rm65_artifact_dir(
            out_dir, br._effective_config_id(params), 100
        ) == Path(first["--output"])

        commands = []
        for _ in range(2):
            result = br.run_batch(spec, workers=1, resume=False, dry_run=True,
                                  data_root=out_dir.parent)
            commands.append(result["tasks"][0]["command"])
        assert commands[0] == commands[1]
        assert _argv_value(commands[0], "--output") == first["--output"]

    def test_dry_run_commands_match_real_commands(
        self, monkeypatch: pytest.MonkeyPatch, out_dir: Path
    ) -> None:
        """dry-run 打印的命令与真实执行的 argv 完全一致。"""
        calls: list[list[str]] = []
        _install_fake_subprocess(monkeypatch, calls)
        spec = _rm65_spec(100, 101)

        planned = br.run_batch(spec, workers=1, resume=False, dry_run=True,
                               data_root=out_dir.parent)
        br.run_batch(spec, workers=1, resume=False, data_root=out_dir.parent)

        assert [task["command"] for task in planned["tasks"]] == calls
        assert Path(planned["csv"]) == out_dir / "results.csv"

    def test_resume_skips_done_task_with_injected_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """resume 身份不含注入的 --output：第二次 run 不重跑已完成任务。"""
        calls: list[list[str]] = []
        _install_fake_subprocess(monkeypatch, calls)
        spec = _rm65_spec(100)

        first = br.run_batch(spec, workers=1, resume=False, data_root=tmp_path / "root")
        second = br.run_batch(spec, workers=1, data_root=tmp_path / "root")

        assert first["launched"] == 1
        assert second["done_before"] == 1
        assert second["launched"] == 0
        assert len(calls) == 1, "已完成任务被注入路径判成新任务重跑了"

    def test_different_configs_do_not_share_artifact_dir(
        self, monkeypatch: pytest.MonkeyPatch, out_dir: Path
    ) -> None:
        """不同配置（--ball-speed 不同）落在不同配置级目录。"""
        calls: list[list[str]] = []
        _install_fake_subprocess(monkeypatch, calls)

        fast = br.run_batch(_rm65_spec(100, extra={"--ball-speed": 7}),
                            workers=1, resume=False, dry_run=True,
                            data_root=out_dir.parent)
        slow = br.run_batch(_rm65_spec(100, extra={"--ball-speed": 9}),
                            workers=1, resume=False, dry_run=True,
                            data_root=out_dir.parent)

        fast_dir = Path(_argv_value(fast["tasks"][0]["command"], "--output")).parent
        slow_dir = Path(_argv_value(slow["tasks"][0]["command"], "--output")).parent
        assert fast_dir != slow_dir
        assert fast_dir.parent == slow_dir.parent == out_dir / "artifacts"
        assert not calls, "dry-run 不应执行子进程"

    def test_artifact_dir_name_is_windows_safe_for_long_config(
        self, out_dir: Path
    ) -> None:
        """226 字符 config_id（exp28 实测）压成 Windows 合法、长度受控目录名。"""
        long_id = "ablationsoftmin_only__alwayscollision__" + "x" * 190
        path = br.rm65_artifact_dir(out_dir, long_id, 100)

        assert len(path.parent.name) <= br.ARTIFACT_SLUG_MAX_CHARS + 12
        assert not set(path.parent.name) & set('<>:"|?*\\')
        assert path.parent.name != br.rm65_artifact_dir(out_dir, long_id + "y", 100).parent.name


class TestExplicitOutputUntouched:
    """已显式给 --output 的任务与其它脚本不被注入。"""

    def test_r1d_explicit_output_kept_verbatim(
        self, monkeypatch: pytest.MonkeyPatch, out_dir: Path
    ) -> None:
        """R1D smoke 任务（已带 --output）保持原路径、只出现一次。"""
        calls: list[list[str]] = []
        _install_fake_subprocess(monkeypatch, calls)
        explicit = str(out_dir.parent / "r1d_run" / "artifacts" / "unit" / "case" / "seed100")
        spec = br.ExperimentSpec(
            name="exp_r1d",
            script=OTHER_SCRIPT,
            grid=[{"--output": explicit, "--seed": 100, "_config_id": "unit__case"}],
        )

        planned = br.run_batch(spec, workers=1, resume=False, dry_run=True,
                               data_root=out_dir.parent)
        command = planned["tasks"][0]["command"]

        assert _argv_value(command, "--output") == explicit
        assert command.count("--output") == 1
        # 与不加注入的纯 build_command 完全一致
        assert command == br.build_command(OTHER_SCRIPT, spec.grid[0])

    def test_r1d_explicit_output_kept_in_real_run(
        self, monkeypatch: pytest.MonkeyPatch, out_dir: Path
    ) -> None:
        """真实执行路径同样不改写显式 --output，CSV 记录原值。"""
        calls: list[list[str]] = []
        _install_fake_subprocess(monkeypatch, calls)
        explicit = str(out_dir.parent / "r1d_run" / "seed100")
        spec = br.ExperimentSpec(
            name="exp_r1d",
            script=OTHER_SCRIPT,
            grid=[{"--output": explicit, "--seed": 100}],
        )

        br.run_batch(spec, workers=1, resume=False, data_root=out_dir.parent)

        assert _argv_value(calls[0], "--output") == explicit
        rows = _read_rows(out_dir.parent / "exp_r1d" / "results.csv")
        assert rows[0]["artifact_dir"] == explicit

    def test_non_v12_script_without_output_not_injected(
        self, monkeypatch: pytest.MonkeyPatch, out_dir: Path
    ) -> None:
        """非 V12 脚本即使没给 --output 也不注入（不猜别人的产物布局）。"""
        calls: list[list[str]] = []
        _install_fake_subprocess(monkeypatch, calls)
        spec = br.ExperimentSpec(
            name="exp_other",
            script=OTHER_SCRIPT,
            grid=[{"--ball-speed": 7, "--seed": 100}],
        )

        planned = br.run_batch(spec, workers=1, resume=False, dry_run=True,
                               data_root=out_dir.parent)

        command = planned["tasks"][0]["command"]
        assert "--output" not in command
        assert not calls

    def test_script_override_by_basename_is_recognized(self, out_dir: Path) -> None:
        """``_script`` 覆盖为绝对路径时仍按 basename 识别 V12 入口。"""
        absolute = str(br.PROJECT / RM65_SCRIPT)
        params = {"_script": absolute, "--seed": 100}

        prepared = br._prepare_task_params(RM65_SCRIPT, params, out_dir)

        assert Path(_argv_value(br.build_command(absolute, prepared), "--output")) \
            .is_relative_to(out_dir / "artifacts")
