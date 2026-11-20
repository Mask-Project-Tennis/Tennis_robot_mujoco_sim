"""R1-A7-D 诊断 CLI 工具测试（阶段 4 Task 4.8）。

覆盖参数面纪律（主代理裁定 1：强制参数/未知参数/不适用参数/显式
默认值/双重运动闸门均退出码 2）与端到端执行（monkeypatch 工厂注入
fake transport + fake clock，不触发真实 SDK 加载或 DDS）。

所有 sign/offset/增益/限值/幅度/频率/时长/阈值均为**测试专用有限
参数，非实机参数**。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

# 沿用 tests/test_multi_robot_sim_entrypoint.py 的导入方式（scripts/ 入 sys.path）
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "tools"))

import test_unitree_r1d as tool  # noqa: E402

from src.real.r1d_command_policy import R1DSlotHold  # noqa: E402
from src.real.r1d_types import R1DStateFrame  # noqa: E402
from src.real.unitree_r1d_interface import UnitreeR1DInterface  # noqa: E402

_LEFT_PROFILE = _REPO_ROOT / "configs" / "robots" / "r1d_left.yaml"

_LEFT_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)

# —— 测试专用参数（非实机参数）——
_TEST_SIGNS = (1, -1, 1, -1, 1, -1, 1)
_TEST_OFFSETS = (0.0, 0.05, -0.1, 0.2, -0.3, 0.4, 0.0)
_TEST_KP = (60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0)
_TEST_KD = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
_TEST_DQ_LIMIT = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
_TEST_Q_BOUND = 1.5
_TEST_TAU_LIMIT = 30.0
_PERIOD = 0.002
_TIMEOUT = 0.05
_STOP_DQ_THRESHOLD = 0.05
_MODE_MACHINE = 7
_RO_DURATION = 0.1  # 硬件配置诊断时长（read_only 端到端，测试专用）
_TRACK_DURATION = 0.12  # 硬件配置诊断时长（tracking 端到端，测试专用）
# 说明：0.12 s → 60 个发送周期，离散升余弦跨度 ≈0.00656 rad ≥ 新校验阈值
# max(0.1·A, 1e-3)=0.005 rad（P1-3，2026-11-18；0.06 s 仅 0.00164 rad 已被
# 接管前拒绝）。

# 显式策略哨兵槽位（manifest null 组）+ 对侧臂/头（manifest 示例数值
# 哨兵——R5 修订后全部 28 个非目标槽位须显式覆盖）
_EXPLICIT_SLOTS = tuple(range(0, 15)) + (31, 32, 33, 34)
_OTHER_ARM_SLOTS = (22, 23, 24, 25, 26, 27, 28)
_HEAD_SLOTS = (29, 30)
_ALL_POLICY_SLOTS = _EXPLICIT_SLOTS + _OTHER_ARM_SLOTS + _HEAD_SLOTS


def _slot_policy_entry(slot: int) -> dict[str, float]:
    """单槽显式策略条目（非目标槽哨兵值；对侧臂/头用 manifest 示例数值）。"""
    if slot in _HEAD_SLOTS:
        kp, kd = ((50.0, 2.0), (10.0, 0.1))[_HEAD_SLOTS.index(slot)]
        return {"mode": 1, "q": 0.0, "dq": 0.0, "kp": kp, "kd": kd, "tau": 0.0}
    if slot in _OTHER_ARM_SLOTS:
        kp = (100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0)[slot - 22]
        return {"mode": 1, "q": 0.0, "dq": 0.0, "kp": kp, "kd": 2.0, "tau": 0.0}
    return {
        "mode": 1,
        "q": 0.007 - 0.031 * slot,
        "dq": 0.0,
        "kp": 11.0 + slot,
        "kd": 0.4 + 0.13 * slot,
        "tau": 0.21 + 0.07 * slot,
    }


# ---------------------------------------------------------------------------
# fake 基础设施（与 test_r1d_diagnostics 同模式，自包含）
# ---------------------------------------------------------------------------


class FakeClock:
    """可注入的可变单调时钟（测试专用）。"""

    def __init__(self, start: float = 2000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += float(dt)


class FakeR1DTransport:
    """测试专用 fake 传输层。"""

    def __init__(self) -> None:
        self.open_count = 0
        self.acquire_count = 0
        self.publish_count = 0
        self.release_count = 0
        self.close_count = 0
        self.publish_frames: list[Any] = []
        self.on_state = None
        self.open_error: Exception | None = None

    def open(self, on_state) -> None:
        if self.open_error is not None:
            raise self.open_error
        self.open_count += 1
        self.on_state = on_state

    def acquire_control(self) -> None:
        self.acquire_count += 1

    def publish(self, command: Any) -> None:
        self.publish_count += 1
        self.publish_frames.append(command)

    def release_control(self) -> None:
        self.release_count += 1

    def close(self) -> None:
        self.close_count += 1


def _measured_q() -> np.ndarray:
    """35 槽唯一哨兵实测姿态。"""
    return np.array([0.1 * slot - 2.0 for slot in range(35)])


class _FakeFactory:
    """替身工厂：构造 fake 驱动并自起泵线程（fake clock + 状态帧投递）。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.handles: list[dict[str, Any]] = []
        # 置 True 时下一次构造的 transport 注入 open 失败（测试失败路径）
        self.open_error_next = False

    def __call__(
        self,
        profile,
        config,
        *,
        acquire_timeout_s: float,
        slot_policies=None,
        stop_confirm_dq_threshold=None,
        expected_mode_machine=None,
        **kwargs,
    ) -> UnitreeR1DInterface:
        self.calls.append(
            {
                "acquire_timeout_s": acquire_timeout_s,
                "slot_policies": slot_policies,
                "stop_confirm_dq_threshold": stop_confirm_dq_threshold,
                "expected_mode_machine": expected_mode_machine,
            }
        )
        clock = FakeClock()
        transport = FakeR1DTransport()
        if self.open_error_next:
            transport.open_error = RuntimeError("simulated open failure")
        arm = UnitreeR1DInterface(
            config,
            profile,
            transport,
            clock=clock,
            slot_policies=slot_policies,
            stop_confirm_dq_threshold=stop_confirm_dq_threshold,
        )
        done = threading.Event()
        pump = threading.Thread(
            target=self._pump, args=(clock, transport, done), daemon=True
        )
        self.handles.append(
            {"arm": arm, "clock": clock, "transport": transport, "done": done}
        )
        pump.start()
        return arm

    @staticmethod
    def _pump(clock: FakeClock, transport: FakeR1DTransport, done: threading.Event) -> None:
        """泵线程：每 2ms 真实时间推进 fake 时钟 0.0005s 并投递状态帧。

        fake 时钟 1/4 速推进：诊断按墙钟节奏 send（每 2ms），发布记录
        的 published_at 间隔按 fake 时钟计 ~0.5ms，远低于期限（=
        硬件配置控制周期），避免 fake 时钟失真造成伪超期（真实场景
        的期限语义由 test_r1d_diagnostics 手算纯函数测试覆盖）。
        """
        deadline = time.monotonic() + 8.0
        while not done.is_set() and time.monotonic() < deadline:
            clock.advance(0.0005)
            if transport.on_state is not None:
                transport.on_state(
                    R1DStateFrame(
                        q=_measured_q(),
                        dq=np.zeros(35),
                        sequence=None,
                        source_time=None,
                        received_at=clock(),
                        mode_machine=_MODE_MACHINE,
                        session_id=1,
                    )
                )
            time.sleep(0.002)

    def stop_all(self) -> None:
        """停止全部泵线程。"""
        for handle in self.handles:
            handle["done"].set()


@pytest.fixture
def fake_factory(monkeypatch: pytest.MonkeyPatch) -> _FakeFactory:
    """注入替身工厂并保证测试结束时停止泵线程。"""
    factory = _FakeFactory()
    monkeypatch.setattr(tool, "build_r1d_arm", factory)
    yield factory
    factory.stop_all()


# ---------------------------------------------------------------------------
# 配置文件构造
# ---------------------------------------------------------------------------


def _hardware_dict(
    *, motion_grade: bool, duration: float, network: str | None = "eth-test"
) -> dict[str, Any]:
    """构造测试硬件配置字典（read_only 级或 motion 级，非实机参数）。"""
    slots = {name: 15 + i for i, name in enumerate(_LEFT_JOINT_NAMES)}
    data: dict[str, Any] = {
        "schema_version": 1,
        "robot_id": "unitree_r1d",
        "side": "left",
        "communication": {
            "dds_domain": 0,
            "network_interface": network,
            "lowcmd_topic": "rt/lowcmd",
            "lowstate_topic": "rt/lowstate",
        },
        "layout": {"layout_verified": True, "joint_slots": slots},
        "timing": {
            "control_period_s": _PERIOD,
            "state_timeout_s": _TIMEOUT,
            "diagnostic_duration_s": duration,
        },
    }
    if motion_grade:
        data["calibration"] = {
            "verified": True,
            "joint_signs": dict(zip(_LEFT_JOINT_NAMES, _TEST_SIGNS)),
            "joint_offsets_rad": dict(zip(_LEFT_JOINT_NAMES, _TEST_OFFSETS)),
        }
        data["limits"] = {
            "verified": True,
            "q_lower_rad": {n: -_TEST_Q_BOUND for n in _LEFT_JOINT_NAMES},
            "q_upper_rad": {n: _TEST_Q_BOUND for n in _LEFT_JOINT_NAMES},
            "dq_limit_rad_s": dict(zip(_LEFT_JOINT_NAMES, _TEST_DQ_LIMIT)),
            "tau_limit_nm": {n: _TEST_TAU_LIMIT for n in _LEFT_JOINT_NAMES},
            "kp_nm_per_rad": dict(zip(_LEFT_JOINT_NAMES, _TEST_KP)),
            "kd_nms_per_rad": dict(zip(_LEFT_JOINT_NAMES, _TEST_KD)),
        }
    else:
        data["calibration"] = {"verified": False, "joint_signs": None, "joint_offsets_rad": None}
        data["limits"] = {"verified": False}
        for field in (
            "q_lower_rad", "q_upper_rad", "dq_limit_rad_s", "tau_limit_nm",
            "kp_nm_per_rad", "kd_nms_per_rad",
        ):
            data["limits"][field] = None
    return data


def _write_yaml(tmp_path: Path, name: str, data: dict[str, Any]) -> Path:
    """写测试 YAML 并返回路径。"""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _slot_policies_file(tmp_path: Path) -> Path:
    """写覆盖全部 28 个非目标槽位的显式策略 YAML（测试哨兵值）。

    R5 修订（评审 I-05）：对侧臂/头槽位也须显式给出（数值取 manifest
    示例哨兵），不再依赖运行时回退。
    """
    entries = {slot: _slot_policy_entry(slot) for slot in _ALL_POLICY_SLOTS}
    return _write_yaml(tmp_path, "slots.yaml", {"slot_policies": entries})


def _read_only_argv(
    tmp_path: Path, hw: Path, out: Path, extra: list[str] | None = None
) -> list[str]:
    """read_only 基础参数（sample 周期必填，测试专用值）。"""
    return [
        "--robot-config", str(_LEFT_PROFILE),
        "--hardware-config", str(hw),
        "--output-dir", str(out),
        "--acquire-timeout-s", "1.0",
        "--sample-period-s", "0.02",
    ] + (extra or [])


def _tracking_argv(
    tmp_path: Path, hw: Path, out: Path, extra: list[str] | None = None
) -> list[str]:
    """tracking 全量合法参数（数值为测试专用值）。

    R4（评审 2026-11-18）：两项验收阈值在 tracking 模式必填（缺省退出 2），
    基础参数即携带宽阈值；缺失用例经 drop 参数显式删除验证拒绝路径。
    """
    return [
        "--robot-config", str(_LEFT_PROFILE),
        "--hardware-config", str(hw),
        "--output-dir", str(out),
        "--acquire-timeout-s", "1.0",
        "--mode", "tracking",
        "--enable-motion",
        "--selected-joint", "left_shoulder_pitch_joint",
        "--amplitude-rad", "0.05",
        "--frequency-hz", "1.0",
        "--stop-confirm-dq-threshold", str(_STOP_DQ_THRESHOLD),
        "--tracking-error-limit-rad", "1.0",
        "--stop-latency-limit-s", "10.0",
        "--slot-policies", str(_slot_policies_file(tmp_path)),
    ] + (extra or [])


# ---------------------------------------------------------------------------
# 参数面纪律（退出码 2）
# ---------------------------------------------------------------------------


class TestArgvRejection:
    """参数纪律：强制/未知/不适用/显式默认值/运动闸门 → 退出码 2。"""

    def test_missing_required_arg_exits_2(self, tmp_path: Path) -> None:
        """缺任一强制参数（argparse）→ SystemExit 2。"""
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=False, duration=_RO_DURATION)
        )
        full = _read_only_argv(tmp_path, hw, tmp_path / "out")
        # 逐项移除 --robot-config/--hardware-config/--output-dir/--acquire-timeout-s
        for flag in ("--robot-config", "--hardware-config", "--output-dir", "--acquire-timeout-s"):
            argv = [a for a in full]
            index = argv.index(flag)
            del argv[index:index + 2]
            with pytest.raises(SystemExit) as exc:
                tool.main(argv)
            assert int(exc.value.code) == 2

    def test_unknown_arg_exits_2(self, tmp_path: Path) -> None:
        """未知参数（argparse 不识别）→ SystemExit 2。"""
        with pytest.raises(SystemExit) as exc:
            tool.main(["--bogus-flag", "1"])
        assert int(exc.value.code) == 2

    @pytest.mark.parametrize(
        "extra",
        [
            ["--selected-joint", "left_elbow_joint"],
            ["--amplitude-rad", "0.05"],
            ["--amplitude-rad", "0"],          # 显式等于"默认概念值"也拒绝
            ["--frequency-hz", "1.0"],
            ["--frequency-hz", "0"],
            ["--stop-confirm-dq-threshold", "0.05"],
            ["--slot-policies", "slots.yaml"],
            ["--enable-motion"],
            ["--tracking-error-limit-rad", "0.01"],  # I-06：验收阈值仅 tracking
            ["--stop-latency-limit-s", "1.0"],       # I-06：同上
        ],
    )
    def test_read_only_rejects_tracking_args(
        self, tmp_path: Path, extra: list[str]
    ) -> None:
        """read_only 下显式给出 tracking 专属参数（任意值）→ 2 且点名。"""
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=False, duration=_RO_DURATION)
        )
        assert tool.main(_read_only_argv(tmp_path, hw, tmp_path / "out", extra)) == 2

    @pytest.mark.parametrize("bad_value", ["-1", "256"])
    def test_expected_mode_machine_out_of_uint8_rejected(
        self, tmp_path: Path, bad_value: str
    ) -> None:
        """I-08：--expected-mode-machine 越出 uint8 [0,255] → 2 且点名。"""
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=False, duration=_RO_DURATION)
        )
        argv = _read_only_argv(
            tmp_path, hw, tmp_path / "out", ["--expected-mode-machine", bad_value]
        )
        assert tool.main(argv) == 2

    def test_read_only_requires_sample_period(self, tmp_path: Path) -> None:
        """read_only 缺 --sample-period-s → 2。"""
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=False, duration=_RO_DURATION)
        )
        argv = [
            "--robot-config", str(_LEFT_PROFILE),
            "--hardware-config", str(hw),
            "--output-dir", str(tmp_path / "out"),
            "--acquire-timeout-s", "1.0",
        ]
        assert tool.main(argv) == 2

    def test_tracking_requires_enable_motion(self, tmp_path: Path) -> None:
        """--mode tracking 无 --enable-motion → 2（双重运动闸门）。"""
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=True, duration=_TRACK_DURATION)
        )
        argv = [
            "--robot-config", str(_LEFT_PROFILE),
            "--hardware-config", str(hw),
            "--output-dir", str(tmp_path / "out"),
            "--acquire-timeout-s", "1.0",
            "--mode", "tracking",
            "--selected-joint", "left_elbow_joint",
            "--amplitude-rad", "0.05",
            "--frequency-hz", "1.0",
            "--stop-confirm-dq-threshold", str(_STOP_DQ_THRESHOLD),
        ]
        assert tool.main(argv) == 2

    def test_tracking_rejects_sample_period(self, tmp_path: Path) -> None:
        """tracking 下 --sample-period-s（read_only 专属）→ 2。"""
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=True, duration=_TRACK_DURATION)
        )
        assert (
            tool.main(
                _tracking_argv(tmp_path, hw, tmp_path / "out", ["--sample-period-s", "0.02"])
            )
            == 2
        )

    @pytest.mark.parametrize(
        "drop_flag",
        [
            "--selected-joint", "--amplitude-rad", "--frequency-hz",
            "--stop-confirm-dq-threshold",
            # R4（评审 2026-11-18）：验收阈值在 tracking 模式必填
            "--tracking-error-limit-rad", "--stop-latency-limit-s",
        ],
    )
    def test_tracking_missing_required_fields(
        self, tmp_path: Path, drop_flag: str
    ) -> None:
        """tracking 缺任一必填参数 → 2。"""
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=True, duration=_TRACK_DURATION)
        )
        argv = _tracking_argv(tmp_path, hw, tmp_path / "out")
        index = argv.index(drop_flag)
        del argv[index:index + 2]
        assert tool.main(argv) == 2

    @pytest.mark.parametrize(
        "flag", ["--tracking-error-limit-rad", "--stop-latency-limit-s"]
    )
    def test_tracking_missing_acceptance_threshold_rejected(
        self, tmp_path: Path, flag: str, capsys: pytest.CaptureFixture
    ) -> None:
        """R4：缺验收阈值 → 退出码 2 且原因点名参数（不得静默进入采集）。

        仅采集（无冻结判据）不得共用验收 pass——CLI 层直接拒绝缺参，
        库级判定同口径兜底 fail（tests/test_r1d_diagnostics.py）。
        """
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=True, duration=_TRACK_DURATION)
        )
        argv = _tracking_argv(tmp_path, hw, tmp_path / "out")
        index = argv.index(flag)
        del argv[index:index + 2]
        assert tool.main(argv) == 2
        assert flag in capsys.readouterr().err

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("--acquire-timeout-s", "0"),
            ("--acquire-timeout-s", "-1"),
            ("--amplitude-rad", "0"),
            ("--frequency-hz", "-2"),
        ],
    )
    def test_non_positive_numeric_values_rejected(
        self, tmp_path: Path, flag: str, value: str
    ) -> None:
        """非正数值参数 → 2。"""
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=True, duration=_TRACK_DURATION)
        )
        argv = _tracking_argv(tmp_path, hw, tmp_path / "out")
        index = argv.index(flag)
        argv[index + 1] = value
        assert tool.main(argv) == 2

    def test_tracking_requires_motion_grade_config(self, tmp_path: Path) -> None:
        """tracking + read_only 级硬件配置（校准 null）→ 2（motion 门槛）。"""
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=False, duration=_TRACK_DURATION)
        )
        assert tool.main(_tracking_argv(tmp_path, hw, tmp_path / "out")) == 2

    def test_read_only_requires_valid_communication(self, tmp_path: Path) -> None:
        """read_only + network_interface null → 2（read_only 门槛）。"""
        hw = _write_yaml(
            tmp_path,
            "hw.yaml",
            _hardware_dict(motion_grade=False, duration=_RO_DURATION, network=None),
        )
        assert tool.main(_read_only_argv(tmp_path, hw, tmp_path / "out")) == 2

    def test_missing_diagnostic_duration_rejected(self, tmp_path: Path) -> None:
        """timing.diagnostic_duration_s null → 2（时长必须现场冻结）。"""
        data = _hardware_dict(motion_grade=False, duration=_RO_DURATION)
        data["timing"]["diagnostic_duration_s"] = None
        hw = _write_yaml(tmp_path, "hw.yaml", data)
        assert tool.main(_read_only_argv(tmp_path, hw, tmp_path / "out")) == 2

    def test_unknown_selected_joint_rejected(self, tmp_path: Path) -> None:
        """--selected-joint 不在 profile 关节名中 → 2。"""
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=True, duration=_TRACK_DURATION)
        )
        argv = _tracking_argv(tmp_path, hw, tmp_path / "out")
        index = argv.index("--selected-joint")
        argv[index + 1] = "bogus_joint"
        assert tool.main(argv) == 2

    def test_tracking_short_waveform_rejected_before_acquire(
        self,
        tmp_path: Path,
        fake_factory: _FakeFactory,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """离散波形不足 2 步（duration=period）→ 配置构造即拒绝，不接管驱动。

        P1-3（2026-11-18）：零运动/单步波形必须在接管之前失败——CLI 在
        ``build_r1d_arm`` 之前构造 ``R1DDiagnosticConfig``，新校验抛出
        ValueError 后经 except 转退出码 2，工厂零调用（未 open/未 acquire）。
        """
        hw = _write_yaml(
            tmp_path,
            "hw.yaml",
            _hardware_dict(motion_grade=True, duration=_PERIOD),
        )
        assert tool.main(_tracking_argv(tmp_path, hw, tmp_path / "out")) == 2
        assert fake_factory.calls == []
        assert fake_factory.handles == []
        stderr = capsys.readouterr().err
        assert "配置加载失败" in stderr
        assert "波形" in stderr


# ---------------------------------------------------------------------------
# 端到端（monkeypatch 工厂 → fake transport/clock）
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """端到端执行（fake）：产物生成、退出码、工厂参数透传。"""

    def test_read_only_run_passes(
        self, tmp_path: Path, fake_factory: _FakeFactory, capsys: pytest.CaptureFixture
    ) -> None:
        """read_only 端到端：退出码 0、零发布零接管、产物齐全。"""
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=False, duration=_RO_DURATION)
        )
        out = tmp_path / "out"
        assert tool.main(_read_only_argv(tmp_path, hw, out)) == 0
        handle = fake_factory.handles[0]
        transport = handle["transport"]
        assert transport.publish_count == 0
        assert transport.acquire_count == 0
        assert transport.close_count == 1
        for name in (
            "diagnostic_summary.json", "diagnostic_publish_records.csv",
            "diagnostic_samples.csv", "diagnostic_send_log.csv", "diagnostic_raw.npz",
        ):
            assert (out / name).is_file(), f"缺产物 {name}"
        stdout = capsys.readouterr().out
        assert "诊断结论: pass" in stdout
        # 摘要含 SDK 标识与代码 SHA（溯源）
        assert "unitree_sdk2py@814556d1" in stdout

    def test_explicit_mode_read_only_accepted(
        self, tmp_path: Path, fake_factory: _FakeFactory
    ) -> None:
        """显式 --mode read_only（等于缺省默认值）不被静默忽略：正常生效。"""
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=False, duration=_RO_DURATION)
        )
        argv = _read_only_argv(
            tmp_path, hw, tmp_path / "out", ["--mode", "read_only"]
        )
        assert tool.main(argv) == 0
        handle = fake_factory.handles[0]
        assert handle["transport"].publish_count == 0

    def test_tracking_run_passes(
        self, tmp_path: Path, fake_factory: _FakeFactory
    ) -> None:
        """tracking 端到端：退出码 0、接管发布发生、产物与工厂参数正确。"""
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=True, duration=_TRACK_DURATION)
        )
        out = tmp_path / "out"
        assert tool.main(_tracking_argv(tmp_path, hw, out)) == 0
        handle = fake_factory.handles[0]
        transport = handle["transport"]
        assert transport.acquire_count == 1
        assert transport.publish_count >= 2
        assert transport.release_count == 1
        # 工厂参数透传：槽位策略解析为显式 R1DSlotHold、停止阈值来自 CLI
        call = fake_factory.calls[0]
        assert call["acquire_timeout_s"] == 1.0
        assert call["stop_confirm_dq_threshold"] == _STOP_DQ_THRESHOLD
        policies = call["slot_policies"]
        # R5 修订：YAML 显式覆盖全部 28 个非目标槽位（含对侧臂/头）
        assert policies is not None and len(policies) == len(_ALL_POLICY_SLOTS)
        assert isinstance(policies[0], R1DSlotHold)
        # 摘要 JSON：选中关节解析为模型序索引 0（肩俯仰 = 首关节）
        import json

        summary = json.loads((out / "diagnostic_summary.json").read_text(encoding="utf-8"))
        assert summary["verdict"] == "pass"
        assert summary["config"]["selected_joint"] == 0
        assert summary["config"]["send_period_s"] == _PERIOD
        assert summary["config"]["duration_s"] == _TRACK_DURATION
        assert summary["provenance"]["side"] == "left"
        assert summary["provenance"]["hardware_config"]["sha256"]

    def test_tracking_thresholds_and_mode_machine_wired(
        self, tmp_path: Path, fake_factory: _FakeFactory
    ) -> None:
        """I-06/I-08（评审 2026-11-17）：验收阈值与冻结机型经 CLI 注入生效。

        --tracking-error-limit-rad/--stop-latency-limit-s 进入诊断配置
        （此前字段存在但无注入通路，规程 pass 判据无法由工具实现）；
        --expected-mode-machine 透传 build_r1d_arm（G3 冻结后的校验通路）。
        """
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=True, duration=_TRACK_DURATION)
        )
        out = tmp_path / "out"
        argv = _tracking_argv(
            tmp_path, hw, out,
            [
                "--tracking-error-limit-rad", "0.05",
                "--stop-latency-limit-s", "3.0",
                "--expected-mode-machine", "7",
            ],
        )
        assert tool.main(argv) == 0
        # I-08：冻结机型透传工厂（缺省调用为 None）
        default_call = fake_factory.calls[0]
        assert default_call["expected_mode_machine"] == 7
        # I-06：阈值进入诊断配置并写入摘要（可判定，实际值 ≤ 阈值 → pass）
        import json

        summary = json.loads((out / "diagnostic_summary.json").read_text(encoding="utf-8"))
        assert summary["config"]["tracking_error_limit_rad"] == 0.05
        assert summary["config"]["stop_latency_limit_s"] == 3.0
        assert summary["verdict"] == "pass"

    def test_failure_returns_exit_1(
        self, tmp_path: Path, fake_factory: _FakeFactory
    ) -> None:
        """诊断失败（传输层 open 失败）→ main 返回 1。"""
        fake_factory.open_error_next = True
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=False, duration=0.02)
        )
        assert tool.main(_read_only_argv(tmp_path, hw, tmp_path / "out")) == 1

    def test_read_only_grade_config_ok_for_read_only_mode(
        self, tmp_path: Path, fake_factory: _FakeFactory
    ) -> None:
        """read_only 级配置（校准 null）在 read_only 模式合法（门槛梯度）。"""
        hw = _write_yaml(
            tmp_path, "hw.yaml", _hardware_dict(motion_grade=False, duration=_RO_DURATION)
        )
        assert tool.main(_read_only_argv(tmp_path, hw, tmp_path / "out")) == 0
