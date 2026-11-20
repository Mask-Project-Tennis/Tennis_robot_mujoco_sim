"""R1-A7-D 只读/低速诊断工具与可追踪指标测试（阶段 4 Task 4.8）。

所有 sign/offset/增益/限值/幅度/频率/时长/阈值均为**测试专用有限参数，
非实机参数**：仅用于验证诊断流程、指标计算与产物 schema，不代表任何
实物标定或已核验数值。全机非目标槽位策略沿用 4.3/4.6 测试的全量显式
哨兵策略。

测试用 fake transport + fake clock 跑完整只读与 tracking 场景（含故障
案例），不触发真实 SDK 加载或 DDS；等待统一走泵线程（推进 fake 时钟
并投递状态帧，短间隔轮询，不依赖长 sleep）。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pytest
import yaml
from numpy.testing import assert_allclose

from src.real.r1d_command_policy import R1DSlotHold
from src.real.r1d_diagnostics import (
    R1D_DIAGNOSTIC_SCHEMA_VERSION,
    R1D_SDK_IDENTITY,
    R1DDiagnosticConfig,
    compute_publish_interval_metrics,
    compute_tracking_error_metrics,
    run_r1d_diagnostic,
)
from src.real.r1d_factory import build_r1d_arm
from src.real.r1d_types import R1DStateFrame
from src.robot.robot_profile import load_robot_profile

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LEFT_PROFILE_PATH = _REPO_ROOT / "configs" / "robots" / "r1d_left.yaml"

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
_TEST_KP = (60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0)  # N·m/rad
_TEST_KD = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)  # N·m·s/rad
_TEST_DQ_LIMIT = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)  # rad/s
_TEST_Q_BOUND = 1.5  # rad
_TEST_TAU_LIMIT = 30.0  # N·m
_PERIOD = 0.002  # s（数值同官方示例，仅测试用）
_TIMEOUT = 0.05  # s（状态新鲜度超时，测试专用）
_STOP_DQ_THRESHOLD = 0.05  # rad/s（停止确认速度阈值，测试专用）
_MODE_MACHINE = 7  # 测试专用机型回读值
_ACQUIRE_TIMEOUT = 1.0  # s（fake 路径不被消费，测试专用）
# tracking 诊断时长：须让离散升余弦波形产生可验证跨度（P1-3，2026-11-18）。
# 0.12 s → 60 个发送周期；离散跨度 0.025·(1−cos(2π·0.118))≈0.00656 rad
# ≥ 阈值 max(0.1·A, 1e-3)=0.005 rad；隐含峰值速度 A·π·f≈0.157 rad/s
# 远小于驱动 dq_limit（测试专用参数，非实机值）。
_DIAG_DURATION = 0.12  # s（tracking 诊断时长，测试专用）
_RO_DURATION = 0.1  # s（read_only 诊断时长，测试专用）
_SAMPLE_PERIOD = 0.02  # s（read_only 采样周期，测试专用）
_AMPLITUDE = 0.05  # rad（tracking 波形幅度，测试专用）
_FREQUENCY = 1.0  # Hz（tracking 波形频率，测试专用）

# 实测姿态哨兵（与 4.6 测试一致）：q[slot] = 0.1*slot - 2.0
# 左臂槽 15-21 实测 SDK = (-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1)
_LEFT_MEASURED_SDK = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1)
# 锚定模型序 q = sign*(q_sdk - offset)（手算）
_Q_ANCHORED = np.array((-0.5, 0.45, -0.2, 0.4, 0.2, 0.4, 0.1))

# 显式策略须覆盖的 19 个 manifest-null 槽位（与 4.3/4.6 测试一致）
_EXPLICIT_SLOTS = tuple(range(0, 15)) + (31, 32, 33, 34)

# 泵线程真实时长上限（软件测试参数）
_PUMP_TIMEOUT_S = 8.0


# ---------------------------------------------------------------------------
# 测试辅助（fake clock / fake transport / 配置与驱动构造 / 泵线程）
# ---------------------------------------------------------------------------


class FakeClock:
    """可注入的可变单调时钟（测试专用）。"""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += float(dt)


class FakeR1DTransport:
    """测试专用 fake 传输层（记录全部调用，支持受控失败）。"""

    def __init__(self) -> None:
        self.open_count = 0
        self.acquire_count = 0
        self.publish_count = 0
        self.release_count = 0
        self.close_count = 0
        self.publish_frames: list[Any] = []
        self.on_state: Callable[[R1DStateFrame], None] | None = None
        self.open_error: Exception | None = None
        self.acquire_error: Exception | None = None
        self.release_error: Exception | None = None
        self.publish_error: Exception | None = None

    def open(self, on_state: Callable[[R1DStateFrame], None]) -> None:
        if self.open_error is not None:
            raise self.open_error
        self.open_count += 1
        self.on_state = on_state

    def acquire_control(self) -> None:
        if self.acquire_error is not None:
            raise self.acquire_error
        self.acquire_count += 1

    def publish(self, command: Any) -> None:
        if self.publish_error is not None:
            raise self.publish_error
        self.publish_count += 1
        self.publish_frames.append(command)

    def release_control(self) -> None:
        if self.release_error is not None:
            raise self.release_error
        self.release_count += 1

    def close(self) -> None:
        self.close_count += 1

    def deliver(self, frame: R1DStateFrame) -> None:
        """测试注入状态帧（模拟 lowstate 回调）。"""
        assert self.on_state is not None, "尚未 open"
        self.on_state(frame)


def _measured_q() -> np.ndarray:
    """35 槽唯一哨兵实测姿态（q[slot] = 0.1*slot - 2.0）。"""
    return np.array([0.1 * slot - 2.0 for slot in range(35)])


def _measured_dq() -> np.ndarray:
    """35 槽非零速度哨兵（dq[slot] = 0.01*(slot+1)，超停止阈值）。"""
    return np.array([0.01 * (slot + 1) for slot in range(35)])


def _state_frame(
    clock: FakeClock,
    *,
    mode_machine: int = _MODE_MACHINE,
    dq: np.ndarray | None = None,
    q: np.ndarray | None = None,
) -> R1DStateFrame:
    """构造以 fake clock 当前时刻接收的整机状态帧（测试专用）。"""
    return R1DStateFrame(
        q=q if q is not None else _measured_q(),
        dq=dq if dq is not None else np.zeros(35, dtype=np.float64),
        sequence=None,
        source_time=None,
        received_at=clock(),
        mode_machine=mode_machine,
        session_id=1,
    )


def _feed(
    transport: FakeR1DTransport,
    clock: FakeClock,
    *,
    dq: np.ndarray | None = None,
) -> None:
    """向驱动投递一帧以当前 fake 时钟计时的状态帧。"""
    transport.deliver(_state_frame(clock, dq=dq))


def _pump(
    clock: FakeClock,
    transport: FakeR1DTransport,
    done: threading.Event,
    *,
    fast_dq: bool = False,
    stop_feed_after_s: float | None = None,
    advance_s: float = 0.001,
) -> None:
    """泵线程：推进 fake 时钟并按需投递状态帧，直到 done 置位或超时。

    fast_dq=True 时投递超停止阈值的高速帧（制造"停止未确认"）；
    stop_feed_after_s 给出时，真实时间超过该值后停止喂帧但继续推进
    时钟（制造"状态流中断"）；advance_s 控制每个真实 2ms 轮次推进的
    fake 时钟步长（默认半速 0.001——tracking 发布间隔远低于期限更
    稳定；流中断测试用同步速率 0.002 保证超时按时触发）。
    """
    start = time.monotonic()
    deadline = start + _PUMP_TIMEOUT_S
    while not done.is_set() and time.monotonic() < deadline:
        clock.advance(advance_s)
        feeding = (
            transport.on_state is not None  # connect 完成前不投递
            and (
                stop_feed_after_s is None
                or time.monotonic() - start < stop_feed_after_s
            )
        )
        if feeding:
            dq = _measured_dq() if fast_dq else None
            _feed(transport, clock, dq=dq)
        time.sleep(0.002)


def _explicit_hold(slot: int) -> R1DSlotHold:
    """测试专用显式槽位保持策略（逐槽唯一哨兵，非实机参数）。"""
    return R1DSlotHold(
        mode=1,
        q=0.007 - 0.031 * slot,
        dq=0.0,
        kp=11.0 + slot,
        kd=0.4 + 0.13 * slot,
        tau=0.21 + 0.07 * slot,
    )


# 对侧臂/头显式哨兵（数值=manifest [公开示例]；R5 修订后须显式给出，
# 不再运行时回退）——左臂驱动的对侧=右臂 22-28，头 29/30
_OTHER_ARM_SLOTS = (22, 23, 24, 25, 26, 27, 28)
_HEAD_SLOTS = (29, 30)


def _manifest_example_hold(slot: int) -> R1DSlotHold:
    """对侧臂/头槽显式哨兵（数值=manifest 示例，非运行时回退）。"""
    if slot in _HEAD_SLOTS:
        offset = _HEAD_SLOTS.index(slot)
        return R1DSlotHold(
            mode=1, q=0.0, dq=0.0, kp=(50.0, 10.0)[offset],
            kd=(2.0, 0.1)[offset], tau=0.0,
        )
    other_kp = (100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0)
    return R1DSlotHold(
        mode=1, q=0.0, dq=0.0, kp=other_kp[slot - 22], kd=2.0, tau=0.0,
    )


def _full_explicit_policies() -> dict[int, R1DSlotHold]:
    """覆盖全部 28 个非目标槽位的显式策略（R5 修订：须全覆盖，测试专用）。"""
    policies = {slot: _explicit_hold(slot) for slot in _EXPLICIT_SLOTS}
    for slot in _OTHER_ARM_SLOTS:
        policies[slot] = _manifest_example_hold(slot)
    for slot in _HEAD_SLOTS:
        policies[slot] = _manifest_example_hold(slot)
    return policies


def _base_config_dict() -> dict[str, Any]:
    """motion 级硬件配置字典（测试专用有限参数，非实机参数）。"""
    slots = {name: 15 + i for i, name in enumerate(_LEFT_JOINT_NAMES)}
    signs = {name: s for name, s in zip(_LEFT_JOINT_NAMES, _TEST_SIGNS)}
    offsets = {name: o for name, o in zip(_LEFT_JOINT_NAMES, _TEST_OFFSETS)}
    return {
        "schema_version": 1,
        "robot_id": "unitree_r1d",
        "side": "left",
        "communication": {
            "dds_domain": 0,
            "network_interface": "eth-test",  # 测试专用，非实机参数
            "lowcmd_topic": "rt/lowcmd",
            "lowstate_topic": "rt/lowstate",
        },
        "layout": {"layout_verified": True, "joint_slots": slots},
        "calibration": {
            "verified": True,
            "joint_signs": signs,
            "joint_offsets_rad": offsets,
        },
        "limits": {
            "verified": True,
            "q_lower_rad": {n: -_TEST_Q_BOUND for n in _LEFT_JOINT_NAMES},
            "q_upper_rad": {n: _TEST_Q_BOUND for n in _LEFT_JOINT_NAMES},
            "dq_limit_rad_s": {n: d for n, d in zip(_LEFT_JOINT_NAMES, _TEST_DQ_LIMIT)},
            "tau_limit_nm": {n: _TEST_TAU_LIMIT for n in _LEFT_JOINT_NAMES},
            "kp_nm_per_rad": {n: k for n, k in zip(_LEFT_JOINT_NAMES, _TEST_KP)},
            "kd_nms_per_rad": {n: k for n, k in zip(_LEFT_JOINT_NAMES, _TEST_KD)},
        },
        "timing": {
            "control_period_s": _PERIOD,
            "state_timeout_s": _TIMEOUT,
            "diagnostic_duration_s": 5.0,
        },
    }


def _make_arm(
    tmp_path: Path,
    clock: FakeClock,
    transport: FakeR1DTransport | None = None,
    *,
    slot_policies: dict[int, R1DSlotHold] | None = None,
):
    """经 build_r1d_arm 构造被测驱动（注入 fake transport + fake clock）。"""
    from src.real.r1d_config import load_r1d_hardware_config

    data = _base_config_dict()
    path = tmp_path / "hw_left.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    config = load_r1d_hardware_config(path)
    profile = load_robot_profile(_LEFT_PROFILE_PATH)
    fake = transport if transport is not None else FakeR1DTransport()
    arm = build_r1d_arm(
        profile,
        config,
        acquire_timeout_s=_ACQUIRE_TIMEOUT,
        transport=fake,
        clock=clock,
        slot_policies=slot_policies,
        stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD,
    )
    return arm, fake


def _read_only_config(**overrides: Any) -> R1DDiagnosticConfig:
    """构造 read_only 诊断配置（测试专用参数，非实机参数）。"""
    values: dict[str, Any] = {
        "mode": "read_only",
        "duration_s": _RO_DURATION,
        "sample_period_s": _SAMPLE_PERIOD,
        "hardware_config_path": "tests://hw_left.yaml",  # 测试来源标记
        "hardware_config_sha256": "0" * 64,  # 测试占位哈希
        "side": "left",
    }
    values.update(overrides)
    return R1DDiagnosticConfig(**values)


def _tracking_config(**overrides: Any) -> R1DDiagnosticConfig:
    """构造 tracking 诊断配置（测试专用参数，非实机参数）。

    publish_deadline_s 用 CLI 的同款条件（= 1×period，不是宽期限）：超期
    判据是"间隔 > 1.5×期限"（发布开销与 fake 时钟量化计入半拍容差，见
    r1d_diagnostics._PUBLISH_INTERVAL_OVERRUN_RATIO，评审 2026-11-18 复
    审）；此配置让端到端用例覆盖 CLI 的真实判定条件，期限语义与超期
    计数正确性另由手算纯函数测试覆盖。

    tracking_error_limit_rad/stop_latency_limit_s 默认给宽阈值（R4，
    评审 2026-11-18：tracking 验收要求完整阈值，缺失即 fail）——阈值
    判定正确性由超限用例（收紧对应阈值）与缺失用例（显式置 None）覆盖。
    """
    values: dict[str, Any] = {
        "mode": "tracking",
        "duration_s": _DIAG_DURATION,
        "send_period_s": _PERIOD,
        "selected_joint": 0,
        "amplitude_rad": _AMPLITUDE,
        "frequency_hz": _FREQUENCY,
        "stop_confirm_dq_threshold": _STOP_DQ_THRESHOLD,
        "publish_deadline_s": _PERIOD,
        "tracking_error_limit_rad": 1.0,  # 宽阈值：仅证明判定未误报
        "stop_latency_limit_s": 10.0,  # 宽阈值：仅证明判定未误报
        "joint_signs": _TEST_SIGNS,
        "joint_offsets_rad": _TEST_OFFSETS,
        "hardware_config_path": "tests://hw_left.yaml",  # 测试来源标记
        "hardware_config_sha256": "0" * 64,  # 测试占位哈希
        "side": "left",
    }
    values.update(overrides)
    return R1DDiagnosticConfig(**values)


def _publish_row(
    *,
    seq: int,
    published_at: float,
    q_sdk: tuple[float, ...] | None,
    kind: str = "active",
    success: bool = True,
) -> dict[str, Any]:
    """构造收集器内部发布记录行（字段与 _consume_publishes 同构）。"""
    return {
        "seq": seq,
        "session_id": 1,
        "kind": kind,
        "published_at": float(published_at),
        "q_sdk": None if q_sdk is None else np.asarray(q_sdk, dtype=np.float64),
        "mode_machine": _MODE_MACHINE,
        "success": bool(success),
        "error": None if success else "fake transport 拒绝",
    }


def _sdk_target(*, joint: int, value: float) -> tuple[float, ...]:
    """7 关节 SDK 侧目标向量：选中关节取给定值，其余为 0。"""
    values = [0.0] * 7
    values[joint] = float(value)
    return tuple(values)


def _finalized_verdict(
    config: R1DDiagnosticConfig,
    output_dir: Path,
    publishes: list[dict[str, Any]],
) -> dict[str, Any]:
    """注入发布记录后直接 finalize（只测判定层，不启动驱动/线程）。

    判定层不依赖 fake 驱动的发送节拍，直接给收集器注入与
    ``_consume_publishes`` 同构的发布记录，避开计时 flake。
    """
    from src.real.r1d_diagnostics import _DiagnosticCollector

    collector = _DiagnosticCollector(config, output_dir)
    collector._completed_normally = True
    collector._publishes = list(publishes)
    return collector.finalize()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


# ---------------------------------------------------------------------------
# 模块纯净性
# ---------------------------------------------------------------------------


def test_module_import_is_pure() -> None:
    """import 诊断模块不得加载 SDK/MuJoCo、不得启动线程。"""
    banned = ("unitree_sdk2py", "cyclonedds", "mujoco")
    saved = {name: sys.modules.get(name) for name in banned}
    for name in banned:
        sys.modules.pop(name, None)
    before_threads = threading.active_count()
    try:
        import src.real.r1d_diagnostics  # noqa: F401

        for name in banned:
            assert name not in sys.modules, f"import 不应加载 {name}"
        assert threading.active_count() == before_threads, "import 不得启动线程"
    finally:
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module


# ---------------------------------------------------------------------------
# 配置校验
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """R1DDiagnosticConfig 校验：不适用字段显式给出即拒绝（不静默忽略）。"""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("selected_joint", 0),
            ("amplitude_rad", _AMPLITUDE),
            ("frequency_hz", _FREQUENCY),
            ("stop_confirm_dq_threshold", _STOP_DQ_THRESHOLD),
            ("send_period_s", _PERIOD),
            ("joint_signs", _TEST_SIGNS),
            ("joint_offsets_rad", _TEST_OFFSETS),
        ],
    )
    def test_read_only_rejects_tracking_fields(
        self, field: str, value: Any
    ) -> None:
        """read_only 模式下显式给出 tracking 专属字段 → 构造拒绝。"""
        with pytest.raises(ValueError, match=field):
            _read_only_config(**{field: value})

    def test_read_only_requires_sample_period(self) -> None:
        """read_only 缺采样周期 → 拒绝。"""
        with pytest.raises(ValueError, match="sample_period_s"):
            _read_only_config(sample_period_s=None)

    @pytest.mark.parametrize(
        "field",
        [
            "send_period_s",
            "amplitude_rad",
            "frequency_hz",
            "stop_confirm_dq_threshold",
            "selected_joint",
            "joint_signs",
            "joint_offsets_rad",
        ],
    )
    def test_tracking_requires_motion_fields(self, field: str) -> None:
        """tracking 缺任一必需字段 → 拒绝。"""
        values: dict[str, Any] = {
            "send_period_s": _PERIOD,
            "amplitude_rad": _AMPLITUDE,
            "frequency_hz": _FREQUENCY,
            "stop_confirm_dq_threshold": _STOP_DQ_THRESHOLD,
            "selected_joint": 0,
            "joint_signs": _TEST_SIGNS,
            "joint_offsets_rad": _TEST_OFFSETS,
        }
        values[field] = None
        with pytest.raises(ValueError, match=field):
            _tracking_config(**values)

    def test_tracking_rejects_sample_period(self) -> None:
        """tracking 不适用采样周期字段（采样随发送周期）→ 拒绝。"""
        with pytest.raises(ValueError, match="sample_period_s"):
            _tracking_config(sample_period_s=_SAMPLE_PERIOD)

    @pytest.mark.parametrize(
        "overrides,match",
        [
            ({"duration_s": 0.0}, "duration_s"),
            ({"selected_joint": 7}, "selected_joint"),
            ({"selected_joint": -1}, "selected_joint"),
            ({"joint_signs": (1,) * 7, "joint_offsets_rad": (0.0,) * 6}, "joint_offsets_rad"),
            ({"joint_signs": (1, 2, 1, 1, 1, 1, 1)}, "joint_signs"),
        ],
    )
    def test_invalid_values_rejected(
        self, overrides: dict[str, Any], match: str
    ) -> None:
        """非法数值（非正时长/越界索引/不成对校准/非 ±1 sign）→ 拒绝。"""
        with pytest.raises(ValueError, match=match):
            _tracking_config(**overrides)

    def test_tracking_rejects_single_step_waveform(self) -> None:
        """duration 只覆盖一个发送周期 → 离散波形不足 2 步，构造即拒绝。"""
        with pytest.raises(ValueError, match="波形") as excinfo:
            _tracking_config(duration_s=_PERIOD, send_period_s=_PERIOD)
        message = str(excinfo.value)
        assert "不足 2 步" in message
        assert "接管前拒绝" in message
        # 消息必须带实际 duration/period，现场可直接定位配置问题
        assert f"duration_s={_PERIOD!r}" in message
        assert f"send_period_s={_PERIOD!r}" in message

    def test_tracking_rejects_insufficient_excursion(self) -> None:
        """离散跨度低于 max(0.1·A, 1e-3) → 构造拒绝（接管前拒绝零运动）。"""
        with pytest.raises(ValueError, match="位移不足") as excinfo:
            _tracking_config(
                duration_s=0.01,  # 5 个发送周期，但波形尚未离开起点
                frequency_hz=1.0,
                amplitude_rad=0.02,
            )
        message = str(excinfo.value)
        assert "span=" in message
        assert "阈值" in message
        assert "steps=5" in message

    def test_tracking_aliasing_rejected(self) -> None:
        """f=500 Hz 与 2ms 发送周期同相：每个采样都在相同相位 → 跨度 0 拒绝。"""
        with pytest.raises(ValueError, match="位移不足") as excinfo:
            _tracking_config(
                duration_s=0.006,  # steps=3
                frequency_hz=500.0,  # 2π·f·period = 2π → 采样恒在相位 0
                amplitude_rad=0.02,
            )
        message = str(excinfo.value)
        assert "steps=3" in message
        assert "span=0" in message

    def test_tracking_minimal_discrete_waveform_accepted(self) -> None:
        """最小合法离散波形：2 步、跨度 0.01 rad（阈值 max(0.1·A,1e-3)=0.001）。"""
        config = _tracking_config(
            duration_s=0.004, frequency_hz=250.0, amplitude_rad=0.01
        )
        assert config.tracking_step_count() == 2
        assert config.expected_target_span_rad() == pytest.approx(0.01, abs=1e-12)

    @pytest.mark.parametrize(
        "duration_s,expected_steps",
        [(0.005, 2), (0.007, 4)],
        ids=["round-half-even-2.5-to-2", "round-half-even-3.5-to-4"],
    )
    def test_tracking_step_count_uses_python_round(
        self, duration_s: float, expected_steps: int
    ) -> None:
        """步数取整沿用 Python round（5/2→2、7/2→4），与 run_tracking 同口径。"""
        config = _tracking_config(
            duration_s=duration_s, frequency_hz=250.0, amplitude_rad=0.01
        )
        assert config.tracking_step_count() == expected_steps

    def test_tracking_helpers_reject_read_only(self) -> None:
        """read_only 下调用 tracking 专用方法 → 明确 ValueError（不静默返回）。"""
        config = _read_only_config()
        for method in (
            config.tracking_step_count,
            config.expected_target_span_rad,
        ):
            with pytest.raises(ValueError, match="tracking"):
                method()


# ---------------------------------------------------------------------------
# 指标纯函数（手工可验小样本）
# ---------------------------------------------------------------------------


class TestMetricsPureFunctions:
    """指标计算数值正确性：固定小样本手算对照。"""

    def test_publish_interval_metrics_handcrafted(self) -> None:
        """间隔 [0.1, 0.15, 0.05]，期限 0.12 → 分位数手算一致，零超期。

        超期判据 = 间隔 > 1.5×期限 = 0.18（复审 2026-11-18）：0.15 是
        "1.25×期限"的轻微超出（发布开销量级），不计超期。
        """
        metrics = compute_publish_interval_metrics(
            [0.0, 0.1, 0.25, 0.3], deadline_s=0.12
        )
        # 手算：intervals = [0.1, 0.15, 0.05]；sorted = [0.05, 0.1, 0.15]
        assert metrics["n_intervals"] == 3
        assert metrics["p50_s"] == pytest.approx(0.1)
        assert metrics["p95_s"] == pytest.approx(0.145)  # 0.1+0.9*(0.15-0.1)
        assert metrics["p99_s"] == pytest.approx(0.149)  # 0.1+0.98*(0.15-0.1)
        assert metrics["max_s"] == pytest.approx(0.15)
        assert metrics["deadline_overruns"] == 0  # 0.15 ≤ 1.5×0.12

    def test_publish_overhead_not_counted_as_overrun(self) -> None:
        """复审复现（2026-11-18）：2 ms 周期 + 20 µs 发布开销 → 零超期。

        驱动按"上一帧 publish 返回后再等一个完整周期"排下一拍，间隔 =
        周期 + 发布/唤醒开销，恒略大于周期。旧判据（> 期限）把每次正常
        发布都记超期；新判据（> 1.5×期限）不得误报。
        """
        period = 0.002
        overhead = 20e-6  # 20 µs 模拟发送耗时
        ats = [i * (period + overhead) for i in range(4)]
        metrics = compute_publish_interval_metrics(ats, deadline_s=period)
        assert metrics["n_intervals"] == 3
        assert metrics["max_s"] == pytest.approx(period + overhead)
        assert metrics["deadline_overruns"] == 0

    @pytest.mark.parametrize(
        ("gap", "expected"),
        [
            (0.002, 0),  # 恰等期限：按时
            (0.002 + 5e-4, 0),  # 1.25×期限：发布开销量级
            (0.003, 0),  # 1.5×期限：严格 > 才计，边界不计
            (0.0031, 1),  # 1.55×期限：错过半拍以上
            (0.004, 1),  # 2×期限：漏一拍
        ],
    )
    def test_publish_overrun_boundary(self, gap: float, expected: int) -> None:
        """超期判据边界：> 1.5×期限才计（半拍容差，不是漏拍不报）。"""
        metrics = compute_publish_interval_metrics(
            [0.0, gap], deadline_s=0.002
        )
        assert metrics["deadline_overruns"] == expected
        assert metrics["max_s"] == pytest.approx(gap)

    @pytest.mark.parametrize("ats", [[], [1.0]])
    def test_publish_interval_metrics_insufficient_unavailable(
        self, ats: list[float]
    ) -> None:
        """发布记录不足 2 条 → 指标 unavailable（None），不得报 0。"""
        metrics = compute_publish_interval_metrics(ats, deadline_s=0.12)
        assert metrics["unavailable"] is True
        assert metrics["p50_s"] is None
        assert metrics["max_s"] is None
        assert metrics["deadline_overruns"] is None

    def test_tracking_error_metrics_handcrafted(self) -> None:
        """两关节两配对样本：RMS/max 手算一致。"""
        pairs = [
            (np.array([0.0, 0.0]), np.array([0.01, 0.0])),
            (np.array([0.1, 0.0]), np.array([0.12, 0.0])),
        ]
        metrics = compute_tracking_error_metrics(pairs)
        assert metrics["n_matched"] == 2
        # 关节 0 误差 [0.01, 0.02]：rms=sqrt((1+4)e-4/2)、max=0.02
        assert metrics["per_joint"][0]["rms_rad"] == pytest.approx(0.01581138830)
        assert metrics["per_joint"][0]["max_rad"] == pytest.approx(0.02)
        # 关节 1 误差全 0（有配对样本时 0 是真实值，不是无样本的冒充）
        assert metrics["per_joint"][1]["rms_rad"] == pytest.approx(0.0)
        assert metrics["per_joint"][1]["max_rad"] == pytest.approx(0.0)

    def test_tracking_error_metrics_no_samples_unavailable(self) -> None:
        """无配对样本 → unavailable（None），不得报零误差"通过"。"""
        metrics = compute_tracking_error_metrics([])
        assert metrics["unavailable"] is True
        assert metrics["n_matched"] == 0
        assert all(item["rms_rad"] is None for item in metrics["per_joint"])


# ---------------------------------------------------------------------------
# read_only 模式
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 受控时间推进（诊断测试确定性调度；评审 2026-11-20）
# ---------------------------------------------------------------------------

_SCHED_STEP_S = 0.001        # 每次推进的最大虚拟时间步（s）
_SCHED_REAL_STEP_S = 0.001   # 每次推进后让出真实时间（≈1:1 实时，防线程饥饿）
_SCHED_IDLE_S = 0.0005       # 无等待者且无停止在途时的真实休眠（s）
_SCHED_YIELD_S = 0.0002      # 等待线程让出/轮询的真实休眠（s）
_SCHED_TIMEOUT_S = 120.0     # 调度线程真实存活上限（防御挂死）
_SCHED_PUBLISH_WAIT_S = 1.0  # 等待单次发布落库的最长真实时间（s）


class _ControlledTimeScheduler:
    """受控时间推进器：由测试唯一决定虚拟时钟前进的时机与幅度。

    与 ``_pump`` 的区别：``_pump`` 按真实墙钟推进虚拟时间、不等待发送
    线程完成本轮，桌面调度抖动会让发布间隔跨过 1.5 倍期限。本调度器在
    每次推进时先投递状态帧并唤醒驱动，推进后等待该时刻到期的发布真正
    落库才继续；诊断侧经 ``run_r1d_diagnostic(wait_until=...)`` 登记目标
    时刻，调度器不越过目标——命令、发布、反馈、采样四者按同一顺序串行
    发生，不再依赖线程调度。

    仅用于测试；生产路径仍走 ``time.sleep``（见
    ``_DiagnosticCollector._wait_until``）。
    """

    def __init__(
        self,
        clock: FakeClock,
        transport: FakeR1DTransport,
        arm: Any,
        *,
        step_s: float = _SCHED_STEP_S,
        real_step_s: float = _SCHED_REAL_STEP_S,
        fast_dq: bool = False,
        delay_after_active_publishes: int | None = None,
        delay_s: float = _PERIOD,
        period_s: float = _PERIOD,
    ) -> None:
        """初始化调度器（不启动线程，避免构造期竞态）。"""
        self._clock = clock
        self._transport = transport
        self._arm = arm
        self._step_s = float(step_s)
        self._real_step_s = float(real_step_s)
        self._fast_dq = bool(fast_dq)
        self._delay_after_active_publishes = delay_after_active_publishes
        self._delay_s = float(delay_s)
        self._period_s = float(period_s)
        self._delay_used = False
        self._cond = threading.Condition()
        self._waiter: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="controlled-time", daemon=True
        )

    # -- 生命周期 -------------------------------------------------------

    def start(self) -> None:
        """启动调度线程。"""
        self._thread.start()

    def stop(self) -> None:
        """停止调度线程（幂等；只置位并唤醒，不阻塞在调度器内部）。"""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        with self._arm._cond:
            self._arm._cond.notify_all()
        self._thread.join(timeout=2.0)

    # -- 诊断侧注入点 ---------------------------------------------------

    def wait_until(self, deadline: float) -> None:
        """登记目标时刻并阻塞，直到虚拟时钟到达 ``deadline``。"""
        deadline = float(deadline)
        with self._cond:
            self._waiter = deadline
            self._cond.notify_all()
            while self._clock() < deadline and not self._stop.is_set():
                self._cond.wait(0.05)
            if self._waiter == deadline:
                self._waiter = None

    # -- 调度线程 -------------------------------------------------------

    def _loop(self) -> None:
        """按等待者/停止在途两种情况推进虚拟时钟。"""
        end = time.monotonic() + _SCHED_TIMEOUT_S
        while not self._stop.is_set() and time.monotonic() < end:
            with self._cond:
                target = self._waiter
            if target is None and not self._stop_in_progress():
                # 无人等待：不推进（避免越过诊断尚未登记的节拍点）
                time.sleep(_SCHED_IDLE_S)
                continue
            now = self._clock()
            if target is not None and now >= target:
                # 已到达目标：唤醒等待者，等其登记下一个目标
                with self._cond:
                    self._cond.notify_all()
                time.sleep(_SCHED_YIELD_S)
                continue
            step = self._step_s
            if target is not None:
                step = min(step, target - now)
            self._advance_to(now + max(step, 1e-9))
            if self._inject_delay_after_publish():
                continue
            # 让出真实时间：诊断/驱动线程需在下一拍前获得锁并观察新时刻，
            # 否则锁竞争会让 50ms 确认窗口被虚拟时钟直接跨过。
            time.sleep(self._real_step_s)

    def _stop_in_progress(self) -> bool:
        """停止请求已登记但尚未回到 READ_ONLY：期间需继续推进时间。"""
        status = self._arm.get_status()
        return status.phase in ("ACTIVE", "STOPPING") and (
            self._arm._stop_request is not None or status.phase == "STOPPING"
        )

    def _inject_delay_after_publish(self) -> bool:
        """在指定次数的 active 发布落库后，把下一次发布整体推迟。

        触发点选在发布**落库之后**：此时下一次发布的期限严格等于
        ``last_published_at + period``。虚拟时钟直接跳到
        ``last_published_at + period + delay_s``——下一次命令与发布都在
        该时刻发生，发布间隔 = 周期 + ``delay_s``。确定性验证超期判据
        仍判失败（红线：不得放宽阈值消除红灯）。
        """
        if self._delay_after_active_publishes is None or self._delay_used:
            return False
        with self._arm._lock:
            active = [
                record
                for record in self._arm._publish_records
                if record.kind == "active" and record.success
            ]
        if len(active) < self._delay_after_active_publishes:
            return False
        jump = active[-1].published_at + self._period_s + self._delay_s
        if jump <= self._clock():
            return False
        self._delay_used = True
        self._advance_to(jump)
        return True

    def _advance_to(self, new_now: float) -> None:
        """推进到 ``new_now``：投帧 → 唤醒驱动 → 等发布落地 → 唤醒等待者。"""
        delta = float(new_now) - self._clock()
        if delta > 0.0:
            self._clock.advance(delta)
        if self._transport.on_state is not None:
            dq = _measured_dq() if self._fast_dq else None
            _feed(self._transport, self._clock, dq=dq)
        self._wake_arm()
        self._drain_publishes()
        with self._cond:
            self._cond.notify_all()

    def _wake_arm(self) -> None:
        """唤醒驱动调度线程（其 wait 带超时，幂等安全）。"""
        with self._arm._cond:
            self._arm._cond.notify_all()

    def _drain_publishes(self) -> None:
        """等待当前时刻到期的发布真正落库（发布记录在驱动锁内追加）。"""
        with self._arm._lock:
            pending = self._arm._pending_target is not None
            due = self._arm._next_publish_at
            records = len(self._arm._publish_records)
        inflight = self._transport.publish_count > records
        if not inflight and (not pending or due is None or due > self._clock()):
            return
        self._wake_arm()
        end = time.monotonic() + _SCHED_PUBLISH_WAIT_S
        while time.monotonic() < end:
            with self._arm._lock:
                landed = len(self._arm._publish_records) > records
                phase = self._arm._phase
            if landed:
                return
            if phase not in ("ACTIVE", "STOPPING"):
                return
            time.sleep(_SCHED_YIELD_S)


def _run_controlled(
    tmp_path: Path,
    clock: FakeClock,
    config: R1DDiagnosticConfig,
    *,
    scheduler_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], FakeR1DTransport, Any]:
    """在受控时间推进下跑一次诊断，返回 (摘要, fake 传输, 驱动)。"""
    arm, transport = _make_arm(
        tmp_path, clock, slot_policies=_full_explicit_policies()
    )
    scheduler = _ControlledTimeScheduler(
        clock, transport, arm, **(scheduler_kwargs or {})
    )
    scheduler.start()
    try:
        result = run_r1d_diagnostic(
            arm,
            config,
            tmp_path / "out",
            clock=clock,
            wait_until=scheduler.wait_until,
        )
    finally:
        scheduler.stop()
    return result, transport, arm


class TestReadOnlyMode:
    """只读诊断：不 activate、不 send——publish/acquire 全 0。"""

    def test_read_only_publishes_and_acquires_zero(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """全程 transport.publish/acquire 计数为 0，verdict=pass。"""
        arm, transport = _make_arm(tmp_path, clock)
        done = threading.Event()
        pump = threading.Thread(
            target=_pump, args=(clock, transport, done), daemon=True
        )
        pump.start()
        try:
            result = run_r1d_diagnostic(
                arm, _read_only_config(), tmp_path / "out"
            )
        finally:
            done.set()
            pump.join(timeout=2.0)
        # 只读不变式：publish/acquire 全 0（测试断言 fake 计数）
        assert transport.publish_count == 0
        assert transport.acquire_count == 0
        assert result["verdict"] == "pass", result["reasons"]
        assert result["counts"]["publish_records_total"] == 0
        # 无发布 → 间隔指标 unavailable（None），不得报 0
        assert result["publish_metrics"]["unavailable"] is True
        assert result["counts"]["samples_ok"] == result["counts"]["samples"]
        assert result["counts"]["samples"] >= 1
        # 清理：断开后通道关闭
        assert transport.close_count == 1
        assert arm.get_status().phase == "CLOSED"

    def test_read_only_insufficient_data_fails(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """无状态帧投递 → 全程采样失败 → verdict=fail（数据不足）。"""
        arm, transport = _make_arm(tmp_path, clock)
        result = run_r1d_diagnostic(
            arm, _read_only_config(duration_s=0.02), tmp_path / "out"
        )
        assert result["verdict"] == "fail"
        assert any("数据不足" in r for r in result["reasons"])
        assert result["counts"]["samples_ok"] == 0
        assert transport.close_count == 1  # 失败也清理资源

    def test_read_only_stream_interrupt_fails(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """状态流中途断流（停喂但时钟继续）→ fail（状态流中断）。"""
        arm, transport = _make_arm(tmp_path, clock)
        done = threading.Event()
        # 同步速率推进（advance=0.002）保证新鲜度超时按真实节奏触发；
        # 诊断窗口 0.2s（10 拍），0.03s 后停喂——后续采样 age 持续增长
        # 超过 state_timeout=0.05s，至少末尾数拍失败
        pump = threading.Thread(
            target=_pump,
            args=(clock, transport, done),
            kwargs={"stop_feed_after_s": 0.03, "advance_s": 0.002},
            daemon=True,
        )
        pump.start()
        try:
            result = run_r1d_diagnostic(
                arm, _read_only_config(duration_s=0.2), tmp_path / "out"
            )
        finally:
            done.set()
            pump.join(timeout=2.0)
        assert result["verdict"] == "fail"
        assert any("状态流中断" in r for r in result["reasons"])
        assert transport.close_count == 1

    def test_read_only_connect_failure_fails(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """传输层 open 失败 → fail（connect 失败）+ 资源清理。"""
        transport = FakeR1DTransport()
        transport.open_error = RuntimeError("simulated open failure")
        arm, transport = _make_arm(tmp_path, clock, transport)
        result = run_r1d_diagnostic(arm, _read_only_config(), tmp_path / "out")
        assert result["verdict"] == "fail"
        assert any("connect 失败" in r for r in result["reasons"])
        assert transport.close_count == 1  # disconnect 清理半开通道

    def test_read_only_records_full_non_target_state(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """采样含完整 35 槽非目标状态与接收时刻（full_state_available）。"""
        arm, transport = _make_arm(tmp_path, clock)
        done = threading.Event()
        pump = threading.Thread(
            target=_pump, args=(clock, transport, done), daemon=True
        )
        pump.start()
        try:
            result = run_r1d_diagnostic(
                arm, _read_only_config(duration_s=0.04), tmp_path / "out"
            )
        finally:
            done.set()
            pump.join(timeout=2.0)
        assert result["verdict"] == "pass", result["reasons"]
        assert result["provenance"]["full_state_available"] is True
        npz = np.load(tmp_path / "out" / "diagnostic_raw.npz")
        full_q = npz["sample_full_q"]
        assert full_q.shape[1] == 35
        # 完整帧哨兵值逐槽对齐（q[slot] = 0.1*slot - 2.0）
        valid_rows = ~np.isnan(full_q).all(axis=1)
        assert valid_rows.any()
        assert_allclose(full_q[valid_rows], np.tile(_measured_q(), (valid_rows.sum(), 1)))


# ---------------------------------------------------------------------------
# tracking 模式
# ---------------------------------------------------------------------------


class TestTrackingMode:
    """单关节低速跟踪：锚定→波形→回起始→停止确认。"""

    def test_tracking_full_run(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """完整 tracking：只动选中关节、其余保持、回起始+停止确认通过。"""
        # 受控时间推进：命令/发布/反馈/采样串行，排除桌面调度抖动
        result, transport, _ = _run_controlled(
            tmp_path, clock, _tracking_config()
        )

        assert result["verdict"] == "pass", result["reasons"]
        # 接管与发布都发生过（tracking 语义）
        assert transport.acquire_count == 1
        assert transport.publish_count >= 2
        # 停止流程：确认 + 退权 + 回 READ_ONLY
        assert result["stop"]["stop_confirmed"] is True
        assert result["stop"]["final_phase"] == "READ_ONLY"
        assert transport.release_count == 1
        assert result["stop"]["feedback_below_threshold_at_s"] is not None
        assert result["stop"]["latency_software_s"] is not None
        assert result["stop"]["latency_feedback_s"] is not None

        # —— 波形断言：发布帧中只有选中关节（模型序 0 → SDK 槽 15）在动 ——
        frames_q = np.stack([np.asarray(f.q) for f in transport.publish_frames])
        # 选中关节槽 15：有变化（波形幅值 > 0.5mm）且回到锚定值
        motion_span = float(frames_q[:, 15].max() - frames_q[:, 15].min())
        assert motion_span > 0.0005, f"选中关节未运动（span={motion_span}）"
        assert frames_q[:, 15].min() == pytest.approx(_LEFT_MEASURED_SDK[0], abs=1e-9)
        assert frames_q[:, 15].max() <= _LEFT_MEASURED_SDK[0] + _AMPLITUDE
        # 其余 6 个目标关节槽 16-21：全程恒等于锚定值（保持）
        for offset, anchored in enumerate(_LEFT_MEASURED_SDK[1:], start=16):
            assert float(frames_q[:, offset].max() - frames_q[:, offset].min()) == (
                pytest.approx(0.0, abs=1e-12)
            ), f"非选中关节槽 {offset} 发生了运动"
            assert frames_q[0, offset] == pytest.approx(anchored, abs=1e-9)
        # 非目标槽位（显式策略哨兵，如槽 0/31）：全程保持
        for slot in (0, 31):
            hold_q = _explicit_hold(slot).q
            assert np.allclose(frames_q[:, slot], hold_q, atol=1e-12)

        # —— 结束回到起始目标：send 日志最后一条目标 = 锚定模型序 ——
        npz = np.load(tmp_path / "out" / "diagnostic_raw.npz")
        send_targets = npz["send_target_model"]
        assert send_targets.shape[0] >= 2
        assert_allclose(send_targets[0], _Q_ANCHORED, atol=1e-12)
        assert_allclose(send_targets[-1], _Q_ANCHORED, atol=1e-12)

        # 发布间隔指标可得且不超期（fake 泵送下间隔 < 期限）
        metrics = result["publish_metrics"]
        assert metrics["unavailable"] is False
        assert metrics["deadline_overruns"] == 0
        # 跟踪误差指标可得（配对样本 > 0；无匹配样本数为 0）
        tracking = result["tracking_metrics"]
        assert tracking["unavailable"] is False
        assert tracking["n_matched"] >= 1
        assert tracking["no_match_samples"] == 0
        assert tracking["alignment_rule"]

    def test_tracking_activate_rejected_fails(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """无显式槽位策略 → activate 拒绝 → fail + 零发布 + 清理。"""
        arm, transport = _make_arm(tmp_path, clock)  # slot_policies=None
        done = threading.Event()
        pump = threading.Thread(
            target=_pump, args=(clock, transport, done), daemon=True
        )
        pump.start()
        try:
            result = run_r1d_diagnostic(
                arm, _tracking_config(), tmp_path / "out"
            )
        finally:
            done.set()
            pump.join(timeout=2.0)
        assert result["verdict"] == "fail"
        assert any("activate 拒绝" in r for r in result["reasons"])
        assert transport.publish_count == 0  # 未获控制权零发布
        assert transport.acquire_count == 0
        assert transport.close_count == 1

    def test_tracking_stop_unconfirmed_fails(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """停止确认窗口无低速反馈 → 停止未确认 → fail。"""
        arm, transport = _make_arm(
            tmp_path, clock, slot_policies=_full_explicit_policies()
        )
        done = threading.Event()
        # fast_dq：全程投递超阈值高速帧——activate/发布不受影响，
        # 停止反馈确认必然失败（确认窗口内无低速新帧）
        pump = threading.Thread(
            target=_pump,
            args=(clock, transport, done),
            kwargs={"fast_dq": True},
            daemon=True,
        )
        pump.start()
        try:
            result = run_r1d_diagnostic(
                arm, _tracking_config(), tmp_path / "out"
            )
        finally:
            done.set()
            pump.join(timeout=2.0)
        assert result["verdict"] == "fail"
        assert any("停止未确认" in r for r in result["reasons"])
        # 未确认停止不退权（不宣称已停）
        assert transport.release_count == 0
        assert result["stop"]["stop_confirmed"] is False

    def test_tracking_error_limit_exceeded_fails(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """跟踪误差超注入阈值 → fail（D-M8，评审 2026-11-17）。

        fake 泵送恒定锚定反馈、目标按升余弦偏离锚定 → 误差 ≈ 末段
        波形偏移 δ(t_end)=A/2·(1−cos(2π·f·t_end))（_DIAG_DURATION 仍短
        于半周期，δ_end≈0.0066 rad）；阈值设 0.0002（约 33× 裕度）必超限。
        判别力：若 _apply_threshold_verdicts 缺跟踪误差判定分支，
        verdict 将误为 pass。
        """
        arm, transport = _make_arm(
            tmp_path, clock, slot_policies=_full_explicit_policies()
        )
        done = threading.Event()
        pump = threading.Thread(
            target=_pump, args=(clock, transport, done), daemon=True
        )
        pump.start()
        try:
            result = run_r1d_diagnostic(
                arm,
                _tracking_config(tracking_error_limit_rad=0.0002),
                tmp_path / "out",
            )
        finally:
            done.set()
            pump.join(timeout=2.0)
        assert result["verdict"] == "fail"
        assert any("跟踪误差超阈值" in r for r in result["reasons"])
        # 指标本身仍如实可得（判定基于实测 max，不是无样本 unavailable）
        assert result["tracking_metrics"]["unavailable"] is False

    def test_stop_latency_limit_exceeded_fails(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """反馈侧停止延迟超注入阈值 → fail（D-M8，评审 2026-11-17）。

        阈值设 1e-6 s：实测反馈延迟含至少一个采样周期的量化
        （latency_feedback ≥ send_period 量级），必然超限。判别力：
        若 _apply_threshold_verdicts 缺停止延迟判定分支，verdict 将误
        为 pass。
        """
        arm, transport = _make_arm(
            tmp_path, clock, slot_policies=_full_explicit_policies()
        )
        done = threading.Event()
        pump = threading.Thread(
            target=_pump, args=(clock, transport, done), daemon=True
        )
        pump.start()
        try:
            result = run_r1d_diagnostic(
                arm,
                _tracking_config(stop_latency_limit_s=1e-6),
                tmp_path / "out",
            )
        finally:
            done.set()
            pump.join(timeout=2.0)
        assert result["verdict"] == "fail"
        assert any("停止延迟超阈值" in r for r in result["reasons"])
        # 判定引用的是反馈侧实测口径（D-M2），遥测含口径注明
        assert result["stop"]["latency_feedback_s"] is not None
        assert result["stop"]["latency_feedback_scope"]


class TestTrackingMotionEvidence:
    """P1-3（2026-11-18）：tracking 验收须同时具备发布间隔与实际运动证据。

    判定层单测直接注入与 ``_consume_publishes`` 同构的发布记录，只验证
    ``_apply_threshold_verdicts`` 的新增两项证据判据——不依赖 fake 驱动
    发送节拍，避免与 timing flake 混淆。
    """

    def test_tracking_verdict_fails_when_publish_unavailable(
        self, tmp_path: Path
    ) -> None:
        """仅一条成功 active 发布 → 发布间隔证据不足 → fail（不得判通过）。"""
        result = _finalized_verdict(
            _tracking_config(),
            tmp_path / "out",
            [
                _publish_row(
                    seq=0,
                    published_at=1.0,
                    q_sdk=_sdk_target(joint=0, value=0.0),
                )
            ],
        )
        assert result["verdict"] == "fail"
        assert any("发布间隔证据不足" in reason for reason in result["reasons"])
        assert result["publish_metrics"]["unavailable"] is True

    def test_tracking_verdict_fails_when_published_targets_do_not_move(
        self, tmp_path: Path
    ) -> None:
        """三条成功 active 目标恒定 → 实际发布跨度 0 → fail（证据非空）。"""
        constant = _sdk_target(joint=0, value=0.3)
        publishes = [
            _publish_row(seq=i, published_at=1.0 + i * _PERIOD, q_sdk=constant)
            for i in range(3)
        ]
        result = _finalized_verdict(_tracking_config(), tmp_path / "out", publishes)
        assert result["verdict"] == "fail"
        reasons = "\n".join(result["reasons"])
        assert "实际发布的 SDK 侧目标（按模型序排列）未覆盖波形" in reasons
        assert "actual=0.000000" in reasons
        assert "expected=" in reasons
        assert result["publish_metrics"]["unavailable"] is False

    @pytest.mark.parametrize(
        "ratio,expect_span_failure",
        [(0.5, False), (0.499, True)],
        ids=["exactly-half-passes-span-item", "just-below-half-fails-span-item"],
    )
    def test_tracking_published_target_span_threshold(
        self, tmp_path: Path, ratio: float, expect_span_failure: bool
    ) -> None:
        """实际跨度门槛：= 0.5×expected 通过该项，0.499×expected 失败。

        本用例只注入发布记录、没有配对反馈样本（不伪造 samples），因此
        整体 verdict 仍因既有"无配对样本"判据 fail——**该项通过 ≠ 验收
        通过**；这里断言的是新增跨度子项的判定语义。
        """
        config = _tracking_config()
        span = ratio * config.expected_target_span_rad()
        publishes = [
            _publish_row(
                seq=0, published_at=1.0, q_sdk=_sdk_target(joint=0, value=0.0)
            ),
            _publish_row(
                seq=1,
                published_at=1.0 + _PERIOD,
                q_sdk=_sdk_target(joint=0, value=span),
            ),
        ]
        result = _finalized_verdict(config, tmp_path / "out", publishes)
        reasons = "\n".join(result["reasons"])
        span_failure = any("未覆盖波形" in reason for reason in result["reasons"])
        assert span_failure is expect_span_failure
        # 两条有效目标都计入，故不是"证据不足"分支
        assert not any("实际发布目标证据不足" in reason for reason in result["reasons"])
        # 无配对样本 → 整体 verdict 必 fail；跨度子项通过不等于验收通过。
        assert result["verdict"] == "fail"
        assert "无配对样本" in reasons
        if not expect_span_failure:
            assert "未覆盖波形" not in reasons

    def test_tracking_published_targets_require_two_valid_records(
        self, tmp_path: Path
    ) -> None:
        """失败/stop/缺目标/NaN 记录不凑数：仅 1 条有效目标 → 证据不足 fail。"""
        publishes = [
            _publish_row(
                seq=0, published_at=1.0, q_sdk=_sdk_target(joint=0, value=0.0)
            ),
            _publish_row(
                seq=1,
                published_at=1.002,
                q_sdk=_sdk_target(joint=0, value=9.0),
                success=False,
            ),
            _publish_row(
                seq=2,
                published_at=1.004,
                q_sdk=_sdk_target(joint=0, value=9.0),
                kind="stop",
            ),
            _publish_row(seq=3, published_at=1.002, q_sdk=None),
            _publish_row(
                seq=4,
                published_at=1.004,
                q_sdk=_sdk_target(joint=0, value=float("nan")),
            ),
        ]
        result = _finalized_verdict(_tracking_config(), tmp_path / "out", publishes)
        reasons = "\n".join(result["reasons"])
        assert result["verdict"] == "fail"
        assert "实际发布目标证据不足" in reasons
        assert "仅 1 条" in reasons
        # 成功 active 记录已有 3 条（间隔证据可得），失败与 NaN 只影响目标证据
        assert result["publish_metrics"]["unavailable"] is False

    def test_tracking_span_uses_only_selected_joint(self, tmp_path: Path) -> None:
        """selected_joint=3 时其它关节运动不算覆盖波形 → fail。"""
        config = _tracking_config(selected_joint=3)
        expected = config.expected_target_span_rad()
        start = _sdk_target(joint=3, value=0.0)
        moved = list(start)
        moved[0] = expected  # 非选中关节大幅运动，选中关节保持不动
        publishes = [
            _publish_row(seq=0, published_at=1.0, q_sdk=start),
            _publish_row(seq=1, published_at=1.0 + _PERIOD, q_sdk=tuple(moved)),
        ]
        result = _finalized_verdict(config, tmp_path / "out", publishes)
        assert any("未覆盖波形" in reason for reason in result["reasons"])

    def test_tracking_span_ignores_sign_and_offset(self, tmp_path: Path) -> None:
        """sign/offset 只做换算：关节 1（sign=-1）SDK 目标跨半幅即通过跨度项。"""
        config = _tracking_config(selected_joint=1)
        half = 0.5 * config.expected_target_span_rad()
        start = _sdk_target(joint=1, value=0.0)
        moved = list(start)
        moved[1] = half
        publishes = [
            _publish_row(seq=0, published_at=1.0, q_sdk=start),
            _publish_row(seq=1, published_at=1.0 + _PERIOD, q_sdk=tuple(moved)),
        ]
        result = _finalized_verdict(config, tmp_path / "out", publishes)
        reasons = "\n".join(result["reasons"])
        assert "未覆盖波形" not in reasons
        assert "实际发布目标证据不足" not in reasons

    def test_tracking_failure_still_writes_artifacts(self, tmp_path: Path) -> None:
        """verdict=fail 时 JSON/CSV/NPZ 仍全部落盘（先判定后写产物）。"""
        import json

        out = tmp_path / "out"
        constant = _sdk_target(joint=0, value=0.0)
        publishes = [
            _publish_row(seq=i, published_at=1.0 + i * _PERIOD, q_sdk=constant)
            for i in range(3)
        ]
        result = _finalized_verdict(_tracking_config(), out, publishes)
        assert result["verdict"] == "fail"
        assert (out / "diagnostic_summary.json").is_file()
        summary = json.loads(
            (out / "diagnostic_summary.json").read_text(encoding="utf-8")
        )
        assert summary["verdict"] == "fail"
        for name in (
            "diagnostic_publish_records.csv",
            "diagnostic_samples.csv",
            "diagnostic_send_log.csv",
            "diagnostic_raw.npz",
        ):
            assert (out / name).is_file(), f"缺产物 {name}"
        npz = np.load(out / "diagnostic_raw.npz")
        assert npz["pub_q_sdk_model_order"].shape == (3, 7)


# ---------------------------------------------------------------------------
# 失败路径（中断/异常）
# ---------------------------------------------------------------------------


class TestFailurePaths:
    """Ctrl+C/异常路径：失败结论 + 资源清理。"""

    def test_keyboard_interrupt_aborts_and_cleans(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """诊断循环中 Ctrl+C → verdict=aborted + disconnect 清理。

        注入点选 publish_records：仅诊断主线程调用（发送线程只 append
        deque），避免实例属性遮蔽影响后台线程。
        """
        arm, transport = _make_arm(tmp_path, clock)
        done = threading.Event()
        pump = threading.Thread(
            target=_pump, args=(clock, transport, done), daemon=True
        )
        pump.start()
        original = arm.publish_records
        calls = {"n": 0}

        def _interrupting():
            calls["n"] += 1
            if calls["n"] >= 2:
                raise KeyboardInterrupt
            return original()

        arm.publish_records = _interrupting  # type: ignore[method-assign]
        try:
            result = run_r1d_diagnostic(
                arm, _read_only_config(duration_s=0.06), tmp_path / "out"
            )
        finally:
            done.set()
            pump.join(timeout=2.0)
            arm.publish_records = original  # 恢复供后续断言使用
        assert result["verdict"] == "aborted"
        assert any("用户中断" in r for r in result["reasons"])
        assert transport.close_count == 1
        assert arm.get_status().phase == "CLOSED"
        # 中止也写产物（失败结论可追溯）
        summary = tmp_path / "out" / "diagnostic_summary.json"
        assert summary.is_file()

    def test_unexpected_exception_fails_and_cleans(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """诊断循环中未预期异常 → verdict=fail + 清理。

        注：get_arm_state 级别的异常会被采样层记录为样本失败（流中断
        语义）；本测试注入点为 publish_records（仅主线程调用），验证
        未被子层包裹的顶层异常路径生成失败结论并清理。
        """
        arm, transport = _make_arm(tmp_path, clock)
        done = threading.Event()
        pump = threading.Thread(
            target=_pump, args=(clock, transport, done), daemon=True
        )
        pump.start()
        original = arm.publish_records
        calls = {"n": 0}

        def _exploding():
            calls["n"] += 1
            if calls["n"] >= 2:
                raise RuntimeError("simulated publish_records crash")
            return original()

        arm.publish_records = _exploding  # type: ignore[method-assign]
        try:
            result = run_r1d_diagnostic(
                arm, _read_only_config(duration_s=0.06), tmp_path / "out"
            )
        finally:
            done.set()
            pump.join(timeout=2.0)
            arm.publish_records = original
        assert result["verdict"] == "fail"
        assert any("诊断过程异常" in r for r in result["reasons"])
        assert transport.close_count == 1


# ---------------------------------------------------------------------------
# 产物 schema
# ---------------------------------------------------------------------------


class TestArtifacts:
    """JSON 摘要 + CSV/NPZ 原始序列的字段完备性。"""

    @pytest.fixture
    def read_only_result(
        self, tmp_path: Path, clock: FakeClock
    ) -> Iterator[dict[str, Any]]:
        """跑一次通过的 read_only 诊断，产出结果与产物目录。"""
        arm, transport = _make_arm(tmp_path, clock)
        done = threading.Event()
        pump = threading.Thread(
            target=_pump, args=(clock, transport, done), daemon=True
        )
        pump.start()
        try:
            result = run_r1d_diagnostic(
                arm, _read_only_config(), tmp_path / "out"
            )
        finally:
            done.set()
            pump.join(timeout=2.0)
        assert result["verdict"] == "pass", result["reasons"]
        yield result

    def test_json_schema_complete(
        self, tmp_path: Path, read_only_result: dict[str, Any]
    ) -> None:
        """JSON 摘要字段覆盖计划清单（溯源/会话/计数/指标/停止/产物）。"""
        result = read_only_result
        assert result["schema"] == f"r1d_diagnostic_v{R1D_DIAGNOSTIC_SCHEMA_VERSION}"
        assert result["verdict"] == "pass"
        assert result["mode"] == "read_only"
        # 配置回显
        assert result["config"]["duration_s"] == _RO_DURATION
        assert result["config"]["sample_period_s"] == _SAMPLE_PERIOD
        # 溯源：SDK 版本（R4）/代码 SHA/硬件配置来源/侧别/单位
        provenance = result["provenance"]
        assert provenance["sdk_version"] == R1D_SDK_IDENTITY
        assert provenance["code_sha"] and len(provenance["code_sha"]) == 40
        assert provenance["hardware_config"]["path"] == "tests://hw_left.yaml"
        assert provenance["hardware_config"]["sha256"]
        assert provenance["side"] == "left"
        units = provenance["units"]
        assert units["position"] == "rad" and units["velocity"] == "rad/s"
        assert provenance["shared_clock"] is True
        # 会话/计数/指标/停止/产物
        assert result["session"]["status_events"]
        # E-M8（评审 2026-11-17）：收紧为唯一期望——final_phase 取最后
        # 一个状态事件，cleanup（disconnect）不记录事件，read_only 正常
        # 通过路径唯一终相是 READ_ONLY；出现其他值即状态机泄漏。
        assert result["session"]["final_phase"] == "READ_ONLY"
        assert set(result["counts"]) >= {
            "samples", "samples_ok", "send_calls", "send_accepted",
            "publish_records_total", "publish_active_success", "publish_stop",
        }
        assert result["publish_metrics"]["unavailable"] is True
        assert result["tracking_metrics"] is None  # read_only 无跟踪指标
        assert result["stop"]["requested"] is False
        artifacts = result["artifacts"]
        for key in ("publish_csv", "samples_csv", "send_csv", "npz"):
            assert (tmp_path / "out" / artifacts[key]).is_file()

    def test_csv_columns(
        self, tmp_path: Path, read_only_result: dict[str, Any]
    ) -> None:
        """CSV 表头覆盖：发布（R9 时刻）/采样（完整 35 槽）/send（接受时刻）。"""
        import csv as csv_mod

        out = tmp_path / "out"
        with (out / "diagnostic_publish_records.csv").open(
            encoding="utf-8"
        ) as handle:
            publish_header = next(csv_mod.reader(handle))
        assert publish_header[:7] == [
            "seq", "session_id", "kind", "published_at", "mode_machine",
            "success", "error",
        ]
        assert "q_sdk_6" in publish_header

        with (out / "diagnostic_samples.csv").open(encoding="utf-8") as handle:
            samples_header = next(csv_mod.reader(handle))
        for column in (
            "index", "t", "received_at", "source_time", "sequence",
            "q_model_6", "dq_model_6", "full_q_34", "full_dq_34",
            "model_error", "full_error",
        ):
            assert column in samples_header, f"采样 CSV 缺列 {column}"

        with (out / "diagnostic_send_log.csv").open(encoding="utf-8") as handle:
            send_header = next(csv_mod.reader(handle))
        assert send_header[:4] == ["step", "accepted_at", "code", "code_name"]
        assert "target_model_6" in send_header

    def test_npz_keys(
        self, tmp_path: Path, read_only_result: dict[str, Any]
    ) -> None:
        """NPZ 覆盖发布/采样/发送三序列（None 哨兵规则见 JSON）。"""
        npz = np.load(tmp_path / "out" / "diagnostic_raw.npz")
        expected = {
            "send_step", "send_accepted_at", "send_code", "send_target_model",
            "sample_index", "sample_t", "sample_q_model", "sample_dq_model",
            "sample_full_q", "sample_full_dq", "sample_received_at",
            "sample_source_time", "sample_sequence",
            "pub_seq", "pub_session_id", "pub_kind_code", "pub_published_at",
            "pub_mode_machine", "pub_success", "pub_q_sdk_model_order",
        }
        assert expected <= set(npz.files)
        assert npz["sample_full_q"].shape[1] == 35
        assert npz["sample_q_model"].shape[1] == 7


# ---------------------------------------------------------------------------
# 评审 2026-11-18 修复轮（复审 844f4c1）：R4 阈值缺失不得 pass /
# R5 反馈配对用同帧接收时刻 / R6 非整周期受限速回程
# ---------------------------------------------------------------------------


class TestAcceptanceThresholds:
    """R4（P2）：tracking 验收阈值缺失不得输出 pass（仅采集 ≠ 验收通过）。"""

    @pytest.mark.parametrize(
        "missing_fields",
        [
            ["tracking_error_limit_rad"],
            ["stop_latency_limit_s"],
            ["tracking_error_limit_rad", "stop_latency_limit_s"],
        ],
    )
    def test_missing_thresholds_fail_not_pass(
        self, tmp_path: Path, clock: FakeClock, missing_fields: list[str]
    ) -> None:
        """缺任一/全部阈值：verdict=fail，原因点名缺失字段与"仅采集"。"""
        overrides = {field: None for field in missing_fields}
        arm, transport = _make_arm(
            tmp_path, clock, slot_policies=_full_explicit_policies()
        )
        done = threading.Event()
        pump = threading.Thread(
            target=_pump, args=(clock, transport, done), daemon=True
        )
        pump.start()
        try:
            result = run_r1d_diagnostic(
                arm, _tracking_config(**overrides), tmp_path / "out"
            )
        finally:
            done.set()
            pump.join(timeout=2.0)
        assert result["verdict"] == "fail", result["reasons"]
        reasons = "\n".join(result["reasons"])
        for field in missing_fields:
            assert field in reasons, f"失败原因未点名缺失阈值 {field}"
        assert "仅完成数据采集" in reasons and "验收通过" in reasons

    def test_thresholds_present_within_limit_pass(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """阈值完整且实际值未超限：pass（判定通路真实接通，非恒 fail）。"""
        # 受控时间推进：不再用真实墙钟泵送（该用例曾因发布间隔 4ms
        # > 1.5×2ms 在桌面调度下间歇失败）
        result, _, _ = _run_controlled(tmp_path, clock, _tracking_config())
        assert result["verdict"] == "pass", result["reasons"]
        assert result["config"]["tracking_error_limit_rad"] == 1.0
        assert result["config"]["stop_latency_limit_s"] == 10.0

    def test_controlled_delay_injection_still_fails_on_overrun(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """确定性注入 2ms 延迟 → 间隔 4ms > 3ms，必须判超期失败。

        红线（评审 2026-11-20）：稳定化只消除测试自身调度不确定性，不放
        宽产品判据——同样的 4ms 间隔在受控时钟下依然判 fail，证明红灯
        不是被"跳过/重试/放宽阈值"消除的。
        """
        result, _, _ = _run_controlled(
            tmp_path,
            clock,
            _tracking_config(),
            scheduler_kwargs={
                "delay_after_active_publishes": 4,
                "delay_s": _PERIOD,
            },
        )
        assert result["publish_metrics"]["deadline_overruns"] >= 1
        assert result["verdict"] == "fail", result["reasons"]
        assert any("超期" in reason for reason in result["reasons"])


class TestFeedbackPairing:
    """R5（P2）：配对键 = 同帧接收时刻；排除发布前缓存与重复反馈。"""

    def test_pre_publish_cache_excluded(self) -> None:
        """发布前到达的旧缓存（received_at ≤ published_at）不计反馈。

        评审复现：published_at=10.0、样本 t=10.01、received_at=9.99
        在旧规则（按读取完成时刻）计为有效匹配；新规则必须 no_match。
        """
        from src.real.r1d_diagnostics import _pair_feedback_frames

        target = np.zeros(7)
        feedback = np.full(7, 0.5)
        pairs, no_match = _pair_feedback_frames(
            [(10.0, target)],
            [(9.99, 1, feedback)],  # 发布前缓存：接收时刻早于发布
        )
        assert pairs == []
        assert no_match == 1

    def test_received_after_publish_matches(self) -> None:
        """接收时刻严格晚于发布时刻：最早未消费帧配对为反馈。"""
        from src.real.r1d_diagnostics import _pair_feedback_frames

        target = np.zeros(7)
        early = np.full(7, 0.1)
        late = np.full(7, 0.2)
        pairs, no_match = _pair_feedback_frames(
            [(10.0, target)],
            [(10.01, 1, early), (10.02, 2, late)],
        )
        assert no_match == 0
        assert len(pairs) == 1
        assert np.array_equal(pairs[0][1], early)  # 最早的发布后帧

    def test_duplicate_feedback_matched_once(self) -> None:
        """同一反馈帧只作一条命令的证据（配对即消费）。"""
        from src.real.r1d_diagnostics import _pair_feedback_frames

        frame = np.full(7, 0.3)
        target = np.zeros(7)
        pairs, no_match = _pair_feedback_frames(
            [(10.0, target), (10.002, target)],
            [(10.004, 1, frame)],  # 唯一反馈帧晚于两条发布
        )
        # 仅第一条命令获得该帧；第二条无可用新反馈
        assert len(pairs) == 1
        assert no_match == 1

    def test_samples_without_frame_time_never_match(self) -> None:
        """无可靠帧时间的样本不参与配对（降级证据，不伪装已对齐）。"""
        from src.real.r1d_diagnostics import _pair_feedback_frames

        pairs, no_match = _pair_feedback_frames(
            [(10.0, np.zeros(7))],
            [(None, None, np.full(7, 0.1))],
        )
        assert pairs == []
        assert no_match == 1

    def test_missing_target_counts_no_match(self) -> None:
        """目标缺失（q_sdk 提取失败）的发布行计无匹配。"""
        from src.real.r1d_diagnostics import _pair_feedback_frames

        pairs, no_match = _pair_feedback_frames(
            [(10.0, None)],
            [(10.01, 1, np.full(7, 0.1))],
        )
        assert pairs == []
        assert no_match == 1

    def test_same_frame_sampled_twice_counts_once(self) -> None:
        """同一物理帧被采样两次仍只作一条命令的证据（复审 2026-11-18）。

        复现：采样周期快于状态流时，连续两次采样读到同一缓存帧
        （received_at 与 sequence 均相同）。旧实现按样本行逐条配对，会把
        这一帧计为两条命令都已匹配，夸大反馈覆盖率；新实现按帧身份
        (received_at, sequence) 去重后，第二条命令应无匹配。
        """
        from src.real.r1d_diagnostics import _pair_feedback_frames

        frame = np.full(7, 0.3)
        target = np.zeros(7)
        pairs, no_match = _pair_feedback_frames(
            [(10.0, target), (10.001, target)],
            [
                (10.004, 7, frame),  # 第一次采样到帧 seq=7
                (10.004, 7, frame.copy()),  # 第二次采样：同一物理帧
            ],
        )
        assert len(pairs) == 1  # 只作第一条命令的证据
        assert no_match == 1  # 第二条命令没有新的反馈帧

    def test_different_frames_same_receive_time_both_match(self) -> None:
        """接收时刻相同但 sequence 不同的两帧是不同物理帧，各作证一次。"""
        from src.real.r1d_diagnostics import _pair_feedback_frames

        first = np.full(7, 0.1)
        second = np.full(7, 0.2)
        target = np.zeros(7)
        pairs, no_match = _pair_feedback_frames(
            [(10.0, target), (10.001, target)],
            [(10.004, 7, first), (10.004, 8, second)],
        )
        assert no_match == 0
        assert len(pairs) == 2
        assert np.array_equal(pairs[0][1], first)
        assert np.array_equal(pairs[1][1], second)
    """R6（P2）：非整周期波形受限速回程，不再一拍跳回起点。"""

    @pytest.mark.parametrize("duration_s", [0.5, 1.0])
    def test_non_integer_period_rate_limited_return(
        self, tmp_path: Path, clock: FakeClock, duration_s: float
    ) -> None:
        """半周期/整周期：全部 active 发布步长合法并回到起点后正常停止。"""
        # 受控时间推进：发布间隔严格等于周期，步长断言不受调度抖动干扰
        result, transport, arm = _run_controlled(
            tmp_path, clock, _tracking_config(duration_s=duration_s)
        )
        assert result["verdict"] == "pass", result["reasons"]
        # 全部 active 发布帧（按发布顺序）槽 15 的相邻步长 ≤ dq_limit·周期
        # （选中关节=模型序 0 → 槽 15；dq_limit=2.0 rad/s 为测试值最小者）
        active_frames = [
            frame
            for frame, record in zip(
                transport.publish_frames,
                [r for r in arm.publish_records() if r.success],
            )
            if record.kind == "active"
        ]
        assert len(active_frames) >= 2
        steps = np.abs(
            np.diff(np.asarray([float(f.q[15]) for f in active_frames]))
        )
        max_step = float(np.min(_TEST_DQ_LIMIT)) * _PERIOD
        assert float(np.max(steps)) <= max_step + 1e-9, (
            f"发布步长 {float(np.max(steps)):.6f} 超过允许 {max_step:.6f}"
        )
        # 末拍目标回到锚定起点（回程收敛）
        npz = np.load(tmp_path / "out" / "diagnostic_raw.npz")
        send_targets = npz["send_target_model"]
        assert_allclose(send_targets[-1], _Q_ANCHORED, atol=1e-12)
        # 停止流程完整（未因回程中断）
        assert result["stop"]["stop_confirmed"] is True
        assert result["stop"]["final_phase"] == "READ_ONLY"
