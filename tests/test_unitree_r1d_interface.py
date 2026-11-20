"""R1-A7-D 驱动生命周期/单一发送器/看门狗测试（阶段 4 Task 4.6）。

所有 sign/offset/增益/限值/阈值/超时均为**测试专用有限参数，非实机参数**：
仅用于验证驱动状态机、发送线程与看门狗逻辑，不代表任何实物标定或
已核验数值。全机非目标槽位策略沿用 4.3 测试的全量显式哨兵策略。

并发测试使用 Barrier/Event + fake clock，不依赖长 sleep；
等待统一走 eventually 轮询（真实秒上限 2s，间隔 2ms）。
"""

from __future__ import annotations

import os
import subprocess
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
from src.real.r1d_state_buffer import R1DStateError
from src.real.r1d_types import R1DStateFrame
from src.real.unitree_r1d_interface import (
    R1DInterfaceError,
    R1DPublishRecord,
    R1D_PUBLISH_RECORD_CAPACITY,
    UnitreeR1DInterface,
)
from src.real.robot_arm_protocol import RobotArmInterface
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
_COMMAND_TIMEOUT = 0.2  # s（目标过期看门狗，测试专用）
_STOP_DQ_THRESHOLD = 0.05  # rad/s（停止确认速度阈值，测试专用）
_MODE_MACHINE = 7  # 测试专用机型回读值
_JOIN_TIMEOUT = 2.0  # s（发送线程 join 上限，测试专用）

# 实测姿态哨兵：q[slot] = 0.1*slot - 2.0（35 槽唯一，测试专用）
# 左臂槽 15-21 实测 SDK = (-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1)
_LEFT_MEASURED_SDK = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1)
# 锚定实测姿态的模型序 q_desired（手算：q_model = sign*(q_sdk - offset)）
_Q_ANCHORED = np.array((-0.5, 0.45, -0.2, 0.4, 0.2, 0.4, 0.1))
# 第二目标：锚定值 + 0.002（单关节步长上限 min=0.004，不超限）
_Q_NEXT = _Q_ANCHORED + 0.002

# 显式策略哨兵槽位（manifest null 组，与 4.3 测试一致）；对侧臂/头
# 另用 manifest 示例数值哨兵（见 _manifest_example_hold）——R5 修订后
# 全部 28 个非目标槽位均须显式覆盖
_EXPLICIT_SLOTS = tuple(range(0, 15)) + (31, 32, 33, 34)


# ---------------------------------------------------------------------------
# 测试辅助（fake clock / fake transport / 配置构造 / eventually）
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
    """测试专用 fake 传输层（记录全部调用，支持受控失败与发布阻塞）。"""

    def __init__(self) -> None:
        self.open_count = 0
        self.acquire_count = 0
        self.publish_count = 0
        # 已进入 publish 调用（含被闸门阻塞中）的次数：测试用它确认
        # 发送线程已实际到达 transport.publish 调用点
        self.publish_entered = 0
        self.release_count = 0
        self.close_count = 0
        # 退权闸门（R3 测试，评审 2026-11-18）：未 set 时 release_control
        # 阻塞——模拟"确认后退权执行期间"（MotionSwitcher RPC 慢）的
        # 可控窗口；release_entered 确认发送线程已到达退权调用点
        self.release_entered = 0
        self.release_gate: threading.Event | None = None
        self.publish_frames: list[Any] = []
        self.on_state: Callable[[R1DStateFrame], None] | None = None
        self.open_error: Exception | None = None
        self.acquire_error: Exception | None = None
        self.release_error: Exception | None = None
        self.publish_error: Exception | None = None
        # 发布闸门：未 set 时所有 publish 阻塞（模拟慢 DDS 写/线程滞留）
        self.publish_gate: threading.Event | None = None
        self.publish_hook: Callable[[], None] | None = None
        # 接管闸门：未 set 时 acquire 阻塞（模拟慢 MotionSwitcher RPC）
        self.acquire_gate: threading.Event | None = None
        self.acquire_entered = 0  # 已进入 acquire 调用（含阻塞中）的次数
        # 已投递状态帧数（E-M6：并发测试用它做谓词等待，替代裸 sleep）
        self.deliver_count = 0

    def open(self, on_state: Callable[[R1DStateFrame], None]) -> None:
        if self.open_error is not None:
            raise self.open_error
        self.open_count += 1
        self.on_state = on_state

    def acquire_control(self) -> None:
        self.acquire_entered += 1
        if self.acquire_gate is not None:
            self.acquire_gate.wait()
        if self.acquire_error is not None:
            raise self.acquire_error
        self.acquire_count += 1

    def publish(self, command: Any) -> None:
        self.publish_entered += 1
        if self.publish_gate is not None:
            self.publish_gate.wait()
        if self.publish_hook is not None:
            self.publish_hook()
        if self.publish_error is not None:
            raise self.publish_error
        self.publish_count += 1
        self.publish_frames.append(command)

    def release_control(self) -> None:
        self.release_entered += 1
        if self.release_gate is not None:
            self.release_gate.wait()
        if self.release_error is not None:
            raise self.release_error
        self.release_count += 1

    def close(self) -> None:
        self.close_count += 1

    def deliver(self, frame: R1DStateFrame) -> None:
        """测试注入状态帧（模拟 lowstate 回调）。"""
        assert self.on_state is not None, "尚未 open"
        self.deliver_count += 1
        self.on_state(frame)


def eventually(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    """真实时钟有界轮询等待（短间隔，不依赖长 sleep）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def _measured_q() -> np.ndarray:
    """35 槽唯一哨兵实测姿态（q[slot] = 0.1*slot - 2.0）。"""
    return np.array([0.1 * slot - 2.0 for slot in range(35)])


def _measured_dq() -> np.ndarray:
    """35 槽非零速度哨兵（dq[slot] = 0.01*(slot+1)）。"""
    return np.array([0.01 * (slot + 1) for slot in range(35)])


def _state_frame(
    clock: FakeClock,
    *,
    mode_machine: int = _MODE_MACHINE,
    dq: np.ndarray | None = None,
    q: np.ndarray | None = None,
    session_id: int = 1,
) -> R1DStateFrame:
    """构造以 fake clock 当前时刻接收的整机状态帧（测试专用）。"""
    return R1DStateFrame(
        q=q if q is not None else _measured_q(),
        dq=dq if dq is not None else np.zeros(35, dtype=np.float64),
        sequence=None,
        source_time=None,
        received_at=clock(),
        mode_machine=mode_machine,
        session_id=session_id,
    )


def _feed(
    transport: FakeR1DTransport,
    clock: FakeClock,
    *,
    mode_machine: int = _MODE_MACHINE,
    dq: np.ndarray | None = None,
    q: np.ndarray | None = None,
) -> None:
    """向驱动投递一帧以当前 fake 时钟计时的状态帧。"""
    transport.deliver(_state_frame(clock, mode_machine=mode_machine, dq=dq, q=q))


def _pump(
    transport: FakeR1DTransport,
    clock: FakeClock,
    predicate: Callable[[], bool],
    *,
    timeout: float = 2.0,
    mode_machine: int = _MODE_MACHINE,
) -> bool:
    """泵送等待：循环推进 fake 时钟并投递新鲜帧，直到谓词成立或超时。

    冻结 fake 时钟下，确认窗口/期限等待不会自行到期——由测试侧推进
    时钟并喂帧驱动状态机前进（不依赖长 sleep，每轮 2ms）。
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        clock.advance(0.001)
        _feed(transport, clock, mode_machine=mode_machine)
        time.sleep(0.002)
    return predicate()


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


# 对侧臂（左臂驱动的对侧=右臂 22-28）与头（29/30）的显式哨兵策略：
# 数值取 manifest [公开示例]（q=0 + 示例增益）。R5 修订（评审 I-05）后
# 这些槽位同样必须显式给出——不再运行时回退 manifest 示例
_OTHER_ARM_SLOTS = (22, 23, 24, 25, 26, 27, 28)
_HEAD_SLOTS = (29, 30)
_OTHER_ARM_HOLD_KP = (100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0)
_HEAD_HOLD_KP = (50.0, 10.0)
_HEAD_HOLD_KD = (2.0, 0.1)


def _manifest_example_hold(slot: int) -> R1DSlotHold:
    """对侧臂/头槽显式哨兵（数值=manifest 示例，非运行时回退）。"""
    if slot in _HEAD_SLOTS:
        offset = _HEAD_SLOTS.index(slot)
        return R1DSlotHold(
            mode=1, q=0.0, dq=0.0,
            kp=_HEAD_HOLD_KP[offset], kd=_HEAD_HOLD_KD[offset], tau=0.0,
        )
    return R1DSlotHold(
        mode=1, q=0.0, dq=0.0,
        kp=_OTHER_ARM_HOLD_KP[slot - 22], kd=2.0, tau=0.0,
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


_UNSET = object()


def _make_arm(
    tmp_path: Path,
    clock: FakeClock,
    transport: FakeR1DTransport | None = None,
    *,
    slot_policies: Any = _UNSET,
    expected_mode_machine: int | None = None,
    command_timeout_s: float | None = None,
    stop_sequence_frames: int = 1,
    stop_confirm_dq_threshold: float | None = None,
    join_timeout_s: float = _JOIN_TIMEOUT,
    config_mutate: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[UnitreeR1DInterface, FakeR1DTransport]:
    """构造被测驱动（默认注入全量显式槽位策略，测试专用）。"""
    from src.real.r1d_config import load_r1d_hardware_config

    data = _base_config_dict()
    if config_mutate is not None:
        config_mutate(data)
    path = tmp_path / "hw_left.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    config = load_r1d_hardware_config(path)
    profile = load_robot_profile(_LEFT_PROFILE_PATH)
    fake = transport if transport is not None else FakeR1DTransport()
    arm = UnitreeR1DInterface(
        config,
        profile,
        fake,
        clock=clock,
        slot_policies=(
            _full_explicit_policies() if slot_policies is _UNSET else slot_policies
        ),
        expected_mode_machine=expected_mode_machine,
        command_timeout_s=command_timeout_s,
        stop_sequence_frames=stop_sequence_frames,
        stop_confirm_dq_threshold=stop_confirm_dq_threshold,
        join_timeout_s=join_timeout_s,
    )
    return arm, fake


def _connect_and_activate(
    arm: UnitreeR1DInterface, transport: FakeR1DTransport, clock: FakeClock
) -> None:
    """连接并接管到 ACTIVE（投递一帧新鲜状态后 activate）。"""
    assert arm.connect() is True
    _feed(transport, clock)
    arm.activate()
    assert arm.get_status().phase == "ACTIVE"


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def armed_full(
    tmp_path: Path, clock: FakeClock
) -> Iterator[tuple[UnitreeR1DInterface, FakeR1DTransport, FakeClock]]:
    """已 ACTIVE 的驱动三件套（arm/transport/clock），用后清理。"""
    arm, transport = _make_arm(
        tmp_path,
        clock,
        stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD,
        command_timeout_s=_COMMAND_TIMEOUT,
    )
    _connect_and_activate(arm, transport, clock)
    try:
        yield arm, transport, clock
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass


def _stop_records(arm: UnitreeR1DInterface) -> list[R1DPublishRecord]:
    return [r for r in arm.publish_records() if r.kind == "stop" and r.success]


def _active_records(arm: UnitreeR1DInterface) -> list[R1DPublishRecord]:
    return [r for r in arm.publish_records() if r.kind == "active"]


def _run_slow_stop_confirmed(
    arm: UnitreeR1DInterface, transport: FakeR1DTransport, clock: FakeClock
) -> None:
    """主线程侧泵送缓停完成：推进时钟 + 投递低速新帧（冻结时钟下期限与确认均需推进）。"""
    done = threading.Event()
    holder: dict[str, BaseException | None] = {"error": None}

    def _stop() -> None:
        try:
            arm.slow_stop()
        except BaseException as exc:  # pragma: no cover - 记录意外异常
            holder["error"] = exc
        done.set()

    threading.Thread(target=_stop, daemon=True).start()
    # 泵送：持续小幅推进时钟并喂帧，驱动停止序列期限与确认窗口前进
    deadline = time.monotonic() + _JOIN_TIMEOUT
    while time.monotonic() < deadline and not done.is_set():
        clock.advance(0.001)
        _feed(transport, clock)  # 发布后到达的低速新帧（dq=0）
        time.sleep(0.002)
    assert done.wait(0.1)
    assert holder["error"] is None


# ---------------------------------------------------------------------------
# 模块纯净性与构造
# ---------------------------------------------------------------------------


def test_module_import_is_pure() -> None:
    """import 驱动模块不得加载 SDK/MuJoCo、不得启动线程（子进程独立验证）。

    I-09 修复（评审 2026-11-17）：原进程内断言命中 sys.modules 缓存，
    不执行模块体，属空操作；参照 test_r1d_transport.py 先例改子进程。
    """
    code = (
        "import sys\n"
        "import threading\n"
        "import src.real.unitree_r1d_interface\n"
        "leaked = [m for m in sys.modules if m == 'unitree_sdk2py'\n"
        "          or m.startswith('unitree_sdk2py.') or m == 'cyclonedds'\n"
        "          or m.startswith('cyclonedds.') or m == 'mujoco'\n"
        "          or m.startswith('mujoco.')]\n"
        "assert not leaked, leaked\n"
        "assert threading.active_count() == 1, threading.enumerate()\n"
        "print('PURE_OK')\n"
    )
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert "PURE_OK" in result.stdout


def test_constructor_does_not_touch_transport(tmp_path: Path, clock: FakeClock) -> None:
    """构造零副作用：不 open/acquire/publish/release/close，不启动线程。"""
    before_threads = threading.active_count()
    arm, transport = _make_arm(tmp_path, clock)
    assert transport.open_count == 0
    assert transport.acquire_count == 0
    assert transport.publish_count == 0
    assert transport.release_count == 0
    assert transport.close_count == 0
    assert threading.active_count() == before_threads
    status = arm.get_status()
    assert status.phase == "CLOSED"
    assert status.session_id == 0
    assert status.stop_confirmed is False
    assert status.fault_reason is None
    assert status.last_publish_at is None
    assert arm.publish_records() == ()


def test_constructor_records_disabled_capabilities(
    tmp_path: Path, clock: FakeClock
) -> None:
    """未注入命令超时/停止确认阈值时：能力禁用并显式记录（R11）。"""
    arm, _ = _make_arm(tmp_path, clock)  # 默认不注入 command_timeout/阈值
    notes = "\n".join(arm.capability_notes)
    assert "命令超时" in notes and "禁用" in notes
    assert "停止确认" in notes and "禁用" in notes
    # 注入后不再记录对应禁用项
    arm2, _ = _make_arm(
        tmp_path,
        clock,
        command_timeout_s=_COMMAND_TIMEOUT,
        stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD,
    )
    notes2 = "\n".join(arm2.capability_notes)
    assert "命令超时" not in notes2
    assert "停止确认" not in notes2


def test_command_timeout_lower_bound_enforced(
    tmp_path: Path, clock: FakeClock
) -> None:
    """Minor 3：命令超时 < 2×控制周期属配置矛盾，构造即拒绝。"""
    # 0.002 < 2×period(0.004)：拒绝
    with pytest.raises(ValueError, match="2 倍控制周期"):
        _make_arm(tmp_path, clock, command_timeout_s=_PERIOD)
    # 恰为 2×周期：允许（边界）
    arm, _ = _make_arm(tmp_path, clock, command_timeout_s=2.0 * _PERIOD)
    assert arm.get_status().phase == "CLOSED"


def test_publish_records_bounded(tmp_path: Path, clock: FakeClock) -> None:
    """Minor 2：发布记录有界（deque maxlen=公开容量常量），记录语义不变。"""
    # 容量常量覆盖典型诊断窗口：500 Hz × ≥200 s（远超 diagnostic_duration）
    assert isinstance(R1D_PUBLISH_RECORD_CAPACITY, int)
    assert R1D_PUBLISH_RECORD_CAPACITY >= 100_000
    arm, _ = _make_arm(tmp_path, clock)
    # 内部容器确为有界 deque 且容量等于公开常量（容量生效性验证，测试窥视
    # 私有字段仅此一处：公开 API 无法观察 maxlen）
    internal = arm._publish_records  # noqa: SLF001
    assert internal.maxlen == R1D_PUBLISH_RECORD_CAPACITY
    assert arm.publish_records() == ()


def test_implements_robot_arm_interface_protocol(
    tmp_path: Path, clock: FakeClock
) -> None:
    """驱动结构性实现 RobotArmInterface（runtime_checkable）。"""
    arm, _ = _make_arm(tmp_path, clock)
    assert isinstance(arm, RobotArmInterface)


# ---------------------------------------------------------------------------
# 状态机：connect / activate / stop / fault / disconnect
# ---------------------------------------------------------------------------


def test_connect_transitions_to_read_only(tmp_path: Path, clock: FakeClock) -> None:
    """CLOSED→connect→READ_ONLY；重复 connect 幂等返回 True。"""
    arm, transport = _make_arm(tmp_path, clock)
    assert arm.connect() is True
    assert arm.get_status().phase == "READ_ONLY"
    assert transport.open_count == 1
    assert arm.connect() is True  # 幂等
    assert transport.open_count == 1
    arm.disconnect()


def test_connect_never_commands_motion(tmp_path: Path, clock: FakeClock) -> None:
    """connect 只订阅：绝不 acquire/publish/release（具名复选框项）。"""
    arm, transport = _make_arm(tmp_path, clock)
    assert arm.connect() is True
    _feed(transport, clock)
    _feed(transport, clock)
    assert transport.acquire_count == 0
    assert transport.publish_count == 0
    assert transport.release_count == 0
    assert arm.get_status().last_publish_at is None
    arm.disconnect()


def test_connect_failure_returns_false(tmp_path: Path, clock: FakeClock) -> None:
    """传输层 open 异常：connect 返回 False（不吞成 True），phase 仍 CLOSED。"""
    transport = FakeR1DTransport()
    transport.open_error = RuntimeError("网卡不存在")
    arm, _ = _make_arm(tmp_path, clock, transport)
    assert arm.connect() is False
    assert arm.get_status().phase == "CLOSED"
    arm.disconnect()


def test_connect_refused_without_read_only_config(
    tmp_path: Path, clock: FakeClock
) -> None:
    """read_only 门槛未满足（网卡 null）：connect 返回 False 且不触传输层。"""

    def mutate(data: dict[str, Any]) -> None:
        data["communication"]["network_interface"] = None

    arm, transport = _make_arm(tmp_path, clock, config_mutate=mutate)
    assert arm.connect() is False
    assert transport.open_count == 0
    assert arm.get_status().phase == "CLOSED"


def test_connect_refused_without_state_timeout(
    tmp_path: Path, clock: FakeClock
) -> None:
    """状态超时未配置（timing.state_timeout_s null）：状态缓存能力禁用（R11），connect 拒绝。"""

    def mutate(data: dict[str, Any]) -> None:
        data["timing"]["state_timeout_s"] = None

    arm, transport = _make_arm(tmp_path, clock, config_mutate=mutate)
    assert "状态超时" in "\n".join(arm.capability_notes)
    assert arm.connect() is False
    assert transport.open_count == 0


def test_activate_before_connect_raises(tmp_path: Path, clock: FakeClock) -> None:
    """CLOSED 下 activate 抛显式错误。"""
    arm, _ = _make_arm(tmp_path, clock)
    with pytest.raises(R1DInterfaceError, match="connect"):
        arm.activate()


def test_activate_success_flow(tmp_path: Path, clock: FakeClock) -> None:
    """READ_ONLY→activate（新鲜状态+机型+参数+停止能力预检）→ACTIVE，不发布任何命令。"""
    arm, transport = _make_arm(tmp_path, clock)
    assert arm.connect() is True
    _feed(transport, clock)
    arm.activate()
    status = arm.get_status()
    assert status.phase == "ACTIVE"
    assert transport.acquire_count == 1
    assert transport.publish_count == 0  # 接管本身不发命令
    arm.disconnect()


def test_activate_refused_without_fresh_state(
    tmp_path: Path, clock: FakeClock
) -> None:
    """无新鲜状态：activate 拒绝（R1DStateError 语义上浮），不接管。"""
    arm, transport = _make_arm(tmp_path, clock)
    assert arm.connect() is True
    # 不投递任何状态帧
    with pytest.raises(R1DInterfaceError, match="新鲜"):
        arm.activate()
    assert arm.get_status().phase == "READ_ONLY"
    assert transport.acquire_count == 0
    arm.disconnect()


def test_activate_refused_with_stale_state(
    tmp_path: Path, clock: FakeClock
) -> None:
    """状态过期后 activate 拒绝。"""
    arm, transport = _make_arm(tmp_path, clock)
    assert arm.connect() is True
    _feed(transport, clock)
    clock.advance(_TIMEOUT + 0.01)
    with pytest.raises(R1DInterfaceError, match="新鲜|过期"):
        arm.activate()
    assert transport.acquire_count == 0
    arm.disconnect()


def test_activate_refused_without_motion_config(
    tmp_path: Path, clock: FakeClock
) -> None:
    """motion 门槛未满足（限值 provisional）：activate 拒绝，不接管。"""

    def mutate(data: dict[str, Any]) -> None:
        data["limits"]["verified"] = False

    arm, transport = _make_arm(tmp_path, clock, config_mutate=mutate)
    assert arm.connect() is True  # read_only 只需通信+布局
    _feed(transport, clock)
    with pytest.raises(R1DInterfaceError, match="motion"):
        arm.activate()
    assert arm.get_status().phase == "READ_ONLY"
    assert transport.acquire_count == 0
    arm.disconnect()


def test_activate_refused_without_slot_policies(
    tmp_path: Path, clock: FakeClock
) -> None:
    """未注入全机槽位策略：停止能力预检失败，拒绝接管（G4 前不动）。"""
    arm, transport = _make_arm(tmp_path, clock, slot_policies=None)
    assert "非目标槽位" in "\n".join(arm.capability_notes)
    assert arm.connect() is True
    _feed(transport, clock)
    with pytest.raises(R1DInterfaceError, match="停止能力预检|停止帧"):
        arm.activate()
    assert transport.acquire_count == 0
    arm.disconnect()


def test_activate_refused_on_mode_machine_mismatch(
    tmp_path: Path, clock: FakeClock
) -> None:
    """冻结期望机型与回读不符：activate 拒绝（机型校验）。"""
    arm, transport = _make_arm(tmp_path, clock, expected_mode_machine=9)
    assert arm.connect() is True
    _feed(transport, clock)  # 回读 mode_machine=7 ≠ 9
    with pytest.raises(R1DInterfaceError, match="机型|新鲜"):
        arm.activate()
    assert transport.acquire_count == 0
    arm.disconnect()


def _control_error(message: str) -> Exception:
    """构造 4.5 异常体系的控制权错误实例。"""
    from src.real.r1d_transport import R1DControlError

    return R1DControlError(message)


def test_activate_failure_rolls_back_no_lowcmd(
    tmp_path: Path, clock: FakeClock
) -> None:
    """接管失败回滚：不发布任何 lowcmd + 报告未获权（manifest failure_rollback）。"""
    transport = FakeR1DTransport()
    transport.acquire_error = _control_error("机载运控持续占用")
    arm, _ = _make_arm(tmp_path, clock, transport)
    assert arm.connect() is True
    _feed(transport, clock)
    with pytest.raises(R1DInterfaceError, match="未获控制权"):
        arm.activate()
    # 回滚：未发布任何 lowcmd；本场景无他人持有控制权 → release 执行（幂等清理）
    assert transport.publish_count == 0
    assert transport.release_count >= 1
    assert arm.get_status().phase == "READ_ONLY"
    assert arm.get_status().fault_reason is None
    arm.disconnect()


def test_concurrent_activate_single_owner_no_release(
    tmp_path: Path, clock: FakeClock
) -> None:
    """I-1：并发双 activate 恰好一个成功；失败方绝不释放先到者控制权。"""
    arm, transport = _make_arm(tmp_path, clock)
    assert arm.connect() is True
    _feed(transport, clock)
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def worker(tag: str) -> None:
        barrier.wait()
        try:
            arm.activate()
            outcomes.append(f"{tag}:ok")
        except R1DInterfaceError as error:
            outcomes.append(f"{tag}:err({str(error)[:20]})")

    threads = [
        threading.Thread(target=worker, args=(tag,), daemon=True)
        for tag in ("A", "B")
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(_JOIN_TIMEOUT)
        assert not thread.is_alive()
    # 恰好一个成功；失败方为显式生命周期错误（非竞态崩溃）
    ok = [o for o in outcomes if o.endswith(":ok")]
    assert len(ok) == 1, outcomes
    assert any(":err(" in o for o in outcomes)
    # 终态一致：ACTIVE、零退权、零发布（接管本身不发命令）
    status = arm.get_status()
    assert status.phase == "ACTIVE"
    assert transport.release_count == 0
    assert transport.publish_count == 0
    arm.disconnect()


def test_reactivate_while_active_rejected_without_touching_control(
    tmp_path: Path, clock: FakeClock
) -> None:
    """I-1 顺序路径：ACTIVE 中重复 activate 在 phase 门槛即拒绝，不触碰控制权。"""
    arm, transport = _make_arm(tmp_path, clock)
    _connect_and_activate(arm, transport, clock)
    acquire_before = transport.acquire_count
    with pytest.raises(R1DInterfaceError, match="已处于 ACTIVE"):
        arm.activate()
    assert transport.acquire_count == acquire_before  # 未重复 acquire
    assert transport.release_count == 0  # 未释放既有控制权
    assert arm.get_status().phase == "ACTIVE"
    arm.disconnect()


def test_activate_race_stale_skips_release(
    tmp_path: Path, clock: FakeClock
) -> None:
    """I-1 stale 路径：接管 RPC 在途时 disconnect——activate 竞态失败且不 release。"""
    arm, transport = _make_arm(tmp_path, clock)
    assert arm.connect() is True
    _feed(transport, clock)
    acquire_gate = threading.Event()
    transport.acquire_gate = acquire_gate  # 使 acquire 阻塞（慢 RPC）
    holder: dict[str, BaseException | None] = {"error": None}

    def worker() -> None:
        try:
            arm.activate()
        except BaseException as exc:
            holder["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert eventually(lambda: transport.acquire_entered >= 1)  # activate 在 RPC 内
    arm.disconnect()  # 世代失效 + 通道关闭
    acquire_gate.set()  # 放行：acquire 返回后 activate 进 stale 竞态分支
    thread.join(_JOIN_TIMEOUT)
    assert not thread.is_alive()
    # 竞态失败为显式错误；不得 release（通道已由 close 清理）
    assert isinstance(holder["error"], R1DInterfaceError)
    assert "竞态" in str(holder["error"])
    assert transport.release_count == 0
    assert arm.get_status().phase == "CLOSED"


def test_slow_stop_confirmed_returns_to_read_only_full(
    armed_full: tuple[UnitreeR1DInterface, FakeR1DTransport, FakeClock],
) -> None:
    """正常缓停（带 clock 的完整版）：确认→退权→READ_ONLY。"""
    arm, transport, clock = armed_full
    _run_slow_stop_confirmed(arm, transport, clock)
    status = arm.get_status()
    assert status.phase == "READ_ONLY"
    assert status.stop_confirmed is True
    assert transport.release_count == 1
    assert len(_stop_records(arm)) == 1  # 默认有限停止序列 1 帧


def test_slow_stop_without_threshold_unconfirmed_fault(
    tmp_path: Path, clock: FakeClock
) -> None:
    """未注入停止确认阈值：只能发布有限停止序列，stop_confirmed=False + 故障锁存（不退权）。"""
    arm, transport = _make_arm(tmp_path, clock)  # 无阈值
    _connect_and_activate(arm, transport, clock)
    arm.slow_stop()
    status = arm.get_status()
    assert status.phase == "FAULT"
    assert status.stop_confirmed is False
    assert status.fault_reason is not None and "停止未确认" in status.fault_reason
    assert transport.release_count == 0  # 未确认不退权
    assert len(_stop_records(arm)) == 1  # 有限序列仍已发布
    arm.disconnect()


def test_slow_stop_without_fresh_state_unconfirmed(
    tmp_path: Path, clock: FakeClock
) -> None:
    """停止时无新鲜状态（build_stop 无法构造）：故障锁存 + stop_confirmed=False。"""
    arm, transport = _make_arm(tmp_path, clock)
    _connect_and_activate(arm, transport, clock)
    clock.advance(_TIMEOUT + 0.01)  # 状态过期
    arm.slow_stop()
    status = arm.get_status()
    assert status.phase == "FAULT"
    assert status.stop_confirmed is False
    assert "无法构造停止帧" in (status.fault_reason or "")
    assert len(_stop_records(arm)) == 0  # 未发布任何停止帧
    arm.disconnect()


def test_release_failure_after_confirmed_stop_latches_fault(
    tmp_path: Path, clock: FakeClock
) -> None:
    """确认后退权失败：FAULT + stop_confirmed=True + 明确原因。"""
    transport = FakeR1DTransport()
    transport.release_error = RuntimeError("publisher Close 失败")
    arm, _ = _make_arm(
        tmp_path, clock, transport, stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD
    )
    _connect_and_activate(arm, transport, clock)
    _run_slow_stop_confirmed(arm, transport, clock)
    status = arm.get_status()
    assert status.phase == "FAULT"
    assert status.stop_confirmed is True
    assert "释放控制权失败" in (status.fault_reason or "")


def test_emergency_stop_interrupts_pending_target_and_latches(
    tmp_path: Path, clock: FakeClock
) -> None:
    """急停打断排队目标：未被拾取的目标永不发布；终态 FAULT，原对象不得再 activate。"""
    arm, transport = _make_arm(
        tmp_path, clock, stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD
    )
    _connect_and_activate(arm, transport, clock)
    gate = threading.Event()
    transport.publish_gate = gate  # 阻塞发布，制造排队窗口
    assert arm.send_joint_command(_Q_ANCHORED) == 0
    # 发送线程已实际进入第一帧 publish 调用（阻塞中）；此时再排队第二个目标
    assert eventually(lambda: transport.publish_entered >= 1)
    queued = _Q_NEXT.copy()
    assert arm.send_joint_command(queued) == 0
    stop_done = threading.Event()
    threading.Thread(
        target=lambda: (arm.emergency_stop(), stop_done.set()), daemon=True
    ).start()
    # 同步点：驱动内部停止完成事件被 clear 与"急停请求登记+排队目标清除"
    # 在同一锁内原子发生——观察到 cleared 即可确定排队目标已清除。
    # （请求登记无公开观察点，故测试窥视该私有事件；发送线程仍阻塞在
    # publish(A) 内部，不存在推进竞态。）
    # C-M6 声明（评审 2026-11-17）：不为测试观测增设公开只读接口——
    # "停止请求是否已登记"是瞬时中间态，公开它会诱导调用方依赖内部
    # 时序；其稳定语义已由 get_status 的 phase/stop_confirmed 覆盖，
    # 窥私属性断言沿用 4.6 评审先例（最小暴露面）。
    assert eventually(lambda: not arm._stop_done.is_set())
    gate.set()  # 放行：第一帧发布完成，发送线程回到决策循环 → 先处理停止请求
    assert _pump(transport, clock, lambda: stop_done.is_set())
    status = arm.get_status()
    assert status.phase == "FAULT"
    # 排队目标 queued 永不发布：所有已发布 active 帧的目标槽数值不含其换算值
    queued_sdk = np.asarray(_TEST_SIGNS) * queued + np.asarray(_TEST_OFFSETS)
    for frame in transport.publish_frames:
        assert not np.allclose(np.asarray(frame.q[15:22]), queued_sdk)
    # 原对象不得再次 activate
    with pytest.raises(R1DInterfaceError, match="不得再次 activate|FAULT"):
        arm.activate()


def test_duplicate_slow_stop_single_sequence(
    armed_full: tuple[UnitreeR1DInterface, FakeR1DTransport, FakeClock],
) -> None:
    """重复缓停幂等：仅执行一次有限停止序列。"""
    arm, transport, clock = armed_full
    stop_count = len(_stop_records(arm))
    _run_slow_stop_confirmed(arm, transport, clock)
    arm.slow_stop()  # READ_ONLY 下重复缓停：no-op
    arm.slow_stop()
    assert len(_stop_records(arm)) == stop_count + 1


def test_duplicate_slow_stop_during_stopping_single_sequence(
    tmp_path: Path, clock: FakeClock
) -> None:
    """STOPPING 进行中重复缓停：不产生第二段停止序列。"""
    arm, transport = _make_arm(
        tmp_path,
        clock,
        stop_sequence_frames=3,
        stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD,
    )
    _connect_and_activate(arm, transport, clock)
    stopper_done = threading.Event()
    threading.Thread(
        target=lambda: (arm.slow_stop(), stopper_done.set()), daemon=True
    ).start()
    assert eventually(lambda: len(_stop_records(arm)) >= 1)
    arm.slow_stop()  # 序列进行中重复调用（阻塞等待在途序列完成）
    # 泵送谓词直接观察终态（测试本地事件会在等待超时返回时误置位）
    assert _pump(
        transport, clock, lambda: arm.get_status().phase in ("READ_ONLY", "FAULT")
    )
    assert stopper_done.wait(_JOIN_TIMEOUT)
    assert arm.get_status().phase == "READ_ONLY"
    assert len(_stop_records(arm)) == 3  # 恰好一段 3 帧序列


def test_duplicate_slow_stop_in_registration_window_no_false_fault(
    tmp_path: Path, clock: FakeClock
) -> None:
    """I-01（评审 2026-11-17）：真实注册窗口内的重复缓停不得误锁 FAULT。

    窗口 = _decide 取走停止请求之后、_run_stop_sequence 主体执行之前。
    实例级 patch 在 _run_stop_sequence 入口同步，使第二次 slow_stop()
    恰好落在窗口内发起。修复前：第二次调用见 phase 仍 ACTIVE → 重新
    登记请求 + clear 完成事件，第一段完成后第二段在 READ_ONLY 下存活
    复查失败 → FAULT / stop_confirmed=False（真实已停却报未确认）；
    修复后：_decide 取走请求的同一锁内已占住 STOPPING，第二次调用走
    幂等分支等待在途完成 → READ_ONLY / stop_confirmed=True。
    """
    arm, transport = _make_arm(
        tmp_path, clock, stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD
    )
    _connect_and_activate(arm, transport, clock)
    original = arm._run_stop_sequence
    entered = threading.Event()
    proceed = threading.Event()

    def _synced_stop_sequence(generation: int, *, trigger: str, final_fault: bool) -> None:
        entered.set()
        assert proceed.wait(_JOIN_TIMEOUT)
        original(generation, trigger=trigger, final_fault=final_fault)

    arm._run_stop_sequence = _synced_stop_sequence  # type: ignore[method-assign]

    errors: list[BaseException] = []
    first_done = threading.Event()
    second_done = threading.Event()

    def _stop_once(done: threading.Event) -> None:
        try:
            arm.slow_stop()
        except BaseException as exc:  # 记录意外异常
            errors.append(exc)
        finally:
            done.set()

    threading.Thread(target=_stop_once, args=(first_done,), daemon=True).start()
    assert eventually(lambda: entered.is_set())
    # 真实窗口内发起第二次缓停（_decide 已取走请求、序列主体未执行）
    threading.Thread(target=_stop_once, args=(second_done,), daemon=True).start()
    # E-M6：登记完成无公开可观察谓词，保留固定短等待（锁操作微秒级，
    # 0.05s 千倍裕度；改小会引入慢机抖动，改条件变量需暴露驱动内部状态）
    time.sleep(0.05)
    proceed.set()
    # 泵送第一段停止序列与确认窗口完成（冻结时钟需推进）
    deadline = time.monotonic() + _JOIN_TIMEOUT
    while time.monotonic() < deadline and not (
        first_done.is_set() and second_done.is_set()
    ):
        clock.advance(0.001)
        _feed(transport, clock)
        time.sleep(0.002)
    assert first_done.wait(0.1)
    assert second_done.wait(0.1)
    assert errors == []
    # 终态稳定确认：若竞态残留（第二段停止被重新登记），发送线程会很快
    # 在 READ_ONLY 下复查失败转入 FAULT
    settle = time.monotonic() + 0.3
    while time.monotonic() < settle:
        if arm.get_status().phase == "FAULT":
            break
        time.sleep(0.005)
    status = arm.get_status()
    assert status.phase == "READ_ONLY", f"误锁 FAULT: {status.fault_reason}"
    assert status.stop_confirmed is True
    arm.disconnect()


def test_stale_async_fault_does_not_leak_into_next_session(
    tmp_path: Path, clock: FakeClock
) -> None:
    """I-03（评审 2026-11-17）：未消费异步故障原因不跨会话泄漏。

    会话 1 的 ACTIVE 期间登记控制权丢失疑似后立即 disconnect（发送
    线程未及消费）——修复前 _async_fault_reason 残留，重连后新发送
    线程在 READ_ONLY 消费它 → 停止序列存活复查失败 → FAULT、
    activate 被"故障锁存"拒绝；修复后 disconnect 清除该字段，新会话
    干净起步（READ_ONLY 稳定、activate 正常）。
    """
    arm, transport = _make_arm(tmp_path, clock)
    assert arm.connect() is True
    _feed(transport, clock)
    arm.activate()
    assert arm.get_status().phase == "ACTIVE"
    # 模拟回调侧已登记、发送线程尚未消费的异步故障，随后立即断开
    arm._async_fault_reason = "控制权丢失疑似（测试注入）"  # noqa: SLF001
    arm.disconnect()
    # 修复点：异步事件随会话终结清除（窥私有属性断言，同套件既有先例）
    assert arm._async_fault_reason is None  # noqa: SLF001
    # 重连：旧事件不得把新会话推入 FAULT。E-M6：观察窗用谓词轮询替代
    # 裸 sleep——泄漏发生时 FAULT 会立即被捕获（失败信息带原因），而非
    # 等满窗口后查一次。
    assert arm.connect() is True
    fault_seen_at: list[str] = []
    deadline = time.monotonic() + 0.1
    while time.monotonic() < deadline:
        status_now = arm.get_status()
        if status_now.phase == "FAULT":
            fault_seen_at.append(str(status_now.fault_reason))
            break
        time.sleep(0.002)
    assert fault_seen_at == [], f"跨会话泄漏触发 FAULT: {fault_seen_at}"
    status = arm.get_status()
    assert status.phase == "READ_ONLY"
    # 新会话 activate 正常（无故障锁存）
    _feed(transport, clock)
    arm.activate()
    assert arm.get_status().phase == "ACTIVE"
    arm.disconnect()


def test_sender_thread_crash_in_read_only_latches_fault(
    tmp_path: Path, clock: FakeClock
) -> None:
    """C-M2（评审 2026-11-17）：READ_ONLY 会话发送线程崩溃不得静默。

    修复前：异常仅记日志，phase 假持 READ_ONLY——唯一发布者已死，
    后续 activate 会在"无发布线程"状态下虚假成功（目标永不上线、
    看门狗失能）；修复后：READ_ONLY 同样锁存 FAULT，activate 被拒。
    """
    arm, transport = _make_arm(tmp_path, clock)

    def _crash(generation: int) -> None:
        raise RuntimeError("测试注入：发送线程内部崩溃")

    # connect 之前注入（实例属性）：发送线程首次决策即崩。若在 connect 后
    # 注入，真实 _decide 已进入 READ_ONLY 条件等待循环（方法内 while 不
    # 返回），实例级替换不会生效
    arm._decide = _crash  # type: ignore[method-assign]
    assert arm.connect() is True
    assert eventually(lambda: arm.get_status().phase == "FAULT")
    status = arm.get_status()
    assert "发送线程内部异常" in (status.fault_reason or "")
    assert status.stop_confirmed is False
    # 故障锁存后 activate 被拒（不得带病接管）
    _feed(transport, clock)
    with pytest.raises(R1DInterfaceError, match="不得再次 activate"):
        arm.activate()
    arm.disconnect()


def test_disconnect_idempotent_closed(
    tmp_path: Path, clock: FakeClock
) -> None:
    """disconnect 幂等 → CLOSED；传输层 close 幂等可重复。"""
    arm, transport = _make_arm(tmp_path, clock)
    assert arm.connect() is True
    arm.disconnect()
    assert arm.get_status().phase == "CLOSED"
    arm.disconnect()
    assert arm.get_status().phase == "CLOSED"
    assert transport.close_count >= 1


# ---------------------------------------------------------------------------
# 状态读取与命令发送语义
# ---------------------------------------------------------------------------


def test_state_returns_14_in_profile_order(
    tmp_path: Path, clock: FakeClock
) -> None:
    """get_arm_state 返回 (14,)=[q(7),dq(7)] 模型序（手算期望表，具名复选框项）。"""
    arm, transport = _make_arm(tmp_path, clock)
    assert arm.connect() is True
    _feed(transport, clock, dq=_measured_dq())
    state = arm.get_arm_state()
    assert state.shape == (14,)
    # 期望表独立手算：q_model = sign*(q_sdk - offset)；dq_model = sign*dq_sdk
    expected_q = tuple(
        s * (sdk - off) for s, sdk, off in zip(_TEST_SIGNS, _LEFT_MEASURED_SDK, _TEST_OFFSETS)
    )
    dq_sdk = tuple(0.01 * (slot + 1) for slot in range(15, 22))
    expected_dq = tuple(s * d for s, d in zip(_TEST_SIGNS, dq_sdk))
    assert_allclose(state[:7], expected_q, atol=1e-12)
    assert_allclose(state[7:], expected_dq, atol=1e-12)
    arm.disconnect()


def test_get_state_failure_raises_explicit_error(
    tmp_path: Path, clock: FakeClock
) -> None:
    """读状态失败（缓存空/未连接）抛显式 R1DStateError，不返回缓存冒充。"""
    arm, transport = _make_arm(tmp_path, clock)
    with pytest.raises(R1DStateError):
        arm.get_arm_state()  # 未连接
    assert arm.connect() is True
    with pytest.raises(R1DStateError, match="缓存为空"):
        arm.get_arm_state()  # 已连接但无帧
    _feed(transport, clock)
    clock.advance(_TIMEOUT + 0.01)
    with pytest.raises(R1DStateError, match="过期"):
        arm.get_arm_state()  # 状态过期
    arm.disconnect()


def test_send_rejected_before_active(tmp_path: Path, clock: FakeClock) -> None:
    """非 ACTIVE 下 send 返回非零拒绝码，且零发布。"""
    arm, transport = _make_arm(tmp_path, clock)
    assert arm.connect() is True
    _feed(transport, clock)
    code = arm.send_joint_command(_Q_ANCHORED)
    assert code != 0
    assert transport.publish_count == 0
    arm.disconnect()


def test_invalid_target_never_published(
    armed_full: tuple[UnitreeR1DInterface, FakeR1DTransport, FakeClock],
) -> None:
    """非法目标（形状/NaN/Inf）同步拒绝（码 2），绝不发布（具名复选框项）。"""
    arm, transport, clock = armed_full
    before = transport.publish_count
    assert arm.send_joint_command(np.full(7, np.nan)) == 2
    assert arm.send_joint_command(np.full(6, 0.1)) == 2
    assert arm.send_joint_command(np.array([np.inf, 0, 0, 0, 0, 0, 0])) == 2
    assert arm.send_joint_command(_Q_ANCHORED + np.nan) == 2
    assert transport.publish_count == before  # 零发布
    assert arm.get_status().phase == "ACTIVE"  # 同步拒绝不触发故障


def test_send_publishes_latest_target_only(
    armed_full: tuple[UnitreeR1DInterface, FakeR1DTransport, FakeClock],
) -> None:
    """send 仅验证并交付最新目标：被更新的旧目标永不发布（单一发送器）。"""
    arm, transport, clock = armed_full
    gate = threading.Event()
    transport.publish_gate = gate
    assert arm.send_joint_command(_Q_ANCHORED) == 0
    assert eventually(lambda: transport.publish_entered >= 1)  # 第一帧进入 publish（阻塞）
    superseded = _Q_NEXT.copy()
    assert arm.send_joint_command(superseded) == 0  # 排队后被覆盖
    latest = _Q_NEXT + 0.001
    assert arm.send_joint_command(latest) == 0
    gate.set()
    # R1（评审 2026-11-18）：发布受控制周期限频——先等首帧实际完成发布
    # （发布期限在成功发布点推进），再推进一个周期，第二帧（最新目标）
    # 才获发布许可
    assert eventually(lambda: len(_active_records(arm)) >= 1)
    clock.advance(_PERIOD)
    assert eventually(lambda: len(_active_records(arm)) >= 2)
    # superseded 永不发布；最新目标被发布
    frames_q = [np.asarray(f.q[15:22]) for f in transport.publish_frames]
    superseded_sdk = np.asarray(_TEST_SIGNS) * superseded + np.asarray(_TEST_OFFSETS)
    latest_sdk = np.asarray(_TEST_SIGNS) * latest + np.asarray(_TEST_OFFSETS)
    for q in frames_q:
        assert not np.allclose(q, superseded_sdk)
    assert any(np.allclose(q, latest_sdk, atol=1e-12) for q in frames_q)


def test_published_frame_carries_converted_target_and_frozen_policy(
    armed_full: tuple[UnitreeR1DInterface, FakeR1DTransport, FakeClock],
) -> None:
    """已发布 active 帧：目标七槽=sign*q+offset+增益；全机 35 槽完整。"""
    arm, transport, clock = armed_full
    assert arm.send_joint_command(_Q_ANCHORED) == 0
    assert eventually(lambda: transport.publish_count >= 1)
    frame = transport.publish_frames[-1]
    assert len(frame.q) == 35
    expected = np.asarray(_TEST_SIGNS) * _Q_ANCHORED + np.asarray(_TEST_OFFSETS)
    assert_allclose(np.asarray(frame.q[15:22]), expected, atol=1e-12)
    assert_allclose(np.asarray(frame.kp[15:22]), np.asarray(_TEST_KP), atol=1e-12)
    # 显式策略槽位（如 0）按注入哨兵保持
    assert_allclose([frame.q[0], frame.kp[0], frame.tau[0]], [0.007, 11.0, 0.21], atol=1e-12)


def test_publish_records_contain_real_publish_times(
    tmp_path: Path, clock: FakeClock
) -> None:
    """R9：发布记录在 transport.publish 调用点计时（含 stop 帧），不冒充 send 接受时刻。"""
    arm, transport = _make_arm(
        tmp_path, clock, stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD
    )
    _connect_and_activate(arm, transport, clock)
    t_accept = clock()
    assert arm.send_joint_command(_Q_ANCHORED) == 0
    assert eventually(lambda: len(_active_records(arm)) >= 1)
    record = _active_records(arm)[0]
    assert record.published_at == pytest.approx(t_accept)
    assert record.session_id == 1
    assert record.success is True
    assert record.q_sdk is not None and record.q_sdk.shape == (7,)
    _run_slow_stop_confirmed(arm, transport, clock)
    stop_records = _stop_records(arm)
    assert len(stop_records) == 1
    assert stop_records[0].kind == "stop"
    assert stop_records[0].q_sdk is not None  # 停止帧也记录锚定实测目标


def test_publish_records_with_delayed_publish_show_real_time(
    armed_full: tuple[UnitreeR1DInterface, FakeR1DTransport, FakeClock],
) -> None:
    """发布延迟发生时，记录时刻=真实发布时刻而非接受时刻（R9 关键区分）。"""
    arm, transport, clock = armed_full
    gate = threading.Event()
    transport.publish_gate = gate
    t_accept = clock()
    assert arm.send_joint_command(_Q_ANCHORED) == 0
    assert eventually(lambda: transport.publish_entered >= 1)
    # 交付期限（接受 + 2 周期 = 0.004s）内的延迟：不构成周期超时，但足以
    # 区分接受时刻与真实发布时刻
    clock.advance(0.003)
    gate.set()
    assert eventually(lambda: len(_active_records(arm)) >= 1)
    record = _active_records(arm)[0]
    assert record.published_at == pytest.approx(t_accept + 0.003)
    assert record.published_at != pytest.approx(t_accept)


# ---------------------------------------------------------------------------
# 看门狗：命令过期/状态过期/周期超时/publish 异常/控制权丢失
# ---------------------------------------------------------------------------


def test_command_timeout_triggers_stop_and_fault(
    armed_full: tuple[UnitreeR1DInterface, FakeR1DTransport, FakeClock],
) -> None:
    """目标命令过期（超过 command_timeout_s 无新目标）：停止策略 + 故障锁存。"""
    arm, transport, clock = armed_full
    assert arm.send_joint_command(_Q_ANCHORED) == 0
    assert eventually(lambda: len(_active_records(arm)) >= 1)
    # 保持状态新鲜地逼近命令超时：先喂帧再小幅越过 command_timeout_s
    clock.advance(_COMMAND_TIMEOUT - 0.01)
    _feed(transport, clock)  # 距上次接受 0.19s，帧龄 0（新鲜）
    clock.advance(0.011)  # 越过 0.2s：命令过期触发，状态仍新鲜（0.011 < 0.05）
    assert _pump(transport, clock, lambda: arm.get_status().phase == "FAULT")
    status = arm.get_status()
    assert "命令过期" in (status.fault_reason or "")
    assert len(_stop_records(arm)) == 1  # 状态新鲜时停止帧可构造并已发布


def test_command_timeout_disabled_by_default(
    tmp_path: Path, clock: FakeClock
) -> None:
    """未注入命令超时（R11 禁用）：空闲 ACTIVE 不因超时故障。"""
    arm, transport = _make_arm(tmp_path, clock)  # 无 command_timeout_s
    _connect_and_activate(arm, transport, clock)
    clock.advance(10.0)
    _feed(transport, clock)  # 状态保持新鲜
    clock.advance(0.001)
    assert eventually(lambda: arm.get_status().phase == "ACTIVE")
    assert arm.get_status().phase == "ACTIVE"
    arm.disconnect()


def test_state_stale_triggers_fault_with_unconfirmed_stop(
    armed_full: tuple[UnitreeR1DInterface, FakeR1DTransport, FakeClock],
) -> None:
    """状态过期：故障锁存；停止帧因无新鲜状态无法构造 → stop_confirmed=False。"""
    arm, transport, clock = armed_full
    clock.advance(_TIMEOUT + 0.01)  # 停止投递帧 → 状态过期
    assert eventually(lambda: arm.get_status().phase == "FAULT")
    status = arm.get_status()
    assert "状态过期" in (status.fault_reason or "")
    assert status.stop_confirmed is False
    assert len(_stop_records(arm)) == 0


def test_period_overrun_drops_target_and_faults(
    tmp_path: Path, clock: FakeClock
) -> None:
    """发送周期超时：滞留发布期间过期的目标被丢弃（不补发）并故障锁存。"""
    arm, transport = _make_arm(
        tmp_path, clock, stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD
    )
    _connect_and_activate(arm, transport, clock)
    gate = threading.Event()
    transport.publish_gate = gate
    assert arm.send_joint_command(_Q_ANCHORED) == 0
    assert eventually(lambda: transport.publish_entered >= 1)  # 第一帧阻塞在 publish 内
    late_target = _Q_NEXT.copy()
    assert arm.send_joint_command(late_target) == 0  # 排队目标
    clock.advance(3.0 * _PERIOD + 0.001)  # 越过交付期限（接受+2 周期）
    gate.set()
    assert _pump(transport, clock, lambda: arm.get_status().phase == "FAULT")
    status = arm.get_status()
    reason = status.fault_reason or ""
    assert "周期超时" in reason
    # Minor 1：消息与判定口径一致（交付期限=接受+1 周期，判限=接受+2 周期）
    assert "交付期限=接受+1 周期" in reason
    assert "判限=接受+2 周期" in reason
    # 过期目标永不发布
    late_sdk = np.asarray(_TEST_SIGNS) * late_target + np.asarray(_TEST_OFFSETS)
    for frame in transport.publish_frames:
        assert not np.allclose(np.asarray(frame.q[15:22]), late_sdk)


def test_publish_exception_triggers_stop_and_fault(
    tmp_path: Path, clock: FakeClock
) -> None:
    """publish 异常：不吞成 0；触发停止策略 + 故障锁存；失链时仅报告尝试。"""
    arm, transport = _make_arm(tmp_path, clock)
    _connect_and_activate(arm, transport, clock)
    transport.publish_error = RuntimeError("DDS Write 失败")
    assert arm.send_joint_command(_Q_ANCHORED) == 0  # 接受（异步发布才失败）
    assert eventually(lambda: arm.get_status().phase == "FAULT")
    status = arm.get_status()
    reason = status.fault_reason or ""
    assert "发布异常" in reason or "停止未确认" in reason
    # 失败的发布尝试也被记录（success=False）
    failed = [r for r in arm.publish_records() if not r.success]
    assert any(r.kind == "active" for r in failed)
    # 停止帧发布同样失败 → 未确认停止（不声称物理急停）
    assert status.stop_confirmed is False
    assert "不声称" in reason or "失链" in reason or "停止未确认" in reason


def test_control_loss_mode_change_triggers_fault(
    tmp_path: Path, clock: FakeClock
) -> None:
    """控制权丢失（ACTIVE 期间 mode_machine 变化）：停止策略 + 故障锁存。"""
    arm, transport = _make_arm(
        tmp_path, clock, stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD
    )
    _connect_and_activate(arm, transport, clock)
    _feed(transport, clock, mode_machine=_MODE_MACHINE + 1)  # 机型字段变化
    assert _pump(
        transport,
        clock,
        lambda: arm.get_status().phase == "FAULT",
        mode_machine=_MODE_MACHINE + 1,
    )
    status = arm.get_status()
    assert "mode_machine" in (status.fault_reason or "")
    assert "控制权丢失" in (status.fault_reason or "")


def test_mode_change_read_only_recorded_no_fault(
    tmp_path: Path, clock: FakeClock
) -> None:
    """READ_ONLY 期间机型变化：仅记录（变化检测），不故障。"""
    arm, transport = _make_arm(tmp_path, clock)
    assert arm.connect() is True
    _feed(transport, clock, mode_machine=_MODE_MACHINE)
    _feed(transport, clock, mode_machine=_MODE_MACHINE + 1)
    clock.advance(0.001)
    assert arm.get_status().phase == "READ_ONLY"
    assert arm.get_status().fault_reason is None
    arm.disconnect()


# ---------------------------------------------------------------------------
# 单一发送器与调度纪律
# ---------------------------------------------------------------------------


def test_stop_sequence_absolute_deadlines_no_burst(
    tmp_path: Path, clock: FakeClock
) -> None:
    """停止序列按绝对期限调度：一次错过多个周期不突发补发（恰 N 帧）。"""
    arm, transport = _make_arm(
        tmp_path,
        clock,
        stop_sequence_frames=3,
        stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD,
    )
    _connect_and_activate(arm, transport, clock)
    calls = {"n": 0}

    def hook() -> None:
        calls["n"] += 1
        clock.advance(10 * _PERIOD)  # 每次发布后时钟大幅前进（错过多个周期）
        _feed(transport, clock)  # 保持状态新鲜并提供确认帧

    transport.publish_hook = hook
    _run_slow_stop_confirmed(arm, transport, clock)
    assert arm.get_status().phase == "READ_ONLY"
    assert len(_stop_records(arm)) == 3  # 恰好 3 帧，无补发


def test_no_motion_commands_after_fault_latch(
    armed_full: tuple[UnitreeR1DInterface, FakeR1DTransport, FakeClock],
) -> None:
    """锁存后没有新的运动命令：send 拒绝（码 3），零新发布。"""
    arm, transport, clock = armed_full
    clock.advance(_TIMEOUT + 0.01)  # 状态过期 → 故障锁存
    assert eventually(lambda: arm.get_status().phase == "FAULT")
    before = transport.publish_count
    assert arm.send_joint_command(_Q_ANCHORED) == 3
    assert arm.send_joint_command(_Q_NEXT) == 3
    clock.advance(0.01)
    assert transport.publish_count == before
    with pytest.raises(R1DInterfaceError):
        arm.activate()


# ---------------------------------------------------------------------------
# 并发（Barrier/Event + fake clock；不持锁等待）
#
# E-M6（评审 2026-11-17）墙钟 sleep 评估结论：本区与缓停测试共 10 处
# time.sleep，逐处评估后分三类处置——
# 1. 谓词驱动泵送循环（while 谓词 + clock.advance + _feed + 2ms 短睡）：
#    fake 时钟冻结，驱动内部期限/确认窗口只能由测试侧推进时钟驱动，
#    条件变量不适用（无墙钟 notify 通路），2ms 轮询即条件等待等价物；
# 2. 负向观察窗（等待"不发生"的事件：FAULT 不出现/记录不新增/线程已
#    退出）：条件变量无法表达"未发生"，必须等满窗口才能下负向结论；
# 3. 有可观察谓词的等待：一律改 eventually/轮询（如 deliver_count 谓词、
#    FAULT 观察窗轮询），不用裸 sleep。
# ---------------------------------------------------------------------------


def test_send_and_slow_stop_concurrent(
    armed_full: tuple[UnitreeR1DInterface, FakeR1DTransport, FakeClock],
) -> None:
    """send 与 slow_stop 并发：首个停止帧之后不得再出现运动命令（不变式）。"""
    arm, transport, clock = armed_full
    gate = threading.Event()
    transport.publish_gate = gate
    barrier = threading.Barrier(3)
    results: list[int] = []
    # E-M5（评审 2026-11-17）：线程异常必须收集——若 sender/stopper 在
    # barrier 后抛异常，daemon 线程静默死亡只会让 results 缺项，不变式
    # 断言可能因"无记录"而伪通过；收集后显式断言两线程零异常退出。
    errors: list[BaseException] = []

    def sender() -> None:
        try:
            barrier.wait()
            results.append(arm.send_joint_command(_Q_ANCHORED))
        except BaseException as exc:  # noqa: BLE001 - 收集后统一断言
            errors.append(exc)

    def stopper() -> None:
        try:
            barrier.wait()
            arm.slow_stop()
        except BaseException as exc:  # noqa: BLE001 - 收集后统一断言
            errors.append(exc)

    threads = [
        threading.Thread(target=sender, daemon=True),
        threading.Thread(target=stopper, daemon=True),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    clock.advance(0.001)
    _feed(transport, clock)
    gate.set()
    # 停止完成后（确认路径）终态 READ_ONLY；泵送推进确认窗口
    deadline = time.monotonic() + _JOIN_TIMEOUT
    while time.monotonic() < deadline and arm.get_status().phase != "READ_ONLY":
        clock.advance(0.001)
        _feed(transport, clock)
        time.sleep(0.002)
    status = arm.get_status()
    assert status.phase == "READ_ONLY"
    # 不变式：首个 stop 记录之后没有新的 active 记录
    records = arm.publish_records()
    first_stop = next(i for i, r in enumerate(records) if r.kind == "stop")
    later_active = [r for r in records[first_stop + 1 :] if r.kind == "active"]
    assert later_active == []
    # E-M5：两线程零异常退出（send 结果本身可为接受或被停拒绝，仅要求无异常）
    for thread in threads:
        thread.join(_JOIN_TIMEOUT)
        assert not thread.is_alive(), "并发线程未在有界时间内退出"
    assert errors == [], f"并发线程抛出异常: {errors}"


def test_disconnect_concurrent_with_state_callback(
    tmp_path: Path, clock: FakeClock
) -> None:
    """disconnect 与状态回调并发：干净关闭，迟到帧被世代门丢弃。"""
    arm, transport = _make_arm(tmp_path, clock)
    assert arm.connect() is True
    stop = threading.Event()
    errors: list[BaseException] = []

    def pump() -> None:
        while not stop.is_set():
            try:
                _feed(transport, clock)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)
                return

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    # E-M6：等泵线程实际投递过至少一帧再 disconnect（谓词等待替代裸
    # sleep——裸 sleep 在慢机上可能等不到投递、在快机上白等）
    assert eventually(lambda: transport.deliver_count >= 1)
    arm.disconnect()
    stop.set()
    thread.join(_JOIN_TIMEOUT)
    assert errors == []
    assert arm.get_status().phase == "CLOSED"
    with pytest.raises(R1DStateError):
        arm.get_arm_state()  # 断开后状态不可用


def test_old_session_thread_no_new_publish_after_reconnect(
    tmp_path: Path, clock: FakeClock
) -> None:
    """旧会话发送线程：滞留发布完成后即退出，不再发布；新会话独立发布。"""
    arm, transport = _make_arm(
        tmp_path, clock, stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD, join_timeout_s=0.05
    )
    assert arm.connect() is True
    _feed(transport, clock)
    arm.activate()
    gate = threading.Event()
    transport.publish_gate = gate
    assert arm.send_joint_command(_Q_ANCHORED) == 0
    assert eventually(lambda: transport.publish_entered >= 1)  # gen1 滞留在 publish 内
    arm.disconnect()  # join 0.05s 超时，线程仍滞留
    assert arm.get_status().phase == "CLOSED"
    gate.set()  # 放行滞留发布
    # gen1 线程完成其在途发布（R10 已知的 publish 调用内窗口）后按世代门退出
    assert eventually(lambda: len(_active_records(arm)) == 1)
    # E-M6：负向观察窗——"此后无新增"只能等满短窗口（线程退出无公开
    # 谓词；记录数已到终值，条件变量无法表达"不再增加"）
    time.sleep(0.02)
    assert len(_active_records(arm)) == 1  # gen1 恰一次在途发布，此后无新增
    # 重连（gen2）后正常发布
    assert arm.connect() is True
    _feed(transport, clock)
    arm.activate()
    assert arm.send_joint_command(_Q_ANCHORED) == 0
    assert eventually(lambda: len(_active_records(arm)) >= 2)
    sessions = [r.session_id for r in _active_records(arm)]
    # disconnect 亦自增世代（失效半建立会话），故重连会话为 3：
    # 关键断言是旧会话 1 恰一次发布、此后无任何会话 1 的新记录
    assert sessions[0] == 1
    assert sessions[1] == 3
    assert len(set(sessions)) == 2
    arm.disconnect()


def test_join_timeout_recorded_and_disconnect_proceeds(
    tmp_path: Path, clock: FakeClock, caplog: Any
) -> None:
    """join 超时：disconnect 仍返回（CLOSED），显式记录滞留线程。"""
    arm, transport = _make_arm(tmp_path, clock, join_timeout_s=0.05)
    assert arm.connect() is True
    _feed(transport, clock)
    arm.activate()
    gate = threading.Event()
    transport.publish_gate = gate
    assert arm.send_joint_command(_Q_ANCHORED) == 0
    assert eventually(lambda: transport.publish_entered >= 1)
    with caplog.at_level("WARNING", logger="src.real.unitree_r1d_interface"):
        arm.disconnect()  # 0.05s join 超时后继续
    assert arm.get_status().phase == "CLOSED"
    assert any("join 超时" in rec.message for rec in caplog.records)
    gate.set()  # 释放滞留线程
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        time.sleep(0.01)
        if arm.get_status().phase == "CLOSED":
            break
    assert arm.get_status().phase == "CLOSED"


def test_no_deadlock_send_stop_disconnect_under_load(
    tmp_path: Path, clock: FakeClock
) -> None:
    """持锁不等待线程：高压 send/stop/disconnect 混发不死锁（有界完成）。"""
    arm, transport = _make_arm(
        tmp_path, clock, stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD, join_timeout_s=0.2
    )
    assert arm.connect() is True
    _feed(transport, clock)
    arm.activate()
    stop = threading.Event()
    errors: list[BaseException] = []

    def worker() -> None:
        i = 0
        while not stop.is_set() and i < 200:
            try:
                arm.send_joint_command(_Q_ANCHORED if i % 2 == 0 else _Q_NEXT)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)
                return
            i += 1
        try:
            arm.slow_stop()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + _JOIN_TIMEOUT
    while time.monotonic() < deadline and any(t.is_alive() for t in threads):
        clock.advance(0.001)
        _feed(transport, clock)
        time.sleep(0.002)
    stop.set()
    for t in threads:
        t.join(_JOIN_TIMEOUT)
        assert not t.is_alive(), "工作线程未在有界时间内完成（疑似死锁）"
    assert errors == []
    arm.disconnect()


# ---------------------------------------------------------------------------
# 评审 2026-11-18 修复轮（复审 844f4c1）：R1 发布限频 / R2 发布许可 /
# R3 退权窗口急停升级
#
# 红线对应评审复现口径：冻结 fake clock + Event 暂停发送线程于构造/
# 传输边界，确定性制造"停止登记先于发布许可"的窗口。窥视内部登记
# 字段（_stop_request/_emergency_latch/_async_fault_reason）仅为确定
# 竞态顺序——停止登记无公开观测点，公开它会诱导依赖内部时序（C-M6
# 先例：不为测试观测增设公开只读接口）。
# ---------------------------------------------------------------------------


class TestPublishCadence:
    """R1（P1）：发布节奏受控制周期限频，高频提交不能绕过步长限速。"""

    def test_burst_within_period_publishes_once(
        self, armed_full: tuple[UnitreeR1DInterface, FakeR1DTransport, FakeClock]
    ) -> None:
        """冻结时钟下同拍连续提交：首帧立即发布，周期内不得多发。"""
        arm, transport, clock = armed_full
        base = _Q_ANCHORED.copy()
        assert arm.send_joint_command(base) == 0
        assert eventually(lambda: len(_active_records(arm)) >= 1)
        # 同拍连续提交 3 个递增目标（累计 0.003 < 单拍步长界 0.004）
        for k in (1, 2, 3):
            assert arm.send_joint_command(base + 0.001 * k) == 0
        # 负向观察窗（评审 2026-11-18 复现：五帧同刻发布）——周期未到，
        # 不得出现第二帧（真实 50ms 窗口内 fake 时钟冻结）
        time.sleep(0.05)
        assert len(_active_records(arm)) == 1, "同一控制周期内出现了多次发布"

    def test_deadline_consumes_latest_target_only(
        self, armed_full: tuple[UnitreeR1DInterface, FakeR1DTransport, FakeClock]
    ) -> None:
        """期限到达消费最新目标：被替换的中间目标永不发布。"""
        arm, transport, clock = armed_full
        base = _Q_ANCHORED.copy()
        assert arm.send_joint_command(base) == 0
        assert eventually(lambda: len(_active_records(arm)) >= 1)
        first = _active_records(arm)[0]
        for k in (1, 2, 3):
            assert arm.send_joint_command(base + 0.001 * k) == 0
            # 真实窗口让"无节奏约束的旧实现"逐帧发出——这样下面的
            # "只发 2 帧 / 中间目标从未发布"才是判别性断言（评审 2026-11-18
            # 修复轮红线自查：旧实现若无间隔会全部发出，本测试必须红）
            time.sleep(0.02)
        clock.advance(_PERIOD)
        assert eventually(lambda: len(_active_records(arm)) >= 2)
        records = _active_records(arm)
        assert len(records) == 2
        # 相邻成功 active 发布间隔 ≥ 控制周期（注入时钟绝对期限）
        assert records[1].published_at >= first.published_at + _PERIOD - 1e-9
        skipped_sdk = (
            np.asarray(_TEST_SIGNS) * (base + 0.002) + np.asarray(_TEST_OFFSETS)
        )
        latest_sdk = (
            np.asarray(_TEST_SIGNS) * (base + 0.003) + np.asarray(_TEST_OFFSETS)
        )
        for frame in transport.publish_frames:
            assert not np.allclose(np.asarray(frame.q[15:22]), skipped_sdk)
        assert np.allclose(
            np.asarray(transport.publish_frames[-1].q[15:22]), latest_sdk, atol=1e-12
        )

    def test_post_lag_publish_still_cadence_limited(
        self, armed_full: tuple[UnitreeR1DInterface, FakeR1DTransport, FakeClock]
    ) -> None:
        """调度延迟后的发布：期限自实际发布时刻顺延，不追赶网格连发。"""
        arm, transport, clock = armed_full
        assert arm.send_joint_command(_Q_ANCHORED) == 0
        assert eventually(lambda: len(_active_records(arm)) >= 1)
        first = _active_records(arm)[0].published_at
        # 迟到 1.5 周期才提交第二目标：交付期限（接受+2 周期）内立即发布
        clock.advance(1.5 * _PERIOD)
        assert arm.send_joint_command(_Q_NEXT) == 0
        assert eventually(lambda: len(_active_records(arm)) >= 2)
        records = _active_records(arm)
        second = records[1].published_at
        assert second - first >= 1.5 * _PERIOD - 1e-9  # 期间确实存在调度延迟
        # 距第二帧实际发布不足一个周期的新目标不得发布（不追赶网格）
        clock.advance(0.9 * _PERIOD)
        assert arm.send_joint_command(_Q_ANCHORED) == 0
        time.sleep(0.03)  # 负向观察窗
        assert len(_active_records(arm)) == 2, "迟到发布后出现追赶连发"
        # 自实际发布时刻顺延满一个周期后才发布
        clock.advance(0.2 * _PERIOD)
        assert eventually(lambda: len(_active_records(arm)) >= 3)
        assert _active_records(arm)[2].published_at - second >= _PERIOD - 1e-9

    def test_high_frequency_producer_spacing_invariant(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """高频生产者不变式：相邻成功 active 发布间隔恒 ≥ 控制周期。

        生产者按"轮内 5 连发无间隔突发"提交（模拟高频调用），主线程
        同步推进 fake 时钟——若限频失效，一个周期内会积累多个允许步长
        并同拍多发（评审复现：五帧同刻发布）。
        """
        arm, transport = _make_arm(
            tmp_path, clock, stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD
        )
        _connect_and_activate(arm, transport, clock)
        stop = threading.Event()
        sent = {"n": 0}
        errors: list[str] = []

        def producer() -> None:
            try:
                for _ in range(40):
                    for _ in range(5):
                        if stop.is_set():
                            return
                        if arm.send_joint_command(_Q_NEXT) != 0:
                            errors.append("send 被拒")
                            return
                        sent["n"] += 1
                    # 轮间真实节拍 1ms：快于主线程进钟速率（2ms/周期），
                    # 使无节奏约束的旧实现按 1ms 模拟时间隔连发（间隔
                    # < 周期 → 红线自查必红）；有节奏约束时发布仍按期限
                    time.sleep(0.001)
            except BaseException as exc:  # noqa: BLE001 - 收集后统一断言
                errors.append(f"producer 异常: {exc}")
            finally:
                stop.set()

        threading.Thread(target=producer, daemon=True).start()
        # 同步速率推进时钟 + 喂帧（冻结时钟下期限不自行到期）
        deadline = time.monotonic() + _JOIN_TIMEOUT
        while time.monotonic() < deadline and not stop.is_set():
            clock.advance(_PERIOD)
            _feed(transport, clock)
            time.sleep(0.002)
        stop.set()
        # 排空在途目标：推进周期至发布数稳定（连续两轮不变即排空）
        drain_deadline = time.monotonic() + 1.0
        last_count = -1
        while time.monotonic() < drain_deadline:
            clock.advance(_PERIOD)
            _feed(transport, clock)
            time.sleep(0.002)
            current = len(_active_records(arm))
            if current == last_count and current >= 2:
                break
            last_count = current
        records = [r for r in _active_records(arm) if r.success]
        assert len(records) >= 5, "高频提交下应发生多次发布"
        assert sent["n"] == 200 and errors == []
        for prev, nxt in zip(records, records[1:]):
            assert (
                nxt.published_at - prev.published_at >= _PERIOD - 1e-9
            ), "相邻发布间隔低于控制周期（限频失效）"
        arm.disconnect()


class TestPublishPermit:
    """R2（P1）：停止登记与发布许可互斥——未获许可的运动帧被撤销。"""

    @pytest.mark.parametrize("stop_kind", ["emergency", "slow", "async_fault"])
    def test_stop_registration_revokes_unpublished_frame(
        self, tmp_path: Path, clock: FakeClock, stop_kind: str
    ) -> None:
        """构造期间登记停止：已构造未发布的运动帧被撤销，零 active 发布。

        评审复现口径：build_active 返回前暂停发送线程，登记急停并确认
        尚未调用 transport.publish——恢复后不得再发布 active。
        """
        arm, transport = _make_arm(
            tmp_path, clock, stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD
        )
        _connect_and_activate(arm, transport, clock)
        in_build = threading.Event()
        release_build = threading.Event()
        original_build = arm._policy.build_active  # noqa: SLF001  # 测试钩子：暂停构造窗口

        def hooked_build(state: Any, target: Any) -> Any:
            in_build.set()
            release_build.wait(_JOIN_TIMEOUT)
            return original_build(state, target)

        arm._policy.build_active = hooked_build  # noqa: SLF001
        assert arm.send_joint_command(_Q_ANCHORED) == 0
        assert in_build.wait(_JOIN_TIMEOUT)  # 发送线程已进入构造（许可判定前）

        stop_thread: threading.Thread | None = None
        if stop_kind == "emergency":
            stop_thread = threading.Thread(target=arm.emergency_stop, daemon=True)
            stop_thread.start()
            # 确定性前提：急停登记完成（锁内置位，公开观测点缺失）
            assert eventually(
                lambda: arm._emergency_latch  # noqa: SLF001
            ), "急停未在恢复构造前完成登记"
        elif stop_kind == "slow":
            stop_thread = threading.Thread(target=arm.slow_stop, daemon=True)
            stop_thread.start()
            assert eventually(
                lambda: arm._stop_request == "slow"  # noqa: SLF001
            ), "缓停未在恢复构造前完成登记"
        else:
            _feed(transport, clock, mode_machine=_MODE_MACHINE + 1)  # 控制权丢失
            assert (
                arm._async_fault_reason is not None  # noqa: SLF001
            ), "异步故障未登记"
        # 登记完成时运动帧尚未进入传输层（评审证据口径）
        assert transport.publish_entered == 0

        release_build.set()
        expected_phase = "READ_ONLY" if stop_kind == "slow" else "FAULT"
        deadline = time.monotonic() + _JOIN_TIMEOUT
        while time.monotonic() < deadline and arm.get_status().phase != expected_phase:
            clock.advance(0.001)
            _feed(transport, clock)
            time.sleep(0.002)
        if stop_thread is not None:
            stop_thread.join(_JOIN_TIMEOUT)
            assert not stop_thread.is_alive()
        status = arm.get_status()
        assert status.phase == expected_phase
        # 核心不变式：停止登记后零运动帧发布（构造中/许可前的帧被撤销）
        assert _active_records(arm) == []
        assert len(_stop_records(arm)) >= 1
        if stop_kind == "emergency":
            # R3 联动：急停终态下原对象不得再次激活
            _feed(transport, clock)
            with pytest.raises(R1DInterfaceError):
                arm.activate()

    def test_permitted_inflight_frame_not_revoked(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """许可边界：已获许可进入传输层的在途帧完成发布（不可撤销）。

        许可授出（进入 transport.publish）之后登记的急停只能约束后续
        帧——该帧按顺序先于停止帧发布，随后停止序列正常执行。
        """
        arm, transport = _make_arm(
            tmp_path, clock, stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD
        )
        _connect_and_activate(arm, transport, clock)
        gate = threading.Event()
        transport.publish_gate = gate
        assert arm.send_joint_command(_Q_ANCHORED) == 0
        # 许可已授出：发送线程滞留在 transport.publish 内（在途帧）
        assert eventually(lambda: transport.publish_entered >= 1)
        stop_thread = threading.Thread(target=arm.emergency_stop, daemon=True)
        stop_thread.start()
        assert eventually(lambda: arm._emergency_latch)  # noqa: SLF001
        gate.set()
        deadline = time.monotonic() + _JOIN_TIMEOUT
        while time.monotonic() < deadline and arm.get_status().phase != "FAULT":
            clock.advance(0.001)
            _feed(transport, clock)
            time.sleep(0.002)
        stop_thread.join(_JOIN_TIMEOUT)
        assert not stop_thread.is_alive()
        assert arm.get_status().phase == "FAULT"
        kinds = [r.kind for r in arm.publish_records() if r.success]
        assert kinds == ["active", "stop"]  # 在途帧 + 停止帧，且仅此二者


class TestEmergencyEscalationDuringRelease:
    """R3（P2）：退权窗口的急停升级不丢失，终态 FAULT 拒绝再激活。"""

    def test_emergency_during_release_escalates_terminal_fault(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """退权执行期间登记急停：终态 FAULT（不回落 READ_ONLY 清锁存）。"""
        arm, transport = _make_arm(
            tmp_path, clock, stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD
        )
        _connect_and_activate(arm, transport, clock)
        assert arm.send_joint_command(_Q_ANCHORED) == 0
        assert eventually(lambda: len(_active_records(arm)) >= 1)
        release_gate = threading.Event()
        transport.release_gate = release_gate
        done = threading.Event()
        stop_errors: list[BaseException] = []

        def _slow_stop() -> None:
            try:
                arm.slow_stop()
            except BaseException as exc:  # noqa: BLE001 - 收集后统一断言
                stop_errors.append(exc)
            done.set()

        threading.Thread(target=_slow_stop, daemon=True).start()
        # 泵送停止序列至退权调用点（停止帧发布 + 反馈确认在退权之前完成）
        deadline = time.monotonic() + _JOIN_TIMEOUT
        while time.monotonic() < deadline and transport.release_entered < 1:
            clock.advance(0.001)
            _feed(transport, clock)
            time.sleep(0.002)
        assert transport.release_entered >= 1, "停止序列未到达退权调用点"

        # 退权执行期间登记急停（评审复现窗口：release_control 阻塞中）
        escalate_thread = threading.Thread(target=arm.emergency_stop, daemon=True)
        escalate_thread.start()
        assert eventually(lambda: arm._emergency_latch)  # noqa: SLF001
        release_gate.set()

        deadline = time.monotonic() + _JOIN_TIMEOUT
        while time.monotonic() < deadline and not done.is_set():
            clock.advance(0.001)
            _feed(transport, clock)
            time.sleep(0.002)
        assert done.wait(0.1) and stop_errors == []
        escalate_thread.join(_JOIN_TIMEOUT)
        assert not escalate_thread.is_alive()

        status = arm.get_status()
        assert status.phase == "FAULT", "退权窗口的急停升级被丢失（终态回落）"
        assert status.stop_confirmed is True  # 停止本身已确认（升级不推翻确认）
        assert "急停升级" in (status.fault_reason or "")
        # 不可恢复语义：补入新鲜状态后原对象仍不得再次激活
        _feed(transport, clock)
        with pytest.raises(R1DInterfaceError, match="故障锁存"):
            arm.activate()


# 退权阻塞窗口登记的异步故障文案（与驱动回调一致，测试专用引用）
_ASYNC_CONTROL_LOSS = (
    "控制权丢失疑似：ACTIVE 期间 mode_machine 7→8（机型字段变化，按控制权丢失处理）"
)


class TestAsyncControlLossDuringRelease:
    """P1-2：退权窗口登记的异步控制权丢失必须并入停止终态（不得回落 READ_ONLY）。"""

    def test_async_control_loss_during_release_is_terminal_fault(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """退权阻塞期间注入机型变化：终态 FAULT 且原因保留原 trigger 与控制权丢失。

        失败口径（修复前）：_finish_stop 只读 _emergency_latch，退权阻塞窗口
        登记的 _async_fault_reason 被丢弃，终态回落 READ_ONLY/fault_reason=None。
        """
        arm, transport = _make_arm(
            tmp_path, clock, stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD
        )
        _connect_and_activate(arm, transport, clock)
        assert arm.send_joint_command(_Q_ANCHORED) == 0
        assert eventually(lambda: len(_active_records(arm)) >= 1)

        release_gate = threading.Event()
        transport.release_gate = release_gate
        done = threading.Event()
        stop_errors: list[BaseException] = []

        def _slow_stop() -> None:
            try:
                arm.slow_stop()
            except BaseException as exc:  # noqa: BLE001 - 收集后统一断言
                stop_errors.append(exc)
            done.set()

        stop_thread = threading.Thread(target=_slow_stop, daemon=True)
        stop_thread.start()
        try:
            # 泵送停止序列至退权调用点（停止帧发布 + 反馈确认先于退权完成）
            deadline = time.monotonic() + _JOIN_TIMEOUT
            while time.monotonic() < deadline and transport.release_entered < 1:
                clock.advance(0.001)
                _feed(transport, clock)
                time.sleep(0.002)
            assert transport.release_entered >= 1, "停止序列未到达退权调用点"

            # 退权阻塞窗口：注入合法（新鲜度/时间戳正常）但机型字段变化的
            # 状态帧——回调在 STOPPING 下登记异步故障（控制权丢失疑似）
            _feed(transport, clock, mode_machine=_MODE_MACHINE + 1)
            assert arm._async_fault_reason is not None, (  # noqa: SLF001
                "退权窗口的控制权丢失未登记（夹具/前置条件错误）"
            )
        finally:
            release_gate.set()
            stop_thread.join(_JOIN_TIMEOUT)
        assert not stop_thread.is_alive()
        assert done.wait(0.1)
        assert stop_errors == []

        status = arm.get_status()
        try:
            assert status.phase == "FAULT", "退权窗口的异步故障被停止收尾丢弃"
            assert status.stop_confirmed is True  # 故障不推翻已确认的停止
            reason = status.fault_reason or ""
            assert "用户缓停请求" in reason, reason  # 原 trigger 不得丢失
            assert "控制权丢失" in reason, reason
            assert arm._async_fault_reason is None  # noqa: SLF001  已并入终态
        finally:
            arm.disconnect()

    def test_mode_change_after_stop_does_not_raise_fault(
        self, tmp_path: Path, clock: FakeClock
    ) -> None:
        """停止完成（READ_ONLY）后到达的机型变化不登记异步故障（防过度触发）。"""
        arm, transport = _make_arm(
            tmp_path, clock, stop_confirm_dq_threshold=_STOP_DQ_THRESHOLD
        )
        _connect_and_activate(arm, transport, clock)
        try:
            _run_slow_stop_confirmed(arm, transport, clock)
            status = arm.get_status()
            assert status.phase == "READ_ONLY"
            assert status.fault_reason is None
            assert status.stop_confirmed is True
            # 回调的机型变化检测条件只含 ACTIVE/STOPPING：停止完成后的
            # 晚到机型变化帧不得把已完成的会话推入 FAULT
            _feed(transport, clock, mode_machine=_MODE_MACHINE + 1)
            assert arm._async_fault_reason is None  # noqa: SLF001
            _feed(transport, clock, mode_machine=_MODE_MACHINE + 2)
            status = arm.get_status()
            assert status.phase == "READ_ONLY"
            assert status.fault_reason is None
        finally:
            arm.disconnect()


class TestStopFinalizationFaultMerge:
    """P1-2：_finish_stop 同锁合并四类故障源并保持确认/清理/日志语义。"""

    @pytest.mark.parametrize(
        "confirmed", [True, False], ids=["confirmed", "unconfirmed"]
    )
    @pytest.mark.parametrize(
        (
            "trigger",
            "async_reason",
            "escalated",
            "outcome",
            "final_fault",
            "expected_fault",
            "expected_fragments",
        ),
        [
            pytest.param(
                "用户缓停请求", None, False, None, False, False, (), id="no-fault"
            ),
            pytest.param(
                "用户缓停请求",
                _ASYNC_CONTROL_LOSS,
                False,
                None,
                False,
                True,
                ("用户缓停请求", "控制权丢失"),
                id="async-only",
            ),
            pytest.param(
                "用户缓停请求",
                None,
                True,
                None,
                False,
                True,
                ("用户缓停请求", "急停升级"),
                id="emergency-only",
            ),
            pytest.param(
                "用户缓停请求",
                None,
                False,
                "停止未确认：测试用停止失败",
                False,
                True,
                ("用户缓停请求", "停止未确认"),
                id="outcome-only",
            ),
            pytest.param(
                "用户缓停请求",
                None,
                False,
                None,
                True,
                True,
                ("用户缓停请求",),
                id="final-fault",
            ),
            pytest.param(
                "用户缓停请求",
                _ASYNC_CONTROL_LOSS,
                True,
                "停止未确认：测试用停止失败",
                False,
                True,
                ("用户缓停请求", "控制权丢失", "急停升级", "停止未确认"),
                id="async-emergency-outcome",
            ),
            pytest.param(
                _ASYNC_CONTROL_LOSS,
                _ASYNC_CONTROL_LOSS,
                False,
                None,
                False,
                True,
                ("控制权丢失",),
                id="async-equals-trigger",
            ),
        ],
    )
    def test_finish_stop_merges_fault_sources(
        self,
        tmp_path: Path,
        clock: FakeClock,
        trigger: str,
        async_reason: str | None,
        escalated: bool,
        outcome: str | None,
        final_fault: bool,
        expected_fault: bool,
        expected_fragments: tuple[str, ...],
        confirmed: bool,
    ) -> None:
        """直接驱动 _finish_stop：终态/合并 reason/确认结果/字段清理按设计落位。"""
        arm, _ = _make_arm(tmp_path, clock)
        accepted_at = clock()
        with arm._lock:  # noqa: SLF001 - 构造待收尾会话现场
            arm._phase = "STOPPING"  # noqa: SLF001
            arm._pending_target = np.full(7, 0.25)  # noqa: SLF001
            arm._pending_accepted_at = accepted_at  # noqa: SLF001
            arm._activation_mode_machine = _MODE_MACHINE  # noqa: SLF001
            arm._emergency_latch = escalated  # noqa: SLF001
            arm._async_fault_reason = async_reason  # noqa: SLF001
            arm._stop_confirmed = not confirmed  # noqa: SLF001 哨兵：须被覆盖
        arm._finish_stop(  # noqa: SLF001 - 被测内部契约
            0, trigger, outcome=outcome, confirmed=confirmed, final_fault=final_fault
        )
        status = arm.get_status()
        assert arm._stop_done.is_set()  # noqa: SLF001
        with arm._lock:  # noqa: SLF001
            assert arm._pending_target is None
            assert arm._pending_accepted_at is None
            assert arm._activation_mode_machine is None
            assert arm._emergency_latch is False
            assert arm._async_fault_reason is None  # 已并入终态，不得残留
        # 停止确认与故障终态是两个正交事实：两条终态分支都保留 confirmed
        assert status.stop_confirmed is confirmed
        if expected_fault:
            assert status.phase == "FAULT"
            reason = status.fault_reason or ""
            assert trigger in reason, reason  # 原 trigger 不得被丢弃
            for fragment in expected_fragments:
                assert fragment in reason, reason
            # 同一段完整文本不得重复拼接（trigger 与 async 同文时只保留一次）
            assert reason.count(_ASYNC_CONTROL_LOSS) <= 1, reason
            assert reason.count("退权窗口收到急停升级（终态强制 FAULT）") <= 1, reason
        else:
            assert status.phase == "READ_ONLY"
            assert status.fault_reason is None

    def test_stale_finish_stop_preserves_new_session_fault(
        self, tmp_path: Path, clock: FakeClock, caplog: Any
    ) -> None:
        """世代失效的旧收尾不得触碰新会话字段（尤其不清异步故障）。"""
        arm, _ = _make_arm(tmp_path, clock)
        new_async_reason = "控制权丢失疑似：新会话已登记（旧会话收尾不得清除）"
        accepted_at = clock()
        with arm._lock:  # noqa: SLF001 - 模拟已存在的新会话现场
            arm._generation = 7  # noqa: SLF001
            arm._phase = "STOPPING"  # noqa: SLF001
            arm._fault_reason = None  # noqa: SLF001
            arm._stop_confirmed = False  # noqa: SLF001
            arm._async_fault_reason = new_async_reason  # noqa: SLF001
            arm._emergency_latch = True  # noqa: SLF001
            arm._pending_target = np.full(7, 0.25)  # noqa: SLF001
            arm._pending_accepted_at = accepted_at  # noqa: SLF001
            arm._activation_mode_machine = _MODE_MACHINE + 1  # noqa: SLF001
            arm._stop_done.clear()  # noqa: SLF001
        with caplog.at_level("INFO", logger="src.real.unitree_r1d_interface"):
            arm._finish_stop(  # noqa: SLF001 - 旧会话（世代 6）收尾
                6,
                "旧会话缓停请求",
                outcome="旧会话停止失败",
                confirmed=True,
                final_fault=True,
            )
        with arm._lock:  # noqa: SLF001
            assert arm._generation == 7
            assert arm._phase == "STOPPING"
            assert arm._fault_reason is None
            assert arm._stop_confirmed is False
            assert arm._async_fault_reason == new_async_reason
            assert arm._emergency_latch is True
            assert arm._pending_target is not None
            assert arm._pending_accepted_at == accepted_at
            assert arm._activation_mode_machine == _MODE_MACHINE + 1
        # 既有完成事件协议保持：stale 收尾仍释放等待者，但不改写会话字段
        assert arm._stop_done.is_set()  # noqa: SLF001
        assert any("停止收尾跳过" in record.message for record in caplog.records)

    def test_async_only_stop_logs_fault_not_read_only(
        self, tmp_path: Path, clock: FakeClock, caplog: Any
    ) -> None:
        """async-only 终态为 FAULT：日志不得按 READ_ONLY 成功口径输出。"""
        arm, _ = _make_arm(tmp_path, clock)
        with arm._lock:  # noqa: SLF001
            arm._phase = "STOPPING"  # noqa: SLF001
            arm._emergency_latch = False  # noqa: SLF001
            arm._async_fault_reason = _ASYNC_CONTROL_LOSS  # noqa: SLF001
            arm._stop_confirmed = False  # noqa: SLF001
        with caplog.at_level("INFO", logger="src.real.unitree_r1d_interface"):
            arm._finish_stop(  # noqa: SLF001
                0, "用户缓停请求", outcome=None, confirmed=True, final_fault=False
            )
        messages = [record.message for record in caplog.records]
        assert not any("停止结束：READ_ONLY" in message for message in messages), (
            messages
        )
        assert any("停止结束：FAULT" in message for message in messages), messages
        status = arm.get_status()
        assert status.phase == "FAULT"
        assert status.stop_confirmed is True
        assert "控制权丢失" in (status.fault_reason or "")
