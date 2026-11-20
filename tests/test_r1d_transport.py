"""SdkR1DTransport 传输适配测试 — 懒加载/生命周期/故障注入（阶段 4 Task 4.5）。

全部通过 fake SDK（_SdkBundle 注入）驱动，不连接真实 DDS 网络、不初始化
真实 ChannelFactory、不触碰硬件；真实 IDL/CRC 序列化对照见
tests/test_r1d_sdk_serialization.py（本文件结论属"普通 Mock 通过"栏）。

协议事实来源：configs/robots/r1d_protocol_manifest.yaml（Task 4.1 冻结）。
测试内数值均为测试专用有限参数，非实机参数。
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
import pytest

import src.real.r1d_transport as r1d_transport_module
from src.real.r1d_config import (
    R1DCalibrationConfig,
    R1DCommunicationConfig,
    R1DHardwareConfig,
    R1DLayoutConfig,
    R1DLimitConfig,
    R1DTimingConfig,
)
from src.real.r1d_transport import (
    R1DChannelError,
    R1DControlError,
    R1DDependencyError,
    R1DPublishError,
    R1DTransportUsageError,
    SdkR1DTransport,
    _SdkBundle,
)
from src.real.r1d_types import (
    R1D_SLOT_COUNT,
    R1DCommandFrame,
    R1DStateFrame,
    R1DTransport,
)

# 仓库根（子进程纯度检查的 cwd/PYTHONPATH）
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 测试专用 fake SDK 组件（记录调用序列 + 可注入故障；非真实协议实现）
# ---------------------------------------------------------------------------


class _FakeClock:
    """单调时钟替身：每次调用前进固定步长（测试专用，非实机时钟）。"""

    def __init__(self, start: float = 100.0, step: float = 0.01) -> None:
        self._now = start
        self._step = step

    def __call__(self) -> float:
        now = self._now
        self._now += self._step
        return now


# 并发测试的通用时限：只作测试上界（防挂起），不作正确性证据
_GATE_TIMEOUT_S = 10.0


class _Gate:
    """测试闸门：首个调用者阻塞到显式放行（有界等待）。

    Attributes:
        entered: 首个调用者已进入闸门。
        release: 放行事件（测试主线程控制）。
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()
        self._claimed = False

    def block_once(self) -> None:
        """首个调用者阻塞到放行；后续调用直接返回（不重复阻塞）。

        Raises:
            RuntimeError: 超过测试时限仍未放行（测试环境异常，不得静默挂起）。
        """
        with self._lock:
            if self._claimed:
                return
            self._claimed = True
        self.entered.set()
        if not self.release.wait(timeout=_GATE_TIMEOUT_S):
            raise RuntimeError("测试闸门等待放行超时")


class _ObservableRLock:
    """RLock 包装：非阻塞探测失败即置 contended（只用于观测争锁，不改变语义）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.contended = threading.Event()

    def __enter__(self) -> _ObservableRLock:
        if not self._lock.acquire(blocking=False):
            self.contended.set()
            self._lock.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._lock.release()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        """代理底层 acquire（供 with/显式探测共用）。"""
        acquired = self._lock.acquire(blocking=blocking, timeout=timeout)
        if not acquired:
            self.contended.set()
        return acquired

    def release(self) -> None:
        """代理底层 release。"""
        self._lock.release()


class _ThreadGatedLock:
    """按线程名在首次取锁前阻塞的锁包装器（fix1 红线：覆盖 publish 入口窗口）。

    仅阻塞指定名称的线程一次，其余线程直通底层 RLock；用于让被测线程停在
    “已进入方法、尚未取得 io 锁”的窗口，主线程完成会话更替后再放行。

    Attributes:
        blocked: 指定线程已到达入口并进入阻断。
    """

    def __init__(self, blocked_thread_name: str, gate: _Gate) -> None:
        self._inner = threading.RLock()
        self._blocked_thread_name = blocked_thread_name
        self._gate = gate
        self.blocked = threading.Event()

    def __enter__(self) -> _ThreadGatedLock:
        if (
            threading.current_thread().name == self._blocked_thread_name
            and not self.blocked.is_set()
        ):
            self.blocked.set()
            self._gate.block_once()
        self._inner.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._inner.release()


class _SequenceClock:
    """剧本时钟：按序返回给定时间值，末尾值重复（close 耗时阈值测试专用）。"""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)
        self._index = 0

    def __call__(self) -> float:
        index = min(self._index, len(self._values) - 1)
        self._index += 1
        return self._values[index]


def _probe_io_lock(io_lock: Any) -> None:
    """跨线程探测 io 锁：Close 内回调线程需取 io 时不得被持锁方阻塞（2 s 界）。

    Args:
        io_lock: 传输层 io 锁（None 表示本测试不探测）。

    Raises:
        RuntimeError: 探测线程 2 s 内无法取得锁（模拟真实 SDK Close 等回调死锁）。
    """
    if io_lock is None:
        return
    done = threading.Event()

    def _probe() -> None:
        with io_lock:
            done.set()

    thread = threading.Thread(target=_probe, name="fake-close-io-probe", daemon=True)
    thread.start()
    if not done.wait(timeout=2.0):
        raise RuntimeError("Close 回调线程 2s 内无法取得 io 锁（疑似死锁）")
    thread.join(timeout=2.0)


def _thread_with_errors(
    target: Callable[[], Any], errors: list[Exception], *, name: str
) -> threading.Thread:
    """构造守护线程：捕获任意异常回传主线程（由主线程统一断言）。"""

    def runner() -> None:
        try:
            target()
        except Exception as exc:  # noqa: BLE001 - 线程异常统一回传主线程，不吞断言
            errors.append(exc)

    return threading.Thread(target=runner, name=name, daemon=True)


def _join_threads(*threads: threading.Thread) -> None:
    """有界 join 后断言已启动的线程全部退出（超时即测试失败，不静默挂起）。

    跳过从未 `start()` 的线程：若早先断言失败导致只启动了部分线程，对未启动
    线程 `join()` 会抛 `RuntimeError: cannot join thread before it is started`，
    掩盖最初的失败原因（fix1 收尾健壮性）。
    """
    started = [thread for thread in threads if thread.ident is not None]
    for thread in started:
        thread.join(timeout=_GATE_TIMEOUT_S)
    alive = [thread.name for thread in started if thread.is_alive()]
    assert not alive, f"线程未在 {_GATE_TIMEOUT_S}s 内退出: {alive}"


def _wait_until(predicate: Callable[[], bool], timeout_s: float) -> bool:
    """短轮询等待条件成立（仅测试观测用，正确性判定不依赖轮询本身）。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


class _FakeMotorCmd:
    """MotorCmd_ 替身（显式全字段构造）。"""

    def __init__(
        self,
        *,
        mode: int,
        q: float,
        dq: float,
        tau: float,
        kp: float,
        kd: float,
        reserve: int = 0,
    ) -> None:
        self.mode = mode
        self.q = q
        self.dq = dq
        self.tau = tau
        self.kp = kp
        self.kd = kd
        self.reserve = reserve


class _FakeLowCmd:
    """LowCmd_ 替身（显式全字段构造）。"""

    def __init__(
        self,
        *,
        mode_pr: int,
        mode_machine: int,
        motor_cmd: list,
        reserve: list,
        crc: int,
    ) -> None:
        self.mode_pr = mode_pr
        self.mode_machine = mode_machine
        self.motor_cmd = list(motor_cmd)
        self.reserve = list(reserve)
        self.crc = crc


class _FakeMotorState:
    """MotorState_ 替身（仅传输层消费的字段）。"""

    def __init__(self, q: float, dq: float) -> None:
        self.q = q
        self.dq = dq


class _FakeLowState:
    """LowState_ 替身（仅传输层消费的字段）。"""

    def __init__(
        self, *, tick: int, mode_machine: int, q: np.ndarray, dq: np.ndarray, crc: int
    ) -> None:
        self.tick = tick
        self.mode_machine = mode_machine
        self.motor_state = [
            _FakeMotorState(float(q[i]), float(dq[i])) for i in range(R1D_SLOT_COUNT)
        ]
        self.crc = crc


def _fake_crc(msg: Any) -> int:
    """fake CRC：确定性校验和（测试专用，非协议 CRC）。

    NaN 容忍：NaN 帧的 CRC 记 0，保证测试的 NaN 走到 R1DStateFrame
    有限性校验路径而非在 fake CRC 处提前崩掉。
    """

    motors = getattr(msg, "motor_state", None)
    if motors is None:
        motors = getattr(msg, "motor_cmd", None)
    if motors is None:
        return 0  # 无电机字段（A-M4 异常消息构造用）：CRC 记 0 不阻断
    total = sum(float(m.q) for m in motors)
    if not np.isfinite(total):
        return 0
    return int(round(total))


class _FakeSubscriber:
    """ChannelSubscriber 替身：记录 handler 与关闭调用。"""

    def __init__(self, sdk: _FakeSdk, fail_init: bool) -> None:
        self._sdk = sdk
        self._fail_init = fail_init
        self.handler: Any = None
        self.queue_len: int | None = None
        self.close_called = 0

    def Init(self, handler: Any, queue_len: int = 0) -> None:
        self._sdk.events.append("subscriber_init")
        if self._fail_init:
            raise RuntimeError("fake subscriber init failure")
        self.handler = handler
        self.queue_len = queue_len

    def Close(self) -> None:
        self._sdk.events.append("subscriber_close")
        _probe_io_lock(self._sdk.close_io_lock)
        gate = self._sdk.subscriber_close_gate
        if gate is not None:
            gate.block_once()
        self.close_called += 1


class _FakePublisher:
    """ChannelPublisher 替身：记录发布消息与关闭调用（可注入阻塞/故障）。"""

    def __init__(self, sdk: _FakeSdk) -> None:
        self._sdk = sdk
        self.init_called = False
        self.messages: list[Any] = []
        self.close_called = 0

    def Init(self) -> None:
        self._sdk.events.append("publisher_init")
        self.init_called = True

    def Write(self, msg: Any) -> bool:
        self._sdk.events.append("publisher_write")
        self.messages.append(msg)
        gate = self._sdk.write_gate
        if gate is not None:
            gate.block_once()
        self._sdk.events.append("publisher_write_done")
        if self._sdk.publisher_fail_write:
            raise RuntimeError("fake write failure")
        return bool(self._sdk.publisher_write_result)

    def Close(self) -> None:
        self._sdk.events.append("publisher_close")
        _probe_io_lock(self._sdk.close_io_lock)
        gate = self._sdk.publisher_close_gate
        if gate is not None:
            gate.block_once()
        self._sdk.events.append("publisher_teardown_done")
        if self._sdk.publisher_fail_close:
            raise RuntimeError("fake publisher close failure")
        self.close_called += 1


class _FakeSwitcherClient:
    """MotionSwitcherClient 替身：脚本化 CheckMode/ReleaseMode。"""

    def __init__(self, sdk: _FakeSdk) -> None:
        self._sdk = sdk
        self.timeout: float | None = None
        self.init_called = False
        self.check_index = 0
        self.release_calls = 0

    def SetTimeout(self, timeout: float) -> None:
        self._sdk.events.append("switcher_set_timeout")
        self.timeout = timeout

    def Init(self) -> None:
        self._sdk.events.append("switcher_init")
        self.init_called = True

    def CheckMode(self) -> tuple[int, Any]:
        self._sdk.events.append("switcher_check")
        gate = self._sdk.check_mode_gate
        if gate is not None:
            gate.block_once()
        script = self._sdk.switcher_checks
        if not script:
            return 0, {"name": ""}
        index = min(self.check_index, len(script) - 1)
        self.check_index += 1
        return script[index]

    def ReleaseMode(self) -> tuple[int, Any]:
        self._sdk.events.append("switcher_release")
        self.release_calls += 1
        return self._sdk.switcher_release_code, None


class _FakeSdk:
    """fake SDK 总装：产出注入 SdkR1DTransport 的 _SdkBundle。

    Attributes:
        events: 调用事件序列（验证"先释放后建 publisher"等顺序契约）。
        factory_result: ChannelFactoryInitialize 返回值（False 模拟失败）。
        factory_raises: ChannelFactoryInitialize 抛出异常（模拟网卡错误）。
        switcher_checks: CheckMode 脚本 [(code, result), ...]。
        switcher_release_code: ReleaseMode 返回码。
    """

    def __init__(self) -> None:
        self.events: list[str] = []
        self.factory_domain: int | None = None
        self.factory_iface: str | None = None
        self.factory_result: Any = True
        self.factory_raises: Exception | None = None
        self.subscriber_fail_init = False
        self.publisher_write_result: Any = True
        self.publisher_fail_write = False
        self.publisher_fail_close = False
        self.switcher_checks: list[tuple[int, Any]] = []
        self.switcher_release_code = 0
        self.publishers: list[_FakePublisher] = []
        self.subscribers: list[_FakeSubscriber] = []
        self.switchers: list[_FakeSwitcherClient] = []
        # 并发测试钩子：闸门让指定 SDK 调用阻塞在途；close_io_lock 模拟
        # Close 内部回调线程需要 io 锁的跨线程竞争。
        self.check_mode_gate: _Gate | None = None
        self.write_gate: _Gate | None = None
        self.publisher_close_gate: _Gate | None = None
        self.subscriber_close_gate: _Gate | None = None
        self.close_io_lock: Any = None

    def bundle(self) -> _SdkBundle:
        sdk = self

        def _factory_initialize(domain: int, iface: str | None) -> Any:
            sdk.events.append("factory_init")
            sdk.factory_domain = domain
            sdk.factory_iface = iface
            if sdk.factory_raises is not None:
                raise sdk.factory_raises
            return sdk.factory_result

        def _create_subscriber(topic: str, msg_type: Any) -> _FakeSubscriber:
            sdk.events.append(f"subscriber_new:{topic}")
            sub = _FakeSubscriber(sdk, fail_init=sdk.subscriber_fail_init)
            sdk.subscribers.append(sub)
            return sub

        def _create_publisher(topic: str, msg_type: Any) -> _FakePublisher:
            sdk.events.append(f"publisher_new:{topic}")
            pub = _FakePublisher(sdk)
            sdk.publishers.append(pub)
            return pub

        def _create_switcher() -> _FakeSwitcherClient:
            sdk.events.append("switcher_new")
            client = _FakeSwitcherClient(sdk)
            sdk.switchers.append(client)
            return client

        return _SdkBundle(
            channel_factory_initialize=_factory_initialize,
            lowcmd_type=_FakeLowCmd,
            lowstate_type=object,
            motor_cmd_factory=lambda **kwargs: _FakeMotorCmd(**kwargs),
            crc_compute=_fake_crc,
            create_publisher=_create_publisher,
            create_subscriber=_create_subscriber,
            create_switcher=_create_switcher,
        )


def _make_transport(sdk: _FakeSdk, **overrides: Any) -> SdkR1DTransport:
    """用 fake bundle 构造 SdkR1DTransport（不触发真实 SDK import）。"""
    params: dict[str, Any] = dict(
        dds_domain=0,
        network_interface="fake0",
        switcher_timeout_s=0.1,
        acquire_timeout_s=0.5,
        release_retry_interval_s=0.0,
        clock=_FakeClock(),
        bundle=sdk.bundle(),
    )
    params.update(overrides)
    return SdkR1DTransport(**params)


def _sentinel_command_frame() -> R1DCommandFrame:
    """哨兵命令帧（float32 可精确表示值；测试专用，非实机参数）。"""
    n = R1D_SLOT_COUNT
    return R1DCommandFrame(
        q=np.array([0.25 * (i + 1) for i in range(n)]),
        dq=np.array([-0.125 * (i + 1) for i in range(n)]),
        kp=np.array([float(i + 1) for i in range(n)]),
        kd=np.array([float(i % 7) for i in range(n)]),
        tau=np.array([0.5 * (i + 1) for i in range(n)]),
        motor_mode=np.array([float(i % 2) for i in range(n)]),
        mode_machine=7,
        mode_pr=0,
    )


def _deliver_state(
    subscriber: _FakeSubscriber,
    *,
    tick: int = 42,
    mode_machine: int = 7,
    corrupt_crc: bool = False,
    nan_q: bool = False,
) -> None:
    """经 fake 订阅回调投递一帧状态（可注入 CRC 错误/NaN）。"""
    q = np.array([0.5 * (i + 1) for i in range(R1D_SLOT_COUNT)])
    if nan_q:
        q[3] = np.nan
    dq = np.array([-0.25 * (i + 1) for i in range(R1D_SLOT_COUNT)])
    msg = _FakeLowState(tick=tick, mode_machine=mode_machine, q=q, dq=dq, crc=0)
    msg.crc = 0 if corrupt_crc else _fake_crc(msg)
    assert subscriber.handler is not None
    subscriber.handler(msg)


# ---------------------------------------------------------------------------
# 懒加载与依赖隔离
# ---------------------------------------------------------------------------


def test_module_import_has_no_sdk_or_dds_side_effect() -> None:
    """模块 import 零副作用：不加载 unitree_sdk2py/cyclonedds（子进程隔离验证）。

    用独立子进程而非进程内断言，避免同进程其他测试模块（如
    test_r1d_sdk_serialization.py）先行 import SDK 造成假阳性/假阴性。
    """
    code = (
        "import sys\n"
        "import src.real.r1d_transport\n"
        "leaked = [m for m in sys.modules if m == 'unitree_sdk2py'\n"
        "          or m.startswith('unitree_sdk2py.') or m == 'cyclonedds'\n"
        "          or m.startswith('cyclonedds.')]\n"
        "assert not leaked, leaked\n"
        "print('PURE_OK')\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert "PURE_OK" in result.stdout


@contextlib.contextmanager
def _sdk_modules_blocked():
    """屏蔽 unitree_sdk2py 模块树，模拟无 SDK 机器（测试专用）。

    sys.modules[名字] = None 会让 import 该名字（含 from-import 父模块）
    立即触发 ImportError，等价于未安装。
    """
    saved: dict[str, Any] = {}
    for key in [
        k
        for k in sys.modules
        if k == "unitree_sdk2py" or k.startswith("unitree_sdk2py.")
    ]:
        saved[key] = sys.modules.pop(key)
    sys.modules["unitree_sdk2py"] = None
    try:
        yield
    finally:
        sys.modules.pop("unitree_sdk2py", None)
        sys.modules.update(saved)


def test_constructor_without_sdk_reports_dependency_error() -> None:
    """无 SDK 机器上真实构造（bundle=None）报明确依赖错误 R1DDependencyError。"""
    with _sdk_modules_blocked():
        with pytest.raises(R1DDependencyError, match="unitree_sdk2py"):
            SdkR1DTransport(dds_domain=0, network_interface="fake0")


def test_transport_implements_r1d_transport_protocol() -> None:
    """SdkR1DTransport 满足 R1DTransport Protocol（runtime_checkable）。"""
    transport = _make_transport(_FakeSdk())
    assert isinstance(transport, R1DTransport)


def test_constructor_rejects_invalid_params() -> None:
    """构造参数校验：负 domain、空网卡字符串、非正超时均拒绝。"""
    sdk = _FakeSdk()
    with pytest.raises(ValueError, match="dds_domain"):
        _make_transport(sdk, dds_domain=-1)
    with pytest.raises(ValueError, match="network_interface"):
        _make_transport(sdk, network_interface="")
    with pytest.raises(ValueError, match="acquire_timeout_s"):
        _make_transport(sdk, acquire_timeout_s=0.0)
    with pytest.raises(ValueError, match="switcher_timeout_s"):
        _make_transport(sdk, switcher_timeout_s=float("nan"))
    with pytest.raises(ValueError, match="release_retry_interval_s"):
        _make_transport(sdk, release_retry_interval_s=-0.5)


# ---------------------------------------------------------------------------
# open：只订阅、失败清理、ChannelFactory/网卡错误转换
# ---------------------------------------------------------------------------


def test_open_subscribes_only_without_acquire_or_publish() -> None:
    """open 只订阅状态：不建 publisher、不建 MotionSwitcher、不发布。"""
    sdk = _FakeSdk()
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    assert sdk.factory_domain == 0
    assert sdk.factory_iface == "fake0"
    assert len(sdk.subscribers) == 1
    assert sdk.subscribers[0].handler is not None
    assert sdk.subscribers[0].queue_len is not None and sdk.subscribers[0].queue_len > 0
    assert sdk.publishers == []
    assert sdk.switchers == []
    assert "publisher_new" not in " ".join(sdk.events)
    assert "switcher_new" not in " ".join(sdk.events)


def test_open_rejects_second_open_and_non_callable() -> None:
    """open 幂等约束：重复 open/非回调入参拒绝。"""
    sdk = _FakeSdk()
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    with pytest.raises(R1DTransportUsageError, match="open"):
        transport.open(lambda frame: None)
    transport2 = _make_transport(_FakeSdk())
    with pytest.raises(ValueError, match="on_state"):
        transport2.open("not-callable")  # type: ignore[arg-type]


def test_open_channel_factory_failure_converted() -> None:
    """ChannelFactory 初始化失败 → 明确 R1DChannelError，零残留资源。

    A-M2（评审 2026-11-17）修正：SDK ChannelFactoryInitialize 失败路径
    **自抛异常**、成功恒返 None（channel.py:298 无 return）——原"返回
    False"模拟与 SDK 实际语义不符，传输层死代码分支已删；失败一律经
    异常路径转换。
    """
    sdk = _FakeSdk()
    sdk.factory_raises = RuntimeError("channel factory init error.")
    transport = _make_transport(sdk)
    with pytest.raises(R1DChannelError, match="ChannelFactory"):
        transport.open(lambda frame: None)
    assert sdk.subscribers == []
    transport.close()  # 失败后 close 幂等可用


def test_open_network_interface_error_converted() -> None:
    """网卡错误（ChannelFactoryInitialize 抛异常）→ 转换为 R1DChannelError。"""
    sdk = _FakeSdk()
    sdk.factory_raises = RuntimeError("bad network interface: no_such_if")
    transport = _make_transport(sdk)
    with pytest.raises(R1DChannelError, match="网卡|network|ChannelFactory"):
        transport.open(lambda frame: None)


def test_open_subscriber_failure_cleans_up_partial_resources() -> None:
    """订阅初始化失败 → 清理已建资源（不留半初始化），异常转换为 R1DChannelError。"""
    sdk = _FakeSdk()
    sdk.subscriber_fail_init = True
    transport = _make_transport(sdk)
    with pytest.raises(R1DChannelError, match="lowstate|订阅"):
        transport.open(lambda frame: None)
    # 已创建的 subscriber 对象被关闭
    assert len(sdk.subscribers) == 1
    assert sdk.subscribers[0].close_called == 1
    # 失败后可正常 close（幂等），也可重新 open
    transport.close()
    sdk.subscriber_fail_init = False
    transport.open(lambda frame: None)
    assert len(sdk.subscribers) == 2
    assert sdk.subscribers[1].handler is not None


# ---------------------------------------------------------------------------
# 状态回调：CRC 校验、帧转换、异常吞噬、会话世代门
# ---------------------------------------------------------------------------


def test_state_callback_converts_and_delivers_frame() -> None:
    """合法状态帧 → CRC 校验 → R1DStateFrame（tick→sequence，数组只读快照）。"""
    sdk = _FakeSdk()
    clock = _FakeClock(start=50.0)
    transport = _make_transport(sdk, clock=clock)
    received: list[R1DStateFrame] = []
    transport.open(received.append)
    _deliver_state(sdk.subscribers[0], tick=42, mode_machine=7)
    assert len(received) == 1
    frame = received[0]
    assert isinstance(frame, R1DStateFrame)
    np.testing.assert_allclose(
        frame.q, [0.5 * (i + 1) for i in range(R1D_SLOT_COUNT)], atol=1e-12
    )
    np.testing.assert_allclose(
        frame.dq, [-0.25 * (i + 1) for i in range(R1D_SLOT_COUNT)], atol=1e-12
    )
    assert frame.sequence == 42  # sequence 承接 LowState_.tick
    assert frame.source_time is None  # 协议无源时间戳，不得编造
    assert frame.mode_machine == 7
    assert frame.received_at == pytest.approx(50.0)
    assert frame.session_id == 1
    # 消息边界复制：交付的是只读快照
    assert frame.q.flags.writeable is False
    assert frame.dq.flags.writeable is False


def test_state_callback_drops_crc_mismatch() -> None:
    """CRC 校验失败帧丢弃（manifest dds.crc.verify_on_receive）。"""
    sdk = _FakeSdk()
    transport = _make_transport(sdk)
    received: list[R1DStateFrame] = []
    transport.open(received.append)
    _deliver_state(sdk.subscribers[0], corrupt_crc=True)
    assert received == []


@pytest.mark.parametrize("bad_kind", ["missing", "short"])
def test_state_callback_rejects_bad_motor_state(bad_kind: str) -> None:
    """A-M4（评审 2026-11-17）：motor_state 缺失/槽数≠35 的帧被拒且不外泄。

    防御路径（绕过 SDK 类型保证的异常消息）：missing → None；
    short → 34 槽。转换抛 R1DTransportError，被回调链路吞噬成丢帧。
    """
    sdk = _FakeSdk()
    transport = _make_transport(sdk)
    received: list[R1DStateFrame] = []
    transport.open(received.append)
    msg = _FakeLowState(
        tick=42,
        mode_machine=7,
        q=np.zeros(R1D_SLOT_COUNT),
        dq=np.zeros(R1D_SLOT_COUNT),
        crc=0,
    )
    if bad_kind == "missing":
        msg.motor_state = None  # type: ignore[assignment]
    else:
        msg.motor_state = msg.motor_state[: R1D_SLOT_COUNT - 1]
    msg.crc = _fake_crc(msg)
    assert sdk.subscribers[0].handler is not None
    sdk.subscribers[0].handler(msg)  # 不得抛出（异常吞噬进日志）
    assert received == []


def test_state_callback_swallows_user_exception_and_nan() -> None:
    """回调链路异常不外泄：用户回调抛错 / 状态含 NaN 均丢弃且不抛出。"""
    sdk = _FakeSdk()
    transport = _make_transport(sdk)

    def bad_callback(_frame: R1DStateFrame) -> None:
        raise RuntimeError("user callback failure")

    transport.open(bad_callback)
    # 用户回调异常被吞噬（不得传入 DDS 监听线程）
    _deliver_state(sdk.subscribers[0])
    # NaN 状态帧被有限性校验拒绝并丢弃
    received: list[R1DStateFrame] = []
    transport.close()
    transport.open(received.append)
    _deliver_state(sdk.subscribers[0], nan_q=True)
    assert received == []


def test_close_blocks_stale_callbacks_across_sessions() -> None:
    """会话世代门：close 后与重开后的旧 handler 回调不再投递。"""
    sdk = _FakeSdk()
    transport = _make_transport(sdk)
    received_first: list[R1DStateFrame] = []
    transport.open(received_first.append)
    stale_handler = sdk.subscribers[0].handler
    transport.close()
    # close 后旧回调（模拟在途/迟到帧）不得写任何会话
    _deliver_state(sdk.subscribers[0])
    assert received_first == []
    # 重开生成新会话；旧 handler 属旧世代，仍被拒
    received_second: list[R1DStateFrame] = []
    transport.open(received_second.append)
    assert stale_handler is not sdk.subscribers[1].handler
    assert sdk.subscribers[0].handler is not sdk.subscribers[1].handler
    _deliver_state(sdk.subscribers[0])  # 旧世代 handler
    assert received_second == []
    _deliver_state(sdk.subscribers[1])  # 新世代 handler
    assert len(received_second) == 1
    assert received_second[0].session_id == 2


def test_close_idempotent_and_safe_when_never_opened() -> None:
    """close 幂等：多次调用、未 open 时调用均安全。"""
    transport = _make_transport(_FakeSdk())
    transport.close()
    transport.close()
    sdk = _FakeSdk()
    transport2 = _make_transport(sdk)
    transport2.open(lambda frame: None)
    transport2.close()
    transport2.close()
    assert " ".join(sdk.events).count("subscriber_close") == 1


# ---------------------------------------------------------------------------
# acquire_control：MotionSwitcher 流程编排（CheckMode→ReleaseMode→重查）
# ---------------------------------------------------------------------------


def test_acquire_occupied_then_release_then_free() -> None:
    """占用→释放→重查空闲→才建 publisher（manifest control.acquisition 流程）。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": "main"}), (0, {"name": ""})]
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    transport.acquire_control()
    assert len(sdk.switchers) == 1
    assert sdk.switchers[0].release_calls == 1
    assert len(sdk.publishers) == 1
    events = " ".join(sdk.events)
    # publisher 建立在最后一次 switcher_check 之后
    assert events.index("publisher_new:rt/lowcmd") > events.rindex("switcher_check")
    assert sdk.switchers[0].timeout == pytest.approx(0.1)
    assert sdk.publishers[0].init_called


def test_acquire_free_immediately_no_release() -> None:
    """空闲（name 为空）直接建 publisher，不调用 ReleaseMode。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": ""})]
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    transport.acquire_control()
    assert sdk.switchers[0].release_calls == 0
    assert len(sdk.publishers) == 1


def test_acquire_checkmode_bad_code_raises() -> None:
    """CheckMode 返回非零码 → R1DControlError，不建 publisher。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(7, None)]
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    with pytest.raises(R1DControlError, match="CheckMode"):
        transport.acquire_control()
    assert sdk.publishers == []


def test_acquire_checkmode_malformed_result_raises() -> None:
    """返回码 0 但结果结构异常（None/缺 name）→ 拒绝，不无条件当成功。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, None)]
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    with pytest.raises(R1DControlError, match="CheckMode"):
        transport.acquire_control()

    sdk2 = _FakeSdk()
    sdk2.switcher_checks = [(0, {"unexpected": 1})]
    transport2 = _make_transport(sdk2)
    transport2.open(lambda frame: None)
    with pytest.raises(R1DControlError, match="CheckMode"):
        transport2.acquire_control()
    assert sdk2.publishers == []


def test_acquire_releasemode_failure_raises_without_publisher() -> None:
    """ReleaseMode 失败 → R1DControlError（未获控制权、未发布任何命令）。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": "main"})]
    sdk.switcher_release_code = -1
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    with pytest.raises(R1DControlError, match="ReleaseMode"):
        transport.acquire_control()
    assert sdk.publishers == []


def test_acquire_timeout_while_persistently_occupied() -> None:
    """持续占用下超时 → R1DControlError 且未建 publisher（失败回滚要求）。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": "main"})]
    transport = _make_transport(sdk, acquire_timeout_s=0.05)
    transport.open(lambda frame: None)
    with pytest.raises(R1DControlError, match="超时|timeout"):
        transport.acquire_control()
    assert sdk.publishers == []
    assert sdk.switchers[0].release_calls >= 1


def test_acquire_requires_open_first_and_rejects_repeat() -> None:
    """未 open 先 acquire 拒绝；重复 acquire 拒绝。"""
    sdk = _FakeSdk()
    transport = _make_transport(sdk)
    with pytest.raises(R1DTransportUsageError):
        transport.acquire_control()
    transport.open(lambda frame: None)
    transport.acquire_control()
    with pytest.raises(R1DTransportUsageError):
        transport.acquire_control()


# ---------------------------------------------------------------------------
# publish / release_control / close 故障分支
# ---------------------------------------------------------------------------


def _acquired_transport(sdk: _FakeSdk) -> SdkR1DTransport:
    """返回已 open+acquire 的传输（fake，空闲模式）。"""
    sdk.switcher_checks = [(0, {"name": ""})]
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    transport.acquire_control()
    return transport


def test_publish_requires_acquire() -> None:
    """未 acquire 先 publish 拒绝（open 只读语义）。"""
    sdk = _FakeSdk()
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    with pytest.raises(R1DTransportUsageError, match="acquire"):
        transport.publish(_sentinel_command_frame())


def test_publish_builds_explicit_lowcmd_and_sets_crc() -> None:
    """publish 构造显式全字段 LowCmd（35 槽逐字段转录 + mode 头字段 + CRC）。"""
    sdk = _FakeSdk()
    transport = _acquired_transport(sdk)
    frame = _sentinel_command_frame()
    transport.publish(frame)
    assert len(sdk.publishers[0].messages) == 1
    msg = sdk.publishers[0].messages[0]
    assert isinstance(msg, _FakeLowCmd)
    assert msg.mode_pr == 0
    assert msg.mode_machine == 7
    assert msg.reserve == [0, 0, 0, 0]
    assert len(msg.motor_cmd) == R1D_SLOT_COUNT
    for i in range(R1D_SLOT_COUNT):
        motor = msg.motor_cmd[i]
        assert motor.mode == i % 2
        assert motor.q == pytest.approx(0.25 * (i + 1))
        assert motor.dq == pytest.approx(-0.125 * (i + 1))
        assert motor.tau == pytest.approx(0.5 * (i + 1))
        assert motor.kp == pytest.approx(float(i + 1))
        assert motor.kd == pytest.approx(float(i % 7))
        assert motor.reserve == 0
    assert msg.crc == _fake_crc(msg)


def test_publish_write_failure_raises() -> None:
    """发布失败（Write 返回 False）→ R1DPublishError。"""
    sdk = _FakeSdk()
    transport = _acquired_transport(sdk)
    sdk.publisher_write_result = False
    with pytest.raises(R1DPublishError):
        transport.publish(_sentinel_command_frame())


def test_publish_write_exception_converted() -> None:
    """发布异常（Write 抛出）→ 转换为 R1DPublishError，不吞成成功。"""
    sdk = _FakeSdk()
    transport = _acquired_transport(sdk)
    sdk.publisher_fail_write = True
    with pytest.raises(R1DPublishError, match="fake write failure"):
        transport.publish(_sentinel_command_frame())


def test_publish_rejects_wrong_frame_type() -> None:
    """publish 只接受 R1DCommandFrame（防止裸数组/错误类型静默序列化）。"""
    sdk = _FakeSdk()
    transport = _acquired_transport(sdk)
    with pytest.raises(ValueError, match="R1DCommandFrame"):
        transport.publish([0.0] * 35)  # type: ignore[arg-type]


def test_publish_after_close_rejected() -> None:
    """close 后 publish 拒绝。"""
    sdk = _FakeSdk()
    transport = _acquired_transport(sdk)
    transport.close()
    with pytest.raises(R1DTransportUsageError):
        transport.publish(_sentinel_command_frame())


def test_release_control_closes_publisher_and_is_idempotent() -> None:
    """release_control 关闭 publisher（不触发未验证 RPC），幂等；随后 publish 拒绝。"""
    sdk = _FakeSdk()
    transport = _acquired_transport(sdk)
    transport.release_control()
    assert " ".join(sdk.events).count("publisher_close") == 1
    # 未验证的机载服务恢复路径（manifest control.release.mechanism=null）
    # 不得调用 SelectMode 之类 RPC —— fake 只可能记录我们实现的调用
    assert "switcher_select" not in " ".join(sdk.events)
    transport.release_control()  # 幂等
    assert " ".join(sdk.events).count("publisher_close") == 1
    with pytest.raises(R1DTransportUsageError):
        transport.publish(_sentinel_command_frame())


def test_release_control_close_failure_converted() -> None:
    """释放失败（publisher.Close 抛出）→ R1DControlError。"""
    sdk = _FakeSdk()
    transport = _acquired_transport(sdk)
    sdk.publisher_fail_close = True
    with pytest.raises(R1DControlError):
        transport.release_control()


def test_close_closes_both_channels_after_acquire() -> None:
    """acquire 后 close 同时关闭 subscriber 与 publisher。"""
    sdk = _FakeSdk()
    transport = _acquired_transport(sdk)
    transport.close()
    events = " ".join(sdk.events)
    assert "publisher_close" in events
    assert "subscriber_close" in events
    # close 后不得再 acquire
    with pytest.raises(R1DTransportUsageError):
        transport.acquire_control()


# ---------------------------------------------------------------------------
# 红线（P1-1）：会话隔离 / 资源拆卸 / 在途发布竞态
# ---------------------------------------------------------------------------


def test_publish_in_flight_cannot_write_new_session_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """红线：旧会话 publish 组包期间 close/重开，旧线程不得写新会话 publisher。

    旧实现（基线 9a5b3f7）：close 立即返回并完成重开，组包放行后旧线程读取
    已被替换的 self._publisher，把旧会话命令写进新会话。
    修复后：publish 在 _io_lock 内完成 状态检查→组包→CRC→Write，close 必须
    等这次发布结束才能拆资源，因此旧线程只可能写旧 publisher。
    """
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": ""})]
    transport = _make_transport(sdk)
    io_lock = _ObservableRLock()
    lifecycle_lock = _ObservableRLock()
    transport._io_lock = io_lock  # type: ignore[attr-defined]
    transport._lifecycle_lock = lifecycle_lock  # type: ignore[attr-defined]

    packing_entered = threading.Event()
    packing_release = threading.Event()
    real_build = r1d_transport_module._build_lowcmd

    def slow_build(command: R1DCommandFrame, bundle: Any) -> Any:
        """阻塞组包阶段：放大 "检查后、写入前" 的竞态窗口。"""
        packing_entered.set()
        if not packing_release.wait(timeout=_GATE_TIMEOUT_S):
            raise RuntimeError("组包放行等待超时")
        return real_build(command, bundle)

    monkeypatch.setattr(r1d_transport_module, "_build_lowcmd", slow_build)

    transport.open(lambda frame: None)
    transport.acquire_control()
    old_publisher = sdk.publishers[-1]
    frame = _sentinel_command_frame()

    publish_errors: list[Exception] = []
    publish_thread = _thread_with_errors(
        lambda: transport.publish(frame), publish_errors, name="old-session-publish"
    )
    reopened = threading.Event()
    reopen_errors: list[Exception] = []

    def reopen_session() -> None:
        try:
            transport.close()
            transport.open(lambda frame: None)
            transport.acquire_control()
        finally:
            reopened.set()

    reopen_thread = _thread_with_errors(
        reopen_session, reopen_errors, name="reopen-session"
    )
    try:
        publish_thread.start()
        assert packing_entered.wait(timeout=_GATE_TIMEOUT_S), (
            "旧 publish 未进入组包阶段"
        )
        reopen_thread.start()
        # 观测：旧实现重开已完成；修复后 close 已开始争抢 io（写入证据在下方断言）
        assert _wait_until(
            lambda: reopened.is_set() or io_lock.contended.is_set(), 5.0
        ), "重开线程既未完成也未争抢 io"
        packing_release.set()
    finally:
        packing_release.set()
        _join_threads(publish_thread, reopen_thread)

    assert publish_errors == [], publish_errors
    assert reopen_errors == [], reopen_errors
    new_publisher = sdk.publishers[-1]
    assert new_publisher is not old_publisher
    assert len(old_publisher.messages) == 1, "旧会话命令未写入旧 publisher"
    assert new_publisher.messages == [], "旧会话命令越界写入新会话 publisher"


def test_stale_acquire_does_not_install_publisher_after_reopen() -> None:
    """红线：重开后迟到的旧 acquire 不得把它的 publisher 装进新会话。

    旧实现：旧 acquire 完成 CheckMode/建 publisher 后无条件写 self._publisher，
    覆盖新会话已安装的 publisher（旧资源泄漏且新会话被劫持）。
    修复后：提交点校验 opened/会话/票据/acquired/publisher，陈旧者自行 Close
    并抛 R1DControlError（并发语义=最新一次 acquire 胜出）。
    """
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": ""})]
    gate = _Gate()
    sdk.check_mode_gate = gate
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)

    stale_errors: list[Exception] = []
    stale_thread = _thread_with_errors(
        transport.acquire_control, stale_errors, name="stale-acquire"
    )
    try:
        stale_thread.start()
        assert gate.entered.wait(timeout=_GATE_TIMEOUT_S), "旧 acquire 未进入 CheckMode"
        transport.close()
        transport.open(lambda frame: None)
        transport.acquire_control()
        new_publisher = sdk.publishers[-1]
        gate.release.set()
    finally:
        gate.release.set()
        _join_threads(stale_thread)

    assert len(stale_errors) == 1, stale_errors
    assert isinstance(stale_errors[0], R1DControlError)
    assert transport._publisher is new_publisher  # type: ignore[attr-defined]
    assert new_publisher.messages == []
    stale_publisher = sdk.publishers[-1]
    assert stale_publisher is not new_publisher
    assert stale_publisher.close_called == 1


def test_open_waits_until_previous_close_tears_down() -> None:
    """红线：close 的 SDK teardown 未完成前，open 不得创建新订阅。

    旧实现：close 在 subscriber.Close() 处阻塞时已清空 _opened/_subscriber，
    并发的 open 立即创建新订阅（新旧资源在途交叠、生命周期状态不一致）。
    修复后：open 全程持 lifecycle 锁，必须等 close 的 teardown 返回。
    """
    sdk = _FakeSdk()
    lifecycle_lock = _ObservableRLock()
    transport = _make_transport(sdk)
    transport._lifecycle_lock = lifecycle_lock  # type: ignore[attr-defined]
    transport.open(lambda frame: None)
    gate = _Gate()
    sdk.subscriber_close_gate = gate

    close_errors: list[Exception] = []
    close_thread = _thread_with_errors(
        transport.close, close_errors, name="teardown-close"
    )
    open_errors: list[Exception] = []
    open_thread = _thread_with_errors(
        lambda: transport.open(lambda frame: None), open_errors, name="concurrent-open"
    )
    try:
        close_thread.start()
        assert gate.entered.wait(timeout=_GATE_TIMEOUT_S), (
            "close 未进入 subscriber teardown"
        )
        open_thread.start()
        # 观测 open 是否已尝试取 lifecycle 锁；正确性由资源数量/顺序断言判定
        lifecycle_lock.contended.wait(timeout=5.0)
        assert len(sdk.subscribers) == 1, "teardown 未完成前 open 已创建新订阅"
        assert transport._subscriber is None  # type: ignore[attr-defined]
    finally:
        gate.release.set()
        _join_threads(close_thread, open_thread)

    assert close_errors == [], close_errors
    assert open_errors == [], open_errors
    assert len(sdk.subscribers) == 2
    events = sdk.events
    close_index = events.index("subscriber_close")
    new_subscriber_indices = [
        i for i, event in enumerate(events) if event.startswith("subscriber_new:")
    ]
    assert new_subscriber_indices
    assert close_index < new_subscriber_indices[-1], "新订阅早于旧订阅 teardown 完成"


def test_publish_entry_window_does_not_write_new_session_publisher() -> None:
    """fix1 红线：publish 已进入方法、尚未取得 io 时被 close→open→acquire 切走。

    旧实现（71bdab3）：publish 在取 _io_lock 前不读 _session_serial，重开完成后
    旧线程取得 io，看到新会话的 opened/acquired/publisher 全部成立，越界把旧会话
    命令写进新会话 publisher。
    修复后：publish 在取 io 前快照 session，io 内复查 session != _session_serial
    即抛 R1DTransportUsageError（消息含“会话已更换”），新 publisher 无任何消息。
    """
    sdk = _FakeSdk()
    transport = _acquired_transport(sdk)
    old_publisher = sdk.publishers[-1]
    gate = _Gate()
    entry_lock = _ThreadGatedLock("stale-publish-entry", gate)
    transport._io_lock = entry_lock  # type: ignore[attr-defined]

    publish_errors: list[Exception] = []
    publish_thread = _thread_with_errors(
        lambda: transport.publish(_sentinel_command_frame()),
        publish_errors,
        name="stale-publish-entry",
    )
    try:
        publish_thread.start()
        assert gate.entered.wait(timeout=_GATE_TIMEOUT_S), "旧 publish 未停在 io 锁入口"
        # 旧 publish 已进入方法、尚未取得 io：主线程完成 close→open→acquire
        transport.close()
        transport.open(lambda frame: None)
        transport.acquire_control()
        new_publisher = sdk.publishers[-1]
        assert new_publisher is not old_publisher
        gate.release.set()
    finally:
        gate.release.set()
        _join_threads(publish_thread)

    assert len(publish_errors) == 1, publish_errors
    assert isinstance(publish_errors[0], R1DTransportUsageError)
    assert "会话已更换" in str(publish_errors[0])
    assert new_publisher.messages == [], "旧会话命令越界写入新会话 publisher"
    assert old_publisher.messages == []
    assert transport._publisher is new_publisher  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 边界与异常（P1-1）：票据失效 / teardown 等待 / 局部失败 / 告警阈值
# ---------------------------------------------------------------------------


def test_release_invalidates_pending_acquire() -> None:
    """release 使阻塞在 RPC 中的 acquire 票据失效，后者自建 publisher 自行关闭。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": ""})]
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    gate = _Gate()
    sdk.check_mode_gate = gate

    acquire_errors: list[Exception] = []
    acquire_thread = _thread_with_errors(
        transport.acquire_control, acquire_errors, name="pending-acquire"
    )
    try:
        acquire_thread.start()
        assert gate.entered.wait(timeout=_GATE_TIMEOUT_S), "acquire 未进入 CheckMode"
        transport.release_control()  # 未持有控制权：递增票据后幂等返回
        gate.release.set()
    finally:
        gate.release.set()
        _join_threads(acquire_thread)

    assert len(acquire_errors) == 1, acquire_errors
    assert isinstance(acquire_errors[0], R1DControlError)
    assert transport._publisher is None  # type: ignore[attr-defined]
    assert transport._acquired is False  # type: ignore[attr-defined]
    assert len(sdk.publishers) == 1
    assert sdk.publishers[0].close_called == 1
    assert sdk.publishers[0].messages == []


def test_acquire_commit_waits_for_release_teardown() -> None:
    """release 的 publisher.Close 未返回前，acquire 不得创建/安装新 publisher。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": ""})]
    transport = _make_transport(sdk)
    lifecycle_lock = _ObservableRLock()
    transport._lifecycle_lock = lifecycle_lock  # type: ignore[attr-defined]
    transport.open(lambda frame: None)
    transport.acquire_control()
    old_publisher = sdk.publishers[-1]
    gate = _Gate()
    sdk.publisher_close_gate = gate

    release_errors: list[Exception] = []
    release_thread = _thread_with_errors(
        transport.release_control, release_errors, name="release-teardown"
    )
    acquire_errors: list[Exception] = []
    acquire_thread = _thread_with_errors(
        transport.acquire_control, acquire_errors, name="waiting-acquire"
    )
    try:
        release_thread.start()
        assert gate.entered.wait(timeout=_GATE_TIMEOUT_S), (
            "release 未进入 publisher.Close"
        )
        acquire_thread.start()
        lifecycle_lock.contended.wait(timeout=5.0)
        assert len(sdk.publishers) == 1, (
            "旧 publisher teardown 未完成即创建新 publisher"
        )
    finally:
        gate.release.set()
        _join_threads(release_thread, acquire_thread)

    assert release_errors == [], release_errors
    assert acquire_errors == [], acquire_errors
    events = sdk.events
    new_publisher_indices = [
        i for i, event in enumerate(events) if event == "publisher_new:rt/lowcmd"
    ]
    assert len(new_publisher_indices) == 2
    assert new_publisher_indices[-1] > events.index("publisher_teardown_done")
    new_publisher = sdk.publishers[-1]
    assert new_publisher is not old_publisher
    assert new_publisher.init_called
    assert old_publisher.close_called == 1
    assert transport._publisher is new_publisher  # type: ignore[attr-defined]
    assert transport._acquired is True  # type: ignore[attr-defined]


def test_acquire_cannot_commit_during_failed_open() -> None:
    """在途 acquire 期间 close 且重新 open 失败：陈旧 acquire 不得留下 publisher。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": ""})]
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    gate = _Gate()
    sdk.check_mode_gate = gate

    acquire_errors: list[Exception] = []
    acquire_thread = _thread_with_errors(
        transport.acquire_control, acquire_errors, name="stale-acquire"
    )
    try:
        acquire_thread.start()
        assert gate.entered.wait(timeout=_GATE_TIMEOUT_S), "acquire 未进入 CheckMode"
        transport.close()
        sdk.subscriber_fail_init = True
        with pytest.raises(R1DChannelError):
            transport.open(lambda frame: None)
        gate.release.set()
    finally:
        gate.release.set()
        _join_threads(acquire_thread)

    assert len(acquire_errors) == 1, acquire_errors
    assert isinstance(acquire_errors[0], R1DControlError)
    assert transport._publisher is None  # type: ignore[attr-defined]
    assert transport._acquired is False  # type: ignore[attr-defined]
    assert transport._opened is False  # type: ignore[attr-defined]
    assert transport._on_state is None  # type: ignore[attr-defined]
    assert len(sdk.publishers) == 1
    assert sdk.publishers[0].close_called == 1
    assert sdk.publishers[0].messages == []


def test_close_waits_for_in_flight_write() -> None:
    """publish 的 Write 在途时 close 必须等它结束，再关闭 publisher。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": ""})]
    transport = _make_transport(sdk)
    io_lock = _ObservableRLock()
    transport._io_lock = io_lock  # type: ignore[attr-defined]
    transport.open(lambda frame: None)
    transport.acquire_control()
    publisher = sdk.publishers[-1]
    frame = _sentinel_command_frame()
    gate = _Gate()
    sdk.write_gate = gate

    publish_errors: list[Exception] = []
    publish_thread = _thread_with_errors(
        lambda: transport.publish(frame), publish_errors, name="in-flight-write"
    )
    close_errors: list[Exception] = []
    close_thread = _thread_with_errors(
        transport.close, close_errors, name="close-waiting"
    )
    try:
        publish_thread.start()
        assert gate.entered.wait(timeout=_GATE_TIMEOUT_S), "publish 未进入 Write"
        close_thread.start()
        io_lock.contended.wait(timeout=5.0)
        assert "publisher_close" not in sdk.events, (
            "Write 在途时 close 已关闭 publisher"
        )
        assert transport._publisher is publisher  # type: ignore[attr-defined]
    finally:
        gate.release.set()
        _join_threads(publish_thread, close_thread)

    assert publish_errors == [], publish_errors
    assert close_errors == [], close_errors
    events = sdk.events
    assert events.index("publisher_write_done") < events.index("publisher_close")
    assert len(publisher.messages) == 1
    assert publisher.close_called == 1
    assert transport._publisher is None  # type: ignore[attr-defined]


def test_release_and_close_close_publisher_once() -> None:
    """release 与 close 并发：publisher 只关闭一次，订阅仍被 close 关闭。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": ""})]
    transport = _make_transport(sdk)
    lifecycle_lock = _ObservableRLock()
    transport._lifecycle_lock = lifecycle_lock  # type: ignore[attr-defined]
    transport.open(lambda frame: None)
    transport.acquire_control()
    publisher = sdk.publishers[-1]
    subscriber = sdk.subscribers[-1]
    gate = _Gate()
    sdk.publisher_close_gate = gate

    release_errors: list[Exception] = []
    release_thread = _thread_with_errors(
        transport.release_control, release_errors, name="concurrent-release"
    )
    close_errors: list[Exception] = []
    close_thread = _thread_with_errors(
        transport.close, close_errors, name="concurrent-close"
    )
    try:
        release_thread.start()
        assert gate.entered.wait(timeout=_GATE_TIMEOUT_S), (
            "release 未进入 publisher.Close"
        )
        close_thread.start()
        lifecycle_lock.contended.wait(timeout=5.0)
        assert sdk.events.count("publisher_close") == 1
    finally:
        gate.release.set()
        _join_threads(release_thread, close_thread)

    assert release_errors == [], release_errors
    assert close_errors == [], close_errors
    assert sdk.events.count("publisher_close") == 1, "publisher 被重复关闭"
    assert sdk.events.count("subscriber_close") == 1
    assert publisher.close_called == 1
    assert subscriber.close_called == 1
    assert transport._publisher is None  # type: ignore[attr-defined]
    assert transport._subscriber is None  # type: ignore[attr-defined]


def test_close_closes_subscriber_even_if_publisher_close_raises() -> None:
    """publisher.Close 抛错时 close 仍关闭订阅，并聚合抛 R1DChannelError。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": ""})]
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    transport.acquire_control()
    sdk.publisher_fail_close = True

    with pytest.raises(R1DChannelError):
        transport.close()

    assert "subscriber_close" in sdk.events
    assert sdk.subscribers[-1].close_called == 1
    assert transport._publisher is None  # type: ignore[attr-defined]
    assert transport._subscriber is None  # type: ignore[attr-defined]
    assert transport._opened is False  # type: ignore[attr-defined]


def test_close_sdk_callbacks_can_acquire_io_lock() -> None:
    """SDK Close 内部回调线程可以取得 io 锁：close 不得持 io 调用 SDK Close。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": ""})]
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    transport.acquire_control()
    sdk.close_io_lock = transport._io_lock  # type: ignore[attr-defined]

    close_errors: list[Exception] = []
    close_thread = _thread_with_errors(
        transport.close, close_errors, name="close-vs-callback"
    )
    close_thread.start()
    _join_threads(close_thread)

    assert close_errors == [], close_errors
    assert "publisher_close" in sdk.events
    assert "subscriber_close" in sdk.events


@pytest.mark.parametrize(
    ("elapsed", "expect_warning"),
    [
        (0.050, False),
        (0.051, True),
    ],
)
def test_close_slow_warning_threshold(
    elapsed: float,
    expect_warning: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """close 入口耗时 >50ms 记录 warning（仅告警，仍正常返回）；0.050 不告警。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": ""})]
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    transport.acquire_control()
    transport._clock = _SequenceClock([0.0, elapsed])  # type: ignore[attr-defined]

    with caplog.at_level(logging.WARNING, logger="src.real.r1d_transport"):
        transport.close()  # 不抛错：告警不改变成功语义

    assert ("拆卸耗时" in caplog.text) is expect_warning, caplog.text
    assert transport._publisher is None  # type: ignore[attr-defined]
    assert transport._subscriber is None  # type: ignore[attr-defined]


def test_publish_exception_then_close_takes_io_lock() -> None:
    """Write 异常路径必须释放 io 锁，否则后续 close 无法取得锁。"""
    sdk = _FakeSdk()
    sdk.switcher_checks = [(0, {"name": ""})]
    transport = _make_transport(sdk)
    transport.open(lambda frame: None)
    transport.acquire_control()
    sdk.publisher_fail_write = True

    with pytest.raises(R1DPublishError):
        transport.publish(_sentinel_command_frame())

    close_errors: list[Exception] = []
    close_thread = _thread_with_errors(
        transport.close, close_errors, name="close-after-error"
    )
    close_thread.start()
    _join_threads(close_thread)

    assert close_errors == [], close_errors
    assert "subscriber_close" in sdk.events


# ---------------------------------------------------------------------------
# from_hardware_config：通信参数从硬件配置注入
# ---------------------------------------------------------------------------


def _test_hardware_config(iface: str | None = "fake0") -> R1DHardwareConfig:
    """测试专用硬件配置（结构合法、数值非实机）。"""
    slots = {f"left_joint{i + 1}": 15 + i for i in range(7)}
    return R1DHardwareConfig(
        schema_version=1,
        robot_id="unitree_r1d",
        side="left",
        communication=R1DCommunicationConfig(
            dds_domain=0,
            network_interface=iface,
            lowcmd_topic="rt/lowcmd",
            lowstate_topic="rt/lowstate",
        ),
        layout=R1DLayoutConfig(
            layout_verified=False, joint_slots=MappingProxyType(slots)
        ),
        calibration=R1DCalibrationConfig(
            verified=False, joint_signs=None, joint_offsets_rad=None
        ),
        limits=R1DLimitConfig(
            verified=False,
            q_lower_rad=None,
            q_upper_rad=None,
            dq_limit_rad_s=None,
            tau_limit_nm=None,
            kp_nm_per_rad=None,
            kd_nms_per_rad=None,
        ),
        timing=R1DTimingConfig(
            control_period_s=None, state_timeout_s=None, diagnostic_duration_s=None
        ),
    )


def test_from_hardware_config_injects_communication_params() -> None:
    """from_hardware_config：domain/网卡/话题从硬件配置注入（网卡现场才有）。"""
    sdk = _FakeSdk()
    transport = SdkR1DTransport.from_hardware_config(
        _test_hardware_config(iface="site_if0"),
        switcher_timeout_s=0.1,
        acquire_timeout_s=0.5,
        clock=_FakeClock(),
        bundle=sdk.bundle(),
    )
    transport.open(lambda frame: None)
    assert sdk.factory_domain == 0
    assert sdk.factory_iface == "site_if0"
    assert any(e.startswith("subscriber_new:rt/lowstate") for e in sdk.events)
    transport.close()
