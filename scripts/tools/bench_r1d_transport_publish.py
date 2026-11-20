"""R1D 传输层发布路径基准 — 对比冻结基线（9a5b3f7 无锁实现）与当前实现（离线，不碰 DDS）。

用途：
- 量化 `_io_lock`（发布临界区）对每拍发布的代价；
- 量化 `close()` 在“有在途 Write”时是否等待写入结束，给停止路径的预算提供依据；
- 自带最小 fake SDK，不 import `tests/`，不加载 unitree_sdk2py/cyclonedds。

用法：
    python scripts/tools/bench_r1d_transport_publish.py
    python scripts/tools/bench_r1d_transport_publish.py --pairs 100000 --rounds 3 --close-block-ms 200
    python scripts/tools/bench_r1d_transport_publish.py --json

对照对象：
- baseline：`_BaselineTransport`，其 `publish()`/`close()` 由基线提交
  `9a5b3f7` 的 `src/real/r1d_transport.py` 逐行等价复制（仅格式适配；旧的无锁
  实现：close 先交换资源再关闭，publish 不持锁），仅补了基准所需的最小
  open/acquire；
- locked：当前生产类 `SdkR1DTransport`（两把 RLock + 会话票据 + teardown 等待）。

自检（强制）：close 等待实验必须证明
- baseline.write_in_flight_at_close is True（旧实现 close 返回时 Write 仍在途，存在竞态窗口）；
- locked.write_in_flight_at_close is False（当前实现 close 等到 Write 结束）；
- 两边 writes == 1 且 errors == []。
任一自检失败、基准执行异常、或发布线程未在时限内退出时，main 返回非零退出码。

边界：这不是性能门禁，不进 pytest，也不证明实机时序；fake DDS 的数字不能外推到
真实 DDS。close 的 50ms 告警只统计返回耗时，不构成看门狗/超时语义。
"""

import argparse
import gc
import json
import math
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

# 仓库根（scripts/tools/<file> → parents[2]）
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402 - 需先插入仓库根到 sys.path

from src.real.r1d_transport import (  # noqa: E402 - 同上
    R1DChannelError,
    R1DPublishError,
    R1DTransportError,
    R1DTransportUsageError,
    SdkR1DTransport,
    _build_lowcmd,
    _SdkBundle,
)
from src.real.r1d_types import R1D_SLOT_COUNT, R1DCommandFrame  # noqa: E402 - 同上


# ---------------------------------------------------------------------------
# 最小 fake SDK（只实现发布路径触及的表面；不记录事件以降低噪声）
# ---------------------------------------------------------------------------


class _FakeMotorCmd:
    """MotorCmd_ 替身（显式全字段）。"""

    __slots__ = ("dq", "kd", "kp", "mode", "q", "reserve", "tau")

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
    """LowCmd_ 替身（显式全字段）。"""

    __slots__ = ("crc", "mode_machine", "mode_pr", "motor_cmd", "reserve")

    def __init__(
        self,
        *,
        mode_pr: int,
        mode_machine: int,
        motor_cmd: list[Any],
        reserve: list[int],
        crc: int,
    ) -> None:
        self.mode_pr = mode_pr
        self.mode_machine = mode_machine
        self.motor_cmd = motor_cmd
        self.reserve = reserve
        self.crc = crc


class _FakePublisher:
    """ChannelPublisher 替身：Write 可阻塞（用于 close 等待实验）。"""

    def __init__(self, block_seconds: float = 0.0) -> None:
        self.messages = 0
        self.block_seconds = block_seconds
        self.entered = threading.Event()
        self.release = threading.Event()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.completed = False

    def Init(self) -> None:
        pass

    def Write(self, msg: Any) -> bool:
        self.messages += 1
        self.started_at = time.perf_counter()
        try:
            if self.block_seconds > 0.0:
                self.entered.set()
                # Windows 的 Event.wait 粒度约 15.6 ms，这里用忙等拿到准确的阻塞时长。
                deadline = self.started_at + self.block_seconds
                while not self.release.is_set() and time.perf_counter() < deadline:
                    pass
            return True
        finally:
            self.finished_at = time.perf_counter()
            self.completed = True

    def Close(self) -> None:
        pass


class _FakeSubscriber:
    """ChannelSubscriber 替身（基准不投递状态）。"""

    def Init(self, handler: Any, queue_len: int = 0) -> None:
        self.handler = handler

    def Close(self) -> None:
        pass


class _FakeSwitcher:
    """MotionSwitcherClient 替身：始终空闲。"""

    def SetTimeout(self, timeout: float) -> None:
        pass

    def Init(self) -> None:
        pass

    def CheckMode(self) -> tuple[int, dict[str, str]]:
        return 0, {"name": ""}

    def ReleaseMode(self) -> tuple[int, None]:
        return 0, None


def _fake_bundle(publisher: _FakePublisher) -> _SdkBundle:
    """组装 fake 绑定：发布器复用同一个实例，便于统计写入数。"""
    return _SdkBundle(
        channel_factory_initialize=lambda domain, iface: True,
        lowcmd_type=_FakeLowCmd,
        lowstate_type=object,
        motor_cmd_factory=lambda **kwargs: _FakeMotorCmd(**kwargs),
        crc_compute=lambda msg: 0,
        create_publisher=lambda topic, msg_type: publisher,
        create_subscriber=lambda topic, msg_type: _FakeSubscriber(),
        create_switcher=lambda: _FakeSwitcher(),
    )


# ---------------------------------------------------------------------------
# 冻结基线：9a5b3f7 的旧 publish/close（无锁）
# ---------------------------------------------------------------------------


class _BaselineTransport:
    """基线对照：publish/close 逐行等价复制（仅格式适配）自提交 9a5b3f7 的旧无锁实现。

    来源：`git show 9a5b3f7:src/real/r1d_transport.py` 中的
    `SdkR1DTransport.publish`（行 649-679）与 `SdkR1DTransport.close`
    （行 683-714）。仅补充基准所需的最小 `__init__`/`open`/`acquire_control`，
    不复制锁、票据等当前实现语义——否则对照就失去意义。
    """

    def __init__(self, *, bundle: _SdkBundle) -> None:
        self._bundle = bundle
        self._subscriber: Any = None
        self._publisher: Any = None
        self._on_state: Any = None
        self._opened = False
        self._acquired = False

    def open(self, on_state: Any) -> None:
        """最小 open：建订阅并登记回调（基准不投递状态）。"""
        self._on_state = on_state
        self._subscriber = self._bundle.create_subscriber(
            "rt/lowstate", self._bundle.lowstate_type
        )
        self._subscriber.Init(lambda msg: None, 10)
        self._opened = True

    def acquire_control(self) -> None:
        """最小 acquire：直接建 publisher（基准不需要 RPC 流程）。"""
        publisher = self._bundle.create_publisher("rt/lowcmd", self._bundle.lowcmd_type)
        publisher.Init()
        self._publisher = publisher
        self._acquired = True

    def publish(self, command: R1DCommandFrame) -> None:
        """发布一帧完整整机命令（基线 9a5b3f7 无锁实现，逐行等价复制并仅格式适配）。

        Args:
            command: 完整 35 槽命令帧。

        Raises:
            ValueError: command 不是 R1DCommandFrame。
            R1DTransportUsageError: 未 acquire（或已 close）。
            R1DPublishError: Write 返回失败或抛出异常。
        """
        if not isinstance(command, R1DCommandFrame):
            raise ValueError(
                f"publish 只接受 R1DCommandFrame，实际 {type(command).__name__}"
            )
        if not self._acquired or self._publisher is None:
            raise R1DTransportUsageError(
                "未持有控制权，须先 acquire_control()（open 只读不发布）"
            )
        try:
            msg = _build_lowcmd(command, self._bundle)
            msg.crc = self._bundle.crc_compute(msg)
            ok = self._publisher.Write(msg)
        except R1DTransportError:
            raise
        except Exception as exc:
            raise R1DPublishError(f"lowcmd 发布异常: {exc}") from exc
        if not ok:
            raise R1DPublishError(
                "lowcmd Write 返回失败（DDS 写入未成功，命令未确认送达）"
            )

    def close(self) -> None:
        """关闭通道（基线 9a5b3f7 无锁实现，逐行等价复制并仅格式适配）。

        Raises:
            R1DChannelError: 任一通道关闭失败（仍会尽量完成全部清理）。
        """
        if not self._opened and self._subscriber is None and self._publisher is None:
            self._on_state = None
            return  # 已处关闭态，幂等
        # 世代门立即生效：此后旧回调（在途/迟到帧）不再投递
        self._opened = False
        self._on_state = None
        errors: list[str] = []

        publisher, self._publisher, self._acquired = self._publisher, None, False
        if publisher is not None:
            try:
                publisher.Close()
            except Exception as exc:
                errors.append(f"publisher 关闭失败: {exc}")
        subscriber, self._subscriber = self._subscriber, None
        if subscriber is not None:
            try:
                subscriber.Close()
            except Exception as exc:
                errors.append(f"subscriber 关闭失败: {exc}")
        if errors:
            raise R1DChannelError("close 清理失败: " + "; ".join(errors))


def _make_transport(publisher: _FakePublisher, *, baseline: bool) -> Any:
    """构造并打开一个已取得控制权的传输实例（基线或当前实现）。"""
    if baseline:
        transport: Any = _BaselineTransport(bundle=_fake_bundle(publisher))
    else:
        transport = SdkR1DTransport(
            dds_domain=0,
            network_interface="bench0",
            switcher_timeout_s=0.1,
            acquire_timeout_s=0.5,
            release_retry_interval_s=0.0,
            clock=time.monotonic,
            bundle=_fake_bundle(publisher),
        )
    transport.open(lambda frame: None)
    transport.acquire_control()
    return transport


def _command_frame() -> R1DCommandFrame:
    """35 槽哨兵命令帧（数值非实机参数）。"""
    n = R1D_SLOT_COUNT
    return R1DCommandFrame(
        q=np.linspace(-0.4, 0.4, n),
        dq=np.zeros(n),
        kp=np.full(n, 20.0),
        kd=np.full(n, 1.0),
        tau=np.zeros(n),
        motor_mode=np.zeros(n),
        mode_machine=7,
        mode_pr=0,
    )


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------


def _percentiles(samples: list[float]) -> dict[str, float]:
    """返回 p50/p95/p99/max（单位与输入一致）。"""
    ordered = sorted(samples)
    n = len(ordered)

    def pick(q: float) -> float:
        return ordered[min(n - 1, max(0, round(q * (n - 1))))]

    return {
        "p50": pick(0.50),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": ordered[-1],
    }


def _publish_round(pairs: int) -> dict[str, Any]:
    """交替测量基线与当前实现的发布耗时（同一轮内配对，抵消 CPU 漂移）。"""
    baseline_pub = _FakePublisher()
    locked_pub = _FakePublisher()
    baseline = _make_transport(baseline_pub, baseline=True)
    locked = _make_transport(locked_pub, baseline=False)
    frame = _command_frame()
    try:
        for _ in range(2000):  # 预热
            baseline.publish(frame)
            locked.publish(frame)
        base_samples: list[float] = []
        lock_samples: list[float] = []
        deltas: list[float] = []
        for _ in range(pairs):
            t0 = time.perf_counter_ns()
            baseline.publish(frame)
            t1 = time.perf_counter_ns()
            locked.publish(frame)
            t2 = time.perf_counter_ns()
            base_samples.append((t1 - t0) / 1000.0)
            lock_samples.append((t2 - t1) / 1000.0)
            deltas.append((t2 - t1 - (t1 - t0)) / 1000.0)
        return {
            "baseline_us": _percentiles(base_samples),
            "locked_us": _percentiles(lock_samples),
            "delta_p50_us": statistics.median(deltas),
            "delta_mean_us": statistics.fmean(deltas),
        }
    finally:
        baseline.close()
        locked.close()


def _build_round(iterations: int) -> dict[str, float]:
    """测量 `_build_lowcmd` + CRC（发布路径的大头）。"""
    bundle = _fake_bundle(_FakePublisher())
    frame = _command_frame()
    for _ in range(1000):
        _build_lowcmd(frame, bundle)
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        msg = _build_lowcmd(frame, bundle)
        msg.crc = bundle.crc_compute(msg)
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1000.0)
    return _percentiles(samples)


def _lock_pair_ns(iterations: int) -> float:
    """测量一次 RLock 加解锁（纳秒）。"""
    lock = threading.RLock()
    for _ in range(1000):
        with lock:
            pass
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        with lock:
            pass
        t1 = time.perf_counter_ns()
        samples.append(float(t1 - t0))
    return statistics.median(samples)


def _publish_and_record(
    transport: Any,
    frame: R1DCommandFrame,
    errors: list[Exception],
) -> None:
    """线程入口：发布一次，把任意异常记入列表（由主线程统一输出）。"""
    try:
        transport.publish(frame)
    except Exception as exc:  # noqa: BLE001 - 基准需记录线程内任意失败
        errors.append(exc)


def _close_wait(block_seconds: float) -> dict[str, Any]:
    """比较“有在途 Write”时基线/当前实现 close 的返回时长与等待事实。"""
    result: dict[str, Any] = {}
    for name, baseline in (("baseline", True), ("locked", False)):
        publisher = _FakePublisher(block_seconds=block_seconds)
        transport = _make_transport(publisher, baseline=baseline)
        frame = _command_frame()
        errors: list[Exception] = []
        thread = threading.Thread(
            target=_publish_and_record,
            args=(transport, frame, errors),
            name=f"bench-publish-{name}",
            daemon=True,
        )
        try:
            thread.start()
            if not publisher.entered.wait(timeout=2.0):
                raise RuntimeError(f"{name}: 发布线程未进入 Write，基准环境异常")
            close_start = time.perf_counter()
            try:
                transport.close()
            except Exception as exc:  # noqa: BLE001 - 记录 close 的任意失败
                errors.append(exc)
            close_ms = (time.perf_counter() - close_start) * 1000.0
            # 关键证据：close 返回的瞬间，Write 是否仍在进行。
            write_in_flight_at_close = not publisher.completed
        finally:
            publisher.release.set()
            thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError(f"{name}: 发布线程 5s 内未退出，基准环境异常")
        result[name] = {
            "close_ms": close_ms,
            "thread_ms": (time.perf_counter() - close_start) * 1000.0,
            "writes": publisher.messages,
            "write_in_flight_at_close": write_in_flight_at_close,
            "errors": [repr(exc) for exc in errors],
        }
    return result


def _self_check(close_wait: dict[str, Any]) -> list[str]:
    """强制自检：close 等待实验的关键事实必须成立，否则基准结论无效。"""
    baseline = close_wait["baseline"]
    locked = close_wait["locked"]
    checks = {
        "baseline.write_in_flight_at_close is True": baseline[
            "write_in_flight_at_close"
        ]
        is True,
        "locked.write_in_flight_at_close is False": locked["write_in_flight_at_close"]
        is False,
        "baseline.writes == 1": baseline["writes"] == 1,
        "locked.writes == 1": locked["writes"] == 1,
        "baseline.errors == []": baseline["errors"] == [],
        "locked.errors == []": locked["errors"] == [],
    }
    return [name for name, ok in checks.items() if not ok]


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def _positive_int(text: str) -> int:
    """argparse 类型：正整数（参数校验，失败即退出码 2）。"""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"必须为正整数，实际 {value}")
    return value


def _positive_finite_float(text: str) -> float:
    """argparse 类型：正的有限浮点数（拒绝 0/负数/inf/nan）。"""
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(f"必须为有限正数，实际 {value}")
    return value


def _git_sha() -> str | None:
    """读取当前提交 SHA（仅作证据标注，失败返回 None，不让基准因此崩溃）。"""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


def main(argv: list[str] | None = None) -> int:
    """命令行入口：跑发布/组包/锁/close 等待基准并打印结果。

    Returns:
        0 表示基准完成且强制自检全部通过；1 表示执行异常或自检失败
        （无效参数由 argparse 以退出码 2 拒绝）。
    """
    parser = argparse.ArgumentParser(description="R1D transport 发布路径基准")
    parser.add_argument(
        "--pairs", type=_positive_int, default=100_000, help="每轮配对数"
    )
    parser.add_argument("--rounds", type=_positive_int, default=3, help="发布基准轮数")
    parser.add_argument(
        "--build-iterations", type=_positive_int, default=20_000, help="组包测量次数"
    )
    parser.add_argument(
        "--lock-iterations", type=_positive_int, default=50_000, help="锁配对测量次数"
    )
    parser.add_argument(
        "--close-block-ms",
        type=_positive_finite_float,
        default=200.0,
        help="close 等待实验中一次 Write 的阻塞毫秒数",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args(argv)

    gc_was_enabled = gc.isenabled()
    gc.disable()  # 测量期间关闭 GC，避免偶发回收污染尾部
    gc_off = True
    try:
        rounds = [_publish_round(args.pairs) for _ in range(args.rounds)]
        build = _build_round(args.build_iterations)
        lock_pair_ns = _lock_pair_ns(args.lock_iterations)
        close_wait = _close_wait(args.close_block_ms / 1000.0)
    except Exception as exc:  # noqa: BLE001 - 基准失败必须非零退出并说明原因
        print(f"基准执行异常（结论无效）: {exc!r}", file=sys.stderr)
        return 1
    finally:
        if gc_was_enabled:
            gc.enable()

    failures = _self_check(close_wait)
    if failures:
        for failure in failures:
            print(f"强制自检失败: {failure}", file=sys.stderr)
        return 1

    payload = {
        "git_sha": _git_sha(),
        "argv": sys.argv[1:],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "params": {
            "pairs": args.pairs,
            "rounds": args.rounds,
            "build_iterations": args.build_iterations,
            "lock_iterations": args.lock_iterations,
            "close_block_ms": args.close_block_ms,
        },
        "rounds": rounds,
        "build_us": build,
        "lock_pair_ns": lock_pair_ns,
        "close_wait": close_wait,
        "self_checks_passed": True,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("R1D transport 发布路径基准（fake SDK / 离线）")
    print(f"  pairs/round={args.pairs}  rounds={args.rounds}  gc_off={gc_off}")
    print()
    print("  轮次   baseline p50   locked p50   delta p50   delta mean   baseline p99")
    for index, row in enumerate(rounds, start=1):
        print(
            f"  {index:>4}   {row['baseline_us']['p50']:>10.2f}   "
            f"{row['locked_us']['p50']:>10.2f}   {row['delta_p50_us']:>+9.2f}   "
            f"{row['delta_mean_us']:>+9.2f}   {row['baseline_us']['p99']:>11.2f}"
        )
    delta_medians = [row["delta_p50_us"] for row in rounds]
    print(f"  delta p50 中位数: {statistics.median(delta_medians):+.2f} us")
    print()
    print(
        f"  _build_lowcmd+CRC: p50 {build['p50']:.2f} us  p95 {build['p95']:.2f} us  "
        f"p99 {build['p99']:.2f} us"
    )
    print(f"  一次 RLock 加解锁: {lock_pair_ns:.0f} ns")
    print()
    print(f"  close 等待（一次 Write 阻塞 {args.close_block_ms:.0f} ms）:")
    for name in ("baseline", "locked"):
        row = close_wait[name]
        print(
            f"    {name:>8}: close {row['close_ms']:>9.2f} ms  "
            f"write_in_flight_at_close={row['write_in_flight_at_close']}  "
            f"writes={row['writes']}  errors={len(row['errors'])}"
        )
    print()
    print(
        "  对照说明：baseline 的 publish/close 逐行等价复制自 9a5b3f7（旧无锁实现，",
    )
    print(
        "  仅格式适配），locked 为当前生产类 SdkR1DTransport。关键证据是 close 返回瞬间 Write"
    )
    print(
        "  是否仍在途：baseline=True（竞态窗口），locked=False（close 等到 Write 完成）。"
    )
    print("  thread_ms 含线程启动/调度开销，仅作参考；fake DDS 数字不外推到真实 DDS。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
