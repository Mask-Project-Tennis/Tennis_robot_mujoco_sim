"""R1DStateBuffer 状态缓存测试（阶段 4 Task 4.4）。

覆盖 brief 全部具名项：空缓存/超时/NaN/长度错误/机型错误/乱序/重复/
旧会话不可用、重复帧不刷新 freshness、回绕接受、重启需新会话、
源/接收时钟不同步、快照双侧不可变、线程安全。

所有时间值均为测试专用 fake clock 注入（latest 显式传 now、帧显式传
received_at），不依赖 sleep，非实机参数。
"""

from __future__ import annotations

import dataclasses
import itertools
import threading
from typing import Any, Callable

import numpy as np
import pytest

from src.real.r1d_state_buffer import R1DStateBuffer, R1DStateError
from src.real.r1d_types import R1DStateFrame, R1D_STATE_SLOT_COUNT

# 测试专用会话标识与新鲜度超时（非实机参数）
SESSION = 1
TIMEOUT = 1.0
# 测试专用机型值（manifest mode_machine_validation=null，仅用于相等校验路径）
MODE = 5


def make_frame(
    *,
    sequence: int | None = 10,
    source_time: float | None = None,
    received_at: float = 0.0,
    mode_machine: int = MODE,
    session_id: int = SESSION,
    q_first: float = 0.0,
) -> R1DStateFrame:
    """构造合法状态帧（q[0] 编码帧身份，用于一致性断言）。"""
    q = np.full(R1D_STATE_SLOT_COUNT, 0.1, dtype=np.float64)
    q[0] = q_first
    dq = np.zeros(R1D_STATE_SLOT_COUNT, dtype=np.float64)
    return R1DStateFrame(
        q=q,
        dq=dq,
        sequence=sequence,
        source_time=source_time,
        received_at=received_at,
        mode_machine=mode_machine,
        session_id=session_id,
    )


def make_raw_frame(**overrides: Any) -> R1DStateFrame:
    """绕过 R1DStateFrame.__post_init__ 构造伪帧。

    仅用于测试 push 的防御性校验（模拟反序列化/篡改绕过类型层的路径），
    生产代码不得使用。
    """
    fields: dict[str, Any] = {
        "q": np.full(R1D_STATE_SLOT_COUNT, 0.1),
        "dq": np.zeros(R1D_STATE_SLOT_COUNT),
        "sequence": 10,
        "source_time": None,
        "received_at": 0.0,
        "mode_machine": MODE,
        "session_id": SESSION,
    }
    fields.update(overrides)
    frame = object.__new__(R1DStateFrame)
    for key, value in fields.items():
        object.__setattr__(frame, key, value)
    return frame


def make_buffer(
    *,
    session_id: int = SESSION,
    state_timeout_s: float = TIMEOUT,
    expected_mode_machine: int | None = None,
) -> R1DStateBuffer:
    """构造被测缓存（参数均为测试专用值）。"""
    return R1DStateBuffer(
        session_id=session_id,
        state_timeout_s=state_timeout_s,
        expected_mode_machine=expected_mode_machine,
    )


# ---------------------------------------------------------------------------
# 正路径
# ---------------------------------------------------------------------------


def test_valid_frame_roundtrip_and_strictly_increasing_accepted() -> None:
    """有效帧入缓存可读回；严格递增序号全部接受。"""
    buffer = make_buffer()
    assert buffer.push(make_frame(sequence=1, received_at=0.0, q_first=1.0)) is True
    assert buffer.push(make_frame(sequence=2, received_at=0.1, q_first=2.0)) is True
    got = buffer.latest(now=0.2, session_id=SESSION)
    assert got.sequence == 2
    assert got.q[0] == pytest.approx(2.0)


def test_sequence_none_frame_uses_receive_freshness_only() -> None:
    """sequence=None 仅凭接收单调时钟判新鲜（seq_semantics 无头序列号支持）。"""
    buffer = make_buffer()
    # 无序号无法判重复，连续 None 序号帧均按接收新鲜度接受
    assert buffer.push(make_frame(sequence=None, received_at=0.0)) is True
    assert buffer.push(make_frame(sequence=None, received_at=0.2)) is True
    assert buffer.latest(now=0.3, session_id=SESSION).sequence is None
    # 超时判据同样只看 received_at（相对最新帧 received_at=0.2）
    with pytest.raises(R1DStateError, match="过期"):
        buffer.latest(now=0.2 + TIMEOUT + 0.01, session_id=SESSION)


# ---------------------------------------------------------------------------
# 无效/失败路径：不产生可用新状态
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        # A-M6（评审 2026-11-17）：bool 是 int 子类，裸 isinstance 校验
        # 会放行——构造层必须显式拒绝（True 曾可冒充 sequence=1）
        ("sequence", True, "整数"),
        ("sequence", False, "整数"),
        # 负 sequence：uint32 序号语义之外的值在构造层拒绝（A-M6 修复
        # 后前移；上界 2^32 仍由缓存层 push 拒绝，见 uint32 域测试）
        ("sequence", -1, "非负"),
        ("mode_machine", True, "uint8"),
        ("mode_machine", False, "uint8"),
        ("mode_machine", -1, "uint8"),
        ("mode_machine", 256, "uint8"),
        ("session_id", True, "整数"),
        ("received_at", float("nan"), "有限"),
    ],
    ids=[
        "sequence-bool-true", "sequence-bool-false", "sequence-negative",
        "mode-bool-true", "mode-bool-false", "mode-negative", "mode-over-255",
        "session-bool", "received-at-nan",
    ],
)
def test_state_frame_constructor_rejects_invalid_scalars(
    field: str, value: Any, match: str
) -> None:
    """R1DStateFrame 标量字段构造校验：bool/负值/越界/NaN 一律拒绝（A-M6）。"""
    kwargs: dict[str, Any] = {field: value}
    with pytest.raises(ValueError, match=match):
        make_frame(**kwargs)


def test_state_frame_constructor_rejects_bool_arrays() -> None:
    """布尔数组不得冒充数值数组（bool dtype 绕过 finite 校验的口子）。"""
    bool_q = np.zeros(R1D_STATE_SLOT_COUNT, dtype=bool)
    with pytest.raises(ValueError):
        R1DStateFrame(
            q=bool_q,
            dq=np.zeros(R1D_STATE_SLOT_COUNT),
            sequence=10,
            source_time=None,
            received_at=0.0,
            mode_machine=MODE,
            session_id=SESSION,
        )


def test_latest_empty_cache_raises() -> None:
    """空缓存 latest 必须显式失败，不返回任何冒充状态。"""
    buffer = make_buffer()
    with pytest.raises(R1DStateError, match="空"):
        buffer.latest(now=0.0, session_id=SESSION)


def test_stale_state_raises_on_timeout() -> None:
    """超过新鲜度超时的缓存帧不可用（now 为注入 fake clock）。"""
    buffer = make_buffer()
    assert buffer.push(make_frame(received_at=0.0)) is True
    with pytest.raises(R1DStateError, match="过期"):
        buffer.latest(now=TIMEOUT + 0.01, session_id=SESSION)


def test_boundary_equal_timeout_is_fresh() -> None:
    """age 恰等于超时阈值仍算新鲜（判据为 age ≤ timeout）。"""
    buffer = make_buffer()
    assert buffer.push(make_frame(received_at=0.0)) is True
    got = buffer.latest(now=TIMEOUT, session_id=SESSION)
    assert got.received_at == pytest.approx(0.0)


def test_nan_frame_cannot_enter_buffer() -> None:
    """NaN 帧在类型层构造即拒绝；绕过构造的伪帧在 push 拒绝且不入缓存。"""
    buffer = make_buffer()
    # 正常路径：4.2 类型层在构造时拒绝 NaN
    bad_q = np.zeros(R1D_STATE_SLOT_COUNT)
    bad_q[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        R1DStateFrame(
            q=bad_q,
            dq=np.zeros(R1D_STATE_SLOT_COUNT),
            sequence=10,
            source_time=None,
            received_at=0.0,
            mode_machine=MODE,
            session_id=SESSION,
        )
    # 防御路径：伪帧（绕过 __post_init__）在 push 被 R1DStateError 拒绝
    with pytest.raises(R1DStateError, match="NaN"):
        buffer.push(make_raw_frame(q=bad_q))
    with pytest.raises(R1DStateError, match="空"):
        buffer.latest(now=0.0, session_id=SESSION)


def test_wrong_length_frame_cannot_enter_buffer() -> None:
    """长度错误帧在类型层构造即拒绝；伪帧在 push 拒绝且不入缓存。"""
    buffer = make_buffer()
    with pytest.raises(ValueError, match="长度"):
        R1DStateFrame(
            q=np.zeros(R1D_STATE_SLOT_COUNT - 1),
            dq=np.zeros(R1D_STATE_SLOT_COUNT),
            sequence=10,
            source_time=None,
            received_at=0.0,
            mode_machine=MODE,
            session_id=SESSION,
        )
    with pytest.raises(R1DStateError, match="长度"):
        buffer.push(
            make_raw_frame(dq=np.zeros(R1D_STATE_SLOT_COUNT + 1))
        )
    with pytest.raises(R1DStateError, match="空"):
        buffer.latest(now=0.0, session_id=SESSION)


def test_mode_machine_mismatch_raises_on_latest() -> None:
    """冻结了期望机型值时，机型不一致的状态不可用。"""
    buffer = make_buffer(expected_mode_machine=MODE)
    assert buffer.push(make_frame(mode_machine=MODE + 1)) is True
    with pytest.raises(R1DStateError, match="机型"):
        buffer.latest(now=0.5, session_id=SESSION)


def test_mode_machine_check_skipped_when_expected_null() -> None:
    """期望机型为 null 时跳过相等断言（manifest mode_machine_validation=null，R6 口径）。"""
    buffer = make_buffer(expected_mode_machine=None)
    assert buffer.push(make_frame(mode_machine=MODE + 1)) is True
    got = buffer.latest(now=0.5, session_id=SESSION)
    assert got.mode_machine == MODE + 1


def test_out_of_order_small_rollback_rejected() -> None:
    """小幅序号回退视为乱序丢弃，缓存保留既有更新帧。"""
    buffer = make_buffer()
    assert buffer.push(make_frame(sequence=100, q_first=100.0)) is True
    assert buffer.push(make_frame(sequence=98, received_at=0.5, q_first=98.0)) is False
    got = buffer.latest(now=0.6, session_id=SESSION)
    assert got.sequence == 100
    assert got.q[0] == pytest.approx(100.0)


def test_duplicate_sequence_rejected_and_does_not_refresh_freshness() -> None:
    """重复序号帧被丢弃且不得刷新 freshness（缓存保留原 received_at）。"""
    buffer = make_buffer()
    assert buffer.push(make_frame(sequence=10, received_at=0.0)) is True
    # 同序号帧晚到：push 拒绝
    assert buffer.push(make_frame(sequence=10, received_at=0.9)) is False
    # 若错误地刷新了 freshness（received_at 变 0.9），now=1.05 仍会"新鲜"；
    # 正确行为：仍按原 received_at=0.0 判过期 → 显式失败
    with pytest.raises(R1DStateError, match="过期"):
        buffer.latest(now=TIMEOUT + 0.05, session_id=SESSION)
    # 未过期窗口内仍可读回原帧
    assert buffer.push(make_frame(sequence=10, received_at=0.0)) is False
    got = buffer.latest(now=0.99, session_id=SESSION)
    assert got.sequence == 10
    assert got.received_at == pytest.approx(0.0)


def test_wraparound_accepted_and_recorded() -> None:
    """大幅回退（≥2^31）按 uint32 回绕接受并计数记录。"""
    buffer = make_buffer()
    high = 2**32 - 3
    assert buffer.push(make_frame(sequence=high, q_first=float(high))) is True
    # 回退量 = 2^32 - 5 ≥ 2^31 → 回绕接受
    assert buffer.push(make_frame(sequence=2, received_at=0.1, q_first=2.0)) is True
    assert buffer.wraparound_count == 1
    got = buffer.latest(now=0.2, session_id=SESSION)
    assert got.sequence == 2
    # 回绕后继续严格递增接受
    assert buffer.push(make_frame(sequence=3, received_at=0.2, q_first=3.0)) is True
    # 相对 last=3 的小幅回退（回退量 2 < 2^31）→ 乱序丢弃，回绕计数不变
    assert buffer.push(make_frame(sequence=1, received_at=0.3)) is False
    assert buffer.wraparound_count == 1


def test_sequence_out_of_uint32_rejected() -> None:
    """序号上界（uint32 物理域）在 push 拒绝；负值在构造层拒绝（A-M6 分层）。"""
    buffer = make_buffer()
    with pytest.raises(R1DStateError, match="uint32"):
        buffer.push(make_frame(sequence=2**32))
    # A-M6：负 sequence 前移到构造层（_require_non_negative_int），
    # 上界仍归缓存层（构造层不知协议 tick 上界职责边界不变）
    with pytest.raises(ValueError, match="非负"):
        make_frame(sequence=-1)
    with pytest.raises(R1DStateError, match="空"):
        buffer.latest(now=0.0, session_id=SESSION)


# ---------------------------------------------------------------------------
# 会话隔离与重启
# ---------------------------------------------------------------------------


def test_old_session_push_rejected_and_latest_raises() -> None:
    """reset 后旧会话帧一律不可用：push 拒绝、latest 显式失败。"""
    buffer = make_buffer(session_id=SESSION)
    assert buffer.push(make_frame(session_id=SESSION)) is True
    buffer.reset(session_id=SESSION + 1)
    # 旧会话帧晚到（正常竞态）：push 拒绝且不复活旧状态
    assert buffer.push(make_frame(session_id=SESSION, received_at=0.5)) is False
    # 旧会话调用方：latest 显式失败
    with pytest.raises(R1DStateError, match="会话"):
        buffer.latest(now=0.6, session_id=SESSION)
    # 新会话但缓存为空：同样显式失败
    with pytest.raises(R1DStateError, match="空"):
        buffer.latest(now=0.6, session_id=SESSION + 1)


def test_restart_requires_new_session() -> None:
    """重启（reset 新会话）后旧状态清空，需新会话新帧才恢复可用。"""
    buffer = make_buffer(session_id=SESSION)
    assert buffer.push(make_frame(received_at=0.0)) is True
    buffer.reset(session_id=SESSION + 1)
    # 序号基准同时清空：新会话首帧不与旧会话序号比较误判乱序
    assert buffer.push(make_frame(sequence=1, session_id=SESSION + 1)) is True
    got = buffer.latest(now=0.1, session_id=SESSION + 1)
    assert got.session_id == SESSION + 1
    assert got.sequence == 1
    assert buffer.wraparound_count == 0


# ---------------------------------------------------------------------------
# 源/接收时钟不同步
# ---------------------------------------------------------------------------


def test_source_time_not_used_for_freshness() -> None:
    """新鲜度唯一判据为接收单调时钟，source_time 极旧/极新均不影响判定。

    源时钟与接收端单调时钟不同步（协议无同步承诺），禁止拿源时间戳
    减本机时钟。
    """
    buffer = make_buffer()
    # 方向一：source_time 极旧，但接收时刻新鲜 → 可用
    assert buffer.push(make_frame(source_time=0.0, received_at=0.0)) is True
    assert buffer.latest(now=0.5, session_id=SESSION).source_time == 0.0
    # 方向二：source_time 极新（未来），不能"救活"过期的接收时刻
    assert buffer.push(
        make_frame(sequence=11, source_time=1e9, received_at=0.0, q_first=11.0)
    ) is True
    with pytest.raises(R1DStateError, match="过期"):
        buffer.latest(now=TIMEOUT + 0.01, session_id=SESSION)


def test_now_must_be_finite() -> None:
    """now 非有限值必须显式失败，防止 NaN 比较吞掉超时判断冒充新鲜。"""
    buffer = make_buffer()
    assert buffer.push(make_frame(received_at=0.0)) is True
    for bad_now in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(R1DStateError, match="有限"):
            buffer.latest(now=bad_now, session_id=SESSION)


# ---------------------------------------------------------------------------
# 快照不可变（双侧）
# ---------------------------------------------------------------------------


def test_snapshot_cannot_be_mutated_by_callback_or_reader() -> None:
    """写入前修改源数组、读取后修改返回对象，均不污染内部快照。"""
    buffer = make_buffer()
    # 写入侧：构造帧的源数组保持可写，写入后修改不得影响缓存快照
    q_src = np.full(R1D_STATE_SLOT_COUNT, 0.1)
    frame = R1DStateFrame(
        q=q_src,
        dq=np.zeros(R1D_STATE_SLOT_COUNT),
        sequence=10,
        source_time=None,
        received_at=0.0,
        mode_machine=MODE,
        session_id=SESSION,
    )
    assert buffer.push(frame) is True
    q_src[0] = 999.0  # 修改源数组（回调路径侧）
    got = buffer.latest(now=0.5, session_id=SESSION)
    assert got.q[0] == pytest.approx(0.1)

    # 读取侧：修改返回对象的数组/字段均被拒绝（只读数组 + frozen dataclass）
    with pytest.raises(ValueError, match="read-only"):
        got.q[0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        got.session_id = 999  # type: ignore[misc]
    # 再次读取：内部快照未受任何影响
    again = buffer.latest(now=0.6, session_id=SESSION)
    assert again.q[0] == pytest.approx(0.1)
    assert again.session_id == SESSION
    assert again.sequence == 10


# ---------------------------------------------------------------------------
# 线程安全
# ---------------------------------------------------------------------------


def test_thread_safety_concurrent_push_and_latest() -> None:
    """多线程并发 push/latest：整帧原子替换、无字段撕裂、无异常（不靠 sleep）。"""
    buffer = make_buffer(state_timeout_s=1000.0)
    n_threads = 4
    frames_per_thread = 300
    total_frames = n_threads * frames_per_thread
    barrier = threading.Barrier(n_threads)
    seq_counter = itertools.count(1)
    state_lock = threading.Lock()  # 保护序号分发与结果收集
    results: list[tuple[int, bool]] = []  # (sequence, 是否接受)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            for _ in range(frames_per_thread):
                with state_lock:
                    seq = next(seq_counter)
                ok = buffer.push(
                    make_frame(sequence=seq, received_at=0.0, q_first=float(seq))
                )
                with state_lock:
                    results.append((seq, ok))
                got = buffer.latest(now=1.0, session_id=SESSION)
                # 同一帧内部一致性：q[0] 与 sequence 必来自同一快照（防撕裂）
                assert got.q[0] == float(got.sequence)
        except BaseException as exc:  # 收集线程内任意失败统一断言
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, f"线程内失败: {errors}"
    # 结果完整性：每个 (seq, ok) 恰好记录一次，序号全局唯一
    assert len(results) == total_frames
    pushed_seqs = {seq for seq, _ in results}
    assert len(pushed_seqs) == total_frames
    # 接受计数下界（E-M4 收紧，评审 2026-11-17：原"1<=n<=total"的上界
    # 对 results 子集恒真、无判别力，删除）：全局最大序号帧 push 时不存在
    # 更大的已接受序号使其乱序，必然被接受
    accepted = [seq for seq, ok in results if ok]
    assert accepted, "全局最大序号帧必须被接受"
    assert max(pushed_seqs) in accepted
    # 最终缓存帧：内部一致（防撕裂）且 sequence 必属实际 push 过的帧
    got = buffer.latest(now=1.0, session_id=SESSION)
    assert got.q[0] == float(got.sequence)
    assert got.sequence in pushed_seqs


# ---------------------------------------------------------------------------
# 会话隔离竞态（I-02，评审 2026-11-17）：代理锁在真实取锁点注入交错
# ---------------------------------------------------------------------------


class _InterleavingLock:
    """代理锁：首次 enter 时执行注入的交错动作（模拟"锁外判定已通过、
    持锁读写前另一线程 reset/换帧"的真实竞态窗口），随后透传真实锁。

    仅首次触发：交错动作自身（reset/push）也走本代理，fired 已置位故
    不会递归。
    """

    def __init__(self, real: threading.Lock, interleave: Callable[[], None]) -> None:
        self._real = real
        self._interleave = interleave
        self._fired = False

    def __enter__(self) -> "_InterleavingLock":
        if not self._fired:
            self._fired = True
            self._interleave()
        self._real.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._real.__exit__(*exc)


def test_push_rechecks_session_inside_lock_under_reset_race() -> None:
    """I-02：push 会话判定与写入同锁原子——reset 交错下旧会话帧不得入新会话缓存。

    交错注入：push(旧会话帧) 完成锁外检查后、取得锁前，另一线程
    reset 切换会话。修复前（锁外比较+锁内不复查）：旧帧与旧序号基准
    写入新会话缓存；修复后（锁内复查）：拒绝返回 False。
    """
    buffer = make_buffer(session_id=SESSION)
    old_frame = make_frame(session_id=SESSION, sequence=100, received_at=0.0)
    real_lock = buffer._lock  # noqa: SLF001  # 竞态注入需要替换内部锁

    def interleave() -> None:
        # push 已通过锁外会话检查（修复前），此刻另一线程 reset 切换会话
        buffer.reset(session_id=SESSION + 1)

    buffer._lock = _InterleavingLock(real_lock, interleave)  # noqa: SLF001
    try:
        assert buffer.push(old_frame) is False  # 锁内复查拒绝旧会话帧
    finally:
        buffer._lock = real_lock  # noqa: SLF001
    # 新会话缓存未被旧帧污染：仍为空（显式报错而非返回旧帧）
    with pytest.raises(R1DStateError, match="空"):
        buffer.latest(now=0.1, session_id=SESSION + 1)


def test_latest_never_returns_cross_session_frame_under_reset_race() -> None:
    """I-02：latest 会话判定与帧读取同锁原子——reset 交错下不得跨会话交付帧。

    交错注入：latest(旧会话) 完成锁外会话检查后、取得锁前，另一线程
    reset 新会话并写入新帧。修复前（锁外比较+返回前不校验帧会话）：
    旧会话调用方拿到新会话帧；修复后：锁内复查显式报"会话不匹配"。
    """
    buffer = make_buffer(session_id=SESSION)
    assert buffer.push(make_frame(session_id=SESSION, received_at=0.0)) is True
    real_lock = buffer._lock  # noqa: SLF001

    def interleave() -> None:
        buffer.reset(session_id=SESSION + 1)
        buffer.push(
            make_frame(session_id=SESSION + 1, received_at=0.0, q_first=7.0)
        )

    buffer._lock = _InterleavingLock(real_lock, interleave)  # noqa: SLF001
    try:
        with pytest.raises(R1DStateError, match="会话"):
            buffer.latest(now=0.5, session_id=SESSION)
    finally:
        buffer._lock = real_lock  # noqa: SLF001
    # 新会话自身读取不受影响（帧会话一致）
    got = buffer.latest(now=0.5, session_id=SESSION + 1)
    assert got.session_id == SESSION + 1


# ---------------------------------------------------------------------------
# 构造参数防御
# ---------------------------------------------------------------------------


def test_constructor_rejects_invalid_parameters() -> None:
    """state_timeout_s 非正/非有限、session_id 非整数、期望机型越界均拒绝。"""
    with pytest.raises(ValueError, match="正"):
        make_buffer(state_timeout_s=0.0)
    with pytest.raises(ValueError, match="有限"):
        make_buffer(state_timeout_s=float("nan"))
    with pytest.raises(ValueError, match="整数"):
        make_buffer(session_id=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="uint8"):
        make_buffer(expected_mode_machine=256)


def test_push_rejects_non_frame_input() -> None:
    """push 只接受 R1DStateFrame，其他类型显式 TypeError。"""
    buffer = make_buffer()
    with pytest.raises(TypeError):
        buffer.push("not-a-frame")  # type: ignore[arg-type]
