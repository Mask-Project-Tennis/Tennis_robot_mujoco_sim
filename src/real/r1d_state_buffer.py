"""R1-A7-D 状态缓存 — 带时效与会话隔离的线程安全快照缓存（阶段 4 Task 4.4）。

协议事实来源：configs/robots/r1d_protocol_manifest.yaml（Task 4.1 冻结）：
- dds.seq_semantics.has_header_seq=false：hg LowState_ 无 go2 式头序列号；
  序号承接 LowState_.tick（uint32），其步长/回绕/重启语义均 null（未公开）。
  因此本模块对序号只做保守判定（见 R1DStateBuffer.push），重启不靠序号
  区分，靠会话标识 session_id（旧会话帧一律不可用）。
- state_validity.stale_state=null：无官方新鲜度标准，本任务自定——
  新鲜度唯一判据 = 注入的 now - frame.received_at ≤ state_timeout_s
  （接收端单调时钟）。禁止拿 source_time 减本机时钟：源时钟与接收端
  单调时钟无同步承诺，source_time 仅透传不参与判定。
- state_validity.crc_required=true：CRC 校验属 4.5 传输层，交付本缓存的
  帧应已通过 CRC；本模块不重复实现 CRC。
- mode_machine_validation=null：期望机型值冻结前，latest 仅在显式配置
  expected_mode_machine 时做相等校验，None 时跳过（R6 口径：冻结前只做
  变化检测+记录，不做相等断言——变化检测属 4.6 生命周期）。

本模块不导入 SDK/MuJoCo、不启动线程、不做网络或控制权副作用。
push 回调路径只做校验/入缓存（R1DStateFrame 在 4.2 构造时已边界复制并
置只读，存储引用即不可变快照），不执行控制权 RPC、不发 DDS、无长耗时
操作。有效性与新鲜度失败抛 R1DStateError，不吞异常、不返回旧缓存冒充
新状态；无序号/源时间时只能证明接收新鲜度，是否足够启用运动按 G1
门槛由 4.6 生命周期裁定。
"""

from __future__ import annotations

import threading
from typing import Union

import numpy as np

from src.real.r1d_types import R1DStateFrame, R1D_STATE_SLOT_COUNT

# 协议 tick 物理域：uint32（manifest lowstate_layout.tick）
_UINT32_MAX = 0xFFFFFFFF
# 回退量 ≥ 2^31（半空间）按回绕接受；tick 回绕语义 manifest 为 null，
# 此阈值为主代理裁定 R7 的保守启发式，非厂商公开语义
_SEQUENCE_WRAPAROUND_THRESHOLD = 2**31
# 机型字段物理域：uint8（manifest lowstate_layout.mode_machine）
_UINT8_MIN = 0
_UINT8_MAX = 255

# 防御性校验中的数值类型口径：注解用别名，isinstance 用元组
_Number = Union[int, float, np.integer, np.floating]
_NUMBER_TYPES = (int, float, np.integer, np.floating)


def _require_plain_int(value: object, name: str) -> int:
    """校验普通整数字段（bool 视为非法）。

    Args:
        value: 输入值。
        name: 字段名（用于错误消息）。

    Returns:
        int 值。

    Raises:
        ValueError: 非整数。
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"字段 {name} 必须是整数")
    return int(value)


class R1DStateError(RuntimeError):
    """R1D 状态缓存访问失败（有效性/新鲜度/会话）。

    抛出场景：空缓存、状态超时、机型不匹配、旧会话访问、now 非有限、
    push 收到绕过类型层校验的无效帧等。缓存层不吞异常、不返回旧缓存
    冒充新状态；调用方须按故障路径处理。
    """


class R1DStateBuffer:
    """带时效与会话隔离的整机状态快照缓存。

    线程安全原子快照：内部持锁整帧替换引用，读取返回同一不可变
    R1DStateFrame 对象（数组在 4.2 构造时已复制并置只读），写入侧与
    读取侧均无法污染内部快照。会话隔离判定（push 的旧会话丢弃、
    latest 的会话不匹配拒绝）与帧读写位于同一锁内（I-02 修复，
    评审 2026-11-17：锁外判定与持锁读写之间的 reset 竞态会跨会话
    污染快照）。

    Attributes:
        session_id: 当前会话标识（reset 更换；旧会话帧/调用方一律不可用）。
        state_timeout_s: 状态新鲜度超时（秒，接收端单调时钟口径）。
        expected_mode_machine: 期望机型值；None 表示协议未冻结、跳过
            相等校验（manifest mode_machine_validation=null，R6 口径）。
        wraparound_count: 本会话内按回绕接受的大幅序号回退次数（诊断记录）。
    """

    def __init__(
        self,
        *,
        session_id: int,
        state_timeout_s: float,
        expected_mode_machine: int | None = None,
    ) -> None:
        """初始化缓存并校验参数。

        Args:
            session_id: 初始会话标识。
            state_timeout_s: 新鲜度超时（秒），必须为正有限值
                （实机数值经硬件配置提供，此处不设默认值）。
            expected_mode_machine: 期望机型值（uint8 语义）或 None 跳过。

        Raises:
            ValueError: 参数类型/范围不合法。
        """
        self._session_id = _require_plain_int(session_id, "session_id")
        if isinstance(state_timeout_s, bool) or not isinstance(
            state_timeout_s, _NUMBER_TYPES
        ):
            raise ValueError("字段 state_timeout_s 必须是数值")
        timeout = float(state_timeout_s)
        if not np.isfinite(timeout):
            raise ValueError("字段 state_timeout_s 必须是有限值")
        if timeout <= 0.0:
            raise ValueError("字段 state_timeout_s 必须为正有限值")
        if expected_mode_machine is not None:
            expected = _require_plain_int(
                expected_mode_machine, "expected_mode_machine"
            )
            if not (_UINT8_MIN <= expected <= _UINT8_MAX):
                raise ValueError(
                    f"字段 expected_mode_machine 超出 uint8 范围 "
                    f"[{_UINT8_MIN}, {_UINT8_MAX}]"
                )
            self._expected_mode_machine: int | None = expected
        else:
            self._expected_mode_machine = None
        self._state_timeout_s = timeout
        self._lock = threading.Lock()
        self._frame: R1DStateFrame | None = None
        self._last_sequence: int | None = None
        self._wraparound_count = 0

    @property
    def session_id(self) -> int:
        """当前会话标识。"""
        return self._session_id

    @property
    def state_timeout_s(self) -> float:
        """状态新鲜度超时（秒）。"""
        return self._state_timeout_s

    @property
    def expected_mode_machine(self) -> int | None:
        """期望机型值；None 表示未冻结、跳过相等校验。"""
        return self._expected_mode_machine

    @property
    def wraparound_count(self) -> int:
        """本会话内按回绕接受的序号回退次数（reset 清零）。"""
        return self._wraparound_count

    def push(self, frame: R1DStateFrame) -> bool:
        """写入一帧状态（回调路径：只做校验/入缓存，无 RPC/DDS/长耗时操作）。

        序号保守判定（依据 manifest seq_semantics：has_header_seq=false，
        tick 步长/回绕/重启语义均 null，主代理裁定 R7）：
        - sequence 为 None（无序号支持）：不参与序号判定，仅凭接收时刻
          入缓存（新鲜度由 latest 按接收单调时钟判定），同时把序号基准
          置空——序号链条自此断裂，后续带序号帧按新链条重新起算，不与
          None 帧之前的序号比较；部署侧 sequence 来源单一（LowState_.tick），
          None/非 None 不会混发，故无实际风险；
        - 与上一帧序号相同：重复帧，返回 False 且不刷新 freshness
          （缓存保留原帧原 received_at）；
        - 严格递增：接受；
        - 大幅回退（回退量 ≥ 2^31）：按 uint32 回绕接受并累加
          wraparound_count 记录；
        - 小幅回退（0 < 回退量 < 2^31）：乱序丢弃，返回 False。

        Args:
            frame: 待写入的状态帧（须属当前会话；数组快照由 4.2 类型层
                边界复制置只读，本方法存储引用即不可变快照）。

        Returns:
            是否接受入缓存。返回 False 仅表示重复/乱序/旧会话帧被丢弃，
            不是错误；数据无效则抛异常。

        Raises:
            TypeError: frame 不是 R1DStateFrame。
            R1DStateError: frame 绕过类型层校验（数组长度/有限性/序号
                uint32 域不合法）——正常构造的 R1DStateFrame 不会触发。
        """
        if not isinstance(frame, R1DStateFrame):
            raise TypeError("push 只接受 R1DStateFrame 实例")
        # 防御性校验：正常路径由 R1DStateFrame.__post_init__ 保证，
        # 此处防反序列化/篡改等绕过构造的路径（深度防御）
        if frame.q.shape != (R1D_STATE_SLOT_COUNT,) or frame.dq.shape != (
            R1D_STATE_SLOT_COUNT,
        ):
            raise R1DStateError(
                f"无效帧：q/dq 必须是一维长度 {R1D_STATE_SLOT_COUNT} 数组，"
                f"实际 {frame.q.shape}/{frame.dq.shape}"
            )
        if not bool(np.all(np.isfinite(frame.q))) or not bool(
            np.all(np.isfinite(frame.dq))
        ):
            raise R1DStateError("无效帧：q/dq 含 NaN 或 Inf")
        if not bool(np.isfinite(frame.received_at)):
            raise R1DStateError("无效帧：received_at 必须是有限值")
        if frame.sequence is not None and not (
            0 <= frame.sequence <= _UINT32_MAX
        ):
            raise R1DStateError(
                f"无效帧：sequence 超出 uint32 范围 [0, {_UINT32_MAX}]"
            )

        with self._lock:
            # I-02 修复（评审 2026-11-17）：会话比较与写入同一锁内原子。
            # 原实现锁外比较后持锁写入不复查——比较通过到取得锁之间
            # reset() 切换会话的竞态会让旧会话帧污染新会话缓存
            # （帧与序号基准一并写入）。
            if frame.session_id != self._session_id:
                # 旧会话帧晚到属正常竞态（close/重连后回调残留），静默丢弃
                return False
            last = self._last_sequence
            sequence = frame.sequence
            if sequence is not None and last is not None:
                if sequence == last:
                    # 重复帧：不替换缓存、不刷新 freshness
                    return False
                if sequence < last:
                    rollback = last - sequence
                    if rollback < _SEQUENCE_WRAPAROUND_THRESHOLD:
                        # 小幅回退：乱序丢弃
                        return False
                    # 大幅回退：按 uint32 回绕接受并记录
                    self._wraparound_count += 1
            self._frame = frame
            self._last_sequence = sequence
            return True

    def latest(self, *, now: float, session_id: int) -> R1DStateFrame:
        """读取最新可用状态帧（整帧原子快照，返回同一不可变对象）。

        检查顺序：会话 → now 有限性 → 缓存非空 → 新鲜度 → 机型。
        新鲜度唯一判据 = now - frame.received_at ≤ state_timeout_s
        （接收端单调时钟；source_time 不参与判定，源/接收时钟不同步）。

        Args:
            now: 调用方注入的单调时钟秒（fake clock 可测）。
            session_id: 调用方会话标识，须等于当前会话。

        Returns:
            最新缓存的不可变状态帧（与内部快照同一对象）。

        Raises:
            R1DStateError: 会话不匹配（旧会话一律不可用）、now 非数值或
                非有限（防 NaN 比较吞掉超时判断）、缓存为空、状态过期、
                机型与冻结期望值不一致。
        """
        # I-02 修复（评审 2026-11-17）：会话比较与帧读取同一锁内原子
        # （原锁外比较后、锁内取帧之间 reset 切换会话的竞态会把新会话
        # 帧交给旧会话调用方）。now 校验保持在会话检查之后（维持
        # "会话→now→空→新鲜度→机型"的原检查顺序）。
        with self._lock:
            if session_id != self._session_id:
                raise R1DStateError(
                    f"会话不匹配：缓存当前会话 {self._session_id}，"
                    f"请求会话 {session_id}；旧会话状态一律不可用"
                )
            if isinstance(now, bool) or not isinstance(now, _NUMBER_TYPES):
                raise R1DStateError("now 必须是数值（单调时钟秒）")
            now_value = float(now)
            if not bool(np.isfinite(now_value)):
                raise R1DStateError("now 必须是有限值（单调时钟秒）")
            frame = self._frame
        if frame is None:
            raise R1DStateError("状态缓存为空：尚无有效状态帧")
        if frame.session_id != session_id:
            # 防御：帧会话与请求会话不一致（正常路径锁内已排除，
            # 仅在绕过本方法的并发写入下可达），显式报错不返回异会话帧
            raise R1DStateError(
                f"会话不匹配：缓存帧属会话 {frame.session_id}，"
                f"请求会话 {session_id}（防御性拒绝，不返回异会话帧）"
            )
        age = now_value - frame.received_at
        if age > self._state_timeout_s:
            raise R1DStateError(
                f"状态过期：age={age:.6f}s > "
                f"timeout={self._state_timeout_s}s（接收单调时钟口径）"
            )
        expected = self._expected_mode_machine
        if expected is not None and frame.mode_machine != expected:
            raise R1DStateError(
                f"机型不匹配：期望 mode_machine={expected}，"
                f"实际 {frame.mode_machine}"
            )
        return frame

    def reset(self, session_id: int) -> None:
        """重置缓存并切换到新会话（重启/重连后调用）。

        清空缓存帧与序号基准（新会话首帧不与旧会话序号比较误判乱序），
        回绕计数一并清零；旧会话的帧与调用方自此一律不可用。

        Args:
            session_id: 新会话标识。

        Raises:
            ValueError: session_id 非整数。
        """
        new_session = _require_plain_int(session_id, "session_id")
        with self._lock:
            self._session_id = new_session
            self._frame = None
            self._last_sequence = None
            self._wraparound_count = 0
