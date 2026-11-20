"""R1-A7-D 单臂驱动：生命周期状态机、单一发送线程与看门狗 — 阶段 4 Task 4.6。

实现 `src/real/robot_arm_protocol.py::RobotArmInterface`（位置目标语义），
组合 4.3 命令策略（R1DCommandPolicy）/ 4.4 状态缓存（R1DStateBuffer）/
4.5 传输层（R1DTransport Protocol），交付可测试的驱动生命周期。

状态机（r1d_types.R1D_PHASES）::

    CLOSED --connect--> READ_ONLY --activate--> ACTIVE
    ACTIVE --slow_stop(反馈确认)--> READ_ONLY
    ACTIVE/READ_ONLY --disconnect--> CLOSED（幂等）
    任一运行态 --故障/急停--> FAULT（终态，原对象不得再次 activate）

关键语义（施工计划 Task 4.6 + 主代理裁定 R9/R10/R11 + 协议 manifest）：

- **单一发送线程持有 publish 权**：全部 transport.publish 调用（含停止帧）
  都在该线程内执行；每次实际发布前重查生命周期与会话世代（R10），
  锁存/断开后不得有新运动命令。运动帧的发布许可 = 同一锁内原子判定
  （世代 + ACTIVE + 无停止请求 + 无急停锁存 + 无异步故障，R2 评审
  2026-11-18）：许可授出后的帧视为"已进入传输层的在途帧"（不可撤销），
  许可之前的停止登记必须撤销该帧——停止登记与发布许可在同一把锁上
  互斥。
- **真实发布记录（R9）**：每次实际 transport.publish 调用点记录目标与时刻
  （含停止帧与失败尝试），供 4.8 诊断消费；send 接受时刻不冒充 DDS 发布时刻。
- **超时参数（R11）**：状态超时/控制周期来自硬件配置 timing；命令超时与
  停止确认速度阈值在 4.2 冻结的硬件配置 schema 中暂无字段，经构造参数
  显式注入，缺省即该能力禁用并显式记录（capability_notes），不硬编码
  实机数值。acquire_timeout_s 属传输层构造参数（4.5），不由本层注入。
- **调度纪律**：全部期限基于注入单调时钟的绝对时刻；错过的周期直接
  跳过（对齐到当前时刻之后的下一个期限），不突发补发一串旧包。
  实际发布节奏受控制周期限频（R1，评审 2026-11-18）：相邻两次成功
  active 发布的注入时钟间隔 ≥ control_period_s——发布许可在绝对期限
  到达时消费**最新**在途目标（被替换的旧目标永不发布），上层调用
  频率或突发提交不能改变实际目标变化速率；停止帧不受该限频约束
  （停止路径不得被人为延迟）。
- **停止语义（manifest control 节）**：
  - 缓停 = 有限停止序列（policy.build_stop 锚定新鲜实测姿态，固定帧数、
    按控制周期绝对期限发布）+ 反馈确认（发布后新帧目标臂速度 ≤ 阈值）
    后才退权（"确认再退权"）；
  - 无法构造停止帧 / 停止帧发布失败 / 确认窗口无低速反馈 → 故障锁存 +
    stop_confirmed=False + 明确记录未确认停止；失链停止仅报告尝试，
    不声称已物理急停（manifest control.link_loss 为 null）；
  - 未确认停止不主动退权（机载恢复机制未公开，release.mechanism=null；
    通道清理统一由 disconnect 的 transport.close 承担）；
  - 退权窗口的急停升级不丢失（R3，评审 2026-11-18）：终态提交的同一
    锁内重新合并急停锁存——退权执行期间（release_control 可阻塞）登记
    的急停把已确认缓停的终态强制升级为 FAULT，原对象不得再次 activate；
  - 接管失败回滚 = 不发布任何 lowcmd + 报告未获权
    （manifest control.failure_rollback.our_policy_requirement）。
- **send_joint_command()==0 仅表目标被本驱动接受**；非零码=拒绝
  （1=phase 非 ACTIVE，2=目标非法，3=故障锁存/停止中）；SDK/传输异常
  绝不吞成 0——异步发布失败按"停止策略 + 故障锁存"处理并记录。
- **机型校验（R6）**：expected_mode_machine 冻结前不做相等断言，仅做
  变化检测：ACTIVE/STOPPING 期间 mode_machine 变化按控制权丢失处理，
  READ_ONLY 期间仅记录（manifest dds.mode_machine_validation=null）。

线程模型：connect 启动发送线程（会话世代绑定，daemon），disconnect 按
join_timeout_s 回收；驱动内部锁只做短暂状态读写，任何 transport 调用
都在锁外执行（避免持锁等待线程死锁）。
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from src.real.r1d_command_policy import (
    R1DCommandPolicy,
    R1DCommandPolicyError,
    R1DSlotHold,
)
from src.real.r1d_config import R1DHardwareConfig, validate_r1d_hardware_config
from src.real.r1d_mapping import R1DJointMapping
from src.real.r1d_state_buffer import R1DStateBuffer, R1DStateError
from src.real.r1d_types import (
    R1DCommandFrame,
    R1DStateFrame,
    R1DStatus,
    R1DTransport,
)
from src.robot.robot_profile import RobotProfile

logger = logging.getLogger(__name__)

# 发送线程条件等待的保活轮询上限（真实秒）。仅当"注入时钟的下一绝对期限
# 距离"超过该值时生效（防 fake clock / 长周期下条件等待睡死）；真实时钟
# 下控制周期 ≤ 该值时永不起作用——它不是实机控制参数，全部期限判定仍以
# 注入时钟的绝对时刻为准。
_SCHEDULER_POLL_S = 0.01
# 发送线程 join 默认超时（真实秒；软件工程参数，非实机数值，可注入覆盖）
_DEFAULT_JOIN_TIMEOUT_S = 5.0
# 真实发布记录容量上限（软件内存上限，非实机控制参数）：500 Hz 连续发布
# 可覆盖约 400 s，远超典型诊断时长（diagnostic_duration_s）；超出丢最老
# 记录。4.8 更长窗口应周期调用 publish_records() 增量消费。
R1D_PUBLISH_RECORD_CAPACITY = 200_000

# send_joint_command 返回码（0=接受；非零一律为拒绝）
SEND_ACCEPTED = 0
SEND_REJECT_PHASE = 1  # 未连接 / 非 ACTIVE / 停止中
SEND_REJECT_INVALID_TARGET = 2  # 目标非法（形状 / NaN / Inf / 非数值）
SEND_REJECT_LATCHED = 3  # 故障锁存后拒绝

# 发布记录种类
_RECORD_ACTIVE = "active"
_RECORD_STOP = "stop"


class R1DInterfaceError(RuntimeError):
    """R1D 驱动生命周期错误（activate 时机不当 / 接管失败未获权等）。"""


@dataclass(frozen=True)
class R1DPublishRecord:
    """单次实际 DDS 发布记录（R9：时刻取自 transport.publish 调用返回点）。

    供 4.8 诊断消费：published_at 是真实发布时刻（注入单调时钟秒，同步
    Write 返回点——最接近帧实际上线的可观测时刻），不是上层
    send_joint_command 的接受时刻；停止帧与失败尝试同样记录。

    Attributes:
        session_id: 驱动会话世代（每次 connect 自增，旧会话记录可区分）。
        kind: "active"（运动命令）或 "stop"（停止帧）。
        published_at: publish 调用返回点时刻（注入单调时钟秒）。
        q_sdk: 目标臂 SDK 侧目标（模型序 7 项，只读副本）；提取失败为 None。
        mode_machine: 命令帧回填的机型字段。
        success: transport.publish 是否成功。
        error: 失败原因描述；成功为 None。
    """

    session_id: int
    kind: str
    published_at: float
    q_sdk: np.ndarray | None
    mode_machine: int
    success: bool
    error: str | None


class UnitreeR1DInterface:
    """R1-A7-D 单臂驱动（实现 RobotArmInterface 位置目标语义）。

    状态缓存与命令策略由构造内部按配置创建；全机非目标槽位策略
    （slot_policies）与停止确认阈值等现场数值经构造参数注入——
    4.2 冻结的硬件配置 schema 暂无对应字段，缺省即能力禁用并显式记录。

    Args:
        config: 已加载的 R1D 硬件配置（timing.state_timeout_s/control_period_s
            由此读取；缺项按 R11 记录禁用）。
        profile: R1D 单侧 7 关节机器人描述。
        transport: R1DTransport 实现（真实 SdkR1DTransport 或测试注入 fake）。
        clock: 单调时钟注入（期限/新鲜度判定的唯一时基）。
        slot_policies: 全机非目标槽位显式保持策略（Ruling R5 第一优先级，
            G4 现场确认后接线）；None 时 activate 停止能力预检将拒绝接管。
        expected_mode_machine: 冻结期望机型值；None 表示未冻结（仅变化检测）。
        command_timeout_s: 目标过期看门狗（秒）；None 禁用并记录（R11）。
            配置了 control_period_s 时须 ≥ 2 倍控制周期（命令超时短于
            周期超时判限"接受+2 周期"必然误报，构造即拒绝）。
        stop_sequence_frames: 有限停止序列帧数（默认 1=最小有限序列）。
        stop_confirm_dq_threshold: 停止反馈确认的速度阈值（rad/s，目标臂
            各关节 |dq| 上界）；None 时停止无法反馈确认（stop_confirmed
            恒 False），None 禁用并记录（R11）。
        join_timeout_s: 发送线程 join / 公开停止调用等待完成的上限（真实秒）。

    Note:
        activate() 全程经内部生命周期锁串行化：并发双 activate 恰好一个
        成功，失败方绝不 release_control（控制权只属于成功登记者；
        `_control_serial` 归属标记 + 世代门双保险）。

    Raises:
        ValueError: clock 不可调用或注入参数非法（含 command_timeout_s
            < 2 倍控制周期的配置矛盾）。
    """

    def __init__(
        self,
        config: R1DHardwareConfig,
        profile: RobotProfile,
        transport: R1DTransport,
        *,
        clock: Callable[[], float],
        slot_policies: Mapping[int, R1DSlotHold] | None = None,
        expected_mode_machine: int | None = None,
        command_timeout_s: float | None = None,
        stop_sequence_frames: int = 1,
        stop_confirm_dq_threshold: float | None = None,
        join_timeout_s: float = _DEFAULT_JOIN_TIMEOUT_S,
    ) -> None:
        """校验参数并创建内部组件（策略/缓存/映射；零传输层副作用）。"""
        if not callable(clock):
            raise ValueError("参数 clock 必须是可调用的单调时钟")
        self._clock = clock
        self._config = config
        self._profile = profile
        self._transport = transport

        # —— R11：超时/阈值来源与能力记录 ——
        notes: list[str] = []
        self._state_timeout_s = config.timing.state_timeout_s
        self._control_period_s = config.timing.control_period_s
        if self._state_timeout_s is None:
            notes.append(
                "状态超时未配置（timing.state_timeout_s null）：状态缓存能力禁用，"
                "connect() 将拒绝（R11：无则禁用并记录）"
            )
        if self._control_period_s is None:
            notes.append(
                "控制周期未配置（timing.control_period_s null）：绝对期限调度与"
                "周期看门狗禁用，motion 门槛亦不可满足（R11）"
            )
        if command_timeout_s is None:
            notes.append(
                "命令超时未注入（硬件配置 timing 暂无该字段）：目标过期看门狗禁用（R11）"
            )
        else:
            command_timeout_s = self._require_positive(
                command_timeout_s, "command_timeout_s"
            )
            # 配置矛盾防御：命令超时短于周期超时判限（接受+2 周期）必然误报
            if (
                self._control_period_s is not None
                and command_timeout_s < 2.0 * self._control_period_s
            ):
                raise ValueError(
                    f"参数 command_timeout_s={command_timeout_s}s 不得小于 2 倍控制周期"
                    f"（{2.0 * self._control_period_s}s）：命令超时短于周期超时判限"
                    "（接受+2 周期）必然误报"
                )
        self._command_timeout_s = command_timeout_s
        if stop_confirm_dq_threshold is None:
            notes.append(
                "停止确认速度阈值未注入：停止无法反馈确认（stop_confirmed 恒 False，"
                "确认路径禁用）（R11）"
            )
        else:
            stop_confirm_dq_threshold = self._require_non_negative(
                stop_confirm_dq_threshold, "stop_confirm_dq_threshold"
            )
        self._stop_confirm_dq_threshold = stop_confirm_dq_threshold
        if slot_policies is None:
            notes.append(
                "未注入非目标槽位显式策略：activate 停止能力预检将拒绝接管"
                "（G4 现场确认后接线，Ruling R5）"
            )
        if expected_mode_machine is None:
            notes.append(
                "期望机型值未冻结：仅做 mode_machine 变化检测+记录，不做相等断言"
                "（R6；manifest dds.mode_machine_validation=null）"
            )
        self._capability_notes: tuple[str, ...] = tuple(notes)

        # stop_sequence_frames：有限停止序列帧数（≥1，默认 1=最小有限序列）
        if (
            isinstance(stop_sequence_frames, bool)
            or not isinstance(stop_sequence_frames, int)
            or stop_sequence_frames < 1
        ):
            raise ValueError(
                f"参数 stop_sequence_frames 必须是 ≥1 的整数，实际 {stop_sequence_frames}"
            )
        self._stop_sequence_frames = stop_sequence_frames
        self._join_timeout_s = self._require_positive(
            join_timeout_s, "join_timeout_s"
        )

        # —— 内部组件（构造内部按配置创建；策略/缓存共享同一注入时钟）——
        self._policy = R1DCommandPolicy(
            config,
            profile,
            slot_policies=slot_policies,
            expected_mode_machine=expected_mode_machine,
            clock=clock,
        )
        self._buffer: R1DStateBuffer | None = None
        if self._state_timeout_s is not None:
            self._buffer = R1DStateBuffer(
                session_id=0,
                state_timeout_s=self._state_timeout_s,
                expected_mode_machine=expected_mode_machine,
            )
        # 模型序映射（校准 null 时延迟到使用处抛显式错误）
        self._mapping: R1DJointMapping | None = None

        # —— 运行状态（内部锁保护；transport 调用一律在锁外）——
        # 生命周期锁：activate 全程串行化（并发双 activate 恰好一个成功，
        # 失败方不触碰控制权）；锁序恒为 lifecycle → lock，单向无死锁
        self._lifecycle_lock = threading.Lock()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._phase = "CLOSED"
        self._generation = 0  # 会话世代：每次 connect 自增，disconnect 亦自增失效
        self._fault_reason: str | None = None
        self._stop_confirmed = False
        self._last_publish_at: float | None = None
        self._pending_target: np.ndarray | None = None
        self._pending_accepted_at: float | None = None
        self._last_accept_at: float | None = None
        self._stop_request: str | None = None  # "slow" | "emergency"
        self._emergency_latch = False
        self._async_fault_reason: str | None = None  # 回调侧异常事件（控制权丢失）
        self._activation_mode_machine: int | None = None
        self._next_tick_deadline: float | None = None
        # R1（评审 2026-11-18）：实际发布节奏期限（注入时钟绝对时刻）。
        # 相邻两次成功 active 发布间隔 ≥ control_period_s；activate 时重置
        # 为接管时刻（首帧锚定目标可立即发布），每次成功发布推进一个周期。
        # None=未配置控制周期（限频禁用；motion 门槛下不可达）。
        self._next_publish_at: float | None = None
        self._sender_thread: threading.Thread | None = None
        self._stop_done = threading.Event()
        self._stop_done.set()
        # 控制权归属标记（I-1 修复）：成功接管的 activate 登记其序号；
        # 退权成功/断开清 None。回滚 release 仅在"世代未变且无归属者"时执行
        self._activate_counter = 0
        self._control_serial: int | None = None
        # 真实发布记录：有界 deque（容量见 R1D_PUBLISH_RECORD_CAPACITY）
        self._publish_records: deque[R1DPublishRecord] = deque(
            maxlen=R1D_PUBLISH_RECORD_CAPACITY
        )

    # ------------------------------------------------------------------
    # 静态校验辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _require_positive(value: float, name: str) -> float:
        """校验正有限浮点参数。"""
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"参数 {name} 必须为正有限值，实际 {value}")
        return number

    @staticmethod
    def _require_non_negative(value: float, name: str) -> float:
        """校验非负有限浮点参数。"""
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"参数 {name} 必须为非负有限值，实际 {value}")
        return number

    # ------------------------------------------------------------------
    # 公开观测
    # ------------------------------------------------------------------

    @property
    def capability_notes(self) -> tuple[str, ...]:
        """构造期确定的能力禁用/降级记录（R11：无则禁用并显式记录）。"""
        return self._capability_notes

    def get_status(self) -> R1DStatus:
        """返回驱动状态快照（线程安全）。

        Note:
            FAULT 为**会话内终态**（C-M8，评审 2026-11-17）：锁存原因仅存于
            本对象内存，重建驱动对象后无历史痕迹。跨对象/跨进程的故障
            审计由 4.8 诊断产物（JSON fail 结论与日志）承担，不在本类型
            范围（R1DStatus 为 4.2 冻结结构，不加字段）。
        """
        with self._lock:
            return R1DStatus(
                phase=self._phase,
                session_id=self._generation,
                fault_reason=self._fault_reason,
                stop_confirmed=self._stop_confirmed,
                last_publish_at=self._last_publish_at,
            )

    def publish_records(self) -> tuple[R1DPublishRecord, ...]:
        """返回全部真实发布记录的元组快照（R9，含停止帧与失败尝试）。

        记录有界：内部 deque 容量 R1D_PUBLISH_RECORD_CAPACITY=200_000
        （软件内存上限，非实机控制参数；500 Hz 连续发布约覆盖 400 s，
        远超典型诊断时长），超出丢最老。4.8 更长诊断窗口应周期调用本方法
        增量消费，而非依赖全量驻留。
        """
        with self._lock:
            return tuple(self._publish_records)

    # ------------------------------------------------------------------
    # RobotArmInterface：connect / disconnect
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """连接：只订阅状态（READ_ONLY），绝不发布或接管。

        Returns:
            是否成功。失败（read_only 门槛未满足 / 状态缓存不可用 /
            传输层 open 异常）返回 False 并记录原因，phase 保持 CLOSED。

        Note:
            READ_ONLY 下重复 connect 幂等返回 True；FAULT/CLOSED 之外的
            运行态调用 connect 返回 False（须先停止/断开）。
        """
        with self._lock:
            if self._phase == "READ_ONLY":
                return True  # 幂等
            if self._phase != "CLOSED":
                logger.warning("connect 拒绝：当前 phase=%s 须先 disconnect", self._phase)
                return False
            if self._buffer is None:
                logger.warning(
                    "connect 拒绝：状态超时未配置，状态缓存能力禁用（见 capability_notes）"
                )
                return False
        try:
            validate_r1d_hardware_config(self._config, self._profile, purpose="read_only")
        except ValueError as error:
            logger.warning("connect 失败：read_only 门槛未满足: %s", error)
            return False

        with self._lock:
            self._generation += 1
            generation = self._generation
            assert self._buffer is not None
            self._buffer.reset(generation)
            self._phase = "READ_ONLY"
            self._fault_reason = None
            self._stop_confirmed = False
            self._activation_mode_machine = None
            self._emergency_latch = False
            self._control_serial = None  # 新会话起点：无控制权归属
            self._pending_target = None
            self._pending_accepted_at = None
            # I-03 修复（评审 2026-11-17）：旧会话未消费的异步故障事件
            # （如 disconnect 前最后一刻登记的控制权丢失疑似）不得泄漏进
            # 新会话——新会话的 _decide 消费它会触发 READ_ONLY 下停止序列
            # 存活复查失败 → 重连即 FAULT
            self._async_fault_reason = None
        try:
            self._transport.open(self._make_state_callback(generation))
        except Exception as error:
            with self._lock:
                self._generation += 1  # 使半建立会话立即失效
                assert self._buffer is not None
                self._buffer.reset(self._generation)
                self._phase = "CLOSED"
            logger.warning("connect 失败：传输层 open 异常: %s", error)
            return False

        thread = threading.Thread(
            target=self._sender_loop,
            args=(generation,),
            name=f"r1d-sender-{generation}",
            daemon=True,
        )
        with self._lock:
            # C-M1 修复（评审 2026-11-17）：open 执行期间可能被并发 disconnect
            # 打断（世代已失效、通道已被 close 清理）——锁内复查世代后再登记，
            # 不启动注定退出的线程，也不向调用方虚报连接成功
            if self._generation != generation:
                interrupted = True
            else:
                self._sender_thread = thread
                interrupted = False
        if interrupted:
            logger.warning("connect 被并发 disconnect 打断：放弃发送线程（未连接）")
            return False
        thread.start()
        logger.info("connect 成功：会话世代 %d，进入 READ_ONLY", generation)
        return True

    def disconnect(self) -> None:
        """断开（幂等）：世代失效 → 回收发送线程（join_timeout_s）→ 关闭通道。

        join 超时不阻塞断开：世代门已保证滞留线程不再发布；
        transport.close 异常仅记录（通道清理不完整风险显式留痕）。

        Warning:
            ACTIVE 下直接 disconnect **不执行停止序列、不释放控制权**
            （C-M3，评审 2026-11-17）：本方法只做世代失效与通道清理，
            不会发布任何停止帧。运动中须先 ``slow_stop()`` 确认停止后
            再断开；未停止即断开时控制权交由机端超时/人工处置
            （manifest control.release.mechanism=null，无已核验远端恢复）。
        """
        with self._lock:
            self._generation += 1
            self._phase = "CLOSED"
            self._pending_target = None
            self._pending_accepted_at = None
            self._stop_request = None
            self._activation_mode_machine = None
            self._emergency_latch = False
            self._async_fault_reason = None  # I-03：异步事件随会话终结，不跨会话泄漏
            self._control_serial = None  # 通道由 transport.close 统一清理
            thread = self._sender_thread
            self._sender_thread = None
            self._cond.notify_all()
            self._stop_done.set()
        if thread is not None and thread.is_alive():
            thread.join(self._join_timeout_s)
            if thread.is_alive():
                logger.warning(
                    "发送线程 join 超时（%.3fs）：线程仍在途，世代门已保证其不再发布",
                    self._join_timeout_s,
                )
        try:
            self._transport.close()
        except Exception as error:
            logger.warning("transport.close 异常（通道清理不完整风险记录）: %s", error)
        logger.info("disconnect 完成：phase=CLOSED")

    # ------------------------------------------------------------------
    # 驱动扩展：activate（接管）
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """校验新鲜状态/机型/参数与停止能力后接管整机控制权（→ACTIVE）。

        全程经内部生命周期锁串行化：并发双 activate 恰好一个成功；失败方
        绝不 release_control（控制权只属于成功登记者）。

        校验顺序：phase 门槛 → motion 级参数 → 新鲜状态（机型由缓存按
        冻结期望校验）→ 停止能力预检（build_stop 必须可构造，覆盖全机
        策略/增益/限位可用性）→ transport.acquire_control。

        Raises:
            R1DInterfaceError: phase 不当、motion 门槛未满足、无新鲜状态、
                停止能力预检失败，或接管失败（回滚：不发布任何 lowcmd +
                报告未获权，manifest failure_rollback）。
        """
        with self._lifecycle_lock:
            self._activate_locked()

    def _activate_locked(self) -> None:
        """activate 主体（已持生命周期锁；见 activate 文档）。"""
        with self._lock:
            phase = self._phase
            generation = self._generation
            # 取本次尝试序号（成功接管时登记为控制权归属标记）
            self._activate_counter += 1
            my_serial = self._activate_counter
        if phase == "CLOSED":
            raise R1DInterfaceError("activate 拒绝：须先 connect() 进入 READ_ONLY")
        if phase == "ACTIVE":
            raise R1DInterfaceError("activate 拒绝：已处于 ACTIVE")
        if phase == "STOPPING":
            raise R1DInterfaceError("activate 拒绝：停止进行中，须等待完成")
        if phase == "FAULT":
            raise R1DInterfaceError(
                "activate 拒绝：故障锁存后原对象不得再次 activate（请重建驱动对象）"
            )

        # 1) motion 级参数门槛（校准/限值/周期等；null 一律拒绝）
        try:
            validate_r1d_hardware_config(self._config, self._profile, purpose="motion")
        except ValueError as error:
            raise R1DInterfaceError(f"activate 拒绝：motion 门槛未满足: {error}") from error

        # 2) 新鲜状态（缓存按 expected_mode_machine 冻结值校验机型）
        buffer = self._require_buffer()
        try:
            state = buffer.latest(now=self._clock(), session_id=generation)
        except R1DStateError as error:
            raise R1DInterfaceError(f"activate 拒绝：无新鲜状态: {error}") from error

        # 3) 停止能力预检：接管前必须能构造有限停止帧（含全机策略可用性）
        preflight_error: Exception | None = None
        try:
            preflight = self._policy.build_stop(state, emergency=True)
        except R1DCommandPolicyError as error:
            preflight, preflight_error = None, error
        if preflight is None:
            raise R1DInterfaceError(
                f"activate 拒绝：停止能力预检失败（无法构造有限停止帧）: {preflight_error}"
            )

        # 4) 接管；失败回滚 = 不发布任何 lowcmd + 报告未获权
        try:
            self._transport.acquire_control()
        except Exception as error:
            released, rollback_error = self._rollback_release_control(generation)
            message = f"activate 失败：未获控制权（未发布任何 lowcmd）: {error}"
            if not released:
                message += "；回滚跳过 release_control（世代已失效或控制权属其他接管者）"
            elif rollback_error is not None:
                message += f"；回滚 release_control 失败: {rollback_error}"
            logger.warning(message)
            raise R1DInterfaceError(message) from error

        # 5) ACTIVE（此刻零发布；首帧运动命令以实测姿态锚定）
        #    竞态回滚区分（I-1）：世代失效→静默退场（close 已清理，不
        #    release）；另一接管者已成功→绝不释放他人控制权。两种情形都
        #    未发布任何命令。
        #    C-M4（评审 2026-11-17）：overtaken 分支经 lifecycle_lock 串行化
        #    后正常路径不可达（同代 activate 串行、异代走 stale），保留为
        #    纵深防御——`_control_serial` 归属检查失效时仍不误释放他人控制权
        with self._lock:
            stale = self._generation != generation
            overtaken = (not stale) and (
                self._phase != "READ_ONLY" or self._control_serial is not None
            )
            if not stale and not overtaken:
                now = self._clock()
                self._phase = "ACTIVE"
                self._fault_reason = None
                self._stop_confirmed = False
                self._activation_mode_machine = state.mode_machine
                # 登记控制权归属：此后仅本序号的退权路径可 release
                self._control_serial = my_serial
                # 命令超时基线：接管后须在 command_timeout_s 内开始发送目标
                self._last_accept_at = now
                if self._control_period_s is not None:
                    self._next_tick_deadline = now + self._control_period_s
                    # R1：发布节奏期限重置为接管时刻——首帧锚定目标可立即
                    # 发布，其后每次成功 active 发布推进一个控制周期
                    self._next_publish_at = now
                self._cond.notify_all()
        if stale or overtaken:
            logger.warning(
                "activate 竞态失败（未发布任何命令，不释放控制权）：%s",
                "会话世代已失效（disconnect 已关闭通道）" if stale else "另一接管者已持有控制权",
            )
            raise R1DInterfaceError(
                "activate 竞态失败：接管期间生命周期已变化（未发布任何命令；"
                "控制权归属未受影响）"
            )
        logger.info(
            "activate 成功：世代 %d 接管（mode_machine=%d，零发布）",
            generation,
            state.mode_machine,
        )

    def _rollback_release_control(self, generation: int) -> tuple[bool, Exception | None]:
        """activate 失败回滚退权：仅当世代未变且无成功接管者时才 release。

        Returns:
            (是否执行了 release_control, release 异常或 None)。
        """
        with self._lock:
            may_release = (
                self._generation == generation and self._control_serial is None
            )
        if not may_release:
            return False, None
        try:
            self._transport.release_control()
        except Exception as error:
            return True, error
        return True, None

    # ------------------------------------------------------------------
    # RobotArmInterface：状态读取 / 命令发送 / 停止
    # ------------------------------------------------------------------

    def get_arm_state(self) -> np.ndarray:
        """读取目标臂关节状态：(14,)=[q(7), dq(7)] 模型序（rad, rad/s）。

        Returns:
            拼接数组（位置在前、速度在后，按 profile.active_joint_names 序）。

        Raises:
            R1DStateError: 未连接、校准 null 无法构造模型序映射、缓存空、
                状态过期或机型不符（显式错误，不返回旧缓存冒充新状态）。
        """
        with self._lock:
            phase = self._phase
            generation = self._generation
        if phase not in ("READ_ONLY", "ACTIVE", "STOPPING"):
            raise R1DStateError(f"状态不可用：当前 phase={phase}（未连接或已断开）")
        # 校准缺失时在此显式失败（错误类型与信息保持既有契约；映射实际
        # 使用点在同帧提取入口 _model_state_from_frame）
        self._require_mapping()
        buffer = self._require_buffer()
        frame = buffer.latest(now=self._clock(), session_id=generation)
        return self._model_state_from_frame(frame)

    def _model_state_from_frame(self, frame: R1DStateFrame) -> np.ndarray:
        """从给定状态帧提取模型序 [q(7), dq(7)]（同帧原子）。

        4.8 诊断采样的同包内部契约入口（与 _clock/_buffer 同级，跨包调用
        方不应依赖）：调用方已从缓存取得帧时，模型序状态必须出自**同一
        帧**——独立两次读取会让 q 与接收时刻分属不同帧，误差配对据此把
        发布前旧缓存伪装成发布后新反馈（R5，评审 2026-11-18）。
        """
        mapping = self._require_mapping()
        return np.concatenate(
            [mapping.extract(frame), mapping.extract_velocities(frame)]
        )

    def send_joint_command(self, q_desired: np.ndarray) -> int:
        """验证并交付最新目标（仅验证；实际发布由单一发送线程执行）。

        Args:
            q_desired: 模型序 7 关节目标位置（rad）。每次调用替换在途目标
                （被替换的旧目标永不发布）。

        Returns:
            状态码：0=目标被本驱动接受（不代表硬件执行成功）；
            1=phase 非 ACTIVE（未连接/只读/停止中）；2=目标非法；
            3=故障锁存后拒绝。
        """
        with self._lock:
            phase = self._phase
        if phase == "FAULT" or (phase == "STOPPING" and self._emergency_latch):
            return SEND_REJECT_LATCHED
        if phase != "ACTIVE":
            return SEND_REJECT_PHASE
        # 同步结构校验（形状/有限性）；限位/步长/新鲜度在发布时由策略复查
        try:
            target = np.array(q_desired, dtype=np.float64, copy=True)
        except (TypeError, ValueError):
            return SEND_REJECT_INVALID_TARGET
        if target.shape != (7,):
            return SEND_REJECT_INVALID_TARGET
        if not bool(np.all(np.isfinite(target))):
            return SEND_REJECT_INVALID_TARGET
        with self._lock:
            if self._phase != "ACTIVE":
                # 竞态：校验期间被停止/锁存
                return (
                    SEND_REJECT_LATCHED
                    if self._phase in ("FAULT",)
                    else SEND_REJECT_PHASE
                )
            self._pending_target = target
            accepted_at = self._clock()
            self._pending_accepted_at = accepted_at
            self._last_accept_at = accepted_at
            self._cond.notify_all()
        return SEND_ACCEPTED

    def slow_stop(self) -> None:
        """缓停（可恢复）：请求有限停止序列，阻塞等待其完成（有界）。

        ACTIVE 下触发停止序列并在反馈确认后退权回 READ_ONLY；
        未确认（无阈值/无新鲜反馈/发布失败）则故障锁存。STOPPING 中重复
        调用幂等（等待在途序列完成）；READ_ONLY/CLOSED/FAULT 下 no-op。
        """
        initiated = self._request_stop(emergency=False)
        if initiated and not self._stop_done.wait(self._join_timeout_s):
            logger.warning("缓停未在 %.3fs 内完成（发送线程仍在途）", self._join_timeout_s)

    def emergency_stop(self) -> None:
        """急停（不可恢复）：打断排队目标，最终锁存 FAULT。

        软件层锁存停止请求并尽力发布有限停止帧；不承诺硬件急停效果，
        失链时仅报告尝试（stop_confirmed 如实反映反馈确认结果）。
        """
        initiated = self._request_stop(emergency=True)
        if initiated and not self._stop_done.wait(self._join_timeout_s):
            logger.warning("急停未在 %.3fs 内完成（发送线程仍在途）", self._join_timeout_s)

    def _request_stop(self, *, emergency: bool) -> bool:
        """登记停止请求（锁内原子），返回是否存在等待中的停止流程。"""
        with self._lock:
            if self._phase == "ACTIVE":
                self._stop_request = "emergency" if emergency else "slow"
                if emergency:
                    self._emergency_latch = True
                # 急停/缓停均打断排队目标（未被拾取的目标永不发布）
                self._pending_target = None
                self._pending_accepted_at = None
                self._stop_done.clear()
                self._cond.notify_all()
                return True
            if self._phase == "STOPPING":
                if emergency:
                    # 急停升级在途缓停：终态强制 FAULT
                    self._emergency_latch = True
                return True  # 等待在途停止完成（重复 stop 幂等）
            return False

    # ------------------------------------------------------------------
    # 内部：状态回调（世代门 + 机型变化检测 + 会话重标记入缓存）
    # ------------------------------------------------------------------

    def _make_state_callback(self, generation: int) -> Callable[[R1DStateFrame], None]:
        """构造绑定会话世代的状态回调（close/重连后旧回调丢弃）。"""

        def on_state(frame: R1DStateFrame) -> None:
            """状态回调：只做世代门/变化检测/入缓存，无 RPC/DDS/长耗时操作。"""
            with self._lock:
                if generation != self._generation or self._phase in ("CLOSED", "FAULT"):
                    return  # 旧会话迟到帧 / 已锁存：丢弃
                # 机型变化检测（R6：冻结前只做变化检测+记录，不做相等断言）
                reference = self._activation_mode_machine
                if (
                    reference is not None
                    and self._phase in ("ACTIVE", "STOPPING")
                    and frame.mode_machine != reference
                    and self._async_fault_reason is None
                ):
                    reason = (
                        f"控制权丢失疑似：ACTIVE 期间 mode_machine {reference}→"
                        f"{frame.mode_machine}（机型字段变化，按控制权丢失处理）"
                    )
                    self._async_fault_reason = reason
                    logger.error("%s", reason)
                    self._cond.notify_all()
                # 会话重标记：传输层会话 → 驱动会话世代（缓存世代门的隔离依据）
                restamped = R1DStateFrame(
                    q=frame.q,
                    dq=frame.dq,
                    sequence=frame.sequence,
                    source_time=frame.source_time,
                    received_at=frame.received_at,
                    mode_machine=frame.mode_machine,
                    session_id=generation,
                )
                buffer = self._buffer
            if buffer is None:  # 防御：connect 已保证 buffer 存在
                return
            try:
                accepted = buffer.push(restamped)
            except (TypeError, R1DStateError) as error:
                logger.error("状态帧入缓存被拒（丢弃该帧）: %s", error)
                return
            with self._lock:
                self._cond.notify_all()  # 唤醒发送线程（新帧可能完成确认/新鲜度判定）
            if not accepted:
                logger.debug("状态帧被缓存丢弃（重复/乱序）")

        return on_state

    # ------------------------------------------------------------------
    # 内部：单一发送线程（唯一 publish 所有者）
    # ------------------------------------------------------------------

    def _sender_loop(self, generation: int) -> None:
        """发送线程主循环：决策（锁内）→ 执行（锁外，transport 调用无锁）。"""
        logger.debug("发送线程启动：世代 %d", generation)
        try:
            while True:
                action = self._decide(generation)
                if action is None:
                    return  # 世代失效 / 已关闭 / 已锁存
                kind, payload = action
                if kind == "stop":
                    self._run_stop_sequence(
                        generation,
                        trigger="用户缓停请求" if payload == "slow" else "用户急停请求",
                        final_fault=payload == "emergency",
                    )
                elif kind == "fault":
                    self._run_stop_sequence(
                        generation, trigger=str(payload), final_fault=True
                    )
                elif kind == "publish":
                    self._attempt_publish(generation)
                elif kind == "watchdog":
                    reason = self._watchdog_checks(generation)
                    if reason is not None:
                        self._run_stop_sequence(generation, trigger=reason, final_fault=True)
        except Exception:
            logger.exception("发送线程内部异常：锁存故障并停止发布运动命令")
            with self._lock:
                # C-M2 修复（评审 2026-11-17）：READ_ONLY 会话同样锁存——
                # 唯一发布者死亡后若静默保持 READ_ONLY，后续 activate 会在
                # "无发布线程"状态下虚假成功（目标永不上线、看门狗失能）
                if self._generation == generation and self._phase in (
                    "ACTIVE",
                    "STOPPING",
                    "READ_ONLY",
                ):
                    read_only = self._phase == "READ_ONLY"
                    self._phase = "FAULT"
                    self._fault_reason = (
                        "发送线程内部异常（只读会话，锁存防止带病 activate）"
                        if read_only
                        else "发送线程内部异常（未确认停止）"
                    )
                    self._stop_confirmed = False
                    self._cond.notify_all()
        finally:
            self._stop_done.set()
            logger.debug("发送线程退出：世代 %d", generation)

    def _decide(self, generation: int) -> tuple[str, object] | None:
        """决策下一步动作（锁内等待；返回 None 表示线程应退出）。"""
        period = self._control_period_s
        with self._cond:
            while True:
                if self._generation != generation or self._phase in ("CLOSED", "FAULT"):
                    return None
                if self._stop_request is not None:
                    action: tuple[str, object] = ("stop", self._stop_request)
                    self._stop_request = None
                    if self._phase == "ACTIVE":
                        # I-01 修复（评审 2026-11-17）：取走停止请求的同一锁内
                        # 先占住 STOPPING。否则"请求已取走、停止序列未开跑"的
                        # 窗口内重复 slow_stop()/emergency_stop() 会重新登记
                        # 请求并 clear 完成事件，第二段停止在 READ_ONLY 下
                        # 存活复查失败，误锁 FAULT 并覆盖第一段的真实停止确认
                        self._phase = "STOPPING"
                    return action
                if self._async_fault_reason is not None:
                    if self._phase in ("ACTIVE", "STOPPING"):
                        action = ("fault", self._async_fault_reason)
                        self._async_fault_reason = None
                        if self._phase == "ACTIVE":
                            # 同 I-01：故障停止同样在消费事件的同一锁内
                            # 占住停止流程，消除重复 stop 登记窗口
                            self._phase = "STOPPING"
                        return action
                    # I-03 防御：READ_ONLY 下出现残留异步故障事件属会话隔离
                    # 缺陷（事件只在 ACTIVE/STOPPING 由回调产生），丢弃留痕，
                    # 不让旧事件把新会话推入停止/FAULT
                    leaked = self._async_fault_reason
                    self._async_fault_reason = None
                    logger.warning(
                        "丢弃 READ_ONLY 下残留的异步故障事件（会话隔离防御）: %s",
                        leaked,
                    )
                if self._phase == "ACTIVE":
                    now = self._clock()
                    deadline = self._next_tick_deadline
                    if period is None or deadline is None or now >= deadline:
                        return ("watchdog", None)
                    if self._pending_target is not None:
                        # R1（评审 2026-11-18）：发布受绝对期限限频——周期
                        # 未到不得多发（pending target 即发送会让高频提交
                        # 在一个周期内累积多个允许步长，绕过策略按周期
                        # 计算的步长限速）。期限到达时消费最新目标（等待
                        # 期间新目标替换旧目标，被替换者永不发布）。
                        publish_at = self._next_publish_at
                        if publish_at is None or now >= publish_at:
                            return ("publish", None)
                        self._cond.wait(
                            min(publish_at - now, deadline - now, _SCHEDULER_POLL_S)
                        )
                        continue
                    self._cond.wait(min(deadline - now, _SCHEDULER_POLL_S))
                else:
                    # READ_ONLY/STOPPING 空闲：等待通知（停止序列在各自流程内调度）
                    self._cond.wait(_SCHEDULER_POLL_S)

    def _watchdog_checks(self, generation: int) -> str | None:
        """看门狗节拍检查（仅 ACTIVE）：命令超时 / 状态过期 / 节拍对齐。

        Returns:
            触发停止的故障原因；无异常返回 None。
        """
        now = self._clock()
        # 节拍对齐：绝对期限推进，错过的周期直接跳过（不补发）
        with self._lock:
            period = self._control_period_s
            deadline = self._next_tick_deadline
            last_accept = self._last_accept_at
        if period is not None and deadline is not None:
            overdue = now - deadline
            if overdue < period:
                with self._lock:
                    self._next_tick_deadline = deadline + period
            else:
                skipped = int(math.ceil(overdue / period))
                with self._lock:
                    self._next_tick_deadline = deadline + skipped * period
                logger.debug("看门狗节拍滞后 %d 个周期（已跳过，不补发）", skipped - 1)
        # 命令超时（未注入则该能力禁用，R11）
        if self._command_timeout_s is not None:
            if last_accept is None or (now - last_accept) > self._command_timeout_s:
                idle = "从未接受目标" if last_accept is None else f"{now - last_accept:.6f}s"
                return (
                    f"目标命令过期：距最近接受 {idle} 超过 "
                    f"command_timeout_s={self._command_timeout_s}s"
                )
        # 状态过期（超时标准来自硬件配置）
        buffer = self._require_buffer()
        try:
            buffer.latest(now=now, session_id=generation)
        except R1DStateError as error:
            return f"状态过期/不可用：{error}"
        return None

    def _attempt_publish(self, generation: int) -> None:
        """发布在途最新目标（发布前重查时效/状态/生命周期/世代）。"""
        with self._lock:
            target = self._pending_target
            accepted_at = self._pending_accepted_at
            self._pending_target = None
            self._pending_accepted_at = None
        if target is None or accepted_at is None:
            return  # 已被停止请求清除
        now = self._clock()
        period = self._control_period_s
        # 目标过期复查（命令超时注入时启用）
        if (
            self._command_timeout_s is not None
            and (now - accepted_at) > self._command_timeout_s
        ):
            self._run_stop_sequence(
                generation,
                trigger=(
                    f"目标命令过期：目标接受后 {(now - accepted_at):.6f}s 未发布，"
                    f"超过 command_timeout_s={self._command_timeout_s}s（过期目标不补发）"
                ),
                final_fault=True,
            )
            return
        # 周期超时复查：交付期限 = 接受 + 1 周期，宽限 1 周期，
        # 判限 = 接受 + 2 周期（全来自配置周期；消息与判定同口径）
        if period is not None and now > accepted_at + 2.0 * period:
            self._run_stop_sequence(
                generation,
                trigger=(
                    f"发送周期超时：目标错过交付期限（接受 {accepted_at:.6f}，"
                    f"当前 {now:.6f}；交付期限=接受+1 周期="
                    f"{accepted_at + period:.6f}，判限=接受+2 周期="
                    f"{accepted_at + 2.0 * period:.6f}），过期目标丢弃不补发"
                ),
                final_fault=True,
            )
            return
        # 状态新鲜度 + 全量构造校验（策略层：新鲜度/机型/限位/步长/全机策略）
        buffer = self._require_buffer()
        try:
            state = buffer.latest(now=self._clock(), session_id=generation)
        except R1DStateError as error:
            self._run_stop_sequence(
                generation,
                trigger=f"状态过期/不可用，拒绝发布运动命令：{error}",
                final_fault=True,
            )
            return
        try:
            frame = self._policy.build_active(state, target)
        except R1DCommandPolicyError as error:
            self._run_stop_sequence(
                generation,
                trigger=f"运动命令构造被拒绝：{error}",
                final_fault=True,
            )
            return
        # R2/R10（评审 2026-11-18）：发布许可 = 同一锁内原子判定。许可
        # 条件在 R10 的世代 + ACTIVE 之上合并停止请求/急停锁存/异步故障
        # ——停止登记（_request_stop / 急停 / 控制权丢失回调）全部在同一
        # 把锁上写入，许可判定在其之后即可确定性地撤销"已构造、尚未进入
        # 传输层"的运动帧。许可授出后本帧即视为在途帧（不可撤销）。
        with self._lock:
            permitted = (
                self._generation == generation
                and self._phase == "ACTIVE"
                and self._stop_request is None
                and not self._emergency_latch
                and self._async_fault_reason is None
            )
        if not permitted:
            logger.info(
                "发布许可拒绝：停止请求/急停/故障/世代或生命周期变化，"
                "撤销未发布运动帧（在途许可帧不受影响）"
            )
            return
        try:
            self._transport.publish(frame)
        except Exception as error:
            # 记录时刻取自 publish 调用返回点（最接近实际上线时刻）
            published_at = self._clock()
            self._record_publish(
                generation,
                _RECORD_ACTIVE,
                None,
                frame.mode_machine,
                published_at,
                False,
                f"{type(error).__name__}: {error}",
            )
            self._run_stop_sequence(
                generation,
                trigger=f"命令发布异常（触发停止策略）：{type(error).__name__}: {error}",
                final_fault=True,
            )
            return
        published_at = self._clock()
        if period is not None:
            with self._lock:
                # R1（评审 2026-11-18）：按实际发布时刻推进下一发布期限
                # （绝对时刻）。发布晚于原期限时自实际时刻顺延一个周期，
                # 不按错过的期限追赶连发——相邻成功 active 发布的注入
                # 时钟间隔恒 ≥ control_period_s。
                self._next_publish_at = published_at + period
        self._record_publish(
            generation,
            _RECORD_ACTIVE,
            self._target_q_sdk(frame),
            frame.mode_machine,
            published_at,
            True,
            None,
        )

    # ------------------------------------------------------------------
    # 内部：有限停止序列 + 反馈确认 + 退权（仅发送线程调用）
    # ------------------------------------------------------------------

    def _run_stop_sequence(self, generation: int, *, trigger: str, final_fault: bool) -> None:
        """执行停止策略：有限停止序列 → 反馈确认 → 确认后退权。

        无法构造停止帧 / 发布失败 / 确认窗口无低速反馈 → 故障锁存 +
        stop_confirmed=False + 明确记录未确认停止；未确认不退权。
        """
        logger.warning("触发停止策略（%s）：%s", "故障" if final_fault else "缓停", trigger)
        with self._lock:
            if self._generation != generation:
                return
            if self._phase == "ACTIVE":
                # 兜底置位：主路径（用户停止请求/异步故障）已在 _decide
                # 取走请求的同一锁内置 STOPPING（I-01）；此处覆盖
                # _attempt_publish/_watchdog_checks 直接调用的路径
                self._phase = "STOPPING"
            self._pending_target = None
            self._pending_accepted_at = None
            self._cond.notify_all()

        # 1) 构造停止帧（锚定新鲜实测姿态；无新鲜状态则无法安全构造）
        build_error: Exception | None = None
        frame: R1DCommandFrame | None = None
        buffer = self._require_buffer()
        try:
            state = buffer.latest(now=self._clock(), session_id=generation)
        except R1DStateError as error:
            state = None
            build_error = error
        if state is not None:
            try:
                frame = self._policy.build_stop(state, emergency=final_fault)
            except R1DCommandPolicyError as error:
                frame = None
                build_error = error
        if frame is None:
            self._finish_stop(
                generation,
                trigger,
                outcome=(
                    f"停止未确认：无法构造停止帧（{build_error}）；未发布停止帧，"
                    "不退权、不声称已停"
                ),
                confirmed=False,
                final_fault=True,
            )
            return

        q_sdk = self._target_q_sdk(frame)
        mode_machine = frame.mode_machine
        count = self._stop_sequence_frames
        period = self._control_period_s
        last_publish_at: float | None = None
        # 2) 有限停止序列：固定帧数，按控制周期绝对期限发布；每帧发布前重查
        for index in range(count):
            if index > 0:
                assert last_publish_at is not None  # 第 0 帧迭代已发布并赋值
                if period is None or not self._wait_deadline(generation, last_publish_at + period):
                    # 世代失效/生命周期变化：中止序列（不补发）
                    self._finish_stop(
                        generation,
                        trigger,
                        outcome="停止序列中止：等待发布期限期间世代/生命周期失效（停止未确认，不声称已停）",
                        confirmed=False,
                        final_fault=True,
                    )
                    return
            with self._lock:
                alive = self._generation == generation and self._phase == "STOPPING"
            if not alive:
                self._finish_stop(
                    generation,
                    trigger,
                    outcome="停止序列中止：发布前生命周期/世代复查失败（停止未确认，不声称已停）",
                    confirmed=False,
                    final_fault=True,
                )
                return
            try:
                self._transport.publish(frame)
            except Exception as error:
                published_at = self._clock()  # publish 调用返回点（最接近上线时刻）
                self._record_publish(
                    generation, _RECORD_STOP, q_sdk, mode_machine,
                    published_at, False, f"{type(error).__name__}: {error}",
                )
                self._finish_stop(
                    generation,
                    trigger,
                    outcome=(
                        f"停止未确认：停止帧发布失败（失链，仅报告尝试，"
                        f"不声称已物理急停）：{error}；不退权"
                    ),
                    confirmed=False,
                    final_fault=True,
                )
                return
            published_at = self._clock()  # publish 调用返回点（最接近上线时刻）
            last_publish_at = published_at
            self._record_publish(
                generation, _RECORD_STOP, q_sdk, mode_machine, published_at, True, None
            )

        # 3) 反馈确认（阈值未注入则无法确认，R11）
        threshold = self._stop_confirm_dq_threshold
        if threshold is None:
            self._finish_stop(
                generation,
                trigger,
                outcome=(
                    "停止未确认：未注入停止确认速度阈值（stop_confirm_dq_threshold），"
                    "仅完成有限停止序列发布，无法以反馈确认停止；不退权、不声称已停"
                ),
                confirmed=False,
                final_fault=True,
            )
            return
        assert last_publish_at is not None
        if not self._wait_stop_confirmed(generation, threshold, last_publish_at):
            self._finish_stop(
                generation,
                trigger,
                outcome=(
                    "停止未确认：确认窗口（state_timeout_s）内未观察到发布后的"
                    "新鲜低速反馈；不退权、不声称已停"
                ),
                confirmed=False,
                final_fault=True,
            )
            return

        # 4) 确认后退权；急停/故障升级在 _finish_stop 终态提交的同一锁内
        #    合并（R3，评审 2026-11-18）：release_control 可阻塞，期间登记
        #    的急停不得因"退权前已缓存的 terminal_fault 快照"而丢失——旧
        #    实现按预读快照收尾 READ_ONLY 并清除锁存，原对象可再次激活，
        #    破坏急停的不可恢复语义。
        release_error: Exception | None = None
        try:
            self._transport.release_control()
        except Exception as error:
            release_error = error
        else:
            with self._lock:
                self._control_serial = None  # 退权完成：控制权归属清除
        if release_error is not None:
            self._finish_stop(
                generation,
                trigger,
                outcome=f"停止已确认但释放控制权失败：{release_error}",
                confirmed=True,
                final_fault=True,
            )
            return
        self._finish_stop(
            generation, trigger, outcome=None, confirmed=True, final_fault=final_fault
        )

    def _wait_deadline(self, generation: int, deadline: float) -> bool:
        """等待到注入时钟的绝对期限；世代/生命周期失效返回 False。"""
        with self._cond:
            while True:
                if self._generation != generation or self._phase != "STOPPING":
                    return False
                gap = deadline - self._clock()
                if gap <= 0.0:
                    return True
                self._cond.wait(min(gap, _SCHEDULER_POLL_S))

    def _wait_stop_confirmed(
        self, generation: int, threshold: float, after_publish_at: float
    ) -> bool:
        """等待反馈确认：发布后到达的新鲜帧目标臂速度 ≤ 阈值。

        确认窗口 = 硬件配置 state_timeout_s（注入时钟口径）。
        """
        try:
            mapping = self._require_mapping()
        except R1DStateError:
            return False  # 理论不可达（build_stop 已要求映射可用），防御处理
        buffer = self._require_buffer()
        assert self._state_timeout_s is not None  # buffer 存在即已配置
        deadline = self._clock() + self._state_timeout_s
        with self._cond:
            while True:
                if self._generation != generation or self._phase != "STOPPING":
                    return False
                now = self._clock()
                if now > deadline:
                    return False
                try:
                    frame = buffer.latest(now=now, session_id=generation)
                except R1DStateError:
                    frame = None
                if frame is not None and frame.received_at > after_publish_at:
                    dq = np.abs(mapping.extract_velocities(frame))
                    if float(np.max(dq)) <= threshold:
                        return True
                gap = deadline - self._clock()
                if gap <= 0.0:
                    return False
                self._cond.wait(min(gap, _SCHEDULER_POLL_S))

    def _finish_stop(
        self,
        generation: int,
        trigger: str,
        *,
        outcome: str | None,
        confirmed: bool,
        final_fault: bool,
    ) -> None:
        """停止流程收尾：状态机落位 + 完成事件 + 日志（发送线程调用）。

        P1-2 修复（设计 §5 / §12.4）：终态在同一把锁内合并急停锁存与回调侧
        异步故障（退权窗口登记的控制权丢失）；stale 调用只记日志，不改写
        新会话的任何字段。日志按锁内快照在锁外记录，避免"FAULT 状态配
        READ_ONLY 成功日志"以及持锁调用日志 handler。
        """
        with self._lock:
            stale = self._generation != generation
            snapshot_faulted = False
            snapshot_reason: str | None = None
            snapshot_confirmed = confirmed
            if not stale:
                # R3（评审 2026-11-18）/ P1-2：终态提交的同一锁内重新合并
                # 急停锁存与异步故障。退权开始前/执行中/返回后但终态尚未
                # 提交——这些窗口内登记的原因都不得因竞争而丢失（退权阻塞
                # 期间的控制权丢失若只读急停锁存会被丢弃并回落 READ_ONLY）。
                escalated = self._emergency_latch
                async_reason = self._async_fault_reason
                snapshot_faulted = (
                    final_fault
                    or outcome is not None
                    or escalated
                    or async_reason is not None
                )
                if snapshot_faulted:
                    # reason 保留全部来源；同一段文字只拼接一次（trigger 可能
                    # 本身就是异步故障文案：_decide 消费事件后作为 trigger）
                    reasons = [trigger]
                    if outcome is not None and outcome not in reasons:
                        reasons.append(outcome)
                    if async_reason is not None and async_reason not in reasons:
                        reasons.append(async_reason)
                    if escalated:
                        escalation = "退权窗口收到急停升级（终态强制 FAULT）"
                        if escalation not in reasons:
                            reasons.append(escalation)
                    snapshot_reason = "；".join(reasons)
                    self._phase = "FAULT"
                    self._fault_reason = snapshot_reason
                else:
                    self._phase = "READ_ONLY"
                    self._fault_reason = None
                # 停止确认与故障终态是两个正交事实：两条终态分支都保留确认结果
                self._stop_confirmed = snapshot_confirmed
                self._pending_target = None
                self._pending_accepted_at = None
                self._activation_mode_machine = None
                self._emergency_latch = False
                self._async_fault_reason = None  # 已并入终态，清除防泄漏
                self._cond.notify_all()
        self._stop_done.set()
        # 锁外按锁内快照记录：不持锁调用日志（handler 可能阻塞发送线程），
        # 也不回读可能已被新会话改写的字段
        if stale:
            logger.info("停止收尾跳过：世代已失效（%s）", trigger)
        elif snapshot_faulted:
            logger.error(
                "停止结束：FAULT（stop_confirmed=%s）：%s",
                snapshot_confirmed,
                snapshot_reason,
            )
        else:
            logger.info("停止结束：READ_ONLY（停止已反馈确认并退权）")

    # ------------------------------------------------------------------
    # 内部：记录与组件解析
    # ------------------------------------------------------------------

    def _record_publish(
        self,
        session_id: int,
        kind: str,
        q_sdk: np.ndarray | None,
        mode_machine: int,
        published_at: float,
        success: bool,
        error: str | None,
    ) -> None:
        """记录一次实际发布（R9：publish 调用点时刻）。"""
        record = R1DPublishRecord(
            session_id=session_id,
            kind=kind,
            published_at=published_at,
            q_sdk=q_sdk,
            mode_machine=mode_machine,
            success=success,
            error=error,
        )
        with self._lock:
            # 记录本身如实保留（含死会话滞留发布的诚实留痕）；
            # 状态字段只反映当前会话
            self._publish_records.append(record)
            if success and self._generation == session_id:
                self._last_publish_at = published_at

    def _target_q_sdk(self, frame: R1DCommandFrame) -> np.ndarray | None:
        """提取命令帧目标臂 SDK 侧目标（模型序，只读副本）；失败返回 None。"""
        try:
            mapping = self._require_mapping()
            idx = np.asarray(mapping.sdk_indices, dtype=np.intp)
            q = np.array(frame.q[idx], dtype=np.float64, copy=True)
            q.setflags(write=False)
            return q
        except Exception:
            return None

    def _require_buffer(self) -> R1DStateBuffer:
        """取状态缓存；未创建（状态超时缺配置）抛显式错误。"""
        buffer = self._buffer
        if buffer is None:
            raise R1DStateError(
                "状态缓存不可用：timing.state_timeout_s 未配置（能力禁用，见 capability_notes）"
            )
        return buffer

    def _require_mapping(self) -> R1DJointMapping:
        """解析模型序映射（延迟构造一次）；校准 null 抛显式错误。"""
        with self._lock:
            mapping = self._mapping
        if mapping is not None:
            return mapping
        try:
            mapping = R1DJointMapping.from_profile(self._profile, self._config)
        except ValueError as error:
            raise R1DStateError(
                f"无法构造七关节映射（校准/布局不可用），不能提供模型序状态: {error}"
            ) from error
        with self._lock:
            if self._mapping is None:
                self._mapping = mapping
            return self._mapping
