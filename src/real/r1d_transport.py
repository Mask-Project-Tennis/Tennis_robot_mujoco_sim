"""R1-A7-D 实际 SDK 传输适配 — SdkR1DTransport（阶段 4 Task 4.5）。

实现 `src/real/r1d_types.py::R1DTransport` Protocol：
`open`（只订阅 rt/lowstate，CRC 校验后交付 R1DStateFrame）→
`acquire_control`（MotionSwitcher CheckMode→ReleaseMode→重查循环，空闲后才建
rt/lowcmd publisher）→ `publish`（显式全字段构造 hg LowCmd_ + 官方 CRC）→
`release_control`（关闭 publisher，不触发未经验证的机载服务恢复 RPC）→
`close`（幂等，会话世代门阻断 close 后旧回调）。

协议事实来源：configs/robots/r1d_protocol_manifest.yaml（Task 4.1 冻结）。

模块纯度（施工计划 Global Constraints）：
- 本模块 import 零副作用：不导入 unitree_sdk2py/cyclonedds、不初始化
  ChannelFactory、不启动线程（tests/test_r1d_transport.py 子进程锁死）；
- SDK 延迟到真实构造（`bundle=None`）时才 import；无 SDK 机器构造报
  R1DDependencyError 明确依赖错误；
- cyclonedds 11.1.0 运行时下生成 IDL 类要求显式全字段构造（无默认值，
  manifest sdk.local_dev_install_note），因此所有 IDL 构造走显式工厂；
- release_control 不实现任何"恢复原模式"RPC——manifest control.release.mechanism
  为 null（官方示例无退出释放路径），本项目不得照抄或臆造未经验证的 fallback。

并发与拆卸（P1-1）：
- 两把可重入锁：`_lifecycle_lock`（open/release/close 全程串行，覆盖 SDK
  资源创建与 teardown）与 `_io_lock`（publish 的 状态检查→组包→CRC→Write
  临界区）；固定锁序 lifecycle → io，publish 只取 io；
- 控制权获取用单调递增 `_acquire_serial` 票据：release/close 使旧票据失效，
  提交点校验 会话+票据+opened+未持有+publisher 空；陈旧者锁外 Close 自建
  publisher 并抛 R1DControlError（并发语义 = 最新一次 acquire 胜出）；
- close 的 SDK Close 在 io 锁外执行（真实 SDK Close 可能等待内部回调线程，
  而该线程可能又需要传输层锁，持 io 调用会死锁），并按入口耗时告警
  （>50ms 仅提示，不引入超时语义）。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.real.r1d_config import R1DHardwareConfig
from src.real.r1d_types import (
    R1D_SLOT_COUNT,
    R1DCommandFrame,
    R1DStateFrame,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 协议冻结常量（manifest dds / control 节）
# ---------------------------------------------------------------------------

# DDS domain（manifest dds.domain=0，双侧示例固定值）
R1D_DDS_DOMAIN_DEFAULT = 0
# 话题名（manifest dds.topics，协议冻结）
R1D_LOWCMD_TOPIC = "rt/lowcmd"
R1D_LOWSTATE_TOPIC = "rt/lowstate"
# MotionSwitcher RPC 单次超时（manifest control.acquisition.timeout_example_s=5.0，
# 官方示例事实，非厂商承诺）
R1D_SWITCHER_TIMEOUT_S_EXAMPLE = 5.0
# ReleaseMode 后重查间隔（manifest control.acquisition：官方示例 sleep 1-5s）
R1D_RELEASE_RETRY_INTERVAL_S_EXAMPLE = 1.0
# 订阅队列深度（官方 A7 示例 ChannelSubscriber.Init(handler, 10)）
R1D_STATE_QUEUE_LEN_EXAMPLE = 10
# close 拆卸耗时告警阈值（秒）：仅 warning，不引入超时/中断语义（P1-1）
_CLOSE_SLOW_WARN_S = 0.05


# ---------------------------------------------------------------------------
# 异常体系（SDK 异常在本层转换为明确项目异常，不吞、不伪造成功）
# ---------------------------------------------------------------------------


class R1DTransportError(RuntimeError):
    """R1D 传输层错误基类。"""


class R1DDependencyError(R1DTransportError):
    """SDK 依赖缺失（unitree_sdk2py/cyclonedds 不可用）。"""


class R1DChannelError(R1DTransportError):
    """DDS 通道错误（ChannelFactory 初始化/网卡/订阅/发布通道创建失败）。"""


class R1DControlError(R1DTransportError):
    """控制权生命周期错误（CheckMode/ReleaseMode 失败或获取超时）。"""


class R1DPublishError(R1DTransportError):
    """命令发布失败（Write 返回失败或抛出异常）。"""


class R1DTransportUsageError(R1DTransportError):
    """使用顺序错误（未 open 先 acquire、未 acquire 先 publish、重复 open 等）。"""


# ---------------------------------------------------------------------------
# SDK 绑定包（真实懒加载 / 测试注入共用一条代码路径）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SdkBundle:
    """unitree_sdk2py 表面的最小绑定集（真实加载或测试注入）。

    Attributes:
        channel_factory_initialize: ChannelFactoryInitialize(domain, iface)。
        lowcmd_type: hg LowCmd_ 类（显式全字段构造）。
        lowstate_type: hg LowState_ 类。
        motor_cmd_factory: MotorCmd_ 显式工厂（mode/q/dq/tau/kp/kd/reserve）。
        crc_compute: 官方 CRC(msg) -> int（发布前计算与接收校验共用）。
        create_publisher: (topic, msg_type) -> ChannelPublisher-like。
        create_subscriber: (topic, msg_type) -> ChannelSubscriber-like。
        create_switcher: () -> MotionSwitcherClient-like。
    """

    channel_factory_initialize: Callable[..., Any]
    lowcmd_type: Any
    lowstate_type: Any
    motor_cmd_factory: Callable[..., Any]
    crc_compute: Callable[[Any], int]
    create_publisher: Callable[[str, Any], Any]
    create_subscriber: Callable[[str, Any], Any]
    create_switcher: Callable[[], Any]


def _load_sdk_bundle() -> _SdkBundle:
    """懒加载真实 unitree_sdk2py 并组装 _SdkBundle。

    仅在 SdkR1DTransport 真实构造（bundle=None）时被调用；模块 import 不触达。

    Returns:
        绑定真实 SDK 的 _SdkBundle。

    Raises:
        R1DDependencyError: SDK 未安装或导入失败（消息含安装指引）。
    """
    try:
        # 惰性导入：ChannelFactory 初始化函数、通道类、MotionSwitcher、CRC、hg IDL
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, MotorCmd_
        from unitree_sdk2py.utils.crc import CRC
    except Exception as exc:  # ImportError 及 SDK 环境破损均属依赖问题
        raise R1DDependencyError(
            "R1D 传输层依赖 unitree_sdk2py 不可用（未安装或导入失败）。"
            "请按 configs/robots/r1d_protocol_manifest.yaml sdk.install_commands "
            "安装（固定 commit 814556d1；本机离线测试环境另见 "
            "sdk.local_dev_install_note）。原始错误: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    def _motor_cmd_factory(
        *,
        mode: int,
        q: float,
        dq: float,
        tau: float,
        kp: float,
        kd: float,
        reserve: int = 0,
    ) -> Any:
        """MotorCmd_ 显式全字段工厂（cyclonedds 11.1.0 无默认值）。"""
        return MotorCmd_(mode=mode, q=q, dq=dq, tau=tau, kp=kp, kd=kd, reserve=reserve)

    def _crc_compute(msg: Any) -> int:
        """官方 CRC 计算包装（CRC 为 Singleton，重复构造安全）。"""
        return int(CRC().Crc(msg))

    return _SdkBundle(
        channel_factory_initialize=ChannelFactoryInitialize,
        lowcmd_type=LowCmd_,
        lowstate_type=LowState_,
        motor_cmd_factory=_motor_cmd_factory,
        crc_compute=_crc_compute,
        create_publisher=lambda topic, msg_type: ChannelPublisher(topic, msg_type),
        create_subscriber=lambda topic, msg_type: ChannelSubscriber(topic, msg_type),
        create_switcher=MotionSwitcherClient,
    )


# ---------------------------------------------------------------------------
# 消息边界转换（帧 <-> IDL，边界复制，逐槽显式转录）
# ---------------------------------------------------------------------------


def _build_lowcmd(command: R1DCommandFrame, bundle: _SdkBundle) -> Any:
    """把 R1DCommandFrame 显式转录为 hg LowCmd_（35 槽逐字段，CRC 置 0 待算）。

    适配器层不补任何默认语义值：mode_pr/mode_machine 来自命令帧
    （取值策略归 4.3），保留字段按 manifest lowcmd_layout"示例写 0"。

    Args:
        command: 完整 35 槽命令帧。
        bundle: SDK 绑定（提供显式工厂与 LowCmd_ 类）。

    Returns:
        已填充全部字段、crc=0 的 LowCmd_（调用方随后写入官方 CRC）。
    """
    motors = [
        bundle.motor_cmd_factory(
            mode=int(command.motor_mode[i]),
            q=float(command.q[i]),
            dq=float(command.dq[i]),
            tau=float(command.tau[i]),
            kp=float(command.kp[i]),
            kd=float(command.kd[i]),
            reserve=0,
        )
        for i in range(R1D_SLOT_COUNT)
    ]
    return bundle.lowcmd_type(
        mode_pr=int(command.mode_pr),
        mode_machine=int(command.mode_machine),
        motor_cmd=motors,
        reserve=[0, 0, 0, 0],
        crc=0,
    )


def _state_frame_from_lowstate(
    msg: Any, *, session_id: int, received_at: float
) -> R1DStateFrame:
    """把 hg LowState_ 转换为 R1DStateFrame（边界复制，tick→sequence）。

    LowState_ 无源时间戳字段，source_time=None（不得编造）；
    q/dq 按 35 槽逐槽转录为只读 float64 快照。

    Args:
        msg: SDK 侧 LowState_ 消息。
        session_id: 当前驱动会话标识（会话世代门内传入）。
        received_at: 接收端单调时钟秒（由调用方注入）。

    Returns:
        R1DStateFrame 快照。

    Raises:
        R1DTransportError: motor_state 缺失或槽数不是 35。
        ValueError: 数组含 NaN/Inf（R1DStateFrame 有限性校验）。
    """
    states = getattr(msg, "motor_state", None)
    if states is None or len(states) != R1D_SLOT_COUNT:
        raise R1DTransportError(
            f"LowState_.motor_state 槽数异常: {None if states is None else len(states)}"
            f"（期望 {R1D_SLOT_COUNT}）"
        )
    q = np.fromiter(
        (float(state.q) for state in states), dtype=np.float64, count=R1D_SLOT_COUNT
    )
    dq = np.fromiter(
        (float(state.dq) for state in states), dtype=np.float64, count=R1D_SLOT_COUNT
    )
    return R1DStateFrame(
        q=q,
        dq=dq,
        sequence=int(msg.tick),
        source_time=None,
        received_at=received_at,
        mode_machine=int(msg.mode_machine),
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# 参数校验辅助
# ---------------------------------------------------------------------------


def _require_positive_finite(value: float, name: str) -> float:
    """校验正有限浮点参数。"""
    number = float(value)
    if not bool(np.isfinite(number)) or number <= 0.0:
        raise ValueError(f"参数 {name} 必须为正有限值，实际 {value}")
    return number


def _require_non_negative_finite(value: float, name: str) -> float:
    """校验非负有限浮点参数。"""
    number = float(value)
    if not bool(np.isfinite(number)) or number < 0.0:
        raise ValueError(f"参数 {name} 必须为非负有限值，实际 {value}")
    return number


def _require_topic(value: str, name: str) -> str:
    """校验非空话题名。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"参数 {name} 必须是非空字符串")
    return value


# ---------------------------------------------------------------------------
# SdkR1DTransport
# ---------------------------------------------------------------------------


class SdkR1DTransport:
    """基于 unitree_sdk2py 的 R1DTransport 实现（真实 DDS 通道）。

    生命周期语义（对齐 Protocol 约束与 manifest control 节）：
    - `open(on_state)`：初始化 ChannelFactory（domain+网卡）并只订阅
      rt/lowstate；不建 publisher、不触发任何控制权 RPC；
    - `acquire_control()`：MotionSwitcherClient 循环 CheckMode→（占用则
      ReleaseMode→重查）→ 空闲后才创建 rt/lowcmd publisher；任一返回码非零、
      结果结构异常或超时均抛 R1DControlError 且不发布任何命令；
    - `publish(command)`：显式全字段构造 LowCmd_ + 官方 CRC 后 Write；
      Write 返回 False 或抛出异常均转换 R1DPublishError；
    - `release_control()`：关闭 publisher 即停止发布；不实现任何未经验证的
      机载服务恢复 RPC（manifest control.release.mechanism=null）；
    - `close()`：幂等；关闭订阅/发布通道并使旧会话回调失效（世代门）。
      close 后旧回调（在途/迟到帧）原则上不再投递；极小竞态窗口内
      （回调已通过世代检查、close 尚未完成）单帧可能迟到一次，由
      4.6 驱动侧会话世代门二次拦截（A-M3 措辞如实化，评审 2026-11-17）。

    Note:
        `switcher_timeout_s`/`release_retry_interval_s` 默认值取自 manifest
        记录的官方示例事实；`acquire_timeout_s` 为本项目自定工程参数
        （非厂商值），均可由调用方（4.6/4.7 层）按现场冻结配置显式注入。
    """

    def __init__(
        self,
        *,
        dds_domain: int = R1D_DDS_DOMAIN_DEFAULT,
        network_interface: str | None = None,
        lowcmd_topic: str = R1D_LOWCMD_TOPIC,
        lowstate_topic: str = R1D_LOWSTATE_TOPIC,
        switcher_timeout_s: float = R1D_SWITCHER_TIMEOUT_S_EXAMPLE,
        acquire_timeout_s: float = 10.0,
        release_retry_interval_s: float = R1D_RELEASE_RETRY_INTERVAL_S_EXAMPLE,
        state_queue_len: int = R1D_STATE_QUEUE_LEN_EXAMPLE,
        clock: Callable[[], float] = time.monotonic,
        bundle: _SdkBundle | None = None,
    ) -> None:
        """构造传输（真实构造触发 SDK 懒加载；注入 bundle 则完全不触 SDK）。

        Args:
            dds_domain: DDS domain（manifest 冻结 0）。
            network_interface: 网卡名；None 交由 SDK 自动选择（真实网卡名
                现场确认后注入，manifest dds.network_interface=null 待确认）。
            lowcmd_topic: 命令话题（协议冻结 rt/lowcmd）。
            lowstate_topic: 状态话题（协议冻结 rt/lowstate）。
            switcher_timeout_s: MotionSwitcher RPC 单次超时（示例事实 5.0）。
            acquire_timeout_s: 获取控制权整体超时（本项目自定工程参数）。
            release_retry_interval_s: ReleaseMode 后重查间隔（示例 1.0）。
            state_queue_len: 订阅回调队列深度（官方示例 10）。
            clock: 单调时钟注入（默认 time.monotonic）。
            bundle: SDK 绑定注入（测试用 fake）；None 时懒加载真实 SDK。

        Raises:
            ValueError: 参数非法（负 domain、空网卡字符串、非正超时等）。
            R1DDependencyError: bundle=None 且本机缺少 unitree_sdk2py。
        """
        if (
            isinstance(dds_domain, bool)
            or not isinstance(dds_domain, int)
            or dds_domain < 0
        ):
            raise ValueError(f"参数 dds_domain 必须为非负整数，实际 {dds_domain}")
        if network_interface is not None and (
            not isinstance(network_interface, str) or not network_interface.strip()
        ):
            raise ValueError("参数 network_interface 必须是非空字符串或 None")
        self._dds_domain = dds_domain
        self._network_interface = network_interface
        self._lowcmd_topic = _require_topic(lowcmd_topic, "lowcmd_topic")
        self._lowstate_topic = _require_topic(lowstate_topic, "lowstate_topic")
        self._switcher_timeout_s = _require_positive_finite(
            switcher_timeout_s, "switcher_timeout_s"
        )
        self._acquire_timeout_s = _require_positive_finite(
            acquire_timeout_s, "acquire_timeout_s"
        )
        self._release_retry_interval_s = _require_non_negative_finite(
            release_retry_interval_s, "release_retry_interval_s"
        )
        if not isinstance(state_queue_len, int) or state_queue_len <= 0:
            raise ValueError(
                f"参数 state_queue_len 必须为正整数，实际 {state_queue_len}"
            )
        self._state_queue_len = state_queue_len
        self._clock = clock
        # 真实构造才触发 SDK import（无 SDK 机器此处报 R1DDependencyError）
        self._bundle = _load_sdk_bundle() if bundle is None else bundle

        # 会话世代门：每次 open 自增；close 后旧 handler 因世代不匹配被丢弃
        self._session_serial = 0
        # 获取控制权票据：每次 acquire 尝试自增；release/close 使旧票据失效
        self._acquire_serial = 0
        self._on_state: Callable[[R1DStateFrame], None] | None = None
        self._opened = False
        self._acquired = False
        self._subscriber: Any = None
        self._publisher: Any = None
        # 锁序固定 lifecycle → io；publish 只取 io；io 内禁止取 lifecycle、
        # 禁止调用用户回调、禁止调用 SDK Close（P1-1 死锁边界）。
        self._lifecycle_lock = threading.RLock()
        self._io_lock = threading.RLock()

    @classmethod
    def from_hardware_config(
        cls,
        config: R1DHardwareConfig,
        *,
        switcher_timeout_s: float = R1D_SWITCHER_TIMEOUT_S_EXAMPLE,
        acquire_timeout_s: float = 10.0,
        release_retry_interval_s: float = R1D_RELEASE_RETRY_INTERVAL_S_EXAMPLE,
        state_queue_len: int = R1D_STATE_QUEUE_LEN_EXAMPLE,
        clock: Callable[[], float] = time.monotonic,
        bundle: _SdkBundle | None = None,
    ) -> SdkR1DTransport:
        """从硬件配置构建传输（通信参数自 config.communication 注入）。

        Args:
            config: R1D 硬件配置（网卡名等现场参数以配置为准）。
            switcher_timeout_s: MotionSwitcher RPC 单次超时。
            acquire_timeout_s: 获取控制权整体超时。
            release_retry_interval_s: ReleaseMode 后重查间隔。
            state_queue_len: 订阅回调队列深度（官方示例 10；现场冻结
                值不同时显式注入，A-M1 评审 2026-11-17）。
            clock: 单调时钟注入。
            bundle: SDK 绑定注入（测试）；None 懒加载真实 SDK。

        Returns:
            SdkR1DTransport 实例。
        """
        return cls(
            dds_domain=config.communication.dds_domain,
            network_interface=config.communication.network_interface,
            lowcmd_topic=config.communication.lowcmd_topic,
            lowstate_topic=config.communication.lowstate_topic,
            switcher_timeout_s=switcher_timeout_s,
            acquire_timeout_s=acquire_timeout_s,
            release_retry_interval_s=release_retry_interval_s,
            state_queue_len=state_queue_len,
            clock=clock,
            bundle=bundle,
        )

    # -- open ---------------------------------------------------------------

    def open(self, on_state: Callable[[R1DStateFrame], None]) -> None:
        """打开通道并注册状态回调（只订阅，不接管、不发布）。

        初始化 ChannelFactory（domain+网卡）后创建 rt/lowstate 订阅；
        CRC 校验失败帧与异常帧均丢弃，不向回调投递。

        Args:
            on_state: 状态回调（接收 CRC 已校验的 R1DStateFrame）。

        Raises:
            ValueError: on_state 不可调用。
            R1DTransportUsageError: 已 open 未 close。
            R1DChannelError: ChannelFactory 初始化失败（网卡/domain 错误）
                或订阅创建失败（此时清理已建资源，不留半初始化）。
        """
        if not callable(on_state):
            raise ValueError("参数 on_state 必须是可调用对象")

        # open 全程持 lifecycle：与 close/release 串行，close 的 SDK teardown
        # 未完成前不得创建新订阅（P1-1：open 必须等上一次 close 拆完）。
        with self._lifecycle_lock:
            with self._io_lock:
                if self._opened:
                    raise R1DTransportUsageError("通道已 open，重复 open 须先 close")

            # 1) ChannelFactory 初始化（网卡/domain 错误在此显式暴露）。
            #    A-M2（评审 2026-11-17）：SDK ChannelFactoryInitialize 失败路径
            #    自抛 Exception、成功恒返 None（源码 channel.py:298 无 return），
            #    原 `if result is False` 分支为死代码，已删除——失败一律经
            #    except 转换为 R1DChannelError
            try:
                self._bundle.channel_factory_initialize(
                    self._dds_domain, self._network_interface
                )
            except Exception as exc:
                raise R1DChannelError(
                    f"ChannelFactory 初始化异常（domain={self._dds_domain}, "
                    f"network_interface={self._network_interface!r}，请核对网卡名）: {exc}"
                ) from exc

            # 2) 创建订阅；失败时回滚会话状态并清理已建资源
            subscriber: Any = None
            try:
                subscriber = self._bundle.create_subscriber(
                    self._lowstate_topic, self._bundle.lowstate_type
                )
                # 先登记会话与回调再 Init：Init 成功即可能开始投递
                with self._io_lock:
                    self._session_serial += 1
                    session = self._session_serial
                    self._on_state = on_state
                    self._opened = True
                    self._subscriber = subscriber
                subscriber.Init(
                    self._make_state_handler(session), self._state_queue_len
                )
            except Exception as exc:
                with self._io_lock:
                    self._on_state = None
                    self._opened = False
                    self._subscriber = None
                if subscriber is not None:
                    try:
                        subscriber.Close()
                    except Exception as close_exc:  # 清理失败仅记录，不掩盖原始异常
                        logger.warning("订阅回滚 Close 失败: %s", close_exc)
                raise R1DChannelError(
                    f"打开 {self._lowstate_topic} 订阅失败: {exc}"
                ) from exc

    def _make_state_handler(self, session: int) -> Callable[[Any], None]:
        """构造绑定指定会话世代的状态回调（世代门 + CRC 校验 + 异常吞噬）。"""

        def handler(msg: Any) -> None:
            # 会话世代门：close/重开后旧世代回调直接丢弃，禁止写任何会话
            if not self._opened or session != self._session_serial:
                return
            try:
                crc_expected = self._bundle.crc_compute(msg)
                if int(getattr(msg, "crc", -1)) != crc_expected:
                    logger.warning("lowstate CRC 校验失败，丢弃该帧")
                    return
                frame = _state_frame_from_lowstate(
                    msg, session_id=session, received_at=self._clock()
                )
                callback = self._on_state
                if callback is None:  # 防御：世代门内正常不应发生
                    logger.error("状态回调缺失（会话 %s），丢弃该帧", session)
                    return
                callback(frame)
            except Exception:
                # 回调链路异常（含 NaN/Inf 状态帧被有限性校验拒绝）不得
                # 外泄进 SDK 监听线程；记录后丢帧。
                logger.exception("lowstate 回调处理异常，丢弃该帧")

        return handler

    # -- acquire / release ---------------------------------------------------

    def acquire_control(self) -> None:
        """获取整机控制权（MotionSwitcher 流程 + 建立 lowcmd publisher）。

        流程（manifest control.acquisition）：CheckMode 返回 name 非空说明
        机载运控占用 → ReleaseMode 释放 → 间隔重查，直到 name 为空才创建
        publisher。返回码非零、结果结构异常或整体超时抛 R1DControlError，
        且此时不创建 publisher、未发布任何命令（失败不发布要求）。

        Warning:
            A-M7（评审 2026-11-17）：若已执行 ReleaseMode（释放机载运控）
            之后 publisher 创建失败，机端停留在"已释放"状态——本项目
            **不实现恢复原模式 RPC**（manifest control.release.mechanism
            为 null，官方无已核验的恢复路径，不得臆造）。此时仅回滚
            publisher 资源并抛 R1DChannelError，机端模式恢复由现场
            人工/机载自身机制处理。

        Raises:
            R1DTransportUsageError: 未先 open 或已持有控制权。
            R1DControlError: MotionSwitcher 初始化/CheckMode/ReleaseMode
                失败、结果结构异常或获取超时。
            R1DChannelError: 控制权获取后创建 publisher 失败（清理回滚）。
        """
        # 入口快照（lifecycle→io 短临界区）：校验可获取并领取本次尝试票据。
        with self._lifecycle_lock:
            with self._io_lock:
                if not self._opened:
                    raise R1DTransportUsageError("须先 open() 再 acquire_control()")
                if self._acquired or self._publisher is not None:
                    raise R1DTransportUsageError("已持有控制权，重复 acquire_control()")
                session = self._session_serial
                self._acquire_serial += 1
                ticket = self._acquire_serial

        client = self._create_switcher_client()
        deadline = self._clock() + self._acquire_timeout_s
        while True:
            name = self._check_mode_free(client)
            if name is None:
                break  # 空闲（name 为空）
            if self._clock() + self._release_retry_interval_s > deadline:
                raise R1DControlError(
                    f"获取控制权超时（{self._acquire_timeout_s}s）：机载运控持续占用，"
                    "未创建 publisher、未发布任何命令"
                )
            self._release_mode(client)
            if self._release_retry_interval_s > 0.0:
                time.sleep(self._release_retry_interval_s)

        publisher: Any = None
        try:
            publisher = self._bundle.create_publisher(
                self._lowcmd_topic, self._bundle.lowcmd_type
            )
            publisher.Init()
        except Exception as exc:
            if publisher is not None:
                try:
                    publisher.Close()
                except Exception as close_exc:
                    logger.warning("publisher 回滚 Close 失败: %s", close_exc)
            raise R1DChannelError(
                f"创建 {self._lowcmd_topic} publisher 失败: {exc}"
            ) from exc

        # 提交点（lifecycle→io 短临界区）：RPC/建流期间会话或票据可能已失效
        with self._lifecycle_lock:
            with self._io_lock:
                stale = (
                    not self._opened
                    or session != self._session_serial
                    or ticket != self._acquire_serial
                    or self._acquired
                    or self._publisher is not None
                )
                if not stale:
                    self._publisher = publisher
                    self._acquired = True
        if not stale:
            return
        # 陈旧提交：锁外关闭自建 publisher，不得装入任何会话
        try:
            publisher.Close()
        except Exception as close_exc:
            logger.warning("陈旧 acquire 回滚 Close 失败: %s", close_exc)
        raise R1DControlError(
            "控制权提交点校验失败：会话/票据已失效（close/release 或更新一次 "
            "acquire 已发生），本次自建 publisher 已关闭、未发布任何命令"
        )

    def _create_switcher_client(self) -> Any:
        """创建并初始化 MotionSwitcherClient（异常转换为 R1DControlError）。"""
        try:
            client = self._bundle.create_switcher()
            client.SetTimeout(self._switcher_timeout_s)
            client.Init()
            return client
        except Exception as exc:
            raise R1DControlError(f"MotionSwitcher 客户端初始化失败: {exc}") from exc

    def _check_mode_free(self, client: Any) -> str | None:
        """执行一次 CheckMode；返回 None 表示空闲，非 None 为当前占用模式名。

        Raises:
            R1DControlError: 返回码非零或结果结构异常（返回码 0 不无条件当成功）。
        """
        try:
            code, result = client.CheckMode()
        except Exception as exc:
            raise R1DControlError(f"CheckMode 调用异常: {exc}") from exc
        if code != 0:
            raise R1DControlError(f"CheckMode 失败，返回码 {code}")
        if not isinstance(result, dict) or "name" not in result:
            raise R1DControlError(
                f"CheckMode 返回码 0 但结果结构异常（期望含 'name' 的映射）: {result!r}"
            )
        name = result["name"]
        if not isinstance(name, str):
            raise R1DControlError(f"CheckMode 返回的 name 非字符串: {name!r}")
        return name if name else None

    def _release_mode(self, client: Any) -> None:
        """执行一次 ReleaseMode（失败即抛，不发布任何命令）。

        Raises:
            R1DControlError: 调用异常或返回码非零。
        """
        try:
            code, _ = client.ReleaseMode()
        except Exception as exc:
            raise R1DControlError(f"ReleaseMode 调用异常: {exc}") from exc
        if code != 0:
            raise R1DControlError(
                f"ReleaseMode 失败，返回码 {code}（未获控制权，未发布任何命令）"
            )

    def release_control(self) -> None:
        """释放控制权：关闭 lowcmd publisher（即停止发布）。

        不实现任何"恢复原模式"RPC——manifest control.release.mechanism 为
        null（官方示例无退出释放路径，机载服务恢复机制未公开），本项目
        不得照抄示例或臆造未经验证的 fallback；停止效果未确认的语义由
        上层（4.6）负责报告。

        Raises:
            R1DControlError: publisher 关闭失败（此时停止发布状态未确认）。
        """
        # release 全程持 lifecycle：SDK Close 未完成前，open/close/另一次
        # acquire 的提交都不得穿插（P1-1）；票据失效在 io 内完成。
        with self._lifecycle_lock:
            with self._io_lock:
                self._acquire_serial += 1  # 使在途 acquire 的票据失效
                publisher, self._publisher, self._acquired = (
                    self._publisher,
                    None,
                    False,
                )
            if publisher is None:
                return  # 未持有控制权，幂等 no-op
            try:
                publisher.Close()
            except Exception as exc:
                raise R1DControlError(
                    f"关闭 {self._lowcmd_topic} publisher 失败（停止发布状态未确认）: {exc}"
                ) from exc

    # -- publish --------------------------------------------------------------

    def publish(self, command: R1DCommandFrame) -> None:
        """发布一帧完整整机命令（显式全字段 LowCmd_ + 官方 CRC + Write）。

        Args:
            command: 完整 35 槽命令帧（mode_machine 等由 4.3 策略填充）。

        Raises:
            ValueError: command 不是 R1DCommandFrame。
            R1DTransportUsageError: 未 acquire_control（或已 close），
                或会话已更换（close/open，旧会话发布已失效）。
            R1DPublishError: Write 返回失败或抛出异常。
        """
        if not isinstance(command, R1DCommandFrame):
            raise ValueError(
                f"publish 只接受 R1DCommandFrame，实际 {type(command).__name__}"
            )
        # 会话世代快照：从方法入口到取得 io 之间可能被 close→open→acquire 切走，
        # 必须在取 io 前记录“调用时所处会话”，io 内复查（fix1）。否则旧会话线程
        # 取得 io 后会看到新会话的 opened/acquired/publisher 全部成立而越界写入。
        session = self._session_serial
        # publish 只取 io：状态检查→组包→CRC→Write 属同一临界区，close 的资源
        # 交换必须等本拍发布结束；Write 前快照 publisher，之后不得重读成员
        # （P1-1：防止旧会话在途发布写进新会话 publisher）。
        with self._io_lock:
            if (
                not self._opened
                or not self._acquired
                or self._publisher is None
                or session != self._session_serial
            ):
                raise R1DTransportUsageError(
                    "未持有控制权，须先 acquire_control()（open 只读不发布）；"
                    "或会话已更换（close/open），旧会话发布已失效"
                )
            publisher = self._publisher
            try:
                msg = _build_lowcmd(command, self._bundle)
                msg.crc = self._bundle.crc_compute(msg)
                ok = publisher.Write(msg)
            except R1DTransportError:
                raise
            except Exception as exc:
                raise R1DPublishError(f"lowcmd 发布异常: {exc}") from exc
        if not ok:
            raise R1DPublishError(
                "lowcmd Write 返回失败（DDS 写入未成功，命令未确认送达）"
            )

    # -- close ----------------------------------------------------------------

    def close(self) -> None:
        """关闭通道（幂等；关闭订阅与发布通道并使旧会话回调失效）。

        并发语义（P1-1）：全程持 lifecycle，等所有在途 publish 结束（io）后
        再交换资源；两个 SDK Close 在 io 锁外执行（真实 Close 可能等待内部
        回调线程，持 io 调用会死锁）。入口耗时超过 50ms 仅记录 warning，
        不引入超时/中断语义（调用方仍须自行保证停止路径可接受更长等待）。

        Raises:
            R1DChannelError: 任一通道关闭失败（仍会尽量完成全部清理）。
        """
        started_at = self._clock()
        with self._lifecycle_lock:
            with self._io_lock:
                if (
                    not self._opened
                    and self._subscriber is None
                    and self._publisher is None
                ):
                    self._on_state = None
                    return  # 已处关闭态，幂等
                # 世代门立即生效：此后旧回调（在途/迟到帧）不再投递；
                # 票据递增使在途 acquire 的提交点失效。
                self._opened = False
                self._on_state = None
                self._acquire_serial += 1
                publisher, self._publisher, self._acquired = (
                    self._publisher,
                    None,
                    False,
                )
                subscriber, self._subscriber = self._subscriber, None
            errors: list[str] = []
            # SDK Close 在 io 锁外（lifecycle 内）执行：真实 Close 可能等待
            # 需要传输层锁的内部回调线程，持 io 调用会死锁。
            if publisher is not None:
                try:
                    publisher.Close()
                except Exception as exc:
                    errors.append(f"publisher 关闭失败: {exc}")
            if subscriber is not None:
                try:
                    subscriber.Close()
                except Exception as exc:
                    errors.append(f"subscriber 关闭失败: {exc}")
            elapsed = self._clock() - started_at
            if elapsed > _CLOSE_SLOW_WARN_S:
                logger.warning(
                    "close 拆卸耗时 %.3fs 超过 %.0fms 告警阈值（仅告警，"
                    "不中断/不超时）：publisher=%s subscriber=%s",
                    elapsed,
                    _CLOSE_SLOW_WARN_S * 1000.0,
                    publisher is not None,
                    subscriber is not None,
                )
            if errors:
                raise R1DChannelError("close 清理失败: " + "; ".join(errors))
