"""R1-A7-D 整机协议数据类型 — 纯数据层（阶段 4 Task 4.2）。

定义整机 35 槽布局下的状态帧、命令帧、驱动状态快照与传输层 Protocol。
本模块不导入 SDK/MuJoCo、不启动线程、不做网络或控制权副作用；
CRC 与 DDS 序列化属 4.5 传输层，此处只做结构、长度与有限性校验。

布局与字段事实来源：configs/robots/r1d_protocol_manifest.yaml（Task 4.1 冻结）。
mode_pr 语义（hg LowCmd_ 首字段 uint8，官方 0-100 权重口径）4.2 只透传不解释，
取值策略归 4.3；本层仅按 uint8 物理范围校验。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

# 整机 35 电机槽位（协议 manifest whole_body.slot_count）
R1D_SLOT_COUNT = 35
# 语义别名：状态帧（LowState_.motor_state）与命令帧（LowCmd_.motor_cmd）同为 35 槽
R1D_STATE_SLOT_COUNT = R1D_SLOT_COUNT
R1D_COMMAND_SLOT_COUNT = R1D_SLOT_COUNT

# 驱动生命周期 phase 合法取值（4.6 状态机使用）
R1D_PHASES = ("CLOSED", "READ_ONLY", "ACTIVE", "STOPPING", "FAULT")

# uint8 物理范围（mode_machine/mode_pr 序列化层为 uint8）
_UINT8_MIN = 0
_UINT8_MAX = 255


def _copy_finite_vector(value: object, name: str, length: int) -> np.ndarray:
    """把输入转成一维 float64 只读数组：边界复制 + 长度 + 有限性校验。

    Args:
        value: 任意可转数组的输入（list/tuple/ndarray）。
        name: 字段名（用于错误消息）。
        length: 期望的一维长度。

    Returns:
        只读（writeable=False）的 float64 数组快照，防调用方修改。

    Raises:
        ValueError: 形状不符、含 NaN/Inf、dtype 为布尔或无法转为数值。
    """
    # A-M6（评审 2026-11-17）：布尔数组是有限"数值"，isfinite 校验会
    # 放行——q/dq/kp/kd/tau 均为物理量浮点数组，bool dtype 属输入错误。
    if isinstance(value, np.ndarray) and value.dtype == bool:
        raise ValueError(
            f"字段 {name} 必须是数值数组，不接受布尔 dtype（物理量语义）"
        )
    arr = np.array(value, dtype=np.float64, copy=True)  # 边界复制，防外部引用
    if arr.shape != (length,):
        raise ValueError(
            f"字段 {name} 必须是一维长度 {length} 数组，实际形状 {arr.shape}"
        )
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError(f"字段 {name} 含 NaN 或 Inf")
    arr.setflags(write=False)
    return arr


def _copy_motor_mode_vector(value: object, name: str, length: int) -> np.ndarray:
    """校验电机模式向量：整数值且在 uint8 范围 [0, 255]。

    协议 MotorCmd_.mode 为 uint8（manifest motor_cmd_layout：1=Enable
    0=Disable）；存储沿用 float64 只读快照以共享向量校验路径，
    协议整型转换属 4.5 序列化层。非整数（如 0.5）与越界（如 256）拒绝。

    Args:
        value: 任意可转数组的输入。
        name: 字段名（用于错误消息）。
        length: 期望的一维长度。

    Returns:
        只读 float64 数组快照（元素均为 uint8 范围内的整数值）。

    Raises:
        ValueError: 形状/有限性不符、含非整数元素或越界。
    """
    arr = _copy_finite_vector(value, name, length)
    rounded = np.round(arr)
    if not bool(np.all(arr == rounded)):
        raise ValueError(
            f"字段 {name} 必须是整数值（协议 uint8 语义），含非整数元素"
        )
    if not bool(np.all((rounded >= _UINT8_MIN) & (rounded <= _UINT8_MAX))):
        raise ValueError(
            f"字段 {name} 超出 uint8 范围 [{_UINT8_MIN}, {_UINT8_MAX}]"
        )
    return arr


def _require_uint8(value: object, name: str) -> int:
    """校验 uint8 语义整数字段。

    Args:
        value: 输入值。
        name: 字段名（用于错误消息）。

    Returns:
        范围 [0, 255] 内的 int。

    Raises:
        ValueError: 非整数（bool 视为非法）或越界。
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"字段 {name} 必须是整数（uint8 语义）")
    number = int(value)
    if not (_UINT8_MIN <= number <= _UINT8_MAX):
        raise ValueError(
            f"字段 {name} 超出 uint8 范围 [{_UINT8_MIN}, {_UINT8_MAX}]: {number}"
        )
    return number


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


def _require_non_negative_int(value: object, name: str) -> int:
    """校验非负整数字段（协议 uint32 序号语义；bool 视为非法）。

    A-M6（评审 2026-11-17）：sequence 语义为源侧 uint32 tick 计数，
    负值属协议域外——构造层与缓存层（uint32 上界）双重拒收。

    Args:
        value: 输入值。
        name: 字段名（用于错误消息）。

    Returns:
        非负 int 值。

    Raises:
        ValueError: 非整数或负数。
    """
    number = _require_plain_int(value, name)
    if number < 0:
        raise ValueError(f"字段 {name} 必须非负（uint32 序号语义）: {number}")
    return number


def _require_finite_float(value: object, name: str) -> float:
    """校验有限浮点字段。

    Args:
        value: 输入值。
        name: 字段名（用于错误消息）。

    Returns:
        有限 float 值。

    Raises:
        ValueError: 非数值或非有限（NaN/Inf）。
    """
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"字段 {name} 必须是数值")
    number = float(value)
    if not bool(np.isfinite(number)):
        raise ValueError(f"字段 {name} 必须是有限值")
    return number


def _require_optional_finite_float(value: object, name: str) -> float | None:
    """校验可选有限浮点字段（None 合法，表示无支持，不编造）。"""
    if value is None:
        return None
    return _require_finite_float(value, name)


@dataclass(frozen=True)
class R1DStateFrame:
    """整机状态帧快照（来自 rt/lowstate，CRC 已在传输层校验后交付）。

    数组为完整 35 槽整机布局；构造时边界复制并置只读，
    快照不可被回调或读者修改（4.4 状态缓存依赖此性质）。

    Attributes:
        q: 全机 35 槽关节位置（rad，SDK 侧符号/零位约定）。
        dq: 全机 35 槽关节速度（rad/s，SDK 侧约定）。
        sequence: 协议序号；无支持时为 None，不得编造。
        source_time: 源侧时间戳（秒）；无支持时为 None，不得编造。
        received_at: 接收端单调时钟秒（必填，用于新鲜度判断）。
        mode_machine: 机型/模式回读字段（uint8；具体枚举值协议未公开）。
        session_id: 本驱动会话标识（重启/重连后更换）。
    """

    q: np.ndarray
    dq: np.ndarray
    sequence: int | None
    source_time: float | None
    received_at: float
    mode_machine: int
    session_id: int

    def __post_init__(self) -> None:
        """校验并转为只读快照（frozen dataclass 经 object.__setattr__ 写入）。"""
        object.__setattr__(
            self, "q", _copy_finite_vector(self.q, "q", R1D_STATE_SLOT_COUNT)
        )
        object.__setattr__(
            self, "dq", _copy_finite_vector(self.dq, "dq", R1D_STATE_SLOT_COUNT)
        )
        if self.sequence is not None:
            object.__setattr__(
                self, "sequence", _require_non_negative_int(self.sequence, "sequence")
            )
        object.__setattr__(
            self,
            "source_time",
            _require_optional_finite_float(self.source_time, "source_time"),
        )
        object.__setattr__(
            self, "received_at", _require_finite_float(self.received_at, "received_at")
        )
        object.__setattr__(
            self,
            "mode_machine",
            _require_uint8(self.mode_machine, "mode_machine"),
        )
        object.__setattr__(
            self, "session_id", _require_plain_int(self.session_id, "session_id")
        )


@dataclass(frozen=True)
class R1DCommandFrame:
    """整机命令帧（发布到 rt/lowcmd 的完整 35 槽报文，CRC 属序列化层）。

    全部字段必填、无默认值——适配器层禁止补神秘默认值；
    未核验的非目标槽位策略由 4.3 CommandPolicy 显式构造。

    Attributes:
        q: 全机 35 槽目标位置（rad）。
        dq: 全机 35 槽目标速度（rad/s）。
        kp: 全机 35 槽位置增益（N·m/rad）。
        kd: 全机 35 槽速度增益（N·m·s/rad）。
        tau: 全机 35 槽前馈力矩（N·m）。
        motor_mode: 全机 35 槽电机模式（协议 uint8：整数值 0-255，
            官方注释 1=Enable 0=Disable；本层校验整数值与 uint8 范围，
            语义解释归 4.3）。
        mode_machine: 机型/模式标识（uint8；须回填自最近状态帧）。
        mode_pr: hg LowCmd_ 首字段（uint8；官方 0-100 权重口径，
            本层只透传不解释，取值策略归 4.3）。
    """

    q: np.ndarray
    dq: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray
    motor_mode: np.ndarray
    mode_machine: int
    mode_pr: int

    def __post_init__(self) -> None:
        """校验并转为只读快照（frozen dataclass 经 object.__setattr__ 写入）。"""
        for name in ("q", "dq", "kp", "kd", "tau"):
            object.__setattr__(
                self,
                name,
                _copy_finite_vector(getattr(self, name), name, R1D_COMMAND_SLOT_COUNT),
            )
        object.__setattr__(
            self,
            "motor_mode",
            _copy_motor_mode_vector(
                self.motor_mode, "motor_mode", R1D_COMMAND_SLOT_COUNT
            ),
        )
        object.__setattr__(
            self,
            "mode_machine",
            _require_uint8(self.mode_machine, "mode_machine"),
        )
        object.__setattr__(self, "mode_pr", _require_uint8(self.mode_pr, "mode_pr"))


@dataclass(frozen=True)
class R1DStatus:
    """驱动状态快照（4.6 生命周期状态机的只读观测）。

    Attributes:
        phase: CLOSED/READ_ONLY/ACTIVE/STOPPING/FAULT 之一。
        session_id: 当前会话标识。
        fault_reason: 故障原因描述；无故障为 None。
        stop_confirmed: 停止是否被反馈确认（未确认时为 False，
            不得以"停止发布"推定机器人已停）。
        last_publish_at: 最近一次实际发布时刻（单调秒）；从未发布为 None。
    """

    phase: str
    session_id: int
    fault_reason: str | None
    stop_confirmed: bool
    last_publish_at: float | None

    def __post_init__(self) -> None:
        """校验 phase 合法性与各字段类型。"""
        if not isinstance(self.phase, str) or self.phase not in R1D_PHASES:
            raise ValueError(
                f"字段 phase 必须是 {list(R1D_PHASES)} 之一，实际 {self.phase!r}"
            )
        object.__setattr__(
            self, "session_id", _require_plain_int(self.session_id, "session_id")
        )
        if self.fault_reason is not None:
            if not isinstance(self.fault_reason, str) or not self.fault_reason.strip():
                raise ValueError("字段 fault_reason 必须是非空字符串或 None")
        if not isinstance(self.stop_confirmed, bool):
            raise ValueError("字段 stop_confirmed 必须是布尔值")
        object.__setattr__(
            self,
            "last_publish_at",
            _require_optional_finite_float(self.last_publish_at, "last_publish_at"),
        )


@runtime_checkable
class R1DTransport(Protocol):
    """R1D 传输层协议（实现归 4.5，本片仅声明契约）。

    语义约束：
    - open 只订阅状态回调，不隐式接管控制权；
    - close 幂等，重复调用安全；
    - 具体获取/释放控制权的厂商 API 由 4.1 冻结 manifest 决定，
      不从本抽象方法名推断。
    """

    def open(self, on_state: Callable[[R1DStateFrame], None]) -> None:
        """打开通道并注册状态回调（只订阅，不接管）。"""
        ...

    def acquire_control(self) -> None:
        """获取整机控制权。"""
        ...

    def publish(self, command: R1DCommandFrame) -> None:
        """发布一帧完整整机命令。"""
        ...

    def release_control(self) -> None:
        """释放整机控制权。"""
        ...

    def close(self) -> None:
        """关闭通道（幂等）。"""
        ...
