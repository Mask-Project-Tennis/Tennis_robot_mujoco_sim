"""R1-A7-D 全机命令构造与非目标部件策略 — 纯策略层（阶段 4 Task 4.3）。

职责：
- 把目标臂七关节的有限位置目标（模型序 rad）转换为**完整 35 槽**整机命令帧
  （R1DCommandFrame），逐槽位显式决定 q/dq/kp/kd/tau/mode 与头字段；
- 显式管理全部非目标槽位的保持/忽略策略，禁止默认清零或复用旧命令。

纯策略层：不导入 SDK/MuJoCo、不启动线程、不做网络/控制权副作用；
CRC 与 DDS 序列化属 4.5 传输层。本类非线程安全，调用方（4.6）须在
单一发送线程中串行调用 build_active/build_stop。

非目标槽位策略来源（主代理 Ruling R5 初版 2026-11-16，经阶段 4 团队评审
I-05 于 2026-11-17 修订）：
    硬件配置显式策略 > 拒绝构造（manifest 示例不再作为运行时回退）

R5 初版曾允许"对侧臂/头两组槽位回退 manifest 冻结示例语义 [公开示例]
（q=0 保持 + 示例增益）"。评审 I-05 指出该回退的危险性并已采纳移除：
官方示例自带"首段 2 s 实测→0 线性插值防跳变"的时序工程行为，本策略
单帧无状态未实现；若 G4 只补齐其余 null 槽的显式策略，首帧/停止帧会以
示例增益把**非零实测姿态**的对侧臂/头阶跃拉向 q=0（无插值防跳变）。
示例数值现仅存于 manifest 作为 G4 现场冻结时的对照参考（[公开示例]
证据等级，不得冒充实机核验值），运行时一律要求显式策略。

当前 4.2 冻结的硬件配置 schema（五节：communication/layout/calibration/
limits/timing）没有非目标槽位策略节，"硬件配置显式策略"经构造参数
``slot_policies`` 注入（由 4.6/4.7 在 G4 现场确认后接线）。
全部 35-7=28 个非目标槽位（对侧臂 7 + 头 2 + waist(13)/lift_column(12)/
chassis(0,1)/未编址(2-11,14,31-34) 共 19）均须显式给出策略——
slot_policies 缺任一槽位时 build_active/build_stop 必须拒绝（缺全机
策略拒绝构造），不得默认猜 mode=0，也不得回退示例数值。

槽位组 → 策略来源映射表（R5 修订版）：
    +---------------------------+----------------------------------------------+
    | 目标臂 7 槽（15-21/22-28） | 计算：q=to_sdk_positions(q_desired) 首帧锚定 |
    |                           | 新鲜实测姿态；kp/kd=limits 配置；dq/tau 恒 0；|
    |                           | mode=1（motor_cmd_layout.mode 1=Enable）     |
    +---------------------------+----------------------------------------------+
    | 全部非目标 28 槽          | 仅硬件配置显式策略（slot_policies）；缺任一   |
    | （对侧臂 7/头 2/waist/    | 槽位即拒绝构造。对侧臂/头的 manifest 示例     |
    | lift/chassis/未编址）     | 数值仅作 G4 冻结对照参考，不作运行时回退     |
    +---------------------------+----------------------------------------------+

头字段（主代理 Ruling R6）：
- mode_machine 一律回填自 state.mode_machine（双侧官方示例一致）；
  仅当构造参数 expected_mode_machine 冻结了期望值且与状态不符时拒绝
  （"机型不匹配"）；未冻结（None）时不做相等断言，只回填透传
  （manifest dds.mode_machine_validation null）。
- mode_pr 默认 0（manifest dds.lowcmd_layout.mode_pr + 低层 Python 示例
  写 0），可经构造参数显式覆盖。

目标七槽数值来源：
- q：mapping.to_sdk_positions(q_desired) 换算；首帧（无在途命令时）变化
  约束以**新鲜实测姿态**为基准（fresh 判定标准从硬件配置
  timing.state_timeout_s 读取），后续帧以在途上一命令为基准——因此运动
  起点只能从实测姿态附近开始（与 4.8 "tracking 从实测姿态开始"呼应）。
- kp/kd：硬件配置 limits.kp_nm_per_rad / limits.kd_nms_per_rad
  （SDK 侧增益，按关节名寻址）。
- dq/tau：恒 0——当前无已核验非零来源，**不得从仿真重力补偿搬数值**
  （附加力矩只按已核验策略设置；非目标槽位的 tau 可经显式策略给出）。
- 目标位置限位 [q_lower_rad, q_upper_rad] 与单周期隐含速度约束
  |Δq|/period ≤ dq_limit_rad_s 均来自硬件配置；过界抛显式拒绝异常，
  **不静默截断**。本策略不做内部插值限速；若上层引入插值，发送前仍须
  以本策略完整校验输出。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from src.real.r1d_config import R1DHardwareConfig
from src.real.r1d_mapping import R1DJointMapping
from src.real.r1d_types import R1DCommandFrame, R1D_SLOT_COUNT, R1DStateFrame
from src.robot.robot_profile import RobotProfile

# ---------------------------------------------------------------------------
# 协议事实常量（configs/robots/r1d_protocol_manifest.yaml，Task 4.1 冻结）
# ---------------------------------------------------------------------------

# whole_body.selected_arm.joint_indices_left/right：左臂 15-21 / 右臂 22-28
# [公开示例]（官方双侧 A7 示例 joint_idx_in_idl）。运行时目标槽位由
# R1DJointMapping 按硬件配置 joint_slots 解析（按名寻址）；本常量仅供
# manifest 一致性对照测试（tests/test_r1d_manifest_consistency.py）与文档
# 参考，不参与运行时寻址
_LEFT_ARM_SLOTS = (15, 16, 17, 18, 19, 20, 21)
_RIGHT_ARM_SLOTS = (22, 23, 24, 25, 26, 27, 28)
# whole_body.head.joint_indices：pitch=29, yaw=30 [公开示例]（同上仅对照用）
_HEAD_SLOTS = (29, 30)
# motor_cmd_layout.mode：1=Enable 0=Disable（示例注释）——目标槽位用 1。
# R1DSlotHold.mode 接受完整 uint8 域（官方仅公开 0/1 语义，其余值属
# 调用方显式给出的责任范围，见该字段 docstring）
_MODE_ENABLE = 1
# dds.lowcmd_layout.mode_pr：低层路径 Python 示例写 0（Ruling R6 默认）
_DEFAULT_MODE_PR = 0

# whole_body.other_arm / whole_body.head 的 hold_semantics_example 数值
# （对侧臂 kp=[100,100,100,100,50,50,50]/kd=2、头 kp=[50,10]/kd=[2,0.1]、
# q=0 保持）**不作运行时默认**（R5 修订 / 评审 I-05，2026-11-17）：
# 官方示例自带"首段 2s 实测→0 线性插值防跳变"，本策略单帧无状态未实现，
# 直接回退会把非零实测姿态的对侧臂/头阶跃拉向 q=0。示例数值仅存
# manifest 供 G4 现场冻结显式策略时对照；非目标槽位一律要求
# slot_policies 显式给出，缺失即拒绝构造。

# 步长比较容差：抵消 sign/offset 换算的浮点噪声
_FLOAT_EPS = 1e-12


class R1DCommandPolicyError(ValueError):
    """命令构造显式拒绝（缺状态/新鲜度/机型不匹配/缺全机策略/约束过界等）。

    上游 4.6 捕获后按故障锁存处理；本异常即"拒绝构造"，绝不以截断或
    默认值替代。
    """


@dataclass(frozen=True)
class R1DSlotHold:
    """单个槽位的保持策略（协议 MotorCmd_ 全字段语义）。

    用于显式给出非目标槽位策略（Ruling R5 第一优先级来源）；数值须来自
    已核验来源（G4 现场确认后冻结），manifest 示例值不得冒充实机核验值。

        Attributes:
            mode: 电机模式（协议 uint8 域；官方注释仅公开 1=Enable /
                0=Disable 两个取值，其余 uint8 值协议语义未公开——显式
                给出超出 0/1 的值属调用方责任范围，本类型只做域校验）。
        q: 目标位置（rad）。
        dq: 目标速度（rad/s）。
        kp: 位置增益（N·m/rad，非负）。
        kd: 速度增益（N·m·s/rad，非负）。
        tau: 前馈力矩（N·m，可为负）。
    """

    mode: int
    q: float
    dq: float
    kp: float
    kd: float
    tau: float

    def __post_init__(self) -> None:
        """校验 mode 整数 uint8、数值有限、增益非负。"""
        if isinstance(self.mode, bool) or not isinstance(self.mode, (int, np.integer)):
            raise ValueError("R1DSlotHold.mode 必须是整数（协议 uint8 语义）")
        mode = int(self.mode)
        if not 0 <= mode <= 255:
            raise ValueError(f"R1DSlotHold.mode 超出 uint8 范围 [0, 255]: {mode}")
        object.__setattr__(self, "mode", mode)
        for name in ("q", "dq", "kp", "kd", "tau"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"R1DSlotHold.{name} 必须是数值")
            number = float(value)
            if not bool(np.isfinite(number)):
                raise ValueError(f"R1DSlotHold.{name} 必须是有限值")
            object.__setattr__(self, name, number)
        if self.kp < 0.0:
            raise ValueError(f"R1DSlotHold.kp 必须非负，实际 {self.kp}")
        if self.kd < 0.0:
            raise ValueError(f"R1DSlotHold.kd 必须非负，实际 {self.kd}")


@dataclass(frozen=True)
class _MotionNumbers:
    """build 路径必需的 motion 级硬件参数（缺失即拒绝，null 不能启用运动）。

    Attributes:
        kp: 每关节位置增益（按关节名，SDK 侧）。
        kd: 每关节速度增益（按关节名，SDK 侧）。
        lower: 每关节位置下限（SDK 侧 rad）。
        upper: 每关节位置上限（SDK 侧 rad）。
        dq_limit: 每关节速度限值（SDK 侧 rad/s）。
        period: 控制周期（秒，正有限）。
    """

    kp: Mapping[str, float]
    kd: Mapping[str, float]
    lower: Mapping[str, float]
    upper: Mapping[str, float]
    dq_limit: Mapping[str, float]
    period: float


def _require_uint8_int(value: object, name: str) -> int:
    """校验 uint8 语义整数（bool 视为非法）。"""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"字段 {name} 必须是整数（uint8 语义）")
    number = int(value)
    if not 0 <= number <= 255:
        raise ValueError(f"字段 {name} 超出 uint8 范围 [0, 255]: {number}")
    return number


class R1DCommandPolicy:
    """R1-A7-D 全机命令构造策略（目标臂有限位置目标 → 完整 35 槽命令帧）。

    构造不触发 motion 级校验（4.6 需在 offline 级配置下也能构造本对象进入
    READ_ONLY）；build_active / build_stop 时逐项拒绝缺数值的 motion 级
    参数（校准/增益/限位/周期/新鲜度，null 一律拒绝）。注意（B-M1，
    评审 2026-11-17）：calibration.verified / limits.verified 布尔门槛
    不在本层复查——该门槛由 4.6 activate 前的 purpose="motion" 校验
    承担；本层只对"已进入 build 路径的数值"做非 null 与有限性防御
    （绕过 activate 直接调 build 属调用方误用）。

    线程安全性：非线程安全——4.6 的单一发送线程持有调用权。

    Args:
        config: 已加载的硬件配置（R1DHardwareConfig）。
        profile: R1D 单侧 7 关节机器人描述。
        slot_policies: 显式非目标槽位策略（Ruling R5 修订版唯一运行时
            来源，槽位索引 → 保持策略）；None 表示不提供，此时全部 28 个
            非目标槽位缺失，build_active/build_stop 将拒绝（manifest
            示例值不作运行时默认，见模块 docstring）。不能包含目标臂槽位。
        expected_mode_machine: 冻结的期望机型值（G3 实机回读后冻结）；
            None 表示未冻结，不做相等断言（Ruling R6）。
        mode_pr: hg LowCmd_ 首字段；None 取默认 0（manifest 低层 Python
            示例写 0），可显式覆盖。
        clock: 单调时钟注入（默认 time.monotonic）；新鲜度判定用。
    """

    def __init__(
        self,
        config: R1DHardwareConfig,
        profile: RobotProfile,
        *,
        slot_policies: Mapping[int, R1DSlotHold] | None = None,
        expected_mode_machine: int | None = None,
        mode_pr: int | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """校验并保存构造参数；mapping 与 motion 级参数延迟到 build 时解析。"""
        self._config = config
        self._profile = profile
        # 显式槽位策略：边界复制为内部 dict，防调用方修改
        explicit: dict[int, R1DSlotHold] = {}
        if slot_policies is not None:
            for slot, hold in slot_policies.items():
                if isinstance(slot, bool) or not isinstance(slot, (int, np.integer)):
                    raise ValueError(
                        f"显式槽位策略的键必须是整数槽位索引，实际 {slot!r}"
                    )
                slot_int = int(slot)
                if not 0 <= slot_int < R1D_SLOT_COUNT:
                    raise ValueError(
                        f"显式槽位策略的槽位 {slot_int} 越界"
                        f"（整机 {R1D_SLOT_COUNT} 槽，合法 0..{R1D_SLOT_COUNT - 1}）"
                    )
                if not isinstance(hold, R1DSlotHold):
                    raise ValueError(
                        f"槽位 {slot_int} 的策略必须是 R1DSlotHold 实例，"
                        f"实际 {type(hold).__name__}"
                    )
                explicit[slot_int] = hold
        self._slot_policies: dict[int, R1DSlotHold] = explicit
        # Ruling R6：冻结期望机型值（None = 未冻结，只回填透传）
        self._expected_mode_machine = (
            None
            if expected_mode_machine is None
            else _require_uint8_int(expected_mode_machine, "expected_mode_machine")
        )
        # Ruling R6：mode_pr 默认 0，可显式覆盖
        self._mode_pr = (
            _DEFAULT_MODE_PR if mode_pr is None else _require_uint8_int(mode_pr, "mode_pr")
        )
        self._clock: Callable[[], float] = time.monotonic if clock is None else clock

        # 延迟解析的映射（未标定时构造仍可成功，build 时拒绝）
        self._mapping: R1DJointMapping | None = None
        # 在途上一命令的目标臂 SDK 侧目标（首帧锚定基准）；None 表示首帧
        self._last_target_q_sdk: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 公开接口（跨任务接口契约，4.2 冻结）
    # ------------------------------------------------------------------

    def build_active(
        self, state: R1DStateFrame, q_desired: np.ndarray
    ) -> R1DCommandFrame:
        """构造一帧完整 35 槽主动控制命令（目标臂执行 q_desired）。

        Args:
            state: 新鲜整机状态帧（fresh 判定标准 = 硬件配置
                timing.state_timeout_s，与注入时钟比较）。
            q_desired: 模型序 7 关节目标位置（rad，有限）。

        Returns:
            完整 35 槽命令帧：目标七槽锚定换算目标，全部非目标槽位按
            已解析保持策略填写；mode_machine 回填自状态，mode_pr 按构造值。

        Raises:
            R1DCommandPolicyError: 缺状态/未标定/缺 motion 级参数/状态不
                新鲜/机型不匹配/缺全机策略/目标限位或变化约束过界——
                一律显式拒绝，不静默截断。
        """
        if state is None or not isinstance(state, R1DStateFrame):
            raise R1DCommandPolicyError(
                "缺状态：build_active 需要新鲜 R1DStateFrame（None 或类型不符拒绝构造）"
            )
        mapping = self._resolve_mapping()
        numbers = self._resolve_motion_numbers()
        self._require_fresh(state)
        self._require_mode_match(state)
        idx = np.asarray(mapping.sdk_indices, dtype=np.intp)
        holds = self._resolve_non_target_holds(idx)
        try:
            q_sdk = mapping.to_sdk_positions(q_desired)
        except ValueError as error:
            raise R1DCommandPolicyError(f"目标 q_desired 非法：{error}") from error
        self._check_position_bounds(mapping.joint_names, q_sdk, numbers)
        self._check_step(mapping.joint_names, q_sdk, state, idx, numbers)
        frame = self._assemble_frame(state, idx, mapping.joint_names, q_sdk, numbers, holds)
        # 记录在途目标（首帧锚定基准推进到本帧）
        self._last_target_q_sdk = q_sdk.copy()
        return frame

    def build_stop(
        self, state: R1DStateFrame | None, *, emergency: bool
    ) -> R1DCommandFrame | None:
        """构造有限停止命令；无法安全构造时返回 None（不伪造保持角度）。

        正常 stop 用新鲜 state 构造锚定实测姿态的有限停止目标（目标七槽
        q=实测、dq=0、tau=0、kp/kd=配置值；非目标槽位与 active 同来源
        ——一律取显式 slot_policies，**不自动保持实测、不回退示例值**，
        G4 冻结显式策略时须为非目标槽决定保持目标与增益）。
        emergency 分支构造同样的有限保持帧——协议侧无已核验的更强急停
        报文语义（manifest control.link_loss / release 均 null），差异化
        的打断/故障锁存语义归 4.6。返回 None 仅表无法安全构造（缺状态/
        不新鲜/机型不匹配/缺全机策略/缺 motion 级参数），调用方须故障
        锁存并报告未确认停止。

        Args:
            state: 新鲜整机状态帧；None 表示无新鲜状态。
            emergency: 是否急停请求（策略层构造同一有限保持帧）。

        Returns:
            完整 35 槽停止命令帧；无法安全构造时 None。
        """
        if state is None or not isinstance(state, R1DStateFrame):
            return None
        try:
            mapping = self._resolve_mapping()
            numbers = self._resolve_motion_numbers()
            self._require_fresh(state)
            self._require_mode_match(state)
            idx = np.asarray(mapping.sdk_indices, dtype=np.intp)
            holds = self._resolve_non_target_holds(idx)
        except R1DCommandPolicyError:
            return None
        # 锚定新鲜实测姿态的有限停止目标（SDK 侧实测值无需换算）
        q_stop = np.asarray(state.q[idx], dtype=np.float64).copy()
        frame = self._assemble_frame(
            state, idx, mapping.joint_names, q_stop, numbers, holds
        )
        # 复位首帧锚定基准：停止后的下一次 active 以新实测为基准重新锚定
        self._last_target_q_sdk = None
        return frame

    @property
    def last_target_q_sdk(self) -> np.ndarray | None:
        """最近一次 build_active 成功构造的目标臂 SDK 侧目标（可记录）。

        返回副本（外部修改不污染内部）；从未构造或最近一次为 build_stop
        时为 None。诊断工具（4.8）用它记录实际发送目标。
        """
        if self._last_target_q_sdk is None:
            return None
        return self._last_target_q_sdk.copy()

    # ------------------------------------------------------------------
    # 内部：延迟解析与校验
    # ------------------------------------------------------------------

    def _resolve_mapping(self) -> R1DJointMapping:
        """解析七关节映射（未标定时拒绝——未知校准不能启用运动）。

        Returns:
            按模型序排列的 R1DJointMapping（成功后缓存）。

        Raises:
            R1DCommandPolicyError: 校准为 null 或按名表不一致（包装上游
                R1DJointMapping.from_profile 的 ValueError）。
        """
        if self._mapping is None:
            try:
                self._mapping = R1DJointMapping.from_profile(self._profile, self._config)
            except ValueError as error:
                raise R1DCommandPolicyError(
                    f"未标定槽位/映射不可用，拒绝构造命令：{error}"
                ) from error
        return self._mapping

    def _resolve_motion_numbers(self) -> _MotionNumbers:
        """解析 build 路径必需的 motion 级参数（null 待确认即拒绝）。

        Returns:
            增益/限位/周期集合（绕过 load 直接构造 config 时做防御性复查）。

        Raises:
            R1DCommandPolicyError: 任一必需字段为 null，或周期非正有限、
                增益/速度限值非正。
        """
        limits = self._config.limits
        timing = self._config.timing
        required: dict[str, object] = {
            "limits.kp_nm_per_rad": limits.kp_nm_per_rad,
            "limits.kd_nms_per_rad": limits.kd_nms_per_rad,
            "limits.q_lower_rad": limits.q_lower_rad,
            "limits.q_upper_rad": limits.q_upper_rad,
            "limits.dq_limit_rad_s": limits.dq_limit_rad_s,
            "timing.control_period_s": timing.control_period_s,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise R1DCommandPolicyError(
                f"硬件配置缺少 motion 级参数（null 待确认不能启用运动）: {missing}"
            )
        period = float(timing.control_period_s)  # type: ignore[arg-type]
        if not bool(np.isfinite(period)) or period <= 0.0:
            raise R1DCommandPolicyError(
                f"timing.control_period_s 必须为正有限值，实际 {period}"
            )
        numbers = _MotionNumbers(
            kp=limits.kp_nm_per_rad,  # type: ignore[arg-type]
            kd=limits.kd_nms_per_rad,  # type: ignore[arg-type]
            lower=limits.q_lower_rad,  # type: ignore[arg-type]
            upper=limits.q_upper_rad,  # type: ignore[arg-type]
            dq_limit=limits.dq_limit_rad_s,  # type: ignore[arg-type]
            period=period,
        )
        # 防御性复查：直接构造 config 时 load 层非负校验不经过
        for name, value in (
            ("limits.kp_nm_per_rad", numbers.kp),
            ("limits.kd_nms_per_rad", numbers.kd),
            ("limits.dq_limit_rad_s", numbers.dq_limit),
        ):
            negative = sorted(k for k, v in value.items() if v < 0.0)  # type: ignore[union-attr]
            if negative:
                raise R1DCommandPolicyError(
                    f"字段 {name} 含负值（物理量须非负）: {negative}"
                )
        dq_nonpositive = sorted(
            k for k, v in numbers.dq_limit.items() if v <= 0.0  # type: ignore[union-attr]
        )
        if dq_nonpositive:
            raise R1DCommandPolicyError(
                f"字段 limits.dq_limit_rad_s 须为正值（零限值使变化约束失效）: "
                f"{dq_nonpositive}"
            )
        return numbers

    def _require_fresh(self, state: R1DStateFrame) -> None:
        """校验状态新鲜度（fresh 判定标准来自硬件配置，Ruling R5/R6）。

        Raises:
            R1DCommandPolicyError: state_timeout_s 为 null（无法判定）、
                received_at 晚于当前时钟（age<0）或超时（age>timeout）。
        """
        timeout = self._config.timing.state_timeout_s
        if timeout is None:
            raise R1DCommandPolicyError(
                "无法判定状态新鲜度：timing.state_timeout_s 为 null"
                "（fresh 判定标准须从硬件配置读取）"
            )
        now = self._clock()
        age = now - state.received_at
        if age < 0.0:
            raise R1DCommandPolicyError(
                f"状态不新鲜：received_at={state.received_at} 晚于当前时钟"
                f" {now}（age={age:.6f}s < 0），无法锚定实测姿态"
            )
        if age > timeout:
            raise R1DCommandPolicyError(
                f"状态不新鲜：age={age:.6f}s 超过 state_timeout_s={timeout}s"
                "（fresh 判定标准来自硬件配置）"
            )

    def _require_mode_match(self, state: R1DStateFrame) -> None:
        """机型校验：仅当冻结了期望值时做相等断言（Ruling R6）。

        Raises:
            R1DCommandPolicyError: 冻结期望值与状态回读不符（机型不匹配）。
        """
        if (
            self._expected_mode_machine is not None
            and state.mode_machine != self._expected_mode_machine
        ):
            raise R1DCommandPolicyError(
                f"机型不匹配：冻结期望 mode_machine={self._expected_mode_machine}，"
                f"状态回读 {state.mode_machine}，拒绝构造"
            )

    def _resolve_non_target_holds(self, target_idx: np.ndarray) -> dict[int, R1DSlotHold]:
        """解析全部非目标槽位的保持策略（仅显式来源；缺失即拒绝）。

        R5 修订（评审 I-05，2026-11-17）：不再回退 manifest 示例语义——
        官方示例的 q=0 保持自带 2 s 插值防跳变（本策略未实现），直接
        回退会把非零实测姿态的对侧臂/头阶跃拉向 0。示例数值仅存 manifest
        供 G4 现场冻结显式策略时对照。

        Args:
            target_idx: 目标臂 7 槽索引数组。

        Returns:
            槽位 → 保持策略（覆盖全部 35-7=28 个非目标槽位）。

        Raises:
            R1DCommandPolicyError: 显式策略覆盖了目标臂槽位，或任一非目标
                槽位缺显式策略（缺全机策略，列出全部缺失槽位——不得默认
                猜 mode=0，也不得回退示例数值）。
        """
        target_set = set(int(slot) for slot in target_idx.tolist())
        overlap = sorted(set(self._slot_policies) & target_set)
        if overlap:
            raise R1DCommandPolicyError(
                f"显式槽位策略不能覆盖目标臂槽位（目标槽位由映射+配置计算）: {overlap}"
            )
        holds: dict[int, R1DSlotHold] = {}
        missing: list[int] = []
        for slot in range(R1D_SLOT_COUNT):
            if slot in target_set:
                continue
            hold = self._slot_policies.get(slot)
            if hold is None:
                missing.append(slot)
            else:
                holds[slot] = hold
        if missing:
            raise R1DCommandPolicyError(
                "缺全机策略：以下非目标槽位无显式保持策略（manifest 示例值"
                "不作运行时默认——官方 q=0 保持自带 2s 插值防跳变，本策略"
                "未实现，回退会造成非零姿态阶跃；G4 现场冻结后经 slot_policies "
                f"显式提供），拒绝构造：{missing}"
            )
        return holds

    def _check_position_bounds(
        self,
        joint_names: tuple[str, ...],
        q_sdk: np.ndarray,
        numbers: _MotionNumbers,
    ) -> None:
        """校验目标位置在硬件配置限位内（SDK 侧，先于步长检查）。

        Raises:
            R1DCommandPolicyError: 任一关节目标越界（含关节名与数值）。
        """
        lower = np.asarray(
            [numbers.lower[name] for name in joint_names], dtype=np.float64
        )
        upper = np.asarray(
            [numbers.upper[name] for name in joint_names], dtype=np.float64
        )
        for mask, bound_name, bound in (
            (q_sdk > upper, "上", upper),
            (q_sdk < lower, "下", lower),
        ):
            over = np.nonzero(mask)[0]
            if over.size:
                j = int(over[0])
                raise R1DCommandPolicyError(
                    f"目标位置限位过界：关节 {joint_names[j]} 目标 "
                    f"{q_sdk[j]:.6f} rad 超出硬件配置{bound_name}限位 "
                    f"{bound[j]:.6f} rad（显式拒绝，不静默截断）"
                )

    def _check_step(
        self,
        joint_names: tuple[str, ...],
        q_sdk: np.ndarray,
        state: R1DStateFrame,
        idx: np.ndarray,
        numbers: _MotionNumbers,
    ) -> None:
        """校验单周期目标步长隐含速度 ≤ dq_limit（过界显式拒绝）。

        基准（Ruling R5）：首帧（无在途命令）以**新鲜实测姿态** state.q
        为基准；后续帧以在途上一命令为基准。不做内部插值限速——上层
        须自行生成平滑目标（4.8 tracking 从实测姿态开始）。

        Raises:
            R1DCommandPolicyError: 任一关节步长隐含速度超限（含关节名、
                步长与允许值）。
        """
        first_frame = self._last_target_q_sdk is None
        reference = state.q[idx] if first_frame else self._last_target_q_sdk
        assert reference is not None  # 上一分支保证非 None
        max_step = np.asarray(
            [numbers.dq_limit[name] * numbers.period for name in joint_names],
            dtype=np.float64,
        )
        delta = np.abs(q_sdk - reference)
        over = np.nonzero(delta > max_step + _FLOAT_EPS)[0]
        if over.size:
            j = int(over[0])
            basis = "首帧以新鲜实测姿态为基准" if first_frame else "以在途上一命令为基准"
            raise R1DCommandPolicyError(
                f"目标位置变化约束过界（{basis}）：关节 {joint_names[j]} "
                f"单周期步长 {delta[j]:.6f} rad 超出 dq_limit*周期="
                f"{max_step[j]:.6f} rad（隐含速度超限），显式拒绝，不静默截断"
            )

    def _assemble_frame(
        self,
        state: R1DStateFrame,
        idx: np.ndarray,
        joint_names: tuple[str, ...],
        q_target_sdk: np.ndarray,
        numbers: _MotionNumbers,
        holds: Mapping[int, R1DSlotHold],
    ) -> R1DCommandFrame:
        """组装完整 35 槽命令帧（构造时由 R1DCommandFrame 全量校验输出）。

        Args:
            state: 状态帧（提供 mode_machine 回填）。
            idx: 目标臂 7 槽索引。
            joint_names: 模型序关节名。
            q_target_sdk: 目标臂 SDK 侧目标位置（模型序）。
            numbers: motion 级参数（目标槽位增益）。
            holds: 全部非目标槽位的保持策略。

        Returns:
            完整 35 槽命令帧。
        """
        q = np.zeros(R1D_SLOT_COUNT, dtype=np.float64)
        dq = np.zeros(R1D_SLOT_COUNT, dtype=np.float64)
        kp = np.zeros(R1D_SLOT_COUNT, dtype=np.float64)
        kd = np.zeros(R1D_SLOT_COUNT, dtype=np.float64)
        tau = np.zeros(R1D_SLOT_COUNT, dtype=np.float64)
        motor_mode = np.zeros(R1D_SLOT_COUNT, dtype=np.float64)

        # 目标七槽：q=换算/停止目标，dq/tau 恒 0（无已核验非零来源，
        # 不得从仿真重力补偿搬数值），kp/kd 来自硬件配置，mode=Enable
        q[idx] = q_target_sdk
        kp[idx] = np.asarray(
            [numbers.kp[name] for name in joint_names], dtype=np.float64
        )
        kd[idx] = np.asarray(
            [numbers.kd[name] for name in joint_names], dtype=np.float64
        )
        motor_mode[idx] = _MODE_ENABLE

        # 非目标 28 槽：按已解析保持策略逐槽填写（唯一来源=显式策略，
        # R5 修订后无 manifest 示例回退）
        for slot, hold in holds.items():
            q[slot] = hold.q
            dq[slot] = hold.dq
            kp[slot] = hold.kp
            kd[slot] = hold.kd
            tau[slot] = hold.tau
            motor_mode[slot] = hold.mode

        # 发送前完整校验输出：R1DCommandFrame 构造即校验长度/有限性/
        # motor_mode 整数 uint8 与头字段 uint8 范围
        return R1DCommandFrame(
            q=q,
            dq=dq,
            kp=kp,
            kd=kd,
            tau=tau,
            motor_mode=motor_mode,
            mode_machine=state.mode_machine,
            mode_pr=self._mode_pr,
        )
