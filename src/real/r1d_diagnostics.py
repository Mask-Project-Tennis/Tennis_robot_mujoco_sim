"""R1-A7-D 只读/低速诊断工具与可追踪指标 — 阶段 4 Task 4.8。

交付 ``run_r1d_diagnostic``：在注入 fake/真实传输的 ``UnitreeR1DInterface``
上执行只读状态流记录（read_only）或单关节低速跟踪诊断（tracking），产出
可追踪指标（JSON 摘要）与原始序列（CSV + NPZ）。全部测试路径使用
fake transport / fake clock，本模块不导入 SDK/MuJoCo、不启动线程。

数值来源纪律（主代理裁定 4）：幅度/频率/时长/采样周期/发送周期/各阈值
全部来自 ``R1DDiagnosticConfig``（现场配置，含来源路径与哈希；未知项
null 不补猜），本模块不硬编码任何实机数值。SDK 标识为 R4 冻结事实常量
（manifest sdk.python_commit）；代码 SHA 经只读 ``git rev-parse HEAD``
获取，失败记 null 并留原因，不编造。

时刻语义（R9）：CSV/NPZ 的发布时刻一律取自
``R1DPublishRecord.published_at``（transport.publish 调用返回点），上层
``send_joint_command`` 的接受时刻在 send 日志中单独记录，两者绝不混用。

时基约定：诊断与驱动共享同一注入单调时钟（调用方构造 arm 时注入同一
实例，4.6 评审 M-3 共享时基的延伸）。本模块经同包内部约定访问
``arm._clock`` / ``arm._buffer`` / ``arm._model_state_from_frame``（后者
保证模型序状态与接收时刻出自同一状态帧，R5 评审 2026-11-18）；跨包
调用方不应依赖这些私有入口——访问不到时降级为 time.monotonic /
独立 get_arm_state 读取 / 无完整帧，并在 JSON provenance 显式记录
（shared_clock=false / frame_aligned=false / full_state_available=false）。

tracking 波形（首版，单关节，升余弦）::

    δ(t) = A/2 · (1 − cos(2π·f·t))
    q_target[j] = q_start[j] + δ(t)，其余关节目标恒等于锚定值

理由：起止时刻 δ=0 且 dδ/dt=0——从实测锚定姿态零速度出发、整周期后
零速度返回，满足"从实测姿态锚定开始 / 结束回到起始目标"；峰值 A 在
半周期处；最大隐含速度 = A·π·f（t=1/(4f) 处），现场配置须保证
A·π·f ≤ 选中关节 dq_limit，超出时驱动步长校验显式拒绝（不静默截断）。
非整周期时长下结束点 δ(T)≠0：回起始段按"波形已实际通过的最大单拍
步长"限速逐拍逼近起点（R6，评审 2026-11-18）——一拍跳回起点的步长
可远超 dq_limit·周期，会被驱动拒绝并中断诊断；受限速回程不放宽任何
驱动限制。

误差对齐规则（同时写入 JSON ``tracking_metrics.alignment_rule``）：
"每条成功发布的 active 记录 r（published_at、q_sdk=SDK 侧数值按模型序
排列）与**帧接收时刻**（received_at，与模型序状态同帧原子取得）严格
晚于 r.published_at 的最早未消费状态样本配对；目标换算到模型序
q_target=(q_sdk−offset)·sign（换算参数来自现场配置的校准节，模型序排
列）；误差 = q_fb_model − q_target（模型序，rad）。同一物理反馈帧（身份 =
received_at + sequence）只作一条命令的反馈证据——重复采样到的同一缓存
帧不为后续命令重复作证；received_at 早于或等于发布时刻的发布前缓存不
参与配对；无可靠帧时间的样本同样不参与配对（降级证据，不伪装已对齐）。
无匹配样本的记录计入 no_match_samples，
不参与误差统计；全部记录无匹配时各关节误差标 unavailable（null），不
得记 0——无样本不能得出零误差通过。"（R5，评审 2026-11-18：旧规则按
采样读取完成时刻配对，发布前到达、尚未超时的旧缓存会被计为发布后反
馈，误差统计形成虚假的新反馈证据；复审 2026-11-18：样本行逐条配对会
把同一缓存帧计为多条命令已匹配，帧身份去重后每帧只作证一次。）

失败路径纪律（裁定 8）：序列中断、数据不足、无法确认停止、Ctrl+C/
异常一律生成失败结论（verdict=fail/aborted + 原因列表）并清理资源
（停止尝试 + disconnect）；资源清理异常如实记录，不掩盖原始原因。
"""

from __future__ import annotations

import csv
import json
import logging
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from src.real.r1d_config import R1D_ARM_JOINT_COUNT
from src.real.unitree_r1d_interface import (
    R1DPublishRecord,
    SEND_ACCEPTED,
    SEND_REJECT_INVALID_TARGET,
    SEND_REJECT_LATCHED,
    SEND_REJECT_PHASE,
    UnitreeR1DInterface,
)

logger = logging.getLogger(__name__)

__all__ = [
    "R1DDiagnosticConfig",
    "R1D_DIAGNOSTIC_MODES",
    "R1D_DIAGNOSTIC_SCHEMA_VERSION",
    "R1D_SDK_IDENTITY",
    "compute_publish_interval_metrics",
    "compute_tracking_error_metrics",
    "run_r1d_diagnostic",
]

# 诊断产物 schema 版本（JSON "schema" 字段）
R1D_DIAGNOSTIC_SCHEMA_VERSION = 1
# 合法诊断模式
R1D_DIAGNOSTIC_MODES = ("read_only", "tracking")
# SDK 标识（R4 冻结事实：manifest sdk.python_commit；协议证据链见
# docs/experiments/reports/2026-11-16-r1d-sdk-control-contract.md G1-02）
R1D_SDK_IDENTITY = "unitree_sdk2py@814556d15970dd2ecf1c9984e845ca02ab07e206"

# send_joint_command 返回码 → 具名常量名（拒绝原因引用具名常量，不裸数字）
_SEND_CODE_NAMES: dict[int, str] = {
    SEND_ACCEPTED: "SEND_ACCEPTED",
    SEND_REJECT_PHASE: "SEND_REJECT_PHASE",
    SEND_REJECT_INVALID_TARGET: "SEND_REJECT_INVALID_TARGET",
    SEND_REJECT_LATCHED: "SEND_REJECT_LATCHED",
}

# 停止后低速反馈探测的采样次数上限（软件工程参数，非实机数值：只决定
# "反馈速度达阈值时刻"的探测窗口长度 = 次数 × 发送周期，来自配置周期）
_STOP_PROBE_SAMPLES = 10
# 发布间隔超期判定的时间容差（秒，纯浮点保护）：间隔恰等于期限是按时
# 发布——R1 限频后实测间隔恰为控制周期，严格 > 在 0.002 vs
# 0.0020000000000000018 的尾差上误报超期（评审 2026-11-18 修复轮）
_PUBLISH_INTERVAL_TOLERANCE_S = 1e-9
# 发布间隔超期的判据容差（相对期限的比例，软件工程参数非实机数值）：
# 驱动按"上一帧 publish 返回后再等一个完整周期"排下一拍（R1 限频保证间
# 隔 ≥ 周期、不补发），因此相邻发布的实际间隔 = 周期 + 发布/唤醒开销，
# 恒略大于周期。若把判据设为"间隔 > 周期"则每次正常发布都算超期（评审
# 2026-11-18 复现：2 ms 周期、20 µs 发送耗时 → 全部区间超期），验收永
# 远不能通过。判据改为"错过半拍以上才算超期"：开销是数十~数百 µs 级，
# 远小于半周期；真实漏拍（错过一个控制周期）表现为 ≈ 2× 期限，必然计入。
# 原始间隔分位（p50/p95/p99/max）仍全量落盘供现场复核，本容差不改变其值。
_PUBLISH_INTERVAL_OVERRUN_RATIO = 0.5
# git rev-parse 只读获取代码 SHA 的超时（真实秒，软件工程参数）
_GIT_SHA_TIMEOUT_S = 10.0

# 发布记录 kind → NPZ 数值编码（CSV 保留原字符串；JSON 说明编码规则）
_PUBLISH_KIND_CODES = {"active": 0, "stop": 1}
# NPZ 数值数组对 None 的哨兵（JSON npz_notes 说明；None 语义不编造）
_NPZ_INT_NONE = -1
_NPZ_FLOAT_NONE = float("nan")


# ---------------------------------------------------------------------------
# 配置（数值全部来自现场；未知项 None 不补猜）
# ---------------------------------------------------------------------------


def _require_positive(value: object, name: str) -> float:
    """校验正有限浮点（配置字段公共校验）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"R1DDiagnosticConfig.{name} 必须是数值，实际 {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(
            f"R1DDiagnosticConfig.{name} 必须为正有限值，实际 {value!r}"
        )
    return number


def _require_non_negative_or_none(value: object, name: str) -> float | None:
    """校验可选非负有限浮点（None 合法=未冻结，不判定）。"""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"R1DDiagnosticConfig.{name} 必须是数值或 null")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            f"R1DDiagnosticConfig.{name} 必须为非负有限值或 null，实际 {value!r}"
        )
    return number


@dataclass(frozen=True)
class R1DDiagnosticConfig:
    """R1-A7-D 诊断配置（现场配置数值容器 + 来源引用）。

    数值来源纪律：全部数值由现场提供（CLI 从硬件配置/显式参数构造，
    来源与回退在 JSON provenance 记录）；未知项 None 不补猜。
    与 ``R1DHardwareConfig.timing.diagnostic_duration_s`` 的关系：
    诊断时长以本字段为执行值；CLI 路径下本字段取自硬件配置
    ``timing.diagnostic_duration_s``（现场冻结值优先，工具不提供 CLI
    覆盖，防止与现场冻结值漂移）；库级调用方直接给定。

    Attributes:
        mode: "read_only"（只记录状态流，不 activate/不 send）或
            "tracking"（单关节低速跟踪）。
        duration_s: 诊断时长（秒，正有限）。
        sample_period_s: read_only 状态采样周期（秒）；仅 read_only 允许
            非 None（tracking 的采样随发送周期，不适用本字段）。
        send_period_s: tracking 发送周期（秒）= 硬件配置控制周期；
            仅 tracking 允许非 None。
        selected_joint: tracking 选中的模型序关节索引（0..6）；
            read_only 必须为 None。
        amplitude_rad: tracking 波形峰值幅度（rad，正有限）。
        frequency_hz: tracking 波形频率（Hz，正有限）。
        stop_confirm_dq_threshold: 停止反馈确认速度阈值（rad/s，非负）；
            与构造 arm 时注入的同名参数同值（来自现场配置）。
        publish_deadline_s: publish 间隔超期限阈值（秒，= 控制周期绝对
            期限）；None 表示未冻结，不判定超期（间隔指标仍计算）。
            超期判据：间隔 > 1.5 × 本期限（半拍容差吸收发布/唤醒开销，
            见 _PUBLISH_INTERVAL_OVERRUN_RATIO）。
        tracking_error_limit_rad: 跟踪误差阈值（rad）；None 不判定。
        stop_latency_limit_s: 停止延迟阈值（秒，对反馈侧延迟判定）；
            None 不判定。
        joint_signs: 模型序 7 项方向符号（±1，来自硬件配置校准节）；
            tracking 必需（SDK→模型目标换算），read_only 必须为 None。
        joint_offsets_rad: 模型序 7 项零位偏移（rad，SDK 侧）；
            与 joint_signs 同时提供或同时为 None。
        hardware_config_path: 现场硬件配置来源路径（未记录为 None）。
        hardware_config_sha256: 现场硬件配置文件 SHA256（未记录为 None）。
        side: 侧别（"left"/"right"；未记录为 None）。

    Raises:
        ValueError: 模式/数值/不适用字段（显式给出但当前模式不接受的
            字段）任何一项不合法，消息点名字段。
    """

    mode: str
    duration_s: float
    sample_period_s: float | None = None
    send_period_s: float | None = None
    selected_joint: int | None = None
    amplitude_rad: float | None = None
    frequency_hz: float | None = None
    stop_confirm_dq_threshold: float | None = None
    publish_deadline_s: float | None = None
    tracking_error_limit_rad: float | None = None
    stop_latency_limit_s: float | None = None
    joint_signs: tuple[int, ...] | None = None
    joint_offsets_rad: tuple[float, ...] | None = None
    hardware_config_path: str | None = None
    hardware_config_sha256: str | None = None
    side: str | None = None

    def __post_init__(self) -> None:
        """校验模式、数值与不适用字段（显式给出即拒绝，不静默忽略）。"""
        if self.mode not in R1D_DIAGNOSTIC_MODES:
            raise ValueError(
                f"R1DDiagnosticConfig.mode 必须是 {list(R1D_DIAGNOSTIC_MODES)} 之一，"
                f"实际 {self.mode!r}"
            )
        _require_positive(self.duration_s, "duration_s")
        # 可选阈值（None=未冻结不判定）
        object.__setattr__(
            self,
            "publish_deadline_s",
            _require_non_negative_or_none(self.publish_deadline_s, "publish_deadline_s")
            if self.publish_deadline_s is not None
            else None,
        )
        object.__setattr__(
            self,
            "tracking_error_limit_rad",
            _require_non_negative_or_none(
                self.tracking_error_limit_rad, "tracking_error_limit_rad"
            )
            if self.tracking_error_limit_rad is not None
            else None,
        )
        object.__setattr__(
            self,
            "stop_latency_limit_s",
            _require_non_negative_or_none(self.stop_latency_limit_s, "stop_latency_limit_s")
            if self.stop_latency_limit_s is not None
            else None,
        )
        # 校准换算参数：成对提供或同时 None
        if (self.joint_signs is None) != (self.joint_offsets_rad is None):
            raise ValueError(
                "R1DDiagnosticConfig.joint_signs 与 joint_offsets_rad 必须同时提供"
                "或同时为 None（换算参数成对来自现场校准节）"
            )
        for name, values in (
            ("joint_signs", self.joint_signs),
            ("joint_offsets_rad", self.joint_offsets_rad),
        ):
            if values is not None:
                if len(values) != R1D_ARM_JOINT_COUNT:
                    raise ValueError(
                        f"R1DDiagnosticConfig.{name} 必须是 {R1D_ARM_JOINT_COUNT} 项"
                        f"（模型序），实际 {len(values)} 项"
                    )
                for item in values:
                    if not math.isfinite(float(item)):
                        raise ValueError(f"R1DDiagnosticConfig.{name} 含非有限值")
                if name == "joint_signs":
                    for item in values:
                        if int(item) not in (1, -1):
                            raise ValueError(
                                f"R1DDiagnosticConfig.joint_signs 只能取 ±1，实际 {item}"
                            )

        if self.mode == "read_only":
            # read_only：不适用字段显式给出即拒绝（不静默忽略）
            _require_positive(
                self.sample_period_s, "sample_period_s（read_only 必填）"
            )
            for name, value in (
                ("send_period_s", self.send_period_s),
                ("selected_joint", self.selected_joint),
                ("amplitude_rad", self.amplitude_rad),
                ("frequency_hz", self.frequency_hz),
                ("stop_confirm_dq_threshold", self.stop_confirm_dq_threshold),
                ("joint_signs", self.joint_signs),
                ("joint_offsets_rad", self.joint_offsets_rad),
            ):
                if value is not None:
                    raise ValueError(
                        f"R1DDiagnosticConfig.{name} 仅适用于 tracking 模式，"
                        "read_only 下显式给出必须拒绝（不静默忽略）"
                    )
            object.__setattr__(self, "selected_joint", None)
        else:
            # tracking：必填字段与不适用字段
            _require_positive(self.send_period_s, "send_period_s（tracking 必填）")
            _require_positive(self.amplitude_rad, "amplitude_rad（tracking 必填）")
            _require_positive(self.frequency_hz, "frequency_hz（tracking 必填）")
            _require_non_negative_or_none(
                self.stop_confirm_dq_threshold,
                "stop_confirm_dq_threshold（tracking 必填）",
            )
            if self.stop_confirm_dq_threshold is None:
                raise ValueError(
                    "R1DDiagnosticConfig.stop_confirm_dq_threshold 在 tracking 模式"
                    "必填（来自现场配置，与构造 arm 时同名参数同值）"
                )
            if self.selected_joint is None:
                raise ValueError(
                    "R1DDiagnosticConfig.selected_joint 在 tracking 模式必填"
                )
            if (
                isinstance(self.selected_joint, bool)
                or not isinstance(self.selected_joint, int)
                or not 0 <= self.selected_joint < R1D_ARM_JOINT_COUNT
            ):
                raise ValueError(
                    f"R1DDiagnosticConfig.selected_joint 必须是 0..{R1D_ARM_JOINT_COUNT - 1}"
                    f" 的模型序关节索引，实际 {self.selected_joint!r}"
                )
            if self.sample_period_s is not None:
                raise ValueError(
                    "R1DDiagnosticConfig.sample_period_s 仅适用于 read_only 模式，"
                    "tracking 的状态采样随发送周期（不适用字段显式给出必须拒绝）"
                )
            if self.joint_signs is None or self.joint_offsets_rad is None:
                raise ValueError(
                    "R1DDiagnosticConfig.joint_signs/joint_offsets_rad 在 tracking "
                    "模式必填（SDK→模型目标换算来自现场校准节，未冻结不能算误差）"
                )
            # P1-3（2026-11-18）：接管前拒绝零/单步/混叠波形——只有离散
            # 采样真正产生可验证位移，tracking 的"目标跟随误差"才有意义；
            # 否则目标从未离开起点，"误差小"会被误判为跟踪通过。
            steps = self.tracking_step_count()
            if steps < 2:
                raise ValueError(
                    "tracking 离散波形不足 2 步（至少需要两个发送周期），"
                    "接管前拒绝："
                    f"duration_s={self.duration_s!r}、"
                    f"send_period_s={self.send_period_s!r} → steps={steps}"
                )
            if self.amplitude_rad is None:  # pragma: no cover - 前置必填校验保证
                raise ValueError(
                    "tracking 模式 amplitude_rad 必填（前置校验未生效，"
                    "无法计算离散波形位移）"
                )
            amplitude = float(self.amplitude_rad)
            span = self.expected_target_span_rad()
            span_threshold = max(0.1 * amplitude, 1e-3)
            if span < span_threshold:
                raise ValueError(
                    "tracking 离散波形位移不足（离散跨度 "
                    f"span={span:.6g} rad < 阈值 {span_threshold:.6g} rad = "
                    "max(0.1×amplitude_rad, 1e-3)）："
                    f"steps={steps}、duration_s={self.duration_s!r}、"
                    f"send_period_s={self.send_period_s!r}、"
                    f"frequency_hz={self.frequency_hz!r}——接管前拒绝"
                )

    def tracking_step_count(self) -> int:
        """返回 tracking 升余弦波形的离散步数（发送拍数）。

        与 ``run_tracking`` 使用同一取整规则
        ``max(1, int(round(duration_s / send_period_s)))``（Python ``round``
        半值取偶，保持既有行为）：配置校验、预期跨度计算与实际发送步数
        必须同源，否则"预期波形"与"实际发布波形"不可比。

        Returns:
            波形发送步数（>= 1）。

        Raises:
            ValueError: 非 tracking 模式（read_only 无发送周期与波形）。
        """
        if self.mode != "tracking":
            raise ValueError(
                "R1DDiagnosticConfig.tracking_step_count 仅适用于 tracking 模式"
                f"（当前 mode={self.mode!r}；read_only 无发送周期与波形）"
            )
        if self.send_period_s is None:  # pragma: no cover - 前置必填校验保证
            raise ValueError(
                "R1DDiagnosticConfig.tracking_step_count 需要 send_period_s"
                "（tracking 模式必填，前置校验未生效）"
            )
        return max(1, int(round(self.duration_s / self.send_period_s)))

    def expected_target_span_rad(self) -> float:
        """返回离散升余弦波形在发送拍上的模型序目标跨度（rad）。

        对 ``k = 0..steps-1`` 采样
        ``0.5·amplitude_rad·(1−cos(2π·frequency_hz·k·send_period_s))``，
        流式维护 min/max 后返回 ``max−min``（不分配与 steps 等长的列表）。
        该跨度是判定层"实际发布目标覆盖波形"的预期基准。

        Returns:
            离散采样点的目标跨度（max−min，rad）；单步波形为 0。

        Raises:
            ValueError: 非 tracking 模式（read_only 无可验证波形）。
        """
        if self.mode != "tracking":
            raise ValueError(
                "R1DDiagnosticConfig.expected_target_span_rad 仅适用于 tracking "
                f"模式（当前 mode={self.mode!r}；read_only 无可验证波形）"
            )
        if (
            self.send_period_s is None
            or self.amplitude_rad is None
            or self.frequency_hz is None
        ):  # pragma: no cover - 前置 tracking 必填校验保证
            raise ValueError(
                "R1DDiagnosticConfig.expected_target_span_rad 需要 send_period_s/"
                "amplitude_rad/frequency_hz（tracking 模式必填，前置校验未生效）"
            )
        steps = self.tracking_step_count()
        period = float(self.send_period_s)
        amplitude = float(self.amplitude_rad)
        frequency = float(self.frequency_hz)
        min_value = math.inf
        max_value = -math.inf
        for step in range(steps):
            value = 0.5 * amplitude * (
                1.0 - math.cos(2.0 * math.pi * frequency * step * period)
            )
            min_value = min(min_value, value)
            max_value = max(max_value, value)
        return float(max_value - min_value)


# ---------------------------------------------------------------------------
# 指标纯函数（公开导出，供测试手工小样本验证）
# ---------------------------------------------------------------------------


def compute_publish_interval_metrics(
    published_ats: Sequence[float], deadline_s: float | None
) -> dict[str, Any]:
    """计算实际发布间隔分位指标与超期限次数。

    Args:
        published_ats: 成功 active 发布的 published_at 序列（注入单调
            时钟秒，升序；来自 R1DPublishRecord，非上层 send 时刻）。
        deadline_s: 超期限阈值（= 控制周期绝对期限）；None 表示未冻结，
            不判定超期。判据见 _PUBLISH_INTERVAL_OVERRUN_RATIO：间隔
            超过 ``deadline_s × (1 + 0.5)`` 才计一次超期（错过半拍以上，
            吸收发布/唤醒开销；真实漏拍 ≈ 2× 期限）。

    Returns:
        指标字典：n_intervals/p50_s/p95_s/p99_s/max_s/deadline_s/
        deadline_overruns；不足 2 条发布记录时各指标为 None 且
        unavailable=True（无样本不能得出 0——间隔指标标 unavailable
        而非 0）。
    """
    count = len(published_ats)
    if count < 2:
        return {
            "n_intervals": max(0, count - 1),
            "p50_s": None,
            "p95_s": None,
            "p99_s": None,
            "max_s": None,
            "deadline_s": deadline_s,
            "deadline_overruns": None,
            "unavailable": True,
            "unavailable_reason": "成功 active 发布记录不足 2 条，无法计算间隔指标",
        }
    intervals = np.diff(np.asarray(published_ats, dtype=np.float64))
    overruns: int | None = None
    if deadline_s is not None:
        # 超期 = 错过半拍以上（见 _PUBLISH_INTERVAL_OVERRUN_RATIO 的推导）：
        # 间隔 > 期限 × 1.5。+绝对容差仅吸收浮点尾差（非放宽判据）。
        overrun_threshold = deadline_s * (1.0 + _PUBLISH_INTERVAL_OVERRUN_RATIO)
        overruns = int(
            np.count_nonzero(
                intervals > overrun_threshold + _PUBLISH_INTERVAL_TOLERANCE_S
            )
        )
    return {
        "n_intervals": int(intervals.size),
        "p50_s": float(np.percentile(intervals, 50)),
        "p95_s": float(np.percentile(intervals, 95)),
        "p99_s": float(np.percentile(intervals, 99)),
        "max_s": float(np.max(intervals)),
        "deadline_s": deadline_s,
        "deadline_overruns": overruns,
        "unavailable": False,
        "unavailable_reason": None,
    }


def compute_tracking_error_metrics(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """按对齐规则配对样本计算各关节 RMS/max 跟踪误差。

    Args:
        pairs: 已配对的 (q_target_model, q_fb_model) 序列（各 (7,) 模型序，
            rad）；配对规则见模块 docstring（发布后最早有效样本）。

    Returns:
        指标字典：per_joint（每关节 rms_rad/max_rad/n_matched）与
        n_matched/no_match 由调用方补充；无配对样本时 per_joint 各值
        None 且 unavailable=True（无样本不能得出零误差"通过"）。
    """
    joint_count = R1D_ARM_JOINT_COUNT
    if not pairs:
        return {
            "per_joint": [
                {"rms_rad": None, "max_rad": None} for _ in range(joint_count)
            ],
            "n_matched": 0,
            "unavailable": True,
            "unavailable_reason": "无配对样本（无成功发布记录或无发布后有效反馈）",
        }
    errors = np.stack(
        [np.asarray(fb, dtype=np.float64) - np.asarray(target, dtype=np.float64)
         for target, fb in pairs]
    )
    joint_count = int(errors.shape[1])
    per_joint = [
        {
            "rms_rad": float(np.sqrt(np.mean(errors[:, j] ** 2))),
            "max_rad": float(np.max(np.abs(errors[:, j]))),
        }
        for j in range(joint_count)
    ]
    return {
        "per_joint": per_joint,
        "n_matched": int(errors.shape[0]),
        "unavailable": False,
        "unavailable_reason": None,
    }


def _pair_feedback_frames(
    publish_rows: Sequence[tuple[float, np.ndarray | None]],
    sample_rows: Sequence[tuple[float | None, int | None, np.ndarray | None]],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """按帧接收时刻把发布记录与反馈样本单调配对（R5 纯函数，供测试）。

    Args:
        publish_rows: 成功 active 发布行 (published_at, 模型序目标)，
            按发布顺序；目标缺失（q_sdk 提取失败）直接计无匹配。
        sample_rows: 状态样本行 (received_at, sequence, 模型序反馈 q)；
            按 received_at 升序（调用方排序）。received_at 为 None 的样本
            无可靠帧时间，不参与配对（降级证据，不伪装已对齐）。

    Returns:
        (配对的 (目标, 反馈) 序列, 无匹配发布数)。规则：每条发布与接收
        时刻**严格晚于**其 published_at 的最早未消费反馈帧配对；帧身份 =
        (received_at, sequence)——同一物理帧被多次采样（采样周期快于状态
        流，连续两次读到同一缓存帧）只算一帧，不得为后续命令重复提供
        "新反馈"证据（评审 2026-11-18：旧实现按样本行配对，同一帧可被
        计为两条命令都已匹配，夸大反馈覆盖率）。发布前缓存
        （received_at ≤ published_at）永不匹配。
    """
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    no_match = 0
    index = 0
    last_consumed: tuple[float, int | None] | None = None
    for published_at, q_target in publish_rows:
        if q_target is None:
            no_match += 1
            continue
        matched: np.ndarray | None = None
        while index < len(sample_rows):
            received_at, sequence, q = sample_rows[index]
            index += 1
            if received_at is None or received_at <= published_at:
                continue  # 无帧时间/发布前缓存：不匹配任何发布
            identity = (float(received_at), sequence)
            if identity == last_consumed:
                continue  # 同一物理帧的重复采样：已为前一条命令作证
            matched = q
            last_consumed = identity
            break
        if matched is None:
            no_match += 1
            continue
        pairs.append((q_target, matched))
    return pairs, no_match


def _git_commit_sha() -> str | None:
    """只读获取当前仓库 HEAD SHA（git rev-parse HEAD）。

    显式指定 cwd=本文件所在仓库根（src/real 上两级），不依赖调用方
    进程的当前目录——诊断可能从任意目录启动，隐式 cwd 会在误启动目录
    恰好也是 git 仓库时取到无关 SHA（D-M1，评审 2026-11-17）。

    Returns:
        40 位 commit SHA；获取失败（非 git 环境/超时）返回 None，
        原因由调用方记录，不编造。
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_SHA_TIMEOUT_S,
            check=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        sha = result.stdout.strip()
        return sha or None
    except Exception as error:  # noqa: BLE001 - 降级为 None，原因留痕
        logger.warning("代码 SHA 获取失败（记 null）: %s", error)
        return None


# ---------------------------------------------------------------------------
# 诊断执行
# ---------------------------------------------------------------------------


class _DiagnosticCollector:
    """单次诊断的执行器：采样/记录/指标/产物写盘（内部类，勿外部使用）。"""

    def __init__(
        self,
        config: R1DDiagnosticConfig,
        output_dir: Path,
        *,
        clock: Callable[[], float] | None = None,
        wait_until: Callable[[float], None] | None = None,
    ) -> None:
        """初始化收集器并解析共享时钟。

        Args:
            config: 现场诊断配置。
            output_dir: 产物输出目录。
            clock: 可选受控时钟（测试注入）。缺省沿用驱动注入时钟
                （``arm._clock``）或 ``time.monotonic``，生产行为不变。
            wait_until: 可选受控等待（测试注入）。参数为推进到的截止
                时刻；缺省按真实墙钟 ``time.sleep`` 推进，生产行为不变。
                注入后由测试调度器确定性地推进命令→发布→反馈→采样，
                不再依赖桌面线程调度。
        """
        self._config = config
        self._clock_override = clock
        self._wait_override = wait_until
        self._output_dir = Path(output_dir)
        self._reasons: list[str] = []
        self._aborted = False
        self._aborted_reason: str | None = None
        self._completed_normally = False
        self._samples: list[dict[str, Any]] = []
        self._sends: list[dict[str, Any]] = []
        self._publishes: list[dict[str, Any]] = []
        self._status_events: list[dict[str, Any]] = []
        self._stop_info: dict[str, Any] = {
            "requested": False,
            "requested_at_s": None,
            "software_detected_at_s": None,
            "feedback_below_threshold_at_s": None,
            "stop_confirmed": None,
            "final_phase": None,
            # 停止来源（D-M5）：diagnostic=正常停止流程，cleanup=清理路径
            # 兜底停止（异常中止/Ctrl+C），两者遥测口径相同、语义不同。
            "source": None,
        }
        self._consumed_records = 0
        # 共享时钟在 _bind_arm（拿到 arm 实例后）解析
        self._clock: Callable[[], float] = time.monotonic
        self._shared_clock = False
        self._full_state_available = False
        self._full_state_probe = False
        # R5（评审 2026-11-18）：模型序状态与帧时间元数据是否同帧原子
        self._frame_aligned = False
        # 同帧原子提取入口（_bind_arm 解析；同包内部契约见模块 docstring）
        self._frame_extractor: Callable[[Any], np.ndarray] | None = None
        # 信息性备注（降级/来源缺失等；不影响 verdict，区别于失败原因）
        self._notes: list[str] = []
        self._git_sha_error: str | None = None

    # -- 基础记录 -------------------------------------------------------

    def _bind_arm(self, arm: UnitreeR1DInterface) -> None:
        """绑定驱动实例并解析共享时钟（同包内部约定，见模块 docstring）。"""
        clock_attr = getattr(arm, "_clock", None)
        if self._clock_override is not None:
            # 测试注入的受控时钟优先；与驱动同一实例时仍如实标记共享时钟
            self._clock = self._clock_override
            self._shared_clock = bool(callable(clock_attr) and clock_attr is self._clock_override)
        elif callable(clock_attr):
            self._clock = clock_attr
            self._shared_clock = True
        else:
            self._clock = time.monotonic
            self._shared_clock = False
            self._notes.append(
                "无法访问驱动注入时钟（arm._clock），降级 time.monotonic，"
                "发布/采样时刻可能跨时基"
            )
        buffer = getattr(arm, "_buffer", None)
        self._full_state_probe = buffer is not None
        if buffer is None:
            self._notes.append(
                "无法访问状态缓存（arm._buffer），完整非目标状态与"
                "反馈接收/源时间列缺失（full_state_available=false）"
            )
        # R5（评审 2026-11-18）：同帧原子提取入口（模型序 q/dq 与
        # received_at 出自同一状态帧）。入口不可达时模型序退化为独立
        # get_arm_state 读取并显式降级（frame_aligned=false）——此时
        # 样本不参与误差配对（无可靠帧时间，不伪装已对齐）。
        self._frame_extractor = getattr(arm, "_model_state_from_frame", None)
        if buffer is not None and not callable(self._frame_extractor):
            self._notes.append(
                "无法访问同帧提取入口（arm._model_state_from_frame），模型序"
                "状态退化为独立 get_arm_state 读取（frame_aligned=false，"
                "该类样本不参与误差配对）"
            )

    def _now(self) -> float:
        """共享时钟当前时刻。"""
        return self._clock()

    def _pacing_now(self) -> float:
        """节拍基准时刻。

        注入受控时钟时用该时钟（与采样/发布同基）；否则沿用真实墙钟
        ``time.monotonic``——与注入前的节拍口径完全一致，生产不变。
        """
        if self._clock_override is not None:
            return self._now()
        return time.monotonic()

    def _wait_until(self, deadline: float) -> None:
        """推进到截止时刻（生产=sleep 真实墙钟；测试=受控调度器推进）。

        Args:
            deadline: 目标时刻（与 :meth:`_pacing_now` 同基）。
        """
        if self._wait_override is not None:
            self._wait_override(deadline)
            return
        gap = deadline - self._pacing_now()
        if gap > 0.0:
            time.sleep(gap)

    def fail(self, reason: str) -> None:
        """记录失败原因（verdict 最终为 fail）。"""
        logger.error("诊断失败原因：%s", reason)
        self._reasons.append(reason)

    def abort(self, reason: str) -> None:
        """记录中止（Ctrl+C/外部中断；亦属失败结论）。"""
        self._aborted = True
        self._aborted_reason = reason
        logger.error("诊断中止：%s", reason)

    def _record_status(self, arm: UnitreeR1DInterface) -> None:
        """记录状态快照（phase/fault/stop_confirmed 变化时追加事件）。"""
        status = arm.get_status()
        last = self._status_events[-1] if self._status_events else None
        key = (status.phase, status.fault_reason, status.stop_confirmed)
        if last is None or (
            last["phase"], last["fault_reason"], last["stop_confirmed"]
        ) != key:
            self._status_events.append(
                {
                    "t": self._now(),
                    "phase": status.phase,
                    "fault_reason": status.fault_reason,
                    "stop_confirmed": status.stop_confirmed,
                    "last_publish_at": status.last_publish_at,
                }
            )

    def _sample(self, arm: UnitreeR1DInterface) -> None:
        """采样一拍：缓存原始帧单次读取（35 槽 + 时间元数据 + 模型序同帧）。

        R5（评审 2026-11-18）：模型序 q/dq 与 received_at/sequence 出自
        **同一状态帧**（经 arm._model_state_from_frame 同包契约）——旧实
        现"get_arm_state + buffer.latest"两次独立读取会让 q 属于旧帧而
        时间戳属于新帧，误差配对据此把发布前缓存伪装成发布后反馈。同帧
        入口不可达（跨包调用方）时退化为独立 get_arm_state 读取，样本
        received_at 保持 None（不参与配对，降级不伪装对齐）。

        样本有效性分层（read_only 场景校准可能未核验）：完整帧或模型序
        任一可得即视为有效样本（error=None）；仅记录失败侧的信息性错误
        （model_error/full_error），两侧都失败才置 error。tracking 的
        误差配对仍要求模型序可得（q 非 None 且 received_at 非 None）。
        """
        entry: dict[str, Any] = {
            # t 在状态读取完成后回填（D-M3，评审 2026-11-17）：样本时刻
            # 须反映状态取得时刻，而非采样开始时刻——停止阈值探测以 t
            # 与请求时刻比较，早记会把读取耗时藏进延迟。误差配对已改用
            # 同帧 received_at（R5），t 仅作读取完成时刻留痕。
            "t": None,
            "q": None,
            "dq": None,
            "full_q": None,
            "full_dq": None,
            "received_at": None,
            "source_time": None,
            "sequence": None,
            "mode_machine": None,
            "error": None,
            "model_error": None,
            "full_error": None,
        }
        model_ok = False
        full_ok = False
        if getattr(self, "_full_state_probe", False):
            buffer = getattr(arm, "_buffer", None)
            if buffer is not None:
                try:
                    frame = buffer.latest(
                        now=self._now(), session_id=arm.get_status().session_id
                    )
                    entry["full_q"] = np.array(frame.q, dtype=np.float64, copy=True)
                    entry["full_dq"] = np.array(frame.dq, dtype=np.float64, copy=True)
                    entry["received_at"] = float(frame.received_at)
                    entry["source_time"] = (
                        float(frame.source_time)
                        if frame.source_time is not None
                        else None
                    )
                    entry["sequence"] = frame.sequence
                    entry["mode_machine"] = int(frame.mode_machine)
                    self._full_state_available = True
                    full_ok = True
                    # R5：同帧原子提取模型序（入口在 _bind_arm 解析）
                    extractor = getattr(self, "_frame_extractor", None)
                    if callable(extractor):
                        try:
                            state = np.asarray(extractor(frame), dtype=np.float64)
                            entry["q"] = state[:R1D_ARM_JOINT_COUNT].copy()
                            entry["dq"] = state[R1D_ARM_JOINT_COUNT:].copy()
                            model_ok = True
                            self._frame_aligned = True
                        except Exception as error:  # noqa: BLE001 - 如实记录
                            entry["model_error"] = f"{type(error).__name__}: {error}"
                except Exception as error:  # noqa: BLE001 - 原始帧失败如实记录
                    entry["full_error"] = f"{type(error).__name__}: {error}"
        if not model_ok and entry["model_error"] is None:
            # 同帧提取未发生或不可用：独立公开 API 读取（该样本无可靠
            # 帧时间配对资格，received_at 为 None 时配对自然排除）
            try:
                state = np.asarray(arm.get_arm_state(), dtype=np.float64)
                entry["q"] = state[:R1D_ARM_JOINT_COUNT].copy()
                entry["dq"] = state[R1D_ARM_JOINT_COUNT:].copy()
                model_ok = True
            except Exception as error:  # noqa: BLE001 - 采样失败如实记录
                entry["model_error"] = f"{type(error).__name__}: {error}"
        if not model_ok and not full_ok:
            entry["error"] = (
                f"模型序与完整帧均不可得：{entry['model_error']}；{entry['full_error']}"
            )
            logger.warning("状态采样失败：%s", entry["error"])
        elif not model_ok or not full_ok:
            logger.debug(
                "采样部分降级：model_ok=%s full_ok=%s（%s）",
                model_ok, full_ok,
                entry["model_error"] or entry["full_error"],
            )
        # D-M3：t 回填为状态读取完成时刻（读取失败时同样取当前时刻，
        # 失败样本的 t 不参与阈值判定，仅作序列定位）。
        entry["t"] = self._now()
        self._samples.append(entry)

    def _consume_publishes(self, arm: UnitreeR1DInterface) -> None:
        """周期增量消费真实发布记录（有界 deque，见驱动 R1D_PUBLISH_RECORD_CAPACITY）。"""
        records: tuple[R1DPublishRecord, ...] = arm.publish_records()
        for record in records[self._consumed_records:]:
            self._publishes.append(
                {
                    "seq": len(self._publishes),
                    "session_id": int(record.session_id),
                    "kind": record.kind,
                    "published_at": float(record.published_at),
                    "q_sdk": (
                        np.array(record.q_sdk, dtype=np.float64, copy=True)
                        if record.q_sdk is not None
                        else None
                    ),
                    "mode_machine": int(record.mode_machine),
                    "success": bool(record.success),
                    "error": record.error,
                }
            )
        self._consumed_records = len(records)

    def _send(self, arm: UnitreeR1DInterface, target: np.ndarray, step: int) -> bool:
        """发送一拍目标并记录 send 日志（接受时刻与发布时刻分开）。"""
        accepted_at = self._now()
        code = int(arm.send_joint_command(target))
        self._sends.append(
            {
                "step": int(step),
                "accepted_at": float(accepted_at),
                "target": np.array(target, dtype=np.float64, copy=True),
                "code": code,
            }
        )
        if code != SEND_ACCEPTED:
            self.fail(
                f"send_joint_command 被拒绝（step={step}，返回码 {code}="
                f"{_SEND_CODE_NAMES.get(code, '未知码')}），目标未被本驱动接受"
            )
            return False
        return True

    # -- read_only 执行 --------------------------------------------------

    def _await_first_state(
        self, arm: UnitreeR1DInterface, timeout_wall: float, poll_s: float
    ) -> bool:
        """有界等待首个可用状态（启动竞态吸收；不记录样本）。

        connect 完成到首帧 lowstate 到达之间存在正常竞态，此期间的
        不可用不构成"状态流中断"；等待超时由调用方按数据不足处理。
        可用判定与采样同分层：模型序（get_arm_state，需校准）或完整
        帧（状态缓存）任一可得即通过——read_only 场景校准允许 null。
        """
        buffer = getattr(arm, "_buffer", None)
        while self._pacing_now() < timeout_wall:
            if self._probe_state_usable(arm, buffer):
                return True
            self._wait_until(
                min(timeout_wall, self._pacing_now() + max(0.0005, poll_s))
            )
        return False

    def _probe_state_usable(
        self, arm: UnitreeR1DInterface, buffer: Any
    ) -> bool:
        """探测状态是否可用（模型序或完整帧任一可得）。"""
        try:
            arm.get_arm_state()
            return True
        except Exception:  # noqa: BLE001 - 模型序不可用（可能校准 null）
            pass
        if buffer is None:
            return False
        try:
            buffer.latest(now=self._now(), session_id=arm.get_status().session_id)
            return True
        except Exception:  # noqa: BLE001 - 完整帧亦不可用
            return False

    def run_read_only(self, arm: UnitreeR1DInterface) -> None:
        """执行只读诊断：连接后仅采样状态流，绝不 activate/不 send。"""
        if not arm.connect():
            self.fail("connect 失败（READ_ONLY 门槛未满足或传输层 open 异常）")
            return
        self._record_status(arm)
        period = float(self._config.sample_period_s)
        duration = float(self._config.duration_s)
        n_samples = max(1, int(round(duration / period)))
        wall_start = self._pacing_now()
        # 首帧有界等待（同 duration 上限）：等待期不采样不计中断
        if not self._await_first_state(arm, wall_start + duration, period):
            self.fail("数据不足：等待首帧超时，状态流不可用（全程无有效样本）")
            return
        for index in range(n_samples):
            self._sample(arm)
            self._consume_publishes(arm)
            self._record_status(arm)
            status = arm.get_status()
            if status.phase != "READ_ONLY":
                self.fail(
                    f"READ_ONLY 期间 phase 异常变化为 {status.phase}"
                    f"（fault={status.fault_reason}），只读约束被破坏"
                )
                return
            if self._publishes:
                self.fail(
                    f"READ_ONLY 期间出现 {len(self._publishes)} 条实际发布记录"
                    "（违反只读不变式：publish/acquire 必须全为 0）"
                )
                return
            self._wait_until(wall_start + (index + 1) * period)
        self._consume_publishes(arm)
        # D-M4（评审 2026-11-17）：末次循环检查与本次消费之间存在收尾
        # 窗口——最后一次 sleep 期间/之后到达的发布记录此前漏检，补一次
        # 只读不变式检查（优先级高于样本统计：不变式破坏即 fail）。
        if self._publishes:
            self.fail(
                f"READ_ONLY 收尾窗口出现 {len(self._publishes)} 条实际发布记录"
                "（违反只读不变式：publish/acquire 必须全为 0）"
            )
            return
        ok_count = sum(1 for s in self._samples if s["error"] is None)
        if ok_count == 0:
            self.fail("数据不足：全程无有效状态样本（状态流不可用）")
        elif ok_count < len(self._samples):
            self.fail(
                f"状态流中断：{len(self._samples) - ok_count} 个采样失败"
                f"（状态过期/不可用；有效 {ok_count}/{len(self._samples)}）"
            )
        else:
            self._completed_normally = True

    # -- tracking 执行 ---------------------------------------------------

    def run_tracking(self, arm: UnitreeR1DInterface) -> None:
        """执行单关节低速跟踪诊断：锚定→升余弦波形→回起始→停止流程。"""
        if not arm.connect():
            self.fail("connect 失败（READ_ONLY 门槛未满足或传输层 open 异常）")
            return
        self._record_status(arm)
        # 首帧有界等待：activate 需要新鲜状态，connect 到首帧之间是启动竞态
        assert self._config.send_period_s is not None  # tracking 构造校验保证
        if not self._await_first_state(
            arm, self._pacing_now() + float(self._config.duration_s),
            float(self._config.send_period_s),
        ):
            self.fail("数据不足：等待首帧超时，无法锚定实测姿态（tracking 前置失败）")
            return
        try:
            arm.activate()
        except Exception as error:  # noqa: BLE001 - activate 拒绝=失败结论
            self.fail(f"activate 拒绝（motion 门槛/新鲜状态/停止能力预检未过）: {error}")
            return
        self._record_status(arm)
        try:
            measured = np.asarray(arm.get_arm_state(), dtype=np.float64)
        except Exception as error:  # noqa: BLE001
            self.fail(f"无法读取实测锚定姿态: {error}")
            return
        q_start = measured[:R1D_ARM_JOINT_COUNT].copy()
        period = float(self._config.send_period_s)
        joint = int(self._config.selected_joint)
        amplitude = float(self._config.amplitude_rad)
        frequency = float(self._config.frequency_hz)
        # 波形隐含最大速度（升余弦导数峰值 A·π·f）：超出 dq_limit 会被
        # 驱动步长校验显式拒绝（现场配置责任；此处只作日志提示，不截断）
        implied_max_speed = amplitude * math.pi * frequency
        logger.info(
            "tracking 波形：关节 %d，A=%.6f rad，f=%.6f Hz，隐含峰值速度 %.6f rad/s"
            "（须 ≤ 该关节 dq_limit，超出将被驱动拒绝）",
            joint, amplitude, frequency, implied_max_speed,
        )
        # P1-3（2026-11-18）：离散步数与配置校验/预期跨度同一来源
        # （tracking_step_count），删除本地重复公式避免取整口径漂移。
        n_steps = self._config.tracking_step_count()
        wall_start = self._pacing_now()
        last_target = q_start.copy()
        # R6（评审 2026-11-18）：波形实际通过的最大单拍步长（选中关节，
        # 相邻发送目标之差）——非整周期回程的限速上界。该步长已被驱动
        # 策略层接受，回程复用同一界必然合法（不放宽任何驱动限制）；
        # 亦除得发布层面的节流因子（发送与发布的节拍比例在回程与波形
        # 段一致，发布步长同界）。
        max_wave_step = 0.0
        for step in range(n_steps):
            t_wave = step * period
            target = q_start.copy()
            target[joint] += 0.5 * amplitude * (
                1.0 - math.cos(2.0 * math.pi * frequency * t_wave)
            )
            if not self._send(arm, target, step):
                return
            max_wave_step = max(
                max_wave_step, abs(float(target[joint] - last_target[joint]))
            )
            last_target = target
            self._wait_until(wall_start + (step + 1) * period)
            self._consume_publishes(arm)
            self._sample(arm)
            self._record_status(arm)
            status = arm.get_status()
            if status.phase != "ACTIVE":
                self.fail(
                    f"tracking 期间 phase 离开 ACTIVE（→ {status.phase}，"
                    f"fault={status.fault_reason}）：序列中断"
                )
                return
        # 结束回到起始目标：波形整周期时自然回 0；非整周期 δ(T)≠0 时按
        # R6 受限速回程逐拍逼近起点——一拍跳回的步长可达 |δ(T)|，超出
        # dq_limit·周期会被驱动拒绝并中断诊断（评审 2026-11-18 复现：
        # 半周期 0.05 rad vs 允许 0.004 rad）。
        if not bool(np.allclose(last_target, q_start, rtol=0.0, atol=1e-12)):
            if max_wave_step <= 0.0:
                self.fail(
                    "回程限速推导失败：波形全程零步长但终点偏离起点"
                    "（配置内部不一致，拒绝继续发送）"
                )
                return
            step_index = n_steps
            current = last_target.copy()
            # 有界回程拍数（防御：正常 ≤ ceil(|δ(T)|/步长)，+2 容纳舍入）
            max_return_steps = (
                int(math.ceil(abs(float(q_start[joint] - current[joint])) / max_wave_step)) + 2
            )
            for _ in range(max_return_steps):
                delta = float(q_start[joint] - current[joint])
                current = current.copy()
                if abs(delta) <= max_wave_step:
                    current[joint] = float(q_start[joint])  # 末拍精确回到起点
                else:
                    current[joint] += math.copysign(max_wave_step, delta)
                if not self._send(arm, current, step_index):
                    return
                step_index += 1
                self._wait_until(wall_start + step_index * period)
                self._consume_publishes(arm)
                self._sample(arm)
                self._record_status(arm)
                status = arm.get_status()
                if status.phase != "ACTIVE":
                    self.fail(
                        f"回起始阶段 phase 离开 ACTIVE（→ {status.phase}，"
                        f"fault={status.fault_reason}）：序列中断"
                    )
                    return
                if bool(np.allclose(current, q_start, rtol=0.0, atol=1e-12)):
                    break
            else:
                self.fail(
                    "受限速回程未在有界拍数内回到起点（内部错误，拒绝继续发送）"
                )
                return
            self._wait_until(self._pacing_now() + period)  # 墙钟等待发送线程拾取末拍发布
            self._consume_publishes(arm)
        self._sample(arm)
        self._record_status(arm)
        status = arm.get_status()
        if status.phase != "ACTIVE":
            self.fail(
                f"波形结束回起始阶段 phase 离开 ACTIVE（→ {status.phase}，"
                f"fault={status.fault_reason}）：序列中断"
            )
            return
        self._run_stop_sequence(arm)
        self._completed_normally = True

    def _run_stop_sequence(self, arm: UnitreeR1DInterface) -> None:
        """执行停止流程并区分软件检测时刻与反馈达阈值时刻。"""
        t_request = self._now()
        self._stop_info["requested"] = True
        self._stop_info["requested_at_s"] = float(t_request)
        self._stop_info["source"] = "diagnostic"
        stop_error: str | None = None
        try:
            arm.slow_stop()
        except Exception as error:  # noqa: BLE001 - 停止异常=失败结论
            stop_error = f"{type(error).__name__}: {error}"
            self.fail(f"slow_stop 异常: {stop_error}")
        t_return = self._now()
        self._stop_info["software_detected_at_s"] = float(t_return)
        self._consume_publishes(arm)
        self._record_status(arm)
        status = arm.get_status()
        self._stop_info["stop_confirmed"] = bool(status.stop_confirmed)
        self._stop_info["final_phase"] = status.phase
        if status.phase != "READ_ONLY" or not status.stop_confirmed:
            self.fail(
                f"停止未确认/未回 READ_ONLY（phase={status.phase}，"
                f"stop_confirmed={status.stop_confirmed}，"
                f"fault={status.fault_reason}）——未确认停止不得宣称通过"
            )
        # 反馈速度达阈值时刻：停止请求后第一个 |dq| ≤ 阈值的有效样本
        threshold = self._config.stop_confirm_dq_threshold
        if threshold is not None:
            assert self._config.send_period_s is not None  # tracking 构造校验保证
            for _ in range(_STOP_PROBE_SAMPLES):
                self._sample(arm)
                sample = self._samples[-1]
                if (
                    sample["error"] is None
                    and sample["dq"] is not None
                    and float(sample["t"]) > float(t_request)
                    and float(np.max(np.abs(sample["dq"]))) <= float(threshold)
                ):
                    self._stop_info["feedback_below_threshold_at_s"] = float(
                        sample["t"]
                    )
                    break
                self._wait_until(self._pacing_now() + float(self._config.send_period_s))
        if self._stop_info["feedback_below_threshold_at_s"] is None:
            self.fail(
                "停止后探测窗口内未观察到反馈速度达阈值的新鲜样本"
                "（反馈侧停止延迟不可得，不编造）"
            )

    # -- 清理与收尾 ------------------------------------------------------

    def cleanup(self, arm: UnitreeR1DInterface) -> None:
        """清理资源：运行态先尽力停止，再 disconnect（幂等）。

        捕获 BaseException（含 KeyboardInterrupt）：Ctrl+C 中止后清理
        仍须完成（disconnect 释放通道），清理失败如实记录但不掩盖
        已记录的中止原因。
        """
        status = None
        try:
            status = arm.get_status()
        except BaseException as error:  # noqa: BLE001 - 清理路径尽力+留痕
            self._reasons.append(f"清理阶段读取状态失败: {type(error).__name__}: {error}")
        if status is not None and status.phase in ("ACTIVE", "STOPPING"):
            # D-M5（评审 2026-11-17）：清理路径的停止同样写入停止遥测
            # （requested/请求时刻/来源标记），否则异常中止的摘要里停止
            # 信息缺失，无法区分"没请求停止"与"请求了但未入库"。
            self._stop_info["requested"] = True
            self._stop_info["requested_at_s"] = float(self._now())
            self._stop_info["source"] = "cleanup"
            try:
                arm.slow_stop()
            except BaseException as error:  # noqa: BLE001 - 清理尽力，如实记录
                self._reasons.append(
                    f"清理阶段 slow_stop 异常: {type(error).__name__}: {error}"
                )
        try:
            arm.disconnect()
        except BaseException as error:  # noqa: BLE001
            self._reasons.append(
                f"清理阶段 disconnect 异常: {type(error).__name__}: {error}"
            )

    # -- 指标与产物 ------------------------------------------------------

    def _publish_interval_metrics(self) -> dict[str, Any]:
        """成功 active 发布的间隔指标（R9 时刻）。"""
        ats = [
            row["published_at"]
            for row in self._publishes
            if row["success"] and row["kind"] == "active"
        ]
        return compute_publish_interval_metrics(ats, self._config.publish_deadline_s)

    def _tracking_error_metrics(self) -> dict[str, Any]:
        """按对齐规则配对计算各关节 RMS/max 误差（规则见模块 docstring）。"""
        signs = np.asarray(self._config.joint_signs, dtype=np.float64)
        offsets = np.asarray(self._config.joint_offsets_rad, dtype=np.float64)
        publish_rows: list[tuple[float, np.ndarray | None]] = []
        for row in self._publishes:
            if not (row["success"] and row["kind"] == "active"):
                continue
            if row["q_sdk"] is None:
                publish_rows.append((float(row["published_at"]), None))
                continue
            # 目标 SDK 数值 → 模型序（q_model=(q_sdk−offset)·sign）
            publish_rows.append(
                (
                    float(row["published_at"]),
                    (row["q_sdk"] - offsets) * signs,
                )
            )
        # R5（评审 2026-11-18）：配对键 = 帧接收时刻（与模型序 q 同帧原子
        # 取得），非采样读取完成时刻——发布前到达、尚未超时的旧缓存不再
        # 被计为发布后反馈；帧身份 (received_at, sequence) 保证同一物理帧
        # 的重复采样只作一条命令的配对证据（评审 2026-11-18 复审：样本行
        # 逐条配对会把一次缓存帧计为多条命令已匹配，夸大覆盖率）。
        sample_rows = sorted(
            (
                (s["received_at"], s["sequence"], s["q"])
                for s in self._samples
                if s["error"] is None
                and s["q"] is not None
                and s["received_at"] is not None
            ),
            key=lambda item: float(item[0]),
        )
        pairs, no_match = _pair_feedback_frames(publish_rows, sample_rows)
        metrics = compute_tracking_error_metrics(pairs)
        metrics["no_match_samples"] = no_match
        metrics["alignment_rule"] = (
            "每条成功发布的 active 记录（published_at、q_sdk=SDK 侧数值按模型序排列）"
            "与帧接收时刻（received_at，与模型序状态同帧原子取得）严格晚于 "
            "published_at 的最早未消费状态样本配对；目标换算到模型序 "
            "q_target=(q_sdk−offset)·sign（换算参数来自现场校准节）；误差 = "
            "q_fb_model − q_target（模型序 rad）。同一物理反馈帧（身份 = "
            "received_at + sequence）只作一条命令的反馈证据——重复采样到的"
            "同一缓存帧不为后续命令重复作证；received_at ≤ 发布时刻的发布前"
            "缓存与无可靠帧时间的样本均不参与配对。无匹配样本记录计入 "
            "no_match_samples，不参与误差统计；无配对样本时各关节误差 "
            "unavailable（不记 0）。"
        )
        return metrics

    def _stop_metrics(self) -> dict[str, Any]:
        """停止指标：软件检测延迟与反馈达阈值延迟分开（口径自描述）。

        双口径（D-M2，评审 2026-11-17；口径文案勘误见评审 2026-11-18 §3.2）：

        - ``latency_software_s``（软件侧/观察上界口径）：停止请求 →
          ``slow_stop`` 返回。驱动缓停内部会等待"停止序列发布完成 +
          发布后低速反馈确认"，因此该值**包含反馈确认等待**（机械减速
          的软件侧观察）与退权/调用开销——它是软件观察上界，不是纯
          计划执行时长，更不得解读为精确机械减速时间；
        - ``latency_feedback_s``（反馈侧/实测口径）：停止请求 → 首个
          采样时刻晚于请求且 |dq|≤阈值的新鲜反馈样本，含机械实际减速
          响应与采样周期量化，是现场验收应引用的实测停止延迟。
        """
        info = self._stop_info
        latency_software: float | None = None
        latency_feedback: float | None = None
        if info["requested_at_s"] is not None and info["software_detected_at_s"] is not None:
            latency_software = float(info["software_detected_at_s"]) - float(
                info["requested_at_s"]
            )
        if (
            info["requested_at_s"] is not None
            and info["feedback_below_threshold_at_s"] is not None
        ):
            latency_feedback = float(info["feedback_below_threshold_at_s"]) - float(
                info["requested_at_s"]
            )
        return {
            "requested": bool(info["requested"]),
            "requested_at_s": info["requested_at_s"],
            "software_detected_at_s": info["software_detected_at_s"],
            "feedback_below_threshold_at_s": info["feedback_below_threshold_at_s"],
            "stop_confirmed": info["stop_confirmed"],
            "final_phase": info["final_phase"],
            "source": info["source"],
            "latency_software_s": latency_software,
            "latency_feedback_s": latency_feedback,
            # 口径注明随遥测落盘，读数方无需回查源码（D-M2；文案勘误
            # 评审 2026-11-18 §3.2：slow_stop 内部等待反馈确认）
            "latency_software_scope": (
                "软件侧/观察上界口径：停止请求→slow_stop 返回"
                "（含有限停止序列发布、发布后低速反馈确认等待与退权/调用"
                "开销；是软件观察上界，不是纯计划执行时长，亦非精确机械"
                "减速时间）"
            ),
            "latency_feedback_scope": (
                "反馈侧/实测口径：停止请求→首个晚于请求且|dq|≤阈值的新鲜"
                "反馈样本（含机械减速响应与采样量化，现场验收引用此值）"
            ),
            "threshold_rad_s": self._config.stop_confirm_dq_threshold,
        }

    def _apply_threshold_verdicts(
        self, publish: dict[str, Any], tracking: dict[str, Any], stop: dict[str, Any]
    ) -> None:
        """按现场阈值把超限写入失败原因（无样本/未冻结不判 0 通过）。"""
        if (
            not publish["unavailable"]
            and publish["deadline_overruns"] is not None
            and int(publish["deadline_overruns"]) > 0
        ):
            self.fail(
                f"publish 间隔超期 {int(publish['deadline_overruns'])} 次"
                f"（判据 > {1.0 + _PUBLISH_INTERVAL_OVERRUN_RATIO:.1f}×期限"
                f"={publish['deadline_s']}s，max={publish['max_s']:.6f}s）"
            )
        if self._config.mode == "tracking":
            # P1-3（2026-11-18）：tracking 验收同时要求发布间隔证据与实际
            # 运动证据——没有成功发布、或发布目标不随波形运动时，"跟踪
            # 误差小"可能只是目标从未离开起点，不能据此判通过。
            if publish["unavailable"]:
                self.fail(
                    "发布间隔证据不足：成功 active 发布记录不足 2 条，"
                    "无法判定发布节奏——不足以判定跟踪验收通过"
                )
            selected_joint = self._config.selected_joint
            if selected_joint is None:  # pragma: no cover - 构造校验保证
                self.fail(
                    "selected_joint 缺失：tracking 判定要求选中关节"
                    "（构造校验未生效，无法比较实际发布目标跨度）"
                )
                return
            actual_targets: list[float] = []
            for row in self._publishes:
                if row["kind"] != "active" or not row["success"]:
                    continue
                q_sdk = row["q_sdk"]
                if q_sdk is None:
                    continue
                try:
                    value = float(q_sdk[selected_joint])
                except (TypeError, ValueError, IndexError):
                    continue
                if math.isfinite(value):
                    actual_targets.append(value)
            if len(actual_targets) < 2:
                self.fail(
                    "实际发布目标证据不足：有效 active 发布目标仅 "
                    f"{len(actual_targets)} 条（至少需要 2 条成功且 "
                    "selected_joint 目标有限的 active 记录才能比较实际跨度）"
                    "——不足以判定跟踪验收通过"
                )
            else:
                actual_span = max(actual_targets) - min(actual_targets)
                expected_span = self._config.expected_target_span_rad()
                if actual_span < 0.5 * expected_span:
                    self.fail(
                        "实际发布的 SDK 侧目标（按模型序排列）未覆盖波形："
                        f"actual={actual_span:.6f} rad < "
                        f"0.5×expected={0.5 * expected_span:.6f} rad"
                        f"（expected={expected_span:.6f} rad）"
                        "——目标未随波形运动，不足以判定跟踪验收通过"
                    )
            # R4（评审 2026-11-18）：tracking 是验收模式——验收阈值缺失
            # 时不得输出 pass。缺任一阈值即 fail 并显式区分"仅完成采集"
            # 与"验收通过"：无冻结判据的运行结果不能被当作 G5 通过。
            missing = [
                name
                for name, value in (
                    ("tracking_error_limit_rad", self._config.tracking_error_limit_rad),
                    ("stop_latency_limit_s", self._config.stop_latency_limit_s),
                )
                if value is None
            ]
            if missing:
                self.fail(
                    f"验收阈值缺失（{', '.join(missing)}）：tracking 验收要求"
                    "完整有限阈值——本次仅完成数据采集，不构成验收通过"
                    "（G4 冻结阈值后重跑方可判定）"
                )
            if tracking["unavailable"]:
                self.fail(
                    "跟踪误差不可得（无配对样本）：无样本不能得出零误差通过——"
                    f"{tracking['unavailable_reason']}"
                )
            elif self._config.tracking_error_limit_rad is not None:
                worst = max(
                    item["max_rad"] for item in tracking["per_joint"]
                )
                if worst > float(self._config.tracking_error_limit_rad):
                    self.fail(
                        f"跟踪误差超阈值：max={worst:.6f} rad > "
                        f"tracking_error_limit_rad={self._config.tracking_error_limit_rad}"
                    )
            if (
                self._config.stop_latency_limit_s is not None
                and stop["latency_feedback_s"] is not None
                and float(stop["latency_feedback_s"])
                > float(self._config.stop_latency_limit_s)
            ):
                self.fail(
                    f"停止延迟超阈值：反馈侧 {stop['latency_feedback_s']:.6f}s > "
                    f"stop_latency_limit_s={self._config.stop_latency_limit_s}"
                )

    def _config_echo(self) -> dict[str, Any]:
        """配置回显（数值来源可追溯；未知项 null）。"""
        return {
            "mode": self._config.mode,
            "duration_s": float(self._config.duration_s),
            "sample_period_s": (
                float(self._config.sample_period_s)
                if self._config.sample_period_s is not None
                else None
            ),
            "send_period_s": (
                float(self._config.send_period_s)
                if self._config.send_period_s is not None
                else None
            ),
            "selected_joint": self._config.selected_joint,
            "amplitude_rad": (
                float(self._config.amplitude_rad)
                if self._config.amplitude_rad is not None
                else None
            ),
            "frequency_hz": (
                float(self._config.frequency_hz)
                if self._config.frequency_hz is not None
                else None
            ),
            "stop_confirm_dq_threshold": self._config.stop_confirm_dq_threshold,
            "publish_deadline_s": self._config.publish_deadline_s,
            "tracking_error_limit_rad": self._config.tracking_error_limit_rad,
            "stop_latency_limit_s": self._config.stop_latency_limit_s,
            "joint_signs": list(self._config.joint_signs)
            if self._config.joint_signs is not None
            else None,
            "joint_offsets_rad": list(self._config.joint_offsets_rad)
            if self._config.joint_offsets_rad is not None
            else None,
        }

    def _write_artifacts(self) -> dict[str, str]:
        """写 CSV + NPZ 原始序列产物，返回相对文件名。"""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        def _fmt(value: Any) -> str:
            """CSV 浮点全精度、None/数组安全格式化。"""
            if value is None:
                return ""
            if isinstance(value, float):
                return repr(value)
            return str(value)

        # 1) 发布记录 CSV（R9：published_at = publish 调用返回点）
        publish_csv = "diagnostic_publish_records.csv"
        with (self._output_dir / publish_csv).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            header = [
                "seq", "session_id", "kind", "published_at", "mode_machine",
                "success", "error",
            ] + [f"q_sdk_{i}" for i in range(R1D_ARM_JOINT_COUNT)]
            writer.writerow(header)
            for row in self._publishes:
                q_cols = (
                    [repr(float(v)) for v in row["q_sdk"]]
                    if row["q_sdk"] is not None
                    else [""] * R1D_ARM_JOINT_COUNT
                )
                writer.writerow(
                    [
                        row["seq"], row["session_id"], row["kind"],
                        repr(row["published_at"]), row["mode_machine"],
                        int(row["success"]), row["error"] or "",
                    ]
                    + q_cols
                )

        # 2) 状态采样 CSV（模型序 + 完整 35 槽 + 接收/源时间）
        samples_csv = "diagnostic_samples.csv"
        full_count = 35
        with (self._output_dir / samples_csv).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            header = (
                ["index", "t", "error", "model_error", "full_error",
                 "received_at", "source_time", "sequence", "mode_machine"]
                + [f"q_model_{i}" for i in range(R1D_ARM_JOINT_COUNT)]
                + [f"dq_model_{i}" for i in range(R1D_ARM_JOINT_COUNT)]
                + [f"full_q_{i}" for i in range(full_count)]
                + [f"full_dq_{i}" for i in range(full_count)]
            )
            writer.writerow(header)
            for index, s in enumerate(self._samples):
                def _vec(arr: np.ndarray | None, n: int) -> list[str]:
                    """数组列格式化（None → 空列）。"""
                    if arr is None:
                        return [""] * n
                    return [repr(float(v)) for v in np.asarray(arr)[:n]]

                writer.writerow(
                    [
                        index, repr(float(s["t"])), s["error"] or "",
                        s["model_error"] or "", s["full_error"] or "",
                        _fmt(s["received_at"]), _fmt(s["source_time"]),
                        _fmt(s["sequence"]), _fmt(s["mode_machine"]),
                    ]
                    + _vec(s["q"], R1D_ARM_JOINT_COUNT)
                    + _vec(s["dq"], R1D_ARM_JOINT_COUNT)
                    + _vec(s["full_q"], full_count)
                    + _vec(s["full_dq"], full_count)
                )

        # 3) send 日志 CSV（目标接受时间——与发布时间分开记录）
        send_csv = "diagnostic_send_log.csv"
        with (self._output_dir / send_csv).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["step", "accepted_at", "code", "code_name"]
                + [f"target_model_{i}" for i in range(R1D_ARM_JOINT_COUNT)]
            )
            for row in self._sends:
                writer.writerow(
                    [
                        row["step"], repr(row["accepted_at"]), row["code"],
                        _SEND_CODE_NAMES.get(row["code"], "UNKNOWN"),
                    ]
                    + [repr(float(v)) for v in row["target"]]
                )

        # 4) NPZ 原始序列（None 用哨兵，规则见 JSON npz_notes）
        npz_name = "diagnostic_raw.npz"
        n_sample = len(self._samples)
        n_pub = len(self._publishes)

        def _stack_or_nan(entries: list[Any], width: int) -> np.ndarray:
            """按行堆叠数组列；None 行填 NaN（如校准 null 时模型序不可得）。"""
            rows = np.full((len(entries), width), np.nan)
            for i, value in enumerate(entries):
                if value is not None:
                    rows[i] = np.asarray(value, dtype=np.float64)[:width]
            return rows

        send_targets = _stack_or_nan([r["target"] for r in self._sends], R1D_ARM_JOINT_COUNT)
        sample_q = _stack_or_nan([s["q"] for s in self._samples], R1D_ARM_JOINT_COUNT)
        sample_dq = _stack_or_nan([s["dq"] for s in self._samples], R1D_ARM_JOINT_COUNT)
        sample_full_q = np.full((n_sample, full_count), np.nan)
        sample_full_dq = np.full((n_sample, full_count), np.nan)
        sample_received = np.full(n_sample, np.nan)
        sample_source = np.full(n_sample, np.nan)
        sample_sequence = np.full(n_sample, _NPZ_INT_NONE, dtype=np.int64)
        for i, s in enumerate(self._samples):
            if s["full_q"] is not None:
                sample_full_q[i] = s["full_q"]
            if s["full_dq"] is not None:
                sample_full_dq[i] = s["full_dq"]
            if s["received_at"] is not None:
                sample_received[i] = s["received_at"]
            if s["source_time"] is not None:
                sample_source[i] = s["source_time"]
            if s["sequence"] is not None:
                sample_sequence[i] = int(s["sequence"])
        pub_q_sdk = np.full((n_pub, R1D_ARM_JOINT_COUNT), np.nan)
        for i, row in enumerate(self._publishes):
            if row["q_sdk"] is not None:
                pub_q_sdk[i] = row["q_sdk"]
        np.savez(
            self._output_dir / npz_name,
            send_step=np.asarray([r["step"] for r in self._sends], dtype=np.int64),
            send_accepted_at=np.asarray(
                [r["accepted_at"] for r in self._sends], dtype=np.float64
            ),
            send_code=np.asarray([r["code"] for r in self._sends], dtype=np.int64),
            send_target_model=send_targets,
            sample_index=np.arange(n_sample, dtype=np.int64),
            sample_t=np.asarray([float(s["t"]) for s in self._samples], dtype=np.float64),
            sample_q_model=sample_q,
            sample_dq_model=sample_dq,
            sample_full_q=sample_full_q,
            sample_full_dq=sample_full_dq,
            sample_received_at=sample_received,
            sample_source_time=sample_source,
            sample_sequence=sample_sequence,
            pub_seq=np.asarray([r["seq"] for r in self._publishes], dtype=np.int64),
            pub_session_id=np.asarray(
                [r["session_id"] for r in self._publishes], dtype=np.int64
            ),
            pub_kind_code=np.asarray(
                [_PUBLISH_KIND_CODES.get(r["kind"], -1) for r in self._publishes],
                dtype=np.int64,
            ),
            pub_published_at=np.asarray(
                [r["published_at"] for r in self._publishes], dtype=np.float64
            ),
            pub_mode_machine=np.asarray(
                [r["mode_machine"] for r in self._publishes], dtype=np.int64
            ),
            pub_success=np.asarray(
                [r["success"] for r in self._publishes], dtype=np.int64
            ),
            pub_q_sdk_model_order=pub_q_sdk,
        )
        return {
            "publish_csv": publish_csv,
            "samples_csv": samples_csv,
            "send_csv": send_csv,
            "npz": npz_name,
        }

    def finalize(self) -> dict[str, Any]:
        """汇总指标、判定 verdict 并写全部产物；返回 JSON 摘要 dict。"""
        publish_metrics = self._publish_interval_metrics()
        tracking_metrics: dict[str, Any] | None = None
        if self._config.mode == "tracking":
            tracking_metrics = self._tracking_error_metrics()
        stop_metrics = self._stop_metrics()
        self._apply_threshold_verdicts(publish_metrics, tracking_metrics, stop_metrics)

        if self._aborted:
            verdict = "aborted"
        elif self._reasons:
            verdict = "fail"
        elif self._completed_normally:
            verdict = "pass"
        else:
            verdict = "fail"
            self._reasons.append("内部错误：流程未正常完成也无失败原因，按失败处理")

        git_sha = _git_commit_sha()
        if git_sha is None:
            self._git_sha_error = "git rev-parse HEAD 获取失败（记 null，不编造）"

        reasons = list(self._reasons)
        if self._aborted_reason is not None:
            reasons.insert(0, f"中止：{self._aborted_reason}")

        summary: dict[str, Any] = {
            "schema": f"r1d_diagnostic_v{R1D_DIAGNOSTIC_SCHEMA_VERSION}",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "reasons": reasons,
            "mode": self._config.mode,
            "config": self._config_echo(),
            "provenance": {
                "sdk_version": R1D_SDK_IDENTITY,
                "code_sha": git_sha,
                "code_sha_error": self._git_sha_error,
                "hardware_config": {
                    "path": self._config.hardware_config_path,
                    "sha256": self._config.hardware_config_sha256,
                },
                "side": self._config.side,
                "units": {
                    "position": "rad",
                    "velocity": "rad/s",
                    "time": "s（驱动注入单调时钟；sample/send 调用点与 publish 调用返回点同基）",
                    "tracking_error": "rad（模型序）",
                },
                "shared_clock": self._shared_clock,
                "full_state_available": self._full_state_available,
                # R5：模型序状态与帧时间是否同帧原子（false 时误差配对
                # 仅用有 received_at 的样本，降级不伪装对齐）
                "frame_aligned": self._frame_aligned,
                "notes": list(self._notes),
            },
            "session": {
                "status_events": self._status_events,
                "final_phase": (
                    self._status_events[-1]["phase"]
                    if self._status_events
                    else None
                ),
            },
            "counts": {
                "samples": len(self._samples),
                "samples_ok": sum(
                    1 for s in self._samples if s["error"] is None
                ),
                "send_calls": len(self._sends),
                "send_accepted": sum(
                    1 for r in self._sends if r["code"] == SEND_ACCEPTED
                ),
                "publish_records_total": len(self._publishes),
                "publish_active_success": sum(
                    1 for r in self._publishes
                    if r["kind"] == "active" and r["success"]
                ),
                "publish_stop": sum(
                    1 for r in self._publishes if r["kind"] == "stop"
                ),
            },
            "publish_metrics": publish_metrics,
            "tracking_metrics": tracking_metrics,
            "stop": stop_metrics,
            "npz_notes": {
                "pub_kind_code": "0=active, 1=stop, -1=未知",
                "sample_sequence": f"{_NPZ_INT_NONE}=None（协议无序号支持，不编造）",
                "sample_source_time": "NaN=None（无源时间支持）",
                "sample_full_q/full_dq": "NaN=该采样无完整帧（缓存不可用）",
                "pub_q_sdk_model_order": "NaN=该记录 q_sdk 提取失败；列为 SDK 侧数值按模型序排列",
            },
        }
        try:
            artifacts = self._write_artifacts()
            summary["artifacts"] = artifacts
            summary_json = "diagnostic_summary.json"
            with (self._output_dir / summary_json).open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
        except Exception as error:  # noqa: BLE001 - 产物写盘失败如实暴露
            summary["artifacts"] = None
            summary["artifact_write_error"] = f"{type(error).__name__}: {error}"
            if summary["verdict"] == "pass":
                summary["verdict"] = "fail"
                summary["reasons"].append(
                    f"产物写盘失败（pass 结论降级为 fail）: {error}"
                )
        return summary


def run_r1d_diagnostic(
    arm: UnitreeR1DInterface,
    config: R1DDiagnosticConfig,
    output_dir: Path,
    *,
    clock: Callable[[], float] | None = None,
    wait_until: Callable[[float], None] | None = None,
) -> dict[str, object]:
    """执行一次 R1-A7-D 只读/低速诊断并产出全部产物。

    Args:
        arm: 已构造（phase=CLOSED）的驱动实例；建议经
            ``src.real.r1d_factory.build_r1d_arm`` 构造并注入与诊断共享的
            单调时钟（fake 测试注入 fake clock）。
        config: 现场诊断配置（数值来源见 R1DDiagnosticConfig）。
        output_dir: 产物输出目录（不存在则创建；写入 JSON 摘要 +
            3 份 CSV + NPZ）。
        clock: 可选受控时钟（测试注入；缺省沿用驱动时钟）。
        wait_until: 可选受控等待（测试注入；缺省按真实墙钟 sleep）。
            注入后命令→发布→反馈→采样由测试调度器确定性推进。

    Returns:
        JSON 摘要 dict（与写盘的 diagnostic_summary.json 同构）：
        verdict（pass/fail/aborted）、reasons、指标与 provenance。

    Note:
        - read_only：不 activate、不 send——全程 publish/acquire 必须为 0，
          出现任何实际发布记录立即判失败；
        - tracking：从实测姿态锚定开始，仅选中关节按升余弦波形运动，
          其余关节目标恒为锚定值（全机非目标槽位保持由 4.3 冻结策略经
          驱动 build 路径承担，本层不另造默认值）；结束回到起始目标并
          走完整停止流程（未确认即失败结论）；
        - 失败路径（序列中断/数据不足/停止未确认/Ctrl+C/异常）一律生成
          失败结论并清理资源（停止尝试 + disconnect）。
    """
    collector = _DiagnosticCollector(
        config, output_dir, clock=clock, wait_until=wait_until
    )
    collector._bind_arm(arm)
    try:
        if config.mode == "read_only":
            collector.run_read_only(arm)
        else:
            collector.run_tracking(arm)
    except KeyboardInterrupt:
        collector.abort("用户中断（Ctrl+C）")
    except Exception as error:  # noqa: BLE001 - 异常路径=失败结论+清理
        collector.fail(f"诊断过程异常：{type(error).__name__}: {error}")
        logger.exception("诊断过程异常（失败结论并清理）")
    finally:
        collector.cleanup(arm)
    return collector.finalize()
