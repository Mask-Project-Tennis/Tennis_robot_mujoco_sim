"""R1-A7-D 独立驱动构造工厂 — build_r1d_arm（阶段 4 Task 4.7）。

与 RM65 既有工厂（`src/real/runner_factory.py::build_robot_arm`）分立：
R1D 不复用 RM65 的能力校验与默认语义，也不把 FakeRobot 的六维行为用于
R1D（主代理裁定 7）；RM65 入口行为零改动。`run_real_robot.py` 仍拒绝
R1D robot-config，不经本工厂启动 MPC/异步规划/SDK（阶段 5 才接线）。

构造顺序（brief 复选框 1 / 裁定 4，失败先于传输创建）：

1. 纯数据校验（purpose="offline" 档）：错误 profile（robot_id 不符）/
   错误 side / 硬件配置与 profile 关节名不一致在传输创建前失败。
   read_only/motion 门槛不在构造期重复——connect/activate 内部自会按
   阶梯校验（4.6），构造期只保证"描述与配置互相匹配"；
2. 传输：注入 transport（测试/离线诊断）直接使用，零 SDK import；
   未注入时经 `SdkR1DTransport.from_hardware_config` 真实构造（触发 SDK
   懒加载，4.5 边界），`acquire_timeout_s` 必填显式透传——禁止静默依赖
   传输层 10.0 工程默认（裁定 1），实机数值冻结时点在 G4 现场配置；
3. 驱动：`UnitreeR1DInterface(config, profile, transport, clock=clock, ...)`
   构造零传输副作用（不 connect / acquire / publish）。

共享时钟（裁定 2 / 4.6 评审 M-3）：同一 clock 可调用传给传输工厂与驱动
两处——停止确认判据跨时基比较（发布时刻 vs 反馈接收时刻）要求同一时基。

`command_timeout_s ≥ 2×control_period_s` 下界校验已在 4.6 驱动构造器
实现，本工厂只透传不重复实现（裁定 3）。
"""

from __future__ import annotations

import time
from typing import Callable, Mapping

from src.real.r1d_command_policy import R1DSlotHold
from src.real.r1d_config import R1DHardwareConfig, validate_r1d_hardware_config
from src.real.r1d_transport import SdkR1DTransport
from src.real.r1d_types import R1DTransport
from src.real.unitree_r1d_interface import UnitreeR1DInterface
from src.robot.robot_profile import RobotProfile

__all__ = ["build_r1d_arm", "R1D_FACTORY_DEFAULT_JOIN_TIMEOUT_S"]

# join 超时默认值：与 unitree_r1d_interface 驱动默认（_DEFAULT_JOIN_TIMEOUT_S）
# 对齐的软件工程参数（非实机控制参数）；模块级常量避免跨模块 import 私有成员。
R1D_FACTORY_DEFAULT_JOIN_TIMEOUT_S = 5.0


def build_r1d_arm(
    profile: RobotProfile,
    config: R1DHardwareConfig,
    *,
    acquire_timeout_s: float,
    transport: R1DTransport | None = None,
    clock: Callable[[], float] = time.monotonic,
    slot_policies: Mapping[int, R1DSlotHold] | None = None,
    expected_mode_machine: int | None = None,
    command_timeout_s: float | None = None,
    stop_sequence_frames: int = 1,
    stop_confirm_dq_threshold: float | None = None,
    join_timeout_s: float = R1D_FACTORY_DEFAULT_JOIN_TIMEOUT_S,
    switcher_timeout_s: float | None = None,
    release_retry_interval_s: float | None = None,
    state_queue_len: int | None = None,
) -> UnitreeR1DInterface:
    """构造 R1-A7-D 单臂驱动（只构造对象，不 connect / acquire / publish）。

    Args:
        profile: R1D 单侧 7 关节机器人描述（如 configs/robots/r1d_left.yaml）。
        config: 已加载的 R1D 硬件配置（4.2 冻结 schema）。
        acquire_timeout_s: 获取整机控制权整体超时（秒）。必填显式参数，
            真实传输路径原样透传给 from_hardware_config——实机数值冻结
            时点在 G4 现场配置，本工厂不硬编码；注入 transport 时不被
            消费（fake transport 自行定义语义），仍要求显式给出。
        transport: 传输层注入（测试 fake / 离线诊断）；None 时经
            from_hardware_config 真实构造（触发 SDK 懒加载）。
        clock: 单调时钟注入（默认 time.monotonic）。同一实例传给传输
            工厂与驱动两处（停止确认判据要求同一时基，裁定 2）。
        slot_policies: 全机非目标槽位显式保持策略（G4 现场确认后接线；
            None 时 activate 停止能力预检拒绝接管，语义归 4.6）。
        expected_mode_machine: 冻结期望机型值；None 表示未冻结（仅变化检测）。
        command_timeout_s: 目标过期看门狗（秒）；None 禁用并记录（R11）。
            下界校验（≥ 2×control_period_s）由驱动构造器承担。
        stop_sequence_frames: 有限停止序列帧数（≥1）。
        stop_confirm_dq_threshold: 停止反馈确认速度阈值（rad/s）；None 禁用。
        join_timeout_s: 发送线程 join 上限（真实秒，软件工程参数）。
        switcher_timeout_s: MotionSwitcher RPC 单次超时（秒）。None 沿用
            传输层默认（manifest 示例事实 5.0）；G4 现场冻结值不同时经
            此参数显式注入（A-M1，评审 2026-11-17：实机面向超时改为
            注入通路，不再依赖硬编码示例默认）。
        release_retry_interval_s: ReleaseMode 后重查间隔（秒）。None 沿用
            传输层默认（manifest 示例事实 1.0），语义同上。
        state_queue_len: 订阅回调队列深度。None 沿用传输层默认（官方
            示例 10），语义同上。

    Returns:
        未连接（phase=CLOSED）的 UnitreeR1DInterface，实现
        RobotArmInterface 位置目标语义。

    Raises:
        ValueError: profile 与硬件配置不匹配（robot_id 不符 / side 与
            关节名冲突 / joint_slots 与 profile 名称不一致等 offline 级
            错误，先于传输创建），或注入 transport 不满足 R1DTransport
            Protocol。
    """
    # 1) 纯数据校验（offline 档；任何失败先于传输创建，brief 复选框 1）
    validate_r1d_hardware_config(config, profile, purpose="offline")

    # 2) 传输：注入优先（零 SDK import）；真实路径显式透传超时与共享时钟
    transport_obj: R1DTransport
    if transport is None:
        # 可选超时参数：None=不传（沿用传输层 manifest 示例默认），
        # 显式给出才进入 from_hardware_config（A-M1 注入通路）
        transport_kwargs: dict[str, object] = {
            "acquire_timeout_s": acquire_timeout_s,
            "clock": clock,
        }
        if switcher_timeout_s is not None:
            transport_kwargs["switcher_timeout_s"] = switcher_timeout_s
        if release_retry_interval_s is not None:
            transport_kwargs["release_retry_interval_s"] = release_retry_interval_s
        if state_queue_len is not None:
            transport_kwargs["state_queue_len"] = state_queue_len
        transport_obj = SdkR1DTransport.from_hardware_config(
            config,
            **transport_kwargs,  # type: ignore[arg-type]
        )
    else:
        if not isinstance(transport, R1DTransport):
            raise ValueError(
                "参数 transport 必须实现 R1DTransport Protocol"
                f"（open/acquire_control/publish/release_control/close），"
                f"实际 {type(transport).__name__}"
            )
        transport_obj = transport

    # 3) 驱动（构造零传输副作用；clock 与传输同一时基）
    return UnitreeR1DInterface(
        config,
        profile,
        transport_obj,
        clock=clock,
        slot_policies=slot_policies,
        expected_mode_machine=expected_mode_machine,
        command_timeout_s=command_timeout_s,
        stop_sequence_frames=stop_sequence_frames,
        stop_confirm_dq_threshold=stop_confirm_dq_threshold,
        join_timeout_s=join_timeout_s,
    )
