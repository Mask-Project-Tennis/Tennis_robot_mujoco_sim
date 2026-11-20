"""R1-A7-D 只读/低速诊断 CLI 工具 — 阶段 4 Task 4.8。

诊断入口：加载机器人描述与硬件配置 → 经 ``build_r1d_arm`` 构造驱动
（真实路径触发 SDK 懒加载，见 4.5/4.7）→ 执行
``src.real.r1d_diagnostics.run_r1d_diagnostic`` → 输出 JSON 摘要 +
CSV/NPZ 原始序列到 ``--output-dir``。

参数纪律（主代理裁定 1）：

- ``--robot-config/--hardware-config/--output-dir/--acquire-timeout-s``
  全部强制显式（数值来自现场，无默认值）；
- ``--mode`` 缺省 read_only；显式 ``--mode read_only`` 合法（与缺省同
  效，不静默忽略）；
- 不适用参数一律退出码 2：read_only 下出现任一 tracking 专属参数
  （含值等于默认概念值，如 ``--amplitude-rad 0``）即拒绝并点名；
  tracking 下出现 ``--sample-period-s`` 即拒绝；
- ``--mode tracking`` 必须同时给 ``--enable-motion``（防误操作双重
  闸门）；``--enable-motion`` 不随 ``--mode tracking`` 给出同样拒绝；
- tracking 验收阈值 ``--tracking-error-limit-rad`` / ``--stop-latency-limit-s``
  可选注入（I-06：缺省不判定；给出后超限按 fail 结论报告实际值与判据）；
  ``--expected-mode-machine`` 两模式可选（I-08：G3 冻结后透传驱动做相等
  断言，缺省仅变化检测）；
- tracking 还需有效验收配置：硬件配置须过 ``purpose="motion"`` 门槛
  （校准/限值/周期/诊断时长全部非 null 且已核验），不满足退出码 2；
- 诊断时长一律取硬件配置 ``timing.diagnostic_duration_s``（现场冻结值
  优先；工具不提供 CLI 覆盖，防止与现场冻结值漂移——未冻结时退出码
  2），发送周期/发布期限取 ``timing.control_period_s``（同一来源）。

退出码：0=pass；1=fail/aborted（含清理异常）；2=参数/配置错误。

用户界面用 print（scripts/ CLI 例外，见 AGENTS.md）；诊断/调试信息
走 logging。
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

# 支持从 scripts/ 目录直接运行，确保 `from src.xxx` 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.real.r1d_command_policy import R1DSlotHold
from src.real.r1d_config import (
    R1DHardwareConfig,
    load_r1d_hardware_config,
    validate_r1d_hardware_config,
)
from src.real.r1d_diagnostics import (
    R1DDiagnosticConfig,
    R1D_SDK_IDENTITY,
    run_r1d_diagnostic,
)
from src.real.r1d_factory import build_r1d_arm
from src.real.r1d_types import R1D_SLOT_COUNT
from src.robot.robot_profile import RobotProfile, load_robot_profile

logger = logging.getLogger(__name__)

# 槽位策略 YAML 的合法字段（协议 MotorCmd_ 全字段语义；数值来自现场）
_SLOT_POLICY_FIELDS = ("mode", "q", "dq", "kp", "kd", "tau")


def _build_parser() -> argparse.ArgumentParser:
    """构造参数解析器（default=None 哨兵以检测显式提供）。"""
    parser = argparse.ArgumentParser(
        prog="test_unitree_r1d.py",
        description=(
            "R1-A7-D 只读/低速诊断工具（阶段 4 Task 4.8）：read_only 记录"
            "状态流（不接管不发布）；tracking 对单个显式选中关节执行低速"
            "升余弦诊断波形并验证停止。产物=JSON 摘要+CSV+NPZ。"
        ),
    )
    parser.add_argument(
        "--robot-config", required=True, type=Path,
        help="机器人描述 YAML（configs/robots/r1d_left.yaml 或 r1d_right.yaml）",
    )
    parser.add_argument(
        "--hardware-config", required=True, type=Path,
        help="硬件配置 YAML（configs/robots/r1d_hardware_left.yaml 现场冻结版）",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="产物输出目录（JSON 摘要 + CSV + NPZ）",
    )
    parser.add_argument(
        "--acquire-timeout-s", required=True, type=float,
        help="获取整机控制权整体超时（秒；数值来自现场，透传传输层）。"
             "与 --robot-config/--hardware-config/--output-dir 同为强制参数"
             "（工具不提供默认值，防现场数值漂移）",
    )
    parser.add_argument(
        "--mode", choices=["read_only", "tracking"], default=None,
        help="诊断模式（缺省 read_only；tracking 需 --enable-motion）",
    )
    parser.add_argument(
        "--sample-period-s", type=float, default=None,
        help="状态采样周期（秒）；仅 read_only 适用（必填）",
    )
    parser.add_argument(
        "--tracking-error-limit-rad", type=float, default=None,
        help="跟踪误差验收阈值（rad，各关节 RMS/max 的判定上界）；仅 "
             "tracking 适用（必填——R4 评审 2026-11-18：验收阈值缺失的 "
             "tracking 只完成采集、不得判定通过，CLI 层直接拒绝缺参）",
    )
    parser.add_argument(
        "--stop-latency-limit-s", type=float, default=None,
        help="停止延迟验收阈值（秒，对反馈侧延迟 latency_feedback_s 的"
             "判定上界）；仅 tracking 适用（必填，同 --tracking-error-limit-rad）",
    )
    parser.add_argument(
        "--expected-mode-machine", type=int, default=None,
        help="冻结期望机型值 mode_machine（uint8；G3 实机回读冻结后给出，"
             "read_only/tracking 均可选）。给出后驱动按相等断言校验，"
             "缺省仅做变化检测（Ruling R6）",
    )
    parser.add_argument(
        "--selected-joint", type=str, default=None,
        help="tracking 选中的模型关节名（profile.active_joint_names 之一）",
    )
    parser.add_argument(
        "--amplitude-rad", type=float, default=None,
        help="tracking 波形峰值幅度（rad）；仅 tracking 适用（必填）",
    )
    parser.add_argument(
        "--frequency-hz", type=float, default=None,
        help="tracking 波形频率（Hz）；仅 tracking 适用（必填）",
    )
    parser.add_argument(
        "--stop-confirm-dq-threshold", type=float, default=None,
        help="停止反馈确认速度阈值（rad/s）；仅 tracking 适用（必填，"
             "与硬件验收配置一致）",
    )
    parser.add_argument(
        "--enable-motion", action="store_true",
        help="运动显式使能闸门；仅 --mode tracking 合法（必给）",
    )
    parser.add_argument(
        "--slot-policies", type=Path, default=None,
        help="全机非目标槽位保持策略 YAML（G4 现场确认后提供；仅 tracking 适用，"
             "缺省时 activate 将被驱动拒绝并如实生成失败结论）",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None,
        help="日志级别（缺省 INFO）",
    )
    return parser


def _require_positive(value: float | None, flag: str) -> float:
    """校验正有限数值参数（缺省/非正 → 参数错误）。"""
    if value is None:
        raise _UsageError(f"参数 {flag} 必填（数值来自现场验收配置，无默认值）")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise _UsageError(f"参数 {flag} 必须是数值，实际 {value!r}")
    if not (number > 0.0) or number == float("inf"):
        raise _UsageError(f"参数 {flag} 必须为正有限值，实际 {value!r}")
    return number


class _UsageError(Exception):
    """参数/配置错误（退出码 2）。"""


def _reject_inapplicable(args: argparse.Namespace, mode: str) -> None:
    """拒绝当前模式下不适用的显式参数（含等于默认概念值，不静默忽略）。"""
    tracking_only = (
        ("--selected-joint", args.selected_joint),
        ("--amplitude-rad", args.amplitude_rad),
        ("--frequency-hz", args.frequency_hz),
        ("--stop-confirm-dq-threshold", args.stop_confirm_dq_threshold),
        ("--slot-policies", args.slot_policies),
        ("--tracking-error-limit-rad", args.tracking_error_limit_rad),
        ("--stop-latency-limit-s", args.stop_latency_limit_s),
    )
    if mode == "read_only":
        for flag, value in tracking_only:
            if value is not None:
                raise _UsageError(
                    f"参数 {flag} 仅适用于 --mode tracking；read_only 下显式给出"
                    "（无论取值，包括等于默认概念值）一律拒绝，不静默忽略"
                )
        if args.enable_motion:
            raise _UsageError(
                "参数 --enable-motion 仅适用于 --mode tracking；read_only 下"
                "显式给出即拒绝（只读诊断不得携带运动使能）"
            )
    else:
        if args.sample_period_s is not None:
            raise _UsageError(
                "参数 --sample-period-s 仅适用于 --mode read_only；tracking 的"
                "状态采样随发送周期（显式给出即拒绝）"
            )


def _load_slot_policies(path: Path) -> dict[int, R1DSlotHold]:
    """解析全机非目标槽位保持策略 YAML（数值来自现场 G4 冻结配置）。

    文件格式::

        slot_policies:
          0: {mode: 1, q: 0.0, dq: 0.0, kp: 10.0, kd: 0.5, tau: 0.0}
          ...
    """
    if not path.is_file():
        raise _UsageError(f"槽位策略文件不存在或不是常规文件: {path}")
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"slot_policies"}:
        raise _UsageError(
            f"槽位策略文件必须只含顶层键 slot_policies（实际键 {sorted(map(str, raw)) if isinstance(raw, dict) else type(raw).__name__}）"
        )
    entries = raw["slot_policies"]
    if not isinstance(entries, dict) or not entries:
        raise _UsageError("slot_policies 必须是非空映射（槽位 → 策略字段）")
    policies: dict[int, R1DSlotHold] = {}
    for key, fields in entries.items():
        try:
            slot = int(key)
        except (TypeError, ValueError):
            raise _UsageError(f"槽位策略键必须是整数槽位，实际 {key!r}")
        if not 0 <= slot < R1D_SLOT_COUNT:
            raise _UsageError(f"槽位 {slot} 越界（整机 {R1D_SLOT_COUNT} 槽）")
        if not isinstance(fields, dict):
            raise _UsageError(f"槽位 {slot} 的策略必须是映射")
        unknown = sorted(set(map(str, fields)) - set(_SLOT_POLICY_FIELDS))
        if unknown:
            raise _UsageError(f"槽位 {slot} 策略含未知字段 {unknown}")
        missing = sorted(set(_SLOT_POLICY_FIELDS) - set(map(str, fields)))
        if missing:
            raise _UsageError(f"槽位 {slot} 策略缺字段 {missing}")
        try:
            policies[slot] = R1DSlotHold(
                mode=int(fields["mode"]),
                q=float(fields["q"]),
                dq=float(fields["dq"]),
                kp=float(fields["kp"]),
                kd=float(fields["kd"]),
                tau=float(fields["tau"]),
            )
        except (TypeError, ValueError) as error:
            raise _UsageError(f"槽位 {slot} 策略数值非法: {error}")
    return policies


def _sha256_of(path: Path) -> str:
    """计算文件 SHA256（现场配置来源哈希，产物溯源用）。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_selected_joint(
    profile: RobotProfile, joint_name: str | None
) -> int:
    """把显式关节名解析为模型序索引（按名寻址，不猜顺序）。"""
    if joint_name is None:
        raise _UsageError(
            "参数 --selected-joint 在 tracking 模式必填（模型关节名，"
            "profile.active_joint_names 之一）"
        )
    names = list(profile.active_joint_names)
    if joint_name not in names:
        raise _UsageError(
            f"--selected-joint {joint_name!r} 不在 profile.active_joint_names"
            f" 中（合法值: {names}）"
        )
    return names.index(joint_name)


def _build_diagnostic_config(
    args: argparse.Namespace,
    profile: RobotProfile,
    hardware: R1DHardwareConfig,
    hardware_path: Path,
    mode: str,
) -> R1DDiagnosticConfig:
    """按模式组装诊断配置（数值来源：CLI 显式参数 + 硬件配置冻结值）。"""
    # 诊断时长：一律取硬件配置冻结值（现场配置引用优先，无 CLI 覆盖）
    duration = hardware.timing.diagnostic_duration_s
    if duration is None:
        raise _UsageError(
            "硬件配置 timing.diagnostic_duration_s 为 null（未冻结）：诊断时长"
            "必须由现场验收配置提供，工具不提供默认值"
        )
    side = hardware.side
    provenance = {
        "hardware_config_path": str(hardware_path),
        "hardware_config_sha256": _sha256_of(hardware_path),
        "side": side,
    }
    if mode == "read_only":
        sample_period = _require_positive(
            args.sample_period_s, "--sample-period-s"
        )
        return R1DDiagnosticConfig(
            mode=mode,
            duration_s=duration,
            sample_period_s=sample_period,
            **provenance,
        )
    # tracking：发送周期/发布期限取硬件配置控制周期（单一来源）
    period = hardware.timing.control_period_s
    if period is None:
        raise _UsageError(
            "硬件配置 timing.control_period_s 为 null：tracking 发送周期必须"
            "来自现场冻结配置（motion 门槛校验亦会拒绝）"
        )
    amplitude = _require_positive(args.amplitude_rad, "--amplitude-rad")
    frequency = _require_positive(args.frequency_hz, "--frequency-hz")
    stop_threshold = _require_positive(
        args.stop_confirm_dq_threshold, "--stop-confirm-dq-threshold"
    )
    # R4（评审 2026-11-18）：tracking 验收阈值必填（缺省即参数错误）。
    # 验收模式要求完整有限阈值；无阈值的采集无法与验收通过区分，库级
    # 判定（r1d_diagnostics）同口径兜底输出 fail + "仅采集不构成验收"。
    error_limit = _require_positive(
        args.tracking_error_limit_rad, "--tracking-error-limit-rad"
    )
    latency_limit = _require_positive(
        args.stop_latency_limit_s, "--stop-latency-limit-s"
    )
    selected_joint = _resolve_selected_joint(profile, args.selected_joint)
    # 校准换算参数（模型序）：来自硬件配置校准节（motion 门槛保证非 null）
    assert hardware.calibration.joint_signs is not None
    assert hardware.calibration.joint_offsets_rad is not None
    names = list(profile.active_joint_names)
    signs = tuple(
        int(hardware.calibration.joint_signs[name]) for name in names
    )
    offsets = tuple(
        float(hardware.calibration.joint_offsets_rad[name]) for name in names
    )
    return R1DDiagnosticConfig(
        mode=mode,
        duration_s=duration,
        send_period_s=period,
        selected_joint=selected_joint,
        amplitude_rad=amplitude,
        frequency_hz=frequency,
        stop_confirm_dq_threshold=stop_threshold,
        publish_deadline_s=period,
        # I-06（评审 2026-11-17）打通注入通路；R4（评审 2026-11-18）
        # 升级为必填——缺失在参数解析层即拒绝（退出码 2），不再允许
        # "无阈值采集"经 CLI 进入验收模式
        tracking_error_limit_rad=error_limit,
        stop_latency_limit_s=latency_limit,
        joint_signs=signs,
        joint_offsets_rad=offsets,
        **provenance,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口：解析参数 → 校验配置 → 构造驱动 → 执行诊断。

    Returns:
        0=pass；1=fail/aborted；2=参数/配置错误。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    log_level = args.log_level or "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        # 1) 模式与不适用参数（显式默认值也拒绝，裁定 1）
        mode = args.mode or "read_only"
        _reject_inapplicable(args, mode)
        if mode == "tracking" and not args.enable_motion:
            raise _UsageError(
                "--mode tracking 需要显式 --enable-motion（运动双重闸门，"
                "防止误操作）"
            )
        if mode != "tracking" and args.enable_motion:
            raise _UsageError(
                "--enable-motion 只能伴随 --mode tracking 给出（read_only "
                "下显式给出即拒绝）"
            )
        _require_positive(args.acquire_timeout_s, "--acquire-timeout-s")
        # I-08 修复（评审 2026-11-17）：expected_mode_machine 冻结值经 CLI
        # 注入透传到驱动（G3 实机回读冻结后给出；缺省仅变化检测，R6）
        if args.expected_mode_machine is not None and not (
            0 <= args.expected_mode_machine <= 255
        ):
            raise _UsageError(
                f"参数 --expected-mode-machine 必须在 uint8 范围 [0, 255]，"
                f"实际 {args.expected_mode_machine}"
            )

        # 2) 加载与门槛校验（read_only→read_only 档；tracking→motion 档）
        if not args.robot_config.is_file():
            raise _UsageError(f"机器人描述文件不存在: {args.robot_config}")
        if not args.hardware_config.is_file():
            raise _UsageError(f"硬件配置文件不存在: {args.hardware_config}")
        profile = load_robot_profile(args.robot_config)
        hardware = load_r1d_hardware_config(args.hardware_config)
        purpose = "motion" if mode == "tracking" else "read_only"
        try:
            validate_r1d_hardware_config(hardware, profile, purpose=purpose)
        except ValueError as error:
            raise _UsageError(
                f"{mode} 模式需要有效验收配置（purpose={purpose} 门槛未满足）: {error}"
            )

        # 3) 槽位策略（仅 tracking；G4 现场确认后提供）
        slot_policies: dict[int, R1DSlotHold] | None = None
        if mode == "tracking":
            slot_policies = (
                _load_slot_policies(args.slot_policies)
                if args.slot_policies is not None
                else None
            )

        # 4) 诊断配置与驱动构造
        diag_config = _build_diagnostic_config(
            args, profile, hardware, args.hardware_config, mode
        )
        arm = build_r1d_arm(
            profile,
            hardware,
            acquire_timeout_s=_require_positive(
                args.acquire_timeout_s, "--acquire-timeout-s"
            ),
            slot_policies=slot_policies,
            expected_mode_machine=args.expected_mode_machine,
            stop_confirm_dq_threshold=(
                diag_config.stop_confirm_dq_threshold
                if mode == "tracking"
                else None
            ),
        )
    except _UsageError as error:
        print(f"参数/配置错误: {error}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as error:
        print(f"配置加载失败: {error}", file=sys.stderr)
        return 2

    # 5) 执行诊断（真实路径：activate 仅 tracking；read_only 零接管零发布）
    result = run_r1d_diagnostic(arm, diag_config, args.output_dir)

    # 6) 摘要（用户界面 print；CLI 例外见 AGENTS.md）
    verdict = result["verdict"]
    print("=" * 62)
    print(f"R1-A7-D 诊断结论: {verdict}（模式 {result['mode']}）")
    print(f"SDK: {R1D_SDK_IDENTITY}")
    print(f"代码 SHA: {result['provenance']['code_sha']}")
    print(
        f"硬件配置: {result['provenance']['hardware_config']['path']} "
        f"(sha256 {str(result['provenance']['hardware_config']['sha256'])[:16]}…)"
    )
    publish = result["publish_metrics"]
    if publish["unavailable"]:
        print("publish 间隔指标: unavailable（无成功发布记录，不报 0）")
    else:
        print(
            f"publish 间隔: p50={publish['p50_s']:.6f}s p95={publish['p95_s']:.6f}s "
            f"p99={publish['p99_s']:.6f}s max={publish['max_s']:.6f}s "
            f"超期={publish['deadline_overruns']} 次（判据 >1.5×期限 "
            f"{publish['deadline_s']}s）"
        )
    tracking = result["tracking_metrics"]
    if tracking is not None:
        if tracking["unavailable"]:
            print("跟踪误差: unavailable（无配对样本，不报零误差）")
        else:
            selected = diag_config.selected_joint
            assert selected is not None
            item = tracking["per_joint"][selected]
            print(
                f"跟踪误差（选中关节 {selected}）: rms={item['rms_rad']:.6f} rad "
                f"max={item['max_rad']:.6f} rad（配对 {tracking['n_matched']}，"
                f"无匹配 {tracking['no_match_samples']}）"
            )
    stop = result["stop"]
    if stop["requested"]:
        print(
            f"停止: confirmed={stop['stop_confirmed']} "
            f"软件延迟={stop['latency_software_s']}s "
            f"反馈达阈值延迟={stop['latency_feedback_s']}s"
        )
    reasons = result["reasons"]
    if reasons:
        print("原因:")
        for reason in reasons:
            print(f"  - {reason}")
    artifacts = result.get("artifacts") or {}
    if artifacts:
        print(f"产物目录: {args.output_dir}")
        for key in ("publish_csv", "samples_csv", "send_csv", "npz"):
            if key in artifacts:
                print(f"  - {artifacts[key]}")
        print("  - diagnostic_summary.json")
    print("=" * 62)
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
