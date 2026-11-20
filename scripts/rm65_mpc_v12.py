#!/usr/bin/env python3
"""RM-65 V12 — EpisodeRunner 管线架构。

V11 的规划逻辑提取到 MPCController（src/ilqt/mpc_controller.py），
仿真执行/诊断封装到 SimComponent，
管线编排由 EpisodeRunner（src/ilqt/episode_runner.py）驱动。
V12 是组装器 + 球生成 + 评估。

=== V12 vs V11 ===
  - 规划逻辑（~800 行 replan + buffer + follow-through）→ MPCController
  - 安全滤波（~130 行 X-wall + safety filter）→ PredictiveSafetyFilter
  - 主循环编排 → EpisodeRunner.run()
  - 仿真专属（碰撞/接触/PD推回/诊断指标）→ SimComponent
  - argparse / 配置 / 球生成 / 评估 → 保留在 V12

用法（与 V11 一致）:
    python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --seed 42
    python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --no-plot
    python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --seed 2 --viewer --position-mode --no-r-decay
"""

from __future__ import annotations

import sys
import time
import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim.rm65_env import RM65Env
from src.ilqt.tube_types import HitWindow
from src.robot.constants import INIT_Q, INIT_Q_LEFT
from src.robot.robot_profile import load_robot_profile
from src.utils.yaml_utils import load_config, merge_configs

# ── 多机器人入口（阶段 3 · Task 3.10）：R1D 仿真路径委托新 runner ──
# 模块级导入使 CLI 分派可被测试 monkeypatch（不复制 V12 主循环）
from src.sim.hit_detection import classify_hit_result
from src.sim.r1d_runner import run_r1d_episode, write_v12_r1d_artifacts

# ── V12 共享管线（Task 5）：RM65/R1D 共用 src/sim/v12_episode.py ──
from src.sim.v12_config import ResolvedV12Config, resolve_v12_config
from src.sim.v12_episode import V12EpisodeResult, run_v12_episode

# ── T6：统一 v2 产物（v12_episode_v2 JSON/NPZ + snapshot）──
from src.sim.episode_artifacts import write_episode_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("src.ilqt.robot_limits").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class _StepTimer:
    """每步主循环计时钩子（post_exec_hooks 间隔 = 主循环周期）。

    用于同步 vs 异步重规划的主循环停顿对比：同步 ``SyncReplanMode.submit``
    内联求解会阻塞 step()，对应步的钩子间隔显著拉长；异步 submit 仅入队，
    间隔保持平稳。间隔本身（而非求解耗时）才是控制循环是否被卡住的指标。
    """

    def __init__(self) -> None:
        """初始化计时器。"""
        self._last: float | None = None
        self.deltas_ms: list[float] = []

    def hook(self, ctx: object) -> None:
        """记录与上次调用的间隔（ms）。

        Args:
            ctx: StepContext（未使用，保持 hook 签名一致）。
        """
        now = time.perf_counter()
        if self._last is not None:
            self.deltas_ms.append((now - self._last) * 1000.0)
        self._last = now

    def summary(self, period_ms: float) -> str:
        """生成可分派解析的单行摘要。

        Args:
            period_ms: 控制周期（ms），用于统计超周期步数。

        Returns:
            形如 ``__STEP_TIMING__: n=.. p50=.. ... n_over_period=..`` 的字符串。
        """
        if not self.deltas_ms:
            return "__STEP_TIMING__: n=0"
        ordered = sorted(self.deltas_ms)

        def _pct(q: float) -> float:
            return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

        over = sum(1 for v in ordered if v > period_ms)
        # deciles 向量：供实时图 ECDF（各 episode 的每步主循环耗时分位点池化）
        deciles = ",".join(f"{_pct(q):.2f}" for q in np.linspace(0.1, 1.0, 10))
        return (f"__STEP_TIMING__: n={len(ordered)} p50={_pct(0.50):.2f} "
                f"p95={_pct(0.95):.2f} p99={_pct(0.99):.2f} "
                f"max={ordered[-1]:.2f} n_over_period={over} "
                f"period_ms={period_ms:.2f} deciles={deciles}")

    def latencies_line(self) -> str:
        """生成逐步耗时原始序列单行（供真 per-step ECDF 图使用）。

        Returns:
            形如 ``__STEP_LATENCIES__: 0.26,0.27,...`` 的字符串（2 位小数，步序保留）。
        """
        return ("__STEP_LATENCIES__: "
                + ",".join(f"{v:.2f}" for v in self.deltas_ms))


# ==============================================================================
# 可视化（从 V11 复用）
# ==============================================================================

def _import_v11_visuals():
    """惰性导入可视化函数（避免无 --viewer/--plot 时的导入开销）。"""
    from src.sim.v11_visuals import visualize_rm65_result, plot_tube_results
    return visualize_rm65_result, plot_tube_results


# ==============================================================================
# 多机器人入口（阶段 3 · Task 3.10）：R1D 仿真路径分派
# ==============================================================================

# R1D 路径明确支持的共享基础参数（其余 RM65 专属旧参数一律拒绝，
# 不能接受后忽略 —— R1D 的来球/步数/权重等全部来自 --sim-task-config YAML）
_R1D_SHARED_FLAGS = frozenset({
    "robot_config", "runtime_config", "sim_task_config", "output",
    "seed", "no_plot", "viewer", "log_level",
    "racket_config", "task_overrides",
    # V12 共享基础参数：R1D overlay 同样接受并真正进入 ResolvedV12Config
    # （serve_box → serve.mode、ball_speed → serve.ball_speed、
    #   no_bounce → scene.use_bounce、position_mode → MPCConfig.is_position_mode、
    #   async_replan → MPCConfig.async_mode；T4/T5 已为 R1D 实现异步重规划，
    #   T9 命令行联调确认必须放开，不能当作 RM65 专属参数拒绝）。
    "serve_box", "ball_speed", "no_bounce", "position_mode", "async_replan",
})

# 最近一次 R1D 分派拒绝的 RM65 专属参数清单（供测试断言；非生产 API）
_last_r1d_rejected_flags: list[str] = []


def _build_parser() -> argparse.ArgumentParser:
    """构造 V12 参数解析器（RM65 参数面与 V11 一致 + 4 个多机器人新参数）。"""
    parser = argparse.ArgumentParser(description="RM-65 V12 EpisodeRunner 管线架构")
    parser.add_argument(
        "--viewer", action="store_true",
        help="计算完成后回放（RM65 按真实速度；R1D 以 0.2× 慢放循环）"
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--fd", action="store_true", help="使用有限差分线性化")
    parser.add_argument("--solver-backend", choices=["auto", "python", "cpp"], default="auto",
                        help="iLQR 求解器后端（默认 auto：C++ 扩展可用则用 C++；"
                             "复现 Python FD 基线时显式传 python）")
    parser.add_argument("--horizon", type=int, default=None, help="短地平线步数")
    parser.add_argument("--iter", type=int, default=None, help="每次重规划迭代数")
    parser.add_argument("--fix-joint5", action="store_true", help="固定第 6 关节")
    parser.add_argument("--backswing", type=float, default=0.6, help="后摆幅度 (rad)")
    parser.add_argument("--bs-ratio", type=float, default=0.35, help="后摆占比")
    parser.add_argument("--use-backswing", action="store_true", help="启用后摆（v11 默认无后摆）")
    parser.add_argument("--r-decay", type=float, default=0.40, help="R 衰减占比")
    parser.add_argument("--no-r-decay", action="store_true", help="禁用 R 退火")
    parser.add_argument("--hit-shift", type=float, default=0.0, help="随挥偏移距离 (m)")
    parser.add_argument("--ball-speed", type=float, default=None, help="球到达击打点时水平速度 (m/s)")
    parser.add_argument("--ball-distance", type=float, default=None, help="球起始位置到击打点的直线距离 (m)")
    parser.add_argument("--approach-angle", type=float, default=0.0, help="球飞来方向角 (度)")
    parser.add_argument("--serve-box", action="store_true", help="使用长方体发球区模式")
    parser.add_argument("--no-bounce", action="store_true", help="禁用地面弹跳")
    parser.add_argument("--serve-distance", type=float, default=8.0, help="发球区 Y 方向距离 (m)")
    parser.add_argument("--serve-height", type=float, default=1.2, help="发球区中心高度 (m)")
    parser.add_argument("--serve-x-size", type=float, default=8.0, help="发球区 X 轴全长 (m)")
    parser.add_argument("--serve-y-size", type=float, default=0.2, help="发球区 Y 轴全长 (m)")
    parser.add_argument("--serve-z-size", type=float, default=0.3, help="发球区 Z 轴全长 (m)")
    parser.add_argument("--Q-tcp-soft", type=float, default=0.0, help="TCP 速度软惩罚权重")
    parser.add_argument("--Q-qdot-limit", type=float, default=0.0, help="关节速度软惩罚权重")
    parser.add_argument("--normal-weight", type=float, default=500000.0, help="拍面法向量代价权重")
    parser.add_argument("--normal-flip", action="store_true", help="翻转法向量方向")
    parser.add_argument("--replan-interval", type=int, default=None, help="重规划间隔步数")
    parser.add_argument("--near-iters", type=int, default=None, help="near阶段 iLQR 迭代次数")
    parser.add_argument("--window-ms", type=float, default=50.0, help="Tube 候选窗口半宽 (ms)")
    parser.add_argument("--softmin-beta", type=float, default=5.0, help="终端 softmin 锐度 β")
    parser.add_argument("--corridor-radius", type=float, default=0.12,
                        help="走廊半宽 (m)；标称 = 拍半径 0.12 m，敏感性实验可改")
    parser.add_argument("--ablation",
                        choices=["full", "tube_only", "softmin_only", "none", "select_one"],
                        default=None, help="消融模式（select_one = 候选选择基线，chat12 C3）")
    parser.add_argument("--no-softmin", action="store_true", help="[已废弃] 禁用 softmin")
    parser.add_argument("--no-tube", action="store_true", help="[已废弃] 禁用 Tube 走廊")
    parser.add_argument("--no-follow-through", action="store_true", help="禁用球轨迹回溯随挥")
    parser.add_argument("--follow-trigger", choices=["planned", "contact"], default="planned",
                        help="随挥触发方式")
    parser.add_argument("--no-v-maximize", action="store_true", help="禁用中点速度最大化")
    parser.add_argument("--fixed-direction", action="store_true", help="使用固定 YAML hit_direction")
    parser.add_argument("--target-speed", type=float, default=1.8, help="终端目标速度 (m/s)")
    parser.add_argument("--no-plot", action="store_true", help="禁用 matplotlib 可视化")
    parser.add_argument("--dump-trajectory", type=str, default=None, help="保存轨迹到 npz 文件（可被 replay_trajectory.py 加载）")
    parser.add_argument("--realtime", action="store_true", help="模拟实时节奏")
    parser.add_argument("--async-replan", action="store_true", help="启用异步重规划")
    parser.add_argument("--time-perturb-ms", type=float, default=0.0, help="球到达时间预测扰动 (ms)")
    parser.add_argument("--space-perturb-m", type=float, default=0.0, help="击打点空间偏移 (m)")
    parser.add_argument("--spatial-perturb-target", choices=["hitpoint", "ballstate"],
                        default="hitpoint",
                        help="空间扰动注入口径: hitpoint=只偏移单点目标（历史口径）; "
                             "ballstate=偏移球状态估计（候选集合/走廊轴/初始化同源）")
    parser.add_argument("--perturb-alpha-min", type=float, default=0.0, help="衰减扰动保底值")
    parser.add_argument("--shared-input-full", action="store_true",
                        help="统一观测扰动（chat12 C2）: ballstate 口径下时序/可达性也消费受扰球态")
    parser.add_argument("--no-freeze-first-plan", action="store_true",
                        help="不冻结时间端到端（chat12 C1）: 首次规划期间球继续飞行，计算延迟计入任务时间")
    parser.add_argument("--always-collision", action="store_true",
                        help="击球前球-拍碰撞常开（chat14 碰撞窗口验证）: 绕过 k_hit/距离窗口判定")
    parser.add_argument("--random-perturb", action="store_true", help="随机扰动")
    parser.add_argument("--perturb-sign", choices=["random", "positive", "negative"],
                        default="random", help="扰动符号方向")
    parser.add_argument("--time-perturb-min-ms", type=float, default=0.0, help="随机扰动时间下限 (ms)")
    parser.add_argument("--space-perturb-min-m", type=float, default=0.0, help="随机扰动空间下限 (m)")
    parser.add_argument("--ball-speed-perturb-pct", type=float, default=0.0, help="球速耦合扰动百分比")
    parser.add_argument("--max-tcp", type=float, default=None, help="TCP 线速度硬限制 (m/s)")
    parser.add_argument("--terminal-exempt-steps", type=int, default=None, help="终段豁免步数（默认沿用 RobotLimits 配置值 20）")
    parser.add_argument("--sim-limits", action="store_true",
        help="(已废弃) 仿真限位现为默认行为，此标志无效")
    parser.add_argument("--limits-config", type=Path, default=None,
        help="限位 YAML(RealRobotConfig 格式). 优先级高于 --sim-limits")
    parser.add_argument("--obs-freq", type=float, default=0, help="观测频率 Hz")
    parser.add_argument("--obs-noise-pos", type=float, default=0, help="观测位置噪声 std (m)")
    parser.add_argument("--obs-noise-vel", type=float, default=0, help="观测速度噪声 std (m/s)")
    parser.add_argument("--obs-use-kf", action="store_true", help="观测启用 KF 滤波")
    parser.add_argument("--position-mode", action="store_true", help="启用位置控制模式")
    parser.add_argument("--no-feedforward", action="store_true", help="禁用前馈补偿")
    parser.add_argument("--kp", type=float, nargs='+', default=None, help="位置模式 PD Kp")
    parser.add_argument("--kd", type=float, nargs='+', default=None, help="位置模式 PD Kd")
    parser.add_argument("--dq-max-fraction", type=float, default=None, help="单步角度变化系数")
    # ── 阶段调度参数（原硬编码）──
    parser.add_argument("--far-threshold", type=int, default=None, help="far/mid 阶段边界步数")
    parser.add_argument("--near-threshold", type=int, default=None, help="mid/near 阶段边界步数（0=自动）")
    parser.add_argument("--first-plan-iters", type=int, default=None, help="首次规划 iLQR 迭代数")
    # ── 代价阶段倍率（原硬编码）──
    parser.add_argument("--Qp-scale-far", type=float, default=None, help="far 阶段位置代价倍率")
    parser.add_argument("--Qv-scale-far", type=float, default=None, help="far 阶段速度代价倍率")
    parser.add_argument("--Qp-scale-near", type=float, default=None, help="near 阶段位置代价倍率")
    parser.add_argument("--Qv-scale-near", type=float, default=None, help="near 阶段速度代价倍率")
    # ── 随挥 PD 增益（原硬编码在 follow_through.py）──
    parser.add_argument("--follow-kp", type=float, default=None, help="随挥 PD 比例增益")
    parser.add_argument("--follow-kd", type=float, default=None, help="随挥 PD 微分增益")
    # ── HitPointRefiner 阈值（原硬编码在 hit_point_refiner.py）──
    parser.add_argument("--hit-lock", type=int, default=None, help="击球点防抖锁定步数")
    parser.add_argument("--hard-margin", type=float, default=None, help="IK 硬裕度 (度)")
    parser.add_argument("--warn-margin", type=float, default=None, help="IK 警告裕度 (度)")
    parser.add_argument("--j1-warn", type=float, default=None, help="关节1 警告裕度 (度)")
    parser.add_argument("--refiner-window", type=int, default=None, help="Refiner 搜索窗口半宽 (步)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别（默认 INFO; 计时分析需 DEBUG 以输出稳态 REPLAN）")
    parser.add_argument("--dump-step-timing", action="store_true",
                        help="输出每步主循环耗时摘要（__STEP_TIMING__, 同步/异步停顿对比用）")
    parser.add_argument("--dump-step-latencies", action="store_true",
                        help="额外输出逐步耗时原始序列（__STEP_LATENCIES__, 真 per-step ECDF 数据源）")
    # ── 多机器人入口（Task 3.10）：R1D 仿真路径（缺省全部 None → RM65 旧路径）──
    parser.add_argument("--robot-config", type=Path, default=None,
                        help="机器人描述 YAML（缺省 RM65 旧路径；显式 rm65.yaml 等价缺省；"
                             "unitree_r1d 进入 R1D 仿真路径，须与 --runtime-config/"
                             "--sim-task-config 同时提供）")
    parser.add_argument("--runtime-config", type=Path, default=None,
                        help="R1D 仿真运行参数 YAML（仅 R1D 路径，如 configs/robots/"
                             "r1d_sim_runtime_left.yaml）")
    parser.add_argument("--sim-task-config", type=Path, default=None,
                        help="R1D 仿真任务 YAML（仅 R1D 路径，如 configs/r1d_sim_left.yaml；"
                             "来球/步数/权重全部来自该文件）")
    parser.add_argument("--output", type=Path, default=None,
                        help="产物输出目录（RM65 与 R1D 路径均生效；RM65 缺省 "
                             "results/，R1D 缺省 results/r1d_sim）")
    # ── V12 共享管线（Task 5）：球拍描述与任务点路径覆盖 ──
    parser.add_argument("--racket-config", type=Path, default=None,
                        help="球拍描述 YAML（共享管线使用；R1D 右臂缺省由注册表"
                             "提供 configs/rackets/blade100_v9_reference.yaml）")
    parser.add_argument("--task-overrides", action="append", default=None,
                        metavar="KEY=VALUE",
                        help="任务参数点路径覆盖（可重复，如 "
                             "--task-overrides mpc.total_horizon=240；值按 YAML 解析）")
    return parser


def _explicit_cli_actions(
    parser: argparse.ArgumentParser, argv: Sequence[str]
) -> list[argparse.Action]:
    """把 argv 中实际出现的选项映射到对应 action（解析语义对齐）。

    收集以 ``--`` 开头的 token（``--opt=value`` 取 ``=`` 前部分），
    裸 ``--`` 之后视为位置参数停止收集；选项名按"精确命中 → 唯一
    前缀缩写"匹配 action（与 argparse 缩写规则一致，歧义 token 由
    parse_args 自行报错，不会到达此处）。
    """
    explicit: list[argparse.Action] = []
    for token in argv:
        if token == "--":
            break
        if not token.startswith("--"):
            continue
        name = token.split("=", 1)[0]
        matched = [a for a in parser._actions if name in a.option_strings]  # noqa: SLF001 - argparse 无公开遍历 API
        if not matched:
            matched = [
                a for a in parser._actions
                if any(o.startswith(name) for o in a.option_strings)
            ]
        if len(matched) == 1:
            explicit.append(matched[0])
    return explicit


def _runtime_explicit_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: Sequence[str],
) -> dict[str, object]:
    """只把命令行真正给出的运行时覆盖转发给 ``run_v12_episode``（F01）。

    旧实现无条件传全部 8 个参数，导致「未传」被当成显式 False/0，覆盖掉
    artifact 里已保存的值；这里以 argv 实际出现的 option 为准，未出现的键
    不进入 explicit，由管线按「已保存值 > 当前默认」继承。

    Args:
        parser: 入口解析器（用于 ``_explicit_cli_actions`` 反查）。
        args: 已解析参数命名空间。
        argv: 原始命令行参数（不含程序名）。

    Returns:
        形如 ``{"first_plan_flight": True, "max_tcp": 1.25}`` 的显式覆盖
        dict；只含用户真正给出的键。
    """
    explicit_dests = {
        action.dest for action in _explicit_cli_actions(parser, argv)
    }
    options: dict[str, object] = {}
    if "no_freeze_first_plan" in explicit_dests:
        # 旧 CLI 语义：--no-freeze-first-plan 表示首规划期间让球飞行
        options["first_plan_flight"] = bool(args.no_freeze_first_plan)
    if "always_collision" in explicit_dests:
        options["always_collision"] = bool(args.always_collision)
    if "ball_speed_perturb_pct" in explicit_dests:
        options["ball_speed_perturb_pct"] = float(args.ball_speed_perturb_pct)
    if "fixed_direction" in explicit_dests:
        options["fixed_direction"] = bool(args.fixed_direction)
    if "limits_config" in explicit_dests:
        options["limits_config"] = (
            None if args.limits_config is None else Path(args.limits_config)
        )
    if "max_tcp" in explicit_dests:
        options["max_tcp"] = (
            None if args.max_tcp is None else float(args.max_tcp)
        )
    if "terminal_exempt_steps" in explicit_dests:
        options["terminal_exempt_steps"] = (
            None if args.terminal_exempt_steps is None
            else int(args.terminal_exempt_steps)
        )
    if "dq_max_fraction" in explicit_dests:
        options["dq_max_fraction"] = (
            None if args.dq_max_fraction is None
            else float(args.dq_max_fraction)
        )
    return options


def _collect_rejected_rm65_flags(
    parser: argparse.ArgumentParser, argv: Sequence[str]
) -> list[str]:
    """收集 R1D 路径下命令行显式传入的 RM65 专属参数（不共享即拒绝）。

    判定"显式设置" = argv 中实际出现该选项（含 ``--opt=value`` 与唯一
    前缀缩写）；显式传入即拒绝，即使值与解析器缺省相同——与缺省值
    比较无法识别"用户敲了默认值"的显式输入（审查 P2：--target-speed
    1.8 / --normal-weight 500000 恰等于缺省值时曾被放行后静默忽略）。
    """
    rejected: list[str] = []
    seen: set[str] = set()
    for action in _explicit_cli_actions(parser, argv):
        if action.dest in _R1D_SHARED_FLAGS or action.dest == "help":
            continue
        option = next(
            (o for o in action.option_strings if o.startswith("--")),
            action.dest,
        )
        if option not in seen:
            seen.add(option)
            rejected.append(option)
    return rejected


def _parse_task_overrides(
    items: Sequence[str] | None,
) -> dict[str, object] | None:
    """把 ``--task-overrides KEY=VALUE`` 列表转成 T3 解析层需要的 dict。

    Args:
        items: argparse ``action="append"`` 收集的原始字符串列表。

    Returns:
        点路径 → 解析后值的 dict；未提供时返回 None。

    Raises:
        ValueError: 缺少 ``=``、键为空或值不是合法 YAML。
    """
    if not items:
        return None
    import yaml

    overrides: dict[str, object] = {}
    for item in items:
        key, sep, raw = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(
                f"--task-overrides 需要 KEY=VALUE 形式，收到 {item!r}"
            )
        try:
            overrides[key] = yaml.safe_load(raw)
        except yaml.YAMLError as exc:  # pragma: no cover - 非法 YAML 输入
            raise ValueError(
                f"--task-overrides {key} 的值不是合法 YAML: {exc}"
            ) from exc
    return overrides


def _config_or_error(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: Sequence[str],
) -> ResolvedV12Config:
    """调用 T3 配置解析器；配置错误统一走 parser.error（退出码 2）。"""
    explicit = _explicit_cli_actions(parser, argv)
    try:
        return resolve_v12_config(args, explicit)
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.error(str(exc))
        raise AssertionError("unreachable")  # pragma: no cover - parser.error 退出


def _derive_r1d_output_dir(args: argparse.Namespace) -> Path:
    """R1D 产物目录：显式 --output 优先，否则 <repo>/results/r1d_sim。"""
    if args.output is not None:
        return Path(args.output)
    return Path(__file__).resolve().parent.parent / "results" / "r1d_sim"


def _print_r1d_result(
    payload: dict[str, object], output_dir: Path
) -> None:
    """打印旧 ``__R1D_RESULT__`` 行（字段/顺序与 T0 一致）。"""
    hit = dict(payload.get("hit") or {})
    episode = dict(payload.get("episode") or {})
    print(
        "__R1D_RESULT__: "
        f"side={payload.get('side')} "
        f"hit_type={hit.get('type')} "
        f"hit_step={hit.get('step')} "
        f"total_steps={episode.get('total_steps')} "
        f"safe_steps={episode.get('safe_steps')} "
        f"failure_reason={episode.get('failure_reason')} "
        f"first_plan_ms={episode.get('first_plan_ms')} "
        f"output={output_dir}"
    )


def _view_r1d_overlay(
    config: ResolvedV12Config, result: V12EpisodeResult
) -> None:
    """R1D overlay 路径的 --viewer 回放（0.2x，产物已写出后）。"""
    from src.sim.r1d_runner import _replay_r1d_episode
    from src.sim.v12_robot_adapter import build_v12_robot

    bundle = build_v12_robot(config)
    _replay_r1d_episode(
        bundle.execution,
        result.X,
        result.ball_pos,
        result.ball_vel,
        float(result.dt),
    )


def _dispatch_robot_selection(
    parser: argparse.ArgumentParser, args: argparse.Namespace,
    argv: Sequence[str],
) -> int | None:
    """机器人分派：RM65 走共享 V12 管线，R1D 固定任务/overlay 双路径。

    Args:
        parser: V12 参数解析器（错误经 parser.error 以退出码 2 呈现）。
        args: 已解析参数（``task_overrides`` 已转换为 dict）。
        argv: 实际传入的参数列表（不含程序名；按"命令行显式出现"判定
            RM65 专属参数，避免"值等于缺省"被静默忽略）。

    Returns:
        None 表示 RM65 旧路径继续 main；
        int 为 R1D 路径完成/失败退出码（0 成功）。

    Raises:
        SystemExit: 配置类错误（缺 runtime/task、侧别错配、RM65 专属参数
            显式出现等）。
    """
    global _last_r1d_rejected_flags
    _last_r1d_rejected_flags = []

    has_r1d_config = (
        args.runtime_config is not None or args.sim_task_config is not None
    )
    if args.robot_config is None:
        if has_r1d_config:
            parser.error(
                "--runtime-config/--sim-task-config 必须与 --robot-config "
                "同时提供"
            )
        return None

    try:
        profile = load_robot_profile(Path(args.robot_config))
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.error(f"无法加载机器人描述 {args.robot_config}: {exc}")
        raise AssertionError("unreachable")  # pragma: no cover

    if profile.robot_id == "rm65":
        if has_r1d_config:
            parser.error(
                "RM65 机器人描述不接受 --runtime-config/--sim-task-config"
            )
        return None
    if profile.robot_id != "unitree_r1d":
        parser.error(
            f"不支持的机器人 robot_id={profile.robot_id!r}"
            "（仅支持 rm65 / unitree_r1d）"
        )

    rejected = _collect_rejected_rm65_flags(parser, argv)
    if rejected:
        _last_r1d_rejected_flags = list(rejected)
        parser.error(
            "R1D 仿真路径不支持 RM65 专属参数（不能接受后忽略）: "
            + ", ".join(rejected)
            + "；共享基础参数仅 seed/no-plot/viewer/output/log-level/"
            "robot-config/runtime-config/sim-task-config/racket-config/"
            "task-overrides"
        )

    output_dir = _derive_r1d_output_dir(args)

    if args.sim_task_config is not None:
        # 旧固定任务兼容模式：三件套委托旧 runner（T6 才迁产物 writer）。
        # 该模式的球初态/执行模式来自任务文件，serve/ball-speed/no-bounce/
        # position-mode 无法被旧 runner 消费，显式给出即报错（不接受后忽略）。
        if args.runtime_config is None:
            parser.error("R1D 固定任务模式须同时提供 --runtime-config")
        runtime_path = Path(args.runtime_config)
        task_path = Path(args.sim_task_config)
        if not runtime_path.is_file():
            parser.error(
                f"R1D 固定任务模式的运行参数文件不存在: {runtime_path}"
            )
        if not task_path.is_file():
            parser.error(f"R1D 固定任务模式的任务文件不存在: {task_path}")
        fixed_explicit = _explicit_cli_actions(parser, argv)
        fixed_dests = {action.dest for action in fixed_explicit}
        # 旧固定任务入口只消费 seed/viewer；球拍/任务覆盖/异步重规划在
        # run_r1d_episode 中无对应入口，显式给出必须报错（否则静默丢弃）。
        unsupported = [
            flag for flag in (
                "racket_config", "task_overrides", "async_replan",
            )
            if flag in fixed_dests
        ]
        if unsupported:
            parser.error(
                "旧固定任务入口（--sim-task-config）不支持 "
                + ", ".join(
                    "--" + flag.replace("_", "-") for flag in unsupported
                )
                + "；球拍/任务覆盖/异步重规划请改用 R1D overlay 模式"
                "（省略 --sim-task-config）"
            )
        fixed_conflicts = [
            flag for flag in (
                "serve_box", "ball_speed", "no_bounce", "position_mode",
            )
            if flag in fixed_dests
        ]
        if fixed_conflicts:
            parser.error(
                "R1D 固定任务模式（--sim-task-config）的来球初态/执行模式由"
                "任务文件决定，不接受 "
                + ", ".join(
                    "--" + flag.replace("_", "-") for flag in fixed_conflicts
                )
                + "；请改用 R1D overlay 模式（省略 --sim-task-config）"
            )
        try:
            result = run_r1d_episode(
                Path(args.robot_config),
                runtime_path,
                task_path,
                output_dir,
                seed=args.seed,
                viewer=args.viewer,
            )
        except (ValueError, FileNotFoundError, OSError) as exc:
            parser.error(str(exc))
            raise AssertionError("unreachable")  # pragma: no cover
        _print_r1d_result(result, output_dir)
        return 0

    # R1D overlay（注册表任务 + serve 覆盖 provisional 球初态）→ 共享 V12 管线
    config = _config_or_error(parser, args, argv)
    runtime_explicit = _runtime_explicit_options(parser, args, argv)
    try:
        result = run_v12_episode(config, **runtime_explicit)
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.error(str(exc))
        raise AssertionError("unreachable")  # pragma: no cover
    payload = write_v12_r1d_artifacts(result, Path(config.output_dir))
    if args.viewer:
        _view_r1d_overlay(config, result)
    _print_r1d_result(payload, Path(config.output_dir))
    return 0


def _restore_hit_window(payload: object) -> HitWindow | None:
    """把共享管线序列化的击球窗口还原为 ``HitWindow``（仅绘图用）。"""
    if not payload:
        return None
    data = dict(payload)  # type: ignore[arg-type]
    return HitWindow(
        best_k=int(data["best_k"]),
        k_candidates=np.asarray(data["k_candidates"], dtype=int),
        p_ball_candidates=np.asarray(data["p_ball_candidates"], dtype=np.float64),
        v_ball_candidates=np.asarray(data["v_ball_candidates"], dtype=np.float64),
        weights=np.asarray(data["weights"], dtype=np.float64),
    )


def _rm65_visuals_config() -> dict:
    """旧可视化函数需要的配置 dict（default→v5→mpc，仅可视化用）。"""
    repo = Path(__file__).resolve().parent.parent
    config_dict = load_config(repo / "configs" / "default.yaml")
    for name in ("v5_active_hit.yaml", "mpc.yaml"):
        path = repo / "configs" / name
        if path.exists():
            config_dict = merge_configs(config_dict, load_config(path))
    return config_dict


def _build_rm65_plot_env(config: ResolvedV12Config) -> RM65Env:
    """重建 RM65 环境（仅供绘图/回放，不参与管线数值）。"""
    repo = Path(__file__).resolve().parent.parent
    env = RM65Env(repo / "src" / "robot" / "rm65_model.xml", dt=float(config.dt))
    if config.actuator_mode == "position":
        env.configure_actuator_mode(
            "position",
            np.asarray(config.kp, dtype=np.float64),
            np.asarray(config.kd, dtype=np.float64),
        )
        # 与 v12_robot_adapter 的执行环境同源：显式 --no-feedforward 必须在
        # 回放环境生效，否则回放动力学与执行侧不一致（审查 R13 实测偏差）。
        env.configure_feedforward(
            True if config.feedforward is None else bool(config.feedforward)
        )
    return env


def _print_rm65_summary(result: V12EpisodeResult) -> None:
    """打印旧 V12 评估块 + ``__RESULT__`` 结构化行（字段/格式不变）。"""
    m = result.metrics
    p_hit = np.asarray(m["p_hit"], dtype=np.float64)
    p_ee_final = np.asarray(m["p_ee_final"], dtype=np.float64)
    histories = dict(m["histories"])
    ball_near_steps = int(sum(bool(x) for x in histories["ball_near"]))
    tube_ready_steps = int(sum(bool(x) for x in histories["tube_ready"]))
    pos_error = float(m["pos_error"])
    pos_error_plan = float(m["pos_error_plan"])
    vel_error = float(m["vel_error"])
    min_dist = float(m["min_dist"])
    ball_near_ms = float(m["ball_near_ms"])
    tube_ready_ms = float(m["tube_ready_ms"])
    hit_time_error = float(m["hit_time_error_ms"])
    hit_position_error = float(m["hit_position_error"])
    max_tcp = float(m["max_tcp_speed"])
    max_qdot = float(m["max_qdot_ratio"])
    max_face = float(m["max_racket_face_speed"])
    hit_type = "主动击球" if result.active_contact else (
        "被动接触" if result.passive_contact else "未触球"
    )
    print("\n========================================")
    print(
        "  "
        + classify_hit_result(
            result.active_contact, result.passive_contact, pos_error
        )
    )
    print("  [V12 EpisodeRunner 管线架构]")
    print(f"  ablation: {m['ablation_mode']}")
    print(f"  击打目标位置: {np.round(p_hit, 3)}")
    print(f"  末端实际位置: {np.round(p_ee_final, 3)}")
    print(f"  规划跟踪误差: {pos_error_plan:.4f} m")
    print(f"  速度误差: {vel_error:.4f} m/s")
    print(f"  MPC 总步数: {int(m['n_mpc_steps'])}")
    print("  --- Tube 专用指标 ---")
    print(f"  最小球拍-球距离: {min_dist:.4f} m")
    print(f"  ball_near 步数: {ball_near_steps} = {ball_near_ms:.1f} ms")
    print(f"  tube_ready 步数: {tube_ready_steps} = {tube_ready_ms:.1f} ms")
    print(f"  击打时间误差: {hit_time_error:.1f} ms")
    print(f"  击打位置误差: {hit_position_error:.4f} m")
    print("  --- 计算性能 ---")
    print(
        f"  总墙钟时间: {float(m['wall_time_s']):.2f}s"
        f" (MPC={float(m['mpc_time_s']):.2f}s)"
    )
    print("  --- 执行层约束 ---")
    print(
        f"  max_qdot={max_qdot:.2f}x, max_tcp={max_tcp:.1f}m/s, "
        f"max_face={max_face:.1f}m/s"
    )
    print(f"  击球类型: {hit_type}")
    print("========================================\n")

    ball_out_speed = (
        float(result.ball_out_speed)
        if result.ball_out_speed is not None else -1.0
    )
    ball_out_vy = (
        float(result.ball_out_vy) if result.ball_out_vy is not None else -99.0
    )
    v_racket_at_hit = float(m["v_racket_at_hit"])
    print(
        f"__RESULT__: pos_error={pos_error:.6f} vel_error={vel_error:.6f} "
        f"min_dist={min_dist:.6f} ball_near_ms={ball_near_ms:.1f} "
        f"tube_ready_ms={tube_ready_ms:.1f} max_tcp={max_tcp:.2f} "
        f"max_qdot={max_qdot:.2f} max_face={max_face:.1f} "
        f"hit_type={result.hit_type} "
        f"hit_time_error_ms={hit_time_error:.1f} "
        f"hit_pos_error={hit_position_error:.6f} "
        f"v_racket_at_hit={v_racket_at_hit:.3f} "
        f"first_plan_ms={float(m['first_plan_ms']):.1f} "
        f"ball_out_speed={ball_out_speed:.3f} "
        f"ball_out_vy={ball_out_vy:.3f}"
    )


def _plot_rm65_episode(
    config: ResolvedV12Config, result: V12EpisodeResult
) -> None:
    """旧 ``plot_tube_results`` 路径（仅绘图，失败只 warning）。"""
    if config.no_plot:
        return
    try:
        _, plot_fn = _import_v11_visuals()
        histories = dict(result.metrics["histories"])
        plot_fn(
            Path(__file__).resolve().parent.parent / "results",
            f"v12_{config.seed or 'default'}",
            np.asarray(result.ball_pos, dtype=np.float64),
            np.asarray(result.racket_pos, dtype=np.float64),
            _restore_hit_window(result.metrics.get("hit_window")),
            histories["distances"],
            histories["normal_align"],
            histories["ball_near"],
            histories["tube_ready"],
            [],  # k_hit_steps_history（V12 未单独追踪）
            histories["pos_error"],
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("可视化失败: %s", exc)


def _view_rm65_episode(
    config: ResolvedV12Config, result: V12EpisodeResult
) -> None:
    """旧 ``replay_trajectory`` + ``visualize_rm65_result`` 回放路径。"""
    try:
        visualize_fn, _ = _import_v11_visuals()
        from src.sim.replay import replay_trajectory

        env = _build_rm65_plot_env(config)
        serve = dict(result.metrics["serve"])
        replay_result = replay_trajectory(
            env,
            np.asarray(result.U, dtype=np.float64),
            np.asarray(INIT_Q, dtype=np.float64),
            np.asarray(INIT_Q_LEFT, dtype=np.float64),
            np.asarray(serve["p0"], dtype=np.float64),
            np.asarray(serve["v0"], dtype=np.float64),
            int(result.hit_step),
        )
        visualize_fn(
            env,
            replay_result.X_replay,
            np.asarray(result.U, dtype=np.float64),
            replay_result.ball_replay,
            _rm65_visuals_config(),
            init_q_left=np.asarray(INIT_Q_LEFT, dtype=np.float64),
            post_hit_steps=int(result.metrics["post_hit_steps"]),
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("回放可视化失败: %s", exc)


def _write_rm65_episode_artifacts(result: V12EpisodeResult) -> None:
    """写出 RM65 回合的 v2 产物（失败不阻断旧 CLI 输出/退出码）。

    Args:
        result: 共享管线回合结果（成功或失败均可）。
    """
    try:
        payload = write_episode_artifacts(result, Path(result.config.output_dir))
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("v2 产物写出失败（旧 CLI 行为不受影响）：%s", exc)
        return
    logger.info(
        "v2 产物已写出：%s（schema=%s, hit=%s, physical=%s）",
        result.config.output_dir,
        payload.get("schema"),
        payload["hit"]["legacy_hit_type"],
        payload["hit"]["physical_hit_type"],
    )


def main() -> None:
    """RM65 V12 主入口：CLI → T3 配置解析 → 共享管线 → 打印/绘图/回放。"""
    parser = _build_parser()
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    logging.getLogger().setLevel(
        getattr(logging, str(args.log_level).upper(), logging.INFO)
    )
    try:
        args.task_overrides = _parse_task_overrides(args.task_overrides)
    except ValueError as exc:
        parser.error(str(exc))

    dispatch_code = _dispatch_robot_selection(parser, args, argv)
    if dispatch_code is not None:
        raise SystemExit(dispatch_code)

    config = _config_or_error(parser, args, argv)
    if config.robot_id != "rm65":  # pragma: no cover - 分派阶段已处理 R1D
        parser.error(
            f"内部错误：非 RM65 配置 {config.robot_id!r} 未在分派阶段处理"
        )

    if bool(args.sim_limits):
        logger.warning(
            "仿真默认关节限位已由 default.yaml / 机器人描述提供，"
            "--sim-limits 已弃用（该参数不再改变行为）"
        )

    step_timer = _StepTimer() if args.dump_step_timing else None
    hooks = (step_timer.hook,) if step_timer is not None else ()
    dump_path = Path(args.dump_trajectory) if args.dump_trajectory else None
    runtime_explicit = _runtime_explicit_options(parser, args, argv)
    try:
        result = run_v12_episode(
            config,
            dump_trajectory=dump_path,
            post_exec_hooks=hooks,
            **runtime_explicit,
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.error(str(exc))
        raise AssertionError("unreachable")  # pragma: no cover

    if bool(result.metrics.get("unreachable", False)):
        serve = dict(result.metrics["serve"])
        print("\n网球不在工作空间内，机械臂不击打！")
        print(f"  球初始位置: {np.round(np.asarray(serve['p0']), 3)}")
        print(f"  球初始速度: {np.round(np.asarray(serve['v0']), 3)}")
        _write_rm65_episode_artifacts(result)
        return

    if step_timer is not None:
        print(step_timer.summary(period_ms=float(config.dt) * 1000.0))
        if args.dump_step_latencies:
            print(step_timer.latencies_line())

    _print_rm65_summary(result)
    _write_rm65_episode_artifacts(result)

    if dump_path is not None:
        logger.info(
            "轨迹已保存至 %s（.npz 格式，%d 步）", dump_path, len(result.U)
        )

    _plot_rm65_episode(config, result)

    if config.viewer:
        _view_rm65_episode(config, result)


if __name__ == "__main__":
    main()
