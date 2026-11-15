"""真机部署入口脚本 — 真实 RM-65B 双臂网球击打。

组装 PlanningEnv + 机器人接口 + 球感知 + 安全监控 + 异步重规划器，
驱动 RealRunner 的 start/step/stop 分步主循环。

用法:
    # Mock 模式（不接真机，用 FakeRobot + SimulatedBallSensor 冒烟测试）
    python scripts/run_real_robot.py --mock --max-steps 10

    # 真机模式（待硬件就绪后实现）
    python scripts/run_real_robot.py --ball-speed 3.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# 支持从 scripts/ 目录直接运行，确保 `from src.xxx` 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ilqt.async_replanner import AsyncReplanner
from src.ilqt.planning_env import PlanningEnv
from src.ilqt.replan_core import do_replan
from src.ilqt.tube_types import ReplanState
from src.real.adaptive_timer import AdaptiveTimer
from src.real.ball_perceiver import BallPerceiver
from src.real.ball_sensor import SimulatedBallSensor
from src.real.config import RealRobotConfig
from src.real.real_runner import RealRunner
from src.real.runner_factory import (
    INIT_Q,
    INIT_Q_LEFT,
    ModelBinding,
    build_real_robot_mpc_config,
    build_robot_arm,
    build_robot_limits,
    build_solver,
    load_validated_rm65_binding,
    validate_rm65_profile,
)
from src.real.safety_monitor import SafetyMonitor
from src.robot.robot_profile import (
    RobotProfile,
    default_rm65_profile_path,
    load_robot_profile,
)

logger = logging.getLogger(__name__)

# Mock 球初始位置 + 来球方向单位向量（[0,2,1] 归一化）
_DEFAULT_BALL_POS = np.array([0.0, -1.5, 1.8], dtype=np.float64)
_BALL_VEL_DIR = np.array([0.0, 2.0, 1.0], dtype=np.float64)
_BALL_VEL_DIR /= float(np.linalg.norm(_BALL_VEL_DIR))
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "real_robot.yaml"


def _resolve_profile(profile: RobotProfile | None) -> tuple[RobotProfile, ModelBinding]:
    """解析并校验机器人描述（身份/名称预检 + 加载绑定 + 运行能力校验）。

    纯模型加载，不启动线程、不接触硬件；在任何驱动构造或
    传感器启动之前调用，使配置错误尽早暴露。先做廉价的纯描述
    预检（validate_rm65_profile）：不受支持的机器人（如阶段 2 的 D2）
    在加载模型之前即得到"仅支持 RM65"的明确错误，而不是 loader
    的模型缺失诊断。

    Args:
        profile: 显式描述；None 时加载默认 configs/robots/rm65.yaml。

    Returns:
        (通过运行能力校验的描述, 对应的校验绑定)。
        绑定复用给 build_robot_arm，避免真机路径重复加载模型。

    Raises:
        ValueError: 描述非法或不受当前入口支持（仅 RM65）。
    """
    if profile is None:
        profile = load_robot_profile(default_rm65_profile_path())
    validate_rm65_profile(profile)
    binding = load_validated_rm65_binding(profile)
    return profile, binding


def create_runner(
    config: RealRobotConfig,
    mock: bool = True,
    ball_pos: np.ndarray | None = None,
    ball_vel: np.ndarray | None = None,
    profile: RobotProfile | None = None,
) -> RealRunner:
    """组装 RealRunner 的全部依赖组件并返回。

    所有可调参数从 config（YAML 加载）读取，不使用硬编码常量。

    Args:
        config: 真机配置（YAML 为唯一真相源）。
        mock: True 使用 FakeRobot；False 使用 RobotInterface（真机 SDK）。
        ball_pos: 初始球位置，默认 [0, -1.5, 1.8]（可达候选点）。
        ball_vel: 初始球速度，默认 ball_speed=3.0 m/s 沿 [0,2,1] 方向。
        profile: 机器人描述；None 时加载默认 RM65 描述（行为与旧入口一致）。

    Returns:
        已组装但尚未 start 的 RealRunner 实例。
    """
    # 0. 解析并校验机器人描述（在任何驱动构造/传感器启动之前）；
    #    校验绑定复用给驱动工厂，真机路径不再重复加载模型
    profile, binding = _resolve_profile(profile)

    if ball_pos is None:
        ball_pos = _DEFAULT_BALL_POS.copy()
    if ball_vel is None:
        ball_vel = 3.0 * _BALL_VEL_DIR.copy()
    ball_pos = np.asarray(ball_pos, dtype=np.float64)
    ball_vel = np.asarray(ball_vel, dtype=np.float64)

    # 1. 规划环境（位置模式）— 模型路径来自已校验 profile；dt/kp/kd 从 config 读取
    env = PlanningEnv(profile.model_path, dt=config.dt)
    env.init_q_left = INIT_Q_LEFT.copy()
    env.configure_actuator_mode("position", kp=config.kp, kd=config.kd)
    env.configure_feedforward(True)
    env.reset(INIT_Q)
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left

    # 2. 机器人接口（工厂构造，不连接硬件；复用第 0 步的校验绑定）
    robot = build_robot_arm(
        config, profile=profile, mock=mock, init_q=INIT_Q.copy(), binding=binding
    )
    if not mock:
        logger.info("create_runner: 使用真机接口 RobotInterface (mock=False)")

    # 3. 球传感器 + 启动
    sensor = SimulatedBallSensor()
    sensor.start()

    # 4. 球感知器（bootstrap：推两次观测 + 两次 update，使有限差分速度可用）
    perceiver = BallPerceiver(sensor, estimator_config=None, dt=config.dt)
    sensor.push(ball_pos, 0.0)
    perceiver.update()
    t_obs = 0.02
    p_obs = ball_pos + ball_vel * t_obs
    sensor.push(p_obs, t_obs)
    perceiver.update()

    # 5. 安全监控 — 直接用 config（config 即真相源）
    robot_limits = build_robot_limits(env, config)
    safety = SafetyMonitor(config, robot=robot)

    # 6. 规划方向（来球反方向，用于 replan_state.current_n_des）
    ball_vel_norm = float(np.linalg.norm(ball_vel))
    if ball_vel_norm > 1e-6:
        d_hat = -ball_vel / ball_vel_norm
    else:
        d_hat = np.array([0.0, 1.0, 0.0])

    # 7. 规划配置：构建 MPCConfig + 直接传给 RealRunner（无翻译层）
    solver = build_solver()
    mpc_config = build_real_robot_mpc_config(config)

    # 8. 重规划状态
    replan_state = ReplanState(
        k_hit_new=mpc_config.total_horizon,
        p_hit_new=ball_pos.copy(),
        v_ball_hit_new=ball_vel.copy(),
        current_n_des=d_hat.copy(),
        U_prev=np.zeros((0, env.NU)),
        is_first_plan=True,
    )

    # 9. 异步重规划器（静态注入 config/robot_limits/solver；
    #    与环境引用同一已校验 profile.model_path）
    replanner = AsyncReplanner(
        env, do_replan, mpc_config, robot_limits, solver,
        state=replan_state, model_path=profile.model_path,
    )

    # 10. 自适应定时器（100Hz 目标频率，匹配 SDK 实测吞吐上限）
    timer = AdaptiveTimer(target_hz=100.0)

    return RealRunner(
        env=env,
        robot=robot,
        ball_perceiver=perceiver,
        safety=safety,
        replanner=replanner,
        config=mpc_config,
        robot_limits=robot_limits,
        solver=solver,
        timer=timer,
        replan_state=replan_state,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数（argv 供测试注入）。"""
    parser = argparse.ArgumentParser(
        description="RM-65B 真机部署入口 — MPC+iLQR+Tube 网球击打主循环",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG_PATH),
        help="真机配置 YAML 路径",
    )
    parser.add_argument(
        "--robot-config",
        type=str,
        default=None,
        help="机器人描述 YAML 路径（默认 configs/robots/rm65.yaml）",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock 模式（不接真机，用 FakeRobot + SimulatedBallSensor）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="主循环最大运行步数（独立于规划时长 total_horizon）",
    )
    parser.add_argument(
        "--ball-speed",
        type=float,
        default=3.0,
        help="模拟球速 m/s（Mock 模式下缩放初始球速度）",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="禁用可视化（兼容 V11 参数，真机入口暂不渲染）",
    )
    return parser.parse_args(argv)


def main() -> None:
    """入口主函数：配置日志 → 校验机器人描述 → 构建 runner → 主循环 → 输出指标。"""
    args = _parse_args()

    # 日志配置（单进程，force=True 覆盖可能存在的基本配置）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    logger.info(
        "启动真机部署入口: mock=%s max_steps=%d ball_speed=%.2f",
        args.mock, args.max_steps, args.ball_speed,
    )

    # 机器人描述：显式 --robot-config 或默认 RM65。
    # 在任何驱动构造/传感器/后台线程之前完成校验（重复一次纯模型加载，可接受）。
    # 配置类错误（文件缺失/字段非法/不受支持）以明确日志 + 退出码 2 呈现，
    # 不向操作员抛 traceback。
    try:
        profile_arg: RobotProfile | None = None
        if args.robot_config is not None:
            profile_arg = load_robot_profile(Path(args.robot_config))
        profile, _ = _resolve_profile(profile_arg)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("机器人描述加载/校验失败: %s", exc)
        sys.exit(2)
    logger.info("机器人描述已校验: robot_id=%s model=%s",
                profile.robot_id, profile.model_path)

    # 加载配置
    config = RealRobotConfig.from_yaml(args.config)
    logger.info("配置已加载: %s (dt=%.3f tcp=%.1f hit=%.1f)",
                args.config, config.dt, config.max_tcp_speed, config.target_hit_speed)

    # 根据 ball_speed 生成初始球速度（方向固定 [0,2,1] 归一化）
    ball_vel = args.ball_speed * _BALL_VEL_DIR

    # 构建 runner
    runner = create_runner(
        config=config,
        mock=args.mock,
        ball_vel=ball_vel,
        profile=profile,
    )

    # 主循环（max_steps 独立于规划时长 total_horizon，仅控制 episode 实际运行步数）
    runner.start()
    if runner.ball_unreachable:
        logger.warning("首次规划判定球不可达，episode 提前结束")
    else:
        step_count = 0
        while not runner.done and step_count < args.max_steps:
            info = runner.step()
            step_count += 1
            if not info.get("safe", True):
                logger.warning("步 %d 安全检查失败，退出主循环", info.get("step", -1))
                break

    # 清理 + 指标
    metrics = runner.stop()

    # 结果打印（允许使用 print，便于终端直接查看）
    print(
        f"总步数: {metrics['total_steps']}, "
        f"安全步数: {metrics['safe_steps']}, "
        f"球不可达: {metrics['ball_unreachable']}, "
        f"安全失败: {metrics['safety_failed']}, "
        f"重规划提交: {metrics['replan_submit_count']}, "
        f"重规划完成: {metrics['replan_complete_count']}"
    )


if __name__ == "__main__":
    main()
