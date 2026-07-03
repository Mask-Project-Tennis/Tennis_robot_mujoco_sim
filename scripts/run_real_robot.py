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
from src.real.fake_robot import FakeRobot
from src.real.real_runner import RealRunner
from src.real.runner_factory import (
    INIT_Q,
    INIT_Q_LEFT,
    build_replan_cfg,
    build_robot_limits,
    build_solver,
)
from src.real.safety_monitor import SafetyMonitor

logger = logging.getLogger(__name__)

# Mock 球初始位置 + 来球方向单位向量（[0,2,1] 归一化）
_DEFAULT_BALL_POS = np.array([0.0, -1.5, 1.8], dtype=np.float64)
_BALL_VEL_DIR = np.array([0.0, 2.0, 1.0], dtype=np.float64)
_BALL_VEL_DIR /= float(np.linalg.norm(_BALL_VEL_DIR))
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "real_robot.yaml"


def create_runner(
    config: RealRobotConfig,
    mock: bool = True,
    ball_pos: np.ndarray | None = None,
    ball_vel: np.ndarray | None = None,
) -> RealRunner:
    """组装 RealRunner 的全部依赖组件并返回。

    所有可调参数从 config（YAML 加载）读取，不使用硬编码常量。

    Args:
        config: 真机配置（YAML 为唯一真相源）。
        mock: True 使用 FakeRobot；False 使用 RobotInterface（真机 SDK）。
        ball_pos: 初始球位置，默认 [0, -1.5, 1.8]（可达候选点）。
        ball_vel: 初始球速度，默认 ball_speed=3.0 m/s 沿 [0,2,1] 方向。

    Returns:
        已组装但尚未 start 的 RealRunner 实例。
    """
    if ball_pos is None:
        ball_pos = _DEFAULT_BALL_POS.copy()
    if ball_vel is None:
        ball_vel = 3.0 * _BALL_VEL_DIR.copy()
    ball_pos = np.asarray(ball_pos, dtype=np.float64)
    ball_vel = np.asarray(ball_vel, dtype=np.float64)

    # 1. 规划环境（位置模式）— dt/kp/kd 从 config 读取
    env = PlanningEnv(dt=config.dt)
    env.init_q_left = INIT_Q_LEFT.copy()
    env.configure_actuator_mode("position", kp=config.kp, kd=config.kd)
    env.configure_feedforward(True)
    env.reset(INIT_Q)
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left

    # 2. 机器人接口
    if mock:
        robot = FakeRobot(init_q=INIT_Q, dt=config.dt)
    else:
        from src.real.robot_interface import RobotInterface

        robot = RobotInterface(config)
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

    # 6. 规划方向（来球反方向）
    ball_vel_norm = float(np.linalg.norm(ball_vel))
    if ball_vel_norm > 1e-6:
        d_hat = -ball_vel / ball_vel_norm
    else:
        d_hat = np.array([0.0, 1.0, 0.0])
    v_hit_desired = config.target_hit_speed * d_hat

    # 7. 规划配置
    solver = build_solver()
    replan_cfg = build_replan_cfg(env, robot_limits, solver, d_hat, v_hit_desired, config)

    # 8. 重规划状态
    replan_state = ReplanState(
        k_hit_new=replan_cfg["k_hit_total"],
        p_hit_new=ball_pos.copy(),
        v_ball_hit_new=ball_vel.copy(),
        current_n_des=d_hat.copy(),
        U_prev=np.zeros((0, env.NU)),
        is_first_plan=True,
    )

    # 9. 异步重规划器
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
    replanner = AsyncReplanner(
        env, do_replan, replan_cfg, state=replan_state, model_path=model_path
    )

    # 10. 自适应定时器（100Hz 目标频率，匹配 SDK 实测吞吐上限）
    timer = AdaptiveTimer(target_hz=100.0)

    return RealRunner(
        env=env,
        robot=robot,
        ball_perceiver=perceiver,
        safety=safety,
        replanner=replanner,
        replan_cfg=replan_cfg,
        timer=timer,
        replan_state=replan_state,
    )


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
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
    return parser.parse_args()


def main() -> None:
    """入口主函数：配置日志 → 构建 runner → 主循环 → 输出指标。"""
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
