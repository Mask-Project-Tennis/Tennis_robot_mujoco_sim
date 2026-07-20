"""MPCController 规划模块测试。

从 V11 main() 提取的完整规划逻辑的端到端验证。
复用 runner_factory 工厂函数构建 PlanningEnv + MPCConfig + 球数据。
"""

from __future__ import annotations

import numpy as np

from src.ilqt.mpc_controller import MPCConfig, MPCController, MPCStepResult
from src.ilqt.planning_env import PlanningEnv
from src.real.config import RealRobotConfig
from src.real.runner_factory import (
    DT,
    INIT_Q,
    INIT_Q_LEFT,
    KD,
    KP,
    build_robot_limits,
)

_CFG = RealRobotConfig()


def _build_env() -> PlanningEnv:
    """构建位置模式 PlanningEnv（左臂维持零位）。"""
    env = PlanningEnv(dt=DT)
    env.init_q_left = INIT_Q_LEFT.copy()
    env.configure_actuator_mode("position", kp=KP, kd=KD)
    env.configure_feedforward(True)
    env.reset(INIT_Q)
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left
    env.update_kinematics()
    return env


def _build_config() -> MPCConfig:
    """构建测试用 MPCConfig（短 horizon 加速测试）。"""
    return MPCConfig(
        is_position_mode=True,
        dt=DT,
        total_horizon=200,
        fixed_horizon=60,
        replan_interval=20,
        max_iter_per_plan=3,
        first_plan_iters=5,
        near_plan_iters=2,
        near_threshold=80,
    )


class TestMPCControllerSmoke:
    """轮 1：start → step 烟雾测试。"""

    def test_mpc_start_step_basic(self) -> None:
        """start + step × 1 返回 MPCStepResult，不崩溃。"""
        env = _build_env()
        config = _build_config()
        mpc = MPCController(env, config, robot_limits=build_robot_limits(env, _CFG))

        ball_pos = np.array([0.0, -1.5, 1.8])
        ball_vel = np.array([0.0, 2.0, 1.0])
        arm_state = env.get_arm_state()

        mpc.start(ball_pos, ball_vel, arm_state)
        result = mpc.step(ball_pos, ball_vel, arm_state)

        assert isinstance(result, MPCStepResult)
        assert result.u_cmd.shape == (6,)
        assert result.phase in ("far", "mid", "near", "follow_through", "done")
        mpc.stop()


class TestMPCControllerReachable:
    """轮 2：可达球 → k_hit > 0, u_cmd 非零。"""

    def test_mpc_reachable_ball(self) -> None:
        """可达球应有 k_hit > 0 且 u_cmd 非零。"""
        env = _build_env()
        config = _build_config()
        mpc = MPCController(env, config, robot_limits=build_robot_limits(env, _CFG))

        ball_pos = np.array([0.0, -1.5, 1.8])
        ball_vel = np.array([0.0, 2.0, 1.0])
        arm_state = env.get_arm_state()

        mpc.start(ball_pos, ball_vel, arm_state)
        result = mpc.step(ball_pos, ball_vel, arm_state)

        assert result.k_hit > 0, "可达球应有 k_hit > 0"
        assert np.any(np.abs(result.u_cmd) > 1e-6), "u_cmd 不应全零"
        mpc.stop()


class TestMPCControllerFollowThrough:
    """轮 3：step_count > total_horizon → phase='follow_through'。"""

    def test_mpc_follow_through(self) -> None:
        """step_count 超过 mpc_horizon → 触发随挥。"""
        env = _build_env()
        config = _build_config()
        config.follow_through_steps = 5  # 启用随挥
        config.total_horizon = 15  # 短 horizon 加速测试
        mpc = MPCController(env, config, robot_limits=build_robot_limits(env, _CFG))

        # 用接近工作空间的球（确保短 horizon 内可达）
        ball_pos = np.array([0.0, -0.6, 1.4])
        ball_vel = np.array([0.0, 0.5, 0.0])
        arm_state = env.get_arm_state()

        mpc.start(ball_pos, ball_vel, arm_state)
        # 执行超过 total_horizon 步
        for _ in range(config.total_horizon + 1):
            result = mpc.step(ball_pos, ball_vel, arm_state)
            if result.phase == "follow_through":
                break
        assert result.phase == "follow_through"
        mpc.stop()


class TestMPCControllerUnreachable:
    """轮 4：远球 → ball_unreachable=True 或 done=True。"""

    def test_mpc_unreachable(self) -> None:
        """球在 [10,10,10]（远离工作空间）→ 不可达。"""
        env = _build_env()
        config = _build_config()
        mpc = MPCController(env, config, robot_limits=build_robot_limits(env, _CFG))

        ball_far = np.array([10.0, 10.0, 10.0])
        ball_vel_zero = np.array([0.0, 0.0, 0.0])
        arm_state = env.get_arm_state()

        mpc.start(ball_far, ball_vel_zero, arm_state)
        # start 可能就标记不可达
        if not mpc.done:
            result = mpc.step(ball_far, ball_vel_zero, arm_state)
            assert result.ball_unreachable or mpc.done
        else:
            assert mpc.done
        mpc.stop()


class TestDependencyDirection:
    """轮 5：依赖方向修复 — build_solver 迁入 src.ilqt.solver，robot_limits 必传。"""

    def test_build_solver_importable_from_ilqt(self) -> None:
        """build_solver 可从 src.ilqt.solver 导入（移出 runner_factory）。"""
        from src.ilqt.solver import build_solver

        solver = build_solver()
        assert solver is not None

    def test_robot_limits_injected(self) -> None:
        """MPCController 接受并使用注入的 robot_limits（Minor #4）。"""
        env = _build_env()
        config = _build_config()
        custom_rl = build_robot_limits(env, _CFG)
        mpc = MPCController(env, config, robot_limits=custom_rl)
        assert mpc._robot_limits is custom_rl
        mpc.stop()

    def test_no_src_real_imports_in_mpc_controller(self) -> None:
        """src.ilqt.mpc_controller 不应导入 src.real（依赖方向）。"""
        import ast
        from pathlib import Path

        source = Path("src/ilqt/mpc_controller.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("src.real"):
                    violations.append(node.module)
        assert not violations, f"发现 src.real 导入: {violations}"
