"""多机器人 MPC 消费链测试（阶段 3 · Task 3.8）。

核心断言组（brief + 施工计划 Task 3.8 / §Global Constraints）：
1. 七维 warm-start：力矩 JT / 位置 JT 在 R1D（n=7）上输出 (N,7) 有限解，
   控制界来自环境（env.control_range，按 binding 执行器序）；
2. R1D 策略决定：不用 fix_joint5、不继承"索引 0 必是后摆轴"（后摆
   warm-start 明确拒绝）、R 退火均匀（无关节 1 特殊衰减）；
   RM65（PlanningEnv，无 binding）同接口保持原行为（回归锁定）；
3. 工作空间中心/半径显式来自当前侧配置（task YAML）并对模型校验
   （中心 IK 可达 + 限位内），不能读 RM65 SHOULDER_POS；
4. IK 结果检查：solve_ik_checked 校验误差与限位，失败报告（ValueError）
   而不是带坏初值继续；随挥位置模式 IK 失败回退 hold 并告警；
5. 同步七维短规划：有限解、成本不发散；滚动轨迹移位（buffer 消费 +
   重规划触发）；消费预测结果后显式恢复臂状态（不依赖 step_from_state
   副作用）；
6. 同序重排关节/执行器/参数后输出物理意义一致（JT warm-start 与
   1 次迭代 iLQR 均按重排列反演一致）；
7. async 决定：本阶段不实现 R1D 异步 —— 入口（MPCController 构造 /
   AsyncReplanner 直接构造）明确拒绝，不静默改同步；
8. use_fast_lin=True + FD 后端回退分支补测（3.5 评审指针）：
   与 use_fast_lin=False（完整 FD）结果一致。

测试环境：R1D 左侧 profile + 阶段 3 仿真 runtime + 任务配置（真实
YAML 文件，provisional 数值不在此复制）；RM65 回归对照用 PlanningEnv。
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from src.ilqt.async_replanner import AsyncReplanner, PlanRequest
from src.ilqt.cost import build_production_cost
from src.ilqt.jt_init import (
    compute_jacobian_init_control_position,
    solve_ik_checked,
)
from src.ilqt.mpc_controller import MPCConfig, MPCController, MPCStepResult
from src.ilqt.mpc_helpers import (
    _backswing_dispatch,
    _fix_joint5_dispatch,
    _jt_init_dispatch,
    compute_jacobian_init_control,
    compute_r_schedule,
    validate_workspace_against_model,
)
from src.ilqt.profile_planning_env import ProfilePlanningEnv
from src.ilqt.replan_core import _resolve_cost_bases, do_replan
from src.ilqt.robot_limits import RobotLimits
from src.ilqt.solver import ILQTSolver, build_solver
from src.ilqt.strategies.follow_through import FollowContext, PlannedFollowThrough
from src.robot.model_factory import load_robot_model
from src.robot.robot_profile import RobotProfile, load_robot_profile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    load_runtime_parameters,
)
from src.sim.sim_config import SimTaskConfig, load_sim_task_config

REPO_ROOT = Path(__file__).resolve().parent.parent

_PROFILE_YAML = REPO_ROOT / "configs" / "robots" / "r1d_left.yaml"
_SIM_RUNTIME_YAML = REPO_ROOT / "configs" / "robots" / "r1d_sim_runtime_left.yaml"
_TASK_YAML = REPO_ROOT / "configs" / "r1d_sim_left.yaml"

# 重排（同序重排一致性测试用）：B 的第 k 关节 = A 的第 PERM[k] 关节
_PERM = (3, 0, 5, 1, 6, 2, 4)


# ---------- 只读配置加载（进程内缓存） ----------


@lru_cache(maxsize=1)
def _profile() -> RobotProfile:
    """加载左侧 RobotProfile。"""
    return load_robot_profile(_PROFILE_YAML)


@lru_cache(maxsize=1)
def _parameters() -> RobotRuntimeParameters:
    """加载左侧仿真 runtime 参数（完整显式，provisional）。"""
    return load_runtime_parameters(_SIM_RUNTIME_YAML, _profile())


@lru_cache(maxsize=1)
def _task() -> SimTaskConfig:
    """加载左侧 SimTaskConfig。"""
    return load_sim_task_config(_TASK_YAML, _profile())


# ---------- 测试辅助构造 ----------


def _planning_env() -> ProfilePlanningEnv:
    """构建左侧规划环境（裸模型 + 虚拟球，力矩模式构造默认）。"""
    bundle = load_robot_model(_profile(), _parameters())
    return ProfilePlanningEnv(bundle)


def _make_limits(env: ProfilePlanningEnv) -> RobotLimits:
    """从环境同源参数构造 7 维 RobotLimits（不依赖 3.7 的新工厂）。"""
    params = env._parameters  # noqa: SLF001 — 同源参数，测试内构造限位
    return RobotLimits(
        q_lower=np.array(params.q_lower, dtype=np.float64),
        q_upper=np.array(params.q_upper, dtype=np.float64),
        qdot_max=np.array(params.qdot_max, dtype=np.float64),
        qddot_max=np.array(params.qddot_max, dtype=np.float64),
        u_min=-np.array(params.torque_max, dtype=np.float64),
        u_max=np.array(params.torque_max, dtype=np.float64),
        max_tcp_speed=float(params.max_tcp_speed),
    )


def _make_config(**overrides: object) -> MPCConfig:
    """构建 R1D 测试用 MPCConfig（工作空间显式来自当前侧 task 配置）。"""
    task = _task()
    values: dict[str, object] = dict(
        dt=0.005,
        total_horizon=200,
        fixed_horizon=16,
        replan_interval=10,
        max_iter_per_plan=2,
        first_plan_iters=2,
        near_plan_iters=2,
        near_threshold=80,
        far_threshold=50,
        R=1.0e-4,
        normal_weight=2.0e5,
        racket_speed=1.2,
        target_speed=1.2,
        shoulder_pos=np.array(task.workspace_center, dtype=np.float64),
        workspace_radius=float(task.workspace_radius),
        cost_config=dict(task.cost_config),
    )
    values.update(overrides)
    return MPCConfig(**values)


def _step_ball_analytic(
    ball_pos: np.ndarray, ball_vel: np.ndarray, dt: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """解析推进球一步（重力抛物线，镜像真实感知回路的来球推进）。

    控制器测试不能把球冻结在初位置——find_hitting_point_physics 按
    remaining_horizon 预测，球不前进会在窗口收缩后判不可达。
    """
    pos = ball_pos + ball_vel * dt
    vel = ball_vel.copy()
    vel[2] -= 9.81 * dt
    return pos, vel


def _first_plan_request(env: ProfilePlanningEnv) -> PlanRequest:
    """构建来自 task YAML 球初态的首次规划请求。"""
    task = _task()
    x0 = env.get_arm_state()
    return PlanRequest(
        x_current=x0.copy(),
        ball_pos=np.array(task.initial_ball_position, dtype=np.float64),
        ball_vel=np.array(task.initial_ball_velocity, dtype=np.float64),
        step=0,
        k_hit_current=200,
        U_prev=np.zeros((0, env.NU)),
        p_hit_current=np.array(task.initial_ball_position, dtype=np.float64),
        v_hit_desired=np.array([1.0, 0.0, 0.0]),
        n_des_current=np.array([1.0, 0.0, 0.0]),
        d_hat=np.array([1.0, 0.0, 0.0]),
        is_first_plan=True,
    )


# ==============================================================================
# 1. 七维 warm-start
# ==============================================================================


class TestSevenDimWarmStart:
    """七维 JT warm-start：形状/有限性/控制界/趋近目标。"""

    def test_torque_jt_shape_and_bounds(self) -> None:
        """力矩 JT：输出 (N,7)，有限且在 env.control_range（±torque_max）内。"""
        env = _planning_env()
        x0 = env.get_arm_state()
        p_hit = np.array(_task().workspace_center, dtype=np.float64)

        U = compute_jacobian_init_control(env, x0, p_hit, horizon=8)

        assert U.shape == (8, env.NU)
        assert np.all(np.isfinite(U))
        cr = env.control_range
        assert np.all(U >= cr[:, 0] - 1e-12)
        assert np.all(U <= cr[:, 1] + 1e-12)

    def test_torque_jt_moves_toward_target(self) -> None:
        """力矩 JT：仿真后末端向击球点趋近（误差单调下降量级）。"""
        env = _planning_env()
        x0 = env.get_arm_state()
        p_hit = np.array(_task().workspace_center, dtype=np.float64)
        env.set_arm_state(x0)
        err_before = np.linalg.norm(env.get_ee_pos() - p_hit)

        U = compute_jacobian_init_control(env, x0, p_hit, horizon=12)
        x = x0.copy()
        for k in range(len(U)):
            x = env.step_from_state(x, U[k])
        env.set_arm_state(x)
        err_after = np.linalg.norm(env.get_ee_pos() - p_hit)

        assert err_after < err_before

    def test_position_jt_shape_and_bounds(self) -> None:
        """位置 JT：输出 q_desired (N,7)，裁剪到关节限位（control_range）。"""
        env = _planning_env()
        params = env._parameters  # noqa: SLF001
        env.configure_actuator_mode(
            "position",
            np.array(params.kp, dtype=np.float64),
            np.array(params.kd, dtype=np.float64),
        )
        env.reset()
        x0 = env.get_arm_state()
        p_hit = np.array(_task().workspace_center, dtype=np.float64)

        U = compute_jacobian_init_control_position(env, x0, p_hit, horizon=8)

        assert U.shape == (8, env.NQ)
        assert np.all(np.isfinite(U))
        cr = env.control_range  # 位置模式 = 关节 jnt_range
        assert np.all(U >= cr[:, 0] - 1e-12)
        assert np.all(U <= cr[:, 1] + 1e-12)

    def test_position_jt_converges(self) -> None:
        """位置 JT：多步后末端位置误差显著缩小。"""
        env = _planning_env()
        params = env._parameters  # noqa: SLF001
        env.configure_actuator_mode(
            "position",
            np.array(params.kp, dtype=np.float64),
            np.array(params.kd, dtype=np.float64),
        )
        env.reset()
        x0 = env.get_arm_state()
        p_hit = np.array(_task().workspace_center, dtype=np.float64)
        env.set_arm_state(x0)
        err_before = np.linalg.norm(env.get_ee_pos() - p_hit)

        U = compute_jacobian_init_control_position(env, x0, p_hit, horizon=20)
        x = x0.copy()
        for k in range(len(U)):
            x = env.step_from_state(x, U[k])
        env.set_arm_state(x)
        err_after = np.linalg.norm(env.get_ee_pos() - p_hit)

        assert err_after < 0.5 * err_before


# ==============================================================================
# 2. R1D 策略决定（fix_joint5 / 后摆 / R 退火）+ RM65 回归锁定
# ==============================================================================


class TestR1DPolicyDecisions:
    """R1D 不继承 RM65 专属策略；RM65 原行为保持。"""

    def test_fix_joint5_rejected_on_bound_env(self) -> None:
        """R1D（绑定环境）+ fix_joint5 → ValueError（明确拒绝）。"""
        env = _planning_env()
        x0 = env.get_arm_state()
        p_hit = np.array(_task().workspace_center, dtype=np.float64)
        with pytest.raises(ValueError, match="fix_joint5"):
            _jt_init_dispatch(env, x0, p_hit, 4, fix_joint5_angle=0.1)
        with pytest.raises(ValueError, match="fix_joint5"):
            _fix_joint5_dispatch(
                np.zeros((4, env.NU)), x0, env, 0.1,
            )

    def test_backswing_rejected_on_bound_env(self) -> None:
        """R1D 不继承"索引 0 必是后摆轴"：后摆 warm-start 明确拒绝。"""
        env = _planning_env()
        x0 = env.get_arm_state()
        p_hit = np.array(_task().workspace_center, dtype=np.float64)
        with pytest.raises(ValueError, match="后摆"):
            _backswing_dispatch(
                env, x0, p_hit, np.array([1.0, 0.0, 0.0]), 4,
            )

    def test_rm65_fix_joint5_and_backswing_still_work(self) -> None:
        """RM65（PlanningEnv，无 binding）：fix_joint5/后摆保持原行为。"""
        from src.ilqt.planning_env import PlanningEnv
        from src.real.runner_factory import DT, INIT_Q, INIT_Q_LEFT, KD, KP

        env = PlanningEnv(dt=DT)
        env.init_q_left = INIT_Q_LEFT.copy()
        env.configure_actuator_mode("position", kp=KP, kd=KD)
        env.reset(INIT_Q)
        env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left
        env.update_kinematics()

        x0 = env.get_arm_state()
        env.set_arm_state(x0)
        p_hit = env.get_ee_pos() + np.array([0.02, 0.02, 0.0])

        # 位置模式 JT + fix_joint5：第 6 关节被钉住
        U = _jt_init_dispatch(env, x0, p_hit, 4, fix_joint5_angle=0.1)
        assert U.shape == (4, 6)
        assert np.allclose(U[:, 5], 0.1)

        # 位置模式后摆 warm-start：仍可生成（关节 0 五次多项式路径）
        U_bs, q_des = _backswing_dispatch(
            env, x0, p_hit, np.array([0.0, 1.0, 0.0]), 4,
        )
        assert U_bs.shape == (4, 6)
        assert q_des.shape == (4, 6)

    def test_r_schedule_uniform_for_r1d(self) -> None:
        """R1D：R 退火均匀（无关节 1 特殊衰减），形状 (N, n)。"""
        sched = compute_r_schedule(40, 1.0e-4, n_joints=7, uniform_decay=True)
        assert sched.shape == (40, 7)
        # 所有关节同列（均匀）
        for j in range(1, 7):
            assert np.allclose(sched[:, j], sched[:, 0])
        # 退火末端趋近 0
        assert sched[-1, 0] < 0.05 * 1.0e-4 + 1e-12

    def test_r_schedule_rm65_default_unchanged(self) -> None:
        """RM65 缺省：形状 (N,6)，关节 1（索引 0）保留特殊衰减。"""
        sched = compute_r_schedule(40, 1.0e-4, decay_ratio=0.40)
        assert sched.shape == (40, 6)
        x = np.linspace(0.0, 1.0, 40 - 24)  # decay_start=40*(1-0.4)=24
        expected_joint0 = 1.0e-4 * (1.0 - x) ** 10.0
        expected_other = 1.0e-4 * (1.0 - x)
        assert np.allclose(sched[24:, 0], expected_joint0)
        for j in range(1, 6):
            assert np.allclose(sched[24:, j], expected_other)
        assert np.allclose(sched[:24], 1.0e-4)


# ==============================================================================
# 3. 工作空间显式配置 + 模型校验
# ==============================================================================


class TestWorkspaceValidation:
    """工作空间中心/半径来自当前侧配置并对模型校验。"""

    def test_valid_side_config_passes(self) -> None:
        """task YAML 的工作区中心（IK 可达 + 限位内）通过校验。"""
        env = _planning_env()
        task = _task()
        validate_workspace_against_model(
            env,
            np.array(task.workspace_center, dtype=np.float64),
            float(task.workspace_radius),
            robot_limits=_make_limits(env),
        )

    def test_unreachable_center_rejected(self) -> None:
        """中心远离模型可达域 → ValueError（报告而非静默）。"""
        env = _planning_env()
        with pytest.raises(ValueError, match="工作空间中心"):
            validate_workspace_against_model(
                env, np.array([5.0, 5.0, 5.0]), 0.45,
                robot_limits=_make_limits(env),
            )

    def test_nonpositive_radius_rejected(self) -> None:
        """半径非正 → ValueError。"""
        env = _planning_env()
        task = _task()
        with pytest.raises(ValueError, match="半径"):
            validate_workspace_against_model(
                env,
                np.array(task.workspace_center, dtype=np.float64),
                0.0,
                robot_limits=_make_limits(env),
            )

    def test_controller_rejects_bad_workspace(self) -> None:
        """MPCController 构造时用错误工作区 → ValueError（R1D 路径校验）。"""
        env = _planning_env()
        bad = _make_config(
            shoulder_pos=np.array([5.0, 5.0, 5.0]),
        )
        with pytest.raises(ValueError, match="工作空间中心"):
            MPCController(env, bad, robot_limits=_make_limits(env))


# ==============================================================================
# 4. IK 结果检查（误差 + 限位，失败报告）
# ==============================================================================


class TestIKChecked:
    """solve_ik_checked：误差与限位校验，失败报告。"""

    def test_reachable_target_within_limits(self) -> None:
        """可达目标：返回 q 有限、误差 < 容差、关节限位内。"""
        env = _planning_env()
        limits = _make_limits(env)
        task = _task()
        target = np.array(task.workspace_center, dtype=np.float64)

        q = solve_ik_checked(
            env, target, q_init=env.get_arm_state()[: env.NQ],
            robot_limits=limits, pos_tol=0.02,
        )

        assert q.shape == (env.NQ,)
        assert np.all(np.isfinite(q))
        assert np.all(q >= limits.q_lower - 1e-9)
        assert np.all(q <= limits.q_upper + 1e-9)
        env.set_arm_state(np.concatenate([q, np.zeros(env.NQ)]))
        assert np.linalg.norm(env.get_ee_pos() - target) < 0.02

    def test_unreachable_target_reports(self) -> None:
        """不可达目标：ValueError 报告（含误差信息），不带坏解继续。"""
        env = _planning_env()
        with pytest.raises(ValueError, match="误差"):
            solve_ik_checked(
                env, np.array([5.0, 5.0, 5.0]),
                q_init=env.get_arm_state()[: env.NQ],
                robot_limits=_make_limits(env), pos_tol=0.02,
            )


# ==============================================================================
# 5. 同步七维短规划 + 滚动轨迹移位 + 重规划
# ==============================================================================


class TestSevenDimReplan:
    """do_replan 七维链路：有限解、成本不发散、显式恢复臂状态。"""

    def test_first_plan_finite_and_cost_nondecreasing(self) -> None:
        """首次规划：解有限；直接求解器成本不发散。"""
        env = _planning_env()
        limits = _make_limits(env)
        config = _make_config()
        solver = build_solver(env=env, backend="python")
        assert isinstance(solver, ILQTSolver)
        assert solver.use_analytical is False  # R1D 强制 Python FD

        task = _task()
        request = _first_plan_request(env)
        from src.ilqt.tube_types import ReplanState

        state = ReplanState(
            k_hit_new=200,
            p_hit_new=np.array(task.initial_ball_position, dtype=np.float64),
            v_ball_hit_new=np.array(task.initial_ball_velocity, dtype=np.float64),
            current_n_des=np.array([1.0, 0.0, 0.0]),
            U_prev=np.zeros((0, env.NU)),
            is_first_plan=True,
        )
        result = do_replan(request, env, state, config, limits, solver)

        assert not result.ball_unreachable
        assert result.k_hit_new > 0
        assert result.U_mpc_full.shape[1] == env.NU
        assert np.all(np.isfinite(result.U_mpc_full))
        assert np.all(np.isfinite(result.U_buffer))

        # 消费预测结果后显式恢复臂状态（不依赖 step_from_state 副作用）
        assert np.allclose(env.get_arm_state(), request.x_current, atol=1e-9)

        # 成本不发散：短视野直接求解（首算成本 vs 终算成本有限且受控）
        p_follow = result.p_hit_new + config.follow_through_length * request.d_hat
        cost_fn = build_production_cost(
            env, config, limits,
            p_hit=p_follow, v_hit=request.v_hit_desired,
            n_des=result.n_des_new,
            Q_p_mat=np.eye(3) * 2000.0, Q_v_mat=np.eye(3) * 100.0,
            cost_config=None,
        )
        x0 = request.x_current
        U_warm = compute_jacobian_init_control(
            env, x0, p_follow, horizon=6, gain=30.0,
        )[:6]
        X1, U1, c1, ok1 = solver.solve_few_iters(
            env, cost_fn, x0, U_warm, max_iter=2, use_fast_lin=False,
        )
        assert np.all(np.isfinite(X1)) and np.all(np.isfinite(U1))
        assert np.all(np.isfinite(c1))
        assert ok1
        assert c1[-1] < c1[0] * 10.0 + 1.0  # 不发散

    def test_far_jt_path_replan(self) -> None:
        """远段重规划：k_hit > far_threshold 时走 JT 快路径（iters=0）。"""
        env = _planning_env()
        limits = _make_limits(env)
        config = _make_config()
        solver = build_solver(env=env, backend="python")
        request = _first_plan_request(env)
        from src.ilqt.tube_types import ReplanState

        state = ReplanState(is_first_plan=False)
        # k_hit_current 大 → 重规划得 k_hit≈88 > far_threshold=50 → JT 快路径
        request_far = dataclasses.replace(
            request, is_first_plan=False, step=10, k_hit_current=190,
        )
        result = do_replan(request_far, env, state, config, limits, solver)
        assert not result.ball_unreachable
        assert result.k_hit_new > config.far_threshold
        assert result.iters_plan == 0  # JT 快路径：未跑 iLQR 迭代
        assert result.U_mpc_full.shape[1] == env.NU
        assert np.all(np.isfinite(result.U_mpc_full))


class TestMPCControllerR1D:
    """MPCController 七维消费链：start/step/滚动 buffer/重规划触发。"""

    def test_start_step_buffer_and_replan(self) -> None:
        """start → step×N：u_cmd (7,) 有限，重规划触发，阶段合法。"""
        env = _planning_env()
        limits = _make_limits(env)
        config = _make_config()
        mpc = MPCController(env, config, robot_limits=limits)

        task = _task()
        ball_pos = np.array(task.initial_ball_position, dtype=np.float64)
        ball_vel = np.array(task.initial_ball_velocity, dtype=np.float64)
        arm_state = env.get_arm_state()

        mpc.start(ball_pos, ball_vel, arm_state)
        assert mpc._solver.use_analytical is False  # noqa: SLF001 — env 注入生效

        phases: set[str] = set()
        replanned_count = 0
        n_steps = config.replan_interval * 2 + 2
        arm = arm_state.copy()
        for sc in range(n_steps):
            result = mpc.step(ball_pos, ball_vel, arm)
            assert isinstance(result, MPCStepResult)
            assert result.u_cmd.shape == (env.NU,)
            assert np.all(np.isfinite(result.u_cmd))
            phases.add(result.phase)
            if result.replanned:
                replanned_count += 1
            # 来球按物理推进（冻结球会在窗口收缩后被判不可达）
            ball_pos, ball_vel = _step_ball_analytic(ball_pos, ball_vel)
        assert replanned_count >= 1, "滚动窗口内应至少触发一次重规划"
        assert phases <= {"far", "mid", "near", "follow_through", "done"}
        mpc.stop()

    def test_follow_through_seven_dim(self) -> None:
        """随挥触发：k_hit 递减到 1 后进入 follow_through，u_cmd (7,) 有限。"""
        env = _planning_env()
        limits = _make_limits(env)
        # total_horizon 需覆盖来球到达（k_hit≈88 步），随挥由 planned 触发
        config = _make_config(
            follow_through_steps=4,
            total_horizon=100,
            follow_trigger="planned",
        )
        mpc = MPCController(env, config, robot_limits=limits)

        task = _task()
        ball_pos = np.array(task.initial_ball_position, dtype=np.float64)
        ball_vel = np.array(task.initial_ball_velocity, dtype=np.float64)
        arm_state = env.get_arm_state()

        mpc.start(ball_pos, ball_vel, arm_state)
        assert not mpc.ball_unreachable
        seen_follow = False
        result = None
        for _ in range(config.total_horizon + 6):
            result = mpc.step(ball_pos, ball_vel, arm_state)
            assert result.u_cmd.shape == (env.NU,)
            assert np.all(np.isfinite(result.u_cmd))
            if result.phase == "follow_through":
                seen_follow = True
                break
            # 来球按物理推进（冻结球会在窗口收缩后被判不可达）
            ball_pos, ball_vel = _step_ball_analytic(ball_pos, ball_vel)
        assert seen_follow, "应在 k_hit<=1 或 horizon 结束时触发随挥"
        mpc.stop()


class TestFollowThroughSevenDim:
    """PlannedFollowThrough 七维：位置模式 IK 校验 + 力矩模式控制界。"""

    def _position_env(self) -> ProfilePlanningEnv:
        """构建位置模式规划环境。"""
        env = _planning_env()
        params = env._parameters  # noqa: SLF001
        env.configure_actuator_mode(
            "position",
            np.array(params.kp, dtype=np.float64),
            np.array(params.kd, dtype=np.float64),
        )
        env.reset()
        return env

    def _ctx(
        self, env: ProfilePlanningEnv, k_hit: int = 1,
        robot_limits: object = None,
    ) -> FollowContext:
        arm = env.get_arm_state()
        return FollowContext(
            step_count=50, mpc_horizon=250, k_hit=k_hit,
            arm_state=arm,
            ball_pos=np.array([0.3, 0.3, 1.4]),
            d_follow=np.array([1.0, 0.0, 0.0]),
            v_hit_desired=np.array([1.2, 0.0, 0.0]),
            env=env,
            robot_limits=robot_limits,
        )

    def test_position_mode_control_within_bounds(self) -> None:
        """位置模式随挥：u=(7,) 且在 control_range（关节限位）内。"""
        env = self._position_env()
        limits = _make_limits(env)
        policy = PlannedFollowThrough(
            follow_through_steps=6, follow_trigger="planned",
            dt=env.dt, is_position_mode=True, NQ=env.NQ, NU=env.NU,
        )
        ctx = self._ctx(env, robot_limits=limits)
        assert policy.should_trigger(ctx) is True
        for step_in in range(3):
            u = policy.compute_control(ctx, step_in)
            assert u.shape == (env.NU,)
            cr = env.control_range
            assert np.all(u >= cr[:, 0] - 1e-9)
            assert np.all(u <= cr[:, 1] + 1e-9)

    def test_position_mode_ik_failure_falls_back_to_hold(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """IK 目标不可达：告警并回退 hold（不带坏指令继续）。"""
        env = self._position_env()
        limits = _make_limits(env)
        policy = PlannedFollowThrough(
            follow_through_steps=6, follow_trigger="planned",
            dt=env.dt, is_position_mode=True, NQ=env.NQ, NU=env.NU,
        )
        ctx = self._ctx(env, robot_limits=limits)
        policy.should_trigger(ctx)
        # 篡改触发位置到不可达远处 → p_des 随挥方向继续远离
        policy._p_ee_at_hit = np.array([10.0, 0.0, 1.5])  # noqa: SLF001
        with caplog.at_level("WARNING"):
            u = policy.compute_control(ctx, 1)
        assert np.allclose(u, ctx.arm_state[: env.NQ])  # hold 回退
        assert any("随挥 IK" in r.message for r in caplog.records)

    def test_torque_mode_control_finite(self) -> None:
        """力矩模式随挥：u=(7,) J^T PD，有限且在 ±torque_max 内。"""
        env = _planning_env()
        policy = PlannedFollowThrough(
            follow_through_steps=6, follow_trigger="planned",
            dt=env.dt, is_position_mode=False, NQ=env.NQ, NU=env.NU,
        )
        ctx = self._ctx(env)
        policy.should_trigger(ctx)
        u = policy.compute_control(ctx, 0)
        assert u.shape == (env.NU,)
        assert np.all(np.isfinite(u))
        cr = env.control_range
        assert np.all(u >= cr[:, 0] - 1e-9)
        assert np.all(u <= cr[:, 1] + 1e-9)


# ==============================================================================
# 6. 同序重排一致性
# ==============================================================================


class TestPermutationConsistency:
    """同序重排关节/执行器/参数后，输出物理意义一致。"""

    @staticmethod
    def _permuted_profile() -> RobotProfile:
        """按 _PERM 重排 active_joint_names/actuator_names 的 profile。"""
        prof = _profile()
        return dataclasses.replace(
            prof,
            active_joint_names=tuple(prof.active_joint_names[i] for i in _PERM),
            actuator_names=tuple(prof.actuator_names[i] for i in _PERM),
        )

    @staticmethod
    def _permuted_parameters() -> RobotRuntimeParameters:
        """按 _PERM 重排关节序参数（数值数组同序重排）。"""
        params = _parameters()
        perm = np.array(_PERM)

        def p(arr: object) -> object:
            if arr is None:
                return None
            arr_np = np.asarray(arr, dtype=np.float64)
            if arr_np.size == 0:
                return arr  # 空数组（如 held_q=[]）原样返回
            return arr_np[perm]  # type: ignore[index]

        return dataclasses.replace(
            params,
            joint_names=tuple(params.joint_names[i] for i in _PERM),
            init_q=p(params.init_q),
            held_q=p(params.held_q) if params.held_q is not None else params.held_q,
            kp=p(params.kp) if params.kp is not None else params.kp,
            kd=p(params.kd) if params.kd is not None else params.kd,
            qdot_max=p(params.qdot_max),
            qddot_max=p(params.qddot_max),
            torque_max=p(params.torque_max),
        )

    def test_binding_addresses_permuted(self) -> None:
        """重排后绑定地址按名称解析（B 第 k 行 = A 第 PERM[k] 关节）。"""
        env_a = _planning_env()
        bundle_b = load_robot_model(
            self._permuted_profile(), self._permuted_parameters(),
        )
        env_b = ProfilePlanningEnv(bundle_b)

        name_by_qpos_a = {}
        for k, jid in enumerate(env_a.binding.joint_ids):
            name_by_qpos_a[k] = env_a.model.joint(jid).name
        import mujoco  # 局部导入：仅重排测试需要按名反查

        for k in range(env_b.NQ):
            jid_b = env_b.binding.joint_ids[k]
            name_b = mujoco.mj_id2name(
                env_b.model, mujoco.mjtObj.mjOBJ_JOINT, jid_b
            )
            assert name_b == name_by_qpos_a[_PERM[k]]

    def test_jt_warm_start_permutation_invariant(self) -> None:
        """JT warm-start：U_B[:, PERM] ≈ U_A（物理意义一致）。"""
        env_a = _planning_env()
        env_b = ProfilePlanningEnv(
            load_robot_model(self._permuted_profile(), self._permuted_parameters())
        )
        p_hit = np.array(_task().workspace_center, dtype=np.float64)

        x0_a = env_a.get_arm_state()
        x0_b = x0_a.copy()
        perm = np.array(_PERM)
        x0_b[: env_a.NQ] = x0_a[: env_a.NQ][perm]
        x0_b[env_a.NQ :] = x0_a[env_a.NQ :][perm]

        U_a = compute_jacobian_init_control(env_a, x0_a, p_hit, horizon=6)
        U_b = compute_jacobian_init_control(env_b, x0_b, p_hit, horizon=6)

        # 方向：B 的第 k 列 = A 的第 PERM[k] 列 ⇒ U_A = U_B[:, argsort(PERM)]
        perm_inv = np.argsort(perm)
        assert U_a.shape == U_b.shape == (6, 7)
        assert np.allclose(U_a, U_b[:, perm_inv], atol=1e-9)

    def test_ilqr_one_iter_permutation_invariant(self) -> None:
        """1 次迭代 iLQR（FD 线性化）：控制序列按重排列反演一致。"""
        env_a = _planning_env()
        env_b = ProfilePlanningEnv(
            load_robot_model(self._permuted_profile(), self._permuted_parameters())
        )
        perm = np.array(_PERM)
        task = _task()
        p_hit = np.array(task.workspace_center, dtype=np.float64)

        def build_inputs(env: ProfilePlanningEnv) -> tuple:
            # get_arm_state() 已按各环境的关节序返回（env_b 重排后 init_q
            # 同步重排），无需再次置换
            x0 = env.get_arm_state()
            U_warm = compute_jacobian_init_control(env, x0, p_hit, horizon=4)
            limits = RobotLimits(
                q_lower=np.array(env._parameters.q_lower),  # noqa: SLF001
                q_upper=np.array(env._parameters.q_upper),  # noqa: SLF001
                qdot_max=np.full(7, 30.0),
                qddot_max=np.full(7, np.inf),
                u_min=-np.array(env._parameters.torque_max),  # noqa: SLF001
                u_max=np.array(env._parameters.torque_max),  # noqa: SLF001
            )
            cfg = MPCConfig(
                dt=0.005, R=1.0e-4, Q_qdot_base=0.001,
                Q_qddot_base=0.0005, Q_du_base=0.001,
            )
            cost = build_production_cost(
                env, cfg, limits, p_hit=p_hit,
                v_hit=np.array([1.0, 0.0, 0.0]), n_des=None,
                Q_p_mat=np.eye(3) * 2000.0, Q_v_mat=np.eye(3) * 100.0,
            )
            solver = build_solver(env=env, backend="python")
            return x0, U_warm, cost, solver

        x0_a, u_a, cost_a, solver_a = build_inputs(env_a)
        x0_b, u_b, cost_b, solver_b = build_inputs(env_b)

        _, U_out_a, _, ok_a = solver_a.solve_few_iters(
            env_a, cost_a, x0_a, u_a, max_iter=1, limits=None,
            use_fast_lin=True,
        )
        _, U_out_b, _, ok_b = solver_b.solve_few_iters(
            env_b, cost_b, x0_b, u_b, max_iter=1, limits=None,
            use_fast_lin=True,
        )
        assert ok_a and ok_b
        assert np.allclose(U_out_a, U_out_b[:, np.argsort(perm)], atol=1e-6)


# ==============================================================================
# 7. async 决定：本阶段明确拒绝 R1D 异步
# ==============================================================================


class TestAsyncRejection:
    """R1D + async → 入口明确拒绝（不静默改同步）。"""

    def test_controller_rejects_async_on_bound_env(self) -> None:
        """MPCController（R1D + async_mode=True）→ ValueError。"""
        env = _planning_env()
        config = _make_config(async_mode=True)
        with pytest.raises(ValueError, match="异步"):
            MPCController(env, config, robot_limits=_make_limits(env))

    def test_async_replanner_rejects_bound_env(self) -> None:
        """AsyncReplanner 直接构造（绑定环境）→ ValueError。"""
        from src.ilqt.robot_limits import RobotLimits as _RL
        from src.ilqt.solver import build_solver as _bs

        env = _planning_env()
        with pytest.raises(ValueError, match="绑定环境"):
            AsyncReplanner(
                env, do_replan, _make_config(), _RL(
                    q_lower=np.zeros(7), q_upper=np.zeros(7),
                    qdot_max=np.zeros(7), qddot_max=np.zeros(7),
                    u_min=np.zeros(7), u_max=np.zeros(7),
                ), _bs(env=env, backend="python"),
            )


# ==============================================================================
# 8. use_fast_lin=True + FD 后端回退分支补测（3.5 评审指针）
# ==============================================================================


class TestFastLinFDFallback:
    """FD 后端下 use_fast_lin 快捷路径回退完整 FD：结果一致。"""

    def test_fast_lin_equals_full_fd(self) -> None:
        """use_fast_lin=True 与 False 在 FD 后端下输出一致。"""
        env = _planning_env()
        task = _task()
        p_hit = np.array(task.workspace_center, dtype=np.float64)
        x0 = env.get_arm_state()
        U_warm = compute_jacobian_init_control(env, x0, p_hit, horizon=4)

        limits = RobotLimits(
            q_lower=np.zeros(7) - 10.0, q_upper=np.zeros(7) + 10.0,
            qdot_max=np.full(7, 100.0), qddot_max=np.full(7, np.inf),
            u_min=-np.full(7, 1e6), u_max=np.full(7, 1e6),
        )
        cfg = MPCConfig(dt=0.005, R=1.0e-4)
        cost = build_production_cost(
            env, cfg, limits, p_hit=p_hit,
            v_hit=np.array([1.0, 0.0, 0.0]), n_des=None,
            Q_p_mat=np.eye(3) * 2000.0, Q_v_mat=np.eye(3) * 100.0,
        )
        solver = build_solver(env=env, backend="python")
        assert solver.use_analytical is False

        X_fast, U_fast, c_fast, ok_fast = solver.solve_few_iters(
            env, cost, x0, U_warm.copy(), max_iter=1, use_fast_lin=True,
        )
        X_full, U_full, c_full, ok_full = solver.solve_few_iters(
            env, cost, x0, U_warm.copy(), max_iter=1, use_fast_lin=False,
        )
        assert ok_fast and ok_full
        assert np.allclose(X_fast, X_full, atol=1e-12)
        assert np.allclose(U_fast, U_full, atol=1e-12)
        assert np.allclose(c_fast, c_full, atol=1e-9)


# ==============================================================================
# 9. cost_config 透传（replan_core → build_production_cost）
# ==============================================================================


class TestCostConfigPassthrough:
    """cost_config 键提取与透传（Q_p/Q_v 进调度基底，其余透传工厂）。"""

    def test_resolve_cost_bases_with_task_config(self) -> None:
        """R1D：Q_p/Q_v 从 cost_config 提取为基底，其余键透传。"""
        cost_cfg = {
            "Q_p": 2000.0, "Q_v": 100.0, "R": 1.0e-2,
            "tube": {"window_half_ms": 50.0},
        }
        config = _make_config(cost_config=cost_cfg)
        q_p_mat, q_v_mat, overrides = _resolve_cost_bases(config, 8.0, 120.0)

        # 终端基准 = cost_config Q_p/Q_v × 调度 scale
        assert np.allclose(q_p_mat, 8.0 * 2000.0 * np.eye(3))
        assert np.allclose(q_v_mat, 120.0 * 100.0 * np.eye(3))
        # Q_p/Q_v 已从透传字典提取（避免与 Q_p_mat 双源）
        assert overrides is not None
        assert "Q_p" not in overrides and "Q_v" not in overrides
        assert overrides.get("R") == 1.0e-2
        assert "tube" in overrides

    def test_resolve_cost_bases_yaml_task_config(self) -> None:
        """真实 task YAML cost_config（Q_p/Q_v/tube）：提取后仅剩 tube。"""
        config = _make_config()  # cost_config=dict(task.cost_config)
        _, _, overrides = _resolve_cost_bases(config, 8.0, 120.0)
        assert overrides is not None
        assert set(overrides) == {"tube"}

    def test_resolve_cost_bases_rm65_default(self) -> None:
        """RM65（cost_config=None）：走 Q_p_base 调度，透传 None。"""
        config = MPCConfig(
            Q_p_base=np.array([1e5, 1e5, 1e5]), Q_v_base=None,
        )
        q_p_mat, q_v_mat, overrides = _resolve_cost_bases(config, 5.0, 3.0)
        assert np.allclose(q_p_mat, 5.0 * np.diag(np.array([1e5, 1e5, 1e5])))
        assert np.allclose(q_v_mat, 3.0 * np.eye(3))  # Q_v_base=None → I₃
        assert overrides is None

    def test_replan_accepts_task_cost_config(self) -> None:
        """do_replan 全链消费 task cost_config（含 tube 子字典）不报错。"""
        env = _planning_env()
        limits = _make_limits(env)
        config = _make_config()
        solver = build_solver(env=env, backend="python")
        request = _first_plan_request(env)
        from src.ilqt.tube_types import ReplanState

        state = ReplanState(is_first_plan=True)
        result = do_replan(request, env, state, config, limits, solver)
        assert not result.ball_unreachable
        assert np.all(np.isfinite(result.U_mpc_full))

    def test_r_decay_base_uses_effective_r(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R 退火基底 = cost_config 覆盖后的有效 R（审查 P2 组合测试）。

        cost_config.R=7、config.R=1e-4 且 use_r_decay 时，退火调度须以
        7 为基底——修复前以 config.R 为基底，七维全一控制代价 0.00035
        （正确值 24.5），机器人专属 R 被退火调度静默覆盖。
        """
        import src.ilqt.replan_core as replan_core
        from src.ilqt.tube_types import ReplanState

        env = _planning_env()
        limits = _make_limits(env)
        task_cfg = dict(_task().cost_config)
        task_cfg["R"] = 7.0
        config = _make_config(
            cost_config=task_cfg, use_r_decay=True, R=1.0e-4,
        )
        solver = build_solver(env=env, backend="python")
        request = _first_plan_request(env)

        captured: dict[str, float] = {}
        real = replan_core.compute_r_schedule

        def _spy(steps_remaining: int, base_R: float, **kwargs: object):
            captured.setdefault("base_R", float(base_R))
            return real(steps_remaining, base_R, **kwargs)

        monkeypatch.setattr(replan_core, "compute_r_schedule", _spy)
        result = do_replan(request, env, ReplanState(is_first_plan=True),
                           config, limits, solver)
        assert not result.ball_unreachable
        assert captured.get("base_R") == 7.0, (
            "R 退火基底应为 cost_config.R（有效 R），"
            f"实际 {captured.get('base_R')}"
        )

    def test_r_decay_base_falls_back_to_config_r(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RM65（cost_config=None）退火基底仍为 config.R（回归锁定）。"""
        import src.ilqt.replan_core as replan_core
        from src.ilqt.tube_types import ReplanState

        env = _planning_env()
        limits = _make_limits(env)
        config = _make_config(cost_config=None, use_r_decay=True, R=1.0e-4)
        solver = build_solver(env=env, backend="python")
        request = _first_plan_request(env)

        captured: dict[str, float] = {}
        real = replan_core.compute_r_schedule

        def _spy(steps_remaining: int, base_R: float, **kwargs: object):
            captured.setdefault("base_R", float(base_R))
            return real(steps_remaining, base_R, **kwargs)

        monkeypatch.setattr(replan_core, "compute_r_schedule", _spy)
        result = do_replan(request, env, ReplanState(is_first_plan=True),
                           config, limits, solver)
        assert not result.ball_unreachable
        assert captured.get("base_R") == pytest.approx(1.0e-4)
