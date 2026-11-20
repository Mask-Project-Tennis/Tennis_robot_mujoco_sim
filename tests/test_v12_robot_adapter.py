"""T4：V12RobotBundle 双机器人适配测试（RM65 / R1D）。

覆盖 brief §5 的 7 类 adapter 断言：
1. RM65 bundle 与 T0 构造事实一致（限位/初始姿态/几何/同源执行规划对象）；
2. RM65 规划环境工厂返回独立实例且执行器配置克隆；
3. R1D 七关节 / NU=7 / 执行 nq=n+7、规划 nq=n / 几何 ID / 保护接触与
   X 墙语义依据；
4. R1D 规划环境工厂每次返回全新 model/data；
5. 七关节 + 自由球乱序按名映射（绝不截前六、不假设前缀布局）；
6. 空 held 集合不读写相邻槽（R1D）与 RM65 左臂槽位不受按名回放影响；
7. R1D 强制 cpp 后端必须明确 ValueError；两机器人位置模式走同一
   adapter 配置分支。
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.ilqt.components.predictive_safety import X_WALL_BODY_NAMES
from src.ilqt.profile_planning_env import ProfilePlanningEnv
from src.robot.constants import INIT_Q, INIT_Q_LEFT
from src.sim.rm65_env import RM65Env
from src.sim.v12_config import resolve_v12_config
from src.sim.v12_robot_adapter import (
    ReplayState,
    SimulationEnv,
    V12RobotBundle,
    build_v12_robot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOTS_DIR = REPO_ROOT / "configs" / "robots"
R1D_PROFILE = ROBOTS_DIR / "r1d_right.yaml"
R1D_RUNTIME = ROBOTS_DIR / "r1d_v12_runtime_right.yaml"

R1D_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
# T7 冻结准备姿态（2026-11-19，configs/robots/r1d_v12_runtime_right.yaml）；
# 该常量只锁定 bundle 按 runtime 原样读取 init_q 的行为，不参与 T7 调参。
R1D_INIT_Q = [0.091, 0.152, 0.021, 0.504, -0.218, 0.321, 0.019]
R1D_KP = [400.0, 400.0, 250.0, 120.0, 60.0, 40.0, 40.0]
R1D_KD = [12.0, 12.0, 8.0, 5.0, 2.5, 1.5, 1.5]
R1D_TORQUE = [60.0, 60.0, 33.0, 33.0, 33.0, 10.0, 10.0]
R1D_PROTECTED_PAIRS = (
    ("right_shoulder_pitch_Link_visual", "torso_box"),
    ("right_shoulder_roll_link_visual", "torso_box"),
    ("right_shoulder_yaw_link_visual", "torso_box"),
    ("right_elbow_link_visual", "torso_box"),
    ("right_wrist_roll_link_visual", "torso_box"),
    ("right_wrist_pitch_link_visual", "torso_box"),
    ("right_wrist_yaw_link_visual", "torso_box"),
    ("racket_face", "torso_box"),
)


def _resolve(args: dict[str, object], flags: set[str]) -> object:
    """用 T3 resolver 构造真实 :class:`ResolvedV12Config`。"""
    return resolve_v12_config(
        argparse.Namespace(**args), frozenset(flags)
    )


def _rm65_config(*, position_mode: bool = False) -> object:
    """T0 基线（serve-box 7 m/s）resolved 配置；可选位置模式。"""
    args: dict[str, object] = {
        "serve_box": True,
        "ball_speed": 7.0,
        "seed": 42,
        "no_plot": True,
    }
    flags = {"--serve-box", "--ball-speed", "--seed", "--no-plot"}
    if position_mode:
        args["position_mode"] = True
        flags.add("--position-mode")
    return _resolve(args, flags)


def _r1d_config(
    *,
    position_mode: bool = False,
    solver_backend: str | None = None,
) -> object:
    """R1D 右臂 resolved 配置；可选位置模式 / 求解后端覆盖。"""
    args: dict[str, object] = {"robot_config": R1D_PROFILE}
    flags = {"--robot-config"}
    if position_mode:
        args["position_mode"] = True
        args["runtime_config"] = R1D_RUNTIME
        flags |= {"--position-mode", "--runtime-config"}
    if solver_backend is not None:
        args["solver_backend"] = solver_backend
        flags.add("--solver-backend")
    return _resolve(args, flags)


@pytest.fixture()
def rm65_bundle() -> V12RobotBundle:
    """RM65 扭矩模式 bundle。"""
    return build_v12_robot(_rm65_config())  # type: ignore[arg-type]


@pytest.fixture()
def r1d_bundle() -> V12RobotBundle:
    """R1D 右臂扭矩模式 bundle。"""
    return build_v12_robot(_r1d_config())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. RM65 bundle 与 T0 构造事实一致
# ---------------------------------------------------------------------------


def test_rm65_bundle_matches_t0_construction_facts(
    rm65_bundle: V12RobotBundle,
) -> None:
    """RM65 bundle 的关节/环境/限位/几何/安全事实与 T0 逐项一致。"""
    bundle = rm65_bundle
    assert bundle.robot_id == "rm65"
    assert bundle.joint_names == (
        "r_joint1",
        "r_joint2",
        "r_joint3",
        "r_joint4",
        "r_joint5",
        "r_joint6",
    )
    assert isinstance(bundle.execution, SimulationEnv)

    # 环境维数：RM65 双臂 12 + 球 6 => nv=18；执行状态为右臂 12 维
    assert (bundle.execution.NQ, bundle.execution.NV) == (6, 18)
    assert (bundle.execution.NU, bundle.execution.NX) == (6, 12)
    # T0：MPCController 以 RM65Env 同时承担执行与规划
    assert bundle.planning is bundle.execution
    assert np.allclose(bundle.init_q, INIT_Q)
    assert np.allclose(bundle.execution.init_q_left, INIT_Q_LEFT)  # type: ignore[attr-defined]

    geometry = bundle.geometry
    model = bundle.execution.model
    assert geometry.racket_center_site == "racket_center"
    assert geometry.racket_face_geom == "r_racket_face"
    assert geometry.ball_geom == "ball"
    assert geometry.racket_body == "r_racket_body"
    assert geometry.flange_body == "r_flange"
    assert geometry.ball_joint == "ball_freejoint"
    assert geometry.racket_geom_names == ("r_racket_handle", "r_racket_face")
    assert geometry.racket_center_site_id == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "racket_center"
    )
    assert geometry.racket_face_geom_id == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "r_racket_face"
    )
    assert geometry.ball_geom_id == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "ball"
    )
    assert geometry.ball_joint_id == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint"
    )

    # T0：无显式保护接触；X 墙为世界 X 平面（X >= -0.1）
    assert bundle.protected_contacts == ()
    assert bundle.x_wall.mode == "world_x"
    assert bundle.x_wall.limit == pytest.approx(-0.1)
    assert bundle.x_wall.body_names == X_WALL_BODY_NAMES
    assert bundle.x_wall.enabled is True
    assert bundle.x_wall.coordinate_frame == "world"
    assert bundle.backend == "auto"

    # 力矩上限与 T0 一致：来自力矩模式 ctrlrange 绝对值
    ctrlrange = bundle.execution.model.actuator_ctrlrange[: bundle.execution.NU]
    assert np.allclose(bundle.execution.control_range, ctrlrange)
    assert np.allclose(
        bundle.execution.torque_max,
        np.maximum(np.abs(ctrlrange[:, 0]), np.abs(ctrlrange[:, 1])),
    )
    assert np.allclose(bundle.execution.torque_max, [60, 60, 30, 10, 10, 10])

    # 仿真限位：T0 RobotLimits.from_config(default.yaml, ctrlrange) 语义
    limits = bundle.robot_limits
    assert np.allclose(
        np.rad2deg(limits.q_lower),
        np.array([-178, -130, -135, -178, -128, -360], dtype=float)
        + np.array([2, 1, 3, 3, 3, 3], dtype=float),
    )
    assert np.allclose(
        np.rad2deg(limits.q_upper),
        np.array([178, 130, 135, 178, 128, 360], dtype=float)
        - np.array([2, 1, 3, 3, 3, 3], dtype=float),
    )
    assert np.allclose(
        np.rad2deg(limits.qdot_max),
        [180, 180, 225, 225, 225, 225],
    )
    assert np.allclose(
        np.rad2deg(limits.qddot_max),
        np.array([400, 400, 500, 500, 500, 500], dtype=float) * 0.85,
    )
    assert np.allclose(limits.u_min, [-60, -60, -30, -10, -10, -10])
    assert np.allclose(limits.u_max, [60, 60, 30, 10, 10, 10])


def test_rm65_planning_factory_returns_independent_env(
    rm65_bundle: V12RobotBundle,
) -> None:
    """RM65 工厂复刻 T0：独立 RM65Env + init_q_left + 执行器配置克隆。"""
    plan_env = rm65_bundle.planning_env_factory()
    assert isinstance(plan_env, RM65Env)
    assert plan_env is not rm65_bundle.execution
    assert plan_env.data is not rm65_bundle.execution.data
    assert np.allclose(plan_env.init_q_left, INIT_Q_LEFT)
    assert np.allclose(
        plan_env.model.actuator_ctrlrange,
        rm65_bundle.execution.model.actuator_ctrlrange,
    )
    # 每次调用都必须全新实例（后台线程独占 MjData）
    second = rm65_bundle.planning_env_factory()
    assert second is not plan_env
    assert second.data is not plan_env.data


def test_unknown_robot_id_rejected() -> None:
    """未知 robot_id 必须明确 ValueError。"""
    config = _rm65_config()
    ghost = dataclasses.replace(config, robot_id="ghost_robot")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="未知机器人"):
        build_v12_robot(ghost)


# ---------------------------------------------------------------------------
# 2. R1D bundle 形态 / 几何 / 保护接触 / X 墙依据
# ---------------------------------------------------------------------------


def test_r1d_bundle_seven_joint_shape_and_geometry(
    r1d_bundle: V12RobotBundle,
) -> None:
    """R1D bundle：七关节按名、执行 nq=n+7、规划 nq=n、几何与安全依据。"""
    bundle = r1d_bundle
    assert bundle.robot_id == "unitree_r1d"
    assert bundle.joint_names == R1D_JOINT_NAMES
    assert isinstance(bundle.execution, SimulationEnv)
    # 规划环境只需满足 RobotEnv 求解接口（brief §4.2），不要求执行侧协议
    assert not isinstance(bundle.planning, SimulationEnv)

    # 执行环境带自由球：nq = 7 + 7 = 14, nv = 7 + 6 = 13；
    # 规划环境为裸模型：nq = nv = 7；执行状态维数均为 2n = 14。
    assert (bundle.execution.NQ, bundle.execution.NV) == (7, 13)
    assert (bundle.execution.NU, bundle.execution.NX) == (7, 14)
    assert bundle.execution.model.nq == 14
    assert bundle.execution.model.nv == 13
    assert (bundle.planning.NQ, bundle.planning.NV) == (7, 7)
    assert bundle.planning.model.nq == 7
    assert bundle.planning is not bundle.execution
    assert bundle.planning.data is not bundle.execution.data

    assert np.allclose(bundle.init_q, R1D_INIT_Q)

    geometry = bundle.geometry
    model = bundle.execution.model
    assert geometry.racket_center_site == "racket_center"
    assert geometry.racket_face_geom == "racket_face"
    assert geometry.ball_geom == "ball"
    assert geometry.racket_body == "r1d_racket"
    assert geometry.flange_body == "right_wrist_yaw_link"
    assert geometry.ball_joint == "ball_freejoint"
    assert "racket_face" in geometry.racket_geom_names
    assert "racket_handle" in geometry.racket_geom_names
    assert geometry.racket_center_site_id == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "racket_center"
    )
    assert geometry.racket_body_id == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "r1d_racket"
    )
    assert geometry.ball_geom_id == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "ball"
    )

    # 保护接触来自 T2 场景保护对（躯干盒 vs 右臂链路/拍面）
    assert bundle.protected_contacts == R1D_PROTECTED_PAIRS
    # X 墙：世界 X 墙对 12.5m 外底座无意义，显式映射为躯干局部对称面
    assert bundle.x_wall.mode == "body_plane"
    assert bundle.x_wall.limit == pytest.approx(0.0)
    assert bundle.x_wall.body_names == ("r1d_torso",)
    assert bundle.x_wall.protected_geom_names == ("torso_box",)
    assert bundle.x_wall.enabled is False
    assert bundle.x_wall.coordinate_frame == "torso_local"

    # 后端事实：R1D 只允许 auto/python（FD），并记录为 python
    assert bundle.backend == "python"
    # 球拍装配事实来自 T2
    assert bundle.racket is not None
    assert bundle.racket.racket_id == "blade100_v9_reference"
    assert bundle.robot_model_bundle is not None
    assert bundle.robot_model_bundle.racket is bundle.racket

    # 限位/力矩以 runtime 为权威
    assert np.allclose(bundle.execution.torque_max, R1D_TORQUE)
    assert np.allclose(bundle.robot_limits.u_min, -np.array(R1D_TORQUE))
    assert np.allclose(bundle.robot_limits.u_max, np.array(R1D_TORQUE))
    assert np.all(np.isfinite(bundle.robot_limits.q_lower))
    assert np.all(np.isfinite(bundle.robot_limits.q_upper))

    # 自由球地址来自场景（球在主动关节之后）：reset 后球回到任务初态
    bundle.reset()
    replay = bundle.get_replay_state()
    assert np.allclose(replay.ball_position, [0.3, -0.6, 2.2])


def test_r1d_planning_factory_returns_independent_env(
    r1d_bundle: V12RobotBundle,
) -> None:
    """R1D 工厂每次返回全新 ProfilePlanningEnv（独立 model 与 data）。"""
    plan_env = r1d_bundle.planning_env_factory()
    assert isinstance(plan_env, ProfilePlanningEnv)
    assert plan_env is not r1d_bundle.execution
    assert plan_env.data is not r1d_bundle.execution.data
    assert plan_env.model is not r1d_bundle.execution.model
    assert (plan_env.NQ, plan_env.NV, plan_env.NU) == (7, 7, 7)
    assert plan_env.model.nq == 7

    second = r1d_bundle.planning_env_factory()
    assert second is not plan_env
    assert second.data is not plan_env.data
    assert second.model is not plan_env.model


# ---------------------------------------------------------------------------
# 3. 七关节 + 自由球乱序按名映射
# ---------------------------------------------------------------------------


def test_named_replay_mapping_handles_out_of_order_seven_joints(
    r1d_bundle: V12RobotBundle,
) -> None:
    """回放状态按关节名读写：字典乱序也必须落到正确 qpos/qvel 地址。"""
    bundle = r1d_bundle
    bundle.reset()
    names = bundle.joint_names
    # 逆序构造映射，验证实现不依赖“前六个/正序”假设
    positions = {
        name: 0.01 * (index + 1) for index, name in enumerate(reversed(names))
    }
    velocities = {
        name: -0.02 * (index + 1) for index, name in enumerate(names)
    }
    state = ReplayState(
        joint_positions=positions,
        joint_velocities=velocities,
        ball_position=np.array([-11.0, -0.5, 1.4]),
        ball_velocity=np.array([2.0, 0.1, -0.2]),
    )
    bundle.set_replay_state(state)

    model = bundle.execution.model
    data = bundle.execution.data
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])
        assert float(data.qpos[qpos_adr]) == pytest.approx(positions[name])
        assert float(data.qvel[qvel_adr]) == pytest.approx(velocities[name])

    replay = bundle.get_replay_state()
    assert np.allclose(replay.ball_position, [-11.0, -0.5, 1.4])
    assert np.allclose(replay.ball_velocity, [2.0, 0.1, -0.2])
    # 每个关节名都能反查 qpos 地址；第 7 关节不是前缀布局的 6
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert bundle.joint_qpos_index(name) == int(model.jnt_qposadr[joint_id])
    assert bundle.joint_qpos_index(names[6]) != 6
    assert bundle.joint_qpos_index(names[0]) != 6


def test_replay_state_returns_independent_copies(
    r1d_bundle: V12RobotBundle,
) -> None:
    """读取回放状态返回副本，修改不污染环境；写入要求完整关节名集合。"""
    bundle = r1d_bundle
    bundle.reset()
    replay = bundle.get_replay_state()
    replay.ball_position[0] = 999.0
    assert bundle.get_replay_state().ball_position[0] != pytest.approx(999.0)

    with pytest.raises(ValueError, match="缺少关节"):
        bundle.set_replay_state(
            ReplayState(
                joint_positions={bundle.joint_names[0]: 0.0},
                joint_velocities={},
                ball_position=np.zeros(3),
                ball_velocity=np.zeros(3),
            )
        )


def test_empty_held_set_does_not_write_neighbour_slots(
    r1d_bundle: V12RobotBundle,
    rm65_bundle: V12RobotBundle,
) -> None:
    """R1D 空 held 集合：写全部 7 个关节不会碰到球槽位。"""
    bundle = r1d_bundle
    assert bundle.execution.binding.held_qpos_indices == ()  # type: ignore[attr-defined]
    bundle.reset()
    model = bundle.execution.model
    arm_qpos_adr = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in bundle.joint_names
    ]
    ball_before = bundle.get_replay_state().ball_position.copy()
    zeros = np.zeros(2 * bundle.execution.NQ)
    bundle.execution.set_arm_state(zeros)
    for adr in arm_qpos_adr:
        assert float(bundle.execution.data.qpos[adr]) == pytest.approx(0.0)
    # 相邻（自由球）槽位未被读写
    assert np.allclose(bundle.get_replay_state().ball_position, ball_before)

    # RM65：按名回放只覆盖 6 个主动关节；左臂槽位由环境自身维护
    assert len(rm65_bundle.get_replay_state().joint_positions) == 6
    rm65_bundle.reset()
    rm65_bundle.set_replay_state(
        ReplayState(
            joint_positions={name: 0.1 for name in rm65_bundle.joint_names},
            joint_velocities={name: 0.0 for name in rm65_bundle.joint_names},
            ball_position=np.array([5.0, 0.0, 2.0]),
            ball_velocity=np.zeros(3),
        )
    )
    data = rm65_bundle.execution.data
    assert np.allclose(data.qpos[6:12], INIT_Q_LEFT)


# ---------------------------------------------------------------------------
# 4. 后端拒绝与两模式同源
# ---------------------------------------------------------------------------


def test_r1d_forced_cpp_backend_rejected() -> None:
    """R1D + 强制 cpp 必须明确 ValueError（不静默降级为 FD）。"""
    with pytest.raises(ValueError, match=r"C\+\+ 后端"):
        build_v12_robot(_r1d_config(solver_backend="cpp"))  # type: ignore[arg-type]


def test_two_robots_share_position_mode_configuration_branch() -> None:
    """两机器人位置模式都经 adapter 同一分支配置执行器（brief §4.5）。"""
    rm65 = build_v12_robot(_rm65_config(position_mode=True))  # type: ignore[arg-type]
    assert rm65.execution.actuator_mode == 1
    assert np.allclose(rm65.execution.kp, [25.0] * 6)
    assert np.allclose(rm65.execution.kd, [2.0] * 6)
    assert rm65.execution.use_feedforward is True
    # RM65 规划与执行同一对象，配置同源
    assert rm65.planning is rm65.execution
    rm65_plan = rm65.planning_env_factory()
    assert rm65_plan.actuator_mode == 1
    assert np.allclose(rm65_plan.kp, [25.0] * 6)

    r1d = build_v12_robot(_r1d_config(position_mode=True))  # type: ignore[arg-type]
    assert r1d.execution.actuator_mode == 1
    assert np.allclose(r1d.execution.kp, R1D_KP)
    assert np.allclose(r1d.execution.kd, R1D_KD)
    assert r1d.execution.use_feedforward is True
    assert r1d.planning.actuator_mode == 1  # type: ignore[attr-defined]
    assert np.allclose(r1d.planning.kp, R1D_KP)  # type: ignore[attr-defined]
    r1d_plan = r1d.planning_env_factory()
    assert r1d_plan.actuator_mode == 1
    assert np.allclose(r1d_plan.kp, R1D_KP)


# ---------------------------------------------------------------------------
# 5. step_from_state 不污染 live 状态
# ---------------------------------------------------------------------------


def test_step_from_state_does_not_pollute_live_state(
    rm65_bundle: V12RobotBundle,
    r1d_bundle: V12RobotBundle,
) -> None:
    """两机器人 trial step 都不得改写执行环境 live MjData。"""
    for bundle in (rm65_bundle, r1d_bundle):
        execution = bundle.execution
        bundle.reset()
        data = execution.data
        qpos_before = data.qpos.copy()
        qvel_before = data.qvel.copy()
        ctrl_before = data.ctrl.copy()
        time_before = float(data.time)
        x0 = execution.get_arm_state().copy()
        execution.step_from_state(  # type: ignore[attr-defined]
            x0, np.full(execution.NU, 3.0)
        )
        assert np.array_equal(data.qpos, qpos_before)
        assert np.array_equal(data.qvel, qvel_before)
        assert np.array_equal(data.ctrl, ctrl_before)
        assert float(data.time) == pytest.approx(time_before)
        assert np.allclose(execution.get_arm_state(), x0)
