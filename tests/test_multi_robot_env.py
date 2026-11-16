"""多机器人环境工厂与带球执行环境测试（阶段 3 · Task 3.4）。

核心断言组（brief + 施工计划 Task 3.4 / §共享接口契约）：
1. ``test_ball_arm_address_isolation`` 等：球/机械臂互不串地址、reset
   重现、球 quaternion 合法、step_full 仅推进一个 dt；调用球预测不得
   改物理球；试探步（step_from_state）不动 live 与物理球。
2. ``test_factory_*``：工厂分别构造执行/规划模型 ——
   ``execution.model is not planning.model``、data 独立，修改执行器或
   碰撞 mask 不串实例；初始化 TCP/惯量/限位/高度一致。
3. ``test_planning_scene_is_ball_free_and_collision_free``：规划路径
   禁用球接触（裸模型无球）、不承担真实碰撞判定（掩码全 0）；执行
   路径保留球物理（球—地面接触可观测）。
4. ``test_set_arm_collision_toggles_only_ball_racket_pair``：
   set_arm_collision 经 pair_margin 通道仅切球—拍接触，不关闭其他
   保护碰撞（球—机身接触在关闭期间仍可观测）。
5. ``test_pipeline_consumed_methods_contract``：共享管线实际消费的
   方法列入协议测试，少方法即失败；禁止 __getattr__ 兜底。
6. ``test_position_mode_step_from_state_independent_mjdata_reference``：
   位置模式 step_from_state 与独立 MjData 单步对照（3.3 评审指针：
   位置模式此前无独立 MjData 参照）。
7. ``test_factory_both_sides_*``：双侧工厂形状/独立性一致（Global
   Constraints：双侧完成环境测试）。

注意：本文件不共享 lru_cache 场景模型给会改 model 状态的测试
（pair_margin / 执行器模式），避免 test_tennis_scene.py 同款停放项。
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.ilqt.ball_predictor import BallPredictor
from src.ilqt.profile_planning_env import ProfilePlanningEnv
from src.ilqt.robot_env_protocol import RobotEnv
from src.robot.robot_profile import RobotProfile, load_robot_profile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    load_runtime_parameters,
)
from src.sim.bound_robot_env import BoundRobotEnv
from src.sim.env_factory import RobotEnvironments, build_r1d_environments
from src.sim.profile_sim_env import ProfileSimEnv
from src.sim.sim_config import SimTaskConfig, load_sim_task_config
from src.sim.tennis_scene import TennisSceneBundle, build_tennis_scene

# 仓库根目录（tests/ 的父目录）
REPO_ROOT = Path(__file__).resolve().parent.parent
SIDES = ("left", "right")

# 双侧描述/仿真 runtime/任务配置（Task 3.1 交付的真实文件）
_PROFILE_YAML = {
    side: REPO_ROOT / "configs" / "robots" / f"r1d_{side}.yaml" for side in SIDES
}
_SIM_RUNTIME_YAML = {
    side: REPO_ROOT / "configs" / "robots" / f"r1d_sim_runtime_{side}.yaml"
    for side in SIDES
}
_TASK_YAML = {
    side: REPO_ROOT / f"configs/r1d_sim_{side}.yaml" for side in SIDES
}

# 球物理常量（rm65_model.xml 逐字来源，与 tennis_scene 一致）
_BALL_RADIUS = 0.033

# 模板固定几何名（阶段 2 契约）
_GROUND_GEOM = "ground"
_TORSO_GEOM = "torso_box"
_RACKET_GEOM = "racket_face"

# 固定随机种子（保证失败可复现）
_RNG_SEED = 20261117


# ---------- 不可变配置加载（进程内缓存，纯只读对象） ----------


@lru_cache(maxsize=2)
def _profile(side: str) -> RobotProfile:
    """加载并缓存该侧 RobotProfile。"""
    return load_robot_profile(_PROFILE_YAML[side])


@lru_cache(maxsize=2)
def _parameters(side: str) -> RobotRuntimeParameters:
    """加载该侧仿真 runtime 参数（完整显式，provisional）。"""
    return load_runtime_parameters(_SIM_RUNTIME_YAML[side], _profile(side))


@lru_cache(maxsize=2)
def _task(side: str) -> SimTaskConfig:
    """加载并缓存该侧 SimTaskConfig。"""
    return load_sim_task_config(_TASK_YAML[side], _profile(side))


# ---------- fixture：只读行为测试共享一份带球环境 ----------


@pytest.fixture(scope="module")
def scene_env_left() -> tuple[ProfileSimEnv, TennisSceneBundle]:
    """module 级共享 ProfileSimEnv（左）+ 其场景（仅 data 层操作测试使用）。

    本 fixture 下的测试只做 reset/step/set_ball_state/predict 等 data 层
    操作，不修改 model 的执行器配置与 pair_margin，故可安全共享。
    """
    scene = build_tennis_scene(_profile("left"), _parameters("left"), _task("left"))
    return ProfileSimEnv(scene), scene


@pytest.fixture(scope="module")
def envs_left() -> RobotEnvironments:
    """module 级共享工厂产物（左）—— 工厂只读断言使用。

    使用后恢复执行器/碰撞状态的测试须自行还原（本组测试仅在
    set_arm_collision 往返测试中改动并以往返恢复）。
    """
    return build_r1d_environments(
        _profile("left"), _parameters("left"), _task("left")
    )


def _contact_pairs(data: mujoco.MjData) -> set[tuple[int, int]]:
    """收集当前接触的 geom id 对（无序对，小 id 在前）。"""
    return {
        (min(int(c.geom1), int(c.geom2)), max(int(c.geom1), int(c.geom2)))
        for i in range(data.ncon)
        for c in (data.contact[i],)
    }


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    """按名取 geom id（测试前提：名称存在）。"""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert gid >= 0, f"模型中不存在 geom {name}"
    return int(gid)


# ---------- 1. 带球执行环境：地址隔离 / reset / quaternion / step_full ----------


def test_ball_arm_address_isolation(
    scene_env_left: tuple[ProfileSimEnv, TennisSceneBundle],
) -> None:
    """球/机械臂互不串地址：set_ball_state 不动臂，set_arm_state 不动球。"""
    env, scene = scene_env_left
    n = env.NQ
    env.reset()

    # set_ball_state 不改臂状态（逐位相等）
    arm_before = env.get_arm_state().copy()
    env.set_ball_state(np.array([1.5, -0.4, 2.0]), np.array([3.0, 0.5, -1.0]))
    np.testing.assert_array_equal(env.get_arm_state(), arm_before)

    # set_arm_state 不改球状态（位置/速度逐位相等）
    ball_pos, ball_vel = env.get_ball_state()
    x = arm_before.copy()
    x[:n] += 0.05
    x[n:] += 0.02
    env.set_arm_state(x)
    np.testing.assert_array_equal(env.get_ball_state()[0], ball_pos)
    np.testing.assert_array_equal(env.get_ball_state()[1], ball_vel)
    # 球 quaternion 也未被触碰（仍为单位四元数）
    quat = env.data.qpos[scene.ball_qpos_start + 3 : scene.ball_qpos_start + 7]
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0, rtol=0, atol=1e-12)


def test_reset_reproducible(
    scene_env_left: tuple[ProfileSimEnv, TennisSceneBundle],
) -> None:
    """reset 重现：任意扰动后 reset 与初始 reset 逐位一致，重复 reset 稳定。"""
    env, _ = scene_env_left
    n = env.NQ
    x0 = env.reset()
    snap0 = (env.data.qpos.copy(), env.data.qvel.copy())

    # 扰动一：步进 + 球位姿改写 + 臂状态改写
    u = np.full(n, 0.2)
    for _ in range(10):
        env.step_full(u)
    env.set_ball_state(np.array([9.0, 9.0, 9.0]), np.array([1.0, 1.0, 1.0]))
    x_perturbed = x0.copy()
    x_perturbed[:n] += 0.1
    env.set_arm_state(x_perturbed)

    x1 = env.reset()
    snap1 = (env.data.qpos.copy(), env.data.qvel.copy())
    np.testing.assert_array_equal(x1, x0)
    np.testing.assert_array_equal(snap1[0], snap0[0])
    np.testing.assert_array_equal(snap1[1], snap0[1])

    # 扰动二后再次 reset：两轮 reset 结果逐位一致
    for _ in range(5):
        env.step_full(np.full(n, -0.1))
    env.reset()
    np.testing.assert_array_equal(env.data.qpos, snap0[0])
    np.testing.assert_array_equal(env.data.qvel, snap0[1])


def test_ball_quaternion_unit_after_set_and_steps(
    scene_env_left: tuple[ProfileSimEnv, TennisSceneBundle],
) -> None:
    """球 quaternion 合法：构造/set_ball_state/多步 step_full 后均为单位四元数。"""
    env, scene = scene_env_left
    bq = scene.ball_qpos_start
    env.reset()

    def _quat_norm() -> float:
        return float(
            np.linalg.norm(env.data.qpos[bq + 3 : bq + 7])
        )

    assert abs(_quat_norm() - 1.0) < 1e-12  # 构造/reset 后

    env.set_ball_state(np.array([1.0, 0.2, 1.8]), np.array([-2.0, 0.0, 0.5]))
    assert abs(_quat_norm() - 1.0) < 1e-12  # set_ball_state 写入 [1,0,0,0]

    u = np.full(env.NQ, 0.1)
    for _ in range(20):
        env.step_full(u)
    assert abs(_quat_norm() - 1.0) < 1e-9  # 物理积分后仍归一


def test_step_full_advances_exactly_one_dt(
    scene_env_left: tuple[ProfileSimEnv, TennisSceneBundle],
) -> None:
    """step_full 仅推进一个 dt；返回三元组形状与读数一致。"""
    env, _ = scene_env_left
    n = env.NQ
    env.reset()
    t0 = float(env.data.time)
    dt = env.dt

    arm, ball_pos, ball_vel = env.step_full(np.zeros(n))

    assert arm.shape == (2 * n,)
    assert ball_pos.shape == (3,)
    assert ball_vel.shape == (3,)
    assert env.data.time == t0 + dt  # 恰好一个时间步
    np.testing.assert_array_equal(arm, env.get_arm_state())
    np.testing.assert_array_equal(ball_pos, env.get_ball_pos())
    np.testing.assert_array_equal(ball_vel, env.get_ball_vel())


def test_predict_ball_keeps_physical_ball_frozen(
    scene_env_left: tuple[ProfileSimEnv, TennisSceneBundle],
) -> None:
    """调用球预测不得改物理球：预测前后 live qpos/qvel 逐位不变。

    同时验证 ProfileSimEnv 的预测委托 BallPredictor（解析抛物线，
    与独立 BallPredictor 实例结果逐位一致）。
    """
    env, _ = scene_env_left
    env.reset()
    env.set_ball_state(np.array([2.0, 0.4, 2.5]), np.array([-4.0, -0.3, -1.0]))
    snap = (env.data.qpos.copy(), env.data.qvel.copy())

    p0 = np.array([0.0, 0.0, 5.0])
    v0 = np.array([0.0, 0.0, -1.0])
    pos_pred, vel_pred = env.predict_ball_trajectory(p0, v0, 50)

    assert pos_pred.shape == (50, 3)
    assert vel_pred.shape == (50, 3)
    np.testing.assert_array_equal(env.data.qpos, snap[0])
    np.testing.assert_array_equal(env.data.qvel, snap[1])

    # 与独立 BallPredictor 逐位一致（复用解析预测，非物理仿真）
    ref = BallPredictor(dt=env.dt)
    pos_ref, vel_ref = ref.predict_from(p0, v0, 50)
    np.testing.assert_array_equal(pos_pred, pos_ref)
    np.testing.assert_array_equal(vel_pred, vel_ref)


def test_step_from_state_keeps_live_and_ball_frozen(
    scene_env_left: tuple[ProfileSimEnv, TennisSceneBundle],
) -> None:
    """试探步用 scratch data：step_from_state 后 live/物理球逐位不变。

    execution 不得作异步线程工作环境 —— 本条验证其预测通道不污染
    live 状态（异步规划须使用工厂的 planning 副本）。
    """
    env, _ = scene_env_left
    n = env.NQ
    env.reset()
    rng = np.random.default_rng(_RNG_SEED)
    snap = (
        env.data.qpos.copy(),
        env.data.qvel.copy(),
        env.data.ctrl.copy(),
        env.data.time,
        env.data.qfrc_applied.copy(),
        env.data.qacc_warmstart.copy(),
    )

    for _ in range(5):
        x = env.get_arm_state().copy()
        x[:n] += 0.02 * rng.standard_normal(n)
        x[n:] += 0.01 * rng.standard_normal(n)
        env.step_from_state(x, np.full(n, 0.5))

    np.testing.assert_array_equal(env.data.qpos, snap[0])
    np.testing.assert_array_equal(env.data.qvel, snap[1])
    np.testing.assert_array_equal(env.data.ctrl, snap[2])
    assert env.data.time == snap[3]
    np.testing.assert_array_equal(env.data.qfrc_applied, snap[4])
    np.testing.assert_array_equal(env.data.qacc_warmstart, snap[5])


def test_execution_preserves_ball_ground_contact(
    scene_env_left: tuple[ProfileSimEnv, TennisSceneBundle],
) -> None:
    """执行路径保留球物理：球—地面接触可观测（MuJoCo 物理碰撞）。"""
    env, scene = scene_env_left
    env.reset()
    ground = _geom_id(env.model, _GROUND_GEOM)
    ball = scene.ball_geom_id

    # 球心放到地面之上且穿透 1mm（margin=0 时 dist<0 即产生接触）
    env.set_ball_state(
        np.array([0.5, 0.5, _BALL_RADIUS - 0.001]), np.zeros(3)
    )
    pairs = _contact_pairs(env.data)
    assert (min(ground, ball), max(ground, ball)) in pairs


# ---------- 2. set_arm_collision：pair_margin 通道仅切球—拍 ----------


def test_set_arm_collision_toggles_only_ball_racket_pair() -> None:
    """set_arm_collision 仅切换球—拍接触（pair_margin=-1.0），不关其他保护。

    独立环境（function 级）：本测试改 model.pair_margin，不与共享
    fixture 串状态。保护碰撞（球—机身）在球—拍关闭期间仍可观测。
    """
    scene = build_tennis_scene(
        _profile("left"), _parameters("left"), _task("left")
    )
    env = ProfileSimEnv(scene)
    model = env.model
    ball = scene.ball_geom_id
    racket = _geom_id(model, _RACKET_GEOM)
    torso = _geom_id(model, _TORSO_GEOM)
    ball_racket = (min(ball, racket), max(ball, racket))
    ball_torso = (min(ball, torso), max(ball, torso))

    # 定位模型中的球—拍显式 pair（3.2 为每个拍 geom 建立一个）
    pair_ids = [
        pid
        for pid in range(model.npair)
        if {int(model.pair_geom1[pid]), int(model.pair_geom2[pid])} == {ball, racket}
    ]
    assert pair_ids, "测试前提：带球模型必须含球—拍显式 pair"

    env.reset()
    # 初始（构造态）margin 为编译默认值 0，球—拍接触可观测
    assert all(float(model.pair_margin[pid]) == 0.0 for pid in pair_ids)
    tcp = env.get_ee_pos()
    env.set_ball_state(tcp, np.zeros(3))  # 球心放到拍面中心（穿透球半径）
    assert ball_racket in _contact_pairs(env.data)

    # 关闭：margin=-1.0，仅球—拍接触消失
    env.set_arm_collision(False)
    assert all(float(model.pair_margin[pid]) == -1.0 for pid in pair_ids)
    env.set_ball_state(tcp, np.zeros(3))
    assert ball_racket not in _contact_pairs(env.data)

    # 其他保护碰撞不受影响：球—机身接触在关闭期间仍可观测
    torso_xpos = np.array(env.data.geom_xpos[torso])
    env.set_ball_state(torso_xpos, np.zeros(3))
    assert ball_torso in _contact_pairs(env.data)

    # 恢复：margin 回编译默认，球—拍接触恢复；反复切换可逆
    env.set_arm_collision(True)
    assert all(float(model.pair_margin[pid]) == 0.0 for pid in pair_ids)
    env.set_ball_state(tcp, np.zeros(3))
    assert ball_racket in _contact_pairs(env.data)
    env.set_arm_collision(False)
    assert all(float(model.pair_margin[pid]) == -1.0 for pid in pair_ids)
    env.set_arm_collision(True)
    assert all(float(model.pair_margin[pid]) == 0.0 for pid in pair_ids)


# ---------- 3. 工厂：模型/data 独立性与初始化一致性 ----------


def test_factory_models_and_data_independent(envs_left: RobotEnvironments) -> None:
    """execution/planning 的 model 与 data 均不共享可变实例。"""
    ex, pl = envs_left.execution, envs_left.planning
    assert ex is not pl
    assert ex.model is not pl.model
    assert ex.data is not pl.data
    assert ex._scratch is not pl._scratch  # 试探步 scratch 也独立

    # 修改执行器配置不串实例：改 execution 增益，planning 不变
    ex_ids = np.asarray(ex.binding.actuator_ids, dtype=int)
    pl_ids = np.asarray(pl.binding.actuator_ids, dtype=int)
    pl_gain_before = pl.model.actuator_gainprm[pl_ids].copy()
    ex_gain_saved = ex.model.actuator_gainprm[ex_ids].copy()
    try:
        ex.model.actuator_gainprm[ex_ids, 0] = 123.0
        np.testing.assert_array_equal(
            pl.model.actuator_gainprm[pl_ids], pl_gain_before
        )
    finally:
        ex.model.actuator_gainprm[ex_ids] = ex_gain_saved  # 恢复共享 fixture

    # 碰撞 mask 不串实例：execution 关球—拍（pair_margin 通道），
    # planning 为裸模型无 pair 通道，几何掩码不受影响
    mask_before = pl.model.geom_contype.copy()
    margin_saved = ex.model.pair_margin.copy()
    ex.set_arm_collision(False)
    try:
        np.testing.assert_array_equal(pl.model.geom_contype, mask_before)
        assert pl.model.npair == 0  # 裸模型无显式 pair
    finally:
        ex.set_arm_collision(True)
        np.testing.assert_array_equal(ex.model.pair_margin, margin_saved)

    # data 不串实例：planning 步进，execution 的 live 不变（含物理球）
    ex_qpos = ex.data.qpos.copy()
    x = pl.get_arm_state().copy()
    x[: pl.NQ] += 0.03
    pl.set_arm_state(x)
    pl.step(np.full(pl.NQ, 0.1))
    np.testing.assert_array_equal(ex.data.qpos, ex_qpos)
    assert ex.data.time == 0.0


def test_factory_initial_state_consistent(envs_left: RobotEnvironments) -> None:
    """初始化 TCP/惯量/限位/高度一致：执行与规划模型逐名逐项对照。"""
    ex, pl = envs_left.execution, envs_left.planning
    # 共享 fixture 防御：前序测试可能移动过 data，先恢复初始态
    #（reset 不重置执行器配置，恢复的是 data/球/臂初始位形）
    ex.reset()
    pl.reset()
    m_ex, m_pl = ex.model, pl.model

    # 时间步长一致（仿真与规划同模型参数）
    assert m_ex.opt.timestep == m_pl.opt.timestep == float(
        _parameters("left").dt
    )

    # 初始 TCP 世界位置一致（reset 后 FK）
    np.testing.assert_allclose(
        ex.data.site_xpos[ex.binding.tcp_site_id],
        pl.data.site_xpos[pl.binding.tcp_site_id],
        rtol=0,
        atol=1e-12,
    )

    # 逐 body：质量/惯量/安装位姿（含高度）一致
    for b in range(m_pl.nbody):
        name = mujoco.mj_id2name(m_pl, mujoco.mjtObj.mjOBJ_BODY, b)
        b_ex = mujoco.mj_name2id(m_ex, mujoco.mjtObj.mjOBJ_BODY, name)
        assert b_ex >= 0, f"执行模型缺少 body {name}"
        assert m_ex.body_mass[b_ex] == m_pl.body_mass[b]
        np.testing.assert_array_equal(
            m_ex.body_inertia[b_ex], m_pl.body_inertia[b]
        )
        np.testing.assert_array_equal(m_ex.body_pos[b_ex], m_pl.body_pos[b])
        np.testing.assert_array_equal(
            m_ex.body_quat[b_ex], m_pl.body_quat[b]
        )

    # 关节限位一致（模型是位置限位唯一权威）
    for j_ex, j_pl in zip(ex.binding.joint_ids, pl.binding.joint_ids):
        np.testing.assert_array_equal(
            m_ex.jnt_range[j_ex], m_pl.jnt_range[j_pl]
        )

    # 力矩上限与控制范围一致（torque 模式默认 = ±torque_max）
    np.testing.assert_array_equal(ex.torque_max, pl.torque_max)
    assert ex.actuator_mode == 0 and pl.actuator_mode == 0
    np.testing.assert_array_equal(ex.control_range, pl.control_range)


def test_factory_execution_ball_initialized_from_task(
    envs_left: RobotEnvironments,
) -> None:
    """execution 物理球初态 = task.initial_ball_position/velocity（reset 基准）。"""
    task = _task("left")
    ex = envs_left.execution
    pos, vel = ex.get_ball_state()
    np.testing.assert_allclose(pos, task.initial_ball_position, rtol=0, atol=0)
    np.testing.assert_allclose(vel, task.initial_ball_velocity, rtol=0, atol=0)

    # 扰动后 reset 回到 task 初态
    ex.set_ball_state(np.zeros(3), np.zeros(3))
    ex.reset()
    np.testing.assert_allclose(
        ex.get_ball_state()[0], task.initial_ball_position, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        ex.get_ball_state()[1], task.initial_ball_velocity, rtol=0, atol=0
    )


def test_factory_planning_actuator_config_not_polluted(
    envs_left: RobotEnvironments,
) -> None:
    """execution 上切换执行器模式不影响 planning（独立 model 实例）。"""
    ex, pl = envs_left.execution, envs_left.planning
    kp = np.asarray(_parameters("left").kp, dtype=np.float64)
    kd = np.asarray(_parameters("left").kd, dtype=np.float64)

    pl_range_before = pl.control_range.copy()
    ex.configure_actuator_mode("position", kp, kd)
    try:
        assert ex.actuator_mode == 1
        assert pl.actuator_mode == 0  # planning 保持力矩模式
        np.testing.assert_array_equal(pl.control_range, pl_range_before)
    finally:
        ex.configure_actuator_mode("torque")  # 恢复共享 fixture
    assert ex.actuator_mode == 0


@pytest.mark.parametrize("side", SIDES)
def test_factory_both_sides_shapes_and_independence(side: str) -> None:
    """双侧工厂产物：类型/维数/独立性与球初态契约一致。"""
    envs = build_r1d_environments(
        _profile(side), _parameters(side), _task(side)
    )
    assert isinstance(envs, RobotEnvironments)
    assert isinstance(envs.execution, ProfileSimEnv)
    assert isinstance(envs.planning, BoundRobotEnv)
    # 规划副本以 ProfilePlanningEnv 交付（绑定球预测适配层）
    assert isinstance(envs.planning, ProfilePlanningEnv)

    assert envs.execution.NQ == 7
    assert envs.execution.NX == 14
    assert envs.execution.NU == 7
    assert envs.planning.NQ == 7
    # 带球模型 nq=14/nv=13；规划裸模型 nq=7/nv=7
    assert envs.execution.model.nq == 14
    assert envs.execution.model.nv == 13
    assert envs.planning.model.nq == 7
    assert envs.planning.model.nv == 7
    assert envs.execution.model is not envs.planning.model
    assert envs.execution.data is not envs.planning.data

    task = _task(side)
    pos, vel = envs.execution.get_ball_state()
    np.testing.assert_allclose(pos, task.initial_ball_position, rtol=0, atol=0)
    np.testing.assert_allclose(vel, task.initial_ball_velocity, rtol=0, atol=0)


# ---------- 4. 规划路径：禁用球接触、不承担真实碰撞判定 ----------


def test_planning_scene_is_ball_free_and_collision_free(
    envs_left: RobotEnvironments,
) -> None:
    """规划路径禁用球接触（裸模型无球），不承担真实碰撞判定（掩码全 0）。

    planning 的球接口是 BallPredictor 虚拟球（解析预测存储），与
    execution 的物理球分离：planning 球操作/预测不串 execution live。
    """
    pl = envs_left.planning
    pl_model = pl.model

    # 裸模型：无球 free joint、无显式 pair、几何碰撞掩码全 0
    for jid in range(pl_model.njnt):
        assert int(pl_model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    assert pl_model.npair == 0
    assert bool(np.all(pl_model.geom_contype == 0))
    assert bool(np.all(pl_model.geom_conaffinity == 0))

    # 虚拟球（解析预测存储）：set/get 往返
    pl.set_ball_state(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    pos, vel = pl.get_ball_state()
    np.testing.assert_allclose(pos, [1.0, 2.0, 3.0], rtol=0, atol=0)
    np.testing.assert_allclose(vel, [4.0, 5.0, 6.0], rtol=0, atol=0)
    pl.set_ball_vel(np.array([0.0, 0.0, -1.0]))
    assert np.allclose(pl.get_ball_vel(), [0.0, 0.0, -1.0])

    # 预测接口：形状正确、与独立 BallPredictor 一致
    pos_pred, vel_pred = pl.predict_ball_trajectory(
        np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0]), 10
    )
    assert pos_pred.shape == (10, 3) and vel_pred.shape == (10, 3)

    # 跨实例隔离：planning 球操作/预测不串 execution 物理球
    ex = envs_left.execution
    ex.reset()
    ex_qpos = ex.data.qpos.copy()
    ex_qvel = ex.data.qvel.copy()
    pl.set_ball_state(np.array([9.0, 9.0, 9.0]), np.array([9.0, 9.0, 9.0]))
    pl.predict_ball_trajectory(
        np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, -1.0]), 30
    )
    np.testing.assert_array_equal(ex.data.qpos, ex_qpos)
    np.testing.assert_array_equal(ex.data.qvel, ex_qvel)


# ---------- 5. 共享管线实际消费方法协议测试 ----------

# 执行环境消费清单（SimComponent/EpisodeRunner 主循环）：
# brief 指定 8 个 + 拍面线/角速度 + 球轨迹预测 + RobotEnv 协议成员。
_EXECUTION_CONSUMED_METHODS: tuple[str, ...] = (
    # brief 指定（球状态 / 拍面速度 / 碰撞开关 / 复合步进）
    "step_full",
    "get_ball_state",
    "set_ball_state",
    "set_ball_vel",
    "get_ball_pos",
    "get_ball_vel",
    "get_racket_face_speed",
    "set_arm_collision",
    # 拍面线/角速度、球轨迹预测
    "get_ee_vel",
    "get_ee_angular_vel",
    "predict_ball_trajectory",
    # RobotEnv 协议成员（求解器/限位/成本消费）
    "get_arm_state",
    "set_arm_state",
    "step",
    "step_from_state",
    "get_ee_pos",
    "get_ee_jacp",
    "get_ee_jacr",
    "get_ee_normal",
    "update_kinematics",
    "solve_ik",
    "reset",
    "configure_actuator_mode",
    "configure_feedforward",
)

# 规划副本消费清单（do_replan/异步规划线程）：虚拟球同步 + 球预测 +
# RobotEnv 协议成员（无物理球步进/拍面碰撞开关 —— 那些属执行路径）。
_PLANNING_CONSUMED_METHODS: tuple[str, ...] = (
    "get_ball_state",
    "set_ball_state",
    "set_ball_vel",
    "get_ball_pos",
    "get_ball_vel",
    "predict_ball_trajectory",
    "get_arm_state",
    "set_arm_state",
    "step",
    "step_from_state",
    "get_ee_pos",
    "get_ee_vel",
    "get_ee_jacp",
    "get_ee_jacr",
    "get_ee_normal",
    "update_kinematics",
    "solve_ik",
    "reset",
    "configure_actuator_mode",
    "configure_feedforward",
)

_CONSUMED_ATTRIBUTES: tuple[str, ...] = (
    "NQ",
    "NX",
    "NU",
    "dt",
    "control_range",
    "model",
    "data",
)


def test_pipeline_consumed_methods_contract(envs_left: RobotEnvironments) -> None:
    """消费方法逐项存在且可调用，少方法即失败；禁止 __getattr__ 兜底。"""
    for env, methods in (
        (envs_left.execution, _EXECUTION_CONSUMED_METHODS),
        (envs_left.planning, _PLANNING_CONSUMED_METHODS),
    ):
        for name in methods:
            # getattr_static 绕过实例 __getattr__，直接查类层定义
            static = inspect.getattr_static(env, name, None)
            assert static is not None, f"{type(env).__name__} 缺少方法 {name}"
            assert callable(getattr(env, name)), f"{name} 不可调用"
        for attr in _CONSUMED_ATTRIBUTES:
            assert hasattr(env, attr), f"{type(env).__name__} 缺少属性 {attr}"
        # 类层（含 MRO）不得定义 __getattr__ 兜底吞缺失方法
        for klass in type(env).__mro__:
            assert "__getattr__" not in vars(klass), (
                f"{klass.__name__} 定义了 __getattr__（禁止兜底吞缺失方法）"
            )
        # 满足 RobotEnv 协议
        assert isinstance(env, RobotEnv)


def test_racket_face_speed_returns_float_scalar(
    scene_env_left: tuple[ProfileSimEnv, TennisSceneBundle],
) -> None:
    """get_racket_face_speed 返回非负 float 标量（拍面中心线速度模）。"""
    env, _ = scene_env_left
    env.reset()
    speed = env.get_racket_face_speed()
    assert isinstance(speed, float)
    assert speed >= 0.0
    # 静止零速状态：拍面速度模为 0
    assert speed == pytest.approx(0.0, abs=1e-12)


# ---------- 6. 位置模式独立 MjData 参照（3.3 评审指针补验） ----------


def test_position_mode_step_from_state_independent_mjdata_reference() -> None:
    """位置模式 step_from_state 与独立 MjData 单步对照 atol=1e-10。

    3.3 评审指针：位置模式此前无独立 MjData 参照对照。本测试作为
    消费者按 bound_robot_env.py 模块 docstring 的 scratch 基态配方
    （位置模式分支）手工复现一步，逐项对照 atol=1e-10。
    独立环境（function 级）：本测试切换执行器模式，不与共享 fixture
    串状态。
    """
    scene = build_tennis_scene(
        _profile("left"), _parameters("left"), _task("left")
    )
    env = ProfileSimEnv(scene)
    model = env.model
    n = env.NQ
    kp = np.asarray(scene.robot.parameters.kp, dtype=np.float64)
    kd = np.asarray(scene.robot.parameters.kd, dtype=np.float64)
    env.configure_actuator_mode("position", kp, kd)

    qpos_idx = np.asarray(env.binding.qpos_indices, dtype=int)
    qvel_idx = np.asarray(env.binding.qvel_indices, dtype=int)
    act_ids = np.asarray(env.binding.actuator_ids, dtype=int)
    joint_ids = np.asarray(env.binding.joint_ids, dtype=int)
    torque_max = env.torque_max

    # 限位内平滑姿态 + 小速度（远离饱和边界，配方可对照）
    rng = np.random.default_rng(_RNG_SEED)
    lo = model.jnt_range[joint_ids, 0]
    hi = model.jnt_range[joint_ids, 1]
    q = lo + (hi - lo) * (0.4 + 0.2 * rng.random(n))
    qdot = 0.2 * (rng.random(n) - 0.5)
    x = np.concatenate([q, qdot])
    u = q + 0.05 * (rng.random(n) - 0.5)  # 目标角在当前角附近

    x_next = env.step_from_state(x, u)

    # —— 独立 MjData：按 scratch 基态配方（位置模式）手工复现一步 ——
    ref = mujoco.MjData(model)
    mujoco.mj_resetData(model, ref)
    ref.qpos[qpos_idx] = x[:n]
    ref.qvel[qvel_idx] = x[n:]
    q_now = np.array(ref.qpos[qpos_idx])
    # ctrl = 限位裁剪 + 误差限幅 |u-q| ≤ torque_max/kp
    u_clipped = np.clip(u, model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1])
    max_err = torque_max / kp
    ref.ctrl[act_ids] = q_now + np.clip(u_clipped - q_now, -max_err, max_err)
    # 前馈偏置力（位置模式默认启用）
    mujoco.mj_forward(model, ref)
    ref.qacc[:] = 0.0
    tau_bias = np.zeros(model.nv)
    mujoco.mj_rne(model, ref, 0, tau_bias)
    ref.qfrc_applied[qvel_idx] = tau_bias[qvel_idx]
    mujoco.mj_step(model, ref)

    x_ref = np.empty(2 * n)
    x_ref[:n] = ref.qpos[qpos_idx]
    x_ref[n:] = ref.qvel[qvel_idx]
    np.testing.assert_allclose(x_next, x_ref, rtol=0, atol=1e-10)
