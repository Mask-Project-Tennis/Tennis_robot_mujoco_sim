"""ProfileSimEnv 地面解析弹跳回归测试（T7 parity 修复）。

背景：R1D 执行环境（``ProfileSimEnv``）此前只让 MuJoCo 物理推进球，缺少
``RM65Env._handle_ball_bounce`` 的解析弹跳，导致执行侧球—地真实碰撞把切向
速度几乎全部耗散；而规划/发球模型（``BallPredictor`` /
``generate_ball_from_serve_box``）假设弹跳后 ``vz`` 反向乘恢复系数、
水平速度乘 0.95 —— 两侧球轨迹严重不一致，R1D 真实回合中球根本无法到达
目标区。

本测试锁定修复后的口径：
- handler 边界（``z < r & vz < 0``、``z >= r``、``vz >= 0``、切向摩擦
  0.95、restitution 来自共享常量）；
- ``step`` / ``step_full`` 后球状态确实应用了解析弹跳；
- 同一球初态下 R1D 与 ``RM65Env`` 的球位置/速度逐步一致（parity），
  至少覆盖一次弹跳；
- 不改变 R1D 6/7 维布局（nq=14 / nv=13 / nu=7）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.robot.constants import BOUNCE_RESTITUTION
from src.robot.robot_profile import load_robot_profile
from src.robot.runtime_parameters import load_runtime_parameters
from src.sim.profile_sim_env import ProfileSimEnv
from src.sim.rm65_env import RM65Env
from src.sim.sim_config import load_sim_task_config
from src.sim.tennis_scene import TennisSceneBundle, build_tennis_scene

REPO_ROOT = Path(__file__).resolve().parent.parent

# 球物理契约常量（与 RM65Env / BallPredictor / TennisScene 一致）
_BALL_RADIUS = 0.033
_TANGENTIAL_FRICTION = 0.95


@pytest.fixture(scope="module")
def scene_env_right() -> tuple[ProfileSimEnv, TennisSceneBundle]:
    """右臂带球执行环境 + 场景（module 级共享；本模块测试只做 data 层操作）。"""
    profile = load_robot_profile(
        REPO_ROOT / "configs" / "robots" / "r1d_right.yaml"
    )
    parameters = load_runtime_parameters(
        REPO_ROOT / "configs" / "robots" / "r1d_sim_runtime_right.yaml", profile
    )
    task = load_sim_task_config(
        REPO_ROOT / "configs" / "r1d_sim_right.yaml", profile
    )
    scene = build_tennis_scene(profile, parameters, task)
    return ProfileSimEnv(scene), scene


@pytest.fixture(scope="module")
def rm65_env() -> RM65Env:
    """RM65 执行环境（仅用于球轨迹 parity 对照，臂保持零力矩）。"""
    return RM65Env(REPO_ROOT / "src" / "robot" / "rm65_model.xml")


# ---------- 1. 布局/半径契约 ----------


def test_r1d_layout_and_ball_radius_contract(
    scene_env_right: tuple[ProfileSimEnv, TennisSceneBundle],
) -> None:
    """R1D 布局不变（nq=14/nv=13/nu=7），球半径读取并与 0.033 契约一致。"""
    env, _scene = scene_env_right
    assert env.NQ == 7
    assert env.NU == 7
    assert env.model.nq == 14
    assert env.model.nv == 13
    assert env._ball_radius == pytest.approx(_BALL_RADIUS, abs=1e-12)


# ---------- 2. handler 边界 ----------


def test_bounce_handler_applies_restitution_and_friction(
    scene_env_right: tuple[ProfileSimEnv, TennisSceneBundle],
) -> None:
    """z<r 且 vz<0：钳到球半径、vz 反向乘 BR、vx/vy 乘 0.95。"""
    env, _scene = scene_env_right
    env.reset()
    env.set_ball_state(
        np.array([2.5, -1.0, 0.01]), np.array([2.0, -1.0, -5.0])
    )

    env._handle_ball_bounce()

    pos, vel = env.get_ball_state()
    np.testing.assert_allclose(pos, [2.5, -1.0, _BALL_RADIUS], atol=1e-12)
    np.testing.assert_allclose(
        vel,
        [
            2.0 * _TANGENTIAL_FRICTION,
            -1.0 * _TANGENTIAL_FRICTION,
            5.0 * BOUNCE_RESTITUTION,
        ],
        atol=1e-12,
    )

    # 弹跳后 vz>0，再次调用不应重复反弹（幂等）
    env._handle_ball_bounce()
    pos2, vel2 = env.get_ball_state()
    np.testing.assert_allclose(pos2, pos, atol=0.0)
    np.testing.assert_allclose(vel2, vel, atol=0.0)


def test_bounce_handler_no_bounce_when_above_ground(
    scene_env_right: tuple[ProfileSimEnv, TennisSceneBundle],
) -> None:
    """z>=r 时即使 vz<0 也不触发解析弹跳。"""
    env, _scene = scene_env_right
    env.reset()
    env.set_ball_state(
        np.array([2.5, -1.0, 0.5]), np.array([2.0, -1.0, -5.0])
    )

    env._handle_ball_bounce()

    pos, vel = env.get_ball_state()
    np.testing.assert_allclose(pos, [2.5, -1.0, 0.5], atol=0.0)
    np.testing.assert_allclose(vel, [2.0, -1.0, -5.0], atol=0.0)


def test_bounce_handler_no_bounce_when_moving_up(
    scene_env_right: tuple[ProfileSimEnv, TennisSceneBundle],
) -> None:
    """z<r 但 vz>=0（已在上抛）时不做反弹。"""
    env, _scene = scene_env_right
    env.reset()
    env.set_ball_state(
        np.array([2.5, -1.0, 0.01]), np.array([2.0, -1.0, 1.0])
    )

    env._handle_ball_bounce()

    pos, vel = env.get_ball_state()
    np.testing.assert_allclose(pos, [2.5, -1.0, 0.01], atol=0.0)
    np.testing.assert_allclose(vel, [2.0, -1.0, 1.0], atol=0.0)


# ---------- 3. step/step_full 集成 ----------


def test_step_full_applies_analytic_bounce(
    scene_env_right: tuple[ProfileSimEnv, TennisSceneBundle],
) -> None:
    """快速自由落球经 step_full 后触发解析弹跳（弱物理反弹不足以解释）。

    使用 serve 量级的入射速度（vz=-5 m/s）。MuJoCo 软接触自身只产生约
    +1.5 m/s 的弱反弹；解析 handler（vz 反向乘 0.75）给出约 +3.7 m/s。
    因此检测到 vz>3.0 时即可确认解析弹跳被应用，并断言球心被钳到球半径。
    """
    env, _scene = scene_env_right
    env.reset()
    env.set_ball_state(
        np.array([2.5, -1.0, 0.5]), np.array([0.5, 0.3, -5.0])
    )

    bounce_step = -1
    for k in range(200):
        _arm, ball_pos, ball_vel = env.step_full(np.zeros(env.NU))
        if ball_vel[2] > 3.0:
            bounce_step = k
            assert ball_pos[2] == pytest.approx(_BALL_RADIUS, abs=1e-12)
            # 解析弹跳：vz ≈ |入射 vz| * BR（入射速度含重力增量，取宽松容差）
            assert 3.0 < ball_vel[2] < 5.0
            break

    assert bounce_step >= 0, "200 步内未检测到解析弹跳（vz>3.0）"


# ---------- 4. 与 RM65 球轨迹 parity ----------


def test_ball_trajectory_matches_rm65_env(
    scene_env_right: tuple[ProfileSimEnv, TennisSceneBundle],
    rm65_env: RM65Env,
) -> None:
    """同一球初态下 R1D（修复后）与 RM65 的球位置/速度逐步一致。

    球放在远离机器人的地面区域（x=2.5）；两环境臂均给零力矩，球只受重力
    与地面接触作用。修复后 R1D 的解析弹跳与 RM65 逐式一致，故两者球状态
    应逐步一致（浮点容差 1e-9），并至少覆盖一次弹跳。
    """
    env, _scene = scene_env_right
    pos0 = np.array([2.5, -1.0, 0.60])
    vel0 = np.array([1.2, 0.6, -3.0])

    env.reset()
    rm65_env.reset()
    env.set_ball_state(pos0, vel0)
    rm65_env.set_ball_state(pos0, vel0)

    prev_vz = float(vel0[2])
    bounce_count = 0
    bounce_max_vz = 0.0
    for k in range(160):
        env.step_full(np.zeros(env.NU))
        rm65_env.step(np.zeros(rm65_env.NU))

        pos_r1d, vel_r1d = env.get_ball_state()
        pos_rm65, vel_rm65 = rm65_env.get_ball_state()

        np.testing.assert_allclose(
            pos_r1d,
            pos_rm65,
            rtol=0.0,
            atol=1e-9,
            err_msg=f"第 {k} 步球位置不一致（R1D vs RM65）",
        )
        np.testing.assert_allclose(
            vel_r1d,
            vel_rm65,
            rtol=0.0,
            atol=1e-9,
            err_msg=f"第 {k} 步球速度不一致（R1D vs RM65）",
        )

        if prev_vz < 0.0 and vel_r1d[2] > 0.0:
            bounce_count += 1
            bounce_max_vz = max(bounce_max_vz, float(vel_r1d[2]))
            assert pos_r1d[2] == pytest.approx(_BALL_RADIUS, abs=1e-12)
        prev_vz = float(vel_r1d[2])

    assert bounce_count >= 1, "160 步内两环境均未发生地面弹跳，parity 未覆盖弹跳"
    # 入射速度约 4.3 m/s：解析弹跳给出 vz≈3.2，弱物理反弹仅约 1.5
    assert bounce_max_vz > 3.0, (
        "parity 覆盖的弹跳不是解析弹跳（反弹 vz 过小），请检查 handler"
    )
