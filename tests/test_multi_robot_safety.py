"""七维限位与独立预测碰撞保护测试（阶段 3 · Task 3.7）。

核心断言组（brief + 施工计划 Task 3.7）：
1. ``TestSevenDimLimits``：仅第七轴（索引 6）超 q/qdot/qddot/torque 的拒绝
   用例 —— check_step_feasibility / strict_braking_check /
   compute_trajectory_metrics（trajectory metrics）/
   check_limits_on_trajectory（qdot history + 旧 ``[6:]`` 切片 bug 必须暴露）
   全部按 limits 维数取状态；状态宽度不匹配显式报错。
2. ``TestRuntimeLimitsAdapter``：``RobotLimits.from_runtime_parameters``
   runtime → RobotLimits 适配 —— 完整参数才能构造；逐值直传（无 margin/
   scale 隐式缩放，模型 q 限位与人为运行约束互不覆盖）；control_range
   只作维数锚定，不顶替 torque_max / q 界。
3. ``TestHasForbiddenContact`` / ``TestForbiddenContactChecker``：合成模型
   的初态穿透 / 分离 / 一步预测后穿透负例；球—障碍接触不在禁止对内
   （球拍击球不视作禁止接触的同型证明）；无效禁止对必须失败；真实 R1D
   双侧场景 init_q 无接触 + 穿透构型检出；检查器使用独立数据，不触碰
   live 球/状态/时钟。
4. ``TestPredictiveSafetyInjection``：PredictiveSafetyFilter 可选碰撞检查
   注入 —— 无注入保持旧语义（全候选失败仍 is_safe=True）；注入后初态
   穿透立即不安全；预测穿透时力矩 beta 与位置插值候选逐一验证、全部
   失败返回不安全（"尝试了制动" ≠ "已验证安全"）；live 隔离；
   X 墙 body 名缺失必须失败（不再空列表空真放行）。

运行：``python -m pytest tests/test_multi_robot_safety.py -q``
"""

from __future__ import annotations

import dataclasses
from collections import deque
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest

from src.ilqt.robot_limits import (
    RobotLimits,
    check_limits_on_trajectory,
    check_step_feasibility,
    compute_trajectory_metrics,
    strict_braking_check,
)
from src.robot.robot_profile import load_robot_profile
from src.robot.runtime_parameters import load_runtime_parameters
from src.sim.sim_config import load_sim_task_config
from src.sim.tennis_scene import build_tennis_scene

# 仓库根目录（tests/ 的父目录）
REPO_ROOT = Path(__file__).resolve().parent.parent
SIDES = ("left", "right")

# R1D 已验证的穿透/干净构型（肩俯仰, 肘），scratch 扫描确定（双侧一致）：
# (0.4, 1.45) 与机身保护对接触；(0.4, 1.30) 干净；init_q 干净。
_PEN_PITCH = 0.4
_PEN_ELBOW = 1.45
_CLEAN_ELBOW = 1.30

# 合成碰撞模型（布局镜像真实场景语义：障碍在独立 body、臂铰链驱动、
# 球 freejoint 存在但不在禁止对内；MuJoCo 3.9.0 实测此布局接触可观测）
_SYNTH_XML = """
<mujoco model="synth_collision">
  <compiler angle="radian" />
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" />
  <worldbody>
    <body name="torso" pos="0.45 0.12 0">
      <geom name="obstacle" type="box" size="0.05 0.05 0.05" contype="1" conaffinity="1" />
    </body>
    <body name="mover" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" range="-3 3" />
      <geom name="arm_geom" type="capsule" fromto="0.30 0 0 0.50 0 0" size="0.02" mass="0.5" contype="1" conaffinity="1" />
    </body>
    <body name="ball_body" pos="0.30 -0.30 0.0">
      <freejoint name="ball_free" />
      <geom name="ball_geom" type="sphere" size="0.02" mass="0.058" contype="1" conaffinity="1" />
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="j1" gear="1" ctrlrange="-10 10" forcerange="-10 10" />
  </actuator>
</mujoco>
"""


# ---------- 共享环境构造（真实 R1D 双侧，缓存复用） ----------


@lru_cache(maxsize=2)
def _profile(side: str) -> Any:
    """加载并缓存该侧 RobotProfile。"""
    return load_robot_profile(REPO_ROOT / "configs" / "robots" / f"r1d_{side}.yaml")


@lru_cache(maxsize=2)
def _raw_parameters(side: str) -> Any:
    """加载 YAML 原样 runtime（q_lower/q_upper 仍为 None，未回填）。"""
    return load_runtime_parameters(
        REPO_ROOT / "configs" / "robots" / f"r1d_sim_runtime_{side}.yaml",
        _profile(side),
    )


@lru_cache(maxsize=2)
def _task(side: str) -> Any:
    """加载并缓存该侧 SimTaskConfig。"""
    return load_sim_task_config(REPO_ROOT / f"configs/r1d_sim_{side}.yaml", _profile(side))


@lru_cache(maxsize=2)
def _scene(side: str) -> Any:
    """构建并缓存该侧带球场景（q 限位已从模型回填进 robot.parameters）。"""
    return build_tennis_scene(_profile(side), _raw_parameters(side), _task(side))


@lru_cache(maxsize=4)
def _exec_env(side: str, mode: str) -> Any:
    """构建并缓存该侧带球执行环境（mode = torque / position）。"""
    from src.sim.profile_sim_env import ProfileSimEnv

    scene = _scene(side)
    env = ProfileSimEnv(scene)
    env.set_initial_ball_state(
        _task(side).initial_ball_position, _task(side).initial_ball_velocity
    )
    if mode == "position":
        pars = scene.robot.parameters
        env.configure_actuator_mode(
            "position",
            np.asarray(pars.kp, dtype=np.float64),
            np.asarray(pars.kd, dtype=np.float64),
        )
    return env


def _r1d_limits(side: str) -> RobotLimits:
    """从回填后的 runtime + 环境控制范围构建七维 RobotLimits。"""
    env = _exec_env(side, "torque")
    return RobotLimits.from_runtime_parameters(
        _scene(side).robot.parameters, env.control_range
    )


def _r1d_q(side: str, pitch: float, elbow: float) -> np.ndarray:
    """init_q 基础上覆写 (肩俯仰, 肘) 的七维关节角。"""
    q = np.array(_scene(side).robot.parameters.init_q, dtype=np.float64)
    q[0] = pitch
    q[3] = elbow
    return q


def _raw_contact_pairs(data: mujoco.MjData) -> set[tuple[int, int]]:
    """收集当前接触的 geom id 无序对（小 id 在前；与 3.2 测试同型）。"""
    return {
        (min(int(c.geom1), int(c.geom2)), max(int(c.geom1), int(c.geom2)))
        for i in range(data.ncon)
        for c in (data.contact[i],)
    }


# ============================================================================
#  1. 七维限位拒绝（q / qdot / qddot / torque 仅第七轴超限）
# ============================================================================


def _limits7(qddot_hard_reject: bool = True) -> RobotLimits:
    """构造七维测试限位（各关节同值，便于仅扰动第七轴）。"""
    return RobotLimits(
        q_lower=np.full(7, -1.0),
        q_upper=np.full(7, 1.0),
        qdot_max=np.full(7, 3.0),
        qddot_max=np.full(7, 50.0),
        u_min=np.full(7, -10.0),
        u_max=np.full(7, 10.0),
        qddot_hard_reject=qddot_hard_reject,
    )


class TestSevenDimLimits:
    """仅第七轴超限必须被全部检查路径拒绝（n 维化回归）。"""

    DT = 0.005

    def test_rejects_seventh_joint_q(self) -> None:
        """x_next[6]=1.5 > q_upper[6]=1 → 拒绝且 reason 指明 joint=6。"""
        limits = _limits7()
        x_prev = np.zeros(14)
        x_next = np.zeros(14)
        x_next[6] = 1.5  # 第七轴位置超上限（其余关节全零）
        ok, reason = check_step_feasibility(x_prev, x_next, np.zeros(7), limits, self.DT)
        assert not ok
        assert "joint=6" in reason

    def test_rejects_seventh_joint_qdot(self) -> None:
        """x_next[13]=4 > qdot_max[6]=3 → 硬拒绝 joint=6。"""
        limits = _limits7()
        x_prev = np.zeros(14)
        x_next = np.zeros(14)
        x_next[13] = 4.0  # 第七轴速度超限
        ok, reason = check_step_feasibility(
            x_prev, x_next, np.zeros(7), limits, self.DT, skip_qdot=False
        )
        assert not ok
        assert "qdot" in reason and "joint=6" in reason

    def test_rejects_seventh_joint_qddot_with_history(self) -> None:
        """qdot history 滑窗显示第七轴加速度 60 > 50 → 硬拒绝 joint=6。"""
        limits = _limits7(qddot_hard_reject=True)
        # 滑窗 3 点：0, 0, Δjoint7=0.6 → qddot_est = 0.6/(2*dt) = 60 > 50
        hist = deque(
            [np.zeros(7), np.zeros(7), np.concatenate([np.zeros(6), [0.6]])],
            maxlen=limits.qddot_window_size + 1,
        )
        x_prev = np.zeros(14)
        x_next = np.zeros(14)
        ok, reason = check_step_feasibility(
            x_prev, x_next, np.zeros(7), limits, self.DT,
            qdot_history=hist, skip_qdot=True, skip_qddot=False,
        )
        assert not ok
        assert "qddot" in reason and "joint=6" in reason

    def test_rejects_seventh_joint_torque(self) -> None:
        """u[6]=15 > u_max[6]=10（力矩模式）→ 拒绝 joint=6。"""
        limits = _limits7()
        x_prev = np.zeros(14)
        x_next = np.zeros(14)
        u = np.zeros(7)
        u[6] = 15.0
        ok, reason = check_step_feasibility(
            x_prev, x_next, u, limits, self.DT, actuator_mode=0
        )
        assert not ok
        assert "u upper" in reason and "joint=6" in reason

    def test_strict_braking_rejects_seventh_joint(self) -> None:
        """执行层 strict_braking_check 对第七轴 q / u 超限均拒绝。"""
        limits = _limits7()
        x_prev = np.zeros(14)

        x_bad_q = np.zeros(14)
        x_bad_q[6] = 1.5
        ok_q, reason_q = strict_braking_check(
            x_prev, x_bad_q, np.zeros(7), limits, self.DT
        )
        assert not ok_q and "joint=6" in reason_q

        u_bad = np.zeros(7)
        u_bad[6] = 15.0
        ok_u, reason_u = strict_braking_check(
            x_prev, np.zeros(14), u_bad, limits, self.DT, actuator_mode=0
        )
        assert not ok_u and "joint=6" in reason_u

        # 位置模式不检查 u 界（由 qdot/forcerange 覆盖），其余合法 → 通过
        ok_pos, _ = strict_braking_check(
            x_prev, np.zeros(14), u_bad, limits, self.DT, actuator_mode=1
        )
        assert ok_pos

    def test_clean_seven_dim_trajectory_passes(self) -> None:
        """干净七维轨迹（零速度、零控制）全检查通过。"""
        limits = _limits7()
        n_steps = 4
        X = np.zeros((n_steps + 1, 14))
        U = np.zeros((n_steps, 7))
        ok, reason = check_step_feasibility(
            X[0], X[1], U[0], limits, self.DT, actuator_mode=0
        )
        assert ok, reason
        ok_traj, k_bad, _ = check_limits_on_trajectory(X, U, limits, self.DT)
        assert ok_traj and k_bad == -1

    def test_trajectory_metrics_sees_seventh_joint_only(self) -> None:
        """仅第七轴 qdot 超限 → max_qdot_joint==6 且 ratio>1（metrics 不丢第七轴）。"""
        limits = _limits7()
        n_steps = 3
        X = np.zeros((n_steps + 1, 14))
        U = np.zeros((n_steps, 7))
        X[2][13] = 4.0  # 仅第七轴速度 4 > 3
        metrics = compute_trajectory_metrics(X, U, limits, self.DT)
        assert metrics.max_qdot_ratio == pytest.approx(4.0 / 3.0)
        assert metrics.max_qdot_joint == 6

    def test_check_limits_on_trajectory_rejects_seventh_joint_q(self) -> None:
        """X[1][6]=1.5（第七轴 q 超上限）→ 轨迹检查拒绝（暴露旧 [6:] 切片）。"""
        limits = _limits7()
        n_steps = 3
        X = np.zeros((n_steps + 1, 14))
        U = np.zeros((n_steps, 7))
        X[1][6] = 1.5  # 旧代码把它当 qdot[0] 读取 → 漏检
        ok, k_bad, reason = check_limits_on_trajectory(X, U, limits, self.DT)
        assert not ok
        assert k_bad == 0
        assert "joint=6" in reason

    def test_check_limits_on_trajectory_rejects_seventh_joint_qdot(self) -> None:
        """X[1][13]=4（第七轴 qdot 超限）→ 轨迹检查拒绝且 history 按维切片。"""
        limits = _limits7()
        n_steps = 3
        X = np.zeros((n_steps + 1, 14))
        U = np.zeros((n_steps, 7))
        X[1][13] = 4.0
        ok, k_bad, reason = check_limits_on_trajectory(X, U, limits, self.DT)
        assert not ok
        assert k_bad == 0
        assert "joint=6" in reason

    def test_state_width_mismatch_raises(self) -> None:
        """12 维状态配 7 维限位 → 显式 ValueError（不静默错切）。"""
        limits = _limits7()
        with pytest.raises(ValueError, match="状态向量"):
            check_step_feasibility(
                np.zeros(12), np.zeros(12), np.zeros(6), limits, self.DT
            )
        with pytest.raises(ValueError, match="状态向量"):
            strict_braking_check(
                np.zeros(12), np.zeros(12), np.zeros(6), limits, self.DT
            )


# ============================================================================
#  2. runtime → RobotLimits 适配
# ============================================================================


class TestRuntimeLimitsAdapter:
    """from_runtime_parameters：完整参数才能构造，逐值直传不互相覆盖。"""

    @pytest.mark.parametrize("side", SIDES)
    def test_complete_runtime_builds_exact_limits(self, side: str) -> None:
        """回填后的完整 runtime → 七维限位逐值等于 runtime 声明（无缩放）。"""
        scene = _scene(side)
        pars = scene.robot.parameters
        limits = RobotLimits.from_runtime_parameters(pars, _exec_env(side, "torque").control_range)

        assert limits.q_lower.shape == (7,)
        # 逐值直传：不加 margin、不乘 scale（人为运行约束不被隐式改写）
        np.testing.assert_array_equal(limits.q_lower, pars.q_lower)
        np.testing.assert_array_equal(limits.q_upper, pars.q_upper)
        np.testing.assert_array_equal(limits.qdot_max, pars.qdot_max)
        np.testing.assert_array_equal(limits.qddot_max, pars.qddot_max)
        np.testing.assert_array_equal(limits.u_min, -np.asarray(pars.torque_max))
        np.testing.assert_array_equal(limits.u_max, np.asarray(pars.torque_max))
        assert limits.max_tcp_speed == pars.max_tcp_speed
        # 派生量：dq_max = qdot_max * dt * 0.5（缺省比例）
        np.testing.assert_allclose(
            limits.dq_max, np.asarray(pars.qdot_max) * pars.dt * 0.5
        )
        # 模型 q 限位权威（阶段 2 回填规则）未被改写
        model = scene.robot.model
        jids = list(scene.robot.binding.joint_ids)
        np.testing.assert_allclose(
            limits.q_lower, model.jnt_range[jids, 0], atol=1e-12
        )
        np.testing.assert_allclose(
            limits.q_upper, model.jnt_range[jids, 1], atol=1e-12
        )

    @pytest.mark.parametrize("side", SIDES)
    def test_incomplete_runtime_rejected(self, side: str) -> None:
        """q_lower/q_upper 为 None（未回填）或 max_tcp_speed 缺失 → 拒绝构造。"""
        raw = _raw_parameters(side)  # YAML 原样：q_lower/q_upper = None
        ctrl = _exec_env(side, "torque").control_range
        with pytest.raises(ValueError, match="q_lower|完整"):
            RobotLimits.from_runtime_parameters(raw, ctrl)

        filled = _scene(side).robot.parameters
        no_tcp = dataclasses.replace(filled, max_tcp_speed=None)
        with pytest.raises(ValueError, match="max_tcp_speed|完整"):
            RobotLimits.from_runtime_parameters(no_tcp, ctrl)

    @pytest.mark.parametrize("side", SIDES)
    def test_no_cross_override_with_control_range(self, side: str) -> None:
        """control_range（如位置模式关节范围）不顶替 torque_max / q 界。"""
        pars = _scene(side).robot.parameters
        # 模拟位置模式 control_range（关节范围而非力矩范围）
        pos_range = np.stack(
            [np.asarray(pars.q_lower), np.asarray(pars.q_upper)], axis=1
        )
        limits = RobotLimits.from_runtime_parameters(pars, pos_range)
        np.testing.assert_array_equal(limits.u_max, np.asarray(pars.torque_max))
        np.testing.assert_array_equal(limits.q_lower, np.asarray(pars.q_lower))

        # 维数不一致（6 行 control_range 配七维参数）→ 拒绝
        with pytest.raises(ValueError, match="control_range"):
            RobotLimits.from_runtime_parameters(pars, pos_range[:6])


# ============================================================================
#  3. has_forbidden_contact（合成模型 + 真实 R1D 场景）
# ============================================================================


@lru_cache(maxsize=1)
def _synth_model() -> mujoco.MjModel:
    """编译并缓存合成碰撞模型（接触窗口 6°–24°，q=0 干净）。"""
    return mujoco.MjModel.from_xml_string(_SYNTH_XML)


def _synth_ids() -> tuple[int, int, int]:
    """返回 (obstacle, arm_geom, ball_geom) 的 geom id。"""
    model = _synth_model()
    return (
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")),
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "arm_geom")),
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")),
    )


def _synth_forward(q: float | None = None, ball_at_obstacle: bool = False) -> mujoco.MjData:
    """独立 MjData 上写入指定关节角/球位并 mj_forward（返回该 data）。"""
    model = _synth_model()
    data = mujoco.MjData(model)
    if q is not None:
        data.qpos[0] = q
    if ball_at_obstacle:
        ball_jnt = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        )
        adr = int(model.jnt_qposadr[ball_jnt])
        data.qpos[adr : adr + 3] = [0.45, 0.12, 0.0]
    mujoco.mj_forward(model, data)
    return data


def _synth_first_contact_angle_deg() -> float:
    """扫描找首个出现 arm—obstacle 接触的关节角（度，自校验几何）。"""
    model = _synth_model()
    data = mujoco.MjData(model)
    obs_id, arm_id, _ball_id = _synth_ids()
    pair = (min(arm_id, obs_id), max(arm_id, obs_id))
    for qdeg in range(2, 60, 2):
        mujoco.mj_resetData(model, data)
        data.qpos[0] = np.deg2rad(qdeg)
        mujoco.mj_forward(model, data)
        if pair in _raw_contact_pairs(data):
            return float(qdeg)
    raise AssertionError("合成模型扫描未找到 arm—obstacle 接触（几何漂移）")


class TestHasForbiddenContact:
    """has_forbidden_contact：只查预先解析的禁止对；无效对必须失败。"""

    def test_separated_state_false(self) -> None:
        """分离位形（q=0）→ 无禁止接触。"""
        from src.sim.collision_checks import has_forbidden_contact

        model = _synth_model()
        obs_id, arm_id, _ball_id = _synth_ids()
        data = _synth_forward(q=0.0)
        assert not has_forbidden_contact(model, data, ((arm_id, obs_id),))

    def test_initial_penetration_true(self) -> None:
        """已有初态穿透（接触窗口内关节角）→ True。"""
        from src.sim.collision_checks import has_forbidden_contact

        model = _synth_model()
        obs_id, arm_id, _ball_id = _synth_ids()
        q_in = np.deg2rad(_synth_first_contact_angle_deg())
        data = _synth_forward(q=q_in)
        assert data.ncon > 0, "接触窗口内应存在物理接触（几何自校验）"
        assert has_forbidden_contact(model, data, ((arm_id, obs_id),))

    def test_predicted_after_step_true(self) -> None:
        """一步预测后穿透：q=0 + 面向接触窗口的初速 → mj_step 后终态检出。"""
        from src.sim.collision_checks import has_forbidden_contact

        model = _synth_model()
        obs_id, arm_id, _ball_id = _synth_ids()
        q_in = np.deg2rad(_synth_first_contact_angle_deg())
        data = mujoco.MjData(model)
        data.qvel[0] = q_in * 1.3 / float(model.opt.timestep)  # 一步越过接触边界
        data.ctrl[0] = 0.0
        mujoco.mj_step(model, data)
        # mj_step 后接触数组停留在步前位形 —— 显式刷新到预测终态再查询
        mujoco.mj_forward(model, data)
        assert has_forbidden_contact(model, data, ((arm_id, obs_id),))

    def test_ball_contact_not_in_pairs_ignored(self) -> None:
        """球—障碍接触存在但不在禁止对内 → False（球拍击球同型语义）。"""
        from src.sim.collision_checks import has_forbidden_contact

        model = _synth_model()
        obs_id, arm_id, ball_id = _synth_ids()
        data = _synth_forward(q=0.0, ball_at_obstacle=True)
        assert data.ncon > 0, "球放障碍中心应产生物理接触（几何自校验）"
        # 只查 arm—obstacle 对：球接触被忽略
        assert not has_forbidden_contact(model, data, ((arm_id, obs_id),))
        # 显式列出球对才报真
        assert has_forbidden_contact(
            model, data, ((min(ball_id, obs_id), max(ball_id, obs_id)),)
        )

    def test_invalid_pairs_raise(self) -> None:
        """越界 id / 两端同 geom → ValueError（无效禁止对必须失败）。"""
        from src.sim.collision_checks import has_forbidden_contact

        model = _synth_model()
        data = _synth_forward(q=0.0)
        ngeom = int(model.ngeom)
        with pytest.raises(ValueError, match="越界"):
            has_forbidden_contact(model, data, ((ngeom, 0),))
        with pytest.raises(ValueError, match="同一"):
            has_forbidden_contact(model, data, ((1, 1),))

    def test_empty_pairs_false(self) -> None:
        """空禁止对元组为合法空集语义（查询层）；注入层的缺失拒绝另测。"""
        from src.sim.collision_checks import has_forbidden_contact

        model = _synth_model()
        data = _synth_forward(q=0.0)
        assert not has_forbidden_contact(model, data, ())


class TestForbiddenContactChecker:
    """ForbiddenContactChecker：独立数据检查器 + 真实 R1D 场景。"""

    @pytest.mark.parametrize("side", SIDES)
    def test_r1d_init_clean_and_penetration_detected(self, side: str) -> None:
        """真实 R1D 场景：init_q 无接触；已验证穿透构型检出。"""
        from src.sim.collision_checks import ForbiddenContactChecker

        scene = _scene(side)
        checker = ForbiddenContactChecker(
            scene.robot.model,
            np.asarray(scene.robot.binding.qpos_indices),
            scene.protected_geom_pairs,
        )
        assert not checker.has_contact_at_q(
            np.asarray(scene.robot.parameters.init_q)
        )
        assert checker.has_contact_at_q(_r1d_q(side, _PEN_PITCH, _PEN_ELBOW))
        # 干净邻位（穿透边界外侧）无接触
        assert not checker.has_contact_at_q(_r1d_q(side, _PEN_PITCH, _CLEAN_ELBOW))

    @pytest.mark.parametrize("side", SIDES)
    def test_r1d_one_step_predicted_penetration(self, side: str) -> None:
        """真实动力学一步预测：干净位形 + 面向机身的肘部初速 → 预测态穿透。"""
        from src.sim.collision_checks import ForbiddenContactChecker

        scene = _scene(side)
        env = _exec_env(side, "torque")
        checker = ForbiddenContactChecker(
            scene.robot.model,
            np.asarray(scene.robot.binding.qpos_indices),
            scene.protected_geom_pairs,
        )
        q_clean = _r1d_q(side, _PEN_PITCH, _CLEAN_ELBOW)
        qdot = np.zeros(7)
        qdot[3] = (_PEN_ELBOW - _CLEAN_ELBOW) * 1.3 / env.dt
        x_pred = env.step_from_state(np.concatenate([q_clean, qdot]), np.zeros(7))
        assert not checker.has_contact_at_q(q_clean)
        assert checker.has_contact_at_q(x_pred[:7])

    def test_from_env_uses_binding(self) -> None:
        """from_env 从环境 binding 取臂 qpos 地址；缺 binding 的环境显式失败。"""
        from src.sim.collision_checks import ForbiddenContactChecker

        env = _exec_env("left", "torque")
        scene = _scene("left")
        checker = ForbiddenContactChecker.from_env(env, scene.protected_geom_pairs)
        assert not checker.has_contact_at_q(np.asarray(scene.robot.parameters.init_q))

        fake = SimpleNamespace(model=env.model)  # 无 binding
        with pytest.raises(ValueError, match="binding"):
            ForbiddenContactChecker.from_env(fake, scene.protected_geom_pairs)

    def test_checker_validates_pairs_and_indices(self) -> None:
        """空禁止对 / 越界 id / 同 geom 对 / 非法 qpos 索引 → ValueError。"""
        from src.sim.collision_checks import ForbiddenContactChecker

        model = _synth_model()
        with pytest.raises(ValueError, match="禁止对"):
            ForbiddenContactChecker(model, np.array([0]), ())
        with pytest.raises(ValueError, match="越界"):
            ForbiddenContactChecker(model, np.array([0]), ((int(model.ngeom), 0),))
        with pytest.raises(ValueError, match="同一"):
            ForbiddenContactChecker(model, np.array([0]), ((1, 1),))
        with pytest.raises(ValueError, match="qpos"):
            ForbiddenContactChecker(model, np.array([int(model.nq)]), ((0, 1),))

    def test_checker_q_shape_validated(self) -> None:
        """q 维数不符 / 含 NaN → ValueError。"""
        from src.sim.collision_checks import ForbiddenContactChecker

        scene = _scene("left")
        checker = ForbiddenContactChecker(
            scene.robot.model,
            np.asarray(scene.robot.binding.qpos_indices),
            scene.protected_geom_pairs,
        )
        with pytest.raises(ValueError, match="shape"):
            checker.has_contact_at_q(np.zeros(6))
        with pytest.raises(ValueError, match="有限"):
            checker.has_contact_at_q(np.full(7, np.nan))

    @pytest.mark.parametrize("side", SIDES)
    def test_checker_does_not_touch_live(self, side: str) -> None:
        """检查器查询前后 live 球/臂状态/时间逐项不变（独立数据）。"""
        from src.sim.collision_checks import ForbiddenContactChecker

        env = _exec_env(side, "torque")
        scene = _scene(side)
        checker = ForbiddenContactChecker.from_env(env, scene.protected_geom_pairs)
        env.reset()
        qpos_before = env.data.qpos.copy()
        qvel_before = env.data.qvel.copy()
        time_before = float(env.data.time)
        checker.has_contact_at_q(np.asarray(scene.robot.parameters.init_q))
        checker.has_contact_at_q(_r1d_q(side, _PEN_PITCH, _PEN_ELBOW))
        np.testing.assert_array_equal(env.data.qpos, qpos_before)
        np.testing.assert_array_equal(env.data.qvel, qvel_before)
        assert env.data.time == time_before


# ============================================================================
#  4. PredictiveSafetyFilter 可选注入（制动 ≠ 已验证安全）
# ============================================================================


class _StubChecker:
    """记录调用并可编程响应的检查器桩（duck-type has_contact_at_q）。"""

    def __init__(self, initial: bool, predicted: bool) -> None:
        self.calls: list[np.ndarray] = []
        self._initial = initial
        self._predicted = predicted

    def has_contact_at_q(self, q: np.ndarray) -> bool:
        self.calls.append(np.asarray(q).copy())
        # 首个调用 = 初态检查；其余 = 预测候选验证
        return self._initial if len(self.calls) == 1 else self._predicted


class TestPredictiveSafetyInjection:
    """可选碰撞检查注入：默认不变，注入后"尝试了制动"≠"已验证安全"。"""

    @pytest.mark.parametrize("side", SIDES)
    def test_safe_command_passes_with_checker(self, side: str) -> None:
        """init_q + 界内小力矩 + 无接触 → is_safe=True。"""
        from src.ilqt.components.predictive_safety import PredictiveSafetyFilter
        from src.sim.collision_checks import ForbiddenContactChecker

        env = _exec_env(side, "torque")
        scene = _scene(side)
        checker = ForbiddenContactChecker.from_env(env, scene.protected_geom_pairs)
        safety = PredictiveSafetyFilter(
            env, _r1d_limits(side),
            is_position_mode=False, arm_nq=7, enable_x_wall=False,
            contact_checker=checker,
        )
        arm_state = env.reset()
        safe_u, is_safe = safety.filter(np.full(7, 2.0), arm_state)
        assert is_safe
        np.testing.assert_allclose(safe_u, np.full(7, 2.0))

    @pytest.mark.parametrize("side", SIDES)
    def test_initial_penetration_unsafe_and_stops(self, side: str) -> None:
        """初态已穿透 → 立即不安全（停止回合），不给未验证制动 is_safe=True。"""
        from src.ilqt.components.predictive_safety import PredictiveSafetyFilter
        from src.sim.collision_checks import ForbiddenContactChecker

        env = _exec_env(side, "torque")
        scene = _scene(side)
        checker = ForbiddenContactChecker.from_env(env, scene.protected_geom_pairs)
        safety = PredictiveSafetyFilter(
            env, _r1d_limits(side),
            is_position_mode=False, arm_nq=7, enable_x_wall=False,
            contact_checker=checker,
        )
        arm_state = np.concatenate([_r1d_q(side, _PEN_PITCH, _PEN_ELBOW), np.zeros(7)])
        safe_u, is_safe = safety.filter(np.full(7, 2.0), arm_state)
        assert not is_safe
        assert safety.emergency_stop_count == 1
        # 力矩模式紧急制动 = -20*qdot（零速 → 零力矩）
        np.testing.assert_allclose(safe_u, np.zeros(7))

    def test_predicted_contact_verifies_every_candidate_torque(self) -> None:
        """预测穿透：每个力矩 beta 候选都实际验证 → 全失败 → is_safe=False。"""
        from src.ilqt.components.predictive_safety import PredictiveSafetyFilter

        env = _exec_env("left", "torque")
        stub = _StubChecker(initial=False, predicted=True)
        safety = PredictiveSafetyFilter(
            env, _r1d_limits("left"),
            is_position_mode=False, arm_nq=7, enable_x_wall=False,
            contact_checker=stub,
        )
        arm_state = env.reset()
        qdot = arm_state[7:]
        safe_u, is_safe = safety.filter(np.full(7, 2.0), arm_state)
        assert not is_safe, "全部候选预测穿透时不得返回安全"
        assert safety.emergency_stop_count == 1
        np.testing.assert_allclose(safe_u, -20.0 * qdot)
        # 初态 1 次 + 6 个候选（beta=1.0 + 5 档递降）逐一经预测态验证
        assert len(stub.calls) == 7
        predicted = stub.calls[1:]
        distinct = {tuple(np.round(q, 9)) for q in predicted}
        assert len(distinct) >= 3, "力矩 beta 候选应产生多个不同预测态"

    def test_predicted_contact_verifies_position_interpolation(self) -> None:
        """位置模式：每个插值候选 q+beta*(u-q) 同样逐一验证 → 全失败不安全。"""
        from src.ilqt.components.predictive_safety import PredictiveSafetyFilter

        env = _exec_env("left", "position")
        stub = _StubChecker(initial=False, predicted=True)
        safety = PredictiveSafetyFilter(
            env, _r1d_limits("left"),
            is_position_mode=True, arm_nq=7, enable_x_wall=False,
            contact_checker=stub,
        )
        arm_state = env.reset()
        q = arm_state[:7].copy()
        u_cmd = q + 0.3
        safe_u, is_safe = safety.filter(u_cmd, arm_state)
        assert not is_safe
        assert safety.emergency_stop_count == 1
        np.testing.assert_allclose(safe_u, q)  # 位置模式紧急制动 = hold q
        assert len(stub.calls) == 7
        predicted = stub.calls[1:]
        distinct = {tuple(np.round(x, 9)) for x in predicted}
        assert len(distinct) >= 3, "位置插值候选应产生多个不同预测态"

    def test_no_injection_keeps_legacy_emergency_safe(self) -> None:
        """无注入（RM65 默认语义）：全部 beta 失败 → 紧急制动仍 is_safe=True。"""
        from src.ilqt.components.predictive_safety import PredictiveSafetyFilter

        env = _exec_env("left", "torque")
        limits = _r1d_limits("left")
        limits.q_upper = np.full(7, -10.0)  # 人为收紧使任何预测都超上限
        safety = PredictiveSafetyFilter(
            env, limits, is_position_mode=False, arm_nq=7, enable_x_wall=False,
        )
        arm_state = env.reset()
        qdot = arm_state[7:]
        safe_u, is_safe = safety.filter(np.full(7, 2.0), arm_state)
        assert is_safe, "无注入时保持旧 RM65 语义（紧急制动视为安全）"
        assert safety.emergency_stop_count == 1
        np.testing.assert_allclose(safe_u, -20.0 * qdot)

    @pytest.mark.parametrize("side", SIDES)
    def test_filter_live_isolation_with_checker(self, side: str) -> None:
        """注入模式下 filter() 前后 live 球/臂/时钟逐项不变。"""
        from src.ilqt.components.predictive_safety import PredictiveSafetyFilter
        from src.sim.collision_checks import ForbiddenContactChecker

        env = _exec_env(side, "torque")
        scene = _scene(side)
        checker = ForbiddenContactChecker.from_env(env, scene.protected_geom_pairs)
        safety = PredictiveSafetyFilter(
            env, _r1d_limits(side),
            is_position_mode=False, arm_nq=7, enable_x_wall=False,
            contact_checker=checker,
        )
        arm_state = env.reset()
        qpos_before = env.data.qpos.copy()
        time_before = float(env.data.time)
        ctrl_before = env.data.ctrl.copy()
        _, is_safe = safety.filter(np.full(7, 2.0), arm_state)
        assert is_safe
        np.testing.assert_array_equal(env.data.qpos, qpos_before)
        np.testing.assert_array_equal(env.data.ctrl, ctrl_before)
        assert env.data.time == time_before

    def test_x_wall_missing_bodies_fail_construction(self) -> None:
        """R1D 模型无 RM65 X 墙 body 名 → 构造失败（不再空列表空真放行）。"""
        from src.ilqt.components.predictive_safety import PredictiveSafetyFilter

        env = _exec_env("left", "torque")
        with pytest.raises(ValueError, match="X.*墙|body"):
            PredictiveSafetyFilter(
                env, _r1d_limits("left"),
                is_position_mode=False, arm_nq=7, enable_x_wall=True,
            )
