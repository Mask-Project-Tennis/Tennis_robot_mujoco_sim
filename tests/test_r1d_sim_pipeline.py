"""R1D 共享执行管线与物理接触判定测试（阶段 3 · Task 3.9）。

核心断言组（brief + 施工计划 Task 3.9）：
1. ``TestRunnerContract``：EpisodeRunner 短回合契约 —— 感知为物理球状态、
   规划控制为 ``(7,)``、安全后才执行、每次执行时间只推进约定 dt（R1D 不走
   旧 X 墙 PD 的额外 step_full）；停止分支可观察。
2. ``TestContactByID``：只按解析出的 ball/racket geom ID 判定**物理**接触
   （``dist <= 0`` 真实接触过滤，钉死 MuJoCo 3.9 斜圆柱显式 pair 在
   margin=-1 下正 gap 仍生成接触点的怪象 —— 窗口关闭期近距离飞球不得
   误判命中）；球碰地面/机身不是命中；球与连杆重叠（臂—球掩码关，无
   contact）不是命中；正误差无接触不是命中；主动命中以接触时拍速
   >0.3 m/s（``get_racket_face_speed``）与被动接触分开记录。
3. ``TestCollisionWindow``：碰撞窗口只切球—拍（geom contype/conaffinity
   全程不变，机身碰撞在窗口关闭期间仍可观测）；窗口关时球穿拍不判定。
4. ``TestProtocolAndLegacy``：``SimExecutionEnv`` 协议（RM65Env /
   ProfileSimEnv 双侧满足）；SimComponent 接触 ID 只给一半必须 ValueError；
   缺省构造走旧子串路径（R1D 模型 geom 名恰含 ball/racket 子串，缺省
   参数仍可判定，证明旧参数兼容）；``classify_hit_result`` 机器人名称
   参数（缺省 RM-65 文本不变）。

运行：``python -m pytest tests/test_r1d_sim_pipeline.py -q``
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.ilqt.components.protocols import (
    ExecutorComponent,
    PerceptionComponent,
    SafetyComponent,
    SimExecutionEnv,
)
from src.ilqt.components.sim_component import SimComponent
from src.ilqt.components.sim_perception import SimPerception
from src.ilqt.episode_runner import EpisodeRunner
from src.ilqt.mpc_controller import MPCStepResult
from src.ilqt.robot_limits import RobotLimits
from src.ilqt.tube_types import TubeConfig
from src.sim.collision_checks import ForbiddenContactChecker
from src.sim.env_factory import RobotEnvironments, build_r1d_environments
from src.sim.hit_detection import classify_hit_result, determine_hit_from_type
from src.robot.robot_profile import load_robot_profile
from src.robot.runtime_parameters import load_runtime_parameters
from src.sim.sim_config import load_sim_task_config

# 仓库根目录（tests/ 的父目录）
REPO_ROOT = Path(__file__).resolve().parent.parent
SIDES = ("left", "right")


# ============================================================================
#  共享组装（真实 R1D 双侧环境，缓存复用；每测试自行 reset 归位）
# ============================================================================


@lru_cache(maxsize=2)
def _envs(side: str) -> RobotEnvironments:
    """构建并缓存该侧环境对（execution 带球 + planning 副本）。"""
    profile = load_robot_profile(REPO_ROOT / "configs" / "robots" / f"r1d_{side}.yaml")
    parameters = load_runtime_parameters(
        REPO_ROOT / "configs" / "robots" / f"r1d_sim_runtime_{side}.yaml", profile
    )
    task = load_sim_task_config(REPO_ROOT / f"configs/r1d_sim_{side}.yaml", profile)
    return build_r1d_environments(profile, parameters, task)


@lru_cache(maxsize=2)
def _limits(side: str) -> RobotLimits:
    """该侧七维 RobotLimits（回填后 runtime + 执行环境 control_range）。"""
    env = _envs(side).execution
    return RobotLimits.from_runtime_parameters(env.scene.robot.parameters, env.control_range)


def _checker(side: str) -> ForbiddenContactChecker:
    """该侧禁止对检查器（独立 MjData，live 零污染）。"""
    env = _envs(side).execution
    return ForbiddenContactChecker.from_env(env, env.scene.protected_geom_pairs)


def _reset_env(side: str) -> Any:
    """reset 执行环境并确保球—拍碰撞窗口处于开启基准态。

    共享缓存实例的跨测试归位点：臂/球恢复 reset 基准，pair_margin
    恢复编译默认（部分测试直接关闭窗口后由本函数下一轮归位）。
    """
    env = _envs(side).execution
    env.reset()
    env.set_arm_collision(True)
    return env


class FakeSevenDimMPC:
    """七维 Mock MPC —— 提供 SimComponent / EpisodeRunner 消费的最小接口。

    SimComponent 只读 ``_k_hit`` / ``_p_hit`` / ``_n_des`` 三个规划属性；
    EpisodeRunner 消费 start/step/done/stop。``done_after`` 步后 done。
    """

    def __init__(
        self,
        n: int = 7,
        k_hit: int = 0,
        done_after: int = 3,
        p_hit: np.ndarray | None = None,
    ) -> None:
        self._n = n
        self._k_hit = k_hit
        self._p_hit = np.zeros(3) if p_hit is None else np.asarray(p_hit, dtype=float)
        self._n_des = np.array([0.0, 0.0, 1.0])
        self._count = 0
        self._done_after = done_after
        self._done = False

    def start(self, ball_pos: Any, ball_vel: Any, arm_state: Any) -> None:
        """记录启动（Mock，无规划）。"""

    def step(self, ball_pos: Any, ball_vel: Any, arm_state: Any) -> MPCStepResult:
        """返回零控制 (n,)，第 done_after 步 done。"""
        self._count += 1
        if self._count >= self._done_after:
            self._done = True
        return MPCStepResult(
            u_cmd=np.zeros(self._n), phase="near", k_hit=self._k_hit,
        )

    def stop(self) -> None:
        """Mock 停止。"""

    @property
    def done(self) -> bool:
        """回合是否结束。"""
        return self._done


class UnsafeSafety:
    """Mock 安全 —— 总是拒绝（停止分支可观察性测试）。"""

    def filter(self, u_cmd: Any, arm_state: Any) -> tuple[np.ndarray, bool]:
        """恒不安全。"""
        return np.asarray(u_cmd), False


def _make_component(
    side: str,
    mpc: FakeSevenDimMPC,
    always_collision: bool = False,
    contact_ids: bool = True,
) -> SimComponent:
    """构造 R1D 配置的 SimComponent（力矩模式，无 X 墙 body）。"""
    env = _envs(side).execution
    scene = env.scene
    return SimComponent(
        env,
        mpc,
        _limits(side),
        init_q=np.asarray(scene.robot.parameters.init_q, dtype=np.float64),
        is_position_mode=False,
        hard_x_body_ids=[],
        initial_hitting_tube=None,
        tube_cfg=TubeConfig(),
        dt=float(env.dt),
        always_collision=always_collision,
        ball_geom_id=scene.ball_geom_id if contact_ids else None,
        racket_geom_ids=scene.racket_geom_ids if contact_ids else None,
    )


def _contact_pairs(env: Any) -> set[tuple[int, int]]:
    """当前 live data 的接触 geom id 无序对集合（小 id 在前）。"""
    return {
        (min(int(c.geom1), int(c.geom2)), max(int(c.geom1), int(c.geom2)))
        for i in range(env.data.ncon)
        for c in (env.data.contact[i],)
    }


def _protected_geom_frequencies(scene: Any) -> dict[int, int]:
    """统计保护对中各 geom 出现次数（机身 geom 与所有对共现）。"""
    counts: dict[int, int] = {}
    for pair in scene.protected_geom_pairs:
        for g in pair:
            counts[int(g)] = counts.get(int(g), 0) + 1
    return counts


def _torso_geom_id(scene: Any) -> int:
    """机身保护 geom id（在全部保护对中共现的那个，非拍/非球）。"""
    counts = _protected_geom_frequencies(scene)
    return next(
        g for g, c in counts.items()
        if c == len(scene.protected_geom_pairs)
        and g not in scene.racket_geom_ids and g != scene.ball_geom_id
    )


def _arm_link_geom_id(scene: Any) -> int:
    """臂连杆保护 geom id（只在单个保护对中出现，非拍/非球）。"""
    counts = _protected_geom_frequencies(scene)
    return next(
        g for g, c in counts.items()
        if c == 1 and g not in scene.racket_geom_ids and g != scene.ball_geom_id
    )


def _run_short_episode(
    side: str, safety: Any, hooks: list[Any] | None = None
) -> tuple[dict, Any, SimComponent, FakeSevenDimMPC]:
    """组装并运行一个 3 步短回合（真 SimComponent + 真/假安全）。"""
    env = _reset_env(side)
    mpc = FakeSevenDimMPC(done_after=3)
    comp = _make_component(side, mpc)
    perception = SimPerception(env)
    runner = EpisodeRunner(
        mpc=mpc,
        perception=perception,
        safety=safety,
        executor=comp,
        post_plan_hooks=hooks or [],
    )
    metrics = runner.run(max_steps=10)
    return metrics, env, comp, mpc


# ============================================================================
#  1. EpisodeRunner 短回合契约（感知 / 维度 / 安全 / 时间推进 / 停止）
# ============================================================================


class TestRunnerContract:
    """R1D EpisodeRunner 短回合契约（真实 ProfileSimEnv + 真安全滤波）。"""

    def test_components_satisfy_protocols(self) -> None:
        """R1D 组件实例满足三个管线 Protocol（按协议而非 RM65 类型消费）。"""
        from src.ilqt.components.predictive_safety import PredictiveSafetyFilter

        env = _reset_env("left")
        mpc = FakeSevenDimMPC()
        comp = _make_component("left", mpc)
        perception = SimPerception(env)
        safety = PredictiveSafetyFilter(
            env,
            _limits("left"),
            is_position_mode=False,
            arm_nq=7,
            enable_x_wall=False,
            contact_checker=_checker("left"),
        )
        assert isinstance(comp, ExecutorComponent)
        assert isinstance(perception, PerceptionComponent)
        assert isinstance(safety, SafetyComponent)

    @pytest.mark.parametrize("side", SIDES)
    def test_perception_returns_physical_ball_state(self, side: str) -> None:
        """感知读到的就是物理球状态（每步与 live data 球地址逐项对照）。"""
        env = _envs(side).execution
        seen: list[tuple[np.ndarray, np.ndarray]] = []

        def hook(ctx: Any) -> None:
            # 感知后、执行前采样：SimPerception 输出应等于物理球真值
            seen.append((np.array(ctx.ball_pos), env.get_ball_pos()))

        metrics, _, _, _ = _run_short_episode(
            side, _real_safety(side), hooks=[hook]
        )
        assert metrics["safe_steps"] >= 1
        assert seen, "post_plan hook 未被调用"
        for perc_pos, phys_pos in seen:
            np.testing.assert_allclose(perc_pos, phys_pos, atol=1e-12)

    @pytest.mark.parametrize("side", SIDES)
    def test_control_dimension_is_seven(self, side: str) -> None:
        """规划控制为 (7,)：U_history 每项 (7,)，X_history 每项 (14,)。"""
        metrics, _, comp, _ = _run_short_episode(side, _real_safety(side))
        assert metrics["safe_steps"] == 3
        assert len(comp.U_history) == 3
        assert all(u.shape == (7,) for u in comp.U_history)
        assert all(x.shape == (14,) for x in comp.X_history)

    @pytest.mark.parametrize("side", SIDES)
    def test_execution_advances_exactly_one_dt_per_step(self, side: str) -> None:
        """每次执行时间只推进约定 dt —— R1D 不走旧 X 墙 PD 的额外 step_full。"""
        env = _reset_env(side)
        dt = float(env.dt)
        time_before = float(env.data.time)
        metrics, env_after, _, _ = _run_short_episode(side, _real_safety(side))
        assert metrics["safe_steps"] == 3
        assert float(env_after.data.time) == pytest.approx(time_before + 3 * dt, abs=1e-9)

    @pytest.mark.parametrize("side", SIDES)
    def test_unsafe_stops_before_any_execution(self, side: str) -> None:
        """安全失败 → 零执行、零时间推进、停止分支可观察（safe_steps=0）。"""
        env = _reset_env(side)
        time_before = float(env.data.time)
        metrics, env_after, comp, _ = _run_short_episode(side, UnsafeSafety())
        assert metrics["safe_steps"] == 0
        assert metrics["total_steps"] == 0
        assert comp.X_history == []
        assert comp.U_history == []
        assert comp.total_mpc_steps == 0
        assert float(env_after.data.time) == pytest.approx(time_before, abs=1e-12)

    @pytest.mark.parametrize("side", SIDES)
    def test_real_safety_filter_passes_short_episode(self, side: str) -> None:
        """注入禁止对检查器的真 PredictiveSafetyFilter 放行 init_q 零力矩。"""
        metrics, _, _, _ = _run_short_episode(side, _real_safety(side))
        assert metrics["safe_steps"] == 3
        assert "error" not in metrics


def _real_safety(side: str) -> Any:
    """该侧真安全滤波（R1D：无 X 墙 + 禁止对检查器注入）。"""
    from src.ilqt.components.predictive_safety import PredictiveSafetyFilter

    return PredictiveSafetyFilter(
        _envs(side).execution,
        _limits(side),
        is_position_mode=False,
        arm_nq=7,
        enable_x_wall=False,
        contact_checker=_checker(side),
    )


# ============================================================================
#  2. 物理接触判定（按解析 ID + 真实接触过滤）
# ============================================================================


class TestContactByID:
    """只按 ball/racket geom ID 判定物理接触（dist<=0 过滤 + 负例组）。"""

    @pytest.mark.parametrize("side", SIDES)
    def test_passive_contact_stationary_racket(self, side: str) -> None:
        """静止拍 + 球贴拍 → 被动命中（拍速 ≤0.3），反弹被记录。"""
        env = _reset_env(side)
        mpc = FakeSevenDimMPC(k_hit=0)
        comp = _make_component(side, mpc)
        env.set_ball_state(env.get_ee_pos().copy(), np.zeros(3))
        comp.execute(np.zeros(7))
        assert comp.ball_was_hit
        assert comp.passive_contact
        assert not comp.active_contact
        assert comp.hit_step == 0
        assert comp.v_ee_at_hit is not None and comp.v_ee_at_hit <= 0.3
        assert comp.ball_out_speed is not None
        assert comp.ball_pos_at_hit is not None

    @pytest.mark.parametrize("side", SIDES)
    def test_active_contact_moving_racket(self, side: str) -> None:
        """拍面带速（>0.3 m/s）+ 球贴拍 → 主动命中，与被动分开记录。"""
        env = _reset_env(side)
        mpc = FakeSevenDimMPC(k_hit=0)
        comp = _make_component(side, mpc)
        scene = env.scene
        qdot = np.full(7, 1.5)
        env.set_arm_state(np.concatenate([scene.robot.parameters.init_q, qdot]))
        env.set_ball_state(env.get_ee_pos().copy(), np.zeros(3))
        # 前提：该构型拍面线速度确实 > 0.3 m/s
        assert env.get_racket_face_speed() > 0.3
        comp.execute(np.zeros(7))
        assert comp.ball_was_hit
        assert comp.active_contact
        assert not comp.passive_contact
        assert comp.v_ee_at_hit is not None and comp.v_ee_at_hit > 0.3
        assert comp.ball_out_speed is not None and comp.ball_out_speed > 0.0

    def test_ball_ground_contact_is_not_hit(self) -> None:
        """球—地面物理接触不是命中（ID 判定不认球—地面对）。"""
        env = _reset_env("left")
        mpc = FakeSevenDimMPC(k_hit=0)
        comp = _make_component("left", mpc)
        ball = int(env.scene.ball_geom_id)
        env.set_ball_state(np.array([1.0, 0.0, 0.01]), np.zeros(3))
        comp.execute(np.zeros(7))
        # 前提：本步确有球—地面接触（负例不是"没碰到"）
        assert (min(ball, 0), max(ball, 0)) in _contact_pairs(env)
        assert not comp.ball_was_hit
        assert not comp.active_contact
        assert not comp.passive_contact

    def test_ball_torso_contact_is_not_hit_even_window_closed(self) -> None:
        """球—机身物理接触不是命中；且窗口关闭期间机身碰撞仍可观测。"""
        env = _reset_env("left")
        mpc = FakeSevenDimMPC(k_hit=999)  # 远步 → 窗口关
        comp = _make_component("left", mpc)
        scene = env.scene
        torso_geom = _torso_geom_id(scene)
        ball = int(scene.ball_geom_id)
        env.update_kinematics()
        torso_pos = np.array(env.data.geom_xpos[torso_geom])
        # 球心放到机身 geom 中心（必然重叠接触）
        env.set_ball_state(torso_pos, np.zeros(3))
        comp.execute(np.zeros(7))
        # 前提：球—机身接触存在（掩码未被窗口关闭破坏）
        pair_key = (min(ball, torso_geom), max(ball, torso_geom))
        assert pair_key in _contact_pairs(env)
        assert not comp.ball_was_hit
        assert not comp.active_contact
        assert not comp.passive_contact

    def test_ball_overlaps_arm_link_without_contact_not_hit(self) -> None:
        """球与臂连杆位置重叠（臂—球掩码关，无 contact）不是命中。"""
        env = _reset_env("left")
        mpc = FakeSevenDimMPC(k_hit=0)
        comp = _make_component("left", mpc)
        scene = env.scene
        # 取一个非拍臂 geom（肘部连杆），把球放到其 geom 中心
        arm_geom = _arm_link_geom_id(scene)
        env.update_kinematics()
        env.set_ball_state(
            np.array(env.data.geom_xpos[arm_geom]), np.zeros(3)
        )
        comp.execute(np.zeros(7))
        assert not comp.ball_was_hit
        assert not comp.active_contact
        assert not comp.passive_contact

    def test_window_closed_skips_contact_scan_positive_gap_not_hit(self) -> None:
        """窗口关闭（远步）不扫描接触：正 gap 幽灵点存在也不判命中。

        集成语义回归：k_hit=999 → ``enable_collision=False`` → execute
        整体跳过接触扫描块（``_is_ball_racket_contact`` 不被调用），
        margin=-1 关闭态怪象留下的 dist>0 幽灵接触点不影响裁决。
        dist<=0 过滤本身的单元锚点见
        ``test_is_ball_racket_contact_filters_positive_dist_ghost``
        （本测试删掉过滤器仍绿，不是过滤器的回归锚点）。
        """
        env = _reset_env("left")
        mpc = FakeSevenDimMPC(k_hit=999)  # 窗口关（margin=-1）
        comp = _make_component("left", mpc)
        scene = env.scene
        racket_geom = int(scene.racket_geom_ids[0])
        env.update_kinematics()
        racket_pos = np.array(env.data.geom_xpos[racket_geom])
        # 沿世界 +Z 把球放到距拍 ~0.5 m（实测该位形在 margin=-1 下会留下
        # dist>0 的幽灵接触点 —— 但窗口关 → 不扫描 → 不判命中）
        env.set_ball_state(racket_pos + np.array([0.0, 0.0, 0.5]), np.zeros(3))
        comp.execute(np.zeros(7))
        assert not comp.ball_was_hit
        assert not comp.active_contact
        assert not comp.passive_contact

    def test_is_ball_racket_contact_filters_positive_dist_ghost(self) -> None:
        """``_is_ball_racket_contact`` 单元钉死 dist<=0 过滤（合成幽灵接触）。

        margin=-1 关闭态怪象形态：球—拍 geom ID 匹配但 ``dist > 0``
        （MuJoCo 3.9 斜圆柱显式 pair 实测可产生，gap 0.23→dist 0.139、
        gap 0.5→dist 0.394）。execute() 当前调用图中窗口关闭即不扫描
        （见 test_positive_gap_contact_point_not_hit），故用合成接触
        对象直接钉判定本身 —— 删掉 ``and float(contact.dist) <= 0.0``
        本测试即红，是 dist 过滤的回归锚点。
        """
        from types import SimpleNamespace

        env = _reset_env("left")
        scene = env.scene
        ball = int(scene.ball_geom_id)
        racket = int(scene.racket_geom_ids[0])
        mpc = FakeSevenDimMPC(k_hit=0)
        comp = _make_component("left", mpc)

        def ghost(g1: int, g2: int, dist: float) -> Any:
            """合成 MuJoCo 接触形态对象（geom1/geom2/dist）。"""
            return SimpleNamespace(geom1=g1, geom2=g2, dist=dist)

        # 幽灵接触（ID 匹配 + dist>0，双向 geom 顺序）→ False
        assert not comp._is_ball_racket_contact(ghost(ball, racket, 0.139))
        assert not comp._is_ball_racket_contact(ghost(racket, ball, 0.394))
        # 正面锚：ID 匹配 + dist<=0（真实物理接触，含 dist=0 边界）→ True
        assert comp._is_ball_racket_contact(ghost(ball, racket, -0.038))
        assert comp._is_ball_racket_contact(ghost(racket, ball, 0.0))
        # ID 不匹配（球—地面 / 球—机身，即使穿透）→ False
        assert not comp._is_ball_racket_contact(ghost(ball, 0, -0.01))
        assert not comp._is_ball_racket_contact(ghost(ball, _torso_geom_id(scene), -0.01))

    def test_close_without_contact_positive_error_case_is_miss(self) -> None:
        """距离小但无 contact 不是命中（pos_error≈0 仍判"未命中"）。"""
        env = _reset_env("left")
        # p_hit 设为拍面位置 → pos_error≈0，球距拍 0.15 m（margin=0 无接触）
        env.update_kinematics()
        ee_pos = env.get_ee_pos().copy()
        mpc = FakeSevenDimMPC(k_hit=0, p_hit=ee_pos)
        comp = _make_component("left", mpc)
        env.set_ball_state(ee_pos + np.array([0.0, 0.0, 0.15]), np.zeros(3))
        comp.execute(np.zeros(7))
        assert not comp.ball_was_hit
        assert not comp.active_contact
        assert not comp.passive_contact
        assert comp.pos_error_history[-1] < 0.05
        message = classify_hit_result(
            comp.active_contact, comp.passive_contact,
            comp.pos_error_history[-1], robot_name="R1-A7-D",
        )
        assert "未命中" in message
        assert not determine_hit_from_type("miss")


# ============================================================================
#  3. 碰撞窗口语义（只切球—拍，机身保护常开）
# ============================================================================


class TestCollisionWindow:
    """碰撞窗口只切换球—拍 pair；R1D 不触碰旧 contype/conaffinity 语义。"""

    def test_window_closed_ball_passes_through_racket(self) -> None:
        """窗口关（远步）：球贴拍带速 → 无命中判定（穿拍而过）。"""
        env = _reset_env("left")
        mpc = FakeSevenDimMPC(k_hit=999)
        comp = _make_component("left", mpc)
        env.set_ball_state(env.get_ee_pos().copy(), np.array([0.0, 0.0, -5.0]))
        comp.execute(np.zeros(7))
        assert not comp.ball_was_hit
        assert not comp.active_contact
        assert not comp.passive_contact
        assert comp.ball_out_speed is None  # 未走反弹分支

    def test_geom_masks_unchanged_across_execute(self) -> None:
        """execute 全程 geom contype/conaffinity 不变（不关机身保护）。"""
        env = _reset_env("left")
        mpc = FakeSevenDimMPC(k_hit=0)
        comp = _make_component("left", mpc)
        ct_before = env.model.geom_contype.copy()
        ca_before = env.model.geom_conaffinity.copy()
        comp.execute(np.zeros(7))
        comp.execute(np.zeros(7))
        np.testing.assert_array_equal(env.model.geom_contype, ct_before)
        np.testing.assert_array_equal(env.model.geom_conaffinity, ca_before)


# ============================================================================
#  4. 协议与旧路径兼容
# ============================================================================


class TestProtocolAndLegacy:
    """SimExecutionEnv 协议 + SimComponent 旧参数兼容 + 结果分类文本。"""

    def test_rm65_env_satisfies_sim_execution_env(self) -> None:
        """RM65Env 满足 SimExecutionEnv 协议（旧环境不被协议排除）。"""
        from src.sim.rm65_env import RM65Env

        env = RM65Env(REPO_ROOT / "src" / "robot" / "rm65_model.xml")
        assert isinstance(env, SimExecutionEnv)

    @pytest.mark.parametrize("side", SIDES)
    def test_profile_sim_env_satisfies_sim_execution_env(self, side: str) -> None:
        """ProfileSimEnv（双侧）满足 SimExecutionEnv 协议。"""
        assert isinstance(_envs(side).execution, SimExecutionEnv)

    def test_partial_contact_ids_rejected(self) -> None:
        """接触 ID 只给一半（ball 无 racket / 反之）必须 ValueError。"""
        env = _reset_env("left")
        scene = env.scene
        mpc = FakeSevenDimMPC()
        with pytest.raises(ValueError, match="同时"):
            SimComponent(
                env, mpc, _limits("left"),
                init_q=np.asarray(scene.robot.parameters.init_q, dtype=np.float64),
                is_position_mode=False, hard_x_body_ids=[],
                initial_hitting_tube=None, tube_cfg=TubeConfig(),
                dt=float(env.dt),
                ball_geom_id=int(scene.ball_geom_id), racket_geom_ids=None,
            )
        with pytest.raises(ValueError, match="同时"):
            SimComponent(
                env, mpc, _limits("left"),
                init_q=np.asarray(scene.robot.parameters.init_q, dtype=np.float64),
                is_position_mode=False, hard_x_body_ids=[],
                initial_hitting_tube=None, tube_cfg=TubeConfig(),
                dt=float(env.dt),
                ball_geom_id=None, racket_geom_ids=scene.racket_geom_ids,
            )

    def test_default_ctor_uses_substring_path_and_still_works(self) -> None:
        """缺省构造（无接触 ID）走旧子串路径，R1D geom 名仍可判定。"""
        env = _reset_env("left")
        mpc = FakeSevenDimMPC(k_hit=0)
        comp = _make_component("left", mpc, contact_ids=False)
        assert comp._contact_geom_ids is None  # 旧路径标志
        env.set_ball_state(env.get_ee_pos().copy(), np.zeros(3))
        comp.execute(np.zeros(7))
        # R1D 模型 geom 名 "ball"/"racket_face" 含子串 → 旧路径同样命中
        assert comp.ball_was_hit
        assert comp.passive_contact

    def test_classify_hit_result_default_robot_name(self) -> None:
        """缺省 robot_name 保持旧 "RM-65" 文本（逐字兼容）。"""
        assert "RM-65 击打成功" in classify_hit_result(True, False, 0.01)
        assert "RM-65 被动命中" in classify_hit_result(False, True, 0.02)
        assert "RM-65 未命中" in classify_hit_result(False, False, 0.03)

    def test_classify_hit_result_custom_robot_name(self) -> None:
        """R1D 输出正确机器人名称。"""
        assert "R1-A7-D 击打成功" in classify_hit_result(
            True, False, 0.01, robot_name="R1-A7-D"
        )
        assert "R1-A7-D 被动命中" in classify_hit_result(
            False, True, 0.02, robot_name="R1-A7-D"
        )
        assert "R1-A7-D 未命中" in classify_hit_result(
            False, False, 0.03, robot_name="R1-A7-D"
        )
