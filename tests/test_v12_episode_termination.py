"""R01/R07/R08 终止语义与球速度历史回归测试。

覆盖三个真实缺陷：

- R01：``EpisodeRunner.run`` 必须回传 ``stop_reason``/``unsafe``，且
  ``_apply_post_hit_phase`` 只在 ``mpc_done``/``max_steps`` 时执行击打后
  20 步未滤波保持控制；``unsafe``/``no_ball`` 必须 0 步截断。
- R07：``_seed_ball_history`` 首帧必须与 ``set_ball_state`` 写入环境的
  初态同源（含 ``ball_speed_perturb_pct`` 扰动）。
- R08：``_resolve_failure_reason`` 是纯函数且 ``mpc_done`` 不得掩盖安全
  拒绝/不可达。
- R16：实际执行的运行选项与生效 ``init_q`` 必须进入
  ``payload["config"]["runtime_overrides"]``（产物同源），而不只是 metrics。

这些测试全部使用假 execution/executor，不加载 MuJoCo 模型；只有最后一个
R16/R07 端到端用例跑 horizon=60 的短回合（不跑验收长回合）。
"""

from __future__ import annotations

import argparse
import types
from typing import Any

import numpy as np
import pytest

from src.ilqt.episode_runner import EpisodeRunner
from src.sim.episode_artifacts import config_to_dict
from src.sim.v12_config import ResolvedV12Config, resolve_v12_config
from src.sim.v12_episode import (
    _POST_HIT_STEPS,
    _apply_post_hit_phase,
    _resolve_failure_reason,
    _run_post_hit_hold,
    _seed_ball_history,
    run_v12_episode,
)

# ---------------------------------------------------------------------------
# EpisodeRunner：stop_reason / unsafe
# ---------------------------------------------------------------------------


class _PerceptionStub:
    """按 step 返回球状态；``missing_at`` 指定的步返回 None。"""

    def __init__(self, missing_at: int | None = None) -> None:
        self.missing_at = missing_at
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def get_ball_state(self, step: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
        if self.missing_at is not None and step >= self.missing_at:
            return None
        return np.zeros(3), np.array([0.0, -7.0, 0.0])


class _MPCStub:
    """第 ``done_after`` 个 step 后置 done；``None`` 表示永不完成。"""

    def __init__(self, done_after: int | None) -> None:
        self.done_after = done_after
        self.steps = 0
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    def start(self, ball_pos: Any, ball_vel: Any, arm_state: Any) -> None:
        pass

    def step(self, ball_pos: Any, ball_vel: Any, arm_state: Any) -> Any:
        self.steps += 1
        if self.done_after is not None and self.steps >= self.done_after:
            self._done = True
        return types.SimpleNamespace(u_cmd=np.zeros(7))

    def stop(self) -> None:
        pass


class _SafetyStub:
    """前 ``accept_steps`` 步放行，之后拒绝。"""

    def __init__(self, accept_steps: int | None = None) -> None:
        self.accept_steps = accept_steps
        self.calls = 0

    def filter(self, u: Any, arm_state: Any) -> tuple[np.ndarray, bool]:
        accepted = self.accept_steps is None or self.calls < self.accept_steps
        self.calls += 1
        return np.asarray(u, dtype=np.float64), accepted


class _ExecutorStub:
    def __init__(self) -> None:
        self.executed: list[np.ndarray] = []

    def get_arm_state(self) -> np.ndarray:
        return np.zeros(14)

    def execute(self, u: Any) -> None:
        self.executed.append(np.asarray(u, dtype=np.float64).copy())

    def get_metrics(self) -> dict[str, Any]:
        return {}


def _make_runner(
    *,
    mpc: _MPCStub,
    safety: _SafetyStub,
    perception: _PerceptionStub,
) -> tuple[EpisodeRunner, _ExecutorStub]:
    executor = _ExecutorStub()
    return (
        EpisodeRunner(
            mpc=mpc,
            perception=perception,
            safety=safety,
            executor=executor,
        ),
        executor,
    )


def test_initial_no_ball_reports_stop_reason_no_ball() -> None:
    """初始球状态不可用：保留 error="no_ball"，并回传 stop_reason。"""
    runner, executor = _make_runner(
        mpc=_MPCStub(done_after=None),
        safety=_SafetyStub(),
        perception=_PerceptionStub(missing_at=0),
    )
    metrics = runner.run(max_steps=10)
    assert metrics["error"] == "no_ball"
    assert metrics["stop_reason"] == "no_ball"
    assert metrics["unsafe"] is False
    assert metrics["total_steps"] == 0
    assert metrics["safe_steps"] == 0
    assert executor.executed == []


def test_safety_reject_sets_unsafe_stop_reason() -> None:
    """安全滤波拒绝：stop_reason="unsafe"，且不再执行/计数该步。"""
    runner, executor = _make_runner(
        mpc=_MPCStub(done_after=None),
        safety=_SafetyStub(accept_steps=2),
        perception=_PerceptionStub(),
    )
    metrics = runner.run(max_steps=10)
    assert metrics["stop_reason"] == "unsafe"
    assert metrics["unsafe"] is True
    assert metrics["total_steps"] == 2
    assert metrics["safe_steps"] == 2
    assert len(executor.executed) == 2


def test_mpc_done_sets_mpc_done_stop_reason() -> None:
    """MPC 正常完成：stop_reason="mpc_done"、unsafe=False。"""
    runner, executor = _make_runner(
        mpc=_MPCStub(done_after=3),
        safety=_SafetyStub(),
        perception=_PerceptionStub(),
    )
    metrics = runner.run(max_steps=10)
    assert metrics["stop_reason"] == "mpc_done"
    assert metrics["mpc_done"] is True
    assert metrics["unsafe"] is False
    assert metrics["total_steps"] == 3
    assert len(executor.executed) == 3


def test_max_steps_sets_max_steps_stop_reason() -> None:
    """达到步数上限且 MPC 未完成：stop_reason="max_steps"。"""
    runner, executor = _make_runner(
        mpc=_MPCStub(done_after=None),
        safety=_SafetyStub(),
        perception=_PerceptionStub(),
    )
    metrics = runner.run(max_steps=4)
    assert metrics["stop_reason"] == "max_steps"
    assert metrics["mpc_done"] is False
    assert metrics["unsafe"] is False
    assert metrics["total_steps"] == 4
    assert len(executor.executed) == 4


def test_midloop_no_ball_sets_stop_reason_no_ball() -> None:
    """循环内球状态丢失：stop_reason="no_ball"，且不带 error 键。"""
    runner, _ = _make_runner(
        mpc=_MPCStub(done_after=None),
        safety=_SafetyStub(),
        perception=_PerceptionStub(missing_at=2),
    )
    metrics = runner.run(max_steps=10)
    assert metrics["stop_reason"] == "no_ball"
    assert metrics["unsafe"] is False
    assert metrics["total_steps"] == 2
    assert "error" not in metrics


# ---------------------------------------------------------------------------
# 击打后保持段：按 stop_reason 门控（R01）
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, nu: int) -> None:
        self.actuator_ctrlrange = np.tile(
            np.array([[-50.0, 50.0]], dtype=np.float64), (nu, 1)
        )


class _FakeExecution:
    """记录 step_full 调用的极简执行环境（7 维臂 + 自由球）。"""

    NQ = 7
    NU = 7
    NX = 14

    def __init__(self) -> None:
        self.model = _FakeModel(self.NU)
        self._x = np.zeros(self.NX, dtype=np.float64)
        # 关节速度非零，便于校验 u_hold = 100*(q_hold-q) - 10*qdot = -10*qdot
        self._x[self.NQ:] = 0.5
        self._ball_pos = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._ball_vel = np.array([0.0, -7.0, 0.0], dtype=np.float64)
        self.step_calls: list[np.ndarray] = []

    def get_arm_state(self) -> np.ndarray:
        return self._x.copy()

    def step_full(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.step_calls.append(np.asarray(u, dtype=np.float64).copy())
        # 位置每步 +0.01，速度不变：便于校验力矩 PD 公式的逐步输出
        self._x = self._x + np.concatenate(
            [np.full(self.NQ, 0.01), np.zeros(self.NQ)]
        )
        self._ball_pos = self._ball_pos + np.array([0.0, -0.035, 0.0])
        return self._x, self._ball_pos, self._ball_vel

    def get_ball_vel(self) -> np.ndarray:
        return self._ball_vel.copy()

    def get_ee_pos(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)


class _FakeRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, **kwargs: Any) -> None:
        self.records.append(dict(kwargs))


def _make_sim_component(
    *,
    p0: np.ndarray | None = None,
    v0: np.ndarray | None = None,
) -> types.SimpleNamespace:
    """构造与 SimComponent 同构的最小 history 容器（首帧已 seed）。"""
    return types.SimpleNamespace(
        X_history=[np.zeros(14, dtype=np.float64)],
        U_history=[],
        ball_pos_history=[
            np.zeros(3) if p0 is None else np.asarray(p0, dtype=np.float64).copy()
        ],
        ball_vel_history=[
            np.zeros(3) if v0 is None else np.asarray(v0, dtype=np.float64).copy()
        ],
    )


@pytest.mark.parametrize("stop_reason", ["unsafe", "no_ball"])
def test_non_normal_stop_reason_skips_post_hit_entirely(stop_reason: str) -> None:
    """R01：unsafe/no_ball 截断，0 步 step_full、0 条 recorder、U 为空。"""
    execution = _FakeExecution()
    sim = _make_sim_component()
    recorder = _FakeRecorder()

    steps, skipped = _apply_post_hit_phase(
        run_metrics={"stop_reason": stop_reason, "unsafe": stop_reason == "unsafe"},
        execution=execution,
        sim_component=sim,  # type: ignore[arg-type]
        is_position_mode=False,
        dt=0.005,
        total_horizon=300,
        follow_through_steps=60,
        recorder=recorder,
        ball_unreachable=False,
    )

    assert steps == 0
    assert skipped == stop_reason
    assert execution.step_calls == []
    assert recorder.records == []
    assert len(sim.X_history) == 1
    assert len(sim.U_history) == 0
    # 截断后仍保持 U 与 X 的配对关系：len(U) == len(X) - 1
    assert len(sim.U_history) == len(sim.X_history) - 1


@pytest.mark.parametrize("stop_reason", ["mpc_done", "max_steps"])
def test_ball_unreachable_skips_post_hit_even_when_done(stop_reason: str) -> None:
    """R01/R08：不可达即使 mpc_done/max_steps 也截断，且失败原因优先不可达。"""
    execution = _FakeExecution()
    sim = _make_sim_component()
    recorder = _FakeRecorder()

    steps, skipped = _apply_post_hit_phase(
        run_metrics={"stop_reason": stop_reason},
        execution=execution,
        sim_component=sim,  # type: ignore[arg-type]
        is_position_mode=False,
        dt=0.005,
        total_horizon=300,
        follow_through_steps=60,
        recorder=recorder,
        ball_unreachable=True,
    )

    assert steps == 0
    assert skipped == "ball_unreachable"
    assert execution.step_calls == []
    assert recorder.records == []
    assert len(sim.U_history) == len(sim.X_history) - 1
    assert (
        _resolve_failure_reason(
            run_metrics={"stop_reason": stop_reason, "total_steps": 300, "safe_steps": 300},
            ball_unreachable=True,
            stop_reason=stop_reason,
        )
        == "ball_unreachable"
    )


def test_unknown_stop_reason_is_treated_as_skipped() -> None:
    """缺失 stop_reason 时保守截断，原因记为 "unknown"。"""
    execution = _FakeExecution()
    sim = _make_sim_component()
    steps, skipped = _apply_post_hit_phase(
        run_metrics={},
        execution=execution,
        sim_component=sim,  # type: ignore[arg-type]
        is_position_mode=False,
        dt=0.005,
        total_horizon=10,
        follow_through_steps=0,
        recorder=None,
        ball_unreachable=False,
    )
    assert steps == 0
    assert skipped == "unknown"
    assert execution.step_calls == []


@pytest.mark.parametrize("stop_reason", ["mpc_done", "max_steps"])
def test_normal_stop_reason_runs_twenty_steps(stop_reason: str) -> None:
    """正常终止仍执行 20 步保持控制，且四路 history 等长推进。"""
    execution = _FakeExecution()
    sim = _make_sim_component()
    recorder = _FakeRecorder()

    steps, skipped = _apply_post_hit_phase(
        run_metrics={"stop_reason": stop_reason},
        execution=execution,
        sim_component=sim,  # type: ignore[arg-type]
        is_position_mode=False,
        dt=0.005,
        total_horizon=300,
        follow_through_steps=60,
        recorder=recorder,
        ball_unreachable=False,
    )

    assert steps == _POST_HIT_STEPS == 20
    assert skipped is None
    assert len(execution.step_calls) == 20
    assert len(recorder.records) == 20
    assert len(sim.X_history) == 21
    assert len(sim.U_history) == 20
    assert len(sim.ball_pos_history) == 21
    assert len(sim.ball_vel_history) == 21
    assert len(sim.U_history) == len(sim.X_history) - 1
    # recorder 时间戳步号沿用主评估段 + 随挥段之后
    assert recorder.records[0]["timestamp"] == pytest.approx(360 * 0.005)
    assert recorder.records[-1]["timestamp"] == pytest.approx(379 * 0.005)


def test_post_hit_hold_keeps_legacy_torque_pd_outputs() -> None:
    """R01 正常路径逐位一致：u_hold 仍为 100*(q_hold-q)-10*qdot。"""
    execution = _FakeExecution()
    sim = _make_sim_component()

    _run_post_hit_hold(
        execution=execution,
        sim_component=sim,  # type: ignore[arg-type]
        is_position_mode=False,
        dt=0.005,
        total_horizon=0,
        follow_through_steps=0,
        recorder=None,
    )

    # q_hold 取当前 q，位置项恒为零；qdot=0.5 → 每步 u_hold = -10*0.5 = -5
    for u in execution.step_calls:
        np.testing.assert_allclose(u, np.full(7, -5.0))


def test_post_hit_hold_position_mode_holds_joint_angles() -> None:
    """位置模式 u_hold == q_hold（不经过力矩 PD）。"""
    execution = _FakeExecution()
    sim = _make_sim_component()

    _run_post_hit_hold(
        execution=execution,
        sim_component=sim,  # type: ignore[arg-type]
        is_position_mode=True,
        dt=0.005,
        total_horizon=0,
        follow_through_steps=0,
        recorder=None,
    )

    np.testing.assert_allclose(execution.step_calls[0], np.zeros(7))
    np.testing.assert_allclose(execution.step_calls[1], np.full(7, 0.01))


# ---------------------------------------------------------------------------
# R08：失败原因判定矩阵
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("run_metrics", "ball_unreachable", "stop_reason", "expected"),
    [
        # error 最优先，且 verbatim 返回
        (
            {"error": "no_ball", "stop_reason": "no_ball", "safe_steps": 0,
             "total_steps": 0},
            False,
            "no_ball",
            "no_ball",
        ),
        # error 优先于 unsafe / unreachable
        (
            {"error": "sensor_lost", "stop_reason": "unsafe", "safe_steps": 3,
             "total_steps": 10},
            True,
            "unsafe",
            "sensor_lost",
        ),
        # 不可达即使 mpc_done 也报 ball_unreachable
        (
            {"stop_reason": "mpc_done", "mpc_done": True, "safe_steps": 10,
             "total_steps": 10},
            True,
            "mpc_done",
            "ball_unreachable",
        ),
        # mpc_done=True 但最后一步安全拒绝 → safety_filter_stop
        (
            {"stop_reason": "unsafe", "mpc_done": True, "safe_steps": 9,
             "total_steps": 10},
            False,
            "unsafe",
            "safety_filter_stop",
        ),
        # 最后一步安全拒绝（safe_steps == total_steps）也不得被 done 掩盖
        (
            {"stop_reason": "unsafe", "mpc_done": True, "safe_steps": 10,
             "total_steps": 10},
            False,
            "unsafe",
            "safety_filter_stop",
        ),
        # 球状态不可用
        (
            {"stop_reason": "no_ball", "safe_steps": 4, "total_steps": 10},
            False,
            "no_ball",
            "ball_state_unavailable",
        ),
        # 正常完成：两种正常终止均为 None
        (
            {"stop_reason": "mpc_done", "mpc_done": True, "safe_steps": 10,
             "total_steps": 10},
            False,
            "mpc_done",
            None,
        ),
        (
            {"stop_reason": "max_steps", "mpc_done": False, "safe_steps": 10,
             "total_steps": 10},
            False,
            "max_steps",
            None,
        ),
    ],
)
def test_resolve_failure_reason_matrix(
    run_metrics: dict[str, Any],
    ball_unreachable: bool,
    stop_reason: str,
    expected: str | None,
) -> None:
    """R08：按旧 r1d_runner 顺序判定失败原因。"""
    assert _resolve_failure_reason(
        run_metrics=run_metrics,
        ball_unreachable=ball_unreachable,
        stop_reason=stop_reason,
    ) == expected


# ---------------------------------------------------------------------------
# R07：球初态 history 与 set_ball_state 同源
# ---------------------------------------------------------------------------


class _BallStateEnv:
    """只实现 set_ball_state/get_ball_pos/get_ball_vel 的球状态假环境。"""

    def __init__(self) -> None:
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)

    def set_ball_state(self, pos: np.ndarray, vel: np.ndarray) -> None:
        self._pos = np.asarray(pos, dtype=np.float64).copy()
        self._vel = np.asarray(vel, dtype=np.float64).copy()

    def get_ball_pos(self) -> np.ndarray:
        return self._pos.copy()

    def get_ball_vel(self) -> np.ndarray:
        return self._vel.copy()


def test_seed_ball_history_uses_perturbed_initial_state() -> None:
    """R07：扰动 10% 时首帧必须等于写入环境的 p0_real/v0_real。"""
    p0 = np.array([0.0, 0.0, 1.0])
    v0 = np.array([0.0, -7.0, 0.5])
    # 与 run_v12_episode 的扰动公式一致：offset = pct/100 * 2.0
    pct = 10.0
    v0_dir = v0 / np.linalg.norm(v0)
    p0_real = p0 + v0_dir * (pct / 100.0 * 2.0)
    v0_real = v0.copy()

    env = _BallStateEnv()
    env.set_ball_state(p0_real, v0_real)
    sim = _make_sim_component()

    _seed_ball_history(
        sim,  # type: ignore[arg-type]
        ball_pos=p0_real,
        ball_vel=v0_real,
    )

    # 首帧与首个 step 前的 env 状态逐位一致
    np.testing.assert_array_equal(sim.ball_pos_history[0], env.get_ball_pos())
    np.testing.assert_array_equal(sim.ball_vel_history[0], env.get_ball_vel())
    # 且确实是扰动后的初态，而不是扰动前的 p0
    assert not np.array_equal(sim.ball_pos_history[0], p0)
    np.testing.assert_allclose(
        sim.ball_pos_history[0], p0 + v0_dir * 0.2
    )


def test_seed_ball_history_copies_inputs() -> None:
    """首帧为独立拷贝：调用后再改输入数组不污染 history。"""
    sim = _make_sim_component()
    pos = np.array([0.1, 0.2, 0.3])
    vel = np.array([0.0, -7.0, 0.0])
    _seed_ball_history(sim, ball_pos=pos, ball_vel=vel)  # type: ignore[arg-type]
    pos[0] = 99.0
    vel[1] = 99.0
    np.testing.assert_array_equal(sim.ball_pos_history[0], [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(sim.ball_vel_history[0], [0.0, -7.0, 0.0])


# ---------------------------------------------------------------------------
# R16 + R07 端到端：运行选项进 config payload，球首帧为扰动后初态
# ---------------------------------------------------------------------------


def _args(**overrides: object) -> argparse.Namespace:
    """构造只含所需字段的局部 Namespace（其余走 resolver 缺省）。"""
    return argparse.Namespace(**overrides)


def _resolve_rm65_short() -> ResolvedV12Config:
    """解析短回合 RM65 配置（horizon=60/2 次迭代，供一次真实闭环用）。"""
    return resolve_v12_config(
        _args(
            serve_box=True,
            ball_speed=7.0,
            seed=42,
            no_plot=True,
            horizon=60,
            iter=2,
            near_iters=2,
            first_plan_iters=2,
            near_threshold=30,
            replan_interval=20,
        ),
        frozenset({"--serve-box", "--ball-speed", "--seed", "--no-plot"}),
    )


def test_run_v12_episode_records_runtime_options_and_initial_ball() -> None:
    """R16/R07：短回合的覆盖项进 config payload，球首帧为扰动后初态。"""
    config = _resolve_rm65_short()
    result = run_v12_episode(
        config,
        ball_speed_perturb_pct=10.0,
        always_collision=True,
        first_plan_flight=False,
        fixed_direction=False,
        hit_direction=None,
    )

    overrides = result.config.runtime_overrides
    assert overrides["ball_speed_perturb_pct"] == pytest.approx(10.0)
    assert overrides["always_collision"] is True
    assert overrides["first_plan_flight"] is False
    assert overrides["fixed_direction"] is False
    assert overrides["hit_direction"] is None
    assert overrides["limits_config"] is None
    assert len(overrides["init_q"]) == 6
    assert len(overrides["qdot_max"]) == 6
    # metrics 与 config 同源
    assert result.metrics["runtime_overrides"] == overrides
    # 产物 payload（T6 writer 用 config_to_dict）同样携带这些键
    payload = {"config": config_to_dict(result.config)}
    recorded = payload["config"]["runtime_overrides"]
    for key in (
        "ball_speed_perturb_pct",
        "always_collision",
        "first_plan_flight",
        "fixed_direction",
        "hit_direction",
        "init_q",
    ):
        assert key in recorded, f"payload 缺少 runtime_overrides[{key!r}]"
    assert recorded["ball_speed_perturb_pct"] == pytest.approx(10.0)

    # R07：ball_initial 的首帧是扰动后的 p0_real / v0（不是解析前 p0）
    serve = result.metrics["serve"]
    p0 = np.asarray(serve["p0"], dtype=np.float64)
    v0 = np.asarray(serve["v0"], dtype=np.float64)
    offset_m = 10.0 / 100.0 * 2.0
    p0_real = p0 + v0 / np.linalg.norm(v0) * offset_m
    np.testing.assert_allclose(result.ball_pos[0], p0_real)
    np.testing.assert_allclose(result.ball_vel[0], v0)
    # 位置/速度逐帧同长（R07 不静默补齐）
    assert result.ball_pos.shape[0] == result.ball_vel.shape[0]
