"""T4：R1D（七关节绑定环境）异步重规划测试。

覆盖任务书 §5 的 8 类异步验收点：

1. 真实后台规划：``submit_count``/``complete_count``/``plan_duration_ms`` 有值，
   执行环境在规划期不被修改，``U_buffer`` 列数按 NU=7。
2. 后台规划环境与执行环境/传入规划环境是不同对象，model/data 不共享。
3. ``submit`` 对请求做深拷贝：提交后主线程改写原数组不影响后台。
4. 后台 replan_fn 抛异常 → ``PlanResult(solver_ok=False)``，主线程不崩。
5. 失效结果不应用（过期/超界/长度不足），``_U_buffer`` 保持不变。
6. 同步 vs 异步单次规划等价（k_hit/p_hit/n_des/U 容差内一致，记录容差来源）。
7. ``stop`` 幂等、重复回合无残留后台线程。
8. ``PlanResult`` 空控制序列按 NU 形状；R1D + 异步无工厂明确拒绝。

环境只使用真实 R1D 固定场景（右臂、torque 模式）；MPC 迭代数压到最小以保持
测试时长，但每条用例仍走真实 ``do_replan``（非 mock solver）。
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from src.ilqt.async_replanner import AsyncReplanner, PlanRequest, PlanResult
from src.ilqt.mpc_controller import MPCConfig, MPCController
from src.ilqt.solver import build_solver
from src.ilqt.tube_types import ReplanState, TubeConfig
from src.sim.v12_config import ResolvedV12Config, resolve_v12_config
from src.sim.v12_robot_adapter import V12RobotBundle, build_v12_robot

_REPO_ROOT = Path(__file__).resolve().parents[1]
_R1D_PROFILE = _REPO_ROOT / "configs" / "robots" / "r1d_right.yaml"

_BALL_POS = np.array([0.3, -0.6, 2.2], dtype=np.float64)
_BALL_VEL = np.array([-11.5, -0.3, 4.25], dtype=np.float64)

# 测试用小迭代预算：仍走真实 do_replan/真实 solver，只是缩短单次规划时长。
_MPC_OVERRIDES: dict[str, object] = {
    "version": "t4_async_test",
    "total_horizon": 250,
    "fixed_horizon": 20,
    "first_plan_iters": 3,
    "max_iter_per_plan": 1,
    "near_plan_iters": 1,
    "replan_interval": 5,
    "shoulder_pos": np.array([-11.75, -0.9, 1.25], dtype=np.float64),
    "workspace_radius": 0.80,
}

# T3 resolved cost_config 键 → MPCConfig 终端基准键（3.1 YAML 命名差异）。
_COST_KEY_TO_BASE: dict[str, str] = {
    "Q_qdot": "Q_qdot_base",
    "Q_qddot": "Q_qddot_base",
    "Q_du": "Q_du_base",
}


def _r1d_config(*, async_replan: bool) -> ResolvedV12Config:
    """用 T3 解析器构造 R1D 右臂 overlay 配置。"""
    args = argparse.Namespace(robot_config=str(_R1D_PROFILE))
    flags = {"--robot-config"}
    if async_replan:
        args.async_replan = True
        flags.add("--async-replan")
    return resolve_v12_config(args, frozenset(flags))


def _translate_cost_config(cost_config: dict[str, object]) -> dict[str, object]:
    """把 T3 的 cost_config 键翻译成 do_replan 可识别的终端基准键。"""
    translated: dict[str, object] = {}
    for key, value in cost_config.items():
        if key == "stage_weights":
            continue
        translated[_COST_KEY_TO_BASE.get(key, key)] = value
    return translated


def _build_mpc_config(
    resolved: ResolvedV12Config,
    *,
    async_mode: bool,
) -> MPCConfig:
    """构造缩小迭代预算的 R1D MPCConfig（真实字段，测试专用数值）。"""
    kwargs = dict(resolved.mpc_config)
    kwargs.pop("async_mode", None)
    solver_config = dict(resolved.solver_config)
    solver_config.pop("backend", None)
    kwargs.update(_MPC_OVERRIDES)
    kwargs.update(
        cost_config=_translate_cost_config(dict(resolved.cost_config)),
        tube_cfg=TubeConfig(**dict(resolved.cost_config["tube"])),
        solver_backend="python",
        solver_config=solver_config,
        async_mode=async_mode,
    )
    return MPCConfig(**kwargs)


def _make_controller(
    bundle: V12RobotBundle,
    resolved: ResolvedV12Config,
    *,
    async_mode: bool,
) -> MPCController:
    """基于 bundle 事实构造 MPCController（唯一机器人来源）。"""
    return MPCController(
        bundle.planning,
        _build_mpc_config(resolved, async_mode=async_mode),
        robot_limits=bundle.robot_limits,
        planning_env_factory=bundle.planning_env_factory,
    )


def _snapshot(env: object) -> dict[str, np.ndarray | float]:
    """快照执行环境的可观测状态（用于断言后台规划不污染执行侧）。"""
    data = env.data  # type: ignore[attr-defined]
    return {
        "qpos": np.array(data.qpos, copy=True),
        "qvel": np.array(data.qvel, copy=True),
        "ctrl": np.array(data.ctrl, copy=True),
        "time": float(data.time),
    }


def _assert_same_snapshot(
    before: dict[str, np.ndarray | float],
    after: dict[str, np.ndarray | float],
) -> None:
    """断言两个执行环境快照逐位一致。"""
    assert np.array_equal(before["qpos"], after["qpos"])
    assert np.array_equal(before["qvel"], after["qvel"])
    assert np.array_equal(before["ctrl"], after["ctrl"])
    assert before["time"] == pytest.approx(after["time"])


def _wait_for(predicate, timeout_s: float = 15.0) -> bool:
    """轮询等待谓词为真（不 sleep 阻塞主线程过久）。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _make_request(step: int = 0) -> PlanRequest:
    """构造最小合法 PlanRequest（七关节 NU=7 空 U_prev）。"""
    return PlanRequest(
        x_current=np.zeros(14, dtype=np.float64),
        ball_pos=_BALL_POS.copy(),
        ball_vel=_BALL_VEL.copy(),
        step=step,
        k_hit_current=250,
        U_prev=np.zeros((0, 7), dtype=np.float64),
        p_hit_current=np.zeros(3, dtype=np.float64),
        v_hit_desired=np.zeros(3, dtype=np.float64),
        n_des_current=np.zeros(3, dtype=np.float64),
        d_hat=np.zeros(3, dtype=np.float64),
    )


class _BlockingReplan:
    """记录后台收到的请求 + 阻塞到释放的 replan_fn 替身（仅测深拷贝语义）。"""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.seen: PlanRequest | None = None

    def __call__(  # noqa: D102 - 测试替身
        self,
        request: PlanRequest,
        env_plan: object,
        state: ReplanState,
        config: MPCConfig,
        robot_limits: object,
        solver: object,
    ) -> PlanResult:
        self.seen = request
        self.started.set()
        self.release.wait(timeout=10.0)
        return PlanResult(
            nu=7,
            k_hit_new=10,
            U_mpc_full=np.zeros((10, 7), dtype=np.float64),
            U_buffer=np.zeros((5, 7), dtype=np.float64),
            U_prev=np.zeros((0, 7), dtype=np.float64),
            request_step=request.step,
        )


@pytest.fixture(scope="function")
def r1d() -> tuple[V12RobotBundle, ResolvedV12Config]:
    """构造真实 R1D 右臂 bundle 与 async overlay 配置。"""
    resolved = _r1d_config(async_replan=True)
    bundle = build_v12_robot(resolved)
    return bundle, resolved


def test_background_plan_reports_progress_and_leaves_execution_untouched(
    r1d: tuple[V12RobotBundle, ResolvedV12Config],
) -> None:
    """真实后台规划：submit/complete/耗时/duration 有值，执行环境不被修改。"""
    bundle, resolved = r1d
    controller = _make_controller(bundle, resolved, async_mode=True)
    replanner = controller.replanner
    assert replanner is not None

    arm_state = bundle.reset()
    bundle.execution.set_ball_state(_BALL_POS.copy(), _BALL_VEL.copy())
    before = _snapshot(bundle.execution)

    try:
        controller.start(_BALL_POS, _BALL_VEL, arm_state)
        assert replanner.env_plan is not None
        assert replanner.env_plan is not bundle.execution
        assert replanner.env_plan.data is not bundle.execution.data

        for _ in range(2000):
            controller.step(_BALL_POS, _BALL_VEL, arm_state)
            if replanner.submit_count > 0 and replanner.complete_count > 0:
                break
            time.sleep(0.002)

        assert replanner.submit_count > 0
        assert replanner.complete_count > 0
        assert len(replanner.replan_durations_ms) == replanner.complete_count
        assert all(duration > 0.0 for duration in replanner.replan_durations_ms)

        # U_buffer 必须是七列（NU=7），而不是 RM65 的六列
        buffer = controller._U_buffer  # noqa: SLF001 - T4 契约字段
        assert buffer.ndim == 2
        assert buffer.shape[1] == 7
        assert replanner.is_planning() is False

        # 主线程只读执行环境；后台环境独立 ⇒ 快照逐位不变
        _assert_same_snapshot(before, _snapshot(bundle.execution))
    finally:
        controller.stop()
        controller.stop()

    assert replanner._thread is None  # noqa: SLF001 - 生命周期断言
    assert replanner.is_planning() is False
    assert replanner.env_plan is not bundle.planning


def test_background_env_is_independent(r1d: tuple[V12RobotBundle, ResolvedV12Config]) -> None:
    """后台 env_plan 与执行环境、传入规划环境均为不同对象且不共享 data/model。"""
    bundle, resolved = r1d
    controller = _make_controller(bundle, resolved, async_mode=True)
    replanner = controller.replanner
    assert replanner is not None

    arm_state = bundle.reset()
    try:
        controller.start(_BALL_POS, _BALL_VEL, arm_state)
        env_plan = replanner.env_plan
        assert env_plan is not None
        assert env_plan is not bundle.execution
        assert env_plan is not bundle.planning
        assert env_plan.data is not bundle.execution.data
        assert env_plan.data is not bundle.planning.data
        assert env_plan.model is not bundle.execution.model
        assert env_plan.model is not bundle.planning.model
    finally:
        controller.stop()

    # 工厂每次调用都返回全新实例（T4 §4.4）
    first = bundle.planning_env_factory()
    second = bundle.planning_env_factory()
    assert first is not second
    assert first.data is not second.data
    assert first.model is not second.model


def test_submit_deep_copies_request(r1d: tuple[V12RobotBundle, ResolvedV12Config]) -> None:
    """submit 后主线程改写请求数组不影响后台看到的提交时刻快照。"""
    bundle, resolved = r1d
    blocking = _BlockingReplan()
    replanner = AsyncReplanner(
        bundle.execution,
        blocking,
        _build_mpc_config(resolved, async_mode=False),
        bundle.robot_limits,
        build_solver(env=bundle.planning, backend="python", config=None),
        planning_env_factory=bundle.planning_env_factory,
    )

    request = _make_request()
    ball_pos_before = request.ball_pos.copy()
    replanner.start()
    try:
        assert replanner.submit(request) is True
        assert blocking.started.wait(15.0) is True

        # 主线程在后台持有请求期间改写原数组/替换 U_prev
        request.ball_pos[:] = -999.0
        request.U_prev = np.full((3, 7), 7.0)

        assert blocking.seen is not None
        assert np.allclose(blocking.seen.ball_pos, ball_pos_before)
        assert blocking.seen.U_prev.shape == (0, 7)
        assert blocking.release.wait(timeout=0.01) is False  # 仍被测试阻塞

        blocking.release.set()
        assert _wait_for(replanner.has_new_plan) is True
        result = replanner.apply_new_plan()
        assert result is not None
        assert result.solver_ok is True
    finally:
        blocking.release.set()
        replanner.stop()
        replanner.stop()
    assert replanner._thread is None  # noqa: SLF001 - 生命周期断言


def test_background_exception_becomes_solver_not_ok(
    r1d: tuple[V12RobotBundle, ResolvedV12Config],
) -> None:
    """后台 replan_fn 抛异常 → PlanResult(solver_ok=False)，主线程不崩。"""

    def _boom(*args: object, **kwargs: object) -> PlanResult:
        raise RuntimeError("T4 测试注入的后台规划异常")

    bundle, resolved = r1d
    replanner = AsyncReplanner(
        bundle.execution,
        _boom,
        _build_mpc_config(resolved, async_mode=False),
        bundle.robot_limits,
        build_solver(env=bundle.planning, backend="python", config=None),
        planning_env_factory=bundle.planning_env_factory,
    )
    replanner.start()
    try:
        assert replanner.submit(_make_request()) is True
        assert _wait_for(replanner.has_new_plan) is True
        result = replanner.apply_new_plan()
        assert result is not None
        assert result.solver_ok is False
        assert result.ball_unreachable is False
        assert result.U_buffer.shape == (0, 7)
        assert result.U_prev.shape == (0, 7)
        assert result.U_mpc_full.shape == (0, 7)
        assert result.plan_duration_ms > 0.0
        assert replanner.complete_count == 1

        # 主线程存活：还能继续提交（异常不杀死线程）
        assert replanner.submit(_make_request(step=5)) is True
        assert _wait_for(replanner.has_new_plan) is True
        second = replanner.apply_new_plan()
        assert second is not None
        assert second.solver_ok is False
    finally:
        replanner.stop()
        replanner.stop()
    assert replanner._thread is None  # noqa: SLF001 - 生命周期断言


def test_invalid_result_is_not_applied(r1d: tuple[V12RobotBundle, ResolvedV12Config]) -> None:
    """过期/超界/长度不足的异步结果被拒绝，_U_buffer 保持不变。"""
    bundle, resolved = r1d
    controller = _make_controller(bundle, resolved, async_mode=True)
    buffer_before = np.full((10, 7), 0.5, dtype=np.float64)
    controller._U_buffer = buffer_before.copy()  # noqa: SLF001 - 构造失效场景
    controller._k_hit = 42  # noqa: SLF001
    # _apply_async_result 会断言规划态存在；直接调用私有方法需手工赋值
    controller._replan_state = ReplanState()  # noqa: SLF001
    try:
        # elapsed=0 < k_hit_new=3，但剩余 U_mpc_full 只有 3 行 < replan_interval=5
        too_short = PlanResult(
            nu=7,
            request_step=5,
            k_hit_new=3,
            U_mpc_full=np.ones((3, 7), dtype=np.float64),
        )
        assert controller._apply_async_result(too_short, step=5) is False  # noqa: SLF001

        # elapsed=20 >= k_hit_new=10 ⇒ 已过期
        expired = PlanResult(
            nu=7,
            request_step=0,
            k_hit_new=10,
            U_mpc_full=np.ones((10, 7), dtype=np.float64),
        )
        assert controller._apply_async_result(expired, step=20) is False  # noqa: SLF001

        # elapsed=30 >= len(U_mpc_full)=30 ⇒ 越界
        overrun = PlanResult(
            nu=7,
            request_step=0,
            k_hit_new=40,
            U_mpc_full=np.ones((30, 7), dtype=np.float64),
        )
        assert controller._apply_async_result(overrun, step=30) is False  # noqa: SLF001

        assert np.array_equal(controller._U_buffer, buffer_before)  # noqa: SLF001
        assert controller._k_hit == 42  # noqa: SLF001
    finally:
        controller.stop()


def test_sync_and_async_single_plan_are_equivalent(
    r1d: tuple[V12RobotBundle, ResolvedV12Config],
) -> None:
    """同步 vs 异步单次规划结果等价（容差来源见模块/报告说明）。"""
    bundle, resolved = r1d
    sync_controller = _make_controller(bundle, resolved, async_mode=False)
    async_controller = _make_controller(bundle, resolved, async_mode=True)
    assert sync_controller.replanner is not None
    assert async_controller.replanner is not None

    arm_state = bundle.reset()
    try:
        sync_controller.start(_BALL_POS, _BALL_VEL, arm_state)
        async_controller.start(_BALL_POS, _BALL_VEL, arm_state)

        # 同步路径不启动后台线程；异步路径第一版规划后启动后台线程
        assert sync_controller.replanner._thread is None  # noqa: SLF001
        assert async_controller.replanner._thread is not None  # noqa: SLF001

        assert sync_controller._k_hit == async_controller._k_hit  # noqa: SLF001
        # 同 do_replan、同 env_plan 参数 ⇒ 命中几何逐位一致（1e-8 给浮点求和）
        assert np.allclose(
            sync_controller._p_hit, async_controller._p_hit, atol=1e-8, rtol=0.0  # noqa: SLF001
        )
        assert np.allclose(
            sync_controller._v_ball_hit,  # noqa: SLF001
            async_controller._v_ball_hit,  # noqa: SLF001
            atol=1e-8,
            rtol=0.0,
        )
        assert np.allclose(
            sync_controller._n_des, async_controller._n_des, atol=1e-8, rtol=0.0  # noqa: SLF001
        )

        # 控制序列：FD 线性化 + BLAS 归约顺序可能与线程/调度相关，容差 1e-6
        sync_u = sync_controller._U_buffer  # noqa: SLF001
        async_u = async_controller._U_buffer  # noqa: SLF001
        assert sync_u.ndim == 2 and async_u.ndim == 2
        assert sync_u.shape[1] == 7 and async_u.shape[1] == 7
        assert sync_u.shape == async_u.shape
        assert np.allclose(sync_u, async_u, atol=1e-6, rtol=0.0)
    finally:
        sync_controller.stop()
        async_controller.stop()


def test_stop_is_idempotent_and_leaves_no_background_thread(
    r1d: tuple[V12RobotBundle, ResolvedV12Config],
) -> None:
    """stop 幂等；重复回合后没有残留后台线程（daemon 线程已 join）。"""
    bundle, resolved = r1d
    controller = _make_controller(bundle, resolved, async_mode=True)
    replanner = controller.replanner
    assert replanner is not None
    baseline = {thread.ident for thread in threading.enumerate() if thread.is_alive()}

    arm_state = bundle.reset()
    worker_threads: list[threading.Thread] = []
    try:
        for _round in range(2):
            controller.start(_BALL_POS, _BALL_VEL, arm_state)
            worker = replanner._thread  # noqa: SLF001
            assert worker is not None
            worker_threads.append(worker)
            for _ in range(5):
                controller.step(_BALL_POS, _BALL_VEL, arm_state)
                time.sleep(0.002)
            controller.stop()
            controller.stop()  # 幂等：重复调用不抛错
            assert replanner._thread is None  # noqa: SLF001
            assert replanner.is_planning() is False
    finally:
        controller.stop()
        controller.stop()

    assert all(not worker.is_alive() for worker in worker_threads)
    leftover = [
        thread
        for thread in threading.enumerate()
        if thread.is_alive() and thread.ident not in baseline
    ]
    assert leftover == []


def test_plan_result_empty_sequences_follow_nu(
    r1d: tuple[V12RobotBundle, ResolvedV12Config],
) -> None:
    """PlanResult 空控制序列按 NU 归一（RM65 默认 6，R1D 7）；非法形状/维度报错。"""
    default_result = PlanResult()
    assert default_result.U_buffer.shape == (0, 6)
    assert default_result.U_prev.shape == (0, 6)
    assert default_result.U_mpc_full.shape == (0, 6)

    r1d_result = PlanResult(nu=7)
    assert r1d_result.U_buffer.shape == (0, 7)
    assert r1d_result.U_prev.shape == (0, 7)
    assert r1d_result.U_mpc_full.shape == (0, 7)

    with pytest.raises(ValueError, match="列数"):
        PlanResult(nu=7, U_buffer=np.zeros((3, 6), dtype=np.float64))
    with pytest.raises(ValueError, match="正"):
        PlanResult(nu=0)

    # R1D 异步 replanner 的后台事实 NU=7，异常结果也按 7 归一（见上一用例）
    bundle, resolved = r1d
    replanner = AsyncReplanner(
        bundle.execution,
        _BlockingReplan(),
        _build_mpc_config(resolved, async_mode=False),
        bundle.robot_limits,
        build_solver(env=bundle.planning, backend="python", config=None),
        planning_env_factory=bundle.planning_env_factory,
    )
    assert int(replanner._NU) == 7  # noqa: SLF001


def test_r1d_async_requires_planning_env_factory(
    r1d: tuple[V12RobotBundle, ResolvedV12Config],
) -> None:
    """R1D 绑定环境 + 异步未提供工厂时明确拒绝（不静默改同步/不降级）。"""
    bundle, resolved = r1d
    with pytest.raises(ValueError, match="planning_env_factory"):
        AsyncReplanner(
            bundle.execution,
            _BlockingReplan(),
            _build_mpc_config(resolved, async_mode=False),
            bundle.robot_limits,
            build_solver(env=bundle.planning, backend="python", config=None),
        )
    with pytest.raises(ValueError, match="planning_env_factory"):
        MPCController(
            bundle.planning,
            _build_mpc_config(resolved, async_mode=True),
            robot_limits=bundle.robot_limits,
        )
