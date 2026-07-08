"""AsyncReplanner 泛化测试 — 验证能创建 PlanningEnv 类型的 env_plan。

原实现硬编码 RM65Env，泛化后用 type(self._env)() 动态创建，
使 AsyncReplanner 同时支持仿真（RM65Env）和真机（PlanningEnv）规划环境。
"""

import numpy as np

from src.ilqt.planning_env import PlanningEnv
from src.ilqt.async_replanner import AsyncReplanner, PlanRequest, PlanResult
from src.ilqt.tube_types import ReplanState


def test_async_replanner_creates_planning_env():
    """AsyncReplanner 传入 PlanningEnv 时，env_plan 也是 PlanningEnv 类型。"""
    from pathlib import Path

    model_path = Path("src/robot/rm65_model.xml")
    env = PlanningEnv()
    # 新签名：replan_fn 接受 6 参数 (request, env_plan, state, config, robot_limits, solver)
    replanner = AsyncReplanner(
        env=env,
        replan_fn=lambda req, env_p, state, config, robot_limits, solver: PlanResult(solver_ok=False),
        config=None,  # type: ignore[arg-type]  # lambda 不消费 config
        robot_limits=None,  # type: ignore[arg-type]
        solver=None,  # type: ignore[arg-type]
        state=ReplanState(),
        model_path=model_path,
    )

    env_plan = replanner._ensure_env_plan()

    assert isinstance(env_plan, PlanningEnv), (
        f"env_plan 应为 PlanningEnv 类型，实际为 {type(env_plan)}"
    )
    assert env_plan is not env, "env_plan 应为独立实例"
    assert env_plan.dt == env.dt, "dt 应一致"

    replanner.stop()
