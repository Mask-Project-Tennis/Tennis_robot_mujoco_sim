"""EpisodeRunner 管线编排器单元测试 — 全 mock，不依赖 MuJoCo。"""

import numpy as np
from src.ilqt.episode_runner import EpisodeRunner
from src.ilqt.mpc_controller import MPCStepResult


class FakeMPC:
    """Mock MPC — step 返回固定结果，第 N 步 done。"""

    def __init__(self) -> None:
        self._step_count = 0
        self._done = False

    def start(self, ball_pos, ball_vel, arm_state) -> None:
        pass

    def step(self, ball_pos, ball_vel, arm_state) -> MPCStepResult:
        self._step_count += 1
        if self._step_count >= 5:
            self._done = True
        return MPCStepResult(
            u_cmd=np.zeros(6), phase="far", k_hit=10,
        )

    def stop(self) -> None:
        pass

    @property
    def done(self) -> bool:
        return self._done


class FakePerception:
    """Mock 感知 — 总返回 (zeros(3), zeros(3))。"""

    def get_ball_state(self):
        return np.zeros(3), np.zeros(3)


class FakeExecutor:
    """Mock 执行器 — 记录所有 execute 调用。"""

    def __init__(self) -> None:
        self.executed: list[np.ndarray] = []

    def get_arm_state(self) -> np.ndarray:
        return np.zeros(12)

    def execute(self, u_cmd: np.ndarray) -> None:
        self.executed.append(u_cmd.copy())


class FakeSafety:
    """Mock 安全 — 总是放行。"""

    def filter(self, u_cmd, arm_state):
        return u_cmd, True


class UnsafeSafety:
    """Mock 安全 — 总是拒绝（用于测试安全停止）。"""

    def filter(self, u_cmd, arm_state):
        return u_cmd, False


def test_runner_basic():
    """tracer: all-mock, run(10) 返回 dict, 5 步后 done。"""
    mpc = FakeMPC()
    executor = FakeExecutor()
    runner = EpisodeRunner(
        mpc=mpc,
        perception=FakePerception(),
        safety=FakeSafety(),
        executor=executor,
    )
    metrics = runner.run(max_steps=10)
    assert metrics["total_steps"] == 5  # FakeMPC 在第 5 步 done
    assert metrics["safe_steps"] == 5
    assert len(executor.executed) == 5


def test_runner_safety_stop():
    """safety 返回 is_safe=False → 第一步就停止，0 safe_steps。"""
    runner = EpisodeRunner(
        mpc=FakeMPC(),
        perception=FakePerception(),
        safety=UnsafeSafety(),
        executor=FakeExecutor(),
    )
    metrics = runner.run(max_steps=10)
    assert metrics["safe_steps"] == 0
    assert metrics["total_steps"] == 0
