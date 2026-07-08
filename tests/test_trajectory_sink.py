"""trajectory_sink.py 单元测试 — Sink 链 + StepState 共享。

测试策略: 手写 Mock（非 MagicMock），验证行为而非实现。
Mock 清单:
    - MockSafetyMonitor: 可配置 is_safe 返回值，记录调用
    - MockTimer: tick_end 返回 0（不 sleep），记录调用次数
    - MockEnv: 固定 jacp/tcp_pos，记录 set_arm_state 调用
    - MockTrajectoryRecorder: 记录 record() 调用参数
"""

import numpy as np

from src.real.fake_robot import FakeRobot
from src.real.trajectory_sink import CommandSink, RecorderSink, RobotSink, TeeSink
from src.real.trajectory_types import StepState


# ── 手写 Mock 类 ──


class MockSafetyMonitor:
    """安全监控 Mock — 可配置 is_safe 返回值，记录调用。"""

    def __init__(self, safe: bool = True) -> None:
        self._safe = safe
        self.is_safe_calls: list[dict] = []

    def is_safe(
        self,
        arm_state: np.ndarray,
        q_desired: np.ndarray,
        tcp_speed: float = 0.0,
    ) -> bool:
        """记录参数并返回配置的安全值。"""
        self.is_safe_calls.append(
            {
                "arm_state": arm_state.copy(),
                "q_desired": q_desired.copy(),
                "tcp_speed": tcp_speed,
            }
        )
        return self._safe


class MockTimer:
    """自适应计时器 Mock — tick_end 返回 0（不 sleep），记录调用。"""

    def __init__(self, sleep_return: float = 0.0) -> None:
        self._sleep_return = sleep_return
        self.tick_start_count: int = 0
        self.tick_end_count: int = 0

    def tick_start(self) -> None:
        self.tick_start_count += 1

    def tick_end(self) -> float:
        self.tick_end_count += 1
        return self._sleep_return


class MockEnv:
    """规划环境 Mock — 固定 jacp/tcp_pos，记录 set_arm_state 调用。"""

    def __init__(self) -> None:
        self.set_arm_state_calls: list[np.ndarray] = []
        self._jacp = np.eye(3, 6)
        self._tcp_pos = np.array([0.5, 0.0, 0.5])

    def set_arm_state(self, x: np.ndarray) -> None:
        self.set_arm_state_calls.append(x.copy())

    def get_ee_jacp(self) -> np.ndarray:
        return self._jacp.copy()

    def get_ee_pos(self) -> np.ndarray:
        return self._tcp_pos.copy()


class MockTrajectoryRecorder:
    """轨迹记录器 Mock — 记录 record() 调用参数。"""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def record(
        self,
        q_desired: np.ndarray,
        q_actual: np.ndarray,
        timestamp: float,
        tcp_pos: np.ndarray,
        ball_pos: np.ndarray | None = None,
    ) -> None:
        """记录一步数据参数。"""
        self.records.append(
            {
                "q_desired": q_desired.copy(),
                "q_actual": q_actual.copy(),
                "timestamp": timestamp,
                "tcp_pos": tcp_pos.copy(),
                "ball_pos": ball_pos.copy() if ball_pos is not None else None,
            }
        )


class StubSink:
    """最小 Sink 桩 — 用于 TeeSink 测试，可配置返回值并记录调用。"""

    def __init__(self, return_value: bool = True) -> None:
        self._return_value = return_value
        self.send_count: int = 0
        self.received_states: list[StepState] = []

    def send(self, state: StepState) -> bool:
        self.send_count += 1
        self.received_states.append(state)
        return self._return_value


# ── 异常注入 Mock 类（Exception path 测试专用）──


class FailingGetArmStateRobot(FakeRobot):
    """FakeRobot 变体 — get_arm_state() 抛出 RuntimeError，模拟通信断开。

    slow_stop 计数继承自 FakeRobot，用于验证异常路径调用了缓停。
    """

    def get_arm_state(self) -> np.ndarray:
        raise RuntimeError("模拟机器人通信断开")


class FailingSendCommandRobot(FakeRobot):
    """FakeRobot 变体 — send_joint_command() 抛出 RuntimeError，模拟下发失败。

    slow_stop 计数继承自 FakeRobot，用于验证异常路径调用了缓停。
    """

    def send_joint_command(self, q_desired: np.ndarray) -> int:
        raise RuntimeError("模拟关节命令下发失败")


class FailingEnv(MockEnv):
    """MockEnv 变体 — set_arm_state() 抛出 RuntimeError，模拟 FK 计算失败。"""

    def set_arm_state(self, x: np.ndarray) -> None:
        raise RuntimeError("模拟 env.set_arm_state 失败")


# ── 工厂函数 ──


def _make_robot_sink(
    robot: FakeRobot | None = None,
    safety: MockSafetyMonitor | None = None,
    timer: MockTimer | None = None,
    env: MockEnv | None = None,
) -> RobotSink:
    """构建 RobotSink，参数缺省时用默认 Mock。"""
    return RobotSink(
        robot=robot or FakeRobot(init_q=np.zeros(6)),
        safety=safety or MockSafetyMonitor(safe=True),
        timer=timer or MockTimer(),
        env=env or MockEnv(),
    )


# ── 测试类 ──


class TestCommandSinkProtocol:
    """验证 RobotSink/RecorderSink/TeeSink 满足 CommandSink 协议。"""

    def test_robot_sink_satisfies_protocol(self) -> None:
        """RobotSink 满足 CommandSink Protocol（runtime_checkable）。"""
        sink = _make_robot_sink()
        assert isinstance(sink, CommandSink)

    def test_recorder_sink_satisfies_protocol(self) -> None:
        """RecorderSink 满足 CommandSink Protocol。"""
        sink = RecorderSink(recorder=MockTrajectoryRecorder())
        assert isinstance(sink, CommandSink)

    def test_tee_sink_satisfies_protocol(self) -> None:
        """TeeSink 满足 CommandSink Protocol。"""
        sink = TeeSink(sinks=[])
        assert isinstance(sink, CommandSink)


class TestRobotSink:
    """RobotSink 行为测试 — 安全检查 + 状态填充 + 节奏控制。"""

    def test_safe_case_sends_command_and_returns_true(self) -> None:
        """安全检查通过时，下发关节命令并返回 True。"""
        robot = FakeRobot(init_q=np.zeros(6))
        sink = _make_robot_sink(robot=robot, safety=MockSafetyMonitor(safe=True))
        q_cmd = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        state = StepState(q_desired=q_cmd, timestamp=0.0)

        result = sink.send(state)

        assert result is True
        assert len(robot.command_history) == 1
        np.testing.assert_array_almost_equal(robot.command_history[0], q_cmd)

    def test_unsafe_case_slow_stops_and_returns_false(self) -> None:
        """安全检查失败时，缓停并返回 False，不下发命令。"""
        robot = FakeRobot(init_q=np.zeros(6))
        sink = _make_robot_sink(robot=robot, safety=MockSafetyMonitor(safe=False))
        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        result = sink.send(state)

        assert result is False
        assert robot.slow_stop_count == 1
        assert len(robot.command_history) == 0

    def test_arm_state_filled_after_send(self) -> None:
        """send 后 state.arm_state 被填充为发送前的机械臂状态。"""
        robot = FakeRobot(init_q=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        expected = robot.get_arm_state().copy()
        sink = _make_robot_sink(robot=robot)
        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        sink.send(state)

        assert state.arm_state is not None
        np.testing.assert_array_equal(state.arm_state, expected)

    def test_tcp_pos_filled_after_send(self) -> None:
        """send 后 state.tcp_pos 被填充为 env.get_ee_pos() 的返回值。"""
        env = MockEnv()
        expected_tcp = env.get_ee_pos().copy()
        sink = _make_robot_sink(env=env)
        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        sink.send(state)

        assert state.tcp_pos is not None
        np.testing.assert_array_equal(state.tcp_pos, expected_tcp)

    def test_timer_tick_start_called_before_command(self) -> None:
        """安全场景下 tick_start 被调用一次（在下发命令之前）。"""
        timer = MockTimer()
        robot = FakeRobot(init_q=np.zeros(6))
        sink = _make_robot_sink(robot=robot, timer=timer)
        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        sink.send(state)

        assert timer.tick_start_count == 1
        assert len(robot.command_history) == 1

    def test_timer_tick_end_called_in_safe_case(self) -> None:
        """安全场景下 tick_end 被调用一次（在下发命令之后）。"""
        timer = MockTimer()
        sink = _make_robot_sink(timer=timer)
        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        sink.send(state)

        assert timer.tick_end_count == 1

    def test_timer_tick_end_not_called_in_unsafe_case(self) -> None:
        """安全失败时 tick_end 不被调用（立即返回）。"""
        timer = MockTimer()
        sink = _make_robot_sink(
            timer=timer, safety=MockSafetyMonitor(safe=False)
        )
        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        sink.send(state)

        assert timer.tick_end_count == 0

    def test_safety_check_called_with_correct_args(self) -> None:
        """safety.is_safe 接收正确的 arm_state、q_desired、tcp_speed。"""
        robot = FakeRobot(init_q=np.array([0.1] * 6))
        safety = MockSafetyMonitor(safe=True)
        env = MockEnv()
        sink = _make_robot_sink(robot=robot, safety=safety, env=env)
        q_cmd = np.array([0.2] * 6)
        state = StepState(q_desired=q_cmd, timestamp=0.0)

        sink.send(state)

        assert len(safety.is_safe_calls) == 1
        call = safety.is_safe_calls[0]
        np.testing.assert_array_equal(call["arm_state"], state.arm_state)
        np.testing.assert_array_equal(call["q_desired"], q_cmd)
        # jacp=eye(3,6), qdot=zeros(6) → tcp_speed=0
        assert call["tcp_speed"] == 0.0

    def test_robot_sink_handles_get_arm_state_exception(self) -> None:
        """robot.get_arm_state() 抛异常时：缓停 + 返回 False + tick_end 不调用。

        场景: 机器人通信断开，get_arm_state() 抛 RuntimeError。
        预期: 尝试 slow_stop()，返回 False，timer.tick_end 不被调用
              （避免 timer 状态被污染）。
        """
        robot = FailingGetArmStateRobot(init_q=np.zeros(6))
        timer = MockTimer()
        sink = _make_robot_sink(robot=robot, timer=timer)
        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        result = sink.send(state)

        assert result is False
        assert robot.slow_stop_count == 1
        assert timer.tick_end_count == 0

    def test_robot_sink_handles_send_joint_command_exception(self) -> None:
        """robot.send_joint_command() 抛异常时：缓停 + 返回 False。

        场景: 安全检查通过，但下发关节命令时通信失败。
        预期: 尝试 slow_stop()，返回 False，tick_end 不被调用。
        """
        robot = FailingSendCommandRobot(init_q=np.zeros(6))
        timer = MockTimer()
        sink = _make_robot_sink(robot=robot, timer=timer)
        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        result = sink.send(state)

        assert result is False
        assert robot.slow_stop_count == 1
        assert timer.tick_end_count == 0


class TestRecorderSink:
    """RecorderSink 行为测试 — 记录数据 + fallback 逻辑。"""

    def test_records_with_present_arm_state_and_tcp_pos(self) -> None:
        """state.arm_state 和 tcp_pos 均存在时，直接记录。"""
        recorder = MockTrajectoryRecorder()
        sink = RecorderSink(recorder=recorder)
        arm_state = np.array([0.1] * 6 + [0.2] * 6)
        tcp_pos = np.array([0.5, 0.0, 0.5])
        q_cmd = np.array([0.3] * 6)
        state = StepState(
            q_desired=q_cmd, timestamp=1.0, arm_state=arm_state, tcp_pos=tcp_pos
        )

        result = sink.send(state)

        assert result is True
        assert len(recorder.records) == 1
        record = recorder.records[0]
        np.testing.assert_array_equal(record["q_desired"], q_cmd)
        np.testing.assert_array_equal(record["q_actual"], arm_state[:6])
        assert record["timestamp"] == 1.0
        np.testing.assert_array_equal(record["tcp_pos"], tcp_pos)

    def test_arm_state_none_with_robot_fallback(self) -> None:
        """arm_state 为 None + robot 提供时，fallback 读取 robot.get_arm_state()。"""
        robot = FakeRobot(init_q=np.array([0.1] * 6))
        recorder = MockTrajectoryRecorder()
        sink = RecorderSink(recorder=recorder, robot=robot)
        state = StepState(
            q_desired=np.zeros(6), timestamp=0.0, tcp_pos=np.zeros(3)
        )

        sink.send(state)

        expected_q = np.array([0.1] * 6)
        np.testing.assert_array_equal(recorder.records[0]["q_actual"], expected_q)

    def test_arm_state_none_no_robot_uses_zeros(self) -> None:
        """arm_state 为 None + robot 未提供时，使用 zeros(12)。"""
        recorder = MockTrajectoryRecorder()
        sink = RecorderSink(recorder=recorder)
        state = StepState(
            q_desired=np.zeros(6), timestamp=0.0, tcp_pos=np.zeros(3)
        )

        sink.send(state)

        np.testing.assert_array_equal(recorder.records[0]["q_actual"], np.zeros(6))

    def test_tcp_pos_none_with_env_fallback(self) -> None:
        """tcp_pos 为 None + env 提供时，fallback 读取 env.get_ee_pos()。"""
        env = MockEnv()
        expected_tcp = env.get_ee_pos().copy()
        recorder = MockTrajectoryRecorder()
        sink = RecorderSink(recorder=recorder, env=env)
        arm_state = np.zeros(12)
        state = StepState(
            q_desired=np.zeros(6), timestamp=0.0, arm_state=arm_state
        )

        sink.send(state)

        np.testing.assert_array_equal(recorder.records[0]["tcp_pos"], expected_tcp)
        assert len(env.set_arm_state_calls) == 1
        np.testing.assert_array_equal(env.set_arm_state_calls[0], arm_state)

    def test_all_none_no_fallbacks_uses_zeros(self) -> None:
        """arm_state 和 tcp_pos 均为 None + 无 robot/env 时，全用 zeros。"""
        recorder = MockTrajectoryRecorder()
        sink = RecorderSink(recorder=recorder)
        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        result = sink.send(state)

        assert result is True
        np.testing.assert_array_equal(recorder.records[0]["q_actual"], np.zeros(6))
        np.testing.assert_array_equal(recorder.records[0]["tcp_pos"], np.zeros(3))

    def test_recorder_sink_env_fallback_exception(self) -> None:
        """tcp_pos=None + env.set_arm_state 抛异常时：降级 zeros(3)，返回 True。

        场景: state.tcp_pos 未填充，fallback 调 env.set_arm_state 时 FK 失败。
        预期: 不抛异常，tcp_pos 降级为 zeros(3)，记录仍完成，返回 True。
        """
        env = FailingEnv()
        recorder = MockTrajectoryRecorder()
        sink = RecorderSink(recorder=recorder, env=env)
        arm_state = np.zeros(12)
        state = StepState(
            q_desired=np.zeros(6), timestamp=0.0, arm_state=arm_state
        )

        result = sink.send(state)

        assert result is True
        assert len(recorder.records) == 1
        np.testing.assert_array_equal(recorder.records[0]["tcp_pos"], np.zeros(3))

    def test_recorder_sink_robot_fallback_exception(self) -> None:
        """arm_state=None + robot.get_arm_state 抛异常时：降级 zeros(12)，返回 True。

        场景: state.arm_state 未填充，fallback 调 robot.get_arm_state 时通信断开。
        预期: 不抛异常，arm_state 降级为 zeros(12)，记录仍完成，返回 True。
        """
        robot = FailingGetArmStateRobot(init_q=np.zeros(6))
        recorder = MockTrajectoryRecorder()
        sink = RecorderSink(recorder=recorder, robot=robot)
        state = StepState(
            q_desired=np.zeros(6), timestamp=0.0, tcp_pos=np.zeros(3)
        )

        result = sink.send(state)

        assert result is True
        assert len(recorder.records) == 1
        np.testing.assert_array_equal(recorder.records[0]["q_actual"], np.zeros(6))


class TestTeeSink:
    """TeeSink 行为测试 — 顺序广播 + 短路失败。"""

    def test_two_sinks_both_succeed(self) -> None:
        """两个 Sink 均成功时，都被调用并返回 True。"""
        sink1 = StubSink(return_value=True)
        sink2 = StubSink(return_value=True)
        tee = TeeSink(sinks=[sink1, sink2])
        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        result = tee.send(state)

        assert result is True
        assert sink1.send_count == 1
        assert sink2.send_count == 1

    def test_first_fails_short_circuits(self) -> None:
        """第一个 Sink 失败时，第二个不被调用（短路），返回 False。"""
        sink1 = StubSink(return_value=False)
        sink2 = StubSink(return_value=True)
        tee = TeeSink(sinks=[sink1, sink2])
        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        result = tee.send(state)

        assert result is False
        assert sink1.send_count == 1
        assert sink2.send_count == 0

    def test_empty_sink_list_returns_true(self) -> None:
        """空 Sink 列表时，返回 True（无操作）。"""
        tee = TeeSink(sinks=[])
        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        result = tee.send(state)

        assert result is True

    def test_single_sink_delegates_correctly(self) -> None:
        """单个 Sink 时，正确委托并传递 state。"""
        inner = StubSink(return_value=True)
        tee = TeeSink(sinks=[inner])
        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        result = tee.send(state)

        assert result is True
        assert inner.send_count == 1
        assert inner.received_states[0] is state

    def test_state_passed_to_all_sinks(self) -> None:
        """同一个 StepState 对象按顺序传递给所有 Sink。"""
        sink1 = StubSink(return_value=True)
        sink2 = StubSink(return_value=True)
        tee = TeeSink(sinks=[sink1, sink2])
        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        tee.send(state)

        assert sink1.received_states[0] is state
        assert sink2.received_states[0] is state


class TestSinkIntegration:
    """Sink 链集成测试 — RobotSink + RecorderSink 状态共享。"""

    def test_robot_sink_fills_state_for_recorder_sink(self) -> None:
        """RobotSink 填充 state.arm_state/tcp_pos，RecorderSink 直接消费。

        验证: RecorderSink 记录的 q_actual 是 RobotSink 读取的发送前状态
              （而非 fallback 重新读取的发送后状态）。
        """
        init_q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        robot = FakeRobot(init_q=init_q)
        env = MockEnv()
        recorder = MockTrajectoryRecorder()

        robot_sink = RobotSink(
            robot=robot,
            safety=MockSafetyMonitor(safe=True),
            timer=MockTimer(),
            env=env,
        )
        recorder_sink = RecorderSink(recorder=recorder)
        tee = TeeSink(sinks=[robot_sink, recorder_sink])

        q_cmd = np.array([0.2] * 6)
        state = StepState(q_desired=q_cmd, timestamp=0.5)

        result = tee.send(state)

        assert result is True
        assert len(recorder.records) == 1
        record = recorder.records[0]
        # q_actual = 发送前 arm_state[:6] = init_q（证明 state 被共享）
        np.testing.assert_array_equal(record["q_actual"], init_q)
        # tcp_pos 由 RobotSink 填充
        np.testing.assert_array_equal(record["tcp_pos"], env.get_ee_pos())
        np.testing.assert_array_equal(record["q_desired"], q_cmd)
        assert record["timestamp"] == 0.5

    def test_recorder_not_called_when_robot_sink_fails(self) -> None:
        """RobotSink 安全检查失败时，RecorderSink 不被调用（短路）。"""
        robot = FakeRobot(init_q=np.zeros(6))
        recorder = MockTrajectoryRecorder()

        robot_sink = RobotSink(
            robot=robot,
            safety=MockSafetyMonitor(safe=False),
            timer=MockTimer(),
            env=MockEnv(),
        )
        recorder_sink = RecorderSink(recorder=recorder)
        tee = TeeSink(sinks=[robot_sink, recorder_sink])

        state = StepState(q_desired=np.zeros(6), timestamp=0.0)

        result = tee.send(state)

        assert result is False
        assert len(recorder.records) == 0


class TestRobotSinkNaNInf:
    """NaN/Inf 安全检查测试。"""

    def test_robot_sink_rejects_nan_q_desired(self) -> None:
        """q_desired 含 NaN → slow_stop + return False，不下发。"""
        robot = FakeRobot(init_q=np.zeros(6))
        robot_sink = RobotSink(
            robot=robot,
            safety=MockSafetyMonitor(safe=True),
            timer=MockTimer(),
            env=MockEnv(),
        )
        state = StepState(
            q_desired=np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]),
            timestamp=0.0,
        )
        result = robot_sink.send(state)
        assert result is False
        assert robot.slow_stop_count > 0
        assert len(robot.command_history) == 0

    def test_robot_sink_rejects_inf_q_desired(self) -> None:
        """q_desired 含 Inf → 同样拒绝。"""
        robot = FakeRobot(init_q=np.zeros(6))
        robot_sink = RobotSink(
            robot=robot,
            safety=MockSafetyMonitor(safe=True),
            timer=MockTimer(),
            env=MockEnv(),
        )
        state = StepState(
            q_desired=np.array([0.0, np.inf, 0.0, 0.0, 0.0, 0.0]),
            timestamp=0.0,
        )
        result = robot_sink.send(state)
        assert result is False
        assert robot.slow_stop_count > 0
        assert len(robot.command_history) == 0
