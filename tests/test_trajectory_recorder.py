"""TrajectoryRecorder + 数据结构单元测试。

测试轨迹记录、保存、加载（含旧 pickle 格式兼容）和 post_exec_hook 生成。
"""

import pickle

import numpy as np
import pytest

from src.real.trajectory_recorder import TrajectoryRecorder
from src.real.trajectory_types import ReplayTrajectory, StepState


# ── 手写 Mock 类（不使用 MagicMock）──


class MockEnv:
    """模拟 RobotEnv Protocol，返回固定 arm_state 和 ee_pos。"""

    def __init__(self, arm_state: np.ndarray, ee_pos: np.ndarray) -> None:
        self._arm_state = np.asarray(arm_state, dtype=float)
        self._ee_pos = np.asarray(ee_pos, dtype=float)

    def get_arm_state(self) -> np.ndarray:
        """返回右臂状态 (12,)。"""
        return self._arm_state.copy()

    def get_ee_pos(self) -> np.ndarray:
        """返回末端位置 (3,)。"""
        return self._ee_pos.copy()


class MockStepContext:
    """模拟 StepContext，仅含 hook 所需字段。

    step_count 默认 None（模拟无该字段的旧式调用），触发墙钟 fallback。
    """

    def __init__(
        self,
        u_cmd: np.ndarray,
        ball_pos: np.ndarray | None,
        step_count: int | None = None,
    ) -> None:
        self.u_cmd = u_cmd
        self.ball_pos = ball_pos
        self.step_count = step_count


# ── 数据结构测试 ──


class TestReplayTrajectory:
    """ReplayTrajectory 数据结构测试。"""

    def test_creation_and_field_access(self):
        """创建 ReplayTrajectory 后各字段可正确访问，形状一致。"""
        traj = ReplayTrajectory(
            q_desired=np.zeros((10, 6)),
            q_actual=np.zeros((10, 6)),
            timestamps=np.arange(10) * 0.005,
            tcp_pos=np.zeros((10, 3)),
            ball_pos=np.zeros((10, 3)),
            init_q=np.zeros(6),
            init_q_left=np.zeros(6),
            dt=0.005,
            hit_step=5,
            metadata={"hit_type": "active"},
        )
        assert traj.q_desired.shape == (10, 6)
        assert traj.q_actual.shape == (10, 6)
        assert traj.timestamps.shape == (10,)
        assert traj.tcp_pos.shape == (10, 3)
        assert traj.ball_pos.shape == (10, 3)
        assert traj.init_q.shape == (6,)
        assert traj.init_q_left.shape == (6,)
        assert traj.dt == 0.005
        assert traj.hit_step == 5
        assert traj.metadata["hit_type"] == "active"


class TestStepState:
    """StepState 数据结构测试。"""

    def test_creation_with_defaults(self):
        """StepState 可选字段默认为 None。"""
        state = StepState(
            q_desired=np.zeros(6),
            timestamp=0.1,
        )
        np.testing.assert_array_equal(state.q_desired, np.zeros(6))
        assert state.timestamp == 0.1
        assert state.arm_state is None
        assert state.tcp_pos is None

    def test_creation_with_all_fields(self):
        """StepState 全字段赋值。"""
        state = StepState(
            q_desired=np.ones(6),
            timestamp=0.2,
            arm_state=np.zeros(12),
            tcp_pos=np.array([0.5, 0.0, 0.3]),
        )
        np.testing.assert_array_equal(state.q_desired, np.ones(6))
        np.testing.assert_array_equal(state.arm_state, np.zeros(12))
        np.testing.assert_array_equal(state.tcp_pos, np.array([0.5, 0.0, 0.3]))


# ── 记录器 record() 测试 ──


class TestTrajectoryRecorderRecord:
    """record() 方法测试。"""

    def test_record_single_step(self):
        """record() 单步记录，to_trajectory() 返回长度 1。"""
        env = MockEnv(
            arm_state=np.concatenate([np.full(6, 0.1), np.zeros(6)]),
            ee_pos=np.array([0.5, 0.0, 0.3]),
        )
        recorder = TrajectoryRecorder(
            env=env,
            init_q=np.zeros(6),
            init_q_left=np.zeros(6),
            dt=0.005,
        )
        recorder.record(
            q_desired=np.full(6, 0.2),
            q_actual=np.full(6, 0.1),
            timestamp=0.005,
            tcp_pos=np.array([0.5, 0.0, 0.3]),
            ball_pos=np.array([1.0, 0.0, 1.0]),
        )
        traj = recorder.to_trajectory()
        assert traj.q_desired.shape == (1, 6)
        np.testing.assert_allclose(traj.q_desired[0], np.full(6, 0.2))
        np.testing.assert_allclose(traj.q_actual[0], np.full(6, 0.1))
        assert traj.timestamps[0] == 0.005
        np.testing.assert_allclose(traj.tcp_pos[0], np.array([0.5, 0.0, 0.3]))
        np.testing.assert_allclose(traj.ball_pos[0], np.array([1.0, 0.0, 1.0]))

    def test_record_multiple_steps_accumulate(self):
        """多次 record() 调用，数据按序累积。"""
        env = MockEnv(
            arm_state=np.concatenate([np.zeros(6), np.zeros(6)]),
            ee_pos=np.zeros(3),
        )
        recorder = TrajectoryRecorder(
            env=env,
            init_q=np.zeros(6),
            init_q_left=np.zeros(6),
            dt=0.005,
        )
        for i in range(5):
            recorder.record(
                q_desired=np.full(6, float(i)),
                q_actual=np.full(6, float(i) * 0.9),
                timestamp=i * 0.005,
                tcp_pos=np.array([float(i), 0.0, 0.0]),
                ball_pos=np.array([float(i), 1.0, 1.0]),
            )
        traj = recorder.to_trajectory()
        assert traj.q_desired.shape == (5, 6)
        np.testing.assert_allclose(traj.q_desired[:, 0], np.arange(5))
        np.testing.assert_allclose(traj.q_actual[:, 0], np.arange(5) * 0.9)
        np.testing.assert_allclose(traj.timestamps, np.arange(5) * 0.005)
        np.testing.assert_allclose(traj.tcp_pos[:, 0], np.arange(5))

    def test_record_ball_pos_none_defaults_to_zeros(self):
        """ball_pos=None 时记录为零向量。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), 0.005)
        recorder.record(
            q_desired=np.zeros(6),
            q_actual=np.zeros(6),
            timestamp=0.0,
            tcp_pos=np.zeros(3),
            ball_pos=None,
        )
        traj = recorder.to_trajectory()
        np.testing.assert_allclose(traj.ball_pos[0], np.zeros(3))

    def test_to_trajectory_preserves_init_and_metadata(self):
        """to_trajectory() 保留 init_q / init_q_left / dt / metadata。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(
            env=env,
            init_q=np.full(6, 0.1),
            init_q_left=np.full(6, 0.2),
            dt=0.002,
            metadata={"hit_type": "active", "ball_speed": 7.0},
        )
        recorder.record(np.zeros(6), np.zeros(6), 0.0, np.zeros(3))
        traj = recorder.to_trajectory(hit_step=3)
        np.testing.assert_allclose(traj.init_q, np.full(6, 0.1))
        np.testing.assert_allclose(traj.init_q_left, np.full(6, 0.2))
        assert traj.dt == 0.002
        assert traj.hit_step == 3
        assert traj.metadata["hit_type"] == "active"
        assert traj.metadata["ball_speed"] == 7.0

    def test_record_copies_input_arrays_no_aliasing(self):
        """record() 对输入数组做 copy，修改原数组不影响内部记录数据。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), 0.005)
        q_desired = np.full(6, 0.5)
        q_actual = np.full(6, 0.3)
        tcp_pos = np.array([0.1, 0.2, 0.3])
        ball_pos = np.array([1.0, 2.0, 3.0])
        recorder.record(q_desired, q_actual, 0.0, tcp_pos, ball_pos)
        # 修改原数组，验证内部数据不受影响
        q_desired[0] = 999.0
        q_actual[0] = 999.0
        tcp_pos[0] = 999.0
        ball_pos[0] = 999.0
        traj = recorder.to_trajectory()
        np.testing.assert_allclose(traj.q_desired[0], np.full(6, 0.5))
        np.testing.assert_allclose(traj.q_actual[0], np.full(6, 0.3))
        np.testing.assert_allclose(traj.tcp_pos[0], np.array([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(traj.ball_pos[0], np.array([1.0, 2.0, 3.0]))


# ── save() / load() 往返测试 ──


class TestTrajectoryRecorderSaveLoad:
    """save() / load() 往返测试。"""

    def test_save_load_roundtrip(self, tmp_path):
        """save() → load() 所有字段保持一致。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(
            env=env,
            init_q=np.full(6, 0.1),
            init_q_left=np.full(6, 0.2),
            dt=0.005,
            metadata={"hit_type": "active", "ball_speed": 7.0},
        )
        for i in range(10):
            recorder.record(
                q_desired=np.full(6, float(i)),
                q_actual=np.full(6, float(i) * 0.95),
                timestamp=i * 0.005,
                tcp_pos=np.array([float(i), 0.0, 0.5]),
                ball_pos=np.array([float(i), 1.0, 1.0]),
            )
        path = tmp_path / "traj.npz"
        recorder.save(path, hit_step=5)

        traj = TrajectoryRecorder.load(path)
        assert traj.q_desired.shape == (10, 6)
        np.testing.assert_allclose(traj.q_desired, np.tile(np.arange(10).reshape(-1, 1), (1, 6)))
        np.testing.assert_allclose(traj.q_actual, np.tile((np.arange(10) * 0.95).reshape(-1, 1), (1, 6)))
        np.testing.assert_allclose(traj.timestamps, np.arange(10) * 0.005)
        np.testing.assert_allclose(traj.tcp_pos[:, 0], np.arange(10))
        np.testing.assert_allclose(traj.init_q, np.full(6, 0.1))
        np.testing.assert_allclose(traj.init_q_left, np.full(6, 0.2))
        assert traj.dt == 0.005
        assert traj.hit_step == 5
        assert traj.metadata["hit_type"] == "active"
        assert traj.metadata["ball_speed"] == 7.0

    def test_save_creates_npz_file(self, tmp_path):
        """save() 生成 .npz 文件，含必要字段。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), 0.005)
        recorder.record(np.zeros(6), np.zeros(6), 0.0, np.zeros(3))
        path = tmp_path / "out.npz"
        recorder.save(path)
        assert path.exists()
        data = np.load(path)
        assert "q_desired" in data.files
        assert "q_actual" in data.files
        assert "metadata" in data.files

    def test_save_auto_creates_parent_dir(self, tmp_path):
        """save() 自动创建不存在的父目录。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), 0.005)
        recorder.record(np.zeros(6), np.zeros(6), 0.0, np.zeros(3))
        path = tmp_path / "sub" / "deep" / "traj.npz"
        assert not path.parent.exists()
        recorder.save(path)
        assert path.exists()

    def test_save_default_hit_step_negative_one(self, tmp_path):
        """save() 不传 hit_step 时默认 -1。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), 0.005)
        recorder.record(np.zeros(6), np.zeros(6), 0.0, np.zeros(3))
        path = tmp_path / "default.npz"
        recorder.save(path)
        traj = TrajectoryRecorder.load(path)
        assert traj.hit_step == -1

    def test_empty_trajectory_save_load_roundtrip(self, tmp_path):
        """空轨迹（0 条记录）save()→load() 保持空数组形状。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), 0.005)
        path = tmp_path / "empty.npz"
        recorder.save(path)
        traj = TrajectoryRecorder.load(path)
        assert traj.q_desired.shape == (0, 6)
        assert traj.q_actual.shape == (0, 6)
        assert traj.timestamps.shape == (0,)
        assert traj.tcp_pos.shape == (0, 3)
        assert traj.ball_pos.shape == (0, 3)
        assert traj.dt == 0.005
        assert traj.hit_step == -1

    def test_metadata_numpy_scalars_roundtrip(self, tmp_path):
        """metadata 含 numpy 标量/数组，save→load 值保持一致（类型转为 Python 原生）。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(
            env=env,
            init_q=np.zeros(6),
            init_q_left=np.zeros(6),
            dt=0.005,
            metadata={
                "ball_speed": np.float64(7.5),
                "hit_count": np.int32(3),
                "pos_array": np.array([0.1, 0.2, 0.3]),
            },
        )
        recorder.record(np.zeros(6), np.zeros(6), 0.0, np.zeros(3))
        path = tmp_path / "np_meta.npz"
        recorder.save(path)
        traj = TrajectoryRecorder.load(path)
        # JSON 不支持 numpy 类型，_json_default 转为 Python 原生类型
        assert traj.metadata["ball_speed"] == 7.5
        assert isinstance(traj.metadata["ball_speed"], float)
        assert traj.metadata["hit_count"] == 3
        assert isinstance(traj.metadata["hit_count"], int)
        # numpy 数组转为 list
        assert traj.metadata["pos_array"] == [0.1, 0.2, 0.3]
        assert isinstance(traj.metadata["pos_array"], list)


# ── make_hook() 测试 ──


class TestTrajectoryRecorderMakeHook:
    """make_hook() 方法测试。"""

    def test_make_hook_returns_callable(self):
        """make_hook() 返回可调用对象。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), 0.005)
        hook = recorder.make_hook()
        assert callable(hook)

    def test_hook_extracts_data_and_records(self):
        """hook 从 StepContext 提取数据并调用 record()。"""
        arm_state = np.concatenate([np.full(6, 0.3), np.zeros(6)])
        ee_pos = np.array([0.5, 0.1, 0.2])
        env = MockEnv(arm_state=arm_state, ee_pos=ee_pos)
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), 0.005)

        hook = recorder.make_hook()
        ctx = MockStepContext(
            u_cmd=np.full(6, 0.4),
            ball_pos=np.array([1.0, 2.0, 3.0]),
        )
        hook(ctx)

        traj = recorder.to_trajectory()
        assert traj.q_desired.shape == (1, 6)
        # u_cmd → q_desired
        np.testing.assert_allclose(traj.q_desired[0], np.full(6, 0.4))
        # arm_state[:6] → q_actual
        np.testing.assert_allclose(traj.q_actual[0], np.full(6, 0.3))
        # ee_pos → tcp_pos
        np.testing.assert_allclose(traj.tcp_pos[0], ee_pos)
        # ball_pos
        np.testing.assert_allclose(traj.ball_pos[0], np.array([1.0, 2.0, 3.0]))
        # timestamp 为非负浮点（perf_counter 差值）
        assert isinstance(traj.timestamps[0], (float, np.floating))
        assert traj.timestamps[0] >= 0.0

    def test_hook_ball_pos_none(self):
        """hook 处理 ball_pos=None（空挥场景）。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), 0.005)
        hook = recorder.make_hook()
        ctx = MockStepContext(u_cmd=np.zeros(6), ball_pos=None)
        hook(ctx)
        traj = recorder.to_trajectory()
        np.testing.assert_allclose(traj.ball_pos[0], np.zeros(3))

    def test_hook_multiple_calls_accumulate(self):
        """hook 多次调用，数据累积。"""
        env = MockEnv(
            arm_state=np.concatenate([np.full(6, 0.5), np.zeros(6)]),
            ee_pos=np.array([0.1, 0.2, 0.3]),
        )
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), 0.005)
        hook = recorder.make_hook()
        for _ in range(3):
            hook(MockStepContext(u_cmd=np.full(6, 0.5), ball_pos=None))
        traj = recorder.to_trajectory()
        assert traj.q_desired.shape == (3, 6)
        assert traj.q_actual.shape == (3, 6)

    def test_hook_uses_simulation_time_when_step_count_present(self):
        """hook 优先用 step_count * dt 作为时间戳（仿真时间）。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), dt=0.005)
        hook = recorder.make_hook()
        ctx = MockStepContext(
            u_cmd=np.zeros(6), ball_pos=None, step_count=10
        )
        hook(ctx)
        traj = recorder.to_trajectory()
        # step_count=10, dt=0.005 → timestamp=0.05（仿真时间）
        assert traj.timestamps[0] == pytest.approx(0.05)

    def test_hook_fallback_wall_clock_when_step_count_none(self):
        """step_count=None 时回退墙钟时间（非负浮点）。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), 0.005)
        hook = recorder.make_hook()
        ctx = MockStepContext(u_cmd=np.zeros(6), ball_pos=None, step_count=None)
        hook(ctx)
        traj = recorder.to_trajectory()
        assert isinstance(traj.timestamps[0], (float, np.floating))
        assert traj.timestamps[0] >= 0.0


# ── 旧 pickle 格式兼容测试 ──


class TestTrajectoryRecorderLoadOldPickle:
    """load() 兼容旧 pickle 格式测试。"""

    def test_load_old_pickle_format(self, tmp_path):
        """加载旧 pickle 格式，字段正确转换为新 ReplayTrajectory。"""
        n_steps = 5
        # X_history 含初始状态（N+1 个），每个 (24,)
        X_history = [np.zeros(24) for _ in range(n_steps + 1)]
        for i in range(n_steps + 1):
            X_history[i][:6] = float(i) * 0.1
        # U_history 含 N 个，每个 (6,)
        U_history = [np.full(6, float(i) * 0.2) for i in range(n_steps)]
        ball_pos_history = [np.array([float(i), 1.0, 1.0]) for i in range(n_steps)]

        old_data = {
            "X_history": X_history,
            "U_history": U_history,
            "ball_pos_history": ball_pos_history,
            "init_q": np.full(6, 0.1),
            "init_q_left": np.full(6, 0.2),
            "pos_error": 0.05,
            "hit_type": "active",
            "p0": np.array([2.0, 0.0, 1.5]),
            "v0": np.array([-7.0, 0.0, 0.0]),
            "hit_step": 3,
            "post_hit_steps": 50,
        }
        path = tmp_path / "old_traj.pkl"
        with open(path, "wb") as f:
            pickle.dump(old_data, f)

        traj = TrajectoryRecorder.load(path)

        # q_desired = U_history
        expected_q_desired = np.array(U_history)
        np.testing.assert_allclose(traj.q_desired, expected_q_desired)

        # q_actual = X_history[1:][:, :6]（跳过初始状态）
        expected_q_actual = np.array([x[:6] for x in X_history[1:]])
        np.testing.assert_allclose(traj.q_actual, expected_q_actual)

        # timestamps = arange * dt（dt 默认 0.005）
        np.testing.assert_allclose(traj.timestamps, np.arange(n_steps) * 0.005)

        # tcp_pos = zeros（旧格式无记录）
        np.testing.assert_allclose(traj.tcp_pos, np.zeros((n_steps, 3)))

        # ball_pos
        np.testing.assert_allclose(traj.ball_pos, np.array(ball_pos_history))

        # dt 默认 0.005
        assert traj.dt == 0.005

        # init_q / init_q_left 保留
        np.testing.assert_allclose(traj.init_q, np.full(6, 0.1))
        np.testing.assert_allclose(traj.init_q_left, np.full(6, 0.2))

        # hit_step 保留
        assert traj.hit_step == 3

        # metadata 从旧字段构建
        assert traj.metadata["hit_type"] == "active"
        assert traj.metadata["pos_error"] == 0.05
        assert traj.metadata["post_hit_steps"] == 50

    def test_load_old_pickle_without_ball_pos(self, tmp_path):
        """旧 pickle 无 ball_pos_history 时，ball_pos 填零。"""
        n_steps = 3
        X_history = [np.zeros(24) for _ in range(n_steps + 1)]
        U_history = [np.full(6, 0.1) for _ in range(n_steps)]

        old_data = {
            "X_history": X_history,
            "U_history": U_history,
            "init_q": np.zeros(6),
            "init_q_left": np.zeros(6),
            "hit_step": -1,
        }
        path = tmp_path / "no_ball.pkl"
        with open(path, "wb") as f:
            pickle.dump(old_data, f)

        traj = TrajectoryRecorder.load(path)
        np.testing.assert_allclose(traj.ball_pos, np.zeros((n_steps, 3)))
        assert traj.hit_step == -1

    def test_load_non_dict_pickle_raises_value_error(self, tmp_path):
        """pickle 内容非 dict → ValueError。"""
        path = tmp_path / "bad.pkl"
        with open(path, "wb") as f:
            pickle.dump([1, 2, 3], f)  # list 而非 dict
        with pytest.raises(ValueError, match="非 dict"):
            TrajectoryRecorder.load(path)

    def test_load_pickle_missing_required_keys_raises_value_error(self, tmp_path):
        """pickle dict 缺少必需键 → ValueError。"""
        path = tmp_path / "missing.pkl"
        with open(path, "wb") as f:
            pickle.dump({"X_history": []}, f)  # 缺 U_history/init_q/init_q_left
        with pytest.raises(ValueError, match="缺少必需键"):
            TrajectoryRecorder.load(path)

    def test_load_corrupted_pickle_raises_value_error(self, tmp_path):
        """非法 pickle 文件 → ValueError（转换自 UnpicklingError）。"""
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"\x00\x01\x02 not a valid pickle stream")
        with pytest.raises(ValueError, match="无法解析"):
            TrajectoryRecorder.load(path)

    def test_load_old_pickle_length_mismatch_truncates(self, tmp_path, caplog):
        """X_history/U_history 长度不匹配时截断到较短长度并告警。"""
        # U_history 有 5 个，X_history 只有 3 个（含初始状态）→ q_actual 仅 2 个
        n_u = 5
        X_history = [np.zeros(24) for _ in range(3)]
        U_history = [np.full(6, 0.1) for _ in range(n_u)]

        old_data = {
            "X_history": X_history,
            "U_history": U_history,
            "init_q": np.zeros(6),
            "init_q_left": np.zeros(6),
            "hit_step": -1,
        }
        path = tmp_path / "mismatch.pkl"
        with open(path, "wb") as f:
            pickle.dump(old_data, f)

        import logging

        with caplog.at_level(logging.WARNING):
            traj = TrajectoryRecorder.load(path)

        # 截断至 min(5, 2) = 2
        assert traj.q_desired.shape == (2, 6)
        assert traj.q_actual.shape == (2, 6)
        assert traj.timestamps.shape == (2,)
        assert traj.tcp_pos.shape == (2, 3)
        assert traj.ball_pos.shape == (2, 3)
        # 应有截断告警日志
        assert any("截断" in rec.message for rec in caplog.records)


class TestOldPicklePositionMode:
    """旧 pickle 格式 is_position_mode 安全标记测试。"""

    def test_old_pickle_marks_position_mode_false(
        self, tmp_path, caplog
    ) -> None:
        """旧 pickle 格式应标记 is_position_mode=False（保守安全）。"""
        old_data = {
            "X_history": [np.zeros(24) for _ in range(5)],
            "U_history": [np.full(6, 0.1) for _ in range(4)],
            "init_q": np.zeros(6),
            "init_q_left": np.zeros(6),
            "hit_step": -1,
        }
        path = tmp_path / "old_mode.pkl"
        with open(path, "wb") as f:
            pickle.dump(old_data, f)

        import logging

        with caplog.at_level(logging.WARNING):
            traj = TrajectoryRecorder.load(path)

        assert traj.metadata.get("is_position_mode") is False
        assert any("控制模式" in rec.message for rec in caplog.records)
