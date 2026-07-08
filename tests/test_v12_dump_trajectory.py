"""V12 --dump-trajectory 集成测试。

不运行完整 V12 仿真，而是模拟 V12 中 TrajectoryRecorder 的使用流程：
    1. 创建 recorder + 注入 post_exec_hook
    2. 模拟 post-hit 手动 record()
    3. 模拟评估后 metadata 更新
    4. 验证旧 pickle 格式兼容（回归测试）
"""

import pickle

import numpy as np

from src.real.trajectory_recorder import TrajectoryRecorder
from src.ilqt.step_context import StepContext


# ── 手写 Mock 类 ──


class MockEnv:
    """模拟 RobotEnv Protocol，返回固定/可变 arm_state 和 ee_pos。

    arm_state 可通过 set_arm_state 更新，模拟仿真步进。
    """

    def __init__(
        self,
        arm_state: np.ndarray | None = None,
        ee_pos: np.ndarray | None = None,
    ) -> None:
        self._arm_state = (
            np.asarray(arm_state, dtype=float)
            if arm_state is not None
            else np.zeros(12)
        )
        self._ee_pos = (
            np.asarray(ee_pos, dtype=float)
            if ee_pos is not None
            else np.zeros(3)
        )

    def get_arm_state(self) -> np.ndarray:
        """返回右臂状态 (12,)。"""
        return self._arm_state.copy()

    def set_arm_state(self, arm_state: np.ndarray) -> None:
        """更新内部 arm_state（模拟仿真步进后状态变化）。"""
        self._arm_state = np.asarray(arm_state, dtype=float).copy()

    def get_ee_pos(self) -> np.ndarray:
        """返回末端位置 (3,)。"""
        return self._ee_pos.copy()


# ── 1. Hook 集成测试 ──


class TestTrajectoryRecorderHookIntegration:
    """模拟 EpisodeRunner post_exec_hook 流程的集成测试。"""

    def test_hook_with_real_step_context(self, tmp_path):
        """用真实 StepContext 调用 hook，验证数据提取和保存。"""
        init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45])
        init_q_left = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45])
        arm_state = np.concatenate([init_q, np.zeros(6)])
        env = MockEnv(arm_state=arm_state, ee_pos=np.array([0.5, 0.0, 0.3]))

        recorder = TrajectoryRecorder(
            env=env,
            init_q=init_q,
            init_q_left=init_q_left,
            dt=0.005,
            metadata={
                "p0": [2.0, 0.0, 1.5],
                "v0": [-7.0, 0.0, 0.0],
                "ball_speed": 7.0,
                "is_position_mode": False,
            },
        )
        hook = recorder.make_hook()

        # 模拟 5 步 EpisodeRunner 执行
        for step_count in range(5):
            u_cmd = np.full(6, 0.1 * (step_count + 1))
            ctx = StepContext(
                step_count=step_count,
                arm_state=arm_state.copy(),
                ball_pos=np.array([2.0 - 0.01 * step_count, 0.0, 1.5]),
                ball_vel=np.array([-7.0, 0.0, 0.0]),
                u_cmd=u_cmd,
            )
            # 模拟执行后状态变化
            new_arm_state = arm_state.copy()
            new_arm_state[:6] += 0.01
            env.set_arm_state(new_arm_state)
            arm_state = new_arm_state
            hook(ctx)

        # 保存并重新加载
        path = tmp_path / "hook_test.npz"
        recorder.save(path, hit_step=3)
        traj = TrajectoryRecorder.load(path)

        assert traj.q_desired.shape == (5, 6)
        assert traj.q_actual.shape == (5, 6)
        # 时间戳应为 step_count * dt = 0, 0.005, 0.01, 0.015, 0.02
        np.testing.assert_allclose(traj.timestamps, np.arange(5) * 0.005)
        # q_desired[0] = u_cmd at step 0 = 0.1
        np.testing.assert_allclose(traj.q_desired[0], np.full(6, 0.1))
        # q_desired[4] = u_cmd at step 4 = 0.5
        np.testing.assert_allclose(traj.q_desired[4], np.full(6, 0.5))
        # init_q 保留
        np.testing.assert_allclose(traj.init_q, init_q)
        np.testing.assert_allclose(traj.init_q_left, init_q_left)
        assert traj.dt == 0.005
        assert traj.hit_step == 3
        # metadata 保留初始字段
        assert traj.metadata["ball_speed"] == 7.0
        assert traj.metadata["is_position_mode"] is False
        assert traj.metadata["p0"] == [2.0, 0.0, 1.5]

    def test_hook_ball_pos_none_records_zeros(self, tmp_path):
        """hook 收到 ball_pos=None 时记录零向量（空挥场景）。"""
        env = MockEnv(arm_state=np.zeros(12), ee_pos=np.zeros(3))
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), 0.005)
        hook = recorder.make_hook()

        ctx = StepContext(
            step_count=0,
            arm_state=np.zeros(12),
            ball_pos=None,
            u_cmd=np.zeros(6),
        )
        hook(ctx)

        traj = recorder.to_trajectory()
        np.testing.assert_allclose(traj.ball_pos[0], np.zeros(3))


# ── 2. Post-hit 手动记录测试 ──


class TestPostHitRecording:
    """模拟 V12 post-hit 循环中手动调用 recorder.record() 的流程。"""

    def test_manual_record_simulates_post_hit_loop(self, tmp_path):
        """模拟 V12 post-hit 循环：手动 record + 保存 + 加载往返。"""
        init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45])
        init_q_left = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45])
        dt = 0.005
        total_horizon = 200
        follow_through_steps = 160
        post_hit_steps = 20

        env = MockEnv(arm_state=np.concatenate([init_q, np.zeros(6)]))

        recorder = TrajectoryRecorder(
            env=env,
            init_q=init_q,
            init_q_left=init_q_left,
            dt=dt,
            metadata={"ball_speed": 7.0},
        )

        # 模拟 runner 阶段（hook 记录 total_horizon + follow_through_steps 步）
        hook = recorder.make_hook()
        total_runner_steps = total_horizon + follow_through_steps
        for step_count in range(total_runner_steps):
            ctx = StepContext(
                step_count=step_count,
                arm_state=np.concatenate([init_q, np.zeros(6)]),
                ball_pos=np.array([1.0, 0.0, 1.0]),
                u_cmd=init_q.copy(),
            )
            hook(ctx)

        # 模拟 post-hit 循环（手动 record）
        x_current = np.concatenate([init_q, np.zeros(6)])
        for i in range(post_hit_steps):
            q_hold = x_current[:6].copy()
            # 模拟状态变化
            x_current = x_current.copy()
            x_current[:6] += 0.001
            env.set_arm_state(x_current)
            step_idx = total_horizon + follow_through_steps + i
            recorder.record(
                q_desired=q_hold,
                q_actual=x_current[:6].copy(),
                timestamp=step_idx * dt,
                tcp_pos=env.get_ee_pos(),
                ball_pos=np.array([0.5, 0.0, 0.5]),
            )

        # 保存并加载
        path = tmp_path / "post_hit.npz"
        recorder.save(path, hit_step=150)
        traj = TrajectoryRecorder.load(path)

        expected_total = total_runner_steps + post_hit_steps
        assert traj.q_desired.shape == (expected_total, 6)
        assert traj.q_actual.shape == (expected_total, 6)
        assert traj.timestamps.shape == (expected_total,)
        assert traj.hit_step == 150

        # runner 阶段时间戳: 0, dt, ..., (total_runner_steps-1)*dt
        np.testing.assert_allclose(
            traj.timestamps[:total_runner_steps],
            np.arange(total_runner_steps) * dt,
        )
        # post-hit 阶段时间戳: total_runner_steps*dt, ..., (total_runner_steps+post_hit_steps-1)*dt
        np.testing.assert_allclose(
            traj.timestamps[total_runner_steps:],
            (np.arange(post_hit_steps) + total_runner_steps) * dt,
        )

    def test_post_hit_record_without_hook(self, tmp_path):
        """仅手动 record（无 hook），数据正确累积。"""
        env = MockEnv(
            arm_state=np.zeros(12),
            ee_pos=np.array([0.3, 0.0, 0.2]),
        )
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), 0.005)

        for i in range(10):
            recorder.record(
                q_desired=np.full(6, float(i) * 0.1),
                q_actual=np.full(6, float(i) * 0.09),
                timestamp=(i + 1) * 0.005,
                tcp_pos=np.array([0.3 + i * 0.01, 0.0, 0.2]),
                ball_pos=np.array([1.0 - i * 0.05, 0.0, 1.0]),
            )

        path = tmp_path / "manual_only.npz"
        recorder.save(path)
        traj = TrajectoryRecorder.load(path)

        assert traj.q_desired.shape == (10, 6)
        np.testing.assert_allclose(traj.q_desired[:, 0], np.arange(10) * 0.1)
        np.testing.assert_allclose(traj.q_actual[:, 0], np.arange(10) * 0.09)
        np.testing.assert_allclose(traj.timestamps, (np.arange(10) + 1) * 0.005)


# ── 3. Metadata 更新测试 ──


class TestMetadataUpdate:
    """模拟 V12 评估后更新 _metadata 的流程。"""

    def test_metadata_update_after_evaluation(self, tmp_path):
        """初始 metadata + 评估后 update → 保存 → 加载，两者均保留。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(
            env=env,
            init_q=np.zeros(6),
            init_q_left=np.zeros(6),
            dt=0.005,
            metadata={
                "p0": [2.0, 0.0, 1.5],
                "v0": [-7.0, 0.0, 0.0],
                "ball_speed": 7.0,
                "is_position_mode": False,
            },
        )
        recorder.record(np.zeros(6), np.zeros(6), 0.0, np.zeros(3))

        # 模拟 V12 评估后更新 metadata
        recorder._metadata.update({
            "hit_type": "active",
            "pos_error": 0.035,
            "hit_step": 150,
            "post_hit_steps": 20,
        })

        path = tmp_path / "meta_update.npz"
        recorder.save(path, hit_step=150)
        traj = TrajectoryRecorder.load(path)

        # 初始字段保留
        assert traj.metadata["p0"] == [2.0, 0.0, 1.5]
        assert traj.metadata["v0"] == [-7.0, 0.0, 0.0]
        assert traj.metadata["ball_speed"] == 7.0
        assert traj.metadata["is_position_mode"] is False
        # 更新字段保留
        assert traj.metadata["hit_type"] == "active"
        assert traj.metadata["pos_error"] == 0.035
        assert traj.metadata["hit_step"] == 150
        assert traj.metadata["post_hit_steps"] == 20

    def test_metadata_overwrite_existing_key(self, tmp_path):
        """update 覆盖已存在的 key（如 hit_step 先无后有）。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(
            env=env,
            init_q=np.zeros(6),
            init_q_left=np.zeros(6),
            dt=0.005,
            metadata={"hit_step": -1},
        )
        recorder.record(np.zeros(6), np.zeros(6), 0.0, np.zeros(3))

        # V12 评估后覆盖 hit_step
        recorder._metadata.update({"hit_step": 142})
        path = tmp_path / "overwrite.npz"
        recorder.save(path, hit_step=142)
        traj = TrajectoryRecorder.load(path)

        assert traj.metadata["hit_step"] == 142
        assert traj.hit_step == 142


# ── 4. 旧 pickle 格式兼容（回归测试） ──


class TestOldPickleCompatibility:
    """V12 旧 --dump-trajectory 输出的 pickle 格式兼容回归测试。

    确保 TrajectoryRecorder.load() 能读取 V12 改造前生成的 pickle 文件。
    """

    def test_load_v12_style_old_pickle(self, tmp_path):
        """加载模拟 V12 旧格式 pickle，所有字段正确映射。"""
        n_steps = 10
        # X_history 含初始状态（N+1 个），每个 (24,)
        X_history = [np.zeros(24) for _ in range(n_steps + 1)]
        for i in range(n_steps + 1):
            X_history[i][:6] = float(i) * 0.1
        # U_history 含 N 个，每个 (6,)
        U_history = [np.full(6, float(i) * 0.2) for i in range(n_steps)]
        ball_pos_history = [
            np.array([2.0 - i * 0.1, 0.0, 1.5]) for i in range(n_steps)
        ]
        init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45])
        init_q_left = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45])

        old_data = {
            "X_history": X_history,
            "U_history": U_history,
            "ball_pos_history": ball_pos_history,
            "init_q": init_q,
            "init_q_left": init_q_left,
            "pos_error": 0.035,
            "hit_type": "active",
            "p0": np.array([2.0, 0.0, 1.5]),
            "v0": np.array([-7.0, 0.0, 0.0]),
            "hit_step": 8,
            "post_hit_steps": 20,
        }
        path = tmp_path / "v12_old.pkl"
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
        np.testing.assert_allclose(traj.init_q, init_q)
        np.testing.assert_allclose(traj.init_q_left, init_q_left)

        # hit_step 保留
        assert traj.hit_step == 8

        # metadata 从旧字段构建
        assert traj.metadata["hit_type"] == "active"
        assert traj.metadata["pos_error"] == 0.035
        assert traj.metadata["post_hit_steps"] == 20

    def test_old_pickle_with_post_hit_steps(self, tmp_path):
        """旧 pickle 含 post_hit_steps 字段时正确映射到 metadata。"""
        n_steps = 5
        X_history = [np.zeros(24) for _ in range(n_steps + 1)]
        U_history = [np.full(6, 0.1) for _ in range(n_steps)]

        old_data = {
            "X_history": X_history,
            "U_history": U_history,
            "init_q": np.zeros(6),
            "init_q_left": np.zeros(6),
            "hit_step": 3,
            "post_hit_steps": 20,
        }
        path = tmp_path / "post_hit_old.pkl"
        with open(path, "wb") as f:
            pickle.dump(old_data, f)

        traj = TrajectoryRecorder.load(path)
        assert traj.metadata["post_hit_steps"] == 20
        assert traj.hit_step == 3

    def test_new_npz_load_does_not_fallback_to_pickle(self, tmp_path):
        """新 npz 格式文件能被 load() 正确识别（不误触发 pickle 回退）。"""
        env = MockEnv(np.zeros(12), np.zeros(3))
        recorder = TrajectoryRecorder(env, np.zeros(6), np.zeros(6), 0.005)
        recorder.record(
            q_desired=np.full(6, 0.5),
            q_actual=np.full(6, 0.3),
            timestamp=0.005,
            tcp_pos=np.array([0.1, 0.2, 0.3]),
            ball_pos=np.array([1.0, 0.0, 1.0]),
        )
        path = tmp_path / "traj.npz"
        recorder.save(path, hit_step=1)
        traj = TrajectoryRecorder.load(path)

        # 应识别为新 npz 格式（有 tcp_pos 数据，旧 pickle tcp_pos 全零）
        np.testing.assert_allclose(traj.tcp_pos[0], np.array([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(traj.q_desired[0], np.full(6, 0.5))
        assert traj.hit_step == 1
