"""多机器人仿真入口测试 — V12 CLI 机器人选择分派 + run_r1d_episode。

Task 3.10 TDD（先 RED 后 GREEN）。覆盖：

1. V12 参数面（红线：RM65 缺省路径逐字不变）：
   - 新参数 ``--robot-config/--runtime-config/--sim-task-config`` 缺省 None；
   - 显式 ``configs/robots/rm65.yaml`` 与缺省等价（继续旧路径）；
   - R1D 三件套缺一、rm65 配 R1D 配置、未支持 robot_id → 构造 runner 前
     退出非零（退出码 2）；
   - RM65 专属旧参数对 R1D 必须**拒绝**（不能接受后忽略），含 ``--fix-joint5``
     点名；共享基础参数（seed/no-plot/output/log-level）明确支持；
2. ``run_r1d_episode``：真实 R1D 左/右配置短回合 → JSON+NPZ 产物
   （字段清单 + simulation-only 标记 + 两次固定参数运行逐位可复现），
   side 由完整关节名前缀校验得出；
3. import guard：r1d_runner 不依赖 ``src.real``（真机模块/SDK），
   无 SDK 机器可运行。

短回合 YAML 副本：测试基于仓库真实 ``configs/r1d_sim_{left,right}.yaml``
生成 tmp 缩减副本（episode 步数/迭代预算削减、保留来球到达所需规划视野、
字段结构不变），不改生产 configs/（r1d_sim_*.yaml 为 3.1 建、3.11 调参独占）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 沿用 tests/test_robot_profile_entrypoint.py 的导入方式（scripts/ 入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# 仓库真实 R1D 配置（左/右双侧；测试前提存在性，不修改）
PROFILE_LEFT = REPO_ROOT / "configs" / "robots" / "r1d_left.yaml"
PROFILE_RIGHT = REPO_ROOT / "configs" / "robots" / "r1d_right.yaml"
RUNTIME_LEFT = REPO_ROOT / "configs" / "robots" / "r1d_sim_runtime_left.yaml"
RUNTIME_RIGHT = REPO_ROOT / "configs" / "robots" / "r1d_sim_runtime_right.yaml"
TASK_LEFT = REPO_ROOT / "configs" / "r1d_sim_left.yaml"
TASK_RIGHT = REPO_ROOT / "configs" / "r1d_sim_right.yaml"
PROFILE_RM65 = REPO_ROOT / "configs" / "robots" / "rm65.yaml"

# run_r1d_episode 产物 JSON 必含的顶层键（Task 3.10 brief 清单）
REQUIRED_JSON_KEYS = {
    "schema", "simulation_only", "robot_id", "side", "joint_names",
    "dt", "mode", "backend", "input_sha256", "provisional_fields",
    "fixed_scene_params", "workspace", "ball_initial",
    "episode", "hit", "metrics", "seed",
}


# 短回合副本保留的规划视野（步）。生产 task 的来球约 t≈0.44s≈88 步到达
# 击球高度（configs/r1d_sim_{left,right}.yaml 文件头抛物线验算），
# find_hitting_point_physics 只在 remaining_horizon=total_horizon-step 内
# 搜可达点：若把 total_horizon 压到 episode 步数以下，第 0 步即判
# ball_unreachable、回合 0 步结束（2026-11-16 起点实测：total_steps=0、
# X 仅 1 行）。取 120 步（0.6s）覆盖来球到达并留余量；episode 步数
# 另行削减以控制测试耗时。
_SHORT_HORIZON = 120


def _short_task_copy(
    tmp_path: Path,
    source: Path,
    *,
    max_steps: int = 60,
    workspace_center: list[float] | None = None,
    workspace_radius: float | None = None,
) -> Path:
    """基于真实 task YAML 生成短回合副本（仅削减步数/迭代，结构不变）。

    Args:
        tmp_path: 目标目录。
        source: 仓库真实 task YAML（configs/r1d_sim_{left,right}.yaml）。
        max_steps: 缩减后的 episode 步数上限。
        workspace_center: 可选工作空间中心覆盖（世界系 (3,)）。仅供右臂
            tmp 副本使用：生产 r1d_sim_right.yaml 的中心过不了
            validate_workspace_against_model（3.8 局部 IK 解超限位，
            见 test_side_right_resolved 的实测依据；3.11 调参范围）。
        workspace_radius: 可选工作空间半径覆盖（m）；与中心一起用于构造
            模型可达的临时工作区（不改生产 YAML）。

    Returns:
        写出的 tmp YAML 路径（provisional 数值原样保留，缩减项仅限
        episode 步数与规划迭代预算 —— 属测试耗时控制，非调参）。
    """
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["max_steps"] = max_steps
    if workspace_center is not None:
        data["workspace_center"] = list(workspace_center)
    if workspace_radius is not None:
        data["workspace_radius"] = float(workspace_radius)
    mpc = data["mpc_config"]
    mpc.update(
        total_horizon=_SHORT_HORIZON,  # 保留来球到达视野（见上方常量注释）
        fixed_horizon=40,              # 单次 iLQR 优化视野上限（测试耗时控制）
        replan_interval=25,
        max_iter_per_plan=2,
        first_plan_iters=2,
        near_plan_iters=2,
        near_threshold=30,
    )
    solver = data["solver_config"]
    solver.update(horizon=40, max_iter=3)  # 与 fixed_horizon 对齐
    out = tmp_path / f"short_task_{source.stem}.yaml"
    out.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return out


def _make_d2_profile(tmp_path: Path) -> Path:
    """生成合法结构但身份为 d2 的 profile（未支持 robot_id 负例）。"""
    data = {
        "schema_version": 1,
        "robot_id": "d2",
        "model_path": str(REPO_ROOT / "src" / "robot" / "model.xml"),
        "active_joint_names": ["joint1"],
        "actuator_names": ["a1"],
        "held_joint_names": [],
        "tcp_site_name": "tcp",
    }
    out = tmp_path / "d2.yaml"
    out.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return out


# ============================================================================
#  1. V12 参数面：缺省行为逐字不变
# ============================================================================


class TestV12LegacyArgface:
    """新参数不改变 RM65 缺省路径的参数面。"""

    def test_new_flags_default_none(self) -> None:
        """四个新参数缺省 None → RM65 旧路径（缺省行为不变）。"""
        import rm65_mpc_v12 as v12

        args = v12._build_parser().parse_args([])
        assert args.robot_config is None
        assert args.runtime_config is None
        assert args.sim_task_config is None
        assert args.output is None

    def test_legacy_defaults_unchanged_sample(self) -> None:
        """抽检关键旧参数缺省值（防手滑改动 RM65 参数面）。"""
        import rm65_mpc_v12 as v12

        args = v12._build_parser().parse_args([])
        assert args.backswing == pytest.approx(0.6)
        assert args.bs_ratio == pytest.approx(0.35)
        assert args.r_decay == pytest.approx(0.40)
        assert args.normal_weight == pytest.approx(500000.0)
        assert args.serve_distance == pytest.approx(8.0)
        assert args.fix_joint5 is False
        assert args.serve_box is False
        assert args.position_mode is False


# ============================================================================
#  2. V12 机器人选择分派（构造 runner 前退出非零）
# ============================================================================


class TestV12RobotSelection:
    """_dispatch_robot_selection：rm65 继续旧路径，R1D 委托新 runner。"""

    @staticmethod
    def _dispatch(argv: list[str]) -> object:
        """以给定 argv 解析并执行分派（返回值：None=RM65 旧路径，int=退出码）。"""
        import rm65_mpc_v12 as v12

        parser = v12._build_parser()
        args = parser.parse_args(argv)
        return v12._dispatch_robot_selection(parser, args, argv)

    def test_default_continues_legacy(self) -> None:
        """无任何新参数 → 返回 None（RM65 旧路径继续，零改动）。"""
        assert self._dispatch([]) is None

    def test_explicit_rm65_profile_continues_legacy(self) -> None:
        """显式 rm65.yaml 与缺省等价：同样继续旧路径。"""
        assert self._dispatch(["--robot-config", str(PROFILE_RM65)]) is None

    def test_rm65_profile_with_runtime_config_rejected(self) -> None:
        """rm65 profile + R1D 运行/任务配置 → 退出码 2（组合非法）。"""
        with pytest.raises(SystemExit) as exc:
            self._dispatch(
                ["--robot-config", str(PROFILE_RM65),
                 "--runtime-config", str(RUNTIME_LEFT),
                 "--sim-task-config", str(TASK_LEFT)]
            )
        assert int(exc.value.code) == 2

    def test_r1d_missing_runtime_config_exits_2(self) -> None:
        """R1D 缺 --runtime-config → 退出码 2，消息指明缺 runtime。"""
        with pytest.raises(SystemExit) as exc:
            self._dispatch(
                ["--robot-config", str(PROFILE_LEFT),
                 "--sim-task-config", str(TASK_LEFT)]
            )
        assert int(exc.value.code) == 2

    def test_r1d_missing_task_config_exits_2(self) -> None:
        """R1D 缺 --sim-task-config → 退出码 2，消息指明缺 task。"""
        with pytest.raises(SystemExit) as exc:
            self._dispatch(
                ["--robot-config", str(PROFILE_LEFT),
                 "--runtime-config", str(RUNTIME_LEFT)]
            )
        assert int(exc.value.code) == 2

    def test_runtime_config_without_robot_config_exits_2(self) -> None:
        """--runtime-config 单独给出（无 --robot-config）→ 退出码 2。"""
        with pytest.raises(SystemExit) as exc:
            self._dispatch(["--runtime-config", str(RUNTIME_LEFT)])
        assert int(exc.value.code) == 2

    def test_task_config_without_robot_config_exits_2(self) -> None:
        """--sim-task-config 单独给出（无 --robot-config）→ 退出码 2。"""
        with pytest.raises(SystemExit) as exc:
            self._dispatch(["--sim-task-config", str(TASK_LEFT)])
        assert int(exc.value.code) == 2

    def test_missing_profile_file_exits_2(
        self, tmp_path: Path
    ) -> None:
        """--robot-config 指向不存在文件 → 退出码 2。"""
        with pytest.raises(SystemExit) as exc:
            self._dispatch(
                ["--robot-config", str(tmp_path / "no_such.yaml"),
                 "--runtime-config", str(RUNTIME_LEFT),
                 "--sim-task-config", str(TASK_LEFT)]
            )
        assert int(exc.value.code) == 2

    def test_unsupported_robot_id_exits_2(self, tmp_path: Path) -> None:
        """合法 YAML 但 robot_id 非 rm65/unitree_r1d → 退出码 2。"""
        with pytest.raises(SystemExit) as exc:
            self._dispatch(
                ["--robot-config", str(_make_d2_profile(tmp_path)),
                 "--runtime-config", str(RUNTIME_LEFT),
                 "--sim-task-config", str(TASK_LEFT)]
            )
        assert int(exc.value.code) == 2


class TestV12R1dExclusiveFlagsRejected:
    """RM65 专属旧参数对 R1D 必须拒绝（不能接受后忽略）。"""

    _BASE = ["--robot-config", "@PROFILE", "--runtime-config", "@RUNTIME",
             "--sim-task-config", "@TASK"]

    @staticmethod
    def _dispatch(argv_extra: list[str]) -> None:
        import rm65_mpc_v12 as v12

        base = [
            "--robot-config", str(PROFILE_LEFT),
            "--runtime-config", str(RUNTIME_LEFT),
            "--sim-task-config", str(TASK_LEFT),
        ]
        parser = v12._build_parser()
        argv = base + argv_extra
        args = parser.parse_args(argv)
        v12._dispatch_robot_selection(parser, args, argv)

    @pytest.mark.parametrize(
        "extra",
        [
            ["--serve-box"],
            ["--fix-joint5"],                       # brief 点名：明确拒绝
            ["--ball-speed", "7"],
            ["--dump-trajectory", "x.npz"],
            ["--position-mode"],
            ["--limits-config", "configs/real_robot.yaml"],
            ["--ablation", "none"],
        ],
    )
    def test_rm65_exclusive_flag_rejected(self, extra: list[str]) -> None:
        """每个 RM65 专属参数与 R1D 三件套同给 → 退出码 2 且点名该参数。"""
        with pytest.raises(SystemExit) as exc:
            self._dispatch(extra)
        assert int(exc.value.code) == 2

    @pytest.mark.parametrize(
        ("extra", "expected_option"),
        [
            # 审查 P2：值恰等于解析器缺省的显式输入也曾被静默放行
            (["--target-speed", "1.8"], "--target-speed"),
            (["--normal-weight", "500000"], "--normal-weight"),
            (["--target-speed=1.8"], "--target-speed"),   # --opt=value 形式
            (["--target-speed", "1.2"], "--target-speed"),  # 值不同同样拒绝
            (["--target-spe", "1.8"], "--target-speed"),  # 唯一前缀缩写
        ],
    )
    def test_rm65_exclusive_flag_default_value_rejected(
        self, extra: list[str], expected_option: str
    ) -> None:
        """显式传入即拒绝，即使值与解析器缺省相同（argv 实际出现判定）。"""
        import rm65_mpc_v12 as v12

        base = [
            "--robot-config", str(PROFILE_LEFT),
            "--runtime-config", str(RUNTIME_LEFT),
            "--sim-task-config", str(TASK_LEFT),
        ]
        parser = v12._build_parser()
        argv = base + extra
        args = parser.parse_args(argv)
        with pytest.raises(SystemExit) as exc:
            v12._dispatch_robot_selection(parser, args, argv)
        assert int(exc.value.code) == 2
        rejected = getattr(v12, "_last_r1d_rejected_flags", None)
        assert rejected is not None and expected_option in rejected

    def test_fix_joint5_message_explicit(self) -> None:
        """--fix-joint5 的拒绝消息明确点名（brief 单列条款）。"""
        import rm65_mpc_v12 as v12

        argv = [
            "--robot-config", str(PROFILE_LEFT),
            "--runtime-config", str(RUNTIME_LEFT),
            "--sim-task-config", str(TASK_LEFT),
            "--fix-joint5"
        ]
        parser = v12._build_parser()
        args = parser.parse_args(argv)
        with pytest.raises(SystemExit) as exc:
            v12._dispatch_robot_selection(parser, args, argv)
        assert int(exc.value.code) == 2
        # 消息须含被拒参数名（argparse error 输出到 stderr；此处校验
        # 分派函数在退出前显式登记的拒绝清单）
        rejected = getattr(v12, "_last_r1d_rejected_flags", None)
        assert rejected is not None and "--fix-joint5" in rejected


class TestV12SharedFlagsAccepted:
    """共享基础参数（seed/no-plot/viewer/output/log-level）对 R1D 明确支持。"""

    def test_shared_flags_reach_runner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """三件套 + 共享参数 → run_r1d_episode 被调用，退出码 0。"""
        import rm65_mpc_v12 as v12

        calls: dict[str, object] = {}

        def _fake_run(
            profile_path: Path,
            runtime_path: Path,
            task_path: Path,
            output_dir: Path,
            *,
            seed: int | None = None,
            viewer: bool = False,
        ) -> dict:
            calls["profile"] = profile_path
            calls["runtime"] = runtime_path
            calls["task"] = task_path
            calls["output"] = output_dir
            calls["viewer"] = viewer
            return {"hit": {"type": "miss"}, "episode": {"total_steps": 0}}

        monkeypatch.setattr(v12, "run_r1d_episode", _fake_run)
        out_dir = tmp_path / "r1d_out"
        parser = v12._build_parser()
        argv = [
            "--robot-config", str(PROFILE_LEFT),
            "--runtime-config", str(RUNTIME_LEFT),
            "--sim-task-config", str(TASK_LEFT),
            "--seed", "42", "--no-plot", "--viewer", "--log-level", "WARNING",
            "--output", str(out_dir)
        ]
        args = parser.parse_args(argv)
        rc = v12._dispatch_robot_selection(parser, args, argv)
        assert rc == 0
        assert calls["profile"] == PROFILE_LEFT
        assert calls["runtime"] == RUNTIME_LEFT
        assert calls["task"] == TASK_LEFT
        assert calls["output"] == out_dir
        assert calls["viewer"] is True

    def test_default_output_dir_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """未给 --output 时使用缺省 results/r1d_sim 目录（仍进 runner）。"""
        import rm65_mpc_v12 as v12

        calls: dict[str, object] = {}

        def _fake_run(*args: object, **kwargs: object) -> dict:
            calls["output"] = args[3]
            return {}

        monkeypatch.setattr(v12, "run_r1d_episode", _fake_run)
        parser = v12._build_parser()
        argv = [
            "--robot-config", str(PROFILE_LEFT),
            "--runtime-config", str(RUNTIME_LEFT),
            "--sim-task-config", str(TASK_LEFT)
        ]
        args = parser.parse_args(argv)
        rc = v12._dispatch_robot_selection(parser, args, argv)
        assert rc == 0
        assert Path(calls["output"]).name == "r1d_sim"


class TestV12SideMismatchExitsBeforeRunner:
    """profile 侧别错配在构造 runner（环境/求解器）前退出非零。"""

    def test_left_profile_right_task_exits_2_before_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """left profile 配 right task → 退出码 2，且环境工厂未被调用。"""
        import rm65_mpc_v12 as v12
        import src.sim.r1d_runner as r1d_runner_mod

        builds = {"n": 0}

        def _spy_build(*args: object, **kwargs: object) -> object:
            builds["n"] += 1
            raise AssertionError("侧别错配不得进入环境构造")

        monkeypatch.setattr(
            r1d_runner_mod, "build_r1d_environments", _spy_build
        )
        parser = v12._build_parser()
        argv = [
            "--robot-config", str(PROFILE_LEFT),
            "--runtime-config", str(RUNTIME_LEFT),
            "--sim-task-config", str(TASK_RIGHT)
        ]
        args = parser.parse_args(argv)
        with pytest.raises(SystemExit) as exc:
            v12._dispatch_robot_selection(parser, args, argv)
        assert int(exc.value.code) == 2
        assert builds["n"] == 0


# ============================================================================
#  3. run_r1d_episode：拒绝语义（构造环境前）
# ============================================================================


class TestRunR1dEpisodeRejections:
    """配置身份/侧别错误在重资产构造前抛 ValueError。"""

    def test_non_r1d_profile_rejected(self) -> None:
        """rm65 profile 进入 R1D runner → ValueError（点名 robot_id）。"""
        from src.sim.r1d_runner import run_r1d_episode

        with pytest.raises(ValueError, match="rm65"):
            run_r1d_episode(
                PROFILE_RM65, RUNTIME_LEFT, TASK_LEFT, Path("unused")
            )

    def test_side_mismatch_rejected_before_env_build(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """left profile + right task → ValueError，build_r1d_environments 0 次。"""
        import src.sim.r1d_runner as r1d_runner_mod
        from src.sim.r1d_runner import run_r1d_episode

        builds = {"n": 0}

        def _spy_build(*args: object, **kwargs: object) -> object:
            builds["n"] += 1
            raise AssertionError("侧别错配不得进入环境构造")

        monkeypatch.setattr(
            r1d_runner_mod, "build_r1d_environments", _spy_build
        )
        with pytest.raises(ValueError, match="joint_names"):
            run_r1d_episode(
                PROFILE_LEFT, RUNTIME_LEFT, TASK_RIGHT, Path("unused")
            )
        assert builds["n"] == 0

    def test_runtime_side_mismatch_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """left profile + right runtime → ValueError，环境工厂 0 次。"""
        import src.sim.r1d_runner as r1d_runner_mod
        from src.sim.r1d_runner import run_r1d_episode

        builds = {"n": 0}
        monkeypatch.setattr(
            r1d_runner_mod, "build_r1d_environments",
            lambda *a, **k: builds.__setitem__("n", builds["n"] + 1),
        )
        with pytest.raises(ValueError):
            run_r1d_episode(
                PROFILE_LEFT, RUNTIME_RIGHT, TASK_LEFT, Path("unused")
            )
        assert builds["n"] == 0

    def test_missing_task_file_raises(self) -> None:
        """task YAML 不存在 → FileNotFoundError（入口层转退出非零）。"""
        from src.sim.r1d_runner import run_r1d_episode

        with pytest.raises(FileNotFoundError):
            run_r1d_episode(
                PROFILE_LEFT, RUNTIME_LEFT,
                Path("no_such_task.yaml"), Path("unused")
            )

    def test_rm65_only_mpc_keys_rejected(
        self, tmp_path: Path
    ) -> None:
        """task mpc_config 含 RM65 专属键（如 fix_joint5_angle）→ ValueError。"""
        from src.sim.r1d_runner import run_r1d_episode

        task = _short_task_copy(tmp_path, TASK_LEFT)
        data = yaml.safe_load(task.read_text(encoding="utf-8"))
        data["mpc_config"]["fix_joint5_angle"] = 0.1
        task.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError, match="fix_joint5"):
            run_r1d_episode(
                PROFILE_LEFT, RUNTIME_LEFT, task, tmp_path / "out"
            )


# ============================================================================
#  4. R1D 回放（无显示环境用 fake viewer 验证状态写入）
# ============================================================================


class TestR1dReplay:
    """R1D --viewer 回放函数：逐帧写状态并同步，不触碰真实窗口。"""

    def test_replay_writes_arm_and_ball_frames(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """fake viewer 下至少写入一帧关节/球状态，并退出循环。"""
        import mujoco.viewer
        from src.sim.r1d_runner import _replay_r1d_episode

        class _Cam:
            def __init__(self) -> None:
                self.distance = 0.0
                self.elevation = 0.0
                self.azimuth = 0.0
                self.lookat = [0.0, 0.0, 0.0]

        class _Viewer:
            def __init__(self) -> None:
                self.cam = _Cam()
                self.checks = 0
                self.synced = 0

            def __enter__(self) -> "_Viewer":
                return self

            def __exit__(self, *args: object) -> bool:
                return False

            def is_running(self) -> bool:
                self.checks += 1
                return self.checks <= 4

            def sync(self) -> None:
                self.synced += 1

        class _Execution:
            model = object()
            data = object()

            def __init__(self) -> None:
                self.arm: list[np.ndarray] = []
                self.ball: list[tuple[np.ndarray, np.ndarray]] = []

            def set_arm_state(self, x: np.ndarray) -> None:
                self.arm.append(np.array(x))

            def set_ball_state(
                self, pos: np.ndarray, vel: np.ndarray
            ) -> None:
                self.ball.append((np.array(pos), np.array(vel)))

        fake_viewer = _Viewer()
        monkeypatch.setattr(
            mujoco.viewer, "launch_passive",
            lambda model, data: fake_viewer,
        )
        execution = _Execution()
        _replay_r1d_episode(
            execution,
            np.zeros((3, 14)),
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            0.01,
            speed=1000.0,
        )
        assert execution.arm
        assert execution.ball
        assert fake_viewer.synced >= 1


# ============================================================================
#  5. run_r1d_episode：短回合产物（JSON + NPZ）
# ============================================================================


class TestRunR1dEpisodeOutputs:
    """真实 R1D 配置短回合：产物字段、标记与可复现性。"""

    @pytest.fixture(scope="class")
    def left_short_episode(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> dict:
        """跑一次左侧短回合（真实全链：规划/安全/执行/接触判定）。

        class 级共享：两个测试类（字段/复现）复用同一次运行的结果目录。
        """
        from src.sim.r1d_runner import run_r1d_episode

        tmp = tmp_path_factory.mktemp("r1d_left_short")
        task = _short_task_copy(tmp, TASK_LEFT)
        result = run_r1d_episode(
            PROFILE_LEFT, RUNTIME_LEFT, task, tmp / "out", seed=7
        )
        return {"result": result, "out_dir": tmp / "out", "task": task}

    def test_json_fields_complete(
        self, left_short_episode: dict
    ) -> None:
        """JSON 产物字段覆盖 brief 清单，且标记 simulation-only。"""
        out_dir: Path = left_short_episode["out_dir"]
        json_path = out_dir / "r1d_episode.json"
        assert json_path.is_file()
        import json

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        missing = REQUIRED_JSON_KEYS - set(payload)
        assert not missing, f"JSON 缺字段: {missing}"
        # 标记：新输出 simulation-only（不作真机轨迹承诺）
        assert payload["simulation_only"] is True
        # 身份与侧别（side 由完整关节名校验得出）
        assert payload["robot_id"] == "unitree_r1d"
        assert payload["side"] == "left"
        assert payload["joint_names"][0] == "left_shoulder_pitch_joint"
        assert len(payload["joint_names"]) == 7
        # 模式/后端/步长
        assert payload["mode"] == "torque"
        assert "python" in payload["backend"]["effective"]
        assert payload["dt"] == pytest.approx(0.005)
        # 输入哈希与 provisional 标记
        sha = payload["input_sha256"]
        assert {"profile_yaml", "runtime_yaml", "task_yaml",
                "model_template"} <= set(sha)
        for name, digest in sha.items():
            assert isinstance(digest, str) and len(digest) == 64, (
                f"{name} 应为 SHA256 十六进制摘要"
            )
            int(digest, 16)  # 非法十六进制在此抛 ValueError
        assert "dt" in payload["provisional_fields"]
        # 固定高度/球拍参数（scene 几何）
        scene = payload["fixed_scene_params"]
        assert scene["base_position"][2] == pytest.approx(1.0)
        assert scene["racket_radius"] == pytest.approx(0.12)
        # 接触类型/时刻/速度 + 规划耗时 + 失败原因字段（brief 输出清单逐项）
        assert {"type", "step", "time_s", "racket_face_speed"} <= set(
            payload["hit"]
        )
        assert {"max_steps", "total_steps", "safe_steps", "first_plan_ms",
                "failure_reason"} <= set(payload["episode"])
        # episode 与接触记录
        assert 0 < payload["episode"]["total_steps"] <= 60
        assert payload["hit"]["type"] in ("active", "passive", "miss")
        assert isinstance(payload["hit"]["step"], int)
        # 拍速：发生接触时为 float，未触球时为 None（如实记录，不造 0）
        assert payload["hit"]["racket_face_speed"] is None or isinstance(
            payload["hit"]["racket_face_speed"], float
        )

    def test_npz_series_shapes(
        self, left_short_episode: dict
    ) -> None:
        """NPZ 时间序列：X/U/ball 形状对齐且数值有限。"""
        import numpy as np

        npz_path: Path = left_short_episode["out_dir"] / "r1d_episode.npz"
        assert npz_path.is_file()
        data = np.load(npz_path)
        assert set(data.files) >= {"X", "U", "ball_pos", "ball_vel", "t"}
        X, U = data["X"], data["U"]
        ball_pos, ball_vel, t = data["ball_pos"], data["ball_vel"], data["t"]
        assert X.ndim == 2 and X.shape[1] == 14
        assert U.ndim == 2 and U.shape[1] == 7
        assert U.shape[0] == X.shape[0] - 1
        assert ball_pos.shape == (X.shape[0], 3)
        assert ball_vel.shape == (X.shape[0], 3)
        assert t.shape == (X.shape[0],)
        assert bool(np.all(np.isfinite(X)))
        assert bool(np.all(np.isfinite(U)))
        assert t[0] == pytest.approx(0.0)
        assert t[1] - t[0] == pytest.approx(0.005)

    def test_result_dict_json_serializable(
        self, left_short_episode: dict
    ) -> None:
        """返回值可直接 json.dumps（契约：JSON 可序列化指标）。"""
        import json

        result: dict = left_short_episode["result"]
        text = json.dumps(result, ensure_ascii=False)
        assert "unitree_r1d" in text

    def test_side_right_resolved(self, tmp_path: Path) -> None:
        """右侧配置运行 → side == "right"（由 joint_names 前缀校验）。

        工作空间中心覆盖依据（实测 IK，不改生产 YAML）：生产
        r1d_sim_right.yaml 的中心 [0.25,-0.25,1.35] 过不了
        validate_workspace_against_model —— 右肩侧摆（joint 1）限位为
        [-142.007°, +12.995°]（r1d_model_right.xml L77，URDF 非对称限位
        → 上界 +0.2268 rad），自规划初值 [0,0.2,0,-0.6,0,0,0] 的局部 IK
        收敛到 q[1]=0.3289 rad 超上界（2026-11-16 起点实测原话：
        工作空间中心/半径对模型校验失败）。**注意**：换 IK 初值
        （如 q[1]=-0.6）存在限位内分支 q[1]=-0.3779、末端误差 0.9mm，
        故这不是"物理绝对不可达"，而是 3.8 校验用单初值局部 IK 的
        分支选择问题——属 3.11 调参/上游校验范围，3.10 不修改生产 YAML
        与上游实现。
        tmp 副本改用生产初值即可通过校验的实测点 [0.25,-0.10,1.35]：
        BoundRobotEnv.solve_ik（规划初值、max_iter=100、eps=5e-3）解
        q=[0.3952,0.0717,0.0619,-0.1524,-0.1892,-0.2833,0.0419]，
        末端误差 4.6mm < 50mm，全关节在限位内（最小裕度 8.9°@joint1），
        validate_workspace_against_model 通过；来球轨迹在该中心
        0.45m 可达半径内（t≈0.44s 球位 [0.30,-0.32,1.41]，距中心 0.23m）。
        """
        from src.sim.r1d_runner import run_r1d_episode

        task = _short_task_copy(
            tmp_path, TASK_RIGHT, max_steps=30,
            workspace_center=[0.25, -0.10, 1.35],
        )
        result = run_r1d_episode(
            PROFILE_RIGHT, RUNTIME_RIGHT, task, tmp_path / "out_r"
        )
        assert result["side"] == "right"
        assert result["joint_names"][0] == "right_shoulder_pitch_joint"

    def test_two_identical_runs_reproducible_bitwise(
        self, tmp_path: Path
    ) -> None:
        """固定参数执行两次短回合：字段/数组逐位一致（同 seed）。

        短回合保留 120 步规划视野、实际执行 40 步（球在生产初态下可达，
        首个规划成功），确保复现性断言作用在非零多步的 X/U/ball 序列上，
        而不是"第 0 步即 ball_unreachable"的空回合。
        """
        import json

        import numpy as np

        from src.sim.r1d_runner import run_r1d_episode

        task = _short_task_copy(tmp_path, TASK_LEFT, max_steps=40)
        out1, out2 = tmp_path / "run1", tmp_path / "run2"
        r1 = run_r1d_episode(PROFILE_LEFT, RUNTIME_LEFT, task, out1, seed=3)
        r2 = run_r1d_episode(PROFILE_LEFT, RUNTIME_LEFT, task, out2, seed=3)

        # NPZ 数组逐位一致（MuJoCo/FD 确定性，无随机源）
        d1 = np.load(out1 / "r1d_episode.npz")
        d2 = np.load(out2 / "r1d_episode.npz")
        assert set(d1.files) == set(d2.files)
        for key in d1.files:
            assert np.array_equal(d1[key], d2[key]), f"数组 {key} 不逐位一致"

        # JSON 确定性字段一致（排除墙钟耗时等非确定字段）
        j1 = json.loads((out1 / "r1d_episode.json").read_text(encoding="utf-8"))
        j2 = json.loads((out2 / "r1d_episode.json").read_text(encoding="utf-8"))
        for field in ("robot_id", "side", "joint_names", "dt", "mode",
                      "backend", "input_sha256", "provisional_fields",
                      "fixed_scene_params", "workspace", "ball_initial",
                      "hit", "seed"):
            assert j1[field] == j2[field], f"JSON 字段 {field} 不一致"
        assert j1["episode"]["total_steps"] == j2["episode"]["total_steps"]
        assert r1["hit"] == r2["hit"]

    def test_failure_reason_recorded_when_no_progress(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """首个执行步被安全滤波拒绝（球可达）→ failure_reason=safety_filter_stop。

        来球在 120 步规划视野内可达（首个规划成功、不触发 ball_unreachable），
        随后注入"恒不安全"的滤波替身，确保停止来自安全拒绝分支而非
        "球不可达"提前结束；可观察标志经 on_unsafe hook 置位。
        """
        import numpy as np

        import src.sim.r1d_runner as r1d_runner_mod
        from src.sim.r1d_runner import run_r1d_episode

        class _AlwaysUnsafe:
            """恒不安全的安全滤波替身（停止分支可观察性）。"""

            def filter(self, u_cmd: np.ndarray, arm_state: np.ndarray):
                """恒拒绝。"""
                return np.asarray(u_cmd), False

        # 最小侵入：替换 r1d_runner 命名空间的 PredictiveSafetyFilter 构造
        original_cls = r1d_runner_mod.PredictiveSafetyFilter

        def _fake_filter(*args: object, **kwargs: object) -> object:
            return _AlwaysUnsafe()

        monkeypatch.setattr(
            r1d_runner_mod, "PredictiveSafetyFilter", _fake_filter
        )
        try:
            task = _short_task_copy(tmp_path, TASK_LEFT, max_steps=20)
            result = run_r1d_episode(
                PROFILE_LEFT, RUNTIME_LEFT, task, tmp_path / "out_f"
            )
        finally:
            monkeypatch.setattr(
                r1d_runner_mod, "PredictiveSafetyFilter", original_cls
            )
        assert result["episode"]["safe_steps"] == 0
        assert result["episode"]["failure_reason"] == "safety_filter_stop"
        assert result["hit"]["type"] == "miss"


# ============================================================================
#  5. import guard：无 SDK / 无 real 模块依赖
# ============================================================================


class TestImportGuard:
    """入口不导入真实 SDK、不开放 unitree 真机入口。"""

    def test_r1d_runner_source_has_no_real_import(self) -> None:
        """r1d_runner 源码不含 src.real import（契约：不连接 real 模块）。"""
        source = (REPO_ROOT / "src" / "sim" / "r1d_runner.py").read_text(
            encoding="utf-8"
        )
        assert "src.real" not in source
        assert "robot_interface" not in source
        assert "RobotInterface" not in source

    def test_r1d_runner_importable_with_real_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """把 src.real.* 全部置为不可导入后，r1d_runner 仍可正常 import。"""
        import importlib

        class _Blocker:
            """按前缀阻断 import 的 meta_path 钩子。"""

            def find_module(self, fullname: str, path: object = None):
                """阻断 src.real.* 的查找。"""
                if fullname.startswith("src.real"):
                    return self
                return None

            def load_module(self, fullname: str):
                """阻断时抛 ImportError。"""
                raise ImportError(f"测试阻断: {fullname}")

        blocker = _Blocker()
        monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
        # 移除已导入的 r1d_runner 强制重导入
        saved = sys.modules.pop("src.sim.r1d_runner", None)
        try:
            module = importlib.import_module("src.sim.r1d_runner")
            assert callable(module.run_r1d_episode)
        finally:
            if saved is not None:
                sys.modules["src.sim.r1d_runner"] = saved

    def test_v12_import_does_not_pull_sdk(self) -> None:
        """子进程隔离：全新解释器 import 入口 → 不拉入真机 SDK/real_runner。

        同一 pytest 进程收集 tests/test_robot_profile_entrypoint.py 时会
        import run_real_robot（其模块级导入 src.real.real_runner），
        令本进程 sys.modules 检查受收集污染——即使被测入口本身不导入。
        改用独立解释器验证（等价"无 SDK 机器直接运行入口"的真实条件）：
        子进程 import V12 入口 + R1D 运行器后，sys.modules 不得出现
        src.real.robot_interface / src.real.real_runner 及 SDK 根模块。
        """
        import json
        import os
        import subprocess

        script = (
            "import json, sys\n"
            "import rm65_mpc_v12\n"
            "import src.sim.r1d_runner\n"
            "blocked = sorted(\n"
            "    name for name in sys.modules\n"
            "    if name.split('.')[0] in ('Robotic_Arm', 'rm_robot_interface')\n"
            "    or name.startswith('src.real.robot_interface')\n"
            "    or name.startswith('src.real.real_runner')\n"
            ")\n"
            "print(json.dumps({'blocked': blocked}))\n"
            "sys.exit(1 if blocked else 0)\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT), str(REPO_ROOT / "scripts"),
             env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        assert proc.returncode == 0, (
            f"入口导入拉入被禁止模块或子进程失败（rc={proc.returncode}）：\n"
            f"stdout={proc.stdout[-2000:]}\nstderr={proc.stderr[-4000:]}"
        )
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        assert payload["blocked"] == []


# ============================================================================
#  6. R15：旧固定任务入口不得静默忽略新共享参数 / 缺文件必须早退
# ============================================================================


class TestV12FixedTaskRejectsNewSharedFlags:
    """``--sim-task-config`` 兼容分支对球拍/任务覆盖/异步重规划显式报错。"""

    _BASE = [
        "--robot-config", str(PROFILE_LEFT),
        "--runtime-config", str(RUNTIME_LEFT),
    ]

    @staticmethod
    def _dispatch(argv: list[str]) -> object:
        """解析并分派给定 argv（返回退出码；错误路径抛 SystemExit(2)）。"""
        import rm65_mpc_v12 as v12

        parser = v12._build_parser()
        args = parser.parse_args(argv)
        return v12._dispatch_robot_selection(parser, args, argv)

    @pytest.mark.parametrize(
        ("extra", "option"),
        [
            (["--racket-config", "does-not-exist.yaml"], "--racket-config"),
            (["--task-overrides", "unknown.option=123"], "--task-overrides"),
            (["--async-replan"], "--async-replan"),
        ],
    )
    def test_explicit_new_flag_rejected(
        self, extra: list[str], option: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """固定任务 + 三个新参数分别退出 2，stderr 点名该参数。"""
        argv = self._BASE + ["--sim-task-config", str(TASK_LEFT)] + extra
        with pytest.raises(SystemExit) as exc:
            self._dispatch(argv)
        assert int(exc.value.code) == 2
        assert option in capsys.readouterr().err

    def test_missing_task_file_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--sim-task-config 指向不存在文件 → 退出 2，不进入 runner。"""
        argv = self._BASE + [
            "--sim-task-config", str(tmp_path / "no_such_task.yaml"),
            "--output", str(tmp_path / "out"),
        ]
        with pytest.raises(SystemExit) as exc:
            self._dispatch(argv)
        assert int(exc.value.code) == 2
        assert "任务文件不存在" in capsys.readouterr().err

    def test_missing_runtime_file_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--runtime-config 指向不存在文件 → 退出 2，不进入 runner。"""
        argv = [
            "--robot-config", str(PROFILE_LEFT),
            "--runtime-config", str(tmp_path / "no_such_runtime.yaml"),
            "--sim-task-config", str(TASK_LEFT),
            "--output", str(tmp_path / "out"),
        ]
        with pytest.raises(SystemExit) as exc:
            self._dispatch(argv)
        assert int(exc.value.code) == 2
        assert "运行参数文件不存在" in capsys.readouterr().err

    def test_no_extra_flags_runs_old_schema(self, tmp_path: Path) -> None:
        """无额外参数仍返回 0，并产出旧 ``r1d_sim_episode_v1`` schema。

        生产左右 task 的 workspace 中心在 T0 基线上过不了
        ``validate_workspace_against_model``（既有 3F+3E，见
        ``docs/experiments/reports/2026-11-18-r1d-v12-parity.md``），与本任务
        R15 无关。为真正走完旧 runner，这里用初始 TCP 位置作为可达中心
        （半径 0.2 m）构造 tmp 副本：仅改测试输入，不改生产 configs。
        """
        import json

        from src.robot.robot_profile import load_robot_profile
        from src.robot.runtime_parameters import load_runtime_parameters
        from src.sim.env_factory import build_r1d_environments
        from src.sim.sim_config import load_sim_task_config

        profile = load_robot_profile(PROFILE_LEFT)
        runtime = load_runtime_parameters(RUNTIME_LEFT, profile)
        base_task = load_sim_task_config(TASK_LEFT, profile)
        envs = build_r1d_environments(profile, runtime, base_task)
        center = envs.execution.get_ee_pos().tolist()

        task = _short_task_copy(
            tmp_path, TASK_LEFT, max_steps=20,
            workspace_center=center, workspace_radius=0.2,
        )
        out_dir = tmp_path / "out"
        rc = self._dispatch(
            [
                "--robot-config", str(PROFILE_LEFT),
                "--runtime-config", str(RUNTIME_LEFT),
                "--sim-task-config", str(task),
                "--seed", "42",
                "--no-plot",
                "--output", str(out_dir),
            ]
        )
        assert rc == 0
        payload = json.loads(
            (out_dir / "r1d_episode.json").read_text(encoding="utf-8")
        )
        assert payload["schema"] == "r1d_sim_episode_v1"
        assert payload["robot_id"] == "unitree_r1d"
