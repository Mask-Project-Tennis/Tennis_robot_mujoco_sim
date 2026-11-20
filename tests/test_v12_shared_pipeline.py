"""V12 共享闭环管线测试（Task 5）。

覆盖 brief §5 要求：

* RM65 共享管线冒烟（n=6、X/U 形状、hit_type 集合、不依赖 active）；
* R1D overlay 冒烟（n=7、joint_names 7、球自由关节地址不截断、允许 miss）；
* RM65 输出保真（``__RESULT__``/打印块 + T0 基线 NPZ 逐位对照）；
* 旧 ``run_r1d_episode`` 产物兼容（mappingproxy 回归：``fixed_scene_params``
  可 JSON 序列化且 ``fixed_joint_poses`` 键值保留）；
* ``SimComponent`` 接近半径参数化（RM65 缺省 0.033/0.12/0.03；R1D 由
  bundle 球拍给出）；
* 发球 RNG 顺序与机器人分支无关（同 seed 直接重算 == 管线结果）；
* ``MPCConfig`` 未知键早失败。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
# 子进程统一 UTF-8：Windows 默认 cp936 会把中文日志写成 GBK，
# 父进程按 UTF-8 解码时会得到替换字符，导致对旧中文打印块的断言假失败。
_UTF8_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("mujoco")

import rm65_mpc_v12 as v12  # noqa: E402
from src.ilqt.components import sim_component as sim_component_module  # noqa: E402
from src.sim import v12_episode as v12_episode_module  # noqa: E402
from src.sim.r1d_runner import _scene_to_jsonable  # noqa: E402
from src.sim.v12_episode import run_v12_episode  # noqa: E402
from src.sim.v12_robot_adapter import build_v12_robot  # noqa: E402
from src.robot.robot_profile import load_robot_profile  # noqa: E402
from src.robot.runtime_parameters import load_runtime_parameters  # noqa: E402

_R1D_RIGHT_PROFILE = ROOT / "configs" / "robots" / "r1d_right.yaml"
_R1D_RIGHT_RUNTIME = ROOT / "configs" / "robots" / "r1d_v12_runtime_right.yaml"
_T0_TORQUE_BASELINE = (
    ROOT / "results" / "r1d_v12_parity" / "baseline" / "rm65_torque_a.npz"
)

_RM65_SHORT_ARGV = [
    "--serve-box",
    "--ball-speed", "7",
    "--seed", "42",
    "--no-plot",
    "--horizon", "60",
    "--iter", "2",
    "--near-iters", "2",
    "--first-plan-iters", "2",
    "--near-threshold", "30",
    "--replan-interval", "20",
]


def _parse_and_resolve(argv: list[str]) -> tuple[argparse.Namespace, object]:
    """用真实 CLI 解析器 + T3 解析层构造配置。

    Args:
        argv: 命令行参数（不含程序名）。

    Returns:
        ``(args, ResolvedV12Config)``。
    """
    parser = v12._build_parser()
    args = parser.parse_args(argv)
    explicit = v12._explicit_cli_actions(parser, argv)
    config = v12.resolve_v12_config(args, explicit)
    return args, config


def _assert_arm_shapes(result: object, n: int) -> None:
    """断言 X/U/ball_pos/ball_vel/time 形状与有限性。"""
    x = result.X
    u = result.U
    assert x.shape[1] == 2 * n
    assert u.shape[1] == n
    assert x.shape[0] == u.shape[0] + 1
    assert result.ball_pos.shape[0] == x.shape[0]
    assert result.ball_vel.shape[0] == x.shape[0]
    assert result.time.shape[0] == x.shape[0]
    for arr in (x, u, result.ball_pos, result.ball_vel, result.time):
        assert np.all(np.isfinite(arr))


# ── 1. RM65 共享管线冒烟 ───────────────────────────────────────────────
def test_v12_episode_rm65_smoke() -> None:
    """RM65（n=6）共享管线可跑通，且不依赖命中 active。"""
    _, config = _parse_and_resolve(_RM65_SHORT_ARGV)
    result = run_v12_episode(config)

    assert result.robot_id == "rm65"
    assert result.mode == "torque"
    assert len(result.joint_names) == 6
    assert result.racket_id is None
    _assert_arm_shapes(result, 6)
    assert result.hit_type in {"active", "passive", "miss"}
    # 不要求 active：本测试仅校验形状/字段/失败原因集合
    assert result.metrics["total_steps"] > 0
    assert result.metrics["failure_reason"] in {
        None, "safety_filter_stop", "ball_state_unavailable", "ball_unreachable",
    }
    # RM65 保持 V12 旧接近半径缺省
    proximity = result.metrics["proximity"]
    assert proximity["ball_radius"] == pytest.approx(0.033)
    assert proximity["racket_radius"] == pytest.approx(0.12)
    assert proximity["proximity_margin"] == pytest.approx(0.03)
    assert proximity["ball_near_threshold"] == pytest.approx(0.033 + 0.12 + 0.03)
    # RM65 shoulder_pos 仍取 V12 常量（T0 事实），不是模型派生
    from src.robot.constants import SHOULDER_POS

    assert result.metrics["shoulder_pos_source"] == "cli/scene.shoulder_pos"
    assert np.allclose(
        np.asarray(result.metrics["shoulder_pos"], dtype=np.float64),
        np.asarray(SHOULDER_POS, dtype=np.float64),
        atol=0.0, rtol=0.0,
    )


# ── 2. R1D overlay 冒烟（七维、按名映射、允许 miss）───────────────────
def test_v12_episode_r1d_overlay_smoke(tmp_path: Path) -> None:
    """R1D 右臂 overlay（n=7）进入共享闭环；球自由关节地址不截断。"""
    overrides = v12._parse_task_overrides(
        [
            "mpc.total_horizon=80",
            "mpc.fixed_horizon=40",
            "mpc.first_plan_iters=2",
            "mpc.max_iter_per_plan=2",
            "mpc.near_plan_iters=2",
        ]
    )
    assert overrides is not None
    parser = v12._build_parser()
    argv = [
        "--robot-config", str(_R1D_RIGHT_PROFILE),
        "--seed", "42",
        "--no-plot",
        "--output", str(tmp_path),
    ]
    args = parser.parse_args(argv)
    args.task_overrides = overrides
    explicit = v12._explicit_cli_actions(parser, argv)
    config = v12.resolve_v12_config(args, explicit)

    result = run_v12_episode(config)

    assert result.robot_id == "unitree_r1d"
    assert len(result.joint_names) == 7
    assert all(name.startswith("right_") for name in result.joint_names)
    _assert_arm_shapes(result, 7)
    # 球自由关节地址来自 bundle（R1D 为 qpos 0-6 / qvel 0-5），不得截断为 RM65 布局
    assert result.metrics["ball_qpos_start"] == 0
    assert result.metrics["ball_qvel_start"] == 0
    assert result.hit_type in {"active", "passive", "miss"}
    # R1D 接近半径来自 bundle 球拍档案（blade100 横向半宽 0.123）
    proximity = result.metrics["proximity"]
    assert proximity["racket_radius"] == pytest.approx(0.123)
    assert proximity["ball_near_threshold"] == pytest.approx(
        0.033 + 0.123 + 0.03,
    )
    # 肩点必须是模型事实（right_shoulder_pitch_Link 的 world xpos），
    # 不得再用 target_center 顶替，也不得复用 RM65 常量。
    expected_shoulder = v12_episode_module._model_shoulder_pos(build_v12_robot(config))
    reported = np.asarray(result.metrics["shoulder_pos"], dtype=np.float64)
    assert np.allclose(reported, expected_shoulder, atol=1e-9, rtol=0.0)
    assert result.metrics["shoulder_pos_source"] == (
        "model:right_shoulder_pitch_Link"
    )
    assert not np.allclose(
        reported, np.asarray(config.target_center, dtype=np.float64), atol=1e-6,
    )


# ── 2b. R1D overlay 接受 V12 共享 CLI 旗标且真正进入配置 ────────────────
def test_r1d_overlay_shared_cli_flags_enter_config() -> None:
    """serve-box/ball-speed/no-bounce/position-mode 进入 ResolvedV12Config。"""
    _, torque_cfg = _parse_and_resolve(
        [
            "--robot-config", str(_R1D_RIGHT_PROFILE),
            "--seed", "42",
            "--no-plot",
            "--serve-box",
            "--ball-speed", "7",
            "--no-bounce",
        ]
    )
    assert torque_cfg.robot_id == "unitree_r1d"
    # 缺省仍为 torque；serve-box/ball-speed/no-bounce 必须真正生效
    assert torque_cfg.actuator_mode == "torque"
    assert torque_cfg.mpc_config["is_position_mode"] is False
    assert torque_cfg.serve_config["mode"] == "serve_box"
    assert torque_cfg.serve_config["ball_speed"] == pytest.approx(7.0)
    assert torque_cfg.scene_config["use_bounce"] is False

    _, position_cfg = _parse_and_resolve(
        [
            "--robot-config", str(_R1D_RIGHT_PROFILE),
            "--seed", "42",
            "--no-plot",
            "--position-mode",
        ]
    )
    # 显式 position 优先；R1D 的 kp/kd 由 bundle 从 runtime 参数派生（T3 约定）
    assert position_cfg.actuator_mode == "position"
    assert position_cfg.mpc_config["is_position_mode"] is True


def test_r1d_shared_flags_not_in_rejected_set() -> None:
    """共享旗标不再被 _collect_rejected_rm65_flags 当作 RM65 专属参数。"""
    parser = v12._build_parser()
    argv = [
        "--robot-config", str(_R1D_RIGHT_PROFILE),
        "--runtime-config", str(_R1D_RIGHT_RUNTIME),
        "--serve-box",
        "--ball-speed", "7",
        "--no-bounce",
        "--position-mode",
        "--async-replan",
        "--seed", "42",
    ]
    rejected = v12._collect_rejected_rm65_flags(parser, argv)
    for option in (
        "--serve-box", "--ball-speed", "--no-bounce", "--position-mode",
        "--async-replan",
    ):
        assert option not in rejected


def test_r1d_async_replan_is_shared_capability() -> None:
    """R1D 异步重规划是共享能力（T4/T5 实现），CLI 不得当作 RM65 专属参数拒绝。"""
    assert "async_replan" in v12._R1D_SHARED_FLAGS
    parser = v12._build_parser()
    argv = [
        "--robot-config", str(_R1D_RIGHT_PROFILE),
        "--runtime-config", str(_R1D_RIGHT_RUNTIME),
        "--serve-box",
        "--ball-speed", "7",
        "--async-replan",
        "--seed", "42",
    ]
    assert "--async-replan" not in v12._collect_rejected_rm65_flags(parser, argv)


def test_r1d_overlay_cli_two_modes_write_legacy_artifacts(tmp_path: Path) -> None:
    """R1D overlay CLI 两模式各跑一个短回合并写旧 r1d_sim_episode_v1 产物。"""
    script = SCRIPTS_DIR / "rm65_mpc_v12.py"
    base_argv = [
        "--robot-config", str(_R1D_RIGHT_PROFILE),
        "--seed", "42",
        "--no-plot",
        "--task-overrides", "mpc.total_horizon=80",
        "--task-overrides", "mpc.fixed_horizon=40",
        "--task-overrides", "mpc.first_plan_iters=2",
        "--task-overrides", "mpc.max_iter_per_plan=2",
        "--task-overrides", "mpc.near_plan_iters=2",
    ]
    for mode, extra, expected_nu in (
        ("torque", [], 7),
        ("position", ["--position-mode"], 7),
    ):
        out_dir = tmp_path / f"r1d_{mode}"
        cmd = [
            sys.executable, str(script), *base_argv,
            *extra, "--output", str(out_dir),
        ]
        proc = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False, env=_UTF8_ENV,
        )
        assert proc.returncode == 0, proc.stderr[-4000:]
        assert "__R1D_RESULT__:" in proc.stdout
        payload = json.loads(
            (out_dir / "r1d_episode.json").read_text(encoding="utf-8")
        )
        assert payload["schema"] == "r1d_sim_episode_v1"
        assert payload["mode"] == mode
        assert len(payload["joint_names"]) == 7
        assert payload["hit"]["type"] in {"active", "passive", "miss"}
        assert payload["episode"]["failure_reason"] in {
            None, "safety_filter_stop", "ball_unreachable",
        }
        with np.load(out_dir / "r1d_episode.npz") as npz:
            assert npz["X"].shape[1] == 14
            assert npz["U"].shape[1] == expected_nu
            assert npz["U"].shape[0] == npz["X"].shape[0] - 1


# ── 3. RM65 输出保真 + T0 基线逐位对照 ────────────────────────────────
def _compare_npz_bitwise(left: Path, right: Path) -> None:
    """逐键比较两个轨迹 NPZ（键集合/形状/dtype/数值完全一致）。"""
    with np.load(left) as a, np.load(right) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            assert a[key].shape == b[key].shape, f"{key} 形状不一致"
            assert a[key].dtype == b[key].dtype, f"{key} dtype 不一致"
            assert np.array_equal(a[key], b[key]), f"{key} 数值不一致"


def test_rm65_torque_matches_t0_baseline(tmp_path: Path) -> None:
    """T0 基线命令（torque seed42）经新管线复跑后逐位一致。"""
    if not _T0_TORQUE_BASELINE.is_file():
        # results/ 被 gitignore：干净 clone 无 T0 基线产物时必须显式 skip，
        # 而不是收集期静默跳过（复审 B2 非阻断项）。
        pytest.skip("T0 baseline artifacts not present")
    out = tmp_path / "rm65_torque_new.npz"
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "rm65_mpc_v12.py"),
        "--serve-box", "--ball-speed", "7", "--seed", "42",
        "--no-plot", "--dump-trajectory", str(out),
    ]
    proc = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False, env=_UTF8_ENV,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    stdout = proc.stdout
    # 旧打印块与 __RESULT__ 语义保持不变
    assert "========================================" in stdout
    assert "MPC 总步数:" in stdout
    assert "__RESULT__:" in stdout
    result_line = stdout.split("__RESULT__:", 1)[1].strip().splitlines()[0]
    parsed = dict(part.split("=", 1) for part in result_line.split())
    expected_keys = {
        "pos_error", "vel_error", "min_dist", "ball_near_ms", "tube_ready_ms",
        "max_tcp", "max_qdot", "max_face", "hit_type", "hit_time_error_ms",
        "hit_pos_error", "v_racket_at_hit", "first_plan_ms",
        "ball_out_speed", "ball_out_vy",
    }
    assert expected_keys.issubset(parsed.keys())
    assert out.is_file()
    _compare_npz_bitwise(_T0_TORQUE_BASELINE, out)


_T0_POSITION_BASELINE = (
    ROOT / "results" / "r1d_v12_parity" / "baseline" / "rm65_position_a.npz"
)


def test_rm65_position_matches_t0_baseline(tmp_path: Path) -> None:
    """T0 基线命令（position seed42）经新管线复跑后逐位一致。"""
    if not _T0_POSITION_BASELINE.is_file():
        pytest.skip("T0 baseline artifacts not present")
    out = tmp_path / "rm65_position_new.npz"
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "rm65_mpc_v12.py"),
        "--serve-box", "--ball-speed", "7", "--seed", "42",
        "--no-plot", "--position-mode", "--dump-trajectory", str(out),
    ]
    proc = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False, env=_UTF8_ENV,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert "__RESULT__:" in proc.stdout
    assert out.is_file()
    _compare_npz_bitwise(_T0_POSITION_BASELINE, out)


# ── 4. 旧 run_r1d_episode 产物兼容：mappingproxy 回归 ─────────────────
def test_r1d_legacy_scene_params_json_serializable() -> None:
    """R1 回归：fixed_scene_params 可 JSON 序列化且 fixed_joint_poses 保留。"""
    profile = load_robot_profile(_R1D_RIGHT_PROFILE)
    runtime = load_runtime_parameters(_R1D_RIGHT_RUNTIME, profile)
    scene = runtime.scene
    # 构造非空固定关节位姿：T2 用 MappingProxyType 表达只读，
    # dataclasses.asdict 会因 deepcopy 抛 TypeError（T4 报告 R1）。
    scene_with_poses = replace(
        scene,
        fixed_joint_poses={
            "right_shoulder_pitch_joint": 0.123,
            "right_shoulder_roll_joint": -0.25,
        },
    )
    assert type(scene_with_poses.fixed_joint_poses).__name__ == "mappingproxy"

    payload = _scene_to_jsonable(scene_with_poses)
    dumped = json.dumps(payload, ensure_ascii=False)
    assert isinstance(dumped, str)
    # R1D 仿真运行参数的 racket_radius 允许暂缺（T2 未知限值显式 null），
    # 但键必须存在且 JSON 化后为 None/float（不得出现 MappingProxyType）
    assert "racket_radius" in payload
    assert payload["racket_radius"] is None or isinstance(
        payload["racket_radius"], float
    )
    poses = payload["fixed_joint_poses"]
    assert isinstance(poses, dict)
    # 键值必须保留（T2 用 MappingProxyType 表达只读，不得丢失/变成字符串）
    assert poses == {
        "right_shoulder_pitch_joint": pytest.approx(0.123),
        "right_shoulder_roll_joint": pytest.approx(-0.25),
    }
    for key, value in poses.items():
        assert isinstance(key, str)
        assert isinstance(value, float)


# ── 5. SimComponent 接近半径参数化（缺省表达式逐位保留）─────────────
def test_sim_component_proximity_defaults() -> None:
    """RM65 缺省半径与旧硬编码表达式一致；参数化入口存在且新值生效。"""
    import inspect

    sig = inspect.signature(sim_component_module.SimComponent.__init__)
    assert sig.parameters["ball_radius"].default == pytest.approx(0.033)
    assert sig.parameters["racket_radius"].default == pytest.approx(0.12)
    assert sig.parameters["proximity_margin"].default == pytest.approx(0.03)
    # 旧执行判据 dist < 0.033 + 0.12 + 0.03（相加次序不变）
    assert sim_component_module.DEFAULT_BALL_RADIUS == pytest.approx(0.033)
    assert sim_component_module.DEFAULT_RACKET_RADIUS == pytest.approx(0.12)
    assert sim_component_module.DEFAULT_PROXIMITY_MARGIN == pytest.approx(0.03)


# ── 6. 发球 RNG 顺序与机器人分支无关 ─────────────────────────────────
def test_serve_rng_order_robot_independent() -> None:
    """同 seed 下直接按配置重算的发球初态与管线结果完全一致。"""
    from src.tennis.ball import generate_ball_from_serve_box

    _, config = _parse_and_resolve(_RM65_SHORT_ARGV)
    result = run_v12_episode(config)

    scene = dict(config.scene_config)
    serve = dict(config.serve_config)
    shoulder = np.asarray(scene["shoulder_pos"], dtype=np.float64)
    rng = np.random.default_rng(int(config.seed))
    p0, v0, _ = generate_ball_from_serve_box(
        serve_box_center=tuple(float(x) for x in serve["serve_box_center"]),
        serve_box_halfsize=tuple(float(x) for x in serve["serve_box_halfsize"]),
        target_center=np.asarray(config.target_center, dtype=np.float64),
        target_offset=float(config.target_offset),
        shoulder_pos=shoulder,
        workspace_radius=float(config.workspace_radius),
        g=scene.get("ball_gravity"),
        ball_speed=serve.get("ball_speed"),
        speed_range=tuple(float(x) for x in serve["speed_range"]),
        use_bounce=bool(scene.get("use_bounce", True)),
        bounce_restitution=float(scene.get("bounce_restitution", 0.75)),
        rng=rng,
    )
    served = result.metrics["serve"]
    assert np.array_equal(np.asarray(served["p0"]), np.asarray(p0))
    assert np.array_equal(np.asarray(served["v0"]), np.asarray(v0))
    # RM65 的发球框位置/尺寸必须是 V12 常量（T0 逐位不变）
    assert v12_episode_module._resolve_serve_box_center(
        "rm65", serve, np.asarray(config.target_center, dtype=np.float64),
    ) == tuple(float(x) for x in serve["serve_box_center"])
    assert v12_episode_module._resolve_serve_box_halfsize("rm65", serve) == (
        tuple(float(x) for x in serve["serve_box_halfsize"])
    )


# ── 6b. R1D 发球几何：球从 +X 侧沿 -X 飞向目标（朝向 +X）─────────────
def test_r1d_serve_box_geometry_points_along_negative_x() -> None:
    """R1D 发球区旋转 90°：p0.x > target.x、v0.x < 0，目标仍在 target 附近。"""
    from src.tennis.ball import generate_ball_from_serve_box

    _, config = _parse_and_resolve(
        [
            "--robot-config", str(_R1D_RIGHT_PROFILE),
            "--seed", "42",
            "--no-plot",
            "--serve-box",
            "--ball-speed", "7",
        ]
    )
    serve = dict(config.serve_config)
    target = np.asarray(config.target_center, dtype=np.float64)
    box_center = v12_episode_module._resolve_serve_box_center(
        "unitree_r1d", serve, target,
    )
    box_halfsize = v12_episode_module._resolve_serve_box_halfsize(
        "unitree_r1d", serve,
    )
    assert box_center == pytest.approx(
        (target[0] + float(serve["serve_distance"]), target[1], float(serve["serve_height"]))
    )
    # 8m 长轴随来球方向旋转到 Y；RM65 的 X 长轴值不得被原地改写
    assert box_halfsize == pytest.approx((0.1, 4.0, 0.15))
    assert tuple(float(x) for x in serve["serve_box_halfsize"]) == pytest.approx(
        (4.0, 0.1, 0.15)
    )

    bundle = build_v12_robot(config)
    bundle.reset()
    shoulder = v12_episode_module._model_shoulder_pos(bundle)
    rng = np.random.default_rng(int(config.seed))
    p0, v0, p_hit = generate_ball_from_serve_box(
        serve_box_center=box_center,
        serve_box_halfsize=box_halfsize,
        target_center=target,
        target_offset=float(config.target_offset),
        shoulder_pos=shoulder,
        workspace_radius=float(config.workspace_radius),
        g=dict(config.scene_config).get("ball_gravity"),
        ball_speed=serve.get("ball_speed"),
        speed_range=tuple(float(x) for x in serve["speed_range"]),
        use_bounce=bool(dict(config.scene_config).get("use_bounce", True)),
        bounce_restitution=float(dict(config.scene_config).get("bounce_restitution", 0.75)),
        rng=rng,
    )
    p0 = np.asarray(p0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    p_hit = np.asarray(p_hit, dtype=np.float64)
    # 球从 +X 侧出发、沿 -X 飞向目标，而不是沿 -Y 侧向掠过
    assert p0[0] > target[0]
    assert v0[0] < 0.0
    assert abs(v0[0]) > abs(v0[1]), f"R1D 来球应以 -X 为主方向，实得 v0={v0}"
    assert np.linalg.norm(p_hit - target) < 1.0


# ── 7. MPCConfig 未知键早失败 ─────────────────────────────────────────
def test_mpc_config_unknown_key_fails() -> None:
    """``mpc_config`` 出现未知键时在构造控制器前早失败。"""
    _, config = _parse_and_resolve(_RM65_SHORT_ARGV)
    bad_mpc = dict(config.mpc_config)
    bad_mpc["totally_unknown_key"] = 1
    bad_config = replace(config, mpc_config=bad_mpc)

    with pytest.raises(ValueError, match="totally_unknown_key"):
        run_v12_episode(bad_config)


def test_v12_episode_config_type_guard() -> None:
    """run_v12_episode 只接受 ResolvedV12Config。"""
    with pytest.raises(TypeError, match="ResolvedV12Config"):
        run_v12_episode(object())  # type: ignore[arg-type]
