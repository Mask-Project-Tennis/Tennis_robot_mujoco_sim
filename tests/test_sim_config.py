"""R1D 仿真任务配置（SimTaskConfig）与严格仿真预检测试。

覆盖（Task 3.1）：
- task 配置加载：字段逐项、schema version、未知/缺失字段拒绝、
  robot_id/joint_names 与 profile 侧别校验（不靠文件名猜测）、
  数组形状/NaN/负数/max_steps 拒绝、actuator_mode 字面量校验；
- 数据类不可变契约：数组复制只读、不与输入别名、dict 浅拷贝；
- 仿真 runtime：stage2 null 限值不能用于 simulation（先模型回填
  q_lower/q_upper，再报缺失动态限值）、长度/NaN/负数/未知键/身份
  不一致拒绝；
- 位置模式缺 kp/kd 拒绝、提供 kp/kd 通过；
- 双侧 happy path：load_runtime_parameters + load_sim_task_config +
  validate_simulation_inputs 全链通过。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import yaml

from src.robot.robot_profile import RobotProfile, load_robot_profile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    load_runtime_parameters,
)
from src.sim.sim_config import (
    SimTaskConfig,
    load_sim_task_config,
    validate_simulation_inputs,
)

# 仓库根目录（tests/ 的父目录）
REPO_ROOT = Path(__file__).resolve().parent.parent

# 双侧描述/仿真参数/任务配置文件（Task 3.1 交付物）
_PROFILE_YAML = {
    "left": REPO_ROOT / "configs" / "robots" / "r1d_left.yaml",
    "right": REPO_ROOT / "configs" / "robots" / "r1d_right.yaml",
}
_SIM_RUNTIME_YAML = {
    "left": REPO_ROOT / "configs" / "robots" / "r1d_sim_runtime_left.yaml",
    "right": REPO_ROOT / "configs" / "robots" / "r1d_sim_runtime_right.yaml",
}
_TASK_YAML = {
    "left": REPO_ROOT / "configs" / "r1d_sim_left.yaml",
    "right": REPO_ROOT / "configs" / "r1d_sim_right.yaml",
}

# 阶段 2 runtime（未知限值 null，不得用于 simulation）
_STAGE2_RUNTIME_YAML = {
    "left": REPO_ROOT / "configs" / "robots" / "r1d_runtime_left.yaml",
    "right": REPO_ROOT / "configs" / "robots" / "r1d_runtime_right.yaml",
}

# 双侧 7 关节名（URDF 原名，与 r1d_{side}.yaml active_joint_names 同序）
_JOINT_NAMES = {
    "left": (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ),
    "right": (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
}


# ---------- 加载 helper（不可变对象，进程内缓存避免重复解析） ----------


def _profile(side: str) -> RobotProfile:
    """加载并缓存该侧 RobotProfile。"""
    return load_robot_profile(_PROFILE_YAML[side])


def _task(side: str) -> SimTaskConfig:
    """加载并缓存该侧 SimTaskConfig。"""
    return load_sim_task_config(_TASK_YAML[side], _profile(side))


def _sim_runtime(side: str) -> RobotRuntimeParameters:
    """加载该侧仿真 runtime 参数。"""
    return load_runtime_parameters(_SIM_RUNTIME_YAML[side], _profile(side))


def _modified_copy(
    source: Path, tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> Path:
    """复制 YAML 到临时文件并应用单点修改（负例构造器）。

    Args:
        source: 原 YAML 文件路径。
        tmp_path: pytest 临时目录。
        mutate: 对解析后字典的原地修改函数。

    Returns:
        修改后副本的路径。
    """
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    mutate(raw)
    target = tmp_path / f"modified_{source.name}"
    target.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return target


# ---------- task 配置：加载与字段校验 ----------


@pytest.mark.parametrize("side", ["left", "right"])
def test_task_config_happy_path(side: str) -> None:
    """双侧 task 配置加载成功，字段逐项与契约一致。"""
    task = _task(side)
    assert task.robot_id == "unitree_r1d"
    assert task.joint_names == _JOINT_NAMES[side]
    assert isinstance(task.ball_joint_name, str) and task.ball_joint_name
    assert isinstance(task.ball_geom_name, str) and task.ball_geom_name
    assert isinstance(task.racket_geom_names, tuple)
    assert len(task.racket_geom_names) >= 1
    assert all(
        isinstance(name, str) and name for name in task.racket_geom_names
    )
    # 保护对：每对是两个非空字符串，且覆盖球拍—机身禁止对
    assert isinstance(task.protected_geom_pairs, tuple)
    assert len(task.protected_geom_pairs) >= 1
    for pair in task.protected_geom_pairs:
        assert isinstance(pair, tuple) and len(pair) == 2
        assert all(isinstance(name, str) and name for name in pair)
    flat = {name for pair in task.protected_geom_pairs for name in pair}
    assert "torso_box" in flat
    assert "racket_face" in flat
    # 数值字段
    assert task.workspace_center.shape == (3,)
    assert task.workspace_radius > 0
    assert task.initial_ball_position.shape == (3,)
    assert task.initial_ball_velocity.shape == (3,)
    assert isinstance(task.max_steps, int) and task.max_steps > 0
    assert task.actuator_mode in ("torque", "position")
    # 三个透传配置为 dict
    for config in (task.solver_config, task.mpc_config, task.cost_config):
        assert isinstance(config, dict)


def test_task_arrays_copied_readonly_and_not_aliased() -> None:
    """手构 SimTaskConfig：数组复制为只读、不与输入别名。"""
    task = _task("left")
    center = np.array([0.25, 0.25, 1.35])
    ball_pos = np.array([2.5, 0.45, 2.8])
    ball_vel = np.array([-5.0, -0.3, -1.0])
    hand = SimTaskConfig(
        robot_id=task.robot_id,
        joint_names=task.joint_names,
        ball_joint_name=task.ball_joint_name,
        ball_geom_name=task.ball_geom_name,
        racket_geom_names=task.racket_geom_names,
        protected_geom_pairs=task.protected_geom_pairs,
        workspace_center=center,
        workspace_radius=0.45,
        initial_ball_position=ball_pos,
        initial_ball_velocity=ball_vel,
        max_steps=600,
        actuator_mode="torque",
        solver_config={"max_iter": 10},
        mpc_config={"dt": 0.005},
        cost_config={"Q_p": 2000.0},
    )
    # 输入数组事后修改不影响内部副本（不别名）
    center[0] = 99.0
    ball_pos[0] = 99.0
    ball_vel[0] = 99.0
    assert hand.workspace_center[0] == pytest.approx(0.25)
    assert hand.initial_ball_position[0] == pytest.approx(2.5)
    assert hand.initial_ball_velocity[0] == pytest.approx(-5.0)
    # 内部数组只读
    for array in (
        hand.workspace_center,
        hand.initial_ball_position,
        hand.initial_ball_velocity,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array[0] = 1.0


def test_task_config_dicts_not_aliased() -> None:
    """solver/mpc/cost dict 浅拷贝：输入事后修改不影响内部值。"""
    task = _task("left")
    solver = {"max_iter": 10}
    hand = SimTaskConfig(
        robot_id=task.robot_id,
        joint_names=task.joint_names,
        ball_joint_name=task.ball_joint_name,
        ball_geom_name=task.ball_geom_name,
        racket_geom_names=task.racket_geom_names,
        protected_geom_pairs=task.protected_geom_pairs,
        workspace_center=np.zeros(3),
        workspace_radius=0.45,
        initial_ball_position=np.zeros(3),
        initial_ball_velocity=np.zeros(3),
        max_steps=600,
        actuator_mode="torque",
        solver_config=solver,
        mpc_config={},
        cost_config={},
    )
    solver["max_iter"] = 999
    assert hand.solver_config["max_iter"] == 10


def test_task_unknown_top_level_key_rejected(tmp_path: Path) -> None:
    """task YAML 顶层未知键拒绝（防拼写错误静默生效）。"""
    path = _modified_copy(
        _TASK_YAML["left"], tmp_path, lambda raw: raw.update(typo_field=1)
    )
    with pytest.raises(ValueError, match="未知字段"):
        load_sim_task_config(path, _profile("left"))


def test_task_missing_field_rejected(tmp_path: Path) -> None:
    """task YAML 缺少契约字段拒绝。"""
    path = _modified_copy(
        _TASK_YAML["left"], tmp_path, lambda raw: raw.pop("max_steps")
    )
    with pytest.raises(ValueError, match="缺少字段 max_steps"):
        load_sim_task_config(path, _profile("left"))


def test_task_bad_schema_version_rejected(tmp_path: Path) -> None:
    """task YAML schema_version 非 1 拒绝。"""
    path = _modified_copy(
        _TASK_YAML["left"], tmp_path, lambda raw: raw.update(schema_version=2)
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_sim_task_config(path, _profile("left"))


def test_task_robot_id_mismatch_rejected(tmp_path: Path) -> None:
    """task robot_id 与 profile 不一致拒绝（不用文件名猜测侧别）。"""
    path = _modified_copy(
        _TASK_YAML["left"], tmp_path, lambda raw: raw.update(robot_id="rm65")
    )
    with pytest.raises(ValueError, match="robot_id"):
        load_sim_task_config(path, _profile("left"))


def test_task_side_mismatch_rejected_by_joint_names() -> None:
    """左 task 配置 + 右 profile：joint_names 不一致拒绝（侧别校验）。"""
    with pytest.raises(ValueError, match="joint_names"):
        load_sim_task_config(_TASK_YAML["left"], _profile("right"))


def test_task_joint_order_mismatch_rejected(tmp_path: Path) -> None:
    """task joint_names 顺序交换后拒绝（须逐项同序）。"""
    def mutate(raw: dict[str, Any]) -> None:
        names = raw["joint_names"]
        names[0], names[1] = names[1], names[0]

    path = _modified_copy(_TASK_YAML["left"], tmp_path, mutate)
    with pytest.raises(ValueError, match="joint_names"):
        load_sim_task_config(path, _profile("left"))


def test_task_joint_count_mismatch_rejected(tmp_path: Path) -> None:
    """task joint_names 缺一个（6/7）拒绝。"""
    def mutate(raw: dict[str, Any]) -> None:
        raw["joint_names"] = raw["joint_names"][:-1]

    path = _modified_copy(_TASK_YAML["left"], tmp_path, mutate)
    with pytest.raises(ValueError, match="joint_names"):
        load_sim_task_config(path, _profile("left"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workspace_center", [0.25, 0.25]),          # 维数错误（2）
        ("initial_ball_position", [1.0, 2.0, 3.0, 4.0]),  # 维数错误（4）
        ("initial_ball_velocity", [-5.0, -0.3]),      # 维数错误（2）
        ("workspace_center", [0.25, float("nan"), 1.35]),  # NaN
        ("initial_ball_position", [2.5, 0.45, float("nan")]),  # NaN
        ("initial_ball_velocity", [float("nan"), -0.3, -1.0]),  # NaN
    ],
)
def test_task_bad_vector_fields_rejected(
    tmp_path: Path, field: str, value: list[float]
) -> None:
    """task 向量字段维数错误或 NaN 拒绝。"""
    path = _modified_copy(
        _TASK_YAML["left"], tmp_path, lambda raw: raw.update({field: value})
    )
    with pytest.raises(ValueError, match=field):
        load_sim_task_config(path, _profile("left"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workspace_radius", 0.0),
        ("workspace_radius", -0.45),
        ("workspace_radius", float("nan")),
        ("max_steps", 0),
        ("max_steps", -10),
    ],
)
def test_task_nonpositive_scalars_rejected(
    tmp_path: Path, field: str, value: Any
) -> None:
    """task 正数标量字段（半径/步数）非正或 NaN 拒绝。"""
    path = _modified_copy(
        _TASK_YAML["left"], tmp_path, lambda raw: raw.update({field: value})
    )
    with pytest.raises(ValueError, match=field):
        load_sim_task_config(path, _profile("left"))


def test_task_bad_actuator_mode_rejected(tmp_path: Path) -> None:
    """actuator_mode 非 torque/position 字面量拒绝。"""
    path = _modified_copy(
        _TASK_YAML["left"],
        tmp_path,
        lambda raw: raw.update(actuator_mode="velocity"),
    )
    with pytest.raises(ValueError, match="actuator_mode"):
        load_sim_task_config(path, _profile("left"))


def test_task_config_section_must_be_mapping(tmp_path: Path) -> None:
    """solver_config 等配置节必须是映射结构（列表拒绝）。"""
    path = _modified_copy(
        _TASK_YAML["left"],
        tmp_path,
        lambda raw: raw.update(solver_config=[1, 2, 3]),
    )
    with pytest.raises(ValueError, match="solver_config"):
        load_sim_task_config(path, _profile("left"))


# ---------- 仿真 runtime：null 拒绝与数值校验 ----------


@pytest.mark.parametrize("side", ["left", "right"])
def test_stage2_runtime_null_rejected_for_simulation(side: str) -> None:
    """stage2 runtime 的 null 限值不能进入 simulation 预检。"""
    params = load_runtime_parameters(_STAGE2_RUNTIME_YAML[side], _profile(side))
    with pytest.raises(ValueError) as excinfo:
        validate_simulation_inputs(_profile(side), params, _task(side))
    message = str(excinfo.value)
    # 动态限值缺失必须点名（q_lower/q_upper 已由模型回填，不在此列）
    for field in ("qdot_max", "qddot_max", "torque_max", "max_tcp_speed"):
        assert field in message
    assert "q_lower" not in message


def test_sim_runtime_missing_qdot_reports_after_backfill(
    tmp_path: Path,
) -> None:
    """qdot_max 置 null：先模型回填 q 限位，再只报动态限值缺失。"""
    path = _modified_copy(
        _SIM_RUNTIME_YAML["left"],
        tmp_path,
        lambda raw: raw.update(qdot_max=None),
    )
    params = load_runtime_parameters(path, _profile("left"))
    with pytest.raises(ValueError) as excinfo:
        validate_simulation_inputs(_profile("left"), params, _task("left"))
    message = str(excinfo.value)
    assert "qdot_max" in message
    assert "q_lower" not in message  # 已回填，不作为缺失项报告


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("init_q", [0.0, 0.2, 0.0, -0.6, 0.0, 0.0]),            # 6 维
        ("init_q", [0.0] * 8),                                   # 8 维
        ("qdot_max", [18.8, 18.8, 33.4, 33.4, 33.4, 37.7]),      # 6 维
        ("torque_max", [60, 60, 33, 33, 33, 10, 10, 10]),        # 8 维
        ("kp", [400.0, 400.0, 250.0]),                           # 3 维
        ("init_q", [0.0, float("nan"), 0.0, -0.6, 0.0, 0.0, 0.0]),  # NaN
        ("qddot_max", [125, 125, 220, 220, float("nan"), 250, 250]),  # NaN
        ("qdot_max", [18.8, -18.8, 33.4, 33.4, 33.4, 37.7, 37.7]),  # 负数
        ("torque_max", [60, 60, 33, 0, 33, 10, 10]),             # 零拒绝
        ("max_tcp_speed", 0.0),                                   # 非正
        ("max_tcp_speed", -1.2),                                  # 负数
        ("kp", [400, 400, -250, 120, 60, 40, 40]),                # kp 负数
    ],
)
def test_sim_runtime_bad_values_rejected(
    tmp_path: Path, field: str, value: Any
) -> None:
    """仿真 runtime 数值校验：长度/NaN/负数/零拒绝。"""
    path = _modified_copy(
        _SIM_RUNTIME_YAML["left"],
        tmp_path,
        lambda raw: raw.update({field: value}),
    )
    with pytest.raises(ValueError, match=field):
        load_runtime_parameters(path, _profile("left"))


def test_sim_runtime_unknown_key_rejected(tmp_path: Path) -> None:
    """仿真 runtime 未知键拒绝。"""
    path = _modified_copy(
        _SIM_RUNTIME_YAML["left"],
        tmp_path,
        lambda raw: raw.update(extra_limit=1.0),
    )
    with pytest.raises(ValueError, match="未知字段"):
        load_runtime_parameters(path, _profile("left"))


def test_sim_runtime_robot_id_mismatch_rejected(tmp_path: Path) -> None:
    """仿真 runtime robot_id 与 profile 不一致在加载时即拒绝。"""
    path = _modified_copy(
        _SIM_RUNTIME_YAML["left"],
        tmp_path,
        lambda raw: raw.update(robot_id="unitree_d2"),
    )
    # 阶段 2 加载器在 kinematics 级即校验身份，不一致不进入后续链路
    with pytest.raises(ValueError, match="robot_id"):
        load_runtime_parameters(path, _profile("left"))


def test_sim_runtime_side_mismatch_rejected() -> None:
    """左 sim runtime + 右 profile：joint_names 不一致拒绝。"""
    params = load_runtime_parameters(_SIM_RUNTIME_YAML["left"], _profile("left"))
    with pytest.raises(ValueError, match="joint_names"):
        validate_simulation_inputs(_profile("right"), params, _task("right"))


def test_explicit_q_limits_must_match_model(tmp_path: Path) -> None:
    """显式 q_lower 与模型 jnt_range 不一致拒绝（模型为唯一权威）。"""
    def mutate(raw: dict[str, Any]) -> None:
        raw["q_lower"] = [-3.0] * 7
        raw["q_upper"] = [3.0] * 7

    path = _modified_copy(_SIM_RUNTIME_YAML["left"], tmp_path, mutate)
    params = load_runtime_parameters(path, _profile("left"))
    with pytest.raises(ValueError, match="q_lower"):
        validate_simulation_inputs(_profile("left"), params, _task("left"))


# ---------- 位置模式 kp/kd 预检 ----------


def test_position_mode_requires_kp_kd(tmp_path: Path) -> None:
    """actuator_mode=position 而 runtime 缺 kp/kd：预检拒绝。"""
    runtime_path = _modified_copy(
        _SIM_RUNTIME_YAML["left"],
        tmp_path,
        lambda raw: (raw.update(kp=None, kd=None)),
    )
    params = load_runtime_parameters(runtime_path, _profile("left"))
    task_path = _modified_copy(
        _TASK_YAML["left"],
        tmp_path,
        lambda raw: raw.update(actuator_mode="position"),
    )
    task = load_sim_task_config(task_path, _profile("left"))
    with pytest.raises(ValueError) as excinfo:
        validate_simulation_inputs(_profile("left"), params, task)
    message = str(excinfo.value)
    assert "kp" in message and "kd" in message
    assert "position" in message


def test_position_mode_with_kp_kd_passes(tmp_path: Path) -> None:
    """actuator_mode=position 且仿真 runtime 提供 kp/kd：预检通过。"""
    task_path = _modified_copy(
        _TASK_YAML["left"],
        tmp_path,
        lambda raw: raw.update(actuator_mode="position"),
    )
    task = load_sim_task_config(task_path, _profile("left"))
    params = load_runtime_parameters(_SIM_RUNTIME_YAML["left"], _profile("left"))
    assert params.kp is not None and params.kd is not None
    # 不抛异常即通过（模型回填 + simulation 完整性 + kp/kd 存在）
    validate_simulation_inputs(_profile("left"), params, task)


# ---------- 双侧全链 happy path ----------


@pytest.mark.parametrize("side", ["left", "right"])
def test_simulation_preflight_happy_path(side: str) -> None:
    """双侧：sim runtime + task 配置全链通过严格仿真预检。"""
    profile = _profile(side)
    params = load_runtime_parameters(_SIM_RUNTIME_YAML[side], profile)
    task = _task(side)
    # 预检前动态限值已显式有限（q_lower/q_upper 仍 null，由模型回填）
    assert params.q_lower is None and params.q_upper is None
    for field in ("qdot_max", "qddot_max", "torque_max"):
        assert getattr(params, field) is not None
    validate_simulation_inputs(profile, params, task)  # 不抛异常即通过


@pytest.mark.parametrize("side", ["left", "right"])
def test_sim_runtime_values_match_documented_source(side: str) -> None:
    """仿真 runtime 数值与 URDF 声明一致（effort/velocity 逐项）。"""
    params = load_runtime_parameters(_SIM_RUNTIME_YAML[side], _profile(side))
    # URDF effort: 肩俯仰/侧摆 60，肩偏航/肘/腕滚转 33，腕俯仰/偏航 10
    assert params.torque_max is not None
    np.testing.assert_allclose(
        params.torque_max, [60.0, 60.0, 33.0, 33.0, 33.0, 10.0, 10.0]
    )
    # URDF velocity: 18.8 / 33.4 / 37.7（r1d.urdf 双侧同名声明）
    assert params.qdot_max is not None
    np.testing.assert_allclose(
        params.qdot_max, [18.8, 18.8, 33.4, 33.4, 33.4, 37.7, 37.7]
    )
