"""R1D 预览与可追溯参数导出工具测试（阶段 2 · Task 2.7）。

覆盖任务卡四组测试步骤：
1. ``test_headless_outputs_match_bundle``：无头运行（双侧各一次）输出的
   JSON 来自实际 bundle——side 字段、r1d 关节名、绑定地址、维数、初态、
   TCP 位姿、回填限位均与测试内重新组装的 bundle 逐项一致；**不得输出
   D2 报告冒充 R1-A7-D**（无 unitree_d2/joint_N 字样）。
2. ``test_config_error_exit_code``：缺文件/坏 YAML/身份不符等异常配置
   退出码 2；缺必传 CLI 参数由 argparse 以 SystemExit(2) 拒绝。
3. ``test_does_not_start_runtime_or_sdk``：运行前后 sys.modules 快照差
   不含 src.real / src.ilqt / src.sim / unitree SDK / mujoco.viewer——
   不触碰 MPC、真机与 SDK。
4. 渲染：mock Renderer 断言 front/side/oblique 三视图调用序列；渲染
   失败非零退出且**已生成 JSON 保留**；真渲染在本机 GL 可用时执行、
   不可用则优雅跳过（策略记录于 task-2.7 报告）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

# 统一导入模式：sys.path.insert + bare name（先例：test_inspect_trajectory.py）
REPO_ROOT = Path(__file__).resolve().parent.parent
_TOOLS_DIR = str(REPO_ROOT / "scripts" / "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from inspect_robot_model import (  # noqa: E402
    EXIT_CONFIG,
    _sdk_entry_for_joint,
    main,
    run_inspection,
)

from src.robot.model_factory import load_robot_model  # noqa: E402
from src.robot.model_kinematics import ModelKinematics  # noqa: E402
from src.robot.racket_profile import load_racket_profile  # noqa: E402
from src.robot.robot_profile import load_robot_profile  # noqa: E402
from src.robot.runtime_parameters import load_runtime_parameters  # noqa: E402

# 双侧配置路径（侧别经显式文件路径表达，无默认侧）
SIDES = ("left", "right")
PROFILE_PATHS = {
    side: REPO_ROOT / "configs" / "robots" / f"r1d_{side}.yaml" for side in SIDES
}
RUNTIME_PATHS = {
    side: REPO_ROOT / "configs" / "robots" / f"r1d_runtime_{side}.yaml"
    for side in SIDES
}
# V12 仿真运行配置（显式升降/固定姿态、无 legacy racket_* 字段）
V12_RUNTIME_PATH = REPO_ROOT / "configs" / "robots" / "r1d_v12_runtime_right.yaml"
# 球拍档案（T1 交付；唯一物理来源）
REFERENCE_RACKET = REPO_ROOT / "configs" / "rackets" / "blade100_v9_reference.yaml"
LEGACY_RACKET = REPO_ROOT / "configs" / "rackets" / "rm65_legacy.yaml"

# 渲染相关被禁模块前缀（运行期不得新增导入）
_FORBIDDEN_MODULE_PREFIXES = (
    "src.real",
    "src.ilqt",
    "src.sim",
    "unitree_sdk",
    "unitree_ros",
)


def _run_headless(side: str, output_dir: Path) -> tuple[int, Path]:
    """以 main() 无头运行指定侧，返回（退出码, JSON 路径）。"""
    rc = main(
        [
            "--robot-config",
            str(PROFILE_PATHS[side]),
            "--runtime-config",
            str(RUNTIME_PATHS[side]),
            "--output-dir",
            str(output_dir),
        ]
    )
    return rc, output_dir / f"inspect_r1d_{side}.json"


# ============================================================================
# 1. 无头输出与实际 bundle 逐项一致（双侧）
# ============================================================================


@pytest.mark.parametrize("side", SIDES)
def test_headless_outputs_match_bundle(side: str, tmp_path: Path) -> None:
    """JSON 报告来自实际 bundle：side/名称/地址/维数/初态/TCP/限位全对上。"""
    rc, json_path = _run_headless(side, tmp_path)
    assert rc == 0, f"无头检查应成功退出（side={side}）"
    assert json_path.is_file(), f"应写出 JSON 报告: {json_path}"

    # json.dumps(allow_nan=False) 往返：禁 NaN/Infinity、可序列化
    text = json_path.read_text(encoding="utf-8")
    report = json.loads(text)
    round_trip = json.dumps(report, ensure_ascii=False, allow_nan=False)
    assert isinstance(round_trip, str)

    # 测试内独立重新组装 bundle（同源入口），逐项对照
    profile = load_robot_profile(PROFILE_PATHS[side])
    parameters = load_runtime_parameters(RUNTIME_PATHS[side], profile)
    bundle = load_robot_model(profile, parameters)

    # --- 身份与侧别 ---
    assert report["robot_id"] == "unitree_r1d"
    assert report["side"] == side
    # 不得输出 D2 报告冒充 R1-A7-D
    assert "unitree_d2" not in text
    assert "d2_flange" not in text

    # --- 名称 ---
    assert report["profile"]["active_joint_names"] == list(
        profile.active_joint_names
    )
    assert report["profile"]["held_joint_names"] == list(profile.held_joint_names)
    assert report["profile"]["actuator_names"] == list(profile.actuator_names)
    # r1d 关节名：该侧 7 个 URDF 原名（left_/right_ 前缀）
    for name in report["profile"]["active_joint_names"]:
        assert name.startswith(f"{side}_") and name.endswith("_joint")

    # --- 维数 ---
    assert report["model"]["nq"] == bundle.model.nq == 7
    assert report["model"]["nv"] == bundle.model.nv == 7
    assert report["model"]["nu"] == bundle.model.nu == 7

    # --- 绑定地址（来自实际 ModelBinding）---
    binding = report["binding"]
    assert binding["qpos_indices"] == list(bundle.binding.qpos_indices)
    assert binding["qvel_indices"] == list(bundle.binding.qvel_indices)
    assert binding["actuator_ids"] == list(bundle.binding.actuator_ids)
    assert binding["joint_ids"] == list(bundle.binding.joint_ids)
    assert binding["held_qpos_indices"] == list(bundle.binding.held_qpos_indices)
    assert binding["tcp_site_id"] == bundle.binding.tcp_site_id
    per_joint_names = [row["name"] for row in binding["per_joint"]]
    assert per_joint_names == list(profile.active_joint_names)

    # --- 初态 ---
    assert report["initial_state"]["init_q"] == list(parameters.init_q)
    assert report["initial_state"]["held_q"] == list(parameters.held_q)
    assert report["initial_state"]["dt"] == parameters.dt

    # --- TCP 位姿：与 ModelKinematics 在同一初态下的结果一致 ---
    kin = ModelKinematics(bundle)
    kin.set_state(np.concatenate([parameters.init_q, np.zeros(bundle.model.nq)]))
    tcp_pos, tcp_rot = kin.tcp_pose()
    np.testing.assert_allclose(
        np.array(report["tcp_pose"]["position_xyz_m"]), tcp_pos, atol=1e-12
    )
    quat = np.array(report["tcp_pose"]["quaternion_wxyz"])
    expected_quat = np.empty(4)
    mujoco.mju_mat2Quat(expected_quat, tcp_rot.flatten())
    np.testing.assert_allclose(quat, expected_quat, atol=1e-12)

    # --- jnt_range 回填限位与未知项 ---
    limits = report["joint_limits"]
    np.testing.assert_allclose(
        np.array(limits["q_lower"]), bundle.parameters.q_lower, atol=1e-15
    )
    np.testing.assert_allclose(
        np.array(limits["q_upper"]), bundle.parameters.q_upper, atol=1e-15
    )
    assert limits["unknown_limits"]["qdot_max"] is None
    assert limits["unknown_limits"]["torque_max"] is None
    assert limits["unknown_limits"]["max_tcp_speed"] is None

    # --- provisional_fields ---
    assert report["provisional_fields"] == list(parameters.provisional_fields)

    # --- 冻结部件清单（六类）---
    frozen = report["frozen_components"]
    for key in ("wheels", "casters", "lift", "waist", "head", "opposite_arm"):
        assert key in frozen, f"冻结部件清单缺少 {key}"
    # 模板恒定 bake 默认 q_L=0.4225；旧运行配置未显式给 lift → 场景默认同值
    assert frozen["lift"]["q_L_m"] == pytest.approx(0.4225, abs=1e-12)
    assert frozen["lift"]["scene_lift_displacement_m"] == pytest.approx(
        0.4225, abs=1e-12
    )
    # T2 起 R1D 完整外观：轮/脚轮/头部/对侧臂都已进入场景
    assert frozen["wheels"]["handling"] == "landing_visual"
    assert frozen["casters"]["handling"] == "landing_visual"
    assert frozen["head"]["handling"] == "frozen_static_in_env_fragment"
    assert frozen["opposite_arm"]["handling"] == "frozen_static_body"
    assert frozen["waist"]["value_rad"] == 0.0

    # --- 球拍节：未传 --racket-config → 旧 API 兼容模式 ---
    racket = report["racket"]
    assert racket["mode"] == "legacy-template"
    assert racket["config_path"] is None and racket["racket_id"] is None
    assert racket["tcp_site"] == "racket_center"
    assert racket["active_joint_count"] == 7
    np.testing.assert_allclose(
        np.array(racket["tcp_local_position"]),
        bundle.model.site_pos[bundle.binding.tcp_site_id],
        atol=1e-15,
    )
    assert racket["legacy_body_mass"] == pytest.approx(0.30, abs=1e-12)

    # --- 数据冲突清单：审计报告 §5 四项原文引用 ---
    conflicts = report["data_conflicts"]
    items = conflicts["items"]
    assert len(items) == 4
    components = " ".join(item["component"] for item in items)
    assert "腰 WAIST_YAW" in components
    assert "左腕俯仰 L_WRIST_PITCH" in components
    assert "右腕俯仰 R_WRIST_PITCH" in components
    assert "左/右腕滚转 WRIST_ROLL" in components
    # 逐字引用核对：右腕俯仰 URDF 不对称 +0.5326、腰 ±2.618 等关键数值原文
    joined = json.dumps(items, ensure_ascii=False)
    assert "±0.6109" in joined and "±2.618" in joined
    assert "[-0.5236, 0.5326]" in joined
    assert "±2.0944" in joined and "±1.9199" in joined
    assert "只记录" in json.dumps(conflicts, ensure_ascii=False) or "不裁断" in json.dumps(
        conflicts, ensure_ascii=False
    )

    # --- SDK 整机索引对照（N0-8 标注）---
    sdk = report["sdk_index_reference"]
    assert "通信地址非 MuJoCo 状态地址" in sdk["annotation"]
    expected_indices = list(range(15, 22)) if side == "left" else list(range(22, 29))
    assert [row["sdk_index"] for row in sdk["selected_side_joints"]] == expected_indices
    assert [row["joint_name"] for row in sdk["selected_side_joints"]] == list(
        profile.active_joint_names
    )

    # --- 资源 SHA256 ---
    files = report["resource_hashes"]["files"]
    robot_cfg_key = next(k for k in files if k.replace("\\", "/").endswith(f"r1d_{side}.yaml"))
    digest = hashlib.sha256(PROFILE_PATHS[side].read_bytes()).hexdigest()
    assert files[robot_cfg_key] == digest
    assert report["resource_hashes"]["files"], "应至少含配置哈希"

    # --- 版本 ---
    assert report["versions"]["mujoco"] == mujoco.__version__


# ============================================================================
# 1b. --racket-config：档案模式报告（球拍唯一物理来源）
# ============================================================================


def test_racket_config_report_matches_profile(tmp_path: Path) -> None:
    """--racket-config 实际传给 load_robot_model：报告装配事实与档案/模型同源。"""
    rc = main(
        [
            "--robot-config",
            str(PROFILE_PATHS["right"]),
            "--runtime-config",
            str(V12_RUNTIME_PATH),
            "--racket-config",
            str(REFERENCE_RACKET),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    report = json.loads(
        (tmp_path / "inspect_r1d_right.json").read_text(encoding="utf-8")
    )
    racket = report["racket"]
    assert racket["mode"] == "profile"
    assert racket["racket_id"] == "blade100_v9_reference"
    assert racket["config_path"].replace("\\", "/") == (
        "configs/rackets/blade100_v9_reference.yaml"
    )
    assert racket["config_sha256"] == hashlib.sha256(
        REFERENCE_RACKET.read_bytes()
    ).hexdigest()
    assert racket["active_joint_count"] == 7
    assert racket["tcp_site"] == "racket_center"
    assert racket["model_path"].replace("\\", "/").endswith(
        "src/robot/r1d_model_right.xml"
    )

    # 测试内独立重装同一配置：报告数值必须与 bundle/模型一致（同一构建函数）
    profile = load_robot_profile(PROFILE_PATHS["right"])
    parameters = load_runtime_parameters(V12_RUNTIME_PATH, profile)
    racket_profile = load_racket_profile(REFERENCE_RACKET)
    bundle = load_robot_model(profile, parameters, racket_profile=racket_profile)
    assert bundle.racket is not None
    assert racket["total_mass"] == pytest.approx(bundle.racket.total_mass, abs=1e-15)
    np.testing.assert_allclose(np.array(racket["com"]), bundle.racket.com, atol=1e-15)
    np.testing.assert_allclose(
        np.array(racket["inertia_diagonal"]),
        bundle.racket.inertia_diagonal,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.array(racket["face_center"]), bundle.racket.face_center, atol=1e-15
    )
    np.testing.assert_allclose(
        np.array(racket["face_semi_axes"]), bundle.racket.face_semi_axes, atol=1e-15
    )
    np.testing.assert_allclose(
        np.array(racket["mount_position"]), bundle.racket.mount_position, atol=1e-15
    )
    np.testing.assert_allclose(
        np.array(racket["mount_wxyz"]), bundle.racket.mount_wxyz, atol=1e-15
    )
    assert racket["connector_mass"] == pytest.approx(
        racket_profile.connector.mass, abs=1e-15
    )
    # TCP site 局部位置 = 拍面几何中心（档案事实）
    np.testing.assert_allclose(
        np.array(racket["tcp_local_position"]), bundle.racket.face_center, atol=1e-15
    )
    # 报告 TCP 世界位姿 = 同源 bundle 的 kinematics 结果
    kin = ModelKinematics(bundle)
    kin.set_state(np.concatenate([parameters.init_q, np.zeros(bundle.model.nq)]))
    tcp_pos, _ = kin.tcp_pose()
    np.testing.assert_allclose(
        np.array(report["tcp_pose"]["position_xyz_m"]), tcp_pos, atol=1e-12
    )


def test_racket_config_hash_in_resource_hashes(tmp_path: Path) -> None:
    """档案 YAML 必须进入资源哈希（复现输入集完整性）。"""
    rc = main(
        [
            "--robot-config",
            str(PROFILE_PATHS["right"]),
            "--runtime-config",
            str(V12_RUNTIME_PATH),
            "--racket-config",
            str(REFERENCE_RACKET),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    report = json.loads(
        (tmp_path / "inspect_r1d_right.json").read_text(encoding="utf-8")
    )
    files = report["resource_hashes"]["files"]
    key = next(
        k for k in files if k.replace("\\", "/").endswith("blade100_v9_reference.yaml")
    )
    assert files[key] == hashlib.sha256(REFERENCE_RACKET.read_bytes()).hexdigest()
    # 复现命令包含 --racket-config，避免预览与运行模型不是同一配置
    assert "--racket-config" in report["invocation"]["reproduce_hint"]


def test_racket_config_legacy_profile_assembles(tmp_path: Path) -> None:
    """rm65_legacy 档案：报告质量/COM/安装来自旧拍档案（legacy_model 来源）。"""
    rc = main(
        [
            "--robot-config",
            str(PROFILE_PATHS["right"]),
            "--runtime-config",
            str(V12_RUNTIME_PATH),
            "--racket-config",
            str(LEGACY_RACKET),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    report = json.loads(
        (tmp_path / "inspect_r1d_right.json").read_text(encoding="utf-8")
    )
    racket = report["racket"]
    legacy = load_racket_profile(LEGACY_RACKET)
    assert racket["racket_id"] == "rm65_legacy"
    assert racket["total_mass"] == pytest.approx(legacy.mass, abs=1e-15)
    assert racket["connector_mass"] == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(np.array(racket["mount_position"]), legacy.mount.position)
    # 旧拍档案的 TCP 在 flange+0.25 m（RM65 原模型口径）
    np.testing.assert_allclose(
        np.array(racket["tcp_local_position"]), legacy.geometry.tcp, atol=1e-15
    )


# ============================================================================
# 1c. SDK 对照按完整关节名跟随（防关节顺序错位）
# ============================================================================

# 审计 §2 语义干顺序（肩俯仰/肩侧摆/肩偏航/肘/腕滚转/腕俯仰/腕偏航）
_SDK_STEMS = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)


@pytest.mark.parametrize("side", SIDES)
def test_sdk_index_follows_joint_name_not_order(
    side: str, tmp_path: Path
) -> None:
    """关节合法重排后 SDK 索引按名跟随：交换肩俯仰/肩侧摆不错标。

    绑定地址按名解析，profile 与 runtime 同步重排关节序是合法配置；
    若 SDK 对照按列表位置配对，侧摆会被错标成俯仰的索引（15/22），
    且 selected_side_joints 行内 joint_name 与 sdk_name 自相矛盾。
    """
    base_index = 15 if side == "left" else 22
    expected = {f"{side}_{stem}_joint": base_index + i for i, stem in enumerate(_SDK_STEMS)}

    profile_cfg = yaml.safe_load(PROFILE_PATHS[side].read_text(encoding="utf-8"))
    profile_cfg["model_path"] = str(
        (REPO_ROOT / "src" / "robot" / f"r1d_model_{side}.xml").resolve()
    )
    profile_cfg["active_joint_names"][0], profile_cfg["active_joint_names"][1] = (
        profile_cfg["active_joint_names"][1],
        profile_cfg["active_joint_names"][0],
    )
    profile_cfg["actuator_names"][0], profile_cfg["actuator_names"][1] = (
        profile_cfg["actuator_names"][1],
        profile_cfg["actuator_names"][0],
    )
    runtime_cfg = yaml.safe_load(RUNTIME_PATHS[side].read_text(encoding="utf-8"))
    runtime_cfg["joint_names"][0], runtime_cfg["joint_names"][1] = (
        runtime_cfg["joint_names"][1],
        runtime_cfg["joint_names"][0],
    )
    runtime_cfg["init_q"][0], runtime_cfg["init_q"][1] = (
        runtime_cfg["init_q"][1],
        runtime_cfg["init_q"][0],
    )
    profile_path = tmp_path / f"r1d_{side}_swapped.yaml"
    runtime_path = tmp_path / f"r1d_runtime_{side}_swapped.yaml"
    profile_path.write_text(yaml.safe_dump(profile_cfg, allow_unicode=True), encoding="utf-8")
    runtime_path.write_text(
        yaml.safe_dump(runtime_cfg, allow_unicode=True), encoding="utf-8"
    )

    _report, json_path, _bundle = run_inspection(profile_path, runtime_path, tmp_path / "out")
    report = json.loads(json_path.read_text(encoding="utf-8"))

    # 两处报告字段都按完整关节名取 SDK 索引
    for row in report["binding"]["per_joint"]:
        assert row["sdk_reference_index"] == expected[row["name"]]
    for row in report["sdk_index_reference"]["selected_side_joints"]:
        assert row["sdk_index"] == expected[row["joint_name"]]
        # 行内自洽：joint_name 语义干 == sdk_name 去 L_/R_ 前缀的小写形式
        stem = row["joint_name"][len(side) + 1 : -len("_joint")]
        assert stem == row["sdk_name"].split("_", 1)[1].lower()
    # 重排后索引全集合仍完整覆盖该侧 7 个连续地址
    assert {row["sdk_index"] for row in report["sdk_index_reference"]["selected_side_joints"]} == set(
        expected.values()
    )


def test_sdk_lookup_rejects_unknown_joint_name() -> None:
    """关节名对不上 SDK 表 → ValueError（配置错误类，main 映射退出码 2）。"""
    with pytest.raises(ValueError, match="SDK 对照表"):
        _sdk_entry_for_joint("left", "left_shoulder_swing_joint")
    with pytest.raises(ValueError, match="形态"):
        _sdk_entry_for_joint("right", "right_shoulder_pitch")


# ============================================================================
# 2. 异常配置退出码 2
# ============================================================================


def test_config_error_missing_profile_file(tmp_path: Path) -> None:
    """robot-config 指向不存在的文件 → 退出码 2。"""
    rc = main(
        [
            "--robot-config",
            str(tmp_path / "no_such.yaml"),
            "--runtime-config",
            str(RUNTIME_PATHS["left"]),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == EXIT_CONFIG == 2


def test_config_error_missing_runtime_file(tmp_path: Path) -> None:
    """runtime-config 指向不存在的文件 → 退出码 2。"""
    rc = main(
        [
            "--robot-config",
            str(PROFILE_PATHS["left"]),
            "--runtime-config",
            str(tmp_path / "no_such_runtime.yaml"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2


def test_config_error_bad_yaml(tmp_path: Path) -> None:
    """坏 YAML（语法错误）→ 退出码 2。"""
    bad = tmp_path / "bad.yaml"
    bad.write_text("robot_id: [unclosed", encoding="utf-8")
    rc = main(
        [
            "--robot-config",
            str(bad),
            "--runtime-config",
            str(RUNTIME_PATHS["left"]),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2


def test_config_error_identity_mismatch(tmp_path: Path) -> None:
    """身份不符（runtime 的 robot_id 与 profile 不一致）→ 退出码 2。"""
    tampered = tmp_path / "r1d_runtime_left_tampered.yaml"
    text = RUNTIME_PATHS["left"].read_text(encoding="utf-8")
    tampered.write_text(
        text.replace("robot_id: unitree_r1d", "robot_id: something_else"),
        encoding="utf-8",
    )
    rc = main(
        [
            "--robot-config",
            str(PROFILE_PATHS["left"]),
            "--runtime-config",
            str(tampered),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    # 异常配置不得写出 JSON
    assert not (tmp_path / "out" / "inspect_r1d_left.json").exists()


def test_config_error_missing_cli_argument(tmp_path: Path) -> None:
    """缺必传参数（--output-dir）→ argparse SystemExit(2)。"""
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--robot-config",
                str(PROFILE_PATHS["left"]),
                "--runtime-config",
                str(RUNTIME_PATHS["left"]),
            ]
        )
    assert exc_info.value.code == 2


# ============================================================================
# 3. 不启动运行时/SDK/viewer
# ============================================================================


def test_does_not_start_runtime_or_sdk(tmp_path: Path) -> None:
    """无头运行不新增 src.real/src.ilqt/src.sim/SDK/mujoco.viewer 模块。"""
    snapshot = set(sys.modules)
    rc, _json_path = _run_headless("left", tmp_path)
    assert rc == 0
    new_modules = {
        name
        for name in set(sys.modules) - snapshot
        if name == "mujoco.viewer" or name.startswith(_FORBIDDEN_MODULE_PREFIXES)
    }
    assert not new_modules, f"无头检查不应触碰运行时/SDK/viewer: {sorted(new_modules)}"


# ============================================================================
# 4. 渲染
# ============================================================================


class _FakeRenderer:
    """mujoco.Renderer 替身：记录 update_scene/render 调用序列。"""

    instances: list["_FakeRenderer"] = []
    calls: list[dict[str, object]] = []

    def __init__(self, model: object, *, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self._closed = False
        type(self).instances.append(self)

    def update_scene(self, data: object, *, camera: object) -> None:
        """记录相机参数（lookat/azimuth/elevation）。"""
        type(self).calls.append(
            {
                "lookat": list(camera.lookat),
                "azimuth": float(camera.azimuth),
                "elevation": float(camera.elevation),
            }
        )

    def render(self) -> np.ndarray:
        """返回非全黑合成像素（便于 PNG 写出与尺寸断言）。"""
        type(self).calls.append({"render": True})
        pixels = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        pixels[:, :, 0] = 200  # 非零内容，证明像素确实被写出
        return pixels

    def close(self) -> None:
        """标记关闭。"""
        self._closed = True


def test_render_mock_sequence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """mock Renderer：front/side/oblique 三视图调用序列与文件名正确。"""
    _FakeRenderer.instances.clear()
    _FakeRenderer.calls.clear()
    monkeypatch.setattr(mujoco, "Renderer", _FakeRenderer)

    rc = main(
        [
            "--robot-config",
            str(PROFILE_PATHS["right"]),
            "--runtime-config",
            str(RUNTIME_PATHS["right"]),
            "--output-dir",
            str(tmp_path),
            "--render",
        ]
    )
    assert rc == 0
    # 三次 update_scene（对应三视图），相机方位角互不相同
    cam_calls = [c for c in _FakeRenderer.calls if "azimuth" in c]
    assert len(cam_calls) == 3
    azimuths = {c["azimuth"] for c in cam_calls}
    assert len(azimuths) == 3, "front/side/oblique 应使用互不相同的方位角"
    # 三张 PNG 已写出且内容非空
    for view in ("front", "side", "oblique"):
        png = tmp_path / f"inspect_r1d_right_{view}.png"
        assert png.is_file(), f"缺少三视图 {png}"
        assert png.read_bytes().startswith(b"\x89PNG")
        assert png.stat().st_size > 0


def test_render_failure_preserves_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """渲染失败 → 非零退出（非 2）且已生成的 JSON 保留。"""

    def _boom(model: object, *, height: int, width: int) -> object:
        raise RuntimeError("GL 上下文不可用（模拟）")

    monkeypatch.setattr(mujoco, "Renderer", _boom)
    rc = main(
        [
            "--robot-config",
            str(PROFILE_PATHS["left"]),
            "--runtime-config",
            str(RUNTIME_PATHS["left"]),
            "--output-dir",
            str(tmp_path),
            "--render",
        ]
    )
    assert rc != 0
    assert rc != EXIT_CONFIG, "渲染失败不应报为配置错误（退出码 2）"
    json_path = tmp_path / "inspect_r1d_left.json"
    assert json_path.is_file(), "渲染失败必须保留已生成的 JSON"
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["side"] == "left"


def test_render_live_or_skip(tmp_path: Path) -> None:
    """真渲染（本机 GL 可用时）；不可用则优雅跳过。"""
    profile = load_robot_profile(PROFILE_PATHS["left"])
    parameters = load_runtime_parameters(RUNTIME_PATHS["left"], profile)
    bundle = load_robot_model(profile, parameters)
    try:
        probe = mujoco.Renderer(bundle.model, height=32, width=32)
        probe.close()
    except Exception as exc:  # noqa: BLE001 — 任意 GL 失败均跳过
        pytest.skip(f"本机 offscreen 渲染不可用，跳过真渲染: {exc}")

    rc = main(
        [
            "--robot-config",
            str(PROFILE_PATHS["left"]),
            "--runtime-config",
            str(RUNTIME_PATHS["left"]),
            "--output-dir",
            str(tmp_path),
            "--render",
        ]
    )
    assert rc == 0
    for view in ("front", "side", "oblique"):
        png = tmp_path / f"inspect_r1d_left_{view}.png"
        assert png.is_file()
        assert png.stat().st_size > 10_000, f"{view} 视图应有实际内容"
