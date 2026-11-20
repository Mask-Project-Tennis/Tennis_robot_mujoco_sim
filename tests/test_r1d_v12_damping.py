"""T7 R1D V12 数值 parity：估计关节阻尼/armature 注入测试。

覆盖 lead T7 裁定（2026-11-19）的硬约束：

* R1D V12 执行/规划模型（含 ``planning_env_factory`` 新建环境）的 7 个主动
  关节被注入估计 ``dof_damping`` / ``dof_armature``；
* 同数值只在 V12 R1D 路径生效：直接 ``load_robot_model`` 的裸模型保持 0；
* RM65 V12 路径逐位不变（六关节 XML 声明值）；
* URDF/XML 资产文件哈希不变（阻尼/armature 不写回资产，estimated 覆盖）；
* 估值常量与 lead 批准初值一致（estimated / provisional）。
"""

from __future__ import annotations


import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("mujoco")

import rm65_mpc_v12 as v12  # noqa: E402
from src.robot.model_factory import load_robot_model  # noqa: E402
from src.robot.robot_profile import load_robot_profile  # noqa: E402
from src.robot.runtime_parameters import load_runtime_parameters  # noqa: E402
from src.sim.v12_robot_adapter import (  # noqa: E402
    _R1D_ESTIMATED_DOF_ARMATURE,
    _R1D_ESTIMATED_DOF_DAMPING,
    build_v12_robot,
)

_R1D_RIGHT_PROFILE = ROOT / "configs" / "robots" / "r1d_right.yaml"
_R1D_RIGHT_RUNTIME = ROOT / "configs" / "robots" / "r1d_v12_runtime_right.yaml"

#: lead 批准的估计初值（参考 RM65 六关节声明值按序延展末项）。
_EXPECTED_DAMPING = (2.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1)
_EXPECTED_ARMATURE = (0.2, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01)

#: RM65 模型六关节 XML 声明值（src/robot/rm65_model.xml 右臂）。
_RM65_DAMPING = (2.0, 2.0, 1.0, 0.5, 0.3, 0.2)
_RM65_ARMATURE = (0.2, 0.2, 0.1, 0.05, 0.03, 0.02)

#: URDF/XML 资产哈希（T7 修复前基线，确保 estimated 参数不写回资产）。
_XML_SHA256 = {
    "src/robot/r1d_model_right.xml": (
        "7a9116cfb1fbaf7863771d7ecd5e9f4219dfc67d7207321c829d95cd2171cec5"
    ),
    "src/robot/r1d_arm_right.xml": (
        "fcaeb323d0ae67ed6f5b80b0ed8ef25bfc166fea11c5523a9e18dc44684799b1"
    ),
    "src/robot/r1d_assets.xml": (
        "b03be72a2c23cf195ed59d09ce8e32947e5d639f2d86c4a40b4095e984d372c3"
    ),
    "src/robot/r1d_env.xml": (
        "a3b44d8789c91c0a9f3d6f8f2c74348b85cc97fc18a6cc6122260cb3a1b9139f"
    ),
}


def _resolve(argv: list[str]):  # noqa: ANN202 - 测试辅助
    """用真实 CLI 解析器 + T3 解析层构造 ResolvedV12Config。"""
    parser = v12._build_parser()
    args = parser.parse_args(argv)
    explicit = v12._explicit_cli_actions(parser, argv)
    return v12.resolve_v12_config(args, explicit)


def _r1d_config():  # noqa: ANN202 - 测试辅助
    """R1D V12 overlay 配置（右臂、seed 42）。"""
    return _resolve(
        [
            "--robot-config",
            "configs/robots/r1d_right.yaml",
            "--seed",
            "42",
            "--no-plot",
        ]
    )


def _rm65_config():  # noqa: ANN202 - 测试辅助
    """RM65 V12 配置（seed 42、serve-box），仅构造 bundle、不跑回合。"""
    return _resolve(
        ["--serve-box", "--ball-speed", "7", "--seed", "42", "--no-plot"]
    )


def _joint_dynamics(model, joint_names) -> tuple[list[float], list[float]]:
    """按关节名顺序读 model 的 dof_damping / dof_armature。"""
    import mujoco

    damping: list[float] = []
    armature: list[float] = []
    for name in joint_names:
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name)))
        assert joint_id >= 0, f"模型缺少关节 {name}"
        dof = int(model.jnt_dofadr[joint_id])
        damping.append(float(model.dof_damping[dof]))
        armature.append(float(model.dof_armature[dof]))
    return damping, armature


def test_estimated_constants_match_lead_approved_values() -> None:
    """estimated 常量必须与 lead 批准的初值逐项一致。"""
    assert _R1D_ESTIMATED_DOF_DAMPING == _EXPECTED_DAMPING
    assert _R1D_ESTIMATED_DOF_ARMATURE == _EXPECTED_ARMATURE


def test_r1d_v12_bundle_injects_estimated_dynamics() -> None:
    """R1D V12 执行/规划/工厂环境都注入估计阻尼与 armature。"""
    bundle = build_v12_robot(_r1d_config())
    for env in (bundle.execution, bundle.planning):
        damping, armature = _joint_dynamics(env.model, bundle.joint_names)
        assert len(damping) == len(_EXPECTED_DAMPING)
        assert damping == pytest.approx(list(_EXPECTED_DAMPING))
        assert armature == pytest.approx(list(_EXPECTED_ARMATURE))

    factory_env = bundle.planning_env_factory()
    damping, armature = _joint_dynamics(factory_env.model, bundle.joint_names)
    assert damping == pytest.approx(list(_EXPECTED_DAMPING))
    assert armature == pytest.approx(list(_EXPECTED_ARMATURE))
    # 工厂环境必须独立于 bundle.planning（不同 MjModel 实例）。
    assert factory_env.model is not bundle.planning.model


def test_r1d_non_v12_bare_model_keeps_zero_dynamics() -> None:
    """非 V12 路径（裸 load_robot_model）仍保持 URDF 行为：damping/armature=0。"""
    profile = load_robot_profile(_R1D_RIGHT_PROFILE)
    parameters = load_runtime_parameters(_R1D_RIGHT_RUNTIME, profile)
    bare = load_robot_model(profile, parameters)
    damping, armature = _joint_dynamics(bare.model, profile.active_joint_names)
    assert damping == pytest.approx([0.0] * len(_EXPECTED_DAMPING))
    assert armature == pytest.approx([0.0] * len(_EXPECTED_ARMATURE))


def test_rm65_v12_bundle_unchanged() -> None:
    """RM65 V12 路径保持 XML 声明值（估计覆盖只对 R1D 生效）。"""
    bundle = build_v12_robot(_rm65_config())
    damping, armature = _joint_dynamics(bundle.execution.model, bundle.joint_names)
    assert len(damping) == len(_RM65_DAMPING)
    assert damping == pytest.approx(list(_RM65_DAMPING))
    assert armature == pytest.approx(list(_RM65_ARMATURE))


def test_r1d_xml_asset_hashes_unchanged() -> None:
    """URDF/XML 资产哈希不变：estimated 参数不写回资产文件。"""
    for relative, expected in _XML_SHA256.items():
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert digest == expected, f"{relative} 资产被修改"
