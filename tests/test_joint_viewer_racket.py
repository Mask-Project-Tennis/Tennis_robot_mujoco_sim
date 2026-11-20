"""R14 回归：关节查看器必须真正装配 ``--racket-config`` 并改位置伺服。

审查 R14 实测：旧 R1D 分支只 ``load_racket_profile`` 校验后丢弃，两套球拍
编译出的 ``geom_size`` / ``body_mass`` / ``site_pos`` 逐位相同。修复后 R1D
分支经 ``load_robot_model(..., racket_profile=racket)`` 装配实际模型，再把
已绑定执行器原地改为位置伺服（kp/kd/ctrlrange 来自 runtime 与真实限位）。
测试只构建模型，不打开 viewer 窗口。
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "tools"))
sys.path.insert(0, str(REPO_ROOT))

import rm65_joint_viewer as viewer  # noqa: E402
from src.robot.racket_profile import load_racket_profile  # noqa: E402
from src.robot.robot_profile import load_robot_profile  # noqa: E402
from src.robot.runtime_parameters import load_runtime_parameters  # noqa: E402

PROFILE_R1D = REPO_ROOT / "configs" / "robots" / "r1d_right.yaml"
RUNTIME_R1D = REPO_ROOT / "configs" / "robots" / "r1d_v12_runtime_right.yaml"
RACKET_BLADE = REPO_ROOT / "configs" / "rackets" / "blade100_v9_reference.yaml"
RACKET_LEGACY = REPO_ROOT / "configs" / "rackets" / "rm65_legacy.yaml"


def _r1d_spec(racket: Path | None = None) -> viewer.JointViewerSpec:
    """按 R1D 右臂描述 + runtime（可选球拍）构建查看器模型。"""
    return viewer.build_position_actuator_model(
        PROFILE_R1D, RUNTIME_R1D, racket
    )


def test_two_rackets_build_different_physical_models() -> None:
    """blade vs legacy：geom_size/body_mass/site_pos 至少一项不同。"""
    blade = _r1d_spec(RACKET_BLADE)
    legacy = _r1d_spec(RACKET_LEGACY)
    assert blade.model.nu == 7
    assert legacy.model.nu == 7
    assert blade.robot_id == "unitree_r1d"

    geom_differs = not np.array_equal(blade.model.geom_size, legacy.model.geom_size)
    mass_differs = not np.allclose(
        blade.model.body_mass, legacy.model.body_mass
    )
    site_differs = not np.allclose(blade.model.site_pos, legacy.model.site_pos)
    assert geom_differs or mass_differs or site_differs, (
        "两套球拍装配出的模型完全相同，说明 --racket-config 仍被丢弃"
    )

    # 质量差严格等于 RacketProfile 的质量差（球拍 + 连接件），证明质量来自档案
    blade_profile = load_racket_profile(RACKET_BLADE.resolve())
    legacy_profile = load_racket_profile(RACKET_LEGACY.resolve())
    blade_total = float(blade_profile.mass + blade_profile.connector.mass)
    legacy_total = float(legacy_profile.mass)
    observed = float(blade.model.body_mass.sum() - legacy.model.body_mass.sum())
    assert observed == pytest.approx(blade_total - legacy_total, abs=1e-9)


def test_r1d_without_racket_still_builds() -> None:
    """不传 --racket-config 仍可构建（使用模板默认拍）。"""
    spec = _r1d_spec()
    assert spec.model.nu == 7
    assert spec.actuator_count == 7
    assert len(spec.init_q) == 7


def test_invalid_racket_path_raises() -> None:
    """非法球拍路径必须报错（不得静默忽略）。"""
    with pytest.raises((FileNotFoundError, ValueError)):
        viewer.build_position_actuator_model(
            PROFILE_R1D, RUNTIME_R1D, REPO_ROOT / "configs" / "rackets" / "no_such.yaml"
        )


def test_r1d_without_runtime_raises() -> None:
    """R1D 缺少 runtime（scene/init_q/kp/kd 来源）→ 明确 ValueError。"""
    with pytest.raises(ValueError, match="--runtime-config"):
        viewer.build_position_actuator_model(PROFILE_R1D, None, RACKET_BLADE)


def test_position_servo_uses_runtime_gains_and_joint_limits() -> None:
    """执行器原地改为位置伺服：gain/bias/ctrlrange 与 runtime/限位一致。"""
    profile = load_robot_profile(PROFILE_R1D.resolve())
    runtime = load_runtime_parameters(RUNTIME_R1D.resolve(), profile)
    spec = _r1d_spec(RACKET_BLADE)
    model = spec.model

    np.testing.assert_allclose(spec.init_q, runtime.init_q, rtol=0.0, atol=0.0)
    assert runtime.kp is not None and runtime.kd is not None
    for index in range(spec.actuator_count):
        joint_id = int(model.actuator_trnid[index, 0])
        assert joint_id >= 0
        assert model.actuator_gainprm[index, 0] == pytest.approx(
            float(runtime.kp[index])
        )
        assert model.actuator_biasprm[index, 1] == pytest.approx(
            -float(runtime.kp[index])
        )
        assert model.actuator_biasprm[index, 2] == pytest.approx(
            -float(runtime.kd[index])
        )
        assert int(model.actuator_biastype[index]) == int(
            mujoco.mjtBias.mjBIAS_AFFINE
        )
        np.testing.assert_allclose(
            model.actuator_ctrlrange[index], model.jnt_range[joint_id],
            rtol=0.0, atol=0.0,
        )
        assert int(model.actuator_ctrllimited[index]) == 1


def test_rm65_default_branch_unchanged() -> None:
    """RM65 缺省分支保持现状：12 个 position 执行器 + 旧初始位姿。"""
    spec = viewer.build_position_actuator_model()
    assert spec.robot_id == "rm65"
    assert spec.actuator_count == 12
    assert spec.model.nu == 12
    np.testing.assert_allclose(spec.init_q, viewer.INIT_Q_RIGHT)
    np.testing.assert_allclose(spec.init_q_left, viewer.INIT_Q_LEFT)
    assert set(np.asarray(spec.model.actuator_biastype).tolist()) == {
        int(mujoco.mjtBias.mjBIAS_AFFINE)
    }

    # RM65 没有按 RacketProfile 装配模型的入口：显式给出必须拒绝，
    # 不能“仅校验后丢弃”造成参数被静默忽略（R14）。
    with pytest.raises(ValueError, match="--racket-config"):
        viewer.build_position_actuator_model(None, None, RACKET_BLADE)


def test_unsupported_robot_id_raises_value_error(tmp_path: Path) -> None:
    """不支持球拍装配入口的 robot_id → 明确 ValueError（不得静默忽略）。"""
    data = {
        "schema_version": 1,
        "robot_id": "d2",
        "model_path": str(REPO_ROOT / "src" / "robot" / "model.xml"),
        "active_joint_names": ["joint1"],
        "actuator_names": ["a1"],
        "held_joint_names": [],
        "tcp_site_name": "tcp",
    }
    profile_path = tmp_path / "d2.yaml"
    profile_path.write_text(
        yaml.safe_dump(data, allow_unicode=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="不支持"):
        viewer.build_position_actuator_model(profile_path)
