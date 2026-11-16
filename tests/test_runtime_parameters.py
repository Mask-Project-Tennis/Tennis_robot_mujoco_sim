"""RobotRuntimeParameters 运行参数模块测试。

覆盖：合法 kinematics 参数、身份/关节顺序拒绝、形状与非有限值拒绝、
YAML 未知键拒绝（含 int/str 混合键）、D2 未知限值保留（无隐式补值）、
simulation 完整性预检、数组不别名输入、手构与 YAML 共用校验、
以及模块导入纯度（不拉入 real/sim/ilqt/MuJoCo/SDK）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import yaml

from src.robot.robot_profile import RobotProfile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    SceneParameters,
    load_runtime_parameters,
    validate_runtime_parameters,
)

# 仓库根目录（tests/ 的父目录）
REPO_ROOT = Path(__file__).resolve().parent.parent

# 仓库中的 D2 临时运行参数文件（本任务交付物之一）
D2_RUNTIME_YAML = REPO_ROOT / "configs" / "robots" / "d2_runtime.yaml"

# D2 七个主动关节名（与 shared-constraints §2.1 一致）
D2_JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 8))

# 可空限值字段（整组未知语义）
_NULLABLE_LIMIT_FIELDS = (
    "kp",
    "kd",
    "q_lower",
    "q_upper",
    "qdot_max",
    "qddot_max",
    "torque_max",
    "max_tcp_speed",
)


# ---------- 合成对象 helper ----------


def _d2_profile(tmp_path: Path) -> RobotProfile:
    """合成 unitree_d2 描述：7 主动关节、无保持关节、TCP=racket_center。"""
    return RobotProfile(
        schema_version=1,
        robot_id="unitree_d2",
        model_path=tmp_path / "d2_model.xml",
        active_joint_names=D2_JOINT_NAMES,
        actuator_names=tuple(f"torque_joint_{i}" for i in range(1, 8)),
        held_joint_names=(),
        tcp_site_name="racket_center",
    )


def _rm65_style_profile(tmp_path: Path) -> RobotProfile:
    """合成六主动 + 六保持关节描述（scene=None 与完整限值场景）。"""
    return RobotProfile(
        schema_version=1,
        robot_id="sixbot",
        model_path=tmp_path / "sixbot.xml",
        active_joint_names=tuple(f"j{i}" for i in range(1, 7)),
        actuator_names=tuple(f"a{i}" for i in range(1, 7)),
        held_joint_names=tuple(f"h{i}" for i in range(1, 7)),
        tcp_site_name="tcp",
    )


def _scene_dict() -> dict[str, Any]:
    """shared-constraints §2.3 表的临时几何数值（D2 场景）。"""
    return {
        "base_position": [0.0, 0.0, 1.0],
        "base_quaternion": [1.0, 0.0, 0.0, 0.0],
        "racket_position": [0.0, 0.0, 0.25],
        "racket_quaternion": [1.0, 0.0, 0.0, 0.0],
        "racket_radius": 0.12,
        "racket_half_thickness": 0.005,
        "racket_mass": 0.30,
        "torso_position": [-0.25, 0.0, -0.30],
        "torso_half_size": [0.15, 0.20, 0.30],
    }


def _build_scene(**overrides: Any) -> SceneParameters:
    """按 §2.3 数值构造 SceneParameters（支持字段覆盖）。"""
    values = _scene_dict() | overrides
    return SceneParameters(**values)


def _d2_provisional_fields() -> tuple[str, ...]:
    """§2.3 表全部临时项（数值项 + null 限值项）的 provisional 名单。"""
    return (
        "dt",
        "init_q",
        "base_position",
        "base_quaternion",
        "racket_position",
        "racket_quaternion",
        "racket_radius",
        "racket_half_thickness",
        "racket_mass",
        "torso_position",
        "torso_half_size",
        "kp",
        "kd",
        "qdot_max",
        "qddot_max",
        "torque_max",
        "max_tcp_speed",
    )


def _d2_runtime_dict() -> dict[str, Any]:
    """与 configs/robots/d2_runtime.yaml 同构的完整运行参数字典。"""
    return {
        "schema_version": 1,
        "robot_id": "unitree_d2",
        "joint_names": list(D2_JOINT_NAMES),
        "dt": 0.005,
        "init_q": [0.0, 1.2, 0.0, -1.2, 0.0, 0.0, 0.0],
        "held_q": [],
        "kp": None,
        "kd": None,
        "q_lower": None,
        "q_upper": None,
        "qdot_max": None,
        "qddot_max": None,
        "torque_max": None,
        "max_tcp_speed": None,
        "scene": _scene_dict(),
        "provisional_fields": list(_d2_provisional_fields()),
    }


# sentinel：区分"未传 scene"（默认 §2.3 场景）与"显式 scene=None"
_UNSET = object()


def _build_d2_params(
    scene: Any = _UNSET,
    **overrides: Any,
) -> RobotRuntimeParameters:
    """手构 D2 运行参数（默认 §2.3 数值含场景，支持字段覆盖）。"""
    scene_value = _build_scene() if scene is _UNSET else scene
    defaults: dict[str, Any] = {
        "robot_id": "unitree_d2",
        "joint_names": D2_JOINT_NAMES,
        "dt": 0.005,
        "init_q": [0.0, 1.2, 0.0, -1.2, 0.0, 0.0, 0.0],
        "held_q": [],
        "kp": None,
        "kd": None,
        "q_lower": None,
        "q_upper": None,
        "qdot_max": None,
        "qddot_max": None,
        "torque_max": None,
        "max_tcp_speed": None,
        "provisional_fields": _d2_provisional_fields(),
        "source": "手构测试对象（unitree_d2）",
    }
    values = defaults | overrides
    return RobotRuntimeParameters(scene=scene_value, **values)


def _build_full_six_dof_params(profile: RobotProfile) -> RobotRuntimeParameters:
    """六维 + 完整限值参数（simulation 预检可通过的样例）。"""
    n = profile.ndof
    return RobotRuntimeParameters(
        robot_id=profile.robot_id,
        joint_names=profile.active_joint_names,
        dt=0.005,
        init_q=[0.0] * n,
        held_q=[0.0] * len(profile.held_joint_names),
        kp=[200.0] * n,
        kd=[10.0] * n,
        q_lower=[-3.0] * n,
        q_upper=[3.0] * n,
        qdot_max=[3.0] * n,
        qddot_max=[10.0] * n,
        torque_max=[50.0] * n,
        max_tcp_speed=1.8,
        scene=None,
        provisional_fields=(),
        source="手构测试对象（完整限值）",
    )


def _write_runtime(
    directory: Path, data: dict[str, Any], filename: str = "runtime.yaml"
) -> Path:
    """把运行参数字典写成 YAML 文件并返回路径。"""
    path = directory / filename
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


# ---------- 测试 ----------


class TestValidKinematicsParameters:
    """合法参数通过 kinematics 级校验。"""

    @pytest.mark.parametrize(
        "variant",
        ["d2_unknown_limits", "six_dof_full_limits", "six_dof_pd_only"],
        ids=["D2七维未知限值带场景", "六维完整限值无场景", "六维仅PD无位置限位"],
    )
    def test_valid_kinematics_parameters(self, tmp_path: Path, variant: str) -> None:
        """三种合法形态均通过 kinematics 校验，维度不被截断。"""
        if variant == "d2_unknown_limits":
            profile = _d2_profile(tmp_path)
            params = _build_d2_params()
            assert params.init_q.shape == (7,)  # 七维完整，无六维截断
            assert params.scene is not None
        elif variant == "six_dof_full_limits":
            profile = _rm65_style_profile(tmp_path)
            params = _build_full_six_dof_params(profile)
            assert params.scene is None  # RM65 风格不覆写旧几何
            assert params.held_q.shape == (6,)
        else:  # six_dof_pd_only：PD 有值、限位未知 → kinematics 仍通过
            profile = _rm65_style_profile(tmp_path)
            params = replace(
                _build_full_six_dof_params(profile),
                q_lower=None,
                q_upper=None,
                qdot_max=None,
                qddot_max=None,
                torque_max=None,
                max_tcp_speed=None,
            )
            assert params.kp is not None and params.kd is not None

        validate_runtime_parameters(params, profile, purpose="kinematics")


class TestIdentityAndOrderMismatch:
    """身份与关节顺序必须与 profile 完全一致。"""

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(
                lambda p: replace(p, robot_id="other_bot"), id="robot_id不一致"
            ),
            pytest.param(
                lambda p: replace(p, joint_names=tuple(reversed(p.joint_names))),
                id="关节顺序颠倒",
            ),
            pytest.param(
                lambda p: replace(p, joint_names=p.joint_names[:-1]),
                id="关节数截断为六维",
            ),
        ],
    )
    def test_rejects_identity_or_order_mismatch(
        self,
        tmp_path: Path,
        mutate: Callable[[RobotRuntimeParameters], RobotRuntimeParameters],
    ) -> None:
        """身份或顺序不匹配 → ValueError，消息含字段名与 profile 身份。"""
        profile = _d2_profile(tmp_path)
        params = mutate(_build_d2_params())

        with pytest.raises(ValueError) as exc_info:
            validate_runtime_parameters(params, profile, purpose="kinematics")
        message = str(exc_info.value)
        assert ("robot_id" in message) or ("joint_names" in message)
        assert profile.robot_id in message  # 错误上下文含 robot_id
        assert params.source in message  # 错误上下文含来源说明


class TestBadShapesAndNonfinite:
    """形状错误与非有限值必须被拒绝。"""

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            pytest.param("init_q", [0.0] * 6, id="init_q六维截断"),
            pytest.param("init_q", [0.0] * 8, id="init_q八维广播"),
            pytest.param("held_q", [0.1], id="held_q长度不符"),
            pytest.param("kp", [100.0] * 6, id="kp维度不符"),
            pytest.param("q_upper", [3.0] * 6, id="q_upper维度不符"),
            pytest.param("init_q", [float("nan")] * 7, id="init_q含NaN"),
            pytest.param("qdot_max", [float("inf")] * 7, id="qdot_max含Inf"),
            pytest.param("dt", float("nan"), id="dt为NaN"),
            pytest.param("dt", 0.0, id="dt为零"),
            pytest.param("dt", -0.005, id="dt为负"),
        ],
    )
    def test_rejects_bad_shapes_and_nonfinite(
        self, tmp_path: Path, field: str, bad_value: Any
    ) -> None:
        """形状/有限性/正数违规 → ValueError，消息含字段名。"""
        profile = _d2_profile(tmp_path)

        with pytest.raises(ValueError) as exc_info:
            # dt 等标量在构造期即被拒绝；数组问题由 validate 拒绝
            params = replace(_build_d2_params(), **{field: bad_value})
            validate_runtime_parameters(params, profile, purpose="kinematics")
        assert field in str(exc_info.value)


class TestPairAndLimitRules:
    """成对提供、限位关系与四元数单位性（共同校验规则）。"""

    @pytest.mark.parametrize(
        ("field", "value", "pair_name"),
        [
            pytest.param("kd", [10.0] * 7, "kp/kd", id="kd单独提供"),
            pytest.param("kp", [100.0] * 7, "kp/kd", id="kp单独提供"),
            pytest.param("q_upper", [3.0] * 7, "q_lower/q_upper", id="q_upper单独提供"),
        ],
    )
    def test_rejects_unpaired_fields(
        self, tmp_path: Path, field: str, value: list[float], pair_name: str
    ) -> None:
        """可空限值必须整组成对提供。"""
        profile = _d2_profile(tmp_path)
        params = replace(_build_d2_params(), **{field: value})

        with pytest.raises(ValueError) as exc_info:
            validate_runtime_parameters(params, profile, purpose="kinematics")
        message = str(exc_info.value)
        assert pair_name in message
        assert field in message

    def test_rejects_bad_position_limits(self, tmp_path: Path) -> None:
        """q_lower >= q_upper → ValueError。"""
        profile = _d2_profile(tmp_path)
        params = _build_d2_params(q_lower=[-3.0] * 7, q_upper=[-3.0] * 7)

        with pytest.raises(ValueError, match="q_lower"):
            validate_runtime_parameters(params, profile, purpose="kinematics")

    def test_rejects_init_q_out_of_range(self, tmp_path: Path) -> None:
        """初态越出已知位置限位 → ValueError。"""
        profile = _d2_profile(tmp_path)
        params = _build_d2_params(
            init_q=[9.9] * 7, q_lower=[-3.0] * 7, q_upper=[3.0] * 7
        )

        with pytest.raises(ValueError) as exc_info:
            validate_runtime_parameters(params, profile, purpose="kinematics")
        assert "init_q" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("quat_field", "bad_value"),
        [
            pytest.param("base_quaternion", [1.1, 0.0, 0.0, 0.0], id="基座非单位"),
            pytest.param("racket_quaternion", [0.0, 0.0, 0.0, 0.0], id="拍面零四元数"),
        ],
    )
    def test_rejects_non_unit_quaternions(
        self, tmp_path: Path, quat_field: str, bad_value: list[float]
    ) -> None:
        """四元数范数偏离 1 超过 1e-8 → 拒绝而非默默归一化。"""
        profile = _d2_profile(tmp_path)
        params = _build_d2_params(scene=_build_scene(**{quat_field: bad_value}))

        with pytest.raises(ValueError) as exc_info:
            validate_runtime_parameters(params, profile, purpose="kinematics")
        assert quat_field in str(exc_info.value)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            pytest.param("kp", [0.0] * 7, id="kp含零"),
            pytest.param("kp", [-100.0] * 7, id="kp含负"),
            pytest.param("kd", [-10.0] * 7, id="kd含负"),
            pytest.param("qdot_max", [0.0] * 7, id="qdot_max含零"),
            pytest.param("qddot_max", [-1.0] * 7, id="qddot_max含负"),
            pytest.param("torque_max", [0.0] * 7, id="torque_max含零"),
        ],
    )
    def test_rejects_nonpositive_limits(
        self, tmp_path: Path, field: str, value: list[float]
    ) -> None:
        """kp>0、kd>=0、速度/加速度/力矩上限必须逐元素为正。"""
        profile = _d2_profile(tmp_path)
        params = _build_d2_params(**{field: value})

        with pytest.raises(ValueError) as exc_info:
            validate_runtime_parameters(params, profile, purpose="kinematics")
        assert field in str(exc_info.value)

    @pytest.mark.parametrize(
        ("scene_field", "value"),
        [
            pytest.param("racket_radius", 0.0, id="拍半径为零"),
            pytest.param("racket_half_thickness", -0.005, id="拍半厚为负"),
            pytest.param("racket_mass", 0.0, id="拍质量为零"),
        ],
    )
    def test_rejects_nonpositive_scene_scalars(
        self, tmp_path: Path, scene_field: str, value: float
    ) -> None:
        """场景正数项（半径/半厚/质量）必须 >0。"""
        profile = _d2_profile(tmp_path)
        params = _build_d2_params(scene=_build_scene(**{scene_field: value}))

        with pytest.raises(ValueError) as exc_info:
            validate_runtime_parameters(params, profile, purpose="kinematics")
        assert scene_field in str(exc_info.value)

    def test_rejects_bad_max_tcp_speed(self, tmp_path: Path) -> None:
        """max_tcp_speed 已提供时必须为正数。"""
        profile = _d2_profile(tmp_path)
        params = _build_d2_params(max_tcp_speed=0.0)

        with pytest.raises(ValueError, match="max_tcp_speed"):
            validate_runtime_parameters(params, profile, purpose="kinematics")

    def test_rejects_unknown_purpose(self, tmp_path: Path) -> None:
        """purpose 只接受 kinematics/simulation；无 real 授权开关。"""
        profile = _d2_profile(tmp_path)
        params = _build_d2_params()

        with pytest.raises(ValueError, match="purpose"):
            validate_runtime_parameters(  # type: ignore[arg-type]
                params, profile, purpose="real"
            )

    def test_rejects_unknown_provisional_names(self, tmp_path: Path) -> None:
        """provisional_fields 中的未知字段名（拼写错误）被拒绝。"""
        profile = _d2_profile(tmp_path)
        params = _build_d2_params(provisional_fields=("init_qq",))

        with pytest.raises(ValueError) as exc_info:
            validate_runtime_parameters(params, profile, purpose="kinematics")
        message = str(exc_info.value)
        assert "provisional_fields" in message
        assert "init_qq" in message


class TestYamlStrictness:
    """YAML 严格解析：未知键、缺字段、坏 schema、坏类型。"""

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda d: d.update(extra_key=1), id="顶层未知str键"),
            pytest.param(
                lambda d: (d.update(extra_str_key="x"), d.__setitem__(1, "int键")),
                id="int与str混合未知键",
            ),
            pytest.param(
                lambda d: d["scene"].update(bad_scene_key=1), id="scene内未知键"
            ),
            pytest.param(lambda d: d.update(source="自报来源"), id="source自报"),
        ],
    )
    def test_rejects_unknown_keys(
        self,
        tmp_path: Path,
        mutate: Callable[[dict[str, Any]], Any],
    ) -> None:
        """未知键（含 int/str 混合键）→ ValueError 而非 TypeError。"""
        data = _d2_runtime_dict()
        mutate(data)
        yaml_path = _write_runtime(tmp_path, data)
        profile = _d2_profile(tmp_path)

        with pytest.raises(ValueError) as exc_info:
            load_runtime_parameters(yaml_path, profile)
        message = str(exc_info.value)
        assert str(yaml_path) in message

    @pytest.mark.parametrize(
        "missing",
        ["kp", "scene", "provisional_fields", "dt", "held_q"],
        ids=["缺kp", "缺scene", "缺provisional_fields", "缺dt", "缺held_q"],
    )
    def test_rejects_missing_fields(self, tmp_path: Path, missing: str) -> None:
        """顶层字段必须全部显式存在（未知限值用 null 表达）。"""
        data = _d2_runtime_dict()
        del data[missing]
        yaml_path = _write_runtime(tmp_path, data)
        profile = _d2_profile(tmp_path)

        with pytest.raises(ValueError) as exc_info:
            load_runtime_parameters(yaml_path, profile)
        message = str(exc_info.value)
        assert missing in message
        assert str(yaml_path) in message

    @pytest.mark.parametrize(
        "bad_version",
        [2, True, "1", None],
        ids=["版本2", "布尔true", "字符串1", "缺失"],
    )
    def test_rejects_bad_schema_version(
        self, tmp_path: Path, bad_version: Any
    ) -> None:
        """schema_version 必须是整数 1（bool 亦拒绝）。"""
        data = _d2_runtime_dict()
        if bad_version is None:
            del data["schema_version"]
        else:
            data["schema_version"] = bad_version
        yaml_path = _write_runtime(tmp_path, data)
        profile = _d2_profile(tmp_path)

        with pytest.raises(ValueError) as exc_info:
            load_runtime_parameters(yaml_path, profile)
        assert "schema_version" in str(exc_info.value)

    def test_rejects_bool_scalar(self, tmp_path: Path) -> None:
        """YAML 布尔标量（dt: true）按 bool 拒绝，不当数值接受。"""
        data = _d2_runtime_dict()
        data["dt"] = True
        yaml_path = _write_runtime(tmp_path, data)
        profile = _d2_profile(tmp_path)

        with pytest.raises(ValueError) as exc_info:
            load_runtime_parameters(yaml_path, profile)
        message = str(exc_info.value)
        assert "dt" in message
        assert str(yaml_path) in message

    def test_rejects_non_numeric_array_element(self, tmp_path: Path) -> None:
        """数组字段含字符串元素 → ValueError。"""
        data = _d2_runtime_dict()
        data["init_q"] = [0.0, "1.2", 0.0, -1.2, 0.0, 0.0, 0.0]
        yaml_path = _write_runtime(tmp_path, data)
        profile = _d2_profile(tmp_path)

        with pytest.raises(ValueError) as exc_info:
            load_runtime_parameters(yaml_path, profile)
        message = str(exc_info.value)
        assert "init_q" in message
        assert str(yaml_path) in message

    def test_rejects_wrong_robot_id_against_profile(self, tmp_path: Path) -> None:
        """YAML robot_id 与 profile 不一致 → ValueError，消息含路径来源。"""
        data = _d2_runtime_dict()
        data["robot_id"] = "someone_else"
        yaml_path = _write_runtime(tmp_path, data)
        profile = _d2_profile(tmp_path)

        with pytest.raises(ValueError) as exc_info:
            load_runtime_parameters(yaml_path, profile)
        message = str(exc_info.value)
        assert "robot_id" in message
        assert str(yaml_path) in message

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """参数文件不存在保留 FileNotFoundError，不转为 ValueError。"""
        with pytest.raises(FileNotFoundError):
            load_runtime_parameters(
                tmp_path / "no_such_runtime.yaml", _d2_profile(tmp_path)
            )


class TestD2RuntimeYaml:
    """仓库 D2 临时配置：未知限值保留、无隐式补值、数值与 §2.3 一致。"""

    def test_unknown_d2_limits_remain_unknown(self, tmp_path: Path) -> None:
        """加载 d2_runtime.yaml：全部限值 None、七维完整、kinematics 通过。"""
        profile = _d2_profile(tmp_path)
        params = load_runtime_parameters(D2_RUNTIME_YAML, profile)

        # 简报逐字断言：未知动态能力不隐式补值
        assert params.qdot_max is None
        assert params.torque_max is None
        for field in _NULLABLE_LIMIT_FIELDS:
            assert getattr(params, field) is None, f"字段 {field} 应保持未知（None）"

        # 七维运动学完整，无六维广播或截断
        assert params.joint_names == D2_JOINT_NAMES
        assert params.init_q.shape == (7,)
        assert params.held_q.shape == (0,)  # held_joint_names=[] → (0,)

        # §2.3 临时数值逐项一致
        assert params.dt == pytest.approx(0.005)
        np.testing.assert_allclose(
            params.init_q, [0.0, 1.2, 0.0, -1.2, 0.0, 0.0, 0.0]
        )
        assert params.scene is not None
        np.testing.assert_allclose(params.scene.base_position, [0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            params.scene.base_quaternion, [1.0, 0.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(params.scene.racket_position, [0.0, 0.0, 0.25])
        np.testing.assert_allclose(
            params.scene.racket_quaternion, [1.0, 0.0, 0.0, 0.0]
        )
        assert params.scene.racket_radius == pytest.approx(0.12)
        assert params.scene.racket_half_thickness == pytest.approx(0.005)
        assert params.scene.racket_mass == pytest.approx(0.30)
        np.testing.assert_allclose(params.scene.torso_position, [-0.25, 0.0, -0.30])
        np.testing.assert_allclose(
            params.scene.torso_half_size, [0.15, 0.20, 0.30]
        )

        # source 由加载路径生成（不由 YAML 自报）
        assert params.source == str(D2_RUNTIME_YAML)

        # §2.3 全部临时项列入 provisional_fields
        for name in _d2_provisional_fields():
            assert name in params.provisional_fields

        # 运动学级校验通过
        validate_runtime_parameters(params, profile, purpose="kinematics")


class TestSimulationPurpose:
    """simulation 完整性预检。"""

    def test_simulation_requires_complete_limits(self, tmp_path: Path) -> None:
        """D2 临时配置 simulation 预检失败，消息含全部缺项名称。"""
        profile = _d2_profile(tmp_path)
        params = load_runtime_parameters(D2_RUNTIME_YAML, profile)

        with pytest.raises(ValueError) as exc_info:
            validate_runtime_parameters(params, profile, purpose="simulation")
        message = str(exc_info.value)
        for name in (
            "q_lower",
            "q_upper",
            "qdot_max",
            "qddot_max",
            "torque_max",
            "max_tcp_speed",
        ):
            assert name in message, f"缺项 {name} 未出现在错误消息中"

    @pytest.mark.parametrize(
        "hole",
        ["q_lower", "q_upper", "qdot_max", "qddot_max", "torque_max", "max_tcp_speed"],
    )
    def test_simulation_missing_single_field_named(
        self, tmp_path: Path, hole: str
    ) -> None:
        """完整参数挖掉单项 → 消息恰好点名该缺项。"""
        profile = _rm65_style_profile(tmp_path)
        params = replace(_build_full_six_dof_params(profile), **{hole: None})

        with pytest.raises(ValueError, match=hole):
            validate_runtime_parameters(params, profile, purpose="simulation")

    def test_simulation_accepts_complete_parameters(self, tmp_path: Path) -> None:
        """完整限值参数通过 simulation 预检。"""
        profile = _rm65_style_profile(tmp_path)
        validate_runtime_parameters(
            _build_full_six_dof_params(profile), profile, purpose="simulation"
        )


class TestManualAndYamlShareValidation:
    """手构对象与 YAML 加载共用同一套校验规则。"""

    def test_manual_and_yaml_rejected_by_same_rule(self, tmp_path: Path) -> None:
        """init_q 越界在手构与 YAML 两条路径上均被同一规则拒绝。"""
        profile = _d2_profile(tmp_path)

        # 手构路径
        manual = _build_d2_params(
            init_q=[9.9] * 7, q_lower=[-3.0] * 7, q_upper=[3.0] * 7
        )
        with pytest.raises(ValueError, match="init_q"):
            validate_runtime_parameters(manual, profile, purpose="kinematics")

        # YAML 路径（load 内部执行 kinematics 校验，直接失败）
        data = _d2_runtime_dict()
        data["init_q"] = [9.9] * 7
        data["q_lower"] = [-3.0] * 7
        data["q_upper"] = [3.0] * 7
        yaml_path = _write_runtime(tmp_path, data)
        with pytest.raises(ValueError) as exc_info:
            load_runtime_parameters(yaml_path, profile)
        message = str(exc_info.value)
        assert "init_q" in message
        assert str(yaml_path) in message

    def test_yaml_roundtrip_matches_manual_defaults(self, tmp_path: Path) -> None:
        """同构 YAML 加载结果与手构默认值字段一致。"""
        profile = _d2_profile(tmp_path)
        yaml_path = _write_runtime(tmp_path, _d2_runtime_dict())
        from_yaml = load_runtime_parameters(yaml_path, profile)
        manual = _build_d2_params()

        assert from_yaml.robot_id == manual.robot_id
        assert from_yaml.joint_names == manual.joint_names
        assert from_yaml.dt == manual.dt
        np.testing.assert_allclose(from_yaml.init_q, manual.init_q)
        assert from_yaml.provisional_fields == manual.provisional_fields


class TestArrayImmutability:
    """数组独立复制且只读；frozen 数据类不可字段赋值。"""

    def test_arrays_do_not_alias_inputs(self) -> None:
        """构造后修改输入数组不影响内部副本，内部数组 writeable=False。"""
        init_q_input = np.array([0.0, 1.2, 0.0, -1.2, 0.0, 0.0, 0.0])
        kp_input = np.array([200.0] * 7)
        base_pos_input = np.array([5.0, 5.0, 5.0])
        scene = _build_scene(base_position=base_pos_input)

        params = _build_d2_params(
            scene=scene, init_q=init_q_input, kp=kp_input, kd=[10.0] * 7
        )

        # 篡改输入数组 → 内部副本不受影响（独立复制）
        init_q_input[0] = 99.0
        kp_input[0] = -1.0
        base_pos_input[0] = 99.0
        assert params.init_q[0] == pytest.approx(0.0)
        assert params.kp is not None and params.kp[0] == pytest.approx(200.0)
        assert params.scene is not None
        assert params.scene.base_position[0] == pytest.approx(5.0)

        # 内部数组只读
        for arr in (
            params.init_q,
            params.held_q,
            params.kp,
            params.kd,
            params.scene.base_position,
            params.scene.base_quaternion,
            params.scene.torso_half_size,
        ):
            assert arr is not None
            assert arr.dtype == np.float64
            assert arr.flags.writeable is False

        with pytest.raises(ValueError):
            params.init_q[0] = 1.0  # 只读数组赋值报错

        with pytest.raises(AttributeError):
            params.dt = 0.01  # type: ignore[misc] — frozen 拒绝字段赋值

    def test_types_are_exported(self) -> None:
        """公共类型可从模块直接导入。"""
        assert RobotRuntimeParameters is not None
        assert SceneParameters is not None


class TestPurity:
    """模块导入纯度：不拉入 real/sim/ilqt/MuJoCo/SDK。"""

    def test_module_import_has_no_forbidden_dependencies(self) -> None:
        """子进程导入模块后 sys.modules 不含被禁顶层模块。"""
        code = (
            "import os, sys\n"
            "sys.path.insert(0, os.environ['STAGE2_REPO_ROOT'])\n"
            "import src.robot.runtime_parameters\n"
            "banned = ('mujoco', 'src.real', 'src.sim', 'src.ilqt', 'rmcl')\n"
            "bad = [m for m in sys.modules\n"
            "       if any(m == b or m.startswith(b + '.') for b in banned)]\n"
            "print('BAD:' + ','.join(bad))\n"
            "sys.exit(1 if bad else 0)\n"
        )
        env = {**os.environ, "PYTHONUTF8": "1", "STAGE2_REPO_ROOT": str(REPO_ROOT)}
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, (
            f"意外副作用: {result.stdout}\n{result.stderr}"
        )
