"""RobotProfile 机器人描述加载器测试。

验证 profile 加载、字段校验、错误诊断、cwd 无关性与模块纯度
（import 期不加载 MuJoCo / 真机 SDK）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from src.robot.robot_profile import (
    RobotProfile,
    default_rm65_profile_path,
    load_robot_profile,
)

# 仓库根目录（tests/ 的父目录）
REPO_ROOT = Path(__file__).resolve().parent.parent


def _base_profile() -> dict[str, Any]:
    """构造一份最小合法的描述字典（供拒绝测试做单字段变异）。

    注：model.xml 由使用方按需创建；TestRejection 的字段变异在
    model_path 存在性检查之前触发，无需真实文件。
    """
    return {
        "schema_version": 1,
        "robot_id": "testbot",
        "model_path": "model.xml",
        "active_joint_names": ["j1", "j2"],
        "actuator_names": ["a1", "a2"],
        "held_joint_names": ["h1"],
        "tcp_site_name": "tcp",
    }


def _write_profile(
    directory: Path, data: dict[str, Any], filename: str = "robot.yaml"
) -> Path:
    """把描述字典写成 YAML 文件并返回路径。"""
    path = directory / filename
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


class TestLoadRm65Profile:
    """默认 RM65 描述的正确性。"""

    def test_load_rm65_profile(self) -> None:
        """加载 configs/robots/rm65.yaml：ndof=6、名称顺序正确、模型路径为仓库绝对路径。"""
        profile = load_robot_profile(default_rm65_profile_path())

        assert profile.schema_version == 1
        assert profile.robot_id == "rm65"
        assert profile.ndof == 6
        assert profile.active_joint_names == tuple(f"r_joint{i}" for i in range(1, 7))
        assert profile.actuator_names == tuple(f"torque_r_joint{i}" for i in range(1, 7))
        assert profile.held_joint_names == tuple(f"l_joint{i}" for i in range(1, 7))
        assert profile.tcp_site_name == "racket_center"

        expected_model = (REPO_ROOT / "src" / "robot" / "rm65_model.xml").resolve()
        assert profile.model_path == expected_model
        assert profile.model_path.is_absolute()
        assert profile.model_path.exists()

    def test_default_path_points_to_repo_yaml(self) -> None:
        """默认路径由模块文件位置定位仓库，指向 configs/robots/rm65.yaml。"""
        assert default_rm65_profile_path() == (
            REPO_ROOT / "configs" / "robots" / "rm65.yaml"
        ).resolve()


class TestModelPathResolution:
    """model_path 相对 YAML 父目录解析与存在性校验。"""

    def test_model_path_resolved_relative_to_yaml_parent(self, tmp_path: Path) -> None:
        """相对 model_path 以 YAML 所在目录为基准解析为绝对路径。"""
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "model.xml").touch()  # 模型文件需存在（存在性校验）
        yaml_path = _write_profile(tmp_path / "sub", _base_profile())
        profile = load_robot_profile(yaml_path)
        assert profile.model_path == (tmp_path / "sub" / "model.xml").resolve()

    def test_missing_model_file_rejected(self, tmp_path: Path) -> None:
        """model_path 指向不存在的文件 → FileNotFoundError，消息含两个路径。"""
        yaml_path = _write_profile(tmp_path, _base_profile())  # model.xml 未创建
        with pytest.raises(FileNotFoundError) as exc_info:
            load_robot_profile(yaml_path)
        message = str(exc_info.value)
        assert "model.xml" in message
        assert str(yaml_path) in message

    def test_directory_model_path_rejected(self, tmp_path: Path) -> None:
        """model_path 指向目录（非常规文件）→ FileNotFoundError，而非 PermissionError。"""
        data = _base_profile()
        data["model_path"] = "."  # YAML 所在目录本身
        yaml_path = _write_profile(tmp_path, data)
        with pytest.raises(FileNotFoundError, match="model_path"):
            load_robot_profile(yaml_path)


class TestRejection:
    """非法描述必须报错且错误消息可定位字段与文件。"""

    @pytest.mark.parametrize(
        "mutate,field",
        [
            (lambda d: d.update(active_joint_names=["j1", "j1"]), "active_joint_names"),
            (lambda d: d.update(held_joint_names=["j2"]), "held_joint_names"),
            (lambda d: d.update(active_joint_names=[]), "active_joint_names"),
            (lambda d: d.update(actuator_names=["a1"]), "actuator_names"),
            (lambda d: d.update(actuator_names=["a1", "a1"]), "actuator_names"),
            (lambda d: d.update(tcp_site_name=""), "tcp_site_name"),
            (lambda d: d.update(schema_version=2), "schema_version"),
            (lambda d: d.update(unknown_field=1), "unknown_field"),
            (lambda d: d.update(model_path=""), "model_path"),
            (lambda d: d.update(robot_id=""), "robot_id"),
        ],
        ids=[
            "重复主动关节",
            "主动保持交集",
            "空主动列表",
            "执行器长度不匹配",
            "执行器重复",
            "空TCP",
            "版本非1",
            "未知字段",
            "空模型路径",
            "空机器人ID",
        ],
    )
    def test_invalid_profile_rejected(
        self, tmp_path: Path, mutate: Callable[[dict[str, Any]], None], field: str
    ) -> None:
        """各非法字段变异均报 ValueError，消息含字段名与文件路径。"""
        data = _base_profile()
        mutate(data)
        yaml_path = _write_profile(tmp_path, data)

        with pytest.raises(ValueError) as exc_info:
            load_robot_profile(yaml_path)
        message = str(exc_info.value)
        assert field in message
        assert str(yaml_path) in message

    def test_mixed_type_unknown_keys_rejected(self, tmp_path: Path) -> None:
        """顶层混合 int/str 未知键 → ValueError（而非 sorted 的 TypeError）。"""
        data: dict[str, Any] = _base_profile()
        data[1] = "int 键"  # type: ignore[assignment] — 故意构造非法键类型
        data["extra_str_key"] = "str 键"
        yaml_path = _write_profile(tmp_path, data)

        with pytest.raises(ValueError) as exc_info:
            load_robot_profile(yaml_path)
        message = str(exc_info.value)
        assert "extra_str_key" in message
        assert str(yaml_path) in message

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """描述文件不存在保留 FileNotFoundError，不转为 ValueError。"""
        with pytest.raises(FileNotFoundError):
            load_robot_profile(tmp_path / "no_such.yaml")


class TestPurityAndIndependence:
    """加载器纯度（无 MuJoCo/SDK 副作用）与 cwd 无关性。"""

    def test_profile_path_is_cwd_independent(self, tmp_path: Path, monkeypatch) -> None:
        """切换工作目录后默认路径与加载结果不变。"""
        monkeypatch.chdir(tmp_path)
        assert default_rm65_profile_path() == (
            REPO_ROOT / "configs" / "robots" / "rm65.yaml"
        ).resolve()
        profile = load_robot_profile(default_rm65_profile_path())
        assert profile.robot_id == "rm65"

    def test_load_from_chinese_path(self, tmp_path: Path) -> None:
        """从含非 ASCII（中文）的目录加载描述成功。"""
        cn_dir = tmp_path / "中文目录"
        cn_dir.mkdir()
        (cn_dir / "model.xml").touch()  # 模型文件需存在（存在性校验）
        yaml_path = _write_profile(cn_dir, _base_profile())
        profile = load_robot_profile(yaml_path)
        assert profile.robot_id == "testbot"
        assert profile.model_path == (cn_dir / "model.xml").resolve()

    def test_module_import_has_no_mujoco_or_real_side_effect(self) -> None:
        """子进程导入模块后 sys.modules 不含 mujoco 与 src.real。"""
        code = (
            "import os, sys\n"
            "sys.path.insert(0, os.environ['STAGE1_REPO_ROOT'])\n"
            "import src.robot.robot_profile\n"
            "bad = [m for m in sys.modules\n"
            "       if m == 'mujoco' or m.startswith('mujoco.')\n"
            "       or m == 'src.real' or m.startswith('src.real.')]\n"
            "print('BAD:' + ','.join(bad))\n"
            "sys.exit(1 if bad else 0)\n"
        )
        env = {**os.environ, "PYTHONUTF8": "1", "STAGE1_REPO_ROOT": str(REPO_ROOT)}
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, f"意外副作用: {result.stdout}\n{result.stderr}"


class TestImmutability:
    """描述数据不可被外部别名修改。"""

    def test_profile_is_frozen_with_tuple_fields(self) -> None:
        """frozen dataclass + tuple：属性赋值报错，集合字段为 tuple。"""
        profile = load_robot_profile(default_rm65_profile_path())

        with pytest.raises(AttributeError):
            profile.robot_id = "other"  # type: ignore[misc]

        assert isinstance(profile.active_joint_names, tuple)
        assert isinstance(profile.actuator_names, tuple)
        assert isinstance(profile.held_joint_names, tuple)

    def test_robot_profile_type_exported(self) -> None:
        """RobotProfile 可从模块直接导入（供类型标注使用）。"""
        assert RobotProfile is not None
