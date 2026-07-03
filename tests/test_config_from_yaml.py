"""RealRobotConfig.from_yaml 解析测试。

验证 YAML 嵌套 section 的短键名（ip/port）经别名映射后正确填入 dataclass，
以及顶层非 section 键（如 joint_zero_offset）不被遗漏。
"""

import textwrap
from pathlib import Path

import numpy as np

from src.real.config import RealRobotConfig


def _write_tmp_yaml(tmp_path: Path, content: str) -> Path:
    """写入临时 YAML 文件并返回路径。"""
    p = tmp_path / "test_config.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


class TestFromYamlAlias:
    """YAML section 内短键名 → dataclass 字段名的别名映射。"""

    def test_robot_ip_from_yaml_ip(self, tmp_path: Path) -> None:
        """YAML robot.ip 应映射到 robot_ip（而非用默认 .18）。"""
        yaml_path = _write_tmp_yaml(tmp_path, """
            robot:
              ip: "192.168.1.99"
              port: 9999
        """)
        cfg = RealRobotConfig.from_yaml(yaml_path)
        assert cfg.robot_ip == "192.168.1.99"
        assert cfg.robot_port == 9999

    def test_default_ip_when_not_specified(self, tmp_path: Path) -> None:
        """不指定 ip 时使用 dataclass 默认值。"""
        yaml_path = _write_tmp_yaml(tmp_path, """
            control:
              dt: 0.005
        """)
        cfg = RealRobotConfig.from_yaml(yaml_path)
        assert cfg.robot_ip == "192.168.1.18"

    def test_port_alias(self, tmp_path: Path) -> None:
        """YAML robot.port 应映射到 robot_port。"""
        yaml_path = _write_tmp_yaml(tmp_path, """
            robot:
              ip: "10.0.0.1"
              port: 1234
        """)
        cfg = RealRobotConfig.from_yaml(yaml_path)
        assert cfg.robot_port == 1234


class TestFromYamlTopLevel:
    """顶层（非 section）键的解析。"""

    def test_toplevel_joint_zero_offset(self, tmp_path: Path) -> None:
        """顶层 joint_zero_offset 列表应被正确读取（非零值）。"""
        yaml_path = _write_tmp_yaml(tmp_path, """
            joint_zero_offset: [0.01, -0.02, 0.03, 0.0, 0.0, 0.0]
        """)
        cfg = RealRobotConfig.from_yaml(yaml_path)
        expected = np.array([0.01, -0.02, 0.03, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(cfg.joint_zero_offset, expected)

    def test_toplevel_scalar_ignored_if_unknown(self, tmp_path: Path) -> None:
        """未知的顶层键应被静默忽略。"""
        yaml_path = _write_tmp_yaml(tmp_path, """
            some_unknown_key: 42
        """)
        cfg = RealRobotConfig.from_yaml(yaml_path)  # 不应抛异常
        assert cfg.robot_ip == "192.168.1.18"


class TestFromYamlNestedSections:
    """嵌套 section 内已知字段的解析（回归测试）。"""

    def test_safety_section(self, tmp_path: Path) -> None:
        """safety section 内的字段应正确解析。"""
        yaml_path = _write_tmp_yaml(tmp_path, """
            safety:
              collision_stage: 3
              max_tcp_speed: 0.5
        """)
        cfg = RealRobotConfig.from_yaml(yaml_path)
        assert cfg.collision_stage == 3
        assert cfg.max_tcp_speed == 0.5

    def test_degree_conversion(self, tmp_path: Path) -> None:
        """q_lower/q_upper 应从度转为弧度。"""
        yaml_path = _write_tmp_yaml(tmp_path, """
            safety:
              q_lower: [-180, -90, 0, -180, -115, -180]
              q_upper: [180, 90, 150, 180, 115, 180]
        """)
        cfg = RealRobotConfig.from_yaml(yaml_path)
        assert cfg.q_lower[0] == np.radians(-180.0)
        assert cfg.q_upper[1] == np.radians(90.0)


class TestFromYamlRealConfig:
    """对项目实际 YAML 配置文件的解析验证。"""

    def test_real_robot_test_yaml_ip(self) -> None:
        """configs/real_robot_test.yaml 的 robot.ip 应正确解析（非默认 .18）。"""
        yaml_path = Path("configs/real_robot_test.yaml")
        if not yaml_path.exists():
            return  # 无该文件时跳过
        cfg = RealRobotConfig.from_yaml(yaml_path)
        # real_robot_test.yaml 中 robot.ip 设为 .19
        assert cfg.robot_ip == "192.168.1.19", (
            f"期望 192.168.1.19（YAML robot.ip），实际 {cfg.robot_ip}（别名映射未生效）"
        )


class TestConfigNoCanfdFields:
    """canfd 相关字段已从 RealRobotConfig 移除。"""

    def test_config_has_no_canfd_fields(self) -> None:
        """RealRobotConfig 不含 control_mode/canfd_* 字段（速度透传已排除）。"""
        from dataclasses import fields as dc_fields

        names = {f.name for f in dc_fields(RealRobotConfig)}
        assert "control_mode" not in names
        assert "canfd_trajectory_mode" not in names
        assert "canfd_smooth_radio" not in names
