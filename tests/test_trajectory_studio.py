"""trajectory_studio 纯函数测试。

仅测试不涉及 IO/交互的纯函数：扫描、路径长度、列表格式化、seed 提取、匹配。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 添加 scripts/ 到路径
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from trajectory_studio import (  # noqa: E402
    _format_list_line,
    _path_length,
    _scan_trajectories,
    _speed_warning_message,
)
from src.real.trajectory_types import ReplayTrajectory  # noqa: E402


class TestScanTrajectories:
    """轨迹扫描分类测试。"""

    def test_empty_dir(self, tmp_path: Path) -> None:
        """空目录返回空列表。"""
        planned, real = _scan_trajectories(tmp_path)
        assert planned == []
        assert real == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        """不存在的目录返回空列表。"""
        planned, real = _scan_trajectories(tmp_path / "nonexistent")
        assert planned == []
        assert real == []

    def test_classification(self, tmp_path: Path) -> None:
        """正确分类规划轨迹和真机录制。"""
        (tmp_path / "stage0_traj_s1.npz").touch()
        (tmp_path / "stage0_traj_swing_s21.npz").touch()
        (tmp_path / "stage3_real_swing_s21_speed050.npz").touch()
        (tmp_path / "not_a_trajectory.txt").touch()  # 非 .npz 忽略

        planned, real = _scan_trajectories(tmp_path)
        assert len(planned) == 2
        assert len(real) == 1
        assert planned[0][0] == "stage0_traj_s1.npz"
        assert real[0][0] == "stage3_real_swing_s21_speed050.npz"

    def test_sorted(self, tmp_path: Path) -> None:
        """结果按文件名排序。"""
        (tmp_path / "stage0_traj_s42.npz").touch()
        (tmp_path / "stage0_traj_s1.npz").touch()
        (tmp_path / "stage0_traj_s7.npz").touch()

        planned, _ = _scan_trajectories(tmp_path)
        names = [n for n, _ in planned]
        assert names == sorted(names)


class TestPathLength:
    """TCP 路径长度计算测试。"""

    def test_zero(self) -> None:
        """空/单点 → 0。"""
        assert _path_length(np.zeros((0, 3))) == 0.0
        assert _path_length(np.zeros((1, 3))) == 0.0

    def test_straight_line(self) -> None:
        """直线运动 → 长度 = 距离。"""
        tcp = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        assert _path_length(tcp) == pytest.approx(2.0)

    def test_zigzag(self) -> None:
        """Z 字运动 → 三段距离之和。"""
        tcp = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        assert _path_length(tcp) == pytest.approx(3.0)


class TestFormatListLine:
    """列表行格式化测试。"""

    def test_planned(self) -> None:
        """规划轨迹行格式正确。"""
        traj = ReplayTrajectory(
            q_desired=np.zeros((131, 6)),
            q_actual=np.zeros((131, 6)),
            timestamps=np.arange(131) * 0.005,
            tcp_pos=np.zeros((131, 3)),
            ball_pos=np.zeros((131, 3)),
            init_q=np.zeros(6),
            init_q_left=np.zeros(6),
            dt=0.005,
            hit_step=65,
        )
        line = _format_list_line("stage0_traj_s1.npz", traj)
        assert "131" in line
        assert "hit@65" in line

    def test_no_hit(self) -> None:
        """hit_step=-1 → no-hit。"""
        traj = ReplayTrajectory(
            q_desired=np.zeros((10, 6)),
            q_actual=np.zeros((10, 6)),
            timestamps=np.arange(10) * 0.005,
            tcp_pos=np.zeros((10, 3)),
            ball_pos=np.zeros((10, 3)),
            init_q=np.zeros(6),
            init_q_left=np.zeros(6),
            dt=0.005,
            hit_step=-1,
        )
        line = _format_list_line("test.npz", traj)
        assert "no-hit" in line

    def test_real(self) -> None:
        """真机录制行含 speed。"""
        traj = ReplayTrajectory(
            q_desired=np.zeros((559, 6)),
            q_actual=np.zeros((559, 6)),
            timestamps=np.arange(559) * 0.005,
            tcp_pos=np.zeros((559, 3)),
            ball_pos=np.zeros((559, 3)),
            init_q=np.zeros(6),
            init_q_left=np.zeros(6),
            dt=0.005,
            hit_step=-1,
            metadata={"speed": 0.5},
        )
        line = _format_list_line("stage3_real.npz", traj, is_real=True)
        assert "559" in line
        assert "speed=0.5" in line


class TestSpeedWarning:
    """I1: _speed_warning_message 严格阈值告警测试。

    覆盖决策点（IO 交互在 _do_replay 内，难测，靠纯函数覆盖）。
    """

    def test_no_warning_when_speed_at_recommendation(self) -> None:
        """speed == rec_speed 不告警（边界含端点）。"""
        # peak_tcp=2.0, firmware_tcp=1.0 → rec_speed=0.5
        msg = _speed_warning_message(speed=0.5, rec_speed=0.5, peak_tcp=2.0, firmware_tcp=1.0)
        assert msg is None

    def test_warning_when_speed_exceeds(self) -> None:
        """speed > rec_speed 触发告警。"""
        msg = _speed_warning_message(speed=0.6, rec_speed=0.5, peak_tcp=2.0, firmware_tcp=1.0)
        assert msg is not None
        assert "预测 TCP 峰值" in msg
        # predicted = 2.0 * 0.6 = 1.2
        assert "1.20" in msg

    def test_no_warning_for_zero_peak(self) -> None:
        """peak_tcp=0 → rec_speed=inf（除零保护），不告警。"""
        # 当 peak_tcp=0，check_tcp_speed 通常返回 rec_speed=inf 或很大值
        # 函数应安全处理（rec_speed > 0 时 speed <= rec_speed 不告警）
        msg = _speed_warning_message(speed=0.5, rec_speed=1e10, peak_tcp=0.0, firmware_tcp=1.0)
        assert msg is None

    def test_no_warning_when_rec_speed_nonpositive(self) -> None:
        """rec_speed <= 0 时跳过（防御非法输入，不告警）。"""
        msg = _speed_warning_message(speed=0.5, rec_speed=0.0, peak_tcp=2.0, firmware_tcp=1.0)
        assert msg is None
        msg_neg = _speed_warning_message(speed=0.5, rec_speed=-1.0, peak_tcp=2.0, firmware_tcp=1.0)
        assert msg_neg is None

    def test_strict_inequality_boundary(self) -> None:
        """严格 > 而非 >=：speed 刚刚超过 rec_speed 即触发。"""
        from math import nextafter
        # speed 比 rec_speed 大 1 ULP（最小浮点增量）
        rec_speed = 0.5
        speed_slightly_above = nextafter(rec_speed, float("inf"))
        msg = _speed_warning_message(
            speed=speed_slightly_above, rec_speed=rec_speed,
            peak_tcp=2.0, firmware_tcp=1.0,
        )
        assert msg is not None

        # speed 比 rec_speed 小 1 ULP：不触发
        speed_slightly_below = nextafter(rec_speed, -float("inf"))
        msg_below = _speed_warning_message(
            speed=speed_slightly_below, rec_speed=rec_speed,
            peak_tcp=2.0, firmware_tcp=1.0,
        )
        assert msg_below is None


class TestUtf8StdoutSafety:
    """I3 回归测试：中文 Windows GBK 控制台无法编码 ✗⚠→ 等符号。

    验证 _safety_card 输出的字符串可被 UTF-8 编码（间接验证 main 启动时的
    TextIOWrapper 包装生效）。完整端到端验证需在 GBK 终端跑 main()，
    此处仅锁住"输出字符串本身可编码"这一基础契约。
    """

    def test_safety_card_output_encodes_utf8(self, tmp_path: Path) -> None:
        """_safety_card 返回的所有行必须可 UTF-8 编码。"""
        from trajectory_studio import _safety_card
        from src.real.trajectory_types import ReplayTrajectory

        # 构造一个会触发超限告警的轨迹（J6 超上限）
        n = 10
        q = np.zeros((n, 6))
        q[:, 5] = np.radians(200)  # J6 远超 180°
        tcp_pos = np.zeros((n, 3))
        tcp_pos[:, 0] = np.arange(n) * 0.3  # TCP 高速

        traj = ReplayTrajectory(
            q_desired=q.copy(),
            q_actual=q.copy(),
            timestamps=np.arange(n) * 0.005,
            tcp_pos=tcp_pos,
            ball_pos=np.zeros((n, 3)),
            init_q=np.zeros(6),
            init_q_left=np.zeros(6),
            dt=0.005,
            hit_step=-1,
            metadata={"is_position_mode": True},
        )

        lines, _, _, _ = _safety_card(traj, q, config_path=None)
        assert len(lines) > 0
        # 每行都必须可 UTF-8 编码（GBK 终端场景下若已包装则不会崩）
        for line in lines:
            line.encode("utf-8", errors="strict")  # 不抛 UnicodeEncodeError


