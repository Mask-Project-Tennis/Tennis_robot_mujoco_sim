"""RacketProfile 球拍参数加载器测试（T1）。

本测试文件是 R1D-V12 parity 任务 T1 的验收依据，覆盖：

* 旧拍 ``configs/rackets/rm65_legacy.yaml`` 与 ``src/robot/rm65_model.xml`` 的独立一致性：
  测试自行解析 XML 并独立换算坐标/惯量，**不调用被测模块的生成逻辑**；
* 参考拍 ``configs/rackets/blade100_v9_reference.yaml`` 的官方规格（Wilson Blade 100 V9）
  与估计项来源标注（estimated + method，不得伪称实测）；
* 非法输入拒绝：未知字段、缺字段、非正质量、非正定/违三角不等式惯量、非单位四元数、
  拍面中心越界、TCP 与拍面中心不一致、拍面面积不自洽等，均必须抛错，
  **禁止静默重置为默认值**；
* 坐标/单位契约：SI 单位、柄底原点、+Z 长轴指向拍头、+X 拍面法向、+Y 右手系、
  TCP = 拍面几何中心、法兰→柄底安装变换。

坐标约定：球拍局部系原点在柄底（手柄末端），+Z 沿长轴指向拍头，+X 为拍面法向，
+Y 由右手系确定。安装变换 ``mount`` 表示 ``p_flange = R(wxyz) @ p_local + position``。
"""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src.robot.racket_profile import (
    LEGACY_RACKET_PROFILE_PATH,
    RACKET_CONFIG_DIR,
    REFERENCE_RACKET_PROFILE_PATH,
    RacketProfile,
    load_racket_profile,
    quat_wxyz_to_rotmat,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RM65_MODEL_PATH = REPO_ROOT / "src" / "robot" / "rm65_model.xml"


# --------------------------------------------------------------------------------------
# 测试内部独立工具（刻意不复用被测模块的实现）
# --------------------------------------------------------------------------------------


def _quat_wxyz_to_rotmat_local(wxyz: list[float]) -> np.ndarray:
    """测试内部独立实现：四元数 (w, x, y, z) → 3×3 旋转矩阵。"""
    w, x, y, z = wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _parse_floats(text: str) -> list[float]:
    """把 MuJoCo XML 中以空格分隔的浮点串解析为 list。"""
    return [float(value) for value in text.split()]


def _xml_legacy_expected() -> dict[str, Any]:
    """从 ``src/robot/rm65_model.xml`` 独立解析旧拍并推导期望值。

    XML 中 ``r_racket_body`` 挂载在 ``r_flange`` 下，其自身坐标系（+Z 长轴、+X 法向）
    即为球拍局部系，但原点在法兰处而非柄底。因此需要：

    1. 用 body 的 quat 得到 ``R_body``（body → flange）；
    2. 用胶囊下端 tip 求出柄底在 body 系中的 z 偏移 ``butt_z``；
    3. 局部系坐标 = body 坐标 - (0, 0, butt_z)，安装平移 = ``R_body @ (0, 0, butt_z)``。
    """
    root = ET.parse(RM65_MODEL_PATH).getroot()
    body = root.find(".//body[@name='r_racket_body']")
    if body is None:  # pragma: no cover - 仅防御，正常 XML 必存在
        raise AssertionError("rm65_model.xml 中缺少 r_racket_body")

    quat = _parse_floats(body.attrib["quat"])
    rot_body_to_flange = _quat_wxyz_to_rotmat_local(quat)
    body_pos = np.array(_parse_floats(body.attrib.get("pos", "0 0 0")), dtype=np.float64)

    inertial = body.find("inertial")
    if inertial is None:  # pragma: no cover
        raise AssertionError("r_racket_body 缺少 inertial")
    mass = float(inertial.attrib["mass"])
    com_body_z = _parse_floats(inertial.attrib["pos"])[2]
    diaginertia = _parse_floats(inertial.attrib["diaginertia"])

    handle = body.find("geom[@name='r_racket_handle']")
    face = body.find("geom[@name='r_racket_face']")
    site = body.find("site[@name='racket_center']")
    if handle is None or face is None or site is None:  # pragma: no cover
        raise AssertionError("r_racket_body 缺少球拍几何或 TCP site")

    handle_fromto = _parse_floats(handle.attrib["fromto"])
    handle_radius = _parse_floats(handle.attrib["size"])[0]
    face_pos = _parse_floats(face.attrib["pos"])
    face_size = _parse_floats(face.attrib["size"])
    tcp_pos = _parse_floats(site.attrib["pos"])

    butt_body_z = handle_fromto[2] - handle_radius
    tcp_body = np.array(tcp_pos, dtype=np.float64)

    return {
        "rot_body_to_flange": rot_body_to_flange,
        "body_pos": body_pos,
        "mass": mass,
        "diaginertia": np.array(diaginertia, dtype=np.float64),
        "com_body": np.array([0.0, 0.0, com_body_z], dtype=np.float64),
        "butt_body_z": butt_body_z,
        "handle_start_body": np.array(handle_fromto[0:3], dtype=np.float64),
        "handle_end_body": np.array(handle_fromto[3:6], dtype=np.float64),
        "handle_radius": handle_radius,
        "face_center_body": np.array(face_pos, dtype=np.float64),
        "face_semi_axes": np.array(face_size, dtype=np.float64),
        "tcp_body": tcp_body,
        "length": face_pos[2] + face_size[2] - butt_body_z,
        "face_area": math.pi * face_size[1] * face_size[2],
    }


def _base_profile() -> dict[str, Any]:
    """构造一份合成的合法球拍字典，供非法输入用例做深拷贝后定向变异。"""
    length = 0.5
    half_len = 0.14
    tcp_z = length - half_len
    base: dict[str, Any] = {
        "schema_version": 1,
        "racket_id": "synthetic_test_racket",
        "description": "合成球拍，仅用于校验用例",
        "geometry": {
            "length": length,
            "tcp": [0.0, 0.0, tcp_z],
            "face_area": math.pi * 0.1 * half_len,
            "face": {
                "shape": "ellipse",
                "center": [0.0, 0.0, tcp_z],
                "semi_axes": [0.005, 0.1, half_len],
            },
            "handle": {
                "shape": "capsule",
                "start": [0.0, 0.0, 0.01],
                "end": [0.0, 0.0, 0.15],
                "radius": 0.01,
            },
            "throat": [],
            "frame": [],
        },
        "mass": 0.3,
        "com": [0.0, 0.0, 0.25],
        "inertia": {"matrix": [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.001]]},
        "mount": {"position": [0.0, 0.0, 0.0], "wxyz": [1.0, 0.0, 0.0, 0.0]},
        "connector": None,
        "provenance": {
            "geometry.length": {
                "source": "estimated",
                "units": "m",
                "reference": None,
                "method": "合成测试值",
            },
            "geometry.tcp": {
                "source": "estimated",
                "units": "m",
                "reference": None,
                "method": "合成测试值：拍面几何中心",
            },
            "geometry.face": {
                "source": "estimated",
                "units": "m",
                "reference": None,
                "method": "合成测试值",
            },
            "geometry.face_area": {
                "source": "estimated",
                "units": "m^2",
                "reference": None,
                "method": "合成测试值: pi * 半宽 * 半长",
            },
            "geometry.handle": {
                "source": "estimated",
                "units": "m",
                "reference": None,
                "method": "合成测试值",
            },
            "mass": {
                "source": "estimated",
                "units": "kg",
                "reference": None,
                "method": "合成测试值",
            },
            "com": {
                "source": "estimated",
                "units": "m",
                "reference": None,
                "method": "合成测试值",
            },
            "inertia": {
                "source": "estimated",
                "units": "kg·m^2",
                "reference": None,
                "method": "合成测试值：均质近似",
            },
            "mount.position": {
                "source": "estimated",
                "units": "m",
                "reference": None,
                "method": "合成测试值",
            },
            "mount.wxyz": {
                "source": "estimated",
                "units": "1",
                "reference": None,
                "method": "合成测试值",
            },
        },
    }
    return base


def _write_profile(tmp_path: Path, data: dict[str, Any], name: str = "racket.yaml") -> Path:
    """把球拍字典写入临时 YAML 文件并返回路径。"""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _load_mutated(tmp_path: Path, mutate: Any) -> RacketProfile:
    """深拷贝合成配置 → 执行变异 → 写临时文件 → 加载。"""
    data = copy.deepcopy(_base_profile())
    mutate(data)
    return load_racket_profile(_write_profile(tmp_path, data))


# --------------------------------------------------------------------------------------
# 模块级常量与工具函数
# --------------------------------------------------------------------------------------


class TestModuleConstants:
    """模块级路径常量与四元数工具函数。"""

    def test_config_dir_and_profile_paths_exist(self) -> None:
        """配置文件目录与两个球拍档案路径必须真实存在。"""
        assert RACKET_CONFIG_DIR.is_dir()
        assert LEGACY_RACKET_PROFILE_PATH.is_file()
        assert REFERENCE_RACKET_PROFILE_PATH.is_file()
        assert RACKET_CONFIG_DIR == REPO_ROOT / "configs" / "rackets"

    def test_quat_wxyz_to_rotmat_is_orthonormal(self) -> None:
        """四元数 → 旋转矩阵工具必须返回正交且行列式为 +1 的矩阵。"""
        quat = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]
        rot = quat_wxyz_to_rotmat(quat)
        assert rot.shape == (3, 3)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        assert math.isclose(float(np.linalg.det(rot)), 1.0, abs_tol=1e-12)
        # 绕 Z 轴 90°：局部 +X 映射到法兰 +Y，局部 +Z 仍为法兰 +Z
        assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
        assert np.allclose(rot @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-12)


# --------------------------------------------------------------------------------------
# 旧拍（rm65_legacy.yaml）与 rm65_model.xml 的独立一致性
# --------------------------------------------------------------------------------------


class TestLegacyProfileMatchesSourceModel:
    """旧拍档案必须能由原始 XML 独立复算得到（测试自己算，不复用模块生成逻辑）。"""

    @pytest.fixture(scope="class")
    def expected(self) -> dict[str, Any]:
        """独立解析 XML 得到的期望值。"""
        return _xml_legacy_expected()

    @pytest.fixture(scope="class")
    def profile(self) -> RacketProfile:
        """加载旧拍档案。"""
        return load_racket_profile(LEGACY_RACKET_PROFILE_PATH)

    def test_identity_and_mass(
        self, profile: RacketProfile, expected: dict[str, Any]
    ) -> None:
        """racket_id 必须标明旧拍，质量必须等于 XML 的 0.35 kg（不是成人拍 0.316）。"""
        assert profile.racket_id == "rm65_legacy"
        assert profile.mass == pytest.approx(expected["mass"], abs=1e-12)
        assert profile.mass == pytest.approx(0.35, abs=1e-12)
        assert profile.mass != pytest.approx(0.316, abs=1e-6)

    def test_length_tcp_and_face_geometry(
        self, profile: RacketProfile, expected: dict[str, Any]
    ) -> None:
        """长度、TCP、拍面中心/半轴、拍面面积必须与 XML 几何一致。"""
        geometry = profile.geometry
        assert geometry.length == pytest.approx(expected["length"], abs=1e-12)
        assert np.allclose(
            geometry.tcp, expected["tcp_body"] - np.array([0.0, 0.0, expected["butt_body_z"]]),
            atol=1e-12,
        )
        assert np.allclose(
            geometry.face.center,
            expected["face_center_body"] - np.array([0.0, 0.0, expected["butt_body_z"]]),
            atol=1e-12,
        )
        assert np.allclose(geometry.face.semi_axes, expected["face_semi_axes"], atol=1e-12)
        assert geometry.face_area == pytest.approx(expected["face_area"], abs=1e-12)
        # TCP 与拍面几何中心同源
        assert geometry.tcp == geometry.face.center

    def test_handle_matches_xml_capsule(
        self, profile: RacketProfile, expected: dict[str, Any]
    ) -> None:
        """手柄胶囊 endpoints/半径必须与 XML fromto/size 一致（柄底为原点）。"""
        handle = profile.geometry.handle
        butt = expected["butt_body_z"]
        assert np.allclose(
            handle.start,
            expected["handle_start_body"] - np.array([0.0, 0.0, butt]),
            atol=1e-12,
        )
        assert np.allclose(
            handle.end,
            expected["handle_end_body"] - np.array([0.0, 0.0, butt]),
            atol=1e-12,
        )
        assert handle.radius == pytest.approx(expected["handle_radius"], abs=1e-12)

    def test_com_and_inertia_match_xml(
        self, profile: RacketProfile, expected: dict[str, Any]
    ) -> None:
        """质心与转动惯量必须与 XML inertial 一致（惯量为 COM 局部系）。"""
        butt = expected["butt_body_z"]
        assert np.allclose(
            profile.com, expected["com_body"] - np.array([0.0, 0.0, butt]), atol=1e-12
        )
        matrix = profile.inertia.as_matrix()
        assert matrix.shape == (3, 3)
        assert np.allclose(matrix, np.diag(expected["diaginertia"]), atol=1e-12)
        assert np.allclose(matrix, matrix.T, atol=1e-15)
        # diaginertia = (0.003, 0.001, 0.003) → 主惯量必须与之逐一对应
        assert np.allclose(
            np.sort(np.asarray(profile.inertia.principal_moments())),
            np.sort(expected["diaginertia"]),
            atol=1e-12,
        )

    def test_mount_reconstructs_xml_flange_frame(
        self, profile: RacketProfile, expected: dict[str, Any]
    ) -> None:
        """安装变换必须把局部系点映射回 XML 的 flange 系（独立对照旋转矩阵）。"""
        rot_mount = quat_wxyz_to_rotmat(profile.mount.wxyz)
        assert np.allclose(rot_mount, expected["rot_body_to_flange"], atol=1e-12)
        butt = expected["butt_body_z"]
        expected_position = expected["rot_body_to_flange"] @ np.array([0.0, 0.0, butt])
        assert np.allclose(
            profile.mount.position, expected_position + expected["body_pos"], atol=1e-12
        )

    def test_reconstructed_geometry_matches_xml_flange_endpoints(
        self, profile: RacketProfile, expected: dict[str, Any]
    ) -> None:
        """用安装变换重建手柄/拍面/TCP 的 flange 坐标，必须逐点等于 XML 端点。"""
        rot = quat_wxyz_to_rotmat(profile.mount.wxyz)
        position = np.asarray(profile.mount.position)

        def to_flange(local: Any) -> np.ndarray:
            """局部系点 → flange 系点。"""
            return rot @ np.asarray(local, dtype=np.float64) + position

        butt = expected["butt_body_z"]
        assert np.allclose(
            to_flange(profile.geometry.handle.start), expected["handle_start_body"],
            atol=1e-12,
        )
        assert np.allclose(
            to_flange(profile.geometry.handle.end), expected["handle_end_body"], atol=1e-12
        )
        assert np.allclose(
            to_flange(profile.geometry.tcp), expected["tcp_body"], atol=1e-12
        )
        # XML 中 body 系原点在局部系中的坐标为 (0, 0, -butt)（柄底在 body 系 z = +butt）
        assert np.allclose(
            to_flange(np.array([0.0, 0.0, -butt])), expected["body_pos"], atol=1e-12
        )

    def test_legacy_has_no_throat_frame_or_connector(self, profile: RacketProfile) -> None:
        """原 XML 没有独立喉部/框体/连接件，档案不得凭空生成。"""
        assert profile.geometry.throat == ()
        assert profile.geometry.frame == ()
        assert profile.connector is None

    def test_legacy_provenance_is_legacy_model(self, profile: RacketProfile) -> None:
        """旧拍所有字段来源为 legacy_model，且必须给出 XML 出处。"""
        required_keys = {
            "geometry.length",
            "geometry.tcp",
            "geometry.face",
            "geometry.face_area",
            "geometry.handle",
            "mass",
            "com",
            "inertia",
            "mount.position",
            "mount.wxyz",
        }
        assert set(profile.provenance) == required_keys
        for key, entry in profile.provenance.items():
            assert entry.source == "legacy_model", key
            assert entry.reference and "rm65_model.xml" in entry.reference, key
            assert entry.method, key
        # 单位表
        assert profile.provenance["geometry.length"].units == "m"
        assert profile.provenance["geometry.face_area"].units == "m^2"
        assert profile.provenance["mass"].units == "kg"
        assert profile.provenance["inertia"].units == "kg·m^2"
        assert profile.provenance["mount.wxyz"].units == "1"

    def test_profile_is_frozen_and_reports_source_path(self, profile: RacketProfile) -> None:
        """RacketProfile 必须冻结（不可变），并记录来源文件绝对路径。"""
        with pytest.raises(Exception):
            profile.mass = 0.3  # type: ignore[misc]
        assert profile.source_path == LEGACY_RACKET_PROFILE_PATH.resolve()
        assert profile.schema_version == 1


# --------------------------------------------------------------------------------------
# 参考拍（Wilson Blade 100 V9）官方规格与估计项标注
# --------------------------------------------------------------------------------------


class TestReferenceProfileSpecs:
    """参考成人拍档案必须对齐官方规格，估计项如实标注。"""

    @pytest.fixture(scope="class")
    def profile(self) -> RacketProfile:
        """加载参考拍档案。"""
        return load_racket_profile(REFERENCE_RACKET_PROFILE_PATH)

    def test_official_specs(self, profile: RacketProfile) -> None:
        """长度 685.8 mm、面积 0.0645 m²、穿线质量 316 g、平衡点 330 mm、框厚 22 mm。"""
        geometry = profile.geometry
        assert profile.racket_id == "blade100_v9_reference"
        assert geometry.length == pytest.approx(0.6858, abs=1e-9)
        assert geometry.face_area == pytest.approx(0.0645, rel=1e-6)
        assert profile.mass == pytest.approx(0.316, abs=1e-12)
        assert profile.com[2] == pytest.approx(0.33, abs=1e-9)
        # 框厚由 4 段圆截面框体半径 2r 表达（视觉/碰撞同源的最小表达）
        assert len(geometry.frame) == 4
        for segment in geometry.frame:
            assert segment.radius == pytest.approx(0.011, abs=1e-12)
            assert 2.0 * segment.radius == pytest.approx(0.022, abs=1e-12)

    def test_mass_is_strung_not_unstrung(self, profile: RacketProfile) -> None:
        """官方规格质量是穿线 316 g，不得写成未穿线 300 g。"""
        assert profile.mass == pytest.approx(0.316, abs=1e-12)
        assert profile.mass != pytest.approx(0.300, abs=1e-6)

    def test_face_area_consistent_with_face_ellipse(self, profile: RacketProfile) -> None:
        """拍面面积必须与拍面椭圆半轴自洽（π·半宽·半长，误差 ≤ 1%）。"""
        half_width = profile.geometry.face.semi_axes[1]
        half_length = profile.geometry.face.semi_axes[2]
        assert profile.geometry.face_area == pytest.approx(
            math.pi * half_width * half_length, rel=1e-2
        )
        assert half_length == pytest.approx(profile.geometry.length - profile.geometry.tcp[2], abs=1e-9)

    def test_geometry_is_self_consistent(self, profile: RacketProfile) -> None:
        """长度、TCP、拍面中心、手柄、喉部、框体必须相互自洽。"""
        geometry = profile.geometry
        assert geometry.tcp == geometry.face.center
        assert geometry.tcp[2] + geometry.face.semi_axes[2] == pytest.approx(
            geometry.length, abs=1e-9
        )
        assert geometry.tcp[2] - geometry.face.semi_axes[2] >= -1e-9
        handle = geometry.handle
        # 柄底（胶囊下端 tip）即局部系原点
        assert math.hypot(handle.start[0], handle.start[1]) <= 1e-9
        assert handle.start[2] - handle.radius == pytest.approx(0.0, abs=1e-9)
        assert handle.end[2] + handle.radius <= geometry.length + 1e-9
        assert len(geometry.throat) == 2
        for segment in list(geometry.throat) + list(geometry.frame):
            assert 0.0 <= segment.start[2] <= geometry.length + 1e-9
            assert 0.0 <= segment.end[2] <= geometry.length + 1e-9
            assert segment.radius > 0.0

    def test_frame_forms_closed_rectangular_ring(self, profile: RacketProfile) -> None:
        """框体必须在 x=0 平面内闭合：横杆沿 ±Y、侧杆沿 ±Z，外沿与 length 一致。

        该用例防止 T2 按 Capsule 直接生成视觉/碰撞时出现"横杆沿 X（拍面法向）
        穿过拍面"的错误几何。
        """
        frame = profile.geometry.frame
        assert len(frame) == 4
        # 所有框体线段都必须位于 x = 0 平面（拍面所在平面）
        for index, segment in enumerate(frame):
            assert abs(segment.start[0]) <= 1e-9, index
            assert abs(segment.end[0]) <= 1e-9, index
        horizontal = [
            segment for segment in frame if abs(segment.start[2] - segment.end[2]) <= 1e-12
        ]
        vertical = [
            segment for segment in frame if abs(segment.start[1] - segment.end[1]) <= 1e-12
        ]
        assert len(horizontal) == 2
        assert len(vertical) == 2
        # 横杆沿 ±Y，位于框体上下边
        for segment in horizontal:
            assert segment.start[1] == pytest.approx(-0.112, abs=1e-12)
            assert segment.end[1] == pytest.approx(0.112, abs=1e-12)
        assert {round(abs(segment.start[2]), 6) for segment in horizontal} == {0.363, 0.6748}
        # 侧杆沿 ±Z，位于框体左右边
        for segment in vertical:
            assert abs(segment.start[1]) == pytest.approx(0.112, abs=1e-12)
            assert segment.end[1] == pytest.approx(segment.start[1], abs=1e-12)
            assert segment.start[2] == pytest.approx(0.363, abs=1e-12)
            assert segment.end[2] == pytest.approx(0.6748, abs=1e-12)
        assert {round(segment.start[1], 6) for segment in vertical} == {-0.112, 0.112}
        # 外沿：max|y| + r ≈ 0.123（拍面半宽）、z ∈ [0.352, 0.6858]（= length）
        max_abs_y = max(
            max(abs(segment.start[1]), abs(segment.end[1])) + segment.radius
            for segment in frame
        )
        min_z = min(
            min(segment.start[2], segment.end[2]) - segment.radius for segment in frame
        )
        max_z = max(
            max(segment.start[2], segment.end[2]) + segment.radius for segment in frame
        )
        assert max_abs_y == pytest.approx(profile.geometry.face.semi_axes[1], abs=1e-12)
        assert max_abs_y == pytest.approx(0.123, abs=1e-12)
        assert min_z == pytest.approx(0.352, abs=1e-12)
        assert max_z == pytest.approx(profile.geometry.length, abs=1e-12)
        assert max_z == pytest.approx(0.6858, abs=1e-12)

    def test_official_fields_have_official_source(self, profile: RacketProfile) -> None:
        """官方规格字段必须标 official 且给出出处；不得伪称实测。"""
        for key in ("geometry.length", "geometry.face_area", "mass", "com"):
            entry = profile.provenance[key]
            assert entry.source == "official", key
            assert entry.reference, key
            assert "wilson" in entry.reference.lower(), key

    def test_estimated_fields_have_method(self, profile: RacketProfile) -> None:
        """所有估计项必须标 estimated 且 method 非空，形状/惯量不得伪称实测。"""
        estimated_keys = {
            "geometry.face",
            "geometry.tcp",
            "geometry.handle",
            "geometry.throat",
            "geometry.frame",
            "inertia",
            "mount.position",
            "mount.wxyz",
            "connector.geometry",
            "connector.mass",
            "connector.com",
            "connector.inertia",
        }
        for key in estimated_keys:
            entry = profile.provenance[key]
            assert entry.source == "estimated", key
            assert entry.method, key
        # 关键估计项的方法说明必须交代几何/惯量近似来源
        assert "椭" in profile.provenance["geometry.face"].method or "近似" in (
            profile.provenance["geometry.face"].method
        )
        assert "平行轴" in profile.provenance["inertia"].method or "近似" in (
            profile.provenance["inertia"].method
        )

    def test_connector_is_independent_rigid_body(self, profile: RacketProfile) -> None:
        """连接件必须是独立刚体参数，且质量不得重复计入球拍质量。"""
        connector = profile.connector
        assert connector is not None
        assert connector.shape == "cylinder"
        assert connector.mass > 0.0
        assert connector.length > 0.0
        assert connector.radius > 0.0
        assert np.linalg.norm(np.asarray(connector.axis)) == pytest.approx(1.0, abs=1e-9)
        assert np.allclose(connector.inertia.as_matrix(), connector.inertia.as_matrix().T)
        assert profile.mass == pytest.approx(0.316, abs=1e-12)

    def test_mount_is_estimated_identity_rotation(self, profile: RacketProfile) -> None:
        """安装变换当前为估计值（T7 标定），旋转部分应为单位四元数。"""
        assert profile.provenance["mount.position"].source == "estimated"
        assert profile.provenance["mount.wxyz"].source == "estimated"
        rot = quat_wxyz_to_rotmat(profile.mount.wxyz)
        assert np.allclose(rot, np.eye(3), atol=1e-9)
        assert np.linalg.norm(np.asarray(profile.mount.wxyz)) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------------------
# 非法输入拒绝（禁止静默重置为默认值）
# --------------------------------------------------------------------------------------


class TestInvalidInputRejected:
    """非法 YAML 输入必须抛错，不得回落默认值。"""

    def test_load_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        """文件不存在时必须抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_racket_profile(tmp_path / "not_exists.yaml")

    def test_malformed_yaml_raises_valueerror_with_path(self, tmp_path: Path) -> None:
        """畸形 YAML 必须包装为含路径的 ValueError（而非泄漏 yaml 库异常类型）。"""
        broken_path = tmp_path / "broken_racket.yaml"
        broken_path.write_text(
            "schema_version: 1\ngeometry: [unclosed\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="broken_racket.yaml"):
            load_racket_profile(broken_path)

    def test_synthetic_base_profile_is_valid(self, tmp_path: Path) -> None:
        """变异用例的基准字典本身必须是合法的（防止基准错误掩盖校验缺陷）。"""
        profile = load_racket_profile(_write_profile(tmp_path, _base_profile()))
        assert profile.racket_id == "synthetic_test_racket"
        assert profile.geometry.length == pytest.approx(0.5)

    def test_unknown_top_level_field_rejected(self, tmp_path: Path) -> None:
        """未知顶层字段必须拒绝（拼写错误不得静默忽略）。"""

        def mutate(data: dict[str, Any]) -> None:
            data["mas"] = 0.3

        with pytest.raises(ValueError, match="未知字段"):
            _load_mutated(tmp_path, mutate)

    def test_unknown_geometry_field_rejected(self, tmp_path: Path) -> None:
        """未知几何子字段必须拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            data["geometry"]["len"] = 0.5

        with pytest.raises(ValueError, match="未知字段"):
            _load_mutated(tmp_path, mutate)

    def test_unknown_provenance_key_rejected(self, tmp_path: Path) -> None:
        """未知 provenance 键必须拒绝（避免来源标注与字段脱节）。"""

        def mutate(data: dict[str, Any]) -> None:
            data["provenance"]["geometry.len"] = {
                "source": "estimated",
                "units": "m",
                "reference": None,
                "method": "伪造来源",
            }

        with pytest.raises(ValueError, match="未知字段"):
            _load_mutated(tmp_path, mutate)

    def test_missing_required_top_level_field_rejected(self, tmp_path: Path) -> None:
        """缺少必需顶层字段必须拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            del data["mass"]

        with pytest.raises(ValueError, match="缺少必需字段"):
            _load_mutated(tmp_path, mutate)

    def test_missing_provenance_key_rejected(self, tmp_path: Path) -> None:
        """已有字段缺少来源标注必须拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            del data["provenance"]["inertia"]

        with pytest.raises(ValueError, match="缺少必需字段"):
            _load_mutated(tmp_path, mutate)

    def test_unsupported_schema_version_rejected(self, tmp_path: Path) -> None:
        """不支持的 schema_version 必须拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            data["schema_version"] = 2

        with pytest.raises(ValueError, match="schema_version"):
            _load_mutated(tmp_path, mutate)

    @pytest.mark.parametrize("bad_mass", [0.0, -0.1, float("nan"), float("inf"), True])
    def test_invalid_mass_rejected(self, tmp_path: Path, bad_mass: Any) -> None:
        """质量必须为正的有限数；0/负数/NaN/Inf/bool 一律拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            data["mass"] = bad_mass

        with pytest.raises(ValueError, match="mass"):
            _load_mutated(tmp_path, mutate)

    def test_non_symmetric_inertia_rejected(self, tmp_path: Path) -> None:
        """非对称惯量矩阵必须拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            data["inertia"]["matrix"][0][1] = 0.002

        with pytest.raises(ValueError, match="对称"):
            _load_mutated(tmp_path, mutate)

    def test_non_positive_definite_inertia_rejected(self, tmp_path: Path) -> None:
        """非正定惯量矩阵必须拒绝（特征值出现负值）。"""

        def mutate(data: dict[str, Any]) -> None:
            data["inertia"]["matrix"] = [
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, -0.001],
            ]

        with pytest.raises(ValueError, match="正定"):
            _load_mutated(tmp_path, mutate)

    def test_triangle_inequality_violating_inertia_rejected(self, tmp_path: Path) -> None:
        """违反主惯量三角不等式（λ₂ > λ₀ + λ₁）的惯量必须拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            data["inertia"]["matrix"] = [
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.3],
            ]

        with pytest.raises(ValueError, match="三角不等式"):
            _load_mutated(tmp_path, mutate)

    @pytest.mark.parametrize(
        "bad_quat",
        [
            [2.0, 0.0, 0.0, 0.0],
            [1.0, 0.01, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    )
    def test_non_unit_quaternion_rejected(
        self, tmp_path: Path, bad_quat: list[float]
    ) -> None:
        """安装四元数范数偏离 1 超过容差必须拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            data["mount"]["wxyz"] = bad_quat

        with pytest.raises(ValueError, match="四元数"):
            _load_mutated(tmp_path, mutate)

    def test_float_rounding_quaternion_is_accepted_and_normalized(self, tmp_path: Path) -> None:
        """MuJoCo 风格的 0.707107 四元数（范数误差 ~3e-7）应接受并归一化。"""

        def mutate(data: dict[str, Any]) -> None:
            data["mount"]["wxyz"] = [0.707107, 0.0, 0.0, 0.707107]

        profile = _load_mutated(tmp_path, mutate)
        assert np.linalg.norm(np.asarray(profile.mount.wxyz)) == pytest.approx(1.0, abs=1e-12)

    def test_face_center_beyond_length_rejected(self, tmp_path: Path) -> None:
        """拍面中心 z 超过球拍长度必须拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            data["geometry"]["face"]["center"] = [0.0, 0.0, 0.6]
            data["geometry"]["tcp"] = [0.0, 0.0, 0.6]

        with pytest.raises(ValueError, match="face.center"):
            _load_mutated(tmp_path, mutate)

    def test_tcp_mismatch_face_center_rejected(self, tmp_path: Path) -> None:
        """TCP 必须等于拍面几何中心，不一致必须拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            data["geometry"]["tcp"] = [0.0, 0.0, 0.35]

        with pytest.raises(ValueError, match="tcp"):
            _load_mutated(tmp_path, mutate)

    def test_face_area_inconsistent_rejected(self, tmp_path: Path) -> None:
        """拍面面积与椭圆半轴不自洽（偏差 > 2%）必须拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            data["geometry"]["face_area"] = 0.1

        with pytest.raises(ValueError, match="face_area"):
            _load_mutated(tmp_path, mutate)

    def test_negative_radius_rejected(self, tmp_path: Path) -> None:
        """负半径必须拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            data["geometry"]["handle"]["radius"] = -0.01

        with pytest.raises(ValueError, match="radius"):
            _load_mutated(tmp_path, mutate)

    def test_handle_not_anchored_at_butt_rejected(self, tmp_path: Path) -> None:
        """手柄下端 tip 不在局部系原点必须拒绝（局部系原点契约）。"""

        def mutate(data: dict[str, Any]) -> None:
            data["geometry"]["handle"]["start"] = [0.0, 0.0, 0.05]

        with pytest.raises(ValueError, match="手柄"):
            _load_mutated(tmp_path, mutate)

    def test_unsupported_capsule_shape_rejected(self, tmp_path: Path) -> None:
        """手柄/喉部/框体的形状标识非 capsule 时必须拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            data["geometry"]["handle"]["shape"] = "box"

        with pytest.raises(ValueError, match="shape"):
            _load_mutated(tmp_path, mutate)

    def test_unsupported_face_shape_rejected(self, tmp_path: Path) -> None:
        """未知拍面形状必须拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            data["geometry"]["face"]["shape"] = "triangle"

        with pytest.raises(ValueError, match="shape"):
            _load_mutated(tmp_path, mutate)

    def test_estimated_without_method_rejected(self, tmp_path: Path) -> None:
        """estimated 来源缺少 method 必须拒绝（估计项必须交代方法）。"""

        def mutate(data: dict[str, Any]) -> None:
            data["provenance"]["inertia"]["method"] = None

        with pytest.raises(ValueError, match="method"):
            _load_mutated(tmp_path, mutate)

    def test_official_without_reference_rejected(self, tmp_path: Path) -> None:
        """official 来源缺少 reference 必须拒绝（官方值必须给出处）。"""

        def mutate(data: dict[str, Any]) -> None:
            data["provenance"]["mass"]["source"] = "official"
            data["provenance"]["mass"]["reference"] = None

        with pytest.raises(ValueError, match="reference"):
            _load_mutated(tmp_path, mutate)

    def test_unknown_source_rejected(self, tmp_path: Path) -> None:
        """未知来源枚举必须拒绝。"""

        def mutate(data: dict[str, Any]) -> None:
            data["provenance"]["mass"]["source"] = "guessed"

        with pytest.raises(ValueError, match="source"):
            _load_mutated(tmp_path, mutate)

    def test_units_mismatch_rejected(self, tmp_path: Path) -> None:
        """来源标注单位与字段单位不一致必须拒绝（SI 单位契约）。"""

        def mutate(data: dict[str, Any]) -> None:
            data["provenance"]["mass"]["units"] = "g"

        with pytest.raises(ValueError, match="units"):
            _load_mutated(tmp_path, mutate)

    def test_invalid_input_does_not_fall_back_to_default(self, tmp_path: Path) -> None:
        """非法输入抛错后不得返回被悄悄修正的 profile（禁止静默重置）。"""

        def mutate(data: dict[str, Any]) -> None:
            data["mass"] = 0.0

        with pytest.raises(ValueError):
            _load_mutated(tmp_path, mutate)
        # 合法文件仍按自身数值加载，说明没有全局默认值污染
        valid = load_racket_profile(_write_profile(tmp_path, _base_profile(), "valid.yaml"))
        assert valid.mass == pytest.approx(0.3, abs=1e-12)
