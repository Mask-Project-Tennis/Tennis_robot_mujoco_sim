"""mujoco_loader 最小覆盖测试。

覆盖范围:
- stale-sweep 清理逻辑（删除 >1h 的崩溃残留目录）
- 模块级缓存复用 / 签名失效
- atexit 清理
- Task 2.4: 内容感知缓存键（D2 资源 + 嵌套 XML 覆盖、等元数据变内容、
  跨项目根隔离、非 ASCII 根 + 换 cwd）与 xml_text 参数化加载
  （不改源模板、不缓存 MjModel 实例）

未覆盖（有意为之，记录为已知缺口）:
- 跨进程竞态: 多进程同时启动时的 stale-sweep 竞争。
  原因: Windows 上多进程模拟脆弱且 CI 不稳定；
  实际场景中同进程第二次调用走缓存路径（line 62-64）在
  stale-sweep 之前返回，本进程永不删自己的目录。
  竞态要求: 同进程两次调用间隔 >1h + 期间另一进程启动，
  概率趋近于零。
- 缓存目录被外部删除后的 .exists() 复查（I1 防御）:
  见 mujoco_loader.py 代码层防御，此处不重复测。
- Linux 分支（sys.platform != win32 直接加载）: 本机无 Linux 环境，
  仅由代码路径审查保证，不宣称已跨平台验收。
"""

from __future__ import annotations

import os
import shutil
import stat
import struct
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import mujoco
import pytest

import src.utils.mujoco_loader as ml


@pytest.fixture(autouse=True)
def _reset_cache() -> Generator[None, None, None]:
    """每个测试前后清理模块级缓存和测试创建的 mj_model_* 临时目录。"""
    ml._cached_tmp_dir = None
    ml._cached_src_hash = None
    # Task 2.4 新增的缓存键组成部分；实现落地前该属性可能不存在，
    # 直接赋值即可安全创建/复位，不影响 RED 阶段的失败归因。
    ml._cached_project_root = None
    tmp_root = Path(tempfile.gettempdir())
    pre_existing = set(tmp_root.glob("mj_model_*"))
    yield
    ml._cached_tmp_dir = None
    ml._cached_src_hash = None
    ml._cached_project_root = None
    post_existing = set(tmp_root.glob("mj_model_*"))
    for d in post_existing - pre_existing:
        shutil.rmtree(d, ignore_errors=True)


def _make_fake_project(tmp_path: Path) -> Path:
    """创建最小化假项目结构（src/robot/ + assets/ 空目录）。"""
    (tmp_path / "src" / "robot").mkdir(parents=True)
    (tmp_path / "assets").mkdir()
    return tmp_path


def test_stale_sweep_deletes_old_dirs(tmp_path: Path) -> None:
    """超过 _STALE_AGE_S 的 mj_model_* 目录被清理，新鲜的保留。"""
    project_root = _make_fake_project(tmp_path)
    tmp_root = Path(tempfile.gettempdir())
    ts = int(time.time() * 1e6)  # 微秒时间戳保证唯一

    old_dir = tmp_root / f"mj_model_test_old_{ts}"
    fresh_dir = tmp_root / f"mj_model_test_fresh_{ts}"
    old_dir.mkdir()
    fresh_dir.mkdir()

    # 将 old_dir 的 mtime 设为 2 小时前（超过 _STALE_AGE_S 阈值）
    old_time = time.time() - ml._STALE_AGE_S - 3600
    os.utime(old_dir, (old_time, old_time))

    ml._setup_temp_copy(project_root)

    assert not old_dir.exists(), f"旧目录未被清理: {old_dir}"
    assert fresh_dir.exists(), f"新鲜目录被误删: {fresh_dir}"


def test_cache_reuse_same_signature(tmp_path: Path) -> None:
    """同签名第二次调用复用缓存，返回相同路径。"""
    project_root = _make_fake_project(tmp_path)

    first = ml._setup_temp_copy(project_root)
    second = ml._setup_temp_copy(project_root)

    assert first == second


def test_cache_invalidation_on_sig_change(tmp_path: Path) -> None:
    """签名变化时重新复制，返回不同路径。"""
    project_root = _make_fake_project(tmp_path)

    with patch("src.utils.mujoco_loader._compute_dir_signature", return_value="sig_aaa"):
        first = ml._setup_temp_copy(project_root)
    with patch("src.utils.mujoco_loader._compute_dir_signature", return_value="sig_bbb"):
        second = ml._setup_temp_copy(project_root)

    assert first != second


def test_cleanup_temp_clears_state(tmp_path: Path) -> None:
    """_cleanup_temp 删除目录并清空模块级缓存。"""
    project_root = _make_fake_project(tmp_path)
    ml._setup_temp_copy(project_root)
    created_dir = ml._cached_tmp_dir
    assert created_dir is not None
    assert created_dir.exists()

    ml._cleanup_temp()

    assert not created_dir.exists()
    assert ml._cached_tmp_dir is None


def test_cleanup_temp_safe_when_no_model_loaded() -> None:
    """_cleanup_temp 在 _cached_tmp_dir=None 时安全 no-op（不应抛异常）。

    回归 M5：模块级 atexit.register 在导入时执行，此时若进程从未加载
    模型（_cached_tmp_dir 仍为 None），退出时 atexit 触发 _cleanup_temp
    必须安全返回，不能 NoneType.rmtree 崩溃。
    """
    ml._cached_tmp_dir = None
    # 多次调用也应幂等
    ml._cleanup_temp()
    ml._cleanup_temp()
    assert ml._cached_tmp_dir is None


# ---------------------------------------------------------------------------
# Task 2.4：内容感知缓存键 + xml_text 参数化加载
# ---------------------------------------------------------------------------


def _write_binary_stl(path: Path, n_triangles: int) -> None:
    """写一个含 n_triangles 个三角形的合法最小二进制 STL 网格。

    第 i 个三角形位于 z=i 平面，各三角形顶点互不重合，
    保证不同 n_triangles 编译出的 mesh_vert 数量可区分。

    Args:
        path: STL 输出路径（自动创建父目录）。
        n_triangles: 三角形数量。
    """
    buf = bytearray(b"\0" * 80)
    buf += struct.pack("<I", n_triangles)
    for i in range(n_triangles):
        buf += struct.pack(
            "<12fH",
            0.0, 0.0, 1.0,          # 法向量
            0.0, 0.0, float(i),     # 顶点 1
            1.0, 0.0, float(i),     # 顶点 2
            0.0, 1.0, float(i),     # 顶点 3
            0,                      # 属性字节计数
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(buf))


def _make_real_project(root: Path) -> Path:
    """创建可真实加载的最小假项目骨架（src/robot/nested + assets）。"""
    (root / "src" / "robot" / "nested").mkdir(parents=True)
    (root / "assets").mkdir()
    return root


def _plain_model_xml(body_name: str = "probe", sphere_size: str = "0.1") -> str:
    """无外部资源的最小 MJCF（用于原路径分支与 override 测试）。"""
    return (
        '<mujoco model="probe">'
        f'<worldbody><body name="{body_name}">'
        f'<geom name="g" type="sphere" size="{sphere_size}"/>'
        "</body></worldbody></mujoco>"
    )


def _d2_mesh_model_xml() -> str:
    """引用 D2 网格 + include 嵌套 XML 的最小 MJCF。

    网格路径相对 src/robot/ 解析（与 RM65 的 ../../assets 布局一致），
    嵌套 include 模拟 Task 2.1 的 d2_arm/d2_assets 片段组织方式。
    """
    return (
        '<mujoco model="probe">'
        '<include file="nested/extra.xml"/>'
        '<asset><mesh name="part" file="../../assets/unitree_d2/meshes/part.STL"/></asset>'
        '<worldbody><body name="arm"><geom type="mesh" mesh="part"/></body></worldbody>'
        "</mujoco>"
    )


def _nested_extra_xml(body_name: str = "extra") -> str:
    """被 include 的嵌套 XML 片本（mujocoinclude）。"""
    return (
        "<mujocoinclude>"
        f'<worldbody><body name="{body_name}"><geom type="sphere" size="0.05"/></body></worldbody>'
        "</mujocoinclude>"
    )


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    """读取指定 body id 的名称（测试断言辅助）。"""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    assert name is not None
    return name


def test_d2_mesh_and_nested_xml_invalidate_cache(tmp_path: Path) -> None:
    """D2 网格与嵌套 XML 变更都必须使缓存失效并加载到新内容。

    旧签名只覆盖 src/robot/*.xml（非递归）和 assets/rm_65 网格，
    assets/unitree_d2 网格与 src/robot 嵌套目录 XML 均不在键内，
    修改后仍会读到旧副本。
    """
    root = _make_real_project(tmp_path / "网球项目")  # 非 ASCII → 临时副本分支
    mesh = root / "assets" / "unitree_d2" / "meshes" / "part.STL"
    _write_binary_stl(mesh, 2)
    model = root / "src" / "robot" / "model.xml"
    model.write_text(_d2_mesh_model_xml(), encoding="utf-8")
    nested = root / "src" / "robot" / "nested" / "extra.xml"
    nested.write_text(_nested_extra_xml("extra"), encoding="utf-8")

    m1 = ml.load_mujoco_model(model)
    assert m1.nbody == 3  # world + arm + extra
    assert m1.nmeshvert == 6  # 2 个三角形 × 3 顶点

    # D2 网格变更 → 缓存必须失效，读到新网格
    _write_binary_stl(mesh, 4)
    m2 = ml.load_mujoco_model(model)
    assert m2.nmeshvert == 12, "D2 网格变更后仍读到旧副本"

    # 嵌套 XML（模拟 d2 片段 include）变更 → 缓存必须失效
    nested.write_text(_nested_extra_xml("extra_renamed"), encoding="utf-8")
    m3 = ml.load_mujoco_model(model)
    names = {_body_name(m3, i) for i in range(m3.nbody)}
    assert "extra_renamed" in names, "嵌套 XML 变更后仍读到旧副本"


def test_equal_metadata_changed_content_invalidates(tmp_path: Path) -> None:
    """文件名与 mtime 完全一致但内容变化时，必须凭内容摘要重新复制。"""
    root = _make_real_project(tmp_path / "网球项目")
    model = root / "src" / "robot" / "model.xml"
    model.write_text(_plain_model_xml(sphere_size="0.1"), encoding="utf-8")

    m1 = ml.load_mujoco_model(model)
    assert m1.geom_size[0][0] == pytest.approx(0.1)

    # 改内容但把 mtime 精确还原，模拟"元数据不变、内容已变"
    st = model.stat()
    model.write_text(_plain_model_xml(sphere_size="0.3"), encoding="utf-8")
    os.utime(model, (st.st_atime, st.st_mtime))

    m2 = ml.load_mujoco_model(model)
    assert m2.geom_size[0][0] == pytest.approx(0.3), "mtime 未变时内容变更未使缓存失效"


def test_cache_does_not_cross_project_roots(tmp_path: Path) -> None:
    """不同项目根（同名文件、同 mtime、不同内容）不得共享缓存副本。"""
    root_a = _make_real_project(tmp_path / "项目甲")
    root_b = _make_real_project(tmp_path / "项目乙")
    model_a = root_a / "src" / "robot" / "model.xml"
    model_b = root_b / "src" / "robot" / "model.xml"
    model_a.write_text(_plain_model_xml(body_name="from_a"), encoding="utf-8")
    model_b.write_text(_plain_model_xml(body_name="from_b"), encoding="utf-8")

    # 抹平 mtime 差异，确保判等不可能依赖时间戳
    stamp = time.time() - 5000
    for p in (model_a, model_b):
        os.utime(p, (stamp, stamp))

    m_a = ml.load_mujoco_model(model_a)
    m_b = ml.load_mujoco_model(model_b)
    assert _body_name(m_a, 1) == "from_a"
    assert _body_name(m_b, 1) == "from_b", "跨项目根复用了同一份缓存副本"


def test_xml_override_keeps_source_unchanged(tmp_path: Path) -> None:
    """xml_text 参数化编译不改源模板、不污染缓存副本、不残留临时 XML。

    分别覆盖原路径分支（ASCII 路径）与临时副本分支（非 ASCII 路径）。
    """
    # -- 分支一：ASCII 原路径 --
    root = _make_real_project(tmp_path / "plain_project")
    model = root / "src" / "robot" / "model.xml"
    original_text = _plain_model_xml(sphere_size="0.1")
    model.write_text(original_text, encoding="utf-8")
    before = sorted(p.name for p in model.parent.iterdir())

    m = ml.load_mujoco_model(model, xml_text=_plain_model_xml(sphere_size="0.3"))
    assert m.geom_size[0][0] == pytest.approx(0.3)
    assert model.read_text(encoding="utf-8") == original_text, "源模板被覆盖"
    assert sorted(p.name for p in model.parent.iterdir()) == before, "原路径分支残留临时 XML"

    # -- 分支二：非 ASCII 路径（先常规加载暖缓存，再 override）--
    root_u = _make_real_project(tmp_path / "网球项目")
    model_u = root_u / "src" / "robot" / "model.xml"
    original_u = _plain_model_xml(sphere_size="0.1")
    model_u.write_text(original_u, encoding="utf-8")
    ml.load_mujoco_model(model_u)  # 暖缓存
    cached_dir = ml._cached_tmp_dir
    assert cached_dir is not None
    cached_template = cached_dir / "src" / "robot" / "model.xml"
    cached_before = sorted(p.name for p in cached_template.parent.iterdir())

    m_u = ml.load_mujoco_model(model_u, xml_text=_plain_model_xml(sphere_size="0.3"))
    assert m_u.geom_size[0][0] == pytest.approx(0.3)
    assert model_u.read_text(encoding="utf-8") == original_u, "非 ASCII 分支源模板被覆盖"
    assert cached_template.read_text(encoding="utf-8") == original_u, "缓存模板被覆盖"
    assert (
        sorted(p.name for p in cached_template.parent.iterdir()) == cached_before
    ), "缓存目录残留临时 XML"


def test_xml_override_falls_back_when_source_dir_readonly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """源模板目录不可写时，xml_text 仍可编译（降级到 ASCII 临时副本）。

    只读安装、只读容器挂载下，"读模型"的 API 不应要求能写源码目录。
    模拟方式：只让发往源目录的 mkstemp 抛 PermissionError，其余照常。
    """
    root = _make_real_project(tmp_path / "plain_project")
    model = root / "src" / "robot" / "model.xml"
    model.write_text(_plain_model_xml(sphere_size="0.1"), encoding="utf-8")
    before = sorted(p.name for p in model.parent.iterdir())

    real_mkstemp = tempfile.mkstemp
    source_dir = model.parent.resolve()

    def _guarded_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        target = kwargs.get("dir")
        if target is not None and Path(str(target)).resolve() == source_dir:
            raise PermissionError(13, "目录只读（测试模拟）", str(target))
        return real_mkstemp(*args, **kwargs)  # type: ignore[return-value]

    monkeypatch.setattr(tempfile, "mkstemp", _guarded_mkstemp)

    m = ml.load_mujoco_model(model, xml_text=_plain_model_xml(sphere_size="0.3"))
    assert m.geom_size[0][0] == pytest.approx(0.3)
    # 源目录未被写入（无临时 XML 残留）
    assert sorted(p.name for p in model.parent.iterdir()) == before
    # 降级路径确实建立了 ASCII 临时副本
    assert ml._cached_tmp_dir is not None
    assert (ml._cached_tmp_dir / "src" / "robot" / "model.xml").exists()


def test_temp_copy_dirs_get_write_permission(tmp_path: Path) -> None:
    """只读（0555）源目录复制的副本目录补足属主写权限，源目录不动。

    shutil.copytree 对每个复制目录执行 copystat，Unix 只读安装/容器
    挂载的目录权限会原样带进副本：xml_text 的只读降级路径会在副本内
    mkstemp 再次 PermissionError，rmtree 也清理不掉。Windows 的 chmod
    不控制目录访问，本用例在该平台退化为权限位级别断言（位会真实
    反映在 stat().st_mode），Linux 上是真实权限行为。
    """
    project_root = _make_fake_project(tmp_path)
    nested = project_root / "src" / "robot" / "deep"
    nested.mkdir()
    (nested / "scene.xml").write_text("<mujoco/>", encoding="utf-8")
    readonly_dirs = [
        tmp_path,
        project_root / "src",
        project_root / "src" / "robot",
        nested,
        project_root / "assets",
    ]
    try:
        for d in readonly_dirs:
            os.chmod(d, 0o555)

        tmp = ml._setup_temp_copy(project_root)

        # 副本内全部目录（含嵌套）都有属主写位
        for p in [tmp, *tmp.rglob("*")]:
            if p.is_dir():
                assert stat.S_IMODE(p.stat().st_mode) & stat.S_IWUSR, (
                    f"副本目录缺属主写权限: {p}"
                )
        # 源目录权限未被修改（只动副本）
        for d in readonly_dirs:
            assert not stat.S_IMODE(d.stat().st_mode) & stat.S_IWUSR
    finally:
        for d in readonly_dirs:
            os.chmod(d, 0o755)  # 恢复写权限，保证 tmp_path 可清理


def test_two_overrides_do_not_reuse_model(tmp_path: Path) -> None:
    """两套 xml_text 各编译各的独立 MjModel，不做实例缓存。"""
    root = _make_real_project(tmp_path / "plain_project")
    model = root / "src" / "robot" / "model.xml"
    model.write_text(_plain_model_xml(body_name="template"), encoding="utf-8")

    m1 = ml.load_mujoco_model(model, xml_text=_plain_model_xml(body_name="one"))
    m2 = ml.load_mujoco_model(model, xml_text=_plain_model_xml(body_name="two"))
    m3 = ml.load_mujoco_model(model, xml_text=_plain_model_xml(body_name="one"))

    assert m1 is not m2, "不同 xml_text 复用了同一 MjModel 实例"
    assert m2 is not m3 and m1 is not m3, "MjModel 实例被缓存复用"
    assert _body_name(m1, 1) == "one"
    assert _body_name(m2, 1) == "two"
    assert _body_name(m3, 1) == "one"

    # 无 xml_text 的常规加载不受 override 影响，读的是磁盘模板
    m4 = ml.load_mujoco_model(model)
    assert _body_name(m4, 1) == "template"


def test_unicode_root_and_changed_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """非 ASCII 项目根下切换 cwd 后重载，仍能凭内容摘要取到新模板。"""
    root = _make_real_project(tmp_path / "网球项目")
    model = root / "src" / "robot" / "model.xml"
    model.write_text(_plain_model_xml(body_name="v1"), encoding="utf-8")
    resolved = model.resolve()

    m1 = ml.load_mujoco_model(resolved)
    assert _body_name(m1, 1) == "v1"

    # 内容改为 v2，但 mtime 还原成与 v1 完全一致
    st = model.stat()
    model.write_text(_plain_model_xml(body_name="v2"), encoding="utf-8")
    os.utime(model, (st.st_atime, st.st_mtime))

    # cwd 切到无关目录后用绝对路径重载
    monkeypatch.chdir(tempfile.gettempdir())
    m2 = ml.load_mujoco_model(resolved)
    assert _body_name(m2, 1) == "v2", "换 cwd 后仍读到旧缓存副本"


def test_r1d_mesh_change_invalidates_loader_cache(tmp_path: Path) -> None:
    """R1D 网格内容变化（相同 mtime/大小）必须使缓存失效（Task 2.5a · N0-6）。

    assets/unitree_r1d 纳入 _SIGNATURE_RESOURCE_DIRS 后，同大小同 mtime
    但内容不同的网格必须凭内容摘要触发重新复制，不得命中旧副本。

    注：MuJoCo 加载 STL 会把顶点平移到 mesh 几何中心（整体平移被归一化
    抵消），因此用顶点 z 的相对构型变化（缩放）构造可观测的内容差异，
    而非整体偏移。
    """
    root = _make_real_project(tmp_path / "网球项目")  # 非 ASCII → 临时副本分支
    mesh = root / "assets" / "unitree_r1d" / "meshes" / "part.STL"

    def _write_stl(path: Path, n_triangles: int, z_scale: float) -> None:
        """写与既有 _write_binary_stl 同布局的最小 STL，顶点 z 按比例缩放。

        同 n_triangles 保证文件大小不变；z_scale 改变顶点相对构型，
        用于构造"元数据与大小完全一致、仅内容不同"的替换。

        Args:
            path: STL 输出路径（自动创建父目录）。
            n_triangles: 三角形数量（决定文件大小）。
            z_scale: 第 i 个三角形顶点的 z 值系数（i*z_scale，决定内容）。
        """
        buf = bytearray(b"\0" * 80)
        buf += struct.pack("<I", n_triangles)
        for i in range(n_triangles):
            z = float(i) * z_scale
            buf += struct.pack(
                "<12fH",
                0.0, 0.0, 1.0,          # 法向量
                0.0, 0.0, z,            # 顶点 1
                1.0, 0.0, z,            # 顶点 2
                0.0, 1.0, z,            # 顶点 3
                0,                      # 属性字节计数
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(buf))

    _write_stl(mesh, 3, 1.0)
    model = root / "src" / "robot" / "model.xml"
    model.write_text(
        '<mujoco model="probe">'
        '<asset><mesh name="part" file="../../assets/unitree_r1d/meshes/part.STL"/></asset>'
        '<worldbody><body name="arm"><geom type="mesh" mesh="part"/></body></worldbody>'
        "</mujoco>",
        encoding="utf-8",
    )

    m1 = ml.load_mujoco_model(model)
    verts_before = m1.mesh_vert[: m1.nmeshvert].copy()

    # 内容变化（顶点 z 从 i*1 变为 i*2），但大小（同三角形数）与
    # mtime 精确还原
    st = mesh.stat()
    _write_stl(mesh, 3, 2.0)
    os.utime(mesh, (st.st_atime, st.st_mtime))

    m2 = ml.load_mujoco_model(model)
    verts_after = m2.mesh_vert[: m2.nmeshvert]
    assert not (verts_before == verts_after).all(), (
        "R1D 网格内容变化（相同 mtime/大小）后仍读到旧缓存副本"
    )
    # 中心化后顶点 z：i*1 → {-1, 0, 1}；i*2 → {-2, 0, 2}
    assert verts_after[:, 2].max() - verts_before[:, 2].max() == pytest.approx(1.0)
