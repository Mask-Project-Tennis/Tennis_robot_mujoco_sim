"""mujoco_loader 最小覆盖测试。

覆盖范围:
- stale-sweep 清理逻辑（删除 >1h 的崩溃残留目录）
- 模块级缓存复用 / 签名失效
- atexit 清理

未覆盖（有意为之，记录为已知缺口）:
- 跨进程竞态: 多进程同时启动时的 stale-sweep 竞争。
  原因: Windows 上多进程模拟脆弱且 CI 不稳定；
  实际场景中同进程第二次调用走缓存路径（line 62-64）在
  stale-sweep 之前返回，本进程永不删自己的目录。
  竞态要求: 同进程两次调用间隔 >1h + 期间另一进程启动，
  概率趋近于零。
- 缓存目录被外部删除后的 .exists() 复查（I1 防御）:
  见 mujoco_loader.py 代码层防御，此处不重复测。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

import src.utils.mujoco_loader as ml


@pytest.fixture(autouse=True)
def _reset_cache() -> Generator[None, None, None]:
    """每个测试前后清理模块级缓存和测试创建的 mj_model_* 临时目录。"""
    ml._cached_tmp_dir = None
    ml._cached_src_hash = None
    tmp_root = Path(tempfile.gettempdir())
    pre_existing = set(tmp_root.glob("mj_model_*"))
    yield
    ml._cached_tmp_dir = None
    ml._cached_src_hash = None
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
