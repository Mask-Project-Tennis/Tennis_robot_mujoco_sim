"""归档完整性测试 — 确保旧脚本归档后路径自洽。

覆盖 4 种失败模式：
1. 文件不在正确位置（archive/root/, archive/sim/, archive/exp/）
2. 保留脚本中残留旧 import 路径
3. 归档的 wrapper 脚本缺少 archive/root sys.path 注入
4. 归档的 batch 脚本 WRAPPER 路径未指向 archive/exp/
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = PROJECT_ROOT / "scripts" / "archive"


# ── Test 1: 文件在正确位置 ──

def test_archive_structure() -> None:
    """归档后 15 个主脚本在 archive/{root,sim}/，旧实验脚本在 archive/exp/。"""
    # root/ 下 9 个
    root_scripts = [
        "rm65_mpc_v6.py", "rm65_mpc_v7.py", "rm65_mpc_v8.py",
        "rm65_mpc_v9.py", "rm65_mpc_v10.py", "rm65_mpc_tube.py",
        "rm65_mpc_tube_constraint.py",
        "rm65_mpc_tube_constraint_realtime.py",
        "rm65_mpc_tube_constraint_realtime_v2.py",
    ]
    for name in root_scripts:
        assert (ARCHIVE / "root" / name).exists(), f"{name} 不在 archive/root/"

    # sim/ 下 6 个
    sim_scripts = [
        "rm65_mpc_tube_constraint_realtime_v4.py",
        "rm65_mpc_tube_constraint_realtime_v5.py",
        "rm65_mpc_v8_softmin_only.py", "rm65_mpc_v8_tuned.py",
        "rm65_mpc_v8_tuned_softmin_only.py",
        "rm65_mpc_v9_softmin_only.py",
    ]
    for name in sim_scripts:
        assert (ARCHIVE / "sim" / name).exists(), f"{name} 不在 archive/sim/"

    # 原位置不存在
    for name in root_scripts:
        assert not (PROJECT_ROOT / "scripts" / name).exists(), \
            f"{name} 仍在 scripts/ 根目录"


# ── Test 2: 保留脚本无悬空引用 ──

def test_no_stale_references() -> None:
    """保留的脚本中不存在 import scripts.rm65_mpc_v[6-9] 等旧路径。"""
    kept = [
        PROJECT_ROOT / "scripts" / "rm65_mpc_v12.py",
        PROJECT_ROOT / "scripts" / "rm65_mpc_v11.py",
        PROJECT_ROOT / "scripts" / "run_20hits_video.py",
        PROJECT_ROOT / "scripts" / "tools" / "scan_ball_params.py",
        PROJECT_ROOT / "scripts" / "tools" / "render_20hits_video.py",
    ]
    stale_patterns = [
        "import scripts.rm65_mpc_v6",
        "import scripts.rm65_mpc_v7",
        "import scripts.rm65_mpc_v8",
        "import scripts.rm65_mpc_v9",
        "import scripts.rm65_mpc_v10",
        "from scripts.rm65_mpc_v9 import",
        "from scripts.rm65_mpc_tube import",
        "import scripts.rm65_mpc_tube_constraint ",
    ]
    for script in kept:
        if not script.exists():
            continue
        content = script.read_text(encoding="utf-8")
        for pattern in stale_patterns:
            assert pattern not in content, \
                f"{script.name} 仍含旧引用: {pattern!r}"


# ── Test 3: 归档的 wrapper 有正确 sys.path ──

def test_archived_wrappers_syspath() -> None:
    """归档的 wrapper 脚本包含 archive/root 的 sys.path 注入。"""
    archive_exp = ARCHIVE / "exp"
    wrappers = [
        "_run_exp1_exempt.py", "_run_exp1_v3_exempt.py",
        "_run_exp2_v3_strict.py", "_run_exp7_kf.py", "_run_exp7_noise.py",
        "run_experiments.py", "_run_strict_experiment.py",
        "run_tcp_limit_experiment.py",
        "run_tcp_limit_experiment_v2.py", "run_tcp_limit_experiment_v3.py",
    ]
    for name in wrappers:
        path = archive_exp / name
        assert path.exists(), f"{name} 不在 archive/exp/"
        content = path.read_text(encoding="utf-8")
        assert "archive" in content and "root" in content, \
            f"{name} 缺少 archive/root sys.path 注入"


# ── Test 4: 归档的 batch WRAPPER 路径 ──

def test_archived_batches_wrapper_path() -> None:
    """归档的 batch 脚本 WRAPPER 路径指向 archive/exp/。"""
    archive_exp = ARCHIVE / "exp"
    batches = [
        "run_exp1_batch.py", "run_exp1_v3_batch.py",
        "run_exp2_v3_batch.py", "run_exp7_batch.py", "run_exp8_batch.py",
    ]
    for name in batches:
        path = archive_exp / name
        assert path.exists(), f"{name} 不在 archive/exp/"
        content = path.read_text(encoding="utf-8")
        assert "archive" in content and "exp" in content, \
            f"{name} WRAPPER 路径未指向 archive/exp/"
