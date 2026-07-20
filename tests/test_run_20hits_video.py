"""run_20hits_video.py 脚本导入完整性测试。"""
import os
import subprocess
import sys
from pathlib import Path


def test_help_exits_zero():
    """run_20hits_video.py --help 应正常退出（验证无 ImportError）。"""
    script = Path(__file__).resolve().parent.parent / "scripts" / "run_20hits_video.py"
    # 强制子进程用 UTF-8 输出，避免 Windows 控制台代码页（GBK/CP936）导致 decode 失败
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        timeout=30,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert result.returncode == 0, (
        f"脚本启动失败（可能 ImportError）:\n{result.stderr}"
    )
    # argparse --help 输出到 stdout，验证标准 usage 模式存在
    assert "usage:" in result.stdout.lower() or "options:" in result.stdout.lower()


def test_no_v9_label_in_output_paths():
    """脚本源码中不应残留 v9 版本标签（输出路径、文件名、logger 名、导入路径）。"""
    script = Path(__file__).resolve().parent.parent / "scripts" / "run_20hits_video.py"
    source = script.read_text(encoding="utf-8")
    assert "v9" not in source.lower(), "脚本中仍残留 v9 版本标签"
