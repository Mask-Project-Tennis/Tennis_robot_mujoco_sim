"""run_20hits_video.py 脚本导入完整性测试。"""
import subprocess
import sys
from pathlib import Path


def test_help_exits_zero():
    """run_20hits_video.py --help 应正常退出（验证无 ImportError）。"""
    script = Path(__file__).resolve().parent.parent / "scripts" / "run_20hits_video.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"脚本启动失败（可能 ImportError）:\n{result.stderr.decode()}"
    )
    # argparse --help 输出到 stdout，验证标准 usage 模式存在
    stdout = result.stdout.decode()
    assert "usage:" in stdout.lower() or "options:" in stdout.lower()


def test_no_v9_label_in_output_paths():
    """脚本源码中不应残留 v9 版本标签（输出路径、文件名、logger 名）。"""
    script = Path(__file__).resolve().parent.parent / "scripts" / "run_20hits_video.py"
    source = script.read_text(encoding="utf-8")
    assert "20hits_v9" not in source, "脚本中仍残留 20hits_v9 路径标签"
