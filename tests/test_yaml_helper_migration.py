"""活跃脚本不应有本地 YAML 工具定义（应从 src.utils.yaml_utils 导入）。

TDD RED 阶段：本测试文件在 Task 6 迁移完成前应当失败。
迁移完成后（所有活跃脚本从 yaml_utils 导入），本测试转为 GREEN 持续守卫。
"""
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT / "scripts"

FORBIDDEN_PATTERNS = ["def load_config(", "def merge_configs("]

# 排除目录：scripts/tools/test_real_robot/ 下的 _connect.py 含领域专用的
# load_config（签名 config_path: str -> RealRobotConfig），它加载 RealRobotConfig
# 数据类而非 YAML dict，不是 yaml_utils.load_config 的重复定义，故排除。
EXCLUDED_DIRS = {"archive", "test_real_robot"}


def test_no_local_yaml_helpers_in_active_scripts() -> None:
    """非归档脚本不应有本地 load_config/merge_configs 定义（应迁移到 src.utils.yaml_utils）。"""
    violators: list[str] = []
    for py in SCRIPTS_DIR.rglob("*.py"):
        if any(excluded in py.parts for excluded in EXCLUDED_DIRS):
            continue
        source = py.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in source:
                violators.append(f"{py.relative_to(PROJECT)}: 含 '{pattern}'")
    assert not violators, (
        "以下非归档脚本仍有本地 YAML 工具定义（应迁移到 src.utils.yaml_utils）:\n"
        + "\n".join(violators)
    )


def test_active_scripts_import_yaml_from_utils() -> None:
    """引用 load_config/merge_configs 的活跃脚本必须从 src.utils.yaml_utils 导入。"""
    violators: list[str] = []
    for py in SCRIPTS_DIR.rglob("*.py"):
        if any(excluded in py.parts for excluded in EXCLUDED_DIRS):
            continue
        source = py.read_text(encoding="utf-8")
        uses_yaml_helpers = "load_config" in source or "merge_configs" in source
        if not uses_yaml_helpers:
            continue
        has_yaml_utils_import = "yaml_utils import" in source
        if not has_yaml_utils_import:
            violators.append(
                f"{py.relative_to(PROJECT)}: 引用 YAML 工具但未从 yaml_utils 导入"
            )
    assert not violators, (
        "以下脚本引用 load_config/merge_configs 但未从 yaml_utils 导入:\n"
        + "\n".join(violators)
    )
