"""跨平台证据标签工具。

同一份代码在 Windows（MSVC）与 Linux（g++）上可能因浮点乘加顺序/归约差异
产生末位不同的结果，因此「逐位一致」类基线产物与冻结期望值必须按平台分开
保存与检索。本模块只提供稳定的平台标签，不参与任何动力学/物理计算。

标签格式：``<os>-<machine>``，例如 ``linux-x86_64``、``windows-amd64``。
"""

from __future__ import annotations

import platform
import sys

# sys.platform 的常见取值归一化（Cygwin 视作 Windows）。
_OS_ALIASES = {
    "win32": "windows",
    "cygwin": "windows",
    "darwin": "macos",
}

# platform.machine() 的常见别名归一化。
# 注意 Windows 上 Python 返回 "AMD64"，保持 amd64（不映射为 x86_64），
# 以与既有证据目录名 windows-amd64 对齐。
_MACHINE_ALIASES = {
    "x64": "amd64",
    "i386": "x86",
    "i686": "x86",
}


def normalize_platform_tag(os_name: str, machine: str) -> str:
    """把 ``sys.platform`` 与 ``platform.machine()`` 归一化为稳定标签。

    Args:
        os_name: ``sys.platform`` 取值（如 ``win32``、``linux``）。
        machine: ``platform.machine()`` 取值（如 ``AMD64``、``x86_64``）。

    Returns:
        归一化标签，例如 ``windows-amd64``、``linux-x86_64``。
    """
    os_part = _OS_ALIASES.get(os_name.lower(), os_name.lower())
    machine_part = _MACHINE_ALIASES.get(machine.lower(), machine.lower())
    return f"{os_part}-{machine_part}"


def platform_tag() -> str:
    """返回当前解释器所在平台的稳定标签。

    Returns:
        例如 ``linux-x86_64``（本机）或 ``windows-amd64``（Windows 源机）。
    """
    return normalize_platform_tag(sys.platform, platform.machine())
