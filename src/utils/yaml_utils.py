"""YAML 配置文件工具模块（加载 + 递归合并）。"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """加载 YAML 配置文件。

    Args:
        config_path: YAML 文件路径。

    Returns:
        解析后的字典。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个配置字典。

    Args:
        base: 基础字典。
        override: 覆盖字典（优先级更高）。

    Returns:
        合并后的新字典（不修改输入）。

    Note:
        使用浅拷贝策略：未被 override 覆盖的嵌套 dict 与 base 共享引用。
        当前所有调用场景（config 只读消费）不受影响。调用者不应修改返回值的嵌套字典。
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
