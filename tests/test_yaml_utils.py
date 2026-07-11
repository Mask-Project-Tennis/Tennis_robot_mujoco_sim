"""yaml_utils 工具模块测试。"""
from pathlib import Path

from src.utils.yaml_utils import load_config, merge_configs


def test_merge_configs_deep_merge():
    """merge_configs 应递归合并嵌套字典。"""
    base = {"a": {"b": 1}, "c": 2}
    override = {"a": {"d": 3}, "e": 4}
    result = merge_configs(base, override)
    assert result == {"a": {"b": 1, "d": 3}, "c": 2, "e": 4}


def test_merge_configs_override_scalar():
    """merge_configs 标量值：override 覆盖 base。"""
    base = {"x": 1, "y": [1, 2]}
    override = {"x": 2, "y": [3]}
    result = merge_configs(base, override)
    assert result["x"] == 2
    assert result["y"] == [3]


def test_merge_configs_not_mutate_input():
    """merge_configs 不应修改输入字典。"""
    base = {"a": {"b": 1}}
    override = {"a": {"c": 2}}
    merge_configs(base, override)
    assert base == {"a": {"b": 1}}
    assert override == {"a": {"c": 2}}


def test_merge_configs_uncovered_nested_shares_reference():
    """未被 override 覆盖的嵌套 dict 与 base 共享引用（浅拷贝，当前设计如此）。

    此测试锁定浅拷贝行为：result 中未被 override 覆盖的嵌套字典
    与 base 中对应字典是同一对象。调用者不应修改返回值的嵌套字典。
    """
    base = {"a": {"x": 1}, "b": {"y": 2}}
    override = {"a": {"x": 10}}
    result = merge_configs(base, override)
    # "b" 未被 override 覆盖 → 共享引用
    assert result["b"] is base["b"]
    # "a" 被 override 覆盖 → 递归合并产生新对象
    assert result["a"] is not base["a"]


def test_load_config_returns_dict(tmp_path: Path):
    """load_config 应正确解析 YAML 文件并返回字典。"""
    yaml_content = "sim:\n  dt: 0.005\nball:\n  speed: 7\n"
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content, encoding="utf-8")
    result = load_config(config_file)
    assert result == {"sim": {"dt": 0.005}, "ball": {"speed": 7}}
