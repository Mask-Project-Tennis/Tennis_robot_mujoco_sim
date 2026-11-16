"""src.robot 包 ``__init__`` 导出策略测试（阶段 2 Task 2.8）。

验证三条契约：
1. **导入纯度**：``import src.robot`` 不隐式加载 mujoco / ilqt / real / sim
   （干净子进程验证，规避全量 pytest 下 sys.modules 已被先行测试污染的
   假阴性——Task 2.7 评审 M2 的教训）；
2. **导出可用**：9 个公共名字（7 个纯 eager + 2 个绑定层 lazy）均可从
   包根导入，且与源模块中的对象同一；
3. **模型工具不进包根**：load_robot_model / RobotModelBundle /
   ModelKinematics 必须经显式模块路径导入。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import src.robot

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestImportPurity:
    """import src.robot 的依赖纯度（干净解释器验证）。"""

    def test_import_package_does_not_load_mujoco_or_heavy_layers(self) -> None:
        """干净子进程 import src.robot 后 sys.modules 无 mujoco/ilqt/real/sim。"""
        code = (
            "import sys, json; "
            "import src.robot; "
            "bad = sorted(m for m in sys.modules "
            "if m == 'mujoco' or m.startswith('mujoco.') "
            "or m.startswith('src.ilqt') or m.startswith('src.real') "
            "or m.startswith('src.sim')); "
            "print(json.dumps(bad))"
        )
        env = {**os.environ, "PYTHONUTF8": "1"}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, f"子进程失败:\n{proc.stderr}"
        assert json.loads(proc.stdout.strip().splitlines()[-1]) == []


class TestExportsAvailable:
    """公共名字可从包根导入且与源模块同一对象。"""

    def test_eager_pure_exports_match_source_modules(self) -> None:
        """7 个纯导出（profile + runtime_parameters）与源模块对象一致。"""
        import src.robot.robot_profile as profile_mod
        import src.robot.runtime_parameters as runtime_mod
        from src.robot import (
            RobotProfile,
            RobotRuntimeParameters,
            SceneParameters,
            default_rm65_profile_path,
            load_robot_profile,
            load_runtime_parameters,
            validate_runtime_parameters,
        )

        assert RobotProfile is profile_mod.RobotProfile is src.robot.RobotProfile
        assert (
            load_robot_profile
            is profile_mod.load_robot_profile
            is src.robot.load_robot_profile
        )
        assert (
            default_rm65_profile_path
            is profile_mod.default_rm65_profile_path
            is src.robot.default_rm65_profile_path
        )
        assert (
            RobotRuntimeParameters
            is runtime_mod.RobotRuntimeParameters
            is src.robot.RobotRuntimeParameters
        )
        assert SceneParameters is runtime_mod.SceneParameters is src.robot.SceneParameters
        assert (
            load_runtime_parameters
            is runtime_mod.load_runtime_parameters
            is src.robot.load_runtime_parameters
        )
        assert (
            validate_runtime_parameters
            is runtime_mod.validate_runtime_parameters
            is src.robot.validate_runtime_parameters
        )
        assert src.robot.load_robot_profile is profile_mod.load_robot_profile
        assert (
            src.robot.default_rm65_profile_path
            is profile_mod.default_rm65_profile_path
        )
        assert src.robot.RobotRuntimeParameters is runtime_mod.RobotRuntimeParameters
        assert src.robot.SceneParameters is runtime_mod.SceneParameters
        assert (
            src.robot.load_runtime_parameters
            is runtime_mod.load_runtime_parameters
        )
        assert (
            src.robot.validate_runtime_parameters
            is runtime_mod.validate_runtime_parameters
        )

    def test_lazy_binding_exports_match_source_module(self) -> None:
        """延迟导出的绑定层名字与 model_binding 模块对象一致。"""
        import src.robot.model_binding as binding_mod
        from src.robot import ModelBinding, bind_robot_model

        assert ModelBinding is binding_mod.ModelBinding is src.robot.ModelBinding
        assert (
            bind_robot_model
            is binding_mod.bind_robot_model
            is src.robot.bind_robot_model
        )
        # __all__ 覆盖 9 个公共名字
        assert len(src.robot.__all__) == 9


class TestModelToolsNotExported:
    """模型工具保持显式模块导入，不进包根。"""

    def test_model_tools_absent_from_package_root(self) -> None:
        """load_robot_model/RobotModelBundle/ModelKinematics 不可从包根获取。"""
        for name in ("load_robot_model", "RobotModelBundle", "ModelKinematics"):
            assert not hasattr(src.robot, name)
            assert name not in src.robot.__all__
