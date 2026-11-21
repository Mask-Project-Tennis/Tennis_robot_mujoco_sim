# src/robot — 机器人描述、模型资产与装配

## 何时读

- 改机器人配置、球拍档案、R1D 模型片段或 MuJoCo 装配逻辑时。
- 排查 mesh/XML 路径、依赖闭包、模型哈希、关节限位或 TCP 绑定时。
- 只需要共享结果类型（`RacketAssembly` / `RobotModelBundle`）时。

## 职责

| 路径 | 职责 |
|------|------|
| `model_assets/` | 模型文件级依赖扫描：引用解析、路径归一化、闭包与缺失清单 |
| `model_building/` | 模型装配：R1D 模板、球拍、场景、加载与绑定 |
| `model_building/types.py` | `RacketAssembly` / `RobotModelBundle` 唯一类定义 |
| `robot_profile.py` | `RobotProfile`：`configs/robots/*.yaml` 的身份/路径/关节语义 |
| `racket_profile.py` | `RacketProfile`：球拍几何/质量/COM/惯量/安装/来源 |
| `runtime_parameters.py` | `RobotRuntimeParameters` / `SceneParameters`：运行参数纯参数层 |
| `model_binding.py` | `ModelBinding`：名称 → qpos/qvel/actuator 地址 |
| `model_factory.py` | 兼容 facade：`load_robot_model` / `collect_model_dependencies` / 模板生成 |
| `model_kinematics.py` | `ModelKinematics`：TCP 位姿/雅可比探针 |
| `rm65_model.xml`、`r1d_*.xml` | MuJoCo 模型与 R1D 片段（生成物勿手改） |

## 稳定入口

```python
from src.robot.model_factory import (
    collect_model_dependencies,
    load_robot_model,
    write_r1d_model_template,
)
from src.robot.model_building import RacketAssembly, RobotModelBundle
from src.robot.robot_profile import load_robot_profile
from src.robot.racket_profile import load_racket_profile
```

- `load_robot_model(profile, parameters, *, racket_profile=None) -> RobotModelBundle`
- `collect_model_dependencies(profile, parameters, *, arm_fragments=()) -> tuple[dict, ...]`
- 旧路径与新包导出同一对象（`is` 相同）；不要复制第二份实现。

## 允许依赖

- `model_assets` 只依赖标准库、`pathlib`、字符串/路径工具；不导入 MuJoCo、`src.sim`、`src.ilqt`。
- `model_building` 依赖 `model_assets`、`robot_profile`、`racket_profile`、`runtime_parameters`、`model_binding` 与 `src.utils.mujoco_loader`。
- `src.robot` 不得 import `src.sim` / `src.ilqt` / `scripts`；import 期不加载模型、不读 YAML。

## 测试

```bash
python -m pytest \
  tests/test_model_asset_imports.py \
  tests/test_model_building_migration.py \
  tests/test_r1d_model.py \
  tests/test_r1d_assets.py \
  tests/test_racket_profile.py \
  tests/test_refactor_shared_types.py -q
```

## 最小用例

```python
from src.robot.robot_profile import load_robot_profile
from src.robot.runtime_parameters import load_runtime_parameters
from src.robot.model_factory import load_robot_model

profile = load_robot_profile("configs/robots/r1d_right.yaml")
params = load_runtime_parameters("configs/robots/r1d_runtime_right.yaml", profile)
bundle = load_robot_model(profile, params)
print(bundle.model.nq, bundle.binding.tcp_site_id)
```

更多细节见 `model_assets/README.md` 与 `model_building/README.md`。
