# model_building — 机器人模型构建层

## 职责

模型装配实现（P1 仅共享类型，A2 起完整拆分）：

| 模块 | 职责 |
|------|------|
| `types.py` | 共享结果类型 `RobotModelBundle` / `RacketAssembly`（唯一类定义） |
| `template.py` | R1D MJCF 模板生成：片段解析、mesh 路径重写、模板注释与固定结构 |
| `racket.py` | 球拍装配：`RacketProfile` 惯性/几何写入与旧版球拍 XML |
| `scene.py` | 场景数值写入、球场白线、升降重烘、关节限位与模型校验 |
| `loading.py` | `load_robot_model`、`_assemble_bound_model`、臂片段定位、依赖清单 |

旧 facade `src/robot/model_factory.py` 只做 `from ... import X as X` 重导出：
新旧导入返回同一对象（`is` 相同），`inspect.signature` 与 `__module__`
保持指向新模块。

## 分层与依赖方向

```
types ← template ← racket ← scene ← loading
```

- `template.py` 不 import `scene`；仅在 `_build_template_element` 调用期延迟
  import `scene`（模板需要场景数值与球场口径）。
- `scene.py` 不 import `loading`；仅在 `_apply_scene_parameters` 调用期延迟
  import `_apply_fixed_joint_poses`（固定姿态重烘在加载层）。
- 所有模块 import 期不加载 MuJoCo 模型、不读 YAML、不起线程。

## 稳定入口

```python
from src.robot.model_factory import load_robot_model, collect_model_dependencies
from src.robot.model_building import RacketAssembly, RobotModelBundle
```

## 测试

```bash
python -m pytest tests/test_model_building_migration.py tests/test_refactor_shared_types.py -q
```


## 何时读

改 R1D 模板/球拍装配/场景数值/模型加载，或排查 mesh、关节限位、TCP 绑定时。

## 允许依赖

`model_building` 可依赖 `model_assets`、`robot_profile`、`racket_profile`、
`runtime_parameters`、`model_binding` 与 `src.utils.mujoco_loader`；
不得 import `src.sim` / `src.ilqt` / `scripts`，import 期不加载模型、不读 YAML。

## 最小用例

```python
from src.robot.robot_profile import load_robot_profile
from src.robot.runtime_parameters import load_runtime_parameters
from src.robot.model_factory import load_robot_model

profile = load_robot_profile("configs/robots/r1d_right.yaml")
params = load_runtime_parameters("configs/robots/r1d_runtime_right.yaml", profile)
bundle = load_robot_model(profile, params)
print(bundle.model.nq)
```
