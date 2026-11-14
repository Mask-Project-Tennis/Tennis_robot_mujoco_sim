# 阶段 1：机器人描述与 RM65 兼容接入 验收报告
日期：2026-11-14（Asia/Shanghai）
状态：5 个任务全部完成并通过验收；默认行为与阶段 0 基线逐位一致。**到此停止，未进入阶段 2。**

## 交付内容（对应任务 1.1–1.5）

| 任务 | 交付 | 提交 |
|---|---|---|
| 1.1 机器人描述 | `src/robot/robot_profile.py`（RobotProfile/load_robot_profile/default_rm65_profile_path）+ `configs/robots/rm65.yaml` + `tests/test_robot_profile.py`（19 tests） | 0fcfd22 |
| 1.2 模型名称绑定 | `src/robot/model_binding.py`（ModelBinding/bind_robot_model，按名称解析、只接受标量关节、校验执行器传动）+ `tests/test_model_binding.py`（11 tests） | d018a07 |
| 1.3 驱动选择 | `validate_rm65_runtime` + `build_robot_arm`（runner_factory.py）+ Protocol docstring 通用化 + `tests/test_robot_factory.py`（10 tests） | 2a09aaf |
| 1.4 入口接入 | `scripts/run_real_robot.py`：`--robot-config` CLI、`_resolve_profile` 前置校验、`create_runner(..., profile=None)` 可选参数、`build_robot_arm` 替换驱动分支、`profile.model_path` 替换硬编码路径 + `tests/test_robot_profile_entrypoint.py`（4 tests） | 099bdeb |
| 1.5 回归交接 | 本报告 + src/README.md、scripts/README.md、AGENTS.md 索引更新 | 本次 |

## 测试结果

解释器 `C:\Users\Nothing\.conda\envs\mujoco_tennis\python.exe`（Python 3.11.15），仓库根目录 `$PY -m pytest`。

### 新增测试（44 passed）
```
$PY -m pytest tests/test_robot_profile.py tests/test_model_binding.py \
    tests/test_robot_factory.py tests/test_robot_profile_entrypoint.py -q
→ 44 passed in 3.76s
```

### 阶段 0 基线七文件（无新增失败）
```
$PY -m pytest tests/test_robot_interface.py tests/test_real_runner.py \
    tests/test_fake_robot.py tests/test_planning_env.py tests/test_replay.py \
    tests/test_actuator_modes.py tests/test_kinematics.py -q
→ 93 passed, 2 skipped, 1 warning in 16.68s
```
与阶段 0 基线（93/2/1）完全一致；2 个跳过仍为 C++ 扩展不可导入（cp313 vs Py3.11，既有问题未修复），不属新增缺陷。

### 入口冒烟（Mock，不接真机）
```
$PY scripts/run_real_robot.py --mock --max-steps 10                  # 缺省
$PY scripts/run_real_robot.py --mock --max-steps 10 --robot-config configs/robots/rm65.yaml
```
两次输出逐项一致（总步数 1 / 安全步数 1 / 球不可达 False / 安全失败 True）。"安全失败: True" 经 `git stash` 回退基线版本入口对比确认为既有 FakeRobot 行为（首步 qdot 超限触发 SafetyMonitor），非本次引入。

## 固定种子轨迹比对（与阶段 0 基线）

```
$PY scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --seed 42 --no-plot \
    --dump-trajectory <工作区>/stage1_seed42.npz
```

| 指标 | 阶段 0 | 阶段 1 |
|---|---:|---:|
| 击球类型 | active | active |
| 击球位置误差 m | 0.022551 | 0.022551 |
| 速度误差 m/s | 1.980156 | 1.980156 |
| 击球时间误差 ms | 15.0 | 15.0 |
| 拍面击球速度 m/s | 1.001 | 1.001 |
| 出球速度 m/s | 5.665 | 5.665 |

数组级比对（u / q_actual / timestamps / tcp_pos / ball_pos / init_q / init_q_left / dt / hit_step，
形状一致 + `allclose(rtol=0, atol=1e-10)` + 整数全等）：**全部通过，9 个字段 max|Δ| 均为 0.000e+00（逐位一致）**。
q_desired 在默认力矩模式下双方为空，未据此声明位置回放可用。

## 代码检查

`ruff check`（0.15.15）对本阶段 9 个改动文件：**All checks passed**。
过程中修复 1 处 E402（测试文件 sys.path 注入后的导入顺序，按 tests/test_create_runner.py 既有惯例重排）。

## 边界与限制（未实现，属阶段 2+）

- D2（宇树）模型、规划适配、SDK 驱动：未实现，未安装任何 SDK，未猜测 D2 限位（URDF 的 0 不是有效约束）。
- 七关节绑定能力仅由合成模型测试证明（Task 1.2），不代表七关节规划/仿真已适配；运行入口仍只接受 RM65。
- 真机驱动（mock=False）路径已具备校验+构造，但未连接任何硬件验收。
- 升降（R1-D）、视觉、灵巧手：未涉及。
- constants.py / RealRobotConfig / MPC 算法 / 轨迹格式 / C++：零改动。

## 执行期决策（Rulings）

1. **main 与 create_runner 各校验一次描述**：重复一次纯模型加载（~0.3s），换取入口与组装两层防御，未为省加载启动任何线程。
2. **build_robot_arm 对 mock 路径同样拒绝非 rm65 身份**：FakeRobot 是 RM65 专属 6 维 mock，放行其他身份会静默错维度；后续机器人 mock 需先扩展。
3. **测试文件两处笔误当场修复**（capsule `fromto` 属性、MuJoCo 单 body ≤6 dof 需嵌套、`is`→`isinstance`、正则大小写）：均为测试自身缺陷，断言意图未变。
4. **直接在 main 分支逐任务提交**：仓库惯例（近 5 次提交均在 main），未 push。
5. **计划文件与 SDD 工作区保持未跟踪**：项目惯例计划不入库。

## 结论

阶段 1 五项任务全部验收通过：描述层→绑定层→校验层→组装层四层就位，真机入口已接入 `--robot-config`，
RM65 默认行为与阶段 0 逐位一致。有证据可开始阶段 2（D2 模型接入与多机器人参数结构）。
