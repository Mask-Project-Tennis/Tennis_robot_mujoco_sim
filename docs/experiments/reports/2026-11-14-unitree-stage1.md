# 阶段 1：机器人描述与 RM65 兼容接入 验收报告
日期：2026-11-14（Asia/Shanghai）；评审处理更新 2026-11-15
状态：5 个任务全部完成并通过验收；默认行为与阶段 0 基线逐位一致。
外部代码评审（0 Critical / 1 Important / 8 Minor）已于 2026-11-15 全部处理（见文末"评审处理"），修复提交 236c49e。
**到此停止，未进入阶段 2。**

阶段 0 基线报告：[2026-11-14-unitree-baseline.md](2026-11-14-unitree-baseline.md)（未跟踪的工作区文档）。

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

### 新增测试（评审处理后 58 passed；初版 44）
```
$PY -m pytest tests/test_robot_profile.py tests/test_model_binding.py \
    tests/test_robot_factory.py tests/test_robot_profile_entrypoint.py -q
→ 58 passed（初版 44 + 评审修复 14）
```

### 阶段 0 基线七文件（无新增失败）
```
$PY -m pytest tests/test_robot_interface.py tests/test_real_runner.py \
    tests/test_fake_robot.py tests/test_planning_env.py tests/test_replay.py \
    tests/test_actuator_modes.py tests/test_kinematics.py -q
→ 93 passed, 2 skipped, 1 warning in 16.68s
```
与阶段 0 基线（93/2/1）完全一致；2 个跳过仍为 C++ 扩展不可导入（cp313 vs Py3.11，既有问题未修复），不属新增缺陷。

评审补充回归（审查者要求补入，直接覆盖 create_runner 组装）：
`tests/test_create_runner.py` → 3 passed。

### 全量套件（评审修复后）
```
$PY -m pytest tests/ -q --ignore=tests/test_cpp_backward_pass.py --ignore=tests/test_cpp_forward_pass.py
→ 700 passed, 2 skipped in 34.68s（--collect-only 702 项）
```
忽略的两个 C++ 文件为基线即坏的既有收集错误（cp313 扩展 vs Py3.11，阶段 0 已裁定）。

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

证据文件（评审 M8 补全）：
`C:\Users\Nothing\Documents\Codex\2026-11-14\new-chat\outputs\stage0-evidence\stage1_seed42.npz`
（52,598 bytes，SHA256 `7c04e4eb3ed041566d0926b639522ce2b05694873de13ef7c4c25fb220b0e154`；
同目录含阶段 0 基线 npz 与比对脚本 compare_npz.py，不入库）。

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

1. **main 与 create_runner 各校验一次描述**：重复纯模型加载换取入口与组装两层防御，未为省加载启动任何线程。
   加载次数全口径（评审 M5/M8 勘误，2026-11-15 实测复核）：
   - 同步校验路径：初版真机 3 次（main / create_runner / build_robot_arm 各 1），mock 2 次（main / create_runner）；
     评审修复后 `build_robot_arm` 复用 create_runner 的校验 binding，真机降为 2 次，与 mock 一致。
   - 运行期另有规划相关加载（create_runner 的 PlanningEnv 1 次 + AsyncReplanner 后台 2 次），与本阶段改动无关。
   - 单次加载实测开销：首次含中文路径临时复制约 2.2 s，此后缓存目录命中约 0.02 s，不影响启动。
2. **build_robot_arm 对 mock 路径同样拒绝非 rm65 描述**（评审 M5 后从"仅身份"收紧为 validate_rm65_profile 全量名称/TCP/路径校验）：FakeRobot 是 RM65 专属 6 维 mock，放行不符描述会静默错维度；后续机器人 mock 需先扩展。
3. **测试文件两处笔误当场修复**（capsule `fromto` 属性、MuJoCo 单 body ≤6 dof 需嵌套、`is`→`isinstance`、正则大小写）：均为测试自身缺陷，断言意图未变。
4. **直接在 main 分支逐任务提交**：仓库惯例（近 5 次提交均在 main），未 push。
5. **计划文件与 SDD 工作区保持未跟踪**：项目惯例计划不入库。

## 结论

阶段 1 五项任务全部验收通过：描述层→绑定层→校验层→组装层四层就位，真机入口已接入 `--robot-config`，
RM65 默认行为与阶段 0 逐位一致。有证据可开始阶段 2（D2 模型接入与多机器人参数结构）。

## 评审处理（2026-11-15，commit 236c49e）

外部评审结论 "Ready to merge: With fixes"（0 Critical / 1 Important / 8 Minor）。逐条处理：

| 项 | 处理 |
|---|---|
| I1 validate 名称/TCP 拒绝分支无测试 | **已修**。新增 `test_rejects_wrong_joint_names`（真实模型 + 左臂全套名 l_joint/torque_l_joint，绑定成功、校验拒绝）与 `test_rejects_wrong_tcp_site`（tcp_site_name=hit_target，模型中真实存在的另一 site）；七关节缺名测试改名归位到 `TestBinderRejection`。突变验证：临时禁用名称比较/TCP 比较后新测试均转红 |
| M1 手构 profile 维度/重名不设防 | **已修**。`bind_robot_model` 开头 `_validate_profile_shape`（执行器数=主动关节数、三组名称组内不重复、主动/保持不重叠）+ 3 个拒绝测试。裁定：校验放绑定层（公共入口）而非 `RobotProfile.__post_init__`（保持数据类纯描述容器），docstring 写明 |
| M2 混合键类型 TypeError | **已修**。`sorted(map(str, ...))` + 测试（int/str 混合未知键 → ValueError 含路径） |
| M3 model_path 存在性 + 校验顺序 | **已修**。loader 加 `is_file()` 检查（FileNotFoundError 含描述文件与模型双路径，位于字段校验之后不遮蔽字段诊断）+ 目录路径测试；`_resolve_profile` 先 `validate_rm65_profile` 廉价预检再加载模型，非 RM65 描述得到"仅支持 RM65"而非 loader 模型诊断 |
| M4 tcp_site_id<0 死条件 | **已修**。删除（bind 的 _lookup 已保证 ≥0），布局校验注释说明 |
| M5 mock 不校验名称 + 加载次数报告失真 | **已修**。拆出 `validate_rm65_profile` 纯描述校验（mock 路径全量过名称/TCP/路径）+ `load_validated_rm65_binding` 公共 helper + `build_robot_arm(binding=...)` 复用（真机同步加载 3→2 次，有 spy 测试证明传入 binding 不再加载且仍拒乱序布局）；Ruling 1 加载口径已写全 |
| M6 入口测试死脚手架 | **已修**。删恒真断言（real_start/noqa）；spy 扩到 `AsyncReplanner.start` 断言 0 次；新增 `TestMainCli` 两个 main() CLI→YAML 路径测试 |
| M7 import/类型标注 | **已修**。入口脚本 import 合并顶层化（删除 `_resolve_profile` 内 4 处函数内导入）；`_require_*` 与 `mutate` 补 `dict[str, Any]`/`Callable` 标注 |
| M8 报告证据弱 | **已修**（本节 + Ruling 1 勘误 + npz 绝对路径与 SHA256 + 阶段 0 基线报告链接） |
| R3 CLI 级错误处理 | **已修**。main() 捕获 --robot-config 加载/校验的 FileNotFoundError/ValueError → logger.error + sys.exit(2)，测试断言退出码 2 且 create_runner 未被调 |
| R2 test_create_runner 入回归 | **已修**（见"测试结果"节） |
| R5 AGENTS.md 测试总数 | **已修**（另提交） |

不采纳/搁置（含理由）：
- **R4 校验下放 `__post_init__`**：见 M1 裁定，绑定层校验已覆盖公共入口路径，数据类保持纯容器。
- **非标量关节 ball 拒绝测试**：与 free 走同一分支（`_SCALAR_JOINT_TYPES` 元组成员检查），评审未要求，不加重复用例。
- **main() 指标 print 输出**：评审自身已 Declined（AGENTS CLI 豁免）。

回归验证：全量 700 passed / 2 skipped（排除 2 个基线即坏 C++ 文件）；ruff 0.15.15 全部改动文件通过；
Mock 入口冒烟缺省 vs 显式 `--robot-config` 最终指标逐项一致（既有 FakeRobot 首步 TCP 超限行为不变）。
