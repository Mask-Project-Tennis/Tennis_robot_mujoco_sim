# 阶段 3 基线核查报告（Task 3.0：阶段交接与基线门槛）

- **日期**：2026-11-16（Asia/Shanghai）
- **工作目录**：`.worktrees/stage3-sim`（分支 `unitree/stage3-sim`，独立 worktree）
- **性质**：只读基线核查，本片不改任何生产代码/测试/配置
- **结论先行**：环境与阶段 2 交接记录完全一致；阶段 2 目标测试 195 passed；全量回归 846 passed / 2 skipped（既有口径）；RM65 seed42 逐位复现，NPZ 整文件 SHA256 与参考值相同；计划"方案选择与已查明缺口"9 条源码证据全部核实成立（1 条措辞级偏差，见"计划-源码偏差清单"）；无交接失败，下游切片（3.1+）可开工。

## 1. git HEAD / status

```
$ git rev-parse HEAD
754ccc0840640836d74141de4c6b6b3378aed715
$ git log --oneline -1
754ccc0 feat(robot): R1-A7-D 模型与运动学链——参数层/安全加载器/双侧场景/运动学/预览 CLI（阶段 2）
$ git branch --show-current
unitree/stage3-sim
$ git status --porcelain
（空 —— 工作树干净，本片写入前无任何未提交改动）
```

- HEAD 与施工计划标注的 checkout `754ccc0` 一致；主树（`F:/刀客塔/研究/网球机器人/工程/Code/Sim`，分支 `main`）同为 `754ccc0`，tracked 文件无修改（仅两个 untracked：2026-11-14 旧基线报告草稿与一个 .lnk），故主树源码与 worktree 逐字节一致。
- HEAD 提交时间 2026-11-16 09:53:43 +0800。

## 2. 解释器与库版本

| 项 | 值 |
|---|---|
| 解释器 | `C:\Users\Nothing\.conda\envs\mujoco_tennis\python.exe`（Python 3.11.15） |
| 平台 | Windows-10-10.0.19045-SP0（Git Bash），`PYTHONUTF8=1` |
| MuJoCo | 3.9.0 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| PyYAML | 6.0.3 |
| pytest | 9.0.3 |
| ruff | 0.15.15 |

与阶段 2 交接报告"环境"节一致（阶段 2 未记 scipy/pytest 版本，本报告补全；无版本漂移）。

## 3. MuJoCo / C++ 导入能力核查

```
$PY -c "import mujoco; from src.utils.mujoco_loader import load_mujoco_model; ..."
→ mujoco import OK: 3.9.0
→ RM65 model load OK: nq=19 nv=18 nu=12   （rm65_model.xml，双臂12+球7）
```

C++ 扩展（**既有问题，非本片新增**）：

```
$PY -c "from src.cpp import iLQR_Core"
→ ImportError: cannot import name 'iLQR_Core' from 'src.cpp'
磁盘文件：src/cpp/iLQR_Core.cp313-win_amd64.pyd（cp313 ABI 标签，Python 3.11 下不可见）
```

- `src/cpp/solver_cpp.py:36-45` 的 `_CPP_AVAILABLE=False` 回退正常生效：`build_solver()`（`src/ilqt/solver.py:387-399`）try C++ → except ImportError 回退纯 Python `ILQTSolver`，当前机器实际运行的是 Python 求解器。
- pytest 收集时同一根因表现为 `ModuleNotFoundError: No module named 'src.cpp.iLQR_Core'`（tests/test_cpp_backward_pass.py 收集错误）。
- C++ 内核维数硬编码（`src/cpp/types.h:16-18` `constexpr int kNQ = 6; kNU = 6;`），印证计划 Global Constraints"不得让七维输入落进六维内核"的必要性。

## 4. 阶段 2 三项 P2 修复在 HEAD 的核对

三项修复（阶段 2 报告 §2.10）符号、语义与回归测试均在 `754ccc0`，无需重复修：

| # | 修复 | 源码证据（文件:行） | 回归测试 |
|---|---|---|---|
| P2-1 | SDK 对照按完整关节名查询（不按列表位置配对） | `scripts/tools/inspect_robot_model.py:211-248` `_sdk_entry_for_joint`（语义干匹配 + 未知名 ValueError→退出码 2）；:431 报告字段消费 | `tests/test_inspect_robot_model.py:233` `test_sdk_index_follows_joint_name_not_order`、:290 `test_sdk_lookup_rejects_unknown_joint_name` |
| P2-2 | 跨盘符导出 mesh 重定位（复制到导出目录 `meshes/` 并改相对引用） | `src/robot/model_factory.py:569-600` `_retarget_mesh_paths`（relpath 失败分支 copyfile + 正斜杠相对引用） | `tests/test_r1d_model.py:651` `test_export_relocates_meshes_when_relative_unavailable`、:695 `test_cross_drive_export_loads_directly`（双侧参数化） |
| P2-3 | 临时副本目录补属主写权限 | `src/utils/mujoco_loader.py:150`（`_setup_temp_copy` 调用）、:158-178 `_ensure_dirs_writable`（仅对副本目录 OR 属主写位） | `tests/test_mujoco_loader.py:356` `test_temp_copy_dirs_get_write_permission` |

上述测试在本片目标测试与全量回归中全部通过（§5）。

## 5. 阶段 2 目标测试与 seed42（实际命令与结果）

### 5.1 阶段 2 目标测试（12 个文件）

```
$PY -m pytest tests/test_robot_profile.py tests/test_model_binding.py tests/test_robot_factory.py \
    tests/test_robot_profile_entrypoint.py tests/test_runtime_parameters.py \
    tests/test_rm65_runtime_parameters.py tests/test_d2_assets.py tests/test_r1d_assets.py \
    tests/test_r1d_model.py tests/test_model_kinematics.py tests/test_inspect_robot_model.py \
    tests/test_robot_package_init.py -q
→ 195 passed in 33.87s
```

逐文件分解（与阶段 2 报告 §"总数分解"精确吻合，合计 195）：
22（robot_profile）+ 14（model_binding）+ 16（robot_factory）+ 8（robot_profile_entrypoint）+ 66（runtime_parameters）+ 6（rm65_runtime_parameters）+ 3（d2_assets）+ 5（r1d_assets）+ 25（r1d_model）+ 12（model_kinematics）+ 14（inspect_robot_model）+ 4（robot_package_init）= **195**。

### 5.2 全量回归（排除口径，阶段 2 同款命令）

```
$PY -m pytest tests/ -q --ignore=tests/test_cpp_backward_pass.py --ignore=tests/test_cpp_forward_pass.py
→ 846 passed, 2 skipped, 4 warnings in 85.36s（复跑第二次 82.39s，同为 846/2）
```

- 2 个 skip 均为 `tests/test_actuator_modes.py:297/:329`（"C++ iLQR_Core 模块未编译"），阶段 1 即有，非新增。
- 4 个 warnings 均为既有（BasicSafetyFilter RuntimeWarning 等）。
- **与阶段 2 交接基线 846 passed / 2 skipped 完全一致，HEAD 无新增失败。**

### 5.3 全量收集口径

```
$PY -m pytest tests/ --collect-only -q
→ 862 tests collected, 1 error in 2.36s
   ERROR tests/test_cpp_backward_pass.py（ModuleNotFoundError: No module named 'src.cpp.iLQR_Core'）
```

与建区冒烟记录一致（862 collected + 1 既有收集错误）。862 = 848（排除双 C++ 文件后 846 passed + 2 skipped）+ 14（test_cpp_forward_pass.py 可收集但全 skip，单独跑验证：`14 skipped in 0.30s`，同一"模块未编译"原因）。

### 5.4 RM65 seed42（brief 验收命令）

```
mkdir -p results/unitree_stage3
$PY scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --seed 42 --no-plot \
    --dump-trajectory results/unitree_stage3/rm65_seed42.npz
→ 击打成功（主动击球）pos_error=0.022551 / vel_error=1.980156 / hit_time_error=15.0ms
   __RESULT__: pos_error=0.022551 vel_error=1.980156 min_dist=0.022551 ... hit_type=active
   hit_time_error_ms=15.0 hit_pos_error=0.022551 v_racket_at_hit=1.001 first_plan_ms=3534.9
```

击打指标与阶段 2 报告记载逐字一致。产物保留在 `results/unitree_stage3/rm65_seed42.npz`（52,598 字节；results/ 为 git-ignored，不提交）。

## 6. NPZ SHA256 对比

```
$PY -c "import hashlib;print(hashlib.sha256(open(r'results/unitree_stage3/rm65_seed42.npz','rb').read()).hexdigest())"
→ 7c04e4eb3ed041566d0926b639522ce2b05694873de13ef7c4c25fb220b0e154
```

| 项 | 值 |
|---|---|
| 参考 SHA256（brief / 阶段 1/2 交接） | `7c04e4eb3ed041566d0926b639522ce2b05694873de13ef7c4c25fb220b0e154` |
| 本机 SHA256 | `7c04e4eb3ed041566d0926b639522ce2b05694873de13ef7c4c25fb220b0e154` |
| 对比 | **逐字符相同** |
| 文件大小 | 52,598 字节（与阶段 1 证据文件一致） |

**结论：本机基线与参考逐位一致，无环境差异需要记录；不换基线、不调参。**

## 7. 可用 / 不可用测试清单

### 可用（全部通过）

- **阶段 2 目标测试 12 文件**：195 passed（§5.1）。
- **全量回归（排除双 C++ 文件）**：846 passed / 2 skipped（§5.2）。
- **历史既有 skip（非新增，不阻塞）**：
  - `tests/test_actuator_modes.py` 2 skip（C++ iLQR_Core 未编译，阶段 1 起）；
  - `tests/test_cpp_forward_pass.py` 14 skip（同根因，可收集）。

### 不可用（既有，阶段 0/1 已裁定，不重编译）

- `tests/test_cpp_backward_pass.py`：**收集错误**（`ModuleNotFoundError: No module named 'src.cpp.iLQR_Core'`；扩展 `iLQR_Core.cp313-win_amd64.pyd` 为 cp313 ABI，当前 Py3.11 环境不可导入）。全量口径需 `--ignore=tests/test_cpp_backward_pass.py --ignore=tests/test_cpp_forward_pass.py`。
- 阶段 3 口径（计划"执行与验收命令约定"）：R1D 默认 Python + FD，不修 C++ ABI；与现状一致。

### 无新增失败

历史 C++ 不可收集/跳过（上述）与本片新增核查**零新增失败**分列：本片所有新跑命令全部通过/一致。

## 8. 计划-源码核对（"方案选择与已查明缺口" + "共享接口契约"逐条）

对照计划 `docs/superpowers/plans/2026-11-16-unitree-stage3-construction.md`（主树，只读）。格式：计划声称 → 源码实际（worktree 文件:行）→ 是否一致。

| # | 计划声称 | 源码实际 | 一致？ |
|---|---|---|---|
| 1 | `RobotModelBundle` 已提供 profile/parameters/model/binding；`load_robot_model` 目前专用于 R1D，不能假设已支持 RM65 | `src/robot/model_factory.py:140-154` 冻结 dataclass 四字段齐备；:205-262 `load_robot_model`，:230-235 `robot_id != "unitree_r1d"` 即 ValueError（错误信息明示"RM65 等继续使用既有模型入口"） | ✅ 一致 |
| 2 | `ModelKinematics` 只有状态/FK/Jacobian，无执行器控制和时间推进，不是完整 RobotEnv | `src/robot/model_kinematics.py:37-199`：成员仅 `n_dof/set_state/get_state/tcp_pose/tcp_jacobian`；:7-8 docstring 明写"无动力学 step、控制器…不实现 RobotEnv 完整协议，不引入 IK/规划" | ✅ 一致 |
| 3 | `PlanningEnv` 有 `qpos[:NQ]`、左臂 `+6` 等布局假设；`RobotEnv` 还要求 LEFT_ARM_NQ/init_q_left | `src/ilqt/planning_env.py:43-45` NQ/NX/NU=6/12/6 类常量；:122/131/140/168 `qpos[: self.NQ]`；:177-178 `qpos[self.NQ : self.NQ+6]=0`（左臂紧跟其后 6 位假设）；:52 `LEFT_ARM_NQ=6`、:102 `init_q_left`；:65-66 默认加载 rm65_model.xml。协议 `src/ilqt/robot_env_protocol.py:32-33` 要求 `LEFT_ARM_NQ: int` / `init_q_left: np.ndarray` | ✅ 一致 |
| 4 | `build_solver()` 当前不接收环境，优先导入 C++；Python 默认 use_analytical=True，不是自动安全的七维路径 | `src/ilqt/solver.py:387` `def build_solver() -> Any:`（零参数）；:396-399 try `src.cpp.solver_cpp` → except ImportError 回退 Python；:25 `use_analytical: bool = True` 默认。补充：`src/cpp/types.h:16-18` kNQ=6/kNU=6 硬编码 | ✅ 一致 |
| 5 | `robot_limits.check_limits_on_trajectory` 仍出现 `[6:]` | `src/ilqt/robot_limits.py:496、505` `X[k][6:]`（q/qdot 分界硬编码 6） | ✅ 一致 |
| 6 | JT 初始化读取执行器/关节前缀 | `src/ilqt/mpc_helpers.py:167-179` 力矩版 `compute_jacobian_init_control(env: "RM65Env", …)` 用 `actuator_ctrlrange[:env.NU]`；`src/ilqt/jt_init.py:16-24` `_get_jnt_range` 用 `actuator_trnid[i,0]`（"前 NU 个执行器按序对应臂关节"的索引前缀假设）；另有 RM65 关节角色硬编码（`mpc_helpers.py:199` `U[k,5]` fix_joint5、:305-306 R_joint1 → `R_schedule[:,0]`） | ⚠️ 基本一致，措辞偏差（见偏差清单 #1） |
| 7 | `PredictiveSafetyFilter` 使用 RM65 body 名和世界 X 墙；找不到 body 时当前可得到空列表 | `src/ilqt/components/predictive_safety.py:35-44` `X_WALL_BODY_NAMES=("r_link1"…"r_link6","r_flange","r_racket_body")`；:123-128 `ids if i >= 0` 静默丢弃未知名（可得空列表）；:171 `all(data.xpos[bid,0] <= -0.1 …)` 空列表时**空真放行** | ✅ 一致 |
| 8 | `SimComponent` 依赖 RM65Env、前缀执行器、球拍/球名称子串与 RM65 推回逻辑；其"碰撞窗口"不能关闭 R1D 的机身碰撞保护 | `src/ilqt/components/sim_component.py:11` import RM65Env、:36/50 构造器类型 `env: RM65Env`；:204-207 geom 名子串判定 `("ball" in g*) and ("racket" in g*)`；:180-196 X 墙 PD 推回（`300·(init_q−q)−20·qdot` + `_hard_x_body_ids`）；:193-194 `actuator_ctrlrange[:self._nu]`（前 NU 执行器）；:172/219 `env.set_arm_collision(enable_collision)` 碰撞窗口。`src/sim/rm65_env.py:223-238` `set_arm_collision(False)` 会把 ball 之前**全部** geom 的 contype/conaffinity 置 0（含机身），且依赖 `body("ball")` 名称——证实计划顾虑：该窗口机制在 R1D 上会连带关掉机身碰撞 | ✅ 一致 |
| 9 | 阶段 2 R1D 模板无球、无执行器范围，球拍 geom 的 contype/conaffinity 为 0；只换 profile 不会产生击球闭环 | `src/robot/r1d_model_{left,right}.xml`：`ball` 子串出现 0 次；actuator 段 7 个 `<motor gear="1">` **无 ctrlrange/forcerange**（right 模板 :125-133）；`racket_face` geom `contype="0" conaffinity="0"`（right :103；实为全模板 geom 均 0，含 torso_box/ground） | ✅ 一致 |
| 10 | 共享接口契约为**拟新增**（"不表示已实现"） | `src/sim/` 仅 `__init__.py/env.py/hit_detection.py/replay.py/rm65_env.py/v11_visuals.py/viewer.py`；`sim_config.py/tennis_scene.py/bound_robot_env.py/env_factory.py` 均不存在；全库 grep 无 `SimTaskConfig/TennisSceneBundle/BoundRobotEnv/ProfileSimEnv/build_tennis_scene/build_r1d_environments/load_sim_task_config/has_forbidden_contact/run_r1d_episode` 任一符号 | ✅ 一致（契约尚未实现，符合计划声明） |

Global Constraints 抽查：`configs/robots/r1d_runtime_left.yaml:22-29` kp/kd/q_lower/q_upper/qdot_max/qddot_max/torque_max/max_tcp_speed 显式 null，未从 RM65 补值 —— 与"未知限值继续保留 null"一致（right 侧同构）。

## 9. 计划-源码偏差清单（需回写计划）

**无实质矛盾；1 条措辞级 + 2 条索引观察**（更新计划由主代理执行，本片只记录）：

1. **[措辞] JT 初始化"执行器/关节前缀"**（计划 §方案选择第 6 条）：源码实际是**索引前缀假设**（`actuator_ctrlrange[:NU]`、`actuator_trnid[i]` 按"前 NU 个执行器属于受控臂"取关节），并叠加 `env: "RM65Env"` 类型标注与 joint1/joint5 角色硬编码；**没有** `r_`/motor 之类的**名称**前缀查询。语义（对执行器布局耦合、非按名绑定）成立，建议计划在 3.8 切片下发时把该句改为"JT 初始化假设受控臂占据前 NU 个执行器（actuator_trnid/ctrlrange 索引前缀），且类型标注 RM65Env、含 joint1/joint5 角色硬编码"，避免实现者去找不存在的名称前缀代码。
2. **[索引观察] `src/sim/v11_visuals.py` 未列入 AGENTS.md 的 src/sim/ 结构清单**（env/rm65_env/viewer/replay/hit_detection 之外的第 6 个文件）。与施工计划无关，属项目文档漂移，建议收尾任务（3.11）统一维护时补列。
3. **[索引观察] 阶段 2 报告未记录 scipy/pytest 版本**（本报告 §2 已补全实测值 1.17.1 / 9.0.3），不构成偏差，仅供后续环境对账引用。

## 10. 未决事项（交接给 3.1+）

1. **C++ ABI**：`iLQR_Core.cp313-win_amd64.pyd` 在 Py3.11 下不可导入（既有裁定：不修、不重编译）。3.5 实现 `build_solver(backend=…)` 时需保证即使磁盘上有可用扩展，七维输入也不会落进 kNQ=6 内核（计划 Global Constraints 已有此条，本片核实内核确为 6 维硬编码）。
2. **空列表放行语义**：`PredictiveSafetyFilter._resolve_x_wall_body_ids` 空列表 → `_check_x_wall` 空真放行（§8#7），3.7 的 R1D 保护查询不得沿用该行为。
3. **碰撞窗口副作用**：`RM65Env.set_arm_collision(False)` 关闭 ball 之前全部 geom 的碰撞（含机身，§8#8），3.2/3.9 的 R1D `set_arm_collision` 语义（仅切换球—拍接触）必须绕开此实现。
4. **R1D 执行器无范围**：模板 motor 无 ctrlrange/forcerange（§8#9），3.1/3.3 预检须拒绝"无限制 motor 被静默放行"（计划 Review Focus #3 已列）。
5. **`r1d_runtime_{left,right}.yaml` 的 null 限值**在 3.1 的新仿真 YAML（provisional 标记）中给值，不得回写 runtime YAML（Global Constraints 已列，此处确认现状为 null）。
6. 本片产物 `results/unitree_stage3/rm65_seed42.npz` 不提交（git-ignored，留盘备查）。

## 11. 本片执行记录（命令汇总）

| 步骤 | 命令 | 结果 |
|---|---|---|
| git | `git rev-parse HEAD` / `status --porcelain` / `branch --show-current` | 754ccc0…；干净；unitree/stage3-sim |
| 版本 | `$PY --version` + 模块版本脚本 | §2 表 |
| 导入 | mujoco import + `load_mujoco_model(rm65_model.xml)` | OK（nq=19/nv=18/nu=12） |
| 导入 | `from src.cpp import iLQR_Core` | ImportError（cp313 ABI，既有） |
| 目标测试 | 12 文件 `-q` | 195 passed in 33.87s |
| 全量 | `tests/ -q --ignore=cpp_backward --ignore=cpp_forward` | 846 passed / 2 skipped（两次） |
| 收集 | `tests/ --collect-only -q` | 862 collected + 1 error（既有） |
| 单文件 | `test_cpp_forward_pass.py -q` | 14 skipped |
| skip 定位 | 全量 `-rs` | 2 skip 均在 test_actuator_modes.py:297/:329 |
| seed42 | `scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --seed 42 --no-plot --dump-trajectory results/unitree_stage3/rm65_seed42.npz` | 击打成功（主动击球），指标与阶段 2 逐字一致 |
| SHA256 | `hashlib.sha256(open(npz).read())` | 与参考值逐位相同（§6） |

## 12. 自审

- **只读约束**：除本报告文件外，worktree 无任何写入（`git status` 仅本文件新增；`results/` git-ignored 产物除外）。主树零改动。
- **基线未动**：环境差异不存在（SHA 逐位一致），未发生"换基线/调参重跑凑数"。
- **证据可复现**：所有命令完整记录于 §11，输出为真实终端输出（含墙钟秒数）。
- **交接失败项**：无。下游 3.1 可开工。
- **限制**：本片未跑 ruff/mypy（无代码改动，不在验收范围）；未验证 Linux 渲染路径（沿阶段 2 边界）。
