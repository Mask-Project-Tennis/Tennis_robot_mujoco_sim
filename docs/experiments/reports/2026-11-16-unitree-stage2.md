# 阶段 2：R1-A7-D 模型链构建 验收报告（Task 2.8 收官）

- **日期**：2026-11-16（Asia/Shanghai）
- **状态**：10 个任务（2.0 审计 + 批 1 通用切片 2.1/2.2/2.3/2.4 + 新版模型切片 2.5a/2.5b/2.6/2.7 + 2.8 回归交接）全部完成并通过评审；RM65 无新增回归（seed42 逐位一致）；R1-A7-D 模型与运动学进入可测试计算入口，**但仅验收模型与运动学层**（详见"边界与限制"）。
- **型号更正**：阶段 2 中途（2026-11-15）用户裁定目标为 **R1-A7-D 整机单臂**，独立 D2 模型+SDK 路线作废；Task 2.0 产出审计报告后重写模型切片（原 2.5-2.7 → 2.5a/2.5b/2.6/2.7 新版），2.8 按修订卡执行。D2 资产保留为历史资产，不作 R1-A7-D 验收依据。
- **到此停止，未进入阶段 3。**

阶段 1 报告：[2026-11-14-unitree-stage1.md](2026-11-14-unitree-stage1.md)。
型号/接口事实源：[2026-11-15-r1a7d-model-interface-audit.md](2026-11-15-r1a7d-model-interface-audit.md)（Task 2.0 审计报告，模型部分引用源）。

## 交付内容（任务 → 提交）

| 任务 | 交付 | 提交 |
|---|---|---|
| 2.0 型号审计 | R1-A7-D 型号/关节表/变换链/冲突清单/SDK 话题审计报告（证据审查验收） | b86b604 |
| 2.1 D2 资产（批 1） | 固定宇树 D2 官方资产 + 可复建七关节裸臂 + manifest（历史资产，型号更正后不作 R1-A7-D 验收） | 57b699a |
| 2.2 参数结构（批 1） | `src/robot/runtime_parameters.py`（RobotRuntimeParameters/SceneParameters + load/validate）+ `configs/robots/d2_runtime.yaml` | 7846a7c |
| 2.3 RM65 适配（批 1） | `src/ilqt/robot_parameters.py` 旧参数来源兼容适配器（sim/real 两来源 → 统一参数结构） | bd4961f |
| 2.4 加载器（批 1） | `mujoco_loader` 内容感知缓存 + `xml_text` 参数化加载 | 888fc8c |
| 2.5a R1D 官方资产 | `assets/unitree_r1d/`（r1d.urdf + 36 STL + manifest）+ `build_r1d_model.py` 导出双侧裸臂/环境（q_L bake）/资产片段 + `src/robot/r1d_*.xml` | 2120ebb |
| 2.5b R1D 场景组装 | `src/robot/model_factory.py`（`load_robot_model`→`RobotModelBundle`）+ 双侧模板 `r1d_model_{left,right}.xml` + `configs/robots/r1d_{left,right}.yaml` 与 `r1d_runtime_{left,right}.yaml` | b759ab4 |
| 2.6 R1D 运动学 | `src/robot/model_kinematics.py`（`ModelKinematics`）+ `tests/r1d_reference.py` 独立整机 URDF FK 对照器 | ef7c5c8 |
| 2.7 R1D 预览 | `scripts/tools/inspect_robot_model.py`（JSON 报告 + 三视图，退出码 0/2/3） | 94a5777 |
| 2.8 回归交接 | 本报告 + 入口隔离/导出纯度测试 + `src/robot/__init__.py` 导出 + 四处索引更新 | 5a1466a + 勘误 f8e74de |
| 2.9 复审修复 | 独立复审 I1-I3：显式 q 限位拒绝不一致、held_q 构造期应用、源目录不可写时 `xml_text` 降级到 ASCII 副本（+4 测试） | 10fc5f2 |
| 2.10 复审修复 | 独立复审 P2×3：SDK 对照按完整关节名查询、跨盘导出 mesh 重定位、临时副本目录补属主写权限（+8 测试） | 合并入成品提交 |

每任务均经"实现者 → 评审（规格+代码两步）→ 主代理提交"流程（SDD ledger：`.superpowers/sdd/2026-11-15-unitree-stage2-construction/progress.md`，git-ignored）。

**提交形态**：收尾时阶段 2 全部提交（上表 2.2 起 13 个 + 2.10 修复）已合并（squash）为**单一成品提交**（即本报告所在提交）；上表提交号为逐任务时点的原始记账，仅供 ledger 追溯，不再存在于当前分支历史。

## 环境

- 解释器 `C:\Users\Nothing\.conda\envs\mujoco_tennis\python.exe`（Python 3.11.15），仓库根运行，`PYTHONUTF8=1`
- MuJoCo 3.9.0 / NumPy 2.4.6 / PyYAML 6.0.3 / ruff 0.15.15 / Windows 10 19045（Git Bash）

## 测试结果

### 全量回归（最终证据）

```
$PY -m pytest tests/ -q --ignore=tests/test_cpp_backward_pass.py --ignore=tests/test_cpp_forward_pass.py
→ 846 passed, 2 skipped, 4 warnings（collect 848 项，78 个文件；2.9 +4、2.10 +8）
```

- **两个 C++ 测试文件的排除口径（2.9 复审更正）**：`tests/test_cpp_backward_pass.py` 收集即 `ModuleNotFoundError: No module named 'src.cpp.iLQR_Core'`（扩展文件为 `src/cpp/iLQR_Core.cp313-win_amd64.pyd`，cp313 ABI 标签对 Py3.11 不可见，与阶段 0/1 裁定一致，非本阶段引入，不重编译）；`tests/test_cpp_forward_pass.py` **可以收集**，其 14 个用例全部 skip（同一"模块未编译"原因）。即：1 个收集错误 + 1 个可收集但全 skip，"834/2（现 838/2）"是**同时排除这两个文件之后**的口径；若只排除收集失败的那个，结果是 838 passed / 16 skipped。
- **2 个 skip** 均为 `tests/test_actuator_modes.py`（"C++ iLQR_Core 模块未编译"），阶段 1 即有，非新增。
- 4 个 warnings 均为既有（BasicSafetyFilter RuntimeWarning 等）。

### 总数分解（精确对齐 846 = 700 + 146）

| 来源 | 文件 | 数量 | 累计 |
|---|---|---:|---:|
| 阶段 1 基线 | — | 700 | 700 |
| 2.1 | tests/test_d2_assets.py | 3 | 703 |
| 2.2 | tests/test_runtime_parameters.py | 66 | 769 |
| 2.3 | tests/test_rm65_runtime_parameters.py | 6 | 775 |
| 2.4 | tests/test_mujoco_loader.py（5→11） | +6 | 781 |
| 批 1 门禁实测 | — | — | 781（与 ledger 记录一致） |
| 2.5a | tests/test_r1d_assets.py 5 + test_mujoco_loader.py（11→12）+1 | 6 | 787 |
| 2.5b | tests/test_r1d_model.py | 19 | 806 |
| 2.6 | tests/test_model_kinematics.py | 11 | 817 |
| 2.7 | tests/test_inspect_robot_model.py | 11 | 828 |
| 2.8 | test_robot_profile_entrypoint.py +2（入口隔离参数化双侧）、test_robot_package_init.py +4 | 6 | 834 |
| 2.9 复审修复 | test_r1d_model.py +2（显式限位双侧）、test_model_kinematics.py +1、test_mujoco_loader.py +1 | 4 | 838 |
| 2.10 复审修复 | test_inspect_robot_model.py +3（SDK 按名跟随双侧 + 未知名拒绝）、test_r1d_model.py +4（mesh 重定位双侧 + 跨盘端到端双侧）、test_mujoco_loader.py +1（副本目录写权限） | 8 | **846** |

备注：ledger 中 2.5a 实现者口径"新 16+1"与提交版 `test_r1d_assets.py` 的 5 个测试函数不一致——2.8 评审终裁：2120ebb 提交版、2.5a 评审包版、当前工作树均为同一 5 函数版本（16 函数版从未存在），真实差异是 ledger 把"17 passed（5 个新 r1d 用例 + loader 文件全部 12 个，其中 11 个为既有用例）"错误记账为"新增 16+1"，**无测试丢失**；本表以当前仓库实测为准，846 可逐文件精确分解。2.7 的 11 个含 live 渲染用例（本机 WGL 实跑 rc=0，无 skip）。

### 入口隔离（2.8 新增，含突变验证）

`test_r1d_profile_still_rejected_by_real_entrypoint`（参数化 r1d_left/r1d_right 真实 profile）：main() 完整解析路径 → 退出码 2，`create_runner` / SDK `RobotInterface.connect` / `AsyncReplanner.start` 调用次数均为零。突变验证：`sys.exit(2)` 改为 `exit(0)` 后双侧用例均转红，还原复绿。

### 导出纯度（2.8 新增，含突变验证）

`import src.robot` 在干净子进程中不加载 mujoco / src.ilqt / src.real / src.sim（ModelBinding/bind_robot_model 经 PEP 562 延迟导出，显式访问才触发 mujoco 加载；模型工具 load_robot_model/RobotModelBundle/ModelKinematics 不进包根）。突变验证：注入 eager `model_binding` 导入后纯度测试转红，还原复绿。

## 固定种子轨迹比对（与阶段 1 证据，数组级）

```
$PY scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --seed 42 --no-plot \
    --dump-trajectory results/unitree_stage2/rm65_seed42.npz
→ 击打成功（主动击球）pos_error=0.022551 / vel_error=1.980156 / hit_time_error=15.0ms
```

旧证据执行前验证：`C:\Users\Nothing\Documents\Codex\2026-11-14\new-chat\outputs\stage0-evidence\stage1_seed42.npz`（52,598 bytes）SHA256 = `7c04e4eb3ed041566d0926b639522ce2b05694873de13ef7c4c25fb220b0e154`，与阶段 1 报告记载一致。

| 字段 | 形状 | allclose(rtol=0, atol=1e-10) | max\|Δ\| |
|---|---|---|---|
| u | (320, 6) | True | 0.000e+00 |
| q_actual | (320, 6) | True | 0.000e+00 |
| timestamps | (320,) | True | 0.000e+00 |
| tcp_pos | (320, 3) | True | 0.000e+00 |
| ball_pos | (320, 3) | True | 0.000e+00 |
| init_q | (6,) | True | 0.000e+00 |
| init_q_left | (6,) | True | 0.000e+00 |
| dt | () 标量 | True | 0.000e+00 |
| hit_step | 整数 | 全等（236=236） | 0 |

**结论：9 个字段逐位一致（最大偏差 0.000e+00）；新旧 npz 整文件 SHA256 相同。** q_desired 双方均为空 (0,6)（力矩模式），不参与比对（阶段 1 先例）。新证据保留在 `results/unitree_stage2/rm65_seed42.npz`（results/ 被 .gitignore 忽略，保留在盘；比对脚本 `.superpowers/sdd/2026-11-15-unitree-stage2-construction/compare_seed42.py`，git-ignored）。

## 入口冒烟（Mock，不接真机）

```
$PY scripts/run_real_robot.py --mock --max-steps 10                                 # 缺省
$PY scripts/run_real_robot.py --mock --max-steps 10 --robot-config configs/robots/rm65.yaml  # 显式
```

两次输出逐项一致：总步数 1 / 安全步数 1 / 球不可达 False / 安全失败 True / 重规划提交 0 / 重规划完成 0。与阶段 1 记录完全一致；"安全失败: True"仍为**既有 FakeRobot 首步安全失败**（首步触发 SafetyMonitor），非本阶段新增缺陷，也不算成功运动。

## 代码检查

`ruff check`（0.15.15）对本任务 3 个新增/修改 Python 文件（`src/robot/__init__.py`、`tests/test_robot_package_init.py`、`tests/test_robot_profile_entrypoint.py`）：**All checks passed**。过程中修复 9 处 F401（纯度测试的包根 from-import 移入测试方法内，语义更贴合"显式访问才触发延迟导出"）。未顺带格式化全库。

## 执行期决策 / Rulings（自 SDD ledger 提炼）

1. **型号更正**（2026-11-15 用户指令）：目标 R1-A7-D 整机单臂，D2 路线作废；停止在跑的 2.5（D2 场景组装），半成品入 `parked-2.5/` 暂存；模型切片以审计报告 + 计划修订版为唯一施工依据。
2. **编译期掩码**（2.5b）：MuJoCo 3.9 运行期改 geom 掩码对已编译模型无效（对照实验：运行期改掩码 ncon=0，编译期 XML 注入 ncon=4）；碰撞诊断一律用编译期 XML 掩码注入的独立诊断模型，不改生产模板（2.6 沿用）。
3. **结构紧凑单元排除对**（2.5b）：官方网格固有交叠（visual=collision 同网格），4 个穿透值 −0.032539/−0.009292/−0.005455/−0.002519 经评审逐项复现，裁定属"有记录的碰撞简化"（3 组相邻排除对），保守替代均不合理。
4. **init_q 肩侧摆 0.2**（2.5b）：任务卡授权范围内将临时准备姿态肩侧摆从 0.3 调至 0.2 rad（右肩 roll 上界 0.2268），消除准备姿态穿透。
5. **LIFT_QL 常量**（2.6）：tests/r1d_reference.py 对照器以 `LIFT_QL=0.0` 常量对齐 r1d_env.xml 的 bake 值；评审裁定取舍可接受（升降冻结于 0，运行时不联动）。
6. **q_L 推导**（2.7）：预览报告的 q_L 从模型 body_pos 推导，两级互检公式 `z(s1)+z(s2) = 2q_L − 0.1688` 经评审独立验证并与 r1d_env.xml 互证。
7. **批 1 并行**：2.1/2.2/2.4 并行派发（文件所有权互斥），批后全量门禁 781/2 精确吻合后提交。
8. **2.5a 实现口径差异**：见测试分解备注（提交版 5 函数，无测试丢失）。

## 边界与限制（阶段 2 只验收模型与运动学）

- **七关节 MPC / RobotEnv 适配 / 仿真入口 = 阶段 3+**：MPCController、RobotLimits、PlanningEnv、RM65Env 均未泛化（Global Constraints 明令留给阶段 3/5）；run_real_robot 入口仍仅支持 RM65（入口隔离测试锁定）。
- **SDK 真机 = 阶段 4+**：未安装 unitree_sdk2、未连硬件；A7 全 7 关节需低层 `rt/lowcmd`（高层 arm_sdk 只覆盖 A5 兼容 13 关节，审计 §6.4）。
- **未知参数清单**（不得包装成能力）：运行限值 kp/kd/qdot_max/qddot_max/torque_max/max_tcp_speed 双侧 runtime YAML 显式 null（URDF effort/velocity 声明值未实机验证，不进安全限值，N0-9）；球拍/torso 为 provisional 几何；击球侧未确认（双侧对称交付，无默认侧）；SDK q=0 与 URDF 零位是否同一物理姿态未确认；实物型号（A5 vs A7）无铭牌证据。
- **指南/URDF 限位冲突 4 项**（腰、双侧腕俯仰、双侧腕滚转）：只记录不裁断，派生模型逐字采用 URDF 值（含右腕俯仰 +0.5326 不对称），真机验收前无解（审计 §5）。
- **Linux 渲染未实测**：2.7 三视图仅本机 Windows WGL 实跑（rc=0，双侧）；osmesa/egl 路径未验证（AGENTS 跨平台规范已遵守，无平台 GL 调用）。
- **模型四行验收的数值证据**在 2.5a/2.5b/2.6 测试内（manifest+哈希 / 双侧真实+乱序合成 / 刚体变换+惯量+负例碰撞 / 独立整机 URDF FK+有限差分）；本报告引用审计报告为模型事实源，不重复罗列数值。
- **延后 Minor 总账：35 项**（2.1×5、2.2×4、2.3×4、2.4×4、2.0×3、2.5a×4、2.5b×6、2.6×1、2.7×4），明细见 SDD progress.md（git-ignored）。分类概述：阶段 3/8 交接项 2（robot_limits inf→None 语义、RM65 身份常量统一归宿）、测试稳健性（模块级 sys.path.insert、固定名临时文件并行竞争、docstring 措辞）、文档完整性（报告数值记录不完整、对照表舍入）、可观测性强化（JSON missing 不可见、缓存摘要重算开销）。
- **D2 历史资产**（assets/unitree_d2/、d2_*.xml、d2_runtime.yaml）保留在库，不作 R1-A7-D 验收依据（原切片处理表裁定）。

## 完成矩阵核对（计划 §6，模型四行按修订版）

| 需求 | 负责切片 | 通过证据 | 状态 |
|---|---|---|---|
| 官方来源、版本、许可状态可追踪 | 2.5a（+2.0 审计） | source_manifest.yaml 37 文件 SHA256 与审计 §1 逐字一致 + 重建测试（两次生成字节一致=仓库成品）；license bsd-3-clause(仓库根) 记录 | ✅ |
| 7 DOF、14 维受控状态、名称地址正确 | 2.5b/2.6 | 双侧真实模型 load 后 nq=nv=nu=7 + 乱序合成模型（qpos 前带 free joint、执行器乱序）按 binding 地址读写 | ✅ |
| 固定高度、球拍安装和简化机身 | 2.5b | base z+0.10 → TCP 增量 [0,0,0.10]（atol 1e-9）、绕 Z 90° 刚体变换、圆盘惯量 Izz/Ixx 公式（atol 1e-12）、初态无非相邻穿透 + 人为重叠负例 | ✅ |
| FK、关节方向、Jacobian | 2.6 | 独立整机 URDF FK 对照（零/init/seed42×8 姿态双侧，位置旋转残差 ≤1.32e-11，容差 1e-8）+ 中心差分雅可比（≤3.4e-10）+ 评审 scipy 第三实现路径复核 | ✅ |
| 未知参数不冒充能力 | 2.2/2.5b/2.7 | null 解析/拒绝分支 + 预览 JSON 参数清单（provisional_fields/冲突四项逐字引用） | ✅ |
| 多机器人参数与 RM65 原值兼容 | 2.2/2.3 | 单位/优先级/旧工厂数值对照（sim J2=1.57 / real J2=1.40、torque_max=ctrlrange、TCP 1.8/1.0） | ✅ |
| 中文路径、资源缓存、独立实例 | 2.4/2.5b | 资源内容变化（相同 mtime/大小）缓存失效（D2+R1D 双目录）+ xml_text 参数化独立实例 | ✅ |
| 可复跑预览和参数清单 | 2.7 | JSON（17 键，r1d_inspect_report/v1）+ front/side/oblique 三视图 + 复现命令双侧 rc=0 | ✅（渲染仅 Windows 实测） |
| RM65 回归与 R1D 入口隔离 | 2.8/2.9/2.10 | 全量 846/2、seed42 逐位一致（2.9/2.10 修复后整文件 SHA256 均仍相同）、真实 r1d profile 入口隔离测试（退出 2 + 三组件零调用） | ✅ |

阶段 3 接收清单：`RobotProfile` / `ModelBinding` / `RobotRuntimeParameters`（含 `SceneParameters`、load/validate）/ `RobotModelBundle`（`load_robot_model`）/ `ModelKinematics` 的真实实现与测试，双侧 profile + runtime YAML ×4，inspect CLI。仍待阶段 3 决定：R1D 仿真速度/加速度/力矩/PD 参数、RobotEnv 适配、球与固定部件索引、准备姿态/工作空间策略、C++ 七维可用性、实际碰撞保护接法、轨迹生产入口配置选择。

## 索引更新（2.8 所有权）

- `src/README.md`：robot/ 模块表 + 详解节补 runtime_parameters.py / model_factory.py / model_kinematics.py / r1d_*.xml / d2_*.xml 行；关键入口补 `load_robot_model` / `ModelKinematics`
- `scripts/README.md`：tools/ 行 14→19（阶段 2 新增 3：build_r1d_model / build_d2_model / inspect_robot_model；既有 2 个 2026-07/09 加入时未计数）+ 新增 tools/ 小节
- `docs/README.md`：**无报告索引节**（仅指向 experiments/README.md），按任务条件不添加条目；experiments/README.md 亦无逐报告清单（阶段 1 报告同样未列）
- `AGENTS.md`：src/robot/ 结构补 6 行（__init__/runtime_parameters/model_factory/model_kinematics/r1d XML/d2 XML）+ configs/robots/ 补 r1d×4 + d2_runtime + tests/ 总数 702/69 → 848 collect/78 文件 + 测试清单补 9 行

## 2.9 独立复审修复（2026-11-16，提交 10fc5f2）

按 `requesting-code-review` 流程对 `d6526d9..f8e74de` 做了**第二条独立评审路径**（不读实施会话历史，评审者自述覆盖 30 个第一方文件 + 38 条 manifest 哈希重算 + URDF 逐关节抽查）。结论为 **With fixes**：模型链本身可信（哈希、1e-11 几何、RM65 整文件 SHA256 零影响、入口隔离均可独立复现），但参数层/加载器有三处"承诺与行为不一致"。三项均已修复并加测试：

| # | 问题（原状） | 影响 | 修复 |
|---|---|---|---|
| I1 | `_populate_joint_limits` 无条件用 `model.jnt_range` 覆盖入参 q_lower/q_upper，而 docstring 写"允许为 None" | 调用方在 YAML 收紧的安全限位先通过 `validate_runtime_parameters`，再被静默放宽回 URDF 全范围；落在阶段 3 参数迁移路径上 | 模型为位置限位唯一权威：显式值与 jnt_range 不一致即 `ValueError`（报关节名），一致才接受；不接受收紧/放宽覆盖 |
| I2 | `held_q` 只校验、从不应用（计划 Task 2.6 卡 L459 明写"初始化保持关节为 held_q、保持速度为零"） | R1D 双侧 `held_joint_names=[]` 今日零影响；任何带保持关节的模型会拿到静默错误的 FK | `ModelKinematics` 构造时按 `binding.held_qpos/qvel_indices` 写入 held_q 并置速度零后再 `mj_forward`；`set_state` 仍只写主动关节 |
| I3 | ASCII 路径分支下 `xml_text` 的临时 XML 写在模板源目录（`src/robot/`） | 只读安装/只读容器挂载下 `load_robot_model` 直接 `PermissionError`；Linux/CI 每次调用写源码树 | `mkstemp` 失败降级到 ASCII 临时副本的对应目录编译（include/mesh 相对基准不变），递归深度由 `allow_temp_copy_fallback` 限死 |

验证：目标测试 46 passed；全量 **838 passed / 2 skipped**（834 + 4）；ruff 改动文件 All checks passed；**seed42 RM65 重跑后新旧 npz 整文件 SHA256 仍为 `7c04e4eb…e154`**，零影响保持。

未修但已记录的复审 Minor（8 项，见 `review-final-independent.md`）：mypy 新代码 12+2 处、RM65 PD 回退值三处重复、通用参数层文档仍以 D2 为主语、`provisional_fields` 含 null 限值、inspect 用 `model.nq` 而非 `kin.n_dof`、缺 R1D 的 `purpose="simulation"` 拒绝用例、`.gitattributes` 未用通配、以及本报告原有的 C++ 排除口径（已在 §测试结果更正）。

## 2.10 独立复审 P2 修复与提交合并（2026-11-16）

对 `d6526d9..64af9bb` 的第三条独立复审（用户侧子代理）结论为**需修复后通过**：3 项 P2，无 P0/P1。三项均先在本仓库复现确认再修复：

| # | 问题 | 调查结论 | 修复 |
|---|---|---|---|
| P2-1 | `inspect_robot_model` 的 SDK 对照按列表位置与 `profile.active_joint_names` 配对 | **属实且更糟**：合法重排（profile/runtime 同序交换肩俯仰与肩侧摆）后，`per_joint` 与 `selected_side_joints` 双双错标，后者行内 `joint_name=…pitch…` 配 `sdk_name=L_SHOULDER_ROLL` 自相矛盾 | 新增 `_sdk_entry_for_joint`：按完整关节名（去侧别前缀/`_joint` 后缀的语义干 ↔ `sdk_name` 去 `L_`/`R_` 前缀）查询，两处报告字段共用映射；未知名 `ValueError`（退出码 2 配置错误类）。规范顺序输出逐字节不变 |
| P2-2 | `write_r1d_model_template` 跨盘符导出时 mesh 退回仓库绝对路径 | **属实**：导出到 C 盘 ASCII 目录后模板引用 `F:/刀客塔/…` 中文绝对路径，MuJoCo C 层 `Error opening file`，导出物不可加载 | `_retarget_mesh_paths` 相对化失败分支改为**重定位依赖资源**：mesh 复制到导出目录 `meshes/` 子目录并改写为相对引用（copyfile 不保留源只读属性）；同盘符回引行为不变（仓库模板逐字节不受影响） |
| P2-3 | 只读目录降级的副本可能仍不可写 | **机制属实**（安装版 stdlib `_copytree` 源码确认：对每个复制目录执行 `copystat`，Unix 0555 权限位原样带入副本）；本 Windows 开发机不可达，**Linux 实跑仍未做** | `_setup_temp_copy` 复制后 `_ensure_dirs_writable`：仅对副本目录 OR 属主写位（不动源目录、不改文件权限），同时修复 rmtree 清理只读副本目录的残留问题；常规可写源目录为 no-op |

验证：三项修复各自先行复现（P2-1/P2-2 复现脚本转红→修复后转绿）；新增针对性回归 8 项（SDK 按名跟随双侧 + 未知名拒绝、mesh 重定位双侧 + 真跨盘端到端双侧 + 字节确定性测试断言更新、副本目录写权限）；全量 **846 passed / 2 skipped**（838 + 8）；ruff 改动文件 All checks passed；**seed42 RM65 重跑整文件 SHA256 仍为 `7c04e4eb…e154`**（P2-3 落在 RM65 加载路径上，逐位零影响复核）。

**提交合并**：应用户要求，阶段 2 全部提交（2.2 起 13 个 + 本次修复）合并为单一成品提交；提交信息描述最终交付形态，不含过程性勘误记录。原始逐任务提交链留存于 SDD ledger。

## 结论

阶段 2 十项任务全部验收通过：R1-A7-D 官方资产可复建、双侧场景模型可加载绑定、FK/雅可比有独立整机 URDF 数值证据、预览可复跑；通用参数结构与加载器就位；RM65 默认行为与阶段 1 逐位一致（seed42 数组级 + 入口冒烟）；运行入口仍明确限制 RM65，临时几何/未知动力约束未被包装成真机能力。**阶段 2 已通过**（模型与运动学层）；七关节 MPC、RobotEnv 适配、SDK 真机均未验收（阶段 3+ 范围）。有证据可开始阶段 3。
