# 阶段 4 交接基线报告（Task 4.0）：R1-A7-D 整机接口下的单臂驱动

- **日期**：2026-11-16
- **任务**：施工计划 `docs/superpowers/plans/2026-11-16-unitree-stage4-construction.md` Task 4.0（基线与阶段 3 修复交接）
- **角色**：subagent-driven-development 实现者（本报告为 Task 4.0 唯一交付物）
- **范围**：只读核对 + 六文件基线测试，不改任何代码/配置；不连硬件、不装 SDK、不执行 git 写操作

---

## 一、可复跑版本（HEAD / 工作树 / 环境）

| 项 | 值 | 采集命令 |
|----|----|---------|
| HEAD | `0be5be752172c44f45ac8923f0384d8782a60695`（短 `0be5be7`，提交标题 `feat(unitree): 阶段 3——R1D 七维仿真管线与多机器人入口`） | `git rev-parse HEAD` |
| 分支 | `unitree/stage4-real`（worktree `.worktrees/stage4-real`） | `git branch --show-current` |
| 工作树 | **干净**（`git status --short` 为空；此为执行时刻快照——其后并行的 Task 4.1 在 worktree 新增 2 个未跟踪交付文件，属后续切片产物，非本基线污染） | `git status --short` |
| 解释器 | `C:/Users/Nothing/.conda/envs/mujoco_tennis/python.exe`，Python 3.11.15（Anaconda，MSC v.1942 64 bit AMD64），`PYTHONUTF8=1` | `python -c "import sys; ..."` |
| MuJoCo | 3.9.0 | `mujoco.__version__` |
| NumPy | 2.4.6 | `numpy.__version__` |
| C++ 扩展 ABI | `src/cpp/iLQR_Core.cp313-win_amd64.pyd` 仅 cp313 ABI，对本环境 Python 3.11 **不可导入**（`ModuleNotFoundError: No module named 'src.cpp.iLQR_Core'`）⇒ 本环境 Python-only（与阶段 3 报告 §二一致，既有环境限制，非本阶段引入） | `ls src/cpp/*.pyd` + `python -c "import src.cpp.iLQR_Core"` |
| 命令形态 | Git Bash：`PYTHONUTF8=1 "C:/Users/Nothing/.conda/envs/mujoco_tennis/python.exe" -m pytest <files> -q`，全部自仓库根 `.worktrees/stage4-real` 执行 | — |

**未提交修复场景**：brief 撰写时（施工计划 §状态，2026-11-16 计划阶段）曾记载阶段 3 工作树存在"三个 P2 对应的未提交修改"；本任务执行时该场景**已不存在**——三项 P2 修复已随阶段 3 squash 单提交 `0be5be7` 进入分支，原修复分支已删除，不存在独立修复 SHA（主代理裁定，本报告照此归属）。工作树干净，**无任何未提交文件需要收入本任务提交**，brief 复选框①中"不把未提交修复自动收入本任务提交"自然满足（零未提交内容）。

**阶段 3 报告链接**：`docs/experiments/reports/2026-11-16-unitree-stage3.md`（本 worktree 内，361 行；验收状态：阶段 3 已通过，GF①–④ 主代理在 `1711cbd` 亲跑、GF⑤ 独立终审 Approved；squash 后等价提交即本基线 `0be5be7`）。

## 二、三项 P2 修复归属与测试证据

三项 P2 均含于阶段 3 squash 提交 `0be5be7`（无独立修复 SHA）。核对方式：`pytest --collect-only -q -k <名>` 确认存在 + 直接运行确认通过 + `grep -n` 取源码定义行号。**结论：三项修复全部在场且全部通过，本任务不重复实现（也无其他会话修复在进行中的遗留）。**

### P2-1：PD+FF 总力矩上限

- 语义：位置模式 + 前馈（FF）下，PD 项与前馈项合成的总力矩受上限约束（防止 PD+FF 叠加超限）。
- 修复归属：含于 `0be5be7`。
- 测试证据（`tests/test_bound_robot_env.py`，文件共 61 用例）：
  - `test_feedforward_total_torque_capped_live`（定义于 :697）——live 环境路径，参数化 `r1d_left` / `synthetic6`，2 用例；
  - `test_feedforward_total_torque_capped_scratch`（定义于 :730）——`step_from_state` scratch 路径，参数化 `r1d_left` / `synthetic6`，2 用例；
  - 实跑：`pytest tests/test_bound_robot_env.py::test_feedforward_total_torque_capped_live tests/test_bound_robot_env.py::test_feedforward_total_torque_capped_scratch -q` → **4 passed in 2.94s**。

### P2-2：R 退火有效基底（R override + decay）

- 语义：R 退火（r_decay）的衰减基底取生效 R（override 后的 effective R），无 override 时回退 config R——修复此前退火基底忽略 override 的错误。
- 修复归属：含于 `0be5be7`。
- 测试证据（`tests/test_multi_robot_mpc.py`，文件共 34 用例，类 `TestCostConfigPassthrough` 定义于 :880）：
  - `TestCostConfigPassthrough::test_r_decay_base_uses_effective_r`（定义于 :932），1 用例；
  - `TestCostConfigPassthrough::test_r_decay_base_falls_back_to_config_r`（定义于 :970），1 用例；
  - 实跑：`pytest tests/test_multi_robot_mpc.py::TestCostConfigPassthrough::test_r_decay_base_uses_effective_r tests/test_multi_robot_mpc.py::TestCostConfigPassthrough::test_r_decay_base_falls_back_to_config_r -q` → **2 passed in 3.44s**。

### P2-3：CLI 显式参数拒绝（R1D 路径互斥旧 flag）

- 语义：R1D 模式（`--robot-config` 指向 `r1d_*.yaml`）下，RM65 专属旧参数无论传非默认值还是显式默认值均拒绝（退出码 2），防止显式传参静默改变 R1D 行为。
- 修复归属：含于 `0be5be7`。
- 测试证据（`tests/test_multi_robot_sim_entrypoint.py`，文件共 43 用例，类 `TestV12R1dExclusiveFlagsRejected` 定义于 :246）：
  - `test_rm65_exclusive_flag_rejected`（定义于 :280），参数化 extra0–extra8，9 用例；
  - `test_rm65_exclusive_flag_default_value_rejected`（定义于 :297），参数化 extra0–extra4，5 用例（含 `--normal-weight`、两种 `--target-speed` 写法与前缀缩写）；
  - 实跑：`pytest <两个具名节点> -q` → **14 passed in 0.42s**。

## 三、六文件基线测试（实际结果，本任务独立重跑）

命令（自仓库根 `.worktrees/stage4-real`）：

```bash
PYTHONUTF8=1 "C:/Users/Nothing/.conda/envs/mujoco_tennis/python.exe" -m pytest \
    tests/test_bound_robot_env.py tests/test_multi_robot_mpc.py \
    tests/test_multi_robot_sim_entrypoint.py tests/test_robot_interface.py \
    tests/test_robot_factory.py tests/test_robot_profile_entrypoint.py -q
```

**结果：172 passed in 17.55s（0 failed / 0 skipped / 0 error）。**

分文件计数（`--collect-only -q`，与实跑合计一致）：

| 文件 | 用例数 | 覆盖内容 |
|------|-------|---------|
| `tests/test_bound_robot_env.py` | 61 | P2-1 所在文件；BoundRobotEnv 绑定/scratch 隔离（阶段 3） |
| `tests/test_multi_robot_mpc.py` | 34 | P2-2 所在文件；七维 RM65↔R1D MPC 回归 |
| `tests/test_multi_robot_sim_entrypoint.py` | 43 | P2-3 所在文件；V12 多机器人 CLI 分派/产物 schema |
| `tests/test_robot_interface.py` | 10 | Realman SDK Mock（RM65 真机接口，阶段 4 不改） |
| `tests/test_robot_factory.py` | 16 | 驱动工厂/运行能力校验（阶段 1） |
| `tests/test_robot_profile_entrypoint.py` | 8 | `run_real_robot --robot-config` 入口 + R1D 入口隔离 |
| **合计** | **172** | **172 passed in 17.55s** |

- 主代理冒烟参考值为 172 passed in 18.85s：**计数一致（172）**，耗时差异（17.55s vs 18.85s）属机器负载波动，不构成偏差。
- 六文件不含 C++ 专属测试文件，未触及 cp313 ABI 排除问题；本基线数字即阶段 4 后续任务回归对照的起点。

## 四、G0 门槛判定

门槛定义（施工计划 §门槛与完成状态）：G0 交接 = 阶段 3 最终 SHA、三项 P2 处置和回归证据。

| 证据项 | 状态 |
|--------|------|
| 阶段 3 最终 SHA | `0be5be7`（= 阶段 3 squash 单提交，worktree HEAD，干净） |
| 三项 P2 处置 | 全部含于 `0be5be7`，6 个具名测试 20 个参数化用例全部通过（§二） |
| 回归证据 | 六文件 172 passed in 17.55s（§三）；阶段 3 全量 1240 passed + 2 skipped 见阶段 3 报告 §3.2 |

**G0 判定：通过。** brief 复选框④的"如修复未结束，标明 G0 未过"分支不触发——修复已结束（含于 `0be5be7`）且回归证据在场；4.1 可继续，无"阶段 3 问题未关闭"的遗留限制。

## 五、已知问题表（自阶段 3 继承，阶段 4 交接须知）

| # | 问题 | 状态/归属 | 对阶段 4 的影响 |
|---|------|-----------|----------------|
| 1 | C++ 扩展 cp313 ABI 对 py3.11 不可导入（`src/cpp/iLQR_Core.cp313-win_amd64.pyd`） | 既有环境限制（阶段 3 报告 §二/§十三）；全量测试需 `--ignore=tests/test_cpp_backward_pass.py --ignore=tests/test_cpp_forward_pass.py` | 六文件基线不受影响；阶段 4 无 C++ 自研控制服务（计划红线），无需修复 |
| 2 | ruff 6 条 E702 于 `src/ilqt/cost.py:133-135`（`self._skew[...]=...; ...` 多语句行） | 阶段 3 之前既有违规（基线 `754ccc0` 同报，见阶段 3 报告 §3.3），主代理裁定留存 | 阶段 4 新写代码须保持 ruff 干净；不重复上报 |
| 3 | `src/ilqt/replan_core.py:199-201` 节流 guard + 首 plan `k_hit` 初始化语义导致 k_hit 被放大 | 上游 3.8 代码 bug；阶段 3 以 YAML `total_horizon=200` 规避（阶段 3 报告 §5.1-3、§十一-1），建议后续切片修复 | 属仿真规划链，阶段 4 只做单臂低速驱动验证（不接击球 MPC），不受此影响 |
| 4 | 右侧生产配置未命中（短回合 62 步 `safety_filter_stop`，min_dist 0.9267 m） | 阶段 3 如实记录，不宣称右侧命中（报告 §六） | 属仿真闭环，阶段 4 范围外 |
| 5 | 3.8 workspace 校验对右侧 y=−0.25 中心的 IK 分支限制 | 以 task 中心 y=−0.10 规避（阶段 3 报告 §六），3.8 校验未修 | 同上，阶段 4 范围外 |
| 6 | scratch/FD 预测中偶发 MuJoCo QACC NaN/instability 警告 | 阶段 3 报告 §十三（不阻断，未逐条根因） | 阶段 4 不使用 FD 规划链，无直接影响 |

## 六、边界与纪律声明

- 本任务只读核对了 HEAD/工作树/环境/三项 P2 测试，并实跑六文件基线；除本报告与汇报文件外未修改任何文件。
- 未执行任何 git 写操作（add/commit 由主代理负责）；未连接硬件、未安装 SDK、未派发子代理。
- 测试一次运行即为记录结果，无失败重试。
- 复现：按 §一命令形态 + §三命令逐字执行即可。
