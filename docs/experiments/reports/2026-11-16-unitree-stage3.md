# 阶段 3 验收报告：R1-A7-D 单臂七关节 MPC 仿真闭环

- **日期**：2026-11-16
- **状态**：**阶段 3 已通过（2026-11-16）**——GF①–④ 由主代理在最终提交 `1711cbd` 复跑通过，GF⑤ 由独立终审 subagent 对 `754ccc0..1711cbd` 全分支包裁决 Approved（0 Critical / 0 Important / 6 Minor，均非阻断）
- **分支 / HEAD**：`unitree/stage3-sim` @ `1711cbd`（12 commits，`754ccc0..1711cbd`）；正文写于 `26fcab3` 工作树，证据与未提交改动由主代理按文件清单提交为 `1711cbd`（8 files +992/−24）
- **spec**：`2026-11-14-unitree-requirements.md`（型号更正版）与 `2026-11-14-unitree-adaptation-plan.md`（阶段 3）；施工计划 `2026-11-16-unitree-stage3-construction.md`
- **上游交接**：`docs/experiments/reports/2026-11-16-unitree-stage2.md`（模型与运动学层）

---

## 一、交付内容（任务 → 提交）

| 任务 | 提交 | 摘要 |
|------|------|------|
| 3.0 | `592eef4` | 阶段 3 交接基线报告（G0：846 passed/2 skipped、RM65 seed42 NPZ 逐位一致） |
| 3.1 | `6023281` | 仿真参数与任务配置严格预检（SimTaskConfig + 双侧 provisional YAML，49 tests） |
| 3.2 | `6b6a4f7` | 带球场景与碰撞集合绑定（TennisSceneBundle，27 tests） |
| 3.3 | `d35a960` | 按名称绑定的维度无关动力学环境（BoundRobotEnv，57 tests） |
| 3.4 | `3befef2` | 带球执行环境 / 规划副本 / 环境工厂（18 tests） |
| 3.5 | `75abf22` | 七维求解器 FD 与 C++ 三层隔离（45 tests） |
| 3.6 | `36e5de2` | 七维成本与 Tube 导数（53 tests） |
| 3.7 | `299d4c6` | 七维限位与禁止对预测碰撞保护（41 tests） |
| 3.8 | `29edc6e` | 七维策略与 MPC 消费链（32 tests） |
| 3.9 | `537991e` | 共享执行管线与物理接触判定（30 tests） |
| 3.10 | `26fcab3` | 多机器人仿真入口与可追踪产物（38 tests） |
| 3.11 | `1711cbd` | 固定可达案例 + 闭环验收测试 + 生产配置调参 + 索引更新 + 本报告（8 files +992/−24） |

本片工作树改动（`git status`，共 5 个已跟踪文件 + 3 个新增）：

| 文件 | 规模 | 说明 |
|------|------|------|
| `configs/scenarios/r1d_stage3_reachable.yaml` | 新增 2113 B | 固定可达案例（seed 42、expected_hit_type: active） |
| `tests/test_r1d_closed_loop.py` | 新增 20010 B / 4 tests | 固定案例闭环 + 右短回合 + RM65 逐位回归 |
| `configs/r1d_sim_left.yaml` | +35 / −11 | 3.11 定稿调参 4 项（含机理注释） |
| `configs/r1d_sim_right.yaml` | +22 / −7 | 右侧中心处置 + 同 4 项调参 |
| `src/README.md` | +17 / −2 | 阶段 3 模块/入口索引 |
| `scripts/README.md` | +20 / −1 | V12 R1D 入口说明 |
| `AGENTS.md` | +34 / −3 | 目录树/测试计数/运行命令/R1D 阶段状态 |
| `docs/experiments/reports/2026-11-16-unitree-stage3.md` | 本文件 | 阶段 3 验收报告 |

本片不提交大体积产物（NPZ/日志/调参脚本均在 `results/stage3_tuning/`、`results/unitree_stage3/`，`.gitignore` 已忽略）。阶段 3 合计 `754ccc0..1711cbd` 12 commits、55 文件、+14804/−357 行（含 3.11 的 8 文件 +992/−24）。

## 二、环境

- 操作系统 / 算力：Windows 本机，CPU（`--no-plot` 无渲染；MuJoCo 内置 viewer 未使用）
- 解释器：`C:/Users/Nothing/.conda/envs/mujoco_tennis/python.exe`（Python 3.11.15，`$env:PYTHONUTF8=1`）
- 依赖：mujoco 3.9.0、numpy 2.4.6、scipy、PyYAML、pytest（版本沿用阶段 2/3 既有环境）
- C++ 加速：`src/cpp/iLQR_Core.cp313-win_amd64.pyd` 对 Python 3.11 不可导入（ABI 为 cp313），本环境为 **Python-only**；R1D 默认即 Python + 有限差分线性化，未以修复 C++ ABI 为前提
- 模型加载：一律经 `src/utils/mujoco_loader.py::load_mujoco_model()`（Windows 中文路径安全）
- 全部命令自仓库根 `F:\刀客塔\研究\网球机器人\工程\Code\Sim\.worktrees\stage3-sim` 执行

## 三、测试结果

### 3.1 目标测试（本片新增，`tests/test_r1d_closed_loop.py`）

```
& "$PY" -m pytest tests/test_r1d_closed_loop.py -q
→ 4 passed in 48.98s（无 skip，RM65 回归用例实际执行）
```

4 个测试：`test_fixed_scenario_contract`、`test_left_fixed_case_active_hit_and_deterministic`、`test_right_production_short_episode`、`test_rm65_seed42_regression`。

RED 数值证据（调参前配置**事后重放**，test-after-tuning）：本片测试是在调参定稿之后建立的，RED 不是 test-first 结果，而是事后重放证据——调参定稿后由脚本 `results/stage3_tuning/make_pre_tuning.py` 从**调参后**的 `configs/r1d_sim_left.yaml` 反向还原出调参前配置副本 `results/stage3_tuning/tasks/pre_tuning_left.yaml`（其值经核对与 `git show HEAD:configs/r1d_sim_left.yaml` 一致），再重跑该副本得到：`hit=miss`、min_dist 0.1243029667 m、total_steps 121、`mpc_done=true`、`failure_reason=ball_unreachable`、first_plan 11306.4 ms、wall 14.28 s、X(122,14)/U(121,7)，与主代理 16:57 探针（min_dist 0.1243 / total 121 / first_plan 12403.5 ms）一致；该配置下闭环断言（`hit_type=active`）必然失败。时间顺序（据产物/文件创建时间）：调参扫描 17:21–17:24 → 生产 YAML 定稿 17:27:45 → 固定案例 scenario 17:29:15 → 闭环测试 17:29:59 → 测试通过日志 17:31:02 → RED 副本生成与重放 17:31:14 / 17:31:29；故本片明确为 **test-after-tuning**，不以“RED 先于实现”作为流程证据。调参后同一测试 4 passed。

### 3.2 全量回归

```
& "$PY" -m pytest tests/ -q --ignore=tests/test_cpp_backward_pass.py --ignore=tests/test_cpp_forward_pass.py
→ 1240 passed, 2 skipped, 4 warnings in 153.28s (0:02:33)
```

- 排除 `tests/test_cpp_backward_pass.py`：该文件收集即 `ModuleNotFoundError: No module named 'src.cpp.iLQR_Core'`（cp313 扩展对 py3.11 不可见，既有环境限制，非本阶段引入）。
- 排除 `tests/test_cpp_forward_pass.py`：可收集但 14 个用例全部 skip（同一 ABI 原因，既有）。
- 相对 3.9/3.10 基线 1236 passed / 2 skipped：**+4**（新增闭环测试，零回归）；2 skipped、4 warnings 均为既有（`test_actuator_modes.py` 等）。
- 测试文件计数：`tests/test_*.py` 从 754ccc0 基线 77 → HEAD 88 → 工作树 89（本片新增 1 个）；阶段 3 共新增 12 个测试文件。
- 日志：`results/stage3_tuning/pytest_full_stage3.log`（git-ignored）。

### 3.3 Ruff

范围 = `git diff --name-only 754ccc0..HEAD -- "*.py"` 的 44 个文件 + 新增 `tests/test_r1d_closed_loop.py`：

```
& "$PY" -m ruff check <上述 45 个文件>
→ exit 1，仅 6 条 E702，全部位于 src/ilqt/cost.py:133:31、133:57、134:31、134:56、135:32、135:56
```

定性：把基线版本取出（`git show 754ccc0:src/ilqt/cost.py`）后同样报 6 条 E702（同一组 `self._skew[...] = ...; ...` 多语句行，基线行号 127–129），且 `git diff 754ccc0..HEAD -- src/ilqt/cost.py` 的 hunk 不含这些行 ⇒ **阶段 3 之前既有的违规，非本片引入**。本片写入边界不含 `src/ilqt/cost.py`，未修改；作为既有违规上报主代理裁定（另立小片修复或作为例外记录）。其余 43 个本阶段改动 Python 文件 + 新增测试文件 ruff 干净。

## 四、固定案例与闭环证据（左侧）

### 4.1 固定案例文件

`configs/scenarios/r1d_stage3_reachable.yaml`（2113 B，SHA256 `49409cdbff31713c2c807fff1c9fea5a44bfafd0fc9653ca0fec58a1b665c711`）：

```yaml
schema_version: 1
profile_path: ../robots/r1d_left.yaml
runtime_path: ../robots/r1d_sim_runtime_left.yaml
task_path: ../r1d_sim_left.yaml
seed: 42
expected_hit_type: active
```

路径按相对 scenario 文件解析；未引入新配置框架。案例在落盘前完成 FK/IK 与固定重力反推验算：

- 来球初始位置 `[2.5, 0.45, 2.8] m`、初始速度 `[-5.0, -0.3, -1.0] m/s`，`t = 0.45 s`（k_hit = 90 @ dt 0.005）时球心 `[0.25, 0.315, 1.357] m`（z 含 0.5·9.81·t² = 0.993 m 下落，与 BallPredictor 逐位一致）。
- 左肩 `[0.059, 0.086, 1.358] m`，`‖p_hit − shoulder‖ ≈ 0.298 m` < workspace_radius 0.45（且 < 0.35 保守界），时间窗 0.45 s < total_horizon 200 步 = 1.0 s，满足有限步长/时间窗要求。
- `p_hit` 的 IK（自 step40 状态起）解 `q = [18.0, −2.24, 11.51, −13.81, −21.51, −0.72, 29.26]°`，FK 回代 `[0.2499, 0.3142, 1.3575] m`（≈0.9 mm）；7 关节均在限位内且远离硬限位（最紧为 wrist_pitch +29.26°/30°）。
- 单 seed 42 固定、未遍历 seeds 挑选；调参尝试/原因/最终配置哈希见 §五。

### 4.2 左侧生产配置连续 3 次重复（同 seed、同 YAML）

命令形态：`results/stage3_tuning/run_repeat.py --task configs/r1d_sim_left.yaml --out results/stage3_tuning/repeat_prod --runs 3`（该脚本仅循环调用生产 `run_r1d_episode`，不修改判定）。

| 次 | hit | step | face_speed (m/s) | active_contact | min_dist (m) | total_steps | mpc_done | failure | first_plan_ms | wall_s |
|----|-----|------|------------------|----------------|--------------|-------------|----------|---------|---------------|--------|
| 1 | active | 90 | 0.722929 | true | 0.0506811 | 111 | true | null | 6188.3 | 10.798 |
| 2 | active | 90 | 0.722929 | true | 0.0506811 | 111 | true | null | 6415.3 | 8.300 |
| 3 | active | 90 | 0.722929 | true | 0.0506811 | 111 | true | null | 6079.5 | 7.928 |

- `passive_contact=false`，`hit.time_s=0.45`，`hit.racket_face_speed=0.7229 m/s`（> 0.3 阈值）。
- 三次 X/U/ball_pos 内容哈希相同：`X_sha16=3c3c1d11a53fd7da`、`U_sha16=ffb8886921946ee3`、`ball_pos_sha16=1c27933e6babdb91`；`array_bitwise_identical=True`。
- 数组形状：X(112,14)、U(111,7)、ball_pos(112,3)、ball_vel(112,3)、t(112)，全部有限。
- 产物：`results/stage3_tuning/repeat_prod/run{1,2,3}/`（JSON + NPZ，git-ignored）。

生产 run1 JSON（`results/stage3_tuning/repeat_prod/run1/r1d_episode.json`）要点：

- schema `r1d_sim_episode_v1`、`simulation_only=true`、side `left`、mode `torque`、dt 0.005、seed 42；
- backend `python` / `python-fd`；`input_sha256`：profile `a0fbf50a…4098c3`、runtime `24739eac…36a341`、task `c2590acf…938f66d3`、model_template `6cdd2884…ef6fd40d`；
- episode：max_steps 600 / total 111 / safe 111 / mpc_done true / first_plan_ms 6188.3 / wall_time_s 7.673 / failure_reason null；
- hit：active、step 90、time 0.45 s、racket_face_speed 0.7229292674836804、`p_ee_at_hit=[0.27070959, 0.29393715, 1.30884771]`、`ball_pos_at_hit=[0.22535087, 0.31381812, 1.31961372]`、`ball_out_speed=5.4127105`、`ball_out_vy=2.2697967`，message「R1-A7-D 击打成功！（主动击球，pos_error=0.0507m）」；
- metrics：n_mpc_steps 111、min_dist 0.0506811048、max_qdot_ratio 0.5015261、max_tcp_speed = max_racket_face_speed = 1.1273201、active_contact true / passive false。

### 4.3 接触物理核验

- step 90 球—拍 `racket_face` 对产生真实 contact：`ncon=1`、`contact.dist = −0.0070 m`（穿入 7 mm，非 margin 幽灵接触）。
- 球心—拍面中心距离 0.0458 m < 拍半径 0.12 m ⇒ 拍面内接触；拍面局部坐标 `(−0.010, +0.032, +0.031) m`，径向 0.0445 m < 0.12、法向 |rz| 0.031 m < 拍半厚 0.005 + 球半径 0.033 m ⇒ 真实几何穿入。
- step 89（接触前一步）拍速 1.127 m/s（击球前加速），接触时刻 `get_racket_face_speed` 0.723 m/s（同一拍面速度定义，均记录于 metrics/JSON）。
- 球出射速度 `5.413 m/s / vy 2.270 m/s` 由生产 `compute_rebound_velocity(v_ball[90]=[-5,-0.3,-5.414], v_ee, n, e=0.8)` 计算得到（与 NPZ `ball_vel[91]=[-2.079,2.27,4.452]` 一致，差值为下一子步重力增量），**非测试伪造**；报告不宣称出球方向物理最优。
- 拍面法向 step88–90 `[0.248,0.225,0.942] → [0.266,0.241,0.933]`（拍面近水平、主要迎向 +z 来球方向）；几何诊断 min gap −0.0330 m @ step91（接触后）。

### 4.4 规划链证据（probe_trace）

插桩（monkeypatch 环境 step_full/碰撞开关、SimComponent.execute、HybridRefiner.refine、MPCController 结果应用，仅旁路记录、不改判定）重跑生产配置，记录到 `results/stage3_tuning/run_d/probe_trace_d.json`：

- replan 序列（`probe_trace` 仅从 step0 的**第二次** replan 起记录：首 plan 由 `MPCController.start()` 直接调用 `do_replan`，不经被 monkeypatch 的 `_apply_sync_result`，故 trace 中没有它；首 plan 的 `first_plan_iters=30`、horizon 90、`first_plan_ms≈6.1–6.4 s`（三次 6188.3/6415.3/6079.5 ms）见 §七）：step0 二次 replan（k_hit=90、iters=0、h=0、buffer=20、p_hit=[0.25,0.315,1.357]）与 step20（k=70）走 far/JT 分支（iters=0）；step40（k=50）→ near iLQR 5 迭代 h=50；step60（k=30）h=30；step80（k=10）h=10、buffer 10。
- refiner：k=90 feasible、70 feasible、50/30/10 locked（不换点）。
- 碰撞窗口：step79 起（k_hit≤10 且 dist<0.35 条件满足）`enable_collision=True`；step90 记录接触；step91 起球已反弹、碰撞对关闭（关闭态仍可能出现正距/幽灵 contact 列表，被 3.9 的 `ball_was_hit` + `dist<=0` 逻辑屏蔽，对 hit 无影响）。
- 与调参前诊断对照：调参前 k_hit 被放大到 380/360/…（见 §五），全程 far/JT、窗口从未开启，球以几何穿模 −0.0252 m 擦过拍面而 `hit=miss`。

### 4.5 禁止机身接触检查

`results/stage3_tuning/check_episode.py --task configs/r1d_sim_left.yaml --npz results/stage3_tuning/repeat_prod/run1/r1d_episode.npz`：

- X(112,14)/U(111,7)/ball_pos(112,3)/ball_vel(112,3) 全部 finite；
- 生产场景 `ForbiddenContactChecker.from_env(execution, scene.protected_geom_pairs)`：protected pairs = 8（7 连杆 + 球拍 对 torso_box）；
- 逐帧 `has_contact_at_q(q)`（独立 MjData，零污染）：**forbidden_frames = 0 / 112** → PASS。

闭环测试内同样以 `build_r1d_environments` + `ForbiddenContactChecker` 逐帧断言 0 禁止接触，并断言 pairs==8。

## 五、调参过程（诊断 → 定向扫描 → 定稿）

### 5.1 诊断（0.1243 m 差距来源）

1. **复现**：主代理生产探针（min_dist 0.124303 / total 121 / first_plan 12403.5 ms）与本片插桩重跑（min_dist 0.12430297 / total 121 / first_plan 13843.4 ms）逐位一致 ⇒ 稳定复现、非偶发。
2. **几何诊断**（`results/stage3_tuning/diag_geometry.py`）：拍面在 step76→102 几乎停在 `[≈0.316, 0.206, 1.333]`（每步仅 1–2 mm，ee_speed 0.53–0.86 m/s 振荡），球在 y≈0.315 从拍面 y≈0.206 旁 0.11 m 处掠过；拍“提前到位后停住”，未按 1.2 m/s 挥出。
3. **根因（决定性，时序索引）**：`src/ilqt/replan_core.py:199-201` 的节流 guard：
   `if k_hit_candidate < max(10, k_hit_new // 4) and k_hit_new > 30: k_hit_candidate = max(1, k_hit_new − config.replan_interval)`。
   首次 plan 由 `MPCController.start()` 用 `ReplanState(k_hit_new=config.total_horizon=400)` 初始化，真实 k_hit≈90 < max(10, 400//4=100) ⇒ 被改写为 380；此后每 20 步重规划的候选（≈90/50/30/10）恒小于当前 k_hit//4，k_hit 恒 ≈ 400−step ⇒ 规划击球时刻永远在 ~1.5 s 之后。后果：k_hit 全程 ≥260 > far_threshold 50 ⇒ 每次 replan 走 far/JT（iters=0、只慢速跟踪，永不进入 near iLQR/随挥）；`sim_component.py:198-209` 碰撞窗口（k_hit≤30 且 dist<0.35，或 k_hit≤10）全程不开启 ⇒ 球以几何穿模擦过拍面却无碰撞对 ⇒ miss。
   该节流 guard 属上游 3.8 代码、初始化语义属 mpc_controller 既有逻辑，均在本片写入边界之外；本片在 R1D 仿真任务 YAML 侧把 `total_horizon` 由 400 调为 200（`90 ≥ max(10, 200//4=50)` 成立，k_hit=90 正确），并在 YAML 注释记录机理。
4. **第二因素（规划末端误差）**：`probe_plan.py` 用 `env_plan.step_from_state` 对 `U_mpc_full` 前向 rollout，实测规划末端误差 0.14–0.19 m（首 plan err=0.1931 m）⇒ 规划器本身就打不到 p_hit，并非执行跟踪偏差；根因是终端权重 `cost_config.Q_p/Q_v = 2000/100` 相对 `normal_weight=2e5` 过小，法向项主导、位置被牺牲。将 Q_p/Q_v 提至 50000/200（对齐 RM65 `configs/default.yaml` L25-26 生产终端权重）后末端误差降至 ~5 cm。
5. **第三因素（安全滤波停回合）**：多次 miss 的第一原因并非“打不到”，而是击球后 `k_hit<=1` 时每步重规划 1 步 horizon 的激进控制触发 `safety_filter_stop`（如 t200 变体在 step92 因 joint6 |qdot|=40.95>37.70 停）。启用 `follow_through_steps=20`（生产已有 `PlannedFollowThrough`，`follow_trigger="planned"`、k_hit≤1 触发）后，step89 触发平滑 PD 随挥，step90 完成接触，随挥 20 步后 `mpc_done=true` 正常结束（111 步）。

### 5.2 定向扫描记录（同一 seed 42、左侧生产 runtime；变体文件 `results/stage3_tuning/tasks/*.yaml`）

| 变体 | 改动（相对调参前生产配置） | hit | min_dist (m) | steps | failure / 备注 | first_plan_ms | wall_s |
|------|---------------------------|-----|--------------|-------|-----------------|---------------|--------|
| pre_tuning（RED 副本） | 无（total 400 / ft 0 / Q_p 2000 / Q_v 100） | miss | 0.1243030 | 121 | ball_unreachable（k_hit 被放大到 380 起） | 11306.4 | 14.278 |
| t200 | total_horizon 200 | **active** | 0.1536 | 92 | safety_filter_stop（joint6 40.95>37.70 @92；击球后 1 步激进重规划） | 7267.5 | 8.56 |
| t200f100 | total 200 + far_threshold 100（起手 iLQR） | miss | 0.1658 | 91 | safety_filter_stop（joint6 43.42>37.70）；跟踪 y 更差（0.19）⇒ 否决 | 8348.3 | — |
| t200q50k | total 200 + Q_p/Q_v 50000/200 | miss | 0.0628 | 90 | safety_filter_stop（joint1 20.50>18.8 @90） | 6011.4 | 10.42 |
| t200q50kn50 | 上者 + normal_weight 5e4 | **active** | 0.0614 | 91 | safety_filter_stop（joint4 42.87>33.4 @91） | — | — |
| t200q100k | total 200 + Q_p/Q_v 100000/400 | miss | 0.0635 | 90 | safety_filter_stop（joint1 21.92>18.8） | 8355.0 | — |
| s08 | normal 5e4 + speed 0.8 + Q_v_near 40 + cost 50000/200 | miss | 0.0511 | 90 | safety_filter_stop（joint4 34.41>33.4 @90） | 7388.0 | 12.75 |
| **s08ft** | s08 + follow_through_steps 20 | **active** | 0.0621 | 111 | none / mpc_done（干净结束） | 8334.1 | 10.76 |
| s06 | speed 0.6 + Q_v_near 20 | miss | 0.0509 | 90 | safety_filter_stop（joint4 34.41>33.4） | — | — |
| long | total 240 + 更远来球 [3.0,0.45,3.38] | miss | 0.0438 | 110 | safety_filter_stop @110 | — | — |
| b | total 200 + normal 5e4 + ft 20 + cost 50000/200 | **active** | 0.06019 | 111 | none / mpc_done | 7947.0 | 13.94 |
| c | total 200 + speed 0.8 + Q_v_near 40 + ft 20 + cost 50000/200 | **active** | 0.052395 | 111 | none / mpc_done | 8244.4 | 10.67 |
| **d（定稿）** | total 200 + ft 20 + cost Q_p/Q_v 50000/200 | **active** | 0.0506811 | 111 | none / mpc_done | 7205.7 | 9.42 |

### 5.3 定稿（最小必要改动集，写入 `configs/r1d_sim_left.yaml`）

仅 4 项（其余生产值全部保留：normal_weight 2e5、racket_speed/target_speed 1.2、Q_v_scale_near 120、fixed_horizon 160、max_steps 600、workspace [0.25,0.25,1.35]/0.45、来球 [2.5,0.45,2.8]/[−5,−0.3,−1]）：

1. `mpc_config.total_horizon`: 400 → **200**（规避上游节流 guard 改写 k_hit 的时序 bug，见 §5.1-3）；
2. `mpc_config.follow_through_steps`: 0 → **20**（planned 触发；替代击球后 1 步 horizon 激进重规划，避免安全滤波停回合）；
3. `cost_config.Q_p`: 2000 → **50000**；
4. `cost_config.Q_v`: 100 → **200**（与 RM65 生产终端权重同量级）。

选择 d 而非 b/c 的理由：b/c/d 三者均 active、均 111 步/mpc_done；d 的改动最小（不牺牲已冻结的法向权重与挥拍速度生产偏好），且最终生产配置 3 次逐位一致（§4.2），作为固定案例的证据最直接。调参面未使用 `mpc_config` 白名单之外的键（M8 警告未触发），未改 3.10 的 `r1d_runner.py` 与 `_MPC_CONFIG_ALLOWED`。

## 六、右侧处置与短回合结果

- **右 task 中心调整**：`workspace_center` y 由 −0.25 改为 **−0.10**（`[0.25,−0.10,1.35]`）。原因：3.8 `validate_workspace_against_model` 在 −0.25 时的局部 IK 解 `q[1]=0.3289 rad > 右肩侧摆上界 +0.2268 rad`，3.10 已订正「并非物理绝对不可达，更换 IK 初值存在限位内分支（q1=−0.3779、误差 0.9 mm）」；3.8 校验属上游、本片无写入权限，故按任务书建议把中心移入右臂可达区，使右侧环境/求解器/短回合测试可真实运行。其余镜像来球/参数同左侧，并同样应用 §5.3 的 4 项调参（total 200 / ft 20 / Q_p 50000 / Q_v 200）。
- **右生产短回合**（真实 `run_r1d_episode`，产物 `results/stage3_tuning/run_right`）：`hit=miss`、total_steps 62、first_plan 6264.4 ms、wall 7.578 s、`failure_reason=safety_filter_stop`（step62 `[EMERGENCY_STOP] tcp speed 1.29 > 1.20 m/s`）、min_dist 0.9267 m、max_qdot_ratio 0.9468；refiner 日志 `[HIT_RISK] p=[0.275,−0.316,1.384] k=89 min_margin=−7.6°` → `[HIT_SWAP] k=89→86 到 [0.35,−0.321,1.463] 裕度 −3.5°`；plans k=89/69/49/29。右臂在镜像来球下的击球点几何裕度为负（右肩侧摆上界 +12.995°），refiner 换点后仍为负 —— 本片**不宣称右侧命中**，仅确认右侧链路（环境/求解器/安全/产物）真实可跑。
- 闭环测试对右侧的断言仅为：side=right、backend python-fd、total_steps>0、n_mpc_steps>0、hit.type ∈ {active, passive, miss}、failure_reason ∈ {None, safety_filter_stop, ball_unreachable}、数组有限/形状正确、禁止机身接触 0（均通过）。

## 七、规划耗时与控制步数（离线，非实时）

| 口径 | first_plan_ms | wall_s（外部计时，run_repeat） | 说明 |
|------|---------------|-------------|------|
| 生产配置（3 次） | 6188.3 / 6415.3 / 6079.5 | 10.798 / 8.300 / 7.928 | 首 plan = iLQR 30 迭代 @ horizon 90；同 3 次 JSON 内部 `episode.wall_time_s` = 7.673 / 7.641 / 7.301（仅回合内执行计时） |
| 调参前（R1D 基线） | 12403.5（主代理）/ 13843.4（插桩）/ 11306.4（RED 副本） | 12.56–14.28（外部计时） | 首 plan horizon 380、30 迭代 |

- 控制步数：total_steps = safe_steps = n_mpc_steps = **111**（仿真时间 0.555 s @ dt 0.005），击球 step 90（t=0.45 s）；平均每控制步 ≈66–97 ms 墙钟（JSON 内部 `wall_time_s` 口径 ≈66–69 ms，外部 run_repeat 口径 ≈71–97 ms），远大于 5 ms 步长。
- 结论：本阶段为**离线仿真验收**，规划/执行墙钟显著大于仿真时间，报告不得称实时；R1D 默认 Python + 有限差分线性化（C++ 后端在此环境不可用）。

## 八、碰撞统计

- `active_contact=true`（hit_step=90 首次即主动接触）、`passive_contact=false`；contact 对数 `ncon=1`（ball–racket_face），`contact.dist=−0.0070 m`（真实穿入）。
- 击球帧球心—拍心距离 5.07 cm（`metrics.min_dist` 是**击球帧** `pos_error` = ‖p_ee_at_hit − ball_pos_at_hit‖ = 0.0506811 m；轨迹逐步采样的最小值是 4.58 cm，见 §4.3）。
- 禁止机身接触：0 / 112 帧（8 组 protected geom pairs，独立 MjData 逐帧检查，PASS）。
- 碰撞窗口策略：step79 起 `enable_collision=True`（k_hit≤10 且 dist<0.35）；击球成功后关闭球—拍 pair。MuJoCo 3.9 在 pair 关闭态仍可能产生正距“幽灵接触”记录（3.9 已记录），本片复核其不影响 hit 判定，如实标注。

## 九、RM65 seed42 逐字段回归

命令（官方默认机器人/权重/模型/时间步，未改动任何默认值）：

```
& "$PY" scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --seed 42 --no-plot \
    --dump-trajectory results/unitree_stage3/rm65_seed42_stage3.npz
→ exit 0；__RESULT__: hit_type=active pos_error=0.022551 vel_error=1.980156
  hit_time_error=15.0ms first_plan_ms=3282.7 ball_out_speed=5.665 v_racket_at_hit=1.001
  max_tcp=1.27 max_qdot=0.94x max_face=1.4
```

与 3.0/阶段 2 基线 `results/unitree_stage3/rm65_seed42.npz`（SHA256 `7c04e4eb…b0e154`）对照（`results/stage3_tuning/compare_rm65.py`，日志 `results/stage3_tuning/rm65_compare_stage3.log`）：

| key | shape | dtype | array_equal | max\|Δ\| |
|-----|-------|-------|-------------|----------|
| ball_pos | (320,3) | float64 | True | 0.000e+00 |
| dt | () | float64 | True | 0.000e+00 |
| hit_step | () | int64 | True（236=236） | 0.000e+00 |
| init_q | (6,) | float64 | True | 0.000e+00 |
| init_q_left | (6,) | float64 | True | 0.000e+00 |
| metadata | () | <U279 | True | 0.000e+00 |
| q_actual | (320,6) | float64 | True | 0.000e+00 |
| q_desired | (0,6) | float64 | True | 0.000e+00 |
| tcp_pos | (320,3) | float64 | True | 0.000e+00 |
| timestamps | (320,) | float64 | True | 0.000e+00 |
| u | (320,6) | float64 | True | 0.000e+00 |

- 键集合一致（11 键），逐键形状/dtype/数值 **逐位一致**（非仅 zip SHA 对比）。
- 新产物整文件 SHA256 = `7C04E4EB3ED041566D0926B639522CE2B05694873DE13EF7C4C25FB220B0E154`，与 3.0 基线 SHA256 相同。
- 闭环测试内以 tmp 产物运行同一对照（逐键 `array_equal`），本机未 skip、通过；基线缺失时该用例会 skip 并注明 `results/` 为 git-ignored。

## 十、临时参数与 provisional 边界

- 本片修改的可调数值全部位于**仿真专用文件**：`configs/r1d_sim_{left,right}.yaml`（来球/工作区/MPC/代价）与 `configs/robots/r1d_sim_runtime_{left,right}.yaml`（runtime）。阶段 2 的描述文件 `configs/robots/r1d_{left,right}.yaml` 与未知限值文件 `configs/robots/r1d_runtime_{left,right}.yaml` **未修改**。
- runtime 文件声明为 provisional 的字段（仿真临时取值，不代表硬件授权值）：dt、init_q、base_position/quaternion、racket_position/quaternion/radius/half_thickness/mass、torso_position/half_size、kp、kd、qdot_max、qddot_max、torque_max、max_tcp_speed。本片未改这些数值（仅沿用并确认其 provisional 标记）。
- 左 task 的 4 项调参均在 YAML 内以注释记录机理与来源；右 task 的中心调整注明「因 3.8 校验限制移入可达区，不代表硬件限值」。
- 报告中的“生产配置”指 R1D 仿真任务配置，阶段 4 真机不得直接沿用；本片不宣称任何硬件限值。

## 十一、执行期决策与偏离（Rulings）

1. **不改上游**：未修改 3.8 `replan_core.py` 节流 guard、未修改 mpc_controller 初始化语义、未修改 3.8 workspace 校验（均在本片写入边界之外）；以 R1D 仿真 YAML `total_horizon=200` 规避并在注释记录完整机理与风险。建议后续由对应切片以“首次 plan 的 k_hit 初始化”语义修复，从而移除该规避。
2. **终端权重提升**：cost Q_p/Q_v 提至 50000/200，对齐 RM65 生产 `configs/default.yaml` L25-26；属 R1D 仿真任务参数，不影响 RM65 默认行为。
3. **启用随挥**：`follow_through_steps=20` 复用生产已有 `PlannedFollowThrough` 机制（trigger planned），属配置层修复，未新增接口/逻辑。
4. **右侧中心处置**：按任务书建议移入右臂可达区，使右侧链路可跑；不宣称右命中、不改 3.8。
5. **未触碰**：3.10 的 `src/sim/r1d_runner.py`、`tests/test_multi_robot_sim_entrypoint.py` 与任何其它 src/tests/scripts 文件均未修改；未使用 `_MPC_CONFIG_ALLOWED` 之外的键；未新增第二套接口。
6. **ruff 既有违规**：`src/ilqt/cost.py` 6 条 E702 经基线版本复核为既有（非本片引入），因写入边界未修，上报主代理裁定（另立小片或例外记录）。
7. **纪律**：未执行 `git add/commit`（按 SDD I3 由主代理提交）；未 push、未连硬件、未安装 SDK、未开发视觉、未推进阶段 4、未派发子代理；未提交大体积结果。

## 十二、索引更新说明

- `src/README.md`（+17/−2）：`sim/` 概览行与小节补充阶段 3 的 7 个模块 + `hit_detection.py`；`ilqt/` 补 `profile_planning_env.py`；“关键入口”补 `load_sim_task_config` / `validate_simulation_inputs` / `build_r1d_environments` / `run_r1d_episode` / `ProfilePlanningEnv`。符号统计一栏保留阶段 2 末口径、本片未重算（统计口径未在本片定义，避免给出不一致数字）。
- `scripts/README.md`（+20/−1）：`rm65_mpc_v12.py` 行注明阶段 3 起兼 R1D 单臂仿真入口；新增“R1D 单臂仿真入口（阶段 3）”小节（入口命令、R1D 模式允许的 7 个 flag 与其余参数退出码 2、产物 JSON/NPZ schema 与形状、固定案例、离线耗时口径）。
- `AGENTS.md`（+34/−3）：目录树补 `src/sim` 7 个阶段 3 文件与 `profile_planning_env.py`、`configs` 的 5 个阶段 3 配置文件、`tests/` 的 12 个新测试文件与测试计数（1240 passed + 2 skipped、89 个测试文件）；运行命令补 R1D 入口与固定案例；新增“### R1-A7-D 单臂仿真（阶段 3）”小节（入口/配置分层/本机实测/验收状态，明确最终宣布权属主代理与独立终审）。
- `docs/README.md` 无报告索引节，本片未添加（与阶段 2 处理一致）。

## 十三、边界与限制（如实）

- **单侧命中**：仅左侧固定案例 seed 42 主动命中；右侧链路可跑但未命中。不得表述为“双侧命中”。
- **离线非实时**：规划/整回合墙钟显著大于仿真时间，未做实时性优化或验收。
- **单 seed 确定性案例**：不是批量命中率统计；批量命中率、论文图表、真机部署均不在本片范围。
- **数值异常**：scratch/FD 预测中出现偶发 MuJoCo QACC NaN/instability 警告（不阻断，未逐条根因）；球出射方向由生产反弹模型给出，不宣称方向物理最优。
- **平台**：仅 Windows 本机实跑；Linux 未实跑（跨平台仅沿用既有 `load_mujoco_model()` 与既有测试断言），报告不宣称 Linux 通过。
- **C++**：cp313 扩展对 py3.11 不可导入（既有），全量排除两个 C++ 测试文件；R1D 默认 Python-FD，性能数字均为 Python 路径。
- **ruff**：`src/ilqt/cost.py` 6 条 E702 为既有违规（边界外未修）。
- **未覆盖**：R1D 位置模式、async 重规划（3.8 显式拒绝）、更多来球多样性/批量统计、真机与视觉感知。

## 十四、完成矩阵（施工计划 Task 3.11 复选框 → 证据）

| # | 计划要求 | 证据 | 状态 |
|---|----------|------|------|
| ① | 先建可达来球场景（FK/IK + 固定重力反推、有限步长/时间窗、先保存再验收、不遍历 seed 只报成功、调参留尝试/原因/最终配置哈希） | scenario 2113 B SHA256 `4940…c711`；t=0.45s 击球点验算 + IK/FK 0.9 mm；单 seed 42；§5.2 扫描表 + §5.3 定稿；各配置文件 SHA256（见 §4.2/§一） | ✅ 本机 |
| ② | 新测试须调起真实预测—重规划—安全—执行链；闭环断言（active 真实接触、拍速>0.3、状态/控制有限形状正确、无禁止机身接触）；不手写挥拍/teleport 球 | `tests/test_r1d_closed_loop.py` 4 tests 4 passed；probe_trace 生产链路；contact dist −0.0070 m、face_speed 0.723 m/s；禁止接触 0/112 | ✅ |
| ③ | 至少一侧连跑 3 次成功、另一侧同等级环境/求解器/短回合测试；不夸大为双侧 | 3 次逐位一致 active（§4.2）；右短回合 62 步真实运行 miss（§六） | ✅（如实区分） |
| ④ | 固定环境重跑 RM65 seed42 并与 3.0 基线逐项对照 | 11 键 array_equal、整文件 SHA256 与基线一致、hit_step 236（§九） | ✅ |
| ⑤ | 全量测试 + Ruff，精确排除/跳过理由，不把 C++ ABI 既有失败计入新增回归，Linux 未跑注明 | 1240P/2S/4W 153.28 s；ruff 6 E702 既有（§3.3）；Linux 未跑（§十三） | ✅（附既有违规上报） |
| ⑥ | 报告规划耗时/控制步数/命中证据/碰撞/临时参数限制；离线非实时；批量命中率/论文/真机不属本片 | §四/§五/§七/§八/§十/§十三 | ✅ |
| ⑦ | 更新索引与阶段状态；独立终审后才可写“阶段 3 已通过”；提交文档与结果索引，不提交大体积结果 | §十二；GF⑤（`754ccc0..1711cbd` 全分支）Approved 后写入阶段状态（§十五）；大体积产物在 git-ignored `results/` | ✅（`1711cbd` 提交） |

## 十五、结论

在本机/本环境（Windows + Python 3.11.15 + MuJoCo 3.9.0 + CPU、Python-FD、离线仿真）下，阶段 3 的 R1-A7-D 单臂七关节闭环：

- 左侧固定可达案例 seed 42 以**真实**预测—MPC/iLQR/Tube 重规划—安全滤波—MuJoCo 执行—物理接触判定链路完成主动击球（`hit_type=active`、`active_contact=true`、接触拍速 0.723 m/s、击球帧球心—拍心距离 5.07 cm（轨迹采样最小 4.58 cm）、111 控制步、`mpc_done=true`、禁止机身接触 0 帧），同配置同 seed 连跑 3 次逐位一致；
- 全量回归 1240 passed / 2 skipped（相对 3.10 基线 +4、零回归；C++ 两文件为既有 ABI 限制而排除）；
- RM65 seed42 NPZ 与 3.0 基线 11 键逐位一致（整文件 SHA256 亦一致）；
- 右侧链路可真实运行但未命中（未宣称）。

综上，**本片 Task 3.11 在本机验收通过**。GF①–④ 由主代理在最终提交 `1711cbd` 上复跑通过（全量 1240P/2S、ruff 仅 6 条基线 E702、RM65 整文件 SHA 一致、闭环 4P），GF⑤ 由独立终审 subagent 对 `754ccc0..1711cbd` 全分支包（55 files +14804/−357）裁决 **Approved**（0 Critical / 0 Important / 6 Minor）→ **阶段 3 已通过（2026-11-16）**。

---

## 附录 A：本片提交清单（已提交 `1711cbd`）

```
M AGENTS.md
M configs/r1d_sim_left.yaml
M configs/r1d_sim_right.yaml
M scripts/README.md
M src/README.md
?? configs/scenarios/r1d_stage3_reachable.yaml   (2113 B)
?? tests/test_r1d_closed_loop.py                 (20010 B)
?? docs/experiments/reports/2026-11-16-unitree-stage3.md   (本文件)
```

（提交后工作树干净，HEAD = `1711cbd`；上表为提交时的 8 文件清单，统计见 §一。）

## 附录 B：证据产物（git-ignored，不提交）

- `results/unitree_stage3/rm65_seed42.npz`（3.0 基线）、`results/unitree_stage3/rm65_seed42_stage3.npz`（本片重跑，SHA256 与基线一致）
- `results/stage3_tuning/`：`pytest_closed_loop.log`、`pytest_full_stage3.log`、`rm65_seed42_stage3.log`、`rm65_compare_stage3.log`、`compare_rm65.py`、`check_episode.py`、`diag_geometry.py`、`probe_trace.py`、`probe_plan.py`、`gen_variants.py`、`run_sweep.py`、`run_repeat.py`、`make_pre_tuning.py`、`tasks/*.yaml`、`repeat_prod/run{1,2,3}/`、`repeat_pre/run1/`、`run_d/`、`run_s08ft/`、`run_t200/`、`run_right/`、`sweep_out/` 等

## 附录 C：复现命令

```powershell
$env:PYTHONUTF8=1
$PY = "C:/Users/Nothing/.conda/envs/mujoco_tennis/python.exe"

# 1) 目标测试（4 passed）
& "$PY" -m pytest tests/test_r1d_closed_loop.py -q

# 2) 固定案例单跑（CLI 入口）
& "$PY" scripts/rm65_mpc_v12.py `
    --robot-config configs/robots/r1d_left.yaml `
    --runtime-config configs/robots/r1d_sim_runtime_left.yaml `
    --sim-task-config configs/r1d_sim_left.yaml `
    --output results/r1d_episode --seed 42 --no-plot

# 3) 全量回归
& "$PY" -m pytest tests/ -q --ignore=tests/test_cpp_backward_pass.py --ignore=tests/test_cpp_forward_pass.py

# 4) RM65 seed42 回归 + 逐键对照
& "$PY" scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --seed 42 --no-plot `
    --dump-trajectory results/unitree_stage3/rm65_seed42_stage3.npz
& "$PY" results/stage3_tuning/compare_rm65.py results/unitree_stage3/rm65_seed42.npz results/unitree_stage3/rm65_seed42_stage3.npz
```
