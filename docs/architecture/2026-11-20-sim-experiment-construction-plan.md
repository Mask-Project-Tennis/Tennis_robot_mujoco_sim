# 仿真与实验子包拆分：施工方案

> **状态：已获用户确认，2026-11-20 23:17 CST 开始执行。**
> P-1 已开始；本文档随基线提交冻结。

**Goal：** 在保持 import、CLI、数值结果与数据语义不变的前提下，用 1 个集成 worktree + 最多 3 个并发任务 worktree，完成仿真与实验链路的子包拆分，并建立按需发现的 README 路由。

**Architecture：** 先冻结基线、契约与共享类型，再按文件独占进行三轮迁移；每轮从同一 gate SHA 创建任务树，独立审查后逐个 cherry-pick 到集成树，集成回归通过后才进入下一轮。实验实现留在 `scripts` 子包；Episode 采用阶段函数组合，不引入继承框架。

**Tech Stack：** Python 3.11+、MuJoCo 3.9、NumPy/SciPy、pytest 9、CodeGraph 1.6.3；不新增运行依赖。

**Spec：**
- 可维护性调查：`docs/architecture/2026-11-20-maintainability-survey.md`
- 原拆分计划：`docs/architecture/2026-11-20-sim-experiment-decomposition.md`
- 并行执行计划：`docs/architecture/2026-11-20-sim-experiment-subagent-plan.md`
- 本施工方案是上述计划的执行层补充；任务职责与行为边界以原拆分计划为准，调度与文件所有权以并行执行计划为准。

## 0. 执行摘要

| 项 | 结论 |
|---|---|
| 当前阶段 | 用户已确认，P-1/P0 执行中；尚未开始 P1 源码迁移 |
| 基线提交 | `main @ 04a502f`（平台双基线已提交）+ 本方案文档提交 |
| 首要前置 | 平台修复已由 `04a502f` 提交；剩余 `docs/README.md` 与四份架构计划未提交，必须在基线分支冻结 |
| 工作树拓扑 | W2：`integration` 1 个 + 每轮最多 3 个任务树；每轮同一 gate SHA |
| 组织 | 主代理（唯一集成者）+ 最多 3 个子代理；审查者占同一槽位预算 |
| 阶段 | P-1 基线冻结 → P0 基线/契约 → P1 公共类型 → R1(A1/B1/C1) → G1 → R2(A2/B2/C2) → G2 → R3(D1/E1) → G3 → F0 |
| 关键路径 | P0 → P1 → B1 → G1 → B2 → G2 → D1 → G3 → F0 |
| 预计工期 | 约 9.5 个施工日；三槽并行后墙钟约 6–7 个工作日（不含用户评审等待） |
| 验收底线 | 目标测试通过；全量失败/错误/跳过 nodeid 集合与 P0 基线相比无新增；数值/产物/schema 不变；旧 import 与 CLI 可用 |
| 当前不做 | Stage4、`src/cpp/`、iLQT 算法重构、视频脚本、真机驱动整体拆分、批量调参 |

## 1. 现状核验（2026-11-20 22:55 CST）

### 1.1 仓库与环境

| 项 | 实测值 | 对施工的影响 |
|---|---|---|
| 仓库根 | `/data/fxy/research/Tennis_robot_mujoco_sim` | 所有命令显式使用该绝对路径 |
| 分支 / HEAD | `main` / `04a502f`（`04a502f` 提交平台双基线；`52b3988` 为其父提交） | gate SHA 必须用完整 SHA，不用 `main` 名称 |
| remote | `git@github.com:Mask-Project-Tennis/Tennis_robot_mujoco_sim.git` | 本轮不自动 push |
| worktree | 仅主工作区 | 需新建集成树与任务树 |
| CodeGraph | `Not initialized` | P0 需 `codegraph init --local-only -y` |
| Python | 3.11.16（`/home/fxy/.local/miniforge3/envs/mujoco_tennis/bin/python`） | 必须显式激活 conda 环境 |
| MuJoCo / NumPy / pytest | 3.9.0 / 2.4.6 / 9.1.1 | 与基线一致 |
| C++ 扩展 | `src/cpp/iLQR_Core.cpython-311-x86_64-linux-gnu.so` 存在（gitignored） | 每个 worktree 必须显式复制，不能依赖自动带过去 |
| 测试规模 | 136 个 `tests/test_*.py`，58,897 行；`--collect-only` = **2351 tests** | 全量运行约 3.5–10 分钟，必须隔离运行 |
| 目标新包 | 全部 MISSING | P1 从零建立，不在旧文件里预埋空目录 |
| 旧文件规模 | `model_factory.py` 2433 行、`v12_config.py` 2158、`v12_episode.py` 2039、`episode_artifacts.py` 2804、`run_r1d_paper_smoke.py` 2868、`paired_stats.py` 2682、`paper_figs.py` 2091 | 拆分后单文件 >600 行或单函数 >150 行需复查职责并记录理由，不为达标造转发层 |

**环境激活命令（所有任务统一）：**

```bash
source /home/fxy/.local/miniforge3/etc/profile.d/conda.sh
conda activate mujoco_tennis
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1
```

未激活时 C++ 扩展会报：

```text
ImportError: libmujoco.so.3.9.0: cannot open shared object file
```

激活后必须自检：

```bash
python -c "import mujoco, numpy, pytest; print(mujoco.__version__, numpy.__version__, pytest.__version__)"
python -c "import src.cpp.iLQR_Core as m; print([x for x in dir(m) if not x.startswith('_')])"
```

预期 C++ 符号：`linearize_analytical_batch`、`forward_pass_single`、`forward_pass_linesearch`、`backward_pass`、`sim_step`、`check_step`。

### 1.2 未提交工作区（必须在 P-1 处理）

`git status --short` 当前（2026-11-20 23:17 CST）：

```text
 M docs/README.md
?? docs/architecture/
?? docs/experiments/reports/2026-11-14-unitree-baseline.md
?? docs/r1d_serve_acceptance.md
?? docs/r1d_situation_brief.md
```

`git diff --stat` = 1 file, +6。

**关键变化：** 平台化修复（6 个文件，含 `src/utils/platform_tag.py`）已由 `04a502f test(baseline): 记录 Windows/Linux 双平台 T0 基线，Linux 为主` 提交到 `main`；`main` 比 `origin/main` 领先 1。P-1 不再需要提交这些文件，也不需要在基线分支重复平台修复。

**必须在基线提交的只剩：**

- `docs/README.md`：增加可维护性/拆分计划链接；
- `docs/architecture/` 四份文档：可维护性调查、子包拆分计划、Subagent 并行执行计划、本施工方案。

**结论：** `04a502f` 已提供平台修复前提；P-1 只需把上述 docs 提交到 `sim-exp/baseline`，即可从同一 SHA 建任务树。若不提交，任务树看不到计划文档；平台测试不会再因修复缺失而假回归。

其余未跟踪文档（`docs/experiments/reports/2026-11-14-unitree-baseline.md`、`docs/r1d_serve_acceptance.md`、`docs/r1d_situation_brief.md`）不参与本次重构，**不纳入基线冻结**，保持原状。

### 1.3 基线测试证据

| 数据 | 结果 | 说明 |
|---|---|---|
| Windows/T0 历史基线 | 12 failed / 1785 passed / 3 errors / 574.13s | 旧环境、旧测试数量；记录在 `docs/r1d_v12_parallel_repair_plan.md` 与 `docs/r1d_v12_usage.md`，**不能直接当 Linux 验收数字** |
| Linux 平台修复前全量 | 18 failed / 2324 passed / 7 skipped / 3 errors / 212.55s | `results/r1d_v12_parity/baseline/platforms/linux-x86_64/full_suite_pytest_pre_platform_fix.log` |
| 本轮定向复跑 | 9 failed / 56 passed / 63.85s | 命令：`pytest tests/test_mujoco_loader.py tests/test_v12_shared_pipeline.py tests/test_r1d_v12_acceptance.py tests/test_multi_robot_mpc.py -q` |
| 平台敏感用例 | `test_v12_shared_pipeline.py`、`test_r1d_v12_acceptance.py` **全部通过** | 证明平台化配置与 `platforms/linux-x86_64` NPZ 生效 |
| 预计当前 Linux 全量 | 约 14 failed / 3 errors | 11 个 Linux 历史已知（6 mpc + 3 entrypoint + 2 closed_loop）+ 3 个 Linux loader；Windows 历史为 12F+3E；**必须由 P0 在隔离基线树实测冻结** |
| 全量旧日志 | `results/r1d_v12_parity/t9/full_final.log` 保存旧 12F+3E 的 nodeid | P0 以该文件与实测结果交叉核对 |

**定向 9F 的准确 nodeid：**

1. `tests/test_mujoco_loader.py::test_d2_mesh_and_nested_xml_invalidate_cache`
2. `tests/test_mujoco_loader.py::test_xml_override_keeps_source_unchanged`
3. `tests/test_mujoco_loader.py::test_r1d_mesh_change_invalidates_loader_cache`
4. `tests/test_multi_robot_mpc.py::TestSevenDimWarmStart::test_position_jt_converges`
5. `tests/test_multi_robot_mpc.py::TestWorkspaceValidation::test_valid_side_config_passes`
6. `tests/test_multi_robot_mpc.py::TestIKChecked::test_reachable_target_within_limits`
7. `tests/test_multi_robot_mpc.py::TestMPCControllerR1D::test_start_step_buffer_and_replan`
8. `tests/test_multi_robot_mpc.py::TestMPCControllerR1D::test_follow_through_seven_dim`
9. `tests/test_multi_robot_mpc.py::TestAsyncRejection::test_controller_rejects_async_on_bound_env`

**分类：**

| 类别 | 数量 | 原因 | 施工处理 |
|---|---|---|---|
| Linux loader 平台差异 | 3 | `tests/test_mujoco_loader.py:20-21` 明写“Linux 分支（`sys.platform != win32` 直接加载）本机无 Linux 环境，仅由代码路径审查保证”；测试却默认非 ASCII 路径会创建临时副本，在 Linux 走直接加载分支，故失败 | 列为 **Linux-only known failures**；不通过改测试、加 skip 或改 loader 来“修复”；重构不得改变这些 nodeid 的失败性质 |
| RM65 workspace IK 已知失败 | 6 | `src/ilqt/mpc_helpers.py:484` 抛 `ValueError: 工作空间中心/半径对模型校验失败（中心 [0.25, 0.25, 1.35]，半径 0.45 m）：工作空间中心IK 解超关节限位：关节索引 [5] ...——拒绝带坏初值继续` | 列为历史已知；本轮不修算法；若重构导致错误文本/类型变化，按回归处理 |
| 旧入口已知失败 | 3F+3E + 2F | `test_multi_robot_sim_entrypoint` 3F+3E；`test_r1d_closed_loop` 2F（第 3 个依赖 `results/unitree_stage3/rm65_seed42.npz` 的用例因基线缺失 skip；Windows 历史曾为 3F） | P0 逐 nodeid 冻结；F0 对照，不新增 |

**重要：** `results/unitree_stage3/rm65_seed42.npz` 当前不存在，`test_r1d_closed_loop` 中依赖它的用例会按设计 skip；不要为了“全绿”去伪造基线。P0 记录实际 skip 集合。

### 1.4 测试污染事件

在**主工作区**直接跑全量时，曾出现嵌套进程：

```text
timeout 320 /home/fxy/.local/miniforge3/envs/mujoco_tennis/bin/python -m pytest tests/ -q -p no:cacheprovider
```

它把 `results/r1d_v12_parity/baseline/platforms/linux-x86_64/full_suite_pytest.log` 从 358KB 截断为 377B。旧完整日志在 `full_suite_pytest_pre_platform_fix.log`，备份在 `/tmp/tennis_baseline_backup_20261120_225117/`。嵌套进程来源未查明（tests 中未搜到 `no:cacheprovider` 直写；仅 `docs/r1d_v12_parallel_repair_plan.md:251` 有该建议命令）。

**第二次污染（2026-11-20 22:57–22:58）：** 主工作区又出现 `timeout 900 ... pytest tests/ -q -p no:cacheprovider`，同一文件再次被覆盖为点号进度；该进程由 tmux 会话 `pytest_suite`（`/tmp/run_suite.sh`）启动。已 `tmux kill-session -t pytest_suite`，终止 PID 3981696/3981725/3982716；`pgrep -af 'pytest|timeout 900'` 无残留。两次事件说明可能有其他会话/后台任务会自行启动全量 pytest；**不得把当前 `full_suite_pytest.log` 当证据**，P0 必须在隔离树重新生成。

**硬规则：**

- 全量 pytest 只在**隔离 worktree** 内运行，禁止在主工作区运行；
- 运行前检查 `pgrep -af 'pytest|timeout 900'`，若有非本任务进程，先确认归属并停止；运行后检查残留；
- 基线证据目录 `results/r1d_v12_parity/baseline/` 在 P-1 先备份并设只读（`chmod -R a-w` 仅对备份副本；主工作区不设只读，避免影响用户）；
- 任何全量测试输出只写到该 worktree 的 `results/refactor_subpackages/<run-id>/`，不写主工作区 `results/`。

## 2. 开工前置 P-1：冻结基线

> **这是本方案唯一需要用户明确点头的 Git 动作。** 未经同意，不执行本节。

### 2.1 推荐方案：基线分支，不推进 main

```bash
cd /data/fxy/research/Tennis_robot_mujoco_sim

# 0) 先确认无残留 pytest / 后台任务
pgrep -af 'pytest|timeout 900' || true

# 1) 从当前 main（04a502f，已含平台双基线）建基线分支
git switch -c sim-exp/baseline

# 2) 只暂存 P-1 剩余的 docs，禁止 git add .
git add -- \
  docs/README.md \
  docs/architecture/2026-11-20-maintainability-survey.md \
  docs/architecture/2026-11-20-sim-experiment-decomposition.md \
  docs/architecture/2026-11-20-sim-experiment-subagent-plan.md \
  docs/architecture/2026-11-20-sim-experiment-construction-plan.md

git diff --cached --stat
git commit -m "docs(refactor): freeze sim/experiment decomposition plans"

BASELINE_SHA=$(git rev-parse HEAD)
echo "BASELINE_SHA=$BASELINE_SHA"

# 3) 打本地标签，便于恢复；不 push
git tag -a sim-exp-baseline-2026-11-20 -m "sim/experiment refactor baseline" "$BASELINE_SHA"

# 4) 留在基线分支；main 引用仍停在 04a502f，不被推进
git status --short
```

**预期结果：**

- `BASELINE_SHA` 是新的基线提交（父提交为 `04a502f`）；
- `main` 分支引用仍为 `04a502f`；
- 当前主工作区 checkout 为 `sim-exp/baseline`，docs 与计划文档可见；
- 平台修复在 `04a502f` 中已存在，`BASELINE_SHA^` 即 `04a502f`；
- 不 push、不合并、不改 remote。

**备选：** 若用户希望 `main` 也前进，则把上述 `git switch -c` 改为直接在 `main` 提交；本方案默认**不**这样做，因为重构尚未评审，不应污染主线。

### 2.2 基线提交后的校验

```bash
# 基线树内容校验
git show --stat --oneline "$BASELINE_SHA"

# 平台标签校验
git show "$BASELINE_SHA:src/utils/platform_tag.py" | head -5

# 确认 ignored 资源未被提交
git show --stat "$BASELINE_SHA" | grep -E 'results/|experiment_data/|paper/|\.so' && echo "UNEXPECTED" || echo "OK: no ignored resources committed"
```

**停止条件：** 若 `git show --stat` 出现 `results/`、`experiment_data/`、`paper/`、`src/cpp/*.so`，或提交中包含用户未同意的文件，立即停止，不进入 P0。

### 2.3 工作树根目录

推荐外部同级目录，避免仓库内扫描、CodeGraph 重复索引与 `pytest` 收集：

```bash
export REPO=/data/fxy/research/Tennis_robot_mujoco_sim
export WT_ROOT=/data/fxy/research/Tennis_robot_mujoco_sim_worktrees
mkdir -p "$WT_ROOT"
```

目录规划：

```text
/data/fxy/research/Tennis_robot_mujoco_sim_worktrees/
├── p0-baseline/                 # P0 一次性基线运行，完成后移除
├── integration/                 # 集成树，保留至交付
├── r1-a1/  r1-b1/  r1-c1/       # 第一轮
├── r2-a2/  r2-b2/  r2-c2/       # 第二轮
└── r3-d1/  r3-e1/               # 第三轮
```

**命名约定（不使用 `codex` 前缀）：**

- 分支统一用 `sim-exp/` 前缀：`sim-exp/baseline`、`sim-exp/integration`、`sim-exp/r1-a1`、`sim-exp/r2-a2`、`sim-exp/r3-d1` 等；
- worktree 目录用短名：`integration`、`p0-baseline`、`r1-a1`、`r2-b2` 等，与分支一一对应；基线提交由主工作区（checkout `sim-exp/baseline`）承载，不额外建 `baseline` 工作树；
- 任务树归档时删除目录、保留 `sim-exp/*` 分支作为审计线索；不使用 `codex/*` 命名。

**不使用** `$REPO/.worktrees/`：虽然 `.gitignore` 已忽略，但测试中存在 `SCRIPTS_DIR.rglob("*.py")`、`ASSETS_DIR.rglob("*")`、`(PROJECT / "paper").rglob("*")` 等扫描；外部目录可彻底避免把任务树源码扫进主仓库测试。

### 2.4 创建集成树与 P0 基线树

```bash
cd "$REPO"

# 集成树：从基线提交创建，命名分支
git worktree add -b sim-exp/integration "$WT_ROOT/integration" "$BASELINE_SHA"

# P0 一次性基线树：detached HEAD，避免多一条长期分支
git worktree add --detach "$WT_ROOT/p0-baseline" "$BASELINE_SHA"

git worktree list
```

**校验：**

```bash
for d in "$WT_ROOT/integration" "$WT_ROOT/p0-baseline"; do
  echo "=== $d ==="
  git -C "$d" rev-parse HEAD
  git -C "$d" status --short
done
```

### 2.5 忽略资源准备（每个树一次）

`git worktree` 不会带过去任何 gitignored 资源；C++ 扩展、平台 NPZ、实验数据、paper 统计样本必须显式复制。

**最小资源清单：**

| 资源 | 源路径 | 目标 | 用途 |
|---|---|---|---|
| C++ 扩展 | `src/cpp/iLQR_Core.cpython-311-x86_64-linux-gnu.so` | `<wt>/src/cpp/` | 所有求解器测试 |
| 平台基线 | `results/r1d_v12_parity/baseline/` | `<wt>/results/r1d_v12_parity/baseline/` | `test_v12_shared_pipeline` 逐位对照 |
| R1D 实验数据 | `experiment_data/r1d_v12_parity/`（6.5MB） | `<wt>/experiment_data/r1d_v12_parity/` | 统计、paper smoke、产物测试 |
| paper 统计样本 | `paper/planning/06-stats-active.json`（若存在） | `<wt>/paper/planning/` | `test_r1d_statistics` 的“不得改写仓库 paper”用例；缺失则测试按设计 skip |
| 旧全量日志 | `results/r1d_v12_parity/t9/full_final.log` | `<wt>/results/r1d_v12_parity/t9/` | P0 对照 12F+3E nodeid；不是测试输入 |
| 基线备份 | `/tmp/tennis_baseline_backup_20261120_225117/` | 不动 | 污染恢复用 |

**不要复制：**

- 整个 `results/`（185MB）——只复制 `baseline/` 与明确的只读证据；
- 整个 `paper/`（161MB，含嵌套 `.git`）——只复制需要的统计 JSON；
- `experiment_data/exp*` 历史实验——本次不消费；
- `build/`——不需要重编译 C++。

**复制脚本（主代理执行，任务树复用）：**

```bash
prepare_resources() {
  local wt="$1"
  mkdir -p "$wt/src/cpp" "$wt/results/r1d_v12_parity" "$wt/experiment_data" "$wt/paper/planning"
  cp -a "$REPO/src/cpp/iLQR_Core.cpython-311-x86_64-linux-gnu.so" "$wt/src/cpp/"
  cp -a "$REPO/results/r1d_v12_parity/baseline" "$wt/results/r1d_v12_parity/"
  cp -a "$REPO/experiment_data/r1d_v12_parity" "$wt/experiment_data/"
  if [ -f "$REPO/results/r1d_v12_parity/t9/full_final.log" ]; then
    mkdir -p "$wt/results/r1d_v12_parity/t9"
    cp -a "$REPO/results/r1d_v12_parity/t9/full_final.log" "$wt/results/r1d_v12_parity/t9/"
  fi
  if [ -f "$REPO/paper/planning/06-stats-active.json" ]; then
    cp -a "$REPO/paper/planning/06-stats-active.json" "$wt/paper/planning/"
  fi
}
```

**资源校验：**

```bash
check_resources() {
  local wt="$1"
  test -f "$wt/src/cpp/iLQR_Core.cpython-311-x86_64-linux-gnu.so" || { echo "MISSING .so"; return 1; }
  test -f "$wt/results/r1d_v12_parity/baseline/platforms/linux-x86_64/rm65_torque_a.npz" || { echo "MISSING baseline npz"; return 1; }
  test -f "$wt/experiment_data/r1d_v12_parity/baseline/manifest.json" || { echo "MISSING experiment baseline"; return 1; }
  echo "OK: $wt"
}
```

**资源指纹：** 对每个树记录：

```bash
(
  cd "$wt"
  find src/cpp results/r1d_v12_parity/baseline experiment_data/r1d_v12_parity paper/planning \
    -type f -print0 2>/dev/null | sort -z | xargs -0 sha256sum
) > "$WT_ROOT/resource-$(basename "$wt").sha256"
```

### 2.6 开工自检

```bash
cd "$WT_ROOT/integration"
source /home/fxy/.local/miniforge3/etc/profile.d/conda.sh
conda activate mujoco_tennis
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1

python -c "import sys; print(sys.executable)"
python -c "import mujoco; print('mujoco', mujoco.__version__)"
python -c "import src.cpp.iLQR_Core; print('cpp ok')"
python -c "import src.sim.v12_config as m; print('v12_config:', m.__file__)"
python -c "import src.sim.episode_artifacts as m; print('artifacts:', m.__file__)"

# 必须落在当前 worktree，不能指回主工作区
python - <<'PY'
from pathlib import Path
import src.sim.v12_config as a
import src.sim.episode_artifacts as b
for m in (a, b):
    p = Path(m.__file__).resolve()
    print(m.__name__, p)
    assert "Tennis_robot_mujoco_sim_worktrees" in str(p), "import 指回了旧树"
PY
```

**停止条件：** 若 `src.sim.v12_config.__file__` 指向主工作区而非当前 worktree，说明 conda 环境存在 editable install / `PYTHONPATH` 污染，必须先修环境，不能开工。

## 3. 组织、槽位与派发

### 3.1 角色

| 角色 | 数量 | 职责 | 不做什么 |
|---|---|---|---|
| 主代理（lead） | 1 | 基线/契约/ledger/共享文件/Git 集成/最终验收/对用户汇报 | 不把领域内部实现全部揽回；不在任务运行中改任务写集 |
| 实现子代理 | 每轮最多 3 | 按 brief 实现、自测、写报告、停止写入 | 不 commit / 不切分支 / 不 merge / 不 push / 不改共享文件 |
| 审查子代理 | 每轮最多 3（与实现者共享槽位） | 只读审查 diff、契约、测试证据，给 `approved` 或 `changes_requested` | 不编辑代码；不把旧审查结论套到新 SHA |

**槽位硬约束：** 同时活跃的子代理总数 ≤ 3（实现 + 审查一起算）。三个实现者都在写时，不得再起审查者；某个实现者停止写入后，才可用其空出的槽位起 reviewer。

### 3.2 每轮槽位时序

| 时间片 | 槽位 1 | 槽位 2 | 槽位 3 | 主代理 |
|---|---|---|---|---|
| R1 开始 | A1 实现 | B1 实现 | C1 实现 | 监控/ledger |
| A1 完成 | A1 reviewer | B1 实现 | C1 实现 | 核对写集 |
| B1 完成 | A1 reviewer | B1 reviewer | C1 实现 | 处理共享补丁 |
| C1 完成 | A1 reviewer | B1 reviewer | C1 reviewer | 准备 G1 |
| G1 集成 | 空闲 | 空闲 | 空闲 | cherry-pick + 全量回归 |

实际调度可按完成顺序调整，但不得突破 3 个槽位，也不得让 reviewer 与同任务实现者同时写同一文件。

### 3.3 模型选型

- 实现者：使用当前运行时可用、能处理多文件接口迁移的标准档模型；
- 最终架构审查：使用可用的高能力档模型；
- 不在文档硬编码可能失效的模型名；派发时按 `list_subagent_models` 的实际可用项选择；
- 子代理使用独立上下文（`subagent`，不是 `subagent_fork`），避免继承本次长对话；
- 子代理只读自己的 brief、指定 contracts 段和指定源码，不要求它重读整份计划。

### 3.4 派发模板

```text
你是本次子包迁移的 {task_id} 实现者，不是总协调者。
工作目录：{absolute_worktree_path}；只在这个目录读写。
先读 {absolute_brief_path}：它是你的完整任务要求。
再读 {absolute_contracts_path} 中 brief 指定的接口段。
基线：{gate_sha}。该 SHA 的前置任务已集成；不要 checkout 它。
允许写入：{explicit_owned_paths}；所有其他文件只读。
隔离规则：你只拥有这个 worktree；不访问其他任务的未集成源码。
禁止 Git 索引/提交/分支操作；跨域修改以建议写进报告。
测试：{exact_commands}。重测试先向主代理申请运行许可；
输出目录 {unique_test_output_path}。不要同时启动批量实验或渲染。
实现目标、接口、验收断言均以 brief 为准；不扩展功能。
报告写到 {absolute_report_path}；最终只回报状态、文件列表摘要、
测试结果、阻碍和报告路径。完成报告后停止写入，等待审查返修。
```

## 4. W2 工作树与 Git 集成

### 4.1 树、分支、gate

| 树 | 分支 | 起点 | 生命周期 |
|---|---|---|---|
| `p0-baseline` | detached | `BASELINE_SHA` | P0 完成后移除；日志转存协调目录 |
| `integration` | `sim-exp/integration` | `BASELINE_SHA` | 保留至交付；P1/G1/G2/G3/F0 在此推进 |
| `r1-a1` | `sim-exp/r1-a1` | `P1_SHA` | G1 集成后移除目录，保留分支 |
| `r1-b1` | `sim-exp/r1-b1` | `P1_SHA` | 同上 |
| `r1-c1` | `sim-exp/r1-c1` | `P1_SHA` | 同上 |
| `r2-a2` | `sim-exp/r2-a2` | `G1_SHA` | G2 后移除目录，保留分支 |
| `r2-b2` | `sim-exp/r2-b2` | `G1_SHA` | 同上 |
| `r2-c2` | `sim-exp/r2-c2` | `G1_SHA` | 同上 |
| `r3-d1` | `sim-exp/r3-d1` | `G2_SHA` | G3 后移除目录，保留分支 |
| `r3-e1` | `sim-exp/r3-e1` | `G2_SHA` | 同上 |

**创建命令模板：**

```bash
git -C "$REPO" worktree add -b "sim-exp/r1-a1" "$WT_ROOT/r1-a1" "$P1_SHA"
```

创建后立即：

```bash
git -C "$WT_ROOT/r1-a1" rev-parse HEAD        # 必须等于 gate SHA
git -C "$WT_ROOT/r1-a1" status --short        # 必须为空
prepare_resources "$WT_ROOT/r1-a1"
check_resources "$WT_ROOT/r1-a1"
```

### 4.2 提交与审查

**实现者停止写入后，主代理提交：**

```bash
cd "$WT_ROOT/r1-a1"
git add -- src/robot/model_assets src/robot/model_factory.py tests/test_model_asset_imports.py
git diff --cached --stat
git commit -m "refactor(robot): extract model asset resolution (A1)"
TASK_SHA=$(git rev-parse HEAD)
echo "A1_TASK_SHA=$TASK_SHA"
```

**审查：**

```bash
git -C "$WT_ROOT/r1-a1" diff "$P1_SHA..$TASK_SHA" --stat
git -C "$WT_ROOT/r1-a1" diff "$P1_SHA..$TASK_SHA"
```

审查者只读；返修由原 owner 追加提交，审查绑定到新 SHA。

**集成：**

```bash
cd "$WT_ROOT/integration"
git cherry-pick "$A1_TASK_SHA"   # 然后 B1、C1
```

### 4.3 冲突与回退

- cherry-pick 冲突 = 写集、上游契约或补丁上下文有遗漏；立即停止，记录冲突文件与 gate SHA；
- 不选择整文件 ours/theirs；不覆盖其他任务改动；
- 语义冲突的修复必须重新审查，不能沿用旧 `approved`；
- 集成失败只回退本轮已 cherry-pick 的提交：`git revert <sha>`，禁止 `reset --hard`；
- 任务实现失败但未集成：保留分支与报告，移除目录；不得删除用户文件；
- 不自动合并 `main`，不 push；最终合并方式由用户决定。

### 4.4 归档

```bash
# 确认该树没有活跃 pytest / 子代理 / 未提交改动
git -C "$WT_ROOT/r1-a1" status --short
pgrep -af "pytest.*r1-a1" || true

git -C "$REPO" worktree remove "$WT_ROOT/r1-a1"
# 保留分支 sim-exp/r1-a1 作为审计线索；不自动删除
```

归档前把报告、原始日志、资源指纹复制到：

```text
.superpowers/sdd/sim-experiment-decomposition/reports/
.superpowers/sdd/sim-experiment-decomposition/reviews/
```

## 5. 阶段与任务

### 5.1 P0：基线、契约、brief、ledger（主代理，串行）

**工作树：** `$WT_ROOT/p0-baseline`（detached `BASELINE_SHA`）
**目标：** 得到可解释的 Linux 基线、冻结接口、写集、资源指纹和每任务 brief。

**步骤：**

- [ ] 记录 `BASELINE_SHA`、Python/MuJoCo/pytest/NumPy 版本、C++ 扩展哈希、资源指纹。
- [ ] 初始化 CodeGraph（在 `integration` 树，而不是一次性 `p0-baseline` 树）：`cd "$WT_ROOT/integration" && codegraph init --local-only -y`；记录 `codegraph status`。
- [ ] 用 `rg` 补查：字符串动态导入、`__file__`、`sys.path` 修改、全局可变配置、pickle/类型路径、`monkeypatch`/`patch(`、脚本间导入、包级重型导入。
- [ ] 建立 ledger：`.superpowers/sdd/sim-experiment-decomposition/{progress.md,contracts.md,briefs/,reports/,reviews/}`；若已存在，先读身份/状态，不重置。
- [ ] `contracts.md` 逐项记录旧路径/新路径/签名/类型字段/公有常量/错误语义/调用者/已知补丁点；稳定函数清单必须逐项覆盖；长签名用 `inspect.signature` 抄录。
- [ ] 冻结字段级基准：schema、run/config 身份、False/0/None、seed/attempt、meshdir、资源哈希、错误分类；比较排除墙钟时间、时间戳、Git 提交号。
- [ ] 在 `p0-baseline` 跑全量：

```bash
cd "$WT_ROOT/p0-baseline"
source /home/fxy/.local/miniforge3/etc/profile.d/conda.sh
conda activate mujoco_tennis
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1

mkdir -p results/refactor_subpackages/p0
python -m pytest tests/ -q -p no:cacheprovider \
  > results/refactor_subpackages/p0/full_suite.log 2>&1
echo "EXIT:$?"
```

- [ ] 从日志提取 `FAILED`、`ERROR`、`SKIPPED` nodeid，生成 `p0-known-failures.txt`、`p0-skips.txt`；与 `results/r1d_v12_parity/t9/full_final.log` 逐 ID 对照。
- [ ] 对每个新增/变化失败判定：环境差异、历史已知、真实回归；真实回归在 P0 关闭后才能进入 P1。
- [ ] 生成每任务 brief（路径 `briefs/A1.md` 等），内联：全局约束、完整步骤、精确写集、测试命令、contracts 签名、上游裁决、报告格式。
- [ ] 记录测试资源策略与重测试许可规则。

**P0 退出条件：**

1. 全量结果有日志、有 nodeid 清单、有分类；
2. 稳定接口签名与类型身份已冻结；
3. 每个任务 brief 可独立执行；
4. 无“待补的跨任务接口”；
5. 用户已确认 Linux-only 3 个 loader 失败作为已知基线。

**P0 产物（协调目录）：**

```text
.superpowers/sdd/sim-experiment-decomposition/
├── progress.md
├── contracts.md
├── p0-known-failures.txt
├── p0-skips.txt
├── p0-full-suite.log
├── briefs/{P1,A1,B1,C1,A2,B2,C2,D1,E1,F0}.md
├── reports/
└── reviews/
```

### 5.2 P1：共享类型前置（串行）

**起点：** `BASELINE_SHA`（P0 只读，不产生代码提交）
**写集：**

```text
src/robot/model_building/{__init__.py,types.py,README.md}
src/sim/configuration/{__init__.py,types.py,README.md}
src/sim/episode/{__init__.py,types.py,README.md}
src/robot/model_factory.py
src/sim/v12_config.py
src/sim/v12_episode.py
tests/test_refactor_shared_types.py
```

**要求：**

- [ ] 迁移 `RacketAssembly`、`RobotModelBundle`、`ResolvedV12Config`、`V12EpisodeResult` 的唯一类定义；纯辅助函数随类型迁移；
- [ ] 旧路径显式重导出同一对象；字段、默认值、冻结语义、实例方法保持一致；
- [ ] 检查 `__module__`/pickle/`isinstance` 真实使用；有持久化依赖时写兼容处理，不假定无人使用；
- [ ] `__init__.py` 只暴露轻量内容，不 import 尚未迁移的实现模块；
- [ ] 三份 README 明示“当前只完成类型迁移”；
- [ ] 跑：

```bash
python -m pytest \
  tests/test_refactor_shared_types.py \
  tests/test_v12_config.py \
  tests/test_v12_robot_adapter.py \
  tests/test_r1d_model.py \
  tests/test_robot_package_init.py -q
```

- [ ] 额外断言新旧类型 `is` 相同；
- [ ] 独立审查，主代理提交；提交 SHA 记为 `P1_SHA`；
- [ ] 将 `P1_SHA` 写入第一轮所有 brief。

**P1 退出条件：** 目标测试通过；旧/新类型身份一致；无反向 import；`P1_SHA` 成为 R1 的唯一 gate。

### 5.3 第一轮 R1（A1 / B1 / C1）

**gate：** `P1_SHA`。三树同起点，写集不相交。

#### A1：模型资源解析

| 项 | 内容 |
|---|---|
| 写集 | `src/robot/model_assets/` 全包；`src/robot/model_factory.py`；`tests/test_model_asset_imports.py` |
| 输入 | P1 类型、原参数/profile、contracts 中的路径解析规则 |
| 输出 | `model_assets.dependencies.collect_model_file_dependencies`，签名与旧函数完全相同；旧工厂重导出；`collect_model_dependencies` 暂留原工厂，交 A2 |
| 必测 | `pytest tests/test_episode_asset_meshdir.py tests/test_episode_asset_closure.py tests/test_r1d_model.py tests/test_model_asset_imports.py -q` |
| 验收 | 纯解析不加载 MuJoCo；中文路径、`meshdir`、注释忽略行为不变；无产物模块修改 |
| 报告 | 旧私有 helper 的外部使用、待主代理修正的调用、README 路由建议 |

#### B1：配置解析

| 项 | 内容 |
|---|---|
| 写集 | `src/sim/configuration/`（含 P1 types，字段冻结）；`src/sim/v12_config.py`；`tests/test_configuration_migration.py` |
| 输入 | P1 `ResolvedV12Config`、现有配置/profile；不读 A1 实现 |
| 输出 | `configuration.resolution.resolve_v12_config`，原参数/返回类型；旧入口/公有常量显式重导出 |
| 必测 | `pytest tests/test_v12_config.py tests/test_configuration_migration.py -q` |
| 验收 | False/0/None、显式 CLI 与保存覆盖、默认值优先级不变；validator 注册不重复；参数名/错误口径不变 |
| 报告 | monkeypatch/动态导入清单、README 导航建议 |

#### C1：论文 smoke 编排

| 项 | 内容 |
|---|---|
| 写集 | `scripts/exp/paper_smoke/` 全包；`scripts/exp/run_r1d_paper_smoke.py`；`tests/test_paper_smoke_import_contracts.py` |
| 输入 | 冻结旧仿真 import/CLI 和产物 schema；已有 `batch_runner.py`、`run_catalog.py` |
| 输出 | `paper_smoke.manifest.load_manifest/validate_manifest`、`paper_smoke.tasks.resolve_manifest/build_specs_from_manifest`、`paper_smoke.cli.main(argv: Sequence[str] | None = None) -> int`；旧脚本保持接口与 main |
| 必测 | import/CLI + manifest/任务 mock 测试；会启动仿真的测试先申请许可 |
| 验收 | 不修改 `batch_runner`/`run_catalog`；JSON/NPZ schema 不变；直接脚本与模块入口 `--help` 均可；非仓库 cwd 资源寻址不变 |
| 报告 | 常量新归属、旧脚本兼容点、完整测试组结果 |

**R1 共同约束：**

- 不改 `src/sim/v12_robot_adapter.py`、`src/sim/episode_runtime.py`、既有 `tests/test_*.py`（新增测试除外）；
- 需要跨域接线时写进报告，由主代理在 G1 统一修改；
- 不通过 import 旧 facade 取常量；常量迁到真正 owner；
- 每个任务完成后停止写入，报告到 `reports/<task>.md`。

### 5.4 G1：第一轮审查与集成（主代理）

- [ ] 三棵树停止写入；主代理核对每个 diff 不超写集；未授权改动先审查，不盲目回滚；
- [ ] 每个任务一个只读 reviewer（最多 3 个并行；与实现者共享槽位）；
- [ ] reviewer 结论必须为 `approved` 或 `changes_requested`，每条问题有文件/行号、违反契约、最小修复方向；
- [ ] 返修回到原 owner，追加提交；审查绑定新 SHA；
- [ ] 主代理处理共享补丁：旧测试 monkeypatch 路径、父 README、外部调用；
- [ ] 逐任务显式路径 cherry-pick 到 `integration`，顺序 A1 → B1 → C1；
- [ ] 运行 R1 联合测试（A1+B1+C1 测试集去重）；
- [ ] 在 `integration` 跑全量 pytest；与 P0 比较 nodeid 集合，要求无新增失败/错误/跳过；
- [ ] 更新 `contracts.md`、`progress.md`；记录 `G1_SHA`；
- [ ] 归档 R1 三棵树（保留分支）；生成 R2 brief，禁止使用旧源码快照。

**G1 失败处理：** 若全量出现新增失败，停止进入 R2；定位是集成顺序、共享补丁还是任务实现问题；修复后重新跑 G1 全套，不接受“下轮再说”。

### 5.5 第二轮 R2（A2 / B2 / C2）

**gate：** `G1_SHA`。

#### A2：模型装配

| 项 | 内容 |
|---|---|
| 写集 | `src/robot/model_building/`（P1 types 不变）；`src/robot/model_factory.py`；`tests/test_model_building_migration.py`；`model_assets/` 只读 |
| 输出 | `model_building.loading.load_robot_model`、`model_building.template.write_r1d_model_template`；旧工厂签名/返回类型/依赖收集入口不变 |
| 必测 | `pytest tests/test_r1d_model.py tests/test_r1d_assets.py tests/test_r1d_v12_scene.py tests/test_racket_profile.py tests/test_court_scene_integration.py tests/test_robot_package_init.py tests/test_model_building_migration.py -q` |
| 验收 | `__file__`/资源定位不变；左右臂结构/姿态/惯量/TCP/限位不变；不改 XML；`collect_model_dependencies` 不产生第二份资源扫描 |

#### B2：回合编排

| 项 | 内容 |
|---|---|
| 写集 | `src/sim/episode/`（P1 types 不变）；`src/sim/v12_episode.py`；`tests/test_episode_decomposition.py` |
| 输出 | `episode.execution.run_v12_episode` 完整保留原签名；旧路径重导出；各阶段输入/输出/状态所有权记录在 README/报告 |
| 必测 | `pytest tests/test_v12_shared_pipeline.py tests/test_v12_episode_termination.py tests/test_v12_episode_identity_fixes.py tests/test_v12_runtime_restore.py tests/test_v12_dump_trajectory.py tests/test_r1d_v12_acceptance.py tests/test_v12_review_integration.py tests/test_episode_decomposition.py -q` |
| 验收 | 不改 adapter/episode_runtime/产物模块；RNG 消费顺序、配置应用顺序、异常 cleanup、async 退出不变；不得只把 778 行函数整体搬迁；每个有状态对象有唯一所有者和释放位置 |

#### C2：统计分析

| 项 | 内容 |
|---|---|
| 写集 | `scripts/extract/paired_analysis/`；`scripts/extract/paired_stats.py`；`tests/test_paired_analysis_imports.py` |
| 输出 | `paired_analysis.cli.main`；旧统计公有函数/类型/异常按 contracts 显式导出；统计 JSON/CSV 语义不变 |
| 必测 | `pytest tests/test_paired_stats_semantics.py tests/test_paper_stats_catalog.py tests/test_stats_attempt_resume.py tests/test_r1d_statistics.py tests/test_paired_analysis_imports.py -q` |
| 验收 | final attempt、mode 隔离、重复记录拒绝、physical 来源冲突、配对键、bootstrap 不变；不修统计公式；输入复制到独立临时目录 |

### 5.6 G2：第二轮审查与集成

- [ ] 三份独立审查 + 返修；
- [ ] 集成顺序 A2 → B2 → C2；
- [ ] 联合必验：R2 三组测试；模型工厂新旧类型身份；RM65/R1D 固定案例与 P0 行为对照；旧产物入口仍能读当前 Episode 输出；
- [ ] 全量 pytest 对照 P0，无新增失败/错误/跳过；
- [ ] 记录 `G2_SHA`；归档 R2 三树；生成 R3 brief。

### 5.7 第三轮 R3（D1 / E1）

**gate：** `G2_SHA`。

#### D1：产物、快照与回放

| 项 | 内容 |
|---|---|
| 写集 | `src/sim/artifacts/`；`src/sim/episode_artifacts.py`；`tests/test_artifact_import_contracts.py` |
| 输出 | `artifacts.snapshot.write_episode_artifacts`、`artifacts.serialization.load_episode_artifacts`、`artifacts.replay.build_replay_bundle/replay_episode_artifacts`；旧入口参数/返回值不变；`EpisodeArtifacts`/`EpisodeReplayResult` 只在 `artifacts/types.py` 定义一次 |
| 必测 | `pytest tests/test_episode_artifacts.py tests/test_episode_asset_closure.py tests/test_episode_asset_meshdir.py tests/test_episode_replay_sources.py tests/test_multi_robot_replay.py tests/test_artifact_import_contracts.py -q` |
| 验收 | 写盘入口组合 serialization/dependencies；回放组合 validation/materialization；不导入 `episode.execution` 取类型；不反向 import 旧 `episode_artifacts`；schema/资源哈希/来源优先级/replay 参数不变；`_DependencyStager` 迁移后逐阶段验证 |

#### E1：论文图表

| 项 | 内容 |
|---|---|
| 写集 | `scripts/plot/paper/`；`scripts/plot/paper_figs.py`；`tests/test_paper_plot_import_contracts.py` |
| 输出 | `paper.cli.main` 与原绘图函数/CLI；调用签名、输出命名不变 |
| 必测 | `pytest tests/test_paper_figs_paths.py tests/test_paper_figs_tex_escape.py tests/test_r1d_paper_pipeline.py tests/test_paper_plot_import_contracts.py -q`；固定小样本无头渲染 |
| 验收 | 纯数据层不初始化 matplotlib；样式只由明确绘图入口启用；不搬历史数据/图资产；TeX 转义/空值/文件名/数据内容不变；历史函数仍可调用 |

### 5.8 G3：第三轮审查与集成

- [ ] 两份独立审查 + 返修；
- [ ] 集成顺序 D1 → E1；
- [ ] 联合必验：R3 测试；旧回放入口；快照/回放；图表小样本；全量 pytest 对照 P0；
- [ ] 记录 `G3_SHA`；归档 R3 两树；
- [ ] 进入 F0。

### 5.9 F0：导航、全链路与最终审查（主代理）

- [ ] 更新 `AGENTS.md`、根 README、`docs/README.md`、`src/README.md`、`scripts/README.md` 及各父级 README；
- [ ] 每个叶子 README 必须含：何时读、职责/不负责什么、稳定入口/内部文件、允许依赖、测试命令、一个最小用例；
- [ ] 顶层不复制整树；README 路由表逐行验收，所有链接/命令存在；
- [ ] 每个兼容 facade 只保留导出/必要调用适配，不保留第二份实现；
- [ ] 静态扫描“新包反向导入旧 facade”；动态导入/全局可变项按 P0 清单人工复查；
- [ ] 运行新旧 import/CLI、无重型副作用检查；
- [ ] 在 `integration` 跑最终全量 pytest，逐 nodeid 对照 P0；
- [ ] 按原任务 9 跑 RM65/R1D 固定案例、快照回放、论文 smoke→统计→图表小链路；
- [ ] 检查科学身份、配置/资源哈希、schema；记录数值容差及理由；
- [ ] `codegraph sync` + 复查依赖；对变更文件执行 `ruff check`；
- [ ] 未参与实现的 reviewer 做一次全分支只读审查（组合/状态所有权/循环导入/兼容性/文档可发现性）；
- [ ] 写最终报告 `docs/architecture/2026-11-20-sim-experiment-decomposition-result.md`（只有真实执行后创建）；
- [ ] 向用户汇报，不自动合并/推送。

## 6. 测试与验收矩阵

### 6.1 阶段测试

| 阶段 | 命令/范围 | 运行位置 | 通过标准 |
|---|---|---|---|
| P0 | `pytest tests/ -q -p no:cacheprovider` | `p0-baseline` | 记录实测结果（预期 14F+3E）与全部 skip；分类完成 |
| P1 | 5 个指定文件 | `integration` | 全通过；新旧类型 `is` 相同 |
| A1 | 4 个指定文件 | `r1-a1` | 全通过 |
| B1 | 2 个指定文件 | `r1-b1` | 全通过 |
| C1 | import/CLI + mock 测试 | `r1-c1` | 全通过；重测试另许可 |
| G1 | R1 联合 + 全量 | `integration` | 目标测试全通过；全量 nodeid 集合无新增 |
| A2 | 7 个指定文件 | `r2-a2` | 全通过 |
| B2 | 8 个指定文件 | `r2-b2` | 全通过 |
| C2 | 5 个指定文件 | `r2-c2` | 全通过 |
| G2 | R2 联合 + 全量 | `integration` | 同上 |
| D1 | 6 个指定文件 | `r3-d1` | 全通过 |
| E1 | 4 个指定文件 + 无头渲染 | `r3-e1` | 全通过；图/表检查通过 |
| G3 | R3 联合 + 全量 | `integration` | 同上 |
| F0 | 全量 + 固定链路 | `integration` | 无新增失败/错误/跳过；数值/产物/schema 不变 |

### 6.2 已知失败规则

1. P0 生成 `p0-known-failures.txt`（nodeid + 错误类型 + 首行）与 `p0-skips.txt`；
2. 每轮全量对比集合，不允许新增 nodeid；
3. 已有失败不得通过 `pytest.skip`、`xfail`、删除断言、放宽阈值来“变绿”；
4. 若某个失败因为重构而消失，记录为“行为变化”，由主代理确认是否可接受，不静默庆祝；
5. 若错误文本/异常类型变化，即使 nodeid 相同也按回归处理；
6. Linux-only 3 个 loader 失败：P0 记录后保持原样；F0 只验证它们没有变成 error/新增失败。

### 6.3 重测试许可

- 纯函数/小测试可并行；
- 会创建 MuJoCo 模型、渲染、固定输出、批处理、长时间仿真的测试，由主代理发放单一运行许可；
- 每个任务输出到 `results/refactor_subpackages/<task-id>/` 或独立 `--basetemp`，不共享；
- 禁止并发跑多个全量 pytest；禁止在任务树里启动批量实验；
- 每次测试报告记录：worktree、HEAD、环境、输出路径、exit code、通过/失败/跳过数、日志路径。

### 6.4 数值与产物对照

| 对象 | 对照方式 | 容差/规则 |
|---|---|---|
| RM65 T0 NPZ | `results/.../platforms/linux-x86_64/rm65_torque_a.npz` 逐键 | 同平台逐位一致；跨平台仅浮点末位，引用 `cross_platform_*.json` |
| R1D 固定案例 | `configs/scenarios/r1d_v12_active.yaml::expected_by_platform` | 按当前平台标签取期望 |
| 产物 schema | `episode_artifacts` JSON/CSV 字段 | 字段名、类型、来源优先级不变 |
| 配置身份 | `run_id`/config hash | LF 归一化后不变；显式 False/0/None 不丢失 |
| 统计输出 | 固定输入 JSON/CSV | 语义内容一致；bootstrap 随机数固定 |
| 图表 | 固定小样本无头渲染 | 文件名/数据/TeX 转义一致；不做脆弱的跨平台逐字节比较 |

## 7. 质量 Gate

### 7.1 实现者自审

- [ ] `git diff --check` 无空白错误；
- [ ] 只改 brief 允许的路径；
- [ ] 无新增依赖、无 `import *`、无 `src/` 导入 `scripts/`；
- [ ] 新包有 `__init__.py`/README；中文 docstring/注释；类型提示；
- [ ] 目标测试命令与 exit code 写入报告；
- [ ] 若发现跨域修改，只写建议，不动手。

### 7.2 Reviewer 清单

- [ ] Spec 合规：写集、接口、行为约束、测试命令是否与 brief/contracts 一致；
- [ ] 是否复制了第二份实现，或从新包反向 import 旧 facade；
- [ ] `__file__`/资源路径/中文路径/meshdir 是否漂移；
- [ ] monkeypatch/动态 import 是否因重导出失效；
- [ ] False/0/None、配置优先级、RNG 顺序是否改变；
- [ ] 续跑/attempt/physical 来源统计是否重复或混配；
- [ ] 包初始化是否引入循环导入/重型副作用；
- [ ] 测试断言是否被削弱、skip/xfail 是否被滥用；
- [ ] 行数/职责是否内聚；>600 行或 >150 行函数是否有理由；
- [ ] 报告是否完整、可复现。

### 7.3 主代理 Gate

- [ ] 写集核对；
- [ ] 目标测试 + 全量对照；
- [ ] `contracts.md`/`progress.md` 更新；
- [ ] 共享补丁/README/外部调用接线；
- [ ] CodeGraph 刷新；
- [ ] `ruff check` 变更文件；
- [ ] 提交 SHA 与审查 SHA 绑定；
- [ ] 下一轮 brief 重新生成。

## 8. 风险登记

| ID | 风险 | 触发信号 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | 任务树从错误 SHA 创建 / 缺计划文档 | `git rev-parse HEAD` 与基线不符；计划文档缺失 | 假回归/计划漂移 | P-1 固定 `BASELINE_SHA`；建树后逐个 `git rev-parse HEAD` 比对；平台测试前置 |
| R2 | ignored 资源缺失 | C++ ImportError、baseline skip | 测试不可比 | `prepare_resources` + `check_resources` + 指纹 |
| R3 | 全量测试污染主工作区 | `full_suite_pytest.log` 被截断 | 基线证据丢失 | 只在隔离树跑；P-1 备份；运行前后 `pgrep` |
| R4 | Linux-only loader 失败被误当回归 | 3 个 loader nodeid 变化 | 阻塞/错误修复 | P0 分类；不修测试；F0 只查新增 |
| R5 | 重导出后 monkeypatch 失效 | 既有测试 patch 点找不到 | 测试假失败 | P0 清单；主代理改补丁点；保留行为断言 |
| R6 | 循环导入/包级重型副作用 | `import src.sim.configuration` 拉起 MuJoCo | 启动变慢/测试收集失败 | 叶子模块纯导入；P1/F0 导入副作用检查 |
| R7 | 配置 False/0/None 或优先级漂移 | `test_v12_config` 失败/产物 hash 变 | 行为变化 | B1 专项；字段级 golden |
| R8 | RNG/续跑/统计重复 | 固定案例结果变、统计行数变 | 论文数据不可复现 | B2/C2 专项；固定输入对照 |
| R9 | 任务超时/返修循环 | 两次 `changes_requested` | 进度拖延 | 停止扩面，主代理诊断/拆任务；不降低验收 |
| R10 | worktree 被测试扫描/CodeGraph 重复 | 收集到重复测试/符号 | 假失败/索引膨胀 | 外部 `WT_ROOT`；主仓库测试不扫外部目录 |
| R11 | cherry-pick 冲突 | 集成失败 | 返工 | 写集不相交；冲突记录；语义冲突重审 |
| R12 | 用户未确认基线提交 | P-1 无法开始 | 全线阻塞 | 汇报后等待确认；不擅自提交 |

## 9. 工期与里程碑

| 阶段 | 施工日 | 并行度 | 交付物 | Gate |
|---|---:|---|---|---|
| P-1 基线冻结 | 0.5 | 主代理 | `BASELINE_SHA`、标签、资源清单 | 用户确认 |
| P0 基线/契约 | 1.0 | 主代理 | 全量基线、contracts、briefs、ledger | P0 退出条件 |
| P1 公共类型 | 0.5 | 串行 | 三套 types + 重导出 + 测试 | `P1_SHA` |
| R1 A1/B1/C1 | 1.5 | 3 实现 | 三个任务分支 + 报告 | 三份 `approved` |
| G1 | 0.5 | 主代理 | 集成 + 全量对照 | `G1_SHA` |
| R2 A2/B2/C2 | 2.0 | 3 实现 | 三个任务分支 + 报告 | 三份 `approved` |
| G2 | 0.5 | 主代理 | 集成 + 全量对照 | `G2_SHA` |
| R3 D1/E1 | 1.5 | 2 实现 | 两个任务分支 + 报告 | 两份 `approved` |
| G3 | 0.5 | 主代理 | 集成 + 全量对照 | `G3_SHA` |
| F0 | 1.0 | 主代理 + 只读审查 | 导航、全链路、最终报告 | 交付 |
| **合计** | **9.5** | | | |

**说明：** 施工日按 8 小时有效工作估算；三槽并行后墙钟约 6–7 个工作日。若出现契约变更、集成冲突或两次返修，按每次 +20% 估。全量测试 3.5–10 分钟/次，不计入主要瓶颈。

## 10. 汇报、Ledger 与恢复

### 10.1 Ledger

路径：

```text
.superpowers/sdd/sim-experiment-decomposition/progress.md
```

每任务记录：

```text
task_id | owner | write_set | gate_sha | state | report | review | task_sha | integrated_sha | notes
```

状态机：

```text
pending → running → ready_for_review → changes_requested/approved → integrated
```

只有 `integrated` 解锁依赖。上下文恢复先读 ledger 与 `git log`，不重复派发已集成任务。

### 10.2 报告格式

每份实现报告必须含：

- task ID / gate SHA / 实际读写文件；
- 公有符号迁移表；
- 对外接口变化（应为空）；
- 新内部接口；
- 状态所有者；
- 精确测试命令与 exit code；
- 通过/失败/跳过数；
- 日志位置；
- 待集成补丁建议；
- README 路由建议；
- 自审风险。

### 10.3 审查记录

`reviews/<task>-<sha>.md`：

- 结论 `approved` / `changes_requested`；
- 每条问题：文件/行号、违反的契约、最小修复方向；
- 审查者必须核对源码与测试断言，不能只看实现者总结。

### 10.4 用户汇报节奏

- P-1 完成：汇报 `BASELINE_SHA`、改动文件、资源指纹；
- P0 完成：汇报全量基线数字、已知失败分类、契约冻结；
- 每轮 G 完成：汇报集成 SHA、测试对照、风险；
- F0 完成：汇报最终报告、残留风险、合并建议。

## 11. 启动清单（用户确认后执行）

> 以下命令按顺序执行；每一步失败即停止，不跳步。

- [ ] 1. `pgrep -af 'pytest|timeout 900'` 确认无残留测试；若发现其他会话的进程，先确认归属并停止，不能带着全量测试进入 P-1；
- [ ] 2. 确认用户同意 P-1 基线提交分支方案；
- [ ] 3. 执行 §2.1 基线分支 + 显式路径提交 + 标签；
- [ ] 4. 校验 `BASELINE_SHA` 不含 ignored 资源；
- [ ] 5. 创建 `WT_ROOT`、`integration`、`p0-baseline`；
- [ ] 6. 对 `p0-baseline` 执行 `prepare_resources` + `check_resources` + 指纹；
- [ ] 7. 环境自检 + `__file__` 断言；
- [ ] 8. `cd "$WT_ROOT/integration" && codegraph init --local-only -y`；
- [ ] 9. P0 全量基线（隔离树）；
- [ ] 10. 生成 `contracts.md`、`p0-known-failures.txt`、P0 报告；
- [ ] 11. 生成 P1 brief，派发 P1；
- [ ] 12. P1 通过后进入 R1，不提前启动 A1/B1/C1。

## 12. 待确认决策

| ID | 决策 | 推荐 | 备选/影响 |
|---|---|---|---|
| D1 | 基线冻结方式 | 基于 `04a502f` 新建 `sim-exp/baseline`，只提交 `docs/README.md` + 四份架构计划；**主工作区留在该分支**，`main` 引用停在 `04a502f` | 直接提交 `main`：主线前进，但重构未评审；不推荐 |
| D2 | worktree 根目录 | 外部同级 `/data/fxy/research/Tennis_robot_mujoco_sim_worktrees/` | 仓库内 `.worktrees/`：可能被测试扫描/CodeGraph 重复索引；不推荐 |
| D3 | Linux-only 3 个 loader 失败 | 接受为已知基线，不改测试/loader；F0 只查新增 | 修测试加平台 skip：会改变测试语义，且不属于本次重构；不推荐 |
| D4 | Git 推送/合并 | 本轮不 push、不合并 `main`；交付后由用户决定 | 自动合并/推送：违反“不自动合并”约束 |
| D5 | G1/G2/G3 是否跑全量 | 每轮 G 都跑全量并对照 P0；F0 再跑一次 | 只在 F0 跑：不满足“集成回归通过再进下一轮”；不推荐 |
| D6 | 无关未跟踪文档 | 不纳入基线提交，保持原状 | 一并提交：扩大范围，可能夹带未评审内容；不推荐 |

**默认执行假设：** 用户确认本方案后，按 D1–D6 推荐项执行；若用户只确认部分项，未确认项不执行。

## 13. 方案自检

- [x] 每个任务有精确写集、接口、测试命令与验收标准；
- [x] 同轮写集不相交；共享文件/旧测试有唯一 owner；
- [x] P1 类型前置，避免 artifacts 等待不存在的 `episode.types`；
- [x] 每轮最多 3 个子代理，审查者占槽位；
- [x] 每个 gate 有全量回归与 nodeid 对照；
- [x] 已知失败有分类，不通过 skip/xfail 掩盖；
- [x] worktree 从固定 gate SHA 创建，禁止实现者 Git 操作；
- [x] 资源复制/指纹/环境自检有命令；
- [x] 测试污染有隔离与恢复规则；
- [x] 汇报、ledger、恢复、归档有契约；
- [x] 用户评审本方案（2026-11-20 23:17 CST 确认开始执行）；
- [ ] P-1 基线冻结（执行中）；
- [ ] P0 实测基线（待执行）。
