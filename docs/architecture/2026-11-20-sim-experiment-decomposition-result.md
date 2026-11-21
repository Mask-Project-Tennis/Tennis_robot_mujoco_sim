# 仿真/实验子包拆分结果与验收（2026-11-20 计划）

## 1. 结论

仿真与实验链路的 7 个大文件已拆为深子包，旧模块路径保留为兼容 facade（同一对象、无第二份实现）。P1、R1/G1、R2/G2、R3/G3 全部完成；F0 文档、静态检查、固定案例与全量回归已完成，首轮 final-review 的 3 项阻断与复审新增的 7 处旧 facade import 均已修复并复跑全量。

- 最终集成分支：`sim-exp/integration`（保留，不 push/不合并 `main`）
- 基线：`sim-exp/baseline` @ `399e568`；`main` 仍停 `04a502f`
- 回归基线：P0 14F+3E+7S；G1/G2/G3 全量均只含 P0 已知失败
- 旧入口：`src.robot.model_factory`、`src.sim.v12_config`、`src.sim.v12_episode`、`src.sim.episode_artifacts`、`scripts/exp/run_r1d_paper_smoke.py`、`scripts/extract/paired_stats.py`、`scripts/plot/paper_figs.py`

## 2. 目标与约束

- 保持 import、CLI、数值结果、产物 schema、统计口径、图名/TeX 不变。
- 拆分后单文件职责清晰；不为行数指标制造纯转发层。
- 依赖单向：`types ← 实现模块 ← facade`；`src/` 不 import `scripts/`；`artifacts` 不反向依赖 `episode`。
- 所有代码注释/docstring 中文；类型提示；绝对导入；pathlib；无新增依赖。
- 每轮独立审查、显式 cherry-pick、全量 gate；实现者不得做 Git 写操作。

## 3. 基线与工作树

| 项 | 值 |
|---|---|
| 仓库 | `/data/fxy/research/Tennis_robot_mujoco_sim` |
| 基线提交 | `399e568`（`docs(refactor): freeze sim/experiment decomposition plans`） |
| 基线父提交 | `04a502f`（Windows/Linux 双平台 T0 基线） |
| 工作树根 | `/data/fxy/research/Tennis_robot_mujoco_sim_worktrees/` |
| 集成树 | `integration` @ `sim-exp/integration` |
| 基线树 | `p0-baseline` @ `sim-exp/p0-baseline` |
| 外部树 | `tuning-prep` @ `359cf4e`，未触碰 |

P0 全量：**14 failed / 2328 passed / 7 skipped / 3 errors**；17 个唯一 FAILED/ERROR nodeid 冻结为 `p0-known-failures.txt`，7 个 skip 冻结为 `p0-skips.txt`。3 个 loader 失败为 Linux-only；其余 11 个为历史已知。

## 4. 任务与提交

| 阶段 | 任务 | 任务提交 | 集成提交 | 目标测试 |
|---|---|---|---|---|
| P1 | 四个共享类型迁移 | `69fb119` | `69fb119` | 87P/2S |
| R1 | A1 模型文件依赖 | `60877f3` | `5978c89` | 60P/2S |
| R1 | B1 配置解析 | `c0a820b` | `7cee586` | 75P |
| R1 | C1 论文 smoke | `1e37866` | `d83ae90` | 26P |
| R2 | A2 模型装配 | `b67077a` | `08fe350` | 154P/2S |
| R2 | B2 回合编排 | `3ac71bb` | `871bdad` | 130P |
| R2 | C2 配对统计 | `6c9823e` | `8fcaf09` | 206P |
| R3 | D1 产物/回放 | `76fd4d7` | `89910ab` | 83P |
| R3 | E1 论文图表 | `d21ab60` | `82bddfd` | 101P/1S |
| F0 | `.gitignore` 修复 + 文档 + final-review 修复 | final-reviewer | `224c82c` + `7faf339` + `542119c` + `5208206` + `8f7c6e3` | approved / 可交付；见 §7 |

### 4.1 拆分结果

| 旧文件 | 新包 | 旧路径 |
|---|---|---|
| `src/robot/model_factory.py`（2433 行） | `src/robot/model_assets/` + `src/robot/model_building/` | 418 行 facade |
| `src/sim/v12_config.py`（2158 行） | `src/sim/configuration/` | 精简 facade |
| `src/sim/v12_episode.py`（2039 行） | `src/sim/episode/`（阶段函数） | 169 行 facade |
| `src/sim/episode_artifacts.py`（2804 行） | `src/sim/artifacts/` | 187 行 facade |
| `scripts/exp/run_r1d_paper_smoke.py`（2868 行） | `scripts/exp/paper_smoke/` | CLI facade |
| `scripts/extract/paired_stats.py`（2682 行） | `scripts/extract/paired_analysis/` | 387 行 facade |
| `scripts/plot/paper_figs.py`（2091 行） | `scripts/plot/paper/` | 198 行 facade |

- `V12EpisodeResult` 在 `src/sim/episode/types.py`；`RacketAssembly`/`RobotModelBundle` 在 `src/robot/model_building/types.py`；`ResolvedV12Config` 在 `src/sim/configuration/types.py`。
- 旧/新入口的对象同一性由 `tests/test_refactor_shared_types.py`、各任务的 import 契约测试与 G1/G2/G3 reviewer 独立核验。
- `src/sim/episode/__init__.py` 只导出 `V12EpisodeResult`，避免 import 期拉起实现模块；`run_v12_episode` 的稳定入口仍是 `src.sim.v12_episode`（facade）。
- `scripts/extract/paired_analysis/__init__.py` 只从 `types` 轻量导出；完整统计 API 从子模块按需导入。

## 5. 关卡与证据

| Gate | 范围 | 全量结果 | 审查 |
|---|---|---|---|
| G1 | `69fb119..7cee586` | 14F / 2382P / 7S / 3E | approved；`reviews/G1.md` |
| G2 | `7cee586..8fcaf09` | 14F / 2509P / 7S / 3E | approved；`reviews/G2.md` |
| G3 | `8fcaf09..224c82c` | 14F / 2586P / 7S / 3E | approved；`reviews/G3.md` |

- 每轮 17 个 FAILED/ERROR nodeid 与 P0 逐 ID 一致，无新增/缺失。
- G3 新增 passed 77 = D1/E1 新测试 collected 77。
- 证据目录：`integration/.superpowers/sdd/sim-experiment-decomposition/`
  - `contracts.md`、`progress.md`
  - `p0-known-failures.txt`、`p0-skips.txt`、`p0-full-suite.log`
  - `f0-known-failures-final.txt`、`f0-skips.txt`、`f0-full-suite-final.log`
  - `reports/{p0-baseline,P1,A1,B1,C1,A2,B2,C2,D1,E1}.md`
  - `reviews/{G1,G2,G3,final}.md`
  - `results/refactor_subpackages/{g1,g2,g3,f0}/full_suite.log`（F0 最终为 `full_suite_p3.log`）

## 6. 兼容性与行为证据

- A1：旧/新模型依赖闭包 JSON 35629 bytes、MD5 一致；`collect_model_dependencies` 只调用一次文件级扫描。
- B1：六配置场景 84740 bytes、MD5 一致；错误文本/来源优先级/False-0-None 不变。
- C1：manifest 计数 126 tasks / 22 units / 19 experiments / 3 dependencies；`main`/`ManifestError` 同一对象。
- A2：6 个 R1D XML sha256 不变；`_arm_fragment_path` 不依赖 CWD。
- B2：同 seed 42 短回合 `X/U/ball_pos/ball_vel/time/racket_pos` 全部 `np.array_equal`；metrics 仅 timing 键不同。
- C2：`--help` 与旧实现逐字节一致；单模式 JSON 语义 diff 0；`--mode both` 5 输出逐字节；坏数据 rc=2、stderr 逐字节。
- D1：`_DependencyStager` AST 逐字迁移；schema/哈希/来源优先级/回放不变；23 个公开/私有名字同一性通过。
- E1：11 图 `diff_px=0`；table1/2/3 字节相同；matplotlib 冷启动不加载；旧 `import paper_figs` 可用。

## 7. F0 验证

- 文档：根 `README.md`/`README.zh-CN.md`、`AGENTS.md`、`docs/README.md`、`src/README.md`、`scripts/README.md`；新增 `src/robot/README.md`、`src/sim/README.md`、`scripts/exp/README.md`、`scripts/extract/README.md`、`scripts/plot/README.md`；8 个叶子 README 补齐六要素（含 `paired_analysis` 的“允许依赖”节）。
- 静态：`ruff check` 改动文件（新 `scripts/extract/paired_analysis/__init__.py` 0 错误；历史 E1 风格告警未新增）；旧 facade 无第二份实现；`src/`→`scripts/` 0 条；无 import 期模型加载/仿真/写文件。
- final-review 阻断已修复：首轮 3 项（6 处 `src.sim.v12_config` 类型导入、README 失败口径/测试规模、`paired_analysis` 允许依赖）；复审新增 7 处旧 facade import（`src/sim/artifacts/dependencies.py` 6 处 `src.robot.model_factory`、`scripts/exp/paper_smoke/manifest.py` 1 处 `_OVERRIDE_VALIDATORS`）改为新子包入口，并新增 `allowed_override_keys()` 公开 helper。
- final-review 最终结论：**approved / 可交付**（`reviews/final.md`；范围 `224c82c..8f7c6e3`；无阻断；新包旧 facade import 扫描 0、`src/`→`scripts/` 0）。
- 审查后 P3 兼容性修复：`scripts/plot/paper_figs.py` 增加 `__all__`，显式包含 `PROJECT`/`DATA`/`OUT` 与全部公有重导出；`test_facade_star_import_exports_dynamic_globals` 覆盖 `from scripts.plot.paper_figs import *`。此项仅恢复旧通配符导出契约，不改绘图行为。
- 冒烟：新旧 import、CLI `--help`（paper smoke/stats/figures 均 rc=0）、RM65/R1D 固定案例、artifact snapshot replay、paper smoke→stats→figs 小链路。
- 全量（P3 修复后）：**14 failed / 2587 passed / 7 skipped / 16 warnings / 3 errors in 215.73s**；17 个 FAILED/ERROR nodeid 与 P0 排序后逐 ID 一致；7 个 skip 的节点/原因与 P0 一致；证据 `f0-known-failures-p3.txt`、`f0-skips.txt`、`f0-full-suite-p3.log`。
- 定向全链路：`tests/test_v12_shared_pipeline.py`、`tests/test_r1d_v12_acceptance.py`、`tests/test_artifact_import_contracts.py`、`tests/test_paper_smoke_data_root.py`、`tests/test_paired_stats_semantics.py`、`tests/test_paper_figs_tex_escape.py` = **90 passed, 1 skipped in 61.08s**；final-review 复审修复后定向 8 文件 **166 passed in 2.23s**（artifact/episode/paper smoke/config/refactor/paired imports）；P3 修复后 `tests/test_paper_plot_import_contracts.py` **70 passed in 1.13s**（含通配符导出测试）。
- P3 修复后组合定向：`test_paper_plot_import_contracts.py test_paper_smoke_import_contracts.py test_paper_smoke_data_root.py` = **80 passed in 1.72s**；无新增失败。
- `codegraph sync` 在 `integration` 树执行：1,512 files / 26,111 nodes / 65,982 edges / 103.95 MB；`status` 正常。

## 8. 已知限制/后续

- 3 个 Linux loader 失败是测试假设 Windows 临时副本分支导致，不在本次拆分修复；不要通过改测试/skip 掩盖。
- 6 个 mpc + 3 entrypoint F+3E + 2 closed-loop 为历史已知；根因见 P0 报告。
- `p0-baseline` 使用分支而非 detached，属计划轻微偏差；归档时记录。
- `integration` 树保留交付；任务树已删除、`sim-exp/*` 分支保留。未 push、未合并 `main`。
- 后续若继续清理：优先处理 `template.py` 929 行、`preparation.py` 852 行、`results.py` 961 行的职责复查，而不是继续制造转发层。

## 9. 验收判定

- 拆分目标：完成。
- 兼容性：旧 import/CLI/对象同一性/数值/产物/统计/图表证据齐全。
- 回归：三轮 gate 与 F0 全量均只含 P0 已知失败。
- 文档：按需发现路由与叶子 README 六要素完成。
- 结论：**可交付**；剩余失败为已冻结的环境/历史基线，不是本次重构新增。
