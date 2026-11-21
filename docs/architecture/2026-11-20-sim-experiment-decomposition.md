# 仿真与实验链路子包拆分实施计划

> **执行说明：** 本文为待评审计划，尚未开始实施。按任务使用 `executing-plans`；若选择逐任务子代理实现与审查，则使用 `subagent-driven-development`。完成项使用复选框记录，不把文档落盘视为重构完成。

> **并行执行入口：** 使用 [Subagent 并行执行计划](2026-11-20-sim-experiment-subagent-plan.md)。它保留本文的目标职责和行为约束，替代下面的调度顺序、共享类型迁移时点、文件所有权和提交方式；执行时优先按 P0/P1 与三轮任务卡派发。

**Goal：** 将仿真与实验链路按职责组织为子包，保留现有调用契约，以分层 README 支持按需发现。

**Architecture：** 保留旧模块和 CLI 作为稳定入口；内部实现单向依赖职责子包。沿用 EpisodeRunner 与策略组合，数据处理采用纯函数和明确数据结构，不增加继承层次。

**Tech Stack：** Python 3.11+、MuJoCo、NumPy/SciPy、pytest、CodeGraph；不新增运行依赖。

**Spec：** [调查与设计依据](2026-11-20-maintainability-survey.md)。用户已选择先处理仿真与实验链路，并要求子包及 README 按需发现；具体包边界和任务顺序属于本计划建议，待评审。

**归档位置：** `docs/superpowers/` 被仓库 `.gitignore` 排除；本次长期维护文档放在可入库的 `docs/architecture/`，不修改忽略规则，不强制添加忽略目录。

## 全局约束

- 不改变物理模型、控制算法、限位、统计定义、默认参数、随机数消费顺序和配置优先级。
- 保留现有 CLI 路径、参数、退出码、公有 import、返回类型和产物 schema；不删除旧入口。
- 新包必须有 `__init__.py` 与 README；绝对导入；中文注释/docstring；函数签名有类型提示。
- 纯配置/统计模块不因包级导入加载 MuJoCo、SDK、matplotlib 或启动线程；禁止 `import *`。
- 内部实现不反向导入兼容入口；`src/` 不导入 `scripts/`；不以局部 import 隐藏新循环依赖。
- 不搬动 XML/YAML/历史实验数据，避免 `__file__` 改变造成资源路径漂移。
- Stage4、`src/cpp/`、`src/ilqt/` 算法重构、视频脚本不在范围内；只允许修正受迁移影响的调用位置。
- 文档中新目录均为目标设计；实施前重新核对 HEAD、工作区与签名，尊重并行任务修改。

## 目标子包与职责

所有下列新包都包含 `__init__.py` 和 `README.md`；仅创建当前任务实际需要的文件，不预建空架构。

| 新包 | 内部文件 | 负责什么 | 保留的外部入口 |
|---|---|---|---|
| `src/robot/model_assets/` | `references.py`, `dependencies.py` | XML 引用解析、路径优先级、资源依赖清单 | `robot/model_factory.py` 中现有依赖收集函数 |
| `src/robot/model_building/` | `types.py`, `template.py`, `racket.py`, `scene.py`, `loading.py` | 模型类型、模板/静态臂、球拍、场景参数、加载校验 | `robot/model_factory.py` |
| `src/sim/artifacts/` | `types.py`, `serialization.py`, `dependencies.py`, `snapshot.py`, `validation.py`, `materialization.py`, `replay.py` | 数据编码、资产闭包、快照、验证、物化与回放 | `sim/episode_artifacts.py` |
| `src/sim/configuration/` | `types.py`, `validation.py`, `selection.py`, `defaults.py`, `overrides.py`, `resolution.py` | 配置类型、校验、机器人选择、默认值与覆盖解析 | `sim/v12_config.py` |
| `src/sim/episode/` | `types.py`, `preparation.py`, `assembly.py`, `hooks.py`, `post_hit.py`, `results.py`, `execution.py` | 回合结果类型、准备、组件接线、hook、击球后处理、结果汇总、顺序编排 | `sim/v12_episode.py` |
| `scripts/exp/paper_smoke/` | `manifest.py`, `tasks.py`, `identity.py`, `resume.py`, `artifacts.py`, `catalog.py`, `cli.py` | 清单校验/展开、身份、续跑、产物验收、catalog、CLI | `scripts/exp/run_r1d_paper_smoke.py` |
| `scripts/extract/paired_analysis/` | `types.py`, `identity.py`, `sources.py`, `dataset.py`, `metrics.py`, `comparisons.py`, `report.py`, `cli.py` | attempt、physical 来源、数据集校验、统计与输出 | `scripts/extract/paired_stats.py` |
| `scripts/plot/paper/` | `data.py`, `style.py`, `trajectory.py`, `comparisons.py`, `diagnostics.py`, `tables.py`, `cli.py` | 数据寻址、样式、轨迹/比较/诊断图、LaTeX 表 | `scripts/plot/paper_figs.py` |

`batch_runner.py`、`run_catalog.py` 保留为已有共享执行/目录能力，不复制到 paper_smoke；旧统计与绘图入口的公有辅助函数按调用清单显式重导出。

## 依赖与组合规则

```text
旧入口 → 子包外观/执行函数 → 子包内部实现
model_building → model_assets + 现有 robot 参数/绑定
configuration → robot 配置类型（不触发模型加载）
episode → configuration + 现有 robot adapter + EpisodeRunner/策略
artifacts → model_assets + configuration + episode.types + robot adapter
paper_smoke → 现有 batch_runner/run_catalog + 稳定仿真入口
paired_analysis → 现有 run_catalog + 纯数据/统计
paper → 已生成统计/轨迹数据（不启动实验）
```

Episode 不反向依赖产物写盘。`V12EpisodeResult` 放入 `episode/types.py`；产物按类型读取，不通过执行模块获得类型。配置类型同理放在叶子模块。模型共享 dataclass 放 `model_building/types.py`，由旧入口重导出同一类对象。

`run_v12_episode` 保留签名，内部按准备、组装、执行、后处理、结果汇总调用函数。运行状态只能有明确所有者，禁止所有模块任意写一个巨型 dict。hook 的共享状态用按职责命名的 dataclass，字段从实际读写集合提取，不引入万能 Context。无需状态的模块不创建类。

## Review Focus

1. `__file__` 深度、中文/跨盘路径、嵌套 include 与 meshdir 优先级漂移：任务 1、2、3。
2. 重导出后 monkeypatch 不再作用于定义模块，以及动态 import/脚本运行方式变化：任务 0、4、5、6。
3. 保存配置中显式 False/0/None、默认覆盖优先级与 RNG 顺序改变：任务 4、5。
4. 续跑、不同 mode、重复 seed/attempt、physical 来源冲突造成统计重复或混配：任务 6、7。
5. 包初始化带来循环导入/重型副作用，或图表文件名、TeX 转义、schema 改变：任务 2、8、9。

## 任务 0：冻结契约与迁移基线

**修改：** 本计划的执行记录；新增 `tests/test_refactor_import_contracts.py`（只覆盖真实兼容需求，不为每个重导出写镜像测试）。

- [ ] 记录实际 HEAD、Python/MuJoCo/扩展环境和未提交修改。更新 CodeGraph，查询目标函数 callers/callees，再用 `rg` 查字符串导入、monkeypatch、`__file__`、sys.path 修改。
- [ ] 对下列稳定入口记录 `inspect.signature`、公有 dataclass 类型身份和调用者；新旧路径必须使用同一类型定义。迁移私有补丁点时把测试改到行为接口或真实定义位置，禁止为了测试保留两份实现。
- [ ] 运行 `python -m pytest tests/ -q` 保存完整基线；失败逐项记录，不能把历史 1627/1675 等数字作为当前基线，也不静默排除失败文件。
- [ ] 记录已有固定案例、回放和论文链路测试的输出；为未覆盖的配置/身份/路径契约补特征测试。产物比较排除时间戳和墙钟耗时，但不排除控制轨迹、命中、限位、身份或配置哈希。

稳定接口必须原样保留：

```python
collect_model_file_dependencies(model_path: Path, *, asset_missing: list[str] | None = None) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]
collect_model_dependencies(profile: RobotProfile, parameters: RobotRuntimeParameters, *, arm_fragments: Sequence[Path] = ()) -> tuple[dict[str, Any], ...]
load_robot_model(profile: RobotProfile, parameters: RobotRuntimeParameters, *, racket_profile: RacketProfile | None = None) -> RobotModelBundle
write_episode_artifacts(result: V12EpisodeResult, output_dir: Path) -> dict[str, Any]
load_episode_artifacts(output_dir: Path) -> EpisodeArtifacts
build_replay_bundle(output_dir: Path) -> V12RobotBundle
replay_episode_artifacts(output_dir: Path, *, speed: float = 1.0, viewer: bool = True, bundle: V12RobotBundle | None = None) -> EpisodeReplayResult
resolve_v12_config(args: argparse.Namespace | None = None, explicit_options: Iterable[str] | None = None) -> ResolvedV12Config
```

`run_v12_episode`、`resolve_manifest`、`build_specs_from_manifest` 参数较多，实施时以任务 0 冻结的原函数签名为准，不在搬迁中增删参数。新内部函数根据实际读写集合定型，在所属任务开始前补充到执行记录；不得提前发明通用扩展框架。

## 任务 1：提取模型资源解析

**创建：** `model_assets` 表中全部文件；`tests/test_model_asset_imports.py`。**修改：** `model_factory.py`、产物模块的内部调用、`src/robot/README.md`（不存在则新建）。

- [ ] 将 `_AssetReference`、引用属性解析、候选路径、`_walk_model_file`、`collect_model_file_dependencies` 迁至新包；保留参数来源相关 `collect_model_dependencies` 装配逻辑，委托新解析模块。
- [ ] 校验新包导入不加载 MuJoCo，中文路径、全局 meshdir、声明目录回退、注释忽略行为不变。
- [ ] 执行 `python -m pytest tests/test_episode_asset_meshdir.py tests/test_episode_asset_closure.py tests/test_r1d_model.py tests/test_model_asset_imports.py -q`。
- [ ] README 说明路径优先级的事实来源和故障排查入口；完成审查后独立提交。

## 任务 2：拆产物、快照与回放

**依赖：** 任务 1。**创建：** `artifacts` 表中全部文件；`tests/test_artifact_import_contracts.py`。**修改：** `episode_artifacts.py`、`src/sim/README.md`。

- [ ] 先迁类型与纯序列化，再迁依赖/快照和校验，最后迁 `_DependencyStager`、模型物化与回放；每一阶段维持旧入口可用。
- [ ] 资产校验结果显式传给物化函数；保持缺资产/哈希不匹配的拒绝语义，不增加静默回退。已有类不按行数拆成 Mixin。
- [ ] 执行 `python -m pytest tests/test_episode_artifacts.py tests/test_episode_asset_closure.py tests/test_episode_asset_meshdir.py tests/test_episode_replay_sources.py tests/test_multi_robot_replay.py tests/test_artifact_import_contracts.py -q`。
- [ ] 验证纯类型/序列化导入副作用，旧回放入口、输出文件/schema/资源哈希不变；README 分开“写产物”“资产损坏”“回放不一致”路由；独立提交。

## 任务 3：拆模型装配

**依赖：** 任务 1；建议任务 2 后进行，减少同文件交叉修改。**创建：** `model_building` 表中全部文件。**修改：** `model_factory.py`、已有模型测试的必要导入位置、`src/robot/README.md`。

- [ ] 模型类型先迁；模板/静态臂、球拍、场景参数、加载/校验分别迁移；旧工厂保持装配顺序和公有函数签名。
- [ ] 路径基于现有 profile/显式根目录解析，不用新文件的 parent 层数猜测原路径；不改写生成 XML 的资产内容。
- [ ] 执行 `python -m pytest tests/test_r1d_model.py tests/test_r1d_assets.py tests/test_r1d_v12_scene.py tests/test_racket_profile.py tests/test_court_scene_integration.py tests/test_robot_package_init.py -q`。
- [ ] 对照左右臂关节/TCP/惯量、固定初态与场景结构；更新装配 README 并独立提交。

## 任务 4：拆配置解析

**创建：** `configuration` 表中全部文件。**修改：** `v12_config.py`、相关测试的补丁位置、`src/sim/README.md`。

- [ ] 迁类型/冻结函数、验证器/规则表、机器人选择、默认值、覆盖和最终解析；保留原 `resolve_v12_config` 与 `ResolvedV12Config` 导出。
- [ ] 不将机器人差异变成配置子类；用现有注册表/数据组合；明确显式 False/0/None 与“未提供”的区别。
- [ ] 执行 `python -m pytest tests/test_v12_config.py tests/test_v12_runtime_overrides.py tests/test_v12_runtime_restore.py tests/test_robot_package_init.py tests/test_refactor_import_contracts.py -q`。
- [ ] README 描述配置来源/优先级和新增参数应修改的位置；独立提交。

## 任务 5：拆回合编排

**依赖：** 任务 2–4。**创建：** `episode` 表中全部文件。**修改：** `v12_episode.py`、必要的类型导入、对应测试补丁点。

- [ ] 先移动 `V12EpisodeResult`，更新产物模块直接依赖类型模块，阻断产物与执行的循环。
- [ ] 为准备、组件组装、hook、击球后处理、结果汇总列出输入/输出/状态所有者；然后分段提取 `run_v12_episode`，不得仅整函数改位置。
- [ ] 保留 EpisodeRunner/StrategyConfig；配置应用顺序、随机数调用顺序、async 资源释放和 unreachable/异常结果均维持原行为。
- [ ] 执行 `python -m pytest tests/test_v12_shared_pipeline.py tests/test_v12_episode_termination.py tests/test_v12_episode_identity_fixes.py tests/test_v12_runtime_restore.py tests/test_v12_dump_trajectory.py tests/test_r1d_v12_acceptance.py tests/test_v12_review_integration.py -q`。
- [ ] 比较任务 0 固定案例的轨迹、命中、接触和配置身份；写清执行阶段与 hook 所属 README；独立提交。

## 任务 6：拆论文 smoke 编排

**依赖：** 任务 2、4、5 稳定后实施。**创建：** `paper_smoke` 表中全部文件；`tests/test_paper_smoke_import_contracts.py`。**修改：** 原 CLI、`scripts/exp/README.md`。

- [ ] manifest 规范化/校验迁 `manifest`；任务展开迁 `tasks`；run/config 身份迁 `identity`；恢复及完成记录选择迁 `resume`；JSON/NPZ/log 交叉检查迁 `artifacts`；catalog 输出迁 `catalog`；main 迁 `cli`。
- [ ] `load_manifest`、`validate_manifest`、`resolve_manifest`、`build_specs_from_manifest` 保留签名与返回结构；复用现有 ExperimentSpec/batch_runner/run_catalog，不新增第二个调度器。
- [ ] 验证直接脚本与模块方式的 `--help` 均成功；从非仓库工作目录使用绝对脚本路径仍保持既有资源寻址行为。
- [ ] 执行 `python -m pytest tests/test_r1d_experiment_runner.py tests/test_r1d_paper_smoke_contract.py tests/test_paper_smoke_data_root.py tests/test_paper_run_identity.py tests/test_batch_output_isolation.py tests/test_r1d_paper_pipeline.py tests/test_paper_smoke_import_contracts.py -q`。
- [ ] README 区分“改实验清单”“续跑失败”“产物不一致”的阅读入口；独立提交。

## 任务 7：拆统计输入与配对分析

**依赖：** 任务 6 的身份契约稳定。**创建：** `paired_analysis` 表中全部文件；`tests/test_paired_analysis_imports.py`。**修改：** `paired_stats.py`、`scripts/extract/README.md`。

- [ ] 数据类/异常迁 `types`；身份与最终 attempt 选择迁 `identity`；CSV/catalog/physical 来源融合迁 `sources`；数据集/模式校验迁 `dataset`；基础检验迁 `metrics`；实验对照/DiD 迁 `comparisons`；格式/表/summary 迁 `report`；入口迁 `cli`。
- [ ] 保留现有 HitPolicy、physical/legacy、配对键、重复记录拒绝、缺失值和样本排除语义；固定 bootstrap 随机数规则。
- [ ] 执行 `python -m pytest tests/test_paired_stats_semantics.py tests/test_paper_stats_catalog.py tests/test_stats_attempt_resume.py tests/test_r1d_statistics.py tests/test_paired_analysis_imports.py -q`。
- [ ] 对相同固定输入比较统计 JSON/CSV 的语义内容与错误分类；README 列出数据身份/统计计算/报告格式三类入口；独立提交。

## 任务 8：拆图表输出

**依赖：** 任务 7 输出契约稳定。**创建：** `paper` 表中全部文件。**修改：** `paper_figs.py`、`scripts/plot/README.md`。

- [ ] 路径和数据读取、样式、轨迹图、对照图、诊断图、LaTeX 表、CLI 分别迁移；保留旧图函数名和 CLI。
- [ ] 不把统计计算复制到绘图包；保存文件名、目录、图中数据及 TeX 转义保持一致；样式/后端初始化保持受控，不在纯数据模块导入时执行绘图。
- [ ] 执行 `python -m pytest tests/test_paper_figs_paths.py tests/test_paper_figs_tex_escape.py tests/test_r1d_paper_pipeline.py -q`。
- [ ] 使用任务 0 固定小样本无头渲染代表性轨迹图/对照图和 TeX 表，检查空值与中文/特殊字符；图片不做脆弱的跨平台逐字节比较。README 与输出示例同步；独立提交。

## 任务 9：导航收口与联合验收

**修改：** `README.md`、`README.zh-CN.md`、`src/README.md`、`scripts/README.md`、`docs/README.md`、`AGENTS.md` 的相关入口；所有本轮新增 README。

- [ ] 顶层只保留领域概览和阅读触发条件，将详细模块清单下沉到所属 README，不在多个层级复制同一整棵目录树。
- [ ] 从源码/脚本总索引到具体模块最多经过两级 README；每个叶子 README 必须含：何时读、职责/不负责什么、稳定入口与内部文件、允许依赖、测试命令、一个最小用例。
- [ ] 新入口落地后才添加链接；检查链接和命令路径存在；对未来未实施子包只写规划文字。`AGENTS.md` 保留安全/语言/事实来源规则，模块细节用精确链接路由。
- [ ] 检查新旧 import/CLI、循环依赖和导入副作用；运行 `python -m pytest tests/ -q`，与任务 0 逐项比较，要求无新增失败/跳过。既有问题必须说明，不能仅挑一次全绿。
- [ ] 同环境复跑固定 RM65/R1D 案例、快照回放、论文 smoke→统计→图表小链路；不开展批量调参。启动仿真前加载项目 sim_run 技能，产物留在约定输出目录。
- [ ] 对变更文件执行 Ruff；若当前基线存在规则问题，记录差异，不顺带全仓库清理。刷新 CodeGraph 并复查迁移入口调用关系。
- [ ] 填写完成报告：迁移文件、兼容入口、验证证据、剩余候选和已知限制；独立提交。不自动合并或推送。

## README 按需发现示例

| 用户/维护者要做什么 | 一级入口 | 叶子说明 |
|---|---|---|
| 排查模型资源缺失、meshdir 或跨盘路径 | `src/robot/README.md` | `model_assets/README.md` |
| 修改球拍或场景装配 | `src/robot/README.md` | `model_building/README.md` |
| 排查配置没有生效 | `src/sim/README.md` | `configuration/README.md` |
| 修改回合 hook 或后处理 | `src/sim/README.md` | `episode/README.md` |
| 排查快照损坏、回放不一致 | `src/sim/README.md` | `artifacts/README.md` |
| 改清单、续跑或验收产物 | `scripts/exp/README.md` | `paper_smoke/README.md` |
| 查重复 seed、attempt 或配对口径 | `scripts/extract/README.md` | `paired_analysis/README.md` |
| 改论文图、表或输出路径 | `scripts/plot/README.md` | `paper/README.md` |

## 交付节奏与回退

建议三个独立评审批次：A=任务 0–3（资源/模型/回放），B=任务 4–5（配置/回合），C=任务 6–9（实验/统计/绘图/导航）。任务 0 的全量基线为三批共同起点，各批在当前基线基础上复查。

每个任务保持入口可运行、独立提交；失败只回退本任务提交，保留此前已验证的子包。兼容入口的删除不属于本计划。执行中发现无法保持语义的循环依赖、资源所有权不清或需要改变 schema 时，先记录具体问题并修订设计，不把行为变更夹进搬迁。

行数不是硬性验收门槛；若拆分后某实现仍超过约 600 行或函数超过约 150 行，复查职责是否内聚并记录理由，不能为达标制造无意义转发层。

## 自审与执行记录

- [x] 调查证据与候选优先级已记录。
- [x] 已区分仿真/实验主线与 Stage4 后续工作。
- [x] 已定义子包归属、兼容入口、依赖方向及 README 路由。
- [x] 五类高风险情形已映射到任务和验证。
- [ ] 用户评审包边界、任务顺序与执行方式。
- [ ] 任务 0 基线完成后补充实际 HEAD、测试结果、动态调用/补丁清单。
- [ ] 实施任务 1–9，逐项填写提交与验证结果。
