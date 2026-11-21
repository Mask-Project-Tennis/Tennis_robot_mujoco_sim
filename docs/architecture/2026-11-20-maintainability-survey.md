# 仿真与实验链路可维护性调查

- 日期：2026-11-20。
- 状态：调查记录；拆分建议尚未实施。
- 主仓库基线：`main @ 20397eb`，V12 已合入；Stage4 位于独立 worktree。
- 范围：第一方 Python 源码、活跃脚本；排除第三方资产和 `scripts/archive/`。测试文件不与产品源码混排。
- 后续：[子包拆分计划](2026-11-20-sim-experiment-decomposition.md)。

## 方法与限制

使用本机 CodeGraph CLI 执行 `status`、`sync`、`callers`、`callees`，用 Python AST 统计文件物理行数、类/函数定义跨度及直接方法数量，并抽查源码职责。索引更新前存在待同步内容；本文调用关系取同步后的结果。CodeGraph 是静态线索，不能完整解析 hook、动态派发、字符串加载和 monkeypatch，不把图节点数当作实际运行调用次数。

行数包含空行、注释和 docstring，不等同于复杂度。AST 全目录扫描遇到旧脚本 Python 3.11 不接受的 f-string 语法时跳过其符号分析；因此不是全仓库语法验收，也没有运行测试。以下候选均已成功解析并核对。

Stage4 的诊断源码和测试在调查时已有未提交修改，数值来自当时工作区，并非纯提交快照；本次调查未修改其代码。它只作为后续候选，不进入首轮实施范围。

## 规模与职责证据

| 路径/符号 | 规模 | 观察 | 决策 |
|---|---:|---|---|
| `scripts/exp/run_r1d_paper_smoke.py` | 2,868 行 | manifest 校验、任务展开、身份、执行恢复、产物验收与 catalog 输出 | 首轮拆分 |
| `src/sim/episode_artifacts.py` | 2,750 行 | 配置序列化、资源闭包、快照、路径物化、回放 | 首轮优先 |
| `scripts/extract/paired_stats.py` | 2,682 行 | 身份/attempt 去重、physical 来源对齐、统计、报告和 CLI | 实验链路第二阶段 |
| `src/robot/model_factory.py` | 2,433 行 | XML 装配、球场、球拍、依赖路径解析、MuJoCo 加载与检查 | 先提取资源解析，再提取装配 |
| `src/sim/v12_config.py` | 2,158 行 | 配置类型、校验、默认值、CLI/覆盖优先级、机器人选择 | 首轮独立任务 |
| `scripts/plot/paper_figs.py` | 2,091 行 | 数据寻址、样式、多组图、LaTeX 表、CLI | 实验链路最后阶段 |
| `src/sim/v12_episode.py` | 2,039 行 | 配置应用、环境构造、来球、管线接线、后处理、结果组装 | 首轮拆分 |
| `run_v12_episode` | 778 行 | 单函数承担多个有顺序要求的阶段 | 保留入口，拆执行阶段 |
| Stage4 `UnitreeR1DInterface` | 1,286 行 / 30 方法 | 生命周期、锁、发送、停止、看门狗共享状态 | 暂缓；不机械分裂状态所有权 |
| Stage4 `_DiagnosticCollector` | 1,134 行 / 24 方法 | 执行、采样、指标、验收、写盘 | 后续独立设计 |
| `MPCController` | 703 行 / 15 方法 | 已有策略注入，仍有较大调度实现 | 首轮不改 |
| `RM65Env` / `BoundRobotEnv` | 673 / 647 行 | 较完整的仿真环境接口 | 不仅因行数拆分 |

另有 `scripts/run_20hits_video.py::run_batch` 895 行、`src/ilqt/replan_core.py::do_replan` 451 行。视频和控制算法不纳入本次结构迁移，避免同时改变算法与工程组织。

## CodeGraph 与源码交叉结论

1. `run_v12_episode` 查询返回 27 条调用方/导入关系和 51 条被调用关系，包含 CLI、测试、配置应用、机器人构建、来球与击打判定。它是稳定入口，也是职责集中的位置；不能仅移动整个函数后宣称完成拆分。
2. `collect_model_file_dependencies` 查询返回 15 条调用方/导入关系，其中 `_model_asset_entries`、`_derive_dependency_entries` 位于产物模块，`collect_model_dependencies` 位于模型工厂。这是可验证的共享职责，适合提取成低层资源解析模块。
3. `episode_artifacts` 的 `_DependencyStager` 单类约 400 行；模块还包含资产校验、快照写入和回放执行，拆分应按资源生命周期，而非平均行数。
4. `EpisodeRunner` 已组合 perception、MPC、safety、executor；`StrategyConfig` 已提供策略注入。应沿用这套组合方式，不新增机器人专属 Episode 继承树。
5. `paired_stats` 同时拥有 `RecordIdentity`、`StatsContext`、attempt 选择、physical 数据融合、配对检验和表格生成；统计口径与输入规范化应分离，但不能在搬迁中改写统计定义。

## 方案取舍

| 方案 | 收益 | 代价 | 结论 |
|---|---|---|---|
| 只增加 README | 快、风险低 | 大函数和跨职责修改仍存在 | 作为每步配套，不能独立解决问题 |
| 保留入口，内部迁入子包 | 可逐步回归、保留调用契约、职责可独立测试 | 需要管理兼容入口和补丁位置 | 推荐 |
| 全量改名、统一新框架/基类 | 表面目录统一 | 导入、CLI、序列化、并发回归范围过大 | 不采用 |

“组合优于继承”在本项目中的落实：复用现有执行器和策略；数据处理优先纯函数；有资源或运行状态才使用对象；只有确有替换需求才引入 Protocol。不为每个小函数创建类，也不引入通用插件注册中心、BaseEpisode 或拆分用 Mixin。

## 按需发现

当前 `src/README.md`、`scripts/README.md` 汇总较多细节，而目标目录缺少分层入口。建议顶层只保留职责和路由，二级 README 解释稳定入口，叶子包 README 说明模块归属、依赖限制、测试和示例。

每条链接必须说明“什么时候读”：例如资源缺失/meshdir 问题进入模型资产文档，回放不一致进入产物与快照文档，重复 seed/attempt 问题进入统计输入文档。未来目录在计划中以代码文字表示，包实际落地后才发布可点击链接，避免把规划误写成已实现。

## 本轮结论与边界

先处理仿真与实验链路；Stage4、安全阈值、控制算法、物理参数、历史实验结果均不变。拆分以依赖方向、行为等价、可定位性为验收条件，行数只作为复查信号。历史测试通过数字不作为此次重构基线，实施前重新记录环境与测试结果。
