# R1D V12 论文实验迁移矩阵

2026-11-18，只读调查形成的实施范围；尚未迁移或运行。执行期T0复核当前文件并补齐每行完整命令。

## 事实入口

paper/main.tex → sections/experiments.tex、results.tex；scripts/extract/analyze_all.py的REPORTED/SUBMODULES；paper/README.md图表清单。旧docs/experiments/README.md不是当前论文范围的完整来源。

## 数据集合

正文统计的19组：exp15_speed_v2、exp16_limits_v2、exp17b_perturb、exp17d_mechanism、exp17e_perturb7、exp17f_mechanism_perturb、exp17g_spatial_power、exp17h_extreme、exp17i_limits_ablation、exp18_sensitivity、exp19_hardmin、exp21_hardmin_tcp10、exp22_shared_spatial、exp23_nofreeze、exp24_shared_full、exp25_select_one、exp26_tcp_grid、exp27_ball_quality、exp28_always_collision。

额外依赖：exp18_fig_assets（同步/异步计时和NPZ）、exp18_tcp_exempt/window_stats.csv（豁免窗口）、exp17a_noise中无噪声none对照（exp25/28读取）。exp17c_obsfreq、exp13_arch、exp14_pd_v2作为后续扩展，不因目录存在自动加入当前正文验收。

## 精确文件清单

| 层 | 需要审查/适配的现有文件 |
|---|---|
| 运行 | scripts/exp/overnight_experiments.py、batch_runner.py、collect_fig_assets.py |
| 专用旧入口 | scripts/exp/run_exp15_speed_sweep.py、run_exp16_limits_comparison.py：T0检查调用方后决定是否转发共享运行器 |
| 分析入口 | scripts/extract/analyze_all.py |
| 分析子模块 | scripts/extract/paired_stats.py、aux_stats.py、extract_chat8_aux.py、check_shared_input.py、analyze_chat12_night.py、analyze_exp28_collision.py |
| 绘图 | scripts/plot/paper_figs.py、scripts/tools/render_swing_hero.py |

## 图表映射

| 资产 | 数据/生成链 | 本次交付 |
|---|---|---|
| fig5_hit_rate_vs_speed.pdf | paper_figs.py --fig 5；统计JSON＋exp26_tcp_grid | 隔离目录预览 |
| fig6_tube_robustness.pdf | paper_figs.py --fig 6；17系列统计JSON | 同上 |
| fig8_realtime_performance.pdf | collect_fig_assets → timing.json → paper_figs --fig 8 | 本机实际计时，不继承RM65实时结论 |
| table1_comparison.tex、table2_ablation.tex | paired_stats → JSON → paper_figs --fig table | 隔离表格；小样本不当论文统计结论 |
| table3_shared_input.tex | 当前手工维护 | 新增数据驱动生成能力及隔离预览，不能仅声称已有链路 |
| fig3_swing_hero.pdf | NPZ → render_swing_hero → PNG/JSON → paper_figs --fig 3hero | 按robot/config建模，七关节兼容 |
| fig1_handdrawn.pdf | ICRA.pptx → paper/tools/pptx2fig1.py | 人工概念资产，本次不更新 |
| fig2_algorithm_flowchart.tex | 手工TikZ | 人工概念资产，本次不更新 |

文件名不等于正文图号，应按引用关系对照。

## 必须消除的隐式依赖

- batch_runner只认__RESULT__；固定60s超时、16workers与七关节FD不匹配。新增timeout/workers显式选项；R1D默认workers=1，超时由T7实际耗时制定并保存在manifest，不能把计算慢视为miss。
- resume现按(config_id,seed)，新增robot/mode/racket/配置hash/metric schema命名空间；重复执行兼容性检查，错配目录报错或新建目录，禁止混入旧结果。
- `--limit 1`仅取一条任务，不是所有实验cell覆盖；smoke按每种cell/tier构造最小配对样本。
- overnight_experiments::_exp28_seed_pool构造时读取旧exp24 CSV；改为基于当前data-root的延迟解析，保持原实验选择规则。依赖数据不足时明确报缺，不退回旧根目录。
- analyze_all及六子模块统一data-root/output参数；analyze_chat12_night导入时不得执行或写文件。
- paired_stats当前硬写paper/planning；paper_figs即使指定stats/out仍有固定DATA读取，必须贯穿数据根；collect_fig_assets同理。
- render_swing_hero不能固定RM65模型或将七维数据塞入六维环境。

## 小样本覆盖规则

各实验每个比较cell至少两个共同seed（实施默认100、101，与T7开发/留出集分离），保证配对链路可计算；特殊条件选择依赖不足时报告不可估计并用合成fixture验证极端统计分支，不制造“合格”实测样本。全miss、缺tier、无共同seed、超时/no-verdict必须显式表现，不输出虚假零置信区间。

T8工具默认采用一个模式（torque）遍历全部正文实验条件，另以position覆盖入口/前馈/七关节统计与图表的代表链；两种模式的模式适用性、完整重跑命令均交付。这是一项控制smoke规模的实施默认，不取消任何模式的可运行能力。需要模式对照的实验保留两模式。

每行最终补齐：论文引用位置、实验矩阵入口、必要依赖、模式、smoke条件数、原始日志/CSV、统计命令、图表命令、full命令、实际验证状态。所有输出在experiment_data/r1d_v12_parity/paper_smoke内，paper/及旧数据前后哈希不变。

## T0 复核状态（2026-11-19，只读复核）

复核报告：`results/r1d_v12_parity/t0_paper_closure.md`（144 行）与 `results/r1d_v12_parity/t0_paper_closure.json`。结论：CSV 级闭包基本成立（抽样重算与论文数值一致），per-episode `raw/*.log` 级不成立（除 `exp18_fig_assets`/`exp18_tcp_exempt`/`exp28_substep` 外，历史 raw 日志因系统重装丢失）。19 组正文实验 **verified 15 / partial 4 / missing 0**；额外依赖 verified 2 / partial 1。

| 实验 | 论文位置 | 数据 | 统计/图表产物类型 | 状态 |
|---|---|---|---|---|
| exp15_speed_v2 | experiments.tex:41-49,83；results.tex:12-20 | results.csv 1803 行 | paired_stats e1→06-stats-active.json；paper_figs --fig5 | verified |
| exp16_limits_v2 | experiments.tex:41-49；results.tex:16-19 | 403 行 | paired_stats e2→JSON；paper_figs --fig5 | verified |
| exp17b_perturb | experiments.tex:96-110；results.tex:22-61 | 4003 行 | paired_stats e4→JSON；fig6/table1 | verified |
| exp17d_mechanism | experiments.tex:96-110；table1 | 600 行 | paired_stats e3→JSON；fig6/table | verified |
| exp17e_perturb7 | experiments.tex:96-110（7m/s 复证） | 4000 行 | **无统计/绘图消费者** | partial |
| exp17f_mechanism_perturb | experiments.tex:96-110；fig6(b)(c) | 4000 行 | paired_stats e4→JSON；fig6 | verified |
| exp17g_spatial_power | experiments.tex:96-110；results.tex:63-105 | 18000 行 | paired_stats e7→E7_corners；fig6 | verified |
| exp17h_extreme | experiments.tex:96-110；results.tex:63-105 | 24000 行+extract_summary | e7→JSON；fig6 | verified |
| exp17i_limits_ablation | experiments.tex:112-122；table2 | 3200 行+extract_summary | e8→JSON；--fig table | verified |
| exp18_sensitivity | results.tex:107-111 | 2600 行 | aux_stats.json；仅文字 | verified |
| exp19_hardmin | results.tex:107-111 | 401 行 | extract_chat8_aux.py **stdout only** | partial |
| exp21_hardmin_tcp10 | results.tex:127 | 800 行+config.yaml | **运行入口缺失，无统计入口** | partial |
| exp22_shared_spatial | 无唯一正文行；planning 文档称 V-B 数据源 | 51000 行（784 重复键） | check_shared_input.py **stdout only** | partial |
| exp23_nofreeze | experiments.tex:124-140；results.tex:152-164 | 3200 行 | analyze_chat12_night.py **导入即执行/stdout** | partial |
| exp24_shared_full | results.tex:83-105；table3 | 2000 行 | analyze_chat12_night stdout；table3 手写 | verified（数值）/统计产物缺失 |
| exp25_select_one | results.tex:41-48 | 600 行 | analyze_chat12_night stdout | verified（数值）/统计产物缺失 |
| exp26_tcp_grid | experiments.tex:77-94,112-122；results.tex:131-139 | 2400 行 | 无 JSON；paper_figs:900/1004 现算 | verified（数值）/统计产物缺失 |
| exp27_ball_quality | results.tex:55-61 | 800 行 | analyze_chat12_night stdout | verified（数值）/统计产物缺失 |
| exp28_always_collision | results.tex:63-105（180/180） | 180 行 | analyze_exp28_collision.py stdout | verified（数值）/统计产物缺失 |

**额外依赖**：`exp18_fig_assets` 数据 verified（timing.json 141866B + manifest + 9 npz + 52 timing logs），但正文 CPU 型号无 artifact 字段 → partial；`exp18_tcp_exempt/window_stats.csv` verified；`exp17a_noise` 无噪声 none 对照 verified。

**T8 必须处理（按优先级）**：
1. **统计产物缺口**：`paper/planning/06-stats-active.json` 仅含 E1–E8；selection/quality（exp25/27）、端到端 shared-input（exp24）、unfrozen（exp23）、hard-min（exp19/21）、速度×cap（exp26）、always-on（exp28）无落盘 JSON，`analyze_chat12_night.py` 导入即执行且零落盘。T8b 必须先补可指定 `--data-root/--output` 的 JSON 落盘（或明确以 stdout+CSV 为统计入口并写清限制），否则不得声称分析链完整。
2. **运行入口缺口**：`exp21_hardmin_tcp10` 已从 `scripts/exp/overnight_experiments.py` 消失（HEAD 无该键），需按 `config.yaml` 的 `config_ids` 恢复定义；`exp17e_perturb7` 无统计/绘图消费者，7m/s 复证句目前不可再生成；`exp22_shared_spatial` 正文归属待作者确认（不得擅自改论文，只能在交付中标注）。
3. **专用旧入口**：`run_exp15_speed_sweep.py`/`run_exp16_limits_comparison.py` 为 legacy（输出目录不同、`--sim-limits` 已废弃），不作为当前运行入口；当前 exp15/16 由 `overnight_experiments.py` + `batch_runner.py` 生成。
4. **已证实的隐式依赖行号**：`batch_runner.py:39` `__RESULT__`、`:40` `RUN_TIMEOUT_S=60`、`:277-292` workers 默认 16、`:121-141` resume `(config_id,seed)`；`overnight_experiments.py:210-250` 模块级读旧 exp24 CSV；`analyze_all.py:38-40` 固定 DATA；`analyze_chat12_night.py` 无 `main()`/无 `__main__`；`paired_stats.py:33-36` 硬写 paper/planning；`paper_figs.py:46-47` 固定 DATA/OUT（另有 exp18 npz、exp26 CSV、timing.json 固定读取）；`collect_fig_assets.py:36-59` 固定根与 180s；`render_swing_hero.py:57/377-380` 固定 RM65 模型与默认 episode。
5. **统计口径**：论文 active-hit = `hit_type=="active"`（`paired_stats.py:125-139`），不是 CSV 的 `hit` 列（含 passive）。R1D 重算必须显式区分 physical/legacy，不得用 `hit` 列导致系统性偏高。
6. **数据边界**：本次只从 CSV 与新跑数据出发；历史 per-episode 日志不可恢复，不得伪造；所有新产物写入 `experiment_data/r1d_v12_parity/`，`paper/` 与旧数据前后哈希不变。

## 实施状态（2026-11-19，T7–T9 交付）

| 链路 | 状态 | 证据/边界 |
|---|---|---|
| T7 R1D 固定案例（seed42 两模式） | ✅ torque/position 均真实 `face` active；`failure_reason=null`；禁止碰/紧急停止 0 | `configs/scenarios/r1d_v12_active.yaml`；`tests/test_r1d_v12_acceptance.py`；报告 `reports/2026-11-18-r1d-v12-parity.md` |
| T7 留出集 0–9×2 | ✅ 20/20 status=ok，无删行；torque 10/10 active（触后 7–11 步 safety_stop）；position 3/10 active | `experiment_data/r1d_v12_parity/validation/`；不设下限、未反向调参 |
| T8a 运行器 | ✅ manifest 19 组+3 依赖、69 cases（68 ready/adapted + exp25 blocked）、默认 torque 126 tasks、position 10；`--dry-run/--limit/--workers/--timeout` | `configs/experiments/r1d_paper_smoke.yaml`；提交 `7185c60` |
| T8b 统计 | ✅ 8 入口 data-root/output、physical/legacy 双口径、缺数据显式不可估计；旧统计 JSON sha 不变 | 提交 `e922faf`；报告/评审 `results/r1d_v12_parity/reviews/t8b_review.md` |
| T8c 绘图/hero | ✅ `paper_figs --data-root/--out/--fig table3`；`render_swing_hero` R1D 7 关节；per-case smoke 图补齐 | 提交 `c7a10f8`、`2fae2d3`；整链 `results/r1d_v12_parity/paper_smoke/chain.log` |
| T8 整链 smoke | ⚠️ 2 seeds×1 case 全 miss（`safety_filter_stop`）；Fig.5/6 因数据不足跳过，table3 五格全 `--`（不可估计） | 按契约如实留空，不补数；不构成论文统计结论 |
| T9 文档/回归 | ✅ 全量 pytest **12 failed / 1579 passed / 3 errors**（12F+3E 与 T0 基线逐 ID 相同，无新增回归）；RM65 seed42 两模式对 T0 11 键 max_abs_diff=0.0；R1D 两模式固定案例 active；paper/旧数据零写入；整分支 final-review PASS（0 阻断） | `results/r1d_v12_parity/t9/full_final.log`；`results/r1d_v12_parity/reviews/final_review.md`；`docs/r1d_v12_usage.md`；提交 `a695f89`、`ebf40c8` |
| paper/旧数据 | ✅ 前后 1448 文件哈希一致；未改 `paper/`、旧 `experiment_data/exp*` | `results/r1d_v12_parity/paper_smoke/paper_sha256_before/after.txt` |
