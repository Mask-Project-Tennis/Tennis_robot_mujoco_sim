# R1D × V12 多 worker 调参实验设计

日期：2026-11-20。状态：**设计存档，尚未实现运行器、尚未启动实验**。

本方案按用户要求保留供后续执行。候选范围、评价阈值和分阶段预算是本次提出的设计值，不是已测得的最佳参数；正式搜索前必须冻结为配置。配套的代码修复、球场白线改造及 subagent 分工见 [并行修复计划](../../r1d_v12_parallel_repair_plan.md)。

## 1. 目的与前提

T7 固定 seed42 已证明两模式能够产生真实拍面 active 接触，但冻结配置的 torque 模式明确通过较短评价时域结束回合；历史留出集还出现触后腕 pitch safety stop。应专门研究“接触后能否安全完成动作”，不能只优化接触成功率。历史留出集不是当前修复版本的新结果，也不能再次作为盲测集。

实验回答三个问题：控制参数能否提高完整安全击球率；收益能否在新来球上重现；对估计阻尼、armature、球拍质量和弹跳系数的失配是否敏感。torque、position 分别选参、分别报告。模型真实物理参数与控制器参数分层，禁止把增加仿真阻尼当作控制性能提升。

启动条件：

- 配套修复计划 F01–F08 关闭，尤其是配置恢复执行、实际资产校验、稳定任务身份和按模式统计。
- 新评价协议能够贯穿接触后的安全过滤和状态记录；安全拒绝后不再推进物理步。旧的正常结束后保持段不能直接充当新协议。
- 实际注入的 damping/armature、执行模型与规划模型的差异均可保存、哈希、还原；更改注入常量会改变实验身份。
- 使用独立工程实验目录 `experiment_data/r1d_v12_tuning/<campaign_id>/`，不修改 `exp*` 历史数据、T7 冻结配置和论文结果。
- 本实验是仿真参数研究，不作为硬件能力、实时性或论文完整复现的证明。

## 2. 采用的并行方式

| 方式 | 取舍 |
|---|---|
| 在现有 paper smoke 中直接放开 workers | 改动少，但该入口明确要求 workers=1，还承载论文依赖链；不足以解决任务身份、恢复评价及续跑契约。此次不采用。 |
| **独立调度器 + 每回合独立 Python 子进程** | 采用。调度器只排队、监控、汇总，MuJoCo 和求解器全部在子进程中；进程结束即释放回合状态，容易隔离超时和崩溃。 |
| 常驻 MuJoCo worker 池 | 可能节省导入时间，但需证明模型、RNG、warm-start 和异常状态被完整重置。仅在启动开销实测成为瓶颈后另做优化，不作为首版前提。 |

实现建议：主进程用有界 `ThreadPoolExecutor` 管理 `Popen`，线程只等待各自子进程，**不在线程中执行仿真**。`workers=W` 表示最多 W 个仿真子进程；待提交队列不超过 `2W`。复用修复后的身份、日志和目录契约，不复用 paper smoke 的串行约束。Python 的 [subprocess](https://docs.python.org/3.11/library/subprocess.html) 与 [concurrent.futures](https://docs.python.org/3.11/library/concurrent.futures.html) 文档是实现参考；取消 Future 本身不会终止已经运行的子进程。

每个子进程建立独立模型、规划环境、RNG、临时目录和输出目录。关闭 viewer、绘图和 async replanner，固定 `first_plan_flight=False`；并发调度不应改变算法输入或取决于墙钟的重规划时机。计时对比另用隔离运行，批量并发的回合 wall time 只用于吞吐与资源报告。

## 3. 固定评价协议 v1

| 项目 | 预注册值 / 规则 |
|---|---|
| 时间步长 | 保持当前 R1D `dt=0.005 s` |
| 评价总时域 | 从同一初始物理状态起 **2.5 s / 500 步**，两模式、所有候选一致；安全失败立即停止，正常规划结束不缩短观察 |
| 接触后观察 | 首次合格接触后至少 **0.5 s / 100 步**；接触晚于 2.0 s 则本回合不能判完整成功 |
| 稳定恢复 | 评价末尾连续 **0.1 s / 20 步**，所有驱动关节 `abs(qdot) <= min(0.5 rad/s, 0.1*qdot_max)`，TCP 速度 `<=0.1 m/s` |
| 固定安全条件 | 保持现有位置、速度、加速度、力矩、TCP、禁止接触检查及阈值；本协议全程不使用终段豁免，安全过滤器的 `k_hit_remaining=max(99, terminal_exempt_steps+1)`，不影响 MPC 内部真实 remaining；禁止扩大限值或忽略 safety stop 提分 |
| 初步超时 | 每回合墙钟上限 900 s；仅允许在 P0 吞吐试运行后、P1 搜索开始前统一修订并冻结 |
| 规划时域 | 保持冻结案例的 `mpc.total_horizon=270`、`fixed_horizon=260`，本轮不搜索；**评价时域独立于这两个值** |

`contact_success` 必须同时满足：`physical_contact=true`、`surface_kind=face`、`other_geom_name=racket_face`、同接触物理帧拍速 `>0.3 m/s`、物理判定为 active。最小距离单独记录，不能用距离近似替代接触事件。

主指标 `safe_complete_success` = 合格接触 + 足够触后观察 + 全评价期无安全拒绝/EMERGENCY/禁止接触/数值异常 + 达到末段稳定标准。`status=ok` 只说明程序完成，不能代替这个指标。未打中、提前结束、观察不足均为主指标 0，并保留具体原因。

评价器必须有显式阶段：原 MPC/随挥 → 安全恢复保持 → 评价结束。原控制策略不提前切换：只有它正常结束后才进入统一的恢复保持；`no_follow_through=true` 也不能跳过恢复。恢复参考是正常结束帧的关节位置，使用该候选的位置 PD 增益（torque 模式使用冻结 runtime 的 kp/kd）及原有前馈约定生成保持控制，逐步经过同一安全过滤。不能在 safety stop 后执行这段控制，不能使用未经过滤的历史 20 步保持代码。若进入保持时约束不允许安全恢复，记录失败。这是新增的评价执行行为，须单独验证，不能声称仅延长 `max_steps` 就已实现。

当前实际 EpisodeRunner 未把 MPC 的 remaining 赋给安全过滤器，后者默认 99；不能因看到注释中的“终段豁免”就给本次基线新增触前豁免。上述全程关闭规则显式冻结当前基线的实际语义，并防止新增恢复流程把负 remaining 传入而跳过检查。该规则只用于新评价协议，不顺带重写 legacy 执行行为。

同时报告接触前/后 safety stop 数与关节名、触后停止延迟、最小关节限位裕量、最大 qdot/TCP、接触拍速、位置误差和恢复时间。失败样本同样保留原始时序。控制量诊断定义为 `control_effort_integral = sum_k(sum_j(tau[k,j]^2))*dt`，单位 N²·m²·s；tau 取驱动关节实际 `qfrc_actuator + qfrc_applied`，不含被动力/约束力，不能将 position 的 U=q_des 当力矩。它仅在完成全部 500 步且记录完整的样本上描述，附有效 n，其他样本为 null；不作为排名或晋级依据，不称其为机械能耗。

## 4. 预算：9,940 回合，按阶段设屏障

| 阶段 | 每模式设计 | 两模式回合数 | 决策 |
|---|---|---:|---|
| P0 诊断 / 吞吐 | 冻结基线 × 50 个新案例 | 100 | 验证完整观察、确定 workers 与统一超时；不据此修改候选范围 |
| P1 粗搜 | 48 组配置（含基线）× 40 个共用案例 | 3,840 | 每模式选择 4 个非基线候选 |
| P2 候选复验 | 4 候选 + 基线 × 200 个新案例 | 2,000 | 每模式固定 1 个候选；其他候选不进入压力筛选 |
| P3 物理失配压力 | 1 候选 + 基线 × 10 条件 × 50 个新案例 | 2,000 | 检查是否保留候选；失败则保留基线，不在本阶段改选第二名 |
| P4 锁参盲测 | 已锁定候选 + 基线 × 500 个新案例 | 2,000 | 最终报告；看过结果后不再调参 |
| **合计** |  | **9,940** | 不含基础设施单元测试、吞吐重复校验与中断恢复试验 |

若 P1 少于 4 个候选满足硬条件，只让实际合格的 1–3 个晋级；若 P1 无合格候选或 P2/P3 无可推广候选，该模式停止、报告原因，不为凑足预算重复跑完全相同的基线。9,940 是完整流程上限；额外诊断回合另记 `diagnostic`，不混入科学样本。

每阶段必须等预定任务全部取得终态或被明确标为缺失后才能选择。不得让先完成、较快成功的配置优先入围。搜索顺序由独立的 `schedule_seed` 打乱；每种模式、候选交错调度。单回合是最小调度单位，不把一个候选的全部 seeds 固定绑在某个 worker。

## 5. 候选范围与选参规则

以修复后有效配置展开值为基线 `B`，保持底座、升降、初始姿态、球拍、工作区、来球分布及安全上限一致。所有乘数先展开成绝对数值写入 candidate 文件，再从原始配置层解析；候选覆盖在 derive/禁用随挥归零之前应用，不能只修改已经归一化的 B。每模式固定 48 组：第 0 组为原 T7 控制参数，另 47 组从下表有限集合独立均匀抽取，RNG 为 NumPy `default_rng(20261120)`，torque 先生成、position 后生成；按**有效配置**去重后继续抽样至 47 组。生成器版本、完整候选列表和摘要在搜索前保存；禁用随挥时不因无效 follow 参数产生多个等价候选。

| 维度 | torque | position | 实际配置映射 |
|---|---|---|---|
| 目标拍速 | 1.0 / 1.5 / 2.0 / 2.5 / 3.0 m/s | 0.75 / 1.0 / 1.25 / 1.5 / 1.8 m/s | `mpc.racket_speed` 与 `mpc.target_speed` 同值；不是放宽 TCP 安全上限 |
| 法向代价 | 0 / 5e3 / 5e4 / 5e5 | 同左 | `mpc.normal_weight` |
| 控制代价 | B × 0.5 / 1 / 2 / 4 | 同左 | `cost.R` |
| 控制变化代价 | B × 1 / 2 / 4 / 8 | 同左 | `cost.Q_du`；阶段乘数保持 B |
| 随挥开关 | 关闭 / 开启 | 保持 B | `mpc.no_follow_through` |
| 随挥比例增益 | 100 / 200 / 300 | 保持 B | `mpc.follow_kp` |
| 随挥阻尼增益 | 20 / 40 / 60 | 保持 B | `mpc.follow_kd` |
| 终端目标前移量 | 0.04 / 0.08 / 0.12 m | 保持 B | `mpc.follow_through_length`，当前用于规划终端目标偏移，不等于实际随挥轨迹总长 |
| 肩肘 kp 比例 | 保持 B | 0.75 / 1 / 1.25 | `actuator.kp[0:4]` |
| 肩肘阻尼比比例 | 保持 B | 0.75 / 1 / 1.5 | `kd[0:4] = B.kd * sqrt(kp比例) * 阻尼比比例` |
| 腕 kp 比例 | 保持 B | 0.5 / 0.75 / 1 / 1.25 | `actuator.kp[4:7]` |
| 腕阻尼比比例 | 保持 B | 1 / 1.5 / 2 | 同上公式，应用 `actuator.kd[4:7]` |

torque 开启随挥的候选额外固定 `mpc.follow_through_steps=80`（0.4 s）、`mpc.follow_through_v_terminal=0.0`、`mpc.follow_trigger=planned`；关闭时 steps/length/v_terminal 全部归零。基线第 0 组仍为原 T7 关闭随挥，position 随挥值保持其原 T7 配置。这样不会把 torque 基线已归零的 steps=0 带进“开启”分支；80 步是本设计的固定选择，不是从实验推得的最优值。

即 torque 搜索 8 维、position 搜索 8 维；不做全笛卡尔积。若基线某个代价为 0，乘法无法形成不同候选，预检必须提示并在搜索前修订设计版本，禁止运行时静默补值。接口必须证明候选覆盖确实进入执行与规划环境，尤其是 position 的 kp/kd，以及 torque 开启随挥时策略实际触发。

P1/P2 按以下顺序排序：完整安全成功率降序 → 禁止接触率升序 → 触后 safety stop 率升序 → candidate_id 字典序。排序中的三种率统一以该候选的全部计划案例数为分母；另报“已发生合格接触案例中的触后停止率”作为条件诊断，不能替换排名分母。先做硬否决：出现禁止接触或数值发散的候选不推广；安全过滤正常拒绝属于失败样本，不删行。P1 不把几十次多重比较的 p 值当结论。控制量积分不参与同分决胜，避免提前失败轨迹因更短而被偏爱。

P2 晋级至少要求候选的完整安全成功率点估计高于基线；否则该模式停止。P3 候选在每个压力条件的完整安全成功率不能低于同条件基线超过 0.10，且不得出现禁止接触/发散；这个 0.10 是探索阶段预注册门槛，不是统计非劣性证明。

P4 只比较每模式一名候选与基线，使用相同案例的成对差值。输出成功率及 Wilson 区间、成对 bootstrap 差值区间、精确 McNemar 检验。两个模式构成两个确认比较，显著性使用 Holm 校正；同时给出未经校正和校正值。只有效果方向正确、校正后支持提升且安全硬条件满足时才称“本测试分布上优于基线”。若证据不足，如实报告；不能继续挑 P2 其他候选来替换 P4 结论。

## 6. 来球集、随机数与物理压力

所有阶段名义分布为当前 serve-box、7 m/s、距离 8 m、弹跳系数 0.75，其他分布边界取冻结场景与有效配置。每个 case 提前保存实际 `p0/v0`、生成参数和噪声流身份，而不是仅记录整数 seed。不同候选、两模式均消费同一 case；候选增益不得参与来球生成。名义噪声配置保持基线，未来新增噪声/速度分布须另建实验版本。

| 分区 | 新 case seed | 用途 |
|---|---|---|
| P0 | 10000–10049 | 诊断 |
| P1 | 11000–11039 | 搜索 |
| P2 | 20000–20199 | 选择复验 |
| P3 | 30000–30049 | 同一批来球跨 10 物理条件配对 |
| P4 | 40000–40499 | 锁参后才生成、运行并查看输出 |

0–9、42、论文 smoke 的 100/101 只用于历史回归，不进入这些集合。P4 的生成器及 seed 范围可预注册，但其状态、轨迹和结果在候选锁定前不用于调整。每个 case 的噪声按 `(case_id, 传感器通道, 物理采样步)` 确定，不能按候选实际调用 RNG 的次数推进；设置观测频率时仍保持可对应的输入时钟。

P3 的 10 条件为：标称；执行侧 damping ×0.5 / ×1.5；执行侧 armature ×0.5 / ×2；球拍质量 ×0.9 / ×1.1；执行侧 restitution=0.65 / 0.85；组合最难条件（damping ×0.5、armature ×2、球拍质量 ×1.1、restitution=0.85）。质量缩放同时按比例缩放惯量，保持几何与质心。**仅执行模型改变，规划模型保留标称估计**，并保存两套实际数组/哈希；若一起改变，就没有测试模型失配。

这些区间用于敏感性试验，不声称来自实机标定。P3 不能反向优化物理常量。之后若获得实测阻尼，应更新物理基线、创建新 campaign，重新验证，而不是覆盖此次候选结果。

## 7. 身份、目录和单写入者

运行前生成不可变任务表，禁止 worker 临时枚举配置。科学身份与执行位置分开：

- `candidate_id`：模式 + 展开后控制配置内容的摘要。
- `case_id`：初始球状态、噪声协议/seed、生成器版本的摘要。
- `task_id`：schema + 代码内容身份 + robot/racket/实际物理模型哈希 + mode + candidate_id + case_id + 压力条件 + 评价协议哈希的 SHA256。
- `campaign_id`：冻结设计、代码身份、初始候选列表、各阶段输入生成规则与 seed 范围的摘要/可读前缀；不能仅使用日期。后续阶段再由晋级记录和实际 case 内容生成独立 `phase_manifest_hash`，不为提前得到 campaign_id 而读取 P4 输入。
- `attempt_id`：一个 task 的第几次实际启动。重启不改变 task_id，不覆盖旧 attempt。

输出路径、worker 数、机器名、启动时间不进入科学 task_id，单独记录为 execution metadata。代码身份优先使用干净提交；若允许未提交代码，必须保存 tracked diff 与参与运行的新增文件的内容清单/hash，不能只写 HEAD。控制配置摘要不包含 seed，case_id 单独承担输入身份。

```text
experiment_data/r1d_v12_tuning/<campaign_id>/
├── config.yaml                  # 冻结设计和资源设置
├── catalog.json                 # 显式索引，不猜父目录
├── candidates/                  # 每模式展开后的配置
├── cases/                       # 分阶段输入；P4 锁参后生成
├── tasks/<phase>.jsonl           # 不可变任务清单
├── raw/<task_id>/attempt-0001/   # stdout.log、result.json、NPZ、快照
├── journal.jsonl                # 仅协调器追加任务状态
├── results.csv                  # 从有效终态重建，按 task_id 排序
├── decisions/                   # P1/P2/P3 晋级及锁参记录
└── reports/                     # 吞吐、分模式指标、缺失与排除原因
```

worker 仅写自己的 attempt 目录，先写临时文件，完整刷新/关闭后在同一目录原子替换 `completion.json`；其中记录 result/NPZ/snapshot 的摘要与退出状态。主进程验证任务身份、schema、文件摘要及输出完整性后才写 journal、生成 CSV。不能仅凭 log 存在判 done，不能让多个 worker 追加同一 CSV，不能在异常时把半份结果当成功。

`catalog.json` 显式记录任务表、results、run_root 与实际 artifact 路径。路径采用相对 catalog 文件的路径，必要时允许绝对路径；搬迁整棵树后相对引用仍有效。跨根数据使用明确引用，禁止猜 `data_root.parent`。最终按模式输出，样本唯一键为完整 task_id；配对用同一 mode/物理条件/评价协议中的 case_id，对比不同 candidate_id，不能把候选配置哈希也放进导致永远配不上的 join key。

## 8. 超时、断点续跑和失败口径

| 状态 | 续跑 / 统计行为 |
|---|---|
| 正常完成（命中或 miss） | 终态；续跑跳过，按真实指标计入 |
| safety stop / 数值失败 / 子进程 error / timeout | 终态；续跑默认跳过，主指标为 0；保留日志、原因和能取得的最后状态 |
| 用户取消 / 协调器崩溃且无有效 completion | 标记 interrupted；下一次续跑创建新 attempt，原记录保留，不将中断伪装成 miss 或成功 |
| completion 已落盘但汇总未写入 | 验证后补记同一终态，不能再启动任务 |
| 已有终态但摘要损坏或多个终态互相冲突 | 隔离并报错；不自动选成功结果、不自动覆盖 |

任何科学失败不自动重试。显式诊断重试保存新 attempt，但主分析仍使用首次终态；若确认基础设施系统性错误使一批实验无效，应给出统一失效规则、保留原数据并另建 campaign，不能逐 seed 挑成功尝试。未完成的计划任务在汇总中计数，阶段未闭合时不能晋级；阶段全部闭合后主指标分母为计划任务数，error/timeout 不删掉。诊断缺少物理判定时同时报 unknown 数，不能冒充观测到的物理 miss。

超时由调度器从实际启动时刻计时，不能只给 `future.result()` 设置 timeout。Windows 对每个任务使用受本运行器拥有的 Job Object（`KILL_ON_JOB_CLOSE`）管理进程树；子进程须先加入容器再开始仿真，禁止 breakaway，失败即拒绝启动。POSIX 使用独立进程组，并在协调器断开时通过启动器心跳/控制管道退出，重启前确认旧任务组已结束。停止只针对本次拥有的进程，不能全局结束 python。实现依据见 [Microsoft Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)。

同一个 campaign root 只允许一个协调器持锁，记录 PID、启动身份与心跳；不能仅因超时就抢活跃锁。崩溃恢复先确认拥有的旧进程已结束，再释放死锁。首版为单机多 worker，不引入分布式队列。未来多机可按 `int(task_id,16) % shard_count` 静态分片，每片独立目录和单写入者，最后核对分片集合无重复/无缺失再合并。

## 9. worker 数量与吞吐验收

子进程导入数值库前固定 `OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`，固定 `PYTHONUTF8=1`。记录 Python/MuJoCo/NumPy/BLAS 版本、CPU、物理核、可用内存及线程环境；不预先宣称 GPU 能加速当前 Python/有限差分求解路径。

P0 的 50 案例/模式划成五组 10 案例，分别测 W=1/2/4/8/16；超过物理核或内存预算的档位不启动，余下组在已验证的最佳档位执行。每个候选档位还用同一小型固定样本作重复吞吐对照，重复启动记为诊断，不计入 9,940。对照要求 W=1 与 W>1 的有效配置、输入、物理判定一致；同机确定性模式下 X/U 一致，若不一致必须先定位，不能直接展开搜索。

W 的上限同时满足 CPU 和内存约束：`W <= 物理核数`，`W * 单回合峰值RSS + 协调器占用 <= 0.7 * 启动时可用内存`。首个高档位前已有单 worker 内存实测。选择无新增异常且吞吐接近最大值（差距不超过 5%）的最小 W，保留系统余量。

记录总吞吐（回合/小时）、完成数、error/timeout、每回合 wall 的 p50/p95/p99、启动耗时、峰值内存、CPU 利用率。预算估计使用实际吞吐：`剩余小时 = 剩余回合数 / 已验证回合每小时`；原先 1 万次 1–2 小时的经验可作目标，但当前 R1D 路径须单独实测。若 P0 达不到目标，仍先保存诊断，不改变科学指标来追求速度。

## 10. 后续实现拆分与契约

本节是未来开发范围，不表示这些脚本/参数现在可用。先完成并行修复计划的基础门禁，再实现调参工具；球场视觉改造可以独立交付，不是搜索的前置条件。

| 工作包 | 新建 / 主要修改文件 | 交付与依赖 |
|---|---|---|
| T0 协议与输入 | `scripts/exp/r1d_tuning_schema.py`、`r1d_tuning_cases.py`、`configs/experiments/r1d_v12_tuning.yaml`；测试 `tests/test_r1d_tuning_schema.py` | 冻结 dataclass/schema、生成候选/cases/tasks、内容身份；其他任务依赖其接口 |
| T1 单回合与恢复 | `scripts/exp/_run_r1d_tuning_episode.py`、`src/sim/episode_evaluation.py`；集成 `v12_episode.py`、`v12_robot_adapter.py`、`episode_artifacts.py`、必要的 `episode_runner.py`；测试 `tests/test_r1d_tuning_evaluation.py` | 消费 T0 的 TaskSpec；实现实际双环境物理覆盖/快照和完整评价，保持 legacy 默认行为 |
| T2 多 worker 调度 | `scripts/exp/run_r1d_tuning.py`、`scripts/exp/episode_process.py`；测试 `tests/test_r1d_tuning_scheduler.py` | 依赖 T0，可用假子进程与 T1 并行开发；单写入者、进程退出、resume |
| T3 提取统计 | `scripts/extract/extract_r1d_tuning_results.py`；测试 `tests/test_r1d_tuning_results.py` | 依赖 T0，可用合成 completion 与 T1/T2 并行；配对、分模式、失败分母、阶段选择 |
| T4 接线验收 | 协调者负责 CLI 接线及 `scripts/README.md`、`src/README.md`、`AGENTS.md` 索引 | T1–T3 完成后一次端到端测试；随后才允许 P0 |

T0 提供不可变 `TaskSpec`（第 7 节全部身份及 config/case/physics/evaluation 输入路径）、`AttemptResult`（task_id、attempt_id、status、failure_reason、各指标、artifact manifest）、`CampaignConfig`。公有接口定为：

```python
load_campaign(path: Path) -> CampaignConfig
build_tasks(config: CampaignConfig, phase: str) -> list[TaskSpec]
load_task(path: Path, task_id: str) -> TaskSpec
run_episode_task(task: TaskSpec, attempt_dir: Path) -> AttemptResult
run_campaign(config: CampaignConfig, *, workers: int, resume: bool) -> int
summarize_campaign(catalog_path: Path) -> dict[str, Any]
```

T0 定义字段和 JSON 序列化后再派 T1/T2/T3；接口改变由协调者一次性更新，不让三个代理各写一套 task schema。T1 写入源文件时，修复计划的 A/B/I1 必须已交付；若 I2 球场也在进行，协调者先释放相关源文件所有权，再派 T1，不能同时修改适配器/产物文件。T1 在双环境初始化后、首步前应用 TaskSpec.physics，保存实际执行与规划两套 damping/armature、质量/惯量和 restitution；回放必须消费保存值而非重新取当前常量。默认无该新协议时沿用旧行为。T0 的身份函数消费这些明确的物理输入，T1 回传实际值作一致性校验。

最小验证集合：workers=1/4 结果等价；并发上限不超 W；一次超时不会挂住整批；取消/崩溃不留拥有的子进程；completion 已写但 CSV 未写能恢复；更换输出根/worker 数仍命中相同 task_id；模型/参数/评价协议变化必须改变 task_id；损坏快照与重复任务明确失败；both 模式不混合；all-miss 推断字段 null+原因；短时域接触不能判完整成功；安全拒绝后物理步数为 0；触后和恢复段不享受终段豁免；候选 kp/kd 与实际模型一致；torque 随挥开启分支实际执行非零步数。

未来 CLI 约定（**尚未实现，请勿当作现有命令执行**）：

```text
python scripts/exp/run_r1d_tuning.py --config configs/experiments/r1d_v12_tuning.yaml --phase P0 --workers 4 --dry-run
python scripts/exp/run_r1d_tuning.py --config configs/experiments/r1d_v12_tuning.yaml --phase P0 --workers 4 --resume
python scripts/extract/extract_r1d_tuning_results.py --catalog <campaign_root>/catalog.json
```

`--dry-run` 仅生成/验证本阶段清单，不启动仿真；P4 dry-run 也必须检查锁参记录，不能提前物化盲测输入。P1–P4 不自动连跑，阶段决策文件记录候选和原因，下一阶段检查其摘要。

## 11. 交付口径

最终报告同时交付所有任务/尝试清单、实际有效配置、双环境物理快照、原始接触与恢复时序、失败原因、分模式表格、成对比较、吞吐和总算力用量。全 miss 时可以报告观察到的 0 成功及二项区间，但退化的成对提升/DiD 推断必须 `null + unavailable_reason=all_miss`，不使用 `[0,0]` 暗示已证明无差异。

如果本轮只提高接触率而完整安全成功率没有改善，结论就是“接触优化有效，恢复控制仍不足”。这时应另开控制策略设计，而不是继续扩大限值或用缩短 horizon 换取通过。设计阶段的验收是文档和接口一致；实际实验结论只能在执行后填写。
