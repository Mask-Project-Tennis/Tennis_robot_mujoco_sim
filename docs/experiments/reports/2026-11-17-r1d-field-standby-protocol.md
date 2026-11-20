# R1-A7-D 现场待命规程：4.9 只读核对 / 4.10 低速跟踪（Task 4.8 交付）

> **状态声明**：现场窗口未定；本规程所引用的全部实机数值（幅度/频率/时长/
> 阈值/槽位策略/校准/限值）**待 G4 冻结**，当前均为 null 或测试专用占位值；
> **本规程不构成实机授权**——任何连接、上电、接管、运动动作须经现场人员
> 与主代理在本规程之上逐项确认后另行授权。

- **日期**：2026-11-17（Task 4.8 软件交付时点编写；I7 裁定：本轮现场任务
  不派发，由本片交付待命规程）
- **适用对象**：4.9（部署机与实物只读核对，现场任务）与 4.10（低速连续
  跟踪与停止验收，现场任务）的现场执行人员
- **工具**：`scripts/tools/test_unitree_r1d.py`（本片交付；命令见 §3/§4）
- **上游依据**：
  - 施工计划 `docs/superpowers/plans/2026-11-16-unitree-stage4-construction.md` Task 4.9/4.10
  - G1 协议报告 `docs/experiments/reports/2026-11-16-r1d-sdk-control-contract.md`（§7 问询清单）
  - 协议 manifest `configs/robots/r1d_protocol_manifest.yaml`（open_questions 节）
  - 硬件配置模板 `configs/robots/r1d_hardware_{left,right}.yaml`（4.2，现场冻结的载体）

---

## 1. 门槛状态与前置条件

| 门槛 | 内容 | 状态（2026-11-17） | 缺失时允许的动作 |
|---|---|---|---|
| G2 离线通过 | 4.2–4.8 故障注入与 RM65 回归 | 软件已交付（本片完成后由主代理汇总裁定） | 只准备本规程，不上机 |
| G3 实机只读 | 4.9 记录：实物/固件/状态/映射/零位 | **未开始**（现场窗口未定） | — |
| G4 运动前冻结 | 现场人员确认的全机保持/接管/停止方案与数值阈值 | **未开始**（清单见 §2） | 只读采集与报告分析；**禁止 tracking** |

**4.9（只读）前置**：G2 通过 + 部署主机就绪（Python/SDK/IDL 与 G1 一致，
`unitree_sdk2py@814556d1` + cyclonedds，部署机按官方 Ubuntu 流程安装锁定
版本）+ 现场人员在位。

**4.10（tracking）前置**：G3 通过 + **G4 全部确认项冻结**（§2 逐项）+
现场人员明确授权运动范围 + 急停通道（硬件急停按钮）有人在位。

## 2. G4 待确认清单（运动前必须冻结；现状全部未确认）

以下各项现状为 null/未确认（引用 manifest `open_questions` 与 G1 报告 §7
问询项编号 Q1–Q11），**禁止用示例值、仿真值或猜测值代填**；G4 未全部
冻结前，工具的 tracking 模式即使运行也只会得到如实的失败结论：

1. **全机非目标槽位保持策略**（Q3）：单臂接管时对侧臂/头/腰/升降/底盘
   的正确保持策略——官方示例的"q=0+kp 零位保持 + 2s 插值"是否即厂商
   意图。冻结载体：`--slot-policies` YAML（逐槽 mode/q/dq/kp/kd/tau）。
2. **失联/停发行为**（Q2）：lowcmd 停发或断网后各部件行为、有无接收端
   看门狗及其阈值——影响停止语义解释，未确认前"停止发布≠已停止"。
3. **控制权释放恢复**（Q4）：ReleaseMode 后机载服务如何恢复（SelectMode
   重选哪个模式名）。
4. **mode_machine 枚举值**（Q1）：R1-A7-D 正确回读值（4.9 只读核对时
   实测记录，冻结后写入 expected_mode_machine）。
5. **实机数值冻结**（G4 现场确认，逐项写入硬件配置 YAML）：
   - `calibration`：joint_signs / joint_offsets_rad（逐关节实测方向与零位）
   - `limits`：q_lower/q_upper/dq_limit/tau_limit/kp/kd（逐关节）
   - `timing`：control_period_s / state_timeout_s / diagnostic_duration_s
   - tracking 参数：选中关节、amplitude_rad、frequency_hz、
     stop_confirm_dq_threshold；以及 `--acquire-timeout-s`
   - tracking 验收阈值：`--tracking-error-limit-rad` 与
     `--stop-latency-limit-s`（评审 2026-11-18 R4：两项均为 tracking
     运行必填；缺失时工具退出码 2，即使绕过 CLI 直接构造配置，诊断在
     数据采集后也会判 fail 并注明"仅完成数据采集，不构成验收通过"——
     无阈值的正常完成不得记 pass）
   - A·π·f ≤ 选中关节 dq_limit（工具会拒绝超速波形，现场预先核算）
6. **部署主机**（Q10）：外置计算单元（192.168.123.164）还是外部 123 网段
   主机；OS/架构/网卡名（写入 `communication.network_interface`）。
7. **500 Hz 可达成性**（Q9）：Python + 部署主机上实际可达成周期——
   read_only 阶段用发布/接收统计实测后再定 control_period_s，不预承诺。
8. **实物铭牌/固件版本**（Q11）：确认 R1-A7-D 智能版 vs 旗舰版。

## 3. 4.9 只读核对规程（每步命令 + 记录物）

工具语义：read_only 模式**不 activate、不 send**——驱动层保证
publish/acquire 全 0（离线测试 `test_read_only_publishes_and_acquires_zero`
断言）；任何实际发布记录出现即诊断立即失败。

**步骤**（每步的记录物都写入 `results/unitree_stage4/<run_id>/`，
`<run_id>` 建议格式 `ro_<side>_<YYYYMMDD_HHMM>`）：

1. **环境核对**（记录物：`env_check.txt`）
   - 部署机 Python 版本、`pip show unitree_sdk2py cyclonedds` 输出、
     `git rev-parse HEAD`（工具会自动把代码 SHA 写进产物 JSON）。
   - 与 G1 冻结版本（SDK `814556d1`、cyclonedds 0.10.2）逐项对照；
     不一致即停止，只记录，不尝试运行。
2. **实物核对**（记录物：`machine_checklist.md`，照片若干）
   - 铭牌/序列号/固件版本 → 对应 Q11；侧别与安装姿态（升降位置）；
     物理急停按钮位置确认。
3. **硬件配置准备**（记录物：冻结版 `r1d_hardware_<side>.yaml` 副本 +
   其 sha256——工具自动写入产物 JSON 的 provenance）
   - `communication.network_interface` = 现场确认网卡名；
   - `layout.layout_verified`：首轮只读采集允许 `false`（§3.1 勘误，
     评审 2026-11-18：只读是逐关节核对的数据来源，read_only 档不强制
     verified——否则首轮原始采集无可执行路径）；逐关节核对（第 5 步）
     完成后置 `true`。**motion 档强制 `true`**：tracking 预检遇 false
     直接拒绝接管/运动。
   - `timing.state_timeout_s` = 现场冻结的状态新鲜度超时（I-07，评审
     2026-11-17：read_only 档校验现强制该字段非 null，缺失/非正数在
     配置加载即失败——connect 门槛依赖新鲜状态判据）。首轮做法：先
     观察状态流节奏（秒级即可），按实测帧间隔的保守倍数现场冻结并
     记录依据；第 6 步统计后复核收紧。禁止沿用 null 或照抄示例值。
   - `timing.diagnostic_duration_s` = 现场选定的采集时长（如 30 s）；
   - calibration/limits 保持 null（read_only 档允许；模型序列在产物中
     为空，完整 35 槽 SDK 侧原始帧仍完整记录）。
4. **执行只读采集**（记录物：`<run_id>/` 目录全套产物）
   ```bash
   python scripts/tools/test_unitree_r1d.py \
     --robot-config configs/robots/r1d_left.yaml \
     --hardware-config configs/robots/r1d_hardware_left.yaml \
     --output-dir results/unitree_stage4/<run_id> \
     --acquire-timeout-s <现场值> \
     --mode read_only \
     --sample-period-s <现场值>
   ```
   - 退出码 0=pass / 1=fail（fail 原因在 stdout 与 JSON `reasons`）。
   - 产物：`diagnostic_summary.json`（verdict/指标/溯源）+
     `diagnostic_publish_records.csv`（read_only 下应为空表）+
     `diagnostic_samples.csv`（完整 35 槽状态流）+ `diagnostic_send_log.csv`
     （应为空表）+ `diagnostic_raw.npz`。
5. **逐关节映射/零位核对**（记录物：`mapping_check.md`）
   - 从 `diagnostic_samples.csv` 的 `full_q_*` 列读各槽位静止读数；
   - 现场人员手动小幅挪动单一关节（断电/示教模式下），对照槽位读数变化
     确认"关节名→槽位→方向"；无法从静态状态证明方向时，列为 4.10 运动
     前受控核验项，不猜。
   - 核对完成后回填 `layout.layout_verified: true` 与校准初值，重跑第 4
     步交叉验证。
6. **新鲜度/频率统计**（记录物：JSON `publish_metrics` 段为
   unavailable——只读无发布；状态接收节奏从 samples CSV 的
   `received_at` 列人工/脚本统计，写入 `frequency_check.md`）
   - 与 Q9（500 Hz 可达成性）对照，为 G4 冻结 control_period_s 提供实测。

**4.9 停止条件**（出现即断开退出，保留已采集产物）：
- 任何非预期 phase 变化或出现实际发布记录（工具自动判 fail）；
- 观察到整机异常响动/冒烟/异位——按 §6 紧急手段处置；
- 状态流中断/数据不足（工具自动判 fail，原因见 JSON）；
- Ctrl+C 中断（工具生成 aborted 结论并清理连接）。

## 4. 4.10 低速跟踪规程（前置：G3+G4 全部冻结）

**逐级放行**：每次只操作一个显式选中关节；首关节建议选远离身体的腕
关节；幅度从最小冻结值开始，验证通过后才允许下一档/下一关节（每档新
`<run_id>` = `tr_<side>_<joint>_<YYYYMMDD_HHMM>`，修改配置即换新
run_id 重新验收，不得试验后放宽阈值追认通过）。

**执行命令**（全部数值来自 G4 冻结配置，现场不得临时改动）：

```bash
python scripts/tools/test_unitree_r1d.py \
  --robot-config configs/robots/r1d_left.yaml \
  --hardware-config configs/robots/r1d_hardware_left.yaml \
  --output-dir results/unitree_stage4/<run_id> \
  --acquire-timeout-s <G4 冻结值> \
  --mode tracking --enable-motion \
  --selected-joint <G4 冻结关节名> \
  --amplitude-rad <G4 冻结值> \
  --frequency-hz <G4 冻结值> \
  --stop-confirm-dq-threshold <G4 冻结值> \
  --tracking-error-limit-rad <G4 冻结值> \
  --stop-latency-limit-s <G4 冻结值> \
  --slot-policies <G4 冻结的槽位策略 YAML>
```

- `--enable-motion` 为运动双重闸门（缺它退出码 2）；
- 未提供 `--slot-policies` 时驱动 activate 将拒绝接管，工具如实生成
  失败结论——这是设计行为（无全机策略不运动）；
- 波形为升余弦（从实测姿态零速度出发、整周期后返回、单关节），隐含
  峰值速度 A·π·f，超出 dq_limit 会被驱动步长校验显式拒绝；
- 采集时长非控制周期整倍数时，回程按**受限速的多拍收敛**补发（每拍
  步长 ≤ 波形内已用最大步长，评审 2026-11-18 R6：不再单拍跳回起点被
  步长校验拒绝→FAULT）；回程拍数有上界并计入发布间隔统计；
- 记录物：`<run_id>/` 全套产物（JSON 含跟踪误差 RMS/max、publish 间隔
  p50/p95/p99/max 与超期次数、停止延迟双口径：软件检测/反馈达阈值、
  发布原始目标与时刻、send 接受时刻、完整非目标状态）。
  超期判据为**间隔 > 1.5 × 冻结期限**（半拍容差：驱动按"上一帧发布
  返回后再等一个完整周期"排下一拍，间隔恒含发布/唤醒开销；错过半拍
  以上才算漏拍，真实漏拍 ≈ 2× 期限必计入。原始 p50/p95/p99/max 全量
  落盘，判定可复核）；

**验收判定**（JSON `verdict`）：
- pass = 波形完成 + 全关节误差 ≤ 冻结阈值 + 停止确认（stop_confirmed
  =true）+ 发布零超期 + 非目标槽位无超限偏移（人工从 full_q 列核对）；
- 任何一项不满足 → fail，原因在 `reasons`；**无样本的指标标
  unavailable，不得记 0 通过**；
- 需要验证的正常停止/用户停止（Ctrl+C→aborted 结论）/命令超时路径：
  先离线注入已覆盖（`tests/test_r1d_diagnostics.py`），现场是否施加
  故障（拔网/释放整机模式）由现场人员裁定，**禁止 agent 自动执行**；
  未实测的故障路径在报告中标"未实机验证"。

**4.10 停止条件**（立即中止并保留产物）：
- 跟踪误差接近冻结阈值的 80%；
- 任何非目标部件（对侧臂/头/腰/升降）出现可见移动或 full_q 列超限；
- 停止未确认（工具自动判 fail 且不退权——此时按 §6 处置）；
- 发布频率统计持续超期（阶段失败，记录 Python/主机调度数据）。

## 5. 记录物存放路径约定

```
results/unitree_stage4/
├── ro_<side>_<时间戳>/          # 4.9 只读采集（全套产物 + env_check.txt 等）
├── tr_<side>_<joint>_<时间戳>/  # 4.10 每档一次（全套产物）
└── field_notes/                 # machine_checklist.md / mapping_check.md /
                                 # frequency_check.md / G4 确认记录
```

最终验收报告（Task 4.9/4.10 正式报告文件）由现场执行后的对应任务交付，
本规程不代填。

## 6. 紧急手段（按序使用）

1. **工具层 Ctrl+C**：诊断生成 aborted 结论并走 disconnect 清理；
2. **软件 slow_stop/急停**：驱动层缓停/锁存（软件层尽力发布停止帧；
   失链时仅报告尝试，**不承诺硬件已停**）；
3. **硬件急停按钮**：任何软件手段未在停止条件内确认停止时，立即按下
   物理急停（位置在 4.9 第 2 步核对时确认）；
4. 事后：保留全部产物与现场记录，停止相关事件按 JSON `stop` 段的
   双口径时刻线复盘。

## 7. 未确认项处理纪律

- 任何 §2 清单项未冻结：对应能力保持关闭（tracking 预检失败是正确
  行为），不得以"先跑一次看看"绕过；
- 工具输出 fail/aborted：如实上报，不得删除产物或改写结论重跑追认；
- 与 URDF/官方指南冲突的实机限制：逐项记录现场裁定依据，写入
  `field_notes/`，由主代理裁定后再改配置；
- provisional 数值（仿真/示例来源）在任何产物中不得冒充实机限值
  （未知一律 null）。

## 8. 本规程未覆盖事项

- 升降调节（阶段 6）、视觉（阶段 7）、在线击球（后续阶段）不在本规程；
- 阶段 3 轨迹 NPZ 不被本工具接收（阶段 5 才做轨迹重演，防提前实现）；
- 本规程的更新权属主代理；现场发现规程与实物不符时，先停止执行并
  上报，不得现场即改规程。
