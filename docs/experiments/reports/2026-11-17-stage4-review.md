# 阶段 4 独立评审报告：R1-A7-D 整机接口下的单臂驱动（软件层）

- **日期**：2026-11-17
- **评审结论**：**Changes requested**——实现完成、验收评审开放（**0 Critical / 10 Important / 36 Minor**，去重约 30）。建议阶段 4 状态表述为「**软件实现完成；10 项 Important 关闭前不标记验收通过；G3/G4/G5 未启动，待实机验收**」。
- **审查范围**：`unitree/stage4-real` @ `5666959`，区间 `0be5be7..5666959`，**31 files / +14130 / −6**（27 新增 + 4 索引修改；既有 `.py` 零改动）
- **工作树**：`F:\刀客塔\研究\网球机器人\工程\Code\Sim\.worktrees\stage4-real`（评审期间 clean @ `5666959`）
- **审查方式**：5 名只读 reviewer 并行分工 + Lead 独立复跑验证；共享任务板 `task-1..task-5` 全部 completed
- **上游文档**：阶段 4 报告 `2026-11-17-unitree-stage4.md`、基线报告 `2026-11-16-unitree-stage4-baseline.md`、施工计划 `docs/superpowers/plans/2026-11-16-unitree-stage4-construction.md`、`configs/robots/r1d_protocol_manifest.yaml`
- **评审纪律**：本次评审**只读**，未修改任何代码/配置/文档；本报告是评审过程的第一处写入

---

## 一、审查范围与方法

### 1.1 审查对象

| 层 | 文件 | 评价重点 |
|----|------|---------|
| 类型与映射 | `src/real/r1d_types.py`、`src/real/r1d_mapping.py` | 帧不可变性、35 槽语义、按名寻址、sign·q+offset 往返 |
| 配置 | `src/real/r1d_config.py`、`configs/robots/r1d_hardware_{left,right}.yaml`、`configs/robots/r1d_protocol_manifest.yaml` | 只读/运动门槛、null 语义、硬件载体 |
| 命令策略 | `src/real/r1d_command_policy.py` | 全机 35 槽策略、非目标槽保持、停止帧 |
| 状态缓存 | `src/real/r1d_state_buffer.py` | 会话隔离、序号单调、快照不可变 |
| 传输 | `src/real/r1d_transport.py` | 懒加载、只读 open、真实 IDL/CRC、close 幂等 |
| 驱动接口 | `src/real/unitree_r1d_interface.py` | 生命周期、单一发送器、五类看门狗、停止/竞态 |
| 工厂与诊断 | `src/real/r1d_factory.py`、`src/real/r1d_diagnostics.py`、`scripts/tools/test_unitree_r1d.py` | 注入面、指标真实性、CLI 双闸门 |
| 测试与索引 | `tests/test_r1d_*.py`、`tests/test_unitree_r1d_*.py`、`AGENTS.md`、`docs/README.md`、`src/README.md`、`scripts/README.md` | 测试判别力、计数可复现、规范与覆盖 |

### 1.2 团队分工

| 任务 | Reviewer | 范围 | 任务板 |
|------|----------|------|--------|
| 协议/传输 | `rev-protocol` | `r1d_types`、`r1d_transport`、manifest、序列化/传输测试 | task-1 |
| 配置/映射/策略 | `rev-mapping` | `r1d_config`、`r1d_mapping`、`r1d_command_policy`、硬件 YAML、三类测试 | task-2 |
| 状态/驱动接口 | `rev-state-interface` | `r1d_state_buffer`、`unitree_r1d_interface`、并发/停止测试 | task-3 |
| 工厂/诊断/CLI | `rev-factory-diag` | `r1d_factory`、`r1d_diagnostics`、`test_unitree_r1d.py`、三类测试 | task-4 |
| 测试/规范/覆盖 | `rev-tests-compliance` | 全部 10 个测试文件、索引与报告、计数复现 | task-5 |

每名 reviewer 输出分级 findings（`file:line` + 证据 + 影响 + 建议）、独立裁决、强项、无法验证项与命令记录；Lead 另行复跑全量回归、计数、ruff、跳过原因、SDK 布局探针与三个并发探针。

---

## 二、总裁决

### 2.1 分级统计

| Reviewer | 裁决 | Critical | Important | Minor | 要点 |
|----------|------|----------|-----------|-------|------|
| rev-protocol | Approved with issues | 0 | 1 | 7 | fake 覆盖与 CRC 成立；wire 布局证据不成立 |
| rev-mapping | Approved with issues | 0 | 1 | 4 | G2 无阻断；G4 前须关非目标臂默认策略 |
| rev-state-interface | **Changes requested** | 0 | 3 | 8 | 3 个并发/会话隔离缺陷可确定性复现 |
| rev-factory-diag | Approved with issues | 0 | 3 | 8 | 4.7 通过；4.8 有 3 个规程/接线缺口 |
| rev-tests-compliance | Approved with issues | 0 | 2 | 9 | 计数全部复现；2 项证据质量/时效问题 |
| **合计** | **Changes requested** | **0** | **10** | **36** | 去重约 30（硬编码常量等跨 reviewer 重复） |

### 2.2 总体判断

- **可以接受的部分**：分层清晰；懒加载与构造零副作用；错误先于传输；失败路径不伪造「已停止」；无「停止后仍发运动命令」路径；官方 CRC 与独立参考实现互验；测试计数、ruff、跳过原因全部可复现。
- **不能接受的声明**：报告宣称的「4.11 独立终审 0 Important」「4.6/4.4 PASS」「G2 真实 IDL 序列化对照冻结布局」**均不能按原文接受**。其中 3 项并发缺陷（I-01/I-02/I-03）由探针确定性复现，直接否定 4.6/4.4 的 PASS 结论。
- **风险等级**：所有 Important 均**不产生危险运动命令**（无控制权抢占、无停止后发布、无越界槽位下发），但会使命中「停止确认」与「状态会话」语义失真；I-05 在 G4 启用运动后可能造成非目标臂阶跃运动，必须在 G4 前关闭。

---

## 三、10 项 Important（附证据与修复要求）

> 编号按门禁修复时点排列；「Rev」列标注首次发现方。所有行号为评审时 `5666959` 工作树内容。

### I-01 并发重复缓停窗口误锁 FAULT 并覆盖真实停止确认（Rev-C，Lead 复现）

- **位置**：`src/real/unitree_r1d_interface.py:799-802`、`:964-971`、`:1020-1030`、`:679-696`、`:1172-1176`
- **机制**：`_decide` 取走 `_stop_request` 后立即返回（`:799-802`），此时 phase 仍为 ACTIVE；`_run_stop_sequence` 仅当 phase==ACTIVE 才置 STOPPING（`:967-968`），且无 else 早退。窗口内第二次 `slow_stop()` 会重新登记请求并 `_stop_done.clear()`（`:679-696`），第二段停止序列在 READ_ONLY 下 alive 复查要求 phase==STOPPING（`:1020-1021`）→ 失败 → `_finish_stop(confirmed=False, final_fault=True)`，把第一段的 `stop_confirmed=True` 覆盖为 False 并锁存 FAULT（`:1022-1030`、`:1172-1176`）。
- **证据（探针 A）**：CONTROL 组 `READ_ONLY / stop_confirmed=True`；RACE 组 `FAULT / False`、`fault_reason=用户缓停请求`；停止序列中止：1 个 stop 帧、0 个 active 帧、发布前生命周期/世代复查失败（停止未确认，不声称已停）。窗口真实存在于 `_decide` 返回与 `_run_stop_sequence` 取锁之间，探针用 50 ms 注入睡眠稳定命中。
- **影响**：违反 4.6「重复 stop 幂等」验收；遥测失真（真实已停却报 FAULT/未确认），需重建接口对象；不产生运动命令。
- **修复方向**：在取走 `_stop_request` 的同一锁内把 phase 置 STOPPING（或对已进入停止流程的重复请求直接返回当前停止状态而非重新登记）；补覆盖真实窗口的并发测试。
- **关闭时点**：**G4 启用运动前**（且应回写 4.6 验收结论）。

### I-02 状态缓存会话隔离存在锁外检查竞态（Rev-C，Lead 复现）

- **位置**：`src/real/r1d_state_buffer.py:189-191`、`:214-230`、`:251-262`、`:277`
- **机制**：`push` 在锁外比较 `frame.session_id`（`:189`），持锁写入时不再复查（`:214-230`）；`latest` 同样在锁外比较会话（`:251`），锁内取 `_frame`（`:261-262`），返回前不校验 `frame.session_id == session_id`（`:277`）。真实路径中状态回调在锁外 push（`src/real/unitree_r1d_interface.py:739-743`），期间 connect/disconnect 可执行 reset（`:391-403`）。
- **证据（探针 B）**：RACE-1：旧会话帧被新会话缓存接受（`accepted=True`）、`latest(新会话)` 返回旧 `session_id=1`；RACE-2：`latest(3)` 返回 `session_id=4`。对照组无交错时行为正确。
- **影响**：旧会话状态可污染新会话的验证/检查；违反 4.4「快照不可变、会话隔离」。不产生危险运动命令。
- **修复方向**：会话比较与写入/读取置于同一锁内；`latest` 返回前校验帧会话；必要时把 session 放入受锁保护的 generation 字段。
- **关闭时点**：**G3 只读准备前**（只读状态的正确性依赖该路径）。

### I-03 未消费异步故障跨会话泄漏，重连即 FAULT（Rev-C，Lead 复现）

- **位置**：`src/real/unitree_r1d_interface.py:391-403`、`:433-445`、`:803-806`、`:489-498`
- **机制**：`connect()`/`disconnect()` 重置 buffer/phase/fault_reason 等，但未清 `_async_fault_reason`；`_decide` 在 READ_ONLY 也消费它（`:803-806`）→ 下一段停止序列 alive 复查失败 → FAULT；`activate` 被「故障锁存」拒绝。
- **证据（探针 C）**：LEAK 组重连后 `stale_async_reason_after_disconnect=True`、`phase=FAULT`、`activate` 抛「故障锁存后原对象不得再次 activate」、零发布；CONTROL 组 `READ_ONLY / activate 正常`。
- **影响**：一次未消费的旧会话故障永久污染新会话，违反 4.6 生命周期与会话隔离；不产生危险运动命令。
- **修复方向**：connect/disconnect 清除异步故障原因（或会话化保存）；`_decide` 仅在 ACTIVE 消费异步故障。
- **关闭时点**：**G4 前**。

### I-04 原生序列化「总长 == 1004/2092」属巧合，冻结布局证据不成立（Rev-A，Lead 独立复现）

- **位置**：`tests/test_r1d_sdk_serialization.py:250-266`（注释 `:261` 称「含 4 B CDR 头总长一致」）
- **证据（Lead 独立探针）**：

| 断言 | 实测 |
|------|------|
| `LowCmd_.serialize()` 总长 | 1004 B = 4 B CDR-LE 头 `00 01 00 00` + **body 1000 B** |
| body 与官方参考布局 | `body == reference[:4] + reference[8:]` 为 True；`body == reference[4:]` 为 False（丢掉 `motor_cmd[0].mode` + 3 B padding） |
| 用官方布局字节反序列化 | `mode_machine=7` 正确，但 `m0.q=0.0`（期望 0.25）、`m21.q=1.401e-45` 垃圾 → **本机 cyclonedds 11.1.0 编解码与冻结布局不互操作** |
| `LowState_.serialize()` | 总长 2092 B / body 2088 B |

- **影响**：本机通过是「总长巧合」而非布局正确；部署机 `cyclonedds 0.10.2` 若序列化出合规 1004 B body，总长应为 1008 B，现有硬编码断言会失败。**代码本身无缺陷**（`r1d_transport._build_lowcmd` 与官方 CRC 正确），但 4.5「真实 IDL 序列化对照冻结布局」证据失效。
- **修复方向**：测试改为 **body 字节对照**（`serialize()[4:]` 与按冻结布局裁剪的参考体逐字节一致，或字段级反序列化对照），总长断言降级为附加检查；本机明确标注 R4（11.1.0 偏差）、部署机复跑；同步修正报告、R4 与 manifest 中「1004 为 serialize 总长」的措辞（1004 是 CRC 结构像长度）。
- **关闭时点**：G2 证据勘误（**即修**）；部署机复跑属 G3 前。

### I-05 非目标对侧臂/头默认下发 manifest 示例增益 + q=0，停止帧不保持实测、无 2 s 插值（Rev-B）

- **位置**：`src/real/r1d_command_policy.py:265-289`（构造 `manifest_defaults`：对侧臂 7 槽 q=0、`kp=_OTHER_ARM_HOLD_KP`、`kd=_OTHER_ARM_HOLD_KD`；头 2 槽 q=0、kp/kd 常量）、`:350-378`（`build_stop` 用 `holds=_resolve_non_target_holds`，仅目标槽 `q_stop=state.q[idx]` 取实测）、`:95-99`/`:123-124`（docstring 自述 2 s 插值未实现）、`:602-609`（首帧锚定/步长约束只覆盖目标 7 槽）
- **影响**：G4 补齐 19 个 null 槽后即可通过预检，届时**首帧与每次停止都会以示例增益把非零姿态的对侧臂/头阶跃拉向 0**；docstring「manifest 示例值不得冒充实机核验值」与默认回退自相矛盾。
- **修复方向**：G4 启用运动前显式冻结全部 28 个非目标槽策略（保持实测、限幅、或实现文档声称的斜坡/插值）；若暂不实现，须在报告与 manifest 中显式登记为 G4 关闭项并移除「2 s 插值」表述。
- **关闭时点**：**G4 前**（运动安全前提）。

### I-06 跟踪误差/停止延迟冻结阈值不可注入，规程 pass 判据无法实现（Rev-D）

- **位置**：`src/real/r1d_diagnostics.py:197-198`（字段存在）、`scripts/tools/test_unitree_r1d.py:309-321`（构造 tracking 配置不传）、硬件 schema/CLI 无对应参数；待命规程 `2026-11-17-r1d-field-standby-protocol.md:154-155`（pass 判据）
- **影响**：G5 自动验收无法由工具给出「≤冻结阈值」结论，只能人工比对；`R1DDiagnosticConfig` 两个阈值零测试。
- **修复方向**：CLI/配置补齐两个阈值参数并接线到 `run_tracking`，报告输出实际值与判据；或明确改规程为人工比对并去掉「自动」措辞。
- **关闭时点**：**G5 前**。

### I-07 按规程准备的只读配置必然 connect 失败（Rev-D，Lead 源码复核）

- **位置**：`src/real/r1d_config.py:550-560`（read_only 门槛只查 network + layout_verified，不查 timing）、`src/real/unitree_r1d_interface.py:380-384`（`_buffer is None` 时直接 return False）、`:267-273`（buffer 仅在 `state_timeout_s` 非 None 时构造）、待命规程 `:81-88`（只读清单未列 `state_timeout_s`；模板 timing 全 null）
- **证据**：按规程填写的只读配置 `validate(read_only)` 通过，但 `connect` 返回 False、phase CLOSED、零样本；测试夹具 `tests/test_unitree_r1d_tool.py:231-233` 填了 `state_timeout_s` 从而掩盖该路径。
- **影响**：G3 首日按规程操作必定连不上，与「只读准备已完成」的声明冲突；不产生运动命令，但会阻塞实机进程。
- **修复方向**：read_only 预检增加 timing/state_timeout_s 必需项（或 connect 允许无 buffer 的只读模式）；规程只读清单列出该字段；修正夹具以覆盖真实规程配置。
- **关闭时点**：**G3 前**。

### I-08 `expected_mode_machine` 冻结无交付通路（Rev-D）

- **位置**：`src/real/r1d_factory.py:52-56`（参数存在）、`scripts/tools/test_unitree_r1d.py:382-394`（CLI 不传）、配置/CLI/manifest 无对应参数（manifest `mode_machine_validation` 为 null）；`unitree_r1d_interface` 仅记录不判等
- **影响**：G4 冻结机型期望值后，G5 的机型校验仍只「记录后放行」，冻结形同虚设。
- **修复方向**：G4 前接线上报通路（CLI/YAML → factory → interface 判等），或显式删除该冻结项并修改验收表述。
- **关闭时点**：**G4 前裁决**。

### I-09 4 个「import 纯净性」测试为空操作（Rev-E）

- **位置**：`tests/test_r1d_config.py:133-155`、`tests/test_r1d_mapping.py:161-`、`tests/test_r1d_command_policy.py:245-265`、`tests/test_unitree_r1d_interface.py:406-419`
- **机制**：被测模块在文件顶层已导入（如 `tests/test_r1d_config.py:18`），测试体 `sys.modules.pop` 后再 import 命中缓存，断言不可能失败（测试体 import 见 `:146`）。
- **影响**：报告 4.2 该复选框证据失效；当前行为经独立子进程验证确实纯净（9 模块 `leaked=[]`、`threads_delta=0`），属**测试证据质量问题**而非实现缺陷。
- **修复方向**：改为子进程断言（参照 `tests/test_r1d_transport.py:352-377`）；报告 4.2 引用同步更新。
- **关闭时点**：**即修**（证据质量）。

### I-10 报告状态行过期且自相矛盾（Rev-E）

- **位置**：`docs/experiments/reports/2026-11-17-unitree-stage4.md:5`、`:200`
- **问题**：写 `@fa340c7 / 10 commits / 26 files「全为新增」`；实际 `@5666959 / 11 commits / 31 files`（27 新增 + 4 修改：`AGENTS.md`、`docs/README.md`、`scripts/README.md`、`src/README.md`），§十「未触碰既有 src/tests/scripts 文件」与 §九/§六 的修改清单矛盾。
- **影响**：实质结论（既有 `.py` 零改动）仍真，但文档不可作为提交链证据；与 4.11 终审记录相互引用时会放大失真。
- **修复方向**：定稿前更正状态行与 §十 措辞（区分「既有 `.py` 未改」与「4 个索引 md 修改」）。
- **关闭时点**：**定稿前**（即修）。

---

## 四、Minor 清单（36 项，去重约 30）

> 编号前缀为 reviewer：A=rev-protocol、B=rev-mapping、C=rev-state-interface、D=rev-factory-diag、E=rev-tests-compliance。均不阻断 G2，但应在对应门禁前清理或登记。

| # | 位置 | 问题 | 建议 |
|---|------|------|------|
| A-M1 | `src/real/r1d_transport.py:318-333`、`src/real/r1d_factory.py:99-103` | 实机面向示例默认（switcher 5.0 / release 1.0 / state_queue 10）硬编码 | G4 前由硬件 YAML 冻结替换 |
| A-M2 | `src/real/r1d_transport.py:453` | `if result is False` 死代码（SDK 返回非 bool） | 删除或改 `not result` 并测试 |
| A-M3 | `src/real/r1d_transport.py:501-505` vs `:309-310` | close 后旧回调绝对化 docstring 与实现不完全一致 | 修正措辞或实现 |
| A-M4 | `src/real/r1d_transport.py:224-229` | `motor_state` 为 None / 长度≠35 无测试 | 补拒绝路径测试 |
| A-M5 | manifest ↔ 代码 | 槽位/话题无自动对照测试 | 补 parametrized 对照测试 |
| A-M6 | `src/real/r1d_types.py:89-127` | bool/负 sequence 未测 | 补边界测试 |
| A-M7 | `src/real/r1d_transport.py:534-563` | acquire_control 建 publisher 失败不恢复原模式（manifest `release.mechanism` 为 null） | 恢复或文档化 |
| B-M1 | `src/real/r1d_command_policy.py:414-471`、`:200-203` | `build_active/build_stop` 不复查 verified，docstring 措辞过强（activate 门槛 `unitree_r1d_interface.py:502` 守住） | 调整措辞或补校验 |
| B-M2 | `src/real/r1d_config.py:308-310,498-589` | 硬件配置与 manifest 话题/槽位无自动对照 | 加交叉校验/测试 |
| B-M3 | `tests/test_r1d_mapping.py:140-153,420-434` | 按名寻址未被测试真正判别（仅顺序槽） | 加乱序/改名测试 |
| B-M4 | `src/real/r1d_command_policy.py:142-149` | 显式 mode 允许 uint8 0..255 | 限 0/1 或文档说明 |
| C-M1 | `src/real/unitree_r1d_interface.py:404-425` | connect 成功路径不复查 phase/generation | 加锁内复查 |
| C-M2 | `src/real/unitree_r1d_interface.py:780-789` | READ_ONLY 发送线程异常静默 | 记录/遥测 |
| C-M3 | `src/real/unitree_r1d_interface.py:433-445` | disconnect 在 ACTIVE 下不停机不退权 | 文档/实现约束 |
| C-M4 | `src/real/unitree_r1d_interface.py:541-567` | overtaken 分支死代码 | 删除或测试 |
| C-M5 | `tests/test_unitree_r1d_interface.py:880-903,1380-1420`、`tests/test_r1d_state_buffer.py:266-296` | 停止/会话测试缺真实竞态窗口判别力 | 补注入/同步测试 |
| C-M6 | `src/real/unitree_r1d_interface.py:854,484` | 测试/实现窥视私有 `_stop_done` 等 | 提供只读查询接口 |
| C-M7 | `src/real/unitree_r1d_interface.py` | 注释/遥测语义失真（如停止原因） | 修正文案 |
| C-M8 | `src/real/unitree_r1d_interface.py:489-498` | FAULT 终态仅会话内，重建后无痕 | 会话外持久化或说明 |
| D-M1 | `src/real/r1d_diagnostics.py:418-437` | `_git_commit_sha` 未指定 `cwd`，依赖调用目录 | 显式 `cwd=仓库根` |
| D-M2 | `src/real/r1d_diagnostics.py:818-853` | 停止延迟双口径（计划/实测未注明） | 遥测字段区分 |
| D-M3 | `src/real/r1d_diagnostics.py:543,558` | 样本时刻早于状态读取 | 时间戳对齐 |
| D-M4 | `src/real/r1d_diagnostics.py:702-711` | 只读收尾窗口存在遗漏风险 | 补收尾检查 |
| D-M5 | `src/real/r1d_diagnostics.py:857-881` | cleanup 停止未入库 | 记录 |
| D-M6 | 待命规程 `:57` | A·π·f 预检缺失 | 补清单 |
| D-M7 | `scripts/tools/test_unitree_r1d.py` | `--acquire-timeout-s` 为第四强制参数（非缺陷） | 文档说明 |
| D-M8 | `tests/test_r1d_diagnostics.py` | 两阈值零测试 + 规程式配置零覆盖 | 补测试 |
| E-M1 | `src/README.md:174` | 类名错写 `R1DWholeBodyCommandPolicy`（实际 `R1DCommandPolicy`） | 修正 |
| E-M2 | `docs/README.md:40-45` | 未列阶段 2/3 报告却称已列 | 补齐或改措辞 |
| E-M3 | 同 A-M1 | 实机面向常量硬编码 | 同 A-M1 |
| E-M4 | `tests/test_r1d_state_buffer.py:409` | 上界恒真断言 | 收紧断言 |
| E-M5 | `tests/test_unitree_r1d_interface.py:1251-1287` | 发送线程异常未收集 | 捕获并断言 |
| E-M6 | 测试套件 | 10 处墙钟 sleep 泵送 | 改条件变量/事件 |
| E-M7 | `tests/test_r1d_config.py:191/193` | `pytest.raises` 无 `match` | 加 `match` |
| E-M8 | `tests/test_r1d_diagnostics.py:898` | `final_phase` 双态接受 | 收紧为唯一期望 |
| E-M9 | 报告 §七 | 已披露遗留复核属实（4.6 overtaken、4.8 降级路径、4.9/4.10） | 保持登记 |

---

## 五、Lead 独立验证证据

全部命令在 worktree 根执行（`PY=C:/Users/Nothing/.conda/envs/mujoco_tennis/python.exe`，Python 3.11.15）。

### 5.1 测试计数与回归

| 项 | 命令 | 结果 |
|----|------|------|
| 全量回归（排除 2 个 C++ ABI 文件） | `pytest tests/ -q --ignore=tests/test_cpp_forward_pass.py --ignore=tests/test_cpp_backward_pass.py` | **1552 passed, 3 skipped**（189.75 s），与报告一致 |
| 阶段 4 测试（11 文件 = 10 个 r1d + `test_robot_interface`） | `pytest <11 files> -q` | **312 passed**，0 skip，与报告一致 |
| 10 个 r1d 文件逐文件 collect | `pytest --collect-only -q` ×10 | 16/54/38/21/31/8/52/12/40/30 = **302**，与报告一致 |
| G0 兼容六文件 | `pytest tests/test_bound_robot_env.py tests/test_multi_robot_mpc.py tests/test_multi_robot_sim_entrypoint.py tests/test_robot_interface.py tests/test_robot_factory.py tests/test_robot_profile_entrypoint.py -q` | **172 passed**（27.66 s），与报告一致 |
| 3 个 skip 亲验 | `pytest tests/test_r1d_closed_loop.py tests/test_actuator_modes.py -q -rs` | 47 passed, 3 skipped：`test_r1d_closed_loop.py:466` 缺 `results/unitree_stage3/rm65_seed42.npz`（git-ignored，设计内条件跳过）；`test_actuator_modes.py:297/329` C++ `iLQR_Core` 未编译（既有） |

### 5.2 Ruff 与 git

- 阶段 4 文件（`src/real/` + `scripts/tools/test_unitree_r1d.py` + 10 个测试）：`All checks passed!`
- 全仓：`Found 711 errors`，与报告「全为既有」一致（逐条抽查无误报新增）
- `git diff --name-status 0be5be7..HEAD`：31 files、+14130/−6、27 A + 4 M；`git diff -- '*.py' 0be5be7..HEAD` 无修改项（既有 `.py` 零改动）；`git status` 空

### 5.3 SDK / 序列化独立探针

- `unitree_sdk2py` 1.0.1 editable 安装，源为 `.superpowers/sdd/2026-11-16-unitree-stage4-construction/sdk-4.5/unitree_sdk2_python`（`.superpowers/` 被主仓 `.gitignore:44` 忽略），git rev **814556d** ✓（符合 R4）。
- 官方 `unitree_sdk2py/utils/crc.py` 格式串 `'<2B2x' + 'B3x5fI'*35 + '5I'`（`#size=1004`）、HGLowState `'<2I2B2xI' + '13fh2x' + 'B3x4f2hf7I'*35 + '40B5I'`（`#size=2092`）；Linux 走 ctypes `lib/crc_amd64.so`，非 Linux 回落纯 Python `_crc_py`。
- 懒加载纯净性：子进程 `import src.real.r1d_transport` → `sdk_loaded=False, cyclone_loaded=False` ✓。
- 布局探针（本次新增，临时文件）：`LowCmd_` ref 1004 / native 1004 / header `00 01 00 00` / `body == ref[:4]+ref[8:]` True / `body == ref[4:]` False；`LowCmd_.deserialize(b'\x00\x01\x00\x00'+ref)` → `m0.q=0.0`、`mode_machine=7`；`LowState_` ref 2092 / native 2092。结论见 I-04。
- 备注：cyclonedds 类无 `__bytes__`，`bytes(LowCmd_())` 报 `cannot convert 'LowCmd_' object to bytes`；测试使用 `.serialize()` 正确，非缺陷。

### 5.4 并发探针复跑

Rev-C 的 3 个探针（`probe_a_dup_stop.py`、`probe_b_buffer_session.py`、`probe_c_async_fault_leak.py`，本机临时目录）由 Lead 独立复跑，结果与其报告一致（见 I-01/I-02/I-03）；探针通过注入 50 ms 睡眠放大窗口，但窗口在未注入时同样存在（代码路径可静态确认）。

---

## 六、对阶段 4 报告与门禁声明的修正

| 原声明 | 评审结论 | 处理 |
|--------|----------|------|
| G2「通过（SDK 序列化条件通过）」 | **保留条件通过，但证据降级**：CRC/参考打包/本机自洽往返成立；「真实 IDL 序列化对照冻结布局」不成立（I-04） | 改写 G2 证据段落；测试改 body 对照；本机标注 R4；部署机 `cyclonedds 0.10.2` 复跑后再升级为「通过」 |
| 「4.6/4.4 PASS」 | **不成立**：I-01（4.6 重复 stop 幂等）、I-02（4.4 会话隔离）、I-03（4.6 生命周期会话泄漏）均可确定性复现 | 4.6/4.4 改回「实现存在、待 I-01/I-02/I-03 关闭后复验」 |
| 「4.8 独立评审 Approved 0C/0I/4M」 | **不成立**：Rev-D 判 0C/**3I**/8M（阈值不可注入、只读配置连不上、mode_machine 无通路） | 4.8 改判；3 项加入门禁清单（I-06/I-07/I-08） |
| 「4.11 独立终审 0 Important」 | **不成立**：本次评审 10 Important，其中 3 项来自 4.6 核心路径 | 4.11 记录勘误；阶段 4 状态改「验收评审开放」 |
| 测试计数 312/302/172/1552+3/skip 原因 | 全部复现 ✓ | 无需修改 |
| 「既有 src/tests/scripts 文件未触碰」 | 实质真（既有 `.py` 零改动），但 4 个索引 md 为修改 | 修正措辞（I-10） |
| 「4.2 零副作用 / import 纯净」 | 行为真，但 4 个测试为空操作（I-09） | 改子进程测试后保留声明 |

---

## 七、门禁处置建议

| 门禁 | 前置修复项 | 说明 |
|------|-----------|------|
| **G2 证据勘误** | I-04、I-09、I-10 | 不改代码即可先做：测试断言改 body 对照、报告状态行/终审声明勘误；G2 保持「条件通过」 |
| **G3 只读准备** | I-02、I-07 | 会话隔离竞态 + 只读 config 连不上；否则首日按规程操作失败/状态不可信 |
| **G4 启用运动/tracking** | I-01、I-03、I-05、I-08 | 重复停止误 FAULT、重连故障泄漏、非目标臂阶跃、机型冻结无通路；四项任一未关闭不应进入 motion |
| **G5 自动验收** | I-06 | 阈值接线 + 超限 fail 测试；或明确降级为人工判据 |
| 持续清理 | 36 项 Minor（去重 ≈30） | 按门禁分批清理；硬编码常量随 G4 冻结统一替换 |

**总体建议**：0 Critical，无需回滚或紧急停止；但按既定评审纪律，**10 项 Important 关闭前不应将阶段 4 标记为「验收通过」**。建议下一步先做「无代码改动的证据勘误 + 报告修正」（I-04/I-09/I-10），再按 G3 → G4 → G5 顺序修并发与配置项。

---

## 八、局限与不可验证项

- **实机/DDS 行为未验证**：G3/G4/G5 未启动，任何真机时序、固件响应、话题可见性结论均不在本报告范围。
- **部署机 wire 布局未验证**：本机 `cyclonedds 11.1.0` 与部署目标 `0.10.2` 不同；I-04 的 1008 B 结论是推断，需部署机复跑确认。
- **Linux `lib/crc_amd64.so` 分支未执行**：本机走纯 Python 回落路径。
- **4.11 原始评审记录不可见**：无法核对原终审是否基于旧提交/旧断言；本报告仅就其声明与当前代码/证据对照给出结论。
- **并发探针为注入放大**：I-01/I-02/I-03 的复现依赖注入睡眠放大窗口；竞态在真实调度下出现概率未量化（代码路径可静态确认存在）。
- **评审只读**：未修改任何被测文件；本报告写入后工作树将出现 1 个未跟踪文件。

---

## 附录 A：复现命令

```powershell
# 工作目录
cd "F:\刀客塔\研究\网球机器人\工程\Code\Sim\.worktrees\stage4-real"
$PY = "C:/Users/Nothing/.conda/envs/mujoco_tennis/python.exe"

# 1) 全量回归（排除 2 个 C++ ABI 文件）
& $PY -m pytest tests/ -q --ignore=tests/test_cpp_forward_pass.py --ignore=tests/test_cpp_backward_pass.py

# 2) 阶段 4 十一文件（312 passed）
& $PY -m pytest tests/test_r1d_config.py tests/test_r1d_mapping.py tests/test_r1d_command_policy.py `
  tests/test_r1d_state_buffer.py tests/test_r1d_transport.py tests/test_r1d_sdk_serialization.py `
  tests/test_r1d_factory.py tests/test_r1d_diagnostics.py tests/test_unitree_r1d_interface.py `
  tests/test_unitree_r1d_tool.py tests/test_robot_interface.py -q

# 3) G0 六文件（172 passed）
& $PY -m pytest tests/test_bound_robot_env.py tests/test_multi_robot_mpc.py `
  tests/test_multi_robot_sim_entrypoint.py tests/test_robot_interface.py `
  tests/test_robot_factory.py tests/test_robot_profile_entrypoint.py -q

# 4) Ruff
& $PY -m ruff check src/real scripts/tools/test_unitree_r1d.py tests/test_r1d_*.py tests/test_unitree_r1d_*.py
& $PY -m ruff check .

# 5) git 范围
git diff --name-status 0be5be7..HEAD
git diff --stat 0be5be7..HEAD
```

## 附录 B：评审分工与任务板

| 任务 | Owner | 裁决 | 关闭 |
|------|-------|------|------|
| task-1 | rev-protocol | Approved with issues（0C/1I/7M） | completed |
| task-2 | rev-mapping | Approved with issues（0C/1I/4M） | completed |
| task-3 | rev-state-interface | Changes requested（0C/3I/8M） | completed |
| task-4 | rev-factory-diag | Approved with issues（0C/3I/8M） | completed |
| task-5 | rev-tests-compliance | Approved with issues（0C/2I/9M） | completed |

## 附录 C：证据索引

- 测试日志：本次评审由 Lead 现场执行，未落盘为文件（命令与输出见 §5.1）。
- SDK 布局探针：`%TEMP%\lead_probe_serialize_layout.py`（临时文件，重启可能丢失；复现脚本内容见 I-04 的断言表）。
- 并发探针：`C:\Users\Nothing\AppData\Local\Temp\revc_probe\`（probe_a/probe_b/probe_c，临时目录；窗口放大手法与断言见 I-01/I-02/I-03）。
- 分支与提交：`unitree/stage4-real` @ `5666959`；基线 `0be5be7`；阶段 4 提交链 11 commits（`14d0798..5666959`）。
