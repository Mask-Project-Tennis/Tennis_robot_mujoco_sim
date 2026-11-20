# 阶段 4 P1 修复实施报告：传输会话隔离、停止收尾与跟踪验收证据

- **日期**：2026-11-18
- **状态**：**实现、独立审查、全量回归与性能基准均完成；未合并、未推送；未连接硬件**。三项 P1 均带判别力红线测试；最终全量回归 **1675 passed + 3 skipped**（第三次干净运行）。G3（实机只读）/ G4（运动前冻结）/ G5（实机验收）未启动，本文不构成任何实机能力声明。
- **分支 / HEAD**：`unitree/stage4-real` @ `fc6119e`，基线 `9a5b3f7`；新增 3 个提交、10 个文件、`+2391/−133`（不含本报告、索引与 `AGENTS.md` 计数更新）。
- **设计 / 计划**：`docs/superpowers/specs/2026-11-18-stage4-p1-fixes-design.md`（§12 为实施期修订）；`docs/superpowers/plans/2026-11-18-stage4-p1-fixes.md`。两份文档位于 Git 忽略目录，不随分支交付。
- **上游**：阶段 4 报告 `2026-11-17-unitree-stage4.md`（§6.6–§6.8 为前两轮修复证据）；两轮评审全文 `2026-11-17-stage4-review.md`、`2026-11-18-stage4-acceptance-review.md`；本轮三项 P1 由用户 2026-11-18 转交的只读评审提出。
- **SDD ledger**：`.superpowers/sdd/2026-11-18-stage4-p1-fixes/progress.md`（含预检表与 R1–R8 裁决、逐任务审查记录；位于 Git 忽略目录）。

---

## 一、裁决与范围

本轮只修三项阶段 4 阻塞，**不修**阶段 5 数据流三项（预测写回物理球状态、诊断消费驱动证据、JSON/NPZ 轨迹自证）与全部减法项；不碰 `RobotExecutor` 返回码、不改公开协议、不启动 G3–G5。

| # | P1 | 评审症状 | 本轮目标 |
|---|----|----------|----------|
| P1-1 | 旧会话命令写进新会话 publisher | fake DDS 中暂停旧线程组包，close→open→acquire 后恢复，两个 publisher 写入数 `[0,1]`；旧 acquire 也能在重开后安装 publisher | SDK 资源归属单一会话；单次发布使用固定 publisher；旧 I/O 回收前拒绝复用同一传输实例 |
| P1-2 | 退权期间登记的控制权丢失被停止收尾丢弃 | 缓停退权阻塞时注入 `mode_machine 7→8`，驱动记录故障，最终却 `READ_ONLY`、`fault_reason=None` | 同一把锁内合并急停与已登记异步故障；独立保留“停止已由反馈确认” |
| P1-3 | 跟踪验收没有产生运动也能通过 | `duration_s=period=0.002 s` + 恒定反馈即 `pass`；只发布首个锚定目标，发布间隔样本数为零 | 接管前校验离散波形确实含运动；判定以实际发布和有效反馈为证据；缺证保留采集结果并判失败 |

**分支策略**：在既有 worktree 的 `unitree/stage4-real` 上追加三个提交，每个 P1 一个提交并附带其红线测试；未新建 worktree、未开独立 PR、未推送。任务 1 的 P1-1 提交同时收编发布基准脚本。

---

## 二、交付总表

| 任务 | 提交 | 文件 | 变更 | 测试结果 | 审查 |
|------|------|------|------|----------|------|
| P1-1 传输会话隔离 | `33f7986` | `src/real/r1d_transport.py`、`tests/test_r1d_transport.py`、`scripts/tools/bench_r1d_transport_publish.py`、`scripts/README.md`、`AGENTS.md` | 5 files `+1584/−113` | transport 47 passed | 首轮 Needs fixes（发现入口窗口缺口）→ 修复后 Approved |
| P1-2 停止收尾合并异步故障 | `5b8c303` | `src/real/unitree_r1d_interface.py`、`tests/test_unitree_r1d_interface.py` | 2 files `+338/−14` | interface 82 passed；四文件 229 passed | Approved（0 Critical / 0 Important / 5 Minor 记账） |
| P1-3 跟踪验收运动证据 | `fc6119e` | `src/real/r1d_diagnostics.py`、`tests/test_r1d_diagnostics.py`、`tests/test_unitree_r1d_tool.py` | 3 files `+469/−6` | diagnostics 76、tool 40、两文件 116 passed；四文件 245 passed | Approved（6 Minor，4 项处置、2 项裁决不做） |

提交链：`9a5b3f7` → `33f7986` → `5b8c303` → `fc6119e`。工作树在当前 HEAD 上仅有本报告、索引与 `AGENTS.md` 计数更新待提交。

---

## 三、P1-1：传输资源归属单一会话

### 3.1 根因

修复前 `publish()`（`src/real/r1d_transport.py:649-679`）的顺序是：先检查 `self._acquired and self._publisher is not None`，再 `_build_lowcmd()` 组包，最后**重新读取** `self._publisher.Write(msg)`。组包期间另一线程可完成 `close()`（置空）→`open()`（新 serial、新订阅）→`acquire_control()`（新 publisher），旧线程最后一步读到的是新 publisher，于是把旧会话命令写进新会话。`_session_serial` 只在 `open()` 自增，`publish()` 从不读它；`acquire_control()` 建好 publisher 后直接 `self._publisher = publisher; self._acquired = True`（原 `:577-578`），没有世代/会话复查。状态回调有世代门，发布路径没有。

### 3.2 修复

采用两把锁，锁序固定为 `_lifecycle_lock → _io_lock`：

- `_lifecycle_lock`：串行化 `open()`/`release_control()`/`close()`，并覆盖 SDK 通道的创建与销毁；`_io_lock` 覆盖 `_publisher/_acquired/_opened/_session_serial` 的读取与写以及整个 publish 段。
- `open()`：在 lifecycle 内完成 ChannelFactory 与订阅创建；session 自增与 `_opened/_on_state` 登记在 io 内一次完成；`subscriber.Init()` 失败则回滚 opened/on_state 并尽力 Close 订阅，异常包为 `R1DChannelError`；成功后才登记 `_subscriber`。
- `publish()`：类型检查后、取 io 前快照 `session = self._session_serial`；io 内条件为 `opened and acquired and publisher is not None and session == self._session_serial`；组包与 Write 使用 io 内取出的**局部** publisher，全程持锁。会话已更换时抛 `R1DTransportUsageError`（消息注明“会话已更换（close/open）”）。
- `close()`：lifecycle 内进入，用 io 完成状态交换（置 `_opened=False`、`_on_state=None`、取走 publisher/subscriber、使 `_acquire_serial` 失效），因此会等待一次在途 Write；随后释放 io，在 lifecycle 内完成双通道 Close，避免 SDK Close 等回调造成死锁。serial 只在 `open()` 自增；close 以 `_opened` + `_acquire_serial` 失效，保留既有 `session_id == 2` 语义（复审裁决）。close 幂等；耗时超过 50 ms 仅记 warning，不改变控制流。
- `acquire_control()`：入口快照 session 与 `_acquire_serial` 票据；MotionSwitcher 的阻塞等待在锁外；提交时短暂取 lifecycle→io，复查 opened/session/ticket/未 acquired/publisher 为空，不匹配则关闭自建 publisher 并抛 `R1DControlError`。并发 acquire 以最新有效票据为准。
- `release_control()`：io 内取走 publisher 并置 `_acquired=False`，io 外 Close；失败按既有语义上抛。

不变式：

| 编号 | 内容 |
|------|------|
| I1 | `_publisher` 非空 ⇒ `_opened` 且 `_acquired` 且属于当前 session；三者在 io 内一起变 |
| I2 | 每次 Write 都在 io 内，且使用本次调用在 io 内取出的局部 publisher；组包后不再读成员 |
| I3 | `_session_serial` 只在 `open()` 持 io 自增；`close()` 通过 `_opened` 与 `_acquire_serial` 使旧调用失效 |
| I4 | `_subscriber/_on_state` 归 lifecycle，open 建立、close 拆除 |
| I5 | acquire 提交同时验 session 与 ticket；旧提交不安装 publisher，且必须释放已取得的控制权 |

### 3.3 红线与证据

| 测试 | 场景 | 修复前 | 修复后 |
|------|------|--------|--------|
| `test_publish_in_flight_cannot_write_new_session_publisher` | 旧 publish 持 io 组包时 close→open→acquire，再放行 Write | 旧 publisher 未收到命令（`assert 0 == 1`） | 旧命令写入旧 publisher，新 publisher 零写入 |
| `test_stale_acquire_does_not_install_publisher_after_reopen` | CheckMode 阻塞的旧 acquire 跨 close/open | 未抛 `R1DControlError` | 旧 acquire 关闭自建 publisher 并抛 `R1DControlError`，新 publisher 不被覆盖 |
| `test_open_waits_until_previous_close_tears_down` | close 拆订阅期间发起 open | 新订阅先于旧 teardown 建立（`assert 2 == 1`） | 事件顺序为旧 Close 完成后才建新订阅 |
| `test_publish_entry_window_does_not_write_new_session_publisher` | publish 已过类型检查、尚未取得 io 时被 close/open/acquire 切走 | `publish_errors=[]`（越界成功） | 抛 `R1DTransportUsageError`，新 publisher 无消息 |

第四条是本轮独立审查发现的**入口窗口**缺口：前三条规定了“已持 io 组包”的窗口，没有覆盖“方法已进入、尚未拿锁”的窗口。修复方式即上面的入口快照；该测试在修复提交 `71bdab3` 上重放为 `1 failed`（`assert len(publish_errors) == 1` → `0 == 1`），修复后 1 passed。最终 `tests/test_r1d_transport.py` **47 passed**。

### 3.4 发布性能

`scripts/tools/bench_r1d_transport_publish.py` 自带最小 fake SDK，不 import 测试、不加载 `unitree_sdk2py`；`_BaselineTransport` 冻结 `9a5b3f7` 的旧 publish/close 作为基线，locked 路径直接使用生产 `SdkR1DTransport`，不再外包锁。最终证据 `results/stage4_p1_fixes/final/bench_final.json`（`git_sha=fc6119e…`，5 轮、每轮 20000 对）：

| 指标 | 基线（9a5b3f7 语义） | 修复后 | 说明 |
|------|---------------------|--------|------|
| publish p50 | 39.4–41.0 µs | 39.7–41.0 µs | 配对差中位 **0.2 µs** |
| publish 配对差 mean | — | 0.027–0.354 µs（中位 0.228） | 单次噪声远大于锁开销 |
| `_build_lowcmd + CRC` p50 | — | 39.5 µs | 发布路径大头 |
| 一次 RLock 加解锁 | — | 约 200 ns | — |
| 空闲 close | 0.0087 ms | 同量级 | — |
| 200 ms 在途 Write 时 close | 0.0087 ms 返回，`write_in_flight_at_close=True` | 185.64 ms 返回，`write_in_flight_at_close=False` | 加锁版等到在途 Write 结束 |
| 写入数 / 错误 / 自检 | — | 1/1、[]、`self_checks_passed=true` | — |

基线在 Write 仍在途时 close 就返回，正是竞态窗口；加锁版的 close 延迟等于一次在途 Write，而不是锁本身的开销。按 2 ms 控制周期计，每拍锁成本约占周期 0.01%，可忽略。边界：fake DDS、Write 空实现、单发布线程、Windows 计时；真实 DDS 序列化/发送时间会同时加到两版，不改变差值。

---

## 四、P1-2：停止收尾合并在途异步故障

### 4.1 根因

`_finish_stop()`（修复前 `src/real/unitree_r1d_interface.py:1290-1332`）只读 `_emergency_latch` 作为升级条件；`_async_fault_reason` 既不参与终态判定，也不被清除。退权（`release_control`）阻塞期间状态回调把 `mode_machine` 变化写入 `_async_fault_reason`；收尾时 `final_fault=False`、`outcome=None`、`escalated=False`，于是走 `READ_ONLY` 并显式 `_fault_reason=None`，同时写入 `_stop_confirmed=True`。残留事件随后被 `_decide()` 的 `READ_ONLY` 防御分支当“会话隔离泄漏”丢弃。

### 4.2 修复

`_finish_stop()` 在**同一把 `_lock`** 内：

1. 先查 `generation != self._generation`；stale 只记日志，不读写新会话的 fault/stop_confirmed；
2. 读取 `escalated = _emergency_latch` 与 `async_reason = _async_fault_reason`；
3. `faulted = final_fault or outcome is not None or escalated or async_reason is not None`；
4. `faulted` 时置 `FAULT`，reason 按 `trigger → outcome → async_reason → “退权窗口收到急停升级（终态强制 FAULT）”` 去重拼接，保留全部原因；否则 `READ_ONLY` 且 `_fault_reason=None`；
5. 两个分支都写 `_stop_confirmed=confirmed`（停止是否被反馈确认，与“是否有故障”正交）；
6. 清 pending/activation/emergency/async，`notify_all()`；锁内取终态与 reason 快照，锁外按快照记日志并 `_stop_done.set()`，避免日志 I/O 持锁。

`_decide()` 的 `READ_ONLY` 残留分支保留为纵深防御；停止完成后新注入的机型变化不会再登记（phase 已不是 ACTIVE/STOPPING）。

### 4.3 红线与证据

| 测试 | 场景 | 修复前 | 修复后 |
|------|------|--------|--------|
| `test_async_control_loss_during_release_is_terminal_fault` | ACTIVE→slow_stop→反馈确认→release 阻塞→注入合法 `mode_machine` 变化→放行 | `READ_ONLY`、`fault_reason=None` | `FAULT`，reason 含原 trigger 与“控制权丢失”，`stop_confirmed=True`，async 已清 |
| `test_mode_change_after_stop_does_not_raise_fault` | 停止完成后注入机型变化 | — | 不误报 FAULT |
| `test_finish_stop_merges_fault_sources`（参数化） | final_fault / outcome / emergency / async 的 7 种组合×confirmed | 多源 reason 丢失 | 所有原因保留，confirmed 恒按入参 |
| `test_stale_finish_stop_preserves_new_session_fault` | stale 收尾遇到新会话故障 | 可能覆盖新会话 | 新会话字段不变 |
| async-only caplog | 仅异步故障 | 输出 READ_ONLY 成功日志 | 不输出该日志 |

修复前同一批测试 **8 failed / 10 passed**，失败均落在上述目标断言（非导入、夹具或超时）；修复后 `tests/test_unitree_r1d_interface.py` **82 passed**，四文件串行 229 passed。独立审查 Approved，0 Critical / 0 Important；5 条 Minor 记账不改（`_stop_done` 共享 Event 的 stale 提前 set 属 base 既有、参数化断言顺序、缺 reason 分段/同窗口多事件、docstring 措辞含共享事件、全量回归归 Lead）。

---

## 五、P1-3：跟踪验收必须包含运动证据

### 5.1 根因

`run_tracking()`（修复前 `src/real/r1d_diagnostics.py:910`）用 `n_steps = max(1, round(duration/period))`。当 `duration=period=0.002 s` 时 `n_steps=1`，唯一目标在 `t=0`：`0.5·A·(1−cos 0)=0`，target 等于 `q_start`，整个波形只发布首个锚定目标，实际位移为零；`max_wave_step=0`，回程也被限成零。判定层（修复前 `:1202`）只在 `not unavailable` 时检查发布超期，`publish_interval.unavailable` 被跳过；tracking 配对误差≈0，于是 `pass`。配置层只要求 `duration_s>0`，没有 `duration_s ≥ send_period_s` 或波形位移约束，因此该输入完全合法。

### 5.2 修复

- `R1DDiagnosticConfig.tracking_step_count() -> int`：仅 tracking 可调，返回与 `run_tracking` 同式的 `max(1, round(duration/send_period))`；`run_tracking` 改用它，消除两处步数算法漂移。
- `R1DDiagnosticConfig.expected_target_span_rad() -> float`：仅 tracking；按 `k=0..steps-1` 的离散余弦目标流式求 min/max，返回跨度；read_only 调用明确抛 `ValueError`。
- `__post_init__` 在既有校验之后增加：`steps >= 2` 且 `span >= max(0.1·A, 1e-3)`，否则 `ValueError`。这是**接管前**拒绝，不会先动硬件再失败。
- `_apply_threshold_verdicts()`：仅 tracking 时，`publish["unavailable"]` 判 fail（“发布间隔证据不足”）；从本次记录中过滤 `kind=="active" and success and q_sdk` 有限、且至少两条，取 `q_sdk[selected_joint]` 的跨度，要求 `>= 0.5 × expected_span`，否则 fail（“实际发布的模型序目标未覆盖波形”）。read_only 仍不因发布缺失失败。
- fail/stop/缺目标/NaN 记录不计入跨度；`selected_joint` 目标非有限直接 fail。
- `finalize()` 先判定后写产物；失败仍保留 JSON/CSV/NPZ，不因判 fail 丢采集结果。

### 5.3 红线与证据

新增/调整 16 个诊断用例与 CLI 用例，覆盖：0.002/0.002 构造即拒绝、频率过低导致跨度不足、单条 active 发布、恒定 q_sdk、采样混叠（period=0.002/f=500/duration=0.006）、最小合法（period=0.002/f=250/duration=0.004/A=0.01，steps=2、span=0.01）、`round(2.5)=2`、0.499/0.5 阈值边界、`selected_joint` 非零、符号与 offset 不影响跨度、失败/stop/NaN 不计、失败落盘、CLI 接管前拒绝短波。修复前 **15 failed / 1 passed**，失败均为目标断言；修复后 diagnostics 76 passed、tool 40 passed、两文件 116 passed、四文件 245 passed。

夹具只把 `_DIAG_DURATION`/`_TRACK_DURATION` 从 0.06 调到 0.12 s（60 步、跨度≈0.00656 ≥ 0.005），幅度、频率、断言与阈值未动——旧夹具本身不满足新约束，属必要适配而非削弱断言。

独立审查 Approved，6 条 Minor：消息措辞（raw q_sdk 不是“模型序目标”，而是“SDK 侧数值按模型序排列”）、0.5× 用例只断言 span 子项、报告缺“成功发布子序列跨度”限制、缺恰好等于阈值的构造用例、5 处 `assert is not None` 在 `-O` 下风险、`expected_target_span_rad` 为 O(steps) 无上限。前四项分别以消息修正、整体 `verdict=="fail"` + “无配对样本”断言、报告限制说明、显式中文 `ValueError` 处置；恰好等值与 O(steps) 两项裁决不做（浮点等值脆弱、步数由配置上界约束），记录在报告 §7.9/§7.10。

---

## 六、测试与回归

### 6.1 基线（`9a5b3f7`，修复前）

四文件回归首跑 `196 passed, 1 failed`、复跑 `195 passed, 2 failed`；`tests/test_r1d_diagnostics.py` 整文件三次为 `1 failed / 61 passed / 2 failed`；失败集中在 `test_tracking_full_run`、`test_thresholds_present_within_limit_pass`、`test_non_integer_period_rate_limited_return[1.0]`，均报 `publish 间隔超期… max=0.004/0.005 s`。单测隔离复跑三次全 pass。结论：`9a5b3f7` 已有 timing flake，根因是 `_pump` 以半速推进 fake clock 与发送线程调度竞争；与本轮改动无关。按裁决 R8 不扩范围修，只如实记录（基线日志 `results/stage4_p1_fixes/baseline_four_files.log`、`..._rerun.log`）。

### 6.2 分任务

| 阶段 | 命令/范围 | 结果 |
|------|-----------|------|
| P1-1 修复前 | 三条红线 + 既有 transport | 3 failed（目标断言） |
| P1-1 修复后 | `tests/test_r1d_transport.py` | 47 passed |
| P1-1 入口窗口 | 新测试在 `71bdab3` 重放 / 修复后 | 1 failed（`publish_errors=[]`）/ 1 passed |
| P1-2 修复前 | 新测试批 | 8 failed / 10 passed |
| P1-2 修复后 | `tests/test_unitree_r1d_interface.py` | 82 passed |
| P1-2 四文件 | transport+sdk_serialization+interface+diagnostics | 229 passed |
| P1-3 修复前 | 新测试批 | 15 failed / 1 passed |
| P1-3 修复后 | diagnostics + tool | 116 passed |
| P1-3 四文件 | 上述四文件 | 245 passed |

### 6.3 全量回归（排除两个 C++ ABI 文件）

命令：`pytest tests/ --ignore=tests/test_cpp_forward_pass.py --ignore=tests/test_cpp_backward_pass.py -q -rs`

| 运行 | 结果 | 失败/说明 |
|------|------|-----------|
| 第 1 次 | `2 failed, 1673 passed, 3 skipped` | `test_non_integer_period_rate_limited_return[0.5]`、`[1.0]`，publish 间隔 4 ms > 1.5×2 ms |
| 第 2 次 | `1 failed, 1674 passed, 3 skipped` | `[0.5]` 同因 |
| 隔离复跑 | 单测 3 次均 `2 passed`；diagnostics 整文件 `76 / 76 / 1 failed+75 passed` | 证明为调度相关 flake |
| 第 3 次 | **`1675 passed, 3 skipped`，exit 0，155.23 s** | 干净；日志 `results/stage4_p1_fixes/final/full_suite_rerun2.log` |

3 个 skip：2 个 C++ 测试文件因扩展 ABI 不匹配被排除后的既有 skip；1 个为 worktree `results/` 为空时 RM65 seed42 基线 NPZ 缺失的设计内条件跳过。三次运行的失败均为同一既有 flake，未新增失败；最终干净运行作为交付数字，前两次如实保留。

### 6.4 计数与 Ruff

最终用例数（实测）：`test_r1d_transport.py` 47、`test_unitree_r1d_interface.py` 82、`test_r1d_diagnostics.py` 76、`test_unitree_r1d_tool.py` 40；全量 1675 passed + 3 skipped。相对基线 1627 的净增为 48（文件级计数含参数化用例合并，以最终运行为准）。

`ruff check`：本轮 8 个涉改动 Python 文件全部通过（`src/real/r1d_transport.py`、`src/real/unitree_r1d_interface.py`、`src/real/r1d_diagnostics.py`、三个测试文件、基准脚本、`scripts/tools/test_unitree_r1d.py`）。`ruff format --check` 仍有 5 个文件 would reformat，均为干净 HEAD 既有基线（本轮新增区 hunk 为 0），未做整文件重排以免污染 diff。

---

## 七、基准证据（与 §3.4 同源）

- 脚本：`scripts/tools/bench_r1d_transport_publish.py`（CLI：`--pairs --rounds --build-iterations --lock-iterations --close-block-ms --json`）。
- 最终 JSON：`results/stage4_p1_fixes/final/bench_final.json`；日志：`.../bench_final.log`。
- 关键自检：`write_in_flight_at_close` baseline=`True`、locked=`False`；writes=`1/1`；errors=`[]`；`self_checks_passed=true`。若基线不再呈现竞态窗口或线程未退出，脚本返回非零，避免“基准通过但测错对象”。
- 结论：每拍加锁成本在噪声以下（配对中位 0.2 µs）；close 的等待等于一次在途 Write，是可解释的上界；不为性能去掉锁。未来若要把发布路径从 39–41 µs 再降，应复用 `LowCmd` 模板与预构造 35 个电机命令，而不是削弱会话隔离。

---

## 八、独立审查与处置

| 轮次 | 范围 | 结论 | 主要发现 |
|------|------|------|----------|
| review-1 | P1-1 `71bdab3` | Spec ❌ / Needs fixes | **Critical**：publish 入口窗口无会话快照；**Important**：设计 I3 要求 close 自增 serial，实现未做；6 Minor |
| review-2 | P1-1 修复 `5deb91d`+`4cee560` | Approved | 8 条全 ADDRESSED；新增 3 Minor（docstring stale-session、design open 伪代码 `_subscriber` 位置、RED 复现命令） |
| review-3 | P1-3 `1b5506b` | Approved | 0C/0I/6M（见 §5.3） |
| review-final | 整分支 `9a5b3f7..fc6119e` | **Ready to merge = Yes** | 0C/0I/5 非阻断 Minor |

**Critical 的裁决**：close 不自增 `_session_serial`（保留既有 `session_id == 2` 语义），改为 publish 入口快照 + io 内复查；设计 §2.1/§4.1 I3/§4.2/§4.3 同步为“serial 只在 open 自增，close 以 `_opened` + `_acquire_serial` 失效”。这样既关闭了入口窗口，又不改变既有会话编号契约。

**终审 5 条非阻断 Minor 的处置**：① `AGENTS.md` 计数过时——已更新为 1675 与 47/82/76/40；② `unitree_r1d_interface.py` 退权成功写 `_control_serial=None` 无世代校验——pre-existing，transport 串行化后该路径不可达，留观察；③ 诊断夹具 0.12 s 略增 flake 曝光——与 §6.1 同一既有 flake，不扩范围；④ `progress.md` 尾部陈旧——ledger 属忽略目录，收尾时整理；⑤ 基准证据仅在未跟踪 `results/`——本报告记录方法与数字，原始 JSON 保留在本机，未纳入 Git。终审另列 10 项 “Declined to judge”（驱动 connect/disconnect 并发、`_stop_done`、旧调用者身份、Write 卡死无 timeout、记录不按 session 过滤、真实 SDK 时序、AGENTS、O(steps)、reasons 累加、open Init 无 timeout），均超出本轮三项 P1 范围，未做承诺。

---

## 九、遗留与限制（如实）

1. **timing flake 仍存在**：`9a5b3f7` 已有的三条 tracking 用例在整文件/全量运行时会偶发发布间隔超期；隔离复跑通过，最终全量干净，但未修复根因（裁决 R8）。若后续要稳定 CI，应单独修 `_pump` 的时钟推进与发送线程同步，不与本轮 P1 混合。
2. **`ruff format` 基线**：5 个文件在干净 HEAD 即 would reformat；本轮只保证新增区无格式 hunk，未整文件重排。
3. **fake DDS 边界**：所有传输测试与基准都基于 fake channel/publisher；未连接 Unitree 硬件、未上电、未外发 DDS。锁的正确性证明是代码级与 fake 级，真实 SDK 的 Write 阻塞/回调时序仍待 G3 验证。
4. **close 可能长时间等待**：若底层 Write 卡死，close/release 会一直等到 Write 返回；当前只有 50 ms warning，没有超时强制断开（强制断开可能留下未完成的 DDS 调用）。该风险已写入设计 §12，未在本轮引入 timeout。
5. **诊断跨度是“成功发布子序列”的跨度**：若发布极稀薄（例如只有两条恰好覆盖两端的记录），跨度检查仍可能通过；判定层已要求至少两条有效目标且跟踪配对样本存在，但没有对“发布密度”设下界。该限制已记录在 task-3 报告 §7.3。
6. **未实机验证**：G3/G4/G5 未启动，不能声称这些修复已在真实控制权切换、真实停止反馈上生效。
7. **阶段 5 三项未做**：预测写回物理球状态、诊断消费驱动同帧证据、JSON/NPZ 轨迹互相绑定仍在下一分支范围；本轮不改 `predictive_safety.py`、`profile_sim_env.py`、`r1d_runner.py`、`robot_executor.py`。

---

## 十、结论

在本机环境（Windows + Python 3.11.15 + fake transport + 真实 SDK 仅离线序列化；全程未连接硬件）下：

- P1-1：传输的 SDK 资源已归属单一会话；publish 的入口快照 + io 内复查关闭了“组包前被切走”的窗口；旧 acquire 不再能安装 publisher；close 会等在途 Write，且在旧 I/O 拆完前不允许新 open。transport 47 passed，基准显示每拍锁成本在噪声以下。
- P1-2：停止收尾在同一把锁内合并 `final_fault/outcome/emergency/async`，reason 保留全部来源，`stop_confirmed` 与故障判定正交；退权期间的控制权丢失不再被 READ_ONLY 吞掉。interface 82 passed。
- P1-3：接管前拒绝零运动波形；判定层把“发布间隔不可用”和“实际发布目标未覆盖波形”判为失败，缺证保留产物。diagnostics 76 + tool 40 passed。
- 整分支终审 **Ready to merge = Yes**（0C/0I/5 非阻断 Minor），全量回归 **1675 passed + 3 skipped**（第三次干净运行；前两次命中既有 timing flake，已如实记录）。

**阶段结论：三项阶段 4 P1 阻塞已关闭，具备合并条件；阶段 4 仍待 G3–G5 实机验收，本文不宣称实机能力。** 下一步为阶段 5 数据流收敛（预测写回、诊断证据、轨迹自证）与 G3 现场只读核对。

---

## 附录 A：复现命令

```powershell
cd "F:/刀客塔/研究/网球机器人/工程/Code/Sim/.worktrees/stage4-real"
$env:PYTHONUTF8 = "1"
$PY = "C:/Users/Nothing/.conda/envs/mujoco_tennis/python.exe"

# 1) 三项 P1 的目标文件
& $PY -m pytest tests/test_r1d_transport.py tests/test_unitree_r1d_interface.py `
    tests/test_r1d_diagnostics.py tests/test_unitree_r1d_tool.py -q

# 2) 全量回归（排除两个 C++ ABI 文件）
& $PY -m pytest tests/ -q -rs `
    --ignore=tests/test_cpp_forward_pass.py --ignore=tests/test_cpp_backward_pass.py

# 3) Ruff
& $PY -m ruff check src/real/r1d_transport.py src/real/unitree_r1d_interface.py `
    src/real/r1d_diagnostics.py tests/test_r1d_transport.py `
    tests/test_unitree_r1d_interface.py tests/test_r1d_diagnostics.py `
    tests/test_unitree_r1d_tool.py scripts/tools/bench_r1d_transport_publish.py

# 4) 发布基准（最终证据）
& $PY scripts/tools/bench_r1d_transport_publish.py --rounds 5 --pairs 20000 `
    --json results/stage4_p1_fixes/final/bench_final.json
```

## 附录 B：提交链

```
9a5b3f7  fix(real): 处理阶段 4 第二轮复验意见（2 Blocking 全关）
 └─ 33f7986  fix(real): isolate R1D transport sessions and teardown
     （P1-1：两锁 + 会话票据 + publish 入口快照；基准脚本收编）
 └─ 5b8c303  fix(real): preserve asynchronous faults during stop finalization
     （P1-2：_finish_stop 同锁合并四类故障源）
 └─ fc6119e  fix(real): require motion evidence for R1D tracking acceptance
     （P1-3：波形前置校验 + 发布证据判定）
```

## 附录 C：证据索引

| 内容 | 路径 |
|------|------|
| 设计（含 §12 修订） | `docs/superpowers/specs/2026-11-18-stage4-p1-fixes-design.md`（Git 忽略） |
| 实施计划 | `docs/superpowers/plans/2026-11-18-stage4-p1-fixes.md`（Git 忽略） |
| SDD ledger | `.superpowers/sdd/2026-11-18-stage4-p1-fixes/progress.md`（Git 忽略） |
| P1-1 报告 / 审查 | `.superpowers/sdd/.../task-1-report.md`、`review-1.md`、`review-2.md` |
| P1-2 报告 / 审查 | `.superpowers/sdd/.../task-2-report.md`、`review-p1-stop` 结论 |
| P1-3 报告 / 审查 | `.superpowers/sdd/.../task-3-report.md`、`review-3.md` |
| 终审 | `.superpowers/sdd/.../review-final.md` |
| 基线日志 | `results/stage4_p1_fixes/baseline_four_files.log`、`..._rerun.log` |
| 全量日志 | `results/stage4_p1_fixes/final/full_suite.log`、`full_suite_rerun.log`、`full_suite_rerun2.log` |
| 基准证据 | `results/stage4_p1_fixes/final/bench_final.json`、`bench_final.log` |
| 任务级 RED/GREEN | `results/stage4_p1_fixes/task1/`、`task2/`（`red_tests.txt`、`green_tests.txt` 等） |
