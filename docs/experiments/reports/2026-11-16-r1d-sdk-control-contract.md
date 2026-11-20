# R1-A7-D SDK、固件与整机控制契约核验报告（Task 4.1）

- **日期**：2026-11-16
- **性质**：文档与公开源码取证（不安装 SDK、不连接硬件、不写业务代码、无 git 写操作）
- **任务来源**：`docs/superpowers/plans/2026-11-16-unitree-stage4-construction.md` Task 4.1；brief：`.superpowers/sdd/2026-11-16-unitree-stage4-construction/task-4.1-brief.md`
- **工作树**：`.worktrees/stage4-real`，基线 `0be5be7`（分支 `unitree/stage4-real`；本片执行时点的 BASE 记录——其后并行的 Task 4.0 已先行提交 `14d0798`，与本片取证内容无关）
- **验收方式**：证据审查。每条结论附来源 URL + commit + 文件与行号 + 取得日期；未确认字段保留 null。
- **交付物**：本报告 + `configs/robots/r1d_protocol_manifest.yaml`
- **取证材料**：`.superpowers/sdd/2026-11-16-unitree-stage4-construction/evidence-4.1/`（网页快照 + 固定 commit 源码文件，git-ignored；目录内未运行任何解释器/构建工具，仅 curl 下载与文本读取）

---

## 0. 结论速览

1. **Python 官方支持成立，A7 低层路径官方双语言在案**：上轮"官方指南正文打开失败"遗留的疑点（高层 ArmSdk 只有 A5 兼容 13 关节是否等于 Python/A7 无路径）本轮消除——官方 Python 仓库 `unitree_sdk2_python` master@`814556d1`（2026-09-21，提交信息 "add r1_A5_low_level_example and r1_A7_low_level_example"）含 **`example/r1/low_level/r1_A7_low_level_example.py`**，走 `rt/lowcmd`/`rt/lowstate` + hg `LowCmd_`/`LowState_` + CRC32 + MotionSwitcher，与 C++ `r1_A7_wrist_swing_example.cpp` 逐项一致；官方指南架构页明文"开发接口 1. DDS，支持C++ 以及 Python"。
2. **唯一实施路径冻结**：七关节臂级控制 = 低层 `rt/lowcmd` 直接寻址整机槽位 15–21（左）/22–28（右）；高层 `rt/arm_sdk` 封装（`r1_pub.h::ArmSdk::JOINTS` 固定 13 关节、止于 WristRoll）不含 A7 特有腕俯仰/腕偏航，不采用。
3. **IDL/CRC/topic/index 契约可离线冻结**（双侧源码逐字段一致）；**实机行为语义**（失联、非目标部件厂商级保持策略、mode_machine 枚举、tick 语义、释放后恢复）公开渠道不可得，列入 §7 问询清单，manifest 置 null。
4. G1 状态：**部分冻结**——协议结构层（sdk/dds 报文与索引）冻结；实机行为层（control 生命周期语义 + 部件保持策略的厂商承诺）未冻结，不阻塞 4.2–4.5 的离线开发与测试，阻塞 G3/G4 前的实机动作。

---

## 1. 取证渠道与固定版本

### 1.1 渠道清单

| # | 渠道 | 访问方式 | 结果 |
|---|---|---|---|
| C1 | 官方文档中心 support.unitree.com（R1-A-D SDK 开发指南） | WebFetch（静态）→ 空壳；JinaReader/web-reader 渲染抓取 | sdk_overview、get_sdk、architecture_description 三页全文成功；quick_develop、dds_communication_interface（猜测 slug）两页渲染后"暂无文章内容" |
| C2 | 官方文档中心 developer/Python 页 | Tavily 站内高级抓取 | 全文成功（内容为 Go2 示例 + cyclonedds 安装 FAQ） |
| C3 | GitHub `unitreerobotics/unitree_sdk2`（C++） | GitHub API + raw.githubusercontent.com 固定 commit 下载 | 成功（API 元数据 + 12 个关键文件；API 响应快照存 github-meta/，修复轮 1 补） |
| C4 | GitHub `unitreerobotics/unitree_sdk2_python`（Python） | 同上 | 成功（注意仓库名为下划线，连字符 `unitree_sdk2-python` 404，响应快照 github-meta/repo_unitree_sdk2-python_404.json；commit 814556d1 快照 github-meta/commits_python_master_top3.json） |
| C5 | PyPI | WebFetch | `pypi.org/project/unitree_sdk2py/` 404——PyPI 上无此名发布页；包仅以源码仓 setup.py（name='unitree_sdk2py', version='1.0.1'）形式存在（WebFetch 抓取结果，未落盘 JSON） |
| C6 | unitree_ros / ROS2 资源 | 本轮未重新取证 | URDF 与整机索引沿用既有审计 `2026-11-15-r1a7d-model-interface-audit.md`（固定 commit `157ff95`），不重复 |

### 1.2 证据等级（沿用既有审计约定）

- **[指南]** = 宇树官方文档中心页面事实（抓取时页面"更新时间"为准）
- **[公开示例]** = 官方开源仓库代码事实（固定 commit）
- **[实机]** = 本台机器人实物验证结论——**本报告零条 [实机] 证据**

### 1.3 固定版本

| 项 | 值 | 来源 |
|---|---|---|
| C++ SDK | `unitree_sdk2` main@`aaf69cb56c9d5ed63d8b284008ec3a8ab999bec1`（2026-11-09 提交，晚于最新 tag 2.0.2，无对应 release tag） | GitHub API commits；既有审计已固定 |
| Python SDK | `unitree_sdk2_python` master@`814556d15970dd2ecf1c9984e845ca02ab07e206`（2026-09-21） | GitHub API commits（本轮新固定） |
| Python 包声明 | `unitree_sdk2py` version `1.0.1`，python_requires `>=3.8`，依赖 `cyclonedds==0.10.2` + numpy + opencv-python | setup.py L3-20 |
| IDL 来源 | Python：仓库内 `unitree_sdk2py/idl/unitree_hg/msg/dds_/*.py`（Cyclone DDS idlc Python Backend v0.11.0 生成，见各文件头注释 L2-3）；C++：`include/unitree/idl/hg/*.hpp` | 两仓库固定 commit |
| OS/架构 | C++ README：Ubuntu 20.04 LTS / aarch64 与 x86_64 / gcc 9.4.0（README.md L5-7）；Python：apt/pip3 流程（Linux） | [公开示例] |
| 安装命令 | `git clone https://github.com/unitreerobotics/unitree_sdk2_python.git && pip3 install -e .`（cyclonedds 0.10.2 需可用，官方 FAQ 给出源码编译 releases/0.10.x + `CYCLONEDDS_HOME`） | README.md L12-39 / developer/Python 页 |
| 离线序列化检查命令 | 见 §5.4（W2 安装后执行，本片未运行） | 本报告设计，待 4.5 落地 |
| 本机安装状态 | **未安装**（主代理裁定：本片禁止安装，安装属 W2 前经用户批准时点） | — |

---

## 2. G1 决策表

> 表中"行号"指取证目录 `evidence-4.1/` 内固定 commit 文件的行号（与 GitHub 该 commit blob 逐字一致；SHA256 见 §8）。

| 编号 | 结论 | 等级 | 来源（URL / commit / 文件:行 / 日期） |
|---|---|---|---|
| G1-01 | 官方 SDK 指向 `unitree_sdk2`（GitHub），指南未列版本号/commit；本项目沿用既有审计固定 `aaf69cb5`，该 commit 为 main HEAD（2026-11-09）且无对应 release tag，版本浮定风险持续存在 | [指南]+[公开示例] | https://support.unitree.com/home/zh/R1-A-D_SDK_Development_Guide/get_sdk （更新时间 2026-11-12 14:47:09，2026-11-16 取得）；GitHub API commits/main |
| G1-02 | 官方 Python SDK 存在：仓库 `unitree_sdk2_python`（**下划线**，连字符名 404），包名 `unitree_sdk2py` v1.0.1；PyPI 无同名发布页（404），以源码 commit 为准冻结 `814556d1` | [公开示例] | https://github.com/unitreerobotics/unitree_sdk2_python ；setup.py:3-4（name/version）；https://pypi.org/project/unitree_sdk2py/ → 404（2026-11-16 取得） |
| G1-03 | 官方支持 OS = Ubuntu（C++ README 明示 20.04 LTS；Python 页基于 apt/pip3）；架构 aarch64/x86_64；Python CRC 快路径仅提供 Linux .so（crc_amd64.so/crc_aarch64.so），非 Linux 回落纯 Python 逐位实现（功能等价、性能未验证）。**Windows/macOS 无官方支持声明**——本项目部署主机 OS/架构未定，列为待确认项 | [公开示例] | cpp/README.md:4-7；python/crc.py:28-37,224-228；python/README.md:4-19（均 2026-11-16 取得） |
| G1-04 | **A7 七关节唯一实施路径 = 低层 `rt/lowcmd`（hg LowCmd_ 整机 35 槽直接寻址）**。理由：(a) 高层 `rt/arm_sdk` 封装 `ArmSdk::JOINTS` 固定 13 关节（左右臂各 5 止于 WristRoll + WaistYaw + 头 2），不含 A7 腕俯仰/腕偏雅槽位 20/21/27/28，物理上无法驱动 7 自由度臂；(b) 低层路径经官方双语言 A7 示例确认（C++ `r1_A7_wrist_swing_example.cpp` + Python `r1_A7_low_level_example.py`），索引映射 {左 15–21, 右 22–28, 头 29,30} 与指南关节限位表一致；(c) Python 支持成立，真实适配**不 blocked**——但开发机未装 SDK（裁定），4.5 的真实序列化专项测试在 W2 安装后才能执行，此前只能交付合成协议离线测试 | [公开示例] | cpp/r1_pub.h:28-42（JOINTS 13 关节）；cpp/r1_A7_wrist_swing_example.cpp:101-105 与 python/r1_A7_low_level_example.py:48-52（joint_idx_in_idl）；https://support.unitree.com/home/zh/R1-A-D_SDK_Development_Guide/architecture_description "开发接口 1. DDS，支持C++ 以及 Python"（更新时间 2026-11-12 14:35:50，2026-11-16 取得） |
| G1-05 | hg IDL 布局冻结（Python 与 C++ 逐字段一致）：`LowCmd_ = {mode_pr:u8, mode_machine:u8, motor_cmd:MotorCmd_[35], reserve:u32[4], crc:u32}`；`LowState_ = {version:u32[2], mode_pr:u8, mode_machine:u8, tick:u32, imu_state:IMUState_, motor_state:MotorState_[35], wireless_remote:u8[40], reserve:u32[4], crc:u32}`；`MotorCmd_ = {mode:u8, q/dq/tau/kp/kd:f32, reserve:u32}`；`MotorState_ = {mode:u8, q/dq/ddq/tau_est:f32, temperature:i16[2], vol:f32, sensor:u32[2], motorstate:u32, reserve:u32[4]}`；`IMUState_ = {quaternion:f32[4], gyroscope:f32[3], accelerometer:f32[3], rpy:f32[3], temperature:i16}`。**与 go2 LowCmd 的关键差异：hg 无 head/seq 头字段**。序列化长度（官方 CRC 打包口径）：hg LowCmd=**1004 B**（官方 crc.py L22 注释 "#size 1004" 与格式串 `'<2B2x'+'B3x5fI'*35+'5I'` 实算 4+35×28+20 一致）、hg LowState=2092 B（L24 注释 "#size 2092" 同样一致）；W2 安装后实测复核 | [公开示例] | python/_LowCmd_.py:25-29；python/_LowState_.py:25-33；python/_MotorCmd_.py:25-31；python/_MotorState_.py:25-34；python/_IMUState_.py:25-30；cpp LowCmd_.hpp:26-30；cpp LowState_.hpp:28-36；cpp MotorCmd_.hpp:23-29；cpp MotorState_.hpp:24-33；长度：python/crc.py:22-25（官方注释 + 格式串实算）（2026-11-16 取得；修复轮 1 更正 524→1004） |
| G1-06 | CRC 契约冻结：算法 = 逐位 MSB-first、多项式 0x04C11DB7、初值 0xFFFFFFFF、无最终异或、非反射；覆盖 = 整包按 4 字节小端字，长度 = (总字节/4 − 1) 字（覆盖除末尾 crc 外全部内容含对齐填充）。**接收侧必须校验 LowState CRC，失败丢帧**（双侧示例一致）。Python 官方实现为手写 struct.pack 布局（非 cyclonedds 原生序列化），存在 SDK 升级失配风险 → W2 必须做官方 crc.py vs 独立实现的向量级对照 | [公开示例] | python/crc.py:22-25,135-152,185-216；python/r1_A7_low_level_example.py:105-108（收帧 CRC 校验）；cpp/r1_A7_wrist_swing_example.cpp:107-127（Crc32Core 内联实现）,195-198（收帧校验）,282（发帧计算）；cpp/crc.h:55-80（公共头文件内联版）（2026-11-16 取得） |
| G1-07 | 控制权生命周期（示例事实，非厂商承诺）：获取 = 发布前 MotionSwitcherClient（服务名 `motion_switcher`，RPC 话题按 `rt/api/<service>/request|response` 规则组装为 `rt/api/motion_switcher/request|response`；api id 1001 CheckMode/1002 SelectMode/1003 ReleaseMode/1004 SetSilent/1005 GetSilent）循环"查到占用→ReleaseMode→sleep 1-5s→重查"直到空闲；超时示例 5 s。失败处理：Python 示例 raise RuntimeError 退出，C++ 示例打印后继续（工程缺陷，本项目不采纳）；**双侧示例均无失败回滚（重新 SelectMode 恢复原模式）与退出释放路径（主线程 while-true 挂起）**——本项目 4.6 必须自行设计回滚与释放，不能照抄 | [公开示例] | python/r1_A7_low_level_example.py:71-85（获取循环）,73（SetTimeout 5.0）；python/motion_switcher_client.py:14-50（API 注册）；python/motion_switcher_api.py:16-20（api id 常量 1001-1005）、:4（服务名）；python/channel_name.py:13-21（RPC 话题组装规则）；cpp/r1_A7_wrist_swing_example.cpp:163-171,344；C++ 侧 b2 命名空间引入见 cpp 文件 L18（2026-11-16 取得；api id 1004/1005 与话题组装规则于修复轮 1 补证） |
| G1-08 | topic/domain/网卡：`rt/lowcmd`（pub，LowCmd_）、`rt/lowstate`（sub，LowState_）、`rt/secondary_imu`（sub，IMUState_，仅 C++ 示例订阅，Python 示例未订阅）；domain=0（`ChannelFactoryInitialize(0, <网卡>)`）；网卡为必传运行参数（示例以 argv 传入），CycloneDDS XML 绑定该网卡名（channel_config.py）；网络环境 = 123 网段（外置开发计算单元 192.168.123.164，unitree/123）。**500 Hz（control_dt_=0.002 s + CreateRecurrentThreadEx 2000 us）是示例事实，不是本项目频率承诺** | [指南]+[公开示例] | python/r1_A7_low_level_example.py:61,88,92,96-98,175；cpp/r1_A7_wrist_swing_example.cpp:20-22,155,160,182-183；python/channel.py:298-301；python/channel_config.py:1-25；architecture_description 页"外置计算单元 192.168.123.164"（2026-11-16 取得） |
| G1-09 | mode_machine 校验：`LowState_.mode_machine()` 为机型/模式标识（C++ 示例打印 "R1 type: <数值>"），命令侧 `LowCmd_.mode_machine` 须**回填最近一帧状态值**（双侧示例一致：先等首帧再启动发送线程）。**枚举值未公开**——冻结前驱动只做变化检测+记录，不做相等断言；R1-A7-D 正确回读值待实机 | [公开示例] | python/r1_A7_low_level_example.py:99-103（等 mode_machine 更新才启动）,111,127；cpp/r1_A7_wrist_swing_example.cpp:222-225,267-268（2026-11-16 取得） |
| G1-10 | 序号语义：hg LowCmd/LowState **无 go2 式 head/seq 字段**（G1-05）；`LowState_.tick:u32` 推测为帧计数，但**步长、回绕、上电重启复位行为未公开**——W2 前只能作接收新鲜度旁证，不得作唯一失效判据；本项目 4.4 的 session 隔离与 monotonic 接收时钟设计依据即此 | [公开示例]+未公开部分 | python/_LowState_.py:28（tick 字段）；两侧示例均未使用 tick 做校验（全文检索无 tick 消费点）（2026-11-16 取得） |
| G1-11 | 全机命令所有者与保持语义（**示例事实**）：官方 A7 示例每帧重写 16 槽（双臂 14 + 头 2）：`mode=1(Enable), q=0, dq=0, tau=0, kp=示例增益, kd=示例阻尼`；首段 2 s 目标从实测 q 线性插值到 0（防跳变，Python L139-143 / C++ L301-306）；代码注释原文 "Reset all controlled joints so inactive joints hold zero posture."（Python L129）。**其余 19 槽（0,1 底盘轮 / 2-11 / 12 升降 / 13 腰 / 14 / 31-34）示例完全不写**，保持默认构造全零（mode=0 Disable，default.py:218-219）——语义未定义。示例增益（肩 [100,100,100,100]→腕 [50,50,50]，臂 Kd=2，头 kp [50,10]/kd [2,0.1]）**是示例值不是安全参数** | [公开示例] | python/r1_A7_low_level_example.py:17-27,129-137,139-143；cpp/r1_A7_wrist_swing_example.cpp:69-80,287-297,301-306；python/default.py:209-210,218-219（默认全零）（2026-11-16 取得） |
| G1-12 | 官方 Python 指南页（developer/Python）存在且内容仅覆盖 Go2 示例（sport/low_level/遥控/摄像头），未提 R1/R1-A-D；R1-A-D 指南"快速开发"页渲染后无内容（暂无文章内容）。即：**R1-A7-D 的 Python 开发无官方文档级教程，仅有仓库示例**——区分"公开示例存在"与"官方文档支持声明"（后者见架构页一句"支持 Python"） | [指南] | https://support.unitree.com/home/zh/developer/Python （Tavily 抓取，2026-11-16 取得）；https://support.unitree.com/home/zh/R1-A-D_SDK_Development_Guide/quick_develop →"暂无文章内容"（2026-11-16 取得） |
| G1-13 | Python 官方 SDK 内置 `comm/motion_switcher`、`idl/unitree_hg`（含 LowCmd/LowState/MotorCmd/MotorState/IMUState/HandCmd 等）、`utils/crc.py`（含 hg 双向 CRC）——4.5 需要的传输、序列化、控制权组件在 Python 侧齐备，无需 C++ 桥 | [公开示例] | GitHub API tree master@814556d1（响应快照已存 evidence-4.1/github-meta/tree_python_master_full.json，375 条目 truncated=false）：unitree_sdk2py/comm/motion_switcher/、unitree_sdk2py/idl/unitree_hg/msg/dds_/_LowCmd_.py 等、unitree_sdk2py/utils/crc.py（2026-11-16 取得；快照落盘于修复轮 1） |
| G1-14 | 高层机载臂动作服务（`rt/api/arm/request|response`，api 7106-7113，错误码 7400 arm_sdk occupied / 7401 holding 须先 release）存在于 C++ 示例，属预设动作/示教回放通道，**非本项目连续控制路径**；官方注释警告勿用 G1 版客户端替换 R1 版（同名服务、动作 ID 与错误码不同） | [公开示例] | 既有审计 §6.1/§6.3 已固定（unitree_sdk2@aaf69cb5 example/r1/high_level/r1_arm_action_example.cpp）；本轮未推翻（2026-11-16 复核引用） |
| G1-15 | 实机支持与公开示例的区分声明：本轮全部证据为 [指南]+[公开示例]；**实物型号（A5 vs A7 版、智能/旗舰版）、固件版本、mode_machine 实际值、失联行为、单臂接管时对侧/腰/升降/底盘的厂商级保持策略均未经实机验证**——上述不得写入冻结值，manifest 对应字段 null | — | 本报告 §7；既有审计未确认项 #1/#5/#6/#7（2026-11-16） |

---

## 3. A5 wrapper 限制 ≠ 服务器限制：疑点闭环过程

既有审计（2026-11-15）结论："高层 `rt/arm_sdk` 封装只覆盖 A5 兼容 13 关节、不含 A7 特有腕俯仰/腕偏航，A7 全 7 关节需低层接口"——但当时未核实 Python 侧是否存在 A7 路径。本轮核验：

1. `unitree_sdk2@aaf69cb5`：`include/unitree/dds_wrapper/robots/r1/r1_pub.h` L28-42 `ArmSdk::JOINTS` 为 13 关节常量（左右臂各 5 + WaistYaw + HeadPitch/HeadYaw）——wrapper 层限制成立；同 commit `example/r1/low_level/r1_A7_wrist_swing_example.cpp` 提供低层 A7 全关节路径。
2. `unitree_sdk2_python@814556d1`（本轮新固定）：**无 r1 ArmSdk 高层封装**（`unitree_sdk2py/r1/` 只有 loco/），但 `example/r1/low_level/r1_A7_low_level_example.py` 存在且完整——Python 低层 A7 路径官方在案。
3. 结论：A7 走低层 `rt/lowcmd` 是 C++/Python 双语言一致的官方工程事实；"wrapper 只有 A5"确为封装层限制而非服务器能力限制（低层直接寻址 20/21/27/28 即可驱动 A7 腕俯仰/腕偏航）。

## 4. Python 支持矩阵与安装/检查命令

| 项 | 冻结值 | 备注 |
|---|---|---|
| 官方支持声明 | 架构页："开发接口 1. DDS，支持C++ 以及 Python"（2026-11-12 更新） | 文档级一句话；无 R1-A7-D Python 专属教程（G1-12） |
| 仓库/commit | `unitree_sdk2_python` master@`814556d15970dd2ecf1c9984e845ca02ab07e206` | 下划线命名 |
| 包名/版本 | `unitree_sdk2py` 1.0.1（setup.py）；PyPI 无同名页 | 以源码为准 |
| 依赖 | cyclonedds==0.10.2、numpy、opencv-python；Python ≥3.8 | 本项目 conda py3.11 满足版本下限，cyclonedds 版本钉死 |
| 平台 | Linux x86_64/aarch64（CRC .so）；非 Linux 回落纯 Python CRC | 部署主机未定（§7-Q10） |
| 安装命令 | `git clone https://github.com/unitreerobotics/unitree_sdk2_python.git && cd unitree_sdk2_python && pip3 install -e .` | cyclonedds 缺失时按官方 FAQ 源码编译 releases/0.10.x + `CYCLONEDDS_HOME`；**W2 时点经用户批准后执行** |
| 离线序列化检查 | 无网络主机上：构造 `unitree_hg_msg_dds__LowCmd_()` 填哨兵值 → `CRC().Crc(cmd)` 与独立参考实现（§5 算法）互验 → cyclonedds 序列化/反序列化往返比对全部 35 槽 | 具体测试脚本属 Task 4.5 `test_r1d_sdk_serialization.py`，本片仅冻结判据 |

## 5. DDS 契约冻结明细

（以下全部 [公开示例] 级；与 manifest `dds:` 节一一对应）

- **话题**：`rt/lowcmd`（发布）、`rt/lowstate`（订阅）、`rt/secondary_imu`（可选订阅，本项目默认不订）。`rt/arm_sdk` 不采用（G1-04）。
- **domain/网卡**：domain 0；网卡名运行参数传入，CycloneDDS XML `<NetworkInterface name=...>` 绑定（channel_config.py L1-14）；不传网卡时 autodetermine（channel_config.py L16-25，多网卡环境有选错风险，本项目要求显式传入）。
- **IDL**：见 G1-05（35 槽数组、字段顺序即 CDR 序列化顺序）。uint8×2 后有 2 字节对齐填充（CRC 打包格式 `'<2B2x'` 证实，crc.py L23）。
- **CRC**：见 G1-06。
- **mode_machine**：见 G1-09（回填最近状态值；枚举待实机）。
- **序号**：见 G1-10（无 seq 头；tick 语义待定）。
- **500 Hz**：见 G1-08（示例事实，不作承诺；4.8 的周期指标阈值由现场配置提供）。

## 6. 逐部件命令所有者与保持语义

（基线 = 官方 A7 示例行为 G1-11；**厂商级语义未公开，最终裁定属 G4 现场确认**）

| 部件 | 槽位 | 命令所有者（本项目） | 保持/忽略语义（示例事实） | 待确认 |
|---|---|---|---|---|
| 所选臂 | 左 15–21 / 右 22–28 | 本项目 lowcmd（MPC 目标） | 目标由控制器给出；示例对非摆动关节写 q=0+kp 保持、首 2 s 从实测 q 插值 | 击球侧（§7-Q 沿用阶段 2 #2）；单臂接管时对侧槽位策略 |
| 另一臂 | 同上（对侧 7 槽） | 本项目 lowcmd（示例一并写） | 同上（q=0 位保持，非"复制反馈 q"） | 厂商是否要求单臂模式显式忽略对侧 |
| 头 | 29(pitch)/30(yaw) | 本项目 lowcmd（示例一并写） | q=0 + kp[50,10]/kd[2,0.1] | 同上 |
| 腰 | 13 | **未定**（示例不写） | 默认 mode=0 全零，语义未定义 | 由谁保持；机械锁 vs 控制保持（阶段 2 遗留 #5） |
| 升降 | 12 | **未定**（示例不写） | 同上 | 指南有独立"底盘与立柱控制"页但本机抓取为空（§7-Q7）；实验前设定+锁定方式 |
| 底盘 | 0(左轮)/1(右轮) | **未定**（示例不写） | 同上 | 独立服务协议未知 |
| 手/夹爪 | 不在 LowCmd_ 内 | 不控制 | hg 有独立 `HandCmd_[7]`（Dex3 口径）但与 LowCmd 分离；R1-D Dex1 二指协议未见公开 | 是否装夹爪、协议通道 |
| 空槽 | 2–11、14、31–34 | 不写 | 无文档 | — |

## 7. 厂家/现场问询清单（上机前必须回答）

| # | 问题 | 影响面 |
|---|---|---|
| Q1 | `mode_machine` 枚举：R1-A7-D（智能/旗舰两版）正确回读值？与 A5/带腿 R1 如何区分？ | 4.4 机型校验、G3 只读核对 |
| Q2 | `rt/lowcmd` 停发或断网后：臂/头/腰/升降/底盘各部件行为？有无接收端命令超时（看门狗）及其阈值？ | 4.6 看门狗、失联报告、G4 |
| Q3 | 单臂接管时，对侧臂/头/腰/升降/底盘槽位的厂商正确保持策略是什么？官方示例的"q=0+kp 零位保持 + 2s 插值"是否即推荐做法？ | 4.3 全机命令策略 |
| Q4 | MotionSwitcher `ReleaseMode` 后机载服务如何恢复？需 `SelectMode` 重选哪个模式名（可选清单）？ | 4.6 释放/回滚 |
| Q5 | `LowState_.tick` 的计数步长、回绕与控制器重启复位行为？ | 4.4 序号新鲜度 |
| Q6 | hg `LowCmd_`（1004 B）/`LowState_`（2092 B）序列化长度与 CRC 覆盖范围在当前固件上与 SDK 814556d1/aaf69cb5 是否一致（固件-IDL 版本配套表）？ | 4.5 序列化对照 |
| Q7 | "底盘与立柱控制"独立服务的协议（官方页面存在但渲染为空）：topic/IDL/安全语义？ | 阶段 6 前置 |
| Q8 | 夹爪（Dex1 二指）控制通道与 IDL？本机是否装夹爪？ | 工具安装 |
| Q9 | 500 Hz 低层循环在 Python（及所选部署主机）上的可达成性与官方建议周期？ | 4.8/4.10 周期阈值 |
| Q10 | 本项目部署主机：外置计算单元（192.168.123.164，unitree/123）还是外部 123 网段主机？OS/架构/网卡名？ | 4.5 传输配置 |
| Q11 | 实物铭牌/序列号/固件版本（确认 R1-A7-D 智能版 vs 旗舰版） | G3 第一项（沿用阶段 2 #1） |

## 8. 证据文件 SHA256（取证目录 `evidence-4.1/`）

关键文件（前 16 位，完整值可现场 `sha256sum` 复核）：

| 文件 | SHA256（前 16 hex） |
|---|---|
| python/r1_A7_low_level_example.py | 8f934a232840e810 |
| python/r1_A5_low_level_example.py | aef8d2c7f77122d3 |
| python/_LowCmd_.py | a23cec88e4bf7822 |
| python/_LowState_.py | 97f746a693aff14a |
| python/crc.py | b95a530423f72c5a |
| python/channel.py | 8a58eea2bc6bb879 |
| python/channel_name.py（修复轮 1 补） | a978ad0fe4e548a2 |
| python/motion_switcher_api.py（修复轮 1 补） | 933210a39b46f930 |
| python/rpc_client_base.py（修复轮 1 补） | 2a815a5c798be95d |
| python/setup.py | f2c73b2981ee918d |
| cpp/r1_A7_wrist_swing_example.cpp | 0a9798797aa51263 |
| cpp/r1_pub.h | 133abafefb11756a |
| cpp/defines.h | 57d923f806b72609 |
| cpp/crc.h | 3175e111c6f8e68b |
| cpp/LowCmd_.hpp | 3e2760c3475c7a7a |
| cpp/LowState_.hpp | e01be29667100a6f |
| official-guide/*.md（4 份快照） | 见目录 |
| github-meta/*.json（6 份 GitHub API 响应快照，修复轮 1 补：两仓库 repo 元数据、连字符仓 404、双仓 commits top3、Python 仓全树 375 条目） | 见目录 |

## 9. 与施工计划"跨任务接口契约（4.2 冻结）"的对齐

- **无字段级变更请求**。已冻结的 `R1DStateFrame`（q/dq、sequence: int|None、mode_machine、session_id 等）与 hg LowState_ 适配良好：`tick` 可映射 sequence（语义待 Q5，None 语义已预留）；`R1DCommandFrame` 的 q/dq/kp/kd/tau/motor_mode 35 槽与 hg MotorCmd_ 字段一一对应，CRC 归序列化层符合本报告结论。
- 两点提示（非变更，供 4.2 实现参考）：(1) `R1DCommandFrame` 需增补 `mode_pr: int` 透传字段（hg LowCmd_ 首字段，低层路径示例写 0/PR；不补则适配器须造默认值，违反"禁止神秘默认值"约束）——该字段在计划契约的"4.1 确认的其他必要头字段"预留范围内，属填充而非结构变更；(2) `mode_machine` 回填语义（取最近状态帧值）建议写入 4.3 策略而非类型层。

## 10. 自审（局限）

- 官方指南三关键页经渲染抓取成功（更新时间 2026-11-12/13）；"快速开发""DDS通信接口"两页渲染后无内容，底盘与立柱控制页未能取证（slug 未知）——Q7 悬置。
- 全部结论不含实机证据；示例行为（保持策略、500 Hz、失败处理）为工程参考基线，G4 现场确认前不得写成厂商承诺。
- PyPI 404 可能受网络环境影响，已用源码 setup.py 交叉（包版本以源码为准的裁定不受影响）。
- 序列化长度 1004/2092 B 与官方 crc.py 注释（L22/L24）及打包格式实算一致（修复轮 1 更正：初版误写 524）；W2 安装后实测复核（Q6）。

---

## 11. 修复轮 1 记录（2026-11-16，评审驱动）

评审裁决：spec 合规 FAIL（单点证据矛盾）+ 1 Important + 3 Minor。修复内容：

1. **[Important] hg LowCmd 序列化长度 524 → 1004 B**：G1-05、Q6、§10 自审及 manifest `lowcmd_layout.serialized_length_bytes` / `crc.python_impl_note` 共 5 处更正。依据 = `evidence-4.1/python/crc.py` L22 官方注释 `#size 1004` + L23 格式串 `'<2B2x' + 'B3x5fI'*35 + '5I'` 实算 4+35×28+20=1004；初版错误根因 = 我误按 MotorCmd 槽 24 B（漏算 reserve 后对齐）推算且未对照官方注释，且算式本身与结论互不自洽（评审指出）。LowState 2092 B 经复核正确（L24 注释 + `'<2I2B2xI'+'13fh2x'+'B3x4f2hf7I'*35+'40B5I'` 实算 16+56+35×56+60=2092），未动。
2. **[Minor-1] MotionSwitcher api id 1004/1005 落证**：补取证 `unitree_sdk2py/comm/motion_switcher/motion_switcher_api.py`（SHA256 933210a3…）到 evidence-4.1/python/，L16-20 常量定义 1001-1005、L4 服务名 "motion_switcher"；G1-07 与 manifest `control.acquisition` 引用其行号。连带修正：RPC 话题名由无据的 "rt/motion_switcher" 更正为按组装规则 `rt/api/motion_switcher/request|response`（新取证 `channel_name.py` L13-21，SHA256 a978ad0f…）。
3. **[Minor-2] GitHub API 元数据落盘**：`evidence-4.1/github-meta/` 新增 6 份响应快照（两仓 repo 元数据、连字符仓 404、双仓 commits top3、Python 仓全树 375 条目 truncated=false）；G1-02/G1-13 与 §1.1 渠道表引用补快照路径。
4. **[Minor-3] 引用路径统一**：报告 G1-03/G1-05/G1-06 中 `python/utils/crc.py` 统一为取证目录实际落盘路径 `python/crc.py`（平铺）；引用字段处不再有 `python/utils/` 写法。

复检：grep 命中的 `524` 与 `python/utils/` 全部位于本节修复说明文字（变更留档），冻结值/引用字段处无残留；无裸 `rt/motion_switcher`（仅存 `rt/api/motion_switcher`）；manifest YAML 解析通过（lowcmd=1004 / lowstate=2092）；1004/2092 与官方注释 crc.py L22/L24 逐一对齐。
