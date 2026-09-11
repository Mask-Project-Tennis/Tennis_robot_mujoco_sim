# Skill: 论文撰写（paper_writing）

## 目的
指导 Agent 撰写 ICRA 会议格式的学术论文，先中文后翻译英文。
调用时机：实验数据就绪后，开始组织论文内容时。

## 目标会议
**ICRA 2027（Seoul）**，截止 2026-09-15 11:59 PST（Paperplaza）；RAL 作为扩展版后备去向
- 格式：IEEE 双栏会议版，**6 页正文 + 参考文献，最多 +2 页（付费）**
- 语言：英文（先中文草稿，后翻译）
- 模板：IEEEtran.cls（`conference` 选项，main.tex 已就位）
- 排期与依赖：见 `paper/planning/schedule.md`；论文配置：`paper/planning/00-paper-config.md`（须经用户确认）

## 论文目录结构

```
paper/                         # 独立 git 仓库（主仓 .gitignore 已排除）
├── main.tex                   # 主文件（ICRA conference 版）
├── references.bib             # 占位；定稿时按实际引用从 Zotero/arXiv 导出生成
├── build.sh                   # 构建脚本（latexmk；full 模式自动装缺包）
├── .latexmkrc                 # latexmk 配置（pdflatex+bibtex，输出 build/）
├── figures/                   # 图片（由 figure_generation skill 生成）
│   ├── fig1_system_overview.pdf
│   ├── fig2_algorithm_flowchart.pdf
│   ├── fig3_tube_corridor.pdf
│   ├── fig4_joint_trajectory.pdf
│   ├── fig5_hit_rate_vs_speed.pdf
│   ├── fig6_tube_robustness.pdf
│   ├── fig7_realtime_performance.pdf
│   ├── fig8_tube_diagnostic.pdf
│   └── table_data/            # LaTeX 表格数据
├── sections/                  # 英文各节（成稿）
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── problem_formulation.tex
│   ├── method.tex
│   ├── experiments.tex
│   ├── results.tex
│   └── conclusion.tex
├── sections_zh/               # 中文草稿（唯一事实源，markdown + LaTeX 公式）
│   └── （与 sections/ 同名 .md）
├── planning/                  # 学术流水线中间产物
│   ├── 00-paper-config.md     #   论文配置记录（academic-paper Phase 0）
│   ├── schedule.md            #   冲刺排期
│   ├── 02-outline-evidence.md #   大纲 + 证据映射（Phase 2）
│   └── 03-argument-blueprint.md # 论证蓝图（Phase 3）
├── discussion/                # 讨论与决策记录
├── reviews/                   # 模拟评审 / revision roadmap（round-N 子目录）
├── notes/                     # 文献调研笔记（43 篇 + 综述 + raw/），已对齐 Zotero 合集 AAYPCPGP
└── build/                     # 构建产物（gitignore）
```

## LaTeX 工程模板

### main.tex

```latex
\documentclass[letterpaper, 10pt, conference]{IEEEtran}
\IEEEoverridecommandlockouts

\usepackage{cite}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{bm}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage[caption=false,font=footnotesize]{subfig}
\usepackage[hidelinks]{hyperref}
\usepackage[capitalize]{cleveref}

\title{Tube-Based Robust MPC with iLQR for Dynamic\\Tennis Hitting with a Robotic Arm}

\author{
\IEEEauthorblockN{Author Name}
\IEEEauthorblockA{Affiliation\\
Email: author@example.com}
}

\begin{document}
\maketitle

\input{sections/abstract}
\input{sections/introduction}
\input{sections/related_work}
\input{sections/problem_formulation}
\input{sections/method}
\input{sections/experiments}
\input{sections/results}
\input{sections/conclusion}

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
```

### 构建命令

```bash
./build.sh        # latexmk 增量构建 → build/main.pdf
./build.sh full   # 缺宏包时自动 tlmgr install（TUNA 镜像）再构建
./build.sh clean  # 清理辅助文件
```

TeX Live 2026 用户态安装于 `~/.local/texlive/2026`（PATH 已写入 `~/.zshrc`）。

## 核心贡献点

基于代码实际能力，论文的核心贡献为：

### 贡献 1：Tube-based Spatial Corridor for Robust Hitting
- 空间走廊式 Tube 代价，不绑定"第 k 步必须到 p_ball(k)"的时间-空间对应
- v8 实现**仅铰链位置项**：`½·s_k·Q_p_tube·max(0, ‖P_⊥(p_ee−p_ball,k)‖ − r_racket)²`
  （`src/ilqt/tube_cost.py::_compute_tube_cost_at_k`）；半宽 = 拍半径 r_racket = 0.12 m，
  走廊内零代价、沿球轨迹线方向的超前/滞后不惩罚
- 时间窗 softmin 终端（β=5.0，±50ms 候选窗）：候选代价含位置/速度/法向三项，
  速度方向与法向对齐**由终端承担，不在走廊项内**
- ⚠️ "不确定性管道 σ(t) = σ₀ + σᵥ·t + σₐ·t²" **当前代码未实现**（2026-09-11 核对源码），
  禁止写入论文；写作红线清单见 `paper/discussion/2026-09-11_*` §9

**中文草稿要点**：
> 传统的 tube MPC 方法通常要求系统在特定时刻到达特定状态，形成时间-空间的一一对应。
> 然而，网球击打场景中，球的到达时间存在预测误差，且球拍有一定的"甜区"面积。
> 本文提出一种空间走廊式的 tube 代价，允许球拍在候选时间窗口内的任意时刻击球，
> 只要球拍保持在球轨迹线附近的空间走廊内。这大幅提升了对时间预测误差的鲁棒性。

### 贡献 2：Multi-Layer Safety Filter for Real-World Deployment
- 关节约束：位置/速度/加速度/力矩四重限制
- TCP 速度硬限制：通过 monkey-patch 安全滤波器实现
- X 平面墙：臂不越过身体中线（物理层面 PD 推回）
- 逐步安全滤波：尝试多个衰减系数 (β=1.0→0.8→...→0.0)，找到最大可行控制
- 弹性反弹模型：球拍-球碰撞后的物理正确反弹

**中文草稿要点**：
> 为确保真实部署安全性，我们设计了多层安全滤波架构。
> 每步控制经过三层约束检查：X 平面墙预判、安全滤波器关节约束、执行后 PD 推回。
> 安全滤波器不简单地裁剪力矩，而是尝试多个衰减系数，找到满足所有约束的最大力矩。

### 贡献 3：Real-Time MPC with Asynchronous Replanning
- 异步重规划：后台线程求解 iLQR，主线程继续执行 buffer
- Buffer 机制：replan_interval 步的控制缓存
- 分阶段迭代策略（`MPCConfig`: first_plan_iters=30, near_plan_iters=5, max_iter_per_plan=10,
  far_threshold=50）：首次规划 30 次迭代，far 阶段（k_hit>50）**仅 JT 控制、零 iLQR 迭代**，
  near 阶段 5 次迭代 + hard_constraints
- 近距 buffer 扩展：k_hit ≤ 30 时 buffer 翻倍，减少重规划频率

**中文草稿要点**：
> 实测（exp18，空闲机器）：同步稳态重规划（far 0 迭代 / near 5 迭代）中位 4ms、
> p95 33ms、上限 41ms，装入 150ms 重规划预算余量充足；首次规划（30 迭代）约 920ms
> 发生于启动阶段，需提前量或离线预热。异步模式求解耗时同量级，但主循环停顿
> 由 1.0% 的步超周期（max 41.9ms）降为 0%（max 2.23ms）。

### 贡献 4：Experimental Validation on RM-65B Constraints
> **数字基准（2026-09-11 重跑，旧数字一律作废）**：见 `docs/experiments/reports/2026-09-1*_exp1*.md`
> 与 `paper/planning/02-outline-evidence.md`；确认后的贡献措辞见 `paper/planning/00-paper-config.md` §2

- 球速能力（exp15，200 seeds）：7/8/9/12/15 m/s → 85.4/94.9/95.9/85.7/60.5%（峰值 8-9 m/s）
- 真机限位（exp16，200 seeds）：TCP 1.0 vs 1.8 m/s → 36.4% vs 85.4%（v_racket 0.87 vs 1.48）
- 鲁棒性（exp17b/17e）：9 m/s 下空间 0.2m −19.5pp、时间 100ms −4.2pp；full vs none 基线 +16.3pp
- 机制消融（exp17d/17f/G）：四档 full/tube_only/softmin_only/none；softmin 主导标称，走廊贡献见 exp17g

---

## 各节撰写指引

> **页数预算说明**：下列各节页数按 8 页版标注；ICRA 6+2 版按 **×0.75 缩放**
> （Intro 0.9 / Related 0.5 / Problem 0.6 / Method 1.7 / Exp 1.1 / Results 0.9 / Concl 0.3），
> 大纲阶段（`planning/02-outline-evidence.md`）再按实际内容精调。

### I. Abstract（中文草稿）

**字数**：150-250 词英文（对应中文 300-500 字）

**结构**：
1. 一句话背景：机器人动态击打是...的挑战
2. 一句话问题：现有方法在...方面的不足
3. 两句话方法：我们提出...，包含三个关键组件
4. 两句话实验：在 RM-65B...约束下验证，结果表明...

**中文草稿模板**：
```
动态球类击打是机器人操作中的重要挑战，要求在毫秒级精度下完成
轨迹规划、力矩优化和安全约束的协同。本文提出一种基于
MPC+iLQR+Tube的鲁棒网球击打框架，用于 RM-65B 六自由度机械臂。
核心创新是空间走廊式 Tube 代价函数，不绑定时间-空间对应，
允许球拍在候选时间窗口内灵活击球。配合多层安全滤波器和异步重规划架构，
系统在真实机械臂约束下实现了 200Hz 控制频率的实时控制。
在 MuJoCo 仿真中（200 seeds 重跑），系统在 9 m/s 球速下达到 95.9% 命中率，
在真机 TCP 限位 1.0 m/s 下为 36.4%（仿真默认 1.8 m/s 为 85.4%）；
鲁棒性网格显示空间 0.2 m 扰动损失 19.5pp、时间 100 ms 扰动仅损失 4.2pp，
两种机制相对点目标 iLQR 基线提升 6.6-7.1pp。
```

### II. Introduction（1.2 页）

**段落结构**（6-7 段）：

1. **背景与动机**（1 段）
   - 动态球类运动是机器人领域的挑战性任务
   - 应用场景：体育训练、娱乐机器人、快速反应操作
   - 核心挑战：预测不确定性 + 执行器约束 + 实时性要求

2. **问题陈述**（1 段）
   - 给定网球飞行轨迹预测（含不确定性），计算最优挥拍轨迹
   - 需同时满足：击打精度、关节约束、TCP 安全、实时性

3. **已有方法的局限**（1-2 段）
   - 传统轨迹跟踪方法（PD/MPC）缺乏对预测不确定性的鲁棒性
   - 强化学习方法需要大量训练，难以加入硬约束
   - 经典 iLQR/MPC 缺乏 tube 机制，对时序误差敏感

4. **本文方法概述**（1 段）
   - MPC+iLQR+Tube 三层框架
   - 空间走廊式 tube 代价
   - 多层安全滤波

5. **贡献列表**（1 段，4 点，加粗编号）
   - **C1**: Tube-based spatial corridor cost
   - **C2**: Multi-layer safety filter architecture
   - **C3**: Real-time MPC with async replanning
   - **C4**: Experimental validation on RM-65B constraints

6. **论文结构**（1 段，简短）

### III. Related Work（0.6 页）

**组织方式**：按主题分三段，不需要子标题

1. **Trajectory Optimization for Robotic Hitting**（1 段）
   - iLQR/DDP 方法在机器人操控中的应用
   - 网球/乒乓球机器人的已有工作
   - 引用：Todorov iLQR、Sentis DDP、桌球机器人相关

2. **Robust MPC and Tube-based Methods**（1 段）
   - Tube MPC 理论（Mayne et al.）
   - 不确定性管道的构建方法
   - 本文与传统 tube MPC 的区别：空间走廊 vs 时间锁定

3. **Safety-Critical Robot Control**（1 段）
   - 控制屏障函数 (CBF)、安全滤波器
   - 关节约束下的轨迹规划
   - 本文的安全滤波与 CBF 的区别

### IV. Problem Formulation（0.8 页）

**内容**：

1. **机器人模型**（0.3 页）
   - RM-65B 6-DOF 参数表
   - 状态空间：x = [q, q̇] ∈ R¹²
   - 控制空间：u = τ ∈ R⁶
   - 动力学：ẋ = f(x, u)

2. **网球场景**（0.2 页）
   - 球轨迹预测：抛物线模型 p_ball(t) = p₀ + v₀t + ½gt²
   - 击打点搜索：p_hit = argmin ‖p_ee - p_ball‖ 满足可达性
   - 期望击打速度：v_hit = v_racket · d̂

3. **约束定义**（0.3 页）
   - 关节约束：q_min ≤ q ≤ q_max, |q̇| ≤ q̇_max, |q̈| ≤ q̈_max
   - TCP 约束：‖v_tcp‖ ≤ v_tcp_max
   - 空间约束：x_body ≤ x_wall（臂不越中线）

**关键公式**：
```
min_{u₀:N-1}  l_N(x_N) + Σ l_k(x_k, u_k)
s.t.  x_{k+1} = f(x_k, u_k)
      u_k ∈ U, x_k ∈ X
```

### V. Method（2.2 页，最核心的一节）

**子节结构**：

#### V-A. iLQR Trajectory Optimization（0.5 页）

**内容要点**：
- 后向传递：Riccati 递推计算增益 K_k, k_k
- 前向传递：线搜索更新轨迹
- 动力学线性化：A_k = ∂f/∂x, B_k = ∂f/∂u（解析/有限差分）
- 代价函数：终端代价 l_N + 运行代价 l_k

**关键公式**：
```
Q_x = l_x + f_x^T V'_x
Q_u = l_u + f_u^T V'_x
Q_xx = l_xx + f_x^T V'_xx f_x
Q_uu = l_uu + f_u^T V'_xx f_u
K_k = -Q_uu^{-1} Q_ux
k_k = -Q_uu^{-1} Q_u
```

#### V-B. Tube-based Spatial Corridor Cost（0.7 页）

**内容要点**：
- 候选击球窗口搜索：以 best_k 为中心，窗口半宽 window_half_ms（默认 50ms）
- 球轨迹线方向：d_ball = normalize(Σ w_i · v_ball_i)
- 垂直投影矩阵：P_perp = I - d_ball · d_ball^T
- **v8 走廊代价（唯一运行项）**：垂直偏离的 hinge loss，走廊半宽 = 拍半径 r_racket
- 速度方向对齐、法向对齐**移入终端 softmin 候选代价**（见 `sections_zh/method.md` 式 4）

**关键公式**（对齐 `src/ilqt/tube_cost.py::_compute_tube_cost_at_k`）：
```
l_corr(x_k) = 0.5 · s_k · Q_p_tube · [max(0, ‖P_perp · (p_ee,k - p_ball,k)‖ - r_racket)]²
             r_racket = 0.12 m（拍半径, 固定）
```
> ⚠️ 旧版曾含 `Q_v · ‖P_perp·v_ee‖²` 与 `Q_n·(1-n·n_des)` 两项及 `σ_max` 项，
> 当前 v8 实现已移除（速度/法向由终端 softmin 承担；σ(t) 管道未实现）

**与传统 tube MPC 的关键区别**：
- 传统：在时间 k 必须到达状态 x_ref(k) 的 tube 内
- 本文：在候选窗口 [k_min, k_max] 内任意时刻，只要在空间走廊内
- 优势：容忍时间预测误差（实测 9 m/s 下 ±100ms 扰动仅损失 4.2pp，exp17b）

#### V-C. MPC Framework with Replanning（0.5 页）

**内容要点**：
- MPC 外循环：每 replan_interval 步重规划
- 异步重规划：后台线程求解 iLQR，主线程执行 buffer
- 分阶段策略：far/near 不同迭代次数和约束严格度
- Warm-start：雅可比转置法 / 后摆五次多项式

#### V-D. Multi-Layer Safety Filter（0.5 页）

**内容要点**：
- 层次结构：X 平面墙预判 → 安全滤波器 → 执行后 PD 推回
- 安全滤波器：逐步衰减 β = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
- 关节约束检查：位置/速度/加速度/力矩四重
- TCP 速度硬限制：‖J_p · q̇‖ ≤ v_tcp_max

### VI. Experiments（1.5 页）

**内容**：

#### VI-A. Setup（0.3 页）
- 机器人：RM-65B 双臂（12 DOF；右臂 6 DOF 驱动 + 左臂保持零位）
- 仿真：MuJoCo，dt=5ms，200Hz
- 球拍：半径 120mm，连杆垂直法兰
- 球：从 8m 外发球区随机位置飞来

#### VI-B. E1: 球速能力与真机限位（exp15 + exp16，0.4 页）
- 球速 7-15 m/s × 200 seeds；真机 TCP 1.0 vs 仿真 1.8 m/s × 200 seeds
- 结果引用 Fig.5（命中率-球速双线）

#### VI-C. E2: 四档机制消融（exp17d + exp17a/17b 基线格，0.3 页）
- full / tube_only / softmin_only / none × 7/9/12 m/s
- 结果引用 Fig.6(a) 柱状 + Table II

#### VI-D. E3: 鲁棒曲面 $(\tau,\rho)$（exp17b/17e/17f/17g，0.3 页）
- 时间 [0,10,25,50,100]ms × 空间 [0,0.05,0.1,0.2]m × 四档；@9 主网格 + @7 复证
- s=0.2 角点高 seeds（3000/格）做走廊独立贡献显著性
- 结果引用 Fig.6(b) 曲面/热图（核心图）

#### VI-E. E4: 感知退化（exp17a + exp17c，0.2 页）
- 噪声 σp × KF × ablation × 球速；观测频率 200→10 Hz
- 结果引用 Fig.8 + Table III

#### VI-F. E5: 实时预算（exp18，0.2 页）
- 30 sync + 10 async episodes：三段式求解耗时 + 主循环停顿对比
- 结果引用 Fig.7

> 图资产来源：`experiment_data/exp18_fig_assets/`（4 类轨迹 NPZ + timing.json）

### VII. Results and Discussion（1.2 页）

**内容**：

#### VII-A. Quantitative Results（0.4 页）
- Table I 数据分析
- 关键发现：TCP 约束是比关节约束更严格的瓶颈

#### VII-B. Tube Ablation Analysis（0.3 页）
- Table II 数据分析
- Tube 代价占比的最优区间
- 窗口大小的敏感度

#### VII-C. Real-Time Analysis（0.3 页）
- Fig.7 数据分析
- 重规划预算满足性
- 真实机器人预估延迟

#### VII-D. Discussion（0.2 页）
- 仿真到真实部署的差距
- TCP 限制是瓶颈（不是关节速度）
- 局限性：球轨迹模型简化、无空气阻力

### VIII. Conclusion（0.5 页）

**结构**：
1. 一句话总结方法和核心结果
2. 四点关键发现
3. 未来工作：真实 RM-65B 部署、视觉感知集成、多球连续击打

---

## 中英文撰写流程

### 步骤 1：中文技术草稿
- 在 `paper/sections_zh/` 下逐节撰写中文内容
- 使用 markdown 格式，嵌入 LaTeX 公式
- 参考上面的各节撰写指引
- 优先写 Method 节（最核心），再写 Experiments/Results，最后 Introduction/Related Work

### 步骤 2：公式排版
- 将中文草稿中的公式转为 LaTeX 格式
- 确保符号一致性（见下方符号表）
- 公式编号从 (1) 开始连续

### 步骤 3：翻译为英文
- 逐节翻译，保留 LaTeX 公式不变
- 注意 IEEE 的时态习惯：Method 用一般现在时，Experiments 用过去时
- 术语使用 IEEE 社区惯用表达

### 步骤 4：整合到 LaTeX
- 将翻译后的内容填入 `paper/sections/*.tex`
- 插入图表的 `\includegraphics` 和 `\input{table}`
- 编译检查：`cd paper && ./build.sh`

---

## 符号表

论文中使用的数学符号必须全文一致：

| 符号 | 含义 | LaTeX |
|------|------|-------|
| x | 状态向量 [q, q̇] | `\bm{x}` |
| u | 控制向量（力矩） | `\bm{u}` |
| q | 关节角度向量 | `\bm{q}` |
| q̇ | 关节角速度向量 | `\dot{\bm{q}}` |
| τ | 关节力矩向量 | `\bm{\tau}` |
| p_ee | 末端执行器位置 | `\bm{p}_{ee}` |
| v_ee | 末端执行器速度 | `\bm{v}_{ee}` |
| n_rack | 球拍法向量 | `\bm{n}_{rack}` |
| p_hit | 击打目标位置 | `\bm{p}_{hit}` |
| v_hit | 期望击打速度 | `\bm{v}_{hit}` |
| d_ball | 球轨迹线方向 | `\hat{\bm{d}}_{ball}` |
| P_perp | 垂直投影矩阵 | `\bm{P}_\perp` |
| σ(t) | 不确定性半径 | `\sigma(t)` |
| K_k | 反馈增益矩阵 | `\bm{K}_k` |
| k_k | 前馈增益向量 | `\bm{k}_k` |
| Q_p, Q_v, Q_n | 代价权重矩阵 | `\bm{Q}_p, \bm{Q}_v, Q_n` |
| R | 控制代价权重 | `R` |
| J_p | 位置雅可比矩阵 | `\bm{J}_p` |
| N | 规划时间步数 | `N` |
| k | 当前时间步索引 | `k` |

---

## 参考文献策略

**`paper/notes/` 的文献笔记是中间手稿，不假定全部引用。** 定稿阶段才生成正式 `references.bib`：
按正文实际 `\cite` 的条目，从 Zotero 合集「网球机器人」（Key: `AAYPCPGP`，已与 notes/ 全部对齐并去重）
导出 BibTeX，arXiv 条目可用 arxiv MCP `export_citations` 导出权威元数据。禁止手写编造条目。

### 必引文献（指示性，最终以 Zotero 导出为准）

```bibtex
@article{todorov2005,
  title={A generalized iterative {LQG} method for locally-optimal feedback control of constrained nonlinear stochastic systems},
  author={Todorov, Emanuel and Li, Weiwei},
  journal={Proc. ACC},
  year={2005}
}

@article{mayne2011,
  title={Tube-based robust model predictive control},
  author={Mayne, David Q and Kerrigan, Eric C and Falugi, Paola},
  journal={Automatica},
  year={2011}
}

@article{tassa2012,
  title={Synthesis and stabilization of complex behaviors through online trajectory optimization},
  author={Tassa, Yuval and Erez, Tom and Todorov, Emanuel},
  journal={Proc. IROS},
  year={2012}
}

% RM-65B 机器人文档
% MuJoCo 仿真引擎
% 网球机器人相关（如有）
% CBF 安全控制（如有）
% iLQR 在机器人操控中的应用（如有）
```

### 文献搜索建议
- Zotero 合集 `AAYPCPGP` 与 `paper/notes/`（43 篇结构化笔记，头部带 DOI/arXiv ID）
- 笔记的综合分析：`paper/notes/线约束iLQR方案-文献对比分析.md`（核心对比对象与 Devil's Advocate）
- 检查已有文档 `docs/rm65_tennis_report.md` 中的引用列表

---

## 写作注意事项

### ICRA 审稿偏好（单盲，6+2 页）
1. **方法创新性**：明确说明与传统方法的区别（特别是 tube 的空间走廊设计）
2. **实验充分性**：多组对比实验 + 消融 + 统计显著性
3. **可复现性**：参数完整公开，代码开源
4. **实际价值**：强调真实 RM-65B 约束下的验证结果

### 常见退稿原因（需避免）
- 方法贡献不够清晰（被评价为"工程实现"而非"研究贡献"）
- 缺少基线对比（应至少与 PD 跟踪和 Jacobian 转矩对比）
- 仿真结果无法延伸到真实机器人（需讨论 Sim-to-Real gap）
- 实验设置不够公平（需固定随机种子、报告统计量）

### 翻译注意事项
- "代价函数" → "cost function"（不是 "price function"）
- "前向传递" → "forward pass"（不是 "forward transmission"）
- "安全滤波器" → "safety filter"（不是 "security filter"）
- "不确定性管道" → "uncertainty tube" 或 "uncertainty envelope"
- "空间走廊" → "spatial corridor"
- "球拍甜区" → "racket sweet spot"
- "命中率" → "hit rate" 或 "hitting success rate"
