# paper/ —— ICRA 2027 论文工作区

独立 git 仓库（主仓 `/data/fxy/research/Tennis_robot_mujoco_sim` 的 `.gitignore:35` 已排除本目录），
存放 ICRA 2027 投稿的全部 LaTeX 源、图表、中文草稿与过程记录。

> **论文标题以 `main.tex` 为准**：*Beyond Point Targets: Spatial-Corridor and Time-Window
> Costs for Robust Robotic Tennis Hitting*。"Tube MPC" 措辞已因概念边界澄清而删除
> （`planning/00-paper-config.md` 记录的 T2 旧标题为历史版本，引用时以 `main.tex` 为准）。

| 项目 | 现状（2026-09-13） |
|------|-------------------|
| 目标会议 / 截止 | ICRA 2027（Seoul）；**2026-09-15 23:59 PST** 经 Paperplaza 提交（用户操作） |
| 页数预算 | **8 页硬上限**（正文 + 图表 + 参考文献全部计入），无付费加页 |
| 模板 | IEEEtran `conference`，letterpaper 10 pt 双栏；`\pdfminorversion=4`（PaperPlaza 合规）；**PaperCept/RAS 打印边距**（首页顶 1 in、其余各边 0.75 in）由 `\documentclass` 前的 `\CLASSINPUT*` 宏设置（chat19，见 `notes/chat19-revision-tracking.md`） |
| 章节状态 | 中文草稿 8/8（`sections_zh/`，唯一事实源，2026-09-12 冻结）→ 英文成稿 8/8（`sections/`） |
| 引用 | `references.bib` 17 条；只从 Zotero 合集 `AAYPCPGP` / arXiv / OpenAlex-CrossRef 导出（逐条 DOI 核验，禁止手写条目） |
| 评审方式 | **双匿名**：正文与 PDF 元数据均不得出现作者/单位/致谢；作者信息仅在 Paperplaza 填写 |
| 提交产物 | 仓库根 `main.pdf`（`build/main.pdf` 的副本，8 页；字节数随修订变化） |
| 异地备份 | GitHub 私有仓库 `Nothing-ray/tennis-paper-icra2027`（经 gh-proxy 通道，见文末「备份与恢复」） |

## 目录

| 目录/文件 | 用途 |
|-----------|------|
| `main.tex` | 主文件（标题、浮体容差参数、双盲约束、hyperref draft 模式的注释都在此） |
| `sections/` | 英文各节 .tex（8/8 成稿；每篇首行注明转写来源与修订轮次） |
| `sections_zh/` | 中文草稿 .md（**先中文后英文**，唯一事实源，已冻结） |
| `figures/` | 出版级图表 + `table_data/`（LaTeX 表格）；规格见主仓 `skills/figure_generation.md` |
| `planning/` | 流水线中间产物：`00-paper-config.md`（论文配置）、`02-outline-evidence.md`、`03-exp17h-*`、`04-narrative-plan.md`、`05-citation-map.md`、`06-stats-active.json`（**全部统计数字的唯一来源**）、`schedule.md` |
| `notes/` | 文献调研笔记（51 篇逐篇摘要 `paper-*.md` + 主题排查清单 + 方案对比分析）+ 审稿轮次记录（`chat1/2/3.md`）+ `raw/` 原始 dump；与 Zotero `AAYPCPGP` 对齐 |
| `discussion/` | 叙事与实验决策留档（含写作红线清单 §9） |
| `reviews/` | 评审报告占位目录；**当前本地为空**，git 不跟踪空目录（克隆后不存在，按 `round-N/` 约定新建） |
| `references.bib` | 17 条核验文献（IEEE 数字式，`IEEEtran.bst`） |
| `build.sh` / `.latexmkrc` | 构建入口与 latexmk 配置（产物强制输出到 `build/`） |
| `build/` | 构建产物（gitignore）；根目录 `main.pdf` 为提交副本 |

## 图表清单

| 文件 | 内容 | 生成方式 | 正文引用（PDF 图号） |
|------|------|---------|---------|
| `figures/fig1_handdrawn.pdf` | 概念图：(a) 点目标 vs (b) 候选窗 + 走廊（MuJoCo 实景渲染 + 误差标注）；**源文件 = `figures/ICRA.pptx`**（PowerPoint 手绘），经 `tools/pptx2fig1.py` 打补丁（错字/字体/媒体兼容）→ LibreOffice 转 PDF → 裁内容 bbox | `tools/pptx2fig1.py` | `introduction.tex`，单栏 Fig.1（`\columnwidth`） |
| `figures/fig1_hero.pdf` | hero 概念图（旧版，chat16 起未被引用） | `--fig 1hero` | — |
| `figures/fig2_algorithm_flowchart.tex` | 算法框图（TikZ，手工维护） | 直接编辑 .tex | `method.tex`，单栏 Fig.2 |
| `figures/fig3_swing_hero.pdf` | **实景 hero**：球的预测线频闪 + 整臂五帧残影 + 走廊管（±0.12 m）+ 候选窗口（±50 ms 橙段）+ 参考点/接触标注 + 右下角颜色图例（**正文现用版本**，chat27） | `scripts/tools/render_swing_hero.py` → `--fig 3hero` | `method.tex`，单栏 Fig.3（`\columnwidth`） |
| `figures/fig3_swing_hero_render.png` + `.json` | hero 的 MuJoCo 渲染素材与关键点像素坐标 | `scripts/tools/render_swing_hero.py` | 上面那行的输入 |
| `figures/fig3_alt_2d.pdf` | 走廊示意 2D 双面板（chat27 起移出正文，artifact） | `--fig 3alt` | — |
| `figures/fig3_tube_corridor.pdf` | 走廊示意 3D 版（历史版本，未被引用） | `--fig 3` | — |
| `figures/fig6_tube_robustness.pdf` | 机制归因四面板（**核心结果**） | `--fig 6` | `experiments.tex`，单栏 Fig.4（`\columnwidth`） |
| `figures/fig5_hit_rate_vs_speed.pdf` | 能力曲线含限速档单点 + 速度×cap 配对增益（chat27 由 4 面板瘦身到 2 面板） | `--fig 5` | `experiments.tex`，单栏 Fig.5（`\columnwidth`） |
| `figures/fig8_realtime_performance.pdf` | 实时性能预算 | `--fig 8` | `experiments.tex`，通栏 Fig.6（`\textwidth`） |
| `reviews/round-1/reviewer-comments.md` | round-1 审稿意见逐字转录 + 解读与处置 | 手工维护 | — |
| `figures/fig_scene_trajectory.png` | 真实 episode 单帧全轨迹 | `scripts/tools/render_ball_trail.py` | 已移出正文（artifact-only，chat16） |
| `figures/fig4_joint_trajectory.pdf` | 关节角偏差 + 命中区放大 | `--fig 4` | 已移出正文（artifact-only，chat14） |
| `figures/fig7_diagnostic.pdf` | 机制诊断（紧凑数值表） | `--fig 7` | 已移出正文（artifact-only，chat10） |
| `figures/fig1_system_overview.png` | 平台渲染截图（历史版本） | MuJoCo 离屏渲染 | — |
| `figures/table_data/table2_ablation.tex` | TABLE I：TCP 限速层内增益 + 配对 DiD（单栏） | `--fig table` | `experiments.tex` |
| `figures/table_data/table1_comparison.tex` | TABLE II：四档消融 / 网格 / 角点对比（通栏 `table*`） | `--fig table` | `experiments.tex` |
| `figures/table_data/table3_shared_input.tex` | TABLE III：shared-input 交叉验证（单栏，**手写维护**） | 手写 | `experiments.tex` |

⚠️ 文件名与 PDF 图号**不同源**（历史重排所致）：正文引用一律 `\Cref{...}`，不要按文件名推图号。
⚠️ `--fig table` 只写 `table1_comparison.tex` + `table2_ablation.tex`，两者自 chat29 起与磁盘版
**逐字一致**（含 table2 里的 chat22 注释行），重跑不再需要 `git checkout` 还原；
`table3_shared_input.tex` 与 `fig1_handdrawn.pdf` **不由 `paper_figs.py` 生成**。
⚠️ `--fig 3hero` 需要 `figures/fig3_swing_hero_render.png/.json`（由 `scripts/tools/render_swing_hero.py` 渲染）。

图表生成脚本在主仓：`scripts/plot/paper_figs.py`（`--fig` 参数见文件头注释；默认不含 `1hero`）。
在**主仓根目录**执行（需要 conda 环境与离屏渲染）：

```bash
MUJOCO_GL=egl /home/fxy/.local/miniforge3/envs/mujoco_tennis/bin/python \
    scripts/plot/paper_figs.py --fig 1hero 3 3alt 4 5 6 7 8 table
```

### 实景渲染配色（chat29）

正文两张实景图（Fig.1 顶栏实景、Fig.3 hero）共用一套「工业灰」运行期调色板：
右臂亮蓝→浅灰、拍面亮绿→近黑（半透明）、立柱近黑→中灰、拍柄→深灰；球与场地保持原色。
调色板定义在 `scripts/tools/render_ball_trail.py::PALETTES`，由 `apply_palette()` 在**渲染时**
覆盖 `geom_rgba`——**不改 `src/robot/rm65_model.xml`**，仿真视频与回放不受影响。
要回到模型原色：渲染时加 `--palette original`（Fig.3 还需重跑 `--fig 3hero`）。

### Fig.1 背景实景（chat29）

`figures/fig1_scene_render.png` 是 Fig.1 两个面板里嵌的实景背景（3:2，2160×1440），
由主仓脚本渲染；随后**在 PowerPoint 里手动**把它替换进 `figures/ICRA.pptx` 的
`ppt/media/image1.jpeg`（两个面板共用同一张图，替换一次即可）：

```bash
MUJOCO_GL=egl /home/fxy/.local/miniforge3/envs/mujoco_tennis/bin/python \
    scripts/tools/render_fig1_scene.py --palette industrial --fovy 36 --elevation 8 \
    --out paper/figures/fig1_scene_render.png
```

脚本复刻了 chat16 的原始配方（episode `exp18_tcp_exempt/raw/full_seed040.npz` 的
`hit_step−12`、az −100 / el 14 / d 2.0 / lookat −0.7 −0.6 0.8，黑天空→252→236 灰渐变），
并输出同名 `.json`（球的频闪位、接触点、0.12 m 的像素尺度），需要精确对位可据此换算。
`--fovy 36 --elevation 8` 是 chat29 的构图（比原机位更紧、柱子占比更小）；
`--fovy 45 --elevation 14` 复现原机位（标注位置不变，便于最小改动替换）。

⚠ 该 `.json` 里的像素是用 `mjvScene.camera[0]` 投影的，与栅格实际相机差 **~1° 方位角
（≈38 px）**；要精确对位请用下面的 `place_fig1_balls.py`（解析相机，标定残差 ≤1 px）。

### Fig.1 球精灵对位（主仓脚本，改 PPTX 几何）

`pptx2fig1.py` 只打文字/字体/媒体/底色类补丁，**不动几何**。压在实景图上的 12 张球精灵
（面板 (a)/(b) 各 6 张）由主仓脚本按相机解析投影改坐标：

```bash
/home/fxy/.local/miniforge3/envs/mujoco_tennis/bin/python scripts/tools/place_fig1_balls.py \
    --check-image --apply      # 只打印 → 就地改写 ICRA.pptx 里 12 个 <p:pic> 的 off/ext
python paper/tools/pptx2fig1.py  # 再重出 fig1_handdrawn.pdf
```

入射段默认取步 160/176/192/208（40 ms 间隔），续飞段取 233/241（按击球前速度的纯抛物线，
代表"没打中、球越过拍面"）；直径取该深度处的投影球径，因此链子天然由小到大。
换机位后重跑本脚本即可，其余标注（红虚线/管带/圆点/箭头/文字）仍由 PowerPoint 手工维护。

### Fig.1 独立生成链（PPTX 源头）

Fig.1 不走 `paper_figs.py`：源文件是本仓库内的 `figures/ICRA.pptx`，与 conda/MuJoCo 无关：

```bash
python paper/tools/pptx2fig1.py        # ICRA.pptx → figures/fig1_handdrawn.pdf（约 15 s）
```

工具集中维护六类补丁（脚本头部有完整说明与原因），原始 PPTX 字节**不动**：

| 补丁 | 内容 | 原因 |
|------|------|------|
| 文字 | `canditates`→`candidates`；全角括号→ASCII | 拼写错误；全角字形 Liberation Sans 缺失会触发中日韩字体回退 |
| 术语/单位 | `corridor`→`tube`（含 `terminal+Corridor`、`(within corridor )`）；`0.1m`/`50ms` → 数字与单位间补**不换行空格**；`≈` 两侧补空格 | chat28 正文全量 corridor→tube 的图内跟随；评审 #17「0.2m 全文要统一」；普通空格会让窄文本框把 "50 ms" 折行 |
| 不换行 | `文本框 130` / `文本框 225` 的 `wrap="square"`→`"none"` | 这两个 "50 ms" 标签框只有 47.5 pt 宽，加空格后 LO 会折成 "50 m / s" |
| 字体 | 全部 typeface → `Liberation Sans`（139 处） | 源文件文本 run 指向主题东亚槽位（→微软雅黑），本机缺失时回退到 Noto Sans CJK，拉丁字形带"中文黑体味" |
| 媒体 | `image3.webp`→PNG；`image4.svg` 补 width/height；4 个 SVG 图片补 `<a:blip>` 主 `r:embed` | LO 不支持 webp；该 SVG 缺尺寸属性；PowerPoint 的 SVG-only 写法只有扩展引用，LO 会整块丢弃 |
| 底色 | `矩形 107` / `矩形 40` 两个「彩色填充」底衬 → `<a:noFill/>` | 源图的整幅淡蓝/淡灰铺底印在论文里与白纸底不一致（合成 #E9EEF9 / #DFDFDF）；面板边框线与所有内容不动 |

导出后还会把被 LibreOffice 重编码的位图换回源文件原始字节（渲染图 48.7 KB → 72.9 KB 无损还原）。
`figures/fig1.emf` 是旧路线（EMF → LO）的中间产物，已被取代：留在磁盘但**不入库**（见 `.gitignore`）。

## 构建

```bash
./build.sh        # latexmk → build/main.pdf
./build.sh full   # 缺宏包时自动 tlmgr 安装（MiKTeX 式按需装包）再构建
./build.sh clean  # 清理辅助文件
```

TeX Live 2026 用户态安装于 `~/.local/texlive/2026`（PATH 已写入 `~/.zshrc`），包仓库指向 TUNA 镜像。
提交前把产物同步到仓库根：`cp build/main.pdf main.pdf`。

构建后自检：

```bash
pdfinfo build/main.pdf | grep Pages        # 必须 ≤ 8
grep -c "Overfull" build/main.log          # 应为 0
pdffonts build/main.pdf | grep -i "Type 3" # 应为空（字体必须 Type 42 嵌入）
python tools/check_margins.py build/main.pdf  # PaperCept 页边距：末行 verdict 必须为 PASS
```

`tools/check_margins.py` 按 PaperCept 规格（首页顶 ≥1 in、其余各边 ≥0.75 in、正文宽 7 in）
以**光栅墨迹**口径逐页判定 —— 不能用矢量提取口径：pdftex 以 Form XObject 嵌入图片，
提取器不裁剪其 /BBox，会把图片内部实际不落墨的几何算成"越界"（详见
`notes/chat19-revision-tracking.md` §5）。

## 数字溯源链

```
experiment_data/expN*/raw/*.log ──► scripts/extract/paired_stats.py ──► planning/06-stats-active.json
                                                                              │
experiment_data/exp18_fig_assets/raw/*.npz + timing.json ──► scripts/plot/paper_figs.py ──► figures/*.pdf
```

正文与图中的每个统计量都必须能追到 `planning/06-stats-active.json`
（active-hit 口径 + 配对 McNemar / 配对 bootstrap CI），不得手算或估算。

## 写作红线（摘要）

完整清单见 `discussion/2026-09-11_narrative-and-experiment-decisions.md` §9 与主仓 `skills/paper_writing.md`。

- 数字只能来自 `06-stats-active.json`；报告样本量 n 与 no_result 行数；
- 0.3 m/s 主动/被动阈值同时出现在 Definition 1 与实验协议，两处措辞保持一致；
- 走廊代价 = 垂直偏离 hinge loss（半宽 0.12 m = 拍半径），**禁止**再写 σ(t) 时变不确定性管道；
- "多段"只用于描述 softmin 候选击球窗口，不得用于"三段式走廊"；
- 引用只从 Zotero/arXiv 导出，禁止手写 BibTeX 条目。

## 备份与恢复

本仓库备份至 GitHub 私有仓库 `Nothing-ray/tennis-paper-icra2027`。
本机 `github.com:443` 直连被阻断，推送与克隆一律经 `gh-proxy.com` 通道：

```bash
git remote -v   # origin → https://gh-proxy.com/https://github.com/Nothing-ray/tennis-paper-icra2027.git
git push        # 经代理推送
```

换机 / 灾难恢复：

```bash
git clone https://gh-proxy.com/https://github.com/Nothing-ray/tennis-paper-icra2027.git paper
```

⚠️ 仓库内含审稿过程记录（`notes/chat*.md`、`discussion/`、`reviews/`），**必须保持 private**
——公开即暴露作者身份，违反 ICRA 双盲规则。

## 匿名 Artifact（投稿链接）

论文脚注引用的匿名 artifact：**https://anonymous.4open.science/r/Tennis_robot_789741**
（a4os 只读镜像，已剥离 git 历史；2026-09-13 由中性名镜像替换旧链接
`.../r/Tennis_robot_mujoco_sim-72A0`——旧链接已失效（HTTP 410），且 slug 保留真实仓库名；
源仓库为主仓 `Mask-Project-Tennis/Tennis_robot_mujoco_sim`）。

- 源仓库推送经 gh-proxy（主仓 `origin` 已配好）；推送前确认 `origin/main == HEAD`；
- 匿名副本是**快照**：主仓后续提交不会同步。数字或脚本变更后需在 a4os 重新匿名化并更新脚注 URL；
- 副本内已实测：统计 JSON sha256 与论文基准一致、两张表格字节一致、无 C++ 扩展也可跑通单 episode
  （见主仓 README「Reproducing the Paper Results」）。

## 写作流程（与学术技能的对接）

1. 配置确认：`planning/00-paper-config.md`（academic-paper Phase 0，IRON RULE）
2. 大纲/文献矩阵/论证蓝图 → `planning/02-*`、`03-*`、`04-*`
3. 中文起草 → `sections_zh/`；英文转写 → `sections/`（转写自中文稿，中文稿是唯一事实源）
4. 图表 → `figures/`（数据来自主仓 `experiment_data/`，生成链见「数字溯源链」）
5. 评审与修订 → 记录进 `notes/chatN.md` 与 `discussion/`（`reviews/round-N/` 按需新建）
6. 定稿导出引用 → `references.bib`；最终 `./build.sh` 出 PDF 并同步根 `main.pdf`
