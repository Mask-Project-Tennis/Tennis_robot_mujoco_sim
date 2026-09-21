# ICRA 2027 随稿视频制作清单与检查表

> **文档日期**：2026-09-20　**配套**：`plan.md`（分镜/数字口径/管线设计），本文档是其执行层清单
> **用法**：每完成一项勾选 `- [x]`；B 系列检查表是**门禁**——上一层全过才进下一阶段；B5 数字核对表每处数字改动后重跑
> **倒排**（截止 9/22 23:59 PST）：9/20 EOD 完成 Phase 0–1 → 9/21 EOD 粗剪全片（Phase 2–4）→ 9/22 定稿上传（Phase 5–6）

---

## A. 制作任务清单（六阶段，含验收标准）

### Phase 0　工程就绪（预计 0.5 h）✅ 2026-09-20 完成
- [x] 建 `paper/video/assets/` + `paper/video/.gitignore`（忽略 `assets/`）
- [x] 建 `scripts/video/` 目录
- [x] 字体确认：DejaVuSans / DejaVuSans-Bold（`/usr/share/fonts/truetype/dejavu/`）
- [x] `MUJOCO_GL=egl` 无头渲染冒烟通过（**须加 `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`**，否则 EGL 初始化失败；解释器退出时 `__del__` 的 EGLError 为无害噪音）
- [x] ffmpeg 二进制：`…/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`，`-version` 正常
- [x] edge-tts 7.2.8 安装 + 试音成功（**pip 走清华镜像** `-i https://pypi.tuna.tsinghua.edu.cn/simple`，7890 代理已死勿用；TTS 服务网络可达）→ **无需触发 §C 降级**

### Phase 1　种子与录像（预计 1–2 h，关键路径）
- [x] `pick_seeds.py`：扫描 exp17 系列 CSV → 12 m/s 分歧种子 + 7 m/s 干净命中种子 → `assets/seeds.json`（2026-09-20）
  - 结果：分歧种子共 35 个，前 3 = **seed 25 / 31 / 9**（均 none=miss、full=active、**softmin_only=active** ✓ 与旁白"candidate window alone"对应、tube_only=miss）；干净命中 = **seed 47**（hit_pos_error 2.4 cm）
- [x] 种子核验：按 `config_json` 重建 CLI 重跑 v12，hit_type 全部一致（seed 25/31/9 none→miss、full→active；seed 47 full→active）（2026-09-20）
- [x] 配对一致性核验：seed 25 四变体（none/softmin_only/tube_only/full）`--dump-trajectory`，**p0/v0 逐位一致** ✓（配对 npz 已存 `assets/paircheck_s25_*.npz`，可作渲染兜底输入）
  - ⚠️ exp17 的 12 m/s 用 v12 默认 `--serve-distance 8.0`（config_json 无覆盖）——render_hits.py 必须对齐，**不能用 run_20hits_video 的 SERVE_DIST_MAP**
- [x] `render_hits.py`：S1 开场段 ✅（2026-09-20 用户已检查通过）
  - seed 47 @7 m/s full，hit_type=active 与核验一致：常速全程 2.24 s + 接触 0.25× 推近慢放 3.6 s，共 5.83 s
  - 视觉定稿：industrial 灰配色（43 规则）、球亮黄+频闪尾迹、慢放推近（dist 2.2 / az 112）、headlight 补光、wrapper ghost 池
- [x] S5 配对 + S9 尾镜 ✅（2026-09-20）——**改走 npz 无漂移管线**：render_hits 复制的 run_20hits 内联循环与 v12（replan_core）存在行为差异，seed 25 none 被翻转成 active（`--expect` 警报拦截）⇒ 新增 `render_npz.py` 运动学回放 v12 自 dump 的轨迹 npz（`paircheck_s25_*.npz` + 新 dump `seed47_full.npz`），画面即实验真实记录
  - `s5_pt_*`（none，miss）：PASS 帧球擦拍而过；`s5_full_*`（full，active）：CONTACT 帧球贴拍反弹——同 seed 来球逐帧一致，t=0.70 起分化
  - **速度（2026-09-20 用户二轮调整定稿）**：全程广角 **0.25×**（4.47 s，两栏并排素材）+ 接触窗口 **0.1×**（~7.9 s，窗口右端截至 episode 末）；preview 带 "0.25×/0.1× playback" 标注
  - **平滑化（用户三轮反馈）**：render_npz 改为**亚步插值**（qpos/球位按小数步线性内插，逐输出帧按需渲染）——0.1×/0.25× 下的「源帧整数驻留」步进感消除（连续 12 帧 md5 全唯一验证）；S5/S9 已重渲，另产 `s1_slow_*`（seed 47 npz，广角 1.0× + 推近 0.25×，作 S1 慢放段的平滑替代，组装时用其替换 s1 的 B 段）
  - **角标增补（用户四轮反馈）**：左上角标 *Simulation* → **MuJoCo simulation**（第三方仿真器名，提升可信度、双盲合规，plan.md §1 已同步）；全部 preview 已重出，组装配文字以 `render_hits.BADGE_TEXT` 为单一来源
  - `s9_*`：seed 47 同球换机位（dist 2.6 / az 118）作收尾镜头，与 S1 呼应
  - S5/S9 全部产物在 `paper/video/assets/`（各 4 mp4 + sheet + contact.png + meta.json）

### Phase 2　动画（预计 2–3 h）✅（2026-09-21）
- [x] `anim_method.py` S2（框架 14 s）/ S3（候选窗 17 s）/ S4（tube 13 s）→ `s2_method/s3_window/s4_tube.mp4`
  - 配色 灰/橙/蓝/绿 统一；无 B1 红线元素（点目标十字在候选窗展开时淡出 = 被"放宽"而非并存；无追逐/无对勾）；右下 "Schematic" 角标明示示意图
  - 屏显常量运行时 import `TubeConfig`（±50 ms / 0.12 m / β=5，`tube_types.py:18/39/48`），启动打印 B5 溯源
  - 视觉修正：S2 死循环占位行重写；S3 softmin 权重改箭头粗细（去拥挤权重条）；S4 tube 标签移左下避内嵌图、上探针标注移左避标题
- [x] `anim_results.py` S5 增益图 12 s / S6 扰动 15 s / S7 限速 27 s / S8 时间轴 21 s → `s5_chart/s6_perturb/s7_limit/s8_timing.mp4`
  - **B5 全过**：数字运行时读取/重算，启动打印审计表——S5/S6/S7 ← `planning/06-stats-active.json`（E3/E4/E8），S8 ← `exp18_fig_assets/timing.json`（928.3/4/32 ms）+ `exp23_nofreeze/results.csv` 同口径重算（999 ms→0.0% n=2368、224 ms→74.5% n=784）；150 ms 预算 = results.tex RQ4 设计常量
  - **舍入口径（本轮发现）**：论文 pp 增益走「2 位小数中间步再 half-up」（17.347→17.35→+17.4），率值/CI 直接 half-up（22.449→22.4）——`r1_pp`/`r1` 分开设，审计表与 V23 逐字对齐
  - 视觉修正：S5 增益标签移组右侧空白（原压橙柱）；S6 "+8.9 pp" 移右缘（原撞 82.8%）；S7 数值标签移条形上方（原被 CI 横线穿过）
- 产物：7 段 mp4（1280×720@30fps CRF17，共 ~2 MB，全部可解码）+ 各 `_sheet.png` QC 图；ruff clean
- **用户一轮反馈修正（2026-09-21，重渲 s3/s4/s6/s7）**：S3 softmin 汇聚箭头橙→深墨色（橙带上橙箭头不可辨）；S4 带外探针偏移加大（3.8/4.4→6.2/6.0）+ 箭头 1.4→2.8 lw / ms 22、从圆点边缘起笔；S6 网格加轴刻度与轴题（列 = temporal bias t (ms)，行 = spatial bias s (m)，**轴值从 E4 per_cell 键运行时解析** t∈{0,10,25,50,100}×s∈{0,0.05,0.1,0.2}，原灰块意义不明）；S7 行名前身份色块删除（蓝块与红/绿条形无对应，误导）
- **用户二轮反馈修正（2026-09-21，重渲 s4/s6/s7）**：S4 三个 "cost 0" 标签移出走廊带（上内侧探针→点上方 +2.9；中/下内侧探针→点正下方 −3.8，原被带底色遮挡）；S6 偏差箭头移到两曲线间隙中部、文字下移至曲线下方空白、虚线预测偏移 3.0/2.0→4.5/3.0（原排版拥挤）；S7 右侧率值列字号加大（11/9.5→15/12，header 9.5→12）
- **用户三轮反馈修正（2026-09-21，重渲 s4/s6/s7）**：S4 中/下内侧 "cost 0" 标签改锚到**走廊下缘外侧**（沿法向 −5.6，斜坡段水平文字的角会蹭回带内，纯竖直下移治不了）+ 0.9 细引导线表明归属；S6 偏差箭头**两端点各加标注**（墨点=true，橙点=biased prediction，箭头加粗 lw 2.8/ms 20），删冗余右上角标（与 biased prediction 标签同行相撞）；S7 除黑条带外全部字号加大（行名 12.5→15、增益值 14→17、CI/p 10→12、率值 15→17、vs 12→13.5、header 12→13.5、"0 pp" 10.5→12.5、banner 15→17）
- **用户四轮反馈修正（2026-09-21，只重渲 s4）**：S4 整体字号加大（tube 标签 14→16.5、cost 0/hinge 10.5→13、slack 12.5/11.5→15.5/14、小图 title 11.5→14 / 刻度 10→13 / free 11→13.5）；hinge 小图整体上抬（y 0.115→0.17，**刻度数字 0/0.12/0.40 m 原贴底部黑条**）；连锁修三处新碰撞——tube 标签加字号后尾部 "(racket radius)" 撞小图左下角（移到左下空区 x=12）、slack 副句尾蹭走廊上缘与探针箭头起点（缩为 "early / late along the line is free"）、小图标题外置被走廊下缘穿过（移入图内 x=0.62 分位，避开 0.12 虚线与 free）+ 小图底色 0.92→0.97 压掉透底的走廊

### Phase 3　旁白与字幕（2026-09-21 完成）
- [x] `narration.md` 定稿（九段，共 264 词英文，全片等效 ~99 wpm）
  - 数字全过 B5：+32.7/+17.4（正文 §VI-B 口径，非表内相减）、53.1→85.7、98/394 seeds、82.8 vs 73.9（20 格网格）、−11.9/−12.7/+5.8（TCP 1.0）、928/4/150/999 ms、0.12 m、±50 ms；无 first/guaranteed 红线
  - 流程：先出中文稿给用户确认 → 用户三项拍板（音色 Christopher / 语速 −10% / 烧录英文字幕）→ 落 `scripts/video/make_narration.py`（SEGS 为唯一事实来源，narration.md 每次运行重新生成防漂移）
- [x] TTS 音轨 9 段（edge-tts en-US-ChristopherNeural，rate −10%）+ `subtitles.srt`（31 条 cue）
  - **实测校准**：Christopher 口播 ~118 wpm（比 Aria 标定的 142 慢）→ 初版 S2 超 0.6 s、S8 超 0.5 s，各砍词后全 PASS（S8 20.4/20.7 最紧）
  - 字幕时间轴用 edge-tts **WordBoundary 词级事件**（100 ns tick）对齐到句/子句 cue，非比例估算；破折号归一为逗号再切分（承接处带标点不粘连）；两行均衡折行（≤48 字符/行）
  - 槽位校验硬门：每段音频 ≤ 槽位 −0.3 s（HEAD_GAP 约定：音频起点 = 槽位起点 + 0.3 s，assemble.py 须遵守同口径，见 `assets/narration/narration_timings.json`）
  - `--only Sx` 重跑时从 timings 缓存回填其余段（防 SRT/文档被残缺覆盖）；edge-tts 直连可用（代理死不影响，微软端点直通）

### Phase 4　组装（2026-09-21 粗剪 + 五轮精修完成，待用户复核）
- [x] `assemble.py`：槽位表 9 段共 4800 帧（160.0 s）→ PIL 逐帧烧录（MuJoCo simulation 角标 / 0.25×·0.10×·0.5× 倍速标签 / S5 变体条件条 / S1 标题叠底 / 全程字幕 / S9 匿名结尾卡）→ 音轨 adelay+amix（HEAD_GAP=0.3 s 与 SRT 同口径）→ H.264 CRF18+VBV950k + AAC 64k + `-map_metadata -1` + faststart
  - 粗剪规格：**160.0 s / 6.25 MB / 312 kbps**（预算 18.5 MB，余量足可换更高画质）
  - **视觉 QC（18 帧，本会话开了视觉直接 Read）**：修 `parse_srt` 把 SRT 两行拼回一行导致字幕溢出画幅（保留 `\n` 修复）；四处动画底部元素撞字幕带（588–654 px）全部上移后重渲——S3 时间尺 y 9.8→15.5、S4 tube 标签 12.4→15.5 + hinge 小图 inset y 0.17→0.235、S6 网格 gy0 14→17.6（行距 4→3.4）、S5 chart 基线 y0 13→17（sc 0.45→0.36）
  - 合规预检：封装/编码/分辨率/帧率/逐行 ✓；元数据仅剩容器 handler（isom/Lavf，无身份信息）✓；字幕匿名词扫描唯一命中为 "available" 含 "lab" 子串（误报）✓
  - **用户五轮反馈精修（2026-09-21，重渲全部动画段 + 重组装）**：① 字幕 30px 两行 → **24px 单行**、底部锚定 y=622（新带 622–652 px；SRT 重生成为单行，>1224 px 的 cue 运行时自动缩字号，全片仅 1 条降到 23px）；② 全局次文字灰 `INK_2` 0.42→0.27（灰小字难辨认，两动画脚本同步）；③ S6 网格灰块 → **按偏置强度橙色渐变**（caption 注明 deeper = stronger bias，颜色编码配置强度而非结果，诚实）+ 刻度 9.5→12 / 轴题/标签全部加大；④ S7 底部结论行 y 11.5→13.5（原被两行字幕遮挡）；⑤ S8 整体放大一号（轴/量表/chips 字号 ×1.2、条高加厚，chips 底缘抬到场景 y 10.8 = 616 px 安全线，绿刻度改贴飞行轴）。新字幕带更靠下，S2 芯片行 / S3 时间尺 / S5 图基线等旧避让元素自动解除遮挡。精修后成片 **160.0 s / 6.35 MB / 318 kbps**。
  - **用户六轮反馈精修（2026-09-21，全片 160→165 s）**：① `INK_2` 再调深 0.27→**0.19**（灰字对比度二连升）；② S2 次级文字整体放大一号（块副标签 10.5→12.5、卡片副行 12→13.5、chips 11.5→13.5 等）；③ **S3 末态驻留 +3 s**（S3_DUR 17→20，槽位表/SEGS 全体后移）；④ **S4 tube 标签移出走廊左下腿**（(12,15.5)→(13.5,12.0)，原标签被弹道起点段穿过）；⑤ **整句字幕**：重写 `split_pieces`（子句贪心打包 ≤92 字符、禁止跨句回并、超长子句中点平衡二分——初版硬切分支会打乱词序已修）+ 新增 `coalesce_sentences`（≤112 字符整句合成单条 cue）；**S3–S7 重合成**拿新词级时间戳（edge-tts 同文本时长逐位一致），cue 31→26，1:11 处 mid-phrase 断句消除；⑥ **S9 结尾 +2 s**：槽位 150–165（成片 **165.0 s**），接触镜头 0.25×→**0.14× 重渲**（3.2→5.7 s），旁白 158.3 s 说完、匿名卡 158.8 s 才淡入（原 153 s 处打断旁白）。精修后成片 **165.0 s / 6.61 MB / 321 kbps**。
  - **结尾卡补匿名代码链接（2026-09-22，用户拍板「加链接 + 欢迎复现」）**：卡片新增两行
    `Code & data: https://anonymous.4open.science/r/Tennis_robot_789741`（与论文脚注同口径）
    + "Easy setup — reproduction welcome"；成片 **165.0 s / 6.62 MB** 不变。措辞用
    Easy setup 而非 one-command（仓库暂无 install.sh，B 纪律诚实呈现；后续若加一键安装
    脚本可升级措辞）。
  - **一键安装脚本 + 措辞升级（2026-09-22，用户拍板「做一个一键安装脚本」）**：
    主仓新增 `install.sh`（conda 环境 → requirements → 可选 C++ 扩展（失败回退 NumPy）→
    单集冒烟测试期 `hit_type=active`，幂等，本机实测通过）；README 中英双语 Environment
    Setup 顶部加一键入口；结尾卡措辞 "Easy setup" → **"One-command setup — reproduction
    welcome"**（此时措辞为真）。成片 **165.0 s / 6.64 MB**。
- [ ] **粗剪全片用户审看**（唯一人工门）→ 精修意见回收

### Phase 5　合规与定稿（预计 1 h）
- [ ] `check_video.py` 全项通过（B3 + B4 程序化部分）
- [ ] 人工实播 QC + 抽帧给用户过目
- [ ] 成片定稿 `paper/video/icra2027_video.mp4`

### Phase 6　提交与留档
- [ ] 用户经 PaperPlaza 上传（**9/22 23:59 PST 前**；先在系统内试传校验 20 MB）
- [ ] git 本地提交（paper 仓 + 主仓，**不 push**）；checklist 勾选完成归档

### 用户侧事项（仅用户能做）
- [ ] 审看粗剪（Phase 4 末）
- [ ] 放行成片（Phase 5 末）
- [ ] PaperPlaza 上传（Phase 6）
- [ ] （可选）TTS 不满意时自录旁白（提供 narration.md 即可，音轨整轨替换）

---

## B. 检查表（门禁）

### B1 素材级检查（每段录像 / 动画逐一过）
- [ ] hit_type 与核验运行一致（录像段）
- [ ] 720p 缩放下球与尾迹清晰可见（录像段）
- [ ] 固定机位、画面无 GUI/坐标轴残留（录像段）
- [ ] 30 fps、时长与分镜匹配（±0.5 s）
- [ ] 配色 = point-target 灰 / candidate 橙 / tube 蓝 / full 绿（动画段）
- [ ] 无红线元素：不画追逐候选点、不显式搜索接触时刻、无对勾暗示必中（动画段）
- [ ] 图表数字读自 JSON 源（动画段）
- [ ] 最小字号在 720p 可读（≥24 px 等效高度）

### B2 组装级检查（粗剪全片过）
- [ ] 九段顺序与时长 = 分镜表（±0.5 s/段）
- [ ] MuJoCo simulation 角标全程常驻、位置固定、不遮任何关键内容
- [ ] 条件条固定上方、字幕固定下方，均不遮接触区
- [ ] 慢放段标注倍速（0.25× playback）
- [ ] 转场：直切为主，机制段（S2–S4）渐显
- [ ] 旁白与字幕同步（偏差 ≤0.3 s）
- [ ] 音频无爆音，峰值 ≤ −3 dB，响度整体一致
- [ ] 片尾：结论句 → 匿名标题卡 → "Details in the submitted paper"

### B3 成片合规检查（硬性，全过才放行）
- [ ] 时长 ≤180 s（目标 ~160 s）　`ffmpeg -i` 解析
- [ ] 大小 ≤20 MB（目标 ≤18.5 MB）　`ls -l`
- [ ] 高度 ≥480 px（实际 720）
- [ ] 帧率 ≥20 fps（实际 30）
- [ ] 逐行扫描（yuv420p、无 interlace 标志）　`ffmpeg -i` 流信息
- [ ] 封装 mp4、H.264 视频 + AAC 音频
- [ ] 元数据已清空（`-map_metadata -1` 后复查无 encoder/author 等字段）
- [ ] 全片实播无花屏/音画不同步（人工）

### B4 匿名专项扫描（双盲）
- [ ] 画面与字幕无作者/单位/实验室名称
- [ ] 无任何 URL、arXiv 号、GitHub、水印、致谢
- [ ] 文件名与元数据无身份信息
- [ ] 结尾仅匿名论文标题 + "Details in the submitted paper"
- [ ] （若换自录音轨）确认无自我介绍/署名；TTS 无此问题

### B5 数字口径核对表（每处屏幕/旁白数字 → 来源，改动必重查）
| 视频中出现 | 来源 | 核对方式 |
|---|---|---|
| 53.1 → 85.7 %（12 m/s nominal） | Table II `tab:ablation` | 对照 `table1_comparison.tex` |
| **+32.7 pp** @12 m/s | results §VI-B / abstract 原文 | grep 原文，勿用表值相减 |
| **+17.4 pp** @9 m/s | 同上 | 同上 |
| 82.8 vs 73.9 %（perturbed grid, 9 m/s） | Table II perturbed 行 | 对照表文件 |
| **−11.9 / −12.7 / +5.8 pp**（TCP 1.0） | Table I `tab:limits` | 对照 `table2_ablation.tex` + JSON E8 |
| n=394（限速段） | Table I / §VI-D | 对照 |
| 4 ms / 32 ms p95 / 150 ms 预算 | §VI-E | 对照 results.tex |
| 928.3 ms 首计划（冻结）/ 999 ms（未冻结）/ 224 ms（5-iter） | §VI-E | 对照 |
| 未冻结 0 命中（2368 eps）/ 74.5 %（n=784） | §VI-E | 对照 |
| ±50 ms 窗口半宽 / 0.12 m tube 半宽 | method 节 | 对照 method.tex |
| 7 / 9 / 12 m/s，TCP 1.0 / 1.8 m/s | 各段条件条 | 对照实验配置 |
| 收尾句原文 | abstract | 逐词对照 |

### B6 上传前放行单（release gate，Phase 6 入口）
- [ ] B1–B5 全部通过
- [ ] 用户实播审看通过（Phase 5）
- [ ] 成片文件在本地播放器二次复验（时长/画质/音轨）
- [ ] PaperPlaza 试传成功（系统侧 20 MB 校验）
- [ ] 留档完成：checklist 全勾 + git 本地提交（不 push）

---

## C. 降级决策记录（触发时填写）

| 触发条件 | 备选 | 决策 | 日期 |
|---|---|---|---|
| edge-tts 不可用 | piper 离线 / 用户自录 | 待填 | |
| 160 s 超 20 MB | 降 25 fps → 砍 S6/S8 至 ~140 s → 960×540 | 待填 | |
| 9/21 18:00 仍无粗剪 | 直接执行压缩分镜（砍 S6/S8） | 待填 | |
| 分歧种子核验全失败 | 用 exp18 npz 素材离线重渲染 | 待填 | |
