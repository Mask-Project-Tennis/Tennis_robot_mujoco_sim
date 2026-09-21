# ICRA 2027 随稿视频制作计划

> **文档日期**：2026-09-20
> **依据**：`paper/notes/video1.md`（两版分镜建议，以最新 2:40 九段版为底稿）+ 仓库现状调研（录制管线 / 论文数字 / 消融开关三路核实，结论见附录 A）
> **用户决策**（2026-09-20 拍板）：① 旁白 = TTS 合成英文；② 组装 = 代码全自动；③ 分镜 = 新版 2:40 九段
> **状态**：计划留档，待批准执行
> **截止**：**2026-09-22 23:59 PST**（北京时间 9/23 15:59）。上传窗口 9/18–22，过时不补；视频强烈鼓励但可选

---

## 1. 目标与硬约束

成片：`paper/video/icra2027_video.mp4`

| 项 | 值 | 官方要求 |
|---|---|---|
| 时长 | ~160 s | ≤180 s |
| 文件大小 | ≤18.5 MB | ≤20 MB（留余量） |
| 封装/编码 | MP4 / H.264 + AAC | mpeg / mp4 / mpg |
| 分辨率 | 1280×720 | 高度 ≥480 px |
| 帧率 | 30 fps | ≥20 fps |
| 扫描 | 逐行（yuv420p） | progressive |
| 码率预算 | 160 s × ≤20 MB ⇒ 总码率 ≈ 0.95 Mbps（视频 ~850k + 音频 64k） | — |

**双盲合规**（IEEE RAS 双盲规则）：
- 不出现作者、单位、实验室名称与标志；模糊人脸（本项目素材全为仿真，天然满足）
- 无泄露身份的链接与文件元数据（导出时 `-map_metadata -1` 清空）
- 机器人型号与研究所必需设备画面可保留（RM-65B + MuJoCo 画面合规）
- 结尾：匿名论文标题 + "Details in the submitted paper"（不放 arXiv/GitHub 链接）

**诚实呈现要求**（video1.md / RAS 视频建议）：
- 全程常驻 **MuJoCo simulation** 角标（左上；2026-09-20 增补仿真器名——MuJoCo 为第三方开源仿真器、论文正文已写明，标注它提升可信度且不破坏双盲）
- 慢放片段标注 *0.25× playback*
- 对照段标注 ablation 名称与实验条件；统计段标注指标名 *active-hit rate*
- 不把多次实验剪辑拼接成一次；失败片段保留上下文（point-target miss 为真实记录）
- 清晰英语旁白 + 英文字幕

---

## 2. 数字口径（已对 V23 提交稿逐项核实，视频引用以此为准）

> ⚠️ 表格 label 与文件名错位：TABLE I = `tab:limits`（文件 `table2_ablation.tex`）、TABLE II = `tab:ablation`（文件 `table1_comparison.tex`）、TABLE III = `tab:shared`。

### 2.1 标称四档对比（TABLE II，active-hit rate %）

| 条件 | full | tube-only | cand-set-only | point-target | n |
|---|---|---|---|---|---|
| Nominal 7 m/s | 83.8 | 78.4 | 83.5 | 81.4 | 97–99 |
| Nominal 9 m/s | 95.9 | 86.7 | 95.9 | 78.6 | 98 |
| Nominal 12 m/s | 85.7 | 74.5 | 85.7 | 53.1 | 98 |
| Perturbed grid（20 格，9 m/s） | 82.6 | 81.2 | 82.8 | 73.9 | 1960–1962 |
| Corner 12 m/s, t=0, s=0.2 m | 64.8 | 61.6 | 64.8 | 54.7 | 2986–2987 |

### 2.2 TCP 限速消融（TABLE I，7 m/s、n=394 seeds，相对 point-target 配对增益 pp）

| config | TCP 1.8 增益 | TCP 1.0 增益 | p (1.0) |
|---|---|---|---|
| tube-only | +3.3 | **+5.8** | 0.027 |
| candidate-set-only | +1.3 | **−11.9** | <0.001 |
| full | +1.5 | **−12.7** | <0.001 |

### 2.3 引用规则

- 标称增益直接引用正文口径 **+17.4 pp @9 m/s、+32.7 pp @12 m/s**（results §VI-B / abstract / conclusion 原文；与表值舍入差 0.1 pp，**勿用表内数字现场相减**）
- 限速段引用 **−11.9 / −12.7 / +5.8 pp**（表文一致；JSON 源 `planning/06-stats-active.json` E8_limit_mechanism）
- abstract 已删 −11.9/−12.7（chat18 导师意见）但正文 §VI-D 保留，视频按正文口径使用
- 勿混用另一组 speed×cap 扫描口径（7 m/s×1.0 组 full/cand-set 均 −20.6 pp [−33.0,−8.2]）
- 背景数字：TCP 约束集整体代价 84.4%→35.9%（−48.7 pp，RQ0）

### 2.4 实时性 / 冻结初始化（RQ4，§VI-E）

- 稳态重规划 4 ms 中位（32 ms p95、43 ms max、291 次），150 ms 预算内；瓶颈 = 首个同步 30 迭代计划 928.3 ms（仿真时间冻结执行）
- 未冻结端到端：首计划 999 ms 中位 → 仅剩 ~100 ms 挥拍 → 2368 episodes 0 命中；首计划限 5 迭代（224 ms 中位）→ 74.5%（n=784）
- 冻结→未冻结命中率：9 m/s 95.9/86.7/95.9/78.6 → 81.6/82.7/86.7/69.4；12 m/s 85.7/74.5/85.7/53.1 → 74.5/68.4/76.5/56.1

### 2.5 收尾句与标题

- 收尾句（abstract 原文）："Which relaxation helps depends on the available execution speed: cost design should follow the speed budget."
- 匿名标题（结尾卡）："Relaxing the Point Target: A Cost-Relaxation Framework for Robust Robotic Ball Hitting"

---

## 3. 分镜表（2:40 九段，160 s）

| 段 | 时间 | 画面 | 素材来源 | 旁白要点（英文草案） | 屏幕文字 |
|---|---|---|---|---|---|
| S1 | 0:00–0:12 | 常速完整挥拍（~5 s）→ 接触前后 0.25× 慢放（~7 s）+ 短标题叠底 | render_hits.py：clean-hit seed，7 m/s，full 模式（备选：`a_hit_clean.npz` 离线重渲染） | "A dual-arm robot swings to return a fast serve, planned entirely by MPC." | MuJoCo simulation 角标；慢放时 0.25× playback |
| S2 | 0:12–0:26 | 简化 MPC-iLQR 回路，只高亮终端代价与运行代价两个插入点 | anim_method.py（参考 fig2 TikZ 逻辑重绘简化版，静态矢量逐项出现） | "Inside, iLQR tracks a terminal hitting cost plus running costs. We modify exactly two cost terms—each independently switchable." | terminal cost / running cost 高亮 |
| S3 | 0:26–0:43 | 抛物线球轨迹 + 灰色单点目标 → 展开为 ±50 ms 橙色候选组 → softmin 汇聚到同一终端状态 | anim_method.py（Fig.1 思想重绘） | "Instead of one predicted contact point, the terminal state targets a ±50 ms window of candidates, aggregated by a softmin." | ±50 ms candidate window |
| S4 | 0:43–0:56 | 蓝色半透明 tube（半宽 0.12 m）沿球轨迹；带内绿点代价 0、带外红点随偏离增长 | anim_method.py | "A tube around the predicted trajectory penalizes lateral deviation—free inside, costly outside." | 0.12 m half-width tube |
| S5 | 0:56–1:24 | 两栏配对录像（12 m/s 同 seed：point-target miss \| full hit，各 ~8 s 真实记录）→ 切标称增益图（~12 s） | render_hits.py（exp17d 分歧种子 ×2 变体）+ anim_results.py | "At 12 m/s serves, the candidate window alone lifts the active-hit rate from 53.1 to 85.7 percent—a paired gain of +32.7 pp; +17.4 at 9 m/s." | 条件条：12 m/s serve · same seed · active-hit rate；图注：paired gain over point-target |
| S6 | 1:24–1:39 | 有偏差的球状态（偏移预测点）同时输入各规划模块示意 | anim_method.py / anim_results.py | "Under a shared prediction bias—a 20-cell perturbation grid at 9 m/s—the gap narrows: 82.8 versus 73.9 percent."（措辞克制） | perturbed prediction · 9 m/s grid |
| S7 | 1:39–2:06 | 限速条件条 + 三条增益标记（带 CI 与显著性） | anim_results.py（TABLE I 数据） | "But tighten the TCP cap to 1.0 m/s at 7 m/s serves and the window turns harmful: −11.9 pp; full −12.7. The tube still gains +5.8. Under this limit, widening the target does not help." | serve 7 m/s · TCP cap 1.0 m/s · n=394 · gain over point-target |
| S8 | 2:06–2:27 | 初始化冻结与稳态重规划时间轴：init frozen（928 ms 首计划）→ 稳态 4 ms vs 150 ms 预算；未冻结 999 ms→0% / 5-iter 224 ms→74.5% | anim_results.py（RQ4 / Fig.6 数据） | "Main results freeze simulated time for the first plan; steady-state replanning then runs at 4 ms, well within a 150 ms budget. Without freezing, a 999 ms first plan leaves no time to swing." | steady-state replanning · 150 ms budget |
| S9 | 2:27–2:40 | 干净击球镜头 + 结论句 + 匿名标题卡 | render_hits.py 尾镜 | "Which relaxation helps depends on the available execution speed: cost design should follow the speed budget." | Details in the submitted paper；结尾卡小字 Simulated in MuJoCo |

旁白总量约 260–320 词 / 160 s（≈100–115 wpm，从容语速）；逐段定稿与实际音频时长反哺分镜微调（每段预留 ~0.3 s 头部余量）。

---

## 4. 视觉与叙事规则

| 项目 | 规则 |
|---|---|
| 变体配色 | point-target 灰、candidate 橙、tube 蓝、full 绿（全片统一） |
| 方法动画 | 静态矢量逐项出现，同视角同色系；不做 3D 动画 |
| 动态对照 | 最多两栏，避免四路小录像 |
| 统计画面 | 一句完整结论 + 少量直接标注的增益与误差条（Assertion–Evidence 风格） |
| 仿真镜头 | 固定相机（复用 run_20hits_video 机位：az135 / el−15 / d3.5 / lookat [0,0,1] + 四灯），隐藏 GUI，渲染球尾迹 |
| 文字位置 | 条件条固定上方、字幕固定下方、MuJoCo simulation 角标左上，不遮挡接触区 |
| 转场 | 直切为主，机制解释段渐显 |
| 声音 | 清晰英语旁白（TTS）+ 校对后的英文字幕，无背景音乐 |
| 叙事红线 | 不画"依次追逐候选点"、不显式搜索接触时刻、不用对勾暗示带内必中、不以成功片段串烧代替并排对照 |

---

## 5. 素材清单

### 现有（直接复用）
- `scripts/run_20hits_video.py`：离屏 MP4 录制管线（mujoco.Renderer + imageio，1280×720，crf18/yuv420p，`--ablation` + `--start-seed`；**默认 fps=200，须显式 `--fps 30`**）
- `scripts/tools/render_ball_trail.py` / `render_swing_hero.py` / `render_fig1_scene.py`：相机/布光/尾迹渲染技术可借用
- `experiment_data/exp17d_mechanism/results.csv`：12 m/s 四档 per-seed 结果 + `config_json`（挑分歧种子与重建 CLI 的依据）；`exp17i_limits_ablation`（限速段，可选）
- `experiment_data/exp18_fig_assets/raw/`：`a_hit_clean.npz`、`pt_miss_s20_s77.npz`、`co_hit_s20_s77.npz`、`pt_miss_t50.npz`、`sm_hit_t50.npz`（当年即为视频所备，离线重渲染兜底素材）
- `paper/figures/`：fig1_handdrawn.pdf（机制示意原型）、fig2 TikZ（框架逻辑）、fig6_tube_robustness.pdf（=Fig.4）、fig5_hit_rate_vs_speed.pdf、fig8_realtime_performance.pdf（数据源参考）
- ffmpeg 7.0.2：imageio-ffmpeg 自带（系统无 ffmpeg/ffprobe），路径 `…/envs/mujoco_tennis/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`

### 待生成（全部由 scripts/video/ 脚本产出）
- S1/S9 击球录像 2 段、S5 配对录像 2 段（render_hits.py）
- 方法动画 3 段 S2–S4（anim_method.py）
- 结果动画 4 段（S5 图表 / S6 / S7 / S8）（anim_results.py）
- 旁白音频 9 段 + subtitles.srt（make_narration.py，edge-tts）
- 成片（assemble.py）+ 合规报告（check_video.py）

---

## 6. 制作管线与脚本

| 脚本 | 职责 | 要点 |
|---|---|---|
| `pick_seeds.py` | 从 exp17d CSV（12 m/s）挑 none=miss & full=hit 分歧种子并核验 | 对齐该行 `config_json` 重建 CLI 重跑核验 hit_type；避开种子 20/21/27（系统性 no_result）；输出 `assets/seeds.json` |
| `render_hits.py` | MuJoCo 击球录像 | 复用 run_20hits_video 管线；**配对镜头必须 `--n-runs 1` + `--start-seed S`**（脚本跨 run 共享 RNG，仅第 1 球与 v12 同 seed 逐位一致）；12 m/s 显式传 `--serve-distance`（SERVE_DIST_MAP 无 12 档）；渲染球尾迹防低码率丢球；30 fps 输出 |
| `anim_method.py` | S2–S4 方法动画 | matplotlib Agg 逐帧 → imageio 合成 |
| `anim_results.py` | S5 图表 + S6–S8 结果动画 | 数据读论文 JSON 源（`planning/06-stats-active.json` 等），不手抄 |
| `make_narration.py` | edge-tts 逐段合成 en-US 音轨 + 生成 subtitles.srt | `pip install edge-tts`；每段实际时长反馈分镜 |
| `assemble.py` | 组装与编码 | PIL 逐帧烧录角标/条件条/字幕（静态 ffmpeg 无 libass，统一 PIL，字体 DejaVu/Noto Sans）→ 拼接+转场 → ffmpeg：H.264 yuv420p、CRF+VBV（maxrate≈950k / bufsize 2000k）+ AAC 64k、`-map_metadata -1`、目标 ≤18.5 MB |
| `check_video.py` | 合规检查 | 时长/大小/分辨率/帧率/逐行/元数据（解析 `ffmpeg -i` stderr，无 ffprobe）+ 字幕与画面匿名词扫描 |

---

## 7. 目录结构

```
paper/video/            # 视频工程（随 paper 嵌套仓；本地提交不 push，沿用审稿轮纪律）
├── plan.md             # 本文档
├── narration.md        # 英文旁白定稿（逐段时间码）
├── subtitles.srt
├── assets/             # 中间素材（.gitignore 大文件）
└── icra2027_video.mp4  # 成片

scripts/video/          # 生成脚本（主仓，遵循 scripts/ 分区惯例）
```

---

## 8. 进度安排

| 日期 | 内容 |
|---|---|
| 9/20（今天） | 步骤 1–3：建工程 → 挑种子核验 → 击球录像；旁白文稿初稿 |
| 9/21 | 步骤 4–7：动画 → TTS 音轨 → 组装出全片粗剪 → 用户审看 → 修订 |
| 9/22 | 步骤 8–9：精修（配色/字幕/转场）→ 合规检查 → 成片 → 用户经 PaperPlaza 上传 |
| 9/23（北京白天） | 最后缓冲，不依赖 |

---

## 9. 合规检查清单（check_video.py + 人工验收口径）

- [ ] 时长 ≤180 s（目标 ~160 s）
- [ ] 文件 ≤20 MB（目标 ≤18.5 MB）
- [ ] 高度 ≥480 px（实际 720）
- [ ] 帧率 ≥20 fps（实际 30）
- [ ] 逐行扫描（yuv420p，无场交织标志）
- [ ] 封装 mp4（H.264 + AAC）
- [ ] 元数据已清空（`-map_metadata -1` 后 `ffmpeg -i` 验证）
- [ ] 匿名：字幕与画面文字无作者/单位/实验室名；无外部链接；结尾为匿名标题 + "Details in the submitted paper"
- [ ] 常驻 MuJoCo simulation 角标；慢放有倍速标注；对照/统计段有条件与指标标注
- [ ] 实播 QC：球轨迹可见（尾迹）、小字 720p 可读、字幕不遮接触区、失败片段为真实记录非拼接
- [ ] 抽帧渲染给用户过目后再定稿

---

## 10. 风险与对策

| 风险 | 对策 |
|---|---|
| 码率紧（160 s@≤20 MB ≈ 0.95 Mbps），快速小球压缩后消失 | 球尾迹渲染；静态/图表段占大头；CRF+VBV 双约束；仍不行则降 25 fps、或砍 S6/S8 至 ~140 s、或 960×540（均仍合规） |
| edge-tts 网络不可用 | 退离线 TTS（piper）或改用户自录（音轨独立，可整轨替换不动画面） |
| 种子核验不一致（run_20hits_video 与 exp17d 配置漂移） | 以 CSV `config_json` 精确重建 CLI；仍不符换种子（分歧种子充足） |
| 静态 ffmpeg 无 libass/fonts | 字幕与角标统一走 PIL 逐帧烧录 |
| 12 m/s serve-distance 映射缺口 | 显式传参与实验配置对齐 |

---

## 附录 A. 关键技术事实（调研留档）

### A.1 同 seed 四变体配对机制（零代码改动）
- `rm65_mpc_v12.py` 的 `rng = np.random.default_rng(args.seed)` 在任何规划前创建；发球采样 `generate_ball_from_serve_box` 只消费该 RNG，四种 ablation 在发球阶段 RNG 消耗序列相同（重试仅由几何拒绝决定）⇒ **同 seed + 同 `--ball-speed` + 同 serve 参数 ⇒ 四变体逐位同一条来球**
- 变体映射：point-target = `--ablation none`、candidate-set-only = `--ablation softmin_only`、tube-only = `--ablation tube_only`、full = `--ablation full`
- `--limits-config configs/real_robot.yaml` = TCP 1.0 m/s（限速段用）
- `--dump-trajectory PATH` 存 npz（q_desired/u/q_actual/tcp_pos/ball_pos + metadata 含 p0/v0/hit_type），可事后核验四条 run 发球一致

### A.2 已知坑
- `run_20hits_video.py` 的 `--fps` 默认 = 1/dt = 200，必须显式 `--fps 30`
- 该脚本跨 run 共享 `default_rng(start_seed)`：仅第 1 球（i=0）与 `rm65_mpc_v12.py --seed S` 逐位一致 ⇒ 配对录像用 `--n-runs 1` 分别跑
- `SERVE_DIST_MAP = {5:5.7, 6:6.8, 7:8.0, 8:9.0, 9:9.5, 10:10.0}` 无 12 档，12 m/s 需显式 `--serve-distance`（与 exp17d config_json 对齐）
- 种子 20/21/27 系统性 `no_result`，选演示种子时避开
- 单 episode ≈ 1.65–2.25 s 仿真时长，墙钟 p50 ≈ 1.2–1.9 s + 进程启动；首计划 ~928 ms
- 无头渲染：`MUJOCO_GL=egl`（项目标准做法）
- rm65_model.xml 无 `<camera>` 元素，机位全靠程序化自由相机（复用现有 configure_camera 套路）

### A.3 工具链
- conda env `mujoco_tennis`：mujoco 3.9.0、matplotlib 3.11.1、imageio 2.37.4、imageio-ffmpeg 0.6.0（自带 ffmpeg 7.0.2，无 ffprobe、无 libass）；无 moviepy / opencv
- 系统 PATH 无 ffmpeg；一切编码走 imageio-ffmpeg 自带二进制

### A.4 数字来源文件
- TABLE II：`paper/figures/table_data/table1_comparison.tex`（label `tab:ablation`）
- TABLE I：`paper/figures/table_data/table2_ablation.tex`（label `tab:limits`）
- 正文口径：`paper/sections/results.tex` §VI-B（+17.4/+32.7）、§VI-D（−11.9/−12.7/+5.8）、§VI-E（RQ4）
- JSON 源：`paper/planning/06-stats-active.json`（E8_limit_mechanism 等）
