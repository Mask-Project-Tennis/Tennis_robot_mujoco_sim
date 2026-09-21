# 旁白定稿（九段）

> 由 `scripts/video/make_narration.py` 生成，勿手改（改文案请改脚本 SEGS）。
> 音色 `en-US-ChristopherNeural` · 语速 `-10%`（实测口播 ~118 wpm / 全片等效 ~99 wpm）· 共 264 词 / 160 s。
> 音频起点约定 = 槽位起点 + 0.3 s；SRT 同口径。

| 段 | 时间 | 词数 | 音频/预算 s | 中文（确认稿）→ 英文（TTS/字幕） |
|---|---|---|---|---|
| S1 | 0:00–0:12 | 15 | 7.4 / 11.7 | 一台双臂机器人挥拍回击一记快速发球——挥拍全程由 MPC 规划。 |
| | | | | A dual-arm robot swings to return a fast serve — planned entirely by model predictive control. |
| S2 | 0:12–0:26 | 21 | 12.9 / 13.7 | 内部是 iLQR 在滚动优化：终端击球代价加运行代价。我们只改动其中两项——每一项都可以独立开关。 |
| | | | | Inside, iLQR solves a rolling optimization: a terminal hitting cost plus running costs. We modify exactly two terms, each independently switchable. |
| S3 | 0:26–0:46 | 22 | 12.7 / 19.7 | 第一项，终端代价：不再绑定单一预测触球点，而是瞄准一个 ±50 毫秒的候选窗口，用 softmin 聚合成同一个终端目标。 |
| | | | | First, the terminal cost. Instead of one predicted contact point, it targets a plus-or-minus fifty-millisecond window of candidates, aggregated by a softmin. |
| S4 | 0:46–0:59 | 21 | 10.9 / 12.7 | 第二项，一条沿预测轨迹的 tube：横向偏离在 12 厘米半宽内零代价，出界后铰链代价随偏离增长；沿轨迹的超前/滞后不受罚。 |
| | | | | Second, a tube around the predicted trajectory: free inside a twelve-centimeter half-width, a hinge penalty outside — and along-track slack is free. |
| S5 | 0:59–1:27 | 53 | 24.3 / 27.7 | 这是同一记 12 m/s 发球、同一随机种子的并排对比：point-target 漏球，完整方法回击成功。98 个种子下，仅候选窗口就把 active-hit rate 从 53.1% 提到 85.7%——配对增益 +32.7 个百分点；9 m/s 下是 +17.4。 |
| | | | | Here is the same twelve-meter-per-second serve, same seed: point-target misses; the full method returns it. Across ninety-eight seeds, the candidate window alone lifts the active-hit rate from fifty-three point one to eighty-five point seven percent — a paired gain of plus thirty-two point seven points; plus seventeen point four at nine meters per second. |
| S6 | 1:27–1:42 | 25 | 12.3 / 14.7 | 但在共享预测偏差下——9 m/s、20 格扰动网格——差距收窄：82.8% 对 73.9%。 |
| | | | | Under a shared prediction bias — a twenty-cell perturbation grid at nine meters per second — the gap narrows: eighty-two point eight, versus seventy-three point nine percent. |
| S7 | 1:42–2:09 | 57 | 25.1 / 26.7 | 但把 TCP 限速收紧到 1.0 m/s（7 m/s 发球、394 个种子），窗口反而有害：仅候选窗口 −11.9 个百分点，完整方法 −12.7；tube 单独仍 +5.8。在这个速度预算下，放宽目标没有帮助。 |
| | | | | But tighten the end-effector speed cap to one meter per second — seven-meter serves, three hundred ninety-four seeds — and the window turns harmful: minus eleven point nine points for the window alone, minus twelve point seven for the full stack. The tube alone still gains plus five point eight. Under this budget, widening the target does not help. |
| S8 | 2:09–2:30 | 34 | 20.4 / 20.7 | 主结果在首个计划期间冻结仿真时间（928 毫秒）；稳态重规划中位数 4 毫秒，在 150 毫秒预算内。若不冻结，999 毫秒的计划让挥拍无暇完成。 |
| | | | | Main results freeze simulated time for the first plan: nine hundred twenty-eight milliseconds. Steady-state replanning runs at a four-millisecond median, within the one-hundred-fifty-millisecond budget. Without freezing, a nine-hundred-ninety-nine-millisecond plan leaves no time to swing. |
| S9 | 2:30–2:45 | 16 | 8.0 / 14.7 | 哪种松弛有用，取决于可用的执行速度——代价设计应该跟随速度预算。 |
| | | | | Which relaxation helps depends on the available execution speed: cost design should follow the speed budget. |
