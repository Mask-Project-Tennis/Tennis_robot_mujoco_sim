# exp17i：限速 × 机制消融配对实验

> **状态**: 待运行（2026-09-12 启动）
> **上游依据**: `paper/planning/04-narrative-plan.md`（Related Work Q5 相对版主张）、
> `docs/experiments/reports/2026-07-09_exp16_v12_limits_rerun.md`（TCP 1.0 = 36.4% @full）

## 目的

验证叙事协商中确认的新主张（相对版，风险表 §8#6）：

> 「限速越紧，鲁棒层相对点目标的增益越大」——鲁棒性预算与速度余量存在替代关系。

绝对主张（「限速下仍成功」）已被 exp16 证伪（TCP 1.0 下 full 仅 36.4%，−49pp），
不得写入论文；本实验检验的是**相对增益**是否随限速放大。

## 设计（配对版，用户已确认）

| 参数 | 值 |
|------|-----|
| 球速 | 7 m/s（对齐 exp16） |
| 限速层 | TCP 1.0（`--limits-config configs/real_robot.yaml`）× TCP 1.8（默认） |
| 消融档 | full / tube_only / softmin_only / none |
| Seeds | 400/格（同 seed 跨限速层 + 跨档对齐） |
| 规模 | 2 × 4 × 400 = **3,200 runs ≈ 7 分钟** |

配对设计让「限速层」成为唯一组间变量：同一批发球（seed 相同）在两层限速下各打一遍。

## 对照锚点与判定规则

- **TCP 1.8 @ 7 m/s 四档**（exp17d，n≈98/档）：tube vs none = −2.0pp（ns）、
  softmin vs none = +4.2pp——本实验同 seed 复测并加功效。
- **Sanity check**：TCP 1.0 的 full 档应复现 exp16 的 ~36.4%（n=400）。
- **判定**：DiD 检验（gain@TCP1.0 − gain@TCP1.8）。放大且 p<0.05 → 主张以相对形式
  进 Discussion；不放大 → 如实写「限速是独立的绝对瓶颈，鲁棒层无额外增益」。

## 统计功效

预期 TCP 1.0 下命中率 25-40% 区间，n=400/格 CI ±4.3~4.8pp；层内 5-8pp 差异可检；
DiD 分辨 ~7pp（两层增益之差）。

## 数据目录与执行

`experiment_data/exp17i_limits_ablation/`；执行
`python scripts/exp/batch_runner.py exp17i_limits_ablation --workers 16`；
提取 `python scripts/extract/extract_exp17i_results.py`
（层内 z 检验 + 跨层增益放大 DiD 检验 → extract_summary.json）。
