# unitree_r1d —— R1D 整机官方固定资产

本目录是阶段 2(R1-A7-D 模型接入,Task 2.5a)的官方上游资产交付物:
原始整机 URDF、URDF 引用的全部网格与许可拷贝。所有文件的来源、URL 与
SHA256 见 [source_manifest.yaml](source_manifest.yaml),其正确性由
`tests/test_r1d_assets.py::test_source_manifest_hashes` 守护;固定事实
(关节表/变换链/限位冲突)的唯一来源是
`docs/experiments/reports/2026-11-15-r1a7d-model-interface-audit.md`。

## 内容

| 文件 | 说明 |
|---|---|
| `r1d.urdf` | 原始整机 URDF,与固定 commit 的 blob 逐字节一致 |
| `meshes/*.STL` | URDF 引用的 36 个网格(双臂/环境链 + 夹爪/相机/轮/脚轮/头) |
| `LICENSE` | unitree_ros 仓库根许可全文拷贝(BSD-3-Clause) |
| `source_manifest.yaml` | 来源/哈希/派生记录清单(勿手改) |

## 上游来源

- 仓库:<https://github.com/unitreerobotics/unitree_ros>
- commit:`157ff950351e5f535e03c5d85a16b7d594b3dbcc`
- 上游路径:`robots/r1d_description/`(URDF 相对引用 `meshes/*.STL`)

## 许可状态

仓库根 LICENSE 为 BSD-3-Clause 全文(版权人 HangZhou YuShu TECHNOLOGY
CO.,LTD,2016-2022);`r1d_description` 子目录无包级声明,根许可覆盖子目录
属合理推断(清单记 `bsd-3-clause(仓库根,子目录无独立声明)`)。本地
研究验证可继续;**对外再分发前需复核并随附版权与免责声明**。

## 派生模型与重建

由本目录整机 URDF 导出的 MJCF 片段位于 `src/robot/`(均为
`<mujocoinclude>`,生成过程字节确定,测试防漂移):

| 片段 | 说明 |
|---|---|
| `r1d_arm_left.xml` / `r1d_arm_right.xml` | 双侧裸臂:根 body `r1d_base`(waist_yaw_link 之上的虚拟固定安装点),该侧 7 关节链(肩俯仰→肩偏航…腕偏航)惯量/网格逐字来自 URDF;wrist_yaw_link 下 `r1d_flange` site = 该侧 dex1_base_joint origin;dex1 夹爪/腕部相机子树移除 |
| `r1d_env.xml` | 冻结环境链:base_link → lift_mount →(stage1@q_L, stage2@q_L)→ pelvis → waist_yaw@0,无任何关节;两级升降按 mimic multiplier=1 的同一 q_L bake 为固定偏移(默认 0,行程 [0, 0.425] m/级,总 0.85 m);轮/脚轮/头/对侧臂不进入 |
| `r1d_assets.xml` | 双侧臂(7×2)+ 环境(6)共 20 个 mesh 声明,支持任一侧场景组装 |

左右臂由同一工具对同一 URDF 仅以 `side` 参数区分生成(独立导出,非镜像
改名)。**限位逐字采用 URDF 值**(含右腕俯仰上界 +0.5326 的官方不对称,
不得"修正";指南与 URDF 的 4 项冲突见审计报告 §5,只记录不裁断)。

重建命令(默认 q_L=0;指定升降 bake 值加 `--qL`,仓库成品以 q_L=0 交付):

```bash
python scripts/tools/build_r1d_model.py
```

## 注意事项

- URDF 的 effort/velocity 非 0(官方声明值,如肩 60 N·m),但**未经实机
  验证,不得直接写成安全限值**;派生片段不设 ctrlrange/forcerange
  (清单 capability_note,N0-9)。
- 未引用网格(夹爪/相机/轮/脚轮/头共 16 个)仅作资产留存,不进派生片段
  (清单 simplifications 节)。
- 网格路径以 `../../assets/unitree_r1d/meshes/*.STL` 相对 `src/robot/`
  引用,与 RM65/D2 模型的相对路径模式一致;本目录移动会破坏派生模型加载。
- 请勿在本目录手工编辑清单哈希;新增文件必须同步登记清单(测试会拦截
  未登记文件)。
