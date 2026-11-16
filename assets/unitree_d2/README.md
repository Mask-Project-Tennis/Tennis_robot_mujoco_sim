# unitree_d2 —— D2 机械臂官方固定资产

本目录是阶段 2（D2 模型接入）的官方上游资产交付物：原始 URDF、URDF 实际引用
的全部网格与许可/来源声明拷贝。所有文件的来源、URL 与 SHA256 见
[source_manifest.yaml](source_manifest.yaml)，其正确性由
`tests/test_d2_assets.py::test_source_manifest_hashes` 守护。

## 内容

| 文件 | 说明 |
|---|---|
| `D2_URDF.urdf` | 原始 URDF，与固定 commit 的 blob 逐字节一致（LF 行尾） |
| `meshes/*.STL` | URDF 引用的 9 个网格（base_link + link_1..link_8） |
| `package.xml` | 上游 ROS 包清单拷贝（含 license 声明） |
| `source_manifest.yaml` | 来源/哈希/派生记录清单（勿手改） |

## 上游来源

- 仓库：<https://github.com/unitreerobotics/D2-Arm>
- commit：`8d505a8c511f2a8dd401bd4c410e37121b1bfd17`
- URDF 上游路径：`urdf/urdf/D2_URDF.urdf`（注意双层 urdf 目录）

## 许可状态（未确认）

上游仓库无 LICENSE 全文；`package.xml` 声明 `<license>BSD</license>`，但
author/maintainer 均为 TODO 占位，BSD 变体与版权人无法确定，故清单记
`license_status=unverified`。本地研究验证可继续；**对外发布或再分发前需向
宇树科技取得书面授权**。

## 派生模型与重建

由本目录 URDF 生成的裸臂 MJCF 片段位于 `src/robot/d2_arm.xml` 与
`src/robot/d2_assets.xml`（`<mujocoinclude>`，根 body 为 `d2_base`，
joint_1..joint_7；joint_8/link_8 固定末端附件已按契约移除并记录于清单）。

重建命令（生成过程字节确定，测试防漂移）：

```bash
python scripts/tools/build_d2_model.py
```

## 注意事项

- URDF 的 velocity/effort 全为 0，表示官方未提供可用能力值：派生模型与其
  派生场景不设任何 ctrlrange/forcerange/速度限值（见清单 unknown_capabilities）。
- 网格路径以 `../../assets/unitree_d2/meshes/*.STL` 相对 `src/robot/` 引用，
  与 RM65 模型的相对路径模式一致；本目录移动会破坏派生模型加载。
- 请勿在本目录手工编辑清单哈希；新增文件必须同步登记清单（测试会拦截未登记文件）。
