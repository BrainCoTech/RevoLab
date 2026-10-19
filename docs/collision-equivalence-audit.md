# Revo3 碰撞表示等价性检查（2026-10-19）

## 结论

本轮没有找到同时满足“接触语义保持不变”和“采样速度已验证提升”的可落地方案。
不修改生产资产、碰撞过滤、自碰撞、接触间隙或训练任务，也不部署之前的凸包候选。
这是对本轮检查路径的结论，不是所有物理引擎优化均不可行的证明。

当前每个运行时凸体本身已经很简单。无损删除冗余顶点的路径未找到候选；
发现了少量被其他凸体包住的形状，但直接重建为独立 USD 凸体时会重新烘焙并
改变保留形状的外轮廓，因此没有继续进行采样跑分或训练。

## 范围与版本

- 仓库分支：`codex/asset-refresh-retrain`；HEAD：`13a52ef76b73a03f65f8bde1cda6e43be8b6fa7b`。
- 描述子模块：`6db7e2ed1d864ee0a3ab6ff62d43a303787137f6`。
- 资产：`revo3_system/usd/revo3_right.usd` 和 `revo3_left.usd`。
- 本机 Isaac Sim：`5.1.0-rc.19+release.26219.9c81211b.gl`；PhysX 扩展：`107.3.26`。
- 原始表示来自 PhysX `request_convex_collision_representation`，保留原生顶点、
  多边形及索引，不先经过 trimesh 的重新凸包化。
- 导出查询的属性适配仅写 USD session layer；没有保存回源资产。
- 几何审计覆盖左右独立手；重新烘焙检查覆盖右手。没有把本轮结果外推为
  完整双臂场景、GPU 动力学或长期 RL 收敛验证。

## 1. 单个凸体是否存在无损减面空间

| 项目 | 右手 | 左手 |
| --- | ---: | ---: |
| 源碰撞 prim | 82 | 82 |
| 凸分解 / 单凸包源 prim | 50 / 32 | 50 / 32 |
| 源碰撞网格顶点总数 | 672,020 | 672,020 |
| PhysX 凸体数 | 833 | 833 |
| 凸体顶点总数 | 9,721 | 9,746 |
| 单凸体顶点数：最小 / 中位 / P95 / 最大 | 4 / 11 / 19 / 39 | 4 / 11 / 19 / 38 |
| 重复顶点或非极点顶点 | 0 | 0 |
| 同一刚体内完全重复的凸体 | 0 | 0 |
| 被同一刚体其他凸体完全包含的凸体 | 36 | 19 |

顶点审计使用 float64 ConvexHull；包含关系以各凸体支撑半空间检查，数值容差
为 `1e-12 m`。这里的“包含”只说明几何集合关系，不自动保证接触点集合或求解行为一致。
没有把同一位置的不同关节刚体合并，也没有跨刚体删除形状。

因此，源网格很密不意味着采样时仍在处理同样多的三角面。只压缩源文件或外观
模型，不能据此宣称运行时采样提速。对本轮检查到的凸体，继续删极点会改变形状。

还筛查了 8 顶点 / 6 面的盒体候选。仅右手一个分解片段的角点符合精确盒形；
它属于一个多凸体源网格，不能单独改其 USD approximation 而保留其他片段。
这不足以构成已验证的优化方案。

## 2. 显式保存原凸体能否保持几何

将右手 833 个凸体的原生顶点及多边形逐个写成独立 Mesh，指定 `convexHull`，
设置顶点上限 64，再查询 PhysX 生成的表示。

- `minThickness=0`：550 个顶点集合完全相同，283 个变化。
- `minThickness=0.001`（该扩展的默认厚度）：结果同上。
- 最大原生凸体顶点数仅 39，因此没有输入凸体超过本次 64 顶点上限。
- 283 个变化中，278 个来自原凸分解，5 个来自原单凸包。
- 顶点相对另一凸体支撑半空间的最大越界为约 **1.966 mm**。
  此指标不是整体外表面的 Hausdorff 距离。

为排除“变化只发生在内部”的解释，还按刚体分别合并其所有凸体的顶点集合，
比较 4,102 个方向的最外支撑位置（六个坐标轴方向加 seed 42 的 4,096 个方向）：

- 22 个刚体均有超过 1 μm 的方向差。
- 最大方向差为 **1.134 mm**。

一个非零支撑差即可反驳该刚体几何并集完全相等；有限方向没有差异则不能证明
凹面也等价。这里检测到了明确差异，因此这条“重新建 Mesh 并烘焙”的路径未通过。
本轮没有测试直接复用 PhysX 二进制缓存，也没有将其加载收益视作采样收益。

## 3. 删除完全内含的凸体是否无代价

选取右手无名指 MCP 的一对真实凸体，内含关系的最小余量约为 22.31 μm。
这两个凸体在重新烘焙检查中顶点集合完全保持一致，避免叠加另一种几何变化。

在同一运动学刚体下放置这两个凸体，用半径 3 mm、质量 20 g 的球接触；比较：
原始双凸体、删除内部凸体、重复原始双凸体。每次使用新场景、相同坐标。
CPU PhysX / TGS，120 Hz，位置/速度迭代 32/1，contactOffset=0.1 mm、restOffset=0。

| 场景 | 原始重复对照 | 接触点数：原始 → 删除后 | 球心最大差 |
| --- | --- | ---: | ---: |
| 表面外 0.05 mm，接近速度 0.01 m/s | 完全一致 | 2 → 1 | 0 |
| 初始穿透 0.5 mm，接近速度 0.1 m/s | 完全一致 | 2 → 1 | 3.84 nm |

穿透测试的合冲量最大差约 `1.18e-9 N·s`。这些微小数值差不能解释为有意义的
任务退化；本例只说明接触报告并非逐项一致，且局部轨迹基本不变。它不证明
全关节范围、自碰撞、不同物体及 GPU 求解下的语义等价。

实际阻碍仍在资产表示：右手 36 个内含凸体散布于 11 个源网格，均不是某个源
网格的全部凸体。删除它们需要重建剩余凸体，而这 11 个网格的剩余凸体均存在
本轮重新烘焙后变化的部分。所以没有通过直接停用整个源 prim 实施删除。

## 对先前试验的限定

旧的 41 块手掌候选扫掠基线，是从提取凸体重新创建的参考碰撞体；重复对照
证明的是该重建参考的可重复性。它不是直接针对原始 USD 的完整运行时接触
等价性认证。本轮重新烘焙检查进一步说明不能忽略这一区别。

此前 26.86 s → 7.98 s 的大幅提速仍是改变几何后的诊断结果；本轮没有产生
新的“无损提速”数字，也未启动训练。当前训练继续使用原资产。

## 产物与复现

产物在 `outputs/collision-equivalence-20261019/`（本机实验目录，不作为生产资产）。
这些原始脚本和数据未随仓库发布；下面的命令仅适用于已持有该实验目录的环境，
不是克隆仓库后即可运行的公开复现入口。

- `export_cooked.py`、`revo3_right.json`、`revo3_left.json`：原生表示。
- `audit.py`、`audit.json`：极点、重复与包含关系；脚本含正负自检。
- `roundtrip.py`、`roundtrip.json`、`roundtrip-default.json`：重新烘焙对照。
- `check_support.py`、`support-check.json`：按刚体的外支撑面差异。
- `contact_counterexample.py`、`contact-counterexample.json`、`contact-penetrating.json`：局部接触对照。
- 同名 `.log`：原始运行日志。

从仓库根目录运行几何检查：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python outputs/collision-equivalence-20261019/audit.py \
  outputs/collision-equivalence-20261019/revo3_right.json \
  outputs/collision-equivalence-20261019/revo3_left.json \
  --output outputs/collision-equivalence-20261019/audit.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python outputs/collision-equivalence-20261019/check_support.py
```

原生查询和接触脚本需通过现有 Isaac Sim 环境启动。
将 `ISAACSIM_PATH` 设为本机 Isaac Sim 安装目录后，例如重做穿透对照：

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PXR_WORK_THREAD_LIMIT=8 \
PYTHONEXE="$PWD/.venv/bin/python" "${ISAACSIM_PATH}/python.sh" \
  outputs/collision-equivalence-20261019/contact_counterexample.py \
  --root outputs/collision-equivalence-20261019 --gap -0.0005 --speed 0.1 \
  --output-name contact-penetrating.json --headless --device cpu \
  --kit_args '--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8'
```
