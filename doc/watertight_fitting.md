# 水密网格贴合原始网格：算法、实验与结论

本文记录将“水密化后的网格（source）”精准贴合到“原始网格（target）”的算法设计、
工程实现、关键实验结论与推荐配置。任务以 case1 为例：

- source（待形变，水密网格）：`/nvme0pnt/lichanghao/chLi/Dataset/watertight/watertight_case/case1_wt_1536.ply`
- target（拟合目标，原始网格）：`/nvme0pnt/lichanghao/chLi/Dataset/watertight/watertight_case/case1_gen.glb`

最终采用的版本为 **v4**，作为当前唯一保留版本（demo 与 `WatertightFitter` 默认参数已对齐 v4）。
在 v4 基础上又新增了两项能力（见第 9、10 节）：**优化过程中的自交防护**与**接近收敛后的误差驱动局部保形细分**。

---

## 1. 任务与数据特性

- 规模为千万级：source ≈ 1445 万顶点 / 2891 万面，target ≈ 1328 万顶点 / 2646 万面。
- source 与 target 在**原始坐标系下既不同位也不同尺度**（source 最大 bbox 边 ≈ 1.647，
  target ≈ 1.000，中心也不同），因此必须先统一坐标系再做刚性/非刚性对齐。
- source 与 target **拓扑不同、顶点不一一对应**，任何依赖同索引的约束（如同索引 fixed
  vertices / target vertices）都不适用。

评估阈值定义：`L` 为原始 target 的 bbox 最大边长（case1 中 `L ≈ 1.0`），
`tau = L / 2048 ≈ 4.88e-4`，在该阈值下计算 F1-Score。

---

## 2. 与原计划相比的关键调整（基于第一性原理）

原计划设想复用 `OptimalMapper`（逐顶点组局部仿射 + 对称表面 Chamfer + 每轮最近邻）。
实测发现它在该数据规模下不可行或不可取：

1. **逐顶点局部仿射 + Python 遍历边的 stiffness 构建**在 4336 万条边上不可行；
   `simplify_quadric_decimation` 简化千万级网格也过慢（数分钟无结果）。
2. **暴力 Chamfer 算子是 O(N×M)**：400k×400k 单次前反向约 2.4s，14M×3M 直接卡死。
3. **朴素对称 Chamfer 形变会让结果变差**：在稀疏随机表面采样上做对称 Chamfer，
   位移场会追逐采样噪声，实测 F1 从 0.358 降到 0.297。

据此重构出**拓扑无关、全分辨率 GPU 可行**的 `WatertightFitter`。

---

## 3. 最终算法（WatertightFitter）

流程：

1. **目标坐标系归一化**：用 target 的 bbox center/scale 同时归一化 source 与 target，
   使两者落入共享坐标系，并记录 `L`，评估时再反归一化回原始 target 坐标系。
2. **刚性初始化**：对 source → target 表面采样点做 Open3D ICP（含 bbox 预对齐）。
3. **逐顶点位移场优化**：优化变量为每顶点位移 `disp ∈ R^{V×3}`。
   每个 outer iter：
   - 用**空间最近邻索引**（Open3D GPU `NearestNeighborSearch`，O(N log M)）为每个当前
     变形顶点找到最近 target 表面点，建立稳定的逐顶点对应；
   - 对超过掩码阈值的远距离对应做 mask（剔除不可靠匹配）；
   - 内层若干步最小化 `掩码点对点数据项 + 边 Laplacian 平滑（位移差）`，可选 point-to-plane；
   - 掩码阈值与 Laplacian 权重按 coarse→fine 退火。
4. **指标守护**：拟合完成后分别评估“仅刚性基线”与“形变结果”，**只有当形变改善
   chamfer_l1 时才保留形变结果**，否则回退到刚性基线，避免把已经很好的刚性对齐改坏。
5. **评估与反归一化**：在原始 target 坐标系下用高密度表面采样计算 Chamfer 与 F1@tau。

关键工程点：

- 用张量化方式构建唯一边（无 Python 循环），4336 万边约 0.6s。
- 全量 14M 顶点对应查询是主要耗时（5M target 索引下约 27–104s/次），因此默认
  `corr_refresh_every`>1，仅周期性刷新对应（相邻刷新之间位移极小，精度损失可忽略）。

涉及文件：

- `non_rigid_icp/Module/watertight_fitter.py`：核心类，已升级为“优化到平台期 → 误差定位 → 局部细分”的多周期控制器，并内置逐步自交守护
- `non_rigid_icp/Method/sampling.py`：目标坐标系归一化、表面采样、可微 deformed 表面采样
- `non_rigid_icp/Method/nn.py`：GPU 空间最近邻索引 `NNIndex`（新增 `queryKNN` 支持碰撞 broad-phase）
- `non_rigid_icp/Method/topology.py`：拓扑原子函数（唯一边、顶点→面、面邻接、面掩码膨胀、共享顶点判定、连通域）
- `non_rigid_icp/Method/geometry.py`：向量化几何核（点-三角/段-段/三角-三角距离、Möller 三角-三角相交判定）
- `non_rigid_icp/Method/collision.py`：自交两阶段检测（质心 kNN broad-phase + 精确 narrow-phase）
- `non_rigid_icp/Method/convergence.py`：平台期检测 `PlateauMonitor`
- `non_rigid_icp/Method/error_field.py`：双向拟合误差 → 高误差面定位（绝对阈值 + 分位上限 + 硬上限 + 连通域/膨胀）
- `non_rigid_icp/Method/subdivision.py`：局部保形细分 `subdivideMarkedFaces`（红/绿模板，无裂缝、保朝向）
- `non_rigid_icp/Loss/surface.py`：对称 Chamfer、point-to-plane、边 Laplacian 损失
- `non_rigid_icp/Loss/collision.py`：可微自交屏障损失 `selfCollisionBarrierLoss`
- `non_rigid_icp/Metric/chamfer.py`：双向 Chamfer + F1@tau（基于快速 NN，支持百万级点）
- `non_rigid_icp/Demo/watertight_fitter.py`：默认即 v4 配置 + 自交守护 + 自适应细分的 demo
- `non_rigid_icp/Eval/evaluate_case1.py`：对任意输出 mesh 的独立评估脚本
- `non_rigid_icp/Test/test_geometry.py`、`non_rigid_icp/Test/test_fitter_smoke.py`：原子函数单元测试与端到端冒烟测试

---

## 4. 实验结果（flux 环境，CUDA_VISIBLE_DEVICES=2，2M 采样评估）

| 版本 | 关键配置 | F1@(L/2048) | chamfer_l1 | 耗时 |
|---|---|---|---|---|
| baseline | 仅 normalize + 刚性 ICP | 0.4508 | 0.0011089 | — |
| 朴素对称 Chamfer | 稀疏随机采样数据项 | 0.297（下降） | 0.001387 | — |
| **v4（最终）** | outer=60, inner=20, target=2M, lap=200, lr=2e-4, refresh=10 | **0.4868** | **0.0010591** | ~5 min |
| v5 | +point-to-plane, refresh=4, target=3M | 0.4890 | 0.001056 | ~18 min |

v4 相对刚性基线：**F1 +8.0%（0.4508 → 0.4868）、chamfer_l1 −4.5%**，指标守护判定为保留形变。
独立评估脚本复现一致（F1 = 0.4871）。

v5/v6 方向（更频繁刷新对应 + point-to-plane 精修主导 + 更密采样）能再微弱提升
（v5 到 0.489），但**收益递减且计算成本数倍增长**，因此选定 v4 为最优折中且作为唯一版本。

---

## 5. 精度的物理上限（重要）

water­tight 网格是 **1536³** voxel 重建得到的，体素边长 ≈ `1/1536 ≈ 6.5e-4 ≈ 1.33×tau`。
也就是说 **source 表面本身的离散化精度就略粗于评估阈值 tau**，单纯 warp 顶点无法突破该
分辨率下限。v4 已接近此上限。若要进一步提升：

- 使用更高分辨率的水密网格（如 2048³），或
- 在拟合后对 source 做局部细分再贴合。

---

## 6. F1 评估的采样密度敏感性（必读）

`tau = L/2048` 阈值极严格。F1@tau 对采样密度高度敏感：即使两个**完全相同**的表面，
若采样过稀，点间距 > tau 也会导致 F1 远低于 1（实测球面自对自：50k 点 F1≈0.012，
2M 点才到 ≈0.38）。因此：

- 评估必须使用高密度采样（**≥ 2M**，越高越接近真实贴合度），或直接用全顶点；
- 报告 F1 时务必同时注明采样点数；
- 不同版本对比必须使用**相同的采样密度与随机种子**，否则不可比。

---

## 7. 自交防护（self-collision guard）

目标：优化过程中**防止水密网格里原本不相交的非相邻面片产生新的自交**；相邻面（共享顶点）
与水密网格初始已存在的接触默认不算“新增自交”。

第一性原理拆解为可复用原子函数（两阶段碰撞检测）：

1. **broad-phase（质心 kNN）**：复用 GPU 空间索引 `NNIndex`，对每个（活动）面取其 k 个最近质心邻面，
   按包围球重叠 + 余量过滤，并用 `facePairsShareVertex` 剔除拓扑相邻对，得到极小的候选面对集合。
   折叠/自穿处对向面片的质心天然落入彼此的 kNN，故能被召回。规模可达千万面级。
2. **narrow-phase（精确判定）**：在候选集合上做向量化的 Möller 三角-三角相交测试
   （`triangleTriangleIntersects`，非共面精确、共面用接触距离兜底）。

两种使用方式：

- **可微软屏障**（`Loss/collision.py::selfCollisionBarrierLoss`）：对候选面对的三角-三角间距
  施加 `relu(margin - dist)^2` 障碍，在面片**穿透前**就把对向面推开。`margin` 取很小
  （默认 `0.25·tau`），只防穿透、不强加人为间隔（避免破坏薄壁结构）。
- **硬回退守护**（`Module/watertight_fitter.py::_guardedOuterStep`）：每个 outer step 后检测“新增自交”
  （当前相交对 − baseline 允许集）。若出现新增，则**回滚到步前位移**、放大屏障权重并重试，
  最多 `max_collision_retries` 次；仍无法消除则保留安全的步前状态。由此保证每一步都不会
  留下新的自交。

工程要点：

- 水密输入按定义无自交，故对**千万级网格跳过全量基线扫描**（`collision_full_scan_max_faces`），
  baseline 取空、候选仅对**移动过的活动面**（位移超阈、并按面邻接膨胀 1 环）周期性
  （`collision_refresh_every`）惰性构建，避免每轮对全网格做 kNN。小网格仍做精确全量基线。
- 单元测试覆盖：相交/分离/共面、共享边相邻过滤、近接未触、折叠双层穿插。

## 8. 自适应局部细分（adaptive subdivision）

目标：当**全局拟合误差下降趋平（接近收敛）**时，自动发现高误差区域并对其局部细分，再继续拟合，
逐步突破“水密网格离散分辨率下限”，把全局误差进一步压低。最多 `K=4` 轮。

1. **平台期感知**（`Method/convergence.py::PlateauMonitor`）：用双窗口均值比较代替单点，
   当相对下降率连续 `patience` 次低于 `plateau_rel_tol` 即判定接近收敛（对噪声鲁棒）。
2. **高误差区域定位**（`Method/error_field.py::localizeHighErrorFaces`）：合并两个方向的误差——
   fit（source→target，贴合偏差）与 coverage（target→source，覆盖不足，回投到最近 source 顶点）——
   取每面误差为两者较大值。阈值以**绝对值 `error_mult·tau` 为主**（误差已在容差内的面不细分），
   分位数 `quantile` 作上限、`max_refine_faces` 作硬上限，避免大网格细分面数爆炸；再用连通域
   过滤碎块、按面邻接膨胀 `dilation_rings` 环以平滑细分边界。
3. **局部保形细分**（`Method/subdivision.py::subdivideMarkedFaces`）：标记面的三条边全部对半分裂，
   邻接面按被分裂边数 1/2/3 用红/绿模板重三角化（分裂成 2/3/4 子面）。共享边的中点是同一新顶点，
   因此**无裂缝（无 T 形接缝）、保持朝向、保持水密拓扑**。中点取边端点均值（几何不变），
   细分后位移场从 0 重启即可。
4. **重建与续拟合**：细分后重建顶点/面/边/邻接/碰撞候选并重置优化器，细分周期用最细 stage 参数续优化。
5. **指标守护**：最终在原始坐标系下分别评估“仅刚性基线（原拓扑）”与“形变+细分结果”，
   只有当 `chamfer_l1` 改善时才保留细分结果，否则回退到刚性基线。

主流程：`normalize → rigid ICP → [ optimize-to-plateau → localize → subdivide ]×K → 反归一化评估`。
单元/冒烟测试验证：四面体单面/全面细分后仍 watertight、winding 一致、Euler=2；合成球面端到端跑通，
平台期触发细分、`new_self_intersections=0`、指标不劣于刚性基线。

## 9. 复现方式

环境与设备：

```bash
cd /home/lichanghao/github/Watertight/non-rigid-icp
export PATH=/vepfs-cnbja62d5d769987/lichanghao/miniconda3/envs/flux/bin:$PATH
# 注意：flux 环境的 ninja 不在默认 PATH，需如上 export 才能 JIT 编译 chamfer 扩展
```

先跑单元测试与端到端冒烟（秒级，验证原子函数与整条流水线）：

```bash
CUDA_VISIBLE_DEVICES=2 python -m non_rigid_icp.Test.test_geometry
CUDA_VISIBLE_DEVICES=2 python -m non_rigid_icp.Test.test_fitter_smoke
```

运行 v4 基础拟合（默认即 v4 配置，仍含自交守护，可关）：

```bash
CUDA_VISIBLE_DEVICES=2 python -c "from non_rigid_icp.Demo.watertight_fitter import demo; demo(max_subdivisions=0)"
```

运行 v4 + 自交守护 + K=4 自适应细分（本次新增能力）：

```bash
CUDA_VISIBLE_DEVICES=2 python -c "from non_rigid_icp.Demo.watertight_fitter import demo; \
  demo(max_subdivisions=4, collision_refresh_every=10, error_quantile=0.97, \
       save_result_folder_path='./output/case1_refine/')"
```

对任意输出 mesh 做独立评估：

```bash
CUDA_VISIBLE_DEVICES=2 python -m non_rigid_icp.Eval.evaluate_case1 \
  --source <fitted_mesh.ply> \
  --target /nvme0pnt/lichanghao/chLi/Dataset/watertight/watertight_case/case1_gen.glb \
  --samples 2000000
```

输出产物（`output/<run>/`）：`fitted_mesh.ply`、`metrics.json`、`config.json`、
`history.json`、`refine_log.json`（每轮细分的面/顶点增量与误差统计）。
`metrics.json` 还含 `final_new_self_intersections`（应为 0）。

---

## 10. 迭代调参准则

- chamfer_l1 越低越好，并拆分 fit（source→target，贴合误差）与 cov（target→source，覆盖误差）。
- precision 低：水密网格有区域偏离原始表面 → 加强数据项 / 收紧掩码 / 增大 point-to-plane。
- recall 低：原始表面被覆盖不足 → 提高覆盖侧权重 / 增大 source 采样密度。
- 局部塌缩或撕裂：提高 Laplacian / 放慢退火 / 降低 lr。
- 整体对不齐：先检查目标坐标系归一化与 ICP，而非直接调非刚性参数。
- 出现新增自交（`final_new_self_intersections>0`）：增大 `collision_weight` / 减小 `collision_refresh_every`
  （更勤刷新候选）/ 增大 `collision_margin_tau` / 降低 lr。
- 细分面数增长过快/显存吃紧：提高 `error_quantile`（如 0.97→0.99）、降低 `max_refine_faces`、
  减小 `dilation_rings`、减小 `max_subdivisions`。
- 平台期判定过早/过晚：调 `plateau_window`、`plateau_rel_tol`、`plateau_patience`。
- 细分未触发：多为误差已普遍低于 `error_mult·tau`（已接近分辨率上限），可适当降低 `error_mult`
  或提高细分预算，但收益受物理上限约束（见第 5 节）。

---

## 11. case1 全量结果（自交守护 + K=4 细分）

> 运行命令见第 9 节；输出在 `output/case1_refine/`。下表在 2M 采样、相同随机种子下与 v4 对比。
> 总耗时（拟合 + 评估）≈ 996s（A800-80GB）。

| 版本 | F1@(L/2048) | precision | recall | chamfer_l1 | chamfer_l2 | 顶点/面 | 新增自交 |
|---|---|---|---|---|---|---|---|
| 刚性基线 | 0.4509 | 0.4506 | 0.4513 | 0.0011088 | 7.565e-7 | 14.45M / 28.91M | 0 |
| v4 | 0.4868 | — | — | 0.0010591 | — | 14.45M / 28.91M | （未约束） |
| **v4 + 守护 + K=4 细分** | **0.4879** | 0.4881 | 0.4877 | **0.0010574** | 7.117e-7 | 23.38M / 46.77M | **0** |

相对刚性基线：**F1 +8.2%（0.4509 → 0.4879）、chamfer_l1 −4.6%**；相对 v4 再小幅提升
（F1 0.4868 → 0.4879、chamfer_l1 0.0010591 → 0.0010574），且**额外保证了全程无新增自交**。
独立评估脚本（`output/case1_refine/eval_standalone.json`）复现一致：F1 = 0.4885、chamfer_l1 = 0.0010566。

四轮自适应细分（`refine_log.json`，每轮标记面均落在 `max_refine_faces=1.5M` 上限内，增长受控）：

| 轮次 | 标记面数 | 面：before → after | 顶点：before → after | 阈值 | 该区最大面误差 |
|---|---|---|---|---|---|
| 0 | 1,136,722 | 28.91M → 33.18M | 14.45M → 16.59M | 9.10e-4 | 0.01929 |
| 1 | 1,257,481 | 33.18M → 37.70M | 16.59M → 18.85M | 1.13e-3 | 0.01930 |
| 2 | 1,348,288 | 37.70M → 42.29M | 18.85M → 21.15M | 1.31e-3 | 0.01930 |
| 3 | 1,393,609 | 42.29M → 46.77M | 21.15M → 23.38M | 1.49e-3 | 0.01930 |

结论要点：

- 自交守护保证整条优化轨迹**不引入新的自交**（`final_new_self_intersections=0`），相邻面与水密
  初始接触不被误判；其代价随**运动量**而非网格规模增长（仅对位移≳tau 的活动面建候选并设硬上限）。
- 自适应细分仅在接近收敛后、针对**误差超过容差 `error_mult·tau`** 的局部区域进行，保形且水密；
  4 轮内面数 28.9M → 46.8M（+62%，受控）。
- 增益幅度有限是因为 source 已接近 1536³ 的离散分辨率上限（见第 5 节）：`face_error_max≈0.0193`
  的硬骨头多为水密网格与原始网格的**拓扑/特征差异**，单纯 warp+细分难以完全消除；
  细分主要把这些高误差局部的离散误差进一步压低，从而带来 F1 与 chamfer 的稳定改善。
