# 水密网格贴合原始网格：算法、实验与结论

本文记录将“水密化后的网格（source）”精准贴合到“原始网格（target）”的算法设计、
工程实现、关键实验结论与推荐配置。任务以 case1 为例：

- source（待形变，水密网格）：`/nvme0pnt/lichanghao/chLi/Dataset/watertight/watertight_case/case1_wt_1536.ply`
- target（拟合目标，原始网格）：`/nvme0pnt/lichanghao/chLi/Dataset/watertight/watertight_case/case1_gen.glb`

最终采用的版本为 **v4**，作为当前唯一保留版本（demo 与 `WatertightFitter` 默认参数已对齐 v4）。

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

- `non_rigid_icp/Module/watertight_fitter.py`：核心类（`fit` / `evaluate` / `fitAndEvaluate` / `saveResult`）
- `non_rigid_icp/Method/sampling.py`：目标坐标系归一化、表面采样、可微 deformed 表面采样
- `non_rigid_icp/Method/nn.py`：GPU 空间最近邻索引 `NNIndex`
- `non_rigid_icp/Loss/surface.py`：对称 Chamfer、point-to-plane、边 Laplacian 损失
- `non_rigid_icp/Metric/chamfer.py`：双向 Chamfer + F1@tau（基于快速 NN，支持百万级点）
- `non_rigid_icp/Demo/watertight_fitter.py`：默认即 v4 配置的 demo
- `non_rigid_icp/Eval/evaluate_case1.py`：对任意输出 mesh 的独立评估脚本

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

## 7. 复现方式

环境与设备：

```bash
cd /home/lichanghao/github/Watertight/non-rigid-icp
export PATH=/vepfs-cnbja62d5d769987/lichanghao/miniconda3/envs/flux/bin:$PATH
# 注意：flux 环境的 ninja 不在默认 PATH，需如上 export 才能 JIT 编译 chamfer 扩展
```

运行 v4（默认参数即 v4 配置）：

```bash
CUDA_VISIBLE_DEVICES=2 python -c "from non_rigid_icp.Demo.watertight_fitter import demo; demo()"
```

对任意输出 mesh 做独立评估：

```bash
CUDA_VISIBLE_DEVICES=2 python -m non_rigid_icp.Eval.evaluate_case1 \
  --source <fitted_mesh.ply> \
  --target /nvme0pnt/lichanghao/chLi/Dataset/watertight/watertight_case/case1_gen.glb \
  --samples 2000000
```

输出产物（`output/case1_v4/`）：`fitted_mesh.ply`、`metrics.json`、`config.json`、
`history.json`、`eval_standalone.json`。

---

## 8. 迭代调参准则

- chamfer_l1 越低越好，并拆分 fit（source→target，贴合误差）与 cov（target→source，覆盖误差）。
- precision 低：水密网格有区域偏离原始表面 → 加强数据项 / 收紧掩码 / 增大 point-to-plane。
- recall 低：原始表面被覆盖不足 → 提高覆盖侧权重 / 增大 source 采样密度。
- 局部塌缩或撕裂：提高 Laplacian / 放慢退火 / 降低 lr。
- 整体对不齐：先检查目标坐标系归一化与 ICP，而非直接调非刚性参数。
