# 公开细胞器数据获取（外接盘专用）

> 目的：补上本项目最硬的约束——**现有 570 个区域全部是同一个类别 `M`（线粒体）**，
> 没有第二类细胞器，就谈不上"哪两类之间有接触/共定位"。
> 本目录把一个或多个公开数据集抓到外接盘，供后续训练与互作分析使用。

---

## 0. 先看清：为什么要抓公开数据

审计已经确认的三条硬事实（见 [`../../docs/AUDIT-REPORT.md`](../../docs/AUDIT-REPORT.md)）：

| 事实 | 后果 |
|---|---|
| 黄色标注在全部 89 张图里都是同一个字形 `M` | **没有多类细胞器**，DETR 的 `num_classes` 只能是 1 |
| 每图细胞器数中位数只有 5 个 | `knn(k=6)` 有 71% 的图退化成完全图 |
| 88 张图分组后时间点数是 13/49/2/20/4 | 三组不足以做动力学分析 |

**第一条是致命的**。项目标题是「细胞器互作网络重建」，但数据里只有一类细胞器——
这决定了必须引入公开的多类标注数据，否则这条研究线走不通。

---

## 1. 数据集目录（体积均已实测核实）

运行 `python tools/data_acquire/fetch.py --list` 可随时查看。摘要：

| 优先级 | 数据集 | 体积 | 模态 | 关键价值 |
|---|---|---|---|---|
| **1** | **CellMap Segmentation Challenge**（matched-res） | **42.02 GB** | eFIB-SEM，4–8 nm | **73 类**，含 `mito` + `er` + `lyso` + `ld` + `perox` + `golgi`；289 个多人交叉校验的训练体 |
| 2 | EMPIAR-12885 | ~32.8 GB | FIB-SEM，TIFF | 主题直指**线粒体接触位点**；TIFF 可直接读 |
| 2 | BioStudies S-BIAD2983 | 待核实 | 多光谱共聚焦 | **8 种细胞器** + 真实时间序列（day7/14/21/28） |
| 3 | CellMap（128 vox padding） | 116.80 GB | eFIB-SEM | 同 tier1 但四周多 128 体素上下文 |
| 3 | EMPIAR-12017 | 2.9 TB（仅取线粒体子集） | FIB-SEM，TIFF | 肝脏 ER–mito 空间关系 |
| 3 | 3DMSL | 待核实 | 模拟荧光 | 线粒体三维形态库 |

### 1.1 首选：CellMap Segmentation Challenge

- 数据门户：<https://openorganelle.janelia.org/>
- 挑战主页：<https://cellmapchallenge.janelia.org/page/about>
- 官方工具库：<https://github.com/janelia-cellmap/cellmap-segmentation-challenge>
- 数据集合 DOI：<https://doi.org/10.25378/janelia.c.7456966>
- 直连地址（**本次实测可匿名访问、支持断点续传**）：
  ```
  https://janelia-cosem-datasets.s3.amazonaws.com/csc-zipped-data/matched_res_no_pad.zip
  ```
- 许可：COSEM/CellMap 官方 registry 标注 **CC-BY-4.0**
  （AWS Open Data 条目 <https://registry.opendata.aws/janelia-cosem/>）

**为什么它正好解决本项目的问题**：其 `classes.csv`
（<https://github.com/janelia-cellmap/cellmap-segmentation-challenge/blob/main/src/cellmap_segmentation_challenge/utils/classes.csv>）
里同时给出：

```
mito, mito_mem, mito_lum, mito_ribo,
er,   er_mem,   er_lum,   er_mem_all,
lyso, lyso_mem, lyso_lum,  ld, ld_mem, ld_lum,
perox, perox_mem, perox_lum, golgi, ves, endo, nuc, np, cell, ...
```

也就是说 **mito 与 er 都有实例/语义标注** → 可以直接做 **mito–ER 接触位点**
（membrane contact site）的接触长度、接触面积、接触对数量统计。
这正是"细胞器互作网络"最经典、最有文献支撑的一条边。

规模（官方 README 与本次实测）：

- 289 个训练体，来自 22 套 eFIB-SEM，物种含小鼠/果蝇/斑马鱼/人，组织含脑/心/肝/肾
- 73 类（其中 47 类在 ≥3 个数据集里出现，是排行榜评估的核心集）
- 体素 4 nm 或 8 nm 各向同性
- 官方默认下载（Zarr 格式）约 **36.7 GB**；本 zip 为 **42.02 GB**（含 zip 开销）

### 1.2 为什么排除几个看起来更大的选项

| 被排除 | 体积 | 理由 |
|---|---|---|
| CellMap `all_res_no_pad.zip` | 1229 GB | 与 matched-res 内容重复但分辨率更高；先做 2D 检测用不到全分辨率 |
| CellMap `all_res_128_pad.zip` | 1571 GB | 同上，且加 padding |
| EMPIAR-10791 | 2.8 TB | 与 EMPIAR-12017 高度重叠（同为肝脏 FIB-SEM），取 12017 的线粒体子集即可 |

---

## 2. 用法

```bash
# 看目录
python tools/data_acquire/fetch.py --list

# 只探测：可达性、真实体积、目标盘空间够不够（不写盘）
python tools/data_acquire/fetch.py --target W:\dachuang_public_data --dry-run

# 下第一优先级（约 42 GB）
python tools/data_acquire/fetch.py --target W:\dachuang_public_data --only cellmap_challenge_matched

# 下 tier<=2 的全部
python tools/data_acquire/fetch.py --target W:\dachuang_public_data --max-tier 2
```

把 `W:` 换成你外接盘的实际盘符。

### 工具做了什么

1. **空间预检**：下载前检查目标盘剩余空间（留 5% 余量），不够直接拒绝，
   不会写到一半磁盘满；
2. **断点续传**：HTTP `Range` + `.part`，中断后重跑同一条命令即可继续
   （已实测：截断到 40% 再续传，结果与原文件**逐字节一致**）；
3. **尺寸校验**：与 catalog 里实测过的字节数比对，不符即报错，
   不留"看着下完了其实是截断"的文件；
4. **清单留痕**：每集目录写 `dataset.json`，根目录写 `manifest.json`，
   记 url / 大小 / sha256 / 抓取时间——论文里可直接引用来源。

---

## 3. 已知的后续工作（抓完不等于能用）

| 事项 | 说明 |
|---|---|
| **zip 解压后是 Zarr** | 不是 TIFF。本项目已装 `zarr 3.3.0`，可直接读；但现有 `tools/morpho` 的输入是 TIFF 掩码，需要写一层 Zarr → 2D TIFF/PNG 切片导出 |
| **3D → 2D 的取舍** | CellMap 是体数据，本项目 DETR 是 2D。做法：沿 z 轴取切片，或用最大密度投影（MIP）。**必须记录切片策略**，否则论文里说不清"你的 2D 图是怎么来的" |
| **类别映射** | CellMap 的 73 类与我方 `M` 需要建立映射表（至少 `mito → M`）；映射表要落盘成文件 |
| **分辨率不一致** | CellMap 4–8 nm/px，本项目 1–5.3 nm/px（`scale_mat` 有三个取值）。跨数据集比较**绝对物理尺寸**仍然无效，只能比无量纲量 |
| **许可与引用** | CC-BY-4.0 要求署名。使用前把 `citation` 字段抄进论文参考文献 |
| **S-BIAD2983 体积未核实** | 其文件清单是 `BioImageArchive-mcz.json`（MCZ 格式），需先解析清单才能知道体积与下载地址 |

---

## 4. 与"ABC 方案"的关系

`docs/INTEGRITY-NOTES.md` 里的 ABC 是**关于伪造实验元数据怎么处理**的三个选项，
与本目录的数据获取是**两件独立的事**：

- **ABC** 处理的是 `data/structure/structure_metadata.json` 里那些
  由随机数生成的 `actual_date` / `perturbation_level` / `experiment_conditions`；
- **本目录**处理的是"数据里只有一类细胞器、样本量太小"。

两者可以并行推进：ABC 决定旧数据怎么处置，公开数据决定新的分析能不能做起来。
但注意：**即使抓了 CellMap，也不能反过来把旧报告里"仅 2–4 个时间点"的
动力学结论洗白**——那是另一个问题。

---

## 5. 当前状态

| 项 | 状态 |
|---|---|
| 数据集评估与体积核实 | ✅ 完成（体积均实测） |
| 下载工具 | ✅ 完成，`--dry-run` / 断点续传 / 尺寸校验 / 清单均已实测 |
| 外接盘 | ❌ **尚未接入**（本机当前只有 `C:`、`D:`） |
| 实际下载 | ⏸ 等外接盘盘符确定后执行 |
| Zarr → 2D 切片导出 | ⏳ 未开始（依赖数据先到位） |

外接盘接入后请运行：

```bash
python tools/data_acquire/fetch.py --target <盘符>:\dachuang_public_data --dry-run
```

先看空间是否够（首推数据集需要 42 GB，tier1+2 合计约 75 GB），再决定下哪些。
