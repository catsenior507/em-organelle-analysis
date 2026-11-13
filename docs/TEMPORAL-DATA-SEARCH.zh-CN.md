# 时序（时间轴）数据集检索结果

[English](TEMPORAL-DATA-SEARCH.md) ｜ **中文**

> 目标：本项目的动力学分析（李雅普诺夫指数、转移熵）在静态电镜快照上原理上不适用，
> 这一点已在 `docs/MATLAB-VERIFICATION-STATUS.zh-CN.md` 与
> `docs/GOLGI-OPENORGANELLE-RESULTS.zh-CN.md` 中记录。要让它成立，唯一出路是拿到
> **真的有时间轴**的数据。本文记录检索的过程与结论。
>
> **结论：目前没有可用的、公开的、带时间轴的细胞器电镜数据。**
> 这个空白是结构性的，不是检索不力。

---

## 1. 检索了哪些来源

| 来源 | 端点/方式 | 结果 |
|---|---|---|
| EMPIAR | `www.ebi.ac.uk/empiar/api/` | 仅暴露 `latest_citations` 一个端点，**无通用搜索 API**；HTML 无查询接口 |
| BioStudies（EMPIAR 的上级索引） | `biostudies/api/v1/search` | 可用，但 EMPIAR 条目不在其可检索集合内（`collection=empiar` 返回 0） |
| IDR（Image Data Resource） | `idr.openmicroscopy.org/api/v0/m/projects/` | **可用**，147 个研究 |
| Cell Tracking Challenge | `celltrackingchallenge.net` | 可用，但为细胞/细胞核追踪荧光数据 |
| BBBC | `bbbc.broadinstitute.org` | 可用，同上 |
| OpenOrganelle | `janelia-cosem-datasets.s3` | 已全量扫描过，**单一时间点**，无时序子集 |
| Valm 2017 细胞器互作组 | PMC 全文 | 论文**无公开数据存放**，仅补充材料 |

### 1.1 一个必须记录的失败

最初我假设了 `empiar/api/search?q=` 与 `idr/api/v0/m/studies/` 两个端点，**都返回 404**。
而我的抓取函数把"HTTP 错误"与"没有结果"都归成空列表，于是输出会显示
"命中 0 条"，看起来像"确实不存在时序数据"。

**这是会直接导致错误结论的失败模式。** 后来加了 `tools/probe_endpoints.py`，
先打印每个候选 URL 的状态码与响应前 120 字节，确认端点真实存在再解析。
实测 IDR 的正确路径是 `m/projects/`（不是 `m/studies/`）。

---

## 2. IDR 的筛选结果

147 个 project 中：

| 类别 | 数量 |
|---|---|
| 含细胞器关键词（mitochondri/golgi/ER/lysosome/organelle/…） | 28 |
| 同时含时序关键词（time-lapse/live cell/dynamics/tracking/…） | **10** |

这 10 个的标题：

| ID | 名称 | 时序关键词 |
|---|---|---|
| 151 | idr0027-dickerson-chromatin | live cell, movie, tracking |
| 352 | idr0038-held-kidneylightsheet | live cell, tracking |
| 904 | idr0067-king-yeastmeiosis | time-lapse |
| 1551 | idr0071-feldman-crisprko | live-cell |
| 1102 | idr0079-hartmann-lateralline | live imaging |
| 1157 | idr0087-paci-nuclearimport | time-lapse |
| 1601 | idr0097-reicher-proteintag | time-lapse |
| 2301 | idr0115-otsuka-nucleoporins | time-lapse |
| 2101 | idr0118-keenan-flylightsheet | dynamics, time-lapse |
| 3351 | idr0167-li-cellcyclenet | dynamics |

**没有一个是"多种细胞器在活细胞中的时序"**——它们是染色质、核孔、核输入、
酵母减数分裂、斑马鱼侧线等。按"细胞器互作"这个目标看，都不对题。

> 关键词匹配必然有假阳性，所以这里给出标题供人工判断，而不是直接下结论。
> 逐个看完这 10 个标题后可以说：没有可用的。

---

## 3. 为什么这个空白是结构性的

电镜需要固定、脱水、切片或离子束减薄。一旦固定，时间轴就消失了。
要在细胞器分辨率上做"活细胞 3D 电镜时序"，目前没有可行的常规手段。

现存的技术与它们的局限：

| 技术 | 有时间轴？ | 为什么用不上 |
|---|---|---|
| 液态池 TEM | 有 | 只适合纳米颗粒、分子，不是细胞器 |
| 原位/操作性 TEM | 有 | 同上 |
| 光镜-电镜关联（CLEM） | 活细胞成像 + 固定后电镜 | 时间轴只存在于光镜那一半，且通常是单时间点 |
| 连续切片/体电镜 | **无** | z 是空间，不是时间 |
| 倾斜序列（tilt series） | **无** | 角度序列，常被误当成"series＝时序" |

**最容易搞错的一点**：`series` 这个词在电镜里几乎总是指倾斜序列或连续切片，
与时间无关。这也是检索时必须显式排除 `tilt series` 的原因。

---

## 4. 对这个项目的含义

### 4.1 强化了已有的诚实结论

`docs/MATLAB-VERIFICATION-STATUS.zh-CN.md` §7 记录了旧脚本"没有网络重建、李雅普诺夫指数是伪造的"。
当时的处理是删除该指标并显式标注 `lyapunov: null`。本文的检索补充了一个更有力的理由：

**即使把代码写对，也没有数据能让它成立。** 这不是实现问题，是数据根本不存在。

### 4.2 论文里该怎么写

结项论文草稿（`the project report` §3.7）已写明"缺少时间维度"这一局限。
可以把本文作为该判断的依据：

> 检索了 EMPIAR、BioStudies、IDR、Cell Tracking Challenge、BBBC、OpenOrganelle
> 六个来源，未找到任何公开的、带时间轴并与细胞器形态学匹配的数据。
> 活细胞成像有（IDR 147 个研究中 10 个含时序与细胞器关键词），但全是荧光而非电镜，
> 且没有一个是多细胞器互作场景。因此本项目的动力学指标只能以空间统计量替代，
> 这一替代关系在文中已明确说明。

### 4.3 如果将来要做

唯一可行的路线是**自己采集**：活细胞荧光时序（多通道标记不同细胞器）+ 固定后单时间点电镜，
用前者提供动力学、后者提供形态学。这已超出一个本科结项项目的范围。

---

## 5. 工具

| 文件 | 作用 |
|---|---|
| `tools/find_temporal_em.py` | 在多个仓库按关键词检索候选（保留为方法记录） |
| `tools/probe_endpoints.py` | **先探端点再解析**，避免把 404 当成"没有数据" |
| `tools/scan_idr.py` | 列 IDR 全部 147 个研究并按细胞器/时序关键词筛选 |

产物：`logs/idr_projects.json`、`logs/temporal_candidates.json`。
