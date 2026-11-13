# 临时可用的公开细胞器数据集：OpenOrganelle 深挖结果

[English](PUBLIC-DATA-OPENORGANELLE.md) ｜ **中文**

> 背景：结项在即，自有标注来不及补齐。本文记录在公开数据里找到的、**经实测确认可用**
> 的细胞器标注，以及它们的真实规模。
>
> **所有结论都来自实测**（S3 前缀列表、`.zarray` 元数据、实际下载并看图核对），
> 不是根据网页介绍推断的。文中特别标出了"目录存在但没有数据"这类陷阱。

---

## 1. 一句话结论

**OpenOrganelle（Janelia CellMap）的 `groundtruth` crop 里有可用的高尔基体、
内质网、线粒体人工标注，且能与 EM 图像精确对齐，1788 层高尔基体切片，
下载量约 0.5 GB。**

但它**不是**"83 个 crop 的高尔基体"——真正有 golgi 数据的只有 **5 个 crop**，
其中同时含 golgi 与 er/mito 的是 **4 个**。这个区别很重要，见 §3。

---

## 2. 找到了什么

### 2.1 数据源

| 项 | 值 |
|---|---|
| 名称 | OpenOrganelle / CellMap（Janelia Research Campus） |
| 地址 | `s3://janelia-cosem-datasets`（公开，无需账号） |
| 许可 | 见站点说明；引用需注明 CellMap |
| 数据集数 | **105 个**（含 `csc-zipped-data` 打包目录） |
| 格式 | zarr v2（groundtruth crop）、N5（全卷标注）、precomputed（网格） |

### 2.2 目录结构（实测）

```
<数据集>/
├── <数据集>.n5/labels/<类型>/          全卷标注，N5（z5 自定义块头，难解析）
├── <数据集>.zarr/recon-1/
│   ├── em/fibsem-uint8/s0..s2/         原始 EM，多尺度
│   └── labels/groundtruth/<crop>/<通道>/s0/   ★ 人工标注 crop，标准 zarr
└── neuroglancer/viewer_states/<crop>.json   ★ crop 坐标（关键）
```

**`groundtruth` 才是可用的那一路**：标准 zarr v2 + zstd，`dimension_separator`
为 `/`，每块 64³。而 `.n5` 的块带 z5 自定义头（`ndim` + `blockShape` 前缀，
压缩方式与属性不一致），逆向成本高且没必要。

### 2.3 每个 crop 有多少类标注

以 `jrc_hela-2/crop1` 为例，共 **64 个通道**：

```
all  bm  cell  cent  cent_all  cent_dapp  cent_sdapp  chrom  cyto  echrom
ecs  endo  endo_lum  endo_mem  er  er_lum  er_mem  er_mem_all  eres  eres_lum
eres_mem  glyco  golgi  golgi_lum  golgi_mem  hchrom  isg  isg_ins  isg_lum
isg_mem  ld  ld_lum  ld_mem  lyso  lyso_lum  lyso_mem  mito  mito_lum  mito_mem
mito_ribo  mt  mt_in  mt_out  ne  ne_lum  ne_mem  ne_mem_all  nechrom  nhchrom
np  np_in  np_out  nuc  nucleo  nucpl  perox  perox_lum  perox_mem  pm  rbc
tbar  ves  ves_lum  ves_mem
```

标注语义写在 `.zattrs` 里：

```json
{"cellmap":{"annotation":{
  "annotation_type":{"encoding":{"absent":0,"present":1,"unknown":255},
                     "type":"semantic_segmentation"},
  "class_name":"golgi","complement_counts":{"absent":160872207,"present":10487793,"unknown":0}}}}
```

即 **0=无、1=有、255=未知**。`unknown` 这一档很有用：它正是我们缺的
"这块没标注"信息，可直接映射为忽略区。

---

## 3. 真实规模：目录存在 ≠ 有数据

zarr 只写非全零的数据块，所以一个 `golgi/s0/` 目录里可能**只有 `.zarray`
和 `.zattrs`，一个数据块都没有**。

实测 `jrc_hela-2/crop1/golgi/s0/` 就是这种情况：按目录名统计会以为"有标注"，
实际取出来全是零。**我第一次就踩了这个坑**，报了"83 个 crop 有高尔基体"，
逐 crop 检查后实际只有 5 个。

### 3.1 逐通道有数据的 crop 数

| 通道 | hela-2 | hela-3 | jurkat-1 | macrophage-2 | 合计 |
|---|---|---|---|---|---|
| **golgi / golgi_mem** | **1** | **0** | **2** | **2** | **5** |
| er / er_mem | 16 | 8 | 10 | 9 | 43 |
| mito / mito_mem | 7 | 5 | 7 | 5 | 24 |
| endo | 14 | 8 | 10 | 8 | 40 |
| ves | 17 | 7 | 8 | 9 | 41 |
| nuc | 8 | 7 | 5 | 4 | 24 |
| pm | 5 | 7 | 6 | 7 | 25 |
| ld（脂滴） | 1 | 0 | 0 | 1 | 2 |
| lyso | 3 | 1 | 4 | 8 | 16 |
| perox | 0 | 0 | 0 | 0 | **0** |

### 3.2 可用的四个 crop（同 crop 内同时有 golgi 与 er/mito）

| 数据集 / crop | 形状 (z,y,x) | golgi 层数 | 分辨率 |
|---|---|---|---|
| `jrc_hela-2` / `crop4` | 476 × 600 × 600 | 476 | 4.0 nm 各向同性 |
| `jrc_jurkat-1` / `crop37` | 400 × 400 × 400 | 400 | 4.0 nm |
| `jrc_macrophage-2` / `crop39` | 400 × 400 × 400 | 400 | 4.0 nm |
| `jrc_macrophage-2` / `crop48` | 512 × 512 × 512 | 512 | 4.0 nm |
| **合计** | | **1788** | |

对比：我们自有的高尔基体标注是 **53 张二维图**。这四份数据是**完整标注**
（不是部分标注），因此正好补上我们最大的短板。

### 3.3 第二次更正：还要看**前景占比**，不能只看"有没有块"

下载完成后逐通道算了前景占比，结果与 §3.2 的预期不符：

| crop | golgi 前景 | 可用性 |
|---|---|---|
| hela-2 / crop4 | **6.12%** | 可用 |
| jurkat-1 / crop37 | **29.17%** | 可用 |
| macrophage-2 / crop39 | **25.63%** | 可用 |
| macrophage-2 / crop48 | **0.05%**（仅 4 个块） | **实质为空** |

`crop48` 在 §3.1 的清单里被算作"有 golgi"，因为它确实有 4 个非空块。
但 0.05% 的前景意味着那只是零星几个体素，做训练没有意义。

同样地，`mito` 通道实际只有 3 个 crop 有内容（crop4 为空，crop37 仅 0.20%）：

| crop | mito 前景 |
|---|---|
| hela-2 / crop4 | 0% |
| jurkat-1 / crop37 | 0.20% |
| macrophage-2 / crop39 | 7.29% |
| macrophage-2 / crop48 | 19.62% |

**结论：真正可用于高尔基体训练的是 3 个 crop（crop4 / crop37 / crop39），
共 1276 层**，而不是 1788 层。§3.1 的"有数据"判据（存在至少一个非元数据键）
粒度太粗，必须配合前景占比一起看。

> 这是同一个陷阱的第二层：先是"目录存在 ≠ 有数据"，再是
> "有数据 ≠ 数据有意义"。两层都需要实际取下来算一算才能发现。

---

## 4. 与 EM 的对齐（这是最关键的一步）

crop 里**只有标注，没有图像**，而 `.zattrs` 里**没有偏移量**。
坐标藏在：

```
neuroglancer/viewer_states/neuroglancer-<数据集>-<crop>.json  →  "position"
```

`position` 是网页查看器跳转到该 crop 时的视点，单位**纳米**，顺序 `(x, y, z)`。
换算：

```
中心体素 = position_nm / resolution_nm        # 分辨率见 em/.../s0/.zattrs
crop 原点 = 中心体素 − crop形状 / 2
EM 全局范围 = crop 原点 .. 原点 + crop形状
```

实测得到的四个 crop 的 EM 范围：

| crop | crop 原点 (z,y,x) | EM 范围 |
|---|---|---|
| hela-2/crop4 | 3465, 50, 5950 | z[3465,3941] y[50,650] x[5950,6550] |
| jurkat-1/crop37 | 4437, 696, 3118 | z[4437,4837] y[696,1096] x[3118,3518] |
| macrophage-2/crop39 | 4405, 603, 5435 | z[4405,4805] y[603,1003] x[5435,5835] |
| macrophage-2/crop48 | 4482, 451, 5979 | z[4482,4994] y[451,963] x[5979,6491] |

### 4.1 两套坐标系（这里踩过一次）

* **EM** 用**全卷**体素坐标
* **crop 标注**用**局部**坐标，永远从 (0,0,0) 起

所以取同一块内容要分别换算：`crop_local = em_global - crop_origin`。
把全局坐标直接喂给标注读取函数，会得到负维度报错，**或者更糟——静默取到错误位置**。

### 4.2 看图核对

坐标是推断出来的，必须验证。做法：在 crop4 最密集的 golgi 层取 256×256 的
EM 与标注叠在一起渲染（`temp/oo_align3/align.png`）。

结果：**黄色标注精确覆盖 EM 中四条平行的暗色池层带及其上方的团块**。
坐标、纳米换算、两套坐标系的映射全部正确。

这一步不能省。图像与标注错位是训练中最难发现的错误——loss 照样下降，
模型只是学到一堆错配。

---

## 5. 下载成本（实测）

| 内容 | 体积 | 耗时 |
|---|---|---|
| crop4 golgi 全部（476×600×600） | **0.8 MB** | 约 10 分钟 |
| crop4 er 全部 | **0.9 MB** | 约 7 分钟 |
| crop4 一小块 EM（4×256×256） | 12.7 MB | 约 1 分钟 |

标注本身极小（稀疏 + zstd）。**EM 才是体积主体**：按 128³ 块取，
一个 476×600×600 的 crop 约 100 个块、未压缩 200 MB，压缩后约 130 MB。

四个 crop 的 EM 估算 **0.5–0.8 GB**，服务器 `/home` 目前剩 56 GB，足够。

> 注意：EM 是噪声图像，压缩率低（约 0.7）。按块取时若只用到块的一小部分，
> 会浪费带宽——例如取 4 层 × 256 × 256 实际拉了 9 个完整的 128³ 块。

---

## 6. 建议的使用方式

### 6.1 优先级 1：高尔基体（补最大短板）

取 4 个 crop 的 `golgi` + EM，切成 2D 切片训练。规模 1788 层，
且是**完整标注**，按我们已有的分层证据（覆盖率 ≥8% 时 Dice 0.60–0.71），
这批数据应当能把高尔基体的指标从 0.30 拉起来。

```bash
# 1) 解析坐标（已做，结果在 logs/oo_locate.json）
python tools/oo_locate.py

# 2) 取标注
python tools/oo_crop_fetch.py --dataset jrc_hela-2 --crop crop4 \
    --channels golgi,er,mito --out data/oo/hela2_crop4

# 3) 取对应 EM（全局坐标）
python tools/oo_fetch.py --dataset jrc_hela-2 --labels "" \
    --x 5950 6550 --y 50 650 --z 3465 3941 --out data/oo/hela2_crop4_em
```

### 6.2 优先级 2：多类互作（er + mito 有 43 / 24 个 crop）

我们此前的多类互作只能用 ASEM 的一份数据。现在可以扩到几十个 crop，
且**每个 crop 内 er/endo/ves/lyso 等多类同时可用**，足以做真正的异类接触统计。

### 6.3 优先级 3：`unknown` 通道 → 忽略区

`.zattrs` 里的 `unknown: 255` 正是"这块没标注"的权威标记。
这可以直接喂给已经实现并单测通过的忽略区机制（`src/test_ignore_mask.py`），
是那个机制第一个**可信的、非人为构造的**输入来源。

### 6.4 不建议的方向

`csc-zipped-data`（CellMap Segmentation Challenge 打包数据）最小的一份是
**42 GB**，最大 1.57 TB。服务器只剩 56 GB，且我们的需求用 crop 就够，
没有必要下载。

`perox`（过氧化物酶体）在所有 4 个数据集里都**没有数据**，不要指望。
`ld`（脂滴）只有 2 个 crop，若要用需另外核实。

---

## 7. 新增的工具

| 工具 | 作用 |
|---|---|
| `tools/oo_scan.py` | 扫描全部 105 个数据集，列出各自标注了哪些细胞器 |
| `tools/oo_inventory.py` | 逐 crop 逐通道检查**是否真有数据块**（避开"目录存在"陷阱） |
| `tools/oo_locate.py` | 从 viewer state 解析 crop 在 EM 中的坐标 |
| `tools/oo_crop_fetch.py` | 取 groundtruth crop 的标注（zarr + zstd） |
| `tools/oo_fetch.py` | 取 EM 体（zarr + zstd）与 N5 标注 |
| `tools/probe_zarr_key.py` | 探测 zarr 的 chunk key 分隔符 |
| `tools/verify_oo_alignment.py` | 出图核对标注与 EM 是否对齐 |

### 7.1 踩过的六个坑（都写进了代码注释）

1. **标注在 `.n5` 下，不在 `.zarr/recon-1/labels/` 下** —— 按后者扫描得到"全部 0 个"。
2. **目录存在 ≠ 有数据** —— zarr 省略全零块，`golgi/s0/` 可能只有 `.zarray`
   和 `.zattrs`。按目录名统计得到 5 个 crop，实际只有 4 个有块。
3. **有数据 ≠ 有意义** —— crop48 有 4 个非空块但前景仅 0.05%，实质为空。
   必须算前景占比才能判断。
4. **chunk key 分隔符是 `/` 不是 `.`** —— 用点号会全部 404，表现为"每块都缺"，
   代码不报错。ASEM 的数据相反，用点号。
5. **EM 与 crop 标注两套坐标系** —— 全局与局部，混用会取到错误位置。
6. **N5 的块有 z5 自定义头** —— 不是裸 gzip，`gzip.decompress` 会报
   `Not a gzipped file`。改用 `.zarr/groundtruth` 绕开。

### 7.2 两个自己写出来的 bug（值得记下）

**(1) 列清单传了完整 URL。** `list-type=2` 的 `prefix` 参数要的是**键前缀**
（`jrc_hela-2/jrc_hela-2.zarr/...`），我直接把含 `https://` 的完整 URL 传了进去。
S3 按字面量匹配，返回 0 条；随后"待取块"为空、保存了**全零数组**。
三个环节（列清单空、取数 0 块、写盘全零）**没有一个报错**，
表现是"下完了 1.7 GB 但前景全是 0.00%"。

**(2) 对稠密体也列了清单。** 标注稀疏，先列清单再取是有益的（crop4 的 golgi
只取 233 块而非 800 块）。但 EM 是稠密图像体，块几乎全部存在，列清单毫无收益：
hela-2 的 EM 有 **127 033 个分块键**，分页耗时 **495–721 秒**，
比真正取数（约 30 秒）慢十倍。现在 `fetch_zarr_volume` 有 `dense` 参数：
标注走清单，EM 直接按网格取。

> 两个 bug 的共同点：**失败是静默的**。第一个存了全零却不报错，
> 第二个只是慢。因此取数后必须验证前景占比，而不是看目录大小——
> 1.7 GB 的全零数组与 1.7 GB 的真实数据在 `du` 里长得一样。
