# 细胞器接触位点分析（互作网络的实证基础）

> 项目标题是「细胞器互作网络重建」。此前 `tools/network/` 用**质心距离**
> （Delaunay / kNN / radius）近似"互作"——那只是**空间邻近**，不是互作。
> 本目录做的是更接近生物学定义的事：**两个细胞器的边界在多少像素上直接相邻**
> （membrane contact site）。

---

## 1. 为什么"邻近"不等于"互作"

用质心距离建图有两个问题：

1. **尺度不敏感**：一个巨大的线粒体与一个远隔的小内质网，质心距离可能和
   两个紧贴的细胞器差不多；
2. **会凭空造边**：只要两个类别都在图里，kNN 总能连上，于是"互作网络"
   在任何数据上都显得很稠密——这不构成任何发现。

接触位点（contact site）是细胞生物学里真实存在、可测量的结构：
两个细胞器的膜之间距离只有十几到几十纳米时才算接触。
在光学/电镜分辨率有限的条件下，我们用**像素级相邻**作为代理，
并明确标注这是代理而非真实膜间距。

---

## 2. 指标定义

对类别对 (A, B)（默认 `mito:er`）：

| 指标 | 定义 | 单位 |
|---|---|---|
| `a_pixels` / `b_pixels` | 各类别的像素面积 | px |
| `a_boundary_px` / `b_boundary_px` | 8-连通意义下的边界像素数 | px |
| `contact_length_px` | A 的边界像素中，其 8-邻域含 B 边界像素的个数 | px |
| `contact_per_a_boundary` | `contact_length_px / a_boundary_px` | 无量纲 [0,1] |
| `contact_tol_frac` | A 的边界中，距 B ≤ `rim_px` 的比例（容差版） | 无量纲 [0,1] |
| `a_components` / `b_components` | 8-连通分量数 | 个 |

**为什么同时给"严格"与"容差"两个版本**：手绘/自动分割的标注之间常有 1–2 像素的
缝隙，严格的 8-邻域相邻会系统性漏计；容差版用显式的 `rim_px` 参数刻画
"贴得很近"，且该参数写进输出清单，可复现。

---

## 3. 零模型对照（这一节决定了结论能不能用）

**只看"接触比例 0.3"是没有意义的**——如果 A 和 B 在图里都占很大面积，
它们必然接触。所以必须回答："若 B 只是被随便放到图里某处，会得到什么？"

本工具的做法：把 B 的前景**整体**搬到图内随机合法位置（不环绕、不裁剪，
像素数与形状严格不变），重算接触比例，得到零分布，再报告：

- `p_value`：零分布中 ≥ 观测值的比例（加一修正）
- `z`：`(观测 − 零中位) / 零标准差`
- `significant_p05`：`p < 0.05`

### 踩过的坑：不能用 `np.roll`

最初用 `np.roll` 做环形平移，结果**相邻方块的观测值 0.294，零模型最大值也能追上**
——因为 roll 会把 B 复制到图的对侧，"刚贴完 A 的部分"又从另一边贴回来。
改用"只在图内合法位置放置"后，相邻情形 `p=0.043`（显著），
远离情形 `p=1.0`（不显著），判别方向正确。

### 已验证的解析解

在 100×100 图上造两个 40×30 方块，一个相邻、一个远离：

```
A 边界像素 = 136   （解析值 2*(40+30)-4 = 136）✓
相邻：contact_length = 40，per_A = 0.294，tol = 0.309
远离：contact_length = 0， per_A = 0.0
零模型：相邻 p=0.043 显著；远离 p=1.0 不显著 ✓
```

---

## 4. 用法

```bash
# 先由 zarr_to_2d.py 导出 2D 切片
python tools/data_acquire/zarr_to_2d.py \
    --root E:\dachuang_public_data\cellmap_extracted\data_matched_res_no_pad \
    --dataset jrc_hela-2 --labels mito,er \
    --out E:\dachuang_public_data\cellmap_2d \
    --strategy slice --z-step 20 --max-per-crop 10

# 再算接触
python tools/interaction/contact_analysis.py \
    --slices E:\dachuang_public_data\cellmap_2d \
    --pairs mito:er \
    --out data/contact_analysis
```

也支持多对：`--pairs mito:er,mito:lyso,er:lyso`。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--pairs` | `mito:er` | 类别对 A:B，逗号分隔 |
| `--rim-px` | 2 | 容差接触的半径（像素） |
| `--null-shifts` | 20 | 零模型抽样次数（建议 ≥200 再下结论） |
| `--min-pixels` | 50 | 任一类像素少于此值则跳过该切片 |

### 输出

```
data/contact_analysis/
├── per_slice.json    每个切片的完整统计（含零模型明细）
├── summary.json      汇总 + 跳过原因 + 单位说明
└── metrics.csv       一行一个切片，便于统计软件处理
```

---

## 5. 怎么读结果（写论文时必须照此说明）

1. **`contact_per_a_boundary` 必须与零模型一起报**。只报观测值等于没有对照。
2. **`frac_significant_p05` 很低**（例如 <0.2）说明大多数切片的接触
   只是"两者都很多"的必然结果，**不构成结构性的接触偏好**。
3. **像素不是物理长度**。`contact_length_px` 要换算成 µm 需乘 µm/px，
   而本项目图像有三个不同的标定值（0.0010042 / 0.002667 / 0.005291 µm/px），
   **跨图比较绝对长度无效**。CellMap 数据是 2/4/8 nm/体素，与项目图像也不同。
   可比的只有无量纲量（`contact_per_a_boundary`、`contact_tol_frac`）。
4. **切片策略影响结论**。`slice` 保留真实像素统计；`mip` 会把 3D 结构压扁，
   接触长度会被高估。策略记录在 `zarr_to_2d.py` 的
   `export_manifest.json` 里，论文中必须说明。
5. **这只是接触，不是功能互作**。要谈"互作"，还需要时间上的共变
   （两个细胞器随时间的形态/位置耦合）或已知的分子机制证据。
   本项目目前**没有**真实时间序列，这一点在 `docs/INTEGRITY-NOTES.md` 中已说明。

---

## 6. 与其它模块的关系

```
tools/data_acquire/   公开数据集 → 抓取 → 提取 → 3D→2D 切片
                                    ↓
tools/interaction/    接触位点统计 + 零模型          ← 本目录
                                    ↓
tools/network/        以接触强度为边权构建互作网络
```

`tools/network/` 的 `boundary_contact` 判据与本目录的 `contact_*` 指标
口径一致（都基于边界像素），但 network 那边是建图，本目录是统计量。
两者可以交叉验证：若某类对在图里连边很多，接触统计也应当偏高。

---

## 7. 当前状态

| 项 | 状态 |
|---|---|
| 2D 指标实现 | ✅ 与解析几何一致 |
| 2D 零模型 | ✅ 修正环绕伪影后方向正确 |
| 3D 指标实现 | ✅ 接触面元与解析解一致（8×8=64 面元、4000−8·18·18=1408 表面体素） |
| 3D 零模型 | ✅ FFT 精确解，与暴力实现逐项相等 |
| 合成数据测试 | ✅ 通过 |
| **真实数据（jrc_mus-kidney 5 个 crop）** | ✅ **已完成，见 `docs/CONTACT-3D-RESULTS.md`** |
| 多类别对 | ✅ 支持 `--pairs a:b,c:d` |
| 3D 膜间距离（欧氏距离变换） | ❌ 未实现；下一步优先项 |

---

## 8. 3D 版本（`contact_analysis_3d.py`）

CellMap 的标签是 **3D 体数据**（如 300³），直接做 3D 比切成 2D 切片更可靠：
2D 切片上的"接触"可能只是 3D 结构在某层的投影，会系统性低估接触量。

### 用法

```bash
python tools/interaction/contact_analysis_3d.py \
    --labels ~/dachuang_organelle/data/labels \
    --pairs mito:er \
    --manifest train_crop_manifest.csv \
    --tol-nm 0,16,32,64 \
    --null-mode fft \
    --out data/contact_3d
```

### 三个必须注意的设计决定

1. **容差用纳米，不用体素。** 8 nm/体素的数据里 `tol=8` 是 64 nm，
   4 nm/体素里只有 32 nm——用体素会得出错误结论。`--tol-nm` 会按每个 crop
   自己的体素尺寸换算。

2. **零模型用环面平移 + FFT 精确解，不用"仅图内合法位置"。**
   后者在结构散布全卷、包围盒≈整卷时**退化**：合法平移范围坍缩为 0，
   零分布与观测完全相同（p 恒为 1.0）。已在真实数据上遇到并用 48³ 合成场景复现。
   FFT 版本还有额外好处：一次算出全部 n³ 个平移，p 值**没有抽样误差**。

3. **输出 `degenerate_null` 标志。** 零分布若完全没有变化，说明该零模型对
   这个结构没有分辨力，此时的 p 值不可解释。自动标出来，避免误读。

### 与 2D 版的指标对应

| 2D | 3D |
|---|---|
| `contact_length_px` | `contact_faces`（共享面元数） |
| `contact_per_a_boundary` | `contact_per_a_surface` |
| `contact_tol_frac` | `contact_tol_per_a_surface` / `tol_profile_nm` |
| `a_components` | `a_components`（6-邻接连通） |

---

## 9. 真实数据的第一批结论（摘要）

完整分析见 [`docs/CONTACT-3D-RESULTS.md`](../../docs/CONTACT-3D-RESULTS.md)。

jrc_mus-kidney 的 5 个 crop（含 mito + er 标注）：

- **严格零距离接触在 4/5 个 crop 里为 0**；唯一非零的 crop129 是 8 nm 体素，
  很可能只是采样把 <16 nm 的间隙抹平了。
- **贴合出现在 32–64 nm 尺度**：64 nm 容差下 9%–31% 的线粒体表面触及内质网。
  这与 MAMs（线粒体相关膜）已知的 10–30 nm 间隙一致。
- **所有 5 个 crop 的接触都低于环面零模型期望**（z = −1.63 … −4.61），
  即线粒体与内质网在这几块肾组织里是**空间分离**的，而不是共定位。

样本量只有 5 个 crop / 1 个数据集 / 1 只小鼠，上述结论不能外推。

