# tools/morpho —— 区域提取与特征计算

> 替代 `code/analyze_annotated_regions.m`、`code/simple_region_features.m`、
> `code/extract_region_features.m`、`code/normalize_features_um.m`、
> `code/find_scale_bar.m` 与 `code/ocr*.py`。
> 原实现的问题清单见 `docs/AUDIT-REPORT.md` §2.1。

---

## 1. 修正了什么

| 方面 | 原实现 | 本实现 |
|---|---|---|
| **区域定义** | `~green_mask`（绿色轮廓的**补集**）→ 提取到的是背景 | `binary_fill_holes(green_mask)` → 轮廓**内部**，即细胞器 |
| **像素存储** | `min(length(...), 1000)` 截断，并写入字面量 `...` → 91/94 个文件不是合法 JSON | 简化轮廓多边形（`cv2.approxPolyDP`，13–95 点），**无损保留形状** |
| **坐标语义** | MATLAB `find` 给出 `[row,col]`，下游按 `(x,y)` 读 | 全流程统一 **1-based `(x, y)` = (列, 行)** |
| **比例尺** | OCR 失败时静默代入 `0.1 µm/px`（比正确值大 20–100 倍） | `ok=False` + 显式告警，**绝不代入默认值** |
| **灰度特征** | 因 xy 互换而采样到转置位置 | 按 `(x,y)` 正确采样 |
| **质量过滤** | 无（4 百万像素的背景块进入训练集） | `quality_flag` 标记 + `max_area_ratio` 上限 |
| **标签** | OCR 未做极性反转，结果全是 `M`/`S`/`co`/空 | 两种极性取高置信度 + 归一化 → 稳定得到 `M` |
| **重复导出** | 91 张图被当成 94 个样本（`_RGB`/`_Channel4` 重复计数） | `list_pairs()` 过滤非规范名，89 张图 |

### 实测效果（`11_0002`）

| | 原实现 | 本实现 |
|---|---|---|
| 提取到的"区域"数 | 248 个连通域，最大一个是 **4 009 029 px**（整图 92.6% 的背景） | **22 个细胞器**，最大 52 862 px |
| 区域面积中位数 | 23 800 px（被背景块主导） | 10 285 px |
| 占据图像面积 | ~93%（几乎整图） | **7.7%**（符合线粒体占比的常识） |

---

## 2. 文件

| 文件 | 作用 |
|---|---|
| `features.py` | 特征的单一定义源。`segment_from_outline()` 做填洞分割；`compute_region_features()` 算形态 + 灰度；`RegionFeatures` / `ScaleCalibration` 数据结构 |
| `extract_regions.py` | CLI：并行提取、比例尺校验、显式处理尺寸不一致、JSON 输出 |
| `extract_labels.py` | CLI：对黄色标注做 OCR，产出 `data/labels.json` |

---

## 3. 用法

```bash
# 1) OCR 标签（可选，但推荐先跑，之后提取就能自动带上类型）
python tools/morpho/extract_labels.py --jobs 6 --out data/labels.json

# 2) 区域与特征提取（约 1 分钟，8 进程）
python tools/morpho/extract_regions.py --out data/annotations_v2 --jobs 8
```

`extract_regions.py` 会**自动**在输出目录的上一级查找 `labels.json`；
找到就按距离把标签关联到区域，找不到才需要加 `--ocr` 在本进程内即时识别。

### 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--min-area-px` | 50 | 最小区域面积，过滤碎片 |
| `--max-area-ratio` | 0.25 | 单区域面积上限（占整图比例），过滤"把背景围起来"的巨型区域 |
| `--on-size-mismatch` | `transpose` | 原图与标注尺寸不一致时的处理。**项目里 `12_0002.tif` 确实是转置的**（原图 2048×2115，标注 2115×2048）。可选 `error` / `transpose` / `skip-gray` |
| `--jobs` | CPU-1 | 并行进程数 |

---

## 4. 输出格式

```
data/annotations_v2/
├── dataset.json          清单：config、每图区域数、失败列表、特征 schema
├── scale_report.json     比例尺标定汇总（便于人工核对）
└── regions/<image>.json  单张图的区域列表
```

`regions/<image>.json` 的结构：

```jsonc
{
  "image": "11_0002",
  "image_shape": [2115, 2048],          // [H, W]
  "original_file": "11_0002.tif",
  "scale": { "um_per_px": 0.005291, "ok": true, "source": "mat", ... },
  "geometry_note": "",                   // 尺寸不一致时的说明
  "num_regions": 22,
  "num_labeled": 21,
  "regions": [
    {
      "region_id": 1,
      // ---- 位置（1-based，单位：像素）----
      "min_x": 811, "max_x": 948, "min_y": 257, "max_y": 393,
      "width": 138, "height": 137,
      "center_x": 879.2, "center_y": 325.1,
      // ---- 形状 ----
      "area": 12985, "perimeter": 412.7,
      "major_axis": 79.66, "minor_axis": 52.89,
      "orientation": -0.42,
      "solidity": 0.986, "eccentricity": 0.748,
      "extent": 0.687, "equivalent_diameter": 125.9,
      // ---- 灰度 ----
      "gray_mean": 112.5, "gray_std": 18.3,
      "gray_min": 62, "gray_max": 191, "cv_gray": 0.163,
      // ---- 旧口径（保留以便与历史结果对照，不建议继续使用）----
      "roughness": 3.62, "complexity": 0.0071,
      // ---- 标签 ----
      "label_pixels": 36, "label_ratio": 0.0028,
      "recognized_text": "M", "organelle_type": "M",
      "ocr_raw": "M", "label_distance_px": 0.0,
      "is_labeled": true,                 // 语义：成功关联到一个标签
      "has_label_pixels": true,           // 区域内是否含黄色像素
      "quality_flag": "ok",
      // ---- 轮廓（1-based [x, y]，可用于重建掩码或可视化）----
      "contour": [[811.0, 300.0], [820.0, 281.0], ...],
      // ---- µm 派生量（标定不可信时为 null）----
      "um": { "per_px": 0.005291, "width": 0.730, "area": 0.364, ... }
    }
  ]
}
```

### 关于 `is_labeled`

原实现的语义是"区域内含黄色像素"，但黄色像素可能只是**邻近标签溢出的边缘**。
实测：562 个区域含黄色像素，其中 543 个能可靠关联到标签。
因此这里把 `is_labeled` 收紧为"**成功关联到一个标签**"，
并保留 `has_label_pixels` 供需要旧语义时使用。

---

## 5. 当前数据集状态

```
89 张图 / 570 个区域 / 0 失败
区域面积：最小 1 458 px，中位 34 209 px，最大 1 018 994 px
标签关联：543 个区域带类型（全部为 "M"），27 个无标签
比例尺：88 张可信，1 张不可信（12_0002）
```

⚠️ **只有一个细胞器类型（M）**。黄色标注在全部 89 张图里都是同一个字形 `M`。
因此：
- DETR 的 `num_classes` 只能设为 1，只能做**检测**，做不了类型分类；
- `tools/network/` 里任何"按类型分层"的指标都是退化的。

若实际数据中确实存在其它细胞器（内质网、溶酶体等），需要检查绘图标注——
很可能它们用了别的颜色，而 `code/separate_annotation_colors.m:31-33` 只定义了三种颜色。

---

## 6. 已知局限

1. **绿色轮廓是手绘的**，因此区域的边界精度取决于标注者的笔触。
   分割正确性依赖"轮廓闭合"，`segment_from_outline(close_iterations=2)`
   用 2 次形态学闭运算弥合小缺口；若某张图轮廓严重断开，会漏掉该细胞器。
2. **颜色分离丢了约 31% 的像素**（`code/separate_annotation_colors.m:85-87`
   用精确颜色相等，抗锯齿边缘被丢弃）。实测见
   `tools/data_integrity/check_color_separation.py`。
   这会影响 `perimeter` / `roughness`（由边界像素导出）以及填洞的闭合判定。
   若要更精确，应改用容差阈值重新分离（该脚本里有实现）。
3. **比例尺标定依赖 OCR 识别的数字**，一次误读就会让整张图的比例尺错一个数量级。
   已把不可信的标定显式标出（`ok=False`），但**可信的标定也建议人工抽查几张**。
4. **`12_0002` 的原图与标注互为转置**，默认 `--on-size-mismatch transpose`
   会转置原图来对齐。这修好了灰度特征，但说明该图的导出流程与其它图不同，
   建议回溯原因。
5. **OCR 归一化有硬编码**：`normalize_label_text()` 里列出了实测到的变体
   （`M` / `Ml` / `hl` / `Il` / `tl` …）。如果将来出现别的细胞器类型，
   这个映射需要相应扩展，否则新类型会被漏掉。原始 OCR 结果保留在
   `ocr_raw` 字段，便于审计。
