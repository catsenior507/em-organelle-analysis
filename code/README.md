# code/ —— 原 MATLAB 流水线（保留但已不是主路径）

> **主路径已迁移到 `tools/`（Python）。** 这里保留 MATLAB 代码的原因：
> 一是历史进度记录引用了它们，二是部分脚本（比例尺标定、颜色分离）
> 仍可作为参考实现。
>
> 完整问题清单见 [`../docs/AUDIT-REPORT.md`](../docs/AUDIT-REPORT.md)。

---

## 目录定位

| 目录 | 状态 |
|---|---|
| `tools/morpho/` | ✅ **唯一正确的区域提取与特征计算**（替代本目录的 `analyze_annotated_regions.m` 等） |
| `tools/detr/` | ✅ Transformer / DETR 阶段（替代 `detr_region_feature.py`，后者不在本目录） |
| `tools/network/` | ✅ 互作网络重建与涌现指标（原项目**从未实现**的部分） |
| `code/`（本目录） | ⚠️ 历史实现，已修关键 bug，但**不建议**再作为主路径 |

---

## 本目录文件的处置

### 保留并已修补

| 文件 | 说明 |
|---|---|
| `project_paths.m` | 🆕 按文件位置推算绝对路径，取代所有写死的 `D:\dachuang` 与 pwd 相对路径 |
| `project_assert_cwd.m` | 🆕 工作目录不符时**显式报错**，而不是静默读错目录 |
| `find_scale_bar.m` | 已修：移除 OCR 失败后静默代入 `0.1 µm/px` 的兜底；修 `strrep(filename,'red','')` 删除所有 `red` 子串的问题 |
| `normalize_features_um.m` | 已修：改为**全数据集**统计量归一化（原为每图独立 min-max，跨样本不可比），并在 CSV 中保留 µm 原值 |
| `analyze_emergent_entropy.m` | 已修：路径改为 `project_paths()`（原写死 `d:\dachuang`，脚本**完全无法运行**） |

### 保留但仅作参考（顶部已加弃用告警）

| 文件 | 弃用原因 |
|---|---|
| `analyze_annotated_regions.m` | 分割取的是**背景**（`~green_mask`）；输出 JSON 含字面量 `...` 而非法；坐标 `[row,col]` 被当 `(x,y)` |
| `simple_region_features.m` | 解析的是上述非法 JSON；`complexity` 退化（恒为 `1/max_y`）；`min_axis`/`minor_axis` 字段名不一致曾导致崩溃 |
| `extract_region_features.m` | **从未成功运行过**，输出是 `{"regions": []}` |
| `separate_annotation_colors.m` | 三通道精确相等导致抗锯齿像素丢失（实测绿色/黄色各丢约 **31%**）；`im2uint8` 在颜色循环内部执行 |
| `representation_learning.m` | 数据泄漏（随机划分同图增强副本）；任务病态（同图所有区域共享同一行输入）；指标无意义 |
| `neural_network_training.m` / `deep_learning_training.m` | 同上；报告 MSE 1.1848 而"预测均值"基线恰好是 1.0；另有 4 次运行是 NaN |
| `run_from_feature_extraction.m` | 上述逻辑的又一份拷贝，且因 `axes.min_axis` 崩溃过 |
| `test_learning.m` / `test_simple.m` | 只处理前 10 个文件（按字母序全是同一张图的增强副本） |
| `check_deep_learning.m` | 诊断脚本，功能正常，保留 |

### 已移入 `../archive/`（跑不通或已废弃）

| 文件 | 原因 |
|---|---|
| `continue_learning.m` | 调用 `representation_learning.m` 的**局部函数**，MATLAB 不可见 → `continue_learning.log:8-11` 报"未定义函数"，**永远跑不通** |
| `test_json_fix.m` | 同样调用局部函数 `parse_json_regions`；且它想测的正则兜底路径根本没被执行 |
| `run_test.m` / `run_test.bat` | 两者运行的是**不同**的脚本；`run_test.bat:2` 的 `cd /d "d:\dachuang\code"` 指向不存在的目录且不检查 errorlevel；`-nodisplay` 是 UNIX 选项；产物 `test_output.txt` 是 **0 字节** |
| `test_deep_learning.m` / `test_deep_learning_model.m` | 用 `rand` 随机目标训练，MSE 与"预测常数"基线无差别，却打印"工作正常" |
| `debug.log` | 输入法噪音，与项目无关 |
| `ocr_scale_marks.py` | 整个文件是一行、含字面量 `\n` 转义 → Python SyntaxError，100% 死代码 |

> 以上全部可用 `git show aeef301:<原路径>` 取回。

---

## 数据源提示

**不要再从 `data/features_json`、`data/features_all`、`data/normalized_features`
读取特征。** 它们分别存在"91/94 个文件非法 JSON""像素被截断到 1000 点"
"每图独立 min-max 导致跨样本不可比"的问题。

唯一正确的数据源是：

```
data/annotations_v2/dataset.json          清单
data/annotations_v2/regions/<image>.json  区域 + 特征 + µm 派生量
```

生成方式：

```bash
python tools/morpho/extract_labels.py --jobs 6 --out data/labels.json
python tools/morpho/extract_regions.py --out data/annotations_v2 --jobs 8
```

`project_paths.m` 已提供对应字段：`P.annotations_v2`、`P.regions_v2`、`P.dataset_json`。
