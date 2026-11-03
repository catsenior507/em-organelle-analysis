# 体系说明：从原始图像到网络分析

> 本文档描述整条流水线的**结构、数据契约与已验证结论**，
> 目的是让任何人（包括几个月后的自己）能在不看代码的情况下跑通并读懂结果。

---

## 1. 一张图看懂

```
data/原图片/*.tif            88 张灰度 EM（1.004 / 2.667 / 5.291 nm/px）
data/分离的标注信息/green_*.tif  绿色描边 = 线粒体（红=比例尺，黄=文字）
        │
        │  ① tiles          tools/build_ourdata_tiles.py
        ▼
data/ourdata_tiles/         2657 个 256×256 tile，统一到 4 nm/px
        │                    按图像分组，含 pos/neg 与覆盖率标注
        │  ② train          tools/server/scripts/train_seg.py
        ▼                     或 finetune_mitonet.py（微调 MitoNet）
runs/seg_ourdata*/          U-Net 权重（2.29 M 参数）
        │
        │  ③ predict        tools/predict_all_ourdata.py
        ▼
data/masks_ourdata/         93 张原分辨率二值掩码 + index.csv
        │
        │  ④ filter         tools/filter_masks.py（物理面积 0.05 µm²）
        ▼
data/masks_ourdata_clean/   过滤后掩码，每图平均 6.3 个线粒体
        │
        │  ⑤ regions        tools/seg_to_regions.py
        ▼
data/annotations_seg/       与 annotations_v2 **同构**的 regions JSON
        │
        │  ⑥ network        tools/network/run_analysis.py --method radius
        ▼
data/network_analysis_seg_radius/   网络指标与报告
        │
        │  ⑦ report         tools/pipeline_report.py
        ▼
logs/PIPELINE_REPORT.md
```

统一入口：

```bash
python tools/run_pipeline.py --status        # 看每步是否已完成
python tools/run_pipeline.py --steps all     # 全流程（已完成的自动跳过）
python tools/run_pipeline.py --from predict --force   # 换模型后重跑后段
```

---

## 2. 统一层：`tools/unified.py` + `datasets.yaml`

### 3.1 它解决什么

我们手上同时存在**三种互不兼容的输入形态**：

| 形态 | 图像 | 标注 | 过去由谁处理 |
|---|---|---|---|
| `green_separated` | 调色板 TIFF（灰度存成 `mode=P`） | 独立绿色描边图 `green_<id>.tif` | `build_ourdata_tiles.py` |
| `green_overlay` | RGB JPEG（实为灰度） | 同名图副本上叠加绿色描边 | `build_golgi_tiles.py` |
| `prebuilt_tiles` | 已是 `<crop>/z####/{em.png,<类>.png}` | 同左 | `train_seg.py` |

结果是**每来一批数据就写一个新脚本**，而且各自处理"转灰度/填洞/划分"的细节
略有出入，产生不可比的结果。

现在收敛成**一份声明 + 一个读取器**：新增数据集只加一段 YAML，不写代码。

```bash
python tools/unified.py list              # 列出所有数据集与可用数量
python tools/unified.py probe all          # 只检查：配对/尺寸/GT覆盖率/跳过原因
python tools/unified.py build all          # 生成 tile
```

### 3.2 已实现的兼容性

| 维度 | 支持 |
|---|---|
| 图像格式 | 调色板 TIFF、RGB、灰度、8/16 位 → 统一 uint8 灰度 |
| 标注形态 | 独立绿掩码 / 叠加式绿描边 / 现成二值掩码 |
| 命名 | 同名配对、`green_<id>` 前缀配对 |
| 尺寸不一致 | 把标注缩放到图像尺寸并记录（**不再丢图**） |
| 尺度 | `scale_json` 按图查表 + `target_nm` 统一重采样 |
| 负样本策略 | `all_if_trusted`（应对部分标注）或 `sampled` |
| 划分 | `val_every`（确定性）或 `val_images`（外部名单） |

### 3.3 兼容性是可回归验证的

`tools/test_unified_compat.py` 把"与既有脚本一致"写成断言：

```
[2] 高尔基体：必须精确复现旧脚本的 tile 计数
  PASS  正 tile = 976      PASS  负 tile = 820      PASS  图像 = 51
[3] 线粒体：逐图与旧脚本比对（12_0002 为已知改进）
  PASS  没有凭空丢失的图像
  PASS  多出的图像只有 12_0002
  PASS  共有图像的 pos/neg 完全一致   [88 张]
  PASS  总 tile 数差异 = 仅 12_0002 的贡献   [旧 2657 新 2778 差 121]
```

**`12_0002` 是已知且经确认的改进，不是回归**：它的标注图是转置的
（2048×2115 对 2115×2048），旧脚本 `if o.shape != img.shape: continue`
把这张有标注的图整张丢弃，统一层改为缩放处理，因此多出 1 张图 / 121 个 tile。

### 3.4 一个真实踩到的坑

统一层最初在 `--out` 给定时输出 `<out>/<name>`、未给时输出 `data/<name>_tiles`。
两套约定导致流水线传了 `--out` 后产物落到 `data/ourdata_mito`，
而状态检查找 `data/ourdata_mito_tiles`，于是**永远显示"待运行"**。
现已统一为 `<out>/<name>_tiles`（`--out` 只作父目录）。

### 3.5 多数据集并存的流水线

```bash
python tools/run_pipeline.py --dataset ourdata_mito  --status
python tools/run_pipeline.py --dataset ourdata_golgi --status
python tools/run_pipeline.py --dataset ourdata_golgi --from predict --force
```

产物路径随数据集派生，两类数据互不覆盖：

| 步骤 | `ourdata_mito` | `ourdata_golgi` |
|---|---|---|
| tiles | `data/ourdata_mito_tiles/` | `data/ourdata_golgi_tiles/` |
| masks | `data/masks_ourdata_mito*` | `data/masks_ourdata_golgi*` |
| regions | `data/annotations_ourdata_mito/` | `data/annotations_ourdata_golgi/` |
| network | `data/network_ourdata_mito/` | `data/network_ourdata_golgi/` |

> 旧的 `data/ourdata_tiles/`、`data/annotations_seg/`、`data/golgi_tiles/`
> 保留作历史对照，用于复现此前发布的 Dice 数字。

---

## 3. 数据契约（每一步的输入/输出格式）

| 环节 | 输入 | 输出 | 关键约定 |
|---|---|---|---|
| ① tiles | `原图片/*.tif`（调色板灰度）、`green_*.tif`（描边） | `<out>/ourdata/<图像>/z####/{em.png,mito.png,meta.json}` | `<图像>` 必须是 crop 级目录，否则按 crop 划分失效 |
| ② train | tile 目录 | `runs/<name>/{best,last}.pt`、`history.json` | `--val-images` 固定验证集；换种子只变训练侧 |
| ③ predict | 权重 + 原图 | `<out>/<图像>.png` + `index.csv` | 掩码为**原分辨率**；`index.csv` 的 `split` 列区分训练/留出 |
| ④ filter | 掩码 + `index.csv` | 过滤后掩码 + 新 `index.csv` | 阈值用**物理面积**，不用像素数 |
| ⑤ regions | 过滤后掩码 | `regions/<图像>.json` + `dataset.json` | 字段与 `annotations_v2` 同构，特征复用 `tools/morpho/features.py` |
| ⑥ network | regions 目录 | `summary.json` 等 | **必须 `--method radius`** |
| ⑦ report | 各处产物 | `logs/PIPELINE_REPORT.md` | 只读取，不重算 |

---

## 4. 已验证的结论（以及每个数字的来源）

| 结论 | 数值 | 来源文件 |
|---|---|---|
| CellMap 单模态模型零样本失败 | Dice 0.1516（0/6） | `logs/fullimage_ourdata.json`、`docs/EMPANADA-TRANSFER-RESULTS.md` |
| **MitoNet 通用模型零样本可用** | Dice **0.7518**（17/18） | `logs/EMPANADA_HEAD2HEAD.txt` |
| **自有 U-Net 从零训练** | Dice **0.7839**（18/18） | `logs/fullimage_ourdata.json` |
| 剔除标注覆盖 <3% 的 1 张 | 0.8125（17 张） | `docs/OURDATA-RESULTS.md` §7 |
| 多种子稳定性 | **0.7890 ± 0.0016** | `docs/OURDATA-RESULTS.md` §8 |
| 形态学：尺寸可用、边界不可用 | 面积 −10.5%，solidity 0.996→0.882 | `logs/MORPHO_COMPARISON.txt` |
| 网络：必须用距离邻近判据 | 硬接触 140→4；radius 491→458 | `docs/OURDATA-RESULTS.md` §6 |
| **高尔基体分割（自有标注）** | Dice **0.4849**，平凡基线 0.2027 | `logs/GOLGI_RESULT.txt` |
| 统一层与旧脚本产出相符 | 高尔基体 976/820 精确复现；线粒体仅多 12_0002 | `tools/test_unified_compat.py` |

**报精度时只能引用上表来源的数字。** 尤其：
**发布版模型（全 88 张训练）没有留出集，不能给它标注 Dice。**

---

## 5. 踩过的坑（都已在代码里修掉，但值得记住）

这些是实际耗费过时间的错误，留在这里避免重复：

| 坑 | 表现 | 修法 |
|---|---|---|
| `ls <dir> >/dev/null` 返回 255 | 目录存在却被判为缺失 | 用 `test -d` |
| `Compress-Archive` 的 zip 用反斜杠 | Linux `unzip` 解出空目录 | 改用 `tar` |
| 模型尺度不匹配 | 1 nm/px 的图 MitoNet **零输出** | 推理前重采样到 8 nm/px |
| 硬接触判据 | 预测网络边数 −97% | 改用 `--method radius` |
| 部分标注 | Dice 被压低，低分图其实是标注不全 | 报告分层与基线，不单看均值 |
| 验证集随种子变化 | 种子间差异被混淆 | `--val-images` 固定 |
| `python ... \| tail -N` | 日志整个训练期不增长，无法监控 | 去掉管道，实时输出 |
| 从零训练 vs 微调 | 目标域从零训练优于源域微调 | 见 `docs/EMPANADA-TRANSFER-RESULTS.md` |

---

## 6. 环境

两个环境，用途不同：

**本机 `.venv`（Python 3.11）** —— 推理、掩码导出、形态学、网络分析
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install empanada-napari==1.2.4 --no-deps
pip install pyyaml requests zarr scikit-image numba opencv-python-headless tqdm cztile imagecodecs
```

> `empanada-napari` 的 `__init__.py` 会导入 Qt 组件，无 GUI 环境下会失败。
> 已替换为最小版本（原文件备份为 `__init__.py.orig`）。

**服务器 `dachuang`（Python 3.10，2× RTX 3080）** —— 训练与微调
- `empanada-napari==1.2.4`（wheel 自带 empanada 核心包）
- `numba` 必须能装到预编译 wheel；若尝试源码编译 `llvmlite` 会失败，
  用 `--only-binary=:all:` 强制
- 权重从 zenodo 下载（服务器可达），落在 `~/.empanada/`

---

## 7. 目录约定

| 路径 | 内容 | 是否可再生 |
|---|---|---|
| `data/原图片/`、`data/分离的标注信息/` | 原始数据 | **不可再生，勿删** |
| `data/ourdata_tiles/` | tile 数据集 | 可再生（①） |
| `data/masks_*/` | 预测掩码 | 可再生（③④） |
| `data/annotations_seg/` | regions JSON | 可再生（⑤） |
| `runs/*/best.pt`、`last.pt` | 模型权重 | 可再生（②），训练耗时 |
| `runs/*/epoch*.pt` | 中间快照 | 可删（已清理过一次，省 4.86 GB） |
| `logs/*.txt`、`docs/*.md` | **结论与证据** | **不可再生，勿删** |

---

## 8. 如何扩展

**加一个新细胞器**：
1. 确认它在图中确实可见（不要重犯"推荐脂滴但图里没有"的错）
2. 标 20–30 个实例，做成 `<root>/<图像>/z####/{em.png,<类别>.png}`
3. `train_seg.py --classes mito,<类别>`（两类一起训练时**必须**加
   `--class-balance`，否则少数类会被压制到低于平凡基线）
4. 后段不变：④→⑦

**加一个新指标**：在 `tools/pipeline_report.py` 里读取对应产物即可，
不要在报告里重算——报告与产物数字必须同源。

**换模型**：只要产出 `data/masks_<name>/`（原分辨率二值掩码）+ `index.csv`，
执行 `python tools/run_pipeline.py --from filter --force` 即可，后段全部适用。

---

## 9. 高尔基体与外部数据的工具链

线粒体之外，体系已经扩展到**第二个细胞器**和**第二个数据源**。
以下工具不在主流水线的八步里，但属于同一体系，按使用顺序列出：

| 工具 | 作用 |
|---|---|
| `tools/extract_golgi.py` | 解压用户提供的高尔基体压缩包（处理 GBK 文件名与 `__MACOSX`） |
| `tools/quantify_text_contamination.py` | 量化插图里烘死的说明文字对 GT 的污染（结论：约 2%，不值得屏蔽） |
| `tools/detect_text_artifacts.py` | 五种文字检测判据的实现（全部失败，作为反面记录保留） |
| `tools/golgi_scale_analysis.py` | 用结构等效直径判断尺度一致性（中间 50% 仅差 1.8 倍） |
| `tools/clean_golgi_labels.py` | 去掉 <500 px 的标注碎片 |
| `tools/asem_inspect.py` | 查 ASEM 各体数据卷的对齐偏移（offset 相减除以 resolution） |
| `tools/asem_fetch.py` | 按 chunk 从公开 S3 拉取所需体数据子区域 |
| `tools/asem_to_tiles.py` | 3D 体数据 -> 2D 训练切片（含小于 tile 时的 padding） |
| `tools/predict_golgi.py` | 全图推理 + **容差 Dice** + 叠加图 |
| `tools/eval_crossdomain.py` | 跨数据集评估（答案：域间几乎不迁移） |
| `tools/golgi_dice_vs_coverage.py` | 按标注覆盖度分层统计 Dice（r = +0.73） |
| `tools/cv_partial.py` | 交叉验证的部分结果（只看已跑完的折） |
| `tools/golgi_report.py` | 从产物自动生成 `logs/GOLGI_STATUS.md` |

**服务器端入口**（都在 `tools/server/`）：
`upload.py`（上传，处理 `~` 与相对路径）、`download.py`（SFTP 取文件，
不经过 shell 以免被 BOM 污染）、`server_exec.py`（执行远端命令）、
`push_run.py`（上传+执行）。

> 这三个小工具是被同一类问题逼出来的：用 `python -c "..."` 内联调远端命令，
> 反复被 PowerShell 的变量展开、引号吞并、CRLF 转换、BOM 污染打断。
> 凡是"要传给 shell 的文本"，都应该写成文件而不是内联字符串。

完整的实验记录与结论见 `docs/GOLGI-IMPROVEMENTS.md` 与 `docs/MULTICLASS-CONTACTS.md`。

### 9.1 高尔基体也能走主流水线

```bash
# 建 tile（用统一层，按 datasets.yaml 的声明）
python tools/run_pipeline.py --dataset ourdata_golgi --only tiles

# 推理（自动切到 tools/predict_golgi.py，不是线粒体那条路径）
python tools/run_pipeline.py --dataset ourdata_golgi --from predict
```

**推理步骤按数据集分派**，这是必须的而不是偷懒：

| | 线粒体 | 高尔基体 |
|---|---|---|
| 尺度 | 有可靠 nm/px（`dataset.json`），统一重采样到 4 nm | **无标定**，必须在原分辨率推理 |
| 权重 | `models/ourdata_release/best.pt` | `runs/golgi_v3/best.pt` 等 |

早期 `step_predict` 硬编码线粒体脚本，于是 `--dataset ourdata_golgi` 也能"跑通"，
但用的是线粒体的重采样逻辑——**一个不会报错的错误**。现在按数据集分派，
且权重也会自动派生（避免拿线粒体模型去推理高尔基体）。

> 注意：高尔基体的训练在服务器上完成，本地要跑推理需先
> `python tools/server/download.py "~/dachuang_organelle/runs/golgi_v3/best.pt" "runs/golgi_v3/best.pt"`。
> 全量 53 张在 CPU 上较慢，建议在服务器 GPU 上跑 `tools/predict_golgi.py`。


