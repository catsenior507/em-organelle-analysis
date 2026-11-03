# EM Organelle Analysis ｜ 电镜细胞器分析

[English](README.md) ｜ **中文**

从电镜切片分割细胞器，再由此重建**细胞器之间的互作**——后者才是这个项目真正想做的事。

二维电镜切片先分割成细胞器掩码，掩码转成实例区域，区域构成接触图。
下面每一个数字都能从 `logs/` 里的产物复现，出处就写在数字旁边。

---

## 仓库内容

| 方向 | 代码 | 结果 |
|---|---|---|
| 线粒体分割（自训 U-Net） | `src/train_seg.py` | 18 张留出图 **Dice 0.7839** |
| 通用电镜模型迁移（MitoNet） | `src/finetune_mitonet.py` | 零样本 0.7518 → 微调后 **0.7847** |
| 高尔基体分割 | `src/train_seg.py`、`tools/predict_golgi.py` | 初始 0.4849，见下文「已知限制」 |
| **线粒体–内质网接触分析** | `tools/network/multiclass_contacts.py` | 线粒体表面 **4.60%** 在 25 nm 内接触内质网 |
| 同类细胞器邻接网络 | `tools/network/run_analysis.py` | 标注 491 条边 vs 预测 458 条 |
| 形态学指标迁移 | `tools/compare_morpho.py` | 面积 −10.5%，长短轴 −2…−5% |
| 混沌 / 熵分析（MATLAB） | `code/*.m` | 见 `docs/` |
| DETR 区域检测 | `tools/detr/` | 探索性，见 `docs/DETR-TRAINING-READINESS.md` |

**负面结果一并保留，没有粉饰**：

* **跨域不迁移。** 在我们自己的 TEM 插图上训练的模型，拿到公开 FIB-SEM 体数据上只有
  0.0779 和 0.2242，而域内是 0.2636。细胞器的外观高度依赖成像条件。
  （`docs/GOLGI-IMPROVEMENTS.md` §11）
* **加外部数据没能提升本域精度。** 随着实验设计逐步收紧，效应估计一路从
  +0.027 → +0.025 → +0.009 → −0.014。（`docs/GOLGI-IMPROVEMENTS.md` §5）
* **内质网这一类放弃了。** 域内与跨器官的检测、分割都失败；灰度 AUC 0.365，
  与它在数据里的占比基本持平，即等同于随机。（`docs/OTHER-ORGANELLES-FEASIBILITY.md`）

---

## 快速开始

```bash
pip install -r requirements.txt

# 查看已声明的数据集，然后按声明建切片
python tools/unified.py list
python tools/unified.py probe ourdata_mito
python tools/unified.py build ourdata_mito

# 训练
python src/train_seg.py --train-root data/ourdata_mito_tiles \
    --val-root split --classes mito --size 256 --epochs 50 --out runs/seg_mito

# 八步全流程
python tools/run_pipeline.py --status
python tools/run_pipeline.py --steps all
```

流水线是 `env → tiles → train → predict → filter → regions → network → report`。
产物已存在就自动跳过，`--force` 重跑，任一步失败即中止——
不会带着半成品往下走。每一步的输入输出契约见 `docs/PIPELINE.md`。

跑单测：

```bash
python src/test_metrics.py      # 指标实现对照已知算例
python src/test_data_paths.py   # 数据集路径解析
python src/test_ignore_mask.py  # 被忽略的像素梯度必须精确为 0
```

### 新增数据集

数据集是**声明式的，不用写代码**（`datasets.yaml`）。读取器已经能处理
调色板 TIFF、RGB、灰度、16 位输入；三种标注形态（独立绿掩码、叠加描边、二值掩码）；
标注与图像尺寸不一致；逐图像素尺寸；以及两种负样本采样策略：

```yaml
ourdata_mito:
  class: mito
  images: data/原图片
  annotations: data/分离的标注信息
  annotation_kind: green_separated
  annotation_pattern: "green_{id}.tif"
  scale_json: data/annotations_v2/dataset.json
  target_nm: 4.0
```

---

## 参考数字，以及它们不能和什么比

只引用能从 `logs/` 复现的数字。下面四种比较是错的，列出来是因为很容易不小心做出：

| 不要这样比 | 原因 |
|---|---|
| 0.4849 与 0.3879 / 0.3605 | 验证划分不同——317 个切片（随机 20%）对 281 个切片（每 5 张留 1 张）；训练图 41 张对 38 张 |
| 切片级 Dice 与全图像素级 Dice | 不是同一个量，切片级大约高 0.06 |
| `best-of-epochs` 与 `last-5-epoch` 均值 | 系统性相差 0.03–0.05。`best` 还是在 60 个 epoch 里挑最大值，自带选择偏差，跨种子也更不稳 |
| 公开 FIB-SEM 上的接触统计与我们自己样品 | 细胞不同（BSC-1），制样不同（化学固定） |

前三项有脚本可以机械核验：
`python tools/verify_report_numbers.py` 与 `python tools/verify_golgi_numbers.py`。

---

## 已知限制：标注覆盖度

这是高尔基体分数偏低的诚实解释，也是最能改善这个项目的一件事。

真值是不完整的。48 张高尔基体图里有 17 张，标注只覆盖不到 3% 的视野，
而模型预测出来的面积是标注的十倍左右——模型找到了真实存在、但没人勾画的细胞器，
这些正确检测全被算作假阳性。Dice 几乎随标注覆盖度单调变化：

| 标注覆盖度 | 图数 | Dice | 预测 / 真值 |
|---|---|---|---|
| < 3% | 17 | **0.0900** | **10.8×** |
| 3–8% | 18 | 0.4370 | 1.77× |
| 8–20% | 15 | **0.6046** | 0.91× |
| ≥ 20% | 3 | **0.7139** | — |

覆盖度与 Dice 的相关系数 **r = +0.730**。在标注充分的图上，同一个模型能到 0.60–0.71，
而且预测轮廓就贴在池层堆叠上。0.30 这个总体数字是被那批标注稀疏的图拉下来的。

线粒体呈现完全相同的形态（同样分层下 0.2969 / 0.7837 / 0.7987 / 0.8504），
这是在不同细胞器、不同数据集上**独立复现**的，所以把它当作数据属性而非建模缺陷。

**后续工作。** 补齐低覆盖度图像的标注。按分层数字推算，把那 17 张补到 8%，
整体均值就能超过 0.5，而且**完全不用动模型**。
模型侧的手段已经全部试过并失败——学习率与训练计划、Dice 权重、外部数据、
标注碎片清洗、剔除稀疏标注图（结果反而更差）、边界容差。

「忽略区」机制已经实现并单测通过（`src/test_ignore_mask.py`），
供标注者记录"这块我没标"之后排除损失：在切片旁放一个 `ignore.png`，
那些像素的梯度就精确为 0。目前尚未在任何地方启用，
因为**用模型自己的预测去定义该忽略哪里，只会固化它当前的错误**。

---

## 关于数字的可追溯性

`docs/` 里的每个数字都是从产物重算的，不是手抄的。
`tools/golgi_report.py` 生成 `logs/GOLGI_STATUS.md` 时只读日志、不重算任何指标，
并且会把旧日志里已被推翻的结论**显式标注出来**，而不是去改历史记录。
`tools/verify_golgi_numbers.py` 把高尔基体文档里 90 个小数逐一与日志比对——
它抓出过一次真实的抄写错误：某次运行的**训练中途值**被当成了最终结果。

---

## 目录结构

```
src/          训练器、数据集读取、单测
tools/        流水线、统一数据层、网络分析、DETR、形态学
code/         MATLAB 混沌/熵分析、比例尺 OCR、特征提取
docs/         分主题的结果文档（英文为主，中文见 *.zh-CN.md）
logs/         文档引用的结果产物
datasets.yaml 数据集声明
```

## 数据

原始电镜图像、标注、模型权重与训练记录**不在**本仓库——
体积大，且部分是第三方数据。代码预期它们在 `data/`、`models/`、`runs/` 下；
`datasets.yaml` 描述目录约定，`docs/PIPELINE.md` §7 有详细说明。

用到的公开数据：[CellMap / OpenOrganelle](https://openorganelle.janelia.org/)
与 [ASEM](https://registry.opendata.aws/asem-project/)（Kirchhausen 实验室，CC BY-SA 4.0）。
拉取了什么、许可条款如何，记在 `docs/GOLGI-DATA-SOURCES.md`。

## 许可

代码为 MIT（见 `LICENSE`）。第三方数据各自的许可不受此覆盖；
ASEM 数据是 CC BY-SA 4.0，再分发必须遵守。
