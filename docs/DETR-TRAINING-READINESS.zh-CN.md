# DETR 训练就绪报告

[English](DETR-TRAINING-READINESS.md) ｜ **中文**

> 服务器：`youruser@your-server-host:22`，环境 `dachuang`
> 日期：2026-10-31
> 结论：**训练链路已用真实数据端到端验证通过**，只差"更多数据"这一步。

---

## 1. 已验证的事实（都是实测，不是推断）

### 1.1 环境

| 项 | 值 |
|---|---|
| GPU | 2× RTX 3080，各 10 018 MiB，compute capability 8.6 |
| torch | **2.6.0+cu118**，`cuda_avail=True`，GPU matmul 通过 |
| 驱动 | 455.38（原生 CUDA 11.1，但 CUDA 11.8 运行时可用） |
| transformers | 5.20.2 + timm（5.x 的 DETR 默认主干需要 timm） |
| DETR 权重 | `facebook/detr-resnet-50`，经 `HF_ENDPOINT=https://hf-mirror.com` 下载（333 MB / 37 s） |

### 1.2 数据

| 项 | 值 |
|---|---|
| 标签来源 | S3 按 chunk 直接取（**未下载 42 GB zip**） |
| 已下载标签体 | 6 个 crop × mito/er = 12 个体积（300³ / 200³） |
| `jrc_mus-kidney` 里同时含 mito+er 的 crop | **17–20 个**（不同统计口径；按 manifest 为 17，按 S3 目录扫描为 20） |
| EM 平面 | 从 `em/fibsem-uint8/s0`（8 nm）按需取，**已与标签对齐** |
| 已导出 2D 切片 | 13 个（crop129 五层 300×300、crop140 五层 100×100 等），202 个实例 |

### 1.3 关键对齐规则（已验证）

```
EM 原点(体素) = translation(nm) / 8
EM 范围(体素) = shape * voxel_size / 8
标签降采样系数 = 8 / crop_voxel    （4 nm 标签 → 2× ；8 nm 标签 → 1×）
```

实测 `jrc_mus-kidney` 的 17 个 mito+er crop **全部落在 EM 范围内**；
crop129 取出的 EM 平面 100% 非零（致密组织），说明映射正确。

### 1.4 训练冒烟测试（真实切片，非合成）

```
13 个切片，202 个实例（mito 48、er 154）
模型 41.5 M 参数，分类头由 COCO 91 类重建为 3（2 类 + no-object）
batch=1, size=480 : 峰值显存  831–1123 MiB
batch=2, size=480 : 峰值显存      1123 MiB  → 按 85% 显存可放大 7–9 倍
loss 5.58 → 3.69（4 步，未调参，仅证明可下降）
```

**结论：10 GB 显存足以做 DETR 微调，batch 可开到 8–14（480px）或 4–6（640px）。**

---

## 2. 还缺什么：数据量

13 个切片**远远不够**训练。DETR 从 COCO 权重微调到新域通常需要
**数千张**图像、**200 epoch** 量级。当前只有 13 张，等于零。

### 可行的扩张路径（按代价排序）

| 步骤 | 切片数 | 预计数据获取时间 |
|---|---|---|
| `jrc_mus-kidney` 全部 17–20 个 crop × 每 crop 30 层 | ~500–600 | 约 10–20 分钟 |
| 再扩到别的数据集（`jrc_mus-liver` 23、`jrc_hela-2` 26 等） | ~2,000–7,000 | 数小时（S3 慢） |

数据获取为什么便宜：一个 100×100 的 4 nm crop 在 8 nm EM 上只覆盖
**1 个 chunk**（128×128），每 crop 只需 2–4 个 chunk。真正贵的是 300³ 的 crop
（需 45 个 chunk）。

---

## 3. 方法论注意事项（会影响结论，必须写进论文）

1. **标签分辨率与 EM 不一致**：17 个 mito+er crop 里 **16 个是 4 nm 标签**，
   而唯一可用的 EM 是 **8 nm**。因此标签必须 2× 降采样才能逐像素对齐。
   降采样用"块内取最大值"（任一前景即为前景），对 ER 细管网比取均值安全——
   取均值会系统性丢掉细结构。

2. **降采样会合并实例**：mito 的标签是**实例 ID**（0–5），2× 降采样后相邻实例
   可能粘连。当前做法是先按类别取二值掩码再降采样，因此训练目标是
   "每类的前景区域"，实例边界会损失。若要用实例级目标，应改为
   **在降采样后的掩码上重新做连通域标定**（`slice_dataset.py` 已按连通域出框，
   所以实际是把"粘连后的块"当作一个实例）。

3. **crop 之间的类别不全相同**：`crop1026`/`crop1027` 有 `mito_ribo`、`er_mem_all`
   但**没有** `mito` 与 `er`。早先按字母序取前 N 个 crop 导致全部 404。
   现在按"目录里实际含有所需类别"筛选。

4. **只有 8 nm 的 EM**：`em/fibsem-uint8` 只有 s0=8 nm 一个数组的分辨率下限。
   若将来需要 4 nm 图像与 4 nm 标签严格对齐，需要另找数据源。

---

## 4. 复现命令

```bash
# 服务器上
conda activate dachuang
cd ~/dachuang_organelle
export HF_HOME=~/dachuang_organelle/models/hf
export HF_ENDPOINT=https://hf-mirror.com

# 1) 标签（已下 6 个 crop，可加 --max-crops 扩到 17）
python scripts/fetch_labels.py --dataset jrc_mus-kidney --labels mito,er \
    --out data/labels --eligible-cache data/eligible_crops_jrc_mus-kidney.json \
    --max-crops 17

# 2) EM 2D 切片（按 crop 的物理范围从 8 nm EM 取，标签自动降采样对齐）
python scripts/fetch_em_slices.py --manifest data/train_crop_manifest.csv \
    --labels data/labels --dataset jrc_mus-kidney \
    --z-stride 10 --max-slices 30 --out data/slices

# 3) 训练冒烟（已验证）
python scripts/smoke_train_detr.py --slices data/slices --classes mito,er \
    --iters 6 --batch 1 --size 480
```

---

## 5. 正式训练的建议配置（基于实测显存）

| 参数 | 建议值 | 依据 |
|---|---|---|
| size（短边） | 480 | 640 也能跑（288 MiB 前向），但 480 更快、batch 更大 |
| batch | 8 | 实测 batch=2 用 1123 MiB，85% 显存可到 7–9 倍 |
| epoch | 150–200 | DETR 微调文献常用值 |
| lr | 1e-4（主干 1e-5） | DETR 微调惯例：主干更小 |
| 梯度裁剪 | 0.1 | 实测未裁剪时 grad_norm 达 1372，必须裁剪 |
| 划分 | **按 crop 分组**，不能按切片随机划分 | 同 crop 的相邻层高度相似，随机划分会严重高估性能 |
| 增强 | 水平/垂直翻转 + 90° 旋转 | 电镜图像方向无先验，旋转安全 |

**分组划分是最容易犯的错**：同一个 crop 的相邻 z 层几乎是同一张图。
若随机划分，验证集里会混入训练集的近邻，指标虚高。必须按 crop 分组。
