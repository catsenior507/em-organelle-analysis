# 高尔基体标注数据：网上现有资源

[English](GOLGI-DATA-SOURCES.md) ｜ **中文**

> 背景：自己标注没时间，需要在公开数据里找高尔基体的 EM 标注。
> 检索日期 2026-11-02。**所有 URL 均已实测可达（2026-11-02）。**

---

## 1. 首选：ASEM（哈佛 Kirchhausen 实验室）

**这是目前最匹配的资源**——它同时提供**训练好的高尔基体模型**和
**高尔基体真值体数据**，且是 EM（FIB-SEM）。

| 项 | 内容 |
|---|---|
| 全称 | Automated Segmentation of Intracellular Substructures in EM |
| 机构 | Kirchhausen Lab, Harvard Medical School |
| 论文 | Gallusser et al., *J Cell Biol* 222(2):e202208005 (2023) |
| 许可 | **CC BY-SA 4.0**（可商用，需署名 + 相同方式共享） |
| 访问 | S3 公开桶，**无需 AWS 账号** |
| 注册页 | https://registry.opendata.aws/asem-project/ |
| 代码 | https://github.com/kirchhausenlab/incasem（**不在 PyPI 上**） |

### 1.1 预训练模型（各 70 MB，已下载到 `models/asem/`）

```
models/pretrained_checkpoints/model_checkpoint_1837_golgi_CF.pt    ← 高尔基体
models/pretrained_checkpoints/model_checkpoint_1841_er_CF.pt
models/pretrained_checkpoints/model_checkpoint_1847_mito_CF.pt
models/pretrained_checkpoints/model_checkpoint_1675_mito_HPF.pt
models/pretrained_checkpoints/model_checkpoint_2000_np_HPF.pt      核孔
models/pretrained_checkpoints/model_checkpoint_1986_ccp_HPF.pt     网格蛋白
models/pretrained_checkpoints/model_checkpoint_2661_clathrin.pt
models/pretrained_checkpoints/model_checkpoint_2632_eres_cop.pt
```

**CF = 化学固定，HPF = 高压冷冻+冷冻置换。** 两种制样方式各有一套模型。
我们自己的图像属于哪种需要判断——这是选择模型的前提。

### 1.2 高尔基体真值数据（关键）

26 个数据集中，**只有 4 个带 `labels/golgi`**：

| 数据集 | 标注卷 | golgi 体形状 | chunk 数 | 估算大小 |
|---|---|---|---|---|
| 46 | er / golgi / mito | [1921, 1100, 4600] | ~5184 | **约 40 GB** |
| 58 | er / golgi / mito | [1982, 1000, 4752] | ~4864 | **约 40 GB** |
| **61** | er / golgi / mito | [210, 283, 371] | **18** | **约 150 MB** |
| **64** | er / golgi / mito | [284, 204, 424] | **24** | **约 200 MB** |

> **61 与 64 的 golgi 标注只是原始体数据的一个子区域**（raw 是 1805×1400×4800），
> 所以体积很小，实用。46/58 是全卷标注，太大。

**这三个类别同时标注（er + golgi + mito），意味着这套数据可以用来做
真正的多类互作分析**——这正是我们自己数据缺的那一块。

### 1.3 一个必须注意的架构问题

ASEM 模型是 **3D U-Net**（5.84 M 参数，3D 卷积），
而我们的图像是**单张 2D 切片**。直接把一张切片喂 3D 模型效果会很差。

两条路：

* **A（推荐）**：把 ASEM 的 golgi 真值体**切成 2D 切片**，
  用我们已有的 `train_seg.py` 训练一个 2D 模型。管线现成，风险最低。
* **B**：如果我们的编号图像（如 11_0002、11_0003…）确实是**连续 z 层**，
  可以堆 3–5 层成小体数据跑 3D 模型。需先确认它们是否连续。

### 1.4 检查点架构（已解析，可自行重建）

`model_checkpoint_1837_golgi_CF.pt` 是**普通 state_dict**（非 TorchScript），
键名是经典 `pytorch-3dunet` 风格：

```
unet.l_conv.0..3   编码器  32→64→128→256，每块 2×(Conv3d3³+ReLU)，块间 MaxPool3d(2)
unet.r_conv.0.2    解码   up(256)→256 ++ skip(l_conv.2=128) = 384 → conv → 128
unet.r_conv.0.1    解码   up(128)→128 ++ skip(l_conv.1= 64) = 192 → conv →  64
unet.r_conv.0.0    解码   up( 64)→ 64 ++ skip(l_conv.0= 32) =  96 → conv →  32
final_conv_pass.conv_pass.0   32 → 2
```

**单个张量形状已全部核对，可以离线重建，不需要 `incasem` 包**
（该包不在 PyPI，只能从 GitHub 装，而 GitHub 在本机与服务器均不可直连）。

> 注：state_dict 里只有 conv 的 weight/bias，**没有 BatchNorm 参数**，
> 说明这一版用的是无 BN 的 3D U-Net。

---

## 2. 次选：CellMap / OpenOrganelle（我们已在用）

CellMap 的 73 类里包含 `golgi`。但：

* 它的形态是**逐 crop 的 figshare 条目**（如 `jrc_mus-kidney` crop164），
  要按 crop 逐个拉取，比较零碎
* 我们的经验是**CellMap 单模态模型迁移很差**（mito 零样本 Dice 0.1516），
  所以即使训出 golgi 模型，迁移到我们数据大概率也不理想

**结论**：CellMap 的 golgi 可以作为补充训练数据，但不应作为首选。

---

## 3. 排除掉的选项

| 资源 | 为何不合适 |
|---|---|
| MitoEM 2.0 | 只有线粒体 |
| EMPIAR-11695 | 有 Golgi 标注，但 434 GB，且是 Cal51 细胞系 FIB-SEM |
| nanotomy | 2D 大视野 EM，有标注但以病理为主，Golgi 覆盖不确定 |
| 各类荧光高尔基体数据集 | 模态不符（我们是 EM） |

---

## 4. 实测的访问性能

| 路径 | 速度 | 备注 |
|---|---|---|
| 本机 → ASEM S3 | **约 47 kB/s** | 70 MB 模型花了 **25 分钟** |
| 服务器 → S3 | 0.12–0.37 MB/s | 此前 CellMap 实测 |

**结论：从 S3 拉大文件很慢。** 61/64 的 golgi 标注（各约 150–200 MB）
按 47 kB/s 计算需要 1–1.5 小时，**建议放到服务器上拉**（快 3–8 倍），
或只拉需要的子区域 chunk。

---

## 5. 建议的下一步

按价值排序：

| 优先级 | 动作 | 理由 |
|---|---|---|
| **1** | **等你传完自己的高尔基体标注**，先用现有管线训一个 2D 模型 | 同一批成像条件，迁移问题最小 |
| **2** | 若自己的标注太少：从 **61/64** 拉 golgi 真值，切成 2D 补充训练 | 同一模态（FIB-SEM），类别正确 |
| 3 | 用 ASEM golgi 模型做**对照基线**（需先重建 3D U-Net） | 提供一个"通用模型零样本"的参照，类似 MitoNet 之于线粒体 |
| 4 | 若确认我们图像是连续 z 层，再考虑直接用 3D 模型 | 需先验证连续性 |

**关键提醒**：先确认我们自己的图像是 **CF 还是 HPF 制样**。
ASEM 的 golgi 模型只有 CF 版本（`1837_golgi_CF`），
若我们是 HPF，直接套用的效果会打折。
