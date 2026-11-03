# Transformer / DETR 阶段：代码说明与服务器运行手册

> 本目录替代 `code/detr_region_feature.py`。
> 原实现在**第一次前向就会崩溃**，且其监督信号来自被截断的像素表。
> 详细缺陷清单见 `docs/AUDIT-REPORT.md` §2.4。

---

## 1. 为什么必须重写

| # | 原实现 | 后果 |
|---|---|---|
| 1 | `collate_fn` 返回 `{'targets': [t0, t1, ...]}`（list），而 `compute_loss` 写 `targets['boxes'][i]`（dict 语义） | **第一次前向必 `TypeError`**，脚本从未成功训练过 |
| 2 | `Dataset.__getitem__` 先做 `Resize+ToTensor+Normalize`，再把 tensor 交给 `DetrImageProcessor` | **双重归一化**；处理器还会把短边缩到 800 px，与声明的 `img_size=256` 完全脱节 |
| 3 | `self.detr.model.decoder.num_queries = 10` 并新建 `query_embed` | 解码器的 `embed_pos` 仍是 100 个查询，前向形状不匹配；且 `DetrDecoder` 会把 `query_embed` 当**位置**嵌入用 |
| 4 | `iou_matrix` 上做 `torch.topk` 贪心匹配 | 不是一对一置换，同一个目标可被多个查询命中；与 `进度3.pdf` 写的匈牙利算法不符 |
| 5 | 损失只有 `CE + MSE(框) + 0.1·MSE(特征)` | 没有 GIoU；无目标分支对 `[Q]` 做 BCE(target=1)，形状与语义都错 |
| 6 | `image_id = torch.tensor([idx])`，`idx` 在该作用域**未定义** | `NameError` |
| 7 | 特征口径与 MATLAB 侧完全不同（roughness / complexity / gray_level 各自一套） | 即使训好也无法与前面的表征学习结果对话 |
| 8 | 读 `change_feature_json`，其中 7404 个区域的 `region_pixels` 已被压成 **1 个点** | 监督信号本身是坏的 |

---

## 2. 文件结构

```
tools/detr/
├── dataset.py    数据管线：读 annotations_v2、单次预处理、按图分组划分
├── losses.py     匈牙利匹配 + 加权 CE + L1 + GIoU + 属性标准化 MSE
├── model.py      DetrRegionModel：预训练 DetrModel + 重建查询 + 检测头 + 属性头
├── train.py      训练/评估入口
├── selftest.py   CPU 自检（无需权重），12 项检查
└── README.md     本文件
```

唯一的数据来源是 `tools/morpho/extract_regions.py` 产出的 `data/annotations_v2/`：

```bash
python tools/morpho/extract_regions.py --out data/annotations_v2 --jobs 8
```

---

## 3. 已在本机（无 GPU、离线）验证的部分

```bash
python tools/detr/selftest.py
```

实测输出：

```
✓ 数据集清单可构建且框归一化正确       87 张图，目标数 1–24，共 562 个目标
✓ 按图分组划分无泄漏且可复现           train=65 val=22，无重叠，可复现
✓ collate_fn 返回 list 语义的 targets
✓ GIoU 数值正确且退化框不产生 NaN      自交=1.0，分离=-0.7778，退化框有限
✓ cxcywh ↔ xyxy 往返一致
✓ 匈牙利匹配是一对一置换               src=[0, 5] dst=[0, 1]（一对一，无重复）
✓ 无目标样本不崩且损失有限             loss=11.5739，梯度有限且非零
✓ 整批无目标不崩                       loss=0.9850，背景类损失有梯度
✓ 损失在可学习问题上下滑               8.7563 → 0.2153
✓ eos_coef 正确作用于背景类
✓ AP 计算合理性                        完美=1.0，错位=0.0，无目标=None
✓ （反例）旧 topk 匹配会重复           旧 topk 总 IoU 1.03 vs 匈牙利 2.09

结果：12/12 通过
```

**未验证的部分**：模型前向、训练收敛、mAP 数值。原因是本机
（a）无 CUDA，（b）无法访问 `huggingface.co`，`facebook/detr-resnet-50`
权重未缓存。`model.py` 中 `DetrDecoder` 的查询/位置嵌入重建逻辑已按
HuggingFace 实现写成，但**首次在服务器上运行时应先执行第 5 节的冒烟测试**。

---

## 4. 服务器环境准备

```bash
# 1) 依赖（建议独立 conda 环境）
conda create -n organelle python=3.11 -y
conda activate organelle
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers scipy numpy pandas pillow opencv-python scikit-learn networkx
# 可选但推荐：标准 COCO 评估
pip install pycocotools

# 2) 缓存 DETR 权重（服务器通常能直连；国内可设镜像）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download facebook/detr-resnet-50

# 3) 确认
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

若服务器也不通外网：在有网机器上 `huggingface-cli download facebook/detr-resnet-50`，
然后把整个 `~/.cache/huggingface/hub/models--facebook--detr-resnet-50` 目录
拷到服务器的同一路径下，并 `export HF_HUB_OFFLINE=1`。

---

## 5. 运行步骤

### 5.1 冒烟测试（务必先做）
用 4 张图、2 个 epoch 验证管线连通性。这一步会暴露 `model.py` 里任何
与服务器上 transformers 版本相关的字段名问题：

```bash
python tools/detr/train.py --smoke-test --batch-size 2 --short-side 480 \
    --device cuda --out runs/detr_smoke
```

期望：能跑完 2 个 epoch 并打印 `val_mAP@50`（数值可能很低，这是正常的——
只用了 4 张图）。**如果这一步报形状错误，请把完整 traceback 发回来。**

### 5.2 正式训练

```bash
python tools/detr/train.py \
    --epochs 200 --batch-size 4 --lr 1e-4 --lr-backbone 1e-5 \
    --weight-decay 1e-4 --lr-drop 150 --grad-clip 0.1 \
    --num-queries 15 --short-side 800 --val-fraction 0.25 --seed 42 \
    --device cuda --out runs/detr_v1
```

超参与 `进度3.pdf` 给出的 config 一致。显存不足时按此顺序降：
`--short-side 600` → `--batch-size 2` → `--num-queries 12`。

### 5.3 只评估已有权重

```bash
python tools/detr/train.py --eval-only --checkpoint runs/detr_v1/best.pth \
    --device cuda --out runs/detr_v1_eval
```

---

## 6. 输出物

| 文件 | 内容 |
|---|---|
| `runs/detr_v1/history.json` | 每个 epoch 的训练损失、`val_mAP@50`、属性 MAE、耗时、学习率 |
| `runs/detr_v1/best.pth` | 最佳权重 + 优化器状态 + **属性 mean/std**（用于把预测反标准化回物理量） |
| `runs/detr_v1/run_config.json` | 全部超参 + 训练/验证的**具体图片名单**（保证划分可复现） |
| `runs/detr_v1/eval.json` | `--eval-only` 时的指标 |

### 指标怎么读

`evaluate()` 除了 mAP，还会给出**属性回归的基线对照**：

- `attr_mae_macro`：属性平均绝对误差（已反标准化到物理量纲）
- `attr_mae_baseline_macro`：**"永远预测均值"**的基线 MAE
- `attr_beats_baseline`：`attr_mae_macro < 0.95 × baseline` 才为 `true`

这一项是刻意加的。原项目里 `neural_network_training.m` 报告 MSE 1.1848，
而"预测均值"的基线**恰好是 1.0**——即模型不如瞎猜，但日志照样打印"训练完成"。
有了这个布尔量，"没学到"就无法被隐藏。

---

## 7. 已知局限（写论文时必须说明）

1. **单类别检测**：`num_classes = 1`。黄色标注全是字形 "M"（线粒体），
   因此现阶段**没有多类别信息**可用。若后续能从标注中恢复更多类型，
   把 `DetrDataConfig`/`OrganelleNetwork.node_types` 接上即可。
2. **样本量极小**：87 张图 / 562 个目标。DETR 原论文在 COCO（118k 图）上训练。
   本项目必然严重过拟合，**不要**期望高 mAP。可行方向：
   （a）把 `--freeze-backbone` 打开只训头；
   （b）先在合成数据上预训练再微调；
   （c）降低 `--num-queries` 到接近每图目标数中位数（实测中位数 5）。
3. **验证集只有 22 张图**：mAP 的方差会很大，别把 0.02 的差异当结论。
4. **框是轮廓的外接矩形**：`simple_ap` 用 IoU≥0.5 判正，而细胞器形状不规则，
   外接框重叠度天然偏低，mAP 会被系统性低估。建议同时报告
   **中心点误差（px / µm）**。
5. **预处理分辨率**：`DetrImageProcessor` 把短边缩到 800（长边上限 1333），
   而原图是 2048×2115，等于**下采样**。线粒体宽约 40–220 px，
   下采样后仍有几十像素，尚可；但更细的结构会丢。
6. **属性损失只在匹配上的查询上计算**，这是 DETR 的设定，不是缺陷；
   但因此属性的样本量小于总目标数。

---

## 8. 建议的后续工作

- [ ] 服务器冒烟测试通过后，把 `runs/detr_smoke` 的日志留档
- [ ] 加 `--freeze-backbone` 的对照实验（小样本下常常更优）
- [ ] 报告中心点误差（µm），与外接框 mAP 并列
- [ ] 若 OCR 能恢复更多标注类型，把 `num_classes` 从 1 提到实际类别数
- [ ] 用 `tools/network/` 把检测结果接进互作网络，形成"检测 → 网络 → 涌现指标"的完整链路
