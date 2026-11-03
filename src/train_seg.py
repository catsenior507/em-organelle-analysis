#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_seg.py —— 细胞器**语义分割**基线（用于验证"ER 该做分割而非检测"）

## 为什么要有这个脚本

前面的检测实验反复显示：内质网（er）在 DETR 框架下失败——
即使最宽松的 IoU 0.3，er AP 最高只有 0.13；IoU 0.75 时约 0.004；
把训练量从 3750 步加到 15000 步也没救回来。

我给出的诊断是：**内质网是连续管网，用"每类若干个轴对齐框"表示它本身不合适**。
但诊断需要证据。本脚本提供对照：同样数据、同样划分，改用**逐像素分割**，
看 er 的分割指标是否正常。若分割下 er 表现良好，则"表示方式错了"成立；
若分割下 er 仍然差，则问题在标注本身，需要另找原因。

## 设计

* 输入：`em.png`，缩放到固定尺寸（默认 256），灰度转 3 通道以复用常规骨干
* 输出：每类一个通道的二值掩码（sigmoid）
* 模型：紧凑 U-Net（约 2 M 参数）
* 损失：BCE + Dice（Dice 对小目标/细结构比纯 BCE 敏感得多）
* 指标：逐类 Dice 与 IoU，另报"前景像素占比"以判断是否退化为全背景
* 划分：与检测实验一致——**kidney 训练 / liver 验证**（跨器官）

## 用法

    python train_seg.py --train-root data/slices --val-root data/slices_liver \
        --classes mito,er --epochs 60 --batch 16 --size 256 --out runs/seg
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------

class SegDataset(Dataset):
    """读 <root>/<dataset>/<crop>/z####/{em.png,<class>.png}，统一缩放到 size。"""

    def __init__(self, root: str, classes: Tuple[str, ...], size: int = 256,
                 augment: bool = False, seed: int = 0, require_all: bool = True):
        self.classes = classes
        self.size = size
        self.augment = augment
        self.rng = np.random.RandomState(seed)
        self.items: List[Dict[str, Any]] = []
        for zdir in sorted(glob.glob(os.path.join(root, "*", "*", "z*"))):
            if not os.path.isdir(zdir):
                continue
            em = os.path.join(zdir, "em.png")
            if not os.path.exists(em):
                continue
            masks = {}
            for c in classes:
                p = os.path.join(zdir, f"{c}.png")
                if os.path.exists(p):
                    masks[c] = p
            if require_all and len(masks) < len(classes):
                continue
            if not masks:
                continue
            self.items.append({"dir": zdir, "em": em, "masks": masks})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        em = Image.open(it["em"]).convert("L").resize(
            (self.size, self.size), Image.BILINEAR)
        x = np.asarray(em, dtype=np.float32) / 255.0
        # 3 通道（保留常规卷积骨干的接口）
        x = np.stack([x, x, x], axis=0)

        y = np.zeros((len(self.classes), self.size, self.size), dtype=np.float32)
        for ci, c in enumerate(self.classes):
            p = it["masks"].get(c)
            if not p:
                continue
            m = Image.open(p).convert("L").resize(
                (self.size, self.size), Image.NEAREST)
            y[ci] = (np.asarray(m) > 0).astype(np.float32)

        # 可选忽略区：a 目录里存在 ignore.png 时启用。
        # 1 表示"该像素不计入损失与 Dice"，用于标注不完整的区域
        # （模型预测出真实但未标注的结构时不该被惩罚）。
        ignore = None
        ip = os.path.join(it["dir"], "ignore.png")
        if os.path.exists(ip):
            im = Image.open(ip).convert("L").resize(
                (self.size, self.size), Image.NEAREST)
            ignore = (np.asarray(im) > 127).astype(np.float32)

        if self.augment:
            if self.rng.rand() < 0.5:
                x = x[:, :, ::-1].copy()
                y = y[:, :, ::-1].copy()
                if ignore is not None:
                    ignore = ignore[:, ::-1].copy()
            if self.rng.rand() < 0.5:
                x = x[:, ::-1, :].copy()
                y = y[:, ::-1, :].copy()
                if ignore is not None:
                    ignore = ignore[::-1, :].copy()
            if self.rng.rand() < 0.5:                 # 90° 旋转（分割下无框坐标问题）
                k = int(self.rng.randint(1, 4))
                x = np.rot90(x, k, axes=(1, 2)).copy()
                y = np.rot90(y, k, axes=(1, 2)).copy()
                if ignore is not None:
                    ignore = np.rot90(ignore, k, axes=(0, 1)).copy()

        if ignore is not None:
            return (torch.from_numpy(x), torch.from_numpy(y),
                    torch.from_numpy(ignore)[None],      # (1,H,W) 便于逐像素加权
                    it["dir"].replace("\\", "/"))
        return (torch.from_numpy(x), torch.from_numpy(y),
                it["dir"].replace("\\", "/"))


# --------------------------------------------------------------------------
# 模型：紧凑 U-Net
# --------------------------------------------------------------------------

def double_conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """4 级下采样的 U-Net（约 2.5 M 参数）。

    通道数与空间尺寸的对应关系（以输入 256×256 为例）：

        s1  32ch  256×256   ──────────────┐
        s2  64ch  128×128   ────────┐     │
        s3 128ch   64×64    ────┐   │     │
        s4 256ch   32×32        │   │     │
        u4 上采样 64×64  256ch → cat(s3) 384 → c4 → 128ch
        u3 上采样 128×128 128ch → cat(s2) 192 → c3 →  64ch
        u2 上采样 256×256  64ch → cat(s1)  96 → c2 →  32ch
        out 1×1 卷积 → cout

    早期版本多写了一级上采样（u1/c1），导致拼接时通道数与空间尺寸都对不上。
    """

    def __init__(self, cin: int = 3, cout: int = 2, base: int = 32):
        super().__init__()
        b = base
        self.d1 = double_conv(cin, b)          # 32
        self.d2 = double_conv(b, b * 2)        # 64
        self.d3 = double_conv(b * 2, b * 4)    # 128
        self.d4 = double_conv(b * 4, b * 8)    # 256
        self.pool = nn.MaxPool2d(2)

        self.u4 = nn.ConvTranspose2d(b * 8, b * 8, 2, stride=2)
        self.c4 = double_conv(b * 8 + b * 4, b * 4)     # 384 -> 128
        self.u3 = nn.ConvTranspose2d(b * 4, b * 4, 2, stride=2)
        self.c3 = double_conv(b * 4 + b * 2, b * 2)     # 192 ->  64
        self.u2 = nn.ConvTranspose2d(b * 2, b * 2, 2, stride=2)
        self.c2 = double_conv(b * 2 + b, b)             #  96 ->  32
        self.out = nn.Conv2d(b, cout, 1)

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        s4 = self.d4(self.pool(s3))
        x = self.c4(torch.cat([self.u4(s4), s3], dim=1))
        x = self.c3(torch.cat([self.u3(x), s2], dim=1))
        x = self.c2(torch.cat([self.u2(x), s1], dim=1))
        return self.out(x)


# --------------------------------------------------------------------------
# 损失与指标
# --------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              eps: float = 1e-6, ignore: torch.Tensor = None) -> torch.Tensor:
    """逐类 soft Dice。对细结构（ER 管网）比纯 BCE 敏感。

    `ignore` 为 1 的像素在分子分母中都被剔除（乘 0），
    因此被忽略的区域既不奖励预测也不惩罚预测。
    """
    p = torch.sigmoid(logits)
    if ignore is not None:
        keep = 1.0 - ignore.clamp(0, 1)
        p = p * keep
        target = target * keep
    num = 2 * (p * target).sum(dim=(2, 3))
    den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1 - (num + eps) / (den + eps)).mean()


def seg_loss(logits, target, bce_w: float = 0.5, class_weights=None,
             ignore=None):
    """BCE + Dice 混合损失，支持"忽略区"。

    ## class_weights 为什么必要

    两类别联合训练时，像素平均的 BCE 会被**多数类主导**：mito 占 17–54%，
    er 只占 4–22%。模型为了压低 BCE 会把 ER 尽量判成背景。

    实测后果：er 的 Recall 只有 0.05、Dice 0.077，
    而"全预测为前景"的平凡基线是 2p/(1+p) ≈ 0.13——
    **模型比平凡策略还差**。这不是「ER 难学」，是损失把 ER 压制了。

    传入 class_weights（取各类前景比例的反比并归一化）可让两类贡献相当。

    ## ignore 解决什么问题

    标注**不完整**时，模型把真实但未标注的细胞器预测出来会被当作假阳性，
    训练时则被当作"应该预测成背景"，从而学到"不要预测它们"。
    这正是高尔基体那批数据的主要限制（标注覆盖 <3% 的图有 17 张，
    详见 docs/GOLGI-IMPROVEMENTS.md §10）。

    `ignore` 是 (B,1,H,W) 的 0/1 张量，1 表示该像素不计入损失。
    注意：**只忽略损失，不改变 GT**——被忽略的像素既不算正也不算负。
    """
    keep = None
    if ignore is not None:
        keep = 1.0 - ignore.clamp(0, 1)
        # 整批都被忽略：没有任何可学像素。
        # 直接返回"零值但带计算图"的张量，使梯度精确为 0；
        # 若返回 Python 的 0.0，backward 会报 "does not require grad"。
        if float(keep.sum()) < 1.0:
            return logits.sum() * 0.0

    if bce_w > 0 or class_weights is not None:
        ew = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=ew.dtype, device=ew.device)
            ew = ew * w.view(1, -1, 1, 1)
        if keep is not None:
            ew = ew * keep
            bce = ew.sum() / keep.sum().clamp(min=1.0)
        else:
            bce = ew.mean()
    else:
        bce = torch.zeros((), device=logits.device)

    return bce_w * bce + (1 - bce_w) * dice_loss(logits, target, ignore=ignore)


@torch.no_grad()
def trivial_baselines(y: torch.Tensor) -> Dict[str, Any]:
    """算出"全预测前景 / 全预测背景"两个**平凡基线**的 Dice。

    这个对照决定了如何解读一个很低的 Dice：
      * 模型 Dice < 全前景基线 -> 模型比平凡策略还差，问题在训练/损失，不在数据
      * 模型 Dice > 全前景基线但绝对值低 -> 才可能是任务本身难
    不做这个对照，0.07 这样的数字会被误读成"ER 太难学"。
    """
    gt = (y.numpy() > 0.5)
    out = {}
    for ci in range(gt.shape[1]):
        g = gt[:, ci]
        n = float(g.sum())
        tot = float(g.size)
        out[str(ci)] = {
            "gt_frac": round(n / tot, 5) if tot else 0.0,
            "dice_all_foreground": (round(2 * n / (tot + n), 4)
                                    if (tot + n) > 0 else None),
            "dice_all_background": 0.0 if n > 0 else None,
        }
    return out


@torch.no_grad()
def evaluate(model, dl, dev, n_classes: int, thr: float = 0.5) -> Dict[str, Any]:
    """逐类 Dice / IoU / precision / recall。"""
    inter = np.zeros(n_classes)
    pred_sum = np.zeros(n_classes)
    gt_sum = np.zeros(n_classes)
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    per_crop: Dict[str, Dict[str, List[float]]] = {}
    n_seen = 0          # 已评估的图像数
    px_per_img = 0      # 每张图的像素数（H*W）
    model.eval()
    for x, y, dirs in dl:
        x = x.to(dev)
        logits = model(x)
        pred = (torch.sigmoid(logits) > thr).cpu().numpy()
        gt = (y.numpy() > 0.5)
        n_seen += int(gt.shape[0])
        px_per_img = int(gt.shape[2] * gt.shape[3])
        for ci in range(n_classes):
            p = pred[:, ci]
            g = gt[:, ci]
            inter[ci] += np.logical_and(p, g).sum()
            pred_sum[ci] += p.sum()
            gt_sum[ci] += g.sum()
            tp[ci] += np.logical_and(p, g).sum()
            fp[ci] += np.logical_and(p, ~g).sum()
            fn[ci] += np.logical_and(~p, g).sum()
        for bi, d in enumerate(dirs):
            crop = d.split("/")[-2] if "/" in d else d
            e = per_crop.setdefault(crop, {})
            for ci in range(n_classes):
                p = pred[bi, ci]
                g = gt[bi, ci]
                i_ = float(np.logical_and(p, g).sum())
                d_ = float(p.sum() + g.sum())
                e.setdefault(str(ci), []).append(
                    2 * i_ / d_ if d_ > 0 else float("nan"))

    out = {"per_class": {}, "per_crop": {}}
    dice_all, iou_all = [], []
    # 总像素数（用于算"全预测前景"的平凡基线）
    tot_px = float(n_seen) * float(px_per_img) if n_seen else 0.0
    for ci in range(n_classes):
        d_ = 2 * inter[ci] / max(1e-9, pred_sum[ci] + gt_sum[ci])
        i_ = inter[ci] / max(1e-9, pred_sum[ci] + gt_sum[ci] - inter[ci])
        prec = tp[ci] / max(1e-9, tp[ci] + fp[ci])
        rec = tp[ci] / max(1e-9, tp[ci] + fn[ci])
        # 平凡基线：把整幅图都判为前景时的 Dice = 2*gt/(tot+gt)
        base1 = (2 * gt_sum[ci] / (tot_px + gt_sum[ci])
                 if tot_px > 0 else None)
        out["per_class"][str(ci)] = {
            "dice": round(float(d_), 4), "iou": round(float(i_), 4),
            "precision": round(float(prec), 4), "recall": round(float(rec), 4),
            "pred_px": int(pred_sum[ci]), "gt_px": int(gt_sum[ci]),
            "dice_all_foreground": (round(float(base1), 4)
                                    if base1 is not None else None),
            "beats_trivial": (bool(d_ > base1) if base1 is not None else None)}
        dice_all.append(d_)
        iou_all.append(i_)
    out["mean_dice"] = round(float(np.mean(dice_all)), 4) if dice_all else 0.0
    out["mean_iou"] = round(float(np.mean(iou_all)), 4) if iou_all else 0.0

    for crop, e in sorted(per_crop.items()):
        row = {}
        for ci in range(n_classes):
            v = [x for x in e.get(str(ci), []) if not np.isnan(x)]
            row[str(ci)] = round(float(np.mean(v)), 4) if v else None
        out["per_crop"][crop] = row
    return out


def split_items_by_crop(items: List[Dict[str, Any]], val_frac: float,
                        seed: int = 0) -> Tuple[List[Dict[str, Any]],
                                                List[Dict[str, Any]],
                                                List[str]]:
    """按 crop 分组划分数据集项，返回 (训练项, 验证项, 验证 crop 列表)。

    **必须按 crop 分组**：同一 crop 的相邻 z 层几乎相同，随机按切片划分会让
    验证集混入训练集的近邻，指标虚高。

    提成独立函数是为了可单测——这段逻辑原先内联在 main() 里，无法验证。
    """
    if not items:
        return [], [], []
    by_crop: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        parts = it["dir"].replace("\\", "/").split("/")
        crop = parts[-2]
        # 多来源合并时，不同来源可能有同名 crop（两个数据集都有 crop129）。
        # 若不区分来源，按 crop 分组会把它们当成同一个 crop，
        # 导致验证集里混入训练来源的同名 crop —— 即跨来源泄漏。
        tag = it.get("source_tag")
        if tag:
            crop = f"{tag}::{crop}"
        by_crop.setdefault(crop, []).append(it)
    crops = sorted(by_crop)
    if len(crops) <= 1:
        # 只有一个 crop 时无法按 crop 分组，全部给训练，验证留空
        return [it for c in crops for it in by_crop[c]], [], []
    rng = np.random.RandomState(seed)
    order = list(crops)
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * float(val_frac))))
    n_val = min(n_val, len(order) - 1)          # 至少留 1 个训练 crop
    val_crops = sorted(order[:n_val])
    vset = set(val_crops)
    tr = [it for c in crops if c not in vset for it in by_crop[c]]
    va = [it for c in crops if c in vset for it in by_crop[c]]
    return tr, va, val_crops


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="细胞器语义分割（U-Net）")
    ap.add_argument("--train-root", default="data/slices")
    ap.add_argument("--train-roots", default=None,
                    help="逗号分隔的**多个**训练根目录，合并成一个训练集。"
                         "用于把不同来源的数据一起训练（例如自有高尔基体插图 + "
                         "ASEM 的 golgi 体切片）。每一层目录结构仍须是 "
                         "<root>/<crop>/z####/{em.png,<类>.png}。"
                         "给定时覆盖 --train-root。")
    ap.add_argument("--val-root", default="data/slices_liver",
                    help="验证集目录；传 'split' 表示同器官内按 crop 分组划分")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="--val-root split 时验证 crop 的比例")
    ap.add_argument("--val-images", default=None,
                    help="固定验证用的 crop 名单（每行一个）。**多种子稳定性实验必须用这个**："
                         "否则换 --seed 会同时改变划分，测出来的差异混合了"
                         "'训练随机性'和'验证集不同'两个因素，"
                         "无法作为 run-to-run 方差（CellMap 那轮就吃了这个亏）")
    ap.add_argument("--classes", default="mito,er")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--class-balance", action="store_true",
                    help="按类别前景比例的反比给 BCE 加权。解决两类联合训练时"
                         "少数类（er，占 4–22%）被多数类（mito，占 17–54%）"
                         "压制的问题；不加时 er 的 Dice 会低于『全猜前景』的平凡基线")
    ap.add_argument("--dice-weight", type=float, default=0.5,
                    help="损失里 Dice 项的权重（其余为 BCE）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/seg_unet")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--resume", default="none")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    classes = tuple(s.strip() for s in args.classes.split(",") if s.strip())

    # 多来源合并：把一个或多个根目录的切片表拼成一个训练集。
    #
    # 不同来源的 crop 名可能重复（比如两个数据集都有 crop129），
    # 因此给每个来源的目录名加前缀，保证按 crop 分组划分时不会把
    # 不同来源的同名 crop 混在一起（那会造成跨来源泄漏）。
    def _load_roots(roots: List[str], augment: bool):
        ds_list = []
        for r in roots:
            d = SegDataset(r, classes, args.size, augment=augment, seed=args.seed)
            tag = os.path.basename(os.path.normpath(r)) or r
            for it in d.items:
                # 仅在多来源时改写 dir 的显示名会影响读取，故只记录 tag，
                # 分组时用 tag+crop 作为键（见下方 split）
                it["source_tag"] = tag
            ds_list.append(d)
        return ds_list

    roots: List[str] = ([s.strip() for s in args.train_roots.split(",") if s.strip()]
                        if args.train_roots else [args.train_root])

    if len(roots) > 1:
        print(f"多来源合并训练：{len(roots)} 个根目录")
        merged = SegDataset(roots[0], classes, args.size, augment=True,
                            seed=args.seed)
        items: List[Dict[str, Any]] = []
        for r in roots:
            d = SegDataset(r, classes, args.size, augment=True, seed=args.seed)
            tag = os.path.basename(os.path.normpath(r))
            for it in d.items:
                it["source_tag"] = tag
                items.append(it)
            print(f"  {r}: {len(d)} 个切片")
        merged.items = items
        ds_tr = merged
    else:
        ds_tr = SegDataset(args.train_root, classes, args.size, augment=True,
                           seed=args.seed)
    # 找不到任何切片时**必须单独报错**。
    # 早期版本这里直接走划分逻辑，然后报"验证集为空（crop 太少）"，
    # 把一个"路径层级不对/数据没找到"的问题伪装成"crop 数量不足"，
    # 排查时容易被带偏（本项目就因此多花了一轮：tar 包顶层多了一层目录）。
    if len(ds_tr) == 0:
        print(f"[错误] 在 {args.train_root} 下没找到任何切片。\n"
              f"        期望结构：<root>/<数据集>/<crop>/z####/em.png\n"
              f"        请检查路径层级是否多了一层，或 --classes={classes} "
              f"对应的掩码文件是否缺失。", file=sys.stderr)
        # 顺手提示实际看到的结构，减少来回排查
        probe = sorted(glob.glob(os.path.join(args.train_root, "*")))
        if probe:
            print(f"        该目录下实际有："
                  f"{[os.path.basename(p) for p in probe[:6]]}", file=sys.stderr)
            deeper = sorted(glob.glob(os.path.join(args.train_root, "*", "*")))
            if deeper:
                print(f"        再下一层："
                      f"{[os.path.basename(p) for p in deeper[:6]]}", file=sys.stderr)
        return 2

    # --val-root split 表示**同器官内按 crop 分组划分**，用于取得"同分布"对照。
    # 这对判断瓶颈至关重要：若同器官 Dice 高而跨器官 Dice 低，说明瓶颈是域偏移；
    # 若同器官也低，说明模型/标注本身有问题。
    if args.val_root == "split":
        if args.val_images:
            # 固定验证 crop：多种子稳定性实验的唯一正确做法。
            # 只让训练侧（初始化 + 增广）随种子变化，验证集保持不变，
            # 这样 seed 之间的差异才能解释为训练随机性。
            fixed = {l.strip() for l in open(args.val_images, encoding="utf-8")
                     if l.strip()}
            def _crop(it):
                c = it["dir"].replace("\\", "/").split("/")[-2]
                tag = it.get("source_tag")
                return f"{tag}::{c}" if tag else c
            va_items = [it for it in ds_tr.items if _crop(it) in fixed]
            tr_items = [it for it in ds_tr.items if _crop(it) not in fixed]
            val_crops = sorted(fixed)
            missing = sorted(fixed - {_crop(it) for it in ds_tr.items})
            print(f"固定验证集：{len(val_crops)} 个 crop，"
                  f"验证 {len(va_items)} 切片 / 训练 {len(tr_items)} 切片")
            if missing:
                print(f"  [警告] 名单里有 {len(missing)} 个 crop 在数据中不存在："
                      f"{missing[:6]}", file=sys.stderr)
            if not va_items:
                print("[错误] --val-images 指定的 crop 在数据里一个都没匹配到",
                      file=sys.stderr)
                return 2
            if not tr_items:
                print("[错误] 固定验证集后训练集为空", file=sys.stderr)
                return 2
        else:
            tr_items, va_items, val_crops = split_items_by_crop(
                ds_tr.items, args.val_frac, args.seed)
        if not va_items:
            print(f"[错误] 按 crop 分组划分后验证集为空：共找到 "
                  f"{len(ds_tr)} 个切片、"
                  f"{len({it['dir'].replace(chr(92), '/').split('/')[-2] for it in ds_tr.items})} "
                  f"个 crop，至少需要 2 个不同 crop 才能划分。", file=sys.stderr)
            return 2
        ds_tr = SegDataset(args.train_root, classes, args.size, augment=True,
                           seed=args.seed)
        ds_tr.items = tr_items
        ds_va = SegDataset(args.train_root, classes, args.size, augment=False)
        ds_va.items = va_items
        val_name = f"split:{val_crops}"
        print(f"同器官分组划分：{len(tr_items)} 训练 / {len(va_items)} 验证"
              f"（验证 crop {val_crops}）")
    elif args.val_root and os.path.isdir(args.val_root):
        ds_va = SegDataset(args.val_root, classes, args.size, augment=False)
        val_name = os.path.basename(os.path.normpath(args.val_root))
    else:
        ds_va = None
        val_name = None
    print(f"训练切片 {len(ds_tr)}（{os.path.basename(os.path.normpath(args.train_root))}）"
          f"；验证切片 {len(ds_va) if ds_va else 0}（{val_name}）")
    if not len(ds_tr):
        print("[错误] 训练集为空", file=sys.stderr)
        return 2

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=False)
    dl_va = (DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
             if ds_va else None)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(3, len(classes), args.base).to(dev)
    npar = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={dev}  模型 {npar:.2f} M 参数  类别 {classes}")

    # 类别权重：按各类前景比例的反比，归一化到均值为 1（保持整体损失量级不变）。
    # 从训练集抽样估计，避免读全量（几百个 256×256 掩码已足够稳定）。
    class_weights = None
    if args.class_balance:
        n_samp = min(len(ds_tr), 200)
        idx = np.linspace(0, len(ds_tr) - 1, n_samp).astype(int)
        fg = np.zeros(len(classes))
        for i in idx:
            _, y_i, _ = ds_tr[int(i)]
            fg += (y_i.numpy() > 0.5).sum(axis=(1, 2)) / float(args.size ** 2)
        fg /= max(1, n_samp)
        inv = 1.0 / np.clip(fg, 1e-3, None)
        w = inv / inv.mean()
        class_weights = w.tolist()
        print(f"类别前景比例 {np.round(fg, 4).tolist()} -> "
              f"BCE 权重 {np.round(w, 3).tolist()}")
    print(f"损失：BCE {(1-args.dice_weight):.2f} + Dice {args.dice_weight:.2f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    start = 0
    ckpt = os.path.join(args.out, "last.pt")
    if args.resume == "auto" and os.path.exists(ckpt):
        c = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        start = c["epoch"] + 1
        print(f"从 epoch {start} 续训")

    hist: List[Dict[str, Any]] = []
    hp = os.path.join(args.out, "history.json")
    if start and os.path.exists(hp):
        hist = json.load(open(hp, encoding="utf-8"))
    best = -1.0
    t0 = time.time()
    for ep in range(start, args.epochs):
        model.train()
        tot = 0.0
        n = 0
        for batch in dl_tr:
            # 数据集在有 ignore.png 时返回 4 元组，否则仍是 3 元组。
            # 这样旧数据（无 ignore）完全不受影响，保持向后兼容。
            if len(batch) == 4:
                x, y, ign, _ = batch
                ign = ign.to(dev, non_blocking=True)
            else:
                x, y, _ = batch
                ign = None
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            logits = model(x)
            loss = seg_loss(logits, y, bce_w=1.0 - args.dice_weight,
                            class_weights=class_weights, ignore=ign)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            n += 1
        sched.step()
        tr = tot / max(1, n)
        rec: Dict[str, Any] = {"epoch": ep, "train_loss": round(tr, 5),
                               "lr": opt.param_groups[0]["lr"],
                               "elapsed_s": round(time.time() - t0, 1)}
        if dl_va:
            ev = evaluate(model, dl_va, dev, len(classes))
            rec["val_mean_dice"] = ev["mean_dice"]
            rec["val"] = ev["per_class"]
            rec["val_per_crop"] = ev["per_crop"]
            line = " ".join(
                f"{classes[int(k)]} dice={v['dice']:.3f}" for k, v in ev["per_class"].items())
            print(f"epoch {ep+1}/{args.epochs}  loss {tr:.4f}  "
                  f"val {line}  mDice {ev['mean_dice']:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            if ev["mean_dice"] > best:
                best = ev["mean_dice"]
                torch.save({"model": model.state_dict(), "epoch": ep,
                            "classes": list(classes), "best": best},
                           os.path.join(args.out, "best.pt"))
        else:
            print(f"epoch {ep+1}/{args.epochs}  loss {tr:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
        hist.append(rec)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "classes": list(classes),
                    "best": best}, ckpt)
        if args.save_every and (ep + 1) % args.save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "classes": list(classes), "best": best},
                       os.path.join(args.out, f"epoch{ep+1:04d}.pt"))
        with open(hp, "w", encoding="utf-8") as fh:
            json.dump(hist, fh, ensure_ascii=False, indent=1)

    print(f"\n完成。最优 val mDice {best:.4f}")
    if dl_va:
        ev = evaluate(model, dl_va, dev, len(classes))
        print("最终逐类：")
        for k, v in ev["per_class"].items():
            print(f"  {classes[int(k)]}: dice={v['dice']:.4f} iou={v['iou']:.4f} "
                  f"P={v['precision']:.4f} R={v['recall']:.4f}")
        print(f"  mean_dice={ev['mean_dice']:.4f} mean_iou={ev['mean_iou']:.4f}")
        print("\n逐 crop Dice（看稳定性）：")
        for crop, row in ev["per_crop"].items():
            vals = " ".join(f"{classes[int(k)]}={v}" for k, v in row.items())
            print(f"  {crop:<14} {vals}")
        with open(os.path.join(args.out, "eval_last.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(ev, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
