#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_crossdomain.py —— 跨数据集评估：模型在另一个域上还剩多少性能

## 为什么值得做

我们有两个高尔基体数据来源：

* **自有**：期刊插图，TEM，JPEG，手绘绿描边（48 张图 / 976 正 tile）
* **ASEM**：BSC-1 细胞，FIB-SEM，5 nm 各向同性，人工三维标注（1052 tile）

在自有数据上训练、再去 ASEM 上评估，可以直接回答：
**"这个模型学到的是高尔基体的通用形态，还是我们这批图的特定外观？"**

## 判据

* 跨域 Dice 接近域内 Dice -> 学到的是通用形态，模型可迁移
* 跨域 Dice 显著更低（接近 0）-> 学到的是域特定特征，不可迁移

这与线粒体那边已经做过的跨器官实验（自研模型 0.7839 vs CellMap 零样本 0.1516）
是同一类问题，但这次两侧都是高尔基体、都是 EM，条件更干净。

## 实现

复用 `train_seg.SegDataset` 保证预处理与训练完全一致
（灰度 -> 复制 3 通道 -> 归一化），避免"评估时预处理不同"这类隐性错误。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _c in (os.path.join(ROOT, "scripts"),
           os.path.join(ROOT, "tools", "server", "scripts"),
           os.path.join(ROOT, "tools")):
    if os.path.isdir(_c) and _c not in sys.path:
        sys.path.insert(0, _c)
# 让 <repo>/src 可被导入（公开版把训练器与数据集读取器放在 src/）。
_SRC = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402
from train_seg import UNet, SegDataset  # noqa: E402


def evaluate(ckpt: str, root: str, classes, size: int, device: str,
             thr: float = 0.5, limit: int = 0) -> dict:
    model = UNet(3, len(classes), 32).to(device)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = SegDataset(root, classes, size, augment=False)
    n = min(len(ds), limit) if limit else len(ds)
    if n == 0:
        return {"n": 0}

    dice_sum = np.zeros(len(classes))
    pred_frac = np.zeros(len(classes))
    gt_frac = np.zeros(len(classes))
    with torch.no_grad():
        for i in range(n):
            x, y, _ = ds[i]
            p = torch.sigmoid(model(x[None].to(device)))[0].cpu().numpy()
            pred = p >= thr
            gt = y.numpy() > 0.5
            for c in range(len(classes)):
                inter = float((pred[c] & gt[c]).sum())
                den = float(pred[c].sum() + gt[c].sum())
                dice_sum[c] += 2 * inter / den if den > 0 else 1.0
                pred_frac[c] += pred[c].mean()
                gt_frac[c] += gt[c].mean()
    return {"n": n, "dice": (dice_sum / n).tolist(),
            "pred_frac": (pred_frac / n).tolist(),
            "gt_frac": (gt_frac / n).tolist()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--roots", required=True,
                    help="逗号分隔的多个 tile 根目录，逐个评估")
    ap.add_argument("--classes", default="golgi")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    classes = tuple(s.strip() for s in args.classes.split(",") if s.strip())
    print(f"权重 {args.ckpt}  device={args.device}  阈值 {args.thr}")
    print(f"{'数据源':<34} {'tile':>5} {'Dice':>8} {'GT%':>7} {'预测%':>7}")
    print("-" * 66)
    out = []
    for root in [r.strip() for r in args.roots.split(",") if r.strip()]:
        p = root if os.path.isabs(root) else os.path.join(ROOT, root)
        if not os.path.isdir(p):
            print(f"{root:<34}  目录不存在")
            continue
        r = evaluate(args.ckpt, p, classes, args.size, args.device,
                     args.thr, args.limit)
        if r.get("n", 0) == 0:
            print(f"{root:<34}  没有切片")
            continue
        print(f"{root:<34} {r['n']:>5} {r['dice'][0]:>8.4f} "
              f"{100*r['gt_frac'][0]:>6.2f}% {100*r['pred_frac'][0]:>6.2f}%")
        out.append((root, r))
    print()
    print("注：这是 **tile 级** Dice，与训练日志口径一致；")
    print("    全图像素级 Dice 更低（见 tools/predict_golgi.py 的输出）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
