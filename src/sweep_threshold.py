#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_threshold.py —— 在多个决策阈值上重算分割指标

## 为什么要查这个

所有分割实验的 er Dice 都远低于"全预测前景"的平凡基线（0.2556），
而 mito 也偏低。这种**两类同时欠预测**的模式不像"任务难"，更像是
**输出概率被系统性压低、而决策阈值 0.5 过高**。

如果降低阈值能大幅提升 Dice，说明模型其实学到了东西，
只是概率校准偏了——那么之前"ER 学不了"的结论就是被一个阈值假象误导的。

## 做法

加载同一个检查点，在阈值 0.1/0.2/0.3/0.4/0.5/0.6/0.7 上各算一次逐类 Dice，
并同时报出各类的预测像素数与真值像素数之比（`pred/gt`）。
`pred/gt` 远小于 1 就是欠预测的直接证据。

## 用法

    python sweep_threshold.py --ckpt runs/seg_indomain_200e/best.pt \
        --root data/slices --split --classes mito,er --size 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "detr"))

from train_seg import SegDataset, UNet, split_items_by_crop  # noqa: E402


@torch.no_grad()
def evaluate_at(model, dl, dev, n_classes: int, thr: float
                ) -> Dict[str, Dict[str, float]]:
    inter = np.zeros(n_classes)
    pred_sum = np.zeros(n_classes)
    gt_sum = np.zeros(n_classes)
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    for x, y, _ in dl:
        x = x.to(dev)
        prob = torch.sigmoid(model(x)).cpu().numpy()
        gt = (y.numpy() > 0.5)
        pred = prob > thr
        for ci in range(n_classes):
            p, g = pred[:, ci], gt[:, ci]
            inter[ci] += np.logical_and(p, g).sum()
            pred_sum[ci] += p.sum()
            gt_sum[ci] += g.sum()
            tp[ci] += np.logical_and(p, g).sum()
            fp[ci] += np.logical_and(p, ~g).sum()
            fn[ci] += np.logical_and(~p, g).sum()
    out = {}
    for ci in range(n_classes):
        out[str(ci)] = {
            "dice": float(2 * inter[ci] / max(1e-9, pred_sum[ci] + gt_sum[ci])),
            "iou": float(inter[ci] / max(1e-9, pred_sum[ci] + gt_sum[ci] - inter[ci])),
            "precision": float(tp[ci] / max(1e-9, tp[ci] + fp[ci])),
            "recall": float(tp[ci] / max(1e-9, tp[ci] + fn[ci])),
            "pred_px": float(pred_sum[ci]),
            "gt_px": float(gt_sum[ci]),
            "pred_over_gt": float(pred_sum[ci] / max(1e-9, gt_sum[ci])),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/slices")
    ap.add_argument("--classes", default="mito,er")
    ap.add_argument("--size", type=int, default=256, help="必须与训练时一致")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    classes = tuple(s.strip() for s in args.classes.split(",") if s.strip())
    ds = SegDataset(args.root, classes, args.size, augment=False)
    tr, va, vc = split_items_by_crop(ds.items, args.val_frac, args.seed)
    ds.items = va
    print(f"验证切片 {len(va)} 个（crop {vc}）")
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    model = UNet(3, len(classes), args.base)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    print(f"检查点 epoch {ck.get('epoch')}  设备 {dev}")

    ths = [float(t) for t in args.thresholds.split(",")]
    rows = {}
    print()
    header = f"{'thr':>6} "
    for c in classes:
        header += f"{c+' dice':>11} {c+' P':>8} {c+' R':>8} {c+' pred/gt':>11} "
    print(header)
    print("-" * len(header))
    for t in ths:
        r = evaluate_at(model, dl, dev, len(classes), t)
        rows[t] = r
        line = f"{t:>6.2f} "
        for ci in range(len(classes)):
            v = r[str(ci)]
            line += (f"{v['dice']:>11.4f} {v['precision']:>8.3f} "
                     f"{v['recall']:>8.3f} {v['pred_over_gt']:>11.3f} ")
        print(line)

    # 平凡基线（与阈值无关）
    r0 = rows[ths[0]]
    print()
    print("全预测前景的平凡基线（与阈值无关）：")
    for ci in range(len(classes)):
        g = r0[str(ci)]["gt_px"]
        tot = g / max(1e-9, r0[str(ci)]["gt_px"]) # placeholder
    # 用 gt_frac 反推：dice_all_fg = 2p/(1+p)，其中 p=gt_frac
    # 这里直接由 gt_px 与总像素估：需要总像素，用 pred/gt 无法得到，改由数据集算
    n_px = len(va) * args.size * args.size
    for ci in range(len(classes)):
        p = r0[str(ci)]["gt_px"] / n_px
        base = 2 * p / (1 + p) if p > 0 else 0.0
        print(f"  类{ci}（{classes[ci]}）占比 {p:.4f} -> 全前景 Dice {base:.4f}")
        rows.setdefault("_baseline", {})[str(ci)] = base

    # 找每类的最优阈值
    print()
    print("每类的最优阈值：")
    for ci in range(len(classes)):
        best = max(ths, key=lambda t: rows[t][str(ci)]["dice"])
        bd = rows[best][str(ci)]["dice"]
        d50 = rows[ths[min(range(len(ths)), key=lambda i: abs(ths[i]-0.5))]][str(ci)]["dice"]
        base = rows["_baseline"][str(ci)]
        print(f"  {classes[ci]}: 最优 thr={best} dice={bd:.4f} "
              f"（thr=0.5 时 {d50:.4f}，提升 {bd-d50:+.4f}）"
              f"  平凡基线 {base:.4f}  超过基线={bd > base}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"ckpt": args.ckpt, "rows": {str(k): v for k, v in rows.items()},
                       "classes": list(classes)}, fh, ensure_ascii=False, indent=1)
        print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
