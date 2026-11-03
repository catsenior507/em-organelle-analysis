#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_masks.py —— 比较两套掩码（如"留出模型" vs "发布版模型"）

## 用途

1. 发布版模型（用全部 88 张训练）在**原来留出的 17 张**上是否预测更完整？
2. 两套掩码的整体差异有多大（面积、区域数）？

## 指标

* 逐图前景面积占比、连通域数
* 两套掩码之间的 Dice（同一张图的两个预测之间的重合度）
* 按 `split` 列分层看差异（heldout 图上的变化最值得看，因为那是发布版
  第一次见到这些图）

## 注意

这里比较的是**两个模型的预测**，不是"预测 vs 真值"。两者互相接近不代表都正确；
真正对真值的评估见 `docs/OURDATA-RESULTS.md` 的留出实验。
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None


def load_index(d: str) -> Dict[str, Dict]:
    p = os.path.join(d, "index.csv")
    if not os.path.exists(p):
        return {}
    return {r["image"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="data/masks_ourdata_clean", help="基准（留出模型）")
    ap.add_argument("--b", default="data/masks_release", help="对比（发布版模型）")
    ap.add_argument("--out", default="logs/MASK_COMPARISON.txt")
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy")
        return 2

    ia = load_index(args.a)
    ib = load_index(args.b)
    names = sorted(set(ia) & set(ib))
    if not names:
        print(f"[错误] 两套掩码没有共同的图像（{args.a} vs {args.b}）")
        return 2

    rows: List[Dict] = []
    held: List[Dict] = []
    for n in names:
        pa = os.path.join(args.a, f"{n}.png")
        pb = os.path.join(args.b, f"{n}.png")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        ma = np.asarray(Image.open(pa)) > 127
        mb = np.asarray(Image.open(pb)) > 127
        if ma.shape != mb.shape:
            continue
        inter = float((ma & mb).sum())
        dice = 2 * inter / (ma.sum() + mb.sum()) if (ma.sum() + mb.sum()) > 0 else float("nan")
        split = ia[n].get("split", "train")
        _, na = ndi.label(ma, structure=np.ones((3, 3)))
        _, nb = ndi.label(mb, structure=np.ones((3, 3)))
        r = {"image": n, "split": split,
             "frac_a": float(ma.mean()), "frac_b": float(mb.mean()),
             "regions_a": int(na), "regions_b": int(nb),
             "dice_ab": dice,
             "frac_delta": float(mb.mean() - ma.mean())}
        rows.append(r)
        if split == "heldout":
            held.append(r)

    def summ(rs: List[Dict], title: str):
        if not rs:
            print(f"\n{title}: 无数据")
            return
        fa = np.array([r["frac_a"] for r in rs])
        fb = np.array([r["frac_b"] for r in rs])
        dd = np.array([r["dice_ab"] for r in rs])
        print(f"\n{title}（n={len(rs)}）")
        print(f"  前景占比  留出模型 {fa.mean()*100:.2f}%  ->  发布版 {fb.mean()*100:.2f}%"
              f"  ({100*(fb.mean()-fa.mean())/max(1e-9,fa.mean()):+.1f}%)")
        print(f"  两者 Dice  均值 {np.nanmean(dd):.4f}  中位 {np.nanmedian(dd):.4f}")
        print(f"  区域数    留出 {np.mean([r['regions_a'] for r in rs]):.1f}"
              f"  ->  发布版 {np.mean([r['regions_b'] for r in rs]):.1f}")
        grew = sum(1 for r in rs if r["frac_delta"] > 0.005)
        shrank = sum(1 for r in rs if r["frac_delta"] < -0.005)
        print(f"  面积变大 {grew} 张 / 变小 {shrank} 张")

    print("=" * 70)
    print("两套掩码对比")
    print("=" * 70)
    print(f"基准 A: {args.a}")
    print(f"对比 B: {args.b}")
    print(f"共同图像 {len(rows)} 张")

    summ(rows, "全部图像")
    summ(held, "仅原来留出的图像（发布版第一次见到它们）")

    if held:
        print("\n留出图像的逐图明细：")
        print(f"{'图像':<10} {'A占比':>7} {'B占比':>7} {'变化':>8} {'两者Dice':>9}")
        for r in sorted(held, key=lambda x: -abs(x["frac_delta"])):
            print(f"{r['image']:<10} {r['frac_a']*100:>6.2f}% {r['frac_b']*100:>6.2f}% "
                  f"{r['frac_delta']*100:>+7.2f}% {r['dice_ab']:>9.4f}")

    print()
    print("说明：这是**两个模型的预测之间**的重合度，不是对真值的准确率。")
    print("      对真值的评估见 docs/OURDATA-RESULTS.md 的留出实验。")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n逐图明细已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
