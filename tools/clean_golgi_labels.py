#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_golgi_labels.py —— 清洗高尔基体标注：去掉碎片、记录粘连

## 依据

`tools/golgi_scale_analysis.py` 发现：Golgi 连通域的等效直径中间 50% 只差
1.8 倍（190–346 px，尺度其实一致），但全范围有 107 倍，两个极端分别来自

* **12.7 px 量级** —— 标注碎片，很可能不是完整高尔基体
* **1354 px 量级** —— 多个相邻 Golgi 被填洞后连成一片

也就是说瓶颈更可能是**标注质量**而不是成像尺度。既然如此，先做最直接的一步：
**把小碎片去掉**，看 Dice 是否变化。

## 处理规则

| 规则 | 阈值 | 理由 |
|---|---|---|
| 去掉小连通域 | 面积 < `--min-area`（默认 500 px） | 12.7 px 直径约合 127 面积，低于此值的几乎不可能是完整 Golgi |
| 保留大连通域 | 不设上限 | 大块可能是真实的大 Golgi 区域，贸然拆分会引入新错误 |
| 填洞 | 始终 | 绿色是描边，必须填成实心 |

> **不自动拆分粘连区域**：拆分需要形状先验，容易引入比原问题更大的偏差。
> 这一步只做"减法"（去碎片），保守且可解释。

## 输出

    data/golgi/标注清洗/<id>.png     二值掩码（0/255）

可直接作为统一层的 `binary_mask` 类型使用。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE8 = np.ones((3, 3), bool)


def green_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (g > 110) & (g > r + 20) & (g > b + 30)


def fill(o: np.ndarray) -> np.ndarray:
    if not o.any():
        return o
    c = ndi.binary_closing(o, structure=np.ones((3, 3)), iterations=2)
    f = ndi.binary_fill_holes(c)
    return c if f.mean() > 0.6 else f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", default="data/golgi/标注")
    ap.add_argument("--out", default="data/golgi/标注清洗")
    ap.add_argument("--min-area", type=int, default=500)
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy")
        return 2
    src = os.path.join(ROOT, args.annot)
    dst = os.path.join(ROOT, args.out)
    os.makedirs(dst, exist_ok=True)

    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(src)
                 if f.lower().endswith(".jpg"))
    print(f"阈值：去掉面积 < {args.min_area} px 的连通域")
    print()
    print(f"{'id':<9} {'域数':>4} {'去前面积':>10} {'去后面积':>10} {'减少':>7}")
    print("-" * 46)

    rows = []
    for i in ids:
        rgb = np.asarray(Image.open(os.path.join(src, i + ".jpg")).convert("RGB"))
        gt = fill(green_mask(rgb))
        if gt.sum() == 0:
            continue
        lab, n = ndi.label(gt, structure=FACE8)
        sizes = np.bincount(lab.ravel())
        keep = np.zeros(sizes.size, bool)
        keep[1:] = sizes[1:] >= args.min_area
        cleaned = keep[lab]
        n_kept = int(keep.sum())
        before, after = int(gt.sum()), int(cleaned.sum())
        drop = 1.0 - after / max(1, before)
        rows.append({"id": i, "n_regions": n, "n_kept": n_kept,
                     "before": before, "after": after, "drop": drop})
        print(f"{i:<9} {n_kept:>4} {before:>10} {after:>10} {100*drop:>6.1f}%")
        Image.fromarray((cleaned * 255).astype(np.uint8)).save(
            os.path.join(dst, f"{i}.png"))

    if rows:
        tot_b = sum(r["before"] for r in rows)
        tot_a = sum(r["after"] for r in rows)
        print()
        print("=" * 60)
        print(f"共 {len(rows)} 张；总面积 {tot_b} -> {tot_a} "
              f"（减少 {100*(1-tot_a/max(1,tot_b)):.2f}%）")
        nr_b = sum(r["n_regions"] for r in rows)
        nr_a = sum(r["n_kept"] for r in rows)
        print(f"连通域总数 {nr_b} -> {nr_a}（去掉 {nr_b-nr_a} 个碎片）")
        zero = [r["id"] for r in rows if r["after"] == 0]
        if zero:
            print(f"注意：清洗后为空的图 {len(zero)} 张：{zero[:8]}")
        out = os.path.join(ROOT, "logs", "golgi_label_cleaning.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"min_area": args.min_area, "rows": rows}, fh,
                      ensure_ascii=False, indent=1)
        print(f"\n已写出 {os.path.relpath(out, ROOT)}")
        print(f"掩码目录 {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
