#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filter_masks.py —— 按**物理面积**过滤模型输出的小碎片

## 为什么必须做

预测掩码里 75.5% 的连通域小于 50 像素，却只占前景面积的 0.1%——
它们是噪声碎片，不是线粒体。不做过滤的话，
"每张图有多少个线粒体"这个数会被碎片严重污染
（实测中位等效直径只有 8.4 nm，而线粒体典型值是 300–1000 nm）。

## 为什么用 µm² 而不是像素

我们的图有三种像素尺寸（1.004 / 2.667 / 5.291 nm/px），
同一像素面积对应的物理面积差 28 倍。用固定像素阈值会让
5.29 nm/px 的图把真实线粒体也滤掉，或让 1 nm/px 的图漏掉噪声。
所以阈值取**物理面积**，默认 0.05 µm²（约等于直径 250 nm 的圆）。
线粒体典型的等效直径是 0.3–1 µm，对应面积 0.07–0.79 µm²，
因此 0.05 µm² 只会滤掉明显过小的碎片。

## 用法

    python tools/filter_masks.py --min-area-um2 0.05 \
        --src data/masks_ourdata --dst data/masks_ourdata_clean
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/masks_ourdata")
    ap.add_argument("--dst", default="data/masks_ourdata_clean")
    ap.add_argument("--min-area-um2", type=float, default=0.05)
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy", file=sys.stderr)
        return 2

    idx_path = os.path.join(args.src, "index.csv")
    if not os.path.exists(idx_path):
        print(f"[错误] 缺 {idx_path}", file=sys.stderr)
        return 2
    rows = list(csv.DictReader(open(idx_path, encoding="utf-8")))
    nm_of = {r["image"]: float(r["nm_per_px"]) for r in rows}

    os.makedirs(args.dst, exist_ok=True)
    out_rows: List[Dict] = []
    tot_before = tot_after = 0
    n_before = n_after = 0

    print(f"过滤阈值 {args.min_area_um2} µm²")
    print(f"{'图像':<10} {'nm/px':>7} {'域数(前)':>9} {'域数(后)':>9} "
          f"{'面积保留':>9} {'最大µm²':>9}")
    print("-" * 62)

    for r in rows:
        name = r["image"]
        nm = nm_of[name]
        nm = nm if nm > 0 else 4.0
        px_um2 = (nm / 1000.0) ** 2
        min_px = args.min_area_um2 / px_um2

        p = os.path.join(args.src, f"{name}.png")
        if not os.path.exists(p):
            continue
        m = np.asarray(Image.open(p)) > 127
        lab, n = ndi.label(m, structure=np.ones((3, 3)))
        if n == 0:
            Image.fromarray((m.astype(np.uint8) * 255)).save(
                os.path.join(args.dst, f"{name}.png"))
            out_rows.append({**r, "n_regions_clean": 0, "pred_frac_clean": 0.0,
                             "max_area_um2_clean": 0.0, "removed_regions": 0})
            continue
        sizes = np.bincount(lab.ravel())
        keep = np.zeros(sizes.size, bool)
        keep[1:] = sizes[1:] >= min_px
        clean = keep[lab]
        removed = int(n - keep[1:].sum())

        Image.fromarray((clean.astype(np.uint8) * 255)).save(
            os.path.join(args.dst, f"{name}.png"))

        cs = sizes[1:][keep[1:]]
        out_rows.append({
            **r,
            "n_regions_clean": int(keep[1:].sum()),
            "pred_frac_clean": float(clean.mean()),
            "max_area_um2_clean": float(cs.max() * px_um2) if cs.size else 0.0,
            "removed_regions": removed})

        tot_before += int(m.sum()); tot_after += int(clean.sum())
        n_before += int(n); n_after += int(keep[1:].sum())
        print(f"{name:<10} {nm:>7.2f} {n:>9} {int(keep[1:].sum()):>9} "
              f"{100.0*clean.sum()/max(1,m.sum()):>8.1f}% "
              f"{out_rows[-1]['max_area_um2_clean']:>9.2f}")

    out_csv = os.path.join(args.dst, "index.csv")
    if out_rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)

    print()
    print("=" * 62)
    print(f"处理 {len(out_rows)} 张")
    print(f"连通域总数：{n_before} -> {n_after}（移除 {n_before-n_after} 个碎片）")
    print(f"前景像素保留：{100.0*tot_after/max(1,tot_before):.2f}%")
    if out_rows:
        nr = np.array([r["n_regions_clean"] for r in out_rows])
        pf = np.array([r["pred_frac_clean"] for r in out_rows])
        print(f"每图线粒体数：均值 {nr.mean():.1f}  中位 {np.median(nr):.0f}  "
              f"范围 [{nr.min()}, {nr.max()}]")
        print(f"线粒体面积占比：均值 {pf.mean()*100:.2f}%  中位 {np.median(pf)*100:.2f}%")
    print(f"清洗后掩码：{args.dst}/")
    print(f"统计表：{out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
