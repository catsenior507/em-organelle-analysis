#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_text_artifacts.py —— 标定"黑色说明文字"的可分性

## 为什么需要先标定

高尔基体那批原图是**已发表论文的插图**，上面烘进了黑色说明文字与箭头
（"Golgi Cisternae"、"Mitochondria"、引线等）。这些像素覆盖了真实组织，
导致该处 GT 不可信。要在训练中屏蔽它们，首先得能**可靠地区分
"黑色文字"与"深色细胞器"**。

两者的差别（待验证）：
* 文字：接近纯黑（0–40），笔画细（几像素宽）、内部灰度平坦
* 细胞器（线粒体等）：深灰（30–90），有内部纹理、尺寸更大、形状成团

本脚本统计各阈值下的连通域尺寸分布与内部灰度方差，
用数据判断阈值取多少合适，而不是拍脑袋。
"""

from __future__ import annotations

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
SRC = os.path.join(ROOT, "data", "golgi", "原图")


def main() -> int:
    if ndi is None:
        print("[错误] 需要 scipy")
        return 2
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(SRC)
                 if f.lower().endswith(".jpg"))
    print(f"原图 {len(ids)} 张")
    print()
    print("各阈值下的前景统计（阈值越低越只抓纯黑）：")
    print(f"{'阈值':>5} {'前景%':>7} {'域数':>6} {'中位面积':>9} "
          f"{'细长域占比':>11} {'域内方差中位':>13}")
    print("-" * 62)

    rows = []
    for thr in (20, 30, 40, 50, 60):
        fracs, counts, med_areas, thin_ratios, var_meds = [], [], [], [], []
        for i in ids[:20]:
            g = np.asarray(Image.open(os.path.join(SRC, i + ".jpg"))
                           .convert("L"), dtype=np.uint8)
            m = g < thr
            fracs.append(m.mean())
            lab, n = ndi.label(m, structure=np.ones((3, 3)))
            counts.append(n)
            if n == 0:
                continue
            sizes = np.bincount(lab.ravel())[1:]
            med_areas.append(float(np.median(sizes)))
            # 细长度：用 bbox 的长宽比近似
            thin = 0
            variances = []
            for rid in range(1, n + 1):
                sel = (lab == rid)
                if sel.sum() < 20:
                    continue
                ys, xs = np.nonzero(sel)
                h = ys.max() - ys.min() + 1
                w = xs.max() - xs.min() + 1
                ar = max(h, w) / max(1, min(h, w))
                if ar > 4:
                    thin += 1
                variances.append(float(g[sel].std()))
            thin_ratios.append(thin / max(1, n))
            if variances:
                var_meds.append(float(np.median(variances)))
        rows.append({"thr": thr, "frac": float(np.mean(fracs)),
                     "n": float(np.mean(counts)),
                     "med_area": float(np.median(med_areas)) if med_areas else 0,
                     "thin": float(np.mean(thin_ratios)) if thin_ratios else 0,
                     "var": float(np.median(var_meds)) if var_meds else 0})
        r = rows[-1]
        print(f"{thr:>5} {100*r['frac']:>6.2f}% {r['n']:>6.0f} "
              f"{r['med_area']:>9.0f} {100*r['thin']:>10.1f}% "
              f"{r['var']:>13.1f}")

    print()
    print("判读：")
    print("  * 文字笔画细、灰度平坦 -> 域内方差小；细胞器有纹理 -> 方差大")
    print("  * 阈值越低，越可能只抓到纯黑文字，但会漏掉抗锯齿边缘")
    print("  * 若某阈值下细长域占比高且域内方差低，说明该阈值偏向文字")
    out = os.path.join(ROOT, "logs", "text_artifact_calibration.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    print(f"\n已写出 {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
