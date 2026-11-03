#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
golgi_scale_analysis.py —— 分析高尔基体标注的**相对尺度**是否一致

## 为什么要查这个

这批图的像素尺寸跨度极大（885×1273 到 6000×6192）。如果它们来自不同成像
条件、不同放大倍数，那么同一个 Golgi 结构在不同图里占的像素数会差好几倍——
模型就在学一个"尺度不固定"的目标，这比数据量少更难克服。

我们没有可靠的 nm/px 标定，但可以**用结构本身的尺寸作为相对标尺**：
高尔基体的池层堆叠长度在生物学上大致是稳定的（约 1–2 µm），
因此"每个 cistemae 区域的等效直径（像素）"能反映成像尺度。

## 判据

* 若各图的等效直径分布**集中** -> 尺度一致，问题不在尺度
* 若**相差数倍** -> 尺度不一致，值得做尺度归一化再训练

## 注意

这只是**相对**尺度推断，不是绝对标定。输出的是"像素尺度因子"，
可用于把各图重采样到统一尺度，但不能声称这是 nm/px。
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


def main() -> int:
    if ndi is None:
        print("[错误] 需要 scipy")
        return 2
    A = os.path.join(ROOT, "data", "golgi", "原图")
    B = os.path.join(ROOT, "data", "golgi", "标注")
    ids = sorted({os.path.splitext(f)[0] for f in os.listdir(A)}
                 & {os.path.splitext(f)[0] for f in os.listdir(B)})

    print("用 Golgi 连通域的等效直径（像素）作为相对尺度指标")
    print("（生物学上池层堆叠约 1–2 µm，故直径应大致稳定）")
    print()
    print(f"{'id':<9} {'尺寸':>13} {'域数':>4} {'面积中位':>9} "
          f"{'等效直径中位':>12}")
    print("-" * 56)

    rows = []
    for i in ids:
        rgb = np.asarray(Image.open(os.path.join(B, i + ".jpg")).convert("RGB"))
        gt = fill(green_mask(rgb))
        if gt.sum() < 200:
            continue
        lab, n = ndi.label(gt, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())[1:]
        sizes = sizes[sizes >= 100]
        if sizes.size == 0:
            continue
        area_med = float(np.median(sizes))
        diam = 2.0 * np.sqrt(area_med / np.pi)
        rows.append({"id": i, "shape": [rgb.shape[1], rgb.shape[0]],
                     "n_regions": int(sizes.size),
                     "area_median": area_med, "eq_diam": float(diam)})
        print(f"{i:<9} {rgb.shape[1]}x{rgb.shape[0]:<7} {sizes.size:>4} "
              f"{area_med:>9.0f} {diam:>12.1f}")

    if not rows:
        print("\n没有可用样本")
        return 1

    d = np.array([r["eq_diam"] for r in rows])
    a = np.array([r["area_median"] for r in rows])
    print()
    print("=" * 60)
    print(f"等效直径（像素）：中位 {np.median(d):.1f}  "
          f"范围 [{d.min():.1f}, {d.max():.1f}]")
    print(f"  最大/最小 = {d.max()/max(d.min(),1e-6):.2f} 倍")
    print(f"  四分位距 {np.percentile(d,25):.1f} – {np.percentile(d,75):.1f}")
    print(f"面积中位：中位 {np.median(a):.0f} px，"
          f"最大/最小 = {a.max()/max(a.min(),1e-6):.1f} 倍")
    print()
    ratio = d.max() / max(d.min(), 1e-6)
    if ratio < 2.0:
        print("判读：尺度差异 < 2 倍 -> 基本一致，**尺度不是瓶颈**")
    elif ratio < 4.0:
        print("判读：尺度差异 2–4 倍 -> 存在中等差异，尺度归一化可能有用")
    else:
        print("判读：尺度差异 > 4 倍 -> **尺度不一致很可能是主要障碍**，")
        print("      应先把各图重采样到统一尺度再训练")
    print()
    print("提醒：这是相对尺度，不是绝对 nm/px 标定。")

    out = os.path.join(ROOT, "logs", "golgi_scale.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows,
                   "eq_diam_median": float(np.median(d)),
                   "max_over_min": float(ratio),
                   "note": "相对尺度指标，非绝对标定"}, fh,
                  ensure_ascii=False, indent=1)
    print(f"\n已写出 {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
