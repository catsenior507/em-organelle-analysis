#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantify_text_contamination.py —— 量化"文字伪影到底有没有污染 GT"

## 为什么先量化再动手

前四种自动检测文字的判据全部失败（见 `detect_text_artifacts.py` 的表格），
原因是这些插图**文字是暗底上的暗字**，而 TEM 组织本身充满暗结构。

在继续投入之前应回答一个更根本的问题：
**文字落在 GT（高尔基体）上的比例有多大？**

* 若文字主要覆盖**背景** -> 它本来就在 GT=0 的地方，模型把文字学成背景
  与真值一致，**不构成污染**，无需屏蔽
* 若文字大量**压在 Golgi 上** -> GT 在那些地方不可信，必须处理

## 判据

把"很暗"的像素（`gray < gthr`）作为文字/深色结构的近似上界：
* `dark_in_gt`：GT 区域内暗像素占 GT 的比例
* `gt_in_dark`：暗像素中属于 GT 的比例
* `dark_frac`：全图暗像素比例

再与"区域内有文字的图"和"没有文字的图"对比，看指标是否差异明显。
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

    print(f"可配对 {len(ids)} 组；用「灰度 < 阈值」近似暗结构（含文字）")
    print()
    header = (f"{'id':<9} {'GT%':>6} " +
              " ".join(f"{'d<'+str(t):>7}" for t in (40, 60, 80)) +
              f" {'GT内暗占比':>11} {'GT可疑度':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    for i in ids:
        rgb = np.asarray(Image.open(os.path.join(B, i + ".jpg")).convert("RGB"))
        g = np.asarray(Image.open(os.path.join(A, i + ".jpg")).convert("L"))
        if rgb.shape[:2] != g.shape:
            continue
        gt = fill(green_mask(rgb))
        if gt.sum() < 200:
            continue
        gt_frac = float(gt.mean())
        # 用阈值 60 作为"暗结构"主判据
        dark = g < 60
        dark_in_gt = float((dark & gt).sum()) / max(1, int(gt.sum()))
        gt_in_dark = float((dark & gt).sum()) / max(1, int(dark.sum()))
        rows.append({"id": i, "gt_frac": gt_frac,
                     "dark40": float((g < 40).mean()),
                     "dark60": float(dark.mean()),
                     "dark80": float((g < 80).mean()),
                     "dark_in_gt": dark_in_gt, "gt_in_dark": gt_in_dark})
        print(f"{i:<9} {100*gt_frac:>5.1f}% "
              f"{100*rows[-1]['dark40']:>6.2f}% {100*rows[-1]['dark60']:>6.2f}% "
              f"{100*rows[-1]['dark80']:>6.2f}% "
              f"{100*dark_in_gt:>10.1f}% {100*gt_in_dark:>8.1f}%")

    if not rows:
        print("\n没有可用样本")
        return 1

    print()
    print("=" * 70)
    dg = np.array([r["dark_in_gt"] for r in rows])
    gd = np.array([r["gt_in_dark"] for r in rows])
    print(f"GT 区域内暗像素占比：均值 {100*dg.mean():.1f}%  "
          f"中位 {100*np.median(dg):.1f}%  范围 [{100*dg.min():.1f}%, {100*dg.max():.1f}%]")
    print(f"暗像素中属于 GT 的比例：均值 {100*gd.mean():.1f}%  "
          f"中位 {100*np.median(gd):.1f}%")
    print()
    print("判读：")
    print("  * 『GT区域内暗像素占比』高，说明 Golgi 本身就有很多暗像素")
    print("    （高尔基体池层在 TEM 里是深色的）——这**不代表**文字污染。")
    print("  * 真正要看的是『暗像素中属于 GT 的比例』：若很低，")
    print("    说明绝大部分暗区（含文字）落在背景上，对 GT 无影响。")
    out = os.path.join(ROOT, "logs", "text_contamination.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    print(f"\n已写出 {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
