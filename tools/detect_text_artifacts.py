#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_text_artifacts.py —— 检测烘进 EM 插图里的说明文字与箭头

## 为什么不能用简单的图像统计

试过三种判据，都失败，记录下来避免重复：

| 判据 | 结果 | 失败原因 |
|---|---|---|
| 灰度 < 阈值 | 阈值 20 时前景 1.15%、中位连通域 **2 像素** | JPEG 把笔画打碎成散点，抓不全也连不起来 |
| 黑帽变换（细暗结构） | **屏蔽 44%** | EM 里全是细暗的**膜结构**，与文字在"细暗"这一维上不可分 |
| 暗 + 局部方差低 | 只 0.04–0.5% | 判据方向反了：**线粒体内部又暗又平**，文字笔画细反而方差高 |
| 暗 + 局部方差高 + 黑帽 | **47–58%** | 膜同样满足，仍然分不开 |

结论：**单像素级的纹理统计无法区分"画上去的文字"与"拍摄到的膜"**。

## 能用的判据：空间聚簇

文字有一个膜不具备的性质——**一个词由许多小笔画密集排列**，
而膜是长而连通的曲线。于是：

1. 取暗像素（`g < gthr`）的连通域
2. 只保留**面积适中**（`amin..amax`）且**不极度细长**（长宽比 ≤ `max_aspect`）的
   —— 排除长膜、也排除细碎噪点
3. 对每个候选，统计其**邻域半径内**的同类候选数；
   邻居数 ≥ `min_neighbors` 的才保留（即"成串出现"）
4. 膨胀若干像素覆盖抗锯齿边缘

孤立的暗斑（单个囊泡、单条膜断片）邻居少，会被排除；
成词出现的字母邻居多，会被保留。

## 局限（必须与结果一起读）

这是**高精度、低召回**的检测器：它抓得住成串的文字，但：
* 单个字母/孤立的箭头可能抓不到
* 图片角落的大号标题可能被 `amax` 排除
* 因此**不能声称"已完整屏蔽所有文字"**，只能说"屏蔽了可被可靠识别的那部分"

对训练而言这个取舍是合理的：宁可少屏蔽（保留少量噪声监督），
也不要把大片真实组织错误地排除出损失。
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


def detect(gray: np.ndarray, gthr: int = 100, amin: int = 5, amax: int = 800,
           max_aspect: float = 12.0, radius: int = 120,
           min_neighbors: int = 4, dilate: int = 2) -> np.ndarray:
    """返回需要屏蔽的布尔掩码（True = 该像素不可信，应从损失与统计中排除）。"""
    dark = gray < gthr
    if not dark.any():
        return np.zeros_like(dark)
    lab, n = ndi.label(dark, structure=FACE8)
    if n == 0:
        return np.zeros_like(dark)

    sizes = np.bincount(lab.ravel())
    objs = ndi.find_objects(lab)

    cand: list = []
    for rid in range(1, n + 1):
        a = int(sizes[rid])
        if a < amin or a > amax:
            continue
        sl = objs[rid - 1]
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if max(h, w) / max(1, min(h, w)) > max_aspect:
            continue
        cand.append(rid)
    if not cand:
        return np.zeros_like(dark)

    # 注意：center_of_mass 传入单个标签时返回 (y, x) 元组而不是列表，
    # 直接解包会 "too many values to unpack"。这里统一成 (N,2) 数组。
    com = np.asarray(ndi.center_of_mass(dark, lab, cand), dtype=np.float64)
    com = com.reshape(-1, 2)
    cy, cx = com[:, 0], com[:, 1]

    keep = np.zeros(len(cand), bool)
    r2 = float(radius) ** 2
    for k in range(len(cand)):
        d2 = (cy - cy[k]) ** 2 + (cx - cx[k]) ** 2
        if int((d2 <= r2).sum()) - 1 >= min_neighbors:
            keep[k] = True
    if not keep.any():
        return np.zeros_like(dark)

    sel = np.isin(lab, np.asarray(cand)[keep])
    if dilate > 0:
        sel = ndi.binary_dilation(sel, iterations=dilate)
    return sel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/golgi/原图")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gthr", type=int, default=100)
    ap.add_argument("--amin", type=int, default=5)
    ap.add_argument("--amax", type=int, default=800)
    ap.add_argument("--radius", type=int, default=120)
    ap.add_argument("--min-neighbors", type=int, default=4)
    ap.add_argument("--dilate", type=int, default=2)
    ap.add_argument("--show", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy")
        return 2

    src = os.path.join(ROOT, args.src)
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(src)
                 if f.lower().endswith((".jpg", ".tif", ".png")))
    if args.limit:
        ids = ids[: args.limit]

    rows = []
    print(f"{'id':<9} {'屏蔽%':>7} {'域数':>6}")
    print("-" * 28)
    for i in ids:
        p = os.path.join(src, i + os.path.splitext(
            [f for f in os.listdir(src) if os.path.splitext(f)[0] == i][0])[1])
        g = np.asarray(Image.open(p).convert("L"))
        m = detect(g, args.gthr, args.amin, args.amax,
                   radius=args.radius, min_neighbors=args.min_neighbors,
                   dilate=args.dilate)
        _, n = ndi.label(m, structure=FACE8)
        rows.append({"id": i, "frac": float(m.mean()), "n": int(n),
                     "shape": list(g.shape)})
        print(f"{i:<9} {100*m.mean():>6.2f}% {n:>6}")
        if args.show:
            outdir = os.path.join(ROOT, args.show)
            os.makedirs(outdir, exist_ok=True)
            rgb = np.stack([g] * 3, -1)
            rgb = (rgb * 0.5).astype(np.uint8)
            rgb[m] = [255, 0, 0]
            im = Image.fromarray(rgb)
            im.thumbnail((900, 900), Image.LANCZOS)
            im.save(os.path.join(outdir, f"{i}.png"))

    fracs = [r["frac"] for r in rows]
    print()
    print(f"共 {len(rows)} 张；屏蔽占比 均值 {100*np.mean(fracs):.2f}%  "
          f"中位 {100*np.median(fracs):.2f}%  "
          f"范围 [{100*min(fracs):.2f}%, {100*max(fracs):.2f}%]")
    print("注意：这是**高精度低召回**的检测，只抓成串出现的文字；")
    print("      单个字母或孤立箭头可能漏检，故不能声称已屏蔽全部文字。")
    if args.json:
        out = os.path.join(ROOT, args.json)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"params": vars(args), "rows": rows}, fh,
                      ensure_ascii=False, indent=1)
        print(f"已写出 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
