#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
organelle_counts.py —— 细胞器个体数随神经元分化天的变化（S-BIAD2983）

## 为什么这个值得单独算

接触分析回答的是"谁和谁相邻"，但同一批实例级标注还能回答一个更基础的问题：
**在神经元分化的不同阶段，各种细胞器的个体数怎么变？**

S-BIAD2983 覆盖 day 7 / 14 / 21 / 28 四个分化阶段，每个阶段 79 至 95 个视野，
实例级标注可以直接给出每类细胞器的个体数。这比接触分析便宜得多——
不需要任何零模型，只需数标签。

## 方法

* 扫描 `data/biad2983/` 里已缓存的分割文件；
* 对每个（视野，通道）数非零标签个数；
* 按分化天聚合，报中位数与四分位；
* 同时给出每个视野的细胞面积占比，用于判断"实例数变化"是否只是"视野里细胞变多"。

## 一个必须提防的混杂

不同日期可能来自不同的生物学重复与不同的成像批次，**实例数差异可能反映制样差异
而非分化本身**。因此本文只报描述性统计，并明确标注重复数与视野数，
不做显著性声称。
"""

from __future__ import annotations

import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import tifffile

CHANNELS = ["nucleus", "mitochondria", "golgi", "lysosome",
            "peroxisome", "er", "lipid_droplet"]
# 文件名里用的缩写
ALIAS = {
    "nuc": "nucleus", "mito": "mitochondria", "golgi": "golgi",
    "lyso": "lysosome", "perox": "peroxisome", "er": "er", "LD": "lipid_droplet",
    "cell": "cell",
}


def main(cache="data/biad2983"):
    sys.stdout.reconfigure(encoding="utf-8")
    files = glob.glob(os.path.join(cache, "*.tiff"))
    print(f"缓存文件 {len(files)} 个")

    # 按 stem 分组
    by_stem = defaultdict(dict)
    for f in files:
        base = os.path.basename(f)[:-5]
        if "-" not in base:
            continue
        stem, ch = base.rsplit("-", 1)
        if ch in ALIAS:
            by_stem[stem][ALIAS[ch]] = f
    print(f"视野 {len(by_stem)} 个")

    rows = []
    for stem, chs in by_stem.items():
        m = re.search(r"Day(\d+)", stem)
        day = f"day{m.group(1)}" if m else None
        if day is None:
            continue
        rec = {"stem": stem, "day": day,
               "replicate": (re.search(r"BR(\d+)", stem).group(1)
                             if re.search(r"BR(\d+)", stem) else "?")}
        for ch, path in chs.items():
            lab = tifffile.imread(path)
            ids = np.unique(lab)
            rec[ch] = int((ids != 0).sum())
            if ch == "cell":
                rec["cell_frac"] = float((lab > 0).mean())
        rows.append(rec)

    if not rows:
        print("没有带 Day 的视野")
        return 1

    days = sorted({r["day"] for r in rows}, key=lambda d: int(d[3:]))
    print(f"\n涉及 {len(rows)} 个视野，分化天 {days}")

    print("\n每个分化天的视野数与生物学重复数：")
    for d in days:
        sub = [r for r in rows if r["day"] == d]
        reps = sorted({r["replicate"] for r in sub})
        print(f"  {d:<7}视野 {len(sub):>3}   重复 BR{reps}")

    print("\n各细胞器个体数（中位数，括号内为视野数）：")
    hdr = f"{'细胞器':<16}" + "".join(f"{d:>14}" for d in days)
    print(hdr)
    print("-" * len(hdr))
    for ch in CHANNELS:
        cells = []
        for d in days:
            v = [r[ch] for r in rows if r["day"] == d and ch in r]
            cells.append(f"{np.median(v):>10.0f} ({len(v)})" if v else f"{'-':>14}")
        print(f"{ch:<16}" + "".join(cells))

    print("\n按细胞占位面积归一化后的个体数密度（每 1% 视野面积，中位数）：")
    print("  —— 这一步是必要的：各分化天的细胞占位面积不同（见下表），")
    print("     不归一化会把“细胞变大”误读成“细胞器变多”。")
    hdr2 = f"{'细胞器':<16}" + "".join(f"{d:>14}" for d in days)
    print(hdr2)
    print("-" * len(hdr2))
    for ch in CHANNELS:
        cells = []
        for d in days:
            v = [r[ch] / max(1e-9, r["cell_frac"] * 100)
                 for r in rows if r["day"] == d and ch in r and "cell_frac" in r]
            cells.append(f"{np.median(v):>14.0f}" if v else f"{'-':>14}")
        print(f"{ch:<16}" + "".join(cells))

    print("\n细胞占视野面积比（中位数）：")
    for d in days:
        v = [r["cell_frac"] for r in rows if r["day"] == d and "cell_frac" in r]
        if v:
            print(f"  {d:<7}{100*float(np.median(v)):.1f}%  （n={len(v)}）")

    print("\n注意：不同分化天来自不同成像批次，实例数差异可能反映制样与取样差异，")
    print("      不是分化本身。此处只作描述，不作显著性声称。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
