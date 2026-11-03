#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_golgi_tiles.py —— 把高尔基体原图/标注图对切成训练 tile

## 数据情况（已核实）

* `data/golgi/原图/<id>.jpg`     64 张灰度 TEM（JPEG，RGB 但 R=G=B）
* `data/golgi/标注/<id>.jpg`     53 张，= 原图 + **绿色轮廓** + 黄色 "G" 文字
* 可配对 53 组（11 张原图没有标注）

**核实过的两点**：
1. 原图里绿色像素占 0%——绿色纯粹是标注叠加，不是图像内容
2. 原图与标注图的暗像素占比几乎相同（21.37% vs 21.30%），
   说明黑色的说明文字与箭头**在原图里就已存在**，属于图源本身

## 由此带来的两条处理约定

1. **训练输入用原图**（不含绿色），GT 从标注图提取
2. 黑色文字无法去除，会作为背景进入训练。这是这批数据固有的噪声，
   需在报告里写明：GT 在文字覆盖处不可信

## GT 提取

绿色轮廓 → 形态学闭运算补齐断口 → `binary_fill_holes` 填成实心区域。
与线粒体那批数据完全一致的逻辑，便于复用同一套评估。

## 输出

    <out>/golgi/<id>/z####/{em.png, golgi.png}

`<id>` 作为 crop 目录，因此 `train_seg.py --val-root split` 是**按图像划分**。
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None


def green_mask(rgb: np.ndarray) -> np.ndarray:
    """提取绿色描边。标注绿约为 (132,175,96)，与背景灰度差异明显。"""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (g > 110) & (g > r + 20) & (g > b + 30)


def fill_outline(o: np.ndarray) -> np.ndarray:
    """闭运算补断口后填洞。若填出来超过 60% 说明描边不闭合，退回闭运算结果。"""
    if not o.any():
        return o
    c = ndi.binary_closing(o, structure=np.ones((3, 3)), iterations=2)
    f = ndi.binary_fill_holes(c)
    return c if f.mean() > 0.6 else f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="data/golgi/原图")
    ap.add_argument("--annot", default="data/golgi/标注")
    ap.add_argument("--out", default="data/golgi_tiles")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--min-pos-frac", type=float, default=0.01,
                    help="tile 内 GT 占比超过此值算正样本")
    ap.add_argument("--neg-ratio", type=float, default=1.0,
                    help="每张图保留的负 tile 数 = 正 tile 数 × 此比例")
    ap.add_argument("--val-every", type=int, default=5,
                    help="按排序后每隔 N 张取一张做验证（确定性划分）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy")
        return 2

    ids = sorted({os.path.splitext(f)[0] for f in os.listdir(args.orig)}
                 & {os.path.splitext(f)[0] for f in os.listdir(args.annot)})
    if not ids:
        print(f"[错误] {args.orig} 与 {args.annot} 没有同名文件可配对")
        return 2

    # 按排序隔 N 取一作为验证，保证可复现且与随机种子无关
    val_ids = {i for k, i in enumerate(ids) if k % args.val_every == 0}
    print(f"可配对 {len(ids)} 组；验证 {len(val_ids)} 张，训练 {len(ids)-len(val_ids)} 张")

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    rng = np.random.RandomState(args.seed)

    T, S = args.tile, args.stride
    stats: Dict[str, int] = {"pos": 0, "neg": 0, "imgs": 0}
    print()
    print(f"{'id':<9} {'子集':>5} {'尺寸':>12} {'GT%':>6} {'正':>4} {'负':>4}")
    print("-" * 50)

    for i in ids:
        sub = "val" if i in val_ids else "train"
        rgb = np.asarray(Image.open(os.path.join(args.annot, i + ".jpg")).convert("RGB"))
        em = np.asarray(Image.open(os.path.join(args.orig, i + ".jpg")).convert("L"))
        if rgb.shape[:2] != em.shape:
            print(f"{i:<9} [跳过] 尺寸不一致 {rgb.shape[:2]} vs {em.shape}")
            continue
        gt = fill_outline(green_mask(rgb))
        if gt.sum() < 200:
            print(f"{i:<9} [跳过] GT 过小（{int(gt.sum())} px）")
            continue

        h, w = em.shape
        ys = list(range(0, max(1, h - T + 1), S)) or [0]
        xs = list(range(0, max(1, w - T + 1), S)) or [0]
        if ys[-1] != max(0, h - T):
            ys.append(max(0, h - T))
        if xs[-1] != max(0, w - T):
            xs.append(max(0, w - T))

        pos, neg = [], []
        for y0 in ys:
            for x0 in xs:
                g = gt[y0:y0 + T, x0:x0 + T]
                if g.shape != (T, T):
                    continue
                (pos if g.mean() >= args.min_pos_frac else neg).append((y0, x0))
        k = int(round(len(pos) * args.neg_ratio))
        keep_neg = ([neg[j] for j in rng.choice(len(neg), min(k, len(neg)), replace=False)]
                    if neg and k > 0 else [])

        crop_dir = os.path.join(args.out, "golgi", i)
        idx = 0
        for y0, x0 in pos + keep_neg:
            is_pos = (y0, x0) in set(pos)
            d = os.path.join(crop_dir, f"z{idx:04d}")
            os.makedirs(d, exist_ok=True)
            Image.fromarray(em[y0:y0 + T, x0:x0 + T]).save(os.path.join(d, "em.png"))
            Image.fromarray((gt[y0:y0 + T, x0:x0 + T].astype(np.uint8) * 255)).save(
                os.path.join(d, "golgi.png"))
            idx += 1
            stats["pos" if is_pos else "neg"] += 1
        stats["imgs"] += 1
        print(f"{i:<9} {sub:>5} {w}x{h:>7} {100*gt.mean():>5.1f}% "
              f"{len(pos):>4} {len(keep_neg):>4}")

    print()
    print("=" * 50)
    print(f"图像 {stats['imgs']} 张；正 tile {stats['pos']}，负 tile {stats['neg']}，"
          f"合计 {stats['pos']+stats['neg']}")
    print(f"输出 {args.out}/golgi/<id>/z####/{{em.png,golgi.png}}")
    print()
    print("注意：黑色的说明文字与箭头在原图中已存在，无法去除；")
    print("      文字覆盖处 GT 不可信，训练与评估都有此噪声。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
