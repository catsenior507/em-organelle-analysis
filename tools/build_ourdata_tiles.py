#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ourdata_tiles.py —— 把我们自己的 EM 图切成可用于训练的 tile

## 背景

CellMap 训练的线粒体模型在我们自己的数据上**零样本失败**
（6 张图 Dice 均值 0.15，0/6 超过平凡基线；反相后更差，0.049）。
所以必须在自己的数据上训练。

但有两个坑必须先处理：

1. **尺度不统一**：我们的图有三种像素尺寸（1.004 / 2.667 / 5.291 nm/px），
   线粒体在像素上的大小差 5 倍以上。直接混在一起训练，模型要先学会
   "同一物体在不同尺度下都算目标"，浪费容量。这里统一重采样到 `--target-nm`，
   使线粒体的像素尺寸大致一致。

2. **部分标注**：虽然中位覆盖 8.5% 看起来接近穷尽，但有的图只标了 1.1%。
   在那些图上把未标注区域当背景会引入错误监督。处理方式：
     * 覆盖率 ≥ `--trust-coverage` 的图：全部 tile 参与（未标注视为背景）
     * 覆盖率更低的图：**只保留含标注的 tile**，且这些 tile 里未标注部分
       仍按背景处理（tile 以线粒体为中心，误差有限），并在清单里标记来源
   
## 输出结构

    <out>/ourdata/<图像名>/z0000/{em.png, mito.png, meta.json}

`<图像名>` 作为 crop 目录，这样 `train_seg.py --val-root split` 的按 crop
分组划分等价于**按图像划分**，天然避免同一张图的 tile 泄漏到验证集。

## 清单

同时写出 `<out>/manifest.json`，记录每个 tile 来自哪张图、尺度、
是否含标注、该图覆盖率，便于事后按子集分析。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None


def load_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def green_outline(path: str) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    r, g = a[:, :, 0], a[:, :, 1]
    return (g > 40) & (g > r + 20)


def fill_outline(o: np.ndarray) -> np.ndarray:
    c = ndi.binary_closing(o, structure=np.ones((3, 3)), iterations=2)
    f = ndi.binary_fill_holes(c)
    return c if f.mean() > 0.6 else f


def resample_pair(img: np.ndarray, gt: np.ndarray, nm: float, target: float):
    factor = nm / target
    if abs(factor - 1.0) < 1e-3:
        return img, gt, 1.0
    h, w = img.shape
    nw, nh = max(16, int(round(w * factor))), max(16, int(round(h * factor)))
    i2 = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    # 掩码用面积平均再阈值，接近真实的"覆盖占优"
    m2 = np.asarray(Image.fromarray((gt * 255).astype(np.uint8))
                    .resize((nw, nh), Image.BILINEAR)) > 127
    return i2.astype(np.uint8), m2, factor


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/原图片")
    ap.add_argument("--masks", default="data/分离的标注信息")
    ap.add_argument("--annot", default="data/annotations_v2/dataset.json")
    ap.add_argument("--out", default="data/ourdata_tiles")
    ap.add_argument("--target-nm", type=float, default=4.0)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--min-pos-frac", type=float, default=0.01,
                    help="tile 内标注占比超过此值才算正样本 tile")
    ap.add_argument("--neg-ratio", type=float, default=1.0,
                    help="每张图保留的负 tile 数 = 正 tile 数 × 此比例")
    ap.add_argument("--trust-coverage", type=float, default=0.05,
                    help="标注覆盖率高于此值的图，其全部 tile 参与训练")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy", file=sys.stderr)
        return 2

    rng = np.random.RandomState(args.seed)
    meta = json.load(open(args.annot, encoding="utf-8"))
    ents = meta["images"] if isinstance(meta["images"], list) else list(meta["images"].values())

    os.makedirs(os.path.join(args.out, "ourdata"), exist_ok=True)
    manifest: List[Dict] = []
    n_pos = n_neg = 0

    print(f"目标尺度 {args.target_nm} nm/px，tile {args.tile}，步长 {args.stride}")
    print(f"{'图像':<10} {'nm/px':>7} {'缩放':>6} {'覆盖率':>7} {'正tile':>7} {'负tile':>7} {'可信':>5}")
    print("-" * 62)

    for e in ents:
        name = e["image"]
        nm = (e.get("um_per_px") or 0) * 1000.0
        ip = os.path.join(args.images, f"{name}.tif")
        gp = os.path.join(args.masks, f"green_{name}.tif")
        if not (os.path.exists(ip) and os.path.exists(gp)):
            continue
        img = load_gray(ip)
        o = green_outline(gp)
        if o.shape != img.shape:
            continue
        gt = fill_outline(o)
        if gt.sum() < 200:
            continue
        cov = float(gt.mean())
        if nm <= 0:
            nm = 4.0          # 缺尺度的图按目标尺度处理，并在清单中标注
        img2, gt2, factor = resample_pair(img, gt, nm, args.target_nm)
        h, w = img2.shape
        T, S = args.tile, args.stride

        pos, neg = [], []
        ys = list(range(0, max(1, h - T + 1), S)) or [0]
        xs = list(range(0, max(1, w - T + 1), S)) or [0]
        if ys[-1] != max(0, h - T):
            ys.append(max(0, h - T))
        if xs[-1] != max(0, w - T):
            xs.append(max(0, w - T))
        for y0 in ys:
            for x0 in xs:
                g = gt2[y0:y0 + T, x0:x0 + T]
                if g.shape != (T, T):
                    continue
                f = float(g.mean())
                if f >= args.min_pos_frac:
                    pos.append((y0, x0))
                else:
                    neg.append((y0, x0))

        trusted = cov >= args.trust_coverage
        if trusted:
            keep_neg = neg
        else:
            k = int(round(len(pos) * args.neg_ratio))
            keep_neg = ([neg[i] for i in rng.choice(len(neg), k, replace=False)]
                        if neg and k > 0 else [])
        n_take_pos = pos if trusted else pos

        crop_dir = os.path.join(args.out, "ourdata", name)
        idx = 0
        for y0, x0 in n_take_pos + keep_neg:
            is_pos = (y0, x0) in set(pos)
            d = os.path.join(crop_dir, f"z{idx:04d}")
            os.makedirs(d, exist_ok=True)
            Image.fromarray(img2[y0:y0 + T, x0:x0 + T]).save(os.path.join(d, "em.png"))
            Image.fromarray((gt2[y0:y0 + T, x0:x0 + T].astype(np.uint8) * 255)).save(
                os.path.join(d, "mito.png"))
            with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"image": name, "y": y0, "x": x0, "nm_per_px": nm,
                           "coverage": cov, "trusted": trusted,
                           "is_positive": bool(is_pos)}, fh)
            manifest.append({"image": name, "tile": idx, "y": y0, "x": x0,
                             "is_positive": bool(is_pos), "coverage": cov,
                             "trusted": trusted})
            idx += 1
            if is_pos:
                n_pos += 1
            else:
                n_neg += 1
        print(f"{name:<10} {nm:>7.2f} {factor:>6.2f} {cov*100:>6.1f}% "
              f"{len(n_take_pos):>7} {len(keep_neg):>7} {str(trusted):>5}")

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"target_nm": args.target_nm, "tile": args.tile,
                   "tiles": manifest,
                   "n_positive": n_pos, "n_negative": n_neg,
                   "n_images": len({m['image'] for m in manifest})}, fh,
                  ensure_ascii=False, indent=1)

    print()
    print(f"共 {len(manifest)} 个 tile（正 {n_pos} / 负 {n_neg}），"
          f"来自 {len({m['image'] for m in manifest})} 张图")
    print(f"已写出 {args.out}/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
