#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oo_to_tiles.py —— 把 OpenOrganelle crop 转成 2D 训练切片

## 两种模式

**单类模式（默认）**：`--classes golgi`
输出 `<out>/golgi/<crop>_z####/z####/{em.png, golgi.png}`。
每个类别各自一棵目录树。

**多类模式**：`--multi --classes golgi,er,endo,ves,lyso`
输出 `<out>/multi/<crop>_z####/z####/{em.png, golgi.png, er.png, ...}`。
**同一瓦片目录里放全部类别的掩码**——这是 `train_seg.py` 多类训练的前提：
`SegDataset` 默认 `require_all=True`，缺任一类别掩码的瓦片会被整块丢弃。
早期版本只有单类模式，多类训练时因此读到 0 个切片，且报错说的是
"没找到任何切片"，看不出根因是"掩码分散在五棵目录树里"。

## 采样判据

* 单类：以该类前景占比 ≥ `--min-pos-frac` 作为正样本
* 多类：以**所有类别前景的并集**作为判据——预训练的目的是学会区分多种细胞器，
  按单一类别采样会让该类恒为前景，学不到"缺席"的情况

## 关于尺度

OpenOrganelle 是 4.0 nm 各向同性。我们自有高尔基体插图估算 3–5 nm/px，
量级相同，因此**不重采样**——在尺度不确定时重采样等于凭空引入一个假设。

## 关于 255=未知

标注里有 255（`unknown`）这一档，写进 `ignore.png`（255 → 忽略），
让训练时不计入损失。这是忽略区机制第一个**权威的、非人为构造**的输入来源。

## 关于强度

逐块做 1%–99% 百分位拉伸。OO 的对比度只有自有的四分之一（标准差 12.3 对 56.9），
不拉伸则混合训练会把模型彻底毁掉（实测连续 10 轮 Dice 为 0.000）。详见
`docs/GOLGI-OPENORGANELLE-RESULTS.md` §3.1。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tile_starts(total: int, tile: int, stride: int) -> List[int]:
    if total <= tile:
        return [0]
    s = list(range(0, total - tile + 1, max(1, stride)))
    if s[-1] != total - tile:
        s.append(total - tile)
    return s


def stretch(a: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """按百分位把灰度拉伸到 0–255。

    实测两批图像的强度分布差得很远：

    | | 均值 | 标准差 | <64 像素 | >192 像素 |
    |---|---|---|---|---|
    | OpenOrganelle | 172.7 | 12.3 | 0.0% | 2.3% |
    | 自有 TEM 插图 | 140.8 | 56.9 | 10.8% | 19.1% |

    OO 的切片几乎没有可见的膜纹理，直接混合训练时模型学不到形态。
    逐块拉伸到同一动态范围后，两批数据对网络而言才可比。
    """
    if a.size == 0:
        return a
    lo_v, hi_v = np.percentile(a, [lo, hi])
    if hi_v <= lo_v:
        return a
    b = (a.astype(np.float32) - lo_v) * (255.0 / (hi_v - lo_v))
    return np.clip(b, 0, 255).astype(np.uint8)


def load_crop(d: str, classes: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray],
                                                    Dict[str, np.ndarray]]:
    """读一个 crop 的 EM 与全部类别掩码。返回 (em, masks, unknowns)。"""
    em = np.load(os.path.join(d, "em.npy"))
    masks, unknowns = {}, {}
    for c in classes:
        p = os.path.join(d, f"{c}.npy")
        if not os.path.exists(p):
            continue
        gt = np.load(p)
        if gt.shape != em.shape:
            print(f"    形状不一致 {c} {gt.shape} vs em {em.shape}，跳过该类")
            continue
        init = (gt == 255)
        masks[c] = ((gt == 1) | (gt.astype(bool) & ~init)).astype(np.uint8)
        if init.any():
            unknowns[c] = init
    return em, masks, unknowns


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/oo")
    ap.add_argument("--out", default="data/oo_tiles")
    ap.add_argument("--classes", default="golgi,er,mito")
    ap.add_argument("--multi", action="store_true",
                    help="多类模式：全部类别掩码写进同一瓦片目录")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--min-pos-frac", type=float, default=0.01)
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--max-slices", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--normalize", action="store_true", default=True)
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    args = ap.parse_args(argv)

    src = os.path.join(ROOT, args.src)
    if not os.path.isdir(src):
        print(f"[错误] 缺 {args.src}")
        return 2
    crops = sorted(d for d in os.listdir(src)
                   if os.path.isdir(os.path.join(src, d)))
    if not crops:
        print(f"[错误] {args.src} 下没有 crop 目录")
        return 2
    print(f"{len(crops)} 个 crop：{crops}")

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    rng = np.random.RandomState(args.seed)
    manifest: Dict = {}

    if args.multi:
        name = "multi"
        out_root = os.path.join(ROOT, args.out, name)
        os.makedirs(out_root, exist_ok=True)
        print(f"\n=== 多类模式：掩码同目录 {classes}")
        rows = []
        tot_pos = tot_neg = 0
        for crop in crops:
            d = os.path.join(src, crop)
            em, masks, unknowns = load_crop(d, classes)
            have = [c for c in classes if c in masks]
            if len(have) < len(classes):
                print(f"  {crop}: 缺 {sorted(set(classes)-set(have))}，跳过")
                continue
            # 采样判据：所有类别前景的并集（逐层，保持三维形状）
            union = np.zeros_like(masks[have[0]])
            for c in have:
                union |= masks[c]
            n_sl = em.shape[0]
            idx = list(range(n_sl))
            if args.max_slices and n_sl > args.max_slices:
                idx = sorted(rng.choice(n_sl, args.max_slices, replace=False).tolist())

            cpos = cneg = 0
            for zi in idx:
                if union[zi].sum() < 50:
                    continue
                h, w = union[zi].shape
                pos, neg = [], []
                for y0 in tile_starts(h, args.tile, args.stride):
                    for x0 in tile_starts(w, args.tile, args.stride):
                        u = union[zi][y0:y0 + args.tile, x0:x0 + args.tile]
                        if u.shape != (args.tile, args.tile):
                            continue
                        (pos if u.mean() >= args.min_pos_frac else neg).append((y0, x0))
                k = int(round(len(pos) * args.neg_ratio))
                keep_neg = ([neg[j] for j in rng.choice(len(neg), min(k, len(neg)),
                                                        replace=False)]
                            if neg and k > 0 else [])
                if not pos:
                    continue
                cropdir = os.path.join(out_root, f"{crop}_z{zi:04d}")
                ti = 0
                for y0, x0 in pos + keep_neg:
                    dd = os.path.join(cropdir, f"z{ti:04d}")
                    os.makedirs(dd, exist_ok=True)
                    patch = em[zi, y0:y0 + args.tile, x0:x0 + args.tile]
                    if args.normalize:
                        patch = stretch(patch)
                    Image.fromarray(patch).save(os.path.join(dd, "em.png"))
                    for c in have:
                        m = masks[c][zi, y0:y0 + args.tile, x0:x0 + args.tile]
                        Image.fromarray(m * 255).save(os.path.join(dd, f"{c}.png"))
                    # 忽略图取各类 unknown 的并集
                    us = None
                    for c in have:
                        if c in unknowns:
                            u = unknowns[c][zi, y0:y0 + args.tile, x0:x0 + args.tile]
                            us = u if us is None else (us | u)
                    if us is not None and us.any():
                        Image.fromarray(us.astype(np.uint8) * 255).save(
                            os.path.join(dd, "ignore.png"))
                    ti += 1
                cpos += len(pos)
                cneg += len(keep_neg)
            rows.append({"crop": crop, "pos": cpos, "neg": cneg})
            tot_pos += cpos
            tot_neg += cneg
            nz = {c: float((masks[c] > 0).mean()) for c in have}
            print(f"  {crop:<22} 正 {cpos:>5} 负 {cneg:>5}  "
                  + " ".join(f"{c}={nz[c]*100:.1f}%" for c in have))
        manifest[name] = {"classes": classes, "rows": rows,
                          "pos": tot_pos, "neg": tot_neg,
                          "tile": args.tile, "stride": args.stride,
                          "sampling": "union of all classes"}
        print(f"  小计：正 {tot_pos} 负 {tot_neg}")
        print(f"\n输出 {args.out}/{name}/<crop>_z<层>/z<瓦片>/"
              f"（含 {len(classes)} 个掩码）")
    else:
        for cls in classes:
            out_root = os.path.join(ROOT, args.out, cls)
            os.makedirs(out_root, exist_ok=True)
            rows = []
            tot_pos = tot_neg = tot_ign = 0
            print(f"\n=== 类别 {cls}")
            for crop in crops:
                d = os.path.join(src, crop)
                em, masks, unknowns = load_crop(d, [cls])
                if cls not in masks:
                    print(f"  {crop}: 缺通道，跳过")
                    continue
                g_all = masks[cls]
                u_all = unknowns.get(cls)
                n_sl = em.shape[0]
                idx = list(range(n_sl))
                if args.max_slices and n_sl > args.max_slices:
                    idx = sorted(rng.choice(n_sl, args.max_slices,
                                            replace=False).tolist())
                cpos = cneg = cign = 0
                for zi in idx:
                    g2 = g_all[zi]
                    if g2.sum() < 50:
                        continue
                    h, w = g2.shape
                    pos, neg = [], []
                    for y0 in tile_starts(h, args.tile, args.stride):
                        for x0 in tile_starts(w, args.tile, args.stride):
                            g = g2[y0:y0 + args.tile, x0:x0 + args.tile]
                            if g.shape != (args.tile, args.tile):
                                continue
                            (pos if g.mean() >= args.min_pos_frac
                             else neg).append((y0, x0))
                    k = int(round(len(pos) * args.neg_ratio))
                    keep_neg = ([neg[j] for j in rng.choice(len(neg), min(k, len(neg)),
                                                            replace=False)]
                                if neg and k > 0 else [])
                    if not pos:
                        continue
                    cropdir = os.path.join(out_root, f"{crop}_z{zi:04d}")
                    ti = 0
                    for y0, x0 in pos + keep_neg:
                        dd = os.path.join(cropdir, f"z{ti:04d}")
                        os.makedirs(dd, exist_ok=True)
                        patch = em[zi, y0:y0 + args.tile, x0:x0 + args.tile]
                        if args.normalize:
                            patch = stretch(patch)
                        Image.fromarray(patch).save(os.path.join(dd, "em.png"))
                        Image.fromarray((g2[y0:y0 + args.tile, x0:x0 + args.tile]
                                         * 255)).save(os.path.join(dd, f"{cls}.png"))
                        if u_all is not None:
                            us = u_all[zi, y0:y0 + args.tile, x0:x0 + args.tile]
                            if us.any():
                                Image.fromarray(us.astype(np.uint8) * 255).save(
                                    os.path.join(dd, "ignore.png"))
                                cign += 1
                        ti += 1
                    cpos += len(pos)
                    cneg += len(keep_neg)
                rows.append({"crop": crop, "pos": cpos, "neg": cneg, "ignore": cign})
                tot_pos += cpos
                tot_neg += cneg
                tot_ign += cign
                print(f"  {crop:<22} 正 {cpos:>5} 负 {cneg:>5} 含忽略 {cign:>4}")
            manifest[cls] = {"rows": rows, "pos": tot_pos, "neg": tot_neg,
                             "ignore": tot_ign, "tile": args.tile,
                             "stride": args.stride,
                             "sampling": f"{cls} foreground"}
            print(f"  小计：正 {tot_pos} 负 {tot_neg}")

    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    mp = os.path.join(ROOT, args.out, "manifest.json")
    old = {}
    if os.path.exists(mp):
        try:
            old = json.load(open(mp, encoding="utf-8"))
        except Exception:
            old = {}
    old.update(manifest)
    with open(mp, "w", encoding="utf-8") as fh:
        json.dump(old, fh, ensure_ascii=False, indent=1)
    print(f"\n清单 {args.out}/manifest.json")
    print("提示：OpenOrganelle 为 4.0 nm/px；自有插图为 3–5 nm/px，故不重采样。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
