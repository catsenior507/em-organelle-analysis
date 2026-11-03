#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asem_to_tiles.py —— 把 ASEM 的 3D 体数据转成 2D 训练切片

## 为什么转成 2D

ASEM 的标注是三维体（5 nm 各向同性），而我们的目标是给**自己的 2D 切片**做分割。
两条路：
* 直接用 ASEM 的 3D U-Net -> 需要伪造三维上下文，且我们的图是单层
* **把体数据沿 z 切成 2D 切片** -> 与我们既有 2D 数据同构，管线现成

选后者：可复用 `train_seg.py`、统一层与所有评估脚本。

## 尺度说明

ASEM 是 5 nm/px 各向同性。我们自己的高尔基体图是期刊插图，尺度未知，
但按 Golgi 结构占 200–400 像素、实际约 1 µm 推算，大致也是 3–5 nm/px。
因此本脚本**默认不重采样**，直接切片；若后续确认尺度差得多，可用 `--scale` 调整。

## 输出

    <out>/asem<ds>/<类别>/z{slice}/z{tile}/{em.png,<类别>.png}

沿用统一层的目录约定（`<class>/<id>/z####/{em.png,<class>.png}`），
所以可以直接把多个来源的目录并列，再用 `train_seg.py --train-root` 的父目录合并训练。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE8 = np.ones((3, 3), bool)


def tile_starts(total: int, tile: int, stride: int) -> List[int]:
    if total <= tile:
        return [0]
    s = list(range(0, total - tile + 1, max(1, stride)))
    if s[-1] != total - tile:
        s.append(total - tile)
    return s


def pad_to_tile(arr: np.ndarray, tile: int) -> np.ndarray:
    """把小于 tile 的切面补到至少 tile×tile（edge 模式，复制边缘像素）。

    见下方调用处的说明：数据集 64 的 golgi 切面是 204×424，
    短边不足 256，不补就会整卷产不出 tile。
    """
    h, w = arr.shape
    ph = max(0, tile - h)
    pw = max(0, tile - w)
    if ph == 0 and pw == 0:
        return arr
    return np.pad(arr, ((0, ph), (0, pw)), mode="edge")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/asem")
    ap.add_argument("--dataset", required=True, help="如 61、64")
    ap.add_argument("--labels", required=True, help="逗号分隔，如 golgi 或 er,mito")
    ap.add_argument("--out", default="data/asem_tiles")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--min-pos-frac", type=float, default=0.01)
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--max-slices", type=int, default=0,
                    help="最多取多少层（0=全部）；用于控制数据量")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    src = os.path.join(ROOT, args.src, f"ds{args.dataset}")
    labs = [s.strip() for s in args.labels.split(",") if s.strip()]

    # raw 按"首个标注体"分别切，文件名带后缀 raw__<label>.npy。
    # 同一数据集的不同标注 ROI 不一定相同（ds64 的 golgi 与 er 就是两个 ROI），
    # 所以不能用同一个 raw。
    raw_p = os.path.join(src, f"raw__{labs[0]}.npy")
    if not os.path.exists(raw_p):
        alt = os.path.join(src, "raw.npy")
        raw_p = alt if os.path.exists(alt) else raw_p
    if not os.path.exists(raw_p):
        print(f"[错误] 缺 raw：期望 {os.path.basename(raw_p)}")
        return 2
    raw = np.load(raw_p)
    print(f"raw {raw.shape} {raw.dtype}（来自 {os.path.basename(raw_p)}）")
    rng = np.random.RandomState(args.seed)
    manifests: Dict[str, List[Dict]] = {}

    for lab in labs:
        lp = os.path.join(src, f"{lab}.npy")
        if not os.path.exists(lp):
            print(f"[跳过] 缺 {lp}")
            continue
        gt = np.load(lp)
        if gt.shape != raw.shape:
            print(f"[跳过] {lab} 形状 {gt.shape} 与 raw {raw.shape} 不一致")
            continue
        binary = (gt > 0)
        print(f"{lab}: {gt.shape} 前景 {100*binary.mean():.2f}%")

        out_root = os.path.join(ROOT, args.out, f"asem{args.dataset}", lab)
        if os.path.isdir(out_root):
            shutil.rmtree(out_root)

        n_slices = raw.shape[0]
        slice_ids = list(range(n_slices))
        if args.max_slices and n_slices > args.max_slices:
            # 均匀取样，保留整卷的分布
            slice_ids = sorted(rng.choice(n_slices, args.max_slices, replace=False).tolist())

        rows: List[Dict] = []
        tot_pos = tot_neg = 0
        for zi in slice_ids:
            em2d = raw[zi]
            gt2d = binary[zi]
            if gt2d.sum() < 50:
                continue
            # 切面小于 tile 时必须补齐，不能跳过。
            #
            # 数据集 64 的 golgi ROI 是 284×204×424，沿 z 切出来是 **204×424**，
            # 短边 204 < 256。原实现里 `g.shape != (tile, tile)` 会 continue，
            # 结果整卷 golgi 一个 tile 都产不出来（正负皆 0）——
            # 这不是"跳过无效切片"，切片是有效的，只是不够大。
            # 用 edge 模式补边：复制边缘像素，不制造虚假结构；
            # GT 补 0（背景），因此不会凭空生成正样本。
            em2d = pad_to_tile(em2d, args.tile) if em2d.shape[0] < args.tile \
                or em2d.shape[1] < args.tile else em2d
            gt2d = pad_to_tile(gt2d, args.tile) if gt2d.shape[0] < args.tile \
                or gt2d.shape[1] < args.tile else gt2d
            h, w = em2d.shape
            pos, neg = [], []
            for y0 in tile_starts(h, args.tile, args.stride):
                for x0 in tile_starts(w, args.tile, args.stride):
                    g = gt2d[y0:y0 + args.tile, x0:x0 + args.tile]
                    if g.shape != (args.tile, args.tile):
                        continue
                    (pos if g.mean() >= args.min_pos_frac else neg).append((y0, x0))
            k = int(round(len(pos) * args.neg_ratio))
            keep_neg = ([neg[j] for j in rng.choice(len(neg), min(k, len(neg)),
                                                    replace=False)]
                        if neg and k > 0 else [])
            if not pos:
                continue
            sid = f"z{zi:04d}"
            crop = os.path.join(out_root, sid)
            idx = 0
            posset = set(pos)
            for y0, x0 in pos + keep_neg:
                d = os.path.join(crop, f"z{idx:04d}")
                os.makedirs(d, exist_ok=True)
                Image.fromarray(em2d[y0:y0 + args.tile, x0:x0 + args.tile]).save(
                    os.path.join(d, "em.png"))
                Image.fromarray((gt2d[y0:y0 + args.tile, x0:x0 + args.tile]
                                 .astype(np.uint8) * 255)).save(
                    os.path.join(d, f"{lab}.png"))
                idx += 1
                if (y0, x0) in posset:
                    tot_pos += 1
                else:
                    tot_neg += 1
            rows.append({"slice": zi, "n_pos": len(pos), "n_neg": len(keep_neg)})

        manifests[lab] = rows
        print(f"  -> {len(rows)} 层，正 tile {tot_pos}，负 tile {tot_neg}")
        print(f"     输出 {os.path.relpath(out_root, ROOT)}")

    os.makedirs(os.path.join(ROOT, args.out, f"asem{args.dataset}"), exist_ok=True)
    with open(os.path.join(ROOT, args.out, f"asem{args.dataset}",
                           "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"dataset": args.dataset, "labels": labs,
                   "raw_shape": list(raw.shape), "tile": args.tile,
                   "stride": args.stride, "slices": manifests},
                  fh, ensure_ascii=False, indent=1)
    print()
    print("提示：ASEM 是 5 nm/px 各向同性；我们自己的高尔基体插图估算 3–5 nm/px，")
    print("      故默认不重采样。若尺度差异明显，需先重采样再并入训练。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
