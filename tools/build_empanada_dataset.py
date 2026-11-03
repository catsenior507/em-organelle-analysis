#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_empanada_dataset.py —— 把我们自己的 tile 转成 empanada 微调所需格式

## empanada 要求的格式

    <out>/<子集>/images/<name>.png   灰度，cv2.imread(...,0) 读取
    <out>/<子集>/masks/<name>.png    整数标签，**每个实例一个唯一 ID**，0 为背景

已验证的实现：
`empanada.data._base._BaseDataset` 用 `glob(<data_dir>/<子集>/images/*)` 与
`masks/*` 配对；`SingleClassInstanceDataset.__getitem__` 用 `cv2.imread(f, 0)`
读图、`skimage.io.imread` 读掩码，再用 `heatmap_and_offsets(mask, sigma)`
从实例标签生成中心热图与偏移量。

**所以掩码必须是实例标签，不能是 0/1 二值。** 我们现有的 `mito.png` 是二值，
这里用连通域标号转成实例 ID。

## 划分

按**图像**划分（不是按 tile）：用 `--val-images` 指定的名单做验证集，
其余做训练。这与此前所有实验的口径一致，避免同一张图的相邻 tile 泄漏。
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

FACE8 = np.ones((3, 3), bool)     # 8 连通，与掩码过滤时一致


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="data/ourdata_tiles/ourdata")
    ap.add_argument("--out", default="data/empanada_ft")
    ap.add_argument("--val-images", default="models/ourdata_mito/val_images.txt")
    ap.add_argument("--min-instance-px", type=int, default=50,
                    help="小于该像素数的实例并入背景。empanada 的 heatmap 会为"
                         "每个实例生成峰，过小的碎块会污染训练")
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy", file=sys.stderr)
        return 2

    val_imgs: Set[str] = set()
    if os.path.exists(args.val_images):
        val_imgs = {l.strip() for l in open(args.val_images, encoding="utf-8")
                    if l.strip()}
    print(f"验证图像名单：{len(val_imgs)} 张")

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    # empanada 的结构是 data_dir -> source_datasets -> images/masks，
    # 即 _BaseDataset 先列 data_dir 的子目录，再在每个子目录下 glob images/*。
    # 所以必须在 images/masks 之上多一层"来源"目录，否则它会去找
    # <data_dir>/images/images/*。
    for sub in ("train", "val"):
        os.makedirs(os.path.join(args.out, sub, "ourdata", "images"), exist_ok=True)
        os.makedirs(os.path.join(args.out, sub, "ourdata", "masks"), exist_ok=True)

    stats: Dict[str, Dict[str, int]] = {
        "train": {"n": 0, "inst": 0, "px": 0},
        "val": {"n": 0, "inst": 0, "px": 0},
    }

    crops = sorted(d for d in os.listdir(args.tiles)
                   if os.path.isdir(os.path.join(args.tiles, d)))
    print(f"图像目录 {len(crops)} 个")
    print()
    print(f"{'图像':<10} {'子集':>6} {'tile数':>7} {'实例数':>7} {'前景%':>7}")
    print("-" * 42)

    for crop in crops:
        sub = "val" if crop in val_imgs else "train"
        tdirs = sorted(glob.glob(os.path.join(args.tiles, crop, "z*")))
        n_tiles = n_inst = 0
        fg = 0.0
        for td in tdirs:
            em_p = os.path.join(td, "em.png")
            mk_p = os.path.join(td, "mito.png")
            if not (os.path.exists(em_p) and os.path.exists(mk_p)):
                continue
            m = np.asarray(Image.open(mk_p)) > 127
            lab, n = ndi.label(m, structure=FACE8)
            if n:
                sizes = np.bincount(lab.ravel())
                keep = np.zeros(sizes.size, bool)
                keep[1:] = sizes[1:] >= args.min_instance_px
                inst = keep[lab]
                # 重新连续编号，避免 ID 有空洞
                if inst.any():
                    lab2, n2 = ndi.label(inst, structure=FACE8)
                    # **必须用 uint8**：empanada 的 SingleClassInstanceDataset
                    # 会执行 (mask > 0).float()，而 torch 不支持 uint16 的比较，
                    # 会抛 `"gt_cpu" not implemented for 'UInt16'`。
                    # 每个 tile 的实例数是个位数，uint8 足够；
                    # 但仍要检查，避免静默截断。
                    if n2 > 255:
                        raise RuntimeError(
                            f"{td} 有 {n2} 个实例，超过 uint8 上限 255")
                    inst_map = lab2.astype(np.uint8)
                else:
                    inst_map = np.zeros_like(lab, dtype=np.uint8)
                    n2 = 0
            else:
                inst_map = np.zeros_like(lab, dtype=np.uint8)
                n2 = 0

            name = f"{crop}_{os.path.basename(td)}.png"
            # 图像：直接复制（灰度）
            shutil.copyfile(em_p, os.path.join(
                args.out, sub, "ourdata", "images", name))
            Image.fromarray(inst_map).save(
                os.path.join(args.out, sub, "ourdata", "masks", name))
            n_tiles += 1
            n_inst += int(n2)
            fg += float((inst_map > 0).mean())
        print(f"{crop:<10} {sub:>6} {n_tiles:>7} {n_inst:>7} "
              f"{(100*fg/max(1,n_tiles)):>6.1f}%")
        stats[sub]["n"] += n_tiles
        stats[sub]["inst"] += n_inst
        stats[sub]["px"] += int(fg * 256 * 256)

    print()
    print("=" * 42)
    for sub in ("train", "val"):
        s = stats[sub]
        print(f"{sub}: {s['n']} 张，实例 {s['inst']} 个，"
              f"前景像素 {s['px']/1e6:.2f} M")
    print(f"输出 {args.out}/{{train,val}}/{{images,masks}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
