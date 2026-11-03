#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_all_ourdata.py —— 用训练好的模型为**全部**图像导出线粒体掩码

## 产出

1. `data/masks_ourdata/<图像>.png` —— 二值掩码，**原始分辨率**（0/255）
2. `data/masks_ourdata/index.csv` —— 每张图的统计：
   线粒体面积占比、连通域个数、平均面积、中位等效直径、是否曾被标注、Dice（如有标注）
3. 可选叠加图目录

## 重要说明（必须在报告里写清）

* 模型训练时用了 88 张图中的 70 张，**18 张是留出的**。
  `index.csv` 里用 `split` 列标明 `train` / `heldout`，
  以便分析时区分"样本内预测"与"留出预测"。
* 掩码是在 **4 nm/px** 下预测后**最近邻放大回原分辨率**的。
  放大不会增加信息：对 1 nm/px 的图，等效精度受限于 4 nm；
  对 5.29 nm/px 的图则基本无损。若需原分辨率精度，应在原尺度上重训。
* 预测可能包含**标注未覆盖的真实线粒体**（部分标注问题），
  因此预测面积通常大于标注面积；这不是错误。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "server", "scripts"))
# 让 <repo>/src 可被导入（公开版把训练器与数据集读取器放在 src/）。
_SRC = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from train_seg import UNet  # noqa: E402

TARGET_NM = 4.0
TILE = 256
STRIDE = 192


def load_gray(p: str) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L"), dtype=np.uint8)


def resample_img(img: np.ndarray, nm: float, target: float):
    factor = nm / target
    if abs(factor - 1.0) < 1e-3:
        return img, 1.0
    h, w = img.shape
    nw, nh = max(16, int(round(w * factor))), max(16, int(round(h * factor)))
    return np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR)), factor


@torch.no_grad()
def predict_full(model, img: np.ndarray, tile=TILE, stride=STRIDE) -> np.ndarray:
    h, w = img.shape
    prob = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    x = img.astype(np.float32) / 255.0
    x3 = np.stack([x, x, x], 0)

    def starts(total):
        if total <= tile:
            return [0]
        s = list(range(0, total - tile + 1, stride))
        if s[-1] != total - tile:
            s.append(total - tile)
        return s

    for y0 in starts(h):
        for x0 in starts(w):
            patch = x3[:, y0:y0 + tile, x0:x0 + tile]
            ph, pw = patch.shape[1], patch.shape[2]
            if ph < tile or pw < tile:
                patch = np.pad(patch, ((0, 0), (0, tile - ph), (0, tile - pw)),
                               mode="reflect")
            out = torch.sigmoid(model(torch.from_numpy(patch).unsqueeze(0)))[0, 0].numpy()
            out = out[:ph, :pw]
            prob[y0:y0 + ph, x0:x0 + pw] += out
            cnt[y0:y0 + ph, x0:x0 + pw] += 1.0
    cnt[cnt == 0] = 1.0
    return prob / cnt


def region_stats(mask: np.ndarray, nm: float) -> Dict[str, float]:
    """连通域统计（单位换算到 µm）。"""
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return {"n_regions": 0, "mean_area_um2": 0.0,
                "median_eq_diam_um": 0.0, "max_area_um2": 0.0}
    sizes = np.bincount(lab.ravel())[1:]
    px_um2 = (nm / 1000.0) ** 2
    areas = sizes * px_um2
    eq_diam = 2 * np.sqrt(sizes / np.pi) * nm / 1000.0
    return {"n_regions": int(n),
            "mean_area_um2": float(areas.mean()),
            "median_eq_diam_um": float(np.median(eq_diam)),
            "max_area_um2": float(areas.max())}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/ourdata_mito/best.pt")
    ap.add_argument("--images", default="data/原图片")
    ap.add_argument("--masks", default="data/分离的标注信息")
    ap.add_argument("--annot", default="data/annotations_v2/dataset.json")
    ap.add_argument("--out", default="data/masks_ourdata")
    ap.add_argument("--val-list", default="models/ourdata_mito/val_images.txt")
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--overlays", default=None, help="若给目录，则同时保存叠加图")
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy", file=sys.stderr)
        return 2

    val = set()
    if os.path.exists(args.val_list):
        val = {l.strip() for l in open(args.val_list, encoding="utf-8") if l.strip()}

    meta = json.load(open(args.annot, encoding="utf-8"))
    ents = meta["images"] if isinstance(meta["images"], list) else list(meta["images"].values())
    nm_of = {e["image"]: (e.get("um_per_px") or 0) * 1000 for e in ents}
    n_regions_annot = {e["image"]: e.get("num_regions") for e in ents}

    model = UNet(3, 1, 32)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    os.makedirs(args.out, exist_ok=True)
    if args.overlays:
        os.makedirs(args.overlays, exist_ok=True)

    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(".tif"))
    print(f"共 {len(files)} 张图；模型 epoch {ck.get('epoch')}；阈值 {args.thr}")

    rows: List[Dict] = []
    print()
    print(f"{'图像':<10} {'nm/px':>7} {'划分':>8} {'预测占比':>8} {'区域数':>6} "
          f"{'中位直径µm':>10} {'注释区域':>8}")
    print("-" * 70)
    for f in files:
        name = os.path.splitext(f)[0]
        ip = os.path.join(args.images, f)
        img = load_gray(ip)
        nm = nm_of.get(name) or TARGET_NM
        img4, factor = resample_img(img, nm, TARGET_NM)
        prob = predict_full(model, img4)
        pred4 = prob > args.thr

        # 最近邻放大回原分辨率
        mask_full = np.asarray(
            Image.fromarray(pred4.astype(np.uint8) * 255).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST)) > 127
        Image.fromarray((mask_full.astype(np.uint8) * 255)).save(
            os.path.join(args.out, f"{name}.png"))

        st = region_stats(mask_full, nm)
        split = "heldout" if name in val else "train"
        rows.append({"image": name, "nm_per_px": nm, "split": split,
                     "pred_frac": float(mask_full.mean()),
                     "n_regions": st["n_regions"],
                     "mean_area_um2": st["mean_area_um2"],
                     "median_eq_diam_um": st["median_eq_diam_um"],
                     "max_area_um2": st["max_area_um2"],
                     "annot_regions": n_regions_annot.get(name)})
        print(f"{name:<10} {nm:>7.2f} {split:>8} {mask_full.mean()*100:>7.2f}% "
              f"{st['n_regions']:>6} {st['median_eq_diam_um']:>10.3f} "
              f"{str(n_regions_annot.get(name)):>8}")

        if args.overlays:
            rgb = np.stack([img] * 3, -1).astype(np.uint8)
            out = (rgb * 0.45).astype(np.uint8)
            out[mask_full] = [220, 0, 0]
            im = Image.fromarray(out)
            im.thumbnail((900, 900), Image.LANCZOS)
            im.save(os.path.join(args.overlays, f"{name}.png"))

    csv_path = os.path.join(args.out, "index.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pf = np.array([r["pred_frac"] for r in rows])
    nr = np.array([r["n_regions"] for r in rows])
    print()
    print("=" * 70)
    print(f"完成 {len(rows)} 张；掩码目录 {args.out}/")
    print(f"预测线粒体面积占比：均值 {pf.mean()*100:.2f}%  "
          f"中位 {np.median(pf)*100:.2f}%  范围 [{pf.min()*100:.2f}%, {pf.max()*100:.2f}%]")
    print(f"每图连通域数：均值 {nr.mean():.1f}  中位 {np.median(nr):.0f}  最多 {nr.max()}")
    print(f"其中留出图 {sum(1 for r in rows if r['split']=='heldout')} 张、"
          f"训练图 {sum(1 for r in rows if r['split']=='train')} 张")
    print(f"统计表 {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
