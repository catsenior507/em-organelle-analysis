#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fullimage_ourdata.py —— 对留出图像做**全图**推理与评估

## 为什么还要做这一步

训练时报告的是 **tile 级** Dice（0.7911），而 tile 大多以线粒体为中心，
因此会高估真实表现。真正要回答的是"给一张完整图，能不能分割好"。

本脚本对**留出图像**（训练时未使用的 18 张）做全图滑窗推理：
按训练时的 4 nm/px 重采样 → 256 tile / 192 步长 → 拼接 → 与填充后的
绿色标注真值比较，报 Dice / IoU / P / R，并保存叠加图供肉眼核对。

## 与 tile 级评估的差别

* tile 级只统计"含标注的 tile"，模型不必面对大片纯净背景
* 全图评估包含全部背景区域，若模型倾向于在大片背景上误报，Dice 会明显下降
* 若两者接近，说明结论稳健
"""

from __future__ import annotations

import argparse
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


def green_outline(p: str) -> np.ndarray:
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.int16)
    r, g = a[:, :, 0], a[:, :, 1]
    return (g > 40) & (g > r + 20)


def fill_outline(o: np.ndarray) -> np.ndarray:
    c = ndi.binary_closing(o, structure=np.ones((3, 3)), iterations=2)
    f = ndi.binary_fill_holes(c)
    return c if f.mean() > 0.6 else f


def resample(img: np.ndarray, gt: np.ndarray, nm: float, target: float):
    factor = nm / target
    if abs(factor - 1.0) < 1e-3:
        return img, gt, 1.0
    h, w = img.shape
    nw, nh = max(16, int(round(w * factor))), max(16, int(round(h * factor)))
    i2 = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    m2 = np.asarray(Image.fromarray((gt * 255).astype(np.uint8))
                    .resize((nw, nh), Image.BILINEAR)) > 127
    return i2.astype(np.uint8), m2, factor


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


def metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    inter = float((pred & gt).sum())
    ps, gs = float(pred.sum()), float(gt.sum())
    return {
        "dice": 2 * inter / (ps + gs) if (ps + gs) > 0 else float("nan"),
        "iou": inter / (ps + gs - inter) if (ps + gs - inter) > 0 else float("nan"),
        "precision": inter / ps if ps > 0 else float("nan"),
        "recall": inter / gs if gs > 0 else float("nan"),
        "gt_frac": gs / gt.size,
        "pred_frac": ps / gt.size,
        "baseline_dice": 2 * gs / (gt.size + gs) if gs > 0 else float("nan"),
    }


def save_overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, path: str):
    """绿=真值、红=预测、黄=重合。"""
    rgb = np.stack([img] * 3, -1).astype(np.uint8)
    out = (rgb * 0.45).astype(np.uint8)
    out[gt] = [0, 200, 0]
    out[pred] = [220, 0, 0]
    out[gt & pred] = [255, 220, 0]
    # 缩小以便查看
    im = Image.fromarray(out)
    im.thumbnail((900, 900), Image.LANCZOS)
    im.save(path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/ourdata_mito/best.pt")
    ap.add_argument("--images", default="data/原图片")
    ap.add_argument("--masks", default="data/分离的标注信息")
    ap.add_argument("--annot", default="data/annotations_v2/dataset.json")
    ap.add_argument("--val-list", default="models/ourdata_mito/val_images.txt",
                    help="留出图像名列表（每行一个）；缺省则用 history 里的 val_per_crop")
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--out-json", default="logs/fullimage_ourdata.json")
    ap.add_argument("--out-dir", default="logs/overlays")
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy", file=sys.stderr)
        return 2

    # 取留出图像列表
    vals: List[str] = []
    if os.path.exists(args.val_list):
        vals = [l.strip() for l in open(args.val_list, encoding="utf-8") if l.strip()]
    if not vals:
        hp = os.path.join(os.path.dirname(args.ckpt), "history.json")
        if os.path.exists(hp):
            h = json.load(open(hp, encoding="utf-8"))
            b = max(h, key=lambda r: r.get("val_mean_dice", -1))
            vals = sorted((b.get("val_per_crop") or {}).keys())
    if not vals:
        print("[错误] 拿不到留出图像列表", file=sys.stderr)
        return 2

    meta = json.load(open(args.annot, encoding="utf-8"))
    ents = meta["images"] if isinstance(meta["images"], list) else list(meta["images"].values())
    nm_of = {e["image"]: (e.get("um_per_px") or 0) * 1000 for e in ents}

    model = UNet(3, 1, 32)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"检查点 epoch {ck.get('epoch')}  验证图像 {len(vals)} 张  thr={args.thr}")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    print()
    print(f"{'图像':<10} {'nm/px':>7} {'GT占比':>7} {'预测占比':>8} "
          f"{'Dice':>7} {'IoU':>7} {'P':>6} {'R':>6} {'基线':>7}")
    print("-" * 78)
    for name in vals:
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
        nm = nm_of.get(name) or TARGET_NM
        img4, gt4, factor = resample(img, gt, nm, TARGET_NM)
        prob = predict_full(model, img4)
        pred = prob > args.thr
        m = metrics(pred, gt4)
        rows.append({"image": name, "nm_per_px": nm, "shape": list(gt4.shape), **m})
        print(f"{name:<10} {nm:>7.2f} {m['gt_frac']*100:>6.1f}% {m['pred_frac']*100:>7.1f}% "
              f"{m['dice']:>7.4f} {m['iou']:>7.4f} {m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} {m['baseline_dice']:>7.4f}")
        save_overlay(img4, gt4, pred, os.path.join(args.out_dir, f"{name}.png"))

    if not rows:
        print("\n没有成功处理的图像")
        return 1
    ds = np.array([r["dice"] for r in rows])
    bs = np.array([r["baseline_dice"] for r in rows])
    print()
    print("=" * 78)
    print(f"全图 Dice 均值 {np.nanmean(ds):.4f}  中位 {np.nanmedian(ds):.4f}  "
          f"范围 [{np.nanmin(ds):.4f}, {np.nanmax(ds):.4f}]")
    print(f"平凡基线均值 {np.nanmean(bs):.4f}；"
          f"超过基线的图数 {int((ds > bs).sum())}/{len(rows)}")
    print(f"叠加图已保存到 {args.out_dir}/")

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows,
                   "summary": {"n": len(rows),
                               "dice_mean": float(np.nanmean(ds)),
                               "dice_median": float(np.nanmedian(ds)),
                               "dice_min": float(np.nanmin(ds)),
                               "dice_max": float(np.nanmax(ds)),
                               "baseline_mean": float(np.nanmean(bs)),
                               "n_beats_baseline": int((ds > bs).sum())}},
                  fh, ensure_ascii=False, indent=1)
    print(f"已写出 {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
