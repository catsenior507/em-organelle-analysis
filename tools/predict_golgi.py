#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_golgi.py —— 用训练好的高尔基体模型做全图推理，产出掩码与叠加图

## 与 predict_all_ourdata.py 的区别

那个脚本是为自有线粒体数据写的，内含两件高尔基体用不上的东西：
1. 按 `annotations_v2/dataset.json` 里的 nm/px **重采样到 4 nm**——
   高尔基体这批插图没有可靠的尺度标定
2. 线粒体专用的输出字段与阈值

因此单独写一个，复用同样的推理方式（`UNet` + 滑窗 + 反射填充），
但在**原分辨率**上推理，不做重采样（尺度未知，宁可不猜）。

## 产出

    <out>/<id>.png          二值掩码（0/255），原分辨率
    <out>/index.csv         每张图的 GT 覆盖率、预测覆盖率、Dice（有标注时）
    <overlays>/<id>.png     原图 + 预测轮廓（红色），用于人工核对

## 必须说清的

* 掩码是**模型预测**，不是标注；面积通常大于标注（部分标注问题）
* `split=val` 的 10 张是训练时留出的，其 Dice 才代表泛化；
  `split=train` 的 Dice 是样本内，**不能**当作精度报告
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# `train_seg.py` 的存放位置在两种环境下不同：
#   * 本机仓库：tools/server/scripts/train_seg.py
#   * 服务器：  <项目>/scripts/train_seg.py
# 早期只加了前者，在服务器上就报 ModuleNotFoundError: No module named 'train_seg'。
for _cand in (os.path.join(ROOT, "scripts"),
              os.path.join(ROOT, "tools", "server", "scripts"),
              os.path.join(ROOT, "tools")):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)
# 让 <repo>/src 可被导入（公开版把训练器与数据集读取器放在 src/）。
_SRC = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch  # noqa: E402
from train_seg import UNet  # noqa: E402

TILE = 256
STRIDE = 192


def predict_full(model, img: np.ndarray, device: str) -> np.ndarray:
    """滑窗推理，返回与原图同尺寸的概率图。"""
    h, w = img.shape
    out = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)

    def starts(total: int) -> List[int]:
        if total <= TILE:
            return [0]
        s = list(range(0, total - TILE + 1, STRIDE))
        if s[-1] != total - TILE:
            s.append(total - TILE)
        return s

    for y0 in starts(h):
        for x0 in starts(w):
            patch = img[y0:y0 + TILE, x0:x0 + TILE].astype(np.float32) / 255.0
            ph, pw = patch.shape
            if ph < TILE or pw < TILE:
                patch = np.pad(patch, ((0, TILE - ph), (0, TILE - pw)), mode="reflect")
            # **必须与训练时的输入一致**：SegDataset 把灰度复制成 3 通道
            # （`np.stack([x, x, x])`），模型 UNet 的 cin 也是 3。
            # 早期这里只喂 1 通道，报
            #   weight of size [32, 3, 3, 3] ... expected 3 channels, got 1
            patch = np.stack([patch, patch, patch], axis=0)
            t = torch.from_numpy(patch)[None].to(device)
            with torch.no_grad():
                p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            out[y0:y0 + ph, x0:x0 + pw] += p[:ph, :pw]
            cnt[y0:y0 + ph, x0:x0 + pw] += 1.0
    return out / np.clip(cnt, 1e-6, None)


def green_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (g > 110) & (g > r + 20) & (g > b + 30)


def tolerant_dice(pred: np.ndarray, gt: np.ndarray, tol: int) -> float:
    """容许 `tol` 像素边界偏差的 Dice。

    ## 为什么需要它

    高尔基体的 GT 是**手画的区域轮廓**，不像线粒体有外膜那样客观的图像边界。
    轮廓画在哪里取决于标注者，因此存在不可消除的边界误差。

    硬 Dice 会把"整体位置对、但边界差 2–3 像素"判成大量错误。
    若把偏差容忍掉之后 Dice 明显上升，说明误差主要来自**边界定义模糊**，
    而不是模型没识别出高尔基体——这是一个重要的定性判断。

    定义（双向容差）：
        pred 中落在 dilate(gt, tol) 内的算 TP
        gt   中落在 dilate(pred, tol) 内的算被覆盖
        TP = |pred ∩ D(gt)|，FP = |pred| − TP，FN = |gt| − |gt ∩ D(pred)|
        Dice = 2TP / (2TP + FP + FN)
    """
    if tol <= 0:
        denom = pred.sum() + gt.sum()
        return float(2 * (pred & gt).sum() / denom) if denom else 0.0
    dg = ndi.binary_dilation(gt, np.ones((3, 3)), iterations=tol)
    dp = ndi.binary_dilation(pred, np.ones((3, 3)), iterations=tol)
    tp = float((pred & dg).sum())
    fp = float(pred.sum()) - tp
    fn = float(gt.sum()) - float((gt & dp).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def fill(o: np.ndarray) -> np.ndarray:
    if not o.any():
        return o
    c = ndi.binary_closing(o, structure=np.ones((3, 3)), iterations=2)
    f = ndi.binary_fill_holes(c)
    return c if f.mean() > 0.6 else f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/golgi_v2/best.pt")
    ap.add_argument("--orig", default="data/golgi/原图")
    ap.add_argument("--annot", default="data/golgi/标注")
    ap.add_argument("--out", default="data/golgi_masks")
    ap.add_argument("--overlays", default="docs/figures/golgi_pred")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tolerance", default="0,1,2,3,5",
                    help="容差 Dice 的边界像素数（逗号分隔）")
    ap.add_argument("--max-imgs", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy")
        return 2
    ckpt = os.path.join(ROOT, args.ckpt)
    if not os.path.exists(ckpt):
        print(f"[错误] 找不到权重 {args.ckpt}")
        return 2
    orig = os.path.join(ROOT, args.orig)
    annot = os.path.join(ROOT, args.annot)
    ids = sorted({os.path.splitext(f)[0] for f in os.listdir(orig)}
                 & {os.path.splitext(f)[0] for f in os.listdir(annot)})
    if args.max_imgs:
        ids = ids[: args.max_imgs]

    # 留出集合：按排序每 5 张取 1 张，与训练时的划分口径一致
    val_ids = {i for k, i in enumerate(ids) if k % 5 == 0}

    model = UNet(3, 1, 32).to(args.device)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"权重 {args.ckpt}  device={args.device}  图像 {len(ids)} 张")
    print(f"留出（val）{len(val_ids)} 张；阈值 {args.thr}")
    print()

    os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
    if args.overlays:
        os.makedirs(os.path.join(ROOT, args.overlays), exist_ok=True)

    rows: List[Dict] = []
    tols = [int(t) for t in str(args.tolerance).split(",") if t.strip()]
    print(f"{'id':<9} {'split':>5} {'GT%':>6} {'pred%':>6} {'Dice':>6} "
          + " ".join(f"{'t'+str(t):>7}" for t in tols))
    print("-" * (42 + 8 * len(tols)))
    for i in ids:
        g = np.asarray(Image.open(os.path.join(orig, i + ".jpg")).convert("L"))
        rgb = np.asarray(Image.open(os.path.join(annot, i + ".jpg")).convert("RGB"))
        gt = fill(green_mask(rgb)) if rgb.shape[:2] == g.shape else None

        prob = predict_full(model, g, args.device)
        pred = prob >= args.thr
        Image.fromarray((pred * 255).astype(np.uint8)).save(
            os.path.join(ROOT, args.out, f"{i}.png"))

        split = "val" if i in val_ids else "train"
        dice = None
        td: Dict[int, float] = {}
        if gt is not None and gt.sum() > 0:
            inter = float((pred & gt).sum())
            denom = float(pred.sum() + gt.sum())
            dice = 2 * inter / denom if denom > 0 else None
            for t in tols:
                td[t] = tolerant_dice(pred, gt, t)
        row = {"id": i, "split": split,
               "gt_frac": float(gt.mean()) if gt is not None else None,
               "pred_frac": float(pred.mean()), "dice": dice}
        for t in tols:
            row[f"dice_t{t}"] = td.get(t)
        rows.append(row)
        print(f"{i:<9} {split:>5} "
              f"{100*(row['gt_frac'] or 0):>5.1f}% "
              f"{100*pred.mean():>5.1f}% "
              f"{(dice if dice is not None else float('nan')):>6.3f} "
              + " ".join(
                  f"{(td.get(t) if td.get(t) is not None else float('nan')):>7.3f}"
                  for t in tols))

        if args.overlays:
            edges = pred & ~ndi.binary_erosion(pred, iterations=2)
            vis = np.stack([g] * 3, -1)
            vis[edges] = [255, 60, 60]
            im = Image.fromarray(vis)
            im.thumbnail((880, 880), Image.LANCZOS)
            im.save(os.path.join(ROOT, args.overlays, f"{i}.png"))

    with open(os.path.join(ROOT, args.out, "index.csv"), "w",
              newline="", encoding="utf-8") as fh:
        fields = ["id", "split", "gt_frac", "pred_frac", "dice"] + \
                 [f"dice_t{t}" for t in tols]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    vals = [r["dice"] for r in rows if r["split"] == "val" and r["dice"] is not None]
    if vals:
        print()
        print(f"留出集硬 Dice 均值 {np.mean(vals):.4f}（{len(vals)} 张），"
              f"中位 {np.median(vals):.4f}")
        print("  ！这是像素级 Dice（全图），与训练日志的 tile 级 Dice 口径不同，")
        print("    不能直接与 0.48 之类的数字比较。")
        print()
        print("边界容差下的 Dice（判断误差是否主要来自边界定义模糊）：")
        for t in tols:
            tv = [r[f"dice_t{t}"] for r in rows
                  if r["split"] == "val" and r.get(f"dice_t{t}") is not None]
            if not tv:
                continue
            gain = np.mean(tv) - np.mean(vals)
            print(f"  容差 {t} px: {np.mean(tv):.4f}  （比硬 Dice {gain:+.4f}）")
        tv3 = [r.get("dice_t3") for r in rows
               if r["split"] == "val" and r.get("dice_t3") is not None]
        if tv3:
            g = np.mean(tv3) - np.mean(vals)
            print()
            if g > 0.15:
                print("  判读：容差 3 px 就让 Dice 大幅上升 -> 误差主要来自**边界定义模糊**，")
                print("        而不是模型没识别出高尔基体。报告应说明这一点。")
            else:
                print("  判读：放宽边界容差收益有限 -> 误差不是边界模糊主导，")
                print("        更可能是模型确实没分准（或数据量不足）。")
    print(f"\n掩码 {os.path.relpath(os.path.join(ROOT, args.out), ROOT)}/")
    if args.overlays:
        print(f"叠加图 {args.overlays}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
