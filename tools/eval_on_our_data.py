#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_on_our_data.py —— 把 CellMap 训练的线粒体分割模型用到**我们自己的 EM 图上**

## 目的

到目前为止所有实验都在公开数据（CellMap kidney/liver）上。项目真正要回答的是
"这个模型能不能用于我们自己的 89 张图"。本脚本做这件事：

1. 从 `data/分离的标注信息/green_*.tif` 提取绿色标注描边，`binary_fill_holes` 填充
   得到真值掩码（这正是 `data/annotations_v2/dataset.json` 里
   `segmentation: "binary_fill_holes(green_outline)"` 的约定；`organelle_type`
   543/570 为 "M"，即线粒体）
2. 按 `um_per_px` 把图与掩码重采样到 **8 nm/px**（与 CellMap 训练尺度一致）
3. 分块送入在 CellMap kidney 上训练的 U-Net，拼接输出
4. 与真值算 Dice / IoU / P / R

## ⚠ 必须一起读的局限

**我们的标注是"部分标注"**：每张图只有约 6.4 个被勾勒的线粒体，
图里其它线粒体**没有被标为背景**。若在全图上算 Dice，这些未标的目标会被算成
假阳性，指标会被人为压低。

因此本脚本只在**标注区域的邻域 ROI** 内评估：
以所有已标注区域的并集为基础，向外扩张 `--roi-margin` 像素，
只在 ROI 内统计。ROI 之外一律不计入任何指标。

这是部分标注下的标准做法，但它衡量的是"在标注过的区域附近，模型预测得准不准"，
**不是**"整张图上分割得多好"。报告结论时必须写明这一点。

## 用法

    python tools/eval_on_our_data.py \
        --ckpt models/cellmap_mito/best.pt \
        --images data/原图片 --masks data/分离的标注信息 \
        --annot data/annotations_v2/dataset.json \
        --limit 12 --roi-margin 64 --out docs/OURDATA-RESULTS.md
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
import torch.nn as nn
import torch.nn.functional as F

TARGET_NM = 8.0          # CellMap 训练尺度
TILE = 256               # 与训练时一致
STRIDE = 192


# --------------------------------------------------------------------------
# 模型结构：**直接复用训练脚本里的定义**
#
# 教训：这里最初我凭记忆重写了一份 UNet，结果 load_state_dict 直接失败——
# 检查点里的键是 d4/c4/d3/c3/u4/u3/u2 这一套，而我写的是 e1/e2/d3/d2/net.* 一套。
# 重实现一个已有类既没必要又危险（哪怕结构"等价"，命名不同也加载不了；
# 若命名碰巧相同而结构略异，会造成更隐蔽的错误）。
# 因此改为 import 训练脚本的 UNet 与 double_conv，保证与检查点逐层一致。
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(HERE, "server", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
# 让 <repo>/src 可被导入（公开版把训练器与数据集读取器放在 src/）。
_SRC = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from train_seg import UNet  # noqa: E402


# --------------------------------------------------------------------------
# 真值提取
# --------------------------------------------------------------------------
def load_gray(path: str) -> np.ndarray:
    """读调色板 TIFF 并转灰度（原图 mode=P，但调色板是灰阶斜坡）。"""
    im = Image.open(path)
    return np.asarray(im.convert("L"), dtype=np.uint8)


def green_outline_mask(path: str) -> np.ndarray:
    """从分离出的绿色描边图提取二值描边。

    绿色描边图为 RGB，非零像素的三元组为 (0,75,180) 一类，
    这里用"G 明显大于 R"来判定，避免把黑色背景算进来。
    """
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    return (g > 40) & (g > r + 20)


def fill_outline(outline: np.ndarray) -> np.ndarray:
    """binary_fill_holes：把闭合描边填成实心区域。

    描边可能有细小缺口，so 先做一次闭运算再填洞，减少漏填。
    若填充后前景占比异常（>60%），说明描边不闭合，退回闭运算结果并标记。
    """
    if not outline.any():
        return outline
    closed = ndi.binary_closing(outline, structure=np.ones((3, 3)), iterations=2)
    filled = ndi.binary_fill_holes(closed)
    if filled.mean() > 0.6:
        return closed
    return filled


def build_roi(gt: np.ndarray, margin: int) -> np.ndarray:
    """标注区域向外扩张 margin 得到评估 ROI。"""
    if not gt.any():
        return gt
    roi = ndi.binary_dilation(gt, iterations=margin)
    return roi


# --------------------------------------------------------------------------
# 重采样到目标尺度
# --------------------------------------------------------------------------
def resample_to_nm(img: np.ndarray, mask: np.ndarray, nm_per_px: float,
                   target_nm: float = TARGET_NM
                   ) -> Tuple[np.ndarray, np.ndarray, float]:
    """把图与掩码缩放到 target_nm/px。返回 (图, 掩码, 实际缩放因子)。

    我们的数据都比 8 nm 细（1.0 / 2.7 / 5.3 nm/px），所以要**降采样**。
    缩放因子 = nm_per_px / target_nm（<1 表示缩小）。
    """
    factor = float(nm_per_px) / float(target_nm)
    if abs(factor - 1.0) < 1e-3:
        return img, mask, 1.0
    h, w = img.shape
    nh, nw = max(8, int(round(h * factor))), max(8, int(round(w * factor)))
    im2 = Image.fromarray(img).resize((nw, nh), Image.BILINEAR)
    # 掩码用 NEAREST，避免插值产生灰边；但降采样会丢失细目标，
    # 因此掩码改用"最大池化"式的面积占优，更接近真实的覆盖
    mk = Image.fromarray((mask * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    m2 = (np.asarray(mk) > 127)
    return np.asarray(im2, dtype=np.uint8), m2, factor


# --------------------------------------------------------------------------
# 分块推理
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_tiled(model: nn.Module, img: np.ndarray, thr: float,
                  tile: int = TILE, stride: int = STRIDE) -> np.ndarray:
    """滑窗推理并拼接。边缘处用重叠平均，避免接缝。"""
    h, w = img.shape
    prob = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    x = img.astype(np.float32) / 255.0
    # 复制成 3 通道（训练时是 RGB 三通道输入）
    x3 = np.stack([x, x, x], axis=0)

    def starts(total: int) -> List[int]:
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
            t = torch.from_numpy(patch).unsqueeze(0)
            out = torch.sigmoid(model(t))[0, 0].numpy()
            out = out[:ph, :pw]
            prob[y0:y0 + ph, x0:x0 + pw] += out
            cnt[y0:y0 + ph, x0:x0 + pw] += 1.0
    cnt[cnt == 0] = 1.0
    return prob / cnt


def dice_iou_p_r(pred: np.ndarray, gt: np.ndarray, roi: np.ndarray
                 ) -> Dict[str, float]:
    p = pred & roi
    g = gt & roi
    inter = float((p & g).sum())
    ps, gs = float(p.sum()), float(g.sum())
    dice = 2 * inter / (ps + gs) if (ps + gs) > 0 else float("nan")
    iou = inter / (ps + gs - inter) if (ps + gs - inter) > 0 else float("nan")
    prec = inter / ps if ps > 0 else float("nan")
    rec = inter / gs if gs > 0 else float("nan")
    return {"dice": dice, "iou": iou, "precision": prec, "recall": rec,
            "pred_px": ps, "gt_px": gs, "roi_px": float(roi.sum())}


def trivial_baseline(gt: np.ndarray, roi: np.ndarray) -> Dict[str, float]:
    """ROI 内"全预测前景"的平凡基线 Dice = 2*gt/(roi+gt)。"""
    g = float((gt & roi).sum())
    r = float(roi.sum())
    all1 = 2 * g / (r + g) if (r + g) > 0 else float("nan")
    return {"dice_all_foreground": all1, "gt_frac_in_roi": g / r if r else float("nan")}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/cellmap_mito/best.pt")
    ap.add_argument("--images", default="data/原图片")
    ap.add_argument("--masks", default="data/分离的标注信息")
    ap.add_argument("--annot", default="data/annotations_v2/dataset.json")
    ap.add_argument("--limit", type=int, default=12, help="最多评估多少张图")
    ap.add_argument("--roi-margin", type=int, default=64)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--thr-sweep", default="0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--invert", action="store_true",
                    help="先把图反相（255-x）再送模型。"
                         "原因：CellMap 的线粒体比背景**暗**（实测 EM 灰度 AUC 0.163，"
                         "即更暗），而我们自己的图里线粒体比背景**亮**"
                         "（GT 内均值比周围高 +7 到 +34 灰度）。"
                         "对比度方向相反会让模型系统性失效，反相后应与训练分布一致")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy", file=sys.stderr)
        return 2

    meta = json.load(open(args.annot, encoding="utf-8"))
    # dataset.json 的 images 是 list
    entries = meta["images"] if isinstance(meta["images"], list) else list(meta["images"].values())
    by_name = {e["image"]: e for e in entries}

    # 选图：优先有 scale 信息的、按 nm/px 分组各取一些，保证覆盖三种尺度
    cand = []
    for name, e in by_name.items():
        nm = (e.get("um_per_px") or 0) * 1000.0
        if nm <= 0:
            continue
        g = os.path.join(args.masks, f"green_{name}.tif")
        o = os.path.join(args.images, f"{name}.tif")
        if os.path.exists(g) and os.path.exists(o):
            cand.append((nm, name))
    if not cand:
        print("[错误] 找不到配对的 原图/绿色标注", file=sys.stderr)
        return 2
    # 按尺度分档交错取样，避免只测到一种尺度
    buckets: Dict[str, List[Tuple[float, str]]] = {}
    for nm, name in cand:
        buckets.setdefault(f"{nm:.1f}", []).append((nm, name))
    picked: List[Tuple[float, str]] = []
    keys = sorted(buckets)
    i = 0
    while len(picked) < args.limit:
        added = False
        for k in keys:
            if i < len(buckets[k]) and len(picked) < args.limit:
                picked.append(buckets[k][i])
                added = True
        if not added:
            break
        i += 1

    print(f"配对可用 {len(cand)} 张；本次评估 {len(picked)} 张")
    print(f"尺度分档：{ {k: len(v) for k, v in buckets.items()} }")

    model = UNet(3, 2, args.base)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"检查点 epoch {ck.get('epoch')}  参数量 "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    rows = []
    print()
    print(f"{'图像':<10} {'nm/px':>7} {'缩放':>6} {'ROI占比':>8} "
          f"{'Dice':>7} {'IoU':>7} {'P':>6} {'R':>6} {'基线':>7}")
    print("-" * 78)
    for nm, name in picked:
        img = load_gray(os.path.join(args.images, f"{name}.tif"))
        outl = green_outline_mask(os.path.join(args.masks, f"green_{name}.tif"))
        if outl.shape != img.shape:
            print(f"  [跳过] {name} 形状不一致 {outl.shape} vs {img.shape}")
            continue
        gt = fill_outline(outl)
        roi_full = build_roi(gt, args.roi_margin)
        img8, gt8, factor = resample_to_nm(img, gt, nm)
        if args.invert:
            img8 = (255 - img8).astype(np.uint8)
        # ROI 也要按同样因子缩放
        roi8 = np.asarray(Image.fromarray((roi_full * 255).astype(np.uint8))
                          .resize((img8.shape[1], img8.shape[0]), Image.NEAREST)) > 127
        prob = predict_tiled(model, img8, args.thr)
        pred = prob > args.thr
        m = dice_iou_p_r(pred, gt8, roi8)
        tb = trivial_baseline(gt8, roi8)
        rows.append({"image": name, "nm_per_px": nm, "factor": factor,
                     "roi_frac": m["roi_px"] / float(roi8.size),
                     **{k: m[k] for k in ("dice", "iou", "precision", "recall")},
                     "baseline_dice": tb["dice_all_foreground"],
                     "gt_frac_in_roi": tb["gt_frac_in_roi"],
                     "prob_mean_in_roi": float(prob[roi8].mean()),
                     "prob_mean_gt": float(prob[gt8].mean()) if gt8.any() else None})
        print(f"{name:<10} {nm:>7.2f} {factor:>6.2f} "
              f"{m['roi_px']/roi8.size:>7.1%} {m['dice']:>7.4f} {m['iou']:>7.4f} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{tb['dice_all_foreground']:>7.4f}")

    if not rows:
        print("\n没有成功评估的图像")
        return 1

    ds = np.array([r["dice"] for r in rows])
    bs = np.array([r["baseline_dice"] for r in rows])
    print()
    print("=" * 78)
    print(f"合计 {len(rows)} 张：Dice 均值 {np.nanmean(ds):.4f} "
          f"中位 {np.nanmedian(ds):.4f} 范围 [{np.nanmin(ds):.4f}, {np.nanmax(ds):.4f}]")
    print(f"平凡基线均值 {np.nanmean(bs):.4f}；"
          f"超过基线的图数 {int((ds > bs).sum())}/{len(rows)}")
    print(f"各类内平均预测概率（应明显高于 ROI 背景）：")
    for r in rows[:5]:
        print(f"  {r['image']}: GT 内 {r['prob_mean_gt']:.3f} vs ROI 平均 {r['prob_mean_in_roi']:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows,
                       "summary": {
                           "n": len(rows),
                           "dice_mean": float(np.nanmean(ds)),
                           "dice_median": float(np.nanmedian(ds)),
                           "dice_min": float(np.nanmin(ds)),
                           "dice_max": float(np.nanmax(ds)),
                           "baseline_mean": float(np.nanmean(bs)),
                           "n_beats_baseline": int((ds > bs).sum())}},
                      fh, ensure_ascii=False, indent=1)
        print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
