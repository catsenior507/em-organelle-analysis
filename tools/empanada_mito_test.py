#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
empanada_mito_test.py —— 用 empanada 的 MitoNet 在我们的数据上做**有真值**的迁移测试

## 为什么这个测试比之前所有的迁移测试都重要

之前证明 CellMap 的线粒体模型零样本失败（Dice 0.1516）。但那是**单一模态**的模型
（FIB-SEM，小鼠肾/肝）。结论"迁移不行"可能只对那个模型成立。

empanada 的 **MitoNet** 是**通用模型**：编码器在约 150 万张无标注细胞 EM 图像上
自监督预训练（SwAV），再用约 13.5 万个线粒体实例精调。它见过的 EM 成像条件
远比 CellMap 单一数据集丰富。

而且——**线粒体我们是有真值的**。所以这是一次可以**量化**的迁移测试：

  若 MitoNet 在自有数据上 Dice 明显 > 0.15，说明"EM 通用的迁移是可行的"，
  之前的失败应归因于 CellMap 那个模型的模态局限，而不是 EM 迁移本身不可行。
  若 MitoNet 也接近 0.15，说明"跨数据集 EM 迁移难"是普遍现象。

## 用法

    python tools/empanada_mito_test.py --model MitoNet_v1_mini --limit 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import ssl
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "server", "scripts"))

from empanada.inference.engines import PanopticDeepLabEngine  # noqa: E402

MODELS = {
    "MitoNet_v1": {
        "url": "https://zenodo.org/record/6861565/files/MitoNet_v1.pth?download=1",
        "padding": 16, "mean": 0.57571, "std": 0.12765},
    "MitoNet_v1_mini": {
        "url": "https://zenodo.org/record/6861565/files/MitoNet_v1_mini.pth?download=1",
        "padding": 128, "mean": 0.57571, "std": 0.12765},
    "DropNet_base_v1": {
        "url": ("https://zenodo.org/records/15298854/files/"
                "PanopticDeepLabPR_b0d99498-6c99-4b45-844e-2b4dca58f24a_"
                "panoptic_deeplab_pointrend.pth?download=1"),
        "padding": 512, "mean": 0.57571, "std": 0.12765},
    "NucleoNet_base_v2": {
        "url": ("https://zenodo.org/records/18142651/files/"
                "PanopticDeepLabPR_5f023685-7433-404c-9f3d-692898aff9ab_"
                "panoptic_deeplab_pointrend.pth?download=1"),
        "padding": 512, "mean": 0.57571, "std": 0.12765},
}


def download(url: str, dst: str) -> str:
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        print(f"  已有权重 {dst}（{os.path.getsize(dst)/1e6:.1f} MB）")
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    print(f"  下载 {url}")
    with urllib.request.urlopen(url, timeout=120, context=ctx) as r, \
            open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                pct = 100.0 * got / total
                print(f"\r    {got/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)",
                      end="", flush=True)
    print()
    return dst


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


def pad_to(arr: np.ndarray, factor: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = arr.shape[:2]
    ph = (-h) % factor
    pw = (-w) % factor
    if arr.ndim == 2:
        out = np.pad(arr, ((0, ph), (0, pw)), mode="reflect")
    else:
        out = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return out, (h, w)


@torch.no_grad()
def segment(model, img: np.ndarray, cfg: Dict, device: str,
            confidence_thr: float = 0.5) -> np.ndarray:
    """返回与输入同尺寸的布尔掩码（任一实例前景）。"""
    h, w = img.shape
    padded, (oh, ow) = pad_to(img, cfg["padding"])
    x = padded.astype(np.float32)
    x = (x / 255.0 - cfg["mean"]) / cfg["std"]
    t = torch.from_numpy(x)[None, None].to(device)      # (1,1,H,W)

    engine = PanopticDeepLabEngine(
        model, thing_list=[1], label_divisor=1000, stuff_area=64,
        nms_threshold=0.1, nms_kernel=7, confidence_thr=confidence_thr)
    pan = engine(t)
    if isinstance(pan, torch.Tensor):
        pan = pan.cpu().numpy()
    pan = np.asarray(pan).squeeze()
    mask = (pan % 1000) > 0        # 非 0 类别标签即前景
    return mask[:oh, :ow]


def rescale_for_model(img: np.ndarray, gt: np.ndarray, nm: float,
                      target_nm: Optional[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    """把图缩放到模型"习惯"的像素尺度。

    MitoNet 训练时的典型分辨率是 8–30 nm/px，线粒体在其中约 20–100 像素。
    我们的 1.00 nm/px 图像视野只有 2.1 µm，同样大小的线粒体会占到 500 像素，
    比训练分布大一个量级——实测模型在这种图上**完全不输出任何实例**（Dice 0）。
    因此需要先降采样，让线粒体的像素尺寸回到模型见过的范围。
    """
    if not target_nm or target_nm <= 0 or abs(nm - target_nm) < 1e-6:
        return img, gt, 1.0
    factor = nm / target_nm                    # <1 表示缩小
    h, w = img.shape
    nw, nh = max(16, int(round(w * factor))), max(16, int(round(h * factor)))
    i2 = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    g2 = np.asarray(Image.fromarray((gt * 255).astype(np.uint8))
                    .resize((nw, nh), Image.BILINEAR)) > 127
    return i2, g2, factor


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MitoNet_v1_mini", choices=list(MODELS))
    ap.add_argument("--ckpt", default=None,
                    help="直接指定本地权重路径（如微调后的 .pth），"
                         "覆盖 --model 的预设下载地址。"
                         "用途：评估微调模型时必须能指定产物路径")
    ap.add_argument("--images", default="data/原图片")
    ap.add_argument("--masks", default="data/分离的标注信息")
    ap.add_argument("--annot", default="data/annotations_v2/dataset.json")
    ap.add_argument("--cache", default="models/empanada")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--image-list", default=None,
                    help="指定图像名单文件（每行一个）。用于与自有模型做**同台对比**："
                         "两边必须跑完全相同的图，否则比较无意义")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--target-nm", type=float, default=None,
                    help="推理前把图重采样到该像素尺度（nm/px）。"
                         "1.00 nm/px 的图视野只有 2.1 µm，线粒体在像素上比 "
                         "MitoNet 训练分布大一个量级，实测完全不输出实例；"
                         "设成 8 通常能恢复正常")
    ap.add_argument("--invert", action="store_true",
                    help="推理前把图反相。用于判断零结果是「确实没有该结构」"
                         "还是「极性不匹配」：若反相后才出结果，说明极性相反")
    ap.add_argument("--out", default="logs/EMPANADA_MITO.txt")
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy", file=sys.stderr)
        return 2

    cfg = MODELS[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"模型 {args.model}  设备 {device}")

    if args.ckpt:
        path = args.ckpt
        if not os.path.exists(path):
            print(f"[错误] 权重不存在：{path}", file=sys.stderr)
            return 2
        print(f"  使用本地权重 {path}")
    else:
        name = args.model
        dst = os.path.join(args.cache, name + ".pth")
        path = download(cfg["url"], dst)
    model = torch.jit.load(path, map_location=device).eval()
    print(f"  已加载，参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # 选图：按尺度分档交错取样
    meta = json.load(open(args.annot, encoding="utf-8"))
    ents = meta["images"] if isinstance(meta["images"], list) else list(meta["images"].values())
    buckets: Dict[str, List[Tuple[float, str]]] = {}
    for e in ents:
        nm = (e.get("um_per_px") or 0) * 1000
        n = e["image"]
        if not (os.path.exists(os.path.join(args.images, f"{n}.tif"))
                and os.path.exists(os.path.join(args.masks, f"green_{n}.tif"))):
            continue
        buckets.setdefault(f"{nm:.1f}", []).append((nm, n))
    picked: List[Tuple[float, str]] = []
    if args.image_list:
        # 指定名单：保持顺序，只保留存在的
        nm_of = {e["image"]: (e.get("um_per_px") or 0) * 1000 for e in ents}
        wanted = [l.strip() for l in open(args.image_list, encoding="utf-8")
                  if l.strip()]
        for n in wanted:
            if os.path.exists(os.path.join(args.images, f"{n}.tif")) and \
                    os.path.exists(os.path.join(args.masks, f"green_{n}.tif")):
                picked.append((nm_of.get(n, 0.0), n))
        print(f"按名单选取 {len(picked)}/{len(wanted)} 张")
    else:
        i = 0
        keys = sorted(buckets)
        while len(picked) < args.limit:
            added = False
            for k in keys:
                if i < len(buckets[k]) and len(picked) < args.limit:
                    picked.append(buckets[k][i]); added = True
            if not added:
                break
            i += 1

    lines: List[str] = []
    def emit(s=""):
        print(s); lines.append(s)

    emit()
    emit("=" * 72)
    emit(f"empanada {args.model} 在自有数据上的零样本表现（有真值）")
    emit("=" * 72)
    emit(f"{'图像':<10} {'nm/px':>7} {'GT占比':>7} {'预测占比':>8} "
         f"{'Dice':>7} {'P':>6} {'R':>6}")
    emit("-" * 72)

    rows = []
    for nm, n in picked:
        img = load_gray(os.path.join(args.images, f"{n}.tif"))
        o = green_outline(os.path.join(args.masks, f"green_{n}.tif"))
        if o.shape != img.shape:
            continue
        gt = fill_outline(o)
        img_r, gt_r, factor = rescale_for_model(img, gt, nm, args.target_nm)
        if args.invert:
            img_r = (255 - img_r).astype(np.uint8)
        pred = segment(model, img_r, cfg, device, args.thr)
        # 结果缩回原尺寸再比较，避免"在缩放后的网格上算指标"掩盖问题
        if pred.shape != gt.shape:
            pred = np.asarray(
                Image.fromarray(pred.astype(np.uint8) * 255).resize(
                    (gt.shape[1], gt.shape[0]), Image.NEAREST)) > 127
        inter = float((pred & gt).sum())
        ps, gs = float(pred.sum()), float(gt.sum())
        dice = 2 * inter / (ps + gs) if (ps + gs) > 0 else float("nan")
        prec = inter / ps if ps > 0 else float("nan")
        rec = inter / gs if gs > 0 else float("nan")
        base = 2 * gs / (gt.size + gs) if gs > 0 else float("nan")
        rows.append({"image": n, "nm_per_px": nm, "gt_frac": gs / gt.size,
                     "pred_frac": ps / gt.size, "dice": dice,
                     "precision": prec, "recall": rec, "baseline": base})
        emit(f"{n:<10} {nm:>7.2f} {gs/gt.size*100:>6.1f}% "
             f"{ps/gt.size*100:>7.1f}% {dice:>7.4f} {prec:>6.3f} {rec:>6.3f}")

    if rows:
        d = np.array([r["dice"] for r in rows])
        b = np.array([r["baseline"] for r in rows])
        emit()
        emit(f"Dice 均值 {np.nanmean(d):.4f}  中位 {np.nanmedian(d):.4f}  "
             f"范围 [{np.nanmin(d):.4f}, {np.nanmax(d):.4f}]")
        emit(f"平凡基线均值 {np.nanmean(b):.4f}；超过基线 {int((d > b).sum())}/{len(rows)}")
        emit()
        emit("对照：")
        emit("  CellMap 同模态模型零样本        0.1516（0/6 超基线）")
        emit("  自有数据从头训练（留出）        0.7839（18/18）")
        emit()
        if np.nanmean(d) > 0.4:
            emit("=> 通用 EM 模型迁移**有效**：说明之前 CellMap 的失败是那个模型的")
            emit("   模态局限，不是'EM 跨数据集迁移不可行'。")
        elif np.nanmean(d) > 0.25:
            emit("=> 迁移**部分有效**：明显好于 CellMap（0.15），但远低于自有训练（0.78）。")
            emit("   可作为预标注加速人工标注。")
        else:
            emit("=> 迁移**基本无效**：与 CellMap 同量级。说明跨数据集 EM 迁移本身困难，")
            emit("   必须用目标域数据训练。")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        with open(args.out.replace(".txt", ".json"), "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, "rows": rows}, fh,
                      ensure_ascii=False, indent=1)
        print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
