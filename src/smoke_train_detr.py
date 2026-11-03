#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_train_detr.py —— 用**真实切片**在 GPU 上跑几步 DETR 训练，验证链路通

## 这一步要证明什么

不是证明"模型训好了"（13 个切片训不出任何东西），而是证明：

1. 真实 EM + 标签能变成 DETR 需要的 `(pixel_values, labels)`；
2. 44M 参数的 DETR-R50 能在 RTX 3080 上完成前向 + 匈牙利匹配 + 反向 + 优化器步；
3. 换掉 COCO 的 91 类分类头、改成 2 类（mito/er）之后不报错；
4. 显存占用落在 10 GB 以内，且能估出"正式训练时 batch 能开到多大"。

## 为什么用 HF 内置 loss

`DetrForObjectDetection` 在传入 `labels` 时直接返回 `loss`（内部的匈牙利匹配 +
分类 CE + L1 + GIoU）。这样冒烟测试的代码量最小、语义与正式训练一致；
本项目自己的 `tools/detr/losses.py` 是同一套逻辑的实现，正式训练时可二选一。

## 用法

    python smoke_train_detr.py --slices data/slices --iters 6 --size 480
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_dataset import (SliceConfig, SliceTargets, find_slices,  # noqa: E402
                           boxes_to_cxcywh_norm, instances_to_boxes_xyxy,
                           mask_to_instances)


def load_sample(rec: Dict[str, Any], cfg: SliceConfig) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
    """读一个切片，返回 (PIL 图像, HF 格式的 labels)。"""
    img = Image.open(rec["image"]).convert("RGB")
    t = SliceTargets.build(rec, cfg)
    labels = {
        "class_labels": torch.as_tensor(t["labels"], dtype=torch.long),
        "boxes": torch.as_tensor(t["boxes_cxcywh"], dtype=torch.float32),
        # area 不是 HF 必需字段，但保留便于调试
        "area": torch.as_tensor(
            (t["boxes_cxcywh"][:, 2] * t["boxes_cxcywh"][:, 3]
             if t["boxes_cxcywh"].size else np.zeros(0)), dtype=torch.float32),
    }
    return img, labels


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DETR 训练冒烟测试（真实切片）")
    ap.add_argument("--slices", required=True)
    ap.add_argument("--classes", default="mito,er")
    ap.add_argument("--min-area", type=int, default=32)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--size", type=int, default=480, help="短边像素")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--model", default="facebook/detr-resnet-50")
    args = ap.parse_args(argv)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import DetrForObjectDetection, DetrImageProcessor

    cfg = SliceConfig(root=args.slices,
                      classes=tuple(s.strip() for s in args.classes.split(",")),
                      min_area=args.min_area)
    slices = find_slices(args.slices, cfg)
    print(f"切片 {len(slices)} 个，类别 {cfg.classes}")
    if not slices:
        print("[错误] 没有切片", file=sys.stderr)
        return 2

    # 只保留至少有一个实例的切片（训练需要真值）
    samples = []
    for rec in slices:
        img, labels = load_sample(rec, cfg)
        if labels["class_labels"].numel() == 0:
            continue
        samples.append((rec["image_id"], img, labels))
    print(f"有实例的切片 {len(samples)} 个；实例数 "
          f"{ {c: sum(int((s[2]['class_labels'] == i).sum()) for s in samples) for i, c in enumerate(cfg.classes)} }")
    if not samples:
        print("[错误] 没有任何切片含实例", file=sys.stderr)
        return 2

    proc = DetrImageProcessor(size={"shortest_edge": args.size,
                                    "longest_edge": int(args.size * 1.5)})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    if dev == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB")

    model = DetrForObjectDetection.from_pretrained(
        args.model,
        num_labels=len(cfg.classes),
        ignore_mismatched_sizes=True,
    ).to(dev)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量 {n_par:.1f} M  分类头 num_labels={len(cfg.classes)}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    model.train()

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    losses: List[float] = []
    t0 = time.time()
    order = list(range(len(samples))) * (args.iters // max(1, len(samples)) + 2)
    step = 0
    for i in range(0, len(order), args.batch):
        if step >= args.iters:
            break
        batch = [samples[j % len(samples)] for j in order[i:i + args.batch]]
        try:
            enc = proc(images=[b[1] for b in batch], return_tensors="pt")
            px = enc["pixel_values"].to(dev)
            labels = [{k: v.to(dev) for k, v in b[2].items()} for b in batch]
            out = model(pixel_values=px, labels=labels)
            loss = out.loss
        except Exception as exc:  # noqa: BLE001
            print(f"[错误] 第 {step} 步前向失败：{type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return 3
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 0.1)
        opt.step()
        losses.append(float(loss.detach()))
        step += 1
        mem = (torch.cuda.max_memory_allocated() / 1024**2) if dev == "cuda" else 0
        print(f"  step {step:>2}  loss {losses[-1]:.4f}  "
              f"grad_norm(before clip) {float(gn):.4f}  "
              f"峰值显存 {mem:.0f} MiB  ({time.time()-t0:.0f}s)", flush=True)

    print()
    print(f"完成 {step} 步。loss: 首 {losses[0]:.4f} -> 末 {losses[-1]:.4f}  "
          f"min {min(losses):.4f}")
    if dev == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"峰值显存 {peak:.0f} / {total:.0f} MiB "
              f"({100*peak/total:.1f}%)")
        est = int(total * 0.85 / max(peak, 1))
        print(f"按 85% 显存估算，此配置（size={args.size}, batch={args.batch}）"
              f"可放大到约 {est} 倍 batch")
    print("结论：训练链路（数据->前向->匈牙利匹配损失->反向->优化器）可用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
