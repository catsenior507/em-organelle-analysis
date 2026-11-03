#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_detr_slices.py —— 在 CellMap 2D 切片上微调 DETR（多类别）

## 数据划分：必须按 crop 分组

同一个 crop 的相邻 z 层几乎是同一张图（电镜连续切片，层间距 6 个体素）。
若按切片随机划分，验证集里会混入训练集的近邻，指标严重虚高。
因此划分单位是 **crop**，同一 crop 的所有层只出现在一边。

## 超参选择依据（均为本机实测，不是抄默认值）

* 显存：batch=2 / 480px 实测峰值 1123 MiB，10 GB 卡按 85% 可到 7–9 倍
  → 默认 batch=8、size=480
* 梯度：未裁剪时 grad_norm 达 1372 → **必须裁剪**，默认 0.1
* 学习率：DETR 微调惯例是主干比头小一个量级 → `--backbone-lr` 默认 lr/10

## 损失

用 transformers 内置的 `DetrForObjectDetection` 在传入 labels 时返回的 loss，
即官方的匈牙利匹配 + 分类 CE + L1 + GIoU。本项目 `tools/detr/losses.py` 是
同一套逻辑的自实现，正式对比实验时可切换。

## 用法

    python train_detr_slices.py --slices data/slices --classes mito,er \
        --epochs 150 --batch 8 --size 480 \
        --out runs/detr_kidney --resume auto
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_dataset import (SliceConfig, SliceTargets, find_slices)  # noqa: E402


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------

class SliceDetrDataset(Dataset):
    """读 `em.png` + 每类掩码，返回 (PIL RGB 图, HF 格式 labels)。

    增强只做**水平/垂直翻转**：对归一化 cxcywh 框有精确的解析变换，
    不会引入坐标错误。旋转 90° 虽然也能解析变换，但实现时容易把 w/h 写反，
    在没有单测覆盖前先不开。
    """

    def __init__(self, records: List[Dict[str, Any]], cfg: SliceConfig,
                 augment: bool = False, seed: int = 0):
        self.records = records
        self.cfg = cfg
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
        rec = self.records[i]
        img = Image.open(rec["image"]).convert("RGB")
        t = SliceTargets.build(rec, self.cfg)
        boxes = torch.as_tensor(t["boxes_cxcywh"], dtype=torch.float32).clone()
        labels = torch.as_tensor(t["labels"], dtype=torch.long).clone()

        if self.augment and boxes.numel():
            if self.rng.random() < 0.5:                 # 水平翻转
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                boxes[:, 0] = 1.0 - boxes[:, 0]
            if self.rng.random() < 0.5:                 # 垂直翻转
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                boxes[:, 1] = 1.0 - boxes[:, 1]

        return img, {"class_labels": labels, "boxes": boxes}


def collate(batch):
    """返回 list（每样本尺寸可能不同，交给 processor 统一 resize）。"""
    return [b[0] for b in batch], [b[1] for b in batch]


def group_split(records: List[Dict[str, Any]], val_frac: float,
                seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按 dataset/crop 分组划分，同一 crop 不会同时出现在两边。"""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(f"{r['dataset']}/{r['crop']}", []).append(r)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_frac))) if len(keys) > 1 else 0
    val_keys = set(keys[:n_val])
    train, val = [], []
    for k in keys:
        (val if k in val_keys else train).extend(groups[k])
    print(f"分组划分：{len(groups)} 个 crop -> 训练 {len(groups)-len(val_keys)} 个 "
          f"({len(train)} 切片) / 验证 {len(val_keys)} 个 ({len(val)} 切片)")
    if val_keys:
        print(f"  验证集 crop：{sorted(val_keys)}")
    return train, val


# --------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------

def make_optimizer(model, lr: float, backbone_lr: float, wd: float):
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if "backbone" in name else head).append(p)
    groups = [
        {"params": backbone, "lr": backbone_lr},
        {"params": head, "lr": lr},
    ]
    print(f"优化器：主干 {len(backbone)} 组参数 lr={backbone_lr:g}；"
          f"头 {len(head)} 组参数 lr={lr:g}")
    return torch.optim.AdamW(groups, weight_decay=wd)


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def run_epoch(model, loader, proc, dev, opt=None, *,
              max_steps: Optional[int] = None,
              base_lrs: Optional[List[float]] = None,
              warmup: int = 0, total: int = 1, gstep_start: int = 0,
              clip: float = 0.1) -> Tuple[float, int]:
    """跑一个 epoch。opt 为 None 时是验证。返回 (平均 loss, 步数)。"""
    training = opt is not None
    model.train(training)
    tot, n = 0.0, 0
    for images, labels in loader:
        enc = proc(images=images, return_tensors="pt")
        px = enc["pixel_values"].to(dev, non_blocking=True)
        tgt = [{k: v.to(dev) for k, v in lb.items()} for lb in labels]
        with torch.set_grad_enabled(training):
            out = model(pixel_values=px, labels=tgt)
            loss = out.loss
        if training:
            if base_lrs:
                f = lr_at(gstep_start + n, total, warmup, 1.0)
                for gi, g in enumerate(opt.param_groups):
                    g["lr"] = base_lrs[gi] * f
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], clip)
            opt.step()
        tot += float(loss.detach())
        n += 1
        if max_steps and n >= max_steps:
            break
    return (tot / max(1, n)), n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="在 CellMap 切片上微调 DETR")
    ap.add_argument("--slices", default="data/slices")
    ap.add_argument("--val-slices", default=None,
                    help="独立验证集目录（如另一个器官）。给定时训练用 --slices 的"
                         "全部 crop，验证用 --val-slices 的全部 crop，"
                         "即**跨数据集验证**——比同组织内随机切分更能说明泛化能力")
    ap.add_argument("--include-crops", default=None,
                    help="只保留这些 crop（逗号分隔）。域适应实验要用："
                         "把同一器官的 crop 拆成『适应用』与『留出评估用』两组")
    ap.add_argument("--exclude-crops", default=None,
                    help="排除这些 crop（逗号分隔）")
    ap.add_argument("--classes", default="mito,er")
    ap.add_argument("--min-area", type=int, default=32)
    ap.add_argument("--out", default="runs/detr_slices")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=None,
                    help="主干学习率，默认 lr/10")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--model", default="facebook/detr-resnet-50")
    ap.add_argument("--resume", default="none",
                    help="none | auto | <checkpoint.pt>")
    ap.add_argument("--save-every", type=int, default=50,
                    help="每 N 个 epoch 存一份快照（0=不存）。"
                         "原因：val loss 与检测 AP 不一定同步，"
                         "事后需要按 AP 在多个快照里挑最优权重")
    ap.add_argument("--max-train-batches", type=int, default=0,
                    help="调试用：每 epoch 只跑这么多步（0=全部）")
    ap.add_argument("--time-budget-min", type=float, default=0,
                    help="调试用：到点后提前结束并保存")
    args = ap.parse_args(argv)

    os.environ.setdefault("HF_HOME", os.path.expanduser("~/dachuang_organelle/models/hf"))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    from transformers import DetrForObjectDetection, DetrImageProcessor

    classes = tuple(s.strip() for s in args.classes.split(",") if s.strip())
    cfg = SliceConfig(root=args.slices, classes=classes, min_area=args.min_area)
    records = find_slices(args.slices, cfg)
    print(f"切片总数 {len(records)}，类别 {classes}")
    if not records:
        print("[错误] 没有切片", file=sys.stderr)
        return 2

    # 按 crop 筛选（域适应实验用：把同一器官拆成"适应用"与"留出评估用"）
    if args.include_crops or args.exclude_crops:
        def _norm(s: str) -> str:
            s = s.strip()
            return s if s.startswith("crop") else f"crop{s}"
        inc = ({_norm(x) for x in args.include_crops.split(",") if x.strip()}
               if args.include_crops else None)
        exc = ({_norm(x) for x in args.exclude_crops.split(",") if x.strip()}
               if args.exclude_crops else set())
        before = len(records)
        records = [r for r in records
                   if (inc is None or r["crop"] in inc) and r["crop"] not in exc]
        print(f"按 crop 筛选：{before} -> {len(records)} 个切片"
              + (f"（保留 {sorted(inc)}）" if inc else "")
              + (f"（排除 {sorted(exc)}）" if exc else ""))
        if not records:
            print("[错误] 筛选后没有切片", file=sys.stderr)
            return 2

    # 只保留有实例的切片
    kept = []
    for rec in records:
        t = SliceTargets.build(rec, cfg)
        if t["num_instances"] > 0:
            kept.append(rec)
    print(f"有实例的切片 {len(kept)} / {len(records)}")
    if not kept:
        print("[错误] 没有含实例的切片", file=sys.stderr)
        return 2

    # 注意：`group_split` 只在**没有** --val-slices 时才调用。
    # 早期版本先无条件调用一次，于是日志里会出现
    # "分组划分：... 验证 3 个 (43 切片)" 然后又打印
    # "跨数据集验证：... 验证 crop 数 15" 的矛盾输出——
    # 前者被后者覆盖，功能上没问题，但日志极容易误读（我自己就被误导过一次，
    # 以为 --val-slices 没生效）。
    if args.val_slices:
        vcfg = SliceConfig(root=args.val_slices, classes=classes,
                           min_area=args.min_area)
        vrecs = find_slices(args.val_slices, vcfg)
        vkept = [r for r in vrecs
                 if SliceTargets.build(r, vcfg)["num_instances"] > 0]
        print(f"跨数据集验证：{args.val_slices} 有 {len(vkept)}/{len(vrecs)} 个含实例切片")
        if not vkept:
            print("[错误] --val-slices 下没有可用切片", file=sys.stderr)
            return 2
        train_recs, val_recs = kept, vkept
        cfg_v = vcfg
        print(f"训练 crop 数 {len({r['crop'] for r in kept})}，"
              f"验证 crop 数 {len({r['crop'] for r in vkept})}")
    else:
        train_recs, val_recs = group_split(kept, args.val_frac, args.seed)
        cfg_v = cfg
    ds_tr = SliceDetrDataset(train_recs, cfg, augment=True, seed=args.seed)
    ds_va = SliceDetrDataset(val_recs, cfg_v, augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.num_workers, collate_fn=collate,
                       drop_last=False, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                       num_workers=args.num_workers, collate_fn=collate,
                       pin_memory=True)

    proc = DetrImageProcessor(size={"shortest_edge": args.size,
                                    "longest_edge": int(args.size * 1.5)})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    if dev == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"  {p.name}  {p.total_memory/1024**3:.1f} GiB")

    # 传入 id2label/label2id：否则 transformers 会警告
    # "num_labels=2 incompatible to id2label map of length 91"，
    # 且保存的 config 里类别名仍是 COCO 的，后续推理会误导。
    id2label = {i: c for i, c in enumerate(classes)}
    label2id = {c: i for i, c in enumerate(classes)}

    ckpt_path = os.path.join(args.out, "last.pt")
    best_path = os.path.join(args.out, "best.pt")
    os.makedirs(args.out, exist_ok=True)

    start_epoch = 0
    resume_from = None
    if args.resume == "auto":
        if os.path.exists(ckpt_path):
            resume_from = ckpt_path
        else:
            print("--resume auto：没有找到 last.pt，从头开始")
    elif args.resume not in ("none", ""):
        resume_from = args.resume

    if resume_from and os.path.exists(resume_from):
        print(f"从 {resume_from} 恢复")
        ck = torch.load(resume_from, map_location="cpu", weights_only=False)
        model = DetrForObjectDetection.from_pretrained(
            args.model, num_labels=len(classes), ignore_mismatched_sizes=True,
            id2label=id2label, label2id=label2id)
        model.load_state_dict(ck["model"], strict=False)
        start_epoch = int(ck.get("epoch", -1)) + 1
        best_val = float(ck.get("best_val", float("inf")))
        print(f"  已恢复到 epoch {start_epoch}（best_val={best_val:.4f}）")
        model = model.to(dev)
    else:
        model = DetrForObjectDetection.from_pretrained(
            args.model, num_labels=len(classes),
            ignore_mismatched_sizes=True,
            id2label=id2label, label2id=label2id).to(dev)
        best_val = float("inf")

    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型 {n_par:.1f} M 参数，num_labels={len(classes)}")

    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr / 10.0
    opt = make_optimizer(model, args.lr, backbone_lr, args.wd)
    base_lrs = [g["lr"] for g in opt.param_groups]

    steps_per_epoch = len(dl_tr)
    if args.max_train_batches:
        steps_per_epoch = min(steps_per_epoch, args.max_train_batches)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * args.warmup_frac))
    print(f"每 epoch {steps_per_epoch} 步，共 {total_steps} 步，warmup {warmup} 步")

    hist: List[Dict[str, Any]] = []
    hist_path = os.path.join(args.out, "history.json")
    if resume_from and os.path.exists(hist_path):
        try:
            hist = json.load(open(hist_path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            hist = []

    t_start = time.time()
    gstep = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        tr_loss, ntr = run_epoch(
            model, dl_tr, proc, dev, opt,
            max_steps=args.max_train_batches or None,
            base_lrs=base_lrs, warmup=warmup, total=total_steps,
            gstep_start=gstep, clip=args.clip)
        gstep += ntr
        va_loss = float("nan")
        if val_recs:
            va_loss, _ = run_epoch(model, dl_va, proc, dev, None,
                                   max_steps=args.max_train_batches or None)
        lr_now = opt.param_groups[-1]["lr"]
        dt = time.time() - t_start
        rec = {"epoch": epoch, "train_loss": round(tr_loss, 5),
               "val_loss": (round(va_loss, 5) if val_recs else None),
               "lr": lr_now, "elapsed_s": round(dt, 1)}
        hist.append(rec)
        print(f"epoch {epoch+1}/{args.epochs}  train {tr_loss:.4f}  "
              f"val {va_loss:.4f}  lr {lr_now:.2e}  {dt:.0f}s", flush=True)

        improved = (not val_recs) or (va_loss < best_val)
        if improved:
            best_val = va_loss if val_recs else tr_loss
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "best_val": best_val, "classes": list(classes),
                    "args": vars(args)}, ckpt_path)
        if improved:
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "best_val": best_val, "classes": list(classes),
                        "args": vars(args)}, best_path)
            print(f"  -> 保存 best.pt (val {best_val:.4f})")
        if args.save_every and (epoch + 1) % args.save_every == 0:
            snap = os.path.join(args.out, f"epoch{epoch+1:04d}.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "best_val": best_val, "classes": list(classes),
                        "args": vars(args)}, snap)
            print(f"  -> 快照 {os.path.basename(snap)}")
        with open(hist_path, "w", encoding="utf-8") as fh:
            json.dump(hist, fh, ensure_ascii=False, indent=1)

        if args.time_budget_min and (time.time() - t_start) / 60 >= args.time_budget_min:
            print(f"达到时间预算 {args.time_budget_min} 分钟，提前结束")
            break

    print(f"\n完成。最优验证 loss {best_val:.4f}，检查点在 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
