#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_detr.py —— 在验证 crop 上计算**真正的检测指标**（不是 loss）

## 为什么必须做这一步

训练日志里的 `val_loss` 是匈牙利匹配后的综合损失（分类 CE + L1 + GIoU），
它**不能直接解释成"检出了多少细胞器"**。loss 从 5.6 降到 2.5，
既可能是模型变好，也可能只是它学会了"少预测、多报背景"。

要回答"模型能不能用"，必须看：

* **AP@0.5（按类别）**：以 IoU≥0.5 为命中，按置信度排序算平均精度。
  这是检测任务的标准指标，可与文献对比。
* **mAP@0.5**：各类 AP 的平均。
* **Precision / Recall @ 置信度阈值**：给一个可操作的固定阈值，
  告诉使用者"取 conf>0.5 时，检出的里面有多少是真的、漏了多少"。

## 匹配规则

DETR 输出固定数量的框，同一目标可能被多个框覆盖。这里用**贪心匹配**：
按置信度从高到低，每个预测框匹配当前 IoU 最大且未被占用的真值框
（同类），IoU ≥ 阈值算命中，否则算误检。这与 COCO 的评估思路一致
（COCO 用匈牙利匹配求最优，贪心是其常用近似，对小目标数场景差别很小）。

## 用法

    python eval_detr.py --slices data/slices --ckpt runs/detr_kidney/best.pt \
        --classes mito,er --size 480 --conf 0.5 --iou 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_dataset import SliceConfig, SliceTargets, find_slices  # noqa: E402
from train_detr_slices import (SliceDetrDataset, collate,  # noqa: E402
                               group_split)


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a:[N,4] b:[M,4] (x1,y1,x2,y2) -> [N,M] IoU。"""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]))
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0)
              * (a[:, 3] - a[:, 1]).clamp(min=0))
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0)
              * (b[:, 3] - b[:, 1]).clamp(min=0))
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9)


def cxcywh_norm_to_xyxy_px(b: torch.Tensor, w: int, h: int) -> torch.Tensor:
    cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h], dim=1)


def average_precision(rec: np.ndarray, prec: np.ndarray) -> float:
    """AP：PR 曲线下的面积（COCO / pycocotools 的算法）。

    ## 为什么不用 101 点插值

    早期版本用 ``np.interp`` 在 101 个 recall 点上插值。但 recall 常常有**重复值**
    （多个预测落在同一 recall 上），而 ``np.interp`` 遇到重复的 x 时会取**最后一个**
    对应的 y。若最后一个恰好是"recall 已达 1、精确率降为 0"的那个点，
    整段精度就被拉低。

    已知答案测试暴露了这一点：
      * 完美检测（1 个 TP、recall=1、prec=1）应得 AP=1.0，旧实现给 **0.9901**；
      * 1 TP + 1 FP 对 2 个真值，应得 AP=0.5，旧实现给 **0.6213**（高估 24%）。

    现在按 pycocotools 的做法：先做精度包络（向右取最大值），再只在
    **recall 发生变化**的区间上积分，重复 recall 不会引入伪影。
    """
    if rec.size == 0:
        return 0.0
    mrec = np.concatenate([[0.0], np.asarray(rec, dtype=np.float64), [1.0]])
    mpre = np.concatenate([[0.0], np.asarray(prec, dtype=np.float64), [0.0]])
    # 精度包络：从右往左取最大值
    for i in range(mpre.size - 1, 0, -1):
        if mpre[i - 1] < mpre[i]:
            mpre[i - 1] = mpre[i]
    # 只在实际发生 recall 变化的点上积分
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def greedy_match(pred_boxes: torch.Tensor, pred_scores: torch.Tensor,
                 gt_boxes: torch.Tensor, iou_thr: float
                 ) -> Tuple[np.ndarray, int]:
    """按置信度降序贪心匹配。返回 (每个预测是否命中, 命中数)。"""
    n = pred_boxes.shape[0]
    hit = np.zeros(n, dtype=bool)
    if gt_boxes.numel() == 0 or n == 0:
        return hit, 0
    iou = box_iou_xyxy(pred_boxes, gt_boxes)
    used = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    order = torch.argsort(pred_scores, descending=True)
    for i in order.tolist():
        row = iou[i].clone()
        row[used] = -1.0
        best_iou, j = float(row.max()), int(row.argmax())
        if best_iou >= iou_thr:
            hit[i] = True
            used[j] = True
    return hit, int(hit.sum())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DETR 检测指标评估")
    ap.add_argument("--slices", default="data/slices")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--classes", default="mito,er")
    ap.add_argument("--min-area", type=int, default=32)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="结果 JSON 路径")
    args = ap.parse_args(argv)

    os.environ.setdefault("HF_HOME", os.path.expanduser("~/dachuang_organelle/models/hf"))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import DetrForObjectDetection, DetrImageProcessor

    classes = tuple(s.strip() for s in args.classes.split(",") if s.strip())
    cfg = SliceConfig(root=args.slices, classes=classes, min_area=args.min_area)
    records = [r for r in find_slices(args.slices, cfg)
               if SliceTargets.build(r, cfg)["num_instances"] > 0]
    if not records:
        print("[错误] 没有可用切片", file=sys.stderr)
        return 2
    # 与训练**完全相同**的分组划分，保证验证集是同一批 crop
    _, val_recs = group_split(records, args.val_frac, args.seed)
    if not val_recs:
        print("[错误] 验证集为空（crop 太少）", file=sys.stderr)
        return 2
    print(f"验证切片 {len(val_recs)} 个")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck_classes = ck.get("classes") or list(classes)
    if list(ck_classes) != list(classes):
        print(f"[警告] 检查点类别 {ck_classes} 与 --classes {classes} 不一致")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", num_labels=len(classes),
        ignore_mismatched_sizes=True,
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)})
    model.load_state_dict(ck["model"], strict=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    print(f"已加载 {args.ckpt}（epoch {ck.get('epoch')}，"
          f"best_val={ck.get('best_val')}）")

    proc = DetrImageProcessor(size={"shortest_edge": args.size,
                                    "longest_edge": int(args.size * 1.5)})
    ds = SliceDetrDataset(val_recs, cfg, augment=False)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2,
                    collate_fn=collate)

    # 逐类别收集 (score, tp) 用于 AP
    per_class_preds: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(len(classes))}
    n_gt = {i: 0 for i in range(len(classes))}
    conf_stats = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(len(classes))}

    with torch.no_grad():
        for images, labels in dl:
            for img, lb in zip(images, labels):
                enc = proc(images=[img], return_tensors="pt")
                px = enc["pixel_values"].to(dev)
                out = model(pixel_values=px)
                w, h = img.size
                # 统一搬到 CPU：模型输出在 GPU，而标签在 CPU，
                # 混用会抛 "Expected all tensors to be on the same device"。
                # 评估不需要 GPU 张量，全部转 CPU 最简单可靠。
                logits = out.logits[0].detach().cpu()          # [Q, C+1]
                boxes = out.pred_boxes[0].detach().cpu()       # [Q, 4] cxcywh norm
                prob = logits.softmax(-1)
                # 每类：该类的分数 = prob[:, c]；背景为最后一位
                for c in range(len(classes)):
                    cls_gt = (lb["class_labels"] == c)
                    gt_norm = lb["boxes"][cls_gt].cpu()
                    gt_px = cxcywh_norm_to_xyxy_px(gt_norm, w, h)
                    n_gt[c] += int(gt_px.shape[0])
                    scores = prob[:, c]
                    pred_px = cxcywh_norm_to_xyxy_px(boxes, w, h)
                    hit, _n = greedy_match(pred_px, scores, gt_px, args.iou)
                    order = torch.argsort(scores, descending=True)
                    for i in order.tolist():
                        per_class_preds[c].append((float(scores[i]), int(hit[i])))
                    # 固定置信度阈值下的 PR
                    sel = scores >= args.conf
                    if int(sel.sum()):
                        h2, _ = greedy_match(pred_px[sel], scores[sel], gt_px,
                                             args.iou)
                        conf_stats[c]["tp"] += int(h2.sum())
                        conf_stats[c]["fp"] += int((~h2).sum())
                    conf_stats[c]["fn"] += int(max(0, gt_px.shape[0] - int(hit.sum())))

    results = {"ckpt": args.ckpt, "iou_thr": args.iou, "conf": args.conf,
               "n_val_images": len(val_recs), "classes": list(classes),
               "per_class": {}}
    aps = []
    print()
    print(f"{'class':<8} {'n_gt':>6} {'AP@'+str(args.iou):>8} "
          f"{'P@'+str(args.conf):>8} {'R@'+str(args.conf):>8} "
          f"{'TP':>5} {'FP':>5} {'FN':>5}")
    for c, name in enumerate(classes):
        preds = per_class_preds[c]
        preds.sort(key=lambda t: -t[0])
        tp_cum = np.cumsum([p[1] for p in preds]) if preds else np.zeros(0)
        fp_cum = np.cumsum([1 - p[1] for p in preds]) if preds else np.zeros(0)
        if n_gt[c] > 0 and preds:
            rec = tp_cum / float(n_gt[c])
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            ap = average_precision(rec, prec)
        else:
            ap = 0.0
        st = conf_stats[c]
        prec_fix = st["tp"] / max(1, st["tp"] + st["fp"])
        rec_fix = st["tp"] / max(1, st["tp"] + st["fn"])
        aps.append(ap)
        print(f"{name:<8} {n_gt[c]:>6} {ap:>8.4f} {prec_fix:>8.4f} "
              f"{rec_fix:>8.4f} {st['tp']:>5} {st['fp']:>5} {st['fn']:>5}")
        results["per_class"][name] = {
            "n_gt": n_gt[c], "AP": round(ap, 4),
            "precision_at_conf": round(prec_fix, 4),
            "recall_at_conf": round(rec_fix, 4), **st}

    results["mAP"] = round(float(np.mean(aps)), 4) if aps else 0.0
    print(f"\nmAP@{args.iou} = {results['mAP']:.4f}")
    results["caveat"] = (
        f"仅 {len(val_recs)} 个验证切片、"
        f"{len({r['crop'] for r in val_recs})} 个验证 crop，"
        "指标方差很大，不能作为最终结论；且验证 crop 与训练 crop 不同个体/区域，"
        "跨 crop 泛化难度高于同 crop 内插值。")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=1)
        print(f"已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
