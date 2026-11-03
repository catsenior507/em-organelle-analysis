#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_ckpt.py —— 对同一批验证切片，逐个评估多个检查点的检测 AP

## 为什么不能用 14 个进程各跑一遍 eval_detr.py

每次启动都要重新 from_pretrained（加载 COCO 权重）+ 建 DataLoader，
14 份权重就要重复 14 次，光启动开销就占掉大半时间。
本脚本**只加载一次模型与验证集**，之后每个检查点只做 state_dict 替换，
因此可以在一次运行里扫完所有快照。

## 要回答的问题

上一轮 150 epoch 时 val_loss 在 epoch 85 见顶、之后回升；
本轮 600 epoch 的 val_loss 更早在 **epoch 57** 见顶，最终 val 4.50 而 train 0.54。
但 **val_loss 是匈牙利匹配的混合损失，与检测 AP 不一定同步**。
所以必须按 AP 选权重，而不是按 loss。本脚本就是做这件事。

## 用法

    python sweep_ckpt.py --slices data/slices --ckpt-dir runs/detr_kidney_e600 \
        --classes mito,er --iou 0.5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_dataset import SliceConfig, SliceTargets, find_slices  # noqa: E402
from train_detr_slices import SliceDetrDataset, collate, group_split  # noqa: E402
from eval_detr import (average_precision,  # noqa: E402
                       cxcywh_norm_to_xyxy_px, greedy_match)


class KeyedDataset(torch.utils.data.Dataset):
    """在 (img, labels) 之外多返回 image_id，便于按 crop 汇总指标。"""

    def __init__(self, base):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        img, lb = self.base[i]
        return img, lb, self.base.records[i]["image_id"]


def collate_keyed(batch):
    return ([b[0] for b in batch], [b[1] for b in batch],
            [b[2] for b in batch])


def evaluate(model, dl, proc, dev, n_classes: int, iou_thr: float,
             conf: float, crops: Optional[List[str]] = None
             ) -> Dict[str, Any]:
    """返回整体指标 + **逐 crop 指标**。

    为什么必须逐 crop 报告：验证集只有 3 个 crop，整体 mAP 在不同检查点之间
    波动 0.17–0.23，这个幅度可能纯粹是采样噪声。若某个检查点在 3 个 crop 上
    **一致**更好，才谈得上真的更好；若排名在 crop 之间翻转，就说明差异不可信。
    """
    model.eval()
    per_class_preds: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(n_classes)}
    n_gt = {i: 0 for i in range(n_classes)}
    conf_stats = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(n_classes)}
    # 逐 crop 收集
    crop_preds: Dict[str, Dict[int, List[Tuple[float, int]]]] = {}
    crop_gt: Dict[str, Dict[int, int]] = {}

    with torch.no_grad():
        for images, labels, keys in dl:
            for img, lb, key in zip(images, labels, keys):
                crop = key.split("/")[-2] if "/" in key else key
                crop_preds.setdefault(crop, {i: [] for i in range(n_classes)})
                crop_gt.setdefault(crop, {i: 0 for i in range(n_classes)})
                enc = proc(images=[img], return_tensors="pt")
                px = enc["pixel_values"].to(dev)
                out = model(pixel_values=px)
                w, h = img.size
                logits = out.logits[0].detach().cpu()
                boxes = out.pred_boxes[0].detach().cpu()
                prob = logits.softmax(-1)
                for c in range(n_classes):
                    gt_norm = lb["boxes"][(lb["class_labels"] == c)].cpu()
                    gt_px = cxcywh_norm_to_xyxy_px(gt_norm, w, h)
                    n_gt[c] += int(gt_px.shape[0])
                    crop_gt[crop][c] += int(gt_px.shape[0])
                    scores = prob[:, c]
                    pred_px = cxcywh_norm_to_xyxy_px(boxes, w, h)
                    hit, _ = greedy_match(pred_px, scores, gt_px, iou_thr)
                    for i in torch.argsort(scores, descending=True).tolist():
                        per_class_preds[c].append((float(scores[i]), int(hit[i])))
                        crop_preds[crop][c].append((float(scores[i]), int(hit[i])))
                    sel = scores >= conf
                    if int(sel.sum()):
                        h2, _ = greedy_match(pred_px[sel], scores[sel], gt_px, iou_thr)
                        conf_stats[c]["tp"] += int(h2.sum())
                        conf_stats[c]["fp"] += int((~h2).sum())
                    conf_stats[c]["fn"] += int(max(0, gt_px.shape[0] - int(hit.sum())))

    def _ap(preds: List[Tuple[float, int]], ng: int) -> float:
        if ng <= 0 or not preds:
            return 0.0
        preds = sorted(preds, key=lambda t: -t[0])
        tp_cum = np.cumsum([p[1] for p in preds])
        fp_cum = np.cumsum([1 - p[1] for p in preds])
        rec = tp_cum / float(ng)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        return average_precision(rec, prec)

    out = {"per_class": {}, "n_gt": {str(k): v for k, v in n_gt.items()}}
    aps = []
    for c in range(n_classes):
        ap = _ap(per_class_preds[c], n_gt[c])
        st = conf_stats[c]
        out["per_class"][str(c)] = {
            "AP": round(ap, 4), "n_gt": n_gt[c], **st,
            "P": round(st["tp"] / max(1, st["tp"] + st["fp"]), 4),
            "R": round(st["tp"] / max(1, st["tp"] + st["fn"]), 4)}
        aps.append(ap)
    out["mAP"] = round(float(np.mean(aps)), 4) if aps else 0.0

    # 原始预测（每类：[(score, 是否命中), ...]），仅供离线重算指标用。
    # 调用方在落盘后应把它从 row 里 pop 掉，避免污染汇总 JSON。
    out["_raw_preds"] = {str(c): [[round(s, 6), int(h)]
                                  for s, h in per_class_preds[c]]
                         for c in range(n_classes)}

    # 逐 crop
    out["per_crop"] = {}
    for crop in sorted(crop_preds):
        caps = []
        entry = {}
        for c in range(n_classes):
            ap = _ap(crop_preds[crop][c], crop_gt[crop][c])
            entry[str(c)] = {"AP": round(ap, 4), "n_gt": crop_gt[crop][c]}
            caps.append(ap)
        entry["mAP"] = round(float(np.mean(caps)), 4) if caps else 0.0
        out["per_crop"][crop] = entry
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="多检查点 AP 扫描")
    ap.add_argument("--slices", default="data/slices")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--classes", default="mito,er")
    ap.add_argument("--min-area", type=int, default=32)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--no-split", action="store_true",
                    help="把 --slices 下**全部**切片当作验证集。"
                         "跨器官场景必须用这个：--slices 已经是留出的另一个器官，"
                         "再按 val-frac 切一刀只会用到其中一小部分"
                         "（实测 15 个 liver crop 被切成 3 个，扫描结果不代表全肝）")
    ap.add_argument("--include-crops", default=None,
                    help="只在这些 crop 上评估（逗号分隔）。域适应三臂实验用："
                         "各臂必须用**完全相同**的留出 crop，否则不可比")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-preds", default=None,
                    help="把每个检查点的**原始预测**（每类每图的 (score, 是否命中) ）"
                         "存成 JSON。教训：AP 算法一改（本次就修了一个 np.interp 的"
                         "重复 recall bug），原先只存聚合结果就只能重跑全部推理；"
                         "存下原始预测后可以离线用任何实现重算 AP、换 IoU 阈值、"
                         "换置信度阈值，无需再碰 GPU")
    args = ap.parse_args(argv)

    os.environ.setdefault("HF_HOME", os.path.expanduser("~/dachuang_organelle/models/hf"))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import DetrForObjectDetection, DetrImageProcessor

    classes = tuple(s.strip() for s in args.classes.split(",") if s.strip())
    cfg = SliceConfig(root=args.slices, classes=classes, min_area=args.min_area)
    records = [r for r in find_slices(args.slices, cfg)
               if SliceTargets.build(r, cfg)["num_instances"] > 0]
    if args.include_crops:
        def _norm(s: str) -> str:
            s = s.strip()
            return s if s.startswith("crop") else f"crop{s}"
        inc = {_norm(x) for x in args.include_crops.split(",") if x.strip()}
        before = len(records)
        records = [r for r in records if r["crop"] in inc]
        print(f"按 crop 筛选：{before} -> {len(records)} 个切片"
              f"（{len({r['crop'] for r in records})} 个 crop）")
        if not records:
            print("[错误] 筛选后没有切片", file=sys.stderr)
            return 2
    if args.no_split:
        val_recs = records
        print(f"验证切片 {len(val_recs)} 个（全量，不做内部划分）")
        print(f"  验证 crop 数 {len({r['crop'] for r in val_recs})}")
    else:
        _, val_recs = group_split(records, args.val_frac, args.seed)
        print(f"验证切片 {len(val_recs)} 个")

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", num_labels=len(classes),
        ignore_mismatched_sizes=True,
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    proc = DetrImageProcessor(size={"shortest_edge": args.size,
                                    "longest_edge": int(args.size * 1.5)})
    ds = SliceDetrDataset(val_recs, cfg, augment=False)
    dl = DataLoader(KeyedDataset(ds), batch_size=4, shuffle=False,
                    num_workers=2, collate_fn=collate_keyed)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt")))
    # last.pt 与 best.pt 放最后，避免重复评估同一权重时混淆
    order = {"best.pt": 0, "last.pt": 1}
    ckpts.sort(key=lambda p: (order.get(os.path.basename(p), 2),
                              os.path.basename(p)))
    print(f"找到 {len(ckpts)} 个检查点，IoU={args.iou} conf={args.conf}")

    rows: List[Dict[str, Any]] = []
    for p in ckpts:
        name = os.path.basename(p)
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(ck["model"], strict=False)
            if missing:
                print(f"  [警告] {name} 缺 {len(missing)} 个键")
        except Exception as exc:  # noqa: BLE001
            print(f"  [跳过] {name}: {type(exc).__name__}: {exc}")
            continue
        res = evaluate(model, dl, proc, dev, len(classes), args.iou, args.conf)
        epoch = int(ck.get("epoch", -1)) + 1
        raw = res.pop("_raw_preds", None)
        row = {"ckpt": name, "epoch": epoch,
               "val_loss_at_save": (round(float(ck["best_val"]), 4)
                                    if ck.get("best_val") is not None else None),
               **res}
        rows.append(row)
        # 落盘原始预测，便于离线重算指标（见 --dump-preds 的说明）
        if args.dump_preds and raw is not None:
            allp = os.path.join(args.dump_preds,
                                f"{os.path.splitext(name)[0]}_iou{args.iou}.json")
            os.makedirs(args.dump_preds, exist_ok=True)
            with open(allp, "w", encoding="utf-8") as fh:
                json.dump({"ckpt": name, "epoch": epoch, "iou": args.iou,
                           "conf": args.conf, "classes": list(classes),
                           "per_class_preds": raw,
                           "n_gt": res.get("n_gt", {})},
                          fh, ensure_ascii=False)
        per = " ".join(f"{classes[int(k)]} AP={v['AP']:.4f} P={v['P']:.2f} R={v['R']:.2f}"
                       for k, v in res["per_class"].items())
        print(f"  {name:<16} epoch {epoch:>4}  mAP={res['mAP']:.4f}  {per}", flush=True)

    print()
    print("=" * 96)
    print(f"{'ckpt':<16} {'epoch':>6} {'mAP':>8} "
          + " ".join(f"{c+' AP':>10} {c+' P':>7} {c+' R':>7}" for c in classes))
    print("-" * 96)
    for r in rows:
        line = f"{r['ckpt']:<16} {r['epoch']:>6} {r['mAP']:>8.4f} "
        for c in range(len(classes)):
            v = r["per_class"].get(str(c), {})
            line += (f"{v.get('AP', 0):>10.4f} {v.get('P', 0):>7.3f} "
                     f"{v.get('R', 0):>7.3f} ")
        print(line)

    if rows:
        best = max(rows, key=lambda r: r["mAP"])
        print()
        print(f"按 mAP 最优：{best['ckpt']}（epoch {best['epoch']}, "
              f"mAP={best['mAP']:.4f}）")

        # 逐 crop 拆解：只有 3 个验证 crop，整体 mAP 的差异可能只是噪声。
        # 若排名在 crop 之间翻转，就不能宣称某个检查点真的更好。
        print()
        print("=" * 96)
        print("逐 crop mAP（判断排名是否稳定）")
        print("-" * 96)
        crops = sorted(best.get("per_crop", {}))
        print(f"{'ckpt':<16} {'epoch':>6} {'整体':>8} "
              + " ".join(f"{c[:12]:>13}" for c in crops))
        for r in rows:
            line = f"{r['ckpt']:<16} {r['epoch']:>6} {r['mAP']:>8.4f} "
            for c in crops:
                pc = (r.get("per_crop") or {}).get(c, {})
                line += f"{pc.get('mAP', float('nan')):>13.4f}"
            print(line)

        # 排名一致性：每个 crop 上 mAP 最高的检查点是否相同
        winners = {}
        for c in crops:
            vals = [((r.get("per_crop") or {}).get(c, {}).get("mAP", -1), r["ckpt"])
                    for r in rows]
            winners[c] = max(vals)[1]
        print()
        print("各 crop 上 mAP 最高的检查点：")
        for c, w in winners.items():
            print(f"  {c:<16} -> {w}")
        uniq = set(winners.values())
        if len(uniq) == 1:
            print(f"  => {list(uniq)[0]} 在全部 {len(crops)} 个 crop 上一致最优，排名可信")
        else:
            print(f"  => 排名在 crop 之间不一致（{len(uniq)} 个不同赢家）"
                  f"，整体 mAP 的差异不能当作真实提升")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"iou": args.iou, "conf": args.conf, "rows": rows}, fh,
                      ensure_ascii=False, indent=1)
        print(f"已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
