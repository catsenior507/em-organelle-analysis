#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py —— DETR 区域检测 + 属性回归的训练/评估入口

替代 ``code/detr_region_feature.py``。与原实现的关键差别
------------------------------------------------------
=========================  ==================================  ============================
方面                       原实现                               本实现
=========================  ==================================  ============================
数据来源                    ``change_feature_json``（像素被压成    ``annotations_v2``（无损轮廓 +
                           1 个点、JSON 来自截断产物）⛔         canonical 特征）
预处理                      ``transform`` + ``processor`` 双重    只走 ``processor``（唯一入口）
                           归一化 ⛔
targets 传递                ``collate`` 返回 list，损失按 dict    ``collate`` 返回 list，损失按
                           索引 → 第一次前向必崩 ⛔              list 语义
匹配                        ``topk`` IoU 贪心（非置换）⛔         匈牙利 ``linear_sum_assignment``
损失                        CE + MSE(框) + 0.1·MSE(特征) ⛔      加权 CE + L1 + GIoU + 属性 MSE
学习率                      backbone 与 head 同一个 lr ⛔          分组 lr（1e-5 / 1e-4）+ 阶梯衰减
梯度裁剪                    无 ⛔                                 ``clip_grad_norm_`` 0.1
随机性                      无种子 ⛔                             ``seed_everything``
验证集                      从增强副本里随机抽样（泄漏）⛔          按图分组的 held-out
评估                        只报未归一化的 MSE ⛔                  mAP@50/75 + 每属性 MAE/R²
                                                                 + "预测均值框"基线
=========================  ==================================  ============================

用法::

    # 先缓存权重（需要联网，或在有网机器上下载后拷贝）
    huggingface-cli download facebook/detr-resnet-50

    # 训练
    python tools/detr/train.py --epochs 200 --batch-size 4

    # 只评估
    python tools/detr/train.py --eval-only --checkpoint runs/detr/best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # 让 `import detr.*` 生效
sys.path.insert(0, _HERE)                            # 也允许直接跑本文件

from detr.dataset import (  # noqa: E402
    DetrDataConfig,
    RegionDetrDataset,
    build_manifest,
    collate_fn,
    compute_attribute_stats,
    group_split,
)
from detr.losses import LossConfig, SetCriterion, box_cxcywh_to_xyxy, simple_ap  # noqa: E402
from detr.model import DetrModelConfig, DetrRegionModel  # noqa: E402

logger = logging.getLogger("detr.train")


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """固定全部随机源。

    原实现没有任何种子，同一份代码两次运行结果不同，日志无法复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        logger.warning("请求 cuda 但不可用，回退到 cpu")
        return torch.device("cpu")
    return dev


def build_argparser() -> argparse.ArgumentParser:
    root = os.path.abspath(os.path.join(_HERE, "..", ".."))
    p = argparse.ArgumentParser(description="DETR 区域检测 + 属性回归")
    p.add_argument("--annotations", default=os.path.join(root, "data", "annotations_v2"))
    p.add_argument("--images", default=os.path.join(root, "data", "原图片"))
    p.add_argument("--out", default=os.path.join(root, "runs", "detr"))
    p.add_argument("--pretrained", default="facebook/detr-resnet-50",
                   help="HuggingFace 权重标识；离线环境请先下载缓存")
    p.add_argument("--num-queries", type=int, default=15)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-drop", type=int, default=150,
                   help="第几个 epoch 把学习率除以 10（原文 lr_drop=150）")
    p.add_argument("--grad-clip", type=float, default=0.1)
    p.add_argument("--short-side", type=int, default=800,
                   help="DETR 处理器把短边缩放到该尺寸；显存吃紧时降到 480/600")
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--smoke-test", action="store_true",
                   help="只用几张图跑 2 个 epoch，验证管线连通性")
    p.add_argument("--log-level", default="INFO")
    return p


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: DetrRegionModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: Optional[SetCriterion] = None,
    score_threshold: float = 0.5,
    attr_names: Optional[List[str]] = None,
    attr_mean: Optional[np.ndarray] = None,
    attr_std: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """检测指标（mAP@50/75）+ 属性回归指标（MAE/R²）。

    原实现只报了一个未归一化的 MSE，既没有基线也没有分特征误差，
    读者无法判断模型是否真的学到了东西。

    属性对齐说明
    ------------
    ``pred_attributes`` 的形状是 ``[B, Q, A]``（Q 个查询），而目标属性是
    ``[B, N_i, A]``（每张图的目标数不同）。两者**不能直接拼接**，必须先用
    匈牙利匹配建立查询↔目标的对应关系。这里复用训练时的 ``criterion``
    做匹配，避免"按 top-k 置信度近似对齐"这种会引入偏差的偷懒做法。
    """
    model.eval()
    aps_50, aps_75 = [], []
    attr_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        outputs = model(pixel_values)
        probs = outputs["logits"].softmax(-1)[..., : model.cfg.num_classes]   # [B,Q,C]
        scores, _ = probs.max(-1)
        boxes_xyxy = box_cxcywh_to_xyxy(outputs["pred_boxes"])                # 归一化

        for b, tgt in enumerate(batch["targets"]):
            orig_h, orig_w = tgt["orig_size"].tolist()
            scale = torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=torch.float32)
            tgt_boxes = box_cxcywh_to_xyxy(tgt["boxes"]).cpu() * scale
            pred_boxes_px = boxes_xyxy[b].cpu() * scale
            pred_scores = scores[b].cpu()

            ap50 = simple_ap(pred_boxes_px, pred_scores, tgt_boxes, 0.50)
            if ap50 is not None:
                aps_50.append(ap50)
                ap75 = simple_ap(pred_boxes_px, pred_scores, tgt_boxes, 0.75)
                aps_75.append(ap75 if ap75 is not None else 0.0)

        # ---- 属性：用匈牙利匹配对齐后再收集 ----
        if attr_names and "pred_attributes" in outputs and criterion is not None:
            targets_dev = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                for t in batch["targets"]
            ]
            matches = criterion._match(outputs, targets_dev)      # noqa: SLF001
            for b, (src, dst) in enumerate(matches):
                if src.numel() == 0 or "attributes" not in batch["targets"][b]:
                    continue
                pred = outputs["pred_attributes"][b][src].cpu().numpy()
                true = batch["targets"][b]["attributes"][dst].cpu().numpy()
                attr_pairs.append((pred, true))

    metrics: Dict[str, Any] = {}
    if aps_50:
        metrics["mAP@50"] = float(np.mean(aps_50))
        metrics["mAP@75"] = float(np.mean(aps_75))
        metrics["num_val_images"] = len(aps_50)

    if attr_names and attr_pairs:
        P = np.concatenate([p for p, _ in attr_pairs], axis=0)
        T = np.concatenate([t for _, t in attr_pairs], axis=0)
        # 反标准化到物理尺度，指标才有意义（原实现输出的是 z-score 却当成坐标）
        if attr_mean is not None and attr_std is not None:
            P = P * attr_std + attr_mean
            T = T * attr_std + attr_mean

        per_attr_mae, per_attr_r2, per_attr_baseline = {}, {}, {}
        for a, name in enumerate(attr_names):
            err = np.abs(P[:, a] - T[:, a])
            per_attr_mae[name] = float(err.mean())
            var = float(np.var(T[:, a]))
            if var > 1e-12:
                per_attr_r2[name] = float(1.0 - np.mean((P[:, a] - T[:, a]) ** 2) / var)
            else:
                per_attr_r2[name] = None
            # "预测训练集均值"的基线：MAE 若不明显优于它，说明模型没学到东西
            per_attr_baseline[name] = float(np.abs(T[:, a] - T[:, a].mean()).mean())

        metrics["attr_n_samples"] = int(P.shape[0])
        metrics["attr_mae"] = per_attr_mae
        metrics["attr_r2"] = per_attr_r2
        metrics["attr_mae_baseline_mean_predictor"] = per_attr_baseline
        metrics["attr_mae_macro"] = float(np.mean(list(per_attr_mae.values())))
        metrics["attr_mae_baseline_macro"] = float(np.mean(list(per_attr_baseline.values())))
        r2s = [v for v in per_attr_r2.values() if v is not None]
        if r2s:
            metrics["attr_r2_macro"] = float(np.mean(r2s))
        # 显式给出结论性判断，避免"数字摆在那里但没人注意"
        metrics["attr_beats_baseline"] = bool(
            metrics["attr_mae_macro"] < metrics["attr_mae_baseline_macro"] * 0.95
        )
    return metrics


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    seed_everything(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = pick_device(args.device)
    logger.info("device=%s", device)
    if device.type == "cpu":
        logger.warning(
            "当前在 CPU 上运行。DETR-ResNet50 在 CPU 上训练 89 张图需要数小时以上，"
            "仅建议用于 --smoke-test 验证管线连通性。"
        )

    data_cfg = DetrDataConfig(
        annotation_dir=args.annotations,
        image_dir=args.images,
        short_side=args.short_side,
    )
    records = build_manifest(data_cfg)
    if not records:
        logger.error("没有可用图像；请先运行 tools/morpho/extract_regions.py")
        return 2
    if args.smoke_test:
        records = records[:4]
        args.epochs = min(args.epochs, 2)
        logger.info("smoke test: 只用 %d 张图，%d 个 epoch", len(records), args.epochs)

    attr_mean, attr_std = compute_attribute_stats(records)
    logger.info("属性均值 %s", np.round(attr_mean, 3).tolist())
    logger.info("属性标准差 %s", np.round(attr_std, 3).tolist())

    train_idx, val_idx = group_split(records, args.val_fraction, args.seed)
    logger.info("按图分组划分：train=%d 图, val=%d 图（无同图增强副本跨集）",
                len(train_idx), len(val_idx))

    from transformers import DetrImageProcessor
    try:
        processor = DetrImageProcessor.from_pretrained(args.pretrained)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "无法加载处理器（%s）。这是**离线环境最常见的问题**：\n"
            "  huggingface-cli download %s\n"
            "或在有网机器上缓存 ~/.cache/huggingface 后拷贝过来。\n"
            "原始错误：%s", args.pretrained, args.pretrained, exc,
        )
        return 3

    train_ds = RegionDetrDataset(
        [records[i] for i in train_idx], processor, data_cfg, attr_mean, attr_std,
        image_ids=train_idx,
    )
    val_ds = RegionDetrDataset(
        [records[i] for i in val_idx], processor, data_cfg, attr_mean, attr_std,
        image_ids=val_idx,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    model_cfg = DetrModelConfig(
        num_queries=args.num_queries,
        num_classes=1,
        num_attributes=len(data_cfg.attribute_names),
        pretrained=args.pretrained,
    )
    try:
        model = DetrRegionModel(model_cfg).to(device)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "构建模型失败：%s\n"
            "最常见原因：本机没有缓存 %s 权重且无法访问 huggingface.co。\n"
            "    huggingface-cli download %s", exc, args.pretrained, args.pretrained,
        )
        return 3

    criterion = SetCriterion(
        num_classes=model_cfg.num_classes,
        num_attributes=model_cfg.num_attributes,
        cfg=LossConfig(),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.lr, args.lr_backbone, args.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)

    start_epoch = 0
    best_map = -1.0
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_map = ckpt.get("best_map", -1.0)
        logger.info("已从 %s 恢复（epoch %d）", args.checkpoint, start_epoch)

    history: List[Dict[str, Any]] = []
    if not args.eval_only:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            t0 = time.time()
            running: Dict[str, float] = {}
            n_batches = 0

            for step, batch in enumerate(train_loader):
                pixel_values = batch["pixel_values"].to(device)
                targets = [
                    {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                    for t in batch["targets"]
                ]

                outputs = model(pixel_values)
                out = criterion(outputs, targets)
                loss = out["loss"]

                if not torch.isfinite(loss):
                    logger.error("epoch %d step %d 出现非有限损失 %s，跳过该 batch",
                                 epoch, step, loss.item())
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                for k, v in out["losses"].items():
                    running[k] = running.get(k, 0.0) + float(v)
                n_batches += 1

            scheduler.step()
            train_metrics = {k: v / max(n_batches, 1) for k, v in running.items()}
            val_metrics = evaluate(
                model, val_loader, device, criterion=criterion,
                attr_names=list(data_cfg.attribute_names),
                attr_mean=attr_mean, attr_std=attr_std,
            )
            record = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": round(time.time() - t0, 1),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(record)
            logger.info(
                "epoch %d/%d  %.1fs  train_loss=%.4f  val_mAP@50=%s  val_attr_MAE=%s",
                epoch + 1, args.epochs, record["seconds"],
                train_metrics.get("loss_total", float("nan")),
                round(val_metrics.get("mAP@50", float("nan")), 4),
                round(val_metrics.get("attr_mae_macro", float("nan")), 4),
            )

            with open(os.path.join(args.out, "history.json"), "w", encoding="utf-8") as fh:
                json.dump(history, fh, ensure_ascii=False, indent=1)

            cur_map = val_metrics.get("mAP@50", -1.0)
            if cur_map > best_map:
                best_map = cur_map
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_map": best_map,
                    "model_cfg": vars(model_cfg),
                    "data_cfg": {
                        "attribute_names": list(data_cfg.attribute_names),
                        "attribute_mean": attr_mean.tolist(),
                        "attribute_std": attr_std.tolist(),
                        "short_side": data_cfg.short_side,
                    },
                }, os.path.join(args.out, "best.pth"))
                logger.info("  保存最佳模型（mAP@50=%.4f）", best_map)
    else:
        val_metrics = evaluate(
            model, val_loader, device, criterion=criterion,
            attr_names=list(data_cfg.attribute_names),
            attr_mean=attr_mean, attr_std=attr_std,
        )
        logger.info("评估结果：%s", json.dumps(val_metrics, ensure_ascii=False, indent=1))
        with open(os.path.join(args.out, "eval.json"), "w", encoding="utf-8") as fh:
            json.dump(val_metrics, fh, ensure_ascii=False, indent=1)

    # 落盘配置，便于复现
    with open(os.path.join(args.out, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "args": vars(args),
            "model_cfg": vars(model_cfg),
            "attribute_names": list(data_cfg.attribute_names),
            "attribute_mean": attr_mean.tolist(),
            "attribute_std": attr_std.tolist(),
            "train_images": [records[i].name for i in train_idx],
            "val_images": [records[i].name for i in val_idx],
        }, fh, ensure_ascii=False, indent=1)
    logger.info("完成。输出目录：%s", os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
