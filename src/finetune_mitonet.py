#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_mitonet.py —— 在自有数据上微调 MitoNet

## 为什么微调而不是从零训练

MitoNet 零样本在全图评估下 Dice 0.7518（同 18 张留出图），
而我们从零训练的 U-Net 是 0.7839——只差 4%。
MitoNet 的编码器在约 150 万张细胞 EM 图上有自监督预训练，
这个起点比随机初始化好得多，微调应当能用更少的数据/时间超过从零训练。

## 配置怎么来的

`empanada_napari.finetune.main(config)` 需要一份**完整**配置；
插件层的 `_finetune.py` 只负责把用户输入补全。
这里直接从已安装的 `empanada_napari/configs/MitoNet_v1.yaml` 读出
`MODEL` 与 `FINETUNE` 两段（保证与官方模型完全一致），再补上 `TRAIN`/`EVAL`。

## 判据（跑之前定好）

对照三个已有结果：

    MitoNet 零样本   0.7518
    自有 U-Net 从零  0.7839
    CellMap 零样本   0.1516

* 微调后 > 0.7839  -> 微调优于从零训练，应作为默认方案
* 微调后约 0.75    -> 微调没带来提升，需检查学习率/层选择
* 微调后 < 0.70    -> 灾难性遗忘，必须减小 max_lr 或只训解码器
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import yaml


def load_base_config() -> dict:
    """从已安装的 empanada_napari 里读官方 MitoNet 配置。"""
    import empanada_napari
    cfg_dir = os.path.join(os.path.dirname(empanada_napari.__file__), "configs")
    path = os.path.join(cfg_dir, "MitoNet_v1.yaml")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_config(args) -> dict:
    base = load_base_config()
    cfg = {
        "model_name": "mitonet_ourdata",
        "MODEL": {k: v for k, v in base.items() if k != "FINETUNE"},
        "FINETUNE": base["FINETUNE"],
        "TRAIN": {
            "train_dir": args.train_dir,
            "additional_train_dirs": None,
            "model_dir": args.model_dir,
            "save_freq": args.save_every,
            "finetune_layer": args.finetune_layer,
            "lr_schedule": "OneCycleLR",
            "schedule_params": {
                "max_lr": args.max_lr,
                "epochs": args.epochs,
                "steps_per_epoch": -1,
                "pct_start": 0.3,
            },
            "amp": True,
            "optimizer": "AdamW",
            "optimizer_params": {"weight_decay": 0.1},
            "print_freq": 50,
            "metrics": [
                {"metric": "IoU", "name": "semantic_iou", "labels": None,
                 "output_key": "sem_logits", "target_key": "sem"},
            ],
            "batch_size": args.batch,
            "workers": args.workers,
            "epochs": args.epochs,
            "augmentations": [
                {"aug": "RandomScale", "scale_limit": [-0.5, 0.5]},
                {"aug": "PadIfNeeded", "min_height": args.patch,
                 "min_width": args.patch, "border_mode": 0},
                {"aug": "RandomCrop", "height": args.patch, "width": args.patch},
                {"aug": "Rotate", "limit": 180, "border_mode": 0},
                {"aug": "RandomBrightnessContrast", "brightness_limit": 0.2,
                 "contrast_limit": 0.2},
                {"aug": "HorizontalFlip"},
                {"aug": "VerticalFlip"},
            ],
        },
        "EVAL": {
            "eval_dir": args.val_dir,
            "epochs_per_eval": args.eval_every,
            "metrics": [
                {"metric": "IoU", "name": "semantic_iou", "labels": None,
                 "output_key": "sem_logits", "target_key": "sem"},
            ],
        },
    }

    # 补全 metric 的 labels（插件层负责这件事，这里必须自己做）
    labels = cfg["MODEL"].get("labels", [1])
    thing_list = cfg["MODEL"].get("thing_list", [1])
    for section in ("TRAIN", "EVAL"):
        for m in cfg[section]["metrics"]:
            m["labels"] = labels if m["metric"] in ("IoU", "PQ") else thing_list

    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="data/empanada_ft/train")
    ap.add_argument("--val-dir", default="data/empanada_ft/val")
    ap.add_argument("--model-dir", default="runs/empanada_ft")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-lr", type=float, default=1e-4,
                    help="微调学习率。太大会灾难性遗忘；"
                         "官方训练用 3e-3，微调应小一到两个数量级")
    ap.add_argument("--finetune-layer", default="all",
                    choices=["all", "none", "stage1", "stage2", "stage3", "stage4"],
                    help="解冻到哪一层。all 解冻整个编码器；stage4 只调解码器加最后一层")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--dump-config", default=None)
    args = ap.parse_args(argv)

    cfg = build_config(args)

    # 先建目录再写配置。早期版本把 dump-config 放在 makedirs 之前，
    # 直接因目标目录不存在而崩，白白浪费一轮启动。
    os.makedirs(args.model_dir, exist_ok=True)
    if args.dump_config:
        d = os.path.dirname(os.path.abspath(args.dump_config))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.dump_config, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
        print(f"配置已写出 {args.dump_config}")

    print("=" * 70)
    print("微调 MitoNet")
    print("=" * 70)
    print(f"  训练目录 {args.train_dir}")
    print(f"  验证目录 {args.val_dir}")
    print(f"  输出     {args.model_dir}")
    print(f"  epochs {args.epochs}  batch {args.batch}  patch {args.patch}")
    print(f"  max_lr {args.max_lr}  解冻 {args.finetune_layer}")
    print(f"  权重     {cfg['MODEL'].get('model')}")
    print()

    try:
        from empanada_napari import finetune
        rc = finetune.main(cfg)
        print(f"\nfinetune.main 返回 {rc}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\n[失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
