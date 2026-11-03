#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
morpho.detr_dataset —— 面向 DETR 的数据集（替代 code/detr_region_feature.py 的数据部分）

修复的原实现缺陷
----------------
============================  ==========================================================
原实现                         本实现
============================  ==========================================================
``transform`` 与 ``processor``  **只调用 processor 一次**（原实现先 Resize+Normalize，
双重归一化图像 ⛔               再用 processor 归一化一次，输入分布完全错误）
``targets`` 用 dict 逐样本索引  ``collate_fn`` 用 padding + mask，损失端按样本遍历
（collate 返回 list → 必崩）⛔
``image_id = idx``（Dataset 训练  真实 ``image_id``，且**按图分组**划分 train/val
/验证随机划分 → 同一张图的增强
副本同时进两个集合，严重泄漏）⛔
``region_pixels`` 截断到 1000 点  读 canonical 轮廓多边形（无损形状）
⛔
特征口径与 MATLAB 不一致 ⛔     直接读 canonical 特征，两边同一个定义
============================  ==========================================================

数据来源：``tools/morpho/extract_regions.py`` 产出的 ``data/annotations_v2/``。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

#: DETR 官方只有一个前景类；类别 0 保留给 "no object"（背景），由 CrossEntropy 隐式处理。
NUM_CLASSES = 1


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

@dataclass
class DetrDataConfig:
    annotation_dir: str = "data/annotations_v2"
    image_dir: str = "data/原图片"
    image_exts: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg")
    #: DETR 处理器会把短边缩放到这个尺寸；800 是官方默认，显存紧张时降到 480/600。
    short_side: int = 800
    max_side: int = 1333
    #: 只使用被标注过的区域作为目标（未标注的区域不参与检测训练）
    require_labeled: bool = True
    #: 附加到每个目标的回归属性（canonical 特征名，见 features.COMPACT_FEATURE_NAMES）
    attribute_names: Tuple[str, ...] = (
        "major_axis", "minor_axis", "solidity",
        "eccentricity", "extent", "gray_mean",
    )
    #: 属性的物理尺度（用于损失内标准化；None = 在该属性上按数据集统计量自动定标）
    attribute_scales: Optional[Dict[str, float]] = None


# --------------------------------------------------------------------------
# 清单
# --------------------------------------------------------------------------

@dataclass
class ImageRecord:
    name: str
    path: str
    boxes: np.ndarray            # [N, 4]，归一化 cxcywh
    boxes_xyxy_px: np.ndarray    # [N, 4]，1-based (x1,y1,x2,y2)，像素
    labels: np.ndarray           # [N]，全 0（唯一前景类）
    attributes: np.ndarray       # [N, A]，原始属性值
    num_regions: int


def _to_cxcywh_norm(box_xyxy: Sequence[float], w: int, h: int) -> List[float]:
    """``(x1,y1,x2,y2)`` 像素（1-based）→ 归一化 ``(cx, cy, bw, bh)``。

    DETR 的 ``no_objects`` 用 cxcywh，与本函数一致。
    """
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = max(x2 - x1, 1.0) / w
    bh = max(y2 - y1, 1.0) / h
    return [cx, cy, bw, bh]


def build_manifest(cfg: DetrDataConfig, verbose: bool = True) -> List[ImageRecord]:
    """读取 ``annotations_v2`` 并构造图像级清单。

    只有**存在有效区域**的图才会进入清单——原实现会把没有标注的增强图也塞进
    DataLoader，产生空 target 样本（约占 6%），白白拉低有效 batch。
    """
    ds_path = os.path.join(cfg.annotation_dir, "dataset.json")
    if not os.path.exists(ds_path):
        raise FileNotFoundError(
            f"未找到 {ds_path}；请先运行 tools/morpho/extract_regions.py"
        )
    with open(ds_path, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)

    records: List[ImageRecord] = []
    skipped_no_image = 0
    skipped_no_region = 0

    for entry in dataset.get("images", []):
        name = entry["image"]
        img_path = None
        for ext in cfg.image_exts:
            cand = os.path.join(cfg.image_dir, name + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            skipped_no_image += 1
            continue

        region_path = os.path.join(cfg.annotation_dir, entry["file"])
        with open(region_path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)

        height, width = rec["image_shape"][0], rec["image_shape"][1]
        boxes, attrs = [], []
        for r in rec["regions"]:
            if cfg.require_labeled and not r.get("is_labeled", False):
                continue
            boxes.append(_to_cxcywh_norm(r["bbox_xyxy"] if "bbox_xyxy" in r else
                                         (r["min_x"], r["min_y"], r["max_x"], r["max_y"]),
                                         width, height))
            attrs.append([float(r.get(a, 0.0)) for a in cfg.attribute_names])

        if not boxes:
            skipped_no_region += 1
            continue

        boxes_arr = np.asarray(boxes, dtype=np.float32)
        attrs_arr = np.asarray(attrs, dtype=np.float32)
        xyxy = np.asarray([
            [b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2]
            for b in boxes_arr
        ], dtype=np.float32) * np.asarray([width, height, width, height], dtype=np.float32)

        records.append(ImageRecord(
            name=name,
            path=img_path,
            boxes=boxes_arr,
            boxes_xyxy_px=xyxy,
            labels=np.zeros(len(boxes), dtype=np.int64),
            attributes=attrs_arr,
            num_regions=rec["num_regions"],
        ))

    if verbose:
        print(f"[manifest] 可用图像 {len(records)} 张；"
              f"缺原图跳过 {skipped_no_image}；无有效区域跳过 {skipped_no_region}")
        if records:
            counts = [len(r.boxes) for r in records]
            print(f"[manifest] 每图目标数 min={min(counts)} "
                  f"median={int(np.median(counts))} max={max(counts)}")
    return records


def compute_attribute_stats(
    records: Sequence[ImageRecord],
) -> Tuple[np.ndarray, np.ndarray]:
    """在整个数据集上统计属性均值/标准差。

    与原实现的关键差别：原 ``neural_network_training.m`` 确实做了标准化，但
    **只保存网络、不保存 mean/std**，导致写出的"预测值"是 z-score 而非物理量
    （``features_pre/11_0002_predict.json`` 里出现 ``roughness: -0.476``）。
    这里把 mean/std 作为数据集的一部分一起落盘。
    """
    all_attrs = np.concatenate([r.attributes for r in records], axis=0)
    mean = all_attrs.mean(axis=0)
    std = all_attrs.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class RegionDetrDataset(Dataset):
    """返回 ``(pixel_values, target, meta)``。

    ``pixel_values`` 由 ``DetrImageProcessor`` 产出——**处理器是唯一的预处理入口**，
    这里刻意不再叠加任何 ``torchvision.transforms``。
    """

    def __init__(
        self,
        records: Sequence[ImageRecord],
        processor: Any,
        cfg: DetrDataConfig,
        attr_mean: Optional[np.ndarray] = None,
        attr_std: Optional[np.ndarray] = None,
        image_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.records = list(records)
        self.processor = processor
        self.cfg = cfg
        self.attr_mean = attr_mean
        self.attr_std = attr_std
        # image_ids 必须显式传入并全局唯一；原实现用 DataLoader 内的局部 idx，
        # 一旦做 train/val 划分，同一个 image_id 会指代两张不同的图。
        self.image_ids = (
            list(image_ids) if image_ids is not None
            else list(range(len(self.records)))
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        from PIL import Image

        rec = self.records[idx]
        image = Image.open(rec.path).convert("RGB")

        encoding = self.processor(
            images=image,
            return_tensors="pt",
            size={"shortest_edge": self.cfg.short_side, "longest_edge": self.cfg.max_side},
        )
        pixel_values = encoding["pixel_values"][0]

        attrs = rec.attributes
        if self.attr_mean is not None and self.attr_std is not None:
            attrs = (attrs - self.attr_mean) / self.attr_std

        target = {
            "boxes": torch.as_tensor(rec.boxes, dtype=torch.float32),
            "labels": torch.as_tensor(rec.labels, dtype=torch.long),
            "attributes": torch.as_tensor(attrs, dtype=torch.float32),
            "image_id": torch.tensor([self.image_ids[idx]], dtype=torch.long),
            "orig_size": torch.tensor(list(image.size[::-1]), dtype=torch.long),  # (h, w)
            "size": torch.tensor(list(pixel_values.shape[-2:]), dtype=torch.long),
        }
        return {
            "pixel_values": pixel_values,
            "target": target,
            "meta": {"image_id": self.image_ids[idx], "name": rec.name},
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把 batch 组织成 **list of targets**，而不是试图 stack 成 dict。

    原实现的 ``collate_fn`` 返回 ``{'targets': [t0, t1, ...]}``（list），
    而 ``compute_loss`` 里写的是 ``targets['boxes'][i]``（dict 语义）——
    两者不匹配，第一次前向就 ``TypeError``。本实现统一为 list 语义。
    """
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "targets": [b["target"] for b in batch],
        "metas": [b["meta"] for b in batch],
    }


# --------------------------------------------------------------------------
# 划分：按图分组
# --------------------------------------------------------------------------

def group_split(
    records: Sequence[ImageRecord],
    val_fraction: float = 0.25,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """按**图像**划分 train/val，而不是按样本行。

    原实现（``detr_region_feature.py`` 用 ``random_split``，
    ``representation_learning.m:1031`` 用 ``randperm`` 划分行）会把同一张原图
    （以及它的旋转/翻转/裁剪副本）同时放进两边。本项目的增强副本与源图
    像素级重叠，随机划分等于把测试集泄漏进训练集。
    """
    n = len(records)
    if n == 0:
        return [], []
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    n_val = min(n_val, max(1, n - 1))
    val_idx = sorted(order[:n_val].tolist())
    train_idx = sorted(order[n_val:].tolist())
    return train_idx, val_idx
