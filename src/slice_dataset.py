#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slice_dataset.py —— 把 CellMap 导出的 2D 切片变成 DETR 的多类别训练数据

## 为什么需要新的 Dataset

现有的 `detr/dataset.py` 读的是本项目自己的 `annotations_v2`（每图一堆 JSON 区域）。
而公开数据（CellMap）经 `zarr_to_2d.py` 导出后是**图像 + 每类掩码**：

    <root>/<dataset>/crop<ID>/z<NNNN>/
        em.png        灰度电镜图
        mito.png      线粒体掩码（像素值 = 类别标记，0 = 背景）
        er.png        内质网掩码
        ...

要喂给 DETR，需要从**掩码的连通分量**反推实例框与类别标签。
本模块就做这件事，并且刻意保持与 `losses.py` 的多类别损失接口一致
（分类头输出 `num_classes + 1`，最后一位是 no-object）。

## 关键设计点

1. **类别由文件名决定，不是靠猜测**。`--classes mito,er` 给出类别表；
   每个标签 PNG 提供该类的实例。
2. **框来自连通分量**，不是来自掩码的整幅包围盒——否则同一类的多个细胞器
   会被合成一个大框，检测任务就废了。
3. **丢弃过小的分量**（`--min-area`）。标注边缘常有 1–2 像素的碎片，
   它们会变成垃圾真值。
4. **保留 `image_id` 与来源**，便于按图分组划分（`group_split` 已有）。

## 用法

    python -c "from tools.detr.slice_dataset import *; ..."

作为库使用；命令行入口仅用于自检：

    python tools/detr/slice_dataset.py --selftest
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover
    ndi = None


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

@dataclass
class SliceConfig:
    """切片数据集的配置。"""
    root: str = ""
    #: 类别名，顺序即类别 id（0..N-1）。背景类是 num_classes（由损失处理）。
    classes: Tuple[str, ...] = ("mito", "er")
    image_name: str = "em.png"
    #: 小于该像素数的连通分量被丢弃（标注碎片）
    min_area: int = 64
    #: 只保留面积占比不超过整图的实例（排除"把整幅图都标成一类"的情形）
    max_area_frac: float = 0.95
    #: 是否把标签值 >0 的所有像素都算该类（否则要求像素值 == 类别索引+1）
    any_nonzero_is_class: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "classes": list(self.classes),
            "image_name": self.image_name,
            "min_area": self.min_area,
            "max_area_frac": self.max_area_frac,
            "any_nonzero_is_class": self.any_nonzero_is_class,
            "note": (
                "类别 id 由 classes 的顺序决定；num_classes = len(classes)，"
                "分类头输出 len(classes)+1（最后一位 no-object）。"
            ),
        }


# --------------------------------------------------------------------------
# 发现切片
# --------------------------------------------------------------------------

def find_slices(root: str, cfg: SliceConfig) -> List[Dict[str, Any]]:
    """扫描 zarr_to_2d.py 的输出，返回可用的切片记录。"""
    out: List[Dict[str, Any]] = []
    pattern = os.path.join(root, "*", "*", "z*")
    for zdir in sorted(glob.glob(pattern)):
        if not os.path.isdir(zdir):
            continue
        zname = os.path.basename(zdir)
        if not (zname.startswith("z") and zname[1:].isdigit()):
            continue
        img = os.path.join(zdir, cfg.image_name)
        if not os.path.exists(img):
            continue
        masks = {}
        for c in cfg.classes:
            p = os.path.join(zdir, f"{c}.png")
            if os.path.exists(p):
                masks[c] = p
        if not masks:
            continue
        parts = zdir.replace("\\", "/").split("/")
        out.append({
            "dir": zdir,
            "dataset": parts[-3],
            "crop": parts[-2],
            "z": int(zname[1:]),
            "image": img,
            "masks": masks,
            # image_id 必须全局唯一且稳定：用 dataset/crop/z 拼
            "image_id": f"{parts[-3]}/{parts[-2]}/z{zname[1:]}",
        })
    return out


# --------------------------------------------------------------------------
# 掩码 -> 实例框
# --------------------------------------------------------------------------

def mask_to_instances(mask: np.ndarray, class_id: int, *,
                      min_area: int = 64,
                      max_area_frac: float = 0.95,
                      any_nonzero: bool = True) -> List[Dict[str, Any]]:
    """把一个类别掩码拆成实例（连通分量），返回框与统计。

    坐标约定：``(x, y)``，**0-based 图像坐标**，供下游换算成
    DETR 的归一化 ``cxcywh``（那里需要除以宽高）。
    """
    if ndi is None:
        raise RuntimeError("需要 scipy")
    m = np.asarray(mask)
    if m.ndim != 2:
        m = m.squeeze()
    fg = (m > 0) if any_nonzero else (m == class_id + 1)
    total = int(fg.size)
    if not fg.any():
        return []

    lab, n = ndi.label(fg, structure=np.ones((3, 3), int))
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())
    out: List[Dict[str, Any]] = []
    for idx, area in enumerate(sizes):
        if idx == 0:
            continue
        area = int(area)
        if area < min_area:
            continue
        if total and area / total > max_area_frac:
            continue
        ys, xs = np.nonzero(lab == idx)
        if ys.size == 0:
            continue
        out.append({
            "class_id": int(class_id),
            "x1": int(xs.min()), "y1": int(ys.min()),
            "x2": int(xs.max()), "y2": int(ys.max()),
            "area": area,
            "centroid": [float(xs.mean()), float(ys.mean())],
        })
    return out


def instances_to_boxes_xyxy(inst: Sequence[Dict[str, Any]]) -> np.ndarray:
    """``[{x1,y1,x2,y2}]`` -> ``[N,4]`` float32（像素，0-based，右上开区间）。"""
    if not inst:
        return np.zeros((0, 4), dtype=np.float32)
    a = np.asarray([[i["x1"], i["y1"], i["x2"], i["y2"]] for i in inst],
                   dtype=np.float32)
    a[:, 2] += 1.0        # 闭区间 -> 半开区间
    a[:, 3] += 1.0
    return a


def boxes_to_cxcywh_norm(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    """像素 xyxy -> 归一化 cxcywh（DETR 目标格式）。"""
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    b = boxes.astype(np.float32).copy()
    out = np.zeros_like(b)
    out[:, 0] = (b[:, 0] + b[:, 2]) / 2.0 / w
    out[:, 1] = (b[:, 1] + b[:, 3]) / 2.0 / h
    out[:, 2] = np.maximum(b[:, 2] - b[:, 0], 1.0) / w
    out[:, 3] = np.maximum(b[:, 3] - b[:, 1], 1.0) / h
    return out


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class SliceTargets:
    """把切片转成 DETR 目标（不依赖 torch，便于单测）。"""

    @staticmethod
    def build(rec: Dict[str, Any], cfg: SliceConfig) -> Dict[str, Any]:
        from PIL import Image

        img = Image.open(rec["image"])
        w, h = img.size
        inst: List[Dict[str, Any]] = []
        per_class: Dict[str, int] = {}
        for cid, cname in enumerate(cfg.classes):
            p = rec["masks"].get(cname)
            if not p:
                per_class[cname] = 0
                continue
            m = np.asarray(Image.open(p))
            got = mask_to_instances(m, cid,
                                    min_area=cfg.min_area,
                                    max_area_frac=cfg.max_area_frac,
                                    any_nonzero=cfg.any_nonzero_is_class)
            per_class[cname] = len(got)
            inst.extend(got)

        boxes_px = instances_to_boxes_xyxy(inst)
        return {
            "image_id": rec["image_id"],
            "size": [h, w],
            "boxes_px": boxes_px,
            "boxes_cxcywh": boxes_to_cxcywh_norm(boxes_px, w, h),
            "labels": np.asarray([i["class_id"] for i in inst], dtype=np.int64),
            "num_instances": len(inst),
            "per_class": per_class,
            "num_classes": len(cfg.classes),
        }


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------

def _selftest() -> int:
    """用合成切片验证：多实例拆分、类别 id、框归一化。"""
    try:
        import tempfile
    except ImportError:  # pragma: no cover
        return 1
    from PIL import Image

    tmp = tempfile.mkdtemp(prefix="slice_ds_")
    root = os.path.join(tmp, "cellmap_2d")
    zdir = os.path.join(root, "jrc_x", "crop1", "z0007")
    os.makedirs(zdir, exist_ok=True)

    H = W = 120
    em = (np.random.RandomState(0).rand(H, W) * 255).astype(np.uint8)
    Image.fromarray(em).save(os.path.join(zdir, "em.png"))

    # mito: 两个分开的方块；er: 一个长条
    mito = np.zeros((H, W), np.uint8)
    mito[10:30, 10:30] = 1
    mito[70:90, 80:100] = 1
    Image.fromarray(mito).save(os.path.join(zdir, "mito.png"))

    er = np.zeros((H, W), np.uint8)
    er[10:14, 10:100] = 2
    Image.fromarray(er).save(os.path.join(zdir, "er.png"))

    cfg = SliceConfig(root=root, classes=("mito", "er"), min_area=16)
    slices = find_slices(root, cfg)
    ok = True
    if len(slices) != 1:
        print(f"FAIL 找到 {len(slices)} 个切片，期望 1")
        return 1
    t = SliceTargets.build(slices[0], cfg)
    print("image_id:", t["image_id"], "size:", t["size"])
    print("per_class:", t["per_class"], "labels:", t["labels"].tolist())

    # 断言 1：mito 应有 2 个实例，er 1 个
    if t["per_class"]["mito"] != 2:
        print(f"FAIL mito 实例数 {t['per_class']['mito']}，期望 2")
        ok = False
    else:
        print("PASS mito 拆成 2 个实例")
    if t["per_class"]["er"] != 1:
        print(f"FAIL er 实例数 {t['per_class']['er']}，期望 1")
        ok = False
    else:
        print("PASS er 1 个实例")

    # 断言 2：类别 id 正确（mito=0, er=1）
    labs = t["labels"].tolist()
    if sorted(labs) != [0, 0, 1]:
        print(f"FAIL 类别 id {labs}，期望 [0,0,1]")
        ok = False
    else:
        print("PASS 类别 id 与 classes 顺序一致")

    # 断言 3：归一化框在 (0,1)
    if t["boxes_cxcywh"].size:
        b = t["boxes_cxcywh"]
        if b.min() < 0 or b.max() > 1:
            print(f"FAIL 归一化框越界：min={b.min():.3f} max={b.max():.3f}")
            ok = False
        else:
            print("PASS 归一化框都在 (0,1)")
        # 断言 4：第一个 mito 框应覆盖 10..30 -> 宽 20
        w_px = b[0, 2] * W
        if abs(w_px - 20) > 1:
            print(f"FAIL 第一个框宽 {w_px:.1f}px，期望约 20")
            ok = False
        else:
            print(f"PASS 第一个框宽 {w_px:.1f}px ≈ 20")

    # 断言 5：min_area 过滤生效
    cfg2 = SliceConfig(root=root, classes=("mito", "er"), min_area=10_000)
    t2 = SliceTargets.build(slices[0], cfg2)
    if t2["num_instances"] != 0:
        print(f"FAIL min_area 未生效，仍有 {t2['num_instances']} 个实例")
        ok = False
    else:
        print("PASS min_area 过滤生效")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("\n结果：", "全部通过" if ok else "有失败项")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CellMap 2D 切片 -> DETR 多类目标")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=None, help="切片根目录")
    ap.add_argument("--classes", default="mito,er")
    ap.add_argument("--min-area", type=int, default=64)
    ap.add_argument("--dump", default=None, help="把首个切片的目标写到该 JSON")
    args = ap.parse_args(argv)

    if args.selftest or not args.root:
        return _selftest()

    cfg = SliceConfig(root=args.root,
                      classes=tuple(s.strip() for s in args.classes.split(",")
                                    if s.strip()),
                      min_area=args.min_area)
    slices = find_slices(args.root, cfg)
    print(f"找到 {len(slices)} 个切片，类别 {cfg.classes}")
    if not slices:
        return 2
    tot: Dict[str, int] = {c: 0 for c in cfg.classes}
    n_inst = 0
    for rec in slices:
        t = SliceTargets.build(rec, cfg)
        n_inst += t["num_instances"]
        for k, v in t["per_class"].items():
            tot[k] += v
    print(f"实例总数 {n_inst}；按类别 {tot}")
    if args.dump:
        t = SliceTargets.build(slices[0], cfg)
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump({**{k: v for k, v in t.items()
                          if k in ("image_id", "size", "num_instances",
                                   "per_class", "num_classes")},
                       "labels": t["labels"].tolist(),
                       "boxes_cxcywh": t["boxes_cxcywh"].tolist()},
                      fh, ensure_ascii=False, indent=1)
        print(f"已写出 {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
