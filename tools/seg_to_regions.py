#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seg_to_regions.py —— 把分割掩码转成 `annotations_v2` 同构的 regions JSON

## 为什么要这一步

项目已有的分析管线（`tools/network/run_analysis.py`）吃的是
`data/annotations_v2/regions/<image>.json`，里面每个 region 带质心、面积、
轮廓、灰度统计等字段。分割模型产出的是**二值掩码**，两者格式不通，
所以需要一层转换把掩码接进这条管线。

转换后可以直接跑：

    python tools/network/run_analysis.py --annotations data/annotations_seg \\
        --out data/network_analysis_seg

## 与标注版的差异（必须在报告里写明）

| 字段 | 标注版 | 掩码版 |
|---|---|---|
| `organelle_type` | OCR 识别出的字母（M/Y 等） | 统一写 `"M"`（模型只做线粒体） |
| `is_labeled` | 是否成功关联到标注文字 | 恒 `True`（预测本身就是"这是线粒体"） |
| `recognized_text` / `ocr_raw` | 原始文字 | 空字符串 |
| `has_label_pixels` / `label_distance_px` | 有实际含义 | `False` / `None` |
| `source`（dataset.json 的 config） | `extract_regions.py` | 本脚本 |

其余几何/灰度特征由 `tools/morpho/features.py` 的 `compute_region_features`
计算，**与标注版完全同一套实现**，因此两边的特征可直接对比。

> `equivalent_diameter` 沿用原实现的定义 `4*area/perimeter`（不是等效圆直径），
> 这里不改动，以免与既有分析结果不可比。

## 用法

    python tools/seg_to_regions.py \\
        --masks data/masks_ourdata_clean --images data/原图片 \\
        --annot data/annotations_v2/dataset.json --out data/annotations_seg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from morpho.features import (  # noqa: E402
    COMPACT_FEATURE_NAMES, EXTENDED_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION, compute_region_features)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default="data/masks_ourdata_clean")
    ap.add_argument("--images", default="data/原图片")
    ap.add_argument("--annot", default="data/annotations_v2/dataset.json")
    ap.add_argument("--out", default="data/annotations_seg")
    ap.add_argument("--organelle", default="M",
                    help="写入 organelle_type 的值；模型只分割线粒体，故默认 M")
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy", file=sys.stderr)
        return 2

    # 标定信息沿用标注版数据集里的 um_per_px（同一批图，同一标定）
    meta = json.load(open(args.annot, encoding="utf-8"))
    ents = meta["images"] if isinstance(meta["images"], list) else list(meta["images"].values())
    scale_of = {}
    for e in ents:
        um = e.get("um_per_px")
        scale_of[e["image"]] = {
            "um_per_px": um,
            "ok": bool(um) and um > 0,
            "bar_px": None, "bar_value": None, "bar_unit": "um",
            "source": "annotations_v2", "note": "",
        }

    os.makedirs(os.path.join(args.out, "regions"), exist_ok=True)
    manifest: List[Dict] = []
    failures: List[Dict] = []
    total = 0

    mask_files = sorted(f for f in os.listdir(args.masks) if f.lower().endswith(".png"))
    print(f"掩码 {len(mask_files)} 个，输出到 {args.out}/")

    for f in mask_files:
        name = os.path.splitext(f)[0]
        mp = os.path.join(args.masks, f)
        ip = os.path.join(args.images, f"{name}.tif")
        try:
            m = np.asarray(Image.open(mp)) > 127
            gray = (np.asarray(Image.open(ip).convert("L"), dtype=np.float64)
                    if os.path.exists(ip) else None)
            if gray is not None and gray.shape != m.shape:
                # 形状不一致时按掩码尺寸缩放灰度（极少见，但会静默出错，故显式处理）
                g = Image.open(ip).convert("L").resize(
                    (m.shape[1], m.shape[0]), Image.BILINEAR)
                gray = np.asarray(g, dtype=np.float64)

            lab, n = ndi.label(m, structure=np.ones((3, 3)))
            scale = scale_of.get(name) or {"um_per_px": None, "ok": False,
                                           "bar_px": None, "bar_value": None,
                                           "bar_unit": "um", "source": "unknown",
                                           "note": "缺标定"}
            regions: List[Dict] = []
            for rid in range(1, n + 1):
                rm = (lab == rid)
                if rm.sum() == 0:
                    continue
                feat = compute_region_features(
                    rm, gray=gray, label_mask=None,
                    region_id=rid, recognized_text="",
                    organelle_type=args.organelle)
                d = feat.to_dict()
                d["ocr_raw"] = ""
                # 语义调整：掩码版没有"标签文字"概念，
                # is_labeled 表示"该区域被判为目标类别"，恒为 True
                d["is_labeled"] = True
                d["has_label_pixels"] = False
                d["label_distance_px"] = None
                d["source"] = "segmentation"
                if scale.get("ok") and scale.get("um_per_px"):
                    k = float(scale["um_per_px"])
                    d["um"] = {
                        "per_px": k,
                        "width": d["width"] * k,
                        "height": d["height"] * k,
                        "center_x": d["center_x"] * k,
                        "center_y": d["center_y"] * k,
                        "major_axis": d["major_axis"] * k,
                        "minor_axis": d["minor_axis"] * k,
                        "equivalent_diameter": d["equivalent_diameter"] * k,
                        "area": d["area"] * k * k,
                        "perimeter": d["perimeter"] * k,
                    }
                else:
                    d["um"] = None
                regions.append(d)

            record = {
                "image": name,
                "image_shape": [int(m.shape[0]), int(m.shape[1])],
                "original_file": os.path.basename(ip) if os.path.exists(ip) else None,
                "scale": scale,
                "geometry_note": "",
                "num_regions": len(regions),
                "num_labeled": len(regions),
                "regions": regions,
                "source": "segmentation",
            }
            with open(os.path.join(args.out, "regions", f"{name}.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False, indent=1)

            flags: Dict[str, int] = {}
            for r in regions:
                q = r.get("quality_flag", "ok")
                flags[q] = flags.get(q, 0) + 1
            manifest.append({
                "image": name,
                "file": f"regions/{name}.json",
                "num_regions": len(regions),
                "num_labeled": len(regions),
                "quality_flags": flags,
                "scale_ok": bool(scale.get("ok")),
                "um_per_px": scale.get("um_per_px"),
                "geometry_note": "",
            })
            total += len(regions)
        except Exception as exc:  # noqa: BLE001
            failures.append({"image": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [失败] {name}: {type(exc).__name__}: {exc}")

    manifest.sort(key=lambda m: m["image"])
    bad_scale = [m["image"] for m in manifest if not m["scale_ok"]]
    dataset = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "generator": "tools/seg_to_regions.py",
        "config": {
            "masks_dir": os.path.abspath(args.masks),
            "originals_dir": os.path.abspath(args.images),
            "segmentation": "model_prediction (U-Net, 4 nm/px, thr=0.4) + min-area 0.05 um2",
            "coordinate_convention": "1-based (x, y)",
            "organelle_type": args.organelle,
            "note": ("掩码来自自有数据训练的线粒体分割模型；"
                     "is_labeled 恒 True 表示被判为目标类别，不代表有人工标签文字"),
        },
        "feature_schemas": {
            "compact_14": list(COMPACT_FEATURE_NAMES),
            "extended_13": list(EXTENDED_FEATURE_NAMES),
        },
        "summary": {
            "num_images": len(manifest),
            "num_regions": total,
            "num_failures": len(failures),
            "images_with_bad_scale": bad_scale,
            "mean_regions_per_image": round(total / max(len(manifest), 1), 2),
        },
        "images": manifest,
        "failures": failures,
    }
    with open(os.path.join(args.out, "dataset.json"), "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=1)

    print()
    print(f"完成：{len(manifest)} 张图，{total} 个区域，失败 {len(failures)}")
    print(f"每图平均区域数 {total / max(len(manifest), 1):.2f}")
    if bad_scale:
        print(f"标定不可信（会被下游跳过微米量）: {bad_scale}")
    print(f"输出 {args.out}/dataset.json 与 {args.out}/regions/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
