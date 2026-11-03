#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_color_separation.py —— 检验 `code/separate_annotation_colors.m` 的颜色分离损失

## 背景

`separate_annotation_colors.m:85-87` 用**三通道精确相等**提取颜色::

    mask = (im(:,:,1) == 237) & (im(:,:,2) == 28) & (im(:,:,3) == 36);

绘图工具写出的标注是抗锯齿的：字符边缘的像素是前景色与背景色的混合，
既不是纯 (255,242,0)，也不是纯灰。精确相等会**丢掉这些边缘像素**，
把一个字符切成若干碎片。

实测影响（`11_0002`，标注 "M"）：

| 组件大小（像素） | 全库出现次数 | 说明 |
|---|---|---|
| 90 | 263 | 完整的 "M" 字形（`11_0002` 有 2 个这样的大标签） |
| 12 / 11 / 4 | 522 / 521 / 521 | **同一个 "M" 被切成三块**（左竖、右竖、中竖） |
| 2 / 3 | 1608 / 542 | 更细的碎片 |
| 1 | 41 | 单像素 |

也就是说：**精确相等把约 87% 的标签切成了碎片**。

## 本脚本做什么

对每张分离出的绿色/黄色掩码，与"从原始标注图用容差阈值重新分离"的结果对比，
量化丢了多少像素、连通域数量膨胀了多少倍。

这里用容差阈值而不是精确相等：

    mask = (R > 200) & (G > 200) & (B < 100)      # 黄色系
    mask = (R < 100) & (G > 120) & (G < 220) & (B < 150)   # 绿色系

## 用法

    python tools/data_integrity/check_color_separation.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "morpho"))


#: 原始标注图中定义的三种颜色（见 separate_annotation_colors.m:31-33）
TARGET_COLORS = {
    "green": (0, 180, 75),
    "yellow": (255, 242, 0),
    "red": (237, 28, 36),
}


def exact_mask(rgb: np.ndarray, color: tuple) -> np.ndarray:
    """原实现的判定方式：三通道精确相等。"""
    return (
        (rgb[:, :, 0] == color[0])
        & (rgb[:, :, 1] == color[1])
        & (rgb[:, :, 2] == color[2])
    )


def tolerant_mask(rgb: np.ndarray, color: tuple, tol: int = 55) -> np.ndarray:
    """容差判定：允许抗锯齿造成的通道偏移。"""
    r = rgb.astype(np.int16)
    diff = np.abs(r - np.array(color, dtype=np.int16))
    # 用切比雪夫距离（各通道最大偏差）作为判据
    return (diff.max(axis=2) <= tol)


def _component_stats(mask: np.ndarray) -> Dict[str, Any]:
    from scipy import ndimage as ndi

    lab, n = ndi.label(mask, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([], dtype=int)
    return {
        "pixels": int(mask.sum()),
        "components": int(n),
        "largest": int(sizes.max()) if sizes.size else 0,
        "components_ge_20px": int((sizes >= 20).sum()) if sizes.size else 0,
    }


def analyse_image(annotated_path: str, separated_path: str,
                  color_name: str, tol: int) -> Optional[Dict[str, Any]]:
    from PIL import Image

    if not os.path.exists(annotated_path) or not os.path.exists(separated_path):
        return None

    rgb = np.asarray(Image.open(annotated_path).convert("RGB"))
    sep = np.any(np.asarray(Image.open(separated_path).convert("RGB")) > 0, axis=2)

    color = TARGET_COLORS[color_name]
    ex = exact_mask(rgb, color)
    tl = tolerant_mask(rgb, color, tol)

    ex_stats = _component_stats(ex)
    tl_stats = _component_stats(tl)
    sep_stats = _component_stats(sep)

    # 容差掩码里有多少像素是精确掩码漏掉的
    missed = int((tl & ~ex).sum())
    return {
        "color": color_name,
        "tolerance": tol,
        "exact": ex_stats,
        "tolerant": tl_stats,
        "separated_file": sep_stats,
        "pixels_lost_by_exact_match": missed,
        "pixels_lost_ratio": round(missed / max(tl_stats["pixels"], 1), 4),
        "component_fragmentation_factor": round(
            ex_stats["components"] / max(tl_stats["components"], 1), 3
        ),
        "separated_matches_tolerant": bool(
            abs(sep_stats["pixels"] - tl_stats["pixels"]) <= max(5, 0.02 * tl_stats["pixels"])
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    p = argparse.ArgumentParser(description="量化颜色分离的像素损失")
    p.add_argument("--annotated", default=os.path.join(root, "data", "标注图片"))
    p.add_argument("--separated", default=os.path.join(root, "data", "分离的标注信息"))
    p.add_argument("--tol", type=int, default=55)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--out", default=os.path.join(root, "docs", "color-separation-report.json"))
    args = p.parse_args(argv)

    from extract_regions import list_pairs

    pairs = list_pairs(args.separated)[: args.limit]
    if not pairs:
        print("未找到成对文件", file=sys.stderr)
        return 2

    rows: List[Dict[str, Any]] = []
    for name, _, _ in pairs:
        annotated = None
        for ext in (".tif", ".png", ".jpg"):
            cand = os.path.join(args.annotated, name + ext)
            if os.path.exists(cand):
                annotated = cand
                break
        if annotated is None:
            continue
        for color_name in ("green", "yellow", "red"):
            sep_path = os.path.join(args.separated, f"{color_name}_{name}.tif")
            rec = analyse_image(annotated, sep_path, color_name, args.tol)
            if rec is None:
                continue
            rec["image"] = name
            rows.append(rec)

    if not rows:
        print("没有可对比的数据", file=sys.stderr)
        return 2

    print(f"{'image':>12} {'color':>7} {'exact_px':>9} {'tol_px':>8} "
          f"{'lost':>6} {'lost%':>7} {'comps_ex':>9} {'comps_tol':>10} {'frag':>6}")
    for r in rows:
        print(f"{r['image']:>12} {r['color']:>7} {r['exact']['pixels']:>9} "
              f"{r['tolerant']['pixels']:>8} {r['pixels_lost_by_exact_match']:>6} "
              f"{r['pixels_lost_ratio'] * 100:>6.1f}% {r['exact']['components']:>9} "
              f"{r['tolerant']['components']:>10} {r['component_fragmentation_factor']:>6.2f}")

    # 汇总
    summary: Dict[str, Dict[str, float]] = {}
    for color_name in ("green", "yellow", "red"):
        sub = [r for r in rows if r["color"] == color_name]
        if not sub:
            continue
        summary[color_name] = {
            "images": len(sub),
            "mean_pixels_lost_ratio": round(
                float(np.mean([r["pixels_lost_ratio"] for r in sub])), 4
            ),
            "mean_fragmentation_factor": round(
                float(np.mean([r["component_fragmentation_factor"] for r in sub])), 3
            ),
            "all_separated_match_tolerant": all(
                r["separated_matches_tolerant"] for r in sub
            ),
        }

    print("\n汇总（精确相等相对容差阈值的损失）:")
    for k, v in summary.items():
        print(f"  {k:>7}: 平均丢 {v['mean_pixels_lost_ratio'] * 100:.1f}% 像素, "
              f"连通域膨胀 {v['mean_fragmentation_factor']:.2f}×")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "tolerance": args.tol,
            "target_colors": TARGET_COLORS,
            "summary": summary,
            "per_image": rows,
        }, fh, ensure_ascii=False, indent=1)
    print(f"\n输出：{os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
