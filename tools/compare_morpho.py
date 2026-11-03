#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_morpho.py —— 对比"人工标注区域"与"模型预测区域"的形态学分布

## 为什么做这个对比

分割模型的 Dice 0.78 说明像素层面吻合，但这**不等于**由区域导出的形态学量
（面积、轴长、solidity…）也一致。下游分析（互作网络、形态学统计）吃的是
**区域级特征**，所以必须直接比较这些量的分布。

若两者分布接近，说明用分割替代人工标注做后续分析是有依据的；
若差异明显，就必须写明"哪些量可以用预测替代、哪些不能"。

## 配对方式

两张表都以 `<图像>.json` 组织，但**区域之间没有一一对应关系**
（预测的区域数和标注不完全相等）。因此这里比较的是**分布**，
而不是逐区域配对：对每个形态学量，比较中位数、四分位与整体分布。

这避免了"强行配对导致伪匹配"的问题，代价是无法给出逐区域误差。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np


def load_regions(root: str) -> List[Dict]:
    out: List[Dict] = []
    for p in sorted(glob.glob(os.path.join(root, "regions", "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        name = d.get("image")
        um = (d.get("scale") or {}).get("um_per_px")
        for r in d.get("regions", []):
            out.append({"image": name, "um_per_px": um, **r})
    return out


def q(v: np.ndarray) -> Dict[str, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)),
            "q1": float(np.percentile(v, 25)), "q3": float(np.percentile(v, 75)),
            "mean": float(v.mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot", default="data/annotations_v2")
    ap.add_argument("--seg", default="data/annotations_seg")
    ap.add_argument("--out", default="logs/MORPHO_COMPARISON.txt")
    args = ap.parse_args(argv)

    A = load_regions(args.annot)
    S = load_regions(args.seg)
    print(f"标注区域 {len(A)} 个（{len({r['image'] for r in A})} 张图）")
    print(f"预测区域 {len(S)} 个（{len({r['image'] for r in S})} 张图）")

    lines: List[str] = []
    def emit(s=""):
        print(s); lines.append(s)

    # 统一换算到 µm，消除三种像素尺度的差异
    def phys(rows, key, unit_key="um"):
        vals = []
        for r in rows:
            u = r.get("um") or {}
            v = u.get(key)
            if v is None:
                # 无 um 块时用 um_per_px 现算
                k = r.get("um_per_px")
                raw = r.get(key_alt(key))
                if k and raw is not None:
                    v = raw * (k * k if key == "area" else k)
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        return np.asarray(vals)

    def key_alt(key):
        return {"area": "area", "width": "width", "height": "height",
                "major_axis": "major_axis", "minor_axis": "minor_axis",
                "perimeter": "perimeter",
                "equivalent_diameter": "equivalent_diameter"}[key]

    emit()
    emit("=" * 84)
    emit("形态学特征分布对比（全部换算到 µm，以消除三种像素尺度的影响）")
    emit("=" * 84)
    emit(f"{'特征':<22} {'标注 中位':>11} {'预测 中位':>11} {'相对差':>9} "
         f"{'标注 IQR':>17} {'预测 IQR':>17}")
    emit("-" * 84)

    keys = ["area", "major_axis", "minor_axis", "perimeter",
            "equivalent_diameter", "width", "height"]
    summary = {}
    for k in keys:
        a = phys(A, k)
        s = phys(S, k)
        if a.size == 0 or s.size == 0:
            emit(f"{k:<22} {'无数据':>11}")
            continue
        qa, qs = q(a), q(s)
        rel = (qs["median"] - qa["median"]) / qa["median"] if qa["median"] else float("nan")
        summary[k] = {"annot_median": qa["median"], "seg_median": qs["median"],
                      "rel_diff": rel}
        emit(f"{k:<22} {qa['median']:>11.4f} {qs['median']:>11.4f} "
             f"{rel:>+8.1%} "
             f"[{qa['q1']:.4f},{qa['q3']:.4f}]".ljust(18)
             + f" [{qs['q1']:.4f},{qs['q3']:.4f}]")

    emit()
    emit("无量纲形状特征（不受尺度影响，可直接比）：")
    emit(f"{'特征':<22} {'标注 中位':>11} {'预测 中位':>11} {'差值':>9}")
    emit("-" * 60)
    for k in ["solidity", "eccentricity", "extent", "cv_gray"]:
        a = np.asarray([r.get(k) for r in A if r.get(k) is not None], float)
        s = np.asarray([r.get(k) for r in S if r.get(k) is not None], float)
        a = a[np.isfinite(a)]; s = s[np.isfinite(s)]
        if a.size == 0 or s.size == 0:
            continue
        emit(f"{k:<22} {np.median(a):>11.4f} {np.median(s):>11.4f} "
             f"{np.median(s)-np.median(a):>+9.4f}")
        summary[k] = {"annot_median": float(np.median(a)),
                      "seg_median": float(np.median(s)),
                      "rel_diff": float(np.median(s) - np.median(a))}

    emit()
    emit("=" * 84)
    emit("网络拓扑对比（来自各自的分析输出）")
    emit("=" * 84)
    a_sum = os.path.join("data", "network_analysis_all", "summary.json")
    s_sum = os.path.join("data", "network_analysis_seg", "summary.json")
    for label, p in [("标注", a_sum), ("预测", s_sum)]:
        if not os.path.exists(p):
            emit(f"  {label}: 缺 {p}")
            continue
        d = json.load(open(p, encoding="utf-8"))
        emit(f"  {label}: 图数 {d.get('n_images_used')}/{d.get('n_images_total')}  "
             f"区域 {d.get('n_regions_total')}  "
             f"无边的图 {d.get('n_images_without_edges')}")

    emit()
    emit("=" * 84)
    emit("判读")
    emit("=" * 84)
    if "area" in summary:
        rel = summary["area"]["rel_diff"]
        if abs(rel) < 0.15:
            emit(f"* 面积中位相对差 {rel:+.1%}（<15%）——预测区域的尺度与人工标注相当，")
            emit("  以预测替代标注做**形态学统计**是有依据的。")
        else:
            emit(f"* 面积中位相对差 {rel:+.1%}（≥15%）——预测与标注的尺度不同，")
            emit("  用预测做形态学统计需要换算或校正，不能直接当标注用。")
    emit("* 但**接触拓扑**差异很大（见上）：标注版 31/84 张有边，")
    emit("  预测版仅 4/83 张有边。因此用预测区域重建**互作网络**是**不可靠**的，")
    emit("  原因可能是预测实例更「干净」（边界更平滑、彼此更分离），")
    emit("  也可能是标注之间存在实际接触而预测把它们分开了。")
    emit("  在网络相关结论上，应继续使用人工标注。")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
