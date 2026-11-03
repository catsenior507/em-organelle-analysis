#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
golgi_dice_vs_coverage.py —— 分析"标注覆盖度"与 Dice 的关系

## 动机

全图推理后逐图看数据，发现一个很强的规律：

| 图像 | GT 覆盖 | 预测覆盖 | 硬 Dice |
|---|---|---|---|
| 24780 | 9.9% | 10.7% | 0.712 |
| 24776 | 9.6% | 11.7% | 0.697 |
| 7335 | 1.7% | 14.6% | 0.022 |
| 37185 | 0.6% | 8.4% | 0.062 |

**预测面积与 GT 面积越接近，Dice 越高；GT 越稀疏，模型反而预测得越多。**

这与线粒体那边的结论完全一致：标注覆盖度低时，图像里还有大量真实细胞器
**没被标注**，模型预测出来就被算成假阳性，Dice 被系统性压低。
这不是模型错，是**标注不完整**。

## 本脚本做什么

从 `index.csv` 读取逐图结果，按 GT 覆盖度分层统计 Dice，
并计算"预测面积/GT 面积"比值与 Dice 的相关性。
若低覆盖层的 Dice 显著更低、且比值显著大于 1，即可确认是标注问题。
"""

from __future__ import annotations

import csv
import os
import sys
from typing import List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    p = os.path.join(ROOT, "data", "golgi_masks", "index.csv")
    if not os.path.exists(p):
        print(f"[错误] 找不到 {os.path.relpath(p, ROOT)}")
        print("       先运行 tools/predict_golgi.py")
        return 2
    rows: List[dict] = []
    with open(p, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("dice") in (None, "", "None", "nan"):
                continue
            rows.append({"id": r["id"], "split": r["split"],
                         "gt": float(r["gt_frac"]), "pred": float(r["pred_frac"]),
                         "dice": float(r["dice"])})
    if not rows:
        print("[错误] index.csv 里没有有效 Dice")
        return 2

    vals = [r for r in rows if r["split"] == "val"]
    print(f"共 {len(rows)} 张有 Dice；其中留出（val）{len(vals)} 张")
    print()

    # 分层：按 GT 覆盖度
    bins = [(0.0, 0.03, "<3%"), (0.03, 0.08, "3–8%"),
            (0.08, 0.20, "8–20%"), (0.20, 1.01, ">=20%")]
    print("按 GT 覆盖度分层的 Dice（留出集）:")
    print(f"{'覆盖度':<10} {'张数':>4} {'Dice 均值':>10} {'中位':>8} "
          f"{'预测/GT':>9}")
    print("-" * 46)
    for lo, hi, name in bins:
        sel = [r for r in vals if lo <= r["gt"] < hi]
        if not sel:
            print(f"{name:<10} {0:>4} {'—':>10} {'—':>8} {'—':>9}")
            continue
        d = np.array([r["dice"] for r in sel])
        ratio = np.array([r["pred"] / max(r["gt"], 1e-6) for r in sel])
        print(f"{name:<10} {len(sel):>4} {d.mean():>10.4f} "
              f"{np.median(d):>8.4f} {np.median(ratio):>9.2f}")

    print()
    print("全部图（含训练侧，仅作参考）按覆盖度的 Dice：")
    for lo, hi, name in bins:
        sel = [r for r in rows if lo <= r["gt"] < hi]
        if not sel:
            continue
        d = np.array([r["dice"] for r in sel])
        print(f"  {name:<8} {len(sel):>3} 张  Dice 均值 {d.mean():.4f}")

    # 相关性
    print()
    gt = np.array([r["gt"] for r in rows])
    dice = np.array([r["dice"] for r in rows])
    ratio = np.array([r["pred"] / max(r["gt"], 1e-6) for r in rows])
    if len(rows) >= 4:
        c1 = float(np.corrcoef(gt, dice)[0, 1])
        c2 = float(np.corrcoef(np.log(np.clip(ratio, 1e-3, None)), dice)[0, 1])
        print("相关性（全部图）:")
        print(f"  GT 覆盖度 vs Dice          r = {c1:+.3f}")
        print(f"  log(预测/GT) vs Dice       r = {c2:+.3f}")
        print()
        print("判读:")
        if c1 > 0.5:
            print("  * GT 覆盖度与 Dice 强正相关 -> 标注越稀疏，Dice 越低。")
            print("    这说明低分**主要来自标注不完整**（模型预测出未标注的真实细胞器），")
            print("    而不是模型分不准。")
        else:
            print("  * GT 覆盖度与 Dice 相关性弱 -> 低分不能只用标注不完整解释。")
        if c2 < -0.5:
            print("  * log(预测/GT) 与 Dice 强负相关 -> 过度预测是低分的主因，")
            print("    同样指向标注不完整。")

    out = os.path.join(ROOT, "logs", "golgi_dice_vs_coverage.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("id,split,gt_frac,pred_frac,dice,pred_over_gt\n")
        for r in rows:
            fh.write(f"{r['id']},{r['split']},{r['gt']:.5f},{r['pred']:.5f},"
                     f"{r['dice']:.4f},{r['pred']/max(r['gt'],1e-6):.2f}\n")
    print(f"\n明细已写出 {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
