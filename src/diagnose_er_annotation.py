#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_er_annotation.py —— 检查 ER 标注本身是否完整、可用

## 为什么需要这个

同器官分割实验给出了关键事实：**er 的 Dice 只有 0.077，和跨器官的 0.066 几乎一样。**
这排除了"域偏移"和"表示方式"两个解释，剩下两个候选：

  (a) 标注本身有问题（只标了一部分、定义不一致、边界不可靠）
  (b) 训练预算不足（60 epoch / 256px 对细管网不够）

本脚本检验 (a)。判据是**标注的空间完整性**：

* **z 覆盖率**：ER 出现在多少比例的 z 层里。若只有少数层有 ER，
  说明是「部分标注」，模型看到的正样本自相矛盾（同样是 ER 区域，有的层标了有的没标），
  这会让 Dice 天然上不去。
* **与 mito 对比**：mito 若 z 覆盖率高而 ER 低，说明不是整块体积没标，
  而是 ER 这一类被单独漏标。
* **连通分量数**：ER 是管网，分量数天然高；但若高到上千且每个很小，
  说明标注高度碎片化，"实例"没有意义（这与检测 AP 接近 0 一致）。
* **前景占比**：极低（< 1%）说明可能只标了少量典型区域。

## 用法

    python diagnose_er_annotation.py --labels data/labels
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover
    ndi = None

FACE6 = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                  [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                  [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=bool)


def analyse(vol: np.ndarray) -> dict:
    m = np.asarray(vol) > 0
    n = int(m.sum())
    if n == 0:
        return {"voxels": 0, "frac": 0.0, "z_cov": 0.0, "components": 0,
                "largest_frac": 0.0, "bbox_frac": 0.0}
    z, y, x = m.shape
    per_z = m.sum(axis=(1, 2))
    z_cov = float((per_z > 0).mean())
    lab, ncomp = ndi.label(m, structure=FACE6)
    sizes = np.bincount(lab.ravel())[1:] if ncomp else np.array([0])
    zs, ys, xs = np.nonzero(m)
    bbox = ((zs.max() - zs.min() + 1) * (ys.max() - ys.min() + 1)
            * (xs.max() - xs.min() + 1))
    return {"voxels": n, "frac": round(n / m.size, 5),
            "z_cov": round(z_cov, 4), "components": int(ncomp),
            "largest_frac": round(float(sizes.max() / n), 4),
            "bbox_frac": round(bbox / m.size, 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels")
    ap.add_argument("--out", default="logs/ER_DIAGNOSIS.txt")
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy", file=sys.stderr)
        return 2

    pairs = {}
    for p in sorted(glob.glob(os.path.join(args.labels, "*", "*", "*.npy"))):
        parts = p.replace("\\", "/").split("/")
        lab = os.path.splitext(parts[-1])[0]
        crop = parts[-2]
        ds = parts[-3]
        pairs.setdefault((ds, crop), {})[lab] = p

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 92)
    emit("ER 标注完整性诊断")
    emit("=" * 92)
    emit()
    emit(f"{'dataset/crop':<34} {'类':<6} {'占比%':>7} {'z覆盖':>7} "
         f"{'分量数':>8} {'最大分量占比':>13} {'bbox占比':>9}")
    emit("-" * 92)

    flagged = []
    for (ds, crop), labs in sorted(pairs.items()):
        stats = {}
        for lab in ("mito", "er"):
            if lab not in labs:
                continue
            v = np.load(labs[lab])
            st = analyse(v)
            stats[lab] = st
            emit(f"{ds+'/'+crop:<34} {lab:<6} {st['frac']*100:>7.3f} "
                 f"{st['z_cov']*100:>6.1f}% {st['components']:>8} "
                 f"{st['largest_frac']:>13.3f} {st['bbox_frac']:>9.3f}")
        # 判据：ER 的 z 覆盖率明显低于 mito，或 ER 占比极低
        if "er" in stats and "mito" in stats:
            e, m = stats["er"], stats["mito"]
            why = []
            if m["z_cov"] - e["z_cov"] > 0.25:
                why.append(f"ER z覆盖比mito低{100*(m['z_cov']-e['z_cov']):.0f}个百分点")
            if e["frac"] < 0.01:
                why.append("ER 占比<1%")
            if e["components"] > 300:
                why.append(f"ER 碎裂为 {e['components']} 个分量")
            if why:
                flagged.append((f"{ds}/{crop}", why))
        emit()

    emit("=" * 92)
    emit("判定")
    emit("=" * 92)
    if flagged:
        emit(f"有 {len(flagged)} 个 crop 的 ER 标注存在可疑迹象：")
        for name, why in flagged:
            emit(f"  {name:<34} {'; '.join(why)}")
        emit()
        emit("解读指引：")
        emit("  * 『ER z覆盖比 mito 低很多』-> 很可能是**部分标注**：")
        emit("    同样的组织里 mito 标了、ER 没标，模型看到自相矛盾的监督信号，")
        emit("    这足以让 Dice 卡在 0.1 以下，且**换模型/加训练都救不回来**。")
        emit("  * 『ER 碎裂为数百个分量』-> 『实例』对 ER 没有意义（与检测 AP≈0 一致），")
        emit("    但这不影响语义分割；分割看的是像素，不看分量。")
    else:
        emit("未发现明显异常：ER 的占比与 z 覆盖率都在合理范围，")
        emit("碎裂程度也在预期内（管网本就如此）。")
        emit("=> 若如此，ER 的分割失败更可能是**训练预算不足**（epoch/分辨率），")
        emit("   而不是标注问题，应优先做长训练对照。")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
