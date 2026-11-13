#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contact_local.py —— 局部接触分析：块内环面平移零模型

## 为什么需要局部版本

全视野的环面零模型给出 z 中位 −1.90（范围 −5.83 … −0.28），即**每一对细胞器
都比随机摆位更少接触**。这个结果本身没错，但它回答的不是"膜是否接触"，
而是"细胞器是否分区占据"：

这些 crop 横跨多个细胞，高尔基体偏一侧、溶酶体偏另一侧。任何随机平移都会把
一类挪到另一类不存在的区域，于是零模型的接触量必然远高于实测。
**大尺度的区域分隔主导了统计量，把局部接触淹没了。**

## 做法

把体积切成边长 L 的重叠方块（默认 96 体素 = 384 nm，步长 L/2）。对每块：

1. 只保留**两类都至少 min_vox 个体素**的块（默认各 200）；
2. 在**块内**做环面平移零模型：
   - 先把 A 膨胀一次，得到 `dilA`
   - 对 B 做 K 次随机环面平移（`np.roll`），每次算 `(rolled_B & dilA).sum()`
   - 得到零分布；实测值 = `(B & dilA).sum()`
3. 记录富集比 `obs / mean(null)` 与单侧 p 值。

块内平移意味着：**大尺度分隔被块边界吸收**——同一个 384 nm 的小块里两类本来就
都在，于是比较的是"在这个局部环境里，两类是否比随机摆位更贴近"。

## 为什么用 np.roll 而不是 FFT

FFT 能给全部 n³ 个平移的精确分布，但块小（96³ ≈ 88 万体素）时
采样 200 次已足够（p 的采样误差 ≈ 1/√200 ≈ 0.07，对判断"是否显著"够用），
而代码复杂度低得多、也不易写错。全视野那一步用的是项目里已有的 FFT 版本
（`contact_analysis_3d.py --null-mode fft`），这里不重复实现。

## 敏感性

块大小与最小体素数会显著影响结果，因此必须扫参数而不是只报一个设置。
`--blocks` 与 `--min-voxes` 接受逗号分隔的多个值，脚本逐组合跑并对比。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

ST6 = np.zeros((3, 3, 3), bool)
ST6[1, 1, 1] = True
ST6[0, 1, 1] = ST6[2, 1, 1] = True
ST6[1, 0, 1] = ST6[1, 2, 1] = True
ST6[1, 1, 0] = ST6[1, 1, 2] = True


def load(d: str, cls: str) -> Optional[np.ndarray]:
    p = os.path.join(d, f"{cls}.npy")
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a == 1) | ((a > 0) & (a != 255))


def analyse_block(A: np.ndarray, B: np.ndarray, n_roll: int,
                  rng: np.random.Generator, tol: int = 2) -> Optional[Dict[str, Any]]:
    """单块：返回实测接触、零分布统计与 p 值。

    `tol` 是容差（体素）。**不能只用 tol=1 的严格相邻**：语义标注在膜两侧
    各留了 1–2 体素空隙，严格相邻的计数在 0 与个位数之间跳，富集比退化成
    0 或 ∞，失去分辨力。默认 tol=2（4 nm 体素 → 8 nm）让计数平滑，
    也更接近"膜间距"这一物理含义。
    """
    if A.sum() == 0 or B.sum() == 0:
        return None
    dilA = binary_dilation(A, ST6, iterations=max(1, tol))
    obs = float((dilA & B).sum())
    shape = A.shape
    null = np.empty(n_roll, np.float32)
    for i in range(n_roll):
        sh = (int(rng.integers(shape[0])),
              int(rng.integers(shape[1])),
              int(rng.integers(shape[2])))
        null[i] = (np.roll(B, sh, axis=(0, 1, 2)) & dilA).sum()
    mu = float(null.mean())
    # 单侧 p：零分布中 >= 实测的比例（+1 校正，避免 p 恰为 0）
    p = float((1 + (null >= obs).sum()) / (1 + n_roll))
    return {"obs": obs, "null_mean": mu,
            "null_sd": float(null.std()),
            "enrichment": (obs / mu) if mu > 0 else None,
            "p_value": p, "tol": tol}


def run_pair(crop_dir: str, pairs: List[Tuple[str, str]],
             block: int, min_vox: int, n_roll: int, max_blocks: int,
             seed: int, tol: int = 2) -> List[Dict[str, Any]]:
    """对单个 crop，逐类别对做块内分析。"""
    out: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    stride = max(1, block // 2)
    for a_cls, b_cls in pairs:
        A = load(crop_dir, a_cls)
        B = load(crop_dir, b_cls)
        if A is None or B is None:
            continue
        z, y, x = A.shape
        starts = [(i, j, k)
                  for i in range(0, max(1, z - block + 1), stride)
                  for j in range(0, max(1, y - block + 1), stride)
                  for k in range(0, max(1, x - block + 1), stride)]
        used = 0
        rows = []
        for (i, j, k) in starts:
            if used >= max_blocks:
                break
            sa = A[i:i+block, j:j+block, k:k+block]
            sb = B[i:i+block, j:j+block, k:k+block]
            if sa.shape != (block, block, block):
                continue
            if sa.sum() < min_vox or sb.sum() < min_vox:
                continue
            r = analyse_block(sa, sb, n_roll, rng, tol)
            if r is None:
                continue
            r.update({"crop": os.path.basename(crop_dir.rstrip("/")),
                      "pair": f"{a_cls}:{b_cls}",
                      "block": block, "min_vox": min_vox, "tol": tol,
                      "origin": [i, j, k]})
            rows.append(r)
            used += 1
        out.extend(rows)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/oo_labels")
    ap.add_argument("--pairs", default="golgi:er,er:mito,golgi:ves,endo:er,er:lyso,golgi:endo")
    ap.add_argument("--blocks", default="64,96,128")
    ap.add_argument("--min-voxes", default="100,500")
    ap.add_argument("--n-roll", type=int, default=200)
    ap.add_argument("--max-blocks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tols", default="4,6",
                    help="容差（体素）候选，逗号分隔。必须 >=4：语义标注在两类之间留了"
                         " >=8 nm 空隙，tol=2(8nm) 时几乎所有块 obs=0")
    ap.add_argument("--tol", type=int, default=4,
                    help="单个容差（--tols 未给时用）")
    ap.add_argument("--out", default="logs/contact_local.json")
    args = ap.parse_args(argv)

    pairs = []
    for spec in args.pairs.split(","):
        if ":" not in spec:
            continue
        a, b = spec.split(":", 1)
        pairs.append((a.strip(), b.strip()))
    blocks = [int(x) for x in args.blocks.split(",") if x.strip()]
    minvox = [int(x) for x in args.min_voxes.split(",") if x.strip()]

    root = os.path.join(ROOT, args.labels)
    if not os.path.isdir(root):
        print(f"[错误] 缺 {args.labels}")
        return 2
    crops = []
    for ds in sorted(os.listdir(root)):
        dp = os.path.join(root, ds)
        if not os.path.isdir(dp):
            continue
        for c in sorted(os.listdir(dp)):
            cp = os.path.join(dp, c)
            if os.path.isdir(cp):
                crops.append(cp)
    if not crops:
        print(f"[错误] {args.labels} 下没有 crop")
        return 2
    print(f"{len(crops)} 个 crop，{len(pairs)} 个类别对，"
          f"块 {blocks}，最小体素 {minvox}，每块 {args.n_roll} 次平移")

    results: List[Dict[str, Any]] = []
    tols = [int(x) for x in args.tols.split(",") if x.strip()] or [args.tol]
    for blk in blocks:
        for mv in minvox:
            for tol in tols:
                t0 = time.time()
                rows: List[Dict[str, Any]] = []
                for cp in crops:
                    rows += run_pair(cp, pairs, blk, mv,
                                     args.n_roll, args.max_blocks, args.seed,
                                     tol)
                nz = sum(1 for r in rows if r["obs"] > 0)
                print(f"  块={blk} 最小体素={mv} 容差={tol}({tol*4}nm): "
                      f"{len(rows)} 个块，其中 obs>0 {nz} 个 "
                      f"({time.time()-t0:.0f}s)")
                results.extend(rows)

    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)) or ".", exist_ok=True)
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)

    # 即时汇总
    print()
    print("=" * 78)
    print("局部富集比（块内环面零模型）")
    print("=" * 78)
    agg = {}
    for r in results:
        k = (r["block"], r["min_vox"], r["pair"])
        agg.setdefault(k, []).append(r)
    print(f"{'块':>4} {'最小素':>6} {'类别对':<14}{'块数':>5}"
          f"{'富集中位':>10}{'p<0.05且富集>1':>16}")
    print("-" * 78)
    for (blk, mv, pr), rs in sorted(agg.items()):
        enr = [x["enrichment"] for x in rs if x["enrichment"] is not None]
        sig = sum(1 for x in rs
                  if x["p_value"] < 0.05 and (x["enrichment"] or 0) > 1)
        m = st.median(enr) if enr else float("nan")
        print(f"{blk:>4} {mv:>6} {pr:<14}{len(rs):>5}{m:>10.2f}"
              f"{sig:>10}/{len(rs):<5}")
    print()
    print(f"结果 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
