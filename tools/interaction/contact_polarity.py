#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contact_polarity.py —— 膜接触位点是集中在特定区域，还是均匀铺在细胞器表面？

## 为什么问这个

`CONTACT-MULTICLASS-OO.md` §9 留了一条未做的事。此前的分析只回答了
"两类膜靠得有多近、比随机多还是少"，没有回答**接触在细胞器表面是怎么分布的**。

这个问题不是细节。膜接触位点（MCS）在细胞生物学里**不是**随机撒在细胞器表面的：
内质网—线粒体接触富集在线粒体与内质网管状网络的交汇处，且由栓系蛋白复合物定位。
若测量只能给出"接触面积占总表面的百分之几"，它无法区分两种完全不同的生物学：

  (a) 整个表面均匀地偶尔贴近 —— 更像随机擦碰；
  (b) 少数几个区域持续贴近 —— 更像特化的接触位点。

## 用什么统计量

对每个（体数据，类别对 A:B，容差 t）：

1. 取 A 的**表面体素**（6 邻域内有非 A 体素的 A 体素）；
2. 接触体素 = 表面上到 B 的距离 ≤ t 的那些；
3. 把接触体素做 6 连通标记，得到若干**接触斑块**；
4. 统计量 **最大斑块占比** = 最大斑块体素数 / 接触体素总数。

判读：若接触集中在特定位点，斑块应当**少而大**，最大占比高；
若均匀铺开，斑块应当**多而碎**，最大占比低。

## 零模型

在 A 的表面上**均匀随机**抽同样多的体素当作接触，重算最大斑块占比，
重复 N 次得到零分布。这个零模型保持了两件事不变：总接触量、表面本身的形状与
相邻关系——所以"表面是连通的"这一因素已被它吸收，剩下的差异只能来自接触的
空间聚集。p 值单侧，小表示接触比随机更聚集。

## 性能：为什么要在包围盒里算

首版直接在全尺寸体数据（400³–512³）上做距离变换并逐次抽样建数组。
512³ 的 float64 距离变换就是 1 GB，200 次抽样每次再分配一个 134 MB 的布尔数组，
单个类别对要跑几十分钟，做不完全部组合。

改为：先取 A 的表面坐标，只在其包围盒（外扩容差 + 2 体素）内裁剪 A 与 B，
之后所有运算都在这个远小的窗口里做。距离变换在窗口上做是**精确的**，
因为窗口已经覆盖了所有可能落在容差内的点。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

CLASSES = ("golgi", "er", "endo", "ves", "lyso")
PAIRS = [(a, b) for i, a in enumerate(CLASSES) for b in CLASSES[i + 1:]]
S6 = ndi.generate_binary_structure(3, 1)


def load_mask(root: str, ds: str, crop: str, cls: str):
    p = os.path.join(root, ds, crop, f"{cls}.npy")
    if not os.path.exists(p):
        return None
    m = np.load(p)
    return m if m.dtype == bool else (m > 0)


def surface_of(mask: np.ndarray) -> np.ndarray:
    """6 邻域意义下的表面：自身为前景、且至少有一个邻居是背景。"""
    er = ndi.binary_erosion(mask, structure=S6, border_value=0)
    return mask & ~er


def largest_share(contact: np.ndarray) -> tuple:
    """接触斑块数与最大斑块占比。"""
    tot = int(contact.sum())
    if tot == 0:
        return 0, 0.0, 0
    lab, n = ndi.label(contact, structure=S6)
    if n == 0:
        return 0, 0.0, 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return int(n), float(sizes.max() / tot), tot


def analyse_pair(A: np.ndarray, B: np.ndarray, tol_vox: float,
                 n_null: int, rng: np.random.Generator) -> dict:
    surf = surface_of(A)
    coords = np.argwhere(surf)
    if coords.size == 0 or not B.any():
        return {}

    # ---- 只保留表面包围盒（外扩容差 + 2）----
    pad = int(np.ceil(tol_vox)) + 2
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, np.array(A.shape))
    sl = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))
    Al, Bl = A[sl], B[sl]
    surfl = surf[sl]

    n_surf = int(surfl.sum())
    if n_surf == 0:
        return {}

    # 窗口内距离变换是精确的：窗口已覆盖所有可能落在容差内的点
    dist = ndi.distance_transform_edt(~Bl)
    contact = surfl & (dist <= tol_vox)
    n_patch, share, tot = largest_share(contact)
    if tot == 0:
        return {"n_surface": n_surf, "n_contact": 0}

    idx = np.flatnonzero(surfl.ravel())

    # ---- 自适应抽样次数 ----
    #
    # 每次零模型抽样都要在窗口上做一次 3D 连通标记。512³ 的体数据里 ER 表面
    # 可达数百万体素，100 次标记单组就要几十分钟（实测卡死过一次）。
    # 这里按表面体素数收缩抽样次数：小表面照常用满，大表面降到下限 20。
    # 代价是 p 值分辨率变粗（下限从 1/101 变成 1/21），但观测与零模型的
    # 差距中位有 51 倍，不依赖高分辨率 p。
    scale = min(1.0, 1_000_000.0 / max(1, n_surf))
    eff_null = max(20, int(n_null * scale))

    null_shares = np.empty(eff_null, dtype=np.float64)
    buf = np.zeros(surfl.size, dtype=bool)
    for k in range(eff_null):
        buf[:] = False
        buf[rng.choice(idx, size=tot, replace=False)] = True
        _, s, _ = largest_share(buf.reshape(surfl.shape))
        null_shares[k] = s
    p = float((null_shares >= share).sum() + 1) / (eff_null + 1)

    return {
        "n_surface": n_surf,
        "n_contact": tot,
        "contact_frac": tot / n_surf,
        "n_patch": n_patch,
        "largest_share": share,
        "null_share_median": float(np.median(null_shares)),
        "null_share_p95": float(np.percentile(null_shares, 95)),
        "p_aggregated": p,
        "n_null_used": eff_null,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/oo_labels")
    ap.add_argument("--tol-nm", type=float, default=24.0)
    ap.add_argument("--voxel-nm", type=float, default=4.0)
    ap.add_argument("--null", type=int, default=100)
    ap.add_argument("--min-contact", type=int, default=30)
    ap.add_argument("--out", default="logs/CONTACT_POLARITY.json")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    tol_vox = a.tol_nm / a.voxel_nm
    rng = np.random.default_rng(20261113)
    rows = []

    # ---- 断点续跑 ----
    #
    # 首版只在全部跑完后写一次 JSON，导致进程被中断时**已完成的组全部丢失**。
    # 现在改为每完成一组就落盘，并在启动时读回已完成集合、跳过它们。
    done = set()
    if os.path.exists(a.out):
        try:
            for r in json.load(open(a.out, encoding="utf-8")):
                done.add((r["dataset"], r["crop"], r["pair"], r.get("tol_nm")))
            rows = list(json.load(open(a.out, encoding="utf-8")))
            print(f"续跑：已有 {len(rows)} 组，将跳过")
        except Exception:
            rows, done = [], set()

    def flush():
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

    datasets = sorted(d for d in os.listdir(a.root)
                      if os.path.isdir(os.path.join(a.root, d)))
    for ds in datasets:
        crops = sorted(d for d in os.listdir(os.path.join(a.root, ds))
                       if os.path.isdir(os.path.join(a.root, ds, d)))
        for crop in crops:
            for x, y in PAIRS:
                key = (ds, crop, f"{x}:{y}", a.tol_nm)
                if key in done:
                    continue
                A = load_mask(a.root, ds, crop, x)
                B = load_mask(a.root, ds, crop, y)
                if A is None or B is None or not A.any() or not B.any():
                    continue
                r = analyse_pair(A, B, tol_vox, a.null, rng)
                if not r or r.get("n_contact", 0) < a.min_contact:
                    continue
                r.update({"dataset": ds, "crop": crop, "pair": f"{x}:{y}",
                          "tol_nm": a.tol_nm})
                rows.append(r)
                done.add(key)
                flush()
                print(f"  {ds}/{crop} {x}:{y}  接触 {r['n_contact']:>6}"
                      f"  占表面 {100*r['contact_frac']:>5.2f}%"
                      f"  斑块 {r['n_patch']:>4}"
                      f"  最大占比 {r['largest_share']:.3f}"
                      f"  零模型 {r['null_share_median']:.3f}"
                      f"  (n={r['n_null_used']})"
                      f"  p={r['p_aggregated']:.3f}", flush=True)

    flush()
    print(f"\n共 {len(rows)} 组写入 {a.out}")
    if rows:
        ps = [r["p_aggregated"] for r in rows]
        sig = sum(1 for p in ps if p < 0.05)
        shares = [r["largest_share"] for r in rows]
        nulls = [r["null_share_median"] for r in rows]
        print(f"  显著聚集（p<0.05）：{sig}/{len(rows)}")
        print(f"  最大斑块占比 中位 {np.median(shares):.3f}"
              f"  零模型中位 {np.median(nulls):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
