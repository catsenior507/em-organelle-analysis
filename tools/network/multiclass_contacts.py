#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multiclass_contacts.py —— 多类细胞器的互作分析（3D 接触统计）

## 这是项目的核心目标

课题是"细胞器互作网络重建与涌现特征分析"。此前我们**只有单类标注**
（自有数据 `organelle_type` 全是 "M"），只能做"同类相邻"网络，
做不了真正的**异类互作**——因为同一视野里没有两类细胞器的真值。

ASEM 数据集 64 提供了这个条件：`labels/er` 与 `labels/mito` 落在**同一 ROI**
（241×476×528，5 nm 各向同性）。

> 这是**外部数据**（BSC-1 细胞、化学固定），验证的是方法与工具链，
> 不是我们样品的生物学结论。

## 关键：接触阈值必须按生物学定义来定

第一次跑用"膨胀 2 体素 = 10 nm"，得到 **0 个接触**，看起来像是数据有问题。
诊断后发现真相：

* 两类标签是**互斥**的（同一体素只属于一类），重叠体素为 0
* 膨胀 1/2/3 体素（5/10/15 nm）-> 接触**全为 0**
* 膨胀 5 体素（25 nm）-> 接触 15058 体素

也就是说两类的最近距离在 15–25 nm 之间。而文献里
**内质网–线粒体接触位点（MCS）的经典定义就是两膜间距 10–30 nm**。
用 10 nm 正好落在窗口下沿之外，所以判为 0——**是阈值设错，不是数据错**。

因此本脚本：
1. **默认按 25 nm 判定**（窗口内），并在输出里写明依据
2. 提供 `--sweep`，把 5–40 nm 的接触量都列出来，让阈值选择**透明可查**
   而不是一个看起来任意的数字
3. 报告"接触表面积占 mito 表面积的比例"，而不只是体素数——
   后者会随膨胀量单调增长，不利于解释

## 标签是语义而非实例

两类标签的取值都是 {0,1}，即**二值语义标注**，不是实例 ID。
因此"实例"只能靠连通域近似：mito 在 241×476×528 里是 3 个大连通域
（线粒体本就是网络状），er 是 15 个。报告里会明确这一点，
避免把"连通域"误解为"单个细胞器"。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACE6 = ndi.generate_binary_structure(3, 1) if ndi is not None else None


def surface(binary: np.ndarray) -> np.ndarray:
    """表面体素 = 自身为前景但 6 邻域里有背景的体素。"""
    er = ndi.binary_erosion(binary, structure=FACE6, border_value=0)
    return binary & ~er


def contact_at(mask_a: np.ndarray, mask_b: np.ndarray, dilate: int) -> dict:
    """在给定膨胀量下统计 A 与 B 的接触。"""
    grown = ndi.binary_dilation(mask_a, structure=FACE6, iterations=dilate) \
        if dilate > 0 else mask_a
    touch = grown & mask_b
    surf_a = surface(mask_a)
    contact = surf_a & ndi.binary_dilation(mask_b, structure=FACE6,
                                           iterations=dilate) \
        if dilate > 0 else surf_a & mask_b
    return {
        "dilate_vox": dilate,
        "dist_nm": dilate * VOXEL_NM_PLACEHOLDER,
        "overlap_voxels": int(touch.sum()),
        "surface_voxels": int(surf_a.sum()),
        "contact_surface_voxels": int(contact.sum()),
        "contact_surface_frac": (float(contact.sum()) / surf_a.sum()
                                 if surf_a.sum() else None),
    }


VOXEL_NM_PLACEHOLDER = 5.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mito", default="data/asem/ds64/mito.npy")
    ap.add_argument("--er", default="data/asem/ds64/er.npy")
    ap.add_argument("--voxel-nm", type=float, default=5.0)
    ap.add_argument("--min-vox", type=int, default=200)
    ap.add_argument("--dilate", type=int, default=5,
                    help="接触判定膨胀体素数；默认 5 体素 = 25 nm，"
                         "落在 MCS 的 10–30 nm 窗口内")
    ap.add_argument("--sweep", action="store_true", default=True,
                    help="同时输出 5–40 nm 的接触量扫描")
    ap.add_argument("--out", default="data/network_multiclass/ds64.json")
    args = ap.parse_args(argv)

    global VOXEL_NM_PLACEHOLDER
    VOXEL_NM_PLACEHOLDER = args.voxel_nm

    if ndi is None:
        print("[错误] 需要 scipy")
        return 2
    mp = os.path.join(ROOT, args.mito)
    ep = os.path.join(ROOT, args.er)
    for p in (mp, ep):
        if not os.path.exists(p):
            print(f"[错误] 缺 {p}")
            return 2
    M = np.load(mp) > 0
    E = np.load(ep) > 0
    if M.shape != E.shape:
        print(f"[错误] 形状不一致 {M.shape} vs {E.shape}")
        return 2
    print(f"体积 {M.shape}  体素 {args.voxel_nm} nm")
    print(f"  mito 前景 {100*M.mean():.2f}%   er 前景 {100*E.mean():.2f}%")
    print(f"  两类重叠体素 {int((M & E).sum())}（互斥标注则为 0）")
    print()

    # 实例（= 连通域，非真实实例 ID）
    ml, nm = ndi.label(M, structure=FACE6)
    el, ne = ndi.label(E, structure=FACE6)
    ms = np.bincount(ml.ravel())[1:]
    es = np.bincount(el.ravel())[1:]
    print(f"连通域：mito {nm} 个（最大 {ms.max()/1e6:.2f}M 体素）；"
          f"er {ne} 个（最大 {es.max()/1e6:.2f}M 体素）")
    print("  ！标签是**二值语义**，连通域不等于单个细胞器"
          "（线粒体本为网络状）")
    print()

    # 距离扫描
    print("接触量随判定距离的变化（阈值选择要可见，不能是个黑盒数字）：")
    print(f"{'膨胀':>5} {'距离':>8} {'重叠体素':>10} "
          f"{'mito表面接触体素':>17} {'占mito表面':>11}")
    print("-" * 60)
    sweep = []
    for d in (0, 1, 2, 3, 4, 5, 6, 8):
        r = contact_at(M, E, d)
        sweep.append(r)
        print(f"{d:>5} {r['dist_nm']:>6.0f}nm {r['overlap_voxels']:>10} "
              f"{r['contact_surface_voxels']:>17} "
              f"{100*(r['contact_surface_frac'] or 0):>10.2f}%")
    print()

    chosen = contact_at(M, E, args.dilate)
    vum = args.voxel_nm / 1000.0
    print("=" * 68)
    print("线粒体 – 内质网 接触统计（ASEM ds64，BSC-1 细胞，化学固定）")
    print(f"判定距离 {chosen['dist_nm']:.0f} nm（膨胀 {args.dilate} 体素）")
    print(" 依据：MCS 经典定义是两膜间距 10–30 nm，故取窗口内的 25 nm")
    print("=" * 68)
    print(f"  接触重叠体素        {chosen['overlap_voxels']}")
    print(f"  接触体积            {chosen['overlap_voxels']*vum**3:.4f} µm³")
    print(f"  mito 表面体素       {chosen['surface_voxels']}")
    print(f"  接触处 mito 表面    {chosen['contact_surface_voxels']} "
          f"({100*(chosen['contact_surface_frac'] or 0):.2f}%)")
    print(f"  mito 表面积（估）   {chosen['surface_voxels']*vum**2:.4f} µm²")
    print(f"  接触面积（估）      "
          f"{chosen['contact_surface_voxels']*vum**2:.4f} µm²")

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "dataset": "ASEM ds64 (BSC-1, chemical fixation)",
            "voxel_nm": args.voxel_nm,
            "shape": list(M.shape),
            "mito_frac": float(M.mean()), "er_frac": float(E.mean()),
            "overlap_voxels": int((M & E).sum()),
            "components": {"mito": int(nm), "er": int(ne)},
            "note_labels_are_semantic": True,
            "chosen": chosen,
            "distance_sweep": sweep,
            "note": "外部数据，验证方法与工具链，非本样品生物学结论",
        }, fh, ensure_ascii=False, indent=1)
    print(f"\n已写出 {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
