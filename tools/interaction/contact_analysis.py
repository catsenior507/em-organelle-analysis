#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contact_analysis.py —— 从 2D 掩码计算细胞器**接触位点**统计（MITO–ER 等）

## 这在项目里的位置

项目标题是「细胞器互作网络重建」。此前的工作里，`tools/network/` 用**质心距离**
（Delaunay / kNN / radius）近似"互作"——那只是空间邻近，不是互作。

本脚本做的是更接近生物学定义的事：**两个细胞器的边界在多少像素上直接相邻**
（membrane contact site）。接触长度与接触比例是真实文献里常用的量。

## 指标定义（每个都给出单位与分母）

对一对类别 (A, B)：

- ``contact_length_px``：A 的边界像素中，其 8-邻域含 B 边界像素的像素数。
  单位是「像素」，物理长度需乘 µm/px。
- ``contact_per_A_boundary``：``contact_length_px / A边界像素数``，
  无量纲，表示"A 的轮廓有多大比例在接触 B"。取值 [0,1]。
- ``a_pixels`` / ``b_pixels``：各类别的像素面积。
- ``a_components`` / ``b_components``：各类别的 8-连通分量数。
- ``contact_tol_frac``：A 的边界中，距离 B 不超过 ``rim_px`` 像素的比例。
  严格相邻会因标注间隙/抗锯齿漏计，容差版本刻画"贴得很近"，rim_px 显式可复现。

## 零模型对照（关键）

随机把 B 的掩码平移到别处再算一遍，得到"若两者只是偶然共处一图会得到什么"。
观测值必须显著高于零模型，才谈得上结构性的接触偏好。
否则"接触比例 0.3"可能只是"两者都很多"的必然结果。

## 用法

    # 单个导出目录（zarr_to_2d.py 的输出）
    python tools/interaction/contact_analysis.py \\
        --slices E:\\dachuang_public_data\\cellmap_2d \\
        --pairs mito:er \\
        --out data/contact_analysis

    # 也支持直接给 (A 掩码, B 掩码) 的目录对，见 --layout
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover
    ndi = None


# --------------------------------------------------------------------------
# 核心测量
# --------------------------------------------------------------------------

def boundary(mask: np.ndarray) -> np.ndarray:
    """8-连通意义下的边界像素（前景且至少有一个 8-邻域是背景或越界）。"""
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m)
    er = ndi.binary_erosion(m, structure=np.ones((3, 3), bool), border_value=0)
    return m & ~er


def dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    if r <= 0:
        return mask.astype(bool)
    return ndi.binary_dilation(mask.astype(bool),
                               structure=np.ones((3, 3), bool), iterations=r)


def contact_stats(mask_a: np.ndarray, mask_b: np.ndarray,
                  rim_px: int = 2) -> Dict[str, Any]:
    """计算 A 与 B 的接触统计。

    ``mask_a`` / ``mask_b`` 必须是同一坐标系的布尔/整数掩码。
    标签 ID 不参与计算，只看非零。
    """
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    ba = boundary(a)
    bb = boundary(b)

    a_px = int(a.sum())
    b_px = int(b.sum())
    ba_px = int(ba.sum())
    bb_px = int(bb.sum())

    # A 的边界中，邻域里有 B 边界的像素
    if ba_px and bb_px:
        touch = ba & dilate(bb, 1)
        contact = int(touch.sum())
    else:
        contact = 0

    # 容差接触：A 的边界中，距离 B 不超过 rim_px 的比例。
    # 与"严格相邻"互补——严格相邻会因标注间隙、抗锯齿而漏计；
    # 容差版本能回答"两者是否贴得很近"，且 rim_px 是显式参数，可复现。
    #
    # 注意：早期版本试图用一个"向内 rim_px 的环"来度量，结果对相邻方块算出 0——
    # 因为环在 A 边界内 2 像素处，而 B 在外面，两者天然差着 rim_px。
    # 那个定义是错的，已改为直接在边界上做容差判定。
    if ba_px and b_px:
        tol_touch = ba & dilate(b, max(1, rim_px))
        contact_tol = int(tol_touch.sum())
    else:
        contact_tol = 0

    la, na = ndi.label(a, structure=np.ones((3, 3), int))
    lb, nb = ndi.label(b, structure=np.ones((3, 3), int))

    return {
        "a_pixels": a_px,
        "b_pixels": b_px,
        "a_boundary_px": ba_px,
        "b_boundary_px": bb_px,
        "a_components": int(na),
        "b_components": int(nb),
        "contact_length_px": contact,
        "contact_per_a_boundary": (contact / ba_px) if ba_px else None,
        "contact_per_b_boundary": (contact / bb_px) if bb_px else None,
        # 容差接触（rim_px 内）
        "contact_tolerance_px": max(1, rim_px),
        "contact_tol_px": contact_tol,
        "contact_tol_frac": (contact_tol / ba_px) if ba_px else None,
        # 若把 A 的边界全算进来，B 只占 A 附近的比例——便于与零模型比
        "a_boundary_near_b_frac": (contact / ba_px) if ba_px else None,
    }


def _place_inside(mask: np.ndarray, h: int, w: int,
                  dy: int, dx: int) -> np.ndarray:
    """把 mask 的**前景**整体平移到 (dy, dx)，不环绕、不裁剪。

    只平移前景像素（保持其相对形状），越界部分直接丢弃前先确保调用方已选好
    合法的 dy/dx —— 这里假定调用方保证能放下。
    """
    out = np.zeros((h, w), dtype=bool)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return out
    ny, nx = ys + dy, xs + dx
    ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
    out[ny[ok], nx[ok]] = True
    return out


def null_model(mask_a: np.ndarray, mask_b: np.ndarray, *,
               n_shifts: int = 20, seed: int = 0,
               rim_px: int = 2) -> Dict[str, Any]:
    """把 B **整体随机搬到图内另一处**后重算，得到接触比例的零分布。

    为什么不用 ``np.roll``：roll 是**环绕**平移，B 会被复制到图的对侧，
    于是"刚贴完 A 的那部分"又从另一边贴回来，零分布出现不合理的高值
    （实测：相邻方块的观测值 0.294，零模型最大值竟也能追上）。
    改用"只在图内合法位置放置、不环绕、不裁剪"，B 的像素数与形状严格不变。

    判据用 **p 值**（零分布中 ≥ 观测的比例）比"是否超过最大值"更稳健：
    样本有限时最大值本身波动很大。
    """
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    h, w = b.shape
    ys, xs = np.nonzero(b)
    if ys.size == 0 or not a.any():
        return {"observed": None, "null_n": 0, "p_value": None}

    b_h = int(ys.max() - ys.min() + 1)
    b_w = int(xs.max() - xs.min() + 1)
    y_lo, x_lo = -int(ys.min()), -int(xs.min())
    y_hi, x_hi = h - 1 - int(ys.max()), w - 1 - int(xs.max())
    if y_hi < y_lo or x_hi < x_lo:
        return {"observed": None, "null_n": 0, "p_value": None}

    rng = np.random.RandomState(seed)
    obs = contact_stats(a, b, rim_px=rim_px)["contact_per_a_boundary"]

    vals: List[float] = []
    for _ in range(n_shifts):
        dy = int(rng.randint(y_lo, y_hi + 1))
        dx = int(rng.randint(x_lo, x_hi + 1))
        moved = _place_inside(b, h, w, dy, dx)
        st = contact_stats(a, moved, rim_px=rim_px)
        v = st["contact_per_a_boundary"]
        if v is not None:
            vals.append(float(v))

    if obs is None or not vals:
        return {"observed": obs, "null_n": len(vals), "p_value": None}

    o = float(obs)
    ge = sum(1 for v in vals if v >= o)
    p = (ge + 1) / (len(vals) + 1)          # 加一修正，避免 p=0
    med = float(statistics.median(vals))
    sd = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return {
        "observed": o,
        "null_n": len(vals),
        "null_median": med,
        "null_max": float(max(vals)),
        "null_sd": sd,
        "z": ((o - med) / sd) if sd > 0 else None,
        "p_value": round(p, 4),
        "significant_p05": p < 0.05,
        "b_bbox": [b_h, b_w],
        "allowed_shift": [y_lo, y_hi, x_lo, x_hi],
    }


# --------------------------------------------------------------------------
# 数据发现
# --------------------------------------------------------------------------

def find_slice_dirs(root: str) -> List[Dict[str, Any]]:
    """定位 zarr_to_2d.py 的输出结构：

        <root>/<dataset>/crop<ID>/z####/{em.png, <label>.png, ...}
    """
    out: List[Dict[str, Any]] = []
    for zdir in sorted(glob.glob(os.path.join(root, "*", "*", "z*"))):
        if not os.path.isdir(zdir):
            continue
        parts = zdir.replace("\\", "/").split("/")
        zname = parts[-1]
        if not (zname.startswith("z") and zname[1:].isdigit()):
            continue
        labels = {}
        for png in glob.glob(os.path.join(zdir, "*.png")):
            labels[os.path.splitext(os.path.basename(png))[0]] = png
        if not labels:
            continue
        out.append({
            "dir": zdir,
            "dataset": parts[-3],
            "crop": parts[-2],
            "z": int(zname[1:]),
            "labels": labels,
        })
    return out


def load_mask(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path))


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="从 2D 掩码计算细胞器接触位点统计")
    ap.add_argument("--slices", required=True,
                    help="zarr_to_2d.py 的输出根目录")
    ap.add_argument("--pairs", default="mito:er",
                    help="类别对，逗号分隔的 A:B（默认 mito:er）")
    ap.add_argument("--out", default="data/contact_analysis")
    ap.add_argument("--rim-px", type=int, default=2)
    ap.add_argument("--null-shifts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-pixels", type=int, default=50,
                    help="任一类像素少于此值则跳过该切片")
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy（pip install scipy）", file=sys.stderr)
        return 2

    pairs = []
    for spec in args.pairs.split(","):
        if ":" not in spec:
            print(f"[错误] 类别对格式应为 A:B，得到 {spec!r}", file=sys.stderr)
            return 2
        a, b = spec.split(":", 1)
        pairs.append((a.strip(), b.strip()))

    slices = find_slice_dirs(args.slices)
    if not slices:
        print(f"[错误] {args.slices} 下没找到 <dataset>/crop*/z*/*.png 结构",
              file=sys.stderr)
        return 2
    if args.limit:
        slices = slices[: args.limit]
    print(f"找到 {len(slices)} 个切片目录")

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    t0 = time.time()

    for i, s in enumerate(slices, 1):
        labels = s["labels"]
        for a_name, b_name in pairs:
            if a_name not in labels or b_name not in labels:
                skipped.append({
                    "dataset": s["dataset"], "crop": s["crop"], "z": s["z"],
                    "reason": f"缺少 {a_name} 或 {b_name}",
                })
                continue
            ma = load_mask(labels[a_name])
            mb = load_mask(labels[b_name])
            if ma.shape != mb.shape:
                skipped.append({"dataset": s["dataset"], "crop": s["crop"],
                                "z": s["z"],
                                "reason": f"形状不符 {ma.shape} vs {mb.shape}"})
                continue
            if int((ma > 0).sum()) < args.min_pixels or \
               int((mb > 0).sum()) < args.min_pixels:
                skipped.append({"dataset": s["dataset"], "crop": s["crop"],
                                "z": s["z"],
                                "reason": "像素数低于 --min-pixels"})
                continue

            st = contact_stats(ma, mb, rim_px=args.rim_px)
            null = null_model(ma, mb, n_shifts=args.null_shifts,
                              seed=args.seed, rim_px=args.rim_px)
            rows.append({
                "dataset": s["dataset"], "crop": s["crop"], "z": s["z"],
                "pair": f"{a_name}:{b_name}",
                **st,
                "null": null,
            })
        if i % 20 == 0:
            print(f"  ...{i}/{len(slices)}", flush=True)

    if not rows:
        print("[错误] 没有任何切片同时含所需类别", file=sys.stderr)
        return 2

    # ---- 汇总 ----
    def col(k: str) -> List[float]:
        return [r[k] for r in rows if r.get(k) is not None]

    def summ(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "median": round(float(statistics.median(vals)), 4),
            "mean": round(float(statistics.fmean(vals)), 4),
            "min": round(float(min(vals)), 4),
            "max": round(float(max(vals)), 4),
        }

    zs = [r["null"]["z"] for r in rows if r["null"].get("z") is not None]
    sig = sum(1 for r in rows if r["null"].get("significant_p05"))

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "tools/interaction/contact_analysis.py",
        "slices_dir": os.path.abspath(args.slices),
        "pairs": [f"{a}:{b}" for a, b in pairs],
        "rim_px": args.rim_px,
        "null_shifts": args.null_shifts,
        "num_slices_scanned": len(slices),
        "num_measurements": len(rows),
        "num_skipped": len(skipped),
        "metrics": {
            "contact_length_px": summ(col("contact_length_px")),
            "contact_per_a_boundary": summ(col("contact_per_a_boundary")),
            "contact_per_b_boundary": summ(col("contact_per_b_boundary")),
            "contact_tol_frac": summ(col("contact_tol_frac")),
            "a_pixels": summ(col("a_pixels")),
            "b_pixels": summ(col("b_pixels")),
            "a_components": summ(col("a_components")),
            "b_components": summ(col("b_components")),
        },
        "null_model": {
            "num_with_z": len(zs),
            "z_median": round(float(statistics.median(zs)), 3) if zs else None,
            "frac_significant_p05": round(sig / len(rows), 3),
            "interpretation": (
                "contact_per_a_boundary 若与零模型无差别（|z| 小、"
                "frac_significant_p05 低），说明观测到的接触只是"
                "『两者都很多』的必然结果，不构成结构性的接触偏好。"
            ),
        },
        "unit_note": (
            "contact_length_px 的单位是像素；要换算成物理长度需乘 µm/px。"
            "本项目图像标定有三个不同取值，跨图比较绝对长度无效。"
        ),
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "per_slice.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "skipped": skipped[:200]}, fh,
                  ensure_ascii=False, indent=1)

    # CSV
    import csv
    keys = ["dataset", "crop", "z", "a_pixels", "b_pixels",
            "a_components", "b_components", "contact_length_px",
            "contact_per_a_boundary", "contact_tol_frac"]
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + ["null_z", "p_value",
                                                  "significant_p05"])
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in keys}
            row["null_z"] = r["null"].get("z")
            row["p_value"] = r["null"].get("p_value")
            row["significant_p05"] = r["null"].get("significant_p05")
            w.writerow(row)

    print("\n" + "=" * 66)
    print(f"测量 {summary['num_measurements']} 次，跳过 {summary['num_skipped']} 次，"
          f"耗时 {time.time() - t0:.1f}s")
    m = summary["metrics"]
    for k, label in [("contact_per_a_boundary", "A边界接触比例"),
                     ("contact_tol_frac", "容差接触比例"),
                     ("contact_length_px", "接触像素数")]:
        s = m[k]
        if s.get("n"):
            print(f"  {label:<14} n={s['n']:4d}  "
                  f"中位={s['median']:<10} 均值={s['mean']}")
    nm = summary["null_model"]
    print(f"  零模型：z 中位={nm['z_median']}  "
          f"达 p<0.05 的比例={nm['frac_significant_p05']}")
    print(f"\n输出：{os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
