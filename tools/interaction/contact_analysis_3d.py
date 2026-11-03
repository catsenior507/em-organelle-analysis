#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contact_analysis_3d.py —— 在 **3D 标签体数据**上计算细胞器接触位点

## 与 2D 版的区别

`contact_analysis.py` 处理的是 2D 切片（PNG）。本脚本处理 CellMap 的
3D 标签体（按 chunk 从 S3 拉下来的 `.npy`，形状如 300³）。

3D 比 2D 切片更可信：
  * 2D 切片上的"接触"可能只是 3D 结构在某一层的投影，接触长度会被系统性低估；
  * 3D 可以直接算**接触面**（共享面元的体素对数），这是膜接触位点的标准代理量。

## 指标定义

对类别对 (A, B)：

| 指标 | 定义 | 单位 |
|---|---|---|
| `a_voxels` / `b_voxels` | 各类别体素数 | voxel |
| `a_surface_voxels` | A 的表面体素（6-邻域中有非 A 的体素） | voxel |
| `contact_faces` | A 与 B 共享的**面元**数（6-邻接体素对数） | faces |
| `contact_per_a_surface` | `contact_faces / a_surface_voxels` | [0,1] |
| `contact_tol_faces` | A 的表面中，距 B ≤ tol 体素的**面元**数 | faces |
| `a_components` / `b_components` | 3D 连通分量数（6-邻接） | 个 |

物理换算：`contact_faces × voxel²` 得到接触面积（nm² 或 µm²），
`voxel` 取该 crop 的 `voxel_size`（来自官方 manifest）。

## 零模型（同样必需）

把 B 的掩码在 3D 体积内随机平移（不环绕、不裁剪），重算 `contact_per_a_surface`，
得到零分布。只看观测值没有意义——两类都很多时必然接触。

## 用法

    python tools/interaction/contact_analysis_3d.py \
        --labels <fetch_labels 输出目录> \
        --pairs mito:er \
        --out data/contact_3d \
        [--null-shifts 50] [--tol 1]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover
    ndi = None

#: 6-邻接（面接触）。用 6 而不是 26，因为"共享面元"才是膜接触的几何对应物。
FACE6 = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                  [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                  [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=bool)


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """6-邻域意义下的表面体素：前景中至少有一个面邻居是背景。"""
    m = mask.astype(bool)
    er = ndi.binary_erosion(m, structure=FACE6, border_value=0)
    return m & ~er


def dilate6(mask: np.ndarray, it: int) -> np.ndarray:
    if it <= 0:
        return mask.astype(bool)
    return ndi.binary_dilation(mask.astype(bool), structure=FACE6, iterations=it)


def contact_stats3d(a: np.ndarray, b: np.ndarray, tol: int = 1) -> Dict[str, Any]:
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    if a.shape != b.shape:
        raise ValueError(f"形状不一致 {a.shape} vs {b.shape}")

    sa = surface_voxels(a)
    sb = surface_voxels(b)
    a_vox = int(a.sum())
    b_vox = int(b.sum())
    sa_vox = int(sa.sum())
    sb_vox = int(sb.sum())

    # 接触面元：A 的表面体素中，其 6-邻域含 B 的体素
    contact = int((sa & dilate6(b, 1)).sum()) if sa_vox and b_vox else 0
    # 反向（从 B 看 A），用于交叉验证对称性
    contact_b = int((sb & dilate6(a, 1)).sum()) if sb_vox and a_vox else 0
    # 容差接触
    tol_faces = int((sa & dilate6(b, max(1, tol))).sum()) if sa_vox and b_vox else 0

    la, na = ndi.label(a, structure=FACE6)
    lb, nb = ndi.label(b, structure=FACE6)

    return {
        "a_voxels": a_vox,
        "b_voxels": b_vox,
        "a_surface_voxels": sa_vox,
        "b_surface_voxels": sb_vox,
        "contact_faces": contact,
        "contact_faces_from_b": contact_b,
        "contact_per_a_surface": (contact / sa_vox) if sa_vox else None,
        "contact_per_b_surface": (contact_b / sb_vox) if sb_vox else None,
        "contact_tol": tol,
        "contact_tol_faces": tol_faces,
        "contact_tol_per_a_surface": (tol_faces / sa_vox) if sa_vox else None,
        "a_components": int(na),
        "b_components": int(nb),
        "a_frac": round(a_vox / a.size, 5),
        "b_frac": round(b_vox / b.size, 5),
    }


def _shift_no_wrap(mask: np.ndarray, dz: int, dy: int, dx: int) -> np.ndarray:
    """只平移前景，不环绕、不裁剪（调用方保证放得下）。"""
    out = np.zeros_like(mask, dtype=bool)
    zs, ys, xs = np.nonzero(mask)
    if zs.size == 0:
        return out
    nz, ny, nx = zs + dz, ys + dy, xs + dx
    ok = ((nz >= 0) & (nz < mask.shape[0])
          & (ny >= 0) & (ny < mask.shape[1])
          & (nx >= 0) & (nx < mask.shape[2]))
    out[nz[ok], ny[ok], nx[ok]] = True
    return out


def _shift_toroidal(mask: np.ndarray, dz: int, dy: int, dx: int) -> np.ndarray:
    """环面平移（periodic shift）：越界部分从对侧回来。

    这是空间统计里常用的 toroidal shift，严格保持前景的**形状与体积**，
    且对任何形状都总有合法的平移量。
    """
    return np.roll(np.roll(np.roll(mask, dz, axis=0), dy, axis=1), dx, axis=2)


def null_model3d(a: np.ndarray, b: np.ndarray, *, n_shifts: int = 20,
                 seed: int = 0, tol: int = 1,
                 mode: str = "toroidal") -> Dict[str, Any]:
    """把 B 随机平移后重算接触比例，得到零分布。

    **为什么默认用环面平移而不是"图内合法位置重放"**：
    第一版用的是后者，结果在真实数据上退化——`er` 有 33 个连通分量散布全卷，
    其包围盒几乎等于整卷，于是"合法平移范围"坍缩为 0，
    50 次抽样全部等于原始位置，零分布与观测值完全相同（p 恒为 1.0）。
    环面平移对任何形状都总有 n³ 种合法平移，且保持形状与体积，是标准做法。
    """
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    if b.shape != a.shape:
        raise ValueError("形状不一致")
    if not b.any() or not a.any():
        return {"observed": None, "null_n": 0, "p_value": None}

    rng = np.random.RandomState(seed)
    obs = contact_stats3d(a, b, tol=tol)["contact_per_a_surface"]
    vals: List[float] = []

    if mode == "toroidal":
        shp = b.shape
        for _ in range(n_shifts):
            dz = int(rng.randint(0, shp[0]))
            dy = int(rng.randint(0, shp[1]))
            dx = int(rng.randint(0, shp[2]))
            v = contact_stats3d(a, _shift_toroidal(b, dz, dy, dx),
                                tol=tol)["contact_per_a_surface"]
            if v is not None:
                vals.append(float(v))
    else:
        zs, ys, xs = np.nonzero(b)
        lo = (-int(zs.min()), -int(ys.min()), -int(xs.min()))
        hi = (b.shape[0] - 1 - int(zs.max()),
              b.shape[1] - 1 - int(ys.max()),
              b.shape[2] - 1 - int(xs.max()))
        if any(h < l for h, l in zip(hi, lo)):
            return {"observed": None, "null_n": 0, "p_value": None}
        for _ in range(n_shifts):
            dz = int(rng.randint(lo[0], hi[0] + 1))
            dy = int(rng.randint(lo[1], hi[1] + 1))
            dx = int(rng.randint(lo[2], hi[2] + 1))
            v = contact_stats3d(a, _shift_no_wrap(b, dz, dy, dx),
                                tol=tol)["contact_per_a_surface"]
            if v is not None:
                vals.append(float(v))

    if obs is None or not vals:
        return {"observed": obs, "null_n": len(vals), "p_value": None}

    o = float(obs)
    ge = sum(1 for v in vals if v >= o)
    med = float(statistics.median(vals))
    sd = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return {
        "observed": o,
        "null_n": len(vals),
        "null_median": med,
        "null_max": float(max(vals)),
        "null_min": float(min(vals)),
        "null_sd": sd,
        "p_value": round((ge + 1) / (len(vals) + 1), 4),
        "significant_p05": ((ge + 1) / (len(vals) + 1)) < 0.05,
        "z": ((o - med) / sd) if sd > 0 else None,
        "mode": mode,
        # 退化检测：若零分布几乎没有变化，说明这个零模型对该结构无分辨力
        "degenerate_null": bool(sd < 1e-12) or bool(len(set(vals)) == 1),
    }


def tolerance_profile(a: np.ndarray, b: np.ndarray,
                      tols: Tuple[int, ...] = (1, 2, 4, 8)) -> Dict[str, Any]:
    """A 的表面在不同容差下"够得着" B 的比例。

    为什么需要：严格相邻（tol=1）可能因为标注之间的 1–2 体素缝隙而给出 0，
    这并不等于"两者互不接触"。给出多容差剖面可以区分：
      * tol=1 就很高        -> 真的贴在一起
      * 仅在 tol≥4 才升高   -> 中间隔着几个体素，可能是真实间隙或标注偏差
      * 所有 tol 都是 0     -> 在该体积内两者相距很远
    """
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    sa = surface_voxels(a)
    sa_vox = int(sa.sum())
    out: Dict[str, Any] = {"a_surface_voxels": sa_vox, "b_voxels": int(b.sum())}
    if sa_vox == 0 or not b.any():
        for t in tols:
            out[f"faces_tol{t}"] = 0
            out[f"frac_tol{t}"] = None
        return out
    for t in tols:
        c = int((sa & dilate6(b, t)).sum())
        out[f"faces_tol{t}"] = c
        out[f"frac_tol{t}"] = round(c / sa_vox, 6)
    return out


def tolerance_profile_nm(a: np.ndarray, b: np.ndarray, voxel_nm: Sequence[float],
                         tols_nm: Sequence[float] = (0, 16, 32, 64)
                         ) -> Dict[str, Any]:
    """以**纳米**为单位给出容差剖面，使不同分辨率的 crop 可比。

    为什么必须这样做：容差用体素数表示时，8 nm/体素的数据里 tol=8 是 64 nm，
    而 4 nm/体素的数据里 tol=8 只有 32 nm——两者根本不在同一个物理尺度上，
    直接比较会得出错误结论。改为按 nm 指定容差，再换算成体素数迭代次数。

    生物学背景：内质网–线粒体接触位点（MAMs）的间隙约 10–30 nm，
    因此 tol=0/16/32/64 nm 这个序列正好覆盖"紧密接触→仅邻近"的范围。
    """
    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    sa = surface_voxels(a)
    sa_vox = int(sa.sum())
    vmean = float(statistics.fmean(voxel_nm)) if voxel_nm else 1.0
    out: Dict[str, Any] = {
        "voxel_nm": list(voxel_nm),
        "voxel_nm_mean": round(vmean, 3),
        "a_surface_voxels": sa_vox,
        "b_voxels": int(b.sum()),
        "profile": [],
    }
    if sa_vox == 0 or not b.any():
        for t in tols_nm:
            out["profile"].append({"tol_nm": t, "iters_voxels": None,
                                   "faces": 0, "frac": None})
        return out

    # 逐轴换算，取最大的迭代次数（保守），因为体素可能各向异性
    for t in tols_nm:
        iters = max(1, int(round(float(t) / max(vmean, 1e-9)))) if t > 0 else 1
        c = int((sa & dilate6(b, iters)).sum())
        out["profile"].append({
            "tol_nm": t,
            "iters_voxels": iters,
            "faces": c,
            "frac": round(c / sa_vox, 6),
        })
    return out


def contact_null_fft(a: np.ndarray, b: np.ndarray, tol: int = 1
                     ) -> Dict[str, Any]:
    """用 **FFT 互相关**一次性算出所有环面平移下的接触值，得到精确零分布。

    ## 为什么需要它

    暴力做法是"每次重定位都重算一遍形态学"：6 个体积 × 200 次 × 300³ 体素，
    实测十分钟都跑不完。而零模型的本质是

        contact(shift) = |S_A ∩ dilate(B_shifted, 1)|
                       = Σ_x S_A(x) · D(x − shift),   D = dilate(B, 1)

    "平移与膨胀可交换"，所以只需算一次 D，再用**一次** FFT 互相关
    就得到所有平移量上的接触值。

    ## 附带好处：精确检验而非抽样

    互相关给出全部 n³ 个平移的接触值，于是 p 值可以按定义直接数：

        p = #{shift : contact(shift) ≥ contact(0)} / n³

    没有抽样误差，也不需要跑 200 次；分辨率是 1/n³ 而不是 1/(n_shifts+1)。

    ## 前提与代价

    * 环面（周期性）平移假设：体积边界被当成与对侧相连。这是空间统计里的
      常用近似，但**必须写进结论**——对贴近体积边界的大结构，周期性会让
      两侧"人为接上"。
    * 内存：300³ float32 约 108 MB，rfftn 复数约 54 MB，可接受。
    """
    from scipy import fft as sfft

    a = np.asarray(a) > 0
    b = np.asarray(b) > 0
    if a.shape != b.shape:
        raise ValueError("形状不一致")
    sa = surface_voxels(a)
    sa_vox = int(sa.sum())
    d = dilate6(b, max(1, tol))
    if sa_vox == 0 or not d.any():
        return {"observed": None, "p_value": None, "n_shifts": 0}

    shape = sa.shape
    F = sfft.rfftn(sa.astype(np.float32))
    G = sfft.rfftn(d.astype(np.float32))
    C = sfft.irfftn(F * np.conj(G), s=shape)
    c = np.rint(C).astype(np.int64)
    c = np.maximum(c, 0)

    observed = int(c[0, 0, 0])
    n_total = int(np.prod(shape))
    ge = int((c >= observed).sum())
    # 用浮点算 p 避免大整数相除的精度问题
    p = float(ge) / float(n_total)
    flat = c.ravel()
    return {
        "observed_faces": observed,
        "observed_frac": (observed / sa_vox) if sa_vox else None,
        "n_shifts": n_total,
        "null_mean": float(flat.mean()),
        "null_median": float(np.median(flat)),
        "null_min": int(flat.min()),
        "null_max": int(flat.max()),
        "null_p95": float(np.percentile(flat, 95)),
        "p_value": round(p, 8),
        "significant_p05": p < 0.05,
        "z": (float((observed - flat.mean()) /
                    (flat.std() if flat.std() > 0 else np.nan))
              if flat.size else None),
        "mode": "fft_exact_toroidal",
        "degenerate_null": bool(flat.std() == 0),
    }


def find_pairs(root: str) -> List[Dict[str, Any]]:
    """扫描 <root>/<dataset>/<crop>/*.npy，返回每个 crop 的类别→路径映射。"""
    out: List[Dict[str, Any]] = []
    pattern = os.path.join(root, "*", "*", "*.npy")
    by_crop: Dict[Tuple[str, str], Dict[str, str]] = {}
    for p in sorted(glob.glob(pattern)):
        parts = p.replace("\\", "/").split("/")
        label = os.path.splitext(parts[-1])[0]
        crop = parts[-2]
        dataset = parts[-3]
        by_crop.setdefault((dataset, crop), {})[label] = p
    for (dataset, crop), labels in sorted(by_crop.items()):
        out.append({"dataset": dataset, "crop": crop, "labels": labels})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="3D 细胞器接触位点分析")
    ap.add_argument("--labels", required=True,
                    help="fetch_labels.py 的输出目录")
    ap.add_argument("--pairs", default="mito:er")
    ap.add_argument("--out", default="data/contact_3d")
    ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--null-shifts", type=int, default=20)
    ap.add_argument("--null-mode", default="fft",
                    choices=["fft", "toroidal", "no_wrap"],
                    help="零模型平移方式。fft=FFT 互相关的精确环面零分布"
                         "（推荐：一次算出全部 n^3 个平移，p 值无抽样误差）；"
                         "toroidal=随机环面平移抽样；"
                         "no_wrap=仅图内合法位置（形状占满体积时会退化）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-voxels", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", default=None,
                    help="官方 train_crop_manifest.csv；给了才能按 nm 出容差剖面")
    ap.add_argument("--tol-nm", default="0,16,32,64",
                    help="以纳米为单位的容差序列（不同分辨率下可比）")
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy", file=sys.stderr)
        return 2

    pairs = []
    for spec in args.pairs.split(","):
        if ":" not in spec:
            print(f"[错误] 类别对格式应为 A:B，得到 {spec!r}", file=sys.stderr)
            return 2
        a, b = spec.split(":", 1)
        pairs.append((a.strip(), b.strip()))

    crops = find_pairs(args.labels)
    if args.limit:
        crops = crops[: args.limit]
    if not crops:
        print(f"[错误] {args.labels} 下没有 <dataset>/<crop>/*.npy", file=sys.stderr)
        return 2
    print(f"找到 {len(crops)} 个 crop 目录")

    # 体素尺寸（nm）：来自官方 manifest。没有它就无法把容差换算成物理尺度，
    # 不同分辨率的 crop 之间也就不可比。
    voxel_map: Dict[Tuple[str, str], List[float]] = {}
    if args.manifest and os.path.exists(args.manifest):
        with open(args.manifest, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                c = str(row["crop_name"])
                key = (row["dataset"], c if c.startswith("crop") else f"crop{c}")
                try:
                    voxel_map[key] = [float(x) for x in
                                      row["voxel_size"].strip().strip("[]").split(";")]
                except Exception:  # noqa: BLE001
                    pass
        print(f"已读入 {len(voxel_map)} 个 crop 的体素尺寸")
    else:
        print("[提示] 未提供 --manifest，将跳过 nm 单位换算（跨分辨率不可比）")

    tols_nm = tuple(float(x) for x in args.tol_nm.split(",") if x.strip())

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    t0 = time.time()

    for rec in crops:
        ds, crop, labels = rec["dataset"], rec["crop"], rec["labels"]
        for an, bn in pairs:
            if an not in labels or bn not in labels:
                skipped.append({"dataset": ds, "crop": crop,
                                "reason": f"缺少 {an} 或 {bn}（有 {list(labels)}）"})
                continue
            A = np.load(labels[an])
            B = np.load(labels[bn])
            if A.shape != B.shape:
                skipped.append({"dataset": ds, "crop": crop,
                                "reason": f"形状不符 {A.shape} vs {B.shape}"})
                continue
            if int((A > 0).sum()) < args.min_voxels or \
               int((B > 0).sum()) < args.min_voxels:
                skipped.append({"dataset": ds, "crop": crop,
                                "reason": "体素数低于 --min-voxels"})
                continue

            st = contact_stats3d(A, B, tol=args.tol)
            if args.null_mode == "fft":
                nm = contact_null_fft(A, B, tol=args.tol)
            else:
                nm = null_model3d(A, B, n_shifts=args.null_shifts,
                                  seed=args.seed, tol=args.tol,
                                  mode=args.null_mode)
            prof = tolerance_profile(A, B)
            vs = voxel_map.get((ds, crop))
            prof_nm = (tolerance_profile_nm(A, B, vs, tols_nm) if vs else None)
            rows.append({"dataset": ds, "crop": crop, "pair": f"{an}:{bn}",
                         "shape": list(A.shape), "voxel_nm": vs, **st,
                         "null": nm, "tol_profile": prof,
                         "tol_profile_nm": prof_nm})
            nm_txt = ""
            if prof_nm:
                nm_txt = " nm剖面=" + str(
                    [(p["tol_nm"], p["faces"]) for p in prof_nm["profile"]])
            print(f"  {crop:<10} {an}:{bn}  A={st['a_voxels']:,} "
                  f"B={st['b_voxels']:,}  接触面={st['contact_faces']:,} "
                  f"perA={st['contact_per_a_surface']:.4f} "
                  f"p={nm.get('p_value')} "
                  f"退化={nm.get('degenerate_null')}{nm_txt} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    if not rows:
        print("[错误] 没有可分析的 crop 对", file=sys.stderr)
        print(json.dumps(skipped[:10], ensure_ascii=False, indent=1),
              file=sys.stderr)
        return 2

    def col(k):
        return [r[k] for r in rows if r.get(k) is not None]

    def summ(v):
        if not v:
            return {"n": 0}
        return {"n": len(v), "median": round(float(statistics.median(v)), 6),
                "mean": round(float(statistics.fmean(v)), 6),
                "min": round(float(min(v)), 6), "max": round(float(max(v)), 6)}

    ps = [r["null"]["p_value"] for r in rows
          if r["null"].get("p_value") is not None]
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "tools/interaction/contact_analysis_3d.py",
        "labels_dir": os.path.abspath(args.labels),
        "pairs": [f"{a}:{b}" for a, b in pairs],
        "tol": args.tol,
        "null_shifts": args.null_shifts,
        "num_measurements": len(rows),
        "num_skipped": len(skipped),
        "metrics": {
            "contact_faces": summ(col("contact_faces")),
            "contact_per_a_surface": summ(col("contact_per_a_surface")),
            "contact_tol_per_a_surface": summ(col("contact_tol_per_a_surface")),
            "a_voxels": summ(col("a_voxels")),
            "b_voxels": summ(col("b_voxels")),
            "a_components": summ(col("a_components")),
            "b_components": summ(col("b_components")),
        },
        "null_model": {
            "n": len(ps),
            "frac_significant_p05": (round(sum(1 for p in ps if p < 0.05)
                                           / len(ps), 3) if ps else None),
            "median_p": round(float(statistics.median(ps)), 4) if ps else None,
        },
        "unit_note": ("contact_faces 单位为面元；乘 voxel² 得面积。"
                      "voxel 大小取自官方 train_crop_manifest.csv。"),
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "per_crop.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "skipped": skipped}, fh,
                  ensure_ascii=False, indent=1)

    # 注意：csv 已在文件顶部 import。这里**不能**再 `import csv`——
    # 在函数体内出现 import 会让 csv 变成整个函数的局部名，
    # 于是前面读 manifest 时抛 UnboundLocalError（已踩过）。
    keys = ["dataset", "crop", "a_voxels", "b_voxels", "a_surface_voxels",
            "contact_faces", "contact_per_a_surface",
            "contact_tol_per_a_surface", "a_components", "b_components"]
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + ["p_value", "significant_p05"])
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in keys}
            row["p_value"] = r["null"].get("p_value")
            row["significant_p05"] = r["null"].get("significant_p05")
            w.writerow(row)

    print("\n" + "=" * 66)
    print(f"测量 {summary['num_measurements']} 次，跳过 {summary['num_skipped']} 次，"
          f"耗时 {time.time()-t0:.1f}s")
    m = summary["metrics"]
    for k, lab in [("contact_per_a_surface", "A表面接触比例"),
                   ("contact_faces", "接触面元数")]:
        s = m[k]
        if s.get("n"):
            print(f"  {lab:<12} n={s['n']} 中位={s['median']} 均值={s['mean']}")
    nm = summary["null_model"]
    print(f"  零模型：n={nm['n']} 显著比例={nm['frac_significant_p05']} "
          f"p 中位={nm['median_p']}")
    print(f"\n输出：{os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
