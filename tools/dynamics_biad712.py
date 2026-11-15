#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dynamics_biad712.py —— 在 S-BIAD712 的活细胞时序上计算细胞器接触动力学

## 这个脚本存在的理由

项目早期报告过李雅普诺夫指数等"动力学"量，复核发现是伪造的，而更根本的问题是
**当时没有任何带时间轴的数据**。论文因此声明动力学不可计算。

现在找到了 S-BIAD712：活细胞、7 种细胞器同时标记、25–100 个时间点。
于是动力学**可以算了**——但只能是光学分辨率下能定义的那几种，不是相空间里的混沌量。

## 算什么

对每个（文件，类别对 A:B）：

1. 逐帧按 Otsu 阈值分割各细胞器通道（去小连通域）；
2. A 在当前帧按 d 像素膨胀后与 B 求交，得到该帧的接触面元数；
3. 得到接触量的**时间序列** `c_t`，t = 0..T-1。

由此导出四个量：

* **接触概率** p = #{t : c_t > 0} / T —— 这段时间里有多少帧确实接触；
* **平均接触分数** = 中位 c_t / A 的周长（按面积估计）—— 接触占 A 表面的比例；
* **生成率 / 消解率** = 0→正 与 正→0 的转换次数 / T —— 接触事件的形成与消失速率；
* **接触寿命** = 连续正段的长度分布（帧数）。

## 零模型：循环时间平移

把 B 的时间序列整体循环平移 k 帧（k 从 1 到 T-1），重算接触序列。
若观测到的接触率与"打乱时间对应关系"后无差别，说明接触只是静态共定位，
没有真正的时间结构；若观测显著更高，说明接触在时间上是协同的。

这个零模型保持了 B 的**全部空间结构与全部运动统计**，只破坏 A 与 B 的
**时间对应**，因此差异只能来自时间协同。

## 局限（必须写清）

* 光学分辨率 ~80 nm/像素，接触判据是"两个分割在 1–2 像素内相邻"，
  与电镜里 10–30 nm 的膜接触位点**不是同一件事**；
* 时程只有 25–100 帧 × 1.9 s，无法观察慢过程；
* 阈值分割代替人工标注，绝对数值不可与电镜结果比较。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import czifile
import numpy as np
from scipy import ndimage as ndi

# 元数据里通道与细胞器的对应（0 基）
CH_ORGANELLE = {
    0: "nucleus", 1: "lysosome", 2: "mitochondria", 3: "golgi",
    4: "peroxisome", 5: "er", 6: "lipid_droplet", 7: "residual",
}
PAIRS = [("mitochondria", "er"), ("mitochondria", "golgi"),
         ("er", "golgi"), ("golgi", "lysosome"), ("er", "lysosome"),
         ("mitochondria", "lysosome"), ("er", "lipid_droplet")]


def otsu(v: np.ndarray) -> float:
    """一维 Otsu 阈值。不引 skimage 以外的新依赖。"""
    from skimage.filters import threshold_otsu
    vv = v[np.isfinite(v)]
    if vv.size == 0 or vv.min() == vv.max():
        return float(vv.max() if vv.size else 0)
    return float(threshold_otsu(vv))


def segment(ch: np.ndarray, min_area: int = 200) -> np.ndarray:
    """单帧单通道分割：Otsu + 去小连通域。"""
    t = otsu(ch)
    m = ch > t
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = np.bincount(lab.ravel())
    keep = np.zeros(sizes.size, dtype=bool)
    keep[1:] = sizes[1:] >= min_area
    return keep[lab]


def load_stack(path: str):
    """读 CZI，返回 (T, C, Y, X) 与元信息。"""
    czi = czifile.CziFile(path)
    arr = czi.asarray().squeeze()
    axes = czi.axes
    meta = czi.metadata()
    if isinstance(meta, bytes):
        meta = meta.decode("utf-8", "replace")
    czi.close()
    if arr.ndim == 4 and arr.shape[1] == 8:
        data = arr
    elif arr.ndim == 4 and arr.shape[0] == 8:
        data = np.moveaxis(arr, 0, 1)
    else:
        raise SystemExit(f"[错误] 形状无法解释: {arr.shape}")
    return data, {"axes": axes, "shape": list(arr.shape)}


def contact_series(A: np.ndarray, B: np.ndarray, d: int) -> np.ndarray:
    """逐帧接触面元数。A/B 形状 (T,Y,X)，已二值化。"""
    st = ndi.generate_binary_structure(2, 1)
    out = np.zeros(A.shape[0], dtype=np.int64)
    for t in range(A.shape[0]):
        Ad = ndi.binary_dilation(A[t], st, iterations=d)
        out[t] = int((Ad & B[t]).sum())
    return out


def transitions(c: np.ndarray) -> tuple:
    """接触序列的 生成 / 消解 次数，以及连续正段长度。"""
    on = c > 0
    if on.size < 2:
        return 0, 0, []
    rise = int(np.sum(~on[:-1] & on[1:]) + (1 if on[0] else 0))
    fall = int(np.sum(on[:-1] & ~on[1:]) + (1 if on[-1] else 0))
    runs, cur = [], 0
    for v in on:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return rise, fall, runs


def analyse(data: np.ndarray, d: int, min_area: int) -> dict:
    T, C = data.shape[0], data.shape[1]
    masks = {}
    for c in range(C):
        name = CH_ORGANELLE.get(c)
        if name in (None, "residual"):
            continue
        per = np.stack([segment(data[t, c].astype(np.float32), min_area)
                        for t in range(T)])
        if per.any():
            masks[name] = per
    out = {"T": int(T), "n_organelles": len(masks), "pairs": {}, "motion": {}}

    # 细胞器整体运动：质心位移 / 帧。这是最直接的时间量，
    # 也是论文里"动力学"最朴素的替代——不涉及相空间，只问它动得多快。
    for name, m in masks.items():
        cents = np.array([ndi.center_of_mass(m[t]) for t in range(T)])
        step = np.linalg.norm(np.diff(cents, axis=0), axis=1) if T > 1 else np.array([0.0])
        out["motion"][name] = {
            "mean_step_px": float(step.mean()),
            "max_step_px": float(step.max()),
            "net_displacement_px": float(np.linalg.norm(cents[-1] - cents[0])),
            # 1 帧 = 1.8956 s，1 px = 79.87 nm
            "mean_speed_nm_per_s": float(step.mean() * 79.87 / 1.8956),
        }
    for a, b in PAIRS:
        if a not in masks or b not in masks:
            continue
        obs = contact_series(masks[a], masks[b], d)
        rise, fall, runs = transitions(obs)
        # 零模型：把 B 的时间序列循环平移
        # 零模型的平移次数：最多取 30 个等间隔位移。
        # T-1 次（100 帧时 99 次）× 7 个类别对 = 693 次 dilate+intersect，
        # 是主要的耗时来源；30 次已足够估计中位数，且保持等间隔覆盖整个周期。
        ks = list(range(1, T))
        if len(ks) > 30:
            step = len(ks) / 30.0
            ks = [ks[int(i * step)] for i in range(30)]
        null_means = []
        for k in ks:
            nb = np.roll(masks[b], k, axis=0)
            nc = contact_series(masks[a], nb, d)
            null_means.append(float(nc.mean()))
        out["pairs"][f"{a}:{b}"] = {
            "contact_frames": int((obs > 0).sum()),
            "contact_prob": float((obs > 0).mean()),
            "mean_contact_px": float(obs.mean()),
            "median_contact_px": float(np.median(obs)),
            "rise": rise, "fall": fall,
            "run_lengths": runs,
            "mean_run": float(np.mean(runs)) if runs else 0.0,
            "null_mean_median": float(np.median(null_means)) if null_means else 0.0,
            "enrichment": (float(obs.mean()) /
                           (np.median(null_means) + 1e-9)) if null_means else 0.0,
            # 时间变异系数：接触量在时间上是否真的在波动。
            # 若 CV 接近 0，说明接触是静态的，"动力学"无从谈起。
            "cv": float(obs.std() / (obs.mean() + 1e-9)),
            # 归一化：接触量除以 A 的周长估计，跨类别对可比
            "contact_frac": (float(obs.mean()) /
                              (np.sqrt(4 * np.pi * masks[a].mean()) + 1e-9)),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+",
                    default=["data/biad712/TS_BR1_N08.czi"])
    ap.add_argument("--dilate", type=int, default=3,
                    help="接触判据：A 膨胀多少像素（1 px ≈ 80 nm）")
    ap.add_argument("--min-area", type=int, default=200)
    ap.add_argument("--out", default="logs/BIAD712_DYNAMICS.json")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    paths = []
    for p in a.paths:
        paths.extend(sorted(glob.glob(p)))
    print(f"待分析文件 {len(paths)} 个，接触判据 {a.dilate} px "
          f"≈ {a.dilate*79.87:.0f} nm")

    # 续跑：JSON 里已有的文件跳过。32 个文件要跑几小时，中断不该从头再来。
    results = []
    done = set()
    if os.path.exists(a.out):
        try:
            results = json.load(open(a.out, encoding="utf-8"))
            done = {r["file"] for r in results}
            print(f"续跑：已有 {len(done)} 个文件，将跳过", flush=True)
        except Exception:
            results, done = [], set()

    for p in paths:
        if os.path.basename(p) in done:
            continue
        print(f"\n=== {os.path.basename(p)} ===", flush=True)
        data, info = load_stack(p)
        print(f"  形状 {data.shape}", flush=True)
        r = analyse(data, a.dilate, a.min_area)
        r["file"] = os.path.basename(p)
        r["dilate_px"] = a.dilate
        r["nm_per_px"] = 79.87
        results.append(r)
        print(f"  可分割细胞器 {r['n_organelles']} 个", flush=True)
        print(f"  {'类别对':<26}{'接触帧':>7}{'概率':>7}{'均值px':>9}"
              f"{'生成':>5}{'消解':>5}{'富集':>8}")
        print(f"  {'类别对':<26}{'接触帧':>7}{'概率':>7}{'均值px':>9}"
              f"{'生成':>5}{'消解':>5}{'CV':>7}{'富集':>8}")
        for k, v in r["pairs"].items():
            print(f"  {k:<26}{v['contact_frames']:>4}/{r['T']:<3}"
                  f"{v['contact_prob']:>7.2f}{v['mean_contact_px']:>9.0f}"
                  f"{v['rise']:>5}{v['fall']:>5}{v['cv']:>7.2f}{v['enrichment']:>8.2f}")
        print(f"  {'--- 细胞器运动 ---'}")
        print(f"  {'细胞器':<26}{'均步长px':>10}{'净位移px':>10}{'速度nm/s':>12}")
        for k, v in r["motion"].items():
            print(f"  {k:<26}{v['mean_step_px']:>10.2f}{v['net_displacement_px']:>10.1f}"
                  f"{v['mean_speed_nm_per_s']:>12.1f}")

        # 每完成一个文件就落盘：32 个文件跑一小时，中断不该丢已完成的
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(results, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print(f"\n写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
