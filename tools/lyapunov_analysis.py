#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lyapunov_analysis.py —— 在 S-BIAD712 上真正计算最大李雅普诺夫指数

## 为什么现在可以算了

论文原本写的是"李雅普诺夫指数在静态快照上原理上不适用"，这是对的：
该指标要求存在随时间演化的轨迹。项目早期给出的数值是把幂律指数查表映射成
"李雅普诺夫指数"，既无相空间重构也无发散曲线，已撤销。

但 S-BIAD712 提供了真正的时间轴（25 至 100 帧），因此这个指标**第一次具备
计算前提**。本文不做"能不能算出数"的演示，而是回答一个更有用的问题：
**这批数据的时间长度，足以支持混沌判定吗？**

## 方法（与 Pham 2014 同一套流程）

1. **观测量**：每帧的各细胞器面积与各类别对接触面积。
2. **延迟 τ**：自互信息（Fraser & Swinney 1986）的**第一个局部极小**。
3. **嵌入维 m**：取 3（帧数太少，更高维无意义）。
4. **最大指数 λ**：Rosenstein 1993，跟踪最近邻对的距离发散，
   对 ln(发散) 的**线性上升段**做最小二乘。
5. **两种代理检验**：
   * **随机重排**：保留分布、破坏全部时间结构。任何平滑序列都会赢过它，
     故它只排除"这是白噪声"。
   * **相位随机化**：保留功率谱（即自相关），只破坏非线性结构。
     **只有 λ_obs 高于这一个，才是确定性混沌的证据。**

## 两个曾经写错、由校准脚本发现的地方

1. `ami_tau` 首版对逻辑斯蒂映射返回 τ=7，而一维映射应为 τ=1。τ 过大会使
   嵌入点无法约束下一步，发散在 j=1 就饱和。
2. 首版对整条发散曲线拟合，饱和段把斜率拉向 0。

两者叠加，**已知混沌（λ=ln2=0.693）被估成 0.002**。是在
`validate_lyapunov.py` 的校准里发现的：估计器必须先在逻辑斯蒂映射上测出
0.69，才有资格用来否定真实数据。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def ami_tau(x, max_lag=20):
    """自互信息的第一个局部极小（Fraser & Swinney 1986）。

    用等频分箱而非等宽分箱——混沌序列取值分布高度不均，等宽分箱会让大多数
    点落进一两个箱，互信息估计失真。
    """
    n = len(x)
    k = min(12, max(4, int(round(np.sqrt(max(4, n // 5))))))
    edges = np.quantile(x, np.linspace(0, 1, k + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    b = np.digitize(x, edges[1:-1])

    mis = []
    for tau in range(1, min(max_lag, max(2, n // 4)) + 1):
        joint = np.zeros((k, k))
        np.add.at(joint, (b[:-tau], b[tau:]), 1.0)
        p = joint / joint.sum()
        px = p.sum(1, keepdims=True)
        py = p.sum(0, keepdims=True)
        nz = p > 0
        mi = float(np.sum(p[nz] * np.log(p[nz] / (px @ py)[nz])))
        mis.append(mi)

    for i in range(1, len(mis) - 1):
        if mis[i] < mis[i - 1] and mis[i] <= mis[i + 1]:
            return i + 1
    return int(np.argmin(mis)) + 1


def embed(x, m, tau):
    n = len(x) - (m - 1) * tau
    if n <= m + 5:
        return np.zeros((0, m))
    return np.column_stack([x[i * tau:i * tau + n] for i in range(m)])


def rosenstein(x, m, tau, max_iter=8, k_nn=8):
    """Rosenstein 1993 的最大李雅普诺夫指数。返回 (lambda, 拟合步数, 曲线)。

    ## 一个关键的下标错误（已修）

    最初写的是 `||X_i - X_{nn+j}||`，正确形式是 **`||X_{i+j} - X_{nn+j}||`**。
    前者把「点 i 的当前状态」与「邻居的 j 步之后」相比，j=1 时两者相差一个
    时间步，距离立刻达到吸引子直径量级，于是发散曲线从第一步就饱和，
    **已知混沌（逻辑斯蒂 λ=ln2）被估成 0.002**。

    修正后同一校准测出 0.694，误差 0.1%。
    """
    X = embed(x, m, tau)
    if len(X) < m + 10:
        return None, 0, []
    tree = cKDTree(X)
    k = min(k_nn, len(X))
    _, idx = tree.query(X, k=k)

    curve = []
    for j in range(1, max_iter + 1):
        dists = []
        for i in range(len(X) - j):
            for kk in range(1, k):
                nb = int(idx[i, kk])
                if abs(nb - i) < (m - 1) * tau + 1:
                    continue
                if nb + j < len(X):
                    dists.append(np.linalg.norm(X[i + j] - X[nb + j]))
                break
        if len(dists) < 10:
            break
        curve.append(float(np.log(np.mean(dists) + 1e-12)))

    if len(curve) < 3:
        return None, 0, curve

    y = np.array(curve)
    # 对初始上升段拟合。固定取前 J 步，与校准脚本保持一致——
    # 这样"同一长度下已知混沌能测出多少"就可以直接作为参照。
    keep = min(len(y), max(3, min(5, len(y))))
    slope = float(np.polyfit(np.arange(1, keep + 1), y[:keep], 1)[0])
    return slope, keep, curve


def analyse_series(x, rng, n_surr=20):
    x = np.asarray(x, dtype=float)
    if len(x) < 12 or x.std() < 1e-9:
        return None
    x = (x - x.mean()) / x.std()
    tau = ami_tau(x)
    m = 3
    lam, npts, _ = rosenstein(x, m, tau)
    if lam is None:
        return None

    def sl(xs):
        v, _, _ = rosenstein(xs, m, tau)
        return v

    shuf = [v for v in (sl(rng.permutation(x)) for _ in range(n_surr))
            if v is not None]

    F = np.fft.rfft(x)
    amp = np.abs(F)
    ph_surr = []
    for _ in range(n_surr):
        ph = rng.uniform(0, 2 * np.pi, len(F))
        ph[0] = 0
        v = sl(np.fft.irfft(amp * np.exp(1j * ph), n=len(x)))
        if v is not None:
            ph_surr.append(v)

    def st(v):
        return ((float(np.mean(v)), float(np.std(v, ddof=1)))
                if len(v) > 1 else (None, None))

    sm, ss = st(shuf)
    pm, ps = st(ph_surr)
    return {"tau": tau, "m": m, "lambda": lam, "fit_points": npts,
            "n": len(x), "n_embed": int(len(x) - (m - 1) * tau),
            "shuffle_mean": sm, "shuffle_sd": ss,
            "phase_mean": pm, "phase_sd": ps,
            "z_phase": ((lam - pm) / ps)
            if (pm is not None and ps and ps > 0) else None}


def load_features(path, size=256, min_area=50):
    import czifile

    with czifile.CziFile(path) as czi:
        arr = czi.asarray()
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[None]
    T = arr.shape[0]

    names = ["nucleus", "mitochondria", "golgi", "lysosome",
             "peroxisome", "er", "lipid_droplet"]
    st = ndi.generate_binary_structure(2, 1)
    pairs = [("mitochondria", "er"), ("mitochondria", "lysosome"),
             ("golgi", "mitochondria"), ("golgi", "lysosome")]

    feats = []
    for t in range(T):
        row, masks = [], {}
        for ci in range(min(7, arr.shape[1])):
            im = arr[t, ci].astype(np.float32)
            if im.shape[0] > size:
                im = im[:size, :size]
            mk = im > np.percentile(im, 80)
            if mk.sum() < min_area:
                mk = np.zeros_like(mk)
            masks[names[ci]] = mk
            row.append(float(mk.sum()))
        for a, b in pairs:
            if masks[a].any() and masks[b].any():
                ca = (ndi.binary_dilation(masks[a], st, 1) & masks[b]).sum()
            else:
                ca = 0
            row.append(float(ca))
        feats.append(row)
    return np.array(feats, dtype=float), names


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="data/biad712/*.czi")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="logs/BIAD712_LYAPUNOV.json")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(20261115)

    files = sorted(glob.glob(a.paths))
    if a.limit > 0:
        files = files[:a.limit]
    print("文件 %d 个\n" % len(files), flush=True)

    results = []
    for i, f in enumerate(files, 1):
        try:
            feats, names = load_features(f)
        except Exception as e:  # noqa: BLE001
            print("[%d/%d] 读取失败 %s" % (i, len(files), type(e).__name__), flush=True)
            continue
        labels = names[:7] + ["mito:er", "mito:lyso", "golgi:mito", "golgi:lyso"]
        rec = {"file": os.path.basename(f), "T": int(feats.shape[0]), "series": {}}
        for j, lab in enumerate(labels):
            if j >= feats.shape[1]:
                break
            r = analyse_series(feats[:, j], rng)
            if r:
                rec["series"][lab] = r
        results.append(rec)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(results, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("[%d/%d] %s  T=%d  序列 %d"
              % (i, len(files), os.path.basename(f)[:30], feats.shape[0],
                 len(rec["series"])), flush=True)

    print("\n写入 %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
