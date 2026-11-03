#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dynamics.py —— 真正的时间序列动力学量（替代伪造的"李亚普诺夫指数"）
====================================================================

为什么需要这个模块
------------------
原 ``code/analyze_emergent_entropy.m`` 里有一个
``infer_lyapunov_from_powerlaw``：它把一条幂律拟合的指数**通过查表**映射成
一个"李亚普诺夫指数"，既没有相空间重构，也没有发散曲线。
那是**编造数字**，比不报更糟。

本模块给出正确的 Rosenstein (1993) 方法。它由五步组成，每一步都显式实现、
单独可调用、单独可检查：

1. ``average_mutual_information`` —— 用**自互信息第一个极小值**估计延迟 τ；
2. ``false_nearest_neighbors`` —— 用**伪最近邻比例降到阈值以下**估计嵌入维 m；
3. ``time_delay_embedding`` —— 由 τ、m 重构相空间 ``X_i = (x_i, x_{i+τ}, …)``；
4. ``rosenstein_divergence`` —— 在 **Theiler 窗**约束下找最近邻，
   计算平均对数发散曲线 ``S(k) = <ln d_i(k)>``；
5. ``lyapunov_rosenstein`` —— 在**用户指定的标度区**内最小二乘拟合斜率，
   返回 ``R²`` 与 ``reliable`` 标志。

**可靠性是第一等公民**：样本 <``min_points``（默认 30）、
拟合窗口内点太少、``R²`` 低于 ``min_r2``（默认 0.9）、或斜率非有限时，
``slope`` 一律返回 ``None`` 且 ``reliable=False``，
并在 ``reason`` 里写明为什么。**绝不返回一个编出来的数字。**

单位
----
* τ：**采样点**（不是秒——本数据集没有采样率）。
* m：**维**。
* 斜率 λ：**nat / 采样点**（发散曲线用自然对数，嵌入空间是欧氏距离）。
  正值 = 邻近轨道指数分离；负值 = 收敛到不动点/周期轨。

**本数据集的重要事实**
----------------------
``annotations_v2`` 是**单时间点**的静态图像分割结果，**没有时间轴**。
因此本模块的所有函数在本项目当前数据上都**不适用于回答生物学问题**。
``run_analysis.py`` 默认 ``--lyapunov-source none`` 就是这个原因；
若强行用"区域编号顺序"或"图像编号顺序"当伪序列，必须显式打开，
且结果会被标记为 ``pseudo_sequence=True``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ==========================================================================
# 兼容性护栏：显式禁止旧的伪造接口
# ==========================================================================

def __getattr__(name: str):
    """阻止任何人再导入旧代码里伪造李亚普诺夫指数的函数名。

    旧实现 ``infer_lyapunov_from_powerlaw`` 通过硬编码查表把幂律指数
    映射成"李亚普诺夫指数"，是纯粹的编造。这里让任何对该名字的访问
    直接失败，并指出替代实现，避免它被无声地复制回来。
    """
    if name in ("infer_lyapunov_from_powerlaw", "lyapunov_from_powerlaw",
                "estimate_lyapunov_exponent_legacy"):
        raise AttributeError(
            f"{name!r} 已被移除：它是一张硬编码查找表，不构成李亚普诺夫指数的估计。"
            "请使用 lyapunov_rosenstein()（Rosenstein 1993 算法），"
            "并检查返回的 reliable 标志。"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ==========================================================================
# 结果容器
# ==========================================================================

@dataclass
class LyapunovResult:
    """Rosenstein 估计的完整结果，含**可靠性判定**。"""

    slope: Optional[float]              # nat / 采样点；不可靠时为 None
    r2: Optional[float]                 # 标度区拟合的决定系数
    reliable: bool
    reason: str
    tau: Optional[int] = None           # 采样点
    m: Optional[int] = None             # 维
    theiler: Optional[int] = None       # 采样点
    fit_start: Optional[int] = None     # 标度区起点（发散步数 k）
    fit_end: Optional[int] = None       # 标度区终点
    n_points: int = 0                   # 参与发散曲线平均的相点数
    n_samples: int = 0                  # 原始序列长度
    n_fit_points: int = 0               # 标度区内的点数
    scaling_rule: str = ""              # 标度区是如何确定的（必须写进报告）
    divergence_k: List[int] = field(default_factory=list)
    divergence_s: List[float] = field(default_factory=list)
    ami: List[float] = field(default_factory=list)
    fnn: List[float] = field(default_factory=list)
    pseudo_sequence: bool = False
    units: str = "nat/sample"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==========================================================================
# 1. 延迟 τ：自互信息第一个极小值
# ==========================================================================

def average_mutual_information(
    x: Sequence[float],
    max_lag: Optional[int] = None,
    n_bins: int = 16,
) -> Dict[str, Any]:
    """自互信息 ``I(τ) = I(x_t ; x_{t+τ})``，单位 **bit**。

    度量的是什么
    ------------
    把 ``x_t`` 与 ``x_{t+τ}`` 各自按**等宽箱**离散成 ``n_bins`` 个符号
    （箱边界由整条序列的 min/max 决定，对所有 τ 固定，保证可比），
    然后计算二者经验联合分布的互信息。

    τ 的选取规则：**第一个局部极小值**（Fraser & Swinney 1986）。
    理由：τ 太小时 ``x_{t+τ} ≈ x_t``，互信息高，重构坐标冗余；
    τ 太大时互信息接近 0，坐标几乎无关，吸引子几何被折叠。
    第一个极小值是冗余与去相关之间的标准折中。

    Returns
    -------
    dict
        ``ami``（长度 ``max_lag`` 的 bit 数组）/ ``tau``（第一个局部极小
        对应的滞后；找不到局部极小时给出 ``argmin`` 并置
        ``local_min_found=False``）/ ``n_bins`` / ``note``。
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if max_lag is None:
        max_lag = int(max(1, min(100, n // 4)))
    out: Dict[str, Any] = {"ami": [], "tau": None, "n_bins": int(n_bins),
                           "local_min_found": False, "note": ""}
    if n < 8:
        out["note"] = f"样本数 {n} < 8，自互信息不可靠"
        return out
    max_lag = int(max(1, min(max_lag, n - 2)))

    lo, hi = float(np.min(v)), float(np.max(v))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out["note"] = "序列为常数或含非有限值，自互信息无定义"
        return out
    edges = np.linspace(lo, hi, int(n_bins) + 1)
    sym = np.clip(np.digitize(v, edges[1:-1]), 0, int(n_bins) - 1)

    ami = np.zeros(max_lag + 1, dtype=np.float64)
    for tau in range(max_lag + 1):
        a, b = sym[: n - tau], sym[tau:]
        joint = np.zeros((int(n_bins), int(n_bins)), dtype=np.float64)
        np.add.at(joint, (a, b), 1.0)
        joint /= joint.sum()
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        mi = 0.0
        for i in range(int(n_bins)):
            if px[i] <= 0:
                continue
            for j in range(int(n_bins)):
                if joint[i, j] > 0 and py[j] > 0:
                    mi += joint[i, j] * math.log2(joint[i, j] / (px[i] * py[j]))
        ami[tau] = mi

    tau_sel: Optional[int] = None
    for t in range(1, max_lag):
        if ami[t] <= ami[t - 1] and ami[t] < ami[t + 1]:
            tau_sel = t
            out["local_min_found"] = True
            break
    if tau_sel is None:
        tau_sel = int(np.argmin(ami[1:]) + 1) if max_lag >= 1 else 1
        out["note"] = "未找到严格局部极小，退化为全局 argmin(τ≥1)"
    if tau_sel < 1:
        tau_sel = 1
    out["ami"] = [float(v_) for v_ in ami]
    out["tau"] = int(tau_sel)
    return out


def select_tau(x: Sequence[float], max_lag: Optional[int] = None,
               n_bins: int = 16) -> int:
    """便捷函数：返回 ``average_mutual_information`` 选出的 τ（采样点）。"""
    return int(average_mutual_information(x, max_lag=max_lag, n_bins=n_bins)["tau"] or 1)


# ==========================================================================
# 2. 嵌入维 m：伪最近邻（FNN）
# ==========================================================================

def time_delay_embedding(
    x: Sequence[float], m: int, tau: int
) -> np.ndarray:
    """时延嵌入：``X_i = (x_i, x_{i+τ}, …, x_{i+(m-1)τ})``。

    Returns
    -------
    np.ndarray
        (N - (m-1)τ, m)；样本不足时返回 (0, m)。
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    m, tau = int(m), int(tau)
    if m < 1 or tau < 1:
        raise ValueError(f"m 与 tau 必须 ≥1，得到 m={m}, tau={tau}")
    n = v.size - (m - 1) * tau
    if n <= 0:
        return np.zeros((0, m), dtype=np.float64)
    idx = np.arange(n)[:, None] + tau * np.arange(m)[None, :]
    return v[idx]


def false_nearest_neighbors(
    x: Sequence[float],
    tau: int,
    max_dim: int = 10,
    rtol: float = 10.0,
    atol: float = 2.0,
    theiler: int = 1,
) -> Dict[str, Any]:
    """Kennel 伪最近邻比例 ``FNN(m)``，无量纲 ∈ [0, 1]。

    度量的是什么
    ------------
    对 m 维嵌入中的每个点，找它的最近邻（排除 Theiler 窗内的自身与
    时间近邻）；若把维数升到 m+1 后，两点距离相对增加超过 ``rtol`` 倍
    **或**新增坐标之差超过 ``atol`` 倍序列标准差，则该点被判为"伪最近邻"
    （即当前的 m 把本该分开的两条轨道压到了一起）。
    ``FNN(m)`` = 伪最近邻点数 / 总点数。

    m 的选取规则：**最小的 m 使得 FNN(m) < 1%**；
    若始终达不到 1%，返回达到最小值时的 m 并置 ``below_threshold=False``。

    Returns
    -------
    dict
        ``fnn``（长度 ``max_dim``，下标 i 对应维数 i+1，``nan`` 表示该维
        不可计算）/ ``m`` / ``below_threshold`` / ``threshold`` / ``note``。
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    threshold = 0.01
    out: Dict[str, Any] = {"fnn": [float("nan")] * int(max_dim), "m": None,
                           "below_threshold": False, "threshold": threshold,
                           "note": ""}
    if n < 20:
        out["note"] = f"样本数 {n} < 20，伪最近邻统计不可靠"
        return out
    sigma = float(np.std(v))
    if sigma <= 0:
        out["note"] = "序列为常数，伪最近邻无定义"
        return out

    for m in range(1, int(max_dim) + 1):
        emb = time_delay_embedding(v, m, tau)
        if emb.shape[0] < 4:
            break
        from scipy.spatial import cKDTree

        tree = cKDTree(emb)
        k = min(2 + int(theiler), emb.shape[0])
        dist, idx = tree.query(emb, k=k)
        if dist.ndim == 1:
            break
        # 取第一个满足 |i-j| > theiler 的邻居
        nn = np.full(emb.shape[0], -1, dtype=np.int64)
        for i in range(emb.shape[0]):
            for c in range(1, k):
                if abs(i - int(idx[i, c])) > int(theiler):
                    nn[i] = int(idx[i, c])
                    break
        ok = nn >= 0
        if not np.any(ok):
            out["fnn"][m - 1] = float("nan")
            continue

        i_idx = np.nonzero(ok)[0]
        j_idx = nn[ok]
        d_m = np.linalg.norm(emb[i_idx] - emb[j_idx], axis=1)
        # m+1 维上的新增坐标
        shift = m * int(tau)
        valid = (i_idx + shift < n) & (j_idx + shift < n)
        if not np.any(valid):
            out["fnn"][m - 1] = float("nan")
            continue
        i2, j2, dm2 = i_idx[valid], j_idx[valid], d_m[valid]
        extra = np.abs(v[i2 + shift] - v[j2 + shift])
        d_m1 = np.sqrt(dm2 ** 2 + extra ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(dm2 > 0, d_m1 / np.maximum(dm2, 1e-12), np.inf)
        is_false = (ratio > rtol) | (extra / sigma > atol)
        out["fnn"][m - 1] = float(np.count_nonzero(is_false) / i2.size)

    finite = [(i + 1, f) for i, f in enumerate(out["fnn"]) if np.isfinite(f)]
    if not finite:
        out["note"] = "所有维度的 FNN 都不可计算"
        return out
    best_m, best_f = min(finite, key=lambda t: (t[1], t[0]))
    if best_f < threshold:
        sel = min(m for m, f in finite if f < threshold)
        out["m"] = int(sel)
        out["below_threshold"] = True
    else:
        out["m"] = int(best_m)
        out["note"] = (f"FNN 最小值 {best_f:.4f} 仍高于阈值 {threshold}；"
                       f"返回最小 FNN 对应的 m={best_m}，标为不可靠候选")
    return out


# ==========================================================================
# 3–5. Rosenstein 发散曲线与斜率
# ==========================================================================

def rosenstein_divergence(
    emb: np.ndarray,
    theiler: int,
    max_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rosenstein 平均对数发散曲线。

    ``S(k) = < ln d_i(k) >_i``，其中
    ``d_i(k) = ||X_{i+k} - X_{j(i)+k}||``，
    ``j(i)`` 是 ``X_i`` 在 **Theiler 窗** ``|i - j| > theiler`` 之外
    的最近邻（Theiler 窗的作用是排除时间上相邻、因而平凡接近的点对，
    否则发散曲线会被自相关污染而系统性偏低）。

    单位：``ln`` 是自然对数，``S`` 的量纲是"距离的对数"（无量纲）。

    Returns
    -------
    (k, S)
        ``k`` = 发散步数（采样点）数组；``S`` 为对应的平均 ln 距离。
        只保留所有轨道对都还有定义的 ``k``。
    """
    from scipy.spatial import cKDTree

    n = emb.shape[0]
    if max_k is None:
        max_k = int(max(1, min(50, n // 10)))
    if n < 4:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    tree = cKDTree(emb)
    kq = min(n, 8 + int(theiler))
    dist, idx = tree.query(emb, k=kq)
    if dist.ndim == 1:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    nn = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        for c in range(1, kq):
            if abs(i - int(idx[i, c])) > int(theiler):
                nn[i] = int(idx[i, c])
                break
    ok = nn >= 0
    i_idx = np.nonzero(ok)[0]
    j_idx = nn[ok]
    if i_idx.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    # 只有前后都能走满 max_k 步的轨道对才能进入发散曲线，
    # 否则 S(k) 在不同 k 上由不同的点子集平均，曲线不可比。
    limit = n - 1 - int(max_k)
    if limit < 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    keep = (i_idx <= limit) & (j_idx <= limit)
    i_idx, j_idx = i_idx[keep], j_idx[keep]
    if i_idx.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    ks = np.arange(0, int(max_k) + 1)
    s = np.zeros(ks.size, dtype=np.float64)
    for a, k in enumerate(ks):
        d = np.linalg.norm(emb[i_idx + k] - emb[j_idx + k], axis=1)
        d = d[d > 0]
        if d.size == 0:
            s[a] = np.nan
        else:
            s[a] = float(np.mean(np.log(d)))
    good = np.isfinite(s)
    return ks[good], s[good]


def auto_scaling_region(
    ks: np.ndarray, s: np.ndarray, min_fit_points: int = 8
) -> Tuple[int, int, str]:
    """自动标度区：**在发散曲线进入饱和平台之前**截断。

    规则（写死在这里，报告必须引用）
    --------------------------------
    1. ``S_min = S(0)``（k=0 处的平均 ln 最近邻距离），
       ``S_max = 95 分位数``（对饱和平台水平的稳健估计，避免被单点尖峰拉高）；
    2. **起点** ``k0`` = 第一个满足 ``S ≥ S_min + 0.05·(S_max − S_min)`` 的 k；
    3. **终点** ``k1`` = 第一个满足 ``S ≥ S_min + 0.90·(S_max − S_min)`` 的 k
       —— 到这一步曲线已经贴到平台，再往后是饱和段，斜率不再是 λ；
    4. 若 ``k1 - k0 + 1 < min_fit_points``，则**判定为"无可用标度区"**，
       返回该区间并由调用方拒绝给出数值。

    为什么需要这一步
    ----------------
    Rosenstein 曲线只在"最近邻距离从初始 ε 指数增长到吸引子尺寸"这一段
    是线性的，之后必然饱和。原实现（也是对 ``nolds.lyap_r`` 默认参数的常见
    误用）直接对整条曲线做最小二乘，等于把饱和段也算进斜率，
    得到的 R² 很低而斜率被系统性拉小。**默认就必须避开饱和段。**

    Returns
    -------
    (k0, k1, rule)
        ``rule`` 是人类可读的规则说明，会写进结果字典。
    """
    if ks.size == 0:
        return 0, 0, "发散曲线为空"
    s_max = float(np.percentile(s, 95))
    s_min = float(s[0])
    span = s_max - s_min
    if not np.isfinite(span) or span <= 0:
        return 0, int(ks.size - 1), "曲线无增长（span ≤ 0），退化为全区间"

    k0 = 0
    for i in range(ks.size):
        if s[i] >= s_min + 0.05 * span:
            k0 = int(i)
            break
    k1 = int(ks.size - 1)
    for i in range(ks.size):
        if s[i] >= s_min + 0.90 * span:
            k1 = int(i)
            break
    if k1 < k0:
        k1 = int(ks.size - 1)
    return k0, k1, (
        f"自动标度区：S_min={s_min:.4f}，S_max(p95)={s_max:.4f}；"
        f"起点 = 首个 S≥S_min+5%·span 的 k，终点 = 首个 S≥S_min+90%·span 的 k "
        f"（避开饱和平台）；要求区间点数 ≥ min_fit_points={min_fit_points}"
    )


def lyapunov_rosenstein(
    x: Sequence[float],
    tau: Optional[int] = None,
    m: Optional[int] = None,
    theiler: Optional[int] = None,
    fit_start: Optional[int] = None,
    fit_end: Optional[int] = None,
    min_points: int = 30,
    min_fit_points: int = 8,
    min_r2: float = 0.90,
    max_k: Optional[int] = None,
    pseudo_sequence: bool = False,
) -> LyapunovResult:
    """Rosenstein (1993) 最大李亚普诺夫指数估计。

    算法
    ----
    1. ``tau=None`` 时用 ``average_mutual_information`` 的第一个局部极小；
    2. ``m=None`` 时用 ``false_nearest_neighbors`` 的 1% 阈值；
    3. ``theiler=None`` 时取 ``tau``（标准做法：窗宽取一个延迟）；
    4. 时延嵌入 → Theiler 窗约束下找最近邻 → 平均对数发散曲线 ``S(k)``；
    5. 在 ``[fit_start, fit_end]``（缺省用 ``auto_scaling_region`` 自动
       避开饱和平台）内做**最小二乘**直线拟合 ``S(k) ≈ λ k + c``，
       返回斜率 λ、``R²`` 与 ``reliable``。

    单位
    ----
    λ 的单位是 **nat / 采样点**（``ln`` 自然对数）。因为本数据集没有采样率，
    无法换算成 nat/s 或 bit/s。

    **可靠性判据（任一不满足即 ``slope=None``、``reliable=False``）**
    ----------------------------------------------------------------
    * ``n_samples < min_points``（默认 30）——**点数不足直接拒绝**，
      这正是本数据集会遇到的情况；
    * 标度区内点数 < ``min_fit_points``（默认 8）——3 个点的"直线"
      不能称为标度区；
    * 发散曲线总点数 < ``min_points``；
    * ``R² < min_r2``（默认 0.90）；
    * 斜率非有限。

    对"数据撑不起这个估计"的情形，函数**返回 ``None`` 并在 ``reason``
    里说明**，而不是给出一个看似合理的数字。

    Parameters
    ----------
    pseudo_sequence
        明确标注输入不是真实时间序列（例如区域编号顺序、图像编号顺序）。
        该标志会被原样写进结果，提醒读者不要做动力学解释。

    Returns
    -------
    LyapunovResult
        见 dataclass 定义。
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = v.size

    res = LyapunovResult(
        slope=None, r2=None, reliable=False, reason="",
        n_samples=int(n), pseudo_sequence=bool(pseudo_sequence),
    )
    if n < int(min_points):
        res.reason = (f"样本数 {n} < min_points={min_points}，"
                      "无法可靠估计李亚普诺夫指数（本数据集为单时间点静态图像，"
                      "本就没有可供重构相空间的时间序列）")
        return res

    ami_info = average_mutual_information(v)
    res.ami = list(ami_info["ami"])
    tau_used = int(tau) if tau is not None else int(ami_info["tau"] or 1)

    if m is None:
        fnn_info = false_nearest_neighbors(v, tau=tau_used)
        res.fnn = [float(f) for f in fnn_info["fnn"]]
        m_used = int(fnn_info["m"] or 2)
    else:
        m_used = int(m)
        res.fnn = []
    if m_used < 1 or tau_used < 1:
        res.reason = f"嵌入参数非法：tau={tau_used}, m={m_used}"
        return res

    theiler_used = int(theiler) if theiler is not None else int(theiler_default(tau_used))
    res.tau, res.m, res.theiler = tau_used, m_used, theiler_used

    emb = time_delay_embedding(v, m_used, tau_used)
    if emb.shape[0] < 10:
        res.reason = (f"嵌入后相点数仅 {emb.shape[0]}（N={n}, m={m_used}, τ={tau_used}），"
                      "不足以构成发散曲线")
        return res

    ks, s = rosenstein_divergence(emb, theiler_used, max_k=max_k)
    res.divergence_k = [int(k) for k in ks]
    res.divergence_s = [float(v_) for v_ in s]
    res.n_points = int(emb.shape[0])
    if ks.size < 3:
        res.reason = f"发散曲线只有 {ks.size} 个点，无法拟合标度区"
        return res

    lo, hi, rule = ((fit_start, fit_end, "用户显式指定标度区")
                    if (fit_start is not None and fit_end is not None)
                    else auto_scaling_region(ks, s, min_fit_points=min_fit_points))
    res.scaling_rule = rule
    lo = max(0, int(lo))
    hi = min(int(hi), ks.size - 1)
    res.fit_start, res.fit_end = lo, hi
    res.n_fit_points = int(hi - lo + 1)
    if res.n_fit_points < int(min_fit_points):
        res.reason = (
            f"标度区 [{lo}, {hi}] 内只有 {res.n_fit_points} 个点 "
            f"< min_fit_points={min_fit_points}：发散曲线在 {res.n_fit_points} 步内"
            "就进入饱和平台，没有可用于估计 λ 的线性段（常见原因是嵌入延迟 τ 过大）"
        )
        return res

    kk = ks[lo:hi + 1].astype(np.float64)
    ss = s[lo:hi + 1]
    A = np.column_stack([kk, np.ones_like(kk)])
    try:
        coef, *_ = np.linalg.lstsq(A, ss, rcond=None)
    except np.linalg.LinAlgError as exc:
        res.reason = f"最小二乘失败：{exc}"
        return res
    slope, intercept = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(np.sum((ss - pred) ** 2))
    ss_tot = float(np.sum((ss - ss.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    res.slope = slope if np.isfinite(slope) else None
    res.r2 = float(r2) if np.isfinite(r2) else None

    problems: List[str] = []
    if res.slope is None:
        problems.append("斜率为非有限值")
    if res.r2 is None or not np.isfinite(r2):
        problems.append("R² 无法计算（标度区内 S 为常数）")
    elif r2 < float(min_r2):
        problems.append(f"R²={r2:.3f} < min_r2={min_r2}")
    if ks.size < int(min_points):
        problems.append(f"发散曲线点数 {ks.size} < min_points={min_points}")

    if problems:
        res.slope = None
        res.reliable = False
        res.reason = ("估计不可靠，已拒绝给出数值：" + "；".join(problems)
                      + "。请提供更长/更干净的时间序列，或放宽阈值后自行判断。")
    else:
        res.reliable = True
        res.reason = (f"标度区 k∈[{lo}, {hi}]（{hi - lo + 1} 点），"
                      f"R²={r2:.4f}，斜率单位 nat/采样点")
        if pseudo_sequence:
            res.reason += "；**输入为伪序列，无动力学含义**"
    return res


def theiler_default(tau: int) -> int:
    """Theiler 窗默认值：一个延迟 τ（标准做法）。"""
    return int(max(1, tau))


def lyapunov_sensitivity_scan(
    x: Sequence[float],
    taus: Optional[Sequence[int]] = None,
    dims: Sequence[int] = (2, 3, 4, 5),
    min_points: int = 30,
    min_fit_points: int = 8,
    min_scan_fit_points: int = 10,
    min_r2: float = 0.90,
    pseudo_sequence: bool = False,
) -> Dict[str, Any]:
    """对 (τ, m) 做**敏感性扫描**，并给出"这个 λ 稳不稳"的判据。

    为什么必须做
    ------------
    Rosenstein 估计对嵌入参数高度敏感：``average_mutual_information``
    的第一个极小值在离散映射（如 logistic 映射）上常常偏大，
    而 τ 一大，发散曲线两三步就饱和，标度区消失，λ 会被严重低估。
    只报一个 (τ, m) 下的数字是**不负责任**的。

    本函数按固定网格重估 λ，返回**完整表格**（不隐藏失败的行），
    并给出稳健性判据：

    * ``n_reliable``——通过全部可靠性检验、**且标度区点数 ≥
      ``min_scan_fit_points``（默认 10）** 的 (τ, m) 组合数。
      只有标度区足够长的估计才携带关于 λ 的信息：3 个点的"直线"
      几乎总能拟合出某个斜率，但它不构成约束，因此**不允许它否决结论**；
    * ``median_slope`` / ``iqr_slope``——上述组合的 λ 中位数与四分位距；
    * ``best_r2_slope`` / ``best_r2_tau`` / ``best_r2_m``——R² 最高的那一个
      估计（**注意：按 R² 选模型本身是一种模型选择，报告时必须写明**，
      不能把它当成"真值"）；
    * ``consistent``——可靠估计中落在中位数 ±25% 内的比例 ≥ 0.6；
    * ``robust``——``n_reliable ≥ 3`` 且 ``consistent``。

    τ 默认网格：``1, 2, …, min(10, ami_tau)`` 再并上 ``ami_tau`` 本身；
    m 默认网格：``(2, 3, 4, 5)``。扫描网格固定、无随机性，可复现。

    Returns
    -------
    dict
        ``table``（每行一个 (τ,m) 的结果，**失败行也保留**）/
        ``n_evaluated`` / ``n_reliable`` / ``median_slope`` / ``iqr_slope`` /
        ``best_r2_slope`` / ``consistent`` / ``robust`` / ``taus`` /
        ``dims`` / ``note``。
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    ami_tau = int(average_mutual_information(v)["tau"] or 1)

    if taus is None:
        hi = int(min(10, max(1, ami_tau)))
        taus = sorted(set(range(1, hi + 1)) | {ami_tau})
    taus = [int(t) for t in taus]

    rows: List[Dict[str, Any]] = []
    slopes: List[float] = []
    best: Optional[Dict[str, Any]] = None
    for tau in taus:
        for m in dims:
            res = lyapunov_rosenstein(
                v, tau=tau, m=int(m), min_points=min_points,
                min_fit_points=min_fit_points, min_r2=min_r2,
                pseudo_sequence=pseudo_sequence,
            )
            rows.append({
                "tau": tau, "m": int(m),
                "slope": res.slope, "r2": res.r2,
                "reliable": res.reliable,
                "n_fit_points": res.n_fit_points,
                "fit_start": res.fit_start, "fit_end": res.fit_end,
                "reason": res.reason,
            })
            if res.reliable and res.slope is not None:
                if res.n_fit_points >= int(min_scan_fit_points):
                    slopes.append(float(res.slope))
                if best is None or (res.r2 or -1) > (best["r2"] or -1):
                    best = {"slope": float(res.slope), "r2": float(res.r2 or 0.0),
                            "tau": tau, "m": int(m),
                            "n_fit_points": res.n_fit_points}

    arr = np.asarray(slopes, dtype=np.float64)
    out: Dict[str, Any] = {
        "table": rows,
        "n_evaluated": len(rows),
        "n_reliable": int(arr.size),
        "median_slope": float(np.median(arr)) if arr.size else None,
        "iqr_slope": (float(np.percentile(arr, 75) - np.percentile(arr, 25))
                      if arr.size else None),
        "best_r2_slope": (best or {}).get("slope"),
        "best_r2": (best or {}).get("r2"),
        "best_r2_tau": (best or {}).get("tau"),
        "best_r2_m": (best or {}).get("m"),
        "consistent": False,
        "robust": False,
        "taus": taus,
        "dims": [int(d) for d in dims],
        "ami_tau": ami_tau,
        "min_scan_fit_points": int(min_scan_fit_points),
        "note": "",
    }
    if arr.size == 0:
        out["note"] = ("扫描网格内没有任何 (τ, m) 组合通过可靠性检验"
                       f"（标度区点数 ≥ {min_scan_fit_points}），"
                       "因此**不给出** λ 的数值")
        return out
    med = float(np.median(arr))
    within = float(np.mean(np.abs(arr - med) <= 0.25 * abs(med))) if med != 0 else 0.0
    out["consistent"] = bool(within >= 0.6)
    out["robust"] = bool(arr.size >= 3 and out["consistent"])
    out["fraction_within_25pct"] = within
    if not out["robust"]:
        out["note"] = (
            f"仅 {arr.size} 个组合可靠，且只有 {within:.0%} 落在中位数 ±25% 内；"
            "λ 对嵌入参数不稳定，不应作为结论使用"
        )
    return out


# ==========================================================================
# 汇总：给 CLI 用
# ==========================================================================

def analyze_series(
    x: Sequence[float],
    pseudo_sequence: bool = False,
    fit_start: Optional[int] = None,
    fit_end: Optional[int] = None,
    min_points: int = 30,
    min_fit_points: int = 8,
    min_scan_fit_points: int = 10,
    min_r2: float = 0.90,
    scan: bool = True,
) -> Dict[str, Any]:
    """一次跑完 AMI + FNN + Rosenstein（+ 可选敏感性扫描）。

    这是给 ``run_analysis.py`` 的接口：无论结果是否可靠，都返回一个字典，
    里面必然包含 ``reliable`` / ``reliability_basis`` / ``reason``，
    便于报告如实呈现。

    返回字典中的三层结论（**互不替代，都要读**）
    -------------------------------------------
    ``primary_*``
        标准流程的单一估计：τ 取自互信息第一个极小，m 取伪最近邻 1% 阈值。
        ``primary_reliable`` 是这个单一估计自身的可靠性。
    ``sensitivity_scan``
        (τ, m) 网格扫描，回答"这个 λ 稳不稳"。
    ``reliable`` / ``reliability_basis`` / ``recommended_slope``
        综合判定。``reliability_basis`` ∈ ``{"primary", "sensitivity_scan",
        "none"}``：

        * ``"primary"``：单一标准估计直接通过全部检验，取它的斜率；
        * ``"sensitivity_scan"``：标准 τ/m 不过关，但扫描显示 λ 在
          有足够长标度区的参数组合上稳健——此时 ``recommended_slope``
          取这些组合的**中位数**，并在 ``reason`` 里写明该结论来自参数扫描
          而非单一 (τ, m)；
        * ``"none"``：两条路都不成立，``recommended_slope = None``。
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    ami = average_mutual_information(v)
    tau = int(ami["tau"] or 1)
    fnn = false_nearest_neighbors(v, tau=tau)
    res = lyapunov_rosenstein(
        v, tau=tau, m=(int(fnn["m"]) if fnn["m"] else None),
        fit_start=fit_start, fit_end=fit_end,
        min_points=min_points, min_fit_points=min_fit_points, min_r2=min_r2,
        pseudo_sequence=pseudo_sequence,
    )
    out = res.to_dict()
    out["primary_reliable"] = bool(res.reliable)
    out["primary_slope"] = res.slope
    out["primary_r2"] = res.r2
    out["primary_tau"] = res.tau
    out["primary_m"] = res.m
    out["ami_tau"] = tau
    out["ami_local_min_found"] = ami["local_min_found"]
    out["fnn_curve"] = [float(f) for f in fnn["fnn"]]
    out["fnn_below_threshold"] = fnn["below_threshold"]
    out["fnn_note"] = fnn["note"]
    out["series_length"] = int(v.size)

    if not scan:
        out["reliable"] = bool(res.reliable)
        out["reliability_basis"] = "primary" if res.reliable else "none"
        out["recommended_slope"] = res.slope
        return out

    sc = lyapunov_sensitivity_scan(
        v, min_points=min_points, min_fit_points=min_fit_points,
        min_scan_fit_points=min_scan_fit_points, min_r2=min_r2,
        pseudo_sequence=pseudo_sequence,
    )
    out["sensitivity_scan"] = dict(sc)

    if res.reliable:
        out["reliable"] = True
        out["reliability_basis"] = "primary"
        out["recommended_slope"] = res.slope
        if not sc["robust"]:
            out["reason"] = (out["reason"] + " | 注意：敏感性扫描未通过（"
                             + sc["note"] + "）").strip()
    elif sc["robust"]:
        out["reliable"] = True
        out["reliability_basis"] = "sensitivity_scan"
        out["recommended_slope"] = sc["median_slope"]
        out["reason"] = (
            f"标准 (τ={res.tau}, m={res.m}) 估计不可靠（{res.reason}）；"
            f"但 (τ, m) 扫描中 {sc['n_reliable']} 个标度区充足的组合给出稳健的 "
            f"λ = {sc['median_slope']:.4f}（IQR {sc['iqr_slope']:.4f}，"
            f"单位 nat/采样点），参数敏感性已量化，结论来自扫描而非单一估计"
        )
    else:
        out["reliable"] = False
        out["reliability_basis"] = "none"
        out["recommended_slope"] = None
        out["reason"] = (res.reason + " | 敏感性扫描：" + sc["note"])
    return out
