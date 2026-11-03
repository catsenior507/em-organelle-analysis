#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py —— 在**真正建好的网络**上计算涌现/网络指标
=====================================================

本模块存在的理由
----------------
项目原有的 ``code/analyze_emergent_entropy.m`` 把
"对每区域特征表的一列做直方图、算直方图熵"称作"网络香农熵"，
把一条幂律拟合的指数称作"李亚普诺夫指数"。**那里没有网络**，
而且用**原始**直方图熵去比较区域数 N 相差几十倍的样本——
直方图熵的上界是 ``log2(N)``，N 越大熵越大，跨样本比较原始值没有意义。

本模块的每一条函数都在文档字符串里写明三件事：
1. 它度量的是**哪个概率分布**；
2. **单位**是什么（bit 还是无量纲）；
3. **退化条件**（什么时候这个数字没有意义）。

统一约定
--------
* 所有熵一律以 **bit**（以 2 为底）为单位；如需 nat 请自行乘 ``ln 2``。
* 所有需要随机性的函数都接受 ``seed`` 参数，默认 0，保证可复现。
* 依赖 ``networkx`` 的函数在缺失时抛出带安装提示的 ``ImportError``；
  模块内同时提供不依赖 networkx 的替代实现（见各函数"降级"说明）。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: 平均最短路径/介数中心性的**硬上限**：超过这个节点数就不算，
#: 因为 networkx 的 ``betweenness_centrality`` 是 O(N·M)，
#: 而 ``average_shortest_path_length`` 全源 BFS 是 O(N·M)，
#: N > 300 时单个指标就会拖垮整个批处理。跳过时返回 ``None`` 并说明原因，
#: **绝不**估一个假数字出来。
MAX_NODES_FOR_EXPENSIVE_METRICS = 300

#: 熵的底数转换因子（nat → bit）
LOG2E = 1.4426950408889634


# ==========================================================================
# 0. 通用小工具
# ==========================================================================

def shannon_entropy_bits(pmf: Sequence[float]) -> float:
    """离散香农熵 ``H = -Σ p log2 p``，单位 **bit**。

    Parameters
    ----------
    pmf
        非负权重（不必已归一化，内部会归一化）。零元素被忽略
        （``0 log 0 := 0``）。

    Returns
    -------
    float
        单位为 bit。空分布或全零分布返回 0.0。
    """
    p = np.asarray(pmf, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def _require_networkx(purpose: str):
    """导入 networkx；缺失时给出明确安装提示。"""
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"{purpose} 需要 networkx（本机未安装）。"
            "请 `pip install networkx`；metrics.py 中的熵类与鲁棒性函数"
            "不依赖 networkx，可单独使用。"
        ) from exc
    return nx


def connected_components(adjacency) -> List[np.ndarray]:
    """无向图的连通分量（不依赖 networkx）。

    Returns
    -------
    list of np.ndarray
        每个元素是一个分量的节点下标数组，按大小降序排列。
    """
    a = np.asarray(adjacency.todense() if hasattr(adjacency, "todense") else adjacency)
    n = a.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps: List[np.ndarray] = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        cur = [s]
        while stack:
            u = stack.pop()
            nb = np.nonzero(a[u])[0]
            for v in nb:
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
                    cur.append(int(v))
        comps.append(np.asarray(sorted(cur), dtype=np.int64))
    comps.sort(key=len, reverse=True)
    return comps


def largest_connected_component(adjacency) -> np.ndarray:
    """最大连通分量的节点下标（空图返回空数组）。"""
    comps = connected_components(adjacency)
    return comps[0] if comps else np.zeros(0, dtype=np.int64)


# ==========================================================================
# 1. 度分布熵
# ==========================================================================

def degree_distribution_entropy(
    degrees: Sequence[float], n_nodes: Optional[int] = None
) -> Dict[str, float]:
    """度分布的香农熵及其**归一化**版本。

    度量的是什么
    ------------
    把每个节点视为等概率抽样（概率 ``1/N``）时，随机抽一个节点，
    其**度**这个随机变量的香农熵。即对度取值做经验 pmf
    ``p_k = n_k / N``（``n_k`` = 度为 k 的节点数），
    ``H_deg = -Σ_k p_k log2 p_k``，单位 **bit**。

    **为什么必须报告归一化版本（原代码的核心 bug）**
    ------------------------------------------------
    ``H_deg`` 的上界是 ``log2(支持集大小) ≤ log2(N)``，因此它随节点数 N
    单调增长：一个 30 节点的图即使结构完全规则，熵也可能高于一个
    5 节点的小图。原 ``analyze_emergent_entropy.m`` 直接跨样本比较
    原始直方图熵，而各图区域数相差几十倍——**那个比较在数学上无效**。
    本函数因此固定返回三个量：

    * ``H_deg_bits``：原始熵（bit）；
    * ``H_deg_max_bits`` = ``log2(N)``：该 N 下的上界
      （当且仅当 N 个节点的度两两不同且各占 1/N 时取到）；
    * ``H_deg_normalized`` = ``H_deg_bits / log2(N)`` ∈ [0, 1]：
      **跨样本比较只能用这个量**。

    Parameters
    ----------
    degrees
        (N,) 每个节点的度。
    n_nodes
        节点总数；缺省时取 ``len(degrees)``。显式给出可在子图上计算时
        仍用全图的 N 做归一化。

    Returns
    -------
    dict
        ``H_deg_bits`` / ``H_deg_max_bits`` / ``H_deg_normalized`` /
        ``n_nodes`` / ``n_degree_values``（不同度取值的个数）/
        ``degenerate``（bool：不同度取值 < 2，即完全正则图，H=0）。
    """
    deg = np.asarray(degrees, dtype=np.float64).ravel()
    n = int(n_nodes if n_nodes is not None else deg.size)
    h = shannon_entropy_bits(deg)
    hmax = math.log2(n) if n > 1 else 0.0
    hn = (h / hmax) if hmax > 0 else 0.0
    _, counts = np.unique(deg, return_counts=True)
    return {
        "H_deg_bits": float(h),
        "H_deg_max_bits": float(hmax),
        "H_deg_normalized": float(hn),
        "n_nodes": n,
        "n_degree_values": int(counts.size),
        "degenerate": bool(counts.size < 2),
    }


# ==========================================================================
# 2. 边权 / 边类型分布熵
# ==========================================================================

def edge_weight_entropy(
    weights: Sequence[float],
    n_bins: int = 4,
    binning: str = "quantile",
) -> Dict[str, Any]:
    """边权分布的香农熵，单位 **bit**。

    度量的是什么
    ------------
    随机抽一条**边**（每条边等概率 ``1/M``），其**权重落在哪个箱**
    这个离散随机变量的香农熵。箱子规则必须显式声明，否则数字不可比——
    所以本函数把 ``binning`` 与 ``bin_edges`` 一起返回。

    分箱规则（``binning``）
    -----------------------
    ``'quantile'``（默认）
        等频分箱：用 ``np.quantile`` 取 ``n_bins-1`` 个分位点作为切点，
        每个箱期望包含 ``M/n_bins`` 条边。**箱内边数尽量相等意味着
        在权重连续且无重复时 H ≈ log2(n_bins)**——因此这个量主要反映的是
        "权重分布是否高度集中在少数几个值上"，而不是"分布有多宽"。
        这正是我们要的：接触长度若有成簇结构（很多边接触长度相同），
        分位箱会把这些质量堆进少数箱，熵下降。
    ``'equal_width'``
        等宽分箱：在 ``[min, max]`` 上等分。熵此时对分布的**离散程度**
        敏感（权重集中 → 低熵）。
    ``'fixed'``
        固定宽度分箱，宽度由 ``bin_width`` 给出（需调用方传入，
        通过 ``binning='fixed'`` 且 ``n_bins`` 解释为宽度）。
        本函数不推荐用它，仅备查。

    Returns
    -------
    dict
        ``H_weight_bits`` / ``H_weight_max_bits`` = ``log2(k)``
        （k = 实际使用的箱数）/ ``H_weight_normalized`` /
        ``n_edges``（**永远返回，因为熵强烈依赖样本量**）/
        ``n_bins`` / ``bin_edges`` / ``binning`` / ``degenerate``
        （所有权重相同 → H=0）。
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    m = int(w.size)
    out: Dict[str, Any] = {
        "n_edges": m, "n_bins": 0, "binning": binning,
        "bin_edges": [], "H_weight_bits": 0.0,
        "H_weight_max_bits": 0.0, "H_weight_normalized": 0.0,
        "degenerate": True,
    }
    if m == 0:
        out["note"] = "没有边，边权熵无定义"
        return out

    w = w[np.isfinite(w)]
    if w.size == 0:
        out["note"] = "所有边权非有限，边权熵无定义"
        return out
    if np.allclose(w, w[0]):
        out.update({
            "H_weight_bits": 0.0, "n_bins": 1,
            "bin_edges": [float(w[0]), float(w[0])],
            "degenerate": True,
            "note": "所有边权完全相同，熵恒为 0（这是真实退化，不是 bug）",
        })
        return out

    k = max(2, int(min(n_bins, w.size)))
    if binning == "quantile":
        qs = np.linspace(0.0, 1.0, k + 1)[1:-1]
        edges = np.unique(np.quantile(w, qs))
    elif binning == "equal_width":
        edges = np.linspace(w.min(), w.max(), k + 1)[1:-1]
    elif binning == "fixed":
        width = float(n_bins) if n_bins > 0 else 1.0
        edges = np.arange(w.min() + width, w.max(), width)
    else:
        raise ValueError(f"未知分箱规则 {binning!r}；可选 quantile/equal_width/fixed")

    counts, edges_used = np.histogram(w, bins=np.concatenate([[w.min() - 1e-12], edges, [w.max() + 1e-12]]))
    k_used = int(np.count_nonzero(counts))
    h = shannon_entropy_bits(counts)
    hmax = math.log2(k_used) if k_used > 1 else 0.0
    out.update({
        "H_weight_bits": float(h),
        "H_weight_max_bits": float(hmax),
        "H_weight_normalized": float(h / hmax) if hmax > 0 else 0.0,
        "n_bins": int(len(edges_used) - 1),
        "n_bins_nonempty": k_used,
        "bin_edges": [float(e) for e in edges_used],
        "bin_counts": [int(c) for c in counts],
        "degenerate": bool(k_used < 2),
        "note": (
            "等频分箱下熵的上界接近 log2(箱数)；此量反映权重是否聚集在少数取值上"
            if binning == "quantile" else ""
        ),
    })
    return out


# ==========================================================================
# 3. 拓扑指标
# ==========================================================================

def clustering_coefficients(adjacency) -> Dict[str, Any]:
    """全局传递性（transitivity）与平均局部聚类系数。

    度量的是什么
    ------------
    * **局部聚类系数** ``C_i`` = 节点 i 的邻居之间实际存在的边数 /
      ``k_i(k_i-1)/2``（``k_i < 2`` 时按 networkx 约定为 0）。
      它是"i 的邻居互为邻居"的条件概率的经验估计。
    * **平均局部** = ``mean_i C_i``（对**所有** N 个节点取平均，
      含度为 0/1 的节点——这一点与 networkx 一致，必须声明，
      因为只对 ``k≥2`` 取平均会得到偏大的值）。
    * **全局传递性** = ``3 × 三角形数 / 连通三元组数``，
      等价于对**三元组**等权平均而不是对节点等权平均。

    两者都是**无量纲**的 [0,1] 量，不是熵。

    Returns
    -------
    dict
        ``clustering_global``（传递性）/ ``clustering_mean_local`` /
        ``n_triangles`` / ``n_nodes_degree_ge2`` / ``degenerate``
        （``n_triangles == 0``）。
    """
    a = np.asarray(adjacency.todense() if hasattr(adjacency, "todense") else adjacency)
    a = (a != 0).astype(np.float64)
    np.fill_diagonal(a, 0.0)
    n = a.shape[0]
    if n < 3:
        return {"clustering_global": 0.0, "clustering_mean_local": 0.0,
                "n_triangles": 0, "n_nodes_degree_ge2": 0, "degenerate": True,
                "note": f"N={n} < 3，聚类系数无定义"}

    deg = a.sum(axis=1)
    # 三角形数：trace(A^3)/6
    a3 = a @ a @ a
    n_tri = float(np.trace(a3) / 6.0)
    tri_per_node = np.diag(a3) / 2.0                     # 每个节点参与的三角形数
    denom = deg * (deg - 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        local = np.where(denom > 0, 2.0 * tri_per_node / np.maximum(denom, 1e-12), 0.0)
    triples = float(np.sum(denom) / 2.0)                 # 连通三元组数
    transitivity = (3.0 * n_tri / triples) if triples > 0 else 0.0

    return {
        "clustering_global": float(transitivity),
        "clustering_mean_local": float(np.mean(local)),
        "n_triangles": int(round(n_tri)),
        "n_nodes_degree_ge2": int(np.count_nonzero(deg >= 2)),
        "degenerate": bool(n_tri == 0),
        "note": "" if n_tri > 0 else "图中没有任何三角形，两个聚类系数都为 0",
    }


def characteristic_path_length(adjacency) -> Dict[str, Any]:
    """特征路径长度——**只在最大连通分量（LCC）上计算**。

    度量的是什么
    ------------
    LCC 上所有节点对的**最短路径长度（边数）**的算术平均，
    单位是 **边（跳数）**，不是物理长度。
    非连通图上的"平均最短路径"在数学上无定义（``inf``），
    因此必须**只取 LCC**；这一点在报告里必须写明，
    否则读者会误以为所有节点都参与了平均。

    只有 1 个节点的 LCC 长度为 0（networkx 约定）。

    Returns
    -------
    dict
        ``path_length_lcc``（边数；LCC<2 节点时为 ``None``）/
        ``n_lcc`` / ``n_components`` / ``lcc_fraction`` /
        ``diameter_lcc`` / ``note``。
    """
    a = np.asarray(adjacency.todense() if hasattr(adjacency, "todense") else adjacency)
    n = a.shape[0]
    if n == 0:
        return {"path_length_lcc": None, "n_lcc": 0, "n_components": 0,
                "lcc_fraction": 0.0, "diameter_lcc": None, "note": "空图"}
    comps = connected_components(a)
    lcc = comps[0]
    out = {
        "n_lcc": int(lcc.size),
        "n_components": int(len(comps)),
        "lcc_fraction": float(lcc.size / n),
        "path_length_lcc": None,
        "diameter_lcc": None,
        "note": "",
    }
    if lcc.size < 2:
        out["note"] = "LCC 只有 1 个节点，路径长度无定义"
        return out

    sub = a[np.ix_(lcc, lcc)]
    # BFS 全源最短路（无权），O(N·M)
    k = lcc.size
    inf = np.inf
    total = 0.0
    diam = 0.0
    for s in range(k):
        dist = np.full(k, inf)
        dist[s] = 0.0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in np.nonzero(sub[u])[0]:
                    if dist[v] == inf:
                        dist[v] = d
                        nxt.append(int(v))
            frontier = nxt
        if np.any(np.isinf(dist)):
            out["note"] = "LCC 内部不连通（邻接矩阵不对称？），路径长度不可信"
            return out
        total += float(dist.sum())
        diam = max(diam, float(dist.max()))

    kk = k
    out["path_length_lcc"] = float(total / (kk * (kk - 1)))
    out["diameter_lcc"] = float(diam)
    out["note"] = f"仅在 LCC（{k}/{n} 个节点）上计算"
    return out


def degree_assortativity(adjacency) -> Dict[str, Any]:
    """度同配系数（Newman's degree assortativity），无量纲，∈[-1, 1]。

    度量的是什么
    ------------
    在所有**边**上（每条边等概率），两端节点**度**的 Pearson 相关系数。
    正 = 高度节点倾向连高度节点（同配）；负 = 倾向连低度节点（异配）；
    0 = 无相关。

    Pearson 相关需要度的方差 > 0；**正则图（所有度相同）上该量无定义**，
    此时返回 ``None`` 而不是 0——0 会被误读成"无同配性"。

    Returns
    -------
    dict
        ``degree_assortativity``（或 ``None``）/ ``degenerate`` / ``note``。
    """
    a = np.asarray(adjacency.todense() if hasattr(adjacency, "todense") else adjacency)
    a = (a != 0).astype(np.float64)
    np.fill_diagonal(a, 0.0)
    deg = a.sum(axis=1)
    iu = np.triu_indices(a.shape[0], k=1)
    sel = a[iu] > 0
    if not np.any(sel):
        return {"degree_assortativity": None, "n_edges": 0, "degenerate": True,
                "note": "没有边，同配系数无定义"}
    x = deg[iu[0][sel]]
    y = deg[iu[1][sel]]
    if np.std(x) == 0 or np.std(y) == 0:
        return {"degree_assortativity": None, "n_edges": int(x.size),
                "degenerate": True,
                "note": "所有端点度相同（正则图），Pearson 相关无定义"}
    r = float(np.corrcoef(x, y)[0, 1])
    return {"degree_assortativity": r, "n_edges": int(x.size),
            "degenerate": False, "note": ""}


def modularity_greedy(
    adjacency, seed: int = 0, resolution: float = 1.0
) -> Dict[str, Any]:
    """贪心模块度社区划分（Clauset-Newman-Moore）及其模块度 Q。

    度量的是什么
    ------------
    模块度 ``Q`` = （社区内部边占比）−（在保持度序列的随机零模型下
    社区内部边占比的期望）。**无量纲**，理论上界 1，
    实际网络里 > 0.3 通常被认为有模块结构。

    **小图警告**：模块度对 N 高度敏感，N < 10 时 Q 的方差极大，
    单张图的 Q 值几乎不可解释；本函数在 N < 10 时给出 ``low_n_warning``。

    降级路径
    --------
    需要 ``networkx.algorithms.community.greedy_modularity_communities``。
    networkx 缺失时返回 ``modularity=None`` 并说明，**不伪造数值**。

    Returns
    -------
    dict
        ``modularity`` / ``n_communities`` / ``community_sizes`` /
        ``community_assignment``（长度 N）/ ``resolution`` / ``seed`` /
        ``low_n_warning``。
    """
    a = np.asarray(adjacency.todense() if hasattr(adjacency, "todense") else adjacency)
    n = a.shape[0]
    try:
        nx = _require_networkx("模块度社区划分")
    except ImportError as exc:
        return {"modularity": None, "n_communities": None, "community_sizes": [],
                "community_assignment": [], "resolution": resolution, "seed": seed,
                "note": str(exc)}
    if n == 0:
        return {"modularity": None, "n_communities": 0, "community_sizes": [],
                "community_assignment": [], "resolution": resolution, "seed": seed,
                "note": "空图"}

    g = nx.from_numpy_array((a != 0).astype(np.float64))
    if g.number_of_edges() == 0:
        return {"modularity": None, "n_communities": n, "community_sizes": [1] * n,
                "community_assignment": list(range(n)), "resolution": resolution,
                "seed": seed, "note": "没有边，每个节点自成一个社区，Q 无定义"}
    coms = nx.algorithms.community.greedy_modularity_communities(
        g, resolution=resolution
    )
    q = nx.algorithms.community.modularity(g, coms, resolution=resolution)
    assign = np.zeros(n, dtype=np.int64)
    for ci, c in enumerate(coms):
        for v in c:
            assign[int(v)] = ci
    return {
        "modularity": float(q),
        "n_communities": int(len(coms)),
        "community_sizes": sorted((len(c) for c in coms), reverse=True),
        "community_assignment": assign.tolist(),
        "resolution": float(resolution),
        "seed": int(seed),
        "low_n_warning": bool(n < 10),
        "note": "N<10，Q 的抽样方差很大，单图 Q 不可单独解释" if n < 10 else "",
    }


# ==========================================================================
# 4. 状态离散化（状态转移熵的输入）
# ==========================================================================

def quantile_state_bins(
    values: Sequence[float], n_bins: int = 3
) -> Dict[str, Any]:
    """把连续量离散成**等频（分位）**状态码。

    分箱规则（必须写死在报告里，否则熵不可比）
    ----------------------------------------
    * 用 ``np.quantile`` 在**有效样本**上取 ``n_bins-1`` 个分位点，
      得到 ``n_bins`` 个等频箱；状态码 0..k-1 按数值**升序**排列。
    * 若有效样本的**不同取值个数** < ``n_bins``，则把箱数降到该个数
      （并在 ``note`` 里说明），保证每个状态都非空。
    * ``NaN`` 样本的状态码为 ``-1``（无效），下游函数会剔除它们
      **并报告剔除了几个**。

    Returns
    -------
    dict
        ``states``（长度 N 的 int 数组，-1 = 无效）/ ``edges``（分位切点）/
        ``n_bins`` / ``n_valid`` / ``n_invalid`` / ``state_counts`` /
        ``rule``（人类可读的规则描述）。
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(v)
    n_valid = int(np.count_nonzero(finite))
    states = np.full(v.size, -1, dtype=np.int64)
    if n_valid == 0:
        return {"states": states, "edges": [], "n_bins": 0, "n_valid": 0,
                "n_invalid": int(v.size), "state_counts": {},
                "rule": "全部样本无效，未做离散化"}

    vv = v[finite]
    uniq = np.unique(vv)
    k = int(min(n_bins, uniq.size))
    if k < 2:
        states[finite] = 0
        return {"states": states, "edges": [], "n_bins": 1, "n_valid": n_valid,
                "n_invalid": int(v.size - n_valid), "state_counts": {0: n_valid},
                "rule": f"有效样本只有 {uniq.size} 个不同取值，退化为单状态（转移熵恒为 0）"}

    qs = np.linspace(0.0, 1.0, k + 1)[1:-1]
    edges = np.unique(np.quantile(vv, qs))
    idx = np.searchsorted(edges, vv, side="right")
    states[finite] = idx.astype(np.int64)
    counts = {int(s): int(c) for s, c in zip(*np.unique(states[finite], return_counts=True))}
    return {
        "states": states,
        "edges": [float(e) for e in edges],
        "n_bins": int(len(edges) + 1),
        "n_valid": n_valid,
        "n_invalid": int(v.size - n_valid),
        "state_counts": counts,
        "rule": (
            f"等频（分位）{len(edges) + 1} 箱，切点 {[round(float(e), 6) for e in edges]}；"
            "状态码 0..k-1 按数值升序"
        ),
    }


# ==========================================================================
# 5. 状态转移熵（"涌现"的核心量）
# ==========================================================================

def state_transition_entropy(
    states: Sequence[int], adjacency, n_states: Optional[int] = None
) -> Dict[str, Any]:
    """**状态转移熵（马尔可夫链熵率）**，单位 **bit / 步**。

    这是本项目里唯一真正配得上"涌现"二字的量：它度量的是
    **局部状态之间的依赖结构**，而不是某个单点特征的分布。

    度量的是什么（精确定义）
    ------------------------
    把每个节点上的离散状态 ``s ∈ {0..k-1}`` 看作马尔可夫链的状态，
    **在图上做随机游走**，转移矩阵

        ``P[i, j] = (Σ_{u: s(u)=i} Σ_{v~u} 1[s(v)=j]) / (Σ_{u: s(u)=i} k_u)``

    即"已知当前节点状态为 i，沿一条随机边走到邻居、其状态为 j"的概率
    （``k_u`` 是节点 u 的度）。这正是一个随机游走在**状态标签**上诱导的链。

    熵率有两种加权方式，本函数**都返回**并明确区分：

    * ``H_rate_stationary``：以该链的**平稳分布 π**（``πP = π``，用
      ``Pᵀ`` 的主特征向量求得）加权的条件熵
      ``H = -Σ_i π_i Σ_j P_ij log2 P_ij``。这是标准的马尔可夫链熵率。
    * ``H_rate_empirical``：以**状态的经验占据率**``n_i / n``加权的条件熵。
      小样本下更稳，但严格说不是平稳链的熵率。

    上界是 ``log2(k)``（k = 实际用到的状态数），因此也返回
    ``H_rate_normalized`` = ``H_rate_stationary / log2(k)``。

    **重要区分**：这里的"转移"是**空间**转移（相邻节点），
    **不是时间转移**。本数据集没有时间轴，任何"随时间演化"的说法都是错的。

    Parameters
    ----------
    states
        长度 N 的整数状态码；``-1`` 表示无效（会被剔除并计数）。
    adjacency
        (N, N) 邻接矩阵（稀疏或稠密），节点顺序与 ``states`` 一致。

    Returns
    -------
    dict
        ``H_rate_stationary`` / ``H_rate_empirical`` / ``H_rate_max_bits`` /
        ``H_rate_normalized`` / ``n_states_used`` / ``n_valid_nodes`` /
        ``transition_matrix`` / ``stationary`` / ``occupancy`` /
        ``degenerate``（有效状态 < 2 时熵恒为 0）/ ``note``。
    """
    s = np.asarray(states, dtype=np.int64).ravel()
    a = np.asarray(adjacency.todense() if hasattr(adjacency, "todense") else adjacency)
    valid = s >= 0
    out: Dict[str, Any] = {
        "H_rate_stationary": 0.0, "H_rate_empirical": 0.0,
        "H_rate_max_bits": 0.0, "H_rate_normalized": 0.0,
        "n_states_used": 0, "n_valid_nodes": int(np.count_nonzero(valid)),
        "transition_matrix": [], "stationary": [], "occupancy": [],
        "degenerate": True, "note": "",
    }
    if not np.any(valid):
        out["note"] = "没有任何有效状态，转移熵无定义"
        return out

    sv = s[valid]
    av = (a[np.ix_(valid, valid)] != 0).astype(np.float64)
    np.fill_diagonal(av, 0.0)
    k = int(n_states if n_states is not None else (sv.max() + 1))
    k = max(k, int(sv.max()) + 1)

    if av.sum() == 0:
        out["n_states_used"] = int(len(np.unique(sv)))
        out["note"] = "有效子图没有任何边，转移矩阵无定义"
        return out

    # 转移计数：从状态 i 出发、沿随机边到达状态 j 的次数
    deg = av.sum(axis=1)                          # k_u
    counts = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        m = sv == i
        if not np.any(m):
            continue
        # 该状态所有节点的邻居状态直方图之和
        nb_idx = np.nonzero(av[m])[1]
        if nb_idx.size:
            counts[i] = np.bincount(sv[nb_idx], minlength=k)
        # 状态 i 的"出发次数" = 这些节点的度之和
        counts[i] /= max(deg[m].sum(), 1e-12)
    # 上面 counts[i] 已按行归一化（除的是同一个标量），即为 P
    p = counts
    row_sums = p.sum(axis=1)
    nz = row_sums > 0
    if not np.any(nz):
        out["note"] = "转移矩阵全零"
        return out

    occ = np.array([np.count_nonzero(sv == i) for i in range(k)], dtype=np.float64)
    occ = occ / occ.sum()
    H_rows = np.array([
        shannon_entropy_bits(p[i]) if nz[i] else 0.0 for i in range(k)
    ])
    H_emp = float(np.sum(occ * H_rows))

    # 平稳分布：Pᵀ 的主特征向量
    try:
        evals, evecs = np.linalg.eig(p.T)
        j = int(np.argmin(np.abs(evals - 1.0)))
        pi = np.real(evecs[:, j])
        pi = np.abs(pi)
        if pi.sum() > 0:
            pi = pi / pi.sum()
        else:
            pi = occ.copy()
    except np.linalg.LinAlgError:
        pi = occ.copy()
        out["note"] = "平稳分布特征分解失败，已退化为经验占据率"
    H_stat = float(np.sum(pi * H_rows))

    k_used = int(np.count_nonzero(np.array([np.count_nonzero(sv == i) for i in range(k)])))
    hmax = math.log2(k_used) if k_used > 1 else 0.0
    out.update({
        "H_rate_stationary": H_stat,
        "H_rate_empirical": H_emp,
        "H_rate_max_bits": float(hmax),
        "H_rate_normalized": float(H_stat / hmax) if hmax > 0 else 0.0,
        "n_states_used": k_used,
        "transition_matrix": [[float(v) for v in row] for row in p],
        "stationary": [float(v) for v in pi],
        "occupancy": [float(v) for v in occ],
        "degenerate": bool(k_used < 2),
        "note": (
            f"{int(np.count_nonzero(~valid))} 个节点因状态无效被剔除；"
            "转移是**空间**相邻关系诱导的，不是时间演化"
            + ("；有效状态 < 2，熵恒为 0（真实退化）" if k_used < 2 else "")
        ),
    })
    return out


def mutual_information_adjacent(
    states_x: Sequence[int],
    states_y: Optional[Sequence[int]] = None,
    adjacency=None,
    n_states: Optional[int] = None,
) -> Dict[str, Any]:
    """**相邻节点**状态之间的互信息，单位 **bit**。

    度量的是什么
    ------------
    以"图上的一条有向邻接关系"为等概率抽样单位，随机取一条邻接对
    ``(u → v)``，联合分布 ``p(X=s(u), Y=s(v))``，
    互信息 ``I(X;Y) = ΣΣ p(x,y) log2 [p(x,y) / (p(x)p(y))]``。

    **这是空间依赖，不是时间依赖。**
    本数据集是单时间点静态图像，没有时间轴；
    ``I > 0`` 只说明"空间上相邻的区域倾向于具有相似/相关的离散状态"，
    **不能**读作"一个细胞器的状态变化驱动了另一个"，
    也不能读作"随时间的信息传递"。任何因果或动力学解释都超出本量的能力范围。

    若给出 ``states_y``，则度量两类状态之间的互信息（如类型 vs 灰度状态）；
    否则 ``y = x``，得到的是同一状态量的空间自信息。

    归一化：``I / min(H(X), H(Y))`` ∈ [0,1]（``min`` 而非 ``sqrt``，
    因为当两个变量支撑集大小差很多时 ``sqrt`` 归一化会系统性偏小）。

    Returns
    -------
    dict
        ``MI_bits`` / ``MI_normalized`` / ``H_x_bits`` / ``H_y_bits`` /
        ``n_pairs``（**互信息的估计强烈依赖样本量，必须报告**）/
        ``n_states_x`` / ``n_states_y`` / ``degenerate`` / ``note``。
    """
    x = np.asarray(states_x, dtype=np.int64).ravel()
    y = x if states_y is None else np.asarray(states_y, dtype=np.int64).ravel()
    if y.size != x.size:
        raise ValueError(f"states_x 与 states_y 长度不一致：{x.size} vs {y.size}")
    if adjacency is None:
        raise ValueError("mutual_information_adjacent 需要 adjacency 才能定义'相邻'")

    a = np.asarray(adjacency.todense() if hasattr(adjacency, "todense") else adjacency)
    a = (a != 0)
    np.fill_diagonal(a, False)
    valid = (x >= 0) & (y >= 0)
    a = a & valid[:, None] & valid[None, :]

    out: Dict[str, Any] = {
        "MI_bits": 0.0, "MI_normalized": 0.0, "H_x_bits": 0.0, "H_y_bits": 0.0,
        "n_pairs": 0, "n_states_x": 0, "n_states_y": 0,
        "degenerate": True, "note": "",
    }
    u, v = np.nonzero(a)
    n_pairs = int(u.size)
    out["n_pairs"] = n_pairs
    if n_pairs == 0:
        out["note"] = "有效邻接对为 0，互信息无定义"
        return out

    xs, ys = x[u], y[v]
    kx = int(n_states if n_states is not None else xs.max() + 1)
    ky = int(ys.max() + 1)
    kx, ky = max(kx, xs.max() + 1), max(ky, ys.max() + 1)

    joint = np.zeros((kx, ky), dtype=np.float64)
    np.add.at(joint, (xs, ys), 1.0)
    joint /= joint.sum()
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for i in range(kx):
        if px[i] <= 0:
            continue
        for j in range(ky):
            if joint[i, j] > 0 and py[j] > 0:
                mi += joint[i, j] * math.log2(joint[i, j] / (px[i] * py[j]))
    hx, hy = shannon_entropy_bits(px), shannon_entropy_bits(py)
    denom = min(hx, hy)
    out.update({
        "MI_bits": float(mi),
        "MI_normalized": float(mi / denom) if denom > 0 else 0.0,
        "H_x_bits": float(hx), "H_y_bits": float(hy),
        "n_states_x": int(np.count_nonzero(px > 0)),
        "n_states_y": int(np.count_nonzero(py > 0)),
        "degenerate": bool(denom == 0),
        "note": (
            "**空间**依赖度量，不是时间/因果度量；"
            f"基于 {n_pairs} 个有向邻接对，样本量小则 MI 严重上偏"
            + ("；某一侧状态无变化，MI 恒为 0" if denom == 0 else "")
        ),
    })
    return out


def sequence_entropy_rate(
    states: Sequence[int], n_states: Optional[int] = None
) -> Dict[str, Any]:
    """按**索引顺序**构成的一阶马尔可夫链的条件熵，单位 **bit/步**。

    !!! 警告：这里的"顺序"是数据表里的行序（``region_id`` 升序），
    **不是时间顺序**。本数据集没有时间轴，因此这个量**没有动力学含义**，
    只作为"若将来接入时间序列，同一套代码可以直接算"的接口保留。
    报告中使用它时必须显式标注为"伪序列"。

    度量的是什么
    ------------
    ``H(X_{t+1} | X_t) = Σ_i (n_i / n) · H(P[i, ·])``，
    其中 ``P[i, j] = count(i → j) / count(i → ·)`` 由**相邻索引**的
    状态转移计数得到；``n_i`` 是状态 i 作为起点的次数。

    Returns
    -------
    dict
        ``H_seq_rate_bits`` / ``H_seq_max_bits`` = ``log2(k)`` /
        ``n_transitions`` / ``n_states_used`` / ``degenerate`` / ``note``。
    """
    s = np.asarray(states, dtype=np.int64).ravel()
    s = s[s >= 0]
    out = {"H_seq_rate_bits": 0.0, "H_seq_max_bits": 0.0, "n_transitions": 0,
           "n_states_used": 0, "degenerate": True,
           "note": "伪序列（行序），无时间含义"}
    if s.size < 2:
        out["note"] += "；有效样本 < 2，无转移"
        return out
    k = int(n_states if n_states is not None else s.max() + 1)
    k = max(k, int(s.max()) + 1)
    counts = np.zeros((k, k), dtype=np.float64)
    np.add.at(counts, (s[:-1], s[1:]), 1.0)
    rows = counts.sum(axis=1)
    used = rows > 0
    if not np.any(used):
        out["note"] += "；无转移"
        return out
    h_rows = np.array([shannon_entropy_bits(counts[i]) if used[i] else 0.0
                       for i in range(k)])
    w = rows / rows.sum()
    h = float(np.sum(w * h_rows))
    k_used = int(np.count_nonzero(np.array([np.count_nonzero(s == i) for i in range(k)])))
    hmax = math.log2(k_used) if k_used > 1 else 0.0
    out.update({"H_seq_rate_bits": h, "H_seq_max_bits": float(hmax),
                "n_transitions": int(s.size - 1), "n_states_used": k_used,
                "degenerate": bool(k_used < 2)})
    return out


# ==========================================================================
# 6. 鲁棒性
# ==========================================================================

def percolation_threshold(
    adjacency,
    mode: str = "node_random",
    seed: int = 0,
    n_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """渗流阈值代理量：最大连通分量跌破 50% 时已移除的比例。

    度量的是什么
    ------------
    按给定顺序逐个移除节点（或边），每次移除后记录
    ``LCC 节点数 / 原始 N``；返回**第一次**该比例 < 0.5 时的
    **移除比例 f ∈ (0, 1]**（无量纲）。

    ``mode``
    --------
    ``'node_random'``
        随机顺序移除节点（``seed`` 固定，可复现）。f 大 = 网络对随机故障鲁棒。
    ``'node_targeted'``
        按**初始度**从大到小移除（静态靶向攻击，不重算度）。
        这是最常用的"故意攻击"代理，但它偏向于高估攻击效果，
        因为真实攻击往往动态重算度——这里如实标注为**静态**靶向。
    ``'edge_random'``
        随机顺序移除边；LCC 仅在节点上定义，因此移除边后仍以节点 LCC 计算。

    未跌破 50% 时返回 ``fraction = 1.0`` 且 ``reached = False``
    （含义是"在移除全部之前网络仍保持连通"），**不是**"阈值 = 1.0"。

    Returns
    -------
    dict
        ``fraction`` / ``reached`` / ``mode`` / ``seed`` / ``n_removed`` /
        ``lcc_curve``（每步的 LCC/N，长度 ``n_steps+1``）。
    """
    a = np.asarray(adjacency.todense() if hasattr(adjacency, "todense") else adjacency)
    a = (a != 0)
    np.fill_diagonal(a, False)
    n = a.shape[0]
    out = {"fraction": 1.0, "reached": False, "mode": mode, "seed": int(seed),
           "n_removed": 0, "lcc_curve": []}
    if n == 0:
        return out

    rng = np.random.default_rng(seed)
    a = a.copy()
    curve = [1.0]

    if mode in ("node_random", "node_targeted"):
        deg = a.sum(axis=1)
        order = (rng.permutation(n) if mode == "node_random"
                 else np.argsort(-deg, kind="stable"))
        steps = int(n_steps or n)
        removed = np.zeros(n, dtype=bool)
        frac = 1.0
        for t in range(min(steps, n)):
            removed[order[t]] = True
            sub = a.copy()
            sub[removed, :] = False
            sub[:, removed] = False
            comps = connected_components(sub)
            lcc = comps[0].size if comps else 0
            frac = lcc / n
            curve.append(float(frac))
            if frac < 0.5:
                out.update({
                    "fraction": float((t + 1) / n),
                    "reached": True,
                    "n_removed": int(t + 1),
                    "lcc_curve": curve,
                })
                return out
        out.update({"fraction": 1.0, "reached": False, "n_removed": int(min(steps, n)),
                    "lcc_curve": curve})
        return out

    if mode == "edge_random":
        iu = np.triu_indices(n, k=1)
        edges = np.column_stack([iu[0][a[iu]], iu[1][a[iu]]])
        m = edges.shape[0]
        if m == 0:
            return out
        perm = rng.permutation(m)
        a2 = a.copy()
        for t in range(m):
            i, j = edges[perm[t]]
            a2[i, j] = a2[j, i] = False
            comps = connected_components(a2)
            frac = (comps[0].size / n) if comps else 0.0
            curve.append(float(frac))
            if frac < 0.5:
                out.update({"fraction": float((t + 1) / max(m, 1)), "reached": True,
                            "n_removed": int(t + 1), "lcc_curve": curve})
                return out
        out.update({"lcc_curve": curve})
        return out

    raise ValueError(f"未知 mode {mode!r}；可选 node_random/node_targeted/edge_random")


def mean_betweenness(
    adjacency, max_nodes: int = MAX_NODES_FOR_EXPENSIVE_METRICS
) -> Dict[str, Any]:
    """平均介数中心性（**只使用无权最短路**）。

    度量的是什么
    ------------
    节点 i 的介数 = 所有节点对 ``(s,t)`` 的最短路中经过 i 的比例之和
    （除以归一化因子，networkx 默认对无向图除以 ``(N-1)(N-2)/2``）。
    平均介数 = 对所有 N 个节点取平均，**无量纲** ∈ [0, 1]。
    它刻画"平均而言一个区域在多少条信息通路上"。

    **代价与跳过规则**
    ------------------
    networkx 的实现是 ``O(N·M)``；本函数在 ``N > max_nodes``
    （默认 300）时**直接跳过**并返回 ``None``，附上原因。
    这是有意的：伪造一个抽样估计出来会让报告失去可信度。

    Returns
    -------
    dict
        ``mean_betweenness``（或 ``None``）/ ``max_betweenness`` /
        ``skipped`` / ``n_nodes`` / ``reason``。
    """
    a = np.asarray(adjacency.todense() if hasattr(adjacency, "todense") else adjacency)
    n = a.shape[0]
    if n > max_nodes:
        return {"mean_betweenness": None, "max_betweenness": None, "skipped": True,
                "n_nodes": int(n),
                "reason": f"N={n} > max_nodes={max_nodes}；介数是 O(N·M)，按预设规则跳过"}
    try:
        nx = _require_networkx("介数中心性")
    except ImportError as exc:
        return {"mean_betweenness": None, "max_betweenness": None, "skipped": True,
                "n_nodes": int(n), "reason": str(exc)}
    if n == 0:
        return {"mean_betweenness": None, "max_betweenness": None, "skipped": True,
                "n_nodes": 0, "reason": "空图"}
    g = nx.from_numpy_array((a != 0).astype(np.float64))
    bc = nx.betweenness_centrality(g, normalized=True)
    vals = np.asarray(list(bc.values()), dtype=np.float64)
    return {"mean_betweenness": float(vals.mean()),
            "max_betweenness": float(vals.max()),
            "skipped": False, "n_nodes": int(n), "reason": ""}


# ==========================================================================
# 7. 汇总入口
# ==========================================================================

#: ``compute_all_metrics`` 接受的状态来源列名（来自区域特征表）
STATE_SOURCES = (
    "gray_mean", "gray_std", "cv_gray", "area", "solidity",
    "eccentricity", "roughness", "label_ratio",
)


def node_state_values(table, source: str) -> np.ndarray:
    """从 ``RegionTable`` 取某个特征的逐节点数值 (N,)。

    未知列名抛 ``ValueError`` 并列出可用列。
    """
    if source not in STATE_SOURCES:
        raise ValueError(f"未知状态来源 {source!r}；可选 {list(STATE_SOURCES)}")
    vals = []
    for r in table.regions:
        if source == "area":
            v = r.area
        else:
            v = r.raw.get(source, np.nan)
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(float("nan"))
    return np.asarray(vals, dtype=np.float64)


def compute_all_metrics(
    net,
    table=None,
    *,
    state_source: str = "gray_mean",
    n_state_bins: int = 3,
    weight_bins: int = 4,
    weight_binning: str = "quantile",
    percolation_seed: int = 0,
    max_betweenness_nodes: int = MAX_NODES_FOR_EXPENSIVE_METRICS,
) -> Dict[str, Any]:
    """对一个 ``OrganelleNetwork`` 计算全套指标。

    Parameters
    ----------
    net
        ``tools.network.build.OrganelleNetwork``。
    table
        对应的 ``RegionTable``；用于取状态来源列。为 ``None`` 时跳过
        所有状态类指标（并在 ``skipped`` 里说明）。
    state_source
        ``STATE_SOURCES`` 之一，默认 ``gray_mean``。
    n_state_bins
        状态离散化的等频箱数（默认 3）。
    weight_bins, weight_binning
        边权熵的分箱（默认 4 箱等频）。
    percolation_seed
        鲁棒性随机移除的种子（固定以保证可复现）。
    max_betweenness_nodes
        介数中心性的节点数上限，超过则跳过。

    Returns
    -------
    dict
        分层组织的指标字典：``graph`` / ``degree`` / ``weight`` /
        ``topology`` / ``state`` / ``robustness`` / ``skipped`` / ``notes``。
        **每个值都保持原始定义**，不做任何跨样本归一化（归一化在汇总层做）。
    """
    adj = net.adjacency
    deg = net.degrees()
    n = net.n_nodes

    res: Dict[str, Any] = {
        "image": net.image,
        "method": net.method,
        "graph": {
            "n_nodes": int(n),
            "n_edges": int(net.n_edges),
            "density": float(net.density()),
            "coordinate_unit": net.coordinate_unit,
            "node_type_source": net.node_type_source,
            "edge_weight_stats": net.edge_weight_stats(),
        },
        "skipped": [],
        "notes": list(net.warnings),
    }

    # --- 度分布熵 --------------------------------------------------------
    res["degree"] = degree_distribution_entropy(deg, n_nodes=n)

    # --- 边权熵 ----------------------------------------------------------
    we = edge_weight_entropy(net.edge_weights, n_bins=weight_bins,
                             binning=weight_binning)
    if net.n_edges == 0:
        res["skipped"].append("edge_weight_entropy: 无边缘")
    res["weight"] = we

    # --- 拓扑 ------------------------------------------------------------
    if n >= 2 and net.n_edges > 0:
        res["topology"] = {
            "clustering": clustering_coefficients(adj),
            "path_length": characteristic_path_length(adj),
            "assortativity": degree_assortativity(adj),
            "modularity": modularity_greedy(adj, seed=percolation_seed),
            "betweenness": mean_betweenness(adj, max_nodes=max_betweenness_nodes),
        }
        if res["topology"]["betweenness"]["skipped"]:
            res["skipped"].append(
                "mean_betweenness: " + res["topology"]["betweenness"]["reason"])
    else:
        res["skipped"].append("拓扑指标: 节点数或边数为 0，全部无定义")
        res["topology"] = None

    # --- 状态转移熵 / 互信息 --------------------------------------------
    if table is None:
        res["state"] = None
        res["skipped"].append("状态类指标: 未提供 RegionTable")
    else:
        values = node_state_values(table, state_source)
        disc = quantile_state_bins(values, n_bins=n_state_bins)
        rate = state_transition_entropy(disc["states"], adj,
                                        n_states=disc["n_bins"])
        mi = (mutual_information_adjacent(disc["states"], None, adj,
                                          n_states=disc["n_bins"])
              if n >= 2 and net.n_edges > 0 else
              {"MI_bits": None, "MI_normalized": None, "n_pairs": 0,
               "note": "无边或节点不足，互信息无定义"})
        types, _ = table.node_types(mode="annotation")
        # 类型编码必须是**确定性**的：Python 的 str hash 每个进程都加盐，
        # 直接用 hash() 会让同一份数据两次运行得到不同的编码。
        # 这里改为按字典序排列后的下标。
        uniq_types = sorted(set(types))
        tmap = {t: i for i, t in enumerate(uniq_types)}
        type_codes = np.asarray([tmap[t] for t in types], dtype=np.int64)
        res["state"] = {
            "state_source": state_source,
            "discretization": {
                "rule": disc["rule"], "edges": disc["edges"],
                "n_bins": disc["n_bins"], "n_valid": disc["n_valid"],
                "n_invalid": disc["n_invalid"], "state_counts": disc["state_counts"],
            },
            "transition_entropy": rate,
            "mutual_information_adjacent": mi,
            "sequence_entropy_rate": sequence_entropy_rate(
                disc["states"], n_states=disc["n_bins"]),
            "node_type_codes": type_codes.tolist(),
            "node_type_vocabulary": uniq_types,
        }

    # --- 鲁棒性 ----------------------------------------------------------
    if n >= 2 and net.n_edges > 0:
        res["robustness"] = {
            "percolation_node_random": percolation_threshold(
                adj, mode="node_random", seed=percolation_seed),
            "percolation_node_targeted": percolation_threshold(
                adj, mode="node_targeted", seed=percolation_seed),
            "percolation_edge_random": percolation_threshold(
                adj, mode="edge_random", seed=percolation_seed),
        }
    else:
        res["robustness"] = None
        res["skipped"].append("鲁棒性指标: 无边缘，渗流曲线全为 1，无信息量")

    return res
