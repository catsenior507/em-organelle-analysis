#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_analysis.py —— 逐图重建细胞器互作网络并汇总涌现指标
========================================================

用法
----
::

    python tools/network/run_analysis.py \
        --annotations data/annotations_v2 \
        --out data/network_analysis \
        --method boundary_contact --min-contact-px 2

产物（全部写在 ``--out`` 下）
-----------------------------
``per_image.json``
    每张图一条记录：图参数（N、M、密度、警告）+ 全部指标 + 失败原因。
``summary.json``
    跨图汇总。**每一个汇总值都带 N**（参与汇总的图数），
    另附参与/跳过清单、标定分布、指标退化检测。
``metrics.csv``
    一张图一行的扁平表，方便直接丢进 pandas/SPSS。
``REPORT.md``
    人类可读报告：方法、参数、密度、逐图数字、
    **"局限 / 这些数字说明了什么、不能说明什么"** 一节。
``skipped.json``
    被跳过的图与逐条原因（不静默丢弃任何样本）。
``contact_geometry.json``
    仅 ``boundary_contact``：全数据集的区域间最小间隙分布（阈值取值的依据）。
``networks/<image>.json|.npz``
    仅在 ``--save-networks`` 时写出（``OrganelleNetwork.save``）。

设计原则
--------
* **不静默丢弃**：任何被跳过的图都会带着原因进入 ``skipped.json`` 与报告。
* **不伪造**：指标无定义时写 ``null``，并说明原因，绝不填 0 充当结果。
* **可复现**：所有随机性都来自 ``--seed`` / ``--percolation-seed``。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 允许 `python tools/network/run_analysis.py` 直接运行
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.network import build as netbuild          # noqa: E402
from tools.network import dynamics, metrics          # noqa: E402

__version__ = "1.0.0"


# ==========================================================================
# 参数
# ==========================================================================

def build_argparser() -> argparse.ArgumentParser:
    """构造命令行参数解析器（``--help`` 可查看全部选项）。"""
    p = argparse.ArgumentParser(
        prog="run_analysis.py",
        description=(
            "从 annotations_v2 的区域表真正重建细胞器互作网络，"
            "并计算度分布熵 / 边权熵 / 聚类 / 路径长度 / 模块度 / "
            "状态转移熵 / 相邻互信息 / 鲁棒性等涌现指标。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--annotations", default=os.path.join(_ROOT, "data", "annotations_v2"),
                   help="annotations_v2 目录（内含 regions/*.json）")
    p.add_argument("--out", default=os.path.join(_ROOT, "data", "network_analysis"),
                   help="输出目录")

    g = p.add_argument_group("网络重建")
    g.add_argument("--method", default=netbuild.DEFAULT_METHOD,
                   help="邻接判据，可用逗号分隔多个："
                        + ",".join(netbuild.METHODS))
    g.add_argument("--min-contact-px", "--contact-tol-px", dest="min_contact_px",
                   type=float, default=netbuild.DEFAULT_CONTACT_TOL_PX,
                   help="boundary_contact 的接触阈值（像素）；"
                        "两点距离 ≤ 该值即判为接触。**注意它是上界**，"
                        "名字沿用项目约定")
    g.add_argument("--contact-tol-percentile", type=float, default=None,
                   help="若给出，则把接触阈值改为全数据集区域间最小间隙分布的"
                        "该分位数（如 5 表示第 5 百分位），由数据而非拍脑袋决定；"
                        "会覆盖 --min-contact-px")
    g.add_argument("--k", "--knn-k", dest="k", type=int, default=6,
                   help="knn 的 k（别名 --knn-k）")
    g.add_argument("--max-dist-px", type=float, default=None,
                   help="knn 的距离截断（像素）")
    g.add_argument("--radius-px", type=float, default=None,
                   help="radius 的固定阈值（像素）；不给则由最近邻距离中位数导出")
    g.add_argument("--radius-um", type=float, default=None,
                   help="radius 的固定阈值（微米），会按每张图自己的 um_per_px "
                        "换算到网络坐标单位；无标定的图会被跳过。"
                        "给出时覆盖 --radius-px")
    g.add_argument("--radius-factor", type=float, default=1.5,
                   help="radius 自动导出时的倍数：median_nn × factor")
    g.add_argument("--distance-unit", choices=["auto", "um", "px"], default="auto",
                   help="网络坐标单位。auto = 有标定用 µm。"
                        "注意 Delaunay/kNN/Gabriel 对全局尺度不变，该选项只影响 "
                        "radius 阈值与绝对距离的报告；本数据集不同图标定相差可达 5 倍，"
                        "跨图比较绝对距离时建议用 px")
    g.add_argument("--node-types", choices=["annotation", "morphological"],
                   default="annotation",
                   help="节点类型来源；annotation = 只用细胞器类型/OCR 文本（缺失记 "
                        "unlabeled，诚实模式）；morphological = 缺失时用形状规则派生")

    g2 = p.add_argument_group("指标")
    g2.add_argument("--state-source", choices=list(metrics.STATE_SOURCES),
                    default="gray_mean", help="状态离散化所用的区域特征列")
    g2.add_argument("--n-state-bins", type=int, default=3,
                    help="状态离散化的等频箱数")
    g2.add_argument("--weight-bins", type=int, default=4, help="边权熵的分箱数")
    g2.add_argument("--weight-binning", choices=["quantile", "equal_width"],
                    default="quantile", help="边权熵的分箱规则")
    g2.add_argument("--max-betweenness-nodes", type=int,
                    default=metrics.MAX_NODES_FOR_EXPENSIVE_METRICS,
                    help="介数中心性的节点数上限，超过则跳过（O(N·M)）")
    g2.add_argument("--null-reps", type=int, default=20,
                    help="空间零模型重复次数（同 N 的均匀随机点过程）；"
                         "0 = 关闭。boundary_contact 不适用（依赖形状而非质心）")

    g3 = p.add_argument_group("李亚普诺夫（默认关闭，因为本数据集没有时间轴）")
    g3.add_argument("--lyapunov-source", choices=["none", "node-order", "image-order"],
                    default="none",
                    help="none = 不计算（默认，正确做法）。"
                         "node-order = 把某图的逐节点特征按 region_id 顺序当伪序列；"
                         "image-order = 把逐图的某个汇总值按图像编号顺序当伪序列。"
                         "两者都**不是时间序列**，结果会标记 pseudo_sequence=True")
    g3.add_argument("--lyapunov-series", default=None,
                    help="node-order 时使用的特征列（默认同 state-source）；"
                         "image-order 时可用 CSV 里的扁平列名，如 density / mean_betweenness")
    g3.add_argument("--lyapunov-min-points", type=int, default=30,
                    help="少于该样本数直接判定不可靠")
    g3.add_argument("--lyapunov-min-r2", type=float, default=0.90,
                    help="标度区拟合 R² 低于该值判定不可靠")

    g4 = p.add_argument_group("运行控制")
    g4.add_argument("--min-regions", type=int, default=2,
                    help="少于该区域数的图直接跳过（记入 skipped.json）")
    g4.add_argument("--seed", type=int, default=0, help="全局随机种子")
    g4.add_argument("--percolation-seed", type=int, default=0,
                    help="渗流/模块度的随机种子")
    g4.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    g4.add_argument("--save-networks", action="store_true",
                    help="把每张图的 OrganelleNetwork 写到 <out>/networks/")
    g4.add_argument("--quiet", action="store_true")
    return p


# ==========================================================================
# 扁平化与聚合
# ==========================================================================

def _g(d: Optional[Dict[str, Any]], *path, default=None):
    """安全地按路径取嵌套字典的值。"""
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


#: CSV/summary 中的扁平指标列：(列名, 取值路径)
FLAT_FIELDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("n_nodes", ("graph", "n_nodes")),
    ("n_edges", ("graph", "n_edges")),
    ("density", ("graph", "density")),
    ("H_deg_bits", ("degree", "H_deg_bits")),
    ("H_deg_max_bits", ("degree", "H_deg_max_bits")),
    ("H_deg_normalized", ("degree", "H_deg_normalized")),
    ("n_degree_values", ("degree", "n_degree_values")),
    ("H_weight_bits", ("weight", "H_weight_bits")),
    ("H_weight_max_bits", ("weight", "H_weight_max_bits")),
    ("H_weight_normalized", ("weight", "H_weight_normalized")),
    ("edge_weight_median", ("graph", "edge_weight_stats", "median")),
    ("clustering_global", ("topology", "clustering", "clustering_global")),
    ("clustering_mean_local", ("topology", "clustering", "clustering_mean_local")),
    ("n_triangles", ("topology", "clustering", "n_triangles")),
    ("path_length_lcc", ("topology", "path_length", "path_length_lcc")),
    ("n_lcc", ("topology", "path_length", "n_lcc")),
    ("n_components", ("topology", "path_length", "n_components")),
    ("diameter_lcc", ("topology", "path_length", "diameter_lcc")),
    ("degree_assortativity", ("topology", "assortativity", "degree_assortativity")),
    ("modularity", ("topology", "modularity", "modularity")),
    ("n_communities", ("topology", "modularity", "n_communities")),
    ("mean_betweenness", ("topology", "betweenness", "mean_betweenness")),
    ("H_rate_stationary", ("state", "transition_entropy", "H_rate_stationary")),
    ("H_rate_empirical", ("state", "transition_entropy", "H_rate_empirical")),
    ("H_rate_max_bits", ("state", "transition_entropy", "H_rate_max_bits")),
    ("H_rate_normalized", ("state", "transition_entropy", "H_rate_normalized")),
    ("n_states_used", ("state", "transition_entropy", "n_states_used")),
    ("MI_bits", ("state", "mutual_information_adjacent", "MI_bits")),
    ("MI_normalized", ("state", "mutual_information_adjacent", "MI_normalized")),
    ("MI_n_pairs", ("state", "mutual_information_adjacent", "n_pairs")),
    ("H_seq_rate_bits", ("state", "sequence_entropy_rate", "H_seq_rate_bits")),
    ("percolation_node_random", ("robustness", "percolation_node_random", "fraction")),
    ("percolation_node_targeted", ("robustness", "percolation_node_targeted", "fraction")),
    ("percolation_edge_random", ("robustness", "percolation_edge_random", "fraction")),
)


def flatten_image_row(image: str, net, m: Dict[str, Any],
                      params: Dict[str, Any]) -> Dict[str, Any]:
    """把一张图的结果摊平成一维字典（CSV 行 / 汇总输入）。"""
    row: Dict[str, Any] = {"image": image, "method": net.method}
    row["um_per_px"] = params.get("um_per_px")
    row["coordinate_unit"] = net.coordinate_unit
    row["node_type_source"] = net.node_type_source
    for col, path in FLAT_FIELDS:
        v = _g(m, *path)
        if isinstance(v, (np.floating, np.integer)):
            v = float(v)
        row[col] = v
    row["n_warnings"] = len(m.get("notes", []))
    row["n_skipped_metrics"] = len(m.get("skipped", []))
    return row


def _finite(vals: Sequence[Any]) -> np.ndarray:
    """取出有限数值（None/NaN 直接剔除）。"""
    out = []
    for v in vals:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out.append(f)
    return np.asarray(out, dtype=np.float64)


#: 这些是诊断计数器/元数据，不是指标；不参与跨图汇总的"指标"表
#: （否则它们会淹没在退化列表里，看起来像发现）。
NON_METRIC_KEYS = frozenset({
    "um_per_px", "coordinate_unit", "node_type_source",
    "n_warnings", "n_skipped_metrics",
})


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """跨图汇总：**每个指标都带 N**。

    对每个指标同时给出 ``n``（有定义的图数）、``n_missing``、
    ``mean`` / ``median`` / ``std`` / ``min`` / ``max`` / ``iqr``，
    以及两个退化标志：

    * ``degenerate``——所有取值**完全相同**；
    * ``near_degenerate``——取值最多的那个值占了 **≥ 80%** 的图
      （``mode_fraction``）。例如 kNN 在 N≈5 而 k=6 时几乎每张图都是完全图，
      密度恒为 1、度分布熵归一化恒为 1、聚类系数恒为 1——
      这些量看着"很稳定"，实际上是**没有区分度**，必须如实标注。
    """
    n_images = len(rows)
    out: Dict[str, Any] = {"n_images": n_images, "metrics": {}}
    if not rows:
        return out
    keys = [k for k in rows[0] if k not in ("image", "method")
            and k not in NON_METRIC_KEYS]
    for k in keys:
        vals = [r.get(k) for r in rows]
        arr = _finite(vals)
        entry: Dict[str, Any] = {
            "n": int(arr.size),
            "n_missing": int(n_images - arr.size),
        }
        if arr.size:
            # 众数占比：用四舍五入到 6 位有效数字后的取值计数，避免浮点噪声
            rounded = np.round(arr, 9)
            _, counts = np.unique(rounded, return_counts=True)
            mode_fraction = float(counts.max() / arr.size)
            entry.update({
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                "mode_fraction": mode_fraction,
                "degenerate": bool(arr.size > 1 and np.allclose(arr, arr[0])),
                "near_degenerate": bool(arr.size >= 5 and mode_fraction >= 0.80),
            })
        out["metrics"][k] = entry
    return out


# ==========================================================================
# 零模型
# ==========================================================================

def spatial_null_metrics(
    table: "netbuild.RegionTable",
    method: str,
    params: Dict[str, Any],
    reps: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """空间零模型：同 N 的**均匀随机点过程**在同样大小的视场里建图。

    为什么需要它
    ------------
    Delaunay 图的平均度恒为 ~3（与点位无关），密度恒为 ~0.42；
    这类"结构性常数"如果不给零模型，很容易被误读成生物学发现。
    零模型给出"若这些区域的位置完全随机，同样的建图规则会得到什么"，
    于是 观测值 − 零模型 = 空间聚集/规则性的贡献。

    **不适用于 ``boundary_contact``**：它依赖区域的形状与真实间隙，
    光靠质心点过程无法复现，因此返回 ``None`` 并说明。

    Returns
    -------
    dict or None
        ``density`` / ``H_deg_normalized`` / ``clustering_mean_local`` /
        ``path_length_lcc`` 的均值与标准差，各带 ``n``。
    """
    if method == "boundary_contact":
        return None
    if reps <= 0 or table.n < 2:
        return None

    h, w = table.image_shape
    if table.um_per_px is not None and params.get("coordinate_unit") == "um":
        k = float(table.um_per_px)
        extent = np.array([w * k, h * k], dtype=np.float64)
    else:
        extent = np.array([w, h], dtype=np.float64)

    rng = np.random.default_rng(seed)
    acc: Dict[str, List[float]] = {"density": [], "H_deg_normalized": [],
                                   "clustering_mean_local": [], "path_length_lcc": []}
    radii_used: List[float] = []
    for _ in range(int(reps)):
        pts = rng.random((table.n, 2)) * extent
        try:
            if method == "delaunay":
                edges, _ = netbuild.build_delaunay(pts)
            elif method == "knn":
                edges, _ = netbuild.build_knn(pts, k=int(params.get("k", 6)),
                                              max_dist=params.get("max_dist"))
            elif method == "gabriel":
                edges, _ = netbuild.build_gabriel(pts)
            elif method == "radius":
                # 关键：若观测网络的半径是由**观测点**的最近邻距离导出的，
                # 那么零模型也必须用**它自己的点**重新导出半径，
                # 否则等于把观测点的聚集程度偷偷塞进了零模型，
                # 观测−零模型的差值会被系统性夸大或缩小。
                r = None
                rule = str(params.get("radius_rule", ""))
                if rule.startswith("自动导出"):
                    nn = netbuild._nn_distances(pts)
                    if nn.size:
                        r = float(np.median(nn)) * float(params.get("radius_factor", 1.5))
                else:
                    r = params.get("radius_used")
                if r is None or not np.isfinite(r):
                    return None
                edges, _ = netbuild.build_radius(pts, float(r))
                radii_used.append(float(r))
            else:
                return None
        except Exception:  # noqa: BLE001 - 零模型失败不应该毁掉主流程
            continue

        adj = netbuild._adjacency_from_edges(edges, table.n)
        deg = np.zeros(table.n, dtype=np.int64)
        for i, j in edges:
            deg[i] += 1
            deg[j] += 1
        dens = (2.0 * len(edges) / (table.n * (table.n - 1))) if table.n > 1 else 0.0
        acc["density"].append(float(dens))
        acc["H_deg_normalized"].append(
            float(metrics.degree_distribution_entropy(deg, table.n)["H_deg_normalized"]))
        cl = metrics.clustering_coefficients(adj)
        acc["clustering_mean_local"].append(float(cl["clustering_mean_local"]))
        pl = metrics.characteristic_path_length(adj)
        if pl["path_length_lcc"] is not None:
            acc["path_length_lcc"].append(float(pl["path_length_lcc"]))

    out: Dict[str, Any] = {}
    for k, v in acc.items():
        arr = np.asarray(v, dtype=np.float64)
        out[k] = {
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else None,
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        }
    if radii_used:
        out["radius_used"] = {
            "n": len(radii_used),
            "mean": float(np.mean(radii_used)),
            "std": (float(np.std(radii_used, ddof=1)) if len(radii_used) > 1 else 0.0),
        }
    return out


# ==========================================================================
# 报告
# ==========================================================================

LIMITATIONS_MD = """\
## 局限 / 这些数字说明了什么、不能说明什么

**1. 建出来的是"空间邻近网络"，不是"互作网络"。**
活细胞里的互作至少包含两类事件：(a) 膜接触位点（MCS，两膜间距 10–30 nm）；
(b) 时间上的共定位/共运动。本数据集是**单时间点静态图像**，
没有时间轴，(b) 无法测量（见第 4 条）。对 (a)，我们能拿到的只是
**分割轮廓之间的像素间隙**，它同时受真实间隙、手绘轮廓误差、
轮廓简化误差、分割遗漏共同影响。因此每条边的正确读法是
"这两个区域在图像上靠得很近"，而**不是**"这两个细胞器发生了互作"。

**2. 没有任何分子/生物学证据支撑边的存在。**
`node_types` 全部来自标注文本；若 OCR 未运行（本数据集即是如此），
所有节点类型都是 `unlabeled`。因此**不存在**"线粒体–内质网接触"这类结论，
本报告也不会给出任何按细胞器类型分组的互作结论。

**3. 绝对距离的跨图可比性受标定限制。**
报告中列出了每张图的 `um_per_px` 分布。若该分布不唯一，
不同图之间的 µm 距离**不可直接比较**（同一像素间隙在不同图里对应不同的物理长度）。
Delaunay / kNN / Gabriel 三种判据对全局尺度不变，因此它们的**拓扑**指标
不受此影响；`radius` 与 `boundary_contact` 的阈值、以及一切以 µm
报告的绝对距离会受影响。

**4. 状态转移熵与相邻互信息是空间量，不是时间量。**
二者都在"图上相邻节点"这一关系上计算。它们衡量的是
**空间上相邻的区域是否倾向具有相似的离散状态**。
本数据集没有时间维度，所以这些数字**不能**解释为
"状态随时间的转移"、"信息随时间的传播"或任何因果过程。
若报告里出现"动力学""演化""混沌"等措辞来描述这些量，那是错的。

**5. 本报告不含李亚普诺夫指数。**
李亚普诺夫指数需要一个**时间序列**做相空间重构。本数据集没有时间序列，
因此默认 `--lyapunov-source none`，报告里也不会给出任何 λ 数值。
`tools/network/dynamics.py` 提供了正确的 Rosenstein 实现；
若用"区域编号顺序"等伪序列强行调用，结果会被标记为不可靠
（样本数远低于 30，直接拒绝），这**正是期望行为**。
项目中旧代码里的 `infer_lyapunov_from_powerlaw` 是硬编码查表，
已被显式移除并加了导入护栏。

**6. 样本量极小，很多指标在单图层面不可解释。**
每张图只有个位数到二十几个节点。模块度、路径长度、同配系数、
渗流阈值在 N < 10 时抽样方差极大；报告里对每张图都标了 N。
凡跨图汇总处都给了 N，**没有 N 的平均值不要引用**。

**7. 阈值是选择出来的，会改变结论。**
接触阈值、k、半径都会改变图的密度进而改变所有拓扑指标。
本报告给出了阈值/方法敏感性表；任何单一阈值下的数字都必须
连同该表一起读。若某个结论只在某一个阈值下成立，它就不是结论。

**8. 区域质量过滤。**
提取器给出的 `quality_flag` 记录了碎片、未标注、形状异常的区域。
本流程默认**全部保留**并在 CSV 中可追溯（`--min-regions` 只用于
区域太少无法建网的图）。若改成只保留 `quality_flag == 'ok'` 的区域，
指标会变——这属于分析者自由度，必须显式声明。
"""


def _fmt(v: Any, nd: int = 4) -> str:
    """报告里的数字格式化：None → ``null``（不伪装成 0）。"""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "null"
    if abs(f) >= 1000 or (abs(f) < 1e-3 and f != 0):
        return f"{f:.3g}"
    return f"{f:.{nd}f}"


def write_report(
    path: str,
    args: argparse.Namespace,
    methods: Sequence[str],
    per_image: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    skipped: Sequence[Dict[str, str]],
    sweep: Optional[List[Dict[str, Any]]],
    calibration: Dict[str, Any],
    lyapunov: Optional[Dict[str, Any]],
    null_summary: Optional[Dict[str, Any]],
    elapsed: float,
) -> None:
    """生成 ``REPORT.md``。"""
    L: List[str] = []
    A = L.append

    A("# 细胞器互作网络重建与涌现特征分析报告\n")
    A(f"生成脚本：`tools/network/run_analysis.py` v{__version__}　|　"
      f"总耗时 {elapsed:.1f} s　|　随机种子 seed={args.seed}，"
      f"percolation_seed={args.percolation_seed}\n")
    A("> 本报告中的每一个数字都由本目录下的 `per_image.json` / `metrics.csv` 可追溯。")
    A("> **报告末尾的『局限』一节是必读部分**——它规定了这些数字能被解释到什么程度。\n")

    # ---- 1 方法 ----
    A("## 1. 网络是怎么建出来的\n")
    A(f"* 输入：`{os.path.abspath(args.annotations)}`")
    A(f"* 邻接判据：`{', '.join(methods)}`")
    A(f"* 坐标单位：`{args.distance_unit}`（auto = 有标定则 µm，否则像素）")
    A(f"* 节点类型来源：`{args.node_types}`")
    if "boundary_contact" in methods:
        tol = summary.get("contact_tolerance_px", {}).get("used")
        A(f"* **接触阈值**：{_fmt(tol, 3)} px"
          + ("（由数据分位数导出）" if args.contact_tol_percentile is not None
             else "（用户指定）"))
        A("  * 判定规则：两区域**轮廓（边界像素）之间的最小欧氏像素距离 ≤ 阈值** → 一条边；"
          "权重 = 接触长度（接触区内边界像素个数 × µm/px，或像素数）")
    if "knn" in methods:
        A(f"* kNN：k = {args.k}" + (f"，距离截断 {_fmt(args.max_dist_px)} px"
                                    if args.max_dist_px else "，无距离截断"))
    if "radius" in methods:
        A(f"* radius：阈值由 `median_nn × {args.radius_factor}` 自动导出"
          if args.radius_px is None else f"* radius：阈值 = {_fmt(args.radius_px)} px（用户指定）")
    if "delaunay" in methods:
        A("* Delaunay：质心三角剖分，无参数（空间点模式的标准最近邻图）")
    if "gabriel" in methods:
        A("* Gabriel：Delaunay 子集，边 ij 存在 ⟺ 以 ij 为直径的圆内无其它质心")
    A("")

    # ---- 2 数据与标定 ----
    A("## 2. 数据与标定\n")
    A(f"* 读取到图像：**{summary.get('n_images_total', 0)}** 张，"
      f"参与建网：**{summary.get('n_images_used', 0)}** 张，"
      f"跳过：**{len(skipped)}** 条记录")
    A(f"* 区域总数（参与建网的图）：**{summary.get('n_regions_total', 0)}**")
    A(f"* 每图区域数：中位数 {_fmt(summary.get('regions_per_image', {}).get('median'), 1)}，"
      f"范围 [{_fmt(summary.get('regions_per_image', {}).get('min'), 0)}, "
      f"{_fmt(summary.get('regions_per_image', {}).get('max'), 0)}]")
    A("* µm/px 取值分布：`"
      + json.dumps(calibration.get("um_per_px_values", {}), ensure_ascii=False) + "`")
    if calibration.get("cross_image_um_comparable") is False:
        A("  * **本数据集存在多个不同的 µm/px 标定值**，"
          "跨图比较**绝对物理距离**无效；只影响 radius 阈值与 µm 距离报告，"
          "不影响 Delaunay/kNN/Gabriel 的拓扑（它们对尺度不变）。")
    if calibration.get("images_without_calibration"):
        A(f"  * 标定不可信的图：{calibration['images_without_calibration']}")
    A("")

    # ---- 2.1 节点类型退化 ----
    ntc = summary.get("node_type_check") or {}
    if ntc:
        A("### 2.1 节点类型是否退化（先查清楚再解释）\n")
        A(f"* 节点类型来源分布：`{json.dumps(ntc.get('node_type_source_census', {}), ensure_ascii=False)}`")
        A(f"* 节点类型取值普查：`{json.dumps(ntc.get('node_type_census', {}), ensure_ascii=False)}`")
        A(f"* 排除 `unlabeled` 后的**真实类别数**：{ntc.get('n_distinct_annotated_types')}")
        A(f"* 标注文本（`recognized_text`）取值普查：`{json.dumps(ntc.get('recognized_text_census', {}), ensure_ascii=False)}`")
        if ntc.get("degenerate"):
            A("")
            A(f"> **{ntc.get('note', '')}**")
            A("> 具体来说：本数据集里带标注的区域 `organelle_type` 全部是同一个值，"
              "OCR 文本（`hl` / `M` / `Ml`）只是识别噪声而非不同的细胞器类别。")
            A("> 因此本报告**不包含**任何按细胞器类型分组的指标，"
              "也**不能**从本数据得到\"某两类细胞器互作更强\"这类结论——"
              "数据里根本没有两类。")
        A("")
    if skipped:
        A("### 2.1 被跳过的图及原因\n")
        A("| 图像 | 原因 |")
        A("| --- | --- |")
        reasons: Dict[str, int] = {}
        for s in skipped[:60]:
            A(f"| `{s['image']}` | {s['reason']} |")
            key = s["reason"].split("（")[0]
            reasons[key] = reasons.get(key, 0) + 1
        if len(skipped) > 60:
            A(f"| … | 另有 {len(skipped) - 60} 条，见 `skipped.json` |")
        A("")
        A("按原因归并：" + "；".join(f"{k} × {v}" for k, v in reasons.items()) + "\n")

    # ---- 3 图密度 ----
    A("## 3. 图密度（阈值好坏的第一诊断量）\n")
    A("密度 = 2M / (N(N−1))，接近 0 说明图快散了，接近 1 说明阈值大到把什么都连上了。\n")
    A("| 方法 | 有边的图数 / 总图数 | 边数中位数 | 密度中位数 | 密度范围 | 空图的张数 |")
    A("| --- | --- | --- | --- | --- | --- |")
    for meth in methods:
        d = summary.get("by_method", {}).get(meth, {})
        A(f"| `{meth}` | {d.get('n_images_with_graph', 0)} / {d.get('n_images', 0)} | "
          f"{_fmt(d.get('median_edges'), 1)} | {_fmt(d.get('median_density'), 4)} | "
          f"[{_fmt(d.get('min_density'), 4)}, {_fmt(d.get('max_density'), 4)}] | "
          f"{d.get('n_empty_graphs', 0)} |")
    A("")

    # ---- 4 阈值 / 方法敏感性 ----
    if sweep:
        A("### 3.1 接触阈值敏感性（同一批图，只改阈值）\n")
        A("| 阈值 (px) | 阈值来源 | 有边的图数 | 总边数 | 密度中位数 |")
        A("| --- | --- | --- | --- | --- |")
        for s in sweep:
            A(f"| {_fmt(s['tol_px'], 2)} | {s['label']} | {s['n_images_with_edges']} | "
              f"{s['n_edges_total']} | {_fmt(s['median_density'], 4)} |")
        A("")
        pc = sweep[0].get("pair_distance_percentiles") or {}
        nn = sweep[0].get("nn_distance_percentiles") or {}
        if pc:
            A("**每个区域对**的最小边界间隙分布（像素，全数据集合并）："
              + "，".join(f"p{int(p)}={_fmt(v, 1)}" for p, v in pc.items()) + "\n")
        if nn:
            A("**每个区域到最近区域**的间隙分布（像素，全数据集合并）："
              + "，".join(f"p{int(p)}={_fmt(v, 1)}" for p, v in nn.items()) + "\n")
    if len(methods) > 1:
        A("### 3.2 方法敏感性（同一批图，只改邻接判据）\n")
        A("| 方法 | 密度中位数 | H_deg 归一化 中位数 | 聚类系数 中位数 | 转移熵 中位数 |")
        A("| --- | --- | --- | --- | --- |")
        for meth in methods:
            d = summary.get("by_method", {}).get(meth, {})
            A(f"| `{meth}` | {_fmt(d.get('median_density'), 4)} | "
              f"{_fmt(d.get('median_H_deg_normalized'), 4)} | "
              f"{_fmt(d.get('median_clustering_mean_local'), 4)} | "
              f"{_fmt(d.get('median_H_rate_stationary'), 4)} |")
        A("")

    # ---- 5 汇总指标 ----
    A("## 4. 跨图汇总指标（每个值都带 N）\n")
    A("`N` = 该指标有定义的图数（其余图因无边缘/节点不足而无定义）；"
      "`退化` = 所有图的取值完全相同，说明该指标在本数据集上没有区分度。\n")
    A("| 指标 | N | 缺失 | 中位数 | 均值 | 标准差 | 范围 | 退化 |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for meth in methods:
        A(f"| **{meth}** | | | | | | | |")
        mset = summary.get("metrics_by_method", {}).get(meth, {}).get("metrics", {})
        for col, _ in FLAT_FIELDS:
            e = mset.get(col)
            if not e or not e.get("n"):
                continue
            A(f"| `{col}` | {e['n']} | {e['n_missing']} | {_fmt(e.get('median'))} | "
              f"{_fmt(e.get('mean'))} | {_fmt(e.get('std'))} | "
              f"[{_fmt(e.get('min'))}, {_fmt(e.get('max'))}] | "
              f"{'**是**' if e.get('degenerate') else '否'} |")
    A("")

    # ---- 6 零模型 ----
    if null_summary:
        A("## 5. 空间零模型对照（同 N 的均匀随机点过程）\n")
        A("零模型回答：若这些区域的位置**完全随机**，同样的建图规则会给出什么？\n")
        A("| 方法 | 指标 | 观测中位数 (N) | 零模型中位数 | 差值 |")
        A("| --- | --- | --- | --- | --- |")
        for meth, tbl in null_summary.items():
            for k, e in tbl.get("aggregate", {}).items():
                A(f"| `{meth}` | `{k}` | {_fmt(e.get('observed_median'))} "
                  f"({e.get('n_observed', 0)}) | {_fmt(e.get('null_median'))} | "
                  f"{_fmt(e.get('difference'))} |")
        A("")
        A("零模型**不适用于 `boundary_contact`**（它依赖区域形状而非质心），"
          "该方法的行不会出现。\n")

    # ---- 7 李亚普诺夫 ----
    A("## 6. 李亚普诺夫指数\n")
    if lyapunov is None:
        A("**未计算（默认且正确的做法）。** 本数据集是单时间点静态图像，"
          "不存在时间序列，无法做相空间重构。报告中**不包含任何 λ 数值**。\n")
    else:
        A(f"数据来源：`{lyapunov.get('source')}`。"
          "**这是伪序列，不是时间序列**，下面的数字没有动力学含义。\n")
        A(f"* 序列长度：{lyapunov.get('series_length')}")
        A(f"* 结论：`reliable = {_fmt(lyapunov.get('reliable'))}`，"
          f"依据 = `{lyapunov.get('reliability_basis')}`")
        A(f"* 推荐斜率：{_fmt(lyapunov.get('recommended_slope'))}（nat/采样点；"
          "单位如此，因为数据集没有采样率）")
        A(f"* 理由：{lyapunov.get('reason')}")
        A("")
        sc = lyapunov.get("sensitivity_scan") or {}
        if sc.get("table"):
            A("(τ, m) 敏感性扫描（前若干行）：\n")
            A("| τ | m | λ | R² | 标度区点数 | 可靠 |")
            A("| --- | --- | --- | --- | --- | --- |")
            for row in sc["table"][:12]:
                A(f"| {row['tau']} | {row['m']} | {_fmt(row['slope'])} | "
                  f"{_fmt(row['r2'])} | {row['n_fit_points']} | "
                  f"{'是' if row['reliable'] else '否'} |")
            A("")

    # ---- 8 逐图 ----
    A("## 7. 逐图结果\n")
    A("| 图像 | N | M | 密度 | H_deg(bits) | H_deg归一 | H_权(bits) | 聚类 | "
      "路径(LCC) | Q | 转移熵 | MI(bits) | 渗流(随机) | 警告 |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in per_image:
        if r.get("status") != "ok":
            continue
        A(f"| `{r['image']}` | {_fmt(r.get('n_nodes'), 0)} | {_fmt(r.get('n_edges'), 0)} | "
          f"{_fmt(r.get('density'))} | {_fmt(r.get('H_deg_bits'))} | "
          f"{_fmt(r.get('H_deg_normalized'))} | {_fmt(r.get('H_weight_bits'))} | "
          f"{_fmt(r.get('clustering_mean_local'))} | {_fmt(r.get('path_length_lcc'))} | "
          f"{_fmt(r.get('modularity'))} | {_fmt(r.get('H_rate_stationary'))} | "
          f"{_fmt(r.get('MI_bits'))} | {_fmt(r.get('percolation_node_random'))} | "
          f"{r.get('n_warnings', 0)} |")
    A("")

    # ---- 9 退化 ----
    A("## 8. 退化与失效的指标（如实列出）\n")
    deg_lines: List[str] = []
    near_lines: List[str] = []
    for meth in methods:
        mset = summary.get("metrics_by_method", {}).get(meth, {}).get("metrics", {})
        for col, e in mset.items():
            if e.get("n", 0) > 1 and e.get("degenerate"):
                deg_lines.append(f"* `{meth}` / `{col}`：{e['n']} 张图的取值"
                                 f"完全相同（={_fmt(e.get('median'))}），**无区分度**")
            elif e.get("near_degenerate"):
                near_lines.append(
                    f"* `{meth}` / `{col}`：**近退化**——{e['mode_fraction']:.0%} 的图"
                    f"（{e['n']} 张中有定义的）取值都等于 {_fmt(e.get('median'))}，"
                    "该指标在本数据集上几乎没有区分度")
            elif e.get("n", 0) == 0:
                deg_lines.append(f"* `{meth}` / `{col}`：**所有图都没有定义**"
                                 f"（{e.get('n_missing')} 张缺失）")
    if deg_lines:
        A("**完全退化（所有图取值相同或全部无定义）：**\n")
        A("\n".join(deg_lines) + "\n")
    if near_lines:
        A("**近退化（≥80% 的图取值相同）：**\n")
        A("\n".join(near_lines) + "\n")
        A("> 近退化**不是**\"结果稳定\"，而是\"这个指标在这批数据上没有信息\"。"
          "最常见的原因是图太小：本数据集每图区域数中位数只有 "
          f"{_fmt(summary.get('regions_per_image', {}).get('median'), 1)}，"
          "在这种规模下 `k=6` 的 kNN 图几乎必然是**完全图**，"
          "于是密度、归一化度熵、聚类系数全部顶到上界。"
          "要得到有区分度的数字，必须先把图做大（更多区域/更合适的方法）。\n")
    if not deg_lines and not near_lines:
        A("没有检测到退化或近退化的指标。\n")

    n_lt3 = 0
    for r in per_image:
        if r.get("status") == "ok" and (r.get("n_nodes") or 0) < 3:
            n_lt3 += 1
    if n_lt3:
        A(f"* 有 {n_lt3} 条记录对应的图**节点数 < 3**："
          "Delaunay 三角剖分与 Gabriel 图在数学上需要至少 3 个点，"
          "因此这些图在这两种判据下**按定义就是空图**（不是 bug，已在 "
          "`warnings` 中逐图说明）。kNN / radius 在 N=2 时仍能给出 1 条边。\n")

    # ---- 10 复现 ----
    A("## 9. 复现命令\n")
    A("```bash")
    A(summary.get("command_line", ""))
    A("```\n")

    A(LIMITATIONS_MD)

    A("\n## 附：指标的精确定义与单位\n")
    A("| 指标 | 度量对象 | 单位 / 范围 | 关键注意 |")
    A("| --- | --- | --- | --- |")
    A("| `H_deg_bits` | 随机抽一个节点其**度**的香农熵 | bit | 上界 log2(N)，**不能跨 N 比较** |")
    A("| `H_deg_normalized` | 上者 / log2(N) | 无量纲 [0,1] | **跨样本比较只能用这个** |")
    A("| `H_weight_bits` | 随机抽一条边其**权重所属分位箱**的熵 | bit | 等频分箱下上界≈log2(箱数)，反映权重是否成簇 |")
    A("| `clustering_global` | 传递性 = 3×三角形/连通三元组 | 无量纲 [0,1] | 对**三元组**等权，不是对节点 |")
    A("| `clustering_mean_local` | 各节点局部聚类系数的算术平均 | 无量纲 [0,1] | 含度为 0/1 的节点（与 networkx 一致） |")
    A("| `path_length_lcc` | LCC 上所有节点对最短路的平均 | **边数（跳）** | 只在 LCC 上算；不是物理长度 |")
    A("| `degree_assortativity` | 每条边两端**度**的 Pearson 相关 | 无量纲 [−1,1] | 正则图上**无定义**（返回 null） |")
    A("| `modularity` | 社区内边占比 − 度序列零模型期望 | 无量纲 | N<10 时方差极大 |")
    A("| `mean_betweenness` | 各节点介数中心性的均值 | 无量纲 [0,1] | O(N·M)，N>300 直接跳过 |")
    A("| `H_rate_stationary` | 图上随机游走在**离散状态**上诱导的马尔可夫链熵率 | bit/步 | **空间**转移，非时间 |")
    A("| `MI_bits` | 相邻节点对之间状态的互信息 | bit | **空间**依赖，非时间/因果 |")
    A("| `percolation_*` | 移除节点/边到 LCC 跌破 50% 时的移除比例 | 无量纲 (0,1] | 未跌破时 fraction=1 且 reached=false |")
    A("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# ==========================================================================
# 主流程
# ==========================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口。"""
    t_start = time.time()
    args = build_argparser().parse_args(argv)
    np.random.seed(args.seed)

    methods = [m.strip() for m in args.method.split(",") if m.strip()]
    bad = [m for m in methods if m not in netbuild.METHODS]
    if bad:
        print(f"[ERROR] 未知方法 {bad}；可选 {list(netbuild.METHODS)}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    # ---- 读数据 ----
    log(f"读取区域表：{os.path.abspath(args.annotations)}")
    tables, load_failures = netbuild.load_region_tables(args.annotations)
    skipped: List[Dict[str, str]] = list(load_failures)
    if not tables:
        msg = (load_failures[0]["reason"] if load_failures
               else "没有可用的区域文件")
        print(f"[ERROR] {msg}", file=sys.stderr)
        print("        请先运行：python tools/morpho/extract_regions.py "
              "--separated \"data/分离的标注信息\" --originals \"data/原图片\" "
              "--scale-dir \"data/scale_mat\" --out data/annotations_v2",
              file=sys.stderr)
        return 2

    n_images_total = len(tables)
    if args.limit:
        tables = tables[: args.limit]
    log(f"  读到 {n_images_total} 张图；本次处理 {len(tables)} 张")

    # ---- 节点类型普查（先查清楚它是不是退化的，再决定报告怎么写）----
    type_census: Dict[str, int] = {}
    text_census: Dict[str, int] = {}
    type_source_census: Dict[str, int] = {}
    for t in tables:
        types, src = t.node_types(mode=args.node_types)
        type_source_census[src] = type_source_census.get(src, 0) + 1
        for x in types:
            type_census[x] = type_census.get(x, 0) + 1
        for r in t.regions:
            key = (r.recognized_text or "<empty>")
            text_census[key] = text_census.get(key, 0) + 1
    node_type_check = {
        "n_distinct_node_types": len(type_census),
        "n_distinct_annotated_types": len([k for k in type_census
                                           if k not in ("unlabeled", "")]),
        "node_type_census": dict(sorted(type_census.items(),
                                        key=lambda kv: -kv[1])),
        "node_type_source_census": type_source_census,
        "recognized_text_census": dict(sorted(text_census.items(),
                                              key=lambda kv: -kv[1])),
        "degenerate": False,
        "note": "",
    }
    # 判据：把 "unlabeled" 这类占位符排除后，若真实类别 ≤ 1，就是退化的。
    # 只数 len(census) 会被占位符蒙混过去（M + unlabeled = 2 → 看着"不退化"），
    # 但生物学上数据里只有一类细胞器，任何按类型分组的指标依然毫无意义。
    n_real = node_type_check["n_distinct_annotated_types"]
    if n_real <= 1:
        real = [k for k in type_census if k not in ("unlabeled", "")]
        only = real[0] if real else "<none>"
        node_type_check["degenerate"] = True
        node_type_check["note"] = (
            f"排除占位符后，本数据集只有 **1 个**真实细胞器类别（`{only}`），"
            f"另有 {type_census.get('unlabeled', 0)} 个区域没有任何类型标注。"
            "因此任何按细胞器类型分组的指标（类型熵、类型间边比例、"
            "按类型的转移熵、类型同配性等）在本数据集上**恒为退化值**，"
            "既不能作为发现，也不能用于区分样本。本流程因此不输出这类指标。"
        )
        log(f"  [节点类型] 真实类别只有 {n_real} 个（{only}）+ "
            f"{type_census.get('unlabeled', 0)} 个 unlabeled ——"
            "类型相关指标全部退化，报告中会显式声明")

    for t in tables:
        if t.n < args.min_regions:
            skipped.append({
                "image": t.image,
                "reason": f"区域数 {t.n} < --min-regions {args.min_regions}，无法建网",
            })
    usable = [t for t in tables if t.n >= args.min_regions]

    # ---- 标定分布 ----
    calib_counts: Dict[str, int] = {}
    no_calib: List[str] = []
    for t in usable:
        if t.um_per_px is None:
            no_calib.append(t.image)
            calib_counts["none"] = calib_counts.get("none", 0) + 1
        else:
            key = f"{t.um_per_px:.6g}"
            calib_counts[key] = calib_counts.get(key, 0) + 1
    calibration = {
        "um_per_px_values": calib_counts,
        "n_distinct_calibrations": len([k for k in calib_counts if k != "none"]),
        "cross_image_um_comparable": len([k for k in calib_counts if k != "none"]) <= 1,
        "images_without_calibration": no_calib,
    }

    # ---- 接触几何（一次算好，供阈值扫描复用）----
    geom: Dict[str, Any] = {}
    sweep: Optional[List[Dict[str, Any]]] = None
    sweep_percentiles = [1, 2, 5, 10, 25]
    contact_tol_used = float(args.min_contact_px)

    if "boundary_contact" in methods:
        log("预计算接触几何（光栅化标签图 + 边界点 + 精确距离矩阵）...")
        for i, t in enumerate(usable, 1):
            try:
                geom[t.image] = netbuild.prepare_contact_geometry(t)
            except Exception as exc:  # noqa: BLE001
                skipped.append({"image": t.image,
                                "reason": f"接触几何计算失败：{type(exc).__name__}: {exc}"})
            if i % 20 == 0:
                log(f"  {i}/{len(usable)}")

        pooled_pairs = (np.concatenate([g.pair_distances() for g in geom.values()])
                        if geom else np.zeros(0))
        pooled_nn = (np.concatenate([g.nn_distances() for g in geom.values()])
                     if geom else np.zeros(0))
        pct = ({int(p): float(np.percentile(pooled_pairs, p)) for p in
                [1, 5, 10, 25, 50, 75, 90, 99]} if pooled_pairs.size else {})
        nn_pct = ({int(p): float(np.percentile(pooled_nn, p)) for p in
                   [1, 5, 10, 25, 50, 75, 90, 99]} if pooled_nn.size else {})

        if args.contact_tol_percentile is not None and pooled_pairs.size:
            contact_tol_used = float(np.percentile(pooled_pairs,
                                                   float(args.contact_tol_percentile)))
            log(f"  接触阈值由数据导出：pair 距离 p{args.contact_tol_percentile:g} "
                f"= {contact_tol_used:.2f} px")

        sweep = []
        tols = sorted(set([float(contact_tol_used)]
                          + [float(pct[p]) for p in sweep_percentiles if p in pct]))
        for tol in tols:
            n_edge_imgs = 0
            tot_edges = 0
            dens: List[float] = []
            label = "用户指定"
            for p in sweep_percentiles:
                if p in pct and abs(pct[p] - tol) < 1e-9:
                    label = f"数据集区域对间隙的 p{p}"
            for t in usable:
                if t.image not in geom:
                    continue
                nn_ = netbuild.build_network(t, method="boundary_contact",
                                             min_contact_px=tol,
                                             contact_geometry=geom[t.image],
                                             node_type_mode=args.node_types)
                dens.append(nn_.density())
                tot_edges += nn_.n_edges
                n_edge_imgs += (nn_.n_edges > 0)
            sweep.append({
                "tol_px": float(tol),
                "label": label,
                "n_images_with_edges": int(n_edge_imgs),
                "n_edges_total": int(tot_edges),
                "median_density": float(np.median(dens)) if dens else 0.0,
                "pair_distance_percentiles": pct,
                "nn_distance_percentiles": nn_pct,
            })
        log(f"  阈值扫描完成：{len(sweep)} 个阈值")

    # ---- 逐图建网 + 指标 ----
    per_image: List[Dict[str, Any]] = []
    used_images: List[str] = []
    for meth in methods:
        log(f"方法 {meth}：开始逐图处理")
        for idx, t in enumerate(usable, 1):
            rec: Dict[str, Any] = {"image": t.image, "method": meth}
            try:
                kw: Dict[str, Any] = {"node_type_mode": args.node_types,
                                      "distance_unit": args.distance_unit}
                if meth == "boundary_contact":
                    if t.image not in geom:
                        skipped.append({"image": t.image,
                                        "reason": "接触几何缺失，跳过 boundary_contact"})
                        continue
                    kw["contact_geometry"] = geom[t.image]
                    kw["min_contact_px"] = contact_tol_used
                elif meth == "knn":
                    kw["k"] = args.k
                    kw["max_dist"] = args.max_dist_px
                elif meth == "radius":
                    if args.radius_um is not None:
                        if t.um_per_px is None:
                            skipped.append({
                                "image": t.image,
                                "reason": "指定了 --radius-um 但该图标定不可信，"
                                          "无法换算成像素，radius 图跳过",
                            })
                            continue
                        kw["radius"] = (args.radius_um / t.um_per_px
                                        if args.distance_unit == "px"
                                        else args.radius_um)
                    else:
                        kw["radius"] = args.radius_px
                    kw["radius_factor"] = args.radius_factor

                net = netbuild.build_network(t, method=meth, **kw)
                m = metrics.compute_all_metrics(
                    net, table=t,
                    state_source=args.state_source,
                    n_state_bins=args.n_state_bins,
                    weight_bins=args.weight_bins,
                    weight_binning=args.weight_binning,
                    percolation_seed=args.percolation_seed,
                    max_betweenness_nodes=args.max_betweenness_nodes,
                )
                row = flatten_image_row(t.image, net, m, net.params)
                nm = spatial_null_metrics(t, meth, net.params, args.null_reps,
                                          args.seed)
                rec.update({
                    "status": "ok",
                    "params": net.params,
                    "warnings": net.warnings,
                    "skipped_metrics": m.get("skipped", []),
                    "metrics": m,
                    "null_model": nm,
                })
                rec.update(row)
                rec["edge_attrs_summary"] = {
                    "n_edges": net.n_edges,
                    "weight_stats": net.edge_weight_stats(),
                }
                if args.save_networks:
                    paths = net.save(os.path.join(args.out, "networks"))
                    rec["network_files"] = paths
            except Exception as exc:  # noqa: BLE001
                rec.update({"status": "error",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=3)})
                skipped.append({"image": t.image,
                                "reason": f"[{meth}] 建网/指标失败："
                                          f"{type(exc).__name__}: {exc}"})
                log(f"  [FAIL] {t.image} ({meth}): {exc}")
            per_image.append(rec)
            if idx % 20 == 0:
                log(f"  {idx}/{len(usable)}")

    ok_records_by_method: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}
    for r in per_image:
        if r.get("status") == "ok":
            ok_records_by_method[r["method"]].append(r)
            if r["image"] not in used_images:
                used_images.append(r["image"])

    # ---- 汇总 ----
    by_method: Dict[str, Any] = {}
    metrics_by_method: Dict[str, Any] = {}
    null_by_method: Dict[str, Any] = {}
    for meth in methods:
        rows = ok_records_by_method[meth]
        agg = aggregate_rows(rows)
        metrics_by_method[meth] = agg
        dens = _finite([r.get("density") for r in rows])
        edges = _finite([r.get("n_edges") for r in rows])
        by_method[meth] = {
            "n_images": len(rows),
            "n_images_with_graph": int(np.count_nonzero(edges > 0)) if edges.size else 0,
            "n_empty_graphs": int(np.count_nonzero(edges == 0)) if edges.size else 0,
            "n_edges_total": int(edges.sum()) if edges.size else 0,
            "median_edges": float(np.median(edges)) if edges.size else None,
            "median_density": float(np.median(dens)) if dens.size else None,
            "min_density": float(dens.min()) if dens.size else None,
            "max_density": float(dens.max()) if dens.size else None,
            "median_H_deg_normalized":
                _g(agg, "metrics", "H_deg_normalized", "median"),
            "median_clustering_mean_local":
                _g(agg, "metrics", "clustering_mean_local", "median"),
            "median_H_rate_stationary":
                _g(agg, "metrics", "H_rate_stationary", "median"),
        }

        nulls = [r["null_model"] for r in rows if r.get("null_model")]
        if nulls:
            agg_null: Dict[str, Any] = {}
            for key in ("density", "H_deg_normalized",
                        "clustering_mean_local", "path_length_lcc"):
                obs = _finite([r.get(key) for r in rows])
                nv = _finite([n[key]["mean"] for n in nulls
                              if n.get(key, {}).get("mean") is not None])
                agg_null[key] = {
                    "n_observed": int(obs.size),
                    "observed_median": float(np.median(obs)) if obs.size else None,
                    "null_median": float(np.median(nv)) if nv.size else None,
                    "difference": (float(np.median(obs) - np.median(nv))
                                   if obs.size and nv.size else None),
                }
            null_by_method[meth] = {"aggregate": agg_null, "reps": args.null_reps}

    # ---- 李亚普诺夫（默认关闭）----
    lyap: Optional[Dict[str, Any]] = None
    if args.lyapunov_source != "none":
        log("李亚普诺夫：注意这是伪序列，不是时间序列")
        try:
            series_name = args.lyapunov_series or args.state_source
            if args.lyapunov_source == "node-order":
                rows0 = ok_records_by_method.get(methods[0], [])
                if rows0:
                    t0 = next(x for x in usable if x.image == rows0[0]["image"])
                    series = metrics.node_state_values(t0, series_name)
                    lyap = dynamics.analyze_series(
                        series, pseudo_sequence=True,
                        min_points=args.lyapunov_min_points,
                        min_r2=args.lyapunov_min_r2)
                    lyap["source"] = (f"伪序列：图像 {t0.image} 的逐节点 "
                                      f"{series_name}（按 region_id 顺序），"
                                      f"长度 {series.size}")
            else:  # image-order：把逐图的某个扁平列按图像名顺序当作伪序列
                key = series_name
                series = _finite([r.get(key)
                                  for r in ok_records_by_method.get(methods[0], [])])
                if series.size:
                    lyap = dynamics.analyze_series(
                        series, pseudo_sequence=True,
                        min_points=args.lyapunov_min_points,
                        min_r2=args.lyapunov_min_r2)
                    lyap["source"] = (f"伪序列：逐图 `{key}`（按图像名排序），"
                                      f"长度 {series.size}")
        except Exception as exc:  # noqa: BLE001
            lyap = {"reliable": False, "reliability_basis": "none",
                    "recommended_slope": None,
                    "reason": f"计算失败：{type(exc).__name__}: {exc}"}

    # ---- 写文件 ----
    command_line = "python " + " ".join(
        [os.path.relpath(os.path.abspath(__file__), os.getcwd())]
        + (list(argv) if argv is not None else sys.argv[1:]))

    regions_per_image = _finite([t.n for t in usable])
    n_without_edges = sum(v["n_empty_graphs"] for v in by_method.values())
    summary: Dict[str, Any] = {
        "schema": "network_analysis/1.0",
        "generator": "tools/network/run_analysis.py",
        "version": __version__,
        "command_line": command_line,
        "arguments": vars(args),
        "methods": methods,
        "n_images_total": n_images_total,
        "n_images_used": len(used_images),
        "n_images_skipped": len(skipped),
        "n_images_without_edges": int(n_without_edges),
        "n_regions_total": int(sum(t.n for t in usable)),
        "regions_per_image": {
            "n": int(regions_per_image.size),
            "median": (float(np.median(regions_per_image))
                       if regions_per_image.size else None),
            "min": (float(regions_per_image.min())
                    if regions_per_image.size else None),
            "max": (float(regions_per_image.max())
                    if regions_per_image.size else None),
            "mean": (float(regions_per_image.mean())
                     if regions_per_image.size else None),
        },
        "calibration": calibration,
        "node_type_check": node_type_check,
        "contact_tolerance_px": {
            "used": contact_tol_used,
            "source": ("data_percentile" if args.contact_tol_percentile is not None
                       else "user"),
            "percentile": args.contact_tol_percentile,
        },
        "by_method": by_method,
        "metrics_by_method": metrics_by_method,
        "null_model_by_method": null_by_method,
        "threshold_sweep": sweep,
        "lyapunov": lyap,
        "metrics_notes": {
            "H_deg_bits": "上界为 log2(N)，不可跨 N 比较；跨样本比较请用 H_deg_normalized",
            "path_length_lcc": "只在最大连通分量上计算，单位是边数（跳）",
            "MI_bits": "相邻节点的空间互信息，不是时间/因果量",
            "H_rate_stationary": "图上随机游走在离散状态上的马尔可夫链熵率，单位 bit/步；空间转移",
            "percolation_*": "LCC 跌破 50% 时已移除的比例；reached=false 时 fraction=1 只表示未跌破",
        },
    }

    with open(os.path.join(args.out, "per_image.json"), "w", encoding="utf-8") as fh:
        json.dump(per_image, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "skipped.json"), "w", encoding="utf-8") as fh:
        json.dump(skipped, fh, ensure_ascii=False, indent=1)
    if sweep:
        with open(os.path.join(args.out, "contact_geometry.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(sweep, fh, ensure_ascii=False, indent=1)

    cols = ["image", "method"] + [c for c, _ in FLAT_FIELDS] + [
        "um_per_px", "coordinate_unit", "node_type_source",
        "n_warnings", "n_skipped_metrics"]
    with open(os.path.join(args.out, "metrics.csv"), "w", encoding="utf-8",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in per_image:
            if r.get("status") == "ok":
                w.writerow({k: r.get(k) for k in cols})

    write_report(os.path.join(args.out, "REPORT.md"), args, methods, per_image,
                 summary, skipped, sweep, calibration, lyap, null_by_method,
                 time.time() - t_start)

    # ---- 终端摘要 ----
    log("")
    log("=" * 68)
    log(f"完成：{len(used_images)}/{n_images_total} 张图建网，"
        f"{summary['n_regions_total']} 个区域，跳过 {len(skipped)} 条记录")
    for meth in methods:
        d = by_method[meth]
        log(f"  {meth:18s} 有边图 {d['n_images_with_graph']}/{d['n_images']}，"
            f"总边数 {d['n_edges_total']}，密度中位数 "
            f"{_fmt(d['median_density'])}")
    if calibration["cross_image_um_comparable"] is False:
        log(f"  [注意] µm/px 标定有 {calibration['n_distinct_calibrations']} "
            "个不同取值，跨图绝对距离不可比")
    log(f"输出：{os.path.abspath(args.out)}")
    log("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
