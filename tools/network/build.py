#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build.py —— 从分割区域**真正重建**细胞器互作网络
================================================

本模块回答一个此前项目里从未被回答的问题：
**「哪些细胞器区域之间存在一条边？」**

交互判据（interaction criterion）
---------------------------------
"互作"在活细胞里指两类可观测事件：

1. **接触**（membrane contact site）：两个细胞器的膜在 ~10–30 nm 内，
   由栓系蛋白（tether）连接；
2. **共定位 / 共运动**：两个信号在时间上同步变化或同步位移。

本数据集是**单时间点**的静态分割标注，没有时间轴，因此第 2 类无法测量。
我们退而使用第 1 类的**形态学代理**：区域轮廓之间的最小像素距离。
默认方法 ``boundary_contact`` 就是在实现这个代理量，并且它是唯一
"几何上直接对应接触"的判据；其余 4 种（Delaunay / kNN / radius / Gabriel）
都是**空间点模式**意义上的邻近图，只用于敏感性分析，不应单独作为生物学结论。

5 种邻接方法
------------
===================  ==========================================================
方法                  判据
===================  ==========================================================
``boundary_contact``  **默认。** 两个区域轮廓（边界像素集合）的最小距离
                      ≤ ``min_contact_px`` 像素。用光栅化标签图的边界像素 + KD 树
                      精确计算，不依赖质心。
``delaunay``          Delaunay 三角剖分（质心）。空间点模式的标准最近邻图。
``knn``               每个质心的 k 个最近邻（默认 k=6），可加距离截断。
``radius``            固定距离阈值。阈值必须由数据导出并报告图密度（见下）。
``gabriel``           Gabriel 图，Delaunay 的子集：边 ij 存在 ⟺ 以 ij 为直径的
                      圆内不含其它任何质心。
===================  ==========================================================

radius 阈值的论证
-----------------
``radius`` 不能随便取。本模块的做法是：若调用方不给阈值，则用**本数据集的
最近邻距离中位数** ``median_nn``（把所有区域的"到最近邻质心的距离"取中位数）
乘以 ``radius_factor``（默认 1.5）。选 1.5 的理由：在完全空间随机（Poisson）
点过程下，最近邻距离中位数约为 ``0.5 / sqrt(λ)``，取 1.5 倍中位数可以让
每个点在随机分布下平均连到约 2–3 个邻居，既不退化成空图也不退化成全连接。
**无论阈值如何得到，``build_network`` 都会在 ``params`` 里回填实际使用的
阈值和结果的图密度 ``density``**，阈值选坏时一眼可见（密度接近 0 或接近 1
都说明阈值失效）。

坐标与单位
----------
- 区域坐标在 ``annotations_v2`` 中是 **1-based (x, y) = (列, 行)**。
- 若某张图的标定可信（``scale.ok`` 且 ``um_per_px`` 非 None），则
  ``OrganelleNetwork.centroids`` 以 **µm** 为单位，``coordinate_unit='um'``；
  否则以 **像素** 为单位，记 ``'px'``。两种情况都额外保留 ``centroids_px``。
- 每条边都同时存 ``dist_px`` 与 ``dist_um``（后者在无标定时为 ``None``）。

序列化
------
``OrganelleNetwork.save(dir)`` 写 ``<dir>/<image>.json`` + ``<dir>/<image>.npz``
（邻接矩阵放在 npz 里，避免 JSON 膨胀与浮点精度损失）。
``OrganelleNetwork.load(path)`` 可读回任意一个。
"""

from __future__ import annotations

import glob
import json
import math
import os
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree, Delaunay, QhullError

#: 所有可选的邻接方法名（CLI 的 ``--method`` 取值范围）
METHODS: Tuple[str, ...] = (
    "boundary_contact",
    "delaunay",
    "knn",
    "radius",
    "gabriel",
)

#: 生物学上最合理的代理判据，作为默认方法
DEFAULT_METHOD = "boundary_contact"

#: ``boundary_contact`` 的默认接触阈值（像素）：
#: 分割轮廓本身有 1–2 px 的手绘/简化误差，取 2 px 允许这个误差，
#: 同时远小于本数据集里区域之间的典型间隙（见 REPORT.md 的距离分布）。
DEFAULT_CONTACT_TOL_PX = 2.0


# ==========================================================================
# 1. 区域表的读取
# ==========================================================================

@dataclass
class RegionRecord:
    """单个区域的、建网所需的字段（其它特征原样保留在 ``raw`` 里）。"""

    region_id: int
    center_x: float          # 1-based，像素
    center_y: float          # 1-based，像素
    area: float              # 像素
    contour: np.ndarray      # (P, 2) 1-based (x, y) 像素
    gray_mean: float
    gray_std: float
    cv_gray: float
    is_labeled: bool
    quality_flag: str
    organelle_type: str
    recognized_text: str
    um: Optional[Dict[str, float]]
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def area_um2(self) -> Optional[float]:
        """区域面积（µm²）；无标定时返回 ``None``（**不猜**）。"""
        if not self.um:
            return None
        v = self.um.get("area")
        return float(v) if v is not None else None


@dataclass
class RegionTable:
    """一张图的全部区域 + 标定信息。

    ``um_per_px`` 为 ``None`` 表示标定不可信；此时所有以 µm 为单位的量
    都不会被计算，而不是像原流水线那样静默代入 0.1。
    """

    image: str
    image_shape: Tuple[int, int]           # (H, W)
    um_per_px: Optional[float]
    scale_ok: bool
    geometry_note: str
    regions: List[RegionRecord]
    source_path: str = ""

    # -- 构造 -------------------------------------------------------------
    @staticmethod
    def from_json(path: str) -> "RegionTable":
        """读取 ``annotations_v2/regions/<image>.json``。"""
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)

        scale = d.get("scale") or {}
        um_per_px = scale.get("um_per_px")
        scale_ok = bool(scale.get("ok")) and um_per_px is not None

        regions: List[RegionRecord] = []
        for r in d.get("regions", []):
            contour = np.asarray(r.get("contour") or [], dtype=np.float64)
            if contour.ndim != 2 or contour.shape[1] != 2:
                contour = np.zeros((0, 2), dtype=np.float64)
            regions.append(
                RegionRecord(
                    region_id=int(r["region_id"]),
                    center_x=float(r["center_x"]),
                    center_y=float(r["center_y"]),
                    area=float(r.get("area", 0.0)),
                    contour=contour,
                    gray_mean=float(r.get("gray_mean", np.nan)),
                    gray_std=float(r.get("gray_std", np.nan)),
                    cv_gray=float(r.get("cv_gray", np.nan)),
                    is_labeled=bool(r.get("is_labeled", False)),
                    quality_flag=str(r.get("quality_flag", "ok")),
                    organelle_type=str(r.get("organelle_type") or ""),
                    recognized_text=str(r.get("recognized_text") or ""),
                    um=r.get("um"),
                    raw=r,
                )
            )

        shape = d.get("image_shape") or [0, 0]
        return RegionTable(
            image=str(d.get("image") or os.path.splitext(os.path.basename(path))[0]),
            image_shape=(int(shape[0]), int(shape[1])),
            um_per_px=float(um_per_px) if scale_ok else None,
            scale_ok=scale_ok,
            geometry_note=str(d.get("geometry_note") or ""),
            regions=regions,
            source_path=os.path.abspath(path),
        )

    # -- 便利视图 ---------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.regions)

    def centroids_px(self) -> np.ndarray:
        """(N, 2) 质心，**像素**，1-based ``(x, y)``。"""
        if not self.regions:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray([[r.center_x, r.center_y] for r in self.regions], dtype=np.float64)

    def centroids_um(self) -> Optional[np.ndarray]:
        """(N, 2) 质心，**µm**。无标定时返回 ``None``。"""
        if self.um_per_px is None:
            return None
        return self.centroids_px() * float(self.um_per_px)

    def node_types(self, mode: str = "annotation") -> Tuple[List[str], str]:
        """节点类型标签 + 该标签的来源说明。

        **有文档的降级链（fallback chain）**：

        ``mode='annotation'``（默认，诚实模式）
            1. ``organelle_type``（若非空）
            2. 否则 ``recognized_text``（若非空）
            3. 否则字面量 ``"unlabeled"``。
            本数据集的 OCR 尚未运行，因此实际全部落到第 3 档——这是**已知的
            数据缺口**，必须如实呈现，而不是用形状特征假装成生物学类型。

        ``mode='morphological'``（显式启用才生效）
            在第 3 档时，用**纯形态学**规则给出一个派生标签，仅用于分组统计：
            ``elongated``（major/minor > 3）、``round``（离心率 < 0.6）、
            其余 ``irregular``。这些标签**是形状类别，不是细胞器身份**
            （线粒体、溶酶体、内质网都可以是 elongated）。

        Returns
        -------
        (types, source)
            ``source`` ∈ ``{'organelle_type', 'recognized_text',
            'morphological_rules', 'none'}``，用于在报告里说明标签可信度。
        """
        types: List[str] = []
        used: set = set()
        for r in self.regions:
            t = (r.organelle_type or "").strip()
            if t:
                types.append(t)
                used.add("organelle_type")
                continue
            t = (r.recognized_text or "").strip()
            if t:
                types.append(t)
                used.add("recognized_text")
                continue
            if mode == "morphological":
                types.append(_morphological_type(r))
                used.add("morphological_rules")
            else:
                types.append("unlabeled")
                used.add("none")

        if used == {"organelle_type"}:
            source = "organelle_type"
        elif used <= {"organelle_type", "recognized_text"} and "organelle_type" in used:
            source = "organelle_type"
        elif used == {"recognized_text"}:
            source = "recognized_text"
        elif used == {"none"}:
            source = "none"
        elif used == {"morphological_rules"}:
            source = "morphological_rules"
        else:
            source = "mixed:" + ",".join(sorted(used))
        return types, source


def _morphological_type(r: RegionRecord) -> str:
    """纯形态学分类（见 ``RegionTable.node_types`` 文档）。"""
    raw = r.raw
    major = float(raw.get("major_axis", 0.0) or 0.0)
    minor = float(raw.get("minor_axis", 0.0) or 0.0)
    ecc = float(raw.get("eccentricity", 0.0) or 0.0)
    if minor > 0 and major / minor > 3.0:
        return "elongated"
    if ecc < 0.6:
        return "round"
    return "irregular"


def load_region_tables(annotations_dir: str) -> Tuple[List[RegionTable], List[Dict[str, str]]]:
    """读取 ``<annotations_dir>/regions/*.json``。

    **优雅降级**：目录不存在、为空、或单个文件损坏时都不抛异常，
    而是返回 ``(成功读取的表, 失败清单)``，失败清单里给出原因。

    Returns
    -------
    (tables, failures)
        ``failures`` 为 ``[{'image': ..., 'reason': ...}, ...]``。
    """
    regions_dir = os.path.join(annotations_dir, "regions")
    failures: List[Dict[str, str]] = []

    if not os.path.isdir(regions_dir):
        failures.append({
            "image": "<all>",
            "reason": (
                f"未找到区域目录 {os.path.abspath(regions_dir)}；"
                "请先运行 tools/morpho/extract_regions.py 生成 annotations_v2"
            ),
        })
        return [], failures

    files = sorted(glob.glob(os.path.join(regions_dir, "*.json")))
    if not files:
        failures.append({
            "image": "<all>",
            "reason": f"目录 {os.path.abspath(regions_dir)} 为空，没有 *.json",
        })
        return [], failures

    tables: List[RegionTable] = []
    for fp in files:
        try:
            tables.append(RegionTable.from_json(fp))
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "image": os.path.splitext(os.path.basename(fp))[0],
                "reason": f"{type(exc).__name__}: {exc}",
            })
    return tables, failures


# ==========================================================================
# 2. 标签图光栅化（boundary_contact 需要）
# ==========================================================================

def rasterize_label_map(table: RegionTable) -> np.ndarray:
    """把区域轮廓多边形光栅化成 ``int32`` 标签图（0 = 背景）。

    校验：返回前会检查光栅化面积与 JSON 里记录的 ``area`` 是否一致
    （比值应在 [0.9, 1.1]）。不一致超过 10% 的区域会在 ``warnings`` 里
    被点名——这说明轮廓简化过头，接触判据会失真。

    Raises
    ------
    RuntimeError
        当所有区域轮廓都少于 3 个点（无法成面）时。
    """
    h, w = table.image_shape
    if h <= 0 or w <= 0:
        raise RuntimeError(f"{table.image}: image_shape 非法 {table.image_shape}")

    lab = np.zeros((h, w), dtype=np.int32)
    filled_any = False
    try:
        import cv2
        have_cv2 = True
    except ImportError:  # pragma: no cover - 退化路径
        have_cv2 = False

    for r in table.regions:
        pts = r.contour
        if pts.shape[0] < 3:
            continue
        if have_cv2:
            poly = np.round(pts - 1.0).astype(np.int32)   # 1-based → 0-based
            cv2.fillPoly(lab, [poly], int(r.region_id))
        else:  # pragma: no cover - 退化路径
            _fill_polygon_numpy(lab, pts - 1.0, int(r.region_id))
        filled_any = True

    if not filled_any:
        raise RuntimeError(f"{table.image}: 所有区域轮廓点数 < 3，无法光栅化标签图")

    _check_raster_area(table, lab)
    return lab


def _fill_polygon_numpy(lab: np.ndarray, pts0: np.ndarray, value: int) -> None:
    """无 cv2 时的多边形填充：画边（Bresenham）后做 ``binary_fill_holes``。

    仅作优雅降级用；本机装有 cv2，正常路径不走这里。
    """
    from scipy import ndimage as ndi

    h, w = lab.shape
    edge = np.zeros((h, w), dtype=bool)
    p = np.round(pts0).astype(np.int64)
    for k in range(len(p)):
        x0, y0 = p[k]
        x1, y1 = p[(k + 1) % len(p)]
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        xs = np.round(np.linspace(x0, x1, n)).astype(np.int64)
        ys = np.round(np.linspace(y0, y1, n)).astype(np.int64)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        edge[ys[ok], xs[ok]] = True
    region = ndi.binary_fill_holes(edge)
    lab[region] = value


def _check_raster_area(table: RegionTable, lab: np.ndarray, tol: float = 0.10) -> None:
    """光栅化面积 vs JSON 记录面积的一致性检查（不一致时告警，不抛异常）。"""
    bad: List[Tuple[int, float]] = []
    for r in table.regions:
        a_json = r.area
        a_ras = float(np.count_nonzero(lab == r.region_id))
        if a_json <= 0:
            continue
        ratio = a_ras / a_json
        if not (1.0 - tol <= ratio <= 1.0 + tol):
            bad.append((r.region_id, ratio))
    if bad:
        warnings.warn(
            f"{table.image}: {len(bad)} 个区域的光栅化面积与记录面积偏差 >{tol:.0%}"
            f"（示例 region_id/ratio: {bad[:3]}）；轮廓简化可能过度，接触判据会失真",
            RuntimeWarning,
            stacklevel=2,
        )


# ==========================================================================
# 3. 网络数据结构
# ==========================================================================

@dataclass
class OrganelleNetwork:
    """一张图的细胞器互作网络（**空间邻近代理网络**）。

    Attributes
    ----------
    image
        图像名。
    node_ids
        节点标识，与 ``RegionTable.regions`` 的 ``region_id`` 一一对应。
    centroids
        (N, 2) 质心。单位见 ``coordinate_unit``（有标定 = µm，无 = 像素）。
    centroids_px
        (N, 2) 质心，始终为像素，便于与图像坐标对照。
    areas_um2
        (N,) 面积（µm²）；无标定时为 ``None``。
    node_types
        长度 N 的类型标签；``node_type_source`` 说明其来源与可信度。
    adjacency
        (N, N) 邻接矩阵。默认 ``scipy.sparse.csr_matrix``；
        ``sparse_adjacency=False`` 时为 numpy 稠密数组。
    edge_list
        ``[[i, j], ...]``，0-based 的**节点下标**（指向 ``node_ids``），``i < j``。
    edge_weights
        (M,) 边权。``boundary_contact`` 下为接触长度（µm 或 px，见下）；
        其余方法下为 ``1 / 距离``（距离单位同 ``coordinate_unit``）。
    edge_attrs
        每条边的完整属性：``dist_px`` / ``dist_um`` / ``weight`` /
        ``weight_unit`` / ``contact_px`` / ``contact_um``。
    method, params
        邻接方法与实际参数（含回填的阈值与图密度）。
    """

    image: str
    node_ids: List[int]
    centroids: np.ndarray
    centroids_px: np.ndarray
    areas_um2: Optional[np.ndarray]
    node_types: List[str]
    node_type_source: str
    adjacency: Any
    edge_list: List[List[int]]
    edge_weights: np.ndarray
    edge_attrs: List[Dict[str, Any]]
    method: str
    params: Dict[str, Any]
    coordinate_unit: str = "px"
    image_shape: Tuple[int, int] = (0, 0)
    geometry_note: str = ""
    warnings: List[str] = field(default_factory=list)

    # -- 基本量 -----------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.edge_list)

    def dense_adjacency(self) -> np.ndarray:
        """返回 (N, N) 稠密邻接（0/1，忽略权重）。"""
        if sparse.issparse(self.adjacency):
            return np.asarray(self.adjacency.todense())
        return np.asarray(self.adjacency)

    def weighted_dense(self) -> np.ndarray:
        """返回 (N, N) 稠密**带权**邻接矩阵。"""
        n = self.n_nodes
        wm = np.zeros((n, n), dtype=np.float64)
        for (i, j), w in zip(self.edge_list, self.edge_weights):
            wm[i, j] = w
            wm[j, i] = w
        return wm

    def density(self) -> float:
        """图密度 = 2M / (N(N-1))；N<2 时定义为 0。

        这是判断邻接阈值是否失效的第一诊断量：接近 0 = 图快散了，
        接近 1 = 阈值大到把什么都连上了。
        """
        n, m = self.n_nodes, self.n_edges
        if n < 2:
            return 0.0
        return float(2.0 * m / (n * (n - 1)))

    def degrees(self) -> np.ndarray:
        """(N,) 每个节点的度。"""
        n = self.n_nodes
        deg = np.zeros(n, dtype=np.int64)
        for i, j in self.edge_list:
            deg[i] += 1
            deg[j] += 1
        return deg

    def edge_weight_stats(self) -> Dict[str, float]:
        """边权的最小/中位/最大/均值，用于报告阈值是否合理。"""
        w = np.asarray(self.edge_weights, dtype=np.float64)
        if w.size == 0:
            return {"n_edges": 0}
        return {
            "n_edges": int(w.size),
            "min": float(np.min(w)),
            "median": float(np.median(w)),
            "mean": float(np.mean(w)),
            "max": float(np.max(w)),
        }

    def to_networkx(self):
        """转成 ``networkx.Graph``（缺少 networkx 时抛 ImportError 并说明）。"""
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "需要 networkx 才能做拓扑指标；请 `pip install networkx`，"
                "或使用 metrics.py 中不依赖 networkx 的熵/鲁棒性函数"
            ) from exc
        g = nx.Graph()
        g.add_nodes_from(range(self.n_nodes))
        for (i, j), w, a in zip(self.edge_list, self.edge_weights, self.edge_attrs):
            g.add_edge(int(i), int(j), weight=float(w), **{
                k: v for k, v in a.items() if k != "weight"
            })
        return g

    # -- 序列化 -----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON 可序列化的字典。**邻接矩阵不在这里**，它进 ``.npz`` sidecar。"""
        return {
            "schema": "organelle_network/1.0",
            "image": self.image,
            "image_shape": list(self.image_shape),
            "geometry_note": self.geometry_note,
            "method": self.method,
            "params": self.params,
            "coordinate_unit": self.coordinate_unit,
            "node_ids": [int(v) for v in self.node_ids],
            "node_types": list(self.node_types),
            "node_type_source": self.node_type_source,
            "centroids": np.asarray(self.centroids, dtype=float).tolist(),
            "centroids_px": np.asarray(self.centroids_px, dtype=float).tolist(),
            "areas_um2": (None if self.areas_um2 is None
                          else np.asarray(self.areas_um2, dtype=float).tolist()),
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "density": self.density(),
            "edge_list": [[int(i), int(j)] for i, j in self.edge_list],
            "edge_weights": np.asarray(self.edge_weights, dtype=float).tolist(),
            "edge_attrs": self.edge_attrs,
            "edge_weight_stats": self.edge_weight_stats(),
            "warnings": list(self.warnings),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any], adjacency: Any = None) -> "OrganelleNetwork":
        """从 ``to_dict()`` 的输出还原；``adjacency`` 从 npz sidecar 传入。"""
        edge_list = [[int(i), int(j)] for i, j in d.get("edge_list", [])]
        weights = np.asarray(d.get("edge_weights", []), dtype=np.float64)
        n = int(d.get("n_nodes", len(d.get("node_ids", []))))
        if adjacency is None:
            adjacency = _adjacency_from_edges(edge_list, n)
        areas = d.get("areas_um2")
        return OrganelleNetwork(
            image=str(d["image"]),
            node_ids=[int(v) for v in d.get("node_ids", [])],
            centroids=np.asarray(d.get("centroids", np.zeros((0, 2))), dtype=np.float64),
            centroids_px=np.asarray(d.get("centroids_px", np.zeros((0, 2))), dtype=np.float64),
            areas_um2=None if areas is None else np.asarray(areas, dtype=np.float64),
            node_types=list(d.get("node_types", [])),
            node_type_source=str(d.get("node_type_source", "unknown")),
            adjacency=adjacency,
            edge_list=edge_list,
            edge_weights=weights,
            edge_attrs=list(d.get("edge_attrs", [])),
            method=str(d.get("method", "")),
            params=dict(d.get("params", {})),
            coordinate_unit=str(d.get("coordinate_unit", "px")),
            image_shape=tuple(d.get("image_shape", (0, 0))),  # type: ignore[arg-type]
            geometry_note=str(d.get("geometry_note", "")),
            warnings=list(d.get("warnings", [])),
        )

    def save(self, path: str) -> Dict[str, str]:
        """保存网络。

        ``path`` 可以是
        * 目录 → 写 ``<dir>/<image>.json`` + ``<dir>/<image>.npz``；
        * 以 ``.json`` 结尾的文件路径 → 同名 ``.npz`` sidecar。

        Returns
        -------
        dict
            ``{'json': ..., 'npz': ...}`` 实际写入的路径。
        """
        if path.lower().endswith(".json"):
            json_path = path
            npz_path = path[:-5] + ".npz"
        else:
            os.makedirs(path, exist_ok=True)
            json_path = os.path.join(path, f"{self.image}.json")
            npz_path = os.path.join(path, f"{self.image}.npz")

        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=1)

        adj = self.adjacency
        if sparse.issparse(adj):
            adj = adj.tocsr()
            np.savez_compressed(
                npz_path,
                format=np.asarray(["csr"]),
                data=adj.data, indices=adj.indices, indptr=adj.indptr,
                shape=np.asarray(adj.shape, dtype=np.int64),
            )
        else:
            np.savez_compressed(npz_path, format=np.asarray(["dense"]),
                                adjacency=np.asarray(adj, dtype=np.float64))
        return {"json": os.path.abspath(json_path), "npz": os.path.abspath(npz_path)}

    @staticmethod
    def load(path: str) -> "OrganelleNetwork":
        """读回 ``save`` 写出的网络（``path`` 指向 ``.json`` 或目录）。"""
        if os.path.isdir(path):
            jsons = sorted(glob.glob(os.path.join(path, "*.json")))
            if not jsons:
                raise FileNotFoundError(f"{path} 下没有网络 json")
            json_path = jsons[0]
        else:
            json_path = path
        if not json_path.lower().endswith(".json"):
            json_path = json_path + ".json"
        npz_path = json_path[:-5] + ".npz"

        with open(json_path, "r", encoding="utf-8") as fh:
            d = json.load(fh)

        adjacency = None
        if os.path.exists(npz_path):
            with np.load(npz_path, allow_pickle=False) as z:
                fmt = str(np.asarray(z["format"]).ravel()[0])
                if fmt == "csr":
                    shape = tuple(int(v) for v in np.asarray(z["shape"]).ravel())
                    adjacency = sparse.csr_matrix(
                        (z["data"], z["indices"], z["indptr"]), shape=shape
                    )
                else:
                    adjacency = np.asarray(z["adjacency"], dtype=np.float64)
        return OrganelleNetwork.from_dict(d, adjacency=adjacency)


def _adjacency_from_edges(edge_list: Sequence[Sequence[int]], n: int):
    """由边表构造对称 CSR 邻接矩阵。"""
    if n <= 0 or not edge_list:
        return sparse.csr_matrix((max(n, 0), max(n, 0)), dtype=np.int8)
    rows = np.asarray([e[0] for e in edge_list] + [e[1] for e in edge_list], dtype=np.int64)
    cols = np.asarray([e[1] for e in edge_list] + [e[0] for e in edge_list], dtype=np.int64)
    data = np.ones(rows.size, dtype=np.int8)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


# ==========================================================================
# 4. 邻接判据
# ==========================================================================

def _pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(Na, 2) × (Nb, 2) → (Na, Nb) 欧氏距离。"""
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def _nn_distances(centroids: np.ndarray) -> np.ndarray:
    """每个点到**其它**点的最近距离 (N,)；N<2 时返回空数组。"""
    n = len(centroids)
    if n < 2:
        return np.zeros(0, dtype=np.float64)
    d = _pairwise_dist(centroids, centroids)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def build_delaunay(centroids: np.ndarray) -> Tuple[List[List[int]], List[str]]:
    """Delaunay 三角剖分边集。

    为什么是它：在空间点模式分析里，Delaunay 图是"自然最近邻图"的标准选择，
    它的边集是所有"互相邻近"点对的一个无参数、确定性的定义
    （Gabriel 图、相对邻域图、最小生成树都是它的子图）。

    **退化处理（显式报告，不静默）**：
    * N < 3 → 空边集，warn ``"点数 < 3，Delaunay 无定义"``；
    * 所有点共线 → Qhull 抛 ``QhullError`` → 空边集 + warn。

    Returns
    -------
    (edges, warnings)
    """
    n = len(centroids)
    warns: List[str] = []
    if n < 3:
        return [], [f"点数 N={n} < 3，Delaunay 三角剖分无定义，返回空边集"]
    try:
        tri = Delaunay(centroids)
    except QhullError as exc:
        return [], [f"Delaunay 退化（点共线或重合）：{exc}；返回空边集"]

    pairs = set()
    for simplex in tri.simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                i, j = int(simplex[a]), int(simplex[b])
                pairs.add((min(i, j), max(i, j)))
    edges = [[i, j] for i, j in sorted(pairs)]
    if n > 3 and not edges:
        warns.append("Delaunay 未产生任何边（所有点重合？）")
    return edges, warns


def build_knn(centroids: np.ndarray, k: int = 6,
              max_dist: Optional[float] = None) -> Tuple[List[List[int]], List[str]]:
    """k 近邻图（对称化：只要 i 在 j 的 kNN 或 j 在 i 的 kNN 中就建边）。

    Parameters
    ----------
    k
        邻居数。**k 必须显式记录在 params 里**，因为图密度随 k 线性增长。
    max_dist
        可选距离上限（单位同 ``centroids``）；超过该距离的 kNN 边会被丢掉，
        这会把 kNN 图变成"带截断的 kNN 图"，更接近真实接触的稀疏性。

    Returns
    -------
    (edges, warnings)
        ``warnings`` 在 ``k >= N`` 时会说明"k 被截断成 N-1，图变成完全图"。
    """
    n = len(centroids)
    warns: List[str] = []
    if n < 2:
        return [], [f"点数 N={n} < 2，kNN 图无定义"]
    k_eff = int(min(k, n - 1))
    if k_eff != k:
        warns.append(f"k={k} ≥ N-1={n - 1}，已截断为 k={k_eff}（图将接近完全图）")

    d = _pairwise_dist(centroids, centroids)
    np.fill_diagonal(d, np.inf)
    pairs = set()
    for i in range(n):
        order = np.argsort(d[i], kind="stable")[:k_eff]
        for j in order:
            j = int(j)
            if max_dist is not None and d[i, j] > max_dist:
                continue
            pairs.add((min(i, j), max(i, j)))
    return [[i, j] for i, j in sorted(pairs)], warns


def build_radius(centroids: np.ndarray, radius: float) -> Tuple[List[List[int]], List[str]]:
    """固定半径图：质心距离 ≤ ``radius`` 即连边。

    阈值论证见模块文档字符串。调用方往往不给阈值：``build_network`` 会用
    ``median_nn * radius_factor`` 自动导出，并把**回填后的实际阈值**写进
    ``params['radius_used']``，同时报告图密度，让坏阈值暴露出来。

    Returns
    -------
    (edges, warnings)
        ``warnings`` 会就"图几乎空"或"图几乎完全"给出明确提示。
    """
    n = len(centroids)
    warns: List[str] = []
    if n < 2:
        return [], [f"点数 N={n} < 2，半径图无定义"]
    if radius is None or not np.isfinite(radius) or radius <= 0:
        return [], [f"半径非法 ({radius})，返回空边集"]

    d = _pairwise_dist(centroids, centroids)
    iu = np.triu_indices(n, k=1)
    sel = d[iu] <= radius
    edges = [[int(i), int(j)] for i, j in zip(iu[0][sel], iu[1][sel])]

    dens = 2.0 * len(edges) / (n * (n - 1))
    if dens < 0.05:
        warns.append(f"半径 {radius:.4g} 下密度仅 {dens:.4f}，图接近空图，阈值可能过小")
    elif dens > 0.9:
        warns.append(f"半径 {radius:.4g} 下密度达 {dens:.4f}，图接近完全图，阈值可能过大")
    return edges, warns


def build_gabriel(centroids: np.ndarray) -> Tuple[List[List[int]], List[str]]:
    """Gabriel 图：Delaunay 子集，边 ij 存在 ⟺ 以 ij 为直径的闭圆内无其它点。

    实现：先取 Delaunay 边（Gabriel 图必是其子图，这保证 O(N log N) 而不是 O(N³)），
    再对每条候选边检查空圆条件。

    Returns
    -------
    (edges, warnings)
    """
    n = len(centroids)
    if n < 3:
        return [], [f"点数 N={n} < 3，Gabriel 图无定义（需先有 Delaunay）"]
    cand, warns = build_delaunay(centroids)
    if not cand:
        return [], warns + ["Delaunay 无边，Gabriel 图亦为空"]

    edges: List[List[int]] = []
    for i, j in cand:
        c = 0.5 * (centroids[i] + centroids[j])
        r2 = float(np.sum((centroids[i] - c) ** 2)) + 1e-12
        others = np.ones(n, dtype=bool)
        others[[i, j]] = False
        d2 = np.sum((centroids[others] - c) ** 2, axis=1)
        if not np.any(d2 <= r2):
            edges.append([i, j])
    return edges, warns


def boundary_pixels(label_map: np.ndarray) -> np.ndarray:
    """返回布尔边界掩码：3×3 邻域内标签不全相同的像素即边界。

    实现用两次形态学极值滤波（O(HW)，与区域数无关），比"逐区域二值腐蚀"
    快一个数量级——后者在 2048×2115 的图上要跑 N 次全图操作。
    """
    from scipy import ndimage as ndi

    own = label_map
    maxf = ndi.maximum_filter(label_map, size=3, mode="constant", cval=-1)
    minf = ndi.minimum_filter(label_map, size=3, mode="constant", cval=-1)
    # 内部像素 ⟺ 邻域内最大标签都等于自己（即 3×3 全同）
    return (maxf != own) | (minf != own)


def boundary_points_by_label(label_map: np.ndarray) -> Dict[int, np.ndarray]:
    """一次遍历取出所有区域的边界像素坐标。

    Returns
    -------
    dict
        ``{region_id: (P, 2) float64 数组，列为 (x, y)}``。
        只包含像素数 > 0 的标签；若某区域整块都是边界（极小区域），
        则退化使用该区域的全部像素。

    这是把"逐区域全图布尔运算"（N 次 O(HW)）压成一次 O(HW) 的关键。
    """
    bnd = boundary_pixels(label_map)
    rows, cols = np.nonzero(bnd)
    labs = label_map[rows, cols]
    out: Dict[int, np.ndarray] = {}
    if rows.size:
        order = np.argsort(labs, kind="stable")
        labs_s, rows_s, cols_s = labs[order], rows[order], cols[order]
        uniq, starts = np.unique(labs_s, return_index=True)
        bounds = list(starts) + [labs_s.size]
        for k, lab in enumerate(uniq):
            lab = int(lab)
            if lab <= 0:
                continue
            sl = slice(bounds[k], bounds[k + 1])
            out[lab] = np.column_stack(
                [cols_s[sl].astype(np.float64), rows_s[sl].astype(np.float64)]
            )

    # 兜底：整块都是边界的极小区域没被 3×3 判据标出来时，用全像素
    for lab in np.unique(label_map):
        lab = int(lab)
        if lab <= 0 or lab in out:
            continue
        rr, cc = np.nonzero(label_map == lab)
        if rr.size:
            out[lab] = np.column_stack([cc.astype(np.float64), rr.astype(np.float64)])
    return out


def boundary_distance_matrices(
    label_map: np.ndarray,
    id_to_index: Optional[Dict[int, int]] = None,
    points: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """区域间的**精确**边界-边界最小像素距离矩阵。

    原理：两个区域的最近点必然都在各自的边界上（区域内部点永远不是最近点），
    因此在边界像素集合上建 ``cKDTree`` 查询即可得到精确的最小平面上距离
    （像素中心之间的欧氏距离，单位为"像素"）。

    Parameters
    ----------
    points
        可选的预计算边界点（``boundary_points_by_label`` 的输出），
        避免重复做 O(HW) 的边界提取。

    Returns
    -------
    (dist, ids)
        ``dist`` 为 (K, K) 对称矩阵，``dist[a, b]`` = 区域 ``ids[a]`` 与
        ``ids[b]`` 边界之间的最小距离（像素）；对角线置 0。
        ``ids`` 为参与计算的 ``region_id`` 列表（按升序，
        **只包含在 ``id_to_index`` 中出现的区域**）。

    复杂度
    ------
    每个区域一次 KD 树构建（O(P log P)）+ 每对区域一次 query。
    本数据集每图 N ≤ 23，代价可忽略。
    """
    if id_to_index is None:
        id_to_index = {int(v): int(v) - 1 for v in np.unique(label_map) if v > 0}
    if points is None:
        points = boundary_points_by_label(label_map)

    ids: List[int] = sorted(int(v) for v in points if int(v) in id_to_index)
    k = len(ids)
    dist = np.zeros((k, k), dtype=np.float64)
    trees = {lab: cKDTree(points[lab]) for lab in ids}
    for a in range(k):
        for b in range(a + 1, k):
            d, _ = trees[ids[b]].query(points[ids[a]], k=1)
            v = float(np.min(d))
            dist[a, b] = dist[b, a] = v
    return dist, ids


@dataclass
class ContactGeometry:
    """一张图的接触几何（光栅化标签图 + 边界点 + 精确距离矩阵）。

    对同一张图做**阈值扫描**时，真正昂贵的只有这一步（O(HW) 光栅化 +
    O(K²) KD 树查询），所以把它单独缓存一次，再用不同阈值反复取边。
    这样 ``boundary_contact`` 的敏感性分析从"每阈值重建一次"的
    O(阈值数 × 建图代价) 降到 O(建图代价 + 阈值数 × 边数)。
    """

    image: str
    label_map: np.ndarray
    points: Dict[int, np.ndarray]
    ids: List[int]
    dist_px: np.ndarray            # (K, K)
    id_to_index: Dict[int, int]

    def pair_distances(self) -> np.ndarray:
        """上三角的成对最小距离（像素），用于阈值取值分析与报告。"""
        k = len(self.ids)
        if k < 2:
            return np.zeros(0, dtype=np.float64)
        iu = np.triu_indices(k, k=1)
        return self.dist_px[iu]

    def nn_distances(self) -> np.ndarray:
        """每个区域到**最近**区域的边界距离（像素），长度 K。"""
        k = len(self.ids)
        if k < 2:
            return np.zeros(0, dtype=np.float64)
        d = self.dist_px.copy()
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1)


def prepare_contact_geometry(table: RegionTable,
                             label_map: Optional[np.ndarray] = None) -> ContactGeometry:
    """计算并缓存一张图的接触几何（供 ``boundary_contact`` 复用）。"""
    if label_map is None:
        label_map = rasterize_label_map(table)
    id_to_index = {int(r.region_id): idx for idx, r in enumerate(table.regions)}
    points = boundary_points_by_label(label_map)
    dist, ids = boundary_distance_matrices(label_map, id_to_index, points=points)
    return ContactGeometry(
        image=table.image, label_map=label_map, points=points,
        ids=ids, dist_px=dist, id_to_index=id_to_index,
    )


def build_boundary_contact(
    label_map: np.ndarray,
    min_contact_px: float = DEFAULT_CONTACT_TOL_PX,
    um_per_px: Optional[float] = None,
    id_to_index: Optional[Dict[int, int]] = None,
    geometry: Optional[ContactGeometry] = None,
) -> Tuple[List[List[int]], List[Dict[str, Any]], List[str]]:
    """**默认判据。** 轮廓（边界像素）最小距离 ≤ 阈值 ⟺ 一条边。

    为什么这是最贴近生物学的代理量
    ------------------------------
    膜接触位点（MCS）的定义就是两张膜之间的距离落在 10–30 nm；
    质心之间的远近与"是否接触"没有单调关系（两个巨大的、紧贴的细胞器
    质心可能相距很远，而两个远处的碎片质心可能很近）。
    因此只有**边界到边界**的距离才在几何上对应"接触"。

    算法（精确，不是近似）
    ----------------------
    1. 由光栅化标签图取**边界像素**（3×3 邻域标签不全同的像素）；
    2. 最近点必在边界上，故用 ``cKDTree`` 在边界像素集合上查询即可得到
       区域间的精确最小像素距离；
    3. 距离矩阵一次算好（``boundary_distance_matrices``），再按阈值取边。

    权重 = 接触长度
    ---------------
    ``contact_px`` 定义为：区域 i 的边界像素中，其到区域 j 边界像素的
    最近距离 ≤ 阈值的**像素个数**；再与 j 对 i 的同一量取 ``max`` 做对称化。
    对一条平直接口来说它约等于接口长度（像素）。这是"接触面积/长度"的
    一维代理，不是真实膜接触面积。

    Parameters
    ----------
    label_map
        (H, W) int32 标签图，0 = 背景。
    min_contact_px
        接触阈值（像素）。**注意：尽管名字里有 "min"，它是一个上界**——
        两点距离 ≤ 该值即判为接触。命名沿用项目约定的 CLI 参数名。
    um_per_px
        标定；给出时才计算 ``contact_um`` / ``dist_um``。
    id_to_index
        ``region_id → 节点下标`` 的映射。缺省时假定 ``region_id`` 为
        1..N 连续编号（``extract_regions.py`` 的重编号保证如此），
        即下标 = ``region_id - 1``。显式传入可避免对编号方式的隐含假设。
    geometry
        预计算的 ``ContactGeometry``；给出时直接复用其边界点与距离矩阵，
        避免重复的 O(HW) 光栅化与 O(K²) KD 查询（阈值扫描时必用）。

    Returns
    -------
    (edges, edge_attrs, warnings)
        ``edges`` 为 **0-based 节点下标**对。
    """
    if id_to_index is None:
        id_to_index = {int(lab): int(lab) - 1 for lab in np.unique(label_map) if lab > 0}

    if geometry is not None:
        points = geometry.points
        dist_px, ids = geometry.dist_px, geometry.ids
    else:
        points = boundary_points_by_label(label_map)
        dist_px, ids = boundary_distance_matrices(label_map, id_to_index, points=points)
    tol = float(min_contact_px)
    if len(ids) < 2:
        return [], [], [f"可参与接触计算的区域数 {len(ids)} < 2，无法建接触边"]

    warned: List[str] = []
    edges: List[List[int]] = []
    attrs: List[Dict[str, Any]] = []

    trees: Dict[int, cKDTree] = {}

    for a in range(len(ids)):
        la = ids[a]
        for b in range(a + 1, len(ids)):
            lb = ids[b]
            dmin = float(dist_px[a, b])
            if dmin > tol:
                continue
            # 接触长度（像素）：边界像素中"到对方边界 ≤ 阈值"的个数
            for _lab in (la, lb):
                if _lab not in trees:
                    trees[_lab] = cKDTree(points[_lab])
            dab, _ = trees[lb].query(points[la], k=1)
            dba, _ = trees[la].query(points[lb], k=1)
            contact_px = max(int(np.count_nonzero(dab <= tol)),
                             int(np.count_nonzero(dba <= tol)))

            ia, ib = id_to_index.get(int(la)), id_to_index.get(int(lb))
            if ia is None or ib is None:
                warned.append(
                    f"区域 region_id={la if ia is None else lb} 未出现在节点表中，"
                    "该接触边被丢弃"
                )
                continue
            edges.append([int(ia), int(ib)])
            attrs.append({
                "region_a": int(la),
                "region_b": int(lb),
                "dist_px": dmin,
                "dist_um": (dmin * um_per_px) if um_per_px else None,
                "contact_px": contact_px,
                "contact_um": (contact_px * um_per_px) if um_per_px else None,
            })

    if not edges:
        min_gap = float(np.min(dist_px[~np.eye(len(ids), dtype=bool)])) if len(ids) > 1 else float("nan")
        warned.append(
            f"接触阈值 {tol:g} px 下没有任何区域对接触（本图最小区域间隙 "
            f"{min_gap:.1f} px）；请调大 --min-contact-px 或使用 "
            f"--contact-tol-percentile 由数据导出阈值"
        )
    return edges, attrs, warned


# ==========================================================================
# 5. 统一入口
# ==========================================================================

def build_network(
    table: RegionTable,
    method: str = DEFAULT_METHOD,
    *,
    min_contact_px: float = DEFAULT_CONTACT_TOL_PX,
    k: int = 6,
    max_dist: Optional[float] = None,
    radius: Optional[float] = None,
    radius_factor: float = 1.5,
    label_map: Optional[np.ndarray] = None,
    contact_geometry: Optional[ContactGeometry] = None,
    node_type_mode: str = "annotation",
    sparse_adjacency: bool = True,
    distance_unit: str = "auto",
) -> OrganelleNetwork:
    """按指定判据把一张图的区域连成 ``OrganelleNetwork``。

    Parameters
    ----------
    table
        ``RegionTable``。
    method
        ``METHODS`` 之一。
    min_contact_px
        ``boundary_contact`` 的接触阈值（像素）。
    k, max_dist
        ``knn`` 参数。``max_dist`` 单位与 ``centroids`` 相同（有标定即 µm）。
    radius, radius_factor
        ``radius`` 参数。``radius`` 给定时直接用（单位同 ``centroids``）；
        否则用 ``median_nn * radius_factor`` 自动导出
        （``median_nn`` = 全数据集意义上本图的最近邻距离中位数）。
    label_map
        已光栅化的标签图；``boundary_contact`` 需要，``None`` 时自动光栅化。
    node_type_mode
        见 ``RegionTable.node_types``。
    distance_unit
        ``'auto'``（有标定用 µm，否则 px）/ ``'um'`` / ``'px'``。
        **注意**：Delaunay / Gabriel / kNN 三种判据对全局尺度不变，
        因此该选择**不影响**它们的拓扑；它只影响 ``radius`` 阈值的单位、
        以及报告出来的绝对距离。本数据集不同图的标定相差可达 5 倍
        （见 REPORT.md），所以跨图比较绝对距离时必须固定为 ``'px'``。

    Returns
    -------
    OrganelleNetwork
        ``params`` 内含实际使用的阈值、``median_nn``、``density`` 与
        光栅化/退化告警。

    Raises
    ------
    ValueError
        ``method`` 不在 ``METHODS`` 中，或 ``distance_unit`` 非法。
    RuntimeError
        ``boundary_contact`` 需要标签图但无法光栅化时。
    """
    if method not in METHODS:
        raise ValueError(f"未知方法 {method!r}；可选：{list(METHODS)}")
    if distance_unit not in ("auto", "um", "px"):
        raise ValueError(f"distance_unit 必须是 auto/um/px，得到 {distance_unit!r}")

    warns: List[str] = []
    n = table.n

    centroids_um = table.centroids_um()
    want_um = distance_unit == "um" or (distance_unit == "auto" and centroids_um is not None)
    if want_um and centroids_um is not None:
        centroids = centroids_um
        unit = "um"
    else:
        centroids = table.centroids_px()
        unit = "px"
        if centroids_um is None:
            warns.append("该图标定不可信（scale.ok=False 或 um_per_px 缺失），"
                         "全部距离以像素为单位；跨图比较距离时务必注意")
        elif distance_unit == "px":
            warns.append("已按 distance_unit='px' 强制使用像素坐标；"
                         "µm 标定仍记录在 edge_attrs.dist_um 中供参考")

    params: Dict[str, Any] = {
        "method": method,
        "coordinate_unit": unit,
        "distance_unit_request": distance_unit,
        "um_per_px": table.um_per_px,
        "n_regions_in_table": n,
    }

    # --- 依据方法算边 ---------------------------------------------------
    edge_attrs: List[Dict[str, Any]] = []
    id_to_index = {int(r.region_id): idx for idx, r in enumerate(table.regions)}
    if method == "boundary_contact":
        if contact_geometry is not None:
            label_map = contact_geometry.label_map
        elif label_map is None:
            label_map = rasterize_label_map(table)
        edges, edge_attrs, w = build_boundary_contact(
            label_map, min_contact_px=min_contact_px, um_per_px=table.um_per_px,
            id_to_index=id_to_index, geometry=contact_geometry,
        )
        warns += w
        params["min_contact_px"] = float(min_contact_px)
        params["contact_criterion"] = (
            "两条区域轮廓（边界像素集合）的最小欧氏像素距离 ≤ min_contact_px"
        )
    elif method == "delaunay":
        edges, w = build_delaunay(centroids)
        warns += w
    elif method == "knn":
        edges, w = build_knn(centroids, k=k, max_dist=max_dist)
        warns += w
        params["k"] = int(k)
        params["max_dist"] = None if max_dist is None else float(max_dist)
    elif method == "gabriel":
        edges, w = build_gabriel(centroids)
        warns += w
    else:  # radius
        nn = _nn_distances(centroids)
        median_nn = float(np.median(nn)) if nn.size else float("nan")
        if radius is None:
            radius_used = median_nn * float(radius_factor) if nn.size else float("nan")
            params["radius_rule"] = (
                f"自动导出：median_nn({median_nn:.4g} {unit}) × factor({radius_factor:g})"
            )
        else:
            radius_used = float(radius)
            params["radius_rule"] = "用户显式指定"
        edges, w = build_radius(centroids, radius_used)
        warns += w
        params["median_nn"] = median_nn
        params["radius_factor"] = float(radius_factor)
        params["radius_used"] = float(radius_used)

    # --- 距离与权重 -----------------------------------------------------
    if method != "boundary_contact":
        d_px = _pairwise_dist(table.centroids_px(), table.centroids_px())
        d_unit = _pairwise_dist(centroids, centroids)
        for i, j in edges:
            dist_px = float(d_px[i, j])
            dist_unit = float(d_unit[i, j])
            edge_attrs.append({
                "dist_px": dist_px,
                "dist_um": (dist_px * table.um_per_px) if table.um_per_px else None,
                "dist": dist_unit,
                "weight_unit": f"1/{unit}",
            })

    for a in edge_attrs:
        if method == "boundary_contact":
            # 权重 = 接触长度；有标定用 µm，否则用像素
            if a.get("contact_um") is not None:
                a["weight"] = float(a["contact_um"])
                a["weight_unit"] = "um"
            else:
                a["weight"] = float(a["contact_px"])
                a["weight_unit"] = "px"
        else:
            d = float(a.get("dist", a.get("dist_px", 0.0)))
            a["weight"] = float(1.0 / d) if d > 0 else float("inf")
            if d == 0:
                warns.append("存在距离为 0 的边对（区域重合），其权重记为 inf（截断为 1e12 参与计算）")
                a["weight"] = 1e12

    edge_list = [[int(i), int(j)] for i, j in edges]
    weights = np.asarray([a["weight"] for a in edge_attrs], dtype=np.float64)

    # --- 邻接矩阵 -------------------------------------------------------
    if sparse_adjacency:
        adjacency = _adjacency_from_edges(edge_list, n)
    else:
        adjacency = np.zeros((n, n), dtype=np.int8)
        for i, j in edge_list:
            adjacency[i, j] = adjacency[j, i] = 1

    types, type_source = table.node_types(mode=node_type_mode)
    areas = None
    if table.um_per_px is not None:
        areas = np.asarray([r.area_um2 if r.area_um2 is not None else np.nan
                            for r in table.regions], dtype=np.float64)

    net = OrganelleNetwork(
        image=table.image,
        node_ids=[int(r.region_id) for r in table.regions],
        centroids=centroids,
        centroids_px=table.centroids_px(),
        areas_um2=areas,
        node_types=types,
        node_type_source=type_source,
        adjacency=adjacency,
        edge_list=edge_list,
        edge_weights=weights,
        edge_attrs=edge_attrs,
        method=method,
        params=params,
        coordinate_unit=unit,
        image_shape=table.image_shape,
        geometry_note=table.geometry_note,
        warnings=warns,
    )

    # 回填密度：阈值好坏的第一诊断量
    net.params["density"] = net.density()
    net.params["n_edges"] = net.n_edges
    net.params["n_nodes"] = net.n_nodes
    if net.n_nodes >= 2:
        d = net.params["density"]
        if net.n_edges == 0:
            net.warnings.append("建成的图没有任何边；该图的网络指标全部不可用")
        elif d < 0.05:
            net.warnings.append(f"图密度仅 {d:.4f}，图非常稀疏，路径/模块度类指标极不稳定")
        elif d > 0.9:
            net.warnings.append(f"图密度高达 {d:.4f}，图接近完全图，拓扑指标失去区分度")
    return net


def build_contact_network_for_table(
    table: RegionTable, min_contact_px: float = DEFAULT_CONTACT_TOL_PX, **kw
) -> OrganelleNetwork:
    """便捷封装：``method='boundary_contact'``。"""
    return build_network(table, method="boundary_contact",
                         min_contact_px=min_contact_px, **kw)
