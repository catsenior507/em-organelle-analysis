#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
morpho.features —— 区域特征的单一定义源（single source of truth）

背景
----
原 MATLAB 流水线（code/analyze_annotated_regions.m 等）存在三类根本性缺陷：

1. **分割反了**：它对绿色掩码取反（``~green_mask``）后做连通域，得到的是背景。
   绿色像素是手绘的细胞器**轮廓线**，真正的区域是轮廓的**内部**。
2. **像素表被截断**：``region_rows`` 被 ``min(length(...), 1000)`` 截断，并在 JSON 里
   写入字面量 ``...``，导致 91/94 个文件不是合法 JSON。
3. **xy 互换**：MATLAB 的 ``find`` 返回 ``[row, col]``，下游却按 ``x=col, y=row`` 读。

本模块提供唯一正确的实现：输入必须是 ``(x, y)`` 语义的像素集合。

坐标约定
--------
本模块中所有坐标均为 **图像坐标，1-based，形如 (x, y) = (列, 行)**，与 MATLAB 的
``(col, row)`` 一致。Numpy 数组按 ``arr[row, col]`` 索引，因此访问像素时使用
``arr[int(y) - 1, int(x) - 1]``。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# 常量：特征定义
# --------------------------------------------------------------------------

#: 14 维"紧凑"特征，顺序与原流水线保持一致，便于对照与回归。
#: 注意：全部为无量纲或原始像素量纲，标定（µm）由调用方另行换算。
COMPACT_FEATURE_NAMES: Tuple[str, ...] = (
    "min_x",          # 1
    "max_x",          # 2
    "min_y",          # 3
    "max_y",          # 4
    "width",          # 5
    "height",         # 6
    "center_x",       # 7
    "center_y",       # 8
    "major_axis",     # 9
    "minor_axis",     # 10
    "roughness",      # 11
    "complexity",     # 12
    "gray_mean",      # 13
    "gray_std",       # 14
)

#: 扩展特征：真正有形态学意义的量，建议作为 Transformer 的回归目标。
EXTENDED_FEATURE_NAMES: Tuple[str, ...] = (
    "area",              # 区域内像素数
    "perimeter",         # 8-连通边界长度（像素）
    "solidity",          # area / convex_hull_area
    "eccentricity",      # 椭圆离心率 [0,1)
    "equivalent_diameter",  # 4*area/perimeter，比 sqrt(area) 更稳健的尺度量
    "orientation",       # 长轴方向，弧度，(-pi/2, pi/2]
    "extent",            # area / (bbox_w * bbox_h)
    "gray_mean",
    "gray_std",
    "gray_min",
    "gray_max",
    "cv_gray",           # gray_std / gray_mean，无量纲对比度
    "label_ratio",       # 黄色标注像素 / area，用于确认区域是否被真正标注
)

FEATURE_SCHEMA_VERSION = "2.0.0"


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------

@dataclass
class RegionFeatures:
    """单个区域的形态 + 灰度特征。

    坐标均为 1-based ``(x, y)``。``contour`` 保存简化后的多边形（``[[x, y], ...]``），
    它无损地保留了形状信息，且体积远小于像素列表——这是替换原 1000 点截断方案的关键。
    """

    region_id: int
    # --- 位置 ---
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width: float
    height: float
    center_x: float
    center_y: float
    # --- 形状 ---
    area: float
    perimeter: float
    major_axis: float
    minor_axis: float
    orientation: float
    solidity: float
    eccentricity: float
    extent: float
    equivalent_diameter: float
    # --- 灰度 ---
    gray_mean: float
    gray_std: float
    gray_min: float
    gray_max: float
    cv_gray: float
    # --- 兼容/诊断 ---
    roughness: float
    complexity: float
    label_pixels: int
    label_ratio: float
    recognized_text: str = ""
    organelle_type: str = ""
    contour: List[List[float]] = field(default_factory=list)
    # 质量标记：该区域是否可信（非背景、非碎片、被标注）
    is_labeled: bool = False
    quality_flag: str = "ok"

    # -- 便利视图 ---------------------------------------------------------
    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        """DETR 用的 ``(x1, y1, x2, y2)``，注意是 xy 而非 yx。"""
        return (self.min_x, self.min_y, self.max_x, self.max_y)

    def compact_vector(self) -> np.ndarray:
        """14 维紧凑向量，顺序与 ``COMPACT_FEATURE_NAMES`` 一致。"""
        return np.asarray(
            [
                self.min_x, self.max_x, self.min_y, self.max_y,
                self.width, self.height,
                self.center_x, self.center_y,
                self.major_axis, self.minor_axis,
                self.roughness, self.complexity,
                self.gray_mean, self.gray_std,
            ],
            dtype=np.float64,
        )

    def extended_vector(self) -> np.ndarray:
        """扩展特征向量，顺序与 ``EXTENDED_FEATURE_NAMES`` 一致。"""
        return np.asarray(
            [
                self.area, self.perimeter, self.solidity, self.eccentricity,
                self.equivalent_diameter, self.orientation, self.extent,
                self.gray_mean, self.gray_std, self.gray_min, self.gray_max,
                self.cv_gray, self.label_ratio,
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScaleCalibration:
    """像素 → 微米标定。

    ``ok=False`` 表示标定不可信；此时 ``um_per_px`` 为 ``None``，
    调用方**必须显式处理**，而不是像原流水线那样静默代入 0.1。
    """

    um_per_px: Optional[float]
    bar_px: Optional[float] = None
    bar_value: Optional[float] = None
    bar_unit: Optional[str] = None
    ok: bool = False
    source: str = "none"
    note: str = ""

    def to_um(self, px: float) -> Optional[float]:
        if not self.ok or self.um_per_px is None:
            return None
        return float(px) * self.um_per_px

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# 特征计算
# --------------------------------------------------------------------------

def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """返回 1-based 图像坐标下的 ``(min_x, max_x, min_y, max_y)``。"""
    rows, cols = np.nonzero(mask)
    return (
        int(cols.min()) + 1,   # min_x (列)
        int(cols.max()) + 1,   # max_x
        int(rows.min()) + 1,   # min_y (行)
        int(rows.max()) + 1,   # max_y
    )


def _perimeter_pixels(mask: np.ndarray) -> float:
    """8-连通下的边界**像素个数**。

    注意：这是边界像素数，不是几何周长（原 ``simple_region_features.m:187``
    混用了这两个概念，导致 roughness 近似常数）。几何周长由轮廓给出。
    """
    from scipy import ndimage as ndi

    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return float(np.count_nonzero(mask & ~eroded))


def compute_region_features(
    mask: np.ndarray,
    gray: Optional[np.ndarray] = None,
    label_mask: Optional[np.ndarray] = None,
    region_id: int = 0,
    recognized_text: str = "",
    organelle_type: str = "",
    contour_epsilon_px: float = 1.5,
) -> RegionFeatures:
    """从单个区域的布尔掩码计算特征。

    Parameters
    ----------
    mask
        形状 ``(H, W)`` 的布尔数组，``True`` 表示属于该区域。
    gray
        可选，同形状灰度图（任意数值类型），用于灰度统计。
    label_mask
        可选，同形状布尔数组，标记黄色标注文字像素。
    contour_epsilon_px
        ``cv2.approxPolyDP`` 的简化容差；越大轮廓点越少。1.5 px 通常能把
        数千点压到几十点而肉眼无差别。
    """
    from scipy import ndimage as ndi

    if mask.dtype != bool:
        mask = mask.astype(bool)
    area = int(np.count_nonzero(mask))
    if area == 0:
        raise ValueError("空掩码无法计算特征")

    min_x, max_x, min_y, max_y = _bbox_from_mask(mask)
    width = float(max_x - min_x + 1)
    height = float(max_y - min_y + 1)
    center_x = float(np.nonzero(mask)[1].mean()) + 1.0
    center_y = float(np.nonzero(mask)[0].mean()) + 1.0

    # --- 轮廓：形状信息的无损紧凑表示 ---
    contour_pts = _extract_contour(mask, contour_epsilon_px)
    perimeter = _contour_perimeter(contour_pts)

    # --- 二阶矩 / 轴长 ---
    major_axis, minor_axis, orientation = _axes_from_mask(mask)
    eccentricity = _eccentricity(major_axis, minor_axis)

    # --- 凸包与 solidity ---
    solidity = _solidity(contour_pts, area)

    extent = area / (width * height) if width * height > 0 else 0.0
    equiv_diameter = (4.0 * area / perimeter) if perimeter > 0 else 0.0

    # --- 灰度 ---
    if gray is not None:
        vals = np.asarray(gray)[mask].astype(np.float64)
        gray_mean = float(vals.mean())
        gray_std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        gray_min = float(vals.min())
        gray_max = float(vals.max())
        cv_gray = gray_std / gray_mean if gray_mean > 1e-12 else 0.0
    else:
        gray_mean = gray_std = gray_min = gray_max = cv_gray = 0.0

    # --- 标注像素 ---
    if label_mask is not None:
        label_pixels = int(np.count_nonzero(mask & label_mask.astype(bool)))
    else:
        label_pixels = 0
    label_ratio = label_pixels / area if area > 0 else 0.0

    # --- 旧口径的两个量，保留以便与历史结果对照（但不建议继续使用）---
    # roughness：周长 / sqrt(面积)，形状复杂度指标
    roughness = perimeter / np.sqrt(area) if area > 0 else 0.0
    # complexity：区域在行方向投影的频谱"主频"，原实现有 off-by-one 且无物理意义，
    # 这里保留同样口径但修正索引，仅作对照。
    complexity = _projection_main_frequency(mask)

    # --- 质量标记 ---
    flag = "ok"
    if area < 30:
        flag = "too_small"
    elif not label_mask is None and label_pixels == 0:
        flag = "unlabeled"
    elif solidity < 0.6 and extent < 0.4:
        flag = "irregular"

    # 极端细长的区域几乎必然是分割/过滤残留，明确标出来而不是静默使用
    if minor_axis > 0 and major_axis / max(minor_axis, 1e-9) > 12:
        flag = "elongated"

    return RegionFeatures(
        region_id=region_id,
        min_x=float(min_x), max_x=float(max_x),
        min_y=float(min_y), max_y=float(max_y),
        width=width, height=height,
        center_x=center_x, center_y=center_y,
        area=float(area),
        perimeter=float(perimeter),
        major_axis=major_axis, minor_axis=minor_axis,
        orientation=orientation, solidity=solidity,
        eccentricity=eccentricity, extent=extent,
        equivalent_diameter=equiv_diameter,
        gray_mean=gray_mean, gray_std=gray_std,
        gray_min=gray_min, gray_max=gray_max, cv_gray=cv_gray,
        roughness=float(roughness), complexity=float(complexity),
        label_pixels=label_pixels, label_ratio=label_ratio,
        recognized_text=recognized_text, organelle_type=organelle_type,
        contour=contour_pts,
        is_labeled=label_pixels > 0,
        quality_flag=flag,
    )


def _extract_contour(mask: np.ndarray, epsilon: float) -> List[List[float]]:
    """提取并简化外轮廓，返回 1-based ``[[x, y], ...]``。"""
    try:
        import cv2
    except ImportError:  # 退化路径：用边界像素代替
        from scipy import ndimage as ndi

        eroded = ndi.binary_erosion(mask, np.ones((3, 3), bool), border_value=0)
        rows, cols = np.nonzero(mask & ~eroded)
        return [[float(c) + 1.0, float(r) + 1.0] for r, c in zip(rows, cols)]

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    if epsilon > 0:
        largest = cv2.approxPolyDP(largest, epsilon, True)
    # cv2 给出 (x, y)，0-based → 转 1-based
    return [[float(p[0][0]) + 1.0, float(p[0][1]) + 1.0] for p in largest]


def _contour_perimeter(contour: List[List[float]]) -> float:
    """闭合多边形的几何周长。"""
    if len(contour) < 2:
        return 0.0
    pts = np.asarray(contour, dtype=np.float64)
    diffs = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())


def _axes_from_mask(mask: np.ndarray) -> Tuple[float, float, float]:
    """用二阶中心矩计算长轴、短轴（2*sqrt(特征值)）与方向（弧度）。"""
    rows, cols = np.nonzero(mask)
    if rows.size < 2:
        return 0.0, 0.0, 0.0
    # 注意：这里把 x 当第一维来做协方差，保证"major_axis 对应图像上的长边"
    pts = np.column_stack([cols.astype(np.float64), rows.astype(np.float64)])
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return 0.0, 0.0, 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)          # eigh：对称矩阵，升序，稳定
    eigvals = np.clip(eigvals, 0.0, None)
    major = 2.0 * float(np.sqrt(eigvals[1]))
    minor = 2.0 * float(np.sqrt(eigvals[0]))
    # 长轴方向：对应最大特征值的特征向量，取 x 轴夹角
    vx, vy = eigvecs[:, 1]
    orientation = float(np.arctan2(vy, vx))
    return major, minor, orientation


def _eccentricity(major: float, minor: float) -> float:
    if major <= 0:
        return 0.0
    ratio = min(max(minor / major, 0.0), 1.0)
    return float(np.sqrt(1.0 - ratio * ratio))


def _solidity(contour: List[List[float]], area: float) -> float:
    if len(contour) < 3 or area <= 0:
        return 0.0
    try:
        import cv2
    except ImportError:
        return 0.0
    pts = np.asarray(contour, dtype=np.float32) - 1.0   # 回到 0-based
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0:
        return 0.0
    return float(min(area / hull_area, 1.0))


def _projection_main_frequency(mask: np.ndarray) -> float:
    """行方向投影的主频（归一化到 [0, 0.5]）。

    修正了原 ``representation_learning.m:825-826`` 的 off-by-one
    （``max()`` 返回的是 ``power_spectrum(2:end)`` 内的下标，
    真实 bin 应为 ``argmax + 1``），并且去掉直流分量后再归一化。
    """
    rows = mask.shape[0]
    if rows < 4:
        return 0.0
    signal = mask.sum(axis=1).astype(np.float64)
    signal = signal - signal.mean()
    if not np.any(signal):
        return 0.0
    spec = np.abs(np.fft.rfft(signal)) ** 2
    if spec.size < 3:
        return 0.0
    k = int(np.argmax(spec[1:])) + 1        # 跳过 DC，且下标修正
    return float(k / rows)


# --------------------------------------------------------------------------
# 分割：轮廓 → 区域
# --------------------------------------------------------------------------

def segment_from_outline(
    outline_mask: np.ndarray,
    min_area_px: int = 50,
    max_area_ratio: float = 0.25,
    close_iterations: int = 2,
) -> Tuple[np.ndarray, int]:
    """把"轮廓线掩码"转成"区域标签图"。

    这是对原流水线最关键的修正：原实现使用 ``~outline_mask``（背景），
    本实现使用 ``fill_holes(outline_mask)``（轮廓内部）。

    Parameters
    ----------
    outline_mask
        布尔数组，``True`` = 轮廓像素（原项目中的绿色像素）。
    min_area_px
        最小区域面积，过滤碎片。
    max_area_ratio
        单个区域面积上限（相对整图），过滤"把背景围起来"的巨型区域。
    close_iterations
        填洞前做几次形态学闭运算，用于弥合手绘轮廓的小缺口。

    Returns
    -------
    labels, n_regions
        ``labels`` 为 int32 标签图（0 = 背景），``n_regions`` 为保留的区域数。
    """
    from scipy import ndimage as ndi

    if outline_mask.dtype != bool:
        outline_mask = outline_mask.astype(bool)

    closed = outline_mask
    if close_iterations > 0:
        closed = ndi.binary_closing(
            outline_mask, structure=np.ones((3, 3), bool), iterations=close_iterations
        )

    filled = ndi.binary_fill_holes(closed)

    labels, n = ndi.label(filled, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return labels.astype(np.int32), 0

    sizes = np.bincount(labels.ravel())
    total = float(outline_mask.size)
    max_area = max_area_ratio * total

    # 去掉碎片与巨型区域；保留的重新编号，保证 1..k 连续
    keep = [
        lab for lab in range(1, n + 1)
        if min_area_px <= sizes[lab] <= max_area
    ]
    if not keep:
        return np.zeros_like(labels, dtype=np.int32), 0

    remap = np.zeros(n + 1, dtype=np.int32)
    remap[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    return remap[labels].astype(np.int32), len(keep)
