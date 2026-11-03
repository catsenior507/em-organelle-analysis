#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_regions.py —— 修正后的区域提取（替代 code/analyze_annotated_regions.m）

与原脚本的差别
--------------
================  ====================================  ==============================
方面              原 MATLAB 实现                        本实现
================  ====================================  ==============================
区域定义          ``~green_mask``（背景）⛔              ``fill_holes(green)``（轮廓内部）
像素存储          截断到 1000 点 + 字面量 ``...`` ⛔      简化轮廓多边形（无损形状）
JSON 合法性       91/94 个文件非法 ⛔                    ``json.load`` 100% 可解析
坐标语义          ``[row, col]`` 被当 ``(x, y)`` 读 ⛔   全流程统一 1-based ``(x, y)``
比例尺失败        静默代入 0.1 µm/px（差 20–100 倍）⛔  ``ok=False`` + 显式告警
OCR              未做极性反转，只得到 ``M``/``S``/``co``  自动极性判定 + 白名单 + 尺寸过滤
质量过滤          无（背景进入训练集）⛔                面积/标注/形状三重标记
================  ====================================  ==============================

输出
----
``<out>/regions/<name>.json``
    单张图的区域列表（含轮廓、特征、质量标记）。
``<out>/dataset.json``
    数据集清单：config、每张图的区域数、scheme 信息、失败列表。
``<out>/scale_report.json``
    比例尺标定汇总，便于人工核对。

用法
----
    python tools/morpho/extract_regions.py \
        --separated "data/分离的标注信息" \
        --originals "data/原图片" \
        --scale-dir "data/scale_mat" \
        --out "data/annotations_v2" \
        --ocr
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# 允许 `python tools/morpho/extract_regions.py` 直接运行时导入同目录的 features
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (  # noqa: E402
    COMPACT_FEATURE_NAMES,
    EXTENDED_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    ScaleCalibration,
    compute_region_features,
    segment_from_outline,
)


# --------------------------------------------------------------------------
# I/O 工具
# --------------------------------------------------------------------------

#: ``分离的标注信息/`` 里存在 ``green_11_0002_RGB.tif`` 这类重复导出，
#: 它们与 ``green_11_0002.tif`` 是同一张图的不同导出通道。默认只保留规范名，
#: 避免同一张原图被重复计数（原流水线把 91 张图当成 94 个样本）。
NON_CANONICAL_SUFFIXES = ("_RGB", "_Channel4")


def list_pairs(separated_dir: str) -> List[Tuple[str, str, str]]:
    """返回 ``[(name, green_path, yellow_path), ...]``，已过滤非规范重复导出。"""
    greens: Dict[str, str] = {}
    for fn in sorted(os.listdir(separated_dir)):
        if not fn.startswith("green_") or not fn.lower().endswith((".tif", ".tiff")):
            continue
        name = os.path.splitext(fn)[0][len("green_"):]
        greens[name] = os.path.join(separated_dir, fn)

    pairs: List[Tuple[str, str, str]] = []
    for name, gpath in sorted(greens.items()):
        if any(name.endswith(s) for s in NON_CANONICAL_SUFFIXES):
            continue
        ypath = os.path.join(separated_dir, f"yellow_{name}.tif")
        if os.path.exists(ypath):
            pairs.append((name, gpath, ypath))
    return pairs


def read_mask(path: str) -> np.ndarray:
    """读入颜色分离图并压成布尔掩码（任何非零通道即视为命中）。"""
    from PIL import Image

    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        return np.any(arr > 0, axis=2)
    return arr > 0


def read_gray(path: str) -> Optional[np.ndarray]:
    from PIL import Image

    if not os.path.exists(path):
        return None
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        # 用加权亮度而不是简单均值，与原 MATLAB rgb2gray 口径一致
        arr = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2])
    return arr.astype(np.float64)


def find_original(originals_dir: str, name: str) -> Optional[str]:
    for ext in (".tif", ".tiff", ".png", ".jpg"):
        p = os.path.join(originals_dir, name + ext)
        if os.path.exists(p):
            return p
    return None


# --------------------------------------------------------------------------
# 比例尺
# --------------------------------------------------------------------------

UNIT_TO_UM = {
    "nm": 1e-3,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "m": 1e6,      # 明确支持，避免原实现把 "mm" 误读成 "m" 后静默当 µm
    "mm": 1e3,
}


def load_scale(scale_dir: str, name: str) -> ScaleCalibration:
    """读取 ``scale__<name>.mat``。

    与原实现的关键区别：**标定失败时返回 ``ok=False``，绝不代入默认值**。
    """
    from scipy.io import loadmat

    path = os.path.join(scale_dir, f"scale__{name}.mat")
    if not os.path.exists(path):
        return ScaleCalibration(None, ok=False, source="missing",
                                note="未找到标定文件")

    try:
        d = loadmat(path)
    except Exception as exc:  # noqa: BLE001
        return ScaleCalibration(None, ok=False, source="unreadable",
                                note=f"读取失败: {exc}")

    def scalar(key: str) -> Optional[float]:
        if key not in d or d[key].size == 0:
            return None
        try:
            return float(np.ravel(d[key])[0])
        except (TypeError, ValueError):
            return None

    def text(key: str) -> Optional[str]:
        if key not in d or d[key].size == 0:
            return None
        v = np.ravel(d[key])[0]
        return str(v).strip()

    um_per_px = scalar("pixel_to_um")
    bar_px = scalar("max_width")
    bar_value = scalar("scale_value")
    unit = text("scale_unit")

    # 原流水线的兜底值就是 0.1，这里显式识别出来并判为不可信
    if um_per_px is not None and abs(um_per_px - 0.1) < 1e-12:
        return ScaleCalibration(
            None, bar_px=bar_px, bar_value=bar_value, bar_unit=unit,
            ok=False, source="legacy_fallback",
            note="原脚本 OCR 失败后的默认值 0.1 µm/px，不可信，需人工标定",
        )

    if unit is not None and unit.lower() not in UNIT_TO_UM:
        return ScaleCalibration(
            None, bar_px=bar_px, bar_value=bar_value, bar_unit=unit,
            ok=False, source="unknown_unit",
            note=f"未知单位 {unit!r}，原实现会静默按 µm 处理（可致 10^6 误差）",
        )

    if um_per_px is None or um_per_px <= 0:
        return ScaleCalibration(None, bar_px=bar_px, bar_value=bar_value,
                                bar_unit=unit, ok=False, source="invalid_value",
                                note="pixel_to_um 缺失或非正")

    # 合理性检查：2048 px 宽的视场若小于 0.5 µm 或大于 5000 µm，几乎必然有误
    fov_um = um_per_px * 2048
    note = ""
    ok = True
    if not (0.5 <= fov_um <= 5000):
        ok = False
        note = f"视场宽度 {fov_um:.3g} µm 超出合理范围，请人工复核"

    return ScaleCalibration(um_per_px, bar_px=bar_px, bar_value=bar_value,
                            bar_unit=unit, ok=ok, source="mat", note=note)


# --------------------------------------------------------------------------
# OCR（可选）
# --------------------------------------------------------------------------

#: 黄色标注文字是绘图工具打的细胞器类型缩写，允许的字符集很小。
ALLOWED_LABEL_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def ocr_label(
    crop: np.ndarray,
    min_pixels: int = 6,
    config: str = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
) -> str:
    """对黄色标注裁剪块做 OCR。

    修正 ``code/ocr.py`` 的两个致命问题：

    1. **极性**：MATLAB 写出的是黑底白字，Tesseract 对白底黑字才准。这里对两种极性
       各跑一次，取置信度更高者。
    2. **无效预处理**：去掉了对已二值图再做 OTSU、以及 1×1 膨胀这类空操作。
    """
    try:
        import cv2
        import pytesseract
    except ImportError:
        return ""

    _ensure_tesseract(pytesseract)

    if crop.size == 0 or np.count_nonzero(crop) < min_pixels:
        return ""

    up = crop.astype(np.uint8) * 255
    # 放大小字，Tesseract 在 x-height < 20 px 时基本失效
    scale = max(1, int(round(40.0 / max(up.shape[0], 1))))
    if scale > 1:
        up = cv2.resize(up, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    up = cv2.copyMakeBorder(up, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

    best, best_conf = "", -1.0
    for candidate in (up, 255 - up):        # 两种极性都试
        try:
            data = pytesseract.image_to_data(
                candidate, config=config, output_type=pytesseract.Output.DICT
            )
        except Exception:  # noqa: BLE001
            continue
        words, confs = [], []
        for txt, conf in zip(data.get("text", []), data.get("conf", [])):
            try:
                c = float(conf)
            except (TypeError, ValueError):
                continue
            cleaned = "".join(ch for ch in (txt or "") if ch in ALLOWED_LABEL_CHARS)
            if cleaned and c > 0:
                words.append(cleaned)
                confs.append(c)
        if words:
            score = float(np.mean(confs))
            if score > best_conf:
                best_conf, best = score, "".join(words)

    return best.strip()


_TESSERACT_READY = False


def _ensure_tesseract(pytesseract_module: Any) -> None:
    """定位 tesseract 可执行文件。

    原 ``code/ocr.py:13`` 把路径写死成 ``D:\\Tesseract-OCR\\tesseract.exe``；
    本机实际的安装位置是 ``C:\\Program Files\\Tesseract-OCR\\tesseract.exe``
    且不在 PATH 上，所以那里永远是"找不到 tesseract"。这里改为自动探测。
    """
    global _TESSERACT_READY
    if _TESSERACT_READY:
        return

    import shutil

    found = shutil.which("tesseract")
    if not found:
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
            os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        ]
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
    if found:
        pytesseract_module.pytesseract.tesseract_cmd = found
    _TESSERACT_READY = True


def label_crop(
    label_mask: np.ndarray, mask: np.ndarray, margin: int = 4
) -> np.ndarray:
    """取区域内黄色像素的包围盒裁剪，用于 OCR。"""
    sel = mask & label_mask
    if not np.any(sel):
        return np.zeros((0, 0), dtype=bool)
    rows, cols = np.nonzero(sel)
    r0 = max(0, rows.min() - margin)
    r1 = min(label_mask.shape[0], rows.max() + margin + 1)
    c0 = max(0, cols.min() - margin)
    c1 = min(label_mask.shape[1], cols.max() + margin + 1)
    return sel[r0:r1, c0:c1]


def normalize_label_text(text: str) -> str:
    """把 OCR 结果归一到标签本身。

    黄色标注是绘图工具打的单字符类型标签。抗锯齿会把细笔画切碎
    （见 ``tools/data_integrity/check_color_separation.py``：
    精确颜色匹配平均丢掉 31% 的像素），于是同一个 "M" 可能被 OCR 读成
    ``"M"``、``"hl"``、``"Ml"`` 等。观察到的全部变体都指向同一个字形，
    这里做一次显式归一，并把**原始 OCR 结果**保留在 ``ocr_raw`` 字段里，
    避免"归一把错误抹平"。
    """
    cleaned = "".join(ch for ch in (text or "") if ch in ALLOWED_LABEL_CHARS)
    if not cleaned:
        return ""
    # 实测变体：M / Ml / hl / Il / tl —— 均由 "M" 的笔画碎片误读而来
    if cleaned in {"M", "Ml", "MI", "hl", "Il", "tl", "H", "N"}:
        return "M"
    return cleaned


def associate_labels(
    label_mask: np.ndarray,
    mask: np.ndarray,
    labels_index: Optional[Dict[str, Any]],
    image_name: str,
    max_distance_px: float = 15.0,
) -> Tuple[str, str, float]:
    """把一张图的 OCR 标签关联到某个区域。

    优先使用 ``data/labels.json`` 里已经 OCR 好的结果（由
    ``tools/morpho/extract_labels.py`` 生成），按**标签中心到区域的距离**
    选取最近的标签——比"标签中心是否落在区域内"稳健，因为标签常常画在
    轮廓线外侧。

    ``max_distance_px`` 默认 15 px：细胞器的短轴中位数约 100 px，
    标签是紧挨着轮廓写的。放宽到几十像素会把远处别的不相干标签拉进来，
    实测放宽到 40 px 时最大匹配距离达到 575 px，等于"随便挑一个"。
    宁可返回空类型，也不要错误的类型。

    Returns
    -------
    (raw_text, normalized_text, distance_px)
        找不到标签时返回 ``("", "", inf)``。
    """
    if not labels_index:
        return "", "", float("inf")

    rec = labels_index.get(image_name)
    if not rec or "labels" not in rec:
        return "", "", float("inf")

    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return "", "", float("inf")

    def _find(px: float, py: float, max_d: float) -> Tuple[str, str, float]:
        """在 (px, py) 附近找最近的标签。"""
        best, best_d = None, float("inf")
        for L in rec["labels"]:
            x1, y1, x2, y2 = L["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            d = float(np.hypot(cx - px, cy - py))
            if d < best_d:
                best, best_d = L, d
        if best is None or best_d > max_d:
            return "", "", best_d
        return best.get("text", ""), normalize_label_text(best.get("text", "")), best_d

    # 策略 1：标签中心落在**标签像素质心**附近 —— 最强证据，距离记为 0。
    # 用黄色标注像素自身的质心（而不是区域质心）来匹配，因为标签通常写在
    # 轮廓线外侧，区域质心可能离标签很远。
    sel = mask & label_mask
    if np.any(sel):
        ys, xs = np.nonzero(sel)
        text, norm, d = _find(float(xs.mean()), float(ys.mean()), max_distance_px)
        if norm:
            return text, norm, 0.0

    # 策略 2：标签**包围盒与区域包围盒相交** —— 标签叠在轮廓线上时成立。
    rows, cols = np.nonzero(mask)
    bx1, bx2 = float(cols.min()), float(cols.max())
    by1, by2 = float(rows.min()), float(rows.max())
    for L in rec["labels"]:
        x1, y1, x2, y2 = L["bbox"]
        if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2):
            return L.get("text", ""), normalize_label_text(L.get("text", "")), 0.0

    # 策略 3：最近邻，严格距离上限。宁可返回空类型，也不要错误的类型。
    text, norm, d = _find((bx1 + bx2) / 2.0, (by1 + by2) / 2.0, max_distance_px)
    if norm:
        return text, norm, d
    return "", "", d


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def process_one(
    name: str,
    green_path: str,
    yellow_path: str,
    originals_dir: str,
    scale: ScaleCalibration,
    min_area_px: int,
    max_area_ratio: float,
    do_ocr: bool,
    on_size_mismatch: str = "error",
    labels_index: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    outline = read_mask(green_path)
    label_mask = read_mask(yellow_path)

    if outline.shape != label_mask.shape:
        raise ValueError(
            f"green/yellow 尺寸不一致: {outline.shape} vs {label_mask.shape}"
        )

    gray = None
    orig_path = find_original(originals_dir, name)
    mismatch_note = ""
    if orig_path is not None:
        gray = read_gray(orig_path)
        if gray is not None and gray.shape != outline.shape:
            # 项目里确实存在这一例：data/原图片/12_0002.tif 是 (2048, 2115)，
            # 而它的标注是 (2115, 2048)——原图与标注被转置了。
            # 原流水线对此**静默**处理（用 ``x <= size(img,2)`` 这类守卫丢掉越界像素），
            # 于是灰度特征悄悄出自错位的像素。这里必须显式决策。
            if on_size_mismatch == "transpose":
                gray = gray.T
                mismatch_note = (
                    f"原图 {os.path.basename(orig_path)} 与标注尺寸互为转置"
                    f"（{gray.T.shape} vs {outline.shape}），已按 --on-size-mismatch=transpose 转置对齐。"
                    "⚠️ 注意：转置只修正了**数组形状**，不等于图像内容与标注对齐——"
                    "实测该图的比例尺文字在转置后仍是侧向的，红色轮廓也未落在真实线粒体上，"
                    "说明该图的导出方向与其它图不同（很可能是旋转 90°）。"
                    "此图的灰度特征仅供参考，形态特征（由标注掩码导出）不受影响。"
                )
            elif on_size_mismatch == "skip-gray":
                mismatch_note = (
                    f"原图 {os.path.basename(orig_path)} 尺寸 {gray.shape} 与标注 "
                    f"{outline.shape} 不一致，已丢弃灰度特征（形态特征仍有效）"
                )
                gray = None
            else:
                raise ValueError(
                    f"原图 {os.path.basename(orig_path)} 尺寸 {gray.shape} "
                    f"与标注 {outline.shape} 不一致；"
                    f"用 --on-size-mismatch={{transpose,skip-gray}} 显式指定处理方式"
                )

    labels, n_regions = segment_from_outline(
        outline, min_area_px=min_area_px, max_area_ratio=max_area_ratio
    )

    regions = []
    for lab in range(1, n_regions + 1):
        mask = labels == lab

        # 标签关联：优先用 data/labels.json 里预 OCR 好的结果（按距离匹配），
        # 否则退回在本进程内即时 OCR。
        raw_text, otype, label_dist = associate_labels(
            label_mask, mask, labels_index, name
        )
        if not raw_text and do_ocr:
            raw_text = ocr_label(label_crop(label_mask, mask))
            otype = normalize_label_text(raw_text)

        feat = compute_region_features(
            mask, gray=gray, label_mask=label_mask,
            region_id=lab, recognized_text=raw_text, organelle_type=otype,
        )
        d = feat.to_dict()
        d["ocr_raw"] = raw_text
        # is_labeled 的语义收紧为"**成功关联到一个标签**"。
        # 原始实现用的是"区域内有黄色像素"，但黄色像素可能只是别的标签
        # 溢出的边缘（实测 562 个区域含黄色像素，其中只有 443 个能可靠关联）。
        d["is_labeled"] = bool(otype)
        d["has_label_pixels"] = feat.label_pixels > 0
        d["label_distance_px"] = (
            None if label_dist == float("inf") else round(float(label_dist), 2)
        )

        # 补上以 µm 为单位的派生量（仅当标定可信时才写）
        if scale.ok and scale.um_per_px is not None:
            k = scale.um_per_px
            d["um"] = {
                "per_px": k,
                "width": d["width"] * k,
                "height": d["height"] * k,
                "center_x": d["center_x"] * k,
                "center_y": d["center_y"] * k,
                "major_axis": d["major_axis"] * k,
                "minor_axis": d["minor_axis"] * k,
                "equivalent_diameter": d["equivalent_diameter"] * k,
                "area": d["area"] * k * k,
                "perimeter": d["perimeter"] * k,
            }
        else:
            d["um"] = None
        regions.append(d)

    labeled = [r for r in regions if r["is_labeled"]]
    return {
        "image": name,
        "image_shape": list(outline.shape),
        "original_file": os.path.basename(orig_path) if orig_path else None,
        "scale": scale.to_dict(),
        "geometry_note": mismatch_note,
        "num_regions": len(regions),
        "num_labeled": len(labeled),
        "regions": regions,
    }


def build_argparser() -> argparse.ArgumentParser:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    p = argparse.ArgumentParser(description="修正后的区域提取（fill-holes 分割）")
    p.add_argument("--separated", default=os.path.join(root, "data", "分离的标注信息"))
    p.add_argument("--originals", default=os.path.join(root, "data", "原图片"))
    p.add_argument("--scale-dir", default=os.path.join(root, "data", "scale_mat"))
    p.add_argument("--out", default=os.path.join(root, "data", "annotations_v2"))
    p.add_argument("--min-area-px", type=int, default=50,
                   help="最小区域面积（像素），过滤碎片")
    p.add_argument("--max-area-ratio", type=float, default=0.25,
                   help="单区域面积上限（占整图比例），过滤背景巨型区域")
    p.add_argument("--ocr", action="store_true", help="启用 OCR 识别标注文字（较慢）")
    p.add_argument("--labels",
                   default=None,
                   help="预 OCR 的标签索引（tools/morpho/extract_labels.py 的输出）；"
                        "提供后按距离把标签关联到区域，无需再跑 OCR")
    p.add_argument("--on-size-mismatch", choices=["error", "transpose", "skip-gray"],
                   default="transpose",
                   help="原图与标注尺寸不一致时的处理方式（项目里 12_0002.tif 是转置的）")
    p.add_argument("--jobs", type=int, default=0,
                   help="并行进程数；0 = CPU 核数-1（至少 1）")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试用）")
    p.add_argument("--quiet", action="store_true")
    return p


def _worker(task: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """子进程入口：处理一张图，返回 ``(record, failure)`` 之一。"""
    try:
        scale = load_scale(task["scale_dir"], task["name"])
        record = process_one(
            task["name"], task["green"], task["yellow"], task["originals"],
            scale, task["min_area_px"], task["max_area_ratio"],
            task["do_ocr"], task["on_size_mismatch"], task.get("labels_index"),
        )
        return record, None
    except Exception as exc:  # noqa: BLE001
        return None, {
            "image": task["name"],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
        }


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    pairs = list_pairs(args.separated)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print(f"[ERROR] 在 {args.separated} 未找到 green_*/yellow_* 成对文件", file=sys.stderr)
        return 2

    regions_dir = os.path.join(args.out, "regions")
    os.makedirs(regions_dir, exist_ok=True)

    # 预 OCR 的标签索引（可选）：必须在构造 tasks 之前加载，
    # 否则子进程拿不到。默认在输出目录的上一级找 labels.json。
    labels_index: Optional[Dict[str, Any]] = None
    labels_path = args.labels
    if labels_path is None:
        default_labels = os.path.join(
            os.path.dirname(os.path.abspath(args.out)), "labels.json"
        )
        if os.path.exists(default_labels):
            labels_path = default_labels
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        labels_index = payload.get("per_image", {})
        print(f"已加载标签索引：{labels_path}"
              f"（{len(labels_index)} 张图）", flush=True)
    elif args.ocr:
        print("未找到标签索引，将在本进程内即时 OCR（较慢）", flush=True)

    tasks = [
        {
            "name": name, "green": gpath, "yellow": ypath,
            "originals": args.originals, "scale_dir": args.scale_dir,
            "min_area_px": args.min_area_px, "max_area_ratio": args.max_area_ratio,
            "do_ocr": bool(args.ocr), "on_size_mismatch": args.on_size_mismatch,
            "labels_index": labels_index,
        }
        for name, gpath, ypath in pairs
    ]

    n_jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) - 1)
    n_jobs = min(n_jobs, len(tasks))

    results: List[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = []
    if n_jobs <= 1:
        for i, task in enumerate(tasks, 1):
            if not args.quiet:
                print(f"[{i}/{len(tasks)}] {task['name']}", flush=True)
            results.append(_worker(task))
    else:
        # 用进程池：每张图的形态学/FFT/cv2 都是纯 Python + C 计算，
        # 单进程约 3.3 s，89 张图串行要 5 分钟；并行后 1 分钟以内。
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if not args.quiet:
            print(f"并行处理 {len(tasks)} 张图，使用 {n_jobs} 个进程", flush=True)
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(_worker, t): t["name"] for t in tasks}
            done = 0
            for fut in as_completed(futures):
                done += 1
                name = futures[fut]
                if not args.quiet:
                    print(f"[{done}/{len(tasks)}] {name}", flush=True)
                results.append(fut.result())

    manifest: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    scales: Dict[str, Any] = {}

    for record, failure in results:
        if failure is not None:
            failures.append(failure)
            print(f"    [FAIL] {failure['image']}: {failure['error']}", file=sys.stderr)
            continue
        assert record is not None
        name = record["image"]
        scales[name] = record["scale"]

        out_path = os.path.join(regions_dir, f"{name}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=1)

        flags: Dict[str, int] = {}
        for r in record["regions"]:
            flags[r["quality_flag"]] = flags.get(r["quality_flag"], 0) + 1

        manifest.append({
            "image": name,
            "file": f"regions/{name}.json",
            "num_regions": record["num_regions"],
            "num_labeled": record["num_labeled"],
            "quality_flags": flags,
            "scale_ok": record["scale"]["ok"],
            "um_per_px": record["scale"]["um_per_px"],
            "geometry_note": record.get("geometry_note", ""),
        })

    manifest.sort(key=lambda m: m["image"])

    total_regions = sum(m["num_regions"] for m in manifest)
    bad_scale = [m["image"] for m in manifest if not m["scale_ok"]]

    dataset = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "generator": "tools/morpho/extract_regions.py",
        "config": {
            "separated_dir": os.path.abspath(args.separated),
            "originals_dir": os.path.abspath(args.originals),
            "scale_dir": os.path.abspath(args.scale_dir),
            "segmentation": "binary_fill_holes(green_outline)",
            "coordinate_convention": "1-based (x, y)",
            "min_area_px": args.min_area_px,
            "max_area_ratio": args.max_area_ratio,
            "ocr_enabled": bool(args.ocr),
            "on_size_mismatch": args.on_size_mismatch,
            "jobs": n_jobs,
        },
        "feature_schemas": {
            "compact_14": list(COMPACT_FEATURE_NAMES),
            "extended_13": list(EXTENDED_FEATURE_NAMES),
        },
        "summary": {
            "num_images": len(manifest),
            "num_regions": total_regions,
            "num_failures": len(failures),
            "images_with_bad_scale": bad_scale,
            "mean_regions_per_image": round(total_regions / max(len(manifest), 1), 2),
        },
        "images": manifest,
        "failures": failures,
    }

    with open(os.path.join(args.out, "dataset.json"), "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "scale_report.json"), "w", encoding="utf-8") as fh:
        json.dump(scales, fh, ensure_ascii=False, indent=1)

    print("\n" + "=" * 60)
    print(f"完成：{len(manifest)} 张图，{total_regions} 个区域")
    print(f"失败：{len(failures)}")
    print(f"比例尺不可信：{len(bad_scale)} 张 {bad_scale[:12]}")
    print(f"输出：{os.path.abspath(args.out)}")
    print("=" * 60)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
