#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_pipeline.py —— 对重构后的数据与代码做一次端到端核对

回答三个问题：

1. **新的标注数据是否自洽**（区域数、面积、轮廓、标签、比例尺、µm 派生量）；
2. **它相对旧数据到底修好了什么**（旧 `features_json` 的非法 JSON、
   1000 点截断、xy 互换、背景被当成区域）；
3. **哪些旧结论必须撤回**（互信息 = log₂(样本数)、伪造的 λ、样本量不足）。

输出既打到 stdout，也写成 `docs/verification-report.json`，
方便放进进度记录里当证据。

用法::

    python tools/verify_pipeline.py
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

def _pct(a: float, b: float) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def section(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


# --------------------------------------------------------------------------
# 1. 旧数据的问题（可量化证据）
# --------------------------------------------------------------------------

def check_old_features_json(path: str = "data/features_json") -> Dict[str, Any]:
    """量化旧 `features_json` 的损坏程度。"""
    files = sorted(glob.glob(os.path.join(path, "*.json")))
    result: Dict[str, Any] = {"dir": path, "num_files": len(files)}

    if not files:
        result["status"] = "missing"
        return result

    parse_ok = parse_fail = 0
    ellipsis_files = 0
    truncated_1000 = 0
    total_regions = 0
    region_pixel_lens: List[int] = []

    for f in files:
        raw = open(f, encoding="utf-8", errors="replace").read()
        if ",..." in raw:
            ellipsis_files += 1
        try:
            data = json.loads(raw)
            parse_ok += 1
        except Exception:  # noqa: BLE001
            parse_fail += 1
            continue
        for r in data.get("regions", []):
            total_regions += 1
            rp = r.get("region_pixels") or []
            n = len(rp) // 2 if rp and isinstance(rp[0], int) else len(rp)
            region_pixel_lens.append(n)
            if n == 1000:
                truncated_1000 += 1

    result.update({
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
        "files_with_literal_ellipsis": ellipsis_files,
        "regions_in_parseable_files": total_regions,
        "regions_with_exactly_1000_points": truncated_1000,
    })
    return result


def check_old_background_contamination(
    separated_dir: str = "data/分离的标注信息",
    limit: int = 3,
) -> Dict[str, Any]:
    """证明旧分割逻辑取到的是背景。

    复现 `analyze_annotated_regions.m:69-75` 的做法（``~green_mask`` 做连通域），
    与正确做法（``binary_fill_holes``）对比。
    """
    from scipy import ndimage as ndi
    from PIL import Image

    out: Dict[str, Any] = {"images": []}
    files = sorted(glob.glob(os.path.join(separated_dir, "green_*.tif")))
    files = [f for f in files
             if not any(f.endswith(s + ".tif") for s in ("_RGB", "_Channel4"))][:limit]

    for f in files:
        name = os.path.basename(f)[len("green_"):-len(".tif")]
        g = np.any(np.asarray(Image.open(f).convert("RGB")) > 0, axis=2)
        total = int(g.size)

        # 旧做法：补集
        lab_old, n_old = ndi.label(~g)
        sizes_old = np.bincount(lab_old.ravel())[1:] if n_old else np.array([0])
        max_old = int(sizes_old.max()) if sizes_old.size else 0

        # 正确做法：填洞
        filled = ndi.binary_fill_holes(
            ndi.binary_closing(g, np.ones((3, 3), bool), iterations=2)
        )
        lab_new, n_new = ndi.label(filled, structure=np.ones((3, 3), int))
        sizes_new = np.bincount(lab_new.ravel())[1:] if n_new else np.array([])
        keep = sizes_new[(sizes_new >= 50) & (sizes_new <= 0.25 * total)]
        max_new = int(keep.max()) if keep.size else 0

        out["images"].append({
            "image": name,
            "image_pixels": total,
            "old_components": int(n_old),
            "old_largest_px": max_old,
            "old_largest_share": round(max_old / total, 4),
            "new_organelles": int(keep.size),
            "new_largest_px": max_new,
            "new_largest_share": round(max_new / total, 4),
            "new_total_share": round(float(keep.sum()) / total, 4),
        })
    return out


# --------------------------------------------------------------------------
# 2. 新数据的自洽性
# --------------------------------------------------------------------------

def check_new_dataset(root: str = "data/annotations_v2") -> Dict[str, Any]:
    ds_path = os.path.join(root, "dataset.json")
    if not os.path.exists(ds_path):
        return {"status": "missing", "root": root}

    ds = json.load(open(ds_path, encoding="utf-8"))
    out: Dict[str, Any] = {
        "root": root,
        "schema_version": ds.get("schema_version"),
        "num_images": ds["summary"]["num_images"],
        "num_regions": ds["summary"]["num_regions"],
        "num_failures": ds["summary"]["num_failures"],
    }

    areas: List[int] = []
    contour_pts: List[int] = []
    parse_fail = 0
    types: collections.Counter = collections.Counter()
    flags: collections.Counter = collections.Counter()
    um_ok = um_missing = 0
    bbox_ok = 0
    center_in_bbox = 0
    scale_ok = 0
    scale_bad: List[str] = []
    zero_area_contour = 0

    for m in ds["images"]:
        p = os.path.join(root, m["file"])
        try:
            rec = json.load(open(p, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            parse_fail += 1
            continue
        if rec["scale"]["ok"]:
            scale_ok += 1
        else:
            scale_bad.append(rec["image"])

        for r in rec["regions"]:
            areas.append(r["area"])
            contour_pts.append(len(r["contour"]))
            types[r["organelle_type"]] += 1
            flags[r["quality_flag"]] += 1
            if r.get("um"):
                um_ok += 1
            else:
                um_missing += 1
            # 包围盒一致性
            if (r["max_x"] - r["min_x"] + 1) == r["width"] and \
               (r["max_y"] - r["min_y"] + 1) == r["height"]:
                bbox_ok += 1
            if r["min_x"] <= r["center_x"] <= r["max_x"] and \
               r["min_y"] <= r["center_y"] <= r["max_y"]:
                center_in_bbox += 1
            # 轮廓必须闭合且面积为正
            if len(r["contour"]) < 3:
                zero_area_contour += 1

    n = len(areas)
    out.update({
        "json_parse_failures": parse_fail,
        "areas": {
            "min": int(min(areas)) if areas else None,
            "median": int(statistics.median(areas)) if areas else None,
            "max": int(max(areas)) if areas else None,
        },
        "contour_points": {
            "min": min(contour_pts) if contour_pts else None,
            "median": int(statistics.median(contour_pts)) if contour_pts else None,
            "max": max(contour_pts) if contour_pts else None,
        },
        "organelle_types": dict(types),
        "quality_flags": dict(flags),
        "regions_with_um": um_ok,
        "regions_without_um": um_missing,
        "bbox_width_height_consistent": _pct(bbox_ok, n),
        "center_inside_bbox": _pct(center_in_bbox, n),
        "contours_with_lt3_points": zero_area_contour,
        "scale_ok_images": scale_ok,
        "scale_bad_images": scale_bad,
        "assertions": {
            "all_json_parseable": parse_fail == 0,
            "no_contour_too_small": zero_area_contour == 0,
            "bbox_consistent": bbox_ok == n,
            "center_in_bbox": center_in_bbox == n,
            "single_organelle_class": len([k for k in types if k]) <= 1,
        },
    })
    return out


# --------------------------------------------------------------------------
# 3. 旧"涌现分析"结论的证伪
# --------------------------------------------------------------------------

def check_entropy_is_log2_n(normalized_dir: str = "data/normalized_features",
                            report_glob: str = "data/emerg_inf_advanced/*/comprehensive_report.txt"
                            ) -> Dict[str, Any]:
    """验证"联合熵 = log₂(该组行数)"。

    旧 `analyze_emergent_entropy.m` 的 `compute_joint_entropy` 用数据自身的
    `unique_bins` 建直方图边界，于是每个不同的行各占一个箱 →
    H = log₂(不同行数)。这里用报告里印出的数字反推，做交叉验证。
    """
    import re

    out: Dict[str, Any] = {"checks": [], "csv_counts": {}}

    files = sorted(glob.glob(os.path.join(normalized_dir, "*.csv")))
    if files:
        per: collections.Counter = collections.Counter()
        rows_per_group: Dict[str, int] = collections.defaultdict(int)
        for f in files:
            base = os.path.basename(f)
            m = re.match(r"normalized_region_analysis__(\d+)_(\d+)\.csv", base)
            if not m:
                continue
            per[m.group(1)] += 1
            with open(f, encoding="utf-8", errors="replace") as fh:
                rows_per_group[m.group(1)] += max(0, sum(1 for _ in fh) - 1)
        out["csv_counts"] = dict(sorted(per.items()))
        out["pooled_rows"] = dict(sorted(rows_per_group.items()))

    # 从报告里抓 "Joint Entropy" 与 "Sample s_NN"
    reports = sorted(glob.glob(report_glob))
    if reports:
        text = open(reports[-1], encoding="utf-8", errors="replace").read()
        cur = None
        for line in text.splitlines():
            ms = re.match(r"\s*Sample (s_\d+)", line)
            if ms:
                cur = ms.group(1).replace("s_", "")
            mj = re.search(r"Joint Entropy:\s*([0-9.]+)", line)
            if mj and cur:
                reported = float(mj.group(1))
                rows = out["pooled_rows"].get(cur)
                expected = math.log2(rows) if rows else None
                out["checks"].append({
                    "group": cur,
                    "pooled_rows": rows,
                    "log2_rows": round(expected, 4) if expected else None,
                    "reported_joint_entropy": reported,
                    "matches_log2_rows": (
                        expected is not None and abs(expected - reported) < 0.01
                    ),
                    "report": os.path.basename(os.path.dirname(reports[-1])),
                })
    return out


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    os.chdir(root)

    parser = argparse.ArgumentParser(description="重构后流水线的端到端核对")
    parser.add_argument("--out", default="docs/verification-report.json")
    args = parser.parse_args(argv)

    report: Dict[str, Any] = {}

    section("1. 旧数据的问题（可量化证据）")
    old_json = check_old_features_json()
    report["old_features_json"] = old_json
    if old_json.get("num_files"):
        print(f"  data/features_json：{old_json['num_files']} 个文件")
        print(f"    可被 json.load 解析：{old_json['parse_ok']}"
              f"    解析失败：{old_json['parse_fail']}")
        print(f"    含字面量 '...' 的文件：{old_json['files_with_literal_ellipsis']}")
        print(f"    恰好 1000 个坐标点的区域："
              f"{old_json['regions_with_exactly_1000_points']}"
              f" / {old_json['regions_in_parseable_files']}（可解析文件内）")

    old_bg = check_old_background_contamination()
    report["old_background_contamination"] = old_bg
    if old_bg["images"]:
        print(f"\n  旧分割逻辑（~green_mask）vs 正确做法（fill_holes）：")
        print(f"    {'image':>12} {'旧:最大块占比':>14} {'新:细胞器数':>12} "
              f"{'新:最大占比':>12} {'新:总面积占比':>14}")
        for r in old_bg["images"]:
            print(f"    {r['image']:>12} {r['old_largest_share'] * 100:>13.1f}% "
                  f"{r['new_organelles']:>12} {r['new_largest_share'] * 100:>11.1f}% "
                  f"{r['new_total_share'] * 100:>13.1f}%")

    section("2. 新数据集的自洽性")
    new_ds = check_new_dataset()
    report["new_dataset"] = new_ds
    if new_ds.get("status") == "missing":
        print("  data/annotations_v2 不存在——先运行 tools/morpho/extract_regions.py")
    else:
        print(f"  {new_ds['num_images']} 张图 / {new_ds['num_regions']} 个区域 / "
              f"{new_ds['num_failures']} 失败")
        print(f"  JSON 解析失败：{new_ds['json_parse_failures']}")
        a = new_ds["areas"]
        print(f"  区域面积：min {a['min']} / med {a['median']} / max {a['max']} px")
        c = new_ds["contour_points"]
        print(f"  轮廓点数：min {c['min']} / med {c['median']} / max {c['max']}"
              f"（旧实现固定 1000，且带省略号）")
        print(f"  包围盒 width/height 自洽：{new_ds['bbox_width_height_consistent']}")
        print(f"  中心点落在包围盒内：{new_ds['center_inside_bbox']}")
        print(f"  带 µm 派生量的区域：{new_ds['regions_with_um']}"
              f"（无标定：{new_ds['regions_without_um']}）")
        print(f"  比例尺可信：{new_ds['scale_ok_images']} 张；"
              f"不可信：{new_ds['scale_bad_images']}")
        print(f"  细胞器类型分布：{new_ds['organelle_types']}")
        print("\n  断言：")
        for k, v in new_ds["assertions"].items():
            print(f"    {'✓' if v else '✗'} {k}")

    section("3. 旧“涌现分析”结论的证伪：联合熵 = log₂(行数)")
    ent = check_entropy_is_log2_n()
    report["entropy_is_log2_n"] = ent
    if ent.get("checks"):
        print(f"  {'group':>6} {'合并行数':>10} {'log2(行数)':>12} "
              f"{'报告中的联合熵':>16} {'吻合':>6}")
        for c in ent["checks"]:
            print(f"  {c['group']:>6} {str(c['pooled_rows']):>10} "
                  f"{str(c['log2_rows']):>12} {c['reported_joint_entropy']:>16.4f} "
                  f"{'是' if c['matches_log2_rows'] else '否':>6}")
    print(f"\n  各样本组的 CSV 数（= “时间点”数）：{ent.get('csv_counts', {})}")
    few = [k for k, v in (ent.get("csv_counts") or {}).items() if v < 5]
    if few:
        print(f"  ⚠️ 时间点少于 5 的组：{few} —— 不足以做任何动力学分析")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"\n完整报告：{os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
