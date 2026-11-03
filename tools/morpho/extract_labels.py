#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_labels.py —— 从黄色标注中恢复细胞器类型标签（OCR）

## 为什么需要单独一个脚本

原 `code/analyze_annotated_regions.m:149-178` 在区域分析过程中内联调用
`code/ocr.py`，而 `ocr.py` 有两个致命缺陷：

1. **未做极性反转**（`ocr.py:31` 直接对掩码图做 OTSU）。MATLAB 写出的中间图是
   **黑底白字**，Tesseract 在**白底黑字**上才准。结果是历史上识别出的全是
   `M` / `S` / `co` / 空字符串（统计：504 空、237 个 `"M"`、36 个 `"S"`、20 个 `"co"`）。
2. **硬编码 tesseract 路径** `D:\\Tesseract-OCR\\tesseract.exe`（`ocr.py:13`），
   而本机实际在 `C:\\Program Files\\Tesseract-OCR\\`，且不在 PATH 上。

本脚本对每个黄色连通域**两种极性各跑一次、取置信度更高者**，
并支持并行，可独立于区域提取运行。

## 用法

    python tools/morpho/extract_labels.py                    # 全库调查
    python tools/morpho/extract_labels.py --out data/labels.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_regions import (  # noqa: E402
    _ensure_tesseract,
    list_pairs,
    ocr_label,
    read_mask,
)


def survey_one(green_path: str, yellow_path: str,
               min_pixels: int = 6,
               merge_radius: int = 6) -> Dict[str, Any]:
    """对一张图的所有标注做 OCR，返回逐个标签的结果。

    关键点：**先做形态学膨胀再取连通域**。
    黄色标注是细笔画文字，抗锯齿会把一个字符切成几段（实测：全库 12 种
    "字形"里有 8 种是宽 1–4 px 的碎片）。直接对原始掩码做 ``ndi.label``
    会把一个 ``M`` 拆成 3–4 个碎片，碎片过小被 ``min_pixels`` 丢掉，
    于是 OCR 全部落空——这正是上一版"识别率异常"的原因。
    这里用 ``binary_dilation`` 把相近笔画合并，再在**原始掩码**上取
    合并后的包围盒做 OCR，既不改变字形，也不会漏字。
    """
    import scipy.ndimage as ndi

    label_mask = read_mask(yellow_path)
    green_mask = read_mask(green_path)
    filled_green = ndi.binary_fill_holes(
        ndi.binary_closing(green_mask, np.ones((3, 3), bool), iterations=2)
    )

    if merge_radius > 0:
        grouped = ndi.binary_dilation(
            label_mask, structure=np.ones((3, 3), bool), iterations=merge_radius
        )
    else:
        grouped = label_mask

    groups, n = ndi.label(grouped, structure=np.ones((3, 3), dtype=int))
    results: List[Dict[str, Any]] = []

    for lab in range(1, n + 1):
        # 该组覆盖的原始标注像素
        sel = label_mask & (groups == lab)
        count = int(sel.sum())
        if count < min_pixels:
            continue
        rows, cols = np.nonzero(sel)
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        crop = sel[r0:r1, c0:c1]

        text = ocr_label(crop)
        cy, cx = (r0 + r1) // 2, (c0 + c1) // 2
        results.append({
            "bbox": [c0 + 1, r0 + 1, c1, r1],      # 1-based (x1,y1,x2,y2)
            "pixels": count,
            "text": text,
            # 标签中心是否落在某个细胞器轮廓内部
            "inside_green_fill": bool(filled_green[cy, cx]),
        })

    counts = collections.Counter(r["text"] for r in results if r["text"])
    return {
        "num_labels": len(results),
        "recognized": dict(counts),
        "labels": results,
    }


def build_argparser() -> argparse.ArgumentParser:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    p = argparse.ArgumentParser(description="OCR 提取黄色标注文字")
    p.add_argument("--separated", default=os.path.join(root, "data", "分离的标注信息"))
    p.add_argument("--out", default=os.path.join(root, "data", "labels.json"))
    p.add_argument("--jobs", type=int, default=0)
    p.add_argument("--min-pixels", type=int, default=6)
    p.add_argument("--merge-radius", type=int, default=6,
                   help="连通域合并的膨胀半径（像素），用于把被抗锯齿切碎的笔画拼回一个字")
    p.add_argument("--limit", type=int, default=0)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    # 提前确认 tesseract 可用，避免跑完才发现
    try:
        import pytesseract
        _ensure_tesseract(pytesseract)
        version = pytesseract.get_tesseract_version()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] tesseract 不可用：{exc}", file=sys.stderr)
        print("  安装：https://github.com/UB-Mannheim/tesseract/wiki", file=sys.stderr)
        return 2

    pairs = list_pairs(args.separated)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print("[ERROR] 未找到 green_*/yellow_* 成对文件", file=sys.stderr)
        return 2

    n_jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) - 1)
    n_jobs = min(n_jobs, len(pairs))

    tasks = [(name, g, y) for name, g, y in pairs]
    results: Dict[str, Any] = {}
    global_counts: collections.Counter = collections.Counter()

    if n_jobs <= 1:
        for i, (name, g, y) in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {name}", flush=True)
            results[name] = survey_one(g, y, args.min_pixels, args.merge_radius)
            global_counts.update(results[name]["recognized"])
    else:
        print(f"并行处理 {len(tasks)} 张图，使用 {n_jobs} 个进程", flush=True)
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(survey_one, g, y, args.min_pixels, args.merge_radius): name
                for name, g, y in tasks
            }
            done = 0
            for fut in as_completed(futures):
                name = futures[fut]
                done += 1
                try:
                    results[name] = fut.result()
                    global_counts.update(results[name]["recognized"])
                except Exception as exc:  # noqa: BLE001
                    results[name] = {"error": f"{type(exc).__name__}: {exc}"}
                if done % 10 == 0 or done == len(tasks):
                    print(f"  已完成 {done}/{len(tasks)}", flush=True)

    payload = {
        "tesseract_version": str(version),
        "min_pixels": args.min_pixels,
        "merge_radius": args.merge_radius,
        "global_label_counts": dict(global_counts.most_common()),
        "per_image": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    print("\n" + "=" * 60)
    print("全局标签统计（OCR 结果 → 出现次数）:")
    for text, count in global_counts.most_common(30):
        print(f"  {text!r}: {count}")
    if not global_counts:
        print("  （未识别出任何文字）")
    print(f"\n输出：{os.path.abspath(args.out)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
