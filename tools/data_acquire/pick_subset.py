#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pick_subset.py —— 依据官方 crop manifest 选出"性价比最高"的数据集子集

## 问题

CellMap 有 22 套数据集、289 个标注 crop。外接盘装不下全部，
而"哪个数据集值得提取"不能靠猜——要看：
  * 该数据集有多少 crop 同时含所需的类别（如 mito + er）；
  * 提取它大约要多少磁盘；
  * 每 GB 能换来多少个可用 crop。

## 做法

只读 `train_crop_manifest.csv`（官方给出每个 crop 的 shape / voxel_size /
class_label），按体素数估算体积。**这是估算，不是实测**：
真实 zip 里还有多分辨率层级（s0..s4，约 1.143×）与压缩，
所以本脚本给出的是**相对排序**，不是精确字节数。

估算公式（保守）：

    bytes(dataset) ≈ Σ_crop  shape_z·shape_y·shape_x · (1 + n_labels) · 1.2

其中 1 是 EM，n_labels 是该 crop 的类别数，1.2 近似多分辨率与压缩的综合影响。
需要精确值请用 `inspect_cellmap_zip.py`（HTTP Range 读 zip 目录，实测字节）。

## 用法

    python tools/data_acquire/pick_subset.py --need mito,er
    python tools/data_acquire/pick_subset.py --need mito,er --budget-gb 40
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(HERE, "cellmap_refs", "train_crop_manifest.csv")


def parse_vec(s: str) -> Tuple[float, ...]:
    return tuple(float(x) for x in s.strip().strip("[]").split(";"))


def load(manifest: str) -> Dict[Tuple[str, str], Dict]:
    """{(dataset, crop): {labels, shape, voxel}}"""
    info: Dict[Tuple[str, str], Dict] = {}
    with open(manifest, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["dataset"], str(r["crop_name"]))
            rec = info.setdefault(key, {"labels": set(),
                                        "shape": parse_vec(r["shape"]),
                                        "voxel": parse_vec(r["voxel_size"])})
            rec["labels"].add(r["class_label"])
    return info


def estimate_gb(info: Dict[Tuple[str, str], Dict]) -> Dict[str, Dict]:
    """按数据集估算体积与 crop 统计。"""
    per: Dict[str, Dict] = {}
    for (ds, crop), rec in info.items():
        sh = rec["shape"]
        vox = sh[0] * sh[1] * sh[2]
        nlab = len(rec["labels"])
        gb = vox * (1 + nlab) * 1.2 / 1e9
        e = per.setdefault(ds, {"gb": 0.0, "crops": 0, "with_need": 0,
                                "labels": set()})
        e["gb"] += gb
        e["crops"] += 1
        e["labels"] |= rec["labels"]
    return per


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="选出最划算的数据集子集")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--need", default="mito,er",
                    help="必须同时具备的类别，逗号分隔")
    ap.add_argument("--budget-gb", type=float, default=None,
                    help="磁盘预算（GB），只推荐合计不超过它的组合")
    args = ap.parse_args(argv)

    if not os.path.exists(args.manifest):
        print(f"[错误] manifest 不存在：{args.manifest}", file=sys.stderr)
        return 2

    need: Set[str] = {s.strip() for s in args.need.split(",") if s.strip()}
    info = load(args.manifest)
    per = estimate_gb(info)

    # 补 with_need 计数
    for (ds, crop), rec in info.items():
        if need <= rec["labels"]:
            per[ds]["with_need"] += 1

    rows = sorted(per.items(), key=lambda kv: kv[1]["gb"])
    print(f"需要同时具备：{sorted(need)}")
    print(f"{'dataset':<28} {'crops':>6} {'含need':>7} {'估GB':>9} {'GB/含need':>10}")
    print("-" * 66)
    for ds, e in rows:
        ratio = (e["gb"] / e["with_need"]) if e["with_need"] else float("inf")
        print(f"{ds:<28} {e['crops']:>6} {e['with_need']:>7} "
              f"{e['gb']:>9.1f} {ratio if ratio != float('inf') else float('nan'):>10.2f}")

    cand = [(ds, e) for ds, e in rows if e["with_need"] > 0]
    if not cand:
        print(f"\n[警告] 没有任何数据集同时具备 {sorted(need)}", file=sys.stderr)
        return 1

    print("\n按『每 GB 换到的可用 crop 数』排序（越靠前越划算）：")
    cand.sort(key=lambda t: t[1]["gb"] / t[1]["with_need"])
    for ds, e in cand:
        print(f"  {ds:<26} {e['gb']:6.1f} GB  {e['with_need']:2d} 个可用 crop"
              f"  -> {e['gb'] / e['with_need']:.2f} GB/crop")

    if args.budget_gb:
        total = 0.0
        chosen: List[str] = []
        for ds, e in cand:
            if total + e["gb"] <= args.budget_gb:
                chosen.append(ds)
                total += e["gb"]
        print(f"\n预算 {args.budget_gb} GB 内的建议组合（合计约 {total:.1f} GB）：")
        for ds in chosen:
            print(f"  --dataset {ds}")
        if chosen:
            print("\n直接可用：")
            print("  python tools/data_acquire/extract_subset.py --zip <zip> "
                  "--out <dir> \\")
            print("      " + " ".join(f"--dataset {d}" for d in chosen))

    print("\n注意：以上体积是**估算**（基于 manifest 的体素数），"
          "真实字节请用 extract_subset.py --plan 或 inspect_cellmap_zip.py。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
