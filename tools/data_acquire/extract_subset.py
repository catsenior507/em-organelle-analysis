#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_subset.py —— 从 CellMap zip 中**只提取选定数据集**，不全量解压

## 为什么不能全量解压

`matched_res_no_pad.zip` 里有 **1,319,617 个文件**（Zarr 分块，平均 ~32 KB）。
往机械硬盘写 130 万个碎文件会很慢（寻道主导），NTFS 元数据还要额外吃掉数 GB。
而本项目实际只需要少数几个同时含 `mito` 与 `er` 的数据集。

## 做法

zip 的中央目录允许随机访问单个条目，所以 `zipfile` 可以只读需要的条目：

    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.filename.startswith(prefix):
                zf.open(info) -> 解压写入

全量列 130 万条目需要一点内存（~250 MB），但只解压匹配前缀的部分。

## 用法

    # 先看有哪些数据集、哪些含 mito+er（不写盘）
    python tools/data_acquire/extract_subset.py --zip E:\\...\\matched_res_no_pad.zip --plan

    # 只提取 jrc_cos7-1a
    python tools/data_acquire/extract_subset.py \\
        --zip E:\\...\\matched_res_no_pad.zip \\
        --out E:\\dachuang_public_data\\cellmap_extracted \\
        --dataset jrc_cos7-1a
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from typing import Any, Dict, List, Optional


def classify(name: str) -> Dict[str, Any]:
    """从 zip 内路径解析 dataset / crop / 类别。

    结构（已逐层实测确认）::

        data_matched_res_no_pad/<dataset>/<dataset>.zarr/recon-1/
            em/fibsem-uint8/s0/...
            labels/groundtruth/<crop>/<label>/s0/...
    """
    parts = name.split("/")
    out = {"dataset": None, "crop": None, "label": None, "kind": None}
    if len(parts) < 3:
        return out
    out["dataset"] = parts[1]
    if "recon-1" not in parts:
        return out
    tail = parts[parts.index("recon-1") + 1:]
    if tail and tail[0] == "em":
        out["kind"] = "em"
    elif len(tail) >= 4 and tail[0] == "labels" and tail[1] == "groundtruth":
        out["kind"] = "label"
        out["crop"] = tail[2]
        out["label"] = tail[3]
    return out


def scan_zip(zp: str) -> Dict[str, Any]:
    """扫描中央目录，汇总 dataset -> {bytes, files, labels, crops}。"""
    t0 = time.time()
    stats: Dict[str, Dict[str, Any]] = {}
    n = 0
    with zipfile.ZipFile(zp) as zf:
        infos = zf.infolist()
        print(f"中央目录条目：{len(infos):,}（读取耗时 {time.time() - t0:.1f}s）")
        for info in infos:
            n += 1
            if info.is_dir():
                continue
            cls = classify(info.filename)
            ds = cls["dataset"]
            if not ds:
                continue
            rec = stats.setdefault(ds, {"bytes": 0, "files": 0,
                                        "labels": set(), "crops": set(),
                                        "has_em": False})
            rec["files"] += 1
            rec["bytes"] += info.file_size
            if cls["kind"] == "em":
                rec["has_em"] = True
            elif cls["kind"] == "label":
                rec["labels"].add(cls["label"])
                if cls["crop"]:
                    rec["crops"].add(cls["crop"])
            if n % 300_000 == 0:
                print(f"  ...扫描 {n:,}", flush=True)
    for rec in stats.values():
        rec["labels"] = sorted(rec["labels"])
        rec["crops"] = sorted(rec["crops"])
        rec["num_crops"] = len(rec["crops"])
        rec["bytes_gb"] = round(rec["bytes"] / 1e9, 2)
        del rec["crops"]
    return stats


def plan(stats: Dict[str, Any], need: List[str]) -> None:
    print(f"\n{'dataset':<30} {'GB':>8} {'files':>9} {'crops':>6}  em  mito er  同时?")
    rows = sorted(stats.items(), key=lambda kv: kv[1]["bytes"])
    both = []
    for name, r in rows:
        has = [k for k in need if k in r["labels"]]
        ok = len(has) == len(need)
        if ok:
            both.append(name)
        print(f"{name:<30} {r['bytes_gb']:>8.2f} {r['files']:>9} "
              f"{r['num_crops']:>6}  {'Y' if r['has_em'] else 'N':>2}  "
              f"{' '.join('Y' if k in r['labels'] else '-' for k in need)}  "
              f"{'<<<' if ok else ''}")
    print(f"\n同时含 {need} 的数据集：{len(both)}")
    for name in both:
        r = stats[name]
        print(f"  {name:<30} {r['bytes_gb']:>7.2f} GB  {r['num_crops']} crops")
    if both:
        smallest = min(both, key=lambda k: stats[k]["bytes"])
        print(f"\n建议先提取最小的：{smallest} "
              f"（{stats[smallest]['bytes_gb']} GB）")


def extract(zp: str, out_root: str, datasets: List[str],
            limit_files: int = 0) -> None:
    wanted = set(datasets)
    t0 = time.time()
    done_files = 0
    done_bytes = 0
    planned = 0

    with zipfile.ZipFile(zp) as zf:
        infos = zf.infolist()
        targets = []
        for info in infos:
            cls = classify(info.filename)
            if cls["dataset"] in wanted and not info.is_dir():
                targets.append(info)
        planned = len(targets)
        total_bytes = sum(i.file_size for i in targets)
        print(f"将提取 {planned:,} 个文件，共 {total_bytes / 1e9:.2f} GB")
        print(f"目标目录：{os.path.abspath(out_root)}")

        last = time.time()
        for i, info in enumerate(targets, 1):
            if limit_files and i > limit_files:
                print(f"  [limit] 只提取前 {limit_files} 个文件")
                break
            dest = os.path.join(out_root, info.filename.replace("/", os.sep))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as fh:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    fh.write(buf)
            done_files += 1
            done_bytes += info.file_size
            now = time.time()
            if now - last >= 5:
                last = now
                rate = done_bytes / max(now - t0, 1e-6)
                print(f"   {done_files:,}/{planned:,} 文件  "
                      f"{done_bytes / 1e9:.2f} GB  {rate / 1e6:.1f} MB/s",
                      flush=True)

    print(f"\n完成：{done_files:,} 个文件，{done_bytes / 1e9:.2f} GB，"
          f"耗时 {time.time() - t0:.0f}s")
    print(f"输出：{os.path.abspath(out_root)}")


def main(argv: Optional[List[str]] = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="从 CellMap zip 中只提取选定数据集")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default=None,
                    help="输出根目录（缺省为 zip 所在目录/cellmap_extracted）")
    ap.add_argument("--dataset", action="append", default=None,
                    help="要提取的数据集名，可重复")
    ap.add_argument("--need", default="mito,er",
                    help="--plan 时检查的类别（默认 mito,er）")
    ap.add_argument("--plan", action="store_true",
                    help="只扫描并给出建议，不写盘")
    ap.add_argument("--limit-files", type=int, default=0,
                    help="只提取前 N 个文件（试跑用）")
    ap.add_argument("--inventory", default=os.path.join(here, "zip_subset_inventory.json"))
    args = ap.parse_args(argv)

    if not os.path.exists(args.zip):
        print(f"[错误] zip 不存在：{args.zip}", file=sys.stderr)
        return 2

    stats = scan_zip(args.zip)
    with open(args.inventory, "w", encoding="utf-8") as fh:
        json.dump({"zip": os.path.abspath(args.zip), "datasets": stats},
                  fh, ensure_ascii=False, indent=1)
    print(f"清单已写入：{args.inventory}")

    need = [s.strip() for s in args.need.split(",") if s.strip()]
    plan(stats, need)

    if args.plan:
        return 0
    if not args.dataset:
        print("\n[提示] 未指定 --dataset，已只做扫描。"
              "要提取请加 --dataset <名字>。", file=sys.stderr)
        return 0

    unknown = [d for d in args.dataset if d not in stats]
    if unknown:
        print(f"[错误] 未知数据集：{unknown}", file=sys.stderr)
        return 2

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.zip)),
                                   "cellmap_extracted")
    extract(args.zip, out, args.dataset, args.limit_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
