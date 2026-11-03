#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_cellmap_zip.py —— 不下载整个 zip，直接读出它的内容清单

## 为什么需要

CellMap 的 `matched_res_no_pad.zip` 有 42 GB，但其中是 **1,319,617 个 Zarr 分块小文件**
（平均 ~32 KB）。这意味着：

- 不能盲目"下载完再解压"：往机械盘写 130 万个碎文件会非常慢，
  而且 NTFS 的元数据开销会额外吃掉数 GB；
- 应该**先知道里面有什么**（哪些数据集、每个数据集有哪些标注类别），
  再决定只提取需要的子集。

zip 格式把完整目录（central directory）放在文件**末尾**，所以可以用 HTTP `Range`
只读末尾几 MB 就能拿到全部条目名 —— 不必等 42 GB 下完。

## 做法

1. 读末尾 100 KB，找到 Zip64 的 EOCD 记录，拿到中央目录的偏移与长度；
2. 分块 Range 读取中央目录，逐条解析 `PK\\x01\\x02` 记录；
3. 汇总成 `dataset -> {labels, em_chunks, bytes}` 的结构并落盘 JSON。

## 用法

    python tools/data_acquire/inspect_cellmap_zip.py
    python tools/data_acquire/inspect_cellmap_zip.py --url <其它 zip>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_URL = ("https://janelia-cosem-datasets.s3.amazonaws.com/"
               "csc-zipped-data/matched_res_no_pad.zip")
UA = "dachuang-data-acquire/1.0"


def fetch_range(url: str, start: int, end: int, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def total_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        cr = r.headers.get("Content-Range") or ""
        if "/" in cr:
            return int(cr.rsplit("/", 1)[1])
    raise RuntimeError("无法获取总大小")


def locate_central_directory(url: str, total: int) -> Dict[str, int]:
    """解析 Zip64 EOCD，返回 central directory 的偏移与长度。"""
    tail = fetch_range(url, max(0, total - 200_000), total - 1)

    j = tail.rfind(b"PK\x06\x07")           # Zip64 EOCD locator
    if j < 0:
        raise RuntimeError("未找到 Zip64 EOCD locator（可能不是 Zip64）")
    _sig, _disk, z64_off, _ndisks = struct.unpack("<IIQI", tail[j:j + 20])

    rec = fetch_range(url, z64_off, z64_off + 55)
    if rec[:4] != b"PK\x06\x06":
        raise RuntimeError("Zip64 EOCD 记录签名不符")
    (_sig2, _size) = struct.unpack("<IQ", rec[:12])
    (ver_made, ver_need, disk, cd_disk, ent_disk, ent_total,
     cd_size, cd_off) = struct.unpack("<HHIIQQQQ", rec[12:12 + 44])

    return {
        "entries_total": int(ent_total),
        "cd_size": int(cd_size),
        "cd_offset": int(cd_off),
        "zip64_eocd_offset": int(z64_off),
    }


_CD_MAGIC = b"PK\x01\x02"
_CD_FIXED = 46


def iter_central_directory(url: str, cd_offset: int, cd_size: int,
                           chunk: int = 8 << 20):
    """流式产出 (name, uncompressed_size, compressed_size, method)。"""
    pos = 0
    carry = b""
    base = cd_offset
    while pos < cd_size:
        end = min(pos + chunk, cd_size) - 1
        buf = carry + fetch_range(url, base + pos, base + end)
        pos = end + 1

        i = 0
        consumed = 0
        while True:
            i = buf.find(_CD_MAGIC, i)
            if i < 0:
                # 保留尾部残片，等下一块拼接
                carry = buf[max(0, len(buf) - 200):]
                break
            if i + _CD_FIXED > len(buf):
                carry = buf[i:]
                break
            (_sig, _vm, _vn, _flags, method, _t, _d, _crc,
             csize, usize, nlen, elen, clen, _disk, _ia, _ea,
             _lho) = struct.unpack("<IHHHHHHIIIHHHHHII", buf[i:i + _CD_FIXED])
            if i + _CD_FIXED + nlen > len(buf):
                carry = buf[i:]
                break
            name = buf[i + _CD_FIXED:i + _CD_FIXED + nlen].decode("utf-8", "replace")
            yield name, usize, csize, method
            i += _CD_FIXED + nlen + elen + clen
            consumed = i
        # carry 已包含未完成记录；下次从干净边界继续
        if consumed:
            carry = buf[consumed:] if i >= len(buf) else b""
    # 收尾：carry 里若还有完整记录也解析掉（大多数情况下为空）
    if carry:
        i = 0
        while True:
            i = carry.find(_CD_MAGIC, i)
            if i < 0 or i + _CD_FIXED > len(carry):
                break
            (_sig, _vm, _vn, _flags, method, _t, _d, _crc,
             csize, usize, nlen, elen, clen, _disk, _ia, _ea,
             _lho) = struct.unpack("<IHHHHHHIIIHHHHHII", carry[i:i + _CD_FIXED])
            if i + _CD_FIXED + nlen > len(carry):
                break
            name = carry[i + _CD_FIXED:i + _CD_FIXED + nlen].decode("utf-8", "replace")
            yield name, usize, csize, method
            i += _CD_FIXED + nlen + elen + clen


def classify(name: str) -> Optional[Dict[str, Any]]:
    """把条目名解析成 (dataset, crop, kind, label)。

    实际结构（已逐层实测确认）::

        data_matched_res_no_pad/<dataset>/<dataset>.zarr/recon-1/
            em/fibsem-uint8/s0/...                        <- 电镜图像（多分辨率）
            labels/groundtruth/<crop>/<label>/s0/...      <- 标注（按 crop 分，多分辨率）

    踩过的坑：最初把 ``recon-1/<x>`` 直接当类别，得到的是 ``labels``；
    再过一层得到 ``groundtruth``；真正的类别在 **crop 之后的下一层**。
    """
    parts = name.split("/")
    if len(parts) < 3:
        return None
    dataset = parts[1]
    kind, label, crop = None, None, None

    if "recon-1" not in parts:
        return {"dataset": dataset, "crop": None, "kind": None, "label": None}
    tail = parts[parts.index("recon-1") + 1:]

    if tail and tail[0] == "em":
        kind = "em"
    elif len(tail) >= 4 and tail[0] == "labels" and tail[1] == "groundtruth":
        kind = "label"
        crop = tail[2]
        label = tail[3]
    return {"dataset": dataset, "crop": crop, "kind": kind, "label": label}


def main(argv: Optional[List[str]] = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="用 Range 读取 zip 内容清单")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", default=os.path.join(here, "cellmap_zip_inventory.json"))
    ap.add_argument("--max-entries", type=int, default=0,
                    help="只解析前 N 条（调试用，0=全部）")
    args = ap.parse_args(argv)

    total = total_size(args.url)
    print(f"zip 总大小：{total:,} 字节（{total / 1e9:.2f} GB）")
    info = locate_central_directory(args.url, total)
    print(f"条目总数：{info['entries_total']:,}")
    print(f"中央目录：{info['cd_size'] / 1e6:.1f} MB @ {info['cd_offset']:,}")

    datasets: Dict[str, Dict[str, Any]] = {}
    files: Dict[str, Dict[str, Any]] = {}
    n = 0
    last_report = 0
    for name, usize, csize, method in iter_central_directory(
            args.url, info["cd_offset"], info["cd_size"]):
        n += 1
        if args.max_entries and n > args.max_entries:
            break
        if name.endswith("/"):
            continue
        cls = classify(name)
        if not cls:
            continue
        ds = datasets.setdefault(cls["dataset"], {
            "files": 0, "bytes": 0, "labels": [], "has_em": False,
            "crops": [],
        })
        ds["files"] += 1
        ds["bytes"] += usize
        if cls["kind"] == "em":
            ds["has_em"] = True
        elif cls["kind"] == "label" and cls["label"]:
            if cls["crop"]:
                ds["crops"].append(cls["crop"])
            key = f"{cls['dataset']}/{cls['crop']}/{cls['label']}"
            rec = files.setdefault(key, {
                "dataset": cls["dataset"], "crop": cls["crop"],
                "label": cls["label"], "files": 0, "bytes": 0,
            })
            rec["files"] += 1
            rec["bytes"] += usize

        if n - last_report >= 200_000:
            last_report = n
            print(f"  ...已解析 {n:,} 条", flush=True)

    # 汇总 label 列表
    for key, rec in files.items():
        datasets[rec["dataset"]]["labels"].append(rec["label"])
    for ds in datasets.values():
        ds["labels"] = sorted(set(ds["labels"]))
        ds["crops"] = sorted(set(ds["crops"]))
        ds["num_crops"] = len(ds["crops"])
        ds["bytes_gb"] = round(ds["bytes"] / 1e9, 2)

    print(f"\n解析条目：{n:,}")
    print(f"数据集：{len(datasets)}")
    print(f"\n{'dataset':<28} {'GB':>8} {'files':>9}  em  labels")
    for name, ds in sorted(datasets.items(), key=lambda kv: -kv[1]["bytes"]):
        labs = ",".join(ds["labels"])[:60]
        print(f"{name:<28} {ds['bytes_gb']:>8.2f} {ds['files']:>9}  "
              f"{'Y' if ds['has_em'] else 'N':>2}  {labs}")

    # 有 mito 且有 er 的数据集 —— 这是本项目最需要的
    both = [k for k, v in datasets.items()
            if "mito" in v["labels"] and "er" in v["labels"]]
    print(f"\n同时含 mito 与 er 的数据集（{len(both)}）：{both}")

    payload = {
        "url": args.url,
        "total_bytes": total,
        "entries": info["entries_total"],
        "parsed": n,
        "datasets": datasets,
        "label_sizes": files,
        "datasets_with_mito_and_er": both,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"\n清单已写入：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
