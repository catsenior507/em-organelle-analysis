#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_dataset_crops.py —— 探测某个数据集有哪些 crop **真的同时含所需类别且有数据**

为什么不能只看目录：CellMap 里有 crop 的类别目录与 `.zarray` 都存在、却**一个
数据块都没有**（实测 jrc_mus-kidney 的 crop149/mito、crop146/mito）。
只按目录存在筛选会把空数组当可用标注。

判断标准（与 fetch_labels.py 的 has_data_chunks 一致）：
除 `.zarray`/`.zgroup`/`.zattrs` 外，前缀下存在以数字开头的 chunk 路径。

用法：
    python probe_dataset_crops.py --dataset jrc_mus-liver --labels mito,er
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List

BUCKET = "https://janelia-cosem-datasets.s3.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
UA = "dachuang-organelle/1.0"


def http_get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def s3_ls(prefix: str, max_keys: int = 12, delimiter: str = "/"):
    url = (f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}"
           f"&max-keys={max_keys}")
    if delimiter:
        url += f"&delimiter={urllib.parse.quote(delimiter)}"
    root = ET.fromstring(http_get(url).decode("utf-8", "replace"))
    dirs = [p.find(NS + "Prefix").text for p in root.findall(NS + "CommonPrefixes")]
    keys = [(c.find(NS + "Key").text, int(c.find(NS + "Size").text))
            for c in root.findall(NS + "Contents")]
    return dirs, keys


def has_data(arr_prefix: str) -> bool:
    """该 Zarr 数组是否有真实数据块。

    **必须用无分隔符列举**。早期版本传了 ``delimiter="/"``，于是
    ``s0/0/0/0`` 这类 chunk 被折叠成 CommonPrefixes（``0/``、``1/``…），
    Contents 里只剩 ``.zarray``/``.zattrs``，导致**有数据也被判成空**。
    实测后果：一度错判 jrc_mus-liver / jrc_mus-liver-zon-1 / jrc_hela-2
    三个数据集"没有 mito+er 数据"，而 liver/crop124/mito/s0 实际有 chunk。
    """
    try:
        _, keys = s3_ls(arr_prefix, max_keys=10, delimiter="")
    except Exception:  # noqa: BLE001
        return False
    for k, _s in keys:
        base = k.rsplit("/", 1)[-1]
        if base.startswith("."):
            continue
        if base.isdigit():
            return True
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labels", default="mito,er")
    ap.add_argument("--resolution", default="s0")
    ap.add_argument("--max-report", type=int, default=40)
    args = ap.parse_args(argv)

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    root = f"{args.dataset}/{args.dataset}.zarr/recon-1/labels/groundtruth/"
    dirs, _ = s3_ls(root, max_keys=1000)
    crops = sorted({d.rstrip("/").split("/")[-1] for d in dirs
                    if d.rstrip("/").split("/")[-1].startswith("crop")},
                   key=lambda s: int(s[4:]) if s[4:].isdigit() else 0)
    print(f"{args.dataset}: groundtruth 下 {len(crops)} 个 crop")

    t0 = time.time()
    ok: List[Dict[str, object]] = []
    for c in crops:
        subs, _ = s3_ls(f"{root}{c}/", max_keys=200)
        have = {s.rstrip("/").split("/")[-1] for s in subs}
        if not all(l in have for l in labels):
            continue
        good = {}
        for l in labels:
            good[l] = has_data(f"{root}{c}/{l}/{args.resolution}/")
        if all(good.values()):
            ok.append({"crop": c, **good})
        else:
            print(f"  {c}: 目录有但数据块缺 {[k for k, v in good.items() if not v]}")
    print(f"\n同时含 {labels} 且**有数据**的 crop：{len(ok)} 个"
          f"（扫描 {time.time()-t0:.0f}s）")
    for r in ok[: args.max_report]:
        print("  ", r["crop"])
    out = os.path.expanduser(
        f"~/dachuang_organelle/data/eligible_{args.dataset.replace('-','_')}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"dataset": args.dataset, "labels": labels,
                   "eligible": [r["crop"] for r in ok],
                   "scanned": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                  fh, ensure_ascii=False, indent=1)
    print(f"\n已写出 {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
