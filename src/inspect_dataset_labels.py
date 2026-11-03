#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_dataset_labels.py —— 查清某数据集各 crop 到底有哪些类别、哪些真含数据

背景：manifest 声称某些 crop 有 mito+er，但实测 jrc_mus-liver 的可用 crop 为 0。
可能原因：
  * 类别名不同（mito_lum / mito_mem 而不是 mito）
  * 目录与 .zarray 存在但无数据块（已知的稀疏空数组问题）
  * manifest 与 S3 实际内容不同步

用法：
    python inspect_dataset_labels.py --dataset jrc_mus-liver --crops 2
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

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
    return ([p.find(NS + "Prefix").text for p in root.findall(NS + "CommonPrefixes")],
            [(c.find(NS + "Key").text, int(c.find(NS + "Size").text))
             for c in root.findall(NS + "Contents")])


def has_data(prefix: str) -> bool:
    """必须用**无分隔符**列举：chunk 形如 ``s0/0/0/0``，带 delimiter 时会被
    折叠成 CommonPrefixes，Contents 里只剩 ``.zarray``/``.zattrs``，
    于是**有数据也会被判成空**（曾因此错判 liver/hela 无 mito+er）。"""
    try:
        _, keys = s3_ls(prefix, max_keys=10, delimiter="")
    except Exception:  # noqa: BLE001
        return False
    return any(not k.rsplit("/", 1)[-1].startswith(".")
               and k.rsplit("/", 1)[-1].isdigit() for k, _ in keys)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--crops", type=int, default=3, help="抽查几个 crop")
    ap.add_argument("--resolution", default="s0")
    args = ap.parse_args(argv)

    root = f"{args.dataset}/{args.dataset}.zarr/recon-1/labels/groundtruth/"
    dirs, _ = s3_ls(root, max_keys=1000)
    crops = sorted({d.rstrip("/").split("/")[-1] for d in dirs
                    if d.rstrip("/").split("/")[-1].startswith("crop")},
                   key=lambda s: int(s[4:]) if s[4:].isdigit() else 0)
    print(f"{args.dataset}: {len(crops)} 个 crop")

    # 统计每个 crop 的类别名
    class_counter = Counter()
    mito_like = Counter()
    er_like = Counter()
    detail = []
    for c in crops[: max(1, args.crops)]:
        subs, _ = s3_ls(f"{root}{c}/", max_keys=200)
        names = [s.rstrip("/").split("/")[-1] for s in subs]
        for n in names:
            class_counter[n] += 1
            if n.startswith("mito"):
                mito_like[n] += 1
            if n.startswith("er"):
                er_like[n] += 1
        # 对 mito/er 精确名检查是否有数据
        info = {}
        for want in ("mito", "er"):
            if want in names:
                info[want] = "有" if has_data(f"{root}{c}/{want}/{args.resolution}/") else "空数组"
            else:
                info[want] = "无此类别"
        detail.append((c, len(names), info))
        print(f"\n--- {c}（{len(names)} 个类别）")
        print(f"    mito: {info['mito']}    er: {info['er']}")
        print(f"    全部类别: {names}")

    print()
    print("=== 各 crop 里出现过的 mito* 类别名 ===")
    for n, k in mito_like.most_common():
        print(f"  {n:<16} 出现在 {k} 个抽查 crop 中")
    print("=== er* ===")
    for n, k in er_like.most_common():
        print(f"  {n:<16} 出现在 {k} 个抽查 crop 中")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
