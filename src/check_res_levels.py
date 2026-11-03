#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_res_levels.py —— 查某数据集某 crop 的某类别**到底哪些分辨率层级有数据**

现象：jrc_mus-liver 与 jrc_mus-liver-zon-1 的 crop 里 `mito`/`er` 目录存在，
但 `s0/` 下没有数据块。可能是数据在 s1/s2，也可能整个类别都是空的。

同时列出该 crop 下所有类别的"非空"情况，便于判断这个数据集到底标注了什么。
"""
from __future__ import annotations

import argparse
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BUCKET = "https://janelia-cosem-datasets.s3.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
UA = "dachuang-organelle/1.0"


def http_get(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def s3_ls(prefix, max_keys=12, delimiter="/"):
    url = (f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}"
           f"&max-keys={max_keys}")
    if delimiter:
        url += f"&delimiter={urllib.parse.quote(delimiter)}"
    root = ET.fromstring(http_get(url).decode("utf-8", "replace"))
    return ([p.find(NS + "Prefix").text for p in root.findall(NS + "CommonPrefixes")],
            [(c.find(NS + "Key").text, int(c.find(NS + "Size").text))
             for c in root.findall(NS + "Contents")])


def has_data(prefix):
    """无分隔符列举，否则 chunk 会被折叠成 CommonPrefixes 而误判为空。"""
    try:
        _, keys = s3_ls(prefix, max_keys=10, delimiter="")
    except Exception:
        return False, 0
    n = sum(1 for k, _ in keys
            if not k.rsplit("/", 1)[-1].startswith(".") and k.rsplit("/", 1)[-1].isdigit())
    return n > 0, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--crop", required=True)
    ap.add_argument("--classes", default="mito,er,mito_mem,er_mem,all")
    args = ap.parse_args(argv)

    root = f"{args.dataset}/{args.dataset}.zarr/recon-1/labels/groundtruth/{args.crop}/"
    subs, _ = s3_ls(root, max_keys=200)
    names = [s.rstrip("/").split("/")[-1] for s in subs]
    print(f"{args.dataset}/{args.crop}: {len(names)} 个类别")

    print("\n=== 指定类别的各分辨率层级 ===")
    for cls in args.classes.split(","):
        cls = cls.strip()
        if cls not in names:
            print(f"  {cls:<12} 无此类别")
            continue
        lv, _ = s3_ls(f"{root}{cls}/", max_keys=20)
        levels = [s.rstrip("/").split("/")[-1] for s in lv]
        if not levels:
            print(f"  {cls:<12} 无分辨率层级子目录")
            continue
        row = []
        for lvname in sorted(levels, key=lambda s: int(s[1:]) if s[1:].isdigit() else 99):
            ok, n = has_data(f"{root}{cls}/{lvname}/")
            row.append(f"{lvname}:{'有' if ok else '空'}({n})")
        print(f"  {cls:<12} {' '.join(row)}")

    print("\n=== 全部类别的 s0 是否非空（前 30 个）===")
    nonempty = []
    for n in names[:30]:
        ok, cnt = has_data(f"{root}{n}/s0/")
        if ok:
            nonempty.append((n, cnt))
    if nonempty:
        for n, c in nonempty:
            print(f"  {n:<16} s0 有数据（前 10 个对象里 {c} 个 chunk）")
    else:
        print("  前 30 个类别在 s0 下**全部为空**")
        print("  => 这个 crop 在 s0 可能确实没有标注，或数据只在更粗层级")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
