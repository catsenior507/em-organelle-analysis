#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_czi.py —— 只看 CZI 文件头部，取出采集维度与时间信息

## 为什么只取头部

S-BIAD712 的每个时序文件 236–944 MB，整包下载 27 GB 不合理，而**判断可用性需要的
信息几乎全在文件头**：Zeiss CZI 在文件开头存放一段元数据 XML，里面有
`SizeT`（时间点数）、`SizeZ`、`SizeC`（通道数）、像素尺寸、以及时间戳。

因此先用 HTTP Range 取前若干 MB，在字节流里搜 XML；拿到维度即可决定是否值得下载。
若 XML 不在前几 MB（CZI 的分段存储可能把元数据放在别处），再逐步扩大范围。

## 安全性

只读。Range 请求不会修改远端，本地只写一个临时片段。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import urllib.parse
import urllib.request

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

BASE = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/712/S-BIAD712/Files/"


def attrs(e):
    return {a["name"]: a["value"] for a in e.get("attributes", [])}


def pick_smallest(meta_path: str):
    d = json.load(open(meta_path, encoding="utf-8", errors="replace"))
    ts = [e for e in d if attrs(e).get("Image Type", "").lower().startswith("time")]
    ts.sort(key=lambda e: e["size"])
    return ts


def fetch_head(url: str, nbytes: int) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes=0-{nbytes-1}"})
    with urllib.request.urlopen(req, timeout=180, context=CTX) as r:
        return r.read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="temp/biad712_meta.json")
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--out", default="temp/czi_head.bin")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    ts = pick_smallest(a.meta)
    print(f"Time series 文件 {len(ts)} 个，最小的三个：")
    for e in ts[:3]:
        at = attrs(e)
        print(f"  {e['size']/1e6:7.0f} MB  cycles={at.get('Zstack slices/cycles Number')}  "
              f"{e['path']}")

    target = ts[0]
    url = BASE + urllib.parse.quote(target["path"])
    print(f"\n取前 {a.mb} MB: {target['path']}")
    raw = fetch_head(url, a.mb * 1024 * 1024)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "wb").write(raw)
    print(f"  实得 {len(raw)/1e6:.2f} MB -> {a.out}")

    print("\n=== 文件头标识 ===")
    print(f"  前 16 字节: {raw[:16].hex()}")

    print("\n=== 在头部搜索 CZI 元数据 XML ===")
    txt = raw.decode("latin-1", "replace")
    # CZI 的图像元数据以 "ImageDocument" 或 "Metadata" 出现
    found = False
    for marker in ("ImageDocument", "Information/Image", "SizeT", "SizeZ", "SizeC"):
        hits = [m.start() for m in re.finditer(re.escape(marker), txt)]
        print(f"  '{marker}': {len(hits)} 处" + (f"  首现偏移 {hits[0]}" if hits else ""))
        if hits:
            found = True

    if not found:
        print("\n  头部未见 XML——CZI 元数据可能在分段中。建议扩大 --mb 或改用 czifile 解析。")
        return 2

    print("\n=== 抽取关键维度字段 ===")
    for pat in (r"<SizeT>(\d+)</SizeT>", r"<SizeZ>(\d+)</SizeZ>",
                r"<SizeC>(\d+)</SizeC>", r"<SizeX>(\d+)</SizeX>",
                r"<SizeY>(\d+)</SizeY>",
                r"<ScalingX>([\d.eE+-]+)</ScalingX>",
                r"<ScalingY>([\d.eE+-]+)</ScalingY>",
                r"<ScalingZ>([\d.eE+-]+)</ScalingZ>",
                r"<TimeSpan>([\d.eE+-]+)</TimeSpan>",
                r"<AcquisitionTime>([^<]+)</AcquisitionTime>"):
        m = re.search(pat, txt)
        if m:
            print(f"  {pat[1:pat.find('>')]:<18} = {m.group(1)}")

    # 时间相关：RelativeTime 标签
    rel = re.findall(r'<RelativeTime>(\d+)</RelativeTime>', txt)
    if rel:
        vals = sorted({int(x) for x in rel})
        print(f"  RelativeTime 标签 {len(rel)} 个，取值 {vals[:6]}{' ...' if len(vals) > 6 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
