#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asem_inspect.py —— 检查 ASEM 各数据集的体数据对齐信息

## 为什么必须先查对齐

ASEM 的标注体（如 `labels/golgi`）通常只覆盖 raw 的一个**子区域**：
数据集 61 的 golgi 标注是 210×283×371，而 raw 是 1805×1400×4800。
若不知道该子区域在 raw 中的**偏移**，就无法把标注和图像对齐——
标签体自己不能当训练输入（它是标签，不是图像）。

ASEM 在 `.zattrs` 里给了 `offset`（单位 nm）与 `resolution`，
据此可以算出 chunk 的索引偏移：

    idx = (offset_label - offset_raw) / resolution

本脚本把每个数据集的 raw 与各标注体的 offset/resolution/shape 列出来，
并算出让标注与 raw 对齐所需的索引偏移。
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.request
from typing import Dict, Optional

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

BUCKET = "https://asem-project.s3.amazonaws.com"


def get(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout, context=ctx) as r:
        return r.read().decode("utf-8", "replace")


def zjson(url: str) -> Optional[dict]:
    try:
        return json.loads(get(url))
    except Exception:
        return None


def vol_info(ds: str, vol: str) -> Optional[Dict]:
    base = f"{BUCKET}/datasets/{ds}/{ds}.zarr/neuroglancer_volumes/{vol}"
    attrs = zjson(f"{base}/s0/.zattrs")
    arr = zjson(f"{base}/s0/.zarray")
    if attrs is None or arr is None:
        return None
    return {"offset": attrs.get("offset"), "resolution": attrs.get("resolution"),
            "shape": arr.get("shape"), "chunks": arr.get("chunks"),
            "dtype": arr.get("dtype")}


def main() -> int:
    ds_list = sys.argv[1:] or ["46", "58", "61", "64"]
    print(f"{'数据集':<6} {'卷':<14} {'shape':>22} {'offset':>22} {'res':>10}")
    print("-" * 78)
    for ds in ds_list:
        raw = vol_info(ds, "raw")
        if not raw:
            print(f"{ds:<6} raw 读取失败")
            continue
        vols = ["raw"] + [v for v in ("labels/golgi", "labels/er", "labels/mito")
                          if vol_info(ds, v)]
        for v in vols:
            info = raw if v == "raw" else vol_info(ds, v)
            if not info:
                continue
            print(f"{ds:<6} {v:<14} {str(info['shape']):>22} "
                  f"{str(info['offset']):>22} {str(info['resolution']):>10}")
        # 计算对齐偏移
        if raw.get("offset") and raw.get("resolution"):
            ro, rr = raw["offset"], raw["resolution"]
            for v in ("labels/golgi", "labels/er", "labels/mito"):
                info = vol_info(ds, v)
                if not info or not info.get("offset"):
                    continue
                lo = info["offset"]
                idx = [round((lo[k] - ro[k]) / rr[k]) for k in range(3)]
                sh = info["shape"]
                print(f"       -> {v:<12} 对齐到 raw 的索引偏移 {idx}；"
                      f"占 raw 的 {sh[0]}×{sh[1]}×{sh[2]} 体素")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
