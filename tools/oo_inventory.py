#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oo_inventory.py —— 统计 OpenOrganelle groundtruth crop 里**真正有数据**的标注

## 为什么需要这一步

每个 crop 的目录里都有 64 个通道目录，但**目录存在不等于有数据**：
zarr 只写非全零的数据块，所以一个 `golgi/s0/` 里可能只有 `.zarray`
和 `.zattrs` 两个元数据文件，一个数据块都没有。

实测 `jrc_hela-2/crop1/golgi/s0/` 就是这种情况：
按目录名统计会以为"有高尔基体标注"，实际取出来全是零。

本脚本对每个 crop 的每个目标通道，只列 3 个 key 就能判断是否为空
（`max-keys=3`：若返回的全是点开头的元数据文件，即为空），
成本极低但结论确定。

## 输出

一张"数据集 × 通道 → 有数据的 crop 数"的表，以及可用的 (crop, 通道) 明细。
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

B = "https://janelia-cosem-datasets.s3.us-east-1.amazonaws.com/"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

DATASETS = ["jrc_hela-2", "jrc_hela-3", "jrc_jurkat-1", "jrc_macrophage-2"]
CHANNELS = ["golgi", "golgi_mem", "er", "er_mem", "mito", "mito_mem",
            "ld", "lyso", "perox", "endo", "ves", "nuc", "pm"]


def get(url: str, timeout: int = 40):
    for _ in range(2):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            continue
    return None


def list_prefixes(prefix: str) -> List[str]:
    t = get(f"{B}?list-type=2&delimiter=/&prefix={prefix}&max-keys=1000")
    if not t:
        return []
    return [p[len(prefix):].rstrip("/")
            for p in re.findall(r"<Prefix>([^<]+)</Prefix>", t)
            if p[len(prefix):].rstrip("/")]


def has_data(ds: str, crop: str, ch: str) -> bool:
    """该通道是否有真实数据块（不只是 .zarray/.zattrs）。"""
    prefix = (f"{ds}/{ds}.zarr/recon-1/labels/groundtruth/{crop}/{ch}/s0/")
    t = get(f"{B}?list-type=2&prefix={prefix}&max-keys=5")
    if not t:
        return False
    keys = [k[len(prefix):] for k in re.findall(r"<Key>([^<]+)</Key>", t)]
    return any(not k.startswith(".") for k in keys)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("检查 4 个含 golgi 的数据集：哪些 crop 的哪些通道真的有数据\n")

    summary: Dict[str, Dict[str, List[str]]] = {}
    for ds in DATASETS:
        gbase = f"{ds}/{ds}.zarr/recon-1/labels/groundtruth/"
        crops = list_prefixes(gbase)
        print(f"{ds}: {len(crops)} 个 crop，逐个查 {len(CHANNELS)} 个通道…")
        res = {ch: [] for ch in CHANNELS}
        jobs = [(c, ch) for c in crops for ch in CHANNELS]
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(has_data, ds, c, ch): (c, ch) for c, ch in jobs}
            for f in as_completed(futs):
                c, ch = futs[f]
                try:
                    if f.result():
                        res[ch].append(c)
                except Exception:
                    pass
        for ch in CHANNELS:
            if res[ch]:
                print(f"    {ch:<10} {len(res[ch]):>2} 个 crop: "
                      + " ".join(sorted(res[ch])[:8])
                      + ("…" if len(res[ch]) > 8 else ""))
        summary[ds] = res

    print()
    print("=" * 70)
    print("汇总：各通道在多少 crop 里真有数据")
    print("=" * 70)
    print(f"{'通道':<12}" + "".join(f"{d.split('_')[-1][:9]:>11}" for d in DATASETS))
    print("-" * 70)
    for ch in CHANNELS:
        row = "".join(f"{len(summary[d][ch]):>11}" for d in DATASETS)
        print(f"{ch:<12}{row}")

    # 可用的 (数据集, crop) 组合：至少同时有 golgi 与 er 或 mito
    print()
    print("可用的组合（同 crop 内同时有 golgi 与 er/mito）：")
    usable = []
    for ds in DATASETS:
        g = set(summary[ds]["golgi"])
        e = set(summary[ds]["er"])
        m = set(summary[ds]["mito"])
        both = sorted(g & (e | m))
        if both:
            usable.append((ds, both))
            print(f"  {ds:<20} {len(both)} 个 crop: {' '.join(both)}")
    if not usable:
        print("  无")

    os.makedirs("logs", exist_ok=True)
    with open("logs/oo_inventory.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print("\n明细已写出 logs/oo_inventory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
