#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
list_biad2983_fields.py —— 枚举 S-BIAD2983 里可用的视野（stem）

## 为什么需要

文件名形如
`01252024_MSi08L_iN_Day7_BR5_N01_Unmixing_0_cmle.ome-<channel>.tiff`

同一 stem 下每个细胞器一个 tiff。要批量跑，必须先知道：
1. 有哪些 stem；
2. 每个 stem **六个必需通道是否齐全**（缺一个就跑不了该视野）；
3. 分组（iNeurons day7/14/21/28、iPSCs）——不同组代表不同分化阶段。

## 输出

`logs/biad2983_fields.json`：每个 stem 的分组、已有通道列表、是否齐全。
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import urllib.request
from collections import defaultdict

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

BASE = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/983/S-BIAD2983/Files/BioImageArchive-mcz/"
GROUPS = ["iNeurons day 7", "iNeurons day 14", "iNeurons day 21",
          "iNeurons day 28", "iPSCs"]
NEED = {"golgi", "mito", "lyso", "perox", "LD", "nuc", "cell"}


def listing(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180, context=CTX) as r:
        t = r.read().decode("utf-8", "replace")
    return [h for h in re.findall(r'href="([^"?][^"]*)"', t)
            if not h.startswith(("/", "http")) and h not in ("../", "./")]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    all_fields = {}
    for g in GROUPS:
        url = BASE + urllib.request.quote(g) + "/Segmented/"
        try:
            files = listing(url)
        except Exception as e:  # noqa: BLE001
            print(f"[跳过] {g}: {type(e).__name__}")
            continue
        stems = defaultdict(set)
        for f in files:
            if not f.endswith(".tiff"):
                continue
            base = f[:-5]
            if "-" not in base:
                continue
            stem, ch = base.rsplit("-", 1)
            stems[stem].add(ch)
        print(f"{g}: {len(stems)} 个视野, {len(files)} 个文件")
        for s, chs in stems.items():
            all_fields[s] = {
                "group": g,
                "channels": sorted(chs),
                "complete": NEED.issubset(chs),
                "missing": sorted(NEED - chs),
            }

    complete = {k: v for k, v in all_fields.items() if v["complete"]}
    print(f"\n总计 {len(all_fields)} 个视野，其中通道齐全 {len(complete)} 个")
    by_group = defaultdict(int)
    for v in complete.values():
        by_group[v["group"]] += 1
    for g, n in by_group.items():
        print(f"  {g:<18}{n}")

    os.makedirs("logs", exist_ok=True)
    json.dump(all_fields, open("logs/biad2983_fields.json", "w",
                               encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n写入 logs/biad2983_fields.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
