#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oo_scan.py —— 扫描 OpenOrganelle 全部数据集，列出各数据集标注了哪些细胞器

## 为什么要扫全量

OpenOrganelle（Janelia CellMap）的 S3 桶公开，里面有 100 多个数据集，
但**不是每个都标注了同一种细胞器**：神经元数据集可能只标突触，
肝脏/HeLa 数据集才可能有高尔基体。要判断能否补我们的标注缺口，
必须先知道哪个数据集真有 golgi。

## 目录结构（实测）

    <ds>/<ds>.n5/labels/<类型>_<seg|pred>/      N5 格式的标注
    <ds>/<ds>.zarr/recon-1/{em,labels}/         zarr 格式的重建与预测

注意：标注在 **`.n5`** 下，不在 `.zarr/recon-1/labels/` 下。
早期版本按后者扫描，得到"全部 0 个"，是一个不看实际结构就下结论的错误。

`_seg` 是人工/半自动的分割真值，`_pred` 是模型预测。补标注需求关心的是 `_seg`。

## 特别关注的标注

* `golgi_seg` —— 高尔基体真值，直接补我们的缺口
* `*_contacts` —— **细胞器之间的接触标注**。这不是细胞器本身，
  而是互作界面，正是"互作网络"所需的一类真值，能用来独立验证我们的接触分析。
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

B = "https://janelia-cosem-datasets.s3.us-east-1.amazonaws.com/"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

TARGETS = ("golgi", "er", "mito", "ld", "lyso", "endo", "nucleus", "pm",
           "vesicle", "ribo", "chrom", "np", "eres", "ecs")


def get(url: str, timeout: int = 45) -> Optional[str]:
    for _ in range(2):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            continue
    return None


def list_prefixes(prefix: str) -> List[str]:
    body = get(f"{B}?list-type=2&delimiter=/&prefix={prefix}&max-keys=1000")
    if not body:
        return []
    return re.findall(r"<Prefix>([^<]+)</Prefix>", body)


def scan_dataset(ds: str) -> Dict:
    ds = ds.rstrip("/")
    base = f"{ds}/{ds}.n5/labels/"
    out = {"dataset": ds, "labels": [], "has_n5": False}
    subs = list_prefixes(base)
    if subs:
        out["has_n5"] = True
        out["labels"] = sorted(p[len(base):].rstrip("/") for p in subs if p != base)
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("列出 OpenOrganelle 全部数据集…")
    tops = [p for p in list_prefixes("") if not p.startswith("csc-zipped-data")]
    print(f"共 {len(tops)} 个数据集，逐个查标注类型…\n")

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(scan_dataset, d) for d in tops]
        for f in as_completed(futs):
            results.append(f.result())
    results.sort(key=lambda r: r["dataset"])

    n_n5 = sum(1 for r in results if r["has_n5"])
    print(f"其中 {n_n5} 个有 .n5 标注目录\n")

    have: Dict[str, List[str]] = {t: [] for t in TARGETS}
    for r in results:
        for lb in r["labels"]:
            if not lb.endswith("_seg"):      # 只看分割真值，不看预测
                continue
            for key in TARGETS:
                if lb.startswith(key):
                    have[key].append(r["dataset"])
                    break

    print("=" * 74)
    print("各细胞器的人工分割真值（*_seg）在哪些数据集里存在")
    print("=" * 74)
    for t in TARGETS:
        ds = sorted(set(have[t]))
        if not ds:
            continue
        print(f"\n{t}（{len(ds)} 个数据集）")
        for i in range(0, len(ds), 3):
            print("   " + "  ".join(ds[i:i + 3]))

    print()
    print("=" * 74)
    print("带细胞器接触标注（*_contacts）的数据集 —— 互作网络的直接真值")
    print("=" * 74)
    any_c = False
    for r in results:
        cs = [lb for lb in r["labels"] if "contacts" in lb]
        if cs:
            any_c = True
            print(f"\n{r['dataset']}（{len(cs)} 种）")
            for i in range(0, len(cs), 3):
                print("   " + "  ".join(cs[i:i + 3]))
    if not any_c:
        print("  无")

    # 汇总成一张"最有用"的清单
    print()
    print("=" * 74)
    print("建议优先取用的数据集")
    print("=" * 74)
    ranked = []
    for r in results:
        segs = {lb for lb in r["labels"] if lb.endswith("_seg")}
        n_contact = sum(1 for lb in r["labels"] if "contacts" in lb)
        score = (("golgi_seg" in segs) * 10 + ("er_seg" in segs) * 5
                 + ("mito_seg" in segs) * 5 + min(n_contact, 10)
                 + len(segs) * 0.1)
        if score > 0:
            ranked.append((score, r["dataset"], len(segs), n_contact,
                           "golgi" if "golgi_seg" in segs else ""))
    ranked.sort(reverse=True)
    print(f"{'数据集':<34}{'分割类':>6}{'接触类':>7}  {'含golgi':>7}")
    print("-" * 60)
    for score, ds, nseg, nc, g in ranked[:15]:
        print(f"{ds:<34}{nseg:>6}{nc:>7}  {g:>7}")

    os.makedirs("logs", exist_ok=True)
    out = "logs/oo_catalog.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    print(f"\n完整清单已写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
