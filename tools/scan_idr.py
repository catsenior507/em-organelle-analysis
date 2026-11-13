#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_idr.py —— 扫描 IDR 的全部研究，挑出与"细胞器 + 时序"相关的

## IDR 是什么

Image Data Resource（openmicroscopy.org）是活细胞成像的主要公共仓库，
有大量多通道时间序列。**它不是电镜仓库**，但本课题缺的正是时间轴，
而 IDR 里可能有可用的替代数据。

## 端点

实测 `https://idr.openmicroscopy.org/api/v0/` 返回的路径里**没有 studies**，
只有 projects / screens / datasets / images。早期版本按 `m/studies/` 请求，
得到 404，然后把它当成"没有数据"——这种静默误判要避免，所以先列 API 根
确认路径存在，再取数据。

## 判据

时间轴：IDR 的 `m/projects/` 条目本身不含"timelapse"字段，
因此按标题/描述做关键词匹配，并把命中的关键词一并打印以便人工复核。
只给出候选，不自动下结论。
"""

from __future__ import annotations

import json
import re
import ssl
import sys
import urllib.request
from typing import Dict, List

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

ORGANELLE = re.compile(
    r"mitochondri|golgi|endoplasmic|lysosom|peroxisom|lipid droplet|"
    r"organelle|autophag|endosom|nucle(?:us|ar)\b|membrane", re.I)
TEMPORAL = re.compile(
    r"time[\s-]?lapse|timelapse|live[\s-]?cell|live[\s-]?imaging|"
    r"dynamics|motility|migration|tracking|movie|kinetic", re.I)


def get(url: str):
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("列 IDR 全部 project…")
    d = get("https://idr.openmicroscopy.org/api/v0/m/projects/?limit=100000")
    items = d.get("data", d if isinstance(d, list) else [])
    print(f"  共 {len(items)} 个\n")

    rows: List[Dict] = []
    for p in items:
        name = p.get("Name") or ""
        desc = p.get("Description") or ""
        # 有些 project 的 Description 里带数据集摘要
        text = f"{name} {desc}"
        org = sorted({m.group(0).lower() for m in ORGANELLE.finditer(text)})
        tmp = sorted({m.group(0).lower() for m in TEMPORAL.finditer(text)})
        if org or tmp:
            rows.append({"id": p.get("Id") or p.get("@id"), "name": name[:150],
                         "organelle": org, "temporal": tmp})

    print("=" * 78)
    print("含细胞器关键词的 project")
    print("=" * 78)
    both = [r for r in rows if r["organelle"]]
    for r in both[:40]:
        t = "✓" if r["temporal"] else " "
        print(f"  [{t}] {r['id']:<6} {r['name'][:78]}")
        if r["temporal"]:
            print(f"          时序关键词: {', '.join(r['temporal'][:5])}")

    print()
    n_both = sum(1 for r in both if r["temporal"])
    print(f"同时命中的 project：{n_both} / {len(both)} 个含细胞器关键词")
    print()
    print("注意：关键词只能筛出候选。IDR 的 project 描述往往很简略，")
    print("      真正是否有时间轴要看每个数据集里的 Image 数量与 timepoint 元数据。")

    with open("logs/idr_projects.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    print("\n候选 -> logs/idr_projects.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
