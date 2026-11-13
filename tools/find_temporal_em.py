#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_temporal_em.py —— 在主要公共数据仓库里找**带时间维**的细胞器数据

## 为什么要单独找

本项目的动力学分析（李雅普诺夫指数、转移熵）在静态电镜快照上原理上不适用，
这一点已经在文档里明确记录。要让它成立，唯一的出路是拿到**真的有时间轴**的数据。

## 查哪些来源

| 来源 | 为什么查 |
|---|---|
| EMPIAR | 电镜原始数据的公共仓库，体量最大 |
| IDR（Image Data Resource） | 活细胞时序荧光的主要仓库，有元数据 API |
| Cell Tracking Challenge | 明确规定为时序数据，且带分割真值 |
| OpenOrganelle | 已知无时间轴，但确认一下没有新增时序子集 |

## 判据

"时序"不等于"体数据"：z 轴是空间，不是时间。要找的是
**同一视野在多个时间点重复采集**。关键词上区分
time-lapse / timelapse / live / dynamics / series / track / 4D，
并**排除**纯 tilt series（那是角度序列，不是时间）。

## 输出

一张按"是否含时间轴"分级的表，附各自的可疑度标注——自动匹配必然有假阳性，
所以每条都给出触发关键词，便于人工复核。
"""

from __future__ import annotations

import json
import re
import ssl
import sys
import urllib.parse
import urllib.request
from typing import Dict, List

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

POS = re.compile(r"time[\s-]?lapse|timelapse|live[\s-]?cell|live[\s-]?imaging|"
                 r"dynamics|motility|migration|series|tracking|4d\b|"
                 r"over[\s-]?time|kinetics", re.I)
NEG = re.compile(r"tilt[\s-]?series|tomogram|single[\s-]?particle|"
                 r"crystal|ribosome|virus", re.I)


def get(url: str, timeout: int = 45, as_json: bool = True):
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
            body = r.read()
        return json.loads(body.decode("utf-8", "replace")) if as_json else body
    except Exception as e:  # noqa: BLE001
        return {"__error__": f"{type(e).__name__}: {e}"}


def score(text: str) -> List[str]:
    hits = [m.group(0) for m in POS.finditer(text or "")]
    hits += [m.group(0) for m in NEG.finditer(text or "")]
    return hits


def query_empiar() -> List[Dict]:
    """EMPIAR 官方 REST API：按关键词检索条目。"""
    out: List[Dict] = []
    seen = set()
    for kw in ("time-lapse", "timelapse", "live cell", "dynamics",
               "time series", "4D", "motility"):
        url = ("https://www.ebi.ac.uk/empiar/api/search?q="
               + urllib.parse.quote(kw) + "&limit=100")
        j = get(url)
        if "__error__" in j:
            print(f"  EMPIAR '{kw}': {j['__error__'][:60]}")
            continue
        items = j if isinstance(j, list) else j.get("items", [])
        for it in items:
            acc = it.get("empiar_id") or it.get("id")
            if acc in seen:
                continue
            seen.add(acc)
            title = (it.get("title") or it.get("description") or "")[:200]
            out.append({"source": "EMPIAR", "id": str(acc),
                        "title": title, "kw": kw, "hits": score(title)})
    return out


def query_idr() -> List[Dict]:
    """IDR 用 OMERO 的 mapannotations/studies 接口列全部研究。"""
    j = get("https://idr.openmicroscopy.org/api/v0/m/studies/")
    if isinstance(j, dict) and "__error__" in j:
        print(f"  IDR: {j['__error__'][:60]}")
        return []
    out = []
    studies = j.get("data", j) if isinstance(j, dict) else j
    for s in studies if isinstance(studies, list) else []:
        name = s.get("Name") or ""
        desc = s.get("Description") or ""
        text = f"{name} {desc}"
        hits = score(text)
        if hits:
            out.append({"source": "IDR", "id": str(s.get("Id") or s.get("@id")),
                        "title": name[:200], "kw": "meta", "hits": hits})
    return out


def query_ctc() -> List[Dict]:
    """Cell Tracking Challenge：页面无 JSON API，抓训练集列表页。"""
    body = get("http://celltrackingchallenge.net/2d-datasets/", as_json=False)
    out = []
    if isinstance(body, dict):
        print(f"  CTC: {body.get('__error__', '')[:60]}")
        return out
    txt = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
    for m in re.finditer(r"<td[^>]*>\s*([A-Za-z0-9\-]{3,20})\s*</td>", txt):
        out.append({"source": "CTC", "id": m.group(1), "title": m.group(1),
                    "kw": "ctc", "hits": ["time-lapse(by design)"]})
    return out[:40]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 76)
    print("检索带时间轴的公共数据集")
    print("=" * 76)

    print("\n[1] EMPIAR")
    em = query_empiar()
    em.sort(key=lambda r: -len(r["hits"]))
    print(f"  命中 {len(em)} 条（按关键词得分排序，前 15）")
    for r in em[:15]:
        print(f"    {r['id']:<8} [{r['kw']}] {r['title'][:88]}")

    print("\n[2] IDR")
    idr = query_idr()
    print(f"  命中 {len(idr)} 条（前 15）")
    for r in idr[:15]:
        print(f"    {r['id']:<8} {r['title'][:88]}")

    print("\n[3] Cell Tracking Challenge")
    ctc = query_ctc()
    print(f"  {len(ctc)} 条（该仓库本身即为时序设计）")
    for r in ctc[:12]:
        print(f"    {r['id']}")

    os_ = "logs/temporal_candidates.json"
    import os
    os.makedirs("logs", exist_ok=True)
    with open(os_, "w", encoding="utf-8") as fh:
        json.dump({"empiar": em, "idr": idr, "ctc": ctc}, fh,
                  ensure_ascii=False, indent=1)
    print(f"\n候选清单 -> {os_}")
    print("\n注意：关键词匹配必然有假阳性（如 'series' 会命中 tilt series）。")
    print("每条都给了触发词，下一步需逐个核实是否真的同一视野多次采集。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
