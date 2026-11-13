#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_endpoints.py —— 探测各数据仓库的真实 API 端点

## 为什么先探

上一版直接假设了 `empiar/api/search?q=` 与 `idr/api/v0/m/studies/`，
两个都 404。这类"地址猜错"的失败有个共同特征：**返回为空但不报错**
（我把它当成"没有结果"），于是会得出"不存在时序数据"这种完全错误的结论。

所以先探测：对每个候选 URL 打印状态码与响应前 120 字节，
确认哪一个真的返回数据，再写解析。
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.error
import urllib.request

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

CANDIDATES = [
    ("EMPIAR entries", "https://www.ebi.ac.uk/empiar/api/entries/?limit=3"),
    ("EMPIAR entry", "https://www.ebi.ac.uk/empiar/api/entry/10001/"),
    ("EMPIAR search", "https://www.ebi.ac.uk/empiar/api/search?q=mitochondria"),
    ("EMPIAR v2", "https://www.ebi.ac.uk/empiar/api/v2/entries?limit=3"),
    ("IDR studies", "https://idr.openmicroscopy.org/api/v0/m/studies/"),
    ("IDR mapann", "https://idr.openmicroscopy.org/api/v0/m/mapannotations/"),
    ("IDR web", "https://idr.openmicroscopy.org/webclient/api/studies/"),
    ("CTC 2d", "http://celltrackingchallenge.net/2d-datasets/"),
    ("CTC api", "http://celltrackingchallenge.net/datasets/"),
    ("BBBC", "https://bbbc.broadinstitute.org/"),
]


def probe(name: str, url: str) -> None:
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/html;q=0.9,*/*;q=0.8"})
        with urllib.request.urlopen(req, timeout=40, context=CTX) as r:
            body = r.read()
            code = r.status
            ctype = r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        print(f"  {name:<16} HTTP {e.code}  {url}")
        return
    except Exception as e:  # noqa: BLE001
        print(f"  {name:<16} {type(e).__name__}  {url}")
        return
    snippet = body[:120].decode("utf-8", "replace").replace("\n", " ")
    ok = "JSON" if "json" in ctype else "HTML"
    print(f"  {name:<16} {code} {ok} {len(body)}B  {snippet[:80]}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("探测数据仓库 API 端点\n")
    for n, u in CANDIDATES:
        probe(n, u)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
