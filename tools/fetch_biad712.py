#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_biad712.py —— 并行下载 S-BIAD712 的全部时序文件

## 为什么要并行

实测单连接约 0.8 MB/s，27 GB 需要约 9 小时，太慢。EBI 的 FTP 对单连接限速，
但允许多连接，因此用线程池并发下载，每个文件一个连接。

## 为什么先小后大

动力学分析是按文件独立的。先下最小的，就能**尽早开始分析并得到初步结论**，
不必等全部下完。列表按 size 升序处理。

## 断点续传与校验

* 已有文件若大小与元数据一致则跳过；
* 大小不一致则用 HTTP Range 从断点续传；
* 下完后再核对一次大小，不符就报出来——**静默失败是本项目踩过最多次的坑**。
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

BASE = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/712/S-BIAD712/Files/"
LOCK = threading.Lock()
DONE = {"n": 0, "bytes": 0, "fail": 0}


def attrs(e):
    return {a["name"]: a["value"] for a in e.get("attributes", [])}


def ts_files(meta_path: str):
    d = json.load(open(meta_path, encoding="utf-8", errors="replace"))
    ts = [e for e in d if attrs(e).get("Image Type", "").lower().startswith("time")]
    ts.sort(key=lambda e: e["size"])
    return ts


def fetch_one(entry, outdir: str, retries: int = 6) -> tuple:
    name = entry["path"]
    want = int(entry["size"])
    dst = os.path.join(outdir, name)

    have = os.path.getsize(dst) if os.path.exists(dst) else 0
    if have == want:
        with LOCK:
            DONE["n"] += 1
            DONE["bytes"] += want
        return name, True, "已有"

    url = BASE + urllib.parse.quote(name)
    for attempt in range(1, retries + 1):
        try:
            have = os.path.getsize(dst) if os.path.exists(dst) else 0
            headers = {"User-Agent": "Mozilla/5.0"}
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300, context=CTX) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(dst, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
            got = os.path.getsize(dst)
            if got == want:
                with LOCK:
                    DONE["n"] += 1
                    DONE["bytes"] += want
                return name, True, f"{got/1e6:.0f} MB"
            # 大小不符，继续重试（不删文件，保留断点）
            time.sleep(3)
        except Exception as e:  # noqa: BLE001
            time.sleep(3)
            last = f"{type(e).__name__}"
        else:
            last = f"大小 {os.path.getsize(dst) if os.path.exists(dst) else 0}≠{want}"

    with LOCK:
        DONE["fail"] += 1
    return name, False, last


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="temp/biad712_meta.json")
    ap.add_argument("--outdir", default="data/biad712")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.outdir, exist_ok=True)

    ts = ts_files(a.meta)
    total = sum(int(e["size"]) for e in ts)
    print(f"时序文件 {len(ts)} 个，合计 {total/1e9:.1f} GB")
    print(f"并发 {a.workers}，输出 {a.outdir}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch_one, e, a.outdir): e["path"] for e in ts}
        for i, fut in enumerate(as_completed(futs), 1):
            name, ok, msg = fut.result()
            el = time.time() - t0
            with LOCK:
                gb = DONE["bytes"] / 1e9
                rate = gb / el * 1000 if el else 0
            print(f"[{i:>2}/{len(ts)}] {'OK ' if ok else 'FAIL'} {name[:56]:<56}"
                  f" {msg:<12} 累计 {gb:.1f}/{total/1e9:.1f} GB "
                  f"({rate:.0f} MB/s, 失败 {DONE['fail']})", flush=True)

    print(f"\n完成，用时 {(time.time()-t0)/60:.1f} 分钟，失败 {DONE['fail']} 个")
    return 0 if DONE["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
