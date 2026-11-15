#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
walk_biad.py —— 遍历 BioImage Archive 任一研究的文件树，判断是否含时序列

用法：
    python tools/walk_biad.py S-BIAD3460
    python tools/walk_biad.py S-BIAD3460 --max-files 6000

## 判断依据

不看描述（描述常写"timelapse 能力"但存的是快照），只看**文件名**：

  * 同一视野出现多个时间编号（`_t01`、`T001`、`timepoint2`）→ 有时序；
  * 只有视野编号（`N01`、`F02`、`pos3`）→ 只有快照。

S-BIAD2983 就是前车之鉴：描述写"timelapse images were acquired"，
实际 4518 个文件里含时间编号的是 0 个。
"""

from __future__ import annotations

import argparse
import re
import ssl
import sys
import urllib.request
from collections import Counter

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

DROP3 = ("983",)  # 仅用于拼 URL 的占位，实际按 accession 拆分


def base_for(acc: str) -> str:
    """S-BIAD3460 -> .../S-BIAD/460/S-BIAD3460/Files/"""
    num = acc.split("S-BIAD")[-1]
    return f"https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/{num}/{acc}/Files/"


def listing(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90, context=CTX) as r:
        t = r.read().decode("utf-8", "replace")
    return [h for h in re.findall(r'href="([^"?][^"]*)"', t)
            if not h.startswith(("/", "http")) and h not in ("../", "./")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("accession")
    ap.add_argument("--max-files", type=int, default=5000)
    ap.add_argument("--depth", type=int, default=5)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    base = base_for(a.accession)
    print(f"根: {base}")
    files, dirs = [], []
    frontier = [base]
    for depth in range(1, a.depth + 1):
        nxt = []
        for u in frontier:
            try:
                entries = listing(u)
            except Exception as e:  # noqa: BLE001
                print(f"  [跳过] {u}  {type(e).__name__}")
                continue
            for e in entries:
                (nxt.append(u + e) if e.endswith("/") else files.append(u + e))
                if e.endswith("/"):
                    dirs.append(u + e)
        print(f"  深度 {depth}: 目录 {len(nxt)}，累计文件 {len(files)}")
        if len(files) >= a.max_files:
            print("  达到上限，停止遍历")
            break
        frontier = nxt
        if not frontier:
            break

    print(f"\n共 {len(files)} 个文件，{len(dirs)} 个目录")
    if not files:
        return 1

    print("\n=== 扩展名 ===")
    for ext, n in Counter(("." + f.rsplit(".", 1)[-1].lower())
                          if "." in f.rsplit("/", 1)[-1] else "(无)"
                          for f in files).most_common(8):
        print(f"  {ext:<10}{n:>6}")

    print("\n=== 时序标记（关键）===")
    tpat = re.compile(r"(?:[_-][tT](\d{2,4})(?=[_.])|_T(\d{3,4})|"
                      r"time[_-]?point[_-]?(\d+)|[_-]t(\d{2,3})[_-])")
    hit = [f for f in files if tpat.search(f)]
    print(f"  含时间编号: {len(hit)} / {len(files)}  "
          f"→ {'**有时序**' if hit else '只有快照'}")
    for f in hit[:6]:
        print(f"    {f[len(base):]}")

    print("\n=== 抽样 10 个文件 ===")
    for f in files[:10]:
        print(f"    {f[len(base):]}")

    print("\n=== 抽样 8 个目录 ===")
    for d in dirs[:8]:
        print(f"    {d[len(base):]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
