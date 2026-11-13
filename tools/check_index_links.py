#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_index_links.py —— 核对 docs/README.md 这个文档索引里的全部链接

## 为什么要单独查

索引的价值全在链接能点开。而索引是唯一一份"描述其它文档"的文档，
它出错时不会有任何征兆——读者点进一个 404 才发现。

`check_readme_links.py` 查的是**公开仓库**（public_repo/）里的引用，
本脚本查的是**私有工作区**的 docs 索引，两者范围不同。
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r"\]\(([^)#]+\.md)(?:#[^)]*)?\)")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    idx = os.path.join(ROOT, "docs", "README.md")
    if not os.path.exists(idx):
        print("[错误] 缺 docs/README.md")
        return 2
    s = open(idx, encoding="utf-8").read()
    links = LINK.findall(s)
    bad = []
    for l in links:
        p = os.path.normpath(os.path.join(ROOT, "docs", l))
        if not os.path.exists(p):
            bad.append((l, p))
    print(f"索引中的文档链接：{len(links)} 个")
    if bad:
        for l, p in bad:
            print(f"  ✗ {l}  ->  {p}")
        print(f"缺失 {len(bad)} 个")
        return 1
    print("全部存在")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
