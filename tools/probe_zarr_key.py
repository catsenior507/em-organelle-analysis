#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_zarr_key.py —— 确定 OpenOrganelle zarr 的 chunk key 分隔符

## 为什么要单独探这个

zarr v2 的 chunk 键由 `.zarray` 里的 `dimension_separator` 决定：点号（旧默认）
或斜杠（新默认）。取错分隔符的表现是**每个块都 404**，而代码本身不报错，
只是结果全零——这类"静默失败"最难查，所以值得先用一个小脚本探明，
而不是跑了十分钟才发现拿到空数组。

实测：OpenOrganelle 的 groundtruth crop 用**斜杠**（`0/0/0`），
而 ASEM 的体数据用**点号**（`0.0.0`）。两者不通用。
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.request

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
B = "https://janelia-cosem-datasets.s3.us-east-1.amazonaws.com/"

DS = "jrc_hela-2"
CROP = "crop1"
CH = "golgi"


def fetch(url: str, n: int = 64, timeout: int = 40):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.read(n), None
    except Exception as exc:  # noqa: BLE001
        return None, type(exc).__name__


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    base = (f"{B}{DS}/{DS}.zarr/recon-1/labels/groundtruth/"
            f"{CROP}/{CH}/s0/")
    raw, err = fetch(base + ".zarray", 4096)
    if err:
        print(f"读 .zarray 失败: {err}")
        return 1
    z = json.loads(raw.decode("utf-8"))
    sep = z.get("dimension_separator", ".")
    print(f".zarray: chunks={z['chunks']} dtype={z['dtype']} "
          f"compressor={(z.get('compressor') or {}).get('id')} "
          f"separator={sep!r}")

    for name, s in (("点号 '.'", "."), ("斜杠 '/'", "/"), ("下划线 '_'", "_")):
        key = s.join(["0", "0", "0"])
        data, e = fetch(base + key, 32)
        mark = f"OK 前8字节={data[:8].hex()}" if data else f"失败 {e}"
        print(f"  {name:<12} key={key:<8} {mark}")

    print()
    print(f"结论：应使用 {sep!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
