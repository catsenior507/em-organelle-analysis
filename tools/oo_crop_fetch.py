#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oo_crop_fetch.py —— 从 OpenOrganelle 的 **groundtruth crop** 里取多类标注

## 为什么用这条路径而不是 .n5

OpenOrganelle 有两套标注存储：

* `<ds>/<ds>.n5/labels/<类型>` —— N5，块头是 z5 的自定义头（不是裸 gzip），
  逆向成本高，且是全卷索引；
* `<ds>/<ds>.zarr/recon-1/labels/groundtruth/<crop>/<细胞器>/s0/` ——
  **标准 zarr v2 + zstd**，每个 crop 是一个小的标注子体积。

后者是 CellMap 的官方评测格式，也是人工标注的原始 crop，**直接可用**。
每个 crop 有 64 个通道，含 `golgi`、`er`、`mito`、`ld`、`lyso`、`perox` 等。

## 数据规模（实测）

| 数据集 | crop 数 |
|---|---|
| jrc_hela-2 | 26 |
| jrc_hela-3 | 19 |
| jrc_jurkat-1 | 20 |
| jrc_macrophage-2 | 18 |
| **合计** | **83** |

每个 crop 约 200×1000×1000 体素（64³ 块、约 1024 块/通道）。
按每 crop 取 4 个通道估算，全取约需数 GB；本脚本支持按 z 范围截取。

## 用法

    # 列出一个数据集有哪些 crop、每个 crop 有哪些通道
    python tools/oo_crop_fetch.py --dataset jrc_hela-2 --list

    # 取某个 crop 的 golgi/er/mito，只取前 64 层
    python tools/oo_crop_fetch.py --dataset jrc_hela-2 --crop crop1 \
        --channels golgi,er,mito --z 0 64 --out data/oo/hela2_crop1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import zstandard as zstd

B = "https://janelia-cosem-datasets.s3.us-east-1.amazonaws.com/"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DTYPE = {"|u1": np.uint8, "<u2": np.uint16, "|u2": np.uint16,
         "<u4": np.uint32, "|f4": np.float32}


def get_bytes(url: str, timeout: int = 120, retries: int = 3) -> Optional[bytes]:
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.read()
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(1.2 * (k + 1))
    return None


def get_json(url: str) -> Optional[dict]:
    b = get_bytes(url)
    if not b:
        return None
    try:
        return json.loads(b.decode("utf-8"))
    except Exception:
        return None


def list_prefixes(prefix: str) -> List[str]:
    b = get_bytes(f"{B}?list-type=2&delimiter=/&prefix={prefix}&max-keys=1000")
    if not b:
        return []
    txt = b.decode("utf-8", "replace")
    return [p[len(prefix):].rstrip("/")
            for p in re.findall(r"<Prefix>([^<]+)</Prefix>", txt)
            if p[len(prefix):].rstrip("/")]


def gt_base(ds: str, crop: str, ch: str) -> str:
    return f"{B}{ds}/{ds}.zarr/recon-1/labels/groundtruth/{crop}/{ch}/s0/"


def decompress(raw: bytes, compressor: dict) -> bytes:
    cid = (compressor or {}).get("id", "")
    if cid == "zstd":
        return zstd.ZstdDecompressor().decompress(raw)
    if cid == "zlib":
        import zlib
        return zlib.decompress(raw)
    if cid == "gzip":
        import gzip
        return gzip.decompress(raw)
    return raw


def fetch_channel(ds: str, crop: str, ch: str,
                  zr: Optional[Tuple[int, int]] = None,
                  yr: Optional[Tuple[int, int]] = None,
                  xr: Optional[Tuple[int, int]] = None,
                  workers: int = 8) -> Tuple[np.ndarray, Dict]:
    base = gt_base(ds, crop, ch)
    z = get_json(base + ".zarray")
    if not z:
        raise RuntimeError(f"读不到 {crop}/{ch} 的 .zarray")
    shape = z["shape"]          # (z,y,x)
    chunks = z["chunks"]
    dt = DTYPE.get(z.get("dtype"), np.uint8)
    comp = z.get("compressor") or {}
    cz, cy, cx = chunks

    z0, z1 = (0, shape[0]) if zr is None else (max(0, zr[0]), min(shape[0], zr[1]))
    y0, y1 = (0, shape[1]) if yr is None else (max(0, yr[0]), min(shape[1], yr[1]))
    x0, x1 = (0, shape[2]) if xr is None else (max(0, xr[0]), min(shape[2], xr[1]))
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), dt)
    got = {"ok": 0, "miss": 0, "bytes": 0}

    tasks = []
    for iz in range(z0 // cz, (z1 - 1) // cz + 1):
        for iy in range(y0 // cy, (y1 - 1) // cy + 1):
            for ix in range(x0 // cx, (x1 - 1) // cx + 1):
                tasks.append((iz, iy, ix))

    def one(t):
        iz, iy, ix = t
        # chunk key 用斜杠：该数据集 .zarray 的 dimension_separator 为 "/"。
        # 用点号会全部 404 —— 表现为“每个块都缺”，但代码不会报错。
        raw = get_bytes(f"{base}{iz}/{iy}/{ix}")
        if raw is None:
            got["miss"] += 1
            return t, None
        got["ok"] += 1
        got["bytes"] += len(raw)
        try:
            buf = decompress(raw, comp)
        except Exception:
            return t, None
        arr = np.frombuffer(buf, dtype=dt)
        if arr.size != cz * cy * cx:
            return t, None
        return t, arr.reshape((cz, cy, cx))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(one, t) for t in tasks]):
            t, arr = f.result()
            if arr is None:
                continue
            iz, iy, ix = t
            gz, gy, gx = iz * cz, iy * cy, ix * cx
            sz0 = max(gz, z0); sz1 = min(gz + cz, z1)
            sy0 = max(gy, y0); sy1 = min(gy + cy, y1)
            sx0 = max(gx, x0); sx1 = min(gx + cx, x1)
            if sz0 >= sz1 or sy0 >= sy1 or sx0 >= sx1:
                continue
            out[sz0 - z0:sz1 - z0, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                arr[sz0 - gz:sz1 - gz, sy0 - gy:sy1 - gy, sx0 - gx:sx1 - gx]
    return out, got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--crop", default=None)
    ap.add_argument("--channels", default="golgi,er,mito")
    ap.add_argument("--z", nargs=2, type=int, default=None)
    ap.add_argument("--y", nargs=2, type=int, default=None)
    ap.add_argument("--x", nargs=2, type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    ds = args.dataset
    gbase = f"{ds}/{ds}.zarr/recon-1/labels/groundtruth/"

    if args.list:
        crops = list_prefixes(gbase)
        print(f"{ds}: {len(crops)} 个 crop")
        for c in crops:
            chans = list_prefixes(f"{gbase}{c}/")
            print(f"  {c:<10} {len(chans)} 通道: {' '.join(chans[:10])}"
                  f"{' ...' if len(chans) > 10 else ''}")
        if crops:
            chans = list_prefixes(f"{gbase}{crops[0]}/")
            print(f"\n完整通道列表（{crops[0]}）:")
            for i in range(0, len(chans), 4):
                print("  " + "  ".join(f"{o:<12}" for o in chans[i:i + 4]))
        return 0

    if not args.crop:
        print("[错误] 需要 --crop（或用 --list 先看有哪些）")
        return 2
    if not args.out:
        args.out = f"data/oo/{ds}_{args.crop}"

    outdir = os.path.join(ROOT, args.out)
    os.makedirs(outdir, exist_ok=True)
    print(f"{ds} / {args.crop}  z={args.z} y={args.y} x={args.x}")

    total = 0
    for ch in [s.strip() for s in args.channels.split(",") if s.strip()]:
        p = os.path.join(outdir, f"{ch}.npy")
        if os.path.exists(p):
            print(f"  已有 {ch}.npy，跳过")
            continue
        t0 = time.time()
        arr, got = fetch_channel(ds, args.crop, ch,
                                 tuple(args.z) if args.z else None,
                                 tuple(args.y) if args.y else None,
                                 tuple(args.x) if args.x else None,
                                 args.workers)
        np.save(p, arr)
        total += got["bytes"]
        nz = float((arr > 0).mean())
        print(f"  {ch:<8} {arr.shape} {arr.dtype}  前景 {100*nz:.2f}%  "
              f"{got['bytes']/1e6:.1f} MB  块 ok={got['ok']} miss={got['miss']}  "
              f"{time.time()-t0:.0f}s")

    print(f"\n输出 {args.out}  共 {total/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
