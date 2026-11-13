#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oo_fetch.py —— 从 OpenOrganelle 的 N5 标注里取子区域并保存为 .npy

## 与 ASEM 的 asem_fetch 的区别

ASEM 用 zarr（`s0/z.y.x` 块 + zlib），OpenOrganelle 用 **N5**：

    <ds>/<ds>.n5/labels/<label>/s0/<bx>/<by>/<bz>

* 块坐标是**块索引**（blockSize 的倍数），不是体素索引
* 压缩是 **gzip**，数据结构是原始字节流
* `attributes.json` 里有 `dimensions`（顺序为 x,y,z）、`blockSize`、`dataType`

## 为什么只取子区域

HeLa 数据集的 EM 体是 6368×1600×12000，全量 122 G 体素。但高尔基体标注
只覆盖很小一块（实测 31 个块、20 MB），完全不必要也不应该拉全量。

## 用法

    python tools/oo_fetch.py --dataset jrc_hela-2 --labels golgi_seg,er_golgi_contacts \
        --x 5120 6144 --y 0 512 --z 2048 3584 --out data/oo/hela2
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

B = "https://janelia-cosem-datasets.s3.us-east-1.amazonaws.com/"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DTYPE = {"uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32,
         "int16": np.int16, "float32": np.float32}


def get_bytes(url: str, timeout: int = 120, retries: int = 3) -> Optional[bytes]:
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.read()
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(1.5 * (k + 1))
    return None


def get_json(url: str) -> Optional[dict]:
    b = get_bytes(url)
    if not b:
        return None
    try:
        return json.loads(b.decode("utf-8"))
    except Exception:
        return None


def n5_attrs(ds: str, label: str) -> Optional[Dict]:
    return get_json(f"{B}{ds}/{ds}.n5/labels/{label}/s0/attributes.json")


def fetch_n5(ds: str, label: str, xr: Tuple[int, int], yr: Tuple[int, int],
             zr: Tuple[int, int], attrs: Dict, workers: int = 8
             ) -> Tuple[np.ndarray, Dict]:
    """按体素范围取 N5 子区域。返回 (数组, 统计)。

    N5 的 `dimensions` 是 **x,y,z** 顺序，而拼出来的数组按 (z,y,x) 存，
    与 zarr 那套习惯一致，便于后续与 EM 对齐。
    """
    block = attrs["blockSize"]                    # [bx, by, bz]，N5 里通常各向同性
    dims = attrs["dimensions"]                    # [nx, ny, nz]
    dt = DTYPE.get(attrs["dataType"], np.uint8)
    bx, by, bz = block

    x0, x1 = max(0, xr[0]), min(dims[0], xr[1])
    y0, y1 = max(0, yr[0]), min(dims[1], yr[1])
    z0, z1 = max(0, zr[0]), min(dims[2], zr[1])
    size = (z1 - z0, y1 - y0, x1 - x0)
    out = np.zeros(size, dt)
    got = {"ok": 0, "miss": 0, "bytes": 0}

    tasks = []
    for ix in range(x0 // bx, (x1 - 1) // bx + 1):
        for iy in range(y0 // by, (y1 - 1) // by + 1):
            for iz in range(z0 // bz, (z1 - 1) // bz + 1):
                tasks.append((ix, iy, iz))

    def one(t):
        ix, iy, iz = t
        url = f"{B}{ds}/{ds}.n5/labels/{label}/s0/{ix}/{iy}/{iz}"
        raw = get_bytes(url)
        if raw is None:
            got["miss"] += 1
            return t, None
        got["ok"] += 1
        got["bytes"] += len(raw)
        try:
            buf = gzip.decompress(raw)
        except Exception:
            buf = raw
        # 该块可能被裁到体积边界内，按实际能容纳的形状还原
        n_expected = bx * by * bz
        arr = np.frombuffer(buf, dtype=dt)
        if arr.size == n_expected:
            arr = arr.reshape((bz, by, bx))     # 转成 (z,y,x)
        else:
            # 边缘块：按实际元素数反推
            nz = arr.size // (bx * by)
            if nz * bx * by != arr.size:
                return t, None
            arr = arr.reshape((nz, by, bx))
        return t, arr

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(one, t) for t in tasks]):
            t, arr = f.result()
            if arr is None:
                continue
            ix, iy, iz = t
            gx, gy, gz = ix * bx, iy * by, iz * bz
            az, ay, ax = arr.shape
            sx0 = max(gx, x0); sx1 = min(gx + ax, x1)
            sy0 = max(gy, y0); sy1 = min(gy + ay, y1)
            sz0 = max(gz, z0); sz1 = min(gz + az, z1)
            if sx0 >= sx1 or sy0 >= sy1 or sz0 >= sz1:
                continue
            out[sz0 - z0:sz1 - z0, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                arr[sz0 - gz:sz1 - gz, sy0 - gy:sy1 - gy, sx0 - gx:sx1 - gx]
    return out, got


def fetch_em(ds: str, xr: Tuple[int, int], yr: Tuple[int, int],
             zr: Tuple[int, int], workers: int = 8
             ) -> Tuple[np.ndarray, Dict]:
    """取对应的 EM 体子区域。EM 在 zarr 下，块为 128³。"""
    base = f"{B}{ds}/{ds}.zarr/recon-1/em/fibsem-uint8/s0/"
    z = get_json(base + ".zarray")
    if not z:
        raise RuntimeError("读不到 EM 的 .zarray")
    shape = z["shape"]           # (z,y,x)
    chunks = z["chunks"]
    cz, cy, cx = chunks
    z0, z1 = max(0, zr[0]), min(shape[0], zr[1])
    y0, y1 = max(0, yr[0]), min(shape[1], yr[1])
    x0, x1 = max(0, xr[0]), min(shape[2], xr[1])
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
    got = {"ok": 0, "miss": 0, "bytes": 0}

    tasks = []
    for iz in range(z0 // cz, (z1 - 1) // cz + 1):
        for iy in range(y0 // cy, (y1 - 1) // cy + 1):
            for ix in range(x0 // cx, (x1 - 1) // cx + 1):
                tasks.append((ix, iy, iz))

    def one(t):
        ix, iy, iz = t
        # 该 zarr 的 dimension_separator 是 "/"，且压缩为 zstd（不是 zlib）。
        raw = get_bytes(f"{base}{iz}/{iy}/{ix}")
        if raw is None:
            got["miss"] += 1
            return t, None
        got["ok"] += 1
        got["bytes"] += len(raw)
        try:
            import zstandard as zstd
            raw = zstd.ZstdDecompressor().decompress(raw)
        except Exception:
            try:
                import zlib
                raw = zlib.decompress(raw)
            except Exception:
                pass
        arr = np.frombuffer(raw, dtype=np.uint8)
        n = cz * cy * cx
        if arr.size != n:
            return t, None
        return t, arr.reshape((cz, cy, cx))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(one, t) for t in tasks]):
            t, arr = f.result()
            if arr is None:
                continue
            ix, iy, iz = t
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
    ap.add_argument("--labels", default="golgi_seg",
                    help="逗号分隔，如 golgi_seg,er_seg,er_golgi_contacts")
    ap.add_argument("--x", nargs=2, type=int, required=True)
    ap.add_argument("--y", nargs=2, type=int, required=True)
    ap.add_argument("--z", nargs=2, type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-em", action="store_true", default=True)
    ap.add_argument("--no-em", dest="with_em", action="store_false")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    outdir = os.path.join(ROOT, args.out)
    os.makedirs(outdir, exist_ok=True)
    xr, yr, zr = tuple(args.x), tuple(args.y), tuple(args.z)
    print(f"数据集 {args.dataset}  区域 x{xr} y{yr} z{zr}")

    if args.with_em:
        p = os.path.join(outdir, "em.npy")
        if os.path.exists(p):
            print("  已有 em.npy，跳过")
        else:
            t0 = time.time()
            arr, got = fetch_em(args.dataset, xr, yr, zr, args.workers)
            np.save(p, arr)
            print(f"  EM: {arr.shape} uint8  {got['bytes']/1e6:.1f} MB  "
                  f"块 ok={got['ok']} miss={got['miss']}  {time.time()-t0:.0f}s")

    for lab in [s.strip() for s in args.labels.split(",") if s.strip()]:
        attrs = n5_attrs(args.dataset, lab)
        if not attrs:
            print(f"  [跳过] {lab}: 读不到 attributes.json")
            continue
        p = os.path.join(outdir, f"{lab}.npy")
        if os.path.exists(p):
            print(f"  已有 {lab}.npy，跳过")
            continue
        t0 = time.time()
        arr, got = fetch_n5(args.dataset, lab, xr, yr, zr, attrs, args.workers)
        np.save(p, arr)
        nz = float((arr > 0).mean())
        uniq = np.unique(arr)
        print(f"  {lab}: {arr.shape} {arr.dtype}  前景 {100*nz:.2f}%  "
              f"取值数 {len(uniq)}  最大 {int(arr.max())}  "
              f"{got['bytes']/1e6:.1f} MB  {time.time()-t0:.0f}s")

    print(f"\n输出目录 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
