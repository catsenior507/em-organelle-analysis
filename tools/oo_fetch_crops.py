#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oo_fetch_crops.py —— 一次性取四个 Golgi crop 的标注与 EM（带清单优化）

## 为什么需要"先列清单"

zarr 省略全零块，所以照网格逐块请求时**大量请求是 404**。实测 crop4 的
golgi：31 个块存在、567 个不存在——三分之二是白跑。而 S3 的 404 与 200
耗时相近，于是时间浪费在等"确认不存在"上。

改为先用 `list-type=2` 一次列出该通道下**实际存在**的键，再只取这些键：

* 请求数从"网格块数"降到"实有块数"
* 4 个 crop × 3 通道 × 约 800 块 → 从上万次降到约 3000 次

## 四个 crop

来自 `logs/oo_locate.json`：真实有 golgi 数据、且同 crop 内含 er/mito 的四个。
坐标已用出图核对过（`temp/oo_align3/align.png`：标注精确覆盖池层带）。

## 断点续传

每个通道存一个 `.npy`，已存在就跳过；EM 同理。中途断开重跑不会重下。

## 用法

    python tools/oo_fetch_crops.py --out data/oo --workers 16
    python tools/oo_fetch_crops.py --out data/oo --labels-only
    python tools/oo_fetch_crops.py --out data/oo --em-only --crop hela-2_crop4
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

# 四个可用 crop：数据集、crop 名、EM 原点 (z,y,x)、形状 (z,y,x)
CROPS = [
    ("jrc_hela-2", "crop4", (3465, 50, 5950), (476, 600, 600)),
    ("jrc_jurkat-1", "crop37", (4437, 696, 3118), (400, 400, 400)),
    ("jrc_macrophage-2", "crop39", (4405, 603, 5435), (400, 400, 400)),
    ("jrc_macrophage-2", "crop48", (4482, 451, 5979), (512, 512, 512)),
]


def slug(ds: str, crop: str) -> str:
    return f"{ds.replace('jrc_', '')}_{crop}"


def get_bytes(url: str, timeout: int = 120, retries: int = 3) -> Optional[bytes]:
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.read()
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(1.0 * (k + 1))
    return None


def get_json(url: str) -> Optional[dict]:
    b = get_bytes(url)
    if not b:
        return None
    try:
        return json.loads(b.decode("utf-8"))
    except Exception:
        return None


def list_keys(prefix: str) -> List[Tuple[str, int]]:
    """列出前缀下所有对象 (相对键, 字节数)。分页处理超过 1000 个的情况。"""
    if prefix.startswith(B):
        prefix = prefix[len(B):]
    out: List[Tuple[str, int]] = []
    token = None
    while True:
        url = f"{B}?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        b = get_bytes(url, timeout=90)
        if not b:
            break
        t = b.decode("utf-8", "replace")
        keys = re.findall(r"<Key>([^<]+)</Key>", t)
        sizes = [int(x) for x in re.findall(r"<Size>(\d+)</Size>", t)]
        for k, s in zip(keys, sizes):
            out.append((k[len(prefix):], s))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", t)
        if not m:
            break
        token = m.group(1)
    return out


def decompress(raw: bytes, cid: str) -> bytes:
    if cid == "zstd":
        return zstd.ZstdDecompressor().decompress(raw)
    if cid == "zlib":
        import zlib
        return zlib.decompress(raw)
    if cid == "gzip":
        import gzip
        return gzip.decompress(raw)
    return raw


def fetch_zarr_volume(base: str, shape: Tuple[int, int, int],
                      want: Optional[Tuple] = None, workers: int = 16,
                      dense: bool = False) -> Tuple[np.ndarray, Dict]:
    """通用：取一个 zarr 体（或子区域）。

    `dense=False`（标注）：先列清单，只取存在的块。标注稀疏，全零块不写，
    逐网格请求会有三分之二是 404。

    `dense=True`（EM）：**不列清单**，直接按网格取。EM 是稠密图像体，
    块几乎全部存在；而列清单要翻遍整卷的分块键——实测 hela-2 的 EM 有
    127 033 个键，分页耗时 495–721 秒，比取数据本身慢十倍。
    """
    z = get_json(base + ".zarray")
    if not z:
        raise RuntimeError(f"读不到 {base}.zarray")
    chunks = z["chunks"]
    dt = DTYPE.get(z.get("dtype"), np.uint8)
    cid = (z.get("compressor") or {}).get("id", "")
    cz, cy, cx = chunks

    if want:
        z0, z1, y0, y1, x0, x1 = want
        z0 = max(0, z0); y0 = max(0, y0); x0 = max(0, x0)
        z1 = min(shape[0], z1); y1 = min(shape[1], y1); x1 = min(shape[2], x1)
    else:
        z0, y0, x0 = 0, 0, 0
        z1, y1, x1 = shape
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), dt)
    got = {"ok": 0, "bytes": 0, "listed": 0}
    t0 = time.time()

    # 只列需要的 z 范围内的块，避免列整卷。
    #
    # 注意：list-type=2 的 prefix 参数要的是**键前缀**（形如
    # jrc_hela-2/jrc_hela-2.zarr/...），不是完整 URL。早期版本直接把
    # base（含 https://...）传进去，S3 按字面量匹配于是返回 0 条，
    # 随后 todo 为空、保存了全零数组 —— 三步都没报错。
    todo = []
    if dense:
        # 稠密体（EM）：直接按网格生成，**不列清单**。
        # EM 的块几乎全部存在，而列清单要翻遍整卷的分块键——实测 hela-2
        # 有 127 033 个键，分页耗时 495–721 秒，比取数据本身慢十倍。
        for iz in range(z0 // cz, (z1 - 1) // cz + 1):
            for iy in range(y0 // cy, (y1 - 1) // cy + 1):
                for ix in range(x0 // cx, (x1 - 1) // cx + 1):
                    todo.append((f"{iz}/{iy}/{ix}", iz, iy, ix,
                                 iz * cz, iy * cy, ix * cx))
    else:
        # 稀疏体（标注）：先列清单，只取存在的块。三分之二的网格是空的。
        key_prefix = base[len(B):] if base.startswith(B) else base
        keys = list_keys(key_prefix)
        got["listed"] = len(keys)
        for rel, _size in keys:
            if rel.endswith((".zarray", ".zattrs", ".zgroup")) or "/" not in rel:
                continue
            parts = rel.strip("/").split("/")
            if len(parts) != 3:
                continue
            try:
                iz, iy, ix = (int(p) for p in parts)
            except ValueError:
                continue
            gz, gy, gx = iz * cz, iy * cy, ix * cx
            # 与目标窗口求交，全不相交则跳过
            if gz + cz <= z0 or gz >= z1 or gy + cy <= y0 or gy >= y1 \
                    or gx + cx <= x0 or gx >= x1:
                continue
            todo.append((rel, iz, iy, ix, gz, gy, gx))

    print(f"    清单 {got['listed']} 项，窗口内需取 {len(todo)} 块 "
          f"（{time.time()-t0:.0f}s）")

    def one(item):
        rel, iz, iy, ix, gz, gy, gx = item
        raw = get_bytes(base + rel)
        if raw is None:
            return item, None
        got["ok"] += 1
        got["bytes"] += len(raw)
        try:
            buf = decompress(raw, cid)
        except Exception:
            return item, None
        arr = np.frombuffer(buf, dtype=dt)
        if arr.size != cz * cy * cx:
            return item, None
        return item, arr.reshape((cz, cy, cx))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(one, t) for t in todo]):
            item, arr = f.result()
            if arr is None:
                continue
            _, _, _, _, gz, gy, gx = item
            sz0 = max(gz, z0); sz1 = min(gz + cz, z1)
            sy0 = max(gy, y0); sy1 = min(gy + cy, y1)
            sx0 = max(gx, x0); sx1 = min(gx + cx, x1)
            if sz0 >= sz1 or sy0 >= sy1 or sx0 >= sx1:
                continue
            out[sz0 - z0:sz1 - z0, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                arr[sz0 - gz:sz1 - gz, sy0 - gy:sy1 - gy, sx0 - gx:sx1 - gx]
    return out, got


def label_base(ds: str, crop: str, ch: str) -> str:
    return f"{B}{ds}/{ds}.zarr/recon-1/labels/groundtruth/{crop}/{ch}/s0/"


def em_base(ds: str) -> str:
    return f"{B}{ds}/{ds}.zarr/recon-1/em/fibsem-uint8/s0/"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/oo")
    ap.add_argument("--channels", default="golgi,er,mito")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--em-only", action="store_true")
    ap.add_argument("--crop", default=None, help="只处理某个 slug，如 hela-2_crop4")
    args = ap.parse_args(argv)

    outroot = os.path.join(ROOT, args.out)
    os.makedirs(outroot, exist_ok=True)
    chans = [c.strip() for c in args.channels.split(",") if c.strip()]
    targets = [c for c in CROPS if args.crop is None or slug(c[0], c[1]) == args.crop]
    if not targets:
        print(f"[错误] 没有匹配的 crop：{args.crop}")
        return 2

    total_bytes = 0
    t_start = time.time()
    for ds, crop, origin, shape in targets:
        sl = slug(ds, crop)
        d = os.path.join(outroot, sl)
        os.makedirs(d, exist_ok=True)
        print(f"\n=== {ds}/{crop}  {shape}  原点 {origin}")

        meta = {"dataset": ds, "crop": crop, "origin_zyx": list(origin),
                "shape_zyx": list(shape), "channels": chans}
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)

        if not args.em_only:
            for ch in chans:
                p = os.path.join(d, f"{ch}.npy")
                if os.path.exists(p):
                    print(f"  {ch}: 已有，跳过")
                    continue
                t0 = time.time()
                try:
                    arr, got = fetch_zarr_volume(label_base(ds, crop, ch),
                                                 shape, None, args.workers)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {ch}: 失败 {type(exc).__name__}")
                    continue
                np.save(p, arr)
                total_bytes += got["bytes"]
                print(f"  {ch}: {arr.shape} {arr.dtype}  前景 "
                      f"{100*float((arr>0).mean()):.2f}%  "
                      f"{got['bytes']/1e6:.1f} MB  ok={got['ok']}  "
                      f"{time.time()-t0:.0f}s")

        if not args.labels_only:
            p = os.path.join(d, "em.npy")
            if os.path.exists(p):
                print("  em: 已有，跳过")
            else:
                t0 = time.time()
                arr, got = fetch_zarr_volume(em_base(ds), shape, None, args.workers)
                np.save(p, arr)
                total_bytes += got["bytes"]
                print(f"  em: {arr.shape} uint8  均值 {arr.mean():.0f}  "
                      f"{got['bytes']/1e6:.1f} MB  ok={got['ok']}  "
                      f"{time.time()-t0:.0f}s")

    print()
    print("=" * 68)
    print(f"本次下载 {total_bytes/1e6:.1f} MB，用时 {(time.time()-t_start)/60:.1f} 分钟")
    print(f"输出 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
