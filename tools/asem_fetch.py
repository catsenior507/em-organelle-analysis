#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asem_fetch.py —— 从 ASEM 的公开 S3 拉取指定体数据并组装为 .npy

## 背景

ASEM（哈佛 Kirchhausen 实验室）的体数据以 **zarr v2 + zlib chunk** 形式存放在
`asem-project` 桶里，无需 AWS 账号。标注体往往只覆盖 raw 的一个子区域，
偏移量记录在 `.zattrs` 的 `offset`（单位 nm），需与 raw 的 offset 相减再除以
resolution 才能得到**索引偏移**。`tools/asem_inspect.py` 已把各数据集的偏移算好：

    61/golgi  raw 偏移 [1000, 410, 2597]
    64/golgi  raw 偏移 [ 777, 291,  954]
    64/er,mito 同 ROI，raw 偏移 [400, 326, 2212]

## 这个脚本做什么

只下载**需要的 chunk**（不是整个桶），按 zarr 的 C order 拼成
`(z, y, x)` 的三维数组，存为 `.npy`。对 raw 会自动取与标注对齐的子区域。

## 为什么要按 chunk 拉而不是整卷

61 的 raw 是 1805×1400×4800（5 nm），整卷 = 12.1 G 体素；而标注只覆盖
210×283×371 的一小块。整卷下载既慢（服务器到 S3 只有 0.12–0.37 MB/s）
也没必要。这里只取与标注对齐的那部分。
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import zlib

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

BUCKET = "https://asem-project.s3.amazonaws.com"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_bytes(url: str, timeout: int = 90, retries: int = 3) -> Optional[bytes]:
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=ctx) as r:
                return r.read()
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(2 * (k + 1))
    return None


def zjson(url: str) -> Optional[dict]:
    b = get_bytes(url)
    if b is None:
        return None
    try:
        return json.loads(b.decode("utf-8"))
    except Exception:
        return None


def vol_meta(ds: str, vol: str) -> Optional[Dict]:
    base = f"{BUCKET}/datasets/{ds}/{ds}.zarr/neuroglancer_volumes/{vol}"
    a = zjson(f"{base}/s0/.zattrs")
    z = zjson(f"{base}/s0/.zarray")
    if a is None or z is None:
        return None
    return {"base": base, "offset": a.get("offset"), "res": a.get("resolution"),
            "shape": z.get("shape"), "chunks": z.get("chunks"),
            "dtype": z.get("dtype"), "compressor": z.get("compressor")}


def aligned_bbox(ds: str, label_vol: str, raw_vol: str = "raw"
                 ) -> Tuple[List[int], List[int], Dict]:
    """返回 (raw 起点, 尺寸, 信息)。尺寸取标注体的完整 extent。"""
    L = vol_meta(ds, label_vol)
    R = vol_meta(ds, raw_vol)
    if L is None or R is None:
        raise RuntimeError(f"无法读取 {ds} 的元数据")
    off_l, off_r = L["offset"], R["offset"]
    res = R["res"]
    start = [int(round((off_l[k] - off_r[k]) / res[k])) for k in range(3)]
    size = list(L["shape"])
    return start, size, {"label": L, "raw": R, "res": res}


def fetch_volume(base: str, shape: List[int], chunks: List[int],
                 start: List[int], size: List[int],
                 dtype: str, compressor: Optional[dict],
                 workers: int = 8) -> Tuple[np.ndarray, Dict[str, int]]:
    """按需下载 chunk 并拼装。`start/size` 是相对于该卷自身坐标的窗口。

    返回 (拼装好的数组, 下载统计)。统计里 `miss` 是被填充 0 的 chunk 数——
    zarr 对全零 chunk 允许不落盘，所以 miss 不一定是错误，
    但如果 miss 比例很高就要怀疑坐标算错了。
    """
    np_dtype = np.dtype(dtype.replace("<", "<").replace("|", "|"))
    out = np.zeros(tuple(size), dtype=np_dtype)
    ck = chunks
    # 需要哪些 chunk
    z0, y0, x0 = start
    z1, y1, x1 = z0 + size[0], y0 + size[1], x0 + size[2]
    cz = range(z0 // ck[0], (z1 - 1) // ck[0] + 1)
    cy = range(y0 // ck[1], (y1 - 1) // ck[1] + 1)
    cx = range(x0 // ck[2], (x1 - 1) // ck[2] + 1)
    tasks = [(a, b, c) for a in cz for b in cy for c in cx]
    got = {"ok": 0, "miss": 0, "bytes": 0}

    def one(t):
        a, b, c = t
        url = f"{base}/s0/{a}.{b}.{c}"
        raw = get_bytes(url)
        if raw is None:
            got["miss"] += 1
            return t, None
        got["ok"] += 1
        got["bytes"] += len(raw)
        if compressor and compressor.get("id") == "zlib":
            raw = zlib.decompress(raw)
        arr = np.frombuffer(raw, dtype=np_dtype)
        n = ck[0] * ck[1] * ck[2]
        if arr.size != n:
            # 边缘 chunk 可能小于满尺寸，按实际大小重塑
            side = arr.size
            import math
            s = int(round(side ** (1 / 3)))
            if s ** 3 == side:
                arr = arr.reshape(s, s, s)
            else:
                return t, None
        else:
            arr = arr.reshape(ck[0], ck[1], ck[2])
        return t, arr

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, t) for t in tasks]
        for f in as_completed(futs):
            t, arr = f.result()
            if arr is None:
                continue
            a, b, c = t
            gz, gy, gx = a * ck[0], b * ck[1], c * ck[2]
            # chunk 在卷坐标中的实际范围
            ach, bch, cch = arr.shape
            # 与目标窗口求交
            sz0 = max(gz, z0); sz1 = min(gz + ach, z1)
            sy0 = max(gy, y0); sy1 = min(gy + bch, y1)
            sx0 = max(gx, x0); sx1 = min(gx + cch, x1)
            if sz0 >= sz1 or sy0 >= sy1 or sx0 >= sx1:
                continue
            out[sz0 - z0:sz1 - z0, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                arr[sz0 - gz:sz1 - gz, sy0 - gy:sy1 - gy, sx0 - gx:sx1 - gx]
    return out, got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="如 61 或 64")
    ap.add_argument("--labels", required=True,
                    help="逗号分隔的标注卷名，如 golgi 或 er,mito")
    ap.add_argument("--out", default="data/asem")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    outdir = os.path.join(ROOT, args.out, f"ds{args.dataset}")
    os.makedirs(outdir, exist_ok=True)

    labs = [s.strip() for s in args.labels.split(",") if s.strip()]

    # raw 必须**按标注体的 bbox 分别切**。
    #
    # 踩过的坑：最初 raw 只按数据集命名（raw.npy），而同一个数据集的
    # 不同标注体往往落在不同 ROI —— 数据集 64 的 golgi 在
    # raw_index_offset [777,291,954]、尺寸 284×204×424，
    # 而 er/mito 在 [400,326,2212]、尺寸 241×476×528。
    # 于是先跑 golgi 生成的 raw.npy 被后续 er/mito 复用，
    # 两者的形状都对不上（(284,204,424) vs (241,476,528)），标签与图像错位。
    # 现在 raw 以"首个标注体"为键命名，转换脚本按同一键取用。
    raw_path = os.path.join(outdir, f"raw__{labs[0]}.npy")
    start, size, info = aligned_bbox(args.dataset, f"labels/{labs[0]}")
    print(f"数据集 {args.dataset}  标注 {labs[0]} 的 raw 窗口 "
          f"start={start} size={size}")
    print(f"  分辨率 {info['res']} nm")

    R = info["raw"]
    if os.path.exists(raw_path):
        print(f"  已有 {os.path.basename(raw_path)}，跳过")
    else:
        t0 = time.time()
        arr, got = fetch_volume(R["base"], R["shape"], R["chunks"],
                                start, size, R["dtype"], R["compressor"],
                                args.workers)
        np.save(raw_path, arr)
        print(f"  raw: {arr.shape} {arr.dtype}  "
              f"chunk ok={got['ok']} miss={got['miss']} "
              f"{got['bytes']/1e6:.1f} MB  {time.time()-t0:.0f}s")

    # 各标注
    for lab in labs:
        volname = f"labels/{lab}"
        s2, sz2, info2 = aligned_bbox(args.dataset, volname)
        L = info2["label"]
        p = os.path.join(outdir, f"{lab}.npy")
        if os.path.exists(p):
            print(f"  已有 {os.path.basename(p)}，跳过")
            continue
        t0 = time.time()
        arr, got = fetch_volume(L["base"], L["shape"], L["chunks"],
                                [0, 0, 0], sz2, L["dtype"], L["compressor"],
                                args.workers)
        np.save(p, arr)
        nz = float((arr > 0).mean())
        print(f"  {lab}: {arr.shape} {arr.dtype} 前景 {100*nz:.2f}%  "
              f"chunk ok={got['ok']} miss={got['miss']} "
              f"{got['bytes']/1e6:.1f} MB  {time.time()-t0:.0f}s")
        # 记录对齐信息，供后续切片使用
        with open(os.path.join(outdir, f"{lab}_align.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"dataset": args.dataset, "label": volname,
                       "label_offset_nm": L["offset"],
                       "raw_offset_nm": info["raw"]["offset"],
                       "resolution_nm": info["res"],
                       "raw_index_offset": s2, "label_shape": sz2},
                      fh, ensure_ascii=False, indent=1)
    print(f"\n输出目录 {os.path.relpath(outdir, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
