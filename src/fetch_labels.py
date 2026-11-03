#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_labels.py —— 直接从 S3 拉 CellMap 的**标签掩码**（不碰 42 GB zip）

## 为什么只拉标签

接触位点分析（`tools/interaction/contact_analysis.py`）只需要**各类别的掩码**：
mito 在哪、er 在哪。EM 灰度图只有训练检测器时才需要。

而 CellMap 的标签在 S3 上是按 crop 组织的小数组：

    <ds>.zarr/recon-1/labels/groundtruth/crop<ID>/<label>/s0/
        .zarray  .zgroup  .zattrs     元数据
        0/0/0  0/0/1 ...              zstd 压缩的 chunk

每个 crop 的单个类别约 5³=125 个 chunk，压缩后合计约 MB 级。
因此**不需要下载 42 GB 的 zip，也不需要 248 MB 的中央目录**——
直接 list + get 命中的对象即可。

## 依赖

`zstandard`（S3 上的 chunk 用 zstd 压缩；Python 标准库没有）。
在服务器上：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple zstandard`

## 用法

    python fetch_labels.py --dataset jrc_mus-kidney \
        --labels mito,er --out data/labels \
        [--crops 129,140] [--max-crops 4] [--resolution s0] [--list-only]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BUCKET = "https://janelia-cosem-datasets.s3.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
UA = "dachuang-organelle/1.0"


def http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def s3_ls(prefix: str, delimiter: str = "", max_keys: int = 1000
          ) -> Tuple[List[str], List[Tuple[str, int]]]:
    """列出 S3 前缀。返回 (子目录前缀, [(key, size)])。"""
    url = (f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}"
           f"&max-keys={max_keys}")
    if delimiter:
        url += f"&delimiter={urllib.parse.quote(delimiter)}"
    root = ET.fromstring(http_get(url).decode("utf-8", "replace"))
    dirs = [p.find(NS + "Prefix").text for p in root.findall(NS + "CommonPrefixes")]
    keys = [(c.find(NS + "Key").text, int(c.find(NS + "Size").text))
            for c in root.findall(NS + "Contents")]
    trunc = root.find(NS + "IsTruncated")
    if trunc is not None and trunc.text == "true":
        print(f"  [警告] 列表被截断（>{max_keys}）：{prefix}")
    return dirs, keys


def parse_zarray(obj: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(obj["raw"].decode("utf-8"))


def decompress(blob: bytes, comp: Optional[Dict[str, Any]]) -> bytes:
    if not comp:
        return blob
    cid = comp.get("id")
    if cid == "zstd":
        try:
            import zstandard as zstd
        except ImportError:
            raise RuntimeError(
                "需要 zstandard：pip install -i "
                "https://pypi.tuna.tsinghua.edu.cn/simple zstandard")
        return zstd.ZstdDecompressor().decompressobj().decompress(blob)
    if cid == "gzip":
        import gzip
        return gzip.decompress(blob)
    if cid in ("blosc", "lz4"):
        raise RuntimeError(f"不支持的压缩器 {cid}")
    raise RuntimeError(f"未知压缩器 {cid}")


def has_data_chunks(prefix: str) -> bool:
    """快速判断一个 Zarr 数组**是否真有数据块**。

    为什么需要：CellMap 里有些 crop 的某个类别目录**存在、`.zarray` 也存在，
    但一个数据块都没有**（实测 jrc_mus-kidney 的 crop149/mito/s0 与
    crop146/mito/s0 都是这种空数组）。只按"目录存在"筛选会把这些当成可用标注，
    下完得到一个全零体积——白等时间，还会污染数据集。

    这里只列前几个对象：除 `.zarray`/`.zgroup`/`.zattrs` 之外，
    只要还有以数字开头的 chunk 路径，就认为有数据。
    """
    try:
        _, keys = s3_ls(prefix, delimiter="", max_keys=8)
    except Exception:  # noqa: BLE001
        return False
    for k, _size in keys:
        base = k.rsplit("/", 1)[-1]
        if base.startswith("."):
            continue
        if base.isdigit():
            return True
    return False


def read_zarr_volume(base_prefix: str, resolution: str,
                     chunk_limit: Optional[int] = None,
                     workers: int = 12) -> Tuple[np.ndarray, Dict]:
    """把一个 Zarr v2 数组读成 numpy（按 chunk 拼接）。

    **并发是必须的**：实测单次 S3 GET 的往返延迟约 3–8 秒，而一个 300³、
    chunk 64³ 的体积有 5³=125 个 chunk——串行要 8 分钟以上（实测 135.8s→
    并发 12 路后约 2 分钟）。瓶颈是延迟而不是带宽。

    **"缺 chunk" 不是错误**：Zarr 对全零的 chunk 通常不落盘，读取时用
    ``fill_value`` 填充。所以 ``chunks_missing`` 很大只说明该体积稀疏
    （标签本身只占少数体素），不代表下载失败。

    只支持 3 维、uint8/uint16、zstd 压缩——CellMap 的标签正是如此。
    """
    import concurrent.futures as cf

    arr_prefix = f"{base_prefix.rstrip('/')}/{resolution}/"
    zarray_key = arr_prefix + ".zarray"
    blob = http_get(f"{BUCKET}/{urllib.parse.quote(zarray_key)}")
    z = json.loads(blob.decode("utf-8"))
    shape = tuple(int(x) for x in z["shape"])
    chunks = tuple(int(x) for x in z["chunks"])
    dtype = np.dtype(z.get("dtype", "|u1"))
    comp = z.get("compressor")
    fill = z.get("fill_value", 0)
    if len(shape) != 3:
        raise RuntimeError(f"只支持 3D，得到 shape={shape}")

    vol = np.full(shape, fill, dtype=dtype)
    ncz = (shape[0] + chunks[0] - 1) // chunks[0]
    ncy = (shape[1] + chunks[1] - 1) // chunks[1]
    ncx = (shape[2] + chunks[2] - 1) // chunks[2]
    coords = [(cz, cy, cx)
              for cz in range(ncz) for cy in range(ncy) for cx in range(ncx)]
    total = len(coords)
    if chunk_limit:
        coords = coords[:chunk_limit]

    def get_one(c):
        cz, cy, cx = c
        key = f"{arr_prefix}{cz}/{cy}/{cx}"
        try:
            raw = http_get(f"{BUCKET}/{urllib.parse.quote(key)}")
            return c, decompress(raw, comp), None
        except Exception as exc:  # noqa: BLE001
            return c, None, f"{type(exc).__name__}: {exc}"

    got = missing = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for c, buf, err in pool.map(get_one, coords):
            if buf is None:
                missing += 1
                continue
            cz, cy, cx = c
            a = np.frombuffer(buf, dtype=dtype)
            side = round(a.size ** (1 / 3))
            if a.size == chunks[0] * chunks[1] * chunks[2]:
                ch = chunks
            elif side and side ** 3 == a.size:
                ch = (side, side, side)
            else:
                missing += 1
                continue
            sl = (slice(cz * chunks[0], min((cz + 1) * chunks[0], shape[0])),
                  slice(cy * chunks[1], min((cy + 1) * chunks[1], shape[1])),
                  slice(cx * chunks[2], min((cx + 1) * chunks[2], shape[2])))
            sub = a.reshape(ch)[: sl[0].stop - sl[0].start,
                                : sl[1].stop - sl[1].start,
                                : sl[2].stop - sl[2].start]
            vol[sl] = sub
            got += 1
    dt = time.time() - t0
    meta = {"shape": list(shape), "chunks": list(chunks), "dtype": str(dtype),
            "compressor": comp, "chunks_fetched": got, "chunks_missing": missing,
            "total_chunks": total, "workers": workers,
            "seconds": round(dt, 1)}
    return vol, meta


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="从 S3 拉 CellMap 标签掩码")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labels", default="mito,er")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", default="s0")
    ap.add_argument("--crops", default=None, help="逗号分隔；缺省取前若干个")
    ap.add_argument("--max-crops", type=int, default=4)
    ap.add_argument("--chunk-limit", type=int, default=0,
                    help="每个体积最多取多少 chunk（调试用，0=全部）")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--eligible-cache", default=None,
                    help="可用 crop 列表缓存路径（避免每次重扫 33 个 crop）")
    ap.add_argument("--workers", type=int, default=12,
                    help="并发下载 chunk 的线程数（单请求延迟高，必须并发）")
    ap.add_argument("--skip-existing", action="store_true",
                    help="已存在且非空的 .npy 直接跳过（扩数据时避免重下）")
    args = ap.parse_args(argv)

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    root = f"{args.dataset}/{args.dataset}.zarr/recon-1/labels/groundtruth/"
    dirs, _ = s3_ls(root, delimiter="/")
    crops = sorted({d.rstrip("/").split("/")[-1] for d in dirs
                    if d.rstrip("/").split("/")[-1].startswith("crop")},
                   key=lambda s: int(s[4:]) if s[4:].isdigit() else 0)
    print(f"{args.dataset}: groundtruth 下 {len(crops)} 个 crop")

    # CellMap 的 crop 之间**类别并不一致**：例如 crop1026 有声母/叶绿体相关类别，
    # 却没有 mito 与 er。按"字母序取前 N 个"会全部 404。这里逐个列出 crop 的子目录，
    # 只保留**真正同时含有所需类别**的 crop。
    #
    # 扫描 33 个 crop 要 100–260 秒（每个 list 一次，链路慢），所以结果要缓存，
    # 否则每次重跑都白等几分钟。
    cache = args.eligible_cache
    eligible: List[str] = []
    if cache and os.path.exists(cache):
        try:
            blob = json.load(open(cache, encoding="utf-8"))
            if blob.get("dataset") == args.dataset and \
                    blob.get("labels") == labels:
                eligible = blob["eligible"]
                print(f"  复用可用 crop 缓存：{cache}"
                      f"（{len(eligible)} 个）")
        except Exception:  # noqa: BLE001
            eligible = []

    if not eligible:
        t_scan = time.time()
        for c in crops:
            subs, _ = s3_ls(f"{root}{c}/", delimiter="/")
            have = {s.rstrip("/").split("/")[-1] for s in subs}
            if all(l in have for l in labels):
                eligible.append(c)
        print(f"  同时含 {labels} 的 crop：{len(eligible)} 个"
              f"（扫描 {time.time()-t_scan:.0f}s）")
        if cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
            with open(cache, "w", encoding="utf-8") as fh:
                json.dump({"dataset": args.dataset, "labels": labels,
                           "eligible": eligible,
                           "scanned": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                          fh, ensure_ascii=False, indent=1)
            print(f"  已缓存 -> {cache}")
    if not eligible:
        print(f"[错误] {args.dataset} 里没有 crop 同时含 {labels}", file=sys.stderr)
        return 2

    if args.crops:
        want = [f"crop{s.strip()}" if not s.strip().startswith("crop") else s.strip()
                for s in args.crops.split(",")]
        crops = [c for c in eligible if c in want]
        if not crops:
            print(f"[错误] 指定的 crop 不在可用列表里。可用：{eligible[:10]}",
                  file=sys.stderr)
            return 2
    else:
        crops = eligible[: args.max_crops]
    print(f"将处理 {len(crops)} 个 crop：{crops}")

    if args.list_only:
        return 0

    os.makedirs(args.out, exist_ok=True)
    written: List[Dict[str, Any]] = []
    empty_pairs: List[Dict[str, Any]] = []
    t0 = time.time()
    for c in crops:
        for lab in labels:
            dest = os.path.join(args.out, args.dataset, c, f"{lab}.npy")
            if args.skip_existing and os.path.exists(dest) and \
                    os.path.getsize(dest) > 0:
                print(f"  {c:<10} {lab:<5} 已存在，跳过", flush=True)
                continue
            pref = f"{root}{c}/{lab}/"
            arr_pref = f"{pref}{args.resolution}/"
            # 先花一次 list 请求判断有没有数据块，避免把空数组整块下下来
            if not has_data_chunks(arr_pref):
                print(f"  {c:<10} {lab:<5} 空数组（无数据块），跳过",
                      flush=True)
                empty_pairs.append({"crop": c, "label": lab,
                                    "reason": "zarr 数组存在但无数据块"})
                continue
            try:
                vol, meta = read_zarr_volume(pref, args.resolution,
                                             args.chunk_limit or None,
                                             workers=args.workers)
            except Exception as exc:  # noqa: BLE001
                print(f"  [跳过] {c}/{lab}: {type(exc).__name__}: {exc}")
                continue
            nz = int(np.count_nonzero(vol))
            if nz == 0:
                # 双保险：即使有 chunk 也全零（理论上不该发生），不落盘
                print(f"  {c:<10} {lab:<5} 全零，跳过不保存", flush=True)
                empty_pairs.append({"crop": c, "label": lab,
                                    "reason": "读到但全为零"})
                continue
            print(f"  {c:<10} {lab:<5} shape={vol.shape} 非零={nz:,} "
                  f"({100.0*nz/vol.size:.2f}%) chunk {meta['chunks_fetched']}/"
                  f"{meta['total_chunks']} 缺 {meta['chunks_missing']} "
                  f"{meta['seconds']}s")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            np.save(dest, vol)
            written.append({"dataset": args.dataset, "crop": c, "label": lab,
                            "path": dest, "shape": list(vol.shape),
                            "nonzero": nz, **meta})
    with open(os.path.join(args.out, "fetch_labels_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"dataset": args.dataset, "labels": labels,
                   "resolution": args.resolution,
                   "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "empty_pairs": empty_pairs,
                   "volumes": written}, fh, ensure_ascii=False, indent=1)
    print(f"\n完成 {len(written)} 个体积，耗时 {time.time()-t0:.0f}s")
    print(f"输出：{os.path.abspath(args.out)}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
