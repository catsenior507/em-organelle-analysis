#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_em_slices.py —— 从 S3 取 EM 灰度平面，与已有标签对齐，输出 2D 切片

## 为什么需要单独这一步

接触分析只需要标签掩码（已完成）。但训练 DETR 需要 **(EM 图像, 目标框)** 配对，
而 EM 在 S3 上是**整卷一个数组**：

    recon-1/em/fibsem-uint8/s0/     shape=[22199, 7976, 12287]  chunks=[64,128,128]  8 nm

所以只取每个 crop 覆盖的那一小块，且只取选定的 z 平面（不必取整个 300³ 体积）。

## 坐标映射（关键，且已验证）

官方 manifest 给出每个 crop 的 `translation`（**纳米**）、`shape`（体素）、
`voxel_size`（纳米/体素）。EM s0 是 8 nm/体素，因此：

    EM 原点(体素)  = translation / 8
    EM 范围(体素)  = shape * voxel_size / 8

实测 jrc_mus-kidney 的 17 个 mito+er crop 全部落在 EM 范围内。

**分辨率不匹配**：17 个 crop 里 16 个是 4 nm 标签，比 8 nm 的 EM 细一倍。
因此标签在进入训练前要**按 2× 降采样**才能与 EM 逐像素对齐。
本脚本用"块内取最大值"（任一前景即为前景），对细结构（如 ER 管网）比取均值安全。

## 输出布局（与 tools/detr/slice_dataset.py 直接对接）

    <out>/<dataset>/<crop>/z<NNNN>/
        em.png      8 位灰度，原始 EM 值不缩放
        mito.png    0/1 掩码
        er.png      0/1 掩码

## 用法

    python fetch_em_slices.py --manifest train_crop_manifest.csv \
        --labels data/labels --dataset jrc_mus-kidney \
        --crops 129,156 --z-stride 20 --max-slices 8 --out data/slices
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

BUCKET = "https://janelia-cosem-datasets.s3.amazonaws.com"
UA = "dachuang-organelle/1.0"
EM_SCALE_NM = 8.0          # 默认值；实际会从 .zattrs 的 multiscales 读取


def read_em_scale(prefix: str, level: str = "s0") -> float:
    """从 EM 的 .zattrs 读出该分辨率层级的体素尺寸（nm）。

    **不能硬编码 8 nm**：jrc_mus-kidney 的 s0 是 8 nm，但别的数据集可能是 4 nm。
    体素尺寸一旦算错，crop 的物理范围映射到 EM 就会整体错位，
    取出来的切片与标签完全不对应（而图像本身看起来仍然"正常"，很难发现）。
    因此改为读 OME-NGFF 的 multiscales 声明：

        multiscales[0].datasets[i].coordinateTransformations[0].scale = [z, y, x]
    """
    arr_base = prefix.rsplit("/", 2)[0] + "/"      # em/<array>/
    try:
        blob = http_get(f"{BUCKET}/{urllib.parse.quote(arr_base + '.zattrs')}")
        attrs = json.loads(blob.decode("utf-8"))
        ms = attrs.get("multiscales") or []
        if ms:
            for ds in ms[0].get("datasets", []):
                if ds.get("path") == level:
                    for t in ds.get("coordinateTransformations", []):
                        if t.get("type") == "scale":
                            return float(t["scale"][0])
    except Exception:  # noqa: BLE001
        pass
    return EM_SCALE_NM


def http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def read_zarray(prefix: str) -> Dict[str, Any]:
    return json.loads(http_get(f"{BUCKET}/{urllib.parse.quote(prefix + '.zarray')}")
                      .decode("utf-8"))


def decompress(blob: bytes, comp: Optional[Dict[str, Any]]) -> bytes:
    if not comp:
        return blob
    cid = comp.get("id")
    if cid == "zstd":
        if zstd is None:
            raise RuntimeError("需要 zstandard")
        return zstd.ZstdDecompressor().decompressobj().decompress(blob)
    if cid == "gzip":
        import gzip
        return gzip.decompress(blob)
    raise RuntimeError(f"不支持的压缩器 {cid}")


def chunk_coords(zarr: Dict[str, Any], zs: Sequence[int],
                 y0: int, x0: int, h: int, w: int) -> set:
    """一个 crop 的若干 z 平面覆盖到的全部 chunk 坐标（去重）。

    **为什么先去重再取**：EM 的 z chunk 厚 64，而 z-stride 常常是 20，
    于是同一个 (cz,cy,cx) chunk 会被多个 z 平面重复需要。串行逐平面取会
    把同一块下载好几遍——实测这正是慢的主因。
    """
    chunks = [int(c) for c in zarr["chunks"]]
    coords = set()
    czs = {int(z) // chunks[0] for z in zs}
    cy0, cy1 = y0 // chunks[1], (y0 + h - 1) // chunks[1]
    cx0, cx1 = x0 // chunks[2], (x0 + w - 1) // chunks[2]
    for cz in czs:
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                coords.add((cz, cy, cx))
    return coords


def fetch_chunks(prefix: str, zarr: Dict[str, Any], coords: set,
                 workers: int = 12, timeout: int = 120) -> Dict[Tuple[int, int, int], np.ndarray]:
    """并发取回一批 chunk，返回 {(cz,cy,cx): 3D array}。

    与 fetch_labels.py 同样的理由：单次 S3 GET 的往返延迟 3–8 秒，
    串行会让"取几十个 chunk"变成十几分钟。
    """
    import concurrent.futures as cf

    chunks = [int(c) for c in zarr["chunks"]]
    dtype = np.dtype(zarr.get("dtype", "|u1"))
    comp = zarr.get("compressor")

    def one(c):
        cz, cy, cx = c
        key = f"{prefix}{cz}/{cy}/{cx}"
        try:
            raw = http_get(f"{BUCKET}/{urllib.parse.quote(key)}", timeout)
        except Exception:  # noqa: BLE001
            return c, None
        try:
            a = np.frombuffer(decompress(raw, comp), dtype=dtype)
        except Exception:  # noqa: BLE001
            return c, None
        if a.size != chunks[0] * chunks[1] * chunks[2]:
            return c, None
        return c, a.reshape(chunks)

    out: Dict[Tuple[int, int, int], np.ndarray] = {}
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for c, arr in pool.map(one, sorted(coords)):
            if arr is not None:
                out[c] = arr
    return out


def plane_from_cache(cache: Dict[Tuple[int, int, int], np.ndarray],
                     zarr: Dict[str, Any], z: int,
                     y0: int, x0: int, h: int, w: int,
                     dtype) -> np.ndarray:
    """从已取回的 chunk 缓存里拼出一个 z 平面。缺的 chunk 视为背景。"""
    chunks = [int(c) for c in zarr["chunks"]]
    cz = z // chunks[0]
    out = np.zeros((h, w), dtype=dtype)
    cy0, cy1 = y0 // chunks[1], (y0 + h - 1) // chunks[1]
    cx0, cx1 = x0 // chunks[2], (x0 + w - 1) // chunks[2]
    for cy in range(cy0, cy1 + 1):
        for cx in range(cx0, cx1 + 1):
            vol = cache.get((cz, cy, cx))
            if vol is None:
                continue
            plane = vol[z - cz * chunks[0]]
            gy0, gx0 = cy * chunks[1], cx * chunks[2]
            iy0, iy1 = max(y0, gy0), min(y0 + h, gy0 + chunks[1])
            ix0, ix1 = max(x0, gx0), min(x0 + w, gx0 + chunks[2])
            if iy1 <= iy0 or ix1 <= ix0:
                continue
            out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = \
                plane[iy0 - gy0:iy1 - gy0, ix0 - gx0:ix1 - gx0]
    return out


def select_z_by_content(labs: Dict[str, np.ndarray], factor: int,
                        z_max: int, max_slices: int, min_sep: int,
                        min_content_frac: float = 0.0
                        ) -> List[Tuple[int, float]]:
    """按**标注内容量**挑选 z 层，而不是按固定步长。

    ## 为什么必须这样做

    早期版本用 `range(0, z_extent, z_stride)` 固定步长取样，实测踩坑：
    crop160 取到的 17 层里线粒体覆盖率是 **0.00%**，crop163/166 只有 0.3%。
    原因不是标注有问题，而是**取样的层正好落在没有该细胞器的区域**——
    这些 crop 当初是按别的细胞器选的。

    把这类切片喂给检测器，等于告诉模型"这些图里没有线粒体"，
    但它们只是**没取到**线粒体所在的层。

    ## 做法

    对每个 z（换算到 EM 体素），统计各类别在该层的前景体素总数，取和；
    然后按内容从多到少贪心挑选，且与已选层保持 `min_sep` 的间隔
    （避免相邻层几乎重复，同 crop 的近邻会造成训练/验证泄漏）。

    返回 [(z_local, content_fraction), ...]，按 z 升序。
    """
    nz = np.zeros(z_max, dtype=np.float64)
    for vol in labs.values():
        dv = downsample_max(vol, factor)
        m = min(z_max, dv.shape[0])
        nz[:m] += (dv[:m] > 0).sum(axis=(1, 2))
    total_vox = None
    for vol in labs.values():
        dv = downsample_max(vol, factor)
        total_vox = dv.shape[1] * dv.shape[2] if total_vox is None else total_vox
        break
    frac = nz / float(max(1, total_vox or 1))

    order = np.argsort(-nz)
    picked: List[int] = []
    for z in order:
        z = int(z)
        if nz[z] <= 0:
            break
        if frac[z] < min_content_frac:
            break
        if all(abs(z - p) >= min_sep for p in picked):
            picked.append(z)
        if len(picked) >= max_slices:
            break
    if not picked:
        # 全都没有内容时，退化为均匀取样（便于诊断）
        picked = list(range(0, z_max, max(1, min_sep)))[:max_slices]
    picked.sort()
    return [(z, round(float(frac[z]), 5)) for z in picked]


def downsample_max(vol: np.ndarray, factor: int) -> np.ndarray:
    """整块取最大值降采样（任一前景即为前景）。

    对掩码用 max 而不是 mean：ER 是细管网，取均值会把很多像素压到 0 以下，
    等于系统性地丢掉细结构。max 会略微"加粗"，但对检测任务更安全。
    """
    if factor <= 1:
        return vol
    z, y, x = vol.shape
    z2, y2, x2 = z // factor, y // factor, x // factor
    v = vol[: z2 * factor, : y2 * factor, : x2 * factor]
    v = v.reshape(z2, factor, y2, factor, x2, factor)
    return v.max(axis=(1, 3, 5))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="取 EM 平面并与标签对齐导出 2D 切片")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True, help="fetch_labels.py 的输出目录")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--crops", default=None, help="逗号分隔；缺省取标签目录里全部")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", default="s0")
    ap.add_argument("--z-stride", type=int, default=20)
    ap.add_argument("--select-by-content", action="store_true",
                    help="按标注内容量选层而不是固定步长（推荐）")
    ap.add_argument("--min-sep", type=int, default=6,
                    help="选中层之间的最小间隔（体素），避免相邻层近乎重复")
    ap.add_argument("--min-content-frac", type=float, default=0.0,
                    help="低于该前景比例的层不选（0=只要非空）")
    ap.add_argument("--max-slices", type=int, default=8, help="每个 crop 最多导出几层")
    ap.add_argument("--workers", type=int, default=12,
                    help="并发下载 chunk 的线程数（单请求延迟 3–8 秒，必须并发）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--classes", default="mito,er",
                    help="需要哪些类别；给 --require-all-labels 时必须全部有标注")
    ap.add_argument("--no-require-all-labels", dest="require_all_labels",
                    action="store_false",
                    help="允许缺少部分类别的 crop（默认不允许）")
    ap.set_defaults(require_all_labels=True)
    args = ap.parse_args(argv)

    if Image is None:
        print("[错误] 需要 pillow", file=sys.stderr)
        return 2
    if zstd is None:
        print("[错误] 需要 zstandard", file=sys.stderr)
        return 2

    def vec(s):
        return [float(x) for x in s.strip().strip("[]").split(";")]

    # --- manifest：每个 crop 的 translation / shape / voxel ---
    info: Dict[str, Dict[str, Any]] = {}
    with open(args.manifest, "r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["dataset"] != args.dataset:
                continue
            c = str(r["crop_name"])
            info.setdefault(c, {"shape": vec(r["shape"]),
                                "voxel": vec(r["voxel_size"]),
                                "translation": vec(r["translation"])})

    # --- 标签目录里实际存在的 crop ---
    lab_dir = os.path.join(args.labels, args.dataset)
    have = sorted(os.listdir(lab_dir)) if os.path.isdir(lab_dir) else []
    crops = [c for c in have if c.startswith("crop") and c[4:] in info]
    if args.crops:
        want = {"crop" + s.strip().lstrip("crop") for s in args.crops.split(",")}
        crops = [c for c in crops if c in want]
    if not crops:
        print(f"[错误] 没有可处理的 crop。标签目录：{lab_dir}", file=sys.stderr)
        return 2
    print(f"{args.dataset}: 将处理 {len(crops)} 个 crop：{crops}")

    em_prefix = (f"{args.dataset}/{args.dataset}.zarr/recon-1/em/"
                 f"fibsem-uint8/{args.resolution}/")
    zarr = read_zarray(em_prefix)
    em_scale = read_em_scale(em_prefix, args.resolution)
    print(f"EM {args.resolution}: shape={zarr['shape']} chunks={zarr['chunks']} "
          f"dtype={zarr['dtype']} scale={em_scale}nm"
          f"（{'来自 .zattrs' if abs(em_scale - EM_SCALE_NM) > 1e-9 or em_scale != 8.0 else '默认值'}）")

    written: List[Dict[str, Any]] = []
    t0 = time.time()
    for c in crops:
        d = info[c[4:]]
        o = [int(round(t / em_scale)) for t in d["translation"]]
        ext = [int(round(s * v / em_scale))
               for s, v in zip(d["shape"], d["voxel"])]
        # 降采样系数：标签比 EM 细多少倍就要缩多少倍。
        # 早期版本写成 voxel/EM（4/8 被 round 成 0，再被 max(1,·) 兜成 1），
        # 于是 4 nm 标签根本没降采样，和 8 nm EM 差了 2 倍尺寸。
        factor = max(1, int(round(em_scale / d["voxel"][0])))
        print(f"\n--- {c}  crop_voxel={d['voxel'][0]:g}nm  EM原点={o}  "
              f"EM范围={ext}  标签降采样 x{factor}")

        labs = {}
        for lab in args.classes.split(","):
            lab = lab.strip()
            p = os.path.join(lab_dir, c, f"{lab}.npy")
            if os.path.exists(p):
                a = np.load(p, mmap_mode="r")
                if int(np.count_nonzero(a)) > 0:
                    labs[lab] = np.load(p)
                else:
                    print(f"    [注意] {lab} 全零 -> 视为未标注")
        if not labs:
            print("    [跳过] 没有非空标签")
            continue
        # 缺某个类别就整块跳过：CellMap 里"某类数组为空"通常是**没有标注**，
        # 而不是"该区域确实没有这个细胞器"。把它当负样本会教模型漏检。
        want = [s.strip() for s in args.classes.split(",") if s.strip()]
        if args.require_all_labels and len(labs) < len(want):
            missing = [w for w in want if w not in labs]
            print(f"    [跳过] 缺少类别 {missing}（疑似未标注，不作负样本）")
            continue

        if args.select_by_content:
            # 按标注内容选层（推荐）：固定步长会取到没有标注的层
            picked = select_z_by_content(
                labs, factor, ext[0], args.max_slices,
                min_sep=max(1, args.min_sep),
                min_content_frac=args.min_content_frac)
            zs = [z for z, _f in picked]
            print(f"    按内容选层：{len(zs)} 层，覆盖率 "
                  f"{[f'{f:.4f}' for _z, f in picked]}")
        else:
            zs = list(range(0, ext[0], max(1, args.z_stride)))[: args.max_slices]
        # 先把这些 z 需要的 chunk 一次性并发取回（去重），再逐层拼平面
        coords = chunk_coords(zarr, [o[0] + z for z in zs],
                              o[1], o[2], ext[1], ext[2])
        print(f"    需要 {len(coords)} 个 EM chunk，并发 {args.workers} 路获取...")
        t1 = time.time()
        cache = fetch_chunks(em_prefix, zarr, coords, workers=args.workers)
        print(f"    取回 {len(cache)}/{len(coords)} 个 chunk，"
              f"{time.time()-t1:.0f}s")

        dtype = np.dtype(zarr.get("dtype", "|u1"))
        for z_local in zs:
            z_em = o[0] + z_local
            plane = plane_from_cache(cache, zarr, z_em,
                                     o[1], o[2], ext[1], ext[2], dtype)
            nz = int(np.count_nonzero(plane))
            if nz == 0:
                print(f"    z={z_local:>4} (EM z={z_em}) 空白，跳过")
                continue
            zl = z_local // factor
            lab_planes = {}
            for lab, vol in labs.items():
                dv = downsample_max(vol, factor)
                if zl >= dv.shape[0]:
                    continue
                lab_planes[lab] = dv[zl]

            if args.dry_run:
                print(f"    z={z_local:>4} (EM z={z_em}) EM 非零 {nz:,}/{plane.size:,} "
                      f"= {100.0*nz/plane.size:.1f}%  "
                      f"标签 {[(k, int((v>0).sum())) for k, v in lab_planes.items()]}")
                continue

            zdir = os.path.join(args.out, args.dataset, c, f"z{z_local:04d}")
            os.makedirs(zdir, exist_ok=True)
            Image.fromarray(plane.astype(np.uint8)).save(os.path.join(zdir, "em.png"))
            for lab, m in lab_planes.items():
                Image.fromarray(((m > 0).astype(np.uint8))).save(
                    os.path.join(zdir, f"{lab}.png"))
            written.append({"crop": c, "z_local": z_local, "z_em": z_em,
                            "dir": zdir, "em_nonzero": nz,
                            "labels": {k: int((v > 0).sum()) for k, v in lab_planes.items()},
                            "downsample": factor})
            print(f"    z={z_local:>4} (EM z={z_em}) 已写  EM非零 {nz:,}/{plane.size:,}  "
                  f"标签 {[(k, int((v>0).sum())) for k, v in lab_planes.items()]}")

    print(f"\n完成：{len(written)} 个切片，耗时 {time.time()-t0:.0f}s")
    if written and not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "slice_manifest.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"dataset": args.dataset, "em_resolution": args.resolution,
                       "em_scale_nm": em_scale,
                       "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "note": ("4nm 标签已按 2x 降采样到 8nm EM 网格；"
                                "em.png 保留原始灰度值，标签为 0/1 掩码"),
                       "slices": written}, fh, ensure_ascii=False, indent=1)
        print(f"清单：{os.path.join(args.out, 'slice_manifest.json')}")
    return 0 if (written or args.dry_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
