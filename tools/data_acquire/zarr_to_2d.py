#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zarr_to_2d.py —— 把 CellMap 的 3D Zarr 标注导出成 2D 切片（EM + 细胞器掩码）

## 为什么需要这一步

CellMap 是**体数据**（每 crop 400³ 体素），而本项目的 DETR 管线是 **2D**。
中间必须做一次 3D → 2D 的降维。这是一个**方法学决策**，不是无所谓的实现细节：

- 沿 z 取单层切片：保留真实像素统计，但标注在某些层可能很稀疏；
- 最大密度投影（MIP）：目标更连续，但把 3D 结构压扁，轴长/面积类形态特征会被高估。

所以本脚本**把策略写进输出清单**（`export_manifest.json`），
让"你的 2D 图是怎么来的"在论文里可追溯。

## 对齐方式（这是最容易出错的地方）

官方 `train_crop_manifest.csv` 给出每个 crop 的：

    translation = [z;y;x] 物理偏移（nm）
    shape       = [z;y;x] 体素数
    voxel_size  = [z;y;x] nm/体素

于是 crop 在 EM 全卷中的**起始体素索引**为 ``translation / voxel_size``。
标签数组本身已经是裁剪后的体积，因此：

    label[crop][k]  <->  EM[ z0 + k ]

只需从 EM 全卷按 ``z0`` 偏移取对应层，就能与标签空间对齐。

## 用法

    python tools/data_acquire/zarr_to_2d.py \\
        --root  E:\\dachuang_public_data\\cellmap_extracted\\data_matched_res_no_pad \\
        --dataset jrc_cos7-1a \\
        --crops 234,236 \\
        --labels mito,er \\
        --out E:\\dachuang_public_data\\cellmap_2d \\
        --strategy slice --z-step 25 --max-per-crop 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Windows 上并发读取 Zarr 块可能让进程硬崩（官方 cellmap-data 文档明确提到）。
# 本脚本本来就单线程顺序读，这里只是把并发压到 1 以防底层库自作聪明。
os.environ.setdefault("CELLMAP_MAX_CONCURRENT_READS", "1")


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def parse_vec(s: str) -> Tuple[float, float, float]:
    """解析 '[a;b;c]' -> (a, b, c)。"""
    return tuple(float(x) for x in s.strip().strip("[]").split(";"))


def load_crop_manifest(path: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """读 train_crop_manifest.csv，返回 {(dataset, crop): {voxel, translation, shape, labels}}。"""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["dataset"], str(row["crop_name"]))
            rec = out.setdefault(key, {
                "voxel": parse_vec(row["voxel_size"]),
                "translation": parse_vec(row["translation"]),
                "shape": tuple(int(float(x)) for x in
                               row["shape"].strip().strip("[]").split(";")),
                "labels": set(),
            })
            rec["labels"].add(row["class_label"])
    for rec in out.values():
        rec["labels"] = sorted(rec["labels"])
    return out


def crop_origin_voxels(rec: Dict[str, Any]) -> Tuple[int, int, int]:
    """crop 在 EM 全卷中的起始体素索引 = translation / voxel_size。"""
    tr = np.asarray(rec["translation"], dtype=np.float64)
    vs = np.asarray(rec["voxel"], dtype=np.float64)
    return tuple(int(round(v)) for v in (tr / vs))


# --------------------------------------------------------------------------
# Zarr 读取
# --------------------------------------------------------------------------

def open_array(path: str):
    """打开一个 Zarr 数组（兼容 zarr v2 目录）。"""
    import zarr
    return zarr.open_array(path, mode="r")


def find_volume_dir(base: str, kind: str, crop: Optional[str] = None,
                    label: Optional[str] = None) -> Optional[str]:
    """定位 em 或 label 的体积目录（含多分辨率子目录 s0/s1…）。"""
    if kind == "em":
        cand = os.path.join(base, "recon-1", "em", "fibsem-uint8", "s0")
        return cand if os.path.isdir(cand) else None
    if kind == "label":
        cand = os.path.join(base, "recon-1", "labels", "groundtruth",
                            str(crop), str(label), "s0")
        return cand if os.path.isdir(cand) else None
    return None


def read_slab(arr, strategy: str, z0: int, depth: int) -> np.ndarray:
    """取一段 z（strategy=slice 取单层；mip 取 max projection）。"""
    z1 = min(z0 + depth, arr.shape[0])
    if z1 <= z0:
        return np.zeros(arr.shape[1:], dtype=arr.dtype)
    slab = np.asarray(arr[z0:z1])
    if strategy == "mip":
        return slab.max(axis=0)
    return slab[0]


# --------------------------------------------------------------------------
# 保存
# --------------------------------------------------------------------------

def save_png(path: str, img: np.ndarray) -> None:
    from PIL import Image
    a = np.asarray(img)
    if a.ndim != 2:
        a = a.squeeze()
    if a.dtype != np.uint8:
        lo, hi = float(a.min()), float(a.max())
        if hi > lo:
            a = (a.astype(np.float64) - lo) / (hi - lo) * 255.0
        else:
            a = a.astype(np.float64) * 0.0
        a = a.astype(np.uint8)
    Image.fromarray(a).save(path)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="把 CellMap 3D Zarr 导出为 2D EM + 细胞器掩码切片")
    ap.add_argument("--root", required=True,
                    help="解压后的根，如 ...\\data_matched_res_no_pad")
    ap.add_argument("--manifest", default=os.path.join(
        here, "cellmap_refs", "train_crop_manifest.csv"))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--crops", default=None,
                    help="crop 列表（逗号分隔）；缺省=该数据集全部含所需类别的 crop")
    ap.add_argument("--labels", default="mito,er")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strategy", choices=["slice", "mip"], default="slice")
    ap.add_argument("--z-step", type=int, default=25,
                    help="每隔多少层取一帧")
    ap.add_argument("--mip-depth", type=int, default=8,
                    help="mip 时每次投影的层数")
    ap.add_argument("--max-per-crop", type=int, default=8)
    ap.add_argument("--start-frac", type=float, default=0.15,
                    help="从 z 的这个比例处开始取（避开 crop 边缘空白）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    crop_meta = load_crop_manifest(args.manifest)

    # 实际布局：<root>/<dataset>/<dataset>.zarr/recon-1/...
    # 兼容两种：带 .zarr 子目录的（真实数据），以及直接把 recon-1 放在 dataset 下的（测试）
    ds_dir = os.path.join(args.root, args.dataset, f"{args.dataset}.zarr")
    if not os.path.isdir(ds_dir):
        alt = os.path.join(args.root, args.dataset)
        if os.path.isdir(os.path.join(alt, "recon-1")):
            ds_dir = alt
        else:
            print(f"[错误] 找不到数据集目录：{ds_dir}", file=sys.stderr)
            print(f"       也试过：{alt}", file=sys.stderr)
            return 2

    # 该数据集里同时含所需类别的 crop
    avail = {c: rec for (d, c), rec in crop_meta.items()
             if d == args.dataset and all(l in rec["labels"] for l in labels)}
    if args.crops:
        wanted = [s.strip() for s in args.crops.split(",")]
        avail = {c: r for c, r in avail.items() if c in wanted}
    if not avail:
        print(f"[错误] {args.dataset} 中没有同时含 {labels} 的 crop",
              file=sys.stderr)
        return 2

    print(f"数据集：{args.dataset}")
    print(f"crop 数：{len(avail)}（同时含 {labels}）")
    print(f"策略：{args.strategy}"
          + (f"，z-step={args.z_step}" if args.strategy == "slice"
             else f"，mip-depth={args.mip_depth}"))
    print(f"输出：{os.path.abspath(args.out)}")

    # EM 全卷
    em_dir = find_volume_dir(ds_dir, "em")
    if em_dir is None:
        print(f"[错误] 找不到 EM 卷：{ds_dir}\\recon-1\\em\\fibsem-uint8\\s0",
              file=sys.stderr)
        return 2
    em = open_array(em_dir)
    print(f"EM 卷形状：{em.shape}  dtype={em.dtype}  chunks={getattr(em, 'chunks', '?')}")

    manifest: Dict[str, Any] = {
        "generator": "tools/data_acquire/zarr_to_2d.py",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": os.path.abspath(args.root),
        "dataset": args.dataset,
        "labels": labels,
        "strategy": args.strategy,
        "z_step": args.z_step,
        "mip_depth": args.mip_depth,
        "start_frac": args.start_frac,
        "em_shape": list(em.shape),
        "em_dtype": str(em.dtype),
        "coordinate_note": (
            "crop 原点体素索引 = translation / voxel_size（来自官方 "
            "train_crop_manifest.csv）。标签数组本身已裁剪，故 label[k] <-> EM[z0+k]。"
        ),
        "crops": {},
    }

    t0 = time.time()
    total_frames = 0
    for crop, rec in sorted(avail.items(), key=lambda kv: int(kv[0])):
        z0, y0, x0 = crop_origin_voxels(rec)
        cz, cy, cx = rec["shape"]
        print(f"\n--- crop {crop}: origin(z,y,x)=({z0},{y0},{x0}) "
              f"shape=({cz},{cy},{cx}) vs={rec['voxel']}")

        # 标签数组
        lab_arrays: Dict[str, Any] = {}
        for lab in labels:
            d = find_volume_dir(ds_dir, "label", crop=crop, label=lab)
            if d is None:
                print(f"    [!] 缺少 {lab} 的 s0 目录，跳过该类别")
                continue
            lab_arrays[lab] = open_array(d)
            print(f"    {lab}: shape={lab_arrays[lab].shape}")

        if not lab_arrays:
            print("    [!] 没有任何标签数组，跳过该 crop")
            continue

        ref = next(iter(lab_arrays.values()))
        nz = int(min(ref.shape[0], cz))
        # 起始层：避开边缘空白
        start = int(nz * args.start_frac)
        depth = args.mip_depth if args.strategy == "mip" else 1
        zs = list(range(start, nz - depth, max(1, args.z_step)))[: args.max_per_crop]
        if not zs:
            zs = [start]

        crop_rec: Dict[str, Any] = {
            "origin_voxels": [z0, y0, x0],
            "shape": list(rec["shape"]),
            "voxel_size_nm": list(rec["voxel"]),
            "translation_nm": list(rec["translation"]),
            "labels": list(lab_arrays.keys()),
            "frames": [],
        }

        for z in zs:
            outdir = os.path.join(args.out, args.dataset, f"crop{crop}",
                                  f"z{z:04d}")
            if not args.dry_run:
                os.makedirs(outdir, exist_ok=True)

            # EM：从全卷按 crop 原点偏移取
            em_z = z0 + z
            if em_z >= em.shape[0]:
                print(f"    [!] EM 缺少 z={em_z}，跳过")
                continue
            em_slab = read_slab(em, args.strategy, em_z, depth)
            # 裁到 crop 的 y/x 范围
            yy0, yy1 = y0, min(y0 + cy, em_slab.shape[0])
            xx0, xx1 = x0, min(x0 + cx, em_slab.shape[1])
            if yy1 <= yy0 or xx1 <= xx0:
                print(f"    [!] crop 的 y/x 超出 EM 边界，跳过")
                continue
            em_crop = em_slab[yy0:yy1, xx0:xx1]

            frame: Dict[str, Any] = {
                "z": z, "em_z": em_z,
                "em_shape": list(em_crop.shape),
                "label_stats": {},
            }
            if not args.dry_run:
                save_png(os.path.join(outdir, "em.png"), em_crop)
            for lab, arr in lab_arrays.items():
                ls = read_slab(arr, args.strategy, z, depth)
                # y/x 也要裁到与 EM 一致（标签比 EM 小时按最小范围）
                h = min(ls.shape[0], em_crop.shape[0])
                w = min(ls.shape[1], em_crop.shape[1])
                ls = ls[:h, :w]
                frame["label_stats"][lab] = {
                    "shape": list(ls.shape),
                    "nonzero": int(np.count_nonzero(ls)),
                    "max": float(ls.max()) if ls.size else 0.0,
                }
                if not args.dry_run:
                    save_png(os.path.join(outdir, f"{lab}.png"), ls)
                total_frames += 1

            crop_rec["frames"].append(frame)
            print(f"    z={z:4d} -> em_z={em_z:4d}  "
                  + "  ".join(f"{l}:{frame['label_stats'][l]['nonzero']}px"
                              for l in frame["label_stats"]))

        manifest["crops"][crop] = crop_rec

    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "export_manifest.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=1)

    print(f"\n完成：{total_frames} 个（帧×类别），耗时 {time.time() - t0:.1f}s")
    if not args.dry_run:
        print(f"清单：{os.path.join(args.out, 'export_manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
