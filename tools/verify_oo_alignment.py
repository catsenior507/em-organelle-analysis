#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_oo_alignment.py —— 验证 crop 坐标与 EM 的对齐

## 这一步为什么关键

`oo_locate.py` 从 neuroglancer 的 viewer state 推出了每个 crop 在 EM 体中的
体素范围，但那只是**推断**：position 是视点中心、单位是纳米、分辨率取自
另一个文件的 zattrs。任何一步假设错，取出的 EM 与标注就对不上。
而"图像和标注错位"是训练中最难发现的一类错误——loss 照样下降，
模型只是学到一堆错配。

因此必须**看图核对**：在 crop 的坐标处取一小块 EM，与同一位置的 golgi 标注
叠加渲染。若标注恰好落在 EM 里那些平行池层堆叠上，对齐成立。

## 两套坐标系（这里踩过一次）

* **EM**：全卷体素坐标，crop4 合法范围 z[3465,3941] y[50,650] x[5950,6550]
* **crop 标注**：局部坐标，永远从 (0,0,0) 起，形状 476×600×600

因此取同一块内容时，两者要分别换算：

    crop_local = em_global - crop_origin

把全局坐标直接喂给 `fetch_channel` 会得到负维度而报错
（或静默取到错误位置，更糟）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oo_crop_fetch import fetch_channel  # noqa: E402
from oo_fetch import fetch_em  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def crop_origin(ds: str, crop: str) -> tuple:
    """从 logs/oo_locate.json 读该 crop 的 EM 起点；没有就现场算。"""
    p = os.path.join(ROOT, "logs", "oo_locate.json")
    if os.path.exists(p):
        for r in json.load(open(p, encoding="utf-8")):
            if r.get("dataset") == ds and r.get("crop") == crop and "em_z" in r:
                return (r["em_z"][0], r["em_y"][0], r["em_x"][0])
    from oo_locate import locate
    r = locate(ds, crop)
    if "error" in r:
        raise RuntimeError(f"无法定位 {ds}/{crop}: {r['error']}")
    return (r["em_z"][0], r["em_y"][0], r["em_x"][0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="jrc_hela-2")
    ap.add_argument("--crop", default="crop4")
    ap.add_argument("--z", nargs=2, type=int, default=[3700, 3724])
    ap.add_argument("--y", nargs=2, type=int, default=[100, 400])
    ap.add_argument("--x", nargs=2, type=int, default=[6000, 6300])
    ap.add_argument("--out", default="temp/oo_align")
    args = ap.parse_args(argv)

    outdir = os.path.join(ROOT, args.out)
    os.makedirs(outdir, exist_ok=True)
    zr, yr, xr = tuple(args.z), tuple(args.y), tuple(args.x)
    oz, oy, ox = crop_origin(args.dataset, args.crop)
    lz = (zr[0] - oz, zr[1] - oz)
    ly = (yr[0] - oy, yr[1] - oy)
    lx = (xr[0] - ox, xr[1] - ox)
    print(f"{args.dataset}/{args.crop}  crop 原点 (z,y,x)={(oz,oy,ox)}")
    print(f"  EM 全局范围      z{zr} y{yr} x{xr}")
    print(f"  crop 局部范围    z{lz} y{ly} x{lx}")
    if min(lz[0], ly[0], lx[0]) < 0:
        print("  [错误] 局部坐标为负，说明请求超出了 crop 范围")
        return 2

    pe = os.path.join(outdir, "em.npy")
    if os.path.exists(pe):
        em = np.load(pe)
        print(f"  EM 已有 {em.shape}")
    else:
        em, got = fetch_em(args.dataset, xr, yr, zr, workers=8)
        np.save(pe, em)
        print(f"  EM: {em.shape} uint8  {got['bytes']/1e6:.1f} MB  "
              f"ok={got['ok']} miss={got['miss']}")

    pg = os.path.join(outdir, "golgi.npy")
    if os.path.exists(pg):
        gt = np.load(pg)
        print(f"  golgi 已有 {gt.shape}")
    else:
        gt, got = fetch_channel(args.dataset, args.crop, "golgi", lz, ly, lx, 8)
        np.save(pg, gt)
        print(f"  golgi: {gt.shape}  前景 {100*(gt>0).mean():.2f}%  "
              f"{got['bytes']/1e6:.1f} MB  ok={got['ok']} miss={got['miss']}")

    if em.shape != gt.shape:
        print(f"  [错误] 形状不一致 EM{em.shape} vs GT{gt.shape}")
        return 1

    zi = int(np.argmax((gt > 0).sum(axis=(1, 2))))
    a, b = em[zi], gt[zi]
    print(f"  取局部层 z={zi}，golgi 前景 {100*(b>0).mean():.2f}%")

    left = np.stack([a] * 3, -1)
    right = left.copy()
    right[b > 0] = [255, 210, 60]
    h, w = a.shape
    canvas = np.zeros((h, w * 2 + 12, 3), np.uint8)
    canvas[:, :w] = left
    canvas[:, w + 12:] = right
    im = Image.fromarray(canvas)
    im.thumbnail((1000, 1000), Image.LANCZOS)
    im.save(os.path.join(outdir, "align.png"))
    print(f"  已保存 {args.out}/align.png（左=EM，右=EM+高尔基体标注）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
