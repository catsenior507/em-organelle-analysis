#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oo_locate.py —— 从 neuroglancer 的 viewer state 解析 crop 在 EM 体中的坐标

## 为什么需要

`groundtruth/crop4/golgi/` 只存标注，不存图像；训练需要"图像 + 标注"配对。
但 `recon-1/labels/groundtruth/<crop>/` 的 `.zattrs` 里**没有**偏移量，
所以无法直接把 crop 叠到 EM 上。

坐标藏在 `neuroglancer/viewer_states/neuroglancer-<ds>-<crop>.json` 的
`position` 字段里——那是 OpenOrganelle 网页查看器跳转到该 crop 时用的视点，
单位是**纳米**。

## 换算

EM 的体素尺寸写在 `recon-1/em/fibsem-uint8/s0/.zattrs` 的 `pixelResolution`
（本数据集为 4.0、4.0、5.24 nm，对应 x、y、z）。

    voxel = position_nm / resolution_nm

`position` 是 crop 的**中心**，因此 EM 取用范围为

    中心 ± 半尺寸，半尺寸 = crop_shape / 2

## 输出

每个 crop 的 EM 体素范围，可直接喂给 oo_crop_fetch 的 --z/--y/--x。
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.request
from typing import Dict, List, Optional, Tuple

B = "https://janelia-cosem-datasets.s3.us-east-1.amazonaws.com/"
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def get(url: str, timeout: int = 45) -> Optional[str]:
    for _ in range(2):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            continue
    return None


def getj(url: str):
    t = get(url)
    if not t:
        return None
    try:
        return json.loads(t)
    except Exception:
        return None


def em_resolution(ds: str) -> Tuple[float, float, float]:
    a = getj(f"{B}{ds}/{ds}.zarr/recon-1/em/fibsem-uint8/s0/.zattrs") or {}
    pr = a.get("pixelResolution") or {}
    d = pr.get("dimensions") or [4.0, 4.0, 4.0]
    # 约定：dimensions 顺序为 x, y, z
    return float(d[0]), float(d[1]), float(d[2])


def em_shape(ds: str) -> Tuple[int, int, int]:
    z = getj(f"{B}{ds}/{ds}.zarr/recon-1/em/fibsem-uint8/s0/.zarray") or {}
    s = z.get("shape") or [0, 0, 0]
    return int(s[0]), int(s[1]), int(s[2])       # (z, y, x)


def crop_shape(ds: str, crop: str, ch: str = "golgi") -> Optional[Tuple[int, int, int]]:
    z = getj(f"{B}{ds}/{ds}.zarr/recon-1/labels/groundtruth/{crop}/{ch}/s0/.zarray")
    if not z:
        return None
    s = z.get("shape")
    return (int(s[0]), int(s[1]), int(s[2])) if s else None


def locate(ds: str, crop: str) -> Dict:
    st = getj(f"{B}{ds}/neuroglancer/viewer_states/"
              f"neuroglancer-{ds}-{crop}.json")
    if not st or "position" not in st:
        return {"dataset": ds, "crop": crop, "error": "无 viewer state 或 position"}
    pos = st["position"]                     # 纳米，顺序 x, y, z
    rx, ry, rz = em_resolution(ds)
    shape = crop_shape(ds, crop)
    if not shape:
        return {"dataset": ds, "crop": crop, "error": "无 crop 形状"}
    cz, cy, cx = shape                       # (z, y, x)

    # 中心（体素）
    mz = pos[2] / rz
    my = pos[1] / ry
    mx = pos[0] / rx
    # 半尺寸
    hz, hy, hx = cz / 2.0, cy / 2.0, cx / 2.0

    ez, ey, ex = em_shape(ds)
    z0 = max(0, int(round(mz - hz)))
    y0 = max(0, int(round(my - hy)))
    x0 = max(0, int(round(mx - hx)))
    z1 = min(ez, z0 + cz)
    y1 = min(ey, y0 + cy)
    x1 = min(ex, x0 + cx)
    return {"dataset": ds, "crop": crop, "position_nm": pos,
            "resolution_nm": [rx, ry, rz], "crop_shape_zyx": list(shape),
            "center_voxel_zyx": [round(mz, 1), round(my, 1), round(mx, 1)],
            "em_z": [z0, z1], "em_y": [y0, y1], "em_x": [x0, x1],
            "clipped": [z1 - z0 != cz, y1 - y0 != cy, x1 - x0 != cx]}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    combos = [("jrc_hela-2", "crop4"),
              ("jrc_jurkat-1", "crop37"),
              ("jrc_macrophage-2", "crop39"),
              ("jrc_macrophage-2", "crop48")]
    out: List[Dict] = []
    for ds, crop in combos:
        r = locate(ds, crop)
        out.append(r)
        if "error" in r:
            print(f"{ds}/{crop}: {r['error']}")
            continue
        print(f"{ds}/{crop}")
        print(f"  分辨率 {r['resolution_nm']} nm   中心体素 {r['center_voxel_zyx']}")
        print(f"  crop 形状 (z,y,x) {r['crop_shape_zyx']}")
        print(f"  EM 范围  z{r['em_z']} y{r['em_y']} x{r['em_x']}"
              + ("   [触边被裁剪]" if any(r["clipped"]) else ""))
    print()
    print("可直接复制的取数命令：")
    for r in out:
        if "error" in r:
            continue
        z, y, x = r["em_z"], r["em_y"], r["em_x"]
        print(f"  python tools/oo_crop_fetch.py --dataset {r['dataset']} "
              f"--crop {r['crop']} --channels golgi,er,mito "
              f"--z {z[0]} {z[1]} --y {y[0]} {y[1]} --x {x[0]} {x[1]}")

    import os
    os.makedirs("logs", exist_ok=True)
    with open("logs/oo_locate.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 logs/oo_locate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
