#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explore_other_structures.py —— 在非线粒体区域里寻找"其它结构"的候选

## 为什么先做这个，而不是直接训练其它细胞器

CellMap 有 73 类细胞器可以训练，但**迁移到我们的数据已被证明失败**
（线粒体零样本 Dice 0.1516，反相后 0.0486）。在花几小时训练别的类之前，
必须先回答一个更基础的问题：

**我们的图像里，除线粒体之外，到底有没有形态上可分辨的结构？**

若没有（例如图像信噪比太低、只有细胞质纹理），那么训练任何别的类都是徒劳；
若有，则可以把这些候选导出给人工确认，用少量标注快速起一个模型。

## 方法（刻意保守）

1. 用已验证的线粒体模型得到 mito 掩码，把它排除
2. 在剩余区域做局部平均平滑（压噪）
3. 按**两类外观**找候选：
   * **亮而均匀**：局部均值显著高于全局中位数，且局部方差低
     （对应脂滴 / 空泡等均质结构）
   * **暗而封闭**：局部均值显著低于中位数
     （对应膜包被的腔室）
4. 面积过滤到 0.05–2 µm²（细胞器尺度）
5. 统计每图候选数，并导出候选裁片供人工查看

## 重要限制

**这**不是**识别，只是"候选提议"。** 输出只有位置和外观统计，
没有任何类别标签。类别必须由人确认。脚本不会声称"找到了溶酶体"，
只会说"这里有一个亮而均匀、面积 0.3 µm² 的结构"。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/原图片")
    ap.add_argument("--masks", default="data/masks_ourdata_clean")
    ap.add_argument("--annot", default="data/annotations_v2/dataset.json")
    ap.add_argument("--out", default="data/other_candidates")
    ap.add_argument("--sheet", default="logs/other_candidates_sheet.png")
    ap.add_argument("--min-area-um2", type=float, default=0.05)
    ap.add_argument("--max-area-um2", type=float, default=2.0)
    ap.add_argument("--smooth-nm", type=float, default=20.0,
                    help="局部平均的物理尺度（nm）。20 nm 约等于几个像素，足以压噪")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0=全部）")
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy", file=sys.stderr)
        return 2

    meta = json.load(open(args.annot, encoding="utf-8"))
    ents = meta["images"] if isinstance(meta["images"], list) else list(meta["images"].values())
    nm_of = {e["image"]: (e.get("um_per_px") or 0) * 1000 for e in ents}

    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in os.listdir(args.masks) if f.endswith(".png"))
    if args.limit:
        files = files[: args.limit]

    rows: List[Dict] = []
    sheet_crops: List[tuple] = []

    print(f"{'图像':<10} {'nm/px':>7} {'非mito%':>8} {'亮候选':>7} {'暗候选':>7} "
          f"{'亮面积µm²':>10}")
    print("-" * 60)

    for f in files:
        name = os.path.splitext(f)[0]
        ip = os.path.join(args.images, f"{name}.tif")
        if not os.path.exists(ip):
            continue
        img = np.asarray(Image.open(ip).convert("L"), dtype=np.float32)
        mp = os.path.join(args.masks, f)
        m = np.asarray(Image.open(mp)) > 127
        if m.shape != img.shape:
            continue
        nm = nm_of.get(name) or 4.0
        if nm <= 0:
            nm = 4.0
        px_um2 = (nm / 1000.0) ** 2

        # 平滑尺度换算成像素
        k = max(1, int(round(args.smooth_nm / nm)))
        sm = ndi.uniform_filter(img, size=k)
        valid = ~ndi.binary_dilation(m, iterations=max(2, k))

        med = float(np.median(sm[valid])) if valid.any() else float(np.median(sm))
        sd = float(sm[valid].std()) if valid.any() else float(sm.std())

        bright_thr = med + 1.0 * sd
        dark_thr = med - 1.0 * sd

        def candidates(thr: float, above: bool) -> np.ndarray:
            band = (sm > thr) if above else (sm < thr)
            band &= valid
            lab, n = ndi.label(band, structure=np.ones((3, 3)))
            if n == 0:
                return np.zeros((0, 4))
            sz = np.bincount(lab.ravel())[1:]
            keep = (sz >= args.min_area_um2 / px_um2) & (sz <= args.max_area_um2 / px_um2)
            idx = np.nonzero(keep)[0] + 1
            out = []
            for rid in idx:
                ys, xs = np.nonzero(lab == rid)
                out.append((float(xs.mean()), float(ys.mean()),
                            float(sz[rid - 1] * px_um2),
                            float(sm[lab == rid].mean())))
            return np.asarray(out) if out else np.zeros((0, 4))

        bright = candidates(bright_thr, True)
        dark = candidates(dark_thr, False)

        rows.append({"image": name, "nm_per_px": nm,
                     "non_mito_frac": float(valid.mean()),
                     "n_bright": int(len(bright)), "n_dark": int(len(dark)),
                     "bright_area_um2_total": float(bright[:, 2].sum()) if len(bright) else 0.0,
                     "bright_mean_gray": float(bright[:, 3].mean()) if len(bright) else None,
                     "median_gray": med, "gray_sd": sd})
        print(f"{name:<10} {nm:>7.2f} {valid.mean()*100:>7.1f}% "
              f"{len(bright):>7} {len(dark):>7} "
              f"{(bright[:,2].sum() if len(bright) else 0):>10.2f}")

        # 收几张亮候选的裁片做人工查看
        if len(bright) and len(sheet_crops) < 24:
            for cx, cy, area, mg in bright[:2]:
                y0 = int(max(0, cy - 48)); x0 = int(max(0, cx - 48))
                crop = img[y0:y0 + 96, x0:x0 + 96]
                if crop.shape == (96, 96):
                    lo, hi = np.percentile(crop, 2), np.percentile(crop, 98)
                    c = np.clip((crop - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
                    sheet_crops.append((f"{name} {area:.2f}um2", c))

    if not rows:
        print("\n没有成功处理的图像")
        return 1

    out_csv = os.path.join(args.out, "candidates.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    nb = np.array([r["n_bright"] for r in rows])
    nd = np.array([r["n_dark"] for r in rows])
    print()
    print("=" * 60)
    print(f"处理 {len(rows)} 张")
    print(f"亮候选（局部显著高于中位）：图均 {nb.mean():.1f}，中位 {np.median(nb):.0f}，最多 {nb.max()}")
    print(f"暗候选（局部显著低于中位）：图均 {nd.mean():.1f}，中位 {np.median(nd):.0f}，最多 {nd.max()}")
    print(f"候选位置与外观统计：{out_csv}")

    # 拼接查看表
    if sheet_crops:
        cols = 6
        rowsn = (len(sheet_crops) + cols - 1) // cols
        sheet = np.zeros((rowsn * 96, cols * 96), np.uint8)
        for i, (label, c) in enumerate(sheet_crops):
            r, cc = divmod(i, cols)
            sheet[r*96:(r+1)*96, cc*96:(cc+1)*96] = c
        os.makedirs(os.path.dirname(os.path.abspath(args.sheet)), exist_ok=True)
        Image.fromarray(sheet).resize(
            (cols*96*2, rowsn*96*2), Image.LANCZOS).save(args.sheet)
        print(f"候选裁片汇总图：{args.sheet}（{len(sheet_crops)} 个，每个 96×96 像素）")

    print()
    print("注意：这是**候选提议**，不是识别。列出的只是"
          "'位置 + 面积 + 局部灰度'，\n      具体是什么细胞器必须由人看过才能确定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
