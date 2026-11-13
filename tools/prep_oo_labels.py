#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prep_oo_labels.py —— 把 OpenOrganelle 的标注整理成 contact_analysis_3d 期望的布局

## 为什么要转换

`tools/interaction/contact_analysis_3d.py` 期望

    <root>/<dataset>/<crop>/<类别>.npy

而取数脚本按 crop 平铺存成

    data/oo/<crop_slug>/<类别>.npy

所以需要建一棵符号链接树，把 slug 拆回 dataset 与 crop 两级。
**用软链接而非复制**：这些 .npy 总计上百 MB，复制没必要。

## 同时产出 manifest

3D 版要靠官方 manifest 把容差换算成纳米。我们的四个 crop 都是 4.0 nm 各向同性，
所以直接写一份最小 manifest（列名与官方一致：dataset / crop_name / voxel_size）。

**这一步不能省**：没有 manifest 时脚本会跳过 nm 换算，而容差用体素表示
在不同分辨率的数据之间不可比——这正是 README §8 第 1 条强调的坑。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOXEL = [4.0, 4.0, 4.0]      # OpenOrganelle 四个 crop 均为 4 nm 各向同性


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/oo")
    ap.add_argument("--out", default="data/oo_labels")
    ap.add_argument("--manifest", default="data/oo_labels/manifest.csv")
    ap.add_argument("--classes", default="golgi,er,mito,endo,ves,lyso")
    args = ap.parse_args(argv)

    src = os.path.join(ROOT, args.src)
    out = os.path.join(ROOT, args.out)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not os.path.isdir(src):
        print(f"[错误] 缺 {args.src}")
        return 2

    rows = []
    n_link = 0
    for slug in sorted(os.listdir(src)):
        d = os.path.join(src, slug)
        if not os.path.isdir(d):
            continue
        # slug 形如 hela-2_crop4 / jurkat-1_crop37 → dataset 用原名，crop 取 cropN
        if "_crop" not in slug:
            continue
        ds_part, crop = slug.rsplit("_", 1)
        dataset = "jrc_" + ds_part                   # 与官方命名一致
        tgt_dir = os.path.join(out, dataset, crop)
        os.makedirs(tgt_dir, exist_ok=True)
        got = []
        for c in classes:
            p = os.path.join(d, f"{c}.npy")
            if not os.path.exists(p):
                continue
            link = os.path.join(tgt_dir, f"{c}.npy")
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(os.path.abspath(p), link)
            got.append(c)
            n_link += 1
        print(f"  {dataset}/{crop}: {len(got)} 类 ({', '.join(got)})")
        rows.append({"dataset": dataset, "crop_name": crop,
                     "voxel_size": "[4.0;4.0;4.0]"})

    mp = os.path.join(ROOT, args.manifest)
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    with open(mp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "crop_name", "voxel_size"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n建了 {n_link} 个符号链接")
    print(f"manifest -> {args.manifest}（{len(rows)} 行，体素 4 nm 各向同性）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
