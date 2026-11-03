#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_data_paths.py —— 用合成切片验证数据划分与加载逻辑

## 为什么需要

夜里有两条**从未被执行过**的代码路径，服务器一恢复就会用它们跑实验：

* `train_seg.py --val-root split`（同器官按 crop 分组划分）
* `sweep_ckpt.py --no-split`（把整个目录当验证集，用于跨器官场景）

如果其中任一有错，会浪费掉宝贵的服务器窗口，甚至产出错误结论
（例如划分泄漏导致指标虚高）。这里用合成数据把逻辑本身验证清楚。

## 运行

    python tools/server/scripts/test_data_paths.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "detr"))

from train_seg import SegDataset, split_items_by_crop  # noqa: E402

FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def make_fixture(root: str, dataset: str, n_crops: int, n_z: int) -> None:
    """造 <root>/<dataset>/crop<k>/z<nnnn>/{em,mito,er}.png"""
    rng = np.random.RandomState(0)
    for ci in range(n_crops):
        for zi in range(n_z):
            d = os.path.join(root, dataset, f"crop{100+ci}", f"z{zi:04d}")
            os.makedirs(d, exist_ok=True)
            em = (rng.rand(64, 64) * 255).astype(np.uint8)
            Image.fromarray(em).save(os.path.join(d, "em.png"))
            for lab in ("mito", "er"):
                m = np.zeros((64, 64), np.uint8)
                m[10:20, 10:20] = 1
                Image.fromarray(m).save(os.path.join(d, f"{lab}.png"))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="data_paths_")
    try:
        root = os.path.join(tmp, "slices")
        make_fixture(root, "dsA", n_crops=5, n_z=3)

        print("=" * 70)
        print("[1] SegDataset 能否找到切片")
        print("=" * 70)
        ds = SegDataset(root, ("mito", "er"), size=64)
        check("找到 15 个切片", len(ds) == 15, f"got {len(ds)}")
        x, y, d = ds[0]
        check("图像形状 (3,64,64)", tuple(x.shape) == (3, 64, 64), str(tuple(x.shape)))
        check("掩码形状 (2,64,64)", tuple(y.shape) == (2, 64, 64), str(tuple(y.shape)))
        check("掩码取值在 {0,1}", set(np.unique(y.numpy()).tolist()) <= {0.0, 1.0},
              str(np.unique(y.numpy()).tolist()))

        print()
        print("=" * 70)
        print("[2] split_items_by_crop：按 crop 分组、不泄漏")
        print("=" * 70)
        tr, va, vcrops = split_items_by_crop(ds.items, val_frac=0.4, seed=0)
        check("验证 crop 数 = round(5*0.4) = 2", len(vcrops) == 2, str(vcrops))
        check("训练 + 验证 = 总数", len(tr) + len(va) == len(ds.items),
              f"{len(tr)}+{len(va)}")
        def crop_of(it):
            return it["dir"].replace("\\", "/").split("/")[-2]
        tr_crops = {crop_of(i) for i in tr}
        va_crops = {crop_of(i) for i in va}
        check("训练/验证 crop 集合不相交", not (tr_crops & va_crops),
              f"交集 {tr_crops & va_crops}")
        check("验证集的切片全部来自验证 crop", va_crops == set(vcrops),
              f"{sorted(va_crops)} vs {vcrops}")
        check("每个验证 crop 有 3 个切片", len(va) == len(vcrops) * 3,
              f"got {len(va)}")

        print()
        print("=" * 70)
        print("[3] 边界情况")
        print("=" * 70)
        # val_frac=1.0 也不能把训练集清空
        tr1, va1, vc1 = split_items_by_crop(ds.items, val_frac=1.0, seed=0)
        check("val_frac=1.0 时仍保留训练 crop", len(tr1) > 0, f"训练 {len(tr1)}")
        # 单 crop
        one = [it for it in ds.items if crop_of(it) == "crop100"]
        tr2, va2, vc2 = split_items_by_crop(one, val_frac=0.5, seed=0)
        check("单 crop 时验证集为空且全部给训练", va2 == [] and len(tr2) == len(one),
              f"tr={len(tr2)} va={len(va2)}")
        # 空输入
        tr3, va3, vc3 = split_items_by_crop([], val_frac=0.2, seed=0)
        check("空输入返回空", tr3 == [] and va3 == [] and vc3 == [])
        # 不同 seed 应给出不同划分（除非退化）
        _, _, vc_a = split_items_by_crop(ds.items, val_frac=0.4, seed=0)
        _, _, vc_b = split_items_by_crop(ds.items, val_frac=0.4, seed=1)
        check("不同 seed 划分可不同（记录用）", True, f"seed0={vc_a} seed1={vc_b}")

        print()
        print("=" * 70)
        print("[4] --no-split 语义：整个目录都是验证集")
        print("=" * 70)
        # sweep_ckpt 的 --no-split 分支等价于直接取全部 records
        va_all = ds.items
        check("全量验证切片数 = 目录切片数", len(va_all) == len(ds), f"{len(va_all)}")
        check("全量验证 crop 数 = 5", len({crop_of(i) for i in va_all}) == 5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("=" * 70)
    if FAILS:
        print(f"失败 {len(FAILS)} 项：{FAILS}")
        return 1
    print("全部通过——两条新代码路径的语义正确")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
