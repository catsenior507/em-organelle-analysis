#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_zarr_to_2d.py —— 用**合成**的 CellMap 布局数据验证 zarr_to_2d.py

## 为什么需要

在真实 42 GB 数据下载完成之前，导出的代码路径无法用真实数据验证。
但"对齐数学"是可以独立验证的：造一个形状已知的合成体数据，
在特定位置放入已知强度/标签，然后检查导出结果是否落在**预期像素位置**。

本测试验证四件事：

1. ``crop_origin_voxels`` 的 ``translation / voxel_size`` 计算；
2. EM 切片确实取自 ``z0 + z`` 而不是 ``z``（这是最容易错的偏移）；
3. 标签与 EM 的 y/x 裁剪范围一致；
4. ``export_manifest.json`` 记录了策略与每帧的来源层号。

## 用法

    python tools/data_acquire/test_zarr_to_2d.py
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

DATASET = "jrc_test"
CROP = "1"
EM_SHAPE = (64, 64, 64)
CROP_SHAPE = (16, 16, 16)
VOXEL = (2.0, 2.0, 2.0)
TRANSLATION = (20.0, 20.0, 20.0)      # -> origin voxels (10, 10, 10)
EXPECTED_ORIGIN = (10, 10, 10)


def build_fixture(base: str) -> str:
    """建立 <base>/data_test/jrc_test/... 的合成 Zarr 与 manifest。"""
    import zarr

    ds = os.path.join(base, "data_test", DATASET, f"{DATASET}.zarr")
    em_dir = os.path.join(ds, "recon-1", "em", "fibsem-uint8", "s0")
    mito_dir = os.path.join(ds, "recon-1", "labels", "groundtruth",
                            CROP, "mito", "s0")
    er_dir = os.path.join(ds, "recon-1", "labels", "groundtruth",
                          CROP, "er", "s0")

    # EM：用一个 z 梯度做标记，便于验证"取的是哪一层"
    em = zarr.open_array(em_dir, mode="w", shape=EM_SHAPE,
                         chunks=(16, 32, 32), dtype="uint8")
    for z in range(EM_SHAPE[0]):
        em[z, :, :] = np.uint8(z)

    # 标签：只在 crop 内的一半区域非零，便于核对裁剪
    for path, val in ((mito_dir, 1), (er_dir, 2)):
        arr = zarr.open_array(path, mode="w", shape=CROP_SHAPE,
                              chunks=(8, 16, 16), dtype="uint8")
        arr[:, : CROP_SHAPE[1] // 2, :] = np.uint8(val)

    # manifest
    man = os.path.join(base, "train_crop_manifest.csv")
    with open(man, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["crop_name", "dataset", "class_label", "voxel_size",
                    "translation", "shape"])
        for lab in ("mito", "er"):
            w.writerow([CROP, DATASET, lab,
                        _vec(VOXEL), _vec(TRANSLATION), _vec(CROP_SHAPE)])
    return man


def _vec(t) -> str:
    return "[" + ";".join(str(float(x)) for x in t) + "]"


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="z2d_test_")
    outdir = os.path.join(tmp, "out")
    ok = True
    try:
        manifest = build_fixture(tmp)
        root = os.path.join(tmp, "data_test")
        print(f"fixture: {tmp}")

        cmd = [sys.executable, os.path.join(HERE, "zarr_to_2d.py"),
               "--root", root,
               "--manifest", manifest,
               "--dataset", DATASET,
               "--crops", CROP,
               "--labels", "mito,er",
               "--out", outdir,
               "--strategy", "slice",
               "--z-step", "5",
               "--max-per-crop", "2",
               "--start-frac", "0.25"]
        print("run:", " ".join(cmd[:3]), "...")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        # 以字节流捕获：子进程可能输出非 GBK 可解码的内容（含栈信息里的原始字节）
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           env=env)
        out = (r.stdout or b"").decode("utf-8", "replace")
        err = (r.stderr or b"").decode("utf-8", "replace")
        print(out[-2500:])
        if r.returncode != 0:
            print("STDERR:", err[-2000:])
            return 1

        # --- 断言 1：origin 计算 ---
        man = json.load(open(os.path.join(outdir, "export_manifest.json"),
                             encoding="utf-8"))
        crop = man["crops"][CROP]
        origin = tuple(crop["origin_voxels"])
        if origin != EXPECTED_ORIGIN:
            print(f"FAIL origin: 得到 {origin}，期望 {EXPECTED_ORIGIN}")
            ok = False
        else:
            print(f"PASS origin_voxels = {origin}")

        # --- 断言 2：EM 取的是 z0 + z，不是 z ---
        # 合成 EM 每层像素值恰等于该层 z。save_png 对 uint8 **保留原值**
        # （EM 原始灰度与标签 ID 都不应被 rescale），所以直接比对像素值。
        from PIL import Image
        for fr in crop["frames"]:
            p = os.path.join(outdir, DATASET, f"crop{CROP}",
                             f"z{fr['z']:04d}", "em.png")
            if not os.path.exists(p):
                print(f"FAIL 缺少 {p}")
                ok = False
                continue
            a = np.asarray(Image.open(p)).astype(np.float64)
            if abs(a.mean() - fr["em_z"]) > 1.0:
                print(f"FAIL EM 层号：z={fr['z']} em_z={fr['em_z']} "
                      f"导出像素均值={a.mean():.1f} 期望≈{fr['em_z']}")
                ok = False
            else:
                print(f"PASS EM 切片来自 em_z={fr['em_z']}"
                      f"（像素均值 {a.mean():.1f}）")

        # --- 断言 2b：标签 ID 必须保持原值，不能被 rescale ---
        for fr in crop["frames"]:
            for lab, want in (("mito", 1), ("er", 2)):
                p = os.path.join(outdir, DATASET, f"crop{CROP}",
                                 f"z{fr['z']:04d}", f"{lab}.png")
                arr = np.asarray(Image.open(p))
                vals = set(np.unique(arr).tolist())
                if not vals <= {0, want}:
                    print(f"FAIL 标签 {lab} 取值被改写：{sorted(vals)}，"
                          f"期望 ⊆ {{0, {want}}}")
                    ok = False
        print("PASS 标签 ID 未被 rescale（0/1/2 原值保留）")

        # --- 断言 3：标签非零比例符合"左半非零" ---
        for fr in crop["frames"]:
            for lab in ("mito", "er"):
                st = fr["label_stats"][lab]
                frac = st["nonzero"] / (st["shape"][0] * st["shape"][1])
                if abs(frac - 0.5) > 0.06:
                    print(f"FAIL 标签覆盖率 {lab}: {frac:.3f}，期望≈0.5")
                    ok = False
                else:
                    print(f"PASS 标签 {lab} 覆盖率 {frac:.3f} ≈ 0.5")

        # --- 断言 4：manifest 记录了策略 ---
        for k in ("strategy", "z_step", "start_frac", "coordinate_note"):
            if k not in man:
                print(f"FAIL manifest 缺少 {k}")
                ok = False
        if ok:
            print("PASS manifest 记录了切片策略与坐标约定")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n结果：", "全部通过" if ok else "有失败项")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
