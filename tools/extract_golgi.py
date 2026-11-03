#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_golgi.py —— 解压 F 目录的高尔基体原图与标注图

## 要点

1. **文件名编码**：zip 不带 UTF-8 标志位，需 cp437→gbk 还原
2. **跳过 `__MACOSX/`**：macOS 打包时附带的元数据目录，不是数据
3. **跳过 `._*` 前缀文件**：macOS 的资源叉文件，0 字节左右
4. 输出两棵平行的目录树，按文件名一一对应：

       data/golgi/原图/<id>.jpg
       data/golgi/标注/<id>.jpg

这样后续可以按 `<id>` 配对，不必依赖解压顺序。
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")

SRC = [
    (r"F/0/高尔基体图片（原图）.zip", r"data/golgi/原图"),
    (r"F/1/高尔基体图片（标注后）.zip", r"data/golgi/标注"),
]


def real_name(info: zipfile.ZipInfo) -> str:
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("gbk")
    except Exception:
        return info.filename


def useful(info: zipfile.ZipInfo) -> bool:
    n = real_name(info)
    if info.is_dir():
        return False
    if "__MACOSX" in n or "/._" in n or n.startswith("._"):
        return False
    base = n.rsplit("/", 1)[-1]
    if base.startswith("."):
        return False
    return True


def main() -> int:
    total = 0
    for zpath, outdir in SRC:
        if not os.path.exists(zpath):
            print(f"[跳过] 不存在 {zpath}")
            continue
        if os.path.isdir(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir, exist_ok=True)
        n = 0
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                if not useful(info):
                    continue
                name = real_name(info)
                base = name.rsplit("/", 1)[-1]
                # 只保留叶子文件名，避免把上游目录结构带进来
                dst = os.path.join(outdir, base)
                with z.open(info) as src, open(dst, "wb") as fh:
                    shutil.copyfileobj(src, fh)
                n += 1
        print(f"{zpath}\n  -> {outdir}: {n} 个文件")
        total += n

    print()
    # 统计配对情况
    a = os.path.join("data/golgi/原图")
    b = os.path.join("data/golgi/标注")
    if os.path.isdir(a) and os.path.isdir(b):
        sa = {os.path.splitext(f)[0] for f in os.listdir(a)}
        sb = {os.path.splitext(f)[0] for f in os.listdir(b)}
        print(f"原图 id 数 {len(sa)}，标注 id 数 {len(sb)}")
        print(f"  两者都有（可配对）{len(sa & sb)}")
        print(f"  只有原图 {len(sa - sb)}")
        print(f"  只有标注 {len(sb - sa)}")
        if sb - sa:
            print(f"  只有标注的 id（原图缺失）：{sorted(sb - sa)[:10]}")
    print(f"\n共解压 {total} 个文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
