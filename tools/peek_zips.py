#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
peek_zips.py —— 检查 F 目录两个压缩包的结构与文件名编码

## 为什么要单独处理编码

Windows 上用资源管理器压缩的中文文件名，zip 里通常**不带 UTF-8 标志位**，
而是用本地代码页（中文系统是 GBK）存放字节流。Python 的 `zipfile` 遇到这种情况
会按 cp437 解码，得到一串乱码；直接解压出来的文件名也就是错的。

这里先判断每条目的 UTF-8 标志位（`flag_bits & 0x800`），
没有标志位就按 cp437 → gbk 还原，这样文件名才与原意一致。
"""

import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")

PATHS = [
    r"F/0/高尔基体图片（原图）.zip",
    r"F/1/高尔基体图片（标注后）.zip",
]


def real_name(info: zipfile.ZipInfo) -> str:
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("gbk")
    except Exception:
        return info.filename


def main() -> int:
    for p in PATHS:
        print("=" * 74)
        print(p)
        with zipfile.ZipFile(p) as z:
            infos = [i for i in z.infolist() if not i.is_dir()]
            dirs = [i for i in z.infolist() if i.is_dir()]
            print(f"  文件 {len(infos)} 条，目录 {len(dirs)} 条")
            sizes = [i.file_size for i in infos]
            if sizes:
                print(f"  原始总大小 {sum(sizes)/1e6:.1f} MB，"
                      f"最大单文件 {max(sizes)/1e6:.2f} MB")
            # 后缀分布
            from collections import Counter
            ext = Counter()
            for i in infos:
                n = real_name(i)
                ext[n.rsplit(".", 1)[-1].lower() if "." in n else "(无后缀)"] += 1
            print(f"  后缀分布: {dict(ext)}")
            print("  前 10 个条目:")
            for i in infos[:10]:
                utf8 = bool(i.flag_bits & 0x800)
                print(f"    {'[utf8]' if utf8 else '[gbk ]'} "
                      f"{real_name(i)}   {i.file_size/1e6:.2f} MB")
            if len(infos) > 10:
                print(f"    ... 还有 {len(infos)-10} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
