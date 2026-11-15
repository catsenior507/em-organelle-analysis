#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_biad712.py —— 验证 S-BIAD712 的一个时序文件能否被正确读出

## 要回答四个问题

1. **维度**：是不是真的 T×C×Z×Y×X = 25×8×1×768×768？
2. **帧间隔**：XML 里写 `Interval = 1 ms`，但单帧就要 1.9 s，物理上不可能。
   真实间隔必须看 CZI 里逐帧的时间戳，不能信那个字段。
3. **通道身份**：8 个通道是不是对应的 7 个细胞器 + 1 个残差通道？
   元数据给了通道标签，但要从数据本身确认（强度分布应当各不相同）。
4. **可用性**：能否按帧取出像素、做后续分析。

## 为什么要先验证再分析

S-BIAD2983 的教训：描述写"timelapse images were acquired"，实际 4518 个文件里
含时间编号的是 0 个。**描述不可信，只有文件本身可信。**
"""

from __future__ import annotations

import re
import sys

import numpy as np

import czifile

PATH = "data/biad712/TS_BR1_N08.czi"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== 1) 打开文件 ===")
    czi = czifile.CziFile(PATH)
    print(f"  形状 {czi.shape}   轴序 {czi.axes}   维度 {'×'.join(map(str, czi.shape))}")

    ax = {a: i for i, a in enumerate(czi.axes)}
    T = czi.shape[ax["T"]] if "T" in ax else 1
    C = czi.shape[ax["C"]] if "C" in ax else 1
    Z = czi.shape[ax.get("Z", 0)] if "Z" in ax else 1
    print(f"  解析: T={T}  C={C}  Z={Z}")

    print("\n=== 2) 元数据里的时间信息 ===")
    meta = czi.metadata()
    if isinstance(meta, bytes):
        meta = meta.decode("utf-8", "replace")
    for pat in (r"<FrameTime>([\d.eE+-]+)</FrameTime>",
                r"<TimeSeries>(\w+)</TimeSeries>",
                r"<SizeT>(\d+)</SizeT>"):
        m = re.search(pat, meta)
        if m:
            print(f"  {pat[1:pat.find('>')]:<14} = {m.group(1)}")

    # 逐帧时间戳：CZI 把它们存在 <TimeStamps> 段里
    ts = re.findall(r"<TimeStamp>([^<]+)</TimeStamp>", meta)
    if not ts:
        # 有些版本是十六进制大整数，单位为 100 ns
        ts = re.findall(r"<TimeStamp>(\d+)</TimeStamp>", meta)
    print(f"  TimeStamp 标签数: {len(ts)}")
    if ts:
        try:
            vals = [int(x) for x in ts]
            if len(vals) > 1:
                d = np.diff(sorted(vals))
                print(f"    首帧 {vals[0]}  末帧 {vals[-1]}")
                print(f"    相邻间隔（原始单位）: 中位 {np.median(d):.0f}  最小 {d.min()}  最大 {d.max()}")
                # CZI 时间戳单位通常为 100 纳秒
                print(f"    若单位为 100 ns: 中位间隔 = {np.median(d)*1e-7:.3f} s")
        except ValueError:
            for x in ts[:5]:
                print(f"    {x}")

    print("\n=== 3) 读入数据并检查通道 ===")
    arr = czi.asarray().squeeze()          # 去掉长度为 1 的轴
    print(f"  读入形状 {arr.shape}   dtype {arr.dtype}")

    # 规范成 (T, C, Y, X)
    if arr.ndim == 4 and arr.shape[1] == C:
        data = arr
    elif arr.ndim == 4 and arr.shape[0] == C:
        data = np.moveaxis(arr, 0, 1)
    else:
        data = arr.reshape(T, C, arr.shape[-2], arr.shape[-1])
    print(f"  规整为 {data.shape} (T, C, Y, X)")

    print("\n=== 4) 每个通道的强度与前景占比（确认 7 个细胞器可分）===")
    names = ["残差(第8)", "脂滴", "内质网", "过氧化物酶体", "高尔基体",
             "线粒体", "溶酶体", "核"]
    print(f"  {'C':>3} {'mean':>8} {'p99':>8} {'>p50占比':>10}")
    for c in range(C):
        v = data[0, c].astype(np.float32)
        p50, p99 = np.percentile(v, 50), np.percentile(v, 99)
        frac = float((v > p50 * 1.5).mean()) if p50 > 0 else 0.0
        print(f"  {c:>3} {v.mean():>8.1f} {p99:>8.1f} {frac:>10.3f}")

    print("\n=== 5) 时间维是否真的有变化（不是重复帧）===")
    for c in range(min(3, C)):
        f0, f_last = data[0, c].astype(np.float32), data[-1, c].astype(np.float32)
        d = np.abs(f_last - f0).mean()
        same = np.array_equal(data[0, c], data[1, c]) if T > 1 else True
        print(f"  通道 {c}: 首末帧平均绝对差 {d:.2f}   首两帧完全相同={same}")

    print("\n=== 6) 结论 ===")
    ok = T > 1 and C >= 7
    print(f"  {'✓' if ok else '✗'} 时序可读: T={T} 帧, C={C} 通道, "
          f"{data.shape[-2]}×{data.shape[-1]} 像素")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
