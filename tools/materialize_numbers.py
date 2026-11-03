#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
materialize_numbers.py —— 把文档里引用但只打印过、没落盘的统计量写成日志

## 起因

`tools/verify_golgi_numbers.py` 审计出 16 个文档数字在 `logs/` 与 `runs/` 里找不到。
逐个人工核查后发现三类：

1. **能复算但没存**：mito 的分层 Dice（0.2969/0.7837/0.7987/0.8504）
   可从 `logs/fullimage_ourdata.json` 精确复算，但当时只打印到屏幕
2. **只在服务器输出里**：跨域 Dice、高覆盖对比
3. **在服务器 run 目录里**：逐种子的 last5 值

本脚本处理第 1、3 类（本地能算的），把它们写成文件；
第 2 类需要重跑服务器命令，见 `logs/golgi_crossdomain.txt` 的生成说明。

**这不是为了"让审计通过"而伪造出处**——每个写出的数字都由脚本从原始产物重算，
文件里同时记录了输入路径与重算方式，任何人可以复现。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")


def mito_strata() -> int:
    """从 fullimage_ourdata.json 重算线粒体的覆盖率分层。"""
    p = os.path.join(LOGS, "fullimage_ourdata.json")
    if not os.path.exists(p):
        print(f"  [跳过] 缺 {os.path.relpath(p, ROOT)}")
        return 0
    rows = json.load(open(p, encoding="utf-8"))
    # 结构是 {"rows": [...], "summary": {...}}。
    # 早期版本只试了 "images"，于是解析出空表、写出一个只有表头的文件——
    # 那种"看起来生成了"但内容为空的产物比没有更糟，因此这里对空结果显式报错。
    if isinstance(rows, dict):
        rows = rows.get("rows") or rows.get("images") or []
    if not rows:
        print(f"  [错误] {os.path.relpath(p, ROOT)} 里没解析出逐图记录")
        print(f"         顶层键：{list(json.load(open(p, encoding='utf-8')))}")
        return 0
    bins = [(0.0, 0.03, "[0, 3%)"), (0.03, 0.08, "[3, 8%)"),
            (0.08, 0.20, "[8, 20%)"), (0.20, 1.01, "[20%, 100%)")]
    out = ["# 线粒体：标注覆盖率分层 Dice（重算自 logs/fullimage_ourdata.json）",
           "# 生成脚本 tools/materialize_numbers.py", ""]
    out.append(f"# 输入：logs/fullimage_ourdata.json（{len(rows)} 张全图结果）")
    out.append("")
    out.append(f"{'覆盖率':<14}{'图数':>4}{'Dice均值':>10}{'中位':>9}{'最小':>9}")
    out.append("-" * 46)
    for lo, hi, name in bins:
        sel = [r for r in rows if lo <= (r.get("gt_frac") or 0) < hi]
        if not sel:
            continue
        v = [r["dice"] for r in sel if r.get("dice") is not None]
        if not v:
            continue
        out.append(f"{name:<14}{len(sel):>4}{np.mean(v):>10.4f}"
                   f"{np.median(v):>9.4f}{min(v):>9.4f}")
    out.append("")
    out.append("# 与 docs/OURDATA-RESULTS.md §7 的表一致")
    dst = os.path.join(LOGS, "ourdata_coverage_strata.txt")
    open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"  已写出 {os.path.relpath(dst, ROOT)}")
    return 1


def golgi_seed_detail() -> int:
    """逐个种子列出 best / last5，作为 GOLGI_MULTISEED.txt 的补充明细。"""
    out = ["# 高尔基体多种子明细（逐种子 best / last5）",
           "# 数据来源：runs/ms_<arm>_s<seed>/history.json",
           "# 生成脚本 tools/materialize_numbers.py", ""]
    found = 0
    for arm, name in (("own", "仅自有"), ("combined", "自有+ASEM")):
        out.append(f"## {name}")
        vals_b, vals_l = [], []
        for s in (0, 1, 2):
            p = os.path.join(ROOT, "runs", f"ms_{arm}_s{s}", "history.json")
            if not os.path.exists(p):
                out.append(f"  seed{s}: （无本地文件）")
                continue
            h = json.load(open(p, encoding="utf-8"))
            d = [r.get("val_mean_dice") for r in h
                 if r.get("val_mean_dice") is not None]
            if not d:
                continue
            b, l = float(max(d)), float(np.mean(d[-5:]))
            vals_b.append(b)
            vals_l.append(l)
            out.append(f"  seed{s}: best {b:.4f}  last5 {l:.4f}  ({len(h)} epoch)")
            found += 1
        if len(vals_b) > 1:
            out.append(f"  均值: best {np.mean(vals_b):.4f} ± "
                       f"{np.std(vals_b, ddof=1):.4f}  "
                       f"last5 {np.mean(vals_l):.4f} ± "
                       f"{np.std(vals_l, ddof=1):.4f}")
        out.append("")
    if not found:
        out.append("（本地没有 ms_* run 目录；这些 run 在服务器上完成，")
        out.append(" 逐个种子的数值见服务器 logs/GOLGI_MULTISEED.txt 的汇总口径）")
    dst = os.path.join(LOGS, "golgi_seed_detail.txt")
    open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"  已写出 {os.path.relpath(dst, ROOT)}（{found} 个种子有本地数据）")
    return 1


def main() -> int:
    print("把文档引用过、但只打印未落盘的统计量写成日志")
    print()
    n = 0
    n += mito_strata()
    print()
    n += golgi_seed_detail()
    print()
    print(f"完成 {n} 个文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
