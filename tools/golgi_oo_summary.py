#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
golgi_oo_summary.py —— 汇总三个高尔基体实验的最终数字

在服务器上跑，直接读 runs/*/history.json，不经过 shell 传参
（这个环境的 PowerShell 会破坏含括号与引号的 Python 单行命令）。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = [
    ("B  OO+自有（归一化）", "runs/oo_plus_ours/history.json", "自有留出 281"),
    ("B'  OO+自有（未归一化）", "runs/oo_plus_ours_rawimgs_history.json", "自有留出 281"),
    ("C  OO 跨细胞类型", "runs/oo_golgi_indomain/history.json", "OO crop39"),
    ("A  仅 OO（归一化）", "runs/oo_golgi_only/history.json", "自有留出 281"),
    ("对照  仅自有（单种子）", "runs/seg_golgi_own/history.json", "自有留出 281"),
]


def load(p):
    p2 = p if os.path.isabs(p) else os.path.join(ROOT, p)
    if not os.path.exists(p2):
        return None
    h = json.load(open(p2, encoding="utf-8"))
    d = [r["val_mean_dice"] for r in h
         if isinstance(r, dict) and r.get("val_mean_dice") is not None]
    return d or None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 84)
    print("OpenOrganelle 高尔基体实验汇总")
    print("=" * 84)
    print(f"{'实验':<26}{'epoch':>6}{'best':>9}{'last5':>9}{'final':>9}"
          f"{'非零轮':>7}  验证集")
    print("-" * 84)
    base = None
    for name, p, vs in RUNS:
        d = load(p)
        if d is None:
            print(f"{name:<26}{'—':>6}{'—':>9}{'—':>9}{'—':>9}{'—':>7}  {vs}")
            continue
        nz = sum(1 for v in d if v > 0)
        best, l5, fin = max(d), float(np.mean(d[-5:])), d[-1]
        if "对照" in name:
            base = l5
        print(f"{name:<26}{len(d):>6}{best:>9.4f}{l5:>9.4f}{fin:>9.4f}"
              f"{nz:>7}  {vs}")

    print()
    print("已有结论（3 种子，同口径）：只有自有时 best 0.4242 ± 0.0506，"
          "last5 0.3180 ± 0.0080")
    b = load("runs/oo_plus_ours/history.json")
    if b and base:
        l5 = float(np.mean(b[-5:]))
        diff = l5 - base
        print(f"\nB 与对照的单种子差：{diff:+.4f}")
        print(f"  3 种子的标准差为 0.0080，两倍标准差为 0.0160。")
        if abs(diff) < 0.016:
            print("  → **落在种子波动内，不能声称改善，也不能声称变差。**")
        elif diff > 0:
            print("  → 超出两倍标准差，可声称改善（但仍需多种子确认）。")
        else:
            print("  → 低于两倍标准差，提示变差。")
    print()
    print("注意：C 的验证集是 OO 的 crop39（巨噬细胞），训练集是 HeLa 与 Jurkat，")
    print("      因此它衡量的是**跨细胞类型**的泛化，不是同分布的域内性能。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
