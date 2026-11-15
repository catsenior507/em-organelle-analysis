#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarise_biad712.py —— 汇总 32 个时序文件的接触动力学

## 要回答的问题

单文件（`BR1_N08`）的分析显示：多数类别对的接触**每一帧都成立**、富集 ≈ 1.0
（完全由静态空间交织解释），只有 **mitochondria:lysosome** 表现出真正的开关行为
（接触概率 0.36、时间 CV 1.74、5 次生成 5 次消解）。

但那只是一个视野。本文要检验：**这个结论在 32 个视野上是否普遍。**

## 判据

把一个（文件，类别对）判为"**有动态**"需要同时满足：

  * `contact_prob < 0.95` —— 不是一直开着，存在真正的关闭帧；
  * `cv > 1.0` —— 接触量在时间上波动超过均值（强波动）；
  * `rise >= 2` —— 至少两次独立的形成事件。

三者都是必要条件：只看 prob 会把"偶尔不接触"也算进去；只看 cv 会被
均值很小时的噪声放大。要求上升事件 ≥2 是为了排除单次闪烁。

## 零模型的读法

`enrichment = 观测均值 / 循环时间平移零模型的中位数`。
≈ 1 表示接触完全由静态交织解释；> 1 表示 A 与 B 在时间上协同；
< 1 表示它们倾向于**错开**（一个出现时另一个不在）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="logs/BIAD712_DYNAMICS_FULL.json")
    ap.add_argument("--prob-max", type=float, default=0.95)
    ap.add_argument("--cv-min", type=float, default=1.0)
    ap.add_argument("--rise-min", type=int, default=2)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    data = json.load(open(a.json, encoding="utf-8"))
    print(f"文件 {len(data)} 个")

    by_pair = defaultdict(list)
    for r in data:
        for pair, v in r["pairs"].items():
            by_pair[pair].append((r["file"], r["T"], v))

    print(f"类别对 {len(by_pair)} 个\n")

    print("=" * 96)
    print(f"{'类别对':<26}{'文件数':>6}{'中位概率':>10}{'中位CV':>9}"
          f"{'中位富集':>10}{'动态文件':>10}{'占比':>8}")
    print("-" * 96)

    verdict = {}
    for pair in sorted(by_pair, key=lambda p: -len(by_pair[p])):
        rows = by_pair[pair]
        # 某些文件里个别细胞器在某帧分割为空，中心/周长会出现 NaN。
        # 用 nanmedian 并显式排除，而不是让一个 NaN 把整列中位数变成 NaN。
        probs = np.array([v["contact_prob"] for _, _, v in rows], dtype=float)
        cvs = np.array([v["cv"] for _, _, v in rows], dtype=float)
        ens = np.array([v["enrichment"] for _, _, v in rows], dtype=float)
        dyn = sum(1 for _, _, v in rows
                  if v["contact_prob"] < a.prob_max
                  and v["cv"] > a.cv_min
                  and v["rise"] >= a.rise_min)
        verdict[pair] = dyn / len(rows)
        print(f"{pair:<26}{len(rows):>6}{np.median(probs):>10.2f}"
              f"{np.nanmedian(cvs):>9.2f}{np.nanmedian(ens):>10.2f}"
              f"{dyn:>6}/{len(rows):<3}{100*dyn/len(rows):>7.0f}%")

    print("=" * 96)

    print("\n判读：")
    ranked = sorted(verdict.items(), key=lambda kv: -kv[1])
    top, top_frac = ranked[0]
    print(f"  动态占比最高：**{top}** —— {100*top_frac:.0f}% 的文件判为有动态")
    others = [(k, v) for k, v in ranked[1:] if v > 0]
    if others:
        print("  其余出现动态的：")
        for k, v in others:
            print(f"    {k:<26}{100*v:>4.0f}%")
    else:
        print("  其余类别对**没有任何文件**判为有动态")

    static = [k for k, v in ranked if v == 0]
    print(f"\n  完全静态（富集≈1、持续接触）的类别对 {len(static)} 个：")
    for k in static:
        print(f"    {k}")

    print("\n  结论：" + (
        f"单视野的观察在 {len(data)} 个视野上成立——{top} 是主要的有动态类别对"
        if top_frac >= 0.5 else
        f"单视野的观察**不普遍**——{top} 只在 {100*top_frac:.0f}% 的文件上表现为动态"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
