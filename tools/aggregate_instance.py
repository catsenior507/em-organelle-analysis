#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_instance.py —— 汇总多个视野的实例级互作结果

## 输出三张表

1. **按类别对**：跨视野的富集中位数、范围、显著（富集 > 1.5）的视野比例；
2. **按分化天**：同一类别对在不同天的变化——这是 S-BIAD2983 唯一的纵向信息
   （它是快照，不是时序，但跨天比较仍有意义）；
3. **实例数随天的变化**：各细胞器个体数是否随分化改变。

## 判据

富集的意义要在多个视野上稳定才算数。本文用两个条件：
* 中位富集 > 1.3
* 且超过一半的视野富集 > 1.2

单看某一个视野的富集会被视野间的差异淹没——这正是前面单视野结论出错的教训。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="logs/INSTANCE_20F.json")
    ap.add_argument("--fields", default="logs/biad2983_fields.json")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    data = json.load(open(a.json, encoding="utf-8"))
    fld = json.load(open(a.fields, encoding="utf-8"))
    print(f"视野 {len(data)} 个\n")

    # ---- 1) 按类别对 ----
    by_pair = defaultdict(list)
    for r in data:
        for p, v in r["pairs"].items():
            if v.get("n_a_total") and v.get("null_draws_used"):
                by_pair[p].append((r["stem"], v))

    print("=" * 92)
    print(f"{'类别对':<12}{'视野':>5}{'中位富集':>10}{'范围':>20}"
          f"{'中位伙伴数':>11}{'富集>1.2 的视野':>16}")
    print("-" * 92)
    ranked = []
    for p in sorted(by_pair):
        rows = by_pair[p]
        en = np.array([v["enrichment"] for _, v in rows])
        dg = np.array([v["mean_deg"] for _, v in rows])
        frac = float((en > 1.2).mean())
        med = float(np.median(en))
        ranked.append((med, p, len(rows), frac))
        print(f"{p:<12}{len(rows):>5}{med:>10.2f}"
              f"{f'{en.min():.2f}–{en.max():.2f}':>20}{np.median(dg):>11.2f}"
              f"{f'{int(frac*100)}%':>16}")
    print("=" * 92)

    print("\n判定（中位富集 > 1.3 且过半视野 > 1.2）：")
    solid = [x for x in ranked if x[0] > 1.3 and x[3] > 0.5]
    if solid:
        for med, p, n, frac in sorted(solid, reverse=True):
            print(f"  **{p}**  中位 {med:.2f}×  {int(frac*100)}% 的视野 > 1.2")
    else:
        print("  无。所有类别对的中位富集都 ≤ 1.3，或未过半。")

    # ---- 2) 按分化天 ----
    def day_of(stem):
        m = re.search(r"Day(\d+)", stem)
        return f"day{m.group(1)}" if m else "?"

    by_day = defaultdict(lambda: defaultdict(list))
    for r in data:
        d = day_of(r["stem"])
        for p, v in r["pairs"].items():
            if v.get("enrichment"):
                by_day[p][d].append(v["enrichment"])

    print("\n按分化天（中位富集）：")
    days = sorted({d for x in by_day.values() for d in x})
    print(f"{'类别对':<12}" + "".join(f"{d:>10}" for d in days))
    for p in sorted(by_day):
        cells = []
        for d in days:
            vals = by_day[p].get(d)
            cells.append(f"{np.median(vals):>10.2f}" if vals else f"{'-':>10}")
        print(f"{p:<12}" + "".join(cells))

    # ---- 3) 实例数随天 ----
    print("\n各细胞器实例数（中位）：")
    inst = defaultdict(lambda: defaultdict(list))
    for r in data:
        d = day_of(r["stem"])
        for c, n in r["instances"].items():
            inst[c][d].append(n)
    print(f"{'细胞器':<10}" + "".join(f"{d:>10}" for d in days))
    for c in sorted(inst):
        cells = []
        for d in days:
            vals = inst[c].get(d)
            cells.append(f"{int(np.median(vals)):>10}" if vals else f"{'-':>10}")
        print(f"{c:<10}" + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
