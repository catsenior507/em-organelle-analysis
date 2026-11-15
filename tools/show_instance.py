#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""show_instance.py —— 打印实例级互作网络的汇总表"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

d = json.load(open("logs/INSTANCE_INTERACTION_3F.json", encoding="utf-8"))
print(f"视野数 {len(d)}")
for r in d:
    stem = r["stem"][:46]
    inst = "  ".join(f"{k}={v}" for k, v in r["instances"].items())
    print(f"\n{stem}\n  {inst}")

print()
hdr = f"{'类别对':<12}{'视野':>6}{'有接触/总':>14}{'平均':>8}{'中位':>7}{'最多':>6}{'零模型':>9}{'富集':>8}"
print(hdr)
print("-" * len(hdr))

pairs = set()
for r in d:
    pairs |= set(r["pairs"].keys())

for p in sorted(pairs):
    for i, r in enumerate(d, 1):
        v = r["pairs"].get(p)
        if not v or not v.get("n_a_total"):
            continue
        en = v["enrichment"]
        if en >= 100:
            ens = "inf"
        elif en > 0:
            ens = f"{en:.1f}x"
        else:
            ens = "-"
        frac = f"{v['n_a_with_contact']}/{v['n_a_total']}"
        print(f"{p:<12}{'F' + str(i):>6}{frac:>14}{v['mean_deg']:>8.2f}"
              f"{v['median_deg']:>7.1f}{v['max_deg']:>6}"
              f"{v['null_mean_deg']:>9.2f}{ens:>8}")
    print()
