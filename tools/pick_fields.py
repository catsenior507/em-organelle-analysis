#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pick_fields.py —— 从 351 个视野里挑出代表性样本

## 选法

按分化天分层，每天取前 N 个（按名字排序）。这样四个时间点都有覆盖，
可以顺便看"实例级接触数是否随分化改变"——这是 S-BIAD2983 的 4 个时间点
唯一的纵向信息（它是快照，不是时序，但跨天比较仍有意义）。

## 为什么按名字排序而不是随机

名字里的 `BR` 是生物学重复、`N` 是视野编号。按名字排序会先取某个 BR 的
连续视野，重复性不足。因此**按 BR 轮转取样**：先在每个 BR 里各取一个，
再取第二轮。这样样本横跨所有生物学重复。

输出：`logs/biad2983_sample.txt`（每行一个 stem）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="logs/biad2983_fields.json")
    ap.add_argument("--per-group", type=int, default=5)
    ap.add_argument("--out", default="logs/biad2983_sample.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    fld = json.load(open(a.fields, encoding="utf-8"))
    ok = {k: v for k, v in fld.items() if v["complete"]}

    by_group = defaultdict(list)
    for stem, v in ok.items():
        by_group[v["group"]].append(stem)

    picked = []
    for g in sorted(by_group):
        stems = sorted(by_group[g])

        # 按 BR 轮转：解析 _BR<数字>_
        def br_of(s):
            m = re.search(r"_BR(\d+)_", s)
            return int(m.group(1)) if m else 99

        buckets = defaultdict(list)
        for s in stems:
            buckets[br_of(s)].append(s)
        order = []
        round_i = 0
        while len(order) < len(stems):
            added = False
            for br in sorted(buckets):
                if round_i < len(buckets[br]):
                    order.append(buckets[br][round_i])
                    added = True
            if not added:
                break
            round_i += 1

        take = order[: a.per_group]
        picked.extend(take)
        print(f"{g:<18}可选 {len(stems):>3}  取 {len(take)}  例: {take[0][:40]}")

    open(a.out, "w", encoding="utf-8").write("\n".join(picked) + "\n")
    print(f"\n共选 {len(picked)} 个视野，写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
