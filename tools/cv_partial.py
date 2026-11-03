#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cv_partial.py —— 交叉验证的部分结果（已完成多少折就先看多少折）

## 为什么需要

`cv_golgi.sh` 只在全部 10 次训练跑完后才输出汇总。
但在跑的过程中，已完成的折已经能提供配对比较——早看到方向，
就能在发现异常时及时止损，而不是等两小时。

## 配对比较的原理

同一折上两臂相减，可以消掉"这一折本身难不难"的影响：

    配对差 = Dice(自有+ASEM, 第 k 折) − Dice(仅自有, 第 k 折)

如果每一折的差都是正（或都是负），说明方向一致，比只看均值可靠得多。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(arm: str, fold: int, need: int = 45):
    """读取某折某臂的结果。**未跑完的 run 返回 None。**

    踩过的坑：早期版本不检查 epoch 数，于是把只跑了 10 个 epoch 的
    中间状态当成结果，算出 −0.3472 这种夸张的"配对差"，
    把原本 2/2 一致的正向结论搅成"不一致"。
    部分中间结果比没有结果更危险，因为它看起来像数据。
    """
    p = os.path.join(ROOT, "runs", f"cv_{arm}_f{fold}", "history.json")
    if not os.path.exists(p):
        return None
    try:
        h = json.load(open(p, encoding="utf-8"))
    except Exception:
        return None
    d = [r.get("val_mean_dice") for r in h if r.get("val_mean_dice") is not None]
    if not d or len(d) < need:
        return None
    return {"n": len(h), "best": float(max(d)), "last5": float(np.mean(d[-5:]))}


def main() -> int:
    print("=" * 66)
    print("交叉验证部分结果（已完成的折）")
    print("=" * 66)
    print(f"{'fold':>5} {'仅自有 last5':>14} {'自有+ASEM last5':>16} {'配对差':>9}")
    print("-" * 50)
    pairs = []
    for f in range(5):
        a = load("own", f)
        b = load("comb", f)
        if not a or not b:
            continue
        diff = b["last5"] - a["last5"]
        pairs.append(diff)
        mark = "" if a["n"] >= 45 and b["n"] >= 45 else " (未跑完)"
        print(f"{f:>5} {a['last5']:>14.4f} {b['last5']:>16.4f} "
              f"{diff:>+9.4f}{mark}")
    if not pairs:
        print("\n还没有任何折完成")
        return 0
    print()
    print(f"已完成 {len(pairs)} 折")
    print(f"配对差均值 {np.mean(pairs):+.4f}"
          + (f" ± {np.std(pairs, ddof=1):.4f}" if len(pairs) > 1 else ""))
    pos = sum(1 for d in pairs if d > 0)
    print(f"ASEM 更好的折数：{pos}/{len(pairs)}")
    print()
    if len(pairs) >= 3:
        if pos >= len(pairs) - 1 and np.mean(pairs) > 0.01:
            print("方向：多数折上 ASEM 更好，证据较一致")
        elif pos <= 1 and np.mean(pairs) < -0.01:
            print("方向：多数折上 ASEM 更差，倾向负迁移")
        else:
            print("方向：不一致或幅度很小，暂不能结论")
    else:
        print("折数不足 3，暂不下判断")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
