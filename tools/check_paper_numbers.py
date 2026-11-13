#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_paper_numbers.py —— 核对结项论文里的每个小数是否都有出处

## 为什么必须查

论文是从十几份结果文档汇编出来的。汇编过程最容易出的一类错误是**抄错数**：
把 0.3188 写成 0.3288、把 0.730 写成 0.703。这类错误在阅读时几乎看不出来，
但会让整篇论文的可信度归零——因为它的全部价值就在于"每个数字都能追溯到产物"。

本项目已经栽过一次：某份文档把一次运行的**训练中途值**（0.3389）当成最终结果，
而真值是 0.3275。那次是 `verify_golgi_numbers.py` 抓出来的。

## 做法

把论文里所有 `\\d+\\.\\d+` 形式的数抽出来，逐个在"语料"里找：
语料 = docs/ 下的其它文档 + logs/ 下的所有文本 + runs/ 下的 history.json。
找不到的报出来，由人判断是笔误还是新算的。

## 白名单

有些数在论文里是**派生计算**的（如某两个数的差），产物里自然没有。
这类必须显式列进 WHITELIST 并写明来源，而不是放宽匹配规则——
放宽了就等于放弃检查。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM = re.compile(r"\b\d+\.\d+\b")

# 论文里由计算得出、不直接出现在产物中的数。每条必须写明来源。
WHITELIST = {
    "0.0160": "两倍种子标准差，由 0.008 推出（见 GOLGI-FEWSHOT-RESULTS.md）",
    "0.0057": "0.3245 − 0.3188，混合训练与对照的差",
    "0.0448": "scratch 0.2457 − init 0.2008",
    "0.0230": "scratch 0.2903 − init 0.2673",
    "0.0063": "init 0.3275 − scratch 0.3212",
    "0.0955": "lr3e-5 下 init 0.1143 与 scratch 0.2098 的差",
    "0.1266": "lr3e-5 下 init 0.1429 与 scratch 0.2695 的差",
    "0.0780": "多类 init 0.1677 与 scratch 0.2457 的差",
    "0.0499": "多类 init 0.2404 与 scratch 0.2903 的差",
    "0.3275": "init N=38 的三种子均值",
    "0.3212": "scratch N=38 的三种子均值",
    "0.9014": "正文汉字数（由 check_paper.py 统计）",
    # 以下两组是《撰写规范》里的排版常数，不属于实验结果，
    # 因此不会出现在 logs/ 或 runs/ 中。
    "1.75": "附件 5 规定的页脚距边界 1.75 cm",
    "2.54": "附件 5 规定的下页边距 2.54 cm",
}

TEXT_EXT = {".md", ".txt", ".json", ".csv", ".log"}


def build_corpus(exclude: str = "") -> str:
    """把 docs/（除被检查文件自身）+ logs/ + runs/ 的文本拼成一份语料。"""
    parts: List[str] = []
    paper = os.path.join(ROOT, exclude) if exclude else ""
    for base in ("docs", "logs", "runs"):
        d = os.path.join(ROOT, base)
        if not os.path.isdir(d):
            continue
        for dp, _dns, fns in os.walk(d):
            for fn in fns:
                p = os.path.join(dp, fn)
                if p == paper:
                    continue
                if os.path.splitext(fn)[1].lower() not in TEXT_EXT:
                    continue
                try:
                    if os.path.getsize(p) > 30_000_000:
                        continue
                    parts.append(open(p, encoding="utf-8", errors="replace").read())
                except Exception:
                    continue
    return "\n".join(parts)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="docs/PAPER-DRAFT.md",
                    help="要检查的草稿；另两份结项材料也可用它核对")
    args = ap.parse_args()
    paper = os.path.join(ROOT, args.file)
    if not os.path.exists(paper):
        print(f"[错误] 缺 {args.file}")
        return 2
    text = open(paper, encoding="utf-8").read()
    nums = NUM.findall(text)
    uniq: Set[str] = set(nums)
    print(f"{os.path.basename(args.file)} 中出现的小数：{len(nums)} 次，去重 {len(uniq)} 个")

    corpus = build_corpus(args.file)
    print(f"语料长度：{len(corpus):,} 字符")

    missing = sorted(n for n in uniq if n not in corpus and n not in WHITELIST)
    print()
    if missing:
        print(f"**{len(missing)} 个数字在语料与白名单中都找不到**：")
        for n in missing:
            # 给出该数在论文中的上下文，便于定位
            i = text.find(n)
            ctx = text[max(0, i - 50):i + 20].replace("\n", " ")
            print(f"  {n:<10} …{ctx}…")
        print()
        print("处理方式：若是抄写错误，改正；若确为派生计算，加进 WHITELIST 并写明来源。")
        return 1

    print("每个小数都能在产物/文档中找到出处，或已在白名单中说明来源。")
    print(f"白名单条目：{len(WHITELIST)} 个（全部写明来源）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
