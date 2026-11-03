#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_golgi_numbers.py —— 审计文档里引用的数字是否都有日志出处

## 为什么需要

本项目反复强调"报告里的数字必须与产物同源"，但文档是人手写的，
很容易出现"改了实验忘记改文档"。更糟的是**文档内部自相矛盾**——
两个章节引用同一个实验结果却给了不同的值。

这个脚本把文档里所有 `0.xxxx` 形式的数字抽出来，逐个检查它们
是否出现在 `logs/` 或 `runs/` 的产物里。找不到的会被列为"待核对"，
由人判断是笔误还是引用了未保存的结果。

## 局限（必须说清）

* 它只检查"这个数字在某个日志里出现过"，**不检查上下文是否对应**。
  一个数字恰好与别处的数值相同也会通过。
* 因此它是**筛查**工具，不是证明工具。它的价值在于把"完全凭空写的数字"
  和"抄错的数字"筛出来。
* 少数数字是有意为之的派生值（如百分比换算、差值），不会直接出现在日志里，
  需要在白名单里注明。
"""

from __future__ import annotations

import os
import re
import sys
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 允许的"派生值"：文档里合理算出、日志里不会直接出现的数字
WHITELIST = {
    # 百分比 / 换算
    "0.1763",   # 类别前景比例，出现在 sweep 日志里其实有 -> 保留以备
    # 差值（由两侧相减得到）
    "0.0157", "0.0252", "0.0253", "0.0186", "0.0142", "0.0139", "0.0092",
    "0.0066", "0.0125", "0.0135", "0.0161", "0.0334", "0.0238", "0.0506",
    "0.0748", "0.0865", "0.0274", "0.0010", "0.0053", "0.0009", "0.0207",
    "0.0300",   # 逐折配对差（由两侧相减）
    # 统计量（脚本算出后打印，可能在不同格式下）
    "0.3127", "0.3219", "0.4242", "0.4103",
    # 有意保留的**历史错误值**：文档在订正说明里引用了它，
    # 用来记录"曾经抄错成什么"。它当然不会出现在产物里——
    # 正是因为它错了。删除这条引用会让错误记录消失。
    "0.3389",
}


def collect_corpus() -> str:
    parts: List[str] = []
    for sub in ("logs", "runs"):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if not os.path.isfile(p):
                continue
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    parts.append(fh.read())
            except Exception:
                pass
    return "\n".join(parts)


def main() -> int:
    doc = os.path.join(ROOT, "docs", "GOLGI-IMPROVEMENTS.md")
    if not os.path.exists(doc):
        print(f"[错误] 找不到 {doc}")
        return 2
    text = open(doc, encoding="utf-8").read()
    corpus = collect_corpus()

    nums = sorted(set(re.findall(r"\b0\.\d{3,4}\b", text)))
    print(f"文档中出现的 0.xxxx 数字：{len(nums)} 个")
    print(f"语料（logs/ + runs/）长度：{len(corpus)} 字符")
    print()

    missing = []
    for n in nums:
        if n in corpus or n in WHITELIST:
            continue
        missing.append(n)

    if not missing:
        print("全部数字都能在产物或白名单中找到")
        return 0

    print(f"未在产物中找到、也不在白名单里的数字（{len(missing)} 个）：")
    for n in missing:
        # 找它在文档中的上下文，便于人工判断
        idx = text.find(n)
        line_no = text[:idx].count("\n") + 1
        line = text.split("\n")[line_no - 1].strip()
        print(f"  {n}   文档 L{line_no}: {line[:74]}")
    print()
    print("判断方法：")
    print("  * 若是笔误/抄错 -> 改正文档")
    print("  * 若是脚本算出的派生值 -> 加进 WHITELIST 并注明来源")
    print("  * 若是没有产物支撑的数字 -> **删掉**，不要留在文档里")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
