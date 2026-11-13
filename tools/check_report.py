#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_report.py —— 核对结题报告草稿的章节完整性与引用有效性

## 为什么单独写

结题报告（附件 3）是**表格模板**，章节标题由学校固定。草稿若漏掉某一节
（例如"四、经费开支情况"），套模板时很容易被忽略，直到答辩前才发现。

本脚本核对：
* 模板要求的章节标题是否齐全（一、…十、）
* 文中引用的 `docs/`、`logs/`、`tools/` 路径是否真实存在
* 字数（供参考，模板未规定）
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CJK = re.compile(r"[\u4e00-\u9fff]")

SECTIONS = [
    ("一、项目研究内容", "研究内容与方案"),
    ("二、项目特色", "特色与创新点"),
    ("三、项目研究取得的成果", "成果情况"),
    ("四、经费开支情况", "经费"),
    ("五、项目组对项目完成情况的自我评价", "自我评价"),
    ("六、体会和心得", "体会"),
    ("八、指导教师意见", "指导教师意见"),
    ("九、院（系）评审建议", "院系评审"),
    ("十、学校评定意见", "学校意见"),
]

# 模板里跳过"七"，此处照模板核对，若真缺"七"应视为模板自身编号问题
NOTE_ABOUT_SEVEN = "模板从『六』直接跳到『八』，无『七』节；本核对照模板原样执行。"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = os.path.join(ROOT, "docs", "REPORT-DRAFT.md")
    if not os.path.exists(p):
        print(f"[错误] 缺 {p}")
        return 2
    s = open(p, encoding="utf-8").read()

    ok = True
    print("模板章节：")
    for title, desc in SECTIONS:
        hit = title in s
        print(f"  {'✓' if hit else '✗'} {title:<34}{desc}")
        ok &= hit
    print(f"  （{NOTE_ABOUT_SEVEN}）")

    print()
    refs = re.findall(r"`((?:docs|logs|tools|data|runs)/[^`]+)`", s)
    bad = sorted({r for r in refs if not os.path.exists(os.path.join(ROOT, r))})
    print(f"路径引用：{len(refs)} 处，缺失 {len(bad)}")
    for b in bad:
        print(f"  ✗ {b}")
    ok &= not bad

    print()
    print(f"中文字数：{len(CJK.findall(s))}（模板未规定）")
    print()
    print("结论：" + ("章节齐全、引用有效" if ok else "有缺失，见上"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
