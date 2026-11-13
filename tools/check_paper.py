#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_paper.py —— 核对结项论文草稿是否满足《撰写规范》的硬性要求

## 规范要求（附件 5）

| 项 | 要求 |
|---|---|
| 标题 | 不超过 20 字 |
| 中文摘要 | 约 300 字 |
| 关键词 | 3–5 个 |
| 正文 | 8000–10000 字 |
| 八大部分 | 封面、标题、作者信息、中英文摘要、正文、参考文献、致谢、意见 |

本脚本只查**可机械核对**的部分（字数、关键词个数、章节是否齐全），
字体字号等排版项需在套 Word 模板时人工处理，列在文末「排版待办」里。
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CJK = re.compile(r"[\u4e00-\u9fff]")


def cjk_len(s: str) -> int:
    return len(CJK.findall(s))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = os.path.join(ROOT, "docs", "PAPER-DRAFT.md")
    if not os.path.exists(p):
        print(f"[错误] 缺 {p}")
        return 2
    s = open(p, encoding="utf-8").read()
    lines = s.split("\n")

    ok = True

    # 标题
    title = next((x.lstrip("# ").strip() for x in lines if x.startswith("# ")), "")
    n_title = cjk_len(title)
    mark = "OK" if n_title <= 20 else "超长"
    print(f"标题        {n_title:>5} 字  （要求 ≤20）  {mark}")
    print(f"            「{title}」")
    ok &= n_title <= 20

    # 中文摘要
    m = re.search(r"## 摘要\s*\n(.*?)\n\*\*关键词\*\*", s, re.S)
    if m:
        n_abs = cjk_len(m.group(1))
        print(f"中文摘要    {n_abs:>5} 字  （要求约 300）")
    else:
        print("中文摘要    未找到")
        ok = False

    # 关键词
    m = re.search(r"\*\*关键词\*\*[：:]\s*(.+)", s)
    if m:
        kws = [x.strip() for x in re.split(r"[，,、]", m.group(1)) if x.strip()]
        print(f"关键词      {len(kws):>5} 个  （要求 3–5）  {'OK' if 3 <= len(kws) <= 5 else '不符'}")
        print(f"            {'，'.join(kws)}")
        ok &= 3 <= len(kws) <= 5

    # 英文摘要关键词
    m = re.search(r"\*\*KEY WORDS\*\*[：:]\s*(.+)", s, re.I)
    if m:
        kws = [x.strip() for x in m.group(1).split(",") if x.strip()]
        print(f"KEY WORDS   {len(kws):>5} 个  （要求 3–5）  {'OK' if 3 <= len(kws) <= 5 else '不符'}")

    # 正文字数：第 1 节到参考文献之间
    i, j = s.find("## 1 绪论"), s.find("## 参考文献")
    if i > 0 and j > i:
        body = s[i:j]
        n_body = cjk_len(body)
        inrange = 8000 <= n_body <= 10000
        print(f"正文        {n_body:>5} 字  （要求 8000–10000）  {'OK' if inrange else '超出/不足'}")
        ok &= inrange
    else:
        print("正文        未找到第 1 节或参考文献")
        ok = False

    # 章节齐全
    need = ["## 摘要", "## ABSTRACT", "## 1 绪论", "## 2 研究内容与方法",
            "## 3 研究结果与讨论", "## 4 结论", "## 参考文献", "## 致谢"]
    missing = [x for x in need if x not in s]
    print(f"章节        {len(need)-len(missing)}/{len(need)} 齐全"
          + (f"  缺 {missing}" if missing else "  OK"))
    ok &= not missing

    # 参考文献条目数
    refs = re.findall(r"^\[(\d+)\]", s, re.M)
    ref_ids = [int(x) for x in refs]
    print(f"参考文献    {len(refs):>5} 条")

    # 引用合规性：《撰写规范》要求"按论文中参考文献出现的次序，用中括号的数字
    # 连续编号"。初稿曾出现**正文一个引用标记都没有**而参考文献却有十条的情况，
    # 这不符合规范，且无法判断哪条被真正使用过。
    m_i, m_j = s.find("## 1 绪论"), s.find("## 参考文献")
    if m_i > 0 and m_j > m_i:
        body = s[m_i:m_j]
        cites = [int(x) for x in re.findall(r"\[(\d+)\]", body)]
        first: list = []
        for c in cites:
            if c not in first:
                first.append(c)
        checks = [
            ("引用编号连续无缺",
             bool(cites) and sorted(set(cites)) == list(range(1, max(cites) + 1))),
            ("首次出现次序与编号一致", first == sorted(first)),
            ("无未被引用的条目", set(ref_ids) <= set(cites)),
            ("无指向不存在条目的引用", set(cites) <= set(ref_ids)),
        ]
        print(f"引用        正文 {len(cites)} 处，涉及 {len(set(cites))} 条")
        for name, good in checks:
            print(f"            {'OK' if good else '不符'}  {name}")
            ok &= good

    print()
    print("结论：" + ("全部可机械核对的项均满足" if ok else "有项目不满足，见上"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
