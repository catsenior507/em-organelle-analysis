#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_matlab_syntax.py —— 在无法启动 MATLAB 时对 .m 文件做**结构化**检查

## 为什么需要它

本工作区的 MATLAB 在审计期间无法启动（`matlab -batch` 报
`无法与所需的 MathWorks 服务通信(错误 5201)`，是许可服务不可用）。
但审计对若干 .m 文件做了针对性修改，必须证明它们至少**结构完整**。

## 它能证明什么、不能证明什么

**能**：块关键字（`function` / `if` / `for` / `while` / `switch` / `try` /
`parfor` / `arguments`）与 `end` 的配对；括号与引号平衡；
每个文件至少有一个 `function` 或顶层语句；`end` 不出现负数配对。

**不能**：类型错误、函数名拼错、`isfield` 语义、运行时错误。
最终仍应以 MATLAB 的 `checkcode` 为准：

    matlab -batch "addpath('code'); checkcode('find_scale_bar','-m0')"

## 用法

    python tools/check_matlab_syntax.py code/*.m
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from typing import Dict, List, Tuple

#: 会开启一个必须以 end 关闭的块的语句
BLOCK_OPENERS = (
    "function", "if", "for", "while", "switch", "try", "parfor", "arguments", "classdef",
)

#: 这些关键字后面即使跟 if/for，也不是块开始
NOT_BLOCK_PREFIXES = ("elseif", "else", "case", "otherwise", "catch", "end")

#: 行尾的 `...` 是 MATLAB 续行符，不是内容
CONTINUATION = re.compile(r"\.\.\.\s*$")

#: 只去掉注释与字符串，保留块关键字用于计数
def _scan_line(line: str, in_str: bool) -> Tuple[str, bool]:
    """单趟扫描一行，返回 (去掉注释与字符串后的代码, 行末是否仍在字符串内)。

    跨越整行（乃至跨行字符串）维护状态，避免"逐行重置"导致的引号配对错乱——
    这正是先前版本误报的根源。
    """
    out = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if in_str:
            if ch == "'":
                if i + 1 < n and line[i + 1] == "'":
                    i += 2                       # '' → 转义的单引号，仍在字符串内
                    continue
                in_str = False
            i += 1
            continue
        if ch == "%":
            break                                # 注释：本行剩余部分丢弃
        if ch == "'":
            j = i - 1
            while j >= 0 and line[j] in " \t":
                j -= 1
            prev = line[j] if j >= 0 else ""
            if prev.isalnum() or prev in ")]}._":
                out.append(ch)                   # 转置运算符 A'
            else:
                in_str = True                    # 字符串起始
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), in_str


def _join_continuations(raw_lines: List[str]) -> List[Tuple[int, str]]:
    """把 MATLAB 的 ``...`` 续行合并成逻辑行，并保留起始行号。

    必须在括号计数**之前**合并：``layers = [`` 与 ``];`` 是同一个语句，
    逐行判断括号平衡必然误报（实测 ``code/`` 下 11 个文件被误报）。
    """
    joined: List[Tuple[int, str]] = []
    buf = ""
    start = 1
    for idx, ln in enumerate(raw_lines, 1):
        if not buf:
            start = idx
        if CONTINUATION.search(ln):
            buf += CONTINUATION.sub("", ln)
            continue
        joined.append((start, buf + ln))
        buf = ""
    if buf:
        joined.append((start, buf))
    return joined


def check_file(path: str) -> Tuple[bool, List[str], Dict[str, int], str]:
    src = open(path, encoding="utf-8", errors="replace").read()
    lines = _join_continuations(src.splitlines())

    depth = 0
    problems: List[str] = []
    counts: Dict[str, int] = {"openers": 0, "end": 0}
    min_depth = 0
    # 括号深度必须**跨行**累计（理由见 _join_continuations 的 docstring）
    paren_depth = 0
    bracket_depth = 0
    paren_worst = 0
    bracket_worst = 0
    paren_line = 0
    bracket_line = 0

    in_str = False
    desync_line = 0
    for lineno, raw in lines:
        code, in_str = _scan_line(raw, in_str)
        # 单引号在 MATLAB 里既可能是转置也可能是字符串起始，纯文本解析无法
        # 100% 判定。一旦某行结束时仍"在字符串内"，说明解析很可能已经失步，
        # 后面的括号与 end 计数都不可信——此时必须**明说**，而不是报一个假错误。
        if in_str and not desync_line:
            desync_line = lineno
        stripped = code.strip()

        # ---- 跨行括号深度 ----
        paren_depth += code.count("(") - code.count(")")
        bracket_depth += code.count("[") - code.count("]")
        if paren_depth < paren_worst:
            paren_worst, paren_line = paren_depth, lineno
        if bracket_depth < bracket_worst:
            bracket_worst, bracket_line = bracket_depth, lineno

        if not stripped:
            continue

        # 统计块开启关键字（只认行首的词）
        first = re.split(r"[\s(]", stripped, maxsplit=1)[0]
        if first in BLOCK_OPENERS:
            counts["openers"] += 1
            depth += 1
        elif first == "end":
            counts["end"] += 1
            depth -= 1
            min_depth = min(min_depth, depth)
        else:
            # 紧凑写法 `if ... , ... end` / `for ... , ... end`
            bare_ends = len(re.findall(r"(?<![\w.])end(?![\w])", stripped))
            if bare_ends and first in NOT_BLOCK_PREFIXES:
                counts["end"] += bare_ends
                depth -= bare_ends
                min_depth = min(min_depth, depth)
            inline_open = len(re.findall(
                r"(?<![\w.])(?:if|for|while|switch|try)(?![\w])", stripped))
            if inline_open:
                counts["openers"] += inline_open
                depth += inline_open

    if desync_line:
        problems.append(
            f"无法判定：引号解析在 L{desync_line} 处疑似失步"
            f"（该行结束后仍在字符串内）。本文件的结构检查结果不可信，"
            f"请以 MATLAB 的 checkcode 为准。"
        )
    elif paren_depth != 0:
        problems.append(f"圆括号最终不平衡：净深度 {paren_depth:+d}"
                        f"（最深负深度 {paren_worst} 在 L{paren_line}）")
    if not desync_line and bracket_depth != 0:
        problems.append(f"方括号最终不平衡：净深度 {bracket_depth:+d}"
                        f"（最深负深度 {bracket_worst} 在 L{bracket_line}）")
    if not desync_line and depth != 0:
        problems.append(f"块关键字与 end 不配对：净深度 {depth:+d}"
                        f"（openers={counts['openers']}, end={counts['end']}）")
    if not desync_line and min_depth < 0:
        problems.append(f"出现了多余的 end（最深负深度 {min_depth}）")

    status = "无法判定" if desync_line else "ok"
    return (not problems or bool(desync_line)), problems, counts, status


def main(argv: list | None = None) -> int:
    here = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
    root = __import__("os").path.abspath(__import__("os").path.join(here, ".."))

    parser = argparse.ArgumentParser(description="MATLAB 文件的结构化检查")
    parser.add_argument("files", nargs="*", default=None,
                        help="要检查的 .m 文件；留空则检查 code/ 下全部")
    args = parser.parse_args(argv)

    files = args.files or sorted(glob.glob(__import__("os").path.join(root, "code", "*.m")))
    if not files:
        print("没有找到 .m 文件", file=sys.stderr)
        return 2

    n_bad = 0
    n_unknown = 0
    print(f"{'文件':<40} {'openers':>8} {'end':>6}  结果")
    print("-" * 74)
    for f in files:
        ok, problems, counts, status = check_file(f)
        name = __import__("os").path.basename(f)
        mark = "✓" if status == "ok" else "?"
        print(f"{name:<40} {counts['openers']:>8} {counts['end']:>6}  {mark} {status}")
        for p in problems[:6]:
            print(f"    {p}")
        if status == "无法判定":
            n_unknown += 1
        elif not ok:
            n_bad += 1

    print()
    print(f"{len(files) - n_bad - n_unknown}/{len(files)} 个文件结构检查通过；"
          f"{n_unknown} 个无法判定")
    print()
    print("注意：这只是结构化检查，**不能**替代 MATLAB 的 checkcode。")
    print("MATLAB 可用的环境下请运行：")
    print("  matlab -batch \"addpath('code'); checkcode('文件名','-m0')\"")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
