#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_all.py —— 一条命令跑完全部自动化检查

## 为什么需要

项目里有十几项检查（单元测试、数字审计、链接检查、隐私扫描、规范符合性……），
分散在不同脚本里。结项前或交接时，没人会记得该跑哪些。

这个脚本把全部检查串起来，逐项报告通过/失败，并在最后给一句总结。
**它不修改任何东西**，只读。

## 检查分三类

| 类别 | 检查 | 失败意味着 |
|---|---|---|
| 正确性 | 单元测试 | 代码逻辑有问题 |
| 可信性 | 数字审计（高尔基体、三份草稿） | 文档里的数与产物对不上 |
| 交付性 | 规范符合性、链接、隐私 | 提交上去会出问题 |

## 用法

    python tools/check_all.py            # 全部
    python tools/check_all.py --fast     # 跳过耗时的（公开仓库构建与隐私扫描）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(script: str, args: List[str] = None) -> Tuple[bool, str, float]:
    """跑一个 python 脚本，返回 (是否成功, 最后一行输出, 耗时)。"""
    p = os.path.join(ROOT, script)
    if not os.path.exists(p):
        return False, f"缺 {script}", 0.0
    t0 = time.time()
    try:
        r = subprocess.run([PY, p] + (args or []), cwd=ROOT,
                           capture_output=True, text=True, timeout=1800,
                           encoding="utf-8", errors="replace")
        out = (r.stdout or "").strip().split("\n")
        last = out[-1] if out else ""
        return r.returncode == 0, last, time.time() - t0
    except subprocess.TimeoutExpired:
        return False, "超时", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}", time.time() - t0


CHECKS = [
    # (类别, 显示名, 脚本, 参数)
    ("正确性", "单元测试 · 指标", "tools/server/scripts/test_metrics.py", None),
    ("正确性", "单元测试 · 数据路径", "tools/server/scripts/test_data_paths.py", None),
    ("正确性", "单元测试 · 忽略区梯度", "tools/server/scripts/test_ignore_mask.py", None),
    ("正确性", "单元测试 · 统一层兼容", "tools/test_unified_compat.py", None),

    ("可信性", "数字审计 · 高尔基体文档", "tools/verify_golgi_numbers.py", None),
    ("可信性", "数字出处 · 论文", "tools/check_paper_numbers.py",
     ["--file", "docs/PAPER-DRAFT.md"]),
    ("可信性", "数字出处 · 结题报告", "tools/check_paper_numbers.py",
     ["--file", "docs/REPORT-DRAFT.md"]),
    ("可信性", "数字出处 · 研究综述", "tools/check_paper_numbers.py",
     ["--file", "docs/REVIEW-DRAFT.md"]),
    ("可信性", "中英文档数字一致", "tools/check_translation.py", None),
    ("可信性", "流水线产物契约", "tools/verify_pipeline.py", None),

    ("交付性", "论文规范符合性", "tools/check_paper.py", None),
    ("交付性", "结题报告完整性", "tools/check_report.py", None),
    ("交付性", "文档索引链接", "tools/check_index_links.py", None),
    ("交付性", "公开仓库链接", "tools/check_readme_links.py", None),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="跳过公开仓库构建与隐私扫描")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    checks = list(CHECKS)
    if not args.fast:
        checks += [
            ("交付性", "公开仓库构建", "tools/build_public.py", None),
            ("交付性", "隐私扫描", "tools/scan_private_info.py",
             ["--path", "public_repo"]),
        ]

    print("=" * 78)
    print("全项目检查")
    print("=" * 78)
    results = []
    cur = None
    for kind, name, script, a in checks:
        if kind != cur:
            print(f"\n[{kind}]")
            cur = kind
        ok, last, dt = run(script, a)
        results.append((kind, name, ok, last, dt))
        print(f"  {'✓' if ok else '✗'} {name:<26}{dt:>6.1f}s   {last[:52]}")

    n_ok = sum(1 for *_, ok, _l, _d in [(r[0], r[1], r[2], r[3], r[4])
                                        for r in results] if ok)
    total = len(results)
    print()
    print("=" * 78)
    print(f"通过 {n_ok} / {total}")
    if n_ok < total:
        print("未通过：")
        for kind, name, ok, last, _dt in results:
            if not ok:
                print(f"  [{kind}] {name}: {last[:70]}")
    print("=" * 78)
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
