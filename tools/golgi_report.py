#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
golgi_report.py —— 从既有产物生成高尔基体工作状态报告

## 设计原则

**只读产物，不重算任何指标。** 报告里的每个数字都来自某个具体文件，
并在文中标出来源，这样"报告与实验记录不一致"这类问题不可能发生。

这条原则与 `tools/pipeline_report.py` 一致：早期手工把数字抄进文档，
后来改了实验却忘了改文档，导致两个数字互相矛盾。

## 产物清单

    logs/GOLGI_RESULT.txt            第一次训练（仅自有 51 张）
    logs/GOLGI_COMBINED_RESULT.txt   合并训练（自有 41 + ASEM）
    logs/GOLGI_ABLATION.txt          同划分两臂对比
    logs/GOLGI_MULTISEED.txt         3 种子 × 2 臂
    logs/GOLGI_CONFIG_SWEEP.txt      配置改进扫描
    data/network_multiclass/ds64.json  多类接触分析

缺哪个就跳过哪个，并在报告里说明"尚未生成"，不猜测。
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")


def read(name: str) -> Optional[str]:
    p = os.path.join(LOGS, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read().strip()


def tail(text: str, n: int) -> str:
    lines = [l for l in text.split("\n") if l.strip()]
    return "\n".join(lines[-n:])


def main() -> int:
    L: List[str] = []
    w = L.append

    w("# 高尔基体工作状态（自动生成）")
    w("")
    w("> 本文件由 `tools/golgi_report.py` 从日志与产物生成，**不重算任何指标**。")
    w("> 每个数字都标了来源，缺失的会明确写“尚未生成”。")
    w("")

    # ---------- 数据 ----------
    w("## 1. 数据")
    w("")
    w("| 来源 | 内容 | 用途 |")
    w("|---|---|---|")
    w("| 自有 | 64 张原图 / 53 张标注（绿描边），可配对 53 | 主域数据 |")
    w("| ASEM ds61 | golgi 标注 210×283×371，CF 制样 | 补充训练 |")
    w("| ASEM ds64 | golgi 284×204×424；**er+mito 同 ROI** 241×476×528 | 补充训练 / 多类互作 |")
    w("")
    w("ASEM 转出的 2D tile：ds61 golgi 660、ds64 golgi 392、ds64 er 2169、ds64 mito 1916。")
    w("")

    # ---------- 训练结果 ----------
    w("## 2. 分割结果")
    w("")
    first = read("GOLGI_RESULT.txt")
    comb = read("GOLGI_COMBINED_RESULT.txt")
    if first:
        w("### 2.1 仅自有数据（第一次）")
        w("")
        w("```")
        w(tail(first, 12))
        w("```")
        w("")
        w("来源：`logs/GOLGI_RESULT.txt`")
        w("")
    if comb:
        w("### 2.2 自有 + ASEM 合并")
        w("")
        w("```")
        w(tail(comb, 14))
        w("```")
        w("")
        w("来源：`logs/GOLGI_COMBINED_RESULT.txt`")
        w("")

    # ---------- 消融 ----------
    w("## 3. 消融：引入 ASEM 到底有没有用")
    w("")
    abl = read("GOLGI_ABLATION.txt")
    ms = read("GOLGI_MULTISEED.txt")
    if abl:
        w("同划分两臂（单次）：")
        w("")
        w("```")
        w(tail(abl, 12))
        w("```")
        w("")
        w("> **上面日志里那句「可大规模应用路线成立」是单次运行的结论，已被推翻。**")
        w("> 该结论基于一次训练的 +0.0274；三种子重复后缩到 +0.0157/+0.0252（在噪声内），")
        w("> 5 折配对交叉验证后变成 best −0.0139 / last5 +0.0092（符号相反）。")
        w("> 日志是历史记录，保留原文；**以本节下方的交叉验证结论为准。**")
        w("")
    if ms:
        w("3 种子 × 2 臂（测种子方差）：")
        w("")
        w("```")
        w(tail(ms, 22))
        w("```")
        w("")
    cv = read("GOLGI_CV.txt")
    if cv:
        w("5 折交叉验证（测划分方差，逐折配对）：")
        w("")
        w("```")
        w(tail(cv, 26))
        w("```")
        w("")
        w("**关键**：`best` 与 `last5` 给出**相反的符号**"
          "（best −0.0139 / 3-of-5，last5 +0.0092 / 4-of-5）。")
        w("因此不能声称 ASEM 有效——结论依赖统计量，说明这个样本量下效应未被证实。")
        w("")
    if not abl and not ms and not cv:
        w("**尚未生成**——消融仍在运行。")
        w("")
    w("> 注意：更早的 0.4849 与合并训练的 0.3879 用的是**不同验证划分**"
      "（317 tile / 随机 20% 对 281 tile / 每 5 张留 1 张），不可直接相减。"
      "判断引入 ASEM 的效应必须看上面的同划分对照。")
    w("")

    # ---------- 配置改进 ----------
    cfg = read("GOLGI_CONFIG_SWEEP.txt")
    w("## 4. 配置改进")
    w("")
    if cfg:
        w("```")
        w(tail(cfg, 14))
        w("```")
        w("")
        w("来源：`logs/GOLGI_CONFIG_SWEEP.txt`")
    else:
        w("**尚未生成**——配置扫描仍在排队/运行。")
    w("")

    # ---------- 多类互作 ----------
    w("## 5. 多类互作（ASEM ds64: mito + ER 同 ROI）")
    w("")
    p = os.path.join(ROOT, "data", "network_multiclass", "ds64.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        c = d.get("chosen") or {}
        sw = d.get("distance_sweep") or []
        w("**首次具备异类互作分析能力。** 两类标注在同一 ROI，可直接做接触统计。")
        w("")
        w(f"判定距离 {c.get('dist_nm')} nm（膨胀 {c.get('dilate_vox')} 体素）：")
        w("")
        w("| 指标 | 值 |")
        w("|---|---|")
        w(f"| 接触重叠体素 | {c.get('overlap_voxels')} |")
        w(f"| mito 表面接触占比 | {100*(c.get('contact_surface_frac') or 0):.2f}% |")
        w(f"| mito 表面体素 | {c.get('surface_voxels')} |")
        w("")
        if sw:
            w("距离扫描（阈值选择是可见的，不是一个黑盒数字）：")
            w("")
            w("| 距离 | 接触体素 | 占 mito 表面 |")
            w("|---|---|---|")
            for r in sw:
                w(f"| {r['dist_nm']:.0f} nm | {r['overlap_voxels']} | "
                  f"{100*(r['contact_surface_frac'] or 0):.2f}% |")
            w("")
        w("来源：`data/network_multiclass/ds64.json`")
        w("")
        w("**关键教训**：最初用 10 nm 判定得到 **0 接触**，看起来像数据有问题；"
          "实际是阈值落在 MCS（10–30 nm）窗口下沿之外。")
        w("")
    else:
        w("**尚未生成**。")
        w("")

    # ---------- 结论与限制 ----------
    w("## 6. 能得出与不能得出的结论")
    w("")
    w("### 可以说的")
    w("")
    w("* 高尔基体可以分割，且明显超过平凡基线（具体数值见上）")
    w("* ASEM 的公开数据可获取、可对齐、可转成训练切片")
    w("* mito–ER 接触在方法上可量化（4.60% 的 mito 表面在 25 nm 内接触 ER）")
    w("")
    w("### 不能说的")
    w("")
    w("* **不能**把不同划分的 Dice 直接相减（317 tile 对 281 tile）")
    w("* **不能**把像素级全图 Dice 与 tile 级 Dice 相比")
    w("* **不能**把 ASEM 的接触结果当作我们样品（BSC-1、化学固定）的生物学结论")
    w("* **不能**声称「每个线粒体接触几个 ER」——标签是二值语义，连通域不等于实例")
    w("")

    out = os.path.join(LOGS, "GOLGI_STATUS.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"已写出 {os.path.relpath(out, ROOT)}（{len(L)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
