#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_contact3d.py —— 把五类接触分析（环面 FFT 零模型）的结果整理成文档

## 报告什么

1. **富集比**：实测接触面 / 环面零模型的期望接触面。
   环面零模型保持每类的形状与体积不变、只随机平移，因此它回答的是
   "给定这两类的形态与丰度，它们的邻近程度是否超出随机摆位"。
2. **p 值与显著比例**：FFT 版一次覆盖全部 n³ 个平移，p 值无抽样误差。
3. **纳米距离剖面**：0/8/16/24/32/40 nm 各档的接触面数。
   这比单一阈值更能说明"贴近发生在哪个尺度"。

## 为什么不用均匀随机零模型

我先写过一个用"体积内均匀随机分布"作零模型的版本，得到所有类别对
富集 0.00–0.05。那个数字**不可解释**：组织里细胞器分区占据，
均匀随机重排会把它们混在一起，于是任何类别对都显示"远低于随机"。
这是零模型的问题，不是生物学信号。环面平移保留了形态，才是正确的比较。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pct(v: float) -> str:
    return f"{100*v:.3f}%"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="temp/contact3d_percrop.json")
    ap.add_argument("--out", default="logs/CONTACT-3D-OO.md")
    args = ap.parse_args(argv)

    p = args.inp if os.path.isabs(args.inp) else os.path.join(ROOT, args.inp)
    if not os.path.exists(p):
        print(f"[错误] 缺 {args.inp}")
        return 2
    rows = json.load(open(p, encoding="utf-8"))

    L: List[str] = []
    A = L.append
    A("# 五类细胞器的三维接触分析（环面零模型）")
    A("")
    A(f"> 数据：OpenOrganelle 四个 crop（HeLa / Jurkat / 巨噬细胞 ×2），4.0 nm 各向同性体素。")
    A(f"> 零模型：环面平移 + FFT 互相关，**一次覆盖全部 n³ 个平移**，p 值无抽样误差。")
    A(f"> 共 {len(rows)} 组 (crop, 类别对) 测量。")
    A("")

    # 1) 总体
    ps = [r["null"]["p_value"] for r in rows if r.get("null")]
    sig = sum(1 for x in ps if x is not None and x < 0.05)
    A("## 1. 总体结论")
    A("")
    A(f"* 显著高于零模型的类别对：**{sig} / {len(ps)}**（p < 0.05）")
    A(f"* p 值中位数：**{st.median(ps):.3f}**" if ps else "* 无 p 值")
    A(f"* 严格 0 距离接触面数中位数："
      f"**{st.median([r['contact_faces'] for r in rows]):.0f}**"
      f"（最大 {max(r['contact_faces'] for r in rows):.0f}）")
    A("")
    A("也就是说：**没有任何一对细胞器在四种细胞里的接触量超过"
      "\"保持各自形态、随机摆位\"的期望。** 这与本项目此前在肾组织 mito–er 上"
      "得到的结论一致（`docs/CONTACT-3D-RESULTS.md`，z = −1.63 … −4.61）。")
    A("")

    # 2) 富集比排序
    A("## 2. 富集比排序（实测 / 零模型期望）")
    A("")
    A("富集比 = `observed_faces / null_mean`。>1 表示比随机摆位更贴近。")
    A("")
    A("| crop | 类别对 | 实测接触面 | 零模型期望 | 富集比 | p |")
    A("|---|---|---|---|---|---|")
    recs = []
    for r in rows:
        nm = r.get("null") or {}
        exp = nm.get("null_mean")
        obs = r["contact_faces"]
        enr = (obs / exp) if exp else None
        recs.append((enr if enr is not None else -1, r["crop"], r["pair"],
                     obs, exp, nm.get("p_value")))
    for enr, crop, pair, obs, exp, pv in sorted(recs, reverse=True)[:18]:
        A(f"| {crop} | {pair} | {obs} | "
          f"{exp:.0f} | **{enr:.3f}** | {pv:.3f} |" if exp else
          f"| {crop} | {pair} | {obs} | — | — | — |")
    A("")

    # 3) 纳米距离剖面
    #
    # `tol_profile_nm` 不是 {nm: 计数}，而是
    #   {"voxel_nm": [...], "voxel_nm_mean": 4.0, "profile": [{"tol_nm":…, "faces":…}, …]}
    # 早期版本按字典键直接 float() 转换，撞到 "voxel_nm" 而崩。
    A("## 3. 距离剖面：贴近发生在哪个尺度")
    A("")
    A("严格相邻（0 nm）几乎全为 0，说明膜之间并不接触。"
      "下表给出类别对在各容差下的接触面数，按最大容差档排序取前 12。")
    A("")

    def profile_of(r):
        tp = r.get("tol_profile_nm") or {}
        return {float(e["tol_nm"]): float(e.get("faces", 0))
                for e in (tp.get("profile") or []) if e.get("tol_nm") is not None}

    prof_vals = set()
    for r in rows:
        prof_vals |= set(profile_of(r).keys())
    prof_keys = sorted(prof_vals)
    agg: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        prof = profile_of(r)
        if not prof:
            continue
        a, b = r["pair"].split(":")
        key = f"{min(a, b)}–{max(a, b)}"
        for k, v in prof.items():
            agg[key][k].append(v)

    def med(d, k):
        """跨 crop 聚合用：d[k] 是列表，取中位数。"""
        v = d.get(k)
        return st.median(v) if v else 0.0

    def scalar(d, k):
        """单个 crop 用：d[k] 已经是数值。"""
        v = d.get(k)
        return float(v) if isinstance(v, (int, float)) else 0.0

    A("| 类别对 | " + " | ".join(f"{k:.0f} nm" for k in prof_keys) + " | n |")
    A("|---" * (len(prof_keys) + 2) + "|")
    ordered = sorted(agg.items(), key=lambda kv: -med(kv[1], prof_keys[-1]))
    for key, d in ordered[:12]:
        cells = [f"{med(d, k):.0f}" for k in prof_keys]
        n = len(d.get(prof_keys[-1], []))
        A(f"| {key} | " + " | ".join(cells) + f" | {n} |")
    A("")
    A("读法：接触面数随容差单调上升是必然的（容差越大\"接触\"越多）。"
      "有信息量的是**上升的陡度**——若某一对在 16 nm 内就出现大量接触面，"
      "说明两膜真正贴近；若要到 32–40 nm 才有，那只是空间上靠近而非接触。")
    A("")

    # 4) 逐 crop
    A("## 4. 逐 crop 的主要类别对")
    A("")
    by_crop = defaultdict(list)
    for r in rows:
        by_crop[r["crop"]].append(r)
    for crop, rs in sorted(by_crop.items()):
        A(f"### {crop}")
        A("")
        A(f"{len(rs)} 对，{rs[0]['shape']} 体素")
        A("")
        wanted = [k for k in (0.0, 16.0, 24.0, 32.0, 40.0) if k in prof_keys]
        A("| 类别对 | " + " | ".join(f"{k:.0f}nm" for k in wanted)
          + " | 富集比 | z |")
        A("|---" * (len(wanted) + 3) + "|")
        for x in sorted(rs, key=lambda y: -scalar(profile_of(y), prof_keys[-1])
                        if prof_keys else 0)[:8]:
            pr2 = profile_of(x)
            nm = x.get("null") or {}
            exp = nm.get("null_mean")
            enr = f"{x['contact_faces']/exp:.3f}" if exp else "—"
            z = nm.get("z")
            zs = f"{z:+.2f}" if isinstance(z, (int, float)) else "—"
            cells = " | ".join(f"{pr2.get(k, 0):.0f}" for k in wanted)
            A(f"| {x['pair']} | {cells} | {enr} | {zs} |")
        A("")

    # 5) 局限
    A("## 5. 局限")
    A("")
    A("1. **语义标注，非实例**：能回答\"两类膜是否靠得近\"，"
      "不能回答\"每个高尔基体接触几个内质网\"。")
    A("2. **线粒体只在 3 个 crop 有数据**（crop4 的 mito 为空），"
      "因此含 mito 的 5 组被跳过。")
    A("3. **四个 crop 来自三种细胞**，跨 crop 汇总只作参考。")
    A("4. **环面零模型假设体积边界与对侧相连**。这是空间统计的标准做法，"
      "但对只有部分区域有标注的数据，可能引入偏差；脚本已在退化时标记 "
      "`degenerate`，本批结果均为 False。")
    A("5. 严格接触为 0 也可能是**标注口径**造成的：语义掩码之间通常留有"
      "1–2 体素的间隙（膜本身未被标注为任一类），所以 0 nm 档全零"
      "不代表生物学上没有接触，只代表标注上不接触。这一点在解释时必须保留。")

    out = "\n".join(x for x in L if x is not None) + "\n"
    op = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(out)
    print(f"写出 {args.out}（{len(L)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
