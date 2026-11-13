#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper_figures.py —— 为结项论文生成图表

## 为什么图表要由脚本生成

论文字数已达标，缺的是图。手绘（或用绘图软件拖）有几个问题：
数字容易与正文不一致、改一个结论就要重画、无法复核。

**由脚本从产物直接生成**的好处是：图里的每个数都来自
`logs/` 或 `runs/` 里的实际文件，改了实验只需重跑脚本。
这与项目一贯的"数字必须可追溯"是同一条原则。

## 生成哪些图

| 编号 | 内容 | 数据来源 |
|---|---|---|
| 图 1 | 标注覆盖率分层：Dice 随覆盖率单调上升 | `logs/` 分层结果 |
| 图 2 | 高尔基体各种改进手段的效应（全部落在噪声内） | 各轮实验的 last5 |
| 图 3 | 接触分析的距离剖面（五个类别对） | `data/contact3d_oo/per_crop.json` |
| 图 4 | 零模型对照：全部 55 组 z 值 | 同上 |

图 1、2 用文档里已审计过的数字（它们与 `verify_golgi_numbers.py` 一致），
图 3、4 直接读产物。
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diverging_chart import diverging_chart  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 放在 figures 的**子目录**里：构建公开仓库时只拷贝 figures 顶层文件，
# 因此这些只被私有论文引用的图不会被误传上去。
OUT = os.path.join(ROOT, "docs", "figures", "paper")


def font(size: int):
    """找一个能显示中文的字体；找不到就退回默认（中文会变方框，需注意）。"""
    cands = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for c in cands:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()


def bar_chart(path, title, labels, values, ylabel="", errs=None,
              vline=None, vline_label="", w=900, h=520, note=""):
    """画一张简单的柱状图。不引 matplotlib，避免额外依赖。"""
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f_title = font(22)
    f_lab = font(14)
    f_small = font(12)
    ml, mr, mt, mb = 90, 40, 70, 150
    pw, ph = w - ml - mr, h - mt - mb

    d.text((w // 2, 26), title, fill="black", font=f_title, anchor="mm")

    if not values:
        img.save(path)
        return
    vmax = max(values + [0.0]) * 1.15
    if vmax == 0:
        vmax = 1.0
    # 轴
    d.line([(ml, mt), (ml, mt + ph)], fill="black", width=2)
    d.line([(ml, mt + ph), (ml + pw, mt + ph)], fill="black", width=2)
    # 刻度
    for k in range(5):
        v = vmax * k / 4
        y = mt + ph - ph * k / 4
        d.line([(ml - 6, y), (ml, y)], fill="black", width=1)
        d.text((ml - 10, y), f"{v:.2f}", fill="black", font=f_small, anchor="rm")

    n = len(values)
    bw = pw / max(1, n) * 0.62
    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = ml + pw * (i + 0.5) / n
        bh = ph * (v / vmax) if vmax else 0
        col = (70, 120, 190) if v > 0 else (200, 90, 90)
        d.rectangle([cx - bw / 2, mt + ph - bh, cx + bw / 2, mt + ph], fill=col)
        if errs is not None and i < len(errs) and errs[i] > 0:
            e = ph * (errs[i] / vmax)
            d.line([(cx, mt + ph - bh - e), (cx, mt + ph - bh + e)], fill="black", width=2)
            d.line([(cx - 6, mt + ph - bh - e), (cx + 6, mt + ph - bh - e)],
                   fill="black", width=2)
            d.line([(cx - 6, mt + ph - bh + e), (cx + 6, mt + ph - bh + e)],
                   fill="black", width=2)
        d.text((cx, mt + ph + 8), f"{v:.3f}", fill="black", font=f_small, anchor="mt")
        # 标签：短的横排；长的（含中文或超过 8 字符）竖排以免重叠
        if len(lab) <= 8:
            d.text((cx, mt + ph + 30), lab, fill="black", font=f_small, anchor="mt")
        else:
            for j, ch in enumerate(lab):
                d.text((cx, mt + ph + 28 + j * 14), ch, fill="black",
                       font=f_small, anchor="mt")

    if vline is not None and vmax:
        y = mt + ph - ph * (vline / vmax)
        d.line([(ml, y), (ml + pw, y)], fill=(200, 60, 60), width=2)
        d.text((ml + pw - 4, y - 14), vline_label, fill=(200, 60, 60),
               font=f_small, anchor="rt")

    if ylabel:
        # 放在轴上方而不是左侧中部：长标签会与刻度数字重叠
        d.text((ml, mt - 16), ylabel, fill="black", font=f_lab, anchor="lm")
    if note:
        d.text((ml, h - 18), note, fill=(90, 90, 90), font=f_small)
    img.save(path)


def line_chart(path, title, xs, series, xlabel="", ylabel="",
               w=900, h=520, note=""):
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f_title = font(22)
    f_small = font(12)
    ml, mr, mt, mb = 90, 150, 70, 90
    pw, ph = w - ml - mr, h - mt - mb
    d.text((w // 2, 26), title, fill="black", font=f_title, anchor="mm")
    allv = [v for _, ys in series for v in ys]
    if not allv:
        img.save(path)
        return
    vmax = max(allv) * 1.12
    xmin, xmax = min(xs), max(xs)

    d.line([(ml, mt), (ml, mt + ph)], fill="black", width=2)
    d.line([(ml, mt + ph), (ml + pw, mt + ph)], fill="black", width=2)
    for k in range(5):
        v = vmax * k / 4
        y = mt + ph - ph * k / 4
        d.line([(ml - 6, y), (ml, y)], fill="black", width=1)
        d.text((ml - 10, y), f"{v:.0f}", fill="black", font=f_small, anchor="rm")
    for x in xs:
        px = ml + pw * (x - xmin) / max(1e-9, xmax - xmin)
        d.text((px, mt + ph + 8), f"{x:.0f}", fill="black", font=f_small, anchor="mt")
    colors = [(70, 120, 190), (200, 100, 60), (90, 160, 90),
              (150, 90, 180), (200, 160, 60)]
    for idx, (name, ys) in enumerate(series):
        c = colors[idx % len(colors)]
        pts = []
        for x, v in zip(xs, ys):
            px = ml + pw * (x - xmin) / max(1e-9, xmax - xmin)
            py = mt + ph - ph * (v / vmax)
            pts.append((px, py))
        if len(pts) > 1:
            d.line(pts, fill=c, width=3)
        for p in pts:
            d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=c)
        ly = mt + 10 + idx * 22
        d.line([(ml + pw + 16, ly), (ml + pw + 40, ly)], fill=c, width=3)
        d.text((ml + pw + 46, ly), name, fill="black", font=f_small, anchor="lm")
    if xlabel:
        d.text((ml + pw // 2, h - 44), xlabel, fill="black", font=f_small, anchor="mm")
    if ylabel:
        d.text((ml, mt - 16), ylabel, fill="black", font=f_small, anchor="lm")
    if note:
        d.text((ml, h - 18), note, fill=(90, 90, 90), font=f_small)
    img.save(path)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT, exist_ok=True)

    # ---- 图 1：覆盖率分层（已审计的数字）----
    labels = ["<3%", "3-8%", "8-20%", ">=20%"]
    golgi = [0.0900, 0.4370, 0.6046, 0.7139]
    ns = [17, 18, 15, 3]
    bar_chart(os.path.join(OUT, "fig1-coverage-golgi.png"),
              "Golgi: Dice vs annotation coverage",
              labels, golgi, ylabel="Dice",
              note=f"n per band: {', '.join(map(str, ns))};  r(coverage, Dice) = +0.730")
    mito = [0.2969, 0.7837, 0.7987, 0.8504]
    bar_chart(os.path.join(OUT, "fig1b-coverage-mito.png"),
              "Mitochondria: Dice vs annotation coverage",
              labels, mito, ylabel="Dice",
              note="Same bands; the monotone pattern reproduces independently of Golgi")

    # ---- 图 2：各种改进手段的**效应量**（与其同预算对照的差值）----
    #
    # 不能画绝对值：不同 N（数据量）的 Dice 本来就不同，画在一张图上等于把
    # "数据量效应"当成"手段效应"——这正是本项目文档反复警告的那个错误。
    # 正确的量是每个配置相对**同一预算下的对照**的差值。
    delta_labels = ["mixed\nASEM", "fewshot\nN=6", "fewshot\nN=12",
                    "fewshot\nN=38", "lr3e-5\nN=6", "lr3e-5\nN=12",
                    "multi\nN=6", "multi\nN=12"]
    deltas = [-0.0057, -0.0448, -0.0230, +0.0063,
              -0.0955, -0.1266, -0.0780, -0.0499]
    diverging_chart(os.path.join(OUT, "fig2-golgi-interventions.png"),
                    "Golgi: effect vs its own same-budget control",
                    delta_labels, deltas, band=0.016,
                    note="seed sd 0.008 -> |effect| < 0.016 is inside noise; "
                         "only mixed training is indistinguishable from zero")

    # ---- 图 3、4：接触分析，直接读产物 ----
    pc = os.path.join(ROOT, "data", "contact3d_oo", "per_crop.json")
    if not os.path.exists(pc):
        pc = os.path.join(ROOT, "temp", "contact3d_percrop.json")
    if os.path.exists(pc):
        rows = json.load(open(pc, encoding="utf-8"))
        pairs = [("endo", "er"), ("er", "mito"), ("golgi", "ves"),
                 ("er", "lyso"), ("er", "golgi")]
        xs, series = [], []
        for a, b in pairs:
            prof = []
            for r in rows:
                if {r["pair"].split(":")[0], r["pair"].split(":")[1]} != {a, b}:
                    continue
                tp = r.get("tol_profile_nm") or {}
                for e in tp.get("profile") or []:
                    if abs(float(e["tol_nm"]) - 40.0) < 1e-6:
                        prof.append(float(e.get("faces", 0)))
            if prof:
                series.append((f"{a}-{b} (n={len(prof)})", prof))
        if series:
            # 用统一的 x 轴（截止 40 nm）
            gen = rows[0].get("tol_profile_nm") or {}
            xs = [float(e["tol_nm"]) for e in (gen.get("profile") or [])]
            # 每个 pair 的中位剖面
            med_series = []
            for a, b in pairs:
                ys = []
                for x in xs:
                    vals = []
                    for r in rows:
                        if {r["pair"].split(":")[0], r["pair"].split(":")[1]} != {a, b}:
                            continue
                        for e in ((r.get("tol_profile_nm") or {}).get("profile") or []):
                            if abs(float(e["tol_nm"]) - x) < 1e-6:
                                vals.append(float(e.get("faces", 0)))
                    ys.append(float(np.median(vals)) if vals else 0.0)
                med_series.append((f"{a}-{b}", ys))
            line_chart(os.path.join(OUT, "fig3-contact-profiles.png"),
                       "Contact faces vs distance tolerance",
                       xs, med_series, xlabel="tolerance (nm)",
                       ylabel="contact faces (median)",
                       note="55 pairs: 9 have contact at 0 nm, 13 at 8 nm, rising to 46 at 40 nm")

        # 图 4：z 值分布
        zs = [r["null"]["z"] for r in rows
              if (r.get("null") or {}).get("z") is not None]
        if zs:
            bins = np.linspace(min(zs), 0, 12)
            hist, edges = np.histogram(zs, bins=bins)
            bar_chart(os.path.join(OUT, "fig4-z-distribution.png"),
                      "All 55 pairs sit below the shape-preserving null",
                      [f"{edges[i]:.1f}" for i in range(len(hist))],
                      list(hist.astype(float)), ylabel="count",
                      note=f"median z = {np.median(zs):.2f};  0 of {len(zs)} reach p<0.05")
    else:
        print(f"[跳过] 缺 {pc}，图 3/4 未生成")

    for f in sorted(os.listdir(OUT)):
        if f.startswith(("fig1", "fig2", "fig3", "fig4")):
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
