#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图 2 用的发散柱状图：以 0 为中心，画每个配置相对其同预算对照的差值。

单独成文件是因为 `make_paper_figures.py` 里已经很长，而这个函数
有一段需要解释的设计理由：**为什么不能画绝对值**。

若把 baseline 0.3245、few-shot N=6 0.2457、N=38 0.3212 画在一起，
读者会以为"few-shot 比 baseline 差"，其实 N=6 的对照是它自己的
scratch（0.2457 对同预算的 scratch 0.2457 附近），与 baseline 不是同一预算。
绝对值图会把"数据量效应"和"手段效应"混为一谈。
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    import os
    for c in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\simsun.ttc"):
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()


def diverging_chart(path, title, labels, values, band=0.016,
                    w=980, h=560, note=""):
    """以 0 为中心的发散柱状图，画出 ±band 的噪声带。"""
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f_title, f_small, f_lab = _font(20), _font(12), _font(13)

    ml, mr, mt, mb = 80, 40, 70, 130
    pw, ph = w - ml - mr, h - mt - mb
    d.text((w // 2, 26), title, fill="black", font=f_title, anchor="mm")

    vmax = max(abs(v) for v in values) * 1.25
    zero_y = mt + ph / 2

    def y_of(v):
        return zero_y - (v / vmax) * (ph / 2)

    # ±band 噪声带
    d.rectangle([ml, y_of(band), ml + pw, y_of(-band)], fill=(245, 245, 235))
    d.text((ml + pw - 4, y_of(band) - 4), f"+-{band:.3f} noise band",
           fill=(150, 140, 90), font=f_small, anchor="rb")

    d.line([(ml, mt), (ml, mt + ph)], fill="black", width=2)
    d.line([(ml, zero_y), (ml + pw, zero_y)], fill="black", width=2)

    # y 刻度
    for k in (-1.0, -0.5, 0.0, 0.5, 1.0):
        v = vmax * k
        y = y_of(v)
        d.line([(ml - 6, y), (ml, y)], fill="black", width=1)
        d.text((ml - 10, y), f"{v:+.2f}", fill="black", font=f_small, anchor="rm")

    n = len(values)
    bw = pw / max(1, n) * 0.58
    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = ml + pw * (i + 0.5) / n
        y = y_of(v)
        col = (200, 90, 90) if v < 0 else (70, 140, 90)
        d.rectangle([cx - bw / 2, min(zero_y, y), cx + bw / 2, max(zero_y, y)],
                    fill=col)
        ty = y - 8 if v < 0 else y + 4
        d.text((cx, ty), f"{v:+.3f}", fill="black", font=f_small,
               anchor="mb" if v < 0 else "mt")
        for j, line in enumerate(lab.split("\n")):
            d.text((cx, mt + ph + 14 + j * 15), line, fill="black",
                   font=f_small, anchor="mt")

    d.text((ml, mt - 16), "delta vs same-budget control", fill="black",
           font=f_lab, anchor="lm")
    if note:
        d.text((ml, h - 18), note, fill=(90, 90, 90), font=f_small)
    img.save(path)
